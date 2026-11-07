"""Stage 4, step 3: PPO fine-tuning of the imitation policy, with domain randomisation (control delay, payload, damping, friction).

  python3 train_ppo.py --init runs/il_policy.npz --out runs/ppo_s1 --seed 1 --workers 4 --iters 100

NumPy only (no PyTorch), so it runs anywhere NumPy and MuJoCo run.  The actor starts as the imitation network (mean action) with
a fixed-start exploration noise; the critic is trained alone for a few iterations first so that the advantages mean something
before the actor is allowed to move.

Each iteration: workers collect whole episodes (each with its own random delay / payload / damping / friction) until ~--steps steps;
GAE advantages; a few epochs of clipped-PPO updates; every --eval-every iterations a deterministic evaluation in two conditions.
Progress is printed as a bar, the log goes to <out>/ppo_log.csv, and the best actor (by evaluation) is saved as <out>/best_policy.npz
(the same format as imitate.py's policy, so  python3 imitate.py --eval <out>/best_policy.npz  works).
"""
import os, time, csv, argparse
import numpy as np
from multiprocessing import Pool
from rl_env import PickPlaceEnv, OBS_DIM
from imitate import MLP, Policy

FRIC = np.array([0.6, 0.8, 0.5, 0.2, 0.15, 0.08])


def random_shift(rng):
    """Mid-episode change at the lift, drawn fresh each episode (None = no change)."""
    if rng.random() < 0.2:
        return None
    s = dict(payload=float(rng.uniform(0.0, 2.0)), damping=float(np.exp(rng.uniform(np.log(0.5), np.log(4.0)))))
    if rng.random() < 0.4:
        s["friction_vec"] = rng.uniform(0.3, 1.5, 6) * FRIC
    return s


# ---------------------------------------------------------------- workers
_ENV = None


def _init(seed):
    global _ENV
    _ENV = PickPlaceEnv(seed=seed + os.getpid() % 1000)


def _policy(w):
    actor = Policy(MLP(None, params=(w["aW"], w["ab"])), w["mu"], w["sd"])
    return actor, w["logstd"]


def collect(args):
    """Run whole episodes until at least n_steps steps.  Returns per-step arrays plus episode boundaries."""
    w, n_steps, seed, dmax = args
    env, actor, logstd = _ENV, *_policy(w)
    rng = np.random.default_rng(seed); std = np.exp(logstd)
    env.rng = np.random.default_rng(seed + 1)
    O, A, LP, R, EP_END, TERM, LASTO, wins, eps, rets = [], [], [], [], [], [], [], 0, 0, []
    while len(O) < n_steps:
        env.delay_steps = int(rng.integers(0, dmax + 1)) if rng.random() < 0.7 else 0
        env.shift_fn = random_shift
        o, _ = env.reset(); ret = 0.0
        while True:
            mean = actor.net.forward(((o - actor.mu) / actor.sd)[None])[0]
            a = mean + std * rng.standard_normal(4)
            lp = float(-0.5 * np.sum(((a - mean) / std) ** 2) - np.sum(logstd) - 2 * np.log(2 * np.pi))
            o2, r, te, tr, info = env.step(np.clip(a, -1, 1))
            O.append(o); A.append(a); LP.append(lp); R.append(r); ret += r
            o = o2
            if te or tr:
                EP_END.append(len(O) - 1); TERM.append(bool(te)); LASTO.append(o); break
        wins += int(info["success"]); eps += 1; rets.append(ret)
    return (np.array(O, np.float32), np.array(A, np.float32), np.array(LP), np.array(R), np.array(EP_END), np.array(TERM), np.array(LASTO, np.float32), wins, eps, rets)


def evaluate(args):
    """Deterministic policy (mean action) on fixed tasks in one condition. Returns (successes, episodes, mean steps)."""
    w, cond, n_eps, seed = args
    actor, _ = _policy(w)
    kw = dict(delay_steps=5) if cond == "delay10" else (dict(shift_fn=lambda r: dict(payload=1.0, damping=3.0)) if cond == "shift" else {})
    env = PickPlaceEnv(seed=seed, **kw); ok, steps = 0, 0
    for _ in range(n_eps):
        o, _ = env.reset()
        for k in range(1300):
            o, r, te, tr, info = env.step(actor.act(o))
            if te or tr: break
        ok += int(info["success"]); steps += k + 1
    return ok, n_eps, steps / n_eps


# ---------------------------------------------------------------- PPO pieces
class Adam:
    def __init__(self, params, lr):
        self.p, self.lr, self.t = params, lr, 0
        self.m = [np.zeros_like(x) for x in params]; self.v = [np.zeros_like(x) for x in params]

    def step(self, grads, max_norm=1.0):
        norm = float(np.sqrt(sum(np.sum(g ** 2) for g in grads)))
        sc = min(1.0, max_norm / (norm + 1e-8)); self.t += 1
        for i, g in enumerate(grads):
            g = g * sc
            self.m[i] = 0.9 * self.m[i] + 0.1 * g; self.v[i] = 0.999 * self.v[i] + 0.001 * g * g
            self.p[i] -= self.lr * (self.m[i] / (1 - 0.9 ** self.t)) / (np.sqrt(self.v[i] / (1 - 0.999 ** self.t)) + 1e-8)
        return norm


def backward(net, hs, g):
    """Gradients of a scalar loss wrt every W then every b, given dLoss/dOutput = g."""
    gW, gb = [None] * len(net.W), [None] * len(net.W)
    for k in range(len(net.W) - 1, -1, -1):
        gW[k], gb[k] = hs[k].T @ g, g.sum(0)
        if k > 0:
            g = (g @ net.W[k].T) * (1 - hs[k] ** 2)
    return gW + gb


def gae(R, V, V_last, ep_end, term, gamma, lam):
    """Advantages for a concatenation of whole episodes. V_last[i] = value of the final observation of episode i (0 if terminal)."""
    n = len(R); adv = np.zeros(n); start = 0
    for i, e in enumerate(ep_end):
        nxt_v, last = (0.0 if term[i] else V_last[i]), 0.0
        for t in range(e, start - 1, -1):
            nv = nxt_v if t == e else V[t + 1]
            delta = R[t] + gamma * nv - V[t]
            last = delta + gamma * lam * last
            adv[t] = last
        start = e + 1
    return adv


def bar(frac, text):
    n = int(20 * frac)
    print(f"{text} [{'#' * n}{'-' * (20 - n)}] {int(100 * frac)}%", flush=True)


def main(a):
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    il = Policy.load(a.init)
    actor = il.net
    mu, sd = il.mu, il.sd
    logstd = np.full(4, np.log(a.std0))
    critic = MLP([OBS_DIM, a.hidden, a.hidden, 1], rng)
    opt_a = Adam(actor.W + actor.b + [logstd], a.lr_actor)
    opt_c = Adam(critic.W + critic.b, a.lr_critic)
    pool = Pool(a.workers, initializer=_init, initargs=(a.seed * 100,))
    log, best, t0, total_steps = [], -1.0, time.time(), 0
    wr = lambda: dict(aW=actor.W, ab=actor.b, mu=mu, sd=sd, logstd=logstd)

    for it in range(a.iters):
        res = pool.map(collect, [(wr(), a.steps // a.workers, 10000 * a.seed + 100 * it + k, a.max_delay) for k in range(a.workers)])
        O = np.concatenate([r[0] for r in res]); A = np.concatenate([r[1] for r in res]); LP = np.concatenate([r[2] for r in res]); R = np.concatenate([r[3] for r in res])
        off, EPE, TERM, LASTO = 0, [], [], []
        for r in res:
            EPE += list(r[4] + off); TERM += list(r[5]); LASTO.append(r[6]); off += len(r[0])
        LASTO = np.concatenate(LASTO); wins, eps = sum(r[7] for r in res), sum(r[8] for r in res)
        mean_ret = float(np.mean(sum([r[9] for r in res], [])))
        total_steps += len(O)
        X = ((O - mu) / sd)
        V = critic.forward(X)[:, 0]; V_last = critic.forward((LASTO - mu) / sd)[:, 0]
        adv = gae(R, V, V_last, EPE, TERM, a.gamma, a.lam)
        ret = adv + V
        advn = (adv - adv.mean()) / (adv.std() + 1e-8)
        n, kl_last, upd_actor = len(O), 0.0, it >= a.critic_warmup
        for ep in range(a.epochs):
            idx = rng.permutation(n)
            for s in range(0, n, a.batch):
                j = idx[s:s + a.batch]; xb = X[j]
                # critic: mean squared error to the return
                v, hc = critic.forward(xb, keep=True)
                opt_c.step(backward(critic, hc, 2 * (v - ret[j][:, None]) / len(j)), max_norm=5.0)
                if not upd_actor:
                    continue
                mean, ha = actor.forward(xb, keep=True)
                std = np.exp(logstd)
                z = (A[j] - mean) / std
                lp = -0.5 * np.sum(z ** 2, 1) - np.sum(logstd) - 2 * np.log(2 * np.pi)
                ratio = np.exp(lp - LP[j])
                active = ~(((ratio > 1 + a.clip) & (advn[j] > 0)) | ((ratio < 1 - a.clip) & (advn[j] < 0)))
                coef = -(advn[j] * ratio * active)[:, None] / len(j)           # d(loss)/d(logp)
                g_mean = coef * z / std
                g_logstd = (coef * (z ** 2 - 1)).sum(0)
                opt_a.step(backward(actor, ha, g_mean) + [g_logstd], max_norm=1.0)
                logstd[:] = np.clip(logstd, np.log(0.02), np.log(0.6))
            if upd_actor:
                mean = actor.forward(X)
                lp = -0.5 * np.sum(((A - mean) / np.exp(logstd)) ** 2, 1) - np.sum(logstd) - 2 * np.log(2 * np.pi)
                kl_last = float(np.mean(LP - lp))
                if kl_last > a.target_kl:
                    break
        row = dict(iter=it, steps=total_steps, train_success=wins / eps, mean_return=mean_ret, kl=kl_last, std=float(np.exp(logstd).mean()), minutes=(time.time() - t0) / 60)
        if (it + 1) % a.eval_every == 0 or it == a.iters - 1:
            conds = ["nominal", "delay10", "shift"]
            ev = pool.map(evaluate, [(wr(), c, a.eval_eps, 777) for c in conds])
            for c, e in zip(conds, ev):
                row[f"eval_{c}"] = e[0] / e[1]
            score = float(np.mean([e[0] / e[1] for e in ev]))
            row["eval_mean"] = score
            if score > best:
                best = score; Policy(actor, mu, sd).save(os.path.join(a.out, "best_policy.npz"))
        Policy(actor, mu, sd).save(os.path.join(a.out, "last_policy.npz"))
        log.append(row)
        with open(os.path.join(a.out, "ppo_log.csv"), "w", newline="") as f:
            cols = sorted({k for r in log for k in r}, key=lambda k: list(row).index(k) if k in row else 99)
            w = csv.DictWriter(f, cols); w.writeheader(); w.writerows(log)
        txt = f"PPO iter {it + 1}/{a.iters}  steps {total_steps}  train success {wins}/{eps}  return {mean_ret:6.2f}  KL {kl_last:.4f}  std {np.exp(logstd).mean():.3f}"
        if "eval_mean" in row:
            txt += f"  EVAL nominal {row['eval_nominal']:.2f} delay {row['eval_delay10']:.2f} shift {row['eval_shift']:.2f}"
        bar((it + 1) / a.iters, txt)
    pool.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default="runs/il_policy.npz"); ap.add_argument("--out", default="runs/ppo_s1"); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--workers", type=int, default=2); ap.add_argument("--iters", type=int, default=100); ap.add_argument("--steps", type=int, default=16384)
    ap.add_argument("--epochs", type=int, default=4); ap.add_argument("--batch", type=int, default=1024); ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--lr-actor", dest="lr_actor", type=float, default=3e-5); ap.add_argument("--lr-critic", dest="lr_critic", type=float, default=1e-3)
    ap.add_argument("--gamma", type=float, default=0.995); ap.add_argument("--lam", type=float, default=0.95); ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--std0", type=float, default=0.15); ap.add_argument("--target-kl", dest="target_kl", type=float, default=0.03)
    ap.add_argument("--critic-warmup", dest="critic_warmup", type=int, default=3); ap.add_argument("--max-delay", dest="max_delay", type=int, default=5)
    ap.add_argument("--eval-every", dest="eval_every", type=int, default=5); ap.add_argument("--eval-eps", dest="eval_eps", type=int, default=10)
    main(ap.parse_args())
