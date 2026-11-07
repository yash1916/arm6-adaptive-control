"""Stage 4, step 2: imitation learning (behaviour cloning + DAgger) of the pick-and-place expert.

  python3 imitate.py --iters 8 --out runs/il            # train
  python3 imitate.py --eval runs/il/policy.npz --episodes 40   # evaluate in four conditions

Behaviour cloning copies the expert on states the EXPERT visits.  A learner makes small mistakes, reaches states the expert never
showed it, and drifts.  DAgger fixes that: run the LEARNER, let the expert label every state it visits, add those to the data, retrain.
NumPy only (a 2-hidden-layer tanh network, Adam), like the stage 2/3 networks.
"""
import os, sys, time, argparse, csv
import numpy as np
from multiprocessing import Pool
from rl_env import PickPlaceEnv, StatelessExpert, OBS_DIM, ACT_DIM


class MLP:
    def __init__(self, sizes, rng=None, params=None):
        if params is not None:
            self.W, self.b = params; return
        rng = rng or np.random.default_rng(0)
        self.W = [rng.standard_normal((i, o)) * np.sqrt(1.0 / i) for i, o in zip(sizes[:-1], sizes[1:])]
        self.W[-1] *= 0.1
        self.b = [np.zeros(o) for o in sizes[1:]]

    def forward(self, x, keep=False):
        hs = [x]
        for W, b in zip(self.W[:-1], self.b[:-1]):
            hs.append(np.tanh(hs[-1] @ W + b))
        y = hs[-1] @ self.W[-1] + self.b[-1]
        return (y, hs) if keep else y

    def fit(self, X, Y, epochs, lr, rng, batch=512, log=None):
        P = self.W + self.b
        m = [np.zeros_like(p) for p in P]; v = [np.zeros_like(p) for p in P]; t = 0
        n = len(X)
        for ep in range(epochs):
            cur = lr * (0.5 * (1 + np.cos(np.pi * ep / epochs)) * 0.9 + 0.1)
            idx = rng.permutation(n); tot = 0.0
            for s in range(0, n, batch):
                j = idx[s:s + batch]; xb, yb = X[j], Y[j]
                y, hs = self.forward(xb, keep=True)
                g = 2 * (y - yb) / len(j); tot += float(np.sum((y - yb) ** 2))
                gW, gb = [None] * len(self.W), [None] * len(self.W)
                for k in range(len(self.W) - 1, -1, -1):
                    gW[k], gb[k] = hs[k].T @ g, g.sum(0)
                    if k > 0:
                        g = (g @ self.W[k].T) * (1 - hs[k] ** 2)
                t += 1
                for i, gr in enumerate(gW + gb):
                    m[i] = 0.9 * m[i] + 0.1 * gr; v[i] = 0.999 * v[i] + 0.001 * gr ** 2
                    P[i] -= cur * (m[i] / (1 - 0.9 ** t)) / (np.sqrt(v[i] / (1 - 0.999 ** t)) + 1e-8)
            if log: log(ep, tot / n / Y.shape[1])


class Policy:
    """Normalised-input network; action = clip(output, -1, 1)."""
    def __init__(self, net, mu, sd):
        self.net, self.mu, self.sd = net, mu, sd

    def act(self, obs):
        return np.clip(self.net.forward(((obs - self.mu) / self.sd)[None])[0], -1, 1)

    def save(self, path):
        np.savez(path, mu=self.mu, sd=self.sd, nW=len(self.net.W), **{f"W{i}": W for i, W in enumerate(self.net.W)}, **{f"b{i}": b for i, b in enumerate(self.net.b)})

    @staticmethod
    def load(path):
        z = np.load(path); n = int(z["nW"])
        return Policy(MLP(None, params=([z[f"W{i}"] for i in range(n)], [z[f"b{i}"] for i in range(n)])), z["mu"], z["sd"])


def rollout(args):
    """One worker: run n episodes. The learner drives (with probability 1-beta per step the learner acts, else the expert); the
    expert labels EVERY visited state.  weights=None -> expert only."""
    seed, weights, beta, n_eps, max_steps = args
    env = PickPlaceEnv(seed=seed); ex = StatelessExpert(env)
    pol = None if weights is None else Policy(MLP(None, params=(weights[0], weights[1])), weights[2], weights[3])
    rng = np.random.default_rng(seed + 99)
    X, Y, wins = [], [], 0
    for _ in range(n_eps):
        o, _ = env.reset()
        for k in range(max_steps):
            a_exp = ex(o)
            X.append(o); Y.append(a_exp)
            a = a_exp if (pol is None or rng.random() < beta) else pol.act(o)
            o, r, te, tr, info = env.step(a)
            if te or tr: break
        wins += int(info["success"])
    return np.array(X, np.float32), np.array(Y, np.float32), wins, n_eps


def bar(frac, text):
    n = int(20 * frac)
    print(f"{text} [{'#' * n}{'-' * (20 - n)}] {int(100 * frac)}%", flush=True)


def train(a):
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    pool = Pool(a.workers)
    X, Y, hist = None, None, []
    pol, net, t0 = None, MLP([OBS_DIM, a.hidden, a.hidden, 4], rng), time.time()
    for it in range(a.iters + 1):
        beta = 1.0 if it == 0 else max(0.0, 0.5 - 0.25 * (it - 1))
        n_eps = a.first_eps if it == 0 else a.eps
        w = None if it == 0 else (pol.net.W, pol.net.b, pol.mu, pol.sd)
        jobs = [(1000 * (it + 1) + 10 * k + a.seed, w, beta, n_eps // a.workers, 1000 if it == 0 else a.max_steps) for k in range(a.workers)]
        res = pool.map(rollout, jobs)
        xn, yn = np.concatenate([r[0] for r in res]), np.concatenate([r[1] for r in res])
        wins, tot = sum(r[2] for r in res), sum(r[3] for r in res)
        X = xn if X is None else np.concatenate([X, xn]); Y = yn if Y is None else np.concatenate([Y, yn])
        mu, sd = (X.mean(0), X.std(0) + 1e-3) if it == 0 else (pol.mu, pol.sd)
        net.fit(((X - mu) / sd).astype(np.float64), Y.astype(np.float64), epochs=a.epochs0 if it == 0 else a.epochs, lr=a.lr, rng=rng)
        pol = Policy(net, mu, sd)
        loss = float(np.mean((net.forward((X - mu) / sd) - Y) ** 2))
        hist.append(dict(iter=it, beta=beta, learner_success=f"{wins}/{tot}" if beta == 0 else "mixed", samples=len(X), loss=loss, minutes=(time.time() - t0) / 60))
        bar((it + 1) / (a.iters + 1), f"DAgger iter {it}/{a.iters}  beta={beta:.2f}  success while driving {wins}/{tot}  data {len(X)}  loss {loss:.4f}")
        pol.save(os.path.join(a.out, "policy.npz"))
    with open(os.path.join(a.out, "dagger_log.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, list(hist[0])); w.writeheader(); w.writerows(hist)
    pool.close()


CASES = {"nominal": dict(), "payload+damping (1 kg, x3)": dict(shift_fn=lambda rng: dict(payload=1.0, damping=3.0)),
         "delay 10 ms": dict(delay_steps=5), "noise 1e-4/1e-2": dict(noise=(1e-4, 1e-2))}


def evaluate(a):
    pol = Policy.load(a.eval)
    print(f"{'case':28s} success  err_mm(median)  policy-steps  effort  ik_fail")
    for name, kw in CASES.items():
        env = PickPlaceEnv(seed=100, **kw); ok, errs, steps, eff, ikf = 0, [], [], [], 0
        for ep in range(a.episodes):
            o, _ = env.reset()
            for k in range(1300):
                o, r, te, tr, info = env.step(pol.act(o))
                if te or tr: break
            ok += info["success"]; errs.append(1000 * info["err_xy"]); steps.append(k + 1); eff.append(info["effort"]); ikf += info["ik_fail"]
        print(f"{name:28s} {ok:3d}/{a.episodes}  {np.median(errs):10.1f}  {np.mean(steps):12.0f}  {np.mean(eff):6.3f}  {ikf:6d}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=8); ap.add_argument("--first-eps", type=int, default=60); ap.add_argument("--eps", type=int, default=30)
    ap.add_argument("--max-steps", type=int, default=900); ap.add_argument("--epochs0", type=int, default=40); ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--hidden", type=int, default=256); ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1); ap.add_argument("--out", default="runs/il"); ap.add_argument("--eval", default=None); ap.add_argument("--episodes", type=int, default=40)
    a = ap.parse_args()
    evaluate(a) if a.eval else train(a)