"""Stage 4, step 4: fair evaluation of learned policies on tasks NONE of them saw in training or model selection.

  python3 eval_policies.py --episodes 40 --workers 4 --out results/stage4_eval.csv \
      --policy il=runs/il_policy.npz --policy ppo1=runs/ppo_s1/last_policy.npz --policy ppo2=runs/ppo_s2/last_policy.npz ... --expert

Conditions (task seed 5000, different from training seeds and from the seed 777 used for model selection during PPO):
  nominal          unchanged plant
  shift            1 kg payload + damping x3 at the lift
  friction         random Coulomb friction + payload + damping at the lift (out of the PPO training mix only in its combination)
  noise            joint sensor noise 1e-4 rad / 1e-2 rad/s
  delay 10 ms      inside the PPO training range (0-10 ms)
  delay 12 ms, 14 ms   just OUTSIDE the training range (16 ms and beyond makes the 60 rad/s controller itself unstable, even for the scripted expert)
Every policy sees exactly the same tasks in each condition.  Reported per policy: success rate, median place error, mean episode length
(policy steps, 50 per second), rms torque effort.  Policies named ppo<N> are pooled for a mean +/- std across seeds.
"""
import argparse, csv, time
import numpy as np
from multiprocessing import Pool
from rl_env import PickPlaceEnv, StatelessExpert
from imitate import Policy
from train_ppo import random_shift

FRIC = np.array([0.6, 0.8, 0.5, 0.2, 0.15, 0.08])
COND = {
    "nominal": dict(),
    "shift": dict(shift_fn=lambda r: dict(payload=1.0, damping=3.0)),
    "friction": dict(shift_fn=lambda r: dict(payload=0.8, damping=2.0, friction_vec=r.uniform(0.5, 1.5, 6) * FRIC)),
    "noise": dict(noise=(1e-4, 1e-2)),
    "delay 10 ms": dict(delay_steps=5),
    "delay 12 ms": dict(delay_steps=6),
    "delay 14 ms": dict(delay_steps=7),
}


def run(args):
    name, path, cond, n_eps, seed = args
    env = PickPlaceEnv(seed=seed, **COND[cond])
    expert = StatelessExpert(env) if path is None else None
    pol = Policy.load(path) if path else None
    ok, errs, steps, eff = 0, [], [], []
    for _ in range(n_eps):
        o, _ = env.reset()
        for k in range(1300):
            a = expert(o) if expert else pol.act(o)
            o, r, te, tr, info = env.step(a)
            if te or tr: break
        ok += int(info["success"]); errs.append(1000 * info["err_xy"]); steps.append(k + 1); eff.append(info["effort"])
    return dict(policy=name, cond=cond, episodes=n_eps, success=ok / n_eps, err_mm_median=float(np.median(errs)), steps=float(np.mean(steps)), effort=float(np.mean(eff)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", action="append", default=[], metavar="NAME=PATH"); ap.add_argument("--expert", action="store_true")
    ap.add_argument("--episodes", type=int, default=40); ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=5000); ap.add_argument("--out", default="results/stage4_eval.csv")
    a = ap.parse_args()
    items = [(n, p) for n, p in (s.split("=") for s in a.policy)] + ([("expert", None)] if a.expert else [])
    jobs = [(n, p, c, a.episodes, a.seed) for n, p in items for c in COND]
    t0 = time.time(); rows = []
    with Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(run, jobs)):
            rows.append(r); f = (i + 1) / len(jobs)
            print(f"Evaluating [{'#' * int(20 * f)}{'-' * (20 - int(20 * f))}] {int(100 * f)}%  {r['policy']} / {r['cond']}: {r['success']:.2f}", flush=True)
    import os; os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0])); w.writeheader(); w.writerows(rows)
    # table: groups ppo1..ppoN pooled
    group = lambda n: "ppo (mean±std over seeds)" if n.startswith("ppo") and n[3:].isdigit() else n
    names = []
    for r in rows:
        g = group(r["policy"])
        if g not in names: names.append(g)
    print(f"\n{'success rate':28s}" + "".join(f"{c:>13s}" for c in COND))
    for g in names:
        line = f"{g:28s}"
        for c in COND:
            v = [r["success"] for r in rows if group(r["policy"]) == g and r["cond"] == c]
            line += f"{np.mean(v):7.2f}±{np.std(v):.2f}" if len(v) > 1 else f"{v[0]:13.2f}"
        print(line)
    print(f"\n{'mean episode length (steps)':28s}" + "".join(f"{c:>13s}" for c in COND))
    for g in names:
        print(f"{g:28s}" + "".join(f"{np.mean([r['steps'] for r in rows if group(r['policy']) == g and r['cond'] == c]):13.0f}" for c in COND))
