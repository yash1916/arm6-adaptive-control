"""Stage 4, step 1: check the RL environment with the scripted policy and measure its speed.
  python3 check_env.py --episodes 40
"""
import argparse, time, numpy as np
from rl_env import PickPlaceEnv, ScriptedPolicy

CASES = {"nominal": dict(), "payload+damping (1 kg, x3)": dict(shift_fn=lambda rng: dict(payload=1.0, damping=3.0)),
         "delay 10 ms": dict(delay_steps=5), "noise 1e-4/1e-2": dict(noise=(1e-4, 1e-2))}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--episodes", type=int, default=40); a = ap.parse_args()
    print(f"{'case':28s} success  err_mm  policy-steps  ik_fail  sim-steps/s")
    for name, kw in CASES.items():
        env = PickPlaceEnv(seed=3, **kw); pol = ScriptedPolicy(env)
        ok, errs, steps, ikf, t0, tot = 0, [], [], 0, time.time(), 0
        for ep in range(a.episodes):
            o, _ = env.reset(); pol.reset()
            for k in range(1300):
                o, r, te, tr, info = env.step(pol(o)); tot += 1
                if te or tr: break
            ok += info["success"]; errs.append(1000 * info["err_xy"]); steps.append(k + 1); ikf += info["ik_fail"]
        print(f"{name:28s} {ok:3d}/{a.episodes}  {np.median(errs):6.1f}  {np.mean(steps):10.0f}  {ikf:7d}  {10 * tot / (time.time() - t0):9.0f}", flush=True)