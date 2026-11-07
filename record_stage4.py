"""Side-by-side video of the scripted expert and a learned policy doing the SAME pick-and-place task (real time, 25 fps).
  python3 record_stage4.py --policy runs/ppo_s1/last_policy.npz --tasks 2 --out stage4_ppo_vs_expert.mp4
Left: scripted expert.  Right: the learned policy.  The clock in each panel stops when that run ends (placed = green)."""
import os, argparse
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np, mujoco, imageio.v3 as iio
from PIL import Image, ImageDraw
from rl_env import PickPlaceEnv, StatelessExpert
from imitate import Policy
from pick_place import sample_task

ap = argparse.ArgumentParser()
ap.add_argument("--policy", required=True); ap.add_argument("--label", default="PPO policy"); ap.add_argument("--tasks", type=int, default=2)
ap.add_argument("--seed", type=int, default=2024); ap.add_argument("--out", default="stage4_ppo_vs_expert.mp4")
a = ap.parse_args()

pol = Policy.load(a.policy)
W, H, FPS = 540, 400, 25
envs = [PickPlaceEnv(seed=1), PickPlaceEnv(seed=1)]
r = [mujoco.Renderer(e.m, H, W) for e in envs]
cam = mujoco.MjvCamera(); cam.lookat[:] = [0.1, 0.0, 0.1]; cam.distance = 1.35; cam.azimuth = 140; cam.elevation = -28
expert = StatelessExpert(envs[0])
rng = np.random.default_rng(a.seed)
frames = []


def tile(i, name, done_t, ok):
    r[i].update_scene(envs[i].d, cam)
    img = Image.fromarray(r[i].render().copy()); dr = ImageDraw.Draw(img)
    col = (200, 255, 200) if ok else (255, 255, 255)
    dr.rectangle([0, 0, W, 40], fill=col)
    dr.text((8, 4), name, fill=(20, 20, 20))
    dr.text((8, 20), f"time {done_t:5.1f} s" + ("   placed" if ok else ""), fill=(20, 20, 20))
    return np.array(img)


for t in range(a.tasks):
    task = sample_task(rng)
    obs = [e.reset(task)[0] for e in envs]
    done, t_end, ok, k = [False, False], [0.0, 0.0], [False, False], 0
    while not all(done) and k < 1300:
        for i, e in enumerate(envs):
            if done[i]: continue
            act = expert(obs[i]) if i == 0 else pol.act(obs[i])
            obs[i], rew, te, tr, info = e.step(act)
            t_end[i] = (k + 1) * 0.02
            if te or tr: done[i], ok[i] = True, bool(info["success"])
        k += 1
        if k % 2 == 0:                                   # 50 policy steps/s -> 25 fps
            frames.append(np.hstack([tile(0, "Scripted expert", t_end[0], ok[0]), tile(1, a.label, t_end[1], ok[1])]))
    for _ in range(FPS):                                 # hold the final frame for a second
        frames.append(frames[-1])
iio.imwrite(a.out, np.stack(frames), fps=FPS)
print("wrote", a.out, len(frames), "frames,", len(frames) / FPS, "s")
os._exit(0)