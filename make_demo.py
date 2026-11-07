import os; os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np, mujoco, imageio.v3 as iio
m = mujoco.MjModel.from_xml_path("arm6.xml"); d = mujoco.MjData(m)
KP = np.array([300, 500, 300, 40, 30, 15.]); KD = np.array([40, 60, 40, 4, 3, 1.5])
A = np.array([0.8, 0.5, 0.7, 0.8, 0.6, 1.0]); W = np.array([0.30, 0.25, 0.35, 0.40, 0.30, 0.50])
r = mujoco.Renderer(m, 480, 640); cam = mujoco.MjvCamera(); cam.lookat[:] = [0.2, -0.04, 0.3]; cam.distance = 1.25; cam.azimuth = 135; cam.elevation = -15
frames, fps = [], 30; step_per_frame = int(1 / fps / m.opt.timestep)
for i in range(int(6 * fps)):
    for _ in range(step_per_frame):
        qt = A * np.sin(W * 2 * np.pi * d.time) * min(1, d.time / 1.0)
        mujoco.mj_forward(m, d)
        d.ctrl[:] = np.clip(KP * (qt - d.qpos) - KD * d.qvel + d.qfrc_bias, -m.actuator_ctrlrange[:, 1], m.actuator_ctrlrange[:, 1]); mujoco.mj_step(m, d)
    r.update_scene(d, cam); frames.append(r.render())
iio.imwrite("arm6_demo.mp4", np.stack(frames), fps=fps); print("wrote arm6_demo.mp4", len(frames), "frames; final max|q|", np.abs(d.qpos).max().round(3))
