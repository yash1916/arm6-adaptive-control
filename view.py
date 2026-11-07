"""Interactive viewer: gravity-compensated PD follows a slow multi-joint sweep.  python3 view.py"""
import time, numpy as np, mujoco, mujoco.viewer
m = mujoco.MjModel.from_xml_path("arm6.xml"); d = mujoco.MjData(m)
KP = np.array([300, 500, 300, 40, 30, 15.]); KD = np.array([40, 60, 40, 4, 3, 1.5])
A = np.array([0.8, 0.5, 0.7, 0.8, 0.6, 1.0]); W = np.array([0.30, 0.25, 0.35, 0.40, 0.30, 0.50])
with mujoco.viewer.launch_passive(m, d) as v:
    while v.is_running():
        t0 = time.time()
        qt = A * np.sin(W * 2 * np.pi * d.time)
        mujoco.mj_forward(m, d)
        d.ctrl[:] = np.clip(KP * (qt - d.qpos) - KD * d.qvel + d.qfrc_bias, -m.actuator_ctrlrange[:, 1], m.actuator_ctrlrange[:, 1])
        mujoco.mj_step(m, d); v.sync()
        time.sleep(max(0, m.opt.timestep - (time.time() - t0)))
