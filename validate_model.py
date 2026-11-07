"""Checks that the MuJoCo model reproduces the CAD data, then a hold/track test with and without gravity compensation."""
import os; os.environ.setdefault("MUJOCO_GL", "egl")
import json, numpy as np, mujoco
meta = json.load(open("cad_props.json")); P = meta["parts"]; O = {k: np.array(v) * 1e-3 for k, v in meta["part_origin_world_mm"].items()}
m = mujoco.MjModel.from_xml_path("arm6.xml"); d = mujoco.MjData(m)
g = 9.81
# ---- 1. total mass and COM at q=0 vs CAD
parts = [("Base","base"),("Link1Turntable","link1_turntable"),("Link2Upper","link2_upper"),("Link3Forearm","link3_forearm"),
         ("Link4ForearmRoll","link4_forearm_roll"),("Link5Wrist","link5_wrist"),("Link6Gripper","link6_gripper"),("FingerL","finger"),("FingerR","finger")]
mass = np.array([P[p]["mass_kg"] for _, p in parts]); com = np.array([O[n] + np.array(P[p]["com_m"]) for n, p in parts])
mujoco.mj_forward(m, d)
print(f"[mass]  CAD {mass.sum():.4f} kg | MuJoCo {m.body_subtreemass[0]:.4f} kg")
print(f"[COM]   CAD {np.round((mass[:,None]*com).sum(0)/mass.sum(),5)} | MuJoCo {np.round(d.subtree_com[0],5)}")
# ---- 2. gravity hold torques at q=0 vs hand calc from CAD
dist = {"joint1": parts[1:], "joint2": parts[2:], "joint3": parts[3:], "joint4": parts[4:], "joint5": parts[5:], "joint6": parts[6:]}
hand = []
for j in range(1, 7):
    jn = f"joint{j}"; p0 = np.array(meta["joints_mm"][jn]["pos"]) * 1e-3; ax = np.array(meta["joints_mm"][jn]["axis"], float)
    tau = 0.0
    for n, p in dist[jn]:
        r = O[n] + np.array(P[p]["com_m"]) - p0; F = np.array([0, 0, -g * P[p]["mass_kg"]])
        tau += -np.dot(np.cross(r, F), ax)                    # torque needed to HOLD = minus the gravity torque
    hand.append(tau)
print("[gravity hold torque at q=0, N m]\n   CAD hand calc:", np.round(hand, 4), "\n   MuJoCo bias  :", np.round(d.qfrc_bias, 4))
print("   max |diff| =", float(np.abs(np.array(hand) - d.qfrc_bias).max()))
# ---- 3. hold a pose: PD vs PD + gravity compensation
KP = np.array([300, 500, 300, 40, 30, 15.]); KD = np.array([40, 60, 40, 4, 3, 1.5]); qT = np.array([0.5, -0.4, 0.8, 0.3, -0.5, 0.4])
def hold(gc, T=4.0):
    mujoco.mj_resetData(m, d); n = int(T / m.opt.timestep); sat = 0
    for _ in range(n):
        tau = KP * (qT - d.qpos) - KD * d.qvel
        if gc:
            mujoco.mj_forward(m, d); tau = tau + d.qfrc_bias
        d.ctrl[:] = np.clip(tau, -m.actuator_ctrlrange[:, 1], m.actuator_ctrlrange[:, 1]); mujoco.mj_step(m, d)
    return np.abs(qT - d.qpos), d.qpos.copy()
e0, _ = hold(False); e1, q1 = hold(True)
print("[hold test]  final |joint error| rad\n   PD only          :", np.round(e0, 4), "\n   PD + gravity comp:", np.round(e1, 5))
# ---- 4. render the posed arm
r = mujoco.Renderer(m, 720, 1280); cam = mujoco.MjvCamera(); cam.lookat[:] = [0.2, -0.05, 0.3]; cam.distance = 1.25; cam.azimuth = 140; cam.elevation = -18
d.qpos[:] = q1; mujoco.mj_forward(m, d); r.update_scene(d, cam); import imageio.v3 as iio
iio.imwrite("arm6_mujoco.png", r.render()); print("rendered arm6_mujoco.png")
