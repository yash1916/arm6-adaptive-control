"""Compare PID, computed torque (CT), CT + integral, and MPC when the controller's model is wrong or the motors are weak.

  python3 run_model_based.py                 # 20 random tasks, all three conditions
  python3 run_model_based.py 30 mismatch     # 30 tasks, one condition  (nominal | mismatch | weak)
  python3 run_model_based.py 20 weak pid ct  # only some controllers

Conditions (the controller always believes the NOMINAL model; the simulation uses the perturbed PLANT):
  nominal   plant == nominal model
  mismatch  arm mass/inertia x1.3, cube 0.4 kg (4x), joint damping x2
  weak      tool speed x3 and motor torque limits x0.10 so that they actually bind (controllers know the limits)
"""
import sys, time
import numpy as np
import mujoco
import pick_place
from pick_place import PickPlace, sample_task
from plant import make_plant
from controllers import PID, ComputedTorque, NominalModel, MPC

KP = np.array([300, 500, 300, 40, 30, 15.]); KD = np.array([40, 60, 40, 4, 3, 1.5])
KI = 2.0 * KP; I_LIM = np.array([20, 20, 15, 3, 2, 1.])

CONDITIONS = {"nominal": dict(),
              "mismatch": dict(mass_scale=1.3, cube_mass=0.4, damping_scale=2.0),
              "weak": dict(torque_scale=float(__import__("os").environ.get("TS","0.10")))}
SPEED = {"nominal": 1.0, "mismatch": 1.0, "weak": float(__import__("os").environ.get("SP", "3.0"))}


def make(name, nominal):
    mdl = NominalModel(nominal)
    return {"pid": lambda: PID(KP, KI, KD, I_LIM),
            "ct": lambda: ComputedTorque(wn=20, model=mdl),
            "cti": lambda: ComputedTorque(wn=20, model=mdl, ki=2000.0),
            "mpc": lambda: MPC(mdl)}[name]()


if __name__ == "__main__":
    args = sys.argv[1:]
    n_ep = int(args[0]) if args else 20
    conds = [a for a in args[1:] if a in CONDITIONS] or list(CONDITIONS)
    names = [a for a in args[1:] if a not in CONDITIONS] or ["pid", "ct", "cti", "mpc"]
    rng = np.random.default_rng(7)
    tasks = [sample_task(rng) for _ in range(n_ep)]
    V0 = (pick_place.V_FAST, pick_place.V_SLOW)
    for cond in conds:
        pick_place.V_FAST, pick_place.V_SLOW = V0[0] * SPEED[cond], V0[1] * SPEED[cond]
        print(f"\n=== condition: {cond}  {CONDITIONS[cond]}  tool speed x{SPEED[cond]}  ({n_ep} tasks, same for all) ===")
        print(f"{'ctrl':5s} {'ok@20mm':>8s} {'ok@5mm':>7s} {'mean err mm':>12s} {'rms track mrad':>15s} {'clipped %':>10s} {'s/episode':>10s}")
        for name in names:
            plant, nominal = make_plant(**CONDITIONS[cond])
            d = mujoco.MjData(plant)
            pp = PickPlace(plant, d, make(name, nominal), nominal=nominal); pp.torque_aware = __import__("os").environ.get("TA", "1") == "1"
            t0 = time.time(); res = [pp.episode(c, y, g) for c, y, g in tasks]; dt = (time.time() - t0) / n_ep
            pl = [r for r in res if r["planned"]]
            ok = [r for r in pl if r["success"]]
            err = np.mean([r["err_xy"] for r in pl]) * 1000
            print(f"{name:5s} {len(ok):4d}/{n_ep:<3d} {sum(r['err_xy'] < 0.005 for r in ok):3d}/{n_ep:<3d} {err:12.1f} "
                  f"{1000*np.mean([r['rms_track'] for r in pl]):15.1f} {100*np.mean([r['sat_frac'] for r in pl]):10.1f} {dt:10.1f}")