"""Run randomized pick-and-place episodes with different controllers and compare success rates.

  python3 run_pick_place.py                 # 20 episodes per controller
  python3 run_pick_place.py 40 pdg pid      # 40 episodes, only these controllers
  python3 run_pick_place.py 40 --speed 3    # 3x faster tool motion (a harder test)
  python3 run_pick_place.py 40 --payload 1  # unknown 1 kg payload on the tool (controllers' models do not know about it)
"""
import sys, time, copy
import numpy as np
import mujoco
from pick_place import PickPlace, sample_task
from controllers import PD, PID, PDGravity, ComputedTorque

KP = np.array([300, 500, 300, 40, 30, 15.]); KD = np.array([40, 60, 40, 4, 3, 1.5])
KI = 2.0 * KP; I_LIM = np.array([20, 20, 15, 3, 2, 1.])


def make(name, m):
    """Build a controller. Model-based ones get a NOMINAL copy of the plant model (taken now, before any payload is added),
    so later changes to the plant are invisible to them -- exactly like real, unknown model error."""
    nom = copy.copy(m)
    return {"pd": lambda: PD(KP, KD),
            "pid": lambda: PID(KP, KI, KD, I_LIM),
            "pdg": lambda: PDGravity(nom, KP, KD),
            "ct": lambda: ComputedTorque(wn=20, model_damping=True, model=nom),
            "cti": lambda: ComputedTorque(wn=20, model_damping=True, model=nom, ki=2000.0)}[name]()


if __name__ == "__main__":
    args = sys.argv[1:]
    speed, payload = 1.0, 0.0
    if "--speed" in args:
        i = args.index("--speed"); speed = float(args[i + 1]); del args[i:i + 2]
    if "--payload" in args:
        i = args.index("--payload"); payload = float(args[i + 1]); del args[i:i + 2]
    import pick_place
    pick_place.V_FAST *= speed; pick_place.V_SLOW *= speed
    n_ep = int(args[0]) if len(args) > 0 else 20
    names = args[1:] or ["pd", "pid", "pdg", "ct", "cti"]
    m = mujoco.MjModel.from_xml_path("arm6_pick.xml"); d = mujoco.MjData(m)
    rng = np.random.default_rng(123)
    tasks = [sample_task(rng) for _ in range(n_ep)]          # the SAME random tasks for every controller
    print(f"{n_ep} random episodes per controller (same tasks for all), tool speed x{speed}, unknown payload {payload} kg\n")
    print(f"{'controller':10s} {'ok@20mm':>8s} {'ok@5mm':>7s} {'lifted':>7s} {'IK fail':>8s} {'mean err':>9s} {'max err':>8s}   (place error in mm)")
    for name in names:
        pp = PickPlace(m, d, make(name, m))
        res, t0 = [], time.time()
        for cube, yaw, goal in tasks:
            res.append(pp.episode(cube, yaw, goal, payload=payload))
        ok = [r for r in res if r["success"]]
        planned = [r for r in res if r["planned"]]
        e = np.array([r["err_xy"] for r in planned]) * 1000 if planned else np.array([np.nan])
        ok5 = sum(1 for r in ok if r["err_xy"] < 0.005)
        print(f"{name:10s} {len(ok):4d}/{n_ep:<3d} {ok5:3d}/{n_ep:<3d} {sum(r['lifted'] for r in res):4d}/{n_ep:<2d} {n_ep - len(planned):8d} {np.mean(e):9.1f} {np.max(e):8.1f}")
        reasons = {}
        for r in res: reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
        print("           outcomes:", reasons)
