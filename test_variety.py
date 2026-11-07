"""Stress test: random 360-degree tasks + random objects (shape, size, mass) + random dynamics (arm mass, damping, payload).
  python3 test_variety.py 40 pid cti
"""
import sys, time
import numpy as np, mujoco
from pick_place import PickPlace, sample_task
from objects import sample_object
from randomize import DynamicsRandomizer
from controllers import PID, ComputedTorque, NominalModel
KP = np.array([300, 500, 300, 40, 30, 15.]); KD = np.array([40, 60, 40, 4, 3, 1.5]); KI = 2 * KP; IL = np.array([20, 20, 15, 3, 2, 1.])

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    names = sys.argv[2:] or ["pid", "cti"]
    rng = np.random.default_rng(int(__import__("os").environ.get("SEED","11")))
    rd = DynamicsRandomizer()
    eps = [(sample_task(rng), sample_object(rng), rd.sample(rng)) for _ in range(n)]       # same episodes for every controller
    for name in names:
        mdl = NominalModel(rd.nominal)
        c = PID(KP, KI, KD, IL) if name == "pid" else ComputedTorque(wn=20, model=mdl, ki=2000.0)
        pp = PickPlace(rd.plant, mujoco.MjData(rd.plant), c); res = []; t = time.time()
        for (cube, yaw, goal), obj, par in eps:
            rd.apply(par); res.append(pp.episode(cube, yaw, goal, obj=obj, payload=par["payload"]))
        ok = [r for r in res if r["success"]]
        print(f"\n{name}: {len(ok)}/{n} success, mean place error {1000*np.mean([r['err_xy'] for r in res if r['planned']]):.1f} mm, {(time.time()-t)/n:.1f} s/episode")
        for shape in ("box", "cyl"):
            rs = [r for r in res if r["obj"]["shape"] == shape]
            print(f"   {shape}: {sum(r['success'] for r in rs)}/{len(rs)}")
        for (task, obj, par), r in zip(eps, res):
            if not r["success"]:
                print("   FAIL", r["reason"], f"{1000*r['err_xy']:.0f}mm", obj["shape"], "half", np.round(obj["half"], 3), "h", round(obj["h"], 3), "m", round(obj["mass"], 2),
                      "| payload", round(par["payload"], 2), "damp", round(par["damping_scale"], 2), "massx", round(par["mass_scale"], 2))