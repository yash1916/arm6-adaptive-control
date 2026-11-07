"""Benchmark of the classical controllers on random 360-degree pick-and-place with random objects.

  python3 benchmark_classical.py --cond varied --seeds 0 1 2 --tasks 12
  python3 benchmark_classical.py --cond nominal varied stress --seeds 0 --tasks 6 --ctrl pid cti      (subset)

Conditions (the controllers always believe the NOMINAL arm; the simulated arm differs):
  nominal  arm as modelled, no payload; objects still random (0.05-0.5 kg, boxes and cylinders)
  varied   arm mass x0.8-1.5, joint damping x0.5-3, tool payload 0-1 kg          (the ranges the network is trained on)
  stress   arm mass x1.5-2.0, joint damping x3-5, tool payload 1-2 kg            (outside those ranges)
For one (condition, seed) every controller runs the SAME tasks, objects and arms.  One CSV row per episode is appended to --out,
so a job can be stopped, restarted or split over cluster nodes (one job per condition/seed) and the files merged later.
Metrics: success (<20 mm and resting on the floor), place error, rms joint tracking error, settling time of the tool tip (1 mm,
after the grasp and place moves), normalised rms torque, torque chatter, share of steps with a clipped command.
"""
import os, sys, csv, time, argparse
import numpy as np, mujoco
from pick_place import PickPlace, sample_task
from objects import sample_object
from randomize import DynamicsRandomizer
from controllers import PD, PID, PDGravity, ComputedTorque, NominalModel, MPC, AdaptiveSL

KP = np.array([300, 500, 300, 40, 30, 15.]); KD = np.array([40, 60, 40, 4, 3, 1.5]); KI = 2 * KP; IL = np.array([20, 20, 15, 3, 2, 1.])
ALL = ["pd", "pdg", "pid", "ct", "cti", "mpc", "ctnn", "ctinn", "asl"]   # asl = adaptive control (Slotine-Li); ctnn / ctinn = computed torque (+ integral) + the trained network (--net)
CONDS = {"nominal": dict(mass_scale=(1.0, 1.0), damping_scale=(1.0, 1.0), payload=(0.0, 0.0)),
         "varied":  dict(mass_scale=(0.8, 1.5), damping_scale=(0.5, 3.0), payload=(0.0, 1.0)),
         "stress":  dict(mass_scale=(1.5, 2.0), damping_scale=(3.0, 5.0), payload=(1.0, 2.0))}
COLS = ["cond", "seed", "ctrl", "ep", "planned", "success", "err_mm", "rms_mrad", "settle_s", "unsettled", "effort", "chatter",
        "clipped", "wall_s", "shape", "obj_mass", "payload", "mass_scale", "damping_scale", "reason"]


def make(name, rd, net=None, res_scale=1.0):
    mdl = NominalModel(rd.nominal)
    return {"pd": lambda: PD(KP, KD), "pdg": lambda: PDGravity(rd.nominal, KP, KD), "pid": lambda: PID(KP, KI, KD, IL),
            "ct": lambda: ComputedTorque(wn=20, model=mdl), "cti": lambda: ComputedTorque(wn=20, model=mdl, ki=2000.0),
            "mpc": lambda: MPC(mdl),
            "asl": lambda: AdaptiveSL(rd.nominal, kd_gain=160.0, gamma=(128.0, 128.0, 128.0)),
            "ctnn": lambda: ComputedTorque(wn=20, model=mdl, residual=net, res_scale=res_scale),
            "ctinn": lambda: ComputedTorque(wn=20, model=mdl, ki=2000.0, residual=net, res_scale=res_scale)}[name]()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", nargs="+", default=["varied"], choices=list(CONDS)); ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--tasks", type=int, default=10); ap.add_argument("--ctrl", nargs="+", default=ALL, choices=ALL)
    ap.add_argument("--out", default="results/classical_raw.csv")
    ap.add_argument("--res-scale", type=float, default=1.0, help="multiply the network correction by this (1 = full, 0.5 = half)")
    ap.add_argument("--net", default="nn_150.npz", help="trained network for ctnn / ctinn (frozen: no learning during evaluation)")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    new = not os.path.exists(a.out)
    f = open(a.out, "a", newline=""); w = csv.DictWriter(f, COLS)
    if new: w.writeheader()
    rd = DynamicsRandomizer()
    net = None
    if any(c in a.ctrl for c in ("ctnn", "ctinn")):
        from residual_net import ResidualNet
        net = ResidualNet.load(a.net)
    for cond in a.cond:
        for seed in a.seeds:
            rng = np.random.default_rng(1000 * seed + list(CONDS).index(cond))
            eps = [(sample_task(rng), sample_object(rng), rd.sample(rng, CONDS[cond])) for _ in range(a.tasks)]
            for name in a.ctrl:
                pp = PickPlace(rd.plant, mujoco.MjData(rd.plant), make(name, rd, net, a.res_scale), nominal=rd.nominal); t_all = time.time(); n_ok = 0
                for i, ((cube, yaw, goal), obj, par) in enumerate(eps):
                    rd.apply(par); t0 = time.time()
                    r = pp.episode(cube, yaw, goal, obj=obj, payload=par["payload"])
                    ok = r["planned"]
                    w.writerow(dict(cond=cond, seed=seed, ctrl=name, ep=i, planned=int(ok), success=int(r["success"]),
                                    err_mm=1000 * r["err_xy"] if ok else "", rms_mrad=1000 * r["rms_track"] if ok else "",
                                    settle_s=r["settle_s"] if ok else "", unsettled=r["unsettled"] if ok else "", effort=r["effort"] if ok else "",
                                    chatter=r["chatter"] if ok else "", clipped=100 * r["sat_frac"] if ok else "", wall_s=time.time() - t0,
                                    shape=obj["shape"], obj_mass=obj["mass"], payload=par["payload"], mass_scale=par["mass_scale"],
                                    damping_scale=par["damping_scale"], reason=r["reason"]))
                    f.flush(); n_ok += r["success"]
                print(f"[{cond} seed {seed}] {name:3s}: {n_ok}/{a.tasks} success, {(time.time() - t_all) / a.tasks:.1f} s/episode", flush=True)