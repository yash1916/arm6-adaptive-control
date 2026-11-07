"""Dynamics that CHANGE in the middle of an episode, unknown to the controller.

The arm starts exactly as modelled (random objects, 360-degree tasks).  When the arm starts to lift the object, the plant changes:
  payload   a payload of 0.8 kg appears at the tool tip                       (inside the training range)
  damping   all joint damping x3                                             (edge of the training range)
  both      payload 1.0 kg and damping x3                                    (combined)
  heavy     a payload of 2.0 kg appears                                      (twice the largest payload seen in training)
Every controller sees the same tasks.  Reported: success, place error, rms joint error over the whole episode, and rms joint
error from the moment of the change to the end of the episode (post-change).

  python3 eval_shift.py --shifts payload damping both heavy --seeds 10 11 12 --tasks 6 --out results/shift_raw.csv \
      --nets hist=runs/h1/nn_hist_s1.npz,runs/h2/nn_hist_s2.npz,runs/h3/nn_hist_s3.npz nohist=...
"""
import os, csv, time, argparse
import numpy as np, mujoco
from pick_place import PickPlace, sample_task
from objects import sample_object
from randomize import DynamicsRandomizer
from controllers import ComputedTorque, NominalModel, PID, AdaptiveSL
from residual_net import ResidualNet

SHIFTS = {"payload": dict(at="lift", payload=0.8), "damping": dict(at="lift", damping=3.0),
          "both": dict(at="lift", payload=1.0, damping=3.0), "heavy": dict(at="lift", payload=2.0),
          # changes OUTSIDE the family the adaptive law estimates (it assumes one mass scale, one damping scale, one payload):
          "perjoint": "perjoint", "permass": "permass", "friction": "friction", "mixed": "mixed"}
NOMINAL = dict(mass_scale=1.0, damping_scale=1.0, payload=0.0)
KP = np.array([300, 500, 300, 40, 30, 15.]); KD = np.array([40, 60, 40, 4, 3, 1.5]); KI = 2 * KP; IL = np.array([20, 20, 15, 3, 2, 1.])
def make_shift(name, seed):
    """Fixed shifts, or random ones from outside the adaptive law's family (same draw for every controller on a task)."""
    spec = SHIFTS[name]
    if isinstance(spec, dict):
        return spec
    g = np.random.default_rng(seed)
    out = dict(at="lift")
    if name in ("perjoint", "mixed"):
        out["damping_vec"] = np.exp(g.uniform(np.log(0.5), np.log(4.0), 6))
    if name in ("permass", "mixed"):
        out["mass_vec"] = g.uniform(0.7, 1.8, 8)
    if name in ("friction", "mixed"):
        out["friction_vec"] = g.uniform(0.5, 1.5, 6) * np.array([0.6, 0.8, 0.5, 0.2, 0.15, 0.08])
    if name == "mixed":
        out["payload"] = 0.8
    return out


COLS = ["delay", "noise", "shift", "seed", "ctrl", "ep", "success", "err_mm", "rms_mrad", "post_mrad", "effort", "chatter", "clipped", "reason"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shifts", nargs="+", default=list(SHIFTS), choices=list(SHIFTS)); ap.add_argument("--seeds", nargs="+", type=int, default=[10])
    ap.add_argument("--tasks", type=int, default=6); ap.add_argument("--out", default="results/shift_raw.csv")
    ap.add_argument("--nets", nargs="*", default=[], help="label=path1,path2,... : each path is one trained network (evaluated separately)")
    ap.add_argument("--no-baselines", action="store_true")
    ap.add_argument("--nets-wn", type=float, default=20.0, help="bandwidth (rad/s) of the computed-torque controller the networks sit on (ki = wn^3/4)")
    ap.add_argument("--delay", type=int, default=0, help="control latency in simulation steps (1 step = 2 ms)")
    ap.add_argument("--cti", nargs="*", default=[], metavar="NAME=wn", help="computed torque + integral at another bandwidth, e.g. ctihi=60 (ki = wn^3/4)")
    ap.add_argument("--adaptive", nargs="*", default=[], metavar="NAME=kd,gamma", help="adaptive controllers to add, e.g. asl=20,4  (kd_gain, adaptation gain)")
    ap.add_argument("--noise", nargs=2, type=float, default=None, metavar=("SIGMA_Q", "SIGMA_QD"), help="Gaussian noise on the joint angle (rad) and velocity (rad/s) readings the controller sees")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    new = not os.path.exists(a.out); f = open(a.out, "a", newline=""); w = csv.DictWriter(f, COLS)
    if new: w.writeheader()
    rd = DynamicsRandomizer(); mdl = NominalModel(rd.nominal); FL0 = rd.plant.dof_frictionloss.copy()
    ctrls = {}
    if not a.no_baselines:
        ctrls["pid"] = lambda: PID(KP, KI, KD, IL)
        ctrls["cti"] = lambda: ComputedTorque(wn=20, model=mdl, ki=2000.0)
    for spec in a.cti:
        name, v = spec.split("="); wn = float(v)
        ctrls[name] = (lambda wn=wn: ComputedTorque(wn=wn, model=mdl, ki=wn ** 3 / 4.0))
    for spec in a.adaptive:
        name, v = spec.split("="); kd, g = (float(x) for x in v.split(","))
        ctrls[name] = (lambda kd=kd, g=g: AdaptiveSL(rd.nominal, kd_gain=kd, gamma=(g, g, g)))
    for spec in a.nets:
        label, paths = spec.split("=")
        for i, p in enumerate(paths.split(",")):
            net = ResidualNet.load(p)
            ctrls[f"{label}{i + 1}"] = (lambda net=net: ComputedTorque(wn=a.nets_wn, model=mdl, ki=a.nets_wn ** 3 / 4.0, residual=net))
    for sh in a.shifts:
        for seed in a.seeds:
            rng = np.random.default_rng(7000 + 100 * list(SHIFTS).index(sh) + seed)
            eps = [(sample_task(rng), sample_object(rng)) for _ in range(a.tasks)]
            for name, mk in ctrls.items():
                n_ok = 0
                for i, ((cube, yaw, goal), obj) in enumerate(eps):
                    rd.apply(NOMINAL); rd.plant.dof_frictionloss[:] = FL0                                   # fresh nominal arm every episode (the shift edits the plant)
                    pp = PickPlace(rd.plant, mujoco.MjData(rd.plant), mk(), nominal=rd.nominal)
                    pp.delay_steps = a.delay
                    if a.noise: pp.noise = (a.noise[0], a.noise[1], 1000 * seed + i)
                    r = pp.episode(cube, yaw, goal, obj=obj, payload=0.0, shift=make_shift(sh, 5000 * seed + i)); ok = r["planned"]
                    w.writerow(dict(delay=a.delay, noise=("%g/%g" % tuple(a.noise)) if a.noise else "0/0", shift=sh, seed=seed, ctrl=name, ep=i, success=int(r["success"]), err_mm=1000 * r["err_xy"] if ok else "",
                                    rms_mrad=1000 * r["rms_track"] if ok else "", post_mrad=1000 * r["post_shift_rms"] if ok else "", effort=r["effort"] if ok else "", chatter=r["chatter"] if ok else "", clipped=100 * r["sat_frac"] if ok else "", reason=r["reason"]))
                    f.flush(); n_ok += r["success"]
                print(f"[{sh} seed {seed}] {name:8s}: {n_ok}/{a.tasks} success", flush=True)