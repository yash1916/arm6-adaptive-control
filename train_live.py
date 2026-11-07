"""Watch the arm work AND the neural network learn, at the same time.

Every task (random 360-degree positions, random object, random arm mass / damping / payload) is run twice on identical
conditions:   1) baseline  computed torque + integral
              2) the same controller + the neural network's torque correction
Every run feeds samples to the network, which trains in small steps while the simulation runs.  The overlay in the corner
shows samples, loss, how much of the model error the network explains on held-out tasks, and the live comparison.

  python3 train_live.py                          live MuJoCo window, runs until you close it
  python3 train_live.py --speed 8                faster than real time (default 5)
  python3 train_live.py --headless --tasks 60    no window, as fast as possible, prints one line per task
  python3 train_live.py --headless --tasks 40 --video demo.mp4 --video-last 2     also record the last 2 tasks
  python3 train_live.py --plot                   also open a live matplotlib window with the learning curves
Results: nn_residual.npz (network), training_log.csv, training_curve.png
"""
import os, sys, time, argparse, csv
ap = argparse.ArgumentParser()
ap.add_argument("--headless", action="store_true"); ap.add_argument("--tasks", type=int, default=None)
ap.add_argument("--speed", type=float, default=5.0); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--updates", type=float, default=0.25, help="gradient steps per recorded sample")
ap.add_argument("--video", default=None); ap.add_argument("--video-last", type=int, default=2)
ap.add_argument("--plot", action="store_true"); ap.add_argument("--out", default="nn_residual.npz")
ap.add_argument("--hidden", type=int, default=128)
ap.add_argument("--noise-train", action="store_true", help="each training task draws a sensor-noise level (none, small, medium, large) and the controller and network see noisy joint readings")
ap.add_argument("--history", action="store_true", help="also feed the network the controller's running averages of the recent model error (37 inputs instead of 19)")
ap.add_argument("--p-nominal", type=float, default=0.25, help="share of training tasks whose arm matches the model exactly, so the network learns to stay quiet when there is nothing to correct")
args = ap.parse_args()
if args.video:
    os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import mujoco
from pick_place import PickPlace, sample_task
from objects import sample_object
from randomize import DynamicsRandomizer, RANGES
from controllers import ComputedTorque, NominalModel
from residual_net import ResidualNet
from data_collect import Buffer, Collector

NOISE_LEVELS = [(0.0, 0.0), (1e-5, 1e-3), (1e-4, 1e-2), (5e-4, 5e-2)]
NOMINAL = dict(mass_scale=(1.0, 1.0), damping_scale=(1.0, 1.0), payload=(0.0, 0.0))
n_tasks = args.tasks if args.tasks is not None else (40 if args.headless else 10**9)
rng = np.random.default_rng(args.seed); rb = np.random.default_rng(args.seed + 1)
rd = DynamicsRandomizer(); mdl = NominalModel(rd.nominal)
N_IN = 37 if args.history else 19
net = ResidualNet(n_in=N_IN, hidden=(args.hidden, args.hidden), seed=args.seed); buf = Buffer(n_in=N_IN)
d = mujoco.MjData(rd.plant)
mk = lambda **kw: ComputedTorque(wn=20, model=mdl, ki=2000.0, i_limit=20.0, **kw)
pp_base, pp_nn = PickPlace(rd.plant, d, mk()), PickPlace(rd.plant, d, mk(residual=net))


class State:                                           # everything the overlay and the log need
    task = 0; run = "-"; val = False; ema = float("nan"); updates = 0; credit = 0.0; r2 = float("nan")
    r2_hist = []; r2_x = []; rms_base = []; rms_nn = []; last = ""; t0 = time.time()
S = State()


def spark(vals, n=28):
    chars = " .:-=+*#%@"; v = vals[-n:]
    return "".join(chars[int(np.clip(x, 0, 0.999) * len(chars))] for x in v) if v else "-"


def sink(x, y):                                        # one new sample arrives; train a little
    buf.add(x, y, validation=S.val)
    if S.val:
        return
    if not net.ready and buf.n >= 2000:
        net.set_norm(buf.X[:buf.n].astype(float), buf.Y[:buf.n].astype(float))
    if not net.ready:
        return
    S.credit += args.updates
    while S.credit >= 1.0:
        Xb, Yb = buf.batch(rb)
        loss = net.train_step(Xb, Yb); S.credit -= 1.0; S.updates += 1
        S.ema = loss if np.isnan(S.ema) else 0.98 * S.ema + 0.02 * loss
        if S.updates % 500 == 0 and buf.nv > 200:
            S.r2 = net.r2(*buf.val()); S.r2_hist.append(S.r2); S.r2_x.append(S.updates)


pp_base.recorder = pp_nn.recorder = Collector(mdl, sink, every=10, history=args.history)


def hud_text():
    w = 20
    nb, nn_ = np.mean(S.rms_base[-w:]) if S.rms_base else float("nan"), np.mean(S.rms_nn[-w:]) if S.rms_nn else float("nan")
    gain = f"{100 * (nn_ / nb - 1):+.0f}%" if S.rms_base and nb > 0 else "-"
    t1 = ["Controller now", "Task", "Samples (train / held-out)", "Gradient steps", "Training loss",
          "Model error explained (R2)", "R2 history", f"Tracking error, avg of last {w} tasks", "  baseline CT+I", "  CT+I + network", "  network vs baseline"]
    t2 = [S.run, f"{S.task}" + (" (held out)" if S.val else ""), f"{buf.n} / {buf.nv}", f"{S.updates}",
          "-" if np.isnan(S.ema) else f"{S.ema:.3f}", "-" if np.isnan(S.r2) else f"{S.r2:.2f}", "[" + spark(S.r2_hist) + "]", "",
          f"{1000 * nb:.1f} mrad" if S.rms_base else "-", f"{1000 * nn_:.1f} mrad" if S.rms_nn else "-", gain]
    return "\n".join(t1), "\n".join(t2)


def make_plot(live):
    try:
        import matplotlib
        if not live:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    if live and "agg" in matplotlib.get_backend().lower():
        print("(no interactive matplotlib window available here; saving training_curve.png instead)"); live = False
    if live:
        plt.ion()
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))

    def draw():
        for a in ax: a.clear()
        ax[0].plot(S.r2_x, S.r2_hist); ax[0].set_ylim(-0.1, 1.0)
        ax[0].set_xlabel("gradient steps"); ax[0].set_ylabel("R2 on held-out tasks"); ax[0].set_title("share of model error explained")
        k = np.arange(1, len(S.rms_base) + 1)
        sm = lambda v: np.convolve(v, np.ones(5) / 5, mode="valid") if len(v) >= 5 else np.array(v)
        ax[1].plot(k[:len(sm(S.rms_base))] + (4 if len(k) >= 5 else 0), 1000 * sm(S.rms_base), label="baseline CT+I")
        ax[1].plot(k[:len(sm(S.rms_nn))] + (4 if len(k) >= 5 else 0), 1000 * sm(S.rms_nn), label="CT+I + network")
        ax[1].set_xlabel("task"); ax[1].set_ylabel("rms joint error (mrad), 5-task average"); ax[1].legend(); ax[1].set_title("tracking, same conditions")
        fig.tight_layout()
        if live:
            fig.canvas.draw_idle(); fig.canvas.flush_events()
        else:
            fig.savefig("training_curve.png", dpi=110)
    return draw


draw = make_plot(args.plot) if (args.plot or args.headless or True) else None
if args.plot is False and draw is not None and not args.headless:
    pass


def finish():
    net.save(args.out)
    if draw: draw()
    with open("training_log.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["task", "rms_baseline_rad", "rms_network_rad"]); w.writerows(zip(range(1, len(S.rms_base) + 1), S.rms_base, S.rms_nn))
    k = min(20, len(S.rms_base))
    if k:
        print(f"\nlast {k} tasks: baseline {1000 * np.mean(S.rms_base[-k:]):.2f} mrad, with network {1000 * np.mean(S.rms_nn[-k:]):.2f} mrad | "
              f"held-out R2 {S.r2:.2f} | {S.updates} gradient steps | {time.time() - S.t0:.0f} s")
    print(f"saved {args.out}, training_log.csv" + (", training_curve.png" if draw else ""))


def run_task(i, pacing=None, frames_for=None):
    S.task, S.val = i, (i % 10 == 9)
    par = rd.sample(rng, NOMINAL if rng.random() < args.p_nominal else RANGES); rd.apply(par); obj = sample_object(rng); cube, yaw, goal = sample_task(rng)
    if args.noise_train:                                # same noise level and seed for the baseline run and the network run
        sq, sv = NOISE_LEVELS[rng.integers(len(NOISE_LEVELS))]
        for pp_ in (pp_base, pp_nn):
            pp_.noise, pp_._noise_rng = ((sq, sv, 10_000 + i) if sq > 0 else None), None
    out = {}
    for label, pp in (("baseline CT+I", pp_base), ("CT+I + network", pp_nn)):
        S.run = label
        if pacing: pacing()
        fr = [] if frames_for is not None else None
        info = pp.episode(cube, yaw, goal, frames=fr, obj=obj, payload=par["payload"])
        out[label] = info
        if frames_for is not None: frames_for(label, fr)
    b, n_ = out["baseline CT+I"], out["CT+I + network"]
    if not (b["planned"] and n_["planned"]):
        print(f"task {i:3d}: skipped (the planner found no solution for this random task)", flush=True)
        return out
    S.rms_base.append(b["rms_track"]); S.rms_nn.append(n_["rms_track"])
    if buf.nv > 200: S.r2 = net.r2(*buf.val()); S.r2_hist.append(S.r2); S.r2_x.append(S.updates)
    size = (f"{2000 * obj['half'][0]:.0f}x{2000 * obj['half'][1]:.0f}x{1000 * obj['h']:.0f}mm" if obj["shape"] == "box"
            else f"d{2000 * obj['half']:.0f}xh{1000 * obj['h']:.0f}mm")
    print(f"task {i:3d}{' [val]' if S.val else '      '} {obj['shape']} {size:>13s} {obj['mass']:.2f}kg | payload {par['payload']:.2f}kg "
          f"mass x{par['mass_scale']:.2f} damp x{par['damping_scale']:.2f} | baseline {1000 * b['rms_track']:5.1f} mrad {1000 * b['err_xy']:4.1f}mm"
          f" | +network {1000 * n_['rms_track']:5.1f} mrad {1000 * n_['err_xy']:4.1f}mm | R2 {S.r2:5.2f} | {buf.n} samples", flush=True)
    return out


if args.headless:
    renderer = None
    if args.video:
        import imageio.v2 as imageio
        from PIL import Image, ImageDraw
        renderer = mujoco.Renderer(rd.plant, 480, 640); cam = mujoco.MjvCamera()
        cam.lookat[:] = [0.2, 0, 0.1]; cam.distance = 1.2; cam.azimuth = 150; cam.elevation = -25
        writer = imageio.get_writer(args.video, fps=30)
        for pp in (pp_base, pp_nn):
            pp._renderer, pp._cam, pp._every = renderer, cam, int(1 / 30 / rd.plant.opt.timestep)

        def annotate(label, frames):
            t1, t2 = hud_text()
            for f in frames:
                im = Image.fromarray(f); dr = ImageDraw.Draw(im)
                dr.text((8, 6), f"{label}   |   task {S.task}   |   R2 {S.r2:.2f}   |   steps {S.updates}", fill=(255, 255, 255))
                writer.append_data(np.asarray(im))
    try:
        for i in range(n_tasks):
            record = args.video and i >= n_tasks - args.video_last
            run_task(i, frames_for=annotate if record else None)
            if draw and (i + 1) % 10 == 0: draw()
    except KeyboardInterrupt:
        pass
    if args.video: writer.close(); print("wrote", args.video)
    finish()
else:
    import mujoco.viewer as mjv
    warned = [False]
    with mjv.launch_passive(rd.plant, d) as v:
        v.cam.lookat[:] = [0.2, 0, 0.1]; v.cam.distance = 1.3; v.cam.azimuth = 150; v.cam.elevation = -25
        wall0 = [time.time()]; nstep = [0]; last_hud = [0.0]

        def show():
            try:
                t1, t2 = hud_text()
                v.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT, t1, t2))
            except Exception as e:
                if not warned[0]: print("(overlay text not available:", e, ")"); warned[0] = True

        def hook():
            nstep[0] += 1
            if nstep[0] % 8 == 0:
                if not v.is_running(): raise SystemExit
                v.sync()
                if time.time() - last_hud[0] > 0.4: show(); last_hud[0] = time.time()
                lag = d.time / args.speed - (time.time() - wall0[0])        # stay at the chosen speed when the simulation is faster
                if lag > 0: time.sleep(min(lag, 0.05))
        pp_base.hook = pp_nn.hook = hook
        print("Close the MuJoCo window or press Ctrl+C to stop; the network is saved when you do.")
        try:
            for i in range(n_tasks):
                run_task(i, pacing=lambda: wall0.__setitem__(0, time.time()))
                if draw: draw()
                if not v.is_running(): break
        except (SystemExit, KeyboardInterrupt):
            pass
    finish()