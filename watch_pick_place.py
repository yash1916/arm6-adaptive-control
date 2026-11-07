"""Watch pick-and-place in MuJoCo and choose the controller (and task) while it runs.

  python3 watch_pick_place.py                      asks you which controller to start with
  python3 watch_pick_place.py mpc                  start with MPC (pd | pid | pdg | ct | cti | mpc | all)
  python3 watch_pick_place.py pid --cond mismatch  plant differs from the controller's model (nominal | mismatch | weak)
  python3 watch_pick_place.py all --video all.mp4 --n 2        no window: record a video, name drawn on every frame
  python3 watch_pick_place.py pid --video t.mp4 --cmds "mpc; task 0.4 0.1 0.35 -0.15; n"    scripted commands

While the window is open, type commands in THIS terminal and press Enter:
  pd | pid | pdg | ct | cti | mpc   switch controller; the SAME task is replayed so you can compare
  all                               every controller does the current task in turn
  n                                 new random task            r   replay the current task
  task cx cy gx gy [yaw]            your own task: cube (cx, cy) -> goal (gx, gy) in metres, optional cube yaw in rad
  help                              list commands             q   quit
Without a command the next random task starts by itself.  Reachable area: a full 360 deg ring, radius 0.30-0.50 m from the base.
"""
import os, sys, time, threading, queue
if "--video" in sys.argv:
    os.environ.setdefault("MUJOCO_GL", "egl")        # headless rendering; must be set before mujoco is imported
import numpy as np
import mujoco
import pick_place
from pick_place import PickPlace, sample_task
from plant import make_plant
from controllers import PD, PID, PDGravity, ComputedTorque, NominalModel, MPC
from run_model_based import CONDITIONS, SPEED, KP, KD, KI, I_LIM

LABEL = {"pd": "PD", "pid": "PID", "pdg": "PD + gravity compensation", "ct": "Computed torque",
         "cti": "Computed torque + integral", "mpc": "Model predictive control (MPC)"}
HELP = "Commands (type + Enter):" + __doc__.split("press Enter:")[1].split("Without a command")[0]


def make(name, nominal):
    mdl = NominalModel(nominal)
    return {"pd": lambda: PD(KP, KD), "pid": lambda: PID(KP, KI, KD, I_LIM),
            "pdg": lambda: PDGravity(nominal, KP, KD),
            "ct": lambda: ComputedTorque(wn=20, model=mdl),
            "cti": lambda: ComputedTorque(wn=20, model=mdl, ki=2000.0),
            "mpc": lambda: MPC(mdl)}[name]()


args = sys.argv[1:]
opt = lambda k, default: (args[args.index(k) + 1] if k in args else default)
first = args[0] if args and not args[0].startswith("--") else None
cond = opt("--cond", "nominal")
seed, n_ep, video, cmds = int(opt("--seed", 7)), int(opt("--n", 2)), opt("--video", None), opt("--cmds", None)
if first is None:                                                  # no controller given: ask
    print("Controllers:", ", ".join(f"{k} = {v}" for k, v in LABEL.items()), ", all")
    first = ""
    while first not in list(LABEL) + ["all"]:
        first = input("Start with which controller? ").strip().lower()
pick_place.V_FAST *= SPEED[cond]; pick_place.V_SLOW *= SPEED[cond]

plant, nominal = make_plant(**CONDITIONS[cond])
d = mujoco.MjData(plant)
pps = {n: PickPlace(plant, d, make(n, nominal), nominal=nominal) for n in LABEL}   # one PickPlace per controller, one shared simulation
rng = np.random.default_rng(seed)
LOOKAT, DIST, AZ, EL = [0.30, 0.0, 0.10], 1.25, 150, -25


class Session:
    """Holds the current mode (one controller or 'all') and task; handle() applies one typed command."""
    def __init__(self):
        self.mode = first
        self.task = sample_task(rng)
        self.run_now = True              # run the current task+mode on the next loop pass
        self.quit = False

    def names(self):
        return list(LABEL) if self.mode == "all" else [self.mode]

    def handle(self, line):
        w = line.strip().lower().split()
        if not w:
            return
        c = w[0]
        if c in LABEL or c == "all":
            self.mode, self.run_now = c, True
            print(f"> controller: {'all' if c == 'all' else LABEL[c]}  (replaying the current task)")
        elif c in ("n", "new"):
            self.task, self.run_now = sample_task(rng), True
        elif c in ("r", "replay"):
            self.run_now = True
        elif c == "task":
            try:
                v = [float(x) for x in w[1:]]
                assert len(v) in (4, 5)
                self.task = (np.array(v[:2]), v[4] if len(v) == 5 else 0.0, np.array(v[2:4])); self.run_now = True
            except Exception:
                print("> usage: task cx cy gx gy [yaw]      e.g.  task 0.40 0.10 0.35 -0.15")
        elif c in ("q", "quit", "exit"):
            self.quit = True
        elif c in ("help", "h", "?"):
            print(HELP)
        else:
            print("> unknown command - type 'help' for the list")


S = Session()


def title(n):
    return f"Controller: {LABEL[n]}   |   plant: {cond}"


def report(n, cube, goal, info):
    if not info["planned"]:
        print(f"[{LABEL[n]:30s}] cube {np.round(cube, 3)} -> goal {np.round(goal, 3)} | NOT REACHABLE ({info['reason']}) - "
              f"keep cube and goal 0.30-0.50 m from the base (any angle)", flush=True)
        return
    print(f"[{LABEL[n]:30s}] cube {np.round(cube, 3)} -> goal {np.round(goal, 3)} | {info['reason']} | "
          f"place error {info['err_xy'] * 1000:.1f} mm | rms track {info['rms_track'] * 1000:.1f} mrad", flush=True)


if video:
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw
    renderer = mujoco.Renderer(plant, 480, 640); cam = mujoco.MjvCamera()
    cam.lookat[:] = LOOKAT; cam.distance = DIST; cam.azimuth = AZ; cam.elevation = EL
    writer = imageio.get_writer(video, fps=30); nfr = 0           # frames are streamed to disk, not kept in memory
    script = [c for c in (cmds or "").split(";")]
    runs = 0
    while runs < (n_ep if not cmds else len(script) + 1) and not S.quit:
        if runs > 0 and cmds:
            S.handle(script[runs - 1])
        elif runs > 0:
            S.task = sample_task(rng)
        cube, yaw, goal = S.task
        for n in S.names():
            pp = pps[n]; pp._renderer, pp._cam = renderer, cam; pp._every = int(1 / 30 / plant.opt.timestep)
            mine = []
            info = pp.episode(cube, yaw, goal, frames=mine)
            for f in mine:                                        # draw the controller name on each frame
                im = Image.fromarray(f); ImageDraw.Draw(im).text((10, 10), title(n), fill=(255, 255, 255))
                writer.append_data(np.asarray(im)); nfr += 1
            report(n, cube, goal, info)
        runs += 1
    writer.close(); print("wrote", video, f"({nfr} frames)")
else:
    import mujoco.viewer as mjv
    inbox = queue.Queue()

    def reader():                                                  # typed commands arrive here while the sim runs
        while True:
            try:
                inbox.put(input())
            except EOFError:
                return
    threading.Thread(target=reader, daemon=True).start()
    print("\nType a command + Enter any time (help = list).  Current controller:", S.mode)

    with mjv.launch_passive(plant, d) as v:
        v.cam.lookat[:] = LOOKAT; v.cam.distance = DIST; v.cam.azimuth = AZ; v.cam.elevation = EL
        wall0 = [time.time()]; n_steps = [0]

        def show(n, extra=""):
            try:
                v.set_texts((mujoco.mjtFontScale.mjFONTSCALE_200, mujoco.mjtGridPos.mjGRID_TOPLEFT, title(n), extra))
            except Exception:
                pass                                              # old viewer without overlay text: the terminal still prints it

        def hook():
            n_steps[0] += 1
            if n_steps[0] % 8 == 0:
                if not v.is_running():
                    raise SystemExit
                v.sync()
                lag = d.time - (time.time() - wall0[0])           # stay in real time when the sim is faster than that
                if lag > 0:
                    time.sleep(lag)
        for pp in pps.values():
            pp.hook = hook
        try:
            while v.is_running() and not S.quit:
                if S.run_now:
                    S.run_now = False
                    cube, yaw, goal = S.task
                    for n in S.names():
                        show(n, "running...")
                        wall0[0] = time.time()
                        info = pps[n].episode(cube, yaw, goal)
                        show(n, ("not reachable" if not info["planned"] else f"{info['reason']}, error {info['err_xy'] * 1000:.1f} mm") + "   (type a controller name to switch)")
                        report(n, cube, goal, info)
                        time.sleep(0.8)
                t_end = time.time() + 1.5                          # wait for a command, else start a new random task
                got = False
                while time.time() < t_end and v.is_running():
                    try:
                        S.handle(inbox.get(timeout=0.1)); got = True; break
                    except queue.Empty:
                        v.sync()
                if not got and not S.run_now:
                    S.task, S.run_now = sample_task(rng), True
        except SystemExit:
            pass