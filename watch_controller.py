"""Watch a controller drive the arm through a sequence of poses.

  python3 watch_controller.py ct_full                  interactive MuJoCo window (controller: pd | pdg | ct | ct_full)
  python3 watch_controller.py pd 1.5                   plain PD, 1.5 s per move
  python3 watch_controller.py ct_full 1.0 --video ct.mp4   no window: record 20 s of video instead
  python3 watch_controller.py ct_full 1.0 --headless 12    no window, no video: just run and print tracking errors
"""
import os, time, argparse
import numpy as np
import mujoco
from common import load, quintic
from controllers import PD, PDGravity, ComputedTorque

POSES = [np.zeros(6),
         np.array([0.9, 0.5, 0.9, 0.0, -0.6, 0.0]),
         np.array([-0.9, 0.9, -0.4, 1.0, 0.5, 1.5]),
         np.array([0.0, -0.6, 1.2, -0.8, -0.8, -1.5]),
         np.array([0.5, -0.4, 0.8, 0.3, -0.5, 0.4])]
HOLD = 0.6                                   # seconds to pause at each pose
KP = np.array([300, 500, 300, 40, 30, 15.]); KD = np.array([40, 60, 40, 4, 3, 1.5])


def make_controller(name, m):
    return {"pd": lambda: PD(KP, KD),
            "pdg": lambda: PDGravity(m, KP, KD),
            "ct": lambda: ComputedTorque(wn=20),
            "ct_full": lambda: ComputedTorque(wn=20, model_damping=True)}[name]()


class Runner:
    """Steps the simulation and walks through POSES forever (pose 0 -> 1 -> 2 -> ... -> 0 ...)."""
    def __init__(self, m, d, controller, t_move):
        self.m, self.d, self.c, self.T = m, d, controller, t_move
        self.i = 0                                   # index of the pose we are leaving
        self.t0 = 0.0                                # simulation time when the current move began
        d.qpos[:] = POSES[0]; mujoco.mj_forward(m, d)
        self.tau_max = m.actuator_ctrlrange[:, 1]

    def step(self):
        m, d = self.m, self.d
        t_local = d.time - self.t0
        if t_local >= self.T + HOLD:                 # move + pause finished -> next segment
            self.i = (self.i + 1) % len(POSES); self.t0 = d.time; t_local = 0.0
            print(f"[t={d.time:6.2f}s] moving to pose {(self.i + 1) % len(POSES)}")
        q_from, q_to = POSES[self.i], POSES[(self.i + 1) % len(POSES)]
        q_ref, qd_ref, qdd_ref = quintic(q_from, q_to, self.T, t_local)
        mujoco.mj_forward(m, d)
        tau = self.c(m, d, q_ref, qd_ref, qdd_ref)
        d.ctrl[:] = np.clip(tau, -self.tau_max, self.tau_max)
        mujoco.mj_step(m, d)
        return q_ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("controller", choices=["pd", "pdg", "ct", "ct_full"])
    ap.add_argument("move_time", nargs="?", type=float, default=2.0)
    ap.add_argument("--video"); ap.add_argument("--headless", type=float, metavar="SECONDS")
    a = ap.parse_args()
    m, d = load(); run = Runner(m, d, make_controller(a.controller, m), a.move_time)
    print(f"controller={a.controller}  move_time={a.move_time}s")

    if a.headless:                                   # run without any display and report the tracking error
        errs = []
        while d.time < a.headless:
            q_ref = run.step(); errs.append(np.abs(q_ref - d.qpos).max())
        print(f"max joint error {max(errs):.4f} rad | mean {np.mean(errs):.4f} rad"); return

    if a.video:                                      # record to a file (needs imageio-ffmpeg)
        os.environ.setdefault("MUJOCO_GL", "egl")
        import imageio.v3 as iio
        r = mujoco.Renderer(m, 480, 640); cam = mujoco.MjvCamera()
        cam.lookat[:] = [0.2, -0.04, 0.3]; cam.distance = 1.3; cam.azimuth = 135; cam.elevation = -15
        frames, per = [], int(1 / 30 / m.opt.timestep)
        while d.time < 20.0:
            for _ in range(per): run.step()
            r.update_scene(d, cam); frames.append(r.render())
        iio.imwrite(a.video, np.stack(frames), fps=30); print("wrote", a.video); return

    import mujoco.viewer as mjv                       # interactive window, real-time
    with mjv.launch_passive(m, d) as v:
        v.cam.lookat[:] = [0.2, -0.04, 0.3]; v.cam.distance = 1.3; v.cam.azimuth = 135; v.cam.elevation = -15
        wall0 = time.time(); n = 0
        while v.is_running():
            run.step(); n += 1
            if n % 8 == 0:                            # redraw about every 16 ms of simulated time
                v.sync()
                lag = d.time - (time.time() - wall0)  # stay in real time
                if lag > 0: time.sleep(lag)


if __name__ == "__main__":
    main()