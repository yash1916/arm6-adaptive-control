"""Pick-and-place sequence for the 6-axis arm.

Plan  : IK turns Cartesian waypoints (tool pointing down) into joint paths; each path is time-scaled with a smooth
        quintic profile.  Any joint-space controller (PD, PID, PD+gravity, computed torque) then tracks the path.
Steps : move above cube -> descend -> close gripper -> lift -> carry above goal -> descend -> open -> retreat
"""
import copy, types
import numpy as np
import mujoco
from scipy.interpolate import CubicSpline
from common import quintic
from ik import IK, tool_down_R
from controllers import NominalModel
from objects import DEFAULT_OBJ, apply_object, grasp_yaw, grasp_height

Z_SAFE = 0.14            # tool height while moving (m)
Z_GRASP = 0.012          # tool height when grasping / placing (cube centre is at 0.015)
CUBE_HALF = 0.015
GRIP_CLOSE = 0.025       # finger servo target (m) when closing; cube blocks the fingers at about 0.013
V_FAST, V_SLOW = 0.20, 0.08     # PEAK tool speeds (m/s) of the quintic profile
TORQUE_USE, MAX_T, HEADROOM, FLOOR_MAX = 0.7, 25.0, 0.15, 0.85   # planner keeps the nominal feed-forward torque below 70 % of each motor's limit
R_MIN, R_MAX, TH_MAX = 0.30, 0.50, np.pi   # workspace for objects and goals: radius from base axis, +/- angle (rad): full 360 deg ring


def sample_task(rng, min_sep=0.12):
    """Random cube (x, y, yaw) and goal (x, y), both inside the reachable ring sector, at least min_sep apart."""
    def pt():
        r, th = rng.uniform(R_MIN, R_MAX), rng.uniform(-TH_MAX, TH_MAX)
        return np.array([r * np.cos(th), r * np.sin(th)])
    cube = pt()
    while True:
        goal = pt()
        if np.linalg.norm(goal - cube) >= min_sep:
            break
    return cube, rng.uniform(-np.pi / 4, np.pi / 4), goal


class Sensed:
    """What the controller is allowed to read: the simulation state, except that joint angles and velocities are noisy."""
    def __init__(self, d, qpos, qvel):
        self._d, self.qpos, self.qvel = d, qpos, qvel

    def __getattr__(self, name):
        return getattr(self._d, name)


class PickPlace:
    def __init__(self, m, d, controller, nominal=None):
        self.m, self.d, self.c = m, d, controller
        self.ik = IK(m)
        self.tau_max = m.actuator_ctrlrange[:6, 1]
        self.cube_adr = m.jnt_qposadr[m.joint("cube_free").id]
        self.tcp_id = m.site("tcp").id
        self.q_ready = self._find_ready()
        self.grip = 0.0
        self.min_link_z = 9.0
        self.hook = None                 # optional callable run after every simulation step (viewer / pacing)
        self.recorder = None             # optional callable(pp) run just before every simulation step (data collection)
        self._trace = None               # (time, tool-tip error) samples while measuring settling time
        self._tcp_target, self._path_end = None, 0.0
        self.delay_steps, self._delay_q = 0, None      # control latency: the torque computed now is applied this many steps later
        self.noise = None                # None, or (sigma_q rad, sigma_qd rad/s, seed): Gaussian noise on the joint readings the controller sees
        self._noise_rng = None
        # Torque-aware timing: the planner checks, with its OWN nominal model (a copy taken now), that the reference motion
        # needs at most TORQUE_USE of each motor's limit, and stretches the duration if it does not.
        self.torque_aware = True
        self._nom = NominalModel(nominal if nominal is not None else copy.copy(m))
        self._nom_d = types.SimpleNamespace(qpos=np.zeros(m.nq), qvel=np.zeros(m.nv))

    # ---------------------------------------------------------------- planning
    def _find_ready(self):
        rng = np.random.default_rng(1)
        for _ in range(100):
            seed = np.array([0, rng.uniform(0.2, 1.0), rng.uniform(0.5, 1.8), 0, rng.uniform(-1.5, 0), 0])
            q, ok = self.ik.solve(np.array([0.40, 0.0, Z_SAFE]), tool_down_R(0.0), seed)
            if ok:
                return q
        raise RuntimeError("could not find a ready pose")

    def cart_path(self, p0, yaw0, p1, yaw1, q_start, spacing=0.02):
        """Path between two tool poses in POLAR form: radius, angle about the base axis, height and yaw all change linearly.
        Vertical moves are straight lines; moves around the base are arcs (never through the base), up to a full turn.
        Returns the joint path (K x 6) by chained IK, or None if IK fails."""
        r0, r1 = np.hypot(*p0[:2]), np.hypot(*p1[:2])
        t0, t1 = np.arctan2(p0[1], p0[0]), np.arctan2(p1[1], p1[0])
        length = np.sqrt((r1 - r0) ** 2 + (0.5 * (r0 + r1) * (t1 - t0)) ** 2 + (p1[2] - p0[2]) ** 2)
        n = max(5, int(np.ceil(length / spacing)) + 1)
        Q, q = [], q_start
        for s in np.linspace(0, 1, n):
            r, th = (1 - s) * r0 + s * r1, (1 - s) * t0 + s * t1
            q, ok = self.ik.solve(np.array([r * np.cos(th), r * np.sin(th), (1 - s) * p0[2] + s * p1[2]]),
                                  tool_down_R((1 - s) * yaw0 + s * yaw1), q)
            if not ok:
                return None
            Q.append(q.copy())
        return np.array(Q)

    # ---------------------------------------------------------------- execution
    def _set_preview(self, fn):
        """Give look-ahead controllers (MPC) a function ahead_seconds -> (q, qd, qdd) of the reference."""
        if getattr(self.c, "uses_preview", False):
            self.c.preview = fn

    def _step(self, q_ref, qd_ref, qdd_ref):
        m, d = self.m, self.d
        mujoco.mj_forward(m, d)
        if self.noise is None:
            seen = d
        else:
            if self._noise_rng is None:
                self._noise_rng = np.random.default_rng(self.noise[2])
            q, qd = d.qpos.copy(), d.qvel.copy()
            q[:6] += self.noise[0] * self._noise_rng.standard_normal(6)
            qd[:6] += self.noise[1] * self._noise_rng.standard_normal(6)
            seen = Sensed(d, q, qd)
        tau = self.c(m, seen, q_ref, qd_ref, qdd_ref)
        self._n += 1
        e2 = float(np.sum((q_ref - d.qpos[:6]) ** 2))
        self._e2 += e2
        if self._shifted:
            self._e2p += e2; self._np += 1
        self._sat += int(np.any(np.abs(tau) > self.tau_max))
        tau_c = np.clip(tau, -self.tau_max, self.tau_max)
        u = tau_c / self.tau_max                                   # torque as a fraction of each motor's limit
        self._eff += float(np.mean(u ** 2))
        if self._tau_prev is not None:
            self._chat += float(np.mean((u - self._tau_prev) ** 2))
        self._tau_prev = u
        if self._trace is not None:
            self._trace.append((d.time, float(np.linalg.norm(d.site_xpos[self.tcp_id] - self._tcp_target))))
        if self.delay_steps > 0:
            if self._delay_q is None:
                self._delay_q = [tau_c.copy() for _ in range(self.delay_steps)]
            self._delay_q.append(tau_c.copy())
            tau_c = self._delay_q.pop(0)
        d.ctrl[:6] = tau_c
        d.ctrl[6:8] = self.grip
        if self.recorder is not None:
            self.recorder(self)
        mujoco.mj_step(m, d)
        if self._frames is not None:
            self._record()
        if self.hook is not None:
            self.hook()
        self.min_link_z = min(self.min_link_z, min(d.xpos[m.body(b).id][2] for b in ("Link3Forearm", "Link4ForearmRoll", "Link5Wrist")))

    def _torque_use(self, spl, T, n=60):
        """Largest ratio |tau_ff| / tau_max over the quintic reference of duration T, tau_ff = M qdd + h from the nominal model."""
        worst = 0.0
        for t in np.linspace(0.0, T, n):
            s_, sd, sdd = quintic(0.0, 1.0, T, t)
            q, qd, qdd = spl(s_), spl(s_, 1) * sd, spl(s_, 2) * sd**2 + spl(s_, 1) * sdd
            self._nom_d.qpos[:6], self._nom_d.qvel[:6] = q, qd
            M, h = self._nom.dynamics(self._nom_d)
            worst = max(worst, float(np.max(np.abs(M @ qdd + h) / self.tau_max)))
        return worst

    def _static_use(self, spl, n=60):
        """Largest |gravity torque| / tau_max along the path (arm at rest): the floor no amount of slowing down can go below."""
        worst = 0.0
        for s_ in np.linspace(0.0, 1.0, n):
            self._nom_d.qpos[:6], self._nom_d.qvel[:6] = spl(s_), 0.0
            _, h = self._nom.dynamics(self._nom_d)
            worst = max(worst, float(np.max(np.abs(h) / self.tau_max)))
        return worst

    def _plan_duration(self, spl, T):
        """Stretch T (x1.12 per step, at most 25 s) until the nominal feed-forward torque leaves room for feedback:
        target = max(TORQUE_USE, gravity floor + HEADROOM).  If gravity alone needs more than FLOOR_MAX of a motor
        (the arm cannot even hold this pose), slowing down cannot help, so the duration is left alone and the path is
        flagged in self.last_infeasible."""
        self.last_stretch, self.last_infeasible = 1.0, False
        if not self.torque_aware:
            return T
        floor = self._static_use(spl)
        if floor > FLOOR_MAX:
            self.last_infeasible = True
            return T
        target, T0 = max(TORQUE_USE, floor + HEADROOM), T
        while T < MAX_T and self._torque_use(spl, T) > target:
            T *= 1.12
        self.last_stretch = T / T0
        return T

    def run_path(self, Q, speed, trace=False):
        """Execute a joint path Q (K x 6) with a quintic time profile; peak tool speed = speed.
        trace=True also records the tool-tip error to the path's end point, for settling_time() after the next hold."""
        d = self.d
        L = np.linalg.norm(np.diff(self.ik.tcp_path(Q), axis=0), axis=1).sum() if len(Q) > 1 else 0.0
        T = max(0.6, 1.875 * L / speed)
        spl = CubicSpline(np.linspace(0, 1, len(Q)), Q)
        T = self._plan_duration(spl, T)
        t0 = d.time
        if trace:
            self._trace, self._tcp_target, self._path_end = [], self.ik.tcp(Q[-1]), t0 + T

        def preview(ahead):                            # reference `ahead` seconds from now (held after the path ends)
            s, sd, sdd = quintic(0.0, 1.0, T, min(d.time - t0 + ahead, T))
            if d.time - t0 + ahead >= T:
                sd = sdd = 0.0
            return spl(s), spl(s, 1) * sd, spl(s, 2) * sd**2 + spl(s, 1) * sdd

        self._set_preview(preview)
        while d.time - t0 < T + 0.2:                  # +0.2 s settling at the end of the path
            t = d.time - t0
            s, sd, sdd = quintic(0.0, 1.0, T, min(t, T))
            if t >= T:
                sd = sdd = 0.0
            self._step(spl(s), spl(s, 1) * sd, spl(s, 2) * sd**2 + spl(s, 1) * sdd)
        self.q_hold = Q[-1]

    def settling_time(self, thresh=0.001):
        """Time from the end of the traced path until the tool tip stays within `thresh` (1 mm) of the target.
        Returns (seconds, settled). If it never settles within the recorded window, seconds = window length, settled = False."""
        tr = np.array(self._trace); self._trace = None
        t, e = tr[:, 0], tr[:, 1]
        keep = t >= self._path_end
        t, e = t[keep], e[keep]
        if len(t) == 0:
            return float("nan"), False
        bad = np.where(e >= thresh)[0]
        if len(bad) == 0:
            return 0.0, True
        if bad[-1] == len(t) - 1:
            return float(t[-1] - self._path_end), False
        return float(t[bad[-1] + 1] - self._path_end), True

    def hold(self, seconds, grip):
        self.grip = grip
        self._set_preview(lambda ahead: (self.q_hold, np.zeros(6), np.zeros(6)))
        t0 = self.d.time
        while self.d.time - t0 < seconds:
            self._step(self.q_hold, np.zeros(6), np.zeros(6))

    def set_payload(self, kg):
        """Unknown payload on the tool tip: raises the mass of the 'payload' body in the PLANT model only."""
        m, d = self.m, self.d
        bid = m.body("payload").id
        m.body_mass[bid] = max(kg, 1e-6)
        m.body_inertia[bid] = [max(kg, 1e-6) * 1e-4] * 3
        mujoco.mj_setConst(m, mujoco.MjData(m))          # scratch data: safe to call in the middle of an episode

    def _maybe_shift(self, shift, phase):
        """Change the plant in the middle of an episode, unknown to the controller: shift = dict(at='lift'|'carry', payload=kg, damping=factor)."""
        if shift is None or shift.get("at") != phase:
            return
        if "payload" in shift:
            self.set_payload(shift["payload"])
        if "damping" in shift:
            self.m.dof_damping[:6] *= shift["damping"]
        if "damping_vec" in shift:                                   # a different factor on every joint
            self.m.dof_damping[:6] *= shift["damping_vec"]
        if "mass_vec" in shift:                                      # a different mass factor on every link (inertia scaled the same)
            from plant import ARM_BODIES
            for b, f in zip(ARM_BODIES, shift["mass_vec"]):
                i = self.m.body(b).id
                self.m.body_mass[i] *= f
                self.m.body_inertia[i] *= f
            mujoco.mj_setConst(self.m, mujoco.MjData(self.m))
        if "friction_vec" in shift:                                  # Coulomb friction (N m) on every joint
            self.m.dof_frictionloss[:6] = shift["friction_vec"]
        self._shifted = True

    def episode(self, cube_xy, cube_yaw, goal_xy, frames=None, cam_fn=None, payload=None, obj=None, shift=None):
        """shift: optional dict that changes the arm mid-episode (see _maybe_shift).  obj: object spec from objects.py (default: the 30 mm, 0.1 kg cube). payload: None keeps the plant as it is."""
        m, d = self.m, self.d
        obj = DEFAULT_OBJ if obj is None else obj
        apply_object(m, d, obj)
        if payload is not None:
            self.set_payload(payload)
        zg = grasp_height(obj)
        mujoco.mj_resetData(m, d)
        d.qpos[:6] = self.q_ready
        a = self.cube_adr
        d.qpos[a:a + 3] = [cube_xy[0], cube_xy[1], obj["h"] / 2]
        d.qpos[a + 3:a + 7] = [np.cos(cube_yaw / 2), 0, 0, np.sin(cube_yaw / 2)]
        d.mocap_pos[0] = [goal_xy[0], goal_xy[1], 0.001]
        self.grip = 0.0; self.min_link_z = 9.0; self._frames = frames; self._cam_fn = cam_fn
        self._n, self._e2, self._sat = 0, 0.0, 0
        self._shifted, self._e2p, self._np = False, 0.0, 0
        self._eff, self._chat, self._tau_prev, self._trace = 0.0, 0.0, None, None
        settle = []
        self.c.reset()
        mujoco.mj_forward(m, d)
        self.q_hold = self.q_ready.copy()
        info = dict(planned=False, lifted=False, success=False, err_xy=np.nan, reason="")

        p_ready = self.ik.tcp(self.q_ready)
        above = lambda xy, z: np.array([xy[0], xy[1], z])
        # ---- plan every motion first (so an unreachable task is reported before anything moves)
        phi_c = float(np.arctan2(cube_xy[1], cube_xy[0]))      # the cube looks the same after a quarter turn, so choose the
        yaw = grasp_yaw(obj, cube_yaw, phi_c)         # fingers close across the object's short side; wrist roll within +/-90 deg (see objects.grasp_yaw)
        gyaw = float(np.arctan2(goal_xy[1], goal_xy[0])) + np.pi / 2   # yaw for placing: position only matters, so use the wrist-neutral yaw
                                                           # (the object turns while carried)
        plan, q = [], self.q_ready
        legs = [("fast", p_ready, 0.0, above(cube_xy, Z_SAFE), yaw),
                ("slow", above(cube_xy, Z_SAFE), yaw, above(cube_xy, zg), yaw),
                ("slow", above(cube_xy, zg), yaw, above(cube_xy, Z_SAFE), yaw),
                ("fast", above(cube_xy, Z_SAFE), yaw, above(goal_xy, Z_SAFE), gyaw),
                ("slow", above(goal_xy, Z_SAFE), gyaw, above(goal_xy, zg + 0.004), gyaw),
                ("slow", above(goal_xy, zg + 0.004), gyaw, above(goal_xy, Z_SAFE), gyaw)]
        for speed, p0, y0, p1, y1 in legs:
            Q = self.cart_path(p0, y0, p1, y1, q)
            if Q is None:
                info["reason"] = "IK failed"; return info
            plan.append((speed, Q)); q = Q[-1]
        info["planned"] = True

        # ---- execute
        (s0, Q0), (s1, Q1), (s2, Q2), (s3, Q3), (s4, Q4), (s5, Q5) = plan
        self.hold(0.4, 0.0)
        self.run_path(Q0, V_FAST if s0 == "fast" else V_SLOW)
        self.run_path(Q1, V_SLOW, trace=True)         # descend
        self.hold(0.8, GRIP_CLOSE)                    # close
        settle.append(self.settling_time())
        self._maybe_shift(shift, "lift")
        self.run_path(Q2, V_SLOW)                     # lift
        info["lifted"] = bool(d.qpos[a + 2] > 0.05)
        self._maybe_shift(shift, "carry")
        self.run_path(Q3, V_FAST)                     # carry
        self.run_path(Q4, V_SLOW, trace=True)         # lower
        self.hold(0.6, 0.0)                           # release
        settle.append(self.settling_time())
        self.run_path(Q5, V_SLOW)                     # retreat
        self.hold(0.5, 0.0)

        cube = d.qpos[a:a + 3].copy()
        info["err_xy"] = float(np.linalg.norm(cube[:2] - goal_xy))
        info["cube_z"] = float(cube[2])
        info["success"] = bool(info["err_xy"] < 0.02 and cube[2] < 0.5 * obj["h"] + 0.01)
        info["reason"] = "ok" if info["success"] else ("dropped / not lifted" if not info["lifted"] else "missed goal")
        info["min_link_z"] = float(self.min_link_z)
        info["obj"] = obj
        info["rms_track"] = float(np.sqrt(self._e2 / max(self._n, 1)))     # rms joint-error norm over the episode (rad)
        info["settle_s"] = float(np.mean([s for s, _ in settle]))          # mean over the two precision moves (grasp, place)
        info["unsettled"] = int(sum(not ok for _, ok in settle))           # how many of them never settled within 1 mm
        info["post_shift_rms"] = float(np.sqrt(self._e2p / self._np)) if self._np else float("nan")   # rms error after the change
        info["effort"] = float(np.sqrt(self._eff / max(self._n, 1)))        # rms torque / torque limit, 0..1
        info["chatter"] = float(np.sqrt(self._chat / max(self._n - 1, 1)))  # rms step-to-step torque change / limit
        info["sat_frac"] = self._sat / max(self._n, 1)                     # fraction of steps with a torque clipped
        return info

    # ---------------------------------------------------------------- video support
    _frames = None

    def _record(self):
        d = self.d
        if int(round(d.time / self.m.opt.timestep)) % self._every == 0:
            self._renderer.update_scene(d, self._cam); self._frames.append(self._renderer.render())