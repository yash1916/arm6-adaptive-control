"""Reinforcement-learning environment for the pick-and-place task (stage 4).

The policy acts at 50 Hz.  Its action is a small motion of the tool-tip target plus a gripper command:
    a = (dx, dy, dz, grip) in [-1, 1]^4      tool target moves by up to STEP_M * a[:3];  grip > 0 closes the gripper.
Below the policy sits the same machinery as the classical stages: damped-least-squares IK turns the tool target into a joint target,
a critically damped filter turns that into a smooth reference (q, qd, qdd), and the tuned computed-torque + integral controller
(60 rad/s) tracks it at 500 Hz.  So the policy decides WHERE to go and WHEN to grip; it does not produce motor torques.

Conveniences given to the policy, as in the scripted planner: the wrist yaw is chosen by the environment (grasp yaw across the
object's short side, wrist-neutral yaw for placing), so the policy never has to learn wrist rotation.

The class follows the Gymnasium API (reset -> obs, info; step -> obs, reward, terminated, truncated, info) without importing it,
so it runs anywhere NumPy and MuJoCo run.  Dynamics shifts and control delay are options, so training can randomise them.
"""
import numpy as np, mujoco
from ik import tool_down_R
from controllers import ComputedTorque, NominalModel
from objects import DEFAULT_OBJ, apply_object, grasp_yaw
from pick_place import sample_task, PickPlace, Z_SAFE, GRIP_CLOSE
from randomize import DynamicsRandomizer

NOMINAL = dict(mass_scale=1.0, damping_scale=1.0, payload=0.0)
CTRL_PER_ACT = 10          # simulation steps (2 ms) per policy action -> 50 Hz
STEP_M = 0.005             # largest tool-target move per action (m) -> 0.25 m/s
REF_WN = 30.0              # bandwidth of the reference filter (rad/s)
YAW_RATE = 1.5             # wrist yaw target rate limit (rad/s)
MAX_STEPS = 1200           # 24 s
R_LO, R_HI = 0.28, 0.55      # reachable ring for the tool target (m from the base axis)
OBS_DIM, ACT_DIM = 41, 4


class PickPlaceEnv:
    obs_dim, act_dim = OBS_DIM, ACT_DIM

    def __init__(self, seed=0, wn=60.0, delay_steps=0, shift_fn=None, noise=None):
        """shift_fn(rng) -> shift dict or None, applied when the cube is first lifted (same dict format as PickPlace._maybe_shift).
        noise = (sigma_q, sigma_qd) adds Gaussian noise to the joint readings the controller AND the policy see."""
        self.rng = np.random.default_rng(seed)
        self.rd = DynamicsRandomizer()
        self.mdl = NominalModel(self.rd.nominal)
        self.fl0 = self.rd.plant.dof_frictionloss.copy()
        self.wn, self.delay_steps, self.shift_fn, self.noise = wn, delay_steps, shift_fn, noise
        self.m = self.rd.plant
        self.d = mujoco.MjData(self.m)
        self.ctrl = ComputedTorque(wn=wn, model=self.mdl, ki=wn ** 3 / 4.0)
        # PickPlace is reused only for its IK, its helper constants and its shift code
        self.pp = PickPlace(self.m, self.d, self.ctrl, nominal=self.rd.nominal)
        self.ik, self.tau_max = self.pp.ik, self.pp.tau_max
        self.cube_adr = self.pp.cube_adr
        self.dt = self.m.opt.timestep

    # ------------------------------------------------------------------ episode control
    def reset(self, task=None):
        rd, m, d, pp = self.rd, self.m, self.d, self.pp
        rd.apply(NOMINAL); rd.plant.dof_frictionloss[:] = self.fl0
        cube, cyaw, goal = task if task is not None else sample_task(self.rng)
        self.cube0, self.goal = np.asarray(cube, float), np.asarray(goal, float)
        apply_object(m, d, DEFAULT_OBJ)
        mujoco.mj_resetData(m, d)
        d.qpos[:6] = pp.q_ready
        a = self.cube_adr
        d.qpos[a:a + 3] = [cube[0], cube[1], DEFAULT_OBJ["h"] / 2]
        d.qpos[a + 3:a + 7] = [np.cos(cyaw / 2), 0, 0, np.sin(cyaw / 2)]
        d.mocap_pos[0] = [goal[0], goal[1], 0.001]
        mujoco.mj_forward(m, d)
        self.ctrl.reset()
        phi_c = float(np.arctan2(cube[1], cube[0]))
        self.yaw_pick = grasp_yaw(DEFAULT_OBJ, cyaw, phi_c)
        self.yaw_place = float(np.arctan2(goal[1], goal[0])) + np.pi / 2
        self.yaw_cmd = 0.0
        self.q_goal = pp.q_ready.copy()
        self.q_ref, self.qd_ref = pp.q_ready.copy(), np.zeros(6)
        self.tool_t = self.ik.tcp(pp.q_ready)
        self.grip, self.grip_cmd = 0.0, 0.0
        self.shift = self.shift_fn(self.rng) if self.shift_fn else None
        self.shifted, self.lifted_once = False, False
        self.noise_rng = np.random.default_rng(int(self.rng.integers(1 << 31)))
        self.delay_q = None
        self.steps, self.prev_a, self.ik_fail = 0, np.zeros(4), 0
        self.sat, self.eff, self.n_sim = 0, 0.0, 0
        self.phi = self._potential()
        self.min_cube_z = 1.0
        return self._obs(), {}

    # ------------------------------------------------------------------ one policy action
    def step(self, action):
        m, d = self.m, self.d
        a = np.clip(np.asarray(action, float), -1, 1)
        # 1. tool target and gripper
        tgt = self.tool_t + STEP_M * a[:3]
        r = np.hypot(tgt[0], tgt[1])
        if r > 1e-6:                                   # keep the target inside the reachable ring and above the table
            rr = np.clip(r, R_LO, R_HI); tgt[:2] *= rr / r
        tgt[2] = np.clip(tgt[2], 0.010, 0.25)
        self.tool_t = tgt
        self.grip_cmd = GRIP_CLOSE if a[3] > 0 else 0.0
        cube = d.qpos[self.cube_adr:self.cube_adr + 3]
        carrying = self.grip_cmd > 0 and cube[2] > 0.03
        want = self.yaw_place if carrying else self.yaw_pick
        self.yaw_cmd += np.clip(want - self.yaw_cmd, -YAW_RATE * CTRL_PER_ACT * self.dt, YAW_RATE * CTRL_PER_ACT * self.dt)
        q, ok = self.ik.solve(self.tool_t, tool_down_R(self.yaw_cmd), self.q_goal)
        if ok:
            self.q_goal = q
        else:
            self.ik_fail += 1
        # 2. simulate CTRL_PER_ACT steps: filter the reference, run the controller
        for _ in range(CTRL_PER_ACT):
            qdd = REF_WN ** 2 * (self.q_goal - self.q_ref) - 2 * REF_WN * self.qd_ref
            self.qd_ref = self.qd_ref + self.dt * qdd
            self.q_ref = self.q_ref + self.dt * self.qd_ref
            mujoco.mj_forward(m, d)
            seen = self._sensed()
            tau = self.ctrl(m, seen, self.q_ref, self.qd_ref, qdd)
            self.sat += int(np.any(np.abs(tau) > self.tau_max))
            tau = np.clip(tau, -self.tau_max, self.tau_max)
            self.eff += float(np.mean((tau / self.tau_max) ** 2)); self.n_sim += 1
            if self.delay_steps > 0:
                if self.delay_q is None:
                    self.delay_q = [tau.copy() for _ in range(self.delay_steps)]
                self.delay_q.append(tau.copy()); tau = self.delay_q.pop(0)
            d.ctrl[:6] = tau
            d.ctrl[6:8] = self.grip_cmd
            mujoco.mj_step(m, d)
            if not self.shifted and self.shift is not None and d.qpos[self.cube_adr + 2] > 0.03 and self.grip_cmd > 0:
                self.pp._maybe_shift(dict(self.shift, at="lift"), "lift"); self.shifted = True
        self.steps += 1
        # 3. reward and termination
        cube = d.qpos[self.cube_adr:self.cube_adr + 3].copy()
        self.min_cube_z = min(self.min_cube_z, cube[2])
        if cube[2] > 0.05:
            self.lifted_once = True
        phi_new = self._potential()
        reward = phi_new - self.phi - 0.002 - 0.01 * float(np.sum((a - self.prev_a) ** 2)) / 4
        self.phi, self.prev_a = phi_new, a
        err_xy = float(np.linalg.norm(cube[:2] - self.goal))
        placed = err_xy < 0.02 and cube[2] < 0.5 * DEFAULT_OBJ["h"] + 0.01 and self.grip_cmd == 0.0 and self._cube_at_rest()
        fell = cube[2] < -0.05
        terminated = bool(placed or fell)
        if placed:
            reward += 10.0
        if fell:
            reward -= 5.0
        truncated = bool(self.steps >= MAX_STEPS and not terminated)
        info = dict(success=bool(placed), err_xy=err_xy, lifted=self.lifted_once, dropped=bool(self.lifted_once and not placed and cube[2] < 0.03 and err_xy > 0.02),
                    ik_fail=self.ik_fail, effort=float(np.sqrt(self.eff / max(self.n_sim, 1))), sat_frac=self.sat / max(self.n_sim, 1))
        return self._obs(), float(reward), terminated, truncated, info

    # ------------------------------------------------------------------ helpers
    def _sensed(self):
        if self.noise is None:
            return self.d
        from pick_place import Sensed
        q, qd = self.d.qpos.copy(), self.d.qvel.copy()
        q[:6] += self.noise[0] * self.noise_rng.standard_normal(6)
        qd[:6] += self.noise[1] * self.noise_rng.standard_normal(6)
        return Sensed(self.d, q, qd)

    def _cube_at_rest(self):
        return float(np.linalg.norm(self.d.qvel[self.cube_adr:self.cube_adr + 3])) < 0.05

    def _potential(self):
        """Progress measure used for shaping: reward = change in potential.  Reach the cube, grasp and lift it, carry it to the goal."""
        d = self.d
        cube = d.qpos[self.cube_adr:self.cube_adr + 3]
        tcp = d.site_xpos[self.pp.tcp_id]
        holding = self.grip_cmd > 0 and np.linalg.norm(cube - tcp) < 0.035 and cube[2] > 0.02
        if holding:
            return 1.0 - 2.0 * np.linalg.norm(cube[:2] - self.goal) - 1.0 * max(0.0, 0.05 - cube[2])
        return -2.0 * np.linalg.norm(tcp - cube)

    def _obs(self):
        d = self.d
        a = self.cube_adr
        q, qd = d.qpos[:6], d.qvel[:6]
        if self.noise is not None:
            q = q + self.noise[0] * self.noise_rng.standard_normal(6)
            qd = qd + self.noise[1] * self.noise_rng.standard_normal(6)
        tcp = d.site_xpos[self.pp.tcp_id]
        cube = d.qpos[a:a + 3]
        fing = float(d.qpos[6] + d.qpos[7])
        # fine-scale features: millimetres matter for grasping and placing, but a network sees metres as nearly constant, so the
        # tool-to-cube and tool-to-goal offsets are also given through tanh(offset / 2 cm), which is sensitive at the mm scale
        fine = np.concatenate([np.tanh((cube - self.tool_t) / 0.02), np.tanh((self.goal - self.tool_t[:2]) / 0.02), [np.tanh((self.tool_t[2] - 0.016) / 0.02)]])
        return np.concatenate([q, 0.2 * qd, tcp, self.tool_t, cube, cube - tcp, self.goal, self.goal - cube[:2], fine,
                               [self.grip_cmd / GRIP_CLOSE, 10 * fing, self.steps / MAX_STEPS], self.prev_a]).astype(np.float32)


# ---------------------------------------------------------------------- scripted policy in the same action space
class ScriptedPolicy:
    """State machine that reproduces the classical pick-and-place using only the environment's observations (the baseline for step 1,
    and the demonstrator for imitation learning in step 2)."""
    def __init__(self, env):
        self.env, self.phase, self.hold = env, 0, 0

    def reset(self):
        self.phase, self.hold = 0, 0

    def _goto(self, p, tol=0.002, vmax=1.0):
        """Move the tool target toward p along the planner's POLAR path (radius, angle about the base axis and height change linearly,
        so moves around the base are arcs, never through it).  vmax scales the peak speed (1 = 0.25 m/s)."""
        t = self.env.tool_t
        r0, r1 = np.hypot(t[0], t[1]), np.hypot(p[0], p[1])
        a0, a1 = np.arctan2(t[1], t[0]), np.arctan2(p[1], p[0])
        d_r, d_a, d_z = r1 - r0, a1 - a0, p[2] - t[2]
        rm = 0.5 * (r0 + r1)
        length = np.sqrt(d_r ** 2 + (rm * d_a) ** 2 + d_z ** 2)
        if length < 1e-9:
            return np.zeros(3), True
        s = min(1.0, vmax * STEP_M / length)
        nxt = np.array([(r0 + s * d_r) * np.cos(a0 + s * d_a), (r0 + s * d_r) * np.sin(a0 + s * d_a), t[2] + s * d_z])
        return np.clip((nxt - t) / STEP_M, -1, 1), length < tol

    def __call__(self, obs=None):
        env = self.env
        cube, goal = env.d.qpos[env.cube_adr:env.cube_adr + 3], env.goal
        zg = 0.012
        up = lambda xy, z: np.array([xy[0], xy[1], z])
        grip = -1.0
        if self.phase == 0:   move, ok = self._goto(up(env.cube0, Z_SAFE)); self.phase += ok
        elif self.phase == 1: move, ok = self._goto(up(env.cube0, zg), vmax=0.32); self.phase += ok
        elif self.phase == 2:
            move, grip = np.zeros(3), 1.0
            self.hold += 1
            if self.hold >= 40: self.phase, self.hold = 3, 0          # 0.8 s closing
        elif self.phase == 3: move, ok = self._goto(up(env.cube0, Z_SAFE), vmax=0.32); grip = 1.0; self.phase += ok
        elif self.phase == 4: move, ok = self._goto(up(goal, Z_SAFE), vmax=0.8); grip = 1.0; self.phase += ok
        elif self.phase == 5: move, ok = self._goto(up(goal, zg + 0.004), vmax=0.32); grip = 1.0; self.phase += ok
        elif self.phase == 6:
            move = np.zeros(3); self.hold += 1
            if self.hold >= 30: self.phase, self.hold = 7, 0          # 0.6 s opening
        else: move, ok = self._goto(up(goal, Z_SAFE), vmax=0.32)
        return np.concatenate([move, [grip]])


# ---------------------------------------------------------------------- stateless expert (labels for imitation learning)
class StatelessExpert:
    """Same pick-and-place, but the action depends ONLY on the current state (no phase counter) and varies SMOOTHLY with it, so it can
    label states that a learner visits (DAgger) and a network can copy it.  (A first version switched abruptly at 4 mm of horizontal
    error; a network cannot reproduce a cliff, hovered at the edge and never descended.)
    Horizontal motion is always active.  Descending is gated by horizontal error (full speed under 4 mm, none above 10 mm), and
    horizontal travel is gated by height (none below 30 mm under the safe height, full above 10 mm under it).
    If the cube is dropped it goes and picks it up again from where it lies."""
    ZG, ZP = 0.012, 0.016            # tool height when grasping / when placing (m)
    CLOSE_AT, OPEN_AT = 0.004, 0.004  # close / release once the tool is this close (m) above ZG / ZP

    def __init__(self, env):
        self.env = env
        self._g = ScriptedPolicy(env)

    def reset(self):
        pass

    def _move(self, xy, z, vh, vz):
        """Action toward (xy, z): horizontal along the polar path, vertical gated as described in the class doc."""
        tt = self.env.tool_t
        exy = float(np.linalg.norm(tt[:2] - xy))
        h, _ = self._g._goto(np.array([xy[0], xy[1], tt[2]]), vmax=vh)
        mag = float(np.linalg.norm(h[:2]))
        if mag > 1e-9:                                   # proportional (not constant-speed) near the target: a smooth label to copy
            h = h * min(1.0, 0.5 * exy / STEP_M / mag)
        hs = 1.0 if exy < 0.02 else float(np.clip((tt[2] - (Z_SAFE - 0.03)) / 0.02, 0.0, 1.0))
        dz = z - tt[2]
        vzc = np.clip(0.5 * dz / STEP_M, -vz, vz)
        if dz < 0:
            vzc *= float(np.clip((0.010 - exy) / 0.006, 0.0, 1.0))
        return np.array([h[0] * hs, h[1] * hs, vzc])

    def __call__(self, obs=None):
        env, d = self.env, self.env.d
        cube = d.qpos[env.cube_adr:env.cube_adr + 3]
        tcp = d.site_xpos[env.pp.tcp_id]
        tt, goal = env.tool_t, env.goal
        fing, fing_v = d.qpos[6], abs(d.qvel[6])
        near_cube = np.linalg.norm(cube - tcp) < 0.03
        closed_on_cube = env.grip_cmd > 0 and fing > 0.0115 and fing_v < 0.02 and near_cube     # fingers have stopped on the cube
        holding = env.grip_cmd > 0 and near_cube and cube[2] > 0.03
        if holding or (closed_on_cube and np.linalg.norm(cube[:2] - goal) > 0.02):
            exy = float(np.linalg.norm(tt[:2] - goal))
            if exy > 0.006 or tt[2] > self.ZP + self.OPEN_AT:
                z = Z_SAFE if exy > 0.02 else self.ZP
                return np.concatenate([self._move(goal, z, 0.8 if exy > 0.05 else 0.32, 0.32 if exy < 0.05 or tt[2] < Z_SAFE - 0.01 else 0.8), [1.0]])
            return np.concatenate([np.zeros(3), [-1.0]])                          # release
        exy = float(np.linalg.norm(tt[:2] - cube[:2]))
        if env.grip_cmd > 0 and not closed_on_cube and tcp[2] < 0.02 and exy < 0.01:
            return np.concatenate([np.zeros(3), [1.0]])                           # still closing: wait
        if exy < 0.006 and tt[2] <= self.ZG + self.CLOSE_AT:
            return np.concatenate([np.zeros(3), [1.0]])                           # at the bottom: close
        z = Z_SAFE if exy > 0.02 else self.ZG
        return np.concatenate([self._move(cube[:2], z, 0.8 if exy > 0.05 else 0.32, 0.32), [-1.0]])