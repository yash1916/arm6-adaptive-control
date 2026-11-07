"""Shared helpers: load the model, a smooth trajectory, the simulation loop and the metrics."""
import numpy as np
import mujoco

N = 6                       # number of joints


def load(path="arm6.xml"):
    m = mujoco.MjModel.from_xml_path(path)
    d = mujoco.MjData(m)
    return m, d


def quintic(q0, q1, T, t):
    """Smooth point-to-point motion q0 -> q1 in T seconds (zero velocity/acceleration at both ends).
    Returns reference position, velocity, acceleration at time t."""
    s = np.clip(t / T, 0.0, 1.0)
    pos = 10 * s**3 - 15 * s**4 + 6 * s**5
    vel = (30 * s**2 - 60 * s**3 + 30 * s**4) / T
    acc = (60 * s - 180 * s**2 + 120 * s**3) / T**2
    if t >= T:
        vel = acc = 0.0
    dq = q1 - q0
    return q0 + pos * dq, vel * dq, acc * dq


def simulate(m, d, controller, q0, q1, T_move=2.0, T_total=4.0):
    """Run one experiment. The controller is any callable (m, d, q_ref, qd_ref, qdd_ref) -> torque vector."""
    mujoco.mj_resetData(m, d)
    d.qpos[:] = q0
    tau_max = m.actuator_ctrlrange[:, 1]
    steps = int(T_total / m.opt.timestep)
    log = {k: [] for k in ("t", "q", "qref", "tau")}
    for _ in range(steps):
        mujoco.mj_forward(m, d)                         # refresh M, bias forces ... from the current state
        q_ref, qd_ref, qdd_ref = quintic(q0, q1, T_move, d.time)
        tau = controller(m, d, q_ref, qd_ref, qdd_ref)
        d.ctrl[:] = np.clip(tau, -tau_max, tau_max)     # motors have torque limits
        log["t"].append(d.time); log["q"].append(d.qpos.copy())
        log["qref"].append(q_ref); log["tau"].append(d.ctrl.copy())
        mujoco.mj_step(m, d)
    return {k: np.array(v) for k, v in log.items()}


def metrics(log, T_move=2.0, tol=0.01):
    """rms/peak/final tracking error (rad), settling time after the move (s), control effort (integral of tau^2)."""
    t, e = log["t"], log["qref"] - log["q"]
    dt = t[1] - t[0]
    err = np.linalg.norm(e, axis=1)                     # error size across all joints at each instant
    bad = np.where(err > tol)[0]
    settle = 0.0 if len(bad) == 0 else max(0.0, t[bad[-1]] - T_move)
    return dict(rms_err=float(np.sqrt(np.mean(err**2))), peak_err=float(err.max()),
                final_err=float(err[-1]), settle_s=float(settle),
                effort=float(np.sum(log["tau"]**2) * dt))