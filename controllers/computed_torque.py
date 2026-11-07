"""Computed-torque (inverse-dynamics) control: a nonlinear controller that cancels the robot's dynamics.

    tau = M(q) * ( qdd_ref + Kp e + Kd ed + Ki * integral(e) ) + h(q, qd)           e = q_ref - q

If the model is exact, M*qdd + h = tau turns into  e'' + Kd e' + Kp e = 0 : a simple linear system you tune directly.
Where the model comes from:
    model=None            -> read M and h from the simulator itself (a PERFECT model: handy for learning, not realistic)
    model=NominalModel()  -> the controller's own model, which may differ from the real plant
ki > 0 adds an integral term that corrects the slow errors a wrong model leaves behind.
"""
import numpy as np
from .computed_torque_utils import mass_matrix
from .model import NominalModel


HIST_TAUS = (0.02, 0.1, 0.5)          # time constants (s) of the running averages of the recent model error


class History:
    """What the controller has learned about its model error from the recent past.

    Each step it estimates the torque the model could not explain, r = tau_applied - M q'' - h, where q'' is the measured
    change in joint velocity over the last step.  It keeps running averages of r at three time scales (20 ms, 100 ms,
    0.5 s).  A heavier arm or payload shows up here as a persistent r, which one instantaneous state cannot reveal."""
    def __init__(self, n=6, dt=0.002, taus=HIST_TAUS, clip=None):
        self.n, self.dt, self.alpha, self.clip = n, dt, [1.0 - np.exp(-dt / t) for t in taus], clip
        self.reset()

    def reset(self):
        self.qd_prev, self.avg = None, np.zeros((len(self.alpha), self.n))

    def update(self, qd, tau_prev, M, h):
        if self.qd_prev is not None:
            r = tau_prev - M @ ((qd - self.qd_prev) / self.dt) - h
            if self.clip is not None:
                r = np.clip(r, -self.clip, self.clip)
            for i, a in enumerate(self.alpha):
                self.avg[i] += a * (r - self.avg[i])
        self.qd_prev = qd.copy()
        return self.avg.ravel().copy()


class ComputedTorque:
    def __init__(self, wn=20.0, zeta=1.0, model=None, model_damping=False, ki=0.0, i_limit=20.0, residual=None, res_frac=0.3, res_scale=1.0):
        """wn = closed-loop bandwidth (rad/s); zeta = damping ratio. Kp = wn^2, Kd = 2 zeta wn for every joint.
        model_damping only matters when model=None (then it also cancels the simulator's joint damping).
        ki: integral gain on the acceleration level (1/s^3 units); i_limit caps the integral term (rad/s^2); it must exceed the acceleration error the model mismatch causes (about 10 rad/s^2 at the shoulder for the test mismatch), or the integral saturates."""
        if model is not None and not hasattr(model, "dynamics"):       # a raw MjModel was given: wrap it
            model = NominalModel(model, include_damping=True)
        self.kp, self.kd, self.ki = wn**2, 2.0 * zeta * wn, ki
        self.model, self.model_damping, self.i_limit = model, model_damping, i_limit
        self.integral = None
        self.last_a = None
        self.history, self.last_hist = None, np.zeros(3 * 6)     # created on first call (needs the time step)
        # residual: object with .predict(x) -> 6 torques (a learned model-error compensator, see residual_net.py);
        # its output is limited to res_frac of each motor's torque limit so a bad prediction cannot do much harm
        self.residual, self.res_frac, self.res_scale = residual, res_frac, res_scale

    def reset(self):
        self.integral = None
        if self.history is not None:
            self.history.reset()
        self.last_hist = np.zeros(3 * 6)

    def __call__(self, m, d, q_ref, qd_ref, qdd_ref):
        n = len(q_ref)                                    # use only the arm joints (extra joints are ignored)
        self.last_q, self.last_qd = d.qpos[:n].copy(), d.qvel[:n].copy()      # what the controller saw (noisy, if the sensors are)
        e, ed = q_ref - d.qpos[:n], qd_ref - d.qvel[:n]
        if self.integral is None:
            self.integral = np.zeros(n)
        if self.ki > 0:
            self.integral = np.clip(self.integral + e * m.opt.timestep, -self.i_limit / self.ki, self.i_limit / self.ki)
        if self.history is None:
            self.history = History(n, m.opt.timestep)
        a = qdd_ref + self.kp * e + self.kd * ed + self.ki * self.integral    # desired joint acceleration
        self.last_a = a
        if self.model is not None:
            M, h = self.model.dynamics(d)
            lim = self.res_frac * m.actuator_ctrlrange[:n, 1]
            self.last_hist = self.history.update(d.qvel[:n], d.ctrl[:n].copy(), M, h)      # d.ctrl still holds the previous command
            tau = M @ a + h
            if self.residual is not None:
                from residual_net import features
                use_hist = self.residual.W[0].shape[0] > 19
                x = features(d.qpos[:n], d.qvel[:n], a, d.ctrl[6] > 0.01, self.last_hist if use_hist else None)
                tau = tau + self.res_scale * np.clip(self.residual.predict(x), -lim, lim)
            return tau
        tau = mass_matrix(m, d)[:n, :n] @ a + d.qfrc_bias[:n]
        if self.model_damping:
            tau = tau - d.qfrc_passive[:n]
        return tau