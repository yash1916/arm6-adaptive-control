"""PID controller: PD plus an integral term that removes steady-state error (e.g. the sag caused by gravity).

    tau = Kp e + Ki * integral(e) + Kd (qd_ref - qd)        e = q_ref - q

Anti-windup: the integral TERM (Ki * integral) is clamped to +/- i_limit (N m), so it cannot grow without bound
while a joint is saturated or blocked.
"""
import numpy as np


class PID:
    def __init__(self, kp, ki, kd, i_limit):
        self.kp, self.ki, self.kd = (np.asarray(x, float) for x in (kp, ki, kd))
        self.i_limit = np.asarray(i_limit, float)
        self.n = len(self.kp)
        self.integral = np.zeros(self.n)

    def reset(self):
        self.integral[:] = 0.0

    def __call__(self, m, d, q_ref, qd_ref, qdd_ref):
        n = self.n
        e = q_ref - d.qpos[:n]
        self.integral += e * m.opt.timestep
        cap = self.i_limit / np.maximum(self.ki, 1e-12)        # largest allowed integral value per joint
        self.integral = np.clip(self.integral, -cap, cap)
        return self.kp * e + self.ki * self.integral + self.kd * (qd_ref - d.qvel[:n])