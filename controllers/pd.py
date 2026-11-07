"""PD controller (works on the first n joints, where n = len(kp); extra joints such as fingers or a cube are ignored).

    tau = Kp (q_ref - q) + Kd (qd_ref - qd)

Every controller in this folder has the same call signature:
    tau = controller(m, d, q_ref, qd_ref, qdd_ref)
"""
import numpy as np


class PD:
    def __init__(self, kp, kd):
        self.kp = np.asarray(kp, float)
        self.kd = np.asarray(kd, float)
        self.n = len(self.kp)

    def reset(self):
        pass

    def __call__(self, m, d, q_ref, qd_ref, qdd_ref):
        n = self.n
        return self.kp * (q_ref - d.qpos[:n]) + self.kd * (qd_ref - d.qvel[:n])