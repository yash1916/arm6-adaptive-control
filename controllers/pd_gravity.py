"""PD plus gravity compensation.

    tau = Kp (q_ref - q) + Kd (qd_ref - qd) + g(q)

g(q) is the torque that holds the arm still against gravity at the current joint angles, computed from `model`.
Pass the NOMINAL model (what the controller believes). If the real robot differs (e.g. an unknown payload),
the compensation is wrong by exactly that difference.
"""
import mujoco
from .pd import PD


class PDGravity(PD):
    def __init__(self, model, kp, kd):
        super().__init__(kp, kd)
        self._model = model
        self._scratch = mujoco.MjData(model)  # separate data, so we do not disturb the simulation

    def gravity(self, m, d):
        s = self._scratch
        s.qpos[:] = d.qpos
        s.qvel[:] = 0.0                        # zero velocity: only gravity remains in qfrc_bias
        mujoco.mj_forward(self._model, s)
        return s.qfrc_bias[:self.n].copy()

    def __call__(self, m, d, q_ref, qd_ref, qdd_ref):
        return super().__call__(m, d, q_ref, qd_ref, qdd_ref) + self.gravity(m, d)
