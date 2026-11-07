"""Adaptive control of a robot arm with unknown mass, friction and payload (Slotine and Li, 1987).

The arm's dynamics are linear in a few unknown parameters:

    tau = s * Y1 + d * Y2 + mp * Y3 + A * qdd          s  = scale of all link masses
                                                        d  = scale of joint damping
                                                        mp = payload mass at the tool (kg)

where Y1, Y2, Y3 are known functions of the motion (the "regressor") and A is the (known) motor armature.
With  e = q - q_ref,  qd_r = qd_ref - Lam e,  qdd_r = qdd_ref - Lam ed,  s_v = qd - qd_r = ed + Lam e :

    tau = Y(q, qd, qd_r, qdd_r) theta_hat + A qdd_r - K_D s_v          control law
    theta_hat_dot = - Gamma Y^T s_v                                   adaptation law (projected onto sensible bounds)

This needs no acceleration measurement and no integral term; the parameter estimates play the role of the integral.
The regressor columns come from MuJoCo's inverse dynamics on two copies of the NOMINAL model (never the plant):
  copy A: links only (no damping, no armature, payload ~ 0)         -> Y1
  copy P: the same with a 1 kg payload                               -> Y3 = Y_P - Y1
The Coriolis term C(q, qd) qd_r is obtained from the quadratic form Q(v) = C(q, v) v by polarisation:
  C(q, qd) qd_r = 0.5 [Q(qd + qd_r) - Q(qd) - Q(qd_r)].
"""
import numpy as np
import mujoco
from .computed_torque_utils import mass_matrix


class AdaptiveSL:
    def __init__(self, nominal, lam=20.0, kd_gain=20.0, gamma=(4.0, 4.0, 4.0), theta0=(1.0, 1.0, 0.0),
                 lo=(0.5, 0.0, 0.0), hi=(3.0, 8.0, 5.0), n=6):
        self.n, self.lam, self.kd_gain = n, lam, kd_gain
        self.gamma, self.theta0 = np.asarray(gamma, float), np.asarray(theta0, float)
        self.lo, self.hi = np.asarray(lo, float), np.asarray(hi, float)
        self.b0 = nominal.dof_damping[:n].copy()
        self.arm = nominal.dof_armature[:n].copy()
        self.mA = self._copy(nominal)
        self.mP = self._copy(nominal)
        pid = nominal.body("payload").id
        self.mA.body_mass[pid], self.mA.body_inertia[pid] = 1e-6, [1e-10] * 3
        self.mP.body_mass[pid], self.mP.body_inertia[pid] = 1.0, [1e-4] * 3
        for mm in (self.mA, self.mP):
            mm.dof_damping[:] = 0.0
            mm.dof_armature[:] = 0.0
            mm.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
            mm.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_ACTUATION
        self.dA, self.dP = mujoco.MjData(self.mA), mujoco.MjData(self.mP)
        self.M = mujoco.MjData(nominal)                               # for K_D (diagonal of the nominal mass matrix)
        self.nom = nominal
        self.reset()

    @staticmethod
    def _copy(m):
        import copy
        return copy.copy(m)

    def reset(self):
        self.theta = self.theta0.copy()

    def _id(self, m, dat, q, v, a):
        n = self.n
        dat.qpos[:n], dat.qvel[:], dat.qacc[:] = q, 0.0, 0.0
        dat.qvel[:n], dat.qacc[:n] = v, a
        mujoco.mj_inverse(m, dat)
        return dat.qfrc_inverse[:n].copy()

    def _regressor(self, q, qd, qd_r, qdd_r):
        cols = []
        for m, dat in ((self.mA, self.dA), (self.mP, self.dP)):
            zero = np.zeros(self.n)
            base = self._id(m, dat, q, zero, qdd_r)                                    # M qdd_r + g
            Q = lambda v: self._id(m, dat, q, v, zero) - self._id(m, dat, q, zero, zero)
            gam = 0.5 * (Q(qd + qd_r) - Q(qd) - Q(qd_r))                                # C(q, qd) qd_r
            cols.append(base + gam)
        Y1, Y3 = cols[0], cols[1] - cols[0]
        Y2 = self.b0 * qd_r
        return np.column_stack([Y1, Y2, Y3])

    def __call__(self, m, d, q_ref, qd_ref, qdd_ref):
        n, dt = self.n, m.opt.timestep
        q, qd = d.qpos[:n].copy(), d.qvel[:n].copy()
        e, ed = q - q_ref, qd - qd_ref
        qd_r, qdd_r = qd_ref - self.lam * e, qdd_ref - self.lam * ed
        s_v = qd - qd_r
        Y = self._regressor(q, qd, qd_r, qdd_r)
        self.M.qpos[:], self.M.qvel[:] = d.qpos, 0.0
        mujoco.mj_forward(self.nom, self.M)
        KD = self.kd_gain * np.diag(mass_matrix(self.nom, self.M))[:n]                         # damping gain: kd_gain x the nominal inertia of each joint
        tau = Y @ self.theta + self.arm * qdd_r - KD * s_v
        self.theta = np.clip(self.theta - dt * self.gamma * (Y.T @ s_v), self.lo, self.hi)    # adaptation, with projection
        return tau