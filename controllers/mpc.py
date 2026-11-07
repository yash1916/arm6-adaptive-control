"""Model predictive control (MPC) on top of inverse dynamics.

Idea
  Inverse dynamics, tau = M(q) a + h(q, qd), makes every joint behave like a double integrator: qdd = a.
  So the MPC only has to choose the joint ACCELERATIONS a over a short horizon, and the torque follows from the model.

  Error dynamics per joint (e = q_ref - q, u = a - qdd_ref):   e_{k+1} = e_k + dt*ed_k - dt^2/2 * u_k
                                                              ed_{k+1} = ed_k - dt * u_k
  Every MPC step solves, for each joint, a least-squares problem over the next N samples:
        minimise   sum_k  qe*e_k^2 + qv*ed_k^2 + r*u_k^2
        subject to a_lo <= qdd_ref(k) + u_k <= a_hi            (acceleration bounds that come from the TORQUE LIMITS)
  and applies only the first acceleration (receding horizon), then solves again `every` simulation steps later.

What the horizon buys you: the acceleration bounds are applied across the whole horizon using the previewed reference,
so the controller knows ahead of time when the reference asks for more than the motors can deliver.
Honest limits: bounds use the nominal model at the current state, held constant over the horizon; coupling between joints
is estimated from the reference accelerations; velocity and position limits are not enforced.
"""
import numpy as np
from scipy.optimize import lsq_linear


class MPC:
    uses_preview = True                       # PickPlace gives us the future reference through self.preview(t_ahead)

    def __init__(self, model, N=25, dt=0.02, wn=20.0, qv=0.01, r_scale=1.0, margin=0.9, every=2):
        """model: NominalModel. N*dt = horizon length (s). wn sets the weights: r = 1/wn^4 (LQR-style closed-loop bandwidth).
        margin: use only this fraction of the torque limit when computing acceleration bounds. every: sim steps per MPC solve."""
        self.model, self.N, self.dt, self.margin, self.every = model, N, dt, margin, every
        qe, r = 1.0, r_scale / wn**4
        # prediction matrices for the state x = [e, ed]:  X = Phi x0 + Gamma U
        A = np.array([[1.0, dt], [0.0, 1.0]]); B = np.array([[-0.5 * dt**2], [-dt]])
        Phi = np.zeros((2 * N, 2)); Gam = np.zeros((2 * N, N)); Ak = np.eye(2)
        for k in range(N):
            Ak = A @ Ak if k else A
            Phi[2 * k:2 * k + 2] = Ak
            for j in range(k + 1):
                Gam[2 * k:2 * k + 2, j] = (np.linalg.matrix_power(A, k - j) @ B).ravel()
        self.Phi_e, self.Phi_v = Phi[0::2], Phi[1::2]
        sq = np.sqrt([qe, qv, r])
        self.A_ls = np.vstack([sq[0] * Gam[0::2], sq[1] * Gam[1::2], sq[2] * np.eye(N)])
        self.sq = sq
        self.preview = None                   # set by PickPlace: preview(t_ahead) -> (q, qd, qdd) of the reference
        self.a_cmd = None
        self.k = 0
        self.last_solve_ok = True

    def reset(self):
        self.a_cmd, self.k = None, 0

    def _solve(self, m, d, q_ref, qd_ref, qdd_ref):
        n, N, dt = len(q_ref), self.N, self.dt
        e0, ed0 = q_ref - d.qpos[:n], qd_ref - d.qvel[:n]
        # previewed reference accelerations over the horizon (N x n)
        if self.preview is not None:
            qdd_prev = np.array([self.preview(k * dt)[2] for k in range(N)])
        else:
            qdd_prev = np.tile(qdd_ref, (N, 1))
        # torque limits -> acceleration limits, joint by joint (coupling estimated with the reference acceleration)
        M, h = self.model.dynamics(d)
        tau_max = self.margin * m.actuator_ctrlrange[:n, 1]
        Md = np.diag(M)
        coupling = M @ qdd_ref - Md * qdd_ref
        a_hi = (tau_max - h - coupling) / Md
        a_lo = (-tau_max - h - coupling) / Md
        a = np.empty(n)
        for j in range(n):
            lb, ub = a_lo[j] - qdd_prev[:, j], a_hi[j] - qdd_prev[:, j]      # bounds on u = a - qdd_ref
            bad = lb > ub
            if bad.any():                                                      # infeasible (should not happen): meet in the middle
                mid = 0.5 * (lb + ub); lb = np.where(bad, mid, lb); ub = np.where(bad, mid, ub); self.last_solve_ok = False
            b = -np.concatenate([self.sq[0] * self.Phi_e @ [e0[j], ed0[j]], self.sq[1] * self.Phi_v @ [e0[j], ed0[j]], np.zeros(N)])
            u = lsq_linear(self.A_ls, b, bounds=(lb, ub), method="bvls").x
            a[j] = qdd_ref[j] + u[0]
        self.a_cmd = a

    def __call__(self, m, d, q_ref, qd_ref, qdd_ref):
        if self.a_cmd is None or self.k % self.every == 0:
            self._solve(m, d, q_ref, qd_ref, qdd_ref)
        self.k += 1
        M, h = self.model.dynamics(d)                       # inverse dynamics with the CURRENT state, every step
        return M @ self.a_cmd + h