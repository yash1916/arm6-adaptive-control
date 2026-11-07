"""Inverse kinematics by damped least squares (DLS), for the 6 arm joints only.

Given a target tool position and orientation, find joint angles q so that the 'tcp' site reaches it.
Each iteration:   dq = J^T (J J^T + lambda^2 I)^-1 * error      (J = 6x6 Jacobian of the tcp site w.r.t. the arm joints)
The damping lambda keeps dq finite near singular poses (arm stretched out or folded).
"""
import numpy as np
import mujoco

NARM = 6


def tool_down_R(yaw=0.0):
    """Tool axis (site x) pointing straight down; 'yaw' turns the finger-opening axis (site y) about the vertical."""
    c, s = np.cos(yaw), np.sin(yaw)
    x_t = np.array([0.0, 0.0, -1.0])
    y_t = np.array([c, s, 0.0])
    return np.column_stack([x_t, y_t, np.cross(x_t, y_t)])


class IK:
    def __init__(self, m, site="tcp"):
        self.m = m
        self.d = mujoco.MjData(m)                    # scratch data: never touches the running simulation
        self.site = m.site(site).id
        self.lo, self.hi = m.jnt_range[:NARM, 0], m.jnt_range[:NARM, 1]

    def pose_error(self, q, pos_t, R_t):
        d = self.d
        d.qpos[:NARM] = q
        mujoco.mj_kinematics(self.m, d); mujoco.mj_comPos(self.m, d)
        R = d.site_xmat[self.site].reshape(3, 3)
        e_rot = 0.5 * sum(np.cross(R[:, i], R_t[:, i]) for i in range(3))
        return np.concatenate([pos_t - d.site_xpos[self.site], e_rot])

    def solve(self, pos_t, R_t, q_init, iters=300, tol_pos=3e-4, tol_rot=2e-3, damping=0.05, step=0.7):
        """Returns (q, success). success means both errors are under tolerance and joint limits respected."""
        m, d = self.m, self.d
        q = np.clip(np.asarray(q_init, float).copy(), self.lo, self.hi)
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        for _ in range(iters):
            e = self.pose_error(q, pos_t, R_t)
            if np.linalg.norm(e[:3]) < tol_pos and np.linalg.norm(e[3:]) < tol_rot:
                return q, True
            mujoco.mj_jacSite(m, d, jp, jr, self.site)
            J = np.vstack([jp[:, :NARM], jr[:, :NARM]])
            dq = J.T @ np.linalg.solve(J @ J.T + damping**2 * np.eye(6), e)
            q = np.clip(q + step * dq, self.lo, self.hi)
        e = self.pose_error(q, pos_t, R_t)
        return q, bool(np.linalg.norm(e[:3]) < 2 * tol_pos and np.linalg.norm(e[3:]) < 2 * tol_rot)

    def tcp(self, q):
        self.d.qpos[:NARM] = q; mujoco.mj_kinematics(self.m, self.d)
        return self.d.site_xpos[self.site].copy()

    def tcp_path(self, Q):
        """Tool positions along a joint path (K x 3)."""
        return np.array([self.tcp(q) for q in Q])