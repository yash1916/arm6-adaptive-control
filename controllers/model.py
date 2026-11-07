"""The controller's OWN model of the robot, kept separate from the simulated plant.

Why: a model-based controller (computed torque, MPC) needs M(q) and h(q, qd). If it read them from the simulator it
controls, the model would be perfect by construction. Here the controller owns a copy of the nominal model, so we
can make the real plant differ from it (heavier links, heavier payload, more friction) and see what happens.

    M(q) qdd + h(q, qd) = tau        h = Coriolis + gravity (+ joint damping, if include_damping=True)
"""
import numpy as np
import mujoco
from .computed_torque_utils import mass_matrix


class NominalModel:
    def __init__(self, nominal_model, n=6, include_damping=True):
        self.m = nominal_model
        self.s = mujoco.MjData(nominal_model)          # scratch data: never touches the simulation
        self.n = n
        self.include_damping = include_damping

    def dynamics(self, d):
        """Mass matrix M (n x n) and h (n,) at the plant's current state d (only positions/velocities are read)."""
        s, n = self.s, self.n
        s.qpos[:] = d.qpos
        s.qvel[:] = d.qvel
        mujoco.mj_forward(self.m, s)
        M = mass_matrix(self.m, s)[:n, :n]
        h = s.qfrc_bias[:n].copy()
        if self.include_damping:
            h -= s.qfrc_passive[:n]                    # qfrc_passive is the damping force (opposes motion)
        return M, h