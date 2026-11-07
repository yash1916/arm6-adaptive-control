"""Small helper shared by the model-based controllers."""
import numpy as np
import mujoco


def mass_matrix(m, d):
    """Full joint-space mass matrix (handles the mj_fullM argument order changing between MuJoCo versions)."""
    M = np.zeros((m.nv, m.nv))
    try:
        mujoco.mj_fullM(m, d, M)          # newer MuJoCo
    except TypeError:
        mujoco.mj_fullM(m, M, d.qM)       # older MuJoCo
    return M