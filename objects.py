"""Random graspable objects: boxes and upright cylinders of different size and mass.

The scene has ONE free body ("cube") with two geoms on it, a box and a cylinder.  apply_object() switches on the one
that is needed, resizes it, and sets mass and inertia, so the same compiled model serves every shape.

An object spec is a dict:
    shape  "box" | "cyl"
    half   box: (hx, hy) half extents in the object's own x, y        cyl: radius r
    h      full height (m)
    mass   kg
Gripper limits that shape the sampling: the jaws open 56 mm, so the width between the fingers must be <= 46 mm; the finger
pads are 70 mm long, so objects are at most 60 mm tall.
"""
import numpy as np
import mujoco

DEFAULT_OBJ = dict(shape="box", half=(0.015, 0.015), h=0.03, mass=0.1)       # the original 30 mm, 0.1 kg cube
RANGES = dict(width=(0.020, 0.046), length_max=0.065, radius=(0.010, 0.023), height=(0.020, 0.060), mass=(0.05, 0.50))


def sample_object(rng, ranges=RANGES):
    r = ranges
    h, mass = rng.uniform(*r["height"]), rng.uniform(*r["mass"])
    if rng.random() < 0.5:
        w = rng.uniform(*r["width"]); L = rng.uniform(w, r["length_max"])     # w = the short side, L = the long side
        half = (0.5 * w, 0.5 * L) if rng.random() < 0.5 else (0.5 * L, 0.5 * w)
        return dict(shape="box", half=half, h=h, mass=mass)
    return dict(shape="cyl", half=rng.uniform(*r["radius"]), h=h, mass=mass)


def inertia(spec):
    m, h = spec["mass"], spec["h"]
    if spec["shape"] == "box":
        hx, hy = spec["half"]; a, b = 2 * hx, 2 * hy
        return m / 12 * np.array([b * b + h * h, a * a + h * h, a * a + b * b])
    r = spec["half"]
    return np.array([m / 12 * (3 * r * r + h * h)] * 2 + [0.5 * m * r * r])


def apply_object(m, d, spec):
    """Write the object into the model (call before mj_resetData)."""
    gb, gc, body = m.geom("cube_geom").id, m.geom("cyl_geom").id, m.body("cube").id
    box = spec["shape"] == "box"
    act, off = (gb, gc) if box else (gc, gb)
    h = spec["h"]
    if box:
        hx, hy = spec["half"]; size = [hx, hy, h / 2]; rb = float(np.linalg.norm(size))
    else:
        r = spec["half"]; size = [r, h / 2, 0.0]; rb = float(np.hypot(r, h / 2))
    m.geom_size[act] = size; m.geom_rbound[act] = rb
    m.geom_aabb[act] = [0, 0, 0, size[0], size[1] if box else size[0], size[2] if box else h / 2]
    m.geom_contype[act] = m.geom_conaffinity[act] = 1
    m.geom_rgba[act] = [0.85, 0.15, 0.15, 1] if box else [0.15, 0.35, 0.85, 1]
    m.geom_contype[off] = m.geom_conaffinity[off] = 0
    m.geom_rgba[off, 3] = 0.0
    m.body_mass[body] = spec["mass"]
    m.body_inertia[body] = inertia(spec)
    mujoco.mj_setConst(m, d)


def grasp_yaw(spec, obj_yaw, phi_c):
    """Tool yaw for grasping. The fingers close along the tool's y axis, which points along (cos psi, sin psi).
    With the wrist roll joint (joint 6) at zero the fingers close TANGENTIALLY, i.e. psi = phi_c + pi/2 (phi_c = direction of the
    object seen from the base). We pick the allowed grasp yaw closest to that, so joint 6 never has to turn more than 90 deg.
    Cylinder: any yaw works. Box: the fingers must close across the SHORT side, which allows psi0 + k*pi.
    Near-square box: psi0 + k*pi/2."""
    nat = phi_c + np.pi / 2
    if spec["shape"] == "cyl":
        return float(nat)
    hx, hy = spec["half"]
    psi0 = obj_yaw if hx <= hy else obj_yaw + np.pi / 2
    period = np.pi / 2 if abs(hx - hy) < 0.001 else np.pi
    return float(psi0 + period * np.round((nat - psi0) / period))


def grasp_height(spec):
    """Tool-tip height for grasping: 12 mm for tall objects, lower for short ones so the pads still overlap them."""
    return float(min(0.012, 0.4 * spec["h"]))