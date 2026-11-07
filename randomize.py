"""Randomise the SIMULATED robot (the plant) between episodes.  Controllers keep their nominal model, so every draw is a
model error they were not told about.

    rd = DynamicsRandomizer()           # rd.plant is the model to simulate, rd.nominal is the controller's belief
    p = rd.sample(rng)                  # {"mass_scale", "damping_scale", "payload"}
    rd.apply(p)                         # write it into rd.plant (cheap, in place)

mass_scale     all moving arm bodies, mass and inertia            (x)
damping_scale  joint viscous damping of the six arm joints        (x)
payload        unknown mass fixed at the tool tip                 (kg)
"""
import numpy as np
import mujoco
from plant import make_plant, ARM_BODIES

RANGES = dict(mass_scale=(0.8, 1.5), damping_scale=(0.5, 3.0), payload=(0.0, 1.0))


class DynamicsRandomizer:
    def __init__(self, xml="arm6_pick.xml"):
        self.plant, self.nominal = make_plant(xml)
        p = self.plant
        self.ids = [p.body(b).id for b in ARM_BODIES]
        self.m0, self.i0 = p.body_mass[self.ids].copy(), p.body_inertia[self.ids].copy()
        self.b0 = p.dof_damping[:6].copy()
        self.pay = p.body("payload").id

    def sample(self, rng, ranges=RANGES):
        lo, hi = ranges["damping_scale"]
        return dict(mass_scale=float(rng.uniform(*ranges["mass_scale"])),
                    damping_scale=float(np.exp(rng.uniform(np.log(lo), np.log(hi)))),      # log-uniform: x0.5 and x2 equally likely
                    payload=float(rng.uniform(*ranges["payload"])))

    def apply(self, params):
        p = self.plant
        p.body_mass[self.ids] = self.m0 * params["mass_scale"]
        p.body_inertia[self.ids] = self.i0 * params["mass_scale"]
        p.dof_damping[:6] = self.b0 * params["damping_scale"]
        kg = max(params["payload"], 1e-6)
        p.body_mass[self.pay] = kg
        p.body_inertia[self.pay] = [kg * 1e-4] * 3
        mujoco.mj_setConst(p, mujoco.MjData(p))