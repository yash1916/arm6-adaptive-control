"""Build the simulated plant and the controller's nominal copy of it, with optional model error.

    plant, nominal = make_plant(mass_scale=1.3, cube_mass=0.4, damping_scale=2.0, torque_scale=0.2)

The controller is given `nominal` (what it believes); the simulation uses `plant` (the truth).
    mass_scale    multiplies the mass and inertia of every moving arm body (links, wrist, gripper, fingers)
    cube_mass     mass of the cube in kg (nominal is 0.1) - an unmodelled payload while it is held
    damping_scale multiplies the joint damping of the arm joints
    torque_scale  multiplies the arm motors' torque limits (the controller knows the real limits)
"""
import copy
import numpy as np
import mujoco

ARM_BODIES = ["Link1Turntable", "Link2Upper", "Link3Forearm", "Link4ForearmRoll", "Link5Wrist", "Link6Gripper", "FingerL", "FingerR"]


def make_plant(xml="arm6_pick.xml", mass_scale=1.0, cube_mass=0.1, damping_scale=1.0, torque_scale=1.0):
    plant = mujoco.MjModel.from_xml_path(xml)
    nominal = copy.deepcopy(plant)
    for b in ARM_BODIES:
        i = plant.body(b).id
        plant.body_mass[i] *= mass_scale
        plant.body_inertia[i] *= mass_scale
    c = plant.body("cube").id
    plant.body_inertia[c] *= cube_mass / plant.body_mass[c]
    plant.body_mass[c] = cube_mass
    plant.dof_damping[:6] *= damping_scale
    plant.actuator_ctrlrange[:6] *= torque_scale
    mujoco.mj_setConst(plant, mujoco.MjData(plant))
    nominal.actuator_ctrlrange[:6] *= torque_scale          # the controller knows the motor limits
    return plant, nominal