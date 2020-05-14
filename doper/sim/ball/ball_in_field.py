"""
All the equations below assumes that the agent is a ball with radius R, and the potential force
is being produced only by L2 distance between current agents location and target point.
The presence of rolling friction is also assumed
"""

import taichi as ti
import numpy as np


ti.init(arch=ti.cpu, default_fp=ti.f32)

constants = {
    "radius": 0.05,
    "g": 9.8,
    "f": 0.001,
    "ro": 1000,
}
constants["volume"] = 4 * np.pi * (constants["radius"] ** 3) / 3
constants["mass"] = constants["volume"] * constants["ro"]

sim_steps = 4000


@ti.func
def compute_potential(
    curr_coordinate, target_coordinate,
):
    """Computes L2 potential value

    Args:
        curr_coordinate (ti.Vector): current agent position (center of masses)
        target_coordinate (ti.Vector): target point

    Returns:
        float: L2 potential
    """

    return ti.mean((target_coordinate - curr_coordinate) ** 2)


@ti.func
def cumpute_l2_force(
    curr_coordinate, target_coordinate,
):
    """Computes force produced by L2 potential

    Args:
        curr_coordinate (ti.Vector): current agent position (center of masses)
        target_coordinate (ti.Vector): target point

    Returns:
        float: the amount of force produced by L2 potential
    """
    return 2 * (target_coordinate - curr_coordinate)


@ti.func
def compute_rolling_friction_force(
    velocity_direction, mass, g, f, radius,
):
    """Computes rolling friction force value, flat land assumed

    Args:
        velocity_direction (int): velocity sign, [-1, 0 or 1]
        mass (float): balls mass, kilogram
        g (float): gravitational acceleration, metre / second ^ 2)
        f (float): rolling friction coefficient, metre
        radius (float): balls radius, metre

    Returns:
        float: the amount of the rolling friction force
    """
    assert velocity_direction in [-1, 0, 1,], "velocity_direction should be in [-1, 0, 1]"
    normal_force = mass * g
    return -velocity_direction * f * normal_force / radius


@ti.func
def compute_acceleration(
    forces, mass,
):
    """compute acceleration given list of forces and mass

    Args:
        forces (list): list of forces
        mass (float): balls mass, kilogram

    Returns:
        float: balls acceleration, metre / second ^ 2
    """
    total_force = 0
    for f in forces:
        total_force += f
    return total_force / mass


@ti.func
def sim_step(
    curr_coordinate, velocity, dt,
):
    """Makes one step of the simulation

    Args:
        curr_coordinate (ti.Vector): current agent position (center of masses)
        velocity (ti.var): balls velocity
        dt (float): time intrval

    Returns:
        list: [description]
    """
    l2_force = cumpute_l2_force(x, target_x,)
    friction_force = compute_rolling_friction_force(
        np.sign(v), constants["mass"], constants["g"], constants["f"], constants["radius"],
    )
    acceleration = compute_acceleration([l2_force, friction_force], constants["mass"],)

    ti.atomic_add(velocity, acceleration * dt)
    ti.atomic_add(curr_coordinate, velocity * dt)
    return curr_coordinate, velocity


@ti.kernel
def run_simulation(x: ti.f32, v: ti.f32):
    t0 = 0
    for t1 in range(0, 20, sim_steps):
        dt = t1 - t0
        x, v = sim_step(curr_coordinate=x, velocity=v, constants=constants, dt=dt,)
        t0 = t1


if __name__ == "__main__":
    target_x = ti.Vector(2, dt=ti.f32, shape=1)
    x = ti.Vector(2, dt=ti.f32, shape=1)
    v = ti.Vector(2, dt=ti.f32, shape=1)

    x[0] = [2.0, 0.5]
    target_x[0] = [0.0, 0.0]
    v[0] = [0.1, 0.5]

    run_simulation(x, v)
