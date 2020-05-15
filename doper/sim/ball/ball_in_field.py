"""
All the equations below assumes that the agent is a ball with radius R, and the potential force
is being produced only by L2 distance between current agents location and target point.
The presence of rolling friction is also assumed
"""

import taichi as ti
import numpy as np


ti.init(arch=ti.cpu, default_fp=ti.f32)
gui = ti.GUI("ball", (1024, 1024))

constants = {
    "radius": 0.1,
    "g": 9.8,
    "f": 0.001,
    "ro": 1000,
}
constants["volume"] = 4 * np.pi * (constants["radius"] ** 3) / 3
constants["mass"] = constants["volume"] * constants["ro"]


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
def cumpute_l2_force(t,):
    """Computes force produced by L2 potential

    Args:
        t (int): time id

    Returns:
        float: the amount of force produced by L2 potential
    """
    return 2 * (target_x - x[t - 1, 0])


@ti.func
def compute_rolling_friction_force(t,):
    """Computes rolling friction force value, flat land assumed

    Args:
        t (int): time id

    Returns:
        float: the amount of the rolling friction force
    """
    normal_force = mass * g

    velocity_direction[None] = v[t - 1, 0]
    velocity_direction[None][0] /= ti.abs(velocity_direction[None][0])
    velocity_direction[None][1] /= ti.abs(velocity_direction[None][1])

    return -velocity_direction * f * normal_force / radius


@ti.kernel
def sim_step(t: ti.i32,):
    """Makes one step of the simulation

    Args:
        t (int): time id

    Returns:
        list: [description]
    """
    l2_force = cumpute_l2_force(t,)
    friction_force = compute_rolling_friction_force(t,)
    print(l2_force[0])
    print(friction_force[0])
    acceleration[t, 0] = (world_scale_coeff * l2_force + friction_force) / mass

    v[t, 0] = v[t - 1, 0] + acceleration[t, 0] * dt
    x[t, 0] = x[t - 1, 0] + v[t, 0] * dt


def run_simulation():
    for t in range(1, sim_steps):
        sim_step(t)

        gui.clear(0x3C733F)

        gui.circle(target_x[None], radius=5, color=0x00000)

        gui.circle(x[t, 0], radius=10, color=0xF20530)

        gui.show()

        print(x[t, 0][0], x[t, 0][1])


@ti.layout
def place():
    ti.root.dense(ti.l, sim_steps).dense(ti.i, 1).place(x, v, acceleration)
    ti.root.place(target_x, velocity_direction)
    ti.root.place(dt, radius, g, f, ro, volume, mass)


if __name__ == "__main__":
    sim_steps = 4000
    max_time = 50
    world_scale_coeff = 10

    target_x = ti.Vector(2, dt=ti.f32)
    velocity_direction = ti.Vector(2, dt=ti.f32)
    x = ti.Vector(2, dt=ti.f32)
    v = ti.Vector(2, dt=ti.f32)
    acceleration = ti.Vector(2, dt=ti.f32)

    dt = ti.var(dt=ti.f32)
    radius = ti.var(dt=ti.f32)
    g = ti.var(dt=ti.f32)
    f = ti.var(dt=ti.f32)
    ro = ti.var(dt=ti.f32)
    volume = ti.var(dt=ti.f32)
    mass = ti.var(dt=ti.f32)

    radius[None] = constants["radius"]
    g[None] = constants["g"]
    f[None] = constants["f"]
    ro[None] = constants["ro"]
    volume[None] = constants["volume"]
    mass[None] = constants["mass"]

    x[0, 0] = [0.9, 0.9]
    target_x[None] = [0.5, 0.5]
    v[0, 0] = [0.1, -0.1]
    acceleration[0, 0] = [0.0, 0.0]
    dt[None] = max_time / sim_steps
    run_simulation()
