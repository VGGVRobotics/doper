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
    "radius": 0.05,
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
        t (int): current timestamp

    Returns:
        float: the amount of force produced by L2 potential
    """
    return 2 * (target_x - x[t, 0])


@ti.func
def compute_rolling_friction_force(velocity_direction,):
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
    normal_force = mass * g
    return -velocity_direction * f * normal_force / radius


@ti.kernel
def sim_step(t: ti.i32,):
    """Makes one step of the simulation

    Args:
        t (float): time id

    Returns:
        list: [description]
    """
    l2_force = cumpute_l2_force(t,)
    friction_force = compute_rolling_friction_force(t,)
    acceleration[t, 0] = (l2_force + friction_force) / mass

    v[t, 0] = v[t - 1, 0] + acceleration[t, 0] * dt
    x[t, 0] = x[t - 1, 0] + v[t, 0] * dt


#@ti.kernel
def run_simulation():
    for t in range(1, sim_steps):
        sim_step(t)

        gui.clear(0x3C733F)

        gui.circle(target_x[None], radius=50 // 2, color=0x00000)

        gui.circle(x[t, 0], radius=50, color=0xF20530)

        gui.show()
        print(x[t, 0][0])
        print(x[t, 0][1])
        print(v[t, 0][0])
        print(v[t, 0][1])
        print(target_x[None][1])

@ti.layout
def place():
    ti.root.dense(ti.l, sim_steps).dense(ti.i, 1).place(x, v, acceleration)
    ti.root.place(target_x, l2_force, friction_force)
    ti.root.place(dt, radius, g, f, ro, volume, mass)


if __name__ == "__main__":
    sim_steps = 40
    max_time = 20

    target_x = ti.Vector(2, dt=ti.f32)
    x = ti.Vector(2, dt=ti.f32)
    v = ti.Vector(2, dt=ti.f32)
    acceleration = ti.Vector(2, dt=ti.f32)
    l2_force = ti.Vector(2, dt=ti.f32)
    friction_force = ti.Vector(2, dt=ti.f32)

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

    x[0, 0] = [2.0, 0.5]
    target_x[None] = [0.0, 0.0]
    v[0, 0] = [0.1, 0.5]
    acceleration[0, 0] = [0, 0]
    dt[None] = max_time / sim_steps
    run_simulation()
