"""
all the equations below assumes that the agent is a ball with radius R, and the potential force
is being produced only by L2 distance between current agents location and target point.
The presence of rolling friction is also assumed
"""

import taichi as ti
import numpy as np


ti.init(arch=ti.cpu)

screen_res = (800, 400)
g = 9.8


@ti.func
def compute_potential(curr_coordinate, target_coordinate):
    """
    Computes L2 potential value
    curr_coordinate: current agent position (center of masses), ti.val (dim,)
    target_coordinate: target point, ti.val (dim,)
    """
    return ti.mean((target_coordinate - curr_coordinate) ** 2)


@ti.func
def cumpute_l2_force(curr_coordinate, target_coordinate):
    """
    Computes force produced by L2 potential
    curr_coordinate: current agent position (center of masses), ti.val (dim,)
    target_coordinate: target point, ti.val (dim,)
    """
    return 2 * (target_coordinate - curr_coordinate)


@ti.func
def compute_rolling_friction_force(velocity_direction, mass, g, f, radius):
    """
    Computes rolling friction force value, flat land assumed
    velocity_direction: int, -1 or 1
    mass: balls mass, float (1,)
    g: gravitational acceleration, float (1,)
    f: rolling friction coefficient (meters), float (1,)
    radius: balls radius (meters), float (1,)
    """
    N = m * g  # normal force
    return - velocity_direction * f * N / radius


@ti.func
def compute_acceleration(x, x_target, m, v, F_fr):
    return (2 * (x_target - x) - np.sign(v.to_numpy()) * F_fr) / m


@ti.func
def update_state(curr_coordinate,
                 target_coordinate,
                 mass,
                 velocity,
                 F_fr,
                 dt,
                ):
    """
    curr_coordinate: current agent position (center of masses), ti.val (dim,)
    target_coordinate: target point, ti.val (dim,)
    mass: balls mass, float (1,)
    velocity: ahents velocity vector, ti.val (dim,)

    """
    a_curr = compute_acceleration(curr_coordinate,
                                  target_coordinate,
                                  mass,
                                  velocity,
                                  F_fr,
                                  )
    velocity += a_curr * dt
    curr_coordinate += velocity * dt
    return curr_coordinate, velocity


@ti.kernel
def run_simulation():
    for idx, t1 in np.linspace(0, sim_length, num_steps)
        update_state()


def render(gui):
    canvas = gui.canvas
    canvas.clear(bg_color)
    pos_np = x.to_numpy()
    gui.circles(pos_np, radius=particle_radius, color=particle_color)
    gui.show()
