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
    """Computes L2 potential value

    Args:
        curr_coordinate (float): current agent position (center of masses)
        target_coordinate (float): target point

    Returns:
        float: L2 potential
    """

    return ti.mean((target_coordinate - curr_coordinate) ** 2)


@ti.func
def cumpute_l2_force(curr_coordinate, target_coordinate):
    """Computes force produced by L2 potential

    Args:
        curr_coordinate (float): current agent position (center of masses)
        target_coordinate (float): target point

    Returns:
        float: the amount of force produced by L2 potential
    """
    return 2 * (target_coordinate - curr_coordinate)


@ti.func
def compute_rolling_friction_force(velocity_direction, mass, g, f, radius):
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
    assert velocity_direction in [-1, 0, 1], 'velocity_direction should be in [-1, 0, 1]'
    N = m * g  # normal force
    return - velocity_direction * f * N / radius


@ti.func
def compute_acceleration(forces, mass):
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
    return total_force / m


@ti.func
def update_state(curr_coordinate,
                 target_coordinate,
                 velocity,
                 constants,
                 dt,
                 ):

    l2_force = cumpute_l2_force(curr_coordinate,
                                target_coordinate,
                                )
    friction_force = compute_rolling_friction_force(np.sign(velocity),
                                                    constants['mass'],
                                                    constants['g'],
                                                    constants['f'],
                                                    constants['radius'],
                                                    )
    acceleration = compute_acceleration([l2_force,
                                         friction_force],
                                        constants['mass'],
                                        )
    velocity += acceleration * dt
    curr_coordinate += velocity * dt
    return curr_coordinate, velocity


@ti.kernel
def run_simulation(x0, v0, target_coordinate):
    t0 = 0
    x = x0.copy()
    v = v0.copy()
    for idx, t1 in np.linspace(0, 20, 4000):
        dt = t1 - t0
        update_state()


def render(gui):
    canvas = gui.canvas
    canvas.clear(bg_color)
    pos_np = x.to_numpy()
    gui.circles(pos_np, radius=particle_radius, color=particle_color)
    gui.show()
