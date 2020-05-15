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


# @ti.func
def mean(var):
    ret = 0
    for i in range(var.n):
        print(i)
        ret += var[i, 0]
    return ret / var.n


@ti.func
def compute_potential_point(coord):
    potential_local = ti.sqr((target_coordinate - coord))
    return potential_local[0] + potential_local[1]


@ti.kernel
def compute_potential_grid():
    for i in range(grid_w):
        for j in range(grid_h):
            potential_grid[i, j][0] = world_scale_coeff * compute_potential_point(coords_grid[i, j])


@ti.kernel
def compute_potential_grad_field():
    # https://numpy.org/doc/stable/reference/generated/numpy.gradient.html?highlight=gradient#numpy.gradient
    for i in range(1, grid_w - 1):
        for j in range(1, grid_h - 1):
            hx = coords_grid[i, j][1] - coords_grid[i - 1, j][1]
            hy = coords_grid[i, j][0] - coords_grid[i, j - 1][0]
            potential_gradient_grid[i, j][0] = (
                potential_grid[i + 1, j][0] - potential_grid[i - 1, j][0]
            ) / (2 * hx)
            potential_gradient_grid[i, j][1] = (
                potential_grid[i, j + 1][0] - potential_grid[i, j - 1][0]
            ) / (2 * hy)


@ti.kernel
def find_cell(t: ti.i32,):
    # TODO find some other way to find the cell id by coordinate
    diff = 1e5
    for i in range(grid_w):
        for j in range(grid_h):
            diff_with_cell = coordinate[t, 0] - coords_grid[i, j]
            if diff_with_cell[0] + diff_with_cell[1] < diff:
                diff = diff_with_cell[0] + diff_with_cell[1]
                idx[None][0] = i
                idx[None][1] = j


@ti.func
def cumpute_l2_force():
    """Computes force produced by L2 potential

    Args:
        t (int): time id

    Returns:
        float: the amount of force produced by L2 potential
    """
    # TODO: compute force only by sampling the potential and calculating the derivative
    print(idx[None][0])
    print(idx[None][1])
    return potential_gradient_grid[idx[None][0], idx[None][1]]


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
    l2_force = cumpute_l2_force()
    friction_force = compute_rolling_friction_force(t,)
    print(l2_force[0])
    print(friction_force[0])
    acceleration[t, 0] = (world_scale_coeff * l2_force + friction_force) / mass

    v[t, 0] = v[t - 1, 0] + acceleration[t, 0] * dt
    coordinate[t, 0] = coordinate[t - 1, 0] + v[t, 0] * dt


def run_simulation():
    compute_potential_grid()
    compute_potential_grad_field()
    print(potential_gradient_grid[100, 100][0])
    print(potential_grid[99, 99][0])
    print(potential_grid[101, 101][0])
    for t in range(1, sim_steps):
        find_cell(t - 1)
        sim_step(t)
        gui.clear(0x3C733F)

        gui.circle(target_coordinate[None], radius=5, color=0x00000)

        gui.circle(coordinate[t, 0], radius=10, color=0xF20530)

        gui.show()

        print(coordinate[t, 0][0], coordinate[t, 0][1])


@ti.layout
def place():
    ti.root.dense(ti.l, sim_steps).dense(ti.i, 1).place(coordinate, v, acceleration)
    ti.root.dense(ti.j, grid_w).dense(ti.k, grid_h).place(
        potential_gradient_grid, potential_grid, coords_grid
    )
    ti.root.place(target_coordinate, velocity_direction, idx)
    ti.root.place(dt, radius, g, f, ro, volume, mass)
    ti.root.place(potential)
    ti.root.lazy_grad()


if __name__ == "__main__":
    sim_steps = 4000
    max_time = 50
    world_scale_coeff = 10
    grid_w, grid_h = (128, 128)
    x_borders = (0, 1)
    y_borders = (0, 1)

    potential_gradient_grid = ti.Vector(2, dt=ti.f32)
    potential_grid = ti.Vector(1, dt=ti.f32)
    coords_grid = ti.Vector(2, dt=ti.f32)

    target_coordinate = ti.Vector(2, dt=ti.f32)
    velocity_direction = ti.Vector(2, dt=ti.f32)
    coordinate = ti.Vector(2, dt=ti.f32)
    v = ti.Vector(2, dt=ti.f32)
    acceleration = ti.Vector(2, dt=ti.f32)
    idx = ti.Vector(2, dt=ti.i32)

    dt = ti.var(dt=ti.f32)
    radius = ti.var(dt=ti.f32)
    g = ti.var(dt=ti.f32)
    f = ti.var(dt=ti.f32)
    ro = ti.var(dt=ti.f32)
    volume = ti.var(dt=ti.f32)
    mass = ti.var(dt=ti.f32)
    potential = ti.var(dt=ti.f32)

    x_c = np.linspace(*x_borders, grid_w)
    y_c = np.linspace(*y_borders, grid_h)
    grid = np.stack(np.meshgrid(x_c, y_c, indexing="xy"), 2)
    coords_grid.from_numpy(grid)

    radius[None] = constants["radius"]
    g[None] = constants["g"]
    f[None] = constants["f"]
    ro[None] = constants["ro"]
    volume[None] = constants["volume"]
    mass[None] = constants["mass"]

    coordinate[0, 0] = [0.2, 0.6]
    target_coordinate[None] = [0.5, 0.5]
    v[0, 0] = [0.1, 0.]
    acceleration[0, 0] = [0.0, 0.0]
    dt[None] = max_time / sim_steps
    run_simulation()
