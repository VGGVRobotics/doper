"""
All the equations below assumes that the agent is a ball with radius R, and the potential force
is being produced only by L2 distance between current agents location and target point.
The presence of rolling friction is also assumed
"""

import taichi as ti
import numpy as np
import matplotlib.pyplot as plt


ti.init(arch=ti.cpu, default_fp=ti.f32)
gui = ti.GUI("ball", (256, 256))

constants = {
    "radius": 0.05,
    "g": 9.8,
    "f": 0.007,
    "ro": 1000,
}
constants["volume"] = 4 * np.pi * (constants["radius"] ** 3) / 3
constants["mass"] = constants["volume"] * constants["ro"]


@ti.func
def compute_potential_point(coord):
    potential_local = ti.sqr((target_coordinate - coord))
    return potential_local[0] + potential_local[1]


@ti.kernel
def compute_potential_grid():
    for i in range(grid_w):
        for j in range(grid_h):
            potential_grid[i, j][0] = compute_potential_point(coords_grid[i, j])


@ti.kernel
def compute_potential_grad_field():
    # https://numpy.org/doc/stable/reference/generated/numpy.gradient.html?highlight=gradient#numpy.gradient
    for i in range(1, grid_w - 1):
        for j in range(1, grid_h - 1):
            potential_gradient_grid[i, j][0] = (
                potential_grid[i + 1, j][0] - potential_grid[i - 1, j][0]
            ) / (2 * hx)
            potential_gradient_grid[i, j][1] = (
                potential_grid[i, j + 1][0] - potential_grid[i, j - 1][0]
            ) / (2 * hy)


@ti.kernel
def find_cell(t: ti.i32,):
    idx[None][0] = coordinate[t, 0][0] // hx
    idx[None][1] = coordinate[t, 0][1] // hy


@ti.func
def cumpute_l2_force():
    """Computes force produced by L2 potential

    Args:
        t (int): time id

    Returns:
        float: the amount of force produced by L2 potential
    """

    return -potential_gradient_grid[idx[None][0], idx[None][1]]


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
    if velocity_direction[None][0] != 0.:
        velocity_direction[None][0] /= ti.abs(velocity_direction[None][0])

    if velocity_direction[None][1] != 0.:
        velocity_direction[None][1] /= ti.abs(velocity_direction[None][1])

    return - velocity_direction * f * normal_force / radius


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
    acceleration[t, 0] = (world_scale_coeff * l2_force + friction_force) / mass

    v[t, 0] = v[t - 1, 0] + acceleration[t, 0] * dt
    coordinate[t, 0] = coordinate[t - 1, 0] + v[t, 0] * dt


def draw_potentials():
    pot_np = potential_grid.to_numpy().reshape(grid_w, grid_h) * world_scale_coeff
    pot_np = pot_np + np.abs(pot_np.min())
    plt.imsave('potential.jpg', pot_np / pot_np.max())
    pot_grad_np = potential_gradient_grid.to_numpy().reshape(grid_w, grid_h, 2) * world_scale_coeff
    pot_grad_np = pot_grad_np + np.abs(pot_grad_np.min())
    plt.imsave('potential_g0.jpg', pot_grad_np[:, :, 0] / pot_grad_np.max())
    plt.imsave('potential_g1.jpg', pot_grad_np[:, :, 1] / pot_grad_np.max())


def run_simulation():
    compute_potential_grid()
    compute_potential_grad_field()
    draw_potentials()
    for t in range(1, sim_steps):
        find_cell(t - 1)
        sim_step(t)
        gui.clear(0x3C733F)

        gui.circle(target_coordinate[None], radius=5, color=0x00000)

        gui.circle(coordinate[t, 0],
                   radius=int(constants["radius"] * world_scale_coeff * 10),
                   color=0xF20530)

        gui.show()

        print(coordinate[t, 0][0], coordinate[t, 0][1])


@ti.layout
def place():
    ti.root.dense(ti.l, sim_steps).dense(ti.i, 1).place(coordinate, v, acceleration)
    ti.root.dense(ti.j, grid_w).dense(ti.k, grid_h).place(
        potential_gradient_grid, potential_grid, coords_grid
    )
    ti.root.place(target_coordinate, velocity_direction, idx)
    ti.root.place(dt, radius, g, f, ro, volume, mass, hx, hy)
    ti.root.place(potential)
    ti.root.lazy_grad()


if __name__ == "__main__":
    sim_steps = 20000
    max_time = 40
    world_scale_coeff = 10
    grid_w, grid_h = (1024, 1024)
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
    hx = ti.var(dt=ti.f32)
    hy = ti.var(dt=ti.f32)

    x_c = np.linspace(*x_borders, grid_w)
    y_c = np.linspace(*y_borders, grid_h)
    grid = np.stack(np.meshgrid(x_c, y_c, indexing="xy"), 2)
    coords_grid.from_numpy(grid)
    hx[None] = np.abs(x_c[1] - x_c[0])
    hy[None] = np.abs(y_c[1] - y_c[0])

    radius[None] = constants["radius"]
    g[None] = constants["g"]
    f[None] = constants["f"]
    ro[None] = constants["ro"]
    volume[None] = constants["volume"]
    mass[None] = constants["mass"]

    coordinate[0, 0] = [0.2, 0.5]
    target_coordinate[None] = [0.5, 0.5]
    v[0, 0] = [0.0, 1.]
    acceleration[0, 0] = [0.0, 0.0]
    dt[None] = max_time / sim_steps
    run_simulation()
