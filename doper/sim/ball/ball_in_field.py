"""
All the equations below assumes that the agent is a ball with radius R, and the potential force
is being produced only by L2 distance between current agents location and target point.
The presence of rolling friction is also assumed
"""
from typing import List, Dict, Tuple, Union

import numpy as np
import taichi as ti
import matplotlib.pyplot as plt


@ti.data_oriented
class RollingBallSim:
    def __init__(self,
                 constants: Dict[str, Union[Union[float, int], str]],
                 sim_steps: int,
                 max_time: int,
                 world_scale_coeff: Union[int, float],
                 grid_resolution: Tuple[int],
                 gui):
        self.sim_steps = sim_steps
        self.max_time = max_time
        self.gui = gui
        self.world_scale_coeff = world_scale_coeff
        self.constants = constants
        self.grid_w, self.grid_h = grid_resolution
        x_borders = (0, 1)
        y_borders = (0, 1)

        self.potential_gradient_grid = ti.Vector(2, dt=ti.f32)
        self.potential_grid = ti.Vector(1, dt=ti.f32)
        self.coords_grid = ti.Vector(2, dt=ti.f32)

        self.target_coordinate = ti.Vector(2, dt=ti.f32)
        self.velocity_direction = ti.Vector(2, dt=ti.f32)
        self.coordinate = ti.Vector(2, dt=ti.f32)
        self.v = ti.Vector(2, dt=ti.f32)
        self.acceleration = ti.Vector(2, dt=ti.f32)
        self.idx = ti.Vector(2, dt=ti.i32)

        self.dt = ti.var(dt=ti.f32)
        self.radius = ti.var(dt=ti.f32)
        self.g = ti.var(dt=ti.f32)
        self.f = ti.var(dt=ti.f32)
        self.ro = ti.var(dt=ti.f32)
        self.volume = ti.var(dt=ti.f32)
        self.mass = ti.var(dt=ti.f32)
        self.potential = ti.var(dt=ti.f32)
        self.hx = ti.var(dt=ti.f32)
        self.hy = ti.var(dt=ti.f32)

        ti.root.dense(ti.l, self.sim_steps).dense(ti.i, 1).place(self.coordinate,
                                                                 self.v,
                                                                 self.acceleration)
        ti.root.dense(ti.j, self.grid_w).dense(ti.k, self.grid_h).place(
            self.potential_gradient_grid, self.potential_grid, self.coords_grid
        )
        ti.root.place(self.target_coordinate, self.velocity_direction, self.idx)
        ti.root.place(self.dt, self.radius, self.g, self.f, self.ro, self.volume, self.mass, self.hx, self.hy)
        ti.root.place(self.potential)
        ti.root.lazy_grad()

        x_c = np.linspace(*x_borders, self.grid_w)
        y_c = np.linspace(*y_borders, self.grid_h)
        grid = np.stack(np.meshgrid(x_c, y_c, indexing="xy"), 2)
        self.coords_grid.from_numpy(grid)
        self.hx[None] = np.abs(x_c[1] - x_c[0])
        self.hy[None] = np.abs(y_c[1] - y_c[0])

        self.radius[None] = self.constants["radius"]
        self.g[None] = self.constants["g"]
        self.f[None] = self.constants["f"]
        self.ro[None] = self.constants["ro"]
        self.volume[None] = self.constants["volume"]
        self.mass[None] = self.constants["mass"]

        self.dt[None] = self.max_time / self.sim_steps


    @ti.func
    def compute_potential_point(self, coord):
        potential_local = ti.sqr((self.target_coordinate - coord))
        return potential_local[0] + potential_local[1]


    @ti.kernel
    def compute_potential_grid(self,):
        for i in range(self.grid_w):
            for j in range(self.grid_h):
                self.potential_grid[i, j][0] = self.compute_potential_point(self.coords_grid[i, j])


    @ti.kernel
    def compute_potential_grad_field(self):
        # https://numpy.org/doc/stable/reference/generated/numpy.gradient.html?highlight=gradient#numpy.gradient
        for i in range(1, self.grid_w - 1):
            for j in range(1, self.grid_h - 1):
                self.potential_gradient_grid[i, j][0] = (
                    self.potential_grid[i + 1, j][0] - self.potential_grid[i - 1, j][0]
                ) / (2 * self.hx)
                self.potential_gradient_grid[i, j][1] = (
                    self.potential_grid[i, j + 1][0] - self.potential_grid[i, j - 1][0]
                ) / (2 * self.hy)


    @ti.kernel
    def find_cell(self, t: ti.i32,):
        self.idx[None][0] = self.coordinate[t, 0][0] // self.hx
        self.idx[None][1] = self.coordinate[t, 0][1] // self.hy


    @ti.func
    def compute_l2_force(self):
        """Computes force produced by L2 potential

        Args:
            t (int): time id

        Returns:
            float: the amount of force produced by L2 potential
        """

        return - self.potential_gradient_grid[self.idx[None][0], self.idx[None][1]]


    @ti.func
    def compute_rolling_friction_force(self, t,):
        """Computes rolling friction force value, flat land assumed

        Args:
            t (int): time id

        Returns:
            float: the amount of the rolling friction force
        """
        normal_force = self.mass * self.g

        self.velocity_direction[None] = self.v[t - 1, 0]
        if self.velocity_direction[None][0] != 0.:
            self.velocity_direction[None][0] /= ti.abs(self.velocity_direction[None][0])

        if self.velocity_direction[None][1] != 0.:
            self.velocity_direction[None][1] /= ti.abs(self.velocity_direction[None][1])

        return - self.velocity_direction * self.f * normal_force / self.radius


    @ti.kernel
    def sim_step(self, t: ti.i32,):
        """Makes one step of the simulation

        Args:
            t (int): time id

        Returns:
            list: [description]
        """
        l2_force = self.compute_l2_force()
        friction_force = self.compute_rolling_friction_force(t,)
        self.acceleration[t, 0] = (self.world_scale_coeff * l2_force + friction_force) / self.mass

        self.v[t, 0] = self.v[t - 1, 0] + self.acceleration[t, 0] * self.dt
        self.coordinate[t, 0] = self.coordinate[t - 1, 0] + self.v[t, 0] * self.dt


    def draw_potentials(self):
        pot_np = self.potential_grid.to_numpy().reshape(self.grid_w, self.grid_h)
        pot_np = pot_np + np.abs(pot_np.min())
        plt.imsave('potential.jpg', pot_np / pot_np.max())
        pot_grad_np = self.potential_gradient_grid.to_numpy().reshape(self.grid_w, self.grid_h, 2)
        pot_grad_np = pot_grad_np + np.abs(pot_grad_np.min())
        plt.imsave('potential_g0.jpg', pot_grad_np[:, :, 0] / pot_grad_np.max())
        plt.imsave('potential_g1.jpg', pot_grad_np[:, :, 1] / pot_grad_np.max())


    def run_simulation(self,
                       initial_coordinate: Union[List[float], np.float32],
                       attraction_coordinate: List[float],
                       initial_speed: List[float]):
        self.coordinate[0, 0] = initial_coordinate
        self.target_coordinate[None] = attraction_coordinate
        self.v[0, 0] = initial_speed
        self.acceleration[0, 0] = [0.0, 0.0]
        self.compute_potential_grid()
        self.compute_potential_grad_field()
        self.draw_potentials()
        for t in range(1, self.sim_steps):
            self.find_cell(t - 1)
            self.sim_step(t)
            self.gui.clear(0x3C733F)

            self.gui.circle(self.target_coordinate[None], radius=5, color=0x00000)

            self.gui.circle(self.coordinate[t, 0],
                    radius=int(self.constants["radius"] * self.world_scale_coeff * 10),
                    color=0xF20530)

            self.gui.show()

            print(self.coordinate[t, 0][0], self.coordinate[t, 0][1])

