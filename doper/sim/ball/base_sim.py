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
class BaseSim:
    def __init__(self, grid_resolution):
        self.potential_gradient_grid = ti.Vector(2, dt=ti.f32)
        self.potential_grid = ti.Vector(1, dt=ti.f32)
        self.coords_grid = ti.Vector(2, dt=ti.f32)

        self.target_coordinate = ti.Vector(2, dt=ti.f32)
        self.velocity_direction = ti.Vector(2, dt=ti.f32)
        self.coordinate = ti.Vector(2, dt=ti.f32)
        self.v = ti.Vector(2, dt=ti.f32)
        self.acceleration = ti.Vector(2, dt=ti.f32)
        self.idx = ti.Vector(2, dt=ti.i32)

        self.hx = ti.var(dt=ti.f32)
        self.hy = ti.var(dt=ti.f32)

        self.grid_w, self.grid_h = grid_resolution


    @ti.func
    def compute_potential_point(self, coord):
        raise NotImplementedError

    @ti.kernel
    def compute_potential_grid(self,):
        for i in range(self.grid_w):
            for j in range(self.grid_h):
                self.potential_grid[i, j][0] = self.compute_potential_point(self.coords_grid[i, j])


    @ti.kernel
    def compute_potential_grad_grid(self):
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


    @ti.kernel
    def sim_step(self, t: ti.i32,):
        """Makes one step of the simulation

        Args:
            t (int): time id

        """
        raise NotImplementedError


    def draw_potentials(self):
        pot_np = self.potential_grid.to_numpy().reshape(self.grid_w, self.grid_h)
        pot_np = pot_np + np.abs(pot_np.min())
        plt.imsave('potential.jpg', pot_np / pot_np.max())
        pot_grad_np = self.potential_gradient_grid.to_numpy().reshape(self.grid_w, self.grid_h, 2)
        pot_grad_np = pot_grad_np + np.abs(pot_grad_np.min())
        plt.imsave('potential_g0.jpg', pot_grad_np[:, :, 0] / pot_grad_np.max())
        plt.imsave('potential_g1.jpg', pot_grad_np[:, :, 1] / pot_grad_np.max())


    def run_simulation(self,):
        raise NotImplementedError
