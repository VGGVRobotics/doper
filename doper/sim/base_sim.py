from typing import Tuple
import os

import numpy as np
import taichi as ti
import matplotlib.pyplot as plt


@ti.data_oriented
class BaseSim:
    def __init__(self, grid_resolution: Tuple[int], output_folder: os.PathLike):
        """Base simulation class. Capable of creating grid of potential and its gradient
        Gradient computation is numerical

        Args:
            grid_resolution (Tuple[int]): Width and height resolution for potential and gradient
            output_folder (os.PathLike): Output folder
        """
        self.output_folder = output_folder
        os.makedirs(output_folder, exist_ok=True)

        self.potential_gradient_grid = ti.Vector(2, dt=ti.f32)
        self.potential_grid = ti.Vector(1, dt=ti.f32)
        self.coords_grid = ti.Vector(2, dt=ti.f32)
        self.obstacle_grid = ti.Vector(1, dt=ti.i32)

        self.target_coordinate = ti.Vector(2, dt=ti.f32)
        self.velocity_direction = ti.Vector(2, dt=ti.f32)
        self.coordinate = ti.Vector(2, dt=ti.f32)
        self.v = ti.Vector(2, dt=ti.f32)
        self.acceleration = ti.Vector(2, dt=ti.f32)
        self.idx = ti.Vector(2, dt=ti.i32)

        self.hx = ti.var(dt=ti.f32)
        self.hy = ti.var(dt=ti.f32)

        self.grid_w, self.grid_h = grid_resolution

        ti.root.dense(ti.j, self.grid_w).dense(ti.k, self.grid_h).place(
            self.potential_gradient_grid, self.potential_grid, self.coords_grid, self.obstacle_grid
        )

    @ti.func
    def compute_potential_point(self):
        """Function should compute potential value given the coorditane
        """
        raise NotImplementedError

    @ti.kernel
    def compute_potential_grid(self):
        """Kernel iterates though all the cells in the grid, stores the potential value
        """
        for i in range(self.grid_w):
            for j in range(self.grid_h):
                self.potential_grid[i, j][0] = self.compute_potential_point(self.coords_grid[i, j])

    @ti.kernel
    def compute_potential_grad_grid(self):
        """Computes gradient grid from the potential grid, generated with compute_potential_grid function
        """
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
    def find_cell(
        self, t: ti.i32,
    ):
        """Stores the id of the cell the agent is in in the time id t

        Args:
            t (ti.i32): time id
        """
        self.idx[None][0] = self.coordinate[t, 0][0] // self.hx
        self.idx[None][1] = self.coordinate[t, 0][1] // self.hy

    @ti.kernel
    def compute_obstacle_grid(self):
        for i in range(self.grid_w):
            for j in range(self.grid_h):
                if (i == 0) | (j == 0) | (i == self.grid_w - 1) | (j == self.grid_h - 1):
                    self.obstacle_grid[i, j][0] = 1

    @ti.kernel
    def sim_step(
        self, t: ti.i32,
    ):
        """Makes one step of the simulation

        Args:
            t (ti.i32): time id

        """
        raise NotImplementedError

    def draw_potentials(self):
        """Saves images of the potential and x and y derivatives
        """
        pot_np = self.potential_grid.to_numpy().reshape(self.grid_w, self.grid_h)
        pot_np = pot_np + np.abs(pot_np.min())
        plt.imsave(os.path.join(self.output_folder, "potential.jpg"), pot_np / pot_np.max())
        pot_grad_np = self.potential_gradient_grid.to_numpy().reshape(self.grid_w, self.grid_h, 2)
        pot_grad_np = pot_grad_np + np.abs(pot_grad_np.min())
        plt.imsave(
            os.path.join(self.output_folder, "potential_g0.jpg"),
            pot_grad_np[:, :, 0] / pot_grad_np.max(),
        )
        plt.imsave(
            os.path.join(self.output_folder, "potential_g1.jpg"),
            pot_grad_np[:, :, 1] / pot_grad_np.max(),
        )

        plt.imsave(
            os.path.join(self.output_folder, "obstacles.jpg"),
            self.obstacle_grid.to_numpy().reshape(self.grid_w, self.grid_h),
        )

    def run_simulation(self):
        """Function used to run the simulation
        """
        raise NotImplementedError
