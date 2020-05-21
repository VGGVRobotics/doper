from typing import List, Dict, Tuple, Union
import os
from time import time

import numpy as np
import taichi as ti
import matplotlib.pyplot as plt

from .base_sim import BaseSim


@ti.data_oriented
class RollingBallSim(BaseSim):
    def __init__(
        self,
        constants: Dict[str, Union[float, int, str]],
        sim_steps: int,
        max_time: int,
        world_scale_coeff: Union[int, float],
        grid_resolution: Tuple[int],
        gui: ti.misc.gui.GUI,
        output_folder: os.PathLike,
    ):
        """Simultaion of the ball in the potential field.
        All the equations below assumes that the agent is a ball with radius R, and the potential force
        is being produced only by L2 distance between current agents location and target point.
        The presence of rolling friction is also assumed


        Args:
            constants (Dict[str, Union[Union[float, int], str]]): dict with basic physical constants
            sim_steps (int): total number of steps in the simulation
            max_time (int): the length of the simulation, seconds
            world_scale_coeff (Union[int, float]): the taichi gui supports only [0, 1],
                                                   so this is used to 'scale' the world
            grid_resolution (Tuple[int]): number of cells accross each axis
            gui (ti.gui): taichi gui
            output_folder (os.PathLike): Output folder, used for visualisation
        """
        super().__init__(grid_resolution, output_folder)
        self.sim_steps = sim_steps
        self.max_time = max_time
        self.gui = gui
        self.world_scale_coeff = world_scale_coeff
        self.constants = constants
        self.grid_w, self.grid_h = grid_resolution
        x_borders = (0, 1)
        y_borders = (0, 1)

        self.dt = ti.var(dt=ti.f32)
        self.radius = ti.var(dt=ti.f32)
        self.g = ti.var(dt=ti.f32)
        self.f = ti.var(dt=ti.f32)
        self.ro = ti.var(dt=ti.f32)
        self.volume = ti.var(dt=ti.f32)
        self.mass = ti.var(dt=ti.f32)
        self.potential = ti.var(dt=ti.f32)
        self.elasticity = ti.var(dt=ti.f32)

        ti.root.dense(ti.l, self.sim_steps).dense(ti.i, 1).place(
            self.coordinate, self.v, self.acceleration
        )
        ti.root.place(self.target_coordinate, self.velocity_direction, self.idx)
        ti.root.place(
            self.dt,
            self.radius,
            self.g,
            self.f,
            self.ro,
            self.volume,
            self.mass,
            self.hx,
            self.hy,
            self.elasticity,
        )
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

        self.elasticity[None] = 0.99

    @ti.func
    def compute_potential_point(self, coord: ti.f32):
        """Computes the potential, defined as L2 distance between
        the current coordinate and target poing

        Args:
            coord (ti.f32): current coordinate

        Returns:
            ti.f32: value of the potential
        """
        potential_local = (self.target_coordinate - coord) ** 2
        return potential_local[0] + potential_local[1]

    @ti.func
    def compute_obstacle_direction(
        self, t: ti.i32, i: int, j: int,
    ):
        # TODO create a structure with obstacle cells coordinates, retrieve closest
        if self.obstacle_grid[i, j][0] == 1:
            obstacle_direction = self.coordinate[t - 1, 0] - self.coords_grid[i, j]
            return obstacle_direction

    @ti.func
    def compute_l2_force(self):
        """Computes force produced by L2 potential

        Returns:
            ti.f32: the amount of force produced by L2 potential
        """

        return -self.potential_gradient_grid[self.idx[None][0], self.idx[None][1]]

    @ti.func
    def compute_rolling_friction_force(
        self, t: ti.f32,
    ):
        """Computes rolling friction force value, flat land assumed

        Args:
            t (ti.i32): time id

        Returns:
            flti.f32oat: the amount of the rolling friction force
        """
        normal_force = self.mass * self.g

        self.velocity_direction[None] = self.v[t - 1, 0]
        if self.velocity_direction[None][0] != 0.0:
            self.velocity_direction[None][0] /= ti.abs(self.velocity_direction[None][0])

        if self.velocity_direction[None][1] != 0.0:
            self.velocity_direction[None][1] /= ti.abs(self.velocity_direction[None][1])

        return -self.velocity_direction * self.f * normal_force / self.radius

    @ti.kernel
    def sim_step(
        self, t: ti.i32,
    ):
        """Makes one step of the simulation

        Args:
            t (ti.i32): time id
        """

        min_dist_norm = 100000.0
        closest_direction = ti.Vector([0.0, 0.0])
        projected_v_n = ti.Vector([0.0, 0.0])
        for i in range(self.grid_w):
            for j in range(self.grid_h):
                obstacle_direction = self.compute_obstacle_direction(t, i, j)
                dist_norm = obstacle_direction.norm()
                if dist_norm < min_dist_norm:
                    min_dist_norm = dist_norm
                    closest_direction = obstacle_direction / dist_norm

        if min_dist_norm < self.radius:
            projected_v_n = closest_direction * (closest_direction.dot(self.v[t - 1, 0]))
            projected_v_p = self.v[t - 1, 0] - projected_v_n
            self.v[t - 1, 0] = projected_v_p - self.elasticity * projected_v_n

        l2_force = self.compute_l2_force()
        friction_force = self.compute_rolling_friction_force(t,)
        self.acceleration[t, 0] = (self.world_scale_coeff * l2_force + friction_force) / self.mass
        self.v[t, 0] = self.v[t - 1, 0] + self.acceleration[t, 0] * self.dt
        self.coordinate[t, 0] = self.coordinate[t - 1, 0] + self.v[t, 0] * self.dt

    def run_simulation(
        self,
        initial_coordinate: Tuple[float, float],
        attraction_coordinate: Tuple[float, float],
        initial_speed: Tuple[float, float],
        visualize: bool = True,
    ):
        """Runs simulation

        Args:
            initial_coordinate (Tuple[float, float]):
                [x, y] starting point for the ball
            attraction_coordinate (Tuple[float, float]):
                [x, y] target point, L2 is being computed with it
            initial_speed (Tuple[float, float]):
                [vx, vy] initial speed of the ball
            visualize (bool): show GUI
        """
        self.coordinate[0, 0] = initial_coordinate
        self.target_coordinate[None] = attraction_coordinate
        self.v[0, 0] = initial_speed
        self.acceleration[0, 0] = [0.0, 0.0]
        start = time()
        self.compute_potential_grid()
        self.compute_potential_grad_grid()
        self.compute_obstacle_grid()
        print(f"initialization time {time() - start}")
        self.draw_potentials()
        for t in range(1, self.sim_steps):
            start = time()
            self.find_cell(t - 1)
            self.sim_step(t)
            print(f"sim_step time {time() - start}")
            if visualize:
                self.gui.clear(0x3C733F)

                self.gui.circle(self.target_coordinate[None], radius=5, color=0x00000)

                self.gui.circle(
                    self.coordinate[t, 0],
                    radius=int(self.constants["radius"] * self.world_scale_coeff * 10),
                    color=0xF20530,
                )

                self.gui.show()

            print(self.coordinate[t, 0][0], self.coordinate[t, 0][1])
