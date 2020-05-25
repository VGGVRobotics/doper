from typing import List, Dict, Tuple, Union
import os
from time import time, sleep

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
        self.tmp_coordinate = ti.Vector(2, dt=ti.f32)
        self.tmp_velocity = ti.Vector(2, dt=ti.f32)
        self.tmp_acceleration = ti.Vector(2, dt=ti.f32)
        ti.root.place(self.tmp_coordinate, self.tmp_velocity, self.tmp_acceleration)
        ti.root.dense(ti.k, self.sim_steps).place(self.coordinate, self.velocity, self.acceleration)
        ti.root.place(self.target_coordinate)
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
        self.elasticity[None] = self.constants["obstacles_elasticity"]

        self.dt[None] = self.max_time / self.sim_steps

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
    def get_toi(self, old_coordinate: ti.f32, old_velocity: ti.f32, obstacle_coordinate: ti.f32):
        """Computes time of impact

        Args:
            old_coordinate (ti.f32): coordinate at step before impact
            old_velocity (ti.f32): impact velocity
            obstacle_coordinate (ti.f32): obstacle world coordinate

        Returns:
            (ti.f32): scalar time of impact
        """
        old_obstacle_vector = obstacle_coordinate - old_coordinate
        old_obstacle_distance = old_obstacle_vector.norm()
        old_obstacle_direction = old_obstacle_vector / old_obstacle_distance
        toi = abs(old_obstacle_distance - self.radius) / abs(
            old_obstacle_direction.dot(old_velocity)
        )
        # looks suspicious but works pretty ok
        toi = min(toi, self.dt)
        return toi

    @ti.func
    def collide(
        self, velocity_at_impact: ti.f32, obstacle_direction: ti.f32,
    ):
        """Computes velocity after collision

        Args:
            velocity_at_impact (ti.f32): impact velocity
            obstacle_direction (ti.f32): obstacle normal direction

        Returns:
            (ti.f32): new velocity vector
        """

        projected_v_n = obstacle_direction * obstacle_direction.dot(velocity_at_impact)
        projected_v_p = velocity_at_impact - projected_v_n
        new_velocity = projected_v_p - self.elasticity * projected_v_n
        return new_velocity

    @ti.func
    def compute_l2_force(self, idx: ti.i32):
        """Computes force produced by L2 potential

        Returns:
            ti.f32: the amount of force produced by L2 potential
        """
        return -self.potential_gradient_grid[idx[0], idx[1]]

    @ti.func
    def compute_rolling_friction_force(self, current_velocity: ti.f32):
        """Computes rolling friction force value, flat land assumed

        Args:
            current_velocity (ti.f32): current velocity vector  

        Returns:
            (ti.f32): rolling friction force vector
        """

        normal_force = self.mass * self.g
        velocity_direction = ti.Vector([0.0, 0.0])
        velocity_direction = current_velocity.copy()
        if velocity_direction[0] != 0.0:
            velocity_direction[0] /= ti.abs(velocity_direction[0])

        if velocity_direction[1] != 0.0:
            velocity_direction[1] /= ti.abs(velocity_direction[1])

        return -velocity_direction * self.f * normal_force / self.radius

    @ti.func
    def compute_acceleration(self, coordinate: ti.f32, velocity: ti.f32):
        """Computes current acceleration vector

        Args:
            coordinate (ti.f32): current coordinate
            velocity (ti.f32): current velocity

        Returns:
            (ti.f32): acceleration vector
        """
        l2_force = self.compute_l2_force(self.find_cell(coordinate))
        friction_force = self.compute_rolling_friction_force(velocity)
        return (self.world_scale_coeff * l2_force + friction_force) / self.mass

    @ti.func
    def get_closest_obstacle(self, coordinate: ti.f32):
        """Finds closest obstacle

        Args:
            coordinate (ti.f32): current coordinate

        Returns:
            (ti.f32): closest obstacle_coordinate
            (ti.f32): distance to the obstacle
        """
        min_dist_norm = float(np.inf)
        closest_coordinate = ti.Vector([0.0, 0.0])
        for i, j in self.obstacle_grid:
            if self.obstacle_grid[i, j][0] == 1:
                obstacle_direction = self.coords_grid[i, j] - coordinate
                dist_norm = obstacle_direction.norm()
                if dist_norm < min_dist_norm:
                    min_dist_norm = dist_norm
                    closest_coordinate = self.coords_grid[i, j]
        return closest_coordinate, min_dist_norm

    @ti.kernel
    def try_step(self, t: ti.i32):
        """Advances physics one step forward without collisions and saves results to tmp variables

        Args:
            t (ti.i32): current time step
        """
        self.tmp_acceleration = self.compute_acceleration(
            self.coordinate[t - 1], self.velocity[t - 1]
        )
        self.tmp_velocity = self.velocity[t - 1] + self.tmp_acceleration * self.dt
        self.tmp_coordinate = self.tmp_velocity * self.dt + self.coordinate[t - 1]

    @ti.kernel
    def resolve_collision(
        self, t: ti.i32,
    ):
        """Checks and resolves collisions if any, stores new coordinate, velocity, acceleration

        Args:
            t (ti.i32): time id
        """
        obstacle_coordinate, distance_to_obstacle = self.get_closest_obstacle(self.tmp_coordinate)
        if distance_to_obstacle <= self.radius:
            toi = self.get_toi(self.coordinate[t - 1], self.tmp_velocity, obstacle_coordinate)
            impact_coordinate = self.coordinate[t - 1] + self.tmp_velocity * toi
            obstacle_direction = obstacle_coordinate - self.coordinate[t - 1]
            obstacle_direction /= obstacle_direction.norm()
            velocity_after_collision = self.collide(self.tmp_velocity, obstacle_direction)
            self.acceleration[t] = self.compute_acceleration(
                impact_coordinate, velocity_after_collision
            )
            self.velocity[t] = velocity_after_collision + self.acceleration[t] * (self.dt - toi)
            self.coordinate[t] = impact_coordinate + self.velocity[t] * (self.dt - toi)
        else:
            self.coordinate[t] = self.tmp_coordinate
            self.velocity[t] = self.tmp_velocity
            self.acceleration[t] = self.tmp_acceleration

    def sim_step(self, t: int):
        """Performs simulation step

        Args:
            t (int): current time step
        """
        self.try_step(t)
        self.resolve_collision(t)

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
        self.coordinate[0] = initial_coordinate
        self.target_coordinate[None] = attraction_coordinate
        self.velocity[0] = initial_speed
        self.acceleration[0] = [0.0, 0.0]
        start = time()
        self.compute_potential_grid()
        self.compute_potential_grad_grid()
        self.compute_obstacle_grid()
        print(f"initialization time {time() - start}")
        self.draw_potentials()
        for t in range(1, self.sim_steps):
            start = time()
            self.sim_step(t)
            # print(f"sim_step time {time() - start}")
            if visualize:
                self.gui.clear(0x3C733F)

                self.gui.circle(self.target_coordinate[None], radius=5, color=0x00000)

                self.gui.circle(
                    self.coordinate[t],
                    radius=int(self.constants["radius"] * self.world_scale_coeff * 10),
                    color=0xF20530,
                )

                self.gui.show()

            print("coord: ", self.coordinate[t][0], self.coordinate[t][1])
