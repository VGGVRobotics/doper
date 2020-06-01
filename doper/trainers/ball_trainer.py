__all__ = ["BallAttractorTrainer"]

import torch
import torch.nn as nn
import jax.numpy as np
from jax import value_and_grad
import numpy as onp
from matplotlib import collections as mc
import matplotlib.pyplot as plt

from doper.networks import LinearReLUController
from doper.sim.ball_sim import compute_loss, run_sim
from doper.world.assets import get_svg_scene
from doper.sim.jax_geometry import JaxScene
from doper.utils.connectors import input_to_pytorch, jax_grads_to_pytorch


class BallAttractorTrainer:
    def __init__(self, config):
        super().__init__(self)
        self.config = config
        self.constants = self.config["constants"]
        self.constants["volume"] = 4 * np.pi * (self.constants["radius"] ** 3) / 3
        self.constants["mass"] = self.constants["volume"] * self.constants["density"]
        self.controller = LinearReLUController(config["model"])
        self.scene = get_svg_scene("../assets/simple_level.svg", px_per_meter=100)
        self.jax_scene = JaxScene(segments=np.array(self.scene.get_all_segments()))
        self.sim_time = config["sim"]["sim_time"]
        self.n_steps = config["sim"]["n_steps"]
        self.optimizer = torch.optim.Adam(self.controller.parameters())

    def forward(self, observation, coordinate_init):
        velocity_init = self.controller(observation)
        final_coordinate, trajectory = run_sim(
            self.sim_time,
            self.n_steps,
            self.jax_scene,
            coordinate_init,
            velocity_init,
            self.config["sim"]["attractor_coordinate"],
            self.config["sim"]["attractor_strength"],
            self.constants,
        )
        return velocity_init, trajectory

    def optimize_parameters(self, observation, coordinate_init):
        velocity_init = self.controller(observation)
        loss_val, v_grad = value_and_grad(compute_loss, 4)(
            self.sim_time,
            self.n_steps,
            self.jax_scene,
            coordinate_init,
            velocity_init,
            self.config["sim"]["coordinate_target"],
            self.config["sim"]["attractor_coordinate"],
            self.config["sim"]["attractor_strength"],
            self.constants,
        )
        pytorch_grad = jax_grads_to_pytorch(v_grad)[None, :]
        self.optimizer.zero_grad()
        velocity_init.backward(pytorch_grad)
        self.optimizer.step()

    def get_observations(self, coordinate_init):
        dist = onp.linalg.norm(coordinate_init - self.config["sim"]["coordinate_target"])
        direction = (coordinate_init - self.config["sim"]["coordinate_target"]) / dist
        sin_direction = onp.sin(direction)
        return input_to_pytorch([dist, direction, sin_direction])

    def write_output(self, trajectory, coordinate_init):
        lines = mc.LineCollection(self.scene.get_all_segments())
        fig, ax = plt.subplots()

        ax.add_collection(lines)
        traj = onp.array(trajectory.coordinate)
        ax.scatter(coordinate_init[0], coordinate_init[1], c="g")
        ax.plot(traj[:, 0], traj[:, 1])
        ax.scatter(coordinate_init[0], coordinate_init[1], c="g", label="init")
        ax.scatter(
            self.config["sim"]["attractor_coordinate"][0],
            self.config["sim"]["attractor_coordinate"][1],
            c="b",
            label="attractor",
        )
        ax.scatter(
            self.config["sim"]["coordinate_target"][0],
            self.config["sim"]["coordinate_target"][1],
            c="r",
            label="target",
        )
        fig.legend()
        plt.savefig(f"trajectory.jpg")
