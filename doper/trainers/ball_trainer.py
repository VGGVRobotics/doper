__all__ = ["BallAttractorTrainer"]

import logging

import jax.numpy as np
import matplotlib.pyplot as plt
import numpy as onp
import torch
from jax import value_and_grad
from matplotlib import collections as mc

from doper.networks import LinearReLUController
from doper.sim.ball_sim import compute_loss, run_sim
from doper.sim.jax_geometry import JaxScene
from doper.utils.connectors import input_to_pytorch, jax_grads_to_pytorch, pytorch_to_jax
from doper.world.assets import get_svg_scene
from doper.world.observations import UndirectedRangeSensor

logger = logging.getLogger(__name__)


class BallAttractorTrainer:
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.constants = self.config["constants"]
        self.constants["volume"] = 4 * np.pi * (self.constants["radius"] ** 3) / 3
        self.constants["mass"] = self.constants["volume"] * self.constants["density"]
        self.controller = LinearReLUController(config["model"]).to(config['train']['device'])
        self.scene = get_svg_scene(config["sim"]["svg_scene_path"], px_per_meter=100)
        self.jax_scene = JaxScene(segments=np.array(self.scene.get_all_segments()))
        self.sim_time = config["sim"]["sim_time"]
        self.n_steps = config["sim"]["n_steps"]
        self.optimizer = torch.optim.Adam(self.controller.parameters())
        self.sensor = UndirectedRangeSensor(self.config["sim"]["distance_range"],
                                            self.config["sim"]["angle_step"])

    def forward(self, observation, coordinate_init):
        velocity_init = self.controller(observation)
        final_coordinate, trajectory = run_sim(
            self.sim_time,
            self.n_steps,
            self.jax_scene,
            coordinate_init,
            pytorch_to_jax(velocity_init[0]),
            np.array(self.config["sim"]["attractor_coordinate"]),
            np.array(self.config["sim"]["attractor_strength"]),
            self.constants,
        )
        return velocity_init, trajectory

    def optimize_parameters(self, observation, coordinate_init, grad_clip=5):
        velocity_init = self.controller(observation)
        loss_val, v_grad = value_and_grad(compute_loss, 4)(
            self.sim_time,
            self.n_steps,
            self.jax_scene,
            coordinate_init,
            pytorch_to_jax(velocity_init[0]),
            np.array(self.config["sim"]["coordinate_target"]),
            np.array(self.config["sim"]["attractor_coordinate"]),
            self.config["sim"]["attractor_strength"],
            self.constants,
        )
        logger.info(f'Gradients from simulation are {v_grad}')
        pytorch_grad = jax_grads_to_pytorch(np.clip(v_grad, -grad_clip, grad_clip))[None, :]
        self.optimizer.zero_grad()
        velocity_init.backward(pytorch_grad)
        self.optimizer.step()
        return loss_val

    def get_observations(self, coordinate_init):
        dist = onp.linalg.norm(coordinate_init - onp.array(self.config["sim"]["coordinate_target"]))
        direction = (coordinate_init - onp.array(self.config["sim"]["coordinate_target"])) / dist
        range_obs = self.sensor.get_observation(position=coordinate_init,
                                                scene=self.scene)
        range_obs /= self.config["sim"]["distance_range"]
        return input_to_pytorch([dist, direction, range_obs])

    def write_output(self, trajectory, output_file_name):
        lines = mc.LineCollection(self.scene.get_all_segments())
        fig, ax = plt.subplots()

        ax.add_collection(lines)
        traj = onp.array(trajectory.coordinate)
        ax.scatter(traj[0, 0], traj[0, 1], c="g", label="init")
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
        ax.plot(traj[:, 0], traj[:, 1], c="black", label="trajectory")
        fig.legend()
        plt.savefig(output_file_name)
        plt.close()

    @staticmethod
    def get_init_state():
        return np.array(onp.random.uniform(-1.0, 5.0, size=(2,)))
