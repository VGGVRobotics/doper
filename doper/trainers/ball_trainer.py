__all__ = ["BallAttractorTrainer"]

import logging

import jax.numpy as np
import numpy as onp
import torch

from doper import networks
from doper.sim.ball_sim import vmapped_grad_and_value, run_sim
from doper.utils.connectors import jax_grads_to_pytorch, pytorch_to_jax

logger = logging.getLogger(__name__)


class BallAttractorTrainer:
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = self.config["train"]["device"]
        self.constants = self.config["constants"]
        self.constants["volume"] = 4 * np.pi * (self.constants["radius"] ** 3) / 3
        self.constants["mass"] = self.constants["volume"] * self.constants["density"]
        self.controller = getattr(networks, config["model"]["controller_name"])(config["model"]).to(config["train"]["device"])
        self.sim_time = config["sim"]["sim_time"]
        self.n_steps = config["sim"]["n_steps"]
        self.optimizer = torch.optim.Adam(self.controller.parameters())
        self.batch_size = self.config["train"]["batch_size"]
        self.num_actions = self.config["train"]["num_actions"]

    def forward(self, observation, coordinate_init, velocity_init, direction_init, scene):
        impulse = self.controller(observation.to(self.device))
        final_coordinate, final_velocity, trajectory = run_sim(
            self.sim_time,
            self.n_steps,
            scene.jax_scene,
            coordinate_init[0],
            pytorch_to_jax(
                (
                    torch.from_numpy(onp.array(velocity_init))
                    + impulse.cpu() / self.constants["mass"]
                )[0]
            ),
            direction_init,
            np.array(self.config["sim"]["attractor_coordinate"]),
            self.constants,
        )
        return final_coordinate, final_velocity, trajectory

    def optimize_parameters(self, agent, scene_handler, grad_clip=5):
        coordinate_init = scene_handler.get_init_state(self.batch_size)
        direction_init = onp.ones_like(coordinate_init)
        direction_init /= onp.linalg.norm(direction_init, axis=1)[:, None]

        self.optimizer.zero_grad()
        velocity_init = onp.random.uniform(-1, 2, (self.batch_size, 2))
        for action_id in range(self.num_actions):
            observation = agent.get_observations(coordinate_init, velocity_init, direction_init, scene_handler)

            impulse = self.controller(observation.to(self.device))

            (loss_val, (coord, velocity)), v_grad = vmapped_grad_and_value(
                self.sim_time,
                self.n_steps,
                scene_handler.jax_scene,
                coordinate_init,
                pytorch_to_jax(
                    torch.from_numpy(velocity_init) + impulse.cpu() / self.constants["mass"]
                ),
                np.array(self.config["sim"]["direction_init"]),
                np.array(self.config["sim"]["coordinate_target"]),
                np.array(self.config["sim"]["attractor_coordinate"]),
                self.constants,
            )
            pytorch_grad = jax_grads_to_pytorch(v_grad).to(self.device)
            impulse.backward(pytorch_grad)  # norm?
            coordinate_init = coord
            velocity_init = velocity_init
        torch.nn.utils.clip_grad_norm_(self.controller.parameters(), grad_clip)
        self.optimizer.step()
        return loss_val
