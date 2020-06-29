__all__ = ["DirectedRangeSensingAgent"]

import numpy as onp

from .observations import SimpleRangeSensor
from doper.utils.connectors import input_to_pytorch


class DirectedRangeSensingAgent:
    def __init__(self, config: dict):
        self.config = config
        self.sensor = SimpleRangeSensor(
            self.config["sim"]["agent_params"]["distance_range"],
            self.config["sim"]["agent_params"]["angle_range"],
            self.config["sim"]["agent_params"]["angle_step"],
        )

    def get_observations(
        self,
        coordinate_init: onp.ndarray,
        velocity_init: onp.ndarray,
        heading_direction: onp.ndarray,
        scene_handler: object,
    ):
        """
        Returns an observation given the current agents state
        Args:
            coordinate_init: [batch_size, 2] numpy.ndarray with agents coordinates
            velocity_init: [batch_size, 2] numpy.ndarray with agents speeds
            heading_direction: [batch_size, 2] numpy.ndarray with agents heading direction
            scene_handler: scene handler objects, that contains scene in it

        Returns:
            torch.Tensor of observations, [batch_size, observation_len]

        """
        dist = onp.linalg.norm(
            coordinate_init - onp.array(self.config["sim"]["coordinate_target"]), axis=1
        ).reshape(-1, 1)
        direction = (coordinate_init - onp.array(self.config["sim"]["coordinate_target"])) / dist
        range_obs = [
            self.sensor.get_observation(
                position=coord, heading_direction=heading_direction, scene=scene_handler.jax_scene
            )
            for coord in coordinate_init
        ]
        range_obs = onp.stack(range_obs, axis=0)
        range_obs /= self.config["sim"]["agent_params"]["distance_range"]
        return input_to_pytorch([dist, direction, coordinate_init, velocity_init, range_obs])
