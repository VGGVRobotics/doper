__all__ = ["RangeSensingAgent"]

import numpy as onp

from doper.world.observations import UndirectedRangeSensor
from doper.utils.connectors import input_to_pytorch


class RangeSensingAgent:
    def __init__(self, config):
        self.config = config
        self.sensor = UndirectedRangeSensor(
            self.config["sim"]["agent_params"]["distance_range"],
            self.config["sim"]["agent_params"]["angle_step"],
        )

    def get_observations(self, coordinate_init, velocity_init, scene_handler):
        dist = onp.linalg.norm(
            coordinate_init - onp.array(self.config["sim"]["coordinate_target"]), axis=1
        ).reshape(-1, 1)
        direction = (coordinate_init - onp.array(self.config["sim"]["coordinate_target"])) / dist
        range_obs = [
            self.sensor.get_observation(position=coord, scene=scene_handler.scene) for coord in coordinate_init
        ]
        range_obs = onp.stack(range_obs, axis=0)
        range_obs /= self.config["sim"]["agent_params"]["distance_range"]
        return input_to_pytorch([dist, direction, coordinate_init, velocity_init, range_obs])
