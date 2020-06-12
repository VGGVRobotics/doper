from typing import Tuple, Union, Optional
import numpy as onp
from doper.sim.jax_geometry import JaxScene, batch_line_ray_intersection_point


class Sensor:
    """Dummy sensor class to use in typehints
    """

    def get_observation(self) -> None:
        """Returns sensor's observation
        """
        pass


class SimpleRangeSensor(Sensor):
    def __init__(self, distance_range: float, angle_range: float, angle_step: float) -> None:
        """Simple range sensor.

        Args:
            distance_range (float): max distance
            angle_range (float): field of view of the sensor
            angle_step (float): ray angular step
        """
        self._distance_range = distance_range
        self._angle_range_rad = angle_range / 180 * onp.pi
        self._angle_step_rad = angle_step / 180 * onp.pi

    def get_observation(
        self,
        position: Union[onp.ndarray, Tuple[float, float]],
        heading_direction: Union[onp.ndarray, Tuple[float, float]],
        scene: JaxScene,
        return_intersection_points: Optional[bool] = False,
    ) -> Union[onp.ndarray, Tuple[onp.ndarray, onp.ndarray]]:
        """Returns sensor's observation

        Args:
            position (Union[onp.ndarray, Tuple[float, float]]): current sensor position in world coordinates.
            heading_direction (Union[nop.ndarray, Tuple[float, float]]): sensor heading direction.
                Corresponds to midpoint of FOV.
            scene (Scene): current scene
            return_intersection_points (Optional[bool], optional): If True returns both intersection
                points and ranges. Defaults to False.

        Returns:
            Union[onp.ndarray, Tuple[onp.ndarray, onp.ndarray]]: If return_intesection_points is False,
                returns array of ranges, else returns tuple (range, intersection_points)
        """
        position = onp.array(position)
        heading_angle = onp.arctan2(heading_direction[1], heading_direction[0])
        ray_angles = onp.arange(
            heading_angle - self._angle_range_rad / 2,
            heading_angle + self._angle_range_rad / 2,
            self._angle_step_rad,
        )
        ray_directions = onp.concatenate(
            [onp.cos(ray_angles)[..., onp.newaxis], onp.sin(ray_angles)[..., onp.newaxis]], axis=-1
        )
        segments = scene.segments
        ray_origins = position.reshape(1, -1).repeat(len(ray_directions), axis=0)
        intersection_points = batch_line_ray_intersection_point(
            ray_origins, ray_directions, segments
        )
        intersection_points = onp.array(intersection_points)
        ray_idx = onp.arange(len(ray_directions))
        ranges = onp.linalg.norm(intersection_points - position.reshape(1, 1, -1), axis=-1)
        if return_intersection_points:

            closest = onp.argmin(ranges, axis=-1)
            points = intersection_points[ray_idx, closest, :]
            ranges = ranges[ray_idx, closest]
            points[ranges > self._distance_range] = onp.array([onp.inf, onp.inf])
            ranges[ranges > self._distance_range] = self._distance_range
            return ranges, points
        ranges = ranges.min(axis=-1)
        ranges[ranges > self._distance_range] = self._distance_range

        return ranges


class UndirectedRangeSensor(SimpleRangeSensor):
    def __init__(self, distance_range: float, angle_step: float) -> None:
        """Omnidirectional range sensor.

        Args:
            distance_range (float): max distance
            angle_step (float): ray angular step
        """
        super().__init__(distance_range=distance_range, angle_range=360, angle_step=angle_step)

    def get_observation(
        self,
        position: Union[onp.ndarray, Tuple[float, float]],
        scene: JaxScene,
        return_intersection_points: Optional[bool] = False,
    ) -> Union[onp.ndarray, Tuple[onp.ndarray, onp.ndarray]]:
        """Returns sensor's observation

        Args:
            position (Union[onp.ndarray, Tuple[float, float]]): current sensor position in world coordinates.
            scene (Scene): current scene
            return_intersection_points (Optional[bool], optional): If True returns both intersection
                points and ranges. Defaults to False.

        Returns:
            Union[onp.ndarray, Tuple[onp.ndarray, onp.ndarray]]: If return_intesection_points is False,
                returns array of ranges, else returns tuple (range, intersection_points)
        """
        return super().get_observation(position, (1, 0), scene, return_intersection_points)
