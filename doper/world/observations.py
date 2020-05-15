from typing import Tuple, Union, Optional
import numpy as np
from .scene import Scene
from .checks import batch_line_ray_intersection_point


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
        self._angle_range_rad = angle_range / 180 * np.pi
        self._angle_step_rad = angle_step / 180 * np.pi

    def get_observation(
        self,
        position: Union[np.ndarray, Tuple[float, float]],
        heading_direction: Union[np.ndarray, Tuple[float, float]],
        scene: Scene,
        return_intersection_points: Optional[bool] = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Returns sensor's observation

        Args:
            position (Union[np.ndarray, Tuple[float, float]]): current sensor position in world coordinates.
            heading_direction (Union[np.ndarray, Tuple[float, float]]): sensor heading direction.
                Corresponds to midpoint of FOV.
            scene (Scene): current scene
            return_intersection_points (Optional[bool], optional): If True returns both intersection
                points and ranges. Defaults to False.

        Returns:
            Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]: If return_intesection_points is False,
                returns array of ranges, else returns tuple (range, intersection_points)
        """
        position = np.array(position)
        heading_angle = np.arctan2(heading_direction[1], heading_direction[0])
        ray_angles = np.arange(
            heading_angle - self._angle_range_rad / 2,
            heading_angle + self._angle_range_rad / 2,
            self._angle_step_rad,
        )
        ray_directions = np.concatenate(
            [np.cos(ray_angles)[..., np.newaxis], np.sin(ray_angles)[..., np.newaxis]], axis=-1
        )
        segments = scene.get_polygons_segments(
            scene.get_polygons_in_radius(position, self._distance_range)
        )
        ray_origins = position.reshape(1, -1).repeat(len(ray_directions), axis=0)
        intersection_points = batch_line_ray_intersection_point(
            ray_origins, ray_directions, segments
        )
        ray_idx = np.arange(len(ray_directions))
        ranges = np.linalg.norm(intersection_points - position.reshape(1, 1, -1), axis=-1)
        if return_intersection_points:
            closest = np.argmin(ranges, axis=-1)
            points = intersection_points[ray_idx, closest, :]
            ranges = ranges[ray_idx, closest]
            points[ranges > self._distance_range] = np.array([np.inf, np.inf])
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
        position: Union[np.ndarray, Tuple[float, float]],
        scene: Scene,
        return_intersection_points: Optional[bool] = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Returns sensor's observation

        Args:
            position (Union[np.ndarray, Tuple[float, float]]): current sensor position in world coordinates.
            scene (Scene): current scene
            return_intersection_points (Optional[bool], optional): If True returns both intersection
                points and ranges. Defaults to False.

        Returns:
            Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]: If return_intesection_points is False,
                returns array of ranges, else returns tuple (range, intersection_points)
        """
        return super().get_observation(position, (1, 0), scene, return_intersection_points)
