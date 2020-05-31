from typing import List, Union, Tuple
import jax.numpy as np
from .shapes import Polygon
from .checks import (
    batch_line_ray_intersection_point,
    polygons_in_rect_area,
    batch_point_to_segment_distance,
)


class Scene:
    def __init__(self, polygons: List[Polygon]):
        """Scene representation. Can perfom basic geometric queries.

        Args:
            polygons (List[Polygon]): list of polygones
        """
        self._polygons = polygons
        self._segment_to_polygon = []
        self._all_segments = []
        self._all_normals = []
        for i, polygon in enumerate(self._polygons):
            segments = polygon.segments
            self._segment_to_polygon.extend([i] * len(segments))
            self._all_segments.append(segments)
            self._all_normals.append(polygon.normals)

        self._all_segments = np.concatenate(self._all_segments, axis=0)
        self._all_normals = np.concatenate(self._all_normals, axis=0)
        self._segment_to_polygon = np.array(self._segment_to_polygon)
        self._segment_idxs = np.arange(len(self._all_segments))

    def get_all_segments(self) -> np.ndarray:
        return self.get_polygons_segments(self.get_all_polygons())

    def get_polygons_segments(self, polygons: List[Polygon]) -> np.ndarray:
        """Get concatenated segments of given polygones.

        Args:
            polygons (List[Polygon]): list of polygons

        Returns:
            np.ndarray: [n_segments, 2, 2] array of segment endpoints
        """
        return np.concatenate([p.segments for p in polygons], axis=0)

    def get_polygons_in_area(
        self,
        lower_left: Union[np.ndarray, Tuple[float, float]],
        upper_right: Union[np.ndarray, Tuple[float, float]],
    ) -> List[Polygon]:
        """Returns scene polygons inside rectangular area

        Args:
            lower_left (Union[np.ndarray, Tuple[float, float]]): ROI left lower corner
            upper_right (Union[np.ndarray, Tuple[float, float]]): ROI right upper corner

        Returns:
            List[Polygon]: list of polygons
        """
        return polygons_in_rect_area(self._polygons, lower_left, upper_right)

    def get_polygons_in_radius(
        self, center: Union[np.ndarray, Tuple[float, float]], radius: float
    ) -> List[Polygon]:
        """Returns all polygons in given radius

        Args:
            center (Union[np.ndarray, Tuple[float, float]]): center of the area
            radius (float): radius of the area

        Returns:
            List[Polygon]: list of polygons
        """
        # TODO: use kd-tree for fast query
        # center = np.array(center)
        distances, _ = batch_point_to_segment_distance(
            center, self._all_segments, self._all_normals
        )
        polygon_idxs = set(self._segment_to_polygon[self._segment_idxs[distances <= radius]])
        return [self._polygons[i] for i in polygon_idxs]

    def get_closest_geometry(
        self, point: Union[np.ndarray, Tuple[float, float]]
    ) -> Tuple[np.ndarray, np.ndarray, float, bool]:
        """Returns closest segment, segment normal and distance to the point

        Args:
            point (Union[np.ndarray, Tuple): point to measure distance to

        Returns:
            Tuple[np.ndarray, np.ndarray, float, bool]: tuple of shapes (2, 2), (2), scalar, segment,
             it's normal, distance value, and flag whether closest point is inside the segment or it's corner
        """

        # TODO: use kd-tree for fast query
        # point = np.array(point)
        distances, is_inner = batch_point_to_segment_distance(
            point, self._all_segments, self._all_normals
        )
        # distances[distances < 0] = np.inf
        closest_idx = np.argmin(distances)
        return (
            self._all_segments[closest_idx],
            self._all_normals[closest_idx],
            distances[closest_idx],
            is_inner[closest_idx],
        )

    def get_all_polygons(self) -> List[Polygon]:
        """Returns all scene polygons

        Returns:
            List[Polygon]: list of polygons
        """
        return self._polygons

    def is_point_inside_any_polygon(self, points: np.ndarray) -> np.ndarray:
        """Checks if given points lie inside any polygon.

        Args:
            points (np.ndarray): [n_points, 2] array of points coordinates

        Returns:
            np.ndarray: [n_points] bool array
        """
        # There is a problem if we have a fence around the scene - any point inside will
        # return true
        if len(points) > 1:
            x_min = points[:, 0].min()
            y_min = points[:, 1].min()
            x_max = points[:, 0].max()
            y_max = points[:, 1].max()
            valid_polygons = polygons_in_rect_area(self._polygons, (x_min, y_min), (x_max, y_max))
        else:
            valid_polygons = self.get_all_polygons()
        all_segments = np.concatenate([p.segments for p in valid_polygons], axis=0)
        directions = np.array([1, 0], dtype=np.float32).reshape(1, -1).repeat(len(points), axis=0)
        intersection_points = batch_line_ray_intersection_point(points, directions, all_segments)
        intersection_mask = np.all(np.abs(intersection_points) != np.inf, axis=-1)
        is_inner_point = np.zeros(len(points)).astype(bool)
        idx = 0
        for p in valid_polygons:
            num_intersections = (
                intersection_mask[:, idx : idx + len(p.segments)].astype(int).sum(axis=1)
            )
            is_inner_point = np.logical_or(is_inner_point, num_intersections % 2 != 0)
            idx += len(p.segments)
        return is_inner_point
