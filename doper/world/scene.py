from typing import List, Union, Tuple
import numpy as np
from .shapes import Polygon
from .checks import batch_line_ray_intersection_point, polygons_in_rect_area


class Scene:
    def __init__(self, polygons: List[Polygon]):
        self._polygons = polygons

    def get_polygons_segments(self, polygons: List[Polygon]) -> np.ndarray:
        return np.concatenate([p.segments for p in polygons], axis=0)

    def get_polygons_in_area(
        self,
        lower_left: Union[np.ndarray, Tuple[float, float]],
        upper_right: Union[np.ndarray, Tuple[float, float]],
    ) -> List[Polygon]:
        return polygons_in_rect_area(self._polygons, lower_left, upper_right)

    def get_polygons_in_radius(
        self, center: Union[np.ndarray, Tuple[float, float]], radius: float
    ) -> List[Polygon]:
        # TODO: add inside radii filtering
        return self._polygons

    def get_all_polygons(self) -> List[Polygon]:
        return self._polygons

    def is_point_inside_any_polygon(self, points: np.ndarray) -> np.ndarray:
        # TODO: can be optimized
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


class Rasterizer:
    def __init__(self, points_per_meter: int = 2, frustrum_size: Tuple[int] = (64, 48)) -> None:
        self._points_per_meter = points_per_meter
        self._frutstrum_size = frustrum_size
        self._frame = np.zeros(
            (frustrum_size[1] * points_per_meter, frustrum_size[0] * points_per_meter, 3)
        )
        xx = np.linspace(0, frustrum_size[0], frustrum_size[0] * points_per_meter)
        yy = np.linspace(0, frustrum_size[1], frustrum_size[1] * points_per_meter)
        gx, gy = np.meshgrid(xx, yy, indexing="xy")
        self._coords = np.concatenate([gx[..., np.newaxis], gy[..., np.newaxis]], axis=-1)

    def rasterize(
        self, scene: Scene, frustrum_left_lower_coords: Union[np.ndarray, Tuple[float]]
    ) -> np.ndarray:
        frustrum_left_lower_coords = np.array(frustrum_left_lower_coords)
        is_inner = scene.is_point_inside_any_polygon(
            self._coords.reshape(-1, 2) + frustrum_left_lower_coords
        )

        self._frame[is_inner.reshape(self._coords.shape[:2])[::-1, :]] = np.array([1, 1, 0])
        return self._frame
