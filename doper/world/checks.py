from typing import List, Union, Tuple
import numpy as np
from .shapes import Polygon

# https://gist.github.com/danieljfarrell/faf7c4cafd683db13cbc

# TODO: we can rewrite using numba scalar to improve perf if needed


def batch_line_ray_intersection_point(
    ray_origins: np.ndarray, ray_directions: np.ndarray, segments: np.ndarray
) -> np.ndarray:
    """Function to check intersection between multiple rays and multiple segments.

    See algorithm description here  http://bit.ly/1CoxdrG

    Args:
        ray_origins (np.ndarray): [n_rays, 2] array of ray origin points
        ray_directions (np.ndarray): [n_rays, 2] array of ray directions, may not
            be normalized
        segments (np.ndarray): [n_segments, 2, 2] array of segments endpoints

    Returns:
        np.ndarray: [n_rays, n_segments, 2] array of segment intersection points.
                    Contains np.inf if ray does not intersect the segment.
    """
    ray_directions = ray_directions / np.linalg.norm(ray_directions, axis=-1)[..., np.newaxis]
    v1 = ray_origins[:, np.newaxis, :] - segments[:, 0, :]
    v2 = segments[:, 1, :] - segments[:, 0, :]
    v3 = ray_directions[:, ::-1]
    v3[:, 0] = -v3[:, 0]
    # v2 [n_segm, 2], v1 [n_origins, n_segments, 2], v3 [n_origins, 2]
    denom = v3 @ v2.T
    t1 = np.cross(v2[np.newaxis, ...], v1)
    t1[denom != 0] = t1[denom != 0] / denom[denom != 0]
    t1[denom == 0] = np.inf
    # 2d cross is scalar, so t1[n_origins, n_segm]
    t2 = (v1 @ v3[..., np.newaxis]).squeeze()
    t2[denom != 0] = t2[denom != 0] / denom[denom != 0]
    t2[denom == 0] = np.inf
    mask = (t1 < 0.0) + (t2 < 0.0) + (t2 > 1.0)
    t1 = np.concatenate([t1[..., np.newaxis], -t1[..., np.newaxis]], axis=-1)
    intersections = ray_origins[:, np.newaxis, :] + t1 * ray_directions[:, np.newaxis, :]
    intersections[mask, :] = np.inf
    return intersections


def polygons_in_rect_area(
    polygons: List[Polygon],
    lower_left: Union[np.ndarray, Tuple[float, float]],
    upper_right: Union[np.ndarray, Tuple[float, float]],
) -> List[Polygon]:
    """Returns only polygons with at least one vertex in given rectangular area

    Args:
        polygons (List[Polygon]): list of polygons
        lower_left (Union[np.ndarray, Tuple[float, float]]): (x, y) lower left corner of rectangular area
        upper_right (Union[np.ndarray, Tuple[float, float]]): (x, y) upper right corner of rectangular area

    Returns:
        List[Polygon]: filtered list of polygons
    """
    res = []
    x_min, y_min = lower_left
    x_max, y_max = upper_right
    for p in polygons:
        segments = p.segments
        is_point_inside_rect = (
            (segments[:, :, 0] >= x_min)
            * (segments[:, :, 1] >= y_min)
            * (segments[:, :, 0] <= x_max)
            * (segments[:, :, 1] <= y_max)
        )
        if np.any(is_point_inside_rect):
            res.append(p)
    return res
