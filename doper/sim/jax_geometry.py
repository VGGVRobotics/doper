from typing import List, Tuple, Union
from collections import namedtuple

import numpy as onp
import jax
import jax.numpy as np

Polygon = namedtuple("Polygon", ["segments"])
JaxScene = namedtuple("JaxScene", ["segments", "polygons"])


def compute_segment_normal_projection(point: np.ndarray, segment: np.ndarray) -> np.ndarray:
    """Computes projection of point to the polygon segment normal

    Args:
        point (np.ndarray): point to be projected
        segment (np.ndarray): segment of polygon

    Returns:
        np.ndarray: normal projection
    """
    normal = np.concatenate([segment[1], -segment[0]])
    normal = normal / np.linalg.norm(normal)
    return np.dot(point, normal) * normal


def compute_segment_projection(
    point: np.ndarray, segment: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Projects point on the polygon segment

    Args:
        point (np.ndarray): point to be projected
        segment (np.ndarray): segment to project on

    Returns:
        Tuple[np.ndarray, np.ndarray]: projection vector and boolean flag if projection is inner point
            or one of the segment's endpoints
    """
    segment_vector = segment[1] - segment[0]
    segment_len_sq = segment_vector @ segment_vector
    point_vector = point - segment[0]
    proj_ratio = (np.dot(point_vector, segment_vector) / segment_len_sq).clip(0, 1)
    point_projection = segment[0] + proj_ratio * segment_vector
    is_inner = np.logical_and(proj_ratio > 0, proj_ratio < 1)
    return point_projection, is_inner


def compute_distance_to_segment(point: np.ndarray, segment: np.ndarray) -> np.ndarray:
    """Computes minimal distance from point to segment

    Args:
        point (np.ndarray): point from which compute a distance
        segment (np.ndarray): segment to which compute a distance

    Returns:
        np.ndarray: distance, scalar array
    """
    point_on_segment, _ = compute_segment_projection(point, segment)
    distance = np.linalg.norm(point - point_on_segment, axis=-1)
    return distance


compute_distance_to_segments_batch = jax.vmap(compute_distance_to_segment, (None, 0), 0)


def find_closest_segment_to_point(
    point: np.ndarray, segments_batch: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Searches for the the segment with minimal distance to a given point

    Args:
        point (np.ndarray): point of interest
        segments_batch (np.ndarray): segments to check

    Returns:
        Tuple[np.ndarray, np.ndarray]: closest segment, distance to closest segment
    """
    distances = compute_distance_to_segments_batch(point, segments_batch)
    closest_idx = np.argmin(distances)
    return segments_batch[closest_idx], distances[closest_idx]


def line_ray_intersection_point(
    ray_origin: np.ndarray, ray_direction: np.ndarray, segment: np.ndarray
) -> np.ndarray:
    """Function to check intersection between ray and segment.

    See algorithm description here  http://bit.ly/1CoxdrG

    Args:
        ray_origin (np.ndarray): (2,) ray origin point
        ray_direction (np.ndarray): (2,) ray direction vector, can be unnormalized
        segments (np.ndarray): (2, 2) segment to check intersection with

    Returns:
        np.ndarray: (2,) intersection point
    """

    ray_direction = ray_direction / np.linalg.norm(ray_direction)
    v1 = ray_origin - segment[0]
    v2 = segment[1] - segment[0]
    v3 = np.array([-ray_direction[1], ray_direction[0]])
    denom = v2 @ v3

    def compute_intersection_point(denom):
        t1 = np.cross(v2, v1) / denom
        t2 = (v1 @ v3) / denom
        condition = np.logical_or(np.logical_or(t1 < 0.0, t2 < 0.0), t2 > 1.0)
        return jax.lax.cond(
            condition,
            true_fun=lambda t: np.array([np.inf, np.inf]),
            false_fun=lambda t: ray_origin + t1 * ray_direction,
            operand=t1,
        )

    return jax.lax.cond(
        denom == 0,
        true_fun=lambda d: np.array([np.inf, np.inf]),
        false_fun=compute_intersection_point,
        operand=denom,
    )


def _unpack_and_apply_lrip(batch_element: np.ndarray) -> np.ndarray:
    """Unpacks batched line ray intersection task and applies lrip function

    Args:
        batch_element (np.ndarray): (n_seg * n_rays, 4, 2) batch of tasks

    Returns:
        np.ndarray: (n_seg * n_rays, 2) batch of intersection points
    """
    ray_origin, ray_direction, segment = batch_element[0], batch_element[1], batch_element[2:]
    return line_ray_intersection_point(ray_origin, ray_direction, segment)


_lrip_with_unpack_map = jax.vmap(_unpack_and_apply_lrip, 0)


@jax.jit
def batch_line_ray_intersection_point(
    ray_origins: np.ndarray, ray_directions: np.ndarray, segments: np.ndarray
) -> np.ndarray:
    """Function to check intersection between multiple rays and multiple segments.
    See algorithm description here  http://bit.ly/1CoxdrG
    Args:
        ray_origins (np.ndarray): (n_rays, 2) array of ray origin points
        ray_directions (np.ndarray): (n_rays, 2) array of ray directions, may not
            be normalized
        segments (np.ndarray): (n_segments, 2, 2) array of segments endpoints
    Returns:
        np.ndarray: (n_rays, n_segments, 2) array of segment intersection points.
                    Contains np.inf if ray does not intersect the segment.
    """
    ray_origins = np.repeat(np.expand_dims(ray_origins, 1), segments.shape[0], 1)
    ray_directions = np.repeat(np.expand_dims(ray_directions, 1), segments.shape[0], 1)
    segments = np.repeat(np.expand_dims(segments, 0), ray_origins.shape[0], 0)
    ray_origins = np.expand_dims(ray_origins, 2)
    ray_directions = np.expand_dims(ray_directions, 2)
    # (n_rays, n_segments, 4, 2)
    batch = np.concatenate([ray_origins, ray_directions, segments], axis=2)
    batch = batch.reshape(-1, 4, 2)
    points_batch = _lrip_with_unpack_map(batch)
    return points_batch.reshape(ray_origins.shape[0], segments.shape[1], -1)


def rotate_polygon(polygon: Polygon, angle: float, center: Tuple[float, float]) -> None:
    """Rotates the polygon around center.

    Args:
        angle (float): rotation angle in degrees
        center (Tuple[float, float]): center point of rotation
    """
    angle_rad = angle / 180.0 * np.pi
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    rotation = np.array(((c, -s), (s, c)))
    transalation = np.array(center)
    segments = (
        (rotation @ (polygon.segments.reshape(-1, 2) - transalation).T).T + transalation
    ).reshape(-1, 2, 2)
    return Polygon(segments=segments)


def create_polygon(
    segments: Union[np.ndarray, onp.ndarray, List[Tuple[Tuple[float, float], Tuple[float, float]]]]
) -> Polygon:
    """Creates instance of jax-compatible polygon.

    Args:
        segments (
            Union[
                np.ndarray,
                onp.ndarray,
                List[Tuple[Tuple[float, float], Tuple[float, float]]]]
            ): input arrat of segments

    Returns:
        Polygon: polygon named tuple
    """
    if isinstance(segments, np.ndarray):
        return Polygon(segments=segments)
    return Polygon(segments=np.ndarray(segments))


def create_scene(polygons: List[Polygon]) -> JaxScene:
    return JaxScene(
        polygons=polygons, segments=np.concatenate([p.segments for p in polygons], axis=0)
    )
