from typing import List, Tuple, Union
from collections import namedtuple

import numpy as onp
import jax
import jax.numpy as np

Polygon = namedtuple("Polygon", ["segments"])
JaxScene = namedtuple("JaxScene", ["segments", "polygons", "polygon_ranges"])


def _compute_segment_normal(segment: np.ndarray) -> np.ndarray:
    """Computes normal direction for the segment

    Args:
        segment (np.ndarray): segment to compute normal direction for

    Returns:
        np.ndarray: normal direction vector
    """
    segment_vector = segment[1] - segment[0]
    normal = np.array([segment_vector[1], -segment_vector[0]])
    normal = normal / np.linalg.norm(normal)
    return normal


def compute_segment_normal_projection(point: jax.numpy.ndarray, segment: np.ndarray) -> np.ndarray:
    """Computes projection of point to the polygon segment normal

    Args:
        point (np.ndarray): point to be projected
        segment (np.ndarray): segment of polygon

    Returns:
        np.ndarray: normal projection
    """
    normal = _compute_segment_normal(segment)
    return np.dot(point, normal) * normal


def compute_segment_normal_projection_sign(point: np.ndarray, segment: np.ndarray) -> np.ndarray:
    """Computes sign of projection on segment's normal direction
    Args:
        point (np.ndarray): point being projected on the normal
        segment (np.ndarray): segment to project to

    Returns:
        np.ndarray: sign of projection
    """
    normal = _compute_segment_normal(segment)
    point_vector = point - segment[0]
    return np.sign(np.dot(point_vector, normal))


def _unpack_and_apply_nps(batch_element: np.ndarray) -> np.ndarray:
    """Unpacks batch for cartesian map of compute_segment_normal_projection sign

    Args:
        batch_element (np.ndarray): single batch element

    Returns:
        np.ndarray: normal projection sign
    """
    point, segment = batch_element[0], batch_element[1:]
    return compute_segment_normal_projection_sign(point, segment)


_batch_unpack_and_apply_nps = jax.vmap(_unpack_and_apply_nps)


def batch_segment_normal_projection_sign(points: np.ndarray, segments: np.ndarray) -> np.ndarray:
    """Batched version of compute_segment_normal_projection_sign

    Args:
        points (np.ndarray): points batch
        segments (np.ndarray): segments batch

    Returns:
        np.ndarray: (n_points, n_segments) array of normal projection signs
    """

    # TODO: may be we can somehow generalize packing and unpacking for cartesian map
    # i.e. map of two arrays with shapes (N, ...) and (M, ...) and result with shape (N, M)
    points = np.repeat(np.expand_dims(points, 1), segments.shape[0], 1)
    segments = np.repeat(np.expand_dims(segments, 0), points.shape[0], 0)
    points = np.expand_dims(points, 2)
    # (n_points, n_segments, 3, 2)
    batch = np.concatenate([points, segments], axis=2)
    batch = batch.reshape(-1, 3, 2)
    signs_batch = _batch_unpack_and_apply_nps(batch)
    return signs_batch.reshape(points.shape[0], segments.shape[1])


# slow without jit, slow compilation with jit if too many polygons (> 10 i think)
# no simple workaround except do it in numba instead of jax
def if_points_inside_any_polygon(points: np.ndarray, scene: JaxScene) -> np.ndarray:
    """Checks if points is inside any polygon

    Args:
        points (np.ndarray): point or batch of points to check
        scene (JaxScene): scene instance

    Returns:
        np.ndarray: bool result
    """

    if points.ndim == 1:
        points = points.reshape(1, -1)
    result = np.zeros(len(points)).astype(bool)
    signs = batch_segment_normal_projection_sign(points, scene.segments)
    signs = signs < 0
    for start, end in scene.polygon_ranges:
        is_inside = np.all(signs[:, start:end], axis=-1)
        result = np.logical_or(is_inside, result)
    return result


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


find_closest_segment_to_points_batch = jax.vmap(find_closest_segment_to_point, (0, None), 0)


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
    polygon_ranges = onp.zeros((len(polygons), 2), dtype=np.int32)
    start = 0
    for i, poly in enumerate(polygons):
        polygon_ranges[i, 0] = start
        polygon_ranges[i, 1] = start + len(poly.segments)
        start += len(poly.segments)
    return JaxScene(
        polygons=polygons,
        segments=np.concatenate([p.segments for p in polygons], axis=0),
        polygon_ranges=np.array(polygon_ranges),
    )
