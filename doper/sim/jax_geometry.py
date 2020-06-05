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
