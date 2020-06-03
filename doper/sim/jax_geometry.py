from typing import Tuple
from collections import namedtuple

import jax
import jax.numpy as np


JaxScene = namedtuple("JaxScene", ["segments"])


def compute_segment_normal_projection(point: np.ndarray, segment: np.ndarray) -> np.ndarray:
    normal = np.concatenate([segment[1], -segment[0]])
    normal = normal / np.linalg.norm(normal)
    return np.dot(point, normal) * normal


def compute_segment_projection(
    point: np.ndarray, segment: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    segment_vector = segment[1] - segment[0]
    segment_len_sq = segment_vector @ segment_vector
    point_vector = point - segment[0]
    proj_ratio = (np.dot(point_vector, segment_vector) / segment_len_sq).clip(0, 1)
    point_projection = segment[0] + proj_ratio * segment_vector
    is_inner = np.logical_and(proj_ratio > 0, proj_ratio < 1)
    return point_projection, is_inner


def compute_distance_to_segment(point: np.ndarray, segment: np.ndarray) -> np.ndarray:
    point_on_segment, _ = compute_segment_projection(point, segment)
    distance = np.linalg.norm(point - point_on_segment, axis=-1)
    return distance


compute_distance_to_segments_batch = jax.vmap(compute_distance_to_segment, (None, 0), 0)


def find_closest_segment_to_point(
    point: np.ndarray, segments_batch: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    distances = compute_distance_to_segments_batch(point, segments_batch)
    closest_idx = np.argmin(distances)
    return segments_batch[closest_idx], distances[closest_idx]
