import numpy as np
from typing import List, Tuple, Union


class Polygon:
    def __init__(
        self, segments: List[List[Union[Tuple, List]]], orientation: str = "counterclockwise"
    ) -> None:
        """Represents polygon

        Args:
            segments (List[List[Union[Tuple, List]]]): list of pairs of segment endpoints coordinates
        """
        self._segments = np.array(segments, dtype=np.float32)
        self._sort_segments(orientation=orientation)
        self._normals = np.zeros((len(self._segments), 2))
        segments_vectors = self._segments[:, 1] - self._segments[:, 0]
        self._normals[:, 0], self._normals[:, 1] = segments_vectors[:, 1], -segments_vectors[:, 0]
        self._normals /= np.linalg.norm(self._normals, axis=-1, keepdims=True)

    @property
    def segments(self) -> np.ndarray:
        """np.ndarray: list of pairs of segment endpoints coordinates
        """
        return self._segments

    @property
    def normals(self) -> np.ndarray:
        """np.ndarray: list of normal vectors to corresponding segments
        """
        return self._normals

    def rotate(self, angle: float, center: Tuple[float, float]) -> None:
        """Rotates this polygon around center.

        Args:
            angle (float): rotation angle in degrees
            center (Tuple[float, float]): center point of rotation
        """
        angle_rad = angle / 180 * np.pi
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        rotation = np.array(((c, -s), (s, c)))
        transalation = np.array(center)
        self._segments = (
            (rotation @ (self._segments.reshape(-1, 2) - transalation).T).T + transalation
        ).reshape(-1, 2, 2)
        self._normals = (rotation @ (self._normals).T).T

    def _sort_segments(self, orientation: str) -> None:
        """Sorts segments and endpoints in counterclockwise or clockwise order

        Args:
            orientation (str): sorting order, either clockwise or counterclockwise
        """
        idxs = np.arange(len(self.segments))
        remaining_idxs = set(idxs)
        order = []
        sorted_idxs = sorted(
            idxs,
            key=lambda i: (
                min(self._segments[i, 0, 0], self.segments[i, 1, 0]),
                min(self._segments[i, 0, 1], self.segments[i, 1, 1]),
            ),
        )
        # choose candidate with lower y for at second endpoint
        candidates = self._segments[sorted_idxs[:2]].copy()
        for i in range(len(candidates)):
            if candidates[i, 0, 0] > candidates[i, 1, 0]:
                candidates[i, 0, 0], candidates[i, 1, 0] = candidates[i, 1, 0], candidates[i, 0, 0]
        if candidates[0, 1, 1] < candidates[1, 1, 1]:
            current_idx = sorted_idxs[0]
        else:
            current_idx = sorted_idxs[1]
        starting_segment = self._segments[current_idx].copy()
        if starting_segment[0, 0] > starting_segment[1, 0] or (
            starting_segment[0, 0] == starting_segment[1, 0]
            and starting_segment[0, 1] < starting_segment[1, 1]
        ):
            first, second = starting_segment[1], starting_segment[0]
        else:
            first, second = starting_segment[0], starting_segment[1]
        if orientation == "counterclockwise":
            self._segments[current_idx, 0], self._segments[current_idx, 1] = first, second
        elif orientation == "clockwise":
            self._segments[current_idx, 0], self._segments[current_idx, 1] = second, first
        else:
            raise ValueError(f"Unknown orientation {orientation}")
        order.append(current_idx)
        remaining_idxs.remove(current_idx)
        while len(remaining_idxs) > 0:
            common_endpoint = self._segments[current_idx, 1]
            current_idx = min(
                remaining_idxs,
                key=lambda i: min(
                    np.linalg.norm(self._segments[i, j] - common_endpoint)
                    for j in range(len(self._segments[i]))
                ),
            )
            if not np.allclose(self._segments[current_idx, 0], common_endpoint):
                self._segments[current_idx] = self._segments[current_idx][::-1]
            order.append(current_idx)
            remaining_idxs.remove(current_idx)
        self._segments[:] = self._segments[order]
