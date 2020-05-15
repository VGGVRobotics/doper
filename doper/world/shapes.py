import numpy as np
from typing import List, Tuple, Union


class Polygon:
    def __init__(self, segments: List[List[Union[Tuple, List]]]) -> None:
        self._segments = np.array(segments, dtype=np.float32)

    @property
    def segments(self) -> List[List[Union[Tuple, List]]]:
        return self._segments

    def rotate(self, angle: float, center: Tuple[float, float]) -> None:
        angle_rad = angle / 180 * np.pi
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        rotation = np.array(((c, -s), (s, c)))
        transalation = np.array(center)
        self._segments = (
            (rotation @ (self._segments.reshape(-1, 2) - transalation).T).T + transalation
        ).reshape(-1, 2, 2)
