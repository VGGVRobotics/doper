import numpy as np
from typing import List, Tuple, Union


class Polygon:
    def __init__(self, segments: List[List[Union[Tuple, List]]]) -> None:
        self._segments = np.array(segments, dtype=np.float32)

    @property
    def segments(self) -> List[List[Union[Tuple, List]]]:
        return self._segments
