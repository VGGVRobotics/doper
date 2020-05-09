import numpy as np
from matplotlib import pyplot as plt

# https://gist.github.com/danieljfarrell/faf7c4cafd683db13cbc


def batch_line_ray_intersection_point(
    ray_origins: np.ndarray, ray_directions: np.ndarray, segments: np.ndarray
) -> np.ndarray:
    # Ray-Line Segment Intersection Test in 2D
    # http://bit.ly/1CoxdrG
    ray_directions = ray_directions / np.linalg.norm(ray_directions, axis=-1)[..., np.newaxis]
    v1 = ray_origins[:, np.newaxis, :] - segments[:, 0, :]
    v2 = segments[:, 1, :] - segments[:, 0, :]
    v3 = ray_directions[:, ::-1]
    v3[:, 0] = -v3[:, 0]
    # v2 [n_segm, 2], v1 [n_origins, n_segments, 2], v3 [n_origins, 2]
    denom = v3 @ v2.T
    # do not want any warnings
    denom_nz = denom.copy()
    denom_nz[denom == 0] = 1
    t1 = np.cross(v2[np.newaxis, ...], v1)
    t1 = np.where(denom == 0, -1, t1 / denom_nz)
    # 2d cross is scalar, so t1[n_origins, n_segm]
    t2 = (v1 @ v3[..., np.newaxis]).squeeze()
    t2 = np.where(denom == 0, np.inf, t2 / denom_nz)
    mask = (t1 < 0.0) + (t2 < 0.0) + (t2 > 1.0)
    intersections = (
        ray_origins[:, np.newaxis, :] + t1[..., np.newaxis] * ray_directions[:, np.newaxis, :]
    )
    intersections[mask] = np.inf
    return intersections
