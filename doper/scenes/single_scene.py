__all__ = ["SingleScene"]

import logging
import numpy as onp
import jax.numpy as np

from doper.utils.assets import get_svg_scene
from doper.sim.jax_geometry import (
    if_points_inside_any_polygon,
    find_closest_segment_to_points_batch,
)

logger = logging.getLogger(__name__)


class SingleScene:
    def __init__(self, config: dict):
        self.config = config
        self.jax_scene = get_svg_scene(
            config["sim"]["scene_params"]["svg_scene_path"],
            px_per_meter=config["sim"]["scene_params"]["px_per_meter"],
        )

    def get_init_state(self, batch_size: int):
        """
        Returns jax object with initial state
        Args:
            batch_size: size of a batch

        Returns:
            [batch_size, 2] jax array with initial coordinates
        """
        eps = 0.05
        onp_segments = onp.asarray(self.jax_scene.segments)
        max_x, min_x = onp.max(onp_segments[:, :, 0]), onp.min(onp_segments[:, :, 0])
        max_y, min_y = onp.max(onp_segments[:, :, 1]), onp.min(onp_segments[:, :, 1])
        while True:
            proposal = onp.random.uniform(
                (min_x - 2, max_x + 2), (min_y - 2, max_y + 2), size=(batch_size, 2)
            )
            proposal = np.array(proposal)
            is_inner = if_points_inside_any_polygon(proposal, self.jax_scene)
            _, distance = find_closest_segment_to_points_batch(proposal, self.jax_scene.segments)
            acceptable = np.logical_not(
                np.logical_or(is_inner, distance <= self.config["constants"]["radius"] + eps)
            )
            if np.all(acceptable):
                break
            logger.debug("Resampling starting position")
        return np.array(proposal)
