__all__ = ["MultipleScenes"]

import logging
import numpy as onp
import jax.numpy as np
from glob import glob

from doper.utils.assets import get_svg_scene
from doper.sim.jax_geometry import (
    if_points_inside_any_polygon,
    find_closest_segment_to_points_batch,
)

logger = logging.getLogger(__name__)


class MultipleScenes:
    def __init__(self, config: dict):
        self.config = config
        scene_paths = glob(config["sim"]["scene_params"]["svg_scene_path"])
        self.jax_scenes = [
            get_svg_scene(scene, 
            px_per_meter=self.config["sim"]["scene_params"]["px_per_meter"],
            ) for scene in scene_paths]

    def get_init_state(self, batch_size: int):
        """
        Returns jax object with initial state
        Args:
            batch_size: size of a batch

        Returns:
            [batch_size, 2] jax array with initial coordinates
        """
        self.jax_scene = self.jax_scenes[onp.random.randint(0, len(self.jax_scenes), 1)[0]]
        eps = self.config["constants"]["radius"] / 2

        max_x, min_x = onp.max(self._onp_segments[:, :, 0]), onp.min(self._onp_segments[:, :, 0])
        max_y, min_y = onp.max(self._onp_segments[:, :, 1]), onp.min(self._onp_segments[:, :, 1])
        remaining_idxs = np.arange(batch_size)
        init_proposal = onp.random.uniform(
            (min_x - 1, min_y - 1), (max_x + 1, max_y + 1), size=(batch_size, 2)
        )
        proposal_jax = np.asarray(init_proposal)
        while True:
            is_inner = if_points_inside_any_polygon(proposal_jax, self.jax_scene)
            _, distance = find_closest_segment_to_points_batch(
                proposal_jax, self.jax_scene.segments
            )
            acceptable = onp.asarray(
                np.logical_not(
                    np.logical_or(is_inner, distance <= self.config["constants"]["radius"] + eps)
                )
            )
            init_proposal[remaining_idxs[acceptable], :] = onp.array(proposal_jax)[acceptable, :]
            if np.all(acceptable):
                break
            logger.debug("Resampling starting position")
            remaining_idxs = remaining_idxs[np.logical_not(acceptable)]
            proposal_jax = np.asarray(
                onp.random.uniform((min_x, min_y), (max_x, max_y), size=(len(remaining_idxs), 2))
            )
        return np.array(init_proposal)
