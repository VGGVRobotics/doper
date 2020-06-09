__all__ = ["SingleScene"]

import numpy as onp
import jax.numpy as np

from doper.world.assets import get_svg_scene


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
        onp_segments = onp.asarray(self.jax_scene.segments)
        max_x, min_x = onp.max(onp_segments[:, :, 0]), onp.min(onp_segments[:, :, 0])
        max_y, min_y = onp.max(onp_segments[:, :, 1]), onp.min(onp_segments[:, :, 1])
        proposal = onp.random.uniform(
            (min_x - 1, max_x + 1), (min_y - 1, max_y + 1), size=(batch_size, 2)
        )

        return np.array(proposal)
