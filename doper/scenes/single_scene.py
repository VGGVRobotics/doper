__all__ = ["SingleScene"]

import numpy as onp
import jax.numpy as np

from doper.sim.jax_geometry import JaxScene
from doper.world.assets import get_svg_scene


class SingleScene:
    def __init__(self, config: dict):
        self.config = config
        self.scene = get_svg_scene(config["sim"]["scene_params"]["svg_scene_path"],
                                   px_per_meter=config["sim"]["scene_params"]["px_per_meter"])
        self.jax_scene = JaxScene(segments=np.array(self.scene.get_all_segments()))

    def get_init_state(self, batch_size: int):
        """
        Returns jax object with initial state
        Args:
            batch_size: size of a batch

        Returns:
            [batch_size, 2] jax array with initial coordinates
        """
        proposal = onp.random.uniform((-4.0, -4.0), (16.0, 0.0), size=(batch_size, 2))
        return np.array(proposal)
