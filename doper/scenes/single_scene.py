__all__ = ["SingleScene"]

import numpy as onp
import jax.numpy as np

from doper.sim.jax_geometry import JaxScene
from doper.world.assets import get_svg_scene


class SingleScene:
    def __init__(self, config):
        self.config = config
        self.scene = get_svg_scene(config["sim"]["scene_params"]["svg_scene_path"],
                                   px_per_meter=config["sim"]["scene_params"]["px_per_meter"])
        self.jax_scene = JaxScene(segments=np.array(self.scene.get_all_segments()))

    def get_init_state(self, batch_size):
        return np.array(onp.random.uniform((-4.0, -4.0), (16.0, 0.0), size=(self.config["train"]["batch_size"], 2)))
