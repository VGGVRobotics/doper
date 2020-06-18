__all__ = ["MultipleScenes"]

import logging
from glob import glob

import numpy as onp

from doper.utils.assets import get_svg_scene
from .single_scene import SingleScene

logger = logging.getLogger(__name__)


class MultipleScenes(SingleScene):
    def __init__(self, config: dict):
        self.config = config
        scene_paths = glob(config["sim"]["scene_params"]["svg_scene_path"])
        self.jax_scenes = [
            get_svg_scene(scene, 
            px_per_meter=self.config["sim"]["scene_params"]["px_per_meter"],
            ) for scene in scene_paths]

    def _get_scene(self):
        return self.jax_scenes[onp.random.randint(0, len(self.jax_scenes), 1)[0]]
