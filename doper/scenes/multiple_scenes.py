__all__ = ["MultipleScenes"]

import logging
from glob import glob

import jax
import numpy as onp

from doper.utils.assets import get_svg_scene
from .single_scene import SingleScene

logger = logging.getLogger(__name__)


class MultipleScenes:
    def __init__(self, config: dict) -> None:
        self.config = config
        scene_paths = glob(config["svg_scene_path"])
        self.single_scenes = []
        for scene_path in scene_paths:
            single_config = config.copy()
            single_config["svg_scene_path"] = scene_path
            self.single_scenes.append(SingleScene(single_config))

    def get_init_state(self, batch_size: int) -> jax.numpy.ndarray:
        single_scene = onp.random.choice(self.single_scenes, 1)[0]
        self.jax_scene = single_scene.jax_scene
        return single_scene.get_init_state(batch_size)

    def __iter__(self):
        for scene in self.single_scenes:
            yield scene.get_init_state(1)

