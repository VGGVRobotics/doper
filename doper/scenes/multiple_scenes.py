__all__ = ["MultipleScenes"]

import logging
from glob import glob

import jax
import numpy as onp

from .single_scene import SingleScene

logger = logging.getLogger(__name__)


class MultipleScenes:
    def __init__(self, svg_scene_path: str, px_per_meter: int, agent_radius: float) -> None:
        scene_paths = glob(svg_scene_path)
        self.single_scenes = []
        for scene_path in scene_paths:
            self.single_scenes.append(
                SingleScene(
                    svg_scene_path=scene_path, px_per_meter=px_per_meter, agent_radius=agent_radius
                )
            )

    def get_init_state(self, batch_size: int) -> jax.numpy.ndarray:
        single_scene = onp.random.choice(self.single_scenes, 1)[0]
        self.jax_scene = single_scene.jax_scene
        return single_scene.get_init_state(batch_size)

    def __iter__(self):
        for scene in self.single_scenes:
            yield scene
