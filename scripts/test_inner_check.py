from doper.utils.assets import get_svg_scene
from time import time
from matplotlib import pyplot as plt
from matplotlib import collections as mc
import jax
import numpy as onp
from doper.sim.jax_geometry import (
    if_points_inside_any_polygon,
    JaxScene,
    find_closest_segment_to_points_batch,
)


if __name__ == "__main__":

    scene = get_svg_scene("assets/map_40.svg", px_per_meter=50)
    # scene = JaxScene(segments=scene.segments, polygons=scene.polygons, polygon_ranges)
    onp_segments = onp.asarray(scene.segments)
    max_x, min_x = onp.max(onp_segments[:, :, 0]), onp.min(onp_segments[:, :, 0])
    max_y, min_y = onp.max(onp_segments[:, :, 1]), onp.min(onp_segments[:, :, 1])
    for i in range(10):
        proposal = onp.random.uniform((min_x, min_y), (max_x, max_y), size=(100, 2))
        start = time()
        proposal = jax.numpy.array(proposal)
        is_inner = if_points_inside_any_polygon(proposal, scene)
        _, distance = find_closest_segment_to_points_batch(proposal, scene.segments)
        acceptable = onp.logical_not(is_inner)
        print(time() - start)
    lines = mc.LineCollection(scene.segments)
    fig, ax = plt.subplots()
    ax.add_collection(lines)
    outer = [proposal[i] for i in range(len(proposal)) if acceptable[i]]
    inner = [proposal[i] for i in range(len(proposal)) if not acceptable[i]]
    ax.scatter([p[0] for p in outer], [p[1] for p in outer], color="b")
    ax.scatter([p[0] for p in inner], [p[1] for p in inner], color="r")
    fig.legend()
    plt.show()
