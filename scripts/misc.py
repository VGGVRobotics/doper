from doper.utils.assets import get_svg_scene
from time import time
from matplotlib import pyplot as plt
from matplotlib import collections as mc
import jax
import numpy as onp
from doper.sim.jax_geometry import (
    if_point_inside_any_polygon,
    JaxScene,
    find_closest_segment_to_point,
)


if __name__ == "__main__":
    points = []
    is_inside = []
    scene = get_svg_scene("../assets/simple_level.svg", px_per_meter=50)
    # scene = JaxScene(segments=scene.segments, polygons=scene.polygons, polygon_ranges)
    for _ in range(100):
        point = onp.random.uniform((0, 7.0), (16.0, 0.0), size=(2,))
        points.append(point)
        start = time()
        result = if_point_inside_any_polygon(point, scene)
        _, d = find_closest_segment_to_point(point, scene.segments)
        if d < 0.2:
            result = True
        is_inside.append(result)
        print(result, time() - start)
    lines = mc.LineCollection(scene.segments)
    fig, ax = plt.subplots()
    ax.add_collection(lines)
    outer = [points[i] for i in range(len(points)) if not is_inside[i]]
    inner = [points[i] for i in range(len(points)) if is_inside[i]]
    ax.scatter([p[0] for p in outer], [p[1] for p in outer], color="b")
    ax.scatter([p[0] for p in inner], [p[1] for p in inner], color="r")
    fig.legend()
    plt.show()
