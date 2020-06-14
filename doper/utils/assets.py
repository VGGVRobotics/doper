from typing import List, Tuple

import numpy as onp
from svgpathtools import Line
from svgpathtools import svg_to_paths as sp
import tripy
from ..sim.jax_geometry import JaxScene, create_polygon, rotate_polygon, create_scene


def line_begin_end(line: Line, scale: float, width: int, height: int) -> List[Tuple[float, float]]:
    """Converts svgpathtools Line to a pair of endpoints in x, y coordinates

    Args:
        line (Line): svgpathtools Line object
        scale (float): scale to divide coordinates
        width (int): svg width attribute
        height (int): svg height attribute

    Returns:
        List[Tuple[float, float]]: list of endpoint tuples
    """
    return [
        (line.start.real / scale, (height - line.start.imag) / scale),
        (line.end.real / scale, (height - line.end.imag) / scale),
    ]


def sort_segments(segments: onp.ndarray, orientation: str) -> onp.ndarray:
    """Sorts segments and endpoints in counterclockwise or clockwise order

    Args:
        segments (onp.ndarray): segments to sort
        orientation (str): sorting order, either clockwise or counterclockwise
    """
    idxs = onp.arange(len(segments))
    remaining_idxs = set(idxs)
    order = []
    sorted_idxs = sorted(
        idxs,
        key=lambda i: (
            min(segments[i, 0, 0], segments[i, 1, 0]),
            min(segments[i, 0, 1], segments[i, 1, 1]),
        ),
    )
    # choose candidate with lower y for at second endpoint
    candidates = segments[sorted_idxs[:2]].copy()
    for i in range(len(candidates)):
        if candidates[i, 0, 0] > candidates[i, 1, 0]:
            candidates[i, 0, 0], candidates[i, 1, 0] = candidates[i, 1, 0], candidates[i, 0, 0]
    if candidates[0, 1, 1] < candidates[1, 1, 1]:
        current_idx = sorted_idxs[0]
    else:
        current_idx = sorted_idxs[1]
    starting_segment = segments[current_idx].copy()
    if starting_segment[0, 0] > starting_segment[1, 0] or (
        starting_segment[0, 0] == starting_segment[1, 0]
        and starting_segment[0, 1] < starting_segment[1, 1]
    ):
        first, second = starting_segment[1], starting_segment[0]
    else:
        first, second = starting_segment[0], starting_segment[1]
    if orientation == "counterclockwise":
        segments[current_idx, 0], segments[current_idx, 1] = first, second
    elif orientation == "clockwise":
        segments[current_idx, 0], segments[current_idx, 1] = second, first
    else:
        raise ValueError(f"Unknown orientation {orientation}")
    order.append(current_idx)
    remaining_idxs.remove(current_idx)
    while len(remaining_idxs) > 0:
        common_endpoint = segments[current_idx, 1]
        current_idx = min(
            remaining_idxs,
            key=lambda i: min(
                onp.linalg.norm(segments[i, j] - common_endpoint) for j in range(len(segments[i]))
            ),
        )
        if not onp.allclose(segments[current_idx, 0], common_endpoint):
            segments[current_idx] = segments[current_idx][::-1]
        order.append(current_idx)
        remaining_idxs.remove(current_idx)
    return segments[order]


def get_svg_scene(fname: str, px_per_meter: float = 50) -> JaxScene:
    """Loads scene representation from svg file

    Args:
        fname (str): path to svg file
        px_per_meter (float, optional): pixels per meters scale. Defaults to 50.

    Returns:
        Scene: scene representation instance
    """
    polygons = []
    paths, attributes, svg_attributes = sp.svg2paths(fname, return_svg_attributes=True)
    w, h = svg_attributes["width"], svg_attributes["height"]
    w, h = int(w.replace("px", "")), int(h.replace("px", ""))
    for path, attr in zip(paths, attributes):
        if not path.isclosed():
            print(f"Found non-closed path {path}, skipping")
            continue
        if not all([isinstance(l, Line) for l in path]):
            print(f"Only simple line figures are currently allowed, skipping")
            continue
        # TODO: check lines color and set orientation
        onp_polygon = sort_segments(
            segments=onp.concatenate(
                [onp.array(line_begin_end(line, px_per_meter, w, h))[onp.newaxis] for line in path],
                axis=0,
            ),
            orientation="counterclockwise",
        )
        # triangulate:
        polygon_points = [s[0] for s in onp_polygon]
        triangles_points = tripy.earclip(polygon_points)
        # print(triangles_points)
        idxs = onp.array([0, 1, 2])
        for triangle in triangles_points:
            segments = onp.zeros((3, 2, 2), dtype=onp_polygon.dtype)
            segments[:, 0] = onp.asarray(triangle)
            segments[:, 1] = segments[(idxs + 1) % 3, 0]
            jax_polygon = create_polygon(segments)
            if "transform" in attr and "rotate" in attr["transform"]:
                angle, cx, cy = eval(attr["transform"].replace("rotate", ""))
                cx, cy = cx, h - cy
                jax_polygon = rotate_polygon(
                    jax_polygon, angle, (cx / px_per_meter, cy / px_per_meter)
                )
            polygons.append(jax_polygon)

    return create_scene(polygons)
