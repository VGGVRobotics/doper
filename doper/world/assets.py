from typing import List, Tuple
from svgpathtools import Line
from svgpathtools import svg_to_paths as sp

from .shapes import Polygon
from .scene import Scene


def svg2polygons(fname: str, scale: float) -> List[Polygon]:
    """Function to convert svg paths to polygons.
    Non-closed paths are ignored, paths containing anything except Line are ignored.

    Args:
        fname (str): svg file name
        scale (float): scale to divide coordinates

    Returns:
        List[Polygon]: resulting list of polygons
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
        poly = Polygon([line_begin_end(line, scale, w, h) for line in path])
        if "transform" in attr and "rotate" in attr["transform"]:
            angle, cx, cy = eval(attr["transform"].replace("rotate", ""))
            cx, cy = cx, h - cy
            poly.rotate(angle, (cx / scale, cy / scale))

        polygons.append(poly)

    return polygons


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


def get_svg_scene(fname: str, px_per_meter: float = 50) -> Scene:
    """Loads scene representation from svg file

    Args:
        fname (str): path to svg file
        px_per_meter (float, optional): pixels per meters scale. Defaults to 50.

    Returns:
        Scene: scene representation instance
    """
    return Scene(svg2polygons(fname, scale=px_per_meter))
