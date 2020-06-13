import os

import numpy as np
from svgpathtools import Line, Path, wsvg


def self_intersects(line, lines):
    for l in lines:
        intersection = line.intersect(l)
        if len(intersection) == 0:
            pass
        else:
            return True
    return False


min_num_figures = 2
max_num_figures = 7
min_edge = .1
max_edge = 5.
min_field = -10.
max_field = 10.
output_folder = '../assets/'
num_maps = 3

for map_id in range(num_maps):
    figures = []
    all_lines = []
    num_figures = np.random.randint(min_num_figures, max_num_figures, 1)[0]
    while len(figures) < num_figures:
        num_lines = np.random.randint(2, 9, 1)[0]
        lines = []
        while len(lines) < num_lines:
            if len(lines) == 0:
                initial = np.random.uniform(min_field, max_field, 2)
                direction = np.random.uniform(min_field, max_field, 2)
                step = direction * np.random.uniform(min_edge, max_edge, 1) / np.linalg.norm(direction)
                second = initial + step
                line = Line(initial[0] + initial[1] * 1j, second[0] + second[1] * 1j)
            else:
                prev_line = lines[-1]
                direction = np.random.uniform(min_field, max_field, 2)
                step = direction * np.random.uniform(min_edge, max_edge, 1) / np.linalg.norm(direction)
                new_line_end = (prev_line.end.real, prev_line.end.imag) + step
                line = Line(prev_line.end, new_line_end[0] + new_line_end[1] * 1j)

            if len(lines) == 0 or not self_intersects(line, all_lines):
                lines.append(line)
                all_lines.append(line)
            else:
                pass

        final_line = Line(lines[-1].end, lines[0].start)
        if not self_intersects(final_line, all_lines):
            lines.append(final_line)
            figures.append(Path(*lines))
            all_lines.append(final_line)
            wsvg(paths=figures,
                 filename=os.path.join(output_folder, f"map_{map_id}.svg"),
                 dimensions=(10, 10))
            print(len(figures))
