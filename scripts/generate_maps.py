import os

import numpy as onp
from svgpathtools import Line, Path, wsvg
from multiprocessing import Pool

def self_intersects(line, lines):
    for l in lines:
        intersection = line.intersect(l)
        if len(intersection) == 0:
            pass
        else:
            return True
    return False


min_num_figures = 2
max_num_figures = 20
min_edge = .1
max_edge = 5.
min_field = -10.
max_field = 10.
output_folder = '../assets/'
num_maps = 100

def generate_maps(map_id):
    onp.random.seed(map_id)
    figures = []
    all_lines = []
    num_figures = onp.random.randint(min_num_figures, max_num_figures, 1)[0]
    print(map_id, num_figures)
    while len(figures) < num_figures:
        num_lines = onp.random.randint(2, 9, 1)[0]
        lines = []
        while len(lines) < num_lines:
            if len(lines) == 0:
                initial = onp.random.uniform(min_field, max_field, 2)
                direction = onp.random.uniform(min_field, max_field, 2)
                step = direction * onp.random.uniform(min_edge, max_edge, 1) / onp.linalg.norm(direction)
                second = initial + step
                line = Line(initial[0] + initial[1] * 1j, second[0] + second[1] * 1j)
            else:
                prev_line = lines[-1]
                direction = onp.random.uniform(min_field, max_field, 2)
                step = direction * onp.random.uniform(min_edge, max_edge, 1) / onp.linalg.norm(direction)
                new_line_end = (prev_line.end.real, prev_line.end.imag) + step
                line = Line(prev_line.end, new_line_end[0] + new_line_end[1] * 1j)

            if len(lines) == 0 or not self_intersects(line, all_lines):
                lines.append(line)
                all_lines.append(line)
            else:
                pass

        final_line = Line(lines[-1].end, lines[0].start)
        # TODO do not discard figure, add more points instead
        if not self_intersects(final_line, all_lines):
            lines.append(final_line)
            figures.append(Path(*lines))
            all_lines.append(final_line)
            wsvg(paths=figures,
                 filename=os.path.join(output_folder, f"map_{map_id}.svg"),
                 dimensions=(10, 10))


if __name__ == '__main__':
    with Pool(16) as pool:
        pool.map(generate_maps, range(num_maps))
