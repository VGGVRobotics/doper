import os

import numpy as onp
import matplotlib.pyplot as plt
from matplotlib import collections as mc


def write_output(
    trajectory: onp.ndarray, output_file_name: os.PathLike, scene_handler: object, config: dict
):
    """
    Writes images with trajectories and scene
    Args:
        trajectory: [steps, 2] numpy.ndarray with agents trajectory
        output_file_name: name of the output image
        scene_handler: scene handler object, containing the scene
        config: configuration dict
    """
    lines = mc.LineCollection(onp.array(scene_handler.jax_scene.segments))
    fig, ax = plt.subplots()

    ax.add_collection(lines)
    ax.scatter(trajectory[0, 0], trajectory[0, 1], c="g", label="init")
    ax.scatter(
        config["sim"]["attractor_coordinate"][0],
        config["sim"]["attractor_coordinate"][1],
        c="b",
        label="attractor",
    )
    ax.scatter(
        config["sim"]["coordinate_target"][0],
        config["sim"]["coordinate_target"][1],
        c="r",
        label="target",
    )
    ax.plot(trajectory[:, 0], trajectory[:, 1], c="black", label="trajectory")
    fig.legend()
    plt.savefig(output_file_name)
    plt.close()
