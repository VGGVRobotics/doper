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


def log_val_rollout(val_runner, agent, val_scene_handler, output_folder, iteration, config):
    save_dir = os.path.join(output_folder, f"iteration_{iteration}")
    for scene_num, scene_handler in enumerate(val_scene_handler):
        os.makedirs(save_dir, exist_ok=True)
        init_state = scene_handler.get_init_state(1)[[0]]
        velocity_init = onp.zeros_like(init_state)
        trajectories = []
        for i in range(val_runner.num_actions):
            observation = agent.get_observations(init_state, velocity_init, scene_handler)
            final_coordinate, velocity, trajectory = val_runner.forward(
                observation, init_state, velocity_init, scene_handler
            )
            init_state = final_coordinate.reshape(1, -1)
            velocity_init = velocity.reshape(1, -1)
            trajectories.append(trajectory.coordinate)
        write_output(
            onp.concatenate(trajectories),
            os.path.join(save_dir, f"scene_{scene_num}_trajectory_iter_{iteration}.jpg"),
            scene_handler,
            config,
        )
