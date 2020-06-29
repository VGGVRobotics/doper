import argparse
import yaml
import os
import logging

import numpy as onp
from jax.config import config as jax_config

from doper import trainers, agents, scenes
from doper.utils.loggers import write_output

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, help="path to config .yaml")
    args = parser.parse_args()
    config_path = args.config

    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    device = config["train"]["device"]
    if "cpu" in device:
        jax_config.update('jax_platform_name', 'cpu')
    elif "cuda" in device:
        jax_config.update('jax_platform_name', 'gpu')
    else:
        raise ValueError(f"Device {device} is not available, use cuda or cpu")

    output_folder = config["train"]["output_folder"]
    os.makedirs(output_folder, exist_ok=True)

    logging.basicConfig(
        filename=os.path.join(output_folder, "log.txt"),
        filemode="w",
        format="%(levelname)s - %(name)s - %(asctime)s:\t%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=getattr(logging, config["train"]["logging_level"].upper()),
    )

    logger = logging.getLogger("root")

    trainer = getattr(trainers, config["train"]["trainer_name"])(config)
    agent = getattr(agents, config["sim"]["agent_name"])(config)
    scene_handler = getattr(scenes, config["sim"]["scene_name"])(config)

    for iteration in range(config["train"]["n_iters"]):
        loss_val = trainer.optimize_parameters(agent, scene_handler)
        logger.info(f"Iteration {iteration}: Loss {loss_val}")
        if iteration % config["train"]["val_iter"] == 0:
            init_state = scene_handler.get_init_state(1)[[0]]
            velocity_init = onp.zeros_like(init_state)
            direction_init = onp.random.uniform(-1, 1, (1, 2))
            direction_init /= onp.linalg.norm(direction_init, axis=1)[:, None]
            trajectories = []
            for i in range(trainer.num_actions):
                observation = agent.get_observations(init_state, velocity_init, scene_handler)
                final_coordinate, velocity, trajectory = trainer.forward(
                    observation, init_state, velocity_init, direction_init, scene_handler
                )
                init_state = final_coordinate.reshape(1, -1)
                velocity_init = velocity.reshape(1, -1)
                trajectories.append(trajectory.coordinate)
            write_output(
                onp.concatenate(trajectories),
                os.path.join(output_folder, f"trajectory_iter_{iteration}.jpg"),
                scene_handler,
                config,
            )
