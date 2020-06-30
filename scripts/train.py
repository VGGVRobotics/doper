import argparse
import yaml
import os
import logging

import numpy as onp
from jax.config import config as jax_config

from doper import trainers, agents, scenes
from doper.utils.loggers import write_output, log_val_rollout

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, help="path to config .yaml")
    args = parser.parse_args()
    config_path = args.config

    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    device = config["train"]["device"]
    if "cpu" in device:
        jax_config.update("jax_platform_name", "cpu")
    elif "cuda" in device:
        jax_config.update("jax_platform_name", "gpu")
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
    scene_handler = getattr(scenes, config["sim"]["train_scene_name"])(
        config["sim"]["train_scene_params"]
    )
    val_scene_handler = getattr(scenes, config["sim"]["val_scene_name"])(
        config["sim"]["val_scene_params"]
    )
    for iteration in range(config["train"]["n_iters"]):
        loss_val = trainer.optimize_parameters(agent, scene_handler)
        logger.info(f"Iteration {iteration}: Loss {loss_val}")
        if iteration % config["train"]["val_iter"] == 0:
            log_val_rollout(trainer, agent, val_scene_handler, output_folder, iteration, config)
