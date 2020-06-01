import argparse
import yaml
import os
import logging

from doper import trainers

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, help="path to config .yaml")
    args = parser.parse_args()
    config_path = args.config

    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    device = config['train']['device']
    output_folder = config["train"]["output_folder"]
    os.makedirs(output_folder, exist_ok=True)

    logging.basicConfig(
        filename=os.path.join(output_folder, "log.txt"),
        filemode="w",
        format="%(levelname)s - %(name)s - %(asctime)s:\t%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=getattr(logging, config["train"]["logging_level"].upper()),
    )

    logger = logging.getLogger('root')

    trainer = getattr(trainers, config["train"]["trainer_name"])(config)

    for iteration in range(config["train"]["n_iters"]):
        init_state = trainer.get_init_state()
        observation = trainer.get_observations(init_state)
        loss_val = trainer.optimize_parameters(observation, init_state)
        logger.info(f'Iteration {iteration}: Loss {loss_val}')
        if iteration % config["train"]["val_iter"] == 0:
            init_state = trainer.get_init_state()
            observation = trainer.get_observations(init_state)
            velocity_init, trajectory = trainer.forward(observation, init_state)
            trainer.write_output(
                trajectory, os.path.join(output_folder, f"trajectory_iter_{iteration}.jpg")
            )
