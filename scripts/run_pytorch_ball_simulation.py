import taichi as ti
import numpy as np

from doper.sim import RollingBallPytorchSim


ti.init(arch=ti.cpu, default_fp=ti.f32)

if __name__ == "__main__":

    gui = ti.GUI("ball", (256, 256))

    constants = {
        "radius": 0.05,
        "g": 9.8,
        "f": 0.007,
        "ro": 1000,
        "obstacles_elasticity": 0.8,
    }
    constants["volume"] = 4 * np.pi * (constants["radius"] ** 3) / 3
    constants["mass"] = constants["volume"] * constants["ro"]

    sim = RollingBallPytorchSim(
        constants=constants,
        sim_steps=20,
        max_time=0.1,
        world_scale_coeff=10,
        grid_resolution=(512, 512),
        gui=gui,
        output_folder="./output",
    )

    with ti.Tape(sim.loss):
        sim.run_simulation(
            initial_coordinate=[0.2, 0.5],
            attraction_coordinate=[0.5, 0.5],
            initial_speed=[0.0, 7.0],
            visualize=False,
        )
        print(sim.initial_coordinate.grad[None][0])
