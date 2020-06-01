from time import time

import matplotlib.pyplot as plt
from matplotlib import collections as mc
import numpy as onp
import jax.numpy as np
from jax import value_and_grad

from doper.sim.ball_sim import compute_loss, run_sim
from doper.world.assets import get_svg_scene
from doper.sim.jax_geometry import JaxScene

constants = {}
constants["radius"] = 0.05
constants["ro"] = 1000.0
constants["volume"] = 4 * np.pi * (constants["radius"] ** 3) / 3
constants["mass"] = constants["volume"] * constants["ro"]
constants["f"] = 0.007
constants["elasticity"] = 0.5
n_steps = 2000.0
sim_time = 2.0
target_coordinate = np.array([3.0, 2.5])
coordinate_init = np.array([0.2, 0.4])
velocity_init = np.array([0.0, 0.0])
attractor = np.array([4.0, 3.5])
scene = get_svg_scene("../assets/simple_level.svg", px_per_meter=100)
jax_scene = JaxScene(segments=np.array(scene.get_all_segments()))
for step in range(2000):
    lr = 0.1
    if step > 8000:
        lr = 0.05
    s = time()
    loss_val, v_grad = value_and_grad(compute_loss, 4)(
        sim_time,
        n_steps,
        jax_scene,
        coordinate_init,
        velocity_init,
        target_coordinate,
        attractor,
        constants,
    )
    velocity_init -= lr * np.clip(v_grad, -10, 10)
    print(time() - s, loss_val, velocity_init, v_grad, coordinate_init)

final_coordinate, trajectory = run_sim(
    sim_time, n_steps, jax_scene, coordinate_init, velocity_init, attractor, constants
)
lines = mc.LineCollection(scene.get_all_segments())
fig, ax = plt.subplots()

ax.add_collection(lines)
traj = onp.array(trajectory.coordinate)
ax.scatter(coordinate_init[0], coordinate_init[1], c="g")
ax.plot(traj[:, 0], traj[:, 1])
ax.scatter(attractor[0], attractor[1], c="b")
ax.scatter(target_coordinate[0], target_coordinate[1], c="r")
plt.savefig(f"trajectory.jpg")
