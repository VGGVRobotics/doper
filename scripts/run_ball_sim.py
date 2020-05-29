from time import time

import matplotlib.pyplot as plt
import numpy as onp
import jax.numpy as np
from jax import value_and_grad

from doper.sim.ball_sim import compute_loss, run_sim

constants = {}
constants['radius'] = 0.05
constants['ro'] = 1000.
constants['volume'] = 4 * np.pi * (constants['radius'] ** 3) / 3
constants['mass'] = constants['volume'] * constants['ro']
constants['f'] = 0.007

target_coordinate = np.array([0.5, 0.5])
coordinate_init = np.array([0.2, 0.4])
velocity_init = np.array([1., 0.])
attractor = np.array([0., 0.])

velocity_init = np.array([1., 0.])
for step in range(100):
    lr = 0.5
    s = time()
    loss_val, v_grad = value_and_grad(compute_loss, 1)(coordinate_init, velocity_init, target_coordinate, attractor, constants)
    velocity_init -= lr * v_grad
    print(time() - s, loss_val, velocity_init, v_grad)

final_coordinate, trajectory = run_sim(coordinate_init, velocity_init, attractor, constants)
fig, ax = plt.subplots()
traj = onp.array(trajectory)
ax.plot(traj[:, 0], traj[:, 1])
ax.scatter(attractor[0], attractor[1], c='b')
ax.scatter(target_coordinate[0], target_coordinate[1], c='r')
plt.savefig('trajectory.jpg')
