import jax.numpy as np
from jax import jit, grad

@jit
def compute_potential_point(coord, target_coord):
    return np.sum((coord - target_coord) ** 2)

@jit
def compute_rolling_friction_force(velocity, mass, radius, f, g=9.8):
    return - np.sign(velocity) * mass * g * radius * f / radius

@jit
def compute_acceleration(potential_force, friction_force, mass):
    return (potential_force + friction_force) / mass

@jit
def get_new_cv(current_coordinate, current_velocity, acceleration, dt):
    new_velocity = current_velocity + acceleration * dt
    new_coordinate = current_coordinate + new_velocity * dt
    return new_coordinate, new_velocity

@jit
def run_sim(coordinate_init, velocity_init, target_coordinate, constants):
    trajectory = []
    sim_time = 0.2
    n_steps = 20
    dt = sim_time / n_steps
    coordinate = coordinate_init
    velocity = velocity_init
    for t in np.linspace(0, sim_time, n_steps):
        trajectory.append(coordinate)
        l2_force = - grad(compute_potential_point)(coordinate, target_coordinate)
        friction_force = compute_rolling_friction_force(velocity,
                                                        constants['mass'],
                                                        constants['radius'],
                                                        constants['f'])
        acceleration = compute_acceleration(l2_force,
                                            friction_force,
                                            constants['mass'])
        coordinate, velocity = get_new_cv(coordinate, velocity, acceleration, dt)
    return coordinate, trajectory

@jit
def compute_loss(coordinate_init, velocity_init, target_coordinate, attractor, constants):
    final_coord, _ = run_sim(coordinate_init, velocity_init, attractor, constants)
    return np.sum(np.abs(final_coord - target_coordinate))
