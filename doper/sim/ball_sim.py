from collections import namedtuple
from functools import partial

import jax
import jax.numpy as np
from jax import grad
from jax import lax

from .jax_geometry import find_closest_segment_to_point, compute_segment_projection

BallState = namedtuple("BallState", ["coordinate", "velocity", "acceleration"])
Constants = namedtuple("SimulationConstants", ["coordinate", "velocity", "acceleration"])


def compute_potential_point(coord, attractor_coord):
    return np.sum((coord - attractor_coord) ** 2)


def compute_rolling_friction_force(velocity, mass, radius, f, g=9.8):
    return (
        -np.clip(velocity, -1, 1) * mass * g * f / radius
    )  # ugly hack to reduce friction after slowing down


def compute_acceleration(potential_force, friction_force, mass):
    return (potential_force + friction_force) / mass


def get_new_state(current_state, new_acceleration, dt):
    new_velocity = current_state.velocity + new_acceleration * dt
    new_coordinate = current_state.coordinate + new_velocity * dt
    return BallState(
        coordinate=new_coordinate, velocity=new_velocity, acceleration=new_acceleration
    )


def resolve_collision(args):
    current_state, new_state, attractor, closest_segment, constants, dt = args
    # find toi
    closest_segment_point, _ = compute_segment_projection(new_state.coordinate, closest_segment)
    old_obstacle_vector = closest_segment_point - current_state.coordinate
    old_obstacle_distance = np.linalg.norm(old_obstacle_vector)
    old_obstacle_direction = old_obstacle_vector / old_obstacle_distance

    toi = np.abs(old_obstacle_distance - constants["radius"]) / np.abs(
        old_obstacle_direction.dot(new_state.velocity)
    )
    # looks suspicious but works pretty ok
    toi = toi.clip(0, dt)
    # make toi step
    sub_state = get_new_state(current_state, new_state.acceleration, toi)
    velocity_at_impact = sub_state.velocity
    # find velocity after impact
    closest_segment_point, _ = compute_segment_projection(sub_state.coordinate, closest_segment)
    obstacle_vector = closest_segment_point - sub_state.coordinate
    obstacle_distance = np.linalg.norm(obstacle_vector)
    obstacle_direction = obstacle_vector / obstacle_distance
    projected_v_n = obstacle_direction * obstacle_direction.dot(velocity_at_impact)
    projected_v_p = velocity_at_impact - projected_v_n
    velocity_after_impact = projected_v_p - constants["walls_elasticity"] * projected_v_n
    # make final step
    state_after_impact = BallState(
        sub_state.coordinate, velocity_after_impact, sub_state.acceleration
    )
    l2_force = -grad(compute_potential_point)(sub_state.coordinate, attractor)
    friction_force = compute_rolling_friction_force(
        state_after_impact.velocity,
        constants["mass"],
        constants["radius"],
        constants["rolling_friction_coefficient"],
    )
    new_acceleration = compute_acceleration(l2_force, friction_force, constants["mass"])
    return get_new_state(state_after_impact, new_acceleration, dt - toi)


def collide(current_state, new_state, attractor, closest_segment, distance, constants, dt):
    # debug
    # if distance <= constants["radius"]:
    #     return resolve_collision(
    #         (current_state, new_state, attractor, closest_segment, constants, dt)
    #     )
    # else:
    #     return new_state
    return lax.cond(
        distance <= constants["radius"],
        true_fun=resolve_collision,
        false_fun=lambda s: s[1],
        operand=(current_state, new_state, attractor, closest_segment, constants, dt),
    )


def sim_step(current_state, t, scene, attractor, constants, dt):
    l2_force = -grad(compute_potential_point)(current_state.coordinate, attractor)
    friction_force = compute_rolling_friction_force(
        current_state.velocity,
        constants["mass"],
        constants["radius"],
        constants["rolling_friction_coefficient"],
    )
    acceleration = compute_acceleration(
        constants["attractor_strength"] * l2_force, friction_force, constants["mass"]
    )
    may_be_state = get_new_state(current_state, acceleration, dt)
    closest_segment, distance = find_closest_segment_to_point(
        may_be_state.coordinate, scene.segments
    )
    current_state = collide(
        current_state, may_be_state, attractor, closest_segment, distance, constants, dt,
    )
    return current_state, current_state


def _run_sim(
    sim_time, n_steps, scene, coordinate_init, velocity_init, attractor, constants,
):
    dt = sim_time / n_steps
    current_state = BallState(
        coordinate=coordinate_init,
        velocity=velocity_init,
        acceleration=np.zeros_like(velocity_init),
    )
    step = partial(sim_step, scene=scene, attractor=attractor, constants=constants, dt=dt,)

    current_state, trajectory = lax.scan(step, init=current_state, xs=None, length=n_steps)
    # useful to have this loop for no jit debug
    # trajectory = []
    # for i in range(n_steps):
    #     # print(current_state)
    #     trajectory.append(current_state)
    #     current_state, _ = step(current_state, i)

    return current_state.coordinate, current_state.velocity, trajectory


def _compute_loss(
    sim_time,
    n_steps,
    scene,
    coordinate_init,
    velocity_init,
    target_coordinate,
    attractor,
    constants,
):
    final_coord, final_velocity, trajectory = run_sim(
        sim_time, n_steps, scene, coordinate_init, velocity_init, attractor, constants,
    )
    return np.sum(np.abs(final_coord - target_coordinate)), final_coord, final_velocity


def reduce_loss(s, n, sc, c, v, t, a, constants):
    loss_out = vmapped_loss(s, n, sc, c, v, t, a, constants)
    return np.sum(loss_out[0]), loss_out[1:]


run_sim = jax.jit(_run_sim, static_argnums=(0, 1))

compute_loss = jax.jit(_compute_loss, static_argnums=(0, 1))

vmapped_loss = jax.vmap(compute_loss, in_axes=(None, None, None, 0, 0, None, None, None))

vmapped_grad_and_value = jax.value_and_grad(
    lambda s, n, sc, c, v, t, a, constants: reduce_loss(s, n, sc, c, v, t, a, constants),
    4,
    has_aux=True,
)
