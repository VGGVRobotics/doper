from collections import namedtuple
from functools import partial
from typing import Sequence, Tuple, Union, List

import jax
import jax.numpy as np
from jax import grad
from jax import lax

from .jax_geometry import find_closest_segment_to_point, compute_segment_projection, JaxScene

BallState = namedtuple("BallState", ["coordinate", "velocity", "acceleration", "direction"])


def compute_potential_point(coord: Sequence, attractor_coord: Sequence) -> float:
    """
    Computes L2 potential at a given point
    Args:
        coord: jax array, current coordinate
        attractor_coord: jax array, center of gaussian

    Returns:
        Value of the potential energy
    """
    return np.sum((coord - attractor_coord) ** 2)


def compute_rolling_friction_force(
    velocity: Sequence, mass: float, radius: float, friction_coeff: float, g: float = 9.8
) -> Sequence:
    """
    Reutrns
    Args:
        velocity: current agents velocity
        mass: agents mass
        radius: agents radius for rolling friction
        friction_coeff: rolling friction coefficient
        g: free fall force constant

    Returns:
        jax array with friction force
    """
    return (
        -np.clip(velocity, -1, 1) * mass * g * friction_coeff / radius
    )  # ugly hack to reduce friction after slowing down


def compute_acceleration(
    potential_force: Sequence, friction_force: Sequence, mass: float
) -> Sequence:
    """
    Computes acceleration given potential and frequency forces
    Args:
        potential_force: jax array with potential force vector
        friction_force: jax array with friction force vector
        mass: agents mass

    Returns:
        jax array with agents acceleration
    """
    return (potential_force + friction_force) / mass


def get_new_state(current_state: BallState, new_acceleration: Sequence, dt: float) -> BallState:
    """
    Returns new agents state
    Args:
        current_state: named tuple BallState with the current state
        new_acceleration: jax array with current acceleration
        dt: time step

    Returns:
        named tuple BallState with new state
    """
    tangent_acceleration = current_state.direction * (new_acceleration @ current_state.direction)
    new_velocity = current_state.velocity + tangent_acceleration * dt
    tangent_velocity = current_state.direction * (new_velocity @ current_state.direction)
    new_coordinate = current_state.coordinate + tangent_velocity * dt
    return BallState(
        coordinate=new_coordinate,
        velocity=new_velocity,
        acceleration=new_acceleration,
        direction=current_state.direction
    )


def resolve_collision(args: Sequence):
    """
    Resolves collision
    Args:
        args:
          current_state: named tuple BallState with the current state
          new_state: named tuple BallState with the new state
          attractor: jax array with L2 attractor coordinate
          closest_segment: closest segment to compute projection
          constants: dict with physical constants
          dt: time step

    Returns:
        named tuple BallState with new state
    """
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
        sub_state.coordinate, velocity_after_impact, sub_state.acceleration, current_state.direction
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


def collide(
    current_state: BallState,
    new_state: BallState,
    attractor: Sequence,
    closest_segment: jax.numpy.ndarray,
    distance: float,
    constants: dict,
    dt: float,
):
    """
    Performs collision check and handling
    Args:
        current_state: named tuple BallState with the current state
        new_state: named tuple BallState with the new state
        attractor: jax array with L2 attractor coordinate
        closest_segment: closest segment to compute projection
        distance: distance to the nearest obstacle
        constants: dict with physical constants
        dt: time step

    """
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


def sim_step(
    current_state: BallState,
    t: float,
    scene: JaxScene,
    attractor: Sequence,
    constants: dict,
    dt: float,
) -> Tuple[BallState, BallState]:
    """
    Performs singe simulation step
    Args:
        current_state: named tuple BallState with the current state
        t: fake variable needed for lax.scan
        scene: JaxScene object with scene geometry
        attractor: jax array with L2 attractor coordinate
        constants: dict with physical constants
        dt: time step

    Returns:
        Final state and final state needed for trajectory aggregation by lax.scan
    """
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
    sim_time: Union[float, int],
    n_steps: int,
    scene: JaxScene,
    coordinate_init: Sequence,
    velocity_init: Sequence,
    direction_init: Sequence,
    attractor: Sequence,
    constants: dict,
) -> Tuple[Sequence, Sequence, Sequence]:
    """
    Wrapper function for lax.scan cycle through simulation steps
    Args:
        sim_time: Time in seconds per simulation
        n_steps: Number of steps per simulation
        scene: JaxScene object with scene geometry
        coordinate_init: jax array with initial agents coordinate
        velocity_init: jax array with initial agents velocity
        attractor: jax array with attractor coordinates
        constants: Physical constants

    Returns:
        jax array with final coordinate [batch_size, 2]
        jax array with final velocity [batch_size, 2]
        jax array with agents trajectory [batch_size, n_steps]
    """
    dt = sim_time / n_steps
    current_state = BallState(
        coordinate=coordinate_init,
        velocity=velocity_init,
        acceleration=np.zeros_like(velocity_init),
        direction=direction_init
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
    sim_time: Union[float, int],
    n_steps: int,
    scene: JaxScene,
    coordinate_init: Sequence,
    velocity_init: Sequence,
    direction: Sequence,
    target_coordinate: Sequence,
    attractor: Sequence,
    constants: dict,
) -> Tuple[Sequence, Sequence, Sequence]:
    """
    Wrapper function for computing gradients through the simulation
    Args:
        sim_time: Time in seconds per simulation
        n_steps: Number of steps per simulation
        scene: JaxScene object with scene geometry
        coordinate_init: jax array with initial agents coordinate
        velocity_init: jax array with initial agents velocity
        target_coordinate: jax array with the target for the agent
        attractor: jax array with attractor coordinates
        constants: Physical constants

    Returns:
        loss: [batch_size, 1] jax array of losses
        final_coordinate: [batch_size, 2] jax array of final coordinates
        final_velocity: [batch_size, 2] jax array of final velocities
    """
    final_coord, final_velocity, trajectory = run_sim(
        sim_time, n_steps, scene, coordinate_init, velocity_init, direction, attractor, constants,
    )
    return np.sum(np.abs(final_coord - target_coordinate)), final_coord, final_velocity


def reduce_loss(
    sim_time: Union[float, int],
    n_steps: int,
    scene: JaxScene,
    coordinate_init: Sequence,
    velocity_init: Sequence,
    direction: Sequence,
    target_coordinate: Sequence,
    attractor: Sequence,
    constants: dict,
) -> Tuple[float, List[Sequence]]:
    """
    Wrapper function that reduces loss across batch in order to produce scalar from the vmapped function
    Args:
        sim_time: Time in seconds per simulation
        n_steps: Number of steps per simulation
        scene: JaxScene object with scene geometry
        coordinate_init: jax array with initial agents coordinate
        velocity_init: jax array with initial agents velocity
        target_coordinate: jax array with the target for the agent
        attractor: jax array with attractor coordinates
        constants: Physical constants

    Returns:
        Single float loss for gradient and tuple with auxilary values
    """
    loss_out = vmapped_loss(
        sim_time,
        n_steps,
        scene,
        coordinate_init,
        velocity_init,
        direction,
        target_coordinate,
        attractor,
        constants,
    )
    return np.sum(loss_out[0]), loss_out[1:]


run_sim = jax.jit(_run_sim, static_argnums=(0, 1))

compute_loss = jax.jit(_compute_loss, static_argnums=(0, 1))

#  only coordinate_init and velocity_init are to be vectorized, everything else is to be broadcasted
vmapped_loss = jax.vmap(compute_loss, in_axes=(None, None, None, 0, 0, None, None, None, None))

vmapped_grad_and_value = jax.value_and_grad(
    lambda s, n, sc, c, v, dir, t, a, constants: reduce_loss(s, n, sc, c, v, dir, t, a, constants),
    4,
    has_aux=True,
)
