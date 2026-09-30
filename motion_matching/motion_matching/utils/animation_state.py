"""Motion matching algorithm helpers: inertialization and trajectory/contact/query utilities."""

from dataclasses import dataclass

from typing import Any, Tuple

import numpy as np
import motion_matching.utils.math_utils as math


# ----------------------------- Inertialization -----------------------------


@dataclass
class InertializerState:
    """Persistent per-character inertializer state.

    Bundles the eight arrays that ``inertialize_pose_reset`` /
    ``inertialize_pose_transition`` / ``inertialize_pose_update`` read and mutate
    in place. They are created together and always passed together, so grouping
    them removes long, error-prone positional/keyword argument lists at the call
    sites. The arrays are mutated in place; this object holds references to them.

    - ``bone_offset_*``: ``(nbones, .)`` decaying spring-damper offsets per bone.
    - ``transition_{src,dst}_*``: the source/destination root frame recorded at
      the most recent transition.
    """

    bone_offset_positions: np.ndarray
    bone_offset_velocities: np.ndarray
    bone_offset_rotations: np.ndarray
    bone_offset_angular_velocities: np.ndarray
    transition_src_position: np.ndarray
    transition_src_rotation: np.ndarray
    transition_dst_position: np.ndarray
    transition_dst_rotation: np.ndarray

    @classmethod
    def create(cls, nbones: int) -> "InertializerState":
        """Allocate zeroed state for ``nbones`` bones (identity quats for rotations)."""
        bone_offset_rotations = np.zeros((nbones, 4), dtype=np.float32)
        bone_offset_rotations[:, 0] = 1.0  # Identity quaternions
        return cls(
            bone_offset_positions=np.zeros((nbones, 3), dtype=np.float32),
            bone_offset_velocities=np.zeros((nbones, 3), dtype=np.float32),
            bone_offset_rotations=bone_offset_rotations,
            bone_offset_angular_velocities=np.zeros((nbones, 3), dtype=np.float32),
            transition_src_position=np.zeros(3, dtype=np.float32),
            transition_src_rotation=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            transition_dst_position=np.zeros(3, dtype=np.float32),
            transition_dst_rotation=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        )


def decay_spring_damper_exact_vec3(x: np.ndarray, v: np.ndarray, halflife: float, dt: float) -> None:
    """Exact spring damper decay for vec3"""
    y = math.halflife_to_damping(halflife) / 2.0
    j0 = x.copy()
    j1 = v + j0 * y
    eydt = math.fast_negexpf(y * dt)

    x[:] = eydt * (j0 + j1 * dt)
    v[:] = eydt * (v - j1 * y * dt)


def decay_spring_damper_exact_quat(x: np.ndarray, v: np.ndarray, halflife: float, dt: float) -> None:
    """Exact spring damper decay for quaternions"""
    y = math.halflife_to_damping(halflife) / 2.0

    j0 = math.quat_to_scaled_angle_axis(x)
    j1 = v + j0 * y

    eydt = math.fast_negexpf(y * dt)

    x[:] = math.quat_from_scaled_angle_axis(eydt * (j0 + j1 * dt))
    v[:] = eydt * (v - j1 * y * dt)


def inertialize_transition_vec3(
    off_x: np.ndarray, off_v: np.ndarray, src_x: np.ndarray, src_v: np.ndarray, dst_x: np.ndarray, dst_v: np.ndarray
) -> None:
    """Transition inertializer for vec3"""
    off_x[:] = (src_x + off_x) - dst_x
    off_v[:] = (src_v + off_v) - dst_v


def inertialize_update_vec3(
    out_x: np.ndarray,
    out_v: np.ndarray,
    off_x: np.ndarray,
    off_v: np.ndarray,
    in_x: np.ndarray,
    in_v: np.ndarray,
    halflife: float,
    dt: float,
) -> None:
    """Update inertializer for vec3"""
    decay_spring_damper_exact_vec3(x=off_x, v=off_v, halflife=halflife, dt=dt)
    out_x[:] = in_x + off_x
    out_v[:] = in_v + off_v


def inertialize_transition_quat(
    off_x: np.ndarray, off_v: np.ndarray, src_x: np.ndarray, src_v: np.ndarray, dst_x: np.ndarray, dst_v: np.ndarray
) -> None:
    """Transition inertializer for quaternions"""
    off_x[:] = math.quat_abs(math.quat_mul(math.quat_mul(off_x, src_x), math.quat_inv(dst_x)))
    off_v[:] = (off_v + src_v) - dst_v


def inertialize_update_quat(
    out_x: np.ndarray,
    out_v: np.ndarray,
    off_x: np.ndarray,
    off_v: np.ndarray,
    in_x: np.ndarray,
    in_v: np.ndarray,
    halflife: float,
    dt: float,
) -> None:
    """Update inertializer for quaternions"""
    decay_spring_damper_exact_quat(x=off_x, v=off_v, halflife=halflife, dt=dt)
    out_x[:] = math.quat_mul(off_x, in_x)
    out_v[:] = off_v + math.quat_mul_vec3(off_x, in_v)


def inertialize_pose_reset(state: InertializerState, root_position: np.ndarray, root_rotation: np.ndarray) -> None:
    """Reset inertializer pose state"""
    bone_offset_positions = state.bone_offset_positions
    bone_offset_velocities = state.bone_offset_velocities
    bone_offset_rotations = state.bone_offset_rotations
    bone_offset_angular_velocities = state.bone_offset_angular_velocities
    transition_src_position = state.transition_src_position
    transition_src_rotation = state.transition_src_rotation
    transition_dst_position = state.transition_dst_position
    transition_dst_rotation = state.transition_dst_rotation

    bone_offset_positions.fill(0.0)
    bone_offset_velocities.fill(0.0)
    bone_offset_rotations.fill(0.0)
    bone_offset_rotations[:, 0] = 1.0  # Set w component to 1 for identity quaternions
    bone_offset_angular_velocities.fill(0.0)

    transition_src_position[:] = root_position
    transition_src_rotation[:] = root_rotation
    transition_dst_position.fill(0.0)
    transition_dst_rotation.fill(0.0)
    transition_dst_rotation[0] = 1.0  # Set w component to 1 for identity quaternion


def inertialize_pose_transition(
    state: InertializerState,
    root_position: np.ndarray,
    root_velocity: np.ndarray,
    root_rotation: np.ndarray,
    root_angular_velocity: np.ndarray,
    bone_src_positions: np.ndarray,
    bone_src_velocities: np.ndarray,
    bone_src_rotations: np.ndarray,
    bone_src_angular_velocities: np.ndarray,
    bone_dst_positions: np.ndarray,
    bone_dst_velocities: np.ndarray,
    bone_dst_rotations: np.ndarray,
    bone_dst_angular_velocities: np.ndarray,
) -> None:
    """Transition the inertializer for the full character"""
    bone_offset_positions = state.bone_offset_positions
    bone_offset_velocities = state.bone_offset_velocities
    bone_offset_rotations = state.bone_offset_rotations
    bone_offset_angular_velocities = state.bone_offset_angular_velocities
    transition_src_position = state.transition_src_position
    transition_src_rotation = state.transition_src_rotation
    transition_dst_position = state.transition_dst_position
    transition_dst_rotation = state.transition_dst_rotation

    # Record the root position and rotation in the animation data
    transition_dst_position[:] = root_position
    transition_dst_rotation[:] = root_rotation
    transition_src_position[:] = bone_dst_positions[0]
    transition_src_rotation[:] = bone_dst_rotations[0]

    # Find the velocities so we can transition the root inertializers
    world_space_dst_velocity = math.quat_mul_vec3(
        transition_dst_rotation, math.quat_inv_mul_vec3(transition_src_rotation, bone_dst_velocities[0])
    )

    world_space_dst_angular_velocity = math.quat_mul_vec3(
        transition_dst_rotation, math.quat_inv_mul_vec3(transition_src_rotation, bone_dst_angular_velocities[0])
    )

    # Transition inertializers recording the offsets for the root joint
    inertialize_transition_vec3(
        off_x=bone_offset_positions[0],
        off_v=bone_offset_velocities[0],
        src_x=root_position,
        src_v=root_velocity,
        dst_x=root_position,
        dst_v=world_space_dst_velocity,
    )

    inertialize_transition_quat(
        off_x=bone_offset_rotations[0],
        off_v=bone_offset_angular_velocities[0],
        src_x=root_rotation,
        src_v=root_angular_velocity,
        dst_x=root_rotation,
        dst_v=world_space_dst_angular_velocity,
    )

    # Transition all the inertializers for each other bone
    for i in range(1, len(bone_offset_positions)):
        inertialize_transition_vec3(
            off_x=bone_offset_positions[i],
            off_v=bone_offset_velocities[i],
            src_x=bone_src_positions[i],
            src_v=bone_src_velocities[i],
            dst_x=bone_dst_positions[i],
            dst_v=bone_dst_velocities[i],
        )

        inertialize_transition_quat(
            off_x=bone_offset_rotations[i],
            off_v=bone_offset_angular_velocities[i],
            src_x=bone_src_rotations[i],
            src_v=bone_src_angular_velocities[i],
            dst_x=bone_dst_rotations[i],
            dst_v=bone_dst_angular_velocities[i],
        )


def inertialize_pose_update(
    bone_positions: np.ndarray,
    bone_velocities: np.ndarray,
    bone_rotations: np.ndarray,
    bone_angular_velocities: np.ndarray,
    state: InertializerState,
    bone_input_positions: np.ndarray,
    bone_input_velocities: np.ndarray,
    bone_input_rotations: np.ndarray,
    bone_input_angular_velocities: np.ndarray,
    halflife: float,
    dt: float,
) -> None:
    """Update the inertializer states"""
    bone_offset_positions = state.bone_offset_positions
    bone_offset_velocities = state.bone_offset_velocities
    bone_offset_rotations = state.bone_offset_rotations
    bone_offset_angular_velocities = state.bone_offset_angular_velocities
    transition_src_position = state.transition_src_position
    transition_src_rotation = state.transition_src_rotation
    transition_dst_position = state.transition_dst_position
    transition_dst_rotation = state.transition_dst_rotation

    world_space_position = (
        math.quat_mul_vec3(
            transition_dst_rotation,
            math.quat_inv_mul_vec3(transition_src_rotation, bone_input_positions[0] - transition_src_position),
        )
        + transition_dst_position
    )

    world_space_velocity = math.quat_mul_vec3(
        transition_dst_rotation, math.quat_inv_mul_vec3(transition_src_rotation, bone_input_velocities[0])
    )

    # Normalize here because quat inv mul can sometimes produce
    # unstable returns when the two rotations are very close.
    world_space_rotation = math.quat_normalize(
        math.quat_mul(transition_dst_rotation, math.quat_inv_mul(transition_src_rotation, bone_input_rotations[0]))
    )

    world_space_angular_velocity = math.quat_mul_vec3(
        transition_dst_rotation, math.quat_inv_mul_vec3(transition_src_rotation, bone_input_angular_velocities[0])
    )

    inertialize_update_vec3(
        out_x=bone_positions[0],
        out_v=bone_velocities[0],
        off_x=bone_offset_positions[0],
        off_v=bone_offset_velocities[0],
        in_x=world_space_position,
        in_v=world_space_velocity,
        halflife=halflife,
        dt=dt,
    )

    inertialize_update_quat(
        out_x=bone_rotations[0],
        out_v=bone_angular_velocities[0],
        off_x=bone_offset_rotations[0],
        off_v=bone_offset_angular_velocities[0],
        in_x=world_space_rotation,
        in_v=world_space_angular_velocity,
        halflife=halflife,
        dt=dt,
    )

    for i in range(1, len(bone_positions)):
        inertialize_update_vec3(
            out_x=bone_positions[i],
            out_v=bone_velocities[i],
            off_x=bone_offset_positions[i],
            off_v=bone_offset_velocities[i],
            in_x=bone_input_positions[i],
            in_v=bone_input_velocities[i],
            halflife=halflife,
            dt=dt,
        )

        inertialize_update_quat(
            out_x=bone_rotations[i],
            out_v=bone_angular_velocities[i],
            off_x=bone_offset_rotations[i],
            off_v=bone_offset_angular_velocities[i],
            in_x=bone_input_rotations[i],
            in_v=bone_input_angular_velocities[i],
            halflife=halflife,
            dt=dt,
        )


# --------------------- Database / trajectory / contact ---------------------


def database_trajectory_index_clamp(db: Any, frame: int, offset: int) -> int:
    """Clamp trajectory index to valid range - matches C++ implementation exactly"""
    for i in range(len(db.range_starts)):
        if frame >= db.range_starts[i] and frame < db.range_stops[i]:
            return math.clamp(value=frame + offset, min_val=db.range_starts[i], max_val=db.range_stops[i] - 1)

    assert False
    return -1


def simulation_rotations_update(
    x: np.ndarray, v: np.ndarray, x_goal: np.ndarray, halflife: float, dt: float
) -> Tuple[np.ndarray, np.ndarray]:
    y = math.halflife_to_damping(halflife) / 2.0

    j0 = math.quat_to_scaled_angle_axis(math.quat_abs(math.quat_mul(x, math.quat_inv(x_goal))))
    j1 = v + j0 * y

    eydt = math.fast_negexpf(y * dt)

    x = math.quat_mul(math.quat_from_scaled_angle_axis(eydt * (j0 + j1 * dt)), x_goal)
    v = eydt * (v - j1 * y * dt)
    return x, v


def orbit_camera_update_azimuth(
    azimuth: float, gamepadstick_right: np.ndarray, desired_strafe: bool, dt: float
) -> float:
    """Update camera azimuth based on gamepad input"""
    gamepadaxis = np.array([0.0, 0.0, 0.0]) if desired_strafe else gamepadstick_right
    return azimuth + 2.0 * dt * (-gamepadaxis[0])


def desired_velocity_update(
    gamepadstick_left: np.ndarray,
    camera_azimuth: float,
    simulation_rotation: np.ndarray,
    fwrd_speed: float,
    side_speed: float,
    back_speed: float,
) -> np.ndarray:
    """Update desired velocity based on gamepad input"""
    camera_rot = math.quat_from_angle_axis(camera_azimuth, np.array([0.0, 0.0, 1.0]))
    global_stick_direction = math.quat_mul_vec3(camera_rot, gamepadstick_left)

    local_stick_direction = math.quat_inv_mul_vec3(simulation_rotation, global_stick_direction)

    if local_stick_direction[0] > 0.0:
        local_desired_velocity = np.array([fwrd_speed, side_speed, 0.0]) * local_stick_direction
    else:
        local_desired_velocity = np.array([back_speed, side_speed, 0.0]) * local_stick_direction

    return math.quat_mul_vec3(simulation_rotation, local_desired_velocity)


def desired_rotation_update(
    desired_rotation: np.ndarray,
    gamepadstick_left: np.ndarray,
    gamepadstick_right: np.ndarray,
    camera_azimuth: float,
    desired_strafe: bool,
    desired_velocity: np.ndarray,
) -> np.ndarray:
    """Update desired rotation based on input"""
    desired_rotation_curr = desired_rotation.copy()

    if desired_strafe:
        desired_direction = math.quat_mul_vec3(
            math.quat_from_angle_axis(camera_azimuth, np.array([0.0, 0.0, 1.0])), np.array([-1.0, 0.0, 0.0])
        )

        if math.length(gamepadstick_right) > 0.01:
            desired_direction = math.quat_mul_vec3(
                math.quat_from_angle_axis(camera_azimuth, np.array([0.0, 0.0, 1.0])), math.normalize(gamepadstick_right)
            )

        return math.quat_from_angle_axis(
            np.arctan2(desired_direction[1], desired_direction[0]), np.array([0.0, 0.0, 1.0])
        )

    elif math.length(gamepadstick_left) > 0.01:
        desired_direction = math.normalize(desired_velocity)
        return math.quat_from_angle_axis(
            np.arctan2(desired_direction[1], desired_direction[0]), np.array([0.0, 0.0, 1.0])
        )

    else:
        return desired_rotation_curr


def trajectory_desired_rotations_predict(
    desired_rotations: np.ndarray,
    desired_velocities: np.ndarray,
    desired_rotation: np.ndarray,
    camera_azimuth: float,
    gamepadstick_left: np.ndarray,
    gamepadstick_right: np.ndarray,
    desired_strafe: bool,
    dt: float,
) -> None:
    """Predict desired rotations for future trajectory points"""
    desired_rotations[0] = desired_rotation

    for i in range(1, len(desired_rotations)):
        desired_rotations[i] = desired_rotation_update(
            desired_rotation=desired_rotations[i - 1],
            gamepadstick_left=gamepadstick_left,
            gamepadstick_right=gamepadstick_right,
            camera_azimuth=orbit_camera_update_azimuth(
                azimuth=camera_azimuth, gamepadstick_right=gamepadstick_right, desired_strafe=desired_strafe, dt=i * dt
            ),
            desired_strafe=desired_strafe,
            desired_velocity=desired_velocities[i],
        )


def trajectory_rotations_predict(
    rotations: np.ndarray,
    angular_velocities: np.ndarray,
    rotation: np.ndarray,
    angular_velocity: np.ndarray,
    desired_rotations: np.ndarray,
    halflife: float,
    dt: float,
) -> None:
    """Predict actual rotations using spring damper"""

    for i in range(1, len(rotations)):
        rotations[i], angular_velocities[i] = simulation_rotations_update(
            x=rotation.copy(), v=angular_velocity.copy(), x_goal=desired_rotations[i], halflife=halflife, dt=i * dt
        )


def trajectory_desired_velocities_predict(
    desired_velocities: np.ndarray,
    trajectory_rotations: np.ndarray,
    desired_velocity: np.ndarray,
    camera_azimuth: float,
    gamepadstick_left: np.ndarray,
    gamepadstick_right: np.ndarray,
    desired_strafe: bool,
    fwrd_speed: float,
    side_speed: float,
    back_speed: float,
    dt: float,
) -> None:
    """Predict desired velocities for future trajectory points"""
    desired_velocities[0] = desired_velocity

    for i in range(1, len(desired_velocities)):
        desired_velocities[i] = desired_velocity_update(
            gamepadstick_left=gamepadstick_left,
            camera_azimuth=orbit_camera_update_azimuth(
                azimuth=camera_azimuth, gamepadstick_right=gamepadstick_right, desired_strafe=desired_strafe, dt=i * dt
            ),
            simulation_rotation=trajectory_rotations[i],
            fwrd_speed=fwrd_speed,
            side_speed=side_speed,
            back_speed=back_speed,
        )


def simulation_positions_update(
    position: np.ndarray,
    velocity: np.ndarray,
    acceleration: np.ndarray,
    desired_velocity: np.ndarray,
    halflife: float,
    dt: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Update simulation positions using exact integration"""
    y = math.halflife_to_damping(halflife) / 2.0
    j0 = velocity - desired_velocity
    j1 = acceleration + j0 * y
    eydt = math.fast_negexpf(y * dt)

    position_prev = position.copy()

    new_position = (
        eydt * (((-j1) / (y * y)) + ((-j0 - j1 * dt) / y))
        + (j1 / (y * y))
        + j0 / y
        + desired_velocity * dt
        + position_prev
    )
    new_velocity = eydt * (j0 + j1 * dt) + desired_velocity
    new_acceleration = eydt * (acceleration - j1 * y * dt)

    return new_position, new_velocity, new_acceleration


def trajectory_positions_predict(
    positions: np.ndarray,
    velocities: np.ndarray,
    accelerations: np.ndarray,
    position: np.ndarray,
    velocity: np.ndarray,
    acceleration: np.ndarray,
    desired_velocities: np.ndarray,
    halflife: float,
    dt: float,
) -> None:
    """Predict trajectory positions using physics simulation"""
    positions[0] = position.copy()
    velocities[0] = velocity.copy()
    accelerations[0] = acceleration.copy()

    for i in range(1, len(positions)):
        positions[i], velocities[i], accelerations[i] = simulation_positions_update(
            position=positions[i - 1].copy(),
            velocity=velocities[i - 1].copy(),
            acceleration=accelerations[i - 1].copy(),
            desired_velocity=desired_velocities[i],
            halflife=halflife,
            dt=dt,
        )


def draw_trajectory(
    server: Any,
    trajectory_positions: np.ndarray,
    trajectory_rotations: np.ndarray,
    color: np.ndarray = np.array([255, 165, 0]),
    name_prefix: str = "trajectory",
) -> None:
    """Draw trajectory for debugging - mimics C++ draw_trajectory function"""
    for i in range(1, len(trajectory_positions)):
        pos = trajectory_positions[i]
        rot = trajectory_rotations[i]

        server.scene.add_icosphere(f"/{name_prefix}_point_{i}", position=pos, radius=0.05, color=color, subdivisions=2)

        forward_vec = math.quat_mul_vec3(rot, np.array([1.0, 0.0, 0.0]))
        end_pos = pos + 0.6 * forward_vec

        server.scene.add_line_segments(
            f"/{name_prefix}_dir_{i}",
            points=np.array([[pos, end_pos]]),
            colors=np.array([[color, color]]),
            line_width=3.0,
        )

        if i >= 1:
            prev_pos = trajectory_positions[i - 1]
            server.scene.add_line_segments(
                f"/{name_prefix}_path_{i}",
                points=np.array([[prev_pos, pos]]),
                colors=np.array([[color, color]]),
                line_width=2.0,
            )


def contact_reset(
    contact_state: np.ndarray,
    contact_lock: np.ndarray,
    contact_position: np.ndarray,
    contact_velocity: np.ndarray,
    contact_point: np.ndarray,
    contact_target: np.ndarray,
    contact_offset_position: np.ndarray,
    contact_offset_velocity: np.ndarray,
    input_contact_position: np.ndarray,
    input_contact_velocity: np.ndarray,
    input_contact_state: bool,
) -> None:
    """Reset contact state for a single contact point (matches C++ contact_reset)"""
    contact_state[...] = False
    contact_lock[...] = False
    contact_position[:] = input_contact_position
    contact_velocity[:] = input_contact_velocity
    contact_point[:] = input_contact_position
    contact_target[:] = input_contact_position
    contact_offset_position[:] = 0.0
    contact_offset_velocity[:] = 0.0


def query_compute_trajectory_position_feature(
    query: np.ndarray,
    offset: int,
    root_position: np.ndarray,
    root_rotation: np.ndarray,
    trajectory_positions: np.ndarray,
) -> int:
    """Compute trajectory position features for motion matching query (matches C++ version)"""
    traj0 = math.quat_inv_mul_vec3(root_rotation, trajectory_positions[1] - root_position)
    traj1 = math.quat_inv_mul_vec3(root_rotation, trajectory_positions[2] - root_position)
    traj2 = math.quat_inv_mul_vec3(root_rotation, trajectory_positions[3] - root_position)

    query[offset + 0] = traj0[0]
    query[offset + 1] = traj0[1]
    query[offset + 2] = traj1[0]
    query[offset + 3] = traj1[1]
    query[offset + 4] = traj2[0]
    query[offset + 5] = traj2[1]

    return offset + 6


def query_compute_trajectory_direction_feature(
    query: np.ndarray, offset: int, root_rotation: np.ndarray, trajectory_rotations: np.ndarray
) -> int:
    """Compute trajectory direction features for motion matching query (matches C++ version)"""
    forward = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    traj0 = math.quat_inv_mul_vec3(root_rotation, math.quat_mul_vec3(trajectory_rotations[1], forward))
    traj1 = math.quat_inv_mul_vec3(root_rotation, math.quat_mul_vec3(trajectory_rotations[2], forward))
    traj2 = math.quat_inv_mul_vec3(root_rotation, math.quat_mul_vec3(trajectory_rotations[3], forward))

    query[offset + 0] = traj0[0]
    query[offset + 1] = traj0[1]
    query[offset + 2] = traj1[0]
    query[offset + 3] = traj1[1]
    query[offset + 4] = traj2[0]
    query[offset + 5] = traj2[1]

    return offset + 6


def adjust_query_ground_height(query: np.ndarray, src_ground_height: float, dst_ground_height: float) -> np.ndarray:
    """Adjust height feature values in query to match the desired ground height"""
    adjust_amount = -src_ground_height + dst_ground_height
    query[..., [2, 5]] += adjust_amount
    return query
