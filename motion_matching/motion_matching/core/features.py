#!/usr/bin/env python3
"""
Feature extraction
Faithful Python port of the C++ implementation, including range_starts and range_stops.
"""

import numpy as np
from typing import Tuple

def quat_mul_vec3(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector by quaternion"""
    w, x, y, z = q[0], q[1], q[2], q[3]
    vx, vy, vz = v[0], v[1], v[2]

    return np.array(
        [
            vx * (1 - 2 * y * y - 2 * z * z) + vy * (2 * x * y - 2 * w * z) + vz * (2 * x * z + 2 * w * y),
            vx * (2 * x * y + 2 * w * z) + vy * (1 - 2 * x * x - 2 * z * z) + vz * (2 * y * z - 2 * w * x),
            vx * (2 * x * z - 2 * w * y) + vy * (2 * y * z + 2 * w * x) + vz * (1 - 2 * x * x - 2 * y * y),
        ]
    )


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two quaternions"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def quat_inv_negate_w(q: np.ndarray) -> np.ndarray:
    """Negate the w-component (NOT a proper quaternion inverse).

    Matches the C++ reference's ``quat_inv`` — a sign trick assuming unit
    quaternions, not full conjugation. Named explicitly to avoid confusion with
    ``motion_matching.utils.math_utils.quat_inv``, which is a proper
    conjugate-divided-by-norm. The two are NOT interchangeable.
    """
    w, x, y, z = q
    return np.array([-w, x, y, z])


def cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cross product of two vectors"""
    return np.array([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]])


def database_trajectory_index_clamp(range_starts: np.ndarray, range_stops: np.ndarray, frame: int, offset: int) -> int:
    """Exact implementation of database_trajectory_index_clamp from C++"""
    for i in range(len(range_starts)):
        if frame >= range_starts[i] and frame < range_stops[i]:
            return max(range_starts[i], min(frame + offset, range_stops[i] - 1))

    # This should never happen
    assert False
    return -1


def forward_kinematics_velocity(
    bone_positions: np.ndarray,
    bone_velocities: np.ndarray,
    bone_rotations: np.ndarray,
    bone_angular_velocities: np.ndarray,
    bone_parents: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute global bone positions and velocities using forward kinematics"""
    nbones = len(bone_parents)
    global_positions = np.zeros((nbones, 3))
    global_velocities = np.zeros((nbones, 3))
    global_rotations = np.zeros((nbones, 4))
    global_angular_velocities = np.zeros((nbones, 3))

    for bone in range(nbones):
        if bone_parents[bone] == -1:  # Root
            global_positions[bone] = bone_positions[bone]
            global_velocities[bone] = bone_velocities[bone]
            global_rotations[bone] = bone_rotations[bone]
            global_angular_velocities[bone] = bone_angular_velocities[bone]
        else:
            parent = bone_parents[bone]
            parent_pos = global_positions[parent]
            parent_vel = global_velocities[parent]
            parent_rot = global_rotations[parent]
            parent_ang_vel = global_angular_velocities[parent]

            # Transform position
            local_pos = bone_positions[bone]
            rotated_pos = quat_mul_vec3(parent_rot, local_pos)
            global_positions[bone] = parent_pos + rotated_pos

            # Transform velocity (account for angular velocity)
            local_vel = bone_velocities[bone]
            rotated_vel = quat_mul_vec3(parent_rot, local_vel)
            # Add velocity contributed by parent bone angular velocity
            angular_contribution = cross(parent_ang_vel, rotated_pos)
            global_velocities[bone] = parent_vel + rotated_vel + angular_contribution

            # Transform rotation
            global_rotations[bone] = quat_mul(parent_rot, bone_rotations[bone])

            # Transform angular velocity
            local_ang_vel = bone_angular_velocities[bone]
            rotated_ang_vel = quat_mul_vec3(parent_rot, local_ang_vel)
            global_angular_velocities[bone] = rotated_ang_vel + parent_ang_vel

    return global_positions, global_velocities, global_rotations, global_angular_velocities


def forward_kinematics(
    bone_positions: np.ndarray, bone_rotations: np.ndarray, bone_parents: np.ndarray, return_rotations: bool = False
) -> np.ndarray:
    """Compute global bone positions"""
    nbones = len(bone_parents)
    global_positions = np.zeros((nbones, 3))
    global_rotations = np.zeros((nbones, 4))

    for bone in range(nbones):
        if bone_parents[bone] == -1:  # Root
            global_positions[bone] = bone_positions[bone]
            global_rotations[bone] = bone_rotations[bone]
        else:
            parent = bone_parents[bone]
            parent_pos = global_positions[parent]
            parent_rot = global_rotations[parent]

            # Transform position
            local_pos = bone_positions[bone]
            rotated_pos = quat_mul_vec3(parent_rot, local_pos)
            global_positions[bone] = parent_pos + rotated_pos

            # Transform rotation
            global_rotations[bone] = quat_mul(parent_rot, bone_rotations[bone])

    if return_rotations:
        return global_positions, global_rotations
    else:
        return global_positions


def normalize_feature_exact(
    features: np.ndarray,
    features_offset: np.ndarray,
    features_scale: np.ndarray,
    offset: int,
    size: int,
    weight: float = 1.0,
) -> None:
    """Exact implementation of normalize_feature from original code"""
    nframes = features.shape[0]

    # First compute the mean value for each feature dimension
    for j in range(size):
        features_offset[offset + j] = 0.0

    for i in range(nframes):
        for j in range(size):
            features_offset[offset + j] += features[i, offset + j] / nframes

    # Now compute the variance of each feature dimension
    vars_array = np.zeros(size)

    for i in range(nframes):
        for j in range(size):
            vars_array[j] += (features[i, offset + j] - features_offset[offset + j]) ** 2 / nframes

    # We compute the overall std of the feature as the average std across all dimensions
    std = 0.0
    for j in range(size):
        std += np.sqrt(vars_array[j]) / size

    # Features with no variation can have zero std which is almost always a bug
    assert std > 0.0

    # The scale of a feature is just the std divided by the weight
    for j in range(size):
        features_scale[offset + j] = std / weight

    # Using the offset and scale we can then normalize the features
    for i in range(nframes):
        for j in range(size):
            features[i, offset + j] = (features[i, offset + j] - features_offset[offset + j]) / features_scale[
                offset + j
            ]


def compute_bone_position_feature(
    bone_positions: np.ndarray, bone_rotations: np.ndarray, bone_parents: np.ndarray, bone_idx: int, weight: float = 1.0
) -> np.ndarray:
    """Compute bone position feature relative to root"""
    nframes = bone_positions.shape[0]
    features = np.zeros((nframes, 3))

    for i in range(nframes):
        # Get global bone position using forward kinematics
        global_positions = forward_kinematics(
            bone_positions=bone_positions[i], bone_rotations=bone_rotations[i], bone_parents=bone_parents
        )
        bone_pos = global_positions[bone_idx]

        # Get root position and rotation
        root_pos = bone_positions[i, 0]
        root_rot = bone_rotations[i, 0]

        # Transform to root space
        relative_pos = bone_pos - root_pos
        relative_pos = quat_mul_vec3(quat_inv_negate_w(root_rot), relative_pos)

        features[i] = relative_pos

    return features


def compute_bone_velocity_feature(
    bone_positions: np.ndarray,
    bone_velocities: np.ndarray,
    bone_rotations: np.ndarray,
    bone_angular_velocities: np.ndarray,
    bone_parents: np.ndarray,
    bone_idx: int,
    weight: float = 1.0,
) -> np.ndarray:
    """Compute bone velocity feature relative to root"""
    nframes = bone_positions.shape[0]
    features = np.zeros((nframes, 3))

    for i in range(nframes):
        # Get global bone velocity using forward kinematics with velocities
        global_positions, global_velocities, global_rotations, global_angular_velocities = forward_kinematics_velocity(
            bone_positions=bone_positions[i],
            bone_velocities=bone_velocities[i],
            bone_rotations=bone_rotations[i],
            bone_angular_velocities=bone_angular_velocities[i],
            bone_parents=bone_parents,
        )

        bone_vel = global_velocities[bone_idx]

        # Get root rotation
        root_rot = bone_rotations[i, 0]

        # Transform to root space
        bone_vel = quat_mul_vec3(quat_inv_negate_w(root_rot), bone_vel)

        features[i] = bone_vel

    return features


def compute_trajectory_position_feature(
    bone_positions: np.ndarray,
    bone_rotations: np.ndarray,
    range_starts: np.ndarray,
    range_stops: np.ndarray,
    weight: float = 1.0,
    axis1: int = 0,
    axis2: int = 2,
) -> np.ndarray:
    """Compute trajectory position feature
    Args:
        axis1: index of the first axis (default 0 = x-axis)
        axis2: index of the second axis (default 2 = z-axis)
    """
    nframes = bone_positions.shape[0]
    features = np.zeros((nframes, 6))  # 3 time points * 2D

    for i in range(nframes):
        # Get future positions using database_trajectory_index_clamp (20, 40, 60 frames ahead)
        t0 = database_trajectory_index_clamp(range_starts=range_starts, range_stops=range_stops, frame=i, offset=20)
        t1 = database_trajectory_index_clamp(range_starts=range_starts, range_stops=range_stops, frame=i, offset=40)
        t2 = database_trajectory_index_clamp(range_starts=range_starts, range_stops=range_stops, frame=i, offset=60)

        root_pos = bone_positions[i, 0]
        root_rot = bone_rotations[i, 0]

        # Transform to root space
        trajectory_pos0 = quat_mul_vec3(quat_inv_negate_w(root_rot), bone_positions[t0, 0] - root_pos)
        trajectory_pos1 = quat_mul_vec3(quat_inv_negate_w(root_rot), bone_positions[t1, 0] - root_pos)
        trajectory_pos2 = quat_mul_vec3(quat_inv_negate_w(root_rot), bone_positions[t2, 0] - root_pos)

        features[i, 0] = trajectory_pos0[axis1]  # t0 axis1
        features[i, 1] = trajectory_pos0[axis2]  # t0 axis2
        features[i, 2] = trajectory_pos1[axis1]  # t1 axis1
        features[i, 3] = trajectory_pos1[axis2]  # t1 axis2
        features[i, 4] = trajectory_pos2[axis1]  # t2 axis1
        features[i, 5] = trajectory_pos2[axis2]  # t2 axis2

    return features


def compute_trajectory_direction_feature(
    bone_positions: np.ndarray,
    bone_rotations: np.ndarray,
    range_starts: np.ndarray,
    range_stops: np.ndarray,
    weight: float = 1.0,
    axis1: int = 0,
    axis2: int = 2,
    forward: np.ndarray = np.array([0, 0, 1]),
) -> np.ndarray:
    """Compute trajectory direction feature
    Args:
        axis1: index of the first axis (default 0 = x-axis)
        axis2: index of the second axis (default 2 = z-axis)
    """
    nframes = bone_positions.shape[0]
    features = np.zeros((nframes, 6))  # 3 time points * 2D

    for i in range(nframes):
        # Get future rotations using database_trajectory_index_clamp (20, 40, 60 frames ahead)
        t0 = database_trajectory_index_clamp(range_starts=range_starts, range_stops=range_stops, frame=i, offset=20)
        t1 = database_trajectory_index_clamp(range_starts=range_starts, range_stops=range_stops, frame=i, offset=40)
        t2 = database_trajectory_index_clamp(range_starts=range_starts, range_stops=range_stops, frame=i, offset=60)

        root_rot = bone_rotations[i, 0]

        # Get forward direction in root space
        trajectory_dir0 = quat_mul_vec3(quat_inv_negate_w(root_rot), quat_mul_vec3(bone_rotations[t0, 0], forward))
        trajectory_dir1 = quat_mul_vec3(quat_inv_negate_w(root_rot), quat_mul_vec3(bone_rotations[t1, 0], forward))
        trajectory_dir2 = quat_mul_vec3(quat_inv_negate_w(root_rot), quat_mul_vec3(bone_rotations[t2, 0], forward))

        features[i, 0] = trajectory_dir0[axis1]  # t0 axis1
        features[i, 1] = trajectory_dir0[axis2]  # t0 axis2
        features[i, 2] = trajectory_dir1[axis1]  # t1 axis1
        features[i, 3] = trajectory_dir1[axis2]  # t1 axis2
        features[i, 4] = trajectory_dir2[axis1]  # t2 axis1
        features[i, 5] = trajectory_dir2[axis2]  # t2 axis2

    return features


def build_motion_matching_features(
    bone_positions: np.ndarray,
    bone_velocities: np.ndarray,
    bone_rotations: np.ndarray,
    bone_angular_velocities: np.ndarray,
    bone_parents: np.ndarray,
    range_starts: np.ndarray,
    range_stops: np.ndarray,
    bone_left_foot: int,
    bone_right_foot: int,
    bone_hips: int,
    traj_axis1: int = 0,
    traj_axis2: int = 2,
    forward: np.ndarray = np.array([0, 0, 1]),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """Build all motion matching features with exact implementation"""
    print("Building motion matching features...")

    nframes = bone_positions.shape[0]

    # Feature weights (from controller.cpp)
    feature_weight_foot_position = 0.75
    feature_weight_foot_velocity = 1.0
    feature_weight_hip_velocity = 1.0
    feature_weight_trajectory_positions = 1.0
    feature_weight_trajectory_directions = 1.5

    # Total features: 3+3+3+3+3+6+6 = 27
    all_features = np.zeros((nframes, 27))
    feature_names = []

    # Initialize normalization arrays
    features_offset = np.zeros(27)
    features_scale = np.zeros(27)

    offset = 0

    # Left foot position (3 features)
    print("  Computing left foot position features...")
    left_foot_pos = compute_bone_position_feature(
        bone_positions=bone_positions,
        bone_rotations=bone_rotations,
        bone_parents=bone_parents,
        bone_idx=bone_left_foot,
        weight=feature_weight_foot_position,
    )
    all_features[:, offset : offset + 3] = left_foot_pos
    feature_names.extend(["LeftFootPos_X", "LeftFootPos_Y", "LeftFootPos_Z"])
    offset += 3

    # Right foot position (3 features)
    print("  Computing right foot position features...")
    right_foot_pos = compute_bone_position_feature(
        bone_positions=bone_positions,
        bone_rotations=bone_rotations,
        bone_parents=bone_parents,
        bone_idx=bone_right_foot,
        weight=feature_weight_foot_position,
    )
    all_features[:, offset : offset + 3] = right_foot_pos
    feature_names.extend(["RightFootPos_X", "RightFootPos_Y", "RightFootPos_Z"])
    offset += 3

    # Left foot velocity (3 features)
    print("  Computing left foot velocity features...")
    left_foot_vel = compute_bone_velocity_feature(
        bone_positions=bone_positions,
        bone_velocities=bone_velocities,
        bone_rotations=bone_rotations,
        bone_angular_velocities=bone_angular_velocities,
        bone_parents=bone_parents,
        bone_idx=bone_left_foot,
        weight=feature_weight_foot_velocity,
    )
    all_features[:, offset : offset + 3] = left_foot_vel
    feature_names.extend(["LeftFootVel_X", "LeftFootVel_Y", "LeftFootVel_Z"])
    offset += 3

    # Right foot velocity (3 features)
    print("  Computing right foot velocity features...")
    right_foot_vel = compute_bone_velocity_feature(
        bone_positions=bone_positions,
        bone_velocities=bone_velocities,
        bone_rotations=bone_rotations,
        bone_angular_velocities=bone_angular_velocities,
        bone_parents=bone_parents,
        bone_idx=bone_right_foot,
        weight=feature_weight_foot_velocity,
    )
    all_features[:, offset : offset + 3] = right_foot_vel
    feature_names.extend(["RightFootVel_X", "RightFootVel_Y", "RightFootVel_Z"])
    offset += 3

    # Hip velocity (3 features)
    print("  Computing hip velocity features...")
    hip_vel = compute_bone_velocity_feature(
        bone_positions=bone_positions,
        bone_velocities=bone_velocities,
        bone_rotations=bone_rotations,
        bone_angular_velocities=bone_angular_velocities,
        bone_parents=bone_parents,
        bone_idx=bone_hips,
        weight=feature_weight_hip_velocity,
    )
    all_features[:, offset : offset + 3] = hip_vel
    feature_names.extend(["HipVel_X", "HipVel_Y", "HipVel_Z"])
    offset += 3

    # Trajectory positions (6 features)
    print("  Computing trajectory position features...")
    traj_pos = compute_trajectory_position_feature(
        bone_positions=bone_positions,
        bone_rotations=bone_rotations,
        range_starts=range_starts,
        range_stops=range_stops,
        weight=feature_weight_trajectory_positions,
        axis1=traj_axis1,
        axis2=traj_axis2,
    )
    all_features[:, offset : offset + 6] = traj_pos
    feature_names.extend(
        ["TrajPos_t0_A", "TrajPos_t0_B", "TrajPos_t1_A", "TrajPos_t1_B", "TrajPos_t2_A", "TrajPos_t2_B"]
    )
    offset += 6

    # Trajectory directions (6 features)
    print("  Computing trajectory direction features...")
    traj_dir = compute_trajectory_direction_feature(
        bone_positions=bone_positions,
        bone_rotations=bone_rotations,
        range_starts=range_starts,
        range_stops=range_stops,
        weight=feature_weight_trajectory_directions,
        axis1=traj_axis1,
        axis2=traj_axis2,
        forward=forward,
    )
    all_features[:, offset : offset + 6] = traj_dir
    feature_names.extend(
        ["TrajDir_t0_A", "TrajDir_t0_B", "TrajDir_t1_A", "TrajDir_t1_B", "TrajDir_t2_A", "TrajDir_t2_B"]
    )
    offset += 6

    print("  Normalizing features...")
    offset = 0

    # Left foot position (3 features)
    normalize_feature_exact(
        features=all_features,
        features_offset=features_offset,
        features_scale=features_scale,
        offset=offset,
        size=3,
        weight=feature_weight_foot_position,
    )
    offset += 3

    # Right foot position (3 features)
    normalize_feature_exact(
        features=all_features,
        features_offset=features_offset,
        features_scale=features_scale,
        offset=offset,
        size=3,
        weight=feature_weight_foot_position,
    )
    offset += 3

    # Left foot velocity (3 features)
    normalize_feature_exact(
        features=all_features,
        features_offset=features_offset,
        features_scale=features_scale,
        offset=offset,
        size=3,
        weight=feature_weight_foot_velocity,
    )
    offset += 3

    # Right foot velocity (3 features)
    normalize_feature_exact(
        features=all_features,
        features_offset=features_offset,
        features_scale=features_scale,
        offset=offset,
        size=3,
        weight=feature_weight_foot_velocity,
    )
    offset += 3

    # Hip velocity (3 features)
    normalize_feature_exact(
        features=all_features,
        features_offset=features_offset,
        features_scale=features_scale,
        offset=offset,
        size=3,
        weight=feature_weight_hip_velocity,
    )
    offset += 3

    # Trajectory positions (6 features)
    normalize_feature_exact(
        features=all_features,
        features_offset=features_offset,
        features_scale=features_scale,
        offset=offset,
        size=6,
        weight=feature_weight_trajectory_positions,
    )
    offset += 6

    # Trajectory directions (6 features)
    normalize_feature_exact(
        features=all_features,
        features_offset=features_offset,
        features_scale=features_scale,
        offset=offset,
        size=6,
        weight=feature_weight_trajectory_directions,
    )
    offset += 6

    print("✅ Features built successfully!")
    print(f"   Shape: {all_features.shape}")
    print(f"   Feature names: {feature_names}")

    return all_features, features_offset, features_scale, feature_names
