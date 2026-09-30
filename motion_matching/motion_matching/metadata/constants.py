"""Package-wide numeric constants for the G1 humanoid motion-matching pipeline.

Bone indices refer to the database hierarchy (simulation root, base link,
then G1 actuated joints). Rotation axes use the URDF actuated-joint order.
"""

import numpy as np


# G1 runtime skeleton (kin.joint_names indexing).
G1_BONE_LEFT_FOOT = 7
G1_BONE_RIGHT_FOOT = 13
G1_BONE_HIPS = 1

# Number of randomized terrain variants generated per scenario.
NUM_TERRAIN_VARIANTS = 10


# Pre-computed rotation axis per joint, in joint_names order. This is a
# property of the G1 URDF and never changes between runs; baking it as a
# constant lets entry-point scripts skip URDF parsing at module import (which
# was both slow on multi-worker spawn and a module-level side effect).
#
# To regenerate (after a URDF change):
#   from motion_matching.kinematics.g1_setup import init_g1_kinematics
#   _, axes = init_g1_kinematics(verbose=False)
#   for i, name in enumerate(kin.joint_names):
#       print(f"    [{axes[i,0]:.6f}, {axes[i,1]:.6f}, {axes[i,2]:.6f}],  # {i}: {name}")
G1_JOINT_AXES = np.array(
    [
        [0.0, 1.0, 0.0],  # 0: left_hip_pitch_joint
        [1.0, 0.0, 0.0],  # 1: left_hip_roll_joint
        [0.0, 0.0, 1.0],  # 2: left_hip_yaw_joint
        [0.0, 1.0, 0.0],  # 3: left_knee_joint
        [0.0, 1.0, 0.0],  # 4: left_ankle_pitch_joint
        [1.0, 0.0, 0.0],  # 5: left_ankle_roll_joint
        [0.0, 1.0, 0.0],  # 6: right_hip_pitch_joint
        [1.0, 0.0, 0.0],  # 7: right_hip_roll_joint
        [0.0, 0.0, 1.0],  # 8: right_hip_yaw_joint
        [0.0, 1.0, 0.0],  # 9: right_knee_joint
        [0.0, 1.0, 0.0],  # 10: right_ankle_pitch_joint
        [1.0, 0.0, 0.0],  # 11: right_ankle_roll_joint
        [0.0, 0.0, 1.0],  # 12: waist_yaw_joint
        [1.0, 0.0, 0.0],  # 13: waist_roll_joint
        [0.0, 1.0, 0.0],  # 14: waist_pitch_joint
        [0.0, 1.0, 0.0],  # 15: left_shoulder_pitch_joint
        [1.0, 0.0, 0.0],  # 16: left_shoulder_roll_joint
        [0.0, 0.0, 1.0],  # 17: left_shoulder_yaw_joint
        [0.0, 1.0, 0.0],  # 18: left_elbow_joint
        [1.0, 0.0, 0.0],  # 19: left_wrist_roll_joint
        [0.0, 1.0, 0.0],  # 20: left_wrist_pitch_joint
        [0.0, 0.0, 1.0],  # 21: left_wrist_yaw_joint
        [0.0, 1.0, 0.0],  # 22: right_shoulder_pitch_joint
        [1.0, 0.0, 0.0],  # 23: right_shoulder_roll_joint
        [0.0, 0.0, 1.0],  # 24: right_shoulder_yaw_joint
        [0.0, 1.0, 0.0],  # 25: right_elbow_joint
        [1.0, 0.0, 0.0],  # 26: right_wrist_roll_joint
        [0.0, 1.0, 0.0],  # 27: right_wrist_pitch_joint
        [0.0, 0.0, 1.0],  # 28: right_wrist_yaw_joint
    ],
    dtype=np.float32,
)
