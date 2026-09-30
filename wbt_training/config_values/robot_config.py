"""G1 motor calibration and joint/body orderings used by PHP.

``build_g1_implicit_actuators`` remains at this path for saved checkpoints.
The literal orderings keep the no-MuJoCo motion converter usable with NumPy
alone. Integration tests check them against the pinned Holosoma robot.
"""

from __future__ import annotations

import numpy as np

G1_32BODY_NAMES = [
    "pelvis",
    "left_hip_pitch_link",
    "left_hip_roll_link",
    "left_hip_yaw_link",
    "left_knee_link",
    "left_ankle_pitch_link",
    "left_ankle_roll_link",
    "left_foot_contact_point",
    "right_hip_pitch_link",
    "right_hip_roll_link",
    "right_hip_yaw_link",
    "right_knee_link",
    "right_ankle_pitch_link",
    "right_ankle_roll_link",
    "right_foot_contact_point",
    "waist_yaw_link",
    "waist_roll_link",
    "torso_link",
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_link",
    "left_wrist_roll_link",
    "left_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
    "right_wrist_roll_link",
    "right_wrist_pitch_link",
    "right_wrist_yaw_link",
]
G1_29DOF_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]
ISAACLAB_TO_MUJOCO_DOF = np.array(
    [
        0,
        3,
        6,
        9,
        13,
        17,
        1,
        4,
        7,
        10,
        14,
        18,
        2,
        5,
        8,
        11,
        15,
        19,
        21,
        23,
        25,
        27,
        12,
        16,
        20,
        22,
        24,
        26,
        28,
    ]
)


# ---------------------------------------------------------------------------
# G1 implicit-actuator motor calibration
# Registered on ``robot.control.implicit_actuator_builder`` when a preset sets
# ``control_mode="implicit_position_target"``. Tuned so IsaacSim's built-in PD
# drive applies torque with the correct stiffness/damping/armature.
# ---------------------------------------------------------------------------
_NATURAL_FREQ = 10 * 2.0 * 3.1415926535
_DAMPING_RATIO = 2.0
_ARMATURE_5020 = 0.003609725
_ARMATURE_7520_14 = 0.010177520
_ARMATURE_7520_22 = 0.025101925
_ARMATURE_4010 = 0.00425
_STIFFNESS_5020 = _ARMATURE_5020 * _NATURAL_FREQ**2
_STIFFNESS_7520_14 = _ARMATURE_7520_14 * _NATURAL_FREQ**2
_STIFFNESS_7520_22 = _ARMATURE_7520_22 * _NATURAL_FREQ**2
_STIFFNESS_4010 = _ARMATURE_4010 * _NATURAL_FREQ**2
_DAMPING_5020 = 2.0 * _DAMPING_RATIO * _ARMATURE_5020 * _NATURAL_FREQ
_DAMPING_7520_14 = 2.0 * _DAMPING_RATIO * _ARMATURE_7520_14 * _NATURAL_FREQ
_DAMPING_7520_22 = 2.0 * _DAMPING_RATIO * _ARMATURE_7520_22 * _NATURAL_FREQ
_DAMPING_4010 = 2.0 * _DAMPING_RATIO * _ARMATURE_4010 * _NATURAL_FREQ


def build_g1_implicit_actuators():
    # Imported lazily so this module is importable without IsaacLab.
    from isaaclab.actuators import ImplicitActuatorCfg

    return {
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
            ],
            effort_limit_sim={
                ".*_hip_yaw_joint": 88.0,
                ".*_hip_roll_joint": 139.0,
                ".*_hip_pitch_joint": 88.0,
                ".*_knee_joint": 139.0,
            },
            velocity_limit_sim={
                ".*_hip_yaw_joint": 32.0,
                ".*_hip_roll_joint": 20.0,
                ".*_hip_pitch_joint": 32.0,
                ".*_knee_joint": 20.0,
            },
            stiffness={
                ".*_hip_pitch_joint": _STIFFNESS_7520_14,
                ".*_hip_roll_joint": _STIFFNESS_7520_22,
                ".*_hip_yaw_joint": _STIFFNESS_7520_14,
                ".*_knee_joint": _STIFFNESS_7520_22,
            },
            damping={
                ".*_hip_pitch_joint": _DAMPING_7520_14,
                ".*_hip_roll_joint": _DAMPING_7520_22,
                ".*_hip_yaw_joint": _DAMPING_7520_14,
                ".*_knee_joint": _DAMPING_7520_22,
            },
            armature={
                ".*_hip_pitch_joint": _ARMATURE_7520_14,
                ".*_hip_roll_joint": _ARMATURE_7520_22,
                ".*_hip_yaw_joint": _ARMATURE_7520_14,
                ".*_knee_joint": _ARMATURE_7520_22,
            },
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            effort_limit_sim=50.0,
            velocity_limit_sim=37.0,
            stiffness=2.0 * _STIFFNESS_5020,
            damping=2.0 * _DAMPING_5020,
            armature=2.0 * _ARMATURE_5020,
        ),
        "waist": ImplicitActuatorCfg(
            joint_names_expr=["waist_roll_joint", "waist_pitch_joint"],
            effort_limit_sim=50.0,
            velocity_limit_sim=37.0,
            stiffness=2.0 * _STIFFNESS_5020,
            damping=2.0 * _DAMPING_5020,
            armature=2.0 * _ARMATURE_5020,
        ),
        "waist_yaw": ImplicitActuatorCfg(
            joint_names_expr=["waist_yaw_joint"],
            effort_limit_sim=88.0,
            velocity_limit_sim=32.0,
            stiffness=_STIFFNESS_7520_14,
            damping=_DAMPING_7520_14,
            armature=_ARMATURE_7520_14,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
                ".*_wrist_roll_joint",
                ".*_wrist_pitch_joint",
                ".*_wrist_yaw_joint",
            ],
            effort_limit_sim={
                ".*_shoulder_pitch_joint": 25.0,
                ".*_shoulder_roll_joint": 25.0,
                ".*_shoulder_yaw_joint": 25.0,
                ".*_elbow_joint": 25.0,
                ".*_wrist_roll_joint": 25.0,
                ".*_wrist_pitch_joint": 5.0,
                ".*_wrist_yaw_joint": 5.0,
            },
            velocity_limit_sim={
                ".*_shoulder_pitch_joint": 37.0,
                ".*_shoulder_roll_joint": 37.0,
                ".*_shoulder_yaw_joint": 37.0,
                ".*_elbow_joint": 37.0,
                ".*_wrist_roll_joint": 37.0,
                ".*_wrist_pitch_joint": 22.0,
                ".*_wrist_yaw_joint": 22.0,
            },
            stiffness={
                ".*_shoulder_pitch_joint": _STIFFNESS_5020,
                ".*_shoulder_roll_joint": _STIFFNESS_5020,
                ".*_shoulder_yaw_joint": _STIFFNESS_5020,
                ".*_elbow_joint": _STIFFNESS_5020,
                ".*_wrist_roll_joint": _STIFFNESS_5020,
                ".*_wrist_pitch_joint": _STIFFNESS_4010,
                ".*_wrist_yaw_joint": _STIFFNESS_4010,
            },
            damping={
                ".*_shoulder_pitch_joint": _DAMPING_5020,
                ".*_shoulder_roll_joint": _DAMPING_5020,
                ".*_shoulder_yaw_joint": _DAMPING_5020,
                ".*_elbow_joint": _DAMPING_5020,
                ".*_wrist_roll_joint": _DAMPING_5020,
                ".*_wrist_pitch_joint": _DAMPING_4010,
                ".*_wrist_yaw_joint": _DAMPING_4010,
            },
            armature={
                ".*_shoulder_pitch_joint": _ARMATURE_5020,
                ".*_shoulder_roll_joint": _ARMATURE_5020,
                ".*_shoulder_yaw_joint": _ARMATURE_5020,
                ".*_elbow_joint": _ARMATURE_5020,
                ".*_wrist_roll_joint": _ARMATURE_5020,
                ".*_wrist_pitch_joint": _ARMATURE_4010,
                ".*_wrist_yaw_joint": _ARMATURE_4010,
            },
        ),
    }
