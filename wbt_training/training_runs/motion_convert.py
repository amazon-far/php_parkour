#!/usr/bin/env python3
"""Convert PHP motion-matching outputs to Holosoma's NPZ format.

Reads the BeyondMimic layout emitted by motion matching and preserves input
body poses without MuJoCo forward kinematics. Inserts the two
foot_contact_point bodies as welded duplicates of their ankle_roll parents,
reorders joint DOFs Isaac→MuJoCo, and writes the holosoma format.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

from wbt_training.config_values.robot_config import (
    G1_29DOF_JOINT_NAMES as G1_JOINT_NAMES_MUJOCO,
    G1_32BODY_NAMES as G1_BODY_NAMES,
    ISAACLAB_TO_MUJOCO_DOF as ISAAC_TO_MUJOCO,
)

# Holosoma 32-body ordering mapped back to BeyondMimic 30-body ordering.
# BM uses DOF-layer-grouped ordering (pelvis, hips_pitch, waist_yaw, hips_roll,
# waist_roll, hips_yaw, torso, knees, shoulders_pitch, ankles_pitch,
# shoulders_roll, ankles_roll, shoulders_yaw, elbows, wrist_rolls,
# wrist_pitches, wrist_yaws). Holosoma uses per-chain contiguous ordering.
# BM[18]=left_ankle_roll also provides holosoma[7]=left_foot_contact_point.
# BM[19]=right_ankle_roll also provides holosoma[14]=right_foot_contact_point.
OUT32_TO_BM30 = [
    0, 1, 4, 7, 10, 14, 18, 18,      # pelvis .. left_foot_contact_point
    2, 5, 8, 11, 15, 19, 19,         # right_hip_pitch .. right_foot_contact_point
    3, 6, 9,                         # waist_yaw, waist_roll, torso
    12, 16, 20, 22, 24, 26, 28,      # left_shoulder_pitch .. left_wrist_yaw
    13, 17, 21, 23, 25, 27, 29,      # right_shoulder_pitch .. right_wrist_yaw
]
assert len(OUT32_TO_BM30) == 32


def convert(in_path: str, out_path: str) -> bool:
    d = np.load(in_path, allow_pickle=True)

    fps_val = d["fps"]
    if fps_val.ndim > 0:
        fps_val = int(fps_val.item() if fps_val.size == 1 else fps_val[0])
    else:
        fps_val = int(fps_val)

    bm_pos = d["body_pos_w"].astype(np.float64)   # [T, 30, 3]
    bm_quat = d["body_quat_w"].astype(np.float64) # [T, 30, 4] wxyz
    bm_lvel = d["body_lin_vel_w"].astype(np.float64)
    bm_avel = d["body_ang_vel_w"].astype(np.float64)

    joint_pos_isaac = d["joint_pos"].astype(np.float64)  # [T, 29]
    joint_vel_isaac = d["joint_vel"].astype(np.float64)

    T = joint_pos_isaac.shape[0]
    if T < 2:
        return False

    # Pass-through reordering: 30 → 32 bodies with foot_contact_points welded
    body_pos_w = bm_pos[:, OUT32_TO_BM30, :]
    body_quat_w = bm_quat[:, OUT32_TO_BM30, :]
    body_lin_vel_w = bm_lvel[:, OUT32_TO_BM30, :]
    body_ang_vel_w = bm_avel[:, OUT32_TO_BM30, :]

    # Root DOFs directly from pelvis (BM body 0 = holosoma body 0)
    root_pos = bm_pos[:, 0, :]       # [T, 3]
    root_quat = bm_quat[:, 0, :]     # [T, 4] wxyz
    root_lvel = bm_lvel[:, 0, :]
    root_avel = bm_avel[:, 0, :]

    # Joint DOFs: Isaac → MuJoCo permutation
    joint_pos_mj = joint_pos_isaac[:, ISAAC_TO_MUJOCO]
    joint_vel_mj = joint_vel_isaac[:, ISAAC_TO_MUJOCO]

    joint_pos_h = np.concatenate([root_pos, root_quat, joint_pos_mj], axis=1)  # [T, 36]
    joint_vel_h = np.concatenate([root_lvel, root_avel, joint_vel_mj], axis=1) # [T, 35]

    out = {
        "fps": np.array([fps_val]),   # shape (1,) to match reference
        "joint_pos": joint_pos_h,
        "joint_vel": joint_vel_h,
        "body_pos_w": body_pos_w,
        "body_quat_w": body_quat_w,
        "body_lin_vel_w": body_lin_vel_w,
        "body_ang_vel_w": body_ang_vel_w,
        "joint_names": np.array(G1_JOINT_NAMES_MUJOCO),
        "body_names": np.array(G1_BODY_NAMES),
    }
    for passthrough in ("vel_cmd", "motion_ends", "motion_idxs"):
        if passthrough in d.files:
            out[passthrough] = d[passthrough]

    np.savez(out_path, **out)
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("input")
    p.add_argument("output")
    args = p.parse_args()

    if os.path.isdir(args.input):
        os.makedirs(args.output, exist_ok=True)
        files = sorted(Path(args.input).glob("*.npz"))
        if not files:
            print(f"No NPZ files in {args.input}")
            sys.exit(1)
        ok_count = 0
        for f in files:
            if convert(str(f), os.path.join(args.output, f.name)):
                ok_count += 1
        print(f"Converted {ok_count}/{len(files)} files.")
    else:
        convert(args.input, args.output)
        print(f"Converted {args.input} -> {args.output}")


if __name__ == "__main__":
    main()
