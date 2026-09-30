"""Diagnose motion NPZ files for compatibility with holosoma training.

Checks:
  1. Array shapes and dtypes
  2. First-frame root pose (height, quaternion validity)
  3. Body position sanity (pelvis height, limb distances)
  4. Quaternion convention detection (wxyz vs xyzw)
  5. Comparison with a known-good reference motion

Usage:
    python diagnose_motion.py /path/to/example_motion.npz
    python diagnose_motion.py /path/to/suspect.npz --reference /path/to/known_good.npz
"""

import argparse

import numpy as np


def load_npz(path):
    with np.load(path, allow_pickle=True) as data:
        return {k: data[k] for k in data}


def check_quat_convention(q):
    """Check if quaternion array is likely wxyz or xyzw.

    For a standing humanoid, the root quaternion should be close to identity.
    Identity wxyz = [1, 0, 0, 0], identity xyzw = [0, 0, 0, 1].
    """
    # Check first element vs last element magnitude
    first_abs_mean = np.abs(q[:, 0]).mean()
    last_abs_mean = np.abs(q[:, 3]).mean()

    if first_abs_mean > 0.8 and last_abs_mean < 0.3:
        return "wxyz (w-first, w≈1)"
    elif last_abs_mean > 0.8 and first_abs_mean < 0.3:
        return "xyzw (w-last, w≈1)"
    else:
        return f"UNCLEAR (first_mean={first_abs_mean:.3f}, last_mean={last_abs_mean:.3f})"


def diagnose(path, reference_path=None):
    print(f"\n{'=' * 70}")
    print(f"  Diagnosing: {path}")
    print(f"{'=' * 70}")

    d = load_npz(path)

    # 1. Basic info
    print("\n--- Basic Info ---")
    print(f"  fps: {d['fps']}")
    print(f"  joint_names ({len(d['joint_names'])}): {d['joint_names'][:5].tolist()} ...")
    print(f"  body_names  ({len(d['body_names'])}): {d['body_names'][:5].tolist()} ...")

    for key in ["joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"]:
        if key in d:
            print(f"  {key:20s}: shape={d[key].shape}, dtype={d[key].dtype}")
        else:
            print(f"  {key:20s}: MISSING!")

    T = d["joint_pos"].shape[0]
    fps = float(np.atleast_1d(d["fps"]).flatten()[0])
    print(f"  Total frames: {T}, duration: {T / fps:.1f}s")

    # 2. Root pose from joint_pos (first 7 = xyz + wxyz)
    print("\n--- Root Pose (from joint_pos, first 7 cols = xyz + quat_wxyz) ---")
    root_xyz = d["joint_pos"][:, :3]
    root_quat_from_jp = d["joint_pos"][:, 3:7]

    print(f"  Frame 0 root_xyz:  {root_xyz[0]}")
    print(f"  Frame 0 root_quat: {root_quat_from_jp[0]}")
    print(f"  Root height range: [{root_xyz[:, 2].min():.3f}, {root_xyz[:, 2].max():.3f}]")
    print(f"  Root quat norm (frame 0): {np.linalg.norm(root_quat_from_jp[0]):.6f}")
    print(f"  Root quat convention (joint_pos): {check_quat_convention(root_quat_from_jp)}")

    # 3. Body positions
    print("\n--- Body Positions ---")
    body_names = d["body_names"].tolist()
    body_pos = d["body_pos_w"]

    # Find pelvis index
    pelvis_idx = body_names.index("pelvis") if "pelvis" in body_names else 0
    pelvis_pos = body_pos[:, pelvis_idx, :]
    print(f"  Pelvis (body {pelvis_idx}) frame 0: {pelvis_pos[0]}")
    print(f"  Pelvis height range: [{pelvis_pos[:, 2].min():.3f}, {pelvis_pos[:, 2].max():.3f}]")

    # Check pelvis matches root
    root_pelvis_diff = np.abs(root_xyz[0] - pelvis_pos[0])
    print(f"  |root_xyz - pelvis_pos| frame 0: {root_pelvis_diff} (should be ~0)")
    if root_pelvis_diff.max() > 0.05:
        print(f"  *** WARNING: Root position and pelvis body position differ by {root_pelvis_diff.max():.3f}m!")

    # Check some body heights
    for name in ["left_ankle_roll_link", "right_ankle_roll_link", "torso_link"]:
        if name in body_names:
            idx = body_names.index(name)
            h = body_pos[0, idx, 2]
            print(f"  {name} height (frame 0): {h:.3f}m")

    # 4. Body quaternions
    print("\n--- Body Quaternions ---")
    body_quat = d["body_quat_w"]
    pelvis_quat = body_quat[:, pelvis_idx, :]
    print(f"  Pelvis quat frame 0: {pelvis_quat[0]}")
    print(f"  Pelvis quat norm (frame 0): {np.linalg.norm(pelvis_quat[0]):.6f}")
    print(f"  Body quat convention (pelvis): {check_quat_convention(pelvis_quat)}")

    # Check root quat from joint_pos matches pelvis body quat
    root_pelvis_quat_diff = np.abs(root_quat_from_jp[0] - pelvis_quat[0])
    print(f"  |root_quat - pelvis_quat| frame 0: {root_pelvis_quat_diff} (should be ~0)")
    if root_pelvis_quat_diff.max() > 0.05:
        print(f"  *** WARNING: Root quat and pelvis body quat differ by {root_pelvis_quat_diff.max():.3f}!")

    # 5. Joint positions (first few)
    print(f"\n--- Joint Positions (frame 0, first 10 of {d['joint_pos'].shape[1] - 7} DOFs) ---")
    joint_pos_dofs = d["joint_pos"][0, 7:]  # skip root DOFs
    joint_names = d["joint_names"].tolist()
    for i, name in enumerate(joint_names[:10]):
        print(f"  {name:35s}: {joint_pos_dofs[i]:.4f}")

    # Check joint range
    jp_min = d["joint_pos"][:, 7:].min(axis=0)
    jp_max = d["joint_pos"][:, 7:].max(axis=0)
    jp_range = jp_max - jp_min
    print("\n  Joint position ranges (min, max, range):")
    for i, name in enumerate(joint_names[:10]):
        print(f"  {name:35s}: [{jp_min[i]:+.3f}, {jp_max[i]:+.3f}] range={jp_range[i]:.3f}")

    # 6. Sanity checks
    print("\n--- Sanity Checks ---")

    # Check pelvis height is reasonable for G1 (~0.6-1.0m standing)
    avg_pelvis_h = pelvis_pos[:, 2].mean()
    if avg_pelvis_h < 0.3 or avg_pelvis_h > 1.5:
        print(f"  *** FAIL: Average pelvis height {avg_pelvis_h:.3f}m is outside [0.3, 1.5]m")
    else:
        print(f"  PASS: Average pelvis height {avg_pelvis_h:.3f}m looks reasonable")

    # Check quaternion norms
    quat_norms = np.linalg.norm(body_quat[:, pelvis_idx, :], axis=-1)
    if np.abs(quat_norms - 1.0).max() > 0.01:
        print(f"  *** FAIL: Pelvis quat norm deviates from 1: range [{quat_norms.min():.4f}, {quat_norms.max():.4f}]")
    else:
        print("  PASS: Pelvis quaternion norms are valid")

    # Check feet are below pelvis
    for foot in ["left_ankle_roll_link", "right_ankle_roll_link"]:
        if foot in body_names:
            foot_idx = body_names.index(foot)
            foot_h = body_pos[0, foot_idx, 2]
            if foot_h > pelvis_pos[0, 2]:
                print(f"  *** FAIL: {foot} ({foot_h:.3f}m) is above pelvis ({pelvis_pos[0, 2]:.3f}m)!")
            else:
                print(f"  PASS: {foot} ({foot_h:.3f}m) is below pelvis ({pelvis_pos[0, 2]:.3f}m)")

    # Check body position spread (max distance between any two bodies)
    max_spread = 0
    for t in [0, T // 2, T - 1]:
        diffs = body_pos[t, :, np.newaxis, :] - body_pos[t, np.newaxis, :, :]
        spread = np.linalg.norm(diffs, axis=-1).max()
        max_spread = max(max_spread, spread)
    if max_spread > 5.0:
        print(f"  *** FAIL: Max body spread {max_spread:.3f}m seems too large (robot flying apart?)")
    else:
        print(f"  PASS: Max body spread {max_spread:.3f}m looks reasonable")

    # 7. Compare with reference
    if reference_path:
        print("\n--- Comparison with Reference ---")
        ref = load_npz(reference_path)

        ref_body_names_list = ref["body_names"].tolist()
        print(f"  Reference body count: {len(ref_body_names_list)}, Test body count: {len(body_names)}")
        print(f"  Reference body names match: {body_names == ref_body_names_list}")

        ref_joint_names = ref["joint_names"].tolist()
        print(f"  Reference joint names match: {joint_names == ref_joint_names}")

        ref_fps = float(np.atleast_1d(ref["fps"]).flatten()[0])
        test_fps = float(np.atleast_1d(d["fps"]).flatten()[0])
        print(f"  Reference fps: {ref_fps}, Test fps: {test_fps}")

        ref_body_names = ref["body_names"].tolist()
        ref_pelvis_idx = ref_body_names.index("pelvis") if "pelvis" in ref_body_names else 0
        ref_pelvis_h = ref["body_pos_w"][0, ref_pelvis_idx, 2]
        test_pelvis_h = pelvis_pos[0, 2]
        print(f"  Reference pelvis height frame 0: {ref_pelvis_h:.3f}m")
        print(f"  Test pelvis height frame 0: {test_pelvis_h:.3f}m")

        ref_pelvis_quat = ref["body_quat_w"][0, ref_pelvis_idx]
        test_pelvis_quat = pelvis_quat[0]
        print(f"  Reference pelvis quat frame 0: {ref_pelvis_quat}")
        print(f"  Test pelvis quat frame 0: {test_pelvis_quat}")
        print(f"  Reference quat convention: {check_quat_convention(ref['body_quat_w'][:, 0, :])}")
        print(f"  Test quat convention: {check_quat_convention(pelvis_quat.reshape(-1, 4))}")

        # Compare shapes
        for key in ["joint_pos", "body_pos_w", "body_quat_w"]:
            ref_shape = ref[key].shape[1:]
            test_shape = d[key].shape[1:]
            match = "✓" if ref_shape == test_shape else "✗"
            print(f"  {key} non-time shape: ref={ref_shape} test={test_shape} {match}")

    print(f"\n{'=' * 70}")
    print("  Diagnosis complete")
    print(f"{'=' * 70}\n")


def main():
    parser = argparse.ArgumentParser(description="Diagnose motion NPZ files")
    parser.add_argument("motion_file", help="Path to motion NPZ file to diagnose")
    parser.add_argument("--reference", default=None, help="Path to known-good reference NPZ")
    args = parser.parse_args()

    diagnose(args.motion_file, args.reference)


if __name__ == "__main__":
    main()
