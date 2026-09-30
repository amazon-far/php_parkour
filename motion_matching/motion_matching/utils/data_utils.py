"""Data utilities: motion resampling, mirroring, terrain processing, OBJ-to-JSON conversion."""

import argparse
import json
import os
import sys
from typing import Any, List, Optional, Tuple

import numpy as np
from scipy.interpolate import griddata, interp1d
from scipy.spatial.transform import Rotation

from motion_matching.kinematics import quat as quat


# --------------------------------- Resample ---------------------------------


def _resample_qpos_at_times(data: np.ndarray, sample_times: np.ndarray) -> np.ndarray:
    """Cubically resample ``(N, 36)`` qpos at ``sample_times`` (in source-frame
    units, i.e. ``np.linspace(0, N-1, ...)``).

    Splits the qpos into root quat / root position / dof, interpolates each with
    ``griddata(method="cubic")``, re-normalizes the (unrolled) root quat, and
    concatenates them back. Shared core of :func:`resample_data` and
    :func:`adjust_data_speed` — they differ only in how ``sample_times`` is built.
    """
    root_wxyz = data[:, :4]
    root_pos = data[:, 4:7]
    dof = data[:, 7:]

    nframes = data.shape[0]
    original_times = np.linspace(0, nframes - 1, nframes)

    root_wxyz = quat.unroll(root_wxyz)
    root_wxyz = griddata(original_times, root_wxyz.reshape([nframes, -1]), sample_times, method="cubic").reshape(
        [len(sample_times), 4]
    )
    root_wxyz = quat.normalize(root_wxyz)

    root_pos = griddata(original_times, root_pos.reshape([nframes, -1]), sample_times, method="cubic").reshape(
        [len(sample_times), 3]
    )
    dof = griddata(original_times, dof.reshape([nframes, -1]), sample_times, method="cubic").reshape(
        [len(sample_times), dof.shape[1]]
    )

    return np.concatenate([root_wxyz, root_pos, dof], axis=1)


def resample_data(data: np.ndarray, required_fps: int, original_fps: int) -> np.ndarray:
    """
    Resample data to required FPS

    data: (N, 36) where N is number of frames
    required_fps: required FPS

    Returns:
    resampled_data: (N, 36) where N is number of frames
    """
    if original_fps == required_fps:
        return data

    nframes = data.shape[0]
    sample_times = np.linspace(0, nframes - 1, int(required_fps / original_fps * nframes))
    return _resample_qpos_at_times(data, sample_times)


def adjust_data_speed(data: np.ndarray, speed_up_ratio: float) -> np.ndarray:
    """
    Adjust speed of data

    data: (N, 36) where N is number of frames
    speed_up_ratio: speed up ratio
    """
    nframes = data.shape[0]
    sample_times = np.linspace(0, nframes - 1, int(1 / speed_up_ratio * nframes))
    return _resample_qpos_at_times(data, sample_times)


def resample_vel_cmd(vel_cmd: np.ndarray, required_fps: int, original_fps: int) -> np.ndarray:
    """Resample vel_cmd to required FPS"""
    if original_fps == required_fps:
        return vel_cmd

    nframes = vel_cmd.shape[0]

    original_times = np.linspace(0, nframes - 1, nframes)
    sample_times = np.linspace(0, nframes - 1, int(required_fps / original_fps * nframes))

    f = interp1d(original_times, vel_cmd, axis=0, kind="zero", fill_value="extrapolate")
    resampled_vel_cmd = f(sample_times)

    return resampled_vel_cmd


def adjust_vel_cmd_speed(vel_cmd: np.ndarray, speed_up_ratio: float) -> np.ndarray:
    """Adjust speed of vel_cmd"""
    nframes = vel_cmd.shape[0]

    original_times = np.linspace(0, nframes - 1, nframes)
    sample_times = np.linspace(0, nframes - 1, int(1 / speed_up_ratio * nframes))

    f = interp1d(original_times, vel_cmd, axis=0, kind="zero", fill_value="extrapolate")
    adjusted_vel_cmd = f(sample_times)

    return adjusted_vel_cmd


# ---------------------------------- Mirror ----------------------------------


def animation_mirror_xz(
    dof: np.ndarray, root_pos: np.ndarray, root_wxyz: np.ndarray, names: list
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    joints_mirror = np.array(
        [
            (
                names.index("left" + n[5:])
                if n.startswith("right")
                else (names.index("right" + n[4:]) if n.startswith("left") else names.index(n))
            )
            for n in names
        ]
    )

    neg_joints = [
        "left_hip_roll_joint",
        "right_hip_roll_joint",
        "left_hip_yaw_joint",
        "right_hip_yaw_joint",
        "left_ankle_roll_joint",
        "right_ankle_roll_joint",
        "waist_yaw_joint",
        "waist_roll_joint",
        "left_shoulder_roll_joint",
        "right_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "right_shoulder_yaw_joint",
        "left_wrist_roll_joint",
        "right_wrist_roll_joint",
        "left_wrist_yaw_joint",
        "right_wrist_yaw_joint",
    ]
    neg_indices = [names.index(n) for n in neg_joints]

    dof = dof[:, joints_mirror]
    dof[..., neg_indices] *= -1

    root_pos[..., 1] *= -1
    root_wxyz[..., [1, 3]] *= -1

    return dof, root_pos, root_wxyz


def mirror_terrain(terrain_file: str) -> str:
    """
    Mirror a terrain OBJ file by the xz plane (negate y-coordinates).
    Saves the mirrored file with '_mirror' suffix before the extension.
    """
    with open(terrain_file, "r") as f:
        lines = f.readlines()

    base, ext = os.path.splitext(terrain_file)
    output_file = base + "_mirror" + ext

    mirrored_lines = []
    for line in lines:
        stripped = line.strip()

        if stripped.startswith("v ") and not stripped.startswith("vn ") and not stripped.startswith("vt "):
            parts = stripped.split()
            if len(parts) >= 4:
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                mirrored_lines.append(f"v {x} {-y} {z}\n")
            else:
                mirrored_lines.append(line)

        elif stripped.startswith("f "):
            parts = stripped.split()
            if len(parts) >= 4:
                reversed_vertices = parts[1:][::-1]
                mirrored_lines.append("f " + " ".join(reversed_vertices) + "\n")
            else:
                mirrored_lines.append(line)

        else:
            mirrored_lines.append(line)

    with open(output_file, "w") as f:
        f.writelines(mirrored_lines)

    print(f"Mirrored terrain: {terrain_file} -> {output_file}")
    return output_file


# ---------------------------------- Terrain ---------------------------------


def randomize_box_x_numpy(
    params: dict,
    max_extent: float = 0.75,
    min_extent_y: Optional[float] = None,
    max_extent_y: Optional[float] = None,
    num_variants: int = 10,
    rng: Optional[Any] = None,
) -> np.ndarray:
    """
    Randomize the box shape along the local X-axis (and optionally Y-axis) and return
    an (num_variants x 10) numpy array:
    [pos_x, pos_y, pos_z,  w, x, y, z,  size_x, size_y, size_z]
    """
    if rng is None:
        rng = np.random.default_rng()

    pos = np.array(params["pos"], dtype=float)
    quat = np.array(params["quat"], dtype=float)
    size = np.array(params["size"], dtype=float)

    quat_xyzw = quat[..., [1, 2, 3, 0]]  # wxyz to xyzw
    R_quat = Rotation.from_quat(quat_xyzw)

    min_extent_x = size[0] / 2.0
    if min_extent_y is None:
        min_extent_y = size[1] / 2.0

    out = np.zeros((num_variants, 10), dtype=float)

    for i in range(num_variants):
        L_x = rng.uniform(min_extent_x, max_extent)
        R_x = rng.uniform(min_extent_x, max_extent)

        new_sx = L_x + R_x

        offset_x = 0.5 * (-L_x + R_x)

        if max_extent_y is not None:
            L_y = rng.uniform(min_extent_y, max_extent_y)
            R_y = rng.uniform(min_extent_y, max_extent_y)
            new_sy = L_y + R_y
            offset_y = 0.5 * (-L_y + R_y)
        else:
            new_sy = size[1]
            offset_y = 0.0

        new_c_local = np.array([offset_x, offset_y, 0.0])
        new_c = R_quat.apply(new_c_local) + pos

        out[i, :] = np.array(
            [new_c[0], new_c[1], new_c[2], quat[0], quat[1], quat[2], quat[3], new_sx, new_sy, size[2]]
        )

    return out


def calculate_obb_pca(vertices: list) -> Tuple[List[float], List[float], List[float]]:
    """
    Find OBB using PCA on XY plane (matching scale_local_x.py logic).
    Returns: center, size, quat (w,x,y,z)
    """
    if not vertices:
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]

    pts = np.array(vertices)
    pts_xy = pts[:, :2]
    pts_z = pts[:, 2]

    centroid_xy = np.mean(pts_xy, axis=0)
    centered_xy = pts_xy - centroid_xy

    covariance_matrix = np.cov(centered_xy.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    pts_local_xy = centered_xy @ eigenvectors

    min_local_xy = np.min(pts_local_xy, axis=0)
    max_local_xy = np.max(pts_local_xy, axis=0)

    size_xy = max_local_xy - min_local_xy
    center_local_xy = (min_local_xy + max_local_xy) / 2.0

    center_global_xy_offset = center_local_xy @ eigenvectors.T
    center_global_xy = center_global_xy_offset + centroid_xy

    min_z, max_z = np.min(pts_z), np.max(pts_z)
    size_z = max_z - min_z
    center_z = (min_z + max_z) / 2.0

    pos = [center_global_xy[0], center_global_xy[1], center_z]
    size = [size_xy[0], size_xy[1], size_z]

    local_x_axis = eigenvectors[:, 0]
    angle = np.arctan2(local_x_axis[1], local_x_axis[0])

    half_angle = angle * 0.5
    quat = [np.cos(half_angle), 0.0, 0.0, np.sin(half_angle)]  # w, x, y, z

    pos = [float(x) for x in pos]
    size = [float(x) for x in size]
    quat = [float(x) for x in quat]

    return pos, size, quat


# ------------------------------- OBJ to JSON --------------------------------


def parse_obj(obj_path: str) -> Tuple[List[List[float]], set]:
    vertices = []
    face_indices = set()

    with open(obj_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("v "):
                parts = line.split()
                try:
                    v = [float(parts[1]), float(parts[2]), float(parts[3])]
                    vertices.append(v)
                except (IndexError, ValueError):
                    continue
            elif line.startswith("f "):
                parts = line.split()
                for p in parts[1:]:
                    v_str = p.split("/")[0]
                    try:
                        idx = int(v_str)
                        if idx > 0:
                            face_indices.add(idx - 1)
                        elif idx < 0:
                            face_indices.add(len(vertices) + idx)
                    except ValueError:
                        continue

    return vertices, face_indices


def main() -> None:
    parser = argparse.ArgumentParser(description="Calculate Oriented Bounding Box (PCA-based) JSON from OBJ file.")
    parser.add_argument("--obj_file", required=True, help="Path to the input OBJ file")
    parser.add_argument("--out", help="Path to the output JSON file")
    parser.add_argument("--swap_xy", action="store_true", help="Swap XY coordinates")

    args = parser.parse_args()

    if not os.path.exists(args.obj_file):
        print(f"Error: File {args.obj_file} not found.")
        sys.exit(1)

    vertices, valid_indices = parse_obj(args.obj_file)

    if not vertices:
        print("Error: No vertices found in OBJ file.")
        sys.exit(1)

    if not valid_indices:
        print("Warning: No faces found. Using all vertices.")
        active_vertices = vertices
    else:
        active_vertices = [vertices[i] for i in valid_indices if 0 <= i < len(vertices)]

    if not active_vertices:
        print("Error: No valid vertices found after filtering.")
        sys.exit(1)

    if args.swap_xy:
        active_vertices = np.asarray(active_vertices)[:, [1, 0, 2]].tolist()
    pos, size, quat = calculate_obb_pca(active_vertices)

    data = {"pos": pos, "quat": quat, "size": size}

    output_path = args.out
    if output_path is None:
        if args.swap_xy:
            output_path = os.path.splitext(args.obj_file)[0] + "_xy_swapped.json"
        else:
            output_path = os.path.splitext(args.obj_file)[0] + ".json"

    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Generated JSON: {output_path}")
    print(f"Position: {pos}")
    print(f"Size: {size}")
    print(f"Quat: {quat}")


if __name__ == "__main__":
    main()
