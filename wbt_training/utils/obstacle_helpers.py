"""On-path obstacle generation and motion offset utilities.

Pure numpy/torch/trimesh
operations with no IsaacLab or RSL-RL dependencies. IsaacLab-specific helpers
(``generate_rigid_object_collection_from_list``, ``add_offpath_obstacle``,
``load_offpath_obstacle_data``) are intentionally omitted.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import torch
import trimesh

# ---------------------------------------------------------------------------
# Pure tensor helpers
# ---------------------------------------------------------------------------


def sample_obstacle(
    env_ids: Sequence[int],
    motion_idx: torch.Tensor,
    obstacle_poses: torch.Tensor,
    obstacle_counts: torch.Tensor,
    object_state: torch.Tensor,
    motion_grid: torch.Tensor,
) -> None:
    """Sample obstacle placements for the given environments.

    Parameters
    ----------
    env_ids : sequence of int
        Indices of the environments to populate.
    motion_idx : Tensor (E,)
        Per-env motion segment index.
    obstacle_poses : Tensor (num_obstacles, num_variants, 10)
        Candidate obstacle poses (xyz + quat_xyzw + scale_xyz).
    obstacle_counts : Tensor (num_motions + 1,)
        Cumulative obstacle count boundaries per motion.
    object_state : Tensor (E, max_obstacles, 10)
        **Modified in-place.** Receives sampled obstacle state.
    motion_grid : Tensor (num_motions, 2)
        XY offsets for each motion segment.
    """
    starts = obstacle_counts[motion_idx]
    ends = obstacle_counts[motion_idx + 1]
    lengths = ends - starts

    E = len(env_ids)
    L = int(lengths.max().item())
    M = obstacle_poses.shape[1]

    base = torch.arange(L)
    idx = starts[:, None] + base[None, :]
    mask = base[None, :] < lengths[:, None]
    e_idx = torch.arange(E)[:, None].expand(E, L)

    selected_variants = torch.randint(0, M, (E, L), dtype=torch.long)

    e_idx_flat = e_idx[mask]
    obj_idx_flat = idx[mask].long()
    vars_flat = selected_variants[mask]

    object_state[e_idx_flat, obj_idx_flat, :10] = obstacle_poses[
        obj_idx_flat, vars_flat, :10
    ]
    mg = motion_grid[:, None, :].expand(-1, L, -1)
    object_state[e_idx_flat, obj_idx_flat, :2] += mg[motion_idx, :, :2].reshape(-1, 2)


def compute_motion_grids(
    data, margin_ratio: float = 0.1
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute per-motion-segment XY grid layout and offsets.

    Returns ``(offsets_xy, per_frame_offsets_xy, seg_id, width_total, height_total)``.
    """
    body_pos_w = torch.tensor(data["body_pos_w"], dtype=torch.float32)

    try:
        motion_ends = torch.tensor(data["motion_ends"], dtype=torch.bool)
    except KeyError:
        motion_ends = torch.zeros(body_pos_w.shape[0], dtype=torch.bool)
        motion_ends[-1] = True

    seg_id_raw = torch.cumsum(motion_ends.to(torch.long), dim=0)
    seg_id = torch.roll(seg_id_raw, 1)
    seg_id[0] = 0

    num_segments = int(seg_id[-1].item()) + 1
    root_xy = body_pos_w[:, 0, :2]

    mins = torch.full((num_segments, 2), float("inf"))
    maxs = torch.full((num_segments, 2), float("-inf"))
    seg_id_2d = seg_id.unsqueeze(1).expand(-1, 2)
    mins = mins.scatter_reduce(0, seg_id_2d, root_xy, reduce="amin")
    maxs = maxs.scatter_reduce(0, seg_id_2d, root_xy, reduce="amax")

    sizes = (maxs - mins).clamp_min(1e-6)
    width = sizes[:, 0].max()
    height = sizes[:, 1].max()

    min_dim_t = torch.tensor(3.0, dtype=sizes.dtype)
    width = torch.maximum(width, min_dim_t)
    height = torch.maximum(height, min_dim_t)

    origins_xy = torch.zeros(num_segments, 2, dtype=torch.float32)
    cols_count = math.ceil(math.sqrt(num_segments))
    for i in range(num_segments):
        row = i // cols_count
        col = i % cols_count
        origins_xy[i, 0] = col * (width * (1.0 + margin_ratio) + 0.2)
        origins_xy[i, 1] = row * (height * (1.0 + margin_ratio) + 0.2)

    width_total = cols_count * (width * (1.0 + margin_ratio) + 0.2)
    height_total = cols_count * (height * (1.0 + margin_ratio) + 0.2)

    offsets_xy = origins_xy - mins
    per_frame_offsets_xy = offsets_xy[seg_id]

    return offsets_xy, per_frame_offsets_xy, seg_id, width_total, height_total


def compute_env_origins_grid(num_envs: int, env_spacing: float) -> np.ndarray:
    """Compute environment origins laid out in a grid.

    Mirrors IsaacLab's ``TerrainImporter._compute_env_origins_grid()``.

    Returns array of shape ``(num_envs, 3)``.
    """
    env_origins = np.zeros((num_envs, 3))
    num_rows = int(np.ceil(num_envs / int(np.sqrt(num_envs))))
    num_cols = int(np.ceil(num_envs / num_rows))

    ii, jj = np.meshgrid(np.arange(num_rows), np.arange(num_cols), indexing="ij")
    ii_flat = ii.flatten()[:num_envs]
    jj_flat = jj.flatten()[:num_envs]

    env_origins[:, 0] = -(ii_flat - (num_rows - 1) / 2) * env_spacing
    env_origins[:, 1] = (jj_flat - (num_cols - 1) / 2) * env_spacing
    env_origins[:, 2] = 0.0
    return env_origins


def offset_motions_to_file(
    data,
    env_origins: np.ndarray,
    per_frame_offsets_xy: torch.Tensor,
    file_path: str = "motion.npz",
) -> None:
    """Apply grid offsets and per-env origins to motion data, then save as NPZ."""
    body_pos_w = torch.tensor(data["body_pos_w"], dtype=torch.float32)
    body_pos_w[:, :, :2] += per_frame_offsets_xy.unsqueeze(1)

    E = int(env_origins.shape[0])
    T = int(body_pos_w.shape[0])
    env_origins_t = torch.tensor(env_origins, dtype=body_pos_w.dtype)

    body_pos_w = body_pos_w.unsqueeze(0) + env_origins_t[:, None, None, :]
    body_pos_w = body_pos_w.reshape(E * T, *body_pos_w.shape[2:])

    replicated_data = {}
    for k, v in data.items():
        if k == "body_pos_w":
            continue
        arr = np.asarray(v)
        if arr.ndim >= 1 and arr.shape[0] == T:
            replicated_data[k] = np.concatenate([arr] * E, axis=0)
        else:
            replicated_data[k] = v

    replicated_data["body_pos_w"] = body_pos_w.numpy()
    np.savez(file_path, **replicated_data)


# ---------------------------------------------------------------------------
# Mesh generation
# ---------------------------------------------------------------------------


def save_obstacles_to_file(
    object_state: torch.Tensor | np.ndarray,
    env_origins: np.ndarray,
    env_spacing: float = 20.0,
) -> tuple[list[trimesh.Trimesh], trimesh.Trimesh]:
    """Build obstacle cube meshes with noise and return per-env + combined meshes.

    Parameters
    ----------
    object_state : array-like (..., 10)
        Each obstacle encoded as ``[cx, cy, cz, qx, qy, qz, qw, sx, sy, sz]``.
    env_origins : ndarray (num_envs, 3)
    env_spacing : float

    Returns
    -------
    tuple
        ``(terrain_per_env_list, terrain_entire)``
    """

    def _quat_to_rotation_matrix(qx, qy, qz, qw):
        xx, yy, zz = qx * qx, qy * qy, qz * qz
        xy, xz, yz = qx * qy, qx * qz, qy * qz
        wx, wy, wz = qw * qx, qw * qy, qw * qz
        return [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ]

    def _quat_from_z_rotation(angle):
        ha = angle / 2.0
        return (math.cos(ha), 0.0, 0.0, math.sin(ha))

    def _quat_multiply(q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        )

    def _reconstruct_corners(center, scale, quat, yaw_noise_range=None):
        hx, hy, hz = (s / 2.0 for s in scale)
        qw, qx, qy, qz = quat

        if yaw_noise_range is not None:
            yaw_noise = np.random.uniform(yaw_noise_range[0], yaw_noise_range[1])
            yaw_quat = _quat_from_z_rotation(yaw_noise)
            qw, qx, qy, qz = _quat_multiply((qw, qx, qy, qz), yaw_quat)

        R = _quat_to_rotation_matrix(qx, qy, qz, qw)
        local_points = [
            (-hx, -hy, -hz),
            (hx, -hy, -hz),
            (hx, hy, -hz),
            (-hx, hy, -hz),
            (-hx, -hy, hz),
            (hx, -hy, hz),
            (hx, hy, hz),
            (-hx, hy, hz),
        ]
        corners = []
        for lx, ly, lz in local_points:
            rx = R[0][0] * lx + R[0][1] * ly + R[0][2] * lz
            ry = R[1][0] * lx + R[1][1] * ly + R[1][2] * lz
            rz = R[2][0] * lx + R[2][1] * ly + R[2][2] * lz
            corners.append((center[0] + rx, center[1] + ry, center[2] + rz))
        return corners

    def _add_uniform_noise_to_vertices(
        vertices,
        height_threshold=0.2,
        uniform_noise_range=(-0.05, 0.05),
        individual_noise_range=(-0.02, 0.02),
    ):
        v_boxes = np.asarray(vertices, dtype=np.float32)
        high_z_mask = v_boxes[..., 2] > height_threshold
        mask3 = high_z_mask[..., None].astype(np.float32)
        box_size = v_boxes.shape[1]
        num_boxes = v_boxes.shape[0]

        uniform_noise = np.random.uniform(
            low=uniform_noise_range[0],
            high=uniform_noise_range[1],
            size=(num_boxes, 1, 3),
        )
        individual_noise = np.random.uniform(
            low=individual_noise_range[0],
            high=individual_noise_range[1],
            size=(num_boxes, box_size, 3),
        )
        total_noise = uniform_noise * mask3 + individual_noise * mask3
        return v_boxes + total_noise

    faces = [
        (0, 3, 2, 1),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (3, 7, 6, 2),
        (0, 4, 7, 3),
        (1, 2, 6, 5),
    ]

    object_state = np.asarray(object_state, dtype=float)
    object_state = object_state.copy()
    object_state[:, :, :3] += env_origins[:, None, :]

    # Add a floor object
    floor_size = env_origins.max(axis=0) - env_origins.min(axis=0) + 3 * env_spacing
    floor_center = (
        env_origins.max(axis=0) + env_origins.min(axis=0)
    ) / 2.0 + env_spacing / 2
    floor_center[2] = -0.05
    floor_scale = np.array([floor_size[0], floor_size[1], 0.1])
    floor_quat = np.array([1.0, 0.0, 0.0, 0.0])
    floor_obj = np.concatenate([floor_center, floor_quat, floor_scale]).reshape(1, 10)

    obj_list = object_state.reshape(-1, 10)
    obj_list = np.concatenate([obj_list, floor_obj], axis=0)

    corners_all = []
    for i, obj in enumerate(obj_list):
        center = obj[0:3].tolist()
        quat = obj[3:7].tolist()
        scale = obj[7:10].tolist()
        yaw_range = (-np.pi, np.pi) if i != obj_list.shape[0] - 1 else None
        corners_all.append(
            _reconstruct_corners(center, scale, quat, yaw_noise_range=yaw_range)
        )

    corners_all_noisy = _add_uniform_noise_to_vertices(corners_all)

    faces_tris = []
    for quad in faces:
        faces_tris.append([quad[0], quad[1], quad[2]])
        faces_tris.append([quad[0], quad[2], quad[3]])
    faces_tris_np = np.array(faces_tris)

    def _generate_terrain_mesh(corners_noisy, ftri):
        num_obstacles = corners_noisy.shape[0]
        all_vertices = corners_noisy.reshape(-1, 3)
        offsets = (np.arange(num_obstacles) * 8)[:, None, None]
        all_faces = ftri[None, :, :] + offsets
        all_faces = all_faces.reshape(-1, 3)
        return trimesh.Trimesh(vertices=all_vertices, faces=all_faces)

    terrain_entire = _generate_terrain_mesh(corners_all_noisy, faces_tris_np)

    terrain_per_env_list: list[trimesh.Trimesh] = []
    num_envs = object_state.shape[0]
    num_obstacles_per_env = object_state.shape[1]
    for i in range(num_envs):
        corner_indices = np.concatenate(
            (
                np.arange(i * num_obstacles_per_env, (i + 1) * num_obstacles_per_env),
                [-1],
            )
        )
        mesh = _generate_terrain_mesh(corners_all_noisy[corner_indices], faces_tris_np)
        mesh.apply_translation(-env_origins[i])
        terrain_per_env_list.append(mesh)

    return terrain_per_env_list, terrain_entire


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def sample_obstacles_for_motions(
    num_envs: int, motion_file: str, obstacle_file: str, seed: int | None = None
) -> tuple[float, str, list[trimesh.Trimesh], trimesh.Trimesh]:
    """Main entry-point: sample obstacles and compute terrain meshes.

    Returns ``(env_spacing, motion_file, terrain_per_env_list, terrain_entire)``.
    """
    with np.load(obstacle_file) as o_data, np.load(motion_file) as m_data:
        obstacle_poses = torch.tensor(o_data["obj_list"], dtype=torch.float32)
        obstacle_counts = torch.tensor(o_data["obj_count_list"], dtype=torch.long)

        E = num_envs
        env_ids = torch.arange(E)
        num_obstacles = obstacle_counts[-1]
        object_state = torch.zeros(E, num_obstacles, 10)

        offsets_xy, per_frame_offsets_xy, _seg_id, width, height = compute_motion_grids(
            m_data
        )
        min_spacing = max(width, height) + 1.0

        num_motions = offsets_xy.shape[0]
        if num_obstacles != 0:
            for motion_id in range(num_motions):
                motion_idx = torch.zeros_like(env_ids) + motion_id
                sample_obstacle(
                    env_ids,
                    motion_idx,
                    obstacle_poses,
                    obstacle_counts,
                    object_state,
                    offsets_xy,
                )

        env_origins = compute_env_origins_grid(E, min_spacing.numpy())

        if seed is not None:
            motion_file = str(motion_file).replace(".npz", f"_{seed}.npz")

        offset_motions_to_file(m_data, env_origins, per_frame_offsets_xy, motion_file)

    terrain_per_env_list, terrain_entire = save_obstacles_to_file(
        object_state, env_origins, min_spacing.numpy()
    )
    return float(min_spacing.numpy()), motion_file, terrain_per_env_list, terrain_entire


# ---------------------------------------------------------------------------
# Standalone add_onpath_obstacle (no env_cfg mutation)
# ---------------------------------------------------------------------------


def add_onpath_obstacle_standalone(
    num_variants: int, motion_file: str, obstacle_file: str, seed: int | None = None
) -> tuple[float, str, trimesh.Trimesh, list[trimesh.Trimesh]]:
    """Process on-path obstacles and return data without modifying any env config.

    Unlike the original ``add_onpath_obstacle()`` which mutated
    ``env_cfg`` directly. The caller is responsible for integrating the
    returned data into the training configuration.

    Parameters
    ----------
    num_variants : int
        Number of terrain/obstacle variants (typically ``num_envs``).
    motion_file : str
        Path to motion NPZ file.
    obstacle_file : str
        Path to terrain/obstacle NPZ file.
    seed : int | None
        Random seed for reproducible obstacle sampling.

    Returns
    -------
    tuple
        ``(env_spacing, motion_file, terrain_mesh, terrain_per_env_list)``
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    env_spacing, motion_file, terrain_per_env_list, terrain_entire = (
        sample_obstacles_for_motions(num_variants, motion_file, obstacle_file, seed)
    )

    return env_spacing, motion_file, terrain_entire, terrain_per_env_list
