"""Terrain-save helpers: combine obstacle meshes and randomize box variants.

Shared by ``process_scenario`` (generation) and ``run_interactive``'s
end-recording handler. Both transform a set of obstacle meshes by their
(position, wxyz) pose, append a large ground plane, and export a single OBJ;
and both produce an ``(num_obstacles, num_variants, 10)`` array of randomized
box parameters. These helpers hold the exact geometry so the two paths agree.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np

from motion_matching.utils.data_utils import randomize_box_x_numpy

# Ground-plane dimensions used for every exported terrain (very thin, centred at
# the origin with its top surface at z=0).
_GROUND_EXTENTS = [200.0, 200.0, 0.1]
_GROUND_TRANSLATION = [0.0, 0.0, -0.05]


def _point_to_polyline_distance(point: np.ndarray, polyline: np.ndarray) -> float:
    """Return the shortest planar distance from ``point`` to ``polyline``."""
    if len(polyline) == 1:
        return float(np.linalg.norm(point - polyline[0]))

    starts = polyline[:-1]
    segments = polyline[1:] - starts
    lengths_sq = np.sum(segments * segments, axis=1)
    offsets = point - starts
    t = np.divide(
        np.sum(offsets * segments, axis=1),
        lengths_sq,
        out=np.zeros_like(lengths_sq),
        where=lengths_sq > 1e-12,
    )
    t = np.clip(t, 0.0, 1.0)
    closest = starts + t[:, None] * segments
    return float(np.min(np.linalg.norm(closest - point, axis=1)))


def _oriented_boxes_overlap_2d(
    center_a: np.ndarray,
    yaw_a: float,
    size_a: np.ndarray,
    center_b: np.ndarray,
    yaw_b: float,
    size_b: np.ndarray,
    gap: float = 0.0,
) -> bool:
    """Return whether two yaw-oriented XY rectangles overlap within ``gap``."""
    axes_a = np.array(
        [[np.cos(yaw_a), -np.sin(yaw_a)], [np.sin(yaw_a), np.cos(yaw_a)]]
    )
    axes_b = np.array(
        [[np.cos(yaw_b), -np.sin(yaw_b)], [np.sin(yaw_b), np.cos(yaw_b)]]
    )
    half_a = np.asarray(size_a[:2], dtype=np.float64) * 0.5
    half_b = np.asarray(size_b[:2], dtype=np.float64) * 0.5
    relative_rotation = axes_a.T @ axes_b
    abs_rotation = np.abs(relative_rotation) + 1e-12
    translation_a = axes_a.T @ (center_b - center_a)

    for axis_idx in range(2):
        radius = half_a[axis_idx] + np.dot(half_b, abs_rotation[axis_idx]) + gap
        if abs(translation_a[axis_idx]) > radius:
            return False
    for axis_idx in range(2):
        projected_translation = abs(np.dot(translation_a, relative_rotation[:, axis_idx]))
        radius = half_b[axis_idx] + np.dot(half_a, abs_rotation[:, axis_idx]) + gap
        if projected_translation > radius:
            return False
    return True


def sample_offpath_box_obstacles(
    root_xy: np.ndarray,
    *,
    rng: Optional[Any] = None,
    min_count: int = 10,
    max_count: int = 30,
    path_clearance: float = 0.65,
    min_center_distance: Optional[float] = None,
    max_path_distance: float = 3.0,
    min_size: Sequence[float] = (0.25, 0.25, 0.20),
    max_size: Sequence[float] = (0.65, 0.65, 1.00),
    obstacle_gap: float = 0.15,
    variant_max_half_extent_x: Optional[float] = 0.75,
    position_jitter: float = 0.0,
    max_attempts: int = 30000,
) -> list[tuple[Any, np.ndarray, np.ndarray, dict]]:
    """Sample nearby box obstacles without intersecting a root trajectory.

    Candidates are anchored along the trajectory and offset along either local
    normal. Collision checks use a conservative bounding circle large enough
    for every X-extent variant produced by :func:`randomize_box_x_numpy`, not
    merely the smaller box written to the reference OBJ.

    Returns tuples in the same ``(mesh, pos, quat_wxyz, metadata)`` format used
    by traversal-skill terrain generation.
    """
    import trimesh

    root_xy = np.asarray(root_xy, dtype=np.float64)
    if root_xy.ndim != 2 or root_xy.shape[1] != 2 or len(root_xy) == 0:
        raise ValueError(f"root_xy must have shape (N, 2) with N > 0, got {root_xy.shape}")
    if not np.all(np.isfinite(root_xy)):
        raise ValueError("root_xy contains non-finite values")
    if min_count < 0 or max_count < min_count:
        raise ValueError("expected 0 <= min_count <= max_count")
    if path_clearance < 0.0 or obstacle_gap < 0.0 or position_jitter < 0.0:
        raise ValueError("clearances must be non-negative")
    if min_center_distance is not None and min_center_distance < 0.0:
        raise ValueError("min_center_distance must be non-negative")

    min_size_arr = np.asarray(min_size, dtype=np.float64)
    max_size_arr = np.asarray(max_size, dtype=np.float64)
    if min_size_arr.shape != (3,) or max_size_arr.shape != (3,):
        raise ValueError("min_size and max_size must each contain three extents")
    if np.any(min_size_arr <= 0.0) or np.any(max_size_arr < min_size_arr):
        raise ValueError("box extents must satisfy 0 < min_size <= max_size")

    # Consecutive duplicate points are common during stand segments. Removing
    # them avoids over-sampling those portions and gives well-defined tangents.
    if len(root_xy) > 1:
        keep = np.concatenate([[True], np.linalg.norm(np.diff(root_xy, axis=0), axis=1) > 1e-6])
        polyline = root_xy[keep]
    else:
        polyline = root_xy.copy()

    segments = np.diff(polyline, axis=0)
    segment_lengths = np.linalg.norm(segments, axis=1)
    cumulative_lengths = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    trajectory_length = float(cumulative_lengths[-1])

    if rng is None:
        rng = np.random.default_rng()
    target_count = int(rng.integers(min_count, max_count + 1))
    obstacles: list[tuple[Any, np.ndarray, np.ndarray, dict]] = []
    accepted: list[tuple[np.ndarray, float]] = []

    for _ in range(max_attempts):
        if len(obstacles) >= target_count:
            break

        if trajectory_length > 1e-6:
            distance_along = float(rng.uniform(0.0, trajectory_length))
            segment_idx = min(
                int(np.searchsorted(cumulative_lengths, distance_along, side="right") - 1),
                len(segments) - 1,
            )
            segment_length = segment_lengths[segment_idx]
            alpha = (distance_along - cumulative_lengths[segment_idx]) / segment_length
            anchor = polyline[segment_idx] + alpha * segments[segment_idx]
            tangent = segments[segment_idx] / segment_length
        else:
            anchor = polyline[0]
            tangent_angle = float(rng.uniform(-np.pi, np.pi))
            tangent = np.array([np.cos(tangent_angle), np.sin(tangent_angle)])

        size = rng.uniform(min_size_arr, max_size_arr)
        # NPY variants may grow to 2 * variant_max_half_extent_x along local X.
        # This circle therefore encloses both the reference box and all variants.
        half_extent_x = (
            size[0] * 0.5 if variant_max_half_extent_x is None else variant_max_half_extent_x
        )
        safety_radius = float(np.hypot(half_extent_x, size[1] * 0.5))
        # Reserve room for every NPY position variant around this reference
        # center. This keeps all variants path-safe. Reference OBJ boxes are
        # kept mutually separated below; NPY poses are independently sampled.
        placement_radius = safety_radius + position_jitter
        reference_radius = float(np.hypot(size[0] * 0.5, size[1] * 0.5))
        geometry_min_center_distance = path_clearance + placement_radius
        requested_min_center_distance = 0.0 if min_center_distance is None else min_center_distance
        effective_min_center_distance = max(requested_min_center_distance, geometry_min_center_distance)
        max_center_distance = max_path_distance - position_jitter
        if effective_min_center_distance >= max_center_distance:
            raise ValueError("max_path_distance is too small for the requested box sizes and path clearance")

        normal = np.array([-tangent[1], tangent[0]])
        side = -1.0 if float(rng.random()) < 0.5 else 1.0
        lateral_distance = float(rng.uniform(effective_min_center_distance, max_center_distance))
        center_xy = anchor + side * lateral_distance * normal

        path_distance = _point_to_polyline_distance(center_xy, polyline)
        if path_distance < effective_min_center_distance or path_distance > max_center_distance + 1e-6:
            continue
        overlaps_obstacle = any(
            np.linalg.norm(center_xy - other_xy) < reference_radius + other_radius + obstacle_gap
            for other_xy, other_radius in accepted
        )
        if overlaps_obstacle:
            continue

        yaw = float(np.arctan2(tangent[1], tangent[0]) + rng.uniform(-np.pi / 2.0, np.pi / 2.0))
        quat = np.array([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)], dtype=np.float64)
        pos = np.array([center_xy[0], center_xy[1], size[2] * 0.5], dtype=np.float64)
        mesh = trimesh.creation.box(extents=size)
        metadata = {"pos": pos.copy(), "quat": quat.copy(), "size": size.copy()}
        obstacles.append((mesh, pos, quat, metadata))
        accepted.append((center_xy, reference_radius))

    if len(obstacles) < min_count:
        raise RuntimeError(
            f"could only place {len(obstacles)} off-path obstacles (minimum requested: {min_count}); "
            "increase max_path_distance or reduce the clearance/box sizes"
        )
    return obstacles


def randomize_offpath_terrain_variants(
    metadata_list: Sequence[dict],
    root_xy: np.ndarray,
    num_variants: int,
    *,
    rng: Optional[Any] = None,
    path_clearance: float = 0.4,
    min_center_distance: Optional[float] = None,
    max_path_distance: float = 2.0,
    min_size: Sequence[float] = (0.25, 0.25, 0.20),
    max_size: Sequence[float] = (0.65, 0.65, 1.00),
    obstacle_gap: float = 0.05,
    max_attempts_per_variant: int = 30000,
) -> np.ndarray:
    """Resample a complete off-path obstacle distribution per variant.

    A variant is a coherent terrain scene, not a collection of independent
    perturbations around one base layout. For every variant we resample all
    path anchors, left/right choices, positions, yaw angles, and box dimensions
    from scratch, then jointly validate path clearance and box-to-box overlap.

    The returned array remains ``(num_boxes, num_variants, 10)`` so existing
    consumers can select a single global variant index for the whole scene.
    ``metadata_list`` determines only the fixed box count; its poses are not
    reused by any variant.
    """
    root_xy = np.asarray(root_xy, dtype=np.float64)
    if root_xy.ndim != 2 or root_xy.shape[1] != 2 or len(root_xy) == 0:
        raise ValueError(f"root_xy must have shape (N, 2) with N > 0, got {root_xy.shape}")
    if rng is None:
        rng = np.random.default_rng()
    if num_variants < 0:
        raise ValueError("num_variants must be non-negative")

    num_boxes = len(metadata_list)
    output = np.zeros((num_boxes, num_variants, 10), dtype=np.float64)

    for variant_idx in range(num_variants):
        scene = sample_offpath_box_obstacles(
            root_xy,
            rng=rng,
            min_count=num_boxes,
            max_count=num_boxes,
            path_clearance=path_clearance,
            min_center_distance=min_center_distance,
            max_path_distance=max_path_distance,
            min_size=min_size,
            max_size=max_size,
            obstacle_gap=obstacle_gap,
            variant_max_half_extent_x=None,
            position_jitter=0.0,
            max_attempts=max_attempts_per_variant,
        )
        for box_idx, (_, _, _, metadata) in enumerate(scene):
            output[box_idx, variant_idx] = np.concatenate(
                [metadata["pos"], metadata["quat"], metadata["size"]]
            )

    return output


def build_combined_terrain_mesh(posed_meshes: Sequence[tuple[Any, np.ndarray, np.ndarray]]) -> Any:
    """Transform each ``(mesh, position, wxyz)`` into place and add a ground plane.

    Args:
        posed_meshes: Sequence of ``(mesh, position, rotation_wxyz)``. Each mesh
            is copied (not mutated) and transformed by the given pose.

    Returns:
        A single combined ``trimesh.Trimesh`` (obstacles + ground plane).
    """
    import trimesh

    meshes_to_combine = []
    for mesh, position, rotation in posed_meshes:
        transform = trimesh.transformations.quaternion_matrix(rotation)
        transform[:3, 3] = position
        mesh_copy = mesh.copy()
        mesh_copy.apply_transform(transform)
        meshes_to_combine.append(mesh_copy)

    ground_plane = trimesh.creation.box(extents=_GROUND_EXTENTS)
    ground_plane.apply_translation(_GROUND_TRANSLATION)
    meshes_to_combine.append(ground_plane)

    return trimesh.util.concatenate(meshes_to_combine)


def randomize_terrain_variants(
    metadata_list: Sequence[dict], num_variants: int, *, rng: Optional[Any] = None
) -> np.ndarray:
    """Stack per-obstacle randomized box parameters into one array.

    Args:
        metadata_list: One box-parameter dict per obstacle.
        num_variants: Number of randomized variants per obstacle.
        rng: Optional numpy Generator. ``None`` (default) preserves production
            behaviour, where ``randomize_box_x_numpy`` seeds its own unseeded
            generator (so the output is non-deterministic run-to-run).

    Returns:
        ``(num_obstacles, num_variants, 10)`` float array. If ``metadata_list``
        is empty, the obstacle dimension is zero while the requested variant
        dimension is preserved.
    """
    random_terrains = []
    for metadata in metadata_list:
        rt = randomize_box_x_numpy(metadata, num_variants=num_variants, rng=rng)
        random_terrains.append(rt[None])

    if random_terrains:
        return np.concatenate(random_terrains, axis=0)
    return np.zeros((0, num_variants, 10), dtype=np.float32)
