"""Unit tests for the extracted io helpers (Stage 5).

The interactive end-recording path that also uses these is not covered by the
generation golden, so exercise the helpers directly here.
"""

from __future__ import annotations

import numpy as np
import pytest

from motion_matching.io import (
    build_combined_terrain_mesh,
    randomize_offpath_terrain_variants,
    randomize_terrain_variants,
    sample_offpath_box_obstacles,
    save_motion_npz,
)
from tests._hashing import hash_array
from motion_matching.io.terrain_io import _oriented_boxes_overlap_2d
from motion_matching.motion_visualizer import reconstruct_box_mesh


def _qpos(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((n, 36)).astype(np.float32)
    q[:, :4] /= np.linalg.norm(q[:, :4], axis=1, keepdims=True)
    return q


def test_terrain_npy_box_mesh() -> None:
    yaw_quat = np.array([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)])

    center = np.array([1.0, 2.0, 0.5])
    size = np.array([0.4, 0.6, 1.0])
    mesh = reconstruct_box_mesh(center, size, yaw_quat)
    np.testing.assert_allclose(mesh.centroid, center, atol=1e-7)
    np.testing.assert_allclose(np.sort(mesh.extents[:2]), np.sort(size[:2]), atol=1e-7)
    assert mesh.extents[2] == pytest.approx(size[2])


class TestSaveMotionNpz:
    def test_writes_tracking_schema(self, tmp_path) -> None:
        out = tmp_path / "m.npz"
        tracking = save_motion_npz(_qpos(20, 1), str(out), target_fps=50, source_fps=60)
        assert out.is_file()
        loaded = np.load(out)
        assert set(loaded.files) == set(tracking)
        assert int(loaded["fps"][0]) == 50

    def test_vel_cmd_attached_when_given(self, tmp_path) -> None:
        out = tmp_path / "m.npz"
        # 50/60 resample of 24 frames -> 20 frames; vel_cmd is stored verbatim.
        vel = np.ones((20, 3), dtype=np.float32)
        tracking = save_motion_npz(_qpos(24, 2), str(out), target_fps=50, source_fps=60, vel_cmd=vel)
        assert "vel_cmd" in tracking
        assert np.load(out)["vel_cmd"].shape == (20, 3)

    def test_no_vel_cmd_key_when_omitted(self, tmp_path) -> None:
        out = tmp_path / "m.npz"
        tracking = save_motion_npz(_qpos(20, 3), str(out), target_fps=50, source_fps=50)
        assert "vel_cmd" not in tracking

    def test_equal_fps_is_passthrough(self, tmp_path) -> None:
        # source==target -> resample_data returns the input unchanged, so the
        # tracking output matches feeding qpos straight to qpos_to_tracking.
        from motion_matching.utils.tracking_format import qpos_to_tracking

        qp = _qpos(12, 4)
        out = tmp_path / "m.npz"
        tracking = save_motion_npz(qp, str(out), target_fps=50, source_fps=50)
        direct = qpos_to_tracking(qp, 50)
        for key in direct:
            assert hash_array(tracking[key]) == hash_array(direct[key]), key


class TestRandomizeTerrainVariants:
    def test_empty_preserves_variant_dimension(self) -> None:
        out = randomize_terrain_variants([], num_variants=10)
        assert out.shape == (0, 10, 10)

    def test_seeded_matches_direct_stack(self) -> None:
        from motion_matching.utils.data_utils import randomize_box_x_numpy

        meta = [
            {"pos": [1.0, 0.0, 0.5], "quat": [1.0, 0.0, 0.0, 0.0], "size": [0.4, 0.6, 0.3]},
            {"pos": [2.0, 1.0, 0.5], "quat": [1.0, 0.0, 0.0, 0.0], "size": [0.5, 0.5, 0.3]},
        ]
        out = randomize_terrain_variants(meta, num_variants=5, rng=np.random.default_rng(0))
        assert out.shape == (2, 5, 10)
        # Same as stacking per-obstacle randomize with the same generator stream.
        rng = np.random.default_rng(0)
        expected = np.stack([randomize_box_x_numpy(m, num_variants=5, rng=rng) for m in meta], axis=0)
        assert hash_array(out) == hash_array(expected)


class TestBuildCombinedTerrainMesh:
    def test_ground_plane_only_when_no_obstacles(self) -> None:
        trimesh = pytest.importorskip("trimesh")
        mesh = build_combined_terrain_mesh([])
        # Ground plane is a box: 8 vertices, 12 triangles.
        assert isinstance(mesh, trimesh.Trimesh)
        assert len(mesh.vertices) == 8
        assert len(mesh.faces) == 12

    def test_obstacle_plus_ground(self) -> None:
        trimesh = pytest.importorskip("trimesh")
        box = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
        identity = np.array([1.0, 0.0, 0.0, 0.0])
        mesh = build_combined_terrain_mesh([(box, np.array([0.0, 0.0, 1.0]), identity)])
        # One obstacle box (8v/12f) + ground plane (8v/12f).
        assert len(mesh.vertices) == 16
        assert len(mesh.faces) == 24


class TestSampleOffpathBoxObstacles:
    def test_explicit_center_distance_range(self) -> None:
        root_xy = np.column_stack([np.linspace(0.0, 20.0, 401), np.zeros(401)])
        obstacles = sample_offpath_box_obstacles(
            root_xy,
            rng=np.random.default_rng(21),
            min_count=8,
            max_count=8,
            path_clearance=0.0,
            min_center_distance=0.3,
            max_path_distance=1.5,
            min_size=(0.25, 0.25, 0.2),
            max_size=(0.65, 0.65, 1.0),
            variant_max_half_extent_x=None,
        )

        for _, pos, _, metadata in obstacles:
            center_distance = abs(pos[1])
            footprint_radius = np.hypot(metadata["size"][0] * 0.5, metadata["size"][1] * 0.5)
            assert 0.3 <= center_distance <= 1.5
            assert center_distance >= footprint_radius

    def test_count_format_and_path_clearance(self) -> None:
        # A long straight path provides enough room to exercise the full 10--30
        # requested range without relying on a generated motion database.
        root_xy = np.column_stack([np.linspace(0.0, 20.0, 401), np.zeros(401)])
        obstacles = sample_offpath_box_obstacles(root_xy, rng=np.random.default_rng(7))

        assert 10 <= len(obstacles) <= 30
        for mesh, pos, quat, metadata in obstacles:
            assert len(mesh.vertices) == 8
            assert pos.shape == (3,)
            assert quat.shape == (4,)
            assert set(metadata) == {"pos", "quat", "size"}

            size = metadata["size"]
            variant_radius = np.hypot(0.75, size[1] * 0.5)
            # For this x-axis path, distance to the path is abs(y).
            assert abs(pos[1]) >= 0.65 + variant_radius - 1e-9
            assert abs(pos[1]) <= 3.0 + 1e-9
            assert pos[2] == pytest.approx(size[2] * 0.5)

        # The saved NPY contains asymmetric X-extent variants whose centers
        # shift relative to the reference boxes. Verify those final oriented
        # rectangles also stay outside the path clearance.
        variants = randomize_terrain_variants(
            [metadata for *_, metadata in obstacles],
            num_variants=10,
            rng=np.random.default_rng(99),
        )
        for box_variants in variants:
            for variant in box_variants:
                yaw = 2.0 * np.arctan2(variant[6], variant[3])
                half_y_projection = (
                    abs(np.sin(yaw)) * variant[7] * 0.5
                    + abs(np.cos(yaw)) * variant[8] * 0.5
                )
                assert abs(variant[1]) - half_y_projection >= 0.65 - 1e-9

    def test_seeded_layout_is_deterministic(self) -> None:
        root_xy = np.column_stack([np.linspace(0.0, 20.0, 401), np.zeros(401)])
        first = sample_offpath_box_obstacles(root_xy, rng=np.random.default_rng(11))
        second = sample_offpath_box_obstacles(root_xy, rng=np.random.default_rng(11))

        first_metadata = np.stack([np.concatenate([m["pos"], m["quat"], m["size"]]) for *_, m in first])
        second_metadata = np.stack([np.concatenate([m["pos"], m["quat"], m["size"]]) for *_, m in second])
        np.testing.assert_array_equal(first_metadata, second_metadata)


class TestRandomizeOffpathTerrainVariants:
    def test_randomizes_position_yaw_and_shape_while_staying_off_path(self) -> None:
        root_xy = np.column_stack([np.linspace(0.0, 20.0, 401), np.zeros(401)])
        obstacles = sample_offpath_box_obstacles(
            root_xy,
            rng=np.random.default_rng(3),
            min_count=10,
            max_count=10,
            path_clearance=0.4,
            max_path_distance=2.0,
        )
        base_positions = np.stack([metadata["pos"][:2] for *_, metadata in obstacles])
        variants = randomize_offpath_terrain_variants(
            [metadata for *_, metadata in obstacles],
            root_xy,
            num_variants=10,
            rng=np.random.default_rng(4),
            path_clearance=0.4,
            max_path_distance=2.0,
        )

        assert variants.shape == (10, 10, 10)
        # Slot identities are deliberately not preserved: each column is a
        # newly sampled scene, so positions can move far beyond local jitter.
        first_scene_displacement = np.linalg.norm(variants[:, 0, :2] - base_positions, axis=1)
        assert np.median(first_scene_displacement) > 0.5
        scene_to_scene_displacement = np.linalg.norm(variants[:, 1, :2] - variants[:, 0, :2], axis=1)
        assert np.median(scene_to_scene_displacement) > 0.5

        for box_variants in variants:
            assert np.ptp(box_variants[:, 0]) > 0.0 or np.ptp(box_variants[:, 1]) > 0.0
            yaws = 2.0 * np.arctan2(box_variants[:, 6], box_variants[:, 3])
            assert np.ptp(np.unwrap(yaws)) > 0.0
            assert np.ptp(box_variants[:, 7]) > 0.0

            for variant, yaw in zip(box_variants, yaws):
                half_y_projection = (
                    abs(np.sin(yaw)) * variant[7] * 0.5
                    + abs(np.cos(yaw)) * variant[8] * 0.5
                )
                assert abs(variant[1]) - half_y_projection >= 0.4 - 1e-9
                assert abs(variant[1]) <= 2.0 + 1e-9

        for variant_idx in range(variants.shape[1]):
            scene = variants[:, variant_idx]
            for i in range(len(scene)):
                for j in range(i):
                    yaw_i = 2.0 * np.arctan2(scene[i, 6], scene[i, 3])
                    yaw_j = 2.0 * np.arctan2(scene[j, 6], scene[j, 3])
                    assert not _oriented_boxes_overlap_2d(
                        scene[i, :2],
                        yaw_i,
                        scene[i, 7:10],
                        scene[j, :2],
                        yaw_j,
                        scene[j, 7:10],
                        0.05 - 1e-9,
                    )
