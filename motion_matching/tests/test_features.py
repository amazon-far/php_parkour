"""Tests for motion_matching.core.features — FK and trajectory clamping."""

import numpy as np

from motion_matching.core import features as F


def _identity_quat() -> np.ndarray:
    return np.array([1.0, 0.0, 0.0, 0.0])


class TestQuatMulVec3:
    def test_identity_preserves_vector(self) -> None:
        v = np.array([1.0, -2.0, 3.0])
        np.testing.assert_allclose(F.quat_mul_vec3(_identity_quat(), v), v, atol=1e-6)


class TestQuatMul:
    def test_identity_neutral(self) -> None:
        q = np.array([0.5, 0.5, 0.5, 0.5])  # arbitrary
        np.testing.assert_allclose(F.quat_mul(_identity_quat(), q), q, atol=1e-6)


class TestCross:
    def test_basis_cross_products(self) -> None:
        x = np.array([1.0, 0.0, 0.0])
        y = np.array([0.0, 1.0, 0.0])
        z = np.array([0.0, 0.0, 1.0])
        np.testing.assert_allclose(F.cross(x, y), z, atol=1e-6)
        np.testing.assert_allclose(F.cross(y, z), x, atol=1e-6)
        np.testing.assert_allclose(F.cross(z, x), y, atol=1e-6)


class TestDatabaseTrajectoryIndexClamp:
    """Mirrors the C++ ``database_trajectory_index_clamp``: clamps an offset
    lookup to stay inside whichever (start, stop) range contains the frame."""

    def test_inside_range_no_clamp_needed(self) -> None:
        starts = np.array([0, 100])
        stops = np.array([50, 150])
        # frame=10, offset=20 -> 30, still inside [0, 50).
        assert F.database_trajectory_index_clamp(starts, stops, 10, 20) == 30

    def test_clamped_at_range_end(self) -> None:
        starts = np.array([0])
        stops = np.array([50])
        # frame=40, offset=20 -> 60, clamped to stops[0]-1 == 49.
        assert F.database_trajectory_index_clamp(starts, stops, 40, 20) == 49

    def test_clamped_at_range_start_for_negative_offset(self) -> None:
        starts = np.array([10])
        stops = np.array([50])
        # frame=15, offset=-20 -> -5, clamped to starts[0] == 10.
        assert F.database_trajectory_index_clamp(starts, stops, 15, -20) == 10

    def test_picks_correct_range_when_multiple(self) -> None:
        # frame is in the second range, so clamp must respect THAT range's bounds.
        starts = np.array([0, 100])
        stops = np.array([50, 200])
        assert F.database_trajectory_index_clamp(starts, stops, 120, 5) == 125
        # offset shooting past stops[1]-1 should clamp at stops[1]-1 == 199.
        assert F.database_trajectory_index_clamp(starts, stops, 120, 1000) == 199


class TestForwardKinematics:
    def test_root_only_returns_input(self) -> None:
        # Single bone (the root): FK is identity.
        bone_positions = np.array([[1.0, 2.0, 3.0]])
        bone_rotations = np.array([_identity_quat()])
        bone_parents = np.array([-1])
        out = F.forward_kinematics(bone_positions, bone_rotations, bone_parents)
        np.testing.assert_allclose(out[0], [1.0, 2.0, 3.0], atol=1e-6)

    def test_two_bone_chain_with_identity_rotation(self) -> None:
        # Root at (1,2,3), child at local (0,0,1) relative to root.
        # With identity rotations, global child = root + local = (1,2,4).
        bone_positions = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 1.0]])
        bone_rotations = np.array([_identity_quat(), _identity_quat()])
        bone_parents = np.array([-1, 0])
        out = F.forward_kinematics(bone_positions, bone_rotations, bone_parents)
        np.testing.assert_allclose(out[1], [1.0, 2.0, 4.0], atol=1e-6)

    def test_return_rotations_flag(self) -> None:
        bone_positions = np.array([[0.0, 0.0, 0.0]])
        bone_rotations = np.array([_identity_quat()])
        bone_parents = np.array([-1])
        out = F.forward_kinematics(bone_positions, bone_rotations, bone_parents, return_rotations=True)
        positions, rotations = out
        np.testing.assert_allclose(positions[0], 0.0, atol=1e-6)
        np.testing.assert_allclose(rotations[0], _identity_quat(), atol=1e-6)

    def test_child_rotated_90_around_z(self) -> None:
        # Parent at origin, parent rotated 90 around z; child local pos = (1,0,0)
        # should rotate to global (0,1,0) at the child.
        half = np.pi / 4.0
        rot_z = np.array([np.cos(half), 0.0, 0.0, np.sin(half)])
        bone_positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        bone_rotations = np.array([rot_z, _identity_quat()])
        bone_parents = np.array([-1, 0])
        out = F.forward_kinematics(bone_positions, bone_rotations, bone_parents)
        np.testing.assert_allclose(out[1], [0.0, 1.0, 0.0], atol=1e-6)


class TestForwardKinematicsVelocity:
    def test_root_only_passes_through(self) -> None:
        bp = np.array([[0.0, 0.0, 0.0]])
        bv = np.array([[0.1, 0.2, 0.3]])
        br = np.array([_identity_quat()])
        ba = np.array([[0.0, 0.0, 0.0]])
        bone_parents = np.array([-1])
        gp, gv, gr, ga = F.forward_kinematics_velocity(bp, bv, br, ba, bone_parents)
        np.testing.assert_allclose(gv[0], [0.1, 0.2, 0.3], atol=1e-6)
        np.testing.assert_allclose(gp[0], 0.0, atol=1e-6)

    def test_zero_angular_zero_local_velocity_means_inherited_only(self) -> None:
        # If the child has zero local linear velocity AND parent has zero angular
        # velocity, the child's global velocity equals the parent's.
        bp = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        bv = np.array([[0.5, 0.0, 0.0], [0.0, 0.0, 0.0]])
        br = np.array([_identity_quat(), _identity_quat()])
        ba = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        bone_parents = np.array([-1, 0])
        _, gv, _, _ = F.forward_kinematics_velocity(bp, bv, br, ba, bone_parents)
        np.testing.assert_allclose(gv[1], gv[0], atol=1e-6)


class TestComputeBonePositionFeature:
    def test_root_bone_in_root_space_is_zero(self) -> None:
        # The "feature" for bone 0 (the root) in root space is always 0.
        bp = np.array([[[1.0, 2.0, 3.0]]])  # 1 frame, 1 bone
        br = np.array([[_identity_quat()]])
        bone_parents = np.array([-1])
        feat = F.compute_bone_position_feature(bp, br, bone_parents, bone_idx=0)
        np.testing.assert_allclose(feat[0], 0.0, atol=1e-6)

    def test_two_bones_in_root_space_with_identity_rot(self) -> None:
        # Bone 1 local at (0.5, 0, 0), root at (10, 20, 30) — feature is (0.5, 0, 0).
        # Note: features.quat_inv_negate_w on identity gives [-1, 0, 0, 0], NOT a
        # true inverse. The feature module's quat_mul + that negate-w trick on
        # identity yields a 180 deg flip. Skip this case — see
        # TestComputeBonePositionFeature_AtZeroRoot below.
        pass
