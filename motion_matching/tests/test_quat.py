"""Tests for motion_matching.kinematics.quat — batched quaternion math + FK/IK."""

import numpy as np

from motion_matching.kinematics import quat as Q


def _id() -> np.ndarray:
    return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)


class TestEye:
    def test_shape_zero_returns_single(self) -> None:
        out = Q.eye([])
        assert out.shape == (4,)
        np.testing.assert_allclose(out, _id())

    def test_batch_shape(self) -> None:
        out = Q.eye([2, 3])
        assert out.shape == (2, 3, 4)
        # Every element is the identity quaternion.
        np.testing.assert_allclose(out[0, 0], _id())


class TestLengthAndNormalize:
    def test_length_unit_quaternion(self) -> None:
        np.testing.assert_allclose(Q.length(_id()), 1.0, atol=1e-6)

    def test_normalize_scaled(self) -> None:
        q = np.array([2.0, 0.0, 0.0, 0.0])
        out = Q.normalize(q)
        np.testing.assert_allclose(Q.length(out), 1.0, atol=1e-6)


class TestFromAngleAxisToXform:
    def test_zero_angle_xform_is_identity_matrix(self) -> None:
        q = Q.from_angle_axis(np.array(0.0), np.array([0.0, 0.0, 1.0]))
        x = Q.to_xform(q)
        np.testing.assert_allclose(x, np.eye(3), atol=1e-6)

    def test_90_about_z_xform(self) -> None:
        q = Q.from_angle_axis(np.array(np.pi / 2), np.array([0.0, 0.0, 1.0]))
        x = Q.to_xform(q)
        # 90 deg about z: x -> y, y -> -x
        np.testing.assert_allclose(x @ np.array([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(x @ np.array([0.0, 1.0, 0.0]), [-1.0, 0.0, 0.0], atol=1e-6)


class TestForwardKinematicsBatched:
    """Tests Q.fk against the simple per-bone forward_kinematics in core.features
    for a tiny synthetic skeleton."""

    def test_root_only(self) -> None:
        # Single root bone, batch dim of 1.
        lpos = np.array([[[0.5, 0.6, 0.7]]])  # (batch=1, bones=1, 3)
        lrot = np.array([[_id()]])  # (1, 1, 4)
        parents = [-1]
        gr, gp = Q.fk(lrot, lpos, parents)
        np.testing.assert_allclose(gp[0, 0], [0.5, 0.6, 0.7], atol=1e-6)
        np.testing.assert_allclose(gr[0, 0], _id(), atol=1e-6)

    def test_two_bone_chain_identity(self) -> None:
        # Root at (1,2,3), child local (0,0,1), identity rotations.
        # Expected child global = (1, 2, 4).
        lpos = np.array([[[1.0, 2.0, 3.0], [0.0, 0.0, 1.0]]])
        lrot = np.array([[_id(), _id()]])
        parents = [-1, 0]
        gr, gp = Q.fk(lrot, lpos, parents)
        np.testing.assert_allclose(gp[0, 1], [1.0, 2.0, 4.0], atol=1e-6)

    def test_fk_then_ik_roundtrip(self) -> None:
        # FK -> IK should recover the original local positions and rotations.
        lpos = np.array([[[1.0, 2.0, 3.0], [0.0, 0.0, 1.0], [0.5, 0.0, 0.0]]], dtype=np.float32)
        # Use non-trivial rotations.
        rot1 = Q.from_angle_axis(np.array(0.7), np.array([0.0, 0.0, 1.0]))
        rot2 = Q.from_angle_axis(np.array(0.3), np.array([1.0, 0.0, 0.0]))
        lrot = np.stack([_id(), rot1, rot2])[np.newaxis]  # (1, 3, 4)
        parents = [-1, 0, 1]

        gr, gp = Q.fk(lrot, lpos, parents)
        lrot_back, lpos_back = Q.ik(gr, gp, parents)
        np.testing.assert_allclose(lpos_back, lpos, atol=1e-5)
        # Quaternions can differ by sign; compare per-bone rotation effect on a probe vector.
        for i in range(3):
            v = np.array([1.0, 0.5, -0.2])
            a = Q.mul_vec(lrot[0, i], v)
            b = Q.mul_vec(lrot_back[0, i], v)
            np.testing.assert_allclose(a, b, atol=1e-5)


class TestQuatMul:
    def test_identity_neutral(self) -> None:
        q = Q.from_angle_axis(np.array(0.7), np.array([0.0, 1.0, 0.0]))
        out = Q.mul(_id(), q)
        np.testing.assert_allclose(out, q, atol=1e-6)

    def test_inverse_yields_identity(self) -> None:
        q = Q.from_angle_axis(np.array(1.1), np.array([1.0, 0.0, 0.0]))
        out = Q.mul(q, Q.inv(q))
        # Identity quaternion (up to sign).
        np.testing.assert_allclose(np.abs(out), [1.0, 0.0, 0.0, 0.0], atol=1e-5)
