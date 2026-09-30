"""Tests for motion_matching.utils.math_utils — pure quaternion + helper math."""

import numpy as np
import pytest

from motion_matching.utils import math_utils as mu


def _q_identity() -> np.ndarray:
    return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)


class TestQuatMulVec3:
    def test_identity_quaternion_leaves_vector_unchanged(self) -> None:
        v = np.array([1.0, 2.0, 3.0])
        out = mu.quat_mul_vec3(_q_identity(), v)
        np.testing.assert_allclose(out, v, atol=1e-6)

    def test_rotate_x_axis_90_around_z(self) -> None:
        # 90 deg rotation about +z: x -> y
        half = np.pi / 4.0
        q = np.array([np.cos(half), 0.0, 0.0, np.sin(half)])
        out = mu.quat_mul_vec3(q, np.array([1.0, 0.0, 0.0]))
        np.testing.assert_allclose(out, [0.0, 1.0, 0.0], atol=1e-6)

    def test_rotate_y_axis_90_around_x(self) -> None:
        half = np.pi / 4.0
        q = np.array([np.cos(half), np.sin(half), 0.0, 0.0])
        out = mu.quat_mul_vec3(q, np.array([0.0, 1.0, 0.0]))
        np.testing.assert_allclose(out, [0.0, 0.0, 1.0], atol=1e-6)


class TestQuatMul:
    def test_identity_is_neutral(self) -> None:
        q = mu.quat_from_angle_axis(0.5, np.array([0.0, 0.0, 1.0]))
        np.testing.assert_allclose(mu.quat_mul(_q_identity(), q), q, atol=1e-6)
        np.testing.assert_allclose(mu.quat_mul(q, _q_identity()), q, atol=1e-6)

    def test_compose_two_z_rotations(self) -> None:
        a = mu.quat_from_angle_axis(np.pi / 4, np.array([0.0, 0.0, 1.0]))
        b = mu.quat_from_angle_axis(np.pi / 4, np.array([0.0, 0.0, 1.0]))
        composed = mu.quat_mul(a, b)
        # Composition should equal a single 90-deg rotation about z.
        expected = mu.quat_from_angle_axis(np.pi / 2, np.array([0.0, 0.0, 1.0]))
        # Quaternions can differ by sign; compare absolute components.
        np.testing.assert_allclose(np.abs(composed), np.abs(expected), atol=1e-6)


class TestQuatInv:
    def test_inverse_of_identity_is_identity(self) -> None:
        np.testing.assert_allclose(mu.quat_inv(_q_identity()), _q_identity(), atol=1e-6)

    def test_q_times_q_inv_is_identity(self) -> None:
        q = mu.quat_from_angle_axis(0.7, np.array([1.0, 2.0, 3.0]) / np.linalg.norm([1.0, 2.0, 3.0]))
        composed = mu.quat_mul(q, mu.quat_inv(q))
        # Identity quaternion: w=1, xyz=0.
        np.testing.assert_allclose(composed[0], 1.0, atol=1e-5)
        np.testing.assert_allclose(composed[1:], 0.0, atol=1e-5)

    def test_zero_norm_returns_identity(self) -> None:
        out = mu.quat_inv(np.zeros(4))
        np.testing.assert_allclose(out, _q_identity(), atol=1e-6)


class TestQuatNormalize:
    def test_already_unit_unchanged(self) -> None:
        q = _q_identity()
        np.testing.assert_allclose(mu.quat_normalize(q), q)

    def test_scaled_unit_renormalized(self) -> None:
        q = np.array([2.0, 0.0, 0.0, 0.0])
        out = mu.quat_normalize(q)
        np.testing.assert_allclose(np.linalg.norm(out), 1.0, atol=1e-6)

    def test_zero_norm_returns_identity(self) -> None:
        out = mu.quat_normalize(np.zeros(4))
        np.testing.assert_allclose(out, _q_identity(), atol=1e-6)


class TestQuatFromAngleAxis:
    def test_zero_angle_is_identity(self) -> None:
        out = mu.quat_from_angle_axis(0.0, np.array([0.0, 0.0, 1.0]))
        np.testing.assert_allclose(out, _q_identity(), atol=1e-6)

    def test_unit_norm_for_unit_axis(self) -> None:
        out = mu.quat_from_angle_axis(1.234, np.array([0.0, 1.0, 0.0]))
        np.testing.assert_allclose(np.linalg.norm(out), 1.0, atol=1e-6)


class TestQuatScaledAngleAxisRoundtrip:
    @pytest.mark.parametrize(
        "angle,axis",
        [
            (0.5, np.array([1.0, 0.0, 0.0])),
            (1.7, np.array([0.0, 1.0, 0.0])),
            (2.3, np.array([0.0, 0.0, 1.0])),
            (0.9, np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)),
        ],
    )
    def test_quat_to_saa_to_quat(self, angle: float, axis: np.ndarray) -> None:
        q = mu.quat_from_angle_axis(angle, axis)
        saa = mu.quat_to_scaled_angle_axis(q)
        # Recover quaternion from scaled-angle-axis and check it rotates the
        # same x-axis to the same place.
        q_back = mu.quat_from_scaled_angle_axis(saa)
        ref = np.array([1.0, 0.5, -0.7])
        out_a = mu.quat_mul_vec3(q, ref)
        out_b = mu.quat_mul_vec3(q_back, ref)
        np.testing.assert_allclose(out_a, out_b, atol=1e-5)

    def test_identity_quat_maps_to_zero_saa(self) -> None:
        out = mu.quat_to_scaled_angle_axis(_q_identity())
        np.testing.assert_allclose(out, np.zeros(3), atol=1e-6)

    def test_zero_saa_maps_to_identity_quat(self) -> None:
        out = mu.quat_from_scaled_angle_axis(np.zeros(3))
        np.testing.assert_allclose(out, _q_identity(), atol=1e-6)

    def test_batch_quat_to_saa(self) -> None:
        q = np.stack([_q_identity(), mu.quat_from_angle_axis(0.5, np.array([0.0, 0.0, 1.0]))])
        out = mu.quat_to_scaled_angle_axis(q)
        assert out.shape == (2, 3)
        np.testing.assert_allclose(out[0], 0.0, atol=1e-6)


class TestClamp:
    def test_inside_range(self) -> None:
        assert mu.clamp(0.5, 0.0, 1.0) == 0.5

    def test_below_min(self) -> None:
        assert mu.clamp(-1.0, 0.0, 1.0) == 0.0

    def test_above_max(self) -> None:
        assert mu.clamp(2.0, 0.0, 1.0) == 1.0


class TestNormalize:
    def test_unit_vector_unchanged(self) -> None:
        v = np.array([1.0, 0.0, 0.0])
        np.testing.assert_allclose(mu.normalize(v), v)

    def test_zero_vector_returned_as_is(self) -> None:
        # The function returns the input unchanged when norm is below epsilon.
        v = np.zeros(3)
        np.testing.assert_allclose(mu.normalize(v), v)

    def test_nonunit_normalized(self) -> None:
        v = np.array([3.0, 4.0, 0.0])
        out = mu.normalize(v)
        np.testing.assert_allclose(np.linalg.norm(out), 1.0, atol=1e-6)


class TestLength:
    def test_length_zero(self) -> None:
        assert mu.length(np.zeros(3)) == 0.0

    def test_length_unit(self) -> None:
        assert mu.length(np.array([1.0, 0.0, 0.0])) == pytest.approx(1.0)

    def test_length_pythagorean(self) -> None:
        assert mu.length(np.array([3.0, 4.0, 0.0])) == pytest.approx(5.0)


class TestHalflifeToDamping:
    def test_positive_halflife_gives_positive_damping(self) -> None:
        assert mu.halflife_to_damping(0.5) > 0.0

    def test_smaller_halflife_means_higher_damping(self) -> None:
        # Shorter halflife -> faster decay -> larger damping coefficient.
        assert mu.halflife_to_damping(0.1) > mu.halflife_to_damping(1.0)


class TestFastNegexp:
    def test_x_zero_is_one(self) -> None:
        assert mu.fast_negexpf(0.0) == pytest.approx(1.0)

    def test_decreases_with_x(self) -> None:
        assert mu.fast_negexpf(0.0) > mu.fast_negexpf(0.5) > mu.fast_negexpf(2.0)

    def test_close_to_exp_for_small_x(self) -> None:
        for x in [0.0, 0.1, 0.5, 1.0]:
            assert mu.fast_negexpf(x) == pytest.approx(np.exp(-x), abs=0.05)
