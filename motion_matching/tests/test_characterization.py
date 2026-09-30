"""Characterization tests — pin the *exact* current outputs of the pure
functions that later refactor stages touch (dedup, rename, param-bundling).

These assert byte-identical content hashes (not ``allclose``) against goldens
captured from the pre-refactor code on 2026-06-01 (commit 0424c06c, mm_hs env).
Each test uses its own independently seeded inputs, so they are self-contained
and order-independent.

If a refactor stage is truly structural, these stay green. A red here means a
numeric change slipped in and must be reverted (or, if deliberate, the golden
updated with an explicit note in the commit).
"""

from __future__ import annotations

import numpy as np

from motion_matching.core import database as DB
from motion_matching.core import features as F
from motion_matching.utils import data_utils as D
from motion_matching.utils.tracking_format import qpos_to_tracking

from tests._hashing import hash_array


def _qpos_like(n: int, seed: int) -> np.ndarray:
    """A fixed ``(n, 36)`` motion-matching qpos array with a valid root quat."""
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((n, 36))
    data[:, :4] /= np.linalg.norm(data[:, :4], axis=1, keepdims=True)
    return data


def _skeleton(seed: int):
    """A small fixed 5-bone skeleton for FK tests."""
    rng = np.random.default_rng(seed)
    parents = np.array([-1, 0, 1, 0, 3])
    bp = rng.standard_normal((5, 3))
    br = rng.standard_normal((5, 4))
    br /= np.linalg.norm(br, axis=1, keepdims=True)
    bv = rng.standard_normal((5, 3))
    bav = rng.standard_normal((5, 3))
    return parents, bp, br, bv, bav


# --------------------------------------------------------------------------- #
# data_utils: resample / speed-adjust (Stage 3a dedups these)
# --------------------------------------------------------------------------- #
class TestDataUtilsResample:
    def test_resample_data(self) -> None:
        assert hash_array(D.resample_data(_qpos_like(40, 1), 50, 60)) == (
            "0be401a93cd93c3f7855664d038517efec5a212925f5523a0061e5e51e780b04"
        )

    def test_adjust_data_speed(self) -> None:
        assert hash_array(D.adjust_data_speed(_qpos_like(40, 1), 1.3)) == (
            "9a54bb95b0a6b1869cae1128b82a3414967eafac4241f9c769a50aaab1d5b538"
        )

    def test_resample_vel_cmd(self) -> None:
        vel_cmd = np.random.default_rng(2).standard_normal((40, 14))
        assert hash_array(D.resample_vel_cmd(vel_cmd, 50, 60)) == (
            "60b23632a1cf862bd277f970d62a1177ff801e9779d2d074dbc600e384a284c1"
        )

    def test_adjust_vel_cmd_speed(self) -> None:
        vel_cmd = np.random.default_rng(2).standard_normal((40, 14))
        assert hash_array(D.adjust_vel_cmd_speed(vel_cmd, 1.3)) == (
            "ff5c11a5e61c2d7dfa9b8494dcca7f5368a2d9d595a5050a4a18a7fc6df9b90d"
        )


# --------------------------------------------------------------------------- #
# features: forward kinematics + the negate-w quat_inv (Stage 3c renames it)
# --------------------------------------------------------------------------- #
class TestForwardKinematics:
    def test_forward_kinematics_positions(self) -> None:
        parents, bp, br, _, _ = _skeleton(3)
        assert hash_array(F.forward_kinematics(bp, br, parents)) == (
            "f30fbe3cf06062dbc7f35f040c47b40bbac95a744996193825ad0eb68aa9d07f"
        )

    def test_forward_kinematics_rotations(self) -> None:
        parents, bp, br, _, _ = _skeleton(3)
        _, gr = F.forward_kinematics(bp, br, parents, return_rotations=True)
        assert hash_array(gr) == ("8f14098fd57b21f96c87c84e53256782d0dd7b21eba01864f04ef2636a8b2bf2")

    def test_forward_kinematics_velocity(self) -> None:
        parents, bp, br, bv, bav = _skeleton(3)
        gp, gv, gr, gav = F.forward_kinematics_velocity(bp, bv, br, bav, parents)
        assert hash_array(gp) == ("f30fbe3cf06062dbc7f35f040c47b40bbac95a744996193825ad0eb68aa9d07f")
        assert hash_array(gv) == ("ba58185c9c35213be515c496821ee2717292b6d58179a36185489c4a8e15333d")
        assert hash_array(gr) == ("8f14098fd57b21f96c87c84e53256782d0dd7b21eba01864f04ef2636a8b2bf2")
        assert hash_array(gav) == ("b155af9ae085ba196fa0aefa7aad9485d9010b717bd5620088b9a3d1fa4eb18a")

    def test_quat_inv_negate_w_semantics(self) -> None:
        # features.quat_inv_negate_w negates only the w-component (NOT a true
        # conjugate). The math must stay identical to the old quat_inv.
        _, _, br, _, _ = _skeleton(3)
        assert hash_array(F.quat_inv_negate_w(br[0])) == (
            "50edc0d70b7172e30377956fc30356ef96b18bf750aac3d9dc89b8c24a1d2104"
        )


# --------------------------------------------------------------------------- #
# database search (Stage 1a removes the unused bound_* params)
# --------------------------------------------------------------------------- #
class TestMotionMatchingSearch:
    def test_search_fixed_picks_min_cost(self) -> None:
        rng = np.random.default_rng(4)
        nframes, ndim = 60, 27
        feats = rng.standard_normal((nframes, ndim))
        foff = rng.standard_normal(ndim)
        fscale = np.abs(rng.standard_normal(ndim)) + 0.5
        qn = rng.standard_normal(ndim)
        bi, bc = DB.motion_matching_search_fixed(
            best_index=-1,
            best_cost=np.inf,
            range_starts=np.array([0, 30]),
            range_stops=np.array([30, 60]),
            features=feats,
            features_offset=foff,
            features_scale=fscale,
            query_normalized=qn,
            transition_cost=0.1,
            ignore_range_end=5,
            ignore_surrounding=5,
        )
        assert int(bi) == 39
        assert float(bc) == 27.409416744247846


# --------------------------------------------------------------------------- #
# tracking_format.qpos_to_tracking (Stage 5a wraps it in save_motion_npz)
# --------------------------------------------------------------------------- #
class TestQposToTracking:
    GOLDENS = {
        "body_ang_vel_w": "c4c65ec5d49aba47ada54c5445c6190fa02232dbe2c457bda45a5197b96f3f49",
        "body_lin_vel_w": "773acf2927ac25158c9572b315915494f54839eb8eb75b1efa48020bc3ff200a",
        "body_pos_w": "cd041dc693fa96b8e89838bbd5c8d59fe858790d7e45c068fd6304f9277092a9",
        "body_quat_w": "1ad4c506c74bccb351ce58f1ed8440a1e474b155aefb2a9504f9c798d358c273",
        "fps": "dd25250d22f40cf884a43b51333f9d1be3dd5e4a6fb3a6a458bb1281f5474987",
        "joint_pos": "3e99f0245182cd8b200646e03eeb39b97440fd4a91c2ad93dc73ac1376124839",
        "joint_vel": "3881019d5778b02e0e5fcd5fe0c193782c3299fb521bd7d9b513d046ab9ff8a6",
    }

    def test_outputs_byte_identical(self) -> None:
        qp = _qpos_like(12, 5).astype(np.float32)
        tracking = qpos_to_tracking(qp, 50)
        assert set(tracking) == set(self.GOLDENS), "output keys changed"
        for key, golden in self.GOLDENS.items():
            assert hash_array(tracking[key]) == golden, key


# --------------------------------------------------------------------------- #
# randomize_box_x_numpy seeded (Stage 5b threads rng through)
# --------------------------------------------------------------------------- #
class TestRandomizeBox:
    def test_seeded_output_is_byte_identical(self) -> None:
        params = {"pos": [1.0, 2.0, 0.5], "quat": [1.0, 0.0, 0.0, 0.0], "size": [0.4, 0.6, 0.3]}
        out = D.randomize_box_x_numpy(params, num_variants=10, rng=np.random.default_rng(0))
        assert hash_array(out) == ("e87ff7c5365a8a3a2ec550ed333bcefd7b147aa1c760d3d8c3e0a719a7c7701b")
