"""Characterization tests for URDFKinematics global/joint pose computation.

Pins the exact FK output of ``compute_global_poses`` and ``compute_joint_poses``
for the G1 at a fixed root transform. These guard:

- the scipy read-only-array fix at ``urdf_kinematics.py`` (adding ``np.array``
  is a mathematical no-op — these hashes must not move), and
- Stage 3b, which extracts the shared xyzw<->wxyz reorder helper.

Requires the G1 URDF asset; skipped if absent.
"""

from __future__ import annotations

import numpy as np
import pytest

from motion_matching.utils import G1_URDF
from tests._hashing import hash_array

pytestmark = pytest.mark.skipif(
    not __import__("pathlib").Path(G1_URDF).is_file(), reason=f"G1 URDF not found at {G1_URDF}"
)

_ROOT_POS = np.array([0.1, 0.2, 0.5])
_ROOT_ORI = np.array([1.0, 0.0, 0.0, 0.0])


@pytest.fixture(scope="module")
def kin():
    from motion_matching.kinematics.g1_setup import init_g1_kinematics

    k, _axes = init_g1_kinematics()
    return k


def test_compute_global_poses_byte_identical(kin) -> None:
    pos, orient = kin.compute_global_poses(
        root_position=_ROOT_POS, root_orientation=_ROOT_ORI, joint_angles=np.zeros(kin.num_joints)
    )
    assert hash_array(np.asarray(pos)) == ("96313fde3781acceddb17c5c5ad751f1d2b676f2499689afb874f61477487fce")
    assert hash_array(np.asarray(orient)) == ("7cd918965b5861b6d9a2e94d95959704d1578434291e013c8112b023cca8019b")


def test_compute_joint_poses_byte_identical(kin) -> None:
    pos, orient = kin.compute_joint_poses(
        root_position=_ROOT_POS, root_orientation=_ROOT_ORI, joint_angles=np.zeros(kin.num_joints)
    )
    assert hash_array(np.asarray(pos)) == ("62d1cc032ca9b4361dc6da6f7c8f16ec41388ea76fdf5a82e210e04f82622859")
    assert hash_array(np.asarray(orient)) == ("f1f634a76f4835727384b2b88569e2bfaf0254aeaaf637cce658dbe31dc2be86")
