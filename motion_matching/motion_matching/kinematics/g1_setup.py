"""Shared boilerplate for entry points that need a configured G1 :class:`URDFKinematics`.

Several entry points (``run.py``, ``play_database.py``) need to:

1. Construct a ``URDFKinematics`` from the packaged G1 URDF
2. Compute a ``(num_joints, 3)`` array of per-joint rotation axes
3. Print a sensible status line per joint

This module collects that boilerplate in a single function. Callers that don't
need joint-axes printing can pass ``verbose=False``.
"""

from __future__ import annotations

import numpy as np

from motion_matching.kinematics.urdf_kinematics import URDFKinematics
from motion_matching.utils import G1_URDF


def init_g1_kinematics(*, verbose: bool = False) -> tuple[URDFKinematics, np.ndarray]:
    """Construct a G1 :class:`URDFKinematics` and compute joint rotation axes.

    Args:
        verbose: If ``True`` (default), print one status line per joint listing
            its rotation axis. Set to ``False`` to silence the boilerplate when
            instantiating from a context where the prints are noise.

    Returns:
        ``(kin, joint_axes)``:

        - ``kin``: configured :class:`URDFKinematics` with the packaged G1 URDF.
            Meshes are NOT loaded (callers that need meshes should pass their
            own ``URDFKinematics`` instance with the appropriate flags).
        - ``joint_axes``: ``(kin.num_joints, 3)`` ``float32`` array. Rows for
            joints without an explicit ``<axis>`` element are zeros.
    """
    kin = URDFKinematics(urdf_path=G1_URDF, load_meshes=False, load_collision_meshes=False, verbose=verbose)

    if verbose:
        print("\nComputing joint rotation axes...")

    joint_axes = np.zeros((kin.num_joints, 3))
    for i, joint_name in enumerate(kin.joint_names):
        joint = kin.urdf.joint_map[joint_name]
        if hasattr(joint, "axis") and joint.axis is not None:
            joint_axes[i] = np.array(joint.axis)
            if verbose:
                ax = joint_axes[i]
                print(f"{i:2d}: {joint_name:30s} -> axis: [{ax[0]:6.3f}, {ax[1]:6.3f}, {ax[2]:6.3f}]")

    return kin, joint_axes
