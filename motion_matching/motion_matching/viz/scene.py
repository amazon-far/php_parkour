"""Shared Viser scene setup for the interactive viewers.

``run.py`` (interactive motion matching) and ``play_database.py`` (database
playback) built the same world frame + grid + robot-root frame + G1 ``ViserUrdf``
by hand. These helpers hold that boilerplate in one place. They only affect the
interactive Viser scene; they produce no saved-file output.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Tuple

import numpy as np
import viser
from viser.extras import ViserUrdf

from motion_matching.utils import G1_URDF


def add_world_scene(server: viser.ViserServer) -> Any:
    """Add the world origin frame, an XY grid, and the robot-root frame.

    Returns the ``/robot_root`` frame handle (callers attach the robot to it and
    drive its pose).
    """
    server.scene.add_frame(
        "/world",
        position=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )
    server.scene.add_grid(
        "/xy_grid", width=100.0, height=100.0, position=np.array([0.0, 0.0, 0.0], dtype=np.float32), plane="xy"
    )
    return server.scene.add_frame("/robot_root", show_axes=True)


def load_g1_viser_urdf(server: viser.ViserServer) -> ViserUrdf:
    """Attach the packaged G1 URDF to ``/robot_root`` for visualization."""
    return ViserUrdf(
        server, urdf_or_path=Path(G1_URDF), root_node_name="/robot_root", load_meshes=True, load_collision_meshes=False
    )


def add_world_scene_with_g1(server: viser.ViserServer) -> Tuple[Any, ViserUrdf]:
    """Convenience: :func:`add_world_scene` + :func:`load_g1_viser_urdf`.

    Returns ``(root_frame, viser_urdf)``.
    """
    root_frame = add_world_scene(server)
    viser_urdf = load_g1_viser_urdf(server)
    return root_frame, viser_urdf
