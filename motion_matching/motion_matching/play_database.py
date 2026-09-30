#!/usr/bin/env python3
"""Motion Matching database visualizer using viser."""

import time
import argparse
import os
from pathlib import Path

import numpy as np
import viser
import trimesh

from motion_matching.core.features import forward_kinematics
from motion_matching.utils import db_path
from motion_matching.utils.math_utils import quat_to_scaled_angle_axis
from motion_matching.core.database import MotionMatchingDatabase
from motion_matching.viz import add_world_scene_with_g1
from motion_matching.metadata.constants import (
    G1_JOINT_AXES as joint_axes,
    G1_BONE_LEFT_FOOT as bone_left_foot,
    G1_BONE_RIGHT_FOOT as bone_right_foot,
)


# NOTE: Edit the terrain files to point to your terrains.
terrain_files = []


def _resolve_database_path(arg: str) -> Path:
    """Resolve a user-supplied --database argument to an absolute path.

    Accepts:
      - an absolute path
      - a path relative to cwd
      - a bare filename like 'database_g1_locomotion.bin' (resolved against
        PHP_MOTION_DATABASE_DIR)

    Raises FileNotFoundError with a helpful message if the file doesn't exist.
    """
    candidate = Path(arg)
    if not candidate.is_absolute():
        # Try as-given (cwd-relative) first; if that fails, try inside PHP_MOTION_DATABASE_DIR.
        if not candidate.exists():
            candidate = Path(db_path(arg))
    candidate = candidate.resolve()
    if not candidate.is_file():
        raise FileNotFoundError(
            f"Database file not found: {arg!r}. Tried both cwd-relative and "
            f"PHP_MOTION_DATABASE_DIR-relative resolution. Final candidate: {candidate}"
        )
    return candidate


def main() -> None:
    parser = argparse.ArgumentParser(description="Motion Matching visualizer for G1 robot")
    parser.add_argument(
        "--database",
        "-d",
        default="database_g1_locomotion.bin",
        help=(
            "Database file. Either an absolute path, a cwd-relative path, or "
            "a bare filename (resolved against PHP_MOTION_DATABASE_DIR). "
            "Default: database_g1_locomotion.bin"
        ),
    )
    args = parser.parse_args()

    database_path = _resolve_database_path(args.database)

    for debug_file in ("search_debug.txt", "bone_positions.txt"):
        if os.path.exists(debug_file):
            os.remove(debug_file)
            print(f"Deleted previous {debug_file}")

    db = MotionMatchingDatabase()
    db.load_from_file(str(database_path))

    server = viser.ViserServer(port=8081, verbose=False)

    with server.gui.add_folder("Controls"):
        auto_play = server.gui.add_checkbox("Auto Play", True)
        speed = server.gui.add_slider("Speed", min=0.1, max=3.0, step=0.1, initial_value=1.0)
        min_frame_slider = server.gui.add_slider("Min Frame", min=0, max=db.nframes() - 1, step=1, initial_value=0)
        max_frame_slider = server.gui.add_slider(
            "Max Frame", min=0, max=db.nframes() - 1, step=1, initial_value=db.nframes() - 1
        )
        frame_slider = server.gui.add_slider("Frame", min=0, max=db.nframes() - 1, step=1, initial_value=0)

        server.gui.add_markdown("---")
        server.gui.add_markdown("Camera Follow")
        camera_follow = server.gui.add_checkbox("Follow Character", False)
        camera_offset_x = server.gui.add_slider("Camera Offset X", min=-5.0, max=5.0, step=0.1, initial_value=-2)
        camera_offset_y = server.gui.add_slider("Camera Offset Y", min=-5.0, max=5.0, step=0.1, initial_value=1.0)
        camera_offset_z = server.gui.add_slider("Camera Offset Z", min=-5.0, max=5.0, step=0.1, initial_value=1.0)
        camera_smoothness = server.gui.add_slider("Camera Smoothness", min=0.01, max=0.5, step=0.01, initial_value=0.05)

        server.gui.add_markdown("---")
        contact_label_text = server.gui.add_text("Contact (L/R)", initial_value="L: -  R: -")
        show_contact_markers = server.gui.add_checkbox("Show Contact Markers", True)

    root_frame, viser_urdf = add_world_scene_with_g1(server)
    print("Loaded G1 URDF for visualization")

    fps = 60.0
    dt = 1.0 / fps
    current_frame = 0

    initial_dof = np.zeros(29, dtype=np.float32)
    viser_urdf.update_cfg(initial_dof)

    for idx, mesh_file in enumerate(terrain_files):
        scene_path = f"/terrain_mesh/{idx}"
        mesh_loaded = trimesh.load_mesh(str(mesh_file))
        mesh = trimesh.Trimesh(vertices=mesh_loaded.vertices, faces=mesh_loaded.faces)
        mesh.merge_vertices()
        server.scene.add_mesh_trimesh(scene_path, mesh=mesh)

    previous_time = time.time()
    while True:
        if auto_play.value:
            current_frame += dt * fps * speed.value

        range_min = int(min(min_frame_slider.value, max_frame_slider.value))
        range_max = int(max(min_frame_slider.value, max_frame_slider.value))

        if auto_play.value:
            if current_frame > range_max:
                current_frame = float(range_max)
            if current_frame < range_min:
                current_frame = float(range_min)
            frame_slider.value = int(current_frame)
        else:
            current_frame = float(frame_slider.value)
            if current_frame > range_max:
                current_frame = float(range_max)
                frame_slider.value = int(current_frame)
            if current_frame < range_min:
                current_frame = float(range_min)
                frame_slider.value = int(current_frame)

        next_frame = int(current_frame)

        final_bone_positions = db.bone_positions[next_frame]
        final_bone_rotations = db.bone_rotations[next_frame]

        try:
            dof_rotations = final_bone_rotations[-29:]
            assert dof_rotations.shape[0] == len(joint_axes)

            dof_aa = quat_to_scaled_angle_axis(dof_rotations)
            dof = np.sum(dof_aa * joint_axes, axis=-1)

            global_positions, global_rotations = forward_kinematics(
                final_bone_positions, final_bone_rotations, db.bone_parents, return_rotations=True
            )

            root_frame.position = global_positions[1]
            root_frame.wxyz = global_rotations[1]

            viser_urdf.update_cfg(dof)

            bone_lines = []
            colors = []
            for child_idx, parent_idx in enumerate(db.bone_parents):
                if parent_idx == 0:
                    continue
                if parent_idx != -1:
                    p = global_positions[parent_idx]
                    c = global_positions[child_idx]
                    if not (np.isfinite(p).all() and np.isfinite(c).all()):
                        continue
                    bone_lines.append([p, c])
                    colors.append([[100, 255, 100], [100, 255, 100]])

            server.scene.add_line_segments(
                "/skeleton", points=np.array(bone_lines), colors=np.array(colors), line_width=4.0
            )

            l_contact, r_contact = db.contact_states[next_frame]
            contact_label_text.value = f"L: {int(l_contact)}  R: {int(r_contact)}"
            BLUE = np.array([0, 100, 255])
            for side, in_contact, bone in (
                ("left", bool(l_contact), bone_left_foot),
                ("right", bool(r_contact), bone_right_foot),
            ):
                node = f"/contact_marker/{side}"
                if show_contact_markers.value and in_contact:
                    server.scene.add_icosphere(
                        node, position=global_positions[bone], radius=0.06, color=BLUE, subdivisions=2
                    )
                else:
                    try:
                        server.scene.remove_by_name(node)
                    except Exception:
                        pass

            root = global_positions[1] if len(global_positions) > 1 else np.array([0.0, 0.0, 0.0])

            if camera_follow.value:
                clients = server.get_clients()
                for client in clients.values():
                    character_pos = root
                    camera_offset = np.array([camera_offset_x.value, camera_offset_y.value, camera_offset_z.value])
                    camera_pos = character_pos + camera_offset

                    client.camera.look_at = character_pos + np.array([0.0, 1.0, 0.0])
                    client.camera.up_direction = np.array([0.0, 0.0, 1.0])

                    current_pos = client.camera.position
                    if np.isfinite(current_pos).all():
                        alpha = camera_smoothness.value
                        client.camera.position = alpha * camera_pos + (1 - alpha) * current_pos
                    else:
                        client.camera.position = camera_pos

        except Exception:
            pass

        current_time = time.time()
        time.sleep(max(0, 1.0 / fps - (current_time - previous_time)))
        previous_time = current_time


if __name__ == "__main__":
    main()
