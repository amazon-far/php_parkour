#!/usr/bin/env python3
"""URDF Animation Script

Animate a URDF robot using motion data from .npz files.

Usage:
    python motion_visualizer.py --motion-file motion.npz --terrain-file terrain.obj
    python motion_visualizer.py --motion-file path/to/directory/

Supported motion file format (FAR-tracking schema, produced by run.py):
- .npz with keys: fps, joint_pos, body_pos_w, body_quat_w, joint_vel,
  body_lin_vel_w, body_ang_vel_w, optional vel_cmd.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, List

import numpy as np
import trimesh
import tyro
from scipy.spatial.transform import Rotation

import viser
from viser.extras import ViserUrdf

from motion_matching.io.motion_io import save_edited_motion_npz
from motion_matching.utils import G1_URDF
from motion_matching.scenarios.builtin import VEL_CMD_MAPPING


# Reverse of the joint reordering applied by qpos_to_tracking — restores the
# URDF's actuated-joint order so ViserUrdf.update_cfg gets the right inputs.
# Mapping copied from tracking_motion_animator.py:108-115.
_TRACKING_JOINT_MAPPING = [
    0,
    6,
    12,
    1,
    7,
    13,
    2,
    8,
    14,
    3,
    9,
    15,
    22,
    4,
    10,
    16,
    23,
    5,
    11,
    17,
    24,
    18,
    25,
    19,
    26,
    20,
    27,
    21,
    28,
]
_REVERSE_JOINT_MAPPING = np.zeros(len(_TRACKING_JOINT_MAPPING), dtype=np.int64)
for _orig, _mapped in enumerate(_TRACKING_JOINT_MAPPING):
    _REVERSE_JOINT_MAPPING[_mapped] = _orig


def reconstruct_box_mesh(
    center: np.ndarray, size: np.ndarray, quat_wxyz: np.ndarray
) -> trimesh.Trimesh:
    """Build a transformed box from center, extents, and wxyz quaternion."""
    box = trimesh.creation.box(extents=size)
    transform = trimesh.transformations.quaternion_matrix(quat_wxyz)
    transform[:3, 3] = center
    box.apply_transform(transform)
    return box


def load_motion_data(
    motion_file: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray | None]:
    """Load motion data in the FAR-tracking schema produced by run.py.

    The tracking schema stores joints in a custom policy order; the loader
    applies the reverse mapping so ``joint_data`` is in URDF-actuated order.
    A synthetic ``qpos_data`` (``[quat_wxyz, pos, joints]``) is reconstructed
    from ``body_pos_w[:, 0]`` / ``body_quat_w[:, 0]`` so the save handler can
    reuse the existing resampling utilities.

    Returns:
        joint_data: (T, 29) joints in URDF order
        root_data:  (T, 7)  [x, y, z, qw, qx, qy, qz]
        qpos_data:  (T, 36) [qw, qx, qy, qz, x, y, z, joints]
        fps:        Frames per second
        vel_cmd_data: optional (T, D)
    """
    if motion_file.suffix != ".npz":
        raise ValueError(f"Unsupported file format: {motion_file.suffix}. Only .npz is supported.")

    with np.load(motion_file, allow_pickle=False) as data:
        required = ("fps", "joint_pos", "body_pos_w", "body_quat_w")
        missing = [k for k in required if k not in data.files]
        if missing:
            raise ValueError(
                f"NPZ file {motion_file} missing required keys {missing}; "
                "provide a generated tracking motion from motion_matching.run"
            )

        fps_val = data["fps"]
        if fps_val.size != 1:
            raise ValueError("fps must contain one positive frame rate")
        fps = float(fps_val.item())
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("fps must be positive and finite")

        joints = data["joint_pos"]
        positions = data["body_pos_w"]
        quaternions = data["body_quat_w"]
        if joints.ndim != 2 or joints.shape[1] != 29 or len(joints) == 0:
            raise ValueError(f"joint_pos must have shape (T, 29) with T > 0, got {joints.shape}")
        num_timesteps = len(joints)
        if (positions.ndim != 3 or positions.shape[0] != num_timesteps
                or positions.shape[1] == 0 or positions.shape[2] != 3):
            raise ValueError("body_pos_w must have shape (T, B, 3) matching joint_pos, with B > 0")
        if quaternions.shape != (*positions.shape[:2], 4):
            raise ValueError("body_quat_w must have shape (T, B, 4) matching body_pos_w")
        if not all(np.isfinite(values).all() for values in (joints, positions, quaternions)):
            raise ValueError("Motion arrays must contain only finite values")
        if np.any(np.linalg.norm(quaternions[:, 0], axis=-1) == 0):
            raise ValueError("Root quaternions must be nonzero")

        joint_data = joints[:, _REVERSE_JOINT_MAPPING].astype(np.float32)
        root_pos = positions[:, 0, :].astype(np.float32)
        root_quat = quaternions[:, 0, :].astype(np.float32)
        root_data = np.concatenate([root_pos, root_quat], axis=1)
        qpos_data = np.concatenate([root_quat, root_pos, joint_data], axis=1).astype(np.float32)

        vel_cmd_data = data["vel_cmd"] if "vel_cmd" in data.files else None
        if vel_cmd_data is not None:
            if vel_cmd_data.ndim != 2 or len(vel_cmd_data) != num_timesteps:
                raise ValueError("vel_cmd must have shape (T, D) matching joint_pos")
            if not np.isfinite(vel_cmd_data).all():
                raise ValueError("vel_cmd must contain only finite values")

        print(f"Loaded NPZ motion data: {num_timesteps} frames @ {fps} FPS")
        print(f"Joints: {joint_data.shape[1]}  Bodies: {positions.shape[1]}")
        if vel_cmd_data is not None:
            print(f"Loaded vel_cmd data: {vel_cmd_data.shape}")

    return joint_data, root_data, qpos_data, fps, vel_cmd_data


def create_animation_controls(
    server: viser.ViserServer,
) -> tuple[viser.GuiInputHandle[bool], viser.GuiInputHandle[float], viser.GuiInputHandle[float], viser.GuiButtonHandle]:
    """Create animation control GUI elements."""

    with server.gui.add_folder("Animation Controls"):
        play_button = server.gui.add_checkbox("Play Animation", initial_value=False)

        speed_slider = server.gui.add_slider("Speed", min=0.1, max=5.0, step=0.1, initial_value=1.0)

        frame_slider = server.gui.add_slider(
            "Frame",
            min=0,
            max=100,  # Will be updated based on motion data
            step=1,
            initial_value=0,
        )

        save_button = server.gui.add_button("Save Motion")

    return play_button, speed_slider, frame_slider, save_button


def create_joint_value_sliders(
    server: viser.ViserServer,
    joint_names: tuple[str, ...],
    joint_limits: dict[str, tuple[float | None, float | None]],
    viser_urdf: ViserUrdf,
    play_button: viser.GuiInputHandle[bool] | None = None,
) -> tuple[list[viser.GuiSliderHandle], dict]:
    """Create interactive DOF slider elements."""

    joint_sliders = []

    # Shared state to prevent callback conflicts
    slider_state = {"updating_from_animation": False, "last_manual_update": 0.0}

    with server.gui.add_folder("Joint Values (DOF)"):
        for joint_name in joint_names:
            # Get joint limits
            lower, upper = joint_limits.get(joint_name, (None, None))
            lower = lower if lower is not None else -np.pi
            upper = upper if upper is not None else np.pi

            # Ensure initial value is within bounds
            initial_value = max(lower, min(upper, 0.0))

            # Create slider for each joint
            joint_slider = server.gui.add_slider(
                label=joint_name,
                min=lower,
                max=upper,
                step=0.001,
                initial_value=initial_value,
                marks=((lower, f"{lower:.2f}"), (0.0, "0"), (upper, f"{upper:.2f}"))
                if abs(upper - lower) > 0.1 and lower <= 0.0 <= upper
                else None,
            )

            joint_sliders.append(joint_slider)

    # Set up callbacks for manual control
    def update_urdf_from_sliders() -> None:
        # Only pause if this is a manual user interaction, not an animation update
        if not slider_state["updating_from_animation"]:
            joint_values = np.array([s.value for s in joint_sliders])
            viser_urdf.update_cfg(joint_values)
            slider_state["last_manual_update"] = time.time()
            # Pause animation when user manually adjusts sliders
            if play_button is not None:
                play_button.value = False

    for slider in joint_sliders:
        slider.on_update(lambda _: update_urdf_from_sliders())

    return joint_sliders, slider_state


def create_root_transform_controls(server: viser.ViserServer) -> tuple[list[viser.GuiSliderHandle], viser.FrameHandle]:
    """Create root transform control sliders and frame handle."""

    root_sliders = []

    with server.gui.add_folder("Root Transform"):
        # Translation controls
        x_slider = server.gui.add_slider("X Position", min=-5.0, max=5.0, step=0.01, initial_value=0.0)
        y_slider = server.gui.add_slider("Y Position", min=-5.0, max=5.0, step=0.01, initial_value=0.0)
        z_slider = server.gui.add_slider("Z Position", min=-2.0, max=3.0, step=0.01, initial_value=0.0)

        # Rotation controls (Euler angles for intuitive control)
        roll_slider = server.gui.add_slider("Roll", min=-np.pi, max=np.pi, step=0.01, initial_value=0.0)
        pitch_slider = server.gui.add_slider("Pitch", min=-np.pi, max=np.pi, step=0.01, initial_value=0.0)
        yaw_slider = server.gui.add_slider("Yaw", min=-np.pi, max=np.pi, step=0.01, initial_value=0.0)

        root_sliders = [x_slider, y_slider, z_slider, roll_slider, pitch_slider, yaw_slider]

    # Create root frame handle
    root_frame = server.scene.add_frame("/robot_root", show_axes=True)

    # Set up callbacks to update root frame
    def update_root_transform() -> None:
        x, y, z, roll, pitch, yaw = [s.value for s in root_sliders]
        root_frame.position = (x, y, z)

        # Convert Euler angles to quaternion
        r = Rotation.from_euler("xyz", [roll, pitch, yaw])
        root_frame.wxyz = r.as_quat()[[3, 0, 1, 2]]  # Convert to wxyz format

    for slider in root_sliders:
        slider.on_update(lambda _: update_root_transform())

    return root_sliders, root_frame


def update_root_transform_from_data(
    root_sliders: list[viser.GuiSliderHandle], root_frame: viser.FrameHandle, root_data: np.ndarray, frame_idx: int
) -> None:
    """Update root transform from motion data [x, y, z, qw, qx, qy, qz]."""
    if root_data is None or len(root_data) == 0:
        return

    # Get root transform for current frame
    root_transform = root_data[frame_idx]
    x, y, z = root_transform[:3]

    # Handle quaternion (qw, qx, qy, qz format)
    if len(root_transform) >= 7:
        qw, qx, qy, qz = root_transform[3:7]

        # Convert quaternion to Euler angles for slider display
        r = Rotation.from_quat([qx, qy, qz, qw])  # scipy uses xyzw format
        roll, pitch, yaw = r.as_euler("xyz")

        # Update sliders (without triggering callbacks to avoid conflicts)
        root_sliders[0].value = float(x)  # X
        root_sliders[1].value = float(y)  # Y
        root_sliders[2].value = float(z)  # Z
        root_sliders[3].value = float(roll)  # Roll
        root_sliders[4].value = float(pitch)  # Pitch
        root_sliders[5].value = float(yaw)  # Yaw

        # Update root frame directly
        root_frame.position = (x, y, z)
        root_frame.wxyz = (qw, qx, qy, qz)
    else:
        # Only translation data
        root_sliders[0].value = float(x)
        root_sliders[1].value = float(y)
        root_sliders[2].value = float(z)
        root_frame.position = (x, y, z)


def update_joint_sliders(
    joint_sliders: list[viser.GuiSliderHandle], joint_values: np.ndarray, slider_state: dict
) -> None:
    """Update the joint slider values (avoiding callback conflicts during animation)."""

    # Set flag to prevent callbacks from pausing animation
    slider_state["updating_from_animation"] = True

    for slider, value in zip(joint_sliders, joint_values):
        slider.value = float(value)

    # Reset flag
    slider_state["updating_from_animation"] = False


def main(
    motion_file: Path,
    output_fps: int | None = None,
    urdf_path: Path | None = None,
    loop: bool = True,
    load_meshes: bool = True,
    load_collision_meshes: bool = False,
    terrain_file: Path | None = None,
    show_vel_cmd: bool = False,
) -> None:
    """Animate a URDF with motion data.

    Args:
        motion_file: Required generated tracking motion (.npz) or directory of generated motions.
        urdf_path: Optional G1 URDF override (same 29-joint order as the packaged model)
        loop: Whether to loop the animation
        load_meshes: Load visual meshes
        load_collision_meshes: Load collision meshes
        terrain_file: Optional OBJ mesh to add to the scene as static terrain
        show_vel_cmd: If True, show velocity command visualization using matplotlib
    """
    if urdf_path is None:
        urdf_path = Path(G1_URDF)
    if not motion_file.exists():
        raise FileNotFoundError(f"Generated motion not found: {motion_file}")

    # Handle directory input
    motion_files: List[Path] = []
    is_directory_mode = motion_file.is_dir()
    if is_directory_mode:
        print(f"Scanning directory: {motion_file}")
        found = [path for path in motion_file.glob("*.npz") if not path.stem.endswith("_terrain")]
        motion_files.extend(found)
        motion_files.sort()
        if not motion_files:
            raise ValueError(f"No supported motion files found in directory: {motion_file}")
        print(f"Found {len(motion_files)} motion files.")
        current_motion_file = motion_files[0]
    else:
        motion_files = [motion_file]
        current_motion_file = motion_file

    # Start viser server
    server = viser.ViserServer()
    print("Viser server started. Open http://localhost:8080 in your browser.")

    print(f"Loading G1 URDF from {urdf_path}")

    # Create root transform controls
    root_sliders, root_frame = create_root_transform_controls(server)

    # Create URDF visualizer (attach to root frame)
    viser_urdf = ViserUrdf(
        server,
        urdf_or_path=urdf_path,
        root_node_name="/robot_root",  # Attach to root frame
        load_meshes=load_meshes,
        load_collision_meshes=load_collision_meshes,
        collision_mesh_color_override=(1.0, 0.0, 0.0, 0.5),
    )

    urdf_joint_names = viser_urdf.get_actuated_joint_names()
    urdf_num_joints = len(urdf_joint_names)

    # --- Animation State ---
    anim_state = {
        "motion_data": None,
        "root_data": None,
        "qpos_data": None,
        "fps": 30.0,
        "num_timesteps": 0,
        "num_joints": 0,
        "current_frame": 0.0,
        "current_file": current_motion_file,
    }

    # --- Terrain State ---
    terrain_state = {
        "mesh_handle": None,
        "format": None,
        "variant_count": 0,
        "active_variant": None,
        "npy_data": None,
    }

    def load_and_process_motion(file_path: Path) -> None:
        print(f"\nLoading: {file_path.name}")
        j_data, r_data, q_data, f_val, v_data = load_motion_data(file_path)

        n_time, n_joints = j_data.shape

        if n_joints != urdf_num_joints:
            raise ValueError(f"Motion has {n_joints} joints; G1 URDF has {urdf_num_joints} actuated joints")

        anim_state["motion_data"] = j_data
        anim_state["root_data"] = r_data
        anim_state["qpos_data"] = q_data
        anim_state["fps"] = f_val
        anim_state["num_timesteps"] = n_time
        anim_state["num_joints"] = n_joints
        anim_state["current_frame"] = 0.0
        anim_state["current_file"] = file_path
        anim_state["vel_cmd_data"] = v_data

        # Reset frame slider max
        frame_slider.max = max(0, n_time - 1)
        frame_slider.value = 0

        # Update info display
        info_file_handle.value = str(file_path.name)
        info_timesteps_handle.value = str(n_time)
        info_duration_handle.value = f"{n_time / f_val:.2f}s @ {f_val} FPS"
        info_vel_cmd_handle.value = "-"

        # Locate generated OBJ/NPY terrain paired with this motion.
        replaced_stem = file_path.stem.replace("_motion", "_terrain")
        appended_stem = file_path.stem + "_terrain"
        candidate_stems = [replaced_stem]
        if appended_stem not in candidate_stems:
            candidate_stems.append(appended_stem)
        terrain_stem = candidate_stems[0]
        for candidate in candidate_stems:
            if any((file_path.parent / f"{candidate}{ext}").exists() for ext in (".npy", ".obj")):
                terrain_stem = candidate
                break

        terrain_npy_file = file_path.parent / f"{terrain_stem}.npy"
        terrain_obj_file = file_path.parent / f"{terrain_stem}.obj"
        if terrain_npy_file.exists():
            load_terrain_npy(terrain_npy_file)
        elif terrain_obj_file.exists():
            load_terrain(terrain_obj_file)
            _sync_terrain_variant_controls(0)
        elif terrain_file is not None and terrain_file.exists():
            # Fallback to global terrain file if provided
            _sync_terrain_variant_controls(0)  # global terrain loaded once at startup
        else:
            # Clear terrain if not found
            clear_terrain()

    def load_terrain_npy(path: Path) -> None:
        try:
            print(f"Loading terrain mesh from {path}")
            terrain_data = np.load(path)  # Shape: n x m x 10
            # n: number of boxes
            # m: number of coherent terrain-scene variants. A single m index
            #    must be shared by all boxes.
            # 10: [pos(3), quat(4), size(3)]

            n_boxes, n_variants, n_dims = terrain_data.shape
            assert n_dims == 10, f"Expected 10 dimensions (pos+quat+size), got {n_dims}"

            print(f"Loaded terrain data: {n_boxes} boxes, {n_variants} scene variants")
            terrain_state.update(
                {
                    "format": "npy",
                    "variant_count": n_variants,
                    "active_variant": None,
                    "npy_data": terrain_data,
                }
            )
            if n_boxes == 0 or n_variants == 0:
                # Empty terrain is meaningful (for example, obstacle-free
                # random_locomotion scenes). Remove any mesh left by the
                # previously selected motion before returning.
                terrain_state["variant_count"] = 0
                _sync_terrain_variant_controls(0)
                _replace_terrain_mesh([], "empty NPY terrain")
                return
            _sync_terrain_variant_controls(n_variants)
            initial_index = int(np.random.randint(0, n_variants))
            terrain_variant_dropdown.value = str(initial_index)
            _apply_npy_terrain_variant(initial_index)
        except Exception as exc:
            print(f"Failed to load terrain mesh from {path}: {exc}")
            import traceback

            traceback.print_exc()

    def _replace_terrain_mesh(meshes: list[trimesh.Trimesh], description: str) -> None:
        if terrain_state["mesh_handle"] is not None:
            terrain_state["mesh_handle"].remove()
            terrain_state["mesh_handle"] = None
        if not meshes:
            print(f"No terrain geometry for {description}")
            return
        terrain_mesh = trimesh.util.concatenate(meshes)
        terrain_mesh.merge_vertices()
        terrain_state["mesh_handle"] = server.scene.add_mesh_trimesh(
            "/terrain_mesh",
            mesh=terrain_mesh,
            position=(0.0, 0.0, 0.0),
            wxyz=(1.0, 0.0, 0.0, 0.0),
        )
        print(f"Loaded {description} with {len(meshes)} mesh(es)")

    def _apply_npy_terrain_variant(variant_index: int) -> None:
        terrain_data = terrain_state["npy_data"]
        if terrain_data is None or terrain_data.shape[1] == 0:
            return
        variant_index = int(np.clip(variant_index, 0, terrain_data.shape[1] - 1))
        if terrain_state["format"] == "npy" and terrain_state["active_variant"] == variant_index:
            return
        meshes = [
            reconstruct_box_mesh(box[0:3], box[7:10], box[3:7])
            for box in terrain_data[:, variant_index, :]
        ]
        _replace_terrain_mesh(meshes, f"NPY terrain variant {variant_index}")
        terrain_state["active_variant"] = variant_index

    def load_terrain(path: Path) -> None:
        try:
            print(f"Loading terrain mesh from {path}")
            loaded_mesh = trimesh.load(path, force="mesh")
            terrain_mesh = trimesh.Trimesh(vertices=loaded_mesh.vertices, faces=loaded_mesh.faces)
            terrain_mesh.merge_vertices()

            # Remove previous terrain if exists
            if terrain_state["mesh_handle"] is not None:
                terrain_state["mesh_handle"].remove()

            terrain_state["mesh_handle"] = server.scene.add_mesh_trimesh(
                "/terrain_mesh", mesh=terrain_mesh, position=(0.0, 0.0, 0.0), wxyz=(1.0, 0.0, 0.0, 0.0)
            )
            terrain_state.update(
                {
                    "format": "obj",
                    "variant_count": 0,
                    "active_variant": None,
                    "npy_data": None,
                }
            )
            _sync_terrain_variant_controls(0)
        except Exception as exc:
            print(f"Failed to load terrain mesh {path}: {exc}")
            import traceback

            traceback.print_exc()

    def clear_terrain() -> None:
        if terrain_state["mesh_handle"] is not None:
            terrain_state["mesh_handle"].remove()
            terrain_state["mesh_handle"] = None
        terrain_state.update(
            {
                "format": None,
                "variant_count": 0,
                "active_variant": None,
                "npy_data": None,
            }
        )
        _sync_terrain_variant_controls(0)

    # --- GUI Setup ---

    # File Selector (if directory)
    autoplay_checkbox = None
    file_dropdown = None
    delete_button = None
    if is_directory_mode:
        with server.gui.add_folder("File Selection"):
            if len(motion_files) > 1:
                file_options = [p.name for p in motion_files]
                file_dropdown = server.gui.add_dropdown(
                    "Motion File", options=file_options, initial_value=current_motion_file.name
                )

                @file_dropdown.on_update
                def _(_: Any) -> None:
                    selected_name = file_dropdown.value
                    # Find full path
                    for p in motion_files:
                        if p.name == selected_name:
                            load_and_process_motion(p)
                            break
            else:
                # Show current file name as text when only one file
                server.gui.add_text("Current File", current_motion_file.name)

            autoplay_checkbox = server.gui.add_checkbox("Autoplay Sequence", initial_value=False)

            delete_button = server.gui.add_button("Delete Current File")

            @delete_button.on_click
            def _(_: Any) -> None:
                """Delete the current motion file and associated terrain files."""
                current_file = anim_state["current_file"]
                if current_file is None or not current_file.exists():
                    print(f"Cannot delete: file {current_file} does not exist")
                    return

                try:
                    # Delete the motion file
                    print(f"Deleting motion file: {current_file}")
                    current_file.unlink()

                    # Delete associated terrain files
                    # Try terrain.obj pattern: {stem}_terrain.obj
                    terrain_obj = current_file.parent / f"{current_file.stem}_terrain.obj"
                    if terrain_obj.exists():
                        print(f"Deleting terrain file: {terrain_obj}")
                        terrain_obj.unlink()

                    # Try terrain.npy pattern: {stem}.npy (with _motion -> _terrain replacement)
                    terrain_npy_pattern1 = current_file.parent / f"{current_file.stem}.npy".replace(
                        "_motion", "_terrain"
                    )
                    if terrain_npy_pattern1.exists():
                        print(f"Deleting terrain file: {terrain_npy_pattern1}")
                        terrain_npy_pattern1.unlink()

                    # Also try direct {stem}_terrain.npy pattern
                    terrain_npy_pattern2 = current_file.parent / f"{current_file.stem}_terrain.npy"
                    if terrain_npy_pattern2.exists():
                        print(f"Deleting terrain file: {terrain_npy_pattern2}")
                        terrain_npy_pattern2.unlink()

                    # Remove from motion_files list
                    motion_files.remove(current_file)

                    # Clear terrain from scene
                    clear_terrain()

                    # Update dropdown options or UI
                    if len(motion_files) > 0:
                        if file_dropdown is not None:
                            file_options = [p.name for p in motion_files]
                            file_dropdown.options = file_options

                            # Load next file (or first file if we deleted the last one)
                            next_file = motion_files[0] if motion_files else None
                            if next_file:
                                file_dropdown.value = next_file.name
                                load_and_process_motion(next_file)
                        else:
                            # Only one file left, reload it
                            next_file = motion_files[0]
                            load_and_process_motion(next_file)
                    else:
                        print("All motion files deleted")
                        if file_dropdown is not None:
                            file_dropdown.options = []
                        anim_state["motion_data"] = None
                        anim_state["root_data"] = None
                        anim_state["qpos_data"] = None
                        anim_state["num_timesteps"] = 0
                        frame_slider.max = 0
                        frame_slider.value = 0
                        info_file_handle.value = "No file loaded"
                        info_timesteps_handle.value = "-"
                        info_duration_handle.value = "-"

                    print(f"Successfully deleted {current_file.name}")

                except Exception as e:
                    print(f"Error deleting file {current_file}: {e}")
                    import traceback

                    traceback.print_exc()

    # Animation Controls
    play_button, speed_slider, frame_slider, save_button = create_animation_controls(server)

    # Info Display
    with server.gui.add_folder("Motion Info"):
        info_file_handle = server.gui.add_text("File", "Loading...")
        info_timesteps_handle = server.gui.add_text("Timesteps", "-")
        server.gui.add_text("Joints", str(urdf_num_joints))
        info_duration_handle = server.gui.add_text("Duration", "-")
        info_vel_cmd_handle = server.gui.add_text("Velocity Command", "-")

    # Joint Sliders
    joint_limits = viser_urdf.get_actuated_joint_limits()
    joint_sliders, slider_state = create_joint_value_sliders(
        server=server,
        joint_names=urdf_joint_names,
        joint_limits=joint_limits,
        viser_urdf=viser_urdf,
        play_button=play_button,
    )

    # Visibility Controls
    with server.gui.add_folder("Visibility"):
        show_meshes_cb = server.gui.add_checkbox("Show meshes", viser_urdf.show_visual)
        show_collision_meshes_cb = server.gui.add_checkbox("Show collision meshes", viser_urdf.show_collision)

    @show_meshes_cb.on_update
    def _(_: Any) -> None:
        viser_urdf.show_visual = show_meshes_cb.value

    @show_collision_meshes_cb.on_update
    def _(_: Any) -> None:
        viser_urdf.show_collision = show_collision_meshes_cb.value

    show_meshes_cb.visible = load_meshes
    show_collision_meshes_cb.visible = load_collision_meshes

    # Terrain variant controls, ported from tracking_motion_animator.py.
    with server.gui.add_folder("Terrain Controls"):
        terrain_variant_dropdown = server.gui.add_dropdown(
            "Variant",
            options=["0"],
            initial_value="0",
        )
        terrain_variant_dropdown.visible = False
        rerandomize_terrain_button = server.gui.add_button("Re-randomize Terrain")
        rerandomize_terrain_button.visible = False

    def _sync_terrain_variant_controls(num_variants: int) -> None:
        options = [str(index) for index in range(max(1, num_variants))]
        terrain_variant_dropdown.options = options
        terrain_variant_dropdown.value = options[0]
        terrain_variant_dropdown.visible = num_variants > 0
        rerandomize_terrain_button.visible = num_variants > 1

    def _apply_active_terrain_variant(variant_index: int) -> None:
        if terrain_state["format"] == "npy":
            _apply_npy_terrain_variant(variant_index)

    @terrain_variant_dropdown.on_update
    def _(_: Any) -> None:
        try:
            variant_index = int(terrain_variant_dropdown.value)
        except ValueError:
            return
        _apply_active_terrain_variant(variant_index)

    @rerandomize_terrain_button.on_click
    def _(_: Any) -> None:
        num_variants = int(terrain_state["variant_count"])
        if num_variants <= 0:
            return
        if num_variants == 1:
            variant_index = 0
        else:
            current_index = int(terrain_variant_dropdown.value)
            variant_index = (current_index + int(np.random.randint(1, num_variants))) % num_variants
        terrain_variant_dropdown.value = str(variant_index)
        _apply_active_terrain_variant(variant_index)

    # Grid
    server.scene.add_grid("/grid", width=200, height=200)

    # --- Initial Load ---

    # Load global terrain first if provided
    if terrain_file is not None and terrain_file.exists():
        load_terrain(terrain_file)

    load_and_process_motion(current_motion_file)

    # --- Save Handler ---
    @save_button.on_click
    def _(_: Any) -> None:
        if anim_state["qpos_data"] is None:
            print("Cannot save: no motion data loaded")
            return

        speed_value = float(speed_slider.value)
        speed_suffix = f"{speed_value:.2f}".rstrip("0").rstrip(".").replace(".", "p")
        if not speed_suffix:
            speed_suffix = "1"

        curr_file = anim_state["current_file"]
        output_filename = f"{curr_file.stem}_speed_{speed_suffix}.npz"
        output_path = curr_file.parent / output_filename

        saved_fps = output_fps if output_fps is not None else anim_state["fps"]
        save_edited_motion_npz(
            anim_state["qpos_data"],
            output_path,
            source_fps=anim_state["fps"],
            target_fps=saved_fps,
            speed=speed_value,
            vel_cmd=anim_state["vel_cmd_data"],
        )
        print(f"Motion saved to {output_path} (speed={speed_value:.2f}, fps={saved_fps})")

    # --- Animation Loop ---
    print("Starting animation loop...")

    # Setup matplotlib for vel_cmd visualization
    fig_vel, ax_vel = None, None
    vel_bars = None
    if show_vel_cmd:
        import matplotlib.pyplot as plt

        plt.ion()
        fig_vel, ax_vel = plt.subplots(figsize=(10, 4))
        ax_vel.set_title("Velocity Command (One-Hot)")
        ax_vel.set_xlabel("Command Index")
        ax_vel.set_ylabel("Activation")
        ax_vel.set_ylim(0, 1.1)

    frame_time = 1.0 / 30.0  # Initial safe default
    next_time = time.perf_counter() + frame_time

    while True:
        # Update fps based on current loaded motion
        current_fps = anim_state["fps"]
        if current_fps <= 0:
            current_fps = 30.0
        frame_time = 1.0 / current_fps

        num_timesteps = anim_state["num_timesteps"]

        if num_timesteps > 0:
            if play_button.value:
                frame_increment = speed_slider.value
                anim_state["current_frame"] += frame_increment

                if anim_state["current_frame"] >= num_timesteps:
                    # Handle autoplay sequence
                    should_advance = False
                    if autoplay_checkbox is not None and autoplay_checkbox.value:
                        try:
                            # Find current file index
                            current_idx = -1
                            for i, f in enumerate(motion_files):
                                if f.name == anim_state["current_file"].name:
                                    current_idx = i
                                    break

                            if current_idx != -1:
                                next_idx = current_idx + 1

                                if next_idx < len(motion_files):
                                    # Advance to next file
                                    next_file = motion_files[next_idx]
                                    if file_dropdown is not None:
                                        file_dropdown.value = next_file.name
                                    load_and_process_motion(next_file)
                                    should_advance = True
                                elif loop:
                                    # Loop back to start of playlist
                                    next_file = motion_files[0]
                                    if file_dropdown is not None:
                                        file_dropdown.value = next_file.name
                                    load_and_process_motion(next_file)
                                    should_advance = True
                        except Exception as e:
                            print(f"Error in autoplay: {e}")

                    if should_advance:
                        # Frame reset is handled by load_and_process_motion
                        pass
                    elif loop:
                        anim_state["current_frame"] %= num_timesteps
                    else:
                        anim_state["current_frame"] = num_timesteps - 1
                        play_button.value = False

                frame_slider.value = int(anim_state["current_frame"])
            else:
                anim_state["current_frame"] = float(frame_slider.value)

            frame_idx = int(anim_state["current_frame"]) % num_timesteps

            if anim_state.get("vel_cmd_data") is not None:
                vel_cmd = anim_state["vel_cmd_data"][frame_idx]
                cmd_idx = int(np.argmax(vel_cmd))
                cmd_str = VEL_CMD_MAPPING.get(cmd_idx, f"Undefined ({cmd_idx})")
                info_vel_cmd_handle.value = cmd_str
                # print(f"Frame {frame_idx} vel_cmd: {cmd_str} (idx: {cmd_idx})")

                # Update matplotlib plot
                if show_vel_cmd and fig_vel is not None and plt.fignum_exists(fig_vel.number):
                    dim = len(vel_cmd)
                    if vel_bars is None or len(vel_bars) != dim:
                        ax_vel.clear()
                        ax_vel.set_title("Velocity Command (One-Hot)")
                        ax_vel.set_xlabel("Command Index")
                        ax_vel.set_ylabel("Activation")
                        ax_vel.set_ylim(0, 1.1)

                        labels = []
                        for i in range(dim):
                            if i in VEL_CMD_MAPPING:
                                labels.append(f"{i}:{VEL_CMD_MAPPING[i]}")
                            else:
                                labels.append(str(i))

                        vel_bars = ax_vel.bar(range(dim), vel_cmd, color=plt.cm.tab20(range(dim)))
                        ax_vel.set_xticks(range(dim))
                        ax_vel.set_xticklabels(labels, rotation=45, ha="right")
                        plt.tight_layout()
                    else:
                        for bar, h in zip(vel_bars, vel_cmd):
                            bar.set_height(h)

                    fig_vel.canvas.draw_idle()
                    fig_vel.canvas.flush_events()
            else:
                info_vel_cmd_handle.value = "No Data"

            # Update pose
            with server.atomic():
                if anim_state["motion_data"] is not None:
                    current_joint_values = anim_state["motion_data"][frame_idx]
                    viser_urdf.update_cfg(current_joint_values)
                    update_joint_sliders(
                        joint_sliders=joint_sliders, joint_values=current_joint_values, slider_state=slider_state
                    )

                if anim_state["root_data"] is not None:
                    update_root_transform_from_data(
                        root_sliders=root_sliders,
                        root_frame=root_frame,
                        root_data=anim_state["root_data"],
                        frame_idx=frame_idx,
                    )
            server.flush()

        # Sleep
        now = time.perf_counter()
        time_to_sleep = next_time - now
        if time_to_sleep > 0:
            time.sleep(time_to_sleep)

        # drift correction
        if now > next_time + frame_time:
            next_time = now + frame_time
        else:
            next_time += frame_time


if __name__ == "__main__":
    tyro.cli(main)
