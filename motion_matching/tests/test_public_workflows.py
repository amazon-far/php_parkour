"""Public CLI, generated-motion editing, and terrain-helper regressions."""

import json
import os
import subprocess
import sys

import numpy as np
import pytest
import trimesh
from motion_matching.io.motion_io import save_edited_motion_npz
from motion_matching.motion_visualizer import load_motion_data
from motion_matching.utils import PKG_ROOT, ROBOT_MOTIONS_DIR
from motion_matching.utils.tracking_format import qpos_to_tracking


def _qpos(n=60):
    qpos = np.zeros((n, 36), dtype=np.float32)
    qpos[:, 0] = 1
    qpos[:, 4] = np.linspace(0, 1, n)
    qpos[:, 6] = 0.8
    qpos[:, 7:] = np.linspace(0, 0.2, 29)
    return qpos


def test_viewer_round_trip_preserves_joint_order_and_commands(tmp_path):
    qpos = _qpos()
    commands = np.eye(14, dtype=np.float32)[np.arange(len(qpos)) % 14]
    source = tmp_path / "source_motion.npz"
    np.savez(source, **qpos_to_tracking(qpos, 30), vel_cmd=commands)
    _, _, loaded, fps, loaded_commands = load_motion_data(source)
    np.testing.assert_allclose(loaded, qpos, atol=1e-7)
    np.testing.assert_array_equal(loaded_commands, commands)
    output = tmp_path / "edited.npz"
    save_edited_motion_npz(loaded, output, source_fps=fps, target_fps=fps, vel_cmd=loaded_commands)
    with np.load(output) as saved:
        np.testing.assert_array_equal(saved["vel_cmd"], commands)
        np.testing.assert_allclose(saved["joint_pos"], qpos_to_tracking(qpos, 30)["joint_pos"], atol=1e-7)


def test_speed_and_fps_edits_preserve_discrete_command_timing(tmp_path):
    qpos = _qpos()
    commands = np.eye(14, dtype=np.float32)[np.arange(len(qpos)) // 5]
    output = tmp_path / "edited.npz"
    save_edited_motion_npz(qpos, output, source_fps=30, target_fps=50, speed=1.5, vel_cmd=commands)
    speed_indices = np.floor(np.linspace(0, 59, 40)).astype(int)
    fps_indices = np.floor(np.linspace(0, 39, 66)).astype(int)
    with np.load(output) as saved:
        assert saved["fps"].item() == 50
        assert saved["joint_pos"].shape[0] == 66
        assert saved["vel_cmd"].dtype == np.float32
        np.testing.assert_array_equal(saved["vel_cmd"], commands[speed_indices[fps_indices]])
        np.testing.assert_array_equal(saved["vel_cmd"].sum(axis=1), np.ones(66))


def test_editing_motion_without_commands_keeps_them_optional(tmp_path):
    output = tmp_path / "edited.npz"
    save_edited_motion_npz(_qpos(), output, source_fps=30, target_fps=30)
    with np.load(output) as saved:
        assert "vel_cmd" not in saved


def test_viewer_rejects_source_clip_with_actionable_error():
    with pytest.raises(ValueError, match="generated tracking motion"):
        load_motion_data(ROBOT_MOTIONS_DIR / "climb_76_high_speed_up.npz")


def test_viewer_rejects_misaligned_commands(tmp_path):
    source = tmp_path / "bad_motion.npz"
    np.savez(source, **qpos_to_tracking(_qpos(), 30), vel_cmd=np.zeros((2, 14)))
    with pytest.raises(ValueError, match="vel_cmd.*matching joint_pos"):
        load_motion_data(source)


def _run_python(args, cwd):
    env = {**os.environ, "PYTHONPATH": str(PKG_ROOT.parent)}
    return subprocess.run([sys.executable, *args], env=env, cwd=cwd, capture_output=True, text=True, check=False)


def test_headless_entry_points_do_not_import_gui_packages(tmp_path):
    result = _run_python(["-c", """
import importlib.abc
import sys
class NoGui(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'tkinter', '_tkinter', 'viser', 'matplotlib', 'robot_descriptions'}:
            raise AssertionError('Headless import requested GUI package: ' + fullname)
sys.meta_path.insert(0, NoGui())
import motion_matching.run
import motion_matching.generate_database
"""], tmp_path)
    assert result.returncode == 0, result.stderr


def test_viewer_requires_explicit_motion_file(tmp_path):
    result = _run_python(["-m", "motion_matching.motion_visualizer"], tmp_path)
    assert result.returncode != 0
    assert "--motion-file" in result.stdout + result.stderr
    assert "Viser server started" not in result.stdout


def test_swap_xy_helper_swaps_geometry_without_gui_imports(tmp_path):
    mesh = trimesh.creation.box(extents=[2.0, 1.0, 0.5])
    transform = trimesh.transformations.rotation_matrix(0.3, [0, 0, 1])
    transform[:3, 3] = [4.0, 2.0, 0.25]
    mesh.apply_transform(transform)
    source = tmp_path / "box.obj"
    mesh.export(source)
    result = _run_python([
        "-m", "motion_matching.utils.data_utils", "--obj_file", str(source), "--swap_xy"
    ], tmp_path)
    assert result.returncode == 0, result.stderr
    data = json.loads(source.with_name("box_xy_swapped.json").read_text())
    np.testing.assert_allclose(data["pos"], [2.0, 4.0, 0.25], atol=1e-7)
    rotation = trimesh.transformations.quaternion_matrix(data["quat"])[:3, :3]
    local_vertices = (mesh.vertices[:, [1, 0, 2]] - data["pos"]) @ rotation
    np.testing.assert_allclose(np.abs(local_vertices), np.broadcast_to(np.asarray(data["size"]) / 2, (8, 3)), atol=1e-7)
