"""Public uploader contracts, exercised without W&B installation or network."""

import io
import os
import shlex
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from motion_matching import upload_dataset as uploader


def motion_arrays(*, native=False, frames=4):
    bodies = 32 if native else 30
    arrays = {
        "fps": np.array([50]),
        "joint_pos": np.arange(
            frames * (36 if native else 29), dtype=np.float32
        ).reshape(frames, -1),
        "joint_vel": np.zeros((frames, 35 if native else 29), dtype=np.float32),
        "body_pos_w": np.zeros((frames, bodies, 3), dtype=np.float32),
        "body_quat_w": np.tile(np.array([1.0, 0, 0, 0]), (frames, bodies, 1)),
        "body_lin_vel_w": np.zeros((frames, bodies, 3)),
        "body_ang_vel_w": np.zeros((frames, bodies, 3)),
        "vel_cmd": np.eye(15, dtype=np.float32)[np.arange(frames) % 15],
    }
    if native:
        arrays["joint_names"] = np.array([f"joint_{i}" for i in range(29)])
        arrays["body_names"] = np.array([f"body_{i}" for i in range(32)])
    return arrays


def write_clip(root, name="sample_motion.npz", *, native=False, paired=True):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **motion_arrays(native=native))
    if paired:
        terrain = path.with_name("terrain".join(path.stem.rsplit("motion", 1)) + ".npy")
        boxes = np.zeros((2, 10, 10), dtype=np.float32)
        boxes[:, :, 3] = 1
        boxes[:, :, 7:] = 0.5
        np.save(terrain, boxes)
        terrain.with_suffix(".obj").write_text("# paired preview\nv 0 0 0\n")
    return path


def write_interactive(root, name="motion_20260912_101010", *, flat=False):
    motion = write_clip(root, name + ".npz", paired=False)
    sidecar = motion.with_suffix("")
    sidecar.mkdir()
    if not flat:
        np.save(sidecar / "terrain.npy", np.zeros((2, 10, 10)))
    (sidecar / "multi_boxes_scaled.obj").write_text("# interactive preview\nv 1 2 3\n")
    return motion, sidecar


def snapshot(root):
    return {
        path.relative_to(root).as_posix(): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in root.rglob("*")
        if path.is_file()
    }


class FakeWandb:
    def __init__(self, *, configured_project=None, fail=None, linked_name=None):
        self.settings = SimpleNamespace(project=configured_project)
        self.fail = fail
        self.linked_name = linked_name
        self.calls = []
        self.contents = {}
        self.paths = {}
        self.artifact = None
        self.logged = SimpleNamespace(
            wait=self.wait, qualified_name="source/project/clips:v99"
        )

    def event(self, action):
        self.calls.append(action)
        if action == self.fail:
            raise RuntimeError(f"{action} failure")

    def setup(self):
        self.event("setup")
        return self

    def init(self, **kwargs):
        self.event("init")
        self.init_options = kwargs
        return self

    def Artifact(self, **kwargs):
        self.event("artifact")
        self.artifact_options = kwargs
        self.artifact = SimpleNamespace(add_file=self.add_file)
        return self.artifact

    def add_file(self, path, name):
        self.event("add_file")
        assert name not in self.contents
        self.contents[name] = Path(path).read_bytes()
        self.paths[name] = Path(path)

    def log_artifact(self, artifact):
        self.event("log")
        assert artifact is self.artifact
        return self.logged

    def wait(self):
        self.event("wait")
        # Staged payloads must remain available through upload completion.
        assert all(path.is_file() for path in self.paths.values())
        return self.logged

    def link_artifact(self, *, artifact, target_path):
        self.event("link")
        assert artifact is self.logged
        assert self.calls.index("wait") < self.calls.index("link")
        self.target_path = target_path
        qualified = (
            self.linked_name
            or "resolved-org/" + "/".join(target_path.split("/")[-2:]) + ":v7"
        )
        return SimpleNamespace(qualified_name=qualified)

    def finish(self, *, exit_code):
        self.exit_code = exit_code
        self.event("finish")


@pytest.fixture
def fake_wandb(monkeypatch):
    fake = FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake)
    return fake


@pytest.fixture
def no_sdk(monkeypatch):
    real_import = uploader.importlib.import_module

    def import_module(name, *args, **kwargs):
        if name == "wandb":
            pytest.fail("W&B imported before validation or during a dry run")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(uploader.importlib, "import_module", import_module)


def args(root, *extra):
    return ["--registry_name", "clips", "--file_folder", str(root), *extra]


def test_recursive_upload_is_sorted_preserves_bytes_and_only_selects_payloads(
    tmp_path, fake_wandb
):
    later = write_clip(tmp_path, "z/nested/repeated_motion.npz", native=True)
    earlier = write_clip(tmp_path, "a/repeated_motion.npz")
    # Deliberately contradict lexical order, unlike the old mtime sorting.
    os.utime(later, (1, 1))
    os.utime(earlier, (999, 999))
    (tmp_path / "motion_notes.json").write_text("private notes")
    (tmp_path / "terrain.png").write_bytes(b"preview")
    (tmp_path / "arbitrary.obj").write_bytes(b"unpaired")
    np.savez(tmp_path / "source.npz", qpos=np.zeros((4, 36)))
    before = snapshot(tmp_path)
    assert uploader.main(args(tmp_path)) == 0
    names = list(fake_wandb.contents)
    assert names == sorted(names)
    assert names == [
        "a/repeated_motion.npz",
        "a/repeated_terrain.npy",
        "a/repeated_terrain.obj",
        "z/nested/repeated_motion.npz",
        "z/nested/repeated_terrain.npy",
        "z/nested/repeated_terrain.obj",
    ]
    for name, content in fake_wandb.contents.items():
        assert content == before[name][0]
    assert snapshot(tmp_path) == before
    assert fake_wandb.artifact_options == {
        "name": "clips",
        "type": "terrains-motions",
        "metadata": {
            "schema_version": 1,
            "clip_count": 2,
            "formats": ["beyondmimic", "native"],
        },
    }
    assert fake_wandb.calls[-4:] == ["log", "wait", "link", "finish"]
    assert fake_wandb.exit_code == 0


def test_interactive_normalization_and_flat_staging_preserve_sources(
    tmp_path, fake_wandb
):
    obstacle_motion, obstacle_dir = write_interactive(
        tmp_path, "nested/motion_20260912_111111"
    )
    flat_motion, flat_dir = write_interactive(
        tmp_path, "nested/motion_20260912_222222", flat=True
    )
    before = snapshot(tmp_path)
    assert uploader.main(args(tmp_path)) == 0
    for motion, sidecar in [(obstacle_motion, obstacle_dir), (flat_motion, flat_dir)]:
        prefix = sidecar.relative_to(tmp_path).as_posix()
        assert fake_wandb.contents[f"{prefix}/sample_motion.npz"] == motion.read_bytes()
        assert (
            fake_wandb.contents[f"{prefix}/sample_terrain.obj"]
            == (sidecar / "multi_boxes_scaled.obj").read_bytes()
        )
    terrain_key = "nested/motion_20260912_111111/sample_terrain.npy"
    assert (
        fake_wandb.contents[terrain_key] == (obstacle_dir / "terrain.npy").read_bytes()
    )
    flat_key = "nested/motion_20260912_222222/sample_terrain.npy"
    empty = np.load(io.BytesIO(fake_wandb.contents[flat_key]), allow_pickle=False)
    assert empty.shape == (0, 10, 10)
    assert empty.dtype == np.float32
    assert not fake_wandb.paths[flat_key].exists()
    assert snapshot(tmp_path) == before
    assert len(fake_wandb.contents) == 6


def test_generated_motion_commands_survive_upload_verbatim(tmp_path, fake_wandb):
    from motion_matching.io.motion_io import save_motion_npz

    qpos = np.zeros((6, 36), dtype=np.float32)
    qpos[:, 0] = 1
    qpos[:, 6] = 0.8
    commands = np.eye(15, dtype=np.float32)[:6]
    source = tmp_path / "generated_motion.npz"
    save_motion_npz(qpos, str(source), target_fps=60, source_fps=60, vel_cmd=commands)
    np.save(tmp_path / "generated_terrain.npy", np.zeros((0, 10, 10)))
    assert uploader.main(args(tmp_path)) == 0
    content = fake_wandb.contents["generated_motion.npz"]
    assert content == source.read_bytes()
    with np.load(io.BytesIO(content)) as saved:
        np.testing.assert_array_equal(saved["vel_cmd"], commands)


@pytest.mark.parametrize("hyphens", [True, False])
def test_dry_run_imports_no_sdk_and_writes_nothing(tmp_path, no_sdk, capsys, hyphens):
    write_interactive(tmp_path, flat=True)
    before = snapshot(tmp_path)
    cli = args(
        tmp_path,
        "--dry-run",
        "--entity",
        "team",
        "--project",
        "public-data",
        "--organization",
        "my-org",
    )
    if hyphens:
        cli = [arg.replace("_", "-") if arg.startswith("--") else arg for arg in cli]
    assert uploader.main(cli) == 0
    output = capsys.readouterr().out
    assert "generated empty terrain (0, 10, 10)" in output
    assert "sample_motion.npz" in output
    assert "my-org/wandb-registry-terrains-motions/clips" in output
    assert "Upload project: public-data; entity: team" in output
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize("missing", [False, True])
def test_empty_or_missing_directory_fails_before_sdk(tmp_path, no_sdk, capsys, missing):
    assert uploader.main(args(tmp_path / "missing" if missing else tmp_path)) == 1
    assert "Upload failed:" in capsys.readouterr().err


@pytest.mark.parametrize(
    "name", ["mesh_terrain.npz", "terrain_123.npz", "nested/terrain.npz"]
)
def test_mesh_terrain_npz_rejected_before_sdk(tmp_path, no_sdk, capsys, name):
    write_clip(tmp_path)
    path = tmp_path / name
    path.parent.mkdir(exist_ok=True)
    np.savez(path, vertices=np.zeros((3, 3)))
    assert uploader.main(args(tmp_path)) == 1
    error = capsys.readouterr().err
    assert "Mesh terrain NPZ is unsupported" in error
    assert "*_terrain.npy" in error


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_key",
        "qpos",
        "few_frames",
        "joint_width",
        "body_count",
        "velocity_length",
        "negative_fps",
        "multiple_fps",
        "nan",
        "object",
        "commands",
        "native_names",
        "duplicate_names",
        "raw_names",
        "corrupt",
    ],
)
def test_invalid_motion_fails_before_sdk(tmp_path, no_sdk, capsys, mutation):
    native = mutation in {"native_names", "duplicate_names"}
    arrays = motion_arrays(native=native, frames=1 if mutation == "few_frames" else 4)
    if mutation == "missing_key":
        del arrays["body_pos_w"]
    elif mutation == "qpos":
        arrays = {"qpos": np.zeros((4, 36))}
    elif mutation == "joint_width":
        arrays["joint_pos"] = np.zeros((4, 28))
    elif mutation == "body_count":
        arrays["body_pos_w"] = np.zeros((4, 32, 3))
    elif mutation == "velocity_length":
        arrays["joint_vel"] = arrays["joint_vel"][:-1]
    elif mutation == "negative_fps":
        arrays["fps"] = np.array(-1)
    elif mutation == "multiple_fps":
        arrays["fps"] = np.array([50, 50])
    elif mutation == "nan":
        arrays["body_lin_vel_w"][0, 0, 0] = np.nan
    elif mutation == "object":
        arrays["joint_pos"] = arrays["joint_pos"].astype(object)
    elif mutation == "commands":
        arrays["vel_cmd"] = np.zeros((3, 15))
    elif mutation == "native_names":
        del arrays["joint_names"]
    elif mutation == "duplicate_names":
        arrays["body_names"][:] = "duplicate"
    elif mutation == "raw_names":
        arrays["joint_names"] = np.array([f"joint{i}" for i in range(29)])
    source = tmp_path / "bad_motion.npz"
    np.savez(source, **arrays)
    if mutation == "corrupt":
        source.write_bytes(b"not an npz")
    assert uploader.main(args(tmp_path)) == 1
    assert "Invalid motion" in capsys.readouterr().err


@pytest.mark.parametrize(
    "terrain",
    [
        np.zeros((2, 10)),
        np.zeros((2, 0, 10)),
        np.zeros((2, 10, 9)),
        np.full((2, 10, 10), np.nan),
        np.zeros((2, 10, 10), dtype=object),
    ],
)
def test_invalid_terrain_fails_before_sdk(tmp_path, no_sdk, capsys, terrain):
    write_clip(tmp_path)
    np.save(tmp_path / "sample_terrain.npy", terrain)
    assert uploader.main(args(tmp_path)) == 1
    assert "Invalid terrain" in capsys.readouterr().err


def test_recursive_flat_bundle_and_empty_terrain_are_supported(tmp_path):
    write_clip(tmp_path, "a/native_motion.npz", native=True, paired=False)
    write_clip(tmp_path, "b/motion_raw.npz", paired=False)
    flat = uploader.prepare_dataset(tmp_path)
    assert len(flat.files) == 2
    np.save(tmp_path / "a/native_terrain.npy", np.zeros((0, 10, 10)))
    with pytest.raises(ValueError, match="Mixed terrain bundle.*unpaired"):
        uploader.prepare_dataset(tmp_path)
    np.save(tmp_path / "b/terrain_raw.npy", np.zeros((0, 10, 10)))
    assert len(uploader.prepare_dataset(tmp_path).files) == 4


def test_terrain_pairs_use_last_motion_component_in_same_directory(tmp_path):
    write_clip(tmp_path, "motion_group/tricky_motion_motion.npz")
    assert [entry.name for entry in uploader.prepare_dataset(tmp_path).files] == [
        "motion_group/tricky_motion_motion.npz",
        "motion_group/tricky_motion_terrain.npy",
        "motion_group/tricky_motion_terrain.obj",
    ]


def test_orphan_terrain_rejected_instead_of_mtime_pairing(tmp_path, no_sdk, capsys):
    write_clip(tmp_path)
    np.save(tmp_path / "orphan_terrain.npy", np.zeros((0, 10, 10)))
    assert uploader.main(args(tmp_path)) == 1
    assert "no matching motion: orphan_terrain.npy" in capsys.readouterr().err


def test_obj_alone_does_not_replace_required_box_parameters(tmp_path):
    write_clip(tmp_path)
    (tmp_path / "sample_terrain.npy").unlink()
    with pytest.raises(ValueError, match="OBJ preview needs"):
        uploader.prepare_dataset(tmp_path)


def test_interactive_and_canonical_collision_fails_before_sdk(tmp_path, no_sdk, capsys):
    _, sidecar = write_interactive(tmp_path)
    write_clip(sidecar, paired=False)
    assert uploader.main(args(tmp_path)) == 1
    assert "Artifact path collision" in capsys.readouterr().err


def test_ambiguous_interactive_and_canonical_pair_rejected(tmp_path):
    motion, _ = write_interactive(tmp_path)
    np.save(
        tmp_path / motion.name.replace("motion", "terrain").replace(".npz", ".npy"),
        np.zeros((0, 10, 10)),
    )
    with pytest.raises(ValueError, match="Ambiguous canonical and interactive"):
        uploader.prepare_dataset(tmp_path)


def test_payload_symlink_cannot_upload_outside_selected_directory(tmp_path):
    outside = write_clip(tmp_path, "outside/source_motion.npz", paired=False)
    root = tmp_path / "selected"
    root.mkdir()
    (root / "source_motion.npz").symlink_to(outside)
    with pytest.raises(ValueError, match="outside dataset folder"):
        uploader.prepare_dataset(root)


@pytest.mark.parametrize(
    "registry",
    [
        "",
        " Public skills",
        "Public skills ",
        "a,b",
        "a:b",
        "a/b",
        "a!b",
        "a\tb",
        "a\nb",
        "a\x00b",
    ],
)
def test_invalid_registry_name_is_rejected_before_sdk(
    tmp_path, no_sdk, capsys, registry
):
    write_clip(tmp_path)
    with pytest.raises(SystemExit) as error:
        uploader.main(args(tmp_path, "--registry", registry))
    assert error.value.code == 2
    assert "registry name may contain" in capsys.readouterr().err


def test_defaults_respect_sdk_project_config_and_omit_account_overrides(
    tmp_path, fake_wandb
):
    write_clip(tmp_path)
    assert uploader.main(args(tmp_path)) == 0
    assert fake_wandb.init_options == {"project": "parkour_registry", "name": "clips"}
    assert fake_wandb.target_path == "wandb-registry-terrains-motions/clips"
    fake_wandb.settings.project = "configured-project"
    fake_wandb.contents.clear()
    assert uploader.main(args(tmp_path)) == 0
    assert fake_wandb.init_options == {"project": "configured-project", "name": "clips"}


def test_explicit_destination_and_link_membership_version_are_printed(
    tmp_path, fake_wandb, capsys
):
    write_clip(tmp_path)
    fake_wandb.settings.project = "configured-project"
    fake_wandb.linked_name = "resolved-org/wandb-registry-Public skills/clips:v7"
    assert (
        uploader.main(
            args(
                tmp_path,
                "--entity",
                "team",
                "--project",
                "data",
                "--registry",
                "Public skills",
                "--organization",
                "My Org",
            )
        )
        == 0
    )
    assert fake_wandb.init_options == {
        "project": "data",
        "name": "clips",
        "entity": "team",
    }
    assert fake_wandb.target_path == "My Org/wandb-registry-Public skills/clips"
    output = capsys.readouterr().out
    assert fake_wandb.linked_name in output
    assert ":v99" not in output
    assignment = next(
        line for line in output.splitlines() if line.startswith("REGISTRY=")
    )
    assert assignment == f"REGISTRY={shlex.quote(fake_wandb.linked_name)}"
    assert shlex.split(assignment) == ["REGISTRY=" + fake_wandb.linked_name]


@pytest.mark.parametrize(
    "step", ["init", "artifact", "add_file", "log", "wait", "link", "finish"]
)
def test_upload_errors_are_nonzero_and_runs_finish(tmp_path, fake_wandb, capsys, step):
    write_interactive(tmp_path, flat=True)
    fake_wandb.fail = step
    before = snapshot(tmp_path)
    assert uploader.main(args(tmp_path)) == 1
    captured = capsys.readouterr()
    assert f"{step} failure" in captured.err
    assert "REGISTRY=" not in captured.out
    if step != "init":
        assert fake_wandb.calls[-1] == "finish"
        assert fake_wandb.exit_code == (0 if step == "finish" else 1)
    assert snapshot(tmp_path) == before
    for path in fake_wandb.paths.values():
        if not path.is_relative_to(tmp_path):
            assert not path.exists()


@pytest.mark.parametrize(
    "linked_name",
    ["org/project/clips:v99", "org/wandb-registry-terrains-motions/clips:latest"],
)
def test_source_or_unversioned_link_result_is_not_reported_as_success(
    tmp_path, fake_wandb, capsys, linked_name
):
    write_clip(tmp_path)
    fake_wandb.linked_name = linked_name
    assert uploader.main(args(tmp_path)) == 1
    assert "versioned linked registry Artifact" in capsys.readouterr().err
    assert fake_wandb.exit_code == 1


def test_missing_link_result_fails_with_sdk_version_hint(
    tmp_path, fake_wandb, monkeypatch, capsys
):
    write_clip(tmp_path)
    monkeypatch.setattr(fake_wandb, "link_artifact", lambda **kwargs: None)
    assert uploader.main(args(tmp_path)) == 1
    assert "wandb>=0.22.1" in capsys.readouterr().err
    assert fake_wandb.exit_code == 1


def test_original_failure_survives_cleanup_failure(
    tmp_path, fake_wandb, monkeypatch, capsys
):
    write_clip(tmp_path)
    fake_wandb.fail = "link"

    def finish(**kwargs):
        raise RuntimeError("cleanup failure")

    monkeypatch.setattr(fake_wandb, "finish", finish)
    assert uploader.main(args(tmp_path)) == 1
    error = capsys.readouterr().err
    assert "Upload failed: link failure" in error
    assert "cleanup failure" in error


def test_missing_sdk_gives_optional_extra_install_instruction(
    tmp_path, monkeypatch, capsys
):
    write_clip(tmp_path)

    def missing(name):
        raise ImportError(name)

    monkeypatch.setattr(uploader.importlib, "import_module", missing)
    assert uploader.main(args(tmp_path)) == 1
    assert "python -m pip install -e '.[registry]'" in capsys.readouterr().err


@pytest.mark.parametrize("entrypoint", ["module", "wrapper"])
def test_cli_from_unrelated_cwd_needs_no_sdk_or_gui(tmp_path, entrypoint):
    root = tmp_path / "dataset with spaces"
    write_clip(root)
    package_root = Path(uploader.__file__).resolve().parents[1]
    # Run a fresh interpreter with an import guard, even on hosts with W&B installed.
    program = """
import builtins, runpy, sys
real_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'wandb', 'viser', 'mujoco', 'tkinter'}:
        raise AssertionError('unexpected import: ' + name)
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded
entrypoint = sys.argv.pop(1)
if entrypoint == 'module':
    runpy.run_module('motion_matching.upload_dataset', run_name='__main__')
else:
    runpy.run_path(entrypoint, run_name='__main__')
"""
    entry = (
        "module"
        if entrypoint == "module"
        else str(package_root / "tracking_utils/upload_npz_newformat.py")
    )
    env = {
        **os.environ,
        "PYTHONPATH": str(package_root) if entrypoint == "module" else "",
    }
    result = subprocess.run(
        [sys.executable, "-c", program, entry, *args(root, "--dry-run")],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "Dry run complete" in result.stdout
    assert "sample_motion.npz" in result.stdout
