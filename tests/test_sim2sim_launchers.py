"""Exercise the PHP-owned sim2sim launchers without starting MuJoCo."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def write_executable(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(0o755)


@pytest.fixture
def fake_runtime(tmp_path):
    holosoma = tmp_path / "holosoma checkout"
    (holosoma / "scripts").mkdir(parents=True)
    (holosoma / "scripts/source_mujoco_setup.sh").write_text(":\n")
    (holosoma / "scripts/source_inference_setup.sh").write_text(":\n")
    sim = holosoma / "src/holosoma/holosoma/run_sim.py"
    policy = holosoma / "src/holosoma_inference/holosoma_inference/run_policy.py"
    sim.parent.mkdir(parents=True)
    policy.parent.mkdir(parents=True)
    sim.write_text("")
    policy.write_text("")

    command_log = tmp_path / "command.log"
    fake_bin = tmp_path / "bin"
    recorder = """#!/usr/bin/env bash
{
    printf 'cwd=%s\\n' "$PWD"
    printf 'arg=%s\\n' "$@"
} > "$COMMAND_LOG"
"""
    write_executable(fake_bin / "python", recorder)
    write_executable(fake_bin / "python3", recorder)
    write_executable(fake_bin / "rm", "#!/usr/bin/env bash\nexit 0\n")

    env = os.environ.copy()
    for key in ("RUN", "STEP", "BACKBONE", "STUDENT", "CONDA_ENV_NAME"):
        env.pop(key, None)
    env.update(
        HOLOSOMA_ROOT=str(holosoma),
        COMMAND_LOG=str(command_log),
        PATH=f"{fake_bin}:{env['PATH']}",
        DISPLAY=":test",
    )
    return holosoma, command_log, env


def recorded(path: Path) -> tuple[str, list[str]]:
    lines = path.read_text().splitlines()
    return lines[0].removeprefix("cwd="), [line.removeprefix("arg=") for line in lines[1:]]


def test_sim_launcher_uses_pinned_holosoma_and_forwards_overrides(fake_runtime, tmp_path):
    holosoma, command_log, env = fake_runtime
    result = subprocess.run(
        [str(ROOT / "run_php_sim.sh"), "--simulator.config.no-render"],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    cwd, args = recorded(command_log)
    assert cwd == str(holosoma)
    assert args == [
        str(holosoma / "src/holosoma/holosoma/run_sim.py"),
        "robot:g1-29dof",
        "sensor.d435i_front_depth:g1-d435i-front-depth",
        "plugin.depth:depth-shm-d435i",
        "terrain:terrain-load-step",
        "--robot.asset.xml-file",
        "g1/g1_29dof_halfspherehand.xml",
        "--simulator.config.bridge.enabled=True",
        "--simulator.config.no-render",
    ]


def test_inference_launcher_resolves_local_models_from_calling_directory(
    fake_runtime, tmp_path
):
    holosoma, command_log, env = fake_runtime
    caller = tmp_path / "caller"
    models = caller / "models"
    models.mkdir(parents=True)
    (models / "depth_backbone.onnx").write_bytes(b"backbone")
    (models / "student.onnx").write_bytes(b"student")
    env.update(BACKBONE="models/depth_backbone.onnx", STUDENT="models/student.onnx")

    result = subprocess.run(
        [str(ROOT / "run_php_inference.sh"), "--task.depth-shm.no-required"],
        cwd=caller,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    cwd, args = recorded(command_log)
    assert cwd == str(holosoma)
    assert args[:5] == [
        str(holosoma / "src/holosoma_inference/holosoma_inference/run_policy.py"),
        "inference:g1-wbt-distillation-d435i",
        "--task.interface",
        "lo",
        "--task.model-path",
    ]
    assert args[5] == f"['{models / 'depth_backbone.onnx'}','{models / 'student.onnx'}']"
    assert args[6] == "--task.depth-shm.no-required"


def test_inference_launcher_builds_wandb_pair(fake_runtime, tmp_path):
    _, command_log, env = fake_runtime
    env.update(RUN="wandb://team/project/run-id", STEP="model_29999")

    result = subprocess.run(
        [str(ROOT / "run_php_inference.sh")],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    _, args = recorded(command_log)
    assert args[5] == (
        "['wandb://team/project/run-id/model_29999/depth_backbone.onnx',"
        "'wandb://team/project/run-id/model_29999/student.onnx']"
    )


@pytest.mark.parametrize("launcher", ["run_php_sim.sh", "run_php_inference.sh"])
def test_launchers_report_missing_holosoma_before_activation(tmp_path, launcher):
    env = os.environ.copy()
    env["HOLOSOMA_ROOT"] = str(tmp_path / "missing")
    result = subprocess.run(
        [str(ROOT / launcher)],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "Holosoma checkout not found" in result.stderr
    assert "git submodule update --init" in result.stderr
