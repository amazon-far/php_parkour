"""Run the public launchers without GPUs, network access, or IsaacSim startup."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "training_runs"
LAUNCHERS = (
    "run_terrain_teacher.sh",
    "run_terrain_warp_distill.sh",
    "eval_teacher.sh",
    "eval_student.sh",
)


def make_bundle(path):
    path.mkdir(parents=True)
    np.savez(
        path / "example_motion.npz",
        fps=np.array([50]),
        joint_pos=np.zeros((2, 36)),
        joint_vel=np.zeros((2, 35)),
        body_pos_w=np.zeros((2, 32, 3)),
        body_quat_w=np.zeros((2, 32, 4)),
        body_lin_vel_w=np.zeros((2, 32, 3)),
        body_ang_vel_w=np.zeros((2, 32, 3)),
        joint_names=np.array([f"joint_{i}" for i in range(29)]),
        body_names=np.array([f"body_{i}" for i in range(32)]),
    )
    np.save(path / "example_terrain.npy", np.zeros((0, 10, 10)))
    return f"file://{path}"


@pytest.fixture
def launcher(tmp_path):
    repo = tmp_path / "release checkout"
    scripts = repo / "wbt_training" / "training_runs"
    scripts.mkdir(parents=True)
    for name in (*LAUNCHERS, "setup_env.sh"):
        shutil.copy2(SCRIPTS / name, scripts / name)
    utils = repo / "wbt_training" / "utils"
    utils.mkdir()
    shutil.copy2(SCRIPTS.parent / "utils" / "registry.py", utils / "registry.py")
    (repo / "wbt_training" / "__init__.py").touch()
    (utils / "__init__.py").touch()
    (repo / "scripts").mkdir()
    (repo / "scripts" / "source_isaacsim_setup.sh").write_text(
        'printf activated > "$ACTIVATED"\n'
    )
    bin_dir = tmp_path / "env" / "bin"
    bin_dir.mkdir(parents=True)
    capture = tmp_path / "capture.py"
    capture.write_text(
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "Path(os.environ['CAPTURE']).write_text(json.dumps({'tool': sys.argv[1], "
        "'args': sys.argv[2:], 'cwd': os.getcwd(), 'env': dict(os.environ)}))\n"
    )
    for name in ("python", "torchrun"):
        script = bin_dir / name
        script.write_text(
            "#!/bin/bash\n"
            f'if [[ "$1" == "-" ]]; then exec {shlex.quote(sys.executable)} "$@"; fi\n'
            f'exec {shlex.quote(sys.executable)} {shlex.quote(str(capture))} {name} "$@"\n'
        )
        script.chmod(0o755)
    (bin_dir / "nvidia-smi").write_text("#!/bin/bash\nprintf '0, 900\n1, 50\n'\n")
    (bin_dir / "nvidia-smi").chmod(0o755)
    modules = tmp_path / "modules"
    modules.mkdir()
    (modules / "wandb.py").write_text(
        """import json, os, shutil
from pathlib import Path

def event(kind, value=''):
    with Path(os.environ['WANDB_EVENTS']).open('a') as stream:
        stream.write(json.dumps([kind, value]) + '\\n')

event('import')
if not os.environ.get('REMOTE_SPEC'):
    raise AssertionError('Local launch must not import wandb')
spec = json.loads(Path(os.environ['REMOTE_SPEC']).read_text())

class Artifact:
    type = 'terrains-motions'
    version = 'v7'
    def __init__(self, name):
        self.qualified_name = name
    def download(self, root):
        event('download', self.qualified_name)
        shutil.copytree(spec['bundle'], root)
        return root

class Run:
    def __init__(self, name):
        self.spec = spec.get('runs', {}).get(name, spec)
        self.config = self.spec.get('config', {})
    def used_artifacts(self):
        return [Artifact(name) for name in self.spec.get('used', [])]

class Api:
    def artifact(self, name):
        event('artifact', name)
        return Artifact(name)
    def run(self, name):
        event('run', name)
        return Run(name)
"""
    )
    registry_a = make_bundle(tmp_path / "data A")
    registry_b = make_bundle(tmp_path / "data B")
    checkpoints = [tmp_path / f"teacher {i}.pt" for i in range(2)]
    for checkpoint in checkpoints:
        checkpoint.write_bytes(
            b"checkpoint fixture; the command spy does not load weights"
        )
    env = {
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "HOME": os.environ["HOME"],
        "HSSIM_PYTHON": str(bin_dir / "python"),
        "PYTHONPATH": str(modules),
        "CAPTURE": str(tmp_path / "command.json"),
        "ACTIVATED": str(tmp_path / "activated"),
        "WANDB_EVENTS": str(tmp_path / "wandb-events"),
        "CONVERT_CACHE_DIR": str(tmp_path / "cache"),
        "CUDA_VISIBLE_DEVICES": "3",
        "REGISTRY": registry_a,
        "CHECKPOINT": str(checkpoints[0]),
        "TEACHER_CHECKPOINT": str(checkpoints[0]),
    }

    def run(name, extra=(), **overrides):
        effective = {**env, **overrides}
        effective = {
            key: value for key, value in effective.items() if value is not None
        }
        for marker in ("CAPTURE", "ACTIVATED", "WANDB_EVENTS"):
            Path(env[marker]).unlink(missing_ok=True)
        result = subprocess.run(
            ["bash", str(scripts / name), *extra],
            cwd=tmp_path,
            env=effective,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        capture_file = Path(env["CAPTURE"])
        command = (
            json.loads(capture_file.read_text()) if capture_file.exists() else None
        )
        return result, command

    run.env = env
    run.registries = [registry_a, registry_b]
    run.checkpoints = checkpoints
    run.repo = repo
    run.bin_dir = bin_dir
    run.tmp = tmp_path
    return run


def option(command, flag):
    return command["args"][command["args"].index(flag) + 1]


@pytest.mark.parametrize("name", LAUNCHERS)
@pytest.mark.parametrize("logger", [None, ""])
def test_launch_defaults_to_wandb_without_inventing_metadata(launcher, name, logger):
    result, command = launcher(name, LOGGER=logger)
    assert result.returncode == 0, result.stderr
    assert "logger:wandb" in command["args"]
    assert "logger:disabled" not in command["args"]
    assert "--logger.project" not in command["args"]
    assert "--logger.entity" not in command["args"]
    assert "--logger.name" not in command["args"]
    assert "WANDB_BASE_URL" not in command["env"]
    assert not Path(launcher.env["WANDB_EVENTS"]).exists()


@pytest.mark.parametrize("name", LAUNCHERS)
def test_local_launch_is_offline_and_uses_disabled_logger(launcher, name):
    result, command = launcher(
        name,
        LOGGER="disabled",
        WANDB_PROJECT="ignored-project",
        WANDB_ENTITY="ignored-team",
        RUN_NAME="ignored-name",
    )
    assert result.returncode == 0, result.stderr
    assert command["cwd"] == str(launcher.repo)
    assert "logger:disabled" in command["args"]
    assert "logger:wandb" not in command["args"]
    assert "--logger.project" not in command["args"]
    assert "--logger.entity" not in command["args"]
    assert "--logger.name" not in command["args"]
    assert "WANDB_BASE_URL" not in command["env"]
    assert command["env"]["CUDA_VISIBLE_DEVICES"] == "3"
    assert not Path(launcher.env["WANDB_EVENTS"]).exists()
    assert option(command, "--training.registry-name") == launcher.registries[0]


@pytest.mark.parametrize("name", LAUNCHERS)
@pytest.mark.parametrize("logger", [None, "wandb"])
def test_wandb_logging_uses_only_supplied_metadata(launcher, name, logger):
    result, command = launcher(
        name,
        LOGGER=logger,
        WANDB_PROJECT="my-project",
        WANDB_ENTITY="my-team",
        RUN_NAME="trial with spaces",
        WANDB_BASE_URL="https://wandb.example.org",
    )
    assert result.returncode == 0, result.stderr
    assert "logger:wandb" in command["args"]
    assert option(command, "--logger.project") == "my-project"
    assert option(command, "--logger.entity") == "my-team"
    assert option(command, "--logger.name") == "trial with spaces"
    assert command["env"]["WANDB_BASE_URL"] == "https://wandb.example.org"
    assert not Path(launcher.env["WANDB_EVENTS"]).exists()


@pytest.mark.parametrize("name", LAUNCHERS)
def test_multiple_registries_keep_user_order(launcher, name):
    ordered = ",".join(reversed(launcher.registries))
    result, command = launcher(
        name,
        REGISTRY=ordered,
        TEACHER_CHECKPOINT=",".join(map(str, launcher.checkpoints)),
    )
    assert result.returncode == 0, result.stderr
    assert option(command, "--training.registry-name") == ordered


@pytest.mark.parametrize(
    ("name", "overrides", "message"),
    [
        ("run_terrain_teacher.sh", {"REGISTRY": None}, "REGISTRY is required"),
        ("eval_teacher.sh", {"REGISTRY": None}, "REGISTRY is required"),
        ("eval_student.sh", {"REGISTRY": None}, "REGISTRY is required"),
        ("eval_teacher.sh", {"CHECKPOINT": None}, "CHECKPOINT is required"),
        ("eval_student.sh", {"CHECKPOINT": None}, "CHECKPOINT is required"),
        (
            "run_terrain_warp_distill.sh",
            {"TEACHER_CHECKPOINT": None},
            "TEACHER_CHECKPOINT is required",
        ),
        ("run_terrain_teacher.sh", {"NGPUS": "0"}, "NGPUS must be"),
        ("run_terrain_teacher.sh", {"LOGGER": "typo"}, "LOGGER must be"),
        ("run_terrain_warp_distill.sh", {"FINETUNE": "maybe"}, "FINETUNE must be"),
        ("run_terrain_warp_distill.sh", {"WARMUP_STEPS": "-1"}, "WARMUP_STEPS must be"),
        ("eval_teacher.sh", {"NUM_ENVS": "no"}, "NUM_ENVS must be"),
        ("eval_teacher.sh", {"MAX_EVAL_STEPS": "0"}, "MAX_EVAL_STEPS must be"),
        ("eval_teacher.sh", {"EXPORT_ONNX": "bad"}, "EXPORT_ONNX must be"),
        ("eval_teacher.sh", {"CHECKPOINT": "/missing.pt"}, "Checkpoint must be"),
        ("run_terrain_teacher.sh", {"REGISTRY": "file:///missing"}, "does not exist"),
        (
            "run_terrain_teacher.sh",
            {"REGISTRY": "https://example.org/artifact"},
            "Invalid REGISTRY",
        ),
        (
            "eval_teacher.sh",
            {"CHECKPOINT": "wandb://team/project"},
            "Invalid W&B checkpoint",
        ),
        (
            "eval_student.sh",
            {"CHECKPOINT": "wandb://team/project/run/model.onnx"},
            "must end in .pt",
        ),
    ],
)
def test_invalid_inputs_fail_before_activation(launcher, name, overrides, message):
    result, command = launcher(name, **overrides)
    assert result.returncode != 0
    assert message in result.stderr
    assert command is None
    assert not Path(launcher.env["ACTIVATED"]).exists()
    assert not Path(launcher.env["WANDB_EVENTS"]).exists()


def test_valid_remote_entry_does_not_hide_later_invalid_local_input(launcher):
    result, _ = launcher(
        "run_terrain_teacher.sh", REGISTRY="team/project/motions:v1,file:///missing"
    )
    assert result.returncode != 0
    assert "does not exist" in result.stderr
    assert not Path(launcher.env["WANDB_EVENTS"]).exists()


@pytest.mark.parametrize("name", LAUNCHERS)
def test_auto_requires_explicit_remote_teacher_usage(launcher, name):
    result, command = launcher(name, REGISTRY="auto")
    assert result.returncode != 0
    assert "REGISTRY=auto requires" in result.stderr
    assert command is None
    assert not Path(launcher.env["WANDB_EVENTS"]).exists()


def test_distillation_rejects_unpaired_teacher_registries(launcher):
    result, _ = launcher(
        "run_terrain_warp_distill.sh", REGISTRY=",".join(launcher.registries)
    )
    assert result.returncode != 0
    assert "one REGISTRY per TEACHER_CHECKPOINT" in result.stderr
    assert not Path(launcher.env["ACTIVATED"]).exists()


@pytest.mark.parametrize("suffix", [",", ",,file:///missing"])
def test_empty_registry_entries_are_rejected(launcher, suffix):
    result, _ = launcher(
        "run_terrain_teacher.sh", REGISTRY=launcher.registries[0] + suffix
    )
    assert result.returncode != 0
    assert "empty entries" in result.stderr


def test_local_bundle_validation(launcher):
    bundle = Path(launcher.registries[0].removeprefix("file://"))
    (bundle / "example_terrain.npy").rename(bundle / "wrong_terrain.npy")
    result, _ = launcher("run_terrain_teacher.sh")
    assert result.returncode != 0
    assert "pair each" in result.stderr
    (bundle / "wrong_terrain.npy").rename(bundle / "example_terrain.npy")
    (bundle / "example_motion.npz").write_bytes(b"not an archive")
    result, _ = launcher("run_terrain_teacher.sh")
    assert result.returncode != 0
    assert "Invalid motion NPZ" in result.stderr


def test_relative_paths_and_shell_metacharacters_remain_data(launcher):
    bundle = launcher.tmp / "data 'quotes' $(touch INJECTED)"
    registry = make_bundle(bundle)
    relative = "file://" + str(bundle.relative_to(launcher.tmp))
    checkpoint = str(launcher.checkpoints[0].relative_to(launcher.tmp))
    result, command = launcher(
        "eval_teacher.sh", REGISTRY=relative, CHECKPOINT=checkpoint
    )
    assert result.returncode == 0, result.stderr
    assert option(command, "--training.registry-name") == registry
    assert option(command, "--checkpoint") == str(launcher.checkpoints[0])
    assert not (launcher.tmp / "INJECTED").exists()


def test_gpu_selection_torchrun_and_finetune_preserve_launch_controls(launcher):
    result, command = launcher("run_terrain_teacher.sh", CUDA_VISIBLE_DEVICES=None)
    assert result.returncode == 0, result.stderr
    assert command["tool"] == "python"
    assert command["env"]["CUDA_VISIBLE_DEVICES"] == "1"
    result, command = launcher(
        "run_terrain_warp_distill.sh",
        extra=("--training.num-envs", "128"),
        NGPUS="4",
        FINETUNE="0",
        WARMUP_STEPS="17",
        ENV_NAME="custom",
    )
    assert result.returncode == 0, result.stderr
    assert command["tool"] == "torchrun"
    assert "--nproc_per_node=4" in command["args"]
    assert "exp:g1-wbt-terrain-warp-distill" in command["args"]
    assert option(command, "--algo.config.distillation-warmup-steps") == "17"
    assert command["args"][-2:] == ["--training.num-envs", "128"]
    assert command["env"]["ENV_NAME"] == "custom"


def test_conda_root_and_env_name_select_interpreter(launcher):
    conda = launcher.tmp / "conda"
    target = conda / "envs" / "custom" / "bin"
    target.mkdir(parents=True)
    shutil.copy2(launcher.bin_dir / "python", target / "python")
    result, command = launcher(
        "run_terrain_teacher.sh",
        HSSIM_PYTHON=None,
        CONDA_ROOT=str(conda),
        ENV_NAME="custom",
    )
    assert result.returncode == 0, result.stderr
    assert command["tool"] == "python"


def test_eval_flags_and_gpu_override(launcher):
    result, command = launcher(
        "eval_teacher.sh", CUDA_VISIBLE_DEVICES=None, GPU_INDEX="2", EXPORT_ONNX="true"
    )
    assert result.returncode == 0, result.stderr
    assert command["env"]["CUDA_VISIBLE_DEVICES"] == "2"
    assert option(command, "--training.export-onnx") == "True"
    assert option(command, "--training.max-eval-steps") == "10000"
    result, command = launcher("eval_student.sh", extra=("--show-depth", "True"))
    assert result.returncode == 0, result.stderr
    assert command["args"][-2:] == ["--show-depth", "True"]
    assert option(command, "--training.max-eval-steps") == "1000"


def remote_spec(launcher, **values):
    spec = launcher.tmp / "remote.json"
    spec.write_text(json.dumps({"bundle": launcher.registries[0][7:], **values}))
    return str(spec)


@pytest.mark.parametrize("registry", [None, ""])
@pytest.mark.parametrize("teachers", [("climb",), ("step", "climb")])
def test_student_training_defaults_to_teacher_datasets(launcher, registry, teachers):
    datasets = [f"team/data/{teacher}:v7" for teacher in teachers]
    checkpoints = ",".join(f"wandb://team/project/{teacher}" for teacher in teachers)
    spec = remote_spec(
        launcher,
        runs={
            f"team/project/{teacher}": {"used": [dataset]}
            for teacher, dataset in zip(teachers, datasets, strict=True)
        },
    )
    result, command = launcher(
        "run_terrain_warp_distill.sh",
        REGISTRY=registry,
        TEACHER_CHECKPOINT=checkpoints,
        REMOTE_SPEC=spec,
    )
    assert result.returncode == 0, result.stderr
    assert option(command, "--training.registry-name") == ",".join(datasets)
    assert option(command, "--training.teacher-checkpoint") == checkpoints


def test_student_training_preserves_explicit_registry_without_teacher_lookup(launcher):
    result, command = launcher(
        "run_terrain_warp_distill.sh",
        REGISTRY=launcher.registries[0],
        TEACHER_CHECKPOINT="wandb://team/project/teacher",
    )
    assert result.returncode == 0, result.stderr
    assert option(command, "--training.registry-name") == launcher.registries[0]
    assert not Path(launcher.env["WANDB_EVENTS"]).exists()


@pytest.mark.parametrize("registry", [None, ""])
def test_student_training_default_requires_remote_teacher_metadata(launcher, registry):
    result, command = launcher("run_terrain_warp_distill.sh", REGISTRY=registry)
    assert result.returncode != 0
    assert "REGISTRY=auto requires" in result.stderr
    assert command is None
    assert not Path(launcher.env["ACTIVATED"]).exists()
    assert not Path(launcher.env["WANDB_EVENTS"]).exists()


def test_remote_cache_includes_namespace_and_reuses_immutable_content(launcher):
    spec = remote_spec(launcher)
    ordered = "wandb://team/first/motions:latest,team/second/motions:latest"
    result, command = launcher(
        "run_terrain_teacher.sh", REGISTRY=ordered, REMOTE_SPEC=spec
    )
    assert result.returncode == 0, result.stderr
    names = option(command, "--training.registry-name")
    assert names == "team/first/motions:v7,team/second/motions:v7"
    paths = sorted(Path(launcher.env["CONVERT_CACHE_DIR"]).glob("*/example_motion.npz"))
    assert len(paths) == 2
    for path in paths:
        assert (
            path.read_bytes()
            == (Path(launcher.registries[0][7:]) / "example_motion.npz").read_bytes()
        )
    result, repeated = launcher(
        "run_terrain_teacher.sh", REGISTRY=ordered, REMOTE_SPEC=spec
    )
    assert result.returncode == 0, result.stderr
    assert option(repeated, "--training.registry-name") == names
    assert '"download"' not in Path(launcher.env["WANDB_EVENTS"]).read_text()


def test_teacher_auto_resolution_and_ambiguous_metadata(launcher):
    spec = remote_spec(launcher, used=["team/data/motions:v7"])
    result, command = launcher(
        "run_terrain_warp_distill.sh",
        REGISTRY="auto",
        REMOTE_SPEC=spec,
        TEACHER_CHECKPOINT="wandb://team/project/runs/run/model_9.pt",
    )
    assert result.returncode == 0, result.stderr
    assert (
        option(command, "--training.teacher-checkpoint")
        == "wandb://team/project/run/model_9.pt"
    )
    assert option(command, "--training.registry-name") == "team/data/motions:v7"
    spec = remote_spec(
        launcher, config={"training": {"registry_name": "file:///missing/cache"}}
    )
    result, command = launcher(
        "eval_teacher.sh",
        REGISTRY="auto",
        REMOTE_SPEC=spec,
        CHECKPOINT="wandb://team/project/run",
    )
    assert result.returncode != 0
    assert "set REGISTRY explicitly" in result.stderr
    assert command is None
    assert not Path(launcher.env["ACTIVATED"]).exists()


def test_student_auto_keeps_recorded_registry_order_and_duplicates(launcher):
    names = ["team/data/step:v7", "team/data/climb:v7", "team/data/step:v7"]
    spec = remote_spec(
        launcher,
        config={"php_registry": {"version": 1, "registry_names": names}},
        used=list(reversed(names[:2])),
    )
    result, command = launcher(
        "eval_student.sh",
        REGISTRY="auto",
        CHECKPOINT="wandb://team/project/student/model_99.pt",
        REMOTE_SPEC=spec,
    )
    assert result.returncode == 0, result.stderr
    assert option(command, "--training.registry-name") == ",".join(names)
    assert option(command, "--checkpoint") == "wandb://team/project/student/model_99.pt"


def test_logger_arguments_match_core_config_union(launcher):
    tyro = pytest.importorskip("tyro")
    from holosoma.config_types.experiment import ExperimentConfig
    from holosoma.utils.tyro_utils import TYRO_CONIFG

    for mode in ("disabled", "wandb"):
        result, command = launcher(
            "eval_teacher.sh", LOGGER=mode, WANDB_PROJECT="example"
        )
        assert result.returncode == 0, result.stderr
        args = command["args"]
        start = args.index(f"logger:{mode}")
        stop = args.index("--training.preprocess-hook-kwargs")
        config = tyro.cli(ExperimentConfig, args=args[start:stop], config=TYRO_CONIFG)
        assert config.logger.type == mode
        assert config.logger.video.enabled
        if mode == "wandb":
            assert config.logger.project == "example"


@pytest.mark.parametrize(
    ("name", "finetune"),
    [
        ("run_terrain_teacher.sh", "1"),
        ("run_terrain_warp_distill.sh", "1"),
        ("run_terrain_warp_distill.sh", "0"),
    ],
)
@pytest.mark.parametrize(
    "logger,expected_logger", [(None, "wandb"), ("disabled", "disabled")]
)
def test_complete_training_command_parses_against_php_presets(
    launcher, name, finetune, logger, expected_logger
):
    import tyro
    from holosoma.utils.tyro_utils import TYRO_CONIFG

    from wbt_training.config_values.experiment import AnnotatedExperimentConfig

    result, command = launcher(name, FINETUNE=finetune, LOGGER=logger)
    assert result.returncode == 0, result.stderr
    args = command["args"]
    config = tyro.cli(
        AnnotatedExperimentConfig, args=args[args.index("-m") + 2 :], config=TYRO_CONIFG
    )
    assert config.training.registry_name == launcher.registries[0]
    assert config.logger.type == expected_logger
    if name == "run_terrain_warp_distill.sh":
        assert config.training.teacher_checkpoint == str(launcher.checkpoints[0])
        assert config.algo.config.distillation_warmup_steps == 0


def test_pairing_preserves_motion_word_inside_clip_name(launcher):
    bundle = Path(launcher.registries[0].removeprefix("file://"))
    (bundle / "example_motion.npz").rename(bundle / "locomotion_motion.npz")
    (bundle / "example_terrain.npy").rename(bundle / "locomotion_terrain.npy")
    result, command = launcher("run_terrain_teacher.sh")
    assert result.returncode == 0, result.stderr
    assert command is not None
