"""CPU regression checks for the public PHP presets and checkpoint paths."""

import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path
from textwrap import dedent

import numpy as np
import pytest
import tyro
from holosoma.config_types.experiment import ExperimentConfig
from holosoma.config_values.logger import disabled, wandb
from holosoma.config_values.robot import g1_29dof as core_g1_robot
from holosoma.config_values.wbt.g1.experiment import g1_29dof_wbt
from holosoma.managers.utils import resolve_callable
from holosoma.utils.helpers import get_class
from holosoma.utils.tyro_utils import TYRO_CONIFG

from wbt_training.config_values import experiment, robot_config

PHP_PRESETS = (
    "g1_wbt_terrain_ref",
    "g1_wbt_terrain_no_heightscan",
    "g1_wbt_terrain_ref_noscale",
    "g1_wbt_terrain_no_heightscan_noscale",
    "g1_wbt_terrain_ref_pelvis",
    "g1_wbt_terrain_no_heightscan_pelvis",
    "g1_wbt_terrain_warp_distill",
    "g1_wbt_terrain_warp_distill_finetune",
)

PHP_CHECKPOINT_PATHS = (
    "wbt_training.config_values.robot_config:build_g1_implicit_actuators",
    "wbt_training.config_values.terrain:height_scan",
    "wbt_training.config_values.terrain:apply_terrain_preprocess",
    "wbt_training.config_values.curriculum:RelaxTerminationThresholds",
    "wbt_training.config_values.motion_command:PhpMotionCommand",
    "wbt_training.config_values.motion_command:velocity_command",
    "wbt_training.config_values.motion_command:which_motion",
    "wbt_training.utils.student_teacher.PhpDistillation",
    "wbt_training.utils.student_teacher.PhpDistillationPPO",
)


def _php_paths(value):
    if isinstance(value, dict):
        for item in value.values():
            yield from _php_paths(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _php_paths(item)
    elif isinstance(value, str) and value.startswith("wbt_training."):
        yield value


def test_package_import_is_inert_without_optional_dependencies():
    # -S excludes site packages, including Holosoma, torch, and the simulators.
    # Check sys.modules so silently caught ImportErrors cannot mask eager imports.
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-S",
            "-c",
            (
                "import sys; before = set(sys.modules); import wbt_training; "
                "assert set(sys.modules) - before == {'wbt_training'}"
            ),
        ],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_converter_works_with_only_numpy(tmp_path):
    # Ignore PYTHONPATH and site initialization. Expose this checkout and NumPy
    # explicitly, rejecting every other third-party import even when installed.
    script = dedent(
        """
        import sys
        from pathlib import Path

        sys.path[:0] = sys.argv[1:3]
        allowed = sys.stdlib_module_names | {"numpy", "wbt_training"}

        class NumpyOnlyImports:
            def find_spec(self, fullname, path=None, target=None):
                if fullname.partition(".")[0] not in allowed:
                    raise ModuleNotFoundError(f"Dependency unavailable: {fullname}", name=fullname)
                return None

        sys.meta_path.insert(0, NumpyOnlyImports())

        import numpy as np
        from wbt_training.config_values.robot_config import (
            G1_29DOF_JOINT_NAMES,
            G1_32BODY_NAMES,
            ISAACLAB_TO_MUJOCO_DOF,
            build_g1_implicit_actuators,
        )
        from wbt_training.training_runs.motion_convert import convert

        assert len(G1_29DOF_JOINT_NAMES) == 29
        assert len(G1_32BODY_NAMES) == 32
        assert callable(build_g1_implicit_actuators)
        directory = Path(sys.argv[3])
        source = directory / "source.npz"
        target = directory / "converted.npz"
        joint_pos = np.arange(58).reshape(2, 29)
        joint_vel = joint_pos + 100
        body_pos = np.arange(180).reshape(2, 30, 3)
        body_quat = np.tile([1.0, 0.0, 0.0, 0.0], (2, 30, 1))
        vel_cmd = np.eye(15)[:2]
        np.savez(
            source,
            fps=np.array([30]),
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            body_pos_w=body_pos,
            body_quat_w=body_quat,
            body_lin_vel_w=np.zeros((2, 30, 3)),
            body_ang_vel_w=np.zeros((2, 30, 3)),
            vel_cmd=vel_cmd,
        )
        assert convert(str(source), str(target))
        with np.load(target, allow_pickle=False) as data:
            assert data["joint_pos"].shape == (2, 36)
            assert data["joint_vel"].shape == (2, 35)
            assert data["body_pos_w"].shape == (2, 32, 3)
            assert data["joint_names"].tolist() == G1_29DOF_JOINT_NAMES
            assert data["body_names"].tolist() == G1_32BODY_NAMES
            assert np.array_equal(data["joint_pos"][:, :3], body_pos[:, 0])
            assert np.array_equal(data["joint_pos"][:, 3:7], body_quat[:, 0])
            assert np.array_equal(data["joint_pos"][:, 7:], joint_pos[:, ISAACLAB_TO_MUJOCO_DOF])
            assert np.array_equal(data["joint_vel"][:, 6:], joint_vel[:, ISAACLAB_TO_MUJOCO_DOF])
            assert np.array_equal(data["body_pos_w"][:, 7], body_pos[:, 18])
            assert np.array_equal(data["body_pos_w"][:, 14], body_pos[:, 19])
            assert np.array_equal(data["vel_cmd"], vel_cmd)
        """
    )
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-B",
            "-c",
            script,
            str(Path(__file__).resolve().parents[2]),
            str(Path(np.__file__).resolve().parents[1]),
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_registry_contains_only_the_eight_php_presets():
    assert set(experiment.DEFAULTS) == set(PHP_PRESETS)
    local_presets = {
        name
        for name, value in vars(experiment).items()
        if isinstance(value, ExperimentConfig) and name.startswith("g1_wbt")
    }
    assert local_presets == set(PHP_PRESETS)
    assert experiment.g1_29dof_wbt is g1_29dof_wbt


@pytest.mark.parametrize("name", PHP_PRESETS)
def test_cli_selects_presets_with_wandb_logging_by_default(name):
    cfg = tyro.cli(
        experiment.AnnotatedExperimentConfig,
        args=[f"exp:{name.replace('_', '-')}"],
        config=TYRO_CONIFG,
    )
    assert cfg == experiment.DEFAULTS[name]
    assert cfg.logger == wandb


@pytest.mark.parametrize("name", PHP_PRESETS)
def test_wandb_logging_can_be_disabled_explicitly(name):
    cfg = tyro.cli(
        experiment.AnnotatedExperimentConfig,
        args=[f"exp:{name.replace('_', '-')}", "logger:disabled"],
        config=TYRO_CONIFG,
    )
    assert cfg == replace(experiment.DEFAULTS[name], logger=disabled)


def test_serialized_presets_use_the_supported_php_checkpoint_paths():
    paths = {
        path for cfg in experiment.DEFAULTS.values() for path in _php_paths(asdict(cfg))
    }
    assert paths == set(PHP_CHECKPOINT_PATHS)


@pytest.mark.parametrize("path", PHP_CHECKPOINT_PATHS)
def test_supported_checkpoint_paths_resolve(path):
    # Use the same resolvers as manager terms and algorithm construction.
    resolved = resolve_callable(path) if ":" in path else get_class(path)
    assert callable(resolved)


def test_converter_joint_and_body_orderings_are_unchanged():
    assert robot_config.G1_29DOF_JOINT_NAMES == core_g1_robot.dof_names
    assert robot_config.G1_32BODY_NAMES == core_g1_robot.body_names
    np.testing.assert_array_equal(
        robot_config.ISAACLAB_TO_MUJOCO_DOF,
        [
            0,
            3,
            6,
            9,
            13,
            17,
            1,
            4,
            7,
            10,
            14,
            18,
            2,
            5,
            8,
            11,
            15,
            19,
            21,
            23,
            25,
            27,
            12,
            16,
            20,
            22,
            24,
            26,
            28,
        ],
    )
