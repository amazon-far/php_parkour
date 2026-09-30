"""Saved PHP configs, strict student eval, and full training resume on CPU."""

import copy
import sys
from types import SimpleNamespace

import pytest
import torch
import yaml
from holosoma.utils.config_utils import CONFIG_NAME
from holosoma.utils.eval_utils import CheckpointConfig

from wbt_training.config_values.experiment import g1_wbt_terrain_warp_distill_finetune
from wbt_training.training_runs import eval_student, eval_teacher, export_distill_onnx
from wbt_training.utils import checkpoint, exporter
from wbt_training.utils.compat import rewrite_legacy_paths
from wbt_training.utils.student_teacher import (
    PhpDistillation,
    PhpDistillationPPO,
    RoutingColumnDepthStudentTeacher,
    RoutingColumnDepthStudentTeacherCritic,
)


@pytest.fixture
def legacy_config():
    config = g1_wbt_terrain_warp_distill_finetune.to_serializable_dict()
    config["algo"]["config"]["module"]["student_hidden_dims"] = [17, 9]
    config["algo"]["_target_"] = (
        "holosoma.agents.distillation_ppo.distillation_ppo.DistillationPPO"
    )
    config["command"]["setup_terms"]["motion_command"]["func"] = (
        "holosoma.managers.command.terms.wbt:MotionCommand"
    )
    config["observation"]["groups"]["actor_obs"]["terms"]["velocity_command"][
        "func"
    ] = "holosoma.managers.observation.terms.wbt:velocity_command"
    config["algo"]["config"]["module"]["depth_backbone"] = (
        "wbt_training.utils.depth_backbone.DepthOnlyFCBackbone58x87Small"
    )
    return config


def assert_php_config(config):
    assert (
        config.algo._target_ == "wbt_training.utils.student_teacher.PhpDistillationPPO"
    )
    assert config.algo.config.module.student_hidden_dims == [17, 9]
    assert config.command.setup_terms["motion_command"].func.endswith(
        ":PhpMotionCommand"
    )
    assert config.algo.config.module.depth_backbone.startswith(
        "holosoma.agents.modules."
    )
    assert config.observation.groups["actor_obs"].terms["velocity_command"].func == (
        "wbt_training.config_values.motion_command:velocity_command"
    )


def test_local_pt_precedes_yaml_and_is_shared_by_eval(
    tmp_path, monkeypatch, legacy_config
):
    path = tmp_path / "model_12345.pt"
    torch.save(
        {"experiment_config": legacy_config, "wandb_run_path": "entity/project/run"},
        path,
    )
    (tmp_path / CONFIG_NAME).write_text("not: the saved config\n")
    monkeypatch.setattr(
        checkpoint,
        "get_cached_file_path",
        lambda *_: pytest.fail("Local config used network cache"),
    )
    config, run = checkpoint.load_saved_experiment_config(
        CheckpointConfig(checkpoint=str(path))
    )
    assert_php_config(config)
    assert run == "entity/project/run"
    assert (
        eval_student._load_saved_experiment_config_prefer_pt
        is checkpoint.load_saved_experiment_config
    )
    assert (
        eval_teacher._load_saved_experiment_config_prefer_pt
        is checkpoint.load_saved_experiment_config
    )
    assert torch.load(path, weights_only=False)["experiment_config"] == legacy_config


def test_yaml_fallback_also_rewrites_before_construction(tmp_path, legacy_config):
    path = tmp_path / "model_legacy.pt"
    torch.save({"model_state_dict": {}}, path)
    (tmp_path / CONFIG_NAME).write_text(yaml.safe_dump(legacy_config))
    config, _ = checkpoint.load_saved_experiment_config(
        CheckpointConfig(checkpoint=str(path))
    )
    assert_php_config(config)


def test_invalid_embedded_config_does_not_fall_back(tmp_path, legacy_config):
    path = tmp_path / "model_invalid.pt"
    torch.save({"experiment_config": []}, path)
    (tmp_path / CONFIG_NAME).write_text(yaml.safe_dump(legacy_config))
    with pytest.raises(TypeError, match="experiment_config"):
        checkpoint.load_saved_experiment_config(CheckpointConfig(checkpoint=str(path)))


@pytest.mark.parametrize("option", ["--checkpoint", "--training.checkpoint"])
def test_export_uses_saved_architecture_then_explicit_overrides(
    tmp_path, legacy_config, option
):
    path = tmp_path / "model_12345.pt"
    torch.save({"experiment_config": legacy_config}, path)
    config, saved, _ = export_distill_onnx._parse_export_config(
        [
            "exp:g1-wbt-terrain-warp-distill-finetune",
            option,
            str(path),
            "--training.num-envs",
            "2",
            "--training.registry-name",
            "file:///local/bundle",
            "--algo.config.module.student-hidden-dims",
            "[11, 7]",
        ]
    )
    assert_php_config(saved)
    assert config.algo.config.module.student_hidden_dims == [11, 7]
    assert config.training.num_envs == 2
    assert config.training.registry_name == "file:///local/bundle"
    assert config.training.checkpoint == str(path)
    assert config.training.headless
    assert saved.training.registry_name == legacy_config["training"]["registry_name"]


def test_latest_wandb_reference_is_pinned_once(tmp_path, monkeypatch, legacy_config):
    from holosoma.utils import wandb_registry

    path = tmp_path / "model_12345.pt"
    torch.save({"experiment_config": legacy_config}, path)
    resolutions = []
    reads = []

    def latest(run):
        resolutions.append(run)
        return None, run, "model_12345.pt"

    def cached(reference):
        reads.append(reference)
        return path

    monkeypatch.setattr(wandb_registry, "get_wandb_run_and_file", latest)
    monkeypatch.setattr(checkpoint, "get_cached_file_path", cached)
    config, _, _ = export_distill_onnx._parse_export_config(
        ["--checkpoint", "wandb://entity/project/run"]
    )
    assert resolutions == ["entity/project/run"]
    assert reads == ["wandb://entity/project/run/model_12345.pt"]
    assert config.training.checkpoint == reads[0]


def test_compat_rewrites_only_path_fields_and_upgrades_command_with_evidence():
    old = "holosoma.managers.command.terms.wbt:MotionCommand"
    plain = {
        "func": old,
        "description": "holosoma.managers.observation.terms.wbt:velocity_command",
    }
    rewritten, changes = rewrite_legacy_paths(copy.deepcopy(plain))
    assert rewritten == plain
    assert changes == []
    plain["nested"] = [
        {"func": "holosoma.managers.observation.terms.wbt:velocity_command"}
    ]
    rewritten, _ = rewrite_legacy_paths(copy.deepcopy(plain))
    assert rewritten["func"].endswith(":PhpMotionCommand")
    assert rewritten["description"] == plain["description"]


def student_algo():
    restored = []
    return SimpleNamespace(
        device="cpu",
        policy=SimpleNamespace(
            student=torch.nn.Linear(3, 2), depth_backbone=torch.nn.Linear(4, 1)
        ),
        _restore_env_state=restored.append,
        restored=restored,
    )


def student_state(algo):
    state = {
        f"student.{k}": v.clone() for k, v in algo.policy.student.state_dict().items()
    }
    state.update(
        {
            f"depth_backbone.{k}": v.clone()
            for k, v in algo.policy.depth_backbone.state_dict().items()
        }
    )
    state.update(
        {
            "teachers.4.weight": torch.ones(8),
            "critic.weight": torch.ones(8),
            "std": torch.ones(2),
        }
    )
    return state


def test_student_only_eval_loads_backbone_strictly_and_restores_env(tmp_path):
    source, target = student_algo(), student_algo()
    path = tmp_path / "student.pt"
    torch.save(
        {
            "model_state_dict": student_state(source),
            "iter": 23,
            "env_state": {"gate": 0.5},
        },
        path,
    )
    eval_student._load_student_only(target, str(path))
    for module in ("student", "depth_backbone"):
        for key, expected in getattr(source.policy, module).state_dict().items():
            torch.testing.assert_close(
                getattr(target.policy, module).state_dict()[key], expected
            )
    assert target.current_learning_iteration == 23
    assert target.restored == [{"gate": 0.5}]


@pytest.mark.parametrize(
    "damage", ["student_missing", "backbone_missing", "backbone_shape", "unknown"]
)
def test_student_only_eval_rejects_incomplete_or_mismatched_weights(tmp_path, damage):
    algo = student_algo()
    state = student_state(algo)
    if damage == "student_missing":
        del state["student.bias"]
    elif damage == "backbone_missing":
        state = {k: v for k, v in state.items() if not k.startswith("depth_backbone.")}
    elif damage == "backbone_shape":
        state["depth_backbone.weight"] = torch.zeros(2, 4)
    else:
        state["unhandled.weight"] = torch.zeros(1)
    path = tmp_path / "bad.pt"
    torch.save({"model_state_dict": state}, path)
    with pytest.raises(RuntimeError):
        eval_student._load_student_only(algo, str(path))


def resume_algo(algo_cls):
    algo = object.__new__(algo_cls)
    ppo = algo_cls is PhpDistillationPPO
    policy_cls = (
        RoutingColumnDepthStudentTeacherCritic
        if ppo
        else RoutingColumnDepthStudentTeacher
    )
    kwargs = {"num_critic_obs": 2, "critic_hidden_dims": [4]} if ppo else {}
    algo.policy = policy_cls(
        num_actor_obs=2,
        num_teacher_obs=3,
        num_actions=1,
        depth_backbone=torch.nn.Linear(2, 1),
        depth_output_dim=1,
        student_hidden_dims=[4],
        teacher_hidden_dims=[4],
        num_teachers=2,
        routing_column=2,
        **kwargs,
    )
    algo.device = "cpu"
    algo.optimizer = torch.optim.Adam(algo.policy.parameters(), lr=0.002)
    algo.current_learning_iteration = 23
    algo.learning_rate = 0.002
    algo.ppo_coef = 0.7
    algo.restored = []
    algo._restore_env_state = algo.restored.append
    return algo


@pytest.mark.parametrize("algo_cls", [PhpDistillation, PhpDistillationPPO])
def test_full_resume_and_export_save_hook_keep_training_state(
    tmp_path, monkeypatch, algo_cls
):
    source, target = resume_algo(algo_cls), resume_algo(algo_cls)
    target.current_learning_iteration = 0
    target.learning_rate = 0.1
    target.ppo_coef = 0.0
    target.optimizer.param_groups[0]["lr"] = 0.1
    source.policy.set_loaded_teachers([True, False])
    loss = sum(
        parameter.square().sum()
        for parameter in source.policy.parameters()
        if parameter.requires_grad
    )
    loss.backward()
    source.optimizer.step()
    source._checkpoint_metadata = lambda **_: {"experiment_config": {"saved": True}}
    source._collect_env_state = lambda: {"php_motion_gate": torch.tensor([0.2, 0.7])}
    source.logging_helper = SimpleNamespace(save_checkpoint_artifact=torch.save)
    path = tmp_path / "model_00023.pt"
    exports = []

    def export_saved(algo, iteration):
        assert path.is_file()
        exports.append((algo, iteration))

    monkeypatch.setattr(exporter, "export_depth_student_bundle", export_saved)
    source.save(str(path))
    assert exports == [(source, 23)]
    target.load(str(path))
    for key, expected in source.policy.state_dict().items():
        torch.testing.assert_close(target.policy.state_dict()[key], expected)
    assert target.policy.loaded_teachers == (True, False)
    assert target.current_learning_iteration == 23
    assert target.learning_rate == 0.002
    assert target.optimizer.param_groups[0]["lr"] == 0.002
    assert target.optimizer.state
    for parameter, expected in zip(
        target.optimizer.state.values(), source.optimizer.state.values(), strict=True
    ):
        for key in expected:
            torch.testing.assert_close(parameter[key], expected[key])
    torch.testing.assert_close(
        target.restored[0]["php_motion_gate"], torch.tensor([0.2, 0.7])
    )
    if algo_cls is PhpDistillationPPO:
        assert target.ppo_coef == 0.7
    assert all(
        not parameter.requires_grad for parameter in target.policy.teachers.parameters()
    )


@pytest.mark.parametrize("show_depth", [False, True])
def test_student_eval_main_preserves_motion_config_and_depth_options(
    tmp_path, monkeypatch, legacy_config, show_depth
):
    from holosoma.managers.command.terms.wbt import MotionCommand

    from wbt_training.config_values.motion_command import PhpMotionCommand

    path = tmp_path / "student.pt"
    torch.save({"experiment_config": legacy_config}, path)
    original_step = MotionCommand.step
    original_velocity = PhpMotionCommand.vel_cmd
    calls = []
    monkeypatch.setattr(eval_student, "init_eval_logging", lambda: None)
    monkeypatch.setattr(
        eval_student, "run_eval", lambda *args, **kwargs: calls.append((args, kwargs))
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_student",
            "--checkpoint",
            str(path),
            "--training.num-envs",
            "2",
            "--training.max-eval-steps",
            "7",
            "--show-depth",
            str(show_depth),
        ],
    )

    eval_student.main()

    assert len(calls) == 1
    (config, checkpoint_cfg, saved, _), options = calls[0]
    assert_php_config(saved)
    assert checkpoint_cfg.checkpoint == str(path)
    assert config.training.num_envs == 2
    assert config.training.max_eval_steps == 7
    assert config.command == saved.command
    assert config.termination == saved.get_eval_config().termination
    assert set(options) == {"eval_cbs_cfg", "depth_viz_cfg"}
    assert options["depth_viz_cfg"].show_depth is show_depth
    assert MotionCommand.step is original_step
    assert PhpMotionCommand.vel_cmd is original_velocity


@pytest.mark.parametrize("legacy_target", [False, True])
@pytest.mark.parametrize("source", ["pt", "yaml"])
def test_pure_dagger_recovers_actual_algorithm_type_and_exact_recipe(
    tmp_path, legacy_target, source
):
    from holosoma.config_types.algo import DistillationAlgoConfig, DistillationConfig

    from wbt_training.config_values.experiment import g1_wbt_terrain_warp_distill

    raw = g1_wbt_terrain_warp_distill.to_serializable_dict()
    # Use non-default saved values so accidental reconstruction from today's
    # preset or the overlapping PPO dataclass cannot pass unnoticed.
    recipe = raw["algo"]["config"]
    recipe.update(
        learning_rate=0.0017,
        num_learning_epochs=7,
        num_learning_iterations=4321,
        init_noise_std=0.0042,
        distillation_warmup_steps=31,
        distill_loss_type="huber",
    )
    recipe["module"]["student_hidden_dims"] = [19, 11]
    raw["algo"]["_target_"] = (
        "holosoma.agents.distillation.distillation.Distillation"
        if legacy_target
        else "wbt_training.utils.student_teacher.PhpDistillation"
    )
    path = tmp_path / "model_04321.pt"
    torch.save(
        {"experiment_config": raw} if source == "pt" else {"model_state_dict": {}}, path
    )
    if source == "yaml":
        (tmp_path / CONFIG_NAME).write_text(yaml.safe_dump(raw))
    config, saved, _ = export_distill_onnx._parse_export_config(
        ["--checkpoint", str(path)]
    )
    for recovered in (config, saved):
        assert type(recovered.algo) is DistillationAlgoConfig
        assert type(recovered.algo.config) is DistillationConfig
        assert (
            recovered.algo._target_
            == "wbt_training.utils.student_teacher.PhpDistillation"
        )
        assert recovered.to_serializable_dict()["algo"]["config"] == recipe
    overridden = checkpoint.parse_saved_config_overrides(
        config, ["--algo.config.learning-rate", "0.0008"]
    )
    assert overridden.algo.config.learning_rate == 0.0008
    assert overridden.algo.config.init_noise_std == recipe["init_noise_std"]
    assert config.algo.config.learning_rate == recipe["learning_rate"]
