"""CPU ONNX execution checks for the existing PHP depth/student interface."""

import dataclasses
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from holosoma.agents.modules.depth_backbone import DepthOnlyFCBackbone58x87Small
from holosoma.config_types.observation import (
    ObservationManagerCfg,
    ObsGroupCfg,
    ObsTermCfg,
)
from holosoma.managers.observation.manager import ObservationManager

from wbt_training.config_values.experiment import g1_wbt_terrain_warp_distill_finetune
from wbt_training.training_runs import export_distill_onnx
from wbt_training.utils import exporter

REFERENCE_NAMES = [
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
]


def observation(env, name):
    return env.observations[name]


def export_fixture():
    motion = SimpleNamespace(motion_ends=torch.tensor([False, True, False, True]))
    for index, name in enumerate(REFERENCE_NAMES):
        shape = (
            (4, 2) if name.startswith("joint_") else (4, 1, 4 if "quat" in name else 3)
        )
        setattr(
            motion,
            name,
            torch.arange(np.prod(shape), dtype=torch.float32).reshape(shape)
            + 100 * index,
        )
    command = SimpleNamespace(
        motion=motion,
        motion_cfg=SimpleNamespace(
            body_name_ref=["pelvis"], body_names_to_track=["pelvis"]
        ),
    )
    action = SimpleNamespace(
        action_scales=torch.tensor([0.2, 0.3]),
        p_gains=torch.tensor([20.0, 30.0]),
        d_gains=torch.tensor([1.0, 2.0]),
    )
    env = SimpleNamespace(
        command_manager=SimpleNamespace(
            get_state=lambda _: command,
            cfg=SimpleNamespace(setup_terms={"motion_command": None}),
        ),
        action_manager=SimpleNamespace(get_term=lambda _: action),
        robot_config=SimpleNamespace(dof_names=["left_joint", "right_joint"]),
        default_dof_pos_base=torch.tensor([0.1, -0.1]),
        observations={
            "z_velocity": torch.tensor([[3.0]]),
            "a_proprio": torch.tensor([[1.0, 2.0]]),
        },
    )
    env.observation_manager = ObservationManager(
        ObservationManagerCfg(
            groups={
                "actor_obs": ObsGroupCfg(
                    terms={
                        name: ObsTermCfg(
                            func=f"{__name__}:observation", params={"name": name}
                        )
                        for name in env.observations
                    }
                )
            }
        ),
        env,
        "cpu",
    )
    policy = SimpleNamespace(
        student=torch.nn.Sequential(
            torch.nn.Linear(35, 8), torch.nn.ELU(), torch.nn.Linear(8, 2)
        ),
        depth_backbone=DepthOnlyFCBackbone58x87Small(32),
    )
    return env, policy, motion


def test_depth_bundle_contract_metadata_and_onnx_numerics(tmp_path):
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    env, policy, motion = export_fixture()
    original = {k: v.clone() for k, v in policy.depth_backbone.state_dict().items()}
    backbone_path, student_path = exporter.export_depth_student_components(
        env, policy, str(tmp_path), (58, 87)
    )
    exporter.attach_onnx_metadata(
        student_path, exporter.build_metadata_from_wbt_env(env)
    )
    for path in (backbone_path, student_path):
        onnx.checker.check_model(onnx.load(path))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    backbone = ort.InferenceSession(
        backbone_path, options, providers=["CPUExecutionProvider"]
    )
    student = ort.InferenceSession(
        student_path, options, providers=["CPUExecutionProvider"]
    )
    assert [(item.name, item.shape) for item in backbone.get_inputs()] == [
        ("depth_image", [1, 58, 87])
    ]
    assert [(item.name, item.shape) for item in backbone.get_outputs()] == [
        ("depth_latent", [1, 32])
    ]
    assert [(item.name, item.shape) for item in student.get_inputs()] == [
        ("obs", [1, 35]),
        ("time_step", [1, 1]),
    ]
    assert [item.name for item in student.get_outputs()] == [
        "actions",
        *REFERENCE_NAMES,
    ]
    assert [item.shape for item in student.get_outputs()] == [
        [1, 2],
        [1, 2],
        [1, 2],
        [1, 1, 3],
        [1, 1, 4],
        [1, 1, 3],
        [1, 1, 3],
    ]
    assert all(
        item.type == "tensor(float)"
        for item in [*student.get_inputs(), *student.get_outputs()]
    )
    depth = torch.randn(1, 58, 87)
    with torch.no_grad():
        expected_latent = policy.depth_backbone(depth)
    latent = backbone.run(None, {"depth_image": depth.numpy()})[0]
    np.testing.assert_allclose(latent, expected_latent.numpy(), rtol=1e-5, atol=1e-6)
    actor_obs = env.observation_manager.compute_group("actor_obs")
    torch.testing.assert_close(actor_obs, torch.tensor([[1.0, 2.0, 3.0]]))
    obs = torch.cat([actor_obs, expected_latent], dim=-1)
    with torch.no_grad():
        expected_actions = policy.student(obs).numpy()
    for time_step in (0, 1, 99):
        result = student.run(
            None,
            {
                "obs": obs.numpy(),
                "time_step": np.array([[time_step]], dtype=np.float32),
            },
        )
        np.testing.assert_allclose(result[0], expected_actions, rtol=1e-5, atol=1e-6)
        for name, output in zip(REFERENCE_NAMES, result[1:], strict=True):
            # The first reference clip ends at frame 1; later indices clamp there.
            np.testing.assert_array_equal(
                output, getattr(motion, name)[[min(time_step, 1)]].numpy()
            )
    metadata = student.get_modelmeta().custom_metadata_map
    assert metadata["observation_names"] == "a_proprio,z_velocity,placeholder"
    assert metadata["joint_names"] == "left_joint,right_joint"
    assert metadata["kp"] == metadata["joint_stiffness"] == "20.000,30.000"
    assert metadata["anchor_body_name"] == "pelvis"
    assert policy.depth_backbone.training
    assert policy.student.training
    for key, value in original.items():
        torch.testing.assert_close(policy.depth_backbone.state_dict()[key], value)


def test_failed_export_preserves_live_backbone_state(tmp_path, monkeypatch):
    _, policy, _ = export_fixture()
    policy.depth_backbone.image_compression[0].eval()  # mixed child modes also survive
    modes = [module.training for module in policy.depth_backbone.modules()]
    parameters = list(policy.depth_backbone.parameters())

    def fail(*_, **__):
        raise RuntimeError("export failed")

    monkeypatch.setattr(torch.onnx, "export", fail)
    with pytest.raises(RuntimeError, match="export failed"):
        exporter.export_depth_backbone_as_onnx(policy, str(tmp_path), (58, 87))
    assert [module.training for module in policy.depth_backbone.modules()] == modes
    assert all(
        before is after
        for before, after in zip(
            parameters, policy.depth_backbone.parameters(), strict=True
        )
    )


def test_export_infers_teacher_slots_before_setup_and_uses_full_loader(
    tmp_path, monkeypatch
):
    from holosoma.utils import experiment_paths, helpers

    path = tmp_path / "model_00023.pt"
    torch.save(
        {
            "model_state_dict": {
                "teachers.0.weight": torch.ones(1),
                "teachers.1.weight": torch.ones(1),
            },
            "iter": 23,
        },
        path,
    )
    preset = g1_wbt_terrain_warp_distill_finetune
    config = dataclasses.replace(
        preset, training=dataclasses.replace(preset.training, checkpoint=str(path))
    )
    events = []

    class Algo:
        def __init__(self, **kwargs):
            self.num_teachers = 1
            self.log_dir = kwargs["log_dir"]

        def setup(self):
            assert self.num_teachers == 2
            events.append("setup")

        def attach_checkpoint_metadata(self, saved, run):
            assert saved is preset
            events.append("metadata")

        def load(self, checkpoint_path):
            assert checkpoint_path == str(path)
            events.append("full_load")
            self.current_learning_iteration = 23

    def preprocess(cfg):
        events.append("preprocess")
        return cfg

    def env_class(*_, **__):
        events.append("env")
        return SimpleNamespace()

    def export(algo, iteration):
        assert iteration == 23
        events.append("export")
        return str(Path(algo.log_dir) / "model_00023")

    monkeypatch.setattr(export_distill_onnx, "init_sim_imports", lambda _: None)
    monkeypatch.setattr(
        export_distill_onnx, "close_simulation_app", lambda _: events.append("close")
    )
    monkeypatch.setattr(export_distill_onnx, "apply_preprocess_hook", preprocess)
    monkeypatch.setattr(export_distill_onnx, "export_depth_student_bundle", export)
    monkeypatch.setattr(
        experiment_paths,
        "get_experiment_dir",
        lambda *_args, **_kwargs: tmp_path / "export",
    )
    monkeypatch.setattr(
        helpers,
        "get_class",
        lambda name: env_class if name == config.env_class else Algo,
    )
    result = export_distill_onnx._export(config, preset)
    assert result == str(tmp_path / "export/model_00023")
    assert events == [
        "preprocess",
        "env",
        "setup",
        "metadata",
        "full_load",
        "export",
        "close",
    ]
