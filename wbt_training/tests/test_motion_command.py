"""CPU regressions for PHP expert sampling against Holosoma motion boundaries."""

from types import SimpleNamespace

import pytest
import torch

from holosoma.managers.command.terms.wbt import MotionCommand
from wbt_training.config_values.depth_distillation import wbt_distill_motion_config
from wbt_training.config_values.motion_command import PhpMotionCommand


class ExpertCommand(SimpleNamespace):
    _build_expert_pools = PhpMotionCommand._build_expert_pools
    _sample_expert_balanced = PhpMotionCommand._sample_expert_balanced


def make_command(num_experts=5, num_envs=4096):
    # Unequal, interleaved expert pools in ONE combined NPZ loader.
    frame_experts = torch.cat([torch.zeros(80, dtype=torch.long), torch.arange(num_experts).repeat(4)])
    total = len(frame_experts)
    identity = torch.tensor([0.0, 0.0, 0.0, 1.0])
    sim = SimpleNamespace(
        robot_root_states=torch.zeros(num_envs, 13),
        dof_state=torch.zeros(num_envs, 1, 2),
        set_actor_root_state_tensor_robots=lambda *args: None,
        set_dof_state_tensor_robots=lambda *args: None,
        refresh_sim_tensors=lambda: None,
    )
    command = ExpertCommand(
        device="cpu",
        num_envs=num_envs,
        motion=SimpleNamespace(
            num_motions=1,
            time_step_total=total,
            motion_idxs=frame_experts,
            motion_start_idx=torch.tensor([0]),
            motion_end_idx=torch.tensor([total]),
            motion_ends=torch.arange(total) == total - 1,
        ),
        motion_cfg=SimpleNamespace(
            freeze_at_timestep_zero_prob=wbt_distill_motion_config.freeze_at_timestep_zero_prob,
            resample_on_motion_end=wbt_distill_motion_config.resample_on_motion_end,
            use_adaptive_timesteps_sampler=False,
            body_names_to_track=["pelvis"],
        ),
        _env=SimpleNamespace(
            episode_length_buf=torch.ones(num_envs, dtype=torch.long),
            simulator=sim,
            is_evaluating=False,
        ),
        _expert_frame_pools=None,
        _expert_pool_lengths=None,
        _num_experts=None,
        time_steps=torch.zeros(num_envs, dtype=torch.long),
        motion_ids=torch.zeros(num_envs, dtype=torch.long),
        motion_end_reset=torch.zeros(num_envs, dtype=torch.bool),
        body_pos_w=torch.zeros(num_envs, 1, 3),
        body_quat_w=identity.repeat(num_envs, 1, 1),
    )
    for name in ["root", "ref", "robot_root", "robot_ref"]:
        setattr(command, name + "_pos_w", torch.zeros(num_envs, 3))
        setattr(command, name + "_quat_w", identity.repeat(num_envs, 1))
    command.reset = lambda env_ids: command.time_steps.__setitem__(env_ids, 0)
    return command


@pytest.mark.parametrize("num_experts", [1, 5])
def test_expert_sampling_can_step_single_loader_motion(num_experts):
    torch.manual_seed(42)
    cmd = make_command(num_experts=num_experts)
    env_ids = torch.arange(cmd.num_envs)
    cmd._sample_expert_balanced(env_ids, cmd.num_envs)
    # Teacher routing stays per frame and must still include every requested expert.
    teachers = cmd.motion.motion_idxs[cmd.time_steps]
    assert set(teachers.tolist()) == set(range(num_experts))
    if num_experts == 5:
        counts = torch.bincount(teachers, minlength=5)
        assert int(counts.min()) > 600  # uniform experts, despite expert zero owning most frames
    # Preserve the legacy expert IDs and exercise the intended clip-marker path.
    torch.testing.assert_close(cmd.motion_ids, teachers)
    MotionCommand.step(cmd)
    assert torch.all(cmd.time_steps < cmd.motion.time_step_total)
    assert torch.isfinite(cmd.body_pos_relative_w).all()


def test_partial_sampling_preserves_other_environments():
    cmd = make_command(num_envs=12)
    untouched = torch.tensor([0, 2, 4, 6, 8, 10])
    selected = torch.tensor([1, 3, 5, 7, 9, 11])
    cmd.time_steps.fill_(7)
    cmd.motion_ids.fill_(123)
    cmd._sample_expert_balanced(selected, len(selected))
    torch.testing.assert_close(
        cmd.motion_ids[selected], cmd.motion.motion_idxs[cmd.time_steps[selected]]
    )
    assert torch.all(cmd.motion_ids[untouched] == 123)
    assert torch.all(cmd.time_steps[untouched] == 7)


def test_distillation_explicitly_keeps_legacy_clip_resampling():
    assert wbt_distill_motion_config.resample_on_motion_end is True
    assert wbt_distill_motion_config.freeze_at_timestep_zero_prob == 0.0


def test_internal_clip_ends_reset_before_combined_file_end():
    cmd = make_command(num_envs=4)
    cmd.motion.motion_ends = torch.arange(cmd.motion.time_step_total) % 10 == 9
    cmd.time_steps[:] = torch.tensor([8, 18, 20, 98])
    cmd.motion_ids[:] = torch.tensor([0, 1, 2, 4])
    MotionCommand.step(cmd)
    assert cmd.motion_end_reset.tolist() == [True, True, False, True]
    assert cmd.time_steps.tolist() == [0, 0, 21, 0]
