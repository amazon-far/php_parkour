"""CPU checks for PHP's integration with the Holosoma compatibility branch."""

from types import SimpleNamespace

import pytest
import torch
from holosoma.config_types.algo import (
    DistillationConfig,
    DistillationPPOConfig,
    StudentTeacherModuleConfig,
)
from holosoma.config_types.observation import (
    ObservationManagerCfg,
    ObsGroupCfg,
    ObsTermCfg,
)
from holosoma.managers.observation.manager import ObservationManager
from holosoma.managers.utils import resolve_callable

from wbt_training.config_values.terrain import g1_29dof_wbt_terrain_termination
from wbt_training.utils.student_teacher import PhpDistillation, PhpDistillationPPO


def observation(env, name):
    return env.observations[name]


def build_algo(algo_cls):
    env = SimpleNamespace(
        observations={
            "which_motion": torch.tensor([[0.0], [1.0], [-1.0]]),
            "proprio": torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        },
    )
    # Insert routing first: the live manager must still sort it after proprio.
    terms = {
        name: ObsTermCfg(func=f"{__name__}:observation", params={"name": name})
        for name in env.observations
    }
    env.observation_manager = ObservationManager(
        ObservationManagerCfg(groups={"teacher_obs": ObsGroupCfg(terms=terms)}),
        env,
        "cpu",
    )
    module = StudentTeacherModuleConfig(
        student_hidden_dims=[8], teacher_hidden_dims=[8], critic_hidden_dims=[8]
    )
    cfg_cls = (
        DistillationPPOConfig if algo_cls is PhpDistillationPPO else DistillationConfig
    )
    algo = object.__new__(algo_cls)
    algo.config = cfg_cls(module=module, init_noise_std=0.0123)
    algo.env = env
    algo.device = "cpu"
    algo.actor_obs_dim = algo.critic_obs_dim = 2
    algo.teacher_obs_dim = 3
    algo.num_act = 1
    algo.num_teachers = 2
    algo._build_policy()
    return algo


@pytest.mark.parametrize("algo_cls", [PhpDistillation, PhpDistillationPPO])
def test_policy_build_preserves_noise_and_named_teacher_routing(algo_cls):
    algo = build_algo(algo_cls)
    policy = algo.policy
    for index, teacher in enumerate(policy.teachers):
        with torch.no_grad():
            for parameter in teacher.parameters():
                parameter.zero_()
            teacher[-1].bias.fill_(index + 1)
    policy.set_loaded_teachers([True, True])

    teacher_obs = algo.env.observation_manager.compute_group("teacher_obs")
    torch.testing.assert_close(
        policy.teacher_act(teacher_obs), torch.tensor([[1.0], [2.0], [0.0]])
    )
    torch.testing.assert_close(policy.std, torch.tensor([0.0123]))


def test_core_gated_update_uses_php_expert_lost_mask():
    algo = build_algo(PhpDistillationPPO)
    algo.config = DistillationPPOConfig(
        module=algo.config.module,
        value_loss_coef=0.0,
        entropy_coef=0.0,
        dagger_loss_coef=1.0,
        schedule="fixed",
    )
    algo.distill_loss_fn = torch.nn.functional.mse_loss
    algo.ppo_coef = 0.8
    algo._warmup_active = False
    algo._optimizer_step = lambda loss: loss.backward()
    with torch.no_grad():
        for parameter in algo.policy.student.parameters():
            parameter.zero_()
        algo.policy.student[-1].bias.fill_(0.5)

    actor_obs = torch.zeros(3, 2)
    depth_obs = torch.zeros(3, 58, 87)
    with torch.no_grad():
        actions = algo.policy.act({"actor_obs": actor_obs, "depth_obs": depth_obs})
        old_log_prob = algo.policy.get_actions_log_prob(actions).unsqueeze(-1)
        old_mu = algo.policy.action_mean.clone()
        old_sigma = algo.policy.action_std.clone()

    losses = algo._update_step(
        {
            "actor_obs": actor_obs,
            "depth_obs": depth_obs,
            "critic_obs": actor_obs,
            "actions": actions,
            "teacher_actions": torch.tensor([[1.0], [2.0], [0.0]]),
            "values": torch.zeros(3, 1),
            "returns": torch.zeros(3, 1),
            "advantages": torch.zeros(3, 1),
            "actions_log_prob": old_log_prob,
            "action_mean": old_mu,
            "action_sigma": old_sigma,
            "ppo_gate": torch.tensor([[1.0], [0.5], [1.0]]),
        }
    )

    # The first two samples have imitation weights 0.2 and 0.6; the lost
    # expert contributes no imitation loss, despite the student's nonzero action.
    torch.testing.assert_close(losses["total_loss"], torch.tensor(1.4 / 3.0))
    torch.testing.assert_close(
        algo.policy.student[-1].bias.grad, torch.tensor([-2.0 / 3.0])
    )


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_terrain_termination_detects_end_effector_error_on_every_axis(axis):
    cfg = g1_29dof_wbt_terrain_termination.terms["bad_tracking"]
    term = resolve_callable(cfg.func)(cfg, SimpleNamespace(device="cpu"))
    positions = torch.zeros(1, len(cfg.params["body_names_to_track"]), 3)
    positions[0, term.bad_motion_body_pos_body_indexes[0], axis] = 0.6
    command = SimpleNamespace(
        body_pos_relative_w=positions,
        robot_body_pos_w=torch.zeros_like(positions),
    )

    assert term.bad_motion_body_pos(command).item()
