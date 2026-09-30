"""PHP terrain teacher and depth-distillation experiment presets.

The teacher derives from Holosoma's G1 whole-body-tracking experiment;
terrain variants and students preserve their existing replace chains.
W&B logging is enabled by default; use ``logger:disabled`` for local-only output.
"""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Annotated

import tyro
from holosoma.config_types.algo import (
    DistillationAlgoConfig,
    DistillationConfig,
    DistillationPPOAlgoConfig,
    DistillationPPOConfig,
    OptimizerConfig,
    StudentTeacherModuleConfig,
)
from holosoma.config_types.experiment import ExperimentConfig
from holosoma.config_values import simulator
from holosoma.config_values.logger import wandb as wandb_logger
from holosoma.config_values.wbt.g1.experiment import g1_29dof_wbt

from wbt_training.config_values.depth_distillation import (
    g1_29dof_wbt_terrain_distill_command,
    g1_29dof_wbt_terrain_distill_curriculum,
    g1_29dof_wbt_terrain_distill_observation,
)
from wbt_training.config_values.terrain import (
    g1_29dof_joint_pos_implicit,
    g1_29dof_wbt_terrain_command,
    g1_29dof_wbt_terrain_command_pelvis,
    g1_29dof_wbt_terrain_observation,
    g1_29dof_wbt_terrain_observation_no_heightscan,
    g1_29dof_wbt_terrain_randomization,
    g1_29dof_wbt_terrain_reward,
    g1_29dof_wbt_terrain_termination,
)

# ---------------------------------------------------------------------------
# Terrain presets. Start from the stock `g1_29dof_wbt` base (ref-matching);
# terrain-specific command/observation/reward/termination presets live in
# `wbt_training.config_values.terrain`.
# ---------------------------------------------------------------------------
_ref_adamw_no_wd = OptimizerConfig(_target_="torch.optim.AdamW", weight_decay=0.0)

g1_wbt_terrain_ref = replace(
    g1_29dof_wbt,
    training=replace(
        g1_29dof_wbt.training,
        project="WBT-Holosoma",
        name="g1_29dof_wbt_terrain_manager",
        num_envs=4096,
        # Application-specific preprocess: resolves training.registry_name via
        # wandb (or file://), samples per-env on-path obstacles, splices the
        # combined terrain OBJ + motion NPZ into the config before env build.
        preprocess_hook="wbt_training.config_values.terrain:apply_terrain_preprocess",
        preprocess_hook_kwargs=json.dumps(
            {
                "add_onpath_obstacle": True,
                # Many envs share each of the 64 obstacle layouts.
                "num_variants": 64,
            }
        ),
    ),
    logger=wandb_logger,
    simulator=replace(
        simulator.isaacsim,
        config=replace(
            simulator.isaacsim.config,
            sim=replace(simulator.isaacsim.config.sim, max_episode_length_s=10.0),
        ),
    ),
    # `scene` was promoted from `simulator.config.scene` to a top-level
    # ExperimentConfig field in holosoma (scene/sensor/plugin extraction).
    scene=replace(g1_29dof_wbt.scene, env_spacing=0.0),
    # g1_29dof_wbt base already has actor/critic hidden_dims=[512,256,128] — no module override needed.
    algo=replace(
        g1_29dof_wbt.algo,
        config=replace(
            g1_29dof_wbt.algo.config,
            actor_learning_rate=1e-3,
            critic_learning_rate=1e-3,
            num_learning_epochs=5,
            num_learning_iterations=20000,
            save_interval=500,
            entropy_coef=0.005,
            init_noise_std=1.0,
            init_at_random_ep_len=False,
            use_symmetry=False,
            empirical_normalization=False,
            actor_optimizer=_ref_adamw_no_wd,
            critic_optimizer=_ref_adamw_no_wd,
        ),
    ),
    robot=replace(
        g1_29dof_wbt.robot,
        control=replace(
            g1_29dof_wbt.robot.control,
            action_scale=0.25,
            action_scales_by_effort_limit_over_p_gain=True,
            control_mode="implicit_position_target",
            implicit_actuator_builder="wbt_training.config_values.robot_config:build_g1_implicit_actuators",
        ),
        asset=replace(
            g1_29dof_wbt.robot.asset,
            urdf_file="g1/main_mesh_arm_collision_halfspherehand.urdf",
        ),
    ),
    # Ref routes position targets to IsaacSim's built-in PD drive via
    # JointPositionTargetActionTerm (not the explicit-PD JointPositionActionTerm
    # inherited from g1_29dof_wbt).
    action=g1_29dof_joint_pos_implicit,
    observation=g1_29dof_wbt_terrain_observation,
    termination=g1_29dof_wbt_terrain_termination,
    randomization=g1_29dof_wbt_terrain_randomization,
    command=g1_29dof_wbt_terrain_command,
    reward=g1_29dof_wbt_terrain_reward,
)

# Terrain preset without the height_scan oracle: actor/critic drop those 289
# ray readings. Same obstacle injection + motion set as g1_wbt_terrain_ref;
# bumped to 60k iterations since training has to solve terrain traversal
# without the heightmap sensor.
g1_wbt_terrain_no_heightscan = replace(
    g1_wbt_terrain_ref,
    observation=g1_29dof_wbt_terrain_observation_no_heightscan,
    algo=replace(
        g1_wbt_terrain_ref.algo,
        config=replace(g1_wbt_terrain_ref.algo.config, num_learning_iterations=60000),
    ),
)

# Ref-intent variant: train with effective unit scaling (action_scale=1.0,
# action_scales_by_effort_limit_over_p_gain=False). The original FAR-Holosoma
# terrain preset set these same values on the *action term* (not robot.control);
# our JointPositionTargetActionTerm now applies control.action_scale, so to
# match the ref math we set the control-level knobs to unit here.
g1_wbt_terrain_ref_noscale = replace(
    g1_wbt_terrain_ref,
    robot=replace(
        g1_wbt_terrain_ref.robot,
        control=replace(
            g1_wbt_terrain_ref.robot.control,
            action_scale=1.0,
            action_scales_by_effort_limit_over_p_gain=False,
        ),
    ),
)

g1_wbt_terrain_no_heightscan_noscale = replace(
    g1_wbt_terrain_no_heightscan,
    robot=replace(
        g1_wbt_terrain_no_heightscan.robot,
        control=replace(
            g1_wbt_terrain_no_heightscan.robot.control,
            action_scale=1.0,
            action_scales_by_effort_limit_over_p_gain=False,
        ),
    ),
)

g1_wbt_terrain_ref_pelvis = replace(
    g1_wbt_terrain_ref, command=g1_29dof_wbt_terrain_command_pelvis
)

g1_wbt_terrain_no_heightscan_pelvis = replace(
    g1_wbt_terrain_no_heightscan, command=g1_29dof_wbt_terrain_command_pelvis
)


# ---------------------------------------------------------------------------
# Depth-distillation presets. Build on g1_wbt_terrain_ref:
#   - replace algo with DistillationPPOAlgoConfig
#   - replace observation with 4-group (actor/critic/teacher/depth_camera)
#   - add curriculum (relax_termination)
#   - drop num_envs to 2048 (depth rendering is ~2x more expensive per env)
# ---------------------------------------------------------------------------


_student_teacher_module = StudentTeacherModuleConfig(
    student_hidden_dims=[2048, 1024, 512, 256, 128],
    teacher_hidden_dims=[
        512,
        256,
        128,
    ],  # MUST match g1_wbt_terrain_ref's actor hidden dims.
    critic_hidden_dims=[512, 256, 128],
    activation="ELU",
    depth_backbone="holosoma.agents.modules.depth_backbone.DepthOnlyFCBackbone58x87Small",
    depth_output_dim=32,
)

_distill_algo_base = DistillationPPOAlgoConfig(
    _target_="wbt_training.utils.student_teacher.PhpDistillationPPO",
    _recursive_=False,
    config=DistillationPPOConfig(
        module=_student_teacher_module,
        num_learning_epochs=2,
        num_mini_batches=96,
        clip_param=0.2,
        gamma=0.99,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.001,
        learning_rate=3e-4,
        max_grad_norm=1.0,
        schedule="adaptive",
        desired_kl=0.01,
        num_steps_per_env=24,
        save_interval=500,
        init_noise_std=0.01,
        num_learning_iterations=30000,
        empirical_normalization=False,
        ppo_start_epoch=0,  # overridden for pure DAgger below
        dagger_end_epoch=10000,
        distill_loss_type="mse",
        dagger_loss_coef=10.0,
        weight_decay=0.0,
    ),
)

# Finetune = PPO ramps up from iter 0 to iter 10000 (plateau at 0.9).
# Build on *_noscale* so action_scale=1.0 and action_scales_by_effort_limit_over_p_gain=False
# match the noscale teacher checkpoints. Using the scaled base here would silently
# apply a 0.25*effort/kp factor that the teacher was never trained with.
#
# num_variants dropped 64 → 3 to match far-tracking's Warp-Distillation-*
# recipe (distillation_env_cfg.py:382). Fewer obstacle variants per motion
# lets the student see each layout more often, cutting early-training BC
# loss — 64 made the student chase a much harder distribution.
_distill_preprocess_hook_kwargs = json.dumps(
    {"add_onpath_obstacle": True, "num_variants": 3}
)

g1_wbt_terrain_warp_distill_finetune = replace(
    g1_wbt_terrain_ref_noscale,
    training=replace(
        g1_wbt_terrain_ref_noscale.training,
        num_envs=2048,
        preprocess_hook_kwargs=_distill_preprocess_hook_kwargs,
    ),
    algo=_distill_algo_base,
    command=g1_29dof_wbt_terrain_distill_command,
    observation=g1_29dof_wbt_terrain_distill_observation,
    curriculum=g1_29dof_wbt_terrain_distill_curriculum,
)

# Pure-DAgger preset: uses the dedicated `Distillation` algorithm (no critic,
# no PPO loss), matching far-tracking's `Warp-Distillation-Flat-G1-v0` recipe.
# Student MLP shrinks to [512, 256, 128] (same size as teacher) since we no
# longer need the extra capacity that PPO exploration would have leveraged.
_student_teacher_module_flat = StudentTeacherModuleConfig(
    student_hidden_dims=[512, 256, 128],
    teacher_hidden_dims=[512, 256, 128],
    critic_hidden_dims=[
        512,
        256,
        128,
    ],  # unused by Distillation, required by the shared cfg type
    activation="ELU",
    depth_backbone="holosoma.agents.modules.depth_backbone.DepthOnlyFCBackbone58x87Small",
    depth_output_dim=32,
)

_distill_algo_pure = DistillationAlgoConfig(
    _target_="wbt_training.utils.student_teacher.PhpDistillation",
    _recursive_=False,
    config=DistillationConfig(
        module=_student_teacher_module_flat,
        num_learning_epochs=2,
        num_mini_batches=96,
        learning_rate=3e-4,
        max_grad_norm=1.0,
        num_steps_per_env=24,
        save_interval=500,
        init_noise_std=0.001,
        num_learning_iterations=10000,
        distill_loss_type="mse",
        distillation_warmup_steps=0,
    ),
)

g1_wbt_terrain_warp_distill = replace(
    g1_wbt_terrain_warp_distill_finetune, algo=_distill_algo_pure
)


DEFAULTS = {
    # Terrain preset for dense WBT with on-path obstacle injection
    "g1_wbt_terrain_ref": g1_wbt_terrain_ref,
    # Terrain preset without the oracle height_scan; 60k iters
    "g1_wbt_terrain_no_heightscan": g1_wbt_terrain_no_heightscan,
    # Same terrain preset but with effective unit action scaling (ref-intent).
    "g1_wbt_terrain_ref_noscale": g1_wbt_terrain_ref_noscale,
    "g1_wbt_terrain_no_heightscan_noscale": g1_wbt_terrain_no_heightscan_noscale,
    # Pelvis-anchored terrain variants (native holosoma convention).
    "g1_wbt_terrain_ref_pelvis": g1_wbt_terrain_ref_pelvis,
    "g1_wbt_terrain_no_heightscan_pelvis": g1_wbt_terrain_no_heightscan_pelvis,
    # Depth distillation: pure DAgger from a trained terrain-ref teacher.
    "g1_wbt_terrain_warp_distill": g1_wbt_terrain_warp_distill,
    # Depth distillation: DAgger + PPO combined-loss finetune.
    "g1_wbt_terrain_warp_distill_finetune": g1_wbt_terrain_warp_distill_finetune,
}

AnnotatedExperimentConfig = Annotated[
    ExperimentConfig,
    tyro.conf.arg(
        constructor=tyro.extras.subcommand_type_from_defaults(
            {f"exp:{name.replace('_', '-')}": cfg for name, cfg in DEFAULTS.items()}
        )
    ),
]
