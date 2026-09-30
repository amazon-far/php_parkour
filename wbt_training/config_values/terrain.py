"""Terrain-specific presets + runtime hooks for G1 whole-body-tracking.

Contents:
 * Preset configs: command / observation / randomization / reward / termination / action
 * ``height_scan``: registered on the observation term ``height_scan``
 * ``apply_terrain_preprocess``: registered on ``training.preprocess_hook``

The G1 implicit-actuator motor calibration
(``build_g1_implicit_actuators``) lives in ``config_values.robot_config``
since it's robot-specific and used by any preset that sets
``control_mode="implicit_position_target"``, not just terrain.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import tempfile
from dataclasses import replace
from typing import TYPE_CHECKING

from holosoma.config_types.action import ActionManagerCfg, ActionTermCfg
from holosoma.config_types.command import CommandTermCfg
from holosoma.config_types.experiment import ExperimentConfig
from holosoma.config_types.observation import (
    ObservationManagerCfg,
    ObsGroupCfg,
    ObsTermCfg,
)
from holosoma.config_types.randomization import (
    RandomizationManagerCfg,
    RandomizationTermCfg,
)
from holosoma.config_types.reward import RewardManagerCfg, RewardTermCfg
from holosoma.config_types.termination import TerminationManagerCfg, TerminationTermCfg
from holosoma.config_values.wbt.g1.command import (
    g1_29dof_wbt_command,
    init_pose_config,
    motion_config,
)
from holosoma.config_values.wbt.g1.observation import critic_obs_shared_terms
from holosoma.config_values.wbt.g1.randomization import base_step_terms
from holosoma.config_values.wbt.g1.reward import g1_29dof_wbt_reward

if TYPE_CHECKING:
    import torch
    from holosoma.envs.wbt.wbt_manager import WholeBodyTrackingManager


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Command preset
# ---------------------------------------------------------------------------
terrain_adaptive_motion_config = replace(
    motion_config,
    start_at_timestep_zero_prob=0.0,
    freeze_at_timestep_zero_prob=0.0,
    adaptive_kernel_size=3,
    adaptive_lambda=0.8,
    adaptive_alpha=0.001,
)

wbt_terrain_init_pose_config = replace(
    init_pose_config, root_lin_vel=[0.1, 0.1, 0.05], root_ang_vel=[0.1, 0.1, 0.1]
)

wbt_terrain_motion_config = replace(
    terrain_adaptive_motion_config,
    enable_default_pose_prepend=False,
    enable_default_pose_append=False,
    adaptive_uniform_ratio=1e-6,
    noise_to_initial_pose=wbt_terrain_init_pose_config,
)

g1_29dof_wbt_terrain_command = replace(
    g1_29dof_wbt_command,
    setup_terms={
        "motion_command": CommandTermCfg(
            func="holosoma.managers.command.terms.wbt:MotionCommand",
            params={"motion_config": wbt_terrain_motion_config},
        )
    },
)

# Pelvis-anchored variant for new training. The torso variant above is
# retained so existing checkpoints remain reproducible.
wbt_terrain_motion_config_pelvis = replace(
    wbt_terrain_motion_config, body_name_ref=["pelvis"]
)

g1_29dof_wbt_terrain_command_pelvis = replace(
    g1_29dof_wbt_command,
    setup_terms={
        "motion_command": CommandTermCfg(
            func="holosoma.managers.command.terms.wbt:MotionCommand",
            params={"motion_config": wbt_terrain_motion_config_pelvis},
        )
    },
)


# ---------------------------------------------------------------------------
# Observation preset (adds height_scan). Holosoma concatenates term names in
# alphabetical order, matching the reference policy's on-disk layout here.
# ---------------------------------------------------------------------------
height_scan_term = ObsTermCfg(
    func="wbt_training.config_values.terrain:height_scan", scale=1.0, noise=0.0
)

wbt_terrain_actor_obs = ObsGroupCfg(
    concatenate=True,
    enable_noise=True,
    history_length=1,
    terms={
        "motion_command": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:motion_command",
            scale=1.0,
            noise=0.0,
        ),
        "motion_ref_pos_b": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:motion_ref_pos_b",
            scale=1.0,
            noise=0.25,
        ),
        "motion_ref_ori_b": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:motion_ref_ori_b",
            scale=1.0,
            noise=0.05,
        ),
        "base_lin_vel": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:base_lin_vel",
            scale=1.0,
            noise=0.5,
        ),
        "base_ang_vel": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:base_ang_vel",
            scale=1.0,
            noise=0.2,
        ),
        "dof_pos": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:dof_pos",
            scale=1.0,
            noise=0.01,
        ),
        "dof_vel": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:dof_vel", scale=1.0, noise=0.5
        ),
        "actions": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:actions", scale=1.0, noise=0.0
        ),
        "height_scan": height_scan_term,
    },
)

critic_obs_terrain_terms = {**critic_obs_shared_terms, "height_scan": height_scan_term}

g1_29dof_wbt_terrain_observation = ObservationManagerCfg(
    groups={
        "actor_obs": wbt_terrain_actor_obs,
        "critic_obs": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=1,
            terms=critic_obs_terrain_terms,
        ),
    }
)


# No-heightscan variant. Same actor/critic layout minus the height_scan rays
# (which are an oracle-style sensor — not realistic on the robot). The actor
# drops from 449 -> 160 dims; critic drops correspondingly.
wbt_terrain_actor_obs_no_hs = replace(
    wbt_terrain_actor_obs,
    terms={k: v for k, v in wbt_terrain_actor_obs.terms.items() if k != "height_scan"},
)

g1_29dof_wbt_terrain_observation_no_heightscan = ObservationManagerCfg(
    groups={
        "actor_obs": wbt_terrain_actor_obs_no_hs,
        "critic_obs": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=1,
            terms=critic_obs_shared_terms,
        ),
    }
)


# ---------------------------------------------------------------------------
# Randomization preset
# ---------------------------------------------------------------------------
wbt_terrain_robot_state_dr_at_setup = {
    "randomize_robot_rigid_body_material_startup": RandomizationTermCfg(
        func="holosoma.managers.randomization.terms.locomotion:randomize_robot_rigid_body_material_startup",
        params={
            "static_friction_range": [0.4, 1.3],
            "dynamic_friction_range": [0.4, 1.1],
            "restitution_range": [0.0, 0.5],
        },
    ),
    "randomize_base_com_startup": RandomizationTermCfg(
        func="holosoma.managers.randomization.terms.locomotion:randomize_base_com_startup",
        params={
            "base_com_range": {
                "x": [-0.025, 0.025],
                "y": [-0.05, 0.05],
                "z": [-0.05, 0.05],
            },
            "enabled": True,
        },
    ),
    "setup_dof_pos_bias": RandomizationTermCfg(
        func="holosoma.managers.randomization.terms.locomotion:setup_dof_pos_bias",
        params={
            "dof_pos_bias_range": [-0.01, 0.01],
            "dof_pos_bias_additive_overrides": {"ankle": [-0.05, 0.05]},
            "enabled": True,
        },
    ),
}

wbt_terrain_setup_terms = {
    "push_randomizer_state": RandomizationTermCfg(
        func="holosoma.managers.randomization.terms.locomotion:PushRandomizerState",
        params={
            "push_interval_s": [0.5, 1.0],
            "max_push_vel": [0.1, 0.1, 0.05, 0.1, 0.1, 0.1],
            "enabled": True,
        },
    ),
    "actuator_randomizer_state": RandomizationTermCfg(
        func="holosoma.managers.randomization.terms.locomotion:ActuatorRandomizerState",
        params={
            "kp_range": [1.0, 1.0],
            "kd_range": [1.0, 1.0],
            "rfi_lim_range": [1.0, 1.0],
            "enable_pd_gain": False,
            "enable_rfi_lim": False,
        },
    ),
    "setup_action_delay_buffers": RandomizationTermCfg(
        func="holosoma.managers.randomization.terms.locomotion:setup_action_delay_buffers",
        params={"ctrl_delay_step_range": [0, 1], "enabled": False},
    ),
    **wbt_terrain_robot_state_dr_at_setup,
}

wbt_terrain_reset_terms = {
    "push_randomizer_state": RandomizationTermCfg(
        func="holosoma.managers.randomization.terms.locomotion:PushRandomizerState"
    ),
    "randomize_push_schedule": RandomizationTermCfg(
        func="holosoma.managers.randomization.terms.locomotion:randomize_push_schedule"
    ),
    "randomize_action_delay": RandomizationTermCfg(
        func="holosoma.managers.randomization.terms.locomotion:randomize_action_delay"
    ),
    "actuator_randomizer_state": RandomizationTermCfg(
        func="holosoma.managers.randomization.terms.locomotion:ActuatorRandomizerState"
    ),
    "randomize_dof_state": RandomizationTermCfg(
        func="holosoma.managers.randomization.terms.locomotion:randomize_dof_state",
        params={
            "joint_pos_scale_range": [1.0, 1.0],
            "joint_vel_range": [0.0, 0.0],
            "joint_pos_bias_range": [0.0, 0.0],
            "randomize_dof_pos_bias": False,
        },
    ),
}

g1_29dof_wbt_terrain_randomization = RandomizationManagerCfg(
    setup_terms={**wbt_terrain_setup_terms},
    reset_terms={**wbt_terrain_reset_terms},
    step_terms={**base_step_terms},
)


# ---------------------------------------------------------------------------
# Reward preset
# ---------------------------------------------------------------------------
g1_29dof_wbt_terrain_reward = RewardManagerCfg(
    terms={
        **g1_29dof_wbt_reward.terms,
        "motion_global_ref_position_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_global_ref_position_error_exp",
            params={"sigma": 0.3},
            weight=1.0,
        ),
        "motion_global_ref_orientation_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_global_ref_orientation_error_exp",
            params={"sigma": 0.4},
            weight=1.0,
        ),
        "limits_dof_pos": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:limits_dof_pos",
            params={"soft_dof_pos_limit": 0.9},
            weight=-10.0,
        ),
        "undesired_contacts": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:UndesiredContacts",
            params={
                "threshold": 1.0,
                "undesired_contacts_body_names": (
                    "^(?!left_foot_contact_point$)(?!right_foot_contact_point$)"
                    "(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)"
                    "(?!left_wrist_yaw_link$)(?!right_wrist_yaw_link$).+$"
                ),
            },
            weight=-0.5,
        ),
    }
)


# ---------------------------------------------------------------------------
# Termination preset
# ---------------------------------------------------------------------------
g1_29dof_wbt_terrain_termination = TerminationManagerCfg(
    terms={
        "timeout": TerminationTermCfg(
            func="holosoma.managers.termination.terms.common:timeout_exceeded",
            is_timeout=True,
        ),
        "bad_tracking": TerminationTermCfg(
            func="holosoma.managers.termination.terms.wbt:BadTracking",
            params={
                "bad_ref_pos_threshold": 0.5,
                "bad_ref_ori_threshold": 0.8,
                "bad_motion_body_pos_threshold": 0.5,
                "body_names_to_track": [
                    "pelvis",
                    "left_hip_roll_link",
                    "left_knee_link",
                    "left_ankle_roll_link",
                    "right_hip_roll_link",
                    "right_knee_link",
                    "right_ankle_roll_link",
                    "torso_link",
                    "left_shoulder_roll_link",
                    "left_elbow_link",
                    "left_wrist_yaw_link",
                    "right_shoulder_roll_link",
                    "right_elbow_link",
                    "right_wrist_yaw_link",
                ],
                "bad_motion_body_pos_body_names": [
                    "left_ankle_roll_link",
                    "right_ankle_roll_link",
                    "left_wrist_yaw_link",
                    "right_wrist_yaw_link",
                ],
                "bad_object_pos_threshold": 0.25,
                "bad_object_ori_threshold": 0.8,
            },
        ),
    }
)


# ---------------------------------------------------------------------------
# Implicit action preset (ref's position-target path; IsaacSim's built-in PD
# drive applies torque from position targets).
# ---------------------------------------------------------------------------
g1_29dof_joint_pos_implicit = ActionManagerCfg(
    terms={
        "joint_control": ActionTermCfg(
            func="holosoma.managers.action.terms.joint_control:JointPositionTargetActionTerm",
            params={},
            scale=1.0,
            clip=None,
        )
    }
)


# ---------------------------------------------------------------------------
# Height-scan observation term
# Registered on the obs term ``height_scan`` above; reads the RayCaster
# sensor registered by holosoma.simulator.isaacsim.isaacsim as
# ``height_scanner`` on the scene.
# ---------------------------------------------------------------------------
def height_scan(env: WholeBodyTrackingManager, offset: float = 0.5) -> torch.Tensor:
    sensor = env.simulator.scene.sensors["height_scanner"]
    return (
        sensor.data.pos_w[:, 2].unsqueeze(1) - sensor.data.ray_hits_w[..., 2] - offset
    )


# ---------------------------------------------------------------------------
# Pre-env-build preprocess hook
# Registered on ``training.preprocess_hook``. Resolves registry_name,
# samples per-env obstacles, exports terrain OBJ, splices motion file into
# command config. This application-specific work lives in PHP.
# ---------------------------------------------------------------------------
def apply_terrain_preprocess(
    cfg: ExperimentConfig,
    *,
    add_onpath_obstacle: bool = False,
    num_variants: int | None = None,
    obstacle_seed: int | None = None,
) -> ExperimentConfig:
    if not cfg.training.registry_name:
        return cfg

    from wbt_training.utils.multi_motion_helpers import pull_paired_from_wandb_registry
    from wbt_training.utils.registry import RegistryResolver, record_registry_usage

    registries = RegistryResolver().resolve(cfg.training.registry_name)
    cfg = dataclasses.replace(
        cfg,
        training=dataclasses.replace(
            cfg.training, registry_name=",".join(r.reference for r in registries)
        ),
    )
    record_registry_usage(registries, cfg.logger.type)

    # pull_paired_* separates .npy obstacle arrays from .obj meshes; plain
    # pull_from_* would feed OBJ paths into combine_terrains and crash on .shape.
    _, motion_file, terrain_npy_file, terrain_obj_files = (
        pull_paired_from_wandb_registry([r.local_uri for r in registries])
    )
    logger.info(f"Registry resolution complete. Motion file: {motion_file}")

    terrain_obj_path: str | None = None
    terrain_mesh_for_depth = None  # kept out of the OBJ branch; populated only when we have a live trimesh in memory
    if add_onpath_obstacle and terrain_npy_file:
        from wbt_training.utils.obstacle_helpers import add_onpath_obstacle_standalone

        n_variants = num_variants or cfg.training.num_envs
        _, motion_file_str, terrain_mesh, _ = add_onpath_obstacle_standalone(
            n_variants, str(motion_file), str(terrain_npy_file), obstacle_seed
        )
        _fd, terrain_obj_path = tempfile.mkstemp(suffix=".obj")
        os.close(_fd)
        terrain_mesh.export(terrain_obj_path)
        motion_file = motion_file_str
        terrain_mesh_for_depth = terrain_mesh
    elif terrain_obj_files:
        terrain_obj_path = str(terrain_obj_files[0])
        # If a depth obs term is wired up, load the pre-built OBJ so we can hand
        # vertices/faces to the warp ray-caster. Skipped cheaply if no depth term.
        if cfg.observation is not None and "depth_camera" in cfg.observation.groups:
            import trimesh  # local import: only needed for depth-distillation presets

            loaded = trimesh.load(terrain_obj_path, force="mesh")
            if isinstance(loaded, trimesh.Trimesh):
                terrain_mesh_for_depth = loaded

    if terrain_obj_path is not None:
        from holosoma.config_types.terrain import MeshType

        cfg = dataclasses.replace(
            cfg,
            terrain=dataclasses.replace(
                cfg.terrain,
                terrain_term=dataclasses.replace(
                    cfg.terrain.terrain_term,
                    mesh_type=MeshType.LOAD_OBJ,
                    obj_file_path=terrain_obj_path,
                    num_rows=1,
                    num_cols=1,
                ),
            ),
            # `scene` is a top-level ExperimentConfig field (was
            # `simulator.config.scene` before holosoma's scene/sensor/plugin split).
            scene=dataclasses.replace(cfg.scene, env_spacing=0.0),
        )

    # Splice resolved motion_file into command.motion_config.motion_file
    if cfg.command is not None:
        setup_terms = cfg.command.setup_terms
        if setup_terms and "motion_command" in setup_terms:
            mc_term = setup_terms["motion_command"]
            mc_params = dict(mc_term.params) if mc_term.params else {}
            if "motion_config" in mc_params:
                mc_cfg_obj = mc_params["motion_config"]
                if isinstance(mc_cfg_obj, dict):
                    mc_cfg_obj = {**mc_cfg_obj, "motion_file": str(motion_file)}
                else:
                    mc_cfg_obj = dataclasses.replace(
                        mc_cfg_obj, motion_file=str(motion_file)
                    )
                mc_params["motion_config"] = mc_cfg_obj
                new_setup_terms = dict(setup_terms)
                new_setup_terms["motion_command"] = dataclasses.replace(
                    mc_term, params=mc_params
                )
                cfg = dataclasses.replace(
                    cfg,
                    command=dataclasses.replace(
                        cfg.command, setup_terms=new_setup_terms
                    ),
                )

    # Plumb the terrain mesh into the Warp depth obs term params if present. The
    # depth term requires non-None vertices/faces at __init__ — we inject them
    # here so the depth-distillation presets don't need extra wiring. Silent
    # no-op when the preset has no depth_camera group.
    if (
        terrain_mesh_for_depth is not None
        and cfg.observation is not None
        and "depth_camera" in cfg.observation.groups
    ):
        depth_group = cfg.observation.groups["depth_camera"]
        depth_terms = depth_group.terms
        for term_name, term_cfg in depth_terms.items():
            # Only update terms whose params already declare depth-mesh slots.
            if (
                "terrain_mesh_vertices" in term_cfg.params
                and "terrain_mesh_faces" in term_cfg.params
            ):
                # The mesh arrays live in ``params`` at runtime but need to be
                # serializable for the YAML config snapshot. Dump the mesh to a
                # temp .npz and stash the path; the term ``__init__`` reads the
                # path and reconstitutes the arrays. Keeps ``to_serializable_dict``
                # fast and avoids embedding 100MB of geometry in the YAML.
                import numpy as np

                _mesh_fd, mesh_path = tempfile.mkstemp(suffix="_depth_mesh.npz")
                os.close(_mesh_fd)
                np.savez(
                    mesh_path,
                    vertices=np.asarray(
                        terrain_mesh_for_depth.vertices, dtype=np.float32
                    ),
                    faces=np.asarray(terrain_mesh_for_depth.faces, dtype=np.int64),
                )
                # Reuse the vertices/faces slots: the term accepts either arrays
                # (legacy) or a string path (current).
                term_cfg.params["terrain_mesh_vertices"] = mesh_path
                term_cfg.params["terrain_mesh_faces"] = mesh_path
                logger.info(
                    f"Plumbed terrain mesh (verts={len(terrain_mesh_for_depth.vertices)}, "
                    f"faces={len(terrain_mesh_for_depth.faces)}) into depth obs term "
                    f"'{term_name}' via {mesh_path}."
                )

    return cfg
