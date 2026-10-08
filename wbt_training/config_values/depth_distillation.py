"""Observation / curriculum presets for depth-based distillation tasks.

Builds on top of ``terrain.py``'s terrain-reference presets. Adds four obs
groups: ``actor_obs`` (proprio only), ``teacher_obs`` (teacher's privileged
view), ``critic_obs`` (privileged + height_scan, same as the teacher's critic),
``depth_camera`` (Warp-rendered 58x87 depth image, latency-sampled).

The depth term name + group name must match the constants in
``holosoma.agents.distillation_ppo.distillation_ppo``::

    ACTOR_GROUP="actor_obs", TEACHER_GROUP="teacher_obs",
    CRITIC_GROUP="critic_obs", DEPTH_GROUP="depth_camera", DEPTH_TERM="depth_cam".
"""

from __future__ import annotations

from dataclasses import replace

from dataclasses import replace as _replace

from holosoma.config_types.command import CommandTermCfg
from holosoma.config_types.curriculum import CurriculumTermCfg
from holosoma.config_types.observation import (
    ObservationManagerCfg,
    ObsGroupCfg,
    ObsTermCfg,
)
from holosoma.config_values.wbt.g1.curriculum import g1_29dof_wbt_curriculum

from wbt_training.config_values.terrain import (
    critic_obs_terrain_terms,
    g1_29dof_wbt_terrain_command,
    wbt_terrain_actor_obs,
    wbt_terrain_motion_config,
)

# NB: we pass the camera config as a dotted-path string rather than an
# instance, because ObsTermCfg.params is serialized to YAML at config-save time
# and the camera-config class is not JSON-serializable.
_G1_FLAT_D435I_PATH = (
    "holosoma.sensors.warp.camera_config.d435i_depth_config.G1FlatRsD435iConfig"
)


# ---------------------------------------------------------------------------
# Actor obs (student): strictly proprioceptive — no motion-ref keypoints, no
# height_scan, no lin_vel. The student only sees joint state + base angular
# velocity + gravity proxy, plus the prior action.
# ---------------------------------------------------------------------------
_student_actor_terms = {
    # Projected gravity in the motion-command anchor frame — 3-dim static
    # orientation cue. Matches far-tracking's Warp-Distillation-Flat policy
    # group (see tracking/mdp/observations.py:82-88); without it the student
    # can confuse different torso attitudes with similar joint states.
    "robot_anchor_projected_gravity": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:robot_anchor_projected_gravity",
        scale=1.0,
        noise=0.05,
    ),
    "base_ang_vel": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:base_ang_vel",
        scale=1.0,
        noise=0.2,
    ),
    "dof_pos": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:dof_pos", scale=1.0, noise=0.01
    ),
    "dof_vel": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:dof_vel", scale=1.0, noise=0.5
    ),
    "actions": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:actions", scale=1.0, noise=0.0
    ),
    # One-hot velocity command (15 dims). Matches far-tracking's `command`
    # obs group (distillation_env_cfg.py:209-216 +
    # utils/my_on_policy_runner.py:285) which concatenates the command
    # group's `velocity_command` onto the student's `policy` group input.
    # Teacher was trained without this; it lives only in the student group.
    "velocity_command": ObsTermCfg(
        func="wbt_training.config_values.motion_command:velocity_command",
        scale=1.0,
        noise=0.0,
    ),
}

wbt_distill_actor_obs = ObsGroupCfg(
    concatenate=True, enable_noise=True, history_length=1, terms=_student_actor_terms
)

# Teacher obs: teacher's former actor_obs (privileged + height_scan) plus the
# ``which_motion`` routing term. Noise is disabled so the teacher generates
# deterministic DAgger labels.
#
# The observation manager concatenates terms in sorted-name order, so
# ``RoutingColumnTeacherMixin`` resolves this term's feature offset from the
# live manager and strips exactly that column before the teacher forward.
# ``which_motion`` also flags off-track experts by setting the routing value to
# ``-1``; core emits a zero teacher action for those rows and
# ``PhpDistillationPPO._dagger_loss`` masks them. This mirrors far-tracking's
# ``PrivilegedCfg.which_motion`` (distillation_env_cfg.py:176,
# observations.py:190-197).
_teacher_terms: dict[str, ObsTermCfg] = {
    "which_motion": ObsTermCfg(
        func="wbt_training.config_values.motion_command:which_motion",
        scale=1.0,
        noise=0.0,
    ),
    **wbt_terrain_actor_obs.terms,
}
wbt_distill_teacher_obs = replace(
    wbt_terrain_actor_obs, enable_noise=False, terms=_teacher_terms
)

# Critic obs: privileged + height_scan, same as the teacher preset's critic.
wbt_distill_critic_obs = ObsGroupCfg(
    concatenate=True,
    enable_noise=False,
    history_length=1,
    terms=critic_obs_terrain_terms,
)

# Depth group: single ``depth_cam`` term backed by the Warp ray-caster. Kept in
# its own non-concatenating group because images are 3D tensors.
depth_cam_term = ObsTermCfg(
    func="holosoma.managers.observation.terms.depth:WarpDepthImageObsTerm",
    scale=1.0,
    noise=0.0,
    params={
        "camera_sensor_cfg": _G1_FLAT_D435I_PATH,
        "command_name": "motion_command",
        "latency_frame": (3, 4),
        "resize": (58, 87),
        "buffer_len": 6,
        "enable_holes": True,
        # Populated by ``apply_terrain_preprocess`` at config-build time — see
        # terrain.py. Non-None required at term __init__.
        "terrain_mesh_vertices": None,
        "terrain_mesh_faces": None,
    },
)

wbt_distill_depth_obs = ObsGroupCfg(
    concatenate=False,
    enable_noise=False,
    history_length=1,
    terms={"depth_cam": depth_cam_term},
)

g1_29dof_wbt_terrain_distill_observation = ObservationManagerCfg(
    groups={
        "actor_obs": wbt_distill_actor_obs,
        "critic_obs": wbt_distill_critic_obs,
        "teacher_obs": wbt_distill_teacher_obs,
        "depth_camera": wbt_distill_depth_obs,
    }
)


# ---------------------------------------------------------------------------
# Curriculum: relax termination thresholds between step 24k and 240k so the
# student survives early training. Loosens the thresholds by 2x over the window.
# ---------------------------------------------------------------------------
wbt_distill_relax_termination_term = CurriculumTermCfg(
    func="wbt_training.config_values.curriculum:RelaxTerminationThresholds",
    params={
        # The terrain preset's termination manager has one term `bad_tracking`
        # (class BadTracking) whose params include three sub-thresholds
        # (bad_ref_pos_threshold, bad_ref_ori_threshold,
        #  bad_motion_body_pos_threshold). All of them get scaled by the same
        # progress factor.
        "term_names": ["bad_tracking"],
        "curriculum_start_step": 24_000,
        "curriculum_end_step": 240_000,
        "target_multiplier": 2.0,
        "decay_type": "linear",
    },
)

g1_29dof_wbt_terrain_distill_curriculum = _replace(
    g1_29dof_wbt_curriculum,
    step_terms={
        **g1_29dof_wbt_curriculum.step_terms,
        "relax_termination": wbt_distill_relax_termination_term,
    },
)


# ---------------------------------------------------------------------------
# Command override: disable the adaptive motion-timestep sampler for
# distillation. Matches far-tracking's
# `config/g1/flat_env_cfg.py:34,51,69` which set
# `self.commands.motion.enable_adaptive_sampling = False` on the distill
# task. The teacher preset keeps adaptive sampling on (it's useful during
# RL to over-weight failing phases); the student sees a uniform-in-time
# distribution of motion frames so Loss/behavior is comparable across the
# two codebases.
wbt_distill_motion_config = replace(
    wbt_terrain_motion_config, use_adaptive_timesteps_sampler=False
)

g1_29dof_wbt_terrain_distill_command = replace(
    g1_29dof_wbt_terrain_command,
    setup_terms={
        "motion_command": CommandTermCfg(
            func="wbt_training.config_values.motion_command:PhpMotionCommand",
            params={
                "motion_config": wbt_distill_motion_config,
                # Residual-gated PPO, driven entirely by the per-phase DAgger
                # residual -- there is no hand-written phase criterion. **On by
                # default**, so every distill run gets it; set this False (or
                # ``ppo_gate_adaptive`` 0) to recover the plain scalar-``ppo_coef``
                # loss exactly. Every key must be present here to be settable at
                # all -- tyro derives the
                # ``--command.setup-terms.motion-command.params.*`` flags from
                # this dict's keys. See PhpMotionCommand._attach_ppo_gate.
                "ppo_gate_enabled": True,
                # Backward smear on the residual table, in seconds (bins are ~1 s,
                # so this is also the bin count). GAE propagates a phase's
                # advantage back in time, so the frames that set up a hard phase
                # need PPO weight too. 0 disables the smear.
                "ppo_gate_pre_s": 1.0,
                # Floor of the table: the PPO weight a phase gets when its residual
                # says imitation is already handling it.
                #
                # Do not set this to 0. Wherever the gate is at its floor,
                # ``which_motion`` can independently mask the DAgger term, and at
                # a floor of 0 that leaves the sample with no gradient from either
                # term -- self-sealing, since the mask fires precisely where the
                # policy is already off-track. A run with floor 0 tracked its
                # gated phases correctly and then collapsed in the ungated ones.
                # Only values near 0 are dangerous; the floor is otherwise a soft
                # knob, and what it buys is ``low * ppo_coef`` on an easy bin --
                # 0.45 here against 0.9 on a hard one.
                "ppo_gate_low": 0.5,
                # How much the residual table modulates the gate, blended against
                # all-ones: 1 = fully residual-driven, 0 = all-ones, i.e. the
                # ungated scalar-``ppo_coef`` loss reproduced exactly. All-ones is
                # also what the gate returns before the EMA warms up -- "no
                # information yet" must mean *full* reward pressure, never none,
                # since a gate reading ~0 everywhere is the pure-DAgger regime,
                # which can lose a skill the policy already had.
                "ppo_gate_adaptive": 1.0,
                # Residual quantiles bracketing the ramp: bins at or below
                # `_lo` sit at the floor, bins at or above `_hi` get full weight.
                # Must be a ramp between two quantiles, not a division by one --
                # with the hard phase a minority of bins, dividing by a single
                # quantile saturates the whole table. Watch `Env/ppo_gate_spread`:
                # it goes to 0 exactly when the table has stopped distinguishing
                # phases, which `Loss/ppo_gate_mean` cannot tell you.
                "ppo_gate_quantile_lo": 0.50,
                "ppo_gate_quantile_hi": 0.95,
                # EMA rate for the per-bin residual, matching
                # AdaptiveTimestepSampler's adaptive_alpha. Slow on purpose:
                # raising PPO where the residual is high changes the residual, so
                # the loop needs a time constant well above one iteration.
                "ppo_gate_ema_alpha": 0.001,
                # Unmasked samples required before the residual table is used at
                # all; until then the gate is all-ones. A cold EMA reads ~0
                # everywhere, and acting on that is the pure-DAgger regime above.
                "ppo_gate_warmup_samples": 5.0e5,
            },
        )
    },
)
