"""php-specific ``MotionCommand`` extensions.

Two behaviours live here rather than in holosoma core, because both are
designs this project introduced for depth distillation rather than parts of
holosoma's motion-tracking contract:

**One-hot velocity command.** ``vel_cmd`` is a 15-dim one-hot per motion frame,
added to match far-tracking's student observation (their
``MotionCommand.vel_cmd``, ``tracking/mdp/commands.py:237-244``). Holosoma's
own NPZ format has no such field, and only this project's ``velocity_command``
obs term consumes it.

**Expert-balanced timestep sampling.** When several teachers are distilled at
once, each env must be assigned an *expert* uniformly rather than a *clip*.
Without it clip counts dominate — one registry with 600 clips against six with
40 takes ~69% of envs, so a single teacher swamps the DAgger signal. The notion
of an "expert" is a distillation concept, not a tracking one.

Wire in by pointing a command term's ``func`` at
``wbt_training.config_values.motion_command:PhpMotionCommand``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from holosoma.managers.command.terms.wbt import MotionCommand
from holosoma.managers.observation.terms.wbt import gravity_vector
from holosoma.utils.rotations import quat_rotate_inverse as _quat_rotate_inverse
from holosoma.utils.rotations import (
    get_euler_xyz,
    quat_from_euler_xyz,
    quat_mul,
)


class PhpMotionCommand(MotionCommand):
    """``MotionCommand`` plus ``vel_cmd`` and expert-balanced resampling.

    ``reset`` is overridden wholesale: upstream's sampling block sits in the
    middle of a ~250-line method with no finer-grained hook, so there is
    nothing smaller to override. :meth:`_assert_upstream_reset_unchanged`
    guards against that copy going stale — see its docstring.
    """

    # Upstream ``MotionCommand.reset`` body length (in lines) that this
    # subclass was written against. Bumping holosoma will trip the guard if
    # the method is edited, which is the signal to re-diff the copy below.
    _UPSTREAM_RESET_LINES = 189

    def __init__(self, cfg: Any, env: Any):
        super().__init__(cfg, env)
        # Lazily-built per-expert frame-index pools (see _resample_time_steps).
        self._expert_frame_pools: list[torch.Tensor] | None = None
        self._expert_pool_lengths: torch.Tensor | None = None
        self._num_experts: int | None = None

    # ------------------------------------------------------------------ setup

    def setup(self) -> None:
        super().setup()
        self._attach_vel_cmd()
        self._attach_ppo_gate()
        self._assert_upstream_reset_unchanged()

    # -------------------------------------------------------------- ppo gate

    def _attach_ppo_gate(self) -> None:
        """Set up the residual-driven PPO gate over the motion buffer.

        ``DistillationPPO`` splits its loss into a PPO term (weight
        ``gate * ppo_coef``) and a DAgger term (weight ``1 - gate * ppo_coef``).
        Reward and imitation do not pay off equally on every part of a motion, and
        a scalar ``ppo_coef`` cannot express that: lowering it globally buys local
        tracking on the easy phases and loses the hard ones outright. The gate
        lets the hard phases keep their reward pressure while the phases imitation
        already handles run closer to pure DAgger.

        **Which phases those are is learned, not declared.** The gate is driven
        entirely by the per-phase DAgger residual: reward pressure pays off
        exactly where imitation cannot do the job, and the residual measures that
        directly.

        This deliberately replaces an earlier hand-written criterion that
        thresholded reference pelvis height in metres. That rule had to be retuned
        per skill, and its first form turned out to gate none of the *descent* of
        an obstacle at all -- its test asked the pelvis to still be below a future
        maximum, which is false once it is on the way down. A residual-driven gate
        has no such blind spot to discover: it cannot be structurally wrong about
        a phase, only under-sampled, which ``ppo_gate_bins_seen`` reports.

        Residuals are accumulated per ~1 s bin of the global concatenated-frame
        index, matching ``AdaptiveTimestepSampler``'s binning so the two share a
        resolution. No motion file is re-read here; only ``time_step_total`` is
        needed, to size the bins.
        """
        params = self.cfg.params or {}
        # Fallback False, not True: the distill preset declares this key as True,
        # but the teacher presets share this command term and never declare it --
        # a True fallback would silently build a gate for them too.
        self.motion._ppo_gate_on = bool(params.get("ppo_gate_enabled", False))
        if not self.motion._ppo_gate_on:
            return

        self._gate_low = float(params.get("ppo_gate_low", 0.5))
        self._gate_adaptive = float(params.get("ppo_gate_adaptive", 1.0))
        self._gate_q_lo = float(params.get("ppo_gate_quantile_lo", 0.50))
        self._gate_q_hi = float(params.get("ppo_gate_quantile_hi", 0.95))
        self._gate_alpha = float(params.get("ppo_gate_ema_alpha", 0.001))
        self._gate_warmup = float(params.get("ppo_gate_warmup_samples", 5.0e5))
        total = int(self.motion.time_step_total)
        fps = max(int(round(1.0 / self._env.dt)), 1)
        # Backward smear on the bin table, in bins. Not slack and not geometry:
        # GAE propagates a phase's advantage back in time, so the frames that
        # *set up* a hard phase must carry PPO weight too, or the update
        # penalises a failure without training the steps that caused it.
        # Bins are ~1 s wide, so seconds and bins convert 1:1.
        self._gate_pre_bins = max(int(round(float(params.get("ppo_gate_pre_s", 1.0)))), 0)
        self._gate_nbins = total // fps + 1
        self._gate_bin_sum = torch.zeros(self._gate_nbins, device=self.device)
        self._gate_bin_cnt = torch.zeros(self._gate_nbins, device=self.device)
        self._gate_seen = 0.0
        # bin index of every frame, precomputed once (used on every access)
        self._gate_frame_bin = torch.clamp(
            (torch.arange(total, device=self.device) * self._gate_nbins) // max(total, 1),
            0,
            self._gate_nbins - 1,
        )

        import logging

        logging.getLogger(__name__).info(
            "PhpMotionCommand: residual-driven PPO gate on %d frames in %d bins "
            "(low=%.2f adaptive=%.2f quantiles=%.2f/%.2f alpha=%g warmup=%g pre_bins=%d)",
            total,
            self._gate_nbins,
            self._gate_low,
            self._gate_adaptive,
            self._gate_q_lo,
            self._gate_q_hi,
            self._gate_alpha,
            self._gate_warmup,
            self._gate_pre_bins,
        )

    def observe_dagger_residual(self, residual: torch.Tensor, keep: torch.Tensor) -> None:
        """Fold per-env DAgger residuals into the per-bin EMA behind the gate.

        ``keep`` must exclude rows where ``which_motion`` marked the expert lost.
        This is not optional bookkeeping: ``teacher_act`` zeroes those rows, so
        their residual is exactly 0, and a *failing* phase would therefore read as
        "imitation is perfect here" and have its gate driven to the floor --
        removing the reward pressure precisely where the skill is dying. That is
        the same self-reinforcing trap that makes a lost skill unrecoverable (see
        which_motion's hardcoded thresholds). Masked rows contribute to neither
        the numerator nor the denominator.
        """
        # getattr, not attribute access: ``_attach_ppo_gate`` returns early when
        # the gate is disabled, so these are never assigned on a non-gated run and
        # a bare ``self._gate_bin_sum`` would AttributeError on every rollout step.
        if getattr(self, "_gate_bin_sum", None) is None:
            return
        # Defensive clamp, out-of-place: ``time_steps`` is dtype long already, so
        # ``.long()`` returns the same tensor and an in-place ``clamp_`` would
        # corrupt the command's own clock.
        b = self._gate_frame_bin[
            torch.clamp(self.time_steps, 0, self._gate_frame_bin.shape[0] - 1)
        ]
        w = keep.float()
        step_sum = torch.zeros_like(self._gate_bin_sum).index_add_(0, b, residual * w)
        step_cnt = torch.zeros_like(self._gate_bin_cnt).index_add_(0, b, w)
        a = self._gate_alpha
        self._gate_bin_sum.mul_(1.0 - a).add_(step_sum, alpha=a)
        self._gate_bin_cnt.mul_(1.0 - a).add_(step_cnt, alpha=a)
        self._gate_seen += float(w.sum())

        # Diagnostics. ``ppo_gate_mean`` alone cannot tell you the gate is
        # working: a table that has collapsed to a constant and one that is
        # sharply differentiating phases can share a mean. ``spread`` is the
        # discriminator -- it goes to 0 exactly when the gate has stopped
        # distinguishing phases, which is the failure mode that matters.
        if hasattr(self._env, "log_dict"):
            table = self._adaptive_gate_table()
            engaged = table is not None
            t = lambda v: torch.tensor(float(v), dtype=torch.float)
            self._env.log_dict["ppo_gate_engaged"] = t(engaged)
            self._env.log_dict["ppo_gate_warmup_frac"] = t(
                min(self._gate_seen / max(self._gate_warmup, 1.0), 1.0)
            )
            self._env.log_dict["ppo_gate_bins_seen"] = t(
                (self._gate_bin_cnt > 1e-8).float().mean()
            )
            self._env.log_dict["ppo_gate_spread"] = t(table.std() if engaged else 0.0)
            self._env.log_dict["ppo_gate_table_mean"] = t(table.mean() if engaged else 0.0)

    def _adaptive_gate_table(self) -> torch.Tensor | None:
        """Per-frame gate implied by the residual EMA, or None while it is cold.

        Returns ``None`` rather than a flat table until the EMA has seen
        ``ppo_gate_warmup_samples`` unmasked samples and at least 8 bins, so
        callers can fall back to plain ``ppo_coef`` instead of acting on noise.
        """
        if self._gate_bin_sum is None or self._gate_seen < self._gate_warmup:
            return None
        r = self._gate_bin_sum / self._gate_bin_cnt.clamp(min=1e-8)
        seen = self._gate_bin_cnt > 1e-8
        if int(seen.sum()) < 8:
            return None
        # Ramp between a baseline and a saturation quantile rather than dividing by
        # one of them. Dividing by a single quantile saturates the whole table
        # whenever the hard phase is a minority of bins: with 12% of bins hot, the
        # 75th percentile lands in the *easy* population, so r/q >= 1 everywhere.
        lo = torch.quantile(r[seen], self._gate_q_lo)
        hi = torch.quantile(r[seen], self._gate_q_hi)
        g = ((r - lo) / (hi - lo).clamp(min=1e-8)).clamp(0.0, 1.0)
        # Unseen bins carry no residual estimate. Give them the *maximum* the
        # table can express rather than the floor: an unvisited phase is one we
        # have no evidence is easy, and under-gating a hard phase is what loses a
        # skill, while over-gating one merely costs some local tracking.
        g = torch.where(seen, g, torch.ones_like(g))
        if self._gate_pre_bins > 0:
            # Backward max-smear so a hard bin also raises the bins leading into
            # it -- see _attach_ppo_gate on GAE credit propagation.
            import torch.nn.functional as F

            k = self._gate_pre_bins
            g = F.max_pool1d(
                F.pad(g.view(1, 1, -1), (0, k), mode="replicate"), k + 1, 1
            ).view(-1)
        table = self._gate_low + (1.0 - self._gate_low) * g
        return table[self._gate_frame_bin]

    @property
    def ppo_gate(self) -> torch.Tensor | None:
        """Per-env PPO gate at the current timestep, or ``None`` when disabled.

        While the residual EMA is cold this returns all-ones, i.e. plain scalar
        ``ppo_coef`` -- the pre-gate behaviour. That fallback matters: a gate that
        reads ~0 everywhere for the first few thousand iterations is the
        pure-DAgger regime that destroys a climb skill within ~500 iterations, so
        "no information yet" must mean *full* reward pressure, never none.

        ``ppo_gate_adaptive`` blends the residual table against those all-ones,
        so 0 reproduces the ungated loss exactly and 1 is fully residual-driven.
        """
        if not getattr(self.motion, "_ppo_gate_on", False):
            return None
        ones = torch.ones(int(self.motion.time_step_total), device=self.device)
        table = self._adaptive_gate_table() if self._gate_adaptive > 0.0 else None
        if table is None:
            gate = ones
        else:
            a = self._gate_adaptive
            gate = ((1.0 - a) * ones + a * table).clamp_(0.0, 1.0)
        return gate[self.time_steps]

    def _attach_vel_cmd(self) -> None:
        """Load ``vel_cmd`` from the motion NPZ(s) onto the loader.

        Done after ``super().setup()`` rather than inside the loader so core's
        ``MotionLoader``/``MultiMotionLoader`` need no knowledge of the field.
        Set to ``None`` when any constituent file lacks it, so a mixed set
        never yields ragged shapes.
        """
        loader = self.motion
        files = self._motion_files_for(loader)
        chunks: list[torch.Tensor] = []
        for path in files:
            with np.load(path) as data:
                if "vel_cmd" not in data.files:
                    loader._vel_cmd = None
                    return
                chunks.append(
                    torch.tensor(
                        data["vel_cmd"], dtype=torch.float32, device=self.device
                    )
                )
        loader._vel_cmd = torch.cat(chunks, dim=0) if chunks else None

    def _motion_files_for(self, loader: Any) -> list[str]:
        """Return the NPZ paths behind ``loader``, in frame order.

        Neither ``MotionLoader`` nor ``MultiMotionLoader`` keeps its source
        paths — both consume them as locals during ``__init__`` — so the config
        is used as the source of truth instead, resolved the same way the
        loaders do. ``motion_dir`` mirrors ``MultiMotionLoader``'s ``sorted()``
        glob so frame order matches the concatenation order.
        """
        from pathlib import Path

        from holosoma.utils.path import resolve_data_file_path

        motion_dir = self.motion_cfg.motion_dir
        if motion_dir:
            dirs = motion_dir if isinstance(motion_dir, (list, tuple)) else [motion_dir]
            files: list[str] = []
            for d in dirs:
                expanded = resolve_data_file_path(str(d))
                files.extend(sorted(str(p) for p in Path(expanded).glob("*.npz")))
            return files
        return [str(resolve_data_file_path(self.motion_cfg.motion_file))]

    @property
    def vel_cmd(self) -> torch.Tensor:
        """One-hot velocity command at each env's current timestep.

        Raises rather than zero-filling when the field is absent: a preset that
        wires the ``velocity_command`` obs term cannot run against motions
        without it, and a silent fallback would mismatch the observation width
        a checkpoint was trained with.
        """
        vel = getattr(self.motion, "_vel_cmd", None)
        if vel is None:
            raise RuntimeError(
                "PhpMotionCommand.vel_cmd: motion file has no 'vel_cmd' field. "
                "Presets using the velocity_command obs term require motions "
                "with one-hot velocity commands."
            )
        return vel[self.time_steps]

    # ----------------------------------------------------------- expert pools

    def _expert_balancing_active(self) -> bool:
        """True when the combined motion file marks more than one expert.

        Only single-file loaders take this path; ``MultiMotionLoader`` already
        balances per clip, so its behaviour is left alone.
        """
        motion_idxs = getattr(self.motion, "motion_idxs", None)
        return (
            self.motion.num_motions == 1
            and motion_idxs is not None
            and int(motion_idxs.max().item()) > 0
        )

    def _build_expert_pools(self) -> None:
        """Cache explicit frame-index pools, one per expert.

        Per-expert frames cannot be assumed contiguous: on-path obstacle
        injection tiles the whole motion buffer ``num_variants`` times, so each
        expert's frames land in that many disjoint interleaved blocks. Explicit
        index pools cost ~O(T) int64 — a few MB — and sidestep the issue.
        """
        motion_idxs = self.motion.motion_idxs
        num_experts = int(motion_idxs.max().item()) + 1
        pools: list[torch.Tensor] = []
        for e in range(num_experts):
            idx = (
                torch.nonzero(motion_idxs == e, as_tuple=False)
                .squeeze(-1)
                .to(self.device)
            )
            if idx.numel() == 0:
                # No frames for this expert; fall back to the whole buffer so
                # sampling cannot crash. Such envs get flagged idx == -1 by the
                # which_motion obs term anyway.
                idx = torch.arange(
                    motion_idxs.shape[0], dtype=torch.long, device=self.device
                )
            pools.append(idx)
        self._expert_frame_pools = pools
        self._expert_pool_lengths = torch.tensor(
            [p.numel() for p in pools], dtype=torch.long, device=self.device
        )
        self._num_experts = num_experts

    def _sample_expert_balanced(self, env_ids: torch.Tensor, n: int):
        """Assign each env an expert uniformly, then a frame within it.

        Returns the ``(start_idx, end_idx)`` pair the rest of ``reset``
        expects. Expert frames are non-contiguous under variant tiling, so
        per-clip ranges are meaningless; the full buffer is reported instead,
        leaving the clip-end handling downstream to fire only on a genuine
        last-global-frame sample.
        """
        if self._expert_frame_pools is None:
            self._build_expert_pools()
        pools = self._expert_frame_pools
        pool_lengths = self._expert_pool_lengths
        num_experts = self._num_experts

        expert_ids = torch.randint(0, num_experts, (n,), device=self.device)
        pool_len_for_env = pool_lengths[expert_ids]
        pool_pos = (torch.rand(n, device=self.device) * pool_len_for_env.float()).long()
        time_steps_new = torch.empty(n, dtype=torch.long, device=self.device)
        for e in range(num_experts):
            mask = expert_ids == e
            if mask.any():
                time_steps_new[mask] = pools[e][pool_pos[mask]]
        self.time_steps[env_ids] = time_steps_new
        self.motion_ids[env_ids] = expert_ids

        start_idx = torch.zeros(n, dtype=torch.long, device=self.device)
        end_idx = torch.full(
            (n,), self.motion.time_step_total, dtype=torch.long, device=self.device
        )
        return start_idx, end_idx

    # ------------------------------------------------------------------ guard

    def _assert_upstream_reset_unchanged(self) -> None:
        """Warn when upstream ``reset`` no longer matches what we copied.

        :meth:`reset` below is a line-for-line copy of
        ``MotionCommand.reset`` with the sampling block swapped out. A
        ``__qualname__`` check would not catch an edit *inside* the method, so
        the body length is compared instead — crude, but it turns silent drift
        into a visible warning.
        """
        import inspect
        import warnings

        try:
            lines = len(inspect.getsource(MotionCommand.reset).splitlines())
        except (OSError, TypeError):
            return
        if lines != self._UPSTREAM_RESET_LINES:
            warnings.warn(
                f"MotionCommand.reset is {lines} lines; PhpMotionCommand.reset was "
                f"written against {self._UPSTREAM_RESET_LINES}. Upstream likely "
                f"changed — re-diff PhpMotionCommand.reset against it, then update "
                f"_UPSTREAM_RESET_LINES.",
                RuntimeWarning,
                stacklevel=2,
            )

    # ------------------------------------------------------------------ reset

    def reset(self, env_ids: torch.Tensor | None) -> None:
        """``MotionCommand.reset`` with expert-balanced timestep sampling.

        When expert balancing is inactive — every non-distill preset — this
        delegates wholesale to upstream, so those presets are unaffected by
        anything below. Only the multi-teacher distill path takes the copied
        branch, which duplicates upstream's post-sampling tail verbatim because
        the sampling it must replace sits in the middle of that method with no
        hook to override.
        """
        if not self._expert_balancing_active():
            super().reset(env_ids)
            return

        env_ids = self._ensure_index_tensor(env_ids)
        if env_ids.numel() == 0:
            return

        # 0. Sample the time steps. `phase` is drawn to keep parity with
        # upstream (the adaptive sampler's failed-bin bookkeeping must still
        # run), but the expert-balanced branch below indexes frame pools
        # directly rather than scaling a phase into a clip range.
        if self.motion_cfg.use_adaptive_timesteps_sampler:
            episode_failed = self._env.termination_manager.terminated[env_ids]
            if torch.any(episode_failed):
                failed_at_time_step = self.time_steps[env_ids][episode_failed]
                self.adaptive_timesteps_sampler.update_current_bin_failed_count(
                    failed_at_time_step
                )
            phase = self.adaptive_timesteps_sampler.sample(env_ids.numel())
        else:
            phase = torch.rand(env_ids.numel(), device=self.device)

        if self._env.is_evaluating:
            phase = torch.zeros_like(phase)

        n = env_ids.numel()
        start_idx, end_idx = self._sample_expert_balanced(env_ids, n)

        # --- from here down: copy of upstream MotionCommand.reset ------------

        # Handle start_at_timestep_zero_prob (reset to start of assigned motion)
        prob = self.motion_cfg.start_at_timestep_zero_prob
        if prob >= 1.0:
            self.time_steps[env_ids] = start_idx
        elif prob > 0.0:
            subset = self.time_steps[env_ids]
            rand_vals = torch.rand_like(subset, dtype=torch.float32)
            subset = torch.where(rand_vals < prob, start_idx, subset)
            self.time_steps[env_ids] = subset

        # Pre-advance any frame that is itself a clip-end so the +1 in step() doesn't
        # land on a discontinuous next-clip frame, then wrap any OOB time_steps to 0.
        env_ids_end = torch.where(self.motion.motion_ends[self.time_steps])[0]
        self.time_steps[env_ids_end] += 1
        self.time_steps[self.time_steps >= self.motion.time_step_total] = 0

        # 1. Get the root/body poses from the motion data
        # Index only reset-env timesteps (raw body index 0 = root, same as root_pos_w property)
        reset_ts = self.time_steps[env_ids]
        env_origins = self._env.simulator.scene.env_origins[env_ids]
        root_pos = self.motion._body_pos_w[reset_ts, 0] + env_origins
        root_rot = self.motion._body_quat_w[reset_ts, 0]
        root_lin_vel = self.motion._body_lin_vel_w[reset_ts, 0]
        root_ang_vel = self.motion._body_ang_vel_w[reset_ts, 0]

        dof_pos = self.motion.get_joint_pos(reset_ts)
        dof_vel = self.motion.get_joint_vel(reset_ts)

        # "learned" strategy: for envs starting at frame 0, init robot in default standing
        # pose. Motion command stays at frame 0 — policy learns the transition autonomously.
        if self.motion_cfg.default_pose_transition_strategy == "learned":
            at_start_mask = self.time_steps[env_ids] == start_idx
            if torch.any(at_start_mask):
                n_start = int(at_start_mask.sum().item())
                # Default joint angles
                dof_pos[at_start_mask] = self._env.default_dof_pos_base.expand(
                    n_start, -1
                )
                dof_vel[at_start_mask] = 0.0
                # Default root: keep motion x/y + env_origin, use config z/roll/pitch
                init_state = self._env.robot_config.init_state
                root_pos[at_start_mask, 2] = (
                    env_origins[at_start_mask, 2] + init_state.pos[2]
                )
                root_lin_vel[at_start_mask] = 0.0
                root_ang_vel[at_start_mask] = 0.0
                # Root rotation: keep motion yaw, replace roll/pitch with config defaults
                init_roll, init_pitch = (
                    self._default_init_roll,
                    self._default_init_pitch,
                )
                _, _, motion_yaw = get_euler_xyz(root_rot[at_start_mask], w_last=True)
                default_rot = quat_from_euler_xyz(
                    init_roll.expand(n_start),
                    init_pitch.expand(n_start),
                    motion_yaw,
                )
                root_rot[at_start_mask] = default_rot

        # 2. Adding noise
        # 2.1 prepare the noise scale
        dof_pos_noise = (
            self.init_pose_cfg.dof_pos * self.init_pose_cfg.overall_noise_scale
        )  # float
        root_pos_noise = (
            torch.tensor(
                self.init_pose_cfg.root_pos,
                device=self.device,
            )
            * self.init_pose_cfg.overall_noise_scale
        )  # (3,)
        root_rot_noise_rpy = (
            torch.tensor(
                self.init_pose_cfg.root_rot,
                device=self.device,
            )
            * self.init_pose_cfg.overall_noise_scale
        )  # (3,)
        if self.use_configured_root_velocity_noise:
            root_vel_noise = (
                torch.tensor(
                    self.init_pose_cfg.root_lin_vel,
                    device=self.device,
                )
                * self.init_pose_cfg.overall_noise_scale
            )  # (3,)
            root_ang_vel_noise_rpy = (
                torch.tensor(
                    self.init_pose_cfg.root_ang_vel,
                    device=self.device,
                )
                * self.init_pose_cfg.overall_noise_scale
            )  # (3,)
        else:
            # Reuse push randomizer velocity limits for reset-state velocity noise
            # . Falls back to zero when the push
            # randomizer is disabled.
            push_state = self._env.randomization_manager.get_state(
                "push_randomizer_state"
            )
            _push_vel = (
                getattr(push_state, "max_push_vel", None)
                if push_state is not None
                else None
            )
            if _push_vel is None:
                max_push_vel = torch.zeros(6, device=self.device)
            else:
                max_push_vel = torch.abs(_push_vel.to(self.device))
            root_vel_noise = max_push_vel[:3]
            root_ang_vel_noise_rpy = max_push_vel[3:6]

        # 2.2 Adding noise to dof_pos, root_pos, root_vel, root_ang_vel, root_rot
        # 1.2.1 dof_pos
        target_dof_pos = (
            dof_pos
            + (torch.rand(dof_pos.shape, device=self.device) - 0.5) * 2 * dof_pos_noise
        )  # (num_envs, num_dofs)
        soft_joint_pos_limits = self._env.simulator.dof_pos_limits  # type: ignore[attr-defined]  # (num_dofs, 2)
        target_dof_pos = torch.clip(
            target_dof_pos, soft_joint_pos_limits[:, 0], soft_joint_pos_limits[:, 1]
        )

        # 1.2.2 dof_vel no noise
        target_dof_vel = dof_vel

        # 1.2.3 root_pos
        target_root_pos = root_pos + (
            torch.rand(root_pos.shape, device=self.device) - 0.5
        ) * 2 * root_pos_noise.unsqueeze(0)  # (num_envs, 3)

        # 1.2.4 root_rot
        rand_sample_rpy = (
            (torch.rand((len(env_ids), 3), device=self.device) - 0.5)
            * 2
            * root_rot_noise_rpy
        )
        orientations_delta = quat_from_euler_xyz(
            rand_sample_rpy[:, 0], rand_sample_rpy[:, 1], rand_sample_rpy[:, 2]
        )  # (num_envs, 4), xyzw
        target_root_rot = quat_mul(
            orientations_delta, root_rot, w_last=True
        )  # (num_envs, 4), xyzw

        # 1.2.5 root_lin_vel
        target_root_lin_vel = root_lin_vel + (
            torch.rand(root_lin_vel.shape, device=self.device) - 0.5
        ) * 2 * root_vel_noise.unsqueeze(0)  # (num_envs, 3)

        # 1.2.6 root_ang_vel
        target_root_ang_vel = root_ang_vel + (
            torch.rand(root_ang_vel.shape, device=self.device) - 0.5
        ) * 2 * root_ang_vel_noise_rpy.unsqueeze(0)  # (num_envs, 3)

        # 3. Set the robot states in simulator
        self._env.simulator.dof_pos[env_ids] = target_dof_pos
        self._env.simulator.dof_vel[env_ids] = target_dof_vel

        self._env.simulator.robot_root_states[env_ids, :3] = target_root_pos
        self._env.simulator.robot_root_states[env_ids, 3:7] = target_root_rot
        self._env.simulator.robot_root_states[env_ids, 7:10] = target_root_lin_vel
        self._env.simulator.robot_root_states[env_ids, 10:13] = target_root_ang_vel

        # 4. Set the object states in simulator
        if self.motion.has_object:
            obj_pos = self.motion.get_object_pos_w(reset_ts) + env_origins
            obj_ori = self.motion.get_object_quat_w(reset_ts)
            obj_lin_vel = self.motion.get_object_lin_vel_w(reset_ts)

            # 4.2 add noise to the object states
            obj_pos_noise = torch.tensor(
                [self.init_pose_cfg.object_pos],
                device=self.device,
            )
            obj_pos_noise = (
                obj_pos_noise * self.init_pose_cfg.overall_noise_scale
            )  # (3,)
            target_obj_pos = (
                obj_pos
                + (torch.rand(obj_pos.shape, device=self.device) - 0.5)
                * 2
                * obj_pos_noise
            )

            object_states = torch.cat(
                [target_obj_pos, obj_ori, obj_lin_vel, torch.zeros_like(obj_lin_vel)],
                dim=-1,
            )  # (num_envs, 7)
            # 4.3 set the object states in simulator
            self._env.simulator.set_actor_states(
                [self.object_name], env_ids, object_states
            )


# ---------------------------------------------------------------------------
# Observation term
# ---------------------------------------------------------------------------


def velocity_command(env: Any) -> "torch.Tensor":
    """One-hot velocity command from the motion data, shape ``[num_envs, 15]``.

    Mirrors far-tracking's ``velocity_command``
    (``tracking/mdp/observations.py:205-208``). Gives the student the commanded
    direction for the clip it is tracking; the teacher gets the same
    information implicitly from its motion-reference inputs.

    Requires the command term to be a :class:`PhpMotionCommand` — the base
    ``MotionCommand`` has no ``vel_cmd``.
    """
    cmd = env.command_manager.get_state("motion_command")
    if not hasattr(cmd, "vel_cmd"):
        raise TypeError(
            f"velocity_command obs term requires PhpMotionCommand, got "
            f"{type(cmd).__name__}. Point the command term's func at "
            f"wbt_training.config_values.motion_command:PhpMotionCommand."
        )
    return cmd.vel_cmd.view(env.num_envs, -1)


# ---------------------------------------------------------------------------
# Teacher-routing observation
# ---------------------------------------------------------------------------


def which_motion(
    env: Any,
    bad_ref_pos_threshold: float = 0.5,
    bad_ref_ori_threshold: float = 0.8,
    bad_motion_body_pos_threshold: float = 0.5,
    bad_motion_body_pos_body_names: tuple[str, ...] = (
        "left_ankle_roll_link",
        "right_ankle_roll_link",
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
    ),
) -> torch.Tensor:
    """Per-env current motion index (shape [num_envs, 1]).

    Used as the named routing term in ``teacher_obs``. The PHP routing mixin
    resolves its actual feature column from the observation manager before
    calling ``StudentTeacher.teacher_act``. When the motion file lacks a
    ``motion_idxs`` field, ``MotionLoader`` already zero-fills
    (command/terms/wbt.py:157), so single-motion runs select ``teachers[0]``.

    When the teacher's anchor / orientation / tracked-body errors exceed the
    given thresholds the env is flagged as "expert lost" and the index is
    overwritten with ``-1``; ``StudentTeacher.teacher_act`` zeros that env's
    teacher action, which ``PhpDistillationPPO._dagger_loss`` masks so the
    DAgger loss is suppressed for that sample. Mirrors far-tracking
    ``observations.py:190-197``.
    """
    cmd = env.command_manager.get_state("motion_command")
    idx = cmd.motion.motion_idxs[cmd.time_steps].to(torch.float32)

    bad_ref_pos = (
        torch.norm(cmd.ref_pos_w - cmd.robot_ref_pos_w, dim=1) > bad_ref_pos_threshold
    )

    g = gravity_vector(env)
    motion_g_b = _quat_rotate_inverse(cmd.ref_quat_w, g, w_last=True)
    robot_g_b = _quat_rotate_inverse(cmd.robot_ref_quat_w, g, w_last=True)
    bad_ref_ori = torch.abs(motion_g_b[:, 2] - robot_g_b[:, 2]) > bad_ref_ori_threshold

    body_names_to_track = list(cmd.motion_cfg.body_names_to_track)
    body_idx = torch.tensor(
        [body_names_to_track.index(n) for n in bad_motion_body_pos_body_names],
        dtype=torch.long,
        device=env.device,
    )
    body_err = torch.norm(
        cmd.body_pos_relative_w[:, body_idx] - cmd.robot_body_pos_w[:, body_idx], dim=-1
    )
    bad_motion_body_pos = torch.any(body_err > bad_motion_body_pos_threshold, dim=-1)

    expert_terminate = bad_ref_pos | bad_ref_ori | bad_motion_body_pos
    idx = torch.where(expert_terminate, torch.full_like(idx, -1.0), idx)
    return idx.view(env.num_envs, 1)
