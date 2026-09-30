"""Routing-column student/teacher modules and the matching DAgger masking.

Core's :class:`~holosoma.agents.modules.student_teacher_modules.StudentTeacher`
routes across teachers via an explicit ``teacher_idx`` argument and applies a
plain behaviour-cloning loss. This project layers two conventions on top,
inherited from far-tracking so that runs remain comparable with it:

1. **The routing index rides in the observation.** One column of ``teacher_obs``
   holds the per-env motion index (produced by the ``which_motion`` term); the
   remaining columns are what the teacher MLPs were pretrained on. Passing it
   in-band avoids threading a second tensor through the env → algo boundary.

   Which column that is follows from the observation manager, which concatenates
   a group in **sorted** term order — so ``which_motion`` sits wherever its name
   sorts (last, for the terrain presets), not necessarily first.
   :func:`routing_column_index` derives it by name at policy-build time and
   :func:`_bind_routing_column` binds it onto the module. Nothing downstream can
   detect a wrong column — the width still matches — so a hard-coded offset
   would silently route every env to teacher 0 and hand the teachers a shifted
   input vector.

2. **A negative index means "the expert is lost here".** ``which_motion``
   overwrites the index with ``-1`` when the teacher's own tracking error
   exceeds threshold. Core turns that into a zero action; the mask below then
   drops those rows from the DAgger loss, so the student never regresses toward
   a target its teacher had already failed to hit.

Note the zero action is not purely a signal: during ``distillation_warmup_steps``
the teacher's actions drive the robot, so flagged envs are commanded to zero for
those iterations. That is upstream behaviour, preserved here deliberately.
"""

from __future__ import annotations

import functools
import sys
from typing import Callable

import torch

from holosoma.agents.distillation.distillation import Distillation
from holosoma.agents.distillation_ppo.distillation_ppo import DistillationPPO
from holosoma.agents.modules.student_teacher_modules import (
    DepthStudentTeacher,
    DepthStudentTeacherCritic,
)


class RoutingColumnTeacherMixin:
    """Read the teacher-routing index from a named column of ``teacher_obs``.

    Reserves one observation column, so the teacher MLPs see one fewer input
    than the group's declared width.

    The column is located by term name via :func:`routing_column_index`, not
    assumed to be column 0: the observation manager concatenates a group in
    sorted term order, so ``which_motion`` lands wherever its name sorts (last,
    for the terrain presets). Deriving the offset keeps this correct under any
    ordering and under later term additions, and keeps the teachers' remaining
    input identical to what they were pretrained on.
    """

    def __init__(self, *args, num_teacher_obs: int, routing_column: int = 0, **kwargs):
        # Declare the width the teachers actually consume; the routing column is
        # stripped in teacher_act before it reaches them.
        super().__init__(*args, num_teacher_obs=num_teacher_obs - 1, **kwargs)
        self.num_teacher_obs_with_routing = num_teacher_obs
        self.routing_column = routing_column

    @torch.no_grad()
    def teacher_act(
        self, teacher_obs: torch.Tensor, teacher_idx: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Split the routing column off ``teacher_obs``, then defer to core.

        ``teacher_idx`` is ignored: the index is taken from the observation, so
        that callers in core need no knowledge of this convention.
        """
        if teacher_obs.shape[-1] != self.num_teacher_obs_with_routing:
            raise RuntimeError(
                f"teacher_act: expected last dim {self.num_teacher_obs_with_routing} "
                f"(privileged obs plus one routing column), got "
                f"{teacher_obs.shape[-1]}"
            )
        col = self.routing_column
        routing = teacher_obs[:, col]
        # Drop just the routing column; the rest keeps its relative order, which is
        # exactly the vector the pretrained teachers consumed.
        if col == 0:
            privileged = teacher_obs[:, 1:]
        elif col == teacher_obs.shape[-1] - 1:
            privileged = teacher_obs[:, :-1]
        else:
            privileged = torch.cat(
                [teacher_obs[:, :col], teacher_obs[:, col + 1 :]], dim=-1
            )
        return super().teacher_act(privileged, routing)


class RoutingColumnDepthStudentTeacherCritic(
    RoutingColumnTeacherMixin, DepthStudentTeacherCritic
):
    """``DepthStudentTeacherCritic`` with in-observation teacher routing."""


class RoutingColumnDepthStudentTeacher(RoutingColumnTeacherMixin, DepthStudentTeacher):
    """``DepthStudentTeacher`` (no critic) with in-observation teacher routing."""


def _bind_routing_column(module_cls: type, env) -> Callable[..., object]:
    """Bind the resolved routing column onto ``module_cls``.

    Core's ``setup_*_student_teacher_module`` helpers forward only their own fixed
    kwargs to ``module_cls``, so the column is bound here instead of threaded through
    them — that keeps the routing convention entirely on this side of the boundary,
    as core's generic module has no notion of it.
    """
    return functools.partial(module_cls, routing_column=routing_column_index(env))


class _ExportOnSaveMixin:
    """Export the deployable student bundle after a core checkpoint save."""

    def save(self, path: str | None = None, name: str = "last.ckpt") -> None:
        super().save(path=path, name=name)

        from wbt_training.utils.exporter import export_depth_student_bundle

        try:
            export_depth_student_bundle(self, self.current_learning_iteration)
        except Exception as error:
            print(
                f"[WARN] Failed to export depth student ONNX at iter "
                f"{self.current_learning_iteration}: {error}",
                file=sys.stderr,
            )


class PhpDistillation(_ExportOnSaveMixin, Distillation):
    """Pure-DAgger ``Distillation`` with in-observation teacher routing.

    Note this deliberately keeps upstream's unmasked loss: rows whose expert was
    lost are zeroed by ``teacher_act`` and still regressed against, which trains
    the student toward a zero action there. That matches far-tracking's
    ``DepthDistillation``, so the two remain comparable — see the module
    docstring.
    """

    def _build_policy(self) -> None:
        from holosoma.agents.modules.module_utils import (
            setup_depth_student_teacher_module,
        )

        self.policy = setup_depth_student_teacher_module(
            num_actor_obs=self.actor_obs_dim,
            num_teacher_obs=self.teacher_obs_dim,
            num_actions=self.num_act,
            module_config=self.config.module,
            device=self.device,
            init_noise_std=self.config.init_noise_std,
            num_teachers=self.num_teachers,
            module_cls=_bind_routing_column(RoutingColumnDepthStudentTeacher, self.env),
        )


class PhpDistillationPPO(_ExportOnSaveMixin, DistillationPPO):
    """``DistillationPPO`` that skips samples whose expert was lost.

    ``teacher_act`` zeroes the action of any env whose routing index is
    negative; those rows are dropped from the behaviour-cloning loss here.
    """

    def _build_policy(self) -> None:
        from holosoma.agents.modules.module_utils import setup_student_teacher_module

        self.policy = setup_student_teacher_module(
            num_actor_obs=self.actor_obs_dim,
            num_teacher_obs=self.teacher_obs_dim,
            num_critic_obs=self.critic_obs_dim,
            num_actions=self.num_act,
            module_config=self.config.module,
            device=self.device,
            init_noise_std=self.config.init_noise_std,
            num_teachers=self.num_teachers,
            module_cls=_bind_routing_column(
                RoutingColumnDepthStudentTeacherCritic, self.env
            ),
        )

    def _dagger_loss_per_sample(
        self, student_mean: torch.Tensor, teacher_actions: torch.Tensor
    ) -> torch.Tensor:
        """Per-sample MSE against the teacher, zeroing rows it marked as lost.

        A multiplicative mask keeps the autograd graph intact. Masked rows
        contribute zero rather than being removed, so the batch mean's
        denominator stays the full batch and the reported value matches
        far-tracking's ``my_distillation.py:1021-1035``.

        Note this is why the reported loss falls as a skill dies: a masked row
        contributes 0 here, and once the student stops succeeding at a phase the
        expert is marked lost across it.
        """
        raw = self.distill_loss_fn(student_mean, teacher_actions, reduction="none").mean(dim=-1)
        expert_lost = torch.all(teacher_actions == 0.0, dim=-1)
        return raw * (~expert_lost).float()

    def _update_ppo_gate_stats(
        self, student_mean: torch.Tensor, teacher_actions: torch.Tensor
    ) -> None:
        """Feed the per-frame DAgger residual to the motion command's gate EMA.

        Uses the *raw* per-sample residual and an explicit keep mask rather than
        ``_dagger_loss_per_sample``: that method already multiplies masked rows by
        zero, which would make a failing phase look perfectly imitated and drive
        its gate to the floor.
        """
        cmd = self.env.command_manager.get_state("motion_command")
        obs = getattr(cmd, "observe_dagger_residual", None)
        if obs is None:
            return
        resid = ((student_mean - teacher_actions) ** 2).mean(dim=-1)
        keep = ~torch.all(teacher_actions == 0.0, dim=-1)
        obs(resid.detach(), keep.detach())

    def _get_ppo_gate(self) -> torch.Tensor:
        """Per-env PPO gate from the motion command's precomputed table.

        Falls back to core's all-ones (scalar-``ppo_coef``) behaviour whenever the
        command does not expose a gate, so enabling this is opt-in via
        ``command.setup_terms.motion_command.params.ppo_gate_enabled``.
        """
        cmd = self.env.command_manager.get_state("motion_command")
        gate = getattr(cmd, "ppo_gate", None)
        if gate is None:
            return super()._get_ppo_gate()
        return gate.view(-1, 1).to(self.device)


ROUTING_TERM = "which_motion"


def routing_column_index(env, term_name: str = ROUTING_TERM) -> int:
    """Column of ``teacher_obs`` holding the routing index, located by term name.

    ``ObservationManager.compute_group`` concatenates a group in sorted term order,
    so the routing term's span starts after the summed widths of every term whose
    name sorts ahead of it. Widths come from the live env (a term's shape is set by
    its function, not its config), so this needs a built env rather than just a
    config — note the offset is a FEATURE column, not the term's rank: with the
    terrain presets ``which_motion`` has rank 9 but starts at column 449.

    Nothing downstream can detect a wrong column: the width still matches, so
    training silently routes every env to teacher 0 and feeds the teacher MLPs a
    shifted vector. Hence deriving it here rather than hard-coding an offset.

    Parameters
    ----------
    env : BaseTask
        Built env, for its ``observation_manager``.
    term_name : str
        Routing term name. Defaults to ``which_motion``.

    Returns
    -------
    int
        Zero-based column index, or ``0`` when there is no ``teacher_obs`` group
        (no routing in play, so the value is unused).

    Raises
    ------
    ValueError
        If ``teacher_obs`` has no such term, or the term is not 1 column wide (it
        is read as a per-env scalar).
    """
    obs_manager = env.observation_manager
    group_cfg = obs_manager.cfg.groups.get("teacher_obs")
    if group_cfg is None:
        return 0

    order = sorted(group_cfg.terms)
    if term_name not in order:
        raise ValueError(
            f"teacher_obs has no '{term_name}' term, so the teacher-routing index has "
            f"no source column. Present terms: {order}."
        )

    history = max(1, group_cfg.history_length)
    start = 0
    for name in order:
        width = (
            obs_manager._compute_term("teacher_obs", name, group_cfg.terms[name]).shape[
                1
            ]
            * history
        )
        if name == term_name:
            if width != 1:
                raise ValueError(
                    f"'{term_name}' must be exactly 1 column wide to serve as the teacher-routing "
                    f"index (it is read as a per-env scalar), but it is {width} wide."
                )
            return start
        start += width
    raise AssertionError("unreachable: term_name was checked to be in order")
