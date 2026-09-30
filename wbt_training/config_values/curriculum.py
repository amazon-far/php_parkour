"""Curriculum term that relaxes termination thresholds over training.

Port of far-tracking's ``relax_termination``. Loosens any ``*_threshold``
params on the configured termination terms between ``curriculum_start_step``
and ``curriculum_end_step`` so the student survives longer early in training,
then returns to the original values for realistic evaluation later.

Holosoma's WBT termination stack uses a single ``bad_tracking`` term whose
params are ``bad_ref_pos_threshold``, ``bad_ref_ori_threshold``,
``bad_motion_body_pos_threshold``, etc. (each a separate sub-threshold). This
curriculum scans each target term's ``params`` dict for *any* key ending in
``_threshold`` or exactly named ``threshold``, stashes the originals on
``setup()``, and rewrites them in-place on every ``step()``.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from holosoma.managers.curriculum.base import CurriculumTermBase


def _threshold_keys(params: dict[str, Any]) -> list[str]:
    """Return the subset of ``params`` keys that look like scalar thresholds."""
    keys: list[str] = []
    for k, v in params.items():
        if k == "threshold" or k.endswith("_threshold"):
            if isinstance(v, (int, float)):
                keys.append(k)
    return keys


class RelaxTerminationThresholds(CurriculumTermBase):
    """Relax termination thresholds linearly over a step window.

    Parameters via ``cfg.params``:
    - ``term_names: list[str]`` — names of termination terms to target. If a
      name isn't registered in the env's TerminationManager it's silently
      skipped. Defaults to ``["bad_tracking"]``.
    - ``curriculum_start_step: int`` — ``common_step_counter`` at which
      relaxation begins (before: thresholds stay at original).
    - ``curriculum_end_step: int`` — step at which relaxation reaches its full
      value (after: thresholds pinned at ``target_multiplier * original``).
    - ``target_multiplier: float`` — final threshold = ``target_multiplier *
      original``. Use >1 to loosen.
    - ``decay_type: str`` — ``linear`` (default), ``exp``, or ``cosine``.

    Thresholds are identified by key name: exact ``threshold`` or any
    ``*_threshold``. Every matching scalar gets scaled by the same progress
    factor (so all three BadTracking sub-thresholds move together).
    """

    DEFAULT_TERMS = ("bad_tracking",)

    def __init__(self, cfg: Any, env: Any):
        super().__init__(cfg, env)
        params = cfg.params or {}
        self.term_names: list[str] = list(params.get("term_names") or self.DEFAULT_TERMS)
        self.start_step: int = int(params.get("curriculum_start_step", 24_000))
        self.end_step: int = int(params.get("curriculum_end_step", 240_000))
        self.target_multiplier: float = float(params.get("target_multiplier", 2.0))
        self.decay_type: str = str(params.get("decay_type", "linear")).lower()
        if self.decay_type not in {"linear", "exp", "cosine"}:
            raise ValueError(f"decay_type must be linear/exp/cosine, got {self.decay_type!r}")
        # {term_name: {threshold_key: original_value}}
        self.original_thresholds: dict[str, dict[str, float]] = {}

    def setup(self) -> None:
        if not hasattr(self.env, "termination_manager"):
            return
        mgr = self.env.termination_manager
        for name in self.term_names:
            if name not in mgr._term_names:
                continue
            idx = mgr._term_names.index(name)
            term_cfg = mgr._term_cfgs[idx]
            keys = _threshold_keys(term_cfg.params)
            if not keys:
                continue
            self.original_thresholds[name] = {k: float(term_cfg.params[k]) for k in keys}

        if not self.original_thresholds:
            # Make the silent-no-op case visible when debugging a preset.
            import logging
            logging.getLogger(__name__).warning(
                "RelaxTerminationThresholds found no matching thresholds on terms=%s; "
                "nothing will be relaxed.",
                self.term_names,
            )

    def reset(self, env_ids) -> None:
        return

    def step(self) -> None:
        if not self.original_thresholds or not hasattr(self.env, "termination_manager"):
            return
        current_step = int(getattr(self.env, "common_step_counter", 0))
        progress = self._progress(current_step)
        factor = 1.0 + progress * (self.target_multiplier - 1.0)
        mgr = self.env.termination_manager
        for name, orig_map in self.original_thresholds.items():
            idx = mgr._term_names.index(name)
            # Mutate both the config params (so YAML snapshots are accurate)
            # AND the termination-term instance's cached attributes. Classes like
            # holosoma.managers.termination.terms.wbt:BadTracking copy thresholds
            # into self.* in __init__ and read from self.* in __call__, so
            # updating only params has no effect on the live comparisons.
            params = mgr._term_cfgs[idx].params
            instance = mgr._term_instances.get(name)
            for k, orig in orig_map.items():
                new_val = orig * factor
                params[k] = new_val
                if instance is not None and hasattr(instance, k):
                    setattr(instance, k, new_val)

        if hasattr(self.env, "log_dict"):
            self.env.log_dict["termination_curriculum_progress"] = torch.tensor(
                progress, dtype=torch.float
            )

    def _progress(self, step: int) -> float:
        if step <= self.start_step:
            return 0.0
        if step >= self.end_step:
            return 1.0
        span = max(self.end_step - self.start_step, 1)
        x = (step - self.start_step) / span
        if self.decay_type == "linear":
            return x
        if self.decay_type == "cosine":
            return 0.5 * (1.0 - math.cos(math.pi * x))
        # exp: smooth exponential ease-in (matches far-tracking's convention).
        return 1.0 - math.exp(-4.0 * x)
