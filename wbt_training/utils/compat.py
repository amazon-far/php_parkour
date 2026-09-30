"""Rewrite dotted paths in configs saved by older checkpoints.

Why this is needed
------------------
A checkpoint stores its whole ``ExperimentConfig`` through
``BaseAlgo.attach_checkpoint_metadata``, and both eval entry points prefer that
exact config from the ``.pt`` over a separately serialized W&B yaml. This lets
compatibility rewrites happen before any dotted paths are resolved or the
environment is constructed.

That means every dotted path a preset used at training time is frozen inside
the checkpoint. When a term *moves modules*, a re-export shim at the old path
is enough. But when the owning *class changes* — as with ``vel_cmd`` moving off
core's ``MotionCommand`` onto ``PhpMotionCommand`` — no shim can help: the old
config names ``holosoma...MotionCommand``, which no longer supplies the field,
so the observation raises at the first step.

Rewriting is conditional on evidence
------------------------------------
``MotionCommand`` is only upgraded when the saved config actually asks for the
``velocity_command`` observation. Rewriting unconditionally would also touch
teacher checkpoints that never used ``vel_cmd``; ``PhpMotionCommand`` is a
behavioural superset (it delegates ``reset`` when expert balancing is inactive,
and never touches ``vel_cmd`` unless the obs term asks), so that would be safe
— but silently swapping a class on configs with no need for it makes eval runs
harder to reason about, so the narrower rule is used.
"""

from __future__ import annotations

from typing import Any

# Paths that only changed module, not behaviour.
#
# The first entry moved core -> extension (vel_cmd is a php design); the rest
# moved extension -> core when the generic depth stack was promoted. Both
# directions are handled the same way: the checkpoint names a module that no
# longer holds the implementation, so the path is rewritten on load. This
# replaces the one-line re-export shims that used to sit at the old paths --
# a single table is easier to audit, and to eventually delete, than modules
# scattered across the package.
_MODULE_MOVES: dict[str, str] = {
    "holosoma.managers.observation.terms.wbt:velocity_command": (
        "wbt_training.config_values.motion_command:velocity_command"
    ),
    "holosoma.managers.observation.terms.wbt:which_motion": (
        "wbt_training.config_values.motion_command:which_motion"
    ),
    "holosoma.agents.distillation_ppo.distillation_ppo.DistillationPPO": (
        "wbt_training.utils.student_teacher.PhpDistillationPPO"
    ),
    "holosoma.agents.distillation.distillation.Distillation": (
        "wbt_training.utils.student_teacher.PhpDistillation"
    ),
    "wbt_training.config_values.depth_obs:WarpDepthImageObsTerm": (
        "holosoma.managers.observation.terms.depth:WarpDepthImageObsTerm"
    ),
    "wbt_training.utils.depth_backbone.DepthOnlyFCBackbone58x87Small": (
        "holosoma.agents.modules.depth_backbone.DepthOnlyFCBackbone58x87Small"
    ),
    "wbt_training.utils.depth_backbone.DepthOnlyFCBackbone58x87": (
        "holosoma.agents.modules.depth_backbone.DepthOnlyFCBackbone58x87"
    ),
    "wbt_training.utils.depth_backbone.RecurrentDepthBackbone": (
        "holosoma.agents.modules.depth_backbone.RecurrentDepthBackbone"
    ),
    "wbt_training.utils.warp_sensors.camera_config.d435i_depth_config.G1FlatRsD435iConfig": (
        "holosoma.sensors.warp.camera_config.d435i_depth_config.G1FlatRsD435iConfig"
    ),
    "wbt_training.utils.warp_sensors.camera_config.d435i_depth_config.RsD435iConfig": (
        "holosoma.sensors.warp.camera_config.d435i_depth_config.RsD435iConfig"
    ),
}

# Applied only when _needs_php_motion_command() finds corroborating evidence.
_CLASS_UPGRADES: dict[str, str] = {
    "holosoma.managers.command.terms.wbt:MotionCommand": (
        "wbt_training.config_values.motion_command:PhpMotionCommand"
    ),
}

_VEL_CMD_MARKERS = ("velocity_command",)

# Config keys whose value is a dotted path. ``func`` covers manager terms and
# ``_target_`` hydra-style instantiation; the other two are plain-string module
# paths resolved by get_class at runtime.
_PATH_KEYS = ("func", "_target_", "depth_backbone", "camera_sensor_cfg")


def _walk_strings(node: Any):
    """Yield ``(container, key, value)`` for every str leaf in a nested dict/list."""
    if isinstance(node, dict):
        for k, v in node.items():
            if isinstance(v, str):
                yield node, k, v
            else:
                yield from _walk_strings(v)
    elif isinstance(node, (list, tuple)):
        for i, v in enumerate(node):
            if isinstance(v, str):
                if isinstance(node, list):
                    yield node, i, v
            else:
                yield from _walk_strings(v)


def needs_php_motion_command(cfg_dict: dict) -> bool:
    """True when the saved config wires an observation that requires ``vel_cmd``."""
    return any(
        any(m in value for m in _VEL_CMD_MARKERS)
        for _container, key, value in _walk_strings(cfg_dict)
        if key in ("func", "_target_")
    )


def rewrite_legacy_paths(cfg_dict: dict) -> tuple[dict, list[str]]:
    """Rewrite stale dotted paths in a checkpoint's serialized config.

    Mutates and returns ``cfg_dict`` alongside a list of human-readable
    descriptions of what changed, so callers can log it rather than silently
    altering a config the user believes they loaded verbatim.
    """
    changes: list[str] = []
    rewrites = dict(_MODULE_MOVES)
    if needs_php_motion_command(cfg_dict):
        rewrites.update(_CLASS_UPGRADES)

    for container, key, value in list(_walk_strings(cfg_dict)):
        if key not in _PATH_KEYS:
            continue
        new = rewrites.get(value)
        if new is not None:
            container[key] = new
            changes.append(f"{key}: {value} -> {new}")
    return cfg_dict, changes
