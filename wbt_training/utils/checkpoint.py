"""Checkpoint recovery shared by PHP evaluation and depth export.

Recover the saved configuration before constructing an environment, including
the narrow PHP path migrations. Training resume continues to use the algorithm's
full ``load``; student-only loading is reserved for evaluation.
"""

from __future__ import annotations

import copy
import dataclasses
import json
from pathlib import Path

import tyro
from holosoma.config_types.algo import DistillationAlgoConfig, DistillationPPOAlgoConfig
from holosoma.config_types.experiment import ExperimentConfig
from holosoma.utils.config_utils import CONFIG_NAME
from holosoma.utils.eval_utils import CheckpointConfig, _parse_wandb_reference
from holosoma.utils.file_cache import get_cached_file_path
from holosoma.utils.tyro_utils import TYRO_CONIFG
from loguru import logger

from wbt_training.utils.compat import rewrite_legacy_paths


def resolve_checkpoint_reference(checkpoint: str | None) -> str:
    """Pin a latest-run URI once so configuration and weights use the same file."""
    if not checkpoint:
        raise ValueError("No checkpoint provided")
    if not checkpoint.startswith("wandb://"):
        return str(Path(checkpoint).expanduser())

    run_path, artifact_path = _parse_wandb_reference(checkpoint)
    if artifact_path is None:
        from holosoma.utils.wandb_registry import get_wandb_run_and_file

        _, run_path, artifact_path = get_wandb_run_and_file(run_path)
    return f"wandb://{run_path}/{artifact_path}"


def load_saved_experiment_config(
    checkpoint_cfg: CheckpointConfig,
) -> tuple[ExperimentConfig, str | None]:
    """Prefer embedded .pt config; fall back to legacy YAML with the same rewrites.

    YAML is read from the checkpoint directory or the explicitly requested W&B
    run. A present but invalid embedded config fails instead of silently using
    a different architecture from a preset or sidecar.
    """
    import torch
    import yaml

    reference = resolve_checkpoint_reference(checkpoint_cfg.checkpoint)
    is_wandb = reference.startswith("wandb://")
    stored_wandb_path = None
    config_data = None
    try:
        checkpoint_path = get_cached_file_path(reference) if is_wandb else reference
        contents = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception:
        # Local file errors should remain actionable. Remote legacy runs may
        # have a readable config even when .pt recovery is unavailable.
        if not is_wandb:
            raise
        logger.warning(
            f"Could not read {reference!r} for config recovery; trying run YAML"
        )
    else:
        if not isinstance(contents, dict):
            raise TypeError("Training checkpoint must contain a dictionary")
        config_data = contents.get("experiment_config")
        stored_wandb_path = contents.get("wandb_run_path")
        if config_data is not None and not isinstance(config_data, dict):
            raise TypeError("Checkpoint experiment_config must be a dictionary")

    if config_data is None:
        if is_wandb:
            run_path, _ = _parse_wandb_reference(reference)
            config_path = Path(
                get_cached_file_path(f"wandb://{run_path}/{CONFIG_NAME}")
            )
            stored_wandb_path = stored_wandb_path or run_path
        else:
            config_path = Path(reference).parent / CONFIG_NAME
        logger.warning(
            f"Checkpoint has no readable experiment_config; loading {config_path}"
        )
        with config_path.open() as stream:
            config_data = yaml.safe_load(stream)
        if not isinstance(config_data, dict):
            raise TypeError("Saved experiment YAML must contain a dictionary")
    else:
        logger.info("Loaded exact saved experiment config from .pt")

    config_data, changes = rewrite_legacy_paths(copy.deepcopy(config_data))
    for change in changes:
        logger.info(f"Rewrote legacy checkpoint path: {change}")
    # Pydantic's AlgoConfig union tries the PPO distillation wrapper first and
    # accepts pure-DAgger dictionaries there too. Select by the actual target
    # before union validation, preserving every serialized recipe parameter.
    algo_data = config_data.get("algo")
    if isinstance(algo_data, dict):
        algo_type = {
            "wbt_training.utils.student_teacher.PhpDistillation": DistillationAlgoConfig,
            "wbt_training.utils.student_teacher.PhpDistillationPPO": DistillationPPOAlgoConfig,
        }.get(algo_data.get("_target_"))
        if algo_type is not None:
            config_data["algo"] = algo_type(**algo_data)
    return ExperimentConfig(**config_data), stored_wandb_path


def parse_saved_config_overrides(
    default: ExperimentConfig,
    args: list[str],
    *,
    description: str = "Override the saved checkpoint config.",
) -> ExperimentConfig:
    """Expose the saved algorithm's fields even when it is absent from the core CLI registry."""
    # The core registry's algo menu contains PPO/FastSAC, while PHP's saved
    # policies use their own distillation classes. Bind the concrete saved type
    # for this parse only; leave simulator and other preset menus available.
    config_type = dataclasses.make_dataclass(
        "CheckpointExperimentConfig",
        [("algo", type(default.algo), dataclasses.field(default=default.algo))],
        bases=(ExperimentConfig,),
        frozen=True,
    )
    fields = {
        field.name: getattr(default, field.name)
        for field in dataclasses.fields(default)
    }
    parsed = tyro.cli(
        config_type,
        default=config_type(**fields),
        args=args,
        description=description,
        config=TYRO_CONIFG,
    )
    # Return the stable public config class, keeping already typed nested values.
    return ExperimentConfig(
        **{
            field.name: getattr(parsed, field.name)
            for field in dataclasses.fields(parsed)
        }
    )


def apply_preprocess_hook(config: ExperimentConfig) -> ExperimentConfig:
    """Rebuild terrain/motion scratch assets after explicit config overrides."""
    if not config.training.preprocess_hook:
        return config
    from holosoma.managers.utils import resolve_callable

    hook = resolve_callable(config.training.preprocess_hook, context="preprocess_hook")
    kwargs = json.loads(config.training.preprocess_hook_kwargs or "{}")
    # Terrain preprocessing updates nested depth-term params. Keep the saved
    # checkpoint metadata separate from regenerated eval/export scratch paths.
    config = hook(copy.deepcopy(config), **kwargs)
    logger.info(f"Applied preprocess_hook={config.training.preprocess_hook!r}")
    return config
