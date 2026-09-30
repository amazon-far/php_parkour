"""Export a saved PHP depth-distillation checkpoint as an ONNX bundle.

Recover the checkpoint's exact config, apply evaluation defaults and explicit
CLI overrides, and regenerate terrain from the supplied bundles before building
the environment. Export loads the full policy with inferred teacher slots.

Usage::

    python -m wbt_training.training_runs.export_distill_onnx \\
        --training.checkpoint /path/model_20000.pt \\
        --training.registry-name file:///path/bundle \\
        --training.num-envs 2

``--checkpoint`` is an alias for ``--training.checkpoint``. Legacy ``exp:...``
labels are accepted, but the saved config always defines the starting model.
Export logic is shared with the training save hook through
``wbt_training.utils.exporter.export_depth_student_bundle``.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys

from holosoma.config_types.experiment import ExperimentConfig
from holosoma.train_agent import infer_num_teachers_from_checkpoint, init_sim_imports
from holosoma.utils.eval_utils import CheckpointConfig
from holosoma.utils.sim_utils import close_simulation_app
from loguru import logger

from wbt_training.utils.checkpoint import (
    apply_preprocess_hook,
    load_saved_experiment_config,
    parse_saved_config_overrides,
    resolve_checkpoint_reference,
)
from wbt_training.utils.exporter import export_depth_student_bundle


def _parse_export_config(
    args: list[str] | None = None,
) -> tuple[ExperimentConfig, ExperimentConfig, str | None]:
    """Read config from the requested weights before parsing model overrides."""
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--checkpoint", "--training.checkpoint", dest="checkpoint")
    checkpoint_args, remaining_args = parser.parse_known_args(args)
    if checkpoint_args.checkpoint is None:
        if any(arg in ("-h", "--help") for arg in remaining_args):
            parser.print_help()
            print("Supply a checkpoint to see its available config overrides.")
            raise SystemExit(0)
        parser.error("--training.checkpoint (or --checkpoint) is required")

    reference = resolve_checkpoint_reference(checkpoint_args.checkpoint)
    saved_config, saved_wandb_path = load_saved_experiment_config(
        CheckpointConfig(checkpoint=reference)
    )
    default = saved_config.get_eval_config()
    default = dataclasses.replace(
        default,
        training=dataclasses.replace(
            default.training, checkpoint=reference, headless=True, num_envs=1
        ),
    )
    # Old export commands selected a preset before providing a checkpoint. Keep
    # accepting the label without letting its defaults replace the saved model.
    preset_labels = [arg for arg in remaining_args if arg.startswith("exp:")]
    if preset_labels:
        logger.info(
            f"Using saved checkpoint config for legacy preset labels: {preset_labels}"
        )
        remaining_args = [arg for arg in remaining_args if not arg.startswith("exp:")]
    config = parse_saved_config_overrides(
        default,
        remaining_args,
        description="Export with overrides on the saved checkpoint config.",
    )
    return config, saved_config, saved_wandb_path


def _export(
    config: ExperimentConfig,
    saved_config: ExperimentConfig | None = None,
    saved_wandb_path: str | None = None,
) -> str | None:
    """Construct the environment and restore a full checkpoint, then export."""
    if config.training.checkpoint is None:
        raise ValueError("--training.checkpoint is required for export")

    simulation_app = init_sim_imports(config)
    try:
        import torch
        from holosoma.config_types.env import get_tyro_env_config
        from holosoma.utils.common import seeding
        from holosoma.utils.eval_utils import load_checkpoint
        from holosoma.utils.experiment_paths import get_experiment_dir, get_timestamp
        from holosoma.utils.helpers import get_class

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        seeding(
            config.training.seed,
            torch_deterministic=config.training.torch_deterministic,
        )
        experiment_dir = get_experiment_dir(
            config.logger, config.training, get_timestamp(), task_name="export"
        )
        experiment_dir.mkdir(exist_ok=True, parents=True)

        ckpt_path = load_checkpoint(config.training.checkpoint, str(experiment_dir))
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict):
            raise TypeError("Training checkpoint must contain a dictionary")
        checkpoint_teacher_count = infer_num_teachers_from_checkpoint(checkpoint)
        del checkpoint

        teacher_checkpoints = config.training.teacher_checkpoint
        teacher_paths = (
            [path.strip() for path in teacher_checkpoints.split(",") if path.strip()]
            if teacher_checkpoints is not None
            else []
        )
        if teacher_checkpoints is not None and not teacher_paths:
            raise ValueError(
                "training.teacher_checkpoint must contain at least one path"
            )
        if (
            teacher_paths
            and checkpoint_teacher_count is not None
            and len(teacher_paths) != checkpoint_teacher_count
        ):
            raise ValueError(
                f"Checkpoint contains {checkpoint_teacher_count} teacher slots, but "
                f"{len(teacher_paths)} teacher checkpoints were provided"
            )
        teacher_count = checkpoint_teacher_count or (
            len(teacher_paths) if teacher_paths else None
        )

        config = apply_preprocess_hook(config)
        env = get_class(config.env_class)(get_tyro_env_config(config), device=device)
        algo = get_class(config.algo._target_)(
            device=device,
            env=env,
            config=config.algo.config,
            log_dir=experiment_dir,
            multi_gpu_cfg=None,
        )
        if teacher_count is not None:
            if hasattr(algo, "num_teachers"):
                algo.num_teachers = teacher_count
            elif teacher_count > 1:
                raise RuntimeError(
                    f"Checkpoint requires {teacher_count} teacher slots, but algorithm "
                    f"{type(algo).__name__} has no num_teachers attribute"
                )
        algo.setup()
        if saved_config is not None:
            algo.attach_checkpoint_metadata(saved_config, saved_wandb_path)
        # Keep strict policy/backbone loading and all optimizer, teacher-readiness,
        # iteration, schedule and environment-state restoration in the core loader.
        algo.load(str(ckpt_path))
        iteration = int(getattr(algo, "current_learning_iteration", 0))
        output_dir = export_depth_student_bundle(algo, iteration)
        print(f"Exported ONNX bundle to: {output_dir}", flush=True)
        return output_dir
    finally:
        close_simulation_app(simulation_app)


def main() -> None:
    config, saved_config, saved_wandb_path = _parse_export_config(sys.argv[1:])
    _export(config, saved_config, saved_wandb_path)


if __name__ == "__main__":
    main()
