"""Evaluate a teacher PPO checkpoint (from ``run_terrain_teacher.sh``) in
the training env.

Mirrors holosoma's ``eval_agent.py`` with one WBT-specific fix that
``eval_agent.py`` doesn't apply:

- Runs the registered ``preprocess_hook`` before env construction so that
  the registry resolves and a fresh terrain OBJ is written. The saved
  checkpoint config still references the *training* run's tempfile
  (``/tmp/tmpXXXX.obj``) which no longer exists at eval time.

Unlike ``eval_student.py``, this calls plain ``algo.load(...)`` because
PPO's ``model_state_dict`` has no teacher MLPs to strip and PPO already
implements ``export()`` / ``actor_onnx_wrapper``.

Usage::

    python -m wbt_training.training_runs.eval_teacher \\
        --checkpoint wandb://<entity>/<project>/<run_id>[/model_NNNNN.pt]

Tips:
  --training.headless True               # required on remote boxes
  --training.num-envs 16                 # eval doesn't need 4096
  --training.max-eval-steps 100          # cap rollout length
  --training.export-onnx False           # skip ONNX export
  --terrain.terrain-term.spawn.randomize-tiles True
                                         # required for OBJ-loaded terrains;
                                         # the eval branch with
                                         # randomize_tiles=False reads
                                         # ``Terrain._env_origins[0,0]``
                                         # which is only populated for
                                         # procedurally-generated terrains.
"""

from __future__ import annotations

import dataclasses

import tyro
from holosoma.agents.base_algo.base_algo import BaseAlgo
from holosoma.config_types.eval_callback import EvalCallbacksConfig
from holosoma.config_types.experiment import ExperimentConfig
from holosoma.config_types.video import SphericalCameraConfig
from holosoma.utils.config_utils import CONFIG_NAME
from holosoma.utils.eval_utils import (
    CheckpointConfig,
    init_eval_logging,
    load_checkpoint,
)
from holosoma.utils.experiment_paths import get_experiment_dir, get_timestamp
from holosoma.utils.helpers import get_class
from holosoma.utils.sim_utils import close_simulation_app, setup_simulation_environment
from loguru import logger

from wbt_training.utils.checkpoint import (
    apply_preprocess_hook,
    parse_saved_config_overrides,
    resolve_checkpoint_reference,
)
from wbt_training.utils.checkpoint import (
    load_saved_experiment_config as _load_saved_experiment_config_prefer_pt,
)


def run_eval(
    tyro_config: ExperimentConfig,
    checkpoint_cfg: CheckpointConfig,
    saved_config: ExperimentConfig,
    saved_wandb_path: str | None,
    eval_cbs_cfg: EvalCallbacksConfig | None = None,
) -> None:
    tyro_config = apply_preprocess_hook(tyro_config)

    viewer = tyro_config.simulator.config.viewer
    if viewer.camera is None:
        object.__setattr__(
            tyro_config.simulator.config,
            "viewer",
            dataclasses.replace(
                viewer,
                enable_tracking=True,
                camera=SphericalCameraConfig(tracking_body_name="auto"),
            ),
        )

    env, device, simulation_app = setup_simulation_environment(tyro_config)

    eval_log_dir = get_experiment_dir(
        tyro_config.logger, tyro_config.training, get_timestamp(), task_name="eval"
    )
    eval_log_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving eval logs to {eval_log_dir}")
    tyro_config.save_config(str(eval_log_dir / CONFIG_NAME))

    if eval_cbs_cfg is not None:
        cb_configs = eval_cbs_cfg.collect_active_callbacks()
        if cb_configs:
            object.__setattr__(tyro_config.algo.config, "eval_callbacks", cb_configs)

    assert checkpoint_cfg.checkpoint is not None
    checkpoint = load_checkpoint(checkpoint_cfg.checkpoint, str(eval_log_dir))

    algo_class = get_class(tyro_config.algo._target_)
    algo: BaseAlgo = algo_class(
        device=device,
        env=env,
        config=tyro_config.algo.config,
        log_dir=str(eval_log_dir),
        multi_gpu_cfg=None,
    )
    algo.setup()
    algo.attach_checkpoint_metadata(saved_config, saved_wandb_path)
    algo.load(str(checkpoint))

    if tyro_config.training.export_onnx:
        import os

        if not hasattr(algo, "export"):
            raise AttributeError(
                f"{algo_class.__name__} has no export(); pass --training.export-onnx False"
            )
        exported_dir = os.path.join(str(eval_log_dir), "exported")
        os.makedirs(exported_dir, exist_ok=True)
        ckpt_name = os.path.basename(str(checkpoint))
        onnx_path = os.path.join(exported_dir, ckpt_name.replace(".pt", ".onnx"))
        algo.export(onnx_file_path=onnx_path)  # type: ignore[attr-defined]
        logger.info(f"Exported policy as onnx to: {onnx_path}")

    algo.evaluate_policy(max_eval_steps=tyro_config.training.max_eval_steps)

    if simulation_app:
        close_simulation_app(simulation_app)


def main() -> None:
    init_eval_logging()
    checkpoint_cfg, remaining_args = tyro.cli(
        CheckpointConfig, return_unknown_args=True, add_help=False
    )
    eval_cbs_cfg, remaining_args = tyro.cli(
        EvalCallbacksConfig,
        return_unknown_args=True,
        add_help=False,
        args=remaining_args,
    )
    checkpoint_cfg = dataclasses.replace(
        checkpoint_cfg,
        checkpoint=resolve_checkpoint_reference(checkpoint_cfg.checkpoint),
    )
    saved_cfg, saved_wandb_path = _load_saved_experiment_config_prefer_pt(
        checkpoint_cfg
    )
    eval_cfg = saved_cfg.get_eval_config()
    overwritten_tyro_config = parse_saved_config_overrides(eval_cfg, remaining_args)

    run_eval(
        overwritten_tyro_config,
        checkpoint_cfg,
        saved_cfg,
        saved_wandb_path,
        eval_cbs_cfg=eval_cbs_cfg,
    )


if __name__ == "__main__":
    main()
