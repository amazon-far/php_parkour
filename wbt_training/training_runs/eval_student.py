"""Evaluate a distillation student checkpoint in the training env.

Mirrors holosoma's ``eval_agent.py`` but tailored for distill checkpoints:

- Skips ``algo.export()``: ``Distillation`` / ``DistillationPPO`` raise
  ``NotImplementedError`` from ``actor_onnx_wrapper``. Use
  ``export_distill_onnx.py`` for ONNX bundles.
- Strips ``teachers.*`` keys from the checkpoint state dict before loading,
  then loads ``student`` and ``depth_backbone`` directly. This avoids the
  need to pass ``--training.teacher-checkpoint`` (and the ``num_teachers``
  ceremony that ``Distillation.__init__`` would otherwise demand). Inference
  only uses ``policy.student``, so the teachers are dead weight at eval.

Usage::

    python -m wbt_training.training_runs.eval_student \\
        --checkpoint wandb://<entity>/<project>/<run_id>[/model_NNNNN.pt]
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

import torch
import tyro
from holosoma.agents.base_algo.base_algo import BaseAlgo
from holosoma.config_types.eval_callback import EvalCallbacksConfig
from holosoma.config_types.experiment import ExperimentConfig
from holosoma.utils.config_utils import CONFIG_NAME
from holosoma.utils.eval_utils import (
    CheckpointConfig,
    init_eval_logging,
    load_checkpoint,
)
from holosoma.utils.experiment_paths import get_experiment_dir, get_timestamp
from holosoma.utils.helpers import get_class
from holosoma.utils.sim_utils import close_simulation_app, setup_simulation_environment
from holosoma.utils.tyro_utils import TYRO_CONIFG
from loguru import logger

from wbt_training.utils.checkpoint import (
    apply_preprocess_hook,
    parse_saved_config_overrides,
    resolve_checkpoint_reference,
)
from wbt_training.utils.checkpoint import (
    load_saved_experiment_config as _load_saved_experiment_config_prefer_pt,
)


@dataclass(frozen=True)
class DepthVizConfig:
    show_depth: bool = False
    """Live OpenCV window showing the delayed depth feed the student sees.

    Tiles all envs into one image, updated each control step. Requires a
    DISPLAY (X server reachable) and ``opencv-python`` installed in the
    hssim env. Off by default — headless runs are unaffected.
    """


def _attach_depth_viewer(algo: BaseAlgo) -> None:
    """Monkey-patch ``algo._post_eval_env_step`` to render the delayed depth.

    Pulls ``obs_dict['depth_camera']['depth_cam']`` (constants from
    ``distillation_ppo.DEPTH_GROUP/DEPTH_TERM``), the same tensor that flows
    into ``policy.depth_backbone`` — i.e. the latency-sampled view, not the
    live raycast. Values are normalized by ``WarpDepthImageObsTerm`` to
    roughly ``[-0.5, 0.5]`` so we map straight to uint8 and apply
    ``COLORMAP_TURBO``.
    """
    try:
        import cv2
    except ImportError as e:
        raise ImportError(
            "show_depth=True requires opencv-python in the hssim env: "
            "pip install opencv-python"
        ) from e
    import numpy as np

    num_envs = algo.env.num_envs
    cols = max(1, math.ceil(math.sqrt(num_envs)))
    rows = math.ceil(num_envs / cols)
    window_title = "eval_student depth (delayed)"
    warned_failure = [False]

    orig_post_eval_env_step = algo._post_eval_env_step
    orig_post_evaluate_policy = algo._post_evaluate_policy

    def _depth_post_eval_env_step(actor_state):
        actor_state = orig_post_eval_env_step(actor_state)
        try:
            depth_entry = actor_state["obs"].get("depth_camera")
            if depth_entry is None:
                return actor_state
            depth = (
                depth_entry["depth_cam"]
                if isinstance(depth_entry, dict)
                else depth_entry
            )
            arr = depth.detach().to("cpu", torch.float32).numpy()
            n, h, w = arr.shape
            grid = np.zeros((rows * h, cols * w), dtype=np.float32)
            for i in range(n):
                r, c = divmod(i, cols)
                grid[r * h : (r + 1) * h, c * w : (c + 1) * w] = arr[i]
            grid = cv2.resize(grid, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
            cv2.imshow(window_title, grid + 0.5)
            cv2.waitKey(1)
        except (cv2.error, RuntimeError, ValueError) as e:
            if not warned_failure[0]:
                logger.warning(f"depth viewer failed: {e}; disabling further frames")
                warned_failure[0] = True
        return actor_state

    def _depth_post_evaluate_policy():
        try:
            cv2.destroyAllWindows()
        except cv2.error as error:
            logger.debug(f"Could not close depth viewer: {error}")
        return orig_post_evaluate_policy()

    algo._post_eval_env_step = _depth_post_eval_env_step
    algo._post_evaluate_policy = _depth_post_evaluate_policy
    logger.info(f"Depth viewer attached: {rows}x{cols} tile grid for {num_envs} env(s)")


def _load_student_only(algo: BaseAlgo, ckpt_path: str) -> None:
    """Load ``student`` + ``depth_backbone`` weights from a distill checkpoint,
    skipping ``teachers.*``.

    Distillation training saves the whole composite policy
    (student + teachers + depth backbone) under ``model_state_dict``. At eval
    we only run ``policy.student`` (via ``act_inference``), so we skip the
    teacher tensors entirely — that sidesteps the ``num_teachers`` shape
    mismatch when the checkpoint was trained with N>1 teachers but
    ``Distillation.__init__`` defaults to ``num_teachers=1``.
    """
    logger.info(f"Loading distill student weights from {ckpt_path}")
    loaded = torch.load(ckpt_path, map_location=algo.device, weights_only=False)
    if not isinstance(loaded, dict) or not isinstance(
        loaded.get("model_state_dict"), dict
    ):
        raise TypeError(f"{ckpt_path} is not a distillation training checkpoint")
    state = loaded["model_state_dict"]

    student_state: dict[str, torch.Tensor] = {}
    backbone_state: dict[str, torch.Tensor] = {}
    skipped_teacher = 0
    skipped_inference_irrelevant = 0
    other_keys: list[str] = []
    for k, v in state.items():
        if k.startswith(("teacher.", "teachers.")):
            skipped_teacher += 1
        elif k.startswith("student."):
            student_state[k[len("student.") :]] = v
        elif k.startswith("depth_backbone."):
            backbone_state[k[len("depth_backbone.") :]] = v
        elif k == "std" or k.startswith("critic."):
            # ``std`` is only used for action sampling during training; ``critic.*``
            # is the privileged value head for PPO. ``act_inference`` touches
            # neither, so dropping them is correct.
            skipped_inference_irrelevant += 1
        else:
            other_keys.append(k)

    if not student_state:
        raise RuntimeError(
            f"No 'student.*' keys found in {ckpt_path}. Top-level keys: "
            f"{list(state.keys())[:10]}..."
        )

    if other_keys:
        raise RuntimeError(
            f"Unhandled policy state keys in {ckpt_path}: {other_keys[:10]}"
        )
    if hasattr(algo.policy, "depth_backbone") and not backbone_state:
        raise RuntimeError(f"No 'depth_backbone.*' keys found in {ckpt_path}")

    algo.policy.student.load_state_dict(student_state, strict=True)
    if hasattr(algo.policy, "depth_backbone"):
        algo.policy.depth_backbone.load_state_dict(backbone_state, strict=True)
    elif backbone_state:
        raise RuntimeError(
            "Checkpoint has a depth backbone but the configured policy does not"
        )
    logger.info(
        f"Loaded student ({len(student_state)} tensors) + "
        f"depth_backbone ({len(backbone_state)} tensors); "
        f"skipped {skipped_teacher} teacher + "
        f"{skipped_inference_irrelevant} inference-irrelevant (std/critic) tensors."
    )

    algo.current_learning_iteration = int(loaded.get("iter", 0))
    algo._restore_env_state(loaded.get("env_state"))


def run_eval(
    tyro_config: ExperimentConfig,
    checkpoint_cfg: CheckpointConfig,
    saved_config: ExperimentConfig,
    saved_wandb_path: str | None,
    eval_cbs_cfg: EvalCallbacksConfig | None = None,
    depth_viz_cfg: DepthVizConfig | None = None,
) -> None:
    # Mirror holosoma.train_agent: presets register a ``preprocess_hook`` that
    # downloads/composes registries and writes a fresh terrain OBJ before env
    # construction. The saved checkpoint's config still points at the *training*
    # run's tempfile (e.g. /tmp/tmpXXXX.obj), which no longer exists at eval
    # time. Re-running the hook regenerates one.
    tyro_config = apply_preprocess_hook(tyro_config)

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

    _load_student_only(algo, str(checkpoint))

    if depth_viz_cfg is not None and depth_viz_cfg.show_depth:
        _attach_depth_viewer(algo)

    algo.evaluate_policy(max_eval_steps=tyro_config.training.max_eval_steps)

    # Report video output paths BEFORE close_simulation_app — Isaac Sim's
    # simulation_app.close() can kill the process abruptly (it patches in a
    # noop close_stage and disables a SimulationContext callback to dodge a
    # known shutdown hang), so anything after it may never reach stdout.
    video_dir = eval_log_dir / "renderings_training"
    print(f"\n[VIDEO] eval log dir: {eval_log_dir}", flush=True)
    if video_dir.exists():
        videos = sorted(video_dir.glob("*.mp4"))
        if videos:
            print(f"[VIDEO] {len(videos)} video(s) saved to {video_dir}:", flush=True)
            for v in videos:
                print(f"[VIDEO]   {v}", flush=True)
        else:
            print(
                f"[VIDEO] {video_dir} exists but contains no .mp4 files yet", flush=True
            )
    else:
        print(
            f"[VIDEO] {video_dir} not found — video recording may not have been enabled "
            "(check --logger.video.enabled)",
            flush=True,
        )

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
    depth_viz_cfg, remaining_args = tyro.cli(
        DepthVizConfig,
        return_unknown_args=True,
        add_help=False,
        args=remaining_args,
        config=TYRO_CONIFG,
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
        depth_viz_cfg=depth_viz_cfg,
    )


if __name__ == "__main__":
    main()
