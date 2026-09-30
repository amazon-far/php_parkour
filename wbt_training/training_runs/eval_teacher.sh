#!/bin/bash
# Evaluate a PHP teacher from its saved config.
# Required: CHECKPOINT=local.pt (or wandb://entity/project/run[/model_N.pt]),
# REGISTRY=file:///absolute/bundle[,file:///another/bundle].
# Explicit W&B artifacts and REGISTRY=auto with a W&B teacher are supported.
# NUM_ENVS=1, MAX_EVAL_STEPS=10000, EXPORT_ONNX=False, GPU_INDEX=0.
# LOGGER=wandb by default; LOGGER=disabled keeps logging local.
# Extra CLI args override evaluation defaults.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/setup_env.sh"
php_launcher_init
NUM_ENVS="${NUM_ENVS:-1}"
MAX_EVAL_STEPS="${MAX_EVAL_STEPS:-10000}"
EXPORT_ONNX=$(php_boolean EXPORT_ONNX "${EXPORT_ONNX:-False}")
php_positive_integer NUM_ENVS "$NUM_ENVS"
php_positive_integer MAX_EVAL_STEPS "$MAX_EVAL_STEPS"
php_prepare_inputs eval_teacher "${CHECKPOINT:-}"
CHECKPOINT="$PHP_CHECKPOINTS"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU_INDEX:-0}}"
php_activate_sim
exec "$HSSIM_PYTHON" -m wbt_training.training_runs.eval_teacher \
    --checkpoint "$CHECKPOINT" \
    --training.num-envs "$NUM_ENVS" \
    --training.max-eval-steps "$MAX_EVAL_STEPS" \
    --training.export-onnx "$EXPORT_ONNX" \
    --training.registry-name "$REGISTRY" \
    --terrain.terrain-term.spawn.randomize-tiles True \
    "${LOGGER_ARGS[@]}" --logger.video.enabled True \
    --training.preprocess-hook-kwargs '{"add_onpath_obstacle": true, "num_variants": 1}' \
    "$@"
