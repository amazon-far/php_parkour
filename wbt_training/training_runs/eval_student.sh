#!/bin/bash
# Evaluate a PHP depth student from its saved config.
# Required: CHECKPOINT=local.pt (or wandb://entity/project/run[/model_N.pt]),
# REGISTRY=file:///absolute/bundle[,file:///another/bundle] in training order.
# REGISTRY=auto recovers the ordered datasets from a W&B student run.
# NUM_ENVS=1, MAX_EVAL_STEPS=1000, GPU_INDEX=0, LOGGER=wandb.
# LOGGER=disabled keeps logging local.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/setup_env.sh"
php_launcher_init
NUM_ENVS="${NUM_ENVS:-1}"
MAX_EVAL_STEPS="${MAX_EVAL_STEPS:-1000}"
php_positive_integer NUM_ENVS "$NUM_ENVS"
php_positive_integer MAX_EVAL_STEPS "$MAX_EVAL_STEPS"
php_prepare_inputs eval_student "${CHECKPOINT:-}"
CHECKPOINT="$PHP_CHECKPOINTS"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU_INDEX:-0}}"
php_activate_sim
exec "$HSSIM_PYTHON" -m wbt_training.training_runs.eval_student \
    --checkpoint "$CHECKPOINT" \
    --training.num-envs "$NUM_ENVS" \
    --training.max-eval-steps "$MAX_EVAL_STEPS" \
    --training.registry-name "$REGISTRY" \
    --terrain.terrain-term.spawn.randomize-tiles True \
    "${LOGGER_ARGS[@]}" --logger.video.enabled True \
    --training.preprocess-hook-kwargs '{"add_onpath_obstacle": true, "num_variants": 3}' \
    "$@"
