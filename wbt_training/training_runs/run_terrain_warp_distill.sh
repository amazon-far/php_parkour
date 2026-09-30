#!/bin/bash
# Distill a PHP D435i student from ordered teacher/data pairs.
# Required: TEACHER_CHECKPOINT (comma-separated local .pt or wandb:// paths).
# REGISTRY defaults to auto and requires W&B teachers with usable data metadata.
# Override REGISTRY with one file:// directory or W&B artifact per teacher.
# FINETUNE=1: DAgger + PPO (default); FINETUNE=0: pure DAgger.
# WARMUP_STEPS=0, NGPUS=2; LOGGER=wandb (LOGGER=disabled keeps logging local).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/setup_env.sh"
php_launcher_init
REGISTRY="${REGISTRY:-auto}"
NGPUS="${NGPUS:-2}"
WARMUP_STEPS="${WARMUP_STEPS:-0}"
FINETUNE=$(php_boolean FINETUNE "${FINETUNE:-1}")
php_positive_integer NGPUS "$NGPUS"
[[ "$WARMUP_STEPS" =~ ^(0|[1-9][0-9]*)$ ]] || { php_error 'WARMUP_STEPS must be a nonnegative integer'; exit 1; }
if [[ "$FINETUNE" == True ]]; then
    EXP_PRESET=exp:g1-wbt-terrain-warp-distill-finetune
else
    EXP_PRESET=exp:g1-wbt-terrain-warp-distill
fi
php_prepare_inputs distill "${TEACHER_CHECKPOINT:-}"
TEACHER_CHECKPOINT="$PHP_CHECKPOINTS"
IFS=',' read -r -a PHP_TEACHERS <<< "$TEACHER_CHECKPOINT"
IFS=',' read -r -a PHP_REGISTRIES <<< "$REGISTRY"
for i in "${!PHP_TEACHERS[@]}"; do
    echo "==> [$i] teacher=${PHP_TEACHERS[$i]} registry=${PHP_REGISTRIES[$i]}"
done
php_training_command
php_select_training_gpu
php_activate_sim
exec "${TRAIN_COMMAND[@]}" -m wbt_training.train_agent \
    "$EXP_PRESET" simulator:isaacsim \
    --training.registry-name "$REGISTRY" \
    --training.teacher-checkpoint "$TEACHER_CHECKPOINT" \
    --algo.config.distillation-warmup-steps "$WARMUP_STEPS" \
    "${LOGGER_ARGS[@]}" --logger.video.enabled False \
    "$@"
