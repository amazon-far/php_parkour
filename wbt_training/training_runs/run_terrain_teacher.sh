#!/bin/bash
# Train the PHP height-scan teacher with unit action scales.
# Required: REGISTRY=file:///absolute/bundle[,file:///another/bundle].
# Defaults: LOGGER=wandb, NGPUS=1; LOGGER=disabled keeps logging local.
# Optional: WANDB_PROJECT, WANDB_ENTITY, RUN_NAME.
# Explicit W&B artifacts (entity/project/artifact:version) are also accepted.
# Additional CLI arguments are passed through to wbt_training.train_agent.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/setup_env.sh"
php_launcher_init
NGPUS="${NGPUS:-1}"
php_positive_integer NGPUS "$NGPUS"
php_prepare_inputs teacher ''
php_training_command
php_select_training_gpu
php_activate_sim
exec "${TRAIN_COMMAND[@]}" -m wbt_training.train_agent \
    exp:g1-wbt-terrain-ref-noscale simulator:isaacsim \
    --training.registry-name "$REGISTRY" \
    "${LOGGER_ARGS[@]}" --logger.video.enabled False \
    "$@"
