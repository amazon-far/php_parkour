#!/bin/bash
# Execute to bootstrap PHP; source to load the shared launcher helpers.
# ENV_NAME defaults to php. CONDA_ROOT, HSSIM_PYTHON and HOLOSOMA_LOCAL_DIR
# may point at an existing installation.

php_error() {
    echo "ERROR: $*" >&2
    return 1
}

php_launcher_init() {
    PHP_TRAINING_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PHP_REPO_ROOT="$(cd "$PHP_TRAINING_DIR/../.." && pwd)"
    CONDA_ROOT="${CONDA_ROOT:-$HOME/.holosoma_deps/miniconda3}"
    HSSIM_PYTHON="${HSSIM_PYTHON:-$CONDA_ROOT/envs/${ENV_NAME:-php}/bin/python}"
    LOGGER="${LOGGER:-wandb}"
    case "$LOGGER" in
        disabled) LOGGER_ARGS=(logger:disabled) ;;
        wandb)
            LOGGER_ARGS=(logger:wandb)
            [[ -z "${WANDB_PROJECT:-}" ]] || LOGGER_ARGS+=(--logger.project "$WANDB_PROJECT")
            [[ -z "${WANDB_ENTITY:-}" ]] || LOGGER_ARGS+=(--logger.entity "$WANDB_ENTITY")
            [[ -z "${RUN_NAME:-}" ]] || LOGGER_ARGS+=(--logger.name "$RUN_NAME")
            ;;
        *) php_error "LOGGER must be disabled or wandb (got '$LOGGER')"; return 1 ;;
    esac
}

php_positive_integer() {
    [[ "$2" =~ ^[1-9][0-9]*$ ]] || php_error "$1 must be a positive integer (got '$2')"
}

php_boolean() {
    case "$2" in
        1|true|True|TRUE|yes|Yes|YES) printf 'True\n' ;;
        0|false|False|FALSE|no|No|NO) printf 'False\n' ;;
        *) php_error "$1 must be true/false or 1/0 (got '$2')" ;;
    esac
}

php_prepare_inputs() {
    # Validate every explicit local input before any download, GPU query or
    # simulator setup. Python arguments are data, never interpolated into code.
    local mode="$1" checkpoints="$2" prepared
    [[ -n "${REGISTRY:-}" ]] || { php_error 'REGISTRY is required (file:///absolute/bundle or an explicit W&B artifact)'; return 1; }
    if [[ "$mode" != teacher && -z "$checkpoints" ]]; then
        if [[ "$mode" == distill ]]; then
            php_error 'TEACHER_CHECKPOINT is required (local .pt path or wandb://entity/project/run[/model_N.pt])'
        else
            php_error 'CHECKPOINT is required (local .pt path or wandb://entity/project/run[/model_N.pt])'
        fi
        return 1
    fi
    [[ -x "$HSSIM_PYTHON" ]] || { php_error "$HSSIM_PYTHON not found; set HSSIM_PYTHON or run setup_env.sh"; return 1; }
    prepared=$("$HSSIM_PYTHON" - "$mode" "$checkpoints" "$REGISTRY" "$PHP_REPO_ROOT" <<'PYTHON'
import sys

# Prefer this checkout even when invoked from another working directory.
sys.path.insert(0, sys.argv.pop())
from wbt_training.utils.registry import main

raise SystemExit(main(sys.argv[1:]))
PYTHON
    ) || return 1
    {
        IFS= read -r PHP_CHECKPOINTS
        IFS= read -r REGISTRY
    } <<< "$prepared"
    echo "==> REGISTRY=$REGISTRY"
}

php_activate_sim() {
    export OMNI_KIT_ACCEPT_EULA=YES
    cd "$PHP_REPO_ROOT"
    # shellcheck disable=SC1091
    source "$PHP_REPO_ROOT/scripts/source_isaacsim_setup.sh"
}

php_training_command() {
    php_positive_integer NGPUS "$NGPUS" || return 1
    if [[ "$NGPUS" -gt 1 ]]; then
        local torchrun="$(dirname "$HSSIM_PYTHON")/torchrun"
        [[ -x "$torchrun" ]] || { php_error "torchrun not found alongside $HSSIM_PYTHON"; return 1; }
        TRAIN_COMMAND=("$torchrun" --standalone --nnodes=1 "--nproc_per_node=$NGPUS")
    else
        TRAIN_COMMAND=("$HSSIM_PYTHON")
    fi
}

php_select_training_gpu() {
    if [[ "$NGPUS" -eq 1 && -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        command -v nvidia-smi >/dev/null || { php_error 'nvidia-smi not found; set CUDA_VISIBLE_DEVICES explicitly'; return 1; }
        CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.used \
            --format=csv,noheader,nounits | sort -t',' -k2 -n | head -n1 | awk -F',' '{gsub(/ /,""); print $1}')
        [[ -n "$CUDA_VISIBLE_DEVICES" ]] || { php_error 'No GPU found by nvidia-smi'; return 1; }
        export CUDA_VISIBLE_DEVICES
        echo "Auto-selected GPU $CUDA_VISIBLE_DEVICES (lowest memory used)"
    fi
}

php_setup_environment() {
    php_launcher_init
    local holosoma_dir="${HOLOSOMA_LOCAL_DIR:-$PHP_REPO_ROOT/thirdparty/holosoma}"
    cd "$PHP_REPO_ROOT"
    if [[ -z "${HOLOSOMA_LOCAL_DIR:-}" ]]; then
        git submodule update --init --recursive
    fi
    [[ -f "$holosoma_dir/scripts/setup_isaacsim.sh" ]] || { php_error "Holosoma setup not found in $holosoma_dir"; return 1; }
    ENV_NAME="${ENV_NAME:-php}" bash "$holosoma_dir/scripts/setup_isaacsim.sh"
    php_activate_sim
    "$HSSIM_PYTHON" -m pip install -e "$PHP_REPO_ROOT"
    echo '[setup] Done. Set REGISTRY=file:///absolute/bundle and run wbt_training/training_runs/run_terrain_teacher.sh.'
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    set -euo pipefail
    php_setup_environment "$@"
fi
