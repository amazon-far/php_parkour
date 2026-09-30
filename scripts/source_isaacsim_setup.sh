#!/bin/bash
# Activate the IsaacSim conda env, via holosoma's own setup script.
#
# Resolves the holosoma checkout relative to this file, so the launchers in
# wbt_training/training_runs/ don't need to know where the repo lives. Set
# HOLOSOMA_LOCAL_DIR to point at a holosoma checkout outside this repo (useful
# when developing core and the extension side by side).
#
# Source it, don't execute it:
#   source scripts/source_isaacsim_setup.sh

_resolve_script_dir() {
  local source="${BASH_SOURCE[0]:-${(%):-%x}}"
  while [ -h "$source" ]; do
    local dir
    dir="$( cd -P "$( dirname "$source" )" >/dev/null && pwd )"
    source="$(readlink "$source")"
    [[ $source != /* ]] && source="$dir/$source"
  done
  cd -P "$( dirname "$source" )" >/dev/null && pwd
}
SCRIPT_DIR="$(_resolve_script_dir)"
unset -f _resolve_script_dir

ROOT_DIR=$(dirname "$SCRIPT_DIR")
HOLOSOMA_SUBMODULE_DIR="$ROOT_DIR/thirdparty/holosoma"

if [[ -n "${HOLOSOMA_LOCAL_DIR:-}" ]]; then
    echo "[php] Using local holosoma from $HOLOSOMA_LOCAL_DIR"
    HOLOSOMA_DIR="$HOLOSOMA_LOCAL_DIR"
else
    echo "[php] Using holosoma from $HOLOSOMA_SUBMODULE_DIR"
    HOLOSOMA_DIR="$HOLOSOMA_SUBMODULE_DIR"
fi

if [ ! -d "$HOLOSOMA_DIR/scripts" ]; then
    echo "ERROR: holosoma not found at $HOLOSOMA_DIR." >&2
    echo "       Run: git submodule update --init --recursive" >&2
    return 1
fi

CONDA_ENV_NAME=${ENV_NAME:-php}
echo "[php] Using conda environment: $CONDA_ENV_NAME"
CONDA_ENV_NAME="$CONDA_ENV_NAME" source "$HOLOSOMA_DIR/scripts/source_isaacsim_setup.sh"
