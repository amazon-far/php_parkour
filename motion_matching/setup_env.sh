#!/usr/bin/env bash
# Create an isolated environment for stage-1 motion matching.
#
# Usage:
#   bash motion_matching/setup_env.sh
#
# Overrides:
#   ENV_NAME=<name>          conda env name (default: php-motion-matching)
#   PYTHON_VERSION=<version> Python version (default: 3.10, stage-1 parity runtime)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${ENV_NAME:-php-motion-matching}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"

if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda is required but was not found on PATH" >&2
    exit 1
fi

if conda run -n "$ENV_NAME" python --version >/dev/null 2>&1; then
    echo "[setup] Reusing conda env: $ENV_NAME"
else
    echo "[setup] Creating conda env: $ENV_NAME (Python $PYTHON_VERSION)"
    conda create --yes --name "$ENV_NAME" "python=$PYTHON_VERSION" pip \
        --override-channels --channel conda-forge
fi

echo "[setup] Installing motion matching and its test dependencies"
conda run -n "$ENV_NAME" python -m pip install --upgrade pip
conda run -n "$ENV_NAME" python -m pip install --editable "$SCRIPT_DIR[test]"

echo "[setup] Done. Activate with:"
echo "    conda activate $ENV_NAME"
