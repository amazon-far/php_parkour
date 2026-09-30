#!/usr/bin/env bash
# Run a PHP D435i student against this checkout's pinned Holosoma inference stack.
set -e

PHP_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CALLER_DIR=$PWD
HOLOSOMA_ROOT=${HOLOSOMA_ROOT:-"$PHP_ROOT/thirdparty/holosoma"}

if [ ! -f "$HOLOSOMA_ROOT/scripts/source_inference_setup.sh" ] || \
   [ ! -f "$HOLOSOMA_ROOT/src/holosoma_inference/holosoma_inference/run_policy.py" ]; then
    echo "Holosoma checkout not found at: $HOLOSOMA_ROOT" >&2
    echo "initialize it with: git submodule update --init thirdparty/holosoma" >&2
    echo "or set HOLOSOMA_ROOT=/path/to/holosoma" >&2
    exit 2
fi
HOLOSOMA_ROOT=$(cd -- "$HOLOSOMA_ROOT" && pwd)

RUN=${RUN:-}
STEP=${STEP:-model_20000}
BACKBONE=${BACKBONE:-}
STUDENT=${STUDENT:-}

if [ -n "$RUN" ]; then
    BACKBONE=${BACKBONE:-"${RUN}/${STEP}/depth_backbone.onnx"}
    STUDENT=${STUDENT:-"${RUN}/${STEP}/student.onnx"}
fi
if [ -z "$BACKBONE" ] || [ -z "$STUDENT" ]; then
    echo "set RUN=wandb://<entity>/<project>/<run_id> or both BACKBONE and STUDENT" >&2
    exit 2
fi

# Resolve local paths from the directory where the PHP launcher was invoked,
# before changing to the Holosoma checkout for its runtime and data discovery.
resolve_model_path() {
    case "$1" in
        wandb://* | https://* | /*) printf '%s\n' "$1" ;;
        *) printf '%s/%s\n' "$CALLER_DIR" "$1" ;;
    esac
}
BACKBONE=$(resolve_model_path "$BACKBONE")
STUDENT=$(resolve_model_path "$STUDENT")

for model in "$BACKBONE" "$STUDENT"; do
    case "$model" in
        wandb://* | https://*) ;;
        *) [ -f "$model" ] || { echo "missing checkpoint: $model" >&2; exit 1; } ;;
    esac
done

cd "$HOLOSOMA_ROOT"
source "$HOLOSOMA_ROOT/scripts/source_inference_setup.sh"

# Both paths belong in one Python-list CLI token; the config accepts one model
# path or a list, and the depth student requires the backbone/student pair.
python3 "$HOLOSOMA_ROOT/src/holosoma_inference/holosoma_inference/run_policy.py" \
    inference:g1-wbt-distillation-d435i \
    --task.interface lo \
    --task.model-path "['${BACKBONE}','${STUDENT}']" \
    "$@"

# For a control-loop smoke test without a depth producer, append
# --task.depth-shm.no-required.
