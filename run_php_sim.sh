#!/usr/bin/env bash
# Run the PHP D435i sim2sim producer against this checkout's pinned Holosoma.
set -e

PHP_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
HOLOSOMA_ROOT=${HOLOSOMA_ROOT:-"$PHP_ROOT/thirdparty/holosoma"}

if [ ! -f "$HOLOSOMA_ROOT/scripts/source_mujoco_setup.sh" ] || \
   [ ! -f "$HOLOSOMA_ROOT/src/holosoma/holosoma/run_sim.py" ]; then
    echo "Holosoma checkout not found at: $HOLOSOMA_ROOT" >&2
    echo "initialize it with: git submodule update --init thirdparty/holosoma" >&2
    echo "or set HOLOSOMA_ROOT=/path/to/holosoma" >&2
    exit 2
fi
HOLOSOMA_ROOT=$(cd -- "$HOLOSOMA_ROOT" && pwd)
cd "$HOLOSOMA_ROOT"

source "$HOLOSOMA_ROOT/scripts/source_mujoco_setup.sh"

if [ -z "${DISPLAY:-}" ]; then
    echo "[run_php_sim] DISPLAY is unset — the MuJoCo window will not open and the" >&2
    echo "              gantry keys (7/8/9) will be unavailable. Export DISPLAY to fix." >&2
fi

# A stale block from a crashed run has the wrong size if the resolution changed.
rm -f /dev/shm/depth_img_shm 2>/dev/null || true

python "$HOLOSOMA_ROOT/src/holosoma/holosoma/run_sim.py" robot:g1-29dof \
    sensor.d435i_front_depth:g1-d435i-front-depth \
    plugin.depth:depth-shm-d435i \
    terrain:terrain-load-step \
    --robot.asset.xml-file g1/g1_29dof_halfspherehand.xml \
    --simulator.config.bridge.enabled=True \
    "$@"

# The sim creates depth_img_shm with shape (1, 1, 58, 87). Start
# run_php_inference.sh only after that message appears.
