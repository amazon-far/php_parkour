"""Train G1 29-DOF whole-body tracking agents.

Usage::

    python -m wbt_training.train_agent \\
        exp:g1-wbt-terrain-ref \\
        simulator:isaacsim \\
        --training.headless True

Set ``LOGURU_LEVEL=INFO`` to restore verbose logging (default is WARNING to
reduce simulator log noise).
"""

from __future__ import annotations

import os
import sys
import time

import tyro
from holosoma.train_agent import TrainingContext, train
from holosoma.utils.sim_utils import setup_simulator_imports
from holosoma.utils.tyro_utils import TYRO_CONIFG

from wbt_training.config_values.experiment import AnnotatedExperimentConfig


def _stagger_multi_gpu_init() -> None:
    """Lightweight stagger for multi-GPU IsaacSim init.

    Each rank delays by (local_rank * INIT_STAGGER_SEC) seconds before starting.
    This avoids all 8 ranks hitting the GPU simultaneously during AppLauncher init.

    With the num_envs-per-rank fix in setup_isaaclab_launcher (each rank only
    allocates its share of envs), the per-rank init is light enough that a short
    stagger (3-5s) suffices. No sequential waiting needed.

    Set INIT_STAGGER_SEC=0 to disable staggering entirely.
    Set INIT_MODE=sequential to fall back to the old sentinel-based sequential init.
    """

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if world_size <= 1:
        return

    init_mode = os.environ.get("INIT_MODE", "stagger")

    if init_mode == "sequential":
        _sequential_multi_gpu_init()
        return

    stagger_sec = int(os.environ.get("INIT_STAGGER_SEC", "5"))
    delay = local_rank * stagger_sec

    if delay > 0:
        print(
            f"[INIT] Rank {local_rank}/{world_size}: staggering {delay}s before IsaacSim init",
            file=sys.stderr,
        )
        time.sleep(delay)

    print(
        f"[INIT] Rank {local_rank}/{world_size}: starting IsaacSim init",
        file=sys.stderr,
    )


def _sequential_multi_gpu_init() -> None:
    """Sequential IsaacSim init across GPUs using file-based signaling (fallback).

    Each rank waits for the previous rank to write a sentinel file before starting.
    Rank 0 starts immediately. Use INIT_MODE=sequential to enable this mode.
    """

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if world_size <= 1:
        return

    sentinel_dir = os.environ.get("INIT_SENTINEL_DIR", "/tmp/wbt_init_sentinels")
    os.makedirs(sentinel_dir, exist_ok=True)

    max_wait = int(os.environ.get("INIT_MAX_WAIT_SEC", "3600"))

    if local_rank > 0:
        prev_sentinel = os.path.join(sentinel_dir, f"rank_{local_rank - 1}_done")
        print(
            f"[SEQ-INIT] Rank {local_rank}/{world_size}: waiting for rank {local_rank - 1}...",
            file=sys.stderr,
        )
        waited = 0
        while not os.path.exists(prev_sentinel):
            time.sleep(5)
            waited += 5
            if waited > max_wait:
                print(
                    f"[SEQ-INIT] Rank {local_rank}: TIMEOUT after {max_wait}s. Proceeding.",
                    file=sys.stderr,
                )
                break
            if waited % 60 == 0:
                print(
                    f"[SEQ-INIT] Rank {local_rank}: still waiting ({waited}s)...",
                    file=sys.stderr,
                )
        print(f"[SEQ-INIT] Rank {local_rank}: starting IsaacSim init", file=sys.stderr)
    else:
        for f in os.listdir(sentinel_dir):
            if f.startswith("rank_") and f.endswith("_done"):
                os.remove(os.path.join(sentinel_dir, f))
        print("[SEQ-INIT] Rank 0: starting IsaacSim init (first)", file=sys.stderr)


def _write_init_sentinel() -> None:
    """Write sentinel file to signal that this rank's IsaacSim init is done."""
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return

    sentinel_dir = os.environ.get("INIT_SENTINEL_DIR", "/tmp/wbt_init_sentinels")
    os.makedirs(sentinel_dir, exist_ok=True)
    sentinel_path = os.path.join(sentinel_dir, f"rank_{local_rank}_done")
    with open(sentinel_path, "w") as f:
        f.write(f"rank {local_rank} IsaacSim init done\n")
    print(
        f"[SEQ-INIT] Rank {local_rank}: IsaacSim init done, wrote {sentinel_path}",
        file=sys.stderr,
    )


def main() -> None:
    """Launch training with G1 WBT presets."""
    os.environ.setdefault("LOGURU_LEVEL", "WARNING")
    _stagger_multi_gpu_init()
    tyro_cfg = tyro.cli(AnnotatedExperimentConfig, config=TYRO_CONIFG)
    setup_simulator_imports(tyro_cfg)

    if os.environ.get("INIT_MODE", "stagger") == "sequential":
        with TrainingContext(tyro_cfg) as context:
            _write_init_sentinel()
            context.train()
    else:
        train(tyro_cfg)


if __name__ == "__main__":
    main()
