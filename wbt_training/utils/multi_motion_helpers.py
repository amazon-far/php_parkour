"""Combine ordered motion/terrain bundles from local paths or explicit W&B registries."""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Motion / terrain combining
# ---------------------------------------------------------------------------

# Allocate scratch space only when preprocessing actually writes a bundle.
_TMP_DIR: Path | None = None


def _scratch_dir() -> Path:
    global _TMP_DIR
    if _TMP_DIR is None:
        _TMP_DIR = Path(tempfile.mkdtemp(prefix="holosoma_motions_"))
    return _TMP_DIR


def combine_motions(input_files: list[str | Path]) -> Path:
    """Concatenate multiple motion NPZ files into one.

    Uses a deterministic hash of input filenames for caching.
    """
    if not input_files:
        raise ValueError("At least one motion file is required")
    data_list = []
    for f in input_files:
        with np.load(f) as data:
            d = {key: data[key] for key in data.files}
            if "motion_ends" not in d:
                d["motion_ends"] = np.zeros((d["joint_pos"].shape[0],), dtype=bool)
                d["motion_ends"][-1] = True
            data_list.append(d)

    # Keys that are scalar metadata — keep from first file, don't concatenate.
    # - fps: scalar
    # - joint_names / body_names: per-motion metadata, identical across files
    #   in the same registry. Converted holosoma NPZs ship these fields;
    #   ref artifact NPZs do not.
    _SCALAR_KEYS = {"fps", "joint_names", "body_names"}

    combined = {}
    for key in data_list[0]:
        if key in _SCALAR_KEYS:
            combined[key] = data_list[0][key]
        else:
            combined[key] = np.concatenate([d[key] for d in data_list], axis=0)

    combined_file_name = hashlib.sha256(
        ",".join(str(f) for f in input_files).encode()
    ).hexdigest()[:16]
    out_path = _scratch_dir() / f"{combined_file_name}_motions.npz"
    np.savez(out_path, **combined)
    return out_path


def combine_terrains(input_files: list[str | Path | None]) -> Path:
    """Combine terrain arrays, retaining empty entries for locomotion clips."""
    obj_list: list[np.ndarray] = []
    obj_count = 0
    obj_count_list = [0]

    for f in input_files:
        data = (
            np.load(f, allow_pickle=False) if f is not None else np.zeros((0, 10, 10))
        )
        if np.prod(data.shape) == 0:
            obj_list.append(np.zeros((0, 10, 10)))
            obj_count_list.append(obj_count)
        else:
            obj_list.append(data)
            obj_count_list.append(obj_count + data.shape[0])
            obj_count += data.shape[0]

    # Determine num_variant from first non-empty entry
    num_variant = 10
    for arr in obj_list:
        if arr.shape[0] != 0:
            num_variant = arr.shape[1]
            break
    for i in range(len(obj_list)):
        if obj_list[i].shape[0] == 0:
            obj_list[i] = np.zeros((0, num_variant, 10))

    combined_obj_list = np.concatenate(obj_list, axis=0)
    obj_count_arr = np.array(obj_count_list)

    combined_file_name = hashlib.sha256(
        ",".join(str(f) for f in input_files).encode()
    ).hexdigest()[:16]
    out_path = _scratch_dir() / f"{combined_file_name}_terrains.npz"
    np.savez(out_path, obj_list=combined_obj_list, obj_count_list=obj_count_arr)
    return out_path


def add_motion_idx_from_motion_counts(
    motion_file: Path, motion_counts: list[int]
) -> None:
    """Add ``motion_idxs`` array to motion file based on per-registry motion counts."""
    with np.load(motion_file) as data:
        T = data["joint_pos"].shape[0]
        motion_ends = data["motion_ends"]
        motion_ends_list = np.nonzero(motion_ends)[0]
        motion_idxs_cumsum = np.cumsum(np.array(motion_counts))

        motion_idxs = np.searchsorted(motion_ends_list, np.arange(T), side="left")
        motion_idxs = np.searchsorted(motion_idxs_cumsum, motion_idxs, side="right")

        new_motion_file = motion_file.parent / f"{motion_file.stem}.npz"
        payload = dict(data)
        payload["motion_idxs"] = motion_idxs
    np.savez(new_motion_file, **payload)


# ---------------------------------------------------------------------------
# WandB registry helpers
# ---------------------------------------------------------------------------


def pull_paired_from_wandb_registry(
    registry_list: str | list[str],
) -> tuple[list[int], Path, Path, list[Path]]:
    """Download paired (motion, terrain) data from WandB registry.

    Each WandB artifact is expected to contain paired files:

    * **Motion** files: any file whose name contains ``"motion"``
      (typically ``motion*.npz``).
    * **Terrain** files: any file whose name contains ``"terrain"``
      (typically ``terrain*.npy`` for the new-format obstacle-pose arrays,
      or ``terrain.obj`` for legacy OBJ mesh artifacts).

    The function separates terrain files by extension:

    * ``.npy`` terrain files are obstacle-pose arrays — they are combined
      via :func:`combine_terrains` into a single NPZ that
      :func:`wbt_training.utils.obstacle_helpers.add_onpath_obstacle_standalone` can
      consume (``obj_list`` + ``obj_count_list``).
    * ``.obj`` terrain files are mesh files — they are **not** combined
      but returned as-is in a separate list.  The caller can pass the
      first OBJ path directly to
      ``--terrain.terrain_term.obj_file_path``.

    Parameters
    ----------
    registry_list : str or list[str]
        Comma-separated WandB registry names (e.g.
        ``"entity/wandb-registry-terrains-motions/walk:v1"``),
        or a Python list of such names.

    Returns
    -------
    motion_counts : list[int]
        Number of motion files contributed by each registry entry.
    motion_file : Path
        Combined motion NPZ (with ``motion_idxs`` added).
    terrain_npy_file : Path
        Combined terrain obstacle NPZ (from ``.npy`` files).
        Will be an empty-data NPZ if no ``.npy`` terrain files were found.
    terrain_obj_files : list[Path]
        Raw ``.obj`` terrain mesh files (empty list if none found).
    """
    from wbt_training.utils.registry import RegistryResolver, bundle_pairs

    registries = RegistryResolver().resolve(registry_list)

    motion_files: list[Path] = []
    terrain_npy_files: list[Path | None] = []
    terrain_obj_files: list[Path] = []
    motion_counts: list[int] = []

    for registry in registries:
        pairs = bundle_pairs(registry.directory)
        motion_counts.append(len(pairs))
        for motion, terrain, obj in pairs:
            motion_files.append(motion)
            terrain_npy_files.append(terrain)
            if obj is not None:
                terrain_obj_files.append(obj)

    # Combine motion NPZ files
    motion_file = combine_motions(motion_files)
    add_motion_idx_from_motion_counts(motion_file, motion_counts)

    # Preserve one obstacle boundary per motion even for motion-only registries.
    terrain_npy_file = combine_terrains(terrain_npy_files)

    return motion_counts, motion_file, terrain_npy_file, terrain_obj_files
