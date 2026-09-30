"""Validate and upload PHP motion/terrain pairs to a W&B registry.

Run with ``python -m motion_matching.upload_dataset --help``. W&B is imported
only after local validation, and never for ``--dry-run``. Existing payloads
are uploaded verbatim; only missing interactive locomotion terrain is staged.
"""

from __future__ import annotations

import argparse
import importlib
import os
import re
import shlex
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_PROJECT = "parkour_registry"
DEFAULT_REGISTRY = "terrains-motions"
ARTIFACT_TYPE = "terrains-motions"
EMPTY_TERRAIN_SHAPE = (0, 10, 10)


@dataclass(frozen=True)
class UploadFile:
    """A source file and its artifact path; None means staged empty terrain."""

    source: Path | None
    name: str


@dataclass(frozen=True)
class Dataset:
    root: Path
    files: tuple[UploadFile, ...]
    clip_count: int
    formats: tuple[str, ...]


def _numeric(array: np.ndarray, label: str) -> None:
    if array.dtype.kind not in "fiu" or not np.isfinite(array).all():
        raise ValueError(f"{label} must contain finite real numbers")


def _validate_motion(path: Path) -> str:
    required = {
        "fps",
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
    }
    try:
        with np.load(path, allow_pickle=False) as data:
            missing = required - set(data.files)
            if missing:
                raise ValueError(
                    f"missing {sorted(missing)}; expected PHP BeyondMimic or native motion NPZ"
                )
            positions = data["joint_pos"]
            if positions.ndim != 2 or positions.shape[0] < 2:
                raise ValueError("joint_pos must be (T, 29) or (T, 36), with T >= 2")
            native = positions.shape[1] == 36
            frames = positions.shape[0]
            bodies = 32 if native else 30
            shapes = {
                "joint_pos": (frames, 36 if native else 29),
                "joint_vel": (frames, 35 if native else 29),
                "body_pos_w": (frames, bodies, 3),
                "body_quat_w": (frames, bodies, 4),
                "body_lin_vel_w": (frames, bodies, 3),
                "body_ang_vel_w": (frames, bodies, 3),
            }
            for key, shape in shapes.items():
                value = data[key]
                if value.shape != shape:
                    raise ValueError(
                        f"{key} must have shape {shape}, got {value.shape}"
                    )
                _numeric(value, key)
            fps = data["fps"]
            _numeric(fps, "fps")
            if fps.shape not in ((), (1,)) or fps.item() <= 0:
                raise ValueError("fps must be a positive scalar or one-element array")
            for key, count in (("joint_names", 29), ("body_names", 32)):
                if native:
                    if key not in data:
                        raise ValueError(f"native motion is missing {key}")
                    names = data[key]
                    if (
                        names.shape != (count,)
                        or names.dtype.kind not in "US"
                        or len(set(names.tolist())) != count
                        or not all(names.tolist())
                    ):
                        raise ValueError(
                            f"{key} must contain {count} unique, nonempty strings"
                        )
                elif key in data:
                    raise ValueError(
                        f"{key} requires the native 36-joint/32-body array layout"
                    )
            if "vel_cmd" in data:
                commands = data["vel_cmd"]
                if (
                    commands.ndim != 2
                    or commands.shape[0] != frames
                    or commands.shape[1] == 0
                ):
                    raise ValueError(
                        "vel_cmd must be (T, C), matching joint_pos, with C > 0"
                    )
                _numeric(commands, "vel_cmd")
    except (OSError, ValueError, TypeError, EOFError) as error:
        raise ValueError(f"Invalid motion {path}: {error}") from error
    return "native" if native else "beyondmimic"


def _validate_terrain(path: Path) -> None:
    try:
        terrain = np.load(path, allow_pickle=False)
        if not isinstance(terrain, np.ndarray):
            terrain.close()
            raise TypeError("expected a box-array NPY, not an NPZ archive")
        if terrain.ndim != 3 or terrain.shape[2] != 10 or terrain.shape[1] == 0:
            raise ValueError(
                "expected (num_boxes, num_variants, 10), with num_variants > 0; "
                f"got {terrain.shape}. Empty locomotion terrain can use {EMPTY_TERRAIN_SHAPE}"
            )
        _numeric(terrain, "terrain")
    except (OSError, ValueError, TypeError, EOFError) as error:
        raise ValueError(f"Invalid terrain {path}: {error}") from error


def _is_payload(stem: str, kind: str) -> bool:
    return stem == kind or stem.startswith(kind + "_") or stem.endswith("_" + kind)


def prepare_dataset(file_folder: str | Path) -> Dataset:
    """Select and validate a dataset without importing W&B or writing files.

    Canonical paths stay relative to file_folder. Interactive ``motion_X.npz``
    plus ``motion_X/`` becomes ``motion_X/sample_motion.npz`` and paired terrain.
    A wholly terrain-free canonical bundle is allowed for flat locomotion.
    Once any terrain is present, all clips must be paired.
    """
    root = Path(file_folder).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Dataset folder does not exist or is not a directory: {root}")
    paths = sorted(
        (p for p in root.rglob("*") if p.is_file()),
        key=lambda p: p.relative_to(root).as_posix(),
    )
    for path in paths:
        if path.suffix == ".npz" and _is_payload(path.stem, "terrain"):
            raise ValueError(
                f"Mesh terrain NPZ is unsupported: {path}. Use PHP box-array *_terrain.npy "
                "(and an optional matching OBJ preview); regenerate the dataset in PHP."
            )
    motions = [
        p
        for p in paths
        if p.suffix == ".npz"
        and (p.stem.startswith("motion_") or p.stem.endswith("_motion"))
    ]
    if not motions:
        raise ValueError(f"No *_motion.npz or motion_*.npz files found in {root}")
    terrains = {
        p for p in paths if p.suffix == ".npy" and _is_payload(p.stem, "terrain")
    }
    used_terrains: set[Path] = set()
    files: dict[str, UploadFile] = {}
    formats: set[str] = set()
    unpaired: list[Path] = []

    def add(source: Path | None, name: str) -> None:
        if name in files:
            raise ValueError(
                f"Artifact path collision at {name}: {files[name].source} and {source}"
            )
        if source is not None and not source.resolve().is_relative_to(root):
            raise ValueError(f"Payload points outside dataset folder: {source}")
        files[name] = UploadFile(source, name)

    for motion in motions:
        formats.add(_validate_motion(motion))
        terrain = motion.with_name(
            "terrain".join(motion.stem.rsplit("motion", 1)) + ".npy"
        )
        preview = terrain.with_suffix(".obj")
        interactive = motion.with_suffix("")
        if motion.stem.startswith("motion_") and interactive.is_dir():
            if terrain.exists() or preview.exists():
                raise ValueError(
                    f"Ambiguous canonical and interactive terrain for {motion}"
                )
            prefix = interactive.relative_to(root).as_posix()
            terrain = interactive / "terrain.npy"
            preview = interactive / "multi_boxes_scaled.obj"
            add(motion, f"{prefix}/sample_motion.npz")
            if terrain.is_file():
                _validate_terrain(terrain)
                used_terrains.add(terrain)
                add(terrain, f"{prefix}/sample_terrain.npy")
            else:
                add(None, f"{prefix}/sample_terrain.npy")
            if preview.is_file():
                add(preview, f"{prefix}/sample_terrain.obj")
        else:
            add(motion, motion.relative_to(root).as_posix())
            if terrain.is_file():
                _validate_terrain(terrain)
                used_terrains.add(terrain)
                add(terrain, terrain.relative_to(root).as_posix())
            else:
                unpaired.append(motion)
                if preview.exists():
                    raise ValueError(
                        f"OBJ preview needs a matching box-array NPY: {terrain}"
                    )
            if preview.is_file():
                add(preview, preview.relative_to(root).as_posix())

    if orphaned := terrains - used_terrains:
        names = ", ".join(sorted(p.relative_to(root).as_posix() for p in orphaned))
        raise ValueError(f"Terrain NPY files have no matching motion: {names}")
    if unpaired and any(entry.name.endswith(".npy") for entry in files.values()):
        names = ", ".join(p.relative_to(root).as_posix() for p in unpaired)
        raise ValueError(
            f"Mixed terrain bundle has unpaired motions: {names}. Add matching empty "
            f"*_terrain.npy arrays shaped {EMPTY_TERRAIN_SHAPE} for flat locomotion."
        )
    return Dataset(
        root,
        tuple(files[name] for name in sorted(files)),
        len(motions),
        tuple(sorted(formats)),
    )


def _path_component(value: str) -> str:
    if not value or value != value.strip() or any(c in value for c in "/\\:\n\r\t"):
        raise argparse.ArgumentTypeError(
            "must be a nonempty name, without slashes, colons or control characters"
        )
    return value


def _collection_name(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
        raise argparse.ArgumentTypeError(
            "collection name may contain letters, numbers, underscores, dots and hyphens"
        )
    return value


def _registry_label(value: str) -> str:
    if value != value.strip() or not re.fullmatch(r"[\w. -]+", value):
        raise argparse.ArgumentTypeError(
            "registry name may contain letters, numbers, underscores, dots, hyphens "
            "and spaces, without leading or trailing whitespace"
        )
    return value


def _target_path(registry_name: str, registry: str, organization: str | None) -> str:
    _collection_name(registry_name)
    _registry_label(registry)
    target = f"wandb-registry-{registry}/{registry_name}"
    return (
        f"{_path_component(organization)}/{target}"
        if organization is not None
        else target
    )


def upload_dataset(
    dataset: Dataset,
    *,
    registry_name: str,
    entity: str | None = None,
    project: str | None = None,
    registry: str = DEFAULT_REGISTRY,
    organization: str | None = None,
) -> str:
    """Upload a prepared dataset and return its linked, versioned registry name."""
    target = _target_path(registry_name, registry, organization)
    try:
        wandb = importlib.import_module("wandb")
    except ImportError as error:
        raise RuntimeError(
            "Uploading requires W&B. From motion_matching/, run: "
            "python -m pip install -e '.[registry]' (wandb>=0.22.1,<1). "
            "Use --dry-run to validate without W&B."
        ) from error
    # Read W&B's merged caller settings (including local settings and env vars)
    # before applying the public fallback. Unspecified entity/org stay with W&B.
    configured_project = wandb.setup().settings.project
    init_options = {
        "project": project or configured_project or DEFAULT_PROJECT,
        "name": registry_name,
    }
    if entity is not None:
        init_options["entity"] = entity

    with tempfile.TemporaryDirectory(prefix="php-upload-") as temporary:
        run = wandb.init(**init_options)
        try:
            artifact = wandb.Artifact(
                name=registry_name,
                type=ARTIFACT_TYPE,
                metadata={
                    "schema_version": 1,
                    "clip_count": dataset.clip_count,
                    "formats": list(dataset.formats),
                },
            )
            empty_path = Path(temporary) / "empty.npy"
            for entry in dataset.files:
                source = entry.source
                if source is None:
                    if not empty_path.exists():
                        np.save(
                            empty_path, np.zeros(EMPTY_TERRAIN_SHAPE, dtype=np.float32)
                        )
                    source = empty_path
                artifact.add_file(str(source), name=entry.name)
            logged = run.log_artifact(artifact)
            logged.wait()
            linked = run.link_artifact(artifact=logged, target_path=target)
            qualified_name = getattr(linked, "qualified_name", None)
            expected = rf"[^/]+/wandb-registry-{re.escape(registry)}/{re.escape(registry_name)}:v\d+"
            if not isinstance(qualified_name, str) or not re.fullmatch(
                expected, qualified_name
            ):
                raise RuntimeError(
                    "W&B did not return a versioned linked registry Artifact. "
                    "Use wandb>=0.22.1,<1 and check registry access; source versions are not registry versions."
                )
        except BaseException:
            try:
                run.finish(exit_code=1)
            except Exception as error:  # noqa: BLE001 - preserve the original upload failure
                print(f"W&B run cleanup also failed: {error}", file=sys.stderr)
            raise
        else:
            run.finish(exit_code=0)
    return qualified_name


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry_name",
        "--registry-name",
        required=True,
        type=_collection_name,
        help="Registry collection name",
    )
    parser.add_argument(
        "--file_folder",
        "--file-folder",
        required=True,
        type=Path,
        help="Dataset directory (searched recursively)",
    )
    parser.add_argument(
        "--entity",
        type=_path_component,
        help="W&B upload team; default: caller's W&B configuration",
    )
    parser.add_argument(
        "--project",
        type=_path_component,
        help=f"Upload project; default: W&B configuration or {DEFAULT_PROJECT}",
    )
    parser.add_argument(
        "--registry",
        type=_registry_label,
        default=DEFAULT_REGISTRY,
        help="Registry name, without the wandb-registry- prefix (default: %(default)s)",
    )
    parser.add_argument(
        "--organization",
        type=_path_component,
        help="Registry organization; default: W&B configuration/discovery",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and list files without importing W&B or uploading",
    )
    args = parser.parse_args(argv)
    try:
        dataset = prepare_dataset(args.file_folder)
        target = _target_path(args.registry_name, args.registry, args.organization)
        project = (
            args.project
            or os.environ.get("WANDB_PROJECT")
            or f"W&B configuration (fallback: {DEFAULT_PROJECT})"
        )
        print(
            f"Validated {dataset.clip_count} clips ({', '.join(dataset.formats)}) in {dataset.root}"
        )
        print(
            f"Upload project: {project}; entity: {args.entity or 'W&B configuration'}"
        )
        print(
            f"Registry collection: {target} (organization resolved by W&B when omitted)"
        )
        for entry in dataset.files:
            source = (
                entry.source.relative_to(dataset.root).as_posix()
                if entry.source is not None
                else f"generated empty terrain {EMPTY_TERRAIN_SHAPE}"
            )
            print(f"  {entry.name} <- {source}")
        if args.dry_run:
            print("Dry run complete; no files written or uploaded.")
            return 0
        qualified_name = upload_dataset(
            dataset,
            registry_name=args.registry_name,
            entity=args.entity,
            project=args.project,
            registry=args.registry,
            organization=args.organization,
        )
    except Exception as error:  # noqa: BLE001 - CLI boundary for SDK/validation failures
        print(f"Upload failed: {error}", file=sys.stderr)
        return 1
    print(f"Linked artifact: {qualified_name}")
    print(f"REGISTRY={shlex.quote(qualified_name)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
