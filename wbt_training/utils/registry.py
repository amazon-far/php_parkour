"""Resolve PHP datasets while keeping their registry identity separate from caches.

This module can run before simulator startup. W&B is imported only for remote
inputs or for recording inputs on an already active W&B training run.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import os
import re
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

ARTIFACT_TYPE = "terrains-motions"
METADATA_KEY = "php_registry"
_IMMUTABLE = re.compile(r":v\d+$")


def entries(text: str, label: str) -> list[str]:
    values = [value.strip() for value in text.split(",")]
    if not all(values) or any("\n" in value or "\r" in value for value in values):
        raise ValueError(
            f"{label} must be a nonempty comma-separated list without empty entries"
        )
    return values


def checkpoint_reference(value: str) -> str:
    if value.startswith("wandb://"):
        parts = value[len("wandb://") :].split("/")
        if len(parts) >= 4 and parts[2] == "runs":
            parts.pop(2)
        if len(parts) not in (3, 4) or not all(
            re.fullmatch(r"[\w.-]+", p) for p in parts
        ):
            raise ValueError(f"Invalid W&B checkpoint URI: {value!r}")
        if len(parts) == 4 and not parts[-1].endswith(".pt"):
            raise ValueError(f"Checkpoint filename must end in .pt: {value!r}")
        return "wandb://" + "/".join(parts)
    if "://" in value:
        raise ValueError(f"Use a local .pt path or wandb:// checkpoint: {value!r}")
    path = Path(value).expanduser().resolve()
    if (
        path.suffix != ".pt"
        or not path.is_file()
        or not os.access(path, os.R_OK)
        or path.stat().st_size == 0
    ):
        raise ValueError(f"Checkpoint must be a readable, nonempty .pt file: {path}")
    return str(path)


def bundle_pairs(root: Path) -> list[tuple[Path, Path | None, Path | None]]:
    """Discover pairs in relative-path order, including motion-only locomotion."""
    if not root.is_dir():
        raise ValueError(f"Local REGISTRY directory does not exist: {root}")
    if any(root.rglob("*terrain*.npz")):
        raise ValueError(
            "Mesh terrain NPZ files are unsupported; use PHP motion NPZ / terrain NPY pairs"
        )
    motions = sorted(
        root.rglob("*motion*.npz"), key=lambda p: p.relative_to(root).as_posix()
    )
    if not motions:
        raise ValueError(f"No *motion*.npz files in REGISTRY {root}")
    terrains = set(root.rglob("*terrain*.npy"))
    pairs = []
    expected = set()
    for motion in motions:
        terrain = motion.with_name(
            "terrain".join(motion.name.rsplit("motion", 1))
        ).with_suffix(".npy")
        expected.add(terrain)
        obj = terrain.with_suffix(".obj")
        pairs.append(
            (
                motion,
                terrain if terrain.is_file() else None,
                obj if obj.is_file() else None,
            )
        )
    if terrains and terrains != expected:
        raise ValueError(
            f"REGISTRY {root} must pair each motion NPZ with its matching terrain NPY"
        )
    return pairs


def validate_bundle(root: Path) -> None:
    required = {
        "fps",
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
        "joint_names",
        "body_names",
    }
    for motion, terrain, _ in bundle_pairs(root):
        try:
            with zipfile.ZipFile(motion) as archive:
                keys = {
                    Path(name).stem
                    for name in archive.namelist()
                    if name.endswith(".npy")
                }
        except (OSError, zipfile.BadZipFile) as error:
            raise ValueError(f"Invalid motion NPZ {motion}: {error}") from error
        if missing := required - keys:
            raise ValueError(
                f"Motion {motion} is missing native Holosoma fields {sorted(missing)}; "
                "convert motion-matching outputs with motion_convert.py first"
            )
        if terrain is not None:
            import numpy as np

            data = np.load(terrain, allow_pickle=False)
            if not isinstance(data, np.ndarray):
                data.close()
                raise ValueError(f"Terrain {terrain} must be a box-array NPY file")
            if (
                data.ndim != 3
                or data.shape[2] != 10
                or data.shape[1] < 1
                or data.dtype.kind not in "fiu"
                or not np.isfinite(data).all()
            ):
                raise ValueError(
                    f"Terrain {terrain} must contain finite (boxes, variants, 10) data"
                )


def registry_reference(value: str) -> str:
    if not isinstance(value, str) or any(c in value for c in ",\r\n\t"):
        raise ValueError(
            "Registry references must be strings without commas or control characters"
        )
    if value.startswith("file://"):
        path = Path(value[len("file://") :]).expanduser().resolve()
        validate_bundle(path)
        return "file://" + str(path)
    name = value.removeprefix("wandb://")
    if not re.fullmatch(
        r"[\w.-]+/(?:[\w.-]+|wandb-registry-[\w. -]+)/[\w.-]+(?::[\w.-]+)?", name
    ):
        raise ValueError(
            f"Invalid REGISTRY {value!r}; use file:///absolute/bundle or entity/project/artifact:version"
        )
    return name if ":" in name else name + ":latest"


@dataclasses.dataclass(frozen=True)
class ResolvedRegistry:
    reference: str
    directory: Path

    @property
    def local_uri(self) -> str:
        return "file://" + str(self.directory)


class RegistryResolver:
    def __init__(self, api=None):
        self._api = api

    @property
    def api(self):
        if self._api is None:
            import wandb

            self._api = wandb.Api()
        return self._api

    def for_run(self, checkpoint: str) -> list[str]:
        run_path = "/".join(checkpoint[len("wandb://") :].split("/")[:3])
        run = self.api.run(run_path)
        config = run.config or {}
        if METADATA_KEY in config:
            metadata = config[METADATA_KEY]
            if (
                not isinstance(metadata, dict)
                or type(metadata.get("version")) is not int
                or metadata["version"] != 1
            ):
                raise ValueError(
                    f"Run {run_path} has unsupported PHP registry metadata; set REGISTRY explicitly"
                )
            names = metadata.get("registry_names")
            if (
                not isinstance(names, list)
                or not names
                or not all(isinstance(n, str) for n in names)
            ):
                raise ValueError(
                    f"Run {run_path} has invalid ordered registry metadata; set REGISTRY explicitly"
                )
            try:
                result = [registry_reference(n) for n in names]
                if any(
                    not r.startswith("file://") and not _IMMUTABLE.search(r)
                    for r in result
                ):
                    raise ValueError(
                        "Recorded remote registries must have immutable versions"
                    )
            except ValueError as error:
                raise ValueError(
                    f"Run {run_path}: {error}; set REGISTRY explicitly"
                ) from error
            return result

        training = config.get("training")
        stored = training.get("registry_name") if isinstance(training, dict) else None
        if isinstance(stored, str) and stored:
            try:
                result = [registry_reference(r) for r in entries(stored, "REGISTRY")]
                # Mutable legacy aliases cannot establish the data used at training time.
                if all(r.startswith("file://") or _IMMUTABLE.search(r) for r in result):
                    return result
            except ValueError:
                pass
        artifacts = [a for a in run.used_artifacts() if a.type == ARTIFACT_TYPE]
        if len(artifacts) == 1:
            try:
                reference = registry_reference(artifacts[0].qualified_name)
                if _IMMUTABLE.search(reference):
                    return [reference]
            except ValueError:
                pass
        raise ValueError(
            f"Run {run_path} has no unambiguous usable ordered registry metadata; set REGISTRY explicitly"
        )

    def resolve(self, references: str | list[str]) -> list[ResolvedRegistry]:
        values = (
            entries(references, "REGISTRY")
            if isinstance(references, str)
            else references
        )
        if not values:
            raise ValueError("REGISTRY must have at least one entry")
        # Check every local input before any network request.
        names = [registry_reference(v) for v in values]
        return [self._resolve_one(name) for name in names]

    @staticmethod
    def _cache_path(reference: str) -> Path:
        identity = os.environ.get("WANDB_BASE_URL", "") + "\n" + reference
        digest = hashlib.sha256(identity.encode()).hexdigest()[:16]
        root = (
            Path(os.environ.get("CONVERT_CACHE_DIR", tempfile.gettempdir()))
            .expanduser()
            .resolve()
        )
        return root / f"holosoma_{digest}_{reference.rsplit(':', 1)[1]}"

    @staticmethod
    def _cache_ready(path: Path, reference: str) -> bool:
        marker = path / ".done"
        if marker.is_file() and marker.read_text() == reference + "\n":
            validate_bundle(path)
            return True
        return False

    def _resolve_one(self, reference: str) -> ResolvedRegistry:
        if reference.startswith("file://"):
            return ResolvedRegistry(reference, Path(reference[len("file://") :]))
        if _IMMUTABLE.search(reference):
            cached = self._cache_path(reference)
            if self._cache_ready(cached, reference):
                return ResolvedRegistry(reference, cached)
        artifact = self.api.artifact(reference)
        pinned = registry_reference(artifact.qualified_name)
        # A registry link and its source project can have different version
        # numbers. Preserve the linked qualified name when it is already pinned.
        if not _IMMUTABLE.search(pinned):
            version = artifact.version
            if not isinstance(version, str) or not re.fullmatch(r"v\d+", version):
                raise ValueError(
                    f"Cannot resolve an immutable artifact version for {reference!r}: {version!r}"
                )
            pinned = pinned.rsplit(":", 1)[0] + ":" + version
        version = pinned.rsplit(":", 1)[1]
        if _IMMUTABLE.search(reference) and reference.rsplit(":", 1)[1] != version:
            raise ValueError(
                f"Requested {reference}, but W&B returned version {version}"
            )
        output = self._cache_path(pinned)
        output.parent.mkdir(parents=True, exist_ok=True)
        import fcntl

        with output.with_suffix(".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not self._cache_ready(output, pinned):
                with tempfile.TemporaryDirectory(
                    prefix="php-convert-", dir=output.parent
                ) as tmp:
                    raw = Path(artifact.download(root=str(Path(tmp) / "raw")))
                    converted = Path(tmp) / "converted"
                    converted.mkdir()
                    for motion, terrain, obj in bundle_pairs(raw):
                        target = converted / motion.relative_to(raw)
                        target.parent.mkdir(parents=True, exist_ok=True)
                        with zipfile.ZipFile(motion) as archive:
                            native = {"body_names.npy", "joint_names.npy"}.issubset(
                                archive.namelist()
                            )
                        if native:
                            shutil.copy2(motion, target)
                        else:
                            from wbt_training.training_runs.motion_convert import (
                                convert,
                            )

                            if not convert(str(motion), str(target)):
                                raise ValueError(
                                    f"Motion {motion} has fewer than two frames"
                                )
                        for sidecar in (terrain, obj):
                            if sidecar is not None:
                                shutil.copy2(
                                    sidecar, converted / sidecar.relative_to(raw)
                                )
                    validate_bundle(converted)
                    (converted / ".done").write_text(pinned + "\n")
                    if output.exists():
                        shutil.rmtree(output)
                    converted.rename(output)
        print(f"==> {pinned} -> file://{output}", file=sys.stderr)
        return ResolvedRegistry(pinned, output)


def record_registry_usage(registries: list[ResolvedRegistry], logger_type: str) -> None:
    """Link unique remote inputs once on rank zero, retaining the full ordering."""
    if logger_type != "wandb" or int(os.environ.get("RANK", "0")) != 0:
        return
    import wandb

    if wandb.run is None:
        return
    names = [r.reference for r in registries]
    for reference in dict.fromkeys(names):
        if not reference.startswith("file://"):
            wandb.run.use_artifact(reference, type=ARTIFACT_TYPE)
    training = dict(wandb.run.config.get("training", {}))
    training["registry_name"] = ",".join(names)
    wandb.run.config.update(
        {METADATA_KEY: {"version": 1, "registry_names": names}, "training": training},
        allow_val_change=True,
    )


def prepare_inputs(
    mode: str, checkpoint_text: str, registry_text: str, *, resolver=None
):
    resolver = resolver or RegistryResolver()
    checkpoints = (
        [checkpoint_reference(c) for c in entries(checkpoint_text, "CHECKPOINT")]
        if checkpoint_text
        else []
    )
    if mode.startswith("eval") and len(checkpoints) != 1:
        raise ValueError("Evaluation requires exactly one CHECKPOINT")
    if registry_text == "auto":
        if (
            mode not in ("distill", "eval_teacher", "eval_student")
            or not checkpoints
            or any(not c.startswith("wandb://") for c in checkpoints)
        ):
            raise ValueError(
                "REGISTRY=auto requires explicit wandb:// checkpoint(s); supply REGISTRY explicitly for local checkpoints"
            )
        if mode == "distill":
            registries = []
            for checkpoint in checkpoints:
                sources = resolver.for_run(checkpoint)
                if len(sources) != 1:
                    raise ValueError(
                        f"Teacher {checkpoint} uses multiple registries; provide one REGISTRY per TEACHER_CHECKPOINT explicitly"
                    )
                registries.extend(sources)
        else:
            registries = resolver.for_run(checkpoints[0])
    else:
        registries = entries(registry_text, "REGISTRY")
    if mode == "distill" and len(checkpoints) != len(registries):
        raise ValueError(
            f"Provide one REGISTRY per TEACHER_CHECKPOINT, in matching order ({len(checkpoints)} teachers, {len(registries)} registries)"
        )
    resolved = resolver.resolve(registries)
    return checkpoints, resolved


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=("teacher", "distill", "eval_teacher", "eval_student")
    )
    parser.add_argument("checkpoints")
    parser.add_argument("registries")
    args = parser.parse_args(argv)
    try:
        with contextlib.redirect_stdout(sys.stderr):
            checkpoints, registries = prepare_inputs(
                args.mode, args.checkpoints, args.registries
            )
    except Exception as error:  # noqa: BLE001 - report SDK and input errors at the CLI boundary
        parser.exit(1, f"ERROR: {error}\n")
    print(",".join(checkpoints))
    print(",".join(r.reference for r in registries))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
