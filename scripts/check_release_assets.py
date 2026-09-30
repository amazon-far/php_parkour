#!/usr/bin/env python3
"""Check staged PHP release archives; pending final models block publication."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from urllib.parse import urlparse

from download_assets import MANIFEST, extract_verified, read_component, sha256


def check(
    manifest: Path, archives: Path, *, allow_pending_student: bool = False
) -> dict:
    data = json.loads(manifest.read_text())
    if data.get("schema_version") != 1:
        raise ValueError("Unsupported release asset manifest version")
    results = {}
    for name in ("databases", "student"):
        item = data.get("components", {}).get(name)
        if item is None:
            raise ValueError(f"Missing asset component: {name}")
        if (
            name == "student"
            and item.get("status") != "ready"
            and allow_pending_student
        ):
            results[name] = "pending final model; publication remains blocked"
            continue
        component = read_component(manifest, name)
        info = component["archive"]
        archive = archives / Path(urlparse(info["url"]).path).name
        if not archive.is_file():
            raise ValueError(f"Missing staged archive: {archive}")
        if archive.stat().st_size != info["size"] or sha256(archive) != info["sha256"]:
            raise ValueError(f"Archive checksum/size mismatch: {archive}")
        with tempfile.TemporaryDirectory(
            prefix=f"php-release-check-{name}-"
        ) as scratch:
            directory = Path(scratch)
            extract_verified(archive, component, directory)
            if name == "databases":
                counts = {
                    suffix: len(list(directory.rglob(f"*{suffix}")))
                    for suffix in (".bin", ".obj", ".json")
                }
                if counts != {".bin": 19, ".obj": 32, ".json": 32}:
                    raise ValueError(f"Unexpected database asset counts: {counts}")
            else:
                import onnx

                from wbt_training.utils.release_metadata import sanitize_model

                for filename in ("student.onnx", "depth_backbone.onnx"):
                    model = onnx.load(directory / filename, load_external_data=False)
                    onnx.checker.check_model(model)
                    if (
                        sanitize_model(model).SerializeToString()
                        != model.SerializeToString()
                    ):
                        raise ValueError(
                            f"Release provenance has not been removed from {filename}"
                        )
            results[name] = "verified"
    return {
        "release_ready": results.get("student") == "verified",
        "components": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archives",
        type=Path,
        required=True,
        help="Directory containing staged release tar.gz files",
    )
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument(
        "--allow-pending-student",
        action="store_true",
        help="Check source/data preparation only; does not certify publication readiness",
    )
    args = parser.parse_args()
    try:
        result = check(
            args.manifest,
            args.archives,
            allow_pending_student=args.allow_pending_student,
        )
    except (OSError, ValueError, KeyError, ImportError) as exc:
        parser.exit(1, f"Release asset check failed: {exc}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
