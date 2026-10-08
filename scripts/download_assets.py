#!/usr/bin/env python3
"""Download and verify public PHP release assets using only the standard library."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from urllib.request import urlopen

MANIFEST = Path(__file__).resolve().parents[1] / "release-assets.json"
MOTION_COMPONENTS = (
    "motions-locomotion",
    "motions-low-step",
    "motions-high-step",
    "motions-low-climb-76",
    "motions-high-climb-76",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_path(value: str) -> Path:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or "\\" in value:
        raise ValueError(f"Unsafe asset path: {value!r}")
    if str(path) != value or value == ".":
        raise ValueError(f"Asset path must be normalized: {value!r}")
    return Path(*path.parts)


def read_component(manifest: Path, name: str) -> dict:
    data = json.loads(manifest.read_text())
    if data.get("schema_version") != 1:
        raise ValueError("Unsupported release asset manifest version")
    components = data.get("components", {})
    if name not in components:
        raise ValueError(
            f"Unknown component {name!r}; choose from {', '.join(components)}"
        )
    component = components[name]
    if component.get("status") != "ready":
        raise ValueError(component.get("message", f"{name} is not available yet"))
    entries = component.get("files", [])
    if not entries:
        raise ValueError(f"No files declared for {name}")
    seen = set()
    for entry in entries:
        relative_path(entry["path"])
        if entry["path"] in seen:
            raise ValueError(f"Duplicate manifest entry: {entry['path']}")
        seen.add(entry["path"])
        if len(entry["sha256"]) != 64 or any(
            char not in "0123456789abcdef" for char in entry["sha256"]
        ):
            raise ValueError(f"Invalid SHA-256 for {entry['path']}")
        if not isinstance(entry["size"], int) or entry["size"] < 0:
            raise ValueError(f"Invalid size for {entry['path']}")
    return component


def destination_for(name: str) -> Path:
    if name == "databases" and os.environ.get("PHP_MOTION_DATABASE_DIR"):
        return Path(os.environ["PHP_MOTION_DATABASE_DIR"]).expanduser()
    return Path.home() / ".cache" / "php-parkour" / name


def verify(component: dict, destination: Path) -> list[str]:
    problems = []
    for entry in component["files"]:
        path = destination / relative_path(entry["path"])
        if not path.is_file() or path.is_symlink():
            problems.append(f"missing regular file: {entry['path']}")
        elif path.stat().st_size != entry["size"]:
            problems.append(f"size mismatch: {entry['path']}")
        elif sha256(path) != entry["sha256"]:
            problems.append(f"checksum mismatch: {entry['path']}")
    return problems


def extract_verified(archive: Path, component: dict, output: Path) -> None:
    expected = {entry["path"]: entry for entry in component["files"]}
    found = set()
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle:
            name = member.name.removesuffix("/") if member.isdir() else member.name
            path = relative_path(name)
            if member.isdir():
                continue
            if not member.isfile() or name not in expected or name in found:
                raise ValueError(f"Unexpected archive member: {member.name}")
            if member.size != expected[name]["size"]:
                raise ValueError(f"Archive size mismatch: {name}")
            target = output / path
            target.parent.mkdir(parents=True, exist_ok=True)
            source = bundle.extractfile(member)
            if source is None:
                raise ValueError(f"Cannot read archive member: {name}")
            with source, target.open("wb") as stream:
                shutil.copyfileobj(source, stream)
            found.add(name)
    if found != set(expected):
        raise ValueError(f"Archive is missing files: {sorted(set(expected) - found)}")
    problems = verify(component, output)
    if problems:
        raise ValueError("; ".join(problems))


def fetch(component: dict, destination: Path) -> bool:
    """Install verified files, preserving unrelated files and generated caches."""
    destination = destination.expanduser().resolve()
    if not verify(component, destination):
        return False
    archive = component["archive"]
    url = archive["url"]
    if not url.startswith(("https://", "file://")):
        raise ValueError("Asset URLs must use HTTPS")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="php-assets-", dir=destination.parent
    ) as scratch:
        scratch = Path(scratch)
        download = scratch / "download.tar.gz"
        total = 0
        with urlopen(url, timeout=60) as response, download.open("wb") as stream:
            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                total += len(chunk)
                if total > archive["size"]:
                    raise ValueError("Download exceeds the declared archive size")
                stream.write(chunk)
        if total != archive["size"] or sha256(download) != archive["sha256"]:
            raise ValueError(
                "Archive checksum/size mismatch; existing assets were not replaced"
            )
        extracted = scratch / "extracted"
        extracted.mkdir()
        extract_verified(download, component, extracted)
        # Validate all destination paths before replacing any file.
        for entry in component["files"]:
            target = destination / relative_path(entry["path"])
            if target.is_symlink() or not target.resolve().is_relative_to(destination):
                raise ValueError(f"Refusing a symlinked destination: {entry['path']}")
        for entry in component["files"]:
            relative = relative_path(entry["path"])
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(extracted / relative, target)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("component", choices=("databases", "student", *MOTION_COMPONENTS))
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument(
        "--verify", action="store_true", help="Verify local files without downloading"
    )
    args = parser.parse_args()
    try:
        component = read_component(args.manifest, args.component)
        destination = (args.destination or destination_for(args.component)).expanduser()
        if args.verify:
            problems = verify(component, destination)
            if problems:
                raise ValueError("; ".join(problems))
        else:
            changed = fetch(component, destination)
            print(
                "Downloaded and verified."
                if changed
                else "Already present and verified."
            )
        print(f"{args.component}: {destination.resolve()}")
    except (OSError, ValueError, KeyError, tarfile.TarError) as exc:
        parser.exit(1, f"Asset setup failed: {exc}\n")


if __name__ == "__main__":
    main()
