"""Exercise the public asset acquisition contract without remote credentials."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import tarfile
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "download_assets",
    Path(__file__).resolve().parents[1] / "scripts/download_assets.py",
)
assets = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(assets)


def make_bundle(tmp_path, files=None):
    files = files or {"terrain/box.json": b'{"size": 1}', "example.bin": b"motion"}
    archive = tmp_path / "assets.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    component = {
        "status": "ready",
        "archive": {
            "url": archive.as_uri(),
            "size": archive.stat().st_size,
            "sha256": assets.sha256(archive),
        },
        "files": [
            {
                "path": name,
                "size": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
            for name, data in files.items()
        ],
    }
    return component


def test_download_verify_and_reuse_without_network(tmp_path, monkeypatch):
    component = make_bundle(tmp_path)
    dest = tmp_path / "installed"
    assert assets.fetch(component, dest)
    assert assets.verify(component, dest) == []
    cache = dest / "features.pkl"
    cache.write_bytes(b"generated locally")

    def no_network(*args, **kwargs):
        raise AssertionError("Verified cache must not require a remote service")

    monkeypatch.setattr(assets, "urlopen", no_network)
    assert not assets.fetch(component, dest)
    assert cache.read_bytes() == b"generated locally"


def test_bad_archive_does_not_overwrite_existing_files(tmp_path):
    component = make_bundle(tmp_path)
    dest = tmp_path / "installed"
    dest.mkdir()
    existing = dest / "example.bin"
    existing.write_bytes(b"keep me")
    component["archive"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="checksum"):
        assets.fetch(component, dest)
    assert existing.read_bytes() == b"keep me"


@pytest.mark.parametrize(
    "name", ["../escape", "/absolute", "nested/../../escape", "nested\\escape"]
)
def test_unsafe_archive_paths_are_rejected(tmp_path, name):
    component = make_bundle(tmp_path, {name: b"bad"})
    with pytest.raises(ValueError, match="Unsafe"):
        assets.fetch(component, tmp_path / "installed")
    assert not (tmp_path / "escape").exists()


def test_unpublished_student_has_clear_readiness_error(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "components": {
                    "student": {
                        "status": "pending",
                        "message": "The final student is not available yet",
                    }
                },
            }
        )
    )
    with pytest.raises(ValueError, match="final student"):
        assets.read_component(path, "student")


def test_manifest_file_digest_is_checked_before_install(tmp_path):
    component = make_bundle(tmp_path)
    component["files"][0]["sha256"] = "0" * 64
    dest = tmp_path / "installed"
    with pytest.raises(ValueError, match="checksum"):
        assets.fetch(component, dest)
    assert not dest.exists()
