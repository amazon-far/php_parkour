"""Content-hashing helpers for the byte-identical refactor tests.

The refactor of ``motion_matching`` is constrained to produce byte-identical
generation outputs. These helpers hash arrays/meshes by *content* (not raw file
bytes) so that incidental container metadata (``np.savez`` zip timestamps,
trimesh export ordering) does not produce spurious mismatches.
"""

from __future__ import annotations

import hashlib

import numpy as np


def hash_array(arr: np.ndarray) -> str:
    """Content hash of a single ndarray: dtype + shape + raw bytes."""
    h = hashlib.sha256()
    h.update(str(arr.dtype).encode())
    h.update(str(arr.shape).encode())
    h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


def hash_npz(path: str) -> str:
    """Content hash of an ``.npz`` archive, independent of zip metadata.

    Hashes ``sorted(keys)`` plus each array's dtype/shape/bytes, so two archives
    with identical arrays but different (timestamped) zip framing hash equal.
    """
    data = np.load(path)
    h = hashlib.sha256()
    for key in sorted(data.files):
        arr = data[key]
        h.update(key.encode())
        h.update(str(arr.dtype).encode())
        h.update(str(arr.shape).encode())
        h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


def hash_npy(path: str) -> str:
    """Content hash of a single ``.npy`` array file."""
    return hash_array(np.load(path))


def hash_obj(path: str) -> str:
    """Content hash of a terrain ``.obj`` mesh by its vertices and faces.

    Hashes parsed geometry rather than file text because trimesh's exporter can
    vary incidental formatting; vertices/faces are the byte-identical target.
    """
    import trimesh

    mesh = trimesh.load(path, process=False)
    h = hashlib.sha256()
    h.update(np.asarray(mesh.vertices, dtype=np.float64).tobytes())
    h.update(np.asarray(mesh.faces, dtype=np.int64).tobytes())
    return h.hexdigest()
