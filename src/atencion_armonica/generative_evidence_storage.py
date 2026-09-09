"""Lossless scene storage and atomic, non-overwriting receipts.

No campaign permissions: callers bind provenance and own the destination.
Factors remain complete canonical JSON; compact arrays have no object/pickle.
"""
from __future__ import annotations

import gzip
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import uuid

import numpy as np

from .partial_compatibility_cache import encoded


def atomic_bytes(path, raw):
    """Publish a completed file with link(2), which cannot replace an existing name."""
    path = Path(path)
    if not isinstance(raw, bytes):
        raise TypeError("atomic publication requires bytes")
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    staging = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    # On an exceptional write/link keep this exact owned staging file recoverable.
    with staging.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    os.link(staging, path)
    staging.unlink()  # Only the just-created extra link, not historical data.
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def write_json(path, value):
    return atomic_bytes(path, encoded(value))


def write_scene(path, value):
    raw = encoded(value)
    compressed = gzip.compress(raw, compresslevel=3, mtime=0)
    return {**atomic_bytes(path, compressed), "decoded_sha256": hashlib.sha256(raw).hexdigest(),
            "decoded_bytes": len(raw), "codec": "canonical-json-gzip3-mtime0"}


def read_scene(path, receipt):
    if (set(receipt) != {"sha256", "bytes", "decoded_sha256", "decoded_bytes", "codec"}
            or receipt["codec"] != "canonical-json-gzip3-mtime0"
            or any(type(receipt[k]) is not int or receipt[k] < 0 for k in ("bytes", "decoded_bytes"))):
        raise ValueError("wrong scene codec or size schema")
    compressed = Path(path).read_bytes()
    if (hashlib.sha256(compressed).hexdigest() != receipt["sha256"]
            or len(compressed) != receipt["bytes"]):
        raise ValueError("compressed scene identity differs")
    # Bounded by a separately pinned receipt, not by the gzip length header.
    with gzip.GzipFile(fileobj=BytesIO(compressed), mode="rb") as stream:
        raw = stream.read(receipt["decoded_bytes"]+1)
    if len(raw) != receipt["decoded_bytes"] or hashlib.sha256(raw).hexdigest() != receipt["decoded_sha256"]:
        raise ValueError("decoded scene identity differs")
    value = json.loads(raw)
    if encoded(value) != raw:
        raise ValueError("scene JSON is not canonical")
    if gzip.compress(raw, compresslevel=3, mtime=0) != compressed:
        raise ValueError("gzip bytes do not implement the declared canonical codec")
    return value


def write_arrays(path, arrays):
    if (not isinstance(arrays, dict) or not arrays
            or any(not isinstance(k, str) or not k or not isinstance(v, np.ndarray)
                   or v.dtype.hasobject for k, v in arrays.items())):
        raise ValueError("array archive must have named NumPy arrays without pickle")
    stream = BytesIO()
    np.savez_compressed(stream, **arrays)
    return atomic_bytes(path, stream.getvalue())


def read_arrays(path, receipt):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != receipt["sha256"] or len(raw) != receipt["bytes"]:
        raise ValueError("array archive identity differs")
    with np.load(BytesIO(raw), allow_pickle=False) as archive:
        if len(archive.files) != len(set(archive.files)):
            raise ValueError("duplicate array names")
        return {k: archive[k] for k in archive.files}
