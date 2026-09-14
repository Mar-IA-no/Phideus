"""Lossless deterministic diagnostic bundles and immutable publication.

Only explicitly initialized diagnostic roots may receive writes. No source
artifact is changed; no orphaned output tree is adopted or repaired.
"""
from __future__ import annotations

import base64
import gzip
import hashlib
from io import BytesIO
import json
import math
import os
from pathlib import Path
import uuid

import numpy as np

from .operator_objective_sources import AuthenticatedReader, encoded


def _pack(value):
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in "biuf" or not np.isfinite(value).all():
            raise ValueError("bundle arrays must be finite primitive numeric data")
        a = np.ascontiguousarray(value)
        return {"__ndarray__": {"dtype": a.dtype.str, "shape": list(a.shape),
                               "data": base64.b64encode(a.tobytes()).decode("ascii")}}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        if any(type(k) is not str for k in value) or "__ndarray__" in value:
            raise ValueError("bundle mappings require string keys without reserved tags")
        return {k: _pack(v) for k, v in sorted(value.items())}
    if isinstance(value, (tuple, list)):
        return [_pack(v) for v in value]
    if value is None or type(value) in (str, bool, int, float):
        return value
    raise ValueError("unsupported scientific bundle value")


def _unpack(value):
    if isinstance(value, dict) and "__ndarray__" in value:
        if set(value) != {"__ndarray__"} or set(value["__ndarray__"]) != {"dtype", "shape", "data"}:
            raise ValueError("invalid bundle array tag")
        a = value["__ndarray__"]
        dtype = np.dtype(a["dtype"])
        shape = a["shape"]
        if (dtype.kind not in "biuf" or not isinstance(shape, list) or len(shape) > 4
                or any(type(i) is not int or i < 0 for i in shape)):
            raise ValueError("invalid bundle array dtype or shape")
        raw = base64.b64decode(a["data"], validate=True)
        if math.prod(shape)*dtype.itemsize != len(raw):
            raise ValueError("bundle array shape differs from decoded bytes")
        result = np.frombuffer(raw, dtype=dtype).reshape(shape).copy()
        if not np.isfinite(result).all():
            raise ValueError("nonfinite bundle array")
        return result
    if isinstance(value, dict):
        return {k: _unpack(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_unpack(v) for v in value]
    return value


def bundle_bytes(value):
    """Canonical JSON+base64 arrays, gzip3/mtime0; arrays retain exact bits."""
    decoded = encoded(_pack(value))
    raw = gzip.compress(decoded, compresslevel=3, mtime=0)
    return raw, {"codec": "diagnostic-json-arrays-gzip3-v1", "bytes": len(raw),
                 "sha256": hashlib.sha256(raw).hexdigest(), "decoded_bytes": len(decoded),
                 "decoded_sha256": hashlib.sha256(decoded).hexdigest()}


def load_bundle(raw, receipt, *, maximum_decoded_bytes=32*1024**2):
    if (receipt["codec"] != "diagnostic-json-arrays-gzip3-v1" or type(receipt["decoded_bytes"]) is not int
            or not 0 <= receipt["decoded_bytes"] <= maximum_decoded_bytes
            or len(raw) != receipt["bytes"] or hashlib.sha256(raw).hexdigest() != receipt["sha256"]):
        raise ValueError("invalid scientific bundle receipt")
    with gzip.GzipFile(fileobj=BytesIO(raw), mode="rb") as stream:
        decoded = stream.read(receipt["decoded_bytes"]+1)
    if len(decoded) != receipt["decoded_bytes"] or hashlib.sha256(decoded).hexdigest() != receipt["decoded_sha256"]:
        raise ValueError("scientific decoded bundle differs")
    value = json.loads(decoded)
    if encoded(value) != decoded or gzip.compress(decoded, compresslevel=3, mtime=0) != raw:
        raise ValueError("scientific bundle codec differs")
    return _unpack(value)


def atomic_bytes(path, raw):
    """Publish via non-replacing hard link; preserve exact staging on failure."""
    path = Path(path)
    if type(raw) is not bytes:
        raise ValueError("publication requires completed bytes")
    staging = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    with staging.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    os.link(staging, path)
    staging.unlink()  # Remove only this invocation's extra link after publication.
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


class DiagnosticStore:
    def __init__(self, root, *, project_root):
        project = Path(project_root)
        root = Path(root)
        if (not project.is_absolute() or not root.is_absolute()
                or root != Path(os.path.abspath(root)) or not root.is_relative_to(project)):
            raise ValueError("diagnostic store requires a canonical project-owned root")
        relative = root.relative_to(project).as_posix()
        canonical = "data/atencion_armonica/operator_objective_alignment_v1"
        fixture_base = ".agent-work/phideus-operator-objective-20260914"
        if relative != canonical and not relative.startswith(fixture_base+"/"):
            raise ValueError("diagnostic store is outside its declared output/staging territory")
        if any(p.is_symlink() for p in [root, *root.parents]):
            raise ValueError("diagnostic output cannot traverse symlinks")
        self.root, self.project = root, project

    def initialize(self, manifest):
        if manifest.get("status") != "PREPARED":
            raise ValueError("initial manifest must be PREPARED")
        if self.root.exists():
            raise FileExistsError("explicit initialize never adopts an existing output root")
        self.root.mkdir(parents=True)
        self.publish_json("manifest.json", manifest, initializing=True)

    def manifest(self):
        path = self.root/"manifest.json"
        if not path.is_file() or path.is_symlink():
            raise ValueError("diagnostic root has no canonical manifest")
        raw = path.read_bytes()
        value = json.loads(raw)
        if encoded(value) != raw or value.get("status") != "PREPARED":
            raise ValueError("diagnostic manifest changed or is not canonical")
        return value, {"path": "manifest.json", "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}

    def path(self, relative):
        parts = AuthenticatedReader.parts(relative)
        path = self.root.joinpath(*parts)
        if any(p.is_symlink() for p in [path, *path.parents]):
            raise ValueError("diagnostic member cannot traverse symlinks")
        return path

    def publish(self, relative, raw, *, initializing=False):
        if not initializing:
            self.manifest()
        path = self.path(relative)
        if path.exists():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return {"path": relative, **atomic_bytes(path, raw)}

    def publish_json(self, relative, value, *, initializing=False):
        return self.publish(relative, encoded(value), initializing=initializing)

    def read(self, ref):
        return AuthenticatedReader(self.root).bytes(ref)

    def json(self, ref):
        return AuthenticatedReader(self.root).json(ref)
