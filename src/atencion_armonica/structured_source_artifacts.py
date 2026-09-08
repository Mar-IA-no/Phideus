"""Small immutable bundle primitives; integrity is not experiment authorization.

Stage-specific consumers must additionally check the prospective authorization,
source snapshot, ordered observation identities and expected role/roster.
"""
from __future__ import annotations

import json
from pathlib import Path, PurePosixPath

import numpy as np

from .partial_compatibility_cache import encoded, sha_file

SCHEMA = "structured-source-bundle-v1"
RESERVED = {"manifest.json", "resources.json", "FAILURE.json", "INCOMPLETE.json"}


def safe_member(root, name):
    root = Path(root).resolve()
    if (not isinstance(name, str) or not name or PurePosixPath(name).is_absolute()
            or any(p in ("", ".", "..") for p in name.split("/")) or "\\" in name):
        raise ValueError("bundle member must be a relative normalized path")
    path = root/name
    if not path.resolve().is_relative_to(root):
        raise ValueError("bundle member escapes root")
    return path


def write_json(path, value):
    with Path(path).open("xb") as handle:
        handle.write(encoded(value))


def write_npz(path, **arrays):
    if not arrays or any(np.asarray(value).dtype.hasobject for value in arrays.values()):
        raise ValueError("raw arrays cannot contain pickled objects")
    with Path(path).open("xb") as handle:
        np.savez_compressed(handle, **arrays)


def _inventory(root):
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("bundle must not depend on symbolic links")
        if path.is_file():
            files[path.relative_to(root).as_posix()] = path
    return files


def seal_bundle(root, *, role, binding, resources):
    root = Path(root)
    if not root.is_dir() or not isinstance(role, str) or not role or not isinstance(binding, dict):
        raise ValueError("invalid bundle root/role/binding")
    files = _inventory(root)
    if any(name in RESERVED or Path(name).name in {"FAILURE.json", "INCOMPLETE.json"} for name in files):
        raise ValueError("bundle already sealed or incomplete")
    if not files:
        raise ValueError("empty bundle")
    manifest = {"schema": SCHEMA, "status": "COMPLETE", "role": role, "binding": binding,
                "artifacts_sha256": {name: sha_file(path) for name, path in files.items()}}
    write_json(root/"resources.json", resources)
    manifest["resources_sha256"] = sha_file(root/"resources.json")
    write_json(root/"manifest.json", manifest)
    return manifest


def verify_bundle(root, expected_sha, *, role):
    root = Path(root)
    path = root/"manifest.json"
    if not isinstance(expected_sha, str) or len(expected_sha) != 64 or sha_file(path) != expected_sha:
        raise ValueError("bundle manifest differs from its pinned hash")
    manifest = json.loads(path.read_bytes())
    if (set(manifest) != {"schema", "status", "role", "binding", "artifacts_sha256", "resources_sha256"}
            or manifest["schema"] != SCHEMA or manifest["status"] != "COMPLETE" or manifest["role"] != role
            or not isinstance(manifest["binding"], dict) or not isinstance(manifest["artifacts_sha256"], dict)
            or not manifest["artifacts_sha256"]):
        raise ValueError("wrong bundle role/schema/completeness")
    expected = set(manifest["artifacts_sha256"]) | {"manifest.json", "resources.json"}
    files = _inventory(root)
    if set(files) != expected or any(Path(n).name in {"FAILURE.json", "INCOMPLETE.json"} for n in files):
        raise ValueError("bundle inventory is incomplete, modified or contains a failure marker")
    for name, digest in manifest["artifacts_sha256"].items():
        if name in RESERVED or sha_file(safe_member(root, name)) != digest:
            raise ValueError("scientific artifact changed")
    if sha_file(root/"resources.json") != manifest["resources_sha256"]:
        raise ValueError("resource artifact changed")
    return manifest


def mark_failure(root, exception):
    """Preserve partial output; never erase or overwrite the original failure."""
    root = Path(root)
    if not root.is_dir():
        raise ValueError("caller must own an existing output directory")
    marker = root/"FAILURE.json"
    if not marker.exists():
        write_json(marker, {"status": "INCOMPLETE", "error": repr(exception)})
