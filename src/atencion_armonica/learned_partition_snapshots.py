"""Atomic, immutable snapshots at completed update boundaries.

Snapshot integrity is NOT permission to resume an attempt. The campaign gate
must separately bind parent terminal receipts, runtime and cumulative budget.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

from .partial_compatibility_cache import encoded, sha_file
from .structured_source_artifacts import safe_member, write_json
from .learned_partition_state import SCHEMA as STATE_SCHEMA, validate_state
from .learned_partition_validation import boundary, memoized, claim_reference

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "learned-partition-snapshot-v1"


def _parents(refs, state, *, visiting=None):
    """Zero or one direct predecessor; ancestors are linked recursively.

    Equal cell identity and increasing update position are necessary integrity
    checks, not proof that the child actually ran from that ancestor. Attempt
    receipts and the campaign's dedicated resume port provide that provenance.
    """
    if not isinstance(refs, (list, tuple)) or len(refs) > 1:
        raise ValueError("snapshot accepts at most one direct predecessor")
    for ref in refs:
        parent, _ = read_snapshot(ref, expected_binding=state["binding"], _visiting=visiting)
        identity = ("arm", "reader_seed", "count", "device", "batch_hash")
        if any(parent[k] != state[k] for k in identity) or parent["steps"] >= state["steps"]:
            raise ValueError("snapshot parent has a different cell or non-increasing position")


def _location(root):
    root = Path(root)
    if root.is_symlink():
        raise ValueError("snapshot root cannot be a symlink")
    root = root.resolve()
    if not root.is_dir() or not any(root.is_relative_to(ROOT/p) for p in (".agent-work", "data/atencion_armonica")):
        raise ValueError("snapshot root must be an existing project-owned attempt")
    return root


def write_snapshot(root, name, kernel, *, parents=()):
    import torch
    root = _location(root)
    if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", name):
        raise ValueError("invalid snapshot name")
    state = kernel.state()  # Reject a partial update before any filesystem write.
    validate_state(state)
    _parents(parents, state)
    destination = root/name
    with (root/"snapshot_publish.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if destination.exists():
            raise FileExistsError("snapshot already exists; never overwrite")
        staging = Path(tempfile.mkdtemp(prefix="snapshot_staging_", dir=root))
        try:
            with (staging/"state.pt").open("xb") as handle:
                torch.save(state, handle)
                handle.flush()
                os.fsync(handle.fileno())
            manifest = {"schema": SCHEMA, "state_schema": STATE_SCHEMA, "status": "COMPLETE", "state_sha256": sha_file(staging/"state.pt"),
                        "binding_sha256": hashlib.sha256(encoded(state["binding"])).hexdigest(),
                        "position": {k: state[k] for k in ("epoch", "next_batch", "steps")},
                        "parents": list(parents)}
            write_json(staging/"manifest.json", manifest)
            with (staging/"manifest.json").open("rb") as handle:
                os.fsync(handle.fileno())
            _parents(parents, state)
            os.rename(staging, destination)  # Only publishers holding this attempt lock may write.
            descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            ref = {"path": (destination/"manifest.json").relative_to(ROOT).as_posix(),
                   "sha256": sha_file(destination/"manifest.json")}
            read_snapshot(ref, expected_binding=state["binding"])
        except BaseException as exc:
            failed = staging if staging.exists() else destination
            if failed.exists():
                write_json(failed/"INCOMPLETE.json", {"error": repr(exc)})
            raise
    return {"path": (destination/"manifest.json").relative_to(ROOT).as_posix(),
            "sha256": sha_file(destination/"manifest.json")}


@boundary
def read_snapshot(ref, *, expected_binding, _visiting=None):
    if not isinstance(ref, dict) or set(ref) != {"path", "sha256"}:
        raise ValueError("snapshot reference needs exact path and hash")
    manifest_path = safe_member(ROOT, ref["path"])
    if _visiting is not None and str(manifest_path) in _visiting:
        raise ValueError("cyclic snapshot ancestry")
    claim_reference(ref, role=SCHEMA, binding=expected_binding)
    return _read_snapshot(ref, expected_binding=expected_binding)


@memoized
def _read_snapshot(ref, *, expected_binding):
    # Recursive cache entries are published only after validating every parent.
    # The validation DAG rejects re-entry into an unfinished ancestor node.
    import torch
    manifest_path = safe_member(ROOT, ref["path"])
    folder = _location(manifest_path.parent)
    if manifest_path.name != "manifest.json" or sha_file(manifest_path) != ref["sha256"]:
        raise ValueError("snapshot manifest changed")
    files = list(folder.iterdir())
    if {p.name for p in files} != {"state.pt", "manifest.json"} or any(p.is_symlink() or not p.is_file() for p in files):
        raise ValueError("snapshot inventory is incomplete or modified")
    m = json.loads(manifest_path.read_bytes())
    if (set(m) != {"schema", "state_schema", "status", "state_sha256", "binding_sha256", "position", "parents"}
            or m["schema"] != SCHEMA or m["status"] != "COMPLETE"
            or m["state_schema"] != STATE_SCHEMA
            or m["binding_sha256"] != hashlib.sha256(encoded(expected_binding)).hexdigest()
            or sha_file(folder/"state.pt") != m["state_sha256"]):
        raise ValueError("snapshot binding or state changed")
    # Trusted local state, verified against the explicit expected SHA first.
    state = torch.load(folder/"state.pt", map_location="cpu", weights_only=False)
    validate_state(state)
    if state.get("binding") != expected_binding or m["position"] != {k: state.get(k) for k in ("epoch", "next_batch", "steps")}:
        raise ValueError("snapshot metadata differs from serialized state")
    _parents(m["parents"], state)
    if sha_file(manifest_path) != ref["sha256"] or sha_file(folder/"state.pt") != m["state_sha256"]:
        raise ValueError("snapshot changed while loading")
    return state, m
