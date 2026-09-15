"""Source/runtime/reuse pins for an explicitly named measurement runner.

Capturing a snapshot is read-only and is not permission to run tests or CUDA.
The complete attention-harmonic source package is pinned, including lazy-import
dependencies; unrelated project packages and private agent memories are excluded.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import platform
from pathlib import Path

from .measurement_reuse import ROOT, load_reuse, read_reference
from .partial_compatibility_cache import encoded

ENTRY = "experiments/atencion_armonica/run_operator_under_measurement.py"
DOCUMENTS = ("experiments/atencion_armonica/PLAN_OPERATOR_UNDER_MEASUREMENT.md",
             "experiments/atencion_armonica/PROTOCOL_OPERATOR_UNDER_MEASUREMENT.md")


def source_paths():
    base = ROOT/"src/atencion_armonica"
    return sorted({ENTRY, *DOCUMENTS, *(p.relative_to(ROOT).as_posix() for p in base.rglob("*.py")),
                   *(p.relative_to(ROOT).as_posix() for p in
                     (ROOT/"experiments/atencion_armonica").glob("test_measurement_*.py"))})


def runtime_versions():
    return {"python": platform.python_version(), **{name: importlib.metadata.version(name)
            for name in ("numpy", "scipy", "scikit-learn", "torch")}}


def capture():
    files = []
    for name in source_paths():
        path = ROOT/name
        if any(p.is_symlink() for p in (path, *path.parents)):
            raise ValueError("source snapshot must not traverse symlinks")
        raw = path.read_bytes()
        files.append({"path": name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})
    reused = load_reuse()
    return {"schema": "measurement-source-snapshot-v1", "entry": ENTRY,
            "files": files, "reuse": reused["references"], "runtime": runtime_versions()}


def verify(store, ref):
    snapshot = store.json(ref)
    if (set(snapshot) != {"schema", "entry", "files", "reuse", "runtime"}
            or snapshot["schema"] != "measurement-source-snapshot-v1" or snapshot["entry"] != ENTRY
            or [r["path"] for r in snapshot["files"]] != source_paths()
            or encoded(snapshot["runtime"]) != encoded(runtime_versions())):
        raise ValueError("source closure, entry point or runtime changed")
    for source in snapshot["files"]:
        read_reference(source)
    expected_reuse = load_reuse()["references"]
    if encoded(snapshot["reuse"]) != encoded(expected_reuse):
        raise ValueError("frozen reuse inventory changed")
    return snapshot
