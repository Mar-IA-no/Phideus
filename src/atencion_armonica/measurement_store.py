"""Immutable measurement artifacts and recoverable payload/receipt publication.

No experiment admission or model execution permission. A caller must hold the
exclusive store lock, authenticate stage inputs and account execution attempts.
Existing historical store allowlists and frozen implementation stay untouched.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import fcntl
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path

import numpy as np

from . import generative_evidence_storage as storage
from .measurement_contract import reference as validate_reference
from .partial_compatibility_cache import encoded

ROOT = Path(__file__).resolve().parents[2]
BASES = (ROOT/"data/atencion_armonica/operator_under_measurement_v1",
         ROOT/".agent-work/phideus-measurement-20260915")
META = "__measurement_metadata__"


class MeasurementStore:
    def __init__(self, root, *, binding):
        candidate = Path(root).absolute()
        if (not any(candidate.is_relative_to(base) for base in BASES)
                or ".." in candidate.parts
                or any(p.is_symlink() for p in (candidate, *candidate.parents))):
            raise ValueError("measurement store requires its owned root without symlinks")
        if not isinstance(binding, dict) or not binding or json.loads(encoded(binding)) != binding:
            raise ValueError("canonical explicit binding required")
        self._root, self._binding_bytes, self._locked = candidate, encoded(binding), False
        self._lock_handle, self._lock_root = None, None
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.path("binding.json")
        if path.exists():
            if self.read(self.reference("binding.json")) != self._binding_bytes:
                raise ValueError("existing measurement store has another binding")
        else:
            if any(self.root.iterdir()):
                raise ValueError("cannot adopt unbound existing artifacts")
            storage.write_json(path, binding)

    @property
    def root(self):
        return self._root

    @property
    def binding(self):
        return json.loads(self._binding_bytes)

    def _check_binding(self):
        if self.path("binding.json").read_bytes() != self._binding_bytes:
            raise ValueError("binding differs from pinned canonical bytes")

    def _require_lock(self):
        if (not self._locked or self._lock_handle is None or self._lock_handle.closed
                or self._lock_root != self.root):
            raise RuntimeError("publication requires exclusive store ownership")
        held = os.fstat(self._lock_handle.fileno())
        current = self.path("operation.lock").stat()
        if (held.st_dev, held.st_ino) != (current.st_dev, current.st_ino):
            raise RuntimeError("store lock no longer names the acquired file")
        self._check_binding()

    def path(self, relative):
        if not any(self.root.is_relative_to(base) for base in BASES) or ".." in self.root.parts:
            raise ValueError("measurement root outside its allowlist")
        validate_reference({"path": relative, "sha256": "0"*64, "bytes": 0})
        result = self.root/relative
        if any(p.is_symlink() for p in (result, *result.parents)):
            raise ValueError("artifact path traverses a symlink")
        return result

    def reference(self, relative):
        raw = self.path(relative).read_bytes()
        return {"path": relative, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}

    def read(self, ref):
        ref = validate_reference(ref)
        raw = self.path(ref["path"]).read_bytes()
        if len(raw) != ref["bytes"] or hashlib.sha256(raw).hexdigest() != ref["sha256"]:
            raise ValueError("artifact differs from pinned bytes")
        return raw

    def json(self, ref):
        raw = self.read(ref)
        value = json.loads(raw)
        if encoded(value) != raw:
            raise ValueError("noncanonical artifact JSON")
        return value

    @contextmanager
    def exclusive(self):
        if self._locked:
            raise RuntimeError("store lock cannot be nested")
        with self.path("operation.lock").open("a+b") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._locked = True
            self._lock_handle, self._lock_root = handle, self.root
            try:
                self._require_lock()
                yield self
            finally:
                self._locked = False
                self._lock_handle, self._lock_root = None, None
                fcntl.flock(handle, fcntl.LOCK_UN)

    def publish_json(self, relative, value):
        self._require_lock()
        path = self.path(relative)
        raw = encoded(value)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            ref = self.reference(relative)
            if self.read(ref) != raw:
                raise ValueError("cannot replace published artifact")
            return ref
        storage.atomic_bytes(path, raw)
        return self.reference(relative)

    def _payload(self, ref, expected_identity):
        raw = self.read(ref)
        with np.load(BytesIO(raw), allow_pickle=False) as archive:
            if len(archive.files) != len(set(archive.files)) or META not in archive.files:
                raise ValueError("payload has duplicate arrays or no metadata")
            meta = archive[META]
            if meta.dtype != np.uint8 or meta.ndim != 1:
                raise ValueError("invalid metadata bytes")
            record = json.loads(meta.tobytes())
            if encoded(record) != meta.tobytes():
                raise ValueError("noncanonical payload metadata")
            names = record.get("array_names")
            if (not isinstance(names, list) or any(not isinstance(k, str) or not k or k == META for k in names)
                    or names != sorted(set(names))
                    or set(archive.files) != {META, *(f"array_{i:06d}" for i in range(len(names)))}):
                raise ValueError("payload logical/physical array roster differs")
            arrays = {k: archive[f"array_{i:06d}"] for i, k in enumerate(names)}
        if (set(record) != {"schema", "binding", "identity", "result", "array_names"}
                or record["schema"] != "measurement-stage-payload-v1"
                or encoded(record["binding"]) != self._binding_bytes
                or encoded(record["identity"]) != encoded(expected_identity)
                or record["array_names"] != sorted(arrays)
                or not isinstance(record["result"], dict)
                or any(a.dtype.hasobject for a in arrays.values())):
            raise ValueError("payload identity, schema or array roster differs")
        return record["result"], arrays

    def completed(self, folder, identity):
        """Return validated stage data, or None if no published payload exists.

An intact orphan payload can close its receipt without calling a producer.
Its binding/operation/inputs must match the independently authenticated caller.
This operation does not by itself reconcile missing execution-cost receipts.
"""
        self._require_lock()
        payload_name, receipt_name = folder+"/payload.npz", folder+"/complete.json"
        identity = deepcopy(identity)
        if self.path(receipt_name).exists():
            receipt = self.json(self.reference(receipt_name))
            if (set(receipt) != {"schema", "binding", "identity", "payload"}
                    or receipt["schema"] != "measurement-stage-complete-v1"
                    or encoded(receipt["binding"]) != self._binding_bytes
                    or encoded(receipt["identity"]) != encoded(identity)
                    or receipt["payload"]["path"] != payload_name):
                raise ValueError("completion receipt identity differs")
            result, arrays = self._payload(receipt["payload"], identity)
            return self.reference(receipt_name), result, arrays
        if not self.path(payload_name).exists():
            return None
        payload_ref = self.reference(payload_name)
        result, arrays = self._payload(payload_ref, identity)
        ref = self.publish_json(receipt_name, {"schema": "measurement-stage-complete-v1",
            "binding": self.binding, "identity": identity, "payload": payload_ref})
        return ref, result, arrays

    def publish_stage(self, folder, identity, result, arrays):
        self._require_lock()
        if (not isinstance(identity, dict) or not identity or not isinstance(result, dict)
                or not isinstance(arrays, dict) or META in arrays
                or any(not isinstance(k, str) or not k or not isinstance(a, np.ndarray)
                       or a.dtype.hasobject for k, a in arrays.items())):
            raise ValueError("invalid stage identity/result/arrays")
        existing = self.completed(folder, identity)
        if existing is not None:
            ref, old, saved = existing
            if (encoded(old) != encoded(result) or set(saved) != set(arrays)
                    or any(saved[k].dtype != arrays[k].dtype or saved[k].shape != arrays[k].shape
                           or saved[k].tobytes() != arrays[k].tobytes() for k in saved)):
                raise ValueError("cannot replace a completed stage")
            return ref
        record = {"schema": "measurement-stage-payload-v1", "binding": self.binding,
                  "identity": deepcopy(identity), "result": result, "array_names": sorted(arrays)}
        packed = {f"array_{i:06d}": arrays[k] for i, k in enumerate(sorted(arrays))}
        packed[META] = np.frombuffer(encoded(record), dtype=np.uint8)
        path = self.path(folder+"/payload.npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        storage.write_arrays(path, packed)
        return self.completed(folder, identity)[0]
