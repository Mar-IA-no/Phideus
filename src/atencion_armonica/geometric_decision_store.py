"""Owned immutable artifacts; no dataset discovery or execution permission.

Checkpoint pickle is loaded only after authenticating a receipt written in
this trusted local store. External/untrusted checkpoints are not an input port.
"""
from __future__ import annotations

import copy
import hashlib
from io import BytesIO
import json
from pathlib import Path
import uuid

import torch

from . import generative_evidence_storage as storage
from .generative_evidence_cell import state_digest
from .generative_evidence_reuse import ROOT
from .partial_compatibility_cache import encoded

BASES = (ROOT/"data/atencion_armonica/geometric_decision_energy_v1",
         ROOT/".agent-work/phideus-geometric-decision-20260914")
STATE_SCHEMA = "geometric-decision-snapshot-v1"


class ArtifactStore:
    def __init__(self, root, *, binding):
        candidate = Path(root).absolute()
        if (not any(candidate.is_relative_to(base) and candidate != base for base in BASES)
                or ".." in candidate.parts or any(p.is_symlink() for p in (candidate, *candidate.parents))):
            raise ValueError("store requires an owned geometric-decision directory without symlinks")
        if not isinstance(binding, dict) or not binding or json.loads(encoded(binding)) != binding:
            raise ValueError("canonical explicit artifact binding required")
        self.root, self.binding = candidate, copy.deepcopy(binding)
        if self.root.exists() and not self.root.is_dir():
            raise ValueError("artifact root must be a directory")
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.path("binding.json")
        if path.exists():
            if self.json(self.reference(path)) != binding:
                raise ValueError("existing store has another binding")
        else:
            if any(self.root.iterdir()):
                raise ValueError("cannot adopt existing unbound artifacts")
            storage.write_json(path, binding)

    def path(self, relative):
        if type(relative) is not str or not relative:
            raise ValueError("explicit relative artifact path required")
        p = Path(relative)
        if p.is_absolute() or ".." in p.parts or p.as_posix() != relative or relative == ".":
            raise ValueError("artifact path escapes or is not canonical")
        result = self.root/p
        if any(q.is_symlink() for q in (result, *result.parents)):
            raise ValueError("artifact path traverses a symlink")
        return result

    def reference(self, path):
        path = self.path(Path(path).relative_to(self.root).as_posix())
        raw = path.read_bytes()
        return {"path": path.relative_to(self.root).as_posix(),
                "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}

    def read(self, ref):
        if (not isinstance(ref, dict) or set(ref) != {"path", "sha256", "bytes"}
                or type(ref["sha256"]) is not str or len(ref["sha256"]) != 64
                or any(c not in "0123456789abcdef" for c in ref["sha256"])
                or type(ref["bytes"]) is not int or ref["bytes"] < 0):
            raise ValueError("authenticated artifact receipt required")
        raw = self.path(ref["path"]).read_bytes()
        if len(raw) != ref["bytes"] or hashlib.sha256(raw).hexdigest() != ref["sha256"]:
            raise ValueError("artifact differs from pinned bytes")
        return raw

    def json(self, ref):
        raw = self.read(ref)
        result = json.loads(raw)
        if encoded(result) != raw:
            raise ValueError("artifact JSON is not canonical")
        return result

    def publish_json(self, relative, value):
        path = self.path(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            ref = self.reference(path)
            if self.read(ref) != encoded(value):
                raise ValueError("cannot replace published artifact")
            return ref
        storage.write_json(path, value)
        return self.reference(path)

    def publish_arrays(self, relative, arrays):
        path = self.path(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        storage.write_arrays(path, arrays)
        return self.reference(path)

    def arrays(self, ref):
        import numpy as np
        with np.load(BytesIO(self.read(ref)), allow_pickle=False) as a:
            if len(a.files) != len(set(a.files)):
                raise ValueError("duplicate array names")
            return {k: a[k] for k in a.files}

    def save_state(self, kernel, previous):
        state = kernel.state()
        if state["binding"] != self.binding:
            raise ValueError("kernel and store bindings differ")
        if previous is not None:
            parent = self.json(previous)
            self._state_record(parent)
            if parent["steps"] >= state["steps"]:
                raise ValueError("snapshot parent must precede current step")
        elif state["steps"] != 0:
            raise ValueError("noninitial snapshot requires its parent")
        digest = state_digest(state)
        relative = f"snapshots/step_{state['steps']:06d}.json"
        path = self.path(relative)
        identity = {"schema": STATE_SCHEMA, "binding": self.binding,
                    **{k: state[k] for k in ("steps", "epoch", "next_batch")},
                    "previous": previous, "state_digest": digest}
        if path.exists():
            ref = self.reference(path)
            record = self.json(ref)
            if (set(record) != set(identity) | {"state"}
                    or any(record[k] != v for k, v in identity.items())
                    or state_digest(self.load_state(ref)) != digest):
                raise ValueError("cannot replace a published snapshot")
            return ref
        stream = BytesIO()
        torch.save(state, stream)
        blob = self.path(f"snapshots/state-{uuid.uuid4().hex}.pt")
        blob.parent.mkdir(parents=True, exist_ok=True)
        storage.atomic_bytes(blob, stream.getvalue())
        blob_ref = self.reference(blob)
        saved = torch.load(BytesIO(self.read(blob_ref)), map_location="cpu", weights_only=False)
        if state_digest(saved) != digest:
            raise ValueError("snapshot serialization changed state")
        return self.publish_json(relative, {**identity, "state": blob_ref})

    def _state_record(self, record):
        if (set(record) != {"schema", "binding", "steps", "epoch", "next_batch", "previous", "state", "state_digest"}
                or record["schema"] != STATE_SCHEMA or record["binding"] != self.binding
                or any(type(record[k]) is not int or record[k] < 0 for k in ("steps", "epoch", "next_batch"))):
            raise ValueError("snapshot receipt schema or binding differs")

    def load_state(self, ref):
        record = self.json(ref)
        self._state_record(record)
        state = torch.load(BytesIO(self.read(record["state"])), map_location="cpu", weights_only=False)
        if (state_digest(state) != record["state_digest"] or state["binding"] != self.binding
                or any(state[k] != record[k] for k in ("steps", "epoch", "next_batch"))):
            raise ValueError("snapshot content does not match receipt")
        return state

    def latest(self, kernel):
        """Restore only a complete linked chain; orphan blobs are not checkpoints."""
        previous, previous_step = None, -1
        for path in sorted(self.path("snapshots").glob("step_*.json")):
            ref = self.reference(path)
            record = self.json(ref)
            self._state_record(record)
            if (path.name != f"step_{record['steps']:06d}.json" or record["previous"] != previous
                    or record["steps"] <= previous_step or (previous is None and record["steps"] != 0)):
                raise ValueError("snapshot chain lost or changed a parent")
            # Authenticate every ancestor's state bytes, not merely its receipt.
            self.read(record["state"])
            previous, previous_step = ref, record["steps"]
        if previous is not None:
            kernel.restore(self.load_state(previous))
        return previous
