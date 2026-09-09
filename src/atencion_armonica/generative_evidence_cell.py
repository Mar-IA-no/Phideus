"""One fixed 50-epoch cell with immutable snapshots and calibration outputs.

No file/data discovery, CUDA selection, fresh tests or scientific selection.
The caller authenticates the full delivered TRAIN/calibration arrays, binds
their references, and supplies an external resource supervisor. This layer
never accepts a reduced roster or stops successfully at a partial epoch.
"""
from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path
import uuid

import numpy as np
import torch

from . import generative_evidence as ge
from . import generative_evidence_cache as cache
from . import generative_evidence_storage as storage
from .generative_evidence_model import collate
from .generative_evidence_reuse import ROOT
from .generative_evidence_training import TrainingKernel, collate_targets
from .partial_compatibility_cache import encoded
from .structured_source_artifacts import safe_member


def state_digest(value):
    """Content identity independent of torch.save zip container/object handles."""
    digest = hashlib.sha256()
    def visit(v):
        if isinstance(v, torch.Tensor):
            digest.update(b"torch")
            visit(v.detach().cpu().numpy())
        elif isinstance(v, np.ndarray):
            digest.update(encoded(["numpy", v.dtype.str, list(v.shape)]))
            digest.update(v.tobytes(order="C"))
        elif isinstance(v, dict):
            digest.update(b"dict[")
            for k in sorted(v, key=lambda k: encoded(k)):
                visit(k)
                visit(v[k])
            digest.update(b"]")
        elif isinstance(v, (list, tuple)):
            digest.update(b"list[" if isinstance(v, list) else b"tuple[")
            for item in v:
                visit(item)
            digest.update(b"]")
        else:
            digest.update(encoded([type(v).__name__, v]))
    visit(value)
    return digest.hexdigest()


class CellData:
    """Authenticated caller data, not an authority to read its referenced files.

    Every row has only identity, delivered input, candidate partitions and
    normalized targets. Counts are exact; empty candidates stay in the roster.
    """
    def __init__(self, train, calibration, *, binding):
        cache._binding(binding)
        self.binding = binding
        self.rows = {"train": train, "calibration": calibration}
        self.eligible = {}
        for split, count in (("train", 4096), ("calibration", 512)):
            rows = self.rows[split]
            if not isinstance(rows, list) or len(rows) != count:
                raise ValueError("cell requires the complete 4096/512 roster")
            seen, eligible = set(), []
            for scene_id, row in enumerate(rows):
                if (set(row) != {"scene_id", "identity", "inputs", "partitions", "targets"}
                        or type(row["scene_id"]) is not int or row["scene_id"] != scene_id
                        or not isinstance(row["identity"], str) or len(row["identity"]) != 64
                        or row["identity"] in seen or set(row["inputs"]) != {"groups", "globals", "evidence", "incidence"}):
                    raise ValueError("cell scene identity, order or schema differs")
                seen.add(row["identity"])
                ps = [ge.law.signature(p) for p in row["partitions"]]
                target = row["targets"]
                if (ps != sorted(set(ps)) or len(ps) > ge.MAX_CANDIDATES
                        or any(not ge.law.supported(p) for p in ps)
                        or not isinstance(target, np.ndarray) or target.dtype != np.float32
                        or target.shape != (len(ps), 2) or not np.isfinite(target).all()
                        or np.any(target < 0) or np.any(target > 1)):
                    raise ValueError("cell candidate/target extent differs")
                x = row["inputs"]
                groups = x["groups"]
                if (any(not isinstance(v, np.ndarray) or v.dtype != np.float32 or not np.isfinite(v).all() for v in x.values())
                        or groups.ndim != 2 or groups.shape[1] != 9 or len(groups) > ge.MAX_GROUPS
                        or x["globals"].shape != (len(ps), 17) or x["evidence"].shape != (len(ps), 6)
                        or x["incidence"].shape != (len(ps), len(groups))):
                    raise ValueError("cell delivered input extent differs")
                if ps:
                    n = sum(map(len, ps[0]))
                    ge.partitions_checked(ps, n)
                    gs = sorted({g for p in ps for g in p})
                    incidence = np.array([[len(g)/n if g in p else 0 for g in gs] for p in ps], np.float32)
                    if len(gs) != len(groups) or not np.array_equal(incidence, x["incidence"]):
                        raise ValueError("cell candidate/group incidence differs")
                    eligible.append(scene_id)
                elif len(groups):
                    raise ValueError("empty candidate scene cannot retain model groups")
            self.eligible[split] = eligible
        if not self.eligible["train"] or not self.eligible["calibration"]:
            raise ValueError("cell requires observable TRAIN and calibration support")


class CellArtifacts:
    def __init__(self, root, *, binding):
        cache._binding(binding)
        self.root = Path(root).resolve()
        canonical = ROOT/"data/atencion_armonica/generative_evidence_reader_v1/training"
        temporary = ROOT/".agent-work/phideus-generative-evidence-20260909"
        if not any(self.root.is_relative_to(base) and self.root != base for base in (canonical, temporary)):
            raise ValueError("cell artifacts require an owned cell directory")
        self.binding = binding
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root/"binding.json"
        if path.exists():
            if self.json(self.reference(path)) != binding:
                raise ValueError("cell artifact binding differs")
        else:
            if any(self.root.iterdir()):
                raise ValueError("cannot adopt an existing unbound cell directory")
            storage.write_json(path, binding)

    def path(self, name):
        result = safe_member(self.root, name)
        if any(p.is_symlink() for p in [result, *result.parents] if p.is_relative_to(self.root)):
            raise ValueError("cell artifacts cannot traverse symlinks")
        return result

    def reference(self, path):
        path = self.path(Path(path).relative_to(self.root).as_posix())
        raw = path.read_bytes()
        return {"path": path.relative_to(self.root).as_posix(), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}

    def read(self, ref):
        if set(ref) != {"path", "sha256", "bytes"}:
            raise ValueError("cell artifact reference schema differs")
        raw = self.path(ref["path"]).read_bytes()
        if len(raw) != ref["bytes"] or hashlib.sha256(raw).hexdigest() != ref["sha256"]:
            raise ValueError("cell artifact bytes differ")
        return raw

    def json(self, ref):
        import json
        raw = self.read(ref)
        value = json.loads(raw)
        if encoded(value) != raw:
            raise ValueError("cell JSON must be canonical")
        return value

    def save_state(self, kernel, previous):
        state = kernel.state()
        if state["binding"] != self.binding:
            raise ValueError("kernel and artifact bindings differ")
        folder = self.path("snapshots")
        folder.mkdir(exist_ok=True)
        path = folder/f"step_{kernel.steps:06d}.json"
        content_hash = state_digest(state)
        if path.exists():
            ref = self.reference(path)
            record = self.json(ref)
            saved = self.load_state(ref)
            if record["state_digest"] != content_hash or state_digest(saved) != content_hash:
                raise ValueError("cannot replace a completed state boundary")
            return ref
        if previous is not None:
            old = self.json(previous)
            if old["binding"] != self.binding or old["steps"] >= kernel.steps:
                raise ValueError("state parent must be an earlier bound update")
        elif kernel.steps != 0:
            raise ValueError("a non-initial state requires its prior snapshot")
        stream = BytesIO()
        torch.save(state, stream)
        blob = folder/f"state-{uuid.uuid4().hex}.pt"
        storage.atomic_bytes(blob, stream.getvalue())
        blob_ref = self.reference(blob)
        restored = torch.load(BytesIO(self.read(blob_ref)), map_location="cpu", weights_only=False)
        if state_digest(restored) != content_hash:
            raise ValueError("state serialization changed its content")
        storage.write_json(path, {"schema": "generative-evidence-cell-state-v1", "binding": self.binding,
            "steps": kernel.steps, "epoch": kernel.epoch, "next_batch": kernel.next_batch,
            "previous": previous, "state": blob_ref, "state_digest": content_hash})
        return self.reference(path)

    def load_state(self, ref):
        record = self.json(ref)
        if record["schema"] != "generative-evidence-cell-state-v1" or record["binding"] != self.binding:
            raise ValueError("state receipt binding differs")
        state = torch.load(BytesIO(self.read(record["state"])), map_location="cpu", weights_only=False)
        if (state_digest(state) != record["state_digest"] or state["binding"] != self.binding
                or any(state[k] != record[k] for k in ("steps", "epoch", "next_batch"))):
            raise ValueError("state does not match its receipt")
        return state

    def latest(self, kernel):
        paths = sorted(self.path("snapshots").glob("step_*.json"))
        previous, expected_previous = None, None
        for path in paths:
            ref = self.reference(path)
            record = self.json(ref)
            if record["previous"] != expected_previous:
                raise ValueError("state chain lost or changed a parent")
            previous = expected_previous = ref
        if previous is not None:
            kernel.restore(self.load_state(previous))
        return previous


def read_calibration(data, store, ref, state_ref, epoch):
    rows = data.rows["calibration"]
    ids = data.eligible["calibration"]
    offsets = np.r_[np.int64(0), np.cumsum([len(r["partitions"]) for r in rows], dtype=np.int64)]
    identity = {"binding": store.binding, "epoch": epoch, "state": state_ref,
                "identities": [r["identity"] for r in rows], "eligible_scene_ids": ids}
    record = store.json(ref)
    if (epoch not in range(5, 51, 5) or set(record) != set(identity) | {"predictions"}
            or any(record[k] != v for k, v in identity.items())):
        raise ValueError("calibration parent or complete roster differs")
    with np.load(BytesIO(store.read(record["predictions"])), allow_pickle=False) as a:
        components, off = a["components"], a["offsets"]
        if (set(a.files) != {"components", "offsets"} or components.dtype != np.float32
                or components.shape != (offsets[-1], 2) or not np.isfinite(components).all()
                or np.any(components < 0) or off.dtype != np.int64 or not np.array_equal(off, offsets)):
            raise ValueError("calibration prediction extent or value differs")
    return components, offsets


def calibration_outputs(kernel, data, store, state_ref, check):
    epoch = kernel.epoch
    if epoch not in range(5, 51, 5) or kernel.next_batch != 0:
        raise ValueError("calibration only at the ten declared epoch boundaries")
    rows, ids = data.rows["calibration"], data.eligible["calibration"]
    offsets = np.r_[np.int64(0), np.cumsum([len(r["partitions"]) for r in rows], dtype=np.int64)]
    folder = store.path(f"calibration/epoch_{epoch:02d}")
    folder.mkdir(parents=True, exist_ok=True)
    path = folder/"index.json"
    identity = {"binding": store.binding, "epoch": epoch, "state": state_ref,
                "identities": [r["identity"] for r in rows], "eligible_scene_ids": ids}
    if path.exists():
        ref = store.reference(path)
        read_calibration(data, store, ref, state_ref, epoch)
        return ref
    components = np.empty((offsets[-1], 2), np.float32)
    kernel.model.eval()
    for start in range(0, len(ids), 32):
        check()
        batch_ids = ids[start:start+32]
        batch = {k: v.to(kernel.device) for k, v in collate([rows[i]["inputs"] for i in batch_ids]).items()}
        with torch.no_grad():
            values = kernel.model(batch).cpu().numpy()
        for j, i in enumerate(batch_ids):
            components[offsets[i]:offsets[i+1]] = values[j, :offsets[i+1]-offsets[i]]
    check()
    blob = folder/f"predictions-{uuid.uuid4().hex}.npz"
    storage.write_arrays(blob, {"components": components, "offsets": offsets})
    storage.write_json(path, {**identity, "predictions": store.reference(blob)})
    ref = store.reference(path)
    read_calibration(data, store, ref, state_ref, epoch)
    return ref


def run_cell(data, store, *, arm, checkpoint_seed, reader_seed, device, check, progress=print):
    if store.binding.get("data") != data.binding:
        raise ValueError("cell must bind its authenticated data references")
    kernel = TrainingKernel(arm, checkpoint_seed, reader_seed, binding=store.binding,
                            scene_ids=data.eligible["train"], device=device)
    previous = store.latest(kernel)
    if previous is None:
        previous = store.save_state(kernel, None)
    try:
        while True:
            if kernel.epoch and kernel.next_batch == 0 and kernel.epoch % 5 == 0:
                calibration_outputs(kernel, data, store, previous, check)
            if kernel.epoch == 50:
                break
            check()
            ids = kernel.expected_scene_ids()
            rows = [data.rows["train"][i] for i in ids]
            batch = collate([r["inputs"] for r in rows])
            target = collate_targets([r["targets"] for r in rows], batch["candidate_mask"])
            kernel.step(batch, target, ids)
            if kernel.next_batch == 0:
                previous = store.save_state(kernel, previous)
                progress({"arm": arm, "checkpoint_seed": checkpoint_seed, "reader_seed": reader_seed,
                          "epoch": kernel.epoch, "steps": kernel.steps, "loss": kernel.history[-1]["loss"]})
    except BaseException:
        if kernel.safe_boundary:
            store.save_state(kernel, previous)
        raise
    # Completion means all 50 last_epoch checkpoints and all ten calibrations,
    # not merely that the latest snapshot says epoch 50.
    snapshots = []
    batches = len(kernel.schedule[0])
    for epoch in range(51):
        ref = store.reference(store.path(f"snapshots/step_{epoch*batches:06d}.json"))
        state = store.load_state(ref)
        if state["epoch"] != epoch or state["next_batch"] != 0:
            raise ValueError("a required epoch checkpoint is missing")
        snapshots.append(ref)
    calibrations = []
    for epoch in range(5, 51, 5):
        ref = store.reference(store.path(f"calibration/epoch_{epoch:02d}/index.json"))
        read_calibration(data, store, ref, snapshots[epoch], epoch)
        calibrations.append(ref)
    value = {"schema": "generative-evidence-cell-complete-v1", "binding": store.binding,
             "status": "TRAINED_NOT_SELECTED", "arm": arm, "checkpoint_seed": checkpoint_seed,
             "reader_seed": reader_seed, "epochs": 50, "steps": kernel.steps,
             "train_eligible": data.eligible["train"], "calibration_eligible": data.eligible["calibration"],
             "snapshots": snapshots, "calibrations": calibrations, "last_epoch": snapshots[-1]}
    path = store.path("complete.json")
    if path.exists():
        if store.json(store.reference(path)) != value:
            raise ValueError("cannot replace a completed cell")
    else:
        storage.write_json(path, value)
    return store.reference(path)
