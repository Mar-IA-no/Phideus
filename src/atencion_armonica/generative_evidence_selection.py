"""Recoverable OPEN calibration selection over 27 completed training cells.

The training supervisor remains the authority for manifests, typed terminal
attempts, resource accounting and complete cell payloads.  This consumer only
reopens the preserved calibration outputs, applies the audited pure estimands
and writes a traceable selection that still grants no fresh-test access.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from . import generative_evidence as ge
from . import generative_evidence_evaluation as evaluation
from . import generative_evidence_storage as storage
from .generative_evidence_cell import CellArtifacts, read_calibration, state_digest
from .generative_evidence_reuse import ROOT, VerifiedBytes
from .partial_compatibility_cache import encoded
from .structured_source_artifacts import safe_member


SELECTION_ROOT = ROOT/"data/atencion_armonica/generative_evidence_reader_v1/selection"
TEMP = ROOT/".agent-work/phideus-generative-evidence-20260909"
STATUS = "CALIBRATION_SELECTED_NOT_TEST_AUTHORIZED"


def _clone(value):
    return json.loads(encoded(value))


def _root_reference(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": hashlib.sha256(raw).hexdigest()}


def _valid_root_ref(ref):
    return (isinstance(ref, dict) and set(ref) == {"path", "sha256"}
            and isinstance(ref["path"], str) and bool(ref["path"])
            and isinstance(ref["sha256"], str) and len(ref["sha256"]) == 64
            and all(c in "0123456789abcdef" for c in ref["sha256"]))


def _same_arrays(first, second):
    return (set(first) == set(second)
            and all(first[k].dtype == second[k].dtype and first[k].shape == second[k].shape
                    and np.array_equal(first[k], second[k], equal_nan=True) for k in first))


class SelectionArtifacts:
    """Immutable selection receipts under the canonical or an owned test root."""

    def __init__(self, root, *, binding):
        if not isinstance(binding, dict) or not binding:
            raise ValueError("selection requires a nonempty provenance binding")
        self.binding = _clone(binding)
        self.root = Path(root).resolve()
        if self.root != SELECTION_ROOT and not (self.root.is_relative_to(TEMP) and self.root != TEMP):
            raise ValueError("selection artifacts require the canonical or an owned temporary root")
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root/"binding.json"
        if path.exists():
            raw = path.read_bytes()
            if path.is_symlink() or encoded(json.loads(raw)) != raw or json.loads(raw) != self.binding:
                raise ValueError("selection artifact binding differs")
        else:
            if any(self.root.iterdir()):
                raise ValueError("cannot adopt an existing unbound selection directory")
            storage.write_json(path, self.binding)

    def path(self, relative):
        path = safe_member(self.root, relative)
        if any(p.is_symlink() for p in [path, *path.parents] if p.is_relative_to(self.root)):
            raise ValueError("selection artifacts cannot traverse symlinks")
        return path

    def reference(self, path):
        path = self.path(Path(path).relative_to(self.root).as_posix())
        raw = path.read_bytes()
        return {"path": path.relative_to(self.root).as_posix(),
                "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}

    def json(self, ref):
        if set(ref) != {"path", "sha256", "bytes"}:
            raise ValueError("invalid selection JSON reference")
        raw = self.path(ref["path"]).read_bytes()
        if len(raw) != ref["bytes"] or hashlib.sha256(raw).hexdigest() != ref["sha256"]:
            raise ValueError("selection JSON bytes differ")
        value = json.loads(raw)
        if encoded(value) != raw:
            raise ValueError("selection JSON must be canonical")
        return value

    def arrays(self, ref):
        if set(ref) != {"path", "sha256", "bytes"}:
            raise ValueError("invalid selection array reference")
        return storage.read_arrays(self.path(ref["path"]), {k: ref[k] for k in ("sha256", "bytes")})

    def write_json(self, relative, value):
        path = self.path(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            ref = self.reference(path)
            if encoded(self.json(ref)) != encoded(value):
                raise ValueError("cannot replace a completed selection JSON boundary")
            return ref
        storage.write_json(path, value)
        return self.reference(path)

    def write_arrays(self, relative, arrays):
        path = self.path(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            ref = self.reference(path)
            if not _same_arrays(self.arrays(ref), arrays):
                raise ValueError("cannot replace a completed selection array boundary")
            return ref
        storage.write_arrays(path, arrays)
        return self.reference(path)


def _validate_authority(verified, manifest, accumulated):
    """Check the public R710 result without reimplementing its private ledger."""
    if (set(verified) != {"manifest", "index", "cells", "delivery"}
            or not _valid_root_ref(verified["manifest"]) or not _valid_root_ref(verified["index"])
            or not isinstance(verified["cells"], list) or len(verified["cells"]) != 27):
        raise ValueError("selection requires the public complete-training authority")
    expected = manifest["roster"]
    cells = verified["cells"]
    paths = [f"data/atencion_armonica/generative_evidence_reader_v1/training/{c['arm']}"
             f"/cp_{c['checkpoint_seed']}/seed_{c['reader_seed']}/complete.json" for c in expected]
    if ([row.get("cell") for row in cells] != expected
            or any(set(row) != {"cell", "complete"} for row in cells)
            or any(not _valid_root_ref(row["complete"]) for row in cells)
            or [row["complete"]["path"] for row in cells] != paths
            or len({encoded(row["complete"]) for row in cells}) != 27):
        raise ValueError("verified training cell roster or complete references differ")
    total, previous, per_cell, cell_parents = accumulated
    ids = [c["cell_id"] for c in expected]
    if (not isinstance(total, (int, float)) or not np.isfinite(total) or total < 0
            or total >= manifest["limit_seconds"] or previous is None
            or set(per_cell) != set(ids) or set(cell_parents) != set(ids)
            or any(not np.isfinite(per_cell[i]) or per_cell[i] < 0
                   or per_cell[i] >= manifest["cell_limit_seconds"] for i in ids)
            or any(cell_parents[i] is None for i in ids)):
        raise ValueError("verified training accounting or terminal parents differ")
    return cells, float(total)


def _prepared_calibration(delivery, checkpoint_seed, check):
    check()
    value = delivery["store"].load_shard("calibration", checkpoint_seed, 0)
    if (value["scene_ids"] != list(range(512)) or len(value["identities"]) != 512
            or len(value["rows"]) != 512 or len(value["metrics"]) != 512):
        raise ValueError("selection requires the complete calibration shard")
    partitions = [row["partitions"] for row in value["rows"]]
    return {"identities": value["identities"], "partitions": partitions,
            "signatures": [[ge.law.signature(p) for p in ps] for ps in partitions],
            "metrics": value["metrics"]}


def _cell_data(delivery, prepared, cell, check):
    check()
    data = delivery["corpus"].load_cell(delivery["delivered"], arm=cell["arm"],
                                        checkpoint_seed=cell["checkpoint_seed"], check=check)
    identities = [row["identity"] for row in data.rows["calibration"]]
    signatures = [[ge.law.signature(p) for p in row["partitions"]]
                  for row in data.rows["calibration"]]
    if identities != prepared["identities"] or signatures != prepared["signatures"]:
        raise ValueError("cell calibration inputs differ from the prepared candidate roster")
    return data


def _open_cell(entry, data):
    value = VerifiedBytes(ROOT).json(entry["complete"])
    cell = entry["cell"]
    keys = {"schema", "binding", "status", "arm", "checkpoint_seed", "reader_seed", "epochs", "steps",
            "train_eligible", "calibration_eligible", "snapshots", "calibrations", "last_epoch"}
    if (set(value) != keys or value["schema"] != "generative-evidence-cell-complete-v1"
            or value["status"] != "TRAINED_NOT_SELECTED"
            or any(value[k] != cell[k] for k in ("arm", "checkpoint_seed", "reader_seed"))
            or value["binding"].get("data") != data.binding
            or len(value["snapshots"]) != 51 or len(value["calibrations"]) != 10
            or value["last_epoch"] != value["snapshots"][-1]):
        raise ValueError("verified cell completion changed before selection")
    root = ROOT/entry["complete"]["path"]
    store = CellArtifacts(root.parent, binding=value["binding"])
    local = store.reference(root)
    if local["sha256"] != entry["complete"]["sha256"] or _root_reference(root) != entry["complete"]:
        raise ValueError("cell completion root reference differs")
    return value, store


def _initial_model(cell, complete, store):
    ref = complete["snapshots"][0]
    state = store.load_state(ref)
    if (state["epoch"] != 0 or state["next_batch"] != 0 or state["steps"] != 0
            or state["arm"] != cell["arm"] or state["checkpoint_seed"] != cell["checkpoint_seed"]
            or state["reader_seed"] != cell["reader_seed"]):
        raise ValueError("cell initial snapshot identity differs")
    return {"cell": cell, "snapshot": _root_reference(store.path(ref["path"])),
            "model_state_digest": state_digest(state["model"])}


def _write_epoch(archive, cell, complete_ref, complete, store, data, prepared, epoch, check):
    check()
    position = evaluation.EPOCHS.index(epoch)
    calibration_ref = complete["calibrations"][position]
    components, offsets = read_calibration(data, store, calibration_ref, complete["snapshots"][epoch], epoch)
    calibration = store.json(calibration_ref)
    chosen = evaluation.chosen_metrics(prepared["partitions"], prepared["metrics"], components, offsets)
    ari = [float(v) if ok else None for v, ok in zip(chosen["metrics"][:, 0], chosen["eligible"])]
    prefix = f"calibration/{cell['cell_id']}/epoch_{epoch:02d}"
    check()
    metrics_ref = archive.write_arrays(prefix+"/metrics.npz",
        {"metrics": chosen["metrics"], "eligible": chosen["eligible"]})
    check()
    decisions_ref = archive.write_json(prefix+"/decisions.json",
        {"identities": prepared["identities"], "decisions": chosen["decisions"]})
    state_ref = complete["snapshots"][epoch]
    value = {"schema": "generative-evidence-calibration-readout-v1", "binding": archive.binding,
        "cell": cell, "epoch": epoch, "split": "calibration",
        "split_seed": evaluation.SPLITS["calibration"][1],
        "identities": prepared["identities"], "eligible_scene_ids": np.flatnonzero(chosen["eligible"]).tolist(),
        "excluded_scene_ids": np.flatnonzero(~chosen["eligible"]).tolist(), "ari": ari,
        "cell_complete": complete_ref, "state": _root_reference(store.path(state_ref["path"])),
        "calibration": _root_reference(store.path(calibration_ref["path"])),
        "predictions": _root_reference(store.path(calibration["predictions"]["path"])),
        "metrics": metrics_ref, "decisions": decisions_ref}
    check()
    record_ref = archive.write_json(prefix+"/index.json", value)
    selector = {"arm": cell["arm"], "checkpoint_seed": cell["checkpoint_seed"],
        "reader_seed": cell["reader_seed"], "epoch": epoch, "split": "calibration",
        "split_seed": evaluation.SPLITS["calibration"][1],
        "identities": prepared["identities"], "ari": ari}
    return record_ref, selector, value


def _initial_groups(entries):
    grouped = {}
    for entry in entries:
        grouped.setdefault(str(entry["cell"]["reader_seed"]), []).append(entry)
    if set(grouped) != {str(s) for s in ge.READER_SEEDS}:
        raise ValueError("initial-state reader-seed roster differs")
    result = {}
    for seed in ge.READER_SEEDS:
        rows = grouped[str(seed)]
        expected = [(a, cp) for a in ge.ARMS for cp in ge.CHECKPOINTS]
        if ([(r["cell"]["arm"], r["cell"]["checkpoint_seed"]) for r in rows] != expected
                or len({r["model_state_digest"] for r in rows}) != 1):
            raise ValueError("initial model is not byte-identical across arm/checkpoint cells")
        result[str(seed)] = {"model_state_digest": rows[0]["model_state_digest"], "cells": rows}
    return result


def _run_selection(supervisor, *, root, check, stage_manifest):
    if not callable(check):
        raise TypeError("selection requires a resource check callback")
    if not _valid_root_ref(stage_manifest):
        raise ValueError("selection requires its authenticated stage manifest")
    check()
    verified = supervisor.verified_training()
    manifest, delivery = supervisor.read_training_manifest(verified["manifest"])
    current_delivery = supervisor.verified_delivery()
    ref_keys = ("manifest", "worker", "delivered", "load_profile")
    if (verified["manifest"].get("path") != supervisor.TRAINING_MANIFEST.relative_to(ROOT).as_posix()
            or any(delivery[k] != verified["delivery"][k] or delivery[k] != current_delivery[k] for k in ref_keys)):
        raise ValueError("training/delivery authority changed before calibration selection")
    accumulated = supervisor.training_accumulated(verified["manifest"])
    cells, total = _validate_authority(verified, manifest, accumulated)
    binding = {"selection_manifest": stage_manifest, "training_manifest": verified["manifest"],
               "training_complete": verified["index"]}
    archive = SelectionArtifacts(root, binding=binding)

    prepared_by_cp, common = {}, None
    for cp in ge.CHECKPOINTS:
        prepared_by_cp[cp] = _prepared_calibration(delivery, cp, check)
        identity = {k: prepared_by_cp[cp][k] for k in ("identities", "signatures", "metrics")}
        if common is not None and encoded(identity) != encoded(common):
            raise ValueError("calibration candidates or metrics differ across checkpoints")
        common = identity
        check()

    current_key, data = None, None
    opened, selector_records, record_refs, initials = {}, [], [], []
    for entry in cells:
        cell, complete_ref = entry["cell"], entry["complete"]
        key = cell["arm"], cell["checkpoint_seed"]
        if key != current_key:
            data = _cell_data(delivery, prepared_by_cp[cell["checkpoint_seed"]], cell, check)
            current_key = key
        complete, cell_store = _open_cell(entry, data)
        opened[cell["cell_id"]] = (complete_ref, complete, cell_store)
        initials.append(_initial_model(cell, complete, cell_store))
        for epoch in evaluation.EPOCHS:
            record_ref, selector, _ = _write_epoch(archive, cell, complete_ref, complete, cell_store, data,
                                                   prepared_by_cp[cell["checkpoint_seed"]], epoch, check)
            record_refs.append({"cell": cell, "epoch": epoch, "record": record_ref})
            selector_records.append(selector)

    selected = evaluation.select_epochs(selector_records)
    selected_states = {}
    for arm in ge.ARMS:
        epoch = selected["selected"][arm]["epoch"]
        rows = []
        for entry in cells:
            cell = entry["cell"]
            if cell["arm"] != arm:
                continue
            check()
            complete_ref, complete, cell_store = opened[cell["cell_id"]]
            record = next(r["record"] for r in record_refs if r["cell"] == cell and r["epoch"] == epoch)
            calibration_ref = complete["calibrations"][evaluation.EPOCHS.index(epoch)]
            calibration = cell_store.json(calibration_ref)
            rows.append({"cell": cell, "cell_complete": complete_ref, "epoch": epoch,
                "state": _root_reference(cell_store.path(complete["snapshots"][epoch]["path"])),
                "calibration": _root_reference(cell_store.path(calibration_ref["path"])),
                "predictions": _root_reference(cell_store.path(calibration["predictions"]["path"])),
                "calibration_readout": record})
        if len(rows) != 9:
            raise ValueError("selected arm lacks its nine cell states")
        selected_states[arm] = {"epoch": epoch, "cells": rows}

    value = {"schema": "generative-evidence-calibration-selection-v1", "binding": archive.binding,
        "status": STATUS, "training_accumulated_seconds": total, "cell_count": 27,
        "calibration_record_count": 270, "calibration_records": record_refs,
        "initial_models": _initial_groups(initials), "selection": selected,
        "selected_states": selected_states, "test_access": False}
    check()
    result = archive.write_json("index.json", value)
    if archive.json(result) != value:
        raise ValueError("selection completion changed after publication")
    return result


def run_selection(*, check, stage_manifest, root=SELECTION_ROOT):
    """Select epochs through the fixed R710 authority; never opens a fresh test."""
    from experiments.atencion_armonica import run_generative_training as supervisor
    return _run_selection(supervisor, root=root, check=check, stage_manifest=stage_manifest)


__all__ = ["SelectionArtifacts", "run_selection", "STATUS", "SELECTION_ROOT"]
