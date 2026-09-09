"""Mechanical selection contracts; no real cells, filesystem, models or test access."""
from copy import deepcopy
import hashlib
import inspect
from pathlib import Path

import numpy as np
import pytest

from experiments.atencion_armonica.test_generative_evidence import fixture
from src.atencion_armonica import generative_evidence as ge
from src.atencion_armonica import generative_evidence_selection as selection
from src.atencion_armonica.generative_evidence_evaluation import EPOCHS
from src.atencion_armonica.generative_evidence_supervision import candidate_supervision
from src.atencion_armonica.partial_compatibility_cache import encoded


def root_ref(path):
    return {"path": str(path), "sha256": hashlib.sha256(str(path).encode()).hexdigest()}


def roster():
    return [{"arm": arm, "checkpoint_seed": cp, "reader_seed": seed,
             "cell_id": f"{arm}-cp{cp}-seed{seed}"}
            for arm in ge.ARMS for cp in ge.CHECKPOINTS for seed in ge.READER_SEEDS]


def verified():
    cells = []
    for cell in roster():
        path = (f"data/atencion_armonica/generative_evidence_reader_v1/training/{cell['arm']}"
                f"/cp_{cell['checkpoint_seed']}/seed_{cell['reader_seed']}/complete.json")
        cells.append({"cell": cell, "complete": root_ref(path)})
    return {"manifest": root_ref(".agent-work/phideus-generative-evidence-20260909/training-control/manifest.json"),
            "index": root_ref("data/atencion_armonica/generative_evidence_reader_v1/training/index.json"),
            "cells": cells, "delivery": {}}


def accounting():
    ids = [c["cell_id"] for c in roster()]
    return 27., root_ref("last.exit.json"), {i: 1. for i in ids}, {i: root_ref(i+".exit.json") for i in ids}


class MemoryArtifacts:
    def __init__(self, binding=None):
        self.binding, self.values = binding or {}, {}

    def _ref(self, path, payload):
        raw = encoded(payload) if not isinstance(payload, dict) or not payload or all(isinstance(k, str) for k in payload) else repr(payload).encode()
        return {"path": path, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}

    def write_json(self, path, value):
        if path in self.values and encoded(self.values[path]) != encoded(value):
            raise ValueError("changed JSON")
        self.values[path] = deepcopy(value)
        return self._ref(path, value)

    def write_arrays(self, path, value):
        if path in self.values and not all(
                np.array_equal(self.values[path][k], value[k], equal_nan=True) for k in value):
            raise ValueError("changed arrays")
        self.values[path] = {k: v.copy() for k, v in value.items()}
        raw = b"".join(v.tobytes() for v in value.values())
        return {"path": path, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}

    def json(self, ref):
        return deepcopy(self.values[ref["path"]])


def test_training_authority_requires_exact_roster_refs_and_terminal_accounting():
    value = verified()
    manifest = {"roster": roster(), "limit_seconds": 43200, "cell_limit_seconds": 1800}
    cells, total = selection._validate_authority(value, manifest, accounting())
    assert cells == value["cells"] and total == 27.
    bad = deepcopy(value)
    bad["cells"].pop()
    with pytest.raises(ValueError):
        selection._validate_authority(bad, manifest, accounting())
    bad = deepcopy(value)
    bad["cells"][0]["complete"] = bad["cells"][1]["complete"]
    with pytest.raises(ValueError):
        selection._validate_authority(bad, manifest, accounting())
    total, previous, per_cell, parents = accounting()
    parents[roster()[0]["cell_id"]] = None
    with pytest.raises(ValueError):
        selection._validate_authority(value, manifest, (total, previous, per_cell, parents))


def test_initial_model_identity_is_grouped_only_by_reader_seed():
    rows = [{"cell": cell, "snapshot": root_ref(cell["cell_id"]+"/initial"),
             "model_state_digest": hashlib.sha256(str(cell["reader_seed"]).encode()).hexdigest()}
            for cell in roster()]
    result = selection._initial_groups(rows)
    assert set(result) == {str(s) for s in ge.READER_SEEDS}
    assert all(len(result[str(s)]["cells"]) == 9 for s in ge.READER_SEEDS)
    bad = deepcopy(rows)
    bad[0]["model_state_digest"] = "0"*64
    with pytest.raises(ValueError, match="byte-identical"):
        selection._initial_groups(bad)


def test_epoch_readout_preserves_empty_scenes_metrics_decisions_and_sources(monkeypatch):
    ps = [fixture()[4][0]]
    metrics = [candidate_supervision(ps, np.repeat(np.arange(2), 4))["metrics"][0]]
    partitions, candidate_metrics = [[] for _ in range(512)], [[] for _ in range(512)]
    partitions[17], candidate_metrics[17] = ps, metrics
    identities = [f"{i:064x}" for i in range(512)]
    offsets = np.zeros(513, np.int64)
    offsets[18:] = 1
    monkeypatch.setattr(selection, "read_calibration",
        lambda data, store, ref, state, epoch: (np.array([[.2, .3]], np.float32), offsets))
    monkeypatch.setattr(selection, "_root_reference", root_ref)

    class Store:
        def json(self, ref): return {"predictions": {"path": "prediction.npz"}}
        def path(self, path): return Path(path)

    cell = roster()[0]
    complete = {"calibrations": [{"path": f"cal-{e}.json"} for e in EPOCHS],
                "snapshots": [{"path": f"state-{e}.json"} for e in range(51)]}
    archive = MemoryArtifacts({"training": "fixture"})
    ref, selector, value = selection._write_epoch(archive, cell, root_ref("complete.json"), complete,
        Store(), object(), {"partitions": partitions, "metrics": candidate_metrics, "identities": identities}, 5, lambda: None)
    arrays = archive.values[value["metrics"]["path"]]
    assert selector["ari"][17] == metrics[0]["ari"] and selector["ari"][0] is None
    assert arrays["eligible"].sum() == 1 and np.isnan(arrays["metrics"][0]).all()
    assert archive.values[value["decisions"]["path"]]["decisions"][17]["candidate_index"] == 0
    assert value["state"]["path"] == "state-5.json" and value["predictions"]["path"] == "prediction.npz"
    assert ref["path"].endswith("epoch_05/index.json")


def test_run_selection_consumes_public_authority_and_never_authorizes_tests(monkeypatch):
    signature = inspect.signature(selection.run_selection)
    assert "supervisor" not in signature.parameters
    assert signature.parameters["check"].default is inspect.Parameter.empty
    assert signature.parameters["stage_manifest"].default is inspect.Parameter.empty
    authority = verified()
    delivery_refs = {k: root_ref(k+".json") for k in ("manifest", "worker", "delivered", "load_profile")}
    delivery = {**delivery_refs, "store": object(), "corpus": object()}
    authority["delivery"] = delivery
    manifest = {"roster": roster(), "limit_seconds": 43200, "cell_limit_seconds": 1800}

    class Supervisor:
        TRAINING_MANIFEST = selection.ROOT/authority["manifest"]["path"]
        def __init__(self): self.calls = []
        def verified_training(self): self.calls.append("verified_training"); return authority
        def read_training_manifest(self, ref): self.calls.append("read_training_manifest"); return manifest, delivery
        def verified_delivery(self): self.calls.append("verified_delivery"); return delivery
        def training_accumulated(self, ref): self.calls.append("training_accumulated"); return accounting()

    archive = MemoryArtifacts()
    identities = [f"{i:064x}" for i in range(512)]
    monkeypatch.setattr(selection, "SelectionArtifacts", lambda root, binding: setattr(archive, "binding", binding) or archive)
    monkeypatch.setattr(selection, "_prepared_calibration", lambda delivery, cp, check:
        {"identities": identities, "signatures": [[] for _ in range(512)], "metrics": [[] for _ in range(512)]})
    monkeypatch.setattr(selection, "_cell_data", lambda delivery, prepared, cell, check: object())
    monkeypatch.setattr(selection, "_root_reference", root_ref)

    class Store:
        def path(self, path): return Path(path)
        def json(self, ref): return {"predictions": {"path": "prediction.npz"}}

    def opened(entry, data):
        complete = {"snapshots": [{"path": f"{entry['cell']['cell_id']}/state-{e}.json"} for e in range(51)],
                    "calibrations": [{"path": f"{entry['cell']['cell_id']}/cal-{e}.json"} for e in EPOCHS]}
        return complete, Store()
    monkeypatch.setattr(selection, "_open_cell", opened)
    monkeypatch.setattr(selection, "_initial_model", lambda cell, complete, store:
        {"cell": cell, "snapshot": root_ref(cell["cell_id"]+"/initial"),
         "model_state_digest": hashlib.sha256(str(cell["reader_seed"]).encode()).hexdigest()})

    def write_epoch(archive, cell, complete_ref, complete, store, data, prepared, epoch, check):
        score = epoch/50
        ref = {"path": f"{cell['cell_id']}/{epoch}.json", "sha256": "a"*64, "bytes": 1}
        row = {"arm": cell["arm"], "checkpoint_seed": cell["checkpoint_seed"],
               "reader_seed": cell["reader_seed"], "epoch": epoch, "split": "calibration",
               "split_seed": 2026090881, "identities": identities, "ari": [score]+[None]*511}
        return ref, row, {}
    monkeypatch.setattr(selection, "_write_epoch", write_epoch)

    supervisor = Supervisor()
    stage_manifest = root_ref("selection-control/manifest.json")
    result = selection._run_selection(supervisor, root="unused", check=lambda: None,
                                      stage_manifest=stage_manifest)
    final = archive.json(result)
    assert supervisor.calls == ["verified_training", "read_training_manifest", "verified_delivery", "training_accumulated"]
    assert final["status"] == selection.STATUS and final["test_access"] is False
    assert final["calibration_record_count"] == 270 and len(final["calibration_records"]) == 270
    assert {a: final["selected_states"][a]["epoch"] for a in ge.ARMS} == {a: 50 for a in ge.ARMS}
    assert all(len(final["selected_states"][a]["cells"]) == 9 for a in ge.ARMS)
    assert final["binding"]["selection_manifest"] == stage_manifest
    assert selection._run_selection(supervisor, root="unused", check=lambda: None,
                                    stage_manifest=stage_manifest) == result
    assert archive.json(result) == final
