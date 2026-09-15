"""Authenticate the fixed 36 selected heads and their frozen dependencies.

Read-only; NumPy archives only, no pickle deserialization, models, CUDA or draws.
The returned ledger is provenance, not a campaign or resource authorization.
"""
from __future__ import annotations

import hashlib
from io import BytesIO
import json
from pathlib import Path

import numpy as np

from .measurement_contract import reference

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = "data/atencion_armonica/geometric_decision_energy_v1/archive"
REFERENCES = {
    "archive": {"path": ARCHIVE+"/heads.json", "bytes": 97200,
                "sha256": "b94acc78fe4f08d28faf6b97600a27895ec46ded018aa437b5e768952ba16774"},
    "normalizers": {"path": "data/atencion_armonica/generative_evidence_reader_v1/normalizers.json", "bytes": 298447,
                    "sha256": "46acc508952f72187c7f26dfe325c816f22bd0df1e5a8c88cb1e81e3b8f50e93"},
    "scale": {"path": "data/atencion_armonica/geometric_decision_energy_v1/open/scale.json", "bytes": 275041,
              "sha256": "25da9bd505efb0b594e88755f0e20c0018919646a4149e8d1b6faae19e26bd8e"},
    "common": {"path": "data/atencion_armonica/learned_partition_reader_v1/authorization/train_calibration_06.json",
               "bytes": 1069504, "sha256": "7740d564c748389f47ce7a818675ed5331e81ae4e0c7e50e6576a2e8df8568b5"},
}
SHAPES = {"group1.weight": (32, 9), "group1.bias": (32,),
          "group2.weight": (16, 32), "group2.bias": (16,),
          "partition1.weight": (32, 41), "partition1.bias": (32,),
          "partition2.weight": (2, 32), "partition2.bias": (2,)}
EPOCHS = {"injection_decision": 30, "geometric_decision": 45,
          "decoupled_decision": 40, "local_decision": 45}
CHECKPOINTS = (2026090721, 2026090722, 2026090723)
READERS = (2026091491, 2026091492, 2026091493)


def read_reference(ref):
    ref = reference(ref)
    path = ROOT/ref["path"]
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("reused artifact traverses a symlink")
    raw = path.read_bytes()
    if len(raw) != ref["bytes"] or hashlib.sha256(raw).hexdigest() != ref["sha256"]:
        raise ValueError(f"reused artifact changed: {ref['path']}")
    return raw


def head_arrays(raw):
    with np.load(BytesIO(raw), allow_pickle=False) as archive:
        if len(archive.files) != len(set(archive.files)):
            raise ValueError("duplicate head tensor names")
        arrays = {k: archive[k] for k in archive.files}
    if set(arrays) != set(SHAPES) or any(
            a.dtype != np.float32 or a.shape != SHAPES[k] or not np.isfinite(a).all()
            for k, a in arrays.items()):
        raise ValueError("head tensor schema or values differ")
    return arrays


def load_reuse():
    ledger = []
    def read(ref):
        raw = read_reference(ref)
        ledger.append(dict(ref))
        return raw
    roots = {key: json.loads(read(ref)) for key, ref in REFERENCES.items()}
    archive, common = roots["archive"], roots["common"]["common"]
    if archive["schema"] != "geometric-decision-head-archive-v1" or archive["count"] != 144:
        raise ValueError("unexpected historical head archive")
    if [c["seed"] for c in common["checkpoints"]] != list(CHECKPOINTS):
        raise ValueError("backbone roster differs")
    for checkpoint in common["checkpoints"]:
        read({**checkpoint["checkpoint"], "bytes": 5854371})  # Hash only; never unpickle.
    head_refs = {r["path"]: r for r in archive["records"]}
    if len(head_refs) != 144:
        raise ValueError("head index has duplicate or missing records")
    heads = []
    for arm, epoch in EPOCHS.items():
        if archive["selected_epochs"][arm] != epoch:
            raise ValueError("frozen common epoch differs")
        for cp in CHECKPOINTS:
            for seed in READERS:
                local_path = f"heads/cp_{cp}/{arm}/seed_{seed}/selected.json"
                ref = {**head_refs[local_path], "path": ARCHIVE+"/"+local_path}
                row = json.loads(read(ref))
                if ((row["checkpoint_seed"], row["arm"], row["reader_seed"], row["stage"], row["epoch"])
                        != (cp, arm, seed, "selected", epoch)):
                    raise ValueError("selected reader identity differs")
                if row["arrays"]["path"] != local_path.removesuffix(".json")+".npz":
                    raise ValueError("reader arrays point outside the selected state")
                array_ref = {**row["arrays"], "path": ARCHIVE+"/"+row["arrays"]["path"]}
                arrays = head_arrays(read(array_ref))
                heads.append({"arm": arm, "checkpoint_seed": cp, "reader_seed": seed,
                              "epoch": epoch, "record": ref, "array_reference": array_ref, "arrays": arrays})
    if roots["scale"]["scale"] != 7.5314913344363035:
        raise ValueError("geometric TRAIN scale differs")
    return {"heads": heads, "checkpoints": common["checkpoints"], "runtime": common["runtime"],
            "normalizers": roots["normalizers"]["normalizers"], "scale": roots["scale"]["scale"],
            "references": ledger}
