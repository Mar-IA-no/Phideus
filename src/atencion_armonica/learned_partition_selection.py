"""One complete 36-cell selection freeze before any prospective test access."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from . import learned_partition_gate as gate
from . import learned_partition_provenance as p
from . import learned_partition_runner as runner
from .learned_partition_budget import accounting, terminal_receipt, CELL_SECONDS, CAMPAIGN_SECONDS
from .learned_partition_campaign import calibration_record, resume_state, snapshot_chain
from .learned_partition_core import ARMS, READER_SEEDS
from .learned_partition_data import _bundle, scene_ids
from .learned_partition_metrics import EPOCHS, SEEDS, select_epochs
from .structured_source_artifacts import safe_member, verify_bundle, write_json
from .learned_partition_validation import boundary, memoized, fresh_pass

DATA_FIELDS = ("data_authorization", "train", "calibration", "normalizers", "normalized_train", "normalized_calibration", "reuse_audit")
FREEZE_FIELDS = {"status", "common", *DATA_FIELDS, "train_data", "calibration_data", "cells", "selection"}


def selection_context(record, common):
    authorization = record["data_authorization"]
    auth, train_m, train_shards = runner.training_corpus(record["train"], "train", authorization=authorization)
    from .learned_partition_reuse import PREPARED_FIELDS, verify_completion
    verify_completion(record["reuse_audit"], authorization,
        {k: authorization if k == "authorization" else record[k] for k in PREPARED_FIELDS})
    _, cal_m, cal_shards = runner.training_corpus(record["calibration"], "calibration", authorization=authorization)
    if (auth["common"] != common or train_m["binding"]["data"] != record["train_data"]
            or cal_m["binding"]["data"] != record["calibration_data"]):
        raise ValueError("selection data chain or common identity differs")
    _, cal_data = _bundle(record["calibration_data"], "learned_observation_split", common)
    if cal_data["binding"]["previous"] != {"train": record["train_data"]}:
        raise ValueError("selection calibration belongs to another training chain")
    runner.read_normalizers(record["normalizers"], common, authorization=authorization, train=record["train"])
    for split, shards in (("train", train_shards), ("calibration", cal_shards)):
        refs = record[f"normalized_{split}"]
        if not isinstance(refs, list) or len(refs) != len(shards):
            raise ValueError("selection normalized corpus roster differs")
        for index, (entry, ref) in enumerate(zip(shards, refs)):
            cache = SimpleNamespace(split=split, shard=index, scene_ids=scene_ids(split, index))
            runner.normalized_shard(ref, cache, common, authorization=authorization, data=entry["data"],
                logits=entry["logits"], scored=entry["scored"], normalizers=record["normalizers"], train=record["train"])
    cal = cal_shards[0]
    scored_root, target_root = (p.verify_reference(cal[k]).parent for k in ("scored", "targets"))
    data = {}
    for seed in SEEDS:
        candidates, ari = [], []
        for i in range(512):
            row = runner.load_rows(scored_root/f"seed_{seed}/{i:05d}_rows.npz")
            candidates.append(row.candidates)
            metrics = json.loads((target_root/f"seed_{seed}/{i:05d}_metrics.json").read_bytes())
            ari.append(np.asarray([m["ari"] for m in metrics["candidate_metrics"]], np.float64))
        data[seed] = {"candidates": candidates, "ari": ari}
    base = {"common": common, "authorization": authorization,
        **{k: record[k] for k in DATA_FIELDS if k != "data_authorization"}, "device": auth["training_device"], "count": 4096}
    return base, data


def verify_cell(result, supervisor, binding, calibration_data):
    path = p.verify_reference(result)
    m = verify_bundle(path.parent, result["sha256"], role="learned_training_cell")
    b = m["binding"]
    if set(b) != {"cell", "request", "permit", "resume"} or b["cell"] != binding:
        raise ValueError("completed cell has different datasets, architecture or runtime")
    terminal = terminal_receipt(supervisor, request_ref=b["request"])
    if terminal["status"] != "COMPLETE" or terminal["result"] != result or terminal["budget"] != b["permit"]:
        raise ValueError("cell completion is not backed by its supervisor and budget")
    request = p.read_reference(b["request"])
    if (safe_member(p.ROOT, request["output"]) != path.parent or request["arguments"]["resume"] != b["resume"]
            or request["arguments"].get("reuse_audit") != binding["reuse_audit"]):
        raise ValueError("cell request differs from the completed attempt")
    ancestry = json.loads((path.parent/"ancestry.json").read_bytes())
    inherited = (None, [], []) if b["resume"] is None else resume_state(b["resume"], binding, calibration_data)
    if ancestry != {"resume": b["resume"], "snapshots": inherited[1], "calibrations": inherited[2]}:
        raise ValueError("completed attempt changed its verified inherited prefix")
    chain = json.loads((path.parent/"chain.json").read_bytes())
    if set(chain) != {"initial", "last_epoch", "snapshots", "calibrations"}:
        raise ValueError("completed training chain schema differs")
    refs = chain["snapshots"]
    if (not refs or chain["initial"] != refs[0] or chain["last_epoch"] != refs[-1]
            or refs[:len(inherited[1])] != inherited[1]
            or chain["calibrations"][:len(inherited[2])] != inherited[2]):
        raise ValueError("completed chain omits or changes its inherited states")
    state, positions = snapshot_chain(refs, binding, complete=True)
    ready = json.loads((path.parent/"training_ready.json").read_bytes())
    beginning = inherited[1][-1] if inherited[1] else refs[0]
    beginning_steps = p.read_reference(beginning)["position"]["steps"]
    if ready != {"snapshot": beginning, "steps": beginning_steps, "binding": binding}:
        raise ValueError("training initialization receipt differs from its actual initial/resumed snapshot")
    if json.loads((path.parent/"history.json").read_bytes()) != state["history"]:
        raise ValueError("training diagnostics differ from last_epoch")
    if len(chain["calibrations"]) != 10:
        raise ValueError("cell omitted calibration epochs")
    records, expected_files = [], {"ancestry.json", "history.json", "chain.json", "training_ready.json"}
    for ref in refs[len(inherited[1]):]:
        location = safe_member(p.ROOT, ref["path"])
        if location.parent.parent != path.parent/"snapshots":
            raise ValueError("new snapshot is outside its owning attempt")
        relative = location.parent.relative_to(path.parent).as_posix()
        expected_files.update({f"{relative}/manifest.json", f"{relative}/state.pt", "snapshots/snapshot_publish.lock"})
    for ref in chain["calibrations"]:
        epoch = p.read_reference(ref)["binding"]["epoch"]
        if epoch not in positions:
            raise ValueError("calibration has no mandatory snapshot")
        records.append(calibration_record(ref, binding, calibration_data, snapshot=positions[epoch]))
        if ref not in inherited[2]:
            location = safe_member(p.ROOT, ref["path"])
            if location.parent != path.parent/f"calibration_{epoch:02d}":
                raise ValueError("new calibration output is outside its owning attempt")
            relative = location.parent.relative_to(path.parent).as_posix()
            expected_files.update(f"{relative}/{name}" for name in ("manifest.json", "resources.json", "predictions.npz"))
    if sorted(r["epoch"] for r in records) != list(EPOCHS) or set(m["artifacts_sha256"]) != expected_files:
        raise ValueError("completed cell scientific inventory or epoch roster differs")
    return records, positions


@boundary
def validate_freeze(record, common):
    if (set(record) != FREEZE_FIELDS or record["status"] != "SELECTION_FROZEN" or record["common"] != common
            or not isinstance(record["cells"], list) or len(record["cells"]) != 36):
        raise ValueError("selection freeze schema or completed-cell count differs")
    base, calibration = selection_context(record, common)
    attempts, total, times = accounting()
    if total > CAMPAIGN_SECONDS or any(t > CELL_SECONDS for t in times.values()):
        raise ValueError("selection cannot authorize an over-budget campaign")
    completed = {r["record"]["result"]["path"]: r for r in attempts if r["record"]["status"] == "COMPLETE"}
    if len(completed) != 36:
        raise ValueError("campaign registry lacks exactly 36 completed cells")
    expected = [(a, c, s) for a in ARMS for c in SEEDS for s in READER_SEEDS]
    records = []
    for row, (arm, checkpoint, reader) in zip(record["cells"], expected):
        if set(row) != {"arm", "checkpoint_seed", "reader_seed", "result", "supervisor"} or (
                row["arm"], row["checkpoint_seed"], row["reader_seed"]) != (arm, checkpoint, reader) or any(
                type(row[k]) is not int for k in ("checkpoint_seed", "reader_seed")):
            raise ValueError("cell roster is missing, duplicated or out of order")
        registry = completed.get(row["result"]["path"])
        if registry is None or registry["terminal"] != row["supervisor"] or registry["record"]["result"] != row["result"]:
            raise ValueError("freeze chose an unregistered or superseded attempt")
        binding = {**base, "arm": arm, "checkpoint_seed": checkpoint, "reader_seed": reader}
        vectors, _ = verify_cell(row["result"], row["supervisor"], binding, calibration[checkpoint])
        records.extend(vectors)
    selected = select_epochs(records)
    if record["selection"] is not None and record["selection"] != selected:
        raise ValueError("freeze selection does not replay exactly from all calibration vectors")
    return selected


def create_freeze(output, *, data_authorization, train, calibration, normalizers,
                  normalized_train, normalized_calibration, cells, reuse_audit=None):
    common = gate.common_binding()
    _, train_m = _bundle(train, "learned_training_corpus", common)
    _, cal_m = _bundle(calibration, "learned_training_corpus", common)
    record = {"status": "SELECTION_FROZEN", "common": common, "data_authorization": data_authorization,
        "train": train, "calibration": calibration, "normalizers": normalizers, "normalized_train": normalized_train,
        "normalized_calibration": normalized_calibration, "train_data": train_m["binding"]["data"],
        "calibration_data": cal_m["binding"]["data"], "cells": cells, "selection": None, "reuse_audit": reuse_audit}
    record["selection"] = validate_freeze(record, common)
    if gate.common_binding() != common:
        raise ValueError("source binding changed during selection freeze")
    write_json(output, record)
    return p.reference(output)


@boundary
@memoized
def verify_selection_chain(ref, common):
    record = p.read_reference(ref)
    if record.get("selection") is None:
        raise ValueError("unselected draft cannot authorize test access")
    validate_freeze(record, common)
    if p.read_reference(ref) != record or fresh_pass(gate.common_binding)() != common:
        raise ValueError("selection freeze or implementation changed while checking")
    return record


def create_test_authorization(output, *, freeze, freeze_audit):
    common = gate.common_binding()
    gate.verify_audit(freeze_audit, common, scope="SELECTION_FREEZE", target=freeze)
    verify_selection_chain(freeze, common)
    record = {"status": "TEST_READY", "common": common, "freeze": freeze, "freeze_audit": freeze_audit}
    write_json(output, record)
    return p.reference(output)
