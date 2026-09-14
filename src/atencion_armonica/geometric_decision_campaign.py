"""Fixed 72-cell campaign atop the audited cell runner; no selection/test port."""
from __future__ import annotations

import time
import uuid

from .generative_evidence import CHECKPOINTS
from .geometric_decision_core import ARMS, READER_SEEDS
from .geometric_decision_cell import EPOCHS, read_calibration, run_cell
from .geometric_decision_store import ArtifactStore

ROSTER = tuple((cp, arm, seed) for cp in CHECKPOINTS for arm in ARMS for seed in READER_SEEDS)


def cell_identity(data, cp, arm, seed, device):
    return {"schema": "geometric-decision-cell-binding-v1", "data": data.binding,
            "arm": arm, "checkpoint_seed": cp, "reader_seed": seed, "device": device}


def validate_complete(data, cell, ref):
    value = cell.json(ref)
    steps = ((len(data.eligible["train"])+31)//32)*50
    if (set(value) != {"schema", "binding", "last_epoch", "steps", "last_state", "calibration", "history"}
            or value["schema"] != "geometric-decision-cell-complete-v1" or value["binding"] != cell.binding
            or value["last_epoch"] != 50 or value["steps"] != steps
            or len(value["calibration"]) != 11 or len(value["history"]) != 50):
        raise ValueError("incomplete cell cannot enter the 72-cell completion")
    state = cell.load_state(value["last_state"])
    if (state["epoch"] != 50 or state["next_batch"] != 0 or state["steps"] != steps
            or state["history"] != value["history"]):
        raise ValueError("cell completion differs from retained final numeric state")
    for epoch, calibration in zip(EPOCHS, value["calibration"]):
        record = cell.json(calibration)
        read_calibration(data, cell, calibration, record["state"], epoch)
    return value


def run_campaign(preparation, complete_ref, store, *, device, attempt, check, progress=print, clock=time.monotonic):
    if (store.binding.get("schema") != "geometric-decision-campaign-v1"
            or store.binding.get("preparation") != preparation.store.binding
            or store.binding.get("open_complete") != complete_ref
            or store.binding.get("device") != device
            or store.binding.get("roster") != [list(v) for v in ROSTER]
            or not isinstance(attempt, str) or not attempt):
        raise ValueError("campaign must bind complete OPEN, fixed roster and admitted backend")
    entries = []
    baseline_support = None
    for cp in CHECKPOINTS:
        check()
        before = clock()
        data = preparation.load_checkpoint(complete_ref, cp, check=check)
        support = data.eligible
        if baseline_support is None:
            baseline_support = {key: list(value) for key, value in support.items()}
        elif support != baseline_support:
            raise ValueError("campaign checkpoints differ in eligible OPEN support")
        if any(len(data.rows[key]) != count for key, count in (("train", 4096), ("calibration", 512))):
            raise ValueError("campaign requires full OPEN scene roster")
        store.publish_json(f"timing/load-{uuid.uuid4().hex}.json", {"attempt": attempt, "checkpoint_seed": cp,
            "seconds": clock()-before, "data_binding": data.binding, "kind": "full_checkpoint_load"})
        data_ref = store.publish_json(f"data/cp_{cp}.json", {"binding": data.binding, "eligible": support})
        for arm in ARMS:
            for seed in READER_SEEDS:
                check()
                cell = ArtifactStore(store.root/f"cells/cp_{cp}/{arm}/seed_{seed}",
                                     binding=cell_identity(data, cp, arm, seed, device))
                segment = clock()
                last_epoch = None
                def event(row):
                    nonlocal segment, last_epoch
                    now = clock()
                    timing = {"attempt": attempt, "event": row, "seconds": now-segment,
                        "previous_epoch_event": last_epoch,
                        "kind": "wall_segment_including_recovery_or_calibration_not_pure_optimizer"}
                    store.publish_json(f"timing/epoch-{uuid.uuid4().hex}.json", timing)
                    segment, last_epoch = now, row["epoch"]
                    progress(row)
                try:
                    ref = run_cell(data, cell, arm=arm, checkpoint_seed=cp, reader_seed=seed,
                                   device=device, check=check, progress=event)
                    validate_complete(data, cell, ref)
                finally:
                    store.publish_json(f"timing/tail-{uuid.uuid4().hex}.json", {"attempt": attempt,
                        "checkpoint_seed": cp, "arm": arm, "reader_seed": seed,
                        "previous_epoch_event": last_epoch, "seconds": clock()-segment,
                        "kind": "terminal_or_interrupted_segment_not_an_epoch_completion"})
                entry = {"checkpoint_seed": cp, "arm": arm, "reader_seed": seed,
                    "root": cell.root.relative_to(store.root).as_posix(), "complete": ref, "data": data_ref}
                store.publish_json(f"completed/cp_{cp}/{arm}/seed_{seed}.json", entry)
                entries.append(entry)
                progress({"stage": "cell_complete", "checkpoint_seed": cp, "arm": arm,
                          "reader_seed": seed, "completed_cells": len(entries), "total_cells": 72})
    check()
    if [(e["checkpoint_seed"], e["arm"], e["reader_seed"]) for e in entries] != list(ROSTER):
        raise ValueError("campaign cannot complete with missing or duplicated cells")
    return store.publish_json("complete.json", {"schema": "geometric-decision-campaign-complete-v1",
        "binding": store.binding, "cells": entries, "selection": "not performed", "fresh_tests": "not opened"})
