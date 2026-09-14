"""Explicit campaign work counts and head cost forecast; never selects science."""
from __future__ import annotations

import math

import numpy as np

from .geometric_decision_core import ARMS, READER_SEEDS
from .generative_evidence import CHECKPOINTS


def training_work(eligible_train, eligible_calibration):
    if (type(eligible_train) is not int or not 1 <= eligible_train <= 4096
            or type(eligible_calibration) is not int or not 1 <= eligible_calibration <= 512):
        raise ValueError("explicit full-corpus eligible counts required")
    batches = math.ceil(eligible_train/32)
    steps = batches*50
    snapshots = {0, *range(32, steps+1, 32), *range(batches, steps+1, batches)}
    cells = len(ARMS)*len(CHECKPOINTS)*len(READER_SEEDS)
    return {"cells": cells, "epochs_per_cell": 50, "updates_per_epoch": batches,
        "last_batch_scenes": eligible_train-(batches-1)*32,
        "updates": cells*steps, "warmup_updates": cells*5, "steady_updates": cells*(steps-5),
        "snapshots": cells*len(snapshots),
        "noninitial_snapshots": cells*(len(snapshots)-1),
        "calibration_batches": cells*11*math.ceil(eligible_calibration/32),
        "checkpoint_corpus_loads": len(CHECKPOINTS)}


def head_forecast(cases, *, eligible_train, eligible_calibration, corpus_load_seconds):
    """Conservative case-mean envelope plus separately measured full-corpus load.

    This is a forecast, not an upper-bound proof or backend admission. The caller
    must authenticate source receipts and apply the fixed stage budget/margin.
    A subset extraction time is not silently relabeled a full CellData load.
    """
    expected = {(kind, objective) for kind in ("envelope", "first_train_batch") for objective in ("mse", "decision")}
    if set(cases) != expected or type(corpus_load_seconds) not in (int, float) or not math.isfinite(corpus_load_seconds) or corpus_load_seconds <= 0:
        raise ValueError("four paired profile cases and measured full-checkpoint load required")
    devices = set()
    for (kind, objective), case in cases.items():
        if (case["status"] != "MECHANICAL_PROFILE_NOT_TRAINED_CELL" or case["objective"] != objective
                or len(case["all_update_seconds"]) != 25 or len(case["snapshot_io_seconds"]) != 2
                or len(case["evaluation_batch_io_seconds"]) != 3 or not case["exact_recovery_digest"]):
            raise ValueError("incomplete head profile cannot supply a cost forecast")
        devices.add(case["device"])
        times = [case["setup_seconds"], *case["all_update_seconds"], *case["snapshot_io_seconds"], *case["evaluation_batch_io_seconds"]]
        if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in times):
            raise ValueError("finite positive measured timings required")
    if len(devices) != 1 or not devices <= {"cpu", "cuda:0"}:
        raise ValueError("one backend per head forecast required")
    work = training_work(eligible_train, eligible_calibration)
    rates = {"warmup_updates": max(float(np.mean(c["all_update_seconds"][:5])) for c in cases.values()),
             "steady_updates": max(float(np.mean(c["all_update_seconds"][5:])) for c in cases.values()),
             "noninitial_snapshots": max(max(c["snapshot_io_seconds"]) for c in cases.values()),
             "calibration_batches": max(max(c["evaluation_batch_io_seconds"]) for c in cases.values()),
             "cells": max(c["setup_seconds"] for c in cases.values()),
             "checkpoint_corpus_loads": float(corpus_load_seconds)}
    costs = {key: value*work[key] for key, value in rates.items()}
    return {"device": next(iter(devices)), "work": work, "measured_unit_seconds": rates,
            "projected_cost_components": costs, "projected_seconds_before_margin": sum(costs.values()),
            "method": "first five updates per cell charged separately; max case mean remaining updates; max snapshot/evaluation/setup; measured corpus load",
            "limits": "timing forecast only; excludes future tests and does not prove an upper bound"}
