"""Closed mechanical resource reports and independently recomputable budgets.

Projections are conservative scheduling estimates, never measured campaign
runtime. Safety envelopes remain enforced by the separate process supervisor.
"""
from __future__ import annotations

import math

from .shared_partial_data import mechanical_fixture

BASE = {"status", "namespace", "observations", "seconds", "peak_rss_bytes"}
GEOMETRY = BASE | {"torch_imported", "case_statistics", "case_timings_seconds", "fit_seconds_by_size",
    "source_validation_seconds", "analysis_seconds", "projected_shard_seconds", "projected_disk_bytes",
    "projected_validation_cell_seconds", "projected_analysis_seconds", "disk_components", "validation_io"}
TRAINING = BASE | {"device", "runtime", "availability", "batch_size", "candidate_padding", "group_padding",
    "heads", "projected_cell_seconds", "projected_campaign", "workload"}
GPU = {"peak_reserved_bytes", "per_checkpoint_seconds", "projected_forward_shard_seconds"}
WORKLOAD = {"cells": 36, "updates_per_cell": 6400, "calibration_epochs": 10,
    "calibration_batches_per_epoch": 16, "snapshot_count": 11,
    "warmup_updates": 5, "measured_updates": 20, "measured_evaluation_batches": 5,
    "test_scenes": 2048, "test_inference_passes": 99, "margin": 2}
# 36 selected heads + 54 zero/rotate heads + 9 original-sham heads = 99.
HEAD = {"parameter_count", "setup_seconds", "initial_io_seconds", "update_seconds",
    "evaluation_batch_io_seconds", "metric_batch_seconds", "snapshot_io_seconds", "projected_cell_seconds",
    "linear_multiply_adds_per_batch", "validation"}


def exact(value, keys):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError("resource report schema is incomplete or contains ambiguous extra fields")


def positive(value):
    if type(value) not in (float, int) or not math.isfinite(value) or value <= 0:
        raise ValueError("resource measurement must be finite and positive")
    return value


def sequence(value, count):
    if not isinstance(value, list) or len(value) != count:
        raise ValueError("resource measurement denominator differs")
    for v in value:
        positive(v)


def equal(value, expected):
    positive(value)
    if value != expected:
        raise ValueError("resource projection cannot be reproduced from its measurements")


def packed_upper_bytes(dim):
    return 512*(4*(94*dim+64*6+64*94)+94+64+8)+6*1024


def validate_validation_io(value):
    exact(value, {"hash_small", "hash_large", "inventory", "bundle", "packed"})
    for name, size in (("hash_small", 256), ("hash_large", 16*1024**2)):
        exact(value[name], {"bytes", "seconds"})
        if type(value[name]["bytes"]) is not int or value[name]["bytes"] != size:
            raise ValueError("hash primitive byte denominator differs")
        sequence(value[name]["seconds"], 3)
    for name, fields in (("inventory", {"entries", "files", "seconds"}),
                         ("bundle", {"entries", "files", "bytes", "seconds"})):
        record = value[name]
        exact(record, fields)
        if any(type(record[k]) is not int or record[k] != v for k, v in (("entries", 3078), ("files", 3075))):
            raise ValueError("bundle/inventory primitive denominator differs")
        sequence(record["seconds"], 3)
    if type(value["bundle"]["bytes"]) is not int:
        raise ValueError("bundle byte count must be integral")
    positive(value["bundle"]["bytes"])
    exact(value["packed"], {"8", "9"})
    for dim in (8, 9):
        record = value["packed"][str(dim)]
        exact(record, {"scene_count", "dim", "compressed_bytes", "uncompressed_upper_bytes", "seconds"})
        expected = {"scene_count": 512, "dim": dim, "uncompressed_upper_bytes": packed_upper_bytes(dim)}
        if any(type(record[k]) is not int or record[k] != v for k, v in expected.items()):
            raise ValueError("packed primitive dimension/byte denominator differs")
        if type(record["compressed_bytes"]) is not int or not 0 < record["compressed_bytes"] <= packed_upper_bytes(dim):
            raise ValueError("packed primitive byte accounting differs")
        sequence(record["seconds"], 3)


def geometry_projections(r):
    timings, cases = r["case_timings_seconds"], r["case_statistics"]
    fit_upper = 186*max(max(v) for v in r["fit_seconds_by_size"].values())
    scene_artifacts = 6656*max(c["bytes"] for c in cases)*4
    predictions = (36*10*512+99*4*512)*64*2*4*4
    components = {"scene_artifacts": scene_artifacts, "predictions": predictions,
                  "checkpoint_allowance": 2*1024**3, "margin": 2}
    return {"projected_shard_seconds": 2*512*(fit_upper+max(sum(t.values()) for t in timings)),
        "projected_disk_bytes": 2*(scene_artifacts+predictions+2*1024**3), "disk_components": components,
        # Each measured scene includes all three checkpoints and four arms.
        "projected_validation_cell_seconds": 2*(sum(r["source_validation_seconds"])
            +4608*max(t["load_normalize_and_input_io"] for t in timings)/3),
        "projected_analysis_seconds": 2*(r["analysis_seconds"]["selection"]
            +8*(r["analysis_seconds"]["one_test_summary"]+r["analysis_seconds"]["one_intervention_summary"]))}


def head_projection(h, geometry):
    return 2*(h["setup_seconds"]+h["initial_io_seconds"]+6400*max(h["update_seconds"])
        +10*h["validation"]["calibration_write_seconds"]
        +10*h["snapshot_io_seconds"])+geometry["projected_validation_cell_seconds"]


def campaign_projection(heads, geometry):
    worst_eval = max(max(h["evaluation_batch_io_seconds"]) for h in heads.values())
    worst_metric = max(max(h["metric_batch_seconds"]) for h in heads.values())
    cells = 36*max(h["projected_cell_seconds"] for h in heads.values())
    # Three backbones, four512-scene tests, at most64 candidates + historical
    # partition per scene. Learned choices reuse these candidate metrics.
    test_metrics = 2*3*64*65*worst_metric
    tests = 2*99*64*worst_eval+test_metrics
    # Replay loads predictions and recomputes metrics, with no new forward.
    replay = test_metrics
    # 63 interventions: inference diagnostics, primary evaluation and replay.
    support = 2*3*63*64*geometry["analysis_seconds"]["support_batch_seconds"]
    validation = 4*geometry["projected_validation_cell_seconds"]
    analysis = geometry["projected_analysis_seconds"]
    return {"training_and_calibration": cells, "test_inference_and_metrics": tests,
            "replay_metrics": replay, "test_artifact_validation": validation,
            "support_diagnostics": support,
            "selection_and_summaries": analysis, "total": cells+tests+replay+validation+analysis+support}


def validate_geometry(r):
    exact(r, GEOMETRY)
    validate_validation_io(r["validation_io"])
    if r["observations"] != [mechanical_fixture(4, deformed=d)[0] for d in (False, True)]:
        raise ValueError("geometry observations do not match the two measured fixtures in order")
    if r["torch_imported"] is not False:
        raise ValueError("geometry profile imported Torch")
    sequence(r["source_validation_seconds"], 2)
    exact(r["analysis_seconds"], {"selection", "one_test_summary", "one_intervention_summary", "support_batch_seconds"})
    for v in r["analysis_seconds"].values():
        positive(v)
    exact(r["fit_seconds_by_size"], map(str, range(3, 9)))
    for times in r["fit_seconds_by_size"].values():
        sequence(times, 3)
    if len(r["case_statistics"]) != 2 or len(r["case_timings_seconds"]) != 2:
        raise ValueError("both observable mechanical fixtures are required")
    for case, times in zip(r["case_statistics"], r["case_timings_seconds"]):
        exact(case, {"n", "candidate_counts", "group_counts", "bytes"})
        if case["n"] != 32 or type(case["bytes"]) is not int:
            raise ValueError("geometry scene dimensions differ")
        positive(case["bytes"])
        for key, bound in (("candidate_counts", 64), ("group_counts", 94)):
            if (not isinstance(case[key], list) or len(case[key]) != 3
                    or any(type(v) is not int or not 1 <= v <= bound for v in case[key])):
                raise ValueError("geometry checkpoint/pool dimensions differ")
        exact(times, {"features", "scoring_and_raw_io", "load_normalize_and_input_io"})
        for value in times.values():
            positive(value)
    for key, value in geometry_projections(r).items():
        if key == "disk_components":
            if r[key] != value:
                raise ValueError("disk component derivation differs")
        else:
            equal(r[key], value)


def validate_training(r, geometry, *, gpu):
    exact(r, TRAINING | (GPU if gpu else set()))
    if r["observations"] != ([mechanical_fixture(4)[0]] if gpu else []):
        raise ValueError("training profile observable roster differs from its forward workload")
    exact(r["runtime"], {"torch", "numpy", "device"} | ({"cuda", "cudnn"} if gpu else set()))
    for key in ("torch", "numpy", "device"):
        if not isinstance(r["runtime"][key], str) or not r["runtime"][key].strip():
            raise ValueError("profile runtime field is empty")
    if r["runtime"]["device"] != r["device"]:
        raise ValueError("runtime and profile devices differ")
    if gpu:
        if (not isinstance(r["runtime"]["cuda"], str) or not r["runtime"]["cuda"].strip()
                or type(r["runtime"]["cudnn"]) is not int or r["runtime"]["cudnn"] <= 0):
            raise ValueError("GPU runtime omitted CUDA/cuDNN versions")
        a = r["availability"]
        exact(a, {"device", "uuid", "used_mib_before", "total_mib", "compute_processes_before"})
        if (a["device"] != r["device"] or not isinstance(a["uuid"], str) or not a["uuid"].startswith("GPU-")
                or len(a["uuid"]) <= 4 or a["compute_processes_before"] != []):
            raise ValueError("GPU availability device identity or process state differs")
        positive(a["total_mib"])
        used = a["used_mib_before"]
        if (type(used) not in (int, float) or not math.isfinite(used) or used < 0
                or a["total_mib"]-used < 2048):
            raise ValueError("GPU availability memory envelope differs")
    elif r["availability"] is not None:
        raise ValueError("CPU profile cannot claim a GPU lease")
    if r["workload"] != WORKLOAD:
        raise ValueError("training resource workload or denominators differ")
    exact(r["heads"], {"pairs_structure", "shared_source"})
    for arm, h in r["heads"].items():
        exact(h, HEAD)
        validation = h["validation"]
        exact(validation, {"snapshot_chain_seconds", "snapshot_unique_counts", "snapshot_chain_bytes",
            "calibration_write_seconds", "calibration_read_seconds", "calibration_scene_count", "calibration_candidate_count"})
        sequence(validation["snapshot_chain_seconds"], 3)
        sequence(validation["calibration_read_seconds"], 3)
        if (validation["snapshot_unique_counts"] != [11]*3 or validation["calibration_scene_count"] != 512
                or validation["calibration_candidate_count"] != 64 or type(validation["snapshot_chain_bytes"]) is not int):
            raise ValueError("snapshot/calibration validation workload differs")
        positive(validation["snapshot_chain_bytes"])
        positive(validation["calibration_write_seconds"])
        dim, width, parameters = (8, 33, 1643) if arm == "pairs_structure" else (9, 32, 1650)
        madds = 32*(94*(dim*width+width*16)+64*(94*16+22*32+32*2))
        if h["parameter_count"] != parameters or h["linear_multiply_adds_per_batch"] != madds:
            raise ValueError("profiled head architecture differs")
        sequence(h["update_seconds"], 20)
        sequence(h["evaluation_batch_io_seconds"], 5)
        sequence(h["metric_batch_seconds"], 5)
        for key in ("setup_seconds", "initial_io_seconds", "snapshot_io_seconds"):
            positive(h[key])
        equal(h["projected_cell_seconds"], head_projection(h, geometry))
    equal(r["projected_cell_seconds"], max(h["projected_cell_seconds"] for h in r["heads"].values()))
    if r["projected_campaign"] != campaign_projection(r["heads"], geometry):
        raise ValueError("campaign operation projections differ")
    if gpu:
        sequence(r["per_checkpoint_seconds"], 3)
        equal(r["projected_forward_shard_seconds"], 2*(4*sum(r["per_checkpoint_seconds"])+r["seconds"]))
