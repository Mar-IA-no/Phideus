"""Closed mechanical resource reports and independently recomputable budgets.

Projections are conservative scheduling estimates, never measured campaign
runtime. Safety envelopes remain enforced by the separate process supervisor.
"""
from __future__ import annotations

import math
import hashlib
from pathlib import Path

from .shared_partial_data import mechanical_fixture
from .partial_compatibility_cache import encoded, sha_file

BASE = {"status", "namespace", "observations", "seconds", "peak_rss_bytes"}
SEMANTIC_OPERATIONS = 32  # Amortized throughput of fixed files, not worst-case single-read latency.
GEOMETRY = BASE | {"torch_imported", "case_statistics", "case_timings_seconds", "fit_seconds_by_size",
    "source_validation_seconds", "analysis_seconds", "projected_shard_seconds", "projected_disk_bytes",
    "projected_validation_cell_seconds", "projected_analysis_seconds", "disk_components", "validation_io",
    "semantic_validation", "metadata_validation", "validation_plan", "validation_units"}
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


def validation_plan():
    """R657 clean-path counts. Retry increments use actual ancestry lengths.

    Bounds below are scheduling allowances, not additional experiment limits.
    The report carries this exact table and the current source hash.
    """
    ledger = {
        "cell_pass": {"corpora": 2, "observation_bundles": 9, "observation_payload_shards": 18,
            "logits_bundles": 9, "scored_bundles": 9, "target_bundles": 9, "normalized_bundles": 9,
            "packed_reads": 9, "normalizers": 1, "rows_target_scene_reads": 18432,
            "scene_count": 4608, "passes_per_cell": 2},
        "publication": {"snapshots": 11, "top_level_reads": 31, "state_loads": 176},
        "freeze": {"corpora": 2, "cells": 36, "state_loads": 396, "calibrations": 360,
            "calibration_choices": 184320, "fresh_source_checks": 1},
        "test_stages": {name: {"worker_freezes": 2, "supervisor_freezes": int(name == "test_inference"),
                               "corpora": 5 if name == "normalized" else 6 if name == "test_inference" else 4}
            for name in ("prepare", "aggregate_data", "forward", "score", "normalized", "test_inference", "test_evaluation")},
        "test_primary": {"splits": 4, "freezes_including_authorization": 61, "corpora": 126,
            "selected_manifest_scans": 1584, "selected_state_loads_upper": 1584,
            "prefix_split_and_shard_checks_each": 84},
        "replay": {"evaluations": 4, "freezes": 8, "corpora": 16},
        "allowances": {"metadata_bytes_per_reference": 2*1024**2, "metadata_reads_per_stage": 512,
            "snapshot_extra_bytes_per_state": 512*1024, "margin": 2},
        "retry": {"policy": "add_actual_ancestry_and_repeat_stage_cost_no_retry_count_cap",
                  "elapsed_authority": "supervisor_and_cumulative_budget"}}
    return {"version": "learned-validation-cost-v2", "source_sha256": sha_file(Path(__file__)),
            "ledger_sha256": hashlib.sha256(encoded(ledger)).hexdigest(), "ledger": ledger}


def validate_semantic_measurements(cases):
    if not isinstance(cases, list) or len(cases) != 2:
        raise ValueError("both fixed cases are needed for semantic cost measurements")
    for case in cases:
        exact(case, {"n", "operations_per_repeat", "observation_feature_seconds", "observation_feature_bytes", "checkpoints"})
        if type(case["operations_per_repeat"]) is not int or case["operations_per_repeat"] != SEMANTIC_OPERATIONS:
            raise ValueError("semantic block operation denominator differs")
        if type(case["n"]) is not int or case["n"] != 32 or len(case["checkpoints"]) != 3:
            raise ValueError("semantic case dimensions or checkpoints differ")
        sequence(case["observation_feature_seconds"], 3)
        if type(case["observation_feature_bytes"]) is not int:
            raise ValueError("semantic payload bytes must be integral")
        positive(case["observation_feature_bytes"])
        for row in case["checkpoints"]:
            exact(row, {"candidate_count", "group_count", "pool_rows_seconds", "target_metrics_seconds",
                "logits_seconds", "model_inputs_seconds", "pool_rows_bytes", "target_metrics_bytes", "logits_bytes"})
            for key, bound in (("candidate_count", 64), ("group_count", 94)):
                if type(row[key]) is not int or not 1 <= row[key] <= bound:
                    raise ValueError("semantic row dimensions differ")
            for name in ("pool_rows", "target_metrics", "logits", "model_inputs"):
                sequence(row[name+"_seconds"], 3)
                if name != "model_inputs":
                    if type(row[name+"_bytes"]) is not int:
                        raise ValueError("semantic payload bytes must be integral")
                    positive(row[name+"_bytes"])


def validation_units(r):
    """Unmargined costs; consumers apply margin once to their stage subtotal.

    The maximum measured per-file bundle cost includes inventory and metadata;
    a separate byte coefficient conservatively accounts for large payloads.
    Semantic costs are scaled to64candidates/94groups, not called measured full
    campaign times. Selected rereads use the fuller reader as an upper estimate.
    """
    io, cases = r["validation_io"], r["semantic_validation"]
    rows = [row for case in cases for row in case["checkpoints"]]
    file_seconds = max(max(io["bundle"]["seconds"])/io["bundle"]["files"],
                       max(io["hash_small"]["seconds"])+max(io["inventory"]["seconds"])/io["inventory"]["entries"])
    byte_seconds = max(io["hash_large"]["seconds"])/io["hash_large"]["bytes"]
    metadata_bytes = validation_plan()["ledger"]["allowances"]["metadata_bytes_per_reference"]
    def checked_files(count, size):
        return count*file_seconds+size*byte_seconds
    def scaled(row, name):
        scale = max(64/row["candidate_count"], 94/row["group_count"])
        return max(row[name+"_seconds"])*scale/SEMANTIC_OPERATIONS
    feature_seconds = max(max(c["observation_feature_seconds"])/SEMANTIC_OPERATIONS for c in cases)
    feature_bytes = max(c["observation_feature_bytes"] for c in cases)
    payload_bytes = {name: max(row[name+"_bytes"]*max(64/row["candidate_count"],94/row["group_count"])
                              for row in rows) for name in ("pool_rows", "target_metrics", "logits")}
    semantic = {name: max(scaled(row, name) for row in rows)
                for name in ("pool_rows", "target_metrics", "logits", "model_inputs")}
    bundles = {
        "observation": checked_files(518, 512*feature_bytes+4*metadata_bytes),
        "scored": checked_files(3075, 1536*payload_bytes["pool_rows"]+metadata_bytes),
        "targets": checked_files(3074, 1536*payload_bytes["target_metrics"]+metadata_bytes),
        "logits": checked_files(6, 1536*payload_bytes["logits"]+metadata_bytes),
        "normalized": checked_files(14, 12*packed_upper_bytes(9)+metadata_bytes),
        "small": checked_files(5, metadata_bytes)}
    # Geometry profile contains the3075-file primitive; both Torch profiles are
    # small snapshot/calibration trees. File/byte allowances stay explicit.
    profiles = checked_files(3400, 128*1024**2)
    authority = max(r["source_validation_seconds"])+profiles
    corpus_pair = authority+4*bundles["small"]+9*sum(bundles[k] for k in ("observation", "scored", "targets", "logits"))
    corpus_pair += 4608*feature_seconds+13824*sum(semantic[k] for k in ("pool_rows", "target_metrics", "logits"))
    packed = max(max(v["seconds"]) for v in io["packed"].values())
    cell_pass = {"corpus_pair": corpus_pair, "observation_reread": 4608*feature_seconds,
        "normalized_bundles": 9*bundles["normalized"], "normalizer": bundles["small"],
        "selected_rows_targets": 4608*(semantic["pool_rows"]+semantic["target_metrics"]),
        "selected_input_parity": 4608*semantic["model_inputs"], "packed_reads": 9*packed}
    selection_context = {"corpus_pair": corpus_pair, "normalized_bundles": 9*bundles["normalized"],
        "normalizer": bundles["small"], "calibration_rows_metrics":1536*(semantic["pool_rows"]+semantic["target_metrics"])}
    metadata_read = max(r["metadata_validation"]["seconds"])*max(1., metadata_bytes/r["metadata_validation"]["bytes"])
    return {"file_seconds": file_seconds, "byte_seconds": byte_seconds, "authority": authority,
        "metadata_reference": checked_files(1, metadata_bytes)+metadata_read, "bundles": bundles,
        "feature_seconds": feature_seconds, "semantic_seconds": semantic, "packed_seconds": packed,
        "cell_pass": cell_pass, "selection_context": selection_context}


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
        "projected_validation_cell_seconds": 2*(2*sum(r["validation_units"]["cell_pass"].values())
            +2*r["validation_units"]["authority"]+2*max(r["source_validation_seconds"])
            +512*r["validation_units"]["metadata_reference"]),
        "projected_analysis_seconds": 2*(r["analysis_seconds"]["selection"]
            +8*(r["analysis_seconds"]["one_test_summary"]+r["analysis_seconds"]["one_intervention_summary"]))}


def head_projection(h, geometry):
    return 2*(h["setup_seconds"]+h["initial_io_seconds"]+6400*max(h["update_seconds"])
        +10*h["validation"]["calibration_write_seconds"]
        +10*h["snapshot_io_seconds"])+geometry["projected_validation_cell_seconds"]


def freeze_projection(heads, geometry):
    """Unmargined freeze cost, with all396states and360calibration bundles."""
    u = geometry["validation_units"]
    v = [h["validation"] for h in heads.values()]
    chain_time = max(max(row["snapshot_chain_seconds"]) for row in v)
    chain_bytes = max(row["snapshot_chain_bytes"] for row in v)+11*512*1024
    cal_time = max(max(row["calibration_read_seconds"]) for row in v)
    cal_bytes = 512*64*2*4+512*32+2*1024**2
    # verify_cell hashes every snapshot/calibration again as part of the cell
    # bundle, separately from their semantic loaders. Metadata allowance covers
    # requests/registry/terminal checks, manifest scans and fifty-epoch history.
    cell_bundle = 59*u["file_seconds"]+(chain_bytes+10*cal_bytes)*u["byte_seconds"]
    terms = {"selection_context": sum(u["selection_context"].values()),
        "cell_bundles": 36*cell_bundle, "snapshot_chains": 36*chain_time,
        "calibration_replay": 360*cal_time,
        "metadata_and_history_allowance": 512*u["metadata_reference"],
        "fresh_source_check": max(geometry["source_validation_seconds"])}
    return {"terms": terms, "subtotal": sum(terms.values()), "state_loads": 396,
            "calibration_bundles": 360, "margin_applied": False}


def validation_stage_projections(heads, geometry):
    """Per-test-stage dependency costs only; compute is added by stage_projection.

    Each row uses the largest predecessor prefix for admission and totals.
    The ledger separately preserves the exact84primary-prefix checks; it must
    not be confused with this deliberately conservative scheduling estimate.
    """
    u = geometry["validation_units"]
    f = freeze_projection(heads, geometry)["subtotal"]
    prefix_unit = u["bundles"]["small"]+u["bundles"]["observation"]+512*u["feature_seconds"]
    own = (sum(u["bundles"].values())+512*u["feature_seconds"]
           +1536*sum(u["semantic_seconds"].values())+12*u["packed_seconds"])
    result = {}
    for stage, count in validation_plan()["ledger"]["test_stages"].items():
        terms = {"freeze_worker": count["worker_freezes"]*f,
            "source_request_checks": 2*max(geometry["source_validation_seconds"]),
            "predecessors_upper": 6*prefix_unit, "own_payloads_upper": 2*own,
            "metadata_allowance": 512*u["metadata_reference"]}
        if stage in ("forward", "score"):
            terms["post_seal_reader_upper"] = own
        if stage == "normalized":
            terms["extra_train_corpus_upper"] = u["selection_context"]["corpus_pair"]
        if stage == "test_inference":
            terms["selected_snapshot_ancestry"] = 36*max(max(h["validation"]["snapshot_chain_seconds"]) for h in heads.values())
            terms["selected_manifest_scans"] = 396*u["metadata_reference"]
        result[stage] = {"terms": terms, "worker_seconds": 2*sum(terms.values()),
            "supervisor_preflight_seconds": 2*count["supervisor_freezes"]*f,
            "worker_freezes": count["worker_freezes"], "corpora_total": count["corpora"]}
    return result


def stage_projection(heads, geometry, *, forward_shard_seconds):
    """Admission estimates include compute and validation, not only FLOPs."""
    stages = validation_stage_projections(heads, geometry)
    eval_batch = max(max(h["evaluation_batch_io_seconds"]) for h in heads.values())
    metric_batch = max(max(h["metric_batch_seconds"]) for h in heads.values())
    support = 2*63*16*geometry["analysis_seconds"]["support_batch_seconds"]
    summary = 2*(geometry["analysis_seconds"]["one_test_summary"]+geometry["analysis_seconds"]["one_intervention_summary"])
    # Both terms already include one margin. Never multiply their sum again.
    compute = {"prepare": 2*512*(max(t["features"] for t in geometry["case_timings_seconds"])
                                 +geometry["validation_units"]["feature_seconds"]), "aggregate_data": 0.,
        "forward": forward_shard_seconds, "score": geometry["projected_shard_seconds"],
        "normalized": 2*1536*geometry["validation_units"]["semantic_seconds"]["model_inputs"],
        "test_inference": 2*99*16*eval_batch+support,
        "test_evaluation": 2*3*16*65*metric_batch+support+summary}
    return {stage: {**row, "compute_seconds": compute[stage],
        "worker_total_seconds": row["worker_seconds"]+compute[stage]} for stage, row in stages.items()}


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
    stages = validation_stage_projections(heads, geometry)
    freeze = freeze_projection(heads, geometry)
    # Includes independent replay evaluations and the initial creation and
    # authorization of the freeze. Stage rows expose their before/after costs.
    validation = 4*sum(row["worker_seconds"]+row["supervisor_preflight_seconds"] for row in stages.values())
    validation += 4*stages["test_evaluation"]["worker_seconds"]+4*freeze["subtotal"]
    analysis = geometry["projected_analysis_seconds"]
    return {"training_and_calibration": cells, "test_inference_and_metrics": tests,
            "replay_metrics": replay, "test_artifact_validation": validation,
            "support_diagnostics": support,
            "selection_and_summaries": analysis, "validation_stages": stages, "freeze_validation": freeze,
            "total": cells+tests+replay+validation+analysis+support}


def validate_geometry(r):
    exact(r, GEOMETRY)
    validate_validation_io(r["validation_io"])
    validate_semantic_measurements(r["semantic_validation"])
    exact(r["metadata_validation"], {"bytes", "seconds"})
    if type(r["metadata_validation"]["bytes"]) is not int:
        raise ValueError("metadata byte denominator must be integral")
    positive(r["metadata_validation"]["bytes"])
    sequence(r["metadata_validation"]["seconds"], 3)
    if encoded(r["validation_plan"]) != encoded(validation_plan()):
        raise ValueError("validation ledger/version/source differs")
    if r["validation_units"] != validation_units(r):
        raise ValueError("validation cost terms do not reproduce their primitives")
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
    for case, times, semantic in zip(r["case_statistics"], r["case_timings_seconds"], r["semantic_validation"]):
        exact(case, {"n", "candidate_counts", "group_counts", "bytes"})
        if case["n"] != 32 or type(case["bytes"]) is not int:
            raise ValueError("geometry scene dimensions differ")
        positive(case["bytes"])
        for key, bound in (("candidate_counts", 64), ("group_counts", 94)):
            if (not isinstance(case[key], list) or len(case[key]) != 3
                    or any(type(v) is not int or not 1 <= v <= bound for v in case[key])):
                raise ValueError("geometry checkpoint/pool dimensions differ")
        if (semantic["n"] != case["n"]
                or [row["candidate_count"] for row in semantic["checkpoints"]] != case["candidate_counts"]
                or [row["group_count"] for row in semantic["checkpoints"]] != case["group_counts"]):
            raise ValueError("semantic timing denominators differ from the measured case/checkpoint")
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
