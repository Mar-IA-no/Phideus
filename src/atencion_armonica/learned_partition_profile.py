"""Three bounded mechanical profiles; no prospective scene generation.

Entry points require the full implementation audit, including this source.
CPU geometry does not import Torch; training profiles import it only locally.
"""
from __future__ import annotations

from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import resource
import sys
import time

import numpy as np

from . import learned_partition_gate as gate
from . import learned_partition_provenance as p
from . import learned_partition_resources as resources
from .learned_partition_cache import load_rows, save_rows
from .learned_partition_core import fit_normalizer, model_inputs, observable_features, partition_errors
from .partial_compatibility_cache import encoded, feature_record, sha_file
from .shared_partial_data import mechanical_fixture
from .source_coherence import GroupFitCache, SourceFitter
from .structured_source_artifacts import mark_failure, seal_bundle, write_json, write_npz, verify_bundle, _inventory
from .structured_source_reader import score_scene
from .structured_source_metrics import partition_metrics


def analysis_measurements():
    """Complete fixed metric-array workloads; no observations or scene draws."""
    from .learned_partition_metrics import (ARMS, SEEDS, READER_SEEDS, EPOCHS, SPLITS,
        METRICS, REFERENCES, select_epochs, summarize_test, bootstrap_indices)
    before = time.monotonic()
    records = [{"arm": a, "checkpoint_seed": c, "reader_seed": s, "epoch": e,
        "split": "calibration", "split_seed": SPLITS["calibration"][1],
        "scene_ids": list(range(512)), "ari": [.5]*512}
        for a in ARMS for c in SEEDS for s in READER_SEEDS for e in EPOCHS]
    select_epochs(records)
    selection = time.monotonic()-before
    before = time.monotonic()
    records = [{"scene_id": i, "checkpoint_seed": c, "reader_seed": s,
        "split": "ood_polyphony", "split_seed": SPLITS["ood_polyphony"][1],
        "metrics": {a: dict.fromkeys(METRICS, .5) for a in (*ARMS, *REFERENCES)}}
        for i in range(512) for c in SEEDS for s in READER_SEEDS]
    summarize_test(records, split="ood_polyphony", indices=bootstrap_indices())
    summary = time.monotonic()-before
    del records
    from .learned_partition_test import inference_roster, summarize_interventions
    before = time.monotonic()
    records = [{**entry, "scene_id": i, "candidate_index": 0, "original_candidate_index": 0,
        "metrics": dict.fromkeys(METRICS, .5), "original_metrics": dict.fromkeys(METRICS, .5),
        "delta_vs_original": dict.fromkeys(METRICS, 0.)}
        for entry in inference_roster() if entry["intervention"] != "original" for i in range(512)]
    summarize_interventions(records, split="ood_polyphony", indices=bootstrap_indices())
    intervention_summary = time.monotonic()-before
    del records
    from .learned_partition_readout import input_support, prediction_dependence
    original = mechanical_inputs(9)
    changed = {k: v.copy() for k, v in original.items()}
    changed["groups"][:, 8] = 0
    candidates = []
    for mask in range(64):
        cuts = [0]+[i+1 for i in range(6) if mask & (1 << i)]+[32]
        candidates.append([list(range(lo, hi)) for lo, hi in zip(cuts[:-1], cuts[1:])])
    values = np.linspace(0, 1, 128, dtype=np.float32).reshape(64, 2)
    before = time.monotonic()
    for _ in range(32):
        json.dumps({"input": input_support(original, changed),
                    "prediction": prediction_dependence(values, values[::-1].copy(), candidates)}, allow_nan=False)
    # Dense94-group incidence overbounds the support operation's actual sparsity.
    return {"selection": selection, "one_test_summary": summary,
            "one_intervention_summary": intervention_summary,
            "support_batch_seconds": time.monotonic()-before}


def _limit(started, *, geometry=False):
    seconds = time.monotonic()-started
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    if seconds > 120 or rss >= (1 if geometry else 4)*1024**3:
        raise RuntimeError("mechanical profile exceeds its fixed time/RSS envelope")
    return seconds, rss


def geometry_profile(output, *, audit):
    started = time.monotonic()
    common = gate.common_binding()
    gate.verify_audit(audit, common, scope="FULL_IMPLEMENTATION")
    p.prior_corpus()
    validation_times = [time.monotonic()-started]
    if "torch" in sys.modules:
        raise RuntimeError("geometry profile requires a separate CPU process without Torch")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    try:
        observations, cases, timings, fit_times, semantic = [], [], [], {}, []
        fitter = SourceFitter()
        for deformed in (False, True):
            obs, truth = mechanical_fixture(4, deformed=deformed)
            observations.append(obs)
            q = np.asarray(obs["log_f"], np.float32)
            before = time.monotonic()
            record = feature_record(obs)
            feature_seconds = time.monotonic()-before
            if not fit_times:
                for size in range(3, 9):
                    measured = []
                    for _ in range(3):
                        before = time.monotonic()
                        fitter.fit(q, list(range(size)))
                        measured.append(time.monotonic()-before)
                        _limit(started, geometry=True)
                    fit_times[str(size)] = measured
            a = np.arange(32*32, dtype=np.float32).reshape(32, 32)
            matrices = [(4*record["pair_support"]-2).astype(np.float32),
                        np.sin(a+a.T).astype(np.float32),
                        (2-4*np.abs(q[:, None]-q[None, :])).astype(np.float32)]
            case = output/f"fixture_{int(deformed)}"
            case.mkdir()
            write_json(case/"observation.json", obs)
            write_npz(case/"features.npz", **record)
            fit_cache = GroupFitCache(fitter)
            before = time.monotonic()
            rows = []
            for i, z in enumerate(matrices):
                scored = score_scene(q, z, record["pair_support"], record["triples"], record["residual_cents"],
                                     split_seed=obs["split_seed"], scene_id=obs["scene_id"], fit_cache=fit_cache)
                row = observable_features(scored, z)
                rows.append(row)
                write_json(case/f"pool_{i}.json", scored)
                write_npz(case/f"logits_{i}.npz", logits=z)
                save_rows(case/f"rows_{i}.npz", row)
            scoring_io_seconds = time.monotonic()-before
            before = time.monotonic()
            for i in range(3):
                for path in (case/f"rows_{i}.npz", case/f"pool_{i}.json", case/f"logits_{i}.npz"):
                    p.reference(path)
                json.loads((case/f"pool_{i}.json").read_bytes())
                loaded = load_rows(case/f"rows_{i}.npz")
                normalizer = fit_normalizer([loaded], expected_count=1)
                for arm in ("pairs_structure", "local_compatibility", "shared_source", "decoupled_source"):
                    write_npz(case/f"normalized_{i}_{arm}.npz", **model_inputs(loaded, normalizer, arm))
            input_seconds = time.monotonic()-before
            semantic.append(semantic_measurements(case, obs, truth, rows, matrices))
            payload_bytes = sum(path.stat().st_size for path in case.iterdir())
            timings.append({"features": feature_seconds, "scoring_and_raw_io": scoring_io_seconds,
                            "load_normalize_and_input_io": input_seconds})
            cases.append({"n": 32, "candidate_counts": [len(r.candidates) for r in rows],
                          "group_counts": [len(r.groups) for r in rows], "bytes": payload_bytes})
            _limit(started, geometry=True)
        validation_io = validation_io_measurements(output/"validation_io")
        from copy import deepcopy
        from .learned_partition_validation import _key
        metadata_path = output/"metadata_fixture.json"
        write_json(metadata_path, {"common": common, "prior": p.prior_corpus(), "audit": audit})
        metadata_ref = p.reference(metadata_path)
        metadata_seconds = timed_repeats(lambda: _key(deepcopy(p.read_reference(metadata_ref))))
        analysis = analysis_measurements()
        before = time.monotonic()
        if gate.common_binding() != common:
            raise ValueError("profile source snapshot changed")
        gate.verify_audit(audit, common, scope="FULL_IMPLEMENTATION")
        p.prior_corpus()
        validation_times.append(time.monotonic()-before)
        seconds, rss = _limit(started, geometry=True)
        report = {"status": "MEASURED", "namespace": "MECHANICAL_NOT_PROSPECTIVE", "observations": observations,
                  "seconds": seconds, "peak_rss_bytes": rss, "case_statistics": cases,
                  "case_timings_seconds": timings, "fit_seconds_by_size": fit_times,
                  "torch_imported": "torch" in sys.modules, "source_validation_seconds": validation_times,
                  "analysis_seconds": analysis, "validation_io": validation_io,
                  "semantic_validation": semantic, "metadata_validation": {
                      "bytes": metadata_path.stat().st_size, "seconds": metadata_seconds},
                  "validation_plan": resources.validation_plan()}
        report["validation_units"] = resources.validation_units(report)
        report.update(resources.geometry_projections(report))
        resources.validate_geometry(report)
        write_json(output/"report.json", report)
        if gate.common_binding() != common or "torch" in sys.modules:
            raise ValueError("profile source snapshot changed or Torch entered geometry process")
        seal_bundle(output, role=gate.PROFILE_ROLES["geometry"], binding={"common": common, "implementation_audit": audit},
                    resources={"seconds": _limit(started, geometry=True)[0], "peak_rss_bytes": rss})
        return p.reference(output/"manifest.json")
    except BaseException as exc:
        mark_failure(output, exc)
        raise


def mechanical_inputs(dim):
    """Oversized tensor fixture, NOT 64 actual partitions of 94 observed groups."""
    if dim not in (8, 9):
        raise ValueError("unknown head input dimension")
    x = np.sin(np.arange(94*9, dtype=np.float32)).reshape(94, 9)
    x[:, 8] = (x[:, 8]+1)/2
    return {"groups": x[:, :dim].copy(),
            "globals": np.cos(np.arange(64*6, dtype=np.float32)).reshape(64, 6),
            "incidence": np.full((64, 94), 1/94, np.float32)}


def timed_repeats(operation, count=3, *, operations_per_repeat=1):
    """Keep raw repetitions; callers declare the unit and payload size."""
    seconds = []
    for _ in range(count):
        before = time.monotonic()
        for _ in range(operations_per_repeat):
            operation()
        seconds.append(time.monotonic()-before)
    return seconds


def semantic_measurements(case, obs, truth, rows, matrices):
    """Exercise real payload readers on the two already declared profile cases.

    No authorization bypass, fabricated corpus or prospective draw. Target
    parsing shares the production pure loader; its truth is this fixed fixture.
    """
    from .learned_partition_runner import read_target_metrics
    from .learned_partition_metrics import candidate_targets
    from .source_artifacts import load_ordered_logits
    from .structured_source_data import validate_record
    from .structured_source_metrics import evaluate_scene
    case = Path(case)
    q = np.asarray(obs["log_f"], np.float32)
    def observation_features():
        value = json.loads((case/"observation.json").read_bytes())
        if set(value) != {"scene_id", "split_seed", "log_f"} or value != obs:
            raise ValueError("mechanical observation changed")
        with np.load(case/"features.npz", allow_pickle=False) as raw:
            features = {k: raw[k] for k in raw.files}
        validate_record(features, q)
    def semantic_repeats(operation):
        return timed_repeats(operation, operations_per_repeat=resources.SEMANTIC_OPERATIONS)
    result = {"n": len(q), "operations_per_repeat": resources.SEMANTIC_OPERATIONS,
              "observation_feature_seconds": semantic_repeats(observation_features),
              "observation_feature_bytes": sum((case/name).stat().st_size for name in ("observation.json", "features.npz")),
              "checkpoints": []}
    for i, (row, matrix) in enumerate(zip(rows, matrices)):
        pool_path, row_path = case/f"pool_{i}.json", case/f"rows_{i}.npz"
        pool = json.loads(pool_path.read_bytes())
        target_path, metric_path, logit_path = case/f"targets_{i}.npz", case/f"metrics_{i}.json", case/f"ordered_logits_{i}.npz"
        write_npz(target_path, **candidate_targets(pool, truth["source_ids"]))
        write_json(metric_path, evaluate_scene(pool, truth["source_ids"], matrix, (.55, .65, .6)[i]))
        import hashlib
        # NumPy-only serialization of the historical logits schema.
        write_npz(logit_path, logits=matrix.ravel(), sizes=np.array([len(q)], np.int64),
            offsets=np.array([0, len(q)**2], np.int64), scene_ids=np.array([obs["scene_id"]], np.int64),
            split_seeds=np.array([obs["split_seed"]], np.int64),
            observation_fingerprints=np.array([hashlib.sha256(q.astype("<f4").tobytes()).hexdigest()], dtype="U64"))
        def pool_rows():
            value = json.loads(pool_path.read_bytes())
            loaded = load_rows(row_path)
            if loaded.candidates != tuple(tuple(tuple(g) for g in c["signature"]) for c in value["candidates"]):
                raise ValueError("pool and cached rows differ")
        norm = fit_normalizer([row], expected_count=1)
        result["checkpoints"].append({"candidate_count": len(row.candidates), "group_count": len(row.groups),
            "pool_rows_seconds": semantic_repeats(pool_rows),
            "target_metrics_seconds": semantic_repeats(lambda: read_target_metrics(target_path, metric_path,
                n=row.n, candidates=row.candidates)),
            "logits_seconds": semantic_repeats(lambda: load_ordered_logits(logit_path, [obs])),
            "model_inputs_seconds": semantic_repeats(lambda: [model_inputs(row, norm, arm) for arm in
                ("pairs_structure", "local_compatibility", "shared_source", "decoupled_source")]),
            "pool_rows_bytes": pool_path.stat().st_size+row_path.stat().st_size,
            "target_metrics_bytes": target_path.stat().st_size+metric_path.stat().st_size,
            "logits_bytes": logit_path.stat().st_size})
    return result


def snapshot_and_calibration_measurements(folder, kernel, refs, row):
    """Eleven unique mechanical states and actual512-scene calibration I/O.

    Snapshot positions are mechanical updates, not fifty trained epochs. The
    calibration's declared epoch is an I/O schema fixture, never a candidate
    selection result. No observation or prospective labels enter this helper.
    """
    from .learned_partition_validation import validation_pass
    from .learned_partition_snapshots import read_snapshot
    from .learned_partition_campaign import save_calibration, calibration_record
    from .learned_partition_metrics import SEEDS
    if len(refs) != 11:
        raise ValueError("mechanical snapshot chain must contain eleven unique states")
    counts = []
    def check_chain():
        with validation_pass() as session:
            for ref in refs:
                read_snapshot(ref, expected_binding=kernel.binding)
            count = session.calls.get("src.atencion_armonica.learned_partition_snapshots._read_snapshot", 0)
            if count != 11:
                raise ValueError("snapshot measurement does not cover eleven unique loads")
            counts.append(count)
    chain_times = timed_repeats(check_chain)
    candidates = []
    for mask in range(64):
        cuts = sorted({0, 8, 16, 24, 32, *[i+1 for i in range(6) if mask & (1 << i)]})
        candidates.append(tuple(tuple(range(lo, hi)) for lo, hi in zip(cuts[:-1], cuts[1:])))
    candidates = tuple(sorted(candidates))
    data = {"inputs": [row]*512, "candidates": [candidates]*512,
            "ari": [np.linspace(0, 1, 64, dtype=np.float64)]*512}
    binding = {**kernel.binding, "arm": kernel.model.arm, "checkpoint_seed": SEEDS[0], "reader_seed": kernel.model.seed}
    before = time.monotonic()
    calibration = save_calibration(folder, 5, refs[-1], kernel.model, data, binding)
    write_seconds = time.monotonic()-before
    cal_times = timed_repeats(lambda: calibration_record(calibration, binding, data, snapshot=refs[-1]))
    paths = [p.ROOT/ref["path"] for ref in refs]
    return {"snapshot_chain_seconds": chain_times, "snapshot_unique_counts": counts,
            "snapshot_chain_bytes": sum(path.stat().st_size+(path.parent/"state.pt").stat().st_size for path in paths),
            "calibration_write_seconds": write_seconds, "calibration_read_seconds": cal_times,
            "calibration_scene_count": 512, "calibration_candidate_count": 64}


def validation_io_measurements(output):
    """Actual filesystem primitives over fixed bytes, not prospective scenes.

    The large blob measures hashing independent of compression. Packed input
    timing uses numerical tensors only; its compressed size is not extrapolated
    to real data. All generated bytes remain in the sealed mechanical profile.
    """
    from .learned_partition_inputs import pack_inputs, read_inputs
    output = Path(output)
    output.mkdir(exist_ok=False)
    small, large = output/"small.bin", output/"large.bin"
    with small.open("xb") as handle:
        handle.write(bytes(range(256)))
    with large.open("xb") as handle:
        block = bytes(range(256))*4096
        for _ in range(16):
            handle.write(block)
    result = {"hash_small": {"bytes": small.stat().st_size, "seconds": timed_repeats(lambda: sha_file(small))},
              "hash_large": {"bytes": large.stat().st_size, "seconds": timed_repeats(lambda: sha_file(large))}}
    bundle = output/"bundle"
    bundle.mkdir()
    for seed in range(3):
        folder = bundle/f"seed_{seed}"
        folder.mkdir()
        for index in range(1024):
            write_json(folder/f"{index:04d}.json", {"mechanical": index})
    write_json(bundle/"index.json", {"namespace": "MECHANICAL_BYTES_ONLY"})
    seal_bundle(bundle, role="mechanical_validation_io", binding={"no_observations": True}, resources={})
    reference = p.reference(bundle/"manifest.json")
    files = _inventory(bundle)
    result["inventory"] = {"entries": len(list(bundle.rglob("*"))), "files": len(files),
        "seconds": timed_repeats(lambda: _inventory(bundle))}
    result["bundle"] = {"entries": result["inventory"]["entries"], "files": len(files),
        "bytes": sum(path.stat().st_size for path in files.values()),
        "seconds": timed_repeats(lambda: verify_bundle(bundle, reference["sha256"], role="mechanical_validation_io"))}
    result["packed"] = {}
    for dim in (8, 9):
        path = output/f"packed_{dim}.npz"
        pack_inputs(path, [mechanical_inputs(dim)]*512, scene_ids=np.arange(512, dtype=np.int64), dim=dim)
        # Includes exact array payloads plus a per-member NPZ/header allowance.
        upper = resources.packed_upper_bytes(dim)
        result["packed"][str(dim)] = {"scene_count": 512, "dim": dim,
            "compressed_bytes": path.stat().st_size, "uncompressed_upper_bytes": upper,
            "seconds": timed_repeats(lambda: read_inputs(path, scene_ids=np.arange(512, dtype=np.int64), dim=dim))}
    resources.validate_validation_io(result)
    return result


def training_profile(output, *, audit, device, geometry=None, gpu_grant=None):
    started = time.monotonic()
    common = gate.common_binding()
    gate.verify_audit(audit, common, scope="FULL_IMPLEMENTATION")
    geometry_root, geometry_manifest = gate._bundle(geometry, gate.PROFILE_ROLES["geometry"], common)
    if geometry_manifest["binding"] != {"common": common, "implementation_audit": audit}:
        raise ValueError("training profile requires the same audited geometry profile")
    geometry_report = json.loads((geometry_root/"report.json").read_bytes())
    resources.validate_geometry(geometry_report)
    if device not in ("cpu", "cuda:0") or (device == "cpu" and gpu_grant is not None):
        raise ValueError("profile requires an explicit CPU or granted3090 device")
    from .structured_source_profile import gpu_lease
    lease = gpu_lease(gpu_grant) if device == "cuda:0" else nullcontext(None)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    try:
        with lease as availability:
            if device == "cuda:0":
                os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            import torch
            from .learned_partition_training import TrainingKernel, collate_inputs, collate_targets
            from .learned_partition_inference import predict_inputs
            from .learned_partition_snapshots import read_snapshot, write_snapshot
            torch.set_num_threads(1)
            torch.use_deterministic_algorithms(True)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            if device == "cuda:0":
                from .structured_source_runner import gpu_runtime, checkpoint_forward
                runtime = gpu_runtime()
            else:
                runtime = {"torch": torch.__version__, "numpy": np.__version__, "device": "cpu"}
            measurements = {}
            for arm, dim in (("pairs_structure", 8), ("shared_source", 9)):
                before = time.monotonic()
                kernel = TrainingKernel(arm, 2026090891, binding={"common": common, "namespace": "MECHANICAL_TENSOR_ONLY"}, device=device)
                setup = time.monotonic()-before
                folder = output/arm
                folder.mkdir()
                row = mechanical_inputs(dim)
                inputs = collate_inputs([row]*32)
                target = np.linspace(0, 1, 128, dtype=np.float32).reshape(64, 2)
                targets = collate_targets([target]*32, inputs["candidate_mask"])
                write_npz(folder/"inputs.npz", **row, target=target)
                before = time.monotonic()
                initial = write_snapshot(folder, "initial", kernel)
                read_snapshot(initial, expected_binding=kernel.binding)
                init_io = time.monotonic()-before
                snapshots = [initial]
                update_times = []
                for step in range(25):
                    before = time.monotonic()
                    ids = kernel.expected_scene_ids()
                    inputs = collate_inputs([row]*len(ids))
                    targets = collate_targets([target]*len(ids), inputs["candidate_mask"])
                    kernel.step(inputs, targets, ids)
                    if device == "cuda:0":
                        torch.cuda.synchronize()
                    if step >= 5:
                        update_times.append(time.monotonic()-before)
                    if step < 9:
                        snapshots.append(write_snapshot(folder, f"mechanical_step_{step+1}", kernel, parents=[snapshots[-1]]))
                    _limit(started)
                evaluation_times = []
                metric_times = []
                _, truth = mechanical_fixture(4)
                partition = [list(range(i, i+8)) for i in range(0, 32, 8)]
                for _ in range(5):
                    before = time.monotonic()
                    predictions = predict_inputs(kernel.model, [row]*32)
                    write_npz(folder/f"predictions_{len(evaluation_times)}.npz", components=np.stack(predictions))
                    evaluation_times.append(time.monotonic()-before)
                    before = time.monotonic()
                    for _ in range(32):
                        partition_metrics(partition, truth["source_ids"])
                        partition_errors(partition, truth["source_ids"])
                    metric_times.append(time.monotonic()-before)
                    _limit(started)
                before = time.monotonic()
                ref = write_snapshot(folder, "step_25", kernel, parents=[snapshots[-1]])
                state, _ = read_snapshot(ref, expected_binding=kernel.binding)
                kernel.restore(state)
                snapshot_io = time.monotonic()-before
                snapshots.append(ref)
                validation = snapshot_and_calibration_measurements(folder, kernel, snapshots, row)
                measurements[arm] = {"parameter_count": sum(v.numel() for v in kernel.model.parameters()),
                    "setup_seconds": setup, "initial_io_seconds": init_io, "update_seconds": update_times,
                    "evaluation_batch_io_seconds": evaluation_times, "snapshot_io_seconds": snapshot_io,
                    "metric_batch_seconds": metric_times, "validation": validation,
                    "linear_multiply_adds_per_batch": 32*(94*(dim*kernel.model.width+kernel.model.width*16)
                                                         +64*(94*16+22*32+32*2))}
                measurements[arm]["projected_cell_seconds"] = resources.head_projection(measurements[arm], geometry_report)
                del state, kernel
            observations, forward_times = [], []
            if device == "cuda:0":
                obs, _ = mechanical_fixture(4)
                observations.append(obs)
                record = feature_record(obs)
                for checkpoint in common["checkpoints"]:
                    before = time.monotonic()
                    matrices = checkpoint_forward(checkpoint, [record]*128, runtime)
                    torch.cuda.synchronize()
                    forward_times.append(time.monotonic()-before)
                    write_npz(output/f"frozen_logits_{checkpoint['seed']}.npz", logits=np.stack(matrices))
                    _limit(started)
                peak = torch.cuda.max_memory_reserved(0)
                if peak >= 2*1024**3:
                    raise RuntimeError("GPU profile exceeds2GiB reserved")
            seconds, rss = _limit(started)
            report = {"status": "MEASURED", "namespace": "MECHANICAL_NOT_PROSPECTIVE", "observations": observations,
                      "device": runtime["device"], "runtime": runtime, "availability": availability,
                      "seconds": seconds, "peak_rss_bytes": rss, "batch_size": 32,
                      "candidate_padding": 64, "group_padding": 94, "heads": measurements,
                      "projected_cell_seconds": max(v["projected_cell_seconds"] for v in measurements.values()),
                      "workload": resources.WORKLOAD,
                      "projected_campaign": resources.campaign_projection(measurements, geometry_report)}
            if device == "cuda:0":
                report.update(peak_reserved_bytes=peak, per_checkpoint_seconds=forward_times,
                              projected_forward_shard_seconds=2*(4*sum(forward_times)+seconds))
            resources.validate_training(report, geometry_report, gpu=device == "cuda:0")
        write_json(output/"report.json", report)
        if gate.common_binding() != common:
            raise ValueError("profile sources changed")
        seal_bundle(output, role=gate.PROFILE_ROLES["cpu" if device == "cpu" else "gpu"],
                    binding={"common": common, "implementation_audit": audit, "gpu_grant": gpu_grant,
                             "geometry": geometry},
                    resources={"seconds": _limit(started)[0], "peak_rss_bytes": rss})
        return p.reference(output/"manifest.json")
    except BaseException as exc:
        mark_failure(output, exc)
        raise
