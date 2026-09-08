"""Mechanical resource profiling; no prospective observations or training."""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import time

import numpy as np

from . import structured_source_gate as gate
from .partial_compatibility_cache import encoded, feature_record
from .shared_partial_data import mechanical_fixture
from .source_coherence import SourceFitter
from .structured_source_artifacts import mark_failure, seal_bundle, write_json, write_npz
from .structured_source_metrics import (SEEDS, bootstrap_indices, calibration_grid, read_selected,
                                        select_gammas, summarize_test)
from .structured_source_runner import checkpoint_forward, gpu_runtime, scene_payload


def verify_gpu_grant(ref):
    grant = gate.read_reference(ref)
    if (grant.get("status") != "AUTHORIZED" or grant.get("project") != "Phideus"
            or grant.get("device") != "NVIDIA GeForce RTX 3090"
            or not isinstance(grant.get("user_directive"), str) or not grant["user_directive"].strip()):
        raise PermissionError("explicit current user GPU grant receipt required")
    return grant


@contextmanager
def gpu_lease(grant_ref):
    """Local serialization + live conflict check; does not grant shared ownership.

    The coordinator must record the actual current user grant, communicate scope
    and check that no later conversation has revoked it before invoking this.
    """
    grant = verify_gpu_grant(grant_ref)
    lock_path = gate.ROOT/".agent-work/phideus-structured-reader-20260908/3090.lock"
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        info = subprocess.run(["nvidia-smi", "--id=0", "--query-gpu=name,uuid,memory.used,memory.total",
                               "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True, timeout=10)
        values = [v.strip() for v in info.stdout.strip().split(",")]
        if len(values) != 4 or values[0] != "NVIDIA GeForce RTX 3090":
            raise RuntimeError("visible device differs from the granted RTX 3090")
        if os.environ.get("CUDA_VISIBLE_DEVICES", "0") not in ("0", values[1]):
            raise RuntimeError("CUDA visibility does not select the granted GPU")
        processes = subprocess.run(["nvidia-smi", "--id=0", "--query-compute-apps=pid,used_gpu_memory",
                                    "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True, timeout=10)
        if processes.stdout.strip():
            raise RuntimeError("GPU has an existing compute process; do not occupy a conflicting window")
        if float(values[3])-float(values[2]) < 2048:
            raise RuntimeError("insufficient free GPU memory for the declared envelope")
        yield {"device": values[0], "uuid": values[1], "used_mib_before": float(values[2]),
               "total_mib": float(values[3]), "compute_processes_before": []}
        if verify_gpu_grant(grant_ref) != grant:
            raise PermissionError("GPU grant changed during the stage")


def _cpu_limit(started):
    seconds = time.monotonic()-started
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    if seconds > 120 or rss >= 1024**3:
        raise RuntimeError("CPU mechanical preflight exceeded 120s/1GiB")
    return seconds, rss


def cpu_preflight(output, *, audit):
    started = time.monotonic()
    common = gate.common_binding()
    gate.verify_audit(audit, common)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    try:
        fitter = SourceFitter()
        fit_times, cell_counts = {}, {}
        q = np.linspace(-1, 1, 32, dtype=np.float32)
        for size in range(3, 9):
            times = []
            for _ in range(3):
                t = time.monotonic()
                fitter.fit(q, list(range(size)))
                times.append(time.monotonic()-t)
                _cpu_limit(started)
            fit_times[str(size)] = times
            cell_counts[str(size)] = math.comb(8, size)*1025
        cases, timings, all_payloads = [], [], []
        for deformed in (False, True):
            obs, truth = mechanical_fixture(4, deformed=deformed)
            t = time.monotonic()
            features = feature_record(obs)
            feature_seconds = time.monotonic()-t
            q = np.asarray(obs["log_f"], np.float32)
            rng = np.random.default_rng(2026090791+int(deformed))
            a = rng.normal(size=(32, 32)).astype(np.float32)
            matrices = {SEEDS[0]: (4*features["pair_support"]-2).astype(np.float32),
                        SEEDS[1]: ((a+a.T)/2).astype(np.float32),
                        SEEDS[2]: (2-4*np.abs(q[:, None]-q[None, :])).astype(np.float32)}
            t = time.monotonic()
            payloads = scene_payload(obs, features, matrices, common["checkpoints"], truth["source_ids"], fitter=fitter)
            score_evaluation_seconds = time.monotonic()-t
            case = {"observation": obs, "sidecar": truth, "payloads": {str(s): p for s, p in payloads.items()},
                    "namespace": "MECHANICAL_NOT_PROSPECTIVE", "deformed": deformed}
            t = time.monotonic()
            serialized = encoded(case)
            with (output/f"fixture_{int(deformed)}.json").open("xb") as handle:
                handle.write(serialized)
            serialization_seconds = time.monotonic()-t
            sizes = {str(m): sum(sum(g["size"] == m for g in p["scored"]["groups"]) for p in payloads.values())
                     for m in range(1, 9)}
            cases.append({"n": 32, "candidate_counts": [len(p["scored"]["candidates"]) for p in payloads.values()],
                          "group_counts_by_size_across_pools": sizes,
                          "unique_evaluable_groups": len({tuple(g["members"]) for p in payloads.values()
                                                           for g in p["scored"]["groups"] if g["size"] >= 3}),
                          "cache_hits": sum(p["scored"]["fit_cache_hits"] for p in payloads.values()),
                          "cache_misses": sum(p["scored"]["fit_cache_misses"] for p in payloads.values()),
                          "serialized_bytes": len(serialized)})
            timings.append({"features": feature_seconds, "pool_fit_evaluation": score_evaluation_seconds,
                            "serialization": serialization_seconds})
            all_payloads.append((obs, payloads))
            _cpu_limit(started)
        obs, payloads = all_payloads[-1]
        # Mechanical values copied into a full-shaped metric fixture; no RNG
        # ever draws calibration/test observations here.
        t = time.monotonic()
        grids = []
        for i in range(256):
            for seed in SEEDS:
                grids.append(calibration_grid(**payloads[seed], observation={**obs, "scene_id": i, "split_seed": 2026090780}, seed=seed))
        selection = select_gammas(grids, split="calibration")
        selection_seconds = time.monotonic()-t
        gammas = {k: v["gamma"] for k, v in selection["factors"].items()}
        t = time.monotonic()
        rows = [{"scene_id": i, "seed": seed, "split_role": "iid", "split_seed": 2026090781,
                 **read_selected(**payloads[seed], gammas=gammas)} for i in range(256) for seed in SEEDS]
        readout_seconds = time.monotonic()-t
        t = time.monotonic()
        indices = bootstrap_indices()
        summary = summarize_test(rows, split="iid", indices=indices)
        bootstrap_seconds = time.monotonic()-t
        write_json(output/"mechanical_selection.json", selection)
        write_json(output/"mechanical_summary.json", summary)
        write_npz(output/"mechanical_bootstrap.npz", indices=indices)
        seconds, rss = _cpu_limit(started)
        # Each of six binary trees has at most 31 non-singleton groups;
        # duplicates/large-group pruning can only reduce this fit count.
        upper_groups = 6*31
        worst_fit = max(max(v) for v in fit_times.values())
        no_hit_fit_seconds = 256*upper_groups*worst_fit
        other_seconds = 256*max(sum(v.values()) for v in timings)+selection_seconds+readout_seconds+bootstrap_seconds
        projected = 2*(no_hit_fit_seconds+other_seconds)
        projected_rss = 2*rss+256*max(c["serialized_bytes"] for c in cases)
        report = {"status": "READY" if projected < 1200 and projected_rss < 2*1024**3 else "RESOURCE_REVIEW_REQUIRED",
                  "namespace": "MECHANICAL_NOT_PROSPECTIVE", "seconds": seconds, "peak_rss_bytes": rss,
                  "projected_cpu_phase_seconds": projected, "projected_peak_rss_bytes": projected_rss,
                  "projected_five_phase_cpu_seconds": 5*projected, "scene_count_per_phase": 256,
                  "case_statistics": cases, "case_timings_seconds": timings, "fit_seconds_by_size": fit_times,
                  "grid_cells_by_size": cell_counts, "maximum_unique_evaluable_groups_per_scene": upper_groups,
                  "projection_policy": "six_trees_no_hits_worst_size_time_plus_all_measured_work_times_two",
                  "selection_seconds": selection_seconds, "readout_seconds": readout_seconds,
                  "bootstrap_seconds": bootstrap_seconds, "preservation": "observations_truth_features_logits_pools_costs_witnesses_readouts_bootstrap"}
        write_json(output/"report.json", report)
        if gate.common_binding() != common:
            raise ValueError("preflight sources changed")
        seal_bundle(output, role="cpu_preflight", binding={"common": common, "implementation_audit": audit},
                    resources={"seconds": _cpu_limit(started)[0], "peak_rss_bytes": _cpu_limit(started)[1]})
        return gate.reference(output/"manifest.json")
    except BaseException as exc:
        mark_failure(output, exc)
        raise


def forward_profile(output, *, audit, gpu_grant):
    started = time.monotonic()
    common = gate.common_binding()
    gate.verify_audit(audit, common)
    # One fixed N32 fixture repeated to fill the historical batch size.
    observation, _ = mechanical_fixture(4)
    features = feature_record(observation)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    try:
        with gpu_lease(gpu_grant) as availability:
            runtime = gpu_runtime()
            import torch
            per_checkpoint = []
            for checkpoint in common["checkpoints"]:
                t = time.monotonic()
                raw = checkpoint_forward(checkpoint, [features]*128, runtime)
                torch.cuda.synchronize()
                per_checkpoint.append(time.monotonic()-t)
                write_npz(output/f"mechanical_logits_{checkpoint['seed']}.npz", logits=np.stack(raw))
            peak = torch.cuda.max_memory_reserved(0)
        total = time.monotonic()-started
        # Ten full batches/checkpoint for all1280scenes, includes model-load
        # cost ten times and a factor-two margin, plus startup again.
        projected = 2*(10*sum(per_checkpoint)+total)
        report = {"status": "READY" if projected < 600 and peak < 2*1024**3 else "RESOURCE_REVIEW_REQUIRED",
                  "namespace": "MECHANICAL_NOT_PROSPECTIVE", "device": runtime["device"],
                  "projected_forward_seconds": projected, "peak_reserved_bytes": peak,
                  "seconds": total, "per_checkpoint_seconds": per_checkpoint, "batch_size": 128,
                  "n": 32, "projected_scenes": 1280, "runtime": runtime, "availability": availability}
        write_json(output/"observation.json", observation)
        write_json(output/"report.json", report)
        if gate.common_binding() != common or time.monotonic()-started > 600:
            raise ValueError("profile sources changed or time envelope exceeded")
        seal_bundle(output, role="forward_profile", binding={"common": common, "implementation_audit": audit, "gpu_grant": gpu_grant},
                    resources={"seconds": time.monotonic()-started, "peak_reserved_bytes": peak})
        return gate.reference(output/"manifest.json")
    except BaseException as exc:
        mark_failure(output, exc)
        raise
