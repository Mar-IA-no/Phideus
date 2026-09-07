"""Fixed 15-cell GPU training campaign. No validation selection or test access here.

The CPU parent freezes sources/receipt and holds the persistent study budget.
The worker loads only open caches, preserves each cell and never falls back to CPU.
Evaluation and its closed-test gate are separate prospective stages.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.atencion_armonica.partial_compatibility_budget import CampaignBudget, verify_inherited_lease
from src.atencion_armonica.partial_compatibility_cache import encoded, sha_file

SOURCE_NAMES = (
    "experiments/atencion_armonica/train_partial_compatibility_campaign.py",
    "experiments/atencion_armonica/PLAN_SHARED_PARTIAL_COMPATIBILITY.md",
    "src/atencion_armonica/__init__.py", "src/atencion_armonica/shared_partial_data.py",
    "src/atencion_armonica/partial_compatibility.py", "src/atencion_armonica/peak_tokens.py",
    "src/atencion_armonica/pairformer.py", "src/atencion_armonica/partial_compatibility_cache.py",
    "src/atencion_armonica/partial_compatibility_learning.py",
    "src/atencion_armonica/partial_compatibility_training.py",
    "src/atencion_armonica/partial_compatibility_budget.py",
    "experiments/atencion_armonica/profile_partial_compatibility_gpu.py",
    "experiments/atencion_armonica/test_partial_cached_learning.py",
)
CACHE_ROOT = ROOT/"data/atencion_armonica/shared_partial_cache_v1"
PROFILE_ROOT = ROOT/"data/atencion_armonica/shared_partial_gpu_profile_v1"
BUDGET_ROOT = ROOT/"data/atencion_armonica/shared_partial_study_v1"
APPROVED_PROFILE_SHA256 = "d7b9600be05c11e8fba293275c6025b3bed74de933694e0b38f0970a8fa10b08"
COMPATIBILITY = {
    "path": "Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/590_partial_cache_implementation_audit.md",
    "sha256": "b3fda0d4648e802c2f22cb66628774bbb62e01248e091af8ca092e80dfed7ef5",
    "changed_source": "src/atencion_armonica/partial_compatibility_learning.py",
    "profile_source": "f9f81990614b33c66b5526baa179e763947adf6f6f8d0e84c709ae1d892e8051",
    "current_source": "dfa9cdd624f277c99f16f20710875cd9ee765c3817ad05f9c0cf7cb0c20999f8",
    "parity_test": "experiments/atencion_armonica/test_partial_cached_learning.py",
    "parity_test_sha256": "a9bfeff993c72383b92f00016995d800422fe2892316d14f2a7607526b06f2d1",
}


def sources():
    return {name: sha_file(ROOT/name) for name in SOURCE_NAMES}


def write_new(path, value):
    with path.open("xb") as handle:
        handle.write(encoded(value))


def verify_request(request):
    if sources() != request["source_sha256"]:
        raise RuntimeError("frozen training source changed")
    if sha_file(Path(request["authorization_path"])) != request["authorization_sha256"]:
        raise RuntimeError("authorization receipt changed")
    for split, digest in request["cache_manifests"].items():
        if sha_file(CACHE_ROOT/split/"manifest.json") != digest:
            raise RuntimeError("open cache manifest changed")
    if (request["profile_manifest_sha256"] != APPROVED_PROFILE_SHA256
            or sha_file(PROFILE_ROOT/"manifest.json") != APPROVED_PROFILE_SHA256
            or request["profile_compatibility"] != COMPATIBILITY
            or sha_file(ROOT/COMPATIBILITY["path"]) != COMPATIBILITY["sha256"]
            or sha_file(ROOT/COMPATIBILITY["parity_test"]) != COMPATIBILITY["parity_test_sha256"]):
        raise RuntimeError("approved profile or its audited compatibility changed")


def verified_profile():
    if sha_file(PROFILE_ROOT/"manifest.json") != APPROVED_PROFILE_SHA256:
        raise ValueError("approved GPU profile manifest changed")
    profile = json.loads((PROFILE_ROOT/"manifest.json").read_text())
    if profile["status"] != "COMPLETE" or profile["scope"] != "GPU_profile_not_full_experiment":
        raise ValueError("GPU profile incomplete or wrong scope")
    for name, digest in profile["artifacts_sha256"].items():
        if Path(name).name != name or sha_file(PROFILE_ROOT/name) != digest:
            raise ValueError("GPU profile artifact hash mismatch")
    if (sha_file(ROOT/COMPATIBILITY["path"]) != COMPATIBILITY["sha256"]
            or sha_file(ROOT/COMPATIBILITY["parity_test"]) != COMPATIBILITY["parity_test_sha256"]):
        raise ValueError("cache parity audit/test changed")
    for name, digest in profile["source_sha256"].items():
        current = sha_file(ROOT/name)
        if current != digest and not (name == COMPATIBILITY["changed_source"]
                                      and digest == COMPATIBILITY["profile_source"]
                                      and current == COMPATIBILITY["current_source"]):
            raise ValueError("unreviewed source change since GPU profile")
    measured = json.loads((PROFILE_ROOT/"profile.json").read_text())
    if (measured["projection"]["status"] != "WITHIN_ESTIMATE"
            or measured["projection"]["peak_reserved_bytes"] >= 8*1024**3
            or measured["projection"]["total_compute_seconds"] > 24*3600):
        raise ValueError("resource profile requires review")
    return profile


def worker(output, deadline, lease_fd):
    verify_inherited_lease(BUDGET_ROOT, output, sha_file(output/"request.json"), lease_fd, deadline)
    request = json.loads((output/"request.json").read_text())
    verify_request(request)  # Before importing Torch or querying a device.
    from src.atencion_armonica.partial_compatibility_cache import ObservationCache, assert_disjoint
    from src.atencion_armonica.partial_compatibility_learning import ARMS
    from src.atencion_armonica.partial_compatibility_training import SEEDS, training_cell
    import torch

    started = time.monotonic()
    caches = {split: ObservationCache(CACHE_ROOT/split, split)
              for split in ("development", "train", "validation")}
    assert_disjoint(*caches.values())
    if not torch.cuda.is_available() or torch.cuda.get_device_name(0) != "NVIDIA GeForce RTX 3090":
        raise RuntimeError("this local campaign is authorized for the RTX 3090 only")
    torch.cuda.reset_peak_memory_stats(0)
    results = []
    initial_by_seed = {}
    for seed in SEEDS:
        for arm in ARMS:
            verify_request(request)
            cell_dir = output/"cells"/f"{arm}__seed_{seed}"
            result = training_cell(cell_dir, caches["train"], arm, seed, request["source_sha256"],
                                   remaining_seconds=deadline-time.monotonic())
            if torch.cuda.max_memory_reserved(0) >= 8*1024**3:
                raise RuntimeError("measured campaign memory exceeded the reviewed resource envelope")
            if ARMS[arm][0] == "B-local":
                digest = result["binding"]["initial_model_sha256"]
                if seed in initial_by_seed and initial_by_seed[seed] != digest:
                    raise RuntimeError("paired architectures did not share initialization")
                initial_by_seed[seed] = digest
            result["artifacts_sha256"] = {name: sha_file(cell_dir/name) for name in
                                         ("config.json", "curve.jsonl", "last_epoch.pt", "epoch_10.pt",
                                          "epoch_25.pt", "epoch_50.pt")}
            write_new(cell_dir/"result.json", result)
            results.append({"arm": arm, "seed": seed, "path": str(cell_dir.relative_to(output)),
                            "result_sha256": sha_file(cell_dir/"result.json"),
                            "last_epoch_sha256": result["last_epoch_sha256"]})
            torch.cuda.empty_cache()  # This worker's allocator only, between cells.
    verify_request(request)
    if len(results) != 15:
        raise RuntimeError("incomplete training roster")
    write_new(output/"training.json", {"status": "TRAINED_NOT_EVALUATED", "cells": results,
              "seconds": time.monotonic()-started,
              "peak_reserved_bytes": torch.cuda.max_memory_reserved(0),
              "peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
              "source_sha256": request["source_sha256"], "test_status": "CLOSED"})


def launch(output, authorization):
    if not authorization.is_file() or not authorization.read_text().strip():
        raise ValueError("explicit user GPU authorization receipt is required")
    profile = verified_profile()
    # The profile manifest itself binds the already audited source and artifacts.
    # No profile re-run or test generation occurs here.
    output.mkdir(parents=True, exist_ok=False)
    budget = None
    child = None
    try:
        request = {"source_sha256": sources(), "authorization_path": str(authorization.resolve()),
                   "authorization_sha256": sha_file(authorization),
                   "profile_manifest_sha256": APPROVED_PROFILE_SHA256,
                   "profile_compatibility": COMPATIBILITY,
                   "cache_manifests": {s: sha_file(CACHE_ROOT/s/"manifest.json")
                                       for s in ("development", "train", "validation")},
                   "scope": "15_trainings_no_test_access", "test_status": "CLOSED"}
        write_new(output/"request.json", request)
        budget = CampaignBudget(BUDGET_ROOT, profile_manifest_sha256=sha_file(PROFILE_ROOT/"manifest.json"),
                                profile_seconds=profile["seconds"])
        budget.reserve(output, sha_file(output/"request.json"))
        deadline = budget.execution_deadline()
        environment = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        with (output/"worker.log").open("xb") as log:
            child = subprocess.Popen([sys.executable, __file__, "--output", str(output.resolve()),
                                      "--worker", "--deadline", str(deadline),
                                      "--lease-fd", str(budget.lock.fileno())],
                                     env=environment, stdout=log, stderr=subprocess.STDOUT,
                                     pass_fds=(budget.lock.fileno(),), start_new_session=True)
            try:
                code = child.wait(timeout=max(0., deadline-time.monotonic()))
            except BaseException:
                # Exact owned child only; it does not spawn further jobs.
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
                raise
        if code:
            raise RuntimeError(f"GPU worker failed with exit {code}; inspect worker.log")
        verify_request(request)
        trained = json.loads((output/"training.json").read_text())
        if trained["status"] != "TRAINED_NOT_EVALUATED" or len(trained["cells"]) != 15:
            raise RuntimeError("worker result is not the full training roster")
        write_new(output/"manifest.json", {"status": "TRAINED_NOT_EVALUATED", "test_status": "CLOSED",
                  "request_sha256": sha_file(output/"request.json"),
                  "training_sha256": sha_file(output/"training.json"),
                  "worker_log_sha256": sha_file(output/"worker.log"),
                  "source_sha256": request["source_sha256"], "budget_path": str(budget.path),
                  "profile_manifest_sha256": APPROVED_PROFILE_SHA256,
                  "profile_compatibility": COMPATIBILITY})
        budget.settle("TRAINED_NOT_EVALUATED")
    except BaseException as exc:
        if budget is not None and budget.started is not None and (child is None or child.poll() is not None):
            budget.settle("INCOMPLETE")
        write_new(output/"FAILURE.json", {"status": "INCOMPLETE", "error": repr(exc), "test_status": "CLOSED"})
        raise
    finally:
        if budget is not None:
            budget.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--deadline", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--lease-fd", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        if args.deadline is None or args.lease_fd is None:
            parser.error("worker needs its inherited lease and deadline")
        worker(args.output, args.deadline, args.lease_fd)
    else:
        if args.authorization is None:
            parser.error("--authorization is required")
        launch(args.output, args.authorization)
