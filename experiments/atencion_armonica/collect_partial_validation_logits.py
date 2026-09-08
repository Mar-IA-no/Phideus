"""GPU forward of all frozen last-epoch models on validation, without selection.

The device is released before the separate CPU reader analysis. Test splits
have no route through this command. All fifteen raw outputs are retained.
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
from experiments.atencion_armonica.train_partial_compatibility_campaign import (
    APPROVED_PROFILE_SHA256, BUDGET_ROOT, CACHE_ROOT, PROFILE_ROOT, SOURCE_NAMES as TRAIN_SOURCES,
    verified_profile, write_new,
)
from src.atencion_armonica.partial_compatibility_budget import CampaignBudget, verify_inherited_lease
from src.atencion_armonica.partial_compatibility_cache import sha_file
from src.atencion_armonica.partial_compatibility_registry import training_registry

SOURCES = (*TRAIN_SOURCES, "experiments/atencion_armonica/collect_partial_validation_logits.py",
           "src/atencion_armonica/partial_compatibility_registry.py",
           "src/atencion_armonica/partial_compatibility_inference.py")


def source_hashes():
    return {name: sha_file(ROOT/name) for name in SOURCES}


def verify_checkpoint_bytes(cell):
    if sha_file(cell["checkpoint"]) != cell["checkpoint_sha256"]:
        raise ValueError("checkpoint bytes changed at inference time")


def verify(request):
    if source_hashes() != request["source_sha256"]:
        raise ValueError("validation forward source changed")
    for path_key, digest_key in (("authorization_path", "authorization_sha256"),
                                 ("training_manifest_path", "training_manifest_sha256")):
        if sha_file(Path(request[path_key])) != request[digest_key]:
            raise ValueError("validation forward authority changed")
    if (sha_file(CACHE_ROOT/"validation"/"manifest.json") != request["validation_manifest_sha256"]
            or sha_file(PROFILE_ROOT/"manifest.json") != request["profile_manifest_sha256"]
            or request["profile_manifest_sha256"] != APPROVED_PROFILE_SHA256):
        raise ValueError("validation data or profile changed")


def worker(output, deadline, lease_fd):
    verify_inherited_lease(BUDGET_ROOT, output, sha_file(output/"request.json"), lease_fd, deadline)
    request = json.loads((output/"request.json").read_text())
    verify(request)
    registry = training_registry(Path(request["training_manifest_path"]).parent)
    if registry["manifest_sha256"] != request["training_manifest_sha256"]:
        raise ValueError("training roster changed")
    for name, digest in registry["request"]["source_sha256"].items():
        if sha_file(ROOT/name) != digest:
            raise ValueError("training implementation changed before validation forward")

    from src.atencion_armonica.partial_compatibility_cache import ObservationCache
    from src.atencion_armonica.partial_compatibility_inference import collect_logits, save_logits
    from src.atencion_armonica.partial_compatibility_learning import ARMS
    from src.atencion_armonica.pairformer import build_model
    import numpy as np
    import torch

    started = time.monotonic()
    cache = ObservationCache(CACHE_ROOT/"validation", "validation")
    if not torch.cuda.is_available() or torch.cuda.get_device_name(0) != "NVIDIA GeForce RTX 3090":
        raise RuntimeError("validation forward requires the authorized RTX 3090")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    runtime = {"torch": torch.__version__, "numpy": np.__version__, "cuda": torch.version.cuda,
               "cudnn": torch.backends.cudnn.version(), "device": torch.cuda.get_device_name(0)}
    torch.cuda.reset_peak_memory_stats(0)
    rows = []
    for (arm, seed), cell in registry["cells"].items():
        if time.monotonic() >= deadline:
            raise TimeoutError("validation forward exhausted remaining GPU budget")
        verify_checkpoint_bytes(cell)
        state = torch.load(cell["checkpoint"], map_location="cpu", weights_only=False)
        if (state["binding"] != cell["binding"] or state["binding"]["runtime"] != runtime
                or state["steps"] != 3200 or state["next_epoch"] != 50 or state["next_batch"] != 0):
            raise ValueError("checkpoint is not the bound last-epoch state")
        model = build_model(ARMS[arm][0]).to("cuda:0")
        model.load_state_dict(state["model"])
        matrices = collect_logits(model, cache.records, device="cuda:0")
        verify_checkpoint_bytes(cell)
        name = f"{arm}__seed_{seed}.npz"
        save_logits(output/name, matrices, cache.observations)
        rows.append({"arm": arm, "seed": seed, "path": name, "sha256": sha_file(output/name),
                     "checkpoint_sha256": cell["checkpoint_sha256"], "scene_count": len(matrices)})
        del state, model, matrices
        torch.cuda.empty_cache()
    verify(request)
    if torch.cuda.max_memory_reserved(0) >= 8*1024**3:
        raise RuntimeError("validation memory exceeded reviewed resource envelope")
    write_new(output/"forward.json", {"status": "RAW_VALIDATION_COMPLETE", "rows": rows,
              "runtime": runtime, "seconds": time.monotonic()-started,
              "peak_reserved_bytes": torch.cuda.max_memory_reserved(0), "test_status": "CLOSED"})


def launch(output, training, authorization):
    if not authorization.is_file() or not authorization.read_text().strip():
        raise ValueError("GPU authorization receipt required")
    profile = verified_profile()
    registry = training_registry(training)
    output.mkdir(parents=True, exist_ok=False)
    budget = child = None
    try:
        request = {"source_sha256": source_hashes(), "authorization_path": str(authorization.resolve()),
                   "authorization_sha256": sha_file(authorization),
                   "training_manifest_path": str((training/"manifest.json").resolve()),
                   "training_manifest_sha256": registry["manifest_sha256"],
                   "validation_manifest_sha256": sha_file(CACHE_ROOT/"validation"/"manifest.json"),
                   "profile_manifest_sha256": APPROVED_PROFILE_SHA256, "test_status": "CLOSED"}
        write_new(output/"request.json", request)
        budget = CampaignBudget(BUDGET_ROOT, profile_manifest_sha256=APPROVED_PROFILE_SHA256,
                                profile_seconds=profile["seconds"])
        budget.reserve(output, sha_file(output/"request.json"))
        deadline = budget.execution_deadline()
        environment = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        with (output/"worker.log").open("xb") as log:
            child = subprocess.Popen([sys.executable, __file__, "--output", str(output.resolve()), "--worker",
                                      "--deadline", str(deadline), "--lease-fd", str(budget.lock.fileno())],
                                     env=environment, stdout=log, stderr=subprocess.STDOUT,
                                     pass_fds=(budget.lock.fileno(),), start_new_session=True)
            try:
                code = child.wait(timeout=max(0., deadline-time.monotonic()))
            except BaseException:
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
                raise
        if code:
            raise RuntimeError(f"validation GPU worker exit {code}")
        verify(request)
        forward = json.loads((output/"forward.json").read_text())
        if forward["status"] != "RAW_VALIDATION_COMPLETE" or len(forward["rows"]) != 15:
            raise ValueError("incomplete validation forward")
        write_new(output/"manifest.json", {"status": "RAW_VALIDATION_COMPLETE", "test_status": "CLOSED",
                  "request_sha256": sha_file(output/"request.json"), "forward_sha256": sha_file(output/"forward.json"),
                  "worker_log_sha256": sha_file(output/"worker.log"), "source_sha256": request["source_sha256"]})
        budget.settle("RAW_VALIDATION_COMPLETE")
    except BaseException as exc:
        if budget is not None and budget.started is not None and (child is None or child.poll() is not None):
            budget.settle("INCOMPLETE")
        write_new(output/"FAILURE.json", {"status": "INCOMPLETE", "error": repr(exc)})
        raise
    finally:
        if budget is not None:
            budget.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--training", type=Path)
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--deadline", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--lease-fd", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        if args.deadline is None or args.lease_fd is None:
            parser.error("worker requires inherited lease and deadline")
        worker(args.output, args.deadline, args.lease_fd)
    else:
        if args.training is None or args.authorization is None:
            parser.error("training and authorization are required")
        launch(args.output, args.training, args.authorization)
