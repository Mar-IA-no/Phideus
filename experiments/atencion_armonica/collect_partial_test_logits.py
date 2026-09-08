"""Frozen held-out GPU inference: all five splits and all fifteen last-epoch models."""
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
    APPROVED_PROFILE_SHA256, BUDGET_ROOT, CACHE_ROOT, verified_profile, write_new,
)
from experiments.atencion_armonica.collect_partial_validation_logits import verify_checkpoint_bytes
from src.atencion_armonica.partial_compatibility_budget import CampaignBudget, verify_inherited_lease
from src.atencion_armonica.partial_compatibility_cache import sha_file
from src.atencion_armonica.partial_compatibility_registry import training_registry
from src.atencion_armonica.partial_compatibility_test_cache import FrozenTestCache, verify_disjoint_manifests
from src.atencion_armonica.partial_compatibility_test_gate import TEST_SPLITS, verify_freeze

TEST_CACHE_ROOT = ROOT/"data/atencion_armonica/shared_partial_test_cache_v1"


def verify_request(request):
    freeze_path = Path(request["freeze_path"])
    if (sha_file(freeze_path) != request["freeze_sha256"]
            or sha_file(Path(request["authorization_path"])) != request["authorization_sha256"]):
        raise ValueError("test forward freeze/authority changed")
    frozen = verify_freeze(freeze_path)
    if request["source_sha256"] != frozen["source_sha256"]:
        raise ValueError("test forward source mismatch")
    for split in TEST_SPLITS:
        if sha_file(TEST_CACHE_ROOT/split/"manifest.json") != request["test_manifest_sha256"][split]:
            raise ValueError("test cache manifest changed")
    return frozen


def worker(output, deadline, lease_fd):
    verify_inherited_lease(BUDGET_ROOT, output, sha_file(output/"request.json"), lease_fd, deadline)
    request = json.loads((output/"request.json").read_text())
    freeze = verify_request(request)
    freeze_path = Path(request["freeze_path"])
    training = Path(freeze["bindings"]["training"]["path"]).parent
    registry = training_registry(training)
    for split, digest in registry["request"]["cache_manifests"].items():
        if sha_file(CACHE_ROOT/split/"manifest.json") != digest:
            raise ValueError("open split manifest changed before cross-split identity check")
    verify_disjoint_manifests([CACHE_ROOT/s for s in ("development", "train", "validation")]
                             + [TEST_CACHE_ROOT/s for s in TEST_SPLITS])
    from src.atencion_armonica.partial_compatibility_inference import collect_logits, save_logits
    from src.atencion_armonica.partial_compatibility_learning import ARMS
    from src.atencion_armonica.pairformer import build_model
    import numpy as np
    import torch

    if not torch.cuda.is_available() or torch.cuda.get_device_name(0) != "NVIDIA GeForce RTX 3090":
        raise RuntimeError("test inference requires the authorized RTX 3090")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    runtime = {"torch": torch.__version__, "numpy": np.__version__, "cuda": torch.version.cuda,
               "cudnn": torch.backends.cudnn.version(), "device": torch.cuda.get_device_name(0)}
    torch.cuda.reset_peak_memory_stats(0)
    started, rows = time.monotonic(), []
    for split in TEST_SPLITS:
        cache = FrozenTestCache(TEST_CACHE_ROOT/split, split, freeze_path)
        (output/split).mkdir()
        for (arm, seed), cell in registry["cells"].items():
            if time.monotonic() >= deadline:
                raise TimeoutError("test inference exhausted remaining GPU budget")
            verify_checkpoint_bytes(cell)
            state = torch.load(cell["checkpoint"], map_location="cpu", weights_only=False)
            if (state["binding"] != cell["binding"] or state["binding"]["runtime"] != runtime
                    or state["steps"] != 3200 or state["next_epoch"] != 50 or state["next_batch"] != 0):
                raise ValueError("test checkpoint is not the frozen last epoch")
            model = build_model(ARMS[arm][0]).to("cuda:0")
            model.load_state_dict(state["model"])
            logits = collect_logits(model, cache.records, device="cuda:0")
            verify_checkpoint_bytes(cell)
            name = f"{split}/{arm}__seed_{seed}.npz"
            save_logits(output/name, logits, cache.observations)
            rows.append({"split": split, "arm": arm, "seed": seed, "path": name,
                         "sha256": sha_file(output/name), "checkpoint_sha256": cell["checkpoint_sha256"],
                         "scene_count": len(logits)})
            del state, model, logits
            torch.cuda.empty_cache()
        verify_request(request)
        print(json.dumps({"split": split, "status": "RAW_SPLIT_COMPLETE", "seconds": time.monotonic()-started}), flush=True)
    if torch.cuda.max_memory_reserved(0) >= 8*1024**3:
        raise RuntimeError("test inference memory exceeded resource envelope")
    write_new(output/"forward.json", {"status": "RAW_TEST_COMPLETE", "rows": rows, "runtime": runtime,
              "seconds": time.monotonic()-started, "peak_reserved_bytes": torch.cuda.max_memory_reserved(0),
              "freeze_sha256": request["freeze_sha256"]})


def launch(output, freeze_path, authorization):
    if not authorization.is_file() or not authorization.read_text().strip():
        raise ValueError("explicit GPU authorization receipt required")
    frozen = verify_freeze(freeze_path)
    profile = verified_profile()
    output.mkdir(parents=True, exist_ok=False)
    budget = child = None
    try:
        request = {"freeze_path": str(freeze_path.resolve()), "freeze_sha256": sha_file(freeze_path),
                   "authorization_path": str(authorization.resolve()), "authorization_sha256": sha_file(authorization),
                   "source_sha256": frozen["source_sha256"],
                   "test_manifest_sha256": {s: sha_file(TEST_CACHE_ROOT/s/"manifest.json") for s in TEST_SPLITS}}
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
            raise RuntimeError(f"test GPU worker exit {code}")
        verify_request(request)
        forward = json.loads((output/"forward.json").read_text())
        expected = {(s, r["arm"], r["seed"]) for s in TEST_SPLITS for r in frozen["readers"] if r["seed"] is not None}
        if (forward["status"] != "RAW_TEST_COMPLETE" or len(forward["rows"]) != 75
                or {(r["split"], r["arm"], r["seed"]) for r in forward["rows"]} != expected):
            raise ValueError("test forward roster incomplete")
        write_new(output/"manifest.json", {"status": "RAW_TEST_COMPLETE", "freeze_sha256": sha_file(freeze_path),
                  "request_sha256": sha_file(output/"request.json"), "forward_sha256": sha_file(output/"forward.json"),
                  "worker_log_sha256": sha_file(output/"worker.log"), "source_sha256": request["source_sha256"]})
        budget.settle("RAW_TEST_COMPLETE")
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--freeze", type=Path)
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
        if args.freeze is None or args.authorization is None:
            parser.error("freeze and authorization required")
        launch(args.output, args.freeze, args.authorization)
