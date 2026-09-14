"""Execute the complete fixed campaign, with an immutable training freeze."""
from __future__ import annotations

import time
LAUNCH_STARTED = time.monotonic()

import fcntl
import json
import math
import os
from pathlib import Path
import platform
import signal
import subprocess

import numpy as np
import torch

from experiments.atencion_armonica.profile_geometric_decision import admitted_open, admitted_profile_inputs, OPEN_FINISH
from experiments.atencion_armonica.prepare_geometric_decision_open import ROOT, PROTOCOL, PROTOCOL_SHA, reference, code_snapshot
from src.atencion_armonica.generative_evidence_profile import gpu_availability
from src.atencion_armonica.geometric_decision_store import ArtifactStore, BASES
from src.atencion_armonica.geometric_decision_admission import head_admission
from src.atencion_armonica.geometric_decision_campaign import run_campaign, ROSTER
from src.atencion_armonica.geometric_decision_budget import StageBudget, BudgetExceeded, STAGES


def sources():
    return sorted([*code_snapshot(), *[reference(p) for p in (
        "experiments/atencion_armonica/train_geometric_decision.py",
        "experiments/atencion_armonica/profile_geometric_decision.py",
        "experiments/atencion_armonica/PLAN_GEOMETRIC_DECISION_PROFILES.md",
        "experiments/atencion_armonica/PLAN_GEOMETRIC_DECISION_CAMPAIGN.md")]], key=lambda r: r["path"])


def runtime_for(device):
    return {"torch": str(torch.__version__), "numpy": np.__version__, "python": platform.python_version(),
        "device": device, "threads": torch.get_num_threads(), "deterministic": torch.are_deterministic_algorithms_enabled(),
        "tf32": torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32,
        "cuda_build": torch.version.cuda, "cudnn_version": torch.backends.cudnn.version() if device == "cuda:0" else None,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "driver": subprocess.check_output(["nvidia-smi", "--query-gpu=uuid,driver_version", "--format=csv,noheader"],
                   text=True, timeout=5) if device == "cuda:0" else None}


def training_charged(control):
    """Reservation sizing only; StageBudget independently validates the full ledger."""
    charged = 0.
    for path in sorted(control.path("attempts").glob("*/start.json")):
        start = control.json(control.reference(path))
        if start["stage"] != "training":
            continue
        finish = path.parent/"finish.json"
        charged += control.json(control.reference(finish))["seconds"] if finish.exists() else start["reservation_seconds"]
    return charged


def execute(preparation, complete_ref, store, control, manifest_ref, *, started_at, reservation, verify, progress=print,
            availability=None):
    """Caller holds operator.lock; reusable fixtures cannot alter the roster."""
    device = store.binding["device"]
    if device == "cuda:0" and not isinstance(availability, dict):
        raise ValueError("CUDA attempt requires its own availability receipt before initialization")
    manifest = control.json(manifest_ref)
    if manifest != {"operation": "training", "campaign_binding": store.binding, "root": str(store.root)}:
        raise ValueError("training manifest does not freeze this exact campaign")
    def vram():
        return torch.cuda.max_memory_reserved(0) if device == "cuda:0" and torch.cuda.is_initialized() else 0
    budget = StageBudget(control, "training", manifest_ref=manifest_ref, reservation_seconds=reservation,
        started_at=started_at, prior_charges=control.binding["prior_charges"],
        output_roots=[Path(p) for p in control.binding["output_roots"]], vram=vram)
    def stop(signum, frame):
        if signum == signal.SIGALRM:
            raise BudgetExceeded("training attempt deadline reached; completed snapshots preserved")
        raise InterruptedError("training paused; complete update boundaries are retained")
    old = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM)}
    try:
        for sig in old:
            signal.signal(sig, stop)
        signal.setitimer(signal.ITIMER_REAL, max(.001, reservation-(time.monotonic()-started_at)))
        control.publish_json(budget.folder+"/owner.json", {"pid": os.getpid(), "workspace": str(ROOT),
            "purpose": "geometric-decision-72-cell-training", "manifest": manifest_ref,
            "device": device, "runtime": store.binding.get("runtime"), "availability": availability})
        verify()
        if device == "cuda:0":
            torch.cuda.set_device(0)
            torch.cuda.set_per_process_memory_fraction(.25, 0)
            torch.cuda.reset_peak_memory_stats(0)
        result = run_campaign(preparation, complete_ref, store, device=device,
            attempt=budget.folder, check=budget.check, progress=progress)
        verify()
        budget.check(force_resources=True)
        output = control.publish_json("outputs/training.json", {"manifest": manifest_ref,
            "root": str(store.root), "complete": result})
        finish = budget.finish("COMPLETE", completion=output)
        return {"output": output, "finish": finish}
    except BaseException as exc:
        if not budget.closed:
            status = "LIMIT_REACHED" if isinstance(exc, BudgetExceeded) else "PAUSED" if isinstance(exc, (InterruptedError, KeyboardInterrupt)) else "FAILED"
            budget.finish(status)
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.)
        for sig, handler in old.items():
            signal.signal(sig, handler)


def main():
    if (Path.cwd().resolve() != ROOT or any(os.environ.get(k) != "1" for k in
                                          ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"))):
        raise RuntimeError("project cwd and one-thread environment required")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    control, preparation, complete_ref = admitted_open()
    with control.path("operator.lock").open("ab") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _, input_provenance = admitted_profile_inputs(control)
        admission = head_admission(control, input_provenance, profile_root=BASES[0]/"profiles")
        device = admission["device"]
        if ((device == "cpu" and os.environ.get("CUDA_VISIBLE_DEVICES") != "") or
                (device == "cuda:0" and os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8")):
            raise RuntimeError("environment differs from the measured preferred backend")
        availability = gpu_availability() if device == "cuda:0" else None
        runtime = runtime_for(device)
        if runtime != {k: v for k, v in admission["profile_runtime"].items() if k != "availability"}:
            raise ValueError("training runtime differs from the selected exact-recovery profile")
        protocol, code = reference(PROTOCOL), sources()
        if protocol["sha256"] != PROTOCOL_SHA:
            raise ValueError("scientific protocol changed")
        if admission["projected_seconds"] > STAGES["training"]:
            raise BudgetExceeded("full roster forecast plus 25% exceeds training budget")
        binding = {"schema": "geometric-decision-campaign-v1", "preparation": preparation.store.binding,
            "open_complete": complete_ref, "open_finish": OPEN_FINISH, "protocol": protocol,
            "code": code, "runtime": runtime, "device": device, "admission": admission,
            "roster": [list(v) for v in ROSTER], "selection": "common epoch per arm; frozen protocol; not performed here"}
        store = ArtifactStore(BASES[0]/"training", binding=binding)
        manifest = control.publish_json("manifests/training.json",
            {"operation": "training", "campaign_binding": binding, "root": str(store.root)})
        remaining = STAGES["training"]-training_charged(control)
        if remaining <= 0:
            raise BudgetExceeded("no training budget remains; no silent reset")
        # Reserve one full projected campaign or the remaining fixed budget on
        # recovery; the cumulative supervisor remains the authoritative limit.
        reservation = min(remaining, max(1800., math.ceil(admission["projected_seconds"])))
        def verify():
            if sources() != code or reference(PROTOCOL) != protocol or runtime_for(device) != runtime:
                raise ValueError("frozen training code, protocol or runtime changed")
        print(json.dumps({"status": "STARTING_FIXED_CAMPAIGN", "device": device, "cells": 72,
            "reservation_seconds": reservation, "availability": availability}), flush=True)
        result = execute(preparation, complete_ref, store, control, manifest, started_at=LAUNCH_STARTED,
                         reservation=reservation, verify=verify, availability=availability,
                         progress=lambda row: print(json.dumps(row), flush=True))
        print(json.dumps({"status": "TRAINING_COMPLETE_SELECTION_PENDING", **result}), flush=True)


if __name__ == "__main__":
    main()
