"""OPEN-only calibration selection after an authenticated training completion."""
from __future__ import annotations

import time
LAUNCH_STARTED = time.monotonic()

import fcntl
import argparse
import json
import math
import os
from pathlib import Path
import platform
import signal

import numpy as np
import torch

from experiments.atencion_armonica.profile_geometric_decision import admitted_open
from experiments.atencion_armonica.prepare_geometric_decision_open import ROOT, PROTOCOL, PROTOCOL_SHA, reference, code_snapshot
from src.atencion_armonica.geometric_decision_campaign import ROSTER
from src.atencion_armonica.geometric_decision_campaign_selection import ReadOnlyCellStore, select_campaign
from src.atencion_armonica.geometric_decision_selection_profile import profile_selection, admitted_selection_profile
from src.atencion_armonica.geometric_decision_budget import StageBudget, BudgetExceeded, STAGES
from src.atencion_armonica.geometric_decision_store import ArtifactStore, BASES
from src.atencion_armonica.partial_compatibility_cache import encoded
import hashlib


def admitted_training(control, *, root):
    candidates = []
    for path in sorted(control.path("attempts").glob("*/finish.json")):
        ref = control.reference(path)
        finish = control.json(ref)
        start = control.json(finish["start"])
        if start["stage"] != "training" or finish["status"] != "COMPLETE":
            continue
        manifest = control.json(start["manifest"])
        output = control.json(finish["completion"])
        if (start["binding"] != control.binding or manifest["operation"] != "training"
                or manifest["root"] != str(root) or output["root"] != str(root)
                or output["manifest"] != start["manifest"]):
            raise ValueError("training completion provenance differs")
        candidates.append((ref, output, manifest))
    if not candidates:
        raise ValueError("no COMPLETE training operator; selection must wait")
    # A replay of an already-complete campaign may authorize the same output,
    # but competing roots/outputs cannot be collapsed into one authority.
    if any(row[1:] != candidates[0][1:] for row in candidates):
        raise ValueError("training attempts authorize different campaigns")
    finish_ref, output, manifest = candidates[0]
    raw = encoded(manifest["campaign_binding"])
    campaign = ReadOnlyCellStore(root, binding_ref={"path": "binding.json", "bytes": len(raw),
                                                   "sha256": hashlib.sha256(raw).hexdigest()})
    complete = campaign.json(output["complete"])
    if (complete["binding"] != campaign.binding or complete["schema"] != "geometric-decision-campaign-complete-v1"
            or [(c["checkpoint_seed"], c["arm"], c["reader_seed"]) for c in complete["cells"]] != list(ROSTER)):
        raise ValueError("training completion does not contain the exact 72 cells")
    return campaign, output["complete"], {"finish": finish_ref, "output": output}


def sources():
    return sorted([*code_snapshot(), reference("experiments/atencion_armonica/select_geometric_decision.py"),
                   reference("experiments/atencion_armonica/PLAN_GEOMETRIC_DECISION_SELECTION.md"),
                   reference("experiments/atencion_armonica/profile_geometric_decision.py")], key=lambda r: r["path"])


def bounded_operation(control, manifest_ref, *, stage, started_at, verify, reservation, operation):
    """Caller holds the common lock; no CUDA or separate budget ledger."""
    budget = StageBudget(control, stage, manifest_ref=manifest_ref, reservation_seconds=reservation,
        started_at=started_at, prior_charges=control.binding["prior_charges"],
        output_roots=[Path(p) for p in control.binding["output_roots"]])
    def stop(signum, frame):
        if signum == signal.SIGALRM:
            raise BudgetExceeded("selection reservation expired")
        raise InterruptedError("selection paused; immutable outputs retained")
    old = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM)}
    try:
        for sig in old:
            signal.signal(sig, stop)
        signal.setitimer(signal.ITIMER_REAL, max(.001, reservation-(time.monotonic()-started_at)))
        verify()
        result_ref = operation(budget.check)
        verify()
        finish = budget.finish("COMPLETE", completion=result_ref)
        return {"output": result_ref, "finish": finish}
    except BaseException as exc:
        if not budget.closed:
            budget.finish("LIMIT_REACHED" if isinstance(exc, BudgetExceeded) else "PAUSED"
                          if isinstance(exc, (InterruptedError, KeyboardInterrupt)) else "FAILED")
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.)
        for sig, handler in old.items():
            signal.signal(sig, handler)


def execute(preparation, open_ref, campaign, campaign_ref, output, control, manifest_ref, *, started_at,
            verify, reservation):
    if control.json(manifest_ref) != {"operation": "calibration-selection", "binding": output.binding, "root": str(output.root)}:
        raise ValueError("selection manifest differs")
    def operation(check):
        result = select_campaign(preparation, open_ref, campaign, campaign_ref, output, check=check)
        return control.publish_json("outputs/selection.json", {"manifest": manifest_ref,
            "root": str(output.root), "selection": result})
    return bounded_operation(control, manifest_ref, stage="evaluation", started_at=started_at,
                             verify=verify, reservation=reservation, operation=operation)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", action="store_true", help="arithmetic CPU profile only, no scientific selection")
    args = parser.parse_args()
    if (Path.cwd().resolve() != ROOT or os.environ.get("CUDA_VISIBLE_DEVICES") != ""
            or any(os.environ.get(k) != "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"))):
        raise RuntimeError("selection requires project cwd, one CPU thread and hidden CUDA")
    torch.set_num_threads(1)
    control, preparation, open_ref = admitted_open()
    with control.path("operator.lock").open("ab") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        protocol, code = reference(PROTOCOL), sources()
        if protocol["sha256"] != PROTOCOL_SHA:
            raise ValueError("frozen scientific protocol differs")
        runtime = {"python": platform.python_version(), "numpy": np.__version__, "torch": str(torch.__version__),
                   "device": "cpu", "threads": torch.get_num_threads()}
        def verify():
            if sources() != code or reference(PROTOCOL) != protocol:
                raise ValueError("selection code/protocol changed")
        profile_root = BASES[0]/"profiles/selection-cpu"
        if args.profile:
            if profile_root.exists():
                raise ValueError("selection profile already exists; no silent repetition")
            profile = ArtifactStore(profile_root, binding={"schema": "geometric-decision-selection-profile-binding-v1",
                "protocol": protocol, "code": code, "runtime": runtime})
            manifest = control.publish_json("manifests/profile-selection.json", {"operation": "profile-selection",
                "binding": profile.binding, "root": str(profile.root)})
            def operation(check):
                result = profile_selection(profile, check=check)
                return control.publish_json("outputs/profile-selection.json", {"manifest": manifest,
                    "root": str(profile.root), "binding": profile.reference(profile.path("binding.json")), "profile": result})
            result = bounded_operation(control, manifest, stage="profile", started_at=LAUNCH_STARTED,
                verify=verify, reservation=120., operation=operation)
            print(json.dumps({"status": "SELECTION_CPU_PROFILE_COMPLETE", **result}), flush=True)
            return
        campaign, campaign_ref, training = admitted_training(control, root=BASES[0]/"training")
        load = campaign.binding["admission"]["input_provenance"]["checkpoint_load"]
        admission = admitted_selection_profile(control, root=profile_root, corpus_load_seconds=load["seconds"])
        if (admission["binding"]["code"] != code or admission["binding"]["runtime"] != runtime
                or admission["binding"]["protocol"] != protocol):
            raise ValueError("selection runtime/code/protocol differs from measured profile")
        projected = admission["forecast"]["projected_seconds"]
        if projected > STAGES["evaluation"]:
            raise BudgetExceeded("selection projection plus 25% exceeds evaluation budget")
        binding = {"schema": "geometric-decision-selection-binding-v1", "campaign": campaign_ref,
            "campaign_binding": campaign.binding, "training": training, "protocol": protocol, "code": code,
            "runtime": runtime, "admission": admission}
        output = ArtifactStore(BASES[0]/"selection", binding=binding)
        manifest = control.publish_json("manifests/selection.json", {"operation": "calibration-selection", "binding": binding, "root": str(output.root)})
        result = execute(preparation, open_ref, campaign, campaign_ref, output, control, manifest,
                         started_at=LAUNCH_STARTED, verify=verify, reservation=max(300., math.ceil(projected)))
        print(json.dumps({"status": "EPOCHS_SELECTED_FRESH_NOT_OPENED", **result}), flush=True)


if __name__ == "__main__":
    main()
