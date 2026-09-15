"""CPU continuation with explicit PRE/POST source inventories; no science changes."""
from __future__ import annotations

import time
LAUNCH_STARTED = time.monotonic()

import argparse
import fcntl
import json
import math
import os
from pathlib import Path

from experiments.atencion_armonica import run_geometric_decision_fresh as frozen_op

OPERATOR = "experiments/atencion_armonica/run_geometric_decision_postseal.py"
PLAN = "experiments/atencion_armonica/PLAN_GEOMETRIC_DECISION_POSTSEAL_SOURCE_GRAPH.md"
CONTRACT_PATH = "postseal/source-contract.json"
LATE_PATHS = (
    "src/atencion_armonica/generative_evidence_supervision.py",
    "src/atencion_armonica/observable_rival_evaluation.py",
)


def closing_origin(control, frozen):
    ref, _, start, manifest, output = frozen_op.completed_operation(control, "profile-closing")
    root = frozen_op.BASES[0]/"profiles/closing-cpu-0"
    forecast = frozen["budget"]["forecast"]
    if (ref != forecast["profile_finish"] or output["result"] != forecast["profile_report"]
            or start["stage"] != "profile" or start["binding"] != control.binding
            or manifest["root"] != str(root) or output["root"] != str(root)
            or output["manifest"] != start["manifest"]):
        raise ValueError("late-source origin differs from the frozen closing profile")
    profile = frozen_op.ReadOnlyStore(root, binding_ref=output["binding"])
    report = profile.json(output["result"])
    if (profile.binding != manifest["binding"] or profile.binding["protocol"] != frozen["protocol"]
            or report["binding"] != profile.binding or report["new_test_access"] is not False
            or report["new_observations"] != 0
            or report["schema"] != "geometric-decision-closing-profile-v1"):
        raise ValueError("late sources lack the original pre-test provenance")
    code = {r["path"]: r for r in profile.binding["code"]}
    original = {r["path"] for r in frozen["code"]}
    if (len(code) != len(profile.binding["code"]) or len(original) != len(frozen["code"])
            or not set(LATE_PATHS).issubset(code) or set(LATE_PATHS) & original):
        raise ValueError("late-source roster differs from the two declared modules")
    return {"finish": ref, "root": str(root), "binding": output["binding"], "result": output["result"]}, [code[p] for p in LATE_PATHS]


def contract_payload(control, freeze_ref):
    frozen = control.json(freeze_ref)
    origin, late = closing_origin(control, frozen)
    return {"schema": "geometric-decision-postseal-source-contract-v1", "test_freeze": freeze_ref,
        "protocol": frozen["protocol"], "original_code": frozen["code"], "closing_origin": origin,
        "late_sources": late,
        "execution_sources": [frozen_op.reference(p) for p in (OPERATOR, PLAN)]}


def validate_contract(control, ref, freeze_ref, *, phase=None):
    """File provenance for readers; exact loaded graph additionally for execution."""
    contract = control.json(ref)
    if contract != contract_payload(control, freeze_ref):
        raise ValueError("postseal execution contract or historical origin changed")
    records = [*contract["original_code"], *contract["late_sources"],
        *contract["execution_sources"], contract["protocol"]]
    for source in records:
        if frozen_op.reference(source["path"]) != source:
            raise ValueError("postseal source bytes changed")
    if contract["protocol"]["sha256"] != frozen_op.PROTOCOL_SHA:
        raise ValueError("postseal protocol differs from the scientific contract")
    if phase is not None:
        if phase not in ("PRE", "POST"):
            raise ValueError("unknown source-inventory phase")
        expected = sorted([*contract["original_code"], *(contract["late_sources"] if phase == "POST" else [])],
            key=lambda r: r["path"])
        if frozen_op.sources() != expected:
            raise ValueError("loaded source graph differs from exact "+phase+" inventory")
    return contract


def remaining_evaluation(control, frozen):
    last = sorted(control.path("attempts").glob("*/finish.json"))[-1]
    charged = control.json(control.reference(last))["charged_after"]["evaluation"]
    baseline = frozen["budget"]["charged_before_freeze"]["evaluation"]
    return min(frozen["budget"]["reservations"]["evaluation"]-(charged-baseline),
        frozen_op.STAGES["evaluation"]-charged)


def execute(control, mode, ctx, freeze_ref, frozen, contract_ref, *, started_at):
    if mode not in ("evaluate", "replay"):
        raise ValueError("postseal continuation supports only evaluate and replay")
    if control.path(f"manifests/{mode}.json").exists():
        raise ValueError("metric stage already attempted; preserve receipts")
    finish_ref, _, _, _, observed = frozen_op.completed_operation(control, "prospective-observables")
    recovery_finish = None
    if mode == "replay":
        frozen_op.completed_operation(control, "evaluate")
        recovery_manifest = control.publish_json("manifests/observable-replay.json", {
            "operation": "observable-replay", "test_freeze": freeze_ref,
            "observable_finish": finish_ref, "seal": observed["seal"]})
        recovered = frozen_op.bounded_operation(control, recovery_manifest, stage="fresh", started_at=started_at,
            verify=lambda: validate_contract(control, contract_ref, freeze_ref, phase="PRE"),
            reservation=math.ceil(frozen["budget"]["forecast"]["operation_seconds"]["observable-replay"]),
            operation=lambda check: frozen_op.observable_replay(control, finish_ref, freeze_ref, frozen, ctx,
                recovery_manifest, check=check))
        started_at = time.monotonic()
        recovery_finish = frozen_op.admitted_observable_replay(control, finish_ref, freeze_ref, observed["seal"])
        if recovery_finish != recovered["finish"]:
            raise ValueError("postseal recovery differs from its new COMPLETE receipt")
    output = frozen_op.ArtifactStore(frozen_op.BASES[0]/"evaluation",
        binding={"test_freeze": freeze_ref, "prediction_seal": observed["seal"]})
    manifest = control.publish_json(f"manifests/{mode}.json", {"operation": mode,
        "test_freeze": freeze_ref, "observable_finish": finish_ref, "root": str(output.root),
        "observable_replay_finish": recovery_finish, "execution_contract": contract_ref})
    phase = "PRE"
    def operation(check):
        nonlocal phase
        if mode == "replay" and frozen_op.admitted_observable_replay(
                control, finish_ref, freeze_ref, observed["seal"]) != recovery_finish:
            raise ValueError("observable recovery authority changed")
        complete = frozen_op.evaluate_fresh(control, finish_ref, freeze_ref, output, check=check, replay=mode == "replay")
        phase = "POST"
        return control.publish_json(f"outputs/{mode}.json", {"manifest": manifest, "root": str(output.root),
            "complete": complete, "test_freeze": freeze_ref, "execution_contract": contract_ref})
    return frozen_op.bounded_operation(control, manifest, stage="evaluation", started_at=started_at,
        verify=lambda: validate_contract(control, contract_ref, freeze_ref, phase=phase),
        reservation=remaining_evaluation(control, frozen), operation=operation)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("evaluate", "replay"), required=True)
    args = parser.parse_args()
    if (Path.cwd().resolve() != frozen_op.ROOT or os.environ.get("CUDA_VISIBLE_DEVICES") != ""
            or any(os.environ.get(k) != "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"))):
        raise ValueError("postseal continuation requires one-thread CPU-only project environment")
    frozen_op.torch.set_num_threads(1)
    frozen_op.torch.use_deterministic_algorithms(True)
    frozen_op.torch.backends.cuda.matmul.allow_tf32 = False
    frozen_op.torch.backends.cudnn.allow_tf32 = False
    view = frozen_op.ReadOnlyStore(frozen_op.BASES[0]/"control", binding_ref=frozen_op.CONTROL_BINDING)
    control = frozen_op.ArtifactStore(view.root, binding=view.binding)
    with control.path("operator.lock").open("ab") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ctx, inputs, _ = frozen_op.admit_launch(control, args.stage, started_at=LAUNCH_STARTED)
        started_at = time.monotonic()
        if not control.path("freezes/prospective.json").exists():
            raise ValueError("postseal continuation cannot create a prospective freeze")
        freeze_ref, frozen = frozen_op.prepare_freeze(control, inputs, ctx)
        contract_ref = control.publish_json(CONTRACT_PATH, contract_payload(control, freeze_ref))
        result = execute(control, args.stage, ctx, freeze_ref, frozen, contract_ref, started_at=started_at)
        print(json.dumps({"stage": args.stage, "execution_contract": contract_ref, **result}), flush=True)


if __name__ == "__main__":
    main()
