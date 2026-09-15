"""Finite prospective supervisor: fresh -> sealed evaluation -> exact replay.

All stages require the completed closing-cost profile. No training, selection,
profile retry or alternate seeds are available through this entry point.
"""
from __future__ import annotations

import time
LAUNCH_STARTED = time.monotonic()

import argparse
from copy import deepcopy
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import shutil

import torch

from experiments.atencion_armonica.prepare_geometric_decision_open import ROOT, PROTOCOL, PROTOCOL_SHA, reference, code_snapshot
from experiments.atencion_armonica.profile_geometric_decision import admitted_open, CONTROL_BINDING
from experiments.atencion_armonica.select_geometric_decision import admitted_training
from experiments.atencion_armonica.profile_geometric_decision_observed import admitted_archive
from src.atencion_armonica.geometric_decision_remaining_cost import projection as closing_projection, allocate_recovery
from src.atencion_armonica.geometric_decision_archive_admission import admitted_selection, verify_heads
from src.atencion_armonica.geometric_decision_observed_admission import admit_profile
from src.atencion_armonica.geometric_decision_budget import STAGES, LIMITS, StageBudget, BudgetExceeded, owned_bytes
from src.atencion_armonica.geometric_decision_store import ArtifactStore, BASES
from src.atencion_armonica.geometric_decision_open import ReadOnlyStore
from src.atencion_armonica.geometric_decision_observed_run import run_observed
from src.atencion_armonica.geometric_decision_release import TEST_ROSTER, seal_outputs, admitted_seal
from src.atencion_armonica.geometric_decision_evaluation import evaluate_fresh
from src.atencion_armonica.geometric_decision_draws import DrawBatch, TESTS
from src.atencion_armonica.generative_evidence_reuse import OpenReuse, VerifiedBytes
from src.atencion_armonica.generative_evidence_profile import gpu_availability
from src.atencion_armonica.structured_source_runner import gpu_runtime, checkpoint_forward
from src.atencion_armonica.learned_partition_data import _draw_scene
from src.atencion_armonica import generative_evidence as ge

OPERATOR = "experiments/atencion_armonica/run_geometric_decision_fresh.py"
AMENDMENT = "experiments/atencion_armonica/AMENDMENT_GEOMETRIC_DECISION_RECOVERY_ACCOUNTING.md"


def sources():
    extra = (OPERATOR, "experiments/atencion_armonica/PLAN_GEOMETRIC_DECISION_RELEASE.md",
        "experiments/atencion_armonica/PLAN_GEOMETRIC_DECISION_FRESH.md", AMENDMENT,
        "experiments/atencion_armonica/PLAN_GEOMETRIC_DECISION_SUPERVISOR.md",
        "experiments/atencion_armonica/profile_geometric_decision_closing.py",
        "experiments/atencion_armonica/profile_geometric_decision_observed.py",
        "experiments/atencion_armonica/profile_geometric_decision.py",
        "experiments/atencion_armonica/select_geometric_decision.py")
    return sorted([*code_snapshot(), *[reference(p) for p in extra]], key=lambda r: r["path"])


def completed_operation(control, operation):
    matches = []
    for path in sorted(control.path("attempts").glob("*/finish.json")):
        ref = control.reference(path)
        finish = control.json(ref)
        start = control.json(finish["start"])
        manifest = control.json(start["manifest"])
        if manifest.get("operation") == operation and finish["status"] == "COMPLETE":
            matches.append((ref, finish, start, manifest, control.json(finish["completion"])))
    if len(matches) != 1:
        raise ValueError(f"requires exactly one COMPLETE {operation}; do not repeat it")
    return matches[0]


def closing_admission(control, observed):
    ref, finish, start, manifest, output = completed_operation(control, "profile-closing")
    root = BASES[0]/"profiles/closing-cpu-0"
    if (start["stage"] != "profile" or start["binding"] != control.binding
            or manifest["root"] != str(root) or output["root"] != str(root)
            or output["manifest"] != start["manifest"]):
        raise ValueError("closing profile completion provenance differs")
    profile = ReadOnlyStore(root, binding_ref=output["binding"])
    report = profile.json(output["result"])
    admission = profile.json(report["admission"])
    if (profile.binding != manifest["binding"] or profile.binding["protocol"] != control.binding["protocol"]
            or admission["binding"] != profile.binding or admission["observed_profile"] != observed["forecast"]
            or report["schema"] != "geometric-decision-closing-profile-v1" or report["binding"] != profile.binding
            or report["new_observations"] != 0 or report["new_test_access"] is not False
            or report["original"] != report["original_replay"] or report["probe"] != report["probe_replay"]
            or profile.json(report["exclusions"]) != observed["exclusions"]):
        raise ValueError("closing profile is not the exact admitted OPEN recovery")
    for source in profile.binding["code"]:
        if reference(source["path"]) != source:
            raise ValueError("closing profile source changed")
    expected = closing_projection(report["timings"], observed["forecast"],
        overhead=report["overhead_seconds"], profile_bytes=report["profile_bytes"])
    if report["forecast"] != expected:
        raise ValueError("closing forecast differs from measured phases")
    tail = finish["seconds"]-report["elapsed_to_forecast_seconds"]
    if not math.isfinite(tail) or tail < 0:
        raise ValueError("closing finish has invalid elapsed tail")
    forecast = {**expected, "fresh_seconds": expected["fresh_seconds"]+5*tail,
        "evaluation_seconds": expected["evaluation_seconds"]+5*tail,
        "closing_finish_tail_seconds": tail, "profile_finish": ref, "profile_report": output["result"]}
    forecast = allocate_recovery(forecast, observed["forecast"], report["timings"])
    forecast["accounting_amendment"] = reference(AMENDMENT)
    return profile, report, forecast


def requested_reservations(forecast):
    operations = forecast.get("operation_seconds")
    if operations is None:
        return {"fresh": math.ceil(forecast["fresh_seconds"]), "evaluation": math.ceil(forecast["evaluation_seconds"])}
    return {"fresh": math.ceil(operations["prospective-observables"])+math.ceil(operations["observable-replay"]),
        "evaluation": math.ceil(operations["metrics-and-replay"])}


def check_budget(control, forecast):
    paths = sorted(control.path("attempts").glob("*/start.json"))
    if not paths:
        raise ValueError("completed preparation ledger required")
    end = paths[-1].parent/"finish.json"
    if not end.is_file():
        raise ValueError("unclosed operator attempt requires reconciliation before new stage")
    charged = control.json(control.reference(end))["charged_after"]
    requested = requested_reservations(forecast)
    if any(requested[k]+charged[k] > STAGES[k] for k in requested):
        raise ValueError("full forecast does not fit remaining budget; document resource revision before tests")
    roots = [Path(p) for p in control.binding["output_roots"]]
    existing_bytes = owned_bytes(roots)
    free_bytes = min(shutil.disk_usage(p).free for p in roots)
    if (existing_bytes+forecast["projected_new_bytes"] > LIMITS["new_bytes"]
            or free_bytes-forecast["projected_new_bytes"] < LIMITS["free_bytes"]
            or sum(charged.values())+sum(requested.values())+STAGES["audit"] > LIMITS["total_seconds"]):
        raise ValueError("full forecast loses disk or audit reserve")
    return {"charged_before_freeze": charged, "reservations": requested,
            "existing_bytes": existing_bytes, "free_bytes": free_bytes,
            "audit_reserved_seconds": STAGES["audit"], "forecast": forecast}


def context(control, *, check):
    check()
    campaign, campaign_ref, training = admitted_training(control, root=BASES[0]/"training")
    check()
    selection, selection_ref, selected = admitted_selection(control, root=BASES[0]/"selection",
        campaign=campaign, campaign_ref=campaign_ref, training=training)
    archive, complete, evidence = admitted_archive(control, selection, selection_ref, selected)
    check()
    observed = admit_profile(control, archive, complete, evidence, VerifiedBytes(ROOT), check=check)
    closing, closing_report, forecast = closing_admission(control, observed)
    check()
    return dict(archive=archive, archive_complete=complete, archive_evidence=evidence,
        selection=selection, selection_ref=selection_ref, observed=observed,
        closing=closing, closing_report=closing_report, forecast=forecast)


def prepare_freeze(control, inputs, ctx):
    path = control.path("freezes/prospective.json")
    code, protocol = sources(), reference(PROTOCOL)
    if protocol["sha256"] != PROTOCOL_SHA:
        raise ValueError("scientific protocol changed")
    admission_ref, admission_finish, _, admission_manifest, admission_output = completed_operation(control, "admission-fresh")
    if (admission_output["inputs"] != inputs or admission_output["forecast"] != ctx["forecast"]
            or admission_manifest["code"] != code or admission_manifest["protocol"] != protocol):
        raise ValueError("freeze inputs differ from the completed fresh admission")
    if path.exists():
        ref = control.reference(path)
        frozen = control.json(ref)
        budget = frozen["budget"]
        if (set(budget) != {"charged_before_freeze", "reservations", "existing_bytes", "free_bytes", "audit_reserved_seconds", "forecast"}
                or budget["charged_before_freeze"] != admission_finish["charged_after"]
                or budget["forecast"] != ctx["forecast"] or budget["reservations"] != requested_reservations(ctx["forecast"])
                or budget["audit_reserved_seconds"] != STAGES["audit"]):
            raise ValueError("frozen budget differs from its authenticated admission")
        exclusions = frozen["exclusions"]
        if control.json(exclusions) != ctx["observed"]["exclusions"]:
            raise ValueError("frozen exclusions differ from admission")
    else:
        budget = check_budget(control, ctx["forecast"])
        exclusions = control.publish_json("freezes/exclusions.json", ctx["observed"]["exclusions"])
    expected = {"schema": "geometric-decision-prospective-freeze-v1", "protocol": protocol, "code": code,
        "test_roster": TEST_ROSTER, "fresh_root": str(BASES[0]/"fresh"), "exclusions": exclusions,
        "head_roster": inputs["head_roster"],
        "archive": ctx["archive_evidence"], "selection": ctx["selection_ref"],
        "checkpoints": inputs["checkpoints"], "runtime": inputs["runtime"], "budget": budget,
        "normalization": inputs["normalization"], "scale": inputs["scale"],
        "input_provenance": {"open": inputs["open"], "closing_profile": ctx["forecast"]["profile_finish"],
            "admission_finish": admission_ref}, "accounting_amendment": reference(AMENDMENT)}
    if path.exists():
        if frozen != expected:
            raise ValueError("existing freeze differs from authenticated inputs or schema")
        return ref, frozen
    return control.publish_json("freezes/prospective.json", expected), expected


def bounded_operation(control, manifest_ref, *, stage, started_at, verify, reservation, operation, vram=None):
    """Single cumulative ledger; CUDA sampling is supplied only by fresh."""
    budget = StageBudget(control, stage, manifest_ref=manifest_ref, reservation_seconds=reservation,
        started_at=started_at, prior_charges=control.binding["prior_charges"],
        output_roots=[Path(p) for p in control.binding["output_roots"]], vram=vram)
    def stop(signum, frame):
        if signum == signal.SIGALRM:
            raise BudgetExceeded("prospective operator deadline")
        raise InterruptedError("prospective operation paused; preserve artifacts")
    old = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM)}
    try:
        for sig in old:
            signal.signal(sig, stop)
        signal.setitimer(signal.ITIMER_REAL, max(.001, reservation-(time.monotonic()-started_at)))
        verify()
        result = operation(budget.check)
        verify()
        return {"output": result, "finish": budget.finish("COMPLETE", completion=result)}
    except BaseException as exc:
        if not budget.closed:
            budget.finish("LIMIT_REACHED" if isinstance(exc, BudgetExceeded) else "PAUSED"
                if isinstance(exc, (InterruptedError, KeyboardInterrupt)) else "FAILED")
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.)
        for sig, handler in old.items():
            signal.signal(sig, handler)


def evaluation_admission_reservation(control, freeze_ref):
    """Small authenticated metadata read; never spend outside frozen allocation."""
    frozen = control.json(freeze_ref)
    budget = frozen["budget"]
    allocation = budget["reservations"]["evaluation"]
    baseline = budget["charged_before_freeze"]["evaluation"]
    if (frozen["schema"] != "geometric-decision-prospective-freeze-v1"
            or type(allocation) not in (float, int) or not math.isfinite(allocation) or allocation <= 0
            or type(baseline) not in (float, int) or not math.isfinite(baseline) or baseline < 0
            or budget["reservations"] != requested_reservations(budget["forecast"])):
        raise ValueError("invalid frozen evaluation allocation")
    starts = sorted(control.path("attempts").glob("*/start.json"))
    if not starts or not (starts[-1].parent/"finish.json").is_file():
        raise ValueError("unclosed ledger before evaluation admission")
    charged = control.json(control.reference(starts[-1].parent/"finish.json"))["charged_after"]["evaluation"]
    if not math.isfinite(charged) or charged < baseline:
        raise ValueError("evaluation ledger precedes frozen baseline")
    remaining = min(allocation-(charged-baseline), STAGES["evaluation"]-charged)
    if remaining <= 0:
        raise BudgetExceeded("frozen evaluation allocation exhausted before admission")
    return min(120., remaining)


def admit_launch(control, mode, *, started_at):
    """Authenticate all heavy inputs with live guards before freezing/drawing."""
    if mode not in ("fresh", "evaluate", "replay"):
        raise ValueError("unknown prospective stage")
    name = "prospective-observables" if mode == "fresh" else mode
    if (control.path(f"manifests/{name}.json").exists()
            or control.path(f"manifests/admission-{mode}.json").exists()):
        raise ValueError("stage already attempted; inspect its receipts, no silent retry")
    observed_output, reservation = None, 120.
    if mode != "fresh":
        _, _, _, _, observed_output = completed_operation(control, "prospective-observables")
    if mode == "replay":
        completed_operation(control, "evaluate")
    if mode != "fresh":
        reservation = evaluation_admission_reservation(control, observed_output["test_freeze"])
    code, protocol = sources(), reference(PROTOCOL)
    manifest = control.publish_json(f"manifests/admission-{mode}.json", {
        "operation": "admission-"+mode, "code": code, "protocol": protocol})
    def verify():
        if protocol["sha256"] != PROTOCOL_SHA or sources() != code or reference(PROTOCOL) != protocol:
            raise ValueError("prospective admission source changed")
    state = {}
    def operation(check):
        check()
        opened, preparation, open_ref = admitted_open()
        if opened.binding != control.binding or opened.root != control.root:
            raise ValueError("OPEN returned a different control")
        check()
        ctx = context(control, check=check)
        reuse = OpenReuse()
        verify_heads(ctx["archive"], ctx["archive_complete"]["heads"], ctx["selection"], ctx["selection_ref"], check=check)
        scale_ref = preparation.completion(open_ref)["scale"]
        inputs = {"open": open_ref, "head_roster": ctx["archive"].json(ctx["archive_complete"]["heads"])["records"],
            "runtime": ctx["observed"]["profile"].json(ctx["observed"]["report"]["runtime"]),
            "checkpoints": reuse.common["checkpoints"],
            "normalization": {"source_root": str(preparation.source.store.root),
                "source": preparation.source.corpus.normalizer_ref,
                "normalizers": {k: deepcopy(preparation.source.corpus.norm[k]) for k in ("common", "evidence")}},
            "scale": {"source_root": str(preparation.store.root), "source": scale_ref,
                "scale": preparation.store.json(scale_ref)["scale"]}}
        check()
        state.update(ctx=ctx, inputs=inputs)
        return control.publish_json(f"outputs/admission-{mode}.json", {
            "manifest": manifest, "inputs": inputs, "forecast": ctx["forecast"]})
    result = bounded_operation(control, manifest, stage="fresh" if mode == "fresh" else "evaluation",
        started_at=started_at, verify=verify, reservation=reservation, operation=operation)
    return state["ctx"], state["inputs"], result


def observable_arguments(store, frozen, ctx, *, check, compute):
    runtime = frozen["runtime"]
    normalizer = store.publish_json("normalizers.json", frozen["normalization"])
    store.publish_json("scale.json", frozen["scale"])
    runtime_ref = store.publish_json("runtime.json", runtime)
    fitter = ge.law.GroupFitter(ge.law.Grid(257, 65, 4), device="cuda", assignment_batch=8) if compute else None
    def forward(cp, records):
        if not compute:
            raise RuntimeError("replay cannot forward")
        check()
        result = checkpoint_forward(cp, records, runtime)
        check()
        return result
    def fit(q, partitions):
        if not compute:
            raise RuntimeError("replay cannot fit")
        check()
        result = ge.law.fit_candidates(q, partitions, fitter)
        check()
        return result
    return dict(checkpoints=frozen["checkpoints"], runtime=runtime, forward=forward,
        normalizers=frozen["normalization"]["normalizers"], normalization_ref=normalizer,
        scale=frozen["scale"]["scale"], fit_candidates=fit,
        fit_origin={"grid": {"beta": 257, "gamma": 65, "coarse_stride": 4}, "assignment_batch": 8,
                    "device": "cuda", "runtime": runtime_ref},
        head_store=ctx["archive"], archive_ref=ctx["archive_complete"]["heads"],
        selection_store=ctx["selection"], selection_ref=ctx["selection_ref"], device="cuda:0", check=check)


def fresh_operation(control, store, freeze_ref, frozen, ctx, manifest, check):
    gpu_availability()
    runtime = gpu_runtime()
    if runtime != frozen["runtime"]:
        raise ValueError("actual CUDA runtime differs from frozen observed profile")
    torch.cuda.set_per_process_memory_fraction(.25, 0)
    args = observable_arguments(store, frozen, ctx, check=check, compute=True)
    draws = DrawBatch(store, freeze_ref, exclusions=control.json(frozen["exclusions"]))
    batches = []
    for split, seed in TESTS:
        check()
        draw_ref = draws.produce(split, draw=_draw_scene, check=check)
        data = draws.observations(split, check=check)
        observed = run_observed(store, "observed/"+split, split, data["observations"],
            expected_seed=seed, observation_origin=data["origin"], **args)
        batches.append({"split": split, "draws": draw_ref, "observed": observed})
        print(json.dumps({"stage": "observed_complete", "split": split, "scenes": 512}), flush=True)
    # Release fitter/model storage before the long CPU-only pre-seal recovery.
    del args
    torch.cuda.empty_cache()
    args = observable_arguments(store, frozen, ctx, check=check, compute=False)
    def recover(split):
        data = draws.observations(split, check=check)
        return run_observed(store, "observed/"+split, split, data["observations"],
            expected_seed=dict(TESTS)[split], observation_origin=data["origin"], recovery_only=True, **args)
    seal = seal_outputs(control, store, freeze_ref, batches, recover=recover, check=check)
    return control.publish_json("outputs/prospective-observables.json", {
        "manifest": manifest, "test_freeze": freeze_ref, "seal": seal})


def observable_replay(control, finish_ref, freeze_ref, frozen, ctx, manifest, *, check):
    """CPU-only postseal recovery, accounted to fresh by the pinned amendment."""
    _, seal, seal_ref = admitted_seal(control, finish_ref, freeze_ref, check=check)
    store = ArtifactStore(BASES[0]/"fresh", binding={"test_freeze": freeze_ref})
    kwargs = observable_arguments(store, frozen, ctx, check=check, compute=False)
    draws = DrawBatch(store, freeze_ref, exclusions=control.json(frozen["exclusions"]))
    recovered_rows = []
    for row in seal["batches"]:
        split = row["split"]
        data = draws.observations(split, check=check)
        recovered = run_observed(store, "observed/"+split, split, data["observations"],
            expected_seed=dict(TESTS)[split], observation_origin=data["origin"], recovery_only=True, **kwargs)
        if recovered != row["observed"]:
            raise ValueError("postseal observable replay changed output")
        recovered_rows.append({"split": split, "observed": recovered})
    return control.publish_json("outputs/observable-replay.json", {"manifest": manifest,
        "test_freeze": freeze_ref, "observable_finish": finish_ref, "seal": seal_ref, "recovered": recovered_rows})


def admitted_observable_replay(control, finish_ref, freeze_ref, seal_ref):
    ref, _, start, manifest, output = completed_operation(control, "observable-replay")
    expected = {"operation": "observable-replay", "test_freeze": freeze_ref,
        "observable_finish": finish_ref, "seal": seal_ref}
    seal = control.json(seal_ref)
    if (start["stage"] != "fresh" or start["binding"] != control.binding or manifest != expected
            or output != {"manifest": start["manifest"], "test_freeze": freeze_ref,
                "observable_finish": finish_ref, "seal": seal_ref,
                "recovered": [{"split": r["split"], "observed": r["observed"]} for r in seal["batches"]]}):
        raise ValueError("metric replay requires the exact completed observable recovery")
    return ref


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("fresh", "evaluate", "replay"), required=True)
    args = parser.parse_args()
    if (Path.cwd().resolve() != ROOT or any(os.environ.get(k) != "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"))
            or (args.stage != "fresh" and os.environ.get("CUDA_VISIBLE_DEVICES") != "")
            or (args.stage == "fresh" and (os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, "0")
                                          or os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8"))):
        raise ValueError("explicit one-thread stage-appropriate CPU/GPU environment required")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    view = ReadOnlyStore(BASES[0]/"control", binding_ref=CONTROL_BINDING)
    control = ArtifactStore(view.root, binding=view.binding)
    with control.path("operator.lock").open("ab") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ctx, inputs, _ = admit_launch(control, args.stage, started_at=LAUNCH_STARTED)
        operation_started = time.monotonic()
        if args.stage != "fresh" and not control.path("freezes/prospective.json").exists():
            raise ValueError("evaluation/replay cannot create the test freeze")
        freeze_ref, frozen = prepare_freeze(control, inputs, ctx)
        def verify():
            if reference(PROTOCOL) != frozen["protocol"] or sources() != frozen["code"] or control.json(freeze_ref) != frozen:
                raise ValueError("prospective source/freeze changed")
        if args.stage == "fresh":
            if control.path("outputs/prospective-observables.json").exists():
                raise ValueError("observable output already exists; reconcile instead of relaunching")
            gpu_availability()
            store = ArtifactStore(BASES[0]/"fresh", binding={"test_freeze": freeze_ref})
            manifest = control.publish_json("manifests/prospective-observables.json", {
                "operation": "prospective-observables", "test_freeze": freeze_ref, "root": str(store.root)})
            result = bounded_operation(control, manifest, stage="fresh", started_at=operation_started,
                verify=verify, reservation=math.ceil(frozen["budget"]["forecast"]["operation_seconds"]["prospective-observables"]),
                vram=lambda: torch.cuda.memory_reserved(0) if torch.cuda.is_initialized() else 0,
                operation=lambda check: fresh_operation(control, store, freeze_ref, frozen, ctx, manifest, check))
        else:
            finish_ref, _, _, _, observed_output = completed_operation(control, "prospective-observables")
            mode = args.stage
            recovery_finish = None
            if mode == "replay":
                recovery_manifest = control.publish_json("manifests/observable-replay.json", {
                    "operation": "observable-replay", "test_freeze": freeze_ref,
                    "observable_finish": finish_ref, "seal": observed_output["seal"]})
                recovered = bounded_operation(control, recovery_manifest, stage="fresh", started_at=operation_started,
                    verify=verify, reservation=math.ceil(frozen["budget"]["forecast"]["operation_seconds"]["observable-replay"]),
                    operation=lambda check: observable_replay(control, finish_ref, freeze_ref, frozen, ctx, recovery_manifest, check=check))
                operation_started = time.monotonic()
                recovery_finish = admitted_observable_replay(control, finish_ref, freeze_ref, observed_output["seal"])
                if recovery_finish != recovered["finish"]:
                    raise ValueError("new recovery finish does not match admitted one")
            output = ArtifactStore(BASES[0]/"evaluation", binding={"test_freeze": freeze_ref, "prediction_seal": observed_output["seal"]})
            manifest = control.publish_json(f"manifests/{mode}.json", {"operation": mode,
                "test_freeze": freeze_ref, "observable_finish": finish_ref, "root": str(output.root),
                "observable_replay_finish": recovery_finish})
            def operation(check):
                if mode == "replay":
                    if admitted_observable_replay(control, finish_ref, freeze_ref, observed_output["seal"]) != recovery_finish:
                        raise ValueError("recovery authority changed before metric replay")
                complete = evaluate_fresh(control, finish_ref, freeze_ref, output, check=check, replay=mode == "replay")
                return control.publish_json(f"outputs/{mode}.json", {"manifest": manifest,
                    "root": str(output.root), "complete": complete, "test_freeze": freeze_ref})
            # A single evaluation budget is shared across evaluation and replay;
            # the remaining allowance is enforced by StageBudget each launch.
            last = sorted(control.path("attempts").glob("*/finish.json"))[-1]
            charged = control.json(control.reference(last))["charged_after"]["evaluation"]
            # Admissions and both metric passes share this frozen allocation.
            baseline = frozen["budget"]["charged_before_freeze"]["evaluation"]
            reserve = min(frozen["budget"]["reservations"]["evaluation"]-(charged-baseline), STAGES["evaluation"]-charged)
            result = bounded_operation(control, manifest, stage="evaluation", started_at=operation_started,
                verify=verify, reservation=reserve, operation=operation)
        print(json.dumps({"stage": args.stage, **result}), flush=True)


if __name__ == "__main__":
    main()
