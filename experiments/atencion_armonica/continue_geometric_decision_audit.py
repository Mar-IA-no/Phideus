"""One-shot adapter revision continuation; never rerun the historical recovery.

Historical pins authenticate the failed attempt and its already published cut.
Reviewed current pins authorize the checker that will actually execute.
"""
from __future__ import annotations

import time
LAUNCH_STARTED = time.monotonic()

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import signal

from experiments.atencion_armonica import audit_geometric_decision as checker
from experiments.atencion_armonica import audit_geometric_decision_core as core
from src.atencion_armonica.geometric_decision_budget import LIMITS, STAGES, BudgetExceeded, StageBudget
from src.atencion_armonica.geometric_decision_store import ArtifactStore

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "data/atencion_armonica/geometric_decision_energy_v1"
OLD_ROOT = BASE / "audit-final-verify"
VERIFY_ROOT = BASE / "audit-final-adapter-verify"
WRAPPER = "experiments/atencion_armonica/continue_geometric_decision_audit.py"
PLAN = "experiments/atencion_armonica/PLAN_GEOMETRIC_DECISION_AUDIT_ADAPTER_CONTINUATION.md"
MANIFEST = "manifests/final-technical-audit-adapter-continuation.json"
OUTPUT = "outputs/final-technical-audit-adapter-continuation.json"
DISCREPANCY = "discrepancies/final-technical-audit-adapter-continuation.json"
RESERVATION = 1500.0
PROJECTION_SECONDS = 327.0271509163656
PINS = {
    "finish": {"path": "attempts/0022/finish.json", "bytes": 435,
        "sha256": "d72866e2d75c2d2a67634582695cc34e59346becef2c1f7c15bd02a94c083d4c"},
    "liability": {"path": "attempts/0021/start.json", "bytes": 1334,
        "sha256": "1f2413a58f93e858c9ebbd9dcc2868683112710261e24be14dfbafecda395810"},
    "discrepancy": {"path": "discrepancies/final-technical-audit-recovery.json", "bytes": 649,
        "sha256": "02e9176894039f0d9a2be855e331b4f6cd3fce4fc508568aa958ddb41aef706d"},
    "binding": {"path": "binding.json", "bytes": 5465,
        "sha256": "0159b5f954b6f06ee200f49ed6da103404d2ea80a421cd494cf80ac69dff3d8c"},
    "precommit": {"path": "precommit.json", "bytes": 19041,
        "sha256": "4af416c714bf2475056af0ac06127d50e55d428ea900647ac00a7fbf599a5fa0"},
}
OLD_CODE = [
    {"path": checker.OPERATOR, "bytes": 131740,
     "sha256": "25e81f578567da757c4a5b071a74fda44fac2ad6df11e27a5ed11f8ed7a3d6cf"},
    {"path": checker.CORE, "bytes": 42824,
     "sha256": "ae09edc9507e89e73df74e03b5345c95572601032e966f83de283d44f2d55dd9"},
]
REVIEWS = {
    "Biblioteca/Geometric_Decision_Energy/agent_reports/780_final_audit_adapter_contract_review.md":
        "b39b3ec3d61162d1a204b06a29c979808aabd345796158ef51484dea802cd685",
    "Biblioteca/Geometric_Decision_Energy/agent_reports/781_adapter_continuation_design_review.md":
        "ffff839ec26a97c457fef7dc3e49a5b320d818f0ecd8ab41838bb8563777c209",
}
ACCESS_RECORD = "Biblioteca/Geometric_Decision_Energy/RESULT_READING_DRAFT_20260915.md"
ACCESS_SHA = "fb6444b9e1af7441929e523a8f0fd2c170b3fd4533257d235ce1fea2e07af602"
FIXED_FIELDS = ("cuts", "terminal_receipts", "freeze_ref", "seal_ref", "contract_ref", "report_result_ref")


def approved_source(path, digest):
    ref = checker.source_reference(path)
    if (type(digest) is not str or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest) or ref["sha256"] != digest):
        raise ValueError("source differs from reviewed pin: " + path)
    return ref


def require_absent(control, verify_root):
    if verify_root.exists() or any(control.path(p).exists() for p in (MANIFEST, OUTPUT, DISCREPANCY)):
        raise ValueError("adapter continuation already attempted; no retry")


def current_checker(expected):
    code = checker.sources()
    if code != expected:
        raise ValueError("current checker differs from approved sources before start")
    return code


def validate_history(control, reader, *, old_root, verify_root, pins, old_code,
                     expected_tail=22):
    """Only small authenticated terminal metadata before the new budget start."""
    require_absent(control, verify_root)
    ledger = checker.read_ledger(reader, control.binding)
    if len(ledger["attempts"]) != expected_tail + 1:
        raise ValueError("failed recovery is not the exact ledger tail")
    tail, liability = ledger["attempts"][-1], ledger["attempts"][-2]
    finish = reader.json(pins["finish"])
    discrepancy = reader.json(pins["discrepancy"])
    incident_start = reader.json(pins["liability"])
    if (tail["finish_ref"] != pins["finish"] or tail["finish"] != finish
            or finish["status"] != "FAILED" or finish["completion"] is not None
            or finish["start"] != tail["start_ref"]
            or ledger["charged"] != finish["charged_after"]
            or tail["manifest"].get("operation") != "final-technical-audit"
            or tail["manifest"].get("root") != str(old_root)
            or liability["start_ref"] != pins["liability"] or liability["start"] != incident_start
            or liability["finish"] is not None
            or liability["manifest"].get("operation") != "final-technical-audit-launch-incident-accounting"
            or discrepancy.get("exception_type") != "KeyError" or discrepancy.get("message") != "'root'"
            or discrepancy.get("manifest") != tail["start"]["manifest"]
            or discrepancy.get("incident_start") != liability["start_ref"]
            or discrepancy.get("incident_manifest") != liability["start"]["manifest"]):
        raise ValueError("failed recovery/incident/discrepancy lineage differs")
    if (not old_root.is_dir() or old_root.is_symlink()
            or {p.name for p in old_root.iterdir()} != {"binding.json", "precommit.json"}):
        raise ValueError("historical verify root topology differs")
    old_store = core.AuditStore("historical-audit", old_root, core.Coverage())
    binding, precommit = old_store.json(pins["binding"]), old_store.json(pins["precommit"])
    preflight = checker.unique_operation(ledger, "final-technical-audit-preflight")
    authority = binding.get("recovery", {})
    if (binding.get("code") != old_code or precommit.get("checker") != old_code
            or precommit.get("binding") != binding
            or precommit.get("schema") != "geometric-decision-final-audit-precommit-v1"
            or authority != tail["manifest"].get("recovery")
            or authority.get("checker") != old_code or authority.get("verify_root") != str(old_root)
            or authority.get("incident_start") != liability["start_ref"]
            or authority.get("incident_manifest") != liability["start"]["manifest"]
            or authority.get("preflight_finish") != preflight["finish_ref"]
            or authority.get("preflight_output") != preflight["finish"]["completion"]
            or preflight["manifest"]["binding"].get("code") != old_code
            or preflight["manifest"]["binding"].get("control_binding") != checker.binding_ref(control.binding)
            or precommit.get("plan") != binding.get("plan")
            or binding.get("inputs") != precommit.get("terminal_receipts")
            or precommit.get("preflight") != binding.get("preflight")):
        raise ValueError("historical checker/precommit/recovery binding differs")
    expected_preflight = {"finish": preflight["finish_ref"], "output": preflight["finish"]["completion"],
        "admission": preflight["output"]["admission"], "projection": preflight["output"]["projection"]}
    if core.encoded(expected_preflight) != core.encoded(precommit["preflight"]):
        raise ValueError("historical preflight reference differs")
    return ledger, preflight, precommit


def start_attempt(control, ledger, *, authority, verify_root, started_at,
                  reservation=RESERVATION, projected_seconds=PROJECTION_SECONDS):
    require_absent(control, verify_root)
    available = min(STAGES["audit"] - ledger["charged"]["audit"],
                    LIMITS["total_seconds"] - sum(ledger["charged"].values()))
    if (not math.isfinite(reservation) or not 0 < reservation <= RESERVATION
            or not math.isfinite(projected_seconds) or not 0 < projected_seconds <= reservation
            or reservation > available):
        raise BudgetExceeded("adapter continuation does not fit fixed ledger budget")
    manifest = control.publish_json(MANIFEST, {
        "schema": "geometric-decision-audit-adapter-continuation-manifest-v1",
        "operation": "final-technical-audit", "root": str(verify_root),
        "continuation": authority, "reservation_seconds": reservation,
        "projected_seconds": projected_seconds})
    budget = StageBudget(control, "audit", manifest_ref=manifest,
        reservation_seconds=reservation, started_at=started_at,
        prior_charges=control.binding["prior_charges"],
        output_roots=[Path(p) for p in control.binding["output_roots"]])
    if budget.charged != ledger["charged"]:
        budget.finish("FAILED")
        raise ValueError("continuation charged_before differs from failed recovery charged_after")
    return manifest, budget


def fixed_precommit(admitted, old, binding, code, plan):
    for key in FIXED_FIELDS:
        if core.encoded(admitted[key]) != core.encoded(old[key]):
            raise ValueError("published structural cut or authority changed: " + key)
    if (core.encoded(binding["preflight"]) != core.encoded(old["preflight"])
            or binding["continuation"]["access_history"].get("result_payloads_opened") is not True):
        raise ValueError("preflight/access history differs")
    return {"schema": "geometric-decision-final-audit-precommit-v1", "binding": binding,
        "plan": plan, "checker": code, "preflight": binding["preflight"],
        **{key: admitted[key] for key in FIXED_FIELDS}, "result_payloads_opened": True}


def close_failure(control, manifest, budget, exc):
    try:
        control.publish_json(DISCREPANCY, {
            "schema": "geometric-decision-audit-adapter-continuation-discrepancy-v1",
            "manifest": manifest, "exception_type": type(exc).__name__, "message": str(exc),
            "authority": "raw technical discrepancy; no scientific conclusion"})
    finally:
        if not budget.closed:
            status = ("LIMIT_REACHED" if isinstance(exc, BudgetExceeded) else "PAUSED"
                      if isinstance(exc, (InterruptedError, KeyboardInterrupt)) else "FAILED")
            budget.finish(status)


def execute(control, reader, ledger, preflight, old_precommit, authority, manifest, budget,
            *, verify_root, coverage, sources_unchanged):
    """Production boundary; tests replace only the scientific callbacks."""
    try:
        check = budget.check
        check()
        preflight_store = checker.open_bound("preserved-preflight", Path(preflight["output"]["root"]),
            preflight["manifest"]["binding"], coverage, check=check)
        metadata = preflight_store.json(preflight["output"]["admission"], check=check)
        projection = preflight_store.json(preflight["output"]["projection"], check=check)
        if (projection.get("schema") != "geometric-decision-final-audit-projection-v1"
                or projection.get("projected_seconds") != PROJECTION_SECONDS):
            raise ValueError("preserved projection differs")
        for name in checker.REQUIRED_COMPLETE:
            if metadata["terminal_receipts"].get(name) != checker.unique_operation(ledger, name)["finish_ref"]:
                raise ValueError("preserved terminal receipt differs: " + name)
        binding = {"schema": "geometric-decision-final-audit-binding-v1",
            "plan": authority["audit_plan"], "protocol": authority["protocol"],
            "code": authority["current_checker"], "inputs": metadata["terminal_receipts"],
            "preflight": old_precommit["preflight"], "continuation": authority}
        store = ArtifactStore(verify_root, binding=binding)
        admitted = checker.precommit_inputs(reader, ledger, coverage, metadata, check=check)
        precommit = fixed_precommit(admitted, old_precommit, binding,
                                   authority["current_checker"], authority["audit_plan"])
        pref = store.publish_json("precommit.json", precommit)
        check(force_resources=True)
        result = checker.verify_all(control, reader, ledger, budget.start_ref, pref, store,
                                    coverage, check=check)
        sources_unchanged()
        result_ref = store.publish_json("result.json", result)
        completion = control.publish_json(OUTPUT, {
            "schema": "geometric-decision-audit-adapter-continuation-output-v1",
            "manifest": manifest, "root": str(verify_root),
            "binding": store.reference(store.path("binding.json")),
            "precommit": pref, "result": result_ref, "continuation": authority})
        return completion, budget.finish("COMPLETE", completion=completion)
    except BaseException as exc:
        close_failure(control, manifest, budget, exc)
        raise


def production_main(wrapper_sha, plan_sha, checker_sha):
    if (Path.cwd().resolve() != ROOT or os.environ.get("CUDA_VISIBLE_DEVICES") != ""
            or any(os.environ.get(k) != "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"))):
        raise ValueError("continuation requires project cwd, hidden CUDA and one-thread CPU")
    reviewed = [approved_source(WRAPPER, wrapper_sha), approved_source(PLAN, plan_sha),
        approved_source(checker.OPERATOR, checker_sha), approved_source(checker.CORE, OLD_CODE[1]["sha256"]),
        approved_source(checker.PLAN, checker.PLAN_SHA), approved_source(checker.PROTOCOL, checker.PROTOCOL_SHA),
        *[approved_source(p, digest) for p, digest in REVIEWS.items()], approved_source(ACCESS_RECORD, ACCESS_SHA)]
    code = current_checker(reviewed[2:4])
    coverage = core.Coverage()
    reader = core.AuditStore("control", BASE / "control", coverage)
    control = ArtifactStore(reader.root, binding=reader.json(checker.CONTROL_BINDING))
    with control.path("operator.lock").open("ab") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ledger, preflight, old = validate_history(control, reader, old_root=OLD_ROOT,
            verify_root=VERIFY_ROOT, pins=PINS, old_code=OLD_CODE)
        if (old["plan"] != reviewed[4] or old["binding"]["protocol"] != reviewed[5]
                or preflight["manifest"]["binding"].get("plan") != reviewed[4]
                or preflight["manifest"]["binding"].get("protocol") != reviewed[5]):
            raise ValueError("historical audit plan/protocol differs")
        authority = {"schema": "geometric-decision-audit-adapter-continuation-authority-v1",
            "wrapper": reviewed[0], "continuation_plan": reviewed[1],
            "current_checker": code, "historical_checker": OLD_CODE,
            "audit_plan": reviewed[4], "protocol": reviewed[5], "reviews": reviewed[6:8],
            "failed_finish": PINS["finish"], "failed_discrepancy": PINS["discrepancy"],
            "liability_start": PINS["liability"],
            "previous_precommit": {"root": str(OLD_ROOT), "binding": PINS["binding"], "precommit": PINS["precommit"]},
            "access_history": {"result_payloads_opened": True, "record": reviewed[8],
                "scope": "coordinator read report metrics after historical PRECOMMIT; exact cut retained"}}
        manifest, budget = start_attempt(control, ledger, authority=authority,
            verify_root=VERIFY_ROOT, started_at=LAUNCH_STARTED)
        previous = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM)}

        def stop(signum, frame):
            if signum == signal.SIGALRM:
                raise BudgetExceeded("adapter continuation reservation exhausted")
            raise InterruptedError("adapter continuation interrupted; no retry")

        def unchanged():
            if checker.sources() != code or any(checker.source_reference(r["path"]) != r for r in reviewed):
                raise ValueError("reviewed continuation sources changed during execution")

        try:
            for sig in previous:
                signal.signal(sig, stop)
            signal.setitimer(signal.ITIMER_REAL, max(.001, RESERVATION - (time.monotonic() - LAUNCH_STARTED)))
            completion, finish = execute(control, reader, ledger, preflight, old, authority,
                manifest, budget, verify_root=VERIFY_ROOT, coverage=coverage, sources_unchanged=unchanged)
            print(json.dumps({"status": "FINAL_TECHNICAL_AUDIT_ADAPTER_CONTINUATION_COMPLETE",
                              "finish": finish, "completion": completion}), flush=True)
        except BaseException as exc:
            if not budget.closed:
                close_failure(control, manifest, budget, exc)
            raise
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0.)
            for sig, handler in previous.items():
                signal.signal(sig, handler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--approved-wrapper-sha256", required=True)
    parser.add_argument("--approved-plan-sha256", required=True)
    parser.add_argument("--approved-checker-sha256", required=True)
    args = parser.parse_args()
    production_main(args.approved_wrapper_sha256, args.approved_plan_sha256, args.approved_checker_sha256)
