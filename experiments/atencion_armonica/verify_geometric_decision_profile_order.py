"""Fixed one-shot VERIFY after attempt 0023; preserves all scientific artifacts."""
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
from experiments.atencion_armonica.continue_geometric_decision_audit import (
    approved_source, current_checker, fixed_precommit)
from src.atencion_armonica.geometric_decision_budget import LIMITS, STAGES, BudgetExceeded, StageBudget
from src.atencion_armonica.geometric_decision_store import ArtifactStore

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "data/atencion_armonica/geometric_decision_energy_v1"
OLD_ROOT = BASE / "audit-final-adapter-verify"
ORIGINAL_ROOT = BASE / "audit-final-verify"
VERIFY_ROOT = BASE / "audit-final-profile-order-verify"
WRAPPER = "experiments/atencion_armonica/verify_geometric_decision_profile_order.py"
PLAN = "experiments/atencion_armonica/PLAN_GEOMETRIC_DECISION_AUDIT_PROFILE_ORDER_CONTINUATION.md"
HELPER = "experiments/atencion_armonica/continue_geometric_decision_audit.py"
HELPER_SHA = "2518c217cb6a80ef2c9ecbe715753cb5a97ff942e83228c035bc4ac73e852244"
CHECKER_SHA = "59d872a03635ec354d35f2f9486984b6db4c151095b7a444ec2b17fc71b571a0"
CORE_SHA = "ae09edc9507e89e73df74e03b5345c95572601032e966f83de283d44f2d55dd9"
MANIFEST = "manifests/final-technical-audit-profile-order.json"
OUTPUT = "outputs/final-technical-audit-profile-order.json"
DISCREPANCY = "discrepancies/final-technical-audit-profile-order.json"
RESERVATION = 1500.0
PROJECTION_SECONDS = 327.0271509163656
FIXED_FIELDS = ("cuts", "terminal_receipts", "freeze_ref", "seal_ref", "contract_ref", "report_result_ref")
PINS = {
    "start": {"path": "attempts/0023/start.json", "bytes": 1328,
        "sha256": "eb4172883c2507709fe717ed00f01082fa9662c3ab4aa85fe2ae93ff7e34e823"},
    "finish": {"path": "attempts/0023/finish.json", "bytes": 434,
        "sha256": "37515c1098a58d057ffff326330d4f47c4d621f0533266cbbc0d5cd7cfd52e90"},
    "manifest": {"path": "manifests/final-technical-audit-adapter-continuation.json", "bytes": 3351,
        "sha256": "71507e0a94578c34a6be50edb85f6281f4fd0c273a8e89b4d59f84d3d53e52c5"},
    "discrepancy": {"path": "discrepancies/final-technical-audit-adapter-continuation.json", "bytes": 401,
        "sha256": "0b9bab7e7744310ae5c30e69bca55fecbdf067f3683cc1c02816fa9993046133"},
    "binding": {"path": "binding.json", "bytes": 6389,
        "sha256": "a997481ccfdff029e27fb068e9669121d56defbf9e2e04c7fa6bdb9c7af9f853"},
    "precommit": {"path": "precommit.json", "bytes": 19964,
        "sha256": "7b973254c40cfe46800eb16eaac993cab1fd0c29a4fa5434849c35c68f917b19"},
}
PREVIOUS_CODE = [
    {"path": checker.OPERATOR, "bytes": 133026,
        "sha256": "7b1622f0fc8c78fefe219cd9c9265db708e5cc53fd074fcec7f5e4a2346e2323"},
    {"path": checker.CORE, "bytes": 42824, "sha256": CORE_SHA},
]
ORIGINAL_CODE = [
    {"path": checker.OPERATOR, "bytes": 131740,
        "sha256": "25e81f578567da757c4a5b071a74fda44fac2ad6df11e27a5ed11f8ed7a3d6cf"},
    {"path": checker.CORE, "bytes": 42824, "sha256": CORE_SHA},
]
REVIEWS = {
    "Biblioteca/Geometric_Decision_Energy/agent_reports/784_profile_inventory_order_review.md":
        "0c894ce9c401231cda3ab4844f74deca7fd646b64a67dd30c4d2123531a58a8f",
    "Biblioteca/Geometric_Decision_Energy/agent_reports/785_profile_order_continuation_design_review.md":
        "fad01224df1967bd3e23bd6df658a794ab1445c1a772453bf90c5bd77f52df75",
}
ACCESS_RECORD = "Biblioteca/Geometric_Decision_Energy/RESULT_READING_DRAFT_20260915.md"
ACCESS_SHA = "fb6444b9e1af7441929e523a8f0fd2c170b3fd4533257d235ce1fea2e07af602"


def require_absent(control, verify_root, *, next_attempt=24):
    if (verify_root.exists() or verify_root.is_symlink()
            or control.path(f"attempts/{next_attempt:04d}").exists()
            or any(control.path(p).exists() for p in (MANIFEST, OUTPUT, DISCREPANCY))):
        raise ValueError("profile-order continuation already attempted; no retry")


def historical_precommit(root, refs, code):
    if (root.is_symlink() or not root.is_dir()
            or {p.name for p in root.iterdir()} != {"binding.json", "precommit.json"}):
        raise ValueError("historical verify root topology differs")
    store = core.AuditStore("historical-verify", root, core.Coverage())
    binding, precommit = store.json(refs["binding"]), store.json(refs["precommit"])
    if (precommit.get("schema") != "geometric-decision-final-audit-precommit-v1"
            or binding.get("schema") != "geometric-decision-final-audit-binding-v1"
            or precommit.get("binding") != binding or binding.get("code") != code
            or precommit.get("checker") != code or precommit.get("plan") != binding.get("plan")
            or precommit.get("preflight") != binding.get("preflight")
            or precommit.get("terminal_receipts") != binding.get("inputs")):
        raise ValueError("historical checker or PRECOMMIT binding differs")
    return binding, precommit


def validate_history(control, reader, *, old_root, original_root, verify_root,
                     pins, previous_code, original_code, expected_tail=23):
    """Authenticate small metadata and both old cuts, not scientific payloads."""
    require_absent(control, verify_root, next_attempt=expected_tail + 1)
    ledger = checker.read_ledger(reader, control.binding)
    if len(ledger["attempts"]) != expected_tail + 1:
        raise ValueError("failed profile verification is not the exact ledger tail")
    tail, original, liability = ledger["attempts"][-1], ledger["attempts"][-2], ledger["attempts"][-3]
    start, finish, manifest, discrepancy = (reader.json(pins[k]) for k in
                                           ("start", "finish", "manifest", "discrepancy"))
    if (tail["start_ref"] != pins["start"] or tail["start"] != start
            or tail["finish_ref"] != pins["finish"] or tail["finish"] != finish
            or start["manifest"] != pins["manifest"] or tail["manifest"] != manifest
            or finish["start"] != pins["start"] or finish["status"] != "FAILED"
            or finish["completion"] is not None or ledger["charged"] != finish["charged_after"]
            or manifest.get("operation") != "final-technical-audit" or manifest.get("root") != str(old_root)
            or manifest.get("schema") != "geometric-decision-audit-adapter-continuation-manifest-v1"
            or discrepancy.get("schema") != "geometric-decision-audit-adapter-continuation-discrepancy-v1"
            or discrepancy.get("manifest") != pins["manifest"]
            or discrepancy.get("exception_type") != "ValueError"
            or discrepancy.get("message") != "closing profile observed inventory roster differs"):
        raise ValueError("failed profile verification lineage differs")
    binding, old = historical_precommit(old_root, pins, previous_code)
    authority = binding.get("continuation", {})
    if (authority != manifest.get("continuation")
            or authority.get("schema") != "geometric-decision-audit-adapter-continuation-authority-v1"
            or authority.get("current_checker") != previous_code
            or authority.get("historical_checker") != original_code
            or authority.get("audit_plan") != binding["plan"]
            or authority.get("protocol") != binding["protocol"]
            or old.get("result_payloads_opened") is not True
            or authority.get("access_history", {}).get("result_payloads_opened") is not True):
        raise ValueError("previous continuation authority differs")
    prior = authority["previous_precommit"]
    if prior.get("root") != str(original_root):
        raise ValueError("original PRECOMMIT root differs")
    original_binding, original_cut = historical_precommit(original_root, prior, original_code)
    failure = reader.json(authority["failed_finish"])
    failure_detail = reader.json(authority["failed_discrepancy"])
    incident = reader.json(authority["liability_start"])
    recovery = original_binding.get("recovery", {})
    preflight = checker.unique_operation(ledger, "final-technical-audit-preflight")
    if (original["finish_ref"] != authority["failed_finish"] or original["finish"] != failure
            or failure["status"] != "FAILED" or failure["completion"] is not None
            or failure["start"] != original["start_ref"]
            or original["manifest"].get("operation") != "final-technical-audit"
            or original["manifest"].get("root") != str(original_root)
            or original["manifest"].get("recovery") != recovery
            or failure_detail.get("manifest") != original["start"]["manifest"]
            or failure_detail.get("exception_type") != "KeyError" or failure_detail.get("message") != "'root'"
            or liability["start_ref"] != authority["liability_start"] or liability["start"] != incident
            or liability["finish"] is not None
            or liability["manifest"].get("operation") != "final-technical-audit-launch-incident-accounting"
            or recovery.get("incident_start") != liability["start_ref"]
            or recovery.get("incident_manifest") != liability["start"]["manifest"]
            or failure_detail.get("incident_start") != liability["start_ref"]
            or failure_detail.get("incident_manifest") != liability["start"]["manifest"]
            or recovery.get("checker") != original_code or recovery.get("verify_root") != str(original_root)
            or recovery.get("preflight_finish") != preflight["finish_ref"]
            or recovery.get("preflight_output") != preflight["finish"]["completion"]
            or preflight["manifest"]["binding"].get("code") != original_code
            or preflight["manifest"]["binding"].get("control_binding") != checker.binding_ref(control.binding)):
        raise ValueError("original recovery/liability/preflight authority differs")
    expected_preflight = {"finish": preflight["finish_ref"], "output": preflight["finish"]["completion"],
        "admission": preflight["output"]["admission"], "projection": preflight["output"]["projection"]}
    for key in (*FIXED_FIELDS, "preflight", "plan"):
        if core.encoded(old[key]) != core.encoded(original_cut[key]):
            raise ValueError("original and previous fixed cut differ: " + key)
    if (old["preflight"] != expected_preflight or binding["protocol"] != original_binding["protocol"]
            or preflight["manifest"]["binding"].get("plan") != old["plan"]
            or preflight["manifest"]["binding"].get("protocol") != binding["protocol"]):
        raise ValueError("preserved preflight/plan/protocol differs")
    return ledger, preflight, old, original_cut


def start_attempt(control, ledger, *, authority, verify_root, started_at,
                  reservation=RESERVATION, projected_seconds=PROJECTION_SECONDS, next_attempt=24):
    require_absent(control, verify_root, next_attempt=next_attempt)
    if len(ledger["attempts"]) != next_attempt:
        raise ValueError("new attempt must immediately follow validated tail")
    available = min(STAGES["audit"] - ledger["charged"]["audit"],
                    LIMITS["total_seconds"] - sum(ledger["charged"].values()))
    if (not math.isfinite(reservation) or not 0 < reservation <= RESERVATION
            or not math.isfinite(projected_seconds) or not 0 < projected_seconds <= reservation
            or reservation > available):
        raise BudgetExceeded("profile-order continuation does not fit fixed ledger budget")
    manifest = control.publish_json(MANIFEST, {
        "schema": "geometric-decision-audit-profile-order-manifest-v1",
        "operation": "final-technical-audit", "root": str(verify_root),
        "continuation": authority, "reservation_seconds": reservation,
        "projected_seconds": projected_seconds})
    budget = StageBudget(control, "audit", manifest_ref=manifest,
        reservation_seconds=reservation, started_at=started_at,
        prior_charges=control.binding["prior_charges"],
        output_roots=[Path(p) for p in control.binding["output_roots"]])
    if budget.charged != ledger["charged"] or budget.folder != f"attempts/{next_attempt:04d}":
        budget.finish("FAILED")
        raise ValueError("continuation charged_before or attempt number differs")
    return manifest, budget


def close_failure(control, manifest, budget, exc):
    try:
        control.publish_json(DISCREPANCY, {
            "schema": "geometric-decision-audit-profile-order-discrepancy-v1",
            "manifest": manifest, "exception_type": type(exc).__name__, "message": str(exc),
            "authority": "raw technical discrepancy; no scientific conclusion"})
    finally:
        if not budget.closed:
            budget.finish("LIMIT_REACHED" if isinstance(exc, BudgetExceeded) else "PAUSED"
                          if isinstance(exc, (InterruptedError, KeyboardInterrupt)) else "FAILED")


def execute(control, reader, ledger, preflight, old, original, authority, manifest, budget,
            *, verify_root, coverage, sources_unchanged):
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
            "preflight": old["preflight"], "continuation": authority}
        store = ArtifactStore(verify_root, binding=binding)
        admitted = checker.precommit_inputs(reader, ledger, coverage, metadata, check=check)
        precommit = fixed_precommit(admitted, old, binding, authority["current_checker"], authority["audit_plan"])
        fixed_precommit(admitted, original, binding, authority["current_checker"], authority["audit_plan"])
        pref = store.publish_json("precommit.json", precommit)
        check(force_resources=True)
        result = checker.verify_all(control, reader, ledger, budget.start_ref, pref, store, coverage, check=check)
        sources_unchanged()
        result_ref = store.publish_json("result.json", result)
        completion = control.publish_json(OUTPUT, {
            "schema": "geometric-decision-audit-profile-order-output-v1",
            "manifest": manifest, "root": str(verify_root), "binding": store.reference(store.path("binding.json")),
            "precommit": pref, "result": result_ref, "continuation": authority})
        return completion, budget.finish("COMPLETE", completion=completion)
    except BaseException as exc:
        close_failure(control, manifest, budget, exc)
        raise


def production_main(wrapper_sha, plan_sha):
    if (Path.cwd().resolve() != ROOT or os.environ.get("CUDA_VISIBLE_DEVICES") != ""
            or any(os.environ.get(k) != "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"))):
        raise ValueError("continuation requires project cwd, hidden CUDA and one-thread CPU")
    reviewed = [approved_source(WRAPPER, wrapper_sha), approved_source(PLAN, plan_sha),
        approved_source(checker.OPERATOR, CHECKER_SHA), approved_source(checker.CORE, CORE_SHA),
        approved_source(checker.PLAN, checker.PLAN_SHA), approved_source(checker.PROTOCOL, checker.PROTOCOL_SHA),
        approved_source(HELPER, HELPER_SHA), *[approved_source(p, sha) for p, sha in REVIEWS.items()],
        approved_source(ACCESS_RECORD, ACCESS_SHA)]
    code = current_checker(reviewed[2:4])
    coverage = core.Coverage()
    reader = core.AuditStore("control", BASE / "control", coverage)
    control = ArtifactStore(reader.root, binding=reader.json(checker.CONTROL_BINDING))
    with control.path("operator.lock").open("ab") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ledger, preflight, old, original = validate_history(control, reader, old_root=OLD_ROOT,
            original_root=ORIGINAL_ROOT, verify_root=VERIFY_ROOT, pins=PINS,
            previous_code=PREVIOUS_CODE, original_code=ORIGINAL_CODE)
        if (old["plan"] != reviewed[4] or old["binding"]["protocol"] != reviewed[5]
                or old["binding"]["continuation"]["access_history"]["record"] != reviewed[-1]
                or old["binding"]["continuation"]["wrapper"] != reviewed[6]):
            raise ValueError("historical plan/protocol/access/helper differs")
        authority = {"schema": "geometric-decision-audit-profile-order-authority-v1",
            "wrapper": reviewed[0], "continuation_plan": reviewed[1], "current_checker": code,
            "previous_checker": PREVIOUS_CODE, "original_checker": ORIGINAL_CODE,
            "audit_plan": reviewed[4], "protocol": reviewed[5], "helper": reviewed[6],
            "reviews": reviewed[7:9], "failed_attempt": {k: PINS[k] for k in ("start", "finish", "manifest", "discrepancy")},
            "previous_precommit": {"root": str(OLD_ROOT), "binding": PINS["binding"], "precommit": PINS["precommit"]},
            "original_precommit": old["binding"]["continuation"]["previous_precommit"],
            "access_history": old["binding"]["continuation"]["access_history"]}

        def unchanged():
            if checker.sources() != code or any(checker.source_reference(r["path"]) != r for r in reviewed):
                raise ValueError("reviewed profile-order sources changed during execution")

        unchanged()
        manifest, budget = start_attempt(control, ledger, authority=authority,
            verify_root=VERIFY_ROOT, started_at=LAUNCH_STARTED)
        previous = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM)}

        def stop(signum, frame):
            if signum == signal.SIGALRM:
                raise BudgetExceeded("profile-order continuation reservation exhausted")
            raise InterruptedError("profile-order continuation interrupted; no retry")

        try:
            for sig in previous:
                signal.signal(sig, stop)
            signal.setitimer(signal.ITIMER_REAL, max(.001, RESERVATION - (time.monotonic() - LAUNCH_STARTED)))
            completion, finish = execute(control, reader, ledger, preflight, old, original, authority,
                manifest, budget, verify_root=VERIFY_ROOT, coverage=coverage, sources_unchanged=unchanged)
            print(json.dumps({"status": "FINAL_TECHNICAL_AUDIT_PROFILE_ORDER_COMPLETE",
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
    args = parser.parse_args()
    production_main(args.approved_wrapper_sha256, args.approved_plan_sha256)
