"""Fixed one-shot VERIFY after attempt 0024; no historical operator is rerun."""
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

from experiments.atencion_armonica.verify_geometric_decision_profile_order import historical_precommit

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "data/atencion_armonica/geometric_decision_energy_v1"
OLD_ROOT = BASE / "audit-final-profile-order-verify"
PRIOR_ROOT = BASE / "audit-final-adapter-verify"
ORIGINAL_ROOT = BASE / "audit-final-verify"
VERIFY_ROOT = BASE / "audit-final-coverage-verify"
WRAPPER = "experiments/atencion_armonica/verify_geometric_decision_coverage.py"
PLAN = "experiments/atencion_armonica/PLAN_GEOMETRIC_DECISION_AUDIT_COVERAGE_CONTINUATION.md"
HELPERS = {
    "experiments/atencion_armonica/continue_geometric_decision_audit.py":
        "2518c217cb6a80ef2c9ecbe715753cb5a97ff942e83228c035bc4ac73e852244",
    "experiments/atencion_armonica/verify_geometric_decision_profile_order.py":
        "692fcc7e43be85fc58ea3adb2ef37cf58f02a027b9e1648574604b0cc8f35c1d",
}
CHECKER_SHA = "aada6600410874e68af3797bd00c525368568a0d2293fa94eb718691a7ec05a9"
CORE_SHA = "ae09edc9507e89e73df74e03b5345c95572601032e966f83de283d44f2d55dd9"
MANIFEST = "manifests/final-technical-audit-coverage.json"
OUTPUT = "outputs/final-technical-audit-coverage.json"
DISCREPANCY = "discrepancies/final-technical-audit-coverage.json"
RESERVATION = 1500.0
PROJECTION_SECONDS = 327.0271509163656
FIXED_FIELDS = ("cuts", "terminal_receipts", "freeze_ref", "seal_ref", "contract_ref", "report_result_ref")
PINS = {key: {"path": path, "bytes": size, "sha256": sha} for key, path, size, sha in (
    ("start", "attempts/0024/start.json", 1320, "cee4a7113942ddf0af01c153afe61c8ce5e4302b8bc6a699d2daddfa8694f8a2"),
    ("finish", "attempts/0024/finish.json", 435, "ddf77779bbbe9d390f5c62d4c82d5b34f7926c2667471c5488b4a1b275131bd5"),
    ("manifest", "manifests/final-technical-audit-profile-order.json", 4432, "0d4691db8baa541ac3b7a9cec57c47010d8dd961a832306cdcae788eb91dcafd"),
    ("discrepancy", "discrepancies/final-technical-audit-profile-order.json", 395, "a588da17b078edf8674764ed07562170f2499367569de029194ea6a90fb72646"),
    ("binding", "binding.json", 7471, "42e71604dfd2633646f67d3ad17d9e239d895fd31a48b60940cebd3621eb54a5"),
    ("precommit", "precommit.json", 21046, "bd5368a55749b58bd31c3765c14e913e58ed28dbd073076f8bbe2103bce80aeb"),
)}
HISTORICAL_CODE = [[
    {"path": checker.OPERATOR, "bytes": size, "sha256": sha},
    {"path": checker.CORE, "bytes": 42824, "sha256": CORE_SHA},
] for size, sha in (
    (133670, "59d872a03635ec354d35f2f9486984b6db4c151095b7a444ec2b17fc71b571a0"),
    (133026, "7b1622f0fc8c78fefe219cd9c9265db708e5cc53fd074fcec7f5e4a2346e2323"),
    (131740, "25e81f578567da757c4a5b071a74fda44fac2ad6df11e27a5ed11f8ed7a3d6cf"),
)]
REVIEWS = {
    "Biblioteca/Geometric_Decision_Energy/agent_reports/787_coverage_namespace_review.md":
        "f9ddd704d4b0491d3a347b1505bba34bd223fb8183da3bf57a64447e99cc2b52",
    "Biblioteca/Geometric_Decision_Energy/agent_reports/788_coverage_continuation_design_review.md":
        "d954f61841a414e3debee5351057976868f7fed44a242f57d5146fba6b917723",
}
ACCESS_RECORD = "Biblioteca/Geometric_Decision_Energy/RESULT_READING_DRAFT_20260915.md"
ACCESS_SHA = "fb6444b9e1af7441929e523a8f0fd2c170b3fd4533257d235ce1fea2e07af602"


def require_absent(control, verify_root, *, next_attempt=25):
    if (verify_root.exists() or verify_root.is_symlink()
            or control.path(f"attempts/{next_attempt:04d}").exists()
            or any(control.path(p).exists() for p in (MANIFEST, OUTPUT, DISCREPANCY))):
        raise ValueError("coverage continuation already attempted; no retry")


def failed_attempt(reader, row, pins, root, schema, exception, message):
    start, finish, manifest, discrepancy = (reader.json(pins[k]) for k in
                                           ("start", "finish", "manifest", "discrepancy"))
    if (row["start_ref"] != pins["start"] or row["start"] != start or start["stage"] != "audit"
            or row["finish_ref"] != pins["finish"] or row["finish"] != finish
            or start["manifest"] != pins["manifest"] or row["manifest"] != manifest
            or finish["start"] != pins["start"] or finish["status"] != "FAILED" or finish["completion"] is not None
            or manifest.get("operation") != "final-technical-audit" or manifest.get("root") != str(root)
            or manifest.get("schema") != schema + "-manifest-v1"
            or discrepancy.get("schema") != schema + "-discrepancy-v1"
            or discrepancy.get("manifest") != pins["manifest"]
            or discrepancy.get("exception_type") != exception or discrepancy.get("message") != message):
        raise ValueError("failed attempt lineage differs")
    return manifest, discrepancy


def validate_history(control, reader, *, roots, verify_root, pins, historical_code, expected_tail=24):
    """Explicit three-cut chain ending at 0024; not a generic retry dispatcher."""
    require_absent(control, verify_root, next_attempt=expected_tail + 1)
    ledger = checker.read_ledger(reader, control.binding)
    if len(ledger["attempts"]) != expected_tail + 1 or len(roots) != 3 or len(historical_code) != 3:
        raise ValueError("exact failed tail and three historical roots required")
    rows = list(reversed(ledger["attempts"][-3:]))
    m24, _ = failed_attempt(reader, rows[0], pins, roots[0], "geometric-decision-audit-profile-order",
        "ValueError", "conflicting identities for profile-head-case:binding.json")
    if ledger["charged"] != rows[0]["finish"]["charged_after"]:
        raise ValueError("ledger charge differs from latest finish")
    b24, p24 = historical_precommit(roots[0], pins, historical_code[0])
    a24 = b24.get("continuation", {})
    if (a24 != m24.get("continuation")
            or a24.get("schema") != "geometric-decision-audit-profile-order-authority-v1"
            or a24.get("current_checker") != historical_code[0]
            or a24.get("previous_checker") != historical_code[1]
            or a24.get("original_checker") != historical_code[2]):
        raise ValueError("0024 authority/checker generations differ")
    refs = [{"root": str(roots[0]), "binding": pins["binding"], "precommit": pins["precommit"]},
            a24["previous_precommit"], a24["original_precommit"]]
    if [r.get("root") for r in refs] != [str(r) for r in roots]:
        raise ValueError("historical root chain differs")
    m23, _ = failed_attempt(reader, rows[1], a24["failed_attempt"], roots[1],
        "geometric-decision-audit-adapter-continuation", "ValueError",
        "closing profile observed inventory roster differs")
    b23, p23 = historical_precommit(roots[1], refs[1], historical_code[1])
    a23 = b23.get("continuation", {})
    if (a23 != m23.get("continuation")
            or a23.get("schema") != "geometric-decision-audit-adapter-continuation-authority-v1"
            or a23.get("current_checker") != historical_code[1]
            or a23.get("historical_checker") != historical_code[2]
            or a23.get("previous_precommit") != refs[2]
            or a23.get("access_history") != a24.get("access_history")):
        raise ValueError("0023 authority or original PRECOMMIT chain differs")
    b22, p22 = historical_precommit(roots[2], refs[2], historical_code[2])
    finish22 = reader.json(a23["failed_finish"])
    start22 = reader.json(finish22["start"])
    pins22 = {"start": finish22["start"], "manifest": start22["manifest"],
              "finish": a23["failed_finish"], "discrepancy": a23["failed_discrepancy"]}
    m22, d22 = failed_attempt(reader, rows[2], pins22, roots[2],
        "geometric-decision-final-audit-recovery", "KeyError", "'root'")
    recovery = b22.get("recovery", {})
    liability = ledger["attempts"][-4]
    incident = reader.json(a23["liability_start"])
    preflight = checker.unique_operation(ledger, "final-technical-audit-preflight")
    if (m22.get("recovery") != recovery or recovery.get("checker") != historical_code[2]
            or recovery.get("verify_root") != str(roots[2])
            or liability["start_ref"] != a23["liability_start"] or liability["start"] != incident
            or liability["finish"] is not None
            or liability["manifest"].get("operation") != "final-technical-audit-launch-incident-accounting"
            or recovery.get("incident_start") != liability["start_ref"]
            or recovery.get("incident_manifest") != liability["start"]["manifest"]
            or d22.get("incident_start") != liability["start_ref"]
            or d22.get("incident_manifest") != liability["start"]["manifest"]
            or recovery.get("preflight_finish") != preflight["finish_ref"]
            or recovery.get("preflight_output") != preflight["finish"]["completion"]
            or preflight["manifest"]["binding"].get("code") != historical_code[2]
            or preflight["manifest"]["binding"].get("control_binding") != checker.binding_ref(control.binding)):
        raise ValueError("original recovery/liability/preflight chain differs")
    expected_preflight = {"finish": preflight["finish_ref"], "output": preflight["finish"]["completion"],
        "admission": preflight["output"]["admission"], "projection": preflight["output"]["projection"]}
    cuts = [p24, p23, p22]
    if [p.get("result_payloads_opened") for p in cuts] != [True, True, False]:
        raise ValueError("historical result access declarations differ")
    for binding, authority in ((b24, a24), (b23, a23)):
        if (binding["plan"] != authority.get("audit_plan") or binding["protocol"] != authority.get("protocol")
                or authority.get("access_history", {}).get("result_payloads_opened") is not True):
            raise ValueError("continuation plan/protocol/access differs")
    for p in cuts:
        if (p["preflight"] != expected_preflight or p["binding"]["protocol"] != b24["protocol"]
                or any(core.encoded(p[k]) != core.encoded(p24[k]) for k in (*FIXED_FIELDS, "preflight", "plan"))):
            raise ValueError("three historical cuts or their authority differ")
    if (preflight["manifest"]["binding"].get("plan") != b24["plan"]
            or preflight["manifest"]["binding"].get("protocol") != b24["protocol"]):
        raise ValueError("preflight plan/protocol differs")
    return ledger, preflight, cuts, refs


def start_attempt(control, ledger, *, authority, verify_root, started_at,
                  reservation=RESERVATION, projected_seconds=PROJECTION_SECONDS, next_attempt=25):
    require_absent(control, verify_root, next_attempt=next_attempt)
    if len(ledger["attempts"]) != next_attempt:
        raise ValueError("new attempt must immediately follow validated tail")
    available = min(STAGES["audit"] - ledger["charged"]["audit"],
                    LIMITS["total_seconds"] - sum(ledger["charged"].values()))
    if (not math.isfinite(reservation) or not 0 < reservation <= RESERVATION
            or not math.isfinite(projected_seconds) or not 0 < projected_seconds <= reservation
            or reservation > available):
        raise BudgetExceeded("coverage continuation does not fit fixed ledger budget")
    manifest = control.publish_json(MANIFEST, {
        "schema": "geometric-decision-audit-coverage-manifest-v1",
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
            "schema": "geometric-decision-audit-coverage-discrepancy-v1",
            "manifest": manifest, "exception_type": type(exc).__name__, "message": str(exc),
            "authority": "raw technical discrepancy; no scientific conclusion"})
    finally:
        if not budget.closed:
            budget.finish("LIMIT_REACHED" if isinstance(exc, BudgetExceeded) else "PAUSED"
                          if isinstance(exc, (InterruptedError, KeyboardInterrupt)) else "FAILED")


def execute(control, reader, ledger, preflight, cuts, authority, manifest, budget,
            *, verify_root, coverage, sources_unchanged):
    try:
        old = cuts[0]
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
        for prior in cuts[1:]:
            fixed_precommit(admitted, prior, binding, authority["current_checker"], authority["audit_plan"])
        pref = store.publish_json("precommit.json", precommit)
        check(force_resources=True)
        result = checker.verify_all(control, reader, ledger, budget.start_ref, pref, store, coverage, check=check)
        sources_unchanged()
        result_ref = store.publish_json("result.json", result)
        completion = control.publish_json(OUTPUT, {
            "schema": "geometric-decision-audit-coverage-output-v1",
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
        *[approved_source(p, sha) for p, sha in HELPERS.items()],
        *[approved_source(p, sha) for p, sha in REVIEWS.items()], approved_source(ACCESS_RECORD, ACCESS_SHA)]
    code = current_checker(reviewed[2:4])
    coverage = core.Coverage()
    reader = core.AuditStore("control", BASE / "control", coverage)
    control = ArtifactStore(reader.root, binding=reader.json(checker.CONTROL_BINDING))
    with control.path("operator.lock").open("ab") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ledger, preflight, cuts, refs = validate_history(control, reader,
            roots=[OLD_ROOT, PRIOR_ROOT, ORIGINAL_ROOT], verify_root=VERIFY_ROOT,
            pins=PINS, historical_code=HISTORICAL_CODE)
        old = cuts[0]["binding"]
        historical = old["continuation"]
        if (old["plan"] != reviewed[4] or old["protocol"] != reviewed[5]
                or historical["access_history"]["record"] != reviewed[-1]
                or historical["wrapper"] != reviewed[7] or historical["helper"] != reviewed[6]):
            raise ValueError("historical plan/protocol/access/helpers differ")
        authority = {"schema": "geometric-decision-audit-coverage-authority-v1",
            "wrapper": reviewed[0], "continuation_plan": reviewed[1], "current_checker": code,
            "checker_0024": HISTORICAL_CODE[0], "checker_0023": HISTORICAL_CODE[1],
            "checker_0022": HISTORICAL_CODE[2], "audit_plan": reviewed[4], "protocol": reviewed[5],
            "helpers": reviewed[6:8], "reviews": reviewed[8:10],
            "failed_attempt": {k: PINS[k] for k in ("start", "finish", "manifest", "discrepancy")},
            "historical_precommits": refs, "access_history": historical["access_history"]}

        def unchanged():
            if checker.sources() != code or any(checker.source_reference(r["path"]) != r for r in reviewed):
                raise ValueError("reviewed coverage sources changed during execution")

        unchanged()
        manifest, budget = start_attempt(control, ledger, authority=authority,
            verify_root=VERIFY_ROOT, started_at=LAUNCH_STARTED)
        previous = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM)}

        def stop(signum, frame):
            if signum == signal.SIGALRM:
                raise BudgetExceeded("coverage continuation reservation exhausted")
            raise InterruptedError("coverage continuation interrupted; no retry")

        try:
            for sig in previous:
                signal.signal(sig, stop)
            signal.setitimer(signal.ITIMER_REAL, max(.001, RESERVATION - (time.monotonic() - LAUNCH_STARTED)))
            completion, finish = execute(control, reader, ledger, preflight, cuts, authority,
                manifest, budget, verify_root=VERIFY_ROOT, coverage=coverage, sources_unchanged=unchanged)
            print(json.dumps({"status": "FINAL_TECHNICAL_AUDIT_COVERAGE_COMPLETE",
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
