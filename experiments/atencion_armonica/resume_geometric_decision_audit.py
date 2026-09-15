"""One-shot recovery wrapper for the geometric-decision final audit.

This module does not replace the reviewed checker.  It records the unmeasured
launch incident conservatively, starts a new bounded control attempt, and then
calls the published PRECOMMIT/VERIFY functions in a sibling output store.
"""
from __future__ import annotations

import time
LAUNCH_STARTED = time.monotonic()

import argparse
from dataclasses import dataclass
import fcntl
import json
import math
import os
from pathlib import Path
import signal
from typing import Any, Callable

from experiments.atencion_armonica import audit_geometric_decision as checker
from experiments.atencion_armonica import audit_geometric_decision_core as core
from src.atencion_armonica.geometric_decision_budget import (
    LIMITS, STAGES as BUDGET_STAGES, BudgetExceeded, StageBudget,
)
from src.atencion_armonica.geometric_decision_store import ArtifactStore


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "data/atencion_armonica/geometric_decision_energy_v1"
CONTROL_ROOT = BASE / "control"
AUDIT_PARENT = BASE / "audit-final"
PREFLIGHT_ROOT = AUDIT_PARENT / "preflight"
VERIFY_ROOT = BASE / "audit-final-verify"
WRAPPER = "experiments/atencion_armonica/resume_geometric_decision_audit.py"
RECOVERY_PLAN = "experiments/atencion_armonica/PLAN_GEOMETRIC_DECISION_AUDIT_RECOVERY.md"
INCIDENT_MANIFEST_PATH = "manifests/final-technical-audit-launch-incident.json"
RECOVERY_MANIFEST_PATH = "manifests/final-technical-audit-recovery.json"
RECOVERY_OUTPUT_PATH = "outputs/final-technical-audit-recovery.json"
RECOVERY_DISCREPANCY_PATH = "discrepancies/final-technical-audit-recovery.json"
RECOVERY_RESERVATION = 1500.0
EXPECTED_PROJECTION_SECONDS = 327.0271509163656
EXPECTED_PREFLIGHT_FINISH = {
    "path": "attempts/0020/finish.json", "bytes": 574,
    "sha256": "e1711560e68acec40a19990c5ce0e61c3d8f7c847889e2d9a374220f7daf709c",
}
EXPECTED_PREFLIGHT_OUTPUT = {
    "path": "outputs/final-technical-audit-preflight.json", "bytes": 649,
    "sha256": "90769c9563a4cafd0814afd4c8ba30ef142f05e7277b5c0ec2ccc8b7721b0d28",
}
EXPECTED_REPORT_FINISH = {
    "path": "attempts/0019/finish.json", "bytes": 551,
    "sha256": "5cb26fb2873387ba3e7f7021f75f29f7a8d1b011c03fb97bf7f9fe4e4682dc64",
}


@dataclass
class RecoveryStart:
    control: ArtifactStore
    control_read: core.AuditStore
    ledger: dict[str, Any]
    preflight: dict[str, Any]
    authority: dict[str, Any]
    incident_manifest: dict[str, Any]
    incident_start: dict[str, Any]
    manifest: dict[str, Any]
    budget: StageBudget
    verify_root: Path


def source_reference(relative: str) -> dict[str, Any]:
    path = ROOT / relative
    digest, size = core.sha256_path(path)
    return {"path": relative, "sha256": digest, "bytes": size}


def require_sha(value: str, label: str) -> str:
    if (type(value) is not str or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)):
        raise ValueError(label + " requires a lowercase sha256")
    return value


def _absent(control: ArtifactStore, relatives: tuple[str, ...], verify_root: Path) -> None:
    present = [relative for relative in relatives if control.path(relative).exists()]
    if present or verify_root.exists():
        raise ValueError("final audit recovery already attempted; preserve receipts and do not retry")


def validate_initial_state(
    control: ArtifactStore,
    control_read: core.AuditStore,
    *,
    audit_parent: Path,
    preflight_root: Path,
    verify_root: Path,
    expected_preflight_finish: dict[str, Any],
    expected_preflight_output: dict[str, Any],
    expected_report_finish: dict[str, Any],
    expected_checker: list[dict[str, Any]],
    expected_plan: dict[str, Any],
    expected_protocol: dict[str, Any],
    expected_control_binding: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Authenticate only terminal metadata and the preserved parent topology."""
    _absent(control, (
        "manifests/final-technical-audit.json", INCIDENT_MANIFEST_PATH,
        RECOVERY_MANIFEST_PATH, RECOVERY_OUTPUT_PATH, RECOVERY_DISCREPANCY_PATH,
    ), verify_root)
    ledger = checker.read_ledger(control_read, control.binding)
    preflight = checker.unique_operation(ledger, "final-technical-audit-preflight")
    report = checker.unique_operation(ledger, "post-replay-report")
    if (preflight["finish_ref"] != expected_preflight_finish
            or preflight["finish"]["completion"] != expected_preflight_output
            or report["finish_ref"] != expected_report_finish
            or ledger["attempts"][-1] is not preflight):
        raise ValueError("terminal preflight/report authority differs")
    p_binding = preflight["manifest"].get("binding")
    if (not isinstance(p_binding, dict)
            or p_binding.get("code") != expected_checker
            or p_binding.get("plan") != expected_plan
            or p_binding.get("protocol") != expected_protocol
            or p_binding.get("control_binding") != expected_control_binding
            or p_binding.get("maximum_combined_seconds") != checker.RESERVATION
            or preflight["output"].get("root") != str(preflight_root)):
        raise ValueError("preflight binding or root differs")
    expected_output = control_read.json(expected_preflight_output)
    if preflight["output"] != expected_output:
        raise ValueError("preflight completion differs")
    if (not audit_parent.is_dir() or audit_parent.is_symlink()
            or {path.name for path in audit_parent.iterdir()} != {preflight_root.name}
            or not preflight_root.is_dir() or preflight_root.is_symlink()
            or {path.name for path in preflight_root.iterdir()}
                != {"binding.json", "admission.json", "projection.json"}
            or (audit_parent / "binding.json").exists()):
        raise ValueError("preserved unbound audit parent topology differs")
    return ledger, preflight


def prospective_budget(ledger: dict[str, Any], preflight_seconds: float,
                       projected_seconds: float, reservation: float) -> dict[str, float]:
    if (type(preflight_seconds) not in (int, float) or not math.isfinite(preflight_seconds)
            or not 0 <= preflight_seconds < checker.RESERVATION
            or type(projected_seconds) not in (int, float) or not math.isfinite(projected_seconds)
            or projected_seconds <= 0 or type(reservation) not in (int, float)
            or not 0 < reservation <= RECOVERY_RESERVATION):
        raise ValueError("recovery timing authority differs")
    liability = checker.RESERVATION - preflight_seconds
    audit_after = ledger["charged"]["audit"] + liability
    total_after = sum(ledger["charged"].values()) + liability
    audit_remaining = BUDGET_STAGES["audit"] - audit_after
    global_remaining = LIMITS["total_seconds"] - total_after
    available = min(audit_remaining, global_remaining)
    if liability <= 0 or reservation > available or projected_seconds > reservation:
        raise BudgetExceeded("recovery liability/projection does not fit fixed audit/global budget")
    return {"liability_seconds": liability, "audit_after_liability": audit_after,
            "total_after_liability": total_after, "audit_remaining": audit_remaining,
            "global_remaining": global_remaining, "reservation_seconds": reservation,
            "projected_seconds": projected_seconds}


def publish_liability(
    control: ArtifactStore,
    control_read: core.AuditStore,
    ledger: dict[str, Any],
    preflight: dict[str, Any],
    *,
    authority: dict[str, Any],
    accounting: dict[str, float],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Publish an explicit unmeasured liability as a start with no finish."""
    manifest = control.publish_json(INCIDENT_MANIFEST_PATH, {
        "schema": "geometric-decision-final-audit-launch-incident-v1",
        "operation": "final-technical-audit-launch-incident-accounting",
        "authority": "accounting liability; not measured elapsed time or a historical scientific execution",
        "failure": {"exception_type": "ValueError",
                    "message": "cannot adopt existing unbound artifacts",
                    "source": "experiments/atencion_armonica/audit_geometric_decision.py:1877"},
        "preflight_finish": preflight["finish_ref"],
        "preflight_output": preflight["finish"]["completion"],
        "recovery": authority,
        "accounting": accounting,
    })
    index = len(ledger["attempts"])
    start = control.publish_json(f"attempts/{index:04d}/start.json", {
        "schema": "geometric-decision-attempt-v1", "binding": control.binding,
        "manifest": manifest, "stage": "audit",
        "reservation_seconds": accounting["liability_seconds"],
        "charged_before": ledger["charged"],
    })
    replayed = checker.read_ledger(control_read, control.binding)
    record = replayed["attempts"][-1]
    expected_audit = ledger["charged"]["audit"] + accounting["liability_seconds"]
    if (record["start_ref"] != start or record["finish"] is not None
            or replayed["charged"]["audit"] != expected_audit):
        raise ValueError("incident liability was not charged exactly once")
    return manifest, start, replayed


def begin_recovery(
    control: ArtifactStore,
    control_read: core.AuditStore,
    ledger: dict[str, Any],
    preflight: dict[str, Any],
    *,
    authority: dict[str, Any],
    verify_root: Path,
    started_at: float,
    projected_seconds: float = EXPECTED_PROJECTION_SECONDS,
    reservation: float = RECOVERY_RESERVATION,
) -> RecoveryStart:
    accounting = prospective_budget(ledger, preflight["finish"]["seconds"],
                                    projected_seconds, reservation)
    incident_manifest, incident_start, amended = publish_liability(
        control, control_read, ledger, preflight, authority=authority,
        accounting=accounting)
    recovery_authority = {**authority, "incident_manifest": incident_manifest,
                          "incident_start": incident_start,
                          "verify_root": str(verify_root), "accounting": accounting}
    manifest = control.publish_json(RECOVERY_MANIFEST_PATH, {
        "schema": "geometric-decision-final-audit-recovery-manifest-v1",
        "operation": "final-technical-audit", "root": str(verify_root),
        "recovery": recovery_authority, "reservation_seconds": reservation,
    })
    budget = StageBudget(control, "audit", manifest_ref=manifest,
        reservation_seconds=reservation, started_at=started_at,
        prior_charges=control.binding["prior_charges"],
        output_roots=[Path(path) for path in control.binding["output_roots"]])
    return RecoveryStart(control, control_read, amended, preflight,
                         recovery_authority, incident_manifest, incident_start,
                         manifest, budget, verify_root)


def close_failure(start: RecoveryStart, exc: BaseException) -> None:
    try:
        start.control.publish_json(RECOVERY_DISCREPANCY_PATH, {
            "schema": "geometric-decision-final-audit-recovery-discrepancy-v1",
            "exception_type": type(exc).__name__, "message": str(exc),
            "incident_manifest": start.incident_manifest,
            "incident_start": start.incident_start,
            "manifest": start.manifest,
            "authority": "raw technical discrepancy; no scientific result",
        })
    finally:
        if not start.budget.closed:
            status = ("LIMIT_REACHED" if isinstance(exc, BudgetExceeded) else "PAUSED"
                      if isinstance(exc, (InterruptedError, KeyboardInterrupt)) else "FAILED")
            start.budget.finish(status)


def open_store_and_run(start: RecoveryStart, binding: dict[str, Any],
                       body: Callable[[ArtifactStore], dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Integrated boundary used by production and real-store startup fixtures."""
    try:
        store = ArtifactStore(start.verify_root, binding=binding)
        completion = body(store)
        finish = start.budget.finish("COMPLETE", completion=completion)
        return completion, finish
    except BaseException as exc:
        close_failure(start, exc)
        raise


def authenticate_preflight(start: RecoveryStart, *, check: Callable[[], None]) -> tuple[dict[str, Any], dict[str, Any]]:
    output = start.preflight["output"]
    store = checker.open_bound("audit-preflight-recovery", Path(output["root"]),
                               start.preflight["manifest"]["binding"],
                               core.Coverage(), check=check)
    metadata = store.json(output["admission"], check=check)
    projection = store.json(output["projection"], check=check)
    if (projection.get("schema") != "geometric-decision-final-audit-projection-v1"
            or projection.get("projected_seconds") != start.authority["accounting"]["projected_seconds"]):
        raise ValueError("authenticated recovery projection differs")
    for name in checker.REQUIRED_COMPLETE:
        if metadata.get("terminal_receipts", {}).get(name) != checker.unique_operation(start.ledger, name)["finish_ref"]:
            raise ValueError("preflight terminal receipt differs: " + name)
    return metadata, projection


def production_main(approved_wrapper_sha256: str, approved_recovery_plan_sha256: str) -> None:
    if (Path.cwd().resolve() != ROOT or os.environ.get("CUDA_VISIBLE_DEVICES") != ""
            or any(os.environ.get(key) != "1" for key in
                   ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"))):
        raise ValueError("audit recovery requires project cwd, hidden CUDA and one-thread CPU")
    wrapper_ref = source_reference(WRAPPER)
    recovery_plan_ref = source_reference(RECOVERY_PLAN)
    if (wrapper_ref["sha256"] != require_sha(approved_wrapper_sha256, "wrapper")
            or recovery_plan_ref["sha256"] != require_sha(approved_recovery_plan_sha256, "recovery plan")):
        raise ValueError("wrapper or recovery plan differs from reviewed source")
    plan_ref, protocol_ref, code = (checker.source_reference(checker.PLAN),
                                    checker.source_reference(checker.PROTOCOL), checker.sources())
    if plan_ref["sha256"] != checker.PLAN_SHA or protocol_ref["sha256"] != checker.PROTOCOL_SHA:
        raise ValueError("accepted audit plan/protocol changed")
    coverage = core.Coverage()
    control_read = core.AuditStore("control", CONTROL_ROOT, coverage)
    binding = control_read.json(checker.CONTROL_BINDING)
    control = ArtifactStore(CONTROL_ROOT, binding=binding)
    with control.path("operator.lock").open("ab") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ledger, preflight = validate_initial_state(control, control_read,
            audit_parent=AUDIT_PARENT, preflight_root=PREFLIGHT_ROOT,
            verify_root=VERIFY_ROOT,
            expected_preflight_finish=EXPECTED_PREFLIGHT_FINISH,
            expected_preflight_output=EXPECTED_PREFLIGHT_OUTPUT,
            expected_report_finish=EXPECTED_REPORT_FINISH,
            expected_checker=code, expected_plan=plan_ref, expected_protocol=protocol_ref,
            expected_control_binding=checker.CONTROL_BINDING)
        authority = {"schema": "geometric-decision-final-audit-recovery-authority-v1",
            "wrapper": wrapper_ref, "recovery_plan": recovery_plan_ref,
            "checker": code, "plan": plan_ref, "protocol": protocol_ref,
            "preflight_finish": preflight["finish_ref"],
            "preflight_output": preflight["finish"]["completion"]}
        start = begin_recovery(control, control_read, ledger, preflight,
            authority=authority, verify_root=VERIFY_ROOT, started_at=LAUNCH_STARTED)
        old = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM)}

        def stop(signum, frame):
            if signum == signal.SIGALRM:
                raise BudgetExceeded("final audit recovery reservation exhausted")
            raise InterruptedError("final audit recovery interrupted; no retry")

        try:
            for sig in old:
                signal.signal(sig, stop)
            signal.setitimer(signal.ITIMER_REAL,
                max(0.001, start.budget.reservation - (time.monotonic() - LAUNCH_STARTED)))
            start.budget.check()
            metadata, _ = authenticate_preflight(start, check=start.budget.check)
            audit_binding = {"schema": "geometric-decision-final-audit-binding-v1",
                "plan": plan_ref, "code": code, "protocol": protocol_ref,
                "inputs": metadata["terminal_receipts"],
                "preflight": {"finish": preflight["finish_ref"],
                    "output": preflight["finish"]["completion"],
                    "admission": preflight["output"]["admission"],
                    "projection": preflight["output"]["projection"]},
                "recovery": start.authority}

            def verify_body(audit_store: ArtifactStore) -> dict[str, Any]:
                admitted = checker.precommit_inputs(control_read, start.ledger, coverage,
                                                     metadata, check=start.budget.check)
                precommit = {"schema": "geometric-decision-final-audit-precommit-v1",
                    "binding": audit_binding, "plan": plan_ref, "checker": code,
                    "terminal_receipts": admitted["terminal_receipts"],
                    "preflight": audit_binding["preflight"],
                    "freeze_ref": admitted["freeze_ref"], "seal_ref": admitted["seal_ref"],
                    "contract_ref": admitted["contract_ref"],
                    "report_result_ref": admitted["report_result_ref"],
                    "cuts": admitted["cuts"], "result_payloads_opened": False}
                precommit_ref = audit_store.publish_json("precommit.json", precommit)
                start.budget.check(force_resources=True)
                result = checker.verify_all(control, control_read, start.ledger,
                    start.budget.start_ref, precommit_ref, audit_store, coverage,
                    check=start.budget.check)
                if (checker.sources() != code or checker.source_reference(checker.PLAN) != plan_ref
                        or source_reference(WRAPPER) != wrapper_ref
                        or source_reference(RECOVERY_PLAN) != recovery_plan_ref):
                    raise ValueError("checker, plan, wrapper or recovery plan changed during recovery")
                result_ref = audit_store.publish_json("result.json", result)
                return control.publish_json(RECOVERY_OUTPUT_PATH, {
                    "manifest": start.manifest, "root": str(VERIFY_ROOT),
                    "binding": audit_store.reference(audit_store.path("binding.json")),
                    "precommit": precommit_ref, "result": result_ref,
                    "recovery": start.authority})

            completion, finish = open_store_and_run(start, audit_binding, verify_body)
            print(json.dumps({"status": "FINAL_TECHNICAL_AUDIT_RECOVERY_COMPLETE",
                              "finish": finish, "completion": completion}), flush=True)
        except BaseException as exc:
            if not start.budget.closed:
                close_failure(start, exc)
            raise
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            for sig, handler in old.items():
                signal.signal(sig, handler)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--approved-wrapper-sha256", required=True)
    parser.add_argument("--approved-recovery-plan-sha256", required=True)
    args = parser.parse_args()
    production_main(args.approved_wrapper_sha256, args.approved_recovery_plan_sha256)


if __name__ == "__main__":
    main()
