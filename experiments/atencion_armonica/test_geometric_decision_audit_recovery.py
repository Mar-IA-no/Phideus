"""Abstract recovery fixtures; never opens campaign data or scientific outputs."""
from __future__ import annotations

import time
from pathlib import Path
import uuid

import pytest

from experiments.atencion_armonica import audit_geometric_decision as checker
from experiments.atencion_armonica import audit_geometric_decision_core as core
from experiments.atencion_armonica import resume_geometric_decision_audit as recovery
from src.atencion_armonica.geometric_decision_budget import LIMITS, StageBudget
from src.atencion_armonica.geometric_decision_store import ArtifactStore


TEST_ROOT = (Path(__file__).resolve().parents[2]
             / ".agent-work/phideus-geometric-decision-20260914/audit778")


@pytest.fixture
def sandbox() -> Path:
    TEST_ROOT.mkdir(parents=True, exist_ok=True)
    root = TEST_ROOT / ("fixture-" + uuid.uuid4().hex)
    root.mkdir()
    return root


def _complete(control: ArtifactStore, stage: str, manifest: dict, output: dict,
              *, reservation: float = 20.0) -> tuple[dict, dict]:
    budget = StageBudget(control, stage, manifest_ref=manifest,
        reservation_seconds=reservation, prior_charges=control.binding["prior_charges"],
        output_roots=[Path(path) for path in control.binding["output_roots"]])
    completion = control.publish_json(output["path"],
                                      {"manifest": manifest, **output["value"]})
    finish = budget.finish("COMPLETE", completion=completion)
    return completion, finish


def fixture_state(root: Path, *, real_authority: bool = False) -> dict:
    campaign, work = root / "campaign", root / "work"
    campaign.mkdir(); work.mkdir()
    control_root = campaign / "control"
    binding = {"schema": "geometric-decision-control-v1", "limits": LIMITS,
        "prior_charges": [], "protocol": {"path": "fixture", "sha256": "0" * 64},
        "output_roots": [str(campaign.resolve()), str(work.resolve())]}
    control = ArtifactStore(control_root, binding=binding)
    control_read = core.AuditStore("fixture-control", control_root, core.Coverage())
    control_binding = control.reference(control.path("binding.json"))
    abstract = {"path": "fixture", "sha256": "1" * 64, "bytes": 1}
    code = checker.sources() if real_authority else [abstract]
    plan = checker.source_reference(checker.PLAN) if real_authority else abstract
    protocol = checker.source_reference(checker.PROTOCOL) if real_authority else abstract
    report_manifest = control.publish_json("manifests/report.json", {
        "operation": "post-replay-report", "fixture": True})
    _, report_finish = _complete(control, "audit", report_manifest,
        {"path": "outputs/report.json", "value": {"root": str(campaign / "report")}})

    audit_parent = campaign / "audit-final"
    preflight_root = audit_parent / "preflight"
    preflight_binding = {"schema": "geometric-decision-final-audit-preflight-binding-v1",
        "code": code, "plan": plan, "protocol": protocol,
        "control_binding": control_binding, "maximum_combined_seconds": 1500.0}
    preflight_store = ArtifactStore(preflight_root, binding=preflight_binding)
    admission = preflight_store.publish_json("admission.json", {
        "terminal_receipts": {"post-replay-report": report_finish}})
    projection = preflight_store.publish_json("projection.json", {
        "schema": "geometric-decision-final-audit-projection-v1",
        "projected_seconds": recovery.EXPECTED_PROJECTION_SECONDS})
    preflight_manifest = control.publish_json("manifests/final-technical-audit-preflight.json", {
        "operation": "final-technical-audit-preflight", "root": str(preflight_root),
        "binding": preflight_binding, "reservation_seconds": 60.0})
    preflight_output, preflight_finish = _complete(control, "audit", preflight_manifest,
        {"path": "outputs/final-technical-audit-preflight.json", "value": {
            "root": str(preflight_root),
            "binding": preflight_store.reference(preflight_store.path("binding.json")),
            "admission": admission, "projection": projection}}, reservation=60.0)
    verify_root = campaign / "audit-final-verify"
    ledger, preflight = recovery.validate_initial_state(control, control_read,
        audit_parent=audit_parent, preflight_root=preflight_root, verify_root=verify_root,
        expected_preflight_finish=preflight_finish,
        expected_preflight_output=preflight_output,
        expected_report_finish=report_finish, expected_checker=code,
        expected_plan=plan, expected_protocol=protocol,
        expected_control_binding=control_binding)
    authority = {"schema": "fixture-recovery-authority-v1", "checker": [abstract],
                 "wrapper": abstract, "recovery_plan": abstract}
    return {"campaign": campaign, "control": control, "control_read": control_read,
            "audit_parent": audit_parent, "preflight_root": preflight_root,
            "verify_root": verify_root, "ledger": ledger, "preflight": preflight,
            "authority": authority, "preflight_finish": preflight_finish,
            "preflight_output": preflight_output, "report_finish": report_finish,
            "abstract": abstract, "code": code, "plan": plan, "protocol": protocol,
            "control_binding": control_binding}


def test_real_store_budget_start_preserves_parent_and_completes_sibling(sandbox):
    state = fixture_state(sandbox)
    launched = time.monotonic()
    start = recovery.begin_recovery(state["control"], state["control_read"],
        state["ledger"], state["preflight"], authority=state["authority"],
        verify_root=state["verify_root"], started_at=launched)
    recovery_start_path = state["control"].path(start.budget.start_ref["path"])
    assert recovery_start_path.exists()
    assert not state["verify_root"].exists()
    assert not (state["audit_parent"] / "binding.json").exists()
    liability_record = checker.read_ledger(state["control_read"],
                                           state["control"].binding)["attempts"][-2]
    assert liability_record["finish"] is None
    assert liability_record["start"]["reservation_seconds"] == (
        checker.RESERVATION - state["preflight"]["finish"]["seconds"])

    def body(store: ArtifactStore) -> dict:
        assert recovery_start_path.exists()
        result = store.publish_json("abstract-result.json", {"status": "COMPLETE"})
        return state["control"].publish_json(recovery.RECOVERY_OUTPUT_PATH, {
            "manifest": start.manifest, "root": str(store.root), "result": result})

    completion, finish = recovery.open_store_and_run(start,
        {"schema": "fixture-final-audit-binding-v1", "recovery": start.authority}, body)
    assert state["control"].json(finish)["status"] == "COMPLETE"
    assert state["control"].json(finish)["completion"] == completion
    assert (state["verify_root"] / "binding.json").is_file()
    assert not (state["audit_parent"] / "binding.json").exists()
    assert {path.name for path in state["audit_parent"].iterdir()} == {"preflight"}


def test_original_parent_adoption_fails_but_recovery_constructor_failure_is_charged(sandbox):
    state = fixture_state(sandbox)
    with pytest.raises(ValueError, match="cannot adopt existing unbound artifacts"):
        ArtifactStore(state["audit_parent"], binding={"schema": "wrong-parent"})
    start = recovery.begin_recovery(state["control"], state["control_read"],
        state["ledger"], state["preflight"], authority=state["authority"],
        verify_root=state["verify_root"], started_at=time.monotonic())
    state["verify_root"].mkdir()
    (state["verify_root"] / "stray").write_bytes(b"preserved")
    with pytest.raises(ValueError, match="cannot adopt existing unbound artifacts"):
        recovery.open_store_and_run(start,
            {"schema": "fixture-final-audit-binding-v1", "recovery": start.authority},
            lambda store: pytest.fail("body must not run"))
    finish = state["control"].json(state["control"].reference(
        state["control"].path(start.budget.folder + "/finish.json")))
    assert finish["status"] == "FAILED"
    assert finish["seconds"] >= 0
    assert state["control"].path(recovery.RECOVERY_DISCREPANCY_PATH).is_file()
    assert (state["verify_root"] / "stray").read_bytes() == b"preserved"
    with pytest.raises(ValueError, match="already attempted"):
        recovery.validate_initial_state(state["control"], state["control_read"],
            audit_parent=state["audit_parent"], preflight_root=state["preflight_root"],
            verify_root=state["verify_root"],
            expected_preflight_finish=state["preflight_finish"],
            expected_preflight_output=state["preflight_output"],
            expected_report_finish=state["report_finish"],
            expected_checker=[state["abstract"]], expected_plan=state["abstract"],
            expected_protocol=state["abstract"],
            expected_control_binding=state["control_binding"])


def test_production_accounting_numbers_and_caps_are_not_reset():
    charged = {"audit": 144.8056539868703, "evaluation": 2530.971776777995,
        "fresh": 16382.972316034953, "open": 54.502750669955276,
        "profile": 215.63404327409808, "training": 4912.186859244946}
    result = recovery.prospective_budget({"charged": charged}, 19.62826516793575,
                                         327.0271509163656, 1500.0)
    assert result == {"liability_seconds": 1480.3717348320642,
        "audit_after_liability": 1625.1773888189346,
        "total_after_liability": 25721.445134820882,
        "audit_remaining": 1974.8226111810654,
        "global_remaining": 23478.554865179118,
        "reservation_seconds": 1500.0, "projected_seconds": 327.0271509163656}
    with pytest.raises(checker.BudgetExceeded, match="fixed audit/global"):
        recovery.prospective_budget({"charged": {**charged, "audit": 3599.0}},
                                    19.62826516793575, 327.0271509163656, 1500.0)


def test_wrapper_hash_requires_exact_lowercase_sha():
    assert recovery.require_sha("a" * 64, "wrapper") == "a" * 64
    for value in ("a" * 63, "A" * 64, True):
        with pytest.raises(ValueError, match="lowercase sha256"):
            recovery.require_sha(value, "wrapper")


def test_preflight_receipt_and_payload_mutations_are_rejected(sandbox):
    state = fixture_state(sandbox)
    bad_finish = {**state["preflight_finish"], "sha256": "0" * 64}
    with pytest.raises(ValueError, match="terminal preflight/report"):
        recovery.validate_initial_state(state["control"], state["control_read"],
            audit_parent=state["audit_parent"], preflight_root=state["preflight_root"],
            verify_root=state["verify_root"], expected_preflight_finish=bad_finish,
            expected_preflight_output=state["preflight_output"],
            expected_report_finish=state["report_finish"], expected_checker=state["code"],
            expected_plan=state["plan"], expected_protocol=state["protocol"],
            expected_control_binding=state["control_binding"])

    start = recovery.begin_recovery(state["control"], state["control_read"],
        state["ledger"], state["preflight"], authority=state["authority"],
        verify_root=state["verify_root"], started_at=time.monotonic())
    projection_path = state["preflight_root"] / state["preflight"]["output"]["projection"]["path"]
    projection_path.write_bytes(b"{}\n")
    with pytest.raises(ValueError, match="authenticated bytes changed") as caught:
        recovery.authenticate_preflight(start, check=start.budget.check)
    recovery.close_failure(start, caught.value)
    finish_path = state["control"].path(start.budget.folder + "/finish.json")
    assert state["control"].json(state["control"].reference(finish_path))["status"] == "FAILED"


def test_production_main_orders_reused_preflight_precommit_verify_with_real_startup(
        sandbox, monkeypatch, capsys):
    state = fixture_state(sandbox, real_authority=True)
    wrapper_ref = recovery.source_reference(recovery.WRAPPER)
    recovery_plan_ref = recovery.source_reference(recovery.RECOVERY_PLAN)
    monkeypatch.setattr(recovery, "CONTROL_ROOT", state["control"].root)
    monkeypatch.setattr(recovery, "AUDIT_PARENT", state["audit_parent"])
    monkeypatch.setattr(recovery, "PREFLIGHT_ROOT", state["preflight_root"])
    monkeypatch.setattr(recovery, "VERIFY_ROOT", state["verify_root"])
    monkeypatch.setattr(recovery, "EXPECTED_PREFLIGHT_FINISH", state["preflight_finish"])
    monkeypatch.setattr(recovery, "EXPECTED_PREFLIGHT_OUTPUT", state["preflight_output"])
    monkeypatch.setattr(recovery, "EXPECTED_REPORT_FINISH", state["report_finish"])
    monkeypatch.setattr(recovery, "LAUNCH_STARTED", time.monotonic())
    monkeypatch.setattr(checker, "CONTROL_BINDING", state["control_binding"])
    monkeypatch.setattr(checker, "REQUIRED_COMPLETE", ("post-replay-report",))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        monkeypatch.setenv(name, "1")
    events = []
    authenticate = recovery.authenticate_preflight

    def observed_authenticate(start, *, check):
        result = authenticate(start, check=check)
        events.append("preflight-reused")
        return result

    def abstract_precommit(control_read, ledger, coverage, metadata, *, check):
        check(); events.append("precommit")
        return {**metadata, "freeze_ref": state["abstract"],
            "seal_ref": state["abstract"], "contract_ref": state["abstract"],
            "report_result_ref": state["abstract"], "cuts": []}

    def abstract_verify(control, control_read, ledger, active_start, precommit_ref,
                        audit_store, coverage, *, check):
        check()
        precommit = audit_store.json(precommit_ref)
        assert precommit["result_payloads_opened"] is False
        assert precommit["checker"] == state["code"]
        events.append("verify")
        return {"schema": "geometric-decision-final-audit-result-v1",
                "status": "COMPLETE", "fixture": True}

    monkeypatch.setattr(recovery, "authenticate_preflight", observed_authenticate)
    monkeypatch.setattr(checker, "precommit_inputs", abstract_precommit)
    monkeypatch.setattr(checker, "verify_all", abstract_verify)
    recovery.production_main(wrapper_ref["sha256"], recovery_plan_ref["sha256"])
    assert events == ["preflight-reused", "precommit", "verify"]
    assert (state["verify_root"] / "precommit.json").is_file()
    assert (state["verify_root"] / "result.json").is_file()
    assert "FINAL_TECHNICAL_AUDIT_RECOVERY_COMPLETE" in capsys.readouterr().out


def test_production_main_rejects_well_formed_but_unapproved_source_hashes(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        monkeypatch.setenv(name, "1")
    wrapper = recovery.source_reference(recovery.WRAPPER)["sha256"]
    plan = recovery.source_reference(recovery.RECOVERY_PLAN)["sha256"]
    wrong_wrapper = ("0" if wrapper[0] != "0" else "1") + wrapper[1:]
    wrong_plan = ("0" if plan[0] != "0" else "1") + plan[1:]
    with pytest.raises(ValueError, match="differs from reviewed source"):
        recovery.production_main(wrong_wrapper, plan)
    with pytest.raises(ValueError, match="differs from reviewed source"):
        recovery.production_main(wrapper, wrong_plan)
