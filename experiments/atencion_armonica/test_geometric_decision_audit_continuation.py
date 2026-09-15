"""Real-store continuation fixtures; no scientific campaign payloads opened."""
from __future__ import annotations

from copy import deepcopy
import time

import pytest

from experiments.atencion_armonica import audit_geometric_decision as checker
from experiments.atencion_armonica import audit_geometric_decision_core as core
from experiments.atencion_armonica import continue_geometric_decision_audit as continuation
from experiments.atencion_armonica import resume_geometric_decision_audit as recovery
from experiments.atencion_armonica.test_geometric_decision_audit_recovery import fixture_state
from src.atencion_armonica.geometric_decision_store import ArtifactStore


def failed_history(root):
    state = fixture_state(root)
    control = state["control"]
    authority = {**state["authority"], "checker": state["code"],
        "plan": state["plan"], "protocol": state["protocol"],
        "preflight_finish": state["preflight_finish"], "preflight_output": state["preflight_output"]}
    start = recovery.begin_recovery(control, state["control_read"], state["ledger"], state["preflight"],
        authority=authority, verify_root=state["verify_root"], started_at=time.monotonic())
    p = state["preflight"]
    metadata = {"terminal_receipts": {"post-replay-report": state["report_finish"]},
        **{key: {"fixture": key} for key in ("freeze_ref", "seal_ref", "contract_ref", "report_result_ref")},
        "cuts": [{"split": str(i), "scene_ids": list(range(7 if i < 3 else 6)),
                  "derived_source_refs": [{"fixture": j} for j in range(4)]} for i in range(4)]}
    binding = {"schema": "geometric-decision-final-audit-binding-v1", "code": state["code"],
        "plan": state["plan"], "protocol": state["protocol"], "inputs": metadata["terminal_receipts"],
        "preflight": {"finish": p["finish_ref"], "output": p["finish"]["completion"],
            "admission": p["output"]["admission"], "projection": p["output"]["projection"]},
        "recovery": start.authority}
    old = {"schema": "geometric-decision-final-audit-precommit-v1", "binding": binding,
        "checker": state["code"], "plan": state["plan"], "preflight": binding["preflight"],
        **metadata, "result_payloads_opened": False}
    refs = {}

    def fail_after_precommit(store):
        refs["binding"] = store.reference(store.path("binding.json"))
        refs["precommit"] = store.publish_json("precommit.json", old)
        raise KeyError("root")

    with pytest.raises(KeyError):
        recovery.open_store_and_run(start, binding, fail_after_precommit)
    refs.update(finish=control.reference(control.path(start.budget.folder + "/finish.json")),
        liability=start.incident_start,
        discrepancy=control.reference(control.path(recovery.RECOVERY_DISCREPANCY_PATH)))
    new_root = state["campaign"] / "audit-final-adapter-verify"
    state.update(old=old, metadata=metadata, pins=refs, new_root=new_root,
                 current_code=[{**state["abstract"], "sha256": "2" * 64}])
    return state


def validated(state):
    return continuation.validate_history(state["control"], state["control_read"],
        old_root=state["verify_root"], verify_root=state["new_root"], pins=state["pins"],
        old_code=state["code"], expected_tail=3)


def new_authority(state):
    return {"schema": "geometric-decision-audit-adapter-continuation-authority-v1",
        "current_checker": state["current_code"], "historical_checker": state["code"],
        "audit_plan": state["plan"], "protocol": state["protocol"],
        "previous_precommit": {"root": str(state["verify_root"]),
            "binding": state["pins"]["binding"], "precommit": state["pins"]["precommit"]},
        "access_history": {"result_payloads_opened": True, "scope": "fixture post-precommit access"}}


def test_complete_continuation_uses_fixed_cut_new_code_and_charges_no_second_liability(tmp_path, monkeypatch):
    state = failed_history(tmp_path)
    ledger, preflight, old = validated(state)
    control = state["control"]
    authority = new_authority(state)
    assert authority["current_checker"] != authority["historical_checker"]
    manifest, budget = continuation.start_attempt(control, ledger, authority=authority,
        verify_root=state["new_root"], started_at=time.monotonic())
    assert budget.folder == "attempts/0004"
    assert control.json(budget.start_ref)["charged_before"] == ledger["attempts"][-1]["finish"]["charged_after"]
    assert not state["new_root"].exists()
    seen = []

    def admitted(*args, **kwargs):
        assert control.path(budget.start_ref["path"]).exists()
        seen.append("admission")
        return state["metadata"]

    def verify(c, r, l, active, pref, store, coverage, **kwargs):
        assert active == budget.start_ref
        precommit = store.json(pref)
        assert precommit["cuts"] == old["cuts"]
        assert precommit["checker"] == state["current_code"]
        assert precommit["result_payloads_opened"] is True
        assert precommit["binding"]["continuation"]["previous_precommit"]["precommit"] == state["pins"]["precommit"]
        seen.append("verify")
        return {"status": "COMPLETE", "authority": "fixture, not scientific validation"}

    monkeypatch.setattr(checker, "REQUIRED_COMPLETE", ("post-replay-report",))
    monkeypatch.setattr(checker, "precommit_inputs", admitted)
    monkeypatch.setattr(checker, "verify_all", verify)
    output, finish = continuation.execute(control, state["control_read"], ledger, preflight, old,
        authority, manifest, budget, verify_root=state["new_root"], coverage=core.Coverage(),
        sources_unchanged=lambda: seen.append("sources"))
    assert seen == ["admission", "verify", "sources"]
    assert control.json(finish)["status"] == "COMPLETE"
    assert control.json(output)["manifest"] == manifest
    after = checker.read_ledger(state["control_read"], control.binding)
    assert len(after["attempts"]) == 5
    assert sum(row["finish"] is None for row in after["attempts"]) == 1
    assert after["charged"]["audit"] == ledger["charged"]["audit"] + control.json(finish)["seconds"]
    with pytest.raises(ValueError, match="no retry"):
        validated(state)


@pytest.mark.parametrize("kind", ("binding", "precommit", "finish", "liability", "discrepancy", "old_code", "tail"))
def test_history_gate_rejects_mutated_pins_or_wrong_tail(tmp_path, kind):
    state = failed_history(tmp_path)
    if kind == "old_code":
        state["code"] = state["current_code"]
    elif kind == "tail":
        ledger, _, _ = validated(state)
        continuation.start_attempt(state["control"], ledger, authority=new_authority(state),
            verify_root=state["new_root"], started_at=time.monotonic())
    else:
        state["pins"][kind] = {**state["pins"][kind], "sha256": "0" * 64}
    with pytest.raises(ValueError):
        validated(state)
    assert not state["new_root"].exists()


@pytest.mark.parametrize("field", continuation.FIXED_FIELDS)
def test_fixed_cut_rejects_changed_identity_even_with_same_27_and_16_counts(field):
    old = {key: {"identity": "old"} for key in continuation.FIXED_FIELDS}
    old["cuts"] = [{"scene_ids": list(range(27)), "derived_source_refs": list(range(16))}]
    old["preflight"] = {"finish": "old"}
    changed = deepcopy(old)
    if field == "cuts":
        changed["cuts"][0]["scene_ids"][-1] = 511  # Same size, different identity.
    else:
        changed[field] = {"identity": "new"}
    binding = {"preflight": old["preflight"], "continuation": {"access_history": {"result_payloads_opened": True}}}
    with pytest.raises(ValueError, match=field):
        continuation.fixed_precommit(changed, old, binding, [], {})


def test_new_precommit_cannot_claim_unopened_results():
    old = {key: [] for key in continuation.FIXED_FIELDS}
    old["preflight"] = {}
    binding = {"preflight": {}, "continuation": {"access_history": {"result_payloads_opened": False}}}
    with pytest.raises(ValueError, match="access history"):
        continuation.fixed_precommit(old, old, binding, [], {})


def test_constructor_failure_has_measured_finish_and_no_retry(tmp_path, monkeypatch):
    state = failed_history(tmp_path)
    ledger, preflight, old = validated(state)
    authority = new_authority(state)
    manifest, budget = continuation.start_attempt(state["control"], ledger, authority=authority,
        verify_root=state["new_root"], started_at=time.monotonic())
    monkeypatch.setattr(checker, "REQUIRED_COMPLETE", ("post-replay-report",))

    def fail(*args, **kwargs):
        assert state["control"].path(budget.start_ref["path"]).exists()
        raise RuntimeError("fixture constructor failure")

    monkeypatch.setattr(continuation, "ArtifactStore", fail)
    with pytest.raises(RuntimeError, match="constructor"):
        continuation.execute(state["control"], state["control_read"], ledger, preflight, old,
            authority, manifest, budget, verify_root=state["new_root"], coverage=core.Coverage(),
            sources_unchanged=lambda: None)
    finish = state["control"].json(state["control"].reference(state["control"].path(budget.folder + "/finish.json")))
    assert finish["status"] == "FAILED" and finish["seconds"] >= 0
    assert finish["charged_after"]["audit"] == ledger["charged"]["audit"] + finish["seconds"]
    with pytest.raises(ValueError, match="no retry"):
        validated(state)


@pytest.mark.parametrize("path", (continuation.WRAPPER, checker.OPERATOR, checker.CORE,
                                  continuation.PLAN, continuation.ACCESS_RECORD))
def test_reviewed_source_hash_required(path):
    with pytest.raises(ValueError, match="reviewed pin"):
        continuation.approved_source(path, "0" * 64)


def test_checker_must_match_reviewed_pair_before_start(monkeypatch):
    expected = [{"path": "checker", "sha256": "a" * 64}]
    monkeypatch.setattr(checker, "sources", lambda: expected)
    assert continuation.current_checker(expected) == expected
    with pytest.raises(ValueError, match="before start"):
        continuation.current_checker([{**expected[0], "sha256": "b" * 64}])
