"""Real stores and budgets; scientific callbacks are abstract fixture boundaries."""
from copy import deepcopy
import time

import pytest

from experiments.atencion_armonica import audit_geometric_decision as checker
from experiments.atencion_armonica import audit_geometric_decision_core as core
from experiments.atencion_armonica import continue_geometric_decision_audit as previous
from experiments.atencion_armonica import verify_geometric_decision_profile_order as current
from experiments.atencion_armonica.test_geometric_decision_audit_continuation import (
    failed_history, validated, new_authority)
from src.atencion_armonica.geometric_decision_budget import BudgetExceeded
from src.atencion_armonica.geometric_decision_store import ArtifactStore


def profile_failure(root, monkeypatch):
    state = failed_history(root)
    ledger, preflight, old = validated(state)
    authority = {**new_authority(state),
        "failed_finish": state["pins"]["finish"],
        "failed_discrepancy": state["pins"]["discrepancy"],
        "liability_start": state["pins"]["liability"]}
    manifest, budget = previous.start_attempt(state["control"], ledger, authority=authority,
        verify_root=state["new_root"], started_at=time.monotonic())

    def fail(*args, **kwargs):
        raise ValueError("closing profile observed inventory roster differs")

    with monkeypatch.context() as patch:
        patch.setattr(checker, "REQUIRED_COMPLETE", ("post-replay-report",))
        patch.setattr(checker, "precommit_inputs", lambda *a, **kw: state["metadata"])
        patch.setattr(checker, "verify_all", fail)
        with pytest.raises(ValueError, match="inventory roster"):
            previous.execute(state["control"], state["control_read"], ledger, preflight, old,
                authority, manifest, budget, verify_root=state["new_root"], coverage=core.Coverage(),
                sources_unchanged=lambda: None)
    control = state["control"]
    old_store = core.AuditStore("previous-cut", state["new_root"], core.Coverage())
    state["profile_pins"] = {
        "start": budget.start_ref, "manifest": manifest,
        "finish": control.reference(control.path(budget.folder + "/finish.json")),
        "discrepancy": control.reference(control.path(previous.DISCREPANCY)),
        "binding": old_store.current_reference("binding.json"),
        "precommit": old_store.current_reference("precommit.json")}
    state["profile_root"] = state["campaign"] / "audit-final-profile-order-verify"
    state["profile_code"] = [{**state["abstract"], "sha256": "3" * 64}]
    return state


def gate(state):
    return current.validate_history(state["control"], state["control_read"],
        old_root=state["new_root"], original_root=state["verify_root"],
        verify_root=state["profile_root"], pins=state["profile_pins"],
        previous_code=state["current_code"], original_code=state["code"], expected_tail=4)


def authority_for(state, old):
    return {"schema": "geometric-decision-audit-profile-order-authority-v1",
        "current_checker": state["profile_code"], "previous_checker": state["current_code"],
        "original_checker": state["code"], "audit_plan": state["plan"], "protocol": state["protocol"],
        "previous_precommit": {"root": str(state["new_root"]),
            **{k: state["profile_pins"][k] for k in ("binding", "precommit")}},
        "original_precommit": old["binding"]["continuation"]["previous_precommit"],
        "access_history": old["binding"]["continuation"]["access_history"]}


def start(state):
    ledger, preflight, old, original = gate(state)
    authority = authority_for(state, old)
    manifest, budget = current.start_attempt(state["control"], ledger, authority=authority,
        verify_root=state["profile_root"], started_at=time.monotonic(), next_attempt=5)
    return ledger, preflight, old, original, authority, manifest, budget


def execute(state, args, **kwargs):
    return current.execute(state["control"], state["control_read"], *args,
        verify_root=state["profile_root"], coverage=core.Coverage(), **kwargs)


def test_complete_new_path_preserves_history_cuts_and_single_liability(tmp_path, monkeypatch):
    state = profile_failure(tmp_path, monkeypatch)
    control = state["control"]
    historical = {str(p): p.read_bytes() for p in state["campaign"].rglob("*") if p.is_file()}
    args = start(state)
    ledger, _, old, original, authority, manifest, budget = args
    assert budget.folder == "attempts/0005"
    assert control.json(budget.start_ref)["charged_before"] == ledger["attempts"][-1]["finish"]["charged_after"]
    assert not state["profile_root"].exists()
    assert len({repr(authority[k]) for k in ("current_checker", "previous_checker", "original_checker")}) == 3
    seen = []

    def verify(c, r, l, active, pref, store, coverage, **kwargs):
        assert active == budget.start_ref and c.path(active["path"]).exists()
        precommit = store.json(pref)
        for field in current.FIXED_FIELDS:
            assert precommit[field] == old[field] == original[field]
        assert precommit["checker"] == state["profile_code"]
        assert precommit["result_payloads_opened"] is True
        assert precommit["binding"]["continuation"] == authority
        seen.append("verify")
        return {"status": "COMPLETE", "authority": "fixture, not campaign verification"}

    monkeypatch.setattr(checker, "REQUIRED_COMPLETE", ("post-replay-report",))
    monkeypatch.setattr(checker, "precommit_inputs", lambda *a, **kw: state["metadata"])
    monkeypatch.setattr(checker, "verify_all", verify)
    output, finish = execute(state, args, sources_unchanged=lambda: seen.append("sources"))
    assert seen == ["verify", "sources"]
    assert control.json(output)["manifest"] == manifest
    assert control.json(finish)["status"] == "COMPLETE"
    after = checker.read_ledger(state["control_read"], control.binding)
    assert len(after["attempts"]) == 6
    assert sum(row["finish"] is None for row in after["attempts"]) == 1
    assert after["charged"]["audit"] == ledger["charged"]["audit"] + control.json(finish)["seconds"]
    for path, raw in historical.items():
        assert type(state["campaign"])(path).read_bytes() == raw
    with pytest.raises(ValueError, match="no retry"):
        gate(state)


@pytest.mark.parametrize("kind", ("start", "finish", "manifest", "discrepancy", "binding", "precommit",
                                 "previous_code", "original_code", "tail", "extra", "original_extra", "symlink"))
def test_gate_rejects_corrupt_or_wrong_history(tmp_path, monkeypatch, kind):
    state = profile_failure(tmp_path, monkeypatch)
    if kind in state["profile_pins"]:
        state["profile_pins"][kind] = {**state["profile_pins"][kind], "sha256": "0" * 64}
    elif kind in ("previous_code", "original_code"):
        state["current_code" if kind == "previous_code" else "code"] = state["profile_code"]
    elif kind == "tail":
        start(state)
    elif kind in ("extra", "original_extra"):
        root = state["new_root"] if kind == "extra" else state["verify_root"]
        ArtifactStore(root, binding=core.AuditStore("fixture", root, core.Coverage()).json(
            core.AuditStore("fixture", root, core.Coverage()).current_reference("binding.json"))).publish_json(
                "extra.json", {"unapproved": True})
    else:
        state["profile_root"].symlink_to(state["new_root"], target_is_directory=True)
    with pytest.raises(ValueError):
        gate(state)


@pytest.mark.parametrize("failure", ("constructor", "cut", "verify", "source"))
def test_execution_failures_are_measured_and_cannot_retry(tmp_path, monkeypatch, failure):
    state = profile_failure(tmp_path, monkeypatch)
    args = start(state)
    ledger, _, _, _, _, manifest, budget = args
    metadata = deepcopy(state["metadata"])
    if failure == "cut":
        metadata["cuts"][0]["scene_ids"][-1] = 511  # Same counts; different identity.

    def fail(*a, **kw):
        assert state["control"].path(budget.start_ref["path"]).exists()
        raise ValueError("fixture " + failure)

    monkeypatch.setattr(checker, "REQUIRED_COMPLETE", ("post-replay-report",))
    monkeypatch.setattr(checker, "precommit_inputs", lambda *a, **kw: metadata)
    monkeypatch.setattr(checker, "verify_all", fail if failure == "verify" else lambda *a, **kw: {"status": "COMPLETE"})
    if failure == "constructor":
        monkeypatch.setattr(current, "ArtifactStore", fail)
    with pytest.raises(ValueError):
        execute(state, args, sources_unchanged=fail if failure == "source" else lambda: None)
    control = state["control"]
    finish = control.json(control.reference(control.path(budget.folder + "/finish.json")))
    assert finish["status"] == "FAILED" and finish["completion"] is None
    assert finish["seconds"] >= 0
    assert finish["charged_after"]["audit"] == ledger["charged"]["audit"] + finish["seconds"]
    assert control.json(control.reference(control.path(current.DISCREPANCY)))["manifest"] == manifest
    assert not control.path(current.OUTPUT).exists()
    with pytest.raises(ValueError, match="no retry"):
        gate(state)


@pytest.mark.parametrize("reservation,projection", ((1501., 1.), (1500., 1501.), (float("nan"), 1.)))
def test_reservation_or_projection_rejected_before_writes(tmp_path, monkeypatch, reservation, projection):
    state = profile_failure(tmp_path, monkeypatch)
    ledger, _, old, _ = gate(state)
    with pytest.raises(BudgetExceeded):
        current.start_attempt(state["control"], ledger, authority=authority_for(state, old),
            verify_root=state["profile_root"], started_at=time.monotonic(), next_attempt=5,
            reservation=reservation, projected_seconds=projection)
    assert not state["control"].path(current.MANIFEST).exists()


def test_insufficient_remaining_budget_rejected(tmp_path, monkeypatch):
    state = profile_failure(tmp_path, monkeypatch)
    ledger, _, old, _ = gate(state)
    ledger["charged"]["audit"] = 2200.
    with pytest.raises(BudgetExceeded):
        current.start_attempt(state["control"], ledger, authority=authority_for(state, old),
            verify_root=state["profile_root"], started_at=time.monotonic(), next_attempt=5)
    assert not state["control"].path(current.MANIFEST).exists()


def test_constructor_guard_closes_real_attempt_on_expired_clock(tmp_path, monkeypatch):
    state = profile_failure(tmp_path, monkeypatch)
    ledger, _, old, _ = gate(state)
    with pytest.raises(BudgetExceeded):
        current.start_attempt(state["control"], ledger, authority=authority_for(state, old),
            verify_root=state["profile_root"], started_at=time.monotonic() - 1., next_attempt=5,
            reservation=.5, projected_seconds=.1)
    control = state["control"]
    finish = control.json(control.reference(control.path("attempts/0005/finish.json")))
    assert finish["status"] == "LIMIT_REACHED" and finish["seconds"] >= 1.
    assert not state["profile_root"].exists()
    with pytest.raises(ValueError, match="no retry"):
        gate(state)


@pytest.mark.parametrize("path", (current.WRAPPER, current.PLAN, current.HELPER, *current.REVIEWS))
def test_wrong_source_pin_rejected(path):
    with pytest.raises(ValueError, match="reviewed pin"):
        current.approved_source(path, "0" * 64)
