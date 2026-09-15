"""Three historical cuts with real stores/budgets, no scientific payloads."""
from copy import deepcopy
from pathlib import Path
import time

import pytest

from experiments.atencion_armonica import audit_geometric_decision as checker
from experiments.atencion_armonica import audit_geometric_decision_core as core
from experiments.atencion_armonica import verify_geometric_decision_profile_order as previous
from experiments.atencion_armonica import verify_geometric_decision_coverage as current
from experiments.atencion_armonica.test_geometric_decision_profile_order import profile_failure, gate, authority_for
from src.atencion_armonica.geometric_decision_budget import BudgetExceeded


def coverage_failure(root, monkeypatch, mutation=None):
    state = profile_failure(root, monkeypatch)
    ledger, preflight, old, original = gate(state)
    authority = {**authority_for(state, old), "failed_attempt": {
        k: state["profile_pins"][k] for k in ("start", "finish", "manifest", "discrepancy")}}
    if mutation == "checker":
        authority["previous_checker"] = state["code"]
    elif mutation == "start_link":
        authority["failed_attempt"]["start"] = ledger["attempts"][-2]["start_ref"]
    elif mutation == "root_link":
        authority["original_precommit"] = authority["previous_precommit"]
    manifest, budget = previous.start_attempt(state["control"], ledger, authority=authority,
        verify_root=state["profile_root"], started_at=time.monotonic(), next_attempt=5)

    def fail(*a, **kw):
        raise ValueError("conflicting identities for profile-head-case:binding.json")

    with monkeypatch.context() as patch:
        patch.setattr(checker, "REQUIRED_COMPLETE", ("post-replay-report",))
        patch.setattr(checker, "precommit_inputs", lambda *a, **kw: state["metadata"])
        patch.setattr(checker, "verify_all", fail)
        with pytest.raises(ValueError, match="conflicting identities"):
            previous.execute(state["control"], state["control_read"], ledger, preflight, old, original,
                authority, manifest, budget, verify_root=state["profile_root"], coverage=core.Coverage(),
                sources_unchanged=lambda: None)
    control = state["control"]
    old_store = core.AuditStore("fixture-0024", state["profile_root"], core.Coverage())
    state["coverage_pins"] = {"start": budget.start_ref, "manifest": manifest,
        "finish": control.reference(control.path(budget.folder + "/finish.json")),
        "discrepancy": control.reference(control.path(previous.DISCREPANCY)),
        "binding": old_store.current_reference("binding.json"),
        "precommit": old_store.current_reference("precommit.json")}
    state["coverage_root"] = state["campaign"] / "audit-final-coverage-verify"
    state["coverage_code"] = [{**state["abstract"], "sha256": "4" * 64}]
    return state


def validated(state):
    return current.validate_history(state["control"], state["control_read"],
        roots=[state["profile_root"], state["new_root"], state["verify_root"]],
        verify_root=state["coverage_root"], pins=state["coverage_pins"],
        historical_code=[state["profile_code"], state["current_code"], state["code"]], expected_tail=5)


def begin(state):
    ledger, preflight, cuts, refs = validated(state)
    authority = {"schema": "geometric-decision-audit-coverage-authority-v1",
        "current_checker": state["coverage_code"], "checker_0024": state["profile_code"],
        "checker_0023": state["current_code"], "checker_0022": state["code"],
        "audit_plan": state["plan"], "protocol": state["protocol"], "historical_precommits": refs,
        "access_history": cuts[0]["binding"]["continuation"]["access_history"]}
    manifest, budget = current.start_attempt(state["control"], ledger, authority=authority,
        verify_root=state["coverage_root"], started_at=time.monotonic(), next_attempt=6)
    return ledger, preflight, cuts, authority, manifest, budget


def execute(state, args, *, sources_unchanged=lambda: None):
    return current.execute(state["control"], state["control_read"], *args,
        verify_root=state["coverage_root"], coverage=core.Coverage(), sources_unchanged=sources_unchanged)


def test_complete_preserves_three_cuts_four_checkers_history_and_charges(tmp_path, monkeypatch):
    state = coverage_failure(tmp_path, monkeypatch)
    historical = {p: p.read_bytes() for p in state["campaign"].rglob("*") if p.is_file()}
    args = begin(state)
    ledger, _, cuts, authority, manifest, budget = args
    assert not state["coverage_root"].exists()
    assert budget.folder == "attempts/0006"
    assert state["control"].json(budget.start_ref)["charged_before"] == ledger["attempts"][-1]["finish"]["charged_after"]
    assert len({repr(authority[k]) for k in ("current_checker", "checker_0024", "checker_0023", "checker_0022")}) == 4

    def verify(c, r, l, active, pref, store, coverage, **kw):
        assert active == budget.start_ref and c.path(active["path"]).exists()
        value = store.json(pref)
        for old in cuts:
            for key in current.FIXED_FIELDS:
                assert core.encoded(value[key]) == core.encoded(old[key])
        assert value["checker"] == state["coverage_code"] and value["result_payloads_opened"] is True
        assert value["binding"]["continuation"]["historical_precommits"] == authority["historical_precommits"]
        return {"status": "COMPLETE", "authority": "fixture only"}

    monkeypatch.setattr(checker, "REQUIRED_COMPLETE", ("post-replay-report",))
    monkeypatch.setattr(checker, "precommit_inputs", lambda *a, **kw: state["metadata"])
    monkeypatch.setattr(checker, "verify_all", verify)
    output, finish = execute(state, args)
    control = state["control"]
    assert control.json(output)["manifest"] == manifest
    assert control.json(finish)["status"] == "COMPLETE"
    after = checker.read_ledger(state["control_read"], control.binding)
    assert len(after["attempts"]) == 7 and sum(r["finish"] is None for r in after["attempts"]) == 1
    assert after["charged"]["audit"] == ledger["charged"]["audit"] + control.json(finish)["seconds"]
    assert all(p.read_bytes() == raw for p, raw in historical.items())
    with pytest.raises(ValueError, match="no retry"):
        validated(state)


@pytest.mark.parametrize("kind", ("start", "finish", "manifest", "discrepancy", "binding", "precommit"))
def test_gate_rejects_mutated_receipts(tmp_path, monkeypatch, kind):
    state = coverage_failure(tmp_path, monkeypatch)
    state["coverage_pins"][kind] = {**state["coverage_pins"][kind], "sha256": "0" * 64}
    with pytest.raises(ValueError):
        validated(state)
    assert not state["coverage_root"].exists()


@pytest.mark.parametrize("kind", ("checker", "start_link", "root_link"))
def test_authenticated_but_wrong_historical_edges_are_rejected(tmp_path, monkeypatch, kind):
    state = coverage_failure(tmp_path, monkeypatch, mutation=kind)
    with pytest.raises(ValueError):
        validated(state)


@pytest.mark.parametrize("which", ("profile_root", "new_root", "verify_root"))
def test_all_three_historical_topologies_are_exact(tmp_path, monkeypatch, which):
    state = coverage_failure(tmp_path, monkeypatch)
    (state[which] / "extra.json").write_text("{}\n")
    with pytest.raises(ValueError, match="topology"):
        validated(state)


@pytest.mark.parametrize("kind", ("constructor", "cut", "verify", "source"))
def test_failures_are_measured_without_retry_or_old_writes(tmp_path, monkeypatch, kind):
    state = coverage_failure(tmp_path, monkeypatch)
    args = begin(state)
    ledger, _, _, _, manifest, budget = args
    metadata = deepcopy(state["metadata"])
    if kind == "constructor":
        state["coverage_root"].mkdir()
        (state["coverage_root"] / "unbound.json").write_text("{}\n")
    if kind == "cut":
        metadata["cuts"][0]["scene_ids"][-1] = 511

    def fail(*a, **kw):
        raise ValueError("fixture " + kind)

    monkeypatch.setattr(checker, "REQUIRED_COMPLETE", ("post-replay-report",))
    monkeypatch.setattr(checker, "precommit_inputs", lambda *a, **kw: metadata)
    monkeypatch.setattr(checker, "verify_all", fail if kind == "verify" else lambda *a, **kw: {"status": "COMPLETE"})
    with pytest.raises(ValueError):
        execute(state, args, sources_unchanged=fail if kind == "source" else lambda: None)
    control = state["control"]
    finish = control.json(control.reference(control.path(budget.folder + "/finish.json")))
    assert finish["status"] == "FAILED" and finish["completion"] is None and finish["seconds"] >= 0
    assert finish["charged_after"]["audit"] == ledger["charged"]["audit"] + finish["seconds"]
    assert control.json(control.reference(control.path(current.DISCREPANCY)))["manifest"] == manifest
    with pytest.raises(ValueError, match="no retry"):
        validated(state)


def test_insufficient_budget_or_expired_constructor(tmp_path, monkeypatch):
    state = coverage_failure(tmp_path, monkeypatch)
    ledger, _, _, _ = validated(state)
    charged = deepcopy(ledger)
    charged["charged"]["audit"] = 2200.
    with pytest.raises(BudgetExceeded):
        current.start_attempt(state["control"], charged, authority={"fixture": True},
            verify_root=state["coverage_root"], started_at=time.monotonic(), next_attempt=6)
    assert not state["control"].path(current.MANIFEST).exists()
    with pytest.raises(BudgetExceeded):
        current.start_attempt(state["control"], ledger, authority={"fixture": True},
            verify_root=state["coverage_root"], started_at=time.monotonic() - 1., next_attempt=6,
            reservation=.5, projected_seconds=.1)
    control = state["control"]
    finish = control.json(control.reference(control.path("attempts/0006/finish.json")))
    assert finish["status"] == "LIMIT_REACHED" and finish["seconds"] >= 1.
    with pytest.raises(ValueError, match="no retry"):
        validated(state)


@pytest.mark.parametrize("path", (current.WRAPPER, current.PLAN, *current.HELPERS, *current.REVIEWS, current.ACCESS_RECORD))
def test_explicit_source_pins_required(path):
    with pytest.raises(ValueError, match="reviewed pin"):
        current.approved_source(path, "0" * 64)
