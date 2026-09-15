"""Supervisor boundaries with CPU fixtures, never real data or CUDA."""
from copy import deepcopy
from types import SimpleNamespace
import time
import subprocess
import sys

import pytest

from experiments.atencion_armonica import run_geometric_decision_fresh as op
from experiments.atencion_armonica.test_geometric_decision_selection_operation import stores


def fixture_finish(control, stage="fresh"):
    manifest = control.publish_json("fixture/manifest.json", {"fixture": True})
    return op.bounded_operation(control, manifest, stage=stage, started_at=time.monotonic(),
        verify=lambda: None, reservation=30.,
        operation=lambda check: control.publish_json("fixture/complete.json", {"complete": True}))


def test_budget_counts_existing_disk_and_keeps_audit_reserve(tmp_path, monkeypatch):
    control, _ = stores(tmp_path)
    fixture_finish(control)
    forecast = {"fresh_seconds": 100., "evaluation_seconds": 100., "projected_new_bytes": 200}
    monkeypatch.setattr(op, "owned_bytes", lambda roots: 100)
    monkeypatch.setattr(op.shutil, "disk_usage", lambda p: SimpleNamespace(free=op.LIMITS["free_bytes"]+200))
    value = op.check_budget(control, forecast)
    assert value["existing_bytes"] == 100 and value["audit_reserved_seconds"] == 3600
    monkeypatch.setattr(op, "owned_bytes", lambda roots: op.LIMITS["new_bytes"]-199)
    with pytest.raises(ValueError, match="disk or audit"):
        op.check_budget(control, forecast)
    monkeypatch.setattr(op, "owned_bytes", lambda roots: 0)
    monkeypatch.setattr(op.shutil, "disk_usage", lambda p: SimpleNamespace(free=op.LIMITS["free_bytes"]+199))
    with pytest.raises(ValueError, match="disk or audit"):
        op.check_budget(control, forecast)


def test_budget_rejects_oversized_evaluation_without_opening_tests(tmp_path):
    control, _ = stores(tmp_path)
    fixture_finish(control)
    with pytest.raises(ValueError, match="does not fit"):
        op.check_budget(control, {"fresh_seconds": 100., "evaluation_seconds": 7201., "projected_new_bytes": 1})
    assert not control.path("freezes/prospective.json").exists()


def test_bounded_operation_samples_supplied_vram_without_cuda(tmp_path):
    control, _ = stores(tmp_path)
    manifest = control.publish_json("manifest.json", {"fixture": "vram-guard"})
    used = [0]
    def operation(check):
        used[0] = op.LIMITS["vram_bytes"]+1
        check()
        raise AssertionError("must not finish after VRAM breach")
    with pytest.raises(op.BudgetExceeded, match="RSS/VRAM"):
        op.bounded_operation(control, manifest, stage="fresh", started_at=time.monotonic(),
            verify=lambda: None, reservation=30., operation=operation, vram=lambda: used[0])
    finish = control.json(control.reference(control.path("attempts/0000/finish.json")))
    assert finish["status"] == "LIMIT_REACHED" and finish["completion"] is None
    assert op.signal.getitimer(op.signal.ITIMER_REAL)[0] == 0


def test_second_attempt_preserves_cumulative_charges(tmp_path):
    control, _ = stores(tmp_path)
    first = fixture_finish(control)
    second = fixture_finish(control)
    a, b = control.json(first["finish"]), control.json(second["finish"])
    assert b["charged_after"]["fresh"] == a["seconds"]+b["seconds"]


def test_admission_failure_has_live_guard_and_cannot_retry_silently(tmp_path, monkeypatch):
    control, _ = stores(tmp_path)
    monkeypatch.setattr(op, "sources", lambda: [])
    monkeypatch.setattr(op, "reference", lambda p: {"sha256": op.PROTOCOL_SHA})
    calls = []
    def material_open():
        assert control.path("attempts/0000/start.json").exists()
        assert 0 < op.signal.getitimer(op.signal.ITIMER_REAL)[0] <= 120
        calls.append("guarded")
        raise InterruptedError("fixture OPEN interruption")
    monkeypatch.setattr(op, "admitted_open", material_open)
    with pytest.raises(InterruptedError):
        op.admit_launch(control, "fresh", started_at=time.monotonic())
    assert calls == ["guarded"]
    finish = control.json(control.reference(control.path("attempts/0000/finish.json")))
    assert finish["status"] == "PAUSED" and finish["completion"] is None
    assert not control.path("freezes/prospective.json").exists()
    with pytest.raises(ValueError, match="already attempted"):
        op.admit_launch(control, "fresh", started_at=time.monotonic())


def test_existing_stage_cannot_repeat_even_before_admission(tmp_path, monkeypatch):
    control, _ = stores(tmp_path)
    control.publish_json("manifests/evaluate.json", {"fixture": "already attempted"})
    monkeypatch.setattr(op, "sources", lambda: pytest.fail("must reject before new admission"))
    with pytest.raises(ValueError, match="already attempted"):
        op.admit_launch(control, "evaluate", started_at=time.monotonic())


def test_replay_arguments_forbid_any_forward_or_fit(tmp_path):
    from src.atencion_armonica.geometric_decision_store import ArtifactStore
    store = ArtifactStore(tmp_path/"fresh", binding={"fixture": True})
    frozen = {"runtime": {"fixture": "cuda-not-executed"}, "normalization": {"normalizers": {}},
        "scale": {"scale": 1.}, "checkpoints": []}
    ctx = {"archive": None, "archive_complete": {"heads": {}}, "selection": None, "selection_ref": {}}
    args = op.observable_arguments(store, frozen, ctx, check=lambda: None, compute=False)
    with pytest.raises(RuntimeError, match="cannot forward"):
        args["forward"](None, None)
    with pytest.raises(RuntimeError, match="cannot fit"):
        args["fit_candidates"](None, None)
    before = store.read(store.reference(store.path("normalizers.json")))
    op.observable_arguments(store, deepcopy(frozen), ctx, check=lambda: None, compute=False)
    assert before == store.read(store.reference(store.path("normalizers.json")))


def test_runner_import_cannot_load_privileged_truth_reader():
    result = subprocess.run([sys.executable, "-c",
        "import sys; import experiments.atencion_armonica.run_geometric_decision_fresh; "
        "assert 'src.atencion_armonica.generative_evidence_supervision' not in sys.modules"],
        capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr


def test_direct_plans_and_amendment_are_source_pinned():
    paths = {row["path"] for row in op.sources()}
    assert "experiments/atencion_armonica/PLAN_GEOMETRIC_DECISION_FRESH.md" in paths
    assert op.AMENDMENT in paths


@pytest.mark.parametrize("field", ["head_roster", "runtime", "checkpoints", "normalization", "scale", "open"])
def test_reopened_freeze_requires_all_authenticated_inputs(tmp_path, monkeypatch, field):
    control, _ = stores(tmp_path)
    monkeypatch.setattr(op, "BASES", [tmp_path])
    code, protocol = [], {"sha256": op.PROTOCOL_SHA}
    monkeypatch.setattr(op, "sources", lambda: code)
    monkeypatch.setattr(op, "reference", lambda p: protocol)
    inputs = {"head_roster": [{"fixture": True}], "runtime": {}, "checkpoints": [],
        "normalization": {}, "scale": {}, "open": {"path": "fixture"}}
    ctx = {"forecast": {"fresh_seconds": 1., "evaluation_seconds": 1., "profile_finish": {}},
        "observed": {"exclusions": {"fixture": True}}, "archive_evidence": {}, "selection_ref": {}}
    charged = dict.fromkeys(op.STAGES, 0.)
    budget = {"charged_before_freeze": charged, "reservations": {"fresh": 1, "evaluation": 1},
        "existing_bytes": 0, "free_bytes": op.LIMITS["free_bytes"], "audit_reserved_seconds": 3600., "forecast": ctx["forecast"]}
    monkeypatch.setattr(op, "check_budget", lambda *a: budget)
    def admission(*args):
        return {}, {"charged_after": charged}, {}, {"code": code, "protocol": protocol}, {
            "inputs": deepcopy(inputs), "forecast": deepcopy(ctx["forecast"])}
    monkeypatch.setattr(op, "completed_operation", admission)
    first, frozen = op.prepare_freeze(control, inputs, ctx)
    assert op.prepare_freeze(control, deepcopy(inputs), deepcopy(ctx)) == (first, frozen)
    changed = deepcopy(inputs)
    changed[field] = {"different": True}
    with pytest.raises(ValueError, match="freeze inputs differ"):
        op.prepare_freeze(control, changed, ctx)


def test_replay_cannot_start_without_completed_evaluation(tmp_path, monkeypatch):
    control, _ = stores(tmp_path)
    calls = []
    def completion(ctrl, operation):
        calls.append(operation)
        if operation == "evaluate":
            raise ValueError("missing completed evaluation")
        return None, None, None, None, {"test_freeze": {}}
    monkeypatch.setattr(op, "completed_operation", completion)
    with pytest.raises(ValueError, match="missing completed evaluation"):
        op.admit_launch(control, "replay", started_at=time.monotonic())
    assert calls == ["prospective-observables", "evaluate"]
    assert not control.path("manifests/admission-replay.json").exists()


@pytest.mark.parametrize("status", ["COMPLETE", "PAUSED"])
def test_metric_replay_requires_completed_bound_observable_recovery(tmp_path, status):
    control, _ = stores(tmp_path)
    frozen, observable_finish = {"fixture": "freeze"}, {"fixture": "observable-finish"}
    seal = control.publish_json("seal.json", {"batches": [{"split": s, "observed": {"fixture": s}} for s, _ in op.TESTS]})
    manifest = control.publish_json("recovery-manifest.json", {"operation": "observable-replay",
        "test_freeze": frozen, "observable_finish": observable_finish, "seal": seal})
    budget = op.StageBudget(control, "fresh", manifest_ref=manifest, reservation_seconds=30.,
        prior_charges=[], output_roots=[tmp_path])
    if status == "COMPLETE":
        output = control.publish_json("recovery-output.json", {"manifest": manifest,
            "test_freeze": frozen, "observable_finish": observable_finish, "seal": seal,
            "recovered": [{"split": s, "observed": {"fixture": s}} for s, _ in op.TESTS]})
        finish = budget.finish(status, completion=output)
        assert op.admitted_observable_replay(control, observable_finish, frozen, seal) == finish
        with pytest.raises(ValueError, match="exact completed"):
            op.admitted_observable_replay(control, {"wrong": True}, frozen, seal)
    else:
        budget.finish(status)
        with pytest.raises(ValueError, match="exactly one COMPLETE"):
            op.admitted_observable_replay(control, observable_finish, frozen, seal)


@pytest.mark.parametrize("remaining", [50., 0., -1.])
def test_admission_cannot_spend_beyond_frozen_evaluation_allocation(tmp_path, monkeypatch, remaining):
    control, _ = stores(tmp_path)
    receipt = fixture_finish(control, stage="evaluation")
    charged = control.json(receipt["finish"])["charged_after"]["evaluation"]
    # Integer allocation, baseline chosen so the shared remainder is exact.
    allocation, baseline = 100, charged+remaining-100
    # Charge a virtual prior interval through the frozen reference fixture;
    # keep a nonnegative baseline and a valid authenticated current ledger.
    if baseline < 0:
        # This fixture needs a nontrivial ledger charge without wall-clock sleep.
        original = op.StageBudget
        manifest = control.publish_json("long-fixture.json", {"fixture": "virtual-time"})
        clock = [1000.]
        budget = original(control, "evaluation", manifest_ref=manifest, reservation_seconds=300.,
            prior_charges=[], output_roots=[tmp_path], clock=lambda: clock[0])
        clock[0] += 200.
        finish = budget.finish("COMPLETE", completion=control.publish_json("long-output.json", {"fixture": True}))
        charged = control.json(finish)["charged_after"]["evaluation"]
        baseline = charged+remaining-allocation
    frozen = control.publish_json("freeze.json", {"schema": "geometric-decision-prospective-freeze-v1",
        "budget": {"reservations": {"fresh": 1, "evaluation": allocation},
            "charged_before_freeze": {"evaluation": baseline},
            "forecast": {"fresh_seconds": 1., "evaluation_seconds": float(allocation)}}})
    monkeypatch.setattr(op, "completed_operation", lambda *a: (None, None, None, None, {"test_freeze": frozen}))
    monkeypatch.setattr(op, "sources", lambda: [])
    monkeypatch.setattr(op, "reference", lambda p: {"sha256": op.PROTOCOL_SHA})
    calls = []
    def bounded(*args, **kwargs):
        calls.append(kwargs["reservation"])
        raise InterruptedError("fixture bounded admission")
    monkeypatch.setattr(op, "bounded_operation", bounded)
    if remaining > 0:
        with pytest.raises(InterruptedError):
            op.admit_launch(control, "evaluate", started_at=time.monotonic())
        assert calls == [pytest.approx(remaining)]
        assert control.path("manifests/admission-evaluate.json").exists()
    else:
        with pytest.raises(op.BudgetExceeded, match="before admission"):
            op.admit_launch(control, "evaluate", started_at=time.monotonic())
        assert calls == []
        assert not control.path("manifests/admission-evaluate.json").exists()
