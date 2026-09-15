"""Postseal orchestration fixtures: no real samples, truth, fitting or CUDA."""
from copy import deepcopy
import subprocess
import sys
import time

import pytest

from experiments.atencion_armonica import run_geometric_decision_postseal as op
from experiments.atencion_armonica.test_geometric_decision_selection_operation import stores


def completed(control, name, stage, manifest, payload):
    ref = control.publish_json(f"manifests/{name}.json", {"operation": name, **manifest})
    return op.frozen_op.bounded_operation(control, ref, stage=stage, started_at=time.monotonic(),
        verify=lambda: None, reservation=30., operation=lambda check: control.publish_json(
            f"outputs/{name}.json", {"manifest": ref, **payload}))


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    control, _ = stores(tmp_path)
    monkeypatch.setattr(op.frozen_op, "BASES", [tmp_path])
    paths = ["src/fixture/original.py", *op.LATE_PATHS, op.OPERATOR, op.PLAN, "fixture-protocol"]
    refs = {p: {"path": p, "bytes": 1, "sha256": f"fixture-{i}"} for i, p in enumerate(paths)}
    refs["fixture-protocol"]["sha256"] = op.frozen_op.PROTOCOL_SHA
    monkeypatch.setattr(op.frozen_op, "reference", lambda p: deepcopy(refs[p]))
    original = [refs[paths[0]]]
    protocol = refs["fixture-protocol"]
    profile = op.frozen_op.ArtifactStore(tmp_path/"profiles/closing-cpu-0",
        binding={"protocol": protocol, "code": [*original, *(refs[p] for p in op.LATE_PATHS)]})
    result = profile.publish_json("result.json", {"schema": "geometric-decision-closing-profile-v1",
        "binding": profile.binding, "new_test_access": False, "new_observations": 0})
    prof = completed(control, "profile-closing", "profile", {"root": str(profile.root), "binding": profile.binding},
        {"root": str(profile.root), "binding": profile.reference(profile.path("binding.json")), "result": result})
    frozen = {"protocol": protocol, "code": original, "budget": {"forecast": {
        "profile_finish": prof["finish"], "profile_report": result,
        "operation_seconds": {"observable-replay": 30.}}, "charged_before_freeze": {"evaluation": 0.},
        "reservations": {"evaluation": 60.}}}
    freeze = control.publish_json("freezes/prospective.json", frozen)
    contract = control.publish_json(op.CONTRACT_PATH, op.contract_payload(control, freeze))
    graph = deepcopy(original)
    monkeypatch.setattr(op.frozen_op, "sources", lambda: deepcopy(sorted(graph, key=lambda r: r["path"])))
    seal = control.publish_json("fixture/seal.json", {"batches": [{"split": "fixture", "observed": "no-data"}]})
    completed(control, "prospective-observables", "fresh", {}, {"seal": seal, "test_freeze": freeze})
    return control, freeze, frozen, contract, graph, refs


def test_contract_has_historical_origin_and_exact_pre_post(fixture):
    control, freeze, frozen, contract, graph, refs = fixture
    value = op.validate_contract(control, contract, freeze, phase="PRE")
    assert value["closing_origin"]["finish"] == frozen["budget"]["forecast"]["profile_finish"]
    assert value["late_sources"] == [refs[p] for p in op.LATE_PATHS]
    graph.extend(refs[p] for p in op.LATE_PATHS)
    op.validate_contract(control, contract, freeze, phase="POST")
    with pytest.raises(ValueError, match="exact PRE"):
        op.validate_contract(control, contract, freeze, phase="PRE")
    # Reporting validates files/provenance, not its own scientific import graph.
    graph.append({"path": "src/fixture/report.py"})
    op.validate_contract(control, contract, freeze)


@pytest.mark.parametrize("mutation", ["missing-original", "one-late", "third-late"])
def test_exact_post_rejects_inventory_mutations(fixture, mutation):
    control, freeze, _, contract, graph, refs = fixture
    graph.extend(refs[p] for p in op.LATE_PATHS)
    if mutation == "missing-original":
        graph.pop(0)
    elif mutation == "one-late":
        graph.pop()
    else:
        graph.append({"path": "src/fixture/unexpected.py"})
    with pytest.raises(ValueError, match="exact POST"):
        op.validate_contract(control, contract, freeze, phase="POST")


@pytest.mark.parametrize("path", [*op.LATE_PATHS, "src/fixture/original.py", op.OPERATOR, op.PLAN])
def test_changed_source_bytes_rejected_even_before_late_import(fixture, path):
    control, freeze, _, contract, _, refs = fixture
    refs[path]["sha256"] = "altered"
    with pytest.raises(ValueError, match="source bytes|contract or historical"):
        op.validate_contract(control, contract, freeze, phase="PRE")


def test_closing_profile_origin_cannot_be_replaced(fixture):
    control, freeze, frozen, contract, _, _ = fixture
    other = deepcopy(frozen)
    other["budget"]["forecast"]["profile_finish"] = {"fixture": "different-origin"}
    other_ref = control.publish_json("fixture/other-freeze.json", other)
    with pytest.raises(ValueError, match="origin differs"):
        op.contract_payload(control, other_ref)
    # A contract for one freeze must not be attached to another.
    duplicate = control.publish_json("fixture/copy-freeze.json", frozen)
    with pytest.raises(ValueError, match="contract or historical"):
        op.validate_contract(control, contract, duplicate)


def fake_evaluation(monkeypatch, fixture, *, add_late=True):
    _, _, _, _, graph, refs = fixture
    calls = []
    def evaluate(control, finish, freeze, output, *, check, replay):
        check()
        calls.append(replay)
        if add_late:
            graph.extend(refs[p] for p in op.LATE_PATHS)
        return output.publish_json("complete.json", {"fixture": "not-scientific"})
    monkeypatch.setattr(op.frozen_op, "evaluate_fresh", evaluate)
    return calls


def test_evaluation_phase_transition_and_supplemental_receipt(fixture, monkeypatch):
    control, freeze, frozen, contract, _, _ = fixture
    calls = fake_evaluation(monkeypatch, fixture)
    result = op.execute(control, "evaluate", {}, freeze, frozen, contract, started_at=time.monotonic())
    finish = control.json(result["finish"])
    row = op.frozen_op.completed_operation(control, "evaluate")
    assert finish["status"] == "COMPLETE" and calls == [False]
    assert row[3]["execution_contract"] == row[4]["execution_contract"] == contract
    assert row[3]["observable_replay_finish"] is None
    assert 0 < finish["charged_after"]["evaluation"] < frozen["budget"]["reservations"]["evaluation"]
    with pytest.raises(ValueError, match="already attempted"):
        op.execute(control, "evaluate", {}, freeze, frozen, contract, started_at=time.monotonic())


def test_failed_terminal_graph_does_not_authorize_written_output_or_retry(fixture, monkeypatch):
    control, freeze, frozen, contract, _, _ = fixture
    fake_evaluation(monkeypatch, fixture, add_late=False)
    with pytest.raises(ValueError, match="exact POST"):
        op.execute(control, "evaluate", {}, freeze, frozen, contract, started_at=time.monotonic())
    assert control.path("outputs/evaluate.json").exists()
    last = sorted(control.path("attempts").glob("*/finish.json"))[-1]
    receipt = control.json(control.reference(last))
    assert receipt["status"] == "FAILED" and receipt["completion"] is None
    with pytest.raises(ValueError, match="COMPLETE"):
        op.frozen_op.completed_operation(control, "evaluate")
    with pytest.raises(ValueError, match="already attempted"):
        op.execute(control, "evaluate", {}, freeze, frozen, contract, started_at=time.monotonic())


def test_replay_requires_completed_evaluation_before_recovery(fixture, monkeypatch):
    control, freeze, frozen, contract, _, _ = fixture
    monkeypatch.setattr(op.frozen_op, "observable_replay", lambda *a, **k: pytest.fail("recovery called early"))
    with pytest.raises(ValueError, match="COMPLETE"):
        op.execute(control, "replay", {}, freeze, frozen, contract, started_at=time.monotonic())
    assert not control.path("manifests/observable-replay.json").exists()


def test_replay_preserves_recovery_stage_and_shared_evaluation_budget(fixture, monkeypatch):
    control, freeze, frozen, contract, graph, _ = fixture
    fake_evaluation(monkeypatch, fixture)
    original_graph = deepcopy(graph)
    evaluated = op.execute(control, "evaluate", {}, freeze, frozen, contract, started_at=time.monotonic())
    prior_charge = control.json(evaluated["finish"])["charged_after"]["evaluation"]
    graph[:] = original_graph  # Simulates the required new CPU process.
    recovered = []
    def recovery(control, finish_ref, freeze_ref, frozen, ctx, manifest, *, check):
        check()
        op.validate_contract(control, contract, freeze, phase="PRE")
        seal = op.frozen_op.completed_operation(control, "prospective-observables")[4]["seal"]
        assert control.json(manifest) == {"operation": "observable-replay", "test_freeze": freeze,
            "observable_finish": finish_ref, "seal": seal}
        recovered.append(True)
        return control.publish_json("outputs/observable-replay.json", {"manifest": manifest,
            "test_freeze": freeze_ref, "observable_finish": finish_ref, "seal": seal,
            "recovered": control.json(seal)["batches"]})
    monkeypatch.setattr(op.frozen_op, "observable_replay", recovery)
    result = op.execute(control, "replay", {}, freeze, frozen, contract, started_at=time.monotonic())
    row = op.frozen_op.completed_operation(control, "replay")
    recovery_row = op.frozen_op.completed_operation(control, "observable-replay")
    assert recovered == [True] and recovery_row[2]["stage"] == "fresh"
    assert row[3]["observable_replay_finish"] == recovery_row[0]
    assert row[3]["execution_contract"] == row[4]["execution_contract"] == contract
    finish = control.json(result["finish"])
    assert finish["charged_after"]["evaluation"] > prior_charge
    assert op.remaining_evaluation(control, frozen) == pytest.approx(60.-finish["charged_after"]["evaluation"])


@pytest.mark.parametrize("mutation", [None, "evaluate-manifest", "evaluate-output", "replay-manifest",
    "replay-output", "recovery", "late-bytes"])
def test_report_authenticates_shared_contract_before_opening_metrics(fixture, monkeypatch, mutation):
    from experiments.atencion_armonica import report_geometric_decision as report
    control, freeze, frozen, contract, _, refs = fixture
    monkeypatch.setattr(report, "BASES", op.frozen_op.BASES)
    monkeypatch.setattr(report, "reference", op.frozen_op.reference)
    fresh = op.frozen_op.completed_operation(control, "prospective-observables")
    seal = fresh[4]["seal"]
    recovered = completed(control, "observable-replay", "fresh", {"test_freeze": freeze,
        "observable_finish": fresh[0], "seal": seal}, {"test_freeze": freeze,
        "observable_finish": fresh[0], "seal": seal, "recovered": control.json(seal)["batches"]})
    output = op.frozen_op.ArtifactStore(op.frozen_op.BASES[0]/"evaluation",
        binding={"test_freeze": freeze, "prediction_seal": seal})
    # Metadata-only stand-in, not evidence of scene-count/metric correctness.
    complete = output.publish_json("complete.json", {"schema": "geometric-decision-evaluation-complete-v1",
        "binding": output.binding, "original_scenes": 2048,
        "results": [{"split": row["split"]} for row in report.TEST_ROSTER]})
    for mode in ("evaluate", "replay"):
        manifest = {"observable_finish": fresh[0], "test_freeze": freeze, "root": str(output.root),
            "execution_contract": contract, "observable_replay_finish": recovered["finish"] if mode == "replay" else None}
        payload = {"root": str(output.root), "test_freeze": freeze, "complete": complete, "execution_contract": contract}
        if mutation == mode+"-manifest":
            manifest["execution_contract"] = {"fixture": "wrong-contract"}
        if mutation == mode+"-output":
            payload["execution_contract"] = {"fixture": "wrong-contract"}
        if mutation == "recovery" and mode == "replay":
            manifest["observable_replay_finish"] = {"fixture": "wrong-recovery"}
        completed(control, mode, "evaluation", manifest, payload)
    if mutation == "late-bytes":
        refs[op.LATE_PATHS[0]]["sha256"] = "modified"
    if mutation is not None:
        monkeypatch.setattr(report, "ReadOnlyStore", lambda *a, **k: pytest.fail("metrics opened before admission"))
        with pytest.raises(ValueError, match="execution contracts|exact observable recovery|source bytes"):
            report.completed_inputs(control)
    else:
        view, value, origin = report.completed_inputs(control)
        assert value == output.json(complete) and view.binding == output.binding
        assert origin["execution_contract"] == contract and origin["observable_replay_finish"] == recovered["finish"]


def test_clean_import_retains_original_graph_and_inert_late_modules_extend_exactly():
    # Real import graph in a clean interpreter; no experimental data is opened.
    script = '''
import sys
from types import ModuleType
from experiments.atencion_armonica import run_geometric_decision_fresh as original
before = original.sources()
from experiments.atencion_armonica import run_geometric_decision_postseal as current
assert current.frozen_op.sources() == before
names = [p.removesuffix('.py').replace('/', '.') for p in current.LATE_PATHS]
assert all(name not in sys.modules for name in names)
for name, path in zip(names, current.LATE_PATHS):
    module = ModuleType(name)
    module.__file__ = str(original.ROOT/path)
    sys.modules[name] = module
expected = sorted(before + [original.reference(p) for p in current.LATE_PATHS], key=lambda r: r['path'])
assert original.sources() == expected
assert not original.torch.cuda.is_initialized()
print('clean PRE unchanged; inert POST exact original+2; CUDA not initialized')
'''
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "POST exact original+2" in result.stdout
