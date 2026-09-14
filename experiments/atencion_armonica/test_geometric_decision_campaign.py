"""Controller topology fixtures; no real corpus, CUDA or scientific cells."""
from types import SimpleNamespace

import pytest

from src.atencion_armonica import geometric_decision_campaign as module
from src.atencion_armonica.geometric_decision_store import ArtifactStore
from experiments.atencion_armonica.test_geometric_decision_cell import dataset, runtime, store_for, run


def fake_preparation():
    loads = []
    binding = {"fixture": "source"}
    def load(ref, cp, *, check):
        check()
        loads.append(cp)
        return SimpleNamespace(binding={"fixture": "data", "cp": cp},
            rows={"train": [None]*4096, "calibration": [None]*512}, eligible={"train": [0], "calibration": [0]})
    return SimpleNamespace(store=SimpleNamespace(binding=binding), load_checkpoint=load, loads=loads)


def campaign_store(tmp_path, prep):
    return ArtifactStore(tmp_path/"campaign", binding={"schema": "geometric-decision-campaign-v1",
        "preparation": prep.store.binding, "open_complete": {"fixture": "complete"},
        "device": "cpu", "roster": [list(v) for v in module.ROSTER]})


def test_roster_exactly_72_and_loads_once_per_backbone(tmp_path, monkeypatch):
    prep = fake_preparation()
    store = campaign_store(tmp_path, prep)
    calls = []
    def cell(data, child, **kwargs):
        calls.append((kwargs["checkpoint_seed"], kwargs["arm"], kwargs["reader_seed"]))
        assert child.binding == module.cell_identity(data, *calls[-1], "cpu")
        return child.publish_json("complete.json", {"fixture": "topology-only-not-training"})
    monkeypatch.setattr(module, "run_cell", cell)
    monkeypatch.setattr(module, "validate_complete", lambda *args: None)
    ref = module.run_campaign(prep, {"fixture": "complete"}, store, device="cpu", attempt="fixture-0",
                              check=lambda: None, progress=lambda row: None)
    assert calls == list(module.ROSTER) and len(calls) == 72
    assert prep.loads == list(module.CHECKPOINTS)
    result = store.json(ref)
    assert len(result["cells"]) == 72 and result["selection"] == "not performed" and result["fresh_tests"] == "not opened"


def test_interrupt_cannot_publish_campaign_completion(tmp_path, monkeypatch):
    prep = fake_preparation()
    store = campaign_store(tmp_path, prep)
    count = 0
    def cell(data, child, **kwargs):
        nonlocal count
        count += 1
        if count == 2:
            raise InterruptedError("fixture pause")
        return child.publish_json("complete.json", {"fixture": "topology-only"})
    monkeypatch.setattr(module, "run_cell", cell)
    monkeypatch.setattr(module, "validate_complete", lambda *args: None)
    with pytest.raises(InterruptedError):
        module.run_campaign(prep, {"fixture": "complete"}, store, device="cpu", attempt="fixture-0",
                            check=lambda: None, progress=lambda row: None)
    assert not store.path("complete.json").exists()
    assert len(list(store.path("completed").rglob("*.json"))) == 1
    assert len(list(store.path("timing").glob("tail-*.json"))) == 2


def test_complete_validator_requires_actual_last_state_and_calibrations(tmp_path, dataset, runtime):
    cell = store_for(tmp_path/"real-kernel-fixture", dataset)
    ref = run(dataset, cell)
    assert module.validate_complete(dataset, cell, ref)["steps"] == 100
    bad = cell.json(ref)
    bad["last_epoch"] = 49
    bad_ref = cell.publish_json("incomplete-fixture.json", bad)
    with pytest.raises(ValueError, match="incomplete cell"):
        module.validate_complete(dataset, cell, bad_ref)


def test_wrong_frozen_roster_rejected_before_loading(tmp_path):
    prep = fake_preparation()
    store = campaign_store(tmp_path, prep)
    with pytest.raises(ValueError, match="fixed roster"):
        module.run_campaign(prep, {"fixture": "other"}, store, device="cpu", attempt="fixture-0",
                            check=lambda: None)
    assert prep.loads == []
