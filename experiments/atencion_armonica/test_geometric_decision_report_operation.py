"""CPU reporting fixtures; abstract <8-event topology, no experimental data."""
from copy import deepcopy

import numpy as np
import pytest

from experiments.atencion_armonica import report_geometric_decision as op
from experiments.atencion_armonica.test_geometric_decision_selection_operation import stores


class MemoryStore:
    def __init__(self, values):
        self.values = values

    def json(self, ref):
        return deepcopy(self.values[ref])

    def arrays(self, ref):
        return {k: v.copy() for k, v in self.values[ref].items()}

    def publish_json(self, path, value):
        assert path not in self.values
        self.values[path] = deepcopy(value)
        return path


def full_fixture(count, *, score_optima=(), math_optima=(), delivered_optima=()):
    return {"candidate_count": count, "chosen": score_optima[0] if score_optima else None,
        "score_order": {"optima": list(score_optima)},
        "oracles": {"tM": {"optima": list(math_optima)}, "tD": {"optima": list(delivered_optima)}},
        "tau": {"status": "abstract-fixture"}, "errors": None, "decision": None}


def test_report_cannot_read_results_without_completed_replay(monkeypatch):
    calls = []
    def operation(control, name):
        calls.append(name)
        if name == "replay":
            raise ValueError("replay not COMPLETE")
        return None
    monkeypatch.setattr(op, "completed_operation", operation)
    monkeypatch.setattr(op, "ReadOnlyStore", lambda *a, **k: pytest.fail("metric store opened prematurely"))
    with pytest.raises(ValueError, match="not COMPLETE"):
        op.completed_inputs(object())
    assert calls == ["prospective-observables", "evaluate", "replay"]


def test_main_incomplete_replay_cannot_publish_report(tmp_path, monkeypatch):
    control, _ = stores(tmp_path)
    monkeypatch.setattr(op, "ROOT", op.Path.cwd().resolve())
    monkeypatch.setattr(op, "BASES", [tmp_path])
    monkeypatch.setattr(op, "CONTROL_BINDING", control.reference(control.path("binding.json")))
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        monkeypatch.setenv(key, "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    def missing(*args):
        raise ValueError("no completed replay")
    monkeypatch.setattr(op, "completed_operation", missing)
    with pytest.raises(ValueError, match="no completed replay"):
        op.main()
    assert not control.path("manifests/report.json").exists()
    assert not (tmp_path/"report").exists()


def test_sham_averages_scenes_not_candidates_and_retains_empty_support():
    def row(sid, mask, donors):
        groups = [{"sizes": [2, 2], "candidate_ids": list(range(len(mask))), "donors": donors}] if mask else []
        return {"scene_id": sid, "candidate_count": len(mask), "diagnostics": {"1": {
            "scalar_changed_mask": mask, "six_channel_sham": {"changed_mask": mask, "donors": donors, "strata": groups}}}}
    store = MemoryStore({"a": row(17, [True, True, False], [1, 2, 0]), "b": row(18, [False], [0]), "c": row(19, [], [])})
    report = op.sham_summary(store, ["a", "b", "c"], check=lambda: None)
    value = report["checkpoint_cells"]["1"]
    assert value["scenes"] == 3 and value["eligible_scenes"] == 2 and value["candidates"] == 4
    assert value["scalar_changed"] == 2 and value["scalar_scene_mean"] == pytest.approx(1/3)
    assert [r["scene_id"] for r in report["scene_checkpoint_rows"]] == [17, 18, 19]
    for ref, result in zip(["a", "b", "c"], report["scene_checkpoint_rows"]):
        assert result["diagnostics"] == store.values[ref]["diagnostics"]["1"]
        assert result["source"] == ref
    assert report["scene_checkpoint_rows"][1]["stratum_support"][0]["singleton"]
    assert report["scene_checkpoint_rows"][2]["candidate_denominator"] == 0
    store.values["a"]["diagnostics"]["1"]["six_channel_sham"]["donors"] = [0, 0, 2]
    with pytest.raises(ValueError, match="donors must permute"):
        op.sham_summary(store, ["a"], check=lambda: None)


def test_strata_extraction_preserves_slices_singletons_undefined_and_empty_scenes():
    metrics = {name: None for name in op.NAMES}
    defined = {**metrics, "regret_tM": 0., "ari": 1.}
    strata = {"k": {"k=1": {"candidate_ids": [0], "chosen": 0, "metrics": defined, "tau_status": "fixture"},
        "k=2": {"candidate_ids": [1, 2], "chosen": 1, "metrics": metrics, "tau_status": "fixture"}}}
    targets = {0: {"candidate_count": 3, "strata": {"full": {"all": [0, 1, 2]}, "k": {"k=1": [0], "k=2": [1, 2]}}},
        1: {"candidate_count": 0, "strata": {"full": {"all": []}, "k": {}}}}
    rows = [{"scene_id": 0, "strata": strata, "full": full_fixture(3, score_optima=(0, 1), math_optima=(0,), delivered_optima=(0, 2))},
        {"scene_id": 1, "strata": {"k": {}}, "full": full_fixture(0)}]
    result = op.stratum_rows(rows, targets, check=lambda: None)
    assert result[0]["strata"] == strata
    assert result[0]["full"] == rows[0]["full"]
    support = result[0]["support"]["k"]
    assert support["strata"] == 2 and support["singleton_strata"] == 1
    assert support["candidate_denominator"] == 3
    assert support["defined_strata"]["regret_tM"] == 1 and support["undefined_strata"]["regret_tM"] == 1
    assert support["defined_strata"]["tau_b"] == 0 and support["undefined_strata"]["tau_b"] == 2
    assert result[1]["eligible_scene"] is False and result[1]["support"]["k"]["strata"] == 0
    targets[0]["strata"]["k"]["k=2"] = [2]
    with pytest.raises(ValueError, match="candidates or metric roster"):
        op.stratum_rows(rows, targets, check=lambda: None)


def test_stratified_operator_exposes_all_head_and_classical_records():
    row = {"scene_id": 0, "strata": {"k": {}}, "full": full_fixture(0)}
    values = {"target": {"scene_id": 0, "candidate_count": 0, "strata": {"full": {"all": []}, "k": {}}},
        "head": {"identity": [1, "fixture", 1, "initial"], "scene_ids": [0], "rows": [row]},
        "classic": [{"scene_id": 0, "values": {name: row for name in ("base", "extended", "z", "d")}}]}
    store, output = MemoryStore(values), MemoryStore({})
    current = {"scene_ids": [0], "targets": ["target"], "heads": ["head"]*144, "classical": "classic",
        "summary": {"classical_order": ["base", "extended", "z", "d"]}}
    index = output.json(op.stratified_report(store, current, output, "strata", check=lambda: None))
    assert len(index["heads"]) == 144 and len(index["classical"]) == 4
    assert index["total_scenes"] == 1 and index["eligible_scenes"] == 0
    assert output.json(index["heads"][0])["identity"] == values["head"]["identity"]
    assert output.json(index["heads"][0])["rows"][0]["full"] == row["full"]
    assert output.json(index["classical"]["base"])["rows"][0]["candidate_denominator"] == 0
    assert output.json(index["classical"]["base"])["rows"][0]["full"] == row["full"]
    # Duplicate identities are loop fixtures only, not an admission of a real roster.


def test_full_extraction_preserves_distinct_head_classical_ties_and_oracles():
    strata = {"k": {}}
    head_full = full_fixture(3, score_optima=(0, 1), math_optima=(0,), delivered_optima=(0, 2))
    classic_full = full_fixture(3, score_optima=(2,), math_optima=(0,), delivered_optima=(0, 2))
    values = {"target": {"scene_id": 0, "candidate_count": 3, "strata": strata},
        "head": {"identity": [1, "fixture", 1, "selected"], "scene_ids": [0],
            "rows": [{"scene_id": 0, "strata": strata, "full": head_full}]},
        "classic": [{"scene_id": 0, "values": {name: {"strata": strata, "full": classic_full}
            for name in ("base", "extended", "z", "d")}}]}
    current = {"scene_ids": [0], "targets": ["target"], "heads": ["head"]*144, "classical": "classic",
        "summary": {"classical_order": ["base", "extended", "z", "d"]}}
    output = MemoryStore({})
    index = output.json(op.stratified_report(MemoryStore(values), current, output, "strata", check=lambda: None))
    for ref in index["heads"]:
        assert output.json(ref)["rows"][0]["full"] == head_full
    for ref in index["classical"].values():
        assert output.json(ref)["rows"][0]["full"] == classic_full
    del values["head"]["rows"][0]["full"]["score_order"]
    with pytest.raises(ValueError, match="full-scene diagnostic"):
        op.stratified_report(MemoryStore(values), current, MemoryStore({}), "strata", check=lambda: None)


def test_probe_extraction_keeps_all_readouts_and_compares_event_topology():
    scene = {"partitions": [[[0, 1], [2, 3]], [[0, 2], [1, 3]]], "canonical_to_observed": [0, 1, 2, 3]}
    values = {
        "sources": {"scene_ids": [0], "sources": ["before"]},
        "derived-sources": {"scene_ids": [0], "sources": ["after"]},
        "before": {"scene": scene},
        "after": {"scene": scene, "derivation": {"parent": "before"}, "coordinates": "coordinates"},
        "coordinates": {"original": np.arange(4, dtype=np.float32), "shifted64": np.arange(4, dtype=np.float64)+.5,
            "shifted32": np.arange(4, dtype=np.float32)+.5, "q_center": np.arange(4, dtype=np.float32)-1.5,
            "q_probe": np.arange(4, dtype=np.float32)-1.5},
        "inputs": {"records": ["before-input"]}, "derived-inputs": {"records": ["after-input"]},
        "before-input": {"arrays": "channels"}, "after-input": {"arrays": "channels"},
        "channels": {"inputs/1/evidence": np.zeros((2, 8), np.float32)},
        "energies": {"energy": np.array([1., 2.]), "offsets": np.array([0, 2], np.int64)},
        "transport-index": {"scene_ids": [0], "records": ["transport"]},
        "transport": {"scene_id": 0, "prediction": "prediction", "diagnostic": {"fixture": True}}}
    prediction = {"head": "head", "checkpoint_seed": 1, "reader_seed": 1, "arm": "fixture", "stage": "initial",
        "arrays": "energies", "choices": [0]}
    values.update(prediction=prediction, **{"derived-prediction": deepcopy(prediction)})
    before = {"head": "head", "prediction": "prediction", "transport": "transport-index"}
    after = {"head": "head", "prediction": "derived-prediction"}
    batch = {"sources": "sources", "inputs": "inputs", "records": [before]*144,
        "roundtrip": {"sources": "derived-sources", "inputs": "derived-inputs", "records": [after]*144}}
    report = op.probe_report(MemoryStore(values), batch, check=lambda: None)
    assert len(report["heads"]) == 144 and report["scene_ids"] == [0]
    assert all(h["rows"][0]["roundtrip"]["same_exact_choice_by_event"] for h in report["heads"])
    assert report["coordinates"][0]["coordinates"]["max_pairwise_log_ratio_change"]["original_to_q_probe"] == 0
    # These duplicate identities are a mechanical loop fixture, not authority
    # for the real roster: the enclosing COMPLETE/seal gate authenticates it.
