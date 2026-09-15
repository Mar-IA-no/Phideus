"""Saved signed states, empty scenes and probes on existing arithmetic fixtures."""
from copy import deepcopy

import numpy as np
import pytest

from experiments.atencion_armonica.test_geometric_decision_core import interface
from experiments.atencion_armonica.test_geometric_decision_inference import empty
from experiments.atencion_armonica.test_geometric_decision_pipeline import inputs, fitted_scene
from experiments.atencion_armonica.test_geometric_decision_scene_store import norms
from src.atencion_armonica import geometric_decision_predictions as module
from src.atencion_armonica import geometric_decision_pipeline as pipeline
from src.atencion_armonica.geometric_decision_head_archive import SHAPES, ROSTER
from src.atencion_armonica.geometric_decision_store import ArtifactStore


def head(store, *, route="geometric", stage="initial"):
    cp, _, seed = ROSTER[0]
    arrays = {k: np.zeros(shape, np.float32) for k, shape in SHAPES.items()}
    if stage == "selected":
        arrays["partition2.bias"][:] = -2
    ref = store.publish_arrays(stage+".npz", arrays)
    return store.publish_json(stage+".json", {"schema": "geometric-decision-frozen-head-v1",
        "binding": store.binding, "checkpoint_seed": cp, "arm": route+"_decision",
        "reader_seed": seed, "stage": stage, "epoch": 0 if stage == "initial" else 5,
        "arrays": ref, "source": {"fixture": "explicit-numeric-parameters"}})


@pytest.fixture
def bundle(tmp_path):
    store = ArtifactStore(tmp_path/"predictions", binding={"fixture": "no-observations-drawn"})
    heads = ArtifactStore(tmp_path/"heads", binding={"fixture": "explicit-heads"})
    row, raw, _ = interface()
    cache = {"ref": {"fixture": "arithmetic-inputs"}, "binding": store.binding,
        "scene_ids": [0, 1, 2], "split": "iid", "split_seed": 11,
        "rows": {cp: [row, empty(), row] for cp in module.ge.CHECKPOINTS},
        "raw": {cp: [raw, {"partitions": []}, raw] for cp in module.ge.CHECKPOINTS},
        "probe_indices": [0, 2]}
    return store, heads, cache


@pytest.mark.parametrize("stage", ["initial", "selected"])
def test_saved_head_not_recreated_initialization_and_recovery(bundle, stage, monkeypatch):
    store, heads, cache = bundle
    head_ref = head(heads, stage=stage)
    kwargs = {"device": "cpu", "runtime": {"fixture": "CPU"}, "check": lambda: None}
    ref = module.preserve_prediction(store, "original", cache, heads, head_ref, **kwargs)
    record = store.json(ref)
    arrays = store.arrays(record["arrays"])
    cp = ROSTER[0][0]
    expected = np.tile(cache["rows"][cp][0]["evidence"][:, 6].astype(np.float64), 2)
    if stage == "selected":
        expected -= 4
    np.testing.assert_array_equal(arrays["energy"], expected)
    assert record["choices"][1] is None
    assert arrays["offsets"][1] == arrays["offsets"][2]
    def forbidden(*args, **kwargs):
        raise AssertionError("recovery must not re-forward or build model")
    monkeypatch.setattr(module, "archived_model", forbidden)
    assert module.preserve_prediction(store, "original", cache, heads, head_ref, **kwargs) == ref


@pytest.mark.parametrize("route", ["injection", "geometric", "decoupled", "local"])
@pytest.mark.parametrize("stage", ["initial", "selected"])
def test_transport_preserves_all_arrays_and_batch_comparison(bundle, monkeypatch, route, stage):
    store, heads, cache = bundle
    head_ref = head(heads, route=route, stage=stage)
    ref = module.preserve_prediction(store, "original", cache, heads, head_ref,
        device="cpu", runtime={"fixture": "CPU"}, check=lambda: None)
    probe = module.preserve_transport(store, "transport", cache, ref, heads, check=lambda: None)
    index = store.json(probe)
    assert index["scene_ids"] == [0, 2]
    for item in index["records"]:
        record = store.json(item)
        arrays = store.arrays(record["arrays"])
        assert record["diagnostic"]["bypass_channel"] == {"geometric": 1, "decoupled": 0}.get(route)
        assert record["diagnostic"]["within_numeric_tolerance"]
        assert record["diagnostic"]["singleton_vs_batch_same_exact_choice"]
        assert {"batched_components", "restored_energy", "channel_order", "weight_column_order",
            "baseline/components", "transported_inputs/evidence"} <= set(arrays)
    def forbidden(*args, **kwargs):
        raise AssertionError("recovered probe must not forward")
    monkeypatch.setattr(module, "transport_prediction", forbidden)
    monkeypatch.setattr(module, "archived_model", forbidden)
    assert module.preserve_transport(store, "transport", cache, ref, heads, check=lambda: None) == probe


def test_orphan_prediction_stops_without_model(bundle, monkeypatch):
    store, heads, cache = bundle
    head_ref = head(heads)
    cp, _, seed = ROSTER[0]
    store.publish_arrays(f"original/cp_{cp}/geometric_decision/seed_{seed}/initial.npz", {"orphan": np.zeros(1)})
    def forbidden(*args, **kwargs):
        raise AssertionError("no model before orphan reconciliation")
    monkeypatch.setattr(module, "archived_model", forbidden)
    with pytest.raises(RuntimeError, match="unreceipted"):
        module.preserve_prediction(store, "original", cache, heads, head_ref,
            device="cpu", runtime={"fixture": "CPU"}, check=lambda: None)


def test_input_batch_loads_real_fixture_sources_once_and_rejects_wrong_ids(inputs):
    store, scene, fitted, observations, args = inputs
    source = pipeline.prepare_sources(store, "original", "iid", observations,
        forward=lambda cp, records: [scene["logits"][cp["seed"]].copy()], **args)
    norm_ref = {"fixture": "unit-norms"}
    ref = pipeline.prepare_inputs(store, "original", source, normalizers=norms(), normalization_ref=norm_ref,
        scale=2., fit_origin={"fixture": "cached5x5"}, fit_candidates=lambda *args: deepcopy(fitted), check=lambda: None)
    cache = module.input_batch(store, ref, normalization_ref=norm_ref, scale=2., check=lambda: None)
    assert cache["scene_ids"] == [0] and cache["probe_indices"] == [0]
    assert set(cache["rows"]) == set(module.ge.CHECKPOINTS)
    value = store.json(ref)
    source_value = store.json(value["sources"])
    source_value["targets"] = {"fixture": "forbidden-extra-supervision"}
    with_targets = {**value, "sources": store.publish_json("source-with-targets.json", source_value)}
    bad = store.publish_json("input-with-targets.json", with_targets)
    with pytest.raises(ValueError, match="source roster"):
        module.input_batch(store, bad, normalization_ref=norm_ref, scale=2., check=lambda: None)
    value["scene_ids"] = [1]
    bad = store.publish_json("wrong-ids.json", value)
    with pytest.raises(ValueError, match="roster"):
        module.input_batch(store, bad, normalization_ref=norm_ref, scale=2., check=lambda: None)


def test_prediction_reader_rejects_tolerance_as_exact_initial_rule(bundle):
    store, heads, cache = bundle
    ref = module.preserve_prediction(store, "original", cache, heads, head(heads),
        device="cpu", runtime={"fixture": "CPU"}, check=lambda: None)
    arrays = store.arrays(store.json(ref)["arrays"])
    arrays["components"][0, 0] += 1e-12
    arrays["energy"] = arrays["components"].sum(-1, dtype=np.float64)
    cp = ROSTER[0][0]
    with pytest.raises(ValueError, match="imposed bypass"):
        module.checked_prediction(arrays, cache["rows"][cp], cache["raw"][cp], route="geometric", initial=True)


@pytest.mark.parametrize("mutation", ["diagnostic", "extra-record", "extra-array", "missing-array",
    "nonfinite", "wrong-shape", "candidate-order", "channels", "weights", "inputs", "energy", "restored", "batched"])
def test_transport_recovery_revalidates_payload_without_forward(bundle, monkeypatch, mutation):
    store, heads, cache = bundle
    ref = module.preserve_prediction(store, "original", cache, heads, head(heads, stage="selected"),
        device="cpu", runtime={"fixture": "CPU"}, check=lambda: None)
    probe = module.preserve_transport(store, "transport", cache, ref, heads, check=lambda: None)
    target = store.json(probe)["records"][0]
    record = store.json(target)
    arrays = store.arrays(record["arrays"])
    if mutation == "diagnostic":
        record["diagnostic"]["max_energy_error"] += 1.
    elif mutation == "extra-record":
        record["labels"] = []
    elif mutation == "extra-array":
        arrays["labels"] = np.zeros(1)
    elif mutation == "missing-array":
        del arrays["restored_components"]
    elif mutation == "nonfinite":
        arrays["transported_components"][0, 0] = np.nan
    elif mutation == "wrong-shape":
        arrays["transported_energy"] = arrays["transported_energy"][:, None]
    else:
        key = {"candidate-order": "candidate_order", "channels": "channel_order", "weights": "weight_column_order",
            "inputs": "transported_inputs/evidence", "energy": "transported_energy",
            "restored": "restored_components", "batched": "batched_components"}[mutation]
        arrays[key].flat[0] += 1
    original_json, original_arrays = store.json, store.arrays
    monkeypatch.setattr(store, "json", lambda r: deepcopy(record) if r == target else original_json(r))
    monkeypatch.setattr(store, "arrays", lambda r: deepcopy(arrays) if r == record["arrays"] else original_arrays(r))
    def forbidden(*args, **kwargs):
        raise AssertionError("no forward or model during malformed transport recovery")
    monkeypatch.setattr(module, "archived_model", forbidden)
    monkeypatch.setattr(module, "transport_prediction", forbidden)
    with pytest.raises(ValueError):
        module.preserve_transport(store, "transport", cache, ref, heads, check=lambda: None)


@pytest.mark.parametrize("indices", [[], [0], [2, 0], [0, 1, 2]])
def test_probe_roster_recomputed_before_model_or_publication(bundle, monkeypatch, indices):
    store, heads, cache = bundle
    ref = module.preserve_prediction(store, "original", cache, heads, head(heads),
        device="cpu", runtime={"fixture": "CPU"}, check=lambda: None)
    cache["probe_indices"] = indices
    def forbidden(*args, **kwargs):
        raise AssertionError("invalid probe roster must stop before model or publication")
    monkeypatch.setattr(module, "archived_model", forbidden)
    monkeypatch.setattr(store, "publish_json", forbidden)
    with pytest.raises(ValueError, match="first four eligible"):
        module.preserve_transport(store, "transport", cache, ref, heads, check=lambda: None)
