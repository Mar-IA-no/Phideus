"""Seed-explicit assembly with old arithmetic q32, no prospective draws."""
from copy import deepcopy

import numpy as np
import pytest

from experiments.atencion_armonica.test_generative_evidence import fixture
from src.atencion_armonica import geometric_decision_observables as module
from src.atencion_armonica.partial_compatibility_cache import feature_record


def source():
    q, z, _, _, ps, _ = fixture()
    obs = {"scene_id": 0, "split_seed": 11, "log_f": q.astype(np.float64).tolist()}
    return obs, feature_record(obs), {cp: z.copy() for cp in module.ge.CHECKPOINTS}, ps


def norms():
    return {"common": {str(cp): {"mean": [0.]*5, "scale": [1.]*5} for cp in module.ge.CHECKPOINTS},
        "evidence": {"mean": [0.]*6, "scale": [1.]*6}}


def test_scene_uses_union_and_normalizes_once_per_backbone_with_one_fit_set(monkeypatch):
    obs, features, logits, ps = source()
    calls = []
    def pool(*args):
        calls.append(1)
        return {"partitions": ps}
    monkeypatch.setattr(module, "build_pool", pool)
    scene = module.scene_from_sources("iid", obs, features, logits, expected_seed=11)
    assert len(calls) == 3 and scene["partitions"] == module.ge.partitions_checked(
        sorted(module.ge.law.signature(r["partition"]) for r in scene["inventory"]["candidates"] if r["status"] == "SUPPORTED"), 8)
    # Bounds are explicit stubs, never claimed to result from geometric fitting.
    fits = [{"partition": p, "status": "FITTED", "branches": {b: {"LB": 1., "UB": 2.}
        for b in module.ge.BRANCHES if len(p) in module.ge.law.BRANCHES[b][2]}} for p in scene["partitions"]]
    result = module.inputs_from_fits(scene, fits, norms(), scale=2., expected_seed=11)
    first = result["inputs"][module.ge.CHECKPOINTS[0]]
    assert first["evidence"].shape == (len(fits), 8)
    for cp in module.ge.CHECKPOINTS:
        np.testing.assert_array_equal(result["inputs"][cp]["evidence"], first["evidence"])
        assert result["inputs"][cp]["evidence"].dtype == np.float32
    # Owned copies stop later source mutation changing a preserved scene.
    features["tokens"][:] = 0
    logits[module.ge.CHECKPOINTS[0]][:] = 0
    assert np.any(scene["features"]["tokens"]) and np.any(scene["logits"][module.ge.CHECKPOINTS[0]])


def test_real_pool_on_arithmetic_features_keeps_canonical_event_mapping():
    obs, features, logits, _ = source()
    scene = module.scene_from_sources("calibration", obs, features, logits, expected_seed=11)
    np.testing.assert_array_equal(scene["q32"], np.sort(np.asarray(obs["log_f"], np.float32)))
    assert scene["identity"] and set(scene["pools"]) == {str(cp) for cp in module.ge.CHECKPOINTS}


def test_empty_supported_universe_remains_empty(monkeypatch):
    obs, features, logits, _ = source()
    monkeypatch.setattr(module, "build_pool", lambda *args: {"partitions": []})
    scene = module.scene_from_sources("iid", obs, features, logits, expected_seed=11)
    assert scene["status"] == "NO_OBSERVABLE_CANDIDATE" and scene["partitions"] == []
    result = module.inputs_from_fits(scene, [], norms(), scale=2., expected_seed=11)
    assert all(v["evidence"].shape == (0, 8) for v in result["inputs"].values())


def test_role_seed_observation_feature_and_logit_guards():
    obs, features, logits, _ = source()
    with pytest.raises(ValueError, match="identity"):
        module.scene_from_sources("iid", obs, features, logits, expected_seed=12)
    with pytest.raises(ValueError, match="schema"):
        module.scene_from_sources("iid", {**obs, "labels": []}, features, logits, expected_seed=11)
    with pytest.raises(ValueError, match="feature schema"):
        module.scene_from_sources("iid", obs, {**features, "labels": np.ones(8)}, logits, expected_seed=11)
    changed = deepcopy(logits)
    changed[module.ge.CHECKPOINTS[0]][0, 1] += 1
    with pytest.raises(ValueError, match="symmetric"):
        module.scene_from_sources("iid", obs, features, changed, expected_seed=11)
    with pytest.raises(ValueError, match="role roster"):
        module.scene_from_sources("iid", {**obs, "scene_id": 512}, features, logits, expected_seed=11)


def test_roundtrip_uses_exact_declared_order_and_preserves_events_not_frequency_matches():
    obs, _, _, _ = source()
    obs["log_f"][1] = obs["log_f"][0]  # Explicit repeated-coordinate arithmetic fixture.
    original = deepcopy(obs)
    result = module.roundtrip_observation(obs, expected_seed=11)
    q = np.asarray(obs["log_f"], np.float32)
    shifted = (q.astype(np.float64)+np.log(2.)).astype(np.float32)
    expected = (shifted.astype(np.float64)-shifted.astype(np.float64).mean(dtype=np.float64)).astype(np.float32)
    assert obs == original
    np.testing.assert_array_equal(np.asarray(result["observation"]["log_f"], np.float32), expected)
    assert result["lineage"]["event_ids"] == list(range(8))
    assert result["lineage"]["original_tied_event_groups"] == [[0, 1]]
    assert result["lineage"]["probe_tied_event_groups"] == [[0, 1]]
    assert result["lineage"]["sidecar_access"] is False


def test_exact_integer_json_coordinates_have_one_canonical_identity():
    obs, _, logits, _ = source()
    obs["log_f"] = list(range(8))
    scene = module.scene_from_sources("iid", obs, feature_record(obs), logits, expected_seed=11)
    restored = module.scene_from_sources("iid", scene["observation"], scene["features"], logits, expected_seed=11)
    assert scene["identity"] == restored["identity"]
    assert all(type(x) is float for x in scene["observation"]["log_f"])
    with pytest.raises(ValueError, match="JSON list"):
        module.validate_observation({**obs, "log_f": np.asarray(obs["log_f"])}, scene_id=0, split_seed=11)
