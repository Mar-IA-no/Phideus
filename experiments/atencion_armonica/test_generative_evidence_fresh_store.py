"""Observable storage fixtures using only already opened q32 observations."""
from copy import deepcopy

import numpy as np
import pytest

from experiments.atencion_armonica.test_generative_evidence_prepared import empty_scene
from src.atencion_armonica import generative_evidence as ge
from src.atencion_armonica import generative_evidence_cache as cache
from src.atencion_armonica import generative_evidence_fresh_store as fresh

BINDING = {"test_freeze": {"path": "fixture/not-a-test-permission.json", "sha256": "a"*64}}
ORIGIN = {"fixture": "already_opened_observations_not_fresh_draws"}


@pytest.fixture(autouse=True)
def forbid_production(monkeypatch):
    from src.atencion_armonica import learned_partition_data
    def forbidden(*a, **kw):
        raise AssertionError("storage fixtures cannot draw or fit campaign data")
    monkeypatch.setattr(learned_partition_data, "_draw_scene", forbidden)
    monkeypatch.setattr(ge.law, "fit_candidates", forbidden)
    monkeypatch.setattr(ge.law.GroupFitter, "fit", forbidden)


def scene(scene_id, split="iid"):
    value = empty_scene(scene_id)
    value["observation"]["split_seed"] = cache.SPLITS[split][1]
    return value


def save_empty(store, scene_id, split="iid"):
    value = scene(scene_id, split)
    store.save_fit(split, scene_id, value, {"fits": [], "group_factors": []}, origin=ORIGIN)
    return store.save_observables(split, scene_id, value)


def test_fresh_roles_and_exact_q32_reject_open_and_old_seed():
    obs = scene(0)["observation"]
    assert np.array_equal(fresh.validate_observation(obs, 0, "iid"), np.arange(8, dtype=np.float32)/8)
    with pytest.raises(PermissionError, match="OPEN"):
        fresh.validate_observation(obs, 0, "train")
    for changed in ({**obs, "split_seed": 2026090882}, {**obs, "scene_id": True},
                    {**obs, "log_f": [.1]+obs["log_f"][1:]}):
        with pytest.raises(ValueError):
            fresh.validate_observation(changed, 0, "iid")


def test_empty_recovery_has_no_supervision_ports_or_test_authority(tmp_path):
    store = fresh.FreshObservableStore(tmp_path/"store", binding=BINDING)
    ref = save_empty(store, 0)
    restored = fresh.FreshObservableStore(store.root, binding=BINDING)
    assert save_empty(restored, 0) == ref
    assert restored.json(ref)["status"] == "NO_OBSERVABLE_CANDIDATE"
    for cp in ge.CHECKPOINTS:
        value = restored.load_row("iid", 0, cp)
        assert value["rows"][0]["evidence"].shape == (0, 6)
    for method in ("save_supervision", "load_supervision", "seal_shard", "seal_split", "load_shard"):
        assert not hasattr(restored, method)
    with pytest.raises(PermissionError):
        restored.load_fit("calibration", 0)
    with pytest.raises(ValueError, match="freeze"):
        fresh.FreshObservableStore(tmp_path/"bad", binding={"purpose": "no-freeze"})
    with pytest.raises(ValueError, match="binding"):
        fresh.FreshObservableStore(store.root, binding={"test_freeze": {**BINDING["test_freeze"], "sha256": "b"*64}})


def test_real_preserved_fit_and_raw_rows_roundtrip_without_forward(tmp_path):
    from src.atencion_armonica.generative_evidence_preparation import ProfileFits
    from src.atencion_armonica.generative_evidence_reuse import OpenReuse
    original = OpenReuse().shard("train", 0).scene(0)
    fitted = ProfileFits().fitted("train", 0, original)
    value = deepcopy(original)
    value["observation"]["split_seed"] = cache.SPLITS["iid"][1]
    store = fresh.FreshObservableStore(tmp_path/"real-profile", binding=BINDING)
    fit = store.save_fit("iid", 0, value, fitted, origin=ORIGIN)
    observable = store.save_observables("iid", 0, value)
    assert store.load_fit("iid", 0)[1] == fit
    assert store.save_observables("iid", 0, value) == observable
    identities = []
    for cp in ge.CHECKPOINTS:
        loaded = store.load_row("iid", 0, cp)
        expected = ge.observable_rows(np.asarray(value["observation"]["log_f"], np.float32),
            value["logits"][cp], value["features"]["triples"], value["features"]["residual_cents"],
            value["partitions"], fitted["fits"])
        assert np.array_equal(loaded["rows"][0]["evidence"], expected["evidence"])
        assert np.array_equal(loaded["rows"][0]["groups"], expected["groups"])
        identities += loaded["identities"]
    assert len(set(identities)) == 1


def test_complete_aggregate_and_corruption_are_not_truth_permission(tmp_path):
    store = fresh.FreshObservableStore(tmp_path/"aggregate", binding=BINDING)
    save_empty(store, 0)
    with pytest.raises(FileNotFoundError):
        store.seal_observables("iid", check=lambda: None)
    assert not store.path("iid/observables.json").exists()
    for i in range(1, 512):
        save_empty(store, i)
    fitted, _ = store._record("iid", 0, "fit")
    factor_path = store.path(fitted["artifact"]["path"])
    original_factor = factor_path.read_bytes()
    factor_path.write_bytes(original_factor[:-1])
    with pytest.raises(ValueError, match="fit artifact"):
        store.seal_observables("iid", check=lambda: None)
    assert not store.path("iid/observables.json").exists()
    factor_path.write_bytes(original_factor)
    ref = store.seal_observables("iid", check=lambda: None)
    assert store.json(ref)["status"] == fresh.STATUS
    assert store.seal_observables("iid", check=lambda: None) == ref
    for cp in ge.CHECKPOINTS:
        result = store.load_observables("iid", cp)
        assert result["scene_ids"] == list(range(512))
        assert all(not row["partitions"] for row in result["rows"])
    index = store.json(ref)
    path = store.path(index["raw"][str(ge.CHECKPOINTS[0])]["path"])
    raw = path.read_bytes()
    path.write_bytes(raw[:-1])  # Only the exact fixture file owned by this test.
    with pytest.raises(ValueError, match="identity"):
        store.seal_observables("iid", check=lambda: None)
    path.write_bytes(raw)
    assert store.seal_observables("iid", check=lambda: None) == ref
    factor_path.write_bytes(original_factor[:-1])
    with pytest.raises(ValueError, match="fit artifact"):
        store.seal_observables("iid", check=lambda: None)
    factor_path.write_bytes(original_factor)
    retained = factor_path.with_suffix(".preserved")
    factor_path.rename(retained)
    with pytest.raises(FileNotFoundError):
        store.load_observables("iid", ge.CHECKPOINTS[0])
    retained.rename(factor_path)
    assert store.seal_observables("iid", check=lambda: None) == ref
