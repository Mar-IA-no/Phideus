"""Observable storage fixtures; tiny CPU fitting, no generator or real data."""
from copy import deepcopy

import numpy as np
import pytest

from experiments.atencion_armonica.test_geometric_decision_observables import source, norms
from src.atencion_armonica import geometric_decision_observables as obsmod
from src.atencion_armonica import geometric_decision_scene_store as module
from src.atencion_armonica.geometric_decision_store import ArtifactStore


def save_fixture_source(store, folder, scene, **kwargs):
    observation = store.publish_json(folder+"/observation.json", scene["observation"])
    return module.save_source(store, folder, scene,
        derivation={"kind": "original", "observation": observation}, **kwargs)


@pytest.fixture(scope="module")
def fitted_scene():
    obs, features, logits, _ = source()
    # Two explicit four-component blocks force a supported observable cut.
    # The previous additive logits produced only unsupported cuts: retaining
    # that fixture would test empty storage, never group-factor preservation.
    for z in logits.values():
        z[:] = -8
        z[:4, :4] = 8
        z[4:, 4:] = 8
    # Actual observable operations, but only the historical arithmetic q8
    # vector and a tiny 5x5 fixture grid. Never a prospective scene or grid.
    scene = obsmod.scene_from_sources("iid", obs, features, logits, expected_seed=11)
    assert scene["partitions"]
    fitter = module.ge.law.GroupFitter(module.ge.law.Grid(5, 5), device="cpu")
    fitted = module.ge.law.fit_candidates(scene["q32"], scene["partitions"], fitter)
    return scene, fitted


def test_roundtrip_source_fit_and_inputs_are_idempotent_without_refit(tmp_path, fitted_scene, monkeypatch):
    scene, fitted = fitted_scene
    store = ArtifactStore(tmp_path/"scene", binding={"fixture": "arithmetic-store-not-test-authority"})
    origin = {"fixture": "explicit-q8-and-logits"}
    source_ref = save_fixture_source(store, "iid/0", scene, origin=origin, expected_seed=11)
    saved = module.load_source(store, source_ref)
    assert module.encoded(module.source_metadata(saved)) == module.encoded(module.source_metadata(scene))
    fit_ref = module.save_fit(store, "iid/0", source_ref, fitted, origin={"grid": "fixture-5x5"})
    assert module.encoded(module.load_fit(store, fit_ref)) == module.encoded(fitted)
    normalization = {"fixture": "explicit-unit-normalizers-and-scale"}
    inputs_ref = module.save_inputs(store, "iid/0", source_ref, fit_ref, norms(), scale=2., normalization_ref=normalization)
    actual = module.load_inputs(store, inputs_ref, normalization_ref=normalization)
    expected = obsmod.inputs_from_fits(scene, fitted["fits"], norms(), scale=2., expected_seed=11)
    for cp in module.ge.CHECKPOINTS:
        module.assert_same_arrays(actual["inputs"][cp], expected["inputs"][cp])
    def forbidden(*args, **kwargs):
        raise AssertionError("storage cannot refit")
    monkeypatch.setattr(module.ge.law.GroupFitter, "fit", forbidden)
    assert save_fixture_source(store, "iid/0", scene, origin=origin, expected_seed=11) == source_ref
    assert module.save_fit(store, "iid/0", source_ref, fitted, origin={"grid": "fixture-5x5"}) == fit_ref
    assert module.save_inputs(store, "iid/0", source_ref, fit_ref, norms(), scale=2., normalization_ref=normalization) == inputs_ref
    with pytest.raises(ValueError, match="normalization"):
        module.load_inputs(store, inputs_ref, normalization_ref={"fixture": "another-normalizer"})


def test_source_rejects_extra_fields_wrong_seed_and_changed_universe(tmp_path, fitted_scene):
    scene, _ = fitted_scene
    store = ArtifactStore(tmp_path/"scene", binding={"fixture": "source-guards"})
    with pytest.raises(ValueError, match="reconstruction"):
        save_fixture_source(store, "bad", {**scene, "targets": []}, origin={"fixture": 1}, expected_seed=11)
    with pytest.raises(ValueError, match="identity"):
        save_fixture_source(store, "bad", scene, origin={"fixture": 1}, expected_seed=12)
    with pytest.raises(ValueError, match="reconstruction"):
        save_fixture_source(store, "bad", {**scene, "identity": "wrong"}, origin={"fixture": 1}, expected_seed=11)


def test_fit_requires_full_factor_roster(tmp_path, fitted_scene):
    scene, fitted = fitted_scene
    assert scene["partitions"], "fixture must exercise nonempty fitting and storage"
    assert fitted["group_factors"]
    bad = deepcopy(fitted)
    bad["group_factors"].pop()
    with pytest.raises(ValueError, match="incomplete"):
        module.validate_fits(scene, bad)
    bad = deepcopy(fitted)
    bad["group_factors"][0]["factor"]["fine"].pop()
    with pytest.raises(ValueError, match="assignments"):
        module.validate_fits(scene, bad)


def test_corrupt_source_payload_rejected_before_loading_inputs(tmp_path, fitted_scene):
    scene, _ = fitted_scene
    store = ArtifactStore(tmp_path/"scene", binding={"fixture": "missing-payload"})
    ref = save_fixture_source(store, "iid/0", scene, origin={"fixture": 1}, expected_seed=11)
    arrays = store.json(ref)["arrays"]
    store.path(arrays["path"]).unlink()  # Only this test's own fixture payload.
    with pytest.raises(FileNotFoundError):
        module.load_source(store, ref)


@pytest.fixture
def delivered_store(tmp_path, fitted_scene):
    scene, fitted = fitted_scene
    store = ArtifactStore(tmp_path/"scene", binding={"fixture": "delivered-guards"})
    source_ref = save_fixture_source(store, "iid/0", scene, origin={"fixture": 1}, expected_seed=11)
    fit_ref = module.save_fit(store, "iid/0", source_ref, fitted, origin={"grid": "fixture-5x5"})
    norm = {"fixture": "unit-normalizers"}
    ref = module.save_inputs(store, "iid/0", source_ref, fit_ref, norms(), scale=2., normalization_ref=norm)
    return store, ref, norm


@pytest.mark.parametrize("mutation", ["bypass", "raw_dtype", "group_ids", "event_map", "donors", "extra"])
def test_delivered_reader_rejects_self_consistent_receipts_with_wrong_rows(delivered_store, mutation):
    store, ref, norm = delivered_store
    record = store.json(ref)
    arrays = store.arrays(record["arrays"])
    cp = str(module.ge.CHECKPOINTS[0])
    if mutation == "bypass":
        arrays[f"inputs/{cp}/evidence"][0, 6] += 1
    elif mutation == "raw_dtype":
        arrays[f"raw/{cp}/groups"] = arrays[f"raw/{cp}/groups"].astype(np.float32)
    elif mutation == "group_ids":
        record["metadata"][cp]["group_ids"].reverse()
    elif mutation == "event_map":
        arrays[f"raw/{cp}/canonical_to_observed"] = arrays[f"raw/{cp}/canonical_to_observed"][::-1]
    elif mutation == "donors":
        record["diagnostics"][cp]["scalar_changed_mask"][0] = not record["diagnostics"][cp]["scalar_changed_mask"][0]
    else:
        record["metadata"][cp]["targets"] = []
    record["arrays"] = store.publish_arrays("tampered.npz", arrays)
    forged = store.publish_json("tampered.json", record)
    with pytest.raises((ValueError, AssertionError)):
        module.load_inputs(store, forged, normalization_ref=norm)


@pytest.mark.parametrize("payload", ["source", "fit"])
def test_delivered_reader_authenticates_underlying_payloads(delivered_store, payload):
    store, ref, norm = delivered_store
    record = store.json(ref)
    linked = store.json(record[payload])
    artifact = linked["arrays"] if payload == "source" else linked["artifact"]
    store.path(artifact["path"]).unlink()  # This test's own fixture bytes only.
    with pytest.raises(FileNotFoundError):
        module.load_inputs(store, ref, normalization_ref=norm)


@pytest.mark.parametrize("mutation", ["candidate_target", "factor_target", "assignment_target", "ub", "witness", "family"])
def test_fit_deep_schema_and_factor_replay_are_closed(fitted_scene, mutation):
    scene, fitted = fitted_scene
    bad = deepcopy(fitted)
    if mutation == "candidate_target":
        bad["fits"][0]["targets"] = [1.]
    elif mutation == "factor_target":
        bad["group_factors"][0]["factor"]["supervision"] = 1.
    elif mutation == "assignment_target":
        bad["group_factors"][0]["factor"]["fine"][0]["labels"] = [0]*4
    elif mutation == "ub":
        next(iter(bad["fits"][0]["branches"].values()))["UB"] += 1
    elif mutation == "witness":
        next(iter(bad["fits"][0]["branches"].values()))["witness"]["prediction"][0] += .01
    else:
        next(iter(bad["fits"][0]["families"].values()))["UB"] += 1
    with pytest.raises(ValueError):
        module.validate_fits(scene, bad)


def test_probe_requires_authenticated_parent_and_preserves_checked_lineage(tmp_path, fitted_scene):
    scene, _ = fitted_scene
    store = ArtifactStore(tmp_path/"scene", binding={"fixture": "typed-roundtrip"})
    original = save_fixture_source(store, "original", scene, origin={"fixture": 1}, expected_seed=11)
    result = obsmod.roundtrip_observation(scene["observation"], expected_seed=11)
    observation = result["observation"]
    # Feature computation is observable, never a sampler.
    from src.atencion_armonica.partial_compatibility_cache import feature_record
    probe = obsmod.scene_from_sources("iid", observation, feature_record(observation), scene["logits"], expected_seed=11)
    with pytest.raises(ValueError, match="parent"):
        module.save_source(store, "bad", probe, origin={"fixture": 1}, expected_seed=11, derivation={"kind": "roundtrip"})
    ref = module.save_source(store, "probe", probe, origin={"fixture": 1}, expected_seed=11,
        derivation={"kind": "roundtrip", "parent": original})
    record = store.json(ref)
    assert record["derivation"]["lineage"] == result["lineage"]
    module.assert_same_arrays(store.arrays(record["coordinates"]), result["coordinates"])
    assert module.load_source(store, ref)["identity"] == probe["identity"]
    record["derivation"]["lineage"]["event_ids"].reverse()
    forged = store.publish_json("wrong-lineage.json", record)
    with pytest.raises(ValueError, match="lineage"):
        module.load_source(store, forged)
    other_obs = {**scene["observation"], "scene_id": 1}
    other = obsmod.scene_from_sources("iid", other_obs, feature_record(other_obs), scene["logits"], expected_seed=11)
    other_ref = save_fixture_source(store, "other", other, origin={"fixture": 1}, expected_seed=11)
    with pytest.raises(ValueError, match="parent transformation"):
        module.save_source(store, "wrong-parent", probe, origin={"fixture": 1}, expected_seed=11,
            derivation={"kind": "roundtrip", "parent": other_ref})
