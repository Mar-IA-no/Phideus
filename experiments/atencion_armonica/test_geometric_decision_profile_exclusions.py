"""Existing q8/roundtrip aliases only; no real profile or fresh test draw."""
from copy import deepcopy

import pytest

from experiments.atencion_armonica.test_geometric_decision_scene_store import fitted_scene
from src.atencion_armonica import geometric_decision_profile_exclusions as module
from src.atencion_armonica.geometric_decision_observables import scene_from_sources, roundtrip_observation
from src.atencion_armonica.geometric_decision_scene_store import save_source
from src.atencion_armonica.geometric_decision_store import ArtifactStore
from src.atencion_armonica.generative_evidence_exclusions import Inventory, fingerprint
from src.atencion_armonica.generative_evidence_reuse import VerifiedBytes


@pytest.fixture
def profile_fixture(tmp_path, fitted_scene):
    scene, _ = fitted_scene
    profile = ArtifactStore(tmp_path/"profile", binding={"fixture": "known-q8-aliases"})
    reader = VerifiedBytes(tmp_path)
    original = {**scene["observation"], "scene_id": 0, "split_seed": 2026090880}
    obs_ref = profile.publish_json("known.json", original)
    source = {"path": "profile/known.json", "sha256": obs_ref["sha256"]}
    inventory = Inventory(reader)
    inventory.add("known", [original["log_f"]], source=source)
    prior = profile.publish_json("prior.json", inventory.result())
    prior_ref = {"path": "profile/prior.json", "sha256": prior["sha256"]}
    originals, derived = [], []
    for sid in range(16):
        obs = {**original, "scene_id": sid}
        obsref = profile.publish_json(f"observations/{sid}.json", obs)
        current = scene_from_sources("train", obs, scene["features"], scene["logits"], expected_seed=2026090880)
        originals.append(save_source(profile, f"original/{sid}", current, expected_seed=2026090880,
            derivation={"kind": "original", "observation": obsref}, origin={"fixture": "q8"}))
        if sid < 4:
            changed = roundtrip_observation(obs, expected_seed=2026090880)
            from src.atencion_armonica.partial_compatibility_cache import feature_record
            current = scene_from_sources("train", changed["observation"], feature_record(changed["observation"]),
                scene["logits"], expected_seed=2026090880)
            derived.append(save_source(profile, f"derived/{sid}", current, expected_seed=2026090880,
                derivation={"kind": "roundtrip", "parent": originals[-1]}, origin={"fixture": "q8-roundtrip"}))
    def batch(name, refs, ids):
        return profile.publish_json(name+".json", {"schema": "geometric-decision-source-batch-v1",
            "binding": profile.binding, "features": {}, "logits": {}, "split": "train", "split_seed": 2026090880,
            "scene_ids": ids, "sources": refs})
    result = profile.publish_json("observed.json", {"schema": "geometric-decision-observed-run-v1",
        "binding": profile.binding, "split": "train", "split_seed": 2026090880, "scene_ids": list(range(16)),
        "truth_access": False, "global_seal": False, "sources": batch("original", originals, list(range(16))),
        "roundtrip_scene_ids": list(range(4)), "roundtrip": {"sources": batch("derived", derived, list(range(4)))}})
    report = profile.publish_json("report.json", {"schema": "geometric-decision-observed-profile-v1",
        "binding": profile.binding, "result": result, "recovery": result, "new_test_observations": 0,
        "exclusion_extension_required": True, "roundtrip_scene_ids": list(range(4))})
    return reader, prior_ref, profile, report


def test_extension_preserves_prior_and_all_delivered_coordinates(profile_fixture):
    reader, prior_ref, profile, report = profile_fixture
    before = reader.read(prior_ref)
    result = module.extend_profile_exclusions(reader, prior_ref, profile, report, check=lambda: None)
    assert reader.read(prior_ref) == before
    assert len(result["groups"]) == 33 and result["test_access"] is False
    assert result["extension"]["delivered_coordinate_vectors"] == 16
    assert all(r["declared_alias"] for r in result["groups"][1:17])
    q = profile.json(profile.reference(profile.path("known.json")))
    expected = {fingerprint(v) for k, v in roundtrip_observation(q, expected_seed=2026090880)["coordinates"].items()
                if k != "shifted64"}
    assert set(result["fingerprints"]) == expected
    assert module.checked_prior(result) == expected


def test_extension_rejects_unrecovered_profile_or_shortened_probes(profile_fixture):
    reader, prior_ref, profile, report = profile_fixture
    value = profile.json(report)
    bad = profile.publish_json("not-recovered.json", {**value, "recovery": {}})
    with pytest.raises(ValueError, match="complete OPEN"):
        module.extend_profile_exclusions(reader, prior_ref, profile, bad, check=lambda: None)
    changed = deepcopy(profile.json(value["result"]))
    changed["roundtrip_scene_ids"] = [0, 1, 2]
    ref = profile.publish_json("shortened.json", changed)
    bad = profile.publish_json("shortened-report.json", {**value, "result": ref, "recovery": ref})
    with pytest.raises(ValueError, match="first-four"):
        module.extend_profile_exclusions(reader, prior_ref, profile, bad, check=lambda: None)
