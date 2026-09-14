"""Full exclusion roster with in-memory q8 aliases, no files or samplers."""
from copy import deepcopy
import hashlib
import json

import numpy as np
import pytest

from src.atencion_armonica import geometric_decision_exclusions as module
from src.atencion_armonica.partial_compatibility_cache import encoded


class Reader:
    def __init__(self):
        self.values, self.consumed = {}, {}

    def add(self, path, value):
        self.values[path] = encoded(value)
        return {"path": path, "sha256": hashlib.sha256(self.values[path]).hexdigest()}

    def read(self, ref):
        # A sidecar access is never an allowed way to build exclusions.
        assert "sidecar" not in ref["path"] and "evaluation" not in ref["path"]
        raw = self.values[ref["path"]]
        assert hashlib.sha256(raw).hexdigest() == ref["sha256"]
        self.consumed[ref["path"]] = ref["sha256"]
        return raw

    def json(self, ref):
        return json.loads(self.read(ref))


@pytest.fixture
def corpus(monkeypatch):
    reader = Reader()
    q = (np.arange(8, dtype=np.float32)/8).tolist()
    source = reader.add("fixture.json", {"q32": q})
    inventory = module.Inventory(reader)
    inventory.add("arithmetic", [q], source=source)
    prior = reader.add("prior.json", inventory.result())
    freeze = reader.add("freeze.json", {"exclusions": prior, "tests": [
        {"split": s, "split_seed": seed, "scene_count": 512, "scene_ids": list(range(512))}
        for s, seed in module.OLD_TESTS.items()]})
    monkeypatch.setattr(module, "PRIOR", prior)
    monkeypatch.setattr(module, "OLD_FREEZE", freeze)
    tests = {}
    for split, seed in module.OLD_TESTS.items():
        records = []
        for sid in range(512):
            ref = reader.add(f"{module.BASE}/fresh/{split}/source/scenes/{sid:05d}.json",
                {"schema": "generative-evidence-fresh-source-scene-v1", "binding": {"test_freeze": freeze},
                 "split": split, "scene_id": sid, "q32": q})
            records.append({"scene_id": sid, "source": ref})
        common = {"binding": {"test_freeze": freeze}, "split": split, "split_seed": seed,
                  "scene_ids": list(range(512)), "truth_access": False}
        choices = reader.add(f"{split}/choices.json", {**common, "records": records})
        predictions = reader.add(f"{split}/predictions.json", {**common, "choices": choices})
        seal = reader.add(f"{split}/seal.json", {"binding": {"test_freeze": freeze}, "split": split,
            "status": "FRESH_PREDICTIONS_SEALED_NO_TRUTH_ACCESS", "truth_access": False,
            "prediction_count": 45, "prediction_index": predictions})
        tests[split] = {"prediction_seal": seal}
    completion = reader.add("completion.json", {"schema": "generative-evidence-fresh-tests-complete-v1",
        "status": "FRESH_TESTS_EVALUATED_REPLAYED_NOT_PROMOTED", "tests": tests,
        "stage_count": 13, "stages": [{"output": freeze}]+[{}]*12})
    monkeypatch.setattr(module, "COMPLETION", completion)
    observation = reader.add("incident/observation.json", {"scene_id": 0, "split_seed": module.RETIRED, "log_f": q})
    incident = reader.add("incident/receipt.json", {"schema": "r754-contaminated-fixture-reconstruction-v1",
        "status": "DETERMINISTIC_RECONSTRUCTION_NOT_ORIGINAL_BYTE_PRESERVATION", "original_bytes_recoverable": False,
        "scope": {"seed": module.RETIRED, "scene_id": 0}, "reconstructed": {
            "observation_q32_sorted_sha256": module.fingerprint(q),
            "observation": {"path": "observation.json", "sha256": observation["sha256"]}}})
    monkeypatch.setattr(module, "INCIDENT", incident)
    catalog = reader.add("mechanical.json", {"schema": "geometric-decision-mechanical-exclusions-v1",
        "scope": "in-memory q8 aliases only; never prospective data", "retired_seed": module.RETIRED,
        "incident_receipt": incident, "records": [{"name": "q8", "source": source, "q32": [q], "alias": True}]})
    reader.consumed.clear()
    return reader, catalog


def test_full_2048_roster_preserves_aliases_and_distinguishes_new_consumption(corpus):
    reader, catalog = corpus
    calls = []
    result = module.extend_exclusions(reader, catalog, check=lambda: calls.append(1))
    assert result["extension"]["newly_read_historical_observations"] == 2048
    assert len(result["groups"]) == 2050 and result["unique_count"] == 1
    assert result["test_access"] is False and len(calls) >= 2048
    assert result["extension"]["retired_seed"] == 2026091482
    assert result["groups"][-1]["declared_alias"] is True
    assert module.checked_prior(result) == set(result["fingerprints"])


def test_missing_historical_payload_stops_instead_of_skipping(corpus):
    reader, catalog = corpus
    del reader.values[f"{module.BASE}/fresh/ood_polyphony/source/scenes/00017.json"]
    with pytest.raises(KeyError):
        module.extend_exclusions(reader, catalog, check=lambda: None)


def test_wrong_incident_authority_is_rejected(corpus):
    reader, catalog = corpus
    value = reader.json(catalog)
    value["retired_seed"] = 2026091582
    bad = reader.add("wrong.json", value)
    with pytest.raises(ValueError, match="retired seed"):
        module.extend_exclusions(reader, bad, check=lambda: None)


def test_prior_union_and_alias_integrity(corpus):
    reader, _ = corpus
    prior = reader.json(module.PRIOR)
    bad = deepcopy(prior)
    bad["groups"][0]["declared_alias"] = True
    with pytest.raises(ValueError, match="groups"):
        module.checked_prior(bad)
    bad = deepcopy(prior)
    bad["unique_count"] += 1
    with pytest.raises(ValueError, match="union"):
        module.checked_prior(bad)
