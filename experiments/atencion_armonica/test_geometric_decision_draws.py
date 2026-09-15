"""Once-only mechanics; callback aliases existing observations, never sampler."""
from copy import deepcopy
import json

import numpy as np
import pytest

from src.atencion_armonica import geometric_decision_draws as module
from src.atencion_armonica.geometric_decision_store import ArtifactStore
from src.atencion_armonica.generative_evidence_reuse import OpenReuse


def exclusions(vectors=()):
    hashes = sorted({module.fingerprint(q) for q in vectors})
    return {"schema": module.SCHEMA, "status": "EXCLUSIONS_ONLY_NOT_TEST_AUTHORIZED",
            "fingerprints": hashes, "unique_count": len(hashes), "test_access": False}


def observation(split, sid, seed):
    return {"scene_id": sid, "split_seed": seed, "log_f": (np.arange(8, dtype=np.float32)/8).tolist()}


def files(tmp_path, inventory):
    freeze = {"path": "explicit-mechanical-fixture-not-a-freeze.json", "sha256": "0"*64, "bytes": 0}
    store = ArtifactStore(tmp_path/"draws", binding={"test_freeze": freeze, "fixture": "no-sampler"})
    return module.DrawBatch(store, freeze, exclusions=inventory)


@pytest.mark.parametrize("frozen", [False, True])
def test_duplicate_preserved_and_recovery_never_redraws(tmp_path, frozen):
    q = observation("iid", 0, 11)["log_f"]
    batch = files(tmp_path, exclusions([q] if frozen else []))
    calls = []
    def draw(split, sid, seed):
        calls.append(sid)
        return observation(split, sid, seed), {"privileged_fixture": sid}
    for _ in range(2):
        with pytest.raises(RuntimeError, match="duplicate preserved"):
            batch.produce("iid", draw=draw, check=lambda: None)
        assert calls == ([0] if frozen else [0, 1])
    sid = 0 if frozen else 1
    record = batch.store.json(batch.store.reference(batch.store.path(f"draws/iid/{sid:05d}/draw.json")))
    assert record["status"] == "DUPLICATE_PRESERVED"
    assert record["duplicate_check"]["matches_frozen"] is frozen
    assert not batch.store.path("draws/iid/index.json").exists()


def test_intent_without_complete_pair_cannot_invoke_sampler(tmp_path):
    batch = files(tmp_path, exclusions())
    batch.store.publish_json("draws/iid/00000/intent.json", batch.intent("iid", 0))
    def forbidden(*args):
        raise AssertionError("never redraw an attempted tuple")
    with pytest.raises(RuntimeError, match="incomplete pair"):
        batch.produce("iid", draw=forbidden, check=lambda: None)


def test_noncontiguous_prefix_cannot_fill_holes(tmp_path):
    batch = files(tmp_path, exclusions())
    batch.store.publish_json("draws/iid/00001/intent.json", batch.intent("iid", 1))
    with pytest.raises(RuntimeError, match="noncontiguous"):
        batch.produce("iid", draw=lambda *args: pytest.fail("no draw"), check=lambda: None)


def test_complete512_open_aliases_and_observation_reader_never_parse_sidecars(tmp_path, monkeypatch):
    # Reuse already OPEN observations. Only role metadata is relabeled; no
    # new numerical vector or scene is created and no labels are consumed.
    reuse = OpenReuse()
    bundle = reuse.bundle(reuse.shards["train"][0]["data"], "learned_observation_shard")
    known = [json.loads(line) for line in bundle.read("observations.jsonl").splitlines()]
    batch = files(tmp_path, exclusions())
    calls = []
    original_json = batch.store.json
    def guarded(ref):
        assert not ref["path"].endswith("/sidecar.json"), "observable port parsed truth"
        return original_json(ref)
    monkeypatch.setattr(batch.store, "json", guarded)
    def alias(split, sid, seed):
        calls.append(sid)
        return {**deepcopy(known[sid]), "split_seed": seed}, {"privileged_fixture": sid}
    ref = batch.produce("iid", draw=alias, check=lambda: None)
    assert calls == list(range(512))
    assert batch.produce("iid", draw=lambda *args: pytest.fail("no redraw"), check=lambda: None) == ref
    observed = batch.observations("iid", check=lambda: None)
    assert observed["index"] == ref and len(observed["observations"]) == 512
    assert observed["origin"]["kind"] == "original"
    for actual, old in zip(observed["observations"], known):
        assert actual["log_f"] == old["log_f"]


def test_unknown_split_and_retired_seed_not_in_roster(tmp_path):
    batch = files(tmp_path, exclusions())
    with pytest.raises(ValueError, match="roster"):
        batch.intent("train", 0)
    assert 2026091482 not in dict(module.TESTS).values()
    assert dict(module.TESTS)["iid"] == 2026091582
