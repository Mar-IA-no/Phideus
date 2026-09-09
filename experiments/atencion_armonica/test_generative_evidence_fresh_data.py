"""Fresh-data mechanics with already seen observations; never a fresh draw."""
from copy import deepcopy
import json

import numpy as np
import pytest

from src.atencion_armonica import generative_evidence_exclusions as exclusions
from src.atencion_armonica import generative_evidence_fresh_data as data
from src.atencion_armonica.generative_evidence_reuse import OpenReuse, VerifiedBytes

FREEZE = {"path": "fixture/not-test-authority.json", "sha256": "a"*64}


def inventory(vectors=()):
    hashes = sorted({exclusions.fingerprint(v) for v in vectors})
    return {"schema": exclusions.SCHEMA, "status": "EXCLUSIONS_ONLY_NOT_TEST_AUTHORIZED",
            "fingerprints": hashes, "unique_count": len(hashes), "test_access": False}


def observation(split, scene_id, seed):
    return {"scene_id": scene_id, "split_seed": seed,
            "log_f": (np.arange(8, dtype=np.float32)/8).tolist()}


@pytest.fixture(autouse=True)
def no_real_draws(monkeypatch):
    from src.atencion_armonica import learned_partition_data
    def forbidden(*args, **kwargs):
        raise AssertionError("mechanical fixtures must never call the actual sampler")
    monkeypatch.setattr(learned_partition_data, "_draw_scene", forbidden)


def test_public_producer_requires_verified_freeze_before_any_files(tmp_path, monkeypatch):
    from src.atencion_armonica import generative_evidence_test_freeze as freeze
    calls = []
    def blocked(ref, *, check):
        calls.append(ref)
        raise PermissionError("incomplete test freeze")
    monkeypatch.setattr(freeze, "verify_freeze", blocked)
    monkeypatch.setattr(data, "DRAW_ROOT", tmp_path/"never-created")
    with pytest.raises(PermissionError, match="freeze"):
        data.produce_test("iid", freeze_ref=FREEZE, check=lambda: None)
    assert calls == [FREEZE] and not data.DRAW_ROOT.exists()


@pytest.mark.parametrize("frozen_duplicate,expected_calls", [(True, 1), (False, 2)])
def test_duplicate_preserved_before_stop_and_no_redraw_on_recovery(tmp_path, frozen_duplicate, expected_calls):
    files = data._DrawFiles(tmp_path/"draws", FREEZE)
    q = observation("iid", 0, 0)["log_f"]
    frozen = inventory([q] if frozen_duplicate else [])
    calls = []
    def draw(split, sid, seed):
        calls.append((split, sid, seed))
        return observation(split, sid, seed), {"privileged_fixture": sid}
    for _ in range(2):
        with pytest.raises(RuntimeError, match="duplicate preserved"):
            data._produce_verified(files, "iid", frozen, check=lambda: None, draw=draw)
        assert len(calls) == expected_calls
    folder = files.folder("iid", expected_calls-1)
    value = files.reader.json(files.reference(folder/"draw.json"))
    assert value["status"] == "DUPLICATE_PRESERVED"
    assert value["duplicate_check"]["matches_frozen"] is frozen_duplicate
    assert value["duplicate_check"]["matches_produced"] is not frozen_duplicate
    assert all((folder/name).is_file() for name in ("observation.json", "sidecar.json"))
    assert not files.path("iid/index.json").exists()


def test_interrupted_pair_is_not_a_license_to_repeat_sampler(tmp_path):
    files = data._DrawFiles(tmp_path/"draws", FREEZE)
    folder = files.folder("iid", 0)
    folder.mkdir(parents=True)
    data.storage.write_json(folder/"intent.json", files.intent("iid", 0))
    data.storage.write_json(folder/"observation.json", observation("iid", 0, data.cache.SPLITS["iid"][1]))
    def forbidden(*args):
        raise AssertionError("incomplete pair must not be redrawn")
    with pytest.raises(RuntimeError, match="incomplete pair"):
        data._produce_verified(files, "iid", inventory(), check=lambda: None, draw=forbidden)
    assert not (folder/"sidecar.json").exists()


def test_complete_open_alias_fixture_and_observation_port_never_parse_truth(tmp_path, monkeypatch):
    # Reassign only role metadata, not numeric observations. This is a mechanical
    # alias of the 512 already opened TRAIN observations, not confirming data.
    reuse = OpenReuse()
    bundle = reuse.bundle(reuse.shards["train"][0]["data"], "learned_observation_shard")
    rows = [json.loads(line) for line in bundle.read("observations.jsonl").splitlines()]
    files = data._DrawFiles(tmp_path/"draws", FREEZE)
    calls = []
    def draw(split, sid, seed):
        calls.append(sid)
        return {**deepcopy(rows[sid]), "split_seed": seed}, {"privileged_fixture": sid}
    frozen = inventory()
    ref = data._produce_verified(files, "iid", frozen, check=lambda: None, draw=draw)
    assert len(calls) == 512
    assert data._produce_verified(files, "iid", frozen, check=lambda: None, draw=draw) == ref
    assert len(calls) == 512
    original = VerifiedBytes.json
    def guarded(self, reference):
        assert not reference["path"].endswith("sidecar.json")
        return original(self, reference)
    monkeypatch.setattr(VerifiedBytes, "json", guarded)
    monkeypatch.setattr(data, "DRAW_ROOT", files.root)
    monkeypatch.setattr(data, "_authority", lambda ref, check: {"freeze": FREEZE, "exclusions": frozen})
    port = data.FreshObservations("iid", freeze_ref=FREEZE, check=lambda: None)
    assert port.observation(26)["log_f"] == rows[26]["log_f"]
    # Corruption is still detected even though this port never parses truth.
    sidecar = files.folder("iid", 26)/"sidecar.json"
    sidecar.rename(sidecar.with_suffix(".preserved-original"))
    data.storage.write_json(sidecar, {"corrupted_fixture": True})
    with pytest.raises(ValueError, match="hash"):
        port.observation(26)


def test_noncontiguous_prefix_rejects_new_draw(tmp_path):
    files = data._DrawFiles(tmp_path/"draws", FREEZE)
    files.folder("iid", 1).mkdir(parents=True)
    with pytest.raises(RuntimeError, match="noncontiguous"):
        data._produce_verified(files, "iid", inventory(), check=lambda: None,
            draw=lambda *args: pytest.fail("a missing earlier ID must not be drawn"))
