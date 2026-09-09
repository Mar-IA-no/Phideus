"""Read-only integration on pinned OPEN train; mechanical corruption fixtures."""
import hashlib
import json

import numpy as np
import pytest

from src.atencion_armonica import generative_evidence as ge
from src.atencion_armonica import generative_evidence_reuse as reuse
from src.atencion_armonica.generative_evidence_storage import atomic_bytes, write_json
from src.atencion_armonica.partial_compatibility_cache import encoded


@pytest.fixture(scope="module")
def open_reuse():
    return reuse.OpenReuse()


def test_open_import_identity_and_test_exclusion(open_reuse):
    assert open_reuse.prepared["authorization"] == reuse.AUTHORIZATION
    assert {k: len(v) for k, v in open_reuse.shards.items()} == {"train": 8, "calibration": 1}
    for split in ("iid", "ood_beta", "ood_polyphony", "deformed_family", "unknown"):
        with pytest.raises(PermissionError, match="never tests"):
            open_reuse.shard(split, 0)
    for shard in (-1, 8, True):
        with pytest.raises(ValueError, match="fixed open roster"):
            open_reuse.shard("train", shard)


def test_open_scene_local_parity_and_exact_consumed_bytes(open_reuse):
    shard = open_reuse.shard("train", 0)
    a = shard.scene(0)  # Predeclared OPEN sample; no generative fitting.
    assert len(a["inventory"]["candidates"]) == 85
    assert len(a["partitions"]) == 48
    # Stub bounds only to exercise tensorization of the observed candidate list.
    fits = [{"partition": p, "status": "FITTED", "branches": {
        b: {"LB": 1., "UB": 2.} for b in ge.BRANCHES if len(p) in ge.law.BRANCHES[b][2]}}
        for p in a["partitions"]]
    for cp in ge.CHECKPOINTS:
        row = ge.observable_rows(np.asarray(a["observation"]["log_f"], np.float32), a["logits"][cp],
                                 a["features"]["triples"], a["features"]["residual_cents"], a["partitions"], fits)
        old = shard.scored.json(f"seed_{cp}/00000_pool.json")
        lookup = {tuple(g["members"]): g for g in old["groups"]}
        overlap = 0
        for g, values in zip(row["group_ids"], row["groups"]):
            if g in lookup:
                assert values[-1] == lookup[g]["costs"]["local_compatibility"]
                overlap += 1
        assert overlap > 0
        assert row["globals"].shape == (48, 17)
    receipt = open_reuse.receipt()
    assert not any("sidecar" in p or "/targets/" in p for p in receipt["consumed_sha256"])
    for path, digest in receipt["consumed_sha256"].items():
        assert hashlib.sha256((reuse.ROOT/path).read_bytes()).hexdigest() == digest


def test_hash_validation_and_path_conflict(tmp_path):
    raw = encoded({"a": 1})
    atomic_bytes(tmp_path/"a.json", raw)
    ref = {"path": "a.json", "sha256": hashlib.sha256(raw).hexdigest()}
    reader = reuse.VerifiedBytes(tmp_path)
    assert reader.json(ref) == {"a": 1}
    assert reader.consumed == {"a.json": ref["sha256"]}
    with pytest.raises(ValueError, match="hash differs"):
        reader.read({**ref, "sha256": "0"*64})
    with pytest.raises(ValueError, match="relative normalized"):
        reader.read({**ref, "path": "../a.json"})


def test_returned_scene_and_receipt_do_not_alias_reader_state(open_reuse):
    shard = open_reuse.shard("train", 0)
    scene = shard.scene(0)
    original = scene["observation"]["log_f"].copy()
    scene["observation"]["log_f"][0] += 1.
    assert shard.observations[0]["log_f"] == original
    assert shard.scene(0)["observation"]["log_f"] == original
    receipt = open_reuse.receipt()
    digest = receipt["corpora"]["train"]["sha256"]
    receipt["corpora"]["train"]["sha256"] = "0"*64
    assert open_reuse.receipt()["corpora"]["train"]["sha256"] == digest


def test_wrong_manifest_role_and_incomplete_marker(tmp_path):
    manifest = {"schema": "structured-source-bundle-v1", "status": "COMPLETE", "role": "observations",
                "binding": {"common": {"version": 1}}, "artifacts_sha256": {"a": "a"*64},
                "resources_sha256": "b"*64}
    report = write_json(tmp_path/"manifest.json", manifest)
    ref = {"path": "manifest.json", "sha256": report["sha256"]}
    reader = reuse.VerifiedBytes(tmp_path)
    reuse.HistoricalBundle(reader, ref, "observations", {"version": 1})
    with pytest.raises(ValueError, match="role, common"):
        reuse.HistoricalBundle(reader, ref, "targets", {"version": 1})
    with pytest.raises(ValueError, match="role, common"):
        reuse.HistoricalBundle(reader, ref, "observations", {"version": 2})
    write_json(tmp_path/"FAILURE.json", {"status": "INCOMPLETE"})
    with pytest.raises(ValueError, match="incomplete marker"):
        reuse.HistoricalBundle(reader, ref, "observations", {"version": 1})
