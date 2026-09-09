"""Ports, immutable publication and post-seal evaluation, using mechanical fixtures."""
from dataclasses import asdict
import hashlib
import json

import numpy as np
import pytest

from src.atencion_armonica import observable_rival_campaign as c
from src.atencion_armonica import observable_rival_evaluation as e
from src.atencion_armonica import observable_source_rivals as r


def test_exact_roster():
    expected = {
        "iid": [13,30,66,131,163,190,228,231,277,297,308,341,348,367,391,398,424,444,445,453,472,482,492,507],
        "ood_beta": [6,9,13,19,49,89,107,113,127,147,167,183,211,231,236,265,271,309,335,337,381,391,399,450],
        "ood_polyphony": [1,18,87,92,139,181,242,257,272,293,304,307,354,367,372,376,391,408,431,451,491,492,494,499],
        "deformed_family": [6,9,35,50,65,81,86,90,111,149,162,169,250,257,278,279,304,350,440,466,471,472,487,500]}
    assert c.selected_ids() == expected


def test_immutable_publication_and_hashes(tmp_path):
    p = tmp_path/"result.json"
    ref = c.write_once(p, {"a": 1})
    assert c.checked_json(ref) == {"a": 1}
    assert c.write_once(p, {"a": 1}) == ref
    with pytest.raises(FileExistsError):
        c.write_once(p, {"a": 2})
    with pytest.raises(ValueError):
        c.checked_json({"path": str(p), "sha256": "0"*64})
    assert not list(tmp_path.glob("*.tmp"))


def test_budget_recovery_refuses_unknown_attempt(tmp_path):
    assert c.elapsed_budget(tmp_path) == 0
    p = tmp_path/"attempts"/"one.json"
    c.write_once(p, {"kind": "observable"})
    with pytest.raises(RuntimeError):
        c.elapsed_budget(tmp_path)
    c.write_once(p.with_suffix(".end.json"), {"seconds": 3.5})
    assert c.elapsed_budget(tmp_path) == 3.5


def test_evaluation_does_not_parse_truth_before_complete_seal(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(c, "read_bundle_member", lambda *args: calls.append(args))
    with pytest.raises(FileNotFoundError):
        e.run(tmp_path)
    assert not calls


def test_observable_inventory_has_no_sidecar_port(tmp_path, monkeypatch):
    # Tiny substituted source inventory, with explicit traps on privileged files.
    ids = {split: [0] for split in c.SPLITS}
    monkeypatch.setattr(c, "selected_ids", lambda: ids)
    q = np.arange(8, dtype=np.float32).tolist()
    p = [[0,1,2,3], [4,5,6,7]]
    entry = {"data": {"path": "data.json", "sha256": "mock"},
             "scored": {"path": "scored.json", "sha256": "mock"}}
    monkeypatch.setattr(c, "checked_json", lambda ref: {"splits": {s: entry for s in c.SPLITS}})
    observed = tmp_path/"observations.jsonl"
    observed.write_bytes(b"\n".join(c.encoded({"scene_id": i, "split_seed": 1, "log_f": q}).strip()
                                   for i in range(512))+b"\n")
    pool = tmp_path/"pool.json"
    c.write_once(pool, {"pool": {"canonical_q32": q, "canonical_to_observed": list(range(8)),
                                 "q_tie_count": 0, "partitions": [p]}})
    counter = 0
    def member(ref, name):
        nonlocal counter
        assert "sidecar" not in name and "metrics" not in name
        split = c.SPLITS[counter//4]
        counter += 1
        path = observed if name == "observations.jsonl" else pool
        return path, c.reference(path), {"binding": {"split": split, "split_seed": 1, "data": entry["data"]}}
    monkeypatch.setattr(c, "read_bundle_member", member)
    result = c.load_observable_inventory()
    assert len(result) == 4 and counter == 16
    assert all(len(s["candidates"]) == 17 for s in result)


def truth_fixture():
    sources = [{"f0": f, "beta": 1e-4, "gamma": 0., "indices": [1,2,3,4]}
               for f in (110., 420.)]
    ideal = []
    for source in sources:
        ns = np.asarray(source["indices"], np.float64)
        ideal.extend((np.log(source["f0"])+np.log(ns)+.5*np.log1p(source["beta"]*ns**2)).tolist())
    ideal = np.asarray(ideal)
    noise = np.linspace(-2, 2, len(ideal))*r.CENTS_TO_LOG
    perm = np.array([5,1,7,0,6,2,4,3])
    center = float((ideal+noise).mean())
    observed = (ideal+noise-center).astype(np.float32)[perm]
    order = np.argsort(observed, kind="stable")
    scene = {"scene_id": 0, "split": "iid", "q32": observed[order].tolist(),
             "canonical_to_observed": order.tolist()}
    truth = {"scene_id": 0, "split_seed": 1, "sigma_cents": 2., "sources": sources,
             "permutation": perm.tolist(), "log_f_ideal": ideal[perm].tolist(),
             "sensor_log_noise": noise[perm].tolist(), "source_ids": np.repeat([0,1], 4)[perm].tolist(),
             "partial_indices": np.tile([1,2,3,4], 2)[perm].tolist(), "mean_log_f_observed": center}
    return scene, truth


def test_preserved_truth_reconstruction_and_no_redraw():
    scene, truth = truth_fixture()
    result = e.validate_truth(scene, truth)
    assert sorted(len(g) for g in result["partition"]) == [4,4]
    changed = json.loads(json.dumps(truth))
    changed["sensor_log_noise"][0] += .01
    with pytest.raises(ValueError):
        e.validate_truth(scene, changed)
    changed = json.loads(json.dumps(truth))
    changed["sources"][0]["f0"] = 1000.
    with pytest.raises(ValueError):
        e.validate_truth(scene, changed)


def test_reader_means_preserve_scene_unit_and_reject_missing_cells(monkeypatch):
    monkeypatch.setattr(c, "selected_ids", lambda: {"iid": [0]})
    records = [{"scene_id": i, "checkpoint_seed": cp, "reader_seed": seed, "split": "iid",
                "metrics": {name: {"ari": .1*j} for name in e.READERS}}
               for i in range(512) for j, cp in enumerate(c.CHECKPOINTS) for seed in e.READER_SEEDS]
    result = e.reader_scene_means(records, "iid")
    assert all(abs(v-.1) < 1e-15 for v in result[0].values())
    with pytest.raises(ValueError):
        e.reader_scene_means(records[:-1], "iid")


def test_seal_checks_every_hash_before_supervision(tmp_path, monkeypatch):
    monkeypatch.setattr(c, "verify_binding", lambda m: None)
    scenes, refs = [], []
    for split, ids in c.selected_ids().items():
        for sid in ids:
            scene = {"split": split, "scene_id": sid}
            scenes.append(scene)
            path = tmp_path/"observable"/split/f"{sid:05d}.json"
            refs.append(c.write_once(path, {"split": split, "scene_id": sid,
                "input_sha256": hashlib.sha256(c.encoded(scene)).hexdigest()}))
    manifest_ref = c.write_once(tmp_path/"campaign.json", {"scenes": scenes,
                               "binding": {"scene_ids": c.selected_ids()}})
    c.write_once(tmp_path/"observable_seal.json", {"count": 96, "campaign": manifest_ref, "scenes": refs})
    assert len(c.verify_seal(tmp_path)[1]) == 96
    # Tamper a referenced immutable payload in this isolated mechanical fixture only.
    from pathlib import Path
    Path(refs[-1]["path"]).write_bytes(b"{}\n")
    with pytest.raises(ValueError):
        c.verify_seal(tmp_path)


def test_roster_rejects_omission_duplication_reordering_and_wrong_ids():
    ids = c.selected_ids()
    scenes = [{"split": split, "scene_id": sid} for split, values in ids.items() for sid in values]
    c.validate_scene_roster(scenes, ids)
    for changed in (scenes[:-1], [scenes[0]]+scenes[:-1], scenes[::-1]):
        with pytest.raises(ValueError):
            c.validate_scene_roster(changed, ids)
    wrong = dict(ids)
    wrong["iid"] = ids["iid"][:-1]+[0]
    with pytest.raises(ValueError):
        c.validate_scene_roster(scenes, wrong)


def test_replay_budget_records_failure_and_rejects_exhaustion(tmp_path, monkeypatch):
    scene = {"q32": list(range(8)), "candidates": []}
    monkeypatch.setattr(c, "verify_seal", lambda root: ({"scenes": [scene]}, ["mock"]))
    monkeypatch.setattr(c, "checked_json", lambda ref: {"group_factors": [], "fits": ["different"]})
    with pytest.raises(ValueError):
        c.replay(tmp_path)
    assert len(list((tmp_path/"attempts").glob("*.end.json"))) == 1
    assert c.elapsed_budget(tmp_path) >= 0
    monkeypatch.setattr(c, "elapsed_budget", lambda root: 7200.)
    with pytest.raises(TimeoutError):
        c.replay(tmp_path)
    assert len(list((tmp_path/"attempts").glob("*.end.json"))) == 1


def test_stale_runtime_profile_is_rejected(tmp_path, monkeypatch):
    fingerprint = c.runtime_fingerprint()
    binding = {"runtime": fingerprint, "sources": {c.SOURCES[0]: "core"}, "grid": asdict(r.Grid())}
    monkeypatch.setattr(c, "source_binding", lambda: binding)
    bad = {**fingerprint, "numpy": "stale"}
    prof = {"runtime": bad, "core_sha256": "core", "grid": binding["grid"], "device": "cpu"}
    monkeypatch.setattr(c, "checked_json", lambda ref: prof)
    with pytest.raises(ValueError, match="runtime"):
        c.initialize(tmp_path, "cpu", {}, [])
    monkeypatch.setattr(c, "sha", lambda path: "core")
    with pytest.raises(ValueError):
        c.compare_profiles({}, {})
