"""Preparation recovery and complete shard fixtures, with all fitting forbidden."""
from copy import deepcopy
from itertools import combinations
import json

import numpy as np
import pytest

from src.atencion_armonica import generative_evidence as ge
from src.atencion_armonica.generative_evidence_prepared import PreparedStore
from src.atencion_armonica.generative_evidence_reuse import OpenReuse, ROOT
from src.atencion_armonica.generative_evidence_storage import read_scene
from src.atencion_armonica.generative_evidence_supervision import candidate_supervision, open_truths

BINDING = {"purpose": "PREPARATION_MECHANICS_NOT_CAMPAIGN", "version": 1}
PROFILE = ROOT/".agent-work/phideus-generative-evidence-20260909/profile-gpu-01"


@pytest.fixture(scope="module")
def real():
    import hashlib
    raw = (PROFILE/"report.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == "c3144164274e154f2dee4ca93eea81410d68558da623545006d751ee473051ea"
    report = json.loads(raw)
    shard = OpenReuse().shard("train", 0)
    truths = open_truths(shard)
    return shard, truths, report


def preserved(real, scene_id):
    shard, truths, report = real
    value = read_scene(PROFILE/f"scene_{scene_id:05d}.json.gz", report["fit_scenes"][scene_id]["scene"])
    return shard.scene(scene_id), {k: value[k] for k in ("fits", "group_factors")}, truths[scene_id]


@pytest.fixture(autouse=True)
def no_fitting(monkeypatch):
    def forbidden(*a, **kw):
        raise AssertionError("preparation codec must not launch a fitter")
    monkeypatch.setattr(ge.law, "fit_candidates", forbidden)
    monkeypatch.setattr(ge.law.GroupFitter, "fit", forbidden)


@pytest.mark.parametrize("scene_id", [0, 26])
def test_persisted_scene_recovery_keeps_fit_raw_and_supervision_separate(tmp_path, real, scene_id):
    scene, fitted, truth = preserved(real, scene_id)
    store = PreparedStore(tmp_path/"store", binding=BINDING)
    fit_ref = store.save_fit("train", scene_id, scene, fitted, origin={"profile": "gpu-01", "scene_id": scene_id})
    store = PreparedStore(tmp_path/"store", binding=BINDING)
    raw_ref = store.save_observables("train", scene_id, scene)
    loaded = [store.load_row("train", scene_id, cp) for cp in ge.CHECKPOINTS]
    assert len({r["identities"][0] for r in loaded}) == 1
    supervised = candidate_supervision(scene["partitions"], truth["labels"])
    target_ref = store.save_supervision("train", scene_id, supervised, identity=loaded[0]["identities"][0], source={"fixture": "open-sidecar"})
    restored = store.load_supervision("train", scene_id)
    for k in ("raw", "targets"):
        assert restored[k].tobytes() == supervised[k].tobytes()
    assert restored["metrics"] == supervised["metrics"]
    assert store.save_fit("train", scene_id, scene, fitted, origin={"profile": "gpu-01", "scene_id": scene_id}) == fit_ref
    assert store.save_observables("train", scene_id, scene) == raw_ref
    assert store.save_supervision("train", scene_id, supervised, identity=loaded[0]["identities"][0], source={"fixture": "open-sidecar"}) == target_ref
    assert "source" not in store.json(raw_ref) and "targets" not in store.json(raw_ref)
    if scene_id == 26:
        assert restored["targets"].shape == (0, 2) and store.json(raw_ref)["status"] == "NO_OBSERVABLE_CANDIDATE"


def test_interruption_after_raw_files_preserves_fit_and_never_overwrites_orphans(tmp_path, real, monkeypatch):
    scene, fitted, _ = preserved(real, 0)
    store = PreparedStore(tmp_path/"store", binding=BINDING)
    fit_ref = store.save_fit("train", 0, scene, fitted, origin={"fixture": "open-profile"})
    publish = store._publish
    def interrupt(split, scene_id, kind, value):
        if kind == "observable":
            raise InterruptedError("mechanical interruption before receipt")
        return publish(split, scene_id, kind, value)
    monkeypatch.setattr(store, "_publish", interrupt)
    with pytest.raises(InterruptedError):
        store.save_observables("train", 0, scene)
    orphans = {p: p.read_bytes() for p in (store.root/"train/00000").glob("raw-*.npz")}
    assert len(orphans) == 3
    recovered = PreparedStore(store.root, binding=BINDING)
    recovered.save_observables("train", 0, scene)
    assert recovered.load_fit("train", 0)[1] == fit_ref
    assert all(p.read_bytes() == raw for p, raw in orphans.items())
    assert len(list((store.root/"train/00000").glob("raw-*.npz"))) == 6


def test_changed_input_wrong_identity_and_fresh_test_are_rejected(tmp_path, real):
    scene, fitted, truth = preserved(real, 0)
    store = PreparedStore(tmp_path/"store", binding=BINDING)
    store.save_fit("train", 0, scene, fitted, origin={"fixture": "open-profile"})
    store.save_observables("train", 0, scene)
    changed = deepcopy(scene)
    changed["logits"][ge.CHECKPOINTS[0]] += np.float32(.01)
    with pytest.raises(ValueError, match="fit or input"):
        store.save_observables("train", 0, changed)
    with pytest.raises(ValueError, match="identity"):
        store.save_supervision("train", 0, candidate_supervision(scene["partitions"], truth["labels"]), identity="0"*64, source={"fixture": "wrong"})
    with pytest.raises(PermissionError, match="fresh test"):
        store.save_fit("iid", 0, scene, fitted, origin={"fixture": "wrong-port"})
    with pytest.raises(ValueError, match="binding"):
        PreparedStore(store.root, binding={"purpose": "another-campaign"})


def empty_scene(scene_id):
    q = np.arange(8, dtype=np.float32)/8
    return {"observation": {"scene_id": scene_id, "split_seed": 2026090881, "log_f": q.tolist()},
            "features": {"triples": np.array(list(combinations(range(8), 3)), np.int64), "residual_cents": np.zeros(56)},
            "logits": {cp: np.zeros((8, 8), np.float32) for cp in ge.CHECKPOINTS},
            "pools": {str(cp): [] for cp in ge.CHECKPOINTS}, "inventory": {"candidates": []}, "partitions": [],
            "canonical_to_observed": np.arange(8), "q32": q, "status": "NO_OBSERVABLE_CANDIDATE"}


def test_complete_512_scene_shard_and_split_no_padding_or_partial_seal(tmp_path):
    store = PreparedStore(tmp_path/"store", binding=BINDING)
    with pytest.raises(FileNotFoundError):
        store.seal_shard("calibration", 0)
    for i in range(512):
        scene = empty_scene(i)
        store.save_fit("calibration", i, scene, {"fits": [], "group_factors": []}, origin={"fixture": "empty-arithmetic"})
        store.save_observables("calibration", i, scene)
        decoded = store.load_row("calibration", i, ge.CHECKPOINTS[0])
        store.save_supervision("calibration", i, {"raw": np.empty((0, 2), np.float64), "targets": np.empty((0, 2), np.float32), "metrics": []},
                               identity=decoded["identities"][0], source={"fixture": "empty-arithmetic"})
    ref = store.seal_shard("calibration", 0)
    assert store.seal_shard("calibration", 0) == ref
    # Corruption/removal after shard seal must not acquire a split seal.
    index = store.json(ref)
    raw_path = store.path(index["raw"][str(ge.CHECKPOINTS[0])]["path"])
    original = raw_path.read_bytes()
    raw_path.write_bytes(original[:-1])
    with pytest.raises(ValueError, match="archive identity"):
        store.seal_split("calibration")
    assert not (store.root/"calibration/index.json").exists()
    raw_path.write_bytes(original)
    moved = raw_path.with_suffix(".temporarily-absent")
    raw_path.rename(moved)
    with pytest.raises(FileNotFoundError):
        store.seal_split("calibration")
    assert not (store.root/"calibration/index.json").exists()
    moved.rename(raw_path)
    split = store.seal_split("calibration")
    assert store.json(split)["shards"] == [ref] and store.json(split)["count"] == 512
    result = store.load_shard("calibration", ge.CHECKPOINTS[2], 0)
    assert result["scene_ids"] == list(range(512)) and len(result["identities"]) == 512
    assert all(r["targets"].shape == (0, 2) for r in result["targets"])
    assert store.raw_shard("calibration", ge.CHECKPOINTS[0], 0)["groups"].shape == (0, 9)
    with pytest.raises(ValueError):
        store.seal_shard("calibration", 1)


def test_missing_factor_assignment_is_not_a_complete_fit(tmp_path, real):
    scene, fitted, _ = preserved(real, 0)
    store = PreparedStore(tmp_path/"store", binding=BINDING)
    fitted["group_factors"][0]["factor"]["fine"].pop()
    with pytest.raises(ValueError, match="assignments"):
        store.save_fit("train", 0, scene, fitted, origin={"fixture": "truncated"})
