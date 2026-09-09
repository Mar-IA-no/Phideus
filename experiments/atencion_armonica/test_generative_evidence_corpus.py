"""Complete-count arithmetic OPEN fixtures; no draw, fitter, CUDA or training."""
from copy import deepcopy

import numpy as np
import pytest

from experiments.atencion_armonica.test_generative_evidence import fixture
from experiments.atencion_armonica.test_generative_evidence_normalization import roster, BINDING
from src.atencion_armonica import generative_evidence as ge
from src.atencion_armonica import generative_evidence_cache as cache
from src.atencion_armonica import generative_evidence_storage as storage
from src.atencion_armonica.generative_evidence_corpus import TrainingCorpus
from src.atencion_armonica.generative_evidence_normalization import fit_training_normalizers
from src.atencion_armonica.generative_evidence_prepared import PreparedStore
from src.atencion_armonica.generative_evidence_reuse import OPEN_SPLITS
from src.atencion_armonica.generative_evidence_supervision import candidate_supervision


@pytest.fixture(scope="module")
def prepared(tmp_path_factory, roster):
    store = PreparedStore(tmp_path_factory.mktemp("prepared"), binding=BINDING)
    f = fixture()
    row, empty = ge.observable_rows(*f), ge.observable_rows(*f[:4], [], [])
    splits = {}
    for split, (count, seed) in OPEN_SPLITS.items():
        shard_refs = []
        for shard in range(count//512):
            raw_refs, decoded = {}, None
            for cp in ge.CHECKPOINTS:
                if split == "train":
                    arrays = roster[cp, shard]
                else:
                    obs = [{"scene_id": i, "split_seed": seed, "log_f": f[0].tolist()} for i in range(512)]
                    rows = [empty if i == 9 else row for i in range(512)]
                    arrays = cache.pack_rows(split, cp, obs, rows, binding=BINDING)
                path = store.path(f"fixture/{split}/{shard}/raw_{cp}.npz")
                path.parent.mkdir(parents=True, exist_ok=True)
                storage.write_arrays(path, arrays)
                raw_refs[str(cp)] = store.reference(path)
                decoded = cache.unpack_rows(arrays, binding=BINDING, split=split, checkpoint_seed=cp)
            supervision = [candidate_supervision(r["partitions"], np.repeat(np.arange(2), 4)) for r in decoded["rows"]]
            path = store.path(f"fixture/{split}/{shard}/targets.npz")
            storage.write_arrays(path, cache.pack_targets(decoded["identities"], supervision, binding=BINDING))
            target_ref = store.reference(path)
            # Dummy scene/factor refs are deliberate: the adapter trusts the
            # caller-pinned completion and only reads aggregate payloads. This
            # fixture is NOT a valid output of the real preparation supervisor.
            index = {"schema": "generative-evidence-prepared-shard-v1", "binding": BINDING,
                "split": split, "shard": shard, "scene_ids": decoded["scene_ids"],
                "records": [{k: {} for k in ("fit", "observable", "supervision")} for _ in range(512)],
                "raw": raw_refs, "targets": target_ref, "metrics": {}}
            path = store.path(f"fixture/{split}/{shard}/index.json")
            storage.write_json(path, index)
            shard_refs.append(store.reference(path))
        path = store.path(f"fixture/{split}/index.json")
        storage.write_json(path, {"schema": "generative-evidence-prepared-split-v1", "binding": BINDING,
            "split": split, "split_seed": seed, "count": count, "shards": shard_refs})
        splits[split] = store.reference(path)
    norms = fit_training_normalizers(lambda cp, shard: roster[cp, shard], binding=BINDING)
    path = store.path("fixture/normalizers.json")
    storage.write_json(path, {"prepared_train": splits["train"], "normalizers": norms})
    complete = {"schema": "generative-evidence-open-preparation-v1", "binding": BINDING,
        "status": "OPEN_PREPARED_NOT_TRAINED_NO_TEST_ACCESS", "splits": splits,
        "normalizers": store.reference(path), "reuse": {"fixture": True}, "profile": {"fixture": True}}
    path = store.path("fixture/open_prepared.json")
    storage.write_json(path, complete)
    corpus = TrainingCorpus(store, store.reference(path))
    delivered = corpus.materialize(check=lambda: None, progress=lambda _: None)
    return store, corpus, delivered


def test_full_materialization_and_cell_data_exact_three_arms(prepared):
    store, corpus, delivered = prepared
    assert len(store.json(delivered)["entries"]) == 27
    first = None
    for cp in ge.CHECKPOINTS:
        for arm in ge.ARMS:
            data = corpus.load_cell(delivered, arm=arm, checkpoint_seed=cp, check=lambda: None)
            assert len(data.rows["train"]) == 4096 and len(data.rows["calibration"]) == 512
            assert len(data.eligible["train"]) == 4095 and len(data.eligible["calibration"]) == 511
            assert data.rows["train"][9]["targets"].shape == (0, 2)
            assert data.binding["arm"] == arm and data.binding["checkpoint_seed"] == cp
            _, _, raw, decoded = corpus.raw("train", cp, 0)
            for i in (0, 9, 511):
                expected = ge.model_inputs(decoded["rows"][i], corpus.norm["common"][str(cp)], corpus.norm["evidence"],
                                           arm, split_seed=OPEN_SPLITS["train"][1], scene_id=i)
                for key in expected:
                    np.testing.assert_array_equal(data.rows["train"][i]["inputs"][key], expected[key])
            if first is None:
                first = data.rows["train"][0]["targets"].copy()
            np.testing.assert_array_equal(data.rows["train"][0]["targets"], first)


def test_materialization_recovery_has_same_receipt_without_new_files(prepared):
    store, corpus, delivered = prepared
    before = sorted(p.relative_to(store.root).as_posix() for p in store.root.rglob("*") if p.is_file())
    assert corpus.materialize(check=lambda: None, progress=lambda _: None) == delivered
    after = sorted(p.relative_to(store.root).as_posix() for p in store.root.rglob("*") if p.is_file())
    assert before == after


@pytest.mark.parametrize("kind", ["status", "missing_calibration", "wrong_train_parent", "norm_role", "norm_digest", "norm_roster"])
def test_incomplete_or_wrong_normalizer_provenance_rejected(prepared, kind):
    store, corpus, _ = prepared
    complete = deepcopy(store.json(corpus.prepared_ref))
    if kind == "status":
        complete["status"] = "INCOMPLETE"
    elif kind == "missing_calibration":
        del complete["splits"]["calibration"]
    else:
        norm = deepcopy(store.json(corpus.normalizer_ref))
        if kind == "wrong_train_parent":
            norm["prepared_train"] = complete["splits"]["calibration"]
        elif kind == "norm_role":
            norm["normalizers"]["split"] = "calibration"
        elif kind == "norm_digest":
            norm["normalizers"]["array_digests"][0]["sha256"] = "0"*64
        else:
            norm["normalizers"]["eligible_scene_ids"] = norm["normalizers"]["eligible_scene_ids"][:-1]
        path = store.path(f"negative/{kind}/normalizers.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        storage.write_json(path, norm)
        complete["normalizers"] = store.reference(path)
    path = store.path(f"negative/{kind}/complete.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    storage.write_json(path, complete)
    with pytest.raises(ValueError):
        changed = TrainingCorpus(store, store.reference(path))
        changed.raw("train", ge.CHECKPOINTS[0], 0)


def test_fresh_access_and_partial_delivered_roster_rejected(prepared):
    store, corpus, delivered = prepared
    with pytest.raises(PermissionError):
        corpus.raw("iid", ge.CHECKPOINTS[0], 0)
    value = deepcopy(store.json(delivered))
    value["entries"] = value["entries"][:-1]
    path = store.path("negative/partial-delivered.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    storage.write_json(path, value)
    with pytest.raises(ValueError, match="27 ordered shards"):
        corpus.load_cell(store.reference(path), arm="local", checkpoint_seed=ge.CHECKPOINTS[0], check=lambda: None)


def test_stop_callback_prevents_materialization_or_loading(prepared):
    _, corpus, delivered = prepared
    def stop():
        raise InterruptedError("fixture stop")
    with pytest.raises(InterruptedError):
        corpus.materialize(check=stop)
    with pytest.raises(InterruptedError):
        corpus.load_cell(delivered, arm="local", checkpoint_seed=ge.CHECKPOINTS[0], check=stop)
