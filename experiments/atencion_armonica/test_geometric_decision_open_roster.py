"""Arithmetic OPEN-adapter fixtures only: no sampler, corpus data, sidecars or GPU."""
import copy
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from src.atencion_armonica import generative_evidence as ge
from src.atencion_armonica import geometric_decision_open as adapter
from src.atencion_armonica.generative_evidence_reuse import OPEN_SPLITS
from src.atencion_armonica.partial_compatibility_cache import encoded


def _write(root, name, raw):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return {"path": name, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def _json(root, name, value):
    return _write(root, name, encoded(value))


def _metadata_fixture(root):
    """Minimal real TrainingCorpus metadata; payload shards deliberately do not exist."""
    binding_value = {"fixture": "arithmetic-open-roster-v1"}
    binding_ref = _json(root, "binding.json", binding_value)
    split_refs = {}
    for split, (count, seed) in OPEN_SPLITS.items():
        shard_refs = [
            {"path": f"unused/{split}/shard_{shard}.json", "sha256": "0" * 64, "bytes": 0}
            for shard in range(count // 512)
        ]
        split_refs[split] = _json(root, f"prepared/{split}.json", {
            "schema": "generative-evidence-prepared-split-v1",
            "binding": binding_value,
            "split": split,
            "split_seed": seed,
            "count": count,
            "shards": shard_refs,
        })

    identities = [f"arithmetic-scene-{i:04d}" for i in range(4096)]
    normalizer = {
        "schema": "generative-evidence-train-normalizers-v1",
        "binding": binding_value,
        "split": "train",
        "split_seed": OPEN_SPLITS["train"][1],
        "scene_count": 4096,
        "eligible_scene_ids": list(range(4096)),
        "excluded_scene_ids": [],
        "scene_identities": identities,
        "common": {
            str(cp): {"mean": [0.0] * 5, "scale": [1.0] * 5,
                      "zero_variance": [False] * 5, "scene_count": [4096] * 5}
            for cp in ge.CHECKPOINTS
        },
        "evidence": {"mean": [0.0] * 6, "scale": [1.0] * 6,
                     "zero_variance": [False] * 6, "scene_count": [4096] * 6},
        "array_digests": [
            {"checkpoint_seed": cp, "shard": shard,
             "sha256": hashlib.sha256(f"{cp}:{shard}".encode()).hexdigest()}
            for cp in ge.CHECKPOINTS for shard in range(8)
        ],
    }
    normalizer_ref = _json(root, "normalizers.json", {
        "prepared_train": split_refs["train"], "normalizers": normalizer,
    })
    prepared_ref = _json(root, "open_prepared.json", {
        "schema": "generative-evidence-open-preparation-v1",
        "binding": binding_value,
        "status": "OPEN_PREPARED_NOT_TRAINED_NO_TEST_ACCESS",
        "splits": split_refs,
        "normalizers": normalizer_ref,
        "reuse": {"fixture": True},
        "profile": {"fixture": True},
    })
    roster = [
        {"split": split, "checkpoint_seed": cp, "shard": shard,
         "index": {"path": f"unused/delivered/{split}/{shard}/{cp}.json",
                   "sha256": hashlib.sha256(f"{split}:{shard}:{cp}".encode()).hexdigest(), "bytes": 0}}
        for split in OPEN_SPLITS
        for shard in range(OPEN_SPLITS[split][0] // 512)
        for cp in ge.CHECKPOINTS
    ]
    completion = {
        "schema": "generative-evidence-delivered-open-v1",
        "binding": binding_value,
        "prepared": prepared_ref,
        "normalizers": normalizer_ref,
        "entries": roster,
    }
    return binding_ref, prepared_ref, completion


def test_complete_arithmetic_roster_is_4096_512_by_three_checkpoints(tmp_path):
    binding_ref, prepared_ref, completion = _metadata_fixture(tmp_path)
    delivered_ref = _json(tmp_path, "delivered.json", completion)
    source = adapter.OpenSource(tmp_path, binding_ref=binding_ref, prepared_ref=prepared_ref,
                                delivered_ref=delivered_ref)
    expected = [
        (split, cp, shard)
        for split in OPEN_SPLITS
        for shard in range(OPEN_SPLITS[split][0] // 512)
        for cp in ge.CHECKPOINTS
    ]
    assert len(expected) == 27
    assert list(source.entries) == expected
    assert sum(split == "train" for split, _, _ in expected) == 8 * 3
    assert sum(split == "calibration" for split, _, _ in expected) == 1 * 3


@pytest.mark.parametrize("mutation", ["missing", "reordered", "duplicate", "extra_field"])
def test_complete_roster_rejects_any_arithmetic_mutation(tmp_path, mutation):
    binding_ref, prepared_ref, completion = _metadata_fixture(tmp_path)
    entries = [dict(entry) for entry in completion["entries"]]
    if mutation == "missing":
        entries.pop()
    elif mutation == "reordered":
        entries[0], entries[1] = entries[1], entries[0]
    elif mutation == "duplicate":
        entries[-1] = dict(entries[-2])
    else:
        entries[0]["unexpected"] = True
    delivered_ref = _json(tmp_path, f"delivered_{mutation}.json", {**completion, "entries": entries})
    with pytest.raises(ValueError, match="all 27 ordered entries"):
        adapter.OpenSource(tmp_path, binding_ref=binding_ref, prepared_ref=prepared_ref,
                           delivered_ref=delivered_ref)


def _cost_row(costs):
    costs = np.asarray(costs, dtype=np.float64)
    evidence = np.zeros((len(costs), 6), dtype=np.float64)
    evidence[:, 1] = costs
    available = np.zeros((len(costs), 3), dtype=np.bool_)
    available[:, 0] = True
    return {"evidence": evidence, "available": available}


def test_full_train_scale_counts_each_scene_once_not_each_candidate():
    identities = [f"identity-{i:04d}" for i in range(4096)]
    eligible = [i for i in range(4096) if i % 4]
    excluded = [i for i in range(4096) if not i % 4]
    source = adapter.OpenSource.__new__(adapter.OpenSource)
    source.delivered_ref = {"fixture": "delivered"}
    source.corpus = SimpleNamespace(
        prepared_ref={"fixture": "prepared"},
        normalizer_ref={"fixture": "normalizer"},
        norm={"scene_identities": identities, "eligible_scene_ids": eligible,
              "excluded_scene_ids": excluded},
    )
    checked = []

    def observable_shard(split, shard, *, check):
        assert split == "train" and 0 <= shard < 8
        check()
        checked.append(shard)
        ids = list(range(shard * 512, (shard + 1) * 512))
        rows = []
        for scene_id in ids:
            # Per four scenes: empty; [1]; [1,3]; [2,2,2].
            costs = ((), (1.0,), (1.0, 3.0), (2.0, 2.0, 2.0))[scene_id % 4]
            rows.append(_cost_row(costs))
        return {"identities": identities[ids[0]:ids[-1] + 1],
                "raw": {cp: rows for cp in ge.CHECKPOINTS}}

    source.observable_shard = observable_shard
    scale = source.geometric_scale(check=lambda: None)
    # Mean within scene first: (1 + mean(1,9) + mean(4,4,4)) / 3 = 10/3.
    assert scale["scale"] == pytest.approx(np.sqrt(10.0 / 3.0), rel=0, abs=1e-15)
    assert scale["eligible_scenes"] == 3072
    assert scale["empty_scenes"] == 1024
    assert checked == list(range(8))


def test_observable_target_port_is_separate_and_interface_has_eight_inputs_and_donors(monkeypatch):
    scene_ids = list(range(512))
    identities = [f"scene-{i}" for i in scene_ids]
    observations = [{"scene_id": i} for i in scene_ids]
    partitions = [((0, 1, 2, 3), (4, 5, 6, 7)),
                  ((0, 1, 2, 4), (3, 5, 6, 7))]
    raw_row = {
        "n": 8,
        "partitions": partitions,
        "evidence": np.array([[0., 2., 0., 4., 0., 6.],
                              [0., 8., 0., 10., 0., 12.]], dtype=np.float64),
        "available": np.ones((2, 3), dtype=np.bool_),
        "incidence": np.eye(2, dtype=np.float32),
        "canonical_to_observed": np.arange(8, dtype=np.int64),
    }
    delivered = {
        "groups": np.arange(18, dtype=np.float32).reshape(2, 9),
        "globals": np.arange(34, dtype=np.float32).reshape(2, 17),
        "incidence": np.eye(2, dtype=np.float32),
        "evidence": np.arange(12, dtype=np.float32).reshape(2, 6),
    }
    decoupled_evidence, inherited_sham = ge.decouple(
        partitions, delivered["evidence"],
        split_seed=OPEN_SPLITS["calibration"][1], scene_id=0,
    )
    decoded = {"scene_ids": scene_ids, "identities": identities, "observations": observations,
               "rows": [raw_row] * 512}
    target_ref = {"path": "targets.npz", "sha256": "1" * 64, "bytes": 1}
    index = {"targets": target_ref,
             "raw": {str(cp): {"fixture": f"raw-{cp}"} for cp in ge.CHECKPOINTS}}

    class Corpus:
        prepared_ref = {"fixture": "prepared"}
        normalizer_ref = {"fixture": "normalizer"}
        norm = {"scene_identities": [f"train-{i}" for i in range(4096)]}
        splits = {"calibration": {"shards": ["prepared-calibration"]}}

        def raw(self, split, cp, shard):
            assert (split, shard) == ("calibration", 0)
            return index, "prepared-calibration", None, decoded

        @staticmethod
        def _identity(split, cp, shard, index_value, shard_ref):
            return {"split": split, "checkpoint_seed": cp, "shard": shard}

        @staticmethod
        def _decode(entry, arrays, decoded_value, identity):
            decoupled = {**delivered, "evidence": decoupled_evidence}
            return {"inputs": {"generative": [delivered] * 512,
                               "decoupled": [decoupled] * 512},
                    "sham": [inherited_sham] * 512}

    class Store:
        binding = {"fixture": "binding"}

        @staticmethod
        def json(ref):
            if ref == "prepared-calibration":
                return index
            return {"inputs": f"arrays-{ref}"}

        @staticmethod
        def arrays(ref):
            return ref

    source = adapter.OpenSource.__new__(adapter.OpenSource)
    source.corpus, source.store = Corpus(), Store()
    source.entries = {("calibration", cp, 0): f"entry-{cp}" for cp in ge.CHECKPOINTS}
    source.delivered_ref = {"fixture": "delivered"}
    target_calls = []

    def unpack_targets(arrays, decoded_value, *, binding):
        target_calls.append((arrays, decoded_value, binding))
        return ["explicit-supervision"] * 512

    monkeypatch.setattr(adapter.cache, "unpack_targets", unpack_targets)
    observable = source.observable_shard("calibration", 0, check=lambda: None)
    assert target_calls == []
    assert observable["targets_ref"] == target_ref
    assert source.supervision_shard(observable) == ["explicit-supervision"] * 512
    assert len(target_calls) == 1

    scale = {"schema": "geometric-decision-open-scale-v1",
             "prepared": source.corpus.prepared_ref, "delivered": source.delivered_ref,
             "normalizers": source.corpus.normalizer_ref,
             "scene_identities": source.corpus.norm["scene_identities"], "scale": 2.0}
    interface = source.interface_shard(observable, scale)
    for cp in ge.CHECKPOINTS:
        assert len(interface[cp]) == 512
        first = interface[cp][0]
        assert first["inputs"]["evidence"].shape == (2, 8)
        np.testing.assert_array_equal(first["inputs"]["evidence"][:, :6], delivered["evidence"])
        np.testing.assert_array_equal(first["inputs"]["evidence"][:, 6:], [[1., 4.], [4., 1.]])
        assert first["sham"]["six_channel_sham"]["donors"] == [1, 0]
        np.testing.assert_array_equal(first["diagnostics"]["decoupled_six"],
                                      delivered["evidence"][[1, 0]])

    cp = ge.CHECKPOINTS[0]
    preserved = observable["decoupled_evidence"][cp][0]
    observable["decoupled_evidence"][cp][0] = preserved.copy()
    observable["decoupled_evidence"][cp][0][0, 0] += np.float32(1)
    with pytest.raises(ValueError, match="preserved delivered diagnostics"):
        source.interface_shard(observable, scale)
    observable["decoupled_evidence"][cp][0] = preserved

    inherited = observable["raw_sham"][cp][0]
    observable["raw_sham"][cp][0] = copy.deepcopy(inherited)
    observable["raw_sham"][cp][0]["donors"] = [0, 1]
    with pytest.raises(ValueError, match="preserved delivered diagnostics"):
        source.interface_shard(observable, scale)
    observable["raw_sham"][cp][0] = inherited
