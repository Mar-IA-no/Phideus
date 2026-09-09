"""Mechanical ragged identity/precision fixtures, not a campaign producer."""
from copy import deepcopy

import numpy as np
import pytest

from experiments.atencion_armonica.test_generative_evidence import fixture
from src.atencion_armonica import generative_evidence as ge
from src.atencion_armonica import generative_evidence_cache as cache
from src.atencion_armonica.generative_evidence_storage import write_arrays, read_arrays
from src.atencion_armonica.generative_evidence_supervision import candidate_supervision

BINDING = {"purpose": "mechanical-codec-fixture", "version": 1}
CP = ge.CHECKPOINTS[0]


def inputs():
    a, b = fixture(), fixture(16)
    rows = [ge.observable_rows(*a), ge.observable_rows(*b), ge.observable_rows(*a[:4], [], [])]
    observations = [{"scene_id": i, "split_seed": cache.SPLITS["train"][1], "log_f": f[0].tolist()}
                    for i, f in zip((0, 2, 3), (a, b, a))]
    return observations, rows


def test_ragged_cache_exact_archive_roundtrip_including_empty(tmp_path):
    observations, rows = inputs()
    arrays = cache.pack_rows("train", CP, observations, rows, binding=BINDING)
    path = tmp_path/"raw.npz"
    ref = write_arrays(path, arrays)
    saved = read_arrays(path, ref)
    decoded = cache.unpack_rows(saved, binding=BINDING, split="train", checkpoint_seed=CP)
    assert decoded["observations"] == observations and decoded["scene_ids"] == [0, 2, 3]
    assert arrays["incidence"].size == sum(r["incidence"].size for r in rows)
    assert arrays["groups"].shape[0] == sum(len(r["groups"]) for r in rows)
    for original, restored in zip(rows, decoded["rows"]):
        for key in (*cache.FIELDS, "incidence", "canonical_to_observed"):
            assert original[key].dtype == restored[key].dtype
            assert original[key].shape == restored[key].shape
            assert original[key].tobytes() == restored[key].tobytes()
        assert ge.law.signature(original["group_ids"]) == ge.law.signature(restored["group_ids"])


def test_targets_are_separate_and_bound_to_candidate_order(tmp_path):
    obs, rows = inputs()
    raw = cache.pack_rows("train", CP, obs, rows, binding=BINDING)
    decoded = cache.unpack_rows(raw, binding=BINDING, split="train", checkpoint_seed=CP)
    supervised = [candidate_supervision(r["partitions"], np.repeat(np.arange(r["n"]//4), 4)) for r in rows]
    arrays = cache.pack_targets(decoded["identities"], supervised, binding=BINDING)
    path = tmp_path/"targets.npz"
    ref = write_arrays(path, arrays)
    result = cache.unpack_targets(read_arrays(path, ref), decoded, binding=BINDING)
    for old, new in zip(supervised, result):
        for k in ("raw", "targets"):
            assert old[k].tobytes() == new[k].tobytes()
    bad = deepcopy(decoded)
    bad["identities"] = bad["identities"][::-1]
    with pytest.raises(ValueError, match="identity/order"):
        cache.unpack_targets(arrays, bad, binding=BINDING)
    contaminated = {**raw, "targets": arrays["targets"]}
    with pytest.raises(ValueError, match="contains supervision"):
        cache.unpack_rows(contaminated, binding=BINDING, split="train", checkpoint_seed=CP)


@pytest.mark.parametrize("corruption", ["binding", "role", "checkpoint", "offset", "tail", "identity", "signature", "incidence", "branch"])
def test_corrupt_raw_cache_rejected(corruption):
    obs, rows = inputs()
    arrays = cache.pack_rows("train", CP, obs, rows, binding=BINDING)
    binding, split, cp = BINDING, "train", CP
    if corruption == "binding":
        binding = {**BINDING, "version": 2}
    elif corruption == "role":
        split = "calibration"
    elif corruption == "checkpoint":
        cp = ge.CHECKPOINTS[1]
    elif corruption == "offset":
        arrays["candidate_offsets"][1] += 1
    elif corruption == "tail":
        arrays["groups"] = np.concatenate([arrays["groups"], arrays["groups"][:1]])
    elif corruption in ("identity", "signature"):
        meta = cache._decode(arrays["metadata"])
        if corruption == "identity":
            meta["scenes"][0]["observation"]["scene_id"] = 1
        else:
            meta["scenes"][0]["partitions"].reverse()
        arrays["metadata"] = cache._metadata(meta)
    elif corruption == "incidence":
        arrays["incidence"][0] = .25
    else:
        arrays["available"][0, 0] = False
    with pytest.raises(ValueError):
        cache.unpack_rows(arrays, binding=binding, split=split, checkpoint_seed=cp)


def test_targets_cannot_disagree_with_raw_entropies():
    obs, rows = inputs()
    raw = cache.pack_rows("train", CP, obs, rows, binding=BINDING)
    decoded = cache.unpack_rows(raw, binding=BINDING, split="train", checkpoint_seed=CP)
    supervised = [candidate_supervision(r["partitions"], np.repeat(np.arange(r["n"]//4), 4)) for r in rows]
    arrays = cache.pack_targets(decoded["identities"], supervised, binding=BINDING)
    arrays["targets"][0, 0] = .5
    with pytest.raises(ValueError, match="normalized entropies"):
        cache.unpack_targets(arrays, decoded, binding=BINDING)
