"""Exact delivered-input fixtures; no campaign normalizers, fitting or CUDA."""
import numpy as np
import pytest

from experiments.atencion_armonica.test_generative_evidence_cache import inputs, BINDING, CP
from src.atencion_armonica import generative_evidence as ge
from src.atencion_armonica import generative_evidence_cache as cache
from src.atencion_armonica.generative_evidence_inputs import pack_inputs, unpack_inputs
from src.atencion_armonica.generative_evidence_storage import write_arrays, read_arrays

RAW_REF = {"fixture": "observable-raw"}
NORM_REF = {"fixture": "mechanical-moments-not-campaign"}
COMMON = {"mean": [0.1]*5, "scale": [2.]*5}
EVIDENCE = {"mean": [0.2]*6, "scale": [3.]*6}


def fixture():
    observations, rows = inputs()
    raw = cache.pack_rows("train", CP, observations, rows, binding=BINDING)
    decoded = cache.unpack_rows(raw, binding=BINDING, split="train", checkpoint_seed=CP)
    expected = {"binding": BINDING, "split": "train", "checkpoint_seed": CP,
                "raw_ref": RAW_REF, "normalizer_ref": NORM_REF, "identities": decoded["identities"]}
    packed = pack_inputs(raw, COMMON, EVIDENCE, **{k: v for k, v in expected.items() if k != "identities"})
    return decoded, expected, packed


def test_persisted_float32_exact_for_three_arms_without_reprocessing(tmp_path):
    decoded, expected, packed = fixture()
    path = tmp_path/"delivered.npz"
    ref = write_arrays(path, packed)
    result = unpack_inputs(read_arrays(path, ref), **expected)
    assert result["scene_ids"] == decoded["scene_ids"]
    for arm in ge.ARMS:
        for i, row in enumerate(decoded["rows"]):
            target = ge.model_inputs(row, COMMON, EVIDENCE, arm, split_seed=cache.SPLITS["train"][1],
                                     scene_id=decoded["scene_ids"][i])
            for name, value in result["inputs"][arm][i].items():
                assert value.dtype == np.float32 and value.tobytes() == target[name].tobytes()
    assert result["inputs"]["local"][-1]["evidence"].shape == (0, 6)
    assert not {"targets", "metrics", "labels"} & set(packed)


def test_unavailable_branch_cannot_be_reintroduced_in_both_channels():
    decoded, expected, packed = fixture()
    index = next(i for i, r in enumerate(decoded["rows"]) if r["partitions"] and len(r["partitions"][0]) == 4)
    lo, hi = packed["candidate_offsets"][index:index+2]
    for arm in ("generative", "decoupled"):
        packed[arm][lo:hi, 2:6] = 1.
    with pytest.raises(ValueError, match="unavailable"):
        unpack_inputs(packed, **expected)


@pytest.mark.parametrize("kind", ["sham", "identity", "normalizer", "source", "incidence", "precision", "offset", "target", "partition"])
def test_corrupted_or_misbound_delivered_inputs_rejected(kind):
    _, expected, packed = fixture()
    if kind == "sham":
        packed["decoupled"][0, 0] += 1
    elif kind == "identity":
        expected["identities"] = expected["identities"][::-1]
    elif kind == "normalizer":
        expected["normalizer_ref"] = {"fixture": "another-normalizer"}
    elif kind == "source":
        expected["raw_ref"] = {"fixture": "another-raw"}
    elif kind == "incidence":
        packed["incidence"][0] = .01
    elif kind == "precision":
        packed["groups"] = packed["groups"].astype(np.float64)
    elif kind == "offset":
        packed["candidate_offsets"][1] += 1
    elif kind == "target":
        packed["targets"] = np.zeros((1, 2), np.float32)
    else:
        meta = cache._decode(packed["metadata"])
        meta["scenes"][0]["partitions"].reverse()
        packed["metadata"] = cache._metadata(meta)
    with pytest.raises(ValueError):
        unpack_inputs(packed, **expected)
