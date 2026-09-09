"""Full-count mechanical roster fixtures; no prospective observation draws."""
from copy import deepcopy

import numpy as np
import pytest

from experiments.atencion_armonica.test_generative_evidence import fixture
from src.atencion_armonica import generative_evidence as ge
from src.atencion_armonica import generative_evidence_cache as cache
from src.atencion_armonica.generative_evidence_normalization import fit_training_normalizers

BINDING = {"purpose": "repeated-arithmetic-fixture-not-a-campaign", "version": 1}


@pytest.fixture(scope="module")
def roster():
    f = fixture()
    row = ge.observable_rows(*f)
    empty = ge.observable_rows(*f[:4], [], [])
    result = {}
    for cp in ge.CHECKPOINTS:
        cp_row = deepcopy(row)
        cp_row["groups"][:, 3:7] += cp-ge.CHECKPOINTS[0]
        cp_row["globals"][:, 2] -= cp-ge.CHECKPOINTS[0]
        for shard in range(8):
            obs = [{"scene_id": i, "split_seed": cache.SPLITS["train"][1], "log_f": f[0].tolist()}
                   for i in range(shard*512, (shard+1)*512)]
            rows = [empty if o["scene_id"] == 9 else cp_row for o in obs]
            result[cp, shard] = cache.pack_rows("train", cp, obs, rows, binding=BINDING)
    return result


def test_full_train_normalization_alignment_support_and_exact_moments(roster):
    norm = fit_training_normalizers(lambda cp, shard: roster[cp, shard], binding=BINDING)
    assert norm["scene_count"] == 4096 and norm["excluded_scene_ids"] == [9]
    assert len(norm["eligible_scene_ids"]) == 4095 and len(norm["array_digests"]) == 24
    base = ge.observable_rows(*fixture())
    expected = ge.fit_common_normalizer(lambda: iter([base]))
    for j, cp in enumerate(ge.CHECKPOINTS):
        np.testing.assert_allclose(norm["common"][str(cp)]["mean"], np.array(expected["mean"])+[j, j, j, j, -j], atol=1e-12)
        assert norm["common"][str(cp)]["scene_count"] == [4095]*5
    assert norm["evidence"]["scene_count"] == [4095]*6


@pytest.mark.parametrize("kind", ["partial", "calibration", "evidence", "metadata", "changed"])
def test_normalizer_rejects_wrong_roster_cross_checkpoint_and_changed_pass(roster, kind):
    calls = {}
    def load(cp, shard):
        calls[cp, shard] = calls.get((cp, shard), 0)+1
        arrays = roster[cp, shard]
        mutate = (cp == ge.CHECKPOINTS[1] and shard == 0) if kind in ("evidence", "metadata") else (cp == ge.CHECKPOINTS[0] and shard == 0)
        if not mutate or (kind == "changed" and calls[cp, shard] == 1):
            return arrays
        arrays = {k: a.copy() for k, a in arrays.items()}
        if kind == "partial":
            decoded = cache.unpack_rows(arrays, binding=BINDING, split="train", checkpoint_seed=cp)
            return cache.pack_rows("train", cp, decoded["observations"][:-1], decoded["rows"][:-1], binding=BINDING)
        if kind == "calibration":
            m = cache._decode(arrays["metadata"])
            m["split"] = "calibration"
            arrays["metadata"] = cache._metadata(m)
        elif kind == "metadata":
            arrays["globals"][0, 6] += .1
        else:
            arrays["evidence"][0, 0] += .01
        return arrays
    with pytest.raises(ValueError):
        fit_training_normalizers(load, binding=BINDING)
