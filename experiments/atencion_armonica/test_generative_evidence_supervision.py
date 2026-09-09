"""OPEN historical sidecars only; reconstruction is not a second generator call."""
from copy import deepcopy
import json

import numpy as np
import pytest

from src.atencion_armonica import generative_evidence_supervision as supervision
from src.atencion_armonica.generative_evidence_reuse import OpenReuse
from src.atencion_armonica.learned_partition_core import partition_errors


@pytest.fixture(scope="module")
def open_shard():
    return OpenReuse().shard("train", 0)


def test_reconstruction_never_redraws_and_targets_cover_new_roster(open_shard, monkeypatch):
    from src.atencion_armonica import learned_partition_data, structured_source_data, shared_partial_data
    def prohibited(*args, **kwargs):
        raise AssertionError("no generator call permitted in supervisor")
    for module, name in ((learned_partition_data, "_draw_scene"), (structured_source_data, "_draw_scene"),
                         (shared_partial_data, "generate_scene"), (shared_partial_data, "_observe")):
        monkeypatch.setattr(module, name, prohibited)
    reconstructed = supervision.open_truths(open_shard)
    assert len(reconstructed) == 512
    scene = open_shard.scene(0)
    y = reconstructed[0]["labels"]
    result = supervision.candidate_supervision(scene["partitions"], y)
    assert result["targets"].shape == (48, 2)
    assert result["raw"].dtype == np.float64 and result["targets"].dtype == np.float32
    for p, raw, target, metrics in zip(scene["partitions"], result["raw"], result["targets"], result["metrics"]):
        e = partition_errors(p, y)
        np.testing.assert_array_equal(raw, e["raw"])
        np.testing.assert_array_equal(target, e["normalized"].astype(np.float32))
        assert len(metrics) == 13 and metrics["sub3_member_fraction"] == 0
    empty = supervision.candidate_supervision([], y)
    assert empty["raw"].shape == (0, 2) and empty["targets"].shape == (0, 2) and empty["metrics"] == []


@pytest.mark.parametrize("change", ["identity", "q32", "source", "permutation", "label_type", "extra"])
def test_corrupted_supervision_rejected(open_shard, change):
    obs = deepcopy(open_shard.observations[0])
    truth = json.loads(open_shard.data.read("sidecars.jsonl").splitlines()[0])
    if change == "identity":
        truth["split_seed"] += 1
    elif change == "q32":
        obs["log_f"][0] = float(np.float32(obs["log_f"][0]+.25))
    elif change == "source":
        truth["sources"][0]["f0"] = 10.
    elif change == "permutation":
        truth["permutation"][0] = truth["permutation"][1]
    elif change == "label_type":
        truth["source_ids"][0] = float(truth["source_ids"][0])
    else:
        truth["unaccounted_information"] = 1
    with pytest.raises(ValueError):
        supervision.reconstruct_truth(obs, truth, "train")


def test_sidecar_reader_requires_open_shard():
    with pytest.raises(PermissionError, match="OPEN"):
        supervision.open_truths(object())
