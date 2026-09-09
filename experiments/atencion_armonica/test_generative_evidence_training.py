"""Mechanical CPU trajectories; no campaign fitting, test draws or model choice."""
from copy import deepcopy
from io import BytesIO

import numpy as np
import pytest
import torch

from experiments.atencion_armonica.test_generative_evidence import fixture, norms
from src.atencion_armonica import generative_evidence as ge
from src.atencion_armonica.generative_evidence_model import collate
from src.atencion_armonica.generative_evidence_training import TrainingKernel, epoch_batches, collate_targets

IDS = [i for i in range(35) if i not in (10, 20)]


def kernel(arm="generative", *, ids=None):
    return TrainingKernel(arm, ge.CHECKPOINTS[0], ge.READER_SEEDS[0],
                          binding={"purpose": "mechanical-fixture-not-campaign", "version": 1},
                          scene_ids=IDS if ids is None else ids, device="cpu")


def batch(k):
    ids = k.expected_scene_ids()
    row = ge.observable_rows(*fixture())
    cn, en = norms(row)
    inp = ge.model_inputs(row, cn, en, k.arm, split_seed=10, scene_id=0)
    inputs = collate([inp for _ in ids])
    targets = collate_targets([np.full((2, 2), (int(i) % 7)/8, np.float32) for i in ids], inputs["candidate_mask"])
    return inputs, targets, ids


def assert_nested_equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, np.ndarray):
        assert a.dtype == b.dtype and np.array_equal(a, b)
    elif isinstance(a, dict):
        assert set(a) == set(b)
        for k in a:
            assert_nested_equal(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert type(a) is type(b) and len(a) == len(b)
        for x, y in zip(a, b):
            assert_nested_equal(x, y)
    else:
        assert a == b


def test_schedule_covers_eligible_roster_once_with_partial_batch():
    for epoch in range(50):
        bs = epoch_batches(ge.READER_SEEDS[0], epoch, IDS)
        assert list(map(len, bs)) == [32, 1]
        assert sorted(np.concatenate(bs).tolist()) == IDS
        expected = np.array(IDS)[np.random.default_rng(np.random.SeedSequence([ge.READER_SEEDS[0], epoch])).permutation(33)]
        np.testing.assert_array_equal(np.concatenate(bs), expected)
    assert len(epoch_batches(ge.READER_SEEDS[0], 0, [3])[0]) == 1
    for ids in ([], [0, 0], [1, 0], [True], [4096]):
        with pytest.raises(ValueError):
            epoch_batches(ge.READER_SEEDS[0], 0, ids)


def test_serialized_mid_epoch_resume_is_bit_exact():
    continuous = kernel()
    initial = continuous.state()
    for _ in range(6):
        continuous.step(*batch(continuous))
    expected = continuous.state()
    interrupted = kernel()
    interrupted.restore(initial)
    interrupted.step(*batch(interrupted))
    assert interrupted.epoch == 0 and interrupted.next_batch == 1
    stream = BytesIO()
    torch.save(interrupted.state(), stream)
    stream.seek(0)
    saved = torch.load(stream, map_location="cpu", weights_only=False)
    resumed = kernel()
    resumed.restore(saved)
    for _ in range(5):
        resumed.step(*batch(resumed))
    assert_nested_equal(expected, resumed.state())


def test_all_50_epochs_and_zero_channel_activity():
    k = kernel("local", ids=[3])
    for _ in range(50):
        k.step(*batch(k))
    assert k.epoch == 50 and k.next_batch == 0 and k.steps == 50
    assert len(k.history) == 50
    for row in k.history:
        assert row["scene_count"] == 1 and row["updates"] == 1
        assert row["moments"]["evidence_inputs"]["mean"] == [0.]*6
        assert row["moments"]["evidence_inputs"]["variance"] == [0.]*6
        assert row["gradient_norms"]["generative_columns"] == 0.
    with pytest.raises(ValueError, match="already complete"):
        k.expected_scene_ids()
    restored = kernel("local", ids=[3])
    restored.restore(k.state())
    assert_nested_equal(k.state(), restored.state())


def test_epoch_loss_accounts_for_partial_batch_scene_mass():
    k = kernel()
    first = k.step(*batch(k))
    second = k.step(*batch(k))
    assert k.history[0]["loss"] == (32*first+second)/33
    assert k.history[0]["scene_count"] == 33


@pytest.mark.parametrize("field", ["binding", "schedule", "steps", "recipe", "model", "rng", "moment"])
def test_corrupt_restore_rejected_before_model_or_rng_mutation(field):
    k = kernel()
    k.step(*batch(k))
    original = k.state()
    bad = deepcopy(original)
    if field == "binding":
        bad["binding"]["version"] += 1
    elif field == "schedule":
        bad["batch_hash"] = "0"*64
    elif field == "steps":
        bad["steps"] += 1
    elif field == "recipe":
        bad["optimizer"]["param_groups"][0]["lr"] *= 2
    elif field == "model":
        bad["model"]["group1.weight"][0, 0] = float("nan")
    elif field == "rng":
        bad["torch_rng"] = torch.zeros(2, dtype=torch.uint8)
    else:
        bad["optimizer"]["state"][0]["exp_avg_sq"][0, 0] = -1
    with pytest.raises(ValueError):
        k.restore(bad)
    assert_nested_equal(original, k.state())


def test_incomplete_update_cannot_be_snapshotted_or_continued():
    k = kernel()
    inputs, targets, ids = batch(k)
    targets[0, 0, 0] = float("nan")
    with pytest.raises(ValueError):
        k.step(inputs, targets, ids)
    with pytest.raises(ValueError, match="incomplete update"):
        k.state()
    with pytest.raises(ValueError, match="incomplete update"):
        k.step(*batch(k))
