"""Finite arithmetic recovery fixtures, not a sampled or scientific campaign."""
from copy import deepcopy

import numpy as np
import pytest
import torch

from experiments.atencion_armonica.test_geometric_decision_core import interface
from src.atencion_armonica.generative_evidence_cell import state_digest
from src.atencion_armonica.geometric_decision_core import ARMS, READER_SEEDS, route_inputs
from src.atencion_armonica.geometric_decision_model import collate
from src.atencion_armonica.geometric_decision_training import TrainingKernel, collate_targets, epoch_batches


@pytest.fixture(autouse=True)
def runtime():
    old = (torch.get_num_threads(), torch.are_deterministic_algorithms_enabled(),
           torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    yield
    torch.set_num_threads(old[0])
    torch.use_deterministic_algorithms(old[1])
    torch.backends.cuda.matmul.allow_tf32 = old[2]
    torch.backends.cudnn.allow_tf32 = old[3]


def kernel(arm="geometric_decision", count=35):
    return TrainingKernel(arm, 2026090721, READER_SEEDS[0], binding={"fixture": "arithmetic-no-sampler"},
                          scene_ids=list(range(count)), device="cpu")


def step(cell):
    row, _, _ = interface()
    row = route_inputs(row, cell.arm.rsplit("_", 1)[0])
    ids = cell.expected_scene_ids()
    batch = collate([row]*len(ids))
    target = np.array([[.7, .6], [.1, .2]], np.float32)
    targets = collate_targets([target]*len(ids), batch["candidate_mask"])
    return cell.step(batch, targets, ids)


@pytest.mark.parametrize("arm", ARMS)
def test_exact_mid_epoch_restore_all_routes_and_objectives(arm):
    cell = kernel(arm)
    initial = cell.state()
    step(cell)
    middle = cell.state()
    assert middle["next_batch"] == 1 and middle["accumulator"]["scene_count"] == 32
    step(cell)  # The partial batch has three scenes, not 32.
    step(cell)
    expected = state_digest(cell.state())
    restored = kernel(arm)
    restored.restore(middle)
    step(restored)
    step(restored)
    assert state_digest(restored.state()) == expected
    assert cell.history[0]["scene_count"] == 35 and cell.history[0]["updates"] == 2
    restored.restore(initial)
    assert restored.steps == restored.epoch == restored.next_batch == 0


def test_fifty_epochs_are_terminal_without_reduced_recipe():
    cell = kernel("injection_mse", count=2)
    for _ in range(50):
        step(cell)
    assert cell.epoch == 50 and cell.steps == 50 and len(cell.history) == 50
    assert cell.next_batch == 0 and cell.accumulator == {}
    other = kernel("injection_mse", count=2)
    other.restore(cell.state())
    assert state_digest(other.state()) == state_digest(cell.state())
    with pytest.raises(ValueError, match="50 epochs"):
        other.expected_scene_ids()


@pytest.mark.parametrize("mutation", ["arm", "position", "recipe", "moments", "runtime", "nan"])
def test_invalid_snapshot_rejected_before_live_state_changes(mutation):
    cell = kernel()
    step(cell)
    state = cell.state()
    before = state_digest(state)
    if mutation == "arm":
        state["arm"] = "injection_mse"
    elif mutation == "position":
        state["steps"] += 1
    elif mutation == "recipe":
        state["optimizer"]["param_groups"][0]["lr"] = 1.
    elif mutation == "moments":
        state["optimizer"]["state"][0]["exp_avg_sq"].fill_(-1.)
    elif mutation == "runtime":
        state["runtime"]["threads"] = 2
    else:
        state["model"]["partition2.bias"][0] = torch.nan
    with pytest.raises(ValueError):
        cell.restore(state)
    assert state_digest(cell.state()) == before


def test_failed_update_requires_restore_not_in_place_retry():
    cell = kernel(count=2)
    initial = cell.state()
    row, _, _ = interface()
    batch = collate([row]*2)
    bad = torch.full((2, 2, 2), float("nan"), dtype=torch.float32)
    with pytest.raises(ValueError):
        cell.step(batch, bad, cell.expected_scene_ids())
    with pytest.raises(ValueError, match="incomplete"):
        cell.state()
    with pytest.raises(ValueError, match="incomplete"):
        step(cell)
    cell.restore(initial)
    step(cell)
    assert cell.steps == 1


def test_paired_full_roster_and_exact_target_masks():
    ids = list(range(4096))
    for epoch in (0, 49):
        batches = epoch_batches(READER_SEEDS[0], epoch, ids)
        assert len(batches) == 128 and all(len(b) == 32 for b in batches)
        assert sorted(np.concatenate(batches).tolist()) == ids
        assert all(np.array_equal(a, b) for a, b in zip(batches, epoch_batches(READER_SEEDS[0], epoch, ids)))
    with pytest.raises(ValueError):
        epoch_batches(READER_SEEDS[0], 50, ids)
    with pytest.raises(ValueError):
        collate_targets([np.zeros((1, 2), np.float32)], torch.tensor([[False, True]]))
