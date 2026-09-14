"""Signed readout and coordinate transport on explicit arithmetic fixtures."""
import copy

import numpy as np
import pytest
import torch

from experiments.atencion_armonica.test_geometric_decision_core import interface
from src.atencion_armonica.geometric_decision_core import READER_SEEDS, ROUTES
from src.atencion_armonica.geometric_decision_model import GeometricDecisionHead
from src.atencion_armonica.geometric_decision_inference import predict, transport_prediction


def empty():
    return {"groups": np.empty((0, 9), np.float32), "globals": np.empty((0, 17), np.float32),
        "evidence": np.empty((0, 8), np.float32), "incidence": np.empty((0, 0), np.float32)}


@pytest.mark.parametrize("route", ROUTES)
def test_initial_readout_exact_bypass_and_full_empty_scene_support(route):
    row, _, _ = interface()
    before = copy.deepcopy(row)
    model = GeometricDecisionHead(READER_SEEDS[0], route)
    rows = [row, *[empty() for _ in range(510)], row]
    outputs = predict(model, rows, device="cpu", check=lambda: None)
    expected = row["evidence"][:, 6 if route == "geometric" else 7].astype(np.float64)
    if route in ("injection", "local"):
        expected[:] = 0
    assert outputs["components"].dtype == np.float64 and outputs["offsets"].shape == (513,)
    np.testing.assert_array_equal(outputs["energy"], np.tile(expected, 2))
    np.testing.assert_array_equal(outputs["components"].sum(-1), outputs["energy"])
    for key in row:
        np.testing.assert_array_equal(row[key], before[key])


def test_signed_negative_outputs_are_not_clipped_and_empty_roster_has_no_forward(monkeypatch):
    row, _, _ = interface()
    model = GeometricDecisionHead(READER_SEEDS[0], "injection")
    with torch.no_grad():
        model.partition2.bias.fill_(-2)
    outputs = predict(model, [row], device="cpu", check=lambda: None)
    assert np.all(outputs["components"] == -2) and np.all(outputs["energy"] == -4)
    def forbidden(*args, **kwargs):
        raise AssertionError("empty scenes must not forward")
    monkeypatch.setattr(model, "forward", forbidden)
    outputs = predict(model, [empty() for _ in range(512)], device="cpu", check=lambda: None)
    assert outputs["components"].shape == (0, 2) and np.all(outputs["offsets"] == 0)


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("initial", [True, False])
def test_transport_moves_weights_and_bypass_preserving_correspondence(route, initial):
    row, raw, _ = interface()
    model = GeometricDecisionHead(READER_SEEDS[0], route)
    if not initial:
        with torch.no_grad():
            model.partition2.weight.copy_(torch.linspace(-.1, .1, 64).reshape(2, 32))
            model.partition2.bias.copy_(torch.tensor([-.2, .3]))
    original_weights = {k: v.detach().clone() for k, v in model.state_dict().items()}
    result = transport_prediction(model, row, raw["partitions"], device="cpu", check=lambda: None)
    assert result["diagnostic"]["within_numeric_tolerance"]
    assert result["diagnostic"]["same_exact_choice"]
    assert result["channel_order"].tolist() == list(range(7, -1, -1))
    assert result["bypass_channel"] == (1 if route == "geometric" else 0 if route == "decoupled" else None)
    for key, value in model.state_dict().items():
        assert torch.equal(value, original_weights[key])


def test_observable_port_rejects_targets_wrong_dtype_and_mismatched_incidence():
    row, raw, _ = interface()
    model = GeometricDecisionHead(READER_SEEDS[0], "geometric")
    with pytest.raises(ValueError, match="supervision"):
        predict(model, [{**row, "targets": np.zeros((2, 2), np.float32)}], device="cpu", check=lambda: None)
    with pytest.raises(ValueError, match="observable"):
        predict(model, [{**row, "evidence": row["evidence"].astype(np.float64)}], device="cpu", check=lambda: None)
    changed = copy.deepcopy(row)
    changed["incidence"] = np.roll(changed["incidence"], 1, axis=1)
    with pytest.raises(ValueError, match="incidence"):
        transport_prediction(model, changed, raw["partitions"], device="cpu", check=lambda: None)
