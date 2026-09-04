from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "experiments/geometria_proporcional/run_proportional_graph_irls_surrogate_fidelity.py"
)
SPEC = importlib.util.spec_from_file_location("irls_surrogate_fidelity", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

from geometria_proporcional.proportional_graph_contract import (  # noqa: E402
    ProportionalGraphConfig,
    generate_graph_views,
)
from geometria_proporcional.proportional_graph_neural import (  # noqa: E402
    differentiable_huber_irls_fixed,
)


def small_view():
    config = ProportionalGraphConfig(
        masters=8,
        train_fraction=0.25,
        calibration_fraction=0.125,
        validation_fraction=0.125,
        n_min=5,
        n_max=5,
        irls_iterations=100,
    )
    return generate_graph_views(config)[0]


def test_torch_fixed_matches_independent_numpy_reference() -> None:
    view = small_view()
    values = view.public.observed_log_ratio.astype(np.float64)
    nx, nw, nm, no, nc = MODULE.numpy_fixed_irls(
        view.public, values, steps=8, delta=1.5, damping=0.5, weight_floor=1e-3
    )
    output = differentiable_huber_irls_fixed(
        view.public,
        torch.tensor(values, dtype=torch.float64),
        steps=8,
        delta=1.5,
        damping=0.5,
        weight_floor=1e-3,
    )
    np.testing.assert_allclose(
        output.x_hat.detach().numpy(), nx, atol=1e-12, rtol=1e-12
    )
    np.testing.assert_allclose(
        output.normalized_weights.detach().numpy(), nw, atol=1e-12, rtol=1e-12
    )
    np.testing.assert_allclose(
        output.min_huber_margin.detach().numpy(), nm, atol=1e-12, rtol=1e-12
    )
    assert np.isclose(float(output.huber_objective), no, atol=1e-12, rtol=1e-12)
    assert np.isclose(float(output.final_solution_change), nc, atol=1e-12, rtol=1e-12)


def test_autograd_matches_central_difference_away_from_kink() -> None:
    view = small_view()
    values = view.public.observed_log_ratio.astype(np.float64)
    tensor = torch.tensor(values, dtype=torch.float64, requires_grad=True)
    output = differentiable_huber_irls_fixed(
        view.public, tensor, steps=4, delta=1.5, damping=0.5, weight_floor=1e-3
    )
    target = torch.tensor(view.private.x_true, dtype=torch.float64)
    torch.mean((output.x_hat - target) ** 2).backward()
    edge = int(np.flatnonzero(view.public.edge_valid)[0])
    h = 1e-6
    losses = []
    for sign in (-1.0, 1.0):
        perturbed = values.copy()
        perturbed[edge] += sign * h
        x_hat, _, _, _, _ = MODULE.numpy_fixed_irls(
            view.public, perturbed, steps=4, delta=1.5, damping=0.5, weight_floor=1e-3
        )
        losses.append(float(np.mean((x_hat - view.private.x_true) ** 2)))
    numeric = (losses[1] - losses[0]) / (2.0 * h)
    assert np.isclose(tensor.grad.detach().numpy()[edge], numeric, atol=1e-7, rtol=1e-5)


def test_stratified_selection_is_deterministic() -> None:
    config = ProportionalGraphConfig(masters=16, n_min=4, n_max=7)
    views = generate_graph_views(config)
    first = MODULE.select_stratified(views, 8)
    second = MODULE.select_stratified(list(reversed(views)), 8)
    assert [v.private.view_id for v in first] == [v.private.view_id for v in second]
    assert len({v.public.n_nodes for v in first}) == 4


def test_official_config_is_strict_and_cpu_only() -> None:
    config = MODULE.load_config(MODULE.DEFAULT_CONFIG)
    assert config["runtime"]["torch_threads"] == 1
    assert config["gradient_depths"] == [16, 64, 256]
