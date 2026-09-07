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
    incidence_matrix,
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


def numpy_fixed_weighted_irls(
    observation, values: np.ndarray, base_weights: np.ndarray, *, steps: int
) -> tuple[np.ndarray, np.ndarray, float]:
    """Independent base-weighted reference for the compatibility extension."""
    valid = np.asarray(observation.edge_valid, dtype=bool)
    incidence = incidence_matrix(observation.n_nodes, observation.edge_index)[valid]
    y = np.asarray(values, dtype=np.float64)[valid]
    base = np.clip(np.asarray(base_weights, dtype=np.float64)[valid], 1e-3, None)
    base /= base.mean()
    variance = np.asarray(observation.edge_variance, dtype=np.float64)[valid]
    scale = max(float(np.sqrt(np.median(variance))), 1e-8)
    threshold = 1.5 * scale
    weights = base.copy()

    def solve(local: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        normalized = np.clip(local, 1e-3, None)
        normalized /= normalized.mean()
        laplacian = incidence.T @ (normalized[:, None] * incidence)
        rhs = incidence.T @ (normalized * y)
        ones = np.ones((observation.n_nodes, 1), dtype=np.float64)
        kkt = np.block([[laplacian, ones], [ones.T, np.zeros((1, 1))]])
        solution = np.linalg.solve(kkt, np.concatenate((rhs, np.zeros(1))))
        return solution[:-1], normalized

    for _ in range(steps):
        x_hat, _ = solve(weights)
        residual = incidence @ x_hat - y
        candidate = np.ones_like(residual)
        large = np.abs(residual) > threshold
        candidate[large] = threshold / np.abs(residual[large])
        candidate = np.clip(candidate, 1e-3, 1.0)
        weights = base * candidate
    x_hat, normalized = solve(weights)
    normalized_residual = (incidence @ x_hat - y) / scale
    magnitude = np.abs(normalized_residual)
    huber = np.where(
        magnitude <= 1.5,
        0.5 * normalized_residual**2,
        1.5 * (magnitude - 0.75),
    )
    full_weights = np.zeros(len(values), dtype=np.float64)
    full_weights[valid] = normalized
    return x_hat, full_weights, float(np.sum(base * huber))


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


def test_base_weighted_torch_matches_independent_numpy_reference() -> None:
    view = small_view()
    values = view.public.observed_log_ratio.astype(np.float64)
    base = np.linspace(0.15, 0.95, len(values), dtype=np.float64)
    nx, nw, no = numpy_fixed_weighted_irls(view.public, values, base, steps=8)
    output = differentiable_huber_irls_fixed(
        view.public,
        torch.tensor(values, dtype=torch.float64),
        base_weights=torch.tensor(base, dtype=torch.float64),
        steps=8,
        delta=1.5,
        damping=1.0,
        weight_floor=1e-3,
    )
    np.testing.assert_allclose(output.x_hat.detach().numpy(), nx, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(
        output.normalized_weights.detach().numpy(), nw, atol=1e-12, rtol=1e-12
    )
    assert np.isclose(float(output.huber_objective), no, atol=1e-12, rtol=1e-12)


def test_base_weight_gradient_matches_central_difference() -> None:
    view = small_view()
    values = view.public.observed_log_ratio.astype(np.float64)
    base = np.linspace(0.15, 0.95, len(values), dtype=np.float64)
    base_tensor = torch.tensor(base, dtype=torch.float64, requires_grad=True)
    output = differentiable_huber_irls_fixed(
        view.public,
        torch.tensor(values, dtype=torch.float64),
        base_weights=base_tensor,
        steps=8,
        delta=1.5,
        damping=1.0,
        weight_floor=1e-3,
    )
    target = torch.tensor(view.private.x_true, dtype=torch.float64)
    torch.mean((output.x_hat - target) ** 2).backward()
    edge = int(np.flatnonzero(view.public.edge_valid)[0])
    h = 1e-6
    losses = []
    for sign in (-1.0, 1.0):
        perturbed = base.copy()
        perturbed[edge] += sign * h
        x_hat, _, _ = numpy_fixed_weighted_irls(
            view.public, values, perturbed, steps=8
        )
        losses.append(float(np.mean((x_hat - view.private.x_true) ** 2)))
    numeric = (losses[1] - losses[0]) / (2.0 * h)
    assert np.isclose(
        base_tensor.grad.detach().numpy()[edge], numeric, atol=1e-7, rtol=1e-5
    )


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
