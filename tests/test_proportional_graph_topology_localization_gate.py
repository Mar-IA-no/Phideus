from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments/geometria_proporcional/run_proportional_graph_topology_localization_gate.py"
SPEC = importlib.util.spec_from_file_location("topology_localization_gate", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def graph_fixture() -> tuple[int, np.ndarray, np.ndarray]:
    edges = np.asarray([[0, 1], [0, 2], [1, 2], [1, 3], [2, 4], [3, 4]], dtype=np.int64)
    delta = np.asarray([0.2, -0.7, 0.1, 0.5, -0.3, 0.9], dtype=np.float64)
    return 5, edges, delta


def relabel_graph(
    n_nodes: int, edges: np.ndarray, delta: np.ndarray, permutation: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    transformed = permutation[edges]
    direction = np.where(transformed[:, 0] < transformed[:, 1], 1.0, -1.0)
    canonical = np.sort(transformed, axis=1)
    order = np.lexsort((canonical[:, 1], canonical[:, 0]))
    return canonical[order], (delta * direction)[order]


def zero_report(width: int) -> dict:
    model = {
        "intercept": -0.1,
        "mean": [0.0] * width,
        "scale": [1.0] * width,
        "coefficients": [0.0] * width,
    }
    return {"models": [model for _ in range(4)]}


def fake_models(cfg: dict) -> dict:
    widths = {
        "correction_scale": 2,
        "public_base": len(MODULE.BASE_FEATURE_ORDER),
        "topology_augmented": len(MODULE.BASE_FEATURE_ORDER) + len(MODULE.TOPOLOGY_FEATURE_ORDER),
    }
    return {
        "arms": {
            arm: {
                "families": {family: zero_report(width) for family, width in widths.items()},
                "topology_permuted": [
                    {"model": zero_report(widths["topology_augmented"])}
                    for _ in range(cfg["control_replicates"])
                ],
                "target_shuffled_topology": [
                    {"model": zero_report(widths["topology_augmented"])}
                    for _ in range(cfg["control_replicates"])
                ],
            }
            for arm in cfg["arms"]
        }
    }


def fake_thresholds(cfg: dict) -> dict:
    return {
        "arms": {
            arm: {
                "families": {
                    family: {"selected_threshold": 0.05} for family in MODULE.FAMILIES
                },
                **{
                    control: [
                        {"selected_threshold": 0.05}
                        for _ in range(cfg["control_replicates"])
                    ]
                    for control in MODULE.CONTROL_FAMILIES
                },
            }
            for arm in cfg["arms"]
        }
    }


def test_config_and_frozen_sources_are_valid_cpu_contracts() -> None:
    cfg = MODULE.load_config(MODULE.DEFAULT_CONFIG)
    MODULE.verify_sources(cfg)
    assert cfg["execution"]["torch_threads"] == 1
    assert cfg["realizations"] == {
        "calibration_seed": 2026090609,
        "selection_seed": 2026090617,
        "adjudication_seed": 2026090629,
        "min_eligible_masters": 220,
    }


def test_topology_features_are_finite_and_zero_safe() -> None:
    n_nodes, edges, delta = graph_fixture()
    features = MODULE.topology_features(n_nodes, edges, delta)
    assert features.shape == (len(MODULE.TOPOLOGY_FEATURE_ORDER),)
    assert np.all(np.isfinite(features))
    assert features[0] == 1.0
    np.testing.assert_array_equal(
        MODULE.topology_features(n_nodes, edges, np.zeros_like(delta)),
        np.zeros(len(MODULE.TOPOLOGY_FEATURE_ORDER)),
    )


def test_topology_features_respect_graph_and_scale_invariances() -> None:
    n_nodes, edges, delta = graph_fixture()
    expected = MODULE.topology_features(n_nodes, edges, delta)
    order = np.asarray([4, 1, 5, 0, 3, 2])
    np.testing.assert_allclose(
        MODULE.topology_features(n_nodes, edges[order], delta[order]), expected, atol=1e-14
    )
    np.testing.assert_allclose(
        MODULE.topology_features(n_nodes, edges[:, ::-1], -delta), expected, atol=1e-14
    )
    relabeled_edges, relabeled_delta = relabel_graph(
        n_nodes, edges, delta, np.asarray([3, 0, 4, 2, 1])
    )
    np.testing.assert_allclose(
        MODULE.topology_features(n_nodes, relabeled_edges, relabeled_delta), expected, atol=1e-14
    )
    np.testing.assert_allclose(
        MODULE.topology_features(n_nodes, edges, 7.3 * delta), expected, atol=1e-14
    )


def test_topology_control_preserves_signed_and_absolute_multisets() -> None:
    _, _, delta = graph_fixture()
    shuffled, permutation = MODULE.permuted_correction(delta, 13)
    assert not np.array_equal(permutation, np.arange(len(delta)))
    np.testing.assert_array_equal(np.sort(shuffled), np.sort(delta))
    np.testing.assert_array_equal(np.sort(np.abs(shuffled)), np.sort(np.abs(delta)))


def test_deployable_actions_do_not_depend_on_solver_targets() -> None:
    cfg = MODULE.load_config(MODULE.DEFAULT_CONFIG)
    rng = np.random.default_rng(23)
    features = rng.normal(size=(4, 2, 10, len(MODULE.BASE_FEATURE_ORDER)))
    topology = {
        "true": rng.normal(size=(4, 2, 10, len(MODULE.TOPOLOGY_FEATURE_ORDER))),
        "control": rng.normal(
            size=(4, 2, cfg["control_replicates"], 10, len(MODULE.TOPOLOGY_FEATURE_ORDER))
        ),
    }
    first = {"features": features, "quotient_rmse": rng.uniform(size=(4, 2, 5, 10))}
    second = {"features": features, "quotient_rmse": rng.uniform(size=(4, 2, 5, 10)) * 100.0}
    models, thresholds = fake_models(cfg), fake_thresholds(cfg)
    actions_a, controls_a, advantages_a = MODULE.apply_models(
        cfg, first, topology, models, thresholds
    )
    actions_b, controls_b, advantages_b = MODULE.apply_models(
        cfg, second, topology, models, thresholds
    )
    for name in actions_a:
        if name != "oracle_per_view":
            np.testing.assert_array_equal(actions_a[name], actions_b[name])
    for name in controls_a:
        np.testing.assert_array_equal(controls_a[name], controls_b[name])
    for name in advantages_a:
        np.testing.assert_array_equal(advantages_a[name], advantages_b[name])
