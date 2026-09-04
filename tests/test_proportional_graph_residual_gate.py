from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT / "experiments/geometria_proporcional/run_proportional_graph_residual_gate.py"
)
SPEC = importlib.util.spec_from_file_location("residual_gate", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

from geometria_proporcional.proportional_graph_contract import (  # noqa: E402
    ProportionalGraphConfig,
    generate_graph_views,
)


def test_config_freezes_cpu_gate_contract() -> None:
    cfg = MODULE.load_config(MODULE.DEFAULT_CONFIG)
    assert cfg["alphas"] == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert cfg["execution"]["torch_threads"] == 1
    assert cfg["shuffle_replicates"] == 16
    assert len(cfg["source_manifest_sha256"]) == 64
    MODULE.verify_source(
        ROOT / cfg["source_loss_contrast"], cfg["source_manifest_sha256"]
    )


def test_public_features_require_only_public_observation() -> None:
    view = generate_graph_views(ProportionalGraphConfig(masters=8))[0]

    class PublicOnly:
        public = view.public

        @property
        def private(self):
            raise AssertionError("private state was accessed")

    features = MODULE.public_features(PublicOnly(), view.public.observed_log_ratio)
    assert features.shape == (15,)
    assert np.all(np.isfinite(features))


def test_alpha_choice_preserves_identity_on_ties() -> None:
    predicted = np.asarray([[0.0, 0.0, 0.0, 0.0], [-0.2, -0.2, 0.1, 0.2]])
    np.testing.assert_array_equal(MODULE.choose_alpha(predicted), np.asarray([0, 1]))


def test_stratified_rotation_is_deranged_and_stratum_preserving() -> None:
    n_nodes = np.asarray([8, 8, 8, 9, 9, 9, 9])
    permutation = MODULE.stratified_rotation(n_nodes, 123)
    assert np.all(permutation != np.arange(len(n_nodes)))
    np.testing.assert_array_equal(n_nodes[permutation], n_nodes)


def test_ridge_gate_shapes_and_determinism() -> None:
    cfg = MODULE.load_config(MODULE.DEFAULT_CONFIG)
    rng = np.random.default_rng(7)
    x = rng.normal(size=(30, 3))
    y = np.column_stack([x[:, 0] * scale for scale in (0.1, 0.2, 0.3, 0.4)])
    masters = np.asarray([f"m{i}" for i in range(len(x))])
    first = MODULE.fit_gate(x, y, x, masters, cfg)
    second = MODULE.fit_gate(x, y, x, masters, cfg)
    assert first[0] == second[0]
    np.testing.assert_array_equal(first[1], second[1])
    np.testing.assert_array_equal(first[2], second[2])
    assert first[1].shape == (30, 4)
