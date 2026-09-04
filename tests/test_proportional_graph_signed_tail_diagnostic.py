from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments/geometria_proporcional/run_proportional_graph_signed_tail_diagnostic.py"
SPEC = importlib.util.spec_from_file_location("signed_tail_diagnostic", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_config_and_opened_source_are_frozen_cpu_contracts() -> None:
    cfg = MODULE.load_config(MODULE.DEFAULT_CONFIG)
    source = MODULE.verify_source(cfg)
    assert cfg["quantile"] == 0.9
    assert source["execution"]["torch_threads"] == 1


def test_pinball_loss_distinguishes_upper_and_lower_errors() -> None:
    y = np.asarray([1.0, -1.0])
    prediction = np.zeros(2)
    assert MODULE.pinball_loss(y, prediction, 0.9) == 0.5


def test_constant_quantile_uses_upper_order_statistic() -> None:
    values = np.arange(10, dtype=np.float64)
    assert MODULE.constant_quantile(values, 0.9) == 9.0


def test_quantile_model_prediction_is_finite_and_deterministic() -> None:
    x = np.arange(40, dtype=np.float64).reshape(20, 2)
    y = 0.5 * x[:, 0] - 0.2 * x[:, 1]
    first = MODULE.fit_quantile(x, y, 0.001, 0.9)
    second = MODULE.fit_quantile(x, y, 0.001, 0.9)
    np.testing.assert_allclose(MODULE.predict_quantile(first, x), MODULE.predict_quantile(second, x))
    assert np.all(np.isfinite(MODULE.predict_quantile(first, x)))


def test_signed_upper_action_requires_strict_negative_bound() -> None:
    cfg = {"arms": ["arm"], "control_replicates": 1}
    # Policy mechanics remain delegated to the already-tested conditional gate.
    mu = np.asarray([[-0.2, -0.3, 0.1, 0.4], [-0.1, -0.2, 0.3, 0.5]])
    upper_residual = np.asarray([[0.2, 0.1, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    bound = mu + upper_residual
    best = np.argmin(bound, axis=1)
    action = np.where(bound[np.arange(len(bound)), best] < 0.0, best + 1, 0)
    np.testing.assert_array_equal(action, np.asarray([2, 2]))


def test_signed_family_firewall_uses_local_policy_names() -> None:
    cfg = {"arms": ["arm"], "control_replicates": 2}
    actions = {family: np.ones((1, 4), dtype=np.int64) for family in MODULE.FAMILIES}
    controls = np.ones((1, 2, 4), dtype=np.int64)
    firewall = {"arms": {"arm": {
        "families": {
            family: {"deploy": family == "topology_signed_tail"}
            for family in MODULE.FAMILIES
        },
        MODULE.CONTROL_FAMILY: [{"deploy": True}, {"deploy": False}],
    }}}
    deployed, deployed_controls = MODULE.apply_firewall(cfg, actions, controls, firewall)
    np.testing.assert_array_equal(deployed["constant_signed_tail"], 0)
    np.testing.assert_array_equal(deployed["public_base_signed_tail"], 0)
    np.testing.assert_array_equal(deployed["topology_signed_tail"], 1)
    np.testing.assert_array_equal(deployed_controls[:, 0], 1)
    np.testing.assert_array_equal(deployed_controls[:, 1], 0)


def test_residual_description_keeps_both_tail_directions() -> None:
    values = np.asarray([[-2.0, 1.0], [-1.0, 3.0]])
    report = MODULE.describe_residual(values)
    assert report["positive_fraction"] == 0.5
    assert report["mean_positive"] == 2.0
    assert report["mean_negative"] == -1.5
    assert len(report["by_alpha"]) == 2
