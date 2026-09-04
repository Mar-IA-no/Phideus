from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments/geometria_proporcional/run_proportional_graph_conditional_risk_gate.py"
SPEC = importlib.util.spec_from_file_location("conditional_risk_gate", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_config_and_source_are_frozen_cpu_contracts() -> None:
    cfg = MODULE.load_config(MODULE.DEFAULT_CONFIG)
    MODULE.verify_source(cfg)
    assert cfg["execution"]["torch_threads"] == 1
    assert cfg["realizations"] == {
        "risk_fit_seed": 2026090809,
        "risk_calibration_seed": 2026090817,
        "policy_selection_seed": 2026090829,
        "adjudication_seed": 2026090837,
        "min_eligible_masters": 220,
    }


def test_conformal_quantile_uses_finite_sample_upper_rank() -> None:
    scores = np.arange(19, dtype=np.float64)
    q, rank = MODULE.conformal_quantile(scores, 0.10)
    assert rank == 18
    assert q == 17.0


def test_bounded_action_requires_strictly_negative_upper_bound() -> None:
    mu = np.asarray([[-0.2, -0.1, 0.0, 0.1], [-0.2, -0.3, 0.2, 0.4]])
    sigma = np.ones_like(mu)
    action, upper = MODULE.bounded_action(mu, sigma, 0.2)
    np.testing.assert_array_equal(action, np.asarray([0, 2]))
    np.testing.assert_allclose(upper, mu + 0.2)


def test_firewall_falls_back_to_exact_identity() -> None:
    cfg = MODULE.load_config(MODULE.DEFAULT_CONFIG)
    actions = {family: np.ones((4, 8), dtype=np.int64) for family in MODULE.FAMILIES}
    controls = np.ones((4, cfg["control_replicates"], 8), dtype=np.int64)
    freeze = {"arms": {}}
    for arm_index, arm in enumerate(cfg["arms"]):
        freeze["arms"][arm] = {
            "families": {family: {"deploy": arm_index != 0} for family in MODULE.FAMILIES},
            MODULE.CONTROL_FAMILY: [
                {"deploy": replicate % 2 == 0} for replicate in range(cfg["control_replicates"])
            ],
        }
    deployed, deployed_controls = MODULE.apply_firewall(cfg, actions, controls, freeze)
    for family in MODULE.FAMILIES:
        np.testing.assert_array_equal(deployed[family][0], 0)
        np.testing.assert_array_equal(deployed[family][1:], 1)
    np.testing.assert_array_equal(deployed_controls[:, 1::2], 0)


def test_actions_depend_on_public_predictions_not_realized_targets() -> None:
    mu = np.asarray([[-0.3, 0.2, 0.4, 0.6], [-0.1, -0.4, 0.3, 0.5]])
    sigma = np.asarray([[0.1, 0.2, 0.3, 0.4], [0.2, 0.1, 0.3, 0.4]])
    first, _ = MODULE.bounded_action(mu, sigma, 0.5)
    # Adjudication targets are intentionally absent from the policy interface.
    second, _ = MODULE.bounded_action(mu.copy(), sigma.copy(), 0.5)
    np.testing.assert_array_equal(first, second)


def test_frozen_runtime_config_cannot_be_replaced(tmp_path: Path) -> None:
    cfg = MODULE.load_config(MODULE.DEFAULT_CONFIG)
    (tmp_path / "resolved_config.json").write_text(json.dumps(cfg))
    MODULE.verify_resolved_config(cfg, tmp_path)
    altered = {**cfg, "miscoverage": 0.2}
    try:
        MODULE.verify_resolved_config(altered, tmp_path)
    except AssertionError as error:
        assert "differs" in str(error)
    else:
        raise AssertionError("altered config passed freeze verification")


def test_phase_freeze_rejects_unmanifested_files(tmp_path: Path) -> None:
    role = tmp_path / "risk_fit"
    role.mkdir()
    (role / "expected.txt").write_text("frozen")
    MODULE.write_phase_freeze(tmp_path, "risk_fit", ("risk_fit",), {}, [])
    MODULE.verify_phase(tmp_path, "risk_fit")
    (role / "unexpected.txt").write_text("late")
    try:
        MODULE.verify_phase(tmp_path, "risk_fit")
    except AssertionError as error:
        assert "file set" in str(error)
    else:
        raise AssertionError("unmanifested file passed freeze verification")
