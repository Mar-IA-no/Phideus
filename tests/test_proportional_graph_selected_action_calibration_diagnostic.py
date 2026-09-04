from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments/geometria_proporcional/run_proportional_graph_selected_action_calibration_diagnostic.py"
SPEC = importlib.util.spec_from_file_location("selected_action_calibration", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_config_and_opened_source_are_frozen_cpu_contracts() -> None:
    cfg = MODULE.load_config(MODULE.DEFAULT_CONFIG)
    source = MODULE.verify_source(cfg)
    assert cfg["miscoverage"] == 0.10
    assert source["execution"]["torch_threads"] == 1


def test_proposer_uses_first_argmin_and_no_target() -> None:
    mu = np.asarray([[-0.2, -0.2, 0.1, 0.3], [-0.1, -0.3, 0.2, 0.4]])
    predicted = np.asarray([[0.1, 0.1, 0.0, 0.0], [0.0, 0.1, 0.0, 0.0]])
    proposal, bound = MODULE.propose(mu, predicted)
    np.testing.assert_array_equal(proposal, np.asarray([1, 2]))
    np.testing.assert_allclose(bound, np.asarray([-0.1, -0.2]))


def test_selected_score_is_bounded_by_simultaneous_score() -> None:
    residual = np.asarray([[0.3, -0.2, 0.1, 0.4], [0.1, 0.2, -0.1, 0.0]])
    predicted = np.asarray([[0.1, 0.0, 0.2, 0.3], [0.0, 0.1, 0.0, 0.2]])
    proposal = np.asarray([1, 2])
    selected = MODULE.selected_score(residual, predicted, proposal)
    simultaneous = np.max(residual - predicted, axis=1)
    assert np.all(selected <= simultaneous)


def test_calibrated_action_requires_strict_negative_bound() -> None:
    proposal = np.asarray([1, 2, 3])
    base = np.asarray([-0.2, -0.1, 0.1])
    action, upper = MODULE.calibrated_action(proposal, base, 0.1)
    np.testing.assert_array_equal(action, np.asarray([1, 0, 0]))
    np.testing.assert_allclose(upper, np.asarray([-0.1, 0.0, 0.2]))


def test_firewall_rejection_is_exact_identity() -> None:
    cfg = {"arms": ["arm"], "control_replicates": 2}
    actions = {
        family: np.ones((1, 4), dtype=np.int64)
        for family in MODULE.FAMILIES
    }
    controls = np.ones((1, 2, 4), dtype=np.int64)
    firewall = {"arms": {"arm": {
        "families": {
            family: {"deploy": family == "topology_selected_action"}
            for family in MODULE.FAMILIES
        },
        MODULE.CONTROL_FAMILY: [{"deploy": True}, {"deploy": False}],
    }}}
    deployed, deployed_controls = MODULE.apply_firewall(cfg, actions, controls, firewall)
    np.testing.assert_array_equal(deployed["constant_selected_action"], 0)
    np.testing.assert_array_equal(deployed["public_base_selected_action"], 0)
    np.testing.assert_array_equal(deployed["topology_selected_action"], 1)
    np.testing.assert_array_equal(deployed_controls[:, 0], 1)
    np.testing.assert_array_equal(deployed_controls[:, 1], 0)
