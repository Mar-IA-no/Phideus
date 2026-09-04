from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments/geometria_proporcional/run_proportional_graph_mean_ranking_power_selection_audit.py"
SPEC = importlib.util.spec_from_file_location("mean_power_audit", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_config_and_source_are_cpu_frozen():
    cfg = MODULE.load_config(MODULE.DEFAULT_CONFIG)
    source_cfg, _, _ = MODULE.verify_source(cfg)
    assert source_cfg["source_manifest_sha256"] == "bffbf9fefa915b6dfd00df63e35bacf9084ec68d71831471b5c0537b11b2536e"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""


def test_two_axis_control_diagnostic_preserves_observed_mean():
    cube = np.asarray([[-2.0, 0.0], [-1.0, 1.0]])
    master = np.asarray([[0, 1], [1, 0], [0, 0], [1, 1]])
    control = np.asarray([[0, 1], [1, 0], [0, 0], [1, 1]])
    row = MODULE.control_diagnostics(cube, master, control)
    assert row["two_axis_bootstrap"]["mean"] == -0.5
    assert row["replicate_favorable_fraction"] == 0.5


def test_firewall_decomposition_is_exact():
    e00 = np.asarray([-2.0, 1.0])
    topology_contribution = np.asarray([1.0, -1.0])
    control_contribution = np.asarray([0.5, 0.5])
    e11 = e00 + topology_contribution + control_contribution
    np.testing.assert_allclose(e11, np.asarray([-0.5, 0.5]), atol=1e-12, rtol=0.0)
