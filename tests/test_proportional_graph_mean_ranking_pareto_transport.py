from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "experiments/geometria_proporcional/run_proportional_graph_mean_ranking_pareto_transport.py"
)
SPEC = importlib.util.spec_from_file_location("mean_pareto_transport", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_config_and_sources_are_cpu_frozen():
    cfg = MODULE.load_config(MODULE.DEFAULT_CONFIG)
    attribution_cfg, _, _ = MODULE.verify_sources(cfg)
    assert (
        attribution_cfg["source_manifest_sha256"]
        == "bffbf9fefa915b6dfd00df63e35bacf9084ec68d71831471b5c0537b11b2536e"
    )
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""


def test_pareto_mask_preserves_tradeoffs_and_exact_ties():
    points = np.asarray(
        [
            [0.0, 2.0],
            [1.0, 1.0],
            [2.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0],
        ]
    )
    np.testing.assert_array_equal(
        MODULE.pareto_mask(points, 1e-12),
        np.asarray([True, True, True, True, False]),
    )


def test_transport_reports_changed_policy_coordinates():
    selection = np.asarray([True, True, False, False, False, False, False])
    adjudication = np.asarray([False, True, True, False, False, False, False])
    selection_values = np.zeros((7, 2, 3))
    adjudication_values = np.ones((7, 2, 3))
    row = MODULE.transport_record(
        selection, adjudication, selection_values, adjudication_values
    )
    assert row["jaccard"] == 1 / 3
    assert row["retention"] == 0.5
    assert row["oracle_recall"] == 0.5
    assert row["lost"][0]["policy_id"] == "identity"
    assert row["lost"][0]["policy_selection"]["iid_delta"] == 0.0
    assert row["added"][0]["policy_id"] == "budget_0.02"
    assert row["added"][0]["adjudication"]["grouped_delta"] == 1.0


def test_set_coverage_statuses_are_directional():
    topology = np.asarray([[0.0, 0.0]])
    worse_control = np.asarray([[1.0, 1.0]])
    tradeoff_control = np.asarray([[-1.0, 1.0]])
    assert MODULE.compare_sets(topology, worse_control, 1e-12) == "TOPOLOGY_DOMINATES"
    assert MODULE.compare_sets(worse_control, topology, 1e-12) == "CONTROL_DOMINATES"
    assert MODULE.compare_sets(topology, topology.copy(), 1e-12) == "EQUIVALENT"
    assert MODULE.compare_sets(topology, tradeoff_control, 1e-12) == "INCOMPARABLE"
