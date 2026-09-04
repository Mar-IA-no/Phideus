from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments/geometria_proporcional/run_proportional_graph_equal_budget_ranking_diagnostic.py"
SPEC = importlib.util.spec_from_file_location("equal_budget_ranking", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_config_and_r365_source_are_frozen_cpu_contracts() -> None:
    cfg = MODULE.load_config(MODULE.DEFAULT_CONFIG)
    source = MODULE.verify_source(cfg)
    assert source["control_replicates"] == 16
    assert cfg["budget_fractions"] == [0.01, 0.02, 0.05, 0.10, 0.20, 0.40]
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""


def test_selected_component_uses_one_based_common_proposal() -> None:
    values = np.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    proposal = np.asarray([2, 1, 2])
    np.testing.assert_array_equal(MODULE.selected_component(values, proposal), [2.0, 3.0, 6.0])


def test_equal_budget_action_uses_stable_index_tie_break() -> None:
    score = np.asarray([0.2, 0.1, 0.1, 0.3, 0.0])
    proposal = np.asarray([1, 2, 3, 4, 2])
    action = MODULE.equal_budget_action(score, proposal, 0.4)
    np.testing.assert_array_equal(action, [0, 2, 0, 0, 2])
    assert np.count_nonzero(action) == 2


def test_jaccard_compares_selected_views_not_alpha_values() -> None:
    left = np.asarray([0, 1, 2, 0])
    right = np.asarray([0, 4, 0, 3])
    assert MODULE.jaccard(left, right) == 1.0 / 3.0


def test_common_proposal_is_public_base_and_shared_by_all_scores() -> None:
    cfg = MODULE.load_config(MODULE.DEFAULT_CONFIG)
    source_cfg = MODULE.transport.load_config(MODULE.source_root(cfg) / "resolved_config.json")
    reconstructed = MODULE.transport.reconstruct_b(source_cfg)
    scores = MODULE.common_proposal_scores(reconstructed, 0)
    state = reconstructed["state"]
    expected, _ = MODULE.selected.propose(
        state["mu"][0], state["predicted"]["public_base_selected_action"][0]
    )
    np.testing.assert_array_equal(scores["proposal"], expected)
    assert scores["topology_permuted"].shape == (16, len(expected))
