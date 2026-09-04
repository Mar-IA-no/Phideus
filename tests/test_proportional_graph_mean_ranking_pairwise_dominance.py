from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "experiments/geometria_proporcional/run_proportional_graph_mean_ranking_pairwise_dominance.py"
)
SPEC = importlib.util.spec_from_file_location("mean_pairwise_dominance", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_config_sources_and_environment_are_frozen_cpu():
    cfg = MODULE.load_config(MODULE.DEFAULT_CONFIG)
    pareto_cfg = MODULE.verify_sources(cfg)
    assert (
        pareto_cfg["source_attribution_manifest_sha256"]
        == cfg["source_attribution_manifest_sha256"]
    )
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""


def test_pairwise_statuses_are_exclusive_and_directional():
    tol = 1e-12
    assert MODULE.classify_one(np.asarray([-1.0, -2.0]), tol) == "EXPANSION_DOMINATES"
    assert MODULE.classify_one(np.asarray([1.0, 2.0]), tol) == "BASE_DOMINATES"
    assert MODULE.classify_one(np.asarray([1.0, -2.0]), tol) == "IID_COST_GROUPED_GAIN"
    assert MODULE.classify_one(np.asarray([-1.0, 2.0]), tol) == "IID_GAIN_GROUPED_COST"
    assert MODULE.classify_one(np.asarray([0.0, 0.0]), tol) == "EQUIVALENT"


def test_pairwise_records_preserve_all_pairs_and_bootstrap_frequencies():
    values = np.zeros((7, 2, 4))
    values[1:, 0] = -1.0
    values[1:, 1] = -2.0
    bootstrap = np.tile(np.arange(4), (5, 1))
    records, increments = MODULE.pairwise_records(values, bootstrap, 1e-12)
    assert len(records) == 21
    assert len(increments) == 21
    first = records[0]
    assert first["pair_id"] == "identity->budget_0.01"
    assert first["point_status"] == "EXPANSION_DOMINATES"
    assert first["bootstrap_status_frequency"]["EXPANSION_DOMINATES"] == 1.0
    assert sum(first["bootstrap_status_frequency"].values()) == 1.0


def test_nested_action_check_rejects_changed_nonzero_action():
    cfg = MODULE.load_config(MODULE.DEFAULT_CONFIG)
    actions = {}
    fractions = (0.01, 0.02, 0.05, 0.10, 0.20, 0.40)
    for regime in cfg["proposal_regimes"]:
        for arm in cfg["arms"]:
            for family in MODULE.MAIN:
                for fraction in fractions:
                    actions[
                        MODULE.action_key(
                            regime, arm, fraction, "policy_selection", family
                        )
                    ] = np.zeros(3, dtype=np.int64)
            for fraction in fractions:
                actions[
                    MODULE.action_key(
                        regime, arm, fraction, "policy_selection", MODULE.CONTROL
                    )
                ] = np.zeros((16, 3), dtype=np.int64)
    key_small = MODULE.action_key(
        cfg["proposal_regimes"][0],
        cfg["arms"][0],
        0.01,
        "policy_selection",
        MODULE.MAIN[0],
    )
    key_large = MODULE.action_key(
        cfg["proposal_regimes"][0],
        cfg["arms"][0],
        0.02,
        "policy_selection",
        MODULE.MAIN[0],
    )
    actions[key_small][0] = 1
    actions[key_large][0] = 2
    try:
        MODULE.assert_nested_actions(cfg, "policy_selection", actions)
    except AssertionError as exc:
        assert "non-nested ranked path" in str(exc)
    else:
        raise AssertionError("changed nonzero action should violate nestedness")


def test_analysis_assembles_policy_selection_and_adjudication_roles():
    cfg = MODULE.load_config(MODULE.DEFAULT_CONFIG)

    def pair(role: str):
        return {
            "pair_id": "identity->budget_0.01",
            "base_policy": "identity",
            "expansion_policy": "budget_0.01",
            "adjacent": True,
            "mean_increment": {"iid": -1.0, "grouped": -1.0},
            "point_status": "EXPANSION_DOMINATES",
            "bootstrap_status_frequency": {
                state: float(state == "EXPANSION_DOMINATES") for state in MODULE.STATES
            },
            "role": role,
        }

    reports = {cohort: {"proposals": {}} for cohort in ("A", "B")}
    for cohort in reports:
        for regime in cfg["proposal_regimes"]:
            reports[cohort]["proposals"][regime] = {"arms": {}}
            for arm in cfg["arms"]:
                roles = {}
                for role in ("policy_selection", "adjudication"):
                    roles[role] = {
                        "families": {"topology_mean": [pair(role)]},
                        MODULE.CONTROL: [
                            {"replicate": replicate, "pairs": [pair(role)]}
                            for replicate in range(cfg["control_replicates"])
                        ],
                    }
                reports[cohort]["proposals"][regime]["arms"][arm] = {"roles": roles}
    analysis = MODULE.build_analysis(cfg, reports)
    fixed = analysis["proposals"][cfg["primary_proposal"]]
    assert (
        fixed["aggregate"]["topology_transition"]["EXPANSION_DOMINATES"][
            "EXPANSION_DOMINATES"
        ]
        == 8
    )
    assert set(fixed["arms"][cfg["arms"][0]]["A"]) == {
        "policy_selection",
        "adjudication",
        "transport",
    }
