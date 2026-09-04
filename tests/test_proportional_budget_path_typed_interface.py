from __future__ import annotations

import copy
import importlib.util
import inspect
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
SCRIPT = (
    ROOT
    / "experiments/geometria_proporcional/run_proportional_budget_path_typed_interface.py"
)
SPEC = importlib.util.spec_from_file_location("budget_path_runner", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

from geometria_proporcional import budget_path_checker as checker_module  # noqa: E402
from geometria_proporcional.budget_path_checker import (  # noqa: E402
    BudgetPathChecker,
    BudgetPathViolation,
    validate_nested_actions,
)
from geometria_proporcional.budget_path_schema import (  # noqa: E402
    BudgetPathArtifact,
)


def representative_artifact():
    cfg = MODULE.load_config(MODULE.DEFAULT_CONFIG)
    cohort = "A"
    attribution_root = ROOT / cfg["source_mean_ranking_attribution"]
    pareto_root = ROOT / cfg["source_pareto_transport"]
    pairwise_root = ROOT / cfg["source_pairwise_dominance"]
    actions = MODULE.read_npz(attribution_root / "cohort_a_selection.npz")
    objectives = MODULE.read_npz(pareto_root / "cohort_a_objectives.npz")
    pareto_report = MODULE.json.loads(
        (pareto_root / "cohort_a_pareto.json").read_text()
    )
    pairwise_report = MODULE.json.loads(
        (pairwise_root / "cohort_a_pairwise.json").read_text()
    )
    artifact = MODULE.build_artifact(
        cfg,
        cohort,
        cfg["proposal_regimes"][0],
        cfg["arms"][0],
        "policy_selection",
        "topology_mean",
        None,
        actions,
        objectives,
        pareto_report,
        pairwise_report,
    )
    return cfg, artifact


def test_config_sources_and_checker_are_cpu_frozen():
    cfg = MODULE.load_config(MODULE.DEFAULT_CONFIG)
    checker = BudgetPathChecker(ROOT, cfg)
    assert checker.config["expected_artifacts"] == 608
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""


def test_checker_reconstructs_representative_artifact():
    cfg, artifact = representative_artifact()
    receipt = BudgetPathChecker(ROOT, cfg).check(artifact)
    assert receipt["checker_status"] == "VALID"
    assert receipt["policy_count"] == 7
    assert receipt["pair_count"] == 21
    assert receipt["decision_status"] == "UNRESOLVED"


def test_schema_rejects_selected_policy_injection():
    _, artifact = representative_artifact()
    artifact["selected_policy_id"] = "budget_0.40"
    with pytest.raises(ValueError, match="artifact schema mismatch"):
        BudgetPathArtifact.from_dict(artifact)


def test_checker_rejects_boolean_disguised_as_numeric_fraction():
    cfg, artifact = representative_artifact()
    artifact["policies"][0]["budget_fraction"] = False
    with pytest.raises(BudgetPathViolation, match="budget fraction mismatch"):
        BudgetPathChecker(ROOT, cfg).check(artifact)


def test_nested_action_checker_rejects_alpha_change():
    actions = [np.zeros(4, dtype=np.int64) for _ in range(7)]
    actions[1][0] = 1
    actions[2][0] = 2
    with pytest.raises(BudgetPathViolation, match="not nested"):
        validate_nested_actions(actions)


def test_eight_frozen_mutations_are_rejected():
    cfg, artifact = representative_artifact()
    results = MODULE.mutation_suite(BudgetPathChecker(ROOT, cfg), artifact)
    assert len(results) == 8
    assert {row["status"] for row in results} == {"REJECTED"}


def test_materialization_has_unique_expected_paths_and_no_decision_field():
    cfg = MODULE.load_config(MODULE.DEFAULT_CONFIG)
    artifacts = MODULE.materialize(cfg)
    assert len(artifacts) == cfg["expected_artifacts"]
    assert len({row["artifact_id"] for row in artifacts}) == len(artifacts)
    keys = set().union(*(MODULE.recursive_keys(row) for row in artifacts))
    assert (
        not {"selected_policy_id", "utility_weight", "scalar_score", "recommendation"}
        & keys
    )
    assert {row["lineage"]["family"] for row in artifacts} == {
        *cfg["main_families"],
        cfg["control_family"],
    }


def test_checker_does_not_import_builder_or_prior_analysis_logic():
    source = inspect.getsource(checker_module)
    assert "run_proportional_budget_path_typed_interface" not in source
    assert "run_proportional_graph_mean_ranking_pareto_transport" not in source
    assert "run_proportional_graph_mean_ranking_pairwise_dominance" not in source
