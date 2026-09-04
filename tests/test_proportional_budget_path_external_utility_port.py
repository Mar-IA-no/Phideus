from __future__ import annotations

import copy
import importlib.util
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
SCRIPT = (
    ROOT
    / "experiments/geometria_proporcional/run_proportional_budget_path_external_utility_port.py"
)
SPEC = importlib.util.spec_from_file_location("budget_path_utility_runner", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

from geometria_proporcional.budget_path_schema import canonical_json  # noqa: E402
from geometria_proporcional.budget_path_utility import (  # noqa: E402
    ExternalUtilityViolation,
    evaluate_external_utility,
    make_declaration,
)


def fixtures():
    return {
        "tradeoff": MODULE.build_fixture("tradeoff", MODULE.TRADEOFF_COORDINATES),
        "singleton": MODULE.build_fixture("singleton", MODULE.SINGLETON_COORDINATES),
    }


def test_config_and_source_are_cpu_frozen():
    cfg = MODULE.load_config(MODULE.DEFAULT_CONFIG)
    MODULE.verify_source(cfg)
    assert cfg["expected_decisions"] == 7
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""


def test_seven_positive_cases_match_declared_sets_and_abstention():
    cases, declarations, decisions = MODULE.execute_positive(fixtures())
    assert len(cases) == len(declarations) == len(decisions) == 7
    assert decisions[1]["selected_policy_ids"] == [
        "budget_0.10",
        "budget_0.20",
        "budget_0.40",
    ]
    assert decisions[5]["selected_policy_ids"] == []
    assert decisions[5]["decision_status"] == "ABSTAIN_NO_FEASIBLE_POLICY"


def test_evaluation_does_not_mutate_artifact_or_receipt():
    artifact, receipt = fixtures()["tradeoff"]
    declaration = make_declaration(
        artifact=artifact,
        checker_receipt=receipt,
        utility_kind="LEXICOGRAPHIC_MINIMIZE",
        parameters={"axis_priority": ["iid_delta", "grouped_delta"]},
    )
    before = canonical_json({"artifact": artifact, "receipt": receipt})
    decision = evaluate_external_utility(artifact, receipt, declaration)
    after = canonical_json({"artifact": artifact, "receipt": receipt})
    assert before == after
    assert set(decision["selected_policy_ids"]) <= set(decision["candidate_policy_ids"])


def test_reader_order_and_dominated_decoy_are_metamorphic_invariants():
    checks = MODULE.metamorphic_checks(fixtures())
    assert len(checks) == 3
    assert {row["status"] for row in checks} == {"PASS"}


def test_twelve_invalid_compositions_are_rejected():
    artifact, receipt = fixtures()["tradeoff"]
    results = MODULE.invalid_suite(artifact, receipt)
    assert len(results) == 12
    assert {row["status"] for row in results} == {"REJECTED"}


@pytest.mark.parametrize(
    "weights, expected",
    [
        ({"iid_delta": -0.1, "grouped_delta": 1.1}, "strictly positive"),
        ({"iid_delta": 0.6, "grouped_delta": 0.6}, "sum to one"),
    ],
)
def test_weight_contract_rejects_negative_and_unnormalized(weights, expected):
    artifact, receipt = fixtures()["tradeoff"]
    declaration = make_declaration(
        artifact=artifact,
        checker_receipt=receipt,
        utility_kind="WEIGHTED_SUM_MINIMIZE",
        parameters={"weights": weights},
    )
    with pytest.raises(ExternalUtilityViolation, match=expected):
        evaluate_external_utility(artifact, receipt, declaration)


def test_user_scope_cannot_be_fabricated():
    artifact, receipt = fixtures()["tradeoff"]
    declaration = make_declaration(
        artifact=artifact,
        checker_receipt=receipt,
        utility_kind="WEIGHTED_SUM_MINIMIZE",
        parameters={"weights": {"iid_delta": 0.5, "grouped_delta": 0.5}},
    )
    changed = copy.deepcopy(declaration)
    changed["scope"] = "USER_DECLARED"
    MODULE.reidentify(changed)
    with pytest.raises(ExternalUtilityViolation, match="SYNTHETIC_FIXTURE"):
        evaluate_external_utility(artifact, receipt, changed)


def test_historical_authority_cannot_be_disguised_with_synthetic_receipt():
    artifact, _ = fixtures()["tradeoff"]
    artifact["lineage"]["scope"] = "OPENED_POSTHOC"
    artifact["authority"]["artifact_status"] = "CHECKABLE_CANDIDATE"
    receipt = MODULE.forged_receipt(artifact)
    declaration = make_declaration(
        artifact=artifact,
        checker_receipt=receipt,
        utility_kind="WEIGHTED_SUM_MINIMIZE",
        parameters={"weights": {"iid_delta": 0.5, "grouped_delta": 0.5}},
    )
    with pytest.raises(ExternalUtilityViolation, match="lineage must be synthetic"):
        evaluate_external_utility(artifact, receipt, declaration)


def test_axis_semantics_are_checked_at_the_port():
    artifact, _ = fixtures()["tradeoff"]
    artifact["axes"][0]["population"] = "grouped"
    receipt = MODULE.forged_receipt(artifact)
    declaration = make_declaration(
        artifact=artifact,
        checker_receipt=receipt,
        utility_kind="WEIGHTED_SUM_MINIMIZE",
        parameters={"weights": {"iid_delta": 0.5, "grouped_delta": 0.5}},
    )
    with pytest.raises(ExternalUtilityViolation, match="axes mismatch"):
        evaluate_external_utility(artifact, receipt, declaration)


def test_synthetic_fixture_has_full_budget_path_surface():
    artifact, receipt = fixtures()["tradeoff"]
    assert len(artifact["policies"]) == 7
    assert len(artifact["pairs"]) == 21
    assert receipt["scope"] == "SYNTHETIC_FIXTURE"
    assert artifact["authority"]["empirical_claim_status"] == "NOT_APPLICABLE"
