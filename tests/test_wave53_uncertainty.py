import importlib.util
import json
from pathlib import Path

import numpy as np

from geometria_proporcional.wave53_uncertainty import (
    apply_coverage_boundary,
    coverage_boundary,
    deranged_within_strata,
    discrete_aurc,
    expected_regret_actions,
    independent_nonempty_mass,
    nonempty_sets,
    ordinal_loss_tensor,
    paired_bootstrap_indices,
    paired_delta_ci,
    stratified_token_split,
)


def load_runner():
    path = (
        Path(__file__).parents[1]
        / "experiments/geometria_proporcional/run_wave53_uncertainty_policy.py"
    )
    spec = importlib.util.spec_from_file_location("wave53_runner_for_tests", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_nonempty_product_mass_is_normalized_and_matches_manual_case():
    sets, mass = independent_nonempty_mass(np.array([[0.5, 0.5]]))
    assert sets.tolist() == [[True, False], [False, True], [True, True]]
    np.testing.assert_allclose(mass, [[1 / 3, 1 / 3, 1 / 3]])


def test_nonempty_product_mass_remains_finite_at_probability_extremes():
    _, mass = independent_nonempty_mass(
        np.array([[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]])
    )
    assert np.all(np.isfinite(mass))
    np.testing.assert_allclose(mass.sum(axis=1), 1.0)


def test_ordinal_loss_matches_wave52_contract():
    sets = nonempty_sets(2)
    utility = np.array([[1.0, -0.2]])
    loss = ordinal_loss_tensor(sets, utility, incompatible_penalty=1.25)
    np.testing.assert_allclose(loss[0, :, 0], [0.0, 1.25])
    np.testing.assert_allclose(loss[0, :, 1], [1.25, 0.0])
    np.testing.assert_allclose(loss[0, :, 2], [0.0, 1.0])


def test_expected_regret_prefers_high_utility_when_both_are_likely():
    result = expected_regret_actions(np.array([[0.9, 0.9]]), np.array([[1.0, -0.2]]))
    assert result["actions"].item() == 0
    np.testing.assert_allclose(result["set_mass"].sum(axis=1), 1.0)
    assert result["margin"].item() > 0.0


def test_stratified_split_is_disjoint_complete_and_deterministic():
    tokens = [f"t{i}" for i in range(12)]
    strata = [("near", i % 2) for i in range(12)]
    left, right = stratified_token_split(tokens, strata, 0.5, 17)
    left2, right2 = stratified_token_split(tokens, strata, 0.5, 17)
    assert set(left).isdisjoint(right)
    assert sorted(np.concatenate([left, right]).tolist()) == list(range(12))
    np.testing.assert_array_equal(left, left2)
    np.testing.assert_array_equal(right, right2)


def test_stratified_split_rejects_singleton_stratum():
    try:
        stratified_token_split(["a", "b", "c"], [("x", 1), ("x", 1), ("y", 2)], 0.5, 17)
    except ValueError as exc:
        assert "singleton" in str(exc)
    else:
        raise AssertionError("singleton stratum must fail closed")


def test_derangement_stays_in_stratum_and_has_no_fixed_points():
    tokens = [f"t{i}" for i in range(8)]
    strata = [("a", 1)] * 4 + [("b", 2)] * 4
    mapping = deranged_within_strata(tokens, strata, 29)
    assert np.all(mapping != np.arange(8))
    assert all(strata[i] == strata[j] for i, j in enumerate(mapping))


def test_coverage_boundary_uses_one_token_mask_for_all_policy_views():
    tokens = ["a", "b", "c", "d"]
    scores = np.array([0.4, 0.1, 0.3, 0.2])
    boundary = coverage_boundary(scores, tokens, 0.5, 31)
    mask = apply_coverage_boundary(scores, tokens, boundary, 31)
    assert mask.tolist() == [False, True, False, True]
    assert boundary["selected_count"] == 2


def test_discrete_aurc_orders_low_predicted_risk_first():
    score = np.array([0.1, 0.2, 0.3])
    risk = np.array([0.0, 0.3, 0.6])
    expected = np.mean([0.0, 0.15, 0.3])
    assert discrete_aurc(score, risk, ["a", "b", "c"]) == expected


def test_paired_bootstrap_reuses_token_indices():
    indices = paired_bootstrap_indices(4, 100, 37)
    result = paired_delta_ci(
        np.array([1.0, 1.0, 1.0, 1.0]),
        np.array([0.0, 0.0, 0.0, 0.0]),
        indices,
    )
    assert result["mean_diff"] == 1.0
    assert result["ci95_low"] == 1.0
    assert result["ci95_high"] == 1.0


def test_selective_coverage_uses_eligible_population_as_denominator():
    runner = load_runner()
    metric = {
        name: np.array([0.0, 1.0, 2.0, 3.0])
        for name in ("accuracy", "compatible", "regret", "worst_regret")
    }
    eligible = np.array([True, True, True, False])
    accepted = np.array([True, False, True, False])
    result = runner.selective_metrics(metric, accepted, eligible)
    assert result["coverage"] == 2 / 3
    assert result["regret"] == 1.0


def test_metadata_loader_prefilters_tokens_and_keeps_only_canonical_rows(tmp_path):
    runner = load_runner()
    rows = [
        {
            "pair_token": "selected",
            "calibration_population": "origin_translation_break",
            "design_stratum": "wrong",
            "oracle_compatible_set": ["a"],
        },
        {
            "pair_token": "selected",
            "calibration_population": "canonical_preserving",
            "design_stratum": "NEAR_RIVAL",
            "oracle_compatible_set": ["a", "b"],
        },
    ]
    path = tmp_path / "labels.jsonl"
    path.write_text(
        "\n".join([*(json.dumps(row) for row in rows), '{"pair_token":"later", bad'])
        + "\n",
        encoding="utf-8",
    )
    assert runner.load_metadata(path, {"selected"}) == {
        "selected": {"design_stratum": "NEAR_RIVAL", "cardinality": 2}
    }


def test_calibration_metrics_serializes_constant_residuals_without_nan():
    runner = load_runner()
    target = np.zeros((4, 4), dtype=float)
    probability = np.full((4, 4), 0.25)
    result = runner.calibration_metrics(probability, target)
    assert result["residual_membership_correlation"] == [[None] * 4 for _ in range(4)]
    assert result["max_abs_offdiagonal_residual_correlation"] is None
    json.dumps(result, allow_nan=False)
