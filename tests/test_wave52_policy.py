from itertools import permutations
import importlib.util
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest
import torch

from geometria_proporcional.wave52_policy import (
    ContextualPolicyDeepSet,
    SharedFamilyReader,
    authorized_actions,
    constrained_regret,
    explicit_set_actions,
    policy_fold,
    score_composition_actions,
    validate_policy_groups,
)

GROUPS = (
    (0, 1, 6, 7, 16, 17, 22, 23),
    (2, 3, 10, 11, 12, 13, 20, 21),
    (4, 5, 8, 9, 14, 15, 18, 19),
)


def load_runner():
    path = (
        Path(__file__).parents[1]
        / "experiments/geometria_proporcional/run_wave52_policy_transport.py"
    )
    spec = importlib.util.spec_from_file_location("wave52_runner_for_tests", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def policies():
    return np.asarray(list(permutations(range(4))), dtype=np.int64)


def test_balanced_policy_groups_and_rotation_cover_all_policies():
    groups = validate_policy_groups(policies(), GROUPS)
    shift = [
        index
        for fold in range(3)
        for index in policy_fold(groups, fold)["policy_shift"]
    ]
    assert sorted(shift) == list(range(24))


def test_invalid_policy_partition_fails_closed():
    bad = (tuple(range(8)), tuple(range(8, 16)), tuple(range(16, 24)))
    with pytest.raises(ValueError, match="every rank twice"):
        validate_policy_groups(policies(), bad)


def test_authorized_action_is_inside_target_and_responds_to_utility():
    targets = np.array([[1, 1, 0, 0], [0, 1, 0, 1]], dtype=bool)
    utilities = np.array([[0.0, 1.0, 2.0, 3.0], [3.0, 2.0, 1.0, 0.0]])
    actions = authorized_actions(targets, utilities)
    assert actions.tolist() == [[1, 0], [3, 1]]
    assert np.all(targets[np.arange(2)[:, None], actions])


def test_authorized_action_rejects_empty_set():
    with pytest.raises(ValueError, match="non-empty"):
        authorized_actions(np.zeros((1, 4), dtype=bool), np.ones((1, 4)))


def test_explicit_policy_uses_predicted_set_and_probability_fallback():
    probability = np.array([[0.8, 0.7, 0.1, 0.2], [0.4, 0.3, 0.2, 0.1]])
    utility = np.array([[0.0, 1.0, 2.0, 3.0], [3.0, 2.0, 1.0, 0.0]])
    actions, fallback = explicit_set_actions(probability, utility, tau=0.5)
    assert actions.tolist() == [[1, 0], [0, 0]]
    assert fallback.tolist() == [False, True]


def test_constrained_regret_penalizes_invalid_more_than_any_valid_action():
    targets = np.array([[1, 1, 0, 0]], dtype=bool)
    utilities = np.array([[1.0, 0.6, 0.2, -0.2]])
    actions = np.array([[2]])
    assert constrained_regret(actions, targets, utilities).item() == 1.25
    valid = constrained_regret(np.array([[1]]), targets, utilities).item()
    assert 0.0 <= valid <= 1.0


def test_score_composition_prefers_utility_when_weight_is_large():
    logits = np.zeros((1, 4))
    utilities = np.array([[0.0, 1.0, 2.0, 3.0]])
    assert score_composition_actions(logits, utilities, 10.0).item() == 3


def test_shared_reader_is_jointly_permutation_equivariant():
    torch.manual_seed(7)
    reader = SharedFamilyReader()
    evidence = torch.randn(3, 4)
    utility = torch.randn(5, 4)
    permutation = torch.tensor([2, 0, 3, 1])
    original = reader(evidence, utility)
    permuted = reader(evidence[:, permutation], utility[:, permutation])
    torch.testing.assert_close(permuted, original[:, :, permutation])


def test_contextual_deepset_is_point_permutation_invariant():
    torch.manual_seed(11)
    model = ContextualPolicyDeepSet()
    points = torch.randn(2, 7, 6)
    mask = torch.ones(2, 7, dtype=torch.bool)
    utility = torch.randn(3, 4)
    expected = model(points, mask, utility)
    order = torch.tensor([4, 1, 6, 0, 2, 5, 3])
    actual = model(points[:, order], mask[:, order], utility)
    torch.testing.assert_close(actual[0], expected[0], atol=1e-6, rtol=0)
    torch.testing.assert_close(actual[1], expected[1], atol=1e-6, rtol=0)


def test_worst_regret_aggregates_as_max_across_policy_folds():
    runner = load_runner()
    store = {"arm": defaultdict(list)}
    base = {
        "pair_token": "token-a",
        "cluster_id": "token-a",
        "design_stratum": "NEAR_RIVAL",
        "cardinality": 2,
        "choice_valid": True,
        "n_policies": 8,
        "action_accuracy": 0.5,
        "compatible_action_rate": 1.0,
        "restricted_regret": 0.2,
    }
    store["arm"]["token-a"].append({**base, "worst_restricted_regret": 0.4})
    store["arm"]["token-a"].append({**base, "worst_restricted_regret": 1.25})
    rows, _ = runner.aggregate_metric_store(store)
    assert rows["arm"][0]["worst_restricted_regret"] == 1.25


def test_counterfactual_intervention_does_not_require_correct_original_action():
    runner = load_runner()
    groups = [
        {
            "pair_token": "token-a",
            "cluster_id": "token-a",
            "design_stratum": "NEAR_RIVAL",
            "cardinality": 2,
            "target": np.array([1, 1, 0, 0], dtype=np.float32),
        }
    ]
    utilities = np.array([[1.0, 0.6, 0.2, -0.2], [0.6, 1.0, 0.2, -0.2]])
    actions = np.array([[2, 1]])
    result, mapping = runner.counterfactual_success(groups, utilities, actions)
    assert result["success_rate"] == 0.5
    assert result["joint_correct_rate"] == 0.0
    assert mapping[0]["intervention_success"] == 1.0
    assert mapping[0]["joint_correct_success"] == 0.0


def test_empty_target_is_reported_without_authorized_choice_metrics():
    runner = load_runner()
    groups = [
        {
            "pair_token": "ooc",
            "cluster_id": "ooc",
            "design_stratum": "OUT_OF_CATALOG",
            "cardinality": 0,
            "target": np.zeros(4, dtype=np.float32),
        }
    ]
    rows = runner.policy_metrics_by_token(
        groups,
        np.array([[1.0, 0.6, 0.2, -0.2]]),
        np.array([[0]]),
        1.25,
    )
    assert rows[0]["choice_valid"] is False
    assert np.isnan(rows[0]["action_accuracy"])
    assert np.isnan(rows[0]["restricted_regret"])


def test_bootstrap_resamples_declared_clusters_not_context_rows():
    runner = load_runner()
    common = {"design_stratum": "NEAR_RIVAL", "cardinality": 2}
    rows_a = [
        {"pair_token": "a", "cluster_id": "shared", "score": 1.0, **common},
        {"pair_token": "b", "cluster_id": "shared", "score": 0.0, **common},
    ]
    rows_b = [
        {"pair_token": "a", "cluster_id": "shared", "score": 0.0, **common},
        {"pair_token": "b", "cluster_id": "shared", "score": 0.0, **common},
    ]
    result = runner.bootstrap_delta(rows_a, rows_b, "score", 20, 7)
    assert result["n_pair_tokens"] == 2
    assert result["n_clusters"] == 1
    assert result["mean_diff"] == 0.5
