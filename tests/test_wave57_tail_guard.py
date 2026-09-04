from __future__ import annotations

import copy

import numpy as np
import pytest
from scipy.special import expit

from geometria_proporcional.wave57_tail_guard import (
    EXPECTED_SKLEARN_VERSION,
    HARM_MODEL_CONTRACT,
    apply_harm_guard,
    conditional_harm_shuffle,
    fit_harm_logistic,
    harm_labels,
    per_set_support,
    proposal_mask,
    score_harm_logistic,
    token_support,
    validate_harm_model_contract,
)


def synthetic_fit(seed: int = 57) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.Generator(np.random.PCG64(seed))
    x = rng.normal(size=(240, 17))
    logits = 0.8 * x[:, 0] - 0.5 * x[:, 3] + 0.2 * x[:, 8]
    y = (rng.random(len(x)) < expit(logits)).astype(np.int8)
    weights = rng.uniform(0.1, 1.0, size=len(x))
    return x, y, weights


def test_harm_label_direction_is_negative_gain() -> None:
    gain = np.asarray([-1.0, -1e-11, -1e-12, 0.0, 2.0])
    np.testing.assert_array_equal(harm_labels(gain), [1, 1, 0, 0, 0])


def test_harm_logistic_contract_and_state_reconstruct_probability() -> None:
    x, y, weights = synthetic_fit()
    state = fit_harm_logistic(x, y, weights)
    probability = score_harm_logistic(state, x)
    assert state["classes"] == [0, 1]
    assert state["contract"] == HARM_MODEL_CONTRACT
    assert state["sklearn_version"] == EXPECTED_SKLEARN_VERSION
    assert probability.shape == (len(x),)
    assert np.all((probability > 0.0) & (probability < 1.0))
    assert np.corrcoef(probability, y)[0, 1] > 0.2


def test_harm_logistic_rejects_contract_drift_and_one_class() -> None:
    x, y, weights = synthetic_fit()
    drift = copy.deepcopy(HARM_MODEL_CONTRACT)
    drift["tol"] = 1e-8
    with pytest.raises(ValueError, match="contract drifted"):
        validate_harm_model_contract(drift)
    with pytest.raises(RuntimeError, match="both classes"):
        fit_harm_logistic(x, np.zeros_like(y), weights)


def test_score_rejects_wrong_positive_class() -> None:
    x, y, weights = synthetic_fit()
    state = fit_harm_logistic(x, y, weights)
    state["classes"] = [1, 0]
    with pytest.raises(ValueError, match="invalid harm-model state"):
        score_harm_logistic(state, x)


def shuffle_fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Eight tokens and three policies create both mixed and homogeneous strata.
    disagreement = np.asarray(
        [
            [1, 1, 0],
            [1, 0, 1],
            [1, 1, 0],
            [1, 0, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 0],
            [1, 0, 1],
        ],
        dtype=bool,
    )
    labels = np.asarray(
        [
            [0, 1, 0],
            [1, 0, 1],
            [1, 1, 0],
            [0, 0, 1],
            [1, 1, 1],
            [0, 0, 0],
            [0, 1, 0],
            [1, 0, 0],
        ],
        dtype=np.int8,
    )
    counts = disagreement.sum(axis=1)
    weights = np.zeros(disagreement.shape, dtype=np.float64)
    weights[disagreement] = np.repeat(1.0 / counts, counts)
    return labels, disagreement, weights


def test_conditional_shuffle_is_exact_deterministic_and_diagnostic() -> None:
    labels, disagreement, weights = shuffle_fixture()
    left = conditional_harm_shuffle(labels, disagreement, weights, 57031)
    right = conditional_harm_shuffle(labels, disagreement, weights, 57031)
    np.testing.assert_array_equal(left["target"], right["target"])
    np.testing.assert_array_equal(left["mapping"], right["mapping"])
    assert left["diagnostics"] == right["diagnostics"]

    counts = disagreement.sum(axis=1)
    for policy in range(disagreement.shape[1]):
        for count in np.unique(counts[disagreement[:, policy]]):
            rows = disagreement[:, policy] & (counts == count)
            assert labels[rows, policy].sum() == left["target"][rows, policy].sum()
    diagnostics = left["diagnostics"]
    assert diagnostics["mixed_strata"] > 0
    assert diagnostics["homogeneous_strata"] > 0
    assert 0.0 <= diagnostics["hamming_global"] <= 1.0
    assert 0.0 <= diagnostics["hamming_weighted"] <= 1.0
    assert len(diagnostics["mapping_sha256"]) == 64
    assert len(diagnostics["target_sha256"]) == 64


def test_guard_uses_fixed_proposals_and_strict_thresholds() -> None:
    mean = np.asarray([[0.2, 0.5, 0.8], [0.1, 0.7, 0.9]])
    disagreement = np.ones_like(mean, dtype=bool)
    proposals = proposal_mask(mean, disagreement, 0.5)
    np.testing.assert_array_equal(proposals, [[False, False, True], [False, True, True]])
    probability = np.asarray([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    hard = np.zeros_like(mean, dtype=np.int64)
    posterior = np.ones_like(hard)
    result = apply_harm_guard(proposals, probability, hard, posterior, 0.5)
    # Probability equal to the threshold is not authorized.
    np.testing.assert_array_equal(result["authorized"], [[False, False, True], [False, False, False]])
    np.testing.assert_array_equal(result["actions"], result["authorized"].astype(np.int64))
    assert token_support(result["authorized"], np.asarray([True, True])) == {
        "rows": 1,
        "tokens": 1,
    }


def test_hard_only_is_exact_identity() -> None:
    mean = np.ones((3, 4))
    disagreement = np.ones((3, 4), dtype=bool)
    proposals = proposal_mask(mean, disagreement, "hard_only")
    assert not proposals.any()
    hard = np.arange(12).reshape(3, 4) % 4
    posterior = (hard + 1) % 4
    result = apply_harm_guard(disagreement, np.zeros((3, 4)), hard, posterior, "hard_only")
    np.testing.assert_array_equal(result["actions"], hard)
    assert not result["authorized"].any()


def test_support_union_cannot_authorize_individual_sets() -> None:
    set_index = np.repeat(np.asarray([0, 4, 8, 10, 12]), 10)
    support = per_set_support(
        set_index, np.ones(len(set_index), dtype=bool), [0, 4, 8, 10, 12], 30
    )
    assert len(set_index) == 50
    assert all(row["tokens"] == 10 for row in support.values())
    assert all(row["status"] == "NOT_EVALUABLE" for row in support.values())
