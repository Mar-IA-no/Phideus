from __future__ import annotations

from pathlib import Path
import json
import sys

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from geometria_proporcional.wave59_hgb_guard_bracket import (  # noqa: E402
    HARM_CONTROL_SEEDS,
    INCOMPATIBILITY_CONTROL_SEEDS,
    apply_calibrated_policies,
    bootstrap_indices,
    calibrate_policies,
    control_family_evaluable,
    factorial_policy_ids,
    inference_safe_view,
    maximum_displacement_control,
    model_id,
    policy_id,
    primary_tokens,
    quantile_higher,
    shard_assignment,
    support,
    validate_inference_safe_view,
    validate_pre_draw_config,
    validate_primary_integrity,
)


def synthetic_inference_data(tokens: int = 12) -> dict[str, np.ndarray]:
    policies = 24
    primary = np.ones(tokens, dtype=bool)
    disagreement = np.ones((tokens, policies), dtype=bool)
    hard = np.zeros((tokens, policies), dtype=np.int64)
    posterior = np.ones((tokens, policies), dtype=np.int64)
    return {
        "pair_token": np.asarray([f"t-{index:03d}" for index in range(tokens)]),
        "primary": primary,
        "disagreement": disagreement,
        "hard_actions": hard,
        "posterior_actions": posterior,
        "gain": np.zeros((tokens, policies), dtype=np.float64),
        "weights": np.full((tokens, policies), 1.0 / policies),
        "design": np.zeros((tokens, policies, 17), dtype=np.float64),
    }


def synthetic_scores(tokens: int = 12) -> dict[str, np.ndarray]:
    base = np.arange(tokens * 24, dtype=np.float64).reshape(tokens, 24)
    base /= float(base.max())
    result = {
        model_id("ridge"): base,
        model_id("hgb"): base + 0.01,
    }
    for target_index, target in enumerate(("harm", "posterior_incompatibility")):
        for guard_index, guard in enumerate(("logistic", "hgb")):
            result[model_id(guard, target)] = (
                base[::-1] + 0.02 * target_index + 0.01 * guard_index
            )
    return result


def test_pre_draw_config_is_bound_and_rejects_drift() -> None:
    path = (
        REPO_ROOT
        / "experiments/geometria_proporcional/configs/wave59_fresh_hgb_guard_bracket.json"
    )
    config = json.loads(path.read_text(encoding="utf-8"))
    validate_pre_draw_config(config)
    config["runtime_budget"]["gpu_allowed"] = True
    with pytest.raises(RuntimeError, match="runtime"):
        validate_pre_draw_config(config)


def test_inference_safe_view_is_closed_and_rejects_truth() -> None:
    source = synthetic_inference_data()
    source["target"] = np.ones((12, 4), dtype=bool)
    safe = inference_safe_view(source)
    assert "target" not in safe
    validate_inference_safe_view(safe)
    safe["gain"] = np.zeros((12, 24), dtype=np.float64)
    with pytest.raises(ValueError, match="allowlist"):
        validate_inference_safe_view(safe)


def test_factorial_has_sixteen_unique_policy_ids_and_separate_model_ids() -> None:
    identifiers = factorial_policy_ids()
    assert len(identifiers) == 16
    assert len(set(identifiers)) == 16
    assert policy_id("hgb", "hgb", "harm", 0.7) in identifiers
    assert policy_id("hgb", "hgb", "posterior_incompatibility", 0.9) in identifiers
    assert model_id("hgb") == "proposer-hgb"
    assert model_id("hgb", "harm") == "guard-hgb-harm"
    with pytest.raises(ValueError):
        policy_id("hgb", "hgb", "harm", 0.8)


def test_primary_universe_excludes_nonprimary_tokens_without_failure() -> None:
    data = synthetic_inference_data()
    data["primary"][[1, 4, 9]] = False
    validate_primary_integrity(data)
    observed = primary_tokens(data)
    assert len(observed) == 9
    assert "t-001" not in observed


def test_integrity_requires_exactly_twenty_four_policy_rows() -> None:
    data = synthetic_inference_data()
    for key in ("disagreement", "hard_actions", "posterior_actions", "gain", "weights"):
        data[key] = data[key][:, :-1]
    with pytest.raises(ValueError, match="exactly 24"):
        validate_primary_integrity(data)


def test_support_keeps_row_and_token_units_distinct() -> None:
    mask = np.zeros((4, 24), dtype=bool)
    mask[0, :20] = True
    mask[2, :10] = True
    observed = support(mask, np.asarray([True, False, True, False]))
    assert observed == {"authorized_rows": 30, "authorized_pair_tokens": 2}


def test_quantile_higher_and_strict_policy_masks() -> None:
    data = synthetic_inference_data(tokens=10)
    scores = synthetic_scores(tokens=10)
    calibration, arrays = calibrate_policies(data, scores)
    assert quantile_higher(np.asarray([0.0, 1.0, 2.0, 3.0]), 0.5) == 2.0
    assert len(calibration["policies"]) == 16
    identifier = policy_id("hgb", "hgb", "harm", 0.7)
    authorized = arrays[f"authorized__{identifier}"]
    threshold = calibration["policies"][identifier]["guard_threshold"]
    proposal = arrays["proposal__hgb"]
    np.testing.assert_array_equal(
        authorized, proposal & (scores[model_id("hgb", "harm")] < threshold)
    )
    assert not np.any(
        authorized & (scores[model_id("hgb", "harm")] == threshold)
    )


def test_apply_uses_frozen_thresholds_on_new_split() -> None:
    validation = synthetic_inference_data(tokens=10)
    calibration, _ = calibrate_policies(validation, synthetic_scores(tokens=10))
    monitor = synthetic_inference_data(tokens=7)
    monitor_scores = synthetic_scores(tokens=7)
    arrays = apply_calibrated_policies(monitor, monitor_scores, calibration)
    identifier = policy_id("ridge", "logistic", "harm", 0.9)
    row = calibration["policies"][identifier]
    proposal = (
        monitor["primary"][:, None]
        & monitor["disagreement"]
        & (
            monitor_scores[model_id("ridge")]
            > calibration["proposers"]["ridge"]["threshold"]
        )
    )
    expected = proposal & (
        monitor_scores[model_id("logistic", "harm")] < row["guard_threshold"]
    )
    np.testing.assert_array_equal(arrays[f"authorized__{identifier}"], expected)


def test_maximum_displacement_preserves_prevalence_and_attains_maximum() -> None:
    tokens, policies = 20, 3
    labels = np.zeros((tokens, policies), dtype=np.int8)
    labels[:4] = 1
    active = np.ones_like(labels, dtype=bool)
    weights = np.full_like(labels, 1.0 / policies, dtype=np.float64)
    pair_tokens = np.asarray([f"c-{index:03d}" for index in range(tokens)])
    control = maximum_displacement_control(labels, active, weights, pair_tokens, 59031)
    for policy in range(policies):
        assert control["target"][:, policy].sum() == labels[:, policy].sum()
    diagnostic = control["diagnostics"]
    assert diagnostic["hamming_global_max"] == pytest.approx(0.4)
    assert diagnostic["hamming_global_attained"] == pytest.approx(0.4)
    assert diagnostic["attained_over_max"] == pytest.approx(1.0)
    replay = maximum_displacement_control(labels, active, weights, pair_tokens, 59031)
    np.testing.assert_array_equal(control["mapping"], replay["mapping"])
    np.testing.assert_array_equal(control["target"], replay["target"])


def test_control_family_requires_five_effective_targets() -> None:
    tokens, policies = 30, 2
    labels = np.zeros((tokens, policies), dtype=np.int8)
    labels[:5] = 1
    active = np.ones_like(labels, dtype=bool)
    weights = np.ones_like(labels, dtype=np.float64)
    pair_tokens = np.asarray([f"u-{index:03d}" for index in range(tokens)])
    controls = [
        maximum_displacement_control(labels, active, weights, pair_tokens, seed)
        for seed in HARM_CONTROL_SEEDS
    ]
    evaluable, failures = control_family_evaluable(controls, "harm")
    assert evaluable is True
    assert failures == []

    balanced = labels.copy()
    balanced[:15] = 1
    repeated_targets = [
        maximum_displacement_control(balanced, active, weights, pair_tokens, seed)
        for seed in INCOMPATIBILITY_CONTROL_SEEDS
    ]
    evaluable, failures = control_family_evaluable(
        repeated_targets, "posterior_incompatibility"
    )
    assert evaluable is False
    assert "target_identity" in failures


def test_shards_and_bootstrap_are_deterministic_and_token_paired() -> None:
    tokens = np.asarray(["a", "b", "c", "d"])
    first = shard_assignment(tokens)
    second = shard_assignment(tokens)
    np.testing.assert_array_equal(first, second)
    assert set(first.tolist()) <= {0, 1}
    indices = bootstrap_indices(tokens, replicates=11)
    np.testing.assert_array_equal(indices, bootstrap_indices(tokens, replicates=11))
    assert indices.shape == (11, 4)
    with pytest.raises(ValueError, match="unique"):
        bootstrap_indices(np.asarray(["a", "a"]), replicates=2)
