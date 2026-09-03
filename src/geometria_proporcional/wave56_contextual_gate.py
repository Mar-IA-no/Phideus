"""Low-capacity contextual residual-gate primitives for Wave 56."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from scipy.special import expit

from .wave52_policy import constrained_regret
from .wave53_uncertainty import nonempty_sets


FEATURE_NAMES = (
    "advantage",
    "hard_risk",
    "minimum_risk",
    "action_risk_margin",
    "posterior_entropy_norm",
    "posterior_top_mass",
    "posterior_top_margin",
    "hard_cardinality",
    "posterior_expected_cardinality",
    "posterior_cardinality_variance",
    "posterior_mass_hard_set",
    "seed_std_mean",
    "seed_std_max",
    "utility_f0",
    "utility_f1",
    "utility_f2",
    "utility_f3",
)


@dataclass(frozen=True)
class WeightedScaler:
    mean: np.ndarray
    scale: np.ndarray

    def transform(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        return (values - self.mean) / self.scale


def fit_weighted_scaler(values: np.ndarray, weights: np.ndarray) -> WeightedScaler:
    """Fit a deterministic weighted standardizer; constants receive scale one."""
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if values.ndim != 2 or weights.shape != (len(values),):
        raise ValueError("values/weights shape mismatch")
    if len(values) == 0 or np.any(weights <= 0.0) or not np.all(np.isfinite(values)):
        raise ValueError("scaler requires finite rows and positive weights")
    mean = np.average(values, axis=0, weights=weights)
    variance = np.average((values - mean) ** 2, axis=0, weights=weights)
    scale = np.sqrt(np.maximum(variance, 0.0))
    scale[scale == 0.0] = 1.0
    return WeightedScaler(mean=mean, scale=scale)


def hard_set_from_logits(logits: np.ndarray, tau: float = 0.5) -> np.ndarray:
    """Return the non-empty hard compatible set used by the policy baseline."""
    logits = np.asarray(logits, dtype=np.float64)
    hard_set = expit(logits) >= float(tau)
    empty = ~hard_set.any(axis=1)
    if np.any(empty):
        hard_set[empty] = False
        hard_set[np.flatnonzero(empty), np.argmax(logits[empty], axis=1)] = True
    return hard_set


def contextual_design(
    *,
    ensemble_logits: np.ndarray,
    per_seed_logits: np.ndarray,
    set_mass: np.ndarray,
    action_risk: np.ndarray,
    hard_actions: np.ndarray,
    posterior_actions: np.ndarray,
    utilities: np.ndarray,
    hard_tau: float = 0.5,
) -> dict[str, np.ndarray]:
    """Build the inference-time Wave 56 design matrix with shape [token,policy,17]."""
    logits = np.asarray(ensemble_logits, dtype=np.float64)
    seed_logits = np.asarray(per_seed_logits, dtype=np.float64)
    mass = np.asarray(set_mass, dtype=np.float64)
    risk = np.asarray(action_risk, dtype=np.float64)
    hard = np.asarray(hard_actions, dtype=np.int64)
    posterior = np.asarray(posterior_actions, dtype=np.int64)
    utilities = np.asarray(utilities, dtype=np.float64)
    n_tokens, n_families = logits.shape
    n_policies = len(utilities)
    if n_families != 4 or seed_logits.ndim != 3 or seed_logits.shape[1:] != logits.shape:
        raise ValueError("expected ensemble [N,4] and per-seed [S,N,4] logits")
    if mass.shape != (n_tokens, 15) or risk.shape != (n_tokens, n_policies, 4):
        raise ValueError("posterior mass/action risk shape mismatch")
    if hard.shape != (n_tokens, n_policies) or posterior.shape != hard.shape:
        raise ValueError("action arrays must have shape [N,P]")

    minimum = np.take_along_axis(risk, posterior[..., None], axis=-1)[..., 0]
    hard_risk = np.take_along_axis(risk, hard[..., None], axis=-1)[..., 0]
    advantage = np.maximum(hard_risk - minimum, 0.0)
    ordered_risk = np.sort(risk, axis=-1)
    action_margin = ordered_risk[..., 1] - ordered_risk[..., 0]

    clipped = np.clip(mass, np.finfo(np.float64).tiny, 1.0)
    entropy = -np.sum(mass * np.log(clipped), axis=1) / np.log(mass.shape[1])
    ordered_mass = np.sort(mass, axis=1)
    top_mass = ordered_mass[:, -1]
    top_margin = ordered_mass[:, -1] - ordered_mass[:, -2]

    sets = nonempty_sets(4).astype(np.float64)
    set_cardinality = sets.sum(axis=1)
    expected_cardinality = mass @ set_cardinality
    cardinality_variance = mass @ (set_cardinality**2) - expected_cardinality**2

    hard_set = hard_set_from_logits(logits, hard_tau)
    hard_cardinality = hard_set.sum(axis=1).astype(np.float64)
    hard_index = (hard_set.astype(np.int64) * (1 << np.arange(4))).sum(axis=1) - 1
    hard_mass = mass[np.arange(n_tokens), hard_index]

    seed_std = np.std(seed_logits, axis=0, ddof=0)
    seed_std_mean = seed_std.mean(axis=1)
    seed_std_max = seed_std.max(axis=1)

    token_features = np.stack(
        [
            entropy,
            top_mass,
            top_margin,
            hard_cardinality,
            expected_cardinality,
            cardinality_variance,
            hard_mass,
            seed_std_mean,
            seed_std_max,
        ],
        axis=-1,
    )
    repeated = np.broadcast_to(token_features[:, None, :], (n_tokens, n_policies, 9))
    utility_features = np.broadcast_to(utilities[None, :, :], (n_tokens, n_policies, 4))
    design = np.concatenate(
        [
            advantage[..., None],
            hard_risk[..., None],
            minimum[..., None],
            action_margin[..., None],
            repeated,
            utility_features,
        ],
        axis=-1,
    )
    if design.shape != (n_tokens, n_policies, len(FEATURE_NAMES)):
        raise AssertionError("Wave 56 feature schema drift")
    if not np.all(np.isfinite(design)):
        raise FloatingPointError("Wave 56 design contains non-finite values")
    return {
        "design": design,
        "hard_set": hard_set,
        "advantage": advantage,
        "disagreement": hard != posterior,
    }


def realized_gain(
    hard_actions: np.ndarray,
    posterior_actions: np.ndarray,
    target: np.ndarray,
    utilities: np.ndarray,
    incompatible_penalty: float,
) -> np.ndarray:
    """Observed regret reduction from replacing hard with posterior action."""
    hard_regret = constrained_regret(hard_actions, target, utilities, incompatible_penalty)
    posterior_regret = constrained_regret(
        posterior_actions, target, utilities, incompatible_penalty
    )
    return hard_regret - posterior_regret


def disagreement_weights(disagreement: np.ndarray) -> np.ndarray:
    """Give every token with at least one disagreement total weight one."""
    disagreement = np.asarray(disagreement, dtype=bool)
    counts = disagreement.sum(axis=1)
    weights = np.zeros(disagreement.shape, dtype=np.float64)
    active = counts > 0
    weights[active] = disagreement[active] / counts[active, None]
    return weights


def stratified_gain_shuffle(
    gain: np.ndarray,
    disagreement: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    """Shuffle gain within (policy, disagreement-count), preserving learner measure."""
    gain = np.asarray(gain, dtype=np.float64)
    disagreement = np.asarray(disagreement, dtype=bool)
    if gain.shape != disagreement.shape:
        raise ValueError("gain and disagreement must align")
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    counts = disagreement.sum(axis=1)
    shuffled = gain.copy()
    mapping = np.full(gain.shape, -1, dtype=np.int64)
    movable = np.zeros(gain.shape, dtype=bool)
    moved = np.zeros(gain.shape, dtype=bool)
    for policy in range(gain.shape[1]):
        for count in np.unique(counts[disagreement[:, policy]]):
            indices = np.flatnonzero(disagreement[:, policy] & (counts == count))
            if len(indices) == 0:
                continue
            permutation = rng.permutation(indices)
            shuffled[indices, policy] = gain[permutation, policy]
            mapping[indices, policy] = permutation
            if len(indices) > 1:
                movable[indices, policy] = True
            moved[indices, policy] = permutation != indices
    active = int(disagreement.sum())
    return {
        "target": shuffled,
        "mapping": mapping,
        "movable_fraction": float(movable.sum() / active) if active else None,
        "moved_fraction": float(moved.sum() / active) if active else None,
    }


def apply_gate(
    scores: np.ndarray,
    hard_actions: np.ndarray,
    posterior_actions: np.ndarray,
    threshold: float | str,
) -> dict[str, np.ndarray]:
    """Apply a strict contextual override or reproduce hard exactly."""
    scores = np.asarray(scores, dtype=np.float64)
    hard = np.asarray(hard_actions, dtype=np.int64)
    posterior = np.asarray(posterior_actions, dtype=np.int64)
    if scores.shape != hard.shape or posterior.shape != hard.shape:
        raise ValueError("scores and actions must align")
    disagreement = hard != posterior
    if threshold == "hard_only":
        override = np.zeros(hard.shape, dtype=bool)
    else:
        value = float(threshold)
        if not np.isfinite(value):
            raise ValueError("threshold must be finite")
        override = disagreement & (scores > value)
    actions = np.where(override, posterior, hard)
    if threshold == "hard_only" and not np.array_equal(actions, hard):
        raise AssertionError("hard_only must reproduce the baseline")
    return {"actions": actions, "override": override}


def weighted_error(
    prediction: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    objective: str,
) -> float:
    """Compute the frozen Stage 0 model-selection objective."""
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if objective == "mse":
        loss = (prediction - target) ** 2
    elif objective == "mae":
        loss = np.abs(prediction - target)
    elif objective == "log_loss":
        p = np.clip(prediction, 1e-12, 1.0 - 1e-12)
        loss = -(target * np.log(p) + (1.0 - target) * np.log(1.0 - p))
    else:
        raise ValueError(objective)
    return float(np.average(loss, weights=weights))


def validate_feature_names(names: Iterable[str]) -> tuple[str, ...]:
    names = tuple(names)
    if names != FEATURE_NAMES:
        raise ValueError("feature schema/order mismatch")
    return names
