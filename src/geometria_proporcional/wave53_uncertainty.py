"""Uncertainty-aware ordinal decision primitives for Wave 53."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Iterable

import numpy as np


def nonempty_sets(n_families: int = 4) -> np.ndarray:
    """Enumerate every non-empty subset in increasing binary-mask order."""
    if n_families < 1:
        raise ValueError("n_families must be positive")
    masks = np.arange(1, 2**n_families, dtype=np.uint64)[:, None]
    bits = np.arange(n_families, dtype=np.uint64)[None, :]
    return ((masks >> bits) & 1).astype(bool)


def independent_nonempty_mass(
    probabilities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return Bernoulli-product mass conditioned on the set being non-empty."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2:
        raise ValueError("probabilities must have shape [tokens, families]")
    if np.any(~np.isfinite(probabilities)) or np.any(
        (probabilities < 0.0) | (probabilities > 1.0)
    ):
        raise ValueError("probabilities must be finite values in [0, 1]")
    sets = nonempty_sets(probabilities.shape[1])
    clipped = np.clip(
        probabilities, np.finfo(np.float64).tiny, 1.0 - np.finfo(np.float64).eps
    )
    log_mass = np.sum(
        np.where(
            sets[None, :, :],
            np.log(clipped)[:, None, :],
            np.log1p(-clipped)[:, None, :],
        ),
        axis=-1,
    )
    log_mass -= np.max(log_mass, axis=1, keepdims=True)
    mass = np.exp(log_mass)
    return sets, mass / mass.sum(axis=1, keepdims=True)


def ordinal_loss_tensor(
    sets: np.ndarray,
    utilities: np.ndarray,
    incompatible_penalty: float = 1.25,
) -> np.ndarray:
    """Loss ``[policy, action, set]`` matching Wave 52 constrained regret."""
    sets = np.asarray(sets, dtype=bool)
    utilities = np.asarray(utilities, dtype=np.float64)
    if sets.ndim != 2 or utilities.ndim != 2 or sets.shape[1] != utilities.shape[1]:
        raise ValueError("sets and utilities must share their family dimension")
    if not np.all(sets.any(axis=1)):
        raise ValueError("sets must be non-empty")
    span = utilities.max(axis=1) - utilities.min(axis=1)
    if np.any(span <= 0.0):
        raise ValueError("every utility policy must have non-zero range")
    optimum = np.max(
        np.where(sets[None, :, :], utilities[:, None, :], -np.inf), axis=-1
    )
    compatible_regret = (optimum[:, None, :] - utilities[:, :, None]) / span[
        :, None, None
    ]
    compatible = sets.T[None, :, :]
    return np.where(compatible, compatible_regret, float(incompatible_penalty))


def expected_regret_actions(
    probabilities: np.ndarray,
    utilities: np.ndarray,
    incompatible_penalty: float = 1.25,
) -> dict[str, np.ndarray]:
    """Choose the lowest expected-loss action for every token and utility policy."""
    sets, set_mass = independent_nonempty_mass(probabilities)
    losses = ordinal_loss_tensor(sets, utilities, incompatible_penalty)
    action_risk = np.einsum("ns,pas->npa", set_mass, losses, optimize=True)
    actions = np.argmin(action_risk, axis=-1).astype(np.int64)
    ordered = np.sort(action_risk, axis=-1)
    return {
        "sets": sets,
        "set_mass": set_mass,
        "action_risk": action_risk,
        "actions": actions,
        "minimum_risk": ordered[..., 0],
        "margin": ordered[..., 1] - ordered[..., 0],
    }


def stable_fraction(token: str, seed: int) -> float:
    """Stable label-independent tie breaker in [0, 1)."""
    digest = hashlib.sha256(f"{seed}:{token}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def stratified_token_split(
    tokens: Iterable[str],
    strata: Iterable[tuple[str, int]],
    fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Split token indices deterministically while preserving declared strata."""
    tokens = list(tokens)
    strata = list(strata)
    if len(tokens) != len(strata):
        raise ValueError("tokens and strata must align")
    if not 0.0 < fraction < 1.0:
        raise ValueError("fraction must be in (0, 1)")
    groups: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index, stratum in enumerate(strata):
        groups[tuple(stratum)].append(index)
    left: list[int] = []
    right: list[int] = []
    for key in sorted(groups):
        ordered = sorted(
            groups[key], key=lambda i: (stable_fraction(tokens[i], seed), tokens[i])
        )
        if len(ordered) < 2:
            raise ValueError(f"cannot split singleton stratum {key}")
        cut = int(round(len(ordered) * fraction))
        cut = min(max(cut, 1), len(ordered) - 1)
        left.extend(ordered[:cut])
        right.extend(ordered[cut:])
    return np.asarray(sorted(left), dtype=np.int64), np.asarray(
        sorted(right), dtype=np.int64
    )


def deranged_within_strata(
    tokens: Iterable[str],
    strata: Iterable[tuple[str, int]],
    seed: int,
) -> np.ndarray:
    """Map each token to a different token in the same stratum by a stable rotation."""
    tokens = list(tokens)
    strata = list(strata)
    groups: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index, stratum in enumerate(strata):
        groups[tuple(stratum)].append(index)
    mapping = np.arange(len(tokens), dtype=np.int64)
    for key in sorted(groups):
        indices = sorted(
            groups[key], key=lambda i: (stable_fraction(tokens[i], seed), tokens[i])
        )
        if len(indices) < 2:
            raise ValueError(f"cannot derange singleton stratum {key}")
        for source, target in zip(indices, indices[1:] + indices[:1], strict=True):
            mapping[source] = target
    if np.any(mapping == np.arange(len(tokens))):
        raise RuntimeError("derangement contains fixed points")
    return mapping


def coverage_boundary(
    scores: np.ndarray,
    tokens: Iterable[str],
    coverage: float,
    seed: int,
) -> dict[str, float]:
    """Fit a lexicographic score boundary at a nominal token coverage."""
    scores = np.asarray(scores, dtype=np.float64)
    tokens = list(tokens)
    if len(scores) != len(tokens) or len(tokens) == 0:
        raise ValueError("scores and non-empty tokens must align")
    if not 0.0 < coverage <= 1.0:
        raise ValueError("coverage must be in (0, 1]")
    tie = np.asarray([stable_fraction(token, seed) for token in tokens])
    order = np.lexsort((np.asarray(tokens), tie, scores))
    retained = max(1, int(round(coverage * len(tokens))))
    boundary_index = int(order[retained - 1])
    return {
        "score": float(scores[boundary_index]),
        "tie": float(tie[boundary_index]),
        "nominal_coverage": float(coverage),
        "selected_count": retained,
        "fit_count": len(tokens),
    }


def apply_coverage_boundary(
    scores: np.ndarray,
    tokens: Iterable[str],
    boundary: dict[str, float],
    seed: int,
) -> np.ndarray:
    """Apply a frozen lexicographic boundary without consulting outcomes."""
    scores = np.asarray(scores, dtype=np.float64)
    tokens = list(tokens)
    tie = np.asarray([stable_fraction(token, seed) for token in tokens])
    return (scores < boundary["score"]) | (
        (scores == boundary["score"]) & (tie <= boundary["tie"])
    )


def discrete_aurc(
    selection_scores: np.ndarray,
    observed_risk: np.ndarray,
    tokens: Iterable[str],
) -> float:
    """Mean cumulative observed risk over token coverages 1/N through N/N."""
    selection_scores = np.asarray(selection_scores, dtype=np.float64)
    observed_risk = np.asarray(observed_risk, dtype=np.float64)
    tokens = np.asarray(list(tokens))
    if not (len(selection_scores) == len(observed_risk) == len(tokens)):
        raise ValueError("AURC inputs must align")
    order = np.lexsort((tokens, selection_scores))
    cumulative = np.cumsum(observed_risk[order]) / np.arange(1, len(order) + 1)
    return float(np.mean(cumulative))


def paired_bootstrap_indices(n_tokens: int, n_boot: int, seed: int) -> np.ndarray:
    """Generate one reusable cluster bootstrap index matrix."""
    if n_tokens < 1 or n_boot < 1:
        raise ValueError("n_tokens and n_boot must be positive")
    return np.random.default_rng(seed).integers(
        0, n_tokens, size=(n_boot, n_tokens), dtype=np.int64
    )


def paired_delta_ci(
    values_a: np.ndarray,
    values_b: np.ndarray,
    indices: np.ndarray,
) -> dict[str, float]:
    """Conditional paired bootstrap interval using precomputed token resamples."""
    values_a = np.asarray(values_a, dtype=np.float64)
    values_b = np.asarray(values_b, dtype=np.float64)
    if values_a.shape != values_b.shape or values_a.ndim != 1:
        raise ValueError("paired values must be aligned vectors")
    if indices.ndim != 2 or indices.shape[1] != len(values_a):
        raise ValueError("bootstrap indices do not match token count")
    delta = values_a - values_b
    draws = delta[indices].mean(axis=1)
    low, high = np.percentile(draws, [2.5, 97.5])
    return {
        "mean_diff": float(delta.mean()),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "fraction_positive": float(np.mean(draws > 0.0)),
        "n_tokens": len(delta),
        "n_boot": len(draws),
    }
