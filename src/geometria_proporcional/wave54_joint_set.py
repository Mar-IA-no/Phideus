"""Regularized joint-set posterior primitives for Wave 54."""

from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

from .wave53_uncertainty import nonempty_sets, ordinal_loss_tensor


STRUCTURES = ("joint_unary", "joint_unary_cardinality", "joint_full")
PAIR_INDICES = tuple(combinations(range(4), 2))


def parameter_size(structure: str) -> int:
    if structure == "joint_unary":
        return 4
    if structure == "joint_unary_cardinality":
        return 7
    if structure == "joint_full":
        return 12
    raise ValueError(f"unknown structure: {structure}")


def reference_parameters(structure: str) -> np.ndarray:
    theta = np.zeros(parameter_size(structure), dtype=np.float64)
    theta[:4] = 1.0
    return theta


def centered_interactions(theta: np.ndarray, structure: str) -> np.ndarray:
    """Return six pair coefficients whose sum is exactly zero."""
    if structure != "joint_full":
        return np.zeros(6, dtype=np.float64)
    free = np.asarray(theta, dtype=np.float64)[7:12]
    return np.concatenate([free, [-float(free.sum())]])


def feature_tensor(logits: np.ndarray, structure: str) -> np.ndarray:
    """Build conditional-set features with shape [token, set, parameter]."""
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 2 or logits.shape[1] != 4 or not np.all(np.isfinite(logits)):
        raise ValueError("logits must be finite with shape [tokens, 4]")
    sets = nonempty_sets(4).astype(np.float64)
    blocks = [logits[:, None, :] * sets[None, :, :]]
    if structure in {"joint_unary_cardinality", "joint_full"}:
        cardinality = sets.sum(axis=1).astype(int)
        blocks.append(
            np.stack([(cardinality == k).astype(np.float64) for k in (2, 3, 4)], axis=1)[
                None, :, :
            ].repeat(len(logits), axis=0)
        )
    if structure == "joint_full":
        pair = np.stack([sets[:, i] * sets[:, j] for i, j in PAIR_INDICES], axis=1)
        # The sixth pair is the reference coordinate, enforcing sum(J_tilde)=0.
        contrast = pair[:, :5] - pair[:, [5]]
        blocks.append(contrast[None, :, :].repeat(len(logits), axis=0))
    if structure not in STRUCTURES:
        raise ValueError(f"unknown structure: {structure}")
    return np.concatenate(blocks, axis=-1)


def target_set_indices(target: np.ndarray) -> np.ndarray:
    target = np.asarray(target, dtype=bool)
    if target.ndim != 2 or target.shape[1] != 4 or not np.all(target.any(axis=1)):
        raise ValueError("target must contain non-empty sets with shape [tokens, 4]")
    masks = (target.astype(np.int64) * (1 << np.arange(4))).sum(axis=1)
    return masks - 1


def posterior_mass(logits: np.ndarray, theta: np.ndarray, structure: str) -> np.ndarray:
    features = feature_tensor(logits, structure)
    theta = np.asarray(theta, dtype=np.float64)
    if theta.shape != (features.shape[-1],) or not np.all(np.isfinite(theta)):
        raise ValueError("theta does not match the selected structure")
    score = np.einsum("nsd,d->ns", features, theta, optimize=True)
    log_mass = score - logsumexp(score, axis=1, keepdims=True)
    mass = np.exp(log_mass)
    if not np.all(np.isfinite(mass)):
        raise FloatingPointError("posterior contains non-finite values")
    return mass


def nll_and_gradient(
    theta: np.ndarray,
    features: np.ndarray,
    target_index: np.ndarray,
    regularization: float,
    theta_reference: np.ndarray,
) -> tuple[float, np.ndarray]:
    theta = np.asarray(theta, dtype=np.float64)
    score = np.einsum("nsd,d->ns", features, theta, optimize=True)
    log_normalizer = logsumexp(score, axis=1)
    rows = np.arange(len(target_index))
    nll = np.mean(log_normalizer - score[rows, target_index])
    mass = np.exp(score - log_normalizer[:, None])
    expected = np.einsum("ns,nsd->nd", mass, features, optimize=True)
    observed = features[rows, target_index]
    delta = theta - theta_reference
    objective = float(nll + 0.5 * regularization * np.dot(delta, delta))
    gradient = (expected - observed).mean(axis=0) + regularization * delta
    if not np.isfinite(objective) or not np.all(np.isfinite(gradient)):
        raise FloatingPointError("non-finite joint posterior objective")
    return objective, gradient


def fit_joint_posterior(
    logits: np.ndarray,
    target: np.ndarray,
    structure: str,
    regularization: float,
    *,
    max_iter: int = 2000,
    gtol: float = 1e-9,
    ftol: float = 1e-12,
) -> dict[str, object]:
    if regularization <= 0.0:
        raise ValueError("regularization must be positive")
    features = feature_tensor(logits, structure)
    target_index = target_set_indices(target)
    reference = reference_parameters(structure)

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        return nll_and_gradient(theta, features, target_index, regularization, reference)

    result = minimize(
        objective,
        reference.copy(),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": int(max_iter), "gtol": float(gtol), "ftol": float(ftol)},
    )
    theta = np.asarray(result.x, dtype=np.float64)
    value, gradient = objective(theta)
    if not result.success or not np.all(np.isfinite(theta)):
        raise RuntimeError(f"joint posterior fit failed: {result.message}")
    return {
        "theta": theta,
        "objective": value,
        "gradient_norm": float(np.linalg.norm(gradient)),
        "iterations": int(result.nit),
        "function_evaluations": int(result.nfev),
        "message": str(result.message),
        "interaction_coefficients": centered_interactions(theta, structure),
    }


def expected_regret_from_mass(
    set_mass: np.ndarray,
    utilities: np.ndarray,
    incompatible_penalty: float,
) -> dict[str, np.ndarray]:
    set_mass = np.asarray(set_mass, dtype=np.float64)
    sets = nonempty_sets(4)
    if set_mass.ndim != 2 or set_mass.shape[1] != len(sets):
        raise ValueError("set_mass must have shape [tokens, 15]")
    if np.any(set_mass < 0.0) or not np.all(np.isfinite(set_mass)):
        raise ValueError("set_mass must be finite and non-negative")
    np.testing.assert_allclose(set_mass.sum(axis=1), 1.0, atol=1e-10)
    loss = ordinal_loss_tensor(sets, utilities, incompatible_penalty)
    action_risk = np.einsum("ns,pas->npa", set_mass, loss, optimize=True)
    actions = np.argmin(action_risk, axis=-1).astype(np.int64)
    ordered = np.sort(action_risk, axis=-1)
    return {
        "action_risk": action_risk,
        "actions": actions,
        "minimum_risk": ordered[..., 0],
        "margin": ordered[..., 1] - ordered[..., 0],
    }


def marginal_probability(set_mass: np.ndarray) -> np.ndarray:
    return np.asarray(set_mass, dtype=np.float64) @ nonempty_sets(4).astype(np.float64)


def empirical_set_prior(target: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    if alpha <= 0.0:
        raise ValueError("alpha must be positive")
    counts = np.bincount(target_set_indices(target), minlength=15).astype(np.float64)
    return (counts + alpha) / (counts.sum() + alpha * len(counts))
