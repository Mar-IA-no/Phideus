"""Conservative posterior-to-policy bridge primitives for Wave 55."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from .wave52_policy import authorized_actions, constrained_regret


HARD_ONLY = "hard_only"
ADVANTAGE_ATOL = 1e-12


def posterior_policy(action_risk: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return Bayes actions and their minimum estimated risk."""
    risk = np.asarray(action_risk, dtype=np.float64)
    if risk.ndim != 3 or risk.shape[-1] != 4 or not np.all(np.isfinite(risk)):
        raise ValueError("action_risk must be finite with shape [tokens, policies, 4]")
    actions = np.argmin(risk, axis=-1).astype(np.int64)
    minimum = np.take_along_axis(risk, actions[..., None], axis=-1)[..., 0]
    return actions, minimum


def posterior_advantage(
    action_risk: np.ndarray,
    hard_actions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate the regret reduction of replacing each hard action by the Bayes action."""
    risk = np.asarray(action_risk, dtype=np.float64)
    hard = np.asarray(hard_actions, dtype=np.int64)
    if hard.shape != risk.shape[:2]:
        raise ValueError("hard_actions must align with action_risk tokens and policies")
    if np.any((hard < 0) | (hard >= risk.shape[-1])):
        raise ValueError("hard_actions contain an invalid family index")
    posterior_actions, minimum = posterior_policy(risk)
    hard_risk = np.take_along_axis(risk, hard[..., None], axis=-1)[..., 0]
    advantage = hard_risk - minimum
    if np.any(advantage < -ADVANTAGE_ATOL):
        raise RuntimeError("posterior minimum risk exceeds hard-action risk")
    return posterior_actions, np.maximum(advantage, 0.0), hard_risk


def bridge_actions(
    action_risk: np.ndarray,
    hard_actions: np.ndarray,
    gamma: float | str,
    *,
    atol: float = ADVANTAGE_ATOL,
) -> dict[str, np.ndarray]:
    """Override the hard action only when estimated advantage exceeds ``gamma``."""
    hard = np.asarray(hard_actions, dtype=np.int64)
    posterior_actions, advantage, hard_risk = posterior_advantage(action_risk, hard)
    if gamma == HARD_ONLY:
        override = np.zeros(hard.shape, dtype=bool)
    else:
        threshold = float(gamma)
        if threshold < 0.0 or not np.isfinite(threshold):
            raise ValueError("gamma must be non-negative and finite")
        override = advantage > threshold + float(atol)
    actions = np.where(override, posterior_actions, hard)
    if gamma == HARD_ONLY and not np.array_equal(actions, hard):
        raise AssertionError("hard_only must reproduce the hard policy exactly")
    return {
        "actions": actions,
        "posterior_actions": posterior_actions,
        "advantage": advantage,
        "hard_risk": hard_risk,
        "override": override,
    }


def action_metric_arrays(
    actions: np.ndarray,
    target: np.ndarray,
    utilities: np.ndarray,
    incompatible_penalty: float,
) -> dict[str, np.ndarray]:
    """Return token-level metrics after averaging the fixed policy measurements."""
    actions = np.asarray(actions, dtype=np.int64)
    target = np.asarray(target, dtype=bool)
    oracle = authorized_actions(target, utilities)
    regret_by_policy = constrained_regret(
        actions, target, utilities, incompatible_penalty
    )
    compatible_by_policy = target[np.arange(len(target))[:, None], actions]
    return {
        "accuracy": np.mean(actions == oracle, axis=1),
        "compatible": np.mean(compatible_by_policy, axis=1),
        "regret": np.mean(regret_by_policy, axis=1),
        "worst_regret": np.max(regret_by_policy, axis=1),
        "regret_by_policy": regret_by_policy,
        "compatible_by_policy": compatible_by_policy,
    }


def override_diagnostics(
    bridge: dict[str, np.ndarray],
    hard_actions: np.ndarray,
    target: np.ndarray,
    utilities: np.ndarray,
    incompatible_penalty: float,
    *,
    token_mask: np.ndarray | None = None,
    atol: float = ADVANTAGE_ATOL,
) -> dict[str, Any]:
    """Describe observed consequences of overrides without treating neutral ties as wins."""
    override = np.asarray(bridge["override"], dtype=bool)
    actions = np.asarray(bridge["actions"], dtype=np.int64)
    hard = np.asarray(hard_actions, dtype=np.int64)
    if override.shape != actions.shape or hard.shape != actions.shape:
        raise ValueError("override, bridge actions, and hard actions must align")
    if token_mask is not None:
        selected = np.asarray(token_mask, dtype=bool)
        if selected.shape != (actions.shape[0],):
            raise ValueError("token_mask must have shape [tokens]")
        override = override[selected]
        actions = actions[selected]
        hard = hard[selected]
        target = np.asarray(target)[selected]
    bridge_regret = constrained_regret(actions, target, utilities, incompatible_penalty)
    hard_regret = constrained_regret(hard, target, utilities, incompatible_penalty)
    delta = bridge_regret - hard_regret
    beneficial = override & (delta < -atol)
    harmful = override & (delta > atol)
    neutral = override & ~(beneficial | harmful)
    total = delta.size
    n_override = int(override.sum())
    n_non_neutral = int((beneficial | harmful).sum())
    return {
        "n_overrides": n_override,
        "n_non_neutral_overrides": n_non_neutral,
        "override_rate": float(n_override / total),
        "beneficial_fraction_all": float(beneficial.sum() / total),
        "neutral_fraction_all": float(neutral.sum() / total),
        "harmful_fraction_all": float(harmful.sum() / total),
        "override_precision": (
            float(beneficial.sum() / n_non_neutral) if n_non_neutral else None
        ),
        "regret_conditioned_on_override": (
            float(bridge_regret[override].mean()) if n_override else None
        ),
        "observed_regret_delta": delta,
        "beneficial": beneficial,
        "neutral": neutral,
        "harmful": harmful,
    }


def select_gamma(
    rows: Iterable[dict[str, Any]],
    *,
    hard_accuracy: float,
    hard_compatible: float,
    accuracy_margin: float = 0.01,
) -> dict[str, Any]:
    """Select a conservative gate using only predeclared feasibility constraints."""
    candidates = []
    for raw in rows:
        row = dict(raw)
        feasible = (
            float(row["accuracy"]) >= float(hard_accuracy) - float(accuracy_margin)
            and float(row["compatible"]) >= float(hard_compatible)
        )
        row["feasible"] = bool(feasible)
        candidates.append(row)
    feasible = [row for row in candidates if row["feasible"]]
    if not feasible:
        raise RuntimeError("gamma grid has no feasible candidate; hard_only must be present")

    def conservatism(value: float | str) -> float:
        return float("inf") if value == HARD_ONLY else float(value)

    selected = min(
        feasible,
        key=lambda row: (float(row["regret"]), -conservatism(row["gamma"])),
    )
    return {"selected": selected, "grid": candidates}


def algebraic_sign(value: float, atol: float = ADVANTAGE_ATOL) -> int:
    """Return the preregistered -1/0/+1 state for selector sensitivity."""
    value = float(value)
    if value > atol:
        return 1
    if value < -atol:
        return -1
    return 0
