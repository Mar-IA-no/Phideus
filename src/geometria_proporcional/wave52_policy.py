"""Ordinal policy-transport primitives for Wave 52."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch
from torch import nn


def validate_policy_groups(
    permutations: np.ndarray,
    groups: Iterable[Iterable[int]],
) -> tuple[tuple[int, ...], ...]:
    """Validate a 3x8 partition with exact family-by-rank balance."""
    permutations = np.asarray(permutations)
    normalized = tuple(tuple(int(index) for index in group) for group in groups)
    if permutations.shape != (24, 4):
        raise ValueError("expected all 24 permutations of four utility levels")
    if len(normalized) != 3 or any(len(group) != 8 for group in normalized):
        raise ValueError("policy partition must contain three groups of eight")
    flat = [index for group in normalized for index in group]
    if sorted(flat) != list(range(24)):
        raise ValueError("policy groups must partition permutation indices 0..23")
    for group in normalized:
        selected = permutations[list(group)]
        for family in range(4):
            counts = np.bincount(selected[:, family], minlength=4)
            if not np.array_equal(counts, np.full(4, 2)):
                raise ValueError("each family must occupy every rank twice per group")
    return normalized


def policy_fold(
    groups: tuple[tuple[int, ...], ...], fold: int
) -> dict[str, tuple[int, ...]]:
    """Rotate balanced groups so every policy is held out exactly once."""
    if fold not in range(3):
        raise ValueError("policy fold must be 0, 1, or 2")
    return {
        "policy_train": groups[fold],
        "policy_val": groups[(fold + 1) % 3],
        "policy_shift": groups[(fold + 2) % 3],
    }


def authorized_actions(targets: np.ndarray, utilities: np.ndarray) -> np.ndarray:
    """Return argmax utility restricted to each non-empty compatible set."""
    targets = np.asarray(targets, dtype=bool)
    utilities = np.asarray(utilities, dtype=np.float64)
    if targets.ndim != 2 or targets.shape[1] != 4:
        raise ValueError("targets must have shape [tokens, 4]")
    if utilities.ndim != 2 or utilities.shape[1] != 4:
        raise ValueError("utilities must have shape [policies, 4]")
    if not np.all(targets.any(axis=1)):
        raise ValueError("authorized actions require non-empty compatible sets")
    masked = np.where(targets[:, None, :], utilities[None, :, :], -np.inf)
    return np.argmax(masked, axis=-1).astype(np.int64)


def explicit_set_actions(
    probabilities: np.ndarray,
    utilities: np.ndarray,
    tau: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply max-utility inside a predicted set, with probability fallback."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    utilities = np.asarray(utilities, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] != 4:
        raise ValueError("probabilities must have shape [tokens, 4]")
    if utilities.ndim != 2 or utilities.shape[1] != 4:
        raise ValueError("utilities must have shape [policies, 4]")
    predicted = probabilities >= float(tau)
    fallback = ~predicted.any(axis=1)
    scores = np.where(predicted[:, None, :], utilities[None, :, :], -np.inf)
    actions = np.argmax(scores, axis=-1).astype(np.int64)
    if np.any(fallback):
        actions[fallback] = np.argmax(probabilities[fallback], axis=1)[:, None]
    return actions, fallback


def score_composition_actions(
    logits: np.ndarray,
    utilities: np.ndarray,
    utility_weight: float,
) -> np.ndarray:
    """Compose log-compatibility and normalized ordinal utility continuously."""
    logits = np.asarray(logits, dtype=np.float64)
    utilities = np.asarray(utilities, dtype=np.float64)
    span = utilities.max(axis=1, keepdims=True) - utilities.min(axis=1, keepdims=True)
    if np.any(span <= 0):
        raise ValueError("utility vectors must have non-zero range")
    normalized = (utilities - utilities.min(axis=1, keepdims=True)) / span
    log_probability = -np.logaddexp(0.0, -logits)
    scores = (
        log_probability[:, None, :] + float(utility_weight) * normalized[None, :, :]
    )
    return np.argmax(scores, axis=-1).astype(np.int64)


def constrained_regret(
    actions: np.ndarray,
    targets: np.ndarray,
    utilities: np.ndarray,
    incompatible_penalty: float = 1.25,
) -> np.ndarray:
    """Regret in [0,1] for compatible actions and a larger fixed invalid penalty."""
    actions = np.asarray(actions, dtype=np.int64)
    targets = np.asarray(targets, dtype=bool)
    utilities = np.asarray(utilities, dtype=np.float64)
    if actions.shape != (targets.shape[0], utilities.shape[0]):
        raise ValueError("actions shape must be [tokens, policies]")
    optimum = np.max(
        np.where(targets[:, None, :], utilities[None, :, :], -np.inf), axis=-1
    )
    chosen = utilities[np.arange(utilities.shape[0])[None, :], actions]
    span = utilities.max(axis=1) - utilities.min(axis=1)
    regret = (optimum - chosen) / span[None, :]
    compatible = targets[np.arange(targets.shape[0])[:, None], actions]
    return np.where(compatible, regret, float(incompatible_penalty))


class SharedFamilyReader(nn.Module):
    """Shared per-family scorer, equivariant to joint family permutation."""

    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, evidence: torch.Tensor, utility: torch.Tensor) -> torch.Tensor:
        """Score [batch, policy, family] from evidence [batch,4] and utility [policy,4]."""
        if evidence.ndim != 2 or evidence.shape[1] != 4:
            raise ValueError("evidence must have shape [batch, 4]")
        if utility.ndim != 2 or utility.shape[1] != 4:
            raise ValueError("utility must have shape [policy, 4]")
        batch, policies = evidence.shape[0], utility.shape[0]
        evidence = evidence[:, None, :].expand(batch, policies, 4)
        utility = utility[None, :, :].expand(batch, policies, 4)
        return self.network(torch.stack([evidence, utility], dim=-1)).squeeze(-1)


class ContextualPolicyDeepSet(nn.Module):
    """DeepSets evidence encoder with set and ordinal-policy outputs."""

    def __init__(
        self,
        input_dim: int = 6,
        hidden_dim: int = 64,
        reader_hidden_dim: int = 32,
    ):
        super().__init__()
        self.point_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.set_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.set_head = nn.Linear(hidden_dim, 4)
        self.choice_evidence_head = nn.Linear(hidden_dim, 4)
        self.policy_reader = SharedFamilyReader(reader_hidden_dim)

    def encode(self, points: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        encoded = self.point_mlp(points)
        float_mask = mask.unsqueeze(-1).to(encoded.dtype)
        mean = (
            (encoded * float_mask).to(torch.float64).sum(dim=1)
            / float_mask.to(torch.float64).sum(dim=1).clamp_min(1.0)
        ).to(encoded.dtype)
        masked = encoded.masked_fill(
            ~mask.unsqueeze(-1), torch.finfo(encoded.dtype).min
        )
        maximum = masked.max(dim=1).values
        maximum = torch.where(
            torch.isfinite(maximum), maximum, torch.zeros_like(maximum)
        )
        return self.set_mlp(torch.cat([mean, maximum], dim=-1))

    def forward(
        self,
        points: torch.Tensor,
        mask: torch.Tensor,
        utilities: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encode(points, mask)
        return self.set_head(latent), self.policy_reader(
            self.choice_evidence_head(latent), utilities
        )
