"""Inference-safe model and feature primitives for Wave 50."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn


EIV_FEATURE_INDICES = (2, 3, 4, 5)


@dataclass(frozen=True)
class FeatureNormalizer:
    mean: np.ndarray
    std: np.ndarray

    def transform(self, features: np.ndarray, no_eiv: bool = False) -> np.ndarray:
        normalized = (features - self.mean) / self.std
        if no_eiv:
            normalized = normalized.copy()
            normalized[:, EIV_FEATURE_INDICES] = 0.0
        return normalized.astype(np.float32)


class DeepSetClassifier(nn.Module):
    """Small permutation-invariant encoder with four matched output logits."""

    def __init__(self, input_dim: int = 6, hidden_dim: int = 64, n_outputs: int = 4):
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
        self.head = nn.Linear(hidden_dim, n_outputs)

    def forward(self, points: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        encoded = self.point_mlp(points)
        float_mask = mask.unsqueeze(-1).to(encoded.dtype)
        mean = (
            (encoded * float_mask).to(torch.float64).sum(dim=1)
            / float_mask.to(torch.float64).sum(dim=1).clamp_min(1.0)
        ).to(encoded.dtype)
        masked = encoded.masked_fill(~mask.unsqueeze(-1), torch.finfo(encoded.dtype).min)
        maximum = masked.max(dim=1).values
        maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
        return self.head(self.set_mlp(torch.cat([mean, maximum], dim=-1)))


def canonical_point_features(row: dict[str, Any]) -> np.ndarray:
    """Build six public point features after declared canonicalization."""
    x = np.asarray(row["x"], dtype=np.float64)
    y = np.asarray(row["y"], dtype=np.float64)
    covariance = np.asarray(row["covariance"], dtype=np.float64)
    semantics = row["coordinate_semantics"]
    inverse = np.diag([
        float(semantics["x_scale_to_canonical"]),
        float(semantics["y_scale_to_canonical"]),
    ])
    observed = np.column_stack([x, y]) @ inverse.T
    covariance = np.einsum("ab,nbc,dc->nad", inverse, covariance, inverse)
    sigma_x = np.sqrt(np.maximum(covariance[:, 0, 0], 0.0))
    sigma_y = np.sqrt(np.maximum(covariance[:, 1, 1], 0.0))
    covariance_known = float(semantics["covariance_knowledge"] == "full")
    known = np.full(len(x), covariance_known, dtype=np.float64)
    return np.column_stack([
        observed[:, 0], observed[:, 1], sigma_x, sigma_y, covariance[:, 0, 1], known
    ])


def collate_examples(
    examples: list[dict[str, Any]],
    indices: np.ndarray,
    point_seed: int | None = None,
) -> tuple[torch.Tensor, ...]:
    selected = [examples[int(index)] for index in indices]
    max_points = max(example["features"].shape[0] for example in selected)
    feature_dim = selected[0]["features"].shape[1]
    points = np.zeros((len(selected), max_points, feature_dim), dtype=np.float32)
    mask = np.zeros((len(selected), max_points), dtype=bool)
    targets = np.zeros((len(selected), 4), dtype=np.float32)
    point_rng = np.random.default_rng(point_seed) if point_seed is not None else None
    for row_index, example in enumerate(selected):
        n = example["features"].shape[0]
        order = np.arange(n) if point_rng is None else point_rng.permutation(n)
        points[row_index, :n] = example["features"][order]
        mask[row_index, :n] = True
        targets[row_index] = example["target"]
    return torch.from_numpy(points), torch.from_numpy(mask), torch.from_numpy(targets)


def permute_example_points(
    examples: list[dict[str, Any]],
    seed: int,
) -> list[dict[str, Any]]:
    """Return examples with independently permuted point order and unchanged semantics."""
    rng = np.random.default_rng(seed)
    permuted = []
    for example in examples:
        features = np.asarray(example["features"])
        permuted.append({**example, "features": features[rng.permutation(len(features))].copy()})
    return permuted


@torch.no_grad()
def predict_logits(model: DeepSetClassifier, examples: list[dict[str, Any]], batch_size: int) -> np.ndarray:
    model.eval()
    outputs = []
    for start in range(0, len(examples), batch_size):
        indices = np.arange(start, min(start + batch_size, len(examples)))
        points, mask, _ = collate_examples(examples, indices)
        outputs.append(model(points, mask).cpu().numpy())
    return np.concatenate(outputs, axis=0) if outputs else np.empty((0, 4))
