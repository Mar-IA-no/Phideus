"""Factored set-membership and partial-choice primitives for Wave 51."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any, Iterable

import numpy as np
import torch
from scipy.special import expit
from torch import nn
from torch.nn import functional as F

from .wave50_model import collate_examples
from .wave50_neural import partial_label_softmax_loss, token_batch_indices


class DualHeadDeepSet(nn.Module):
    """Permutation-invariant encoder with separate membership and choice heads."""

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
        self.set_head = nn.Linear(hidden_dim, n_outputs)
        self.choice_head = nn.Linear(hidden_dim, n_outputs)

    def encode(self, points: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        encoded = self.point_mlp(points)
        float_mask = mask.unsqueeze(-1).to(encoded.dtype)
        mean = (
            (encoded * float_mask).to(torch.float64).sum(dim=1)
            / float_mask.to(torch.float64).sum(dim=1).clamp_min(1.0)
        ).to(encoded.dtype)
        masked = encoded.masked_fill(~mask.unsqueeze(-1), torch.finfo(encoded.dtype).min)
        maximum = masked.max(dim=1).values
        maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
        return self.set_mlp(torch.cat([mean, maximum], dim=-1))

    def forward(
        self, points: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encode(points, mask)
        return self.set_head(latent), self.choice_head(latent)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def state_dict_digest(
    model: nn.Module,
    prefixes: tuple[str, ...] | None = None,
) -> str:
    """Hash tensor values canonically, independent of torch serialization metadata."""
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        if prefixes is not None and not name.startswith(prefixes):
            continue
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def make_optimizer(
    parameters: Iterable[nn.Parameter], learning_rate: float, weight_decay: float
) -> torch.optim.Optimizer:
    return torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)


def train_epochs(
    model: DualHeadDeepSet,
    examples: list[dict[str, Any]],
    objective: str,
    seed: int,
    start_epoch: int,
    epochs: int,
    batch_tokens: int,
    optimizer: torch.optim.Optimizer,
) -> list[dict[str, float]]:
    """Train a fixed objective while preserving Wave 50 batch/permutation recipes."""
    history: list[dict[str, float]] = []
    model.train()
    for epoch in range(start_epoch, start_epoch + epochs):
        losses: list[float] = []
        grad_norms: list[float] = []
        for batch_index, indices in enumerate(
            token_batch_indices(examples, batch_tokens, seed, epoch)
        ):
            points, mask, targets = collate_examples(
                examples,
                indices,
                point_seed=seed * 1_000_000 + epoch * 10_000 + batch_index,
            )
            optimizer.zero_grad(set_to_none=True)
            set_logits, choice_logits = model(points, mask)
            if objective == "set_bce":
                loss = F.binary_cross_entropy_with_logits(set_logits, targets)
            elif objective == "choice_partial":
                loss = partial_label_softmax_loss(choice_logits, targets)
            elif objective == "joint_equal":
                loss = (
                    F.binary_cross_entropy_with_logits(set_logits, targets)
                    + partial_label_softmax_loss(choice_logits, targets)
                )
            else:
                raise ValueError(f"unknown objective: {objective}")
            loss.backward()
            squared = sum(
                float(parameter.grad.detach().square().sum())
                for parameter in model.parameters()
                if parameter.grad is not None
            )
            optimizer.step()
            losses.append(float(loss.detach()))
            grad_norms.append(squared**0.5)
        history.append({
            "epoch": float(epoch + 1),
            "mean_loss": float(np.mean(losses)),
            "mean_grad_norm": float(np.mean(grad_norms)),
        })
    return history


@torch.no_grad()
def predict_dual_logits(
    model: DualHeadDeepSet,
    examples: list[dict[str, Any]],
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    set_outputs: list[np.ndarray] = []
    choice_outputs: list[np.ndarray] = []
    for start in range(0, len(examples), batch_size):
        indices = np.arange(start, min(start + batch_size, len(examples)))
        points, mask, _ = collate_examples(examples, indices)
        set_logits, choice_logits = model(points, mask)
        set_outputs.append(set_logits.cpu().numpy())
        choice_outputs.append(choice_logits.cpu().numpy())
    if not set_outputs:
        empty = np.empty((0, 4), dtype=np.float32)
        return empty, empty.copy()
    return np.concatenate(set_outputs), np.concatenate(choice_outputs)


def choice_metrics(
    examples: list[dict[str, Any]],
    set_logits: np.ndarray,
    choice_logits: np.ndarray,
    set_activation: str,
    tau: float,
) -> dict[str, float]:
    """Token-averaged unconstrained and predicted-set-gated top-1 compatibility."""
    if set_activation == "softmax":
        shifted = set_logits - set_logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        set_probability = exp / exp.sum(axis=1, keepdims=True)
    elif set_activation == "sigmoid":
        set_probability = expit(set_logits)
    else:
        raise ValueError(f"unknown set activation: {set_activation}")

    rows: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for example, probability, scores in zip(
        examples, set_probability, choice_logits, strict=True
    ):
        target = np.asarray(example["target"], dtype=bool)
        unconstrained = int(np.argmax(scores))
        candidates = np.flatnonzero(probability >= tau)
        gated = unconstrained if len(candidates) == 0 else int(candidates[np.argmax(scores[candidates])])
        rows[example["pair_token"]].append((float(target[unconstrained]), float(target[gated])))
    token_rows = [
        (float(np.mean([row[0] for row in group])), float(np.mean([row[1] for row in group])))
        for group in rows.values()
    ]
    return {
        "n_pair_tokens": len(token_rows),
        "choice_top1_compatible": float(np.mean([row[0] for row in token_rows])),
        "choice_top1_gated_compatible": float(np.mean([row[1] for row in token_rows])),
    }


def token_metric_rows(
    examples: list[dict[str, Any]],
    set_logits: np.ndarray,
    choice_logits: np.ndarray,
    set_activation: str,
    tau: float,
) -> list[dict[str, Any]]:
    """Return canonical-view-averaged set and choice metrics for every pair token."""
    if set_activation == "softmax":
        shifted = set_logits - set_logits.max(axis=1, keepdims=True)
        exponentiated = np.exp(shifted)
        set_probability = exponentiated / exponentiated.sum(axis=1, keepdims=True)
    elif set_activation == "sigmoid":
        set_probability = expit(set_logits)
    else:
        raise ValueError(f"unknown set activation: {set_activation}")

    by_token: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for example, probability, scores in zip(
        examples, set_probability, choice_logits, strict=True
    ):
        target = np.asarray(example["target"], dtype=bool)
        predicted = probability >= tau
        unconstrained = int(np.argmax(scores))
        candidates = np.flatnonzero(predicted)
        gated = (
            unconstrained
            if len(candidates) == 0
            else int(candidates[np.argmax(scores[candidates])])
        )
        width = int(predicted.sum())
        incompatible = int(np.logical_and(predicted, ~target).sum())
        by_token[example["pair_token"]].append({
            "design_stratum": example.get("design_stratum"),
            "cardinality": int(target.sum()),
            "set_recall": float(np.logical_and(predicted, target).sum() / target.sum()),
            "complete_coverage": float(np.all(~target | predicted)),
            "exact_set": float(np.array_equal(predicted, target)),
            "any_incompatible": float(incompatible > 0),
            "incompatible_fraction": float(incompatible / max(width, 1)),
            "width": float(width),
            "empty": float(width == 0),
            "choice_top1_compatible": float(target[unconstrained]),
            "choice_top1_gated_compatible": float(target[gated]),
        })

    metric_names = (
        "set_recall",
        "complete_coverage",
        "exact_set",
        "any_incompatible",
        "incompatible_fraction",
        "width",
        "empty",
        "choice_top1_compatible",
        "choice_top1_gated_compatible",
    )
    result = []
    for pair_token in sorted(by_token):
        rows = by_token[pair_token]
        if len({row["design_stratum"] for row in rows}) != 1:
            raise ValueError(f"design stratum differs across canonical views for {pair_token}")
        if len({row["cardinality"] for row in rows}) != 1:
            raise ValueError(f"target cardinality differs across canonical views for {pair_token}")
        result.append({
            "pair_token": pair_token,
            "design_stratum": rows[0]["design_stratum"],
            "cardinality": rows[0]["cardinality"],
            "n_canonical_views": len(rows),
            **{
                name: float(np.mean([row[name] for row in rows]))
                for name in metric_names
            },
        })
    return result
