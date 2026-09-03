"""Matched neural primitives for the Wave 50 development smoke.

The module consumes only visible observations plus explicitly authorized labels.
It contains no generator or oracle implementation.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

from .wave49_schema import CATALOG_FAMILIES, SCHEMA_VERSION, read_jsonl
from .wave50_model import (
    EIV_FEATURE_INDICES,
    DeepSetClassifier,
    FeatureNormalizer,
    canonical_point_features,
    collate_examples,
    permute_example_points,
    predict_logits,
)


FAMILY_TO_INDEX = {family: index for index, family in enumerate(CATALOG_FAMILIES)}
ALLOWED_SMOKE_SPLITS = frozenset({"train", "val"})
TARGET_SCHEMA_VERSION = "oracle_compatible_set_v1"
TARGET_COMPATIBILITY_DISTANCE = 4.0


def partial_label_softmax_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Negative log probability mass assigned to the non-empty compatible set."""
    if logits.shape != targets.shape:
        raise ValueError("logits and targets must have the same shape")
    compatible = targets.to(torch.bool)
    if not torch.all(compatible.any(dim=-1)):
        raise ValueError("partial-label targets must be non-empty")
    selected = logits.masked_fill(~compatible, -torch.inf)
    return (torch.logsumexp(logits, dim=-1) - torch.logsumexp(selected, dim=-1)).mean()


def matched_loss(arm: str, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    if arm == "softmax_partial":
        return partial_label_softmax_loss(logits, targets)
    if arm == "sigmoid_set":
        return F.binary_cross_entropy_with_logits(logits, targets, reduction="mean")
    raise ValueError(f"unknown arm: {arm}")


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def target_vector(families: Iterable[str]) -> np.ndarray:
    target = np.zeros(len(CATALOG_FAMILIES), dtype=np.float32)
    for family in families:
        target[FAMILY_TO_INDEX[family]] = 1.0
    if target.sum() == 0:
        raise ValueError("in-catalog compatible target cannot be empty")
    return target


def load_smoke_records(
    benchmark_dir: Path,
    split: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Join visible rows with opened historical labels for train/val only."""
    if split not in ALLOWED_SMOKE_SPLITS:
        raise ValueError(f"smoke split must be one of {sorted(ALLOWED_SMOKE_SPLITS)}")
    benchmark_dir = Path(benchmark_dir).resolve()
    visible_path = (benchmark_dir / "visible" / f"{split}.jsonl").resolve(strict=True)
    label_path = (benchmark_dir / "sealed" / "oracle" / f"{split}.jsonl").resolve(strict=True)
    forbidden = {
        (benchmark_dir / "visible" / "lockbox.jsonl").resolve(strict=True),
        (benchmark_dir / "sealed" / "oracle" / "lockbox.jsonl").resolve(strict=True),
    }
    if visible_path in forbidden or label_path in forbidden:
        raise ValueError("smoke path resolves to lockbox content")
    config_path = (benchmark_dir / "protocol_config.json").resolve(strict=True)
    return load_labeled_records(visible_path, label_path, config_path, split)


def load_labeled_records(
    visible_path: Path,
    label_path: Path,
    config_path: Path | None,
    split: str,
    expected_schema: str | None = None,
    expected_compatibility_distance: float | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Join an explicitly staged visible split with its authorized labels."""
    visible_path = Path(visible_path).resolve(strict=True)
    label_path = Path(label_path).resolve(strict=True)
    reads = [str(visible_path), str(label_path)]
    if config_path is not None:
        config_path = Path(config_path).resolve(strict=True)
        protocol = json.loads(config_path.read_text(encoding="utf-8"))
        schema = protocol.get("schema_version")
        compatibility_distance = protocol.get("oracle_compatibility_distance")
        reads.append(str(config_path))
    else:
        schema = expected_schema
        compatibility_distance = expected_compatibility_distance
    if schema != SCHEMA_VERSION:
        raise ValueError("unexpected benchmark schema")
    if float(compatibility_distance if compatibility_distance is not None else np.nan) != TARGET_COMPATIBILITY_DISTANCE:
        raise ValueError("unexpected oracle compatibility distance")

    visible_rows = read_jsonl(visible_path)
    if any(row.get("split") != split or row.get("schema_version") != SCHEMA_VERSION for row in visible_rows):
        raise ValueError(f"visible row scope/schema mismatch in {split}")
    visible = {row["fixture_id"]: row for row in visible_rows}
    if len(visible) != len(visible_rows):
        raise ValueError(f"duplicate visible fixture_id in {split}")
    labels = read_jsonl(label_path)
    if any(row.get("split") != split or row.get("schema_version") != SCHEMA_VERSION for row in labels):
        raise ValueError(f"label row scope/schema mismatch in {split}")
    if len({row["fixture_id"] for row in labels}) != len(labels):
        raise ValueError(f"duplicate label fixture_id in {split}")
    records: list[dict[str, Any]] = []
    for label in labels:
        if label["is_out_of_catalog"]:
            continue
        if label["calibration_population"] != "canonical_preserving":
            continue
        if label.get("oracle_input_scope") != "sealed_truth+public_parameter_catalog":
            raise ValueError("unexpected target provenance")
        if label.get("selector_output_dependency") is not False:
            raise ValueError("target depends on selector output")
        row = visible[label["fixture_id"]]
        records.append({
            "fixture_id": label["fixture_id"],
            "pair_token": label["pair_token"],
            "features": canonical_point_features(row),
            "target": target_vector(label["oracle_compatible_set"]),
            "target_families": tuple(label["oracle_compatible_set"]),
            "family_id": label["family_id"],
            "design_stratum": label["design_stratum"],
            "representation": label["representation"],
            "covariance_knowledge": label["covariance_knowledge"],
            "n": int(label["n"]),
            "noise_mode": label["noise_mode"],
            "covariance_mode": label["covariance_mode"],
            "range_mode": label["range_mode"],
        })
    _validate_token_groups(records)
    return records, reads


def load_unlabeled_records(
    visible_path: Path,
    config_path: Path,
    split: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Load public observations for a frozen inference pass without labels."""
    visible_path = Path(visible_path).resolve(strict=True)
    config_path = Path(config_path).resolve(strict=True)
    protocol = json.loads(config_path.read_text(encoding="utf-8"))
    if protocol.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unexpected benchmark schema")
    rows = read_jsonl(visible_path)
    if any(row.get("split") != split or row.get("schema_version") != SCHEMA_VERSION for row in rows):
        raise ValueError(f"visible row scope/schema mismatch in {split}")
    if len({row["fixture_id"] for row in rows}) != len(rows):
        raise ValueError(f"duplicate visible fixture_id in {split}")
    records = []
    for row in rows:
        semantics = row["coordinate_semantics"]
        records.append({
            "fixture_id": row["fixture_id"],
            "features": canonical_point_features(row),
            "target": np.zeros(len(CATALOG_FAMILIES), dtype=np.float32),
            "n": int(row["n"]),
            "calibration_population": semantics["calibration_population"],
            "covariance_knowledge": semantics["covariance_knowledge"],
        })
    return records, [str(visible_path), str(config_path)]


def _validate_token_groups(records: list[dict[str, Any]]) -> None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["pair_token"]].append(record)
    for token, group in grouped.items():
        targets = {record["target_families"] for record in group}
        strata = {record["design_stratum"] for record in group}
        if len(targets) != 1 or len(strata) != 1:
            raise ValueError(f"inconsistent canonical views for pair_token {token}")


def stratified_token_subset(
    records: list[dict[str, Any]],
    max_fixtures: int,
    seed: int,
    include_target_hash: bool = False,
) -> list[dict[str, Any]]:
    """Deterministically sample whole tokens with proportional stratum quotas."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["pair_token"]].append(record)
    group_size = {len(group) for group in grouped.values()}
    if len(group_size) != 1:
        raise ValueError("smoke subset expects equal canonical view counts per token")
    views_per_token = next(iter(group_size))
    budget = min(len(grouped), max_fixtures // views_per_token)
    if budget < 1:
        raise ValueError("fixture budget is smaller than one complete token group")

    strata: dict[tuple[Any, ...], list[str]] = defaultdict(list)
    for token, group in grouped.items():
        key: tuple[Any, ...] = (
            group[0]["design_stratum"], len(group[0]["target_families"])
        )
        if include_target_hash:
            key += (group[0]["target_families"],)
        strata[key].append(token)
    rng = np.random.default_rng(seed)
    for tokens in strata.values():
        rng.shuffle(tokens)

    raw = {key: budget * len(tokens) / len(grouped) for key, tokens in strata.items()}
    quota = {key: min(len(strata[key]), int(np.floor(value))) for key, value in raw.items()}
    remaining = budget - sum(quota.values())
    order = sorted(strata, key=lambda key: (-(raw[key] - np.floor(raw[key])), key))
    for key in order:
        if remaining == 0:
            break
        if quota[key] < len(strata[key]):
            quota[key] += 1
            remaining -= 1
    selected = {token for key, tokens in strata.items() for token in tokens[:quota[key]]}
    return [record for record in records if record["pair_token"] in selected]


def split_tokens(
    records: list[dict[str, Any]],
    fraction_first: float,
    seed: int,
    exact_first_tokens: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split whole pair-token groups with bounded proportional allocation by stratum."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["pair_token"]].append(record)
    strata: dict[tuple[str, int], list[str]] = defaultdict(list)
    for token, group in grouped.items():
        strata[(group[0]["design_stratum"], len(group[0]["target_families"]))].append(token)
    rng = np.random.default_rng(seed)
    total = len(grouped)
    target = exact_first_tokens if exact_first_tokens is not None else int(round(total * fraction_first))
    if not 0 < target < total:
        raise ValueError("token split must leave at least one token in each partition")
    for tokens in strata.values():
        if len(tokens) < 2:
            raise ValueError("each validation stratum needs at least two pair tokens")
        rng.shuffle(tokens)
    raw = {key: target * len(tokens) / total for key, tokens in strata.items()}
    counts = {
        key: min(max(int(np.floor(raw[key])), 1), len(tokens) - 1)
        for key, tokens in strata.items()
    }
    while sum(counts.values()) < target:
        candidates = [key for key, tokens in strata.items() if counts[key] < len(tokens) - 1]
        if not candidates:
            raise RuntimeError("cannot allocate requested first-partition token count")
        key = max(candidates, key=lambda item: (raw[item] - counts[item], item))
        counts[key] += 1
    while sum(counts.values()) > target:
        candidates = [key for key in strata if counts[key] > 1]
        if not candidates:
            raise RuntimeError("cannot reduce first partition to requested token count")
        key = max(candidates, key=lambda item: (counts[item] - raw[item], item))
        counts[key] -= 1
    first: set[str] = set()
    for key, tokens in strata.items():
        first.update(tokens[:counts[key]])
    return (
        [record for record in records if record["pair_token"] in first],
        [record for record in records if record["pair_token"] not in first],
    )


def fit_normalizer(records: list[dict[str, Any]]) -> FeatureNormalizer:
    values = np.concatenate([record["features"] for record in records], axis=0)
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return FeatureNormalizer(mean=mean.astype(np.float64), std=std.astype(np.float64))


def prepare_examples(
    records: list[dict[str, Any]],
    normalizer: FeatureNormalizer,
    no_eiv: bool = False,
    target_by_token: dict[str, np.ndarray] | None = None,
) -> list[dict[str, Any]]:
    examples = []
    for record in records:
        target = record["target"] if target_by_token is None else target_by_token[record["pair_token"]]
        examples.append({
            **record,
            "features": normalizer.transform(record["features"], no_eiv=no_eiv),
            "target": np.asarray(target, dtype=np.float32),
        })
    return examples


def balanced_target_derangement(
    records: list[dict[str, Any]],
    seed: int,
) -> tuple[set[str], dict[str, np.ndarray], dict[str, Any]]:
    """Create a zero-match token-level derangement on a balanced eligible subset."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["pair_token"]].append(record)
    strata: dict[tuple[str, int], dict[tuple[str, ...], list[str]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for token, group in grouped.items():
        target = group[0]["target_families"]
        strata[(group[0]["design_stratum"], len(target))][target].append(token)

    rng = np.random.default_rng(seed)
    selected: set[str] = set()
    mapping: dict[str, np.ndarray] = {}
    excluded: list[dict[str, Any]] = []
    for stratum, by_target in sorted(strata.items()):
        eligible = dict(by_target)
        while len(eligible) >= 3:
            minimum_needed = len(eligible) - 1
            reduced = {
                target: tokens for target, tokens in eligible.items()
                if len(tokens) >= minimum_needed
            }
            if len(reduced) == len(eligible):
                break
            eligible = reduced
        target_types = sorted(eligible)
        if len(target_types) < 3:
            excluded.append({"stratum": list(stratum), "reason": "fewer_than_three_target_hashes"})
            continue
        replacement_count = len(target_types) - 1
        size = min(len(eligible[target]) for target in target_types)
        size -= size % replacement_count
        if size == 0:
            excluded.append({"stratum": list(stratum), "reason": "insufficient_balanced_tokens"})
            continue
        chosen: dict[tuple[str, ...], list[str]] = {}
        for target in target_types:
            tokens = list(eligible[target])
            rng.shuffle(tokens)
            chosen[target] = tokens[:size]
        for target in target_types:
            replacements = [candidate for candidate in target_types if candidate != target]
            rng.shuffle(replacements)
            for token_index, token in enumerate(chosen[target]):
                replacement = replacements[token_index % len(replacements)]
                selected.add(token)
                mapping[token] = target_vector(replacement)

    residual = 0
    replacements_by_original: dict[tuple[int, ...], set[tuple[int, ...]]] = defaultdict(set)
    for token in selected:
        original = grouped[token][0]["target"]
        residual += int(np.array_equal(original, mapping[token]))
        original_hash = tuple(np.asarray(original, dtype=int).tolist())
        replacement_hash = tuple(np.asarray(mapping[token], dtype=int).tolist())
        replacements_by_original[original_hash].add(replacement_hash)
    if residual:
        raise RuntimeError("target derangement retained original targets")
    replacement_counts = {
        "".join(map(str, original)): len(replacements)
        for original, replacements in sorted(replacements_by_original.items())
    }
    return selected, mapping, {
        "n_selected_tokens": len(selected),
        "n_excluded_tokens": len(grouped) - len(selected),
        "residual_target_matches": residual,
        "replacement_counts_by_original_hash": replacement_counts,
        "minimum_replacements_per_original_hash": min(replacement_counts.values(), default=0),
        "excluded_strata": excluded,
    }


def token_batch_indices(
    examples: list[dict[str, Any]],
    batch_tokens: int,
    seed: int,
    epoch: int,
) -> list[np.ndarray]:
    """Return batches containing complete token groups in deterministic order."""
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, example in enumerate(examples):
        grouped[example["pair_token"]].append(index)
    tokens = sorted(grouped)
    order = np.random.default_rng(seed + epoch).permutation(len(tokens))
    return [
        np.asarray([
            index
            for token_index in order[start:start + batch_tokens]
            for index in grouped[tokens[int(token_index)]]
        ])
        for start in range(0, len(order), batch_tokens)
    ]


def train_fixed_recipe(
    model: DeepSetClassifier,
    examples: list[dict[str, Any]],
    arm: str,
    seed: int,
    epochs: int,
    batch_tokens: int,
    learning_rate: float,
    weight_decay: float,
) -> tuple[list[dict[str, float]], dict[str, Any]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    history: list[dict[str, float]] = []
    model.train()
    for epoch in range(epochs):
        losses: list[float] = []
        grad_norms: list[float] = []
        batches = token_batch_indices(examples, batch_tokens, seed, epoch)
        for batch_index, indices in enumerate(batches):
            points, mask, targets = collate_examples(
                examples,
                indices,
                point_seed=seed * 1_000_000 + epoch * 10_000 + batch_index,
            )
            optimizer.zero_grad(set_to_none=True)
            loss = matched_loss(arm, model(points, mask), targets)
            loss.backward()
            squared = sum(
                float(parameter.grad.detach().square().sum())
                for parameter in model.parameters()
                if parameter.grad is not None
            )
            optimizer.step()
            losses.append(float(loss.detach()))
            grad_norms.append(squared ** 0.5)
        history.append({
            "epoch": float(epoch + 1),
            "mean_loss": float(np.mean(losses)),
            "mean_grad_norm": float(np.mean(grad_norms)),
        })
    return history, optimizer.state_dict()


def select_smoke_tau(
    examples: list[dict[str, Any]],
    logits: np.ndarray,
    arm: str,
    tau_grid: Iterable[float],
) -> tuple[float, dict[str, Any]]:
    """Choose a diagnostic-only threshold by token-averaged exact-set rate."""
    candidates = []
    for tau in tau_grid:
        metrics = smoke_metrics(examples, logits, arm, tau=float(tau))
        overall = metrics["overall"]
        candidates.append((
            float(tau),
            float(overall["exact_set"]),
            float(overall["width"]),
            metrics,
        ))
    chosen = max(candidates, key=lambda row: (row[1], -row[2], row[0]))
    return chosen[0], {
        "selection_scope": "SMOKE_DIAGNOSTIC_ONLY",
        "objective": "token_mean_exact_set_then_min_width_then_max_tau",
        "tau": chosen[0],
        "exact_set": chosen[1],
        "mean_width": chosen[2],
    }


def smoke_metrics(
    examples: list[dict[str, Any]],
    logits: np.ndarray,
    arm: str,
    tau: float = 0.5,
) -> dict[str, Any]:
    probabilities = (
        torch.softmax(torch.from_numpy(logits), dim=-1).numpy()
        if arm == "softmax_partial"
        else torch.sigmoid(torch.from_numpy(logits)).numpy()
    )
    fixture_rows: list[dict[str, Any]] = []
    for example, probability in zip(examples, probabilities, strict=True):
        target = example["target"].astype(bool)
        top1 = int(np.argmax(probability))
        predicted = probability >= tau
        width = int(predicted.sum())
        intersection = int(np.logical_and(predicted, target).sum())
        incompatible = int(np.logical_and(predicted, ~target).sum())
        fixture_rows.append({
            "pair_token": example["pair_token"],
            "cardinality": int(target.sum()),
            "top1_compatible": float(target[top1]),
            "set_recall": float(intersection / target.sum()),
            "complete_coverage": float(np.all(~target | predicted)),
            "exact_set": float(np.array_equal(predicted, target)),
            "incompatible_fraction": float(incompatible / max(width, 1)),
            "any_incompatible": float(incompatible > 0),
            "width": float(width),
            "empty": float(width == 0),
        })
    by_token: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in fixture_rows:
        by_token[row["pair_token"]].append(row)
    metric_names = [
        "top1_compatible", "set_recall", "complete_coverage", "exact_set",
        "incompatible_fraction", "any_incompatible", "width", "empty",
    ]
    token_rows = []
    for token, rows in by_token.items():
        token_rows.append({
            "pair_token": token,
            "cardinality": rows[0]["cardinality"],
            **{name: float(np.mean([row[name] for row in rows])) for name in metric_names},
        })
    token_probabilities = []
    token_targets = []
    by_token_indices: dict[str, list[int]] = defaultdict(list)
    for index, example in enumerate(examples):
        by_token_indices[example["pair_token"]].append(index)
    for token in sorted(by_token_indices):
        indices = by_token_indices[token]
        targets = np.stack([examples[index]["target"] for index in indices])
        if not np.all(targets == targets[0]):
            raise ValueError(f"target differs across canonical views for {token}")
        token_targets.append(targets[0])
        token_probabilities.append(probabilities[indices].mean(axis=0))
    token_targets_array = np.stack(token_targets)
    token_probabilities_array = np.stack(token_probabilities)
    family_auc = []
    family_ap = []
    for family_index in range(len(CATALOG_FAMILIES)):
        truth = token_targets_array[:, family_index]
        score = token_probabilities_array[:, family_index]
        family_auc.append(
            float(roc_auc_score(truth, score)) if len(np.unique(truth)) == 2 else float("nan")
        )
        family_ap.append(
            float(average_precision_score(truth, score)) if truth.sum() > 0 else float("nan")
        )
    overall = {name: float(np.mean([row[name] for row in token_rows])) for name in metric_names}
    overall.update({
        "membership_macro_auc": float(np.nanmean(family_auc)),
        "membership_macro_ap": float(np.nanmean(family_ap)),
        "membership_auc_by_family": dict(zip(CATALOG_FAMILIES, family_auc, strict=True)),
        "membership_ap_by_family": dict(zip(CATALOG_FAMILIES, family_ap, strict=True)),
    })
    by_cardinality = {
        str(cardinality): {
            "n_pair_tokens": sum(row["cardinality"] == cardinality for row in token_rows),
            **{
                name: float(np.mean([row[name] for row in token_rows if row["cardinality"] == cardinality]))
                for name in metric_names
            },
        }
        for cardinality in sorted({row["cardinality"] for row in token_rows})
    }
    return {"n_pair_tokens": len(token_rows), "overall": overall, "by_cardinality": by_cardinality}


def logits_payload(examples: list[dict[str, Any]], logits: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "fixture_id": np.asarray([example["fixture_id"] for example in examples]),
        "pair_token": np.asarray([example["pair_token"] for example in examples]),
        "target": np.stack([example["target"] for example in examples]),
        "logits": np.asarray(logits, dtype=np.float32),
    }


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(payload).hexdigest()
