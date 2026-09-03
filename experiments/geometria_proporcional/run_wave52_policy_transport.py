#!/usr/bin/env python3
"""Run the Wave 52 CPU-only ordinal policy-transport development smoke."""

from __future__ import annotations

import argparse
import copy
import json
import os
import resource
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from itertools import permutations
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.special import expit
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.nn import functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from geometria_proporcional.wave49_schema import sha256_file, write_json  # noqa: E402
from geometria_proporcional.wave50_model import collate_examples  # noqa: E402
from geometria_proporcional.wave50_neural import (  # noqa: E402
    TARGET_COMPATIBILITY_DISTANCE,
    TARGET_SCHEMA_VERSION,
    fit_normalizer,
    load_labeled_records,
    prepare_examples,
    select_smoke_tau,
    split_tokens,
    stable_hash,
    stratified_token_subset,
)
from geometria_proporcional.wave51_factored import (  # noqa: E402
    DualHeadDeepSet,
    predict_dual_logits,
)
from geometria_proporcional.wave52_policy import (  # noqa: E402
    ContextualPolicyDeepSet,
    SharedFamilyReader,
    authorized_actions,
    constrained_regret,
    explicit_set_actions,
    policy_fold,
    score_composition_actions,
    validate_policy_groups,
)

SYSTEM_ARMS = (
    "explicit_set_policy",
    "learned_reader_same_set",
    "direct_contextual_choice",
    "joint_set_contextual_choice",
    "utility_ignored",
    "score_composition",
    "oracle_set_then_utility",
    "explicit_context_masked_eval",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wave50-dir", type=Path, required=True)
    parser.add_argument("--wave51-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT
        / "experiments/geometria_proporcional/configs/wave52_policy_transport.json",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def validate_output_path(output_dir: Path, inputs: list[Path]) -> None:
    output = output_dir.resolve()
    root = (REPO_ROOT / "data/geometria_proporcional").resolve()
    if root not in output.parents:
        raise ValueError(f"output must be a child of {root}")
    for raw in inputs:
        source = raw.resolve()
        if output == source or output in source.parents or source in output.parents:
            raise ValueError("output must be disjoint from all input trees")


def assert_no_lockbox(paths: list[Path]) -> None:
    for path in paths:
        resolved = path.resolve(strict=True)
        if any("lockbox" in part.lower() for part in resolved.parts):
            raise ValueError(f"Wave 52 input resolves to lockbox content: {resolved}")


def require_hash(path: Path, expected: str) -> dict[str, str]:
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"source binding mismatch for {path}: {actual} != {expected}")
    return {"path": str(path), "sha256": actual}


def validate_source_binding(
    wave50: Path, wave51: Path, config: dict
) -> list[dict[str, str]]:
    binding = config["source_binding"]
    checked = [
        require_hash(
            wave50 / "execution_manifest.json",
            binding["wave50"]["execution_manifest_sha256"],
        ),
        require_hash(
            wave50 / "training_manifest.json",
            binding["wave50"]["training_manifest_sha256"],
        ),
    ]
    for relative, expected in sorted(binding["wave50"]["files"].items()):
        checked.append(require_hash(wave50 / relative, expected))
    checked.extend(
        [
            require_hash(
                wave51 / "package_manifest.json",
                binding["wave51"]["package_manifest_sha256"],
            ),
            require_hash(
                wave51 / "normalizer.npz", binding["wave51"]["normalizer_sha256"]
            ),
            require_hash(
                wave51 / "split_manifest.json",
                binding["wave51"]["split_manifest_sha256"],
            ),
        ]
    )
    for seed, expected in sorted(binding["wave51"]["checkpoints"].items()):
        checked.append(
            require_hash(
                wave51 / "checkpoints" / f"seed{seed}__sigmoid_only.pt", expected
            )
        )
    for filename, expected in sorted(binding["wave51"]["logits"].items()):
        checked.append(require_hash(wave51 / "logits" / filename, expected))
    assert_no_lockbox([Path(row["path"]) for row in checked])
    return checked


def execution_sources(config_path: Path) -> list[dict[str, str]]:
    paths = [
        Path(__file__).resolve(),
        (REPO_ROOT / "src/geometria_proporcional/wave52_policy.py").resolve(),
        (REPO_ROOT / "src/geometria_proporcional/wave51_factored.py").resolve(),
        (REPO_ROOT / "src/geometria_proporcional/wave50_model.py").resolve(),
        (REPO_ROOT / "src/geometria_proporcional/wave50_neural.py").resolve(),
        (REPO_ROOT / "src/geometria_proporcional/wave49_generator.py").resolve(),
        (
            REPO_ROOT / "Biblioteca/Geometria_Proporcional_Ground_Truth/waves/"
            "WAVE_52_UTILITY_CONDITIONED_POLICY_TRANSPORT_PLAN.md"
        ).resolve(),
        config_path.resolve(),
    ]
    result = []
    for path in paths:
        relative = path.relative_to(REPO_ROOT)
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", str(relative)],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        if tracked.returncode != 0:
            raise RuntimeError(f"execution source is not tracked: {relative}")
        dirty = subprocess.run(
            ["git", "diff", "--quiet", "HEAD", "--", str(relative)], cwd=REPO_ROOT
        )
        if dirty.returncode != 0:
            raise RuntimeError(f"execution source differs from HEAD: {relative}")
        result.append({"path": str(relative), "sha256": sha256_file(path)})
    return result


def group_examples(examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for example in examples:
        grouped[example["pair_token"]].append(example)
    result = []
    for token in sorted(grouped):
        views = grouped[token]
        targets = {tuple(np.asarray(view["target"], dtype=int)) for view in views}
        strata = {view["design_stratum"] for view in views}
        if len(targets) != 1 or len(strata) != 1:
            raise ValueError(f"inconsistent token group: {token}")
        result.append(
            {
                "pair_token": token,
                "cluster_id": token,
                "views": views,
                "target": np.asarray(views[0]["target"], dtype=np.float32),
                "design_stratum": views[0]["design_stratum"],
                "cardinality": int(np.asarray(views[0]["target"]).sum()),
            }
        )
    return result


def token_logits(
    examples: list[dict[str, Any]], logits: np.ndarray
) -> tuple[list[str], np.ndarray]:
    grouped: dict[str, list[np.ndarray]] = defaultdict(list)
    for example, row in zip(examples, logits, strict=True):
        grouped[example["pair_token"]].append(np.asarray(row, dtype=np.float64))
    tokens = sorted(grouped)
    return tokens, np.stack([np.mean(grouped[token], axis=0) for token in tokens])


def batch_token_latent(
    model: ContextualPolicyDeepSet,
    groups: list[dict[str, Any]],
    indices: np.ndarray,
    point_seed: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    flat: list[dict[str, Any]] = []
    owners: list[int] = []
    for owner, index in enumerate(indices):
        views = groups[int(index)]["views"]
        flat.extend(views)
        owners.extend([owner] * len(views))
    points, mask, _ = collate_examples(
        flat, np.arange(len(flat)), point_seed=point_seed
    )
    latent_views = model.encode(points, mask)
    owner_tensor = torch.as_tensor(owners, dtype=torch.long)
    latent = torch.zeros(
        (len(indices), latent_views.shape[1]), dtype=latent_views.dtype
    )
    latent.index_add_(0, owner_tensor, latent_views)
    counts = (
        torch.bincount(owner_tensor, minlength=len(indices))
        .to(latent.dtype)
        .unsqueeze(1)
    )
    latent = latent / counts
    targets = torch.from_numpy(
        np.stack([groups[int(index)]["target"] for index in indices])
    )
    return latent, targets


def train_contextual_model(
    model: ContextualPolicyDeepSet,
    groups: list[dict[str, Any]],
    input_utilities: np.ndarray,
    target_utilities: np.ndarray,
    objective: str,
    seed: int,
    fold: int,
    config: dict,
) -> tuple[list[dict[str, float]], dict[str, Any]]:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    input_utility = torch.from_numpy(input_utilities.astype(np.float32))
    all_targets = np.stack([group["target"] for group in groups])
    choice_valid = all_targets.astype(bool).any(axis=1)
    target_actions = np.full((len(groups), len(target_utilities)), -1, dtype=np.int64)
    target_actions[choice_valid] = authorized_actions(
        all_targets[choice_valid], target_utilities
    )
    history = []
    for epoch in range(int(config["model_epochs"])):
        order = np.random.default_rng(
            seed * 100_000 + fold * 1_000 + epoch
        ).permutation(len(groups))
        losses = []
        for batch_index, start in enumerate(
            range(0, len(order), int(config["batch_tokens"]))
        ):
            indices = order[start : start + int(config["batch_tokens"])]
            latent, targets = batch_token_latent(
                model,
                groups,
                indices,
                point_seed=seed * 1_000_000
                + fold * 100_000
                + epoch * 1_000
                + batch_index,
            )
            set_logits = model.set_head(latent)
            scores = model.policy_reader(
                model.choice_evidence_head(latent), input_utility
            )
            valid_batch = choice_valid[indices]
            if np.any(valid_batch):
                action = torch.from_numpy(target_actions[indices][valid_batch]).reshape(
                    -1
                )
                choice_loss = F.cross_entropy(
                    scores[valid_batch].reshape(-1, 4), action
                )
            else:
                choice_loss = scores.sum() * 0.0
            if objective == "choice":
                loss = choice_loss
            elif objective == "joint":
                loss = (
                    F.binary_cross_entropy_with_logits(set_logits, targets)
                    + choice_loss
                )
            else:
                raise ValueError(objective)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        history.append({"epoch": epoch + 1, "mean_loss": float(np.mean(losses))})
    return history, optimizer.state_dict()


@torch.no_grad()
def predict_contextual_scores(
    model: ContextualPolicyDeepSet,
    groups: list[dict[str, Any]],
    utilities: np.ndarray,
    batch_tokens: int,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    utility = torch.from_numpy(utilities.astype(np.float32))
    set_rows, score_rows = [], []
    for start in range(0, len(groups), batch_tokens):
        indices = np.arange(start, min(start + batch_tokens, len(groups)))
        latent, _ = batch_token_latent(model, groups, indices, point_seed=None)
        set_rows.append(model.set_head(latent).cpu().numpy())
        score_rows.append(
            model.policy_reader(model.choice_evidence_head(latent), utility)
            .cpu()
            .numpy()
        )
    return np.concatenate(set_rows), np.concatenate(score_rows)


def train_frozen_reader(
    set_logits: np.ndarray,
    targets: np.ndarray,
    utilities: np.ndarray,
    seed: int,
    fold: int,
    config: dict,
) -> tuple[SharedFamilyReader, list[dict[str, float]], dict[str, Any]]:
    torch.manual_seed(seed * 10_000 + fold + 52)
    reader = SharedFamilyReader()
    optimizer = torch.optim.AdamW(
        reader.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    valid = targets.astype(bool).any(axis=1)
    action = np.full((len(targets), len(utilities)), -1, dtype=np.int64)
    action[valid] = authorized_actions(targets[valid], utilities)
    utility = torch.from_numpy(utilities.astype(np.float32))
    evidence = torch.from_numpy(set_logits.astype(np.float32))
    history = []
    for epoch in range(int(config["reader_epochs"])):
        order = np.random.default_rng(
            seed * 100_000 + fold * 1_000 + epoch
        ).permutation(len(targets))
        losses = []
        for start in range(0, len(order), int(config["batch_tokens"])):
            indices = order[start : start + int(config["batch_tokens"])]
            selected = indices[valid[indices]]
            if len(selected) == 0:
                continue
            scores = reader(evidence[selected], utility)
            loss = F.cross_entropy(
                scores.reshape(-1, 4), torch.from_numpy(action[selected]).reshape(-1)
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        history.append({"epoch": epoch + 1, "mean_loss": float(np.mean(losses))})
    return reader, history, optimizer.state_dict()


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    history: list[dict[str, float]],
    optimizer_state: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "history": history,
            "optimizer_state": optimizer_state,
            **metadata,
        },
        path,
    )


def policy_metrics_by_token(
    groups: list[dict[str, Any]],
    utilities: np.ndarray,
    actions: np.ndarray,
    incompatible_penalty: float,
) -> list[dict[str, Any]]:
    targets = np.stack([group["target"] for group in groups]).astype(bool)
    valid = targets.any(axis=1)
    truth = np.full(actions.shape, -1, dtype=np.int64)
    regret = np.full(actions.shape, np.nan, dtype=np.float64)
    compatible = np.zeros(actions.shape, dtype=bool)
    if np.any(valid):
        truth[valid] = authorized_actions(targets[valid], utilities)
        regret[valid] = constrained_regret(
            actions[valid], targets[valid], utilities, incompatible_penalty
        )
        compatible[valid] = targets[valid][
            np.arange(int(valid.sum()))[:, None], actions[valid]
        ]
    rows = []
    for index, group in enumerate(groups):
        rows.append(
            {
                "pair_token": group["pair_token"],
                "design_stratum": group["design_stratum"],
                "cardinality": group["cardinality"],
                "cluster_id": group["cluster_id"],
                "choice_valid": bool(valid[index]),
                "n_policies": utilities.shape[0],
                "action_accuracy": (
                    float(np.mean(actions[index] == truth[index]))
                    if valid[index]
                    else float("nan")
                ),
                "compatible_action_rate": (
                    float(np.mean(compatible[index])) if valid[index] else float("nan")
                ),
                "restricted_regret": (
                    float(np.mean(regret[index])) if valid[index] else float("nan")
                ),
                "worst_restricted_regret": (
                    float(np.max(regret[index])) if valid[index] else float("nan")
                ),
            }
        )
    return rows


def append_metric_rows(
    store: dict[str, dict[str, list[dict[str, float]]]], arm: str, rows: list[dict]
) -> None:
    for row in rows:
        store[arm][row["pair_token"]].append(row)


def aggregate_metric_store(
    store: dict[str, dict[str, list[dict[str, float]]]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, float]]]:
    per_token: dict[str, list[dict[str, Any]]] = {}
    summaries: dict[str, dict[str, float]] = {}
    metric_names = (
        "action_accuracy",
        "compatible_action_rate",
        "restricted_regret",
        "worst_restricted_regret",
    )
    for arm, by_token in store.items():
        rows = []
        for token in sorted(by_token):
            parts = by_token[token]
            n_policy = sum(int(row["n_policies"]) for row in parts)
            rows.append(
                {
                    "pair_token": token,
                    "cluster_id": parts[0]["cluster_id"],
                    "design_stratum": parts[0]["design_stratum"],
                    "cardinality": int(parts[0]["cardinality"]),
                    "choice_valid": bool(parts[0]["choice_valid"]),
                    "n_policies": n_policy,
                    **{
                        name: (
                            float(max(row[name] for row in parts))
                            if name == "worst_restricted_regret"
                            else float(
                                sum(row[name] * row["n_policies"] for row in parts)
                                / n_policy
                            )
                        )
                        for name in metric_names
                    },
                }
            )
        per_token[arm] = rows
        primary = [
            row
            for row in rows
            if row["design_stratum"] == "NEAR_RIVAL" and row["cardinality"] >= 2
        ]
        summaries[arm] = {
            "n_pair_tokens": len(primary),
            "n_policy_evaluations": sum(row["n_policies"] for row in primary),
            **{
                name: float(np.mean([row[name] for row in primary]))
                for name in metric_names
            },
        }
    return per_token, summaries


def set_integrity_metrics(
    groups: list[dict[str, Any]], logits: np.ndarray, tau: float
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    targets = np.stack([group["target"] for group in groups]).astype(bool)
    probabilities = expit(np.asarray(logits, dtype=np.float64))
    predicted = probabilities >= float(tau)
    rows = []
    for index, group in enumerate(groups):
        cardinality = int(targets[index].sum())
        width = int(predicted[index].sum())
        incompatible = int(np.logical_and(predicted[index], ~targets[index]).sum())
        rows.append(
            {
                "pair_token": group["pair_token"],
                "cluster_id": group["cluster_id"],
                "design_stratum": group["design_stratum"],
                "cardinality": cardinality,
                "set_recall": (
                    float(
                        np.logical_and(predicted[index], targets[index]).sum()
                        / cardinality
                    )
                    if cardinality
                    else float("nan")
                ),
                "width": float(width),
                "any_incompatible": float(incompatible > 0),
                "incompatible_fraction": float(incompatible / max(width, 1)),
                "fallback": float(width == 0),
            }
        )
    family_auc, family_ap = [], []
    for family in range(targets.shape[1]):
        truth = targets[:, family].astype(int)
        score = probabilities[:, family]
        family_auc.append(
            float(roc_auc_score(truth, score))
            if len(np.unique(truth)) == 2
            else float("nan")
        )
        family_ap.append(
            float(average_precision_score(truth, score))
            if truth.sum()
            else float("nan")
        )
    valid_rows = [row for row in rows if row["cardinality"] > 0]
    summary = {
        "n_pair_tokens": len(rows),
        "n_out_of_catalog_tokens": len(rows) - len(valid_rows),
        "set_recall": float(np.nanmean([row["set_recall"] for row in rows])),
        "width": float(np.mean([row["width"] for row in rows])),
        "any_incompatible": float(np.mean([row["any_incompatible"] for row in rows])),
        "incompatible_fraction": float(
            np.mean([row["incompatible_fraction"] for row in rows])
        ),
        "fallback_rate": float(np.mean([row["fallback"] for row in rows])),
        "membership_macro_auc": float(np.nanmean(family_auc)),
        "membership_macro_ap": float(np.nanmean(family_ap)),
    }
    return rows, summary


def policy_rows(
    groups: list[dict[str, Any]],
    utilities: np.ndarray,
    policy_indices: tuple[int, ...],
    actions: np.ndarray,
    arm: str,
    fold: int,
    regime: str,
    penalty: float,
    seed: int | str = "ensemble",
) -> list[dict[str, Any]]:
    targets = np.stack([group["target"] for group in groups]).astype(bool)
    valid = targets.any(axis=1)
    truth = authorized_actions(targets[valid], utilities)
    regret = constrained_regret(actions[valid], targets[valid], utilities, penalty)
    compatible = targets[valid][np.arange(int(valid.sum()))[:, None], actions[valid]]
    primary = np.asarray(
        [
            group["design_stratum"] == "NEAR_RIVAL" and group["cardinality"] >= 2
            for group in groups
        ]
    )[valid]
    rows = []
    for local, policy_index in enumerate(policy_indices):
        rows.append(
            {
                "fold": fold,
                "regime": regime,
                "arm": arm,
                "seed": seed,
                "policy_index": int(policy_index),
                "n_pair_tokens": int(primary.sum()),
                "action_accuracy": float(
                    np.mean((actions[valid, local] == truth[:, local])[primary])
                ),
                "compatible_action_rate": float(np.mean(compatible[:, local][primary])),
                "restricted_regret": float(np.mean(regret[:, local][primary])),
            }
        )
    return rows


def bootstrap_metric_ci(
    rows: list[dict[str, Any]], metric: str, n_boot: int, seed: int
) -> dict[str, float]:
    primary = [
        row
        for row in rows
        if row["design_stratum"] == "NEAR_RIVAL" and row["cardinality"] >= 2
    ]
    clusters: dict[str, list[float]] = defaultdict(list)
    for row in primary:
        clusters[row["cluster_id"]].append(float(row[metric]))
    values = np.asarray([np.mean(clusters[key]) for key in sorted(clusters)])
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for index in range(n_boot):
        sample = rng.integers(0, len(values), len(values))
        draws[index] = float(np.mean(values[sample]))
    low, high = np.percentile(draws, [2.5, 97.5])
    return {
        "mean": float(np.mean(values)),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "n_clusters": len(values),
    }


def bootstrap_delta(
    rows_a: list[dict[str, Any]],
    rows_b: list[dict[str, Any]],
    metric: str,
    n_boot: int,
    seed: int,
) -> dict[str, float]:
    a = {row["pair_token"]: row for row in rows_a}
    b = {row["pair_token"]: row for row in rows_b}
    tokens = sorted(
        token
        for token in a.keys() & b.keys()
        if a[token]["design_stratum"] == "NEAR_RIVAL" and a[token]["cardinality"] >= 2
    )
    clusters: dict[str, list[str]] = defaultdict(list)
    for token in tokens:
        if a[token]["cluster_id"] != b[token]["cluster_id"]:
            raise ValueError(f"cluster mismatch for {token}")
        clusters[a[token]["cluster_id"]].append(token)
    cluster_ids = sorted(clusters)
    observed = np.asarray(
        [
            np.mean(
                [a[token][metric] - b[token][metric] for token in clusters[cluster]]
            )
            for cluster in cluster_ids
        ]
    )
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for index in range(n_boot):
        sample = rng.integers(0, len(cluster_ids), len(cluster_ids))
        draws[index] = float(np.mean(observed[sample]))
    low, high = np.percentile(draws, [2.5, 97.5])
    return {
        "n_pair_tokens": len(tokens),
        "n_clusters": len(cluster_ids),
        "mean_diff": float(np.mean(observed)),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "fraction_positive": float(np.mean(draws > 0)),
    }


def counterfactual_success(
    groups: list[dict[str, Any]], utilities: np.ndarray, actions: np.ndarray
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    targets = np.stack([group["target"] for group in groups]).astype(bool)
    truth = authorized_actions(targets, utilities)
    intervention_events = []
    joint_events = []
    mapping = []
    for row, group in enumerate(groups):
        if group["design_stratum"] != "NEAR_RIVAL" or group["cardinality"] < 2:
            continue
        for left in range(utilities.shape[0]):
            right = next(
                (
                    candidate
                    for candidate in range(utilities.shape[0])
                    if truth[row, candidate] != truth[row, left]
                ),
                None,
            )
            if right is None:
                continue
            intervention = float(
                actions[row, right] != actions[row, left]
                and actions[row, right] == truth[row, right]
            )
            joint = float(
                actions[row, left] == truth[row, left]
                and actions[row, right] == truth[row, right]
                and actions[row, left] != actions[row, right]
            )
            intervention_events.append(intervention)
            joint_events.append(joint)
            mapping.append(
                {
                    "pair_token": group["pair_token"],
                    "cluster_id": group["cluster_id"],
                    "policy_original_local": left,
                    "policy_counterfactual_local": right,
                    "target_original": int(truth[row, left]),
                    "target_counterfactual": int(truth[row, right]),
                    "action_original": int(actions[row, left]),
                    "action_counterfactual": int(actions[row, right]),
                    "intervention_success": intervention,
                    "joint_correct_success": joint,
                }
            )
    return {
        "n_pair_tokens": len({row["pair_token"] for row in mapping}),
        "n_counterfactual_pairs": len(mapping),
        "success_rate": (
            float(np.mean(intervention_events)) if intervention_events else float("nan")
        ),
        "joint_correct_rate": (
            float(np.mean(joint_events)) if joint_events else float("nan")
        ),
    }, mapping


def replay_saved_arrays(
    output_dir: Path,
    split_groups: dict[str, list[dict[str, Any]]],
    config: dict,
    levels: np.ndarray,
    tau: float,
    lambda_selection: dict[str, Any],
) -> tuple[
    dict[str, Any],
    dict[tuple[int, str], dict[str, np.ndarray]],
    dict[tuple[int, str], dict[str, np.ndarray]],
]:
    """Reload last-epoch checkpoints and reproduce all saved learned arrays exactly."""
    checks: list[dict[str, Any]] = []
    replayed_actions: dict[tuple[int, str], dict[str, np.ndarray]] = {}
    replayed_seed_actions: dict[tuple[int, str], dict[str, np.ndarray]] = {}
    monitor_groups = split_groups["val_monitor"]
    for fold in range(3):
        for regime in ("policy_seen", "policy_shift"):
            path = output_dir / "raw_predictions" / f"fold{fold}__{regime}__monitor.npz"
            with np.load(path) as raw:
                utility = raw["utility"]
                ensemble_set = raw["ensemble_set_logits"]
                reader_actual_rows = []
                contextual_actual: dict[str, list[np.ndarray]] = {
                    arm: []
                    for arm in (
                        "direct_contextual_choice",
                        "joint_set_contextual_choice",
                        "utility_ignored",
                    )
                }
                for seed_index, seed in enumerate(config["seeds"]):
                    checkpoint = torch.load(
                        output_dir
                        / "checkpoints"
                        / f"fold{fold}__seed{seed}__learned_reader_same_set.pt",
                        map_location="cpu",
                        weights_only=False,
                    )
                    reader = SharedFamilyReader()
                    reader.load_state_dict(checkpoint["model_state"])
                    with torch.no_grad():
                        actual = reader(
                            torch.from_numpy(ensemble_set.astype(np.float32)),
                            torch.from_numpy(utility.astype(np.float32)),
                        ).numpy()
                    expected = raw["learned_reader_scores_by_seed"][seed_index]
                    reader_actual_rows.append(actual)
                    checks.append(
                        {
                            "fold": fold,
                            "regime": regime,
                            "seed": int(seed),
                            "arm": "learned_reader_same_set",
                            "array_exact": bool(np.array_equal(actual, expected)),
                        }
                    )
                    for arm, score_key, set_key in (
                        (
                            "direct_contextual_choice",
                            "direct_scores_by_seed",
                            "direct_set_logits_by_seed",
                        ),
                        (
                            "joint_set_contextual_choice",
                            "joint_scores_by_seed",
                            "joint_set_logits_by_seed",
                        ),
                        (
                            "utility_ignored",
                            "ignored_scores_by_seed",
                            "ignored_set_logits_by_seed",
                        ),
                    ):
                        checkpoint = torch.load(
                            output_dir
                            / "checkpoints"
                            / f"fold{fold}__seed{seed}__{arm}.pt",
                            map_location="cpu",
                            weights_only=False,
                        )
                        model = ContextualPolicyDeepSet()
                        model.load_state_dict(checkpoint["model_state"])
                        input_utility = (
                            np.repeat(levels[None, :], len(utility), axis=0)
                            if arm == "utility_ignored"
                            else utility
                        )
                        actual_set, actual_scores = predict_contextual_scores(
                            model,
                            monitor_groups,
                            input_utility,
                            int(config["inference_batch_tokens"]),
                        )
                        exact = bool(
                            np.array_equal(actual_scores, raw[score_key][seed_index])
                            and np.array_equal(actual_set, raw[set_key][seed_index])
                        )
                        checks.append(
                            {
                                "fold": fold,
                                "regime": regime,
                                "seed": int(seed),
                                "arm": arm,
                                "array_exact": exact,
                            }
                        )
                        contextual_actual[arm].append(actual_scores)
                targets = raw["target"].astype(bool)
                actions = {
                    "explicit_set_policy": explicit_set_actions(
                        expit(ensemble_set), utility, tau
                    )[0],
                    "learned_reader_same_set": np.argmax(
                        np.mean(np.stack(reader_actual_rows), axis=0), axis=-1
                    ),
                    "direct_contextual_choice": np.argmax(
                        np.mean(
                            np.stack(contextual_actual["direct_contextual_choice"]),
                            axis=0,
                        ),
                        axis=-1,
                    ),
                    "joint_set_contextual_choice": np.argmax(
                        np.mean(
                            np.stack(contextual_actual["joint_set_contextual_choice"]),
                            axis=0,
                        ),
                        axis=-1,
                    ),
                    "utility_ignored": np.argmax(
                        np.mean(np.stack(contextual_actual["utility_ignored"]), axis=0),
                        axis=-1,
                    ),
                    "score_composition": score_composition_actions(
                        ensemble_set,
                        utility,
                        lambda_selection[str(fold)]["selected"]["weight"],
                    ),
                    "oracle_set_then_utility": authorized_actions(targets, utility),
                    "explicit_context_masked_eval": explicit_set_actions(
                        expit(ensemble_set),
                        np.repeat(levels[None, :], len(utility), axis=0),
                        tau,
                    )[0],
                }
                for arm, action in actions.items():
                    exact = bool(np.array_equal(action, raw[f"ensemble_action__{arm}"]))
                    checks.append(
                        {
                            "fold": fold,
                            "regime": regime,
                            "seed": "ensemble",
                            "arm": f"{arm}__action",
                            "array_exact": exact,
                        }
                    )
                frozen_by_seed = raw["per_seed_frozen_set_logits"]
                per_seed_actions = {
                    "explicit_set_policy": np.stack(
                        [
                            explicit_set_actions(expit(row), utility, tau)[0]
                            for row in frozen_by_seed
                        ]
                    ),
                    "learned_reader_same_set": np.argmax(
                        np.stack(reader_actual_rows), axis=-1
                    ),
                    "direct_contextual_choice": np.argmax(
                        np.stack(contextual_actual["direct_contextual_choice"]), axis=-1
                    ),
                    "joint_set_contextual_choice": np.argmax(
                        np.stack(contextual_actual["joint_set_contextual_choice"]),
                        axis=-1,
                    ),
                    "utility_ignored": np.argmax(
                        np.stack(contextual_actual["utility_ignored"]), axis=-1
                    ),
                    "score_composition": np.stack(
                        [
                            score_composition_actions(
                                row,
                                utility,
                                lambda_selection[str(fold)]["selected"]["weight"],
                            )
                            for row in frozen_by_seed
                        ]
                    ),
                    "oracle_set_then_utility": np.repeat(
                        actions["oracle_set_then_utility"][None, ...],
                        len(config["seeds"]),
                        axis=0,
                    ),
                    "explicit_context_masked_eval": np.stack(
                        [
                            explicit_set_actions(
                                expit(row),
                                np.repeat(levels[None, :], len(utility), axis=0),
                                tau,
                            )[0]
                            for row in frozen_by_seed
                        ]
                    ),
                }
                for arm, action in per_seed_actions.items():
                    exact = bool(np.array_equal(action, raw[f"per_seed_action__{arm}"]))
                    checks.append(
                        {
                            "fold": fold,
                            "regime": regime,
                            "seed": "all",
                            "arm": f"{arm}__per_seed_action",
                            "array_exact": exact,
                        }
                    )
                replayed_actions[(fold, regime)] = actions
                replayed_seed_actions[(fold, regime)] = per_seed_actions
    for fold in range(3):
        for split_name, policy_key in (
            ("train", "policy_train"),
            ("val_threshold", "policy_val"),
        ):
            eval_groups = split_groups[split_name]
            path = (
                output_dir
                / "raw_predictions"
                / f"fold{fold}__{split_name}__{policy_key}.npz"
            )
            with np.load(path) as raw:
                utility = raw["utility"]
                ensemble_set = raw["ensemble_set_logits"]
                for seed_index, seed in enumerate(config["seeds"]):
                    checkpoint = torch.load(
                        output_dir
                        / "checkpoints"
                        / f"fold{fold}__seed{seed}__learned_reader_same_set.pt",
                        map_location="cpu",
                        weights_only=False,
                    )
                    reader = SharedFamilyReader()
                    reader.load_state_dict(checkpoint["model_state"])
                    with torch.no_grad():
                        actual = reader(
                            torch.from_numpy(ensemble_set.astype(np.float32)),
                            torch.from_numpy(utility.astype(np.float32)),
                        ).numpy()
                    checks.append(
                        {
                            "fold": fold,
                            "regime": split_name,
                            "seed": int(seed),
                            "arm": "learned_reader_same_set",
                            "array_exact": bool(
                                np.array_equal(
                                    actual,
                                    raw["learned_reader_scores_by_seed"][seed_index],
                                )
                            ),
                        }
                    )
                    for arm in (
                        "direct_contextual_choice",
                        "joint_set_contextual_choice",
                        "utility_ignored",
                    ):
                        checkpoint = torch.load(
                            output_dir
                            / "checkpoints"
                            / f"fold{fold}__seed{seed}__{arm}.pt",
                            map_location="cpu",
                            weights_only=False,
                        )
                        model = ContextualPolicyDeepSet()
                        model.load_state_dict(checkpoint["model_state"])
                        input_utility = (
                            np.repeat(levels[None, :], len(utility), axis=0)
                            if arm == "utility_ignored"
                            else utility
                        )
                        actual_set, actual_scores = predict_contextual_scores(
                            model,
                            eval_groups,
                            input_utility,
                            int(config["inference_batch_tokens"]),
                        )
                        exact = bool(
                            np.array_equal(
                                actual_set,
                                raw[f"{arm}__set_logits_by_seed"][seed_index],
                            )
                            and np.array_equal(
                                actual_scores, raw[f"{arm}__scores_by_seed"][seed_index]
                            )
                        )
                        checks.append(
                            {
                                "fold": fold,
                                "regime": split_name,
                                "seed": int(seed),
                                "arm": arm,
                                "array_exact": exact,
                            }
                        )
    if not all(row["array_exact"] for row in checks):
        failed = [row for row in checks if not row["array_exact"]]
        raise RuntimeError(f"checkpoint replay differs for {failed[:3]}")
    return (
        {
            "status": "PASS",
            "array_exact": True,
            "n_checkpoint_array_checks": len(checks),
            "checks": checks,
        },
        replayed_actions,
        replayed_seed_actions,
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def label_scope_inventory(paths: dict[str, Path]) -> dict[str, dict[str, int]]:
    """Record source-scope exclusions without broadening the inherited Wave 51 dataset."""
    inventory = {}
    for split, path in paths.items():
        counts = {"total": 0, "out_of_catalog": 0, "canonical_in_catalog": 0}
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                counts["total"] += 1
                if row.get("is_out_of_catalog"):
                    counts["out_of_catalog"] += 1
                elif row.get("calibration_population") == "canonical_preserving":
                    counts["canonical_in_catalog"] += 1
        inventory[split] = counts
    return inventory


def write_training_shuffle_manifests(
    output_dir: Path, train_groups: list[dict[str, Any]], config: dict
) -> dict[str, Any]:
    """Persist every deterministic token order and point-shuffle seed used in training."""
    directory = output_dir / "training_shuffles"
    directory.mkdir(parents=True, exist_ok=True)
    n_tokens = len(train_groups)
    batch_tokens = int(config["batch_tokens"])
    n_batches = (n_tokens + batch_tokens - 1) // batch_tokens
    tokens = np.asarray([group["pair_token"] for group in train_groups])
    fixture_ids = np.asarray(
        [[view["fixture_id"] for view in group["views"]] for group in train_groups]
    )
    point_counts = np.asarray(
        [[len(view["features"]) for view in group["views"]] for group in train_groups],
        dtype=np.int64,
    )
    files = []
    for fold in range(3):
        for seed in config["seeds"]:
            contextual_orders = np.stack(
                [
                    np.random.default_rng(
                        int(seed) * 100_000 + fold * 1_000 + epoch
                    ).permutation(n_tokens)
                    for epoch in range(int(config["model_epochs"]))
                ]
            )
            reader_orders = contextual_orders[: int(config["reader_epochs"])].copy()
            point_seeds = np.asarray(
                [
                    [
                        int(seed) * 1_000_000
                        + fold * 100_000
                        + epoch * 1_000
                        + batch_index
                        for batch_index in range(n_batches)
                    ]
                    for epoch in range(int(config["model_epochs"]))
                ],
                dtype=np.int64,
            )
            path = directory / f"fold{fold}__seed{seed}.npz"
            np.savez_compressed(
                path,
                pair_token=tokens,
                fixture_id=fixture_ids,
                point_count=point_counts,
                contextual_token_order=contextual_orders,
                reader_token_order=reader_orders,
                contextual_point_seed=point_seeds,
                batch_tokens=np.asarray(batch_tokens, dtype=np.int64),
                point_permutation_algorithm=np.asarray(
                    "numpy.default_rng(seed).permutation(n)"
                ),
            )
            files.append(str(path.relative_to(output_dir)))
    return {
        "status": "COMPLETE",
        "n_files": len(files),
        "n_tokens": n_tokens,
        "n_contextual_epochs": int(config["model_epochs"]),
        "n_reader_epochs": int(config["reader_epochs"]),
        "files": files,
    }


def write_report(output_dir: Path, summary: dict) -> None:
    metrics = summary["policy_shift_metrics"]
    lines = [
        "# Ola 52 — transporte de política ordinal",
        "",
        "> Smoke CPU sobre train/val históricos abiertos. No leyó lockbox y no decide GO/NO-GO.",
        "",
        "## Población primaria",
        "",
        f"`NEAR_RIVAL` con cardinalidad compatible >=2: `{summary['data']['primary_monitor_tokens']}` pair tokens, "
        f"cada uno evaluado sobre las 24 políticas held-out de tres folds.",
        "",
        "## Resultado",
        "",
        "| sistema | accuracy | compatible | regret restringido | peor regret |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm in SYSTEM_ARMS:
        row = metrics[arm]
        lines.append(
            f"| `{arm}` | {row['action_accuracy']:.3f} | {row['compatible_action_rate']:.3f} | "
            f"{row['restricted_regret']:.3f} | {row['worst_restricted_regret']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Contrastes preregistrados",
            "",
        ]
    )
    for name, value in summary["contrasts"].items():
        if isinstance(value, dict) and "mean_diff" in value:
            lines.append(
                f"- `{name}`: {value['mean_diff']:+.4f}, IC95 "
                f"[{value['ci95_low']:+.4f}, {value['ci95_high']:+.4f}]."
            )
    lines.extend(
        [
            f"- éxito contrafactual del reader explícito: `{summary['counterfactual']['success_rate']:.3f}`.",
            f"- patrón diagnóstico: `{summary['diagnostic_pattern']['policy_transport_promising']}`.",
            "",
            "## Alcance",
            "",
            "Una eventual ventaja del reader explícito es una ventaja de composición conocida del sistema; no prueba "
            "superioridad causal de la representación, utilidad natural ni geometría universal.",
        ]
    )
    (output_dir / "REPORT_WAVE52_POLICY_TRANSPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    wave50 = args.wave50_dir.resolve()
    wave51 = args.wave51_dir.resolve()
    output_dir = args.output_dir.resolve()
    config_path = args.config.resolve()
    validate_output_path(output_dir, [wave50, wave51])
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("status") != "SMOKE_ONLY_OPENED_HISTORICAL":
        raise ValueError("config must declare SMOKE_ONLY_OPENED_HISTORICAL")
    binding_receipt = validate_source_binding(wave50, wave51, config)
    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.force:
            raise SystemExit(
                f"non-empty output exists: {output_dir}; pass --force to replace"
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if torch.cuda.is_initialized():
        raise RuntimeError(
            "Wave 52 smoke must start before any CUDA context is initialized"
        )
    torch.set_num_threads(int(config["cpu_threads"]))
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()

    # Bind executable sources and the label-blind policy panel before authorized labels are read.
    sources = execution_sources(config_path)
    ranks = np.asarray(list(permutations(range(4))), dtype=np.int64)
    groups = validate_policy_groups(ranks, config["policy_groups"])
    levels = np.asarray(config["policy_levels"], dtype=np.float64)
    policies = levels[ranks]
    policy_manifest = {
        "levels": levels.tolist(),
        "rank_permutations": ranks.tolist(),
        "groups": [list(group) for group in groups],
        "folds": [policy_fold(groups, fold) for fold in range(3)],
        "generation_scope": "label-blind-before-authorized-action-derivation",
        "execution_sources": sources,
    }
    write_json(output_dir / "policy_manifest.json", policy_manifest)
    policy_manifest_sha256 = sha256_file(output_dir / "policy_manifest.json")

    protocol = wave50 / "benchmark/protocol_config.json"
    input_paths = [
        wave50 / "benchmark/visible/train.jsonl",
        wave50 / "authorized_labels/train.jsonl",
        wave50 / "benchmark/visible/val.jsonl",
        wave50 / "authorized_labels/val.jsonl",
        protocol,
        config_path,
    ]
    assert_no_lockbox(input_paths)
    source_scope = label_scope_inventory(
        {"train": input_paths[1], "val": input_paths[3]}
    )
    train_all, train_reads = load_labeled_records(
        input_paths[0],
        input_paths[1],
        protocol,
        "train",
        expected_schema=TARGET_SCHEMA_VERSION,
        expected_compatibility_distance=TARGET_COMPATIBILITY_DISTANCE,
    )
    val_all, val_reads = load_labeled_records(
        input_paths[2],
        input_paths[3],
        protocol,
        "val",
        expected_schema=TARGET_SCHEMA_VERSION,
        expected_compatibility_distance=TARGET_COMPATIBILITY_DISTANCE,
    )
    if validate_source_binding(wave50, wave51, config) != binding_receipt:
        raise RuntimeError("source changed while authorized data were loading")
    if execution_sources(config_path) != sources:
        raise RuntimeError(
            "execution source changed while authorized data were loading"
        )
    train_records = stratified_token_subset(
        train_all, int(config["max_train_fixtures"]), int(config["subset_seed"])
    )
    val_records = stratified_token_subset(
        val_all, int(config["max_val_fixtures"]), int(config["subset_seed"]) + 1
    )
    threshold_records, monitor_records = split_tokens(
        val_records, float(config["val_threshold_fraction"]), int(config["split_seed"])
    )
    split_tokens_actual = {
        "train": sorted({row["pair_token"] for row in train_records}),
        "val_threshold": sorted({row["pair_token"] for row in threshold_records}),
        "val_monitor": sorted({row["pair_token"] for row in monitor_records}),
    }
    canonical_split = json.loads(
        (wave51 / "split_manifest.json").read_text(encoding="utf-8")
    )
    if split_tokens_actual != canonical_split:
        raise RuntimeError("Wave 52 token splits do not reproduce Wave 51")
    if set(split_tokens_actual["val_threshold"]) & set(
        split_tokens_actual["val_monitor"]
    ):
        raise RuntimeError("val_threshold and val_monitor overlap")
    write_json(output_dir / "split_manifest.json", split_tokens_actual)
    split_cluster_sets = {
        split: set(tokens) for split, tokens in split_tokens_actual.items()
    }
    if any(
        split_cluster_sets[left] & split_cluster_sets[right]
        for left, right in (
            ("train", "val_threshold"),
            ("train", "val_monitor"),
            ("val_threshold", "val_monitor"),
        )
    ):
        raise RuntimeError("generative clusters overlap across observation splits")
    resampling_manifest = {
        "unit": "pair_token",
        "cluster_rule": "cluster_id equals pair_token",
        "evidence": (
            "wave49_generator creates one pair_token from split+ordinal+family+condition "
            "before emitting canonical representation/covariance views"
        ),
        "generator_source": {
            "path": "src/geometria_proporcional/wave49_generator.py",
            "sha256": sha256_file(
                REPO_ROOT / "src/geometria_proporcional/wave49_generator.py"
            ),
        },
        "split_disjoint": True,
        "mapping": [
            {"pair_token": token, "cluster_id": token, "split": split}
            for split, tokens in split_tokens_actual.items()
            for token in tokens
        ],
    }
    write_json(output_dir / "resampling_manifest.json", resampling_manifest)

    normalizer = fit_normalizer(train_records)
    with np.load(wave51 / "normalizer.npz") as expected:
        if not np.array_equal(normalizer.mean, expected["mean"]) or not np.array_equal(
            normalizer.std, expected["std"]
        ):
            raise RuntimeError("normalizer does not reproduce Wave 51")
    np.savez(output_dir / "normalizer.npz", mean=normalizer.mean, std=normalizer.std)
    train_examples = prepare_examples(train_records, normalizer)
    threshold_examples = prepare_examples(threshold_records, normalizer)
    monitor_examples = prepare_examples(monitor_records, normalizer)
    train_groups = group_examples(train_examples)
    threshold_groups = group_examples(threshold_examples)
    monitor_groups = group_examples(monitor_examples)
    train_targets = np.stack([group["target"] for group in train_groups])
    threshold_targets = np.stack([group["target"] for group in threshold_groups])
    monitor_targets = np.stack([group["target"] for group in monitor_groups])

    split_examples = {
        "train": train_examples,
        "val_threshold": threshold_examples,
        "val_monitor": monitor_examples,
    }
    split_groups = {
        "train": train_groups,
        "val_threshold": threshold_groups,
        "val_monitor": monitor_groups,
    }
    frozen_token_logits: dict[int, dict[str, np.ndarray]] = {}
    exact_checks = []
    for seed in config["seeds"]:
        checkpoint_path = wave51 / "checkpoints" / f"seed{seed}__sigmoid_only.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model = DualHeadDeepSet()
        model.load_state_dict(checkpoint["model_state"])
        frozen_token_logits[int(seed)] = {}
        for split_name, examples in split_examples.items():
            set_logits, choice_logits = predict_dual_logits(
                model, examples, int(config["inference_batch_tokens"])
            )
            tokens, averaged = token_logits(examples, set_logits)
            choice_tokens, averaged_choice = token_logits(examples, choice_logits)
            expected_tokens = [
                group["pair_token"] for group in split_groups[split_name]
            ]
            if tokens != expected_tokens:
                raise RuntimeError(
                    f"token alignment failed for seed {seed}/{split_name}"
                )
            if choice_tokens != expected_tokens:
                raise RuntimeError(
                    f"choice-token alignment failed for seed {seed}/{split_name}"
                )
            frozen_token_logits[int(seed)][split_name] = averaged
            frozen_path = (
                output_dir / "raw_eval" / "frozen_set" / f"seed{seed}__{split_name}.npz"
            )
            frozen_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                frozen_path,
                pair_token=np.asarray(tokens),
                cluster_id=np.asarray(tokens),
                target=np.stack(
                    [group["target"] for group in split_groups[split_name]]
                ),
                set_logits=averaged,
                choice_logits=averaged_choice,
            )
            if split_name in {"val_threshold", "val_monitor"}:
                with np.load(
                    wave51 / "logits" / f"seed{seed}__sigmoid_only__{split_name}.npz"
                ) as saved:
                    exact = np.array_equal(set_logits, saved["set_logits"])
                    exact_checks.append(exact)
                    if not exact:
                        raise RuntimeError(
                            f"set re-forward differs from Wave 51 for seed {seed}/{split_name}"
                        )

    ensemble_set = {
        split: np.mean(
            np.stack(
                [frozen_token_logits[int(seed)][split] for seed in config["seeds"]]
            ),
            axis=0,
        )
        for split in split_groups
    }
    tau, tau_selection = select_smoke_tau(
        threshold_groups,
        ensemble_set["val_threshold"],
        "sigmoid_set",
        config["diagnostic_tau_grid"],
    )
    set_probability = {split: expit(logits) for split, logits in ensemble_set.items()}

    lambda_selection = {}
    for fold in range(3):
        fold_spec = policy_fold(groups, fold)
        val_utility = policies[list(fold_spec["policy_val"])]
        val_truth = authorized_actions(threshold_targets, val_utility)
        val_mask = np.asarray(
            [
                group["design_stratum"] == "NEAR_RIVAL" and group["cardinality"] >= 2
                for group in threshold_groups
            ]
        )
        candidates = []
        for weight in config["utility_weight_grid"]:
            action = score_composition_actions(
                ensemble_set["val_threshold"], val_utility, weight
            )
            compatible = threshold_targets.astype(bool)[
                np.arange(len(threshold_groups))[:, None], action
            ]
            regret = constrained_regret(
                action,
                threshold_targets,
                val_utility,
                float(config["incompatible_regret_penalty"]),
            )
            candidates.append(
                {
                    "weight": float(weight),
                    "accuracy": float(np.mean((action == val_truth)[val_mask])),
                    "compatible": float(np.mean(compatible[val_mask])),
                    "regret": float(np.mean(regret[val_mask])),
                }
            )
        selected = max(
            candidates,
            key=lambda row: (
                row["accuracy"],
                row["compatible"],
                -row["regret"],
                -row["weight"],
            ),
        )
        lambda_selection[str(fold)] = {"selected": selected, "grid": candidates}

    shuffle_manifest = write_training_shuffle_manifests(
        output_dir, train_groups, config
    )
    fold_models: dict[int, dict[str, list[Any]]] = {}
    training_runs = []
    for fold in range(3):
        fold_spec = policy_fold(groups, fold)
        train_policy = policies[list(fold_spec["policy_train"])]
        fold_models[fold] = {
            arm: []
            for arm in (
                "learned_reader_same_set",
                "direct_contextual_choice",
                "joint_set_contextual_choice",
                "utility_ignored",
            )
        }
        for seed in config["seeds"]:
            seed = int(seed)
            reader, history, optimizer_state = train_frozen_reader(
                ensemble_set["train"], train_targets, train_policy, seed, fold, config
            )
            fold_models[fold]["learned_reader_same_set"].append(reader)
            path = (
                output_dir
                / "checkpoints"
                / f"fold{fold}__seed{seed}__learned_reader_same_set.pt"
            )
            save_checkpoint(
                path,
                reader,
                history,
                optimizer_state,
                {
                    "fold": fold,
                    "seed": seed,
                    "arm": "learned_reader_same_set",
                    "policy_manifest_sha256": policy_manifest_sha256,
                },
            )
            training_runs.append(
                {
                    "fold": fold,
                    "seed": seed,
                    "arm": "learned_reader_same_set",
                    "checkpoint": str(path.relative_to(output_dir)),
                }
            )

            torch.manual_seed(seed * 10_000 + fold + 520)
            initial = copy.deepcopy(ContextualPolicyDeepSet().state_dict())
            for arm, objective, input_utility in (
                ("direct_contextual_choice", "choice", train_policy),
                ("joint_set_contextual_choice", "joint", train_policy),
                (
                    "utility_ignored",
                    "choice",
                    np.repeat(levels[None, :], len(train_policy), axis=0),
                ),
            ):
                model = ContextualPolicyDeepSet()
                model.load_state_dict(copy.deepcopy(initial))
                history, optimizer_state = train_contextual_model(
                    model,
                    train_groups,
                    input_utility,
                    train_policy,
                    objective,
                    seed,
                    fold,
                    config,
                )
                fold_models[fold][arm].append(model)
                path = output_dir / "checkpoints" / f"fold{fold}__seed{seed}__{arm}.pt"
                save_checkpoint(
                    path,
                    model,
                    history,
                    optimizer_state,
                    {
                        "fold": fold,
                        "seed": seed,
                        "arm": arm,
                        "policy_manifest_sha256": policy_manifest_sha256,
                    },
                )
                training_runs.append(
                    {
                        "fold": fold,
                        "seed": seed,
                        "arm": arm,
                        "checkpoint": str(path.relative_to(output_dir)),
                    }
                )

    development_raw_dir = output_dir / "raw_predictions"
    development_raw_dir.mkdir(parents=True, exist_ok=True)
    for fold in range(3):
        fold_spec = policy_fold(groups, fold)
        for split_name, policy_key in (
            ("train", "policy_train"),
            ("val_threshold", "policy_val"),
        ):
            eval_groups = split_groups[split_name]
            utility = policies[list(fold_spec[policy_key])]
            targets = np.stack([group["target"] for group in eval_groups])
            valid = targets.astype(bool).any(axis=1)
            truth = np.full((len(eval_groups), len(utility)), -1, dtype=np.int64)
            truth[valid] = authorized_actions(targets[valid], utility)
            explicit, fallback = explicit_set_actions(
                expit(ensemble_set[split_name]), utility, tau
            )
            masked_utility = np.repeat(levels[None, :], len(utility), axis=0)
            masked, _ = explicit_set_actions(
                expit(ensemble_set[split_name]), masked_utility, tau
            )
            score_action = score_composition_actions(
                ensemble_set[split_name],
                utility,
                lambda_selection[str(fold)]["selected"]["weight"],
            )
            reader_scores = []
            for reader in fold_models[fold]["learned_reader_same_set"]:
                with torch.no_grad():
                    reader_scores.append(
                        reader(
                            torch.from_numpy(
                                ensemble_set[split_name].astype(np.float32)
                            ),
                            torch.from_numpy(utility.astype(np.float32)),
                        ).numpy()
                    )
            payload: dict[str, np.ndarray] = {
                "pair_token": np.asarray(
                    [group["pair_token"] for group in eval_groups]
                ),
                "cluster_id": np.asarray(
                    [group["cluster_id"] for group in eval_groups]
                ),
                "policy_index": np.asarray(fold_spec[policy_key], dtype=np.int64),
                "utility": utility,
                "target": targets,
                "authorized_action": truth,
                "choice_valid": valid,
                "ensemble_set_logits": ensemble_set[split_name],
                "per_seed_frozen_set_logits": np.stack(
                    [
                        frozen_token_logits[int(seed)][split_name]
                        for seed in config["seeds"]
                    ]
                ),
                "explicit_fallback": fallback,
                "ensemble_action__explicit_set_policy": explicit,
                "per_seed_action__explicit_set_policy": np.stack(
                    [
                        explicit_set_actions(
                            expit(frozen_token_logits[int(seed)][split_name]),
                            utility,
                            tau,
                        )[0]
                        for seed in config["seeds"]
                    ]
                ),
                "ensemble_action__score_composition": score_action,
                "per_seed_action__score_composition": np.stack(
                    [
                        score_composition_actions(
                            frozen_token_logits[int(seed)][split_name],
                            utility,
                            lambda_selection[str(fold)]["selected"]["weight"],
                        )
                        for seed in config["seeds"]
                    ]
                ),
                "ensemble_action__oracle_set_then_utility": truth,
                "ensemble_action__explicit_context_masked_eval": masked,
                "learned_reader_scores_by_seed": np.stack(reader_scores),
                "ensemble_action__learned_reader_same_set": np.argmax(
                    np.mean(np.stack(reader_scores), axis=0), axis=-1
                ),
                "per_seed_action__learned_reader_same_set": np.argmax(
                    np.stack(reader_scores), axis=-1
                ),
            }
            for arm in (
                "direct_contextual_choice",
                "joint_set_contextual_choice",
                "utility_ignored",
            ):
                set_rows, score_rows = [], []
                input_utility = (
                    np.repeat(levels[None, :], len(utility), axis=0)
                    if arm == "utility_ignored"
                    else utility
                )
                for model in fold_models[fold][arm]:
                    set_logits, scores = predict_contextual_scores(
                        model,
                        eval_groups,
                        input_utility,
                        int(config["inference_batch_tokens"]),
                    )
                    set_rows.append(set_logits)
                    score_rows.append(scores)
                payload[f"{arm}__set_logits_by_seed"] = np.stack(set_rows)
                payload[f"{arm}__scores_by_seed"] = np.stack(score_rows)
                payload[f"per_seed_action__{arm}"] = np.argmax(
                    np.stack(score_rows), axis=-1
                )
                payload[f"ensemble_action__{arm}"] = np.argmax(
                    np.mean(np.stack(score_rows), axis=0), axis=-1
                )
            np.savez_compressed(
                development_raw_dir / f"fold{fold}__{split_name}__{policy_key}.npz",
                **payload,
            )

    metric_stores = {
        regime: {arm: defaultdict(list) for arm in SYSTEM_ARMS}
        for regime in ("policy_seen", "policy_shift")
    }
    seed_metric_stores = {
        int(seed): {
            regime: {arm: defaultdict(list) for arm in SYSTEM_ARMS}
            for regime in ("policy_seen", "policy_shift")
        }
        for seed in config["seeds"]
    }
    counterfactual_parts = []
    counterfactual_by_seed: dict[int, list[dict[str, float]]] = {
        int(seed): [] for seed in config["seeds"]
    }
    counterfactual_mapping_rows = []
    per_policy_rows = []
    raw_dir = output_dir / "raw_predictions"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for fold in range(3):
        fold_spec = policy_fold(groups, fold)
        selected = lambda_selection[str(fold)]["selected"]

        for regime, policy_key in (
            ("policy_seen", "policy_train"),
            ("policy_shift", "policy_shift"),
        ):
            policy_indices = tuple(fold_spec[policy_key])
            utility = policies[list(policy_indices)]
            truth = authorized_actions(monitor_targets, utility)
            explicit, fallback = explicit_set_actions(
                set_probability["val_monitor"], utility, tau
            )
            masked_utility = np.repeat(levels[None, :], len(utility), axis=0)
            explicit_by_seed = []
            masked_by_seed = []
            score_by_seed = []
            for seed in config["seeds"]:
                seed_logits = frozen_token_logits[int(seed)]["val_monitor"]
                seed_explicit, _ = explicit_set_actions(
                    expit(seed_logits), utility, tau
                )
                seed_masked, _ = explicit_set_actions(
                    expit(seed_logits), masked_utility, tau
                )
                explicit_by_seed.append(seed_explicit)
                masked_by_seed.append(seed_masked)
                score_by_seed.append(
                    score_composition_actions(seed_logits, utility, selected["weight"])
                )
            masked, _ = explicit_set_actions(
                set_probability["val_monitor"], masked_utility, tau
            )
            oracle = truth.copy()
            score_action = score_composition_actions(
                ensemble_set["val_monitor"], utility, selected["weight"]
            )

            reader_scores = []
            for reader in fold_models[fold]["learned_reader_same_set"]:
                with torch.no_grad():
                    reader_scores.append(
                        reader(
                            torch.from_numpy(
                                ensemble_set["val_monitor"].astype(np.float32)
                            ),
                            torch.from_numpy(utility.astype(np.float32)),
                        ).numpy()
                    )
            learned = np.argmax(np.mean(np.stack(reader_scores), axis=0), axis=-1)

            learned_system_actions = {}
            contextual_scores_by_seed = {}
            contextual_set_logits_by_seed = {}
            for arm in (
                "direct_contextual_choice",
                "joint_set_contextual_choice",
                "utility_ignored",
            ):
                scores = []
                set_outputs = []
                for model in fold_models[fold][arm]:
                    input_utility = utility
                    if arm == "utility_ignored":
                        input_utility = np.repeat(levels[None, :], len(utility), axis=0)
                    set_output, output = predict_contextual_scores(
                        model,
                        monitor_groups,
                        input_utility,
                        int(config["inference_batch_tokens"]),
                    )
                    set_outputs.append(set_output)
                    scores.append(output)
                contextual_scores_by_seed[arm] = np.stack(scores)
                contextual_set_logits_by_seed[arm] = np.stack(set_outputs)
                learned_system_actions[arm] = np.argmax(
                    np.mean(np.stack(scores), axis=0), axis=-1
                )

            actions_by_arm = {
                "explicit_set_policy": explicit,
                "learned_reader_same_set": learned,
                "direct_contextual_choice": learned_system_actions[
                    "direct_contextual_choice"
                ],
                "joint_set_contextual_choice": learned_system_actions[
                    "joint_set_contextual_choice"
                ],
                "utility_ignored": learned_system_actions["utility_ignored"],
                "score_composition": score_action,
                "oracle_set_then_utility": oracle,
                "explicit_context_masked_eval": masked,
            }
            for arm, actions in actions_by_arm.items():
                rows = policy_metrics_by_token(
                    monitor_groups,
                    utility,
                    actions,
                    float(config["incompatible_regret_penalty"]),
                )
                append_metric_rows(metric_stores[regime], arm, rows)
                per_policy_rows.extend(
                    policy_rows(
                        monitor_groups,
                        utility,
                        policy_indices,
                        actions,
                        arm,
                        fold,
                        regime,
                        float(config["incompatible_regret_penalty"]),
                    )
                )

            per_seed_actions = {
                "explicit_set_policy": np.stack(explicit_by_seed),
                "learned_reader_same_set": np.argmax(np.stack(reader_scores), axis=-1),
                "direct_contextual_choice": np.argmax(
                    contextual_scores_by_seed["direct_contextual_choice"], axis=-1
                ),
                "joint_set_contextual_choice": np.argmax(
                    contextual_scores_by_seed["joint_set_contextual_choice"], axis=-1
                ),
                "utility_ignored": np.argmax(
                    contextual_scores_by_seed["utility_ignored"], axis=-1
                ),
                "score_composition": np.stack(score_by_seed),
                "oracle_set_then_utility": np.repeat(
                    oracle[None, ...], len(config["seeds"]), axis=0
                ),
                "explicit_context_masked_eval": np.stack(masked_by_seed),
            }
            for seed_index, seed in enumerate(config["seeds"]):
                for arm, seed_actions in per_seed_actions.items():
                    append_metric_rows(
                        seed_metric_stores[int(seed)][regime],
                        arm,
                        policy_metrics_by_token(
                            monitor_groups,
                            utility,
                            seed_actions[seed_index],
                            float(config["incompatible_regret_penalty"]),
                        ),
                    )
                    per_policy_rows.extend(
                        policy_rows(
                            monitor_groups,
                            utility,
                            policy_indices,
                            seed_actions[seed_index],
                            arm,
                            fold,
                            regime,
                            float(config["incompatible_regret_penalty"]),
                            seed=int(seed),
                        )
                    )

            np.savez_compressed(
                raw_dir / f"fold{fold}__{regime}__monitor.npz",
                pair_token=np.asarray(
                    [group["pair_token"] for group in monitor_groups]
                ),
                cluster_id=np.asarray(
                    [group["cluster_id"] for group in monitor_groups]
                ),
                policy_index=np.asarray(policy_indices, dtype=np.int64),
                utility=utility,
                authorized_action=truth,
                target=monitor_targets,
                ensemble_set_logits=ensemble_set["val_monitor"],
                per_seed_frozen_set_logits=np.stack(
                    [
                        frozen_token_logits[int(seed)]["val_monitor"]
                        for seed in config["seeds"]
                    ]
                ),
                learned_reader_scores_by_seed=np.stack(reader_scores),
                direct_scores_by_seed=contextual_scores_by_seed[
                    "direct_contextual_choice"
                ],
                joint_scores_by_seed=contextual_scores_by_seed[
                    "joint_set_contextual_choice"
                ],
                ignored_scores_by_seed=contextual_scores_by_seed["utility_ignored"],
                direct_set_logits_by_seed=contextual_set_logits_by_seed[
                    "direct_contextual_choice"
                ],
                joint_set_logits_by_seed=contextual_set_logits_by_seed[
                    "joint_set_contextual_choice"
                ],
                ignored_set_logits_by_seed=contextual_set_logits_by_seed[
                    "utility_ignored"
                ],
                **{
                    f"ensemble_action__{arm}": actions
                    for arm, actions in actions_by_arm.items()
                },
                **{
                    f"per_seed_action__{arm}": actions
                    for arm, actions in per_seed_actions.items()
                },
                explicit_fallback=fallback,
            )
            if regime == "policy_shift":
                cf, mapping = counterfactual_success(monitor_groups, utility, explicit)
                counterfactual_parts.append(cf)
                for row in mapping:
                    row.update(
                        {
                            "fold": fold,
                            "policy_indices": list(policy_indices),
                            "system": "ensemble",
                        }
                    )
                counterfactual_mapping_rows.extend(mapping)
                for seed_index, seed in enumerate(config["seeds"]):
                    seed_cf, seed_mapping = counterfactual_success(
                        monitor_groups,
                        utility,
                        per_seed_actions["explicit_set_policy"][seed_index],
                    )
                    counterfactual_by_seed[int(seed)].append(seed_cf)
                    for row in seed_mapping:
                        row.update(
                            {
                                "fold": fold,
                                "policy_indices": list(policy_indices),
                                "system": f"seed{seed}",
                            }
                        )
                    counterfactual_mapping_rows.extend(seed_mapping)

    per_token = {}
    regime_metrics = {}
    metric_intervals = {}
    for regime in metric_stores:
        per_token[regime], regime_metrics[regime] = aggregate_metric_store(
            metric_stores[regime]
        )
        metric_intervals[regime] = {}
        for arm, rows in per_token[regime].items():
            write_jsonl(
                output_dir / "metrics_by_token" / f"{regime}__{arm}.jsonl", rows
            )
            metric_intervals[regime][arm] = {
                metric: bootstrap_metric_ci(
                    rows,
                    metric,
                    int(config["n_boot"]),
                    52_000 + list(SYSTEM_ARMS).index(arm) * 10 + index,
                )
                for index, metric in enumerate(
                    (
                        "action_accuracy",
                        "compatible_action_rate",
                        "restricted_regret",
                        "worst_restricted_regret",
                    )
                )
            }

    seed_sensitivity = {}
    for seed, seed_store in seed_metric_stores.items():
        seed_sensitivity[str(seed)] = {}
        for regime, store in seed_store.items():
            _, summaries = aggregate_metric_store(store)
            seed_sensitivity[str(seed)][regime] = summaries
    write_json(output_dir / "seed_sensitivity.json", seed_sensitivity)
    write_jsonl(output_dir / "metrics_by_policy.jsonl", per_policy_rows)
    write_jsonl(
        output_dir / "counterfactual_mapping.jsonl", counterfactual_mapping_rows
    )

    set_rows, set_integrity = set_integrity_metrics(
        monitor_groups, ensemble_set["val_monitor"], tau
    )
    write_jsonl(output_dir / "set_integrity_by_token.jsonl", set_rows)
    set_integrity_by_seed = {}
    for seed in config["seeds"]:
        _, seed_set = set_integrity_metrics(
            monitor_groups, frozen_token_logits[int(seed)]["val_monitor"], tau
        )
        set_integrity_by_seed[str(seed)] = seed_set

    checkpoint_replay, replayed_actions, replayed_seed_actions = replay_saved_arrays(
        output_dir, split_groups, config, levels, tau, lambda_selection
    )
    replay_metric_stores = {
        regime: {arm: defaultdict(list) for arm in SYSTEM_ARMS}
        for regime in ("policy_seen", "policy_shift")
    }
    for fold in range(3):
        for regime in ("policy_seen", "policy_shift"):
            with np.load(raw_dir / f"fold{fold}__{regime}__monitor.npz") as raw:
                for arm in SYSTEM_ARMS:
                    append_metric_rows(
                        replay_metric_stores[regime],
                        arm,
                        policy_metrics_by_token(
                            monitor_groups,
                            raw["utility"],
                            replayed_actions[(fold, regime)][arm],
                            float(config["incompatible_regret_penalty"]),
                        ),
                    )
    replay_regime_metrics = {
        regime: aggregate_metric_store(store)[1]
        for regime, store in replay_metric_stores.items()
    }
    if stable_hash(replay_regime_metrics) != stable_hash(regime_metrics):
        raise RuntimeError("raw-artifact metric replay differs from in-memory metrics")

    replay_seed_stores = {
        int(seed): {
            regime: {arm: defaultdict(list) for arm in SYSTEM_ARMS}
            for regime in ("policy_seen", "policy_shift")
        }
        for seed in config["seeds"]
    }
    replay_policy_rows = []
    for fold in range(3):
        for regime in ("policy_seen", "policy_shift"):
            with np.load(raw_dir / f"fold{fold}__{regime}__monitor.npz") as raw:
                policy_indices = tuple(int(value) for value in raw["policy_index"])
                utility = raw["utility"]
                for seed_index, seed in enumerate(config["seeds"]):
                    for arm in SYSTEM_ARMS:
                        actions = replayed_seed_actions[(fold, regime)][arm][seed_index]
                        append_metric_rows(
                            replay_seed_stores[int(seed)][regime],
                            arm,
                            policy_metrics_by_token(
                                monitor_groups,
                                utility,
                                actions,
                                float(config["incompatible_regret_penalty"]),
                            ),
                        )
                        replay_policy_rows.extend(
                            policy_rows(
                                monitor_groups,
                                utility,
                                policy_indices,
                                actions,
                                arm,
                                fold,
                                regime,
                                float(config["incompatible_regret_penalty"]),
                                seed=int(seed),
                            )
                        )
                for arm in SYSTEM_ARMS:
                    replay_policy_rows.extend(
                        policy_rows(
                            monitor_groups,
                            utility,
                            policy_indices,
                            replayed_actions[(fold, regime)][arm],
                            arm,
                            fold,
                            regime,
                            float(config["incompatible_regret_penalty"]),
                        )
                    )
    replay_seed_sensitivity = {}
    for seed, seed_store in replay_seed_stores.items():
        replay_seed_sensitivity[str(seed)] = {}
        for regime, store in seed_store.items():
            replay_seed_sensitivity[str(seed)][regime] = aggregate_metric_store(store)[
                1
            ]
    if stable_hash(replay_seed_sensitivity) != stable_hash(seed_sensitivity):
        raise RuntimeError("checkpoint replay differs for per-seed metrics")
    policy_row_key = lambda row: (  # noqa: E731 - compact deterministic artifact key
        row["fold"],
        row["regime"],
        str(row["seed"]),
        row["arm"],
        row["policy_index"],
    )
    replay_policy_rows_sorted = sorted(replay_policy_rows, key=policy_row_key)
    per_policy_rows_sorted = sorted(per_policy_rows, key=policy_row_key)
    if stable_hash(replay_policy_rows_sorted) != stable_hash(per_policy_rows_sorted):
        raise RuntimeError("checkpoint replay differs for per-policy metrics")

    primary_rows = per_token["policy_shift"]
    n_boot = int(config["n_boot"])
    contrasts = {
        "explicit_minus_direct_accuracy": bootstrap_delta(
            primary_rows["explicit_set_policy"],
            primary_rows["direct_contextual_choice"],
            "action_accuracy",
            n_boot,
            5201,
        ),
        "explicit_minus_direct_regret": bootstrap_delta(
            primary_rows["explicit_set_policy"],
            primary_rows["direct_contextual_choice"],
            "restricted_regret",
            n_boot,
            5202,
        ),
        "explicit_minus_direct_compatible": bootstrap_delta(
            primary_rows["explicit_set_policy"],
            primary_rows["direct_contextual_choice"],
            "compatible_action_rate",
            n_boot,
            5203,
        ),
        "direct_minus_ignored_accuracy": bootstrap_delta(
            primary_rows["direct_contextual_choice"],
            primary_rows["utility_ignored"],
            "action_accuracy",
            n_boot,
            5204,
        ),
        "explicit_minus_masked_accuracy": bootstrap_delta(
            primary_rows["explicit_set_policy"],
            primary_rows["explicit_context_masked_eval"],
            "action_accuracy",
            n_boot,
            5205,
        ),
        "explicit_minus_learned_reader_accuracy": bootstrap_delta(
            primary_rows["explicit_set_policy"],
            primary_rows["learned_reader_same_set"],
            "action_accuracy",
            n_boot,
            5206,
        ),
    }
    total_cf = sum(row["n_counterfactual_pairs"] for row in counterfactual_parts)
    counterfactual = {
        "n_pair_tokens": sum(
            group["design_stratum"] == "NEAR_RIVAL" and group["cardinality"] >= 2
            for group in monitor_groups
        ),
        "n_counterfactual_pairs": total_cf,
        "success_rate": float(
            sum(
                row["success_rate"] * row["n_counterfactual_pairs"]
                for row in counterfactual_parts
            )
            / total_cf
        ),
        "joint_correct_rate": float(
            sum(
                row["joint_correct_rate"] * row["n_counterfactual_pairs"]
                for row in counterfactual_parts
            )
            / total_cf
        ),
        "by_fold": counterfactual_parts,
        "by_checkpoint": counterfactual_by_seed,
    }
    criteria = config["diagnostic_criteria"]
    accuracy = contrasts["explicit_minus_direct_accuracy"]
    regret = contrasts["explicit_minus_direct_regret"]
    compatible = contrasts["explicit_minus_direct_compatible"]
    direct_control = contrasts["direct_minus_ignored_accuracy"]
    explicit_control = contrasts["explicit_minus_masked_accuracy"]
    conditions = {
        "accuracy_advantage": accuracy["mean_diff"]
        >= criteria["accuracy_improvement_min"]
        and accuracy["ci95_low"] > 0,
        "regret_advantage": regret["mean_diff"] <= -criteria["regret_reduction_min"]
        and regret["ci95_high"] < 0,
        "compatible_noninferiority": compatible["mean_diff"]
        >= criteria["compatible_rate_noninferiority"],
        "set_logits_array_exact": all(exact_checks),
        "direct_uses_context": direct_control["mean_diff"]
        >= criteria["context_control_improvement_min"],
        "explicit_uses_context": explicit_control["mean_diff"]
        >= criteria["context_control_improvement_min"],
        "counterfactual_success": counterfactual["success_rate"]
        >= criteria["counterfactual_success_min"],
    }
    diagnostic_pattern = {
        **conditions,
        "policy_transport_promising": all(conditions.values()),
    }

    checkpoint_replay.update(
        {
            "metric_hash": stable_hash(regime_metrics),
            "metric_replay_hash": stable_hash(replay_regime_metrics),
            "metrics_array_exact": True,
            "seed_metric_hash": stable_hash(seed_sensitivity),
            "seed_metric_replay_hash": stable_hash(replay_seed_sensitivity),
            "per_policy_hash": stable_hash(per_policy_rows_sorted),
            "per_policy_replay_hash": stable_hash(replay_policy_rows_sorted),
            "per_seed_metrics_array_exact": True,
            "per_policy_metrics_array_exact": True,
        }
    )
    write_json(output_dir / "replay_receipt.json", checkpoint_replay)

    contextual = ContextualPolicyDeepSet()
    reader_probe = SharedFamilyReader()
    contextual_total = sum(parameter.numel() for parameter in contextual.parameters())
    set_head_params = sum(
        parameter.numel() for parameter in contextual.set_head.parameters()
    )
    reader_params = sum(parameter.numel() for parameter in reader_probe.parameters())
    contextual_choice_active = contextual_total - set_head_params
    contextual_steps = (
        (len(train_groups) + int(config["batch_tokens"]) - 1)
        // int(config["batch_tokens"])
        * int(config["model_epochs"])
    )
    reader_steps = (
        (len(train_groups) + int(config["batch_tokens"]) - 1)
        // int(config["batch_tokens"])
        * int(config["reader_epochs"])
    )
    training_budget = {
        "replicates_per_learned_arm": 9,
        "fixed_epochs": {
            "contextual": int(config["model_epochs"]),
            "reader": int(config["reader_epochs"]),
        },
        "optimizer_steps_per_run": {
            "contextual": contextual_steps,
            "reader": reader_steps,
        },
        "arms": {
            "explicit_set_policy": {
                "total_parameters": 0,
                "active_parameters": 0,
                "backprops": 0,
                "operation_proxy_parameter_backprops": 0,
            },
            "learned_reader_same_set": {
                "total_parameters": reader_params,
                "active_parameters": reader_params,
                "backprops": reader_steps * 9,
                "operation_proxy_parameter_backprops": reader_params * reader_steps * 9,
            },
            "direct_contextual_choice": {
                "total_parameters": contextual_total,
                "active_parameters": contextual_choice_active,
                "backprops": contextual_steps * 9,
                "operation_proxy_parameter_backprops": contextual_choice_active
                * contextual_steps
                * 9,
            },
            "joint_set_contextual_choice": {
                "total_parameters": contextual_total,
                "active_parameters": contextual_total,
                "backprops": contextual_steps * 9,
                "operation_proxy_parameter_backprops": contextual_total
                * contextual_steps
                * 9,
            },
            "utility_ignored": {
                "total_parameters": contextual_total,
                "active_parameters": contextual_choice_active,
                "backprops": contextual_steps * 9,
                "operation_proxy_parameter_backprops": contextual_choice_active
                * contextual_steps
                * 9,
            },
        },
        "note": "Operation proxy is active-parameter count times optimizer backprops; it is not FLOPs.",
    }
    artifact_contract = {
        "checkpoint_count": len(list((output_dir / "checkpoints").glob("*.pt"))),
        "training_run_count": len(training_runs),
        "frozen_raw_count": len(
            list((output_dir / "raw_eval" / "frozen_set").glob("*.npz"))
        ),
        "development_raw_count": len(list(raw_dir.glob("fold*__train__*.npz")))
        + len(list(raw_dir.glob("fold*__val_threshold__*.npz"))),
        "monitor_raw_count": len(list(raw_dir.glob("fold*__policy_*__monitor.npz"))),
        "training_shuffle_count": len(
            list((output_dir / "training_shuffles").glob("*.npz"))
        ),
        "per_policy_row_count": len(per_policy_rows),
        "counterfactual_mapping_count": len(counterfactual_mapping_rows),
    }
    artifact_contract["complete"] = bool(
        artifact_contract["checkpoint_count"] == 36
        and artifact_contract["training_run_count"] == 36
        and artifact_contract["frozen_raw_count"] == 9
        and artifact_contract["development_raw_count"] == 6
        and artifact_contract["monitor_raw_count"] == 6
        and artifact_contract["training_shuffle_count"] == 9
        and artifact_contract["per_policy_row_count"] == 1536
        and artifact_contract["counterfactual_mapping_count"] > 0
        and checkpoint_replay["array_exact"]
        and checkpoint_replay["metrics_array_exact"]
        and checkpoint_replay["per_seed_metrics_array_exact"]
        and checkpoint_replay["per_policy_metrics_array_exact"]
    )
    if not artifact_contract["complete"]:
        raise RuntimeError(f"artifact contract incomplete: {artifact_contract}")
    conditions["artifact_contract_complete"] = True
    conditions["set_integrity_recomputed"] = bool(all(exact_checks))
    diagnostic_pattern = {
        **conditions,
        "policy_transport_promising": all(conditions.values()),
    }

    accessed = sorted(
        {
            *(Path(path).resolve() for path in train_reads + val_reads),
            *(Path(row["path"]).resolve() for row in binding_receipt),
            config_path,
        }
    )
    assert_no_lockbox(accessed)
    if validate_source_binding(wave50, wave51, config) != binding_receipt:
        raise RuntimeError("bound source artifact changed during execution")
    if execution_sources(config_path) != sources:
        raise RuntimeError("execution source changed during execution")
    write_json(
        output_dir / "access_receipt.json",
        {
            "status": config["status"],
            "allowed_splits": ["train", "val"],
            "lockbox_accessed": False,
            "policy_manifest_sha256_before_label_load": policy_manifest_sha256,
            "source_scope_inventory": source_scope,
            "files_read": [
                {"path": str(path), "sha256": sha256_file(path)} for path in accessed
            ],
        },
    )
    write_json(output_dir / "training_runs.json", training_runs)
    write_json(output_dir / "lambda_selection.json", lambda_selection)

    artifact_files = sorted(
        path
        for path in output_dir.rglob("*")
        if path.is_file()
        and path.name
        not in {
            "artifact_manifest.json",
            "summary.json",
            "REPORT_WAVE52_POLICY_TRANSPORT.md",
        }
    )
    write_json(
        output_dir / "artifact_manifest.json",
        {
            "execution_sources": sources,
            "files": [
                {"path": str(path.relative_to(output_dir)), "sha256": sha256_file(path)}
                for path in artifact_files
            ],
        },
    )
    summary = {
        "status": config["status"],
        "scientific_claim_allowed": False,
        "automatic_go": False,
        "decision_authority": "user",
        "git_commit": git_commit(),
        "execution_sources": sources,
        "source_binding": binding_receipt,
        "policy_manifest_sha256": policy_manifest_sha256,
        "config": config,
        "data": {
            "train_pair_tokens": len(train_groups),
            "val_threshold_pair_tokens": len(threshold_groups),
            "val_monitor_pair_tokens": len(monitor_groups),
            "primary_monitor_tokens": sum(
                group["design_stratum"] == "NEAR_RIVAL" and group["cardinality"] >= 2
                for group in monitor_groups
            ),
            "near_rival_singletons": sum(
                group["design_stratum"] == "NEAR_RIVAL" and group["cardinality"] == 1
                for group in monitor_groups
            ),
            "source_scope_inventory": source_scope,
            "generative_resampling_unit": "pair_token",
        },
        "tau": tau,
        "tau_selection": tau_selection,
        "policy_seen_metrics": regime_metrics["policy_seen"],
        "policy_shift_metrics": regime_metrics["policy_shift"],
        "metric_intervals": metric_intervals,
        "seed_sensitivity": seed_sensitivity,
        "set_integrity": set_integrity,
        "set_integrity_by_seed": set_integrity_by_seed,
        "training_budget": training_budget,
        "training_shuffle_manifest": shuffle_manifest,
        "replay": checkpoint_replay,
        "artifact_contract": artifact_contract,
        "contrasts": contrasts,
        "counterfactual": counterfactual,
        "diagnostic_pattern": diagnostic_pattern,
        "runtime": {
            "device": "cpu",
            "cpu_threads": int(config["cpu_threads"]),
            "wall_seconds": time.perf_counter() - started,
            "cuda_initialized": torch.cuda.is_initialized(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "max_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
        },
    }
    summary["artifact_hash"] = stable_hash(
        {
            "config": config,
            "data": summary["data"],
            "tau": tau,
            "policy_seen_metrics": summary["policy_seen_metrics"],
            "policy_shift_metrics": summary["policy_shift_metrics"],
            "contrasts": contrasts,
            "counterfactual": counterfactual,
            "set_integrity": set_integrity,
            "training_budget": training_budget,
            "replay": checkpoint_replay,
            "artifact_contract": artifact_contract,
            "diagnostic_pattern": diagnostic_pattern,
        }
    )
    write_json(output_dir / "summary.json", summary)
    write_report(output_dir, summary)
    write_json(
        output_dir / "package_manifest.json",
        {
            "phase": "WAVE52_SMOKE_FINAL_PACKAGE_ROOT",
            "files": [
                {"path": name, "sha256": sha256_file(output_dir / name)}
                for name in (
                    "artifact_manifest.json",
                    "summary.json",
                    "REPORT_WAVE52_POLICY_TRANSPORT.md",
                )
            ],
        },
    )


if __name__ == "__main__":
    main()
