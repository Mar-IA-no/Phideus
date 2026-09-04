#!/usr/bin/env python3
"""Fit, select, and adjudicate a public topology-localization residual gate on CPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = ""

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments/geometria_proporcional"))

import run_proportional_graph_fresh_mixed_gate as mixed  # noqa: E402
import run_proportional_graph_frozen_adapters as frozen  # noqa: E402
import run_proportional_graph_residual_gate as historical  # noqa: E402
import run_proportional_graph_safe_abstention_gate as safe  # noqa: E402
import torch  # noqa: E402
from geometria_proporcional.proportional_graph_contract import incidence_matrix  # noqa: E402

DEFAULT_CONFIG = ROOT / "experiments/geometria_proporcional/configs/proportional_graph_topology_localization_gate_v1.json"
SOURCE_FILES = (
    "experiments/geometria_proporcional/PLAN_PROPORTIONAL_TOPOLOGY_LOCALIZATION_GATE_CPU.md",
    "experiments/geometria_proporcional/configs/proportional_graph_topology_localization_gate_v1.json",
    "experiments/geometria_proporcional/run_proportional_graph_topology_localization_gate.py",
    "experiments/geometria_proporcional/run_proportional_graph_fresh_mixed_gate.py",
    "experiments/geometria_proporcional/run_proportional_graph_safe_abstention_gate.py",
    "experiments/geometria_proporcional/run_proportional_graph_residual_gate.py",
    "experiments/geometria_proporcional/run_proportional_graph_neural_smoke.py",
    "src/geometria_proporcional/proportional_graph_contract.py",
)
BASE_FEATURE_ORDER = mixed.FEATURE_ORDER
TOPOLOGY_FEATURE_ORDER = (
    "correction_location_defined",
    "correction_edge_entropy_normalized",
    "correction_node_mass_entropy_normalized",
    "correction_node_mass_max_share",
    "correction_node_mass_effective_fraction",
    "correction_linegraph_product_ratio",
    "correction_linegraph_smoothness",
    "correction_divergence_entropy_normalized",
    "correction_divergence_max_share",
)
FAMILIES = ("correction_scale", "public_base", "topology_augmented")
CONTROL_FAMILIES = ("topology_permuted", "target_shuffled_topology")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def load_config(path: Path) -> dict[str, Any]:
    cfg = json.loads(path.read_text())
    expected = {
        "schema_version", "source_loss_contrast", "source_loss_manifest_sha256",
        "source_historical_gate", "source_historical_gate_manifest_sha256",
        "source_smoke_config", "arms", "seeds", "alphas", "realizations",
        "ridge_lambdas", "ridge_folds", "control_replicates",
        "topology_control_seed", "target_shuffle_seed", "threshold_quantiles",
        "selection_bootstrap_replicates", "selection_bootstrap_seed",
        "selection_upper_percentile", "evaluation_bootstrap_replicates",
        "evaluation_bootstrap_seed", "execution",
    }
    if set(cfg) != expected or cfg["schema_version"] != "proportional-graph-topology-localization-gate-v1":
        raise ValueError("invalid topology gate schema")
    if cfg["arms"] != ["raw_generic", "raw_typed", "closure_generic", "closure_typed"]:
        raise ValueError("arms changed")
    if cfg["seeds"] != [104729, 130363] or cfg["alphas"] != [0.0, 0.25, 0.5, 0.75, 1.0]:
        raise ValueError("neural seed or alpha contract changed")
    if cfg["realizations"] != {
        "calibration_seed": 2026090609, "selection_seed": 2026090617,
        "adjudication_seed": 2026090629, "min_eligible_masters": 220,
    }:
        raise ValueError("realization contract changed")
    if cfg["ridge_lambdas"] != [0.0, 0.01, 0.1, 1.0, 10.0, 100.0] or cfg["ridge_folds"] != 5:
        raise ValueError("ridge contract changed")
    if cfg["control_replicates"] != 16:
        raise ValueError("control replicate count changed")
    if cfg["topology_control_seed"] != 2026090631 or cfg["target_shuffle_seed"] != 2026090637:
        raise ValueError("control seed contract changed")
    if cfg["threshold_quantiles"] != [0.25, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975, 0.99]:
        raise ValueError("threshold grid changed")
    if (
        cfg["selection_bootstrap_replicates"] != 2000
        or cfg["selection_bootstrap_seed"] != 2026090641
        or cfg["selection_upper_percentile"] != 95.0
        or cfg["evaluation_bootstrap_replicates"] != 2000
        or cfg["evaluation_bootstrap_seed"] != 2026090647
    ):
        raise ValueError("bootstrap contract changed")
    if cfg["execution"] != {
        "torch_threads": 1, "max_seconds_per_phase": 720,
        "max_rss_gib": 4.0, "sample_every_views": 64,
    }:
        raise ValueError("execution contract changed")
    return cfg


def source_inputs(cfg: dict[str, Any]) -> list[Path]:
    return mixed.source_inputs(cfg)


def verify_sources(cfg: dict[str, Any]) -> None:
    mixed.verify_sources(cfg)


def require_clean(development: bool) -> None:
    if not development and subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout:
        raise RuntimeError("official phase requires clean worktree")


def git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout.strip()


def normalized_entropy(mass: np.ndarray) -> float:
    mass = np.asarray(mass, dtype=np.float64)
    total = float(mass.sum())
    if total <= 0.0 or len(mass) <= 1:
        return 0.0
    probability = mass / total
    return float(-np.sum(probability * np.log(np.maximum(probability, np.finfo(np.float64).tiny))) / math.log(len(mass)))


def topology_features(n_nodes: int, edge_index: np.ndarray, correction: np.ndarray) -> np.ndarray:
    edges = np.asarray(edge_index, dtype=np.int64)
    delta = np.asarray(correction, dtype=np.float64)
    if edges.shape != (len(delta), 2) or not np.all(np.isfinite(delta)):
        raise ValueError("invalid correction topology input")
    absolute = np.abs(delta)
    total = float(absolute.sum())
    if total <= 0.0:
        return np.zeros(len(TOPOLOGY_FEATURE_ORDER), dtype=np.float64)
    incidence = incidence_matrix(n_nodes, edges)
    node_mass = np.abs(incidence).T @ absolute
    node_probability = node_mass / node_mass.sum()
    node_effective = float(1.0 / (n_nodes * np.sum(node_probability * node_probability)))
    adjacent_pairs = [
        (left, right)
        for left in range(len(edges))
        for right in range(left + 1, len(edges))
        if len(set(edges[left].tolist()) & set(edges[right].tolist())) > 0
    ]
    if adjacent_pairs:
        left = np.asarray([pair[0] for pair in adjacent_pairs], dtype=np.int64)
        right = np.asarray([pair[1] for pair in adjacent_pairs], dtype=np.int64)
        denominator = float(np.mean(absolute * absolute))
        line_product = float(np.mean(absolute[left] * absolute[right]) / denominator)
        line_smoothness = float(np.mean((absolute[left] - absolute[right]) ** 2) / denominator)
    else:
        line_product = 0.0
        line_smoothness = 0.0
    divergence = np.abs(incidence.T @ delta)
    divergence_total = float(divergence.sum())
    divergence_max = float(divergence.max() / divergence_total) if divergence_total > 0.0 else 0.0
    result = np.asarray(
        [
            1.0,
            normalized_entropy(absolute),
            normalized_entropy(node_mass),
            float(node_probability.max()),
            node_effective,
            line_product,
            line_smoothness,
            normalized_entropy(divergence),
            divergence_max,
        ],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(result)):
        raise ValueError("non-finite topology feature")
    return result


def permuted_correction(delta: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    delta = np.asarray(delta, dtype=np.float64)
    permutation = np.random.default_rng(seed).permutation(len(delta))
    if len(delta) > 1 and np.array_equal(permutation, np.arange(len(delta))):
        permutation = np.roll(permutation, 1)
    return delta[permutation], permutation


def public_topology_hash(n_nodes: int, edge_index: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(n_nodes, dtype=np.int64).tobytes())
    digest.update(np.asarray(edge_index, dtype=np.int64).tobytes())
    return digest.hexdigest()


def topology_feature_package(
    role: str, views: list[Any], data: dict[str, np.ndarray], cfg: dict[str, Any], output: Path,
) -> dict[str, np.ndarray]:
    n_arms, n_seeds, n_views = len(cfg["arms"]), len(cfg["seeds"]), len(views)
    true = np.full((n_arms, n_seeds, n_views, len(TOPOLOGY_FEATURE_ORDER)), np.nan)
    control = np.full((n_arms, n_seeds, cfg["control_replicates"], n_views, len(TOPOLOGY_FEATURE_ORDER)), np.nan)
    permutations: list[np.ndarray] = []
    permutation_offsets = [0]
    for arm_index, arm in enumerate(cfg["arms"]):
        for seed_index, neural_seed in enumerate(cfg["seeds"]):
            with np.load(output / role / "forward" / f"{arm}|seed={neural_seed}.npz", allow_pickle=False) as saved:
                corrected = saved["post_irls_corrected_relation"].astype(np.float64)
                offsets = saved["edge_offset"]
                if not np.array_equal(saved["view_id"], np.asarray([view.private.view_id for view in views])):
                    raise AssertionError("forward view order mismatch")
            for view_index, view in enumerate(views):
                start, stop = int(offsets[view_index]), int(offsets[view_index + 1])
                observed = np.asarray(view.public.observed_log_ratio, dtype=np.float64)
                delta = corrected[start:stop] - observed
                true[arm_index, seed_index, view_index] = topology_features(
                    view.public.n_nodes, view.public.edge_index, delta
                )
                for replicate in range(cfg["control_replicates"]):
                    shuffled, permutation = permuted_correction(
                        delta,
                        mixed.stable_seed(
                            cfg["topology_control_seed"], role, arm, neural_seed,
                            public_topology_hash(view.public.n_nodes, view.public.edge_index), replicate,
                        ),
                    )
                    control[arm_index, seed_index, replicate, view_index] = topology_features(
                        view.public.n_nodes, view.public.edge_index, shuffled
                    )
                    permutations.append(permutation.astype(np.int64))
                    permutation_offsets.append(permutation_offsets[-1] + len(permutation))
    edge_offsets = [0]
    for view in views:
        edge_offsets.append(edge_offsets[-1] + len(view.public.edge_index))
    package = {
        "base_feature_order": np.asarray(BASE_FEATURE_ORDER),
        "topology_feature_order": np.asarray(TOPOLOGY_FEATURE_ORDER),
        "true": true,
        "control": control,
        "control_permutation": np.concatenate(permutations),
        "control_permutation_offset": np.asarray(permutation_offsets, dtype=np.int64),
        "control_permutation_axis_order": np.asarray(["arm", "neural_seed", "view", "replicate"]),
        "arm": np.asarray(cfg["arms"]),
        "neural_seed": np.asarray(cfg["seeds"], dtype=np.int64),
        "replicate": np.arange(cfg["control_replicates"], dtype=np.int64),
        "edge_offset": np.asarray(edge_offsets, dtype=np.int64),
        "edge_index": np.concatenate([view.public.edge_index for view in views]),
        "observed_log_ratio": np.concatenate([view.public.observed_log_ratio for view in views]),
        "n_nodes": np.asarray([view.public.n_nodes for view in views], dtype=np.int64),
        "view_id": np.asarray([view.private.view_id for view in views]),
    }
    frozen.save_npz(output / role / "topology_features.npz", package)
    if not np.all(np.isfinite(true)) or not np.all(np.isfinite(control)):
        raise RuntimeError("topology feature extraction failed")
    return package


def feature_matrix(
    data: dict[str, np.ndarray], topology: dict[str, np.ndarray], arm_index: int,
    family: str, replicate: int | None = None,
) -> np.ndarray:
    base = data["features"].mean(axis=1)[arm_index]
    if family == "correction_scale":
        return base[:, list(mixed.REDUCED_COLUMNS)]
    if family == "public_base":
        return base
    if family == "topology_augmented":
        return np.column_stack((base, topology["true"][arm_index].mean(axis=0)))
    if family == "topology_permuted":
        if replicate is None:
            raise ValueError("topology control needs replicate")
        return np.column_stack((base, topology["control"][arm_index, :, replicate].mean(axis=0)))
    if family == "target_shuffled_topology":
        return np.column_stack((base, topology["true"][arm_index].mean(axis=0)))
    raise ValueError(f"unknown family: {family}")


def fit_calibration(
    cfg: dict[str, Any], views: list[Any], data: dict[str, np.ndarray], topology: dict[str, np.ndarray],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    quotient = data["quotient_rmse"].mean(axis=1)
    targets = np.transpose(quotient[:, 1:] - quotient[:, :1], (0, 2, 1))
    masters = np.asarray([view.private.master_id for view in views])
    n_arms, n_views, n_outputs = len(cfg["arms"]), len(views), len(cfg["alphas"]) - 1
    packed: dict[str, np.ndarray] = {"target_delta": targets}
    models: dict[str, Any] = {
        "base_feature_order": list(BASE_FEATURE_ORDER),
        "topology_feature_order": list(TOPOLOGY_FEATURE_ORDER),
        "arms": {},
    }
    for arm_index, arm in enumerate(cfg["arms"]):
        record: dict[str, Any] = {"families": {}, "topology_permuted": [], "target_shuffled_topology": []}
        for family in FAMILIES:
            x = feature_matrix(data, topology, arm_index, family)
            report, prediction, oof = historical.fit_gate(x, targets[arm_index], x, masters, cfg)
            report["oof_mse"] = float(np.mean((oof - targets[arm_index]) ** 2))
            record["families"][family] = report
            packed[f"{arm}|{family}|prediction"] = prediction
            packed[f"{arm}|{family}|oof_prediction"] = oof
            packed[f"{arm}|{family}|oof_action"] = historical.choose_alpha(oof)
        for replicate in range(cfg["control_replicates"]):
            x_control = feature_matrix(data, topology, arm_index, "topology_permuted", replicate)
            report, prediction, oof = historical.fit_gate(
                x_control, targets[arm_index], x_control, masters, cfg
            )
            report["oof_mse"] = float(np.mean((oof - targets[arm_index]) ** 2))
            record["topology_permuted"].append({"replicate": replicate, "model": report})
            packed[f"{arm}|topology_permuted={replicate}|prediction"] = prediction
            packed[f"{arm}|topology_permuted={replicate}|oof_prediction"] = oof
            packed[f"{arm}|topology_permuted={replicate}|oof_action"] = historical.choose_alpha(oof)
            permutation = mixed.paired_master_permutation(
                views, mixed.stable_seed(cfg["target_shuffle_seed"], arm, replicate)
            )
            x_true = feature_matrix(data, topology, arm_index, "target_shuffled_topology")
            report, prediction, oof = historical.fit_gate(
                x_true, targets[arm_index, permutation], x_true, masters, cfg
            )
            report["oof_mse"] = float(np.mean((oof - targets[arm_index, permutation]) ** 2))
            record["target_shuffled_topology"].append(
                {"replicate": replicate, "permutation": permutation.tolist(), "model": report}
            )
            packed[f"{arm}|target_shuffled_topology={replicate}|prediction"] = prediction
            packed[f"{arm}|target_shuffled_topology={replicate}|oof_prediction"] = oof
            packed[f"{arm}|target_shuffled_topology={replicate}|oof_action"] = historical.choose_alpha(oof)
        models["arms"][arm] = record
    return models, packed


def select_one(
    report: dict[str, Any], calibration_x: np.ndarray, selection_x: np.ndarray,
    quotient: np.ndarray, bootstrap: np.ndarray, cfg: dict[str, Any], key: str,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    _, _, calibration_advantage = safe.predicted_action_and_advantage(report, calibration_x)
    thresholds = safe.threshold_grid(calibration_advantage, cfg["threshold_quantiles"])
    predictions, base_action, advantage = safe.predicted_action_and_advantage(report, selection_x)
    actions = [safe.threshold_action(base_action, advantage, threshold) for threshold in thresholds]
    stats = safe.candidate_statistics(quotient, actions, bootstrap, cfg["selection_upper_percentile"])
    selected = stats["selected_index"]
    record = {
        "thresholds": thresholds,
        "mean_iid": stats["mean_iid"].tolist(),
        "mean_balanced": stats["mean_balanced"].tolist(),
        "simultaneous_q95": stats["simultaneous_q95"],
        "upper_iid": stats["upper_iid"].tolist(),
        "admissible": stats["admissible"].tolist(),
        "selected_index": selected,
        "selected_threshold": thresholds[selected],
    }
    packed = {
        f"{key}|iid_delta_by_candidate": stats["iid_delta_by_candidate"],
        f"{key}|balanced_delta_by_candidate": stats["balanced_delta_by_candidate"],
        f"{key}|upper_iid": stats["upper_iid"],
        f"{key}|admissible": stats["admissible"],
        f"{key}|predicted_nonzero_delta": predictions,
        f"{key}|advantage": advantage,
        f"{key}|base_action": base_action,
        f"{key}|candidate_actions": np.asarray(actions),
        f"{key}|selected_action": actions[selected],
    }
    return record, packed


def select_thresholds(
    cfg: dict[str, Any], calibration_data: dict[str, np.ndarray], calibration_topology: dict[str, np.ndarray],
    selection_data: dict[str, np.ndarray], selection_topology: dict[str, np.ndarray], models: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    n_masters = selection_data["quotient_rmse"].shape[-1] // 2
    bootstrap = np.random.default_rng(cfg["selection_bootstrap_seed"]).integers(
        0, n_masters, size=(cfg["selection_bootstrap_replicates"], n_masters)
    )
    freeze: dict[str, Any] = {"arms": {}}
    packed: dict[str, np.ndarray] = {"bootstrap_indices": bootstrap}
    for arm_index, arm in enumerate(cfg["arms"]):
        record: dict[str, Any] = {"families": {}, "topology_permuted": [], "target_shuffled_topology": []}
        quotient = selection_data["quotient_rmse"][arm_index]
        for family in FAMILIES:
            entry, arrays = select_one(
                models["arms"][arm]["families"][family],
                feature_matrix(calibration_data, calibration_topology, arm_index, family),
                feature_matrix(selection_data, selection_topology, arm_index, family),
                quotient, bootstrap, cfg, f"{arm}|{family}",
            )
            record["families"][family] = entry
            packed.update(arrays)
        for control_family in CONTROL_FAMILIES:
            for replicate in range(cfg["control_replicates"]):
                model_record = models["arms"][arm][control_family][replicate]
                entry, arrays = select_one(
                    model_record["model"],
                    feature_matrix(calibration_data, calibration_topology, arm_index, control_family, replicate),
                    feature_matrix(selection_data, selection_topology, arm_index, control_family, replicate),
                    quotient, bootstrap, cfg, f"{arm}|{control_family}={replicate}",
                )
                record[control_family].append({"replicate": replicate, **entry})
                packed.update(arrays)
        freeze["arms"][arm] = record
    return freeze, packed


def load_phase_data(output: Path, role: str) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    with np.load(output / role / "features.npz", allow_pickle=False) as saved:
        features = saved["features"]
    with np.load(output / role / "alpha_metrics.npz", allow_pickle=False) as saved:
        data = {key: saved[key] for key in saved.files}
    data["features"] = features
    with np.load(output / role / "topology_features.npz", allow_pickle=False) as saved:
        topology = {key: saved[key] for key in saved.files}
    return data, topology


def apply_models(
    cfg: dict[str, Any], data: dict[str, np.ndarray], topology: dict[str, np.ndarray],
    models: dict[str, Any], thresholds: dict[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    n_arms, n_views = len(cfg["arms"]), data["quotient_rmse"].shape[-1]
    actions: dict[str, np.ndarray] = {
        "identity": np.zeros((n_arms, n_views), dtype=np.int64),
        "oracle_per_view": np.zeros((n_arms, n_views), dtype=np.int64),
    }
    advantages: dict[str, np.ndarray] = {}
    for family in FAMILIES:
        actions[f"unconstrained_{family}"] = np.zeros((n_arms, n_views), dtype=np.int64)
        actions[f"safe_{family}"] = np.zeros((n_arms, n_views), dtype=np.int64)
        advantages[family] = np.full((n_arms, n_views), np.nan)
    controls: dict[str, np.ndarray] = {}
    for control_family in CONTROL_FAMILIES:
        controls[f"unconstrained_{control_family}"] = np.zeros(
            (n_arms, cfg["control_replicates"], n_views), dtype=np.int64
        )
        controls[f"safe_{control_family}"] = np.zeros_like(controls[f"unconstrained_{control_family}"])
    quotient = data["quotient_rmse"].mean(axis=1)
    for arm_index, arm in enumerate(cfg["arms"]):
        actions["oracle_per_view"][arm_index] = np.argmin(quotient[arm_index], axis=0)
        for family in FAMILIES:
            x = feature_matrix(data, topology, arm_index, family)
            _, base, advantage = safe.predicted_action_and_advantage(
                models["arms"][arm]["families"][family], x
            )
            chosen = thresholds["arms"][arm]["families"][family]["selected_threshold"]
            actions[f"unconstrained_{family}"][arm_index] = safe.threshold_action(base, advantage, 0.0)
            actions[f"safe_{family}"][arm_index] = safe.threshold_action(base, advantage, chosen)
            advantages[family][arm_index] = advantage
        for control_family in CONTROL_FAMILIES:
            for replicate in range(cfg["control_replicates"]):
                x = feature_matrix(data, topology, arm_index, control_family, replicate)
                _, base, advantage = safe.predicted_action_and_advantage(
                    models["arms"][arm][control_family][replicate]["model"], x
                )
                chosen = thresholds["arms"][arm][control_family][replicate]["selected_threshold"]
                controls[f"unconstrained_{control_family}"][arm_index, replicate] = safe.threshold_action(base, advantage, 0.0)
                controls[f"safe_{control_family}"][arm_index, replicate] = safe.threshold_action(base, advantage, chosen)
    return actions, controls, advantages


def selected_values(quotient: np.ndarray, action: np.ndarray) -> np.ndarray:
    return mixed.selected_values(quotient, action)


def summarize(
    cfg: dict[str, Any], data: dict[str, np.ndarray], actions: dict[str, np.ndarray],
    controls: dict[str, np.ndarray], advantages: dict[str, np.ndarray], bootstrap: np.ndarray,
) -> tuple[dict[str, Any], dict[str, Any]]:
    effects: dict[str, Any] = {"arms": {}}
    diagnostics: dict[str, Any] = {"arms": {}}
    iid = np.arange(0, data["quotient_rmse"].shape[-1], 2)
    grouped = np.arange(1, data["quotient_rmse"].shape[-1], 2)
    for arm_index, arm in enumerate(cfg["arms"]):
        quotient = data["quotient_rmse"][arm_index]
        values = {name: selected_values(quotient, action[arm_index]) for name, action in actions.items()}
        for name, control_action in controls.items():
            values[name] = np.mean(
                [selected_values(quotient, control_action[arm_index, replicate]) for replicate in range(cfg["control_replicates"])],
                axis=0,
            )
        arm_effect: dict[str, Any] = {}
        master_values_by_cell: dict[str, dict[str, np.ndarray]] = {}
        for cell, indices in (("iid", iid), ("grouped", grouped), ("balanced", None)):
            per_policy: dict[str, np.ndarray] = {}
            for policy, raw in values.items():
                per_policy[policy] = (
                    raw[:, indices].mean(axis=0)
                    if indices is not None
                    else (0.5 * (raw[:, iid] + raw[:, grouped])).mean(axis=0)
                )
            master_values_by_cell[cell] = per_policy
            arm_effect[cell] = {}
            for policy, master_values in per_policy.items():
                draws = master_values[bootstrap].mean(axis=1)
                row: dict[str, Any] = {
                    "n_masters": len(master_values),
                    "mean_rmse": float(master_values.mean()),
                    "ci95_mean": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
                }
                for baseline, baseline_values in per_policy.items():
                    if baseline == policy:
                        continue
                    delta = master_values - baseline_values
                    delta_draws = delta[bootstrap].mean(axis=1)
                    row[f"minus_{baseline}"] = {
                        "mean": float(delta.mean()),
                        "ci95": [float(np.percentile(delta_draws, 2.5)), float(np.percentile(delta_draws, 97.5))],
                    }
                arm_effect[cell][policy] = row
        arm_effect["action_fractions"] = {}
        for cell, indices in (("iid", iid), ("grouped", grouped)):
            arm_effect["action_fractions"][cell] = {
                name: {str(alpha): float(np.mean(action[arm_index, indices] == alpha_index)) for alpha_index, alpha in enumerate(cfg["alphas"])}
                for name, action in actions.items()
            }
            for name, control_action in controls.items():
                arm_effect["action_fractions"][cell][name] = {
                    str(alpha): float(np.mean(control_action[arm_index, :, indices] == alpha_index))
                    for alpha_index, alpha in enumerate(cfg["alphas"])
                }
        effects["arms"][arm] = arm_effect
        quotient_mean = quotient.mean(axis=0)
        diagnostics["arms"][arm] = {}
        for family in FAMILIES:
            base_action = actions[f"unconstrained_{family}"][arm_index]
            realized = quotient_mean[0] - quotient_mean[base_action, np.arange(len(base_action))]
            diagnostics["arms"][arm][family] = {
                "iid_advantage_benefit_correlation": historical.correlation(advantages[family][arm_index, iid], realized[iid]),
                "grouped_advantage_benefit_correlation": historical.correlation(advantages[family][arm_index, grouped], realized[grouped]),
            }
        safe_topology_iid = master_values_by_cell["iid"]["safe_topology_augmented"] - master_values_by_cell["iid"]["identity"]
        safe_topology_grouped = master_values_by_cell["grouped"]["safe_topology_augmented"] - master_values_by_cell["grouped"]["identity"]
        interaction = safe_topology_grouped - safe_topology_iid
        diagnostics["arms"][arm]["safe_topology_grouped_minus_iid"] = {
            "mean": float(interaction.mean()),
            "ci95": [
                float(np.percentile(interaction[bootstrap].mean(axis=1), 2.5)),
                float(np.percentile(interaction[bootstrap].mean(axis=1), 97.5)),
            ],
        }
    return effects, diagnostics


def deterministic_files(output: Path, roles: tuple[str, ...]) -> list[Path]:
    return sorted(path for role in roles for path in (output / role).rglob("*") if path.is_file())


def source_hashes() -> dict[str, str]:
    return {relative: sha(ROOT / relative) for relative in SOURCE_FILES}


def input_hashes(cfg: dict[str, Any]) -> dict[str, str]:
    return {str(path.relative_to(ROOT)): sha(path) for path in source_inputs(cfg)}


def calibration_phase(cfg: dict[str, Any], output: Path, development: bool) -> None:
    require_clean(development)
    if output.exists():
        raise FileExistsError(output)
    if not development and (ROOT / "data/geometria_proporcional").resolve() not in output.resolve().parents:
        raise ValueError("official output must be below data/geometria_proporcional")
    verify_sources(cfg)
    output.mkdir(parents=True)
    started = time.monotonic()
    samples = [mixed.resource_sample(started, "calibration_start")]
    torch.set_num_threads(cfg["execution"]["torch_threads"])
    torch.use_deterministic_algorithms(True)
    source_cfg, _, models = mixed.source_context(cfg)
    seed = cfg["realizations"]["calibration_seed"]
    views = mixed.fresh_views(cfg, source_cfg, seed)
    write_json(output / "calibration/view_index.json", mixed.view_index(views, seed))
    data = mixed.run_universe("calibration", views, models, source_cfg, cfg, output, started, samples)
    topology = topology_feature_package("calibration", views, data, cfg, output)
    gate_models, packed = fit_calibration(cfg, views, data, topology)
    write_json(output / "calibration/gate_models.json", gate_models)
    frozen.save_npz(output / "calibration/gate_fit.npz", packed)
    write_json(output / "resolved_config.json", cfg)
    files = deterministic_files(output, ("calibration",))
    manifest = {
        "schema_version": "topology-gate-calibration-manifest-v1", "git_head": git_head(),
        "source_hashes": source_hashes(), "input_hashes": input_hashes(cfg),
        "calibration_seed": seed,
        "calibration_files": {str(path.relative_to(output)): sha(path) for path in files},
    }
    write_json(output / "calibration_manifest.json", manifest)
    freeze = {
        "schema_version": "topology-gate-calibration-freeze-v1", "git_head": git_head(),
        "config_sha256": sha(output / "resolved_config.json"),
        "calibration_manifest_sha256": sha(output / "calibration_manifest.json"),
        "gate_models_sha256": sha(output / "calibration/gate_models.json"),
        "future_materialized": False,
    }
    write_json(output / "calibration_freeze.json", freeze)
    write_json(output / "phase_receipt.json", {
        "phase": "calibrate", "calibration_freeze_sha256": sha(output / "calibration_freeze.json"),
        "selection_materialized": False, "adjudication_materialized": False,
    })
    samples.append(mixed.resource_sample(started, "calibration_complete"))
    mixed.enforce(samples[-1], cfg)
    write_json(output / "resource_samples_calibrate.json", samples)
    print(json.dumps(samples[-1], sort_keys=True))


def verify_calibration(cfg: dict[str, Any], output: Path) -> None:
    freeze = json.loads((output / "calibration_freeze.json").read_text())
    manifest = json.loads((output / "calibration_manifest.json").read_text())
    if freeze["git_head"] != git_head() or manifest["git_head"] != git_head():
        raise AssertionError("git HEAD changed after calibration freeze")
    if json.loads((output / "resolved_config.json").read_text()) != cfg or freeze["config_sha256"] != sha(output / "resolved_config.json"):
        raise AssertionError("config changed after calibration freeze")
    if freeze["calibration_manifest_sha256"] != sha(output / "calibration_manifest.json"):
        raise AssertionError("calibration manifest changed")
    if freeze["gate_models_sha256"] != sha(output / "calibration/gate_models.json"):
        raise AssertionError("gate models changed")
    for relative, expected in manifest["calibration_files"].items():
        if sha(output / relative) != expected:
            raise AssertionError(f"calibration file changed: {relative}")


def selection_phase(cfg: dict[str, Any], output: Path, development: bool) -> None:
    require_clean(development)
    if not output.is_dir() or (output / "selection").exists() or (output / "adjudication").exists():
        raise RuntimeError("selection requires calibration-only output")
    verify_calibration(cfg, output)
    started = time.monotonic()
    samples = [mixed.resource_sample(started, "selection_start")]
    torch.set_num_threads(cfg["execution"]["torch_threads"])
    torch.use_deterministic_algorithms(True)
    source_cfg, _, models = mixed.source_context(cfg)
    seed = cfg["realizations"]["selection_seed"]
    views = mixed.fresh_views(cfg, source_cfg, seed)
    calibration_ids = {row["master_id"] for row in json.loads((output / "calibration/view_index.json").read_text())}
    if calibration_ids & {view.private.master_id for view in views}:
        raise AssertionError("calibration and selection masters overlap")
    write_json(output / "selection/view_index.json", mixed.view_index(views, seed))
    data = mixed.run_universe("selection", views, models, source_cfg, cfg, output, started, samples)
    topology = topology_feature_package("selection", views, data, cfg, output)
    calibration_data, calibration_topology = load_phase_data(output, "calibration")
    gate_models = json.loads((output / "calibration/gate_models.json").read_text())
    threshold_freeze, packed = select_thresholds(
        cfg, calibration_data, calibration_topology, data, topology, gate_models
    )
    write_json(output / "selection/thresholds.json", threshold_freeze)
    frozen.save_npz(output / "selection/threshold_statistics.npz", packed)
    files = deterministic_files(output, ("selection",))
    manifest = {
        "schema_version": "topology-gate-selection-manifest-v1", "git_head": git_head(),
        "calibration_freeze_sha256": sha(output / "calibration_freeze.json"),
        "selection_seed": seed,
        "selection_files": {str(path.relative_to(output)): sha(path) for path in files},
    }
    write_json(output / "selection_manifest.json", manifest)
    freeze = {
        "schema_version": "topology-gate-selection-freeze-v1", "git_head": git_head(),
        "calibration_freeze_sha256": sha(output / "calibration_freeze.json"),
        "selection_manifest_sha256": sha(output / "selection_manifest.json"),
        "thresholds_sha256": sha(output / "selection/thresholds.json"),
        "future_materialized": False,
    }
    write_json(output / "selection_freeze.json", freeze)
    write_json(output / "phase_receipt.json", {
        "phase": "select", "selection_freeze_sha256": sha(output / "selection_freeze.json"),
        "adjudication_materialized": False,
    })
    samples.append(mixed.resource_sample(started, "selection_complete"))
    mixed.enforce(samples[-1], cfg)
    write_json(output / "resource_samples_select.json", samples)
    print(json.dumps(samples[-1], sort_keys=True))


def verify_selection(cfg: dict[str, Any], output: Path) -> None:
    verify_calibration(cfg, output)
    freeze = json.loads((output / "selection_freeze.json").read_text())
    manifest = json.loads((output / "selection_manifest.json").read_text())
    receipt = json.loads((output / "phase_receipt.json").read_text())
    if receipt != {
        "phase": "select", "selection_freeze_sha256": sha(output / "selection_freeze.json"),
        "adjudication_materialized": False,
    }:
        raise AssertionError("selection receipt mismatch")
    if freeze["git_head"] != git_head() or manifest["git_head"] != git_head():
        raise AssertionError("git HEAD changed after selection freeze")
    if freeze["calibration_freeze_sha256"] != sha(output / "calibration_freeze.json"):
        raise AssertionError("calibration freeze changed")
    if freeze["selection_manifest_sha256"] != sha(output / "selection_manifest.json"):
        raise AssertionError("selection manifest changed")
    if freeze["thresholds_sha256"] != sha(output / "selection/thresholds.json"):
        raise AssertionError("thresholds changed")
    for relative, expected in manifest["selection_files"].items():
        if sha(output / relative) != expected:
            raise AssertionError(f"selection file changed: {relative}")


def write_replay(output: Path, cfg: dict[str, Any], head: str) -> None:
    checks = "\n".join(
        f"printf '%s  %s\\n' '{sha(ROOT / relative)}' \"$repo/{relative}\" | sha256sum -c -"
        for relative in SOURCE_FILES
    )
    checks += "\n" + "\n".join(
        f"printf '%s  %s\\n' '{sha(path)}' \"$repo/{path.relative_to(ROOT)}\" | sha256sum -c -"
        for path in source_inputs(cfg)
    )
    replay = f'''#!/bin/sh
set -eu
repo=$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd)
[ "$(git -C "$repo" rev-parse HEAD)" = "{head}" ] || {{ echo "wrong git HEAD" >&2; exit 2; }}
[ -z "$(git -C "$repo" status --porcelain)" ] || {{ echo "dirty worktree" >&2; exit 2; }}
{checks}
: "${{OUTPUT_DIR:?set OUTPUT_DIR to a new path}}"
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 "$repo/venv/bin/python" "$repo/experiments/geometria_proporcional/run_proportional_graph_topology_localization_gate.py" --phase calibrate --config "$repo/experiments/geometria_proporcional/configs/proportional_graph_topology_localization_gate_v1.json" --output "$OUTPUT_DIR"
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 "$repo/venv/bin/python" "$repo/experiments/geometria_proporcional/run_proportional_graph_topology_localization_gate.py" --phase select --config "$repo/experiments/geometria_proporcional/configs/proportional_graph_topology_localization_gate_v1.json" --output "$OUTPUT_DIR"
exec env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 "$repo/venv/bin/python" "$repo/experiments/geometria_proporcional/run_proportional_graph_topology_localization_gate.py" --phase evaluate --config "$repo/experiments/geometria_proporcional/configs/proportional_graph_topology_localization_gate_v1.json" --output "$OUTPUT_DIR"
'''
    (output / "replay.sh").write_text(replay)
    (output / "replay.sh").chmod(0o755)


def evaluation_phase(cfg: dict[str, Any], output: Path, development: bool) -> None:
    require_clean(development)
    if not output.is_dir() or (output / "adjudication").exists():
        raise RuntimeError("evaluation requires selected output")
    verify_selection(cfg, output)
    started = time.monotonic()
    samples = [mixed.resource_sample(started, "adjudication_start")]
    torch.set_num_threads(cfg["execution"]["torch_threads"])
    torch.use_deterministic_algorithms(True)
    source_cfg, _, models = mixed.source_context(cfg)
    seed = cfg["realizations"]["adjudication_seed"]
    views = mixed.fresh_views(cfg, source_cfg, seed)
    previous = {
        row["master_id"]
        for role in ("calibration", "selection")
        for row in json.loads((output / role / "view_index.json").read_text())
    }
    if previous & {view.private.master_id for view in views}:
        raise AssertionError("adjudication overlaps an earlier phase")
    write_json(output / "adjudication/view_index.json", mixed.view_index(views, seed))
    data = mixed.run_universe("adjudication", views, models, source_cfg, cfg, output, started, samples)
    topology = topology_feature_package("adjudication", views, data, cfg, output)
    gate_models = json.loads((output / "calibration/gate_models.json").read_text())
    thresholds = json.loads((output / "selection/thresholds.json").read_text())
    actions, controls, advantages = apply_models(cfg, data, topology, gate_models, thresholds)
    frozen.save_npz(output / "adjudication/decisions.npz", {**actions, **controls})
    frozen.save_npz(output / "adjudication/advantages.npz", advantages)
    n_masters = len(views) // 2
    bootstrap = np.random.default_rng(cfg["evaluation_bootstrap_seed"]).integers(
        0, n_masters, size=(cfg["evaluation_bootstrap_replicates"], n_masters)
    )
    frozen.save_npz(output / "bootstrap_indices.npz", {
        "master_id": np.asarray([views[index].private.master_id for index in range(0, len(views), 2)]),
        f"complete_n_{n_masters}": bootstrap,
    })
    effects, diagnostics = summarize(cfg, data, actions, controls, advantages, bootstrap)
    write_json(output / "effects.json", effects)
    write_json(output / "prediction_diagnostics.json", diagnostics)
    write_replay(output, cfg, git_head())
    samples.append(mixed.resource_sample(started, "adjudication_complete"))
    mixed.enforce(samples[-1], cfg)
    write_json(output / "resource_samples_evaluate.json", samples)
    deterministic = sorted(
        path for path in output.rglob("*") if path.is_file() and path.name not in {
            "manifest.json", "resource_samples_calibrate.json", "resource_samples_select.json",
            "resource_samples_evaluate.json", "runtime_observation.json",
        }
    )
    manifest = {
        "schema_version": cfg["schema_version"], "git_head": git_head(),
        "source_hashes": source_hashes(), "input_hashes": input_hashes(cfg),
        "deterministic_files": {str(path.relative_to(output)): sha(path) for path in deterministic},
        "runtime_exclusions": [
            "resource_samples_calibrate.json", "resource_samples_select.json",
            "resource_samples_evaluate.json", "runtime_observation.json",
        ],
    }
    write_json(output / "manifest.json", manifest)
    runtime = mixed.resource_sample(started, "adjudication_complete")
    write_json(output / "runtime_observation.json", runtime)
    print(json.dumps(runtime, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("calibrate", "select", "evaluate"), required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--development", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    output = args.output.resolve()
    if args.phase == "calibrate":
        calibration_phase(cfg, output, args.development)
    elif args.phase == "select":
        selection_phase(cfg, output, args.development)
    else:
        evaluation_phase(cfg, output, args.development)


if __name__ == "__main__":
    main()
