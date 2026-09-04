#!/usr/bin/env python3
"""Fit, calibrate, firewall, and adjudicate a conditional residual-risk gate on CPU."""

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
import run_proportional_graph_topology_localization_gate as topology_gate  # noqa: E402
import torch  # noqa: E402

DEFAULT_CONFIG = ROOT / "experiments/geometria_proporcional/configs/proportional_graph_conditional_risk_gate_v1.json"
SOURCE_FILES = (
    "experiments/geometria_proporcional/PLAN_PROPORTIONAL_CONDITIONAL_RISK_GATE_CPU.md",
    "experiments/geometria_proporcional/configs/proportional_graph_conditional_risk_gate_v1.json",
    "experiments/geometria_proporcional/run_proportional_graph_conditional_risk_gate.py",
    "experiments/geometria_proporcional/run_proportional_graph_topology_localization_gate.py",
    "experiments/geometria_proporcional/run_proportional_graph_fresh_mixed_gate.py",
    "experiments/geometria_proporcional/run_proportional_graph_residual_gate.py",
    "experiments/geometria_proporcional/run_proportional_graph_safe_abstention_gate.py",
    "experiments/geometria_proporcional/run_proportional_graph_frozen_adapters.py",
    "experiments/geometria_proporcional/run_proportional_graph_solver_interface_diagnostic.py",
    "experiments/geometria_proporcional/run_proportional_graph_neural_smoke.py",
    "src/geometria_proporcional/proportional_graph_contract.py",
)
FAMILIES = ("constant_scale", "public_base_scale", "topology_scale")
CONTROL_FAMILY = "topology_permuted_scale"
BASE_FEATURE_ORDER = topology_gate.BASE_FEATURE_ORDER
TOPOLOGY_FEATURE_ORDER = topology_gate.TOPOLOGY_FEATURE_ORDER


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def load_config(path: Path) -> dict[str, Any]:
    cfg = json.loads(path.read_text())
    expected = {
        "schema_version", "source_topology_gate", "source_topology_manifest_sha256",
        "arms", "seeds", "alphas", "realizations", "ridge_lambdas", "ridge_folds",
        "residual_epsilon", "scale_clip", "miscoverage", "control_replicates",
        "topology_control_seed", "selection_bootstrap_replicates",
        "selection_bootstrap_seed", "selection_upper_percentile",
        "evaluation_bootstrap_replicates", "evaluation_bootstrap_seed", "execution",
    }
    if set(cfg) != expected or cfg["schema_version"] != "proportional-graph-conditional-risk-gate-v1":
        raise ValueError("invalid conditional risk schema")
    if cfg["arms"] != ["raw_generic", "raw_typed", "closure_generic", "closure_typed"]:
        raise ValueError("arms changed")
    if cfg["seeds"] != [104729, 130363] or cfg["alphas"] != [0.0, 0.25, 0.5, 0.75, 1.0]:
        raise ValueError("neural factorial changed")
    if cfg["realizations"] != {
        "risk_fit_seed": 2026090809, "risk_calibration_seed": 2026090817,
        "policy_selection_seed": 2026090829, "adjudication_seed": 2026090837,
        "min_eligible_masters": 220,
    }:
        raise ValueError("realization contract changed")
    if cfg["ridge_lambdas"] != [0.0, 0.01, 0.1, 1.0, 10.0, 100.0] or cfg["ridge_folds"] != 5:
        raise ValueError("ridge contract changed")
    if cfg["residual_epsilon"] != 1e-6 or cfg["scale_clip"] != [1e-6, 1.0]:
        raise ValueError("scale contract changed")
    if cfg["miscoverage"] != 0.10 or cfg["control_replicates"] != 16:
        raise ValueError("calibration contract changed")
    if (
        cfg["selection_bootstrap_replicates"] != 2000
        or cfg["selection_upper_percentile"] != 95.0
        or cfg["evaluation_bootstrap_replicates"] != 2000
    ):
        raise ValueError("bootstrap contract changed")
    if cfg["execution"] != {
        "torch_threads": 1, "max_seconds_per_phase": 720,
        "max_rss_gib": 4.0, "sample_every_views": 64,
    }:
        raise ValueError("execution contract changed")
    return cfg


def source_root(cfg: dict[str, Any]) -> Path:
    return ROOT / cfg["source_topology_gate"]


def source_inputs(cfg: dict[str, Any]) -> list[Path]:
    root = source_root(cfg)
    return [root / "manifest.json", root / "resolved_config.json", root / "calibration/gate_models.json"]


def verify_source(cfg: dict[str, Any]) -> dict[str, Any]:
    root = source_root(cfg)
    if sha(root / "manifest.json") != cfg["source_topology_manifest_sha256"]:
        raise AssertionError("topology source manifest hash mismatch")
    manifest = json.loads((root / "manifest.json").read_text())
    for path in source_inputs(cfg)[1:]:
        relative = str(path.relative_to(root))
        if manifest["deterministic_files"].get(relative) != sha(path):
            raise AssertionError(f"topology source mismatch: {relative}")
    source_cfg = json.loads((root / "resolved_config.json").read_text())
    for key in ("arms", "seeds", "alphas"):
        if source_cfg[key] != cfg[key]:
            raise AssertionError(f"source {key} changed")
    return source_cfg


def require_clean(development: bool) -> None:
    if not development and subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout:
        raise RuntimeError("official phase requires clean worktree")


def git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout.strip()


def phase_context(cfg: dict[str, Any]) -> tuple[dict[str, Any], dict[str, dict[int, torch.nn.Module]], dict[str, Any]]:
    source_gate_cfg = verify_source(cfg)
    source_cfg, _, models = mixed.source_context(source_gate_cfg)
    mean_models = json.loads((source_root(cfg) / "calibration/gate_models.json").read_text())
    return source_cfg, models, mean_models


def topology_cfg(cfg: dict[str, Any], source_gate_cfg: dict[str, Any]) -> dict[str, Any]:
    result = dict(source_gate_cfg)
    result.update({
        "arms": cfg["arms"], "seeds": cfg["seeds"], "alphas": cfg["alphas"],
        "realizations": cfg["realizations"], "control_replicates": cfg["control_replicates"],
        "topology_control_seed": cfg["topology_control_seed"], "execution": cfg["execution"],
    })
    return result


def generate_phase(
    role: str, seed_key: str, cfg: dict[str, Any], output: Path,
    source_cfg: dict[str, Any], forward_models: dict[str, dict[int, torch.nn.Module]],
    started: float, samples: list[dict[str, Any]],
) -> tuple[list[Any], dict[str, np.ndarray], dict[str, np.ndarray]]:
    seed = cfg["realizations"][seed_key]
    views = mixed.fresh_views(cfg, source_cfg, seed)
    write_json(output / role / "view_index.json", mixed.view_index(views, seed))
    data = mixed.run_universe(role, views, forward_models, source_cfg, cfg, output, started, samples)
    source_gate_cfg = verify_source(cfg)
    topology = topology_gate.topology_feature_package(
        role, views, data, topology_cfg(cfg, source_gate_cfg), output
    )
    return views, data, topology


def load_phase_data(output: Path, role: str) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    with np.load(output / role / "features.npz", allow_pickle=False) as saved:
        features = saved["features"]
    with np.load(output / role / "alpha_metrics.npz", allow_pickle=False) as saved:
        data = {key: saved[key] for key in saved.files}
    data["features"] = features
    with np.load(output / role / "topology_features.npz", allow_pickle=False) as saved:
        topology = {key: saved[key] for key in saved.files}
    return data, topology


def feature_matrix(
    data: dict[str, np.ndarray], topology: dict[str, np.ndarray], arm_index: int,
    family: str, replicate: int | None = None,
) -> np.ndarray:
    base = data["features"].mean(axis=1)[arm_index]
    if family == "public_base_scale":
        return base
    if family == "topology_scale":
        return np.column_stack((base, topology["true"][arm_index].mean(axis=0)))
    if family == CONTROL_FAMILY:
        if replicate is None:
            raise ValueError("topology control requires replicate")
        return np.column_stack((base, topology["control"][arm_index, :, replicate].mean(axis=0)))
    raise ValueError(f"family has no feature matrix: {family}")


def realized_deltas(data: dict[str, np.ndarray]) -> np.ndarray:
    quotient = data["quotient_rmse"].mean(axis=1)
    return np.transpose(quotient[:, 1:] - quotient[:, :1], (0, 2, 1))


def mean_predictions(
    cfg: dict[str, Any], data: dict[str, np.ndarray], topology: dict[str, np.ndarray],
    mean_models: dict[str, Any],
) -> np.ndarray:
    result = np.full((len(cfg["arms"]), data["quotient_rmse"].shape[-1], len(cfg["alphas"]) - 1), np.nan)
    for arm_index, arm in enumerate(cfg["arms"]):
        x = topology_gate.feature_matrix(data, topology, arm_index, "topology_augmented")
        result[arm_index] = mixed.predict_report(
            mean_models["arms"][arm]["families"]["topology_augmented"], x
        )
    if not np.all(np.isfinite(result)):
        raise RuntimeError("frozen mean prediction failed")
    return result


def fit_scale_models(
    cfg: dict[str, Any], views: list[Any], data: dict[str, np.ndarray],
    topology: dict[str, np.ndarray], mean_models: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    iid = np.arange(0, len(views), 2)
    masters = np.asarray([views[index].private.master_id for index in iid])
    mu = mean_predictions(cfg, data, topology, mean_models)
    delta = realized_deltas(data)
    z = np.log(np.abs(delta - mu) + cfg["residual_epsilon"])
    record: dict[str, Any] = {
        "base_feature_order": list(BASE_FEATURE_ORDER),
        "topology_feature_order": list(TOPOLOGY_FEATURE_ORDER), "arms": {},
    }
    packed: dict[str, np.ndarray] = {"mu": mu, "delta": delta, "log_absolute_residual": z, "iid_index": iid}
    for arm_index, arm in enumerate(cfg["arms"]):
        arm_record: dict[str, Any] = {"families": {}, CONTROL_FAMILY: []}
        for family in ("public_base_scale", "topology_scale"):
            x = feature_matrix(data, topology, arm_index, family)[iid]
            model, prediction, oof = historical.fit_gate(x, z[arm_index, iid], x, masters, cfg)
            model["oof_mse"] = float(np.mean((oof - z[arm_index, iid]) ** 2))
            arm_record["families"][family] = model
            packed[f"{arm}|{family}|prediction"] = prediction
            packed[f"{arm}|{family}|oof_prediction"] = oof
        for replicate in range(cfg["control_replicates"]):
            x = feature_matrix(data, topology, arm_index, CONTROL_FAMILY, replicate)[iid]
            model, prediction, oof = historical.fit_gate(x, z[arm_index, iid], x, masters, cfg)
            model["oof_mse"] = float(np.mean((oof - z[arm_index, iid]) ** 2))
            arm_record[CONTROL_FAMILY].append({"replicate": replicate, "model": model})
            packed[f"{arm}|{CONTROL_FAMILY}={replicate}|prediction"] = prediction
            packed[f"{arm}|{CONTROL_FAMILY}={replicate}|oof_prediction"] = oof
        record["arms"][arm] = arm_record
    return record, packed


def predict_scale(report: dict[str, Any], x: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    low, high = cfg["scale_clip"]
    return np.clip(np.exp(mixed.predict_report(report, x)), low, high)


def all_scales(
    cfg: dict[str, Any], data: dict[str, np.ndarray], topology: dict[str, np.ndarray],
    scale_models: dict[str, Any],
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    n_arms, n_views, n_outputs = len(cfg["arms"]), data["quotient_rmse"].shape[-1], len(cfg["alphas"]) - 1
    scales = {"constant_scale": np.ones((n_arms, n_views, n_outputs))}
    for family in ("public_base_scale", "topology_scale"):
        scales[family] = np.full((n_arms, n_views, n_outputs), np.nan)
    controls = np.full((n_arms, cfg["control_replicates"], n_views, n_outputs), np.nan)
    for arm_index, arm in enumerate(cfg["arms"]):
        for family in ("public_base_scale", "topology_scale"):
            scales[family][arm_index] = predict_scale(
                scale_models["arms"][arm]["families"][family],
                feature_matrix(data, topology, arm_index, family), cfg,
            )
        for replicate in range(cfg["control_replicates"]):
            controls[arm_index, replicate] = predict_scale(
                scale_models["arms"][arm][CONTROL_FAMILY][replicate]["model"],
                feature_matrix(data, topology, arm_index, CONTROL_FAMILY, replicate), cfg,
            )
    if any(not np.all(np.isfinite(value)) for value in scales.values()) or not np.all(np.isfinite(controls)):
        raise RuntimeError("scale prediction failed")
    return scales, controls


def conformal_quantile(scores: np.ndarray, miscoverage: float) -> tuple[float, int]:
    values = np.sort(np.asarray(scores, dtype=np.float64))
    rank = int(math.ceil((len(values) + 1) * (1.0 - miscoverage)))
    if not len(values) or rank > len(values):
        raise ValueError("invalid split-conformal sample size")
    return float(values[rank - 1]), rank


def calibrate_risk(
    cfg: dict[str, Any], data: dict[str, np.ndarray], topology: dict[str, np.ndarray],
    mean_models: dict[str, Any], scale_models: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    iid = np.arange(0, data["quotient_rmse"].shape[-1], 2)
    delta, mu = realized_deltas(data), mean_predictions(cfg, data, topology, mean_models)
    scales, controls = all_scales(cfg, data, topology, scale_models)
    calibration: dict[str, Any] = {"miscoverage": cfg["miscoverage"], "arms": {}}
    packed: dict[str, np.ndarray] = {"delta": delta, "mu": mu, "iid_index": iid, CONTROL_FAMILY: controls}
    for family, value in scales.items():
        packed[family] = value
    for arm_index, arm in enumerate(cfg["arms"]):
        arm_record: dict[str, Any] = {"families": {}, CONTROL_FAMILY: []}
        for family in FAMILIES:
            scores = np.max((delta[arm_index, iid] - mu[arm_index, iid]) / scales[family][arm_index, iid], axis=1)
            q, rank = conformal_quantile(scores, cfg["miscoverage"])
            arm_record["families"][family] = {"q": q, "rank": rank, "n": len(scores)}
            packed[f"{arm}|{family}|score"] = scores
        for replicate in range(cfg["control_replicates"]):
            scores = np.max((delta[arm_index, iid] - mu[arm_index, iid]) / controls[arm_index, replicate, iid], axis=1)
            q, rank = conformal_quantile(scores, cfg["miscoverage"])
            arm_record[CONTROL_FAMILY].append({"replicate": replicate, "q": q, "rank": rank, "n": len(scores)})
            packed[f"{arm}|{CONTROL_FAMILY}={replicate}|score"] = scores
        calibration["arms"][arm] = arm_record
    return calibration, packed


def bounded_action(mu: np.ndarray, sigma: np.ndarray, q: float) -> tuple[np.ndarray, np.ndarray]:
    upper = mu + q * sigma
    best = np.argmin(upper, axis=1)
    action = np.where(upper[np.arange(len(upper)), best] < 0.0, best + 1, 0).astype(np.int64)
    return action, upper


def calibrated_actions(
    cfg: dict[str, Any], data: dict[str, np.ndarray], topology: dict[str, np.ndarray],
    mean_models: dict[str, Any], scale_models: dict[str, Any], calibration: dict[str, Any],
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, np.ndarray], np.ndarray]:
    mu = mean_predictions(cfg, data, topology, mean_models)
    scales, control_scales = all_scales(cfg, data, topology, scale_models)
    actions = {family: np.zeros(mu.shape[:2], dtype=np.int64) for family in FAMILIES}
    uppers = {family: np.full_like(mu, np.nan) for family in FAMILIES}
    controls = np.zeros((len(cfg["arms"]), cfg["control_replicates"], mu.shape[1]), dtype=np.int64)
    control_uppers = np.full_like(control_scales, np.nan)
    for arm_index, arm in enumerate(cfg["arms"]):
        for family in FAMILIES:
            actions[family][arm_index], uppers[family][arm_index] = bounded_action(
                mu[arm_index], scales[family][arm_index], calibration["arms"][arm]["families"][family]["q"]
            )
        for replicate in range(cfg["control_replicates"]):
            controls[arm_index, replicate], control_uppers[arm_index, replicate] = bounded_action(
                mu[arm_index], control_scales[arm_index, replicate],
                calibration["arms"][arm][CONTROL_FAMILY][replicate]["q"],
            )
    return actions, controls, uppers, control_uppers


def selected_delta(quotient: np.ndarray, action: np.ndarray) -> np.ndarray:
    values = mixed.selected_values(quotient, action)
    return values.mean(axis=0) - quotient[:, 0].mean(axis=0)


def firewall_one(delta: np.ndarray, bootstrap: np.ndarray, percentile: float) -> dict[str, Any]:
    iid_delta = delta[np.arange(0, len(delta), 2)]
    observed = float(iid_delta.mean())
    boot = iid_delta[bootstrap].mean(axis=1)
    radius = float(np.percentile(boot - observed, percentile))
    upper = observed + radius
    return {"mean_iid": observed, "bootstrap_upper_radius": radius, "upper_iid": upper, "deploy": bool(upper <= 0.0)}


def select_policies(
    cfg: dict[str, Any], data: dict[str, np.ndarray], topology: dict[str, np.ndarray],
    mean_models: dict[str, Any], scale_models: dict[str, Any], calibration: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    actions, controls, uppers, control_uppers = calibrated_actions(
        cfg, data, topology, mean_models, scale_models, calibration
    )
    mu = mean_predictions(cfg, data, topology, mean_models)
    scales, control_scales = all_scales(cfg, data, topology, scale_models)
    n_masters = data["quotient_rmse"].shape[-1] // 2
    bootstrap = np.random.default_rng(cfg["selection_bootstrap_seed"]).integers(
        0, n_masters, size=(cfg["selection_bootstrap_replicates"], n_masters)
    )
    freeze: dict[str, Any] = {"arms": {}}
    packed: dict[str, np.ndarray] = {
        "bootstrap_indices": bootstrap, "mu": mu,
        "control_scale": control_scales, "control_upper": control_uppers,
    }
    for family, value in uppers.items():
        packed[f"{family}|upper"] = value
        packed[f"{family}|scale"] = scales[family]
    for arm_index, arm in enumerate(cfg["arms"]):
        quotient = data["quotient_rmse"][arm_index]
        record: dict[str, Any] = {"families": {}, CONTROL_FAMILY: []}
        for family in FAMILIES:
            entry = firewall_one(selected_delta(quotient, actions[family][arm_index]), bootstrap, cfg["selection_upper_percentile"])
            record["families"][family] = entry
            packed[f"{arm}|{family}|calibrated_action"] = actions[family][arm_index]
        for replicate in range(cfg["control_replicates"]):
            entry = firewall_one(selected_delta(quotient, controls[arm_index, replicate]), bootstrap, cfg["selection_upper_percentile"])
            record[CONTROL_FAMILY].append({"replicate": replicate, **entry})
            packed[f"{arm}|{CONTROL_FAMILY}={replicate}|calibrated_action"] = controls[arm_index, replicate]
        freeze["arms"][arm] = record
    return freeze, packed


def apply_firewall(
    cfg: dict[str, Any], actions: dict[str, np.ndarray], controls: np.ndarray, freeze: dict[str, Any],
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    deployed = {family: value.copy() for family, value in actions.items()}
    deployed_controls = controls.copy()
    for arm_index, arm in enumerate(cfg["arms"]):
        for family in FAMILIES:
            if not freeze["arms"][arm]["families"][family]["deploy"]:
                deployed[family][arm_index] = 0
        for replicate in range(cfg["control_replicates"]):
            if not freeze["arms"][arm][CONTROL_FAMILY][replicate]["deploy"]:
                deployed_controls[arm_index, replicate] = 0
    return deployed, deployed_controls


def interval(values: np.ndarray, bootstrap: np.ndarray) -> dict[str, Any]:
    draws = values[bootstrap].mean(axis=1)
    return {"mean": float(values.mean()), "ci95": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]}


def summarize(
    cfg: dict[str, Any], data: dict[str, np.ndarray], delta: np.ndarray, mu: np.ndarray,
    scales: dict[str, np.ndarray], control_scales: np.ndarray,
    actions: dict[str, np.ndarray], controls: np.ndarray,
    deployed: dict[str, np.ndarray], deployed_controls: np.ndarray,
    calibration: dict[str, Any], bootstrap: np.ndarray,
) -> tuple[dict[str, Any], dict[str, Any]]:
    iid = np.arange(0, data["quotient_rmse"].shape[-1], 2)
    grouped = np.arange(1, data["quotient_rmse"].shape[-1], 2)
    effects: dict[str, Any] = {"arms": {}}
    diagnostics: dict[str, Any] = {"coverage_jurisdiction": "marginal IID per-view simultaneous across nonidentity alphas", "arms": {}}
    for arm_index, arm in enumerate(cfg["arms"]):
        quotient = data["quotient_rmse"][arm_index]
        value: dict[str, np.ndarray] = {
            "identity": quotient[:, 0],
            "oracle_per_view": mixed.selected_values(quotient, np.argmin(quotient.mean(axis=0), axis=0)),
        }
        for family in FAMILIES:
            value[f"calibrated_{family}"] = mixed.selected_values(quotient, actions[family][arm_index])
            value[f"deployed_{family}"] = mixed.selected_values(quotient, deployed[family][arm_index])
        value[f"calibrated_{CONTROL_FAMILY}"] = np.mean([
            mixed.selected_values(quotient, controls[arm_index, replicate])
            for replicate in range(cfg["control_replicates"])
        ], axis=0)
        value[f"deployed_{CONTROL_FAMILY}"] = np.mean([
            mixed.selected_values(quotient, deployed_controls[arm_index, replicate])
            for replicate in range(cfg["control_replicates"])
        ], axis=0)
        arm_effect: dict[str, Any] = {"cells": {}, "action_fractions": {}}
        cell_values: dict[str, dict[str, np.ndarray]] = {}
        for cell, indices in (("iid", iid), ("grouped", grouped), ("balanced", None)):
            per_policy = {
                name: (raw[:, indices].mean(axis=0) if indices is not None else (0.5 * (raw[:, iid] + raw[:, grouped])).mean(axis=0))
                for name, raw in value.items()
            }
            cell_values[cell] = per_policy
            arm_effect["cells"][cell] = {}
            for name, values in per_policy.items():
                row = {"n_masters": len(values), "mean_rmse": interval(values, bootstrap)}
                row["minus_identity"] = interval(values - per_policy["identity"], bootstrap)
                if name.startswith(("calibrated_topology_scale", "deployed_topology_scale")):
                    prefix = name.split("topology_scale")[0]
                    for baseline in ("constant_scale", "public_base_scale", CONTROL_FAMILY):
                        other = per_policy[f"{prefix}{baseline}"]
                        row[f"minus_{prefix}{baseline}"] = interval(values - other, bootstrap)
                arm_effect["cells"][cell][name] = row
        for cell, indices in (("iid", iid), ("grouped", grouped)):
            arm_effect["action_fractions"][cell] = {}
            for family in FAMILIES:
                for stage, action in (("calibrated", actions[family][arm_index]), ("deployed", deployed[family][arm_index])):
                    arm_effect["action_fractions"][cell][f"{stage}_{family}"] = {
                        str(alpha): float(np.mean(action[indices] == alpha_index))
                        for alpha_index, alpha in enumerate(cfg["alphas"])
                    }
            for stage, action in (("calibrated", controls[arm_index]), ("deployed", deployed_controls[arm_index])):
                arm_effect["action_fractions"][cell][f"{stage}_{CONTROL_FAMILY}"] = {
                    str(alpha): float(np.mean(action[:, indices] == alpha_index))
                    for alpha_index, alpha in enumerate(cfg["alphas"])
                }
        effects["arms"][arm] = arm_effect
        arm_diag: dict[str, Any] = {"coverage": {}, "acted_harm": {}, "interaction_grouped_minus_iid": {}}
        for family in FAMILIES:
            q = calibration["arms"][arm]["families"][family]["q"]
            covered = np.all(delta[arm_index] <= mu[arm_index] + q * scales[family][arm_index], axis=1)
            arm_diag["coverage"][family] = {
                "iid": float(np.mean(covered[iid])), "grouped": float(np.mean(covered[grouped]))
            }
            for stage, action in (("calibrated", actions[family][arm_index]), ("deployed", deployed[family][arm_index])):
                realized = selected_delta(quotient, action)
                arm_diag["acted_harm"][f"{stage}_{family}"] = {}
                for cell, indices in (("iid", iid), ("grouped", grouped)):
                    acted = action[indices] > 0
                    acted_delta = realized[indices][acted]
                    arm_diag["acted_harm"][f"{stage}_{family}"][cell] = {
                        "n": int(acted.sum()),
                        "mean_delta": float(acted_delta.mean()) if len(acted_delta) else None,
                        "positive_fraction": float(np.mean(acted_delta > 0)) if len(acted_delta) else None,
                    }
                iid_effect = cell_values["iid"][f"{stage}_{family}"] - cell_values["iid"]["identity"]
                grouped_effect = cell_values["grouped"][f"{stage}_{family}"] - cell_values["grouped"]["identity"]
                arm_diag["interaction_grouped_minus_iid"][f"{stage}_{family}"] = interval(grouped_effect - iid_effect, bootstrap)
        control_coverage = []
        for replicate in range(cfg["control_replicates"]):
            q = calibration["arms"][arm][CONTROL_FAMILY][replicate]["q"]
            control_coverage.append(np.all(
                delta[arm_index] <= mu[arm_index] + q * control_scales[arm_index, replicate], axis=1
            ))
        stacked = np.asarray(control_coverage)
        arm_diag["coverage"][CONTROL_FAMILY] = {
            "iid": float(np.mean(stacked[:, iid])), "grouped": float(np.mean(stacked[:, grouped]))
        }
        diagnostics["arms"][arm] = arm_diag
    return effects, diagnostics


def deterministic_files(output: Path, roles: tuple[str, ...]) -> list[Path]:
    return sorted(path for role in roles for path in (output / role).rglob("*") if path.is_file())


def source_hashes() -> dict[str, str]:
    return {relative: sha(ROOT / relative) for relative in SOURCE_FILES}


def input_hashes(cfg: dict[str, Any]) -> dict[str, str]:
    return {str(path.relative_to(ROOT)): sha(path) for path in source_inputs(cfg)}


def write_phase_freeze(
    output: Path, phase: str, roles: tuple[str, ...], protected: dict[str, Path], future: list[str]
) -> None:
    files = deterministic_files(output, roles)
    manifest = {
        "schema_version": f"conditional-risk-{phase}-manifest-v1", "git_head": git_head(),
        "roles": list(roles),
        "files": {str(path.relative_to(output)): sha(path) for path in files},
    }
    write_json(output / f"{phase}_manifest.json", manifest)
    freeze = {
        "schema_version": f"conditional-risk-{phase}-freeze-v1", "git_head": git_head(),
        "manifest_sha256": sha(output / f"{phase}_manifest.json"),
        "protected": {name: sha(path) for name, path in protected.items()},
        "future_materialized": {name: (output / name).exists() for name in future},
    }
    if any(freeze["future_materialized"].values()):
        raise AssertionError("future phase materialized before freeze")
    write_json(output / f"{phase}_freeze.json", freeze)


def verify_phase(output: Path, phase: str) -> None:
    manifest_path, freeze_path = output / f"{phase}_manifest.json", output / f"{phase}_freeze.json"
    manifest, freeze = json.loads(manifest_path.read_text()), json.loads(freeze_path.read_text())
    if manifest["git_head"] != git_head() or freeze["git_head"] != git_head():
        raise AssertionError("git HEAD changed after freeze")
    if freeze["manifest_sha256"] != sha(manifest_path):
        raise AssertionError("phase manifest changed")
    actual = {
        str(path.relative_to(output))
        for role in manifest["roles"] for path in (output / role).rglob("*") if path.is_file()
    }
    if actual != set(manifest["files"]):
        raise AssertionError("frozen phase file set changed")
    for relative, expected in manifest["files"].items():
        if sha(output / relative) != expected:
            raise AssertionError(f"frozen file changed: {relative}")
    for relative, expected in freeze["protected"].items():
        if sha(output / relative) != expected:
            raise AssertionError(f"protected file changed: {relative}")


def verify_resolved_config(cfg: dict[str, Any], output: Path) -> None:
    if json.loads((output / "resolved_config.json").read_text()) != cfg:
        raise AssertionError("runtime config differs from frozen config")


def prior_ids(output: Path, roles: tuple[str, ...]) -> set[str]:
    return {
        row["master_id"] for role in roles
        for row in json.loads((output / role / "view_index.json").read_text())
    }


def setup() -> None:
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def risk_fit_phase(cfg: dict[str, Any], output: Path, development: bool) -> None:
    require_clean(development)
    if output.exists():
        raise FileExistsError(output)
    if not development and (ROOT / "data/geometria_proporcional").resolve() not in output.resolve().parents:
        raise ValueError("official output must be below data/geometria_proporcional")
    output.mkdir(parents=True)
    started, samples = time.monotonic(), []
    setup()
    source_cfg, forward_models, mean_models = phase_context(cfg)
    views, data, topology = generate_phase(
        "risk_fit", "risk_fit_seed", cfg, output, source_cfg, forward_models, started, samples
    )
    models, packed = fit_scale_models(cfg, views, data, topology, mean_models)
    write_json(output / "risk_fit/scale_models.json", models)
    frozen.save_npz(output / "risk_fit/scale_fit.npz", packed)
    write_json(output / "resolved_config.json", cfg)
    write_phase_freeze(output, "risk_fit", ("risk_fit",), {
        "resolved_config.json": output / "resolved_config.json",
        "risk_fit/scale_models.json": output / "risk_fit/scale_models.json",
    }, ["risk_calibration", "policy_selection", "adjudication"])
    write_json(output / "phase_receipt.json", {"phase": "risk_fit", "next": "risk_calibration"})
    samples.append(mixed.resource_sample(started, "risk_fit_complete")); mixed.enforce(samples[-1], cfg)
    write_json(output / "resource_samples_risk_fit.json", samples)
    print(json.dumps(samples[-1], sort_keys=True))


def risk_calibration_phase(cfg: dict[str, Any], output: Path, development: bool) -> None:
    require_clean(development); verify_phase(output, "risk_fit"); verify_resolved_config(cfg, output)
    if (output / "risk_calibration").exists():
        raise RuntimeError("risk calibration already exists")
    started, samples = time.monotonic(), []; setup()
    source_cfg, forward_models, mean_models = phase_context(cfg)
    views, data, topology = generate_phase(
        "risk_calibration", "risk_calibration_seed", cfg, output, source_cfg, forward_models, started, samples
    )
    if prior_ids(output, ("risk_fit",)) & {view.private.master_id for view in views}:
        raise AssertionError("risk fit and calibration overlap")
    scale_models = json.loads((output / "risk_fit/scale_models.json").read_text())
    calibration, packed = calibrate_risk(cfg, data, topology, mean_models, scale_models)
    write_json(output / "risk_calibration/quantiles.json", calibration)
    frozen.save_npz(output / "risk_calibration/calibration_scores.npz", packed)
    write_phase_freeze(output, "risk_calibration", ("risk_calibration",), {
        "risk_fit_freeze.json": output / "risk_fit_freeze.json",
        "risk_calibration/quantiles.json": output / "risk_calibration/quantiles.json",
    }, ["policy_selection", "adjudication"])
    write_json(output / "phase_receipt.json", {"phase": "risk_calibration", "next": "policy_selection"})
    samples.append(mixed.resource_sample(started, "risk_calibration_complete")); mixed.enforce(samples[-1], cfg)
    write_json(output / "resource_samples_risk_calibration.json", samples)
    print(json.dumps(samples[-1], sort_keys=True))


def policy_selection_phase(cfg: dict[str, Any], output: Path, development: bool) -> None:
    require_clean(development); verify_phase(output, "risk_fit"); verify_phase(output, "risk_calibration")
    verify_resolved_config(cfg, output)
    if (output / "policy_selection").exists() or (output / "adjudication").exists():
        raise RuntimeError("selection requires risk-calibration-only output")
    started, samples = time.monotonic(), []; setup()
    source_cfg, forward_models, mean_models = phase_context(cfg)
    views, data, topology = generate_phase(
        "policy_selection", "policy_selection_seed", cfg, output, source_cfg, forward_models, started, samples
    )
    if prior_ids(output, ("risk_fit", "risk_calibration")) & {view.private.master_id for view in views}:
        raise AssertionError("policy selection overlaps prior phase")
    scale_models = json.loads((output / "risk_fit/scale_models.json").read_text())
    calibration = json.loads((output / "risk_calibration/quantiles.json").read_text())
    freeze, packed = select_policies(cfg, data, topology, mean_models, scale_models, calibration)
    write_json(output / "policy_selection/firewall.json", freeze)
    frozen.save_npz(output / "policy_selection/selection_statistics.npz", packed)
    write_phase_freeze(output, "policy_selection", ("policy_selection",), {
        "risk_fit_freeze.json": output / "risk_fit_freeze.json",
        "risk_calibration_freeze.json": output / "risk_calibration_freeze.json",
        "policy_selection/firewall.json": output / "policy_selection/firewall.json",
    }, ["adjudication"])
    write_json(output / "phase_receipt.json", {"phase": "policy_selection", "next": "adjudication"})
    samples.append(mixed.resource_sample(started, "policy_selection_complete")); mixed.enforce(samples[-1], cfg)
    write_json(output / "resource_samples_policy_selection.json", samples)
    print(json.dumps(samples[-1], sort_keys=True))


def write_replay(output: Path, cfg: dict[str, Any], head: str) -> None:
    checks = "\n".join(
        f"printf '%s  %s\\n' '{sha(ROOT / relative)}' \"$repo/{relative}\" | sha256sum -c -"
        for relative in SOURCE_FILES
    )
    checks += "\n" + "\n".join(
        f"printf '%s  %s\\n' '{sha(path)}' \"$repo/{path.relative_to(ROOT)}\" | sha256sum -c -"
        for path in source_inputs(cfg)
    )
    command = 'env CUDA_VISIBLE_DEVICES=\'\' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 "$repo/venv/bin/python" "$repo/experiments/geometria_proporcional/run_proportional_graph_conditional_risk_gate.py"'
    replay = f'''#!/bin/sh
set -eu
repo=$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd)
[ "$(git -C "$repo" rev-parse HEAD)" = "{head}" ] || {{ echo "wrong git HEAD" >&2; exit 2; }}
[ -z "$(git -C "$repo" status --porcelain)" ] || {{ echo "dirty worktree" >&2; exit 2; }}
{checks}
: "${{OUTPUT_DIR:?set OUTPUT_DIR to a new path}}"
{command} --phase risk-fit --config "$repo/experiments/geometria_proporcional/configs/proportional_graph_conditional_risk_gate_v1.json" --output "$OUTPUT_DIR"
{command} --phase risk-calibrate --config "$repo/experiments/geometria_proporcional/configs/proportional_graph_conditional_risk_gate_v1.json" --output "$OUTPUT_DIR"
{command} --phase select --config "$repo/experiments/geometria_proporcional/configs/proportional_graph_conditional_risk_gate_v1.json" --output "$OUTPUT_DIR"
exec {command} --phase evaluate --config "$repo/experiments/geometria_proporcional/configs/proportional_graph_conditional_risk_gate_v1.json" --output "$OUTPUT_DIR"
'''
    (output / "replay.sh").write_text(replay); (output / "replay.sh").chmod(0o755)


def evaluation_phase(cfg: dict[str, Any], output: Path, development: bool) -> None:
    require_clean(development)
    for phase in ("risk_fit", "risk_calibration", "policy_selection"):
        verify_phase(output, phase)
    verify_resolved_config(cfg, output)
    if (output / "adjudication").exists():
        raise RuntimeError("adjudication already exists")
    started, samples = time.monotonic(), []; setup()
    source_cfg, forward_models, mean_models = phase_context(cfg)
    views, data, topology = generate_phase(
        "adjudication", "adjudication_seed", cfg, output, source_cfg, forward_models, started, samples
    )
    if prior_ids(output, ("risk_fit", "risk_calibration", "policy_selection")) & {view.private.master_id for view in views}:
        raise AssertionError("adjudication overlaps prior phase")
    scale_models = json.loads((output / "risk_fit/scale_models.json").read_text())
    calibration = json.loads((output / "risk_calibration/quantiles.json").read_text())
    firewall = json.loads((output / "policy_selection/firewall.json").read_text())
    delta, mu = realized_deltas(data), mean_predictions(cfg, data, topology, mean_models)
    scales, control_scales = all_scales(cfg, data, topology, scale_models)
    actions, controls, uppers, control_uppers = calibrated_actions(
        cfg, data, topology, mean_models, scale_models, calibration
    )
    deployed, deployed_controls = apply_firewall(cfg, actions, controls, firewall)
    arrays: dict[str, np.ndarray] = {"mu": mu, "delta": delta, "control_scale": control_scales,
                                          "calibrated_control": controls, "deployed_control": deployed_controls,
                                          "control_upper": control_uppers}
    for family in FAMILIES:
        arrays[f"scale|{family}"] = scales[family]
        arrays[f"upper|{family}"] = uppers[family]
        arrays[f"calibrated|{family}"] = actions[family]
        arrays[f"deployed|{family}"] = deployed[family]
    frozen.save_npz(output / "adjudication/decisions_and_risk.npz", arrays)
    n_masters = len(views) // 2
    bootstrap = np.random.default_rng(cfg["evaluation_bootstrap_seed"]).integers(
        0, n_masters, size=(cfg["evaluation_bootstrap_replicates"], n_masters)
    )
    frozen.save_npz(output / "bootstrap_indices.npz", {"indices": bootstrap})
    effects, diagnostics = summarize(
        cfg, data, delta, mu, scales, control_scales, actions, controls,
        deployed, deployed_controls, calibration, bootstrap,
    )
    write_json(output / "effects.json", effects); write_json(output / "risk_diagnostics.json", diagnostics)
    write_replay(output, cfg, git_head())
    samples.append(mixed.resource_sample(started, "adjudication_complete")); mixed.enforce(samples[-1], cfg)
    write_json(output / "resource_samples_evaluate.json", samples)
    deterministic = sorted(path for path in output.rglob("*") if path.is_file() and path.name not in {
        "manifest.json", "resource_samples_risk_fit.json", "resource_samples_risk_calibration.json",
        "resource_samples_policy_selection.json", "resource_samples_evaluate.json", "runtime_observation.json",
        "phase_receipt.json",
    })
    write_json(output / "manifest.json", {
        "schema_version": cfg["schema_version"], "git_head": git_head(),
        "source_hashes": source_hashes(), "input_hashes": input_hashes(cfg),
        "deterministic_files": {str(path.relative_to(output)): sha(path) for path in deterministic},
        "runtime_exclusions": ["resource_samples_risk_fit.json", "resource_samples_risk_calibration.json",
                               "resource_samples_policy_selection.json", "resource_samples_evaluate.json",
                               "runtime_observation.json", "phase_receipt.json"],
    })
    runtime = mixed.resource_sample(started, "adjudication_complete"); write_json(output / "runtime_observation.json", runtime)
    write_json(output / "phase_receipt.json", {"phase": "complete", "manifest_sha256": sha(output / "manifest.json")})
    print(json.dumps(runtime, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("risk-fit", "risk-calibrate", "select", "evaluate"), required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--development", action="store_true")
    args = parser.parse_args()
    cfg, output = load_config(args.config.resolve()), args.output.resolve()
    if args.phase == "risk-fit": risk_fit_phase(cfg, output, args.development)
    elif args.phase == "risk-calibrate": risk_calibration_phase(cfg, output, args.development)
    elif args.phase == "select": policy_selection_phase(cfg, output, args.development)
    else: evaluation_phase(cfg, output, args.development)


if __name__ == "__main__":
    main()
