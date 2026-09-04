#!/usr/bin/env python3
"""Diagnose a one-sided signed-residual risk interface from opened CPU artifacts."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
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

import run_proportional_graph_conditional_risk_gate as conditional  # noqa: E402
import run_proportional_graph_fresh_mixed_gate as mixed  # noqa: E402
import run_proportional_graph_frozen_adapters as frozen  # noqa: E402
import run_proportional_graph_residual_gate as historical  # noqa: E402
from sklearn.linear_model import QuantileRegressor  # noqa: E402

DEFAULT_CONFIG = ROOT / "experiments/geometria_proporcional/configs/proportional_graph_signed_tail_diagnostic_v1.json"
SOURCE_FILES = (
    "experiments/geometria_proporcional/PLAN_PROPORTIONAL_SIGNED_TAIL_DIAGNOSTIC_CPU.md",
    "experiments/geometria_proporcional/configs/proportional_graph_signed_tail_diagnostic_v1.json",
    "experiments/geometria_proporcional/run_proportional_graph_signed_tail_diagnostic.py",
    "experiments/geometria_proporcional/run_proportional_graph_conditional_risk_gate.py",
    "experiments/geometria_proporcional/run_proportional_graph_topology_localization_gate.py",
    "experiments/geometria_proporcional/run_proportional_graph_fresh_mixed_gate.py",
    "experiments/geometria_proporcional/run_proportional_graph_residual_gate.py",
    "experiments/geometria_proporcional/run_proportional_graph_solver_interface_diagnostic.py",
    "experiments/geometria_proporcional/run_proportional_graph_frozen_adapters.py",
)
FAMILIES = ("constant_signed_tail", "public_base_signed_tail", "topology_signed_tail")
CONTROL_FAMILY = "topology_permuted_signed_tail"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def load_config(path: Path) -> dict[str, Any]:
    cfg = json.loads(path.read_text())
    expected = {
        "schema_version", "source_conditional_gate", "source_manifest_sha256",
        "arms", "alphas", "quantile", "miscoverage", "l1_grid", "folds",
        "control_replicates", "selection_upper_percentile", "execution",
    }
    if set(cfg) != expected or cfg["schema_version"] != "proportional-graph-signed-tail-diagnostic-v1":
        raise ValueError("invalid signed-tail diagnostic schema")
    if cfg["arms"] != ["raw_generic", "raw_typed", "closure_generic", "closure_typed"]:
        raise ValueError("arms changed")
    if cfg["alphas"] != [0.0, 0.25, 0.5, 0.75, 1.0]:
        raise ValueError("alpha grid changed")
    if cfg["quantile"] != 0.90 or cfg["miscoverage"] != 0.10:
        raise ValueError("tail probability changed")
    if cfg["l1_grid"] != [0.0, 0.0001, 0.001, 0.01, 0.1, 1.0] or cfg["folds"] != 5:
        raise ValueError("quantile fit contract changed")
    if cfg["control_replicates"] != 16 or cfg["selection_upper_percentile"] != 95.0:
        raise ValueError("control or firewall contract changed")
    if cfg["execution"] != {"max_seconds": 720, "max_rss_gib": 4.0}:
        raise ValueError("execution contract changed")
    return cfg


def source_root(cfg: dict[str, Any]) -> Path:
    return ROOT / cfg["source_conditional_gate"]


def verify_source(cfg: dict[str, Any]) -> dict[str, Any]:
    root = source_root(cfg)
    if sha(root / "manifest.json") != cfg["source_manifest_sha256"]:
        raise AssertionError("conditional gate manifest hash mismatch")
    manifest = json.loads((root / "manifest.json").read_text())
    for relative, expected in manifest["deterministic_files"].items():
        if sha(root / relative) != expected:
            raise AssertionError(f"conditional source mismatch: {relative}")
    source_cfg = json.loads((root / "resolved_config.json").read_text())
    if source_cfg["arms"] != cfg["arms"] or source_cfg["alphas"] != cfg["alphas"]:
        raise AssertionError("source factorial changed")
    if source_cfg["control_replicates"] != cfg["control_replicates"]:
        raise AssertionError("source controls changed")
    return source_cfg


def require_clean(development: bool) -> None:
    if not development and subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout:
        raise RuntimeError("official diagnostic requires clean worktree")


def git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout.strip()


def read_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key] for key in saved.files}


def phase_arrays(root: Path, role: str) -> dict[str, np.ndarray]:
    if role == "risk_fit":
        return read_npz(root / role / "scale_fit.npz")
    if role == "risk_calibration":
        return read_npz(root / role / "calibration_scores.npz")
    if role == "policy_selection":
        arrays = read_npz(root / role / "selection_statistics.npz")
        data, _ = conditional.load_phase_data(root, role)
        arrays["delta"] = conditional.realized_deltas(data)
        return arrays
    return read_npz(root / role / "decisions_and_risk.npz")


def phase_features(
    root: Path, role: str, cfg: dict[str, Any], family: str, arm_index: int,
    replicate: int | None = None,
) -> np.ndarray:
    data, topology = conditional.load_phase_data(root, role)
    if family == "public_base_signed_tail":
        return conditional.feature_matrix(data, topology, arm_index, "public_base_scale")
    if family == "topology_signed_tail":
        return conditional.feature_matrix(data, topology, arm_index, "topology_scale")
    if family == CONTROL_FAMILY:
        return conditional.feature_matrix(data, topology, arm_index, "topology_permuted_scale", replicate)
    raise ValueError(f"family has no features: {family}")


def pinball_loss(y: np.ndarray, prediction: np.ndarray, quantile: float) -> float:
    error = np.asarray(y) - np.asarray(prediction)
    return float(np.mean(np.maximum(quantile * error, (quantile - 1.0) * error)))


def standardize(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(x, axis=0)
    scale = np.std(x, axis=0)
    scale = np.where(scale > 0.0, scale, 1.0)
    return (x - mean) / scale, mean, scale


def fit_quantile(x: np.ndarray, y: np.ndarray, alpha: float, quantile: float) -> dict[str, Any]:
    normalized, mean, scale = standardize(np.asarray(x, dtype=np.float64))
    estimator = QuantileRegressor(quantile=quantile, alpha=alpha, solver="highs", fit_intercept=True)
    estimator.fit(normalized, np.asarray(y, dtype=np.float64))
    if not np.all(np.isfinite(estimator.coef_)) or not np.isfinite(estimator.intercept_):
        raise RuntimeError("non-finite quantile model")
    return {
        "intercept": float(estimator.intercept_), "mean": mean.tolist(),
        "scale": scale.tolist(), "coefficients": estimator.coef_.tolist(),
    }


def predict_quantile(model: dict[str, Any], x: np.ndarray) -> np.ndarray:
    mean = np.asarray(model["mean"], dtype=np.float64)
    scale = np.asarray(model["scale"], dtype=np.float64)
    coefficient = np.asarray(model["coefficients"], dtype=np.float64)
    return float(model["intercept"]) + ((np.asarray(x) - mean) / scale) @ coefficient


def constant_quantile(y: np.ndarray, quantile: float) -> float:
    return float(np.quantile(np.asarray(y), quantile, method="higher"))


def fit_constant(y: np.ndarray, folds: np.ndarray, quantile: float) -> tuple[dict[str, Any], np.ndarray]:
    oof = np.full_like(y, np.nan)
    for fold in sorted(set(folds.tolist())):
        train, held = folds != fold, folds == fold
        for output in range(y.shape[1]):
            oof[held, output] = constant_quantile(y[train, output], quantile)
    return {
        "quantile": quantile,
        "intercepts": [constant_quantile(y[:, output], quantile) for output in range(y.shape[1])],
        "oof_pinball": pinball_loss(y, oof, quantile),
    }, oof


def fit_family(
    x: np.ndarray, y: np.ndarray, master_ids: np.ndarray, cfg: dict[str, Any],
) -> tuple[dict[str, Any], np.ndarray]:
    folds = historical.interface._fold_ids(master_ids, cfg["folds"])
    candidates, predictions = [], {}
    for alpha in cfg["l1_grid"]:
        oof = np.full_like(y, np.nan)
        for fold in range(cfg["folds"]):
            train, held = folds != fold, folds == fold
            for output in range(y.shape[1]):
                model = fit_quantile(x[train], y[train, output], alpha, cfg["quantile"])
                oof[held, output] = predict_quantile(model, x[held])
        loss = pinball_loss(y, oof, cfg["quantile"])
        candidates.append({"alpha": alpha, "oof_pinball": loss})
        predictions[float(alpha)] = oof
    best = min(row["oof_pinball"] for row in candidates)
    selected = max(row["alpha"] for row in candidates if row["oof_pinball"] <= best + 1e-12)
    models = [fit_quantile(x, y[:, output], selected, cfg["quantile"]) for output in range(y.shape[1])]
    report = {
        "selected_alpha": selected, "candidates": candidates,
        "fold_id": folds.tolist(), "models": models,
        "oof_pinball": pinball_loss(y, predictions[float(selected)], cfg["quantile"]),
    }
    return report, predictions[float(selected)]


def predict_report(report: dict[str, Any], x: np.ndarray) -> np.ndarray:
    if "intercepts" in report:
        return np.tile(np.asarray(report["intercepts"]), (len(x), 1))
    return np.column_stack([predict_quantile(model, x) for model in report["models"]])


def fit_models(cfg: dict[str, Any], root: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    arrays = phase_arrays(root, "risk_fit")
    iid = arrays["iid_index"].astype(np.int64)
    residual = arrays["delta"] - arrays["mu"]
    view_index = json.loads((root / "risk_fit/view_index.json").read_text())
    masters = np.asarray([view_index[index]["master_id"] for index in iid])
    folds = historical.interface._fold_ids(masters, cfg["folds"])
    models: dict[str, Any] = {"arms": {}}
    packed: dict[str, np.ndarray] = {"iid_index": iid, "residual": residual}
    for arm_index, arm in enumerate(cfg["arms"]):
        y = residual[arm_index, iid]
        constant, oof = fit_constant(y, folds, cfg["quantile"])
        record: dict[str, Any] = {"families": {"constant_signed_tail": constant}, CONTROL_FAMILY: []}
        packed[f"{arm}|constant_signed_tail|oof"] = oof
        for family in ("public_base_signed_tail", "topology_signed_tail"):
            x = phase_features(root, "risk_fit", cfg, family, arm_index)[iid]
            report, oof = fit_family(x, y, masters, cfg)
            record["families"][family] = report
            packed[f"{arm}|{family}|oof"] = oof
        for replicate in range(cfg["control_replicates"]):
            x = phase_features(root, "risk_fit", cfg, CONTROL_FAMILY, arm_index, replicate)[iid]
            report, oof = fit_family(x, y, masters, cfg)
            record[CONTROL_FAMILY].append({"replicate": replicate, "model": report})
            packed[f"{arm}|{CONTROL_FAMILY}={replicate}|oof"] = oof
        models["arms"][arm] = record
    return models, packed


def predictions(
    cfg: dict[str, Any], root: Path, role: str, models: dict[str, Any],
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    n_views = phase_arrays(root, role)["mu"].shape[1]
    result = {family: np.full((len(cfg["arms"]), n_views, 4), np.nan) for family in FAMILIES}
    control = np.full((len(cfg["arms"]), cfg["control_replicates"], n_views, 4), np.nan)
    for arm_index, arm in enumerate(cfg["arms"]):
        result["constant_signed_tail"][arm_index] = predict_report(
            models["arms"][arm]["families"]["constant_signed_tail"], np.empty((n_views, 0))
        )
        for family in ("public_base_signed_tail", "topology_signed_tail"):
            x = phase_features(root, role, cfg, family, arm_index)
            result[family][arm_index] = predict_report(models["arms"][arm]["families"][family], x)
        for replicate in range(cfg["control_replicates"]):
            x = phase_features(root, role, cfg, CONTROL_FAMILY, arm_index, replicate)
            control[arm_index, replicate] = predict_report(
                models["arms"][arm][CONTROL_FAMILY][replicate]["model"], x
            )
    return result, control


def calibrate(
    cfg: dict[str, Any], root: Path, models: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    arrays = phase_arrays(root, "risk_calibration")
    iid = arrays["iid_index"].astype(np.int64)
    residual = arrays["delta"] - arrays["mu"]
    predicted, controls = predictions(cfg, root, "risk_calibration", models)
    result: dict[str, Any] = {"arms": {}}
    packed: dict[str, np.ndarray] = {"residual": residual, "iid_index": iid, "control_prediction": controls}
    for family, value in predicted.items():
        packed[f"prediction|{family}"] = value
    for arm_index, arm in enumerate(cfg["arms"]):
        record: dict[str, Any] = {"families": {}, CONTROL_FAMILY: []}
        for family in FAMILIES:
            score = np.max(residual[arm_index, iid] - predicted[family][arm_index, iid], axis=1)
            q, rank = conditional.conformal_quantile(score, cfg["miscoverage"])
            record["families"][family] = {"q": q, "rank": rank, "n": len(score)}
            packed[f"{arm}|{family}|score"] = score
        for replicate in range(cfg["control_replicates"]):
            score = np.max(residual[arm_index, iid] - controls[arm_index, replicate, iid], axis=1)
            q, rank = conditional.conformal_quantile(score, cfg["miscoverage"])
            record[CONTROL_FAMILY].append({"replicate": replicate, "q": q, "rank": rank, "n": len(score)})
            packed[f"{arm}|{CONTROL_FAMILY}={replicate}|score"] = score
        result["arms"][arm] = record
    return result, packed


def actions_for_role(
    cfg: dict[str, Any], root: Path, role: str, models: dict[str, Any], calibration: dict[str, Any],
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, np.ndarray], np.ndarray]:
    arrays = phase_arrays(root, role)
    mu = arrays["mu"]
    predicted, control_prediction = predictions(cfg, root, role, models)
    actions = {family: np.zeros(mu.shape[:2], dtype=np.int64) for family in FAMILIES}
    upper = {family: np.full_like(mu, np.nan) for family in FAMILIES}
    controls = np.zeros((len(cfg["arms"]), cfg["control_replicates"], mu.shape[1]), dtype=np.int64)
    control_upper = np.full_like(control_prediction, np.nan)
    for arm_index, arm in enumerate(cfg["arms"]):
        for family in FAMILIES:
            q = calibration["arms"][arm]["families"][family]["q"]
            upper[family][arm_index] = mu[arm_index] + predicted[family][arm_index] + q
            best = np.argmin(upper[family][arm_index], axis=1)
            actions[family][arm_index] = np.where(
                upper[family][arm_index, np.arange(mu.shape[1]), best] < 0.0, best + 1, 0
            )
        for replicate in range(cfg["control_replicates"]):
            q = calibration["arms"][arm][CONTROL_FAMILY][replicate]["q"]
            control_upper[arm_index, replicate] = mu[arm_index] + control_prediction[arm_index, replicate] + q
            best = np.argmin(control_upper[arm_index, replicate], axis=1)
            controls[arm_index, replicate] = np.where(
                control_upper[arm_index, replicate, np.arange(mu.shape[1]), best] < 0.0, best + 1, 0
            )
    return actions, controls, upper, control_upper


def selection_firewall(
    cfg: dict[str, Any], root: Path, models: dict[str, Any], calibration: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    actions, controls, upper, control_upper = actions_for_role(
        cfg, root, "policy_selection", models, calibration
    )
    data, _ = conditional.load_phase_data(root, "policy_selection")
    source = phase_arrays(root, "policy_selection")
    bootstrap = source["bootstrap_indices"].astype(np.int64)
    result: dict[str, Any] = {"arms": {}}
    packed: dict[str, np.ndarray] = {"bootstrap_indices": bootstrap, "control_upper": control_upper}
    for family, value in upper.items():
        packed[f"upper|{family}"] = value
    for arm_index, arm in enumerate(cfg["arms"]):
        quotient = data["quotient_rmse"][arm_index]
        record: dict[str, Any] = {"families": {}, CONTROL_FAMILY: []}
        for family in FAMILIES:
            delta = conditional.selected_delta(quotient, actions[family][arm_index])
            record["families"][family] = conditional.firewall_one(delta, bootstrap, cfg["selection_upper_percentile"])
            packed[f"{arm}|{family}|action"] = actions[family][arm_index]
        for replicate in range(cfg["control_replicates"]):
            delta = conditional.selected_delta(quotient, controls[arm_index, replicate])
            record[CONTROL_FAMILY].append({
                "replicate": replicate,
                **conditional.firewall_one(delta, bootstrap, cfg["selection_upper_percentile"]),
            })
            packed[f"{arm}|{CONTROL_FAMILY}={replicate}|action"] = controls[arm_index, replicate]
        result["arms"][arm] = record
    return result, packed


def apply_firewall(
    cfg: dict[str, Any], actions: dict[str, np.ndarray], controls: np.ndarray,
    firewall: dict[str, Any],
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    deployed = {family: value.copy() for family, value in actions.items()}
    deployed_controls = controls.copy()
    for arm_index, arm in enumerate(cfg["arms"]):
        for family in FAMILIES:
            if not firewall["arms"][arm]["families"][family]["deploy"]:
                deployed[family][arm_index] = 0
        for replicate in range(cfg["control_replicates"]):
            if not firewall["arms"][arm][CONTROL_FAMILY][replicate]["deploy"]:
                deployed_controls[arm_index, replicate] = 0
    return deployed, deployed_controls


def interval(values: np.ndarray, bootstrap: np.ndarray) -> dict[str, Any]:
    draws = values[bootstrap].mean(axis=1)
    return {"mean": float(values.mean()), "ci95": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]}


def describe_residual(values: np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    flat = array.reshape(-1)
    positive, negative = flat[flat > 0.0], flat[flat < 0.0]
    return {
        "n": len(flat), "positive_fraction": float(np.mean(flat > 0.0)),
        "zero_fraction": float(np.mean(flat == 0.0)),
        "mean_positive": float(positive.mean()) if len(positive) else None,
        "mean_negative": float(negative.mean()) if len(negative) else None,
        "quantiles": {
            str(q): float(np.quantile(flat, q)) for q in (0.1, 0.5, 0.9, 0.95, 0.99)
        },
        "by_alpha": [
            {
                "positive_fraction": float(np.mean(array[:, output] > 0.0)),
                "q90": float(np.quantile(array[:, output], 0.9)),
                "q95": float(np.quantile(array[:, output], 0.95)),
            }
            for output in range(array.shape[1])
        ],
    }


def summarize(
    cfg: dict[str, Any], root: Path, models: dict[str, Any], calibration: dict[str, Any],
    firewall: dict[str, Any], actions: dict[str, np.ndarray], controls: np.ndarray,
    upper: dict[str, np.ndarray], control_upper: np.ndarray,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    data, _ = conditional.load_phase_data(root, "adjudication")
    arrays = phase_arrays(root, "adjudication")
    delta, mu = arrays["delta"], arrays["mu"]
    bootstrap = read_npz(root / "bootstrap_indices.npz")["indices"].astype(np.int64)
    deployed, deployed_controls = apply_firewall(cfg, actions, controls, firewall)
    iid, grouped = np.arange(0, delta.shape[1], 2), np.arange(1, delta.shape[1], 2)
    source_absolute = arrays["deployed|topology_scale"].astype(np.int64)
    fit_arrays = phase_arrays(root, "risk_fit")
    fit_residual = fit_arrays["delta"] - fit_arrays["mu"]
    report: dict[str, Any] = {"status": "OPENED_POSTHOC_DESIGN_DIAGNOSTIC", "arms": {}}
    packed: dict[str, np.ndarray] = {
        "control_upper": control_upper, "calibrated_control": controls,
        "deployed_control": deployed_controls,
    }
    for family, value in upper.items():
        packed[f"upper|{family}"] = value
        packed[f"calibrated|{family}"] = actions[family]
        packed[f"deployed|{family}"] = deployed[family]
    for arm_index, arm in enumerate(cfg["arms"]):
        quotient = data["quotient_rmse"][arm_index]
        values: dict[str, np.ndarray] = {"identity": quotient[:, 0]}
        for family in FAMILIES:
            values[f"calibrated_{family}"] = mixed.selected_values(quotient, actions[family][arm_index])
            values[f"deployed_{family}"] = mixed.selected_values(quotient, deployed[family][arm_index])
        for stage, control_action in (("calibrated", controls), ("deployed", deployed_controls)):
            values[f"{stage}_{CONTROL_FAMILY}"] = np.mean([
                mixed.selected_values(quotient, control_action[arm_index, replicate])
                for replicate in range(cfg["control_replicates"])
            ], axis=0)
        values["absolute_topology_scale"] = mixed.selected_values(quotient, source_absolute[arm_index])
        arm_report: dict[str, Any] = {"cells": {}, "coverage": {}, "action_fractions": {}, "acted_harm": {}}
        for cell, indices in (("iid", iid), ("grouped", grouped), ("balanced", None)):
            per_policy = {
                name: (raw[:, indices].mean(axis=0) if indices is not None else (0.5 * (raw[:, iid] + raw[:, grouped])).mean(axis=0))
                for name, raw in values.items()
            }
            arm_report["cells"][cell] = {}
            for name, value in per_policy.items():
                row = {"minus_identity": interval(value - per_policy["identity"], bootstrap)}
                if name in ("calibrated_topology_signed_tail", "deployed_topology_signed_tail"):
                    stage = name.split("_topology_signed_tail")[0]
                    for baseline in ("constant_signed_tail", "public_base_signed_tail", CONTROL_FAMILY):
                        other = f"{stage}_{baseline}"
                        row[f"minus_{other}"] = interval(value - per_policy[other], bootstrap)
                    row["minus_absolute_topology_scale"] = interval(
                        value - per_policy["absolute_topology_scale"], bootstrap
                    )
                arm_report["cells"][cell][name] = row
        residual = delta[arm_index] - mu[arm_index]
        for family in FAMILIES:
            covered = np.all(delta[arm_index] <= upper[family][arm_index], axis=1)
            arm_report["coverage"][family] = {"iid": float(np.mean(covered[iid])), "grouped": float(np.mean(covered[grouped]))}
            arm_report["acted_harm"][family] = {}
            arm_report["action_fractions"][family] = {}
            for stage, action in (("calibrated", actions[family][arm_index]), ("deployed", deployed[family][arm_index])):
                realized = conditional.selected_delta(quotient, action)
                arm_report["action_fractions"][family][stage] = {
                    "iid": float(np.mean(action[iid] > 0)), "grouped": float(np.mean(action[grouped] > 0))
                }
                arm_report["acted_harm"][family][stage] = {}
                for cell, indices in (("iid", iid), ("grouped", grouped)):
                    acted = action[indices] > 0
                    selected = realized[indices][acted]
                    arm_report["acted_harm"][family][stage][cell] = {
                        "n": int(acted.sum()), "mean_delta": float(selected.mean()) if len(selected) else None,
                        "positive_fraction": float(np.mean(selected > 0)) if len(selected) else None,
                    }
        arm_report["residual_asymmetry"] = {
            "risk_fit_iid": describe_residual(fit_residual[arm_index, ::2]),
            "adjudication_iid": describe_residual(residual[iid]),
            "adjudication_grouped": describe_residual(residual[grouped]),
        }
        report["arms"][arm] = arm_report
    return report, packed


def write_replay(output: Path, cfg: dict[str, Any], head: str) -> None:
    checks = "\n".join(
        f"printf '%s  %s\\n' '{sha(ROOT / relative)}' \"$repo/{relative}\" | sha256sum -c -"
        for relative in SOURCE_FILES
    )
    checks += f"\nprintf '%s  %s\\n' '{sha(source_root(cfg) / 'manifest.json')}' \"$repo/{cfg['source_conditional_gate']}/manifest.json\" | sha256sum -c -"
    replay = f'''#!/bin/sh
set -eu
repo=$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd)
[ "$(git -C "$repo" rev-parse HEAD)" = "{head}" ] || {{ echo "wrong git HEAD" >&2; exit 2; }}
[ -z "$(git -C "$repo" status --porcelain)" ] || {{ echo "dirty worktree" >&2; exit 2; }}
{checks}
: "${{OUTPUT_DIR:?set OUTPUT_DIR to a new path}}"
exec env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 "$repo/venv/bin/python" "$repo/experiments/geometria_proporcional/run_proportional_graph_signed_tail_diagnostic.py" --config "$repo/experiments/geometria_proporcional/configs/proportional_graph_signed_tail_diagnostic_v1.json" --output "$OUTPUT_DIR"
'''
    (output / "replay.sh").write_text(replay)
    (output / "replay.sh").chmod(0o755)


def run(cfg: dict[str, Any], output: Path, development: bool) -> None:
    require_clean(development)
    if output.exists():
        raise FileExistsError(output)
    if not development and (ROOT / "data/geometria_proporcional").resolve() not in output.resolve().parents:
        raise ValueError("official output must be below data/geometria_proporcional")
    started = time.monotonic()
    root = source_root(cfg)
    verify_source(cfg)
    output.mkdir(parents=True)
    write_json(output / "environment.json", {
        "python": platform.python_version(), "numpy": importlib.metadata.version("numpy"),
        "scipy": importlib.metadata.version("scipy"),
        "scikit_learn": importlib.metadata.version("scikit-learn"),
        "quantile_solver": "highs", "threads": 1, "cuda_visible_devices": "",
    })
    models, fit = fit_models(cfg, root)
    write_json(output / "quantile_models.json", models)
    frozen.save_npz(output / "fit_diagnostics.npz", fit)
    calibration, scores = calibrate(cfg, root, models)
    write_json(output / "calibration.json", calibration)
    frozen.save_npz(output / "calibration_scores.npz", scores)
    firewall, selection = selection_firewall(cfg, root, models, calibration)
    write_json(output / "selection_firewall.json", firewall)
    frozen.save_npz(output / "selection_diagnostics.npz", selection)
    actions, controls, upper, control_upper = actions_for_role(cfg, root, "adjudication", models, calibration)
    analysis, adjudication = summarize(
        cfg, root, models, calibration, firewall, actions, controls, upper, control_upper
    )
    write_json(output / "analysis.json", analysis)
    frozen.save_npz(output / "adjudication_diagnostics.npz", adjudication)
    write_json(output / "resolved_config.json", cfg)
    write_replay(output, cfg, git_head())
    runtime = {
        "elapsed_seconds": time.monotonic() - started,
        "max_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
    }
    if runtime["elapsed_seconds"] > cfg["execution"]["max_seconds"] or runtime["max_rss_gib"] > cfg["execution"]["max_rss_gib"]:
        raise RuntimeError("diagnostic resource budget exceeded")
    write_json(output / "runtime_observation.json", runtime)
    deterministic = sorted(path for path in output.rglob("*") if path.is_file() and path.name not in {"manifest.json", "runtime_observation.json"})
    write_json(output / "manifest.json", {
        "schema_version": cfg["schema_version"], "git_head": git_head(),
        "source_manifest_sha256": cfg["source_manifest_sha256"],
        "source_hashes": {relative: sha(ROOT / relative) for relative in SOURCE_FILES},
        "deterministic_files": {str(path.relative_to(output)): sha(path) for path in deterministic},
        "runtime_exclusions": ["runtime_observation.json"],
    })
    print(json.dumps(runtime, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--development", action="store_true")
    args = parser.parse_args()
    run(load_config(args.config.resolve()), args.output.resolve(), args.development)


if __name__ == "__main__":
    main()
