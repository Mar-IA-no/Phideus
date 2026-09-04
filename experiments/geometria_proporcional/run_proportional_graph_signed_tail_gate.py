#!/usr/bin/env python3
"""Fit and adjudicate paired signed-tail and absolute-scale risk gates on fresh CPU data."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
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
import run_proportional_graph_signed_tail_diagnostic as signed  # noqa: E402

DEFAULT_CONFIG = ROOT / "experiments/geometria_proporcional/configs/proportional_graph_signed_tail_gate_v1.json"
SOURCE_FILES = (
    "experiments/geometria_proporcional/PLAN_PROPORTIONAL_SIGNED_TAIL_GATE_CPU.md",
    "experiments/geometria_proporcional/configs/proportional_graph_signed_tail_gate_v1.json",
    "experiments/geometria_proporcional/run_proportional_graph_signed_tail_gate.py",
    "experiments/geometria_proporcional/run_proportional_graph_signed_tail_diagnostic.py",
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
INTERFACES = ("signed_tail", "absolute_scale")
FAMILY_BASES = ("constant", "public_base", "topology")
CONTROL_BASE = "topology_permuted"
SIGNED_FAMILY = {
    "constant": "constant_signed_tail",
    "public_base": "public_base_signed_tail",
    "topology": "topology_signed_tail",
}
ABSOLUTE_FAMILY = {
    "constant": "constant_scale",
    "public_base": "public_base_scale",
    "topology": "topology_scale",
}


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
        "residual_epsilon", "scale_clip", "quantile", "l1_grid", "folds",
        "miscoverage", "control_replicates", "topology_control_seed",
        "selection_bootstrap_replicates", "selection_bootstrap_seed",
        "selection_upper_percentile", "evaluation_bootstrap_replicates",
        "evaluation_bootstrap_seed", "execution",
    }
    if set(cfg) != expected or cfg["schema_version"] != "proportional-graph-signed-tail-gate-v1":
        raise ValueError("invalid signed-tail gate schema")
    if cfg["arms"] != ["raw_generic", "raw_typed", "closure_generic", "closure_typed"]:
        raise ValueError("arms changed")
    if cfg["seeds"] != [104729, 130363] or cfg["alphas"] != [0.0, 0.25, 0.5, 0.75, 1.0]:
        raise ValueError("neural factorial changed")
    if cfg["realizations"] != {
        "risk_fit_seed": 2026091109, "risk_calibration_seed": 2026091117,
        "policy_selection_seed": 2026091129, "adjudication_seed": 2026091137,
        "min_eligible_masters": 220,
    }:
        raise ValueError("fresh realization contract changed")
    if cfg["ridge_lambdas"] != [0.0, 0.01, 0.1, 1.0, 10.0, 100.0] or cfg["ridge_folds"] != 5:
        raise ValueError("absolute-scale fit contract changed")
    if cfg["residual_epsilon"] != 1e-6 or cfg["scale_clip"] != [1e-6, 1.0]:
        raise ValueError("absolute-scale transform changed")
    if cfg["quantile"] != 0.90 or cfg["l1_grid"] != [0.0, 0.0001, 0.001, 0.01, 0.1, 1.0]:
        raise ValueError("signed-tail fit contract changed")
    if cfg["folds"] != 5 or cfg["miscoverage"] != 0.10 or cfg["control_replicates"] != 16:
        raise ValueError("calibration contract changed")
    if cfg["topology_control_seed"] != 2026091141:
        raise ValueError("topology control seed changed")
    if (
        cfg["selection_bootstrap_replicates"] != 2000
        or cfg["selection_bootstrap_seed"] != 2026091147
        or cfg["selection_upper_percentile"] != 95.0
        or cfg["evaluation_bootstrap_replicates"] != 2000
        or cfg["evaluation_bootstrap_seed"] != 2026091153
    ):
        raise ValueError("bootstrap contract changed")
    if cfg["execution"] != {
        "torch_threads": 1, "max_seconds_per_phase": 900,
        "max_rss_gib": 4.0, "sample_every_views": 64,
    }:
        raise ValueError("CPU execution contract changed")
    return cfg


def verify_source(cfg: dict[str, Any]) -> dict[str, Any]:
    return conditional.verify_source(cfg)


def require_clean(development: bool) -> None:
    if not development and subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout:
        raise RuntimeError("official phase requires clean worktree")


def git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout.strip()


def feature_matrix(
    data: dict[str, np.ndarray], topology: dict[str, np.ndarray], arm_index: int,
    family: str, replicate: int | None = None,
) -> np.ndarray:
    mapped = {
        "public_base": "public_base_scale",
        "topology": "topology_scale",
        CONTROL_BASE: conditional.CONTROL_FAMILY,
    }[family]
    return conditional.feature_matrix(data, topology, arm_index, mapped, replicate)


def fit_signed_models(
    cfg: dict[str, Any], views: list[Any], data: dict[str, np.ndarray],
    topology: dict[str, np.ndarray], mean_models: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    iid = np.arange(0, len(views), 2)
    masters = np.asarray([views[index].private.master_id for index in iid])
    mu = conditional.mean_predictions(cfg, data, topology, mean_models)
    delta = conditional.realized_deltas(data)
    residual = delta - mu
    folds = signed.historical.interface._fold_ids(masters, cfg["folds"])
    models: dict[str, Any] = {"arms": {}}
    packed: dict[str, np.ndarray] = {
        "mu": mu, "delta": delta, "residual": residual, "iid_index": iid,
    }
    for arm_index, arm in enumerate(cfg["arms"]):
        y = residual[arm_index, iid]
        constant, oof = signed.fit_constant(y, folds, cfg["quantile"])
        record: dict[str, Any] = {
            "families": {SIGNED_FAMILY["constant"]: constant},
            signed.CONTROL_FAMILY: [],
        }
        packed[f"{arm}|{SIGNED_FAMILY['constant']}|oof"] = oof
        for family in ("public_base", "topology"):
            x = feature_matrix(data, topology, arm_index, family)[iid]
            report, oof = signed.fit_family(x, y, masters, cfg)
            record["families"][SIGNED_FAMILY[family]] = report
            packed[f"{arm}|{SIGNED_FAMILY[family]}|oof"] = oof
        for replicate in range(cfg["control_replicates"]):
            x = feature_matrix(data, topology, arm_index, CONTROL_BASE, replicate)[iid]
            report, oof = signed.fit_family(x, y, masters, cfg)
            record[signed.CONTROL_FAMILY].append({"replicate": replicate, "model": report})
            packed[f"{arm}|{signed.CONTROL_FAMILY}={replicate}|oof"] = oof
        models["arms"][arm] = record
    return models, packed


def signed_predictions(
    cfg: dict[str, Any], data: dict[str, np.ndarray], topology: dict[str, np.ndarray],
    models: dict[str, Any],
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    n_arms = len(cfg["arms"])
    n_views = data["quotient_rmse"].shape[-1]
    n_outputs = len(cfg["alphas"]) - 1
    predictions = {
        family: np.full((n_arms, n_views, n_outputs), np.nan)
        for family in FAMILY_BASES
    }
    controls = np.full((n_arms, cfg["control_replicates"], n_views, n_outputs), np.nan)
    for arm_index, arm in enumerate(cfg["arms"]):
        predictions["constant"][arm_index] = signed.predict_report(
            models["arms"][arm]["families"][SIGNED_FAMILY["constant"]], np.empty((n_views, 0))
        )
        for family in ("public_base", "topology"):
            x = feature_matrix(data, topology, arm_index, family)
            predictions[family][arm_index] = signed.predict_report(
                models["arms"][arm]["families"][SIGNED_FAMILY[family]], x
            )
        for replicate in range(cfg["control_replicates"]):
            x = feature_matrix(data, topology, arm_index, CONTROL_BASE, replicate)
            controls[arm_index, replicate] = signed.predict_report(
                models["arms"][arm][signed.CONTROL_FAMILY][replicate]["model"], x
            )
    if any(not np.all(np.isfinite(value)) for value in predictions.values()) or not np.all(np.isfinite(controls)):
        raise RuntimeError("non-finite signed-tail prediction")
    return predictions, controls


def calibrate_signed(
    cfg: dict[str, Any], data: dict[str, np.ndarray], topology: dict[str, np.ndarray],
    mean_models: dict[str, Any], models: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    iid = np.arange(0, data["quotient_rmse"].shape[-1], 2)
    delta = conditional.realized_deltas(data)
    mu = conditional.mean_predictions(cfg, data, topology, mean_models)
    residual = delta - mu
    predictions, controls = signed_predictions(cfg, data, topology, models)
    result: dict[str, Any] = {"miscoverage": cfg["miscoverage"], "arms": {}}
    packed: dict[str, np.ndarray] = {
        "delta": delta, "mu": mu, "residual": residual,
        "iid_index": iid, "control_prediction": controls,
    }
    for family, value in predictions.items():
        packed[f"prediction|{family}"] = value
    for arm_index, arm in enumerate(cfg["arms"]):
        record: dict[str, Any] = {"families": {}, CONTROL_BASE: []}
        for family in FAMILY_BASES:
            score = np.max(residual[arm_index, iid] - predictions[family][arm_index, iid], axis=1)
            q, rank = conditional.conformal_quantile(score, cfg["miscoverage"])
            record["families"][family] = {"q": q, "rank": rank, "n": len(score)}
            packed[f"{arm}|{family}|score"] = score
        for replicate in range(cfg["control_replicates"]):
            score = np.max(residual[arm_index, iid] - controls[arm_index, replicate, iid], axis=1)
            q, rank = conditional.conformal_quantile(score, cfg["miscoverage"])
            record[CONTROL_BASE].append({"replicate": replicate, "q": q, "rank": rank, "n": len(score)})
            packed[f"{arm}|{CONTROL_BASE}={replicate}|score"] = score
        result["arms"][arm] = record
    return result, packed


def signed_action(mu: np.ndarray, prediction: np.ndarray, q: float) -> tuple[np.ndarray, np.ndarray]:
    upper = mu + prediction + q
    best = np.argmin(upper, axis=1)
    action = np.where(upper[np.arange(len(upper)), best] < 0.0, best + 1, 0).astype(np.int64)
    return action, upper


def absolute_action(mu: np.ndarray, scale: np.ndarray, q: float) -> tuple[np.ndarray, np.ndarray]:
    return conditional.bounded_action(mu, scale, q)


def signed_actions(
    cfg: dict[str, Any], data: dict[str, np.ndarray], topology: dict[str, np.ndarray],
    mean_models: dict[str, Any], models: dict[str, Any], calibration: dict[str, Any],
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, np.ndarray], np.ndarray, dict[str, np.ndarray], np.ndarray]:
    mu = conditional.mean_predictions(cfg, data, topology, mean_models)
    predictions, control_predictions = signed_predictions(cfg, data, topology, models)
    actions = {family: np.zeros(mu.shape[:2], dtype=np.int64) for family in FAMILY_BASES}
    uppers = {family: np.full_like(mu, np.nan) for family in FAMILY_BASES}
    controls = np.zeros((len(cfg["arms"]), cfg["control_replicates"], mu.shape[1]), dtype=np.int64)
    control_uppers = np.full_like(control_predictions, np.nan)
    for arm_index, arm in enumerate(cfg["arms"]):
        for family in FAMILY_BASES:
            actions[family][arm_index], uppers[family][arm_index] = signed_action(
                mu[arm_index], predictions[family][arm_index],
                calibration["arms"][arm]["families"][family]["q"],
            )
        for replicate in range(cfg["control_replicates"]):
            controls[arm_index, replicate], control_uppers[arm_index, replicate] = signed_action(
                mu[arm_index], control_predictions[arm_index, replicate],
                calibration["arms"][arm][CONTROL_BASE][replicate]["q"],
            )
    return actions, controls, uppers, control_uppers, predictions, control_predictions


def paired_actions(
    cfg: dict[str, Any], data: dict[str, np.ndarray], topology: dict[str, np.ndarray],
    mean_models: dict[str, Any], risk_models: dict[str, Any], calibration: dict[str, Any],
) -> tuple[
    dict[str, dict[str, np.ndarray]], dict[str, np.ndarray],
    dict[str, dict[str, np.ndarray]], dict[str, np.ndarray],
    dict[str, dict[str, np.ndarray]], dict[str, np.ndarray],
]:
    signed_result = signed_actions(
        cfg, data, topology, mean_models, risk_models["signed_tail"],
        calibration["interfaces"]["signed_tail"],
    )
    signed_a, signed_c, signed_u, signed_cu, signed_r, signed_cr = signed_result
    absolute_raw = conditional.calibrated_actions(
        cfg, data, topology, mean_models, risk_models["absolute_scale"],
        calibration["interfaces"]["absolute_scale"],
    )
    absolute_a_raw, absolute_c, absolute_u_raw, absolute_cu = absolute_raw
    absolute_scales_raw, absolute_cr = conditional.all_scales(
        cfg, data, topology, risk_models["absolute_scale"]
    )
    absolute_a = {base: absolute_a_raw[ABSOLUTE_FAMILY[base]] for base in FAMILY_BASES}
    absolute_u = {base: absolute_u_raw[ABSOLUTE_FAMILY[base]] for base in FAMILY_BASES}
    absolute_r = {base: absolute_scales_raw[ABSOLUTE_FAMILY[base]] for base in FAMILY_BASES}
    return (
        {"signed_tail": signed_a, "absolute_scale": absolute_a},
        {"signed_tail": signed_c, "absolute_scale": absolute_c},
        {"signed_tail": signed_u, "absolute_scale": absolute_u},
        {"signed_tail": signed_cu, "absolute_scale": absolute_cu},
        {"signed_tail": signed_r, "absolute_scale": absolute_r},
        {"signed_tail": signed_cr, "absolute_scale": absolute_cr},
    )


def select_policies(
    cfg: dict[str, Any], data: dict[str, np.ndarray], topology: dict[str, np.ndarray],
    mean_models: dict[str, Any], risk_models: dict[str, Any], calibration: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    actions, controls, uppers, control_uppers, risks, control_risks = paired_actions(
        cfg, data, topology, mean_models, risk_models, calibration
    )
    n_masters = data["quotient_rmse"].shape[-1] // 2
    bootstrap = np.random.default_rng(cfg["selection_bootstrap_seed"]).integers(
        0, n_masters, size=(cfg["selection_bootstrap_replicates"], n_masters)
    )
    firewall: dict[str, Any] = {"interfaces": {}}
    packed: dict[str, np.ndarray] = {"bootstrap_indices": bootstrap}
    for interface in INTERFACES:
        interface_record: dict[str, Any] = {"arms": {}}
        packed[f"{interface}|control_upper"] = control_uppers[interface]
        packed[f"{interface}|control_risk"] = control_risks[interface]
        for family in FAMILY_BASES:
            packed[f"{interface}|{family}|upper"] = uppers[interface][family]
            packed[f"{interface}|{family}|risk"] = risks[interface][family]
        for arm_index, arm in enumerate(cfg["arms"]):
            quotient = data["quotient_rmse"][arm_index]
            record: dict[str, Any] = {"families": {}, CONTROL_BASE: []}
            for family in FAMILY_BASES:
                delta = conditional.selected_delta(quotient, actions[interface][family][arm_index])
                record["families"][family] = conditional.firewall_one(
                    delta, bootstrap, cfg["selection_upper_percentile"]
                )
                packed[f"{interface}|{arm}|{family}|calibrated_action"] = actions[interface][family][arm_index]
            for replicate in range(cfg["control_replicates"]):
                delta = conditional.selected_delta(quotient, controls[interface][arm_index, replicate])
                record[CONTROL_BASE].append({
                    "replicate": replicate,
                    **conditional.firewall_one(delta, bootstrap, cfg["selection_upper_percentile"]),
                })
                packed[f"{interface}|{arm}|{CONTROL_BASE}={replicate}|calibrated_action"] = controls[interface][arm_index, replicate]
            interface_record["arms"][arm] = record
        firewall["interfaces"][interface] = interface_record
    return firewall, packed


def apply_firewall(
    cfg: dict[str, Any], actions: dict[str, dict[str, np.ndarray]],
    controls: dict[str, np.ndarray], firewall: dict[str, Any],
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, np.ndarray]]:
    deployed = {
        interface: {family: value.copy() for family, value in families.items()}
        for interface, families in actions.items()
    }
    deployed_controls = {interface: value.copy() for interface, value in controls.items()}
    for interface in INTERFACES:
        for arm_index, arm in enumerate(cfg["arms"]):
            record = firewall["interfaces"][interface]["arms"][arm]
            for family in FAMILY_BASES:
                if not record["families"][family]["deploy"]:
                    deployed[interface][family][arm_index] = 0
            for replicate in range(cfg["control_replicates"]):
                if not record[CONTROL_BASE][replicate]["deploy"]:
                    deployed_controls[interface][arm_index, replicate] = 0
    return deployed, deployed_controls


def interval(values: np.ndarray, bootstrap: np.ndarray) -> dict[str, Any]:
    draws = values[bootstrap].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci95": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
    }


def fit_quality(cfg: dict[str, Any], risk_models: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {"arms": {}}
    signed_models = risk_models["signed_tail"]
    absolute_models = risk_models["absolute_scale"]
    for arm in cfg["arms"]:
        signed_arm = signed_models["arms"][arm]
        absolute_arm = absolute_models["arms"][arm]
        result["arms"][arm] = {
            "signed_tail_oof_pinball": {
                family: signed_arm["families"][SIGNED_FAMILY[family]]["oof_pinball"]
                for family in FAMILY_BASES
            } | {
                CONTROL_BASE: float(np.mean([
                    row["model"]["oof_pinball"] for row in signed_arm[signed.CONTROL_FAMILY]
                ]))
            },
            "absolute_scale_oof_mse": {
                "constant": None,
                "public_base": absolute_arm["families"][ABSOLUTE_FAMILY["public_base"]]["oof_mse"],
                "topology": absolute_arm["families"][ABSOLUTE_FAMILY["topology"]]["oof_mse"],
                CONTROL_BASE: float(np.mean([
                    row["model"]["oof_mse"] for row in absolute_arm[conditional.CONTROL_FAMILY]
                ])),
            },
        }
    return result


def policy_values(
    cfg: dict[str, Any], quotient: np.ndarray,
    actions: dict[str, dict[str, np.ndarray]], controls: dict[str, np.ndarray],
    deployed: dict[str, dict[str, np.ndarray]], deployed_controls: dict[str, np.ndarray],
    arm_index: int,
) -> dict[str, np.ndarray]:
    values: dict[str, np.ndarray] = {
        "identity": quotient[:, 0],
        "oracle_per_view": mixed.selected_values(quotient, np.argmin(quotient.mean(axis=0), axis=0)),
    }
    for interface in INTERFACES:
        for stage, stage_actions in (("calibrated", actions), ("deployed", deployed)):
            for family in FAMILY_BASES:
                key = f"{stage}|{interface}|{family}"
                values[key] = mixed.selected_values(quotient, stage_actions[interface][family][arm_index])
        for stage, stage_controls in (("calibrated", controls), ("deployed", deployed_controls)):
            key = f"{stage}|{interface}|{CONTROL_BASE}"
            values[key] = np.mean([
                mixed.selected_values(quotient, stage_controls[interface][arm_index, replicate])
                for replicate in range(cfg["control_replicates"])
            ], axis=0)
    return values


def summarize(
    cfg: dict[str, Any], data: dict[str, np.ndarray], delta: np.ndarray, mu: np.ndarray,
    risk_models: dict[str, Any], calibration: dict[str, Any], firewall: dict[str, Any],
    actions: dict[str, dict[str, np.ndarray]], controls: dict[str, np.ndarray],
    deployed: dict[str, dict[str, np.ndarray]], deployed_controls: dict[str, np.ndarray],
    uppers: dict[str, dict[str, np.ndarray]], control_uppers: dict[str, np.ndarray],
    bootstrap: np.ndarray,
) -> tuple[dict[str, Any], dict[str, Any]]:
    iid = np.arange(0, data["quotient_rmse"].shape[-1], 2)
    grouped = np.arange(1, data["quotient_rmse"].shape[-1], 2)
    effects: dict[str, Any] = {
        "status": "FRESH_PROSPECTIVE_ADJUDICATION",
        "interval_scope": "pointwise paired master bootstrap",
        "risk_fit": fit_quality(cfg, risk_models),
        "selection_firewall": firewall,
        "arms": {},
    }
    diagnostics: dict[str, Any] = {
        "coverage_jurisdiction": "marginal IID per-view simultaneous across nonidentity alphas",
        "arms": {},
    }
    for arm_index, arm in enumerate(cfg["arms"]):
        quotient = data["quotient_rmse"][arm_index]
        values = policy_values(cfg, quotient, actions, controls, deployed, deployed_controls, arm_index)
        arm_effect: dict[str, Any] = {"cells": {}, "action_fractions": {}, "action_overlap": {}}
        cell_values: dict[str, dict[str, np.ndarray]] = {}
        for cell, indices in (("iid", iid), ("grouped", grouped), ("balanced", None)):
            per_policy = {
                name: (
                    raw[:, indices].mean(axis=0)
                    if indices is not None
                    else (0.5 * (raw[:, iid] + raw[:, grouped])).mean(axis=0)
                )
                for name, raw in values.items()
            }
            cell_values[cell] = per_policy
            arm_effect["cells"][cell] = {}
            for name, value in per_policy.items():
                row: dict[str, Any] = {
                    "n_masters": len(value),
                    "mean_rmse": interval(value, bootstrap),
                    "minus_identity": interval(value - per_policy["identity"], bootstrap),
                }
                parts = name.split("|")
                if len(parts) == 3 and parts[2] == "topology":
                    stage, interface, _ = parts
                    for baseline in ("constant", "public_base", CONTROL_BASE):
                        other = f"{stage}|{interface}|{baseline}"
                        row[f"minus_{interface}_{baseline}"] = interval(value - per_policy[other], bootstrap)
                    if interface == "signed_tail":
                        other = f"{stage}|absolute_scale|topology"
                        row["minus_absolute_scale_topology"] = interval(value - per_policy[other], bootstrap)
                arm_effect["cells"][cell][name] = row
        for cell, indices in (("iid", iid), ("grouped", grouped)):
            arm_effect["action_fractions"][cell] = {}
            for interface in INTERFACES:
                for stage, stage_actions in (("calibrated", actions), ("deployed", deployed)):
                    for family in FAMILY_BASES:
                        action = stage_actions[interface][family][arm_index]
                        arm_effect["action_fractions"][cell][f"{stage}|{interface}|{family}"] = {
                            str(alpha): float(np.mean(action[indices] == alpha_index))
                            for alpha_index, alpha in enumerate(cfg["alphas"])
                        }
                    control_action = (
                        controls[interface][arm_index]
                        if stage == "calibrated"
                        else deployed_controls[interface][arm_index]
                    )
                    arm_effect["action_fractions"][cell][f"{stage}|{interface}|{CONTROL_BASE}"] = {
                        str(alpha): float(np.mean(control_action[:, indices] == alpha_index))
                        for alpha_index, alpha in enumerate(cfg["alphas"])
                    }
        for stage, stage_actions in (("calibrated", actions), ("deployed", deployed)):
            signed_topology = stage_actions["signed_tail"]["topology"][arm_index]
            absolute_topology = stage_actions["absolute_scale"]["topology"][arm_index]
            arm_effect["action_overlap"][stage] = {
                "same_action_fraction": float(np.mean(signed_topology == absolute_topology)),
                "both_act_fraction": float(np.mean((signed_topology > 0) & (absolute_topology > 0))),
                "signed_only_fraction": float(np.mean((signed_topology > 0) & (absolute_topology == 0))),
                "absolute_only_fraction": float(np.mean((signed_topology == 0) & (absolute_topology > 0))),
            }
        effects["arms"][arm] = arm_effect

        arm_diag: dict[str, Any] = {
            "coverage": {}, "acted_harm": {}, "interaction_grouped_minus_iid": {},
            "residual_asymmetry": {
                "adjudication_iid": signed.describe_residual((delta[arm_index] - mu[arm_index])[iid]),
                "adjudication_grouped": signed.describe_residual((delta[arm_index] - mu[arm_index])[grouped]),
            },
        }
        for interface in INTERFACES:
            arm_diag["coverage"][interface] = {}
            arm_diag["acted_harm"][interface] = {}
            arm_diag["interaction_grouped_minus_iid"][interface] = {}
            for family in FAMILY_BASES:
                covered = np.all(delta[arm_index] <= uppers[interface][family][arm_index], axis=1)
                arm_diag["coverage"][interface][family] = {
                    "iid": float(np.mean(covered[iid])),
                    "grouped": float(np.mean(covered[grouped])),
                }
                for stage, stage_actions in (("calibrated", actions), ("deployed", deployed)):
                    action = stage_actions[interface][family][arm_index]
                    realized = conditional.selected_delta(quotient, action)
                    key = f"{stage}|{family}"
                    arm_diag["acted_harm"][interface][key] = {}
                    for cell, indices in (("iid", iid), ("grouped", grouped)):
                        acted = action[indices] > 0
                        acted_delta = realized[indices][acted]
                        arm_diag["acted_harm"][interface][key][cell] = {
                            "n": int(acted.sum()),
                            "mean_delta": float(acted_delta.mean()) if len(acted_delta) else None,
                            "positive_fraction": float(np.mean(acted_delta > 0)) if len(acted_delta) else None,
                        }
                    iid_effect = cell_values["iid"][f"{stage}|{interface}|{family}"] - cell_values["iid"]["identity"]
                    grouped_effect = cell_values["grouped"][f"{stage}|{interface}|{family}"] - cell_values["grouped"]["identity"]
                    arm_diag["interaction_grouped_minus_iid"][interface][key] = interval(
                        grouped_effect - iid_effect, bootstrap
                    )
            control_coverage = []
            for replicate in range(cfg["control_replicates"]):
                control_coverage.append(np.all(
                    delta[arm_index] <= control_uppers[interface][arm_index, replicate], axis=1
                ))
            stacked = np.asarray(control_coverage)
            arm_diag["coverage"][interface][CONTROL_BASE] = {
                "iid": float(np.mean(stacked[:, iid])),
                "grouped": float(np.mean(stacked[:, grouped])),
            }
        diagnostics["arms"][arm] = arm_diag
    return effects, diagnostics


def deterministic_files(output: Path, roles: tuple[str, ...]) -> list[Path]:
    return sorted(path for role in roles for path in (output / role).rglob("*") if path.is_file())


def source_hashes() -> dict[str, str]:
    return {relative: sha(ROOT / relative) for relative in SOURCE_FILES}


def input_hashes(cfg: dict[str, Any]) -> dict[str, str]:
    return {str(path.relative_to(ROOT)): sha(path) for path in conditional.source_inputs(cfg)}


def write_phase_freeze(
    output: Path, phase: str, roles: tuple[str, ...], protected: dict[str, Path], future: list[str],
) -> None:
    files = deterministic_files(output, roles)
    manifest = {
        "schema_version": f"signed-tail-gate-{phase}-manifest-v1",
        "git_head": git_head(), "roles": list(roles),
        "files": {str(path.relative_to(output)): sha(path) for path in files},
    }
    write_json(output / f"{phase}_manifest.json", manifest)
    freeze = {
        "schema_version": f"signed-tail-gate-{phase}-freeze-v1",
        "git_head": git_head(), "manifest_sha256": sha(output / f"{phase}_manifest.json"),
        "protected": {name: sha(path) for name, path in protected.items()},
        "future_materialized": {name: (output / name).exists() for name in future},
    }
    if any(freeze["future_materialized"].values()):
        raise AssertionError("future phase materialized before freeze")
    write_json(output / f"{phase}_freeze.json", freeze)


def verify_phase(output: Path, phase: str) -> None:
    manifest_path = output / f"{phase}_manifest.json"
    freeze_path = output / f"{phase}_freeze.json"
    manifest = json.loads(manifest_path.read_text())
    freeze = json.loads(freeze_path.read_text())
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


def setup_phase(
    cfg: dict[str, Any], output: Path, role: str, seed_key: str,
) -> tuple[list[Any], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any], float, list[dict[str, Any]]]:
    started, samples = time.monotonic(), []
    conditional.setup()
    source_cfg, forward_models, mean_models = conditional.phase_context(cfg)
    views, data, topology = conditional.generate_phase(
        role, seed_key, cfg, output, source_cfg, forward_models, started, samples
    )
    return views, data, topology, mean_models, started, samples


def close_phase_resources(
    cfg: dict[str, Any], output: Path, role: str, started: float, samples: list[dict[str, Any]],
) -> None:
    sample = mixed.resource_sample(started, f"{role}_complete")
    mixed.enforce(sample, cfg)
    samples.append(sample)
    write_json(output / f"resource_samples_{role}.json", samples)
    print(json.dumps(sample, sort_keys=True))


def risk_fit_phase(cfg: dict[str, Any], output: Path, development: bool) -> None:
    require_clean(development)
    if output.exists():
        raise FileExistsError(output)
    if not development and (ROOT / "data/geometria_proporcional").resolve() not in output.resolve().parents:
        raise ValueError("official output must be below data/geometria_proporcional")
    output.mkdir(parents=True)
    views, data, topology, mean_models, started, samples = setup_phase(
        cfg, output, "risk_fit", "risk_fit_seed"
    )
    absolute_models, absolute_fit = conditional.fit_scale_models(cfg, views, data, topology, mean_models)
    signed_models, signed_fit = fit_signed_models(cfg, views, data, topology, mean_models)
    write_json(output / "risk_fit/absolute_models.json", absolute_models)
    write_json(output / "risk_fit/signed_models.json", signed_models)
    frozen.save_npz(output / "risk_fit/absolute_fit.npz", absolute_fit)
    frozen.save_npz(output / "risk_fit/signed_fit.npz", signed_fit)
    write_json(output / "risk_fit/environment.json", {
        "python": platform.python_version(),
        "numpy": importlib.metadata.version("numpy"),
        "scipy": importlib.metadata.version("scipy"),
        "scikit_learn": importlib.metadata.version("scikit-learn"),
        "quantile_solver": "highs", "threads": 1, "cuda_visible_devices": "",
    })
    write_json(output / "resolved_config.json", cfg)
    write_phase_freeze(output, "risk_fit", ("risk_fit",), {
        "resolved_config.json": output / "resolved_config.json",
        "risk_fit/absolute_models.json": output / "risk_fit/absolute_models.json",
        "risk_fit/signed_models.json": output / "risk_fit/signed_models.json",
    }, ["risk_calibration", "policy_selection", "adjudication"])
    write_json(output / "phase_receipt.json", {"phase": "risk_fit", "next": "risk_calibration"})
    close_phase_resources(cfg, output, "risk_fit", started, samples)


def load_risk_models(output: Path) -> dict[str, Any]:
    return {
        "absolute_scale": json.loads((output / "risk_fit/absolute_models.json").read_text()),
        "signed_tail": json.loads((output / "risk_fit/signed_models.json").read_text()),
    }


def risk_calibration_phase(cfg: dict[str, Any], output: Path, development: bool) -> None:
    require_clean(development)
    verify_phase(output, "risk_fit")
    verify_resolved_config(cfg, output)
    if (output / "risk_calibration").exists():
        raise RuntimeError("risk calibration already exists")
    views, data, topology, mean_models, started, samples = setup_phase(
        cfg, output, "risk_calibration", "risk_calibration_seed"
    )
    if conditional.prior_ids(output, ("risk_fit",)) & {view.private.master_id for view in views}:
        raise AssertionError("risk fit and calibration overlap")
    risk_models = load_risk_models(output)
    absolute, absolute_arrays = conditional.calibrate_risk(
        cfg, data, topology, mean_models, risk_models["absolute_scale"]
    )
    signed_calibration, signed_arrays = calibrate_signed(
        cfg, data, topology, mean_models, risk_models["signed_tail"]
    )
    calibration = {"interfaces": {"signed_tail": signed_calibration, "absolute_scale": absolute}}
    write_json(output / "risk_calibration/quantiles.json", calibration)
    frozen.save_npz(output / "risk_calibration/absolute_scores.npz", absolute_arrays)
    frozen.save_npz(output / "risk_calibration/signed_scores.npz", signed_arrays)
    write_phase_freeze(output, "risk_calibration", ("risk_calibration",), {
        "risk_fit_freeze.json": output / "risk_fit_freeze.json",
        "risk_calibration/quantiles.json": output / "risk_calibration/quantiles.json",
    }, ["policy_selection", "adjudication"])
    write_json(output / "phase_receipt.json", {"phase": "risk_calibration", "next": "policy_selection"})
    close_phase_resources(cfg, output, "risk_calibration", started, samples)


def policy_selection_phase(cfg: dict[str, Any], output: Path, development: bool) -> None:
    require_clean(development)
    verify_phase(output, "risk_fit")
    verify_phase(output, "risk_calibration")
    verify_resolved_config(cfg, output)
    if (output / "policy_selection").exists() or (output / "adjudication").exists():
        raise RuntimeError("selection requires risk-calibration-only output")
    views, data, topology, mean_models, started, samples = setup_phase(
        cfg, output, "policy_selection", "policy_selection_seed"
    )
    if conditional.prior_ids(output, ("risk_fit", "risk_calibration")) & {
        view.private.master_id for view in views
    }:
        raise AssertionError("policy selection overlaps prior phase")
    risk_models = load_risk_models(output)
    calibration = json.loads((output / "risk_calibration/quantiles.json").read_text())
    firewall, packed = select_policies(cfg, data, topology, mean_models, risk_models, calibration)
    write_json(output / "policy_selection/firewall.json", firewall)
    frozen.save_npz(output / "policy_selection/selection_statistics.npz", packed)
    write_phase_freeze(output, "policy_selection", ("policy_selection",), {
        "risk_fit_freeze.json": output / "risk_fit_freeze.json",
        "risk_calibration_freeze.json": output / "risk_calibration_freeze.json",
        "policy_selection/firewall.json": output / "policy_selection/firewall.json",
    }, ["adjudication"])
    write_json(output / "phase_receipt.json", {"phase": "policy_selection", "next": "adjudication"})
    close_phase_resources(cfg, output, "policy_selection", started, samples)


def write_replay(output: Path, cfg: dict[str, Any], head: str) -> None:
    checks = "\n".join(
        f"printf '%s  %s\\n' '{sha(ROOT / relative)}' \"$repo/{relative}\" | sha256sum -c -"
        for relative in SOURCE_FILES
    )
    checks += "\n" + "\n".join(
        f"printf '%s  %s\\n' '{sha(path)}' \"$repo/{path.relative_to(ROOT)}\" | sha256sum -c -"
        for path in conditional.source_inputs(cfg)
    )
    command = (
        "env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "
        "OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \"$repo/venv/bin/python\" "
        "\"$repo/experiments/geometria_proporcional/run_proportional_graph_signed_tail_gate.py\""
    )
    replay = f'''#!/bin/sh
set -eu
repo=$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd)
[ "$(git -C "$repo" rev-parse HEAD)" = "{head}" ] || {{ echo "wrong git HEAD" >&2; exit 2; }}
[ -z "$(git -C "$repo" status --porcelain)" ] || {{ echo "dirty worktree" >&2; exit 2; }}
{checks}
: "${{OUTPUT_DIR:?set OUTPUT_DIR to a new path}}"
{command} --phase risk-fit --config "$repo/experiments/geometria_proporcional/configs/proportional_graph_signed_tail_gate_v1.json" --output "$OUTPUT_DIR"
{command} --phase risk-calibrate --config "$repo/experiments/geometria_proporcional/configs/proportional_graph_signed_tail_gate_v1.json" --output "$OUTPUT_DIR"
{command} --phase select --config "$repo/experiments/geometria_proporcional/configs/proportional_graph_signed_tail_gate_v1.json" --output "$OUTPUT_DIR"
exec {command} --phase evaluate --config "$repo/experiments/geometria_proporcional/configs/proportional_graph_signed_tail_gate_v1.json" --output "$OUTPUT_DIR"
'''
    (output / "replay.sh").write_text(replay)
    (output / "replay.sh").chmod(0o755)


def evaluation_phase(cfg: dict[str, Any], output: Path, development: bool) -> None:
    require_clean(development)
    for phase in ("risk_fit", "risk_calibration", "policy_selection"):
        verify_phase(output, phase)
    verify_resolved_config(cfg, output)
    if (output / "adjudication").exists():
        raise RuntimeError("adjudication already exists")
    views, data, topology, mean_models, started, samples = setup_phase(
        cfg, output, "adjudication", "adjudication_seed"
    )
    if conditional.prior_ids(output, ("risk_fit", "risk_calibration", "policy_selection")) & {
        view.private.master_id for view in views
    }:
        raise AssertionError("adjudication overlaps prior phases")
    risk_models = load_risk_models(output)
    calibration = json.loads((output / "risk_calibration/quantiles.json").read_text())
    firewall = json.loads((output / "policy_selection/firewall.json").read_text())
    delta = conditional.realized_deltas(data)
    mu = conditional.mean_predictions(cfg, data, topology, mean_models)
    actions, controls, uppers, control_uppers, risks, control_risks = paired_actions(
        cfg, data, topology, mean_models, risk_models, calibration
    )
    deployed, deployed_controls = apply_firewall(cfg, actions, controls, firewall)
    packed: dict[str, np.ndarray] = {"mu": mu, "delta": delta}
    for interface in INTERFACES:
        packed[f"{interface}|control_risk"] = control_risks[interface]
        packed[f"{interface}|control_upper"] = control_uppers[interface]
        packed[f"{interface}|calibrated_control"] = controls[interface]
        packed[f"{interface}|deployed_control"] = deployed_controls[interface]
        for family in FAMILY_BASES:
            packed[f"{interface}|{family}|risk"] = risks[interface][family]
            packed[f"{interface}|{family}|upper"] = uppers[interface][family]
            packed[f"{interface}|{family}|calibrated"] = actions[interface][family]
            packed[f"{interface}|{family}|deployed"] = deployed[interface][family]
    frozen.save_npz(output / "adjudication/decisions_and_risk.npz", packed)
    n_masters = len(views) // 2
    bootstrap = np.random.default_rng(cfg["evaluation_bootstrap_seed"]).integers(
        0, n_masters, size=(cfg["evaluation_bootstrap_replicates"], n_masters)
    )
    frozen.save_npz(output / "bootstrap_indices.npz", {"indices": bootstrap})
    effects, diagnostics = summarize(
        cfg, data, delta, mu, risk_models, calibration, firewall,
        actions, controls, deployed, deployed_controls, uppers, control_uppers, bootstrap,
    )
    write_json(output / "effects.json", effects)
    write_json(output / "risk_diagnostics.json", diagnostics)
    write_replay(output, cfg, git_head())
    close_phase_resources(cfg, output, "adjudication", started, samples)
    deterministic = sorted(
        path for path in output.rglob("*") if path.is_file() and path.name not in {
            "manifest.json", "resource_samples_risk_fit.json",
            "resource_samples_risk_calibration.json", "resource_samples_policy_selection.json",
            "resource_samples_adjudication.json", "runtime_observation.json", "phase_receipt.json",
        }
    )
    write_json(output / "manifest.json", {
        "schema_version": cfg["schema_version"], "git_head": git_head(),
        "source_hashes": source_hashes(), "input_hashes": input_hashes(cfg),
        "deterministic_files": {str(path.relative_to(output)): sha(path) for path in deterministic},
        "runtime_exclusions": [
            "resource_samples_risk_fit.json", "resource_samples_risk_calibration.json",
            "resource_samples_policy_selection.json", "resource_samples_adjudication.json",
            "runtime_observation.json", "phase_receipt.json",
        ],
    })
    runtime = mixed.resource_sample(started, "adjudication_complete")
    write_json(output / "runtime_observation.json", runtime)
    write_json(output / "phase_receipt.json", {
        "phase": "complete", "manifest_sha256": sha(output / "manifest.json"),
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("risk-fit", "risk-calibrate", "select", "evaluate"), required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--development", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config.resolve())
    output = args.output.resolve()
    if args.phase == "risk-fit":
        risk_fit_phase(cfg, output, args.development)
    elif args.phase == "risk-calibrate":
        risk_calibration_phase(cfg, output, args.development)
    elif args.phase == "select":
        policy_selection_phase(cfg, output, args.development)
    else:
        evaluation_phase(cfg, output, args.development)


if __name__ == "__main__":
    main()
