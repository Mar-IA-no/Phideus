#!/usr/bin/env python3
"""Audit selected-action transport and fixed-effect power from two opened CPU cohorts."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
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
import run_proportional_graph_selected_action_calibration_diagnostic as selected  # noqa: E402
import run_proportional_graph_signed_tail_diagnostic as signed  # noqa: E402

DEFAULT_CONFIG = ROOT / "experiments/geometria_proporcional/configs/proportional_graph_selected_action_transport_power_audit_v1.json"
SOURCE_FILES = (
    "experiments/geometria_proporcional/PLAN_PROPORTIONAL_SELECTED_ACTION_TRANSPORT_POWER_AUDIT_CPU.md",
    "experiments/geometria_proporcional/configs/proportional_graph_selected_action_transport_power_audit_v1.json",
    "experiments/geometria_proporcional/run_proportional_graph_selected_action_transport_power_audit.py",
    "experiments/geometria_proporcional/run_proportional_graph_selected_action_calibration_diagnostic.py",
    "experiments/geometria_proporcional/run_proportional_graph_signed_tail_diagnostic.py",
    "experiments/geometria_proporcional/run_proportional_graph_conditional_risk_gate.py",
    "experiments/geometria_proporcional/run_proportional_graph_fresh_mixed_gate.py",
    "experiments/geometria_proporcional/run_proportional_graph_frozen_adapters.py",
)
FAMILIES = selected.FAMILIES
CONTROL_FAMILY = selected.CONTROL_FAMILY
BASE_NAME = selected.BASE_NAME
SIGNED_NAME = {
    "constant_selected_action": "constant_signed_tail",
    "public_base_selected_action": "public_base_signed_tail",
    "topology_selected_action": "topology_signed_tail",
}
CONTRASTS = ("minus_identity", "minus_public_base", "minus_topology_permuted")
CELLS = ("iid", "grouped", "balanced")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def read_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key] for key in saved.files}


def load_config(path: Path) -> dict[str, Any]:
    cfg = json.loads(path.read_text())
    expected = {
        "schema_version", "cohort_a_data", "cohort_a_data_manifest_sha256",
        "cohort_a_models", "cohort_a_models_manifest_sha256", "cohort_b_data",
        "cohort_b_data_manifest_sha256", "cohort_b_selected",
        "cohort_b_selected_manifest_sha256", "arms", "alphas", "miscoverage",
        "control_replicates", "selection_upper_percentile",
        "numerical_zero_tolerance", "historical_realizable_masters", "execution",
    }
    if set(cfg) != expected or cfg["schema_version"] != "proportional-graph-selected-action-transport-power-audit-v1":
        raise ValueError("invalid selected-action transport audit schema")
    if cfg["arms"] != ["raw_generic", "raw_typed", "closure_generic", "closure_typed"]:
        raise ValueError("arms changed")
    if cfg["alphas"] != [0.0, 0.25, 0.5, 0.75, 1.0]:
        raise ValueError("alpha grid changed")
    if cfg["miscoverage"] != 0.10 or cfg["control_replicates"] != 16:
        raise ValueError("selected-action calibration contract changed")
    if cfg["selection_upper_percentile"] != 95.0:
        raise ValueError("firewall contract changed")
    if cfg["numerical_zero_tolerance"] != 1e-12 or cfg["historical_realizable_masters"] != 250:
        raise ValueError("transport interpretation contract changed")
    if cfg["execution"] != {"max_seconds": 300, "max_rss_gib": 4.0}:
        raise ValueError("CPU execution contract changed")
    return cfg


def roots(cfg: dict[str, Any]) -> dict[str, Path]:
    return {
        key: ROOT / cfg[key]
        for key in ("cohort_a_data", "cohort_a_models", "cohort_b_data", "cohort_b_selected")
    }


def verify_manifest(root: Path, expected_sha: str) -> dict[str, Any]:
    if sha(root / "manifest.json") != expected_sha:
        raise AssertionError(f"manifest hash mismatch: {root}")
    manifest = json.loads((root / "manifest.json").read_text())
    for relative, expected in manifest["deterministic_files"].items():
        if sha(root / relative) != expected:
            raise AssertionError(f"opened source mismatch: {root / relative}")
    return manifest


def verify_sources(cfg: dict[str, Any]) -> dict[str, Any]:
    source_roots = roots(cfg)
    manifests = {
        key: verify_manifest(source_roots[key], cfg[f"{key}_manifest_sha256"])
        for key in source_roots
    }
    configs = {
        key: json.loads((root / "resolved_config.json").read_text())
        for key, root in source_roots.items()
    }
    for key, source_cfg in configs.items():
        if source_cfg["arms"] != cfg["arms"] or source_cfg["alphas"] != cfg["alphas"]:
            raise AssertionError(f"factorial mismatch: {key}")
        if source_cfg["control_replicates"] != cfg["control_replicates"]:
            raise AssertionError(f"control mismatch: {key}")
    if configs["cohort_a_models"]["miscoverage"] != cfg["miscoverage"]:
        raise AssertionError("cohort A coverage mismatch")
    if configs["cohort_b_selected"]["miscoverage"] != cfg["miscoverage"]:
        raise AssertionError("cohort B coverage mismatch")
    return {"manifests": manifests, "configs": configs}


def require_clean(development: bool) -> None:
    if not development and subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout:
        raise RuntimeError("official audit requires clean worktree")


def git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout.strip()


def normalize_a_state(
    cfg: dict[str, Any], data_root: Path, role: str, model_cfg: dict[str, Any], models: dict[str, Any],
) -> dict[str, Any]:
    arrays = signed.phase_arrays(data_root, role)
    predictions, controls = signed.predictions(model_cfg, data_root, role, models)
    data, _ = conditional.load_phase_data(data_root, role)
    predicted = {family: predictions[SIGNED_NAME[family]] for family in FAMILIES}
    values = [arrays["mu"], arrays["delta"], controls, *predicted.values()]
    if any(not np.all(np.isfinite(value)) for value in values):
        raise RuntimeError(f"non-finite cohort A state: {role}")
    return {
        "data": data, "mu": arrays["mu"], "delta": arrays["delta"],
        "predicted": predicted, "control": controls,
    }


def calibrate_generic(
    cfg: dict[str, Any], state: dict[str, Any], simultaneous: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    iid = np.arange(0, state["delta"].shape[1], 2)
    residual = state["delta"] - state["mu"]
    result: dict[str, Any] = {"miscoverage": cfg["miscoverage"], "arms": {}}
    packed: dict[str, np.ndarray] = {"iid_index": iid}
    for arm_index, arm in enumerate(cfg["arms"]):
        record: dict[str, Any] = {"families": {}, CONTROL_FAMILY: []}
        for family in FAMILIES:
            prediction = state["predicted"][family][arm_index]
            proposal, base_bound = selected.propose(state["mu"][arm_index], prediction)
            score = selected.selected_score(residual[arm_index], prediction, proposal)
            q, rank = conditional.conformal_quantile(score[iid], cfg["miscoverage"])
            q_sim = simultaneous["arms"][arm]["families"][SIGNED_NAME[family]]["q"]
            if q > q_sim + cfg["numerical_zero_tolerance"]:
                raise AssertionError("cohort A selected quantile exceeds simultaneous quantile")
            record["families"][family] = {
                "q_selected": q, "q_simultaneous": q_sim, "rank": rank, "n": len(iid),
            }
            packed[f"{arm}|{family}|proposal"] = proposal
            packed[f"{arm}|{family}|base_bound"] = base_bound
            packed[f"{arm}|{family}|score"] = score[iid]
        for replicate in range(cfg["control_replicates"]):
            prediction = state["control"][arm_index, replicate]
            proposal, base_bound = selected.propose(state["mu"][arm_index], prediction)
            score = selected.selected_score(residual[arm_index], prediction, proposal)
            q, rank = conditional.conformal_quantile(score[iid], cfg["miscoverage"])
            q_sim = simultaneous["arms"][arm][signed.CONTROL_FAMILY][replicate]["q"]
            if q > q_sim + cfg["numerical_zero_tolerance"]:
                raise AssertionError("cohort A selected control quantile exceeds simultaneous quantile")
            record[CONTROL_FAMILY].append({
                "replicate": replicate, "q_selected": q, "q_simultaneous": q_sim,
                "rank": rank, "n": len(iid),
            })
            packed[f"{arm}|{CONTROL_FAMILY}={replicate}|proposal"] = proposal
            packed[f"{arm}|{CONTROL_FAMILY}={replicate}|base_bound"] = base_bound
            packed[f"{arm}|{CONTROL_FAMILY}={replicate}|score"] = score[iid]
        result["arms"][arm] = record
    return result, packed


def actions_generic(
    cfg: dict[str, Any], state: dict[str, Any], calibration: dict[str, Any],
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, np.ndarray], np.ndarray, dict[str, np.ndarray], np.ndarray]:
    shape = state["mu"].shape[:2]
    actions = {family: np.zeros(shape, dtype=np.int64) for family in FAMILIES}
    proposals = {family: np.zeros(shape, dtype=np.int64) for family in FAMILIES}
    uppers = {family: np.full(shape, np.nan) for family in FAMILIES}
    controls = np.zeros((len(cfg["arms"]), cfg["control_replicates"], shape[1]), dtype=np.int64)
    control_proposals = np.zeros_like(controls)
    control_uppers = np.full_like(controls, np.nan, dtype=np.float64)
    for arm_index, arm in enumerate(cfg["arms"]):
        for family in FAMILIES:
            proposal, base_bound = selected.propose(
                state["mu"][arm_index], state["predicted"][family][arm_index]
            )
            action, upper = selected.calibrated_action(
                proposal, base_bound, calibration["arms"][arm]["families"][family]["q_selected"]
            )
            proposals[family][arm_index], actions[family][arm_index], uppers[family][arm_index] = proposal, action, upper
        for replicate in range(cfg["control_replicates"]):
            proposal, base_bound = selected.propose(
                state["mu"][arm_index], state["control"][arm_index, replicate]
            )
            action, upper = selected.calibrated_action(
                proposal, base_bound, calibration["arms"][arm][CONTROL_FAMILY][replicate]["q_selected"]
            )
            control_proposals[arm_index, replicate] = proposal
            controls[arm_index, replicate] = action
            control_uppers[arm_index, replicate] = upper
    return actions, controls, proposals, control_proposals, uppers, control_uppers


def firewall_generic(
    cfg: dict[str, Any], state: dict[str, Any], calibration: dict[str, Any], bootstrap: np.ndarray,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    actions, controls, proposals, control_proposals, uppers, control_uppers = actions_generic(
        cfg, state, calibration
    )
    result: dict[str, Any] = {"arms": {}}
    packed: dict[str, np.ndarray] = {
        "bootstrap_indices": bootstrap, "control_proposal": control_proposals,
        "control_action": controls, "control_upper": control_uppers,
    }
    for family in FAMILIES:
        packed[f"{family}|proposal"] = proposals[family]
        packed[f"{family}|action"] = actions[family]
        packed[f"{family}|upper"] = uppers[family]
    for arm_index, arm in enumerate(cfg["arms"]):
        quotient = state["data"]["quotient_rmse"][arm_index]
        record: dict[str, Any] = {"families": {}, CONTROL_FAMILY: []}
        for family in FAMILIES:
            delta = conditional.selected_delta(quotient, actions[family][arm_index])
            record["families"][family] = conditional.firewall_one(
                delta, bootstrap, cfg["selection_upper_percentile"]
            )
        for replicate in range(cfg["control_replicates"]):
            delta = conditional.selected_delta(quotient, controls[arm_index, replicate])
            record[CONTROL_FAMILY].append({
                "replicate": replicate,
                **conditional.firewall_one(delta, bootstrap, cfg["selection_upper_percentile"]),
            })
        result["arms"][arm] = record
    return result, packed


def pack_adjudication(
    actions: dict[str, np.ndarray], controls: np.ndarray, proposals: dict[str, np.ndarray],
    control_proposals: np.ndarray, uppers: dict[str, np.ndarray], control_uppers: np.ndarray,
    deployed: dict[str, np.ndarray], deployed_controls: np.ndarray,
) -> dict[str, np.ndarray]:
    packed = {
        "control_proposal": control_proposals, "control_upper": control_uppers,
        "calibrated_control": controls, "deployed_control": deployed_controls,
    }
    for family in FAMILIES:
        packed[f"{family}|proposal"] = proposals[family]
        packed[f"{family}|upper"] = uppers[family]
        packed[f"{family}|calibrated"] = actions[family]
        packed[f"{family}|deployed"] = deployed[family]
    return packed


def reconstruct_a(cfg: dict[str, Any]) -> dict[str, Any]:
    source_roots = roots(cfg)
    data_root, model_root = source_roots["cohort_a_data"], source_roots["cohort_a_models"]
    model_cfg = json.loads((model_root / "resolved_config.json").read_text())
    models = json.loads((model_root / "quantile_models.json").read_text())
    states = {
        role: normalize_a_state(cfg, data_root, role, model_cfg, models)
        for role in ("risk_calibration", "policy_selection", "adjudication")
    }
    simultaneous = json.loads((model_root / "calibration.json").read_text())
    calibration, calibration_pack = calibrate_generic(cfg, states["risk_calibration"], simultaneous)
    selection_bootstrap = signed.phase_arrays(data_root, "policy_selection")["bootstrap_indices"].astype(np.int64)
    firewall, selection_pack = firewall_generic(
        cfg, states["policy_selection"], calibration, selection_bootstrap
    )
    actions, controls, proposals, control_proposals, uppers, control_uppers = actions_generic(
        cfg, states["adjudication"], calibration
    )
    deployed, deployed_controls = selected.apply_firewall(cfg, actions, controls, firewall)
    adjudication_pack = pack_adjudication(
        actions, controls, proposals, control_proposals, uppers, control_uppers,
        deployed, deployed_controls,
    )
    bootstrap = read_npz(data_root / "bootstrap_indices.npz")["indices"].astype(np.int64)
    return {
        "calibration": calibration, "firewall": firewall, "data": states["adjudication"]["data"],
        "state": states["adjudication"],
        "bootstrap": bootstrap, "actions": actions, "controls": controls,
        "deployed": deployed, "deployed_controls": deployed_controls,
        "calibration_pack": calibration_pack, "selection_pack": selection_pack,
        "adjudication_pack": adjudication_pack,
    }


def assert_arrays_exact(actual: dict[str, np.ndarray], expected_path: Path, label: str) -> None:
    expected = read_npz(expected_path)
    if set(actual) != set(expected):
        raise AssertionError(f"{label} array keys differ")
    for key in actual:
        if not np.array_equal(actual[key], expected[key], equal_nan=True):
            raise AssertionError(f"{label} array differs: {key}")


def reconstruct_b(cfg: dict[str, Any]) -> dict[str, Any]:
    selected_root = roots(cfg)["cohort_b_selected"]
    selected_cfg = selected.load_config(selected_root / "resolved_config.json")
    calibration, calibration_pack = selected.calibrate(selected_cfg)
    firewall, selection_pack = selected.selection_firewall(selected_cfg, calibration)
    actions, controls, proposals, control_proposals, uppers, control_uppers = selected.actions_for_role(
        selected_cfg, "adjudication", calibration
    )
    analysis, adjudication_pack = selected.summarize(
        selected_cfg, calibration, firewall, actions, controls, proposals,
        control_proposals, uppers, control_uppers,
    )
    if calibration != json.loads((selected_root / "calibration.json").read_text()):
        raise AssertionError("cohort B calibration JSON does not reproduce R364")
    if firewall != json.loads((selected_root / "selection_firewall.json").read_text()):
        raise AssertionError("cohort B firewall JSON does not reproduce R364")
    if analysis != json.loads((selected_root / "analysis.json").read_text()):
        raise AssertionError("cohort B analysis JSON does not reproduce R364")
    assert_arrays_exact(calibration_pack, selected_root / "calibration_diagnostics.npz", "cohort B calibration")
    assert_arrays_exact(selection_pack, selected_root / "selection_diagnostics.npz", "cohort B selection")
    assert_arrays_exact(adjudication_pack, selected_root / "adjudication_diagnostics.npz", "cohort B adjudication")
    deployed, deployed_controls = selected.apply_firewall(selected_cfg, actions, controls, firewall)
    state = selected.role_state(selected_cfg, "adjudication")
    normalized_state = {
        **state,
        "predicted": {family: state["predicted"][BASE_NAME[family]] for family in FAMILIES},
    }
    bootstrap = read_npz(roots(cfg)["cohort_b_data"] / "bootstrap_indices.npz")["indices"].astype(np.int64)
    return {
        "calibration": calibration, "firewall": firewall, "data": state["data"],
        "state": normalized_state,
        "bootstrap": bootstrap, "actions": actions, "controls": controls,
        "deployed": deployed, "deployed_controls": deployed_controls,
        "calibration_pack": calibration_pack, "selection_pack": selection_pack,
        "adjudication_pack": adjudication_pack, "exact_reproduction": True,
    }


def interval(values: np.ndarray, bootstrap: np.ndarray) -> dict[str, Any]:
    draws = values[bootstrap].mean(axis=1)
    return {
        "mean": float(np.mean(values)),
        "ci95": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
    }


def cell_values(raw: np.ndarray, cell: str) -> np.ndarray:
    iid = np.arange(0, raw.shape[1], 2)
    grouped = np.arange(1, raw.shape[1], 2)
    if cell == "iid":
        return raw[:, iid].mean(axis=0)
    if cell == "grouped":
        return raw[:, grouped].mean(axis=0)
    if cell == "balanced":
        return (0.5 * (raw[:, iid] + raw[:, grouped])).mean(axis=0)
    raise ValueError(f"unknown cell: {cell}")


def summarize_cohort(
    cfg: dict[str, Any], name: str, reconstructed: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    report: dict[str, Any] = {
        "cohort": name, "n_masters": int(reconstructed["bootstrap"].shape[1]), "arms": {},
    }
    packed: dict[str, np.ndarray] = {"bootstrap_indices": reconstructed["bootstrap"]}
    for arm_index, arm in enumerate(cfg["arms"]):
        quotient = reconstructed["data"]["quotient_rmse"][arm_index]
        state = reconstructed["state"]
        residual = state["delta"][arm_index] - state["mu"][arm_index]
        arm_report: dict[str, Any] = {
            "calibration": reconstructed["calibration"]["arms"][arm],
            "firewall": reconstructed["firewall"]["arms"][arm],
            "coverage": {}, "stages": {},
        }
        iid = np.arange(0, residual.shape[0], 2)
        grouped = np.arange(1, residual.shape[0], 2)
        for family in FAMILIES:
            proposal = reconstructed["adjudication_pack"][f"{family}|proposal"][arm_index]
            index = proposal - 1
            rows = np.arange(len(index))
            chosen_residual = residual[rows, index]
            chosen_prediction = state["predicted"][family][arm_index, rows, index]
            q = reconstructed["calibration"]["arms"][arm]["families"][family]["q_selected"]
            covered = chosen_residual <= chosen_prediction + q
            arm_report["coverage"][family] = {
                "iid": float(np.mean(covered[iid])),
                "grouped": float(np.mean(covered[grouped])),
            }
        control_coverage = {"iid": [], "grouped": []}
        for replicate in range(cfg["control_replicates"]):
            proposal = reconstructed["adjudication_pack"]["control_proposal"][arm_index, replicate]
            index = proposal - 1
            rows = np.arange(len(index))
            chosen_residual = residual[rows, index]
            chosen_prediction = state["control"][arm_index, replicate, rows, index]
            q = reconstructed["calibration"]["arms"][arm][CONTROL_FAMILY][replicate]["q_selected"]
            covered = chosen_residual <= chosen_prediction + q
            control_coverage["iid"].append(float(np.mean(covered[iid])))
            control_coverage["grouped"].append(float(np.mean(covered[grouped])))
        arm_report["coverage"][CONTROL_FAMILY] = {
            cell: {
                "mean": float(np.mean(values)), "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
            for cell, values in control_coverage.items()
        }
        for stage, actions, controls in (
            ("calibrated", reconstructed["actions"], reconstructed["controls"]),
            ("deployed", reconstructed["deployed"], reconstructed["deployed_controls"]),
        ):
            raw = {
                "identity": quotient[:, 0],
                "public_base": mixed.selected_values(
                    quotient, actions["public_base_selected_action"][arm_index]
                ),
                "topology": mixed.selected_values(
                    quotient, actions["topology_selected_action"][arm_index]
                ),
                "topology_permuted": np.mean([
                    mixed.selected_values(quotient, controls[arm_index, replicate])
                    for replicate in range(cfg["control_replicates"])
                ], axis=0),
            }
            stage_report: dict[str, Any] = {"action": {}, "acted_harm": {}, "cells": {}}
            topology_action = actions["topology_selected_action"][arm_index]
            for cell, indices in (("iid", iid), ("grouped", grouped), ("balanced", np.arange(len(topology_action)))):
                stage_report["action"][cell] = {
                    "acted_views": int(np.sum(topology_action[indices] > 0)),
                    "total_views": int(len(indices)),
                    "rate": float(np.mean(topology_action[indices] > 0)),
                    "alpha_counts": {
                        str(alpha): int(np.sum(topology_action[indices] == alpha_index))
                        for alpha_index, alpha in enumerate(cfg["alphas"])
                    },
                }
            for family in FAMILIES:
                family_action = actions[family][arm_index]
                realized = conditional.selected_delta(quotient, family_action)
                stage_report["acted_harm"][family] = {}
                for cell, indices in (("iid", iid), ("grouped", grouped)):
                    acted = family_action[indices] > 0
                    selected_delta = realized[indices][acted]
                    stage_report["acted_harm"][family][cell] = {
                        "n": int(np.sum(acted)),
                        "mean_delta": float(np.mean(selected_delta)) if len(selected_delta) else None,
                        "positive_fraction": float(np.mean(selected_delta > 0.0)) if len(selected_delta) else None,
                    }
            for cell in CELLS:
                values = {policy: cell_values(value, cell) for policy, value in raw.items()}
                contrasts = {
                    "minus_identity": values["topology"] - values["identity"],
                    "minus_public_base": values["topology"] - values["public_base"],
                    "minus_topology_permuted": values["topology"] - values["topology_permuted"],
                }
                stage_report["cells"][cell] = {
                    contrast: interval(value, reconstructed["bootstrap"])
                    for contrast, value in contrasts.items()
                }
                for contrast, value in contrasts.items():
                    packed[f"{name}|{arm}|{stage}|{cell}|{contrast}"] = value
            arm_report["stages"][stage] = stage_report
        report["arms"][arm] = arm_report
    return report, packed


def classify_transport(mean_a: float, mean_b: float, tolerance: float) -> str:
    if abs(mean_a) <= tolerance or abs(mean_b) <= tolerance:
        return "IDENTITY_OR_NUMERICAL_ZERO"
    if mean_a < 0.0 and mean_b < 0.0:
        return "FAVORABLE_BOTH"
    if mean_a > 0.0 and mean_b > 0.0:
        return "ADVERSE_BOTH"
    return "SIGN_UNSTABLE"


def project_fixed_effect(row: dict[str, Any], n: int) -> dict[str, Any]:
    mean, upper = float(row["mean"]), float(row["ci95"][1])
    radius = upper - mean
    if mean >= 0.0:
        return {"status": "NO_FAVORABLE_FINITE_PROJECTION", "n_current": n, "n_projected": None, "upper_radius": radius}
    raw = int(math.ceil(n * (radius / abs(mean)) ** 2))
    projected = n if upper < 0.0 else max(n, raw)
    return {
        "status": "CURRENT_N_SUFFICIENT" if upper < 0.0 else "FIXED_EFFECT_PROJECTION",
        "n_current": n, "n_projected": projected, "upper_radius": radius,
    }


def build_transport(
    cfg: dict[str, Any], cohorts: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "status": "OPENED_TWO_COHORT_TRANSPORT_POWER_AUDIT",
        "primary_estimand": "deployed topology minus mean deployed topology-permuted",
        "projection_scope": "fixed observed effect and variance; diagnostic only",
        "arms": {},
    }
    counts = {label: 0 for label in (
        "FAVORABLE_BOTH", "ADVERSE_BOTH", "SIGN_UNSTABLE", "IDENTITY_OR_NUMERICAL_ZERO"
    )}
    for arm in cfg["arms"]:
        arm_report: dict[str, Any] = {"stages": {}}
        for stage in ("calibrated", "deployed"):
            stage_report: dict[str, Any] = {"cells": {}}
            for cell in CELLS:
                cell_report: dict[str, Any] = {}
                for contrast in CONTRASTS:
                    rows = {
                        cohort: cohorts[cohort]["arms"][arm]["stages"][stage]["cells"][cell][contrast]
                        for cohort in ("A", "B")
                    }
                    classification = classify_transport(
                        rows["A"]["mean"], rows["B"]["mean"], cfg["numerical_zero_tolerance"]
                    )
                    projections = {
                        cohort: project_fixed_effect(rows[cohort], cohorts[cohort]["n_masters"])
                        for cohort in ("A", "B")
                    }
                    common = None
                    if classification == "FAVORABLE_BOTH":
                        common = max(projections["A"]["n_projected"], projections["B"]["n_projected"])
                    expected: dict[str, Any] = {}
                    for cohort in ("A", "B"):
                        action = cohorts[cohort]["arms"][arm]["stages"][stage]["action"][cell]
                        multiplier = 2 if cell == "balanced" else 1
                        expected[cohort] = None if common is None else float(
                            common * multiplier * action["rate"]
                        )
                    cell_report[contrast] = {
                        "classification": classification, "cohorts": rows,
                        "fixed_effect_projection": projections,
                        "transport_aware_n_projected": common,
                        "historical_realizable_masters": cfg["historical_realizable_masters"],
                        "expected_acted_views_at_transport_n": expected,
                    }
                    if stage == "deployed" and contrast == "minus_topology_permuted":
                        counts[classification] += 1
                stage_report["cells"][cell] = cell_report
            arm_report["stages"][stage] = stage_report
        report["arms"][arm] = arm_report
    report["deployed_primary_transport_counts"] = counts
    return report


def prefix_pack(prefix: str, arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {f"{prefix}|{key}": value for key, value in arrays.items()}


def write_replay(output: Path, cfg: dict[str, Any], head: str) -> None:
    checks = "\n".join(
        f"printf '%s  %s\\n' '{sha(ROOT / relative)}' \"$repo/{relative}\" | sha256sum -c -"
        for relative in SOURCE_FILES
    )
    for key, root in roots(cfg).items():
        checks += f"\nprintf '%s  %s\\n' '{sha(root / 'manifest.json')}' \"$repo/{cfg[key]}/manifest.json\" | sha256sum -c -"
    replay = f'''#!/bin/sh
set -eu
repo=$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd)
[ "$(git -C "$repo" rev-parse HEAD)" = "{head}" ] || {{ echo "wrong git HEAD" >&2; exit 2; }}
[ -z "$(git -C "$repo" status --porcelain)" ] || {{ echo "dirty worktree" >&2; exit 2; }}
{checks}
: "${{OUTPUT_DIR:?set OUTPUT_DIR to a new path}}"
exec env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 "$repo/venv/bin/python" "$repo/experiments/geometria_proporcional/run_proportional_graph_selected_action_transport_power_audit.py" --config "$repo/experiments/geometria_proporcional/configs/proportional_graph_selected_action_transport_power_audit_v1.json" --output "$OUTPUT_DIR"
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
    verify_sources(cfg)
    output.mkdir(parents=True)
    cohort_a, cohort_b = reconstruct_a(cfg), reconstruct_b(cfg)
    reports: dict[str, dict[str, Any]] = {}
    effects: dict[str, np.ndarray] = {}
    for name, cohort in (("A", cohort_a), ("B", cohort_b)):
        reports[name], cohort_effects = summarize_cohort(cfg, name, cohort)
        effects.update(cohort_effects)
        reconstructed = {}
        for phase in ("calibration", "selection", "adjudication"):
            reconstructed.update(prefix_pack(phase, cohort[f"{phase}_pack"]))
        frozen.save_npz(output / f"cohort_{name.lower()}_reconstruction.npz", reconstructed)
        write_json(output / f"cohort_{name.lower()}_summary.json", reports[name])
    frozen.save_npz(output / "effects_by_master.npz", effects)
    transport = build_transport(cfg, reports)
    transport["cohort_b_exact_r364_reproduction"] = bool(cohort_b["exact_reproduction"])
    write_json(output / "analysis.json", transport)
    write_json(output / "environment.json", {
        "python": platform.python_version(), "numpy": importlib.metadata.version("numpy"),
        "scipy": importlib.metadata.version("scipy"),
        "scikit_learn": importlib.metadata.version("scikit-learn"),
        "threads": 1, "cuda_visible_devices": "", "refit": False,
        "new_views": 0, "new_solves": 0, "gpu_queried": False,
    })
    write_json(output / "resolved_config.json", cfg)
    write_replay(output, cfg, git_head())
    runtime = {
        "elapsed_seconds": time.monotonic() - started,
        "max_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
    }
    if runtime["elapsed_seconds"] > cfg["execution"]["max_seconds"] or runtime["max_rss_gib"] > cfg["execution"]["max_rss_gib"]:
        raise RuntimeError("transport audit resource budget exceeded")
    write_json(output / "runtime_observation.json", runtime)
    deterministic = sorted(
        path for path in output.rglob("*")
        if path.is_file() and path.name not in {"manifest.json", "runtime_observation.json"}
    )
    write_json(output / "manifest.json", {
        "schema_version": cfg["schema_version"], "git_head": git_head(),
        "source_manifests": {
            key: cfg[f"{key}_manifest_sha256"] for key in roots(cfg)
        },
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
