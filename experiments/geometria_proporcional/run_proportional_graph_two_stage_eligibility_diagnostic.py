#!/usr/bin/env python3
"""Evaluate fixed mean ranking with tail-only eligibility on opened CPU cohorts."""

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
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments/geometria_proporcional")]

import run_proportional_graph_conditional_risk_gate as conditional  # noqa: E402
import run_proportional_graph_equal_budget_ranking_diagnostic as ranking  # noqa: E402
import run_proportional_graph_fresh_mixed_gate as mixed  # noqa: E402
import run_proportional_graph_frozen_adapters as frozen  # noqa: E402
import run_proportional_graph_mean_only_ranking_ablation as ablation  # noqa: E402
import run_proportional_graph_selected_action_calibration_diagnostic as selected  # noqa: E402
import run_proportional_graph_selected_action_transport_power_audit as transport  # noqa: E402
import run_proportional_graph_signed_tail_diagnostic as signed  # noqa: E402

DEFAULT_CONFIG = ROOT / "experiments/geometria_proporcional/configs/proportional_graph_two_stage_eligibility_diagnostic_v1.json"
SOURCE_FILES = (
    "experiments/geometria_proporcional/PLAN_PROPORTIONAL_TWO_STAGE_ELIGIBILITY_DIAGNOSTIC_CPU.md",
    "experiments/geometria_proporcional/configs/proportional_graph_two_stage_eligibility_diagnostic_v1.json",
    "experiments/geometria_proporcional/run_proportional_graph_two_stage_eligibility_diagnostic.py",
    "experiments/geometria_proporcional/run_proportional_graph_mean_only_ranking_ablation.py",
    "experiments/geometria_proporcional/run_proportional_graph_equal_budget_ranking_diagnostic.py",
    "experiments/geometria_proporcional/run_proportional_graph_selected_action_transport_power_audit.py",
    "experiments/geometria_proporcional/run_proportional_graph_selected_action_calibration_diagnostic.py",
    "experiments/geometria_proporcional/run_proportional_graph_signed_tail_diagnostic.py",
    "experiments/geometria_proporcional/run_proportional_graph_conditional_risk_gate.py",
    "experiments/geometria_proporcional/run_proportional_graph_fresh_mixed_gate.py",
    "experiments/geometria_proporcional/run_proportional_graph_frozen_adapters.py",
)
FILTERS = ("constant_filter", "public_base_filter", "topology_filter")
CONTROL = "topology_permuted_filter"
PREDICTOR = {
    "constant_filter": "constant_selected_action",
    "public_base_filter": "public_base_selected_action",
    "topology_filter": "topology_selected_action",
}
CONTRASTS = (
    "topology_minus_identity", "topology_minus_mean_only", "topology_minus_constant",
    "topology_minus_public_base", "topology_minus_permuted",
)
CELLS = ranking.CELLS


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
        "schema_version", "source_mean_only_ablation", "source_manifest_sha256", "arms",
        "alphas", "budget_fractions", "miscoverage", "control_replicates",
        "selection_upper_percentile", "numerical_zero_tolerance", "execution",
    }
    if set(cfg) != expected or cfg["schema_version"] != "proportional-graph-two-stage-eligibility-diagnostic-v1":
        raise ValueError("invalid two-stage eligibility schema")
    if cfg["arms"] != ["raw_generic", "raw_typed", "closure_generic", "closure_typed"]:
        raise ValueError("arms changed")
    if cfg["alphas"] != [0.0, 0.25, 0.5, 0.75, 1.0] or cfg["budget_fractions"] != [0.01, 0.02, 0.05, 0.10, 0.20, 0.40]:
        raise ValueError("proposal or budget grid changed")
    if cfg["miscoverage"] != 0.10 or cfg["control_replicates"] != 16 or cfg["selection_upper_percentile"] != 95.0:
        raise ValueError("calibration contract changed")
    if cfg["numerical_zero_tolerance"] != 1e-12 or cfg["execution"] != {"max_seconds": 300, "max_rss_gib": 4.0}:
        raise ValueError("execution contract changed")
    return cfg


def source_root(cfg: dict[str, Any]) -> Path:
    return ROOT / cfg["source_mean_only_ablation"]


def verify_source(cfg: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    root = source_root(cfg)
    if sha(root / "manifest.json") != cfg["source_manifest_sha256"]:
        raise AssertionError("R367 manifest mismatch")
    manifest = json.loads((root / "manifest.json").read_text())
    for relative, expected in manifest["deterministic_files"].items():
        if sha(root / relative) != expected:
            raise AssertionError(f"R367 source mismatch: {relative}")
    r367_cfg = ablation.load_config(root / "resolved_config.json")
    r366_cfg = ablation.verify_source(r367_cfg)
    r365_cfg = transport.load_config(ranking.source_root(r366_cfg) / "resolved_config.json")
    return r366_cfg, r365_cfg


def require_clean(development: bool) -> None:
    if not development and subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout:
        raise RuntimeError("official two-stage diagnostic requires clean worktree")


def git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout.strip()


def normalize_b_state(selected_cfg: dict[str, Any], role: str) -> dict[str, Any]:
    state = selected.role_state(selected_cfg, role)
    return {
        **state,
        "predicted": {family: state["predicted"][selected.BASE_NAME[family]] for family in selected.FAMILIES},
    }


def load_cohort(cfg: dict[str, Any], source_cfg: dict[str, Any], name: str) -> dict[str, Any]:
    source_roots = transport.roots(source_cfg)
    if name == "A":
        data_root, model_root = source_roots["cohort_a_data"], source_roots["cohort_a_models"]
        model_cfg = json.loads((model_root / "resolved_config.json").read_text())
        models = json.loads((model_root / "quantile_models.json").read_text())
        states = {
            role: transport.normalize_a_state(source_cfg, data_root, role, model_cfg, models)
            for role in ("risk_calibration", "policy_selection", "adjudication")
        }
        selection_bootstrap = signed.phase_arrays(data_root, "policy_selection")["bootstrap_indices"].astype(np.int64)
        adjudication_bootstrap = read_npz(data_root / "bootstrap_indices.npz")["indices"].astype(np.int64)
        simultaneous = json.loads((model_root / "calibration.json").read_text())
    else:
        data_root, selected_root = source_roots["cohort_b_data"], source_roots["cohort_b_selected"]
        selected_cfg = selected.load_config(selected_root / "resolved_config.json")
        states = {role: normalize_b_state(selected_cfg, role) for role in ("risk_calibration", "policy_selection", "adjudication")}
        selection_bootstrap = read_npz(data_root / "policy_selection/selection_statistics.npz")["bootstrap_indices"].astype(np.int64)
        adjudication_bootstrap = read_npz(data_root / "bootstrap_indices.npz")["indices"].astype(np.int64)
        simultaneous = json.loads((data_root / "risk_calibration/quantiles.json").read_text())
    return {
        "name": name, "data_root": data_root, "states": states,
        "selection_bootstrap": selection_bootstrap, "adjudication_bootstrap": adjudication_bootstrap,
        "simultaneous": simultaneous,
    }


def simultaneous_q(cohort: dict[str, Any], arm: str, family: str, replicate: int | None = None) -> float:
    source = cohort["simultaneous"]
    if cohort["name"] == "A":
        if replicate is None:
            return float(source["arms"][arm]["families"][transport.SIGNED_NAME[PREDICTOR[family]]]["q"])
        return float(source["arms"][arm][signed.CONTROL_FAMILY][replicate]["q"])
    base = source["interfaces"]["signed_tail"]["arms"][arm]
    if replicate is None:
        return float(base["families"][selected.BASE_NAME[PREDICTOR[family]]]["q"])
    return float(base["topology_permuted"][replicate]["q"])


def common_proposal(state: dict[str, Any], arm_index: int) -> np.ndarray:
    proposal, _ = selected.propose(
        state["mu"][arm_index], state["predicted"]["public_base_selected_action"][arm_index]
    )
    return proposal


def calibrate(cfg: dict[str, Any], cohort: dict[str, Any]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    state = cohort["states"]["risk_calibration"]
    iid = np.arange(0, state["delta"].shape[1], 2)
    residual = state["delta"] - state["mu"]
    report: dict[str, Any] = {"arms": {}}
    packed: dict[str, np.ndarray] = {"iid_index": iid}
    for arm_index, arm in enumerate(cfg["arms"]):
        proposal = common_proposal(state, arm_index)
        record: dict[str, Any] = {"filters": {}, CONTROL: []}
        packed[f"{arm}|proposal"] = proposal
        for family in FILTERS:
            prediction = state["predicted"][PREDICTOR[family]][arm_index]
            score = selected.selected_score(residual[arm_index], prediction, proposal)
            q, rank = conditional.conformal_quantile(score[iid], cfg["miscoverage"])
            q_sim = simultaneous_q(cohort, arm, family)
            if q > q_sim + cfg["numerical_zero_tolerance"]:
                raise AssertionError("selected filter quantile exceeds simultaneous quantile")
            record["filters"][family] = {"q": q, "q_simultaneous": q_sim, "rank": rank, "n": len(iid)}
            packed[f"{arm}|{family}|score"] = score[iid]
        for replicate in range(cfg["control_replicates"]):
            prediction = state["control"][arm_index, replicate]
            score = selected.selected_score(residual[arm_index], prediction, proposal)
            q, rank = conditional.conformal_quantile(score[iid], cfg["miscoverage"])
            q_sim = simultaneous_q(cohort, arm, CONTROL, replicate)
            if q > q_sim + cfg["numerical_zero_tolerance"]:
                raise AssertionError("selected control quantile exceeds simultaneous quantile")
            record[CONTROL].append({"replicate": replicate, "q": q, "q_simultaneous": q_sim, "rank": rank, "n": len(iid)})
            packed[f"{arm}|{CONTROL}={replicate}|score"] = score[iid]
        report["arms"][arm] = record
    return report, packed


def filtered_action(order: np.ndarray, proposal: np.ndarray, eligible: np.ndarray, budget: int) -> np.ndarray:
    chosen = order[np.asarray(eligible, dtype=bool)[order]][:budget]
    action = np.zeros(len(order), dtype=np.int64)
    action[chosen] = proposal[chosen]
    return action


def actions(cfg: dict[str, Any], cohort: dict[str, Any], role: str, calibration: dict[str, Any]) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, np.ndarray]]:
    state = cohort["states"][role]
    n = state["delta"].shape[1]
    main = {family: np.zeros((len(cfg["arms"]), len(cfg["budget_fractions"]), n), dtype=np.int64) for family in ("mean_only", *FILTERS)}
    controls = np.zeros((len(cfg["arms"]), len(cfg["budget_fractions"]), cfg["control_replicates"], n), dtype=np.int64)
    diagnostics: dict[str, np.ndarray] = {}
    for arm_index, arm in enumerate(cfg["arms"]):
        proposal = common_proposal(state, arm_index)
        rank_score = ranking.selected_component(state["mu"][arm_index], proposal)
        order = np.lexsort((np.arange(n, dtype=np.int64), rank_score))
        diagnostics[f"{arm}|proposal"] = proposal
        diagnostics[f"{arm}|rank_score"] = rank_score
        for family in FILTERS:
            prediction = state["predicted"][PREDICTOR[family]][arm_index]
            bound = rank_score + ranking.selected_component(prediction, proposal) + calibration["arms"][arm]["filters"][family]["q"]
            diagnostics[f"{arm}|eligible|{family}"] = bound < 0.0
            diagnostics[f"{arm}|bound|{family}"] = bound
        for replicate in range(cfg["control_replicates"]):
            prediction = state["control"][arm_index, replicate]
            bound = rank_score + ranking.selected_component(prediction, proposal) + calibration["arms"][arm][CONTROL][replicate]["q"]
            diagnostics[f"{arm}|eligible|{CONTROL}={replicate}"] = bound < 0.0
            diagnostics[f"{arm}|bound|{CONTROL}={replicate}"] = bound
        for budget_index, fraction in enumerate(cfg["budget_fractions"]):
            budget = int(math.ceil(fraction * n))
            main["mean_only"][arm_index, budget_index] = filtered_action(order, proposal, np.ones(n, dtype=bool), budget)
            for family in FILTERS:
                main[family][arm_index, budget_index] = filtered_action(order, proposal, diagnostics[f"{arm}|eligible|{family}"], budget)
            for replicate in range(cfg["control_replicates"]):
                controls[arm_index, budget_index, replicate] = filtered_action(order, proposal, diagnostics[f"{arm}|eligible|{CONTROL}={replicate}"], budget)
    return main, controls, diagnostics


def describe(values: list[float | int]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {"mean": float(np.mean(array)), "min": float(np.min(array)), "max": float(np.max(array))}


def describe_optional(values: list[float | None]) -> dict[str, float | int | None]:
    finite = np.asarray([value for value in values if value is not None], dtype=np.float64)
    if not len(finite):
        return {"mean": None, "min": None, "max": None, "n_defined": 0}
    return {
        "mean": float(np.mean(finite)), "min": float(np.min(finite)), "max": float(np.max(finite)),
        "n_defined": int(len(finite)),
    }


def acted_harm(quotient: np.ndarray, action: np.ndarray, indices: np.ndarray) -> dict[str, Any]:
    realized = conditional.selected_delta(quotient, action)
    acted = action[indices] > 0
    selected_delta = realized[indices][acted]
    return {
        "n": int(np.sum(acted)),
        "mean_delta": float(np.mean(selected_delta)) if len(selected_delta) else None,
        "positive_fraction": float(np.mean(selected_delta > 0.0)) if len(selected_delta) else None,
    }


def coverage_diagnostics(
    cfg: dict[str, Any], cohort: dict[str, Any], calibration: dict[str, Any], arm_index: int, arm: str,
) -> dict[str, Any]:
    state = cohort["states"]["adjudication"]
    proposal = common_proposal(state, arm_index)
    residual = ranking.selected_component(state["delta"][arm_index] - state["mu"][arm_index], proposal)
    iid = np.arange(0, len(proposal), 2)
    grouped = np.arange(1, len(proposal), 2)
    result: dict[str, Any] = {
        "jurisdiction": "marginal selected-action coverage; IID calibrated, grouped diagnostic only",
        "filters": {}, CONTROL: {"replicates": [], "aggregate": {}},
    }
    for family in FILTERS:
        predicted = ranking.selected_component(state["predicted"][PREDICTOR[family]][arm_index], proposal)
        covered = residual <= predicted + calibration["arms"][arm]["filters"][family]["q"]
        result["filters"][family] = {
            "iid": float(np.mean(covered[iid])), "grouped": float(np.mean(covered[grouped])),
        }
    for replicate in range(cfg["control_replicates"]):
        predicted = ranking.selected_component(state["control"][arm_index, replicate], proposal)
        covered = residual <= predicted + calibration["arms"][arm][CONTROL][replicate]["q"]
        result[CONTROL]["replicates"].append({
            "replicate": replicate,
            "iid": float(np.mean(covered[iid])), "grouped": float(np.mean(covered[grouped])),
        })
    for cell in ("iid", "grouped"):
        result[CONTROL]["aggregate"][cell] = describe([
            row[cell] for row in result[CONTROL]["replicates"]
        ])
    return result


def firewall(cfg: dict[str, Any], cohort: dict[str, Any], calibration: dict[str, Any]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    main, controls, diagnostics = actions(cfg, cohort, "policy_selection", calibration)
    state = cohort["states"]["policy_selection"]
    bootstrap = cohort["selection_bootstrap"]
    report: dict[str, Any] = {"arms": {}}
    packed = {"bootstrap_indices": bootstrap, **diagnostics}
    for arm_index, arm in enumerate(cfg["arms"]):
        quotient = state["data"]["quotient_rmse"][arm_index]
        arm_record: dict[str, Any] = {"budgets": {}}
        for budget_index, fraction in enumerate(cfg["budget_fractions"]):
            label = f"{fraction:.2f}"
            row: dict[str, Any] = {"policies": {}, CONTROL: []}
            for family, values in main.items():
                action = values[arm_index, budget_index]
                row["policies"][family] = conditional.firewall_one(conditional.selected_delta(quotient, action), bootstrap, cfg["selection_upper_percentile"])
                packed[f"{arm}|budget={label}|action|{family}"] = action
            for replicate in range(cfg["control_replicates"]):
                action = controls[arm_index, budget_index, replicate]
                row[CONTROL].append({"replicate": replicate, **conditional.firewall_one(conditional.selected_delta(quotient, action), bootstrap, cfg["selection_upper_percentile"])})
                packed[f"{arm}|budget={label}|action|{CONTROL}={replicate}"] = action
            arm_record["budgets"][label] = row
        report["arms"][arm] = arm_record
    return report, packed


def deploy(cfg: dict[str, Any], main: dict[str, np.ndarray], controls: np.ndarray, gate: dict[str, Any]) -> tuple[dict[str, np.ndarray], np.ndarray]:
    deployed = {key: value.copy() for key, value in main.items()}
    deployed_controls = controls.copy()
    for arm_index, arm in enumerate(cfg["arms"]):
        for budget_index, fraction in enumerate(cfg["budget_fractions"]):
            row = gate["arms"][arm]["budgets"][f"{fraction:.2f}"]
            for family in deployed:
                if not row["policies"][family]["deploy"]:
                    deployed[family][arm_index, budget_index] = 0
            for replicate in range(cfg["control_replicates"]):
                if not row[CONTROL][replicate]["deploy"]:
                    deployed_controls[arm_index, budget_index, replicate] = 0
    return deployed, deployed_controls


def summarize(cfg: dict[str, Any], cohort: dict[str, Any], calibration: dict[str, Any], gate: dict[str, Any]) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, np.ndarray]]:
    main, controls, diagnostics = actions(cfg, cohort, "adjudication", calibration)
    deployed, deployed_controls = deploy(cfg, main, controls, gate)
    state = cohort["states"]["adjudication"]
    bootstrap = cohort["adjudication_bootstrap"]
    report: dict[str, Any] = {"cohort": cohort["name"], "n_masters": int(bootstrap.shape[1]), "arms": {}}
    effects: dict[str, np.ndarray] = {"bootstrap_indices": bootstrap}
    packed = {**diagnostics}
    for arm_index, arm in enumerate(cfg["arms"]):
        quotient = state["data"]["quotient_rmse"][arm_index]
        n = state["delta"].shape[1]
        iid = np.arange(0, n, 2); grouped = np.arange(1, n, 2)
        proposal = diagnostics[f"{arm}|proposal"]
        arm_report: dict[str, Any] = {
            "calibration": calibration["arms"][arm],
            "coverage": coverage_diagnostics(cfg, cohort, calibration, arm_index, arm),
            "proposal_alpha_fractions": {
                cell: {
                    str(alpha): float(np.mean(proposal[indices] == alpha_index))
                    for alpha_index, alpha in enumerate(cfg["alphas"]) if alpha_index > 0
                }
                for cell, indices in (("iid", iid), ("grouped", grouped))
            },
            "eligible_counts": {
                "filters": {
                    family: {
                        "total": int(np.sum(diagnostics[f"{arm}|eligible|{family}"])),
                        "iid": int(np.sum(diagnostics[f"{arm}|eligible|{family}"][iid])),
                        "grouped": int(np.sum(diagnostics[f"{arm}|eligible|{family}"][grouped])),
                    }
                    for family in FILTERS
                },
                CONTROL: {
                    cell: describe([
                        int(np.sum(diagnostics[f"{arm}|eligible|{CONTROL}={replicate}"][indices]))
                        for replicate in range(cfg["control_replicates"])
                    ])
                    for cell, indices in (("total", np.arange(n)), ("iid", iid), ("grouped", grouped))
                },
            },
            "budgets": {}, "budget_average": {},
        }
        aggregate = {
            stage: {cell: {contrast: [] for contrast in CONTRASTS} for cell in CELLS}
            for stage in ("calibrated", "deployed")
        }
        for budget_index, fraction in enumerate(cfg["budget_fractions"]):
            label = f"{fraction:.2f}"
            budget_report: dict[str, Any] = {
                "requested_action_budget": int(math.ceil(fraction * n)),
                "firewall": gate["arms"][arm]["budgets"][label], "stages": {},
            }
            for stage, stage_main, stage_controls in (("calibrated", main, controls), ("deployed", deployed, deployed_controls)):
                raw = {key: mixed.selected_values(quotient, values[arm_index, budget_index]) for key, values in stage_main.items()}
                raw[CONTROL] = np.mean([mixed.selected_values(quotient, stage_controls[arm_index, budget_index, replicate]) for replicate in range(cfg["control_replicates"])], axis=0)
                raw["identity"] = quotient[:, 0]
                stage_report: dict[str, Any] = {"action_counts": {}, "acted_harm": {}, "cells": {}}
                for family, values in stage_main.items():
                    action = values[arm_index, budget_index]
                    stage_report["action_counts"][family] = {"total": int(np.count_nonzero(action)), "iid": int(np.count_nonzero(action[iid])), "grouped": int(np.count_nonzero(action[grouped]))}
                    stage_report["acted_harm"][family] = {
                        "iid": acted_harm(quotient, action, iid),
                        "grouped": acted_harm(quotient, action, grouped),
                    }
                    packed[f"{arm}|budget={label}|{stage}|action|{family}"] = action
                control_harm: list[dict[str, Any]] = []
                control_counts = {cell: [] for cell in ("total", "iid", "grouped")}
                for replicate in range(cfg["control_replicates"]):
                    control_action = stage_controls[arm_index, budget_index, replicate]
                    control_counts["total"].append(int(np.count_nonzero(control_action)))
                    control_counts["iid"].append(int(np.count_nonzero(control_action[iid])))
                    control_counts["grouped"].append(int(np.count_nonzero(control_action[grouped])))
                    control_harm.append({
                        "replicate": replicate,
                        "iid": acted_harm(quotient, control_action, iid),
                        "grouped": acted_harm(quotient, control_action, grouped),
                    })
                stage_report["action_counts"][CONTROL] = {
                    cell: describe(values) for cell, values in control_counts.items()
                }
                stage_report["acted_harm"][CONTROL] = {
                    "replicates": control_harm,
                    "aggregate": {
                        cell: {
                            "n": describe([row[cell]["n"] for row in control_harm]),
                            "mean_delta": describe_optional([row[cell]["mean_delta"] for row in control_harm]),
                            "positive_fraction": describe_optional([row[cell]["positive_fraction"] for row in control_harm]),
                        }
                        for cell in ("iid", "grouped")
                    },
                }
                packed[f"{arm}|budget={label}|{stage}|action|{CONTROL}"] = stage_controls[arm_index, budget_index]
                for cell in CELLS:
                    value = {key: ranking.policy_cell(array, cell) for key, array in raw.items()}
                    contrasts = {
                        "topology_minus_identity": value["topology_filter"] - value["identity"],
                        "topology_minus_mean_only": value["topology_filter"] - value["mean_only"],
                        "topology_minus_constant": value["topology_filter"] - value["constant_filter"],
                        "topology_minus_public_base": value["topology_filter"] - value["public_base_filter"],
                        "topology_minus_permuted": value["topology_filter"] - value[CONTROL],
                    }
                    stage_report["cells"][cell] = {key: ranking.interval(array, bootstrap) for key, array in contrasts.items()}
                    for key, array in contrasts.items():
                        effects[f"{cohort['name']}|{arm}|budget={label}|{stage}|{cell}|{key}"] = array
                        aggregate[stage][cell][key].append(array)
                budget_report["stages"][stage] = stage_report
            arm_report["budgets"][label] = budget_report
        for stage in ("calibrated", "deployed"):
            arm_report["budget_average"][stage] = {}
            for cell in CELLS:
                arm_report["budget_average"][stage][cell] = {}
                for contrast in CONTRASTS:
                    value = np.mean(aggregate[stage][cell][contrast], axis=0)
                    arm_report["budget_average"][stage][cell][contrast] = ranking.interval(value, bootstrap)
                    effects[f"{cohort['name']}|{arm}|budget_average|{stage}|{cell}|{contrast}"] = value
        report["arms"][arm] = arm_report
    return report, effects, packed


def transport_analysis(cfg: dict[str, Any], reports: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": "OPENED_TWO_STAGE_ELIGIBILITY_DIAGNOSTIC",
        "authority": "post hoc diagnostic; no prospective no-harm guarantee or architecture promotion",
        "primary_estimand": "topology filter minus mean permuted filter", "arms": {},
    }
    counts = {stage: {label: 0 for label in ("FAVORABLE_BOTH", "ADVERSE_BOTH", "SIGN_UNSTABLE", "IDENTITY_OR_NUMERICAL_ZERO")} for stage in ("calibrated", "deployed")}
    average_counts = {stage: {label: 0 for label in ("FAVORABLE_BOTH", "ADVERSE_BOTH", "SIGN_UNSTABLE", "IDENTITY_OR_NUMERICAL_ZERO")} for stage in ("calibrated", "deployed")}
    for arm in cfg["arms"]:
        arm_result: dict[str, Any] = {"budgets": {}, "budget_average": {}}
        for fraction in cfg["budget_fractions"]:
            label = f"{fraction:.2f}"; arm_result["budgets"][label] = {"stages": {}}
            for stage in ("calibrated", "deployed"):
                stage_result: dict[str, Any] = {"cells": {}}
                for cell in CELLS:
                    stage_result["cells"][cell] = {}
                    for contrast in CONTRASTS:
                        rows = {name: reports[name]["arms"][arm]["budgets"][label]["stages"][stage]["cells"][cell][contrast] for name in ("A", "B")}
                        classification = transport.classify_transport(rows["A"]["mean"], rows["B"]["mean"], cfg["numerical_zero_tolerance"])
                        stage_result["cells"][cell][contrast] = {"classification": classification, "cohorts": rows}
                        if contrast == "topology_minus_permuted": counts[stage][classification] += 1
                arm_result["budgets"][label]["stages"][stage] = stage_result
        for stage in ("calibrated", "deployed"):
            arm_result["budget_average"][stage] = {}
            for cell in CELLS:
                arm_result["budget_average"][stage][cell] = {}
                for contrast in CONTRASTS:
                    rows = {
                        name: reports[name]["arms"][arm]["budget_average"][stage][cell][contrast]
                        for name in ("A", "B")
                    }
                    classification = transport.classify_transport(
                        rows["A"]["mean"], rows["B"]["mean"], cfg["numerical_zero_tolerance"]
                    )
                    arm_result["budget_average"][stage][cell][contrast] = {
                        "classification": classification, "cohorts": rows,
                    }
                    if contrast == "topology_minus_permuted":
                        average_counts[stage][classification] += 1
        result["arms"][arm] = arm_result
    result["primary_transport_counts"] = counts
    result["primary_budget_average_transport_counts"] = average_counts
    return result


def write_replay(output: Path, cfg: dict[str, Any], head: str) -> None:
    checks = "\n".join(f"printf '%s  %s\\n' '{sha(ROOT / relative)}' \"$repo/{relative}\" | sha256sum -c -" for relative in SOURCE_FILES)
    checks += f"\nprintf '%s  %s\\n' '{sha(source_root(cfg) / 'manifest.json')}' \"$repo/{cfg['source_mean_only_ablation']}/manifest.json\" | sha256sum -c -"
    replay = f'''#!/bin/sh
set -eu
repo=$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd)
[ "$(git -C "$repo" rev-parse HEAD)" = "{head}" ] || exit 2
[ -z "$(git -C "$repo" status --porcelain)" ] || exit 2
{checks}
: "${{OUTPUT_DIR:?set OUTPUT_DIR}}"
exec env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 "$repo/venv/bin/python" "$repo/experiments/geometria_proporcional/run_proportional_graph_two_stage_eligibility_diagnostic.py" --config "$repo/experiments/geometria_proporcional/configs/proportional_graph_two_stage_eligibility_diagnostic_v1.json" --output "$OUTPUT_DIR"
'''
    (output / "replay.sh").write_text(replay); (output / "replay.sh").chmod(0o755)


def run(cfg: dict[str, Any], output: Path, development: bool) -> None:
    require_clean(development)
    if output.exists(): raise FileExistsError(output)
    if not development and (ROOT / "data/geometria_proporcional").resolve() not in output.resolve().parents: raise ValueError("official output path invalid")
    started = time.monotonic(); _, source_cfg = verify_source(cfg); output.mkdir(parents=True)
    cohorts = {name: load_cohort(cfg, source_cfg, name) for name in ("A", "B")}
    reports: dict[str, Any] = {}; all_effects: dict[str, np.ndarray] = {}
    for name, cohort in cohorts.items():
        calibration, calibration_pack = calibrate(cfg, cohort); write_json(output / f"cohort_{name.lower()}_calibration.json", calibration); frozen.save_npz(output / f"cohort_{name.lower()}_calibration.npz", calibration_pack)
        gate, gate_pack = firewall(cfg, cohort, calibration); write_json(output / f"cohort_{name.lower()}_firewall.json", gate); frozen.save_npz(output / f"cohort_{name.lower()}_selection.npz", gate_pack)
        report, effects, adjudication_pack = summarize(cfg, cohort, calibration, gate); reports[name] = report; all_effects.update(effects); write_json(output / f"cohort_{name.lower()}_summary.json", report); frozen.save_npz(output / f"cohort_{name.lower()}_adjudication.npz", adjudication_pack)
    frozen.save_npz(output / "effects_by_master.npz", all_effects); write_json(output / "analysis.json", transport_analysis(cfg, reports))
    write_json(output / "environment.json", {"python": platform.python_version(), "numpy": importlib.metadata.version("numpy"), "scipy": importlib.metadata.version("scipy"), "scikit_learn": importlib.metadata.version("scikit-learn"), "threads": 1, "cuda_visible_devices": "", "refit": False, "new_views": 0, "new_solves": 0, "gpu_queried": False})
    write_json(output / "resolved_config.json", cfg); write_replay(output, cfg, git_head())
    runtime = {"elapsed_seconds": time.monotonic() - started, "max_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2}
    if runtime["elapsed_seconds"] > cfg["execution"]["max_seconds"] or runtime["max_rss_gib"] > cfg["execution"]["max_rss_gib"]: raise RuntimeError("resource budget exceeded")
    write_json(output / "runtime_observation.json", runtime)
    deterministic = sorted(path for path in output.rglob("*") if path.is_file() and path.name not in {"manifest.json", "runtime_observation.json"})
    write_json(output / "manifest.json", {"schema_version": cfg["schema_version"], "git_head": git_head(), "source_manifest_sha256": cfg["source_manifest_sha256"], "source_hashes": {relative: sha(ROOT / relative) for relative in SOURCE_FILES}, "deterministic_files": {str(path.relative_to(output)): sha(path) for path in deterministic}, "runtime_exclusions": ["runtime_observation.json"]})
    print(json.dumps(runtime, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG); parser.add_argument("--output", type=Path, required=True); parser.add_argument("--development", action="store_true"); args = parser.parse_args(); run(load_config(args.config.resolve()), args.output.resolve(), args.development)


if __name__ == "__main__": main()
