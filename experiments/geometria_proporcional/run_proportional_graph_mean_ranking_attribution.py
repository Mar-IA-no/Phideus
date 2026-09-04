#!/usr/bin/env python3
"""Attribute frozen mean-ranking signal across matched feature families on CPU."""

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
import run_proportional_graph_selected_action_transport_power_audit as transport  # noqa: E402
import run_proportional_graph_topology_localization_gate as topology_gate  # noqa: E402
import run_proportional_graph_two_stage_eligibility_diagnostic as two_stage  # noqa: E402

DEFAULT_CONFIG = ROOT / "experiments/geometria_proporcional/configs/proportional_graph_mean_ranking_attribution_v1.json"
SOURCE_FILES = (
    "experiments/geometria_proporcional/PLAN_PROPORTIONAL_MEAN_RANKING_ATTRIBUTION_CPU.md",
    "experiments/geometria_proporcional/configs/proportional_graph_mean_ranking_attribution_v1.json",
    "experiments/geometria_proporcional/run_proportional_graph_mean_ranking_attribution.py",
    "experiments/geometria_proporcional/run_proportional_graph_two_stage_eligibility_diagnostic.py",
    "experiments/geometria_proporcional/run_proportional_graph_mean_only_ranking_ablation.py",
    "experiments/geometria_proporcional/run_proportional_graph_equal_budget_ranking_diagnostic.py",
    "experiments/geometria_proporcional/run_proportional_graph_selected_action_transport_power_audit.py",
    "experiments/geometria_proporcional/run_proportional_graph_topology_localization_gate.py",
    "experiments/geometria_proporcional/run_proportional_graph_conditional_risk_gate.py",
    "experiments/geometria_proporcional/run_proportional_graph_fresh_mixed_gate.py",
    "experiments/geometria_proporcional/run_proportional_graph_frozen_adapters.py",
)
MAIN = ("reduced_mean", "public_base_mean", "topology_mean")
CONTROLS = ("topology_permuted_mean", "target_shuffled_topology_mean")
MODEL_FAMILY = {
    "reduced_mean": "correction_scale",
    "public_base_mean": "public_base",
    "topology_mean": "topology_augmented",
    "topology_permuted_mean": "topology_permuted",
    "target_shuffled_topology_mean": "target_shuffled_topology",
}
CONTRASTS = (
    "topology_minus_identity", "topology_minus_reduced", "topology_minus_public_base",
    "topology_minus_permuted", "topology_minus_target_shuffled",
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
        "schema_version", "source_two_stage_eligibility", "source_manifest_sha256",
        "source_topology_models", "source_topology_manifest_sha256", "arms", "alphas",
        "proposal_regimes", "fixed_alpha_index", "budget_fractions", "control_replicates",
        "selection_upper_percentile", "numerical_zero_tolerance", "execution",
    }
    if set(cfg) != expected or cfg["schema_version"] != "proportional-graph-mean-ranking-attribution-v1":
        raise ValueError("invalid mean-ranking attribution schema")
    if cfg["arms"] != ["raw_generic", "raw_typed", "closure_generic", "closure_typed"]:
        raise ValueError("arms changed")
    if cfg["alphas"] != [0.0, 0.25, 0.5, 0.75, 1.0]:
        raise ValueError("alpha grid changed")
    if cfg["proposal_regimes"] != ["fixed_alpha_0.25", "r368_common"] or cfg["fixed_alpha_index"] != 1:
        raise ValueError("proposal contract changed")
    if cfg["budget_fractions"] != [0.01, 0.02, 0.05, 0.10, 0.20, 0.40]:
        raise ValueError("budget grid changed")
    if cfg["control_replicates"] != 16 or cfg["selection_upper_percentile"] != 95.0:
        raise ValueError("control or firewall contract changed")
    if cfg["numerical_zero_tolerance"] != 1e-12 or cfg["execution"] != {"max_seconds": 300, "max_rss_gib": 4.0}:
        raise ValueError("execution contract changed")
    return cfg


def source_root(cfg: dict[str, Any]) -> Path:
    return ROOT / cfg["source_two_stage_eligibility"]


def model_root(cfg: dict[str, Any]) -> Path:
    return ROOT / cfg["source_topology_models"]


def verify_manifest(root: Path, expected_sha: str, label: str) -> dict[str, Any]:
    if sha(root / "manifest.json") != expected_sha:
        raise AssertionError(f"{label} manifest mismatch")
    manifest = json.loads((root / "manifest.json").read_text())
    for relative, expected in manifest["deterministic_files"].items():
        if sha(root / relative) != expected:
            raise AssertionError(f"{label} source mismatch: {relative}")
    return manifest


def verify_sources(cfg: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    verify_manifest(source_root(cfg), cfg["source_manifest_sha256"], "R368")
    verify_manifest(model_root(cfg), cfg["source_topology_manifest_sha256"], "R360")
    r368_cfg = two_stage.load_config(source_root(cfg) / "resolved_config.json")
    _, reconstruction_cfg = two_stage.verify_source(r368_cfg)
    models = json.loads((model_root(cfg) / "calibration/gate_models.json").read_text())
    if models["base_feature_order"] != list(topology_gate.BASE_FEATURE_ORDER):
        raise AssertionError("R360 base feature order changed")
    if models["topology_feature_order"] != list(topology_gate.TOPOLOGY_FEATURE_ORDER):
        raise AssertionError("R360 topology feature order changed")
    return r368_cfg, reconstruction_cfg, models


def require_clean(development: bool) -> None:
    if not development and subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout:
        raise RuntimeError("official mean-ranking attribution requires clean worktree")


def git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout.strip()


def load_cohort(
    cfg: dict[str, Any], r368_cfg: dict[str, Any], reconstruction_cfg: dict[str, Any], name: str,
) -> dict[str, Any]:
    cohort = two_stage.load_cohort(r368_cfg, reconstruction_cfg, name)
    cohort["topology"] = {
        role: read_npz(cohort["data_root"] / role / "topology_features.npz")
        for role in ("policy_selection", "adjudication")
    }
    if cfg["arms"] != r368_cfg["arms"] or cfg["alphas"] != r368_cfg["alphas"]:
        raise AssertionError("R368 factorial changed")
    return cohort


def predict_report(models: dict[str, Any], arm: str, family: str, x: np.ndarray, replicate: int | None) -> np.ndarray:
    if family in MAIN:
        report = models["arms"][arm]["families"][MODEL_FAMILY[family]]
    else:
        if replicate is None:
            raise ValueError("control prediction requires replicate")
        report = models["arms"][arm][MODEL_FAMILY[family]][replicate]["model"]
    return mixed.predict_report(report, x)


def mean_predictions(
    cfg: dict[str, Any], cohort: dict[str, Any], models: dict[str, Any], role: str,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], float]:
    state = cohort["states"][role]
    topology = cohort["topology"][role]
    n_arms, n_views, n_outputs = len(cfg["arms"]), state["delta"].shape[1], len(cfg["alphas"]) - 1
    main = {family: np.full((n_arms, n_views, n_outputs), np.nan) for family in MAIN}
    controls = {
        family: np.full((n_arms, cfg["control_replicates"], n_views, n_outputs), np.nan)
        for family in CONTROLS
    }
    for arm_index, arm in enumerate(cfg["arms"]):
        for family in MAIN:
            x = topology_gate.feature_matrix(state["data"], topology, arm_index, MODEL_FAMILY[family])
            main[family][arm_index] = predict_report(models, arm, family, x, None)
        for family in CONTROLS:
            for replicate in range(cfg["control_replicates"]):
                x = topology_gate.feature_matrix(
                    state["data"], topology, arm_index, MODEL_FAMILY[family], replicate
                )
                controls[family][arm_index, replicate] = predict_report(models, arm, family, x, replicate)
    if any(not np.all(np.isfinite(value)) for value in main.values()):
        raise RuntimeError("nonfinite main mean prediction")
    if any(not np.all(np.isfinite(value)) for value in controls.values()):
        raise RuntimeError("nonfinite control mean prediction")
    max_diff = float(np.max(np.abs(main["topology_mean"] - state["mu"])))
    if max_diff > cfg["numerical_zero_tolerance"]:
        raise AssertionError("R360 topology mean does not reproduce preserved mu")
    return main, controls, max_diff


def proposal_for(
    cfg: dict[str, Any], state: dict[str, Any], arm_index: int, proposal_regime: str,
) -> np.ndarray:
    if proposal_regime == "fixed_alpha_0.25":
        return np.full(state["delta"].shape[1], cfg["fixed_alpha_index"], dtype=np.int64)
    if proposal_regime == "r368_common":
        return two_stage.common_proposal(state, arm_index)
    raise ValueError(f"unknown proposal regime: {proposal_regime}")


def ranked_actions(
    cfg: dict[str, Any], cohort: dict[str, Any], role: str,
    main_predictions: dict[str, np.ndarray], control_predictions: dict[str, np.ndarray],
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[str, np.ndarray]], dict[str, np.ndarray]]:
    state = cohort["states"][role]
    n = state["delta"].shape[1]
    main = {
        regime: {family: np.zeros((len(cfg["arms"]), len(cfg["budget_fractions"]), n), dtype=np.int64) for family in MAIN}
        for regime in cfg["proposal_regimes"]
    }
    controls = {
        regime: {
            family: np.zeros((len(cfg["arms"]), len(cfg["budget_fractions"]), cfg["control_replicates"], n), dtype=np.int64)
            for family in CONTROLS
        }
        for regime in cfg["proposal_regimes"]
    }
    diagnostics: dict[str, np.ndarray] = {}
    for regime in cfg["proposal_regimes"]:
        for arm_index, arm in enumerate(cfg["arms"]):
            proposal = proposal_for(cfg, state, arm_index, regime)
            diagnostics[f"{regime}|{arm}|proposal"] = proposal
            scores = {
                family: ranking.selected_component(values[arm_index], proposal)
                for family, values in main_predictions.items()
            }
            control_scores = {
                family: np.stack([
                    ranking.selected_component(values[arm_index, replicate], proposal)
                    for replicate in range(cfg["control_replicates"])
                ])
                for family, values in control_predictions.items()
            }
            for family, score in scores.items():
                diagnostics[f"{regime}|{arm}|score|{family}"] = score
            for family, score in control_scores.items():
                diagnostics[f"{regime}|{arm}|score|{family}"] = score
            for budget_index, fraction in enumerate(cfg["budget_fractions"]):
                for family, score in scores.items():
                    main[regime][family][arm_index, budget_index] = ranking.equal_budget_action(
                        score, proposal, fraction
                    )
                for family, score in control_scores.items():
                    for replicate in range(cfg["control_replicates"]):
                        controls[regime][family][arm_index, budget_index, replicate] = ranking.equal_budget_action(
                            score[replicate], proposal, fraction
                        )
    return main, controls, diagnostics


def firewall(
    cfg: dict[str, Any], cohort: dict[str, Any], main: dict[str, dict[str, np.ndarray]],
    controls: dict[str, dict[str, np.ndarray]], diagnostics: dict[str, np.ndarray],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    state = cohort["states"]["policy_selection"]
    bootstrap = cohort["selection_bootstrap"]
    report: dict[str, Any] = {"proposals": {}}
    packed: dict[str, np.ndarray] = {"bootstrap_indices": bootstrap, **diagnostics}
    for regime in cfg["proposal_regimes"]:
        regime_report: dict[str, Any] = {"arms": {}}
        for arm_index, arm in enumerate(cfg["arms"]):
            quotient = state["data"]["quotient_rmse"][arm_index]
            arm_report: dict[str, Any] = {"budgets": {}}
            for budget_index, fraction in enumerate(cfg["budget_fractions"]):
                label = f"{fraction:.2f}"
                row: dict[str, Any] = {"policies": {}, "controls": {}}
                for family in MAIN:
                    action = main[regime][family][arm_index, budget_index]
                    row["policies"][family] = conditional.firewall_one(
                        conditional.selected_delta(quotient, action), bootstrap, cfg["selection_upper_percentile"]
                    )
                    packed[f"{regime}|{arm}|budget={label}|action|{family}"] = action
                for family in CONTROLS:
                    row["controls"][family] = []
                    for replicate in range(cfg["control_replicates"]):
                        action = controls[regime][family][arm_index, budget_index, replicate]
                        row["controls"][family].append({
                            "replicate": replicate,
                            **conditional.firewall_one(
                                conditional.selected_delta(quotient, action), bootstrap,
                                cfg["selection_upper_percentile"],
                            ),
                        })
                    packed[f"{regime}|{arm}|budget={label}|action|{family}"] = controls[regime][family][arm_index, budget_index]
                arm_report["budgets"][label] = row
            regime_report["arms"][arm] = arm_report
        report["proposals"][regime] = regime_report
    return report, packed


def deploy(
    cfg: dict[str, Any], main: dict[str, dict[str, np.ndarray]],
    controls: dict[str, dict[str, np.ndarray]], gate: dict[str, Any],
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[str, np.ndarray]]]:
    deployed_main = {regime: {family: value.copy() for family, value in rows.items()} for regime, rows in main.items()}
    deployed_controls = {regime: {family: value.copy() for family, value in rows.items()} for regime, rows in controls.items()}
    for regime in cfg["proposal_regimes"]:
        for arm_index, arm in enumerate(cfg["arms"]):
            for budget_index, fraction in enumerate(cfg["budget_fractions"]):
                row = gate["proposals"][regime]["arms"][arm]["budgets"][f"{fraction:.2f}"]
                for family in MAIN:
                    if not row["policies"][family]["deploy"]:
                        deployed_main[regime][family][arm_index, budget_index] = 0
                for family in CONTROLS:
                    for replicate in range(cfg["control_replicates"]):
                        if not row["controls"][family][replicate]["deploy"]:
                            deployed_controls[regime][family][arm_index, budget_index, replicate] = 0
    return deployed_main, deployed_controls


def describe(values: list[float | int]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {"mean": float(np.mean(array)), "min": float(np.min(array)), "max": float(np.max(array))}


def summarize(
    cfg: dict[str, Any], cohort: dict[str, Any], main: dict[str, dict[str, np.ndarray]],
    controls: dict[str, dict[str, np.ndarray]], diagnostics: dict[str, np.ndarray], gate: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, np.ndarray]]:
    deployed_main, deployed_controls = deploy(cfg, main, controls, gate)
    state = cohort["states"]["adjudication"]
    bootstrap = cohort["adjudication_bootstrap"]
    report: dict[str, Any] = {"cohort": cohort["name"], "n_masters": int(bootstrap.shape[1]), "proposals": {}}
    effects: dict[str, np.ndarray] = {"bootstrap_indices": bootstrap}
    packed: dict[str, np.ndarray] = {**diagnostics}
    for regime in cfg["proposal_regimes"]:
        regime_report: dict[str, Any] = {"arms": {}}
        for arm_index, arm in enumerate(cfg["arms"]):
            quotient = state["data"]["quotient_rmse"][arm_index]
            proposal = diagnostics[f"{regime}|{arm}|proposal"]
            arm_report: dict[str, Any] = {
                "proposal_alpha_fractions": {
                    str(alpha): float(np.mean(proposal == index))
                    for index, alpha in enumerate(cfg["alphas"]) if index > 0
                },
                "budgets": {}, "budget_average": {},
            }
            aggregate = {
                stage: {cell: {contrast: [] for contrast in CONTRASTS} for cell in CELLS}
                for stage in ("ranked", "deployed")
            }
            for budget_index, fraction in enumerate(cfg["budget_fractions"]):
                label = f"{fraction:.2f}"
                budget_report: dict[str, Any] = {
                    "action_budget": int(math.ceil(fraction * len(proposal))),
                    "firewall": gate["proposals"][regime]["arms"][arm]["budgets"][label],
                    "stages": {},
                }
                for stage, stage_main, stage_controls in (
                    ("ranked", main, controls), ("deployed", deployed_main, deployed_controls),
                ):
                    raw = {
                        family: mixed.selected_values(quotient, stage_main[regime][family][arm_index, budget_index])
                        for family in MAIN
                    }
                    for family in CONTROLS:
                        raw[family] = np.mean([
                            mixed.selected_values(
                                quotient, stage_controls[regime][family][arm_index, budget_index, replicate]
                            )
                            for replicate in range(cfg["control_replicates"])
                        ], axis=0)
                    raw["identity"] = quotient[:, 0]
                    topology_action = stage_main[regime]["topology_mean"][arm_index, budget_index]
                    stage_report: dict[str, Any] = {
                        "action_counts": {
                            family: int(np.count_nonzero(stage_main[regime][family][arm_index, budget_index]))
                            for family in MAIN
                        },
                        "overlap": {
                            family: ranking.jaccard(
                                topology_action, stage_main[regime][family][arm_index, budget_index]
                            )
                            for family in ("reduced_mean", "public_base_mean")
                        },
                        "cells": {},
                    }
                    for family in CONTROLS:
                        counts = []
                        overlaps = []
                        for replicate in range(cfg["control_replicates"]):
                            action = stage_controls[regime][family][arm_index, budget_index, replicate]
                            counts.append(int(np.count_nonzero(action)))
                            overlaps.append(ranking.jaccard(topology_action, action))
                        stage_report["action_counts"][family] = describe(counts)
                        stage_report["overlap"][family] = describe(overlaps)
                    for family in MAIN:
                        packed[f"{regime}|{arm}|budget={label}|{stage}|action|{family}"] = stage_main[regime][family][arm_index, budget_index]
                    for family in CONTROLS:
                        packed[f"{regime}|{arm}|budget={label}|{stage}|action|{family}"] = stage_controls[regime][family][arm_index, budget_index]
                    for cell in CELLS:
                        value = {key: ranking.policy_cell(array, cell) for key, array in raw.items()}
                        contrasts = {
                            "topology_minus_identity": value["topology_mean"] - value["identity"],
                            "topology_minus_reduced": value["topology_mean"] - value["reduced_mean"],
                            "topology_minus_public_base": value["topology_mean"] - value["public_base_mean"],
                            "topology_minus_permuted": value["topology_mean"] - value["topology_permuted_mean"],
                            "topology_minus_target_shuffled": value["topology_mean"] - value["target_shuffled_topology_mean"],
                        }
                        stage_report["cells"][cell] = {
                            contrast: ranking.interval(array, bootstrap) for contrast, array in contrasts.items()
                        }
                        for contrast, array in contrasts.items():
                            effects[f"{cohort['name']}|{regime}|{arm}|budget={label}|{stage}|{cell}|{contrast}"] = array
                            aggregate[stage][cell][contrast].append(array)
                    budget_report["stages"][stage] = stage_report
                arm_report["budgets"][label] = budget_report
            for stage in ("ranked", "deployed"):
                arm_report["budget_average"][stage] = {}
                for cell in CELLS:
                    arm_report["budget_average"][stage][cell] = {}
                    for contrast in CONTRASTS:
                        array = np.mean(aggregate[stage][cell][contrast], axis=0)
                        arm_report["budget_average"][stage][cell][contrast] = ranking.interval(array, bootstrap)
                        effects[f"{cohort['name']}|{regime}|{arm}|budget_average|{stage}|{cell}|{contrast}"] = array
            regime_report["arms"][arm] = arm_report
        report["proposals"][regime] = regime_report
    return report, effects, packed


def transport_analysis(cfg: dict[str, Any], reports: dict[str, Any]) -> dict[str, Any]:
    labels = ("FAVORABLE_BOTH", "ADVERSE_BOTH", "SIGN_UNSTABLE", "IDENTITY_OR_NUMERICAL_ZERO")
    result: dict[str, Any] = {
        "status": "OPENED_MEAN_RANKING_ATTRIBUTION_DIAGNOSTIC",
        "authority": "post hoc attribution; no architecture promotion, prospective safety, or GO/NO-GO",
        "primary_estimand": "fixed alpha 0.25 topology mean minus mean topology-permuted controls",
        "proposals": {},
    }
    primary_counts: dict[str, Any] = {}
    average_counts: dict[str, Any] = {}
    for regime in cfg["proposal_regimes"]:
        regime_result: dict[str, Any] = {"arms": {}}
        primary_counts[regime] = {stage: {label: 0 for label in labels} for stage in ("ranked", "deployed")}
        average_counts[regime] = {stage: {label: 0 for label in labels} for stage in ("ranked", "deployed")}
        for arm in cfg["arms"]:
            arm_result: dict[str, Any] = {"budgets": {}, "budget_average": {}}
            for fraction in cfg["budget_fractions"]:
                budget = f"{fraction:.2f}"
                arm_result["budgets"][budget] = {"stages": {}}
                for stage in ("ranked", "deployed"):
                    stage_result: dict[str, Any] = {"cells": {}}
                    for cell in CELLS:
                        stage_result["cells"][cell] = {}
                        for contrast in CONTRASTS:
                            rows = {
                                name: reports[name]["proposals"][regime]["arms"][arm]["budgets"][budget]["stages"][stage]["cells"][cell][contrast]
                                for name in ("A", "B")
                            }
                            classification = transport.classify_transport(
                                rows["A"]["mean"], rows["B"]["mean"], cfg["numerical_zero_tolerance"]
                            )
                            stage_result["cells"][cell][contrast] = {
                                "classification": classification, "cohorts": rows,
                            }
                            if contrast == "topology_minus_permuted":
                                primary_counts[regime][stage][classification] += 1
                    arm_result["budgets"][budget]["stages"][stage] = stage_result
            for stage in ("ranked", "deployed"):
                arm_result["budget_average"][stage] = {}
                for cell in CELLS:
                    arm_result["budget_average"][stage][cell] = {}
                    for contrast in CONTRASTS:
                        rows = {
                            name: reports[name]["proposals"][regime]["arms"][arm]["budget_average"][stage][cell][contrast]
                            for name in ("A", "B")
                        }
                        classification = transport.classify_transport(
                            rows["A"]["mean"], rows["B"]["mean"], cfg["numerical_zero_tolerance"]
                        )
                        arm_result["budget_average"][stage][cell][contrast] = {
                            "classification": classification, "cohorts": rows,
                        }
                        if contrast == "topology_minus_permuted":
                            average_counts[regime][stage][classification] += 1
            regime_result["arms"][arm] = arm_result
        result["proposals"][regime] = regime_result
    result["primary_transport_counts"] = primary_counts
    result["primary_budget_average_transport_counts"] = average_counts
    return result


def write_replay(output: Path, cfg: dict[str, Any], head: str) -> None:
    checks = "\n".join(
        f"printf '%s  %s\\n' '{sha(ROOT / relative)}' \"$repo/{relative}\" | sha256sum -c -"
        for relative in SOURCE_FILES
    )
    checks += f"\nprintf '%s  %s\\n' '{sha(source_root(cfg) / 'manifest.json')}' \"$repo/{cfg['source_two_stage_eligibility']}/manifest.json\" | sha256sum -c -"
    checks += f"\nprintf '%s  %s\\n' '{sha(model_root(cfg) / 'manifest.json')}' \"$repo/{cfg['source_topology_models']}/manifest.json\" | sha256sum -c -"
    replay = f'''#!/bin/sh
set -eu
repo=$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd)
[ "$(git -C "$repo" rev-parse HEAD)" = "{head}" ] || exit 2
[ -z "$(git -C "$repo" status --porcelain)" ] || exit 2
{checks}
: "${{OUTPUT_DIR:?set OUTPUT_DIR}}"
exec env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 "$repo/venv/bin/python" "$repo/experiments/geometria_proporcional/run_proportional_graph_mean_ranking_attribution.py" --config "$repo/experiments/geometria_proporcional/configs/proportional_graph_mean_ranking_attribution_v1.json" --output "$OUTPUT_DIR"
'''
    (output / "replay.sh").write_text(replay)
    (output / "replay.sh").chmod(0o755)


def run(cfg: dict[str, Any], output: Path, development: bool) -> None:
    require_clean(development)
    if output.exists():
        raise FileExistsError(output)
    if not development and (ROOT / "data/geometria_proporcional").resolve() not in output.resolve().parents:
        raise ValueError("official output path invalid")
    started = time.monotonic()
    r368_cfg, reconstruction_cfg, models = verify_sources(cfg)
    output.mkdir(parents=True)
    reports: dict[str, Any] = {}
    all_effects: dict[str, np.ndarray] = {}
    reproduction: dict[str, Any] = {}
    for name in ("A", "B"):
        cohort = load_cohort(cfg, r368_cfg, reconstruction_cfg, name)
        selection_main, selection_controls, selection_diff = mean_predictions(cfg, cohort, models, "policy_selection")
        adjud_main, adjud_controls, adjud_diff = mean_predictions(cfg, cohort, models, "adjudication")
        reproduction[name] = {"policy_selection_max_abs_diff": selection_diff, "adjudication_max_abs_diff": adjud_diff}
        prediction_pack = {
            **{f"policy_selection|{family}": value for family, value in selection_main.items()},
            **{f"policy_selection|{family}": value for family, value in selection_controls.items()},
            **{f"adjudication|{family}": value for family, value in adjud_main.items()},
            **{f"adjudication|{family}": value for family, value in adjud_controls.items()},
        }
        selection_actions, selection_control_actions, selection_diagnostics = ranked_actions(
            cfg, cohort, "policy_selection", selection_main, selection_controls
        )
        gate, gate_pack = firewall(
            cfg, cohort, selection_actions, selection_control_actions, selection_diagnostics
        )
        adjud_actions, adjud_control_actions, adjud_diagnostics = ranked_actions(
            cfg, cohort, "adjudication", adjud_main, adjud_controls
        )
        report, effects, adjud_pack = summarize(
            cfg, cohort, adjud_actions, adjud_control_actions, adjud_diagnostics, gate
        )
        reports[name] = report
        all_effects.update(effects)
        write_json(output / f"cohort_{name.lower()}_firewall.json", gate)
        frozen.save_npz(output / f"cohort_{name.lower()}_selection.npz", gate_pack)
        frozen.save_npz(output / f"cohort_{name.lower()}_predictions.npz", prediction_pack)
        write_json(output / f"cohort_{name.lower()}_summary.json", report)
        frozen.save_npz(output / f"cohort_{name.lower()}_adjudication.npz", adjud_pack)
    write_json(output / "model_reproduction.json", reproduction)
    frozen.save_npz(output / "effects_by_master.npz", all_effects)
    write_json(output / "analysis.json", transport_analysis(cfg, reports))
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
        raise RuntimeError("resource budget exceeded")
    write_json(output / "runtime_observation.json", runtime)
    deterministic = sorted(
        path for path in output.rglob("*")
        if path.is_file() and path.name not in {"manifest.json", "runtime_observation.json"}
    )
    write_json(output / "manifest.json", {
        "schema_version": cfg["schema_version"], "git_head": git_head(),
        "source_manifest_sha256": cfg["source_manifest_sha256"],
        "source_topology_manifest_sha256": cfg["source_topology_manifest_sha256"],
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
