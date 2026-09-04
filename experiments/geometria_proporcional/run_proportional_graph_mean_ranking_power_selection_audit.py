#!/usr/bin/env python3
"""Audit power and firewall nonlinearity for frozen mean-ranking effects on CPU."""

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
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments/geometria_proporcional")]

import run_proportional_graph_equal_budget_ranking_diagnostic as ranking  # noqa: E402
import run_proportional_graph_fresh_mixed_gate as mixed  # noqa: E402
import run_proportional_graph_frozen_adapters as frozen  # noqa: E402
import run_proportional_graph_mean_ranking_attribution as attribution  # noqa: E402
import run_proportional_graph_selected_action_transport_power_audit as transport  # noqa: E402

DEFAULT_CONFIG = ROOT / "experiments/geometria_proporcional/configs/proportional_graph_mean_ranking_power_selection_audit_v1.json"
SOURCE_FILES = (
    "experiments/geometria_proporcional/PLAN_PROPORTIONAL_MEAN_RANKING_POWER_SELECTION_AUDIT_CPU.md",
    "experiments/geometria_proporcional/configs/proportional_graph_mean_ranking_power_selection_audit_v1.json",
    "experiments/geometria_proporcional/run_proportional_graph_mean_ranking_power_selection_audit.py",
    "experiments/geometria_proporcional/run_proportional_graph_mean_ranking_attribution.py",
    "experiments/geometria_proporcional/run_proportional_graph_two_stage_eligibility_diagnostic.py",
    "experiments/geometria_proporcional/run_proportional_graph_equal_budget_ranking_diagnostic.py",
    "experiments/geometria_proporcional/run_proportional_graph_selected_action_transport_power_audit.py",
    "experiments/geometria_proporcional/run_proportional_graph_fresh_mixed_gate.py",
    "experiments/geometria_proporcional/run_proportional_graph_frozen_adapters.py",
)
EFFECTS = ("E00_ranked", "E10_topology_firewall", "E01_control_firewalls", "E11_deployed")
CONTRIBUTIONS = ("topology_firewall_contribution", "control_firewalls_contribution")
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
        "schema_version", "source_mean_ranking_attribution", "source_manifest_sha256", "arms",
        "proposal_regimes", "primary_proposal", "budget_fractions", "control_replicates",
        "control_bootstrap_seed", "control_bootstrap_replicates", "planning_references",
        "numerical_zero_tolerance", "execution",
    }
    if set(cfg) != expected or cfg["schema_version"] != "proportional-graph-mean-ranking-power-selection-audit-v1":
        raise ValueError("invalid mean-ranking power audit schema")
    if cfg["arms"] != ["raw_generic", "raw_typed", "closure_generic", "closure_typed"]:
        raise ValueError("arms changed")
    if cfg["proposal_regimes"] != ["fixed_alpha_0.25", "r368_common"] or cfg["primary_proposal"] != "fixed_alpha_0.25":
        raise ValueError("proposal contract changed")
    if cfg["budget_fractions"] != [0.01, 0.02, 0.05, 0.10, 0.20, 0.40]:
        raise ValueError("budget grid changed")
    if cfg["control_replicates"] != 16 or cfg["control_bootstrap_replicates"] != 2000:
        raise ValueError("control bootstrap changed")
    if cfg["control_bootstrap_seed"] != 2026091103 or cfg["planning_references"] != [300, 500, 1000, 2000]:
        raise ValueError("planning contract changed")
    if cfg["numerical_zero_tolerance"] != 1e-12 or cfg["execution"] != {"max_seconds": 300, "max_rss_gib": 4.0}:
        raise ValueError("execution contract changed")
    return cfg


def source_root(cfg: dict[str, Any]) -> Path:
    return ROOT / cfg["source_mean_ranking_attribution"]


def verify_source(cfg: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    root = source_root(cfg)
    if sha(root / "manifest.json") != cfg["source_manifest_sha256"]:
        raise AssertionError("R369 manifest mismatch")
    manifest = json.loads((root / "manifest.json").read_text())
    for relative, expected in manifest["deterministic_files"].items():
        if sha(root / relative) != expected:
            raise AssertionError(f"R369 source mismatch: {relative}")
    source_cfg = attribution.load_config(root / "resolved_config.json")
    r368_cfg, reconstruction_cfg, _ = attribution.verify_sources(source_cfg)
    if cfg["arms"] != source_cfg["arms"] or cfg["proposal_regimes"] != source_cfg["proposal_regimes"]:
        raise AssertionError("R369 design changed")
    return source_cfg, r368_cfg, reconstruction_cfg


def require_clean(development: bool) -> None:
    if not development and subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout:
        raise RuntimeError("official power audit requires clean worktree")


def git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout.strip()


def interval(values: np.ndarray, bootstrap: np.ndarray) -> dict[str, Any]:
    return ranking.interval(np.asarray(values), bootstrap)


def describe(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)), "std": float(np.std(array)),
        "min": float(np.min(array)), "median": float(np.median(array)), "max": float(np.max(array)),
    }


def bootstrap_interval(draws: np.ndarray, observed: float) -> dict[str, Any]:
    return {
        "mean": float(observed),
        "ci95": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
    }


def control_diagnostics(
    cube: np.ndarray, master_bootstrap: np.ndarray, control_bootstrap: np.ndarray,
) -> dict[str, Any]:
    replicate_means = np.mean(cube, axis=1)
    master_draws = np.mean(cube[:, master_bootstrap], axis=2).T
    rows = np.arange(len(control_bootstrap))[:, None]
    joint_draws = np.mean(master_draws[rows, control_bootstrap], axis=1)
    control_draws = np.mean(replicate_means[control_bootstrap], axis=1)
    return {
        "replicate_mean_effect": describe(replicate_means),
        "replicate_favorable_fraction": float(np.mean(replicate_means < 0.0)),
        "two_axis_bootstrap": bootstrap_interval(joint_draws, float(np.mean(cube))),
        "control_axis_only": bootstrap_interval(control_draws, float(np.mean(replicate_means))),
    }


def cell_values(raw: np.ndarray, cell: str) -> np.ndarray:
    return ranking.policy_cell(raw, cell)


def policy_raw(quotient: np.ndarray, action: np.ndarray) -> np.ndarray:
    return mixed.selected_values(quotient, action)


def audit_cohort(
    cfg: dict[str, Any], source_cfg: dict[str, Any], r368_cfg: dict[str, Any],
    reconstruction_cfg: dict[str, Any], name: str,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    root = source_root(cfg)
    cohort = attribution.load_cohort(source_cfg, r368_cfg, reconstruction_cfg, name)
    actions = read_npz(root / f"cohort_{name.lower()}_adjudication.npz")
    source_effects = read_npz(root / "effects_by_master.npz")
    source_gate = json.loads((root / f"cohort_{name.lower()}_firewall.json").read_text())
    bootstrap = cohort["adjudication_bootstrap"]
    control_bootstrap = np.random.default_rng(cfg["control_bootstrap_seed"]).integers(
        0, cfg["control_replicates"],
        size=(cfg["control_bootstrap_replicates"], cfg["control_replicates"]),
    )
    report: dict[str, Any] = {"cohort": name, "n_masters": int(bootstrap.shape[1]), "proposals": {}}
    packed: dict[str, np.ndarray] = {
        "master_bootstrap_indices": bootstrap, "control_bootstrap_indices": control_bootstrap,
    }
    for regime in cfg["proposal_regimes"]:
        regime_report: dict[str, Any] = {"arms": {}}
        for arm_index, arm in enumerate(cfg["arms"]):
            quotient = cohort["states"]["adjudication"]["data"]["quotient_rmse"][arm_index]
            arm_report: dict[str, Any] = {"budgets": {}, "budget_average": {}}
            aggregates = {
                cell: {key: [] for key in (*EFFECTS, *CONTRIBUTIONS, "replicate_cube")}
                for cell in CELLS
            }
            for fraction in cfg["budget_fractions"]:
                budget = f"{fraction:.2f}"
                prefix = f"{regime}|{arm}|budget={budget}"
                top_ranked = policy_raw(quotient, actions[f"{prefix}|ranked|action|topology_mean"])
                top_deployed = policy_raw(quotient, actions[f"{prefix}|deployed|action|topology_mean"])
                control_ranked = np.stack([
                    policy_raw(quotient, action)
                    for action in actions[f"{prefix}|ranked|action|topology_permuted_mean"]
                ])
                control_deployed = np.stack([
                    policy_raw(quotient, action)
                    for action in actions[f"{prefix}|deployed|action|topology_permuted_mean"]
                ])
                gate_row = source_gate["proposals"][regime]["arms"][arm]["budgets"][budget]
                budget_report: dict[str, Any] = {
                    "firewall": {
                        "topology_deploy": gate_row["policies"]["topology_mean"]["deploy"],
                        "control_deploy_count": int(sum(
                            row["deploy"] for row in gate_row["controls"]["topology_permuted_mean"]
                        )),
                        "control_deploy_fraction": float(np.mean([
                            row["deploy"] for row in gate_row["controls"]["topology_permuted_mean"]
                        ])),
                    },
                    "cells": {},
                }
                for cell in CELLS:
                    tr = cell_values(top_ranked, cell); td = cell_values(top_deployed, cell)
                    cr = np.stack([cell_values(raw, cell) for raw in control_ranked])
                    cd = np.stack([cell_values(raw, cell) for raw in control_deployed])
                    values = {
                        "E00_ranked": tr - np.mean(cr, axis=0),
                        "E10_topology_firewall": td - np.mean(cr, axis=0),
                        "E01_control_firewalls": tr - np.mean(cd, axis=0),
                        "E11_deployed": td - np.mean(cd, axis=0),
                    }
                    values["topology_firewall_contribution"] = values["E10_topology_firewall"] - values["E00_ranked"]
                    values["control_firewalls_contribution"] = values["E01_control_firewalls"] - values["E00_ranked"]
                    reconstructed = values["E00_ranked"] + values["topology_firewall_contribution"] + values["control_firewalls_contribution"]
                    if not np.allclose(reconstructed, values["E11_deployed"], atol=cfg["numerical_zero_tolerance"], rtol=0.0):
                        raise AssertionError("firewall decomposition is not additive")
                    source_key = f"{name}|{regime}|{arm}|budget={budget}|ranked|{cell}|topology_minus_permuted"
                    if not np.allclose(values["E00_ranked"], source_effects[source_key], atol=cfg["numerical_zero_tolerance"], rtol=0.0):
                        raise AssertionError("R369 ranked effect reproduction failed")
                    cube = tr[None, :] - cr
                    row: dict[str, Any] = {
                        "effects": {key: interval(value, bootstrap) for key, value in values.items()},
                    }
                    if regime == cfg["primary_proposal"]:
                        row["control_diagnostics"] = control_diagnostics(cube, bootstrap, control_bootstrap)
                    budget_report["cells"][cell] = row
                    for key, value in values.items():
                        packed[f"{prefix}|{cell}|{key}"] = value
                        aggregates[cell][key].append(value)
                    packed[f"{prefix}|{cell}|replicate_cube"] = cube
                    aggregates[cell]["replicate_cube"].append(cube)
                arm_report["budgets"][budget] = budget_report
            for cell in CELLS:
                averaged = {
                    key: np.mean(aggregates[cell][key], axis=0) for key in (*EFFECTS, *CONTRIBUTIONS)
                }
                cube = np.mean(aggregates[cell]["replicate_cube"], axis=0)
                row = {"effects": {key: interval(value, bootstrap) for key, value in averaged.items()}}
                if regime == cfg["primary_proposal"]:
                    row["control_diagnostics"] = control_diagnostics(cube, bootstrap, control_bootstrap)
                arm_report["budget_average"][cell] = row
                for key, value in averaged.items():
                    packed[f"{regime}|{arm}|budget_average|{cell}|{key}"] = value
                packed[f"{regime}|{arm}|budget_average|{cell}|replicate_cube"] = cube
            regime_report["arms"][arm] = arm_report
        report["proposals"][regime] = regime_report
    return report, packed


def classify(rows: dict[str, Any], tolerance: float) -> str:
    return transport.classify_transport(rows["A"]["mean"], rows["B"]["mean"], tolerance)


def build_analysis(cfg: dict[str, Any], reports: dict[str, Any]) -> dict[str, Any]:
    labels = ("FAVORABLE_BOTH", "ADVERSE_BOTH", "SIGN_UNSTABLE", "IDENTITY_OR_NUMERICAL_ZERO")
    result: dict[str, Any] = {
        "status": "OPENED_MEAN_RANKING_POWER_SELECTION_AUDIT",
        "authority": "fixed-effect planning and firewall diagnosis only; no promotion or GO/NO-GO",
        "primary_estimand": "fixed alpha ranked topology mean minus mean topology-permuted",
        "projection_scope": "fixed observed master effect and variance conditional on 16 frozen controls",
        "proposals": {},
    }
    projections: list[int] = []
    primary_favorable = 0
    joint_resolved = 0
    control_axis_resolved = 0
    reversals = 0
    for regime in cfg["proposal_regimes"]:
        counts = {effect: {label: 0 for label in labels} for effect in EFFECTS}
        average_counts = {effect: {label: 0 for label in labels} for effect in EFFECTS}
        regime_result: dict[str, Any] = {"arms": {}, "transport_counts": counts, "budget_average_transport_counts": average_counts}
        for arm in cfg["arms"]:
            arm_result: dict[str, Any] = {"budgets": {}, "budget_average": {}}
            for fraction in cfg["budget_fractions"]:
                budget = f"{fraction:.2f}"
                arm_result["budgets"][budget] = {"cells": {}}
                for cell in CELLS:
                    cell_result: dict[str, Any] = {"effects": {}}
                    for effect in EFFECTS:
                        rows = {
                            name: reports[name]["proposals"][regime]["arms"][arm]["budgets"][budget]["cells"][cell]["effects"][effect]
                            for name in ("A", "B")
                        }
                        classification = classify(rows, cfg["numerical_zero_tolerance"])
                        counts[effect][classification] += 1
                        cell_result["effects"][effect] = {"classification": classification, "cohorts": rows}
                    if cell_result["effects"]["E00_ranked"]["classification"] == "FAVORABLE_BOTH" and cell_result["effects"]["E11_deployed"]["classification"] != "FAVORABLE_BOTH":
                        reversals += regime == cfg["primary_proposal"]
                    if regime == cfg["primary_proposal"]:
                        ranked_rows = cell_result["effects"]["E00_ranked"]["cohorts"]
                        projection = {
                            name: transport.project_fixed_effect(ranked_rows[name], reports[name]["n_masters"])
                            for name in ("A", "B")
                        }
                        common = None
                        if cell_result["effects"]["E00_ranked"]["classification"] == "FAVORABLE_BOTH":
                            primary_favorable += 1
                            common = max(projection[name]["n_projected"] for name in ("A", "B"))
                            projections.append(common)
                        cell_result["fixed_effect_projection"] = {"cohorts": projection, "transport_aware_n_projected": common}
                        diagnostics = {
                            name: reports[name]["proposals"][regime]["arms"][arm]["budgets"][budget]["cells"][cell]["control_diagnostics"]
                            for name in ("A", "B")
                        }
                        cell_result["control_diagnostics"] = diagnostics
                        if all(diagnostics[name]["two_axis_bootstrap"]["ci95"][1] < 0.0 for name in ("A", "B")):
                            joint_resolved += 1
                        if all(diagnostics[name]["control_axis_only"]["ci95"][1] < 0.0 for name in ("A", "B")):
                            control_axis_resolved += 1
                    arm_result["budgets"][budget]["cells"][cell] = cell_result
            for cell in CELLS:
                cell_result = {"effects": {}}
                for effect in EFFECTS:
                    rows = {
                        name: reports[name]["proposals"][regime]["arms"][arm]["budget_average"][cell]["effects"][effect]
                        for name in ("A", "B")
                    }
                    classification = classify(rows, cfg["numerical_zero_tolerance"])
                    average_counts[effect][classification] += 1
                    cell_result["effects"][effect] = {"classification": classification, "cohorts": rows}
                if regime == cfg["primary_proposal"]:
                    cell_result["control_diagnostics"] = {
                        name: reports[name]["proposals"][regime]["arms"][arm]["budget_average"][cell]["control_diagnostics"]
                        for name in ("A", "B")
                    }
                arm_result["budget_average"][cell] = cell_result
            regime_result["arms"][arm] = arm_result
        result["proposals"][regime] = regime_result
    projection_summary: dict[str, Any] = {
        "n_favorable_transport_cells": primary_favorable,
        "n_two_axis_jointly_resolved_cells": joint_resolved,
        "n_control_axis_jointly_resolved_cells": control_axis_resolved,
        "n_ranked_favorable_lost_after_deployment": reversals,
        "planning_references": {
            str(reference): int(sum(value <= reference for value in projections))
            for reference in cfg["planning_references"]
        },
    }
    if projections:
        projection_summary["transport_aware_n_projected"] = describe(np.asarray(projections))
    result["primary_summary"] = projection_summary
    return result


def write_replay(output: Path, cfg: dict[str, Any], head: str) -> None:
    checks = "\n".join(
        f"printf '%s  %s\\n' '{sha(ROOT / relative)}' \"$repo/{relative}\" | sha256sum -c -"
        for relative in SOURCE_FILES
    )
    checks += f"\nprintf '%s  %s\\n' '{sha(source_root(cfg) / 'manifest.json')}' \"$repo/{cfg['source_mean_ranking_attribution']}/manifest.json\" | sha256sum -c -"
    replay = f'''#!/bin/sh
set -eu
repo=$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd)
[ "$(git -C "$repo" rev-parse HEAD)" = "{head}" ] || exit 2
[ -z "$(git -C "$repo" status --porcelain)" ] || exit 2
{checks}
: "${{OUTPUT_DIR:?set OUTPUT_DIR}}"
exec env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 "$repo/venv/bin/python" "$repo/experiments/geometria_proporcional/run_proportional_graph_mean_ranking_power_selection_audit.py" --config "$repo/experiments/geometria_proporcional/configs/proportional_graph_mean_ranking_power_selection_audit_v1.json" --output "$OUTPUT_DIR"
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
    source_cfg, r368_cfg, reconstruction_cfg = verify_source(cfg)
    output.mkdir(parents=True)
    reports: dict[str, Any] = {}
    for name in ("A", "B"):
        report, packed = audit_cohort(cfg, source_cfg, r368_cfg, reconstruction_cfg, name)
        reports[name] = report
        write_json(output / f"cohort_{name.lower()}_audit.json", report)
        frozen.save_npz(output / f"cohort_{name.lower()}_effects.npz", packed)
    write_json(output / "analysis.json", build_analysis(cfg, reports))
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
