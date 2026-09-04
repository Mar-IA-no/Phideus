#!/usr/bin/env python3
"""Diagnose topology ranking under a shared alpha proposal and equal CPU action budgets."""

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

import run_proportional_graph_fresh_mixed_gate as mixed  # noqa: E402
import run_proportional_graph_frozen_adapters as frozen  # noqa: E402
import run_proportional_graph_selected_action_calibration_diagnostic as selected  # noqa: E402
import run_proportional_graph_selected_action_transport_power_audit as transport  # noqa: E402

DEFAULT_CONFIG = ROOT / "experiments/geometria_proporcional/configs/proportional_graph_equal_budget_ranking_diagnostic_v1.json"
SOURCE_FILES = (
    "experiments/geometria_proporcional/PLAN_PROPORTIONAL_EQUAL_BUDGET_RANKING_DIAGNOSTIC_CPU.md",
    "experiments/geometria_proporcional/configs/proportional_graph_equal_budget_ranking_diagnostic_v1.json",
    "experiments/geometria_proporcional/run_proportional_graph_equal_budget_ranking_diagnostic.py",
    "experiments/geometria_proporcional/run_proportional_graph_selected_action_transport_power_audit.py",
    "experiments/geometria_proporcional/run_proportional_graph_selected_action_calibration_diagnostic.py",
    "experiments/geometria_proporcional/run_proportional_graph_signed_tail_diagnostic.py",
    "experiments/geometria_proporcional/run_proportional_graph_conditional_risk_gate.py",
    "experiments/geometria_proporcional/run_proportional_graph_fresh_mixed_gate.py",
    "experiments/geometria_proporcional/run_proportional_graph_frozen_adapters.py",
)
FAMILIES = ("public_base", "topology")
CONTRASTS = ("minus_identity", "minus_public_base", "minus_topology_permuted")
CELLS = ("iid", "grouped", "balanced")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def load_config(path: Path) -> dict[str, Any]:
    cfg = json.loads(path.read_text())
    expected = {
        "schema_version", "source_transport_audit", "source_manifest_sha256",
        "arms", "alphas", "budget_fractions", "control_replicates",
        "numerical_zero_tolerance", "execution",
    }
    if set(cfg) != expected or cfg["schema_version"] != "proportional-graph-equal-budget-ranking-diagnostic-v1":
        raise ValueError("invalid equal-budget ranking schema")
    if cfg["arms"] != ["raw_generic", "raw_typed", "closure_generic", "closure_typed"]:
        raise ValueError("arms changed")
    if cfg["alphas"] != [0.0, 0.25, 0.5, 0.75, 1.0]:
        raise ValueError("alpha grid changed")
    if cfg["budget_fractions"] != [0.01, 0.02, 0.05, 0.10, 0.20, 0.40]:
        raise ValueError("action budget grid changed")
    if cfg["control_replicates"] != 16 or cfg["numerical_zero_tolerance"] != 1e-12:
        raise ValueError("matched control contract changed")
    if cfg["execution"] != {"max_seconds": 300, "max_rss_gib": 4.0}:
        raise ValueError("CPU execution contract changed")
    return cfg


def source_root(cfg: dict[str, Any]) -> Path:
    return ROOT / cfg["source_transport_audit"]


def verify_source(cfg: dict[str, Any]) -> dict[str, Any]:
    root = source_root(cfg)
    if sha(root / "manifest.json") != cfg["source_manifest_sha256"]:
        raise AssertionError("R365 manifest hash mismatch")
    manifest = json.loads((root / "manifest.json").read_text())
    for relative, expected in manifest["deterministic_files"].items():
        if sha(root / relative) != expected:
            raise AssertionError(f"R365 source mismatch: {relative}")
    source_cfg = transport.load_config(root / "resolved_config.json")
    if source_cfg["arms"] != cfg["arms"] or source_cfg["alphas"] != cfg["alphas"]:
        raise AssertionError("source factorial changed")
    if source_cfg["control_replicates"] != cfg["control_replicates"]:
        raise AssertionError("source controls changed")
    transport.verify_sources(source_cfg)
    return source_cfg


def require_clean(development: bool) -> None:
    if not development and subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout:
        raise RuntimeError("official ranking diagnostic requires clean worktree")


def git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout.strip()


def selected_component(values: np.ndarray, proposal: np.ndarray) -> np.ndarray:
    index = np.asarray(proposal, dtype=np.int64) - 1
    rows = np.arange(len(index))
    return np.asarray(values)[rows, index]


def common_proposal_scores(reconstructed: dict[str, Any], arm_index: int) -> dict[str, np.ndarray]:
    state = reconstructed["state"]
    mu = state["mu"][arm_index]
    public_prediction = state["predicted"]["public_base_selected_action"][arm_index]
    proposal, _ = selected.propose(mu, public_prediction)
    scores = {
        "proposal": proposal,
        "public_base": selected_component(mu + public_prediction, proposal),
        "topology": selected_component(
            mu + state["predicted"]["topology_selected_action"][arm_index], proposal
        ),
    }
    controls = np.column_stack([
        selected_component(mu + state["control"][arm_index, replicate], proposal)
        for replicate in range(state["control"].shape[1])
    ]).T
    scores["topology_permuted"] = controls
    if any(not np.all(np.isfinite(value)) for value in scores.values()):
        raise RuntimeError("non-finite common-proposal score")
    return scores


def equal_budget_action(score: np.ndarray, proposal: np.ndarray, fraction: float) -> np.ndarray:
    n = len(score)
    k = int(math.ceil(fraction * n))
    order = np.lexsort((np.arange(n, dtype=np.int64), np.asarray(score)))
    action = np.zeros(n, dtype=np.int64)
    action[order[:k]] = np.asarray(proposal, dtype=np.int64)[order[:k]]
    if np.count_nonzero(action) != k:
        raise AssertionError("equal action budget not realized")
    return action


def jaccard(left: np.ndarray, right: np.ndarray) -> float:
    a, b = np.asarray(left) > 0, np.asarray(right) > 0
    union = np.sum(a | b)
    return float(np.sum(a & b) / union) if union else 1.0


def interval(values: np.ndarray, bootstrap: np.ndarray) -> dict[str, Any]:
    draws = values[bootstrap].mean(axis=1)
    return {
        "mean": float(np.mean(values)),
        "ci95": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
    }


def policy_cell(values: np.ndarray, cell: str) -> np.ndarray:
    return transport.cell_values(values, cell)


def summarize_cohort(
    cfg: dict[str, Any], name: str, reconstructed: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, np.ndarray]]:
    report: dict[str, Any] = {
        "cohort": name, "n_masters": int(reconstructed["bootstrap"].shape[1]), "arms": {},
    }
    effects: dict[str, np.ndarray] = {"bootstrap_indices": reconstructed["bootstrap"]}
    rankings: dict[str, np.ndarray] = {}
    for arm_index, arm in enumerate(cfg["arms"]):
        score = common_proposal_scores(reconstructed, arm_index)
        proposal = score["proposal"]
        quotient = reconstructed["data"]["quotient_rmse"][arm_index]
        rankings[f"{arm}|common_proposal"] = proposal
        for family in FAMILIES:
            rankings[f"{arm}|score|{family}"] = score[family]
        rankings[f"{arm}|score|topology_permuted"] = score["topology_permuted"]
        arm_report: dict[str, Any] = {
            "common_alpha_counts": {
                str(alpha): int(np.sum(proposal == alpha_index))
                for alpha_index, alpha in enumerate(cfg["alphas"])
            },
            "budgets": {}, "budget_average": {},
        }
        budget_effects: dict[str, dict[str, list[np.ndarray]]] = {
            cell: {contrast: [] for contrast in CONTRASTS} for cell in CELLS
        }
        for fraction in cfg["budget_fractions"]:
            label = f"{fraction:.2f}"
            actions = {
                family: equal_budget_action(score[family], proposal, fraction)
                for family in FAMILIES
            }
            controls = np.stack([
                equal_budget_action(score["topology_permuted"][replicate], proposal, fraction)
                for replicate in range(cfg["control_replicates"])
            ])
            for family, action in actions.items():
                rankings[f"{arm}|budget={label}|action|{family}"] = action
            rankings[f"{arm}|budget={label}|action|topology_permuted"] = controls
            topology = mixed.selected_values(quotient, actions["topology"])
            public = mixed.selected_values(quotient, actions["public_base"])
            control = np.mean([
                mixed.selected_values(quotient, controls[replicate])
                for replicate in range(cfg["control_replicates"])
            ], axis=0)
            identity = quotient[:, 0]
            iid = np.arange(0, len(proposal), 2)
            grouped = np.arange(1, len(proposal), 2)
            topology_action = actions["topology"]
            overlap = [jaccard(topology_action, controls[replicate]) for replicate in range(cfg["control_replicates"])]
            budget_report: dict[str, Any] = {
                "total_action_budget": int(np.count_nonzero(topology_action)),
                "topology_action_by_slice": {
                    "iid": int(np.count_nonzero(topology_action[iid])),
                    "grouped": int(np.count_nonzero(topology_action[grouped])),
                },
                "overlap": {
                    "topology_vs_public_base_jaccard": jaccard(topology_action, actions["public_base"]),
                    "topology_vs_permuted_jaccard": {
                        "mean": float(np.mean(overlap)), "min": float(np.min(overlap)),
                        "max": float(np.max(overlap)),
                    },
                },
                "cells": {},
            }
            for cell in CELLS:
                values = {
                    "identity": policy_cell(identity, cell),
                    "public_base": policy_cell(public, cell),
                    "topology": policy_cell(topology, cell),
                    "topology_permuted": policy_cell(control, cell),
                }
                contrasts = {
                    "minus_identity": values["topology"] - values["identity"],
                    "minus_public_base": values["topology"] - values["public_base"],
                    "minus_topology_permuted": values["topology"] - values["topology_permuted"],
                }
                budget_report["cells"][cell] = {
                    contrast: interval(value, reconstructed["bootstrap"])
                    for contrast, value in contrasts.items()
                }
                for contrast, value in contrasts.items():
                    effects[f"{name}|{arm}|budget={label}|{cell}|{contrast}"] = value
                    budget_effects[cell][contrast].append(value)
            arm_report["budgets"][label] = budget_report
        for cell in CELLS:
            arm_report["budget_average"][cell] = {}
            for contrast in CONTRASTS:
                value = np.mean(budget_effects[cell][contrast], axis=0)
                arm_report["budget_average"][cell][contrast] = interval(value, reconstructed["bootstrap"])
                effects[f"{name}|{arm}|budget_average|{cell}|{contrast}"] = value
        report["arms"][arm] = arm_report
    return report, effects, rankings


def transport_report(cfg: dict[str, Any], cohorts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    report: dict[str, Any] = {
        "status": "OPENED_EQUAL_BUDGET_RANKING_DIAGNOSTIC",
        "policy_scope": "ranking only; no no-harm threshold or firewall",
        "primary_estimand": "topology minus mean topology-permuted at equal total action budget",
        "arms": {},
    }
    counts = {label: 0 for label in (
        "FAVORABLE_BOTH", "ADVERSE_BOTH", "SIGN_UNSTABLE", "IDENTITY_OR_NUMERICAL_ZERO"
    )}
    for arm in cfg["arms"]:
        arm_report: dict[str, Any] = {"budgets": {}, "budget_average": {}}
        for fraction in cfg["budget_fractions"]:
            label = f"{fraction:.2f}"
            arm_report["budgets"][label] = {"cells": {}}
            for cell in CELLS:
                arm_report["budgets"][label]["cells"][cell] = {}
                for contrast in CONTRASTS:
                    rows = {
                        cohort: cohorts[cohort]["arms"][arm]["budgets"][label]["cells"][cell][contrast]
                        for cohort in ("A", "B")
                    }
                    classification = transport.classify_transport(
                        rows["A"]["mean"], rows["B"]["mean"], cfg["numerical_zero_tolerance"]
                    )
                    arm_report["budgets"][label]["cells"][cell][contrast] = {
                        "classification": classification, "cohorts": rows,
                    }
                    if contrast == "minus_topology_permuted":
                        counts[classification] += 1
        for cell in CELLS:
            arm_report["budget_average"][cell] = {}
            for contrast in CONTRASTS:
                rows = {
                    cohort: cohorts[cohort]["arms"][arm]["budget_average"][cell][contrast]
                    for cohort in ("A", "B")
                }
                arm_report["budget_average"][cell][contrast] = {
                    "classification": transport.classify_transport(
                        rows["A"]["mean"], rows["B"]["mean"], cfg["numerical_zero_tolerance"]
                    ),
                    "cohorts": rows,
                }
        report["arms"][arm] = arm_report
    report["primary_transport_counts_over_72_arm_budget_cells"] = counts
    return report


def write_replay(output: Path, cfg: dict[str, Any], head: str) -> None:
    checks = "\n".join(
        f"printf '%s  %s\\n' '{sha(ROOT / relative)}' \"$repo/{relative}\" | sha256sum -c -"
        for relative in SOURCE_FILES
    )
    checks += f"\nprintf '%s  %s\\n' '{sha(source_root(cfg) / 'manifest.json')}' \"$repo/{cfg['source_transport_audit']}/manifest.json\" | sha256sum -c -"
    replay = f'''#!/bin/sh
set -eu
repo=$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd)
[ "$(git -C "$repo" rev-parse HEAD)" = "{head}" ] || {{ echo "wrong git HEAD" >&2; exit 2; }}
[ -z "$(git -C "$repo" status --porcelain)" ] || {{ echo "dirty worktree" >&2; exit 2; }}
{checks}
: "${{OUTPUT_DIR:?set OUTPUT_DIR to a new path}}"
exec env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 "$repo/venv/bin/python" "$repo/experiments/geometria_proporcional/run_proportional_graph_equal_budget_ranking_diagnostic.py" --config "$repo/experiments/geometria_proporcional/configs/proportional_graph_equal_budget_ranking_diagnostic_v1.json" --output "$OUTPUT_DIR"
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
    source_cfg = verify_source(cfg)
    output.mkdir(parents=True)
    reconstructed = {"A": transport.reconstruct_a(source_cfg), "B": transport.reconstruct_b(source_cfg)}
    reports: dict[str, dict[str, Any]] = {}
    all_effects: dict[str, np.ndarray] = {}
    for name in ("A", "B"):
        report, effects, rankings = summarize_cohort(cfg, name, reconstructed[name])
        reports[name] = report
        all_effects.update(effects)
        write_json(output / f"cohort_{name.lower()}_summary.json", report)
        frozen.save_npz(output / f"cohort_{name.lower()}_rankings.npz", rankings)
    frozen.save_npz(output / "effects_by_master.npz", all_effects)
    analysis = transport_report(cfg, reports)
    analysis["cohort_b_exact_r364_reproduction"] = bool(reconstructed["B"]["exact_reproduction"])
    write_json(output / "analysis.json", analysis)
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
        raise RuntimeError("ranking diagnostic resource budget exceeded")
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
