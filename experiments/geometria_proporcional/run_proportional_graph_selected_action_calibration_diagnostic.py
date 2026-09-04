#!/usr/bin/env python3
"""Diagnose propose-then-calibrate selected-action bounds from opened CPU artifacts."""

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

DEFAULT_CONFIG = ROOT / "experiments/geometria_proporcional/configs/proportional_graph_selected_action_calibration_diagnostic_v1.json"
SOURCE_FILES = (
    "experiments/geometria_proporcional/PLAN_PROPORTIONAL_SELECTED_ACTION_CALIBRATION_DIAGNOSTIC_CPU.md",
    "experiments/geometria_proporcional/configs/proportional_graph_selected_action_calibration_diagnostic_v1.json",
    "experiments/geometria_proporcional/run_proportional_graph_selected_action_calibration_diagnostic.py",
    "experiments/geometria_proporcional/run_proportional_graph_signed_tail_gate.py",
    "experiments/geometria_proporcional/run_proportional_graph_signed_tail_diagnostic.py",
    "experiments/geometria_proporcional/run_proportional_graph_conditional_risk_gate.py",
    "experiments/geometria_proporcional/run_proportional_graph_fresh_mixed_gate.py",
    "experiments/geometria_proporcional/run_proportional_graph_frozen_adapters.py",
)
FAMILIES = (
    "constant_selected_action", "public_base_selected_action", "topology_selected_action",
)
CONTROL_FAMILY = "topology_permuted_selected_action"
BASE_NAME = {
    "constant_selected_action": "constant",
    "public_base_selected_action": "public_base",
    "topology_selected_action": "topology",
}


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
        "schema_version", "source_signed_tail_gate", "source_manifest_sha256",
        "arms", "alphas", "miscoverage", "control_replicates",
        "selection_upper_percentile", "execution",
    }
    if set(cfg) != expected or cfg["schema_version"] != "proportional-graph-selected-action-calibration-diagnostic-v1":
        raise ValueError("invalid selected-action calibration schema")
    if cfg["arms"] != ["raw_generic", "raw_typed", "closure_generic", "closure_typed"]:
        raise ValueError("arms changed")
    if cfg["alphas"] != [0.0, 0.25, 0.5, 0.75, 1.0]:
        raise ValueError("alpha grid changed")
    if cfg["miscoverage"] != 0.10 or cfg["control_replicates"] != 16:
        raise ValueError("calibration contract changed")
    if cfg["selection_upper_percentile"] != 95.0:
        raise ValueError("firewall contract changed")
    if cfg["execution"] != {"max_seconds": 180, "max_rss_gib": 4.0}:
        raise ValueError("CPU execution contract changed")
    return cfg


def source_root(cfg: dict[str, Any]) -> Path:
    return ROOT / cfg["source_signed_tail_gate"]


def verify_source(cfg: dict[str, Any]) -> dict[str, Any]:
    root = source_root(cfg)
    if sha(root / "manifest.json") != cfg["source_manifest_sha256"]:
        raise AssertionError("signed-tail gate manifest hash mismatch")
    manifest = json.loads((root / "manifest.json").read_text())
    for relative, expected in manifest["deterministic_files"].items():
        if sha(root / relative) != expected:
            raise AssertionError(f"signed-tail source mismatch: {relative}")
    source_cfg = json.loads((root / "resolved_config.json").read_text())
    if source_cfg["arms"] != cfg["arms"] or source_cfg["alphas"] != cfg["alphas"]:
        raise AssertionError("source factorial changed")
    if source_cfg["miscoverage"] != cfg["miscoverage"]:
        raise AssertionError("source coverage level changed")
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


def role_state(cfg: dict[str, Any], role: str) -> dict[str, Any]:
    root = source_root(cfg)
    data, _ = conditional.load_phase_data(root, role)
    if role == "risk_calibration":
        arrays = read_npz(root / role / "signed_scores.npz")
        mu, delta = arrays["mu"], arrays["delta"]
        predicted = {family: arrays[f"prediction|{family}"] for family in ("constant", "public_base", "topology")}
        control = arrays["control_prediction"]
    elif role == "policy_selection":
        arrays = read_npz(root / role / "selection_statistics.npz")
        quantiles = json.loads((root / "risk_calibration/quantiles.json").read_text())
        predicted = {family: arrays[f"signed_tail|{family}|risk"] for family in ("constant", "public_base", "topology")}
        control = arrays["signed_tail|control_risk"]
        mu = np.full_like(predicted["topology"], np.nan)
        for arm_index, arm in enumerate(cfg["arms"]):
            q = quantiles["interfaces"]["signed_tail"]["arms"][arm]["families"]["topology"]["q"]
            mu[arm_index] = arrays["signed_tail|topology|upper"][arm_index] - predicted["topology"][arm_index] - q
        delta = conditional.realized_deltas(data)
    elif role == "adjudication":
        arrays = read_npz(root / role / "decisions_and_risk.npz")
        mu, delta = arrays["mu"], arrays["delta"]
        predicted = {family: arrays[f"signed_tail|{family}|risk"] for family in ("constant", "public_base", "topology")}
        control = arrays["signed_tail|control_risk"]
    else:
        raise ValueError(f"unsupported role: {role}")
    values = [mu, delta, control, *predicted.values()]
    if any(not np.all(np.isfinite(value)) for value in values):
        raise RuntimeError(f"non-finite opened state: {role}")
    return {"data": data, "mu": mu, "delta": delta, "predicted": predicted, "control": control}


def propose(mu: np.ndarray, predicted: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    all_bounds = np.asarray(mu) + np.asarray(predicted)
    index = np.argmin(all_bounds, axis=1)
    return (index + 1).astype(np.int64), all_bounds[np.arange(len(index)), index]


def selected_score(residual: np.ndarray, predicted: np.ndarray, proposal: np.ndarray) -> np.ndarray:
    index = np.asarray(proposal, dtype=np.int64) - 1
    rows = np.arange(len(index))
    return np.asarray(residual)[rows, index] - np.asarray(predicted)[rows, index]


def calibrated_action(proposal: np.ndarray, base_bound: np.ndarray, q: float) -> tuple[np.ndarray, np.ndarray]:
    upper = np.asarray(base_bound) + q
    action = np.where(upper < 0.0, proposal, 0).astype(np.int64)
    return action, upper


def calibrate(cfg: dict[str, Any]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    state = role_state(cfg, "risk_calibration")
    iid = np.arange(0, state["delta"].shape[1], 2)
    residual = state["delta"] - state["mu"]
    simultaneous = json.loads((source_root(cfg) / "risk_calibration/quantiles.json").read_text())
    result: dict[str, Any] = {"miscoverage": cfg["miscoverage"], "arms": {}}
    packed: dict[str, np.ndarray] = {"iid_index": iid}
    for arm_index, arm in enumerate(cfg["arms"]):
        record: dict[str, Any] = {"families": {}, CONTROL_FAMILY: []}
        for family in FAMILIES:
            base = BASE_NAME[family]
            proposal, base_bound = propose(state["mu"][arm_index], state["predicted"][base][arm_index])
            score = selected_score(residual[arm_index], state["predicted"][base][arm_index], proposal)
            q, rank = conditional.conformal_quantile(score[iid], cfg["miscoverage"])
            q_sim = simultaneous["interfaces"]["signed_tail"]["arms"][arm]["families"][base]["q"]
            if q > q_sim + 1e-12:
                raise AssertionError("selected-action quantile exceeds simultaneous quantile")
            record["families"][family] = {
                "q_selected": q, "q_simultaneous": q_sim, "rank": rank, "n": len(iid),
            }
            packed[f"{arm}|{family}|proposal"] = proposal
            packed[f"{arm}|{family}|base_bound"] = base_bound
            packed[f"{arm}|{family}|score"] = score[iid]
        for replicate in range(cfg["control_replicates"]):
            prediction = state["control"][arm_index, replicate]
            proposal, base_bound = propose(state["mu"][arm_index], prediction)
            score = selected_score(residual[arm_index], prediction, proposal)
            q, rank = conditional.conformal_quantile(score[iid], cfg["miscoverage"])
            q_sim = simultaneous["interfaces"]["signed_tail"]["arms"][arm]["topology_permuted"][replicate]["q"]
            if q > q_sim + 1e-12:
                raise AssertionError("selected control quantile exceeds simultaneous quantile")
            record[CONTROL_FAMILY].append({
                "replicate": replicate, "q_selected": q, "q_simultaneous": q_sim,
                "rank": rank, "n": len(iid),
            })
            packed[f"{arm}|{CONTROL_FAMILY}={replicate}|proposal"] = proposal
            packed[f"{arm}|{CONTROL_FAMILY}={replicate}|base_bound"] = base_bound
            packed[f"{arm}|{CONTROL_FAMILY}={replicate}|score"] = score[iid]
        result["arms"][arm] = record
    return result, packed


def actions_for_role(
    cfg: dict[str, Any], role: str, calibration: dict[str, Any],
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, np.ndarray], np.ndarray, dict[str, np.ndarray], np.ndarray]:
    state = role_state(cfg, role)
    shape = state["mu"].shape[:2]
    actions = {family: np.zeros(shape, dtype=np.int64) for family in FAMILIES}
    proposals = {family: np.zeros(shape, dtype=np.int64) for family in FAMILIES}
    uppers = {family: np.full(shape, np.nan) for family in FAMILIES}
    controls = np.zeros((len(cfg["arms"]), cfg["control_replicates"], shape[1]), dtype=np.int64)
    control_proposals = np.zeros_like(controls)
    control_uppers = np.full_like(controls, np.nan, dtype=np.float64)
    for arm_index, arm in enumerate(cfg["arms"]):
        for family in FAMILIES:
            base = BASE_NAME[family]
            proposal, base_bound = propose(state["mu"][arm_index], state["predicted"][base][arm_index])
            action, upper = calibrated_action(
                proposal, base_bound, calibration["arms"][arm]["families"][family]["q_selected"]
            )
            proposals[family][arm_index] = proposal
            actions[family][arm_index] = action
            uppers[family][arm_index] = upper
        for replicate in range(cfg["control_replicates"]):
            proposal, base_bound = propose(state["mu"][arm_index], state["control"][arm_index, replicate])
            action, upper = calibrated_action(
                proposal, base_bound, calibration["arms"][arm][CONTROL_FAMILY][replicate]["q_selected"]
            )
            control_proposals[arm_index, replicate] = proposal
            controls[arm_index, replicate] = action
            control_uppers[arm_index, replicate] = upper
    return actions, controls, proposals, control_proposals, uppers, control_uppers


def selection_firewall(
    cfg: dict[str, Any], calibration: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    state = role_state(cfg, "policy_selection")
    actions, controls, proposals, control_proposals, uppers, control_uppers = actions_for_role(
        cfg, "policy_selection", calibration
    )
    source = read_npz(source_root(cfg) / "policy_selection/selection_statistics.npz")
    bootstrap = source["bootstrap_indices"].astype(np.int64)
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
    return {
        "mean": float(values.mean()),
        "ci95": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
    }


def summarize(
    cfg: dict[str, Any], calibration: dict[str, Any], firewall: dict[str, Any],
    actions: dict[str, np.ndarray], controls: np.ndarray,
    proposals: dict[str, np.ndarray], control_proposals: np.ndarray,
    uppers: dict[str, np.ndarray], control_uppers: np.ndarray,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    root = source_root(cfg)
    state = role_state(cfg, "adjudication")
    source_decisions = read_npz(root / "adjudication/decisions_and_risk.npz")
    bootstrap = read_npz(root / "bootstrap_indices.npz")["indices"].astype(np.int64)
    deployed, deployed_controls = apply_firewall(cfg, actions, controls, firewall)
    iid = np.arange(0, state["delta"].shape[1], 2)
    grouped = np.arange(1, state["delta"].shape[1], 2)
    report: dict[str, Any] = {
        "status": "OPENED_POSTHOC_INTERFACE_DIAGNOSTIC",
        "coverage_scope": "marginal IID selected-action only",
        "calibration": calibration, "selection_firewall": firewall, "arms": {},
    }
    packed: dict[str, np.ndarray] = {
        "control_proposal": control_proposals, "control_upper": control_uppers,
        "calibrated_control": controls, "deployed_control": deployed_controls,
    }
    for family in FAMILIES:
        packed[f"{family}|proposal"] = proposals[family]
        packed[f"{family}|upper"] = uppers[family]
        packed[f"{family}|calibrated"] = actions[family]
        packed[f"{family}|deployed"] = deployed[family]
    for arm_index, arm in enumerate(cfg["arms"]):
        quotient = state["data"]["quotient_rmse"][arm_index]
        values: dict[str, np.ndarray] = {
            "identity": quotient[:, 0],
            "simultaneous_calibrated_topology": mixed.selected_values(
                quotient, source_decisions["signed_tail|topology|calibrated"][arm_index]
            ),
            "simultaneous_deployed_topology": mixed.selected_values(
                quotient, source_decisions["signed_tail|topology|deployed"][arm_index]
            ),
        }
        for stage, stage_actions in (("calibrated", actions), ("deployed", deployed)):
            for family in FAMILIES:
                values[f"{stage}_{family}"] = mixed.selected_values(
                    quotient, stage_actions[family][arm_index]
                )
            control_action = controls if stage == "calibrated" else deployed_controls
            values[f"{stage}_{CONTROL_FAMILY}"] = np.mean([
                mixed.selected_values(quotient, control_action[arm_index, replicate])
                for replicate in range(cfg["control_replicates"])
            ], axis=0)
        arm_report: dict[str, Any] = {
            "cells": {}, "action_fractions": {}, "coverage": {},
            "acted_harm": {}, "interaction_grouped_minus_iid": {}, "action_overlap": {},
        }
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
            arm_report["cells"][cell] = {}
            for name, value in per_policy.items():
                row = {"minus_identity": interval(value - per_policy["identity"], bootstrap)}
                if name in ("calibrated_topology_selected_action", "deployed_topology_selected_action"):
                    stage = name.split("_topology_selected_action")[0]
                    for baseline in ("constant_selected_action", "public_base_selected_action", CONTROL_FAMILY):
                        row[f"minus_{stage}_{baseline}"] = interval(
                            value - per_policy[f"{stage}_{baseline}"], bootstrap
                        )
                    row["minus_simultaneous_topology"] = interval(
                        value - per_policy[f"simultaneous_{stage}_topology"], bootstrap
                    )
                arm_report["cells"][cell][name] = row
        residual = state["delta"][arm_index] - state["mu"][arm_index]
        for family in FAMILIES:
            base = BASE_NAME[family]
            proposed = proposals[family][arm_index]
            chosen_delta = residual[np.arange(len(proposed)), proposed - 1]
            chosen_prediction = state["predicted"][base][arm_index][np.arange(len(proposed)), proposed - 1]
            covered = chosen_delta <= chosen_prediction + calibration["arms"][arm]["families"][family]["q_selected"]
            arm_report["coverage"][family] = {
                "iid": float(np.mean(covered[iid])), "grouped": float(np.mean(covered[grouped])),
            }
            arm_report["action_fractions"][family] = {}
            arm_report["acted_harm"][family] = {}
            for stage, action in (("calibrated", actions[family][arm_index]), ("deployed", deployed[family][arm_index])):
                arm_report["action_fractions"][family][stage] = {
                    cell: {
                        str(alpha): float(np.mean(action[indices] == alpha_index))
                        for alpha_index, alpha in enumerate(cfg["alphas"])
                    }
                    for cell, indices in (("iid", iid), ("grouped", grouped))
                }
                realized = conditional.selected_delta(quotient, action)
                arm_report["acted_harm"][family][stage] = {}
                for cell, indices in (("iid", iid), ("grouped", grouped)):
                    acted = action[indices] > 0
                    selected = realized[indices][acted]
                    arm_report["acted_harm"][family][stage][cell] = {
                        "n": int(acted.sum()),
                        "mean_delta": float(selected.mean()) if len(selected) else None,
                        "positive_fraction": float(np.mean(selected > 0)) if len(selected) else None,
                    }
                iid_effect = cell_values["iid"][f"{stage}_{family}"] - cell_values["iid"]["identity"]
                grouped_effect = cell_values["grouped"][f"{stage}_{family}"] - cell_values["grouped"]["identity"]
                arm_report["interaction_grouped_minus_iid"][f"{stage}_{family}"] = interval(
                    grouped_effect - iid_effect, bootstrap
                )
        for stage, action in (("calibrated", actions["topology_selected_action"]), ("deployed", deployed["topology_selected_action"])):
            simultaneous = source_decisions[f"signed_tail|topology|{stage}"][arm_index]
            selected = action[arm_index]
            arm_report["action_overlap"][stage] = {
                "same_action_fraction": float(np.mean(selected == simultaneous)),
                "both_act_fraction": float(np.mean((selected > 0) & (simultaneous > 0))),
                "selected_only_fraction": float(np.mean((selected > 0) & (simultaneous == 0))),
                "simultaneous_only_fraction": float(np.mean((selected == 0) & (simultaneous > 0))),
            }
        report["arms"][arm] = arm_report
    return report, packed


def write_replay(output: Path, cfg: dict[str, Any], head: str) -> None:
    checks = "\n".join(
        f"printf '%s  %s\\n' '{sha(ROOT / relative)}' \"$repo/{relative}\" | sha256sum -c -"
        for relative in SOURCE_FILES
    )
    checks += f"\nprintf '%s  %s\\n' '{sha(source_root(cfg) / 'manifest.json')}' \"$repo/{cfg['source_signed_tail_gate']}/manifest.json\" | sha256sum -c -"
    replay = f'''#!/bin/sh
set -eu
repo=$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd)
[ "$(git -C "$repo" rev-parse HEAD)" = "{head}" ] || {{ echo "wrong git HEAD" >&2; exit 2; }}
[ -z "$(git -C "$repo" status --porcelain)" ] || {{ echo "dirty worktree" >&2; exit 2; }}
{checks}
: "${{OUTPUT_DIR:?set OUTPUT_DIR to a new path}}"
exec env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 "$repo/venv/bin/python" "$repo/experiments/geometria_proporcional/run_proportional_graph_selected_action_calibration_diagnostic.py" --config "$repo/experiments/geometria_proporcional/configs/proportional_graph_selected_action_calibration_diagnostic_v1.json" --output "$OUTPUT_DIR"
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
    verify_source(cfg)
    output.mkdir(parents=True)
    calibration, calibration_arrays = calibrate(cfg)
    write_json(output / "calibration.json", calibration)
    frozen.save_npz(output / "calibration_diagnostics.npz", calibration_arrays)
    firewall, selection_arrays = selection_firewall(cfg, calibration)
    write_json(output / "selection_firewall.json", firewall)
    frozen.save_npz(output / "selection_diagnostics.npz", selection_arrays)
    actions, controls, proposals, control_proposals, uppers, control_uppers = actions_for_role(
        cfg, "adjudication", calibration
    )
    analysis, adjudication_arrays = summarize(
        cfg, calibration, firewall, actions, controls, proposals,
        control_proposals, uppers, control_uppers,
    )
    write_json(output / "analysis.json", analysis)
    frozen.save_npz(output / "adjudication_diagnostics.npz", adjudication_arrays)
    write_json(output / "environment.json", {
        "python": platform.python_version(), "numpy": importlib.metadata.version("numpy"),
        "scipy": importlib.metadata.version("scipy"),
        "scikit_learn": importlib.metadata.version("scikit-learn"),
        "threads": 1, "cuda_visible_devices": "", "refit": False, "new_solves": 0,
    })
    write_json(output / "resolved_config.json", cfg)
    write_replay(output, cfg, git_head())
    runtime = {
        "elapsed_seconds": time.monotonic() - started,
        "max_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
    }
    if runtime["elapsed_seconds"] > cfg["execution"]["max_seconds"] or runtime["max_rss_gib"] > cfg["execution"]["max_rss_gib"]:
        raise RuntimeError("diagnostic resource budget exceeded")
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
