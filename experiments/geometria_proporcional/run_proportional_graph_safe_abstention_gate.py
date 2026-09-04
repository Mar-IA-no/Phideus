#!/usr/bin/env python3
"""Select and evaluate a residual abstention rule under an IID no-harm constraint."""

from __future__ import annotations

import argparse
import hashlib
import json
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
import torch  # noqa: E402

DEFAULT_CONFIG = ROOT / "experiments/geometria_proporcional/configs/proportional_graph_safe_abstention_gate_v1.json"
SOURCE_FILES = (
    "experiments/geometria_proporcional/PLAN_PROPORTIONAL_SAFE_ABSTENTION_GATE_CPU.md",
    "experiments/geometria_proporcional/configs/proportional_graph_safe_abstention_gate_v1.json",
    "experiments/geometria_proporcional/run_proportional_graph_safe_abstention_gate.py",
    "experiments/geometria_proporcional/run_proportional_graph_fresh_mixed_gate.py",
    "experiments/geometria_proporcional/run_proportional_graph_residual_gate.py",
    "experiments/geometria_proporcional/run_proportional_graph_neural_smoke.py",
    "src/geometria_proporcional/proportional_graph_contract.py",
)
FEATURE_ORDER = mixed.FEATURE_ORDER
REDUCED_COLUMNS = mixed.REDUCED_COLUMNS
FAMILIES = ("public_mixed_ridge_gate", "correction_scale_ridge")
POLICIES = (
    "identity", "unconstrained_full", "safe_full",
    "unconstrained_reduced", "safe_reduced", "safe_shuffled",
    "oracle_per_view",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def load_config(path: Path) -> dict[str, Any]:
    cfg = json.loads(path.read_text())
    expected = {
        "schema_version", "source_mixed_gate", "source_mixed_gate_manifest_sha256",
        "source_smoke_config", "arms", "seeds", "alphas", "realizations",
        "threshold_quantiles", "selection_bootstrap_replicates",
        "selection_bootstrap_seed", "selection_upper_percentile",
        "evaluation_bootstrap_replicates", "evaluation_bootstrap_seed", "execution",
    }
    if set(cfg) != expected or cfg["schema_version"] != "proportional-graph-safe-abstention-gate-v1":
        raise ValueError("invalid safe abstention schema")
    if cfg["arms"] != ["raw_generic", "raw_typed", "closure_generic", "closure_typed"]:
        raise ValueError("arms must remain frozen")
    if cfg["seeds"] != [104729, 130363] or cfg["alphas"] != [0.0, 0.25, 0.5, 0.75, 1.0]:
        raise ValueError("seed and alpha contracts must remain frozen")
    if cfg["threshold_quantiles"] != [0.25, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975, 0.99]:
        raise ValueError("threshold quantiles must remain frozen")
    realization = cfg["realizations"]
    if realization != {
        "selection_seed": 2026090529,
        "adjudication_seed": 2026090537,
        "min_eligible_masters": 220,
    }:
        raise ValueError("invalid realization contract")
    if (
        cfg["selection_bootstrap_replicates"] != 2000
        or cfg["selection_bootstrap_seed"] != 2026090541
        or cfg["selection_upper_percentile"] != 95.0
        or cfg["evaluation_bootstrap_replicates"] != 2000
        or cfg["evaluation_bootstrap_seed"] != 2026090547
    ):
        raise ValueError("bootstrap and selection contracts must remain frozen")
    if cfg["execution"] != {
        "torch_threads": 1,
        "max_seconds_per_phase": 720,
        "max_rss_gib": 4.0,
        "sample_every_views": 64,
    }:
        raise ValueError("execution must remain single-threaded")
    return cfg


def source_inputs(cfg: dict[str, Any]) -> list[Path]:
    source = ROOT / cfg["source_mixed_gate"]
    return [
        source / "manifest.json", source / "resolved_config.json",
        source / "calibration/gate_models.json", source / "calibration/features.npz",
    ]


def verify_source(cfg: dict[str, Any]) -> dict[str, Any]:
    source = ROOT / cfg["source_mixed_gate"]
    if sha(source / "manifest.json") != cfg["source_mixed_gate_manifest_sha256"]:
        raise AssertionError("mixed gate manifest hash mismatch")
    manifest = json.loads((source / "manifest.json").read_text())
    for path in source_inputs(cfg)[1:]:
        relative = str(path.relative_to(source))
        if manifest["deterministic_files"].get(relative) != sha(path):
            raise AssertionError(f"mixed gate source mismatch: {relative}")
    mixed_cfg = json.loads((source / "resolved_config.json").read_text())
    if mixed_cfg["arms"] != cfg["arms"] or mixed_cfg["seeds"] != cfg["seeds"] or mixed_cfg["alphas"] != cfg["alphas"]:
        raise AssertionError("safe gate and source factorial differ")
    return mixed_cfg


def source_context(cfg: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[int, torch.nn.Module]]]:
    mixed_cfg = verify_source(cfg)
    source_cfg, _, models = mixed.source_context(mixed_cfg)
    return mixed_cfg, source_cfg, models


def require_clean(development: bool) -> None:
    if not development and subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True,
        text=True, check=True,
    ).stdout:
        raise RuntimeError("official phase requires clean worktree")


def git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
        text=True, check=True,
    ).stdout.strip()


def feature_columns(family: str) -> list[int]:
    if family == "public_mixed_ridge_gate":
        return list(range(len(FEATURE_ORDER)))
    if family == "correction_scale_ridge":
        return list(REDUCED_COLUMNS)
    raise ValueError(f"unknown family: {family}")


def predicted_action_and_advantage(report: dict[str, Any], x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    predictions = mixed.predict_report(report, x)
    action = mixed.historical_gate.choose_alpha(predictions)
    advantage = np.maximum(0.0, -np.min(predictions, axis=1))
    return predictions, action, advantage


def threshold_grid(advantage: np.ndarray, quantiles: list[float]) -> list[float | None]:
    positive = np.asarray(advantage, dtype=np.float64)
    positive = positive[positive > 0]
    values = [0.0]
    if len(positive):
        values.extend(float(x) for x in np.quantile(positive, quantiles))
    unique = sorted(set(values))
    return [*unique, None]


def threshold_action(base_action: np.ndarray, advantage: np.ndarray, threshold: float | None) -> np.ndarray:
    if threshold is None:
        return np.zeros_like(base_action)
    return np.where(advantage > threshold, base_action, 0).astype(np.int64)


def candidate_statistics(
    quotient: np.ndarray, actions: list[np.ndarray], bootstrap: np.ndarray,
    percentile: float,
) -> dict[str, Any]:
    quotient_mean = quotient.mean(axis=0)
    identity = quotient_mean[0]
    iid = np.arange(0, quotient.shape[2], 2)
    grouped = np.arange(1, quotient.shape[2], 2)
    iid_delta, balanced_delta = [], []
    for action in actions:
        selected = quotient_mean[action, np.arange(len(action))]
        delta = selected - identity
        iid_delta.append(delta[iid])
        balanced_delta.append(0.5 * (delta[iid] + delta[grouped]))
    iid_array = np.asarray(iid_delta)
    balanced_array = np.asarray(balanced_delta)
    observed_iid = iid_array.mean(axis=1)
    observed_balanced = balanced_array.mean(axis=1)
    nonidentity = len(actions) - 1
    if nonidentity:
        boot_means = np.asarray([
            iid_array[index][bootstrap].mean(axis=1) for index in range(nonidentity)
        ]).T
        deviations = boot_means - observed_iid[:nonidentity]
        q_upper = float(np.percentile(np.max(deviations, axis=1), percentile))
    else:
        q_upper = 0.0
    upper = [float(value + q_upper) for value in observed_iid[:nonidentity]] + [0.0]
    admissible = [value <= 0.0 for value in upper[:-1]] + [True]
    eligible = np.flatnonzero(admissible)
    best = float(np.min(observed_balanced[eligible]))
    selected_index = int(eligible[np.flatnonzero(observed_balanced[eligible] <= best + 1e-15)[-1]])
    return {
        "iid_delta_by_candidate": iid_array,
        "balanced_delta_by_candidate": balanced_array,
        "mean_iid": observed_iid,
        "mean_balanced": observed_balanced,
        "simultaneous_q95": q_upper,
        "upper_iid": np.asarray(upper),
        "admissible": np.asarray(admissible, dtype=bool),
        "selected_index": selected_index,
    }


def source_feature_means(cfg: dict[str, Any]) -> np.ndarray:
    with np.load(ROOT / cfg["source_mixed_gate"] / "calibration/features.npz", allow_pickle=False) as saved:
        if list(map(str, saved["feature_order"])) != list(FEATURE_ORDER):
            raise AssertionError("source feature order mismatch")
        return saved["features"].mean(axis=1)


def select_thresholds(
    cfg: dict[str, Any], views: list[Any], data: dict[str, np.ndarray],
    gate_models: dict[str, Any], source_features: np.ndarray,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    feature_mean = data["features"].mean(axis=1)
    n_masters = len(views) // 2
    bootstrap = np.random.default_rng(cfg["selection_bootstrap_seed"]).integers(
        0, n_masters, size=(cfg["selection_bootstrap_replicates"], n_masters)
    )
    freeze: dict[str, Any] = {"feature_order": list(FEATURE_ORDER), "arms": {}}
    packed: dict[str, np.ndarray] = {"bootstrap_indices": bootstrap}
    for arm_index, arm in enumerate(cfg["arms"]):
        record: dict[str, Any] = {"families": {}, "shuffled": []}
        for family in FAMILIES:
            report = gate_models["arms"][arm]["models"][family]
            columns = feature_columns(family)
            _, _, source_advantage = predicted_action_and_advantage(report, source_features[arm_index][:, columns])
            thresholds = threshold_grid(source_advantage, cfg["threshold_quantiles"])
            predictions, base_action, advantage = predicted_action_and_advantage(report, feature_mean[arm_index][:, columns])
            actions = [threshold_action(base_action, advantage, threshold) for threshold in thresholds]
            stats = candidate_statistics(
                data["quotient_rmse"][arm_index], actions, bootstrap,
                cfg["selection_upper_percentile"],
            )
            key = f"{arm}|{family}"
            for name in ("iid_delta_by_candidate", "balanced_delta_by_candidate", "upper_iid", "admissible"):
                packed[f"{key}|{name}"] = stats[name]
            packed[f"{key}|advantage"] = advantage
            packed[f"{key}|predicted_nonzero_delta"] = predictions
            packed[f"{key}|base_action"] = base_action
            packed[f"{key}|candidate_actions"] = np.asarray(actions)
            packed[f"{key}|selected_action"] = actions[stats["selected_index"]]
            selected = stats["selected_index"]
            record["families"][family] = {
                "thresholds": thresholds,
                "mean_iid": stats["mean_iid"].tolist(),
                "mean_balanced": stats["mean_balanced"].tolist(),
                "simultaneous_q95": stats["simultaneous_q95"],
                "upper_iid": stats["upper_iid"].tolist(),
                "admissible": stats["admissible"].tolist(),
                "selected_index": selected,
                "selected_threshold": thresholds[selected],
            }
        for replicate, shuffled in enumerate(gate_models["arms"][arm]["shuffled_target_mixed_ridge"]):
            report = shuffled["model"]
            _, _, source_advantage = predicted_action_and_advantage(report, source_features[arm_index])
            thresholds = threshold_grid(source_advantage, cfg["threshold_quantiles"])
            predictions, base_action, advantage = predicted_action_and_advantage(report, feature_mean[arm_index])
            actions = [threshold_action(base_action, advantage, threshold) for threshold in thresholds]
            stats = candidate_statistics(
                data["quotient_rmse"][arm_index], actions, bootstrap,
                cfg["selection_upper_percentile"],
            )
            key = f"{arm}|shuffle={replicate}"
            packed[f"{key}|iid_delta_by_candidate"] = stats["iid_delta_by_candidate"]
            packed[f"{key}|balanced_delta_by_candidate"] = stats["balanced_delta_by_candidate"]
            packed[f"{key}|predicted_nonzero_delta"] = predictions
            packed[f"{key}|advantage"] = advantage
            packed[f"{key}|base_action"] = base_action
            packed[f"{key}|candidate_actions"] = np.asarray(actions)
            packed[f"{key}|selected_action"] = actions[stats["selected_index"]]
            selected = stats["selected_index"]
            record["shuffled"].append({
                "replicate": replicate, "thresholds": thresholds,
                "mean_iid": stats["mean_iid"].tolist(),
                "mean_balanced": stats["mean_balanced"].tolist(),
                "simultaneous_q95": stats["simultaneous_q95"],
                "upper_iid": stats["upper_iid"].tolist(),
                "admissible": stats["admissible"].tolist(),
                "selected_index": selected, "selected_threshold": thresholds[selected],
            })
        freeze["arms"][arm] = record
    return freeze, packed


def deterministic_phase_files(output: Path, role: str) -> list[Path]:
    return sorted(path for path in (output / role).rglob("*") if path.is_file())


def selection_phase(cfg: dict[str, Any], output: Path, development: bool) -> None:
    require_clean(development)
    if output.exists():
        raise FileExistsError(output)
    if not development and (ROOT / "data/geometria_proporcional").resolve() not in output.resolve().parents:
        raise ValueError("official output must be below data/geometria_proporcional")
    output.mkdir(parents=True)
    started = time.monotonic()
    samples = [mixed.resource_sample(started, "select_start")]
    torch.set_num_threads(cfg["execution"]["torch_threads"])
    torch.use_deterministic_algorithms(True)
    mixed_cfg, source_cfg, models = source_context(cfg)
    seed = cfg["realizations"]["selection_seed"]
    views = mixed.fresh_views(cfg, source_cfg, seed)
    write_json(output / "selection/view_index.json", mixed.view_index(views, seed))
    data = mixed.run_universe("selection", views, models, source_cfg, cfg, output, started, samples)
    gate_models = json.loads((ROOT / cfg["source_mixed_gate"] / "calibration/gate_models.json").read_text())
    threshold_freeze, packed = select_thresholds(
        cfg, views, data, gate_models, source_feature_means(cfg)
    )
    write_json(output / "selection/thresholds.json", threshold_freeze)
    frozen.save_npz(output / "selection/threshold_statistics.npz", packed)
    write_json(output / "resolved_config.json", cfg)
    selection_files = deterministic_phase_files(output, "selection")
    selection_manifest = {
        "schema_version": "safe-abstention-selection-manifest-v1",
        "git_head": git_head(), "source_mixed_config_sha256": sha(ROOT / cfg["source_mixed_gate"] / "resolved_config.json"),
        "source_hashes": {path: sha(ROOT / path) for path in SOURCE_FILES},
        "input_hashes": {str(path.relative_to(ROOT)): sha(path) for path in source_inputs(cfg)},
        "selection_seed": seed,
        "selection_files": {str(path.relative_to(output)): sha(path) for path in selection_files},
    }
    write_json(output / "selection_manifest.json", selection_manifest)
    freeze = {
        "schema_version": "safe-abstention-policy-freeze-v1",
        "git_head": git_head(), "config_sha256": sha(output / "resolved_config.json"),
        "selection_manifest_sha256": sha(output / "selection_manifest.json"),
        "thresholds_sha256": sha(output / "selection/thresholds.json"),
        "future_materialized": False,
    }
    write_json(output / "policy_freeze.json", freeze)
    write_json(output / "phase_receipt.json", {
        "phase": "select", "policy_freeze_sha256": sha(output / "policy_freeze.json"),
        "selection_manifest_sha256": sha(output / "selection_manifest.json"),
        "adjudication_seed_materialized": False,
    })
    samples.append(mixed.resource_sample(started, "select_complete"))
    mixed.enforce(samples[-1], cfg)
    write_json(output / "resource_samples_select.json", samples)
    print(json.dumps(samples[-1], sort_keys=True))


def verify_freeze(cfg: dict[str, Any], output: Path) -> None:
    receipt = json.loads((output / "phase_receipt.json").read_text())
    freeze = json.loads((output / "policy_freeze.json").read_text())
    manifest = json.loads((output / "selection_manifest.json").read_text())
    if receipt != {
        "phase": "select", "policy_freeze_sha256": sha(output / "policy_freeze.json"),
        "selection_manifest_sha256": sha(output / "selection_manifest.json"),
        "adjudication_seed_materialized": False,
    }:
        raise AssertionError("selection receipt mismatch")
    if freeze["git_head"] != git_head() or manifest["git_head"] != git_head():
        raise AssertionError("git HEAD changed after threshold freeze")
    if freeze["config_sha256"] != sha(output / "resolved_config.json") or json.loads((output / "resolved_config.json").read_text()) != cfg:
        raise AssertionError("config changed after threshold freeze")
    if freeze["selection_manifest_sha256"] != sha(output / "selection_manifest.json"):
        raise AssertionError("selection manifest changed after freeze")
    if freeze["thresholds_sha256"] != sha(output / "selection/thresholds.json"):
        raise AssertionError("thresholds changed after freeze")
    for relative, expected in manifest["selection_files"].items():
        if sha(output / relative) != expected:
            raise AssertionError(f"selection file changed: {relative}")


def apply_frozen_policies(
    cfg: dict[str, Any], data: dict[str, np.ndarray], gate_models: dict[str, Any],
    threshold_freeze: dict[str, Any],
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, np.ndarray]]:
    features = data["features"].mean(axis=1)
    quotient = data["quotient_rmse"].mean(axis=1)
    n_arms, _, n_views = quotient.shape
    actions = {policy: np.zeros((n_arms, n_views), dtype=np.int64) for policy in POLICIES if policy != "safe_shuffled"}
    shuffled_actions = np.zeros((n_arms, len(gate_models["arms"][cfg["arms"][0]]["shuffled_target_mixed_ridge"]), n_views), dtype=np.int64)
    advantages = {
        "full": np.full((n_arms, n_views), np.nan),
        "reduced": np.full((n_arms, n_views), np.nan),
    }
    for arm_index, arm in enumerate(cfg["arms"]):
        for family, unconstrained, safe, advantage_key in (
            ("public_mixed_ridge_gate", "unconstrained_full", "safe_full", "full"),
            ("correction_scale_ridge", "unconstrained_reduced", "safe_reduced", "reduced"),
        ):
            columns = feature_columns(family)
            report = gate_models["arms"][arm]["models"][family]
            _, base, advantage = predicted_action_and_advantage(report, features[arm_index][:, columns])
            chosen = threshold_freeze["arms"][arm]["families"][family]["selected_threshold"]
            actions[unconstrained][arm_index] = threshold_action(base, advantage, 0.0)
            actions[safe][arm_index] = threshold_action(base, advantage, chosen)
            advantages[advantage_key][arm_index] = advantage
        for replicate, shuffled in enumerate(gate_models["arms"][arm]["shuffled_target_mixed_ridge"]):
            _, base, advantage = predicted_action_and_advantage(shuffled["model"], features[arm_index])
            chosen = threshold_freeze["arms"][arm]["shuffled"][replicate]["selected_threshold"]
            shuffled_actions[arm_index, replicate] = threshold_action(base, advantage, chosen)
        actions["oracle_per_view"][arm_index] = np.argmin(quotient[arm_index], axis=0)
    return actions, shuffled_actions, advantages


def policy_values(
    quotient: np.ndarray, actions: dict[str, np.ndarray], shuffled: np.ndarray,
    arm_index: int, cfg: dict[str, Any],
) -> dict[str, np.ndarray]:
    values = {
        policy: mixed.selected_values(quotient[arm_index], action[arm_index])
        for policy, action in actions.items()
    }
    values["safe_shuffled"] = np.mean([
        mixed.selected_values(quotient[arm_index], shuffled[arm_index, replicate])
        for replicate in range(shuffled.shape[1])
    ], axis=0)
    return values


def summarize(
    cfg: dict[str, Any], data: dict[str, np.ndarray], actions: dict[str, np.ndarray],
    shuffled: np.ndarray, advantages: dict[str, np.ndarray], views: list[Any],
    bootstrap: np.ndarray,
) -> tuple[dict[str, Any], dict[str, Any]]:
    effects: dict[str, Any] = {"arms": {}}
    diagnostics: dict[str, Any] = {"arms": {}}
    iid, grouped = np.arange(0, len(views), 2), np.arange(1, len(views), 2)
    for arm_index, arm in enumerate(cfg["arms"]):
        values = policy_values(data["quotient_rmse"], actions, shuffled, arm_index, cfg)
        arm_effect: dict[str, Any] = {}
        for cell, indices in (("iid", iid), ("grouped", grouped), ("balanced", None)):
            per_policy = {}
            for policy in POLICIES:
                raw = values[policy]
                per_policy[policy] = (
                    raw[:, indices].mean(axis=0) if indices is not None
                    else (0.5 * (raw[:, iid] + raw[:, grouped])).mean(axis=0)
                )
            arm_effect[cell] = {}
            for policy, master_values in per_policy.items():
                draws = master_values[bootstrap].mean(axis=1)
                row: dict[str, Any] = {
                    "n_masters": len(master_values), "mean_rmse": float(master_values.mean()),
                    "ci95_mean": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
                }
                for baseline in ("identity", "unconstrained_full", "unconstrained_reduced", "safe_reduced", "safe_shuffled", "oracle_per_view"):
                    if baseline == policy:
                        continue
                    delta = master_values - per_policy[baseline]
                    delta_draws = delta[bootstrap].mean(axis=1)
                    row[f"minus_{baseline}"] = {
                        "mean": float(delta.mean()),
                        "ci95": [float(np.percentile(delta_draws, 2.5)), float(np.percentile(delta_draws, 97.5))],
                    }
                arm_effect[cell][policy] = row
        full_iid = arm_effect["iid"]["safe_full"]["minus_identity"]
        full_grouped = arm_effect["grouped"]["safe_full"]["minus_identity"]
        delta_iid = values["safe_full"].mean(axis=0)[iid] - values["identity"].mean(axis=0)[iid]
        delta_grouped = values["safe_full"].mean(axis=0)[grouped] - values["identity"].mean(axis=0)[grouped]
        interaction = delta_grouped - delta_iid
        arm_effect["grouped_minus_iid_safe_full_effect"] = {
            "mean": float(interaction.mean()),
            "ci95": [float(np.percentile(interaction[bootstrap].mean(axis=1), 2.5)), float(np.percentile(interaction[bootstrap].mean(axis=1), 97.5))],
            "iid_effect": full_iid, "grouped_effect": full_grouped,
        }
        arm_effect["action_fractions"] = {}
        for cell, indices in (("iid", iid), ("grouped", grouped)):
            arm_effect["action_fractions"][cell] = {
                policy: {str(alpha): float(np.mean(action[arm_index, indices] == alpha_index)) for alpha_index, alpha in enumerate(cfg["alphas"])}
                for policy, action in actions.items()
            }
            arm_effect["action_fractions"][cell]["safe_shuffled"] = {
                str(alpha): float(np.mean(shuffled[arm_index, :, indices] == alpha_index))
                for alpha_index, alpha in enumerate(cfg["alphas"])
            }
        effects["arms"][arm] = arm_effect
        diagnostics["arms"][arm] = {}
        quotient_mean = data["quotient_rmse"][arm_index].mean(axis=0)
        for family, base_policy, key in (
            ("full", "unconstrained_full", "full"),
            ("reduced", "unconstrained_reduced", "reduced"),
        ):
            base = quotient_mean[actions[base_policy][arm_index], np.arange(len(views))]
            realized = quotient_mean[0] - base
            diagnostics["arms"][arm][family] = {
                cell: {"advantage_benefit_correlation": mixed.historical_gate.correlation(advantages[key][arm_index, indices], realized[indices])}
                for cell, indices in (("iid", iid), ("grouped", grouped))
            }
    return effects, diagnostics


def write_replay(output: Path, cfg: dict[str, Any], head: str) -> None:
    checks = "\n".join(
        f"printf '%s  %s\\n' '{sha(ROOT / path)}' \"$repo/{path}\" | sha256sum -c -"
        for path in SOURCE_FILES
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
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 "$repo/venv/bin/python" "$repo/experiments/geometria_proporcional/run_proportional_graph_safe_abstention_gate.py" --phase select --config "$repo/experiments/geometria_proporcional/configs/proportional_graph_safe_abstention_gate_v1.json" --output "$OUTPUT_DIR"
exec env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 "$repo/venv/bin/python" "$repo/experiments/geometria_proporcional/run_proportional_graph_safe_abstention_gate.py" --phase evaluate --config "$repo/experiments/geometria_proporcional/configs/proportional_graph_safe_abstention_gate_v1.json" --output "$OUTPUT_DIR"
'''
    (output / "replay.sh").write_text(replay)
    (output / "replay.sh").chmod(0o755)


def evaluation_phase(cfg: dict[str, Any], output: Path, development: bool) -> None:
    require_clean(development)
    if not output.is_dir() or (output / "adjudication").exists():
        raise RuntimeError("evaluation requires selected, unevaluated output")
    verify_freeze(cfg, output)
    started = time.monotonic()
    samples = [mixed.resource_sample(started, "evaluate_start")]
    torch.set_num_threads(cfg["execution"]["torch_threads"])
    torch.use_deterministic_algorithms(True)
    _, source_cfg, models = source_context(cfg)
    seed = cfg["realizations"]["adjudication_seed"]
    views = mixed.fresh_views(cfg, source_cfg, seed)
    selection_masters = {row["master_id"] for row in json.loads((output / "selection/view_index.json").read_text())}
    if selection_masters & {view.private.master_id for view in views}:
        raise AssertionError("selection and adjudication masters overlap")
    write_json(output / "adjudication/view_index.json", mixed.view_index(views, seed))
    data = mixed.run_universe("adjudication", views, models, source_cfg, cfg, output, started, samples)
    source = ROOT / cfg["source_mixed_gate"]
    gate_models = json.loads((source / "calibration/gate_models.json").read_text())
    threshold_freeze = json.loads((output / "selection/thresholds.json").read_text())
    actions, shuffled, advantages = apply_frozen_policies(cfg, data, gate_models, threshold_freeze)
    frozen.save_npz(output / "adjudication/decisions.npz", {**actions, "safe_shuffled": shuffled})
    frozen.save_npz(output / "adjudication/advantages.npz", advantages)
    n_masters = len(views) // 2
    bootstrap = np.random.default_rng(cfg["evaluation_bootstrap_seed"]).integers(
        0, n_masters, size=(cfg["evaluation_bootstrap_replicates"], n_masters)
    )
    frozen.save_npz(output / "bootstrap_indices.npz", {
        "master_id": np.asarray([views[index].private.master_id for index in range(0, len(views), 2)]),
        f"complete_n_{n_masters}": bootstrap,
    })
    effects, diagnostics = summarize(cfg, data, actions, shuffled, advantages, views, bootstrap)
    write_json(output / "effects.json", effects)
    write_json(output / "prediction_diagnostics.json", diagnostics)
    samples.append(mixed.resource_sample(started, "evaluate_complete"))
    mixed.enforce(samples[-1], cfg)
    write_json(output / "resource_samples_evaluate.json", samples)
    write_replay(output, cfg, git_head())
    deterministic = sorted(
        path for path in output.rglob("*") if path.is_file() and path.name not in {
            "manifest.json", "resource_samples_select.json", "resource_samples_evaluate.json", "runtime_observation.json",
        }
    )
    manifest = {
        "schema_version": cfg["schema_version"], "git_head": git_head(),
        "source_hashes": {path: sha(ROOT / path) for path in SOURCE_FILES},
        "input_hashes": {str(path.relative_to(ROOT)): sha(path) for path in source_inputs(cfg)},
        "deterministic_files": {str(path.relative_to(output)): sha(path) for path in deterministic},
        "runtime_exclusions": ["resource_samples_select.json", "resource_samples_evaluate.json", "runtime_observation.json"],
    }
    write_json(output / "manifest.json", manifest)
    runtime = mixed.resource_sample(started, "evaluate_complete")
    write_json(output / "runtime_observation.json", runtime)
    print(json.dumps(runtime, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("select", "evaluate"), required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--development", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    output = args.output.resolve()
    if args.phase == "select":
        selection_phase(cfg, output, args.development)
    else:
        evaluation_phase(cfg, output, args.development)


if __name__ == "__main__":
    main()
