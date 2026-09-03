#!/usr/bin/env python3
"""Run the Wave 54 CPU-only regularized joint-set posterior smoke."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import importlib.metadata
import itertools
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import expit

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
PLAN_PATH = (
    REPO_ROOT
    / "Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_54_JOINT_SET_POSTERIOR_PLAN.md"
)
PRIMITIVES_PATH = SRC_ROOT / "geometria_proporcional/wave54_joint_set.py"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from geometria_proporcional.wave52_policy import (  # noqa: E402
    authorized_actions,
    constrained_regret,
    explicit_set_actions,
)
from geometria_proporcional.wave53_uncertainty import (  # noqa: E402
    apply_coverage_boundary,
    coverage_boundary,
    deranged_within_strata,
    discrete_aurc,
    independent_nonempty_mass,
    nonempty_sets,
    paired_bootstrap_indices,
    paired_delta_ci,
)
from geometria_proporcional.wave54_joint_set import (  # noqa: E402
    STRUCTURES,
    empirical_set_prior,
    expected_regret_from_mass,
    fit_joint_posterior,
    marginal_probability,
    posterior_mass,
    target_set_indices,
)

POSTERIOR_ARMS = (
    "independent_raw",
    "independent_platt",
    "joint_unary",
    "joint_unary_cardinality",
    "joint_full",
    "empirical_set_prior",
    "joint_full_target_shuffled",
)
ACTION_ARMS = ("hard_set_policy",) + POSTERIOR_ARMS + ("oracle_set_then_utility",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--wave52-dir", type=Path, required=True)
    parser.add_argument("--wave53-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT
        / "experiments/geometria_proporcional/configs/wave54_joint_set.json",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def require_hash(path: Path, expected: str) -> dict[str, str]:
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"hash mismatch for {path}: expected {expected}, got {actual}")
    return {"path": str(path), "sha256": actual}


def file_receipt(path: Path) -> dict[str, str]:
    path = path.resolve(strict=True)
    return {"path": str(path), "sha256": sha256_file(path)}


def paths_overlap(left: Path, right: Path) -> bool:
    left = left.resolve()
    right = right.resolve()
    return left == right or left in right.parents or right in left.parents


def reject_lockbox_paths(paths: list[Path]) -> None:
    for path in paths:
        resolved = path.resolve()
        if any("lockbox" in part.lower() for part in resolved.parts):
            raise ValueError(f"Wave 54 must not access a lockbox path: {resolved}")


def require_sources_at_head(paths: list[Path]) -> None:
    for path in paths:
        resolved = path.resolve(strict=True)
        try:
            relative = resolved.relative_to(REPO_ROOT)
        except ValueError as error:
            raise RuntimeError(f"execution source is outside the repository: {resolved}") from error
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", str(relative)],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        if tracked.returncode != 0:
            raise RuntimeError(f"execution source is not tracked at HEAD: {relative}")
        changed = subprocess.run(
            ["git", "status", "--porcelain", "--", str(relative)],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        if changed:
            raise RuntimeError(f"execution source differs from HEAD: {relative}")


def prepare_output_directory(output: Path, force: bool) -> Path | None:
    archived = None
    if output.exists():
        if not force:
            raise FileExistsError(f"output exists: {output}; use --force")
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        archived = output.with_name(f"{output.name}.superseded_{stamp}")
        if archived.exists():
            raise FileExistsError(f"archive target already exists: {archived}")
        output.rename(archived)
    output.mkdir(parents=True)
    return archived


def validate_output_path(
    output: Path,
    inputs: list[Path],
    execution_sources: list[Path],
    repo_root: Path = REPO_ROOT,
) -> None:
    if any(paths_overlap(output, path) for path in inputs):
        raise ValueError("output cannot overlap an input or reference path")
    if output == repo_root or output in repo_root.parents:
        raise ValueError("output cannot be the repository or one of its ancestors")
    if any(output == path.resolve() or output in path.resolve().parents for path in execution_sources):
        raise ValueError("output cannot contain an execution source")


def validate_bundle_separation(
    fit: dict[str, np.ndarray], monitor: dict[str, np.ndarray]
) -> None:
    fit_tokens = fit["pair_token"].astype(str)
    monitor_tokens = monitor["pair_token"].astype(str)
    if len(set(fit_tokens)) != len(fit_tokens):
        raise RuntimeError("duplicate pair_token in fit/select bundle")
    if len(set(monitor_tokens)) != len(monitor_tokens):
        raise RuntimeError("duplicate pair_token in sealed monitor bundle")
    overlap = set(fit_tokens) & set(monitor_tokens)
    if overlap:
        raise RuntimeError(f"fit/select and monitor overlap on {len(overlap)} pair_token values")


def git_state() -> tuple[str, str]:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=REPO_ROOT,
        text=True,
    ).strip()
    return commit, dirty


def load_bundle(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        result = {key: data[key] for key in data.files}
    required = {
        "pair_token",
        "cluster_id",
        "target",
        "per_seed_logits",
        "ensemble_logits",
        "design_stratum",
        "cardinality",
    }
    missing = required - set(result)
    if missing:
        raise RuntimeError(f"bundle missing keys: {sorted(missing)}")
    if not np.all(np.isfinite(result["ensemble_logits"])):
        raise RuntimeError("bundle contains non-finite logits")
    return result


def platt_probability(logits: np.ndarray, info: dict[str, Any]) -> np.ndarray:
    return expit(float(info["coefficient"]) * logits + float(info["intercept"]))


def exact_set_nll(set_mass: np.ndarray, target: np.ndarray) -> np.ndarray:
    index = target_set_indices(target)
    return -np.log(np.clip(set_mass[np.arange(len(index)), index], np.finfo(float).tiny, 1.0))


def set_metric_arrays(set_mass: np.ndarray, target: np.ndarray) -> dict[str, np.ndarray]:
    sets = nonempty_sets(4)
    marginal = marginal_probability(set_mass)
    clipped = np.clip(marginal, np.finfo(float).tiny, 1.0 - np.finfo(float).eps)
    card = sets.sum(axis=1)
    expected_cardinality = set_mass @ card
    return {
        "set_nll": exact_set_nll(set_mass, target),
        "set_accuracy": (np.argmax(set_mass, axis=1) == target_set_indices(target)).astype(float),
        "marginal_brier": np.mean((marginal - target) ** 2, axis=1),
        "marginal_nll": -np.mean(
            target * np.log(clipped) + (~target) * np.log1p(-clipped), axis=1
        ),
        "cardinality_abs_error": np.abs(expected_cardinality - target.sum(axis=1)),
    }


def action_metric_arrays(
    actions: np.ndarray,
    target: np.ndarray,
    utilities: np.ndarray,
    penalty: float,
) -> dict[str, np.ndarray]:
    oracle = authorized_actions(target, utilities)
    regret = constrained_regret(actions, target, utilities, penalty)
    compatible = target[np.arange(len(target))[:, None], actions]
    return {
        "accuracy": np.mean(actions == oracle, axis=1),
        "compatible": np.mean(compatible, axis=1),
        "regret": np.mean(regret, axis=1),
        "token_worst_regret": np.max(regret, axis=1),
    }


def cardinality_l1(set_mass: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    card = nonempty_sets(4).sum(axis=1)
    predicted = np.asarray([set_mass[mask][:, card == k].sum(axis=1).mean() for k in range(1, 5)])
    empirical = np.asarray([np.mean(target[mask].sum(axis=1) == k) for k in range(1, 5)])
    return float(np.abs(predicted - empirical).sum())


def summarize(
    set_metrics: dict[str, dict[str, np.ndarray]],
    action_metrics: dict[str, dict[str, np.ndarray]],
    set_mass: dict[str, np.ndarray],
    target: np.ndarray,
    mask: np.ndarray,
) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for arm in ACTION_ARMS:
        row = {
            key if key != "token_worst_regret" else "mean_token_worst_regret": float(values[mask].mean())
            for key, values in action_metrics[arm].items()
        }
        if arm in set_metrics:
            row.update({key: float(values[mask].mean()) for key, values in set_metrics[arm].items()})
            row["cardinality_distribution_l1"] = cardinality_l1(set_mass[arm], target, mask)
        output[arm] = row
    return output


def select_regularization(
    rows: list[dict[str, Any]], selector: str
) -> dict[str, Any]:
    key = f"{selector}_nll"
    return min(rows, key=lambda row: (row[key], -row["regularization"]))


def bootstrap_contrasts(
    set_metrics: dict[str, dict[str, np.ndarray]],
    action_metrics: dict[str, dict[str, np.ndarray]],
    primary_indices: np.ndarray,
    bootstrap_indices: np.ndarray,
    best_independent: str,
) -> dict[str, Any]:
    pairs = {
        "joint_full_minus_best_independent": ("joint_full", best_independent),
        "joint_full_minus_unary_cardinality": ("joint_full", "joint_unary_cardinality"),
        "joint_full_minus_hard": ("joint_full", "hard_set_policy"),
        "joint_full_minus_prior": ("joint_full", "empirical_set_prior"),
        "joint_full_minus_shuffled": ("joint_full", "joint_full_target_shuffled"),
    }
    output = {}
    for name, (left, right) in pairs.items():
        metrics = {}
        common = set(action_metrics[left]) & set(action_metrics[right])
        if left in set_metrics and right in set_metrics:
            common |= set(set_metrics[left]) & set(set_metrics[right])
        for metric in sorted(common):
            source = set_metrics if metric in set_metrics.get(left, {}) else action_metrics
            metrics[metric] = paired_delta_ci(
                source[left][metric][primary_indices],
                source[right][metric][primary_indices],
                bootstrap_indices,
            )
        output[name] = metrics
    return output


def fit_grid(
    fit: dict[str, np.ndarray],
    config: dict[str, Any],
    shuffled_target: np.ndarray,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[float, dict[str, Any]]]]:
    calibration = fit["split_role"].astype(str) == "calibration_fit"
    decision = fit["split_role"].astype(str) == "decision_select"
    primary = decision & (fit["design_stratum"].astype(str) == "NEAR_RIVAL") & (fit["cardinality"] >= 2)
    rows: dict[str, list[dict[str, Any]]] = {}
    models: dict[str, dict[float, dict[str, Any]]] = {}
    for structure in STRUCTURES:
        rows[structure] = []
        models[structure] = {}
        for regularization in config["regularization_grid"]:
            model = fit_joint_posterior(
                fit["ensemble_logits"][calibration],
                fit["target"][calibration],
                structure,
                float(regularization),
                max_iter=config["optimizer"]["max_iter"],
                gtol=config["optimizer"]["gtol"],
                ftol=config["optimizer"]["ftol"],
            )
            mass = posterior_mass(fit["ensemble_logits"][decision], model["theta"], structure)
            nll = exact_set_nll(mass, fit["target"][decision])
            row = {
                "regularization": float(regularization),
                "primary_nll": float(nll[primary[decision]].mean()),
                "global_nll": float(nll.mean()),
                "theta": model["theta"].tolist(),
                "interaction_coefficients": model["interaction_coefficients"].tolist(),
                "objective": model["objective"],
                "gradient_norm": model["gradient_norm"],
                "iterations": model["iterations"],
                "function_evaluations": model["function_evaluations"],
                "optimizer_message": model["message"],
            }
            rows[structure].append(row)
            models[structure][float(regularization)] = model

    selected_lambda = select_regularization(rows["joint_full"], "primary")["regularization"]
    shuffled_model = fit_joint_posterior(
        fit["ensemble_logits"][calibration],
        shuffled_target,
        "joint_full",
        float(selected_lambda),
        max_iter=config["optimizer"]["max_iter"],
        gtol=config["optimizer"]["gtol"],
        ftol=config["optimizer"]["ftol"],
    )
    models["joint_full_target_shuffled"] = {float(selected_lambda): shuffled_model}
    return rows, models


def stable_analysis_hash(arrays: list[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        canonical = np.ascontiguousarray(array)
        digest.update(str(canonical.dtype).encode())
        digest.update(str(canonical.shape).encode())
        digest.update(canonical.tobytes())
    return digest.hexdigest()


def compare_npz(reference: Path, candidate: Path) -> dict[str, Any]:
    with np.load(reference, allow_pickle=False) as left, np.load(candidate, allow_pickle=False) as right:
        if left.files != right.files:
            return {"exact": False, "reason": "key mismatch"}
        mismatches = [key for key in left.files if not np.array_equal(left[key], right[key], equal_nan=True)]
    return {"exact": not mismatches, "mismatches": mismatches}


def main() -> None:
    started = time.time()
    args = parse_args()
    commit, dirty = git_state()
    if dirty:
        raise RuntimeError("tracked worktree must be clean before running Wave 54")
    output = args.output_dir.resolve()
    reference = args.reference_dir.resolve(strict=True) if args.reference_dir else None
    config_path = args.config.resolve(strict=True)
    bundle = args.bundle_dir.resolve(strict=True)
    wave52 = args.wave52_dir.resolve(strict=True)
    wave53 = args.wave53_dir.resolve(strict=True)
    inputs = [bundle, wave52, wave53, config_path]
    if reference is not None:
        inputs.append(reference)
    reject_lockbox_paths([*inputs, output])
    execution_sources = [Path(__file__), PRIMITIVES_PATH, PLAN_PATH, config_path]
    require_sources_at_head(execution_sources)
    validate_output_path(output, inputs, execution_sources)
    os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    bundle_binding = config.get("bundle_binding")
    if not bundle_binding:
        raise RuntimeError("bundle_binding is not frozen in config")
    manifest_receipt = require_hash(
        bundle / "input_bundle_manifest.json", bundle_binding["manifest_sha256"]
    )
    policy_receipt = require_hash(
        wave52 / "policy_manifest.json", config["source_binding"]["wave52_policy_manifest_sha256"]
    )
    platt_receipt = require_hash(
        wave53 / "platt_calibrator.json", config["source_binding"]["wave53_platt_calibrator_sha256"]
    )
    archived_output = prepare_output_directory(output, args.force)
    shutil.copy2(config_path, output / "frozen_config.json")
    expected_fit = {
        "path": str(bundle / "fit_select_bundle.npz"),
        "sha256": bundle_binding["fit_select_bundle_sha256"],
    }
    expected_monitor = {
        "path": str(bundle / "sealed_monitor_bundle.npz"),
        "sha256": bundle_binding["sealed_monitor_bundle_sha256"],
    }
    analysis_freeze = {
        "chronology": "written-before-fit-select-or-sealed-monitor-bundle-access",
        "git_commit": commit,
        "config_sha256": sha256_file(config_path),
        "execution_sources": [
            file_receipt(path) for path in execution_sources
        ],
        "bundle_manifest": manifest_receipt,
        "expected_fit_select": expected_fit,
        "expected_sealed_monitor": expected_monitor,
        "policy_manifest": policy_receipt,
        "platt_calibrator": platt_receipt,
        "structures": list(STRUCTURES),
        "regularization_grid": config["regularization_grid"],
        "diagnostic_criteria": config["diagnostic_criteria"],
        "fit_select_accessed": False,
        "sealed_monitor_accessed": False,
        "lockbox_accessed": False,
        "superseded_output": str(archived_output) if archived_output else None,
    }
    write_json(output / "analysis_freeze.json", analysis_freeze)

    fit_receipt = require_hash(
        bundle / "fit_select_bundle.npz", bundle_binding["fit_select_bundle_sha256"]
    )
    fit = load_bundle(bundle / "fit_select_bundle.npz")
    calibration = fit["split_role"].astype(str) == "calibration_fit"
    decision = fit["split_role"].astype(str) == "decision_select"
    decision_primary = decision & (fit["design_stratum"].astype(str) == "NEAR_RIVAL") & (fit["cardinality"] >= 2)
    shuffle_local = deranged_within_strata(
        fit["pair_token"][calibration].astype(str),
        list(zip(fit["design_stratum"][calibration].astype(str), fit["cardinality"][calibration], strict=True)),
        int(config["shuffle_seed"]),
    )
    shuffled_target = fit["target"][calibration][shuffle_local]
    grid_rows, fitted = fit_grid(fit, config, shuffled_target)

    platt_info = json.loads((wave53 / "platt_calibrator.json").read_text(encoding="utf-8"))
    decision_logits = fit["ensemble_logits"][decision]
    decision_target = fit["target"][decision]
    decision_primary_local = decision_primary[decision]
    independent_decision = {
        "independent_raw": independent_nonempty_mass(expit(decision_logits))[1],
        "independent_platt": independent_nonempty_mass(platt_probability(decision_logits, platt_info))[1],
    }
    independent_nll = {
        arm: float(exact_set_nll(mass, decision_target)[decision_primary_local].mean())
        for arm, mass in independent_decision.items()
    }
    best_independent = min(independent_nll, key=lambda arm: (independent_nll[arm], arm != "independent_raw"))
    selections = {
        structure: {
            "primary": select_regularization(rows, "primary"),
            "global_sensitivity": select_regularization(rows, "global"),
        }
        for structure, rows in grid_rows.items()
    }
    selected_models = {
        structure: fitted[structure][selections[structure]["primary"]["regularization"]]
        for structure in STRUCTURES
    }
    selected_models["joint_full_target_shuffled"] = fitted["joint_full_target_shuffled"][
        selections["joint_full"]["primary"]["regularization"]
    ]
    model_summary = {}
    for structure, model in selected_models.items():
        model_summary[structure] = {
            "theta": model["theta"].tolist(),
            "interaction_coefficients": model["interaction_coefficients"].tolist(),
            "interaction_l2": float(np.linalg.norm(model["interaction_coefficients"])),
            "objective": model["objective"],
            "gradient_norm": model["gradient_norm"],
            "iterations": model["iterations"],
        }
    selection_freeze = {
        "chronology": "written-before-sealed-monitor-bundle-access",
        "best_independent": best_independent,
        "independent_primary_nll": independent_nll,
        "grid": grid_rows,
        "selections": selections,
        "selected_models": model_summary,
        "shuffled_source_index_within_calibration": shuffle_local.tolist(),
        "expected_sealed_monitor": expected_monitor,
        "sealed_monitor_accessed": False,
    }
    write_json(output / "selection_freeze.json", selection_freeze)

    monitor_receipt = require_hash(
        bundle / "sealed_monitor_bundle.npz", bundle_binding["sealed_monitor_bundle_sha256"]
    )
    monitor = load_bundle(bundle / "sealed_monitor_bundle.npz")
    validate_bundle_separation(fit, monitor)
    with open(wave52 / "policy_manifest.json", encoding="utf-8") as handle:
        policy_manifest = json.load(handle)
    utilities = np.asarray(policy_manifest["levels"], dtype=np.float64)[
        np.asarray(policy_manifest["rank_permutations"], dtype=np.int64)
    ]
    penalty = float(config["incompatible_regret_penalty"])
    raw_probability = expit(monitor["ensemble_logits"])
    platt_probability_monitor = platt_probability(monitor["ensemble_logits"], platt_info)
    set_mass = {
        "independent_raw": independent_nonempty_mass(raw_probability)[1],
        "independent_platt": independent_nonempty_mass(platt_probability_monitor)[1],
    }
    for structure in STRUCTURES:
        set_mass[structure] = posterior_mass(
            monitor["ensemble_logits"], selected_models[structure]["theta"], structure
        )
    prior = empirical_set_prior(fit["target"][calibration], float(config["empirical_prior_alpha"]))
    set_mass["empirical_set_prior"] = np.repeat(prior[None, :], len(monitor["target"]), axis=0)
    set_mass["joint_full_target_shuffled"] = posterior_mass(
        monitor["ensemble_logits"],
        selected_models["joint_full_target_shuffled"]["theta"],
        "joint_full",
    )

    posterior_decisions = {
        arm: expected_regret_from_mass(mass, utilities, penalty)
        for arm, mass in set_mass.items()
    }
    hard_actions, hard_fallback = explicit_set_actions(
        raw_probability, utilities, float(config["hard_set_tau"])
    )
    actions = {arm: result["actions"] for arm, result in posterior_decisions.items()}
    actions["hard_set_policy"] = hard_actions
    actions["oracle_set_then_utility"] = authorized_actions(monitor["target"], utilities)
    set_metrics = {arm: set_metric_arrays(mass, monitor["target"]) for arm, mass in set_mass.items()}
    action_metrics = {
        arm: action_metric_arrays(action, monitor["target"], utilities, penalty)
        for arm, action in actions.items()
    }
    primary = (monitor["design_stratum"].astype(str) == "NEAR_RIVAL") & (monitor["cardinality"] >= 2)
    all_catalog = monitor["target"].any(axis=1)
    summaries = {
        "primary_near_rival_cardinality_ge2": summarize(
            set_metrics, action_metrics, set_mass, monitor["target"], primary
        ),
        "all_in_catalog": summarize(set_metrics, action_metrics, set_mass, monitor["target"], all_catalog),
    }
    primary_indices = np.flatnonzero(primary)
    bootstrap_indices = paired_bootstrap_indices(
        len(primary_indices), int(config["n_boot"]), int(config["bootstrap_seed"])
    )
    contrasts = bootstrap_contrasts(
        set_metrics, action_metrics, primary_indices, bootstrap_indices, best_independent
    )

    observed_fit = np.zeros(15, dtype=bool)
    observed_fit[np.unique(target_set_indices(fit["target"][calibration]))] = True
    unseen = ~observed_fit
    joint_decision_mass = posterior_mass(
        fit["ensemble_logits"][decision], selected_models["joint_full"]["theta"], "joint_full"
    )
    unseen_mass = {
        "unseen_set_indices": np.flatnonzero(unseen).tolist(),
        "decision_select": {
            "joint_full": float(joint_decision_mass[:, unseen].sum(axis=1).mean()),
            "independent_raw": float(independent_decision["independent_raw"][:, unseen].sum(axis=1).mean()),
        },
        "val_monitor": {
            "joint_full": float(set_mass["joint_full"][:, unseen].sum(axis=1).mean()),
            "independent_raw": float(set_mass["independent_raw"][:, unseen].sum(axis=1).mean()),
        },
        "observed_fit_count": int(observed_fit.sum()),
    }

    global_lambda = selections["joint_full"]["global_sensitivity"]["regularization"]
    global_model = fitted["joint_full"][global_lambda]
    global_mass = posterior_mass(monitor["ensemble_logits"], global_model["theta"], "joint_full")
    global_set_metrics = set_metric_arrays(global_mass, monitor["target"])
    global_actions = expected_regret_from_mass(global_mass, utilities, penalty)["actions"]
    global_action_metrics = action_metric_arrays(global_actions, monitor["target"], utilities, penalty)
    def mean(values: np.ndarray) -> float:
        return float(values[primary].mean())

    sensitivity_values = {
        "nll_vs_best_independent": (
            mean(set_metrics[best_independent]["set_nll"])
            - mean(set_metrics["joint_full"]["set_nll"]),
            mean(set_metrics[best_independent]["set_nll"])
            - mean(global_set_metrics["set_nll"]),
        ),
        "cardinality_l1_vs_best_independent": (
            cardinality_l1(set_mass[best_independent], monitor["target"], primary)
            - cardinality_l1(set_mass["joint_full"], monitor["target"], primary),
            cardinality_l1(set_mass[best_independent], monitor["target"], primary)
            - cardinality_l1(global_mass, monitor["target"], primary),
        ),
        "nll_vs_unary_cardinality": (
            mean(set_metrics["joint_unary_cardinality"]["set_nll"])
            - mean(set_metrics["joint_full"]["set_nll"]),
            mean(set_metrics["joint_unary_cardinality"]["set_nll"])
            - mean(global_set_metrics["set_nll"]),
        ),
        "regret_vs_hard": (
            mean(action_metrics["hard_set_policy"]["regret"])
            - mean(action_metrics["joint_full"]["regret"]),
            mean(action_metrics["hard_set_policy"]["regret"])
            - mean(global_action_metrics["regret"]),
        ),
        "accuracy_vs_hard": (
            mean(action_metrics["joint_full"]["accuracy"])
            - mean(action_metrics["hard_set_policy"]["accuracy"]),
            mean(global_action_metrics["accuracy"])
            - mean(action_metrics["hard_set_policy"]["accuracy"]),
        ),
        "compatible_vs_hard": (
            mean(action_metrics["joint_full"]["compatible"])
            - mean(action_metrics["hard_set_policy"]["compatible"]),
            mean(global_action_metrics["compatible"])
            - mean(action_metrics["hard_set_policy"]["compatible"]),
        ),
        "accuracy_vs_empirical_prior": (
            mean(action_metrics["joint_full"]["accuracy"])
            - mean(action_metrics["empirical_set_prior"]["accuracy"]),
            mean(global_action_metrics["accuracy"])
            - mean(action_metrics["empirical_set_prior"]["accuracy"]),
        ),
        "accuracy_vs_target_shuffled": (
            mean(action_metrics["joint_full"]["accuracy"])
            - mean(action_metrics["joint_full_target_shuffled"]["accuracy"]),
            mean(global_action_metrics["accuracy"])
            - mean(action_metrics["joint_full_target_shuffled"]["accuracy"]),
        ),
    }
    sensitivity_signs = {
        name: {
            "primary_selected_advantage": primary_value,
            "global_selected_advantage": global_value,
            "sign_changed": bool(np.sign(primary_value) != np.sign(global_value)),
        }
        for name, (primary_value, global_value) in sensitivity_values.items()
    }
    selector_sensitive = any(row["sign_changed"] for row in sensitivity_signs.values())

    boundaries = {}
    selective = {}
    coverage_masks = {}
    fit_decision_mass = posterior_mass(
        fit["ensemble_logits"][decision], selected_models["joint_full"]["theta"], "joint_full"
    )
    fit_decision = expected_regret_from_mass(fit_decision_mass, utilities, penalty)
    decision_score = np.max(fit_decision["minimum_risk"], axis=1)
    monitor_score = np.max(posterior_decisions["joint_full"]["minimum_risk"], axis=1)
    for coverage in config["selection_coverages"]:
        boundary = coverage_boundary(
            decision_score[decision_primary_local],
            fit["pair_token"][decision][decision_primary_local].astype(str),
            float(coverage),
            int(config["selection_tie_seed"]),
        )
        accepted_primary = apply_coverage_boundary(
            monitor_score[primary],
            monitor["pair_token"][primary].astype(str),
            boundary,
            int(config["selection_tie_seed"]),
        )
        accepted = np.zeros(len(primary), dtype=bool)
        accepted[primary] = accepted_primary
        key = f"coverage_{coverage:.2f}"
        boundaries[key] = boundary
        coverage_masks[key] = accepted
        selective[key] = {
            "coverage": float(accepted_primary.mean()),
            "regret": float(action_metrics["joint_full"]["regret"][accepted].mean()),
            "accuracy": float(action_metrics["joint_full"]["accuracy"][accepted].mean()),
            "compatible": float(action_metrics["joint_full"]["compatible"][accepted].mean()),
        }
    selective["aurc"] = discrete_aurc(
        monitor_score[primary],
        action_metrics["joint_full"]["regret"][primary],
        monitor["pair_token"][primary].astype(str),
    )

    criteria = config["diagnostic_criteria"]
    nll_delta = contrasts["joint_full_minus_best_independent"]["set_nll"]
    interaction_delta = contrasts["joint_full_minus_unary_cardinality"]["set_nll"]
    hard_delta = contrasts["joint_full_minus_hard"]
    prior_delta = contrasts["joint_full_minus_prior"]["accuracy"]
    shuffled_delta = contrasts["joint_full_minus_shuffled"]["accuracy"]
    primary_summary = summaries["primary_near_rival_cardinality_ge2"]
    replay_exact = reference is not None
    checks = {
        "joint_nll_advantage": nll_delta["mean_diff"] <= -criteria["joint_nll_reduction_min"] and nll_delta["ci95_high"] < 0,
        "cardinality_l1_advantage": primary_summary["joint_full"]["cardinality_distribution_l1"] <= primary_summary[best_independent]["cardinality_distribution_l1"] - criteria["cardinality_l1_reduction_min"],
        "interaction_nll_advantage": interaction_delta["mean_diff"] <= -criteria["interaction_nll_reduction_min"] and interaction_delta["ci95_high"] < 0,
        "regret_advantage_over_hard": hard_delta["regret"]["mean_diff"] <= -criteria["regret_reduction_min"] and hard_delta["regret"]["ci95_high"] < 0,
        "accuracy_noninferiority_to_hard": hard_delta["accuracy"]["mean_diff"] >= criteria["accuracy_noninferiority"],
        "compatible_noninferiority_to_hard": hard_delta["compatible"]["mean_diff"] >= criteria["compatible_rate_noninferiority"],
        "prior_control": prior_delta["mean_diff"] >= criteria["control_accuracy_improvement_min"],
        "shuffled_control": shuffled_delta["mean_diff"] >= criteria["control_accuracy_improvement_min"],
        "unseen_mass_control": unseen_mass["decision_select"]["joint_full"] <= unseen_mass["decision_select"]["independent_raw"] + config["unseen_mass_allowance"],
        "selector_not_sign_sensitive": not selector_sensitive,
        "external_replay_exact": replay_exact,
    }

    np.savez_compressed(
        output / "posterior_state.npz",
        pair_token=monitor["pair_token"],
        target=monitor["target"],
        utilities=utilities,
        **{f"mass__{arm}": mass for arm, mass in set_mass.items()},
        **{f"actions__{arm}": action for arm, action in actions.items()},
        **{
            f"action_risk__{arm}": result["action_risk"]
            for arm, result in posterior_decisions.items()
        },
        **{
            f"minimum_risk__{arm}": result["minimum_risk"]
            for arm, result in posterior_decisions.items()
        },
        **{
            f"decision_margin__{arm}": result["margin"]
            for arm, result in posterior_decisions.items()
        },
        **{
            f"regret_by_policy__{arm}": constrained_regret(
                action, monitor["target"], utilities, penalty
            )
            for arm, action in actions.items()
        },
        **{
            f"compatible_by_policy__{arm}": monitor["target"][
                np.arange(len(monitor["target"]))[:, None], action
            ]
            for arm, action in actions.items()
        },
        monitor_selection_score=monitor_score,
        **{f"accepted__{name}": mask for name, mask in coverage_masks.items()},
        hard_fallback=hard_fallback,
    )
    np.savez_compressed(
        output / "bootstrap_indices.npz",
        pair_token=monitor["pair_token"][primary_indices],
        indices=bootstrap_indices,
        seed=np.asarray(config["bootstrap_seed"], dtype=np.int64),
    )
    np.savez_compressed(
        output / "selection_state.npz",
        **{f"theta__{name}": model["theta"] for name, model in selected_models.items()},
        **{
            f"grid_theta__{structure}__lambda_{regularization:g}".replace(".", "p"): model["theta"]
            for structure, by_lambda in fitted.items()
            if structure in STRUCTURES
            for regularization, model in by_lambda.items()
        },
        empirical_prior=prior,
        observed_fit=observed_fit,
        shuffled_source_index=shuffle_local,
    )
    np.savez_compressed(
        output / "metrics_state.npz",
        pair_token=monitor["pair_token"],
        primary_mask=primary,
        **{
            f"set__{arm}__{metric}": values
            for arm, metrics in set_metrics.items()
            for metric, values in metrics.items()
        },
        **{
            f"action__{arm}__{metric}": values
            for arm, metrics in action_metrics.items()
            for metric, values in metrics.items()
        },
    )
    analysis_hash = stable_analysis_hash(
        [
            *(set_mass[arm] for arm in POSTERIOR_ARMS),
            *(actions[arm] for arm in ACTION_ARMS),
            *(set_metrics[arm][metric] for arm in sorted(set_metrics) for metric in sorted(set_metrics[arm])),
            *(action_metrics[arm][metric] for arm in sorted(action_metrics) for metric in sorted(action_metrics[arm])),
            bootstrap_indices,
            *(selected_models[name]["theta"] for name in sorted(selected_models)),
        ]
    )
    summary = {
        "status": "EXECUTED_CPU_ONLY_OPENED_HISTORICAL",
        "git_commit": commit,
        "analysis_hash": analysis_hash,
        "best_independent": best_independent,
        "summaries": summaries,
        "contrasts": contrasts,
        "unseen_mass": unseen_mass,
        "selector_sensitivity": {"sensitive": selector_sensitive, "contrasts": sensitivity_signs},
        "selective": selective,
        "selection_boundaries": boundaries,
        "diagnostic_checks": checks,
        "joint_set_posterior_promising": bool(all(checks.values())),
        "automatic_go": False,
        "scope": "historical observed set support; no claim for unseen true sets",
        "runtime_seconds": time.time() - started,
        "lockbox_accessed": False,
        "gpu_used": False,
    }
    write_json(output / "fit_diagnostics.json", {"grid": grid_rows, "selected_models": model_summary})
    write_json(
        output / "access_receipt.json",
        {
            "analysis_freeze_sha256": sha256_file(output / "analysis_freeze.json"),
            "selection_freeze_sha256": sha256_file(output / "selection_freeze.json"),
            "ordered_access": [fit_receipt, monitor_receipt],
            "monitor_opened_after_selection_freeze": True,
            "lockbox_accessed": False,
        },
    )
    write_json(output / "summary.json", summary)
    write_json(
        output / "package_manifest.json",
        {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": {
                name: importlib.metadata.version(name)
                for name in ("numpy", "scipy", "scikit-learn")
            },
        },
    )

    replay_receipt = {"reference_provided": reference is not None, "exact": False}
    if reference is not None:
        comparisons = {
            name: compare_npz(reference / name, output / name)
            for name in (
                "posterior_state.npz",
                "bootstrap_indices.npz",
                "selection_state.npz",
                "metrics_state.npz",
            )
        }
        reference_summary = json.loads((reference / "summary.json").read_text(encoding="utf-8"))
        replay_receipt = {
            "reference_provided": True,
            "npz": comparisons,
            "analysis_hash_matches": reference_summary["analysis_hash"] == analysis_hash,
        }
        replay_receipt["exact"] = replay_receipt["analysis_hash_matches"] and all(
            row["exact"] for row in comparisons.values()
        )
        checks["external_replay_exact"] = replay_receipt["exact"]
        summary["diagnostic_checks"] = checks
        summary["joint_set_posterior_promising"] = bool(all(checks.values()))
        write_json(output / "summary.json", summary)
    write_json(output / "replay_receipt.json", replay_receipt)

    report = [
        "# Ola 54 — posterior conjunto regularizado",
        "",
        f"- pattern conjunto: `{summary['joint_set_posterior_promising']}`",
        f"- best independent congelado: `{best_independent}`",
        f"- selector-sensitive: `{selector_sensitive}`",
        f"- analysis hash: `{analysis_hash}`",
        "",
        "## Población primaria",
        "",
        "| brazo | set NLL | set acc | card L1 | action acc | compatible | regret |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ACTION_ARMS:
        row = primary_summary[arm]
        report.append(
            f"| `{arm}` | {row.get('set_nll', float('nan')):.4f} | {row.get('set_accuracy', float('nan')):.4f} | "
            f"{row.get('cardinality_distribution_l1', float('nan')):.4f} | {row['accuracy']:.4f} | "
            f"{row['compatible']:.4f} | {row['regret']:.4f} |"
        )
    report.extend(
        [
            "",
            "## Alcance",
            "",
            "Smoke CPU sobre logits históricos congelados. El resultado se restringe al soporte de conjuntos observado; no usa lockbox, no revalida el encoder upstream y no decide GO/NO-GO.",
        ]
    )
    (output / "REPORT_WAVE54_JOINT_SET.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    artifact_files = sorted(path for path in output.iterdir() if path.name != "artifact_manifest.json")
    write_json(
        output / "artifact_manifest.json",
        {
            "analysis_hash": analysis_hash,
            "files": [{"path": path.name, "sha256": sha256_file(path)} for path in artifact_files],
        },
    )
    print(json.dumps({"analysis_hash": analysis_hash, "pattern": summary["joint_set_posterior_promising"]}, sort_keys=True))


if __name__ == "__main__":
    main()
