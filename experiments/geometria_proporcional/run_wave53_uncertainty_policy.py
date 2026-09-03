#!/usr/bin/env python3
"""Run the Wave 53 CPU-only uncertainty-aware ordinal policy smoke."""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import scipy
import sklearn
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from geometria_proporcional.wave49_schema import sha256_file  # noqa: E402
from geometria_proporcional.wave50_neural import stable_hash  # noqa: E402
from geometria_proporcional.wave52_policy import (  # noqa: E402
    authorized_actions,
    constrained_regret,
    explicit_set_actions,
    score_composition_actions,
)
from geometria_proporcional.wave53_uncertainty import (  # noqa: E402
    apply_coverage_boundary,
    coverage_boundary,
    deranged_within_strata,
    discrete_aurc,
    expected_regret_actions,
    independent_nonempty_mass,
    paired_bootstrap_indices,
    paired_delta_ci,
    stratified_token_split,
)

ARMS = (
    "hard_set_policy",
    "score_composition",
    "marginal_expected_regret",
    "raw_expected_regret",
    "probability_shuffled",
    "utility_masked",
    "oracle_set_then_utility",
)

PAIR_TOKEN_PATTERN = re.compile(r'"pair_token"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wave50-dir", type=Path, required=True)
    parser.add_argument("--wave52-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--reference-dir",
        type=Path,
        help="Independent prior run to compare for exact external replay.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT
        / "experiments/geometria_proporcional/configs/wave53_uncertainty_policy.json",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def validate_output_path(output_dir: Path, inputs: list[Path]) -> None:
    output = output_dir.resolve()
    root = (REPO_ROOT / "data/geometria_proporcional").resolve()
    if root not in output.parents:
        raise ValueError(f"output must be a child of {root}")
    for raw in inputs:
        source = raw.resolve()
        if output == source or output in source.parents or source in output.parents:
            raise ValueError("output must be disjoint from all input trees")


def require_hash(path: Path, expected: str) -> dict[str, str]:
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"source binding mismatch for {path}: {actual} != {expected}")
    if "lockbox" in str(path.resolve()).lower():
        raise ValueError(f"Wave 53 cannot read lockbox content: {path}")
    return {"path": str(path), "sha256": actual}


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def execution_sources(config_path: Path) -> list[dict[str, str]]:
    paths = [
        Path(__file__).resolve(),
        (REPO_ROOT / "src/geometria_proporcional/wave49_schema.py").resolve(),
        (REPO_ROOT / "src/geometria_proporcional/wave50_neural.py").resolve(),
        (REPO_ROOT / "src/geometria_proporcional/wave52_policy.py").resolve(),
        (REPO_ROOT / "src/geometria_proporcional/wave53_uncertainty.py").resolve(),
        (
            REPO_ROOT / "Biblioteca/Geometria_Proporcional_Ground_Truth/waves/"
            "WAVE_53_UNCERTAINTY_AWARE_POLICY_PLAN.md"
        ).resolve(),
        config_path.resolve(),
    ]
    rows = []
    for path in paths:
        relative = path.relative_to(REPO_ROOT)
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", str(relative)],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        if tracked.returncode != 0:
            raise RuntimeError(f"execution source is not tracked: {relative}")
        dirty = subprocess.run(
            ["git", "diff", "--quiet", "HEAD", "--", str(relative)], cwd=REPO_ROOT
        )
        if dirty.returncode != 0:
            raise RuntimeError(f"execution source differs from HEAD: {relative}")
        rows.append({"path": str(relative), "sha256": sha256_file(path)})
    return rows


def validate_sources(
    wave52: Path, config: dict[str, Any]
) -> tuple[list[dict[str, str]], dict[str, str]]:
    """Validate only manifests that are safe to inspect before the analysis freeze."""
    binding = config["source_binding"]
    checked = [
        require_hash(
            wave52 / "artifact_manifest.json",
            binding["wave52_artifact_manifest_sha256"],
        ),
        require_hash(
            wave52 / "package_manifest.json",
            binding["wave52_package_manifest_sha256"],
        ),
        require_hash(wave52 / "summary.json", binding["wave52_summary_sha256"]),
        require_hash(
            wave52 / "policy_manifest.json",
            binding["wave52_policy_manifest_sha256"],
        ),
        require_hash(
            wave52 / "split_manifest.json",
            binding["wave52_split_manifest_sha256"],
        ),
    ]
    manifest = json.loads((wave52 / "artifact_manifest.json").read_text())
    file_hashes = {row["path"]: row["sha256"] for row in manifest["files"]}
    return checked, file_hashes


def validate_split_sources(
    wave52: Path,
    split: str,
    seeds: list[int],
    file_hashes: dict[str, str],
) -> list[dict[str, str]]:
    """Hash-check one prediction split only after its access is authorized by the freeze."""
    checked = []
    for seed in seeds:
        relative = f"raw_eval/frozen_set/seed{seed}__{split}.npz"
        if relative not in file_hashes:
            raise ValueError(f"artifact manifest does not bind {relative}")
        checked.append(require_hash(wave52 / relative, file_hashes[relative]))
    return checked


def build_access_receipt(
    prefreeze: list[dict[str, str]],
    label: dict[str, str],
    threshold: list[dict[str, str]],
    monitor: list[dict[str, str]],
    analysis_freeze_sha256: str,
    split_manifest_sha256: str,
) -> dict[str, Any]:
    """Preserve staged access chronology and one flat source-binding inventory."""
    stages = {
        "before_analysis_freeze": prefreeze,
        "after_analysis_freeze_before_split_freeze": [label, *threshold],
        "after_split_freeze": monitor,
    }
    return {
        "stages": stages,
        "files_read": [row for rows in stages.values() for row in rows],
        "analysis_freeze_sha256": analysis_freeze_sha256,
        "split_manifest_sha256": split_manifest_sha256,
        "lockbox_accessed": False,
    }


def load_metadata(path: Path, allowed_tokens: set[str]) -> dict[str, dict[str, Any]]:
    """Load only canonical rows for explicitly allowed tokens.

    The token is extracted before JSON parsing so rows from a later frozen split are not
    semantically inspected while preparing an earlier split.
    """
    metadata: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            match = PAIR_TOKEN_PATTERN.search(line)
            if match is None:
                continue
            token = json.loads(f'"{match.group(1)}"')
            if token not in allowed_tokens:
                continue
            row = json.loads(line)
            if row.get("calibration_population") != "canonical_preserving":
                continue
            if row.get("pair_token") != token:
                raise ValueError("pair_token prefilter disagrees with parsed JSON")
            current = {
                "design_stratum": row["design_stratum"],
                "cardinality": len(row["oracle_compatible_set"]),
            }
            if token in metadata and metadata[token] != current:
                raise ValueError(f"inconsistent metadata for {token}")
            metadata[token] = current
    if set(metadata) != allowed_tokens:
        missing = sorted(allowed_tokens - set(metadata))
        raise ValueError(f"metadata missing {len(missing)} allowed tokens")
    return metadata


def load_split(wave52: Path, split: str, seeds: list[int]) -> dict[str, np.ndarray]:
    per_seed = []
    for seed in seeds:
        with np.load(
            wave52 / "raw_eval/frozen_set" / f"seed{seed}__{split}.npz"
        ) as data:
            per_seed.append({key: data[key].copy() for key in data.files})
    base = per_seed[0]
    for row in per_seed[1:]:
        for key in ("pair_token", "cluster_id", "target"):
            if not np.array_equal(row[key], base[key]):
                raise RuntimeError(f"seed alignment failed for {split}/{key}")
    return {
        "pair_token": base["pair_token"],
        "cluster_id": base["cluster_id"],
        "target": base["target"].astype(bool),
        "per_seed_logits": np.stack([row["set_logits"] for row in per_seed]),
        "ensemble_logits": np.mean(
            np.stack([row["set_logits"] for row in per_seed]), axis=0
        ),
    }


def attach_metadata(
    split: dict[str, np.ndarray], metadata: dict[str, dict[str, Any]]
) -> None:
    rows = []
    for token, target in zip(split["pair_token"], split["target"], strict=True):
        token = str(token)
        if token not in metadata:
            raise ValueError(f"missing source metadata for {token}")
        row = metadata[token]
        if int(np.sum(target)) != row["cardinality"]:
            raise ValueError(f"target cardinality mismatch for {token}")
        rows.append(row)
    split["design_stratum"] = np.asarray([row["design_stratum"] for row in rows])
    split["cardinality"] = np.asarray(
        [row["cardinality"] for row in rows], dtype=np.int64
    )


def fit_platt(
    logits: np.ndarray, target: np.ndarray, config: dict[str, Any]
) -> tuple[LogisticRegression, dict[str, Any]]:
    x = np.asarray(logits, dtype=np.float64).reshape(-1, 1)
    y = np.asarray(target, dtype=np.int64).ravel()
    if len(np.unique(y)) != 2:
        raise RuntimeError("Platt calibration requires both classes")
    model = LogisticRegression(
        C=float(config["C"]),
        penalty="l2",
        solver=config["solver"],
        max_iter=int(config["max_iter"]),
        class_weight=config["class_weight"],
        random_state=5301,
    ).fit(x, y)
    if int(model.n_iter_[0]) >= int(config["max_iter"]):
        raise RuntimeError("Platt calibration did not converge")
    return model, {
        "coefficient": float(model.coef_[0, 0]),
        "intercept": float(model.intercept_[0]),
        "n_token_family_rows": len(y),
        "positive_fraction": float(np.mean(y)),
        "n_iter": int(model.n_iter_[0]),
        "recipe": config,
    }


def platt_probability(model: LogisticRegression, logits: np.ndarray) -> np.ndarray:
    shape = np.asarray(logits).shape
    return model.predict_proba(np.asarray(logits).reshape(-1, 1))[:, 1].reshape(shape)


def token_metric_arrays(
    actions: np.ndarray,
    targets: np.ndarray,
    utilities: np.ndarray,
    penalty: float,
) -> dict[str, np.ndarray]:
    truth = authorized_actions(targets, utilities)
    regret = constrained_regret(actions, targets, utilities, penalty)
    compatible = targets[np.arange(len(targets))[:, None], actions]
    return {
        "accuracy": np.mean(actions == truth, axis=1),
        "compatible": np.mean(compatible, axis=1),
        "regret": np.mean(regret, axis=1),
        "token_worst_regret": np.max(regret, axis=1),
    }


def summarize_metrics(
    metrics: dict[str, dict[str, np.ndarray]], mask: np.ndarray
) -> dict[str, dict[str, float]]:
    return {
        arm: {
            (
                "mean_token_worst_regret" if name == "token_worst_regret" else name
            ): float(np.mean(values[mask]))
            for name, values in row.items()
        }
        for arm, row in metrics.items()
    }


def calibration_metrics(probability: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    probability = np.clip(np.asarray(probability, dtype=np.float64), 1e-12, 1 - 1e-12)
    target = np.asarray(target, dtype=np.float64)
    if probability.shape != target.shape or probability.ndim != 2:
        raise ValueError(
            "calibration arrays must have aligned [tokens, families] shape"
        )
    if len(probability) == 0:
        raise ValueError("calibration population cannot be empty")
    brier = np.mean((probability - target) ** 2)
    nll = -np.mean(
        target * np.log(probability) + (1 - target) * np.log(1 - probability)
    )
    _, set_mass = independent_nonempty_mass(probability)
    set_cardinality = np.sum(
        np.arange(1, 16, dtype=np.uint64)[:, None]
        >> np.arange(4, dtype=np.uint64)[None, :]
        & 1,
        axis=1,
    )
    predicted_dist = {
        str(k): float(np.mean(np.sum(set_mass[:, set_cardinality == k], axis=1)))
        for k in range(1, 5)
    }
    true_cardinality = target.sum(axis=1).astype(int)
    empirical_dist = {
        str(k): float(np.mean(true_cardinality == k)) for k in range(1, 5)
    }
    residual = target - probability
    correlations: list[list[float | None]] = []
    finite_off_diagonal = []
    for left in range(residual.shape[1]):
        row = []
        for right in range(residual.shape[1]):
            if np.std(residual[:, left]) == 0.0 or np.std(residual[:, right]) == 0.0:
                value = None
            else:
                value = float(np.corrcoef(residual[:, left], residual[:, right])[0, 1])
                if left < right:
                    finite_off_diagonal.append(abs(value))
            row.append(value)
        correlations.append(row)
    return {
        "brier": float(brier),
        "nll": float(nll),
        "expected_cardinality_mae": float(
            np.mean(np.abs(probability.sum(axis=1) - true_cardinality))
        ),
        "predicted_cardinality_distribution": predicted_dist,
        "empirical_cardinality_distribution": empirical_dist,
        "cardinality_distribution_l1": float(
            sum(
                abs(predicted_dist[str(k)] - empirical_dist[str(k)])
                for k in range(1, 5)
            )
        ),
        "residual_membership_correlation": correlations,
        "max_abs_offdiagonal_residual_correlation": (
            float(max(finite_off_diagonal)) if finite_off_diagonal else None
        ),
    }


def select_lambda(
    logits: np.ndarray,
    targets: np.ndarray,
    utilities: np.ndarray,
    mask: np.ndarray,
    grid: list[float],
    penalty: float,
) -> dict[str, Any]:
    candidates = []
    truth = authorized_actions(targets, utilities)
    for weight in grid:
        action = score_composition_actions(logits, utilities, weight)
        compatible = targets[np.arange(len(targets))[:, None], action]
        regret = constrained_regret(action, targets, utilities, penalty)
        candidates.append(
            {
                "weight": float(weight),
                "accuracy": float(np.mean((action == truth)[mask])),
                "compatible": float(np.mean(compatible[mask])),
                "regret": float(np.mean(regret[mask])),
            }
        )
    selected = max(
        candidates,
        key=lambda row: (
            row["accuracy"],
            row["compatible"],
            -row["regret"],
            -row["weight"],
        ),
    )
    return {"selected": selected, "grid": candidates}


def selective_metrics(
    metric: dict[str, np.ndarray], accepted: np.ndarray, eligible: np.ndarray
) -> dict[str, float]:
    if accepted.shape != eligible.shape or np.any(accepted & ~eligible):
        raise ValueError("accepted tokens must be a subset of the eligible population")
    if not np.any(accepted):
        return {
            "coverage": 0.0,
            "accuracy": float("nan"),
            "compatible": float("nan"),
            "regret": float("nan"),
            "global_worst_regret": float("nan"),
        }
    return {
        "coverage": float(np.sum(accepted) / np.sum(eligible)),
        "accuracy": float(np.mean(metric["accuracy"][accepted])),
        "compatible": float(np.mean(metric["compatible"][accepted])),
        "regret": float(np.mean(metric["regret"][accepted])),
        "global_worst_regret": float(np.max(metric["token_worst_regret"][accepted])),
    }


def bootstrap_contrasts(
    metrics: dict[str, dict[str, np.ndarray]],
    primary_indices: np.ndarray,
    bootstrap_indices: np.ndarray,
) -> dict[str, Any]:
    pairs = {
        "marginal_minus_hard": ("marginal_expected_regret", "hard_set_policy"),
        "marginal_minus_raw": ("marginal_expected_regret", "raw_expected_regret"),
        "marginal_minus_shuffled": ("marginal_expected_regret", "probability_shuffled"),
        "marginal_minus_masked": ("marginal_expected_regret", "utility_masked"),
        "marginal_minus_score": ("marginal_expected_regret", "score_composition"),
    }
    result = {}
    for label, (left, right) in pairs.items():
        result[label] = {
            metric: paired_delta_ci(
                metrics[left][metric][primary_indices],
                metrics[right][metric][primary_indices],
                bootstrap_indices,
            )
            for metric in (
                "accuracy",
                "compatible",
                "regret",
                "token_worst_regret",
            )
        }
    return result


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def write_json(path: Path, data: Any) -> None:
    """Write standards-compliant JSON and fail closed on non-finite values."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def artifact_manifest(
    output_dir: Path, execution: list[dict[str, str]]
) -> dict[str, Any]:
    excluded = {"artifact_manifest.json", "package_manifest.json"}
    files = [
        {"path": str(path.relative_to(output_dir)), "sha256": sha256_file(path)}
        for path in sorted(output_dir.rglob("*"))
        if path.is_file() and str(path.relative_to(output_dir)) not in excluded
    ]
    return {"execution_sources": execution, "files": files}


def risk_coverage_rows(
    selector: str,
    scores: np.ndarray,
    tokens: np.ndarray,
    metrics: dict[str, dict[str, np.ndarray]],
    primary_indices: np.ndarray,
) -> list[dict[str, Any]]:
    """Materialize the complete common-mask risk-coverage curve."""
    primary_tokens = tokens[primary_indices]
    order = np.lexsort((primary_tokens, scores[primary_indices]))
    rows = []
    for retained in range(1, len(order) + 1):
        selected = primary_indices[order[:retained]]
        row: dict[str, Any] = {
            "selector": selector,
            "retained_tokens": retained,
            "eligible_tokens": len(order),
            "coverage": float(retained / len(order)),
            "boundary_score": float(scores[selected[-1]]),
            "boundary_token": str(tokens[selected[-1]]),
        }
        for arm in ARMS:
            row[f"{arm}__accuracy"] = float(np.mean(metrics[arm]["accuracy"][selected]))
            row[f"{arm}__compatible"] = float(
                np.mean(metrics[arm]["compatible"][selected])
            )
            row[f"{arm}__regret"] = float(np.mean(metrics[arm]["regret"][selected]))
        rows.append(row)
    return rows


def compare_external_replay(
    output: Path, reference: Path, analysis_hash: str
) -> dict[str, Any]:
    """Compare a fresh process output with a prior independent run."""
    reference = reference.resolve(strict=True)
    if output == reference:
        raise ValueError("reference replay directory must differ from output")
    reference_summary = json.loads((reference / "summary.json").read_text())
    semantic_exact = reference_summary["analysis_hash"] == analysis_hash
    npz_files = (
        "probabilities.npz",
        "bootstrap_indices.npz",
        "per_seed_actions.npz",
        "selection_state.npz",
        "uncertainty_state.npz",
    )
    npz_checks = {}
    for name in npz_files:
        with np.load(output / name) as actual, np.load(reference / name) as expected:
            keys_exact = actual.files == expected.files
            arrays_exact = keys_exact
            if keys_exact:
                for key in actual.files:
                    left, right = actual[key], expected[key]
                    if left.dtype.kind in "fc":
                        equal = np.array_equal(left, right, equal_nan=True)
                    else:
                        equal = np.array_equal(left, right)
                    arrays_exact = arrays_exact and equal
            npz_checks[name] = {
                "keys_exact": keys_exact,
                "arrays_exact": bool(arrays_exact),
                "array_count": len(actual.files),
            }
    byte_files = (
        "analysis_freeze.json",
        "frozen_config.json",
        "split_manifest.json",
        "platt_calibrator.json",
        "lambda_selection.json",
        "selection_boundaries.json",
        "calibration_diagnostics.json",
        "seed_sensitivity.json",
        "metrics_by_token.jsonl",
        "metrics_by_token_policy.jsonl",
        "metrics_by_token_policy_seed.jsonl",
        "risk_coverage_curve.jsonl",
        "control_manifest.json",
    )
    byte_checks = {
        name: sha256_file(output / name) == sha256_file(reference / name)
        for name in byte_files
    }
    exact = (
        semantic_exact
        and all(row["arrays_exact"] for row in npz_checks.values())
        and all(byte_checks.values())
    )
    return {
        "reference_dir": str(reference),
        "semantic_analysis_hash_exact": semantic_exact,
        "npz": npz_checks,
        "byte_exact": byte_checks,
        "external_replay_exact": bool(exact),
    }


def main() -> None:
    started = time.time()
    args = parse_args()
    wave50 = args.wave50_dir.resolve(strict=True)
    wave52 = args.wave52_dir.resolve(strict=True)
    reference = args.reference_dir.resolve(strict=True) if args.reference_dir else None
    output = args.output_dir.resolve()
    config_path = args.config.resolve(strict=True)
    validate_output_path(
        output, [wave50, wave52] + ([reference] if reference is not None else [])
    )
    if output.exists():
        if not args.force:
            raise FileExistsError(f"output exists: {output}; use --force")
        shutil.rmtree(output)
    output.mkdir(parents=True)
    config = json.loads(config_path.read_text())
    os.environ["OMP_NUM_THREADS"] = str(4)
    os.environ["OPENBLAS_NUM_THREADS"] = str(4)
    os.environ["MKL_NUM_THREADS"] = str(4)

    execution = execution_sources(config_path)
    prefreeze_checked, file_hashes = validate_sources(wave52, config)
    expected_split = json.loads((wave52 / "split_manifest.json").read_text())
    expected_eval_hashes = {}
    for split in ("val_threshold", "val_monitor"):
        expected_eval_hashes[split] = {
            str(seed): file_hashes[f"raw_eval/frozen_set/seed{seed}__{split}.npz"]
            for seed in config["seeds"]
        }
    shutil.copy2(config_path, output / "frozen_config.json")
    write_json(
        output / "analysis_freeze.json",
        {
            "chronology": (
                "written-before-authorized-label-or-evaluation-prediction-access"
            ),
            "execution_sources": execution,
            "config_sha256": sha256_file(config_path),
            "expected_inputs": {
                "wave50_val_labels_sha256": config["source_binding"][
                    "wave50_val_labels_sha256"
                ],
                "wave52_eval_sha256": expected_eval_hashes,
                "wave52_val_threshold_tokens": expected_split["val_threshold"],
                "wave52_val_monitor_tokens": expected_split["val_monitor"],
            },
            "split_recipe": {
                "unit": "pair_token",
                "stratification": [
                    "design_stratum",
                    "compatible_cardinality",
                ],
                "calibration_fraction": config["calibration_fraction"],
                "split_seed": config["split_seed"],
            },
            "selection_recipes": {
                "platt": config["platt"],
                "utility_weight_grid": config["utility_weight_grid"],
                "selection_coverages": config["selection_coverages"],
                "diagnostic_criteria": config["diagnostic_criteria"],
            },
            "authorized_labels_accessed": False,
            "threshold_predictions_accessed": False,
            "monitor_predictions_accessed": False,
        },
    )

    label_checked = require_hash(
        wave50 / "authorized_labels/val.jsonl",
        config["source_binding"]["wave50_val_labels_sha256"],
    )
    threshold_checked = validate_split_sources(
        wave52, "val_threshold", config["seeds"], file_hashes
    )

    threshold = load_split(wave52, "val_threshold", config["seeds"])
    threshold_tokens = {str(token) for token in threshold["pair_token"]}
    if threshold_tokens != set(expected_split["val_threshold"]):
        raise RuntimeError("val_threshold tokens differ from the bound Wave 52 split")
    threshold_metadata = load_metadata(
        wave50 / "authorized_labels/val.jsonl", threshold_tokens
    )
    attach_metadata(threshold, threshold_metadata)

    strata = list(
        zip(threshold["design_stratum"], threshold["cardinality"], strict=True)
    )
    calibration_idx, decision_idx = stratified_token_split(
        threshold["pair_token"],
        strata,
        float(config["calibration_fraction"]),
        int(config["split_seed"]),
    )
    split_manifest = {
        "unit": "pair_token",
        "stratification": ["design_stratum", "compatible_cardinality"],
        "calibration_fit": threshold["pair_token"][calibration_idx].tolist(),
        "decision_select": threshold["pair_token"][decision_idx].tolist(),
        "val_monitor": expected_split["val_monitor"],
        "disjoint": True,
    }
    named_sets = {
        key: set(split_manifest[key])
        for key in ("calibration_fit", "decision_select", "val_monitor")
    }
    if any(
        named_sets[left] & named_sets[right]
        for left, right in (
            ("calibration_fit", "decision_select"),
            ("calibration_fit", "val_monitor"),
            ("decision_select", "val_monitor"),
        )
    ):
        raise RuntimeError("Wave 53 calibration, selection and monitor splits overlap")
    write_json(output / "split_manifest.json", split_manifest)
    monitor_checked = validate_split_sources(
        wave52, "val_monitor", config["seeds"], file_hashes
    )

    monitor = load_split(wave52, "val_monitor", config["seeds"])
    monitor_tokens = {str(token) for token in monitor["pair_token"]}
    if monitor_tokens != named_sets["val_monitor"]:
        raise RuntimeError("val_monitor tokens differ from the frozen split")
    monitor_metadata = load_metadata(
        wave50 / "authorized_labels/val.jsonl", monitor_tokens
    )
    attach_metadata(monitor, monitor_metadata)
    access_receipt = build_access_receipt(
        prefreeze_checked,
        label_checked,
        threshold_checked,
        monitor_checked,
        sha256_file(output / "analysis_freeze.json"),
        sha256_file(output / "split_manifest.json"),
    )
    write_json(output / "access_receipt.json", access_receipt)

    platt, platt_info = fit_platt(
        threshold["ensemble_logits"][calibration_idx],
        threshold["target"][calibration_idx],
        config["platt"],
    )
    raw_probability = {
        "threshold": expit(threshold["ensemble_logits"]),
        "monitor": expit(monitor["ensemble_logits"]),
    }
    calibrated_probability = {
        "threshold": platt_probability(platt, threshold["ensemble_logits"]),
        "monitor": platt_probability(platt, monitor["ensemble_logits"]),
    }
    platt_info["fit_tokens"] = threshold["pair_token"][calibration_idx].tolist()
    write_json(output / "platt_calibrator.json", platt_info)
    np.savez_compressed(
        output / "probabilities.npz",
        threshold_pair_token=threshold["pair_token"],
        monitor_pair_token=monitor["pair_token"],
        threshold_target=threshold["target"],
        monitor_target=monitor["target"],
        threshold_per_seed_logits=threshold["per_seed_logits"],
        monitor_per_seed_logits=monitor["per_seed_logits"],
        threshold_per_seed_raw_probability=expit(threshold["per_seed_logits"]),
        monitor_per_seed_raw_probability=expit(monitor["per_seed_logits"]),
        threshold_per_seed_platt_probability=np.stack(
            [
                platt_probability(platt, logits)
                for logits in threshold["per_seed_logits"]
            ]
        ),
        monitor_per_seed_platt_probability=np.stack(
            [platt_probability(platt, logits) for logits in monitor["per_seed_logits"]]
        ),
        threshold_ensemble_logits=threshold["ensemble_logits"],
        monitor_ensemble_logits=monitor["ensemble_logits"],
        threshold_raw_probability=raw_probability["threshold"],
        monitor_raw_probability=raw_probability["monitor"],
        threshold_platt_probability=calibrated_probability["threshold"],
        monitor_platt_probability=calibrated_probability["monitor"],
    )

    policy_manifest = json.loads((wave52 / "policy_manifest.json").read_text())
    levels = np.asarray(policy_manifest["levels"], dtype=np.float64)
    ranks = np.asarray(policy_manifest["rank_permutations"], dtype=np.int64)
    utilities = levels[ranks]
    penalty = float(config["incompatible_regret_penalty"])
    decision_primary = (threshold["design_stratum"][decision_idx] == "NEAR_RIVAL") & (
        threshold["cardinality"][decision_idx] >= 2
    )
    lambda_selection = select_lambda(
        threshold["ensemble_logits"][decision_idx],
        threshold["target"][decision_idx],
        utilities,
        decision_primary,
        config["utility_weight_grid"],
        penalty,
    )
    write_json(output / "lambda_selection.json", lambda_selection)

    raw_decision = expected_regret_actions(
        raw_probability["threshold"][decision_idx], utilities, penalty
    )
    calibrated_decision = expected_regret_actions(
        calibrated_probability["threshold"][decision_idx], utilities, penalty
    )
    raw_monitor = expected_regret_actions(
        raw_probability["monitor"], utilities, penalty
    )
    calibrated_monitor = expected_regret_actions(
        calibrated_probability["monitor"], utilities, penalty
    )
    tau = float(json.loads((wave52 / "summary.json").read_text())["tau"])
    hard_actions, hard_fallback = explicit_set_actions(
        raw_probability["monitor"], utilities, tau
    )
    score_actions = score_composition_actions(
        monitor["ensemble_logits"], utilities, lambda_selection["selected"]["weight"]
    )
    shuffle_map = deranged_within_strata(
        monitor["pair_token"],
        list(zip(monitor["design_stratum"], monitor["cardinality"], strict=True)),
        int(config["shuffle_seed"]),
    )
    shuffled = expected_regret_actions(
        calibrated_probability["monitor"][shuffle_map], utilities, penalty
    )
    masked_utilities = np.repeat(levels[None, :], len(utilities), axis=0)
    masked = expected_regret_actions(
        calibrated_probability["monitor"], masked_utilities, penalty
    )
    write_json(
        output / "control_manifest.json",
        {
            "probability_shuffled": {
                "unit": "pair_token probability vector",
                "mapping": "stable cyclic derangement within design_stratum x true_cardinality",
                "target_aware_matching": True,
                "purpose": "conservative negative control only; never deployable or used for selection",
                "fixed_points": int(np.sum(shuffle_map == np.arange(len(shuffle_map)))),
                "source_index_by_target_index": shuffle_map.tolist(),
            },
            "utility_masked": {
                "fixed_utility": levels.tolist(),
                "repeated_policy_count": len(utilities),
                "evaluation_uses_actual_policy": True,
            },
        },
    )
    actions = {
        "hard_set_policy": hard_actions,
        "score_composition": score_actions,
        "marginal_expected_regret": calibrated_monitor["actions"],
        "raw_expected_regret": raw_monitor["actions"],
        "probability_shuffled": shuffled["actions"],
        "utility_masked": masked["actions"],
        "oracle_set_then_utility": authorized_actions(monitor["target"], utilities),
    }
    metrics = {
        arm: token_metric_arrays(action, monitor["target"], utilities, penalty)
        for arm, action in actions.items()
    }
    primary = (monitor["design_stratum"] == "NEAR_RIVAL") & (
        monitor["cardinality"] >= 2
    )
    all_catalog = monitor["target"].any(axis=1)
    summaries = {
        "primary_near_rival_cardinality_ge2": summarize_metrics(metrics, primary),
        "all_in_catalog": summarize_metrics(metrics, all_catalog),
    }

    primary_indices = np.flatnonzero(primary)
    bootstrap_indices = paired_bootstrap_indices(
        len(primary_indices), int(config["n_boot"]), int(config["bootstrap_seed"])
    )
    np.savez_compressed(
        output / "bootstrap_indices.npz",
        pair_token=monitor["pair_token"][primary_indices],
        indices=bootstrap_indices,
        seed=np.asarray(int(config["bootstrap_seed"])),
    )
    contrasts = bootstrap_contrasts(metrics, primary_indices, bootstrap_indices)

    seed_sensitivity = {}
    per_seed_actions = []
    for seed_index, seed in enumerate(config["seeds"]):
        seed_probability = platt_probability(
            platt, monitor["per_seed_logits"][seed_index]
        )
        seed_decision = expected_regret_actions(seed_probability, utilities, penalty)
        seed_metric = token_metric_arrays(
            seed_decision["actions"], monitor["target"], utilities, penalty
        )
        seed_sensitivity[str(seed)] = {
            (
                "mean_token_worst_regret" if name == "token_worst_regret" else name
            ): float(np.mean(values[primary]))
            for name, values in seed_metric.items()
        }
        per_seed_actions.append(seed_decision["actions"])
    write_json(output / "seed_sensitivity.json", seed_sensitivity)
    np.savez_compressed(
        output / "per_seed_actions.npz",
        seeds=np.asarray(config["seeds"], dtype=np.int64),
        pair_token=monitor["pair_token"],
        marginal_expected_regret=np.stack(per_seed_actions),
    )

    decision_scores = {
        "raw_expected_regret": np.max(raw_decision["minimum_risk"], axis=1),
        "marginal_expected_regret": np.max(calibrated_decision["minimum_risk"], axis=1),
    }
    monitor_scores = {
        "raw_expected_regret": np.max(raw_monitor["minimum_risk"], axis=1),
        "marginal_expected_regret": np.max(calibrated_monitor["minimum_risk"], axis=1),
    }
    selection = {}
    selective = {}
    risk_curve = []
    for arm in ("raw_expected_regret", "marginal_expected_regret"):
        selection[arm] = {}
        selective[arm] = {
            "full_coverage": {
                "arms": {
                    compared_arm: selective_metrics(
                        metrics[compared_arm], primary, primary
                    )
                    for compared_arm in ARMS
                }
            },
            "aurc": discrete_aurc(
                monitor_scores[arm][primary],
                metrics[arm]["regret"][primary],
                monitor["pair_token"][primary],
            ),
            "secondary_min_margin_mean": float(
                np.mean(
                    np.min(
                        (
                            raw_monitor
                            if arm == "raw_expected_regret"
                            else calibrated_monitor
                        )["margin"],
                        axis=1,
                    )[primary]
                )
            ),
        }
        for coverage in config["selection_coverages"]:
            fit_tokens = threshold["pair_token"][decision_idx][decision_primary]
            boundary = coverage_boundary(
                decision_scores[arm][decision_primary],
                fit_tokens,
                float(coverage),
                int(config["selection_tie_seed"]),
            )
            accepted_primary = apply_coverage_boundary(
                monitor_scores[arm][primary],
                monitor["pair_token"][primary],
                boundary,
                int(config["selection_tie_seed"]),
            )
            full_mask = np.zeros(len(monitor["pair_token"]), dtype=bool)
            full_mask[primary_indices] = accepted_primary
            key = f"coverage_{coverage:.2f}"
            selection[arm][key] = boundary
            selective[arm][key] = {
                "arms": {
                    compared_arm: selective_metrics(
                        metrics[compared_arm], full_mask, primary
                    )
                    for compared_arm in ARMS
                }
            }
        risk_curve.extend(
            risk_coverage_rows(
                arm,
                monitor_scores[arm],
                monitor["pair_token"],
                metrics,
                primary_indices,
            )
        )
    write_json(output / "selection_boundaries.json", selection)
    write_jsonl(output / "risk_coverage_curve.jsonl", risk_curve)
    np.savez_compressed(
        output / "selection_state.npz",
        primary_pair_token=monitor["pair_token"][primary],
        raw_selection_score=monitor_scores["raw_expected_regret"][primary],
        marginal_selection_score=monitor_scores["marginal_expected_regret"][primary],
        raw_min_margin=np.min(raw_monitor["margin"], axis=1)[primary],
        marginal_min_margin=np.min(calibrated_monitor["margin"], axis=1)[primary],
        **{
            f"accepted__{arm}__{coverage:.2f}": apply_coverage_boundary(
                monitor_scores[arm][primary],
                monitor["pair_token"][primary],
                selection[arm][f"coverage_{coverage:.2f}"],
                int(config["selection_tie_seed"]),
            )
            for arm in ("raw_expected_regret", "marginal_expected_regret")
            for coverage in config["selection_coverages"]
        },
    )

    calibration = {}
    for split_name, source, data, indices in (
        ("calibration_fit", "threshold", threshold, calibration_idx),
        ("decision_select", "threshold", threshold, decision_idx),
        ("val_monitor", "monitor", monitor, np.arange(len(monitor["pair_token"]))),
    ):
        split_primary = (data["design_stratum"][indices] == "NEAR_RIVAL") & (
            data["cardinality"][indices] >= 2
        )
        calibration[split_name] = {}
        for population, population_mask in (
            ("all_in_catalog", np.ones(len(indices), dtype=bool)),
            ("primary_near_rival_cardinality_ge2", split_primary),
        ):
            calibration[split_name][population] = {
                "n_tokens": int(np.sum(population_mask)),
                "raw": calibration_metrics(
                    raw_probability[source][indices][population_mask],
                    data["target"][indices][population_mask],
                ),
                "platt": calibration_metrics(
                    calibrated_probability[source][indices][population_mask],
                    data["target"][indices][population_mask],
                ),
            }
    write_json(output / "calibration_diagnostics.json", calibration)

    metric_rows = []
    for index, token in enumerate(monitor["pair_token"]):
        for arm in ARMS:
            metric_rows.append(
                {
                    "pair_token": str(token),
                    "cluster_id": str(monitor["cluster_id"][index]),
                    "design_stratum": str(monitor["design_stratum"][index]),
                    "cardinality": int(monitor["cardinality"][index]),
                    "arm": arm,
                    **{
                        name: float(values[index])
                        for name, values in metrics[arm].items()
                    },
                }
            )
    write_jsonl(output / "metrics_by_token.jsonl", metric_rows)
    metric_policy_rows = []
    truth = authorized_actions(monitor["target"], utilities)
    for arm in ARMS:
        regret = constrained_regret(actions[arm], monitor["target"], utilities, penalty)
        compatible = monitor["target"][
            np.arange(len(monitor["target"]))[:, None], actions[arm]
        ]
        for token_index, token in enumerate(monitor["pair_token"]):
            for policy_index in range(len(utilities)):
                metric_policy_rows.append(
                    {
                        "pair_token": str(token),
                        "cluster_id": str(monitor["cluster_id"][token_index]),
                        "design_stratum": str(monitor["design_stratum"][token_index]),
                        "cardinality": int(monitor["cardinality"][token_index]),
                        "arm": arm,
                        "policy_index": policy_index,
                        "action": int(actions[arm][token_index, policy_index]),
                        "authorized_action": int(truth[token_index, policy_index]),
                        "action_accuracy": float(
                            actions[arm][token_index, policy_index]
                            == truth[token_index, policy_index]
                        ),
                        "compatible_action": float(
                            compatible[token_index, policy_index]
                        ),
                        "restricted_regret": float(regret[token_index, policy_index]),
                    }
                )
    write_jsonl(output / "metrics_by_token_policy.jsonl", metric_policy_rows)
    metric_seed_rows = []
    for seed_index, seed in enumerate(config["seeds"]):
        seed_action = per_seed_actions[seed_index]
        seed_regret = constrained_regret(
            seed_action, monitor["target"], utilities, penalty
        )
        seed_compatible = monitor["target"][
            np.arange(len(monitor["target"]))[:, None], seed_action
        ]
        for token_index, token in enumerate(monitor["pair_token"]):
            for policy_index in range(len(utilities)):
                metric_seed_rows.append(
                    {
                        "seed": int(seed),
                        "pair_token": str(token),
                        "cluster_id": str(monitor["cluster_id"][token_index]),
                        "design_stratum": str(monitor["design_stratum"][token_index]),
                        "cardinality": int(monitor["cardinality"][token_index]),
                        "arm": "marginal_expected_regret",
                        "policy_index": policy_index,
                        "action": int(seed_action[token_index, policy_index]),
                        "authorized_action": int(truth[token_index, policy_index]),
                        "action_accuracy": float(
                            seed_action[token_index, policy_index]
                            == truth[token_index, policy_index]
                        ),
                        "compatible_action": float(
                            seed_compatible[token_index, policy_index]
                        ),
                        "restricted_regret": float(
                            seed_regret[token_index, policy_index]
                        ),
                    }
                )
    write_jsonl(output / "metrics_by_token_policy_seed.jsonl", metric_seed_rows)
    np.savez_compressed(
        output / "uncertainty_state.npz",
        pair_token=monitor["pair_token"],
        target=monitor["target"],
        utilities=utilities,
        sets=calibrated_monitor["sets"],
        set_mass=calibrated_monitor["set_mass"],
        action_risk=calibrated_monitor["action_risk"],
        actions=calibrated_monitor["actions"],
        minimum_risk=calibrated_monitor["minimum_risk"],
        margin=calibrated_monitor["margin"],
        raw_set_mass=raw_monitor["set_mass"],
        raw_action_risk=raw_monitor["action_risk"],
        raw_actions=raw_monitor["actions"],
        shuffled_source_index=shuffle_map,
        hard_fallback=hard_fallback,
    )

    criteria = config["diagnostic_criteria"]
    marginal_hard = contrasts["marginal_minus_hard"]
    marginal_raw = contrasts["marginal_minus_raw"]
    marginal_shuffled = contrasts["marginal_minus_shuffled"]
    marginal_masked = contrasts["marginal_minus_masked"]
    selective_75 = selective["marginal_expected_regret"]["coverage_0.75"]["arms"][
        "marginal_expected_regret"
    ]
    marginal_full = selective["marginal_expected_regret"]["full_coverage"]["arms"][
        "marginal_expected_regret"
    ]
    primary_calibration = calibration["val_monitor"][
        "primary_near_rival_cardinality_ge2"
    ]
    platt_monitor = primary_calibration["platt"]
    raw_calibration_monitor = primary_calibration["raw"]
    conditions = {
        "regret_advantage_over_hard": marginal_hard["regret"]["mean_diff"]
        <= -criteria["regret_reduction_min"]
        and marginal_hard["regret"]["ci95_high"] < 0.0,
        "accuracy_noninferiority_to_hard": marginal_hard["accuracy"]["mean_diff"]
        >= criteria["accuracy_noninferiority"],
        "compatible_noninferiority_to_hard": marginal_hard["compatible"]["mean_diff"]
        >= criteria["compatible_rate_noninferiority"],
        "platt_improves_brier_or_nll": platt_monitor["brier"]
        < raw_calibration_monitor["brier"]
        or platt_monitor["nll"] < raw_calibration_monitor["nll"],
        "platt_regret_tolerated": marginal_raw["regret"]["mean_diff"]
        <= criteria["platt_regret_tolerance"],
        "selective_75_reduces_regret": selective_75["regret"] < marginal_full["regret"],
        "selective_75_coverage_in_range": criteria["selective_coverage_low"]
        <= selective_75["coverage"]
        <= criteria["selective_coverage_high"],
        "probability_control": marginal_shuffled["accuracy"]["mean_diff"]
        >= criteria["control_accuracy_improvement_min"],
        "utility_control": marginal_masked["accuracy"]["mean_diff"]
        >= criteria["control_accuracy_improvement_min"],
    }
    core_signature = stable_hash(
        {
            "summaries": summaries,
            "contrasts": contrasts,
            "calibration": calibration,
            "selection": selection,
            "selective": selective,
            "conditions_without_external_replay": conditions,
            "action_hash": stable_hash(
                {arm: value.tolist() for arm, value in actions.items()}
            ),
        }
    )
    if reference is None:
        replay = {
            "status": "PENDING_EXTERNAL_REPLAY",
            "external_replay_exact": None,
            "core_signature": core_signature,
        }
    else:
        replay = compare_external_replay(output, reference, core_signature)
        replay["status"] = "PASS" if replay["external_replay_exact"] else "FAIL"
        replay["core_signature"] = core_signature
    write_json(output / "replay_receipt.json", replay)
    diagnostic = {
        **conditions,
        "external_replay_exact": replay["external_replay_exact"],
        "uncertainty_policy_promising": (
            all(conditions.values()) and bool(replay["external_replay_exact"])
            if replay["external_replay_exact"] is not None
            else None
        ),
    }

    summary = {
        "status": config["status"],
        "automatic_go": False,
        "scientific_claim_allowed": False,
        "git_commit": git_commit(),
        "execution_sources": execution,
        "source_binding": access_receipt["files_read"],
        "data": {
            "calibration_fit_tokens": len(calibration_idx),
            "decision_select_tokens": len(decision_idx),
            "monitor_tokens": len(monitor["pair_token"]),
            "primary_monitor_tokens": int(primary.sum()),
            "policy_count": len(utilities),
        },
        "tau_frozen_from_wave52": tau,
        "lambda_selection": lambda_selection,
        "platt": platt_info,
        "seed_sensitivity": seed_sensitivity,
        "metrics": summaries,
        "calibration": calibration,
        "selective": selective,
        "contrasts": contrasts,
        "diagnostic_pattern": diagnostic,
        "analysis_hash": core_signature,
        "replay": replay,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "runtime": {
            "wall_seconds": float(time.time() - started),
            "max_rss_mib": float(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
            ),
            "device": "cpu",
        },
        "artifact_contract": {
            "monitor_token_rows": len(metric_rows),
            "monitor_token_policy_rows": len(metric_policy_rows),
            "monitor_token_policy_seed_rows": len(metric_seed_rows),
            "per_seed_action_arrays": len(per_seed_actions),
            "bootstrap_draws": int(config["n_boot"]),
            "risk_coverage_rows": len(risk_curve),
            "raw_eval_reused_without_reforward": True,
        },
    }
    write_json(output / "summary.json", summary)

    primary_summary = summaries["primary_near_rival_cardinality_ge2"]
    report = [
        "# Ola 53 — política ordinal sensible a incertidumbre",
        "",
        f"> Estado: `{config['status']} / CPU-ONLY / NO-GO-NOGO`",
        "",
        "## Resultado diagnóstico",
        "",
        "| Brazo | accuracy | compatible | regret | promedio del peor regret por token |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        row = primary_summary[arm]
        report.append(
            f"| `{arm}` | {row['accuracy']:.4f} | {row['compatible']:.4f} | {row['regret']:.4f} | {row['mean_token_worst_regret']:.4f} |"
        )
    report.extend(
        [
            "",
            "## Contrastes principales",
            "",
            f"- regret marginal − hard: {marginal_hard['regret']['mean_diff']:+.4f} "
            f"IC95 [{marginal_hard['regret']['ci95_low']:+.4f}, {marginal_hard['regret']['ci95_high']:+.4f}]",
            f"- accuracy marginal − hard: {marginal_hard['accuracy']['mean_diff']:+.4f} "
            f"IC95 [{marginal_hard['accuracy']['ci95_low']:+.4f}, {marginal_hard['accuracy']['ci95_high']:+.4f}]",
            f"- accuracy marginal − shuffled: {marginal_shuffled['accuracy']['mean_diff']:+.4f}",
            f"- accuracy marginal − utility_masked: {marginal_masked['accuracy']['mean_diff']:+.4f}",
            "",
            "## Abstención empírica",
            "",
            f"- full coverage regret: {marginal_full['regret']:.4f}",
            f"- nominal 75%: coverage efectiva {selective_75['coverage']:.4f}, regret {selective_75['regret']:.4f}",
            f"- AURC discreta: {selective['marginal_expected_regret']['aurc']:.4f}",
            "",
            "## Adjudicación",
            "",
            f"- patrón diagnóstico conjunto: `{diagnostic['uncertainty_policy_promising']}`",
            "- Este patrón organiza un smoke de desarrollo históricamente abierto. No promueve arquitectura, no declara GO/NO-GO y no ofrece garantía conformal.",
        ]
    )
    (output / "REPORT_WAVE53_UNCERTAINTY_POLICY.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )

    manifest = artifact_manifest(output, execution)
    write_json(output / "artifact_manifest.json", manifest)
    write_json(
        output / "package_manifest.json",
        {
            "phase": "WAVE53_SMOKE_FINAL_PACKAGE_ROOT",
            "files": [
                {"path": name, "sha256": sha256_file(output / name)}
                for name in (
                    "summary.json",
                    "REPORT_WAVE53_UNCERTAINTY_POLICY.md",
                    "artifact_manifest.json",
                )
            ],
        },
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "analysis_hash": core_signature,
                "diagnostic": diagnostic,
                "runtime_seconds": summary["runtime"]["wall_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
