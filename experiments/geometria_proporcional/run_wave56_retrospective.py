#!/usr/bin/env python3
"""Run the opened-data Wave 56 contextual residual-gate diagnostic on CPU."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import importlib.metadata
import itertools
import json
import platform
from pathlib import Path
import subprocess
import sys
import warnings
from typing import Any

import numpy as np
from scipy.special import expit
from scipy.stats import pearsonr, spearmanr
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import HuberRegressor, LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from geometria_proporcional.wave49_schema import sha256_file, write_json  # noqa: E402
from geometria_proporcional.wave52_policy import authorized_actions, constrained_regret  # noqa: E402
from geometria_proporcional.wave54_joint_set import (  # noqa: E402
    expected_regret_from_mass,
    posterior_mass,
)
from geometria_proporcional.wave55_policy_bridge import action_metric_arrays  # noqa: E402
from geometria_proporcional.wave56_contextual_gate import (  # noqa: E402
    FEATURE_NAMES,
    apply_gate,
    contextual_design,
    disagreement_weights,
    fit_weighted_scaler,
    realized_gain,
    stratified_gain_shuffle,
    weighted_error,
)

PLAN_PATH = REPO_ROOT / "Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_56_CONTEXTUAL_RESIDUAL_GATE_PLAN.md"
NOTE_PATH = REPO_ROOT / "Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_56_CONTEXTUAL_RESIDUAL_GATE_RESEARCH_NOTE.md"
CONFIG_DEFAULT = REPO_ROOT / "experiments/geometria_proporcional/configs/wave56_contextual_gate.json"
PRIMITIVES = REPO_ROOT / "src/geometria_proporcional/wave56_contextual_gate.py"
HARD_ONLY = "hard_only"
CONTEXTUAL_FAMILIES = ("ridge_contextual", "huber_contextual", "logistic_contextual")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wave55-dir", type=Path, required=True)
    parser.add_argument("--wave54-dir", type=Path, required=True)
    parser.add_argument("--wave52-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=CONFIG_DEFAULT)
    parser.add_argument("--reference-dir", type=Path)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def digest(path: Path) -> str:
    return sha256_file(path.resolve(strict=True))


def require_hash(path: Path, expected: str) -> None:
    actual = digest(path)
    if actual != expected:
        raise RuntimeError(f"hash mismatch for {path}: {actual} != {expected}")


def write_json_strict(path: Path, payload: Any) -> None:
    """Reject non-finite JSON numbers instead of emitting non-standard NaN tokens."""
    def validate(value: Any, location: str) -> None:
        if isinstance(value, (float, np.floating)) and not np.isfinite(value):
            raise ValueError(f"non-finite JSON value at {location}")
        if isinstance(value, dict):
            for key, item in value.items():
                validate(item, f"{location}.{key}")
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                validate(item, f"{location}[{index}]")

    validate(payload, "$")
    write_json(path, payload)


def require_sources_at_head(paths: list[Path]) -> str:
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO_ROOT, text=True
    ).strip()
    if dirty:
        raise RuntimeError("tracked worktree must be clean before Wave 56 Stage 0")
    for path in paths:
        relative = path.resolve(strict=True).relative_to(REPO_ROOT)
        subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", str(relative)],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
        )
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()


def prepare_output(path: Path, force: bool) -> Path | None:
    archived = None
    if path.exists():
        if not force:
            raise FileExistsError(path)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        archived = path.with_name(f"{path.name}.superseded_{stamp}")
        if archived.exists():
            raise FileExistsError(archived)
        path.rename(archived)
    path.mkdir(parents=True)
    return archived


def load_bundle(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        out = {name: data[name] for name in data.files}
    required = {
        "pair_token", "target", "per_seed_logits", "ensemble_logits",
        "design_stratum", "cardinality",
    }
    if required - set(out):
        raise RuntimeError(f"bundle missing {sorted(required - set(out))}")
    return out


def load_utilities(path: Path) -> tuple[np.ndarray, list[list[int]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    levels = np.asarray(payload["levels"], dtype=np.float64)
    permutations = np.asarray(payload["rank_permutations"], dtype=np.int64)
    groups = payload["groups"]
    expected = set(itertools.permutations(range(4)))
    if permutations.shape != (24, 4) or set(map(tuple, permutations)) != expected:
        raise RuntimeError("policy manifest is not the frozen 24-policy catalogue")
    flat_groups = [int(index) for group in groups for index in group]
    if len(groups) != 3 or any(len(group) != 8 for group in groups) or sorted(flat_groups) != list(range(24)):
        raise RuntimeError("policy groups must be a disjoint 3x8 partition of the catalogue")
    return levels[permutations], groups


def make_dataset(
    bundle: dict[str, np.ndarray], theta: np.ndarray, utilities: np.ndarray, config: dict[str, Any]
) -> dict[str, np.ndarray]:
    logits = bundle["ensemble_logits"].astype(np.float64)
    mass = posterior_mass(logits, theta, "joint_full")
    decision = expected_regret_from_mass(
        mass, utilities, float(config["incompatible_regret_penalty"])
    )
    from geometria_proporcional.wave52_policy import explicit_set_actions

    hard, _ = explicit_set_actions(expit(logits), utilities, float(config["hard_set_tau"]))
    posterior = decision["actions"]
    design = contextual_design(
        ensemble_logits=logits,
        per_seed_logits=bundle["per_seed_logits"],
        set_mass=mass,
        action_risk=decision["action_risk"],
        hard_actions=hard,
        posterior_actions=posterior,
        utilities=utilities,
        hard_tau=float(config["hard_set_tau"]),
    )
    gain = realized_gain(
        hard,
        posterior,
        bundle["target"],
        utilities,
        float(config["incompatible_regret_penalty"]),
    )
    population = config["primary_population"]
    primary = (
        (bundle["design_stratum"].astype(str) == str(population["design_stratum"]))
        & (bundle["cardinality"] >= int(population["minimum_true_cardinality"]))
    )
    return {
        **bundle,
        **design,
        "posterior_actions": posterior,
        "hard_actions": hard,
        "gain": gain,
        "weights": disagreement_weights(design["disagreement"]),
        "primary": primary,
    }


def check_minimums(data: dict[str, np.ndarray], config: dict[str, Any], label: str) -> dict[str, int]:
    mask = data["primary"]
    counts = {
        "tokens": int(mask.sum()),
        "disagreement_rows": int((data["disagreement"] & mask[:, None]).sum()),
    }
    minimums = config["minimums"]
    if counts["tokens"] < int(minimums["tokens"]) or counts["disagreement_rows"] < int(minimums["disagreement_rows"]):
        raise RuntimeError(f"{label} is NOT_EVALUABLE: {counts}")
    return counts


def make_folds(data: dict[str, np.ndarray], config: dict[str, Any]) -> np.ndarray:
    primary_indices = np.flatnonzero(data["primary"])
    splitter = StratifiedKFold(
        n_splits=int(config["folds"]["n_splits"]),
        shuffle=True,
        random_state=int(config["folds"]["random_state"]),
    )
    assignment = np.full(len(data["pair_token"]), -1, dtype=np.int64)
    labels = data["cardinality"][primary_indices].astype(str)
    for fold, (_, holdout) in enumerate(splitter.split(primary_indices, labels)):
        token_indices = primary_indices[holdout]
        if len(token_indices) < int(config["minimums"]["tokens_per_fold"]):
            raise RuntimeError("Stage 0 fold is NOT_EVALUABLE")
        assignment[token_indices] = fold
    if np.any(assignment[primary_indices] < 0):
        raise AssertionError("incomplete token fold assignment")
    return assignment


def selected_columns(spec: dict[str, Any]) -> np.ndarray:
    if spec["columns"] == "all":
        return np.arange(len(FEATURE_NAMES), dtype=np.int64)
    names = list(FEATURE_NAMES)
    return np.asarray([names.index(name) for name in spec["columns"]], dtype=np.int64)


def parameter_grid(spec: dict[str, Any]) -> list[dict[str, float]]:
    if spec["kind"] == "ridge":
        return [{"alpha": float(alpha)} for alpha in spec["alpha"]]
    if spec["kind"] == "huber":
        return [
            {"alpha": float(alpha), "epsilon": float(epsilon)}
            for alpha in spec["alpha"]
            for epsilon in spec["epsilon"]
        ]
    if spec["kind"] == "logistic":
        return [{"C": float(value)} for value in spec["C"]]
    raise ValueError(spec["kind"])


def fit_model(
    spec: dict[str, Any], params: dict[str, float], x: np.ndarray, y: np.ndarray, weights: np.ndarray
) -> dict[str, Any]:
    scaler = fit_weighted_scaler(x, weights)
    xs = scaler.transform(x)
    kind = spec["kind"]
    if kind == "ridge":
        model = Ridge(alpha=params["alpha"], solver="svd").fit(xs, y, sample_weight=weights)
    elif kind == "huber":
        model = HuberRegressor(
            alpha=params["alpha"],
            epsilon=params["epsilon"],
            max_iter=int(spec["max_iter"]),
            tol=float(spec["tol"]),
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(xs, y, sample_weight=weights)
        if any(issubclass(item.category, ConvergenceWarning) for item in caught):
            raise RuntimeError("Huber did not converge")
    elif kind == "logistic":
        binary = (y > 1e-12).astype(np.int64)
        if np.unique(binary).size < 2:
            raise RuntimeError("logistic fit has one class")
        model = LogisticRegression(
            C=params["C"],
            class_weight=spec["class_weight"],
            solver=spec["solver"],
            max_iter=int(spec["max_iter"]),
            random_state=int(spec["random_state"]),
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(xs, binary, sample_weight=weights)
        if any(issubclass(item.category, ConvergenceWarning) for item in caught):
            raise RuntimeError("logistic did not converge")
    else:
        raise ValueError(kind)
    return {"model": model, "scaler": scaler}


def predict_model(state: dict[str, Any], spec: dict[str, Any], x: np.ndarray) -> np.ndarray:
    xs = state["scaler"].transform(x)
    if spec["kind"] == "logistic":
        return state["model"].predict_proba(xs)[:, 1]
    return state["model"].predict(xs)


def model_state_json(state: dict[str, Any]) -> dict[str, Any]:
    model = state["model"]
    return {
        "mean": state["scaler"].mean.tolist(),
        "scale": state["scaler"].scale.tolist(),
        "coef": np.asarray(model.coef_).tolist(),
        "intercept": np.asarray(model.intercept_).tolist(),
        "n_iter": np.asarray(getattr(model, "n_iter_", [])).tolist(),
    }


def fit_rows(data: dict[str, np.ndarray], token_mask: np.ndarray, policy_mask: np.ndarray | None = None) -> np.ndarray:
    rows = data["disagreement"] & token_mask[:, None]
    if policy_mask is not None:
        rows &= policy_mask[None, :]
    return rows


def crossfit_fixed(
    data: dict[str, np.ndarray],
    folds: np.ndarray,
    spec: dict[str, Any],
    params: dict[str, float],
    target: np.ndarray,
    policy_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    columns = selected_columns(spec)
    scores = np.full(data["disagreement"].shape, np.nan, dtype=np.float64)
    fold_models = []
    n_folds = int(folds.max()) + 1
    for fold in range(n_folds):
        train_tokens = data["primary"] & (folds != fold)
        holdout_tokens = data["primary"] & (folds == fold)
        train_rows = fit_rows(data, train_tokens, policy_mask)
        holdout_rows = fit_rows(data, holdout_tokens, policy_mask)
        state = fit_model(
            spec,
            params,
            data["design"][train_rows][:, columns],
            target[train_rows],
            data["weights"][train_rows],
        )
        scores[holdout_rows] = predict_model(
            state, spec, data["design"][holdout_rows][:, columns]
        )
        fold_models.append({"fold": fold, **model_state_json(state)})
    expected = fit_rows(data, data["primary"], policy_mask)
    if np.any(~np.isfinite(scores[expected])):
        raise RuntimeError("cross-fitting left non-finite OOF scores")
    return {"scores": scores, "fold_models": fold_models}


def choose_hyperparameters(
    data: dict[str, np.ndarray],
    folds: np.ndarray,
    spec: dict[str, Any],
    policy_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    rows = fit_rows(data, data["primary"], policy_mask)
    target = (data["gain"] > 1e-12).astype(np.float64) if spec["kind"] == "logistic" else data["gain"]
    candidates = []
    for params in parameter_grid(spec):
        try:
            crossfit = crossfit_fixed(data, folds, spec, params, data["gain"], policy_mask)
            scores = crossfit["scores"]
            value = weighted_error(
                scores[rows], target[rows], data["weights"][rows], spec["objective"]
            )
            candidates.append({
                "params": params,
                "objective": value,
                "status": "PASS",
                "scores": scores,
                "fold_models": crossfit["fold_models"],
            })
        except Exception as error:  # preserved as an auditable failed candidate
            candidates.append({"params": params, "objective": None, "status": "FIT_FAILED", "error": str(error)})
    valid = [row for row in candidates if row["status"] == "PASS"]
    if not valid:
        return {"status": "NOT_EVALUABLE", "candidates": candidates}
    best = min(float(row["objective"]) for row in valid)
    selected = next(row for row in valid if float(row["objective"]) <= best + 1e-12)
    return {"status": "PASS", "selected": selected, "candidates": candidates}


def metric_arrays_for_policies(
    actions: np.ndarray,
    data: dict[str, np.ndarray],
    utilities: np.ndarray,
    penalty: float,
    policy_indices: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    if policy_indices is None:
        return action_metric_arrays(actions, data["target"], utilities, penalty)
    idx = np.asarray(policy_indices, dtype=np.int64)
    selected_actions = actions[:, idx]
    selected_utilities = utilities[idx]
    oracle = authorized_actions(data["target"], selected_utilities)
    regret = constrained_regret(selected_actions, data["target"], selected_utilities, penalty)
    compatible = data["target"][np.arange(len(data["target"]))[:, None], selected_actions]
    return {
        "accuracy": np.mean(selected_actions == oracle, axis=1),
        "compatible": np.mean(compatible, axis=1),
        "regret": np.mean(regret, axis=1),
        "worst_regret": np.max(regret, axis=1),
        "regret_by_policy": regret,
        "compatible_by_policy": compatible,
    }


def summarize(metrics: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, float]:
    return {name: float(values[mask].mean()) for name, values in metrics.items() if values.ndim == 1}


def paired_bootstrap_indices(n_tokens: int, config: dict[str, Any]) -> np.ndarray:
    rng = np.random.Generator(np.random.PCG64(int(config["bootstrap"]["seed"])))
    return rng.integers(
        0,
        n_tokens,
        size=(int(config["bootstrap"]["replicates"]), n_tokens),
        endpoint=False,
        dtype=np.int64,
    )


def paired_delta_ci(left: np.ndarray, right: np.ndarray, indices: np.ndarray) -> dict[str, float]:
    delta = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    sampled = delta[indices].mean(axis=1)
    low, high = np.percentile(sampled, [2.5, 97.5])
    return {
        "mean_diff": float(delta.mean()),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "fraction_positive": float(np.mean(sampled > 0.0)),
    }


def override_summary(
    actions: np.ndarray,
    override: np.ndarray,
    data: dict[str, np.ndarray],
    utilities: np.ndarray,
    config: dict[str, Any],
) -> dict[str, float | int | None]:
    mask = data["primary"][:, None]
    active = np.asarray(override, dtype=bool) & mask
    gain = data["gain"]
    beneficial = active & (gain > 1e-12)
    harmful = active & (gain < -1e-12)
    neutral = active & ~(beneficial | harmful)
    available_beneficial = (data["disagreement"] & mask & (gain > 1e-12)).sum()
    n_non_neutral = int((beneficial | harmful).sum())
    n_override = int(active.sum())
    return {
        "n_overrides": n_override,
        "coverage": float(active.sum() / mask.repeat(active.shape[1], axis=1).sum()),
        "beneficial": int(beneficial.sum()),
        "harmful": int(harmful.sum()),
        "neutral": int(neutral.sum()),
        "precision_beneficial": float(beneficial.sum() / n_non_neutral) if n_non_neutral else None,
        "recall_beneficial": float(beneficial.sum() / available_beneficial) if available_beneficial else None,
        "mean_gain_on_override": float(gain[active].mean()) if n_override else None,
        "mean_beneficial_gain": float(gain[beneficial].mean()) if beneficial.any() else None,
        "mean_harmful_gain": float(gain[harmful].mean()) if harmful.any() else None,
    }


def feature_schema() -> list[dict[str, Any]]:
    domains = {
        "advantage": "nonnegative posterior-risk difference",
        "hard_risk": "finite expected regret",
        "minimum_risk": "finite expected regret",
        "action_risk_margin": "nonnegative expected-regret margin",
        "posterior_entropy_norm": "[0,1]",
        "posterior_top_mass": "[0,1]",
        "posterior_top_margin": "[0,1]",
        "hard_cardinality": "{1,2,3,4}",
        "posterior_expected_cardinality": "[1,4]",
        "posterior_cardinality_variance": "nonnegative",
        "posterior_mass_hard_set": "[0,1]",
        "seed_std_mean": "nonnegative logit dispersion",
        "seed_std_max": "nonnegative logit dispersion",
        "utility_f0": "ordinal utility level",
        "utility_f1": "ordinal utility level",
        "utility_f2": "ordinal utility level",
        "utility_f3": "ordinal utility level",
    }
    return [
        {
            "index": index,
            "name": name,
            "dtype": "float64",
            "normalization": "weighted z-score fit within training fold; constants scale=1",
            "domain": domains[name],
        }
        for index, name in enumerate(FEATURE_NAMES)
    ]


def select_operating_point(
    scores: np.ndarray,
    data: dict[str, np.ndarray],
    utilities: np.ndarray,
    config: dict[str, Any],
    *,
    token_mask: np.ndarray,
    policy_indices: np.ndarray | None = None,
) -> dict[str, Any]:
    penalty = float(config["incompatible_regret_penalty"])
    policy_mask = None
    if policy_indices is not None:
        policy_mask = np.zeros(scores.shape[1], dtype=bool)
        policy_mask[np.asarray(policy_indices, dtype=np.int64)] = True
    eligible_rows = fit_rows(data, token_mask, policy_mask)
    values = scores[eligible_rows]
    if len(values) == 0 or np.any(~np.isfinite(values)):
        raise RuntimeError("no finite scores for threshold selection")
    hard_metrics = metric_arrays_for_policies(
        data["hard_actions"], data, utilities, penalty, policy_indices
    )
    hard_summary = summarize(hard_metrics, token_mask)
    rows = []
    for quantile in config["threshold_quantiles"]:
        threshold = float(np.quantile(values, float(quantile), method="higher"))
        gated = apply_gate(scores, data["hard_actions"], data["posterior_actions"], threshold)
        metrics = metric_arrays_for_policies(gated["actions"], data, utilities, penalty, policy_indices)
        summary = summarize(metrics, token_mask)
        if policy_indices is None:
            coverage = float(gated["override"][token_mask].mean())
        else:
            coverage = float(gated["override"][token_mask][:, policy_indices].mean())
        feasible = (
            summary["accuracy"] >= hard_summary["accuracy"] - float(config["selection"]["accuracy_noninferiority_margin"])
            and summary["compatible"]
            >= hard_summary["compatible"]
            - float(config["selection"]["compatible_noninferiority_margin"])
        )
        rows.append({"q": float(quantile), "threshold": threshold, "coverage": coverage, "feasible": bool(feasible), **summary})
    rows.append({"q": HARD_ONLY, "threshold": HARD_ONLY, "coverage": 0.0, "feasible": True, **hard_summary})
    feasible = [row for row in rows if row["feasible"]]
    atol = float(config["selection"]["tie_atol"])
    best_regret = min(float(row["regret"]) for row in feasible)
    tied = [row for row in feasible if abs(float(row["regret"]) - best_regret) <= atol]
    selected = min(tied, key=lambda row: float(row["coverage"]))
    return {"selected": selected, "grid": rows}


def fit_full_and_evaluate(
    fit_data: dict[str, np.ndarray],
    eval_data: dict[str, np.ndarray],
    spec: dict[str, Any],
    params: dict[str, float],
    q: float | str,
    target: np.ndarray,
    utilities: np.ndarray,
    config: dict[str, Any],
    fit_policy_indices: np.ndarray | None = None,
) -> dict[str, Any]:
    columns = selected_columns(spec)
    policy_mask = None
    if fit_policy_indices is not None:
        policy_mask = np.zeros(fit_data["disagreement"].shape[1], dtype=bool)
        policy_mask[np.asarray(fit_policy_indices, dtype=np.int64)] = True
    rows = fit_rows(fit_data, fit_data["primary"], policy_mask)
    state = fit_model(
        spec,
        params,
        fit_data["design"][rows][:, columns],
        target[rows],
        fit_data["weights"][rows],
    )
    fit_scores = np.full(fit_data["disagreement"].shape, np.nan)
    eval_scores = np.full(eval_data["disagreement"].shape, np.nan)
    fit_disagreement = fit_data["disagreement"]
    eval_disagreement = eval_data["disagreement"]
    fit_scores[fit_disagreement] = predict_model(state, spec, fit_data["design"][fit_disagreement][:, columns])
    eval_scores[eval_disagreement] = predict_model(state, spec, eval_data["design"][eval_disagreement][:, columns])
    if q == HARD_ONLY:
        threshold: float | str = HARD_ONLY
    else:
        threshold = float(np.quantile(fit_scores[rows], float(q), method="higher"))
    gated = apply_gate(eval_scores, eval_data["hard_actions"], eval_data["posterior_actions"], threshold)
    metrics = action_metric_arrays(
        gated["actions"], eval_data["target"], utilities, float(config["incompatible_regret_penalty"])
    )
    return {
        "state": state,
        "fit_scores": fit_scores,
        "eval_scores": eval_scores,
        "threshold": threshold,
        "actions": gated["actions"],
        "override": gated["override"],
        "metrics": metrics,
        "summary": summarize(metrics, eval_data["primary"]),
        "coverage": float(gated["override"][eval_data["primary"]].mean()),
    }


def correlation_summary(scores: np.ndarray, data: dict[str, np.ndarray]) -> dict[str, float | None]:
    rows = fit_rows(data, data["primary"])
    x, y = scores[rows], data["gain"][rows]
    if len(x) < 3 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return {"pearson": None, "spearman": None}
    return {"pearson": float(pearsonr(x, y).statistic), "spearman": float(spearmanr(x, y).statistic)}


def system_sensitivities(
    actions: np.ndarray,
    metrics: dict[str, np.ndarray],
    data: dict[str, np.ndarray],
    utilities: np.ndarray,
) -> dict[str, Any]:
    """Report per-policy and observational-slice metrics without changing selection."""
    primary = data["primary"]
    oracle = authorized_actions(data["target"], utilities)
    regret = metrics["regret_by_policy"]
    compatible = metrics["compatible_by_policy"]
    by_policy = []
    for policy in range(actions.shape[1]):
        by_policy.append({
            "policy_index": policy,
            "accuracy": float(np.mean(actions[primary, policy] == oracle[primary, policy])),
            "compatible": float(np.mean(compatible[primary, policy])),
            "regret": float(np.mean(regret[primary, policy])),
        })
    by_slice = {}
    for stratum in sorted(set(data["design_stratum"].astype(str))):
        for cardinality in sorted(set(data["cardinality"].astype(int))):
            mask = (
                (data["design_stratum"].astype(str) == stratum)
                & (data["cardinality"].astype(int) == cardinality)
            )
            if np.any(mask):
                by_slice[f"{stratum}|cardinality={cardinality}"] = summarize(metrics, mask)
    return {"primary_by_policy": by_policy, "all_in_catalog_by_observational_slice": by_slice}


def analyze_population(
    label: str,
    fit_data: dict[str, np.ndarray],
    eval_data: dict[str, np.ndarray],
    utilities: np.ndarray,
    config: dict[str, Any],
    arrays: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Run the frozen Stage 0 family selection for one token population."""
    folds = make_folds(fit_data, config)
    arrays[f"{label}__dev_fit_fold"] = folds
    model_results: dict[str, Any] = {}
    fitted: dict[str, Any] = {}
    for name, spec in config["models"].items():
        eval_rows = fit_rows(eval_data, eval_data["primary"])
        if spec["kind"] == "logistic" and np.unique((eval_data["gain"][eval_rows] > 1e-12)).size < 2:
            model_results[name] = {"status": "NOT_EVALUABLE", "reason": "dev_eval has one gain class"}
            continue
        selection = choose_hyperparameters(fit_data, folds, spec)
        serial_candidates = []
        for candidate_index, row in enumerate(selection["candidates"]):
            serialized = {key: value for key, value in row.items() if key != "scores"}
            serial_candidates.append(serialized)
            if row["status"] == "PASS":
                arrays[f"{label}__oof_candidate__{name}__{candidate_index}"] = row["scores"]
        if selection["status"] != "PASS":
            model_results[name] = {"status": selection["status"], "candidates": serial_candidates}
            continue
        chosen = selection["selected"]
        oof_scores = chosen["scores"]
        operating = select_operating_point(
            oof_scores, fit_data, utilities, config, token_mask=fit_data["primary"]
        )
        result = fit_full_and_evaluate(
            fit_data,
            eval_data,
            spec,
            chosen["params"],
            operating["selected"]["q"],
            fit_data["gain"],
            utilities,
            config,
        )
        arrays[f"{label}__oof_score__{name}"] = oof_scores
        arrays[f"{label}__eval_score__{name}"] = result["eval_scores"]
        arrays[f"{label}__eval_action__{name}"] = result["actions"]
        arrays[f"{label}__eval_override__{name}"] = result["override"]
        for metric, values in result["metrics"].items():
            arrays[f"{label}__eval_metric__{name}__{metric}"] = values
        fitted[name] = {"spec": spec, "params": chosen["params"], **result}
        model_results[name] = {
            "status": "PASS",
            "selected_params": chosen["params"],
            "oof_objective": float(chosen["objective"]),
            "candidates": serial_candidates,
            "operating_point_oof": operating,
            "full_model": model_state_json(result["state"]),
            "eval_threshold": result["threshold"],
            "dev_eval": {**result["summary"], "coverage": result["coverage"]},
            "override_diagnostics": override_summary(
                result["actions"], result["override"], eval_data, utilities, config
            ),
            "correlation_oof": correlation_summary(oof_scores, fit_data),
            "correlation_dev_eval": correlation_summary(result["eval_scores"], eval_data),
            "sensitivities": system_sensitivities(
                result["actions"], result["metrics"], eval_data, utilities
            ),
        }

    scalar_operating = select_operating_point(
        fit_data["advantage"], fit_data, utilities, config, token_mask=fit_data["primary"]
    )
    scalar_q = scalar_operating["selected"]["q"]
    scalar_threshold = HARD_ONLY if scalar_q == HARD_ONLY else float(
        np.quantile(
            fit_data["advantage"][fit_rows(fit_data, fit_data["primary"])],
            float(scalar_q),
            method="higher",
        )
    )
    scalar_gate = apply_gate(
        eval_data["advantage"], eval_data["hard_actions"], eval_data["posterior_actions"], scalar_threshold
    )
    scalar_metrics = action_metric_arrays(
        scalar_gate["actions"], eval_data["target"], utilities, float(config["incompatible_regret_penalty"])
    )
    arrays[f"{label}__eval_action__scalar_advantage"] = scalar_gate["actions"]
    arrays[f"{label}__eval_override__scalar_advantage"] = scalar_gate["override"]
    for metric, values in scalar_metrics.items():
        arrays[f"{label}__eval_metric__scalar_advantage__{metric}"] = values
    scalar_result = {
        "operating_point": scalar_operating,
        "threshold": scalar_threshold,
        "dev_eval": {
            **summarize(scalar_metrics, eval_data["primary"]),
            "coverage": float(scalar_gate["override"][eval_data["primary"]].mean()),
        },
        "override_diagnostics": override_summary(
            scalar_gate["actions"], scalar_gate["override"], eval_data, utilities, config
        ),
    }

    hard_eval = action_metric_arrays(
        eval_data["hard_actions"], eval_data["target"], utilities, float(config["incompatible_regret_penalty"])
    )
    pure_eval = action_metric_arrays(
        eval_data["posterior_actions"], eval_data["target"], utilities, float(config["incompatible_regret_penalty"])
    )
    hard_summary = summarize(hard_eval, eval_data["primary"])
    tie_order = {name: index for index, name in enumerate(config["selection"]["family_tie_order"])}
    eligible = []
    for name in CONTEXTUAL_FAMILIES:
        row = model_results.get(name, {})
        if row.get("status") != "PASS":
            continue
        metrics = row["dev_eval"]
        feasible = (
            metrics["accuracy"] >= hard_summary["accuracy"] - float(config["selection"]["accuracy_noninferiority_margin"])
            and metrics["compatible"]
            >= hard_summary["compatible"]
            - float(config["selection"]["compatible_noninferiority_margin"])
        )
        row["dev_eval_feasible"] = bool(feasible)
        if feasible:
            eligible.append(name)
    selected_family = None
    if eligible:
        atol = float(config["selection"]["tie_atol"])
        best_regret = min(float(model_results[name]["dev_eval"]["regret"]) for name in eligible)
        tied = [name for name in eligible if float(model_results[name]["dev_eval"]["regret"]) <= best_regret + atol]
        selected_family = min(
            tied,
            key=lambda name: (tie_order[name], float(model_results[name]["dev_eval"]["coverage"])),
        )
    return {
        "folds": folds,
        "models": model_results,
        "fitted": fitted,
        "scalar": scalar_result,
        "scalar_metrics": scalar_metrics,
        "scalar_actions": scalar_gate["actions"],
        "hard_metrics": hard_eval,
        "pure_metrics": pure_eval,
        "pure_summary": summarize(pure_eval, eval_data["primary"]),
        "hard_summary": hard_summary,
        "selected_family": selected_family,
    }


def run_shuffled_controls(
    label: str,
    fit_data: dict[str, np.ndarray],
    eval_data: dict[str, np.ndarray],
    analysis: dict[str, Any],
    utilities: np.ndarray,
    config: dict[str, Any],
    arrays: dict[str, np.ndarray],
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray] | None, dict[str, float] | None]:
    """Run the five frozen shuffled-gain controls for one analysis population."""
    selected_family = analysis["selected_family"]
    shuffled_results: list[dict[str, Any]] = []
    shuffled_metric_arrays: list[dict[str, np.ndarray]] = []
    if selected_family is None:
        return shuffled_results, None, None
    selected = analysis["fitted"][selected_family]
    for seed in config["shuffle_seeds"]:
        sub = stratified_gain_shuffle(
            fit_data["gain"][fit_data["primary"]],
            fit_data["disagreement"][fit_data["primary"]],
            int(seed),
        )
        shuffled_target = fit_data["gain"].copy()
        shuffled_target[fit_data["primary"]] = sub["target"]
        arrays[f"{label}__shuffle_mapping__{seed}"] = sub["mapping"]
        arrays[f"{label}__shuffle_target__{seed}"] = shuffled_target
        if sub["movable_fraction"] < float(config["minimums"]["shuffle_movable_fraction"]):
            shuffled_results.append({
                "seed": int(seed),
                "status": "NOT_EVALUABLE",
                "movable_fraction": sub["movable_fraction"],
                "moved_fraction": sub["moved_fraction"],
            })
            continue
        crossfit = crossfit_fixed(
            fit_data, analysis["folds"], selected["spec"], selected["params"], shuffled_target
        )
        oof = crossfit["scores"]
        operating = select_operating_point(
            oof, fit_data, utilities, config, token_mask=fit_data["primary"]
        )
        result = fit_full_and_evaluate(
            fit_data,
            eval_data,
            selected["spec"],
            selected["params"],
            operating["selected"]["q"],
            shuffled_target,
            utilities,
            config,
        )
        arrays[f"{label}__shuffle_oof_score__{seed}"] = oof
        arrays[f"{label}__shuffle_eval_score__{seed}"] = result["eval_scores"]
        arrays[f"{label}__shuffle_eval_action__{seed}"] = result["actions"]
        arrays[f"{label}__shuffle_eval_override__{seed}"] = result["override"]
        for metric, values in result["metrics"].items():
            arrays[f"{label}__shuffle_eval_metric__{seed}__{metric}"] = values
        shuffled_metric_arrays.append(result["metrics"])
        shuffled_results.append({
            "seed": int(seed),
            "status": "PASS",
            "movable_fraction": sub["movable_fraction"],
            "moved_fraction": sub["moved_fraction"],
            "operating_point_oof": operating,
            "selected_q": operating["selected"]["q"],
            "threshold": result["threshold"],
            "fold_models": crossfit["fold_models"],
            "full_model": model_state_json(result["state"]),
            "dev_eval": {**result["summary"], "coverage": result["coverage"]},
            "override_diagnostics": override_summary(
                result["actions"], result["override"], eval_data, utilities, config
            ),
        })
    if not shuffled_metric_arrays:
        return shuffled_results, None, None
    averaged = {
        key: np.mean(np.stack([row[key] for row in shuffled_metric_arrays]), axis=0)
        for key in ("accuracy", "compatible", "regret", "worst_regret")
    }
    for key, value in averaged.items():
        arrays[f"{label}__shuffle_average_metric__{key}"] = value
    return shuffled_results, averaged, summarize(averaged, eval_data["primary"])


def contrast_signs(
    analysis: dict[str, Any],
    shuffled_average: dict[str, float] | None,
    atol: float,
) -> dict[str, int] | None:
    """Return every predeclared system-contrast sign for a sensitivity population."""
    family = analysis["selected_family"]
    if family is None:
        return None
    contextual = analysis["models"][family]["dev_eval"]
    hard = analysis["hard_summary"]
    scalar = analysis["scalar"]["dev_eval"]
    advantage = analysis["models"].get("ridge_advantage_only", {}).get("dev_eval")
    pure = analysis["pure_summary"]

    def sign(value: float) -> int:
        return 0 if abs(value) <= atol else int(np.sign(value))

    values = {
        "regret_vs_hard": contextual["regret"] - hard["regret"],
        "accuracy_vs_hard": contextual["accuracy"] - hard["accuracy"],
        "compatible_vs_hard": contextual["compatible"] - hard["compatible"],
        "regret_vs_scalar": contextual["regret"] - scalar["regret"],
        "accuracy_vs_scalar": contextual["accuracy"] - scalar["accuracy"],
        "compatible_vs_scalar": contextual["compatible"] - scalar["compatible"],
        "regret_vs_pure_joint": contextual["regret"] - pure["regret"],
        "accuracy_vs_pure_joint": contextual["accuracy"] - pure["accuracy"],
    }
    if advantage is not None:
        values["regret_vs_advantage_only"] = contextual["regret"] - advantage["regret"]
    if shuffled_average is not None:
        values["regret_vs_shuffle_average"] = contextual["regret"] - shuffled_average["regret"]
    return {name: sign(float(value)) for name, value in values.items()}


def compare_reference(output: Path, reference: Path) -> dict[str, Any]:
    def arrays_equal(left: np.ndarray, right: np.ndarray) -> bool:
        if left.dtype.kind in "fc" and right.dtype.kind in "fc":
            return bool(np.array_equal(left, right, equal_nan=True))
        return bool(np.array_equal(left, right))

    checks = {}
    for name in ("analysis_core.json", "selection_freeze.json", "feature_schema.json"):
        checks[name] = (output / name).read_bytes() == (reference / name).read_bytes()
    with np.load(output / "result_arrays.npz", allow_pickle=False) as left, np.load(reference / "result_arrays.npz", allow_pickle=False) as right:
        checks["result_arrays.npz"] = set(left.files) == set(right.files) and all(
            arrays_equal(left[key], right[key]) for key in left.files
        )
    if not all(checks.values()):
        raise RuntimeError(f"Wave 56 Stage 0 replay mismatch: {checks}")
    return {"checks": checks, "all_exact": True}


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    config_path = args.config.resolve(strict=True)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    commit = require_sources_at_head([Path(__file__), PRIMITIVES, PLAN_PATH, NOTE_PATH, config_path])
    wave55 = args.wave55_dir.resolve(strict=True)
    wave54 = args.wave54_dir.resolve(strict=True)
    wave52 = args.wave52_dir.resolve(strict=True)
    binding = config["source_binding"]
    fit_path = wave55 / "bundles/decision_select.npz"
    eval_path = wave55 / "bundles/sealed_monitor.npz"
    require_hash(fit_path, binding["wave55_decision_select_sha256"])
    require_hash(eval_path, binding["wave55_sealed_monitor_sha256"])
    require_hash(wave54 / "selection_state.npz", binding["wave54_selection_state_sha256"])
    require_hash(wave54 / "selection_freeze.json", binding["wave54_selection_freeze_sha256"])
    require_hash(wave52 / "policy_manifest.json", binding["wave52_policy_manifest_sha256"])
    archived = prepare_output(output, args.force)

    utilities, policy_groups = load_utilities(wave52 / "policy_manifest.json")
    with np.load(wave54 / "selection_state.npz", allow_pickle=False) as state:
        theta = state["theta__joint_full"]
    fit_data = make_dataset(load_bundle(fit_path), theta, utilities, config)
    eval_data = make_dataset(load_bundle(eval_path), theta, utilities, config)
    counts = {
        "dev_fit": check_minimums(fit_data, config, "dev_fit"),
        "dev_eval": check_minimums(eval_data, config, "dev_eval"),
    }
    arrays: dict[str, np.ndarray] = {}
    for role, data in (("dev_fit", fit_data), ("dev_eval", eval_data)):
        for key in (
            "pair_token", "target", "per_seed_logits", "ensemble_logits", "design_stratum",
            "cardinality", "design", "gain", "disagreement", "weights", "primary",
            "hard_actions", "posterior_actions", "hard_set", "advantage",
        ):
            arrays[f"{role}__{key}"] = data[key]

    primary = analyze_population("primary", fit_data, eval_data, utilities, config, arrays)
    selected_family = primary["selected_family"]

    fit_all = dict(fit_data)
    eval_all = dict(eval_data)
    fit_all["primary"] = np.ones(len(fit_data["pair_token"]), dtype=bool)
    eval_all["primary"] = np.ones(len(eval_data["pair_token"]), dtype=bool)
    global_sensitivity = analyze_population(
        "all_in_catalog", fit_all, eval_all, utilities, config, arrays
    )

    shuffled_results, averaged_shuffle_arrays, shuffled_average = run_shuffled_controls(
        "primary", fit_data, eval_data, primary, utilities, config, arrays
    )
    (
        global_shuffled_results,
        global_averaged_shuffle_arrays,
        global_shuffled_average,
    ) = run_shuffled_controls(
        "all_in_catalog", fit_all, eval_all, global_sensitivity, utilities, config, arrays
    )

    # Genuine leave-policy-group-out: every selection step sees only the 16 train policies.
    leave_policy_rows = []
    heldout = {key: np.full((len(eval_data["pair_token"]), 24), np.nan) for key in ("accuracy", "compatible", "regret")}
    all_policies = np.arange(24)
    for group_index, heldout_group in enumerate(policy_groups):
        heldout_group = np.asarray(heldout_group, dtype=np.int64)
        train_policies = np.setdiff1d(all_policies, heldout_group)
        train_policy_mask = np.zeros(24, dtype=bool)
        train_policy_mask[train_policies] = True
        local_data = dict(fit_data)
        local_disagreement = fit_data["disagreement"] & train_policy_mask[None, :]
        local_data["weights"] = disagreement_weights(local_disagreement)
        family_rows = {}
        hard_train_metrics = metric_arrays_for_policies(
            eval_data["hard_actions"], eval_data, utilities,
            float(config["incompatible_regret_penalty"]), train_policies,
        )
        hard_train_summary = summarize(hard_train_metrics, eval_data["primary"])
        for name in CONTEXTUAL_FAMILIES:
            spec = config["models"][name]
            eval_train_rows = fit_rows(eval_data, eval_data["primary"], train_policy_mask)
            if (
                spec["kind"] == "logistic"
                and np.unique((eval_data["gain"][eval_train_rows] > 1e-12)).size < 2
            ):
                continue
            selection = choose_hyperparameters(
                local_data, primary["folds"], spec, train_policy_mask
            )
            if selection["status"] != "PASS":
                continue
            serialized_candidates = []
            for candidate_index, candidate in enumerate(selection["candidates"]):
                serialized_candidates.append({
                    key: value for key, value in candidate.items() if key != "scores"
                })
                if candidate["status"] == "PASS":
                    arrays[
                        f"leave_policy__group{group_index}__oof_candidate__{name}__{candidate_index}"
                    ] = candidate["scores"]
            chosen = selection["selected"]
            operating = select_operating_point(
                chosen["scores"], local_data, utilities, config,
                token_mask=local_data["primary"], policy_indices=train_policies,
            )
            evaluated = fit_full_and_evaluate(
                local_data,
                eval_data,
                spec,
                chosen["params"],
                operating["selected"]["q"],
                local_data["gain"],
                utilities,
                config,
                fit_policy_indices=train_policies,
            )
            train_metrics = metric_arrays_for_policies(
                evaluated["actions"], eval_data, utilities,
                float(config["incompatible_regret_penalty"]), train_policies,
            )
            train_summary = summarize(train_metrics, eval_data["primary"])
            arrays[f"leave_policy__group{group_index}__eval_score__{name}"] = evaluated["eval_scores"]
            arrays[f"leave_policy__group{group_index}__eval_action__{name}"] = evaluated["actions"]
            arrays[f"leave_policy__group{group_index}__eval_override__{name}"] = evaluated["override"]
            feasible = (
                train_summary["accuracy"]
                >= hard_train_summary["accuracy"]
                - float(config["selection"]["accuracy_noninferiority_margin"])
                and train_summary["compatible"]
                >= hard_train_summary["compatible"]
                - float(config["selection"]["compatible_noninferiority_margin"])
            )
            family_rows[name] = {
                "spec": spec, "params": chosen["params"], "operating": operating,
                "evaluated": evaluated, "dev_eval_train": train_summary,
                "feasible": bool(feasible), "candidates": serialized_candidates,
            }
        eligible_rows = {name: row for name, row in family_rows.items() if row["feasible"]}
        if not eligible_rows:
            leave_policy_rows.append({"group": group_index, "status": "NOT_EVALUABLE"})
            continue
        best_regret = min(row["dev_eval_train"]["regret"] for row in eligible_rows.values())
        atol = float(config["selection"]["tie_atol"])
        tie_order = {name: index for index, name in enumerate(config["selection"]["family_tie_order"])}
        selected_lp = min(
            [
                name for name, row in eligible_rows.items()
                if row["dev_eval_train"]["regret"] <= best_regret + atol
            ],
            key=lambda name: tie_order[name],
        )
        row = family_rows[selected_lp]
        evaluated = row["evaluated"]
        q = row["operating"]["selected"]["q"]
        selected_actions = evaluated["actions"][:, heldout_group]
        selected_utilities = utilities[heldout_group]
        oracle = authorized_actions(eval_data["target"], selected_utilities)
        regret = constrained_regret(
            selected_actions, eval_data["target"], selected_utilities,
            float(config["incompatible_regret_penalty"]),
        )
        compatible = eval_data["target"][np.arange(len(eval_data["target"]))[:, None], selected_actions]
        heldout["accuracy"][:, heldout_group] = selected_actions == oracle
        heldout["compatible"][:, heldout_group] = compatible
        heldout["regret"][:, heldout_group] = regret
        leave_policy_rows.append({
            "group": group_index, "status": "PASS", "heldout": heldout_group.tolist(),
            "selected_family": selected_lp, "selected_params": row["params"],
            "selected_q": q, "threshold": evaluated["threshold"],
            "model": model_state_json(evaluated["state"]),
            "dev_eval_train": row["dev_eval_train"],
            "family_candidates": {
                name: {
                    "selected_params": candidate["params"],
                    "selected_q": candidate["operating"]["selected"]["q"],
                    "dev_eval_train": candidate["dev_eval_train"],
                    "feasible": candidate["feasible"],
                    "operating_point_oof": candidate["operating"],
                    "full_model": model_state_json(candidate["evaluated"]["state"]),
                    "candidates": candidate["candidates"],
                }
                for name, candidate in family_rows.items()
            },
        })
    leave_policy_summary = None
    if all(row["status"] == "PASS" for row in leave_policy_rows):
        leave_policy_summary = {
            metric: float(np.nanmean(values[eval_data["primary"]])) for metric, values in heldout.items()
        }
        for metric, values in heldout.items():
            arrays[f"leave_policy__{metric}"] = values

    primary_indices = np.flatnonzero(eval_data["primary"])
    bootstrap = paired_bootstrap_indices(len(primary_indices), config)
    arrays["bootstrap_indices"] = bootstrap
    contrasts = {}
    if selected_family is not None:
        selected_metrics = primary["fitted"][selected_family]["metrics"]
        references = {
            "hard": primary["hard_metrics"],
            "pure_joint": primary["pure_metrics"],
            "scalar_advantage": primary["scalar_metrics"],
        }
        advantage_model = primary["fitted"].get("ridge_advantage_only")
        if advantage_model:
            references["advantage_only"] = advantage_model["metrics"]
        if averaged_shuffle_arrays:
            references["shuffle_average"] = averaged_shuffle_arrays
        for reference_name, reference_metrics in references.items():
            contrasts[f"contextual_minus_{reference_name}"] = {
                metric: paired_delta_ci(
                    selected_metrics[metric][primary_indices],
                    reference_metrics[metric][primary_indices],
                    bootstrap,
                )
                for metric in ("accuracy", "compatible", "regret", "worst_regret")
            }

    sign_atol = float(config["selection"]["tie_atol"])
    global_summary = {
        "selected_family": global_sensitivity["selected_family"],
        "selected_params": (
            global_sensitivity["models"][global_sensitivity["selected_family"]]["selected_params"]
            if global_sensitivity["selected_family"] else None
        ),
        "primary_signs": contrast_signs(primary, shuffled_average, sign_atol),
        "all_in_catalog_signs": contrast_signs(
            global_sensitivity, global_shuffled_average, sign_atol
        ),
        "shuffle_controls": global_shuffled_results,
        "shuffle_average_dev_eval": global_shuffled_average,
    }
    global_summary["changed"] = (
        global_summary["selected_family"] != selected_family
        or global_summary["primary_signs"] != global_summary["all_in_catalog_signs"]
        or global_summary["selected_params"] != (
            primary["models"][selected_family]["selected_params"] if selected_family else None
        )
    )

    selection_freeze = {
        "phase": "retrospective-family-selection-only",
        "selected_family": selected_family,
        "selected_params": primary["models"][selected_family]["selected_params"] if selected_family else None,
        "prospective_generation_authorized": False,
        "second_independent_audit_required": True,
    }
    write_json_strict(output / "selection_freeze.json", selection_freeze)
    write_json_strict(output / "feature_schema.json", {"features": feature_schema()})
    core = {
        "status": "RETROSPECTIVE_OPENED_DEVELOPMENT",
        "counts": counts,
        "feature_names": list(FEATURE_NAMES),
        "models": primary["models"],
        "scalar_advantage": primary["scalar"],
        "hard_dev_eval": primary["hard_summary"],
        "selected_family": selected_family,
        "contrasts": contrasts,
        "shuffle_controls": shuffled_results,
        "shuffle_average_dev_eval": shuffled_average,
        "leave_policy_group_out": {"folds": leave_policy_rows, "aggregate": leave_policy_summary},
        "all_in_catalog_sensitivity": global_summary,
        "all_in_catalog_models": global_sensitivity["models"],
        "claim_scope": "opened development data; fixed 24-policy catalogue; no prospective adjudication",
    }
    write_json_strict(output / "analysis_core.json", core)
    np.savez_compressed(output / "result_arrays.npz", **arrays)
    replay = compare_reference(output, args.reference_dir.resolve(strict=True)) if args.reference_dir else None
    if replay:
        write_json_strict(output / "replay_receipt.json", replay)
    write_json_strict(
        output / "runtime.json",
        {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "scikit-learn")},
            "device": "cpu",
        },
    )
    write_json_strict(
        output / "REPORT_WAVE56_RETROSPECTIVE.json",
        {
            **core,
            "git_commit": commit,
            "config_sha256": digest(config_path),
            "plan_sha256": digest(PLAN_PATH),
            "replay": replay,
            "superseded_output": str(archived) if archived else None,
            "decision_authority": "user",
        },
    )
    files = sorted(path for path in output.iterdir() if path.is_file() and path.name != "artifact_manifest.json")
    write_json_strict(
        output / "artifact_manifest.json",
        {"files": {path.name: {"sha256": digest(path), "bytes": path.stat().st_size} for path in files}},
    )
    print(json.dumps({"selected_family": selected_family, "counts": counts}, sort_keys=True))


if __name__ == "__main__":
    main()
