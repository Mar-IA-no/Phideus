"""CPU-only primitives for the adaptive Wave 58 open-data diagnostic."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import itertools
import json
from typing import Any, Iterable
import warnings

import numpy as np
from scipy.special import expit
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Ridge

from .wave52_policy import authorized_actions, constrained_regret
from .wave55_policy_bridge import action_metric_arrays
from .wave56_contextual_gate import fit_weighted_scaler


EPSILON = 1e-12
EXPECTED_SKLEARN_VERSION = "1.8.0"
Q_PROPOSER = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975)
Q_GUARD_8 = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
Q_GUARD_9 = (*Q_GUARD_8, 0.9)
Q_GUARD_WAVE57 = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8)
TARGET_ORDER = (
    "harm",
    "compatibility_loss",
    "posterior_incompatibility",
    "accuracy_loss",
    "tail_breach",
    "max_harm_tail",
)
RIDGE_KWARGS = {
    "alpha": 1.0,
    "copy_X": True,
    "fit_intercept": True,
    "max_iter": None,
    "positive": False,
    "random_state": None,
    "solver": "svd",
    "tol": 1e-4,
}
LOGISTIC_KWARGS = {
    "C": 1.0,
    "class_weight": None,
    "dual": False,
    "fit_intercept": True,
    "intercept_scaling": 1,
    "l1_ratio": 0.0,
    "max_iter": 2000,
    "n_jobs": None,
    "penalty": "deprecated",
    "random_state": None,
    "solver": "lbfgs",
    "tol": 1e-10,
    "verbose": 0,
    "warm_start": False,
}
HGB_COMMON = {
    "categorical_features": "from_dtype",
    "early_stopping": False,
    "interaction_cst": None,
    "l2_regularization": 1.0,
    "learning_rate": 0.05,
    "max_bins": 255,
    "max_depth": None,
    "max_features": 1.0,
    "max_iter": 100,
    "max_leaf_nodes": 7,
    "min_samples_leaf": 20,
    "monotonic_cst": None,
    "n_iter_no_change": 10,
    "scoring": "loss",
    "tol": 1e-7,
    "validation_fraction": 0.1,
    "verbose": 0,
    "warm_start": False,
}
HGB_REGRESSOR_KWARGS = {
    **HGB_COMMON,
    "loss": "squared_error",
    "quantile": None,
    "random_state": 5801,
}
HGB_CLASSIFIER_KWARGS = {
    **HGB_COMMON,
    "class_weight": None,
    "loss": "log_loss",
}


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    header = json.dumps(
        {"dtype": array.dtype.str, "shape": list(array.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(header + b"\0" + array.tobytes()).hexdigest()


def derive_targets(
    data: dict[str, np.ndarray], utilities: np.ndarray, penalty: float
) -> dict[str, np.ndarray]:
    target = np.asarray(data["target"], dtype=bool)
    hard = np.asarray(data["hard_actions"], dtype=np.int64)
    posterior = np.asarray(data["posterior_actions"], dtype=np.int64)
    gain = np.asarray(data["gain"], dtype=np.float64)
    if hard.shape != posterior.shape or gain.shape != hard.shape:
        raise ValueError("Wave 58 action/target arrays do not align")
    rows = np.arange(len(target))[:, None]
    oracle = authorized_actions(target, utilities)
    hard_compatible = target[rows, hard]
    posterior_compatible = target[rows, posterior]
    hard_regret = constrained_regret(hard, target, utilities, penalty)
    posterior_regret = constrained_regret(posterior, target, utilities, penalty)
    reconstructed_gain = hard_regret - posterior_regret
    np.testing.assert_allclose(gain, reconstructed_gain, rtol=0.0, atol=2e-15)
    result = {
        "gain": gain.copy(),
        "harm": gain < -EPSILON,
        "compatibility_loss": hard_compatible & ~posterior_compatible,
        "posterior_incompatibility": ~posterior_compatible,
        "accuracy_loss": (hard == oracle) & (posterior != oracle),
        "tail_breach": posterior_regret
        > hard_regret.max(axis=1)[:, None] + EPSILON,
    }
    for name, values in result.items():
        if values.shape != hard.shape or (name != "gain" and values.dtype != bool):
            raise AssertionError(f"invalid target {name}")
    return result


def fit_mask(data: dict[str, np.ndarray]) -> np.ndarray:
    return np.asarray(data["primary"], dtype=bool)[:, None] & np.asarray(
        data["disagreement"], dtype=bool
    )


def fit_ridge_state(
    design: np.ndarray, target: np.ndarray, weights: np.ndarray
) -> tuple[dict[str, Any], Ridge | None]:
    raw = np.asarray(design, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    sample_weight = np.asarray(weights, dtype=np.float64)
    if (
        raw.ndim != 2
        or y.shape != (len(raw),)
        or sample_weight.shape != (len(raw),)
        or not len(raw)
        or np.any(sample_weight <= 0.0)
        or not np.all(np.isfinite(raw))
        or not np.all(np.isfinite(y))
        or not np.all(np.isfinite(sample_weight))
    ):
        return {
            "kind": "ridge",
            "status": "NOT_EVALUABLE",
            "reason": "invalid_fit_arrays",
            "kwargs": RIDGE_KWARGS,
        }, None
    # Preserve the Wave 57 all-column advanced-index layout: NumPy reductions
    # can differ by a few ulps between C/F traversal even for identical values.
    x = raw[:, np.arange(raw.shape[1], dtype=np.int64)]
    scaler = fit_weighted_scaler(x, weights)
    model = Ridge(**RIDGE_KWARGS).fit(
        scaler.transform(x), y, sample_weight=sample_weight
    )
    state = {
        "kind": "ridge",
        "kwargs": RIDGE_KWARGS,
        "mean": scaler.mean.tolist(),
        "scale": scaler.scale.tolist(),
        "coef": np.asarray(model.coef_, dtype=np.float64).tolist(),
        "intercept": float(model.intercept_),
    }
    return state, model


def score_linear_state(state: dict[str, Any], design: np.ndarray) -> np.ndarray:
    mean = np.asarray(state["mean"], dtype=np.float64)
    scale = np.asarray(state["scale"], dtype=np.float64)
    coef = np.asarray(state["coef"], dtype=np.float64)
    x = np.asarray(design, dtype=np.float64)
    if state["kind"] == "ridge":
        x = x[:, np.arange(x.shape[1], dtype=np.int64)]
    xs = (x - mean) / scale
    raw = xs @ coef + float(state["intercept"])
    return expit(raw) if state["kind"] == "logistic" else raw


def fit_logistic_state(
    design: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    *,
    class_weight: str | None = None,
) -> tuple[dict[str, Any], LogisticRegression] | tuple[dict[str, Any], None]:
    y = np.asarray(target, dtype=np.int8)
    x = np.asarray(design, dtype=np.float64)
    sample_weight = np.asarray(weights, dtype=np.float64)
    kwargs = {**LOGISTIC_KWARGS, "class_weight": class_weight}
    if (
        x.ndim != 2
        or y.shape != (len(x),)
        or sample_weight.shape != (len(x),)
        or not len(x)
        or np.any(sample_weight <= 0.0)
        or not np.all(np.isfinite(x))
        or not np.all(np.isfinite(sample_weight))
    ):
        return {
            "kind": "logistic",
            "status": "NOT_EVALUABLE",
            "reason": "invalid_fit_arrays",
            "kwargs": kwargs,
        }, None
    unique = np.unique(y)
    if not np.array_equal(unique, np.asarray([0, 1], dtype=np.int8)):
        return {
            "kind": "logistic",
            "status": "NOT_EVALUABLE",
            "reason": "single_class",
            "classes_observed": unique.astype(int).tolist(),
            "kwargs": kwargs,
        }, None
    scaler = fit_weighted_scaler(x, sample_weight)
    model = LogisticRegression(**kwargs)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(scaler.transform(x), y, sample_weight=sample_weight)
    if any(issubclass(item.category, ConvergenceWarning) for item in caught):
        return {
            "kind": "logistic",
            "status": "NOT_EVALUABLE",
            "reason": "non_convergence",
            "kwargs": kwargs,
        }, None
    state = {
        "kind": "logistic",
        "status": "PASS",
        "kwargs": kwargs,
        "classes": model.classes_.astype(int).tolist(),
        "mean": scaler.mean.tolist(),
        "scale": scaler.scale.tolist(),
        "coef": np.asarray(model.coef_[0], dtype=np.float64).tolist(),
        "intercept": float(model.intercept_[0]),
        "n_iter": np.asarray(model.n_iter_, dtype=np.int64).tolist(),
    }
    return state, model


def _export_hgb(
    name: str, model: HistGradientBoostingRegressor | HistGradientBoostingClassifier
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    arrays: dict[str, np.ndarray] = {}
    tree_keys = []
    for iteration, predictors in enumerate(model._predictors):
        if len(predictors) != 1:
            raise RuntimeError("Wave 58 expects one HGB tree per iteration")
        predictor = predictors[0]
        key = f"hgb__{name}__tree__{iteration:03d}"
        arrays[key] = predictor.nodes.copy()
        arrays[f"{key}__binned"] = predictor.binned_left_cat_bitsets.copy()
        arrays[f"{key}__raw"] = predictor.raw_left_cat_bitsets.copy()
        tree_keys.append(key)
    state = {
        "kind": "hgb_classifier" if isinstance(model, HistGradientBoostingClassifier) else "hgb_regressor",
        "status": "PASS",
        "kwargs": model.get_params(deep=False),
        "baseline_prediction": np.asarray(model._baseline_prediction, dtype=np.float64).tolist(),
        "n_iter": int(model.n_iter_),
        "n_features": int(model.n_features_in_),
        "tree_keys": tree_keys,
        "transport_only": True,
        "score_authority": "preserved_float64_scores_per_split",
        "classes": model.classes_.astype(int).tolist() if isinstance(model, HistGradientBoostingClassifier) else None,
    }
    return state, arrays


def score_hgb_state(
    state: dict[str, Any], arrays: dict[str, np.ndarray], design: np.ndarray
) -> np.ndarray:
    x = np.asarray(design, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != int(state["n_features"]):
        raise ValueError("HGB design shape mismatch")
    raw = np.full(len(x), float(np.asarray(state["baseline_prediction"]).ravel()[0]))
    for key in state["tree_keys"]:
        nodes = arrays[key]
        if np.any(nodes["is_categorical"]):
            raise RuntimeError("Wave 58 portable HGB transport forbids categorical splits")
        position = np.zeros(len(x), dtype=np.int64)
        active = np.ones(len(x), dtype=bool)
        while np.any(active):
            row_ids = np.flatnonzero(active)
            current = nodes[position[row_ids]]
            leaves = current["is_leaf"].astype(bool)
            if np.any(leaves):
                active[row_ids[leaves]] = False
            branch_ids = row_ids[~leaves]
            if not len(branch_ids):
                continue
            branch = current[~leaves]
            feature = branch["feature_idx"].astype(np.int64)
            values = x[branch_ids, feature]
            missing_left = branch["missing_go_to_left"].astype(bool)
            go_left = np.where(
                np.isnan(values), missing_left, values <= branch["num_threshold"]
            )
            position[branch_ids] = np.where(
                go_left, branch["left"], branch["right"]
            ).astype(np.int64)
        raw += nodes[position]["value"]
    return expit(raw) if state["kind"] == "hgb_classifier" else raw


def fit_hgb_state(
    name: str,
    design: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    *,
    classifier: bool,
    seed: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    y = np.asarray(target)
    x = np.asarray(design, dtype=np.float64)
    sample_weight = np.asarray(weights, dtype=np.float64)
    if (
        x.ndim != 2
        or y.shape != (len(x),)
        or sample_weight.shape != (len(x),)
        or not len(x)
        or np.any(sample_weight <= 0.0)
        or not np.all(np.isfinite(x))
        or not np.all(np.isfinite(y))
        or not np.all(np.isfinite(sample_weight))
    ):
        return {
            "kind": "hgb_classifier" if classifier else "hgb_regressor",
            "status": "NOT_EVALUABLE",
            "reason": "invalid_fit_arrays",
        }, {}
    if classifier and not np.array_equal(np.unique(y), np.asarray([False, True])):
        return {
            "kind": "hgb_classifier",
            "status": "NOT_EVALUABLE",
            "reason": "single_class",
            "classes_observed": np.unique(y).astype(int).tolist(),
        }, {}
    if classifier:
        model = HistGradientBoostingClassifier(
            **HGB_CLASSIFIER_KWARGS, random_state=int(seed)
        )
    else:
        model = HistGradientBoostingRegressor(**HGB_REGRESSOR_KWARGS)
        if seed != 5801:
            raise ValueError("HGB regressor seed drifted")
    model.fit(x, y, sample_weight=sample_weight)
    state, arrays = _export_hgb(name, model)
    direct = model.predict_proba(x)[:, 1] if classifier else model.predict(x)
    np.testing.assert_allclose(
        score_hgb_state(state, arrays, x), direct, rtol=0.0, atol=2e-15
    )
    return state, arrays


def score_model(
    state: dict[str, Any], arrays: dict[str, np.ndarray], design: np.ndarray
) -> np.ndarray:
    if state.get("status", "PASS") != "PASS":
        return np.full(len(design), np.nan, dtype=np.float64)
    if str(state["kind"]).startswith("hgb_"):
        return score_hgb_state(state, arrays, design)
    return score_linear_state(state, design)


def score_grid(
    state: dict[str, Any], arrays: dict[str, np.ndarray], data: dict[str, np.ndarray]
) -> np.ndarray:
    result = np.full(data["disagreement"].shape, np.nan, dtype=np.float64)
    active = np.asarray(data["disagreement"], dtype=bool)
    result[active] = score_model(state, arrays, data["design"][active])
    return result


def summarize_actions(
    actions: np.ndarray,
    data: dict[str, np.ndarray],
    utilities: np.ndarray,
    penalty: float,
    token_mask: np.ndarray | None = None,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    metrics = action_metric_arrays(actions, data["target"], utilities, penalty)
    mask = np.asarray(data["primary"] if token_mask is None else token_mask, dtype=bool)
    summary = {
        key: float(np.asarray(metrics[key])[mask].mean())
        for key in ("accuracy", "compatible", "regret", "worst_regret")
    }
    return summary, metrics


def token_support(mask: np.ndarray, token_mask: np.ndarray) -> dict[str, int]:
    selected = np.asarray(mask, dtype=bool) & np.asarray(token_mask, dtype=bool)[:, None]
    return {"rows": int(selected.sum()), "tokens": int(selected.any(axis=1).sum())}


def shard_assignment(tokens: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            hashlib.sha256((str(token) + "wave57-shard").encode()).digest()[-1] & 1
            for token in np.asarray(tokens).astype(str)
        ],
        dtype=np.int8,
    )


def _feasible(
    summary: dict[str, float], hard: dict[str, float]
) -> bool:
    return bool(
        summary["accuracy"] >= hard["accuracy"] - 0.01
        and summary["compatible"] >= hard["compatible"]
        and summary["worst_regret"] <= hard["worst_regret"] + 0.01
    )


def _threshold(values: np.ndarray, q: float) -> float:
    finite = np.asarray(values, dtype=np.float64)
    if not len(finite) or np.any(~np.isfinite(finite)):
        raise ValueError("threshold population must be finite and nonempty")
    return float(np.quantile(finite, float(q), method="higher"))


def _not_evaluable_selection(
    selector: str, guard_names: list[str], reason: str
) -> dict[str, Any]:
    return {
        "status": "NOT_EVALUABLE",
        "reason": reason,
        "selector": selector,
        "selected": {
            "cell_index": 0,
            "terminal": "HARD_ONLY",
            "proposer_q": "HARD_ONLY",
            "proposer_threshold": "HARD_ONLY",
            "guard_names": guard_names,
            "guard_qs": [],
            "guard_thresholds": [],
            "proposal_support": {"rows": 0, "tokens": 0},
            "authorization_support": {"rows": 0, "tokens": 0},
            "evaluable": False,
            "feasible": False,
            "summary": None,
            "shards": {},
        },
        "grid": [],
    }


def select_candidate(
    proposer_score: np.ndarray,
    guard_scores: list[tuple[str, np.ndarray]],
    data: dict[str, np.ndarray],
    utilities: np.ndarray,
    penalty: float,
    *,
    selector: str,
    proposer_quantiles: Iterable[float] = Q_PROPOSER,
    guard_quantiles: Iterable[float] = Q_GUARD_9,
    guard_quantile_grids: Iterable[Iterable[float]] | None = None,
    fixed_proposer_threshold: float | None = None,
) -> dict[str, Any]:
    primary = np.asarray(data["primary"], dtype=bool)
    disagreement = np.asarray(data["disagreement"], dtype=bool)
    hard_summary, _ = summarize_actions(data["hard_actions"], data, utilities, penalty)
    assignments = shard_assignment(data["pair_token"])
    qps = (None,) if fixed_proposer_threshold is not None else tuple(proposer_quantiles)
    if guard_quantile_grids is None:
        guard_grids = [tuple(guard_quantiles) for _ in guard_scores]
    else:
        guard_grids = [tuple(values) for values in guard_quantile_grids]
        if len(guard_grids) != len(guard_scores):
            raise ValueError("one guard quantile grid is required per guard score")
    rows: list[dict[str, Any]] = []
    active_scores = proposer_score[primary[:, None] & disagreement]
    if not len(active_scores) or np.any(~np.isfinite(active_scores)):
        return _not_evaluable_selection(
            selector, [name for name, _ in guard_scores], "invalid_proposer_scores"
        )
    active = primary[:, None] & disagreement
    if any(np.any(~np.isfinite(score[active])) for _, score in guard_scores):
        return _not_evaluable_selection(
            selector, [name for name, _ in guard_scores], "invalid_guard_scores"
        )
    for cell_index, combo in enumerate(itertools.product(qps, *guard_grids)):
        proposer_q, *guard_qs = combo
        proposer_threshold = (
            float(fixed_proposer_threshold)
            if fixed_proposer_threshold is not None
            else _threshold(active_scores, float(proposer_q))
        )
        proposals = disagreement & (proposer_score > proposer_threshold)
        proposal_support = token_support(proposals, primary)
        if proposal_support["rows"] == 0:
            rows.append(
                {
                    "cell_index": cell_index,
                    "proposer_q": proposer_q,
                    "proposer_threshold": proposer_threshold,
                    "guard_names": [name for name, _ in guard_scores],
                    "guard_qs": list(guard_qs),
                    "guard_thresholds": [],
                    "proposal_support": proposal_support,
                    "authorization_support": {"rows": 0, "tokens": 0},
                    "evaluable": False,
                    "feasible": False,
                    "summary": hard_summary,
                    "shards": {},
                    "reason": "empty_proposal",
                }
            )
            continue
        guard_thresholds = [
            _threshold(score[proposals & primary[:, None]], q)
            for (_, score), q in zip(guard_scores, guard_qs, strict=True)
        ]
        authorized = proposals.copy()
        for (_, score), threshold in zip(guard_scores, guard_thresholds, strict=True):
            authorized &= score < threshold
        authorization_support = token_support(authorized, primary)
        actions = np.where(authorized, data["posterior_actions"], data["hard_actions"])
        summary, _ = summarize_actions(actions, data, utilities, penalty)
        evaluable = proposal_support["tokens"] >= 40 and authorization_support["tokens"] >= 25
        feasible = evaluable and _feasible(summary, hard_summary)
        shard_rows: dict[str, Any] = {}
        if selector == "JOINT_SHARD_ROBUST":
            for shard in (0, 1):
                mask = primary & (assignments == shard)
                counts = {
                    "tokens": int(mask.sum()),
                    "disagreement_rows": int((mask[:, None] & disagreement).sum()),
                    "disagreement_tokens": int((mask & disagreement.any(axis=1)).sum()),
                }
                ps = token_support(proposals, mask)
                aus = token_support(authorized, mask)
                shard_summary, _ = summarize_actions(actions, data, utilities, penalty, mask)
                shard_hard, _ = summarize_actions(data["hard_actions"], data, utilities, penalty, mask)
                local_ok = bool(
                    counts["tokens"] >= 40
                    and counts["disagreement_rows"] >= 120
                    and counts["disagreement_tokens"] >= 50
                    and ps["tokens"] >= 20
                    and aus["tokens"] >= 12
                    and _feasible(shard_summary, shard_hard)
                )
                shard_rows[str(shard)] = {
                    "counts": counts,
                    "proposal_support": ps,
                    "authorization_support": aus,
                    "summary": shard_summary,
                    "hard": shard_hard,
                    "feasible": local_ok,
                }
                feasible &= local_ok
        rows.append(
            {
                "cell_index": cell_index,
                "proposer_q": proposer_q,
                "proposer_threshold": proposer_threshold,
                "guard_names": [name for name, _ in guard_scores],
                "guard_qs": list(guard_qs),
                "guard_thresholds": guard_thresholds,
                "proposal_support": proposal_support,
                "authorization_support": authorization_support,
                "evaluable": bool(evaluable),
                "feasible": bool(feasible),
                "summary": summary,
                "shards": shard_rows,
            }
        )
    rows.append(
        {
            "cell_index": len(rows),
            "terminal": "HARD_ONLY",
            "proposer_q": "HARD_ONLY",
            "proposer_threshold": "HARD_ONLY",
            "guard_names": [name for name, _ in guard_scores],
            "guard_qs": [],
            "guard_thresholds": [],
            "proposal_support": {"rows": 0, "tokens": 0},
            "authorization_support": {"rows": 0, "tokens": 0},
            "evaluable": True,
            "feasible": True,
            "summary": hard_summary,
            "shards": {},
        }
    )
    feasible_rows = [row for row in rows if row["feasible"]]
    best_regret = min(row["summary"]["regret"] for row in feasible_rows)
    tied = [row for row in feasible_rows if row["summary"]["regret"] <= best_regret + EPSILON]

    def key(row: dict[str, Any]) -> tuple[Any, ...]:
        terminal = row.get("terminal") == "HARD_ONLY"
        if terminal:
            return (0, 0, 0, 0, 0, 0.0, 0.0, (), (), row["cell_index"])
        return (
            1,
            row["authorization_support"]["rows"],
            row["authorization_support"]["tokens"],
            row["proposal_support"]["rows"],
            row["proposal_support"]["tokens"],
            -float(row["proposer_threshold"]),
            -float(row["proposer_q"]) if row["proposer_q"] is not None else 0.0,
            tuple(float(value) for value in row["guard_thresholds"]),
            tuple(float(value) for value in row["guard_qs"]),
            row["cell_index"],
        )

    selected = min(tied, key=key)
    return {"selector": selector, "selected": selected, "grid": rows}


def select_sequential(
    proposer_score: np.ndarray,
    guard_scores: list[tuple[str, np.ndarray]],
    data: dict[str, np.ndarray],
    utilities: np.ndarray,
    penalty: float,
    *,
    proposer_quantiles: Iterable[float] = Q_PROPOSER,
    guard_quantiles: Iterable[float] = Q_GUARD_9,
) -> dict[str, Any]:
    primary = np.asarray(data["primary"], dtype=bool)
    disagreement = np.asarray(data["disagreement"], dtype=bool)
    hard_summary, _ = summarize_actions(data["hard_actions"], data, utilities, penalty)
    active = primary[:, None] & disagreement
    if (
        not np.any(active)
        or np.any(~np.isfinite(proposer_score[active]))
        or any(np.any(~np.isfinite(score[active])) for _, score in guard_scores)
    ):
        return _not_evaluable_selection(
            "SEQUENTIAL", [name for name, _ in guard_scores], "invalid_model_scores"
        )
    proposer_rows = []
    for cell, q in enumerate(proposer_quantiles):
        threshold = _threshold(proposer_score[active], float(q))
        proposals = disagreement & (proposer_score > threshold)
        support = token_support(proposals, primary)
        actions = np.where(proposals, data["posterior_actions"], data["hard_actions"])
        summary, _ = summarize_actions(actions, data, utilities, penalty)
        evaluable = support["tokens"] >= 40
        feasible = bool(
            evaluable
            and summary["accuracy"] >= hard_summary["accuracy"] - 0.01
            and summary["compatible"] >= hard_summary["compatible"]
        )
        proposer_rows.append(
            {
                "cell_index": cell,
                "q": float(q),
                "threshold": threshold,
                "support": support,
                "evaluable": evaluable,
                "feasible": feasible,
                "summary": summary,
            }
        )
    proposer_rows.append(
        {
            "cell_index": len(proposer_rows),
            "q": "HARD_ONLY",
            "threshold": "HARD_ONLY",
            "support": {"rows": 0, "tokens": 0},
            "evaluable": True,
            "feasible": True,
            "summary": hard_summary,
        }
    )
    feasible = [row for row in proposer_rows if row["feasible"]]
    best = min(row["summary"]["regret"] for row in feasible)
    tied = [row for row in feasible if row["summary"]["regret"] <= best + EPSILON]

    def proposer_key(row: dict[str, Any]) -> tuple[Any, ...]:
        if row["threshold"] == "HARD_ONLY":
            return (0, 0, 0.0, 0.0, row["cell_index"])
        return (
            row["support"]["rows"],
            row["support"]["tokens"],
            -float(row["threshold"]),
            -float(row["q"]),
            row["cell_index"],
        )

    proposer = min(tied, key=proposer_key)
    if proposer["threshold"] == "HARD_ONLY":
        return {
            "selector": "SEQUENTIAL",
            "proposer": {"selected": proposer, "grid": proposer_rows},
            "selected": {
                "cell_index": 0,
                "terminal": "HARD_ONLY",
                "proposer_q": "HARD_ONLY",
                "proposer_threshold": "HARD_ONLY",
                "guard_names": [name for name, _ in guard_scores],
                "guard_qs": [],
                "guard_thresholds": [],
                "proposal_support": {"rows": 0, "tokens": 0},
                "authorization_support": {"rows": 0, "tokens": 0},
                "evaluable": True,
                "feasible": True,
                "summary": hard_summary,
                "shards": {},
            },
            "grid": [],
        }
    guarded = select_candidate(
        proposer_score,
        guard_scores,
        data,
        utilities,
        penalty,
        selector="JOINT",
        proposer_quantiles=(),
        guard_quantiles=guard_quantiles,
        fixed_proposer_threshold=float(proposer["threshold"]),
    )
    guarded["selector"] = "SEQUENTIAL"
    guarded["proposer"] = {"selected": proposer, "grid": proposer_rows}
    guarded["selected"]["proposer_q"] = proposer["q"]
    return guarded


def apply_selection(
    selection: dict[str, Any],
    proposer_score: np.ndarray,
    guard_scores: dict[str, np.ndarray],
    data: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    chosen = selection["selected"]
    if chosen.get("terminal") == "HARD_ONLY":
        empty = np.zeros(data["disagreement"].shape, dtype=bool)
        return data["hard_actions"].copy(), empty, empty
    proposals = data["disagreement"] & (
        proposer_score > float(chosen["proposer_threshold"])
    )
    authorized = proposals.copy()
    for name, threshold in zip(
        chosen["guard_names"], chosen["guard_thresholds"], strict=True
    ):
        authorized &= guard_scores[name] < float(threshold)
    return (
        np.where(authorized, data["posterior_actions"], data["hard_actions"]),
        proposals,
        authorized,
    )


def paired_bootstrap_indices(tokens: np.ndarray, replicates: int = 5000) -> np.ndarray:
    values = np.asarray(tokens).astype(str)
    if values.tolist() != sorted(values.tolist()):
        raise ValueError("bootstrap tokens must be lexicographically ordered")
    rng = np.random.Generator(np.random.PCG64(5807))
    return rng.integers(0, len(values), size=(int(replicates), len(values)), dtype=np.int64)


def paired_delta_ci(left: np.ndarray, right: np.ndarray, indices: np.ndarray) -> dict[str, float]:
    delta = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    sampled = delta[indices].mean(axis=1)
    low, high = np.percentile(sampled, [2.5, 97.5])
    return {"mean_diff": float(delta.mean()), "ci95_low": float(low), "ci95_high": float(high)}


def canonical_candidate_ids() -> list[str]:
    return [
        f"C-{proposer}-{risk}-{guards}-{selector}"
        for proposer in ("RIDGE", "HGB")
        for risk in ("LOGISTIC", "HGB")
        for guards in ("HARM", "INCOMPATIBILITY", "HARM_AND_INCOMPATIBILITY")
        for selector in ("SEQUENTIAL", "JOINT", "JOINT_SHARD_ROBUST")
    ]


def validate_runtime_contract() -> None:
    if sklearn.__version__ != EXPECTED_SKLEARN_VERSION:
        raise RuntimeError("Wave 58 scikit-learn version drifted")
    if Ridge(**RIDGE_KWARGS).get_params(deep=False) != RIDGE_KWARGS:
        raise RuntimeError("Wave 58 Ridge kwargs drifted")
    if LogisticRegression(**LOGISTIC_KWARGS).get_params(deep=False) != LOGISTIC_KWARGS:
        raise RuntimeError("Wave 58 Logistic kwargs drifted")
    if HistGradientBoostingRegressor(**HGB_REGRESSOR_KWARGS).get_params(deep=False) != HGB_REGRESSOR_KWARGS:
        raise RuntimeError("Wave 58 HGB regressor kwargs drifted")
    expected_classifier = {**HGB_CLASSIFIER_KWARGS, "random_state": 5802}
    if HistGradientBoostingClassifier(**expected_classifier).get_params(deep=False) != expected_classifier:
        raise RuntimeError("Wave 58 HGB classifier kwargs drifted")
