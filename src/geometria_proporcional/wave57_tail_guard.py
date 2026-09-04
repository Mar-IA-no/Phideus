"""Inference-safe primitives for the prospective Wave 57 harm guard."""

from __future__ import annotations

import hashlib
import json
import warnings
from typing import Any

import numpy as np
from scipy.special import expit
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from .wave56_contextual_gate import fit_weighted_scaler


HARM_EPSILON = 1e-12
EXPECTED_SKLEARN_VERSION = "1.8.0"
HARM_MODEL_CONTRACT = {
    "C": 1.0,
    "l1_ratio": 0.0,
    "dual": False,
    "solver": "lbfgs",
    "class_weight": None,
    "fit_intercept": True,
    "max_iter": 2000,
    "tol": 1e-10,
    "warm_start": False,
}
FROZEN_CONFIG_SHA256 = "a986b642cd75cb66120232de0df628b82a21c734f189c96e9ac1146bcdedbdda"


def frozen_config_sha256(config: dict[str, Any]) -> str:
    """Hash the complete result-affecting and boundary contract."""
    payload = json.dumps(
        config,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_wave57_frozen_config(config: dict[str, Any]) -> None:
    """Validate the analytical constants that may not drift after the draw."""
    from .wave56_contextual_gate import FEATURE_NAMES

    if frozen_config_sha256(config) != FROZEN_CONFIG_SHA256:
        raise RuntimeError("Wave 57 complete frozen config drifted")

    if config.get("schema_version") != "wave57-contextual-harm-guard-v1":
        raise RuntimeError("Wave 57 schema drifted")
    if config.get("device") != "cpu" or config.get("seeds") != [17, 29, 43]:
        raise RuntimeError("Wave 57 execution identity drifted")
    if tuple(config.get("feature_names", ())) != FEATURE_NAMES:
        raise RuntimeError("Wave 57 feature schema drifted")
    harm = config.get("harm_model", {})
    if {key: harm.get(key) for key in HARM_MODEL_CONTRACT} != HARM_MODEL_CONTRACT:
        raise RuntimeError("Wave 57 Logistic contract drifted")
    if harm.get("positive_class") != "gain_lt_negative_1e-12":
        raise RuntimeError("Wave 57 harm class drifted")
    if harm.get("sklearn_version") != EXPECTED_SKLEARN_VERSION:
        raise RuntimeError("Wave 57 sklearn version drifted")
    if config.get("proposer_quantiles") != [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975]:
        raise RuntimeError("Wave 57 proposer grid drifted")
    if config.get("guard_acceptance_quantiles") != [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8]:
        raise RuntimeError("Wave 57 guard grid drifted")
    if config.get("shuffle_seeds") != [57031, 57032, 57033, 57034, 57035]:
        raise RuntimeError("Wave 57 shuffle seeds drifted")
    if config.get("physical_splits") != {
        "train": "gate_fit",
        "val": "gate_select",
        "lockbox": "sealed_monitor",
    }:
        raise RuntimeError("Wave 57 split roles drifted")
    if config.get("arms") != [
        "hard_set_policy",
        "pure_joint_full",
        "mean_proposer_only",
        "mean_plus_harm_guard",
        "mean_plus_shuffled_harm_guard",
        "advantage_only_value_gate",
        "oracle_positive_gain",
    ]:
        raise RuntimeError("Wave 57 arm inventory drifted")
    bootstrap = config.get("bootstrap", {})
    if bootstrap.get("replicates") != 5000 or bootstrap.get("seed") != 5707:
        raise RuntimeError("Wave 57 bootstrap drifted")
    minimums = config.get("minimums", {})
    expected_minimums = {
        "gate_fit_tokens": 100,
        "gate_fit_disagreement_rows": 400,
        "gate_fit_disagreement_tokens": 120,
        "gate_fit_harm_tokens": 80,
        "gate_fit_nonharm_tokens": 50,
        "gate_select_tokens": 80,
        "gate_select_disagreement_rows": 300,
        "gate_select_disagreement_tokens": 120,
        "gate_select_proposal_tokens": 40,
        "gate_select_authorized_tokens": 25,
        "gate_select_shard_tokens": 40,
        "gate_select_shard_disagreement_rows": 120,
        "gate_select_shard_disagreement_tokens": 50,
        "gate_select_shard_proposal_tokens": 20,
        "gate_select_shard_authorized_tokens": 12,
        "sealed_monitor_tokens": 100,
        "sealed_monitor_disagreement_rows": 300,
        "sealed_monitor_disagreement_tokens": 120,
        "shuffle_permutable_fraction": 0.8,
        "shuffle_hamming_global": 0.25,
        "shuffle_hamming_weighted": 0.25,
        "absent_support_tokens_per_set": 30,
    }
    if minimums != expected_minimums:
        raise RuntimeError("Wave 57 minimums drifted")


def harm_labels(gain: np.ndarray) -> np.ndarray:
    """Return one exactly when replacing hard by posterior causes negative gain."""
    values = np.asarray(gain, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("gain must be finite")
    return (values < -HARM_EPSILON).astype(np.int8)


def _array_digest(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    header = json.dumps(
        {"dtype": array.dtype.str, "shape": list(array.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(header + b"\0" + array.tobytes()).hexdigest()


def validate_harm_model_contract(contract: dict[str, Any]) -> None:
    if contract != HARM_MODEL_CONTRACT:
        raise ValueError("harm-model contract drifted")
    if sklearn.__version__ != EXPECTED_SKLEARN_VERSION:
        raise RuntimeError(
            f"scikit-learn version drifted: {sklearn.__version__} != "
            f"{EXPECTED_SKLEARN_VERSION}"
        )


def fit_harm_logistic(
    design: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fit the frozen positive-is-harm Logistic and return a JSON-safe state."""
    x = np.asarray(design, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int8)
    sample_weight = np.asarray(weights, dtype=np.float64)
    if x.ndim != 2 or y.shape != (len(x),) or sample_weight.shape != (len(x),):
        raise ValueError("harm fit arrays do not align")
    if len(x) == 0 or np.any(sample_weight <= 0.0):
        raise ValueError("harm fit requires rows with positive weights")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(sample_weight)):
        raise ValueError("harm fit arrays must be finite")
    if not np.array_equal(np.unique(y), np.asarray([0, 1], dtype=np.int8)):
        raise RuntimeError("harm logistic requires both classes 0 and 1")

    frozen = dict(HARM_MODEL_CONTRACT if contract is None else contract)
    validate_harm_model_contract(frozen)
    scaler = fit_weighted_scaler(x, sample_weight)
    xs = scaler.transform(x)
    model = LogisticRegression(**frozen)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(xs, y, sample_weight=sample_weight)
    if any(issubclass(item.category, ConvergenceWarning) for item in caught):
        raise RuntimeError("harm logistic did not converge")
    if not np.array_equal(model.classes_, np.asarray([0, 1])):
        raise RuntimeError("harm logistic class order is not [0,1]")

    probability = model.predict_proba(xs)[:, 1]
    reconstructed = expit(xs @ np.asarray(model.coef_[0]) + float(model.intercept_[0]))
    np.testing.assert_allclose(probability, reconstructed, rtol=0.0, atol=2e-15)
    numeric = (
        scaler.mean,
        scaler.scale,
        model.coef_,
        model.intercept_,
        probability,
    )
    if not all(np.all(np.isfinite(value)) for value in numeric):
        raise FloatingPointError("harm logistic produced a non-finite state")
    return {
        "kind": "logistic_harm",
        "sklearn_version": sklearn.__version__,
        "contract": frozen,
        "classes": model.classes_.astype(int).tolist(),
        "mean": scaler.mean.tolist(),
        "scale": scaler.scale.tolist(),
        "coef": np.asarray(model.coef_[0], dtype=np.float64).tolist(),
        "intercept": float(model.intercept_[0]),
        "n_iter": np.asarray(model.n_iter_, dtype=np.int64).tolist(),
        "fit_probability_sha256": _array_digest(probability),
    }


def score_harm_logistic(state: dict[str, Any], design: np.ndarray) -> np.ndarray:
    """Reconstruct positive-is-harm probabilities from the preserved state."""
    if state.get("kind") != "logistic_harm" or state.get("classes") != [0, 1]:
        raise ValueError("invalid harm-model state")
    validate_harm_model_contract(state.get("contract", {}))
    x = np.asarray(design, dtype=np.float64)
    mean = np.asarray(state["mean"], dtype=np.float64)
    scale = np.asarray(state["scale"], dtype=np.float64)
    coef = np.asarray(state["coef"], dtype=np.float64)
    intercept = float(state["intercept"])
    if x.ndim != 2 or x.shape[1:] != mean.shape or scale.shape != mean.shape:
        raise ValueError("harm score feature shape mismatch")
    probability = expit(((x - mean) / scale) @ coef + intercept)
    if not np.all(np.isfinite(probability)) or np.any((probability < 0) | (probability > 1)):
        raise FloatingPointError("invalid harm probability")
    return probability


def conditional_harm_shuffle(
    labels: np.ndarray,
    disagreement: np.ndarray,
    weights: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    """Permute binary harm labels within (policy, disagreement-count)."""
    y = np.asarray(labels, dtype=np.int8)
    active = np.asarray(disagreement, dtype=bool)
    w = np.asarray(weights, dtype=np.float64)
    if y.shape != active.shape or w.shape != active.shape or y.ndim != 2:
        raise ValueError("shuffle arrays must align as [token,policy]")
    if np.any((y != 0) & (y != 1)) or np.any(w[active] <= 0.0):
        raise ValueError("shuffle requires binary labels and positive active weights")

    rng = np.random.Generator(np.random.PCG64(int(seed)))
    counts = active.sum(axis=1)
    shuffled = y.copy()
    mapping = np.full(y.shape, -1, dtype=np.int64)
    permutable = np.zeros(y.shape, dtype=bool)
    moved = np.zeros(y.shape, dtype=bool)
    strata: list[dict[str, Any]] = []
    for policy in range(y.shape[1]):
        for count in sorted(np.unique(counts[active[:, policy]]).astype(int).tolist()):
            indices = np.flatnonzero(active[:, policy] & (counts == count))
            permutation = rng.permutation(indices)
            mapping[indices, policy] = permutation
            shuffled[indices, policy] = y[permutation, policy]
            if len(indices) > 1:
                permutable[indices, policy] = True
            moved[indices, policy] = permutation != indices
            changed = shuffled[indices, policy] != y[indices, policy]
            mixed = bool(np.unique(y[indices, policy]).size == 2)
            strata.append(
                {
                    "policy_index": policy,
                    "disagreement_count": count,
                    "rows": int(len(indices)),
                    "positive": int(y[indices, policy].sum()),
                    "mixed": mixed,
                    "changed": int(changed.sum()),
                    "hamming": float(changed.mean()) if len(indices) else 0.0,
                    "weight": float(w[indices, policy].sum()),
                    "changed_weight": float(w[indices, policy][changed].sum()),
                }
            )
    n_active = int(active.sum())
    changed = active & (shuffled != y)
    active_weight = float(w[active].sum())
    mixed = [row for row in strata if row["mixed"]]
    homogeneous = [row for row in strata if not row["mixed"]]
    diagnostics = {
        "seed": int(seed),
        "active_rows": n_active,
        "permutable_fraction": float(permutable.sum() / n_active) if n_active else None,
        "mapping_nonidentity_fraction": float(moved.sum() / n_active) if n_active else None,
        "hamming_global": float(changed.sum() / n_active) if n_active else None,
        "hamming_weighted": float(w[changed].sum() / active_weight) if active_weight else None,
        "mixed_strata": len(mixed),
        "homogeneous_strata": len(homogeneous),
        "mixed_weight": float(sum(float(row["weight"]) for row in mixed)),
        "homogeneous_weight": float(sum(float(row["weight"]) for row in homogeneous)),
        "mapping_sha256": _array_digest(mapping),
        "target_sha256": _array_digest(shuffled),
        "strata": strata,
    }
    return {"target": shuffled, "mapping": mapping, "diagnostics": diagnostics}


def proposal_mask(
    mean_score: np.ndarray,
    disagreement: np.ndarray,
    threshold: float | str,
) -> np.ndarray:
    scores = np.asarray(mean_score, dtype=np.float64)
    active = np.asarray(disagreement, dtype=bool)
    if scores.shape != active.shape:
        raise ValueError("proposal score/disagreement mismatch")
    if threshold == "hard_only":
        return np.zeros(active.shape, dtype=bool)
    value = float(threshold)
    finite = active & np.isfinite(scores)
    if np.any(active & ~finite) or not np.isfinite(value):
        raise ValueError("proposal requires finite active scores and threshold")
    return active & (scores > value)


def apply_harm_guard(
    proposals: np.ndarray,
    harm_probability: np.ndarray,
    hard_actions: np.ndarray,
    posterior_actions: np.ndarray,
    threshold: float | str,
) -> dict[str, np.ndarray]:
    proposed = np.asarray(proposals, dtype=bool)
    probability = np.asarray(harm_probability, dtype=np.float64)
    hard = np.asarray(hard_actions, dtype=np.int64)
    posterior = np.asarray(posterior_actions, dtype=np.int64)
    if probability.shape != proposed.shape or hard.shape != proposed.shape or posterior.shape != proposed.shape:
        raise ValueError("guard arrays must align")
    if threshold == "hard_only":
        authorized = np.zeros(proposed.shape, dtype=bool)
    else:
        value = float(threshold)
        if not np.isfinite(value) or np.any(proposed & ~np.isfinite(probability)):
            raise ValueError("guard requires finite proposed probabilities")
        authorized = proposed & (probability < value)
    actions = np.where(authorized, posterior, hard)
    return {"actions": actions, "authorized": authorized, "proposed": proposed.copy()}


def token_support(mask: np.ndarray, token_mask: np.ndarray) -> dict[str, int]:
    values = np.asarray(mask, dtype=bool) & np.asarray(token_mask, dtype=bool)[:, None]
    return {"rows": int(values.sum()), "tokens": int(values.any(axis=1).sum())}


def per_set_support(
    set_index: np.ndarray,
    token_mask: np.ndarray,
    requested: list[int] | tuple[int, ...],
    minimum: int,
) -> dict[str, dict[str, Any]]:
    """Adjudicate support independently; union support never authorizes a set."""
    indices = np.asarray(set_index, dtype=np.int64)
    active = np.asarray(token_mask, dtype=bool)
    if indices.shape != active.shape or indices.ndim != 1:
        raise ValueError("set support arrays must align by token")
    if int(minimum) <= 0 or len(set(int(value) for value in requested)) != len(requested):
        raise ValueError("set support contract is invalid")
    return {
        str(int(value)): {
            "set_index": int(value),
            "tokens": int((active & (indices == int(value))).sum()),
            "required": int(minimum),
            "status": (
                "EVALUABLE"
                if int((active & (indices == int(value))).sum()) >= int(minimum)
                else "NOT_EVALUABLE"
            ),
        }
        for value in requested
    }
