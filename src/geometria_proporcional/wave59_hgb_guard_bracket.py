"""CPU-only primitives for the prospective Wave 59 HGB guard bracket.

This module contains no filesystem or oracle orchestration.  It implements the
frozen statistical objects that phase workers call after their access boundary
has been established.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import re
from copy import deepcopy
from typing import Any, Iterable

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from .wave58_open_diagnostic import (
    HGB_CLASSIFIER_KWARGS,
    HGB_REGRESSOR_KWARGS,
    _export_hgb,
    array_sha256,
    derive_targets,
    fit_logistic_state,
    fit_mask,
    fit_ridge_state,
    paired_delta_ci,
    score_grid,
    score_linear_state,
    summarize_actions,
)


# Shared closed-world phase inputs.  The worker stages exactly these names and
# the recovery validator uses the same frozen source contract rather than
# trusting a journal to declare its own coverage.
PHASE_FILES = {
    "fit": frozenset(
        {
            "phase_request.json",
            "config.json",
            "source_bindings.json",
            "preparation_freeze.json",
            "bundle.npz",
            "utilities.npy",
        }
    ),
    "calibrate_scores": frozenset(
        {
            "phase_request.json",
            "config.json",
            "source_bindings.json",
            "preparation_freeze.json",
            "inference_bundle.npz",
            "model_states_manifest.json",
            "model_state_arrays.npz",
            "fit_freeze.json",
        }
    ),
    "validate": frozenset(
        {
            "phase_request.json",
            "config.json",
            "source_bindings.json",
            "preparation_freeze.json",
            "inference_bundle.npz",
            "truth_bundle.npz",
            "validation_scores.npz",
            "validation_policy_arrays.npz",
            "calibration_freeze.json",
            "utilities.npy",
        }
    ),
    "monitor_apply": frozenset(
        {
            "phase_request.json",
            "config.json",
            "source_bindings.json",
            "preparation_freeze.json",
            "inference_bundle.npz",
            "model_states_manifest.json",
            "model_state_arrays.npz",
            "fit_freeze.json",
            "calibration_freeze.json",
            "validation_freeze.json",
        }
    ),
    "monitor_evaluate": frozenset(
        {
            "phase_request.json",
            "config.json",
            "source_bindings.json",
            "preparation_freeze.json",
            "truth_bundle.npz",
            "monitor_policy_arrays.npz",
            "monitor_action_freeze.json",
            "utilities.npy",
        }
    ),
}


SCHEMA_VERSION = "wave59-fresh-hgb-guard-bracket-v1"
Q_PROPOSER = 0.8
Q_GUARDS = (0.7, 0.9)
Q_LEGACY_HARM = 0.4
TARGETS = ("harm", "posterior_incompatibility")
PROPOSERS = ("ridge", "hgb")
GUARDS = ("logistic", "hgb")
HARM_CONTROL_SEEDS = (59031, 59032, 59033, 59034, 59035)
INCOMPATIBILITY_CONTROL_SEEDS = (59041, 59042, 59043, 59044, 59045)
BOOTSTRAP_SEED = 5907
SHARD_SALT = "wave59-bracket-shard"
PLAN_SHA256 = "7e74f892bf27c4c51fa5f44e4e04b4564f7d63d986d8cc69c7316663bc5eabfb"
PLAN_AUDIT_SHA256 = "c7d1ed28554bb3b1174bfb787ae9469ee20392062f0c36deda7d0d10dcab0134"
FROZEN_STATUS = "FROZEN_PROSPECTIVE_PROTOCOL_PRE_KEY_DRAW"
LEGACY_CONFIG_SOURCE_RELATIVE = (
    "experiments/geometria_proporcional/configs/"
    "wave59_fresh_hgb_guard_bracket.json"
)
SUCCESSOR_CONFIG_SOURCE_RELATIVE = (
    "experiments/geometria_proporcional/configs/"
    "wave59_fresh_hgb_guard_bracket_replay_normalized.json"
)
CONFIG_SOURCE_RELATIVES = frozenset(
    {LEGACY_CONFIG_SOURCE_RELATIVE, SUCCESSOR_CONFIG_SOURCE_RELATIVE}
)
SUCCESSOR_PAIR_TOKEN_COUNT_BASIS = "eligible_unique_pair_tokens"
INFERENCE_SAFE_KEYS = (
    "pair_token",
    "primary",
    "disagreement",
    "design",
    "hard_actions",
    "posterior_actions",
)
FORBIDDEN_INFERENCE_KEYS = (
    "target",
    "gain",
    "harm",
    "posterior_incompatibility",
    "oracle",
    "regret",
    "metric",
    "utilities",
)


class PortableLinearEstimator:
    """Joblib-safe executable copy of the authoritative portable linear scorer."""

    _wave59_portable = True

    def __init__(self, state: dict[str, Any]):
        if state.get("kind") not in {"ridge", "logistic"}:
            raise ValueError("portable linear estimator requires ridge or logistic state")
        self.state = state

    def predict(self, design: np.ndarray) -> np.ndarray:
        return np.asarray(score_linear_state(self.state, design), dtype=np.float64)

    def predict_proba(self, design: np.ndarray) -> np.ndarray:
        if self.state["kind"] != "logistic":
            raise AttributeError("predict_proba is defined only for logistic state")
        positive = self.predict(design)
        return np.column_stack((1.0 - positive, positive))


def config_source_relative(config: dict[str, Any]) -> str:
    """Return the one exact Wave 59 config self-source declared by the config."""
    required = config.get("required_execution_sources")
    if not isinstance(required, list):
        raise RuntimeError("Wave 59 execution-source manifest is absent")
    candidates = [path for path in required if path in CONFIG_SOURCE_RELATIVES]
    if len(candidates) != 1:
        raise RuntimeError("Wave 59 config self-binding path drifted")
    return candidates[0]


def is_successor_config(config: dict[str, Any]) -> bool:
    return config_source_relative(config) == SUCCESSOR_CONFIG_SOURCE_RELATIVE


def validate_pre_draw_config(config: dict[str, Any]) -> None:
    """Reject drift in the result-affecting Wave 59 contract before escrow."""
    if config.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError("Wave 59 schema drifted")
    if config.get("device") != "cpu" or config.get("cpu_threads") != 4:
        raise RuntimeError("Wave 59 CPU contract drifted")
    if config.get("penalty") != 1.25:
        raise RuntimeError("Wave 59 utility penalty drifted")
    if config.get("plan", {}).get("sha256") != PLAN_SHA256:
        raise RuntimeError("Wave 59 plan binding drifted")
    if config.get("accepted_plan_audit", {}).get("sha256") != PLAN_AUDIT_SHA256:
        raise RuntimeError("Wave 59 audit binding drifted")
    if config.get("physical_splits") != {
        "train": "gate_fit",
        "val": "gate_select",
        "lockbox": "sealed_monitor",
    }:
        raise RuntimeError("Wave 59 physical split contract drifted")
    if config.get("phases") != [
        "prepare",
        "fit",
        "calibrate_scores",
        "validate",
        "monitor_apply",
        "monitor_evaluate",
    ]:
        raise RuntimeError("Wave 59 phase contract drifted")
    quantiles = config.get("quantiles", {})
    if (
        quantiles.get("method") != "higher"
        or quantiles.get("comparison") != "strict"
        or quantiles.get("proposer") != Q_PROPOSER
        or quantiles.get("guard_factorial") != list(Q_GUARDS)
        or quantiles.get("legacy_harm") != Q_LEGACY_HARM
    ):
        raise RuntimeError("Wave 59 quantile contract drifted")
    factorial = config.get("factorial", {})
    if (
        factorial.get("proposer") != list(PROPOSERS)
        or factorial.get("guard") != list(GUARDS)
        or factorial.get("target") != list(TARGETS)
        or factorial.get("guard_quantile") != list(Q_GUARDS)
        or factorial.get("cell_count") != len(factorial_policy_ids())
        or factorial.get("selector") != "sequential_only"
    ):
        raise RuntimeError("Wave 59 factorial contract drifted")
    controls = config.get("controls", {})
    if controls.get("harm_seeds") != list(HARM_CONTROL_SEEDS) or controls.get(
        "incompatibility_seeds"
    ) != list(INCOMPATIBILITY_CONTROL_SEEDS):
        raise RuntimeError("Wave 59 control seeds drifted")
    if config.get("main_policies") != {
        "mean": policy_id("hgb", "hgb", "posterior_incompatibility", 0.9),
        "tail": policy_id("hgb", "hgb", "harm", 0.7),
    }:
        raise RuntimeError("Wave 59 main policies drifted")
    expected_minimums = {
        "gate_fit_tokens": 100,
        "gate_fit_disagreement_rows": 400,
        "gate_fit_disagreement_tokens": 120,
        "gate_fit_harm_tokens": 80,
        "gate_fit_nonharm_tokens": 50,
        "gate_fit_incompatibility_rows": 20,
        "gate_fit_incompatibility_tokens": 8,
        "gate_select_tokens": 80,
        "gate_select_disagreement_rows": 300,
        "gate_select_disagreement_tokens": 120,
        "gate_select_proposal_tokens": 40,
        "gate_select_authorized_tokens_per_main": 25,
        "gate_select_shard_tokens": 40,
        "gate_select_shard_disagreement_rows": 120,
        "gate_select_shard_disagreement_tokens": 50,
        "gate_select_shard_proposal_tokens": 20,
        "gate_select_shard_authorized_tokens_per_main": 12,
        "sealed_monitor_tokens": 100,
        "sealed_monitor_disagreement_rows": 300,
        "sealed_monitor_disagreement_tokens": 120,
    }
    if config.get("minimums") != expected_minimums:
        raise RuntimeError("Wave 59 minimums drifted")
    bootstrap = config.get("bootstrap", {})
    if (
        bootstrap.get("replicates") != 5000
        or bootstrap.get("seed") != BOOTSTRAP_SEED
        or bootstrap.get("interval") != [2.5, 97.5]
        or bootstrap.get("unit") != "pair_token_in_T_primary"
    ):
        raise RuntimeError("Wave 59 bootstrap contract drifted")
    shards = config.get("shards", {})
    if shards.get("count") != 2 or shards.get("salt") != SHARD_SALT:
        raise RuntimeError("Wave 59 shard contract drifted")
    artifacts = config.get("artifact_classes", {})
    if (
        artifacts.get("schema") != "wave59-artifact-classes-v1"
        or artifacts.get("closed_world") is not True
        or set(artifacts.get("comparisons", {}))
        != {
            "scientific_exact",
            "scientific_array_exact",
            "functional_state",
            "operational_semantic",
            "secret_excluded_from_public_manifest",
            "self_reference",
        }
    ):
        raise RuntimeError("Wave 59 artifact-class contract drifted")
    runtime = config.get("runtime_budget", {})
    if runtime.get("gpu_allowed") is not False or runtime.get(
        "max_seconds_per_run"
    ) != 1800 or runtime.get("max_rss_bytes") != 8 * 1024**3:
        raise RuntimeError("Wave 59 runtime contract drifted")
    if config.get("status") == FROZEN_STATUS:
        binding = config.get("implementation_binding", {})
        if (
            binding.get("status") != "ACCEPTED_IMPLEMENTATION_AUDIT"
            or re.fullmatch(r"[0-9a-f]{40}", str(binding.get("commit", ""))) is None
            or not isinstance(binding.get("audit_path"), str)
            or re.fullmatch(r"[0-9a-f]{64}", str(binding.get("audit_sha256", "")))
            is None
        ):
            raise RuntimeError("Wave 59 accepted implementation audit is not bound")
        required = config.get("required_execution_sources")
        hashes = config.get("source_sha256")
        if (
            not isinstance(required, list)
            or not required
            or len(required) != len(set(required))
            or not isinstance(hashes, dict)
            or set(hashes) != set(required)
            or any(re.fullmatch(r"[0-9a-f]{64}", str(value)) is None for value in hashes.values())
        ):
            raise RuntimeError("Wave 59 execution-source hashes are incomplete")
        if binding["audit_path"] not in required:
            raise RuntimeError("Wave 59 implementation audit is not an execution source")
        config_path = config_source_relative(config)
        fresh = config.get("fresh_benchmark", {})
        if config_path == SUCCESSOR_CONFIG_SOURCE_RELATIVE:
            if (
                fresh.get("pair_token_count_basis")
                != SUCCESSOR_PAIR_TOKEN_COUNT_BASIS
                or not isinstance(config.get("successor_authority"), dict)
            ):
                raise RuntimeError("Wave 59 successor identity contract drifted")
        elif (
            "pair_token_count_basis" in fresh
            or "successor_authority" in config
        ):
            raise RuntimeError("Wave 59 legacy config acquired successor semantics")
        if hashes[config_path] != config_self_binding_sha256(config, config_path):
            raise RuntimeError("Wave 59 config self-binding drifted")
        if importlib.metadata.version("scikit-learn") != config.get("models", {}).get(
            "sklearn_version"
        ):
            raise RuntimeError("Wave 59 scikit-learn version drifted")


def canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def config_self_binding_sha256(config: dict[str, Any], relative_path: str) -> str:
    """Hash a config with only its own recursive digest normalized to zeros."""
    normalized = deepcopy(config)
    hashes = normalized.get("source_sha256")
    if not isinstance(hashes, dict) or relative_path not in hashes:
        raise RuntimeError("Wave 59 config self-binding entry is absent")
    hashes[relative_path] = "0" * 64
    return canonical_json_sha256(normalized)


def inference_safe_view(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    missing = [key for key in INFERENCE_SAFE_KEYS if key not in data]
    if missing:
        raise ValueError(f"inference-safe source missing {missing}")
    view = {key: np.asarray(data[key]).copy() for key in INFERENCE_SAFE_KEYS}
    validate_inference_safe_view(view)
    return view


def validate_inference_safe_view(data: dict[str, np.ndarray]) -> None:
    keys = set(data)
    if keys != set(INFERENCE_SAFE_KEYS):
        raise ValueError("inference-safe bundle key allowlist mismatch")
    lowered = [key.lower() for key in keys]
    if any(fragment in key for fragment in FORBIDDEN_INFERENCE_KEYS for key in lowered):
        raise ValueError("inference-safe bundle contains a forbidden semantic field")
    primary = np.asarray(data["primary"], dtype=bool)
    disagreement = np.asarray(data["disagreement"], dtype=bool)
    design = np.asarray(data["design"], dtype=np.float64)
    if disagreement.shape != np.asarray(data["hard_actions"]).shape or disagreement.shape != np.asarray(
        data["posterior_actions"]
    ).shape:
        raise ValueError("inference-safe action arrays do not align")
    if design.shape[:2] != disagreement.shape or primary.shape != (len(design),):
        raise ValueError("inference-safe design arrays do not align")
    if not np.all(np.isfinite(design)):
        raise ValueError("inference-safe design must be finite")


def quantile_higher(values: np.ndarray, q: float) -> float:
    raw = np.asarray(values, dtype=np.float64)
    if raw.ndim != 1 or not len(raw) or not np.all(np.isfinite(raw)):
        raise ValueError("quantile input must be a non-empty finite vector")
    if not 0.0 <= float(q) <= 1.0:
        raise ValueError("quantile must be in [0,1]")
    return float(np.quantile(raw, float(q), method="higher"))


def primary_tokens(data: dict[str, np.ndarray]) -> np.ndarray:
    primary = np.asarray(data["primary"], dtype=bool)
    tokens = np.asarray(data["pair_token"]).astype(str)
    if primary.shape != (len(tokens),):
        raise ValueError("primary and pair_token do not align")
    selected = tokens[primary]
    if len(np.unique(tokens)) != len(tokens):
        raise ValueError("pair_token must be unique before policy expansion")
    return np.sort(selected)


def validate_primary_integrity(data: dict[str, np.ndarray], policies: int = 24) -> None:
    primary = np.asarray(data["primary"], dtype=bool)
    disagreement = np.asarray(data["disagreement"], dtype=bool)
    if disagreement.ndim != 2 or disagreement.shape[0] != len(primary):
        raise ValueError("primary/disagreement shape mismatch")
    if disagreement.shape[1] != int(policies):
        raise ValueError("each primary token must carry exactly 24 policies")
    for key in ("hard_actions", "posterior_actions", "gain", "weights"):
        if np.asarray(data[key]).shape != disagreement.shape:
            raise ValueError(f"{key} does not align with policy rows")
    if not len(primary_tokens(data)):
        raise ValueError("T_primary is empty")


def support(mask: np.ndarray, primary: np.ndarray) -> dict[str, int]:
    selected = np.asarray(mask, dtype=bool)
    population = np.asarray(primary, dtype=bool)
    if selected.ndim != 2 or selected.shape[0] != len(population):
        raise ValueError("support arrays do not align")
    active = selected & population[:, None]
    return {
        "authorized_rows": int(active.sum()),
        "authorized_pair_tokens": int(active.any(axis=1).sum()),
    }


def _active_entry_digest(values: np.ndarray) -> str:
    return array_sha256(np.asarray(values))


def maximum_displacement_control(
    labels: np.ndarray,
    active: np.ndarray,
    weights: np.ndarray,
    pair_tokens: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    """Maximize conditional Hamming distance while preserving stratum prevalence.

    Mapping values are source token indices for each destination token/policy;
    inactive entries are -1.  Strata are (policy_index, disagreement_count).
    """
    y = np.asarray(labels, dtype=np.int8)
    mask = np.asarray(active, dtype=bool)
    w = np.asarray(weights, dtype=np.float64)
    tokens = np.asarray(pair_tokens).astype(str)
    if y.ndim != 2 or mask.shape != y.shape or w.shape != y.shape:
        raise ValueError("control arrays must align as [token,policy]")
    if tokens.shape != (len(y),) or len(np.unique(tokens)) != len(tokens):
        raise ValueError("control pair_token identity is invalid")
    if np.any((y != 0) & (y != 1)) or np.any(w[mask] <= 0.0):
        raise ValueError("control needs binary labels and positive active weights")
    if not np.all(np.isfinite(w[mask])):
        raise ValueError("control weights must be finite")

    rng = np.random.Generator(np.random.PCG64(int(seed)))
    counts = mask.sum(axis=1)
    mapping = np.full(y.shape, -1, dtype=np.int64)
    target = y.copy()
    permutable = np.zeros(y.shape, dtype=bool)
    swappable = np.zeros(y.shape, dtype=bool)
    changed = np.zeros(y.shape, dtype=bool)
    maximum = np.zeros(y.shape, dtype=bool)
    strata: list[dict[str, Any]] = []

    for policy in range(y.shape[1]):
        observed_counts = sorted(np.unique(counts[mask[:, policy]]).astype(int).tolist())
        for disagreement_count in observed_counts:
            indices = np.flatnonzero(mask[:, policy] & (counts == disagreement_count))
            indices = np.asarray(
                sorted(indices.tolist(), key=lambda index: (tokens[index], int(index))),
                dtype=np.int64,
            )
            mapping[indices, policy] = indices
            if len(indices) >= 2:
                permutable[indices, policy] = True
            zeros = indices[y[indices, policy] == 0]
            ones = indices[y[indices, policy] == 1]
            mixed = bool(len(zeros) and len(ones))
            if mixed:
                swappable[indices, policy] = True
            n_pairs = min(len(zeros), len(ones))
            if n_pairs:
                if len(ones) <= len(zeros):
                    minority, majority = ones, zeros
                else:
                    minority, majority = zeros, ones
                # Weights are constant in the declared strata.  Sorting first
                # keeps the tie permutation reproducible across array layouts.
                majority = np.asarray(
                    sorted(majority.tolist(), key=lambda index: (tokens[index], int(index))),
                    dtype=np.int64,
                )
                receiver_order = rng.permutation(len(majority))
                receivers = majority[receiver_order[:n_pairs]]
                source_order = rng.permutation(len(minority))
                sources = minority[source_order]
                for source, receiver in zip(sources.tolist(), receivers.tolist(), strict=True):
                    mapping[source, policy] = receiver
                    mapping[receiver, policy] = source
                target[indices, policy] = y[mapping[indices, policy], policy]
                changed[indices, policy] = target[indices, policy] != y[indices, policy]
                maximum[np.concatenate((minority, receivers)), policy] = True
            strata.append(
                {
                    "policy_index": int(policy),
                    "disagreement_count": int(disagreement_count),
                    "n": int(len(indices)),
                    "n0": int(len(zeros)),
                    "n1": int(len(ones)),
                    "max_changed": int(2 * n_pairs),
                    "attained_changed": int(changed[indices, policy].sum()),
                    "mixed": mixed,
                }
            )

    active_count = int(mask.sum())
    weight_total = float(w[mask].sum())
    max_weight = float(w[maximum].sum())
    attained_weight = float(w[changed & mask].sum())
    positives = mask & (y == 1)
    diagnostics = {
        "seed": int(seed),
        "active_entries": active_count,
        "positive_entries": int(positives.sum()),
        "mapping_permutable_fraction": float(permutable[mask].mean()) if active_count else None,
        "label_swappable_fraction": float(swappable[mask].mean()) if active_count else None,
        "hamming_global_max": float(maximum[mask].mean()) if active_count else None,
        "hamming_global_attained": float(changed[mask].mean()) if active_count else None,
        "hamming_weighted_max": max_weight / weight_total if weight_total else None,
        "hamming_weighted_attained": attained_weight / weight_total if weight_total else None,
        "attained_over_max": attained_weight / max_weight if max_weight else None,
        "positive_displacement_fraction": (
            float(changed[positives].mean()) if np.any(positives) else None
        ),
        "mapping_sha256": _active_entry_digest(mapping),
        "target_sha256": _active_entry_digest(target),
        "strata": strata,
    }
    if max_weight and not np.isclose(
        diagnostics["attained_over_max"], 1.0, rtol=0.0, atol=1e-12
    ):
        raise AssertionError("maximum displacement was not attained")
    return {"mapping": mapping, "target": target, "diagnostics": diagnostics}


def control_family_evaluable(
    controls: Iterable[dict[str, Any]], target_name: str
) -> tuple[bool, list[str]]:
    rows = list(controls)
    failures: list[str] = []
    expected = 0.25 if target_name == "harm" else 0.02
    if len(rows) != 5:
        failures.append("control_count")
    mapping_hashes = {row["diagnostics"]["mapping_sha256"] for row in rows}
    target_hashes = {row["diagnostics"]["target_sha256"] for row in rows}
    if len(mapping_hashes) != 5:
        failures.append("mapping_identity")
    if len(target_hashes) != 5:
        failures.append("target_identity")
    for row in rows:
        diag = row["diagnostics"]
        if float(diag.get("mapping_permutable_fraction") or 0.0) < 0.80:
            failures.append("mapping_permutable_fraction")
        if float(diag.get("hamming_weighted_max") or 0.0) < expected:
            failures.append("hamming_weighted_max")
        if not np.isclose(
            float(diag.get("attained_over_max") or 0.0), 1.0, rtol=0.0, atol=1e-12
        ):
            failures.append("attained_over_max")
    return not failures, sorted(set(failures))


def model_id(kind: str, target: str | None = None, seed: int | None = None) -> str:
    target_slug = "incompatibility" if target == "posterior_incompatibility" else target
    if kind in PROPOSERS and target is None and seed is None:
        return f"proposer-{kind}"
    if kind in GUARDS and target in TARGETS and seed is None:
        return f"guard-{kind}-{target_slug}"
    if kind == "control-hgb" and target in TARGETS and seed is not None:
        return f"control-hgb-{target_slug}-{int(seed)}"
    raise ValueError("invalid Wave 59 model identity")


def policy_id(proposer: str, guard: str, target: str, q_guard: float) -> str:
    if proposer not in PROPOSERS or guard not in GUARDS or target not in TARGETS:
        raise ValueError("invalid Wave 59 policy identity")
    if float(q_guard) not in Q_GUARDS:
        raise ValueError("invalid Wave 59 guard quantile")
    q = "70" if float(q_guard) == 0.7 else "90"
    target_slug = "INCOMPATIBILITY" if target == "posterior_incompatibility" else target.upper()
    return f"P-{proposer.upper()}-{guard.upper()}-{target_slug}-Q{q}"


def factorial_policy_ids() -> list[str]:
    return [
        policy_id(proposer, guard, target, q)
        for proposer in PROPOSERS
        for guard in GUARDS
        for target in TARGETS
        for q in Q_GUARDS
    ]


def fit_true_models(
    data: dict[str, np.ndarray],
    utilities: np.ndarray,
    penalty: float,
    *,
    return_objects: bool = False,
) -> tuple[dict[str, dict[str, Any]], dict[str, np.ndarray], dict[str, np.ndarray]] | tuple[
    dict[str, dict[str, Any]], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]
]:
    validate_primary_integrity(data)
    targets = derive_targets(data, utilities, penalty)
    rows = fit_mask(data)
    x = np.asarray(data["design"], dtype=np.float64)[rows]
    weights = np.asarray(data["weights"], dtype=np.float64)[rows]
    gain = np.asarray(targets["gain"], dtype=np.float64)[rows]
    states: dict[str, dict[str, Any]] = {}
    arrays: dict[str, np.ndarray] = {}
    objects: dict[str, Any] = {}

    ridge_id = model_id("ridge")
    states[ridge_id], objects[ridge_id] = fit_ridge_state(x, gain, weights)
    hgb_id = model_id("hgb")
    states[hgb_id], exported, objects[hgb_id] = _fit_hgb_object(
        hgb_id, x, gain, weights, classifier=False, seed=5801
    )
    arrays.update(exported)
    for target_name, seed in (("harm", 5802), ("posterior_incompatibility", 5803)):
        y = np.asarray(targets[target_name], dtype=bool)[rows]
        logistic_id = model_id("logistic", target_name)
        states[logistic_id], objects[logistic_id] = fit_logistic_state(x, y, weights)
        hgb_guard_id = model_id("hgb", target_name)
        states[hgb_guard_id], exported, objects[hgb_guard_id] = _fit_hgb_object(
            hgb_guard_id, x, y, weights, classifier=True, seed=seed
        )
        arrays.update(exported)
    if return_objects:
        return states, arrays, targets, objects
    return states, arrays, targets


def _fit_hgb_object(
    name: str,
    design: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    *,
    classifier: bool,
    seed: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray], Any | None]:
    x = np.asarray(design, dtype=np.float64)
    y = np.asarray(target)
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
        }, {}, None
    if classifier and not np.array_equal(np.unique(y), np.asarray([False, True])):
        return {
            "kind": "hgb_classifier",
            "status": "NOT_EVALUABLE",
            "reason": "single_class",
            "classes_observed": np.unique(y).astype(int).tolist(),
        }, {}, None
    if classifier:
        model = HistGradientBoostingClassifier(
            **HGB_CLASSIFIER_KWARGS, random_state=int(seed)
        )
    else:
        if int(seed) != 5801:
            raise ValueError("HGB regressor seed drifted")
        model = HistGradientBoostingRegressor(**HGB_REGRESSOR_KWARGS)
    model.fit(x, y, sample_weight=sample_weight)
    state, arrays = _export_hgb(name, model)
    direct = model.predict_proba(x)[:, 1] if classifier else model.predict(x)
    np.testing.assert_allclose(
        score_grid(
            state,
            arrays,
            {
                "disagreement": np.ones((len(x), 1), dtype=bool),
                "design": x[:, None, :],
            },
        )[:, 0],
        direct,
        rtol=0.0,
        atol=2e-15,
        equal_nan=True,
    )
    return state, arrays, model


def fit_control_models(
    data: dict[str, np.ndarray],
    targets: dict[str, np.ndarray],
    *,
    return_objects: bool = False,
) -> tuple[dict[str, dict[str, Any]], dict[str, np.ndarray], dict[str, Any]] | tuple[
    dict[str, dict[str, Any]], dict[str, np.ndarray], dict[str, Any], dict[str, Any]
]:
    rows = fit_mask(data)
    x = np.asarray(data["design"], dtype=np.float64)[rows]
    weights = np.asarray(data["weights"], dtype=np.float64)[rows]
    tokens = np.asarray(data["pair_token"]).astype(str)
    states: dict[str, dict[str, Any]] = {}
    arrays: dict[str, np.ndarray] = {}
    metadata: dict[str, Any] = {"families": {}}
    objects: dict[str, Any] = {}
    active = np.asarray(data["disagreement"], dtype=bool) & np.asarray(
        data["primary"], dtype=bool
    )[:, None]
    for target_name, seeds, fit_seed in (
        ("harm", HARM_CONTROL_SEEDS, 5802),
        ("posterior_incompatibility", INCOMPATIBILITY_CONTROL_SEEDS, 5803),
    ):
        labels = np.asarray(targets[target_name], dtype=np.int8)
        controls = [
            maximum_displacement_control(
                labels, active, data["weights"], tokens, mapping_seed
            )
            for mapping_seed in seeds
        ]
        evaluable, failures = control_family_evaluable(controls, target_name)
        family_rows: list[dict[str, Any]] = []
        for mapping_seed, control in zip(seeds, controls, strict=True):
            identifier = model_id("control-hgb", target_name, mapping_seed)
            state, exported, fitted = _fit_hgb_object(
                identifier,
                x,
                np.asarray(control["target"], dtype=bool)[rows],
                weights,
                classifier=True,
                seed=fit_seed,
            )
            if state.get("status") != "PASS":
                evaluable = False
                failures.append(f"fit:{identifier}")
            states[identifier] = state
            objects[identifier] = fitted
            arrays.update(exported)
            arrays[f"mapping__{identifier}"] = np.asarray(control["mapping"])
            arrays[f"target__{identifier}"] = np.asarray(control["target"])
            family_rows.append(
                {
                    "model_id": identifier,
                    "mapping_seed": int(mapping_seed),
                    "fit_random_state": int(fit_seed),
                    "diagnostics": control["diagnostics"],
                }
            )
        metadata["families"][target_name] = {
            "status": "PASS" if evaluable else "NOT_EVALUABLE",
            "failures": sorted(set(failures)),
            "controls": family_rows,
        }
    if return_objects:
        return states, arrays, metadata, objects
    return states, arrays, metadata


def score_true_models(
    states: dict[str, dict[str, Any]],
    arrays: dict[str, np.ndarray],
    data: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    return {name: score_grid(state, arrays, data) for name, state in states.items()}


def calibrate_policies(
    data: dict[str, np.ndarray], scores: dict[str, np.ndarray]
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    primary = np.asarray(data["primary"], dtype=bool)
    disagreement = np.asarray(data["disagreement"], dtype=bool)
    active = primary[:, None] & disagreement
    hard = np.asarray(data["hard_actions"], dtype=np.int64)
    posterior = np.asarray(data["posterior_actions"], dtype=np.int64)
    if hard.shape != active.shape or posterior.shape != active.shape:
        raise ValueError("calibration action arrays do not align")
    arrays: dict[str, np.ndarray] = {}
    metadata: dict[str, Any] = {"proposers": {}, "policies": {}, "controls": {}}

    proposals: dict[str, np.ndarray] = {}
    for proposer in PROPOSERS:
        proposer_name = model_id(proposer)
        tau = quantile_higher(np.asarray(scores[proposer_name])[active], Q_PROPOSER)
        proposal = active & (np.asarray(scores[proposer_name]) > tau)
        proposals[proposer] = proposal
        arrays[f"proposal__{proposer}"] = proposal
        metadata["proposers"][proposer] = {
            "model_id": proposer_name,
            "q": Q_PROPOSER,
            "threshold": tau,
            "support": support(proposal, primary),
        }

    for proposer in PROPOSERS:
        proposal = proposals[proposer]
        for guard in GUARDS:
            for target_name in TARGETS:
                guard_name = model_id(guard, target_name)
                guard_scores = np.asarray(scores[guard_name], dtype=np.float64)
                for q_guard in Q_GUARDS:
                    identifier = policy_id(proposer, guard, target_name, q_guard)
                    tau = quantile_higher(guard_scores[proposal], q_guard)
                    authorized = proposal & (guard_scores < tau)
                    actions = np.where(authorized, posterior, hard)
                    arrays[f"authorized__{identifier}"] = authorized
                    arrays[f"actions__{identifier}"] = actions.astype(np.int64)
                    metadata["policies"][identifier] = {
                        "proposer_model_id": model_id(proposer),
                        "guard_model_id": guard_name,
                        "target": target_name,
                        "q_guard": float(q_guard),
                        "guard_threshold": tau,
                        "support": support(authorized, primary),
                    }
    legacy_scores = np.asarray(scores[model_id("logistic", "harm")], dtype=np.float64)
    legacy_tau = quantile_higher(legacy_scores[proposals["ridge"]], Q_LEGACY_HARM)
    legacy_authorized = proposals["ridge"] & (legacy_scores < legacy_tau)
    arrays["authorized__LEGACY-W57"] = legacy_authorized
    arrays["actions__LEGACY-W57"] = np.where(legacy_authorized, posterior, hard).astype(
        np.int64
    )
    metadata["legacy"] = {
        "proposer_model_id": model_id("ridge"),
        "guard_model_id": model_id("logistic", "harm"),
        "q_guard": Q_LEGACY_HARM,
        "guard_threshold": legacy_tau,
        "support": support(legacy_authorized, primary),
    }

    for target_name, seeds, q_guard in (
        ("harm", HARM_CONTROL_SEEDS, 0.7),
        ("posterior_incompatibility", INCOMPATIBILITY_CONTROL_SEEDS, 0.9),
    ):
        for seed in seeds:
            identifier = model_id("control-hgb", target_name, seed)
            if identifier not in scores:
                continue
            guard_scores = np.asarray(scores[identifier], dtype=np.float64)
            tau = quantile_higher(guard_scores[proposals["hgb"]], q_guard)
            authorized = proposals["hgb"] & (guard_scores < tau)
            arrays[f"authorized__{identifier}"] = authorized
            arrays[f"actions__{identifier}"] = np.where(
                authorized, posterior, hard
            ).astype(np.int64)
            metadata["controls"][identifier] = {
                "target": target_name,
                "mapping_seed": int(seed),
                "q_guard": q_guard,
                "guard_threshold": tau,
                "support": support(authorized, primary),
            }
    arrays["actions__HARD-SET"] = hard.copy()
    arrays["actions__PURE-POSTERIOR"] = posterior.copy()
    for proposer in PROPOSERS:
        arrays[f"actions__{proposer.upper()}-PROPOSER-ONLY"] = np.where(
            proposals[proposer], posterior, hard
        ).astype(np.int64)
    return metadata, arrays


def apply_calibrated_policies(
    data: dict[str, np.ndarray],
    scores: dict[str, np.ndarray],
    calibration: dict[str, Any],
) -> dict[str, np.ndarray]:
    primary = np.asarray(data["primary"], dtype=bool)
    disagreement = np.asarray(data["disagreement"], dtype=bool)
    active = primary[:, None] & disagreement
    hard = np.asarray(data["hard_actions"], dtype=np.int64)
    posterior = np.asarray(data["posterior_actions"], dtype=np.int64)
    arrays: dict[str, np.ndarray] = {
        "actions__HARD-SET": hard.copy(),
        "actions__PURE-POSTERIOR": posterior.copy(),
    }
    proposals: dict[str, np.ndarray] = {}
    for proposer, row in calibration["proposers"].items():
        proposal = active & (
            np.asarray(scores[row["model_id"]], dtype=np.float64)
            > float(row["threshold"])
        )
        proposals[proposer] = proposal
        arrays[f"proposal__{proposer}"] = proposal
        arrays[f"actions__{proposer.upper()}-PROPOSER-ONLY"] = np.where(
            proposal, posterior, hard
        ).astype(np.int64)
    for identifier, row in calibration["policies"].items():
        authorized = proposals[row["proposer_model_id"].removeprefix("proposer-")] & (
            np.asarray(scores[row["guard_model_id"]], dtype=np.float64)
            < float(row["guard_threshold"])
        )
        arrays[f"authorized__{identifier}"] = authorized
        arrays[f"actions__{identifier}"] = np.where(authorized, posterior, hard).astype(
            np.int64
        )
    legacy = calibration["legacy"]
    legacy_authorized = proposals["ridge"] & (
        np.asarray(scores[legacy["guard_model_id"]], dtype=np.float64)
        < float(legacy["guard_threshold"])
    )
    arrays["authorized__LEGACY-W57"] = legacy_authorized
    arrays["actions__LEGACY-W57"] = np.where(
        legacy_authorized, posterior, hard
    ).astype(np.int64)
    for identifier, row in calibration.get("controls", {}).items():
        authorized = proposals["hgb"] & (
            np.asarray(scores[identifier], dtype=np.float64)
            < float(row["guard_threshold"])
        )
        arrays[f"authorized__{identifier}"] = authorized
        arrays[f"actions__{identifier}"] = np.where(authorized, posterior, hard).astype(
            np.int64
        )
    return arrays


def shard_assignment(pair_tokens: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            hashlib.sha256((str(token) + SHARD_SALT).encode("utf-8")).digest()[-1] & 1
            for token in np.asarray(pair_tokens).astype(str)
        ],
        dtype=np.int8,
    )


def bootstrap_indices(pair_tokens: np.ndarray, replicates: int = 5000) -> np.ndarray:
    tokens = np.sort(np.asarray(pair_tokens).astype(str))
    if not len(tokens) or len(np.unique(tokens)) != len(tokens):
        raise ValueError("bootstrap needs unique non-empty T_primary")
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    return rng.integers(0, len(tokens), size=(int(replicates), len(tokens)), dtype=np.int64)


def evaluate_actions(
    data: dict[str, np.ndarray],
    utilities: np.ndarray,
    penalty: float,
    policy_arrays: dict[str, np.ndarray],
) -> tuple[dict[str, dict[str, float]], dict[str, np.ndarray]]:
    summaries: dict[str, dict[str, float]] = {}
    arrays: dict[str, np.ndarray] = {}
    for key, actions in sorted(policy_arrays.items()):
        if not key.startswith("actions__"):
            continue
        identifier = key.removeprefix("actions__")
        summary, metrics = summarize_actions(actions, data, utilities, penalty)
        summaries[identifier] = summary
        for metric, values in metrics.items():
            arrays[f"metric__{identifier}__{metric}"] = np.asarray(values)
    return summaries, arrays


def ordered_primary_metric(
    data: dict[str, np.ndarray], values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    tokens = np.asarray(data["pair_token"]).astype(str)
    primary = np.asarray(data["primary"], dtype=bool)
    metric = np.asarray(values, dtype=np.float64)
    if metric.shape != (len(tokens),):
        raise ValueError("metric must have one value per pair_token")
    indices = np.flatnonzero(primary)
    order = np.argsort(tokens[indices], kind="stable")
    return tokens[indices][order], metric[indices][order]


def mean_control_metric(
    metric_arrays: dict[str, np.ndarray], model_ids: Iterable[str], metric: str
) -> np.ndarray:
    values = [
        np.asarray(metric_arrays[f"metric__{identifier}__{metric}"], dtype=np.float64)
        for identifier in model_ids
    ]
    if len(values) != 5 or any(value.shape != values[0].shape for value in values):
        raise ValueError("control mean requires five aligned token metrics")
    return np.stack(values, axis=0).mean(axis=0)


def aggregate_ternary(conditions: Iterable[bool | str]) -> bool | str:
    values = list(conditions)
    if any(value == "NOT_EVALUABLE" for value in values):
        return "NOT_EVALUABLE"
    if not all(isinstance(value, (bool, np.bool_)) for value in values):
        raise ValueError("invalid ternary condition")
    return bool(all(bool(value) for value in values))


def delta_summary(
    left: np.ndarray, right: np.ndarray, indices: np.ndarray
) -> dict[str, float]:
    return paired_delta_ci(np.asarray(left), np.asarray(right), np.asarray(indices))
