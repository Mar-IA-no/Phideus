#!/usr/bin/env python3
"""Isolated CPU worker for Wave 59 analytical phases.

The coordinator owns oracle materialization and stages only the exact files
listed in ``phase_request.json``.  This worker never discovers repository or
benchmark paths.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import joblib
import numpy as np
from threadpoolctl import threadpool_info

from geometria_proporcional.wave59_hgb_guard_bracket import (
    HARM_CONTROL_SEEDS,
    INCOMPATIBILITY_CONTROL_SEEDS,
    PHASE_FILES,
    PortableLinearEstimator,
    aggregate_ternary,
    apply_calibrated_policies,
    bootstrap_indices,
    calibrate_policies,
    derive_targets,
    evaluate_actions,
    fit_control_models,
    fit_true_models,
    inference_safe_view,
    mean_control_metric,
    model_id,
    ordered_primary_metric,
    primary_tokens,
    policy_id,
    score_true_models,
    shard_assignment,
    support,
    validate_inference_safe_view,
    validate_pre_draw_config,
    validate_primary_integrity,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def write_json(path: Path, payload: Any) -> None:
    encoded = json.dumps(
        payload, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False
    ) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw = tempfile.mkstemp(dir=path.parent, suffix=".npz")
    os.close(descriptor)
    temporary = Path(raw)
    try:
        np.savez(temporary, **{key: np.asarray(value) for key, value in sorted(arrays.items())})
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def validate_stage(stage: Path, phase: str) -> dict[str, Any]:
    if phase not in PHASE_FILES:
        raise ValueError("unknown Wave 59 phase")
    actual = {path.name for path in stage.iterdir() if path.is_file()}
    if actual != PHASE_FILES[phase]:
        raise RuntimeError(f"stage allowlist mismatch: {sorted(actual)}")
    if any(path.is_symlink() for path in stage.iterdir()):
        raise RuntimeError("stage may not contain symlinks")
    request = load_json(stage / "phase_request.json")
    if request.get("phase") != phase or set(request.get("allowed_files", [])) != actual:
        raise RuntimeError("phase request does not bind the stage inventory")
    hashes = request.get("sha256", {})
    for name in sorted(actual - {"phase_request.json"}):
        if hashes.get(name) != sha256_file(stage / name):
            raise RuntimeError(f"stage hash mismatch: {name}")
    return request


def validate_freeze(
    freeze_path: Path,
    *,
    expected_phase: str,
    bindings: dict[str, Path],
    require_all_frozen_files: bool = False,
) -> dict[str, Any]:
    """Authenticate frozen inputs instead of trusting a newly staged request."""
    freeze = load_json(freeze_path)
    if freeze.get("schema_version") != "wave59-phase-freeze-v1":
        raise RuntimeError(f"{freeze_path.name} schema drifted")
    if freeze.get("phase") != expected_phase:
        raise RuntimeError(f"{freeze_path.name} phase drifted")
    frozen = freeze.get("files")
    if not isinstance(frozen, dict) or not frozen:
        raise RuntimeError(f"{freeze_path.name} has no frozen files")
    if require_all_frozen_files and set(bindings) != set(frozen):
        raise RuntimeError(f"{freeze_path.name} binding coverage drifted")
    if not set(bindings).issubset(frozen):
        raise RuntimeError(f"{freeze_path.name} lacks required bindings")
    for frozen_name, path in sorted(bindings.items()):
        if frozen.get(frozen_name) != sha256_file(path):
            raise RuntimeError(f"{freeze_path.name} hash mismatch: {frozen_name}")
    return freeze


def _validate_provenance_chain(stage: Path, freeze: dict[str, Any]) -> None:
    expected = {
        "config_sha256": sha256_file(stage / "config.json"),
        "source_bindings_sha256": sha256_file(stage / "source_bindings.json"),
        "preparation_freeze_sha256": sha256_file(stage / "preparation_freeze.json"),
    }
    for field, digest in expected.items():
        if freeze.get(field) != digest:
            raise RuntimeError(f"freeze provenance mismatch: {field}")


def _jsonable_states(states: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {"models": states, "model_ids": sorted(states)}


def _linear_arrays(states: dict[str, dict[str, Any]]) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for identifier, state in states.items():
        if state.get("kind") not in {"ridge", "logistic"} or state.get("status", "PASS") != "PASS":
            continue
        for key in ("mean", "scale", "coef", "intercept"):
            arrays[f"linear__{identifier}__{key}"] = np.asarray(state[key], dtype=np.float64)
    return arrays


def _freeze(output: Path, name: str, phase: str, files: list[str], extra: dict[str, Any]) -> None:
    write_json(
        output / name,
        {
            "schema_version": "wave59-phase-freeze-v1",
            "phase": phase,
            "files": {path: sha256_file(output / path) for path in files},
            **extra,
        },
    )


def _population_counts(data: dict[str, np.ndarray]) -> dict[str, int]:
    primary = np.asarray(data["primary"], dtype=bool)
    disagreement = np.asarray(data["disagreement"], dtype=bool)
    active = primary[:, None] & disagreement
    return {
        "primary_tokens": int(primary.sum()),
        "disagreement_rows": int(active.sum()),
        "disagreement_tokens": int(active.any(axis=1).sum()),
    }


def _require_minimums(
    observed: dict[str, int], required: dict[str, int], phase: str
) -> dict[str, Any]:
    failed = [name for name, minimum in required.items() if observed.get(name, -1) < minimum]
    result = {
        "phase": phase,
        "observed": observed,
        "required": required,
        "failed": failed,
        "status": "PASS" if not failed else "NOT_EVALUABLE",
    }
    return result


def _oracle_policy_arrays(
    data: dict[str, np.ndarray], policy_arrays: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    arrays = {key: np.asarray(value) for key, value in policy_arrays.items()}
    positive = np.asarray(data["gain"], dtype=np.float64) > 1e-12
    arrays["actions__ORACLE-POSITIVE-GAIN"] = np.where(
        positive,
        np.asarray(data["posterior_actions"], dtype=np.int64),
        np.asarray(data["hard_actions"], dtype=np.int64),
    )
    return arrays


def run_fit(stage: Path, output: Path, config: dict[str, Any]) -> str:
    data = load_npz(stage / "bundle.npz")
    utilities = np.load(stage / "utilities.npy", allow_pickle=False)
    validate_primary_integrity(data)
    counts = _population_counts(data)
    targets_preview = derive_targets(
        data, utilities, float(config.get("penalty", 1.25))
    )
    primary = np.asarray(data["primary"], dtype=bool)
    active = primary[:, None] & np.asarray(data["disagreement"], dtype=bool)
    harm = np.asarray(targets_preview["harm"], dtype=bool)
    incompatibility = np.asarray(
        targets_preview["posterior_incompatibility"], dtype=bool
    )
    counts.update(
        {
            "harm_positive_tokens": int((harm & active).any(axis=1).sum()),
            "harm_nonpositive_tokens": int(((~harm) & active).any(axis=1).sum()),
            "incompatibility_positive_rows": int((incompatibility & active).sum()),
            "incompatibility_positive_tokens": int(
                (incompatibility & active).any(axis=1).sum()
            ),
        }
    )
    minimums = config["minimums"]
    minimum_status = _require_minimums(
        counts,
        {
            "primary_tokens": int(minimums["gate_fit_tokens"]),
            "disagreement_rows": int(minimums["gate_fit_disagreement_rows"]),
            "disagreement_tokens": int(minimums["gate_fit_disagreement_tokens"]),
            "harm_positive_tokens": int(minimums["gate_fit_harm_tokens"]),
            "harm_nonpositive_tokens": int(minimums["gate_fit_nonharm_tokens"]),
            "incompatibility_positive_rows": int(
                minimums["gate_fit_incompatibility_rows"]
            ),
            "incompatibility_positive_tokens": int(
                minimums["gate_fit_incompatibility_tokens"]
            ),
        },
        "fit",
    )
    if minimum_status["status"] != "PASS":
        write_json(output / "fit_not_evaluable.json", minimum_status)
        return "NOT_EVALUABLE"
    true_states, state_arrays, targets, true_objects = fit_true_models(
        data, utilities, float(config.get("penalty", 1.25)), return_objects=True
    )
    control_states, control_arrays, control_meta, control_objects = fit_control_models(
        data, targets, return_objects=True
    )
    states = {**true_states, **control_states}
    objects = {**true_objects, **control_objects}
    state_arrays.update(control_arrays)
    mappings = {
        key: value
        for key, value in state_arrays.items()
        if key.startswith("mapping__") or key.startswith("target__")
    }
    for key in mappings:
        del state_arrays[key]
    state_arrays.update(_linear_arrays(states))
    scores = score_true_models(states, state_arrays, data)

    write_npz(output / "model_state_arrays.npz", state_arrays)
    write_npz(output / "train_scores.npz", scores)
    write_npz(output / "max_displacement_mappings.npz", mappings)
    write_json(output / "max_displacement_diagnostics.json", control_meta)
    model_dir = output / "model_states"
    model_dir.mkdir()
    manifest: dict[str, Any] = {"models": {}, "portable_states": states}
    for identifier, model in sorted(objects.items()):
        if model is None:
            raise RuntimeError(f"model not evaluable: {identifier}")
        if states[identifier]["kind"] in {"ridge", "logistic"}:
            model = PortableLinearEstimator(states[identifier])
        path = model_dir / f"{identifier}.joblib"
        joblib.dump(model, path, compress=0)
        manifest["models"][identifier] = {
            "path": f"model_states/{path.name}",
            "sha256": sha256_file(path),
        }
    write_json(model_dir / "manifest.json", manifest)
    write_json(
        output / "feature_schema.json",
        {"count": int(config["features"]["count"]), "names_source": config["features"]["source"]},
    )
    files = [
        "feature_schema.json",
        "model_state_arrays.npz",
        "train_scores.npz",
        "max_displacement_mappings.npz",
        "max_displacement_diagnostics.json",
        "model_states/manifest.json",
    ]
    _freeze(
        output,
        "fit_freeze.json",
        "fit",
        files,
        {
            "T_primary": primary_tokens(data).tolist(),
            "minimums": minimum_status,
            "control_status": control_meta,
            "config_sha256": sha256_file(stage / "config.json"),
            "source_bindings_sha256": sha256_file(stage / "source_bindings.json"),
            "preparation_freeze_sha256": sha256_file(stage / "preparation_freeze.json"),
            "fit_bundle_sha256": sha256_file(stage / "bundle.npz"),
        },
    )
    return "FIT_COMPLETE"


def _load_states(stage: Path) -> tuple[dict[str, dict[str, Any]], dict[str, np.ndarray]]:
    payload = load_json(stage / "model_states_manifest.json")
    return payload["portable_states"], load_npz(stage / "model_state_arrays.npz")


def run_calibrate(stage: Path, output: Path, config: dict[str, Any]) -> str:
    fit_freeze = validate_freeze(
        stage / "fit_freeze.json",
        expected_phase="fit",
        bindings={
            "model_states/manifest.json": stage / "model_states_manifest.json",
            "model_state_arrays.npz": stage / "model_state_arrays.npz",
        },
    )
    _validate_provenance_chain(stage, fit_freeze)
    data = load_npz(stage / "inference_bundle.npz")
    validate_inference_safe_view(data)
    states, state_arrays = _load_states(stage)
    scores = score_true_models(states, state_arrays, data)
    calibration, policy_arrays = calibrate_policies(data, scores)
    observed = _population_counts(data)
    observed["proposal_tokens"] = int(
        np.asarray(policy_arrays["proposal__hgb"], dtype=bool).any(axis=1).sum()
    )
    main = config["main_policies"]
    for name, identifier in main.items():
        observed[f"authorized_tokens__{name}"] = calibration["policies"][identifier][
            "support"
        ]["authorized_pair_tokens"]
    minimums = config["minimums"]
    minimum_status = _require_minimums(
        observed,
        {
            "primary_tokens": int(minimums["gate_select_tokens"]),
            "disagreement_rows": int(minimums["gate_select_disagreement_rows"]),
            "disagreement_tokens": int(
                minimums["gate_select_disagreement_tokens"]
            ),
            "proposal_tokens": int(minimums["gate_select_proposal_tokens"]),
            "authorized_tokens__mean": int(
                minimums["gate_select_authorized_tokens_per_main"]
            ),
            "authorized_tokens__tail": int(
                minimums["gate_select_authorized_tokens_per_main"]
            ),
        },
        "calibrate_scores",
    )
    if minimum_status["status"] != "PASS":
        write_json(output / "calibration_not_evaluable.json", minimum_status)
        return "NOT_EVALUABLE"
    write_npz(output / "validation_scores.npz", scores)
    write_npz(output / "validation_policy_arrays.npz", policy_arrays)
    _freeze(
        output,
        "calibration_freeze.json",
        "calibrate_scores",
        ["validation_scores.npz", "validation_policy_arrays.npz"],
        {
            "T_primary": primary_tokens(data).tolist(),
            "calibration": calibration,
            "minimums": minimum_status,
            "fit_freeze_sha256": sha256_file(stage / "fit_freeze.json"),
            "control_status": fit_freeze["control_status"],
            "config_sha256": sha256_file(stage / "config.json"),
            "source_bindings_sha256": sha256_file(stage / "source_bindings.json"),
            "preparation_freeze_sha256": sha256_file(stage / "preparation_freeze.json"),
            "validation_inference_bundle_sha256": sha256_file(
                stage / "inference_bundle.npz"
            ),
        },
    )
    return "CALIBRATION_FROZEN"


def run_validate(stage: Path, output: Path, config: dict[str, Any]) -> str:
    calibration_freeze = validate_freeze(
        stage / "calibration_freeze.json",
        expected_phase="calibrate_scores",
        bindings={
            "validation_scores.npz": stage / "validation_scores.npz",
            "validation_policy_arrays.npz": stage / "validation_policy_arrays.npz",
        },
        require_all_frozen_files=True,
    )
    _validate_provenance_chain(stage, calibration_freeze)
    data = load_npz(stage / "truth_bundle.npz")
    safe = load_npz(stage / "inference_bundle.npz")
    validate_inference_safe_view(safe)
    for key, expected in inference_safe_view(data).items():
        np.testing.assert_array_equal(safe[key], expected)
    arrays = load_npz(stage / "validation_policy_arrays.npz")
    scores = load_npz(stage / "validation_scores.npz")
    utilities = np.load(stage / "utilities.npy", allow_pickle=False)
    summaries, metrics = evaluate_actions(
        data,
        utilities,
        float(config.get("penalty", 1.25)),
        _oracle_policy_arrays(data, arrays),
    )
    assignments = shard_assignment(data["pair_token"])
    shard_summaries: dict[str, Any] = {}
    for shard in range(int(config["shards"]["count"])):
        token_mask = assignments == shard
        shard_safe = {key: np.asarray(value)[token_mask] for key, value in safe.items()}
        shard_truth = {
            key: np.asarray(value)[token_mask]
            for key, value in data.items()
            if np.asarray(value).shape[:1] == (len(token_mask),)
        }
        shard_scores = {key: np.asarray(value)[token_mask] for key, value in scores.items()}
        shard_calibration, shard_policies = calibrate_policies(shard_safe, shard_scores)
        shard_observed = _population_counts(shard_safe)
        shard_observed["proposal_tokens"] = int(
            np.asarray(shard_policies["proposal__hgb"], dtype=bool).any(axis=1).sum()
        )
        for name, identifier in config["main_policies"].items():
            shard_observed[f"authorized_tokens__{name}"] = shard_calibration["policies"][identifier]["support"]["authorized_pair_tokens"]
        minimums = config["minimums"]
        shard_minimums = _require_minimums(
            shard_observed,
            {
                "primary_tokens": int(minimums["gate_select_shard_tokens"]),
                "disagreement_rows": int(minimums["gate_select_shard_disagreement_rows"]),
                "disagreement_tokens": int(minimums["gate_select_shard_disagreement_tokens"]),
                "proposal_tokens": int(minimums["gate_select_shard_proposal_tokens"]),
                "authorized_tokens__mean": int(minimums["gate_select_shard_authorized_tokens_per_main"]),
                "authorized_tokens__tail": int(minimums["gate_select_shard_authorized_tokens_per_main"]),
            },
            f"validation_shard_{shard}",
        )
        shard_summary: dict[str, Any] = {
            "minimums": shard_minimums,
            "calibration": shard_calibration,
            "status": shard_minimums["status"],
        }
        if shard_minimums["status"] == "PASS":
            evaluated, evaluated_arrays = evaluate_actions(
                shard_truth,
                utilities,
                float(config.get("penalty", 1.25)),
                _oracle_policy_arrays(shard_truth, shard_policies),
            )
            shard_summary["policies"] = evaluated
            mean_id = config["main_policies"]["mean"]
            tail_id = config["main_policies"]["tail"]
            hgb_proposer = "HGB-PROPOSER-ONLY"

            def scalar_delta(left: str, right: str, metric: str) -> float:
                return float(evaluated[left][metric] - evaluated[right][metric])

            incompatibility_controls = [
                model_id("control-hgb", "posterior_incompatibility", seed)
                for seed in INCOMPATIBILITY_CONTROL_SEEDS
            ]
            harm_controls = [
                model_id("control-hgb", "harm", seed) for seed in HARM_CONTROL_SEEDS
            ]
            mean_control_regret = float(
                np.mean([evaluated[name]["regret"] for name in incompatibility_controls])
            )
            mean_control_worst = float(
                np.mean([evaluated[name]["worst_regret"] for name in harm_controls])
            )
            deltas = {
                "incompatibility_vs_hard_regret": scalar_delta(mean_id, "HARD-SET", "regret"),
                "harm_vs_hard_worst_regret": scalar_delta(tail_id, "HARD-SET", "worst_regret"),
                "incompatibility_vs_harm_regret": scalar_delta(mean_id, tail_id, "regret"),
                "incompatibility_vs_mean_control_regret": float(evaluated[mean_id]["regret"] - mean_control_regret),
                "harm_vs_mean_control_worst_regret": float(evaluated[tail_id]["worst_regret"] - mean_control_worst),
                "incompatibility_vs_hgb_proposer_regret": scalar_delta(mean_id, hgb_proposer, "regret"),
                "harm_vs_hgb_proposer_worst_regret": scalar_delta(tail_id, hgb_proposer, "worst_regret"),
            }
            shard_summary["directional_deltas"] = {
                name: {"value": value, "sign": int(np.sign(value))}
                for name, value in deltas.items()
            }
            shard_summary["nonidentity_vs_global"] = {}
            for name, identifier in config["main_policies"].items():
                local_actions = np.asarray(shard_policies[f"actions__{identifier}"])
                global_actions = np.asarray(arrays[f"actions__{identifier}"])[token_mask]
                changed = local_actions != global_actions
                shard_summary["nonidentity_vs_global"][name] = {
                    "changed_rows": int(changed.sum()),
                    "changed_pair_tokens": int(changed.any(axis=1).sum()),
                    "identical": bool(not np.any(changed)),
                }
            for key, value in evaluated_arrays.items():
                metrics[f"shard{shard}__{key}"] = value
        shard_summaries[str(shard)] = shard_summary
    write_npz(output / "validation_metrics.npz", metrics)
    write_json(
        output / "validation_summary.json",
        {"policies": summaries, "shards": shard_summaries},
    )
    _freeze(
        output,
        "validation_freeze.json",
        "validate",
        ["validation_metrics.npz", "validation_summary.json"],
        {
            "calibration_freeze_sha256": sha256_file(stage / "calibration_freeze.json"),
            "config_sha256": sha256_file(stage / "config.json"),
            "source_bindings_sha256": sha256_file(stage / "source_bindings.json"),
            "preparation_freeze_sha256": sha256_file(stage / "preparation_freeze.json"),
            "validation_truth_bundle_sha256": sha256_file(stage / "truth_bundle.npz"),
        },
    )
    return "VALIDATION_COMPLETE"


def run_monitor_apply(stage: Path, output: Path, config: dict[str, Any]) -> str:
    fit_freeze = validate_freeze(
        stage / "fit_freeze.json",
        expected_phase="fit",
        bindings={
            "model_states/manifest.json": stage / "model_states_manifest.json",
            "model_state_arrays.npz": stage / "model_state_arrays.npz",
        },
    )
    _validate_provenance_chain(stage, fit_freeze)
    calibration_freeze = validate_freeze(
        stage / "calibration_freeze.json",
        expected_phase="calibrate_scores",
        bindings={},
    )
    _validate_provenance_chain(stage, calibration_freeze)
    validation_freeze = validate_freeze(
        stage / "validation_freeze.json",
        expected_phase="validate",
        bindings={},
    )
    _validate_provenance_chain(stage, validation_freeze)
    data = load_npz(stage / "inference_bundle.npz")
    validate_inference_safe_view(data)
    minimums = config["minimums"]
    minimum_status = _require_minimums(
        _population_counts(data),
        {
            "primary_tokens": int(minimums["sealed_monitor_tokens"]),
            "disagreement_rows": int(minimums["sealed_monitor_disagreement_rows"]),
            "disagreement_tokens": int(
                minimums["sealed_monitor_disagreement_tokens"]
            ),
        },
        "monitor_apply",
    )
    if minimum_status["status"] != "PASS":
        write_json(output / "monitor_not_evaluable.json", minimum_status)
        return "NOT_EVALUABLE"
    states, state_arrays = _load_states(stage)
    calibration = calibration_freeze["calibration"]
    scores = score_true_models(states, state_arrays, data)
    arrays = apply_calibrated_policies(data, scores, calibration)
    supports = {
        key.removeprefix("authorized__"): support(value, data["primary"])
        for key, value in arrays.items()
        if key.startswith("authorized__")
    }
    write_npz(output / "monitor_scores.npz", scores)
    write_npz(output / "monitor_policy_arrays.npz", arrays)
    _freeze(
        output,
        "monitor_action_freeze.json",
        "monitor_apply",
        ["monitor_scores.npz", "monitor_policy_arrays.npz"],
        {
            "calibration_freeze_sha256": sha256_file(stage / "calibration_freeze.json"),
            "fit_freeze_sha256": sha256_file(stage / "fit_freeze.json"),
            "validation_freeze_sha256": sha256_file(stage / "validation_freeze.json"),
            "config_sha256": sha256_file(stage / "config.json"),
            "source_bindings_sha256": sha256_file(stage / "source_bindings.json"),
            "preparation_freeze_sha256": sha256_file(stage / "preparation_freeze.json"),
            "model_states_manifest_sha256": sha256_file(
                stage / "model_states_manifest.json"
            ),
            "model_state_arrays_sha256": sha256_file(
                stage / "model_state_arrays.npz"
            ),
            "monitor_inference_bundle_sha256": sha256_file(
                stage / "inference_bundle.npz"
            ),
            "control_status": calibration_freeze["control_status"],
            "supports": supports,
            "minimums": minimum_status,
            "T_primary": primary_tokens(data).tolist(),
            "truth_materialized": False,
        },
    )
    return "MONITOR_ACTIONS_FROZEN"


def _delta(
    data: dict[str, np.ndarray], metrics: dict[str, np.ndarray], left: str, right: str,
    metric: str, indices: np.ndarray
) -> dict[str, float]:
    tokens_left, values_left = ordered_primary_metric(data, metrics[f"metric__{left}__{metric}"])
    tokens_right, values_right = ordered_primary_metric(data, metrics[f"metric__{right}__{metric}"])
    np.testing.assert_array_equal(tokens_left, tokens_right)
    delta = values_left - values_right
    sampled = delta[indices].mean(axis=1)
    low, high = np.percentile(sampled, [2.5, 97.5])
    return {"mean_diff": float(delta.mean()), "ci95_low": float(low), "ci95_high": float(high)}


def _delta_values(
    data: dict[str, np.ndarray], left: np.ndarray, right: np.ndarray, indices: np.ndarray
) -> dict[str, float]:
    tokens_left, values_left = ordered_primary_metric(data, left)
    tokens_right, values_right = ordered_primary_metric(data, right)
    np.testing.assert_array_equal(tokens_left, tokens_right)
    delta = values_left - values_right
    sampled = delta[indices].mean(axis=1)
    low, high = np.percentile(sampled, [2.5, 97.5])
    return {"mean_diff": float(delta.mean()), "ci95_low": float(low), "ci95_high": float(high)}


def _factorial_contrasts(
    data: dict[str, np.ndarray], metrics: dict[str, np.ndarray], indices: np.ndarray
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    metric_names = ("accuracy", "compatible", "regret", "worst_regret")

    def add(name: str, left: str, right: str) -> None:
        result[name] = {
            metric: _delta(data, metrics, left, right, metric, indices)
            for metric in metric_names
        }

    for proposer in ("ridge", "hgb"):
        for guard in ("logistic", "hgb"):
            for q in (0.7, 0.9):
                add(
                    f"target__{proposer}__{guard}__q{int(q*100)}",
                    policy_id(proposer, guard, "posterior_incompatibility", q),
                    policy_id(proposer, guard, "harm", q),
                )
            for target in ("harm", "posterior_incompatibility"):
                add(
                    f"quantile__{proposer}__{guard}__{target}",
                    policy_id(proposer, guard, target, 0.9),
                    policy_id(proposer, guard, target, 0.7),
                )
    for guard in ("logistic", "hgb"):
        for target in ("harm", "posterior_incompatibility"):
            for q in (0.7, 0.9):
                add(
                    f"proposer__{guard}__{target}__q{int(q*100)}",
                    policy_id("hgb", guard, target, q),
                    policy_id("ridge", guard, target, q),
                )
    for proposer in ("ridge", "hgb"):
        for target in ("harm", "posterior_incompatibility"):
            for q in (0.7, 0.9):
                add(
                    f"guard__{proposer}__{target}__q{int(q*100)}",
                    policy_id(proposer, "hgb", target, q),
                    policy_id(proposer, "logistic", target, q),
                )
    for proposer in ("ridge", "hgb"):
        for guard in ("logistic", "hgb"):
            name = f"target_x_quantile__{proposer}__{guard}"
            result[name] = {}
            for metric in metric_names:
                ids = {
                    (target, q): policy_id(proposer, guard, target, q)
                    for target in ("harm", "posterior_incompatibility")
                    for q in (0.7, 0.9)
                }
                values = {
                    key: metrics[f"metric__{identifier}__{metric}"]
                    for key, identifier in ids.items()
                }
                interaction = (
                    values[("posterior_incompatibility", 0.9)]
                    - values[("harm", 0.9)]
                    - values[("posterior_incompatibility", 0.7)]
                    + values[("harm", 0.7)]
                )
                result[name][metric] = _delta_values(
                    data, interaction, np.zeros_like(interaction), indices
                )
    if len(result) != 36:
        raise AssertionError("Wave 59 factorial contrast count drifted")
    return result


def _prospective_patterns(
    data: dict[str, np.ndarray],
    metrics: dict[str, np.ndarray],
    indices: np.ndarray,
    config: dict[str, Any],
    action_freeze: dict[str, Any],
) -> dict[str, Any]:
    mean_id = config["main_policies"]["mean"]
    tail_id = config["main_policies"]["tail"]
    mean_hard = {
        metric: _delta(data, metrics, mean_id, "HARD-SET", metric, indices)
        for metric in ("accuracy", "compatible", "regret", "worst_regret")
    }
    tail_hard = {
        metric: _delta(data, metrics, tail_id, "HARD-SET", metric, indices)
        for metric in ("accuracy", "compatible", "regret", "worst_regret")
    }
    control_deltas: dict[str, dict[str, float] | str] = {}
    for target_name, seeds, main_id, metric in (
        ("posterior_incompatibility", INCOMPATIBILITY_CONTROL_SEEDS, mean_id, "regret"),
        ("harm", HARM_CONTROL_SEEDS, tail_id, "worst_regret"),
    ):
        status = action_freeze["control_status"]["families"][target_name]["status"]
        if status != "PASS":
            control_deltas[target_name] = "NOT_EVALUABLE"
            continue
        ids = [model_id("control-hgb", target_name, seed) for seed in seeds]
        control_mean = mean_control_metric(metrics, ids, metric)
        control_deltas[target_name] = _delta_values(
            data, metrics[f"metric__{main_id}__{metric}"], control_mean, indices
        )

    minimum = int(config["pattern"]["authorized_pair_tokens_min"])
    mean_control = control_deltas["posterior_incompatibility"]
    tail_control = control_deltas["harm"]
    mean_conditions: dict[str, bool | str] = {
        "regret_mean_at_most_negative_0_005": mean_hard["regret"]["mean_diff"] <= -0.005,
        "regret_ci95_high_below_zero": mean_hard["regret"]["ci95_high"] < 0.0,
        "accuracy_ci95_low_at_least_negative_0_01": mean_hard["accuracy"]["ci95_low"] >= -0.01,
        "compatibility_ci95_low_at_least_zero": mean_hard["compatible"]["ci95_low"] >= 0.0,
        "worst_regret_ci95_high_at_most_0_01": mean_hard["worst_regret"]["ci95_high"] <= 0.01,
        "regret_vs_mean_max_displacement_ci95_high_below_zero": (
            "NOT_EVALUABLE"
            if mean_control == "NOT_EVALUABLE"
            else mean_control["ci95_high"] < 0.0
        ),
        "authorized_pair_tokens_at_least_25": action_freeze["supports"][mean_id]["authorized_pair_tokens"] >= minimum,
        "replay_exact": "PENDING",
    }
    tail_conditions: dict[str, bool | str] = {
        "worst_regret_mean_at_most_negative_0_01": tail_hard["worst_regret"]["mean_diff"] <= -0.01,
        "worst_regret_ci95_high_below_zero": tail_hard["worst_regret"]["ci95_high"] < 0.0,
        "regret_ci95_high_at_most_zero": tail_hard["regret"]["ci95_high"] <= 0.0,
        "accuracy_ci95_low_at_least_negative_0_01": tail_hard["accuracy"]["ci95_low"] >= -0.01,
        "compatibility_ci95_low_at_least_zero": tail_hard["compatible"]["ci95_low"] >= 0.0,
        "worst_regret_vs_mean_max_displacement_ci95_high_below_zero": (
            "NOT_EVALUABLE"
            if tail_control == "NOT_EVALUABLE"
            else tail_control["ci95_high"] < 0.0
        ),
        "authorized_pair_tokens_at_least_25": action_freeze["supports"][tail_id]["authorized_pair_tokens"] >= minimum,
        "replay_exact": "PENDING",
    }
    mean_without_replay = [value for key, value in mean_conditions.items() if key != "replay_exact"]
    tail_without_replay = [value for key, value in tail_conditions.items() if key != "replay_exact"]
    return {
        "incompatibility": {
            "conditions": mean_conditions,
            "aggregate_without_replay": aggregate_ternary(mean_without_replay),
            "control_delta": mean_control,
            "replay_exact": "PENDING",
            "aggregate_with_replay": None,
        },
        "harm": {
            "conditions": tail_conditions,
            "aggregate_without_replay": aggregate_ternary(tail_without_replay),
            "control_delta": tail_control,
            "replay_exact": "PENDING",
            "aggregate_with_replay": None,
        },
    }


def run_monitor_evaluate(stage: Path, output: Path, config: dict[str, Any]) -> str:
    action_freeze = validate_freeze(
        stage / "monitor_action_freeze.json",
        expected_phase="monitor_apply",
        bindings={"monitor_policy_arrays.npz": stage / "monitor_policy_arrays.npz"},
    )
    _validate_provenance_chain(stage, action_freeze)
    data = load_npz(stage / "truth_bundle.npz")
    arrays = load_npz(stage / "monitor_policy_arrays.npz")
    utilities = np.load(stage / "utilities.npy", allow_pickle=False)
    summaries, metrics = evaluate_actions(
        data,
        utilities,
        float(config.get("penalty", 1.25)),
        _oracle_policy_arrays(data, arrays),
    )
    tokens = primary_tokens(data)
    indices = bootstrap_indices(tokens, int(config["bootstrap"]["replicates"]))
    contrasts: dict[str, Any] = {}
    main = config["main_policies"]
    for name, identifier in main.items():
        contrasts[f"{name}_vs_hard"] = {
            metric: _delta(data, metrics, identifier, "HARD-SET", metric, indices)
            for metric in ("accuracy", "compatible", "regret", "worst_regret")
        }
        contrasts[f"{name}_vs_hgb_proposer_only"] = {
            metric: _delta(
                data, metrics, identifier, "HGB-PROPOSER-ONLY", metric, indices
            )
            for metric in ("accuracy", "compatible", "regret", "worst_regret")
        }
    contrasts["head_to_head"] = {
        metric: _delta(data, metrics, main["mean"], main["tail"], metric, indices)
        for metric in ("accuracy", "compatible", "regret", "worst_regret")
    }
    contrasts["factorial"] = _factorial_contrasts(data, metrics, indices)
    patterns = _prospective_patterns(data, metrics, indices, config, action_freeze)
    write_npz(output / "bootstrap_indices.npz", {"indices": indices, "pair_token": tokens})
    write_npz(output / "analysis_arrays.npz", metrics)
    write_json(
        output / "analysis.json",
        {
            "status": "COMPLETE",
            "summaries": summaries,
            "contrasts": contrasts,
            "prospective_patterns": patterns,
            "scientific_decision": None,
            "decision_authority": "user",
        },
    )
    return "COMPLETE"


def execute(stage: Path, output: Path, phase: str) -> str:
    validate_stage(stage, phase)
    config = load_json(stage / "config.json")
    validate_pre_draw_config(config)
    if output.exists():
        if any(output.iterdir()):
            raise RuntimeError("worker output must start empty")
    else:
        output.mkdir(parents=True)
    if phase == "fit":
        return run_fit(stage, output, config)
    if phase == "calibrate_scores":
        return run_calibrate(stage, output, config)
    if phase == "validate":
        return run_validate(stage, output, config)
    if phase == "monitor_apply":
        return run_monitor_apply(stage, output, config)
    if phase == "monitor_evaluate":
        return run_monitor_evaluate(stage, output, config)
    raise ValueError(phase)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--forbidden-probe", type=Path, action="append", default=[])
    return parser.parse_args()


def _security_state() -> dict[str, Any]:
    status = Path("/proc/self/status").read_text(encoding="utf-8")
    fields = {}
    for line in status.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            fields[key] = value.strip()
    return {
        "effective_capabilities_hex": fields.get("CapEff"),
        "no_new_privileges": int(fields.get("NoNewPrivs", "-1")),
        "supplementary_groups": os.getgroups(),
    }


def _probe_forbidden(paths: list[Path]) -> list[dict[str, Any]]:
    result = []
    for raw in paths:
        path = raw.resolve(strict=False)
        denied = False
        error_type = None
        try:
            with path.open("rb") as handle:
                handle.read(1)
        except (PermissionError, FileNotFoundError) as error:
            denied = True
            error_type = type(error).__name__
        result.append({"path_sha256": hashlib.sha256(str(path).encode()).hexdigest(), "denied": denied, "error_type": error_type})
    return result


def main() -> None:
    args = parse_args()
    stage = args.stage.resolve(strict=True)
    request = load_json(stage / "phase_request.json")
    if os.environ.get("WAVE59_STAGED_RUNTIME") != "1":
        raise RuntimeError("Wave 59 worker requires staged runtime")
    if os.geteuid() != 65534 or os.getegid() != 65534:
        raise RuntimeError("Wave 59 worker must run as nobody/nogroup")
    status = execute(stage, args.output.resolve(), str(request["phase"]))
    inventory = {
        str(path.relative_to(args.output.resolve())): sha256_file(path)
        for path in sorted(args.output.resolve().rglob("*"))
        if path.is_file()
    }
    write_json(
        args.output.resolve() / "access_receipt.json",
        {
            "phase": request["phase"],
            "status": status,
            "effective_uid": os.geteuid(),
            "effective_gid": os.getegid(),
            "process_security": _security_state(),
            "threadpools": threadpool_info(),
            "stage_hashes": {
                name: sha256_file(stage / name)
                for name in sorted(PHASE_FILES[str(request["phase"])])
            },
            "forbidden_probes": _probe_forbidden(args.forbidden_probe),
            "output_inventory_before_receipt": inventory,
            "benchmark_root_received": False,
        },
    )
    print(json.dumps({"phase": request["phase"], "status": status}, sort_keys=True))


if __name__ == "__main__":
    main()
    aggregate_ternary,
    mean_control_metric,
    policy_id,
