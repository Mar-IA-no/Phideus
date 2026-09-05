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

from geometria_proporcional.wave59_hgb_guard_bracket import (
    HARM_CONTROL_SEEDS,
    INCOMPATIBILITY_CONTROL_SEEDS,
    aggregate_ternary,
    apply_calibrated_policies,
    bootstrap_indices,
    calibrate_policies,
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
    support,
    validate_inference_safe_view,
    validate_pre_draw_config,
    validate_primary_integrity,
)


PHASE_FILES = {
    "fit": {"phase_request.json", "config.json", "bundle.npz", "utilities.npy"},
    "calibrate_scores": {
        "phase_request.json",
        "config.json",
        "inference_bundle.npz",
        "model_states.json",
        "model_state_arrays.npz",
        "fit_freeze.json",
    },
    "validate": {
        "phase_request.json",
        "config.json",
        "truth_bundle.npz",
        "validation_policy_arrays.npz",
        "calibration_freeze.json",
        "utilities.npy",
    },
    "monitor_apply": {
        "phase_request.json",
        "config.json",
        "inference_bundle.npz",
        "model_states.json",
        "model_state_arrays.npz",
        "fit_freeze.json",
        "calibration_freeze.json",
    },
    "monitor_evaluate": {
        "phase_request.json",
        "config.json",
        "truth_bundle.npz",
        "monitor_policy_arrays.npz",
        "monitor_action_freeze.json",
        "utilities.npy",
    },
}


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


def run_fit(stage: Path, output: Path, config: dict[str, Any]) -> str:
    data = load_npz(stage / "bundle.npz")
    utilities = np.load(stage / "utilities.npy", allow_pickle=False)
    validate_primary_integrity(data)
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

    write_json(output / "model_states.json", _jsonable_states(states))
    write_npz(output / "model_state_arrays.npz", state_arrays)
    write_npz(output / "train_scores.npz", scores)
    write_npz(output / "max_displacement_mappings.npz", mappings)
    write_json(output / "max_displacement_diagnostics.json", control_meta)
    model_dir = output / "model_states"
    model_dir.mkdir()
    manifest: dict[str, Any] = {"models": {}}
    for identifier, model in sorted(objects.items()):
        if model is None:
            raise RuntimeError(f"model not evaluable: {identifier}")
        path = model_dir / f"{identifier}.joblib"
        joblib.dump(model, path, compress=0)
        manifest["models"][identifier] = {
            "path": f"model_states/{path.name}",
            "sha256": sha256_file(path),
        }
    write_json(output / "model_states_manifest.json", manifest)
    files = [
        "model_states.json",
        "model_state_arrays.npz",
        "train_scores.npz",
        "max_displacement_mappings.npz",
        "max_displacement_diagnostics.json",
        "model_states_manifest.json",
    ]
    _freeze(
        output,
        "fit_freeze.json",
        "fit",
        files,
        {"T_primary": primary_tokens(data).tolist(), "control_status": control_meta},
    )
    return "FIT_COMPLETE"


def _load_states(stage: Path) -> tuple[dict[str, dict[str, Any]], dict[str, np.ndarray]]:
    payload = load_json(stage / "model_states.json")
    return payload["models"], load_npz(stage / "model_state_arrays.npz")


def run_calibrate(stage: Path, output: Path, config: dict[str, Any]) -> str:
    data = load_npz(stage / "inference_bundle.npz")
    validate_inference_safe_view(data)
    states, state_arrays = _load_states(stage)
    fit_freeze = load_json(stage / "fit_freeze.json")
    scores = score_true_models(states, state_arrays, data)
    calibration, policy_arrays = calibrate_policies(data, scores)
    main = config["main_policies"]
    for identifier in (main["mean"], main["tail"]):
        if calibration["policies"][identifier]["support"]["authorized_pair_tokens"] < int(
            config["minimums"]["gate_select_authorized_tokens_per_main"]
        ):
            return "NOT_EVALUABLE"
    write_npz(output / "validation_scores.npz", scores)
    write_npz(output / "validation_policy_arrays.npz", policy_arrays)
    write_json(output / "calibration.json", calibration)
    _freeze(
        output,
        "calibration_freeze.json",
        "calibrate_scores",
        ["validation_scores.npz", "validation_policy_arrays.npz", "calibration.json"],
        {
            "T_primary": primary_tokens(data).tolist(),
            "calibration": calibration,
            "fit_freeze_sha256": sha256_file(stage / "fit_freeze.json"),
            "control_status": fit_freeze["control_status"],
        },
    )
    return "CALIBRATION_FROZEN"


def run_validate(stage: Path, output: Path, config: dict[str, Any]) -> str:
    data = load_npz(stage / "truth_bundle.npz")
    arrays = load_npz(stage / "validation_policy_arrays.npz")
    utilities = np.load(stage / "utilities.npy", allow_pickle=False)
    summaries, metrics = evaluate_actions(data, utilities, float(config.get("penalty", 1.25)), arrays)
    write_npz(output / "validation_metrics.npz", metrics)
    write_json(output / "validation_summary.json", {"policies": summaries})
    _freeze(
        output,
        "validation_freeze.json",
        "validate",
        ["validation_metrics.npz", "validation_summary.json"],
        {"calibration_freeze_sha256": sha256_file(stage / "calibration_freeze.json")},
    )
    return "VALIDATION_COMPLETE"


def run_monitor_apply(stage: Path, output: Path, config: dict[str, Any]) -> str:
    data = load_npz(stage / "inference_bundle.npz")
    validate_inference_safe_view(data)
    states, state_arrays = _load_states(stage)
    calibration_freeze = load_json(stage / "calibration_freeze.json")
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
            "control_status": calibration_freeze["control_status"],
            "supports": supports,
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
    mean_conditions: list[bool | str] = [
        mean_hard["regret"]["mean_diff"] <= -0.005,
        mean_hard["regret"]["ci95_high"] < 0.0,
        mean_hard["accuracy"]["ci95_low"] >= -0.01,
        mean_hard["compatible"]["ci95_low"] >= 0.0,
        mean_hard["worst_regret"]["ci95_high"] <= 0.01,
        (
            "NOT_EVALUABLE"
            if mean_control == "NOT_EVALUABLE"
            else mean_control["ci95_high"] < 0.0
        ),
        action_freeze["supports"][mean_id]["authorized_pair_tokens"] >= minimum,
    ]
    tail_conditions: list[bool | str] = [
        tail_hard["worst_regret"]["mean_diff"] <= -0.01,
        tail_hard["worst_regret"]["ci95_high"] < 0.0,
        tail_hard["regret"]["ci95_high"] <= 0.0,
        tail_hard["accuracy"]["ci95_low"] >= -0.01,
        tail_hard["compatible"]["ci95_low"] >= 0.0,
        (
            "NOT_EVALUABLE"
            if tail_control == "NOT_EVALUABLE"
            else tail_control["ci95_high"] < 0.0
        ),
        action_freeze["supports"][tail_id]["authorized_pair_tokens"] >= minimum,
    ]
    return {
        "incompatibility": {
            "conditions_without_replay": mean_conditions,
            "aggregate_without_replay": aggregate_ternary(mean_conditions),
            "control_delta": mean_control,
            "replay_exact": "PENDING",
            "aggregate_with_replay": None,
        },
        "harm": {
            "conditions_without_replay": tail_conditions,
            "aggregate_without_replay": aggregate_ternary(tail_conditions),
            "control_delta": tail_control,
            "replay_exact": "PENDING",
            "aggregate_with_replay": None,
        },
    }


def run_monitor_evaluate(stage: Path, output: Path, config: dict[str, Any]) -> str:
    data = load_npz(stage / "truth_bundle.npz")
    arrays = load_npz(stage / "monitor_policy_arrays.npz")
    utilities = np.load(stage / "utilities.npy", allow_pickle=False)
    summaries, metrics = evaluate_actions(data, utilities, float(config.get("penalty", 1.25)), arrays)
    tokens = primary_tokens(data)
    indices = bootstrap_indices(tokens, int(config["bootstrap"]["replicates"]))
    contrasts: dict[str, Any] = {}
    main = config["main_policies"]
    for name, identifier in main.items():
        contrasts[f"{name}_vs_hard"] = {
            metric: _delta(data, metrics, identifier, "HARD-SET", metric, indices)
            for metric in ("accuracy", "compatible", "regret", "worst_regret")
        }
    contrasts["head_to_head"] = {
        metric: _delta(data, metrics, main["mean"], main["tail"], metric, indices)
        for metric in ("accuracy", "compatible", "regret", "worst_regret")
    }
    contrasts["factorial"] = _factorial_contrasts(data, metrics, indices)
    action_freeze = load_json(stage / "monitor_action_freeze.json")
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
