#!/usr/bin/env python3
"""Physically isolated CPU worker for the three Wave 60 phases."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

from geometria_proporcional.wave56_contextual_gate import FEATURE_NAMES
from geometria_proporcional.wave60_frozen_policy_transport import (
    EVALUATE_SCHEMA,
    PHASE_FILES,
    PLAN_COMMIT,
    PLAN_SHA256,
    SCORE_APPLY_SCHEMA,
    SOURCE_HASHES,
    SOURCE_LAW_RECOVERY_BINDING,
    SOURCE_LAW_RECOVERY_REQUEST_SCHEMA,
    SOURCE_LAW_SCHEMA,
    USED_MODELS,
    WAVE59_SOURCE_COMMIT,
    apply_transport_policies,
    evaluate_transport_actions,
    expected_policy_array_keys,
    file_sha256,
    retrospective_source_verification,
    require_exact_keys,
    score_transport_models,
    validate_frozen_policy_spec,
    validate_transport_manifest,
)

SOURCE_BINDING_FILES = {
    "source_law_freeze.json",
    "transport_law_manifest.json",
    "transport_law_arrays.npz",
    "frozen_policy_spec.json",
    "feature_schema.json",
    "verify_source_law_receipt.json",
    "source_law_attestation.json",
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def write_json(path: Path, payload: Any) -> None:
    encoded = (
        json.dumps(
            payload, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False
        )
        + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(raw)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".npz", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(raw)
    try:
        np.savez(
            temporary,
            **{key: np.asarray(value) for key, value in sorted(arrays.items())},
        )
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _security_state() -> tuple[str, int]:
    fields: dict[str, str] = {}
    for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            fields[key] = value.strip()
    return fields.get("CapEff", ""), int(fields.get("NoNewPrivs", "-1"))


def _probe_forbidden(paths: list[Path]) -> list[dict[str, Any]]:
    result = []
    for raw in paths:
        resolved = raw.resolve(strict=False)
        denied = False
        error_type = None
        try:
            with resolved.open("rb") as handle:
                handle.read(1)
        except (PermissionError, FileNotFoundError) as error:
            denied = True
            error_type = type(error).__name__
        result.append(
            {
                "path_sha256": hashlib.sha256(str(resolved).encode()).hexdigest(),
                "denied": denied,
                "error_type": error_type,
            }
        )
    return result


def _stage_inventory(stage: Path) -> set[str]:
    if any(path.is_symlink() for path in stage.iterdir()):
        raise RuntimeError("Wave 60 stage may not contain symlinks")
    if any(not path.is_file() for path in stage.iterdir()):
        raise RuntimeError("Wave 60 stage must be flat and file-only")
    return {path.name for path in stage.iterdir()}


def validate_stage(stage: Path, phase: str) -> dict[str, Any]:
    if phase not in PHASE_FILES:
        raise RuntimeError("unknown Wave 60 phase")
    actual = _stage_inventory(stage)
    if actual != set(PHASE_FILES[phase]):
        raise RuntimeError(f"Wave 60 stage allowlist mismatch: {sorted(actual)}")
    request_name = (
        "source_law_request.json"
        if phase == "verify_source_law"
        else "phase_request.json"
    )
    request = load_json(stage / request_name)
    if request.get("phase", phase) != phase:
        raise RuntimeError("Wave 60 phase request drifted")
    if phase == "verify_source_law":
        recovery_request = (
            request.get("schema_version") == SOURCE_LAW_RECOVERY_REQUEST_SCHEMA
        )
        expected = {
            "schema_version",
            "plan_commit",
            "plan_sha256",
            "implementation_commit",
            "implementation_audit_commit",
            "implementation_audit_sha256",
            "source_paths",
            "source_sha256",
            "output_path",
            "runtime_budget",
        }
        if recovery_request:
            expected.add("recovery")
        if set(request) != expected:
            raise RuntimeError("Wave 60 source-law request keys drifted")
        if request["schema_version"] not in {
            SOURCE_LAW_SCHEMA,
            SOURCE_LAW_RECOVERY_REQUEST_SCHEMA,
        }:
            raise RuntimeError("Wave 60 source-law request schema drifted")
        if recovery_request:
            if request["recovery"] != SOURCE_LAW_RECOVERY_BINDING:
                raise RuntimeError("Wave 60 source-law recovery binding drifted")
            if request["output_path"] != (
                "data/geometria_proporcional/"
                "wave60_frozen_policy_transport_source_law_v2"
            ):
                raise RuntimeError("Wave 60 recovery output path drifted")
        for field in ("implementation_commit", "implementation_audit_commit"):
            value = request[field]
            if (
                not isinstance(value, str)
                or len(value) != 40
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise RuntimeError(f"Wave 60 source-law request {field} drifted")
        audit_hash = request["implementation_audit_sha256"]
        if (
            not isinstance(audit_hash, str)
            or len(audit_hash) != 64
            or any(character not in "0123456789abcdef" for character in audit_hash)
        ):
            raise RuntimeError("Wave 60 source-law implementation audit hash drifted")
        if request["runtime_budget"] != {
            "max_seconds": 900,
            "max_rss_bytes": 1610612736,
        }:
            raise RuntimeError("Wave 60 source-law runtime budget drifted")
        if set(request["source_sha256"]) != set(SOURCE_HASHES):
            raise RuntimeError("Wave 60 source-law request hash roster drifted")
        if (
            request["plan_commit"] != PLAN_COMMIT
            or request["plan_sha256"] != PLAN_SHA256
        ):
            raise RuntimeError("Wave 60 source-law request plan binding drifted")
        expected_source_names = set(PHASE_FILES[phase]) - {"source_law_request.json"}
        if (
            not isinstance(request["source_paths"], dict)
            or set(request["source_paths"]) != expected_source_names
        ):
            raise RuntimeError("Wave 60 source-law request path roster drifted")
        for name, expected_hash in SOURCE_HASHES.items():
            if request["source_sha256"][name] != expected_hash:
                raise RuntimeError(f"Wave 60 requested source hash drifted: {name}")
            if file_sha256(stage / name) != expected_hash:
                raise RuntimeError(f"Wave 60 physical source hash drifted: {name}")
    else:
        expected = {"schema_version", "phase", "allowed_files", "sha256", "run_role"}
        if set(request) != expected or set(request["allowed_files"]) != actual:
            raise RuntimeError("Wave 60 phase request inventory drifted")
        if request["run_role"] not in {"primary", "replay"}:
            raise RuntimeError("Wave 60 run role drifted")
        for name in sorted(actual - {"phase_request.json"}):
            if request["sha256"].get(name) != file_sha256(stage / name):
                raise RuntimeError(f"Wave 60 staged input hash drifted: {name}")
    return request


def _freeze_hashes(output: Path, names: list[str]) -> dict[str, str]:
    return {name: file_sha256(output / name) for name in names}


def validate_source_binding(
    stage: Path, request: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    binding = load_json(stage / "source_bindings.json")
    require_exact_keys(
        binding,
        {
            "schema_version",
            "run_role",
            "config_sha256",
            "source_authority_path_sha256",
            "source_law_freeze_sha256",
            "source_law_attestation_sha256",
            "copied_output_hashes",
            "hardlink_checks",
        },
        "source binding",
    )
    if (
        binding["schema_version"] != "wave60-source-binding-v1"
        or binding["run_role"] != request["run_role"]
        or binding["config_sha256"] != file_sha256(stage / "config.snapshot.json")
        or set(binding["copied_output_hashes"]) != SOURCE_BINDING_FILES
        or set(binding["hardlink_checks"]) != SOURCE_BINDING_FILES
        or any(binding["hardlink_checks"].values())
    ):
        raise RuntimeError("Wave 60 source binding drifted")
    for name in SOURCE_BINDING_FILES & {path.name for path in stage.iterdir()}:
        if binding["copied_output_hashes"][name] != file_sha256(stage / name):
            raise RuntimeError(f"Wave 60 staged source binding drifted: {name}")
    config = load_json(stage / "config.snapshot.json")
    return binding, config


def run_verify_source_law(stage: Path, output: Path, request: dict[str, Any]) -> str:
    if WAVE59_SOURCE_COMMIT not in (stage / "r454_audit.md").read_text(
        encoding="utf-8"
    ):
        raise RuntimeError(
            "Wave 60 source audit does not bind the Wave 59 execution commit"
        )
    manifest = load_json(stage / "wave59_model_states_manifest.json")
    state_arrays = load_npz(stage / "wave59_model_state_arrays.npz")
    calibration = load_json(stage / "wave59_calibration_freeze.json")
    inference = load_npz(stage / "wave59_monitor_inference_bundle.npz")
    scores = load_npz(stage / "wave59_monitor_scores.npz")
    policies = load_npz(stage / "wave59_monitor_policy_arrays.npz")
    action_freeze = load_json(stage / "wave59_monitor_action_freeze.json")
    expected_action_bindings = {
        "fit_freeze_sha256": SOURCE_HASHES["wave59_fit_freeze.json"],
        "calibration_freeze_sha256": SOURCE_HASHES["wave59_calibration_freeze.json"],
        "model_states_manifest_sha256": SOURCE_HASHES[
            "wave59_model_states_manifest.json"
        ],
        "model_state_arrays_sha256": SOURCE_HASHES["wave59_model_state_arrays.npz"],
        "monitor_inference_bundle_sha256": file_sha256(
            stage / "wave59_monitor_inference_bundle.npz"
        ),
    }
    if (
        action_freeze.get("schema_version") != "wave59-phase-freeze-v1"
        or action_freeze.get("phase") != "monitor_apply"
    ):
        raise RuntimeError("Wave 60 source monitor action freeze identity drifted")
    for field, expected_hash in expected_action_bindings.items():
        if action_freeze.get(field) != expected_hash:
            raise RuntimeError(f"Wave 60 source action binding drifted: {field}")
    action_files = action_freeze.get("files", {})
    if (
        action_files.get("monitor_scores.npz")
        != SOURCE_HASHES["wave59_monitor_scores.npz"]
        or action_files.get("monitor_policy_arrays.npz")
        != SOURCE_HASHES["wave59_monitor_policy_arrays.npz"]
    ):
        raise RuntimeError("Wave 60 source action output binding drifted")
    artifact_manifest = load_json(stage / "wave59_artifact_manifest.json")
    if artifact_manifest.get("closed_world") is not True:
        raise RuntimeError("Wave 60 source artifact manifest is not closed-world")
    expected_manifest_paths = {
        "fit/fit_freeze.json": SOURCE_HASHES["wave59_fit_freeze.json"],
        "fit/model_states/manifest.json": SOURCE_HASHES[
            "wave59_model_states_manifest.json"
        ],
        "fit/model_state_arrays.npz": SOURCE_HASHES["wave59_model_state_arrays.npz"],
        "calibration/calibration_freeze.json": SOURCE_HASHES[
            "wave59_calibration_freeze.json"
        ],
        "adjudication/monitor_scores.npz": SOURCE_HASHES["wave59_monitor_scores.npz"],
        "adjudication/monitor_policy_arrays.npz": SOURCE_HASHES[
            "wave59_monitor_policy_arrays.npz"
        ],
        "config.snapshot.json": SOURCE_HASHES["wave59_config_snapshot.json"],
    }
    classified: dict[str, str] = {}
    for class_rows in artifact_manifest.get("classes", {}).values():
        if isinstance(class_rows, dict):
            for relative, record in class_rows.items():
                if isinstance(record, dict) and "sha256" in record:
                    classified[relative] = record["sha256"]
    for relative, expected_hash in expected_manifest_paths.items():
        if classified.get(relative) != expected_hash:
            raise RuntimeError(f"Wave 60 source manifest binding drifted: {relative}")
    transport_manifest, transport_arrays, verification = (
        retrospective_source_verification(
            manifest, state_arrays, calibration, inference, scores, policies
        )
    )
    spec = verification["spec"]
    feature_schema = {
        "schema_version": SOURCE_LAW_SCHEMA,
        "feature_names": list(FEATURE_NAMES),
        "feature_count": len(FEATURE_NAMES),
        "dtype": "float64",
        "order_authority": "wave56_contextual_gate.FEATURE_NAMES",
    }
    write_json(output / "transport_law_manifest.json", transport_manifest)
    write_npz(output / "transport_law_arrays.npz", transport_arrays)
    write_json(output / "frozen_policy_spec.json", spec)
    write_json(output / "feature_schema.json", feature_schema)
    source_freeze = {
        "schema_version": SOURCE_LAW_SCHEMA,
        "phase": "verify_source_law",
        "source_law_request_sha256": file_sha256(stage / "source_law_request.json"),
        "source_commit": WAVE59_SOURCE_COMMIT,
        "implementation_commit": request["implementation_commit"],
        "implementation_audit_sha256": request["implementation_audit_sha256"],
        "source_hashes": dict(SOURCE_HASHES),
        "roster": {
            "used_models": list(USED_MODELS),
            "unused_models": transport_manifest["unused_models"],
            "reproduction": verification["diagnostics"],
        },
        "feature_schema_sha256": file_sha256(output / "feature_schema.json"),
        "transport_law_manifest_sha256": file_sha256(
            output / "transport_law_manifest.json"
        ),
        "transport_law_arrays_sha256": file_sha256(output / "transport_law_arrays.npz"),
        "frozen_policy_spec_sha256": file_sha256(output / "frozen_policy_spec.json"),
    }
    write_json(output / "source_law_freeze.json", source_freeze)
    return "SOURCE_LAW_VERIFIED"


def run_score_apply(stage: Path, output: Path, request: dict[str, Any]) -> str:
    binding, _ = validate_source_binding(stage, request)
    manifest = load_json(stage / "transport_law_manifest.json")
    arrays = load_npz(stage / "transport_law_arrays.npz")
    spec = load_json(stage / "frozen_policy_spec.json")
    inference = load_npz(stage / "sealed_monitor_inference_bundle.npz")
    validate_transport_manifest(manifest, arrays)
    validate_frozen_policy_spec(spec)
    feature_schema = load_json(stage / "feature_schema.json")
    require_exact_keys(
        feature_schema,
        {
            "schema_version",
            "feature_names",
            "feature_count",
            "dtype",
            "order_authority",
        },
        "feature schema",
    )
    if feature_schema != {
        "schema_version": SOURCE_LAW_SCHEMA,
        "feature_names": list(FEATURE_NAMES),
        "feature_count": len(FEATURE_NAMES),
        "dtype": "float64",
        "order_authority": "wave56_contextual_gate.FEATURE_NAMES",
    }:
        raise RuntimeError("Wave 60 feature schema drifted")
    scores = score_transport_models(manifest, arrays, inference)
    policies = apply_transport_policies(inference, scores, spec)
    evaluation_index = {
        "pair_token": np.asarray(inference["pair_token"]),
        "primary": np.asarray(inference["primary"], dtype=bool),
        "score_mask": np.asarray(inference["disagreement"], dtype=bool),
        "decision_mask": np.asarray(inference["primary"], dtype=bool)[:, None]
        & np.asarray(inference["disagreement"], dtype=bool),
    }
    write_npz(output / "monitor_scores.npz", scores)
    write_npz(output / "monitor_policy_arrays.npz", policies)
    write_npz(output / "evaluation_index.npz", evaluation_index)
    freeze = {
        "schema_version": SCORE_APPLY_SCHEMA,
        "phase": "score_apply",
        "source_law_freeze_sha256": binding["source_law_freeze_sha256"],
        "inference_bundle_sha256": file_sha256(
            stage / "sealed_monitor_inference_bundle.npz"
        ),
        "scores_sha256": file_sha256(output / "monitor_scores.npz"),
        "policy_arrays_sha256": file_sha256(output / "monitor_policy_arrays.npz"),
        "evaluation_index_sha256": file_sha256(output / "evaluation_index.npz"),
    }
    write_json(output / "monitor_action_freeze.json", freeze)
    if set(policies) != set(expected_policy_array_keys()):
        raise AssertionError("Wave 60 policy output count drifted")
    return "LOCKBOX_ACTIONS_FROZEN"


def run_evaluate(stage: Path, output: Path, request: dict[str, Any]) -> str:
    binding, config = validate_source_binding(stage, request)
    action_freeze = load_json(stage / "monitor_action_freeze.json")
    if (
        set(action_freeze)
        != {
            "schema_version",
            "phase",
            "source_law_freeze_sha256",
            "inference_bundle_sha256",
            "scores_sha256",
            "policy_arrays_sha256",
            "evaluation_index_sha256",
        }
        or action_freeze["schema_version"] != SCORE_APPLY_SCHEMA
        or action_freeze["phase"] != "score_apply"
        or action_freeze["source_law_freeze_sha256"]
        != binding["source_law_freeze_sha256"]
    ):
        raise RuntimeError("Wave 60 action freeze drifted")
    if action_freeze["policy_arrays_sha256"] != file_sha256(
        stage / "monitor_policy_arrays.npz"
    ) or action_freeze["evaluation_index_sha256"] != file_sha256(
        stage / "evaluation_index.npz"
    ):
        raise RuntimeError("Wave 60 action freeze binding failed")
    policy_arrays = load_npz(stage / "monitor_policy_arrays.npz")
    index = load_npz(stage / "evaluation_index.npz")
    if set(policy_arrays) != set(expected_policy_array_keys()) or set(index) != {
        "pair_token",
        "primary",
        "score_mask",
        "decision_mask",
    }:
        raise RuntimeError("Wave 60 pre-truth evaluation inventory drifted")
    primary = np.asarray(index["primary"], dtype=bool)
    score_mask = np.asarray(index["score_mask"], dtype=bool)
    decision_mask = np.asarray(index["decision_mask"], dtype=bool)
    pair_token = np.asarray(index["pair_token"])
    if (
        primary.ndim != 1
        or pair_token.shape != primary.shape
        or score_mask.ndim != 2
        or score_mask.shape[0] != len(primary)
        or decision_mask.shape != score_mask.shape
        or not np.array_equal(decision_mask, primary[:, None] & score_mask)
        or len(np.unique(pair_token.astype(str))) != len(pair_token)
    ):
        raise RuntimeError("Wave 60 pre-truth evaluation index drifted")
    for key, raw in policy_arrays.items():
        array = np.asarray(raw)
        if array.shape != score_mask.shape:
            raise RuntimeError(f"Wave 60 pre-truth policy shape drifted: {key}")
        if key.startswith(("proposal__", "authorized__")):
            if array.dtype != np.bool_ or np.any(array[~decision_mask]):
                raise RuntimeError(f"Wave 60 pre-truth policy mask drifted: {key}")
        elif not np.issubdtype(array.dtype, np.integer):
            raise RuntimeError(f"Wave 60 pre-truth action dtype drifted: {key}")
    hard = np.asarray(policy_arrays["actions__HARD-SET"])
    if any(
        np.any(np.asarray(array)[~decision_mask] != hard[~decision_mask])
        for key, array in policy_arrays.items()
        if key.startswith("actions__")
    ):
        raise RuntimeError("Wave 60 pre-truth nondecision action drifted")
    truth = load_npz(stage / "sealed_monitor_truth_bundle.npz")
    if not np.array_equal(
        index["pair_token"].astype(str), truth["pair_token"].astype(str)
    ):
        raise RuntimeError("Wave 60 truth/evaluation pair-token order drifted")
    if not np.array_equal(index["primary"].astype(bool), truth["primary"].astype(bool)):
        raise RuntimeError("Wave 60 truth/evaluation primary mask drifted")
    utilities = np.load(stage / "utilities.npy", allow_pickle=False)
    analysis, metrics, bootstrap = evaluate_transport_actions(
        truth,
        utilities,
        float(config["penalty"]),
        policy_arrays,
        replicates=int(config["bootstrap"]["replicates"]),
    )
    write_npz(output / "bootstrap_indices.npz", bootstrap)
    write_npz(output / "analysis_arrays.npz", metrics)
    write_json(output / "analysis.json", analysis)
    freeze = {
        "schema_version": EVALUATE_SCHEMA,
        "phase": "evaluate",
        "truth_bundle_sha256": file_sha256(stage / "sealed_monitor_truth_bundle.npz"),
        "action_freeze_sha256": file_sha256(stage / "monitor_action_freeze.json"),
        "policy_arrays_sha256": file_sha256(stage / "monitor_policy_arrays.npz"),
        "evaluation_index_sha256": file_sha256(stage / "evaluation_index.npz"),
        "utilities_sha256": file_sha256(stage / "utilities.npy"),
        "bootstrap_sha256": file_sha256(output / "bootstrap_indices.npz"),
        "analysis_arrays_sha256": file_sha256(output / "analysis_arrays.npz"),
        "analysis_sha256": file_sha256(output / "analysis.json"),
    }
    write_json(output / "evaluation_freeze.json", freeze)
    return "EVALUATED_IMMUTABLE"


def execute(stage: Path, output: Path, phase: str, request: dict[str, Any]) -> str:
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("Wave 60 worker output must start empty")
    output.mkdir(parents=True, exist_ok=True)
    if phase == "verify_source_law":
        return run_verify_source_law(stage, output, request)
    if phase == "score_apply":
        return run_score_apply(stage, output, request)
    if phase == "evaluate":
        return run_evaluate(stage, output, request)
    raise RuntimeError("unknown Wave 60 phase")


def _receipt(
    stage: Path,
    output: Path,
    phase: str,
    status: str,
    forbidden: list[Path],
    started_at: str,
) -> dict[str, Any]:
    capabilities, no_new_privs = _security_state()
    return {
        "schema_version": (
            SOURCE_LAW_SCHEMA
            if phase == "verify_source_law"
            else (SCORE_APPLY_SCHEMA if phase == "score_apply" else EVALUATE_SCHEMA)
        ),
        "phase": phase,
        "status": status,
        "uid": os.geteuid(),
        "gid": os.getegid(),
        "capabilities": capabilities,
        "no_new_privs": no_new_privs,
        "inputs": {
            name: file_sha256(stage / name) for name in sorted(PHASE_FILES[phase])
        },
        "outputs": {
            str(path.relative_to(output)): file_sha256(path)
            for path in sorted(output.rglob("*"))
            if path.is_file()
        },
        "opened_paths": [
            hashlib.sha256(str((stage / name).resolve()).encode()).hexdigest()
            for name in sorted(PHASE_FILES[phase])
        ],
        "denied_path_probes": _probe_forbidden(forbidden),
        "started_at": started_at,
        "completed_at": _now(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--phase",
        choices=("verify_source_law", "score_apply", "evaluate"),
        required=True,
    )
    parser.add_argument("--forbidden-probe", type=Path, action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if os.environ.get("WAVE60_STAGED_RUNTIME") != "1":
        raise RuntimeError("Wave 60 worker requires staged runtime")
    if os.geteuid() != 65534 or os.getegid() != 65534:
        raise RuntimeError("Wave 60 worker must run as nobody/nogroup")
    started_at = _now()
    stage = args.stage.resolve(strict=True)
    output = args.output.resolve()
    request = validate_stage(stage, args.phase)
    status = execute(stage, output, args.phase, request)
    receipt_name = {
        "verify_source_law": "verify_source_law_receipt.json",
        "score_apply": "score_apply_receipt.json",
        "evaluate": "evaluate_receipt.json",
    }[args.phase]
    write_json(
        output / receipt_name,
        _receipt(stage, output, args.phase, status, args.forbidden_probe, started_at),
    )
    print(json.dumps({"phase": args.phase, "status": status}, sort_keys=True))


if __name__ == "__main__":
    main()
