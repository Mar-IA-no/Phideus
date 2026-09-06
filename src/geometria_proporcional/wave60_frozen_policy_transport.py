"""Frozen-law primitives for the prospective Wave 60 transport experiment.

The module is deliberately CPU-only and contains no benchmark discovery.  It
projects the accepted Wave 59 HGB states, applies their frozen thresholds to an
inference-safe bundle, and evaluates the already-frozen actions after truth is
made available by the coordinator.
"""

from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .wave56_contextual_gate import FEATURE_NAMES
from .wave58_open_diagnostic import score_grid
from .wave59_hgb_guard_bracket import (
    HARM_CONTROL_SEEDS,
    INCOMPATIBILITY_CONTROL_SEEDS,
    aggregate_ternary,
    apply_calibrated_policies,
    evaluate_actions,
    mean_control_metric,
    model_id,
    ordered_primary_metric,
    primary_tokens,
    score_true_models,
    support,
    validate_inference_safe_view,
)

SCHEMA_VERSION = "wave60-frozen-policy-transport-v1"
SOURCE_LAW_SCHEMA = "wave60-source-law-v1"
SCORE_APPLY_SCHEMA = "wave60-score-apply-v1"
EVALUATE_SCHEMA = "wave60-evaluate-v1"
SOURCE_BINDING_SCHEMA = "wave60-source-binding-v1"
BOOTSTRAP_SEED = 6007
BOOTSTRAP_REPLICATES = 5000
WAVE59_SOURCE_COMMIT = "8edc23d1120a91981e01aeb8c385c344da42b3fb"
PLAN_COMMIT = "f8bd1d656587875e5b50a8c1ab33b32181eb5af5"
PLAN_SHA256 = "4edaf638ab73191bf51d35def8c1ef298f400086c1ad45783f73318352c8d459"
PLAN_AUDIT_COMMIT = "a28a077db78fe68ff98f1c35e68333dd967ddcf5"
PLAN_AUDIT_SHA256 = "2fd21ae36db087614412e19387e18b15fadbecf43141d8db0248724a621246f1"
MAIN_POLICIES = {
    "mean": "P-HGB-HGB-INCOMPATIBILITY-Q90",
    "tail": "P-HGB-HGB-HARM-Q70",
}
PROPOSER_MODEL = "proposer-hgb"
MAIN_GUARD_MODELS = {
    MAIN_POLICIES["mean"]: "guard-hgb-incompatibility",
    MAIN_POLICIES["tail"]: "guard-hgb-harm",
}
HARM_CONTROLS = tuple(
    model_id("control-hgb", "harm", seed) for seed in HARM_CONTROL_SEEDS
)
INCOMPATIBILITY_CONTROLS = tuple(
    model_id("control-hgb", "posterior_incompatibility", seed)
    for seed in INCOMPATIBILITY_CONTROL_SEEDS
)
CONTROL_MODELS = HARM_CONTROLS + INCOMPATIBILITY_CONTROLS
USED_MODELS = (PROPOSER_MODEL, *MAIN_GUARD_MODELS.values(), *CONTROL_MODELS)
UNUSED_MODELS = (
    "proposer-ridge",
    "guard-logistic-harm",
    "guard-logistic-incompatibility",
)
REFERENCE_ACTIONS = ("HARD-SET", "HGB-PROPOSER-ONLY")
FROZEN_THRESHOLDS = {
    PROPOSER_MODEL: 0.3352357782028221,
    MAIN_POLICIES["mean"]: 0.05147286376407591,
    MAIN_POLICIES["tail"]: 0.4714312385695055,
    "control-hgb-harm-59031": 0.774991519974731,
    "control-hgb-harm-59032": 0.8006289805270264,
    "control-hgb-harm-59033": 0.7988491561377532,
    "control-hgb-harm-59034": 0.7855736564669745,
    "control-hgb-harm-59035": 0.8026326479327117,
    "control-hgb-incompatibility-59041": 0.06306052990071899,
    "control-hgb-incompatibility-59042": 0.052948173822232034,
    "control-hgb-incompatibility-59043": 0.09258428157572807,
    "control-hgb-incompatibility-59044": 0.08564156818348924,
    "control-hgb-incompatibility-59045": 0.0921077041003903,
}
TRANSPORT_ACTION_IDS = (
    *REFERENCE_ACTIONS,
    MAIN_POLICIES["mean"],
    MAIN_POLICIES["tail"],
    *CONTROL_MODELS,
)
SOURCE_HASHES = {
    "wave59_fit_freeze.json": "1cd2807970a8c8f6b2d0e88abf212c4a0ba797ed67e9f36cb115a90fe941db8a",
    "wave59_model_states_manifest.json": "942651e39c69a65378185de60fdbf770b2df221e62eaf6c6a614055cb719b5c7",
    "wave59_model_state_arrays.npz": "6539ac1fed2f2d030ba8c8dcb0b76c64aed2a76795f63cff245e1f16e371b3a4",
    "wave59_calibration_freeze.json": "6ad2c2c9512c206e0982bede3b0dfaee694187292fb1badeaecb69cd3451367a",
    "wave59_monitor_scores.npz": "61e691babb482360e5e4ea4dd112e3c3c16c9a8e25c0670e2f618db46438d787",
    "wave59_monitor_policy_arrays.npz": "e44fe6d3a0ecc5510c814703ad9eced822c76e7c3bef7a0f827a3a9055fcfb74",
    "wave59_artifact_manifest.json": "909361b45e9fb51977229063afcdf9587144bfb80280fd3c70dda86de19a6371",
    "wave59_config_snapshot.json": "f6edfd2106fe87c8150562d096469e29b64a108a73de2dae0d371bd689a4a9b6",
    "r454_audit.md": "e320d48c1c1b198fd2e324a0df7d534c088f417502d1bb20167c4050a29fe926",
}
SOURCE_PHASE_FILES = frozenset(
    {
        "source_law_request.json",
        "wave59_fit_freeze.json",
        "wave59_model_states_manifest.json",
        "wave59_model_state_arrays.npz",
        "wave59_calibration_freeze.json",
        "wave59_monitor_inference_bundle.npz",
        "wave59_monitor_scores.npz",
        "wave59_monitor_policy_arrays.npz",
        "wave59_monitor_action_freeze.json",
        "wave59_artifact_manifest.json",
        "wave59_config_snapshot.json",
        "r454_audit.md",
    }
)
PHASE_FILES = {
    "verify_source_law": SOURCE_PHASE_FILES,
    "score_apply": frozenset(
        {
            "phase_request.json",
            "config.snapshot.json",
            "source_bindings.json",
            "transport_law_manifest.json",
            "transport_law_arrays.npz",
            "frozen_policy_spec.json",
            "feature_schema.json",
            "sealed_monitor_inference_bundle.npz",
        }
    ),
    "evaluate": frozenset(
        {
            "phase_request.json",
            "config.snapshot.json",
            "source_bindings.json",
            "evaluation_index.npz",
            "monitor_policy_arrays.npz",
            "monitor_action_freeze.json",
            "sealed_monitor_truth_bundle.npz",
            "utilities.npy",
        }
    ),
}
HEX64 = re.compile(r"[0-9a-f]{64}")
METRICS = ("accuracy", "compatible", "regret", "worst_regret")


def canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def config_self_binding_sha256(config: Mapping[str, Any], relative_path: str) -> str:
    normalized = deepcopy(dict(config))
    hashes = normalized.get("source_sha256")
    if not isinstance(hashes, dict) or relative_path not in hashes:
        raise RuntimeError("Wave 60 config self-binding entry is absent")
    hashes[relative_path] = "0" * 64
    return canonical_json_sha256(normalized)


def require_exact_keys(
    payload: Any, expected: Iterable[str], label: str
) -> dict[str, Any]:
    wanted = set(expected)
    if not isinstance(payload, dict) or set(payload) != wanted:
        observed = (
            sorted(payload) if isinstance(payload, dict) else type(payload).__name__
        )
        raise RuntimeError(
            f"Wave 60 {label} keys drifted: expected={sorted(wanted)}, observed={observed}"
        )
    return payload


def require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or HEX64.fullmatch(value) is None:
        raise RuntimeError(f"Wave 60 {label} is not a lowercase SHA-256 digest")
    return value


def _array_equal(left: np.ndarray, right: np.ndarray) -> bool:
    a = np.asarray(left)
    b = np.asarray(right)
    return (
        a.dtype == b.dtype
        and a.shape == b.shape
        and np.array_equal(a, b, equal_nan=True)
    )


def expected_authorized_ids() -> tuple[str, ...]:
    return (MAIN_POLICIES["mean"], MAIN_POLICIES["tail"], *CONTROL_MODELS)


def expected_policy_array_keys() -> frozenset[str]:
    return frozenset(
        {
            "proposal__hgb",
            *(f"authorized__{identifier}" for identifier in expected_authorized_ids()),
            *(f"actions__{identifier}" for identifier in TRANSPORT_ACTION_IDS),
        }
    )


def selected_wave59_array_keys() -> frozenset[str]:
    """The 26 selected arrays, excluding the separately checked hard reference."""
    return expected_policy_array_keys() - {"actions__HARD-SET"}


def validate_source_model_manifest(
    manifest: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    require_exact_keys(
        manifest, {"models", "portable_states"}, "Wave 59 model manifest"
    )
    models = manifest["models"]
    states = manifest["portable_states"]
    expected = set(USED_MODELS) | set(UNUSED_MODELS)
    if not isinstance(models, dict) or set(models) != expected:
        raise RuntimeError("Wave 59 model record roster drifted")
    if not isinstance(states, dict) or set(states) != expected:
        raise RuntimeError("Wave 59 portable-state roster drifted")
    used: dict[str, dict[str, Any]] = {}
    seen_tree_keys: set[str] = set()
    for identifier in USED_MODELS:
        state = states[identifier]
        if not isinstance(state, dict) or state.get("status") != "PASS":
            raise RuntimeError(f"transport state is not PASS: {identifier}")
        expected_kind = (
            "hgb_regressor" if identifier == PROPOSER_MODEL else "hgb_classifier"
        )
        if (
            state.get("kind") != expected_kind
            or state.get("transport_only") is not True
        ):
            raise RuntimeError(f"transport state kind drifted: {identifier}")
        if int(state.get("n_features", -1)) != len(FEATURE_NAMES):
            raise RuntimeError(f"transport feature count drifted: {identifier}")
        keys = state.get("tree_keys")
        if (
            not isinstance(keys, list)
            or len(keys) != 100
            or len(set(keys)) != len(keys)
        ):
            raise RuntimeError(f"transport tree roster drifted: {identifier}")
        overlap = seen_tree_keys.intersection(keys)
        if overlap:
            raise RuntimeError(f"tree key shared by models: {sorted(overlap)[:1]}")
        seen_tree_keys.update(keys)
        used[identifier] = state
    if len(seen_tree_keys) != 1300:
        raise RuntimeError("transport tree-key count drifted")
    return used


def project_transport_law(
    source_manifest: Mapping[str, Any], source_arrays: Mapping[str, np.ndarray]
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    states = validate_source_model_manifest(source_manifest)
    tree_keys = [
        key for identifier in USED_MODELS for key in states[identifier]["tree_keys"]
    ]
    expected_arrays = {
        suffix for key in tree_keys for suffix in (key, f"{key}__binned", f"{key}__raw")
    }
    missing = expected_arrays - set(source_arrays)
    if missing:
        raise RuntimeError(f"transport state arrays missing: {sorted(missing)[:1]}")
    projected = {
        key: np.asarray(source_arrays[key]).copy() for key in sorted(expected_arrays)
    }
    for key in tree_keys:
        nodes = projected[key]
        if nodes.dtype.names is None or "is_categorical" not in nodes.dtype.names:
            raise RuntimeError(f"HGB node dtype drifted: {key}")
        if np.any(nodes["is_categorical"]):
            raise RuntimeError(f"categorical split is not transportable: {key}")
    manifest = {
        "schema_version": SOURCE_LAW_SCHEMA,
        "used_models": list(USED_MODELS),
        "unused_models": list(UNUSED_MODELS),
        "model_states": {identifier: states[identifier] for identifier in USED_MODELS},
        "array_keys": sorted(projected),
        "feature_names": list(FEATURE_NAMES),
        "source_bindings": {
            "wave59_model_states_manifest_sha256": SOURCE_HASHES[
                "wave59_model_states_manifest.json"
            ],
            "wave59_model_state_arrays_sha256": SOURCE_HASHES[
                "wave59_model_state_arrays.npz"
            ],
        },
    }
    validate_transport_manifest(manifest, projected)
    return manifest, projected


def validate_transport_manifest(
    manifest: Mapping[str, Any], arrays: Mapping[str, np.ndarray]
) -> dict[str, dict[str, Any]]:
    require_exact_keys(
        manifest,
        {
            "schema_version",
            "used_models",
            "unused_models",
            "model_states",
            "array_keys",
            "feature_names",
            "source_bindings",
        },
        "transport manifest",
    )
    if manifest["schema_version"] != SOURCE_LAW_SCHEMA:
        raise RuntimeError("transport manifest schema drifted")
    if manifest["used_models"] != list(USED_MODELS) or manifest[
        "unused_models"
    ] != list(UNUSED_MODELS):
        raise RuntimeError("transport manifest roster drifted")
    states = manifest["model_states"]
    if not isinstance(states, dict) or set(states) != set(USED_MODELS):
        raise RuntimeError("transport manifest state set drifted")
    if manifest["feature_names"] != list(FEATURE_NAMES):
        raise RuntimeError("transport feature order drifted")
    keys = manifest["array_keys"]
    if not isinstance(keys, list) or keys != sorted(keys) or len(keys) != 3900:
        raise RuntimeError("transport array-key inventory drifted")
    if set(arrays) != set(keys):
        raise RuntimeError("transport arrays are not closed-world")
    seen: set[str] = set()
    for identifier in USED_MODELS:
        state = states[identifier]
        if state.get("status") != "PASS" or int(state.get("n_features", -1)) != len(
            FEATURE_NAMES
        ):
            raise RuntimeError(f"transport state drifted: {identifier}")
        tree_keys = state.get("tree_keys")
        if not isinstance(tree_keys, list) or len(tree_keys) != 100:
            raise RuntimeError(f"transport tree roster drifted: {identifier}")
        if seen.intersection(tree_keys):
            raise RuntimeError("transport tree key is shared")
        seen.update(tree_keys)
        for key in tree_keys:
            if not {key, f"{key}__binned", f"{key}__raw"}.issubset(arrays):
                raise RuntimeError(f"transport tree arrays missing: {key}")
            nodes = np.asarray(arrays[key])
            if nodes.dtype.names is None or np.any(nodes["is_categorical"]):
                raise RuntimeError(f"transport tree is categorical or malformed: {key}")
    if len(seen) != 1300:
        raise RuntimeError("transport tree count drifted")
    return {identifier: states[identifier] for identifier in USED_MODELS}


def frozen_policy_spec(calibration_freeze: Mapping[str, Any]) -> dict[str, Any]:
    calibration = calibration_freeze.get("calibration")
    if not isinstance(calibration, dict):
        raise RuntimeError("Wave 59 calibration payload is absent")
    proposer = calibration.get("proposers", {}).get("hgb")
    if not isinstance(proposer, dict) or proposer.get("model_id") != PROPOSER_MODEL:
        raise RuntimeError("frozen proposer calibration drifted")
    main: dict[str, Any] = {}
    for role, identifier in MAIN_POLICIES.items():
        row = calibration.get("policies", {}).get(identifier)
        if not isinstance(row, dict) or row.get("proposer_model_id") != PROPOSER_MODEL:
            raise RuntimeError(f"frozen main policy drifted: {identifier}")
        if row.get("guard_model_id") != MAIN_GUARD_MODELS[identifier]:
            raise RuntimeError(f"frozen main guard drifted: {identifier}")
        main[identifier] = {
            "role": role,
            "guard_model_id": row["guard_model_id"],
            "threshold": float(row["guard_threshold"]),
            "source_q_guard": float(row["q_guard"]),
        }
    controls: dict[str, Any] = {}
    source_controls = calibration.get("controls")
    if not isinstance(source_controls, dict) or set(source_controls) != set(
        CONTROL_MODELS
    ):
        raise RuntimeError("frozen control calibration roster drifted")
    for identifier in CONTROL_MODELS:
        row = source_controls[identifier]
        controls[identifier] = {
            "guard_model_id": identifier,
            "target": row["target"],
            "mapping_seed": int(row["mapping_seed"]),
            "threshold": float(row["guard_threshold"]),
            "source_q_guard": float(row["q_guard"]),
        }
    thresholds = {
        PROPOSER_MODEL: float(proposer["threshold"]),
        **{identifier: row["threshold"] for identifier, row in main.items()},
        **{identifier: row["threshold"] for identifier, row in controls.items()},
    }
    spec = {
        "schema_version": SOURCE_LAW_SCHEMA,
        "proposer": {
            "model_id": PROPOSER_MODEL,
            "threshold": thresholds[PROPOSER_MODEL],
        },
        "main_policies": main,
        "controls": controls,
        "references": list(REFERENCE_ACTIONS),
        "thresholds": thresholds,
        "comparison_operators": {"proposer": ">", "guards": "<"},
        "score_mask": "disagreement",
        "decision_mask": "primary AND disagreement",
    }
    validate_frozen_policy_spec(spec)
    return spec


def validate_frozen_policy_spec(spec: Mapping[str, Any]) -> None:
    require_exact_keys(
        spec,
        {
            "schema_version",
            "proposer",
            "main_policies",
            "controls",
            "references",
            "thresholds",
            "comparison_operators",
            "score_mask",
            "decision_mask",
        },
        "frozen policy spec",
    )
    if spec["schema_version"] != SOURCE_LAW_SCHEMA:
        raise RuntimeError("frozen policy schema drifted")
    if spec["references"] != list(REFERENCE_ACTIONS):
        raise RuntimeError("frozen references drifted")
    if spec["comparison_operators"] != {"proposer": ">", "guards": "<"}:
        raise RuntimeError("frozen comparison operators drifted")
    if spec["score_mask"] != "disagreement" or spec["decision_mask"] != (
        "primary AND disagreement"
    ):
        raise RuntimeError("frozen mask semantics drifted")
    proposer = spec["proposer"]
    if (
        set(proposer) != {"model_id", "threshold"}
        or proposer["model_id"] != PROPOSER_MODEL
    ):
        raise RuntimeError("frozen proposer spec drifted")
    if set(spec["main_policies"]) != set(MAIN_POLICIES.values()):
        raise RuntimeError("frozen main-policy roster drifted")
    if set(spec["controls"]) != set(CONTROL_MODELS):
        raise RuntimeError("frozen control roster drifted")
    expected_threshold_ids = {PROPOSER_MODEL, *MAIN_POLICIES.values(), *CONTROL_MODELS}
    if set(spec["thresholds"]) != expected_threshold_ids:
        raise RuntimeError("frozen threshold roster drifted")
    if spec["thresholds"] != FROZEN_THRESHOLDS:
        raise RuntimeError("frozen threshold values drifted")
    if float(proposer["threshold"]) != FROZEN_THRESHOLDS[PROPOSER_MODEL]:
        raise RuntimeError("frozen proposer threshold drifted")
    for identifier, value in spec["thresholds"].items():
        if not np.isfinite(float(value)):
            raise RuntimeError(f"frozen threshold is not finite: {identifier}")
    for role, identifier in MAIN_POLICIES.items():
        row = spec["main_policies"][identifier]
        if set(row) != {"role", "guard_model_id", "threshold", "source_q_guard"}:
            raise RuntimeError(f"frozen main policy shape drifted: {identifier}")
        expected_q = 0.9 if role == "mean" else 0.7
        if row != {
            "role": role,
            "guard_model_id": MAIN_GUARD_MODELS[identifier],
            "threshold": FROZEN_THRESHOLDS[identifier],
            "source_q_guard": expected_q,
        }:
            raise RuntimeError(f"frozen main policy values drifted: {identifier}")
    for identifier in CONTROL_MODELS:
        row = spec["controls"][identifier]
        target = "harm" if identifier in HARM_CONTROLS else "posterior_incompatibility"
        seed = int(identifier.rsplit("-", 1)[-1])
        expected_q = 0.7 if target == "harm" else 0.9
        if row != {
            "guard_model_id": identifier,
            "target": target,
            "mapping_seed": seed,
            "threshold": FROZEN_THRESHOLDS[identifier],
            "source_q_guard": expected_q,
        }:
            raise RuntimeError(f"frozen control values drifted: {identifier}")


def score_transport_models(
    manifest: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    data: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    validate_inference_safe_view(dict(data))
    states = validate_transport_manifest(manifest, arrays)
    result = {
        identifier: score_grid(states[identifier], dict(arrays), dict(data))
        for identifier in USED_MODELS
    }
    if set(result) != set(USED_MODELS):
        raise AssertionError("transport score roster drifted")
    disagreement = np.asarray(data["disagreement"], dtype=bool)
    for identifier, values in result.items():
        score = np.asarray(values, dtype=np.float64)
        if score.shape != disagreement.shape:
            raise RuntimeError(f"transport score shape drifted: {identifier}")
        if not np.all(np.isfinite(score[disagreement])) or not np.all(
            np.isnan(score[~disagreement])
        ):
            raise RuntimeError(f"transport score mask drifted: {identifier}")
    return result


def apply_transport_policies(
    data: Mapping[str, np.ndarray],
    scores: Mapping[str, np.ndarray],
    spec: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    validate_inference_safe_view(dict(data))
    validate_frozen_policy_spec(spec)
    if set(scores) != set(USED_MODELS):
        raise RuntimeError("transport scorer input is not closed-world")
    disagreement = np.asarray(data["disagreement"], dtype=bool)
    primary = np.asarray(data["primary"], dtype=bool)
    hard = np.asarray(data["hard_actions"], dtype=np.int64)
    posterior = np.asarray(data["posterior_actions"], dtype=np.int64)
    for identifier, raw in scores.items():
        values = np.asarray(raw, dtype=np.float64)
        if values.shape != disagreement.shape:
            raise RuntimeError(f"transport score shape drifted: {identifier}")
        if not np.all(np.isfinite(values[disagreement])) or not np.all(
            np.isnan(values[~disagreement])
        ):
            raise RuntimeError(f"transport score mask drifted: {identifier}")
    decision = primary[:, None] & disagreement
    proposer_score = np.asarray(scores[PROPOSER_MODEL], dtype=np.float64)
    proposal = decision & (proposer_score > float(spec["thresholds"][PROPOSER_MODEL]))
    result: dict[str, np.ndarray] = {
        "proposal__hgb": proposal,
        "actions__HARD-SET": hard.copy(),
        "actions__HGB-PROPOSER-ONLY": np.where(proposal, posterior, hard).astype(
            np.int64
        ),
    }
    guard_map = {
        identifier: row["guard_model_id"]
        for identifier, row in spec["main_policies"].items()
    }
    guard_map.update(
        {
            identifier: row["guard_model_id"]
            for identifier, row in spec["controls"].items()
        }
    )
    for identifier in expected_authorized_ids():
        guard_score = np.asarray(scores[guard_map[identifier]], dtype=np.float64)
        authorized = proposal & (guard_score < float(spec["thresholds"][identifier]))
        result[f"authorized__{identifier}"] = authorized
        result[f"actions__{identifier}"] = np.where(authorized, posterior, hard).astype(
            np.int64
        )
    if set(result) != expected_policy_array_keys():
        raise AssertionError("transport policy output roster drifted")
    for key, value in result.items():
        array = np.asarray(value)
        if array.shape != disagreement.shape:
            raise RuntimeError(f"transport output shape drifted: {key}")
        if key.startswith(("proposal__", "authorized__")) and np.any(array[~decision]):
            raise RuntimeError(f"decision mask escaped: {key}")
        if key.startswith("actions__") and np.any(array[~decision] != hard[~decision]):
            raise RuntimeError(f"nondecision action differs from hard: {key}")
    return result


def retrospective_source_verification(
    source_manifest: Mapping[str, Any],
    source_arrays: Mapping[str, np.ndarray],
    calibration_freeze: Mapping[str, Any],
    inference_data: Mapping[str, np.ndarray],
    preserved_scores: Mapping[str, np.ndarray],
    preserved_policy_arrays: Mapping[str, np.ndarray],
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    transport_manifest, transport_arrays = project_transport_law(
        source_manifest, source_arrays
    )
    if set(preserved_scores) != set(USED_MODELS) | set(UNUSED_MODELS):
        raise RuntimeError("preserved Wave 59 score roster drifted")
    full_scores = score_true_models(
        dict(source_manifest["portable_states"]),
        dict(source_arrays),
        dict(inference_data),
    )
    for identifier in sorted(preserved_scores):
        if not _array_equal(full_scores[identifier], preserved_scores[identifier]):
            raise RuntimeError(
                f"Wave 59 portable score reproduction failed: {identifier}"
            )
    spec = frozen_policy_spec(calibration_freeze)
    transport_scores = score_transport_models(
        transport_manifest, transport_arrays, inference_data
    )
    for identifier in USED_MODELS:
        if not _array_equal(transport_scores[identifier], preserved_scores[identifier]):
            raise RuntimeError(f"transport score reproduction failed: {identifier}")
    full_policy = apply_calibrated_policies(
        dict(inference_data), full_scores, calibration_freeze["calibration"]
    )
    for key, expected in preserved_policy_arrays.items():
        if key not in full_policy or not _array_equal(full_policy[key], expected):
            raise RuntimeError(f"Wave 59 full policy reproduction failed: {key}")
    transported = apply_transport_policies(inference_data, transport_scores, spec)
    checked = sorted(selected_wave59_array_keys() | {"actions__HARD-SET"})
    for key in checked:
        if key not in preserved_policy_arrays or not _array_equal(
            transported[key], preserved_policy_arrays[key]
        ):
            raise RuntimeError(f"transport policy reproduction failed: {key}")
    diagnostics = {
        "full_model_scores_exact": len(full_scores),
        "transport_model_scores_exact": len(transport_scores),
        "selected_policy_arrays_exact": len(selected_wave59_array_keys()),
        "hard_reference_exact": True,
        "tree_keys": 1300,
        "transport_arrays": 3900,
        "score_mask": "disagreement",
        "decision_mask": "primary AND disagreement",
    }
    return (
        transport_manifest,
        transport_arrays,
        {"spec": spec, "diagnostics": diagnostics},
    )


def wave60_bootstrap_indices(
    pair_token: np.ndarray, replicates: int = BOOTSTRAP_REPLICATES
) -> np.ndarray:
    tokens = np.sort(np.asarray(pair_token).astype(str))
    if not len(tokens) or len(np.unique(tokens)) != len(tokens):
        raise ValueError("bootstrap needs unique non-empty T_primary")
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    return rng.integers(
        0, len(tokens), size=(int(replicates), len(tokens)), dtype=np.int64
    )


def _delta(
    data: Mapping[str, np.ndarray],
    left: np.ndarray,
    right: np.ndarray,
    indices: np.ndarray,
) -> dict[str, float]:
    tokens_left, values_left = ordered_primary_metric(dict(data), left)
    tokens_right, values_right = ordered_primary_metric(dict(data), right)
    np.testing.assert_array_equal(tokens_left, tokens_right)
    difference = values_left - values_right
    sampled = difference[np.asarray(indices, dtype=np.int64)].mean(axis=1)
    low, high = np.percentile(sampled, [2.5, 97.5])
    return {
        "mean_diff": float(difference.mean()),
        "ci95_low": float(low),
        "ci95_high": float(high),
    }


def evaluate_transport_actions(
    data: Mapping[str, np.ndarray],
    utilities: np.ndarray,
    penalty: float,
    policy_arrays: Mapping[str, np.ndarray],
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, np.ndarray]]:
    if set(policy_arrays) != expected_policy_array_keys():
        raise RuntimeError("evaluation policy arrays are not closed-world")
    summaries, raw_metric_arrays = evaluate_actions(
        dict(data), np.asarray(utilities), float(penalty), dict(policy_arrays)
    )
    metric_arrays = {
        key: value
        for key, value in raw_metric_arrays.items()
        if key.rsplit("__", 1)[-1] in METRICS
    }
    if set(summaries) != set(TRANSPORT_ACTION_IDS):
        raise RuntimeError("evaluation action roster drifted")
    tokens = primary_tokens(dict(data))
    indices = wave60_bootstrap_indices(tokens, replicates)
    deltas: dict[str, Any] = {}
    for role, identifier in MAIN_POLICIES.items():
        deltas[f"{role}_vs_hard"] = {}
        deltas[f"{role}_vs_hgb_proposer_only"] = {}
        for metric in METRICS:
            left = metric_arrays[f"metric__{identifier}__{metric}"]
            deltas[f"{role}_vs_hard"][metric] = _delta(
                data, left, metric_arrays[f"metric__HARD-SET__{metric}"], indices
            )
            deltas[f"{role}_vs_hgb_proposer_only"][metric] = _delta(
                data,
                left,
                metric_arrays[f"metric__HGB-PROPOSER-ONLY__{metric}"],
                indices,
            )
    control_deltas: dict[str, Any] = {}
    for target, ids, main, metric in (
        (
            "posterior_incompatibility",
            INCOMPATIBILITY_CONTROLS,
            MAIN_POLICIES["mean"],
            "regret",
        ),
        ("harm", HARM_CONTROLS, MAIN_POLICIES["tail"], "worst_regret"),
    ):
        mean_control = mean_control_metric(metric_arrays, ids, metric)
        control_deltas[target] = _delta(
            data, metric_arrays[f"metric__{main}__{metric}"], mean_control, indices
        )
    supports = {
        identifier: support(policy_arrays[f"authorized__{identifier}"], data["primary"])
        for identifier in expected_authorized_ids()
    }
    mean_hard = deltas["mean_vs_hard"]
    tail_hard = deltas["tail_vs_hard"]
    mean_support = supports[MAIN_POLICIES["mean"]]["authorized_pair_tokens"] >= 25
    tail_support = supports[MAIN_POLICIES["tail"]]["authorized_pair_tokens"] >= 25
    mean_conditions: dict[str, bool] = {
        "authorized_pair_tokens_at_least_25": mean_support,
        "accuracy_ci95_low_at_least_negative_0_01": mean_hard["accuracy"]["ci95_low"]
        >= -0.01,
        "compatibility_ci95_low_at_least_zero": mean_hard["compatible"]["ci95_low"]
        >= 0.0,
        "regret_mean_at_most_negative_0_005": mean_hard["regret"]["mean_diff"]
        <= -0.005,
        "regret_ci95_high_below_zero": mean_hard["regret"]["ci95_high"] < 0.0,
        "worst_regret_ci95_high_at_most_0_01": mean_hard["worst_regret"]["ci95_high"]
        <= 0.01,
        "regret_vs_mean_max_displacement_ci95_high_below_zero": control_deltas[
            "posterior_incompatibility"
        ]["ci95_high"]
        < 0.0,
    }
    tail_conditions: dict[str, bool] = {
        "authorized_pair_tokens_at_least_25": tail_support,
        "accuracy_ci95_low_at_least_negative_0_01": tail_hard["accuracy"]["ci95_low"]
        >= -0.01,
        "compatibility_ci95_low_at_least_zero": tail_hard["compatible"]["ci95_low"]
        >= 0.0,
        "regret_ci95_high_at_most_zero": tail_hard["regret"]["ci95_high"] <= 0.0,
        "worst_regret_mean_at_most_negative_0_01": tail_hard["worst_regret"][
            "mean_diff"
        ]
        <= -0.01,
        "worst_regret_ci95_high_below_zero": tail_hard["worst_regret"]["ci95_high"]
        < 0.0,
        "worst_regret_vs_mean_max_displacement_ci95_high_below_zero": control_deltas[
            "harm"
        ]["ci95_high"]
        < 0.0,
    }
    patterns: dict[str, bool | str] = {
        "incompatibility": (
            aggregate_ternary(mean_conditions.values())
            if mean_support
            else "NOT_EVALUABLE"
        ),
        "harm": (
            aggregate_ternary(tail_conditions.values())
            if tail_support
            else "NOT_EVALUABLE"
        ),
    }
    diagnostics = {
        "score_transport_scope": "CONDITIONAL_ON_NEW_DRAW_AND_FROZEN_W59_LAW",
        "supports": supports,
        "control_deltas": control_deltas,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": int(replicates),
    }
    analysis = {
        "schema_version": EVALUATE_SCHEMA,
        "status": "COMPLETE",
        "estimand": "frozen_W59_policy_minus_fresh_hard_and_frozen_control_mean",
        "population": {"primary_pair_tokens": int(len(tokens)), "unit": "pair_token"},
        "policies": dict(MAIN_POLICIES),
        "controls": {
            "posterior_incompatibility": list(INCOMPATIBILITY_CONTROLS),
            "harm": list(HARM_CONTROLS),
        },
        "references": list(REFERENCE_ACTIONS),
        "metrics": summaries,
        "deltas": deltas,
        "intervals": "paired_PC64_pair_token_percentile_95",
        "core_conditions": {
            "incompatibility": mean_conditions,
            "harm": tail_conditions,
        },
        "core_patterns": patterns,
        "diagnostics": diagnostics,
        "scientific_decision": None,
        "decision_authority": "user",
        "limitations": [
            "synthetic_generator_only",
            "pipeline_transport_does_not_identify_target_effect",
            "conditional_intervals_without_multiplicity_correction",
            "replay_exact_pending_pair_finalize",
        ],
    }
    require_exact_keys(
        analysis,
        {
            "schema_version",
            "status",
            "estimand",
            "population",
            "policies",
            "controls",
            "references",
            "metrics",
            "deltas",
            "intervals",
            "core_conditions",
            "core_patterns",
            "diagnostics",
            "scientific_decision",
            "decision_authority",
            "limitations",
        },
        "analysis",
    )
    bootstrap = {"indices": indices, "pair_token": tokens}
    return analysis, metric_arrays, bootstrap


def finalize_patterns(
    analysis: Mapping[str, Any], replay_exact: bool
) -> tuple[dict[str, Any], dict[str, Any]]:
    conditions: dict[str, Any] = {}
    patterns: dict[str, Any] = {}
    for name in ("incompatibility", "harm"):
        local = dict(analysis["core_conditions"][name])
        local["replay_exact"] = bool(replay_exact)
        conditions[name] = local
        if analysis["core_patterns"][name] == "NOT_EVALUABLE":
            patterns[name] = "NOT_EVALUABLE"
        else:
            patterns[name] = aggregate_ternary(local.values())
    return conditions, patterns


def validate_pre_draw_config(config: Mapping[str, Any]) -> None:
    required = {
        "schema_version",
        "status",
        "device",
        "cpu_threads",
        "penalty",
        "bootstrap",
        "runtime_budget",
        "main_policies",
        "source_law_authority",
        "attempt",
        "implementation_binding",
        "final_audit",
        "plan_binding",
        "fresh_benchmark",
        "physical_splits",
        "seeds",
        "inference_batch_size",
        "feature_names",
        "source_binding",
        "required_execution_sources",
        "source_sha256",
        "primary_output",
        "primary_output_name",
        "replay_output",
        "replay_output_name",
        "output_parent_relative",
    }
    require_exact_keys(config, required, "config")
    if config["schema_version"] != SCHEMA_VERSION or config["status"] != (
        "FROZEN_PROSPECTIVE_PROTOCOL_PRE_KEY_DRAW"
    ):
        raise RuntimeError("Wave 60 config identity drifted")
    if config["device"] != "cpu" or int(config["cpu_threads"]) != 4:
        raise RuntimeError("Wave 60 CPU contract drifted")
    if float(config["penalty"]) != 1.25:
        raise RuntimeError("Wave 60 penalty drifted")
    if config["main_policies"] != MAIN_POLICIES:
        raise RuntimeError("Wave 60 main policies drifted")
    if config["feature_names"] != list(FEATURE_NAMES):
        raise RuntimeError("Wave 60 feature schema drifted")
    if config["physical_splits"] != {
        "train": "unused_train",
        "val": "unused_validation",
        "lockbox": "sealed_monitor",
    }:
        raise RuntimeError("Wave 60 physical split contract drifted")
    if config["seeds"] != [17, 29, 43] or int(config["inference_batch_size"]) != 256:
        raise RuntimeError("Wave 60 inference contract drifted")
    fresh = config["fresh_benchmark"]
    if fresh != {
        "protocol": "wave49-relational-benchmark-v2",
        "expected_visible_fixtures_per_split": 4992,
        "expected_eligible_pair_tokens_per_split": 768,
        "pair_token_count_basis": "eligible_unique_pair_tokens",
        "no_redraw_after_escrow": True,
        "sealed_directory_mode": "0700",
        "escrow_file_mode": "0600",
        "inference_uid": 65534,
        "inference_gid": 65534,
        "inference_user": "nobody",
        "staging_parent": "/tmp",
    }:
        raise RuntimeError("Wave 60 fresh benchmark contract drifted")
    plan = config["plan_binding"]
    if plan != {
        "commit": PLAN_COMMIT,
        "sha256": PLAN_SHA256,
        "audit_commit": PLAN_AUDIT_COMMIT,
        "audit_path": (
            "Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/"
            "463_wave60_plan_final_focal_pass.md"
        ),
        "audit_sha256": PLAN_AUDIT_SHA256,
    }:
        raise RuntimeError("Wave 60 plan authority drifted")
    implementation = config["implementation_binding"]
    if not isinstance(implementation, dict) or set(implementation) != {
        "status",
        "commit",
        "audit_commit",
        "audit_path",
        "audit_sha256",
    }:
        raise RuntimeError("Wave 60 implementation authority drifted")
    if implementation["status"] != "ACCEPTED_IMPLEMENTATION_AUDIT":
        raise RuntimeError("Wave 60 implementation is not accepted")
    for field in ("commit", "audit_commit"):
        if (
            not isinstance(implementation[field], str)
            or re.fullmatch(r"[0-9a-f]{40}", implementation[field]) is None
        ):
            raise RuntimeError(f"Wave 60 implementation {field} drifted")
    require_sha256(implementation["audit_sha256"], "implementation audit")
    final_audit = config["final_audit"]
    if (
        not isinstance(final_audit, dict)
        or set(final_audit) != {"audit_id", "audit_path"}
        or not isinstance(final_audit["audit_id"], str)
        or re.fullmatch(r"R[0-9]+", final_audit["audit_id"]) is None
        or not isinstance(final_audit["audit_path"], str)
        or Path(final_audit["audit_path"]).is_absolute()
        or ".." in Path(final_audit["audit_path"]).parts
        or Path(final_audit["audit_path"]).parent
        != Path(
            "Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports"
        )
        or Path(final_audit["audit_path"]).suffix != ".md"
    ):
        raise RuntimeError("Wave 60 final config-audit authority drifted")
    attempt = config["attempt"]
    require_exact_keys(
        attempt,
        {"version", "container", "primary", "replay", "pair", "recovery"},
        "attempt namespace",
    )
    version = int(attempt["version"])
    expected_container = (
        "data/geometria_proporcional/"
        f"wave60_frozen_policy_transport_attempt_v{version}"
    )
    if (
        version < 1
        or attempt["container"] != expected_container
        or attempt["primary"] != "primary"
        or attempt["replay"] != "replay"
        or attempt["pair"] != "pair"
    ):
        raise RuntimeError("Wave 60 attempt namespace drifted")
    if version == 1:
        if attempt["recovery"] is not None:
            raise RuntimeError("Wave 60 v1 cannot claim recovery authority")
    else:
        recovery = require_exact_keys(
            attempt["recovery"],
            {
                "schema_version",
                "prior_attempt_container",
                "prior_pair_failure_sha256",
                "amendment_path",
                "amendment_sha256",
                "amendment_audit_commit",
                "amendment_audit_path",
                "amendment_audit_sha256",
                "preserved_draw_sha256",
            },
            "recovery authority",
        )
        if (
            recovery["schema_version"] != "wave60-pretruth-recovery-v1"
            or recovery["prior_attempt_container"] == expected_container
            or not recovery["prior_attempt_container"].startswith(
                "data/geometria_proporcional/"
                "wave60_frozen_policy_transport_attempt_v"
            )
            or not isinstance(recovery["preserved_draw_sha256"], dict)
            or not recovery["preserved_draw_sha256"]
        ):
            raise RuntimeError("Wave 60 recovery authority drifted")
        for field in (
            "prior_pair_failure_sha256",
            "amendment_sha256",
            "amendment_audit_sha256",
        ):
            require_sha256(recovery[field], f"recovery {field}")
        if re.fullmatch(r"[0-9a-f]{40}", recovery["amendment_audit_commit"]) is None:
            raise RuntimeError("Wave 60 recovery audit commit drifted")
        for field in ("amendment_path", "amendment_audit_path"):
            candidate = Path(recovery[field])
            if (
                candidate.is_absolute()
                or candidate.as_posix() != recovery[field]
                or ".." in candidate.parts
            ):
                raise RuntimeError(f"Wave 60 recovery {field} drifted")
        for relative, digest in recovery["preserved_draw_sha256"].items():
            candidate = Path(relative)
            if candidate.is_absolute() or ".." in candidate.parts:
                raise RuntimeError("Wave 60 recovery draw path drifted")
            require_sha256(digest, f"recovery draw {relative}")
    if (
        config["primary_output"] != f"{attempt['container']}/primary"
        or config["replay_output"] != f"{attempt['container']}/replay"
    ):
        raise RuntimeError("Wave 60 output paths drifted")
    if (
        config["primary_output_name"] != "primary"
        or config["replay_output_name"] != "replay"
    ):
        raise RuntimeError("Wave 60 output names drifted")
    if config["output_parent_relative"] != attempt["container"]:
        raise RuntimeError("Wave 60 output parent drifted")
    bootstrap = config["bootstrap"]
    if bootstrap != {
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED,
        "interval": [2.5, 97.5],
        "unit": "pair_token_in_T_primary",
    }:
        raise RuntimeError("Wave 60 bootstrap contract drifted")
    runtime = config["runtime_budget"]
    if runtime != {
        "gpu_allowed": False,
        "max_seconds_total": 900,
        "max_rss_bytes_per_process": 1610612736,
    }:
        raise RuntimeError("Wave 60 runtime budget drifted")
    authority = config["source_law_authority"]
    if not isinstance(authority, dict) or set(authority) != {
        "path",
        "source_law_freeze_sha256",
        "source_law_attestation_sha256",
        "transport_law_manifest_sha256",
        "transport_law_arrays_sha256",
        "frozen_policy_spec_sha256",
        "feature_schema_sha256",
        "source_authority_manifest_sha256",
        "audit_commit",
        "audit_path",
        "audit_sha256",
    }:
        raise RuntimeError("Wave 60 source-law authority binding drifted")
    for key, value in authority.items():
        if key.endswith("sha256"):
            require_sha256(value, f"source-law authority {key}")
    if authority["path"] != (
        "data/geometria_proporcional/wave60_frozen_policy_transport_source_law_v1"
    ):
        raise RuntimeError("Wave 60 source-law authority path drifted")
    if (
        not isinstance(authority["audit_commit"], str)
        or re.fullmatch(r"[0-9a-f]{40}", authority["audit_commit"]) is None
        or not isinstance(authority["audit_path"], str)
        or Path(authority["audit_path"]).is_absolute()
        or ".." in Path(authority["audit_path"]).parts
    ):
        raise RuntimeError("Wave 60 source-law audit binding drifted")
    required_sources = config["required_execution_sources"]
    hashes = config["source_sha256"]
    expected_sources = {
        "experiments/geometria_proporcional/configs/wave60_frozen_policy_transport.json",
        "src/geometria_proporcional/wave60_frozen_policy_transport.py",
        "experiments/geometria_proporcional/run_wave60_frozen_policy_transport.py",
        "experiments/geometria_proporcional/_wave60_phase_worker.py",
        "experiments/geometria_proporcional/prepare_wave56_fresh.py",
        "tests/test_wave60_frozen_policy_transport.py",
        implementation["audit_path"],
        authority["audit_path"],
    }
    if (
        not isinstance(required_sources, list)
        or set(required_sources) != expected_sources
        or len(required_sources) != len(set(required_sources))
        or not isinstance(hashes, dict)
        or set(hashes) != set(required_sources)
    ):
        raise RuntimeError("Wave 60 execution source manifest drifted")
    for relative, digest in hashes.items():
        if (
            not isinstance(relative, str)
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
        ):
            raise RuntimeError("Wave 60 execution source path drifted")
        require_sha256(digest, f"execution source {relative}")
    if not isinstance(config["source_binding"], dict):
        raise RuntimeError("Wave 60 upstream source binding is absent")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def opaque_file_identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    stat = resolved.stat()
    if path.is_symlink() or not resolved.is_file():
        raise RuntimeError(
            f"opaque identity target is not a regular non-symlink: {path}"
        )
    return {
        "resolved_path_sha256": hashlib.sha256(str(resolved).encode()).hexdigest(),
        "device": int(stat.st_dev),
        "inode": int(stat.st_ino),
        "bytes": int(stat.st_size),
        "sha256": file_sha256(resolved),
    }


def require_independent_files(
    fresh: Path, antecedents: Iterable[Path]
) -> dict[str, Any]:
    """Compare opaque bytes and filesystem identities without parsing protected content."""
    current = opaque_file_identity(fresh)
    comparisons = []
    for antecedent in antecedents:
        if not antecedent.exists():
            continue
        previous = opaque_file_identity(antecedent)
        same_inode = (current["device"], current["inode"]) == (
            previous["device"],
            previous["inode"],
        )
        same_bytes = current["sha256"] == previous["sha256"]
        if same_inode or same_bytes:
            raise RuntimeError("INVALID_NEW_DRAW_IDENTITY")
        comparisons.append(
            {
                "antecedent_path_sha256": previous["resolved_path_sha256"],
                "same_inode": same_inode,
                "same_bytes": same_bytes,
            }
        )
    return {"fresh": current, "comparisons": comparisons}
