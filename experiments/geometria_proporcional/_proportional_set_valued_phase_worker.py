#!/usr/bin/env python3
"""Unprivileged staged worker for the physical set-valued CPU preflight.

The coordinator gives this program one closed stage and one writable scratch
directory.  It does not discover repository paths and never imports the
coordinator or the historical runner.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import itertools
import json
import os
from pathlib import Path
import resource
import sys
import time
from typing import Any
import zipfile

import numpy as np
from threadpoolctl import threadpool_info

from geometria_proporcional import proportional_set_valued_native as native


PHASES = (
    "posterior_fit",
    "policy_fit",
    "selection_propose",
    "selection_evaluate",
    "selection_freeze",
    "evaluation_apply",
    "evaluation_truth",
)
HEX64 = frozenset("0123456789abcdef")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(
            jsonable(payload), sort_keys=True, ensure_ascii=False,
            separators=(",", ":"), allow_nan=False,
        ) + "\n"
    ).encode("utf-8")


def fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.pending")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    fsync_dir(path.parent)


def write_json(path: Path, payload: Any) -> None:
    write_bytes(path, json_bytes(payload))


def write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in sorted(arrays):
            array = np.ascontiguousarray(arrays[name])
            if array.dtype.hasobject:
                raise TypeError(f"object array forbidden: {name}")
            raw = io.BytesIO()
            np.lib.format.write_array(raw, array, allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, raw.getvalue(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    write_bytes(path, buffer.getvalue())


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_digest(value: np.ndarray) -> str:
    return native.array_digest(np.asarray(value))


def validate_stage(stage: Path, phase: str) -> dict[str, Any]:
    if stage.is_symlink() or not stage.is_dir():
        raise RuntimeError("stage must be one real directory")
    request = read_json(stage / "phase_request.json")
    if request.get("schema_version") != "proportional-physical-phase-request-v1":
        raise RuntimeError("phase request schema drifted")
    if request.get("phase") != phase:
        raise RuntimeError("phase request phase drifted")
    actual = {path.name for path in stage.iterdir()}
    expected = set(request["allowed_files"])
    if actual != expected or "phase_request.json" not in actual:
        raise RuntimeError(f"stage allowlist mismatch: {sorted(actual)} != {sorted(expected)}")
    for path in stage.iterdir():
        if path.is_symlink() or not path.is_file() or path.stat().st_mode & 0o222:
            raise RuntimeError(f"unsafe staged object: {path.name}")
    hashes = request["sha256"]
    if set(hashes) != actual - {"phase_request.json"}:
        raise RuntimeError("stage hash coverage drifted")
    for name, expected_hash in hashes.items():
        if sha256_file(stage / name) != expected_hash:
            raise RuntimeError(f"stage hash mismatch: {name}")
    return request


def validate_runtime(config: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CUDA_VISIBLE_DEVICES is not empty")
    thread_names = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
    if {key: os.environ.get(key) for key in thread_names} != {key: "1" for key in thread_names}:
        raise RuntimeError("thread environment drifted")
    if "torch" in sys.modules:
        raise RuntimeError("torch imported")
    versions = native.validate_runtime_versions()
    if versions != config["versions"]:
        raise RuntimeError("runtime versions drifted")
    status = Path("/proc/self/status").read_text(encoding="utf-8")
    fields: dict[str, str] = {}
    for line in status.splitlines():
        key, _, value = line.partition(":")
        if key in {"Uid", "Gid", "Groups", "NoNewPrivs", "CapInh", "CapPrm", "CapEff", "CapBnd", "CapAmb"}:
            fields[key] = value.strip()
    if os.getuid() != 65534 or os.getgid() != 65534 or fields.get("Groups"):
        raise RuntimeError("worker identity or groups drifted")
    if fields.get("NoNewPrivs") != "1" or any(fields.get(key) != "0000000000000000" for key in ("CapInh", "CapPrm", "CapEff", "CapBnd", "CapAmb")):
        raise RuntimeError("worker privilege boundary drifted")
    pools = threadpool_info()
    if any(int(row.get("num_threads", 0)) > 1 for row in pools):
        raise RuntimeError("threadpool exceeds one thread")
    module_files = {}
    for name, module in sorted(sys.modules.items()):
        if name == "geometria_proporcional" or name.startswith("geometria_proporcional."):
            location = getattr(module, "__file__", None)
            if location:
                module_files[name] = {"path": str(Path(location).resolve()), "sha256": sha256_file(Path(location))}
    if set(module_files) != set(request["runtime_modules"]):
        raise RuntimeError(f"runtime module inventory drifted: {sorted(module_files)}")
    return {"versions": versions, "identity": fields, "threadpools": pools, "modules": module_files, "sys_path": sys.path, "environment": {key: os.environ.get(key) for key in request["environment_keys"]}, "torch_imported": False, "gpu_used_or_queried": False}


def run_probes(request: dict[str, Any]) -> list[dict[str, str]]:
    rows = []
    for encoded in request["probe_paths"]:
        path = Path(encoded)
        outcome = "OPENED"
        try:
            with path.open("rb") as handle:
                handle.read(1)
        except PermissionError:
            outcome = "PermissionError"
        except FileNotFoundError:
            outcome = "FileNotFoundError"
        if outcome not in {"PermissionError", "FileNotFoundError"}:
            raise RuntimeError("forbidden path probe opened")
        rows.append({"path_sha256": hashlib.sha256(str(path).encode()).hexdigest(), "outcome": outcome})
    return rows


def require_freeze(stage: Path, name: str, files: dict[str, Path]) -> dict[str, Any]:
    freeze = read_json(stage / name)
    frozen = freeze.get("files")
    if not isinstance(frozen, dict) or not set(files).issubset(frozen):
        raise RuntimeError(f"{name} binding coverage drifted")
    for label, path in files.items():
        if frozen[label] != sha256_file(path):
            raise RuntimeError(f"{name} mismatch: {label}")
    return freeze


def freeze(output: Path, name: str, phase: str, files: list[str], extra: dict[str, Any] | None = None) -> None:
    write_json(output / name, {"schema_version": "proportional-physical-phase-freeze-v1", "phase": phase, "files": {item: sha256_file(output / item) for item in files}, **(extra or {})})


def prefix(prefix_name: str, arrays: dict[str, np.ndarray], destination: dict[str, np.ndarray]) -> None:
    for name, value in arrays.items():
        destination[f"{prefix_name}__{name}"] = np.asarray(value)


def posterior_bundle(stage: Path) -> dict[str, Any]:
    states = read_json(stage / "posterior_states.json")
    arrays = load_npz(stage / "posterior_state_arrays.npz")
    return {"states": states, "arrays": arrays}


def posterior_mass(name: str, posterior: dict[str, Any], logits: np.ndarray, shuffled: bool = False) -> np.ndarray:
    role = "target_shuffled" if shuffled else "real"
    if name == "marginal":
        return native.marginal_set_mass(posterior["states"]["marginal"][role], logits)
    return native.joint_set_mass(
        posterior["states"]["joint"][role],
        posterior["arrays"][f"joint_{role}__final_theta"], logits,
    )


def portable_policy_arrays(states: dict[str, Any]) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for posterior_name in ("marginal", "joint"):
        family = states[posterior_name]
        rows = [("true", family["true"])] + [(f"control_{row['seed']}", row) for row in family["controls"]]
        for role, bundle in rows:
            for model_name, state in bundle["states"].items():
                base = f"{posterior_name}__{role}__{model_name}"
                for key in ("mean", "scale", "coef"):
                    arrays[f"{base}__{key}"] = np.asarray(state[key], dtype=np.float64)
                arrays[f"{base}__intercept"] = np.asarray([state["intercept"]], dtype=np.float64)
                if "n_iter" in state:
                    arrays[f"{base}__n_iter"] = np.asarray(state["n_iter"], dtype=np.int64)
    return arrays


def phase_posterior(stage: Path, output: Path, config: dict[str, Any]) -> None:
    data = load_npz(stage / "posterior_fit_truth.npz")
    logits = np.asarray(data["ensemble_logits"], dtype=np.float64)
    target = np.asarray(data["target"], dtype=bool)
    tokens = np.asarray(data["pair_token"]).astype(str)
    folds = native.posterior_fold_ids(tokens, data["design_stratum"], data["cardinality"])
    shuffled = native.target_derangement_v1(tokens, folds, data["design_stratum"], data["cardinality"], seed=int(config["posterior"]["target_shuffle"]["seed"]))
    if shuffled["permutable_fraction"] < float(config["posterior"]["target_shuffle"]["minimum_permutable_fraction"]):
        raise RuntimeError("NOT_EVALUABLE_POSTERIOR")
    shuffled_target = target[np.asarray(shuffled["donor_index"], dtype=np.int64)]
    marginal_real = native.fit_marginal_state(logits, target)
    marginal_shuffled = native.fit_marginal_state(logits, shuffled_target)
    joint_real = native.fit_joint_cv(logits, target, folds)
    joint_shuffled = native.fit_joint_cv(logits, shuffled_target, folds)
    states = {"schema_version": "proportional-posterior-states-v1", "marginal": {"real": marginal_real, "target_shuffled": marginal_shuffled}, "joint": {"real": joint_real["state"], "target_shuffled": joint_shuffled["state"]}, "target_shuffle": {"seed": int(config["posterior"]["target_shuffle"]["seed"]), "fixture_sha256": config["posterior"]["target_shuffle"]["fixture_sha256"], "permutable_fraction": float(shuffled["permutable_fraction"]), "singletons": shuffled["singletons"], "same_map_for_representations": True}}
    all_arrays: dict[str, np.ndarray] = {}
    prefix("joint_real", joint_real["arrays"], all_arrays)
    prefix("joint_target_shuffled", joint_shuffled["arrays"], all_arrays)
    private = {key: value for key, value in all_arrays.items() if any(part in key for part in ("fold_id", "oof_", "fold_"))}
    handoff = {key: value for key, value in all_arrays.items() if key not in private}
    write_json(output / "posterior_states.json", states)
    write_npz(output / "posterior_state_arrays.npz", handoff)
    write_npz(output / "posterior_oof_arrays.npz", private)
    write_json(output / "target_shuffle_map.json", shuffled["rows"])
    write_npz(output / "target_shuffle_arrays.npz", {"donor_index": shuffled["donor_index"], "permutable": shuffled["permutable"], "target_shuffled": shuffled_target})
    write_json(output / "posterior_fit_diagnostics.json", {"status": "EVALUABLE", "rows": len(tokens), "permutable_fraction": float(shuffled["permutable_fraction"])})
    names = ["posterior_states.json", "posterior_state_arrays.npz", "posterior_oof_arrays.npz", "target_shuffle_map.json", "target_shuffle_arrays.npz", "posterior_fit_diagnostics.json"]
    freeze(output, "posterior_fit_freeze.json", "posterior_fit", names)


def phase_policy(stage: Path, output: Path, config: dict[str, Any], utilities: np.ndarray) -> None:
    require_freeze(stage, "posterior_fit_freeze.json", {"posterior_states.json": stage / "posterior_states.json", "posterior_state_arrays.npz": stage / "posterior_state_arrays.npz"})
    posterior = posterior_bundle(stage)
    data = load_npz(stage / "policy_fit_truth.npz")
    logits = np.asarray(data["ensemble_logits"], dtype=np.float64)
    seed_logits = np.asarray(data["per_seed_logits"], dtype=np.float64)
    target = np.asarray(data["target"], dtype=bool)
    tokens = np.asarray(data["pair_token"]).astype(str)
    penalty = float(config["reader"]["penalty"])
    states: dict[str, Any] = {"schema_version": "proportional-contextual-reader-states-v1", "feature_names": list(native.FEATURE_NAMES)}
    private_scores: dict[str, np.ndarray] = {}
    private_controls: dict[str, np.ndarray] = {}
    control_maps: dict[str, Any] = {"schema_version": "proportional-matched-control-maps-v1"}
    diagnostics: dict[str, Any] = {"status": "EVALUABLE", "posteriors": {}}
    for posterior_name in ("marginal", "joint"):
        mass = posterior_mass(posterior_name, posterior, logits)
        training = native.reader_training_data(ensemble_logits=logits, per_seed_logits=seed_logits, target=target, set_mass=mass, utilities=utilities, penalty=penalty)
        true_fit = native.fit_reader_states(training)
        controls = native.fit_control_states(training, tokens)["controls"]
        states[posterior_name] = {"true": true_fit, "controls": [{"seed": row["seed"], "states": row["states"], "diagnostics": row["diagnostics"]} for row in controls]}
        control_maps[posterior_name] = [row["diagnostics"] for row in controls]
        true_scores = native.score_reader_states(true_fit["states"], training["design"], training["disagreement"])
        prefix(f"{posterior_name}__true", true_scores, private_scores)
        for key in ("design", "weights", "disagreement", "gain", "harm", "incompatibility", "hard_actions", "posterior_actions"):
            private_scores[f"{posterior_name}__{key}"] = np.asarray(training[key])
        for row in controls:
            seed = int(row["seed"])
            prefix(f"{posterior_name}__control_{seed}", native.score_reader_states(row["states"], training["design"], training["disagreement"]), private_scores)
            private_controls[f"{posterior_name}__control_{seed}__mapping"] = np.asarray(row["mapping"], dtype=np.int64)
            for key in ("gain", "harm", "incompatibility"):
                private_controls[f"{posterior_name}__control_{seed}__{key}"] = np.asarray(row[key])
        diagnostics["posteriors"][posterior_name] = {"active_rows": int(true_fit["active_rows"]), "active_tokens": int(true_fit["active_tokens"]), "controls": len(controls)}
    if tuple(config["reader"]["feature_names"]) != native.FEATURE_NAMES:
        raise RuntimeError("feature schema drifted")
    write_json(output / "feature_schema.json", {"schema_version": "proportional-contextual-map-features-v1", "count": len(native.FEATURE_NAMES), "feature_names": list(native.FEATURE_NAMES), "hard_adapter": "HARD_MAP_SET", "weighting": "one_total_weight_per_active_token"})
    write_json(output / "policy_states.json", states)
    write_npz(output / "policy_state_arrays.npz", portable_policy_arrays(states))
    write_npz(output / "policy_fit_private.npz", private_scores)
    write_json(output / "control_maps.json", control_maps)
    write_npz(output / "control_arrays.npz", private_controls)
    write_json(output / "policy_fit_diagnostics.json", diagnostics)
    names = ["feature_schema.json", "policy_states.json", "policy_state_arrays.npz", "policy_fit_private.npz", "control_maps.json", "control_arrays.npz", "policy_fit_diagnostics.json"]
    freeze(output, "policy_fit_freeze.json", "policy_fit", names)


def phase_selection_propose(stage: Path, output: Path, config: dict[str, Any], utilities: np.ndarray) -> None:
    require_freeze(stage, "posterior_fit_freeze.json", {"posterior_states.json": stage / "posterior_states.json", "posterior_state_arrays.npz": stage / "posterior_state_arrays.npz"})
    require_freeze(stage, "policy_fit_freeze.json", {"feature_schema.json": stage / "feature_schema.json", "policy_states.json": stage / "policy_states.json", "policy_state_arrays.npz": stage / "policy_state_arrays.npz"})
    posterior = posterior_bundle(stage)
    policies = read_json(stage / "policy_states.json")
    public = load_npz(stage / "decision_select_public.npz")
    logits = np.asarray(public["ensemble_logits"], dtype=np.float64)
    seed_logits = np.asarray(public["per_seed_logits"], dtype=np.float64)
    penalty = float(config["reader"]["penalty"])
    selection_keys: dict[str, Any] = {"schema_version": "proportional-selection-key-metadata-v1", "posteriors": {}}
    apply_metadata: dict[str, Any] = {"schema_version": "proportional-apply-metadata-v1", "posteriors": {}}
    arrays: dict[str, np.ndarray] = {"pair_token": np.asarray(public["pair_token"], dtype="<U64")}
    for name in ("marginal", "joint"):
        mass = posterior_mass(name, posterior, logits)
        shuffled_mass = posterior_mass(name, posterior, logits, True)
        pdata = native.reader_public_data(ensemble_logits=logits, per_seed_logits=seed_logits, set_mass=mass, utilities=utilities, penalty=penalty)
        scores = native.score_reader_states(policies[name]["true"]["states"], pdata["design"], pdata["disagreement"])
        grid = native.candidate_grid(scores, pdata["disagreement"], pdata["hard_actions"], pdata["posterior_actions"])
        selection_keys["posteriors"][name] = [{"candidate_index": index, "kind": row["kind"], "proposer_quantile": row["proposer_quantile"], "harm_quantile": row["harm_quantile"], "incompatibility_quantile": row["incompatibility_quantile"]} for index, row in enumerate(grid["metadata"])]
        apply_metadata["posteriors"][name] = [{"candidate_index": index, **row} for index, row in enumerate(grid["metadata"])]
        arrays[f"{name}__actions"] = grid["actions"]
        arrays[f"{name}__override"] = grid["override"]
        arrays[f"{name}__hard_actions"] = pdata["hard_actions"]
        arrays[f"{name}__posterior_actions"] = pdata["posterior_actions"]
        arrays[f"{name}__disagreement"] = pdata["disagreement"]
        arrays[f"{name}__design"] = pdata["design"]
        arrays[f"{name}__weights"] = pdata["weights"]
        arrays[f"{name}__set_mass_real"] = mass
        arrays[f"{name}__set_mass_target_shuffled"] = shuffled_mass
        prefix(f"{name}__score", scores, arrays)
    write_json(output / "selection_key_metadata.json", selection_keys)
    write_json(output / "apply_metadata.json", apply_metadata)
    write_npz(output / "candidate_public.npz", arrays)
    freeze(output, "candidate_freeze.json", "selection_propose", ["selection_key_metadata.json", "apply_metadata.json", "candidate_public.npz"], {"candidate_count_per_posterior": 344, "truth_received": False})


def phase_selection_evaluate(stage: Path, output: Path, config: dict[str, Any], utilities: np.ndarray) -> None:
    require_freeze(stage, "candidate_freeze.json", {"selection_key_metadata.json": stage / "selection_key_metadata.json", "candidate_public.npz": stage / "candidate_public.npz"})
    truth = load_npz(stage / "decision_select_truth.npz")
    metadata = read_json(stage / "selection_key_metadata.json")
    arrays = load_npz(stage / "candidate_public.npz")
    target = np.asarray(truth["target"], dtype=bool)
    if not np.array_equal(np.asarray(truth["pair_token"]).astype(str), np.asarray(arrays["pair_token"]).astype(str)):
        raise RuntimeError("selection token identity drifted")
    penalty = float(config["reader"]["penalty"])
    decision: dict[str, Any] = {"schema_version": "proportional-selection-decision-v1", "posteriors": {}}
    private_metrics: dict[str, np.ndarray] = {}
    raw: dict[str, np.ndarray] = {"pair_token": truth["pair_token"], "target": target}
    for name in ("marginal", "joint"):
        actions = arrays[f"{name}__actions"]
        overrides = arrays[f"{name}__override"]
        key_rows = metadata["posteriors"][name]
        grid = {"actions": actions, "override": overrides, "metadata": [{key: row[key] for key in ("kind", "proposer_quantile", "harm_quantile", "incompatibility_quantile")} for row in key_rows]}
        evaluated = native.evaluate_candidate_grid(grid, target, utilities, penalty, arrays[f"{name}__hard_actions"])
        index = int(evaluated["selected_index"])
        selected = evaluated["selected"]
        decision["posteriors"][name] = {"selected_index": index, "mean_regret": selected["mean_regret"], "incompatibility_rate": selected["incompatibility_rate"], "harm_rate": selected["harm_rate"], "authorized_rows": selected["authorized_rows"], "candidate_freeze_sha256": sha256_file(stage / "candidate_freeze.json"), "selected_actions_sha256": array_digest(actions[index]), "selected_override_sha256": array_digest(overrides[index])}
        prefix(name, evaluated["arrays"], private_metrics)
        metrics = native.action_metric_arrays(actions[index], target, utilities, penalty)
        prefix(name, metrics, raw)
    write_json(output / "selection_decision.json", decision)
    write_npz(output / "candidate_metrics_private.npz", private_metrics)
    write_npz(output / "selection_target_aligned_private.npz", raw)
    freeze(output, "selection_decision_freeze.json", "selection_evaluate", ["selection_decision.json", "candidate_metrics_private.npz", "selection_target_aligned_private.npz"], {"candidate_freeze_sha256": sha256_file(stage / "candidate_freeze.json")})


def phase_selection_freeze(stage: Path, output: Path, config: dict[str, Any], utilities: np.ndarray) -> None:
    require_freeze(stage, "posterior_fit_freeze.json", {"posterior_states.json": stage / "posterior_states.json", "posterior_state_arrays.npz": stage / "posterior_state_arrays.npz"})
    require_freeze(stage, "policy_fit_freeze.json", {"feature_schema.json": stage / "feature_schema.json", "policy_states.json": stage / "policy_states.json", "policy_state_arrays.npz": stage / "policy_state_arrays.npz"})
    require_freeze(stage, "candidate_freeze.json", {"apply_metadata.json": stage / "apply_metadata.json", "candidate_public.npz": stage / "candidate_public.npz"})
    require_freeze(stage, "selection_decision_freeze.json", {"selection_decision.json": stage / "selection_decision.json"})
    posterior = posterior_bundle(stage)
    policies = read_json(stage / "policy_states.json")
    public = load_npz(stage / "decision_select_public.npz")
    candidate = load_npz(stage / "candidate_public.npz")
    apply_meta = read_json(stage / "apply_metadata.json")
    decision = read_json(stage / "selection_decision.json")
    logits = np.asarray(public["ensemble_logits"], dtype=np.float64)
    seeds = np.asarray(public["per_seed_logits"], dtype=np.float64)
    penalty = float(config["reader"]["penalty"])
    policy: dict[str, Any] = {"schema_version": "proportional-selection-policy-v1", "posteriors": {}}
    selected_arrays: dict[str, np.ndarray] = {}
    match_arrays: dict[str, np.ndarray] = {}
    for name in ("marginal", "joint"):
        mass = posterior_mass(name, posterior, logits)
        pdata = native.reader_public_data(ensemble_logits=logits, per_seed_logits=seeds, set_mass=mass, utilities=utilities, penalty=penalty)
        scores = native.score_reader_states(policies[name]["true"]["states"], pdata["design"], pdata["disagreement"])
        grid = native.candidate_grid(scores, pdata["disagreement"], pdata["hard_actions"], pdata["posterior_actions"])
        if grid["metadata"] != [{key: row[key] for key in ("kind", "proposer_quantile", "harm_quantile", "incompatibility_quantile", "proposer_threshold", "harm_threshold", "incompatibility_threshold")} for row in apply_meta["posteriors"][name]]:
            raise RuntimeError("apply metadata reconstitution drifted")
        if not np.array_equal(grid["actions"], candidate[f"{name}__actions"]) or not np.array_equal(grid["override"], candidate[f"{name}__override"]):
            raise RuntimeError("candidate reconstitution drifted")
        index = int(decision["posteriors"][name]["selected_index"])
        actions = grid["actions"][index]
        override = grid["override"][index]
        if array_digest(actions) != decision["posteriors"][name]["selected_actions_sha256"] or array_digest(override) != decision["posteriors"][name]["selected_override_sha256"]:
            raise RuntimeError("selected action binding drifted")
        selected = grid["metadata"][index]
        controls = []
        valid_masks = []
        for row in policies[name]["controls"]:
            seed = int(row["seed"])
            control_scores = native.score_reader_states(row["states"], pdata["design"], pdata["disagreement"])
            if selected["kind"] == "hard_only":
                thresholds = {"proposer_quantile": 2.0, "harm_quantile": 2.0, "incompatibility_quantile": 2.0, "proposer_threshold": None, "harm_threshold": None, "incompatibility_threshold": None}
                matched = {"actions": pdata["hard_actions"], "selected": np.zeros_like(override), "authorized_universe": np.zeros_like(override), "match_valid": np.ones(len(override), dtype=bool), "requested_k": np.zeros(len(override), dtype=np.int64)}
            else:
                quantiles = (float(selected["proposer_quantile"]), float(selected["harm_quantile"]), float(selected["incompatibility_quantile"]))
                thresholds = native.threshold_triplet(control_scores, pdata["disagreement"], quantiles)
                matched = native.matched_control_actions(true_override=override, scores=control_scores, thresholds=thresholds, disagreement=pdata["disagreement"], hard_actions=pdata["hard_actions"], candidate_actions=pdata["posterior_actions"])
            controls.append({"seed": seed, "thresholds": thresholds})
            valid_masks.append(np.asarray(matched["match_valid"], dtype=bool))
            for key, value in matched.items():
                match_arrays[f"{name}__control_{seed}__{key}"] = np.asarray(value)
        u_true = override.any(axis=1)
        common = u_true.copy()
        for valid in valid_masks:
            common &= valid
        coverage = float(common.sum() / max(1, u_true.sum()))
        support = "NOT_EVALUABLE_NO_TRUE_OVERRIDES" if not u_true.any() else ("EVALUABLE" if coverage >= float(config["matched_controls"]["minimum_common_coverage"]) else "NOT_EVALUABLE_CONTROL_SUPPORT")
        policy["posteriors"][name] = {"selected_index": index, "selected": selected, "controls": controls, "u_true_count": int(u_true.sum()), "u_common_count": int(common.sum()), "common_coverage": coverage, "common_support_status": support}
        selected_arrays[f"{name}__actions"] = actions
        selected_arrays[f"{name}__override"] = override
        selected_arrays[f"{name}__hard_actions"] = pdata["hard_actions"]
        match_arrays[f"{name}__u_true"] = u_true
        match_arrays[f"{name}__u_common"] = common
    write_json(output / "selection_policy.json", policy)
    write_npz(output / "selected_actions.npz", selected_arrays)
    write_npz(output / "selection_matches.npz", match_arrays)
    freeze(output, "selection_policy_freeze.json", "selection_freeze", ["selection_policy.json", "selected_actions.npz", "selection_matches.npz"], {"truth_received": False})


def phase_evaluation_apply(stage: Path, output: Path, config: dict[str, Any], utilities: np.ndarray) -> None:
    require_freeze(stage, "posterior_fit_freeze.json", {"posterior_states.json": stage / "posterior_states.json", "posterior_state_arrays.npz": stage / "posterior_state_arrays.npz"})
    require_freeze(stage, "policy_fit_freeze.json", {"feature_schema.json": stage / "feature_schema.json", "policy_states.json": stage / "policy_states.json", "policy_state_arrays.npz": stage / "policy_state_arrays.npz"})
    require_freeze(stage, "selection_policy_freeze.json", {"selection_policy.json": stage / "selection_policy.json", "selected_actions.npz": stage / "selected_actions.npz", "selection_matches.npz": stage / "selection_matches.npz"})
    posterior = posterior_bundle(stage)
    policies = read_json(stage / "policy_states.json")
    policy = read_json(stage / "selection_policy.json")
    public = load_npz(stage / "evaluate_public.npz")
    logits = np.asarray(public["ensemble_logits"], dtype=np.float64)
    seeds = np.asarray(public["per_seed_logits"], dtype=np.float64)
    penalty = float(config["reader"]["penalty"])
    actions: dict[str, np.ndarray] = {}
    masses: dict[str, np.ndarray] = {}
    sensitivity: dict[str, np.ndarray] = {}
    apply_status: dict[str, Any] = {}
    for name in ("marginal", "joint"):
        mass = posterior_mass(name, posterior, logits)
        shuffled_mass = posterior_mass(name, posterior, logits, True)
        masses[f"{name}__real"] = mass
        masses[f"{name}__target_shuffled"] = shuffled_mass
        pdata = native.reader_public_data(ensemble_logits=logits, per_seed_logits=seeds, set_mass=mass, utilities=utilities, penalty=penalty)
        scores = native.score_reader_states(policies[name]["true"]["states"], pdata["design"], pdata["disagreement"])
        selected = policy["posteriors"][name]["selected"]
        if selected["kind"] == "hard_only":
            contextual = pdata["hard_actions"].copy()
            override = np.zeros_like(pdata["hard_actions"], dtype=bool)
        else:
            applied = native.apply_threshold_triplet(scores, pdata["disagreement"], pdata["hard_actions"], pdata["posterior_actions"], selected)
            contextual, override = applied["actions"], applied["override"]
        actions[f"{name}__hard_actions"] = pdata["hard_actions"]
        actions[f"{name}__contextual_actions"] = contextual
        actions[f"{name}__true_override"] = override
        valid_masks = []
        for control_row, state_row in zip(policy["posteriors"][name]["controls"], policies[name]["controls"], strict=True):
            seed = int(control_row["seed"])
            if seed != int(state_row["seed"]):
                raise RuntimeError("control state order drifted")
            control_scores = native.score_reader_states(state_row["states"], pdata["design"], pdata["disagreement"])
            if selected["kind"] == "hard_only":
                matched = {"actions": pdata["hard_actions"], "selected": np.zeros_like(override), "authorized_universe": np.zeros_like(override), "match_valid": np.ones(len(override), dtype=bool), "requested_k": np.zeros(len(override), dtype=np.int64)}
            else:
                matched = native.matched_control_actions(true_override=override, scores=control_scores, thresholds=control_row["thresholds"], disagreement=pdata["disagreement"], hard_actions=pdata["hard_actions"], candidate_actions=pdata["posterior_actions"])
            valid_masks.append(np.asarray(matched["match_valid"], dtype=bool))
            for key, value in matched.items():
                actions[f"{name}__control_{seed}__{key}"] = np.asarray(value)
        u_true = override.any(axis=1)
        common = u_true.copy()
        for valid in valid_masks:
            common &= valid
        actions[f"{name}__u_true"] = u_true
        actions[f"{name}__u_common"] = common
        coverage = float(common.sum() / max(1, u_true.sum()))
        apply_status[name] = {"u_true_count": int(u_true.sum()), "u_common_count": int(common.sum()), "common_coverage": coverage, "common_support_status": "NOT_EVALUABLE_NO_TRUE_OVERRIDES" if not u_true.any() else ("EVALUABLE" if coverage >= float(config["matched_controls"]["minimum_common_coverage"]) else "NOT_EVALUABLE_CONTROL_SUPPORT")}
        for checkpoint_index, epoch in enumerate(config["checkpoint_epochs"]):
            checkpoint_logits = seeds[checkpoint_index]
            checkpoint_mass = posterior_mass(name, posterior, checkpoint_logits)
            sensitivity[f"checkpoint_{epoch}__{name}__mass"] = checkpoint_mass
            cpdata = native.reader_public_data(ensemble_logits=checkpoint_logits, per_seed_logits=seeds, set_mass=checkpoint_mass, utilities=utilities, penalty=penalty)
            cp_scores = native.score_reader_states(policies[name]["true"]["states"], cpdata["design"], cpdata["disagreement"])
            if selected["kind"] == "hard_only":
                cp_contextual = cpdata["hard_actions"]
            else:
                cp_contextual = native.apply_threshold_triplet(cp_scores, cpdata["disagreement"], cpdata["hard_actions"], cpdata["posterior_actions"], selected)["actions"]
            sensitivity[f"checkpoint_{epoch}__{name}__hard_actions"] = cpdata["hard_actions"]
            sensitivity[f"checkpoint_{epoch}__{name}__contextual_actions"] = cp_contextual
    metadata = {"pair_token": np.asarray(public["pair_token"], dtype="<U64"), "design_stratum": np.asarray(public["design_stratum"], dtype="<U16"), "cardinality": np.asarray(public["cardinality"], dtype="<i8")}
    write_npz(output / "evaluation_metadata.npz", metadata)
    write_npz(output / "evaluation_actions.npz", actions)
    write_npz(output / "evaluation_masses.npz", masses)
    write_npz(output / "evaluation_sensitivities.npz", sensitivity)
    write_json(output / "evaluation_apply_status.json", apply_status)
    freeze(output, "evaluation_action_freeze.json", "evaluation_apply", ["evaluation_metadata.npz", "evaluation_actions.npz", "evaluation_masses.npz", "evaluation_sensitivities.npz", "evaluation_apply_status.json"], {"truth_received": False, "public_pair_token_sha256": array_digest(metadata["pair_token"])})


def metric_summary(metrics: dict[str, np.ndarray]) -> dict[str, float]:
    return {key: float(np.mean(metrics[key])) for key in ("accuracy", "incompatibility", "regret", "worst_regret")}


def estimand(row_id: str, instance: str, left_name: str, right_name: str, left: np.ndarray, right: np.ndarray, indices: np.ndarray | None, allow_zero: bool, evaluable: bool = True, diagnostic: bool = False) -> dict[str, Any]:
    if not evaluable or indices is None or not len(left):
        return {"id": row_id, "instance": instance, "left": left_name, "right": right_name, "orientation": "left_minus_right", "status": "NOT_EVALUABLE", "diagnostic_only": diagnostic, "n_tokens": int(len(left))}
    summary = native.paired_delta_summary(left, right, indices)
    status = "DESCRIPTIVE_ONLY" if diagnostic else native.classify_loss_delta(summary, allow_zero_upper=allow_zero, support_ok=True)
    return {"id": row_id, "instance": instance, "left": left_name, "right": right_name, "orientation": "left_minus_right", "status": status, "diagnostic_only": diagnostic, "allow_zero_upper": allow_zero, **summary}


def phase_evaluation_truth(stage: Path, output: Path, config: dict[str, Any], utilities: np.ndarray) -> None:
    require_freeze(stage, "evaluation_action_freeze.json", {name: stage / name for name in ("evaluation_metadata.npz", "evaluation_actions.npz", "evaluation_masses.npz", "evaluation_sensitivities.npz", "evaluation_apply_status.json")})
    truth = load_npz(stage / "evaluate_truth.npz")
    metadata = load_npz(stage / "evaluation_metadata.npz")
    actions = load_npz(stage / "evaluation_actions.npz")
    masses = load_npz(stage / "evaluation_masses.npz")
    sensitivity = load_npz(stage / "evaluation_sensitivities.npz")
    status = read_json(stage / "evaluation_apply_status.json")
    tokens = np.asarray(metadata["pair_token"]).astype(str)
    target = np.asarray(truth["target"], dtype=bool)
    if not np.array_equal(tokens, np.asarray(truth["pair_token"]).astype(str)) or not np.array_equal(metadata["cardinality"], target.sum(axis=1).astype("<i8")):
        raise RuntimeError("evaluation identity/cardinality mismatch")
    penalty = float(config["reader"]["penalty"])
    global_boot = native.bootstrap_indices(len(tokens), int(config["bootstrap"]["replicates"]), int(config["bootstrap"]["global_seed"]))
    boot: dict[str, np.ndarray] = {"global_pair_token_index": global_boot, "global_pair_token": tokens}
    raw: dict[str, np.ndarray] = {"pair_token": tokens, "target": target}
    metrics_doc: dict[str, Any] = {"schema_version": "proportional-physical-diagnostic-metrics-v1", "status": "OPENED_DATA_PHYSICAL_PREFLIGHT", "posteriors": {}}
    set_metrics: dict[str, Any] = {}
    action_metrics: dict[str, Any] = {}
    cells: dict[str, Any] = {}
    common_boot: dict[str, Any] = {}
    for index, name in enumerate(("marginal", "joint")):
        real = native.exact_set_metric_arrays(masses[f"{name}__real"], target)
        shuffled = native.exact_set_metric_arrays(masses[f"{name}__target_shuffled"], target)
        hard = native.action_metric_arrays(actions[f"{name}__hard_actions"], target, utilities, penalty)
        contextual = native.action_metric_arrays(actions[f"{name}__contextual_actions"], target, utilities, penalty)
        controls = []
        for seed in config["matched_controls"]["seeds"]:
            cm = native.action_metric_arrays(actions[f"{name}__control_{seed}__actions"], target, utilities, penalty)
            controls.append(cm)
            prefix(f"{name}__control_{seed}", cm, raw)
        set_metrics[name] = {"real": real, "target_shuffled": shuffled}
        action_metrics[name] = {"hard": hard, "contextual": contextual, "controls": controls}
        for reader, values in (("hard", hard), ("contextual", contextual)):
            cells[f"{name}_{reader}"] = values
            prefix(f"{name}__{reader}", values, raw)
        prefix(f"{name}__real", real, raw)
        prefix(f"{name}__target_shuffled", shuffled, raw)
        common = np.asarray(actions[f"{name}__u_common"], dtype=bool)
        support_tokens = tokens[common]
        if len(support_tokens):
            seed_key = "marginal_common_seed" if name == "marginal" else "joint_common_seed"
            current_boot = native.bootstrap_indices(len(support_tokens), int(config["bootstrap"]["replicates"]), int(config["bootstrap"][seed_key]))
        else:
            current_boot = None
        common_boot[name] = current_boot
        boot[f"{name}__common_index"] = np.empty((int(config["bootstrap"]["replicates"]), 0), dtype=np.int64) if current_boot is None else current_boot
        boot[f"{name}__common_pair_token"] = support_tokens
        metrics_doc["posteriors"][name] = {"set_real": {key: float(np.mean(value)) for key, value in real.items()}, "set_target_shuffled": {key: float(np.mean(value)) for key, value in shuffled.items()}, "hard": metric_summary(hard), "contextual": metric_summary(contextual), "matched_controls": [{"seed": int(seed), **metric_summary(value)} for seed, value in zip(config["matched_controls"]["seeds"], controls, strict=True)], **status[name]}
    rows = [
        estimand("SET_JOINT_NLL", "joint_minus_marginal", "joint_real_exact_set_nll", "marginal_real_exact_set_nll", set_metrics["joint"]["real"]["exact_set_nll"], set_metrics["marginal"]["real"]["exact_set_nll"], global_boot, False),
        estimand("SET_JOINT_BRIER", "joint_minus_marginal", "joint_real_marginal_brier", "marginal_real_marginal_brier", set_metrics["joint"]["real"]["marginal_brier"], set_metrics["marginal"]["real"]["marginal_brier"], global_boot, True),
    ]
    for name in ("marginal", "joint"):
        rows.append(estimand("SET_SHUFFLE", name, f"{name}_real_exact_set_nll", f"{name}_target_shuffled_exact_set_nll", set_metrics[name]["real"]["exact_set_nll"], set_metrics[name]["target_shuffled"]["exact_set_nll"], global_boot, False))
        for row_id, metric_name, allow in (("READER_REGRET", "regret", False), ("READER_COMPAT", "incompatibility", True), ("READER_WORST", "worst_regret", True)):
            rows.append(estimand(row_id, name, f"{name}_contextual_{metric_name}", f"{name}_hard_{metric_name}", action_metrics[name]["contextual"][metric_name], action_metrics[name]["hard"][metric_name], global_boot, allow))
        common = np.asarray(actions[f"{name}__u_common"], dtype=bool)
        control_regret = np.stack([row["regret"] for row in action_metrics[name]["controls"]])
        rows.append(estimand("READER_CONTROL", name, f"{name}_contextual_regret", f"{name}_matched_control_mean_regret", action_metrics[name]["contextual"]["regret"][common], np.mean(control_regret[:, common], axis=0), common_boot[name], False, status[name]["common_support_status"] == "EVALUABLE"))
    rows.append(estimand("FACTOR_INTERACTION", "joint_reader_delta_minus_marginal_reader_delta", "joint_contextual_minus_hard_regret", "marginal_contextual_minus_hard_regret", action_metrics["joint"]["contextual"]["regret"] - action_metrics["joint"]["hard"]["regret"], action_metrics["marginal"]["contextual"]["regret"] - action_metrics["marginal"]["hard"]["regret"], global_boot, True, diagnostic=True))
    if {row["id"] for row in rows} != set(config["required_decision_rows"]):
        raise RuntimeError("estimand coverage drifted")
    def satisfied(row_id: str, instance: str | None = None) -> bool:
        selected = [row for row in rows if row["id"] == row_id and (instance is None or row["instance"] == instance)]
        return bool(selected) and all(row["status"] == "CONDITION_SATISFIED" for row in selected)
    estimands = {"schema_version": "proportional-set-valued-estimands-v1", "status": "OPENED_DATA_PHYSICAL_PREFLIGHT", "bootstrap_unit": "pair_token", "training_seed_population_claimed": False, "scientific_decision": None, "architecture_promoted": False, "prospective_evidence": False, "decision_authority": "user", "rows": rows, "patterns": {"JOINT_PATTERN_PRESENT": all((satisfied("SET_JOINT_NLL"), satisfied("SET_JOINT_BRIER"), satisfied("SET_SHUFFLE", "joint"))), "CONTEXTUAL_PATTERN_PRESENT": {name: all(satisfied(row_id, name) for row_id in ("READER_REGRET", "READER_COMPAT", "READER_WORST", "READER_CONTROL")) for name in ("marginal", "joint")}, "interpretation": "OPENED_DATA_PHYSICAL_PREFLIGHT_ONLY"}}
    sensitivity_rows = []
    cardinality = np.asarray(metadata["cardinality"], dtype=np.int64)
    for epoch in config["checkpoint_epochs"]:
        for name in ("marginal", "joint"):
            for reader in ("hard", "contextual"):
                metric = native.action_metric_arrays(sensitivity[f"checkpoint_{epoch}__{name}__{reader}_actions"], target, utilities, penalty)
                prefix(f"checkpoint_{epoch}__{name}__{reader}", metric, raw)
                for card in sorted(np.unique(cardinality).tolist()):
                    mask = cardinality == card
                    sensitivity_rows.append({"checkpoint_epoch": int(epoch), "checkpoint_is_population_seed": False, "posterior": name, "reader": reader, "cardinality": int(card), "n_tokens": int(mask.sum()), "mean_regret": float(metric["regret"][mask].mean()), "mean_incompatibility": float(metric["incompatibility"][mask].mean()), "mean_accuracy": float(metric["accuracy"][mask].mean())})
    metrics_doc["checkpoint_sensitivity"] = sensitivity_rows
    duplications = []
    cell_actions = {"marginal_hard": actions["marginal__hard_actions"], "marginal_contextual": actions["marginal__contextual_actions"], "joint_hard": actions["joint__hard_actions"], "joint_contextual": actions["joint__contextual_actions"]}
    for left, right in itertools.combinations(sorted(cells), 2):
        duplications.append({"left": left, "right": right, "actions_exact": bool(np.array_equal(cell_actions[left], cell_actions[right])), "action_position_equal_fraction": float(np.mean(cell_actions[left] == cell_actions[right])), "regret_exact": bool(np.array_equal(cells[left]["regret_by_policy"], cells[right]["regret_by_policy"]))})
    write_json(output / "diagnostic_metrics.json", metrics_doc)
    write_npz(output / "diagnostic_arrays.npz", raw)
    write_npz(output / "bootstrap_indices.npz", boot)
    write_json(output / "estimand_table.json", estimands)
    write_json(output / "cell_duplications.json", {"schema_version": "proportional-cell-duplications-v1", "cells_retained_even_if_equal": True, "comparisons": duplications})
    names = ["diagnostic_metrics.json", "diagnostic_arrays.npz", "bootstrap_indices.npz", "estimand_table.json", "cell_duplications.json"]
    freeze(output, "estimand_freeze.json", "evaluation_truth", names, {"maximum_status": "PHYSICAL_PROSPECTIVE_PACKAGE_PREFLIGHT_VALID"})


def main() -> int:
    args = parse_args()
    stage = args.stage.resolve(strict=True)
    output = args.output.resolve(strict=True)
    if any(output.iterdir()):
        raise RuntimeError("worker output must begin empty")
    request = validate_stage(stage, args.phase)
    config = read_json(stage / "config.json")
    runtime = validate_runtime(config, request)
    probes = run_probes(request)
    utilities = np.load(stage / "utilities.npy", allow_pickle=False) if (stage / "utilities.npy").exists() else np.empty((0, 0))
    started = time.monotonic()
    dispatch = {
        "posterior_fit": lambda: phase_posterior(stage, output, config),
        "policy_fit": lambda: phase_policy(stage, output, config, utilities),
        "selection_propose": lambda: phase_selection_propose(stage, output, config, utilities),
        "selection_evaluate": lambda: phase_selection_evaluate(stage, output, config, utilities),
        "selection_freeze": lambda: phase_selection_freeze(stage, output, config, utilities),
        "evaluation_apply": lambda: phase_evaluation_apply(stage, output, config, utilities),
        "evaluation_truth": lambda: phase_evaluation_truth(stage, output, config, utilities),
    }
    dispatch[args.phase]()
    if "torch" in sys.modules:
        raise RuntimeError("torch imported during phase")
    actual = {path.name for path in output.iterdir() if path.is_file()}
    expected = set(request["expected_outputs"])
    if actual != expected:
        raise RuntimeError(f"output allowlist mismatch: {sorted(actual)} != {sorted(expected)}")
    receipt = {"schema_version": "proportional-physical-worker-receipt-v1", "phase": args.phase, "wall_seconds": time.monotonic() - started, "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024, "runtime": runtime, "probes": probes, "stage_files": request["sha256"], "output_files": {name: sha256_file(output / name) for name in sorted(actual)}}
    write_json(output / "worker_receipt.json", receipt)
    print(json.dumps({"phase": args.phase, "status": "PASS", "wall_seconds": receipt["wall_seconds"], "peak_rss_bytes": receipt["peak_rss_bytes"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
