#!/usr/bin/env python3
"""Coordinate the seven physically separated set-valued CPU phases."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import io
import itertools
import json
import os
from pathlib import Path
import pwd
import resource
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from typing import Any
import zipfile

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DEFAULT = REPO_ROOT / "experiments/geometria_proporcional/configs/proportional_set_valued_physical_preflight_v1.json"
SOURCE_FREEZE_DEFAULT = REPO_ROOT / "experiments/geometria_proporcional/configs/proportional_set_valued_physical_source_freeze_v1.json"
INPUT_DEFAULT = REPO_ROOT / "data/geometria_proporcional/proportional_set_valued_physical_input_v1"
OUTPUT_DEFAULT = REPO_ROOT / "data/geometria_proporcional/proportional_set_valued_physical_preflight_v1"
WORKER_SOURCE = REPO_ROOT / "experiments/geometria_proporcional/_proportional_set_valued_phase_worker.py"
PYTHON = REPO_ROOT / "venv/bin/python"
RUNTIME_SOURCES = (
    "experiments/geometria_proporcional/_proportional_set_valued_phase_worker.py",
    "src/geometria_proporcional/__init__.py",
    "src/geometria_proporcional/wave49_schema.py",
    "src/geometria_proporcional/proportional_set_valued_native.py",
    "src/geometria_proporcional/wave53_uncertainty.py",
    "src/geometria_proporcional/wave54_joint_set.py",
)
RUNTIME_MODULES = (
    "geometria_proporcional",
    "geometria_proporcional.proportional_set_valued_native",
    "geometria_proporcional.wave49_schema",
    "geometria_proporcional.wave53_uncertainty",
    "geometria_proporcional.wave54_joint_set",
)
PHASES = (
    "posterior_fit", "policy_fit", "selection_propose", "selection_evaluate",
    "selection_freeze", "evaluation_apply", "evaluation_truth",
)
STATE_AFTER = {
    "posterior_fit": "POSTERIOR_FIT_COMPLETE",
    "policy_fit": "POLICY_FIT_COMPLETE",
    "selection_propose": "SELECTION_CANDIDATES_FROZEN",
    "selection_evaluate": "SELECTION_DECISION_FROZEN",
    "selection_freeze": "SELECTION_POLICY_FROZEN",
    "evaluation_apply": "EVALUATION_ACTIONS_FROZEN",
    "evaluation_truth": "COMPLETE",
}
TRUTH_LEVEL = {
    "posterior_fit": "POSTERIOR_FIT", "policy_fit": "POLICY_FIT",
    "selection_propose": "POLICY_FIT", "selection_evaluate": "DECISION_SELECT",
    "selection_freeze": "DECISION_SELECT", "evaluation_apply": "DECISION_SELECT",
    "evaluation_truth": "EVALUATE",
}
EXPECTED_OUTPUTS = {
    "posterior_fit": ("posterior_states.json", "posterior_state_arrays.npz", "posterior_oof_arrays.npz", "target_shuffle_map.json", "target_shuffle_arrays.npz", "posterior_fit_diagnostics.json", "posterior_fit_freeze.json"),
    "policy_fit": ("feature_schema.json", "policy_states.json", "policy_state_arrays.npz", "policy_fit_private.npz", "control_maps.json", "control_arrays.npz", "policy_fit_diagnostics.json", "policy_fit_freeze.json"),
    "selection_propose": ("selection_key_metadata.json", "apply_metadata.json", "candidate_public.npz", "candidate_freeze.json"),
    "selection_evaluate": ("selection_decision.json", "candidate_metrics_private.npz", "selection_target_aligned_private.npz", "selection_decision_freeze.json"),
    "selection_freeze": ("selection_policy.json", "selected_actions.npz", "selection_matches.npz", "selection_policy_freeze.json"),
    "evaluation_apply": ("evaluation_metadata.npz", "evaluation_actions.npz", "evaluation_masses.npz", "evaluation_sensitivities.npz", "evaluation_apply_status.json", "evaluation_action_freeze.json"),
    "evaluation_truth": ("diagnostic_metrics.json", "diagnostic_arrays.npz", "bootstrap_indices.npz", "estimand_table.json", "cell_duplications.json", "estimand_freeze.json"),
}
HANDOFF = {
    "posterior_fit": ("posterior_states.json", "posterior_state_arrays.npz", "posterior_fit_freeze.json"),
    "policy_fit": ("feature_schema.json", "policy_states.json", "policy_state_arrays.npz", "policy_fit_freeze.json"),
    "selection_propose": ("selection_key_metadata.json", "apply_metadata.json", "candidate_public.npz", "candidate_freeze.json"),
    "selection_evaluate": ("selection_decision.json", "selection_decision_freeze.json"),
    "selection_freeze": ("selection_policy.json", "selected_actions.npz", "selection_matches.npz", "selection_policy_freeze.json"),
    "evaluation_apply": ("evaluation_metadata.npz", "evaluation_actions.npz", "evaluation_masses.npz", "evaluation_sensitivities.npz", "evaluation_apply_status.json", "evaluation_action_freeze.json"),
}
ENVIRONMENT_KEYS = (
    "PATH", "LANG", "LC_ALL", "PYTHONPATH", "PYTHONNOUSERSITE", "PYTHONHASHSEED",
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
    "KMP_DUPLICATE_LIB_OK", "KMP_INIT_AT_FORK", "CUDA_VISIBLE_DEVICES", "PHIDEUS_STAGED_RUNTIME",
)
PRIVATE_NAMES = frozenset({
    "posterior_oof_arrays.npz", "target_shuffle_map.json", "target_shuffle_arrays.npz",
    "policy_fit_private.npz", "control_maps.json", "control_arrays.npz",
    "candidate_metrics_private.npz", "selection_target_aligned_private.npz",
    "diagnostic_arrays.npz", "bootstrap_indices.npz",
})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_DEFAULT)
    parser.add_argument("--source-freeze", type=Path, default=SOURCE_FREEZE_DEFAULT)
    parser.add_argument("--input-package", type=Path, default=INPUT_DEFAULT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--reference-dir", type=Path)
    parser.add_argument("--execution-class", default="OPENED_DATA_PHYSICAL_PREFLIGHT")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--inject-crash-after-promotion", choices=PHASES)
    parser.add_argument("--inject-crash-after-journal", choices=PHASES)
    return parser.parse_args()


def jsonable(value: Any) -> Any:
    if isinstance(value, dict): return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)): return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, Path): return str(value)
    return value


def json_bytes(payload: Any) -> bytes:
    return (json.dumps(jsonable(payload), sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try: os.fsync(descriptor)
    finally: os.close(descriptor)


def write_bytes(path: Path, payload: bytes, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(raw)
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path); fsync_dir(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True); raise


def write_json(path: Path, payload: Any, mode: int = 0o644) -> None:
    write_bytes(path, json_bytes(payload), mode)


def write_json_exclusive(path: Path, payload: Any, mode: int = 0o444) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(json_bytes(payload)); handle.flush(); os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True); raise
    fsync_dir(path.parent)


def write_npz(path: Path, arrays: dict[str, np.ndarray], mode: int = 0o644) -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in sorted(arrays):
            array = np.ascontiguousarray(arrays[name])
            if array.dtype.hasobject: raise TypeError(f"object array forbidden: {name}")
            raw = io.BytesIO(); np.lib.format.write_array(raw, array, allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0)); info.compress_type = zipfile.ZIP_DEFLATED; info.external_attr = 0o600 << 16
            archive.writestr(info, raw.getvalue(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    write_bytes(path, buffer.getvalue(), mode)


def write_npy(path: Path, array: np.ndarray, mode: int = 0o644) -> None:
    buffer = io.BytesIO(); np.lib.format.write_array(buffer, np.ascontiguousarray(array), allow_pickle=False)
    write_bytes(path, buffer.getvalue(), mode)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive: return {key: archive[key].copy() for key in archive.files}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True, stderr=subprocess.STDOUT).strip()


def ensure_child(path: Path) -> Path:
    resolved = path.resolve()
    if resolved == REPO_ROOT or REPO_ROOT not in resolved.parents: raise ValueError("path must be a strict repository child")
    return resolved


def archive_path(path: Path, suffix: str = "archived") -> Path:
    index = 1
    while True:
        candidate = path.with_name(f"{path.name}.{suffix}.{index:03d}")
        if not candidate.exists(): path.rename(candidate); fsync_dir(path.parent); return candidate
        index += 1


def source_preflight(config_path: Path, freeze_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    freeze = read_json(freeze_path)
    if freeze.get("schema_version") != "proportional-set-valued-physical-source-freeze-v1": raise RuntimeError("source freeze schema drifted")
    freeze_rel = freeze_path.relative_to(REPO_ROOT).as_posix()
    freeze_commit = git("log", "-1", "--format=%H", "--", freeze_rel)
    if git("rev-parse", f"{freeze_commit}^") != freeze["implementation_commit"]: raise RuntimeError("source freeze parent is not implementation commit")
    if git("diff-tree", "--no-commit-id", "--name-only", "-r", freeze_commit).splitlines() != [freeze_rel]: raise RuntimeError("source freeze commit is not freeze-only")
    if subprocess.run(["git", "merge-base", "--is-ancestor", freeze_commit, "HEAD"], cwd=REPO_ROOT).returncode: raise RuntimeError("source freeze commit is not an ancestor of HEAD")
    for relative, expected in freeze["files"].items():
        path = (REPO_ROOT / relative).resolve(strict=True)
        if path.relative_to(REPO_ROOT).as_posix() != relative or sha256_file(path) != expected: raise RuntimeError(f"source freeze mismatch: {relative}")
        if subprocess.run(["git", "diff", "--quiet", "HEAD", "--", relative], cwd=REPO_ROOT).returncode: raise RuntimeError(f"linked worktree path is dirty: {relative}")
    if sha256_file(config_path) != freeze["files"][config_path.relative_to(REPO_ROOT).as_posix()]: raise RuntimeError("config is not frozen")
    observed = {}
    for name, binding in config["source_bindings"].items():
        path = (REPO_ROOT / binding["path"]).resolve(strict=True)
        actual = sha256_file(path)
        if actual != binding["sha256"]: raise RuntimeError(f"historical source mismatch: {name}")
        observed[name] = {"path": binding["path"], "sha256": actual, "bytes": path.stat().st_size}
    return {"schema_version": "proportional-physical-bindings-v1", "source_freeze_sha256": sha256_file(freeze_path), "source_freeze_commit": freeze_commit, "implementation_commit": freeze["implementation_commit"], "historical_sources": observed}


def canonical_role(data: dict[str, np.ndarray], mask: np.ndarray, role: str) -> dict[str, np.ndarray]:
    result = {}
    for key in ("pair_token", "cluster_id", "design_stratum", "cardinality", "ensemble_logits", "per_seed_logits", "target"):
        value = np.asarray(data[key]); selected = value[:, mask] if key == "per_seed_logits" else value[mask]
        if key in {"pair_token", "cluster_id"}: selected = selected.astype("<U64")
        elif key == "design_stratum": selected = selected.astype("<U16")
        elif key == "cardinality": selected = selected.astype("<i8")
        elif key in {"ensemble_logits", "per_seed_logits"}: selected = selected.astype("<f8")
        elif key == "target": selected = selected.astype(bool)
        result[key] = np.ascontiguousarray(selected)
    result["split_role"] = np.full(len(result["pair_token"]), role, dtype="<U24")
    return result


def canonical_full(data: dict[str, np.ndarray], role: str) -> dict[str, np.ndarray]:
    return canonical_role(data, np.ones(len(data["pair_token"]), dtype=bool), role)


def public_view(full: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {key: full[key].copy() for key in ("pair_token", "design_stratum", "cardinality", "ensemble_logits", "per_seed_logits")}


def truth_view(full: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {"pair_token": full["pair_token"].copy(), "target": full["target"].copy()}


def validate_role(name: str, data: dict[str, np.ndarray], expected: int, truth: bool = True) -> None:
    n = len(data["pair_token"])
    if n != expected or len(np.unique(data["pair_token"])) != n: raise RuntimeError(f"{name} count/identity drifted")
    if data["pair_token"].dtype != np.dtype("<U64") or data["design_stratum"].dtype != np.dtype("<U16") or data["cardinality"].dtype != np.dtype("<i8"): raise RuntimeError(f"{name} canonical dtype drifted")
    if data["ensemble_logits"].dtype != np.dtype("<f8") or data["per_seed_logits"].dtype != np.dtype("<f8") or data["ensemble_logits"].shape != (n, 4) or data["per_seed_logits"].shape != (3, n, 4): raise RuntimeError(f"{name} logits shape/dtype drifted")
    if not np.array_equal(data["ensemble_logits"], np.mean(data["per_seed_logits"], axis=0, dtype=np.float64)): raise RuntimeError("LOGIT_ENSEMBLE_MISMATCH")
    if set(data["design_stratum"].astype(str)) != {"FAR_RIVAL", "NEAR_RIVAL"}: raise RuntimeError("STRATUM_VOCABULARY_INVALID")
    if truth:
        if data["target"].shape != (n, 4) or data["target"].dtype != bool or not np.array_equal(data["cardinality"], data["target"].sum(axis=1).astype("<i8")): raise RuntimeError("CARDINALITY_TARGET_MISMATCH")
        if not np.array_equal(data["cluster_id"], data["pair_token"]): raise RuntimeError("TOKEN_IDENTITY_INVALID")


def validate_input_package(config: dict[str, Any], bindings: dict[str, Any], input_path: Path) -> Path:
    expected_files = {
        "opened_fixture_escrow.json": 0o400,
        "preparation_freeze.json": 0o444,
        "preparation_receipt.json": 0o444,
        "public_manifest.json": 0o444,
        "journals/prepare.json": 0o444,
        "prepared/public/decision_select_public.npz": 0o444,
        "prepared/public/evaluate_public.npz": 0o444,
        "prepared/public/utilities.npy": 0o444,
        "prepared/truth/decision_select_truth.npz": 0o400,
        "prepared/truth/evaluate_truth.npz": 0o400,
        "prepared/truth/policy_fit_truth.npz": 0o400,
        "prepared/truth/posterior_fit_truth.npz": 0o400,
    }
    expected_directories = {".": 0o700, "journals": 0o555, "prepared": 0o500, "prepared/public": 0o555, "prepared/truth": 0o500}
    actual_files = {path.relative_to(input_path).as_posix(): path for path in input_path.rglob("*") if path.is_file()}
    actual_dirs = {".": input_path, **{path.relative_to(input_path).as_posix(): path for path in input_path.rglob("*") if path.is_dir()}}
    all_objects = {path.relative_to(input_path).as_posix() for path in input_path.rglob("*")}
    if set(actual_files) != set(expected_files) or set(actual_dirs) != set(expected_directories) or all_objects != set(expected_files) | (set(expected_directories) - {"."}):
        raise RuntimeError("existing input package inventory drifted")
    for relative, mode in expected_directories.items():
        path = actual_dirs[relative]; info = path.lstat()
        if path.is_symlink() or not stat.S_ISDIR(info.st_mode) or stat.S_IMODE(info.st_mode) != mode or info.st_uid != 0 or info.st_gid != 0:
            raise RuntimeError(f"existing input directory metadata drifted: {relative}")
    for relative, mode in expected_files.items():
        path = actual_files[relative]; info = path.lstat()
        if path.is_symlink() or not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or stat.S_IMODE(info.st_mode) != mode or info.st_uid != 0 or info.st_gid != 0:
            raise RuntimeError(f"existing input file metadata drifted: {relative}")
    prep = read_json(input_path / "preparation_freeze.json")
    escrow = read_json(input_path / "opened_fixture_escrow.json")
    public = read_json(input_path / "public_manifest.json")
    receipt = read_json(input_path / "preparation_receipt.json")
    journal = read_json(input_path / "journals/prepare.json")
    prepared = {relative: {"bytes": actual_files[relative].stat().st_size, "sha256": sha256_file(actual_files[relative])} for relative in expected_files if relative.startswith("prepared/")}
    package_id = hashlib.sha256(json_bytes({"source_freeze": bindings["source_freeze_sha256"], "files": prepared})).hexdigest()
    roles = config["opened_fixture_roles"]
    expected_prep = {"schema_version": "proportional-physical-preparation-freeze-v1", "status": "OPENED_DATA_PHYSICAL_PREFLIGHT", "package_id": package_id, "source_freeze_sha256": bindings["source_freeze_sha256"], "checkpoint_axis": config["checkpoint_epochs"], "files": {name: row["sha256"] for name, row in prepared.items()}, "prospective_evidence": False}
    if prep != expected_prep: raise RuntimeError("existing preparation freeze drifted")
    if escrow != {"schema_version": "proportional-opened-fixture-escrow-v1", "status": "OPENED_DATA_PHYSICAL_PREFLIGHT", "prospective_evidence": False, "generation_escrow": False, "package_id": package_id, "historical_sources": bindings["historical_sources"], "prepared_files": prepared, "pairwise_overlap": escrow.get("pairwise_overlap")}:
        raise RuntimeError("existing opened escrow drifted")
    expected_public = {"schema_version": "proportional-physical-public-manifest-v1", "package_id": package_id, "roles": roles, "files": prepared, "truth_commitments_only": True}
    if public != expected_public: raise RuntimeError("existing public manifest drifted")
    if receipt != {"schema_version": "proportional-physical-preparation-receipt-v1", "package_id": package_id, "status": "PREPARED", "cross_array_checks": "PASS", "pairwise_disjoint": True}: raise RuntimeError("existing preparation receipt drifted")
    if journal != {"schema_version": "proportional-physical-journal-v1", "phase": "prepare", "previous_state": "INITIALIZED", "new_state": "PREPARED", "maximum_truth_materialized": "NONE", "package_id": package_id}: raise RuntimeError("existing preparation journal drifted")
    posterior = load_npz(input_path / "prepared/truth/posterior_fit_truth.npz")
    policy = load_npz(input_path / "prepared/truth/policy_fit_truth.npz")
    decision_public = load_npz(input_path / "prepared/public/decision_select_public.npz")
    decision_truth = load_npz(input_path / "prepared/truth/decision_select_truth.npz")
    evaluate_public = load_npz(input_path / "prepared/public/evaluate_public.npz")
    evaluate_truth = load_npz(input_path / "prepared/truth/evaluate_truth.npz")
    validate_role("posterior_fit", posterior, int(roles["posterior_fit"])); validate_role("policy_fit", policy, int(roles["policy_fit"]))
    for name, public_bundle, truth_bundle in (("decision_select", decision_public, decision_truth), ("evaluate", evaluate_public, evaluate_truth)):
        expected_public_keys = {"pair_token", "design_stratum", "cardinality", "ensemble_logits", "per_seed_logits"}
        if set(public_bundle) != expected_public_keys or set(truth_bundle) != {"pair_token", "target"}: raise RuntimeError(f"{name} view schema drifted")
        joined = {**public_bundle, "cluster_id": public_bundle["pair_token"], "target": truth_bundle["target"], "split_role": np.full(len(public_bundle["pair_token"]), name, dtype="<U24")}
        if not np.array_equal(public_bundle["pair_token"], truth_bundle["pair_token"]): raise RuntimeError(f"{name} public/truth identity drifted")
        validate_role(name, joined, int(roles[name]))
    token_sets = {name: set(bundle["pair_token"].astype(str)) for name, bundle in (("posterior", posterior), ("policy", policy), ("decision", decision_public), ("evaluate", evaluate_public))}
    overlaps = {f"{left}__{right}": len(token_sets[left] & token_sets[right]) for left, right in itertools.combinations(token_sets, 2)}
    if any(overlaps.values()) or escrow["pairwise_overlap"] != overlaps: raise RuntimeError("existing input role overlap drifted")
    utility = np.load(input_path / "prepared/public/utilities.npy", allow_pickle=False)
    if utility.dtype != np.dtype("<f8") or utility.shape != (24, 4) or not np.isfinite(utility).all(): raise RuntimeError("utility catalogue drifted")
    return input_path


def prepare_input(config: dict[str, Any], bindings: dict[str, Any], input_path: Path) -> Path:
    if input_path.exists():
        return validate_input_package(config, bindings, input_path)
    preparing = input_path.with_name(f"{input_path.name}.preparing")
    if preparing.exists(): archive_path(preparing, "failed")
    preparing.mkdir(parents=True, mode=0o700)
    public_dir = preparing / "prepared/public"; truth_dir = preparing / "prepared/truth"; journals = preparing / "journals"
    public_dir.mkdir(parents=True); truth_dir.mkdir(parents=True); journals.mkdir()
    source = {name: REPO_ROOT / row["path"] for name, row in config["source_bindings"].items()}
    w54 = load_npz(source["wave54_fit_select"])
    roles = np.asarray(w54["split_role"]).astype(str)
    posterior = canonical_role(w54, roles == "calibration_fit", "posterior_fit")
    evaluate = canonical_role(w54, roles == "decision_select", "evaluate_fixture")
    policy = canonical_full(load_npz(source["wave59_policy_fit"]), "policy_fit")
    decision = canonical_full(load_npz(source["wave59_decision_truth"]), "decision_select")
    expected = config["opened_fixture_roles"]
    for name, data in (("posterior_fit", posterior), ("policy_fit", policy), ("decision_select", decision), ("evaluate", evaluate)):
        validate_role(name, data, int(expected[name]))
    reference = load_npz(source["wave59_decision_public_reference"])
    if not np.array_equal(reference["pair_token"].astype(str), decision["pair_token"].astype(str)): raise RuntimeError("decision public reference identity drifted")
    token_sets = {name: set(data["pair_token"].astype(str)) for name, data in (("posterior", posterior), ("policy", policy), ("decision", decision), ("evaluate", evaluate))}
    overlaps = {f"{left}__{right}": len(token_sets[left] & token_sets[right]) for left, right in itertools.combinations(token_sets, 2)}
    if any(overlaps.values()): raise RuntimeError(f"physical role overlap: {overlaps}")
    bundles = {
        truth_dir / "posterior_fit_truth.npz": posterior,
        truth_dir / "policy_fit_truth.npz": policy,
        public_dir / "decision_select_public.npz": public_view(decision),
        truth_dir / "decision_select_truth.npz": truth_view(decision),
        public_dir / "evaluate_public.npz": public_view(evaluate),
        truth_dir / "evaluate_truth.npz": truth_view(evaluate),
    }
    for path, arrays in bundles.items(): write_npz(path, arrays, 0o400 if truth_dir in path.parents else 0o444)
    policy_manifest = read_json(source["policy_manifest"])
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from geometria_proporcional.proportional_set_valued_native import utilities_from_manifest
    utilities = utilities_from_manifest(policy_manifest)
    write_npy(public_dir / "utilities.npy", utilities, 0o444)
    file_rows = {path.relative_to(preparing).as_posix(): {"bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in sorted(bundles)}
    file_rows["prepared/public/utilities.npy"] = {"bytes": (public_dir / "utilities.npy").stat().st_size, "sha256": sha256_file(public_dir / "utilities.npy")}
    package_id = hashlib.sha256(json_bytes({"source_freeze": bindings["source_freeze_sha256"], "files": file_rows})).hexdigest()
    escrow = {"schema_version": "proportional-opened-fixture-escrow-v1", "status": "OPENED_DATA_PHYSICAL_PREFLIGHT", "prospective_evidence": False, "generation_escrow": False, "package_id": package_id, "historical_sources": bindings["historical_sources"], "prepared_files": file_rows, "pairwise_overlap": overlaps}
    write_json(preparing / "opened_fixture_escrow.json", escrow, 0o400)
    public_manifest = {"schema_version": "proportional-physical-public-manifest-v1", "package_id": package_id, "roles": {"posterior_fit": len(posterior["pair_token"]), "policy_fit": len(policy["pair_token"]), "decision_select": len(decision["pair_token"]), "evaluate": len(evaluate["pair_token"])}, "files": file_rows, "truth_commitments_only": True}
    write_json(preparing / "public_manifest.json", public_manifest, 0o444)
    preparation = {"schema_version": "proportional-physical-preparation-freeze-v1", "status": "OPENED_DATA_PHYSICAL_PREFLIGHT", "package_id": package_id, "source_freeze_sha256": bindings["source_freeze_sha256"], "checkpoint_axis": [17, 29, 43], "files": {path: row["sha256"] for path, row in file_rows.items()}, "prospective_evidence": False}
    write_json(preparing / "preparation_freeze.json", preparation, 0o444)
    write_json(preparing / "preparation_receipt.json", {"schema_version": "proportional-physical-preparation-receipt-v1", "package_id": package_id, "status": "PREPARED", "cross_array_checks": "PASS", "pairwise_disjoint": True}, 0o444)
    write_json_exclusive(journals / "prepare.json", {"schema_version": "proportional-physical-journal-v1", "phase": "prepare", "previous_state": "INITIALIZED", "new_state": "PREPARED", "maximum_truth_materialized": "NONE", "package_id": package_id}, 0o444)
    for directory in (public_dir, journals): directory.chmod(0o555); fsync_dir(directory)
    truth_dir.chmod(0o500); fsync_dir(truth_dir)
    fsync_dir(preparing / "prepared"); (preparing / "prepared").chmod(0o500)
    fsync_dir(preparing); preparing.chmod(0o700)
    os.replace(preparing, input_path); fsync_dir(input_path.parent)
    return validate_input_package(config, bindings, input_path)


def phase_inputs(phase: str, input_package: Path, output: Path, config_path: Path, bindings_path: Path) -> dict[str, Path]:
    common = {"config.json": config_path, "bindings.json": bindings_path, "preparation_freeze.json": input_package / "preparation_freeze.json", "utilities.npy": input_package / "prepared/public/utilities.npy"}
    truth = input_package / "prepared/truth"; public = input_package / "prepared/public"
    if phase == "posterior_fit": common["posterior_fit_truth.npz"] = truth / "posterior_fit_truth.npz"
    elif phase == "policy_fit":
        common["policy_fit_truth.npz"] = truth / "policy_fit_truth.npz"
        for name in HANDOFF["posterior_fit"]: common[name] = output / "posterior_fit" / name
    elif phase == "selection_propose":
        common["decision_select_public.npz"] = public / "decision_select_public.npz"
        for prior in ("posterior_fit", "policy_fit"):
            for name in HANDOFF[prior]: common[name] = output / prior / name
    elif phase == "selection_evaluate":
        common["decision_select_truth.npz"] = truth / "decision_select_truth.npz"
        for name in ("selection_key_metadata.json", "candidate_public.npz", "candidate_freeze.json"): common[name] = output / "selection_propose" / name
    elif phase == "selection_freeze":
        common["decision_select_public.npz"] = public / "decision_select_public.npz"
        for prior in ("posterior_fit", "policy_fit", "selection_propose", "selection_evaluate"):
            for name in HANDOFF[prior]: common[name] = output / prior / name
    elif phase == "evaluation_apply":
        common["evaluate_public.npz"] = public / "evaluate_public.npz"
        for prior in ("posterior_fit", "policy_fit", "selection_freeze"):
            for name in HANDOFF[prior]: common[name] = output / prior / name
    elif phase == "evaluation_truth":
        common["evaluate_truth.npz"] = truth / "evaluate_truth.npz"
        for name in HANDOFF["evaluation_apply"]: common[name] = output / "evaluation_apply" / name
    return common


def probe_paths(phase: str, input_package: Path, output: Path) -> list[str]:
    truth = input_package / "prepared/truth"
    mapping = {
        "posterior_fit": [truth / "policy_fit_truth.npz", truth / "decision_select_truth.npz", truth / "evaluate_truth.npz"],
        "policy_fit": [truth / "posterior_fit_truth.npz", truth / "decision_select_truth.npz", truth / "evaluate_truth.npz"],
        "selection_propose": [truth / name for name in ("posterior_fit_truth.npz", "policy_fit_truth.npz", "decision_select_truth.npz", "evaluate_truth.npz")],
        "selection_evaluate": [output / "posterior_fit/posterior_states.json", output / "policy_fit/policy_states.json", truth / "evaluate_truth.npz"],
        "selection_freeze": [truth / "decision_select_truth.npz", output / "selection_evaluate/candidate_metrics_private.npz", truth / "evaluate_truth.npz"],
        "evaluation_apply": [truth / "decision_select_truth.npz", output / "selection_evaluate/candidate_metrics_private.npz", truth / "evaluate_truth.npz"],
        "evaluation_truth": [output / "posterior_fit/posterior_states.json", output / "policy_fit/policy_states.json", truth / "decision_select_truth.npz", output / "selection_freeze/selection_policy.json"],
    }
    return [str(path.resolve()) for path in mapping[phase]]


def stage_runtime(envelope: Path, freeze: dict[str, Any]) -> Path:
    runtime = envelope / "runtime"; package = runtime / "geometria_proporcional"; package.mkdir(parents=True)
    for relative in RUNTIME_SOURCES:
        source = REPO_ROOT / relative
        destination = runtime / WORKER_SOURCE.name if relative.startswith("experiments/") else package / Path(relative).name
        shutil.copyfile(source, destination); destination.chmod(0o444)
        if sha256_file(destination) != freeze["files"][relative]: raise RuntimeError(f"staged runtime mismatch: {relative}")
    package.chmod(0o555); runtime.chmod(0o555)
    return runtime


def process_rss(pid: int) -> int:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith(("VmRSS:", "VmHWM:")): return int(line.split()[1]) * 1024
    except (FileNotFoundError, ProcessLookupError): pass
    return 0


def run_phase(phase: str, input_package: Path, output: Path, config_path: Path, bindings_path: Path, source_freeze: dict[str, Any], config: dict[str, Any], inject_promotion: str | None, inject_journal: str | None) -> dict[str, Any]:
    staging_root = REPO_ROOT / ".physical_set_valued_stages"
    staging_root.mkdir(mode=0o711, exist_ok=True)
    staging_root.chmod(0o711)
    envelope = Path(tempfile.mkdtemp(prefix=f"physical-{phase}-", dir=staging_root)); envelope.chmod(0o711)
    stage = envelope / "stage"; scratch = envelope / "scratch"; stage.mkdir(); scratch.mkdir()
    nobody = pwd.getpwnam("nobody"); os.chown(scratch, nobody.pw_uid, nobody.pw_gid); scratch.chmod(0o700)
    runtime = stage_runtime(envelope, source_freeze)
    inputs = phase_inputs(phase, input_package, output, config_path, bindings_path)
    for name, source in inputs.items(): shutil.copyfile(source, stage / name); (stage / name).chmod(0o444)
    request = {"schema_version": "proportional-physical-phase-request-v1", "phase": phase, "allowed_files": sorted([*inputs, "phase_request.json"]), "sha256": {name: sha256_file(stage / name) for name in sorted(inputs)}, "expected_outputs": list(EXPECTED_OUTPUTS[phase]), "runtime_modules": list(RUNTIME_MODULES), "environment_keys": list(ENVIRONMENT_KEYS), "probe_paths": probe_paths(phase, input_package, output)}
    write_json(stage / "phase_request.json", request, 0o444); stage.chmod(0o555)
    environment = {"PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "PYTHONPATH": str(runtime), "PYTHONNOUSERSITE": "1", "PYTHONHASHSEED": "0", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1", "KMP_DUPLICATE_LIB_OK": "True", "KMP_INIT_AT_FORK": "FALSE", "CUDA_VISIBLE_DEVICES": "", "PHIDEUS_STAGED_RUNTIME": "1"}
    command = ["setpriv", "--reuid=65534", "--regid=65534", "--clear-groups", "--no-new-privs", "--bounding-set=-all", "--inh-caps=-all", "--ambient-caps=-all", str(PYTHON), "-s", "-P", str(runtime / WORKER_SOURCE.name), "--phase", phase, "--stage", str(stage), "--output", str(scratch)]
    started = time.monotonic(); process = subprocess.Popen(command, cwd=stage, env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    peak = 0; deadline = float(config["phase_wall_seconds"][phase])
    while process.poll() is None:
        peak = max(peak, process_rss(process.pid))
        if time.monotonic() - started > deadline:
            process.kill(); process.wait(); raise TimeoutError(f"phase wall budget exceeded: {phase}")
        time.sleep(0.02)
    stdout, stderr = process.communicate(); elapsed = time.monotonic() - started; peak = max(peak, process_rss(process.pid))
    if process.returncode: raise RuntimeError(f"worker {phase} failed ({process.returncode}): {stderr[-4000:]}")
    actual = {path.name for path in scratch.iterdir()}
    expected = set(EXPECTED_OUTPUTS[phase]) | {"worker_receipt.json"}
    if actual != expected or any(path.is_symlink() or not path.is_file() for path in scratch.iterdir()): raise RuntimeError(f"worker output inventory drifted: {phase}")
    receipt = read_json(scratch / "worker_receipt.json")
    peak = max(peak, int(receipt["peak_rss_bytes"]))
    if peak > int(config["budgets"]["per_process_rss_bytes"]): raise RuntimeError(f"worker RSS budget exceeded: {phase}")
    pending = output / f".{phase}.pending"
    if pending.exists(): archive_path(pending, "failed")
    shutil.copytree(scratch, pending)
    for path in pending.rglob("*"):
        if path.is_file(): os.chown(path, 0, 0); path.chmod(0o400 if path.name in PRIVATE_NAMES else 0o444)
    for directory in sorted((path for path in pending.rglob("*") if path.is_dir()), reverse=True): os.chown(directory, 0, 0); directory.chmod(0o500); fsync_dir(directory)
    os.chown(pending, 0, 0); pending.chmod(0o500); fsync_dir(pending)
    os.replace(pending, output / phase); fsync_dir(output)
    if inject_promotion == phase:
        shutil.rmtree(envelope)
        raise RuntimeError(f"INJECTED_CRASH_AFTER_PROMOTION:{phase}")
    previous = "PREPARED" if phase == PHASES[0] else STATE_AFTER[PHASES[PHASES.index(phase) - 1]]
    journal = {"schema_version": "proportional-physical-journal-v1", "phase": phase, "previous_state": previous, "new_state": STATE_AFTER[phase], "maximum_truth_materialized": TRUTH_LEVEL[phase], "package_id": read_json(input_package / "preparation_freeze.json")["package_id"], "input_hashes": request["sha256"], "output_hashes": {name: sha256_file(output / phase / name) for name in sorted(actual)}, "worker_receipt_sha256": sha256_file(output / phase / "worker_receipt.json"), "wall_seconds": elapsed, "peak_rss_bytes": peak}
    write_json_exclusive(output / "journals" / f"{phase}.json", journal, 0o444)
    if inject_journal == phase:
        shutil.rmtree(envelope)
        raise RuntimeError(f"INJECTED_CRASH_AFTER_JOURNAL:{phase}")
    shutil.rmtree(envelope)
    return {"phase": phase, "wall_seconds": elapsed, "peak_rss_bytes": peak, "stdout": stdout.strip()}


def validate_completed_prefix(output: Path, package_id: str, config: dict[str, Any], bindings: dict[str, Any]) -> int:
    journals = output / "journals"
    if read_json(output / "config.snapshot.json") != config or read_json(output / "bindings.json") != bindings: raise RuntimeError("resume authority snapshot drifted")
    actual_journals = {path.name for path in journals.iterdir()} if journals.exists() else set()
    phase_presence = [bool((output / phase).exists()) for phase in PHASES]
    journal_presence = [f"{phase}.json" in actual_journals for phase in PHASES]
    if actual_journals - {f"{phase}.json" for phase in PHASES}: raise RuntimeError("unknown journal exists")
    complete = 0; previous = "PREPARED"
    for index, phase in enumerate(PHASES):
        phase_dir = output / phase; journal = journals / f"{phase}.json"
        if phase_dir.exists() != journal.exists(): raise RuntimeError(f"orphan phase or journal: {phase}")
        if not phase_dir.exists():
            if any(phase_presence[index + 1:]) or any(journal_presence[index + 1:]): raise RuntimeError("future phase or journal exists")
            break
        payload = read_json(journal)
        receipt = read_json(phase_dir / "worker_receipt.json")
        if payload.get("schema_version") != "proportional-physical-journal-v1" or payload.get("phase") != phase or payload.get("previous_state") != previous or payload.get("new_state") != STATE_AFTER[phase] or payload.get("maximum_truth_materialized") != TRUTH_LEVEL[phase] or payload.get("package_id") != package_id or payload.get("worker_receipt_sha256") != sha256_file(phase_dir / "worker_receipt.json") or payload.get("input_hashes") != receipt.get("stage_files") or set(payload["output_hashes"]) != {path.name for path in phase_dir.iterdir()} or any(sha256_file(phase_dir / name) != digest for name, digest in payload["output_hashes"].items()): raise RuntimeError(f"completed phase invalid: {phase}")
        previous = STATE_AFTER[phase]
        complete += 1
    return complete


def artifact_class(relative: str) -> str:
    name = Path(relative).name
    if name in PRIVATE_NAMES: return "fit_private_audit" if relative.startswith(("posterior_fit/", "policy_fit/")) else ("selection_private_audit" if relative.startswith("selection_evaluate/") else "evaluation_private_audit")
    if relative.startswith("journals/"): return "journal"
    if name.endswith("freeze.json"): return "freeze"
    if name == "worker_receipt.json": return "runtime_receipt"
    if name == "REPORT.md": return "regenerable_report"
    if name in {"config.snapshot.json", "bindings.json"}: return "source_snapshot"
    if name == "artifact_manifest.json": return "self_reference"
    return "public_handoff" if relative.split("/", 1)[0] in PHASES[:-1] else "derived_diagnostic"


def build_manifest(output: Path) -> dict[str, Any]:
    entries = []
    for path in sorted(output.rglob("*")):
        relative = path.relative_to(output).as_posix(); info = path.lstat()
        entries.append({"path": relative, "type": "directory" if path.is_dir() else "file", "class": artifact_class(relative), "bytes": int(info.st_size) if path.is_file() else 0, "sha256": sha256_file(path) if path.is_file() and path.name != "artifact_manifest.json" else None, "mode": stat.S_IMODE(info.st_mode), "uid": info.st_uid, "gid": info.st_gid, "phase": relative.split("/", 1)[0]})
    return {"schema_version": "proportional-physical-artifact-manifest-v1", "self_excluded": True, "files_and_directories": entries}


def compare_reference(output: Path, reference: Path | None) -> dict[str, Any]:
    if reference is None: return {"schema_version": "proportional-physical-replay-receipt-v2", "mode": "primary", "reference_supplied": False, "byte_exact": None, "excluded_paths": [], "excluded_fields": []}
    reference = reference.resolve(strict=True)
    deferred = {"artifact_manifest.json", "runtime.json", "replay_receipt.json", "recovery_origin.json"}
    def files(root: Path) -> dict[str, Path]: return {path.relative_to(root).as_posix(): path for path in root.rglob("*") if path.is_file() and path.relative_to(root).as_posix() not in deferred}
    def normalize(relative: str, payload: Any) -> Any:
        payload = json.loads(json.dumps(payload))
        if relative.endswith("/worker_receipt.json"):
            payload.pop("wall_seconds", None); payload.pop("peak_rss_bytes", None)
            runtime = payload["runtime"]; runtime["cwd"] = "$STAGE"; runtime["stage_metadata"]["cwd"] = "$STAGE"; runtime["stage_metadata"]["files"]["phase_request.json"]["sha256"] = "$PHASE_REQUEST"; runtime["environment"]["PYTHONPATH"] = "$RUNTIME"; runtime["sys_path"][0] = "$RUNTIME"
            for row in runtime["modules"].values(): row["path"] = f"$RUNTIME/{Path(row['path']).name}"
            for row in runtime["runtime_files"].values(): row["path"] = f"$RUNTIME/{Path(row['path']).name}"
            for row in payload["probes"]: row["path_sha256"] = "$ABSOLUTE_PROBE"
            payload["stage_contract"]["probe_path_sha256"] = ["$ABSOLUTE_PROBE"] * len(payload["stage_contract"]["probe_path_sha256"])
        elif relative.startswith("journals/"):
            payload.pop("wall_seconds", None); payload.pop("peak_rss_bytes", None); payload.pop("worker_receipt_sha256", None); payload["output_hashes"]["worker_receipt.json"] = "$WORKER_RECEIPT"
        return payload
    current, prior = files(output), files(reference)
    if set(current) != set(prior): raise RuntimeError("replay normalized inventory differs")
    normalized = 0
    for relative in current:
        if relative.endswith(".json"):
            left = json_bytes(normalize(relative, read_json(current[relative]))); right = json_bytes(normalize(relative, read_json(prior[relative]))); normalized += 1
        else: left = current[relative].read_bytes(); right = prior[relative].read_bytes()
        if left != right: raise RuntimeError(f"replay normalized bytes differ: {relative}")
    return {"schema_version": "proportional-physical-replay-receipt-v2", "mode": "replay", "reference_supplied": True, "compared_files": len(current), "normalized_json_files": normalized, "byte_exact": True, "excluded_paths": sorted(deferred), "excluded_fields": ["absolute_probe_path_hashes", "peak_rss_bytes", "phase_request_sha256", "runtime_stage_paths", "wall_seconds", "worker_receipt_sha256", "worker_receipt_output_hash"], "semantic_exclusions_valid": True, "reference_manifest_sha256": sha256_file(reference / "artifact_manifest.json")}


def r564_parity(output: Path, config: dict[str, Any]) -> dict[str, Any]:
    root = REPO_ROOT / "data/geometria_proporcional/proportional_set_valued_native_preflight_v1"
    checks = []
    def same_file(current: Path, old: Path, label: str) -> None:
        equal = sha256_file(current) == sha256_file(old); checks.append({"object": label, "comparator": "byte_exact", "equal": equal})
        if not equal: raise RuntimeError(f"R564 parity mismatch: {label}")
    same_file(output / "posterior_fit/posterior_states.json", root / "posterior_fit/states.json", "posterior_states")
    same_file(output / "posterior_fit/posterior_state_arrays.npz", root / "posterior_fit/state_arrays.npz", "posterior_state_arrays")
    same_file(output / "posterior_fit/posterior_oof_arrays.npz", root / "posterior_fit/oof_arrays.npz", "posterior_oof_arrays")
    same_file(output / "posterior_fit/target_shuffle_map.json", root / "posterior_fit/target_shuffle_map.json", "target_shuffle_map")
    same_file(output / "posterior_fit/target_shuffle_arrays.npz", root / "posterior_fit/target_shuffle_arrays.npz", "target_shuffle_arrays")
    same_file(output / "policy_fit/feature_schema.json", root / "policy_fit/feature_schema.json", "feature_schema")
    same_file(output / "policy_fit/policy_states.json", root / "policy_fit/states.json", "policy_states")
    same_file(output / "policy_fit/policy_state_arrays.npz", root / "policy_fit/state_arrays.npz", "policy_state_arrays")
    same_file(output / "policy_fit/control_maps.json", root / "policy_fit/control_maps.json", "control_maps")
    same_file(output / "policy_fit/control_arrays.npz", root / "policy_fit/control_arrays.npz", "control_arrays")
    old_candidate = load_npz(root / "decision_select/candidate_metrics.npz"); candidate = load_npz(output / "selection_propose/candidate_public.npz"); private = load_npz(output / "selection_evaluate/candidate_metrics_private.npz")
    mappings = {}
    for name in ("marginal", "joint"):
        mappings[f"{name}__actions"] = candidate[f"{name}__actions"]
        mappings[f"{name}__override"] = candidate[f"{name}__override"]
        for metric in ("mean_regret", "incompatibility_rate", "harm_rate", "authorized_rows"): mappings[f"{name}__{metric}"] = private[f"{name}__{metric}"]
    for key, value in mappings.items():
        equal = np.array_equal(value, old_candidate[key]); checks.append({"object": f"candidate_metrics:{key}", "comparator": "array_equal", "equal": equal})
        if not equal: raise RuntimeError(f"R564 candidate parity mismatch: {key}")
    old_freeze = read_json(root / "decision_select/selection_freeze.json"); policy = read_json(output / "selection_freeze/selection_policy.json"); decision = read_json(output / "selection_evaluate/selection_decision.json")
    for name in ("marginal", "joint"):
        fields = {"selected_index": policy["posteriors"][name]["selected_index"], "selected": {**policy["posteriors"][name]["selected"], **{key: decision["posteriors"][name][key] for key in ("mean_regret", "incompatibility_rate", "harm_rate", "authorized_rows")}}, "control_thresholds": {str(row["seed"]): row["thresholds"] for row in policy["posteriors"][name]["controls"]}, "u_true_count": policy["posteriors"][name]["u_true_count"], "u_common_count": policy["posteriors"][name]["u_common_count"], "common_coverage": policy["posteriors"][name]["common_coverage"], "common_support_status": policy["posteriors"][name]["common_support_status"]}
        for key, value in fields.items():
            equal = value == old_freeze[name][key]; checks.append({"object": f"selection:{name}:{key}", "comparator": "json_exact", "equal": equal})
            if not equal: raise RuntimeError(f"R564 selection parity mismatch: {name}/{key}")
    return {"schema_version": "proportional-physical-r564-parity-v1", "r564_commit": "f7ad9227868f83f381ebbc0a8995fefa5a1a272f", "r564_report_sha256": "2221b3938b03728e28133ac0ac5b05918c56b5a62845966293ac113aea4479cb", "checks": checks, "passed": sum(row["equal"] for row in checks), "total": len(checks), "all_exact": True}


def write_report(output: Path) -> None:
    estimands = read_json(output / "evaluation_truth/estimand_table.json")
    lines = ["# Preflight físico CPU de la rama set-valued", "", "Estado: `PHYSICAL_PROSPECTIVE_PACKAGE_PREFLIGHT_VALID`.", "", "El paquete ejercita siete procesos separados sobre cuatro poblaciones históricas abiertas. No crea un draw prospectivo, no usa monitor o lockbox, no promueve una arquitectura y no decide `GO/NO-GO`.", "", "| ID | Instancia | Estado | N | Media izquierda-derecha | CI95 |", "|---|---|---|---:|---:|---|"]
    for row in estimands["rows"]:
        lines.append(f"| {row['id']} | {row['instance']} | {row['status']} | {row.get('n_tokens', 0)} | {row.get('mean_diff', 'n/a')} | {('[%s, %s]' % (row['ci95_low'], row['ci95_high'])) if 'ci95_low' in row else 'n/a'} |")
    lines += ["", "Estos resultados sólo acreditan el preflight físico sobre datos abiertos (`prospective_evidence=false`).", ""]
    write_bytes(output / "REPORT.md", "\n".join(lines).encode("utf-8"), 0o444)


def main() -> int:
    args = parse_args()
    if args.execution_class == "FRESH_PROSPECTIVE":
        print(json.dumps({"status": "REJECTED", "reason_code": "FRESH_PROSPECTIVE_NOT_AUTHORIZED_V1"}, sort_keys=True)); return 65
    if args.execution_class != "OPENED_DATA_PHYSICAL_PREFLIGHT": raise ValueError("unknown execution class")
    if os.geteuid() != 0 or os.getegid() != 0: raise PermissionError("physical coordinator requires root")
    config_path = args.config.resolve(strict=True); freeze_path = args.source_freeze.resolve(strict=True); config = read_json(config_path)
    if config.get("schema_version") != "proportional-set-valued-physical-preflight-v1" or config.get("enabled_execution_class") != args.execution_class: raise RuntimeError("config authority drifted")
    source_bindings = source_preflight(config_path, freeze_path, config); source_freeze = read_json(freeze_path)
    input_package = ensure_child(args.input_package); output = ensure_child(args.output_dir)
    input_package.parent.mkdir(parents=True, exist_ok=True); output.parent.mkdir(parents=True, exist_ok=True)
    prepare_input(config, source_bindings, input_package)
    package_id = read_json(input_package / "preparation_freeze.json")["package_id"]
    archived = None
    if output.exists() and args.force: archived = archive_path(output)
    if output.exists() and not args.resume: raise FileExistsError(f"output exists: {output}")
    if not output.exists(): output.mkdir(mode=0o700); (output / "journals").mkdir(mode=0o700); write_json(output / "config.snapshot.json", config, 0o444); write_json(output / "bindings.json", source_bindings, 0o444)
    try: complete = validate_completed_prefix(output, package_id, config, source_bindings)
    except RuntimeError:
        if not args.resume: raise
        failed = archive_path(output, "failed"); output.mkdir(mode=0o700); (output / "journals").mkdir(mode=0o700); write_json(output / "config.snapshot.json", config, 0o444); write_json(output / "bindings.json", source_bindings, 0o444); complete = 0
        write_json(output / "recovery_origin.json", {"schema_version": "proportional-physical-recovery-origin-v1", "archived_inventory_sha256": hashlib.sha256("\n".join(sorted(path.relative_to(failed).as_posix() for path in failed.rglob("*"))).encode()).hexdigest(), "reason": "ORPHAN_OR_DIVERGENT_PHASE_ARCHIVED"}, 0o444)
    started = time.monotonic(); phase_rows = []
    for phase in PHASES[complete:]: phase_rows.append(run_phase(phase, input_package, output, config_path, output / "bindings.json", source_freeze, config, args.inject_crash_after_promotion, args.inject_crash_after_journal))
    parity = r564_parity(output, config); write_json(output / "r564_parity_receipt.json", parity, 0o444)
    write_report(output)
    replay = compare_reference(output, args.reference_dir); write_json(output / "replay_receipt.json", replay, 0o444)
    wall = time.monotonic() - started; peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    runtime = {"schema_version": "proportional-physical-runtime-v1", "status": "PHYSICAL_PROSPECTIVE_PACKAGE_PREFLIGHT_VALID", "execution_class": args.execution_class, "wall_seconds": wall, "coordinator_peak_rss_bytes": peak, "phases_executed": phase_rows, "phases_reused": list(PHASES[:complete]), "archived_previous_output": None if archived is None else str(archived), "fresh_draw_created_or_opened": False, "monitor_or_lockbox_opened": False, "gpu_used_or_queried": False, "torch_imported": False, "architecture_promoted": False, "scientific_decision": None, "decision_authority": "user", "prospective_evidence": False}
    write_json(output / "runtime.json", runtime, 0o444)
    write_json(output / "artifact_manifest.json", build_manifest(output), 0o444)
    if args.reference_dir:
        total = wall + float(read_json(Path(args.reference_dir) / "runtime.json")["wall_seconds"])
        if total > float(config["budgets"]["primary_plus_replay_seconds"]): raise RuntimeError("primary plus replay wall budget exceeded")
    print(json.dumps({"status": runtime["status"], "output": str(output), "wall_seconds": wall, "phases_executed": len(phase_rows), "replay": replay["byte_exact"], "r564_parity_checks": parity["total"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
