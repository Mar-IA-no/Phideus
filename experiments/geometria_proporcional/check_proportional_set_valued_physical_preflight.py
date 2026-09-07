#!/usr/bin/env python3
"""Independent checker for the physical set-valued CPU package."""

from __future__ import annotations

import argparse
import ast
import hashlib
import itertools
import json
import os
from pathlib import Path
import resource
import site
import stat
import subprocess
import sys
import sysconfig
import time
from typing import Any, Callable
import zipfile

import numpy as np
import scipy
import sklearn


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))
import check_proportional_set_valued_native_preflight as independent  # noqa: E402


CONFIG_DEFAULT = REPO_ROOT / "experiments/geometria_proporcional/configs/proportional_set_valued_physical_preflight_v1.json"
FREEZE_DEFAULT = REPO_ROOT / "experiments/geometria_proporcional/configs/proportional_set_valued_physical_source_freeze_v1.json"
INPUT_DEFAULT = REPO_ROOT / "data/geometria_proporcional/proportional_set_valued_physical_input_v1"
PHASES = (
    "posterior_fit", "policy_fit", "selection_propose", "selection_evaluate",
    "selection_freeze", "evaluation_apply", "evaluation_truth",
)
STATES = (
    "POSTERIOR_FIT_COMPLETE", "POLICY_FIT_COMPLETE", "SELECTION_CANDIDATES_FROZEN",
    "SELECTION_DECISION_FROZEN", "SELECTION_POLICY_FROZEN",
    "EVALUATION_ACTIONS_FROZEN", "COMPLETE",
)
REASONS = {
    "P1_AUTHORITY": "AUTHORITY_INVALID", "P2_PREPARATION": "PREPARATION_INVALID",
    "P3_PHYSICAL_BOUNDARY": "PHYSICAL_BOUNDARY_INVALID", "P4_STATE_MACHINE": "STATE_MACHINE_INVALID",
    "P5_POSTERIOR": "POSTERIOR_INVALID", "P6_POLICY": "POLICY_INVALID",
    "P7_SELECTION_PROPOSE": "SELECTION_PROPOSE_INVALID", "P8_SELECTION_EVALUATE": "SELECTION_EVALUATE_INVALID",
    "P9_SELECTION_FREEZE": "SELECTION_FREEZE_INVALID", "P10_EVALUATION_APPLY": "EVALUATION_APPLY_INVALID",
    "P11_EVALUATION_TRUTH": "EVALUATION_TRUTH_INVALID", "P12_RESTART_REPLAY": "RESTART_OR_REPLAY_INVALID",
    "P13_INVENTORY": "INVENTORY_INVALID", "P14_SCOPE": "SCOPE_INVALID", "P15_COST": "COST_INVALID",
}
TRUTH_FORBIDDEN = ("target", "truth", "oracle", "label", "gain", "regret", "harm", "compatible", "authorized")
RUNTIME_SOURCES = {
    "_proportional_set_valued_phase_worker.py": "experiments/geometria_proporcional/_proportional_set_valued_phase_worker.py",
    "__init__.py": "src/geometria_proporcional/__init__.py",
    "wave49_schema.py": "src/geometria_proporcional/wave49_schema.py",
    "proportional_set_valued_native.py": "src/geometria_proporcional/proportional_set_valued_native.py",
    "wave53_uncertainty.py": "src/geometria_proporcional/wave53_uncertainty.py",
    "wave54_joint_set.py": "src/geometria_proporcional/wave54_joint_set.py",
}
RUNTIME_MODULES = {
    "geometria_proporcional": "__init__.py",
    "geometria_proporcional.proportional_set_valued_native": "proportional_set_valued_native.py",
    "geometria_proporcional.wave49_schema": "wave49_schema.py",
    "geometria_proporcional.wave53_uncertainty": "wave53_uncertainty.py",
    "geometria_proporcional.wave54_joint_set": "wave54_joint_set.py",
}
ENVIRONMENT = {
    "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
    "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "PYTHONNOUSERSITE": "1", "PYTHONHASHSEED": "0",
    "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
    "KMP_DUPLICATE_LIB_OK": "True", "KMP_INIT_AT_FORK": "FALSE",
    "CUDA_VISIBLE_DEVICES": "", "PHIDEUS_STAGED_RUNTIME": "1",
}
ENVIRONMENT_KEYS = ("PATH", "LANG", "LC_ALL", "PYTHONPATH", "PYTHONNOUSERSITE", "PYTHONHASHSEED", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "KMP_DUPLICATE_LIB_OK", "KMP_INIT_AT_FORK", "CUDA_VISIBLE_DEVICES", "PHIDEUS_STAGED_RUNTIME")
PRIVATE_NAMES = frozenset({"posterior_oof_arrays.npz", "target_shuffle_map.json", "target_shuffle_arrays.npz", "policy_fit_private.npz", "control_maps.json", "control_arrays.npz", "candidate_metrics_private.npz", "selection_target_aligned_private.npz", "diagnostic_arrays.npz", "bootstrap_indices.npz"})
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
HISTORICAL_REFERENCE_ROOT = REPO_ROOT / "data/geometria_proporcional/proportional_set_valued_native_preflight_v1"


class CheckFailure(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--input-package", type=Path, default=INPUT_DEFAULT)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--config", type=Path, default=CONFIG_DEFAULT)
    parser.add_argument("--source-freeze", type=Path, default=FREEZE_DEFAULT)
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--only", choices=tuple(REASONS))
    return parser.parse_args()


def reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant forbidden: {value}")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)


def json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def normalized_replay_json(relative: str, payload: Any) -> Any:
    payload = json.loads(json.dumps(payload))
    if relative.endswith("/worker_receipt.json"):
        payload.pop("wall_seconds", None); payload.pop("peak_rss_bytes", None)
        runtime = payload["runtime"]
        runtime["cwd"] = "$STAGE"; runtime["stage_metadata"]["cwd"] = "$STAGE"
        runtime["environment"]["PYTHONPATH"] = "$RUNTIME"
        runtime["sys_path"][0] = "$RUNTIME"
        runtime["stage_metadata"]["files"]["phase_request.json"]["sha256"] = "$PHASE_REQUEST"
        for row in runtime["modules"].values(): row["path"] = f"$RUNTIME/{Path(row['path']).name}"
        for row in runtime["runtime_files"].values(): row["path"] = f"$RUNTIME/{Path(row['path']).name}"
        for row in payload["probes"]: row["path_sha256"] = "$ABSOLUTE_PROBE"
        payload["stage_contract"]["probe_path_sha256"] = ["$ABSOLUTE_PROBE"] * len(payload["stage_contract"]["probe_path_sha256"])
    elif relative.startswith("journals/"):
        payload.pop("wall_seconds", None); payload.pop("peak_rss_bytes", None); payload.pop("worker_receipt_sha256", None); payload["output_hashes"]["worker_receipt.json"] = "$WORKER_RECEIPT"
    elif relative == "runtime.json":
        expected_keys = {"schema_version", "status", "execution_class", "wall_seconds", "coordinator_peak_rss_bytes", "phases_executed", "phases_reused", "archived_previous_output", "fresh_draw_created_or_opened", "monitor_or_lockbox_opened", "gpu_used_or_queried", "torch_imported", "architecture_promoted", "scientific_decision", "decision_authority", "prospective_evidence"}
        if set(payload) != expected_keys or payload["schema_version"] != "proportional-physical-runtime-v1": raise CheckFailure("runtime replay schema drifted")
        executed = payload.pop("phases_executed"); reused = payload.pop("phases_reused")
        if not isinstance(executed, list) or not isinstance(reused, list) or reused + [row.get("phase") for row in executed if isinstance(row, dict)] != list(PHASES) or len(set(reused)) != len(reused): raise CheckFailure("runtime phase coverage drifted")
        if not isinstance(payload["wall_seconds"], (int, float)) or not np.isfinite(payload["wall_seconds"]) or payload["wall_seconds"] < 0: raise CheckFailure("runtime wall drifted")
        if not isinstance(payload["coordinator_peak_rss_bytes"], int) or isinstance(payload["coordinator_peak_rss_bytes"], bool) or payload["coordinator_peak_rss_bytes"] < 0: raise CheckFailure("runtime RSS drifted")
        if payload["archived_previous_output"] is not None and not isinstance(payload["archived_previous_output"], str): raise CheckFailure("runtime archive field drifted")
        for row in executed:
            if set(row) != {"phase", "wall_seconds", "peak_rss_bytes", "stdout"}: raise CheckFailure("runtime phase row drifted")
            if not isinstance(row["wall_seconds"], (int, float)) or not np.isfinite(row["wall_seconds"]) or row["wall_seconds"] < 0 or not isinstance(row["peak_rss_bytes"], int) or isinstance(row["peak_rss_bytes"], bool) or row["peak_rss_bytes"] < 0 or not isinstance(row["stdout"], str): raise CheckFailure("runtime phase resources drifted")
            worker = json.loads(row["stdout"])
            if set(worker) != {"status", "phase", "wall_seconds", "peak_rss_bytes"} or worker["status"] != "PASS" or worker["phase"] != row["phase"]: raise CheckFailure("runtime worker stdout drifted")
            if not isinstance(worker["wall_seconds"], (int, float)) or not np.isfinite(worker["wall_seconds"]) or worker["wall_seconds"] < 0 or not isinstance(worker["peak_rss_bytes"], int) or isinstance(worker["peak_rss_bytes"], bool) or worker["peak_rss_bytes"] < 0: raise CheckFailure("runtime worker resources drifted")
        payload["phase_coverage"] = list(PHASES); payload["wall_seconds"] = "$WALL"; payload["coordinator_peak_rss_bytes"] = "$RSS"; payload["archived_previous_output"] = "$ARCHIVE"
    return payload


def replay_files(root: Path) -> dict[str, Path]:
    deferred = {"artifact_manifest.json", "replay_receipt.json", "recovery_origin.json"}
    return {path.relative_to(root).as_posix(): path for path in root.rglob("*") if path.is_file() and path.relative_to(root).as_posix() not in deferred}


def compare_replay_semantics(current_root: Path, reference_root: Path) -> tuple[int, int]:
    for root in (current_root, reference_root):
        origin = root / "recovery_origin.json"
        if origin.exists():
            value = read_json(origin)
            digest = value.get("archived_inventory_sha256")
            if set(value) != {"schema_version", "archived_inventory_sha256", "reason"} or value["schema_version"] != "proportional-physical-recovery-origin-v1" or value["reason"] != "ORPHAN_OR_DIVERGENT_PHASE_ARCHIVED" or not isinstance(digest, str) or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest): raise CheckFailure("recovery origin drifted")
    current, previous = replay_files(current_root), replay_files(reference_root)
    if set(current) != set(previous): raise CheckFailure("replay normalized inventory mismatch")
    normalized = 0
    for relative in current:
        if relative.endswith(".json"):
            left = json_bytes(normalized_replay_json(relative, read_json(current[relative])))
            right = json_bytes(normalized_replay_json(relative, read_json(previous[relative])))
            normalized += 1
        else:
            left = current[relative].read_bytes(); right = previous[relative].read_bytes()
        if left != right: raise CheckFailure(f"replay normalized mismatch: {relative}")
    return len(current), normalized


def assert_array(left: np.ndarray, right: np.ndarray, label: str) -> None:
    if left.dtype != right.dtype or left.shape != right.shape or not np.array_equal(left, right):
        raise CheckFailure(f"array mismatch: {label}")


def assert_close_array(left: np.ndarray, right: np.ndarray, label: str, atol: float = 5e-12, *, equal_nan: bool = False) -> None:
    try:
        np.testing.assert_allclose(left, right, rtol=0.0, atol=atol, equal_nan=equal_nan)
    except AssertionError as error:
        raise CheckFailure(f"array mismatch: {label}") from error


def assert_value(left: Any, right: Any, label: str) -> None:
    if left != right:
        raise CheckFailure(f"value mismatch: {label}")


def assert_nested_close(left: Any, right: Any, label: str, atol: float = 5e-12) -> None:
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right): raise CheckFailure(f"object keys mismatch: {label}")
        for key in left: assert_nested_close(left[key], right[key], f"{label}/{key}", atol)
    elif isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right): raise CheckFailure(f"list length mismatch: {label}")
        for index, (lvalue, rvalue) in enumerate(zip(left, right, strict=True)): assert_nested_close(lvalue, rvalue, f"{label}/{index}", atol)
    elif isinstance(left, float) and isinstance(right, (float, int)):
        if not np.isfinite(left) or not np.isfinite(right) or abs(left - right) > atol: raise CheckFailure(f"numeric mismatch: {label}")
    elif left != right:
        raise CheckFailure(f"value mismatch: {label}")


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def validate_historical_reference(freeze: dict[str, Any]) -> None:
    parity_entries = freeze.get("historical_reference", {}).get("parity_entries", {})
    if len(parity_entries) != 12:
        raise CheckFailure("historical parity digest inventory drifted")
    reference_root = HISTORICAL_REFERENCE_ROOT.resolve(strict=True)
    for relative, digest in parity_entries.items():
        path = (reference_root / relative).resolve(strict=True)
        if reference_root not in path.parents or sha256_file(path) != digest:
            raise CheckFailure(f"historical parity source drifted: {relative}")


def expected_artifact_class(relative: str) -> str:
    name = Path(relative).name
    if name in PRIVATE_NAMES: return "fit_private_audit" if relative.startswith(("posterior_fit/", "policy_fit/")) else ("selection_private_audit" if relative.startswith("selection_evaluate/") else "evaluation_private_audit")
    if relative.startswith("journals/"): return "journal"
    if name.endswith("freeze.json"): return "freeze"
    if name == "worker_receipt.json": return "runtime_receipt"
    if name == "REPORT.md": return "regenerable_report"
    if name in {"config.snapshot.json", "bindings.json"}: return "source_snapshot"
    if name == "artifact_manifest.json": return "self_reference"
    return "public_handoff" if relative.split("/", 1)[0] in PHASES[:-1] else "derived_diagnostic"


class Checker:
    def __init__(self, artifact: Path, input_package: Path, reference: Path | None, config: Path, source_freeze: Path):
        self.root = artifact.resolve(strict=True)
        self.input = input_package.resolve(strict=True)
        self.reference = None if reference is None else reference.resolve(strict=True)
        self.config_path = config.resolve(strict=True)
        self.freeze_path = source_freeze.resolve(strict=True)
        self.config = read_json(self.config_path)
        self.freeze = read_json(self.freeze_path)
        self.posterior_truth = load_npz(self.input / "prepared/truth/posterior_fit_truth.npz")
        self.policy_truth = load_npz(self.input / "prepared/truth/policy_fit_truth.npz")
        self.decision_public = load_npz(self.input / "prepared/public/decision_select_public.npz")
        self.decision_truth = load_npz(self.input / "prepared/truth/decision_select_truth.npz")
        self.evaluate_public = load_npz(self.input / "prepared/public/evaluate_public.npz")
        self.evaluate_truth = load_npz(self.input / "prepared/truth/evaluate_truth.npz")
        self.utility = np.load(self.input / "prepared/public/utilities.npy", allow_pickle=False)
        self.posterior_states = read_json(self.root / "posterior_fit/posterior_states.json")
        self.posterior_arrays = load_npz(self.root / "posterior_fit/posterior_state_arrays.npz")
        self.policy_states = read_json(self.root / "policy_fit/policy_states.json")
        self.candidate = load_npz(self.root / "selection_propose/candidate_public.npz")
        self.selection_keys = read_json(self.root / "selection_propose/selection_key_metadata.json")
        self.apply_metadata = read_json(self.root / "selection_propose/apply_metadata.json")
        self.decision = read_json(self.root / "selection_evaluate/selection_decision.json")
        self.policy = read_json(self.root / "selection_freeze/selection_policy.json")
        self.eval_metadata = load_npz(self.root / "evaluation_apply/evaluation_metadata.npz")
        self.eval_actions = load_npz(self.root / "evaluation_apply/evaluation_actions.npz")
        self.eval_masses = load_npz(self.root / "evaluation_apply/evaluation_masses.npz")
        self.eval_sensitivity = load_npz(self.root / "evaluation_apply/evaluation_sensitivities.npz")
        self.apply_status = read_json(self.root / "evaluation_apply/evaluation_apply_status.json")
        self.raw = load_npz(self.root / "evaluation_truth/diagnostic_arrays.npz")
        self.boot = load_npz(self.root / "evaluation_truth/bootstrap_indices.npz")
        self.estimands = read_json(self.root / "evaluation_truth/estimand_table.json")
        self.penalty = float(self.config["reader"]["penalty"])
        self.selection_public: dict[str, dict[str, np.ndarray]] = {}
        self.selection_scores: dict[str, dict[str, np.ndarray]] = {}
        self.evaluation_public: dict[str, dict[str, np.ndarray]] = {}

    def mass(self, name: str, logits: np.ndarray, shuffled: bool = False) -> np.ndarray:
        role = "target_shuffled" if shuffled else "real"
        if name == "marginal":
            return independent.marginal_mass(self.posterior_states["marginal"][role], logits)
        return independent.posterior_mass(logits, self.posterior_arrays[f"joint_{role}__final_theta"], "joint_full")

    def p1(self) -> None:
        if self.config["schema_version"] != "proportional-set-valued-physical-preflight-v1" or self.freeze["schema_version"] != "proportional-set-valued-physical-source-freeze-v1": raise CheckFailure("authority schema drifted")
        freeze_rel = self.freeze_path.relative_to(REPO_ROOT).as_posix(); commit = git("log", "-1", "--format=%H", "--", freeze_rel)
        if git("rev-parse", f"{commit}^") != self.freeze["implementation_commit"]: raise CheckFailure("freeze parent drifted")
        if git("diff-tree", "--no-commit-id", "--name-only", "-r", commit).splitlines() != [freeze_rel]: raise CheckFailure("freeze commit scope drifted")
        for relative, digest in self.freeze["files"].items():
            if sha256_file(REPO_ROOT / relative) != digest: raise CheckFailure(f"frozen file drifted: {relative}")
        validate_historical_reference(self.freeze)
        bindings = read_json(self.root / "bindings.json")
        if bindings["source_freeze_sha256"] != sha256_file(self.freeze_path): raise CheckFailure("run source freeze binding drifted")
        if read_json(self.root / "config.snapshot.json") != self.config: raise CheckFailure("config snapshot drifted")
        observed_sources = {}
        for name, binding in self.config["source_bindings"].items():
            path = (REPO_ROOT / binding["path"]).resolve(strict=True)
            if sha256_file(path) != binding["sha256"]: raise CheckFailure(f"historical binding drifted: {name}")
            observed_sources[name] = {"path": binding["path"], "sha256": binding["sha256"], "bytes": path.stat().st_size}
        if bindings.get("historical_sources") != observed_sources: raise CheckFailure("run historical source binding drifted")
        source_text = Path(__file__).read_text(encoding="utf-8")
        imported = []
        for node in ast.walk(ast.parse(source_text)):
            if isinstance(node, ast.Import): imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module: imported.append(node.module)
        if any(name.endswith(("proportional_set_valued_native", "run_proportional_set_valued_physical_preflight", "_proportional_set_valued_phase_worker")) for name in imported): raise CheckFailure("physical checker imports tested implementation")

    def _validate_full_role(self, name: str, bundle: dict[str, np.ndarray], expected_rows: int, expected_split_role: str) -> None:
        expected = {"pair_token", "cluster_id", "design_stratum", "cardinality", "ensemble_logits", "per_seed_logits", "target", "split_role"}
        if set(bundle) != expected: raise CheckFailure(f"bundle keys drifted: {name}")
        n = len(bundle["pair_token"])
        dtypes = {"pair_token": "<U64", "cluster_id": "<U64", "design_stratum": "<U16", "cardinality": "<i8", "ensemble_logits": "<f8", "per_seed_logits": "<f8", "target": "|b1", "split_role": "<U24"}
        if n != expected_rows or len(np.unique(bundle["pair_token"])) != n or any(bundle[key].dtype != np.dtype(dtype) for key, dtype in dtypes.items()): raise CheckFailure(f"bundle row/dtype drifted: {name}")
        if bundle["ensemble_logits"].shape != (n, 4) or bundle["per_seed_logits"].shape != (3, n, 4) or bundle["target"].shape != (n, 4): raise CheckFailure(f"bundle shape drifted: {name}")
        if not np.isfinite(bundle["ensemble_logits"]).all() or not np.isfinite(bundle["per_seed_logits"]).all(): raise CheckFailure(f"bundle nonfinite: {name}")
        if not np.array_equal(bundle["ensemble_logits"], np.mean(bundle["per_seed_logits"], axis=0, dtype=np.float64)): raise CheckFailure(f"ensemble relation drifted: {name}")
        if not np.array_equal(bundle["cluster_id"], bundle["pair_token"]) or not np.array_equal(bundle["cardinality"], bundle["target"].sum(axis=1).astype("<i8")): raise CheckFailure(f"identity/cardinality drifted: {name}")
        if set(bundle["design_stratum"].astype(str)) != {"FAR_RIVAL", "NEAR_RIVAL"} or not np.all((bundle["cardinality"] >= 1) & (bundle["cardinality"] <= 4)): raise CheckFailure(f"vocabulary/range drifted: {name}")
        if set(bundle["split_role"].astype(str)) != {expected_split_role}: raise CheckFailure(f"split role drifted: {name}")

    def p2(self) -> None:
        expected = {
            "posterior": (self.posterior_truth, {"pair_token", "cluster_id", "design_stratum", "cardinality", "ensemble_logits", "per_seed_logits", "target", "split_role"}),
            "policy": (self.policy_truth, {"pair_token", "cluster_id", "design_stratum", "cardinality", "ensemble_logits", "per_seed_logits", "target", "split_role"}),
            "decision_public": (self.decision_public, {"pair_token", "design_stratum", "cardinality", "ensemble_logits", "per_seed_logits"}),
            "decision_truth": (self.decision_truth, {"pair_token", "target"}),
            "evaluate_public": (self.evaluate_public, {"pair_token", "design_stratum", "cardinality", "ensemble_logits", "per_seed_logits"}),
            "evaluate_truth": (self.evaluate_truth, {"pair_token", "target"}),
        }
        for name, (bundle, keys) in expected.items():
            if set(bundle) != keys: raise CheckFailure(f"bundle keys drifted: {name}")
        for public, truth, name in ((self.decision_public, self.decision_truth, "decision"), (self.evaluate_public, self.evaluate_truth, "evaluate")):
            assert_array(public["pair_token"], truth["pair_token"], f"{name} identity")
            if not np.array_equal(public["cardinality"], truth["target"].sum(axis=1).astype("<i8")): raise CheckFailure(f"{name} cardinality drifted")
            if any(fragment in key.lower() for key in public for fragment in TRUTH_FORBIDDEN): raise CheckFailure(f"{name} public semantic leak")
        token_sets = [set(bundle["pair_token"].astype(str)) for bundle in (self.posterior_truth, self.policy_truth, self.decision_public, self.evaluate_public)]
        if any(left & right for left, right in itertools.combinations(token_sets, 2)): raise CheckFailure("role overlap")
        rows = self.config["opened_fixture_roles"]
        if [len(item) for item in token_sets] != [rows["posterior_fit"], rows["policy_fit"], rows["decision_select"], rows["evaluate"]]: raise CheckFailure("role counts drifted")
        self._validate_full_role("posterior", self.posterior_truth, rows["posterior_fit"], "posterior_fit"); self._validate_full_role("policy", self.policy_truth, rows["policy_fit"], "policy_fit")
        for name, split_role, public, truth in (("decision", "decision_select", self.decision_public, self.decision_truth), ("evaluate", "evaluate_fixture", self.evaluate_public, self.evaluate_truth)):
            full = {**public, "cluster_id": public["pair_token"], "target": truth["target"], "split_role": np.full(len(public["pair_token"]), split_role, dtype="<U24")}
            self._validate_full_role(name, full, rows["decision_select" if name == "decision" else "evaluate"], split_role)
        expected_files = {
            "opened_fixture_escrow.json": 0o400, "preparation_freeze.json": 0o444, "preparation_receipt.json": 0o444, "public_manifest.json": 0o444, "journals/prepare.json": 0o444,
            "prepared/public/decision_select_public.npz": 0o444, "prepared/public/evaluate_public.npz": 0o444, "prepared/public/utilities.npy": 0o444,
            "prepared/truth/decision_select_truth.npz": 0o400, "prepared/truth/evaluate_truth.npz": 0o400, "prepared/truth/policy_fit_truth.npz": 0o400, "prepared/truth/posterior_fit_truth.npz": 0o400,
        }
        expected_dirs = {".": 0o700, "journals": 0o555, "prepared": 0o500, "prepared/public": 0o555, "prepared/truth": 0o500}
        files = {path.relative_to(self.input).as_posix(): path for path in self.input.rglob("*") if path.is_file()}; dirs = {".": self.input, **{path.relative_to(self.input).as_posix(): path for path in self.input.rglob("*") if path.is_dir()}}
        all_objects = {path.relative_to(self.input).as_posix() for path in self.input.rglob("*")}
        if set(files) != set(expected_files) or set(dirs) != set(expected_dirs) or all_objects != set(expected_files) | (set(expected_dirs) - {"."}): raise CheckFailure("input inventory drifted")
        for relative, mode in expected_dirs.items():
            info = dirs[relative].lstat()
            if dirs[relative].is_symlink() or not stat.S_ISDIR(info.st_mode) or stat.S_IMODE(info.st_mode) != mode or info.st_uid or info.st_gid: raise CheckFailure(f"input directory metadata drifted: {relative}")
        for relative, mode in expected_files.items():
            info = files[relative].lstat()
            if files[relative].is_symlink() or not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or stat.S_IMODE(info.st_mode) != mode or info.st_uid or info.st_gid: raise CheckFailure(f"input file metadata drifted: {relative}")
        prepared = {relative: {"bytes": files[relative].stat().st_size, "sha256": sha256_file(files[relative])} for relative in expected_files if relative.startswith("prepared/")}
        bindings = read_json(self.root / "bindings.json")
        package_id = hashlib.sha256(json_bytes({"source_freeze": bindings["source_freeze_sha256"], "files": prepared})).hexdigest()
        prep = read_json(self.input / "preparation_freeze.json"); escrow = read_json(self.input / "opened_fixture_escrow.json"); manifest = read_json(self.input / "public_manifest.json")
        expected_prep = {"schema_version": "proportional-physical-preparation-freeze-v1", "status": "OPENED_DATA_PHYSICAL_PREFLIGHT", "package_id": package_id, "source_freeze_sha256": bindings["source_freeze_sha256"], "checkpoint_axis": self.config["checkpoint_epochs"], "files": {name: row["sha256"] for name, row in prepared.items()}, "prospective_evidence": False}
        if prep != expected_prep or escrow.get("package_id") != package_id or escrow.get("prepared_files") != prepared or escrow.get("historical_sources") != bindings["historical_sources"] or escrow.get("generation_escrow") is not False or escrow.get("prospective_evidence") is not False: raise CheckFailure("preparation authority drifted")
        if manifest != {"schema_version": "proportional-physical-public-manifest-v1", "package_id": package_id, "roles": rows, "files": prepared, "truth_commitments_only": True}: raise CheckFailure("public input manifest drifted")
        if read_json(self.input / "preparation_receipt.json") != {"schema_version": "proportional-physical-preparation-receipt-v1", "package_id": package_id, "status": "PREPARED", "cross_array_checks": "PASS", "pairwise_disjoint": True}: raise CheckFailure("preparation receipt drifted")
        if read_json(self.input / "journals/prepare.json") != {"schema_version": "proportional-physical-journal-v1", "phase": "prepare", "previous_state": "INITIALIZED", "new_state": "PREPARED", "maximum_truth_materialized": "NONE", "package_id": package_id}: raise CheckFailure("preparation journal drifted")
        overlaps = {f"{left}__{right}": len(token_sets[index] & token_sets[jndex]) for index, left in enumerate(("posterior", "policy", "decision", "evaluate")) for jndex, right in enumerate(("posterior", "policy", "decision", "evaluate")) if index < jndex}
        if escrow.get("pairwise_overlap") != overlaps or any(overlaps.values()): raise CheckFailure("escrow overlap drifted")
        if self.utility.dtype != np.dtype("<f8") or self.utility.shape != (24, 4) or not np.isfinite(self.utility).all(): raise CheckFailure("utility catalogue drifted")

    def _stage_inputs(self, phase: str) -> dict[str, Path]:
        common = {"config.json": self.config_path, "bindings.json": self.root / "bindings.json", "preparation_freeze.json": self.input / "preparation_freeze.json", "utilities.npy": self.input / "prepared/public/utilities.npy"}
        truth = self.input / "prepared/truth"; public = self.input / "prepared/public"
        if phase == "posterior_fit": common["posterior_fit_truth.npz"] = truth / "posterior_fit_truth.npz"
        elif phase == "policy_fit":
            common["policy_fit_truth.npz"] = truth / "policy_fit_truth.npz"
            for name in HANDOFF["posterior_fit"]: common[name] = self.root / "posterior_fit" / name
        elif phase == "selection_propose":
            common["decision_select_public.npz"] = public / "decision_select_public.npz"
            for prior in ("posterior_fit", "policy_fit"):
                for name in HANDOFF[prior]: common[name] = self.root / prior / name
        elif phase == "selection_evaluate":
            common["decision_select_truth.npz"] = truth / "decision_select_truth.npz"
            for name in ("selection_key_metadata.json", "candidate_public.npz", "candidate_freeze.json"): common[name] = self.root / "selection_propose" / name
        elif phase == "selection_freeze":
            common["decision_select_public.npz"] = public / "decision_select_public.npz"
            for prior in ("posterior_fit", "policy_fit", "selection_propose", "selection_evaluate"):
                for name in HANDOFF[prior]: common[name] = self.root / prior / name
        elif phase == "evaluation_apply":
            common["evaluate_public.npz"] = public / "evaluate_public.npz"
            for prior in ("posterior_fit", "policy_fit", "selection_freeze"):
                for name in HANDOFF[prior]: common[name] = self.root / prior / name
        else:
            common["evaluate_truth.npz"] = truth / "evaluate_truth.npz"
            for name in HANDOFF["evaluation_apply"]: common[name] = self.root / "evaluation_apply" / name
        return common

    def _probe_hashes(self, phase: str) -> list[str]:
        truth = self.input / "prepared/truth"
        mapping = {
            "posterior_fit": [truth / "policy_fit_truth.npz", truth / "decision_select_truth.npz", truth / "evaluate_truth.npz"],
            "policy_fit": [truth / "posterior_fit_truth.npz", truth / "decision_select_truth.npz", truth / "evaluate_truth.npz"],
            "selection_propose": [truth / name for name in ("posterior_fit_truth.npz", "policy_fit_truth.npz", "decision_select_truth.npz", "evaluate_truth.npz")],
            "selection_evaluate": [self.root / "posterior_fit/posterior_states.json", self.root / "policy_fit/policy_states.json", truth / "evaluate_truth.npz"],
            "selection_freeze": [truth / "decision_select_truth.npz", self.root / "selection_evaluate/candidate_metrics_private.npz", truth / "evaluate_truth.npz"],
            "evaluation_apply": [truth / "decision_select_truth.npz", self.root / "selection_evaluate/candidate_metrics_private.npz", truth / "evaluate_truth.npz"],
            "evaluation_truth": [self.root / "posterior_fit/posterior_states.json", self.root / "policy_fit/policy_states.json", truth / "decision_select_truth.npz", self.root / "selection_freeze/selection_policy.json"],
        }
        return [hashlib.sha256(str(path.resolve()).encode()).hexdigest() for path in mapping[phase]]

    def p3(self) -> None:
        if stat.S_IMODE(self.root.stat().st_mode) != 0o700 or self.root.stat().st_uid != 0: raise CheckFailure("run root permissions drifted")
        for phase in PHASES:
            directory = self.root / phase
            if directory.is_symlink() or stat.S_IMODE(directory.stat().st_mode) != 0o500 or directory.stat().st_uid != 0: raise CheckFailure(f"phase directory boundary drifted: {phase}")
            receipt = read_json(directory / "worker_receipt.json"); runtime = receipt["runtime"]
            if receipt.get("schema_version") != "proportional-physical-worker-receipt-v2" or receipt.get("phase") != phase: raise CheckFailure(f"worker receipt schema drifted: {phase}")
            identity = runtime["identity"]
            if identity["Uid"].split()[0] != "65534" or identity["Gid"].split()[0] != "65534" or identity["Groups"] != "" or identity["NoNewPrivs"] != "1": raise CheckFailure(f"worker identity drifted: {phase}")
            if any(identity[key] != "0000000000000000" for key in ("CapInh", "CapPrm", "CapEff", "CapBnd", "CapAmb")): raise CheckFailure(f"capability set drifted: {phase}")
            if runtime["torch_imported"] or runtime["gpu_used_or_queried"]: raise CheckFailure(f"CPU boundary drifted: {phase}")
            runtime_path = Path(runtime["cwd"]).parent / "runtime"
            envelope = runtime_path.parent
            if envelope.parent != REPO_ROOT / ".physical_set_valued_stages" or not envelope.name.startswith(f"physical-{phase}-"): raise CheckFailure(f"runtime envelope drifted: {phase}")
            expected_environment = {key: ({**ENVIRONMENT, "PYTHONPATH": str(runtime_path)})[key] for key in ENVIRONMENT_KEYS}
            if runtime["environment"] != expected_environment or runtime["cwd"] != runtime["stage_metadata"]["cwd"] or Path(runtime["cwd"]).name != "stage" or runtime_path.parent.name != Path(runtime["cwd"]).parent.name: raise CheckFailure(f"runtime environment/cwd drifted: {phase}")
            if set(runtime["modules"]) != set(RUNTIME_MODULES) or set(runtime["runtime_files"]) != set(RUNTIME_SOURCES): raise CheckFailure(f"runtime module set drifted: {phase}")
            for name, relative in RUNTIME_SOURCES.items():
                row = runtime["runtime_files"][name]
                if Path(row["path"]).parent not in {runtime_path, runtime_path / "geometria_proporcional"} or Path(row["path"]).name != name or row["sha256"] != self.freeze["files"][relative]: raise CheckFailure(f"runtime blob drifted: {phase}/{name}")
            for module, filename in RUNTIME_MODULES.items():
                row = runtime["modules"][module]
                if Path(row["path"]).name != filename or row["sha256"] != self.freeze["files"][RUNTIME_SOURCES[filename]]: raise CheckFailure(f"runtime module drifted: {phase}/{module}")
            expected_sys = [str(runtime_path), str(Path(sysconfig.get_path("stdlib")).parent / f"python{sys.version_info.major}{sys.version_info.minor}.zip"), sysconfig.get_path("stdlib"), str(Path(sysconfig.get_path("stdlib")) / "lib-dynload"), site.getsitepackages()[0]]
            if runtime["sys_path"] != expected_sys: raise CheckFailure(f"worker sys.path drifted: {phase}")
            inputs = self._stage_inputs(phase); stage_hashes = {name: sha256_file(path) for name, path in sorted(inputs.items())}
            contract = receipt.get("stage_contract", {})
            if receipt.get("stage_files") != stage_hashes or contract.get("allowed_files") != sorted([*inputs, "phase_request.json"]) or contract.get("expected_outputs") != list(EXPECTED_OUTPUTS[phase]) or contract.get("runtime_modules") != list(RUNTIME_MODULES) or contract.get("environment_keys") != list(ENVIRONMENT_KEYS) or contract.get("probe_path_sha256") != self._probe_hashes(phase): raise CheckFailure(f"stage contract drifted: {phase}")
            metadata = runtime["stage_metadata"]
            if metadata.get("mode") != 0o555 or metadata.get("uid") != 0 or metadata.get("gid") != 0 or set(metadata.get("files", {})) != set(contract["allowed_files"]): raise CheckFailure(f"stage metadata drifted: {phase}")
            for name, row in metadata["files"].items():
                if row["mode"] != 0o444 or row["uid"] != 0 or row["gid"] != 0 or (name != "phase_request.json" and row["sha256"] != stage_hashes[name]): raise CheckFailure(f"stage file metadata drifted: {phase}/{name}")
            if len(metadata["files"]["phase_request.json"]["sha256"]) != 64: raise CheckFailure(f"phase request digest drifted: {phase}")
            if [row["path_sha256"] for row in receipt["probes"]] != self._probe_hashes(phase) or any(row["outcome"] not in {"PermissionError", "FileNotFoundError"} for row in receipt["probes"]): raise CheckFailure(f"probe contract drifted: {phase}")
            actual_output = {path.name: sha256_file(path) for path in directory.iterdir() if path.is_file() and path.name != "worker_receipt.json"}
            if receipt.get("output_files") != actual_output: raise CheckFailure(f"worker output receipt drifted: {phase}")
            if any(int(pool.get("num_threads", 0)) > 1 for pool in runtime["threadpools"]): raise CheckFailure(f"threadpool drifted: {phase}")

    def p4(self) -> None:
        previous = "PREPARED"; levels = ("POSTERIOR_FIT", "POLICY_FIT", "POLICY_FIT", "DECISION_SELECT", "DECISION_SELECT", "DECISION_SELECT", "EVALUATE")
        package_id = read_json(self.input / "preparation_freeze.json")["package_id"]
        top_directories = {path.name for path in self.root.iterdir() if path.is_dir()}
        if top_directories != {"journals", *PHASES}: raise CheckFailure("phase directory inventory drifted")
        journal_files = {path.name for path in (self.root / "journals").iterdir()}
        if journal_files != {f"{phase}.json" for phase in PHASES}: raise CheckFailure("journal inventory drifted")
        for phase, state, level in zip(PHASES, STATES, levels, strict=True):
            journal = read_json(self.root / "journals" / f"{phase}.json")
            expected_keys = {"schema_version", "phase", "previous_state", "new_state", "maximum_truth_materialized", "package_id", "input_hashes", "output_hashes", "worker_receipt_sha256", "wall_seconds", "peak_rss_bytes"}
            if set(journal) != expected_keys or journal["schema_version"] != "proportional-physical-journal-v1" or journal["phase"] != phase or journal["previous_state"] != previous or journal["new_state"] != state or journal["maximum_truth_materialized"] != level or journal["package_id"] != package_id: raise CheckFailure(f"journal transition drifted: {phase}")
            actual = {path.name: sha256_file(path) for path in (self.root / phase).iterdir() if path.is_file()}
            if journal["input_hashes"] != {name: sha256_file(path) for name, path in sorted(self._stage_inputs(phase).items())} or journal["output_hashes"] != dict(sorted(actual.items())) or journal["worker_receipt_sha256"] != sha256_file(self.root / phase / "worker_receipt.json"): raise CheckFailure(f"journal hash drifted: {phase}")
            previous = state

    def p5(self) -> None:
        validate_historical_reference(self.freeze)
        parity = read_json(self.root / "r564_parity_receipt.json")
        if not parity["all_exact"] or parity["passed"] != parity["total"] or parity["total"] != 36 or len(parity.get("checks", [])) != 36 or any(set(row) != {"object", "comparator", "equal"} or row["equal"] is not True for row in parity["checks"]): raise CheckFailure("R564 posterior parity incomplete")
        reference = HISTORICAL_REFERENCE_ROOT / "posterior_fit"
        for current_name, old_name in (("posterior_states.json", "states.json"), ("posterior_state_arrays.npz", "state_arrays.npz"), ("posterior_oof_arrays.npz", "oof_arrays.npz"), ("target_shuffle_map.json", "target_shuffle_map.json"), ("target_shuffle_arrays.npz", "target_shuffle_arrays.npz")):
            if sha256_file(self.root / "posterior_fit" / current_name) != sha256_file(reference / old_name): raise CheckFailure(f"R564 posterior bytes drifted: {current_name}")
        shuffled = load_npz(self.root / "posterior_fit/target_shuffle_arrays.npz")
        folds = independent.fold_ids(self.posterior_truth["pair_token"], self.posterior_truth["design_stratum"], self.posterior_truth["cardinality"])
        donor, rows, permutable = independent.target_derangement(self.posterior_truth["pair_token"], folds, self.posterior_truth["design_stratum"], self.posterior_truth["cardinality"], 53602)
        assert_array(donor, shuffled["donor_index"], "posterior donor"); assert_array(permutable, shuffled["permutable"], "posterior permutable")
        assert_array(self.posterior_truth["target"][donor], shuffled["target_shuffled"], "posterior shuffled target")
        if rows != read_json(self.root / "posterior_fit/target_shuffle_map.json"): raise CheckFailure("posterior semantic shuffle map drifted")

        logits = self.posterior_truth["ensemble_logits"].astype(np.float64)
        targets = {"real": self.posterior_truth["target"].astype(bool), "target_shuffled": shuffled["target_shuffled"].astype(bool)}
        expected_states: dict[str, Any] = {"schema_version": "proportional-posterior-states-v1", "marginal": {}, "joint": {}, "target_shuffle": {"seed": 53602, "fixture_sha256": self.config["posterior"]["target_shuffle"]["fixture_sha256"], "permutable_fraction": float(permutable.mean()), "singletons": [str(self.posterior_truth["pair_token"][index]) for index in np.flatnonzero(~permutable)], "same_map_for_representations": True}}
        expected_handoff: dict[str, np.ndarray] = {}
        expected_oof: dict[str, np.ndarray] = {}
        for role, target in targets.items():
            model = independent.LogisticRegression(**independent.MARGINAL_CONTRACT).fit(logits.reshape(-1, 1), target.reshape(-1).astype(np.int64))
            probability = model.predict_proba(logits.reshape(-1, 1))[:, 1]
            expected_states["marginal"][role] = {"kind": "pooled_platt", "contract": dict(independent.MARGINAL_CONTRACT), "sklearn_version": sklearn.__version__, "classes": model.classes_.astype(int).tolist(), "coefficient": float(model.coef_[0, 0]), "intercept": float(model.intercept_[0]), "n_iter": int(model.n_iter_[0]), "n_rows": int(target.size), "positive_fraction": float(target.reshape(-1).mean()), "fit_probability_sha256": independent.array_digest(probability)}

            oof_nll = np.empty((6, len(logits)), dtype=np.float64); oof_brier = np.empty_like(oof_nll)
            fold_theta = np.empty((6, 4, 12), dtype=np.float64); fold_objective = np.empty((6, 4), dtype=np.float64); fold_gradient = np.empty((6, 4), dtype=np.float64)
            fold_iterations = np.empty((6, 4), dtype=np.int64); fold_evaluations = np.empty((6, 4), dtype=np.int64); grid_rows = []
            for grid_index, regularization in enumerate(independent.JOINT_GRID):
                for fold in range(4):
                    train, holdout = folds != fold, folds == fold
                    fit = independent.fit_joint_posterior(logits[train], target[train], "joint_full", regularization, max_iter=2000, gtol=1e-9, ftol=1e-12)
                    metric = independent.set_metrics(independent.posterior_mass(logits[holdout], fit["theta"], "joint_full"), target[holdout])
                    oof_nll[grid_index, holdout] = metric["exact_set_nll"]; oof_brier[grid_index, holdout] = metric["marginal_brier"]
                    fold_theta[grid_index, fold] = fit["theta"]; fold_objective[grid_index, fold] = fit["objective"]; fold_gradient[grid_index, fold] = fit["gradient_norm"]
                    fold_iterations[grid_index, fold] = fit["iterations"]; fold_evaluations[grid_index, fold] = fit["function_evaluations"]
                grid_rows.append({"regularization": regularization, "mean_oof_exact_set_nll": float(oof_nll[grid_index].mean()), "mean_oof_marginal_brier": float(oof_brier[grid_index].mean()), "negative_regularization": -regularization})
            chosen = min(range(6), key=lambda index: (grid_rows[index]["mean_oof_exact_set_nll"], grid_rows[index]["mean_oof_marginal_brier"], grid_rows[index]["negative_regularization"]))
            final = independent.fit_joint_posterior(logits, target, "joint_full", independent.JOINT_GRID[chosen], max_iter=2000, gtol=1e-9, ftol=1e-12)
            expected_states["joint"][role] = {"kind": "joint_full", "regularization_grid": list(independent.JOINT_GRID), "selected_index": chosen, "selected_regularization": independent.JOINT_GRID[chosen], "selection_key": ["mean_oof_exact_set_nll", "mean_oof_marginal_brier", "negative_regularization"], "optimizer": {"method": "L-BFGS-B", "max_iter": 2000, "gtol": 1e-9, "ftol": 1e-12}, "grid_metrics": grid_rows, "final_objective": float(final["objective"]), "final_gradient_norm": float(final["gradient_norm"]), "final_iterations": int(final["iterations"]), "final_function_evaluations": int(final["function_evaluations"]), "final_message": str(final["message"])}
            prefix = f"joint_{role}__"
            expected_oof.update({prefix + "fold_id": folds, prefix + "oof_exact_set_nll": oof_nll, prefix + "oof_marginal_brier": oof_brier, prefix + "fold_theta": fold_theta, prefix + "fold_objective": fold_objective, prefix + "fold_gradient_norm": fold_gradient, prefix + "fold_iterations": fold_iterations, prefix + "fold_function_evaluations": fold_evaluations})
            expected_handoff[prefix + "final_theta"] = np.asarray(final["theta"], dtype=np.float64)
            expected_handoff[prefix + "final_interaction_coefficients"] = independent.centered_interactions(final["theta"], "joint_full")
        assert_nested_close(expected_states, self.posterior_states, "posterior states", 2e-10)
        saved_oof = load_npz(self.root / "posterior_fit/posterior_oof_arrays.npz")
        if set(saved_oof) != set(expected_oof) or set(self.posterior_arrays) != set(expected_handoff): raise CheckFailure("posterior recomputation inventory drifted")
        for key, value in expected_oof.items(): assert_close_array(value, saved_oof[key], f"posterior OOF {key}", 2e-10)
        for key, value in expected_handoff.items(): assert_close_array(value, self.posterior_arrays[key], f"posterior state {key}", 2e-10)

    def p6(self) -> None:
        validate_historical_reference(self.freeze)
        schema = read_json(self.root / "policy_fit/feature_schema.json")
        if schema != {"schema_version": "proportional-contextual-map-features-v1", "feature_names": list(independent.FEATURE_NAMES), "count": 17, "hard_adapter": "HARD_MAP_SET", "weighting": "one_total_weight_per_active_token"}: raise CheckFailure("policy feature schema drifted")
        if len(self.policy_states["marginal"]["controls"]) != 5 or len(self.policy_states["joint"]["controls"]) != 5: raise CheckFailure("policy controls drifted")
        reference = HISTORICAL_REFERENCE_ROOT / "policy_fit"
        for current_name, old_name in (("feature_schema.json", "feature_schema.json"), ("policy_states.json", "states.json"), ("policy_state_arrays.npz", "state_arrays.npz"), ("policy_fit_private.npz", "fit_scores.npz"), ("control_maps.json", "control_maps.json"), ("control_arrays.npz", "control_arrays.npz")):
            if sha256_file(self.root / "policy_fit" / current_name) != sha256_file(reference / old_name): raise CheckFailure(f"R564 policy bytes drifted: {current_name}")

        fit_arrays = load_npz(self.root / "policy_fit/policy_fit_private.npz"); control_arrays = load_npz(self.root / "policy_fit/control_arrays.npz"); state_arrays = load_npz(self.root / "policy_fit/policy_state_arrays.npz")
        expected_state_arrays: dict[str, np.ndarray] = {}; expected_control_maps: dict[str, Any] = {"schema_version": "proportional-matched-control-maps-v1"}
        logits = self.policy_truth["ensemble_logits"].astype(np.float64); seed_logits = self.policy_truth["per_seed_logits"].astype(np.float64); target = self.policy_truth["target"].astype(bool); tokens = self.policy_truth["pair_token"].astype(str)
        for posterior_name in ("marginal", "joint"):
            mass = self.mass(posterior_name, logits); public = independent.public_design(logits, seed_logits, mass, self.utility, self.penalty); active = public["disagreement"]
            hard_values = independent.regret(public["hard_actions"], target, self.utility, self.penalty); candidate_values = independent.regret(public["posterior_actions"], target, self.utility, self.penalty)
            gain = hard_values - candidate_values; harm = gain < -1e-12; incompatibility = ~target[np.arange(len(target))[:, None], public["posterior_actions"]]
            expected_public = {**public, "gain": gain, "harm": harm, "incompatibility": incompatibility}
            for key in ("design", "weights", "disagreement", "gain", "harm", "incompatibility", "hard_actions", "posterior_actions"):
                assert_close_array(np.asarray(expected_public[key], dtype=float), np.asarray(fit_arrays[f"{posterior_name}__{key}"], dtype=float), f"policy training {posterior_name}/{key}", 5e-12)
            family = self.policy_states[posterior_name]
            if [row["seed"] for row in family["controls"]] != self.config["matched_controls"]["seeds"]: raise CheckFailure("policy seed order drifted")
            x, weights = public["design"][active], public["weights"][active]
            true_states = family["true"]["states"]
            fitted = {"proposer": independent.fit_ridge(x, gain[active], weights), "harm": independent.fit_guard(x, harm[active], weights), "incompatibility": independent.fit_guard(x, incompatibility[active], weights)}
            for model_name, state in true_states.items():
                independent.compare_fitted_state(state, fitted[model_name], f"{posterior_name} true {model_name}")
                score = np.full(active.shape, np.nan); score[active] = independent.score_state(state, x)
                assert_close_array(score, fit_arrays[f"{posterior_name}__true__{model_name}"], f"policy true score {posterior_name}/{model_name}", 5e-12, equal_nan=True)
            map_rows = []; counts = active.sum(axis=1)
            for control in family["controls"]:
                seed = int(control["seed"]); prefix = f"{posterior_name}__control_{seed}"
                mapping = independent.matched_map(gain, harm, incompatibility, active, tokens, seed)
                assert_array(mapping, control_arrays[f"{prefix}__mapping"], f"control mapping {prefix}")
                rows_active, policies = np.where(active); donors = mapping[rows_active, policies]
                transported = {"gain": gain.copy(), "harm": harm.copy(), "incompatibility": incompatibility.copy()}
                for key in transported: transported[key][rows_active, policies] = {"gain": gain, "harm": harm, "incompatibility": incompatibility}[key][donors, policies]
                for key, value in transported.items(): assert_close_array(value.astype(float), control_arrays[f"{prefix}__{key}"].astype(float), f"control target {prefix}/{key}", 0.0)
                semantic_rows = [(str(tokens[row]), int(policy), str(tokens[mapping[row, policy]]), int(counts[row])) for row, policy in zip(rows_active.tolist(), policies.tolist(), strict=True)]; semantic_rows.sort(key=lambda item: (item[0].encode(), item[1]))
                mapping_sha = hashlib.sha256(json.dumps(semantic_rows, ensure_ascii=False, sort_keys=False, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
                triplet = hashlib.sha256()
                for key in ("gain", "harm", "incompatibility"): triplet.update(np.ascontiguousarray(transported[key][active]).tobytes())
                permutable = np.zeros_like(active); stratum_rows = []; total_hamming = 0
                for policy in range(active.shape[1]):
                    for count in sorted(np.unique(counts[active[:, policy]]).tolist()):
                        indices = np.asarray(sorted(np.flatnonzero(active[:, policy] & (counts == count)).tolist(), key=lambda index: tokens[index].encode()))
                        singleton = len(indices) == 1
                        if not singleton: permutable[indices, policy] = True
                        signatures = np.column_stack([np.ascontiguousarray(gain[indices, policy], dtype="<f8").view("<u8"), harm[indices, policy].astype(np.uint64), incompatibility[indices, policy].astype(np.uint64)])
                        selected_cost = 0 if singleton else int(np.sum(signatures != signatures[[np.where(indices == mapping[index, policy])[0][0] for index in indices]])); total_hamming += selected_cost
                        stratum_rows.append({"policy_index": int(policy), "disagreement_count": int(count), "rows": int(len(indices)), "singleton": singleton, "maximum_hamming": selected_cost})
                diagnostics = {"seed": seed, "active_rows": int(active.sum()), "strata": len(stratum_rows), "singleton_rows": int(np.sum(active & ~permutable)), "permutable_fraction": float(permutable[active].mean()), "maximum_hamming": total_hamming, "mapping_sha256": mapping_sha, "target_triplet_sha256": triplet.hexdigest(), "stratum_rows": stratum_rows}
                if control["diagnostics"] != diagnostics: raise CheckFailure(f"control diagnostics drifted: {prefix}")
                map_rows.append(diagnostics)
                states = control["states"]; fitted_control = {"proposer": independent.fit_ridge(x, transported["gain"][active], weights), "harm": independent.fit_guard(x, transported["harm"][active], weights), "incompatibility": independent.fit_guard(x, transported["incompatibility"][active], weights)}
                for model_name, state in states.items():
                    independent.compare_fitted_state(state, fitted_control[model_name], f"{prefix} {model_name}")
                    score = np.full(active.shape, np.nan); score[active] = independent.score_state(state, x)
                    assert_close_array(score, fit_arrays[f"{prefix}__{model_name}"], f"control score {prefix}/{model_name}", 5e-12, equal_nan=True)
            expected_control_maps[posterior_name] = map_rows
            for role, bundle in [("true", family["true"]), *[(f"control_{row['seed']}", row) for row in family["controls"]]]:
                for model_name, state in bundle["states"].items():
                    base = f"{posterior_name}__{role}__{model_name}"
                    for key in ("mean", "scale", "coef"): expected_state_arrays[f"{base}__{key}"] = np.asarray(state[key], dtype=np.float64)
                    expected_state_arrays[f"{base}__intercept"] = np.asarray([state["intercept"]], dtype=np.float64)
                    if "n_iter" in state: expected_state_arrays[f"{base}__n_iter"] = np.asarray(state["n_iter"], dtype=np.int64)
        if read_json(self.root / "policy_fit/control_maps.json") != expected_control_maps: raise CheckFailure("control map document drifted")
        if set(state_arrays) != set(expected_state_arrays): raise CheckFailure("portable policy array inventory drifted")
        for key, value in expected_state_arrays.items(): assert_close_array(value, state_arrays[key], f"portable policy array {key}", 0.0)

    def p7(self) -> None:
        expected_candidate_keys = {"pair_token"}
        for name in ("marginal", "joint"):
            expected_candidate_keys.update({f"{name}__{key}" for key in ("actions", "design", "disagreement", "hard_actions", "override", "posterior_actions", "set_mass_real", "set_mass_target_shuffled", "weights")})
            expected_candidate_keys.update({f"{name}__score__{key}" for key in ("proposer", "harm", "incompatibility")})
        if set(self.candidate) != expected_candidate_keys or self.apply_metadata.get("schema_version") != "proportional-apply-metadata-v1" or set(self.apply_metadata) != {"schema_version", "posteriors"} or self.selection_keys.get("schema_version") != "proportional-selection-key-metadata-v1" or set(self.selection_keys) != {"schema_version", "posteriors"}: raise CheckFailure("candidate public inventory/schema drifted")
        assert_array(self.decision_public["pair_token"], self.candidate["pair_token"], "candidate pair token")
        for name in ("marginal", "joint"):
            mass = self.mass(name, self.decision_public["ensemble_logits"])
            shuffled = self.mass(name, self.decision_public["ensemble_logits"], True)
            pdata = independent.public_design(self.decision_public["ensemble_logits"], self.decision_public["per_seed_logits"], mass, self.utility, self.penalty)
            scores = independent.score_triplet(self.policy_states[name]["true"]["states"], pdata)
            self.selection_public[name] = pdata; self.selection_scores[name] = scores
            assert_close_array(mass, self.candidate[f"{name}__set_mass_real"], f"{name} candidate real mass")
            assert_close_array(shuffled, self.candidate[f"{name}__set_mass_target_shuffled"], f"{name} candidate shuffled mass")
            for key in ("design", "weights", "disagreement", "hard_actions", "posterior_actions"):
                assert_close_array(np.asarray(pdata[key], dtype=float), np.asarray(self.candidate[f"{name}__{key}"], dtype=float), f"{name} candidate public {key}")
            for key, value in scores.items(): assert_close_array(value, self.candidate[f"{name}__score__{key}"], f"{name} candidate score {key}", equal_nan=True)
            rows = self.apply_metadata["posteriors"][name]; key_rows = self.selection_keys["posteriors"][name]
            if len(rows) != 344 or len(key_rows) != 344: raise CheckFailure("candidate count drifted")
            rebuilt_actions = []; rebuilt_override = []
            for index, row in enumerate(rows):
                if row["candidate_index"] != index or set(key_rows[index]) != {"candidate_index", "kind", "proposer_quantile", "harm_quantile", "incompatibility_quantile"}: raise CheckFailure("candidate metadata schema drifted")
                expected_key = {key: row[key] for key in ("candidate_index", "kind", "proposer_quantile", "harm_quantile", "incompatibility_quantile")}
                if key_rows[index] != expected_key: raise CheckFailure("candidate key order/content drifted")
                if row["kind"] == "hard_only": action = pdata["hard_actions"]; override = np.zeros_like(action, dtype=bool)
                else:
                    expected_threshold = independent.thresholds(scores, pdata["disagreement"], (row["proposer_quantile"], row["harm_quantile"], row["incompatibility_quantile"]))
                    if {key: row[key] for key in expected_threshold} != expected_threshold: raise CheckFailure("candidate threshold drifted")
                    action, override = independent.apply_thresholds(scores, pdata, row)
                rebuilt_actions.append(action); rebuilt_override.append(override)
            assert_array(np.asarray(rebuilt_actions), self.candidate[f"{name}__actions"], f"{name} candidate actions")
            assert_array(np.asarray(rebuilt_override), self.candidate[f"{name}__override"], f"{name} candidate override")

    def p8(self) -> None:
        private = load_npz(self.root / "selection_evaluate/candidate_metrics_private.npz")
        aligned = load_npz(self.root / "selection_evaluate/selection_target_aligned_private.npz")
        if set(self.decision) != {"schema_version", "posteriors"} or self.decision["schema_version"] != "proportional-selection-decision-v1" or set(self.decision["posteriors"]) != {"marginal", "joint"}: raise CheckFailure("selection decision schema drifted")
        target = self.decision_truth["target"]
        expected_aligned: dict[str, np.ndarray] = {"pair_token": self.decision_truth["pair_token"], "target": target}
        for name in ("marginal", "joint"):
            actions = self.candidate[f"{name}__actions"]; overrides = self.candidate[f"{name}__override"]; hard = self.candidate[f"{name}__hard_actions"]
            hard_regret = independent.regret(hard, target, self.utility, self.penalty)
            metrics = {"mean_regret": [], "incompatibility_rate": [], "harm_rate": [], "authorized_rows": []}
            for index, candidate in enumerate(actions):
                row = independent.action_metrics(candidate, target, self.utility, self.penalty)
                metrics["mean_regret"].append(float(row["regret"].mean())); metrics["incompatibility_rate"].append(float(row["incompatibility_by_policy"].mean())); metrics["harm_rate"].append(float(np.mean(row["regret_by_policy"] > hard_regret + 1e-12))); metrics["authorized_rows"].append(int(overrides[index].sum()))
            for key, values in metrics.items(): assert_array(np.asarray(values, dtype=np.int64 if key == "authorized_rows" else np.float64), private[f"{name}__{key}"], f"{name} candidate {key}")
            key_rows = self.selection_keys["posteriors"][name]
            selected = min(range(344), key=lambda index: (metrics["mean_regret"][index], metrics["incompatibility_rate"][index], metrics["harm_rate"][index], -metrics["authorized_rows"][index], key_rows[index]["proposer_quantile"], key_rows[index]["harm_quantile"], key_rows[index]["incompatibility_quantile"]))
            selected_metrics = independent.action_metrics(actions[selected], target, self.utility, self.penalty)
            for key, value in selected_metrics.items(): expected_aligned[f"{name}__{key}"] = value
            decision = self.decision["posteriors"][name]
            expected_decision = {"selected_index": selected, "mean_regret": metrics["mean_regret"][selected], "incompatibility_rate": metrics["incompatibility_rate"][selected], "harm_rate": metrics["harm_rate"][selected], "authorized_rows": metrics["authorized_rows"][selected], "candidate_freeze_sha256": sha256_file(self.root / "selection_propose/candidate_freeze.json"), "selected_actions_sha256": independent.array_digest(actions[selected]), "selected_override_sha256": independent.array_digest(overrides[selected])}
            if decision != expected_decision: raise CheckFailure("minimal selection decision drifted")
        if set(private) != {f"{name}__{metric}" for name in ("marginal", "joint") for metric in ("mean_regret", "incompatibility_rate", "harm_rate", "authorized_rows")} or set(aligned) != set(expected_aligned): raise CheckFailure("selection private inventory drifted")
        for key, value in expected_aligned.items(): assert_array(np.asarray(value), aligned[key], f"selection aligned {key}")

    def p9(self) -> None:
        selected = load_npz(self.root / "selection_freeze/selected_actions.npz"); matches = load_npz(self.root / "selection_freeze/selection_matches.npz")
        expected_selected: dict[str, np.ndarray] = {}; expected_matches: dict[str, np.ndarray] = {}
        for name in ("marginal", "joint"):
            mass = self.mass(name, self.decision_public["ensemble_logits"])
            self.selection_public[name] = independent.public_design(self.decision_public["ensemble_logits"], self.decision_public["per_seed_logits"], mass, self.utility, self.penalty)
            index = self.decision["posteriors"][name]["selected_index"]
            assert_array(self.candidate[f"{name}__actions"][index], selected[f"{name}__actions"], f"{name} selected action")
            assert_array(self.candidate[f"{name}__override"][index], selected[f"{name}__override"], f"{name} selected override")
            expected_selected[f"{name}__actions"] = self.candidate[f"{name}__actions"][index]; expected_selected[f"{name}__override"] = self.candidate[f"{name}__override"][index]; expected_selected[f"{name}__hard_actions"] = self.candidate[f"{name}__hard_actions"]
            u_true = selected[f"{name}__override"].any(axis=1); common = u_true.copy(); expected_matches[f"{name}__u_true"] = u_true
            for control, states in zip(self.policy["posteriors"][name]["controls"], self.policy_states[name]["controls"], strict=True):
                scores = independent.score_triplet(states["states"], self.selection_public[name])
                matched = independent.matched_actions(selected[f"{name}__override"], scores, self.selection_public[name], control["thresholds"])
                for key, value in matched.items():
                    expected_matches[f"{name}__control_{control['seed']}__{key}"] = np.asarray(value); assert_array(np.asarray(value), matches[f"{name}__control_{control['seed']}__{key}"], f"{name} control {control['seed']} {key}")
                common &= matched["match_valid"]
            expected_matches[f"{name}__u_common"] = common; assert_array(common, matches[f"{name}__u_common"], f"{name} common support")
            status = "NOT_EVALUABLE_NO_TRUE_OVERRIDES" if not u_true.any() else ("EVALUABLE" if float(common.sum() / max(1, u_true.sum())) >= self.config["matched_controls"]["minimum_common_coverage"] else "NOT_EVALUABLE_CONTROL_SUPPORT")
            row = self.policy["posteriors"][name]
            if row["selected_index"] != index or row["selected"] != self.apply_metadata["posteriors"][name][index] or row["u_true_count"] != int(u_true.sum()) or row["u_common_count"] != int(common.sum()) or row["common_coverage"] != float(common.sum() / max(1, u_true.sum())) or row["common_support_status"] != status: raise CheckFailure(f"selection policy status drifted: {name}")
        if set(selected) != set(expected_selected) or set(matches) != set(expected_matches): raise CheckFailure("selection freeze inventory drifted")
        for key, value in expected_selected.items(): assert_array(value, selected[key], f"selection freeze {key}")
        for key, value in expected_matches.items(): assert_array(value, matches[key], f"selection match {key}")

    def p10(self) -> None:
        assert_array(self.evaluate_public["pair_token"], self.eval_metadata["pair_token"], "evaluation metadata token")
        assert_array(self.evaluate_public["design_stratum"], self.eval_metadata["design_stratum"], "evaluation metadata stratum")
        assert_array(self.evaluate_public["cardinality"], self.eval_metadata["cardinality"], "evaluation metadata cardinality")
        if set(self.eval_metadata) != {"pair_token", "design_stratum", "cardinality"}: raise CheckFailure("evaluation metadata widened")
        expected_actions: dict[str, np.ndarray] = {}; expected_masses: dict[str, np.ndarray] = {}; expected_sensitivity: dict[str, np.ndarray] = {}; expected_status = {}
        for name in ("marginal", "joint"):
            real = self.mass(name, self.evaluate_public["ensemble_logits"]); shuffled = self.mass(name, self.evaluate_public["ensemble_logits"], True)
            assert_array(real, self.eval_masses[f"{name}__real"], f"{name} evaluation mass"); assert_array(shuffled, self.eval_masses[f"{name}__target_shuffled"], f"{name} evaluation shuffled mass")
            expected_masses[f"{name}__real"] = real; expected_masses[f"{name}__target_shuffled"] = shuffled
            pdata = independent.public_design(self.evaluate_public["ensemble_logits"], self.evaluate_public["per_seed_logits"], real, self.utility, self.penalty)
            self.evaluation_public[name] = pdata
            scores = independent.score_triplet(self.policy_states[name]["true"]["states"], pdata); selected = self.policy["posteriors"][name]["selected"]
            if selected["kind"] == "hard_only": contextual = pdata["hard_actions"]; override = np.zeros_like(contextual, dtype=bool)
            else: contextual, override = independent.apply_thresholds(scores, pdata, selected)
            assert_array(pdata["hard_actions"], self.eval_actions[f"{name}__hard_actions"], f"{name} evaluation hard"); assert_array(contextual, self.eval_actions[f"{name}__contextual_actions"], f"{name} evaluation contextual")
            expected_actions[f"{name}__hard_actions"] = pdata["hard_actions"]; expected_actions[f"{name}__contextual_actions"] = contextual; expected_actions[f"{name}__true_override"] = override
            valid = []
            for control, states in zip(self.policy["posteriors"][name]["controls"], self.policy_states[name]["controls"], strict=True):
                control_scores = independent.score_triplet(states["states"], pdata)
                if selected["kind"] == "hard_only": matched = {"actions": pdata["hard_actions"], "selected": np.zeros_like(override), "authorized_universe": np.zeros_like(override), "match_valid": np.ones(len(override), dtype=bool), "requested_k": np.zeros(len(override), dtype=np.int64)}
                else: matched = independent.matched_actions(override, control_scores, pdata, control["thresholds"])
                valid.append(matched["match_valid"])
                for key, value in matched.items(): expected_actions[f"{name}__control_{control['seed']}__{key}"] = np.asarray(value)
            u_true = override.any(axis=1); common = u_true.copy()
            for value in valid: common &= value
            expected_actions[f"{name}__u_true"] = u_true; expected_actions[f"{name}__u_common"] = common
            coverage = float(common.sum() / max(1, u_true.sum())); status = "NOT_EVALUABLE_NO_TRUE_OVERRIDES" if not u_true.any() else ("EVALUABLE" if coverage >= self.config["matched_controls"]["minimum_common_coverage"] else "NOT_EVALUABLE_CONTROL_SUPPORT")
            expected_status[name] = {"u_true_count": int(u_true.sum()), "u_common_count": int(common.sum()), "common_coverage": coverage, "common_support_status": status}
            for cp_index, epoch in enumerate(self.config["checkpoint_epochs"]):
                cp_mass = self.mass(name, self.evaluate_public["per_seed_logits"][cp_index]); expected_sensitivity[f"checkpoint_{epoch}__{name}__mass"] = cp_mass; assert_array(cp_mass, self.eval_sensitivity[f"checkpoint_{epoch}__{name}__mass"], f"{name} checkpoint mass {epoch}")
                cpdata = independent.public_design(self.evaluate_public["per_seed_logits"][cp_index], self.evaluate_public["per_seed_logits"], cp_mass, self.utility, self.penalty); cp_scores = independent.score_triplet(self.policy_states[name]["true"]["states"], cpdata)
                cp_contextual = cpdata["hard_actions"] if selected["kind"] == "hard_only" else independent.apply_thresholds(cp_scores, cpdata, selected)[0]
                expected_sensitivity[f"checkpoint_{epoch}__{name}__hard_actions"] = cpdata["hard_actions"]; expected_sensitivity[f"checkpoint_{epoch}__{name}__contextual_actions"] = cp_contextual
                assert_array(cpdata["hard_actions"], self.eval_sensitivity[f"checkpoint_{epoch}__{name}__hard_actions"], f"{name} checkpoint hard {epoch}"); assert_array(cp_contextual, self.eval_sensitivity[f"checkpoint_{epoch}__{name}__contextual_actions"], f"{name} checkpoint contextual {epoch}")
        if set(self.eval_actions) != set(expected_actions) or set(self.eval_masses) != set(expected_masses) or set(self.eval_sensitivity) != set(expected_sensitivity) or self.apply_status != expected_status: raise CheckFailure("evaluation apply inventory/status drifted")
        for key, value in expected_actions.items(): assert_array(value, self.eval_actions[key], f"evaluation action {key}")

    def p11(self) -> None:
        target = self.evaluate_truth["target"]
        assert_array(self.evaluate_truth["pair_token"], self.eval_metadata["pair_token"], "evaluation truth identity")
        n_boot = int(self.config["bootstrap"]["replicates"])
        global_boot = np.random.Generator(np.random.PCG64(self.config["bootstrap"]["global_seed"])).integers(0, len(target), size=(n_boot, len(target)), dtype=np.int64)
        assert_array(global_boot, self.boot["global_pair_token_index"], "global bootstrap")
        expected_raw: dict[str, np.ndarray] = {"pair_token": self.eval_metadata["pair_token"].astype(str), "target": target}
        expected_boot: dict[str, np.ndarray] = {"global_pair_token_index": global_boot, "global_pair_token": self.eval_metadata["pair_token"].astype(str)}
        set_rows: dict[str, Any] = {}; action_rows: dict[str, Any] = {}; control_rows: dict[str, list[dict[str, np.ndarray]]] = {}; common_boot: dict[str, np.ndarray | None] = {}
        metrics_doc: dict[str, Any] = {"schema_version": "proportional-physical-diagnostic-metrics-v1", "status": "OPENED_DATA_PHYSICAL_PREFLIGHT", "posteriors": {}}
        for name in ("marginal", "joint"):
            set_rows[name] = {}
            for role in ("real", "target_shuffled"):
                values = independent.set_metrics(self.eval_masses[f"{name}__{role}"], target); set_rows[name][role] = values
                for key, value in values.items(): expected_raw[f"{name}__{role}__{key}"] = value
            for reader in ("hard", "contextual"):
                values = independent.action_metrics(self.eval_actions[f"{name}__{reader}_actions"], target, self.utility, self.penalty); action_rows[f"{name}_{reader}"] = values
                for key, value in values.items(): expected_raw[f"{name}__{reader}__{key}"] = value
            controls = []
            for seed in self.config["matched_controls"]["seeds"]:
                values = independent.action_metrics(self.eval_actions[f"{name}__control_{seed}__actions"], target, self.utility, self.penalty)
                controls.append(values)
                for key, value in values.items(): expected_raw[f"{name}__control_{seed}__{key}"] = value
            control_rows[name] = controls
            common = np.asarray(self.eval_actions[f"{name}__u_common"], dtype=bool); tokens = self.eval_metadata["pair_token"].astype(str)[common]
            if len(tokens):
                seed_key = "marginal_common_seed" if name == "marginal" else "joint_common_seed"
                current = np.random.Generator(np.random.PCG64(self.config["bootstrap"][seed_key])).integers(0, len(tokens), size=(n_boot, len(tokens)), dtype=np.int64)
            else: current = None
            common_boot[name] = current
            expected_boot[f"{name}__common_index"] = np.empty((n_boot, 0), dtype=np.int64) if current is None else current
            expected_boot[f"{name}__common_pair_token"] = tokens
            status = self.apply_status[name]
            metrics_doc["posteriors"][name] = {
                "set_real": {key: float(np.mean(value)) for key, value in set_rows[name]["real"].items()},
                "set_target_shuffled": {key: float(np.mean(value)) for key, value in set_rows[name]["target_shuffled"].items()},
                "hard": {key: float(np.mean(action_rows[f"{name}_hard"][key])) for key in ("accuracy", "incompatibility", "regret", "worst_regret")},
                "contextual": {key: float(np.mean(action_rows[f"{name}_contextual"][key])) for key in ("accuracy", "incompatibility", "regret", "worst_regret")},
                "matched_controls": [{"seed": int(seed), **{key: float(np.mean(value[key])) for key in ("accuracy", "incompatibility", "regret", "worst_regret")}} for seed, value in zip(self.config["matched_controls"]["seeds"], controls, strict=True)],
                **status,
            }
        for epoch in self.config["checkpoint_epochs"]:
            for name in ("marginal", "joint"):
                for reader in ("hard", "contextual"):
                    values = independent.action_metrics(self.eval_sensitivity[f"checkpoint_{epoch}__{name}__{reader}_actions"], target, self.utility, self.penalty)
                    for key, value in values.items(): expected_raw[f"checkpoint_{epoch}__{name}__{reader}__{key}"] = value
        if set(expected_raw) != set(self.raw): raise CheckFailure("evaluation raw inventory drifted")
        for key, value in expected_raw.items():
            if not np.array_equal(np.asarray(value), self.raw[key]): raise CheckFailure(f"evaluation raw drifted: {key}")
        if set(self.boot) != set(expected_boot): raise CheckFailure("bootstrap inventory drifted")
        for key, value in expected_boot.items(): assert_array(value, self.boot[key], f"bootstrap {key}")

        def expected_estimand(row_id: str, instance: str, left_name: str, right_name: str, left: np.ndarray, right: np.ndarray, indices: np.ndarray | None, allow_zero: bool, evaluable: bool = True, diagnostic: bool = False) -> dict[str, Any]:
            if not evaluable or indices is None or not len(left): return {"id": row_id, "instance": instance, "left": left_name, "right": right_name, "orientation": "left_minus_right", "status": "NOT_EVALUABLE", "diagnostic_only": diagnostic, "n_tokens": int(len(left))}
            delta = left - right; draws = delta[indices].mean(axis=1); low, high = np.percentile(draws, self.config["bootstrap"]["percentiles"])
            summary = {"mean_diff": float(delta.mean()), "ci95_low": float(low), "ci95_high": float(high), "n_tokens": int(len(delta)), "n_boot": n_boot}
            if diagnostic: status = "DESCRIPTIVE_ONLY"
            elif summary["ci95_high"] <= 0.0 if allow_zero else summary["ci95_high"] < 0.0: status = "CONDITION_SATISFIED"
            elif summary["ci95_low"] > 0.0: status = "ADVERSE"
            else: status = "NOT_RESOLVED"
            return {"id": row_id, "instance": instance, "left": left_name, "right": right_name, "orientation": "left_minus_right", "status": status, "diagnostic_only": diagnostic, "allow_zero_upper": allow_zero, **summary}

        rows = [
            expected_estimand("SET_JOINT_NLL", "joint_minus_marginal", "joint_real_exact_set_nll", "marginal_real_exact_set_nll", set_rows["joint"]["real"]["exact_set_nll"], set_rows["marginal"]["real"]["exact_set_nll"], global_boot, False),
            expected_estimand("SET_JOINT_BRIER", "joint_minus_marginal", "joint_real_marginal_brier", "marginal_real_marginal_brier", set_rows["joint"]["real"]["marginal_brier"], set_rows["marginal"]["real"]["marginal_brier"], global_boot, True),
        ]
        for name in ("marginal", "joint"):
            rows.append(expected_estimand("SET_SHUFFLE", name, f"{name}_real_exact_set_nll", f"{name}_target_shuffled_exact_set_nll", set_rows[name]["real"]["exact_set_nll"], set_rows[name]["target_shuffled"]["exact_set_nll"], global_boot, False))
            for row_id, metric_name, allow in (("READER_REGRET", "regret", False), ("READER_COMPAT", "incompatibility", True), ("READER_WORST", "worst_regret", True)):
                rows.append(expected_estimand(row_id, name, f"{name}_contextual_{metric_name}", f"{name}_hard_{metric_name}", action_rows[f"{name}_contextual"][metric_name], action_rows[f"{name}_hard"][metric_name], global_boot, allow))
            common = np.asarray(self.eval_actions[f"{name}__u_common"], dtype=bool); control_regret = np.stack([value["regret"] for value in control_rows[name]])
            rows.append(expected_estimand("READER_CONTROL", name, f"{name}_contextual_regret", f"{name}_matched_control_mean_regret", action_rows[f"{name}_contextual"]["regret"][common], np.mean(control_regret[:, common], axis=0), common_boot[name], False, self.apply_status[name]["common_support_status"] == "EVALUABLE"))
        rows.append(expected_estimand("FACTOR_INTERACTION", "joint_reader_delta_minus_marginal_reader_delta", "joint_contextual_minus_hard_regret", "marginal_contextual_minus_hard_regret", action_rows["joint_contextual"]["regret"] - action_rows["joint_hard"]["regret"], action_rows["marginal_contextual"]["regret"] - action_rows["marginal_hard"]["regret"], global_boot, True, diagnostic=True))
        if len(rows) != 13 or {(row["id"], row["instance"]) for row in rows} != {(row["id"], row["instance"]) for row in self.estimands.get("rows", [])}: raise CheckFailure("estimand row inventory drifted")
        def satisfied(row_id: str, instance: str | None = None) -> bool:
            values = [row for row in rows if row["id"] == row_id and (instance is None or row["instance"] == instance)]
            return bool(values) and all(row["status"] == "CONDITION_SATISFIED" for row in values)
        expected_estimands = {"schema_version": "proportional-set-valued-estimands-v1", "status": "OPENED_DATA_PHYSICAL_PREFLIGHT", "bootstrap_unit": "pair_token", "training_seed_population_claimed": False, "scientific_decision": None, "architecture_promoted": False, "prospective_evidence": False, "decision_authority": "user", "rows": rows, "patterns": {"JOINT_PATTERN_PRESENT": all((satisfied("SET_JOINT_NLL"), satisfied("SET_JOINT_BRIER"), satisfied("SET_SHUFFLE", "joint"))), "CONTEXTUAL_PATTERN_PRESENT": {name: all(satisfied(row_id, name) for row_id in ("READER_REGRET", "READER_COMPAT", "READER_WORST", "READER_CONTROL")) for name in ("marginal", "joint")}, "interpretation": "OPENED_DATA_PHYSICAL_PREFLIGHT_ONLY"}}
        if self.estimands != expected_estimands: raise CheckFailure("estimand values/status/pattern drifted")
        sensitivity_rows = []; cardinality = self.eval_metadata["cardinality"].astype(np.int64)
        for epoch in self.config["checkpoint_epochs"]:
            for name in ("marginal", "joint"):
                for reader in ("hard", "contextual"):
                    metric = independent.action_metrics(self.eval_sensitivity[f"checkpoint_{epoch}__{name}__{reader}_actions"], target, self.utility, self.penalty)
                    for card in sorted(np.unique(cardinality).tolist()):
                        mask = cardinality == card
                        sensitivity_rows.append({"checkpoint_epoch": int(epoch), "checkpoint_is_population_seed": False, "posterior": name, "reader": reader, "cardinality": int(card), "n_tokens": int(mask.sum()), "mean_regret": float(metric["regret"][mask].mean()), "mean_incompatibility": float(metric["incompatibility"][mask].mean()), "mean_accuracy": float(metric["accuracy"][mask].mean())})
        metrics_doc["checkpoint_sensitivity"] = sensitivity_rows
        if read_json(self.root / "evaluation_truth/diagnostic_metrics.json") != metrics_doc: raise CheckFailure("diagnostic metric summary drifted")
        cell_actions = {"marginal_hard": self.eval_actions["marginal__hard_actions"], "marginal_contextual": self.eval_actions["marginal__contextual_actions"], "joint_hard": self.eval_actions["joint__hard_actions"], "joint_contextual": self.eval_actions["joint__contextual_actions"]}
        comparisons = []
        for left, right in itertools.combinations(sorted(cell_actions), 2):
            comparisons.append({"left": left, "right": right, "actions_exact": bool(np.array_equal(cell_actions[left], cell_actions[right])), "action_position_equal_fraction": float(np.mean(cell_actions[left] == cell_actions[right])), "regret_exact": bool(np.array_equal(action_rows[left]["regret_by_policy"], action_rows[right]["regret_by_policy"]))})
        if read_json(self.root / "evaluation_truth/cell_duplications.json") != {"schema_version": "proportional-cell-duplications-v1", "cells_retained_even_if_equal": True, "comparisons": comparisons}: raise CheckFailure("cell duplication inventory drifted")

    def p12(self) -> None:
        receipt = read_json(self.root / "replay_receipt.json")
        if self.reference is None:
            if receipt != {"schema_version": "proportional-physical-replay-receipt-v2", "mode": "primary", "reference_supplied": False, "byte_exact": None, "excluded_paths": [], "excluded_fields": []}: raise CheckFailure("primary replay receipt drifted")
            return
        expected_fields = ["absolute_probe_path_hashes", "archived_previous_output", "peak_rss_bytes", "phase_execution_partition", "phase_request_sha256", "recovery_archive_inventory_sha256", "runtime_stage_paths", "wall_seconds", "worker_receipt_sha256", "worker_receipt_output_hash"]
        if receipt.get("schema_version") != "proportional-physical-replay-receipt-v2" or receipt["mode"] != "replay" or receipt["byte_exact"] is not True or not receipt["semantic_exclusions_valid"] or receipt.get("excluded_fields") != expected_fields: raise CheckFailure("replay receipt drifted")
        count, normalized = compare_replay_semantics(self.root, self.reference)
        if receipt.get("compared_files") != count or receipt.get("normalized_json_files") != normalized or receipt.get("excluded_paths") != ["artifact_manifest.json", "recovery_origin.json", "replay_receipt.json"]: raise CheckFailure("replay comparison receipt drifted")

    def p13(self) -> None:
        manifest = read_json(self.root / "artifact_manifest.json")
        if set(manifest) != {"schema_version", "self_excluded", "files_and_directories"} or manifest["schema_version"] != "proportional-physical-artifact-manifest-v1" or manifest["self_excluded"] is not True: raise CheckFailure("artifact manifest schema drifted")
        actual = {path.relative_to(self.root).as_posix(): path for path in self.root.rglob("*") if path.name != "artifact_manifest.json"}
        recorded = {row["path"]: row for row in manifest["files_and_directories"]}
        if len(recorded) != len(manifest["files_and_directories"]) or set(actual) != set(recorded): raise CheckFailure("artifact inventory drifted")
        for relative, path in actual.items():
            row = recorded[relative]; info = path.lstat()
            if set(row) != {"path", "type", "class", "bytes", "sha256", "mode", "uid", "gid", "phase"}: raise CheckFailure(f"artifact manifest row schema drifted: {relative}")
            if stat.S_ISREG(info.st_mode): object_type = "file"
            elif stat.S_ISDIR(info.st_mode): object_type = "directory"
            else: raise CheckFailure(f"artifact special object forbidden: {relative}")
            if row["path"] != relative or row["mode"] != stat.S_IMODE(info.st_mode) or row["uid"] != info.st_uid or row["gid"] != info.st_gid or row["type"] != object_type or row["class"] != expected_artifact_class(relative) or row["phase"] != relative.split("/", 1)[0]: raise CheckFailure(f"artifact metadata drifted: {relative}")
            if object_type == "file" and (info.st_nlink != 1 or row["sha256"] != sha256_file(path) or row["bytes"] != info.st_size): raise CheckFailure(f"artifact bytes drifted: {relative}")
            if object_type == "directory" and (row["sha256"] is not None or row["bytes"] != 0): raise CheckFailure(f"artifact directory row drifted: {relative}")
        for path in self.root.rglob("*.json"):
            if path.read_bytes() != json_bytes(read_json(path)): raise CheckFailure(f"noncanonical JSON: {path.relative_to(self.root)}")
        for path in self.root.rglob("*.npz"):
            with zipfile.ZipFile(path) as archive:
                infos = archive.infolist()
                if [row.filename for row in infos] != sorted(row.filename for row in infos) or any(row.date_time != (1980, 1, 1, 0, 0, 0) or row.compress_type != zipfile.ZIP_DEFLATED or row.external_attr != (0o600 << 16) for row in infos): raise CheckFailure(f"noncanonical NPZ: {path.relative_to(self.root)}")

    def p14(self) -> None:
        runtime = read_json(self.root / "runtime.json")
        normalized_replay_json("runtime.json", runtime)
        expected_false = ("fresh_draw_created_or_opened", "monitor_or_lockbox_opened", "gpu_used_or_queried", "torch_imported", "architecture_promoted", "prospective_evidence")
        if runtime["status"] != "PHYSICAL_PROSPECTIVE_PACKAGE_PREFLIGHT_VALID" or runtime["execution_class"] != "OPENED_DATA_PHYSICAL_PREFLIGHT" or any(runtime[key] is not False for key in expected_false) or runtime["scientific_decision"] is not None or runtime["decision_authority"] != "user": raise CheckFailure("scope claim drifted")
        if "torch" in sys.modules or os.environ.get("CUDA_VISIBLE_DEVICES") != "": raise CheckFailure("checker CPU scope drifted")
        report = (self.root / "REPORT.md").read_text(encoding="utf-8")
        if "prospective_evidence=false" not in report or "no promueve una arquitectura" not in report: raise CheckFailure("report claim boundary drifted")

    def p15(self) -> None:
        runtime = read_json(self.root / "runtime.json")
        limit = int(self.config["budgets"]["per_process_rss_bytes"])
        if runtime["coordinator_peak_rss_bytes"] > limit or any(row["peak_rss_bytes"] > limit for row in runtime["phases_executed"]): raise CheckFailure("run RSS budget exceeded")
        if self.reference is not None:
            total = runtime["wall_seconds"] + read_json(self.reference / "runtime.json")["wall_seconds"]
            if total > self.config["budgets"]["primary_plus_replay_seconds"]: raise CheckFailure("primary plus replay budget exceeded")
        if int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024 > limit: raise CheckFailure("checker RSS budget exceeded")
        roots = (self.root,) if self.reference is None else (self.root, self.reference)
        if sum(tree_bytes(root) for root in roots) > int(self.config["budgets"]["primary_plus_replay_bytes"]): raise CheckFailure("output disk budget exceeded")
        if any(path.stat().st_size > int(self.config["budgets"]["single_file_bytes"]) for root in roots for path in root.rglob("*") if path.is_file()): raise CheckFailure("single file budget exceeded")


PREDICATES: tuple[tuple[str, str], ...] = tuple((name, f"p{index}") for index, name in enumerate(REASONS, start=1))


def _nonnegative_number(value: Any, label: str, *, integer: bool = False) -> None:
    wanted = int if integer else (int, float)
    if not isinstance(value, wanted) or isinstance(value, bool) or (not integer and not np.isfinite(value)) or value < 0:
        raise CheckFailure(f"invalid nonnegative resource: {label}")


def _validate_versions(value: Any, config: dict[str, Any]) -> None:
    expected = {"python": sys.version.split()[0], **config["versions"]}
    if value != expected: raise CheckFailure("receipt versions drifted")


def _validate_hash_rows(value: Any, expected_paths: list[Path], label: str) -> None:
    expected = {path.resolve().relative_to(REPO_ROOT).as_posix(): {"bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in expected_paths}
    if value != expected: raise CheckFailure(f"receipt {label} hashes drifted")


def _validate_checker_stdout(value: Any) -> None:
    if not isinstance(value, dict) or set(value) != {"status", "passed", "total", "checks", "wall_seconds", "peak_rss_bytes"}: raise CheckFailure("checker stdout schema drifted")
    if value["status"] != "PASS" or value["passed"] != 15 or value["total"] != 15: raise CheckFailure("checker stdout result drifted")
    checks = value["checks"]
    if not isinstance(checks, list) or len(checks) != 15 or [row.get("id") for row in checks if isinstance(row, dict)] != [name for name, _ in PREDICATES]: raise CheckFailure("checker predicate coverage drifted")
    if any(set(row) != {"id", "status", "reason_code"} or row["status"] != "PASS" or row["reason_code"] is not None for row in checks): raise CheckFailure("checker predicate row drifted")
    _nonnegative_number(value["wall_seconds"], "checker stdout wall"); _nonnegative_number(value["peak_rss_bytes"], "checker stdout RSS", integer=True)


def _validate_short_suite_receipt(row: Any, suite: str, argv: list[str], inputs: list[Path], outputs: list[Path], total: int, freeze_sha: str, config: dict[str, Any]) -> None:
    keys = {"schema_version", "suite", "source_freeze_sha256", "status", "argv", "inputs", "outputs", "versions", "exit", "wall_seconds", "peak_rss_bytes", "peak_temporary_bytes", "preserved_bytes_before_receipt", "gpu_used_or_queried", "stdout_last_json", "stdout_sha256", "stderr_sha256", "passed", "total"}
    if not isinstance(row, dict) or set(row) != keys or row["schema_version"] != "proportional-physical-suite-receipt-v2" or row["suite"] != suite or row["source_freeze_sha256"] != freeze_sha: raise CheckFailure(f"{suite} receipt schema/freeze drifted")
    if row["status"] != "PASS" or row["exit"] != 0 or row["gpu_used_or_queried"] is not False or row["passed"] != total or row["total"] != total or row["argv"] != argv: raise CheckFailure(f"{suite} receipt result/argv drifted")
    _validate_versions(row["versions"], config); _validate_hash_rows(row["inputs"], inputs, f"{suite} input"); _validate_hash_rows(row["outputs"], outputs, f"{suite} output")
    for name in ("stdout_sha256", "stderr_sha256"):
        value = row[name]
        if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value): raise CheckFailure(f"{suite} stream digest drifted")
    _nonnegative_number(row["wall_seconds"], f"{suite} wall"); _nonnegative_number(row["peak_rss_bytes"], f"{suite} RSS", integer=True); _nonnegative_number(row["peak_temporary_bytes"], f"{suite} temporary", integer=True); _nonnegative_number(row["preserved_bytes_before_receipt"], f"{suite} preserved", integer=True)
    if row["peak_temporary_bytes"] != 0 or row["preserved_bytes_before_receipt"] != sum(path.stat().st_size for path in outputs): raise CheckFailure(f"{suite} byte accounting drifted")
    if suite == "unit_test":
        if row["stdout_last_json"] is not None: raise CheckFailure("unit receipt stdout payload drifted")
    else: _validate_checker_stdout(row["stdout_last_json"])


def check_evidence(path: Path) -> dict[str, Any]:
    root = path.resolve(strict=True); manifest = read_json(root / "evidence_manifest.json")
    required = {"unit_test_receipt.json", "primary_check_receipt.json", "replay_check_receipt.json", "mutation_receipt.json", "recovery_receipt.json"}
    actual = {item.name for item in root.iterdir() if item.is_file() and item.name != "evidence_manifest.json"}
    if actual != required: raise CheckFailure("evidence receipt inventory drifted")
    if not isinstance(manifest, dict) or set(manifest) != {"schema_version", "self_excluded", "source_freeze_sha256", "primary_artifact_manifest_sha256", "replay_artifact_manifest_sha256", "files"} or manifest["schema_version"] != "proportional-physical-evidence-manifest-v1" or manifest["self_excluded"] is not True or not isinstance(manifest["files"], list): raise CheckFailure("evidence manifest schema drifted")
    if any(not isinstance(row, dict) or set(row) != {"path", "bytes", "sha256"} for row in manifest["files"]): raise CheckFailure("evidence manifest row schema drifted")
    recorded = {row["path"]: row for row in manifest["files"]}
    if len(recorded) != len(manifest["files"]) or set(recorded) != actual or any(sha256_file(root / name) != row["sha256"] or (root / name).stat().st_size != row["bytes"] for name, row in recorded.items()): raise CheckFailure("evidence manifest drifted")
    freeze = FREEZE_DEFAULT.resolve(strict=True); freeze_sha = sha256_file(freeze)
    config = read_json(CONFIG_DEFAULT)
    unit = read_json(root / "unit_test_receipt.json")
    primary = read_json(root / "primary_check_receipt.json")
    replay = read_json(root / "replay_check_receipt.json")
    mutations = read_json(root / "mutation_receipt.json")
    recovery = read_json(root / "recovery_receipt.json")
    input_package = REPO_ROOT / "data/geometria_proporcional/proportional_set_valued_physical_input_v1"
    primary_root = REPO_ROOT / "data/geometria_proporcional/proportional_set_valued_physical_preflight_v1"
    replay_root = REPO_ROOT / "data/geometria_proporcional/proportional_set_valued_physical_preflight_replay_v1"
    test_path = REPO_ROOT / "tests/test_proportional_set_valued_physical.py"
    python = str(REPO_ROOT / "venv/bin/python"); checker_path = str(Path(__file__).resolve())
    _validate_short_suite_receipt(unit, "unit_test", [python, "-m", "unittest", "tests.test_proportional_set_valued_physical", "-v"], [freeze], [test_path], 15, freeze_sha, config)
    _validate_short_suite_receipt(primary, "primary_check", [python, checker_path, "--artifact", str(primary_root), "--input-package", str(input_package)], [freeze, input_package / "preparation_freeze.json", primary_root / "artifact_manifest.json"], [primary_root / "artifact_manifest.json"], 15, freeze_sha, config)
    _validate_short_suite_receipt(replay, "replay_check", [python, checker_path, "--artifact", str(replay_root), "--input-package", str(input_package), "--reference", str(primary_root)], [freeze, input_package / "preparation_freeze.json", primary_root / "artifact_manifest.json", replay_root / "artifact_manifest.json"], [replay_root / "artifact_manifest.json"], 15, freeze_sha, config)
    mutation_keys = {"schema_version", "catalogue_version", "catalogue_sha256", "status", "argv", "source_freeze_sha256", "inputs", "outputs", "versions", "exit", "baseline", "cases", "passed", "total", "wall_seconds", "peak_rss_bytes", "children_peak_rss_bytes", "peak_temporary_bytes", "preserved_bytes_before_receipt", "gpu_used_or_queried"}
    recovery_keys = {"schema_version", "status", "argv", "source_freeze_sha256", "inputs", "outputs", "versions", "exit", "cases", "passed", "total", "wall_seconds", "peak_rss_bytes", "children_peak_rss_bytes", "peak_temporary_bytes", "preserved_bytes_before_receipt", "gpu_used_or_queried"}
    if set(mutations) != mutation_keys or set(recovery) != recovery_keys: raise CheckFailure("long receipt schema drifted")
    for label, row in (("mutation", mutations), ("recovery", recovery)):
        if row["status"] != "PASS" or row["exit"] != 0 or row["gpu_used_or_queried"] is not False: raise CheckFailure(f"{label} receipt result drifted")
        _validate_versions(row["versions"], config)
        for field in ("wall_seconds",): _nonnegative_number(row[field], f"{label} {field}")
        for field in ("peak_rss_bytes", "children_peak_rss_bytes", "peak_temporary_bytes", "preserved_bytes_before_receipt"): _nonnegative_number(row[field], f"{label} {field}", integer=True)
    expected_mutation_argv = ["tests/run_proportional_set_valued_physical_mutations.py", "--artifact", "data/geometria_proporcional/proportional_set_valued_physical_preflight_replay_v1", "--input-package", "data/geometria_proporcional/proportional_set_valued_physical_input_v1", "--reference", "data/geometria_proporcional/proportional_set_valued_physical_preflight_v1", "--receipt", "data/geometria_proporcional/proportional_set_valued_physical_evidence_v1/mutation_receipt.json"]
    expected_recovery_argv = ["tests/run_proportional_set_valued_physical_recovery.py", "--reference", "data/geometria_proporcional/proportional_set_valued_physical_preflight_v1", "--input-package", "data/geometria_proporcional/proportional_set_valued_physical_input_v1", "--receipt", "data/geometria_proporcional/proportional_set_valued_physical_evidence_v1/recovery_receipt.json"]
    if mutations["argv"] != expected_mutation_argv or recovery["argv"] != expected_recovery_argv: raise CheckFailure("long receipt argv drifted")
    catalogue = config["mutation_catalogue"]
    mutation_ids = sorted(row.get("case_id", "") for row in mutations.get("cases", []))
    mutation_sha = hashlib.sha256(("\n".join(mutation_ids) + "\n").encode()).hexdigest()
    mutation_case_keys = {"case_id", "single_mutation", "mutation_object", "predicate_expected", "reason_code_expected", "reason_code_observed", "exit", "passed", "requirements"}
    if mutations.get("source_freeze_sha256") != freeze_sha or mutations.get("schema_version") != "proportional-physical-mutation-receipt-v3" or mutations.get("catalogue_version") != catalogue["version"] or mutations.get("catalogue_sha256") != catalogue["case_ids_sha256"] or mutation_sha != catalogue["case_ids_sha256"] or len(set(mutation_ids)) != catalogue["case_count"] or mutations.get("passed") != mutations.get("total") or mutations.get("total") != catalogue["case_count"] or mutations["outputs"] != {"case_rows": catalogue["case_count"]} or any(set(row) != mutation_case_keys for row in mutations["cases"]):
        raise CheckFailure("mutation receipt coverage/freeze drifted")
    expected_baseline_argv = [python, checker_path, "--artifact", str(replay_root), "--input-package", str(input_package), "--reference", str(primary_root)]
    baseline = mutations.get("baseline")
    expected_baseline_checks = [{"id": name, "status": "PASS", "reason_code": None} for name, _ in PREDICATES]
    if not isinstance(baseline, dict) or set(baseline) != {"argv", "exit", "status", "passed", "total", "checks"} or baseline["argv"] != expected_baseline_argv or baseline["exit"] != 0 or baseline["status"] != "PASS" or baseline["passed"] != 15 or baseline["total"] != 15 or baseline["checks"] != expected_baseline_checks:
        raise CheckFailure("mutation baseline drifted")
    coverage = catalogue["coverage"]
    if not isinstance(coverage, dict) or any(not isinstance(ids, list) or not ids or not set(ids).issubset(set(mutation_ids)) for ids in coverage.values()) or set().union(*(set(ids) for ids in coverage.values())) != set(mutation_ids): raise CheckFailure("mutation normative coverage drifted")
    reverse_coverage = {case_id: sorted(requirement for requirement, ids in coverage.items() if case_id in ids) for case_id in mutation_ids}
    if any(not row.get("passed") or row.get("single_mutation") is not True or row.get("exit") == 0 or row.get("reason_code_expected") != row.get("reason_code_observed") or row.get("requirements") != reverse_coverage[row["case_id"]] for row in mutations.get("cases", [])):
        raise CheckFailure("mutation case result drifted")
    recovery_catalogue = config["recovery_catalogue"]
    recovery_pairs = {(row.get("crash_kind"), row.get("crash_point")) for row in recovery.get("cases", [])}
    expected_pairs = {(kind, phase) for kind in recovery_catalogue["crash_intervals"] for phase in PHASES}
    recovery_case_keys = {"crash_kind", "crash_point", "crash_exit", "orphan_phase_present", "journal_absent", "recovery_action", "reference_manifest_sha256", "recovered_manifest_sha256", "scientific_byte_exact", "normalized_operational_check"}
    expected_long_inputs = {"reference_manifest_sha256": sha256_file(primary_root / "artifact_manifest.json"), "input_preparation_sha256": sha256_file(input_package / "preparation_freeze.json")}
    if recovery.get("source_freeze_sha256") != freeze_sha or recovery.get("schema_version") != "proportional-physical-recovery-receipt-v2" or recovery_pairs != expected_pairs or recovery.get("passed") != recovery.get("total") or recovery.get("total") != recovery_catalogue["case_count"] or recovery["inputs"] != expected_long_inputs or recovery["outputs"] != {"recovered_runs": 14} or any(set(row) != recovery_case_keys for row in recovery["cases"]):
        raise CheckFailure("recovery receipt coverage/freeze drifted")
    if any(row["crash_exit"] == 0 or row["orphan_phase_present"] is not True or row["scientific_byte_exact"] is not True or row["normalized_operational_check"] is not True or row["journal_absent"] != (row["crash_kind"] == "after_promotion") or row["reference_manifest_sha256"] != expected_long_inputs["reference_manifest_sha256"] or len(row["recovered_manifest_sha256"]) != 64 for row in recovery.get("cases", [])):
        raise CheckFailure("recovery case result drifted")
    expected_mutation_inputs = {"artifact_manifest_sha256": sha256_file(replay_root / "artifact_manifest.json"), **expected_long_inputs}
    if mutations["inputs"] != expected_mutation_inputs: raise CheckFailure("mutation receipt input hashes drifted")
    limits = config["budgets"]
    if unit["wall_seconds"] > limits["unit_and_permissions_seconds"]:
        raise CheckFailure("unit campaign wall budget exceeded")
    if primary["wall_seconds"] + replay["wall_seconds"] > limits["primary_plus_replay_seconds"]:
        raise CheckFailure("checker campaign wall budget exceeded")
    if mutations["wall_seconds"] > limits["checker_plus_mutations_seconds"] or recovery["wall_seconds"] > limits["recovery_seconds"]:
        raise CheckFailure("mutation/recovery wall budget exceeded")
    rss_fields = [unit["peak_rss_bytes"], primary["peak_rss_bytes"], replay["peak_rss_bytes"], mutations["peak_rss_bytes"], mutations["children_peak_rss_bytes"], recovery["peak_rss_bytes"], recovery["children_peak_rss_bytes"]]
    if any(value > limits["per_process_rss_bytes"] for value in rss_fields):
        raise CheckFailure("evidence campaign RSS budget exceeded")
    if mutations["peak_temporary_bytes"] > limits["scratch_bytes"] or recovery["peak_temporary_bytes"] > limits["scratch_bytes"]:
        raise CheckFailure("evidence scratch budget exceeded")
    if sum((root / name).stat().st_size for name in actual | {"evidence_manifest.json"}) > limits["evidence_bytes"]:
        raise CheckFailure("preserved evidence budget exceeded")
    if any((root / name).stat().st_size > limits["single_file_bytes"] for name in actual | {"evidence_manifest.json"}):
        raise CheckFailure("evidence single-file budget exceeded")
    if manifest.get("source_freeze_sha256") != freeze_sha or manifest.get("primary_artifact_manifest_sha256") != sha256_file(primary_root / "artifact_manifest.json") or manifest.get("replay_artifact_manifest_sha256") != sha256_file(replay_root / "artifact_manifest.json"):
        raise CheckFailure("evidence root binding drifted")
    return {"status": "PASS", "files": len(actual), "unit": "15/15", "primary": "15/15", "replay": "15/15", "mutations": f"{mutations['total']}/{mutations['total']}", "recovery": "14/14", "manifest_sha256": sha256_file(root / "evidence_manifest.json")}


def main() -> int:
    args = parse_args(); started = time.monotonic()
    if args.evidence:
        try: payload = check_evidence(args.evidence); print(json.dumps(payload, sort_keys=True)); return 0
        except Exception as error: print(json.dumps({"status": "FAIL", "reason_code": "INVENTORY_INVALID", "error": str(error)}, sort_keys=True)); return 1
    if args.artifact is None: raise SystemExit("--artifact is required unless --evidence is used")
    results = []
    try: checker = Checker(args.artifact, args.input_package, args.reference, args.config, args.source_freeze)
    except Exception as error:
        print(json.dumps({"status": "FAIL", "reason_code": "AUTHORITY_INVALID", "error": str(error)}, sort_keys=True)); return 1
    predicates = PREDICATES if args.only is None else tuple(row for row in PREDICATES if row[0] == args.only)
    for name, method in predicates:
        try:
            getattr(checker, method)(); results.append({"id": name, "status": "PASS", "reason_code": None})
        except Exception as error:
            results.append({"id": name, "status": "FAIL", "reason_code": REASONS[name], "error": f"{type(error).__name__}: {error}"})
            print(json.dumps({"status": "FAIL", "checks": results, "failed": name, "reason_code": REASONS[name], "wall_seconds": time.monotonic() - started}, sort_keys=True)); return 1
    print(json.dumps({"status": "PASS", "passed": len(results), "total": len(predicates), "checks": results, "wall_seconds": time.monotonic() - started, "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024}, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
