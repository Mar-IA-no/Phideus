#!/usr/bin/env python3
"""Run single-fault mutations against each physical-package predicate."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import resource
import shutil
import stat
import subprocess
import sys
import time
from typing import Any, Callable
import zipfile

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CHECKER = ROOT / "experiments/geometria_proporcional/check_proportional_set_valued_physical_preflight.py"
REASONS = {
    "P1_AUTHORITY": "AUTHORITY_INVALID", "P2_PREPARATION": "PREPARATION_INVALID", "P3_PHYSICAL_BOUNDARY": "PHYSICAL_BOUNDARY_INVALID", "P4_STATE_MACHINE": "STATE_MACHINE_INVALID", "P5_POSTERIOR": "POSTERIOR_INVALID", "P6_POLICY": "POLICY_INVALID", "P7_SELECTION_PROPOSE": "SELECTION_PROPOSE_INVALID", "P8_SELECTION_EVALUATE": "SELECTION_EVALUATE_INVALID", "P9_SELECTION_FREEZE": "SELECTION_FREEZE_INVALID", "P10_EVALUATION_APPLY": "EVALUATION_APPLY_INVALID", "P11_EVALUATION_TRUTH": "EVALUATION_TRUTH_INVALID", "P12_RESTART_REPLAY": "RESTART_OR_REPLAY_INVALID", "P13_INVENTORY": "INVENTORY_INVALID", "P14_SCOPE": "SCOPE_INVALID", "P15_COST": "COST_INVALID",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--input-package", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument(
        "--work-root",
        type=Path,
        default=ROOT / ".agent-work/proportional-set-valued-physical-campaign/mutations",
    )
    parser.add_argument("--receipt", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8")


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive: return {key: archive[key].copy() for key in archive.files}


def write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in sorted(arrays):
            raw = io.BytesIO(); np.lib.format.write_array(raw, np.ascontiguousarray(arrays[name]), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0)); info.compress_type = zipfile.ZIP_DEFLATED; info.external_attr = 0o600 << 16
            archive.writestr(info, raw.getvalue(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    path.write_bytes(buffer.getvalue())


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def object_snapshot(artifact: Path, input_package: Path, config: Path, freeze: Path) -> dict[str, tuple[Any, ...]]:
    snapshot: dict[str, tuple[Any, ...]] = {}
    for namespace, root in (("artifact", artifact), ("input", input_package)):
        for path in sorted(root.rglob("*")):
            relative = path.relative_to(root).as_posix()
            info = path.lstat()
            if stat.S_ISREG(info.st_mode): kind, content = "file", sha256_file(path)
            elif stat.S_ISDIR(info.st_mode): kind, content = "directory", None
            elif stat.S_ISLNK(info.st_mode): kind, content = "symlink", os.readlink(path)
            elif stat.S_ISFIFO(info.st_mode): kind, content = "fifo", None
            else: kind, content = "special", None
            snapshot[f"{namespace}/{relative}"] = (
                kind, stat.S_IMODE(info.st_mode), info.st_uid, info.st_gid,
                info.st_nlink, content,
            )
    for namespace, path in (("config", config), ("freeze", freeze)):
        snapshot[namespace] = ("file", path.stat().st_mode & 0o777, sha256_file(path))
    return snapshot


def writable(root: Path) -> None:
    for path in root.rglob("*"):
        path.chmod(0o700 if path.is_dir() else 0o600)
    root.chmod(0o700)


def rebind_probe_hashes(artifact: Path, input_package: Path) -> None:
    truth = input_package / "prepared/truth"
    mapping = {
        "posterior_fit": [truth / "policy_fit_truth.npz", truth / "decision_select_truth.npz", truth / "evaluate_truth.npz"],
        "policy_fit": [truth / "posterior_fit_truth.npz", truth / "decision_select_truth.npz", truth / "evaluate_truth.npz"],
        "selection_propose": [truth / name for name in ("posterior_fit_truth.npz", "policy_fit_truth.npz", "decision_select_truth.npz", "evaluate_truth.npz")],
        "selection_evaluate": [artifact / "posterior_fit/posterior_states.json", artifact / "policy_fit/policy_states.json", truth / "evaluate_truth.npz"],
        "selection_freeze": [truth / "decision_select_truth.npz", artifact / "selection_evaluate/candidate_metrics_private.npz", truth / "evaluate_truth.npz"],
        "evaluation_apply": [truth / "decision_select_truth.npz", artifact / "selection_evaluate/candidate_metrics_private.npz", truth / "evaluate_truth.npz"],
        "evaluation_truth": [artifact / "posterior_fit/posterior_states.json", artifact / "policy_fit/policy_states.json", truth / "decision_select_truth.npz", artifact / "selection_freeze/selection_policy.json"],
    }
    for phase, paths in mapping.items():
        receipt_path = artifact / phase / "worker_receipt.json"; receipt = read_json(receipt_path)
        hashes = [hashlib.sha256(str(path.resolve()).encode()).hexdigest() for path in paths]
        receipt["stage_contract"]["probe_path_sha256"] = hashes
        for row, digest in zip(receipt["probes"], hashes, strict=True): row["path_sha256"] = digest
        write_json(receipt_path, receipt)


def jmut(relative: str, change: Callable[[Any], None], *, input_side: bool = False) -> Callable[[Path, Path, Path, Path], None]:
    def apply(artifact: Path, input_package: Path, config: Path, freeze: Path) -> None:
        path = (input_package if input_side else artifact) / relative; payload = read_json(path); change(payload); write_json(path, payload)
    return apply


def nmut(relative: str, key: str, change: Callable[[np.ndarray], None], *, input_side: bool = False) -> Callable[[Path, Path, Path, Path], None]:
    def apply(artifact: Path, input_package: Path, config: Path, freeze: Path) -> None:
        path = (input_package if input_side else artifact) / relative; arrays = load_npz(path); change(arrays[key]); write_npz(path, arrays)
    return apply


def npz_mut(relative: str, change: Callable[[dict[str, np.ndarray]], None], *, input_side: bool = False) -> Callable[[Path, Path, Path, Path], None]:
    def apply(artifact: Path, input_package: Path, config: Path, freeze: Path) -> None:
        path = (input_package if input_side else artifact) / relative; arrays = load_npz(path); change(arrays); write_npz(path, arrays)
    return apply


def npy_mut(relative: str, change: Callable[[np.ndarray], None], *, input_side: bool = False) -> Callable[[Path, Path, Path, Path], None]:
    def apply(artifact: Path, input_package: Path, config: Path, freeze: Path) -> None:
        path = (input_package if input_side else artifact) / relative; array = np.load(path, allow_pickle=False); change(array); np.save(path, array, allow_pickle=False)
    return apply


def config_mut(change: Callable[[Any], None]) -> Callable[[Path, Path, Path, Path], None]:
    def apply(artifact: Path, input_package: Path, config: Path, freeze: Path) -> None:
        payload = read_json(config); change(payload); write_json(config, payload)
    return apply


def freeze_mut(change: Callable[[Any], None]) -> Callable[[Path, Path, Path, Path], None]:
    def apply(artifact: Path, input_package: Path, config: Path, freeze: Path) -> None:
        payload = read_json(freeze); change(payload); write_json(freeze, payload)
    return apply


def cases() -> list[tuple[str, str, Callable[[Path, Path, Path, Path], None]]]:
    rows: list[tuple[str, str, Callable[[Path, Path, Path, Path], None]]] = []
    add = lambda predicate, name, fn: rows.append((f"{predicate.lower()}_{name}", predicate, fn))
    add("P1_AUTHORITY", "config_schema", config_mut(lambda p: p.__setitem__("schema_version", "mutated")))
    add("P1_AUTHORITY", "config_class", config_mut(lambda p: p.__setitem__("enabled_execution_class", "FRESH_PROSPECTIVE")))
    add("P1_AUTHORITY", "freeze_schema", freeze_mut(lambda p: p.__setitem__("schema_version", "mutated")))
    add("P1_AUTHORITY", "freeze_digest", freeze_mut(lambda p: p["files"].__setitem__(next(iter(p["files"])), "0" * 64)))
    add("P1_AUTHORITY", "freeze_parent", freeze_mut(lambda p: p.__setitem__("implementation_commit", "0" * 40)))
    add("P2_PREPARATION", "decision_cardinality", nmut("prepared/public/decision_select_public.npz", "cardinality", lambda a: a.__setitem__(0, a[0] + 1), input_side=True))
    add("P2_PREPARATION", "decision_token", nmut("prepared/truth/decision_select_truth.npz", "pair_token", lambda a: a.__setitem__(0, "mutated"), input_side=True))
    add("P2_PREPARATION", "evaluate_cardinality", nmut("prepared/public/evaluate_public.npz", "cardinality", lambda a: a.__setitem__(0, a[0] + 1), input_side=True))
    add("P2_PREPARATION", "evaluate_token", nmut("prepared/truth/evaluate_truth.npz", "pair_token", lambda a: a.__setitem__(0, "mutated"), input_side=True))
    add("P2_PREPARATION", "ensemble", nmut("prepared/public/evaluate_public.npz", "ensemble_logits", lambda a: a.__setitem__((0, 0), a[0, 0] + 1e-3), input_side=True))
    add("P2_PREPARATION", "stratum", nmut("prepared/public/decision_select_public.npz", "design_stratum", lambda a: a.__setitem__(0, "UNKNOWN"), input_side=True))
    add("P2_PREPARATION", "duplicate_token", nmut("prepared/truth/policy_fit_truth.npz", "pair_token", lambda a: a.__setitem__(1, a[0]), input_side=True))
    add("P2_PREPARATION", "cluster_identity", nmut("prepared/truth/posterior_fit_truth.npz", "cluster_id", lambda a: a.__setitem__(0, "mutated"), input_side=True))
    add("P2_PREPARATION", "full_key_extra", npz_mut("prepared/truth/posterior_fit_truth.npz", lambda a: a.__setitem__("undeclared", np.zeros(len(a["pair_token"]), dtype="<f8")), input_side=True))
    add("P2_PREPARATION", "public_truth_key", npz_mut("prepared/public/decision_select_public.npz", lambda a: a.__setitem__("target", np.zeros((len(a["pair_token"]), 4), dtype=bool)), input_side=True))
    add("P2_PREPARATION", "dtype", npz_mut("prepared/truth/policy_fit_truth.npz", lambda a: a.__setitem__("cardinality", a["cardinality"].astype("<i4")), input_side=True))
    add("P2_PREPARATION", "shape", npz_mut("prepared/truth/policy_fit_truth.npz", lambda a: a.__setitem__("ensemble_logits", a["ensemble_logits"][:, :3]), input_side=True))
    add("P2_PREPARATION", "split_role", nmut("prepared/truth/policy_fit_truth.npz", "split_role", lambda a: a.__setitem__(0, "evaluate"), input_side=True))
    add("P2_PREPARATION", "nonfinite", nmut("prepared/truth/policy_fit_truth.npz", "per_seed_logits", lambda a: a.__setitem__((0, 0, 0), np.nan), input_side=True))
    add("P2_PREPARATION", "checkpoint_axis", jmut("preparation_freeze.json", lambda p: p["checkpoint_axis"].__setitem__(0, p["checkpoint_axis"][0] + 1), input_side=True))
    add("P2_PREPARATION", "generation_escrow", jmut("opened_fixture_escrow.json", lambda p: p.__setitem__("generation_escrow", True), input_side=True))
    def input_extra(artifact: Path, input_package: Path, config: Path, freeze: Path) -> None: (input_package / "undeclared.bin").write_bytes(b"extra")
    add("P2_PREPARATION", "file_extra", input_extra)
    def input_symlink(artifact: Path, input_package: Path, config: Path, freeze: Path) -> None: (input_package / "undeclared-link").symlink_to("preparation_freeze.json")
    add("P2_PREPARATION", "symlink", input_symlink)
    def input_hardlink(artifact: Path, input_package: Path, config: Path, freeze: Path) -> None:
        external = input_package.parent / "external-hardlink-source"; external.write_bytes(b"external"); os.link(external, input_package / "undeclared-hardlink")
    add("P2_PREPARATION", "hardlink_external", input_hardlink)
    def input_fifo(artifact: Path, input_package: Path, config: Path, freeze: Path) -> None: os.mkfifo(input_package / "undeclared-fifo")
    add("P2_PREPARATION", "fifo", input_fifo)
    def input_empty_dir(artifact: Path, input_package: Path, config: Path, freeze: Path) -> None: (input_package / "undeclared-empty").mkdir()
    add("P2_PREPARATION", "empty_directory", input_empty_dir)
    add("P2_PREPARATION", "prepared_hash", jmut("preparation_freeze.json", lambda p: p["files"].__setitem__(next(iter(p["files"])), "0" * 64), input_side=True))
    add("P2_PREPARATION", "public_manifest_hash", jmut("public_manifest.json", lambda p: p["files"][next(iter(p["files"]))].__setitem__("sha256", "0" * 64), input_side=True))
    add("P2_PREPARATION", "escrow_package", jmut("opened_fixture_escrow.json", lambda p: p.__setitem__("package_id", "0" * 64), input_side=True))
    add("P2_PREPARATION", "receipt_claim", jmut("preparation_receipt.json", lambda p: p.__setitem__("pairwise_disjoint", False), input_side=True))
    def input_mode(artifact: Path, input_package: Path, config: Path, freeze: Path) -> None: (input_package / "prepared/public/evaluate_public.npz").chmod(0o666)
    add("P2_PREPARATION", "public_mode", input_mode)
    for index, field in enumerate(("Uid", "Gid", "Groups", "NoNewPrivs", "CapEff")):
        value = "0\t0\t0\t0" if field in {"Uid", "Gid"} else ("1" if field == "Groups" else ("0" if field == "NoNewPrivs" else "0000000000000001"))
        add("P3_PHYSICAL_BOUNDARY", f"identity_{index}", jmut("posterior_fit/worker_receipt.json", lambda p, f=field, v=value: p["runtime"]["identity"].__setitem__(f, v)))
    add("P3_PHYSICAL_BOUNDARY", "cuda_env", jmut("policy_fit/worker_receipt.json", lambda p: p["runtime"]["environment"].__setitem__("CUDA_VISIBLE_DEVICES", "0")))
    add("P3_PHYSICAL_BOUNDARY", "probe", jmut("selection_propose/worker_receipt.json", lambda p: p["probes"][0].__setitem__("outcome", "OPENED")))
    add("P3_PHYSICAL_BOUNDARY", "runtime_hash", jmut("posterior_fit/worker_receipt.json", lambda p: p["runtime"]["runtime_files"]["wave49_schema.py"].__setitem__("sha256", "0" * 64)))
    add("P3_PHYSICAL_BOUNDARY", "runtime_path", jmut("policy_fit/worker_receipt.json", lambda p: p["runtime"]["runtime_files"]["wave54_joint_set.py"].__setitem__("path", "/tmp/wave54_joint_set.py")))
    add("P3_PHYSICAL_BOUNDARY", "module_hash", jmut("selection_propose/worker_receipt.json", lambda p: p["runtime"]["modules"]["geometria_proporcional.wave53_uncertainty"].__setitem__("sha256", "0" * 64)))
    add("P3_PHYSICAL_BOUNDARY", "sys_path", jmut("selection_evaluate/worker_receipt.json", lambda p: p["runtime"]["sys_path"].append("/tmp/user-site")))
    add("P3_PHYSICAL_BOUNDARY", "environment_extra", jmut("selection_freeze/worker_receipt.json", lambda p: p["runtime"]["environment"].__setitem__("HOME", "/root")))
    add("P3_PHYSICAL_BOUNDARY", "cwd", jmut("selection_freeze/worker_receipt.json", lambda p: p["runtime"].__setitem__("cwd", "/tmp")))
    add("P3_PHYSICAL_BOUNDARY", "torch_import", jmut("evaluation_apply/worker_receipt.json", lambda p: p["runtime"].__setitem__("torch_imported", True)))
    add("P3_PHYSICAL_BOUNDARY", "threadpool", jmut("evaluation_truth/worker_receipt.json", lambda p: p["runtime"]["threadpools"][0].__setitem__("num_threads", 2)))
    add("P3_PHYSICAL_BOUNDARY", "stage_extra", jmut("evaluation_apply/worker_receipt.json", lambda p: p["stage_contract"]["allowed_files"].append("truth.npz")))
    add("P3_PHYSICAL_BOUNDARY", "stage_missing", jmut("selection_evaluate/worker_receipt.json", lambda p: p["runtime"]["stage_metadata"]["files"].pop(next(iter(p["runtime"]["stage_metadata"]["files"])))))
    add("P3_PHYSICAL_BOUNDARY", "private_metrics_to_freeze", jmut("selection_freeze/worker_receipt.json", lambda p: p["stage_contract"]["allowed_files"].append("candidate_metrics_private.npz")))
    add("P3_PHYSICAL_BOUNDARY", "stage_hash", jmut("evaluation_truth/worker_receipt.json", lambda p: p["stage_files"].__setitem__(next(iter(p["stage_files"])), "0" * 64)))
    add("P3_PHYSICAL_BOUNDARY", "stage_mode", jmut("posterior_fit/worker_receipt.json", lambda p: p["runtime"]["stage_metadata"].__setitem__("mode", 0o777)))
    add("P3_PHYSICAL_BOUNDARY", "probe_omit", jmut("policy_fit/worker_receipt.json", lambda p: p["probes"].pop()))
    add("P3_PHYSICAL_BOUNDARY", "receipt_schema", jmut("selection_propose/worker_receipt.json", lambda p: p.__setitem__("schema_version", "fabricated")))
    add("P3_PHYSICAL_BOUNDARY", "output_hash", jmut("selection_freeze/worker_receipt.json", lambda p: p["output_files"].__setitem__(next(iter(p["output_files"])), "0" * 64)))
    add("P4_STATE_MACHINE", "previous", jmut("journals/policy_fit.json", lambda p: p.__setitem__("previous_state", "PREPARED")))
    add("P4_STATE_MACHINE", "new", jmut("journals/selection_propose.json", lambda p: p.__setitem__("new_state", "COMPLETE")))
    add("P4_STATE_MACHINE", "truth_level", jmut("journals/evaluation_apply.json", lambda p: p.__setitem__("maximum_truth_materialized", "EVALUATE")))
    add("P4_STATE_MACHINE", "package", jmut("journals/posterior_fit.json", lambda p: p.__setitem__("package_id", "0" * 64)))
    add("P4_STATE_MACHINE", "input_hash", jmut("journals/selection_freeze.json", lambda p: p["input_hashes"].__setitem__(next(iter(p["input_hashes"])), "0" * 64)))
    def future_journal(artifact: Path, input_package: Path, config: Path, freeze: Path) -> None: write_json(artifact / "journals/future.json", {"phase": "future"})
    add("P4_STATE_MACHINE", "unknown_journal", future_journal)
    def missing_journal(artifact: Path, input_package: Path, config: Path, freeze: Path) -> None: (artifact / "journals/evaluation_apply.json").unlink()
    add("P4_STATE_MACHINE", "output_without_journal", missing_journal)
    def future_output(artifact: Path, input_package: Path, config: Path, freeze: Path) -> None: (artifact / "future_phase").mkdir()
    add("P4_STATE_MACHINE", "future_output", future_output)
    add("P5_POSTERIOR", "donor", nmut("posterior_fit/target_shuffle_arrays.npz", "donor_index", lambda a: a.__setitem__(0, a[0] + 1)))
    add("P5_POSTERIOR", "permutable", nmut("posterior_fit/target_shuffle_arrays.npz", "permutable", lambda a: a.__setitem__(0, ~a[0])))
    add("P5_POSTERIOR", "target", nmut("posterior_fit/target_shuffle_arrays.npz", "target_shuffled", lambda a: a.__setitem__((0, 0), ~a[0, 0])))
    add("P5_POSTERIOR", "map", jmut("posterior_fit/target_shuffle_map.json", lambda p: p[0].__setitem__("donor_pair_token", "mutated")))
    add("P5_POSTERIOR", "state", jmut("posterior_fit/posterior_states.json", lambda p: p["marginal"]["real"].__setitem__("intercept", p["marginal"]["real"]["intercept"] + 1e-6)))
    add("P5_POSTERIOR", "state_array", nmut("posterior_fit/posterior_state_arrays.npz", next(iter(load_npz(ROOT / "data/geometria_proporcional/proportional_set_valued_physical_preflight_v1/posterior_fit/posterior_state_arrays.npz"))), lambda a: a.flat.__setitem__(0, a.flat[0] + 1e-6)))
    add("P5_POSTERIOR", "oof", nmut("posterior_fit/posterior_oof_arrays.npz", "joint_real__oof_exact_set_nll", lambda a: a.flat.__setitem__(0, a.flat[0] + 1e-6)))
    add("P6_POLICY", "feature_count", jmut("policy_fit/feature_schema.json", lambda p: p.__setitem__("count", 16)))
    add("P6_POLICY", "feature_name", jmut("policy_fit/feature_schema.json", lambda p: p["feature_names"].__setitem__(0, "mutated")))
    add("P6_POLICY", "control_remove", jmut("policy_fit/policy_states.json", lambda p: p["marginal"]["controls"].pop()))
    add("P6_POLICY", "control_seed", jmut("policy_fit/policy_states.json", lambda p: p["marginal"]["controls"][0].__setitem__("seed", 1)))
    add("P6_POLICY", "state_array", nmut("policy_fit/policy_state_arrays.npz", next(iter(load_npz(ROOT / "data/geometria_proporcional/proportional_set_valued_physical_preflight_v1/policy_fit/policy_state_arrays.npz"))), lambda a: a.flat.__setitem__(0, a.flat[0] + 1e-6)))
    add("P6_POLICY", "fit_score", nmut("policy_fit/policy_fit_private.npz", "joint__gain", lambda a: a.flat.__setitem__(0, a.flat[0] + 1e-6)))
    add("P6_POLICY", "control_array", nmut("policy_fit/control_arrays.npz", next(iter(load_npz(ROOT / "data/geometria_proporcional/proportional_set_valued_physical_preflight_v1/policy_fit/control_arrays.npz"))), lambda a: a.flat.__setitem__(0, a.flat[0] + 1)))
    add("P6_POLICY", "handoff_private_key", npz_mut("policy_fit/policy_state_arrays.npz", lambda p: p.__setitem__("gain", np.zeros(1))))
    add("P7_SELECTION_PROPOSE", "threshold", jmut("selection_propose/apply_metadata.json", lambda p: p["posteriors"]["marginal"][0].__setitem__("proposer_threshold", p["posteriors"]["marginal"][0]["proposer_threshold"] + 1e-3)))
    add("P7_SELECTION_PROPOSE", "action", nmut("selection_propose/candidate_public.npz", "marginal__actions", lambda a: a.__setitem__((0, 0, 0), (a[0, 0, 0] + 1) % 4)))
    add("P7_SELECTION_PROPOSE", "override", nmut("selection_propose/candidate_public.npz", "joint__override", lambda a: a.__setitem__((0, 0, 0), ~a[0, 0, 0])))
    add("P7_SELECTION_PROPOSE", "key_extra", jmut("selection_propose/selection_key_metadata.json", lambda p: p["posteriors"]["joint"][0].__setitem__("threshold", 0.0)))
    add("P7_SELECTION_PROPOSE", "candidate_omitted", npz_mut("selection_propose/candidate_public.npz", lambda p: p.pop("joint__actions")))
    add("P7_SELECTION_PROPOSE", "metadata_reordered", jmut("selection_propose/selection_key_metadata.json", lambda p: p["posteriors"]["marginal"].reverse()))
    add("P7_SELECTION_PROPOSE", "private_metric", npz_mut("selection_propose/candidate_public.npz", lambda p: p.__setitem__("mean_regret", np.zeros(344))))
    add("P8_SELECTION_EVALUATE", "mean", nmut("selection_evaluate/candidate_metrics_private.npz", "marginal__mean_regret", lambda a: a.__setitem__(0, a[0] + 1e-3)))
    add("P8_SELECTION_EVALUATE", "harm", nmut("selection_evaluate/candidate_metrics_private.npz", "joint__harm_rate", lambda a: a.__setitem__(0, a[0] + 1e-3)))
    add("P8_SELECTION_EVALUATE", "selected", jmut("selection_evaluate/selection_decision.json", lambda p: p["posteriors"]["marginal"].__setitem__("selected_index", (p["posteriors"]["marginal"]["selected_index"] + 1) % 344)))
    add("P8_SELECTION_EVALUATE", "decision_extra", jmut("selection_evaluate/selection_decision.json", lambda p: p["posteriors"]["joint"].__setitem__("threshold", 0.0)))
    add("P8_SELECTION_EVALUATE", "decision_scalar", jmut("selection_evaluate/selection_decision.json", lambda p: p["posteriors"]["joint"].__setitem__("mean_regret", p["posteriors"]["joint"]["mean_regret"] + 1e-6)))
    add("P8_SELECTION_EVALUATE", "decision_digest", jmut("selection_evaluate/selection_decision.json", lambda p: p["posteriors"]["marginal"].__setitem__("selected_actions_sha256", "0" * 64)))
    add("P8_SELECTION_EVALUATE", "aligned", nmut("selection_evaluate/selection_target_aligned_private.npz", "joint__regret", lambda a: a.__setitem__(0, a[0] + 1e-6)))
    add("P8_SELECTION_EVALUATE", "selector_model", jmut("selection_evaluate/selection_decision.json", lambda p: p.__setitem__("model_state", {})))
    add("P9_SELECTION_FREEZE", "selected_action", nmut("selection_freeze/selected_actions.npz", "marginal__actions", lambda a: a.__setitem__((0, 0), (a[0, 0] + 1) % 4)))
    add("P9_SELECTION_FREEZE", "selected_override", nmut("selection_freeze/selected_actions.npz", "joint__override", lambda a: a.__setitem__((0, 0), ~a[0, 0])))
    add("P9_SELECTION_FREEZE", "match_valid", nmut("selection_freeze/selection_matches.npz", "marginal__control_53611__match_valid", lambda a: a.__setitem__(0, ~a[0])))
    add("P9_SELECTION_FREEZE", "control_threshold", jmut("selection_freeze/selection_policy.json", lambda p: p["posteriors"]["joint"]["controls"][0]["thresholds"].__setitem__("proposer_threshold", p["posteriors"]["joint"]["controls"][0]["thresholds"]["proposer_threshold"] + 1e-3)))
    add("P10_EVALUATION_APPLY", "metadata_token", nmut("evaluation_apply/evaluation_metadata.npz", "pair_token", lambda a: a.__setitem__(0, "mutated")))
    add("P10_EVALUATION_APPLY", "metadata_card", nmut("evaluation_apply/evaluation_metadata.npz", "cardinality", lambda a: a.__setitem__(0, a[0] + 1)))
    add("P10_EVALUATION_APPLY", "metadata_stratum", nmut("evaluation_apply/evaluation_metadata.npz", "design_stratum", lambda a: a.__setitem__(0, "UNKNOWN")))
    add("P10_EVALUATION_APPLY", "metadata_logits", npz_mut("evaluation_apply/evaluation_metadata.npz", lambda p: p.__setitem__("logits", np.zeros((len(p["pair_token"]), 4)))))
    add("P10_EVALUATION_APPLY", "utility", npy_mut("prepared/public/utilities.npy", lambda a: a.__setitem__((0, 0), a[0, 0] + 1e-3), input_side=True))
    add("P10_EVALUATION_APPLY", "mass", nmut("evaluation_apply/evaluation_masses.npz", "joint__real", lambda a: a.__setitem__((0, 0), a[0, 0] + 1e-3)))
    add("P10_EVALUATION_APPLY", "action", nmut("evaluation_apply/evaluation_actions.npz", "marginal__contextual_actions", lambda a: a.__setitem__((0, 0), (a[0, 0] + 1) % 4)))
    add("P10_EVALUATION_APPLY", "sensitivity", nmut("evaluation_apply/evaluation_sensitivities.npz", "checkpoint_17__joint__hard_actions", lambda a: a.__setitem__((0, 0), (a[0, 0] + 1) % 4)))
    add("P10_EVALUATION_APPLY", "control_action", nmut("evaluation_apply/evaluation_actions.npz", "joint__control_53611__actions", lambda a: a.__setitem__((0, 0), (a[0, 0] + 1) % 4)))
    add("P10_EVALUATION_APPLY", "common", nmut("evaluation_apply/evaluation_actions.npz", "marginal__u_common", lambda a: a.__setitem__(0, ~a[0])))
    add("P10_EVALUATION_APPLY", "status", jmut("evaluation_apply/evaluation_apply_status.json", lambda p: p["joint"].__setitem__("u_common_count", p["joint"]["u_common_count"] + 1)))
    add("P10_EVALUATION_APPLY", "evaluator_model", npz_mut("evaluation_apply/evaluation_actions.npz", lambda p: p.__setitem__("model_state", np.zeros(1))))
    add("P11_EVALUATION_TRUTH", "raw", nmut("evaluation_truth/diagnostic_arrays.npz", "marginal__hard__regret", lambda a: a.__setitem__(0, a[0] + 1e-3)))
    add("P11_EVALUATION_TRUTH", "bootstrap", nmut("evaluation_truth/bootstrap_indices.npz", "global_pair_token_index", lambda a: a.__setitem__((0, 0), (a[0, 0] + 1) % a.shape[1])))
    add("P11_EVALUATION_TRUTH", "decision", jmut("evaluation_truth/estimand_table.json", lambda p: p.__setitem__("scientific_decision", "GO")))
    add("P11_EVALUATION_TRUTH", "promotion", jmut("evaluation_truth/estimand_table.json", lambda p: p.__setitem__("architecture_promoted", True)))
    add("P11_EVALUATION_TRUTH", "common_bootstrap", nmut("evaluation_truth/bootstrap_indices.npz", "joint__common_index", lambda a: a.__setitem__((0, 0), (a[0, 0] + 1) % a.shape[1])))
    add("P11_EVALUATION_TRUTH", "estimand_mean", jmut("evaluation_truth/estimand_table.json", lambda p: p["rows"][0].__setitem__("mean_diff", p["rows"][0]["mean_diff"] + 1e-6)))
    add("P11_EVALUATION_TRUTH", "estimand_ci", jmut("evaluation_truth/estimand_table.json", lambda p: p["rows"][0].__setitem__("ci95_high", p["rows"][0]["ci95_high"] + 1e-6)))
    add("P11_EVALUATION_TRUTH", "estimand_status", jmut("evaluation_truth/estimand_table.json", lambda p: p["rows"][0].__setitem__("status", "ADVERSE")))
    add("P11_EVALUATION_TRUTH", "not_evaluable_reinterpreted", jmut("evaluation_truth/estimand_table.json", lambda p: next(row for row in p["rows"] if row["status"] == "NOT_EVALUABLE").__setitem__("status", "ADVERSE")))
    add("P11_EVALUATION_TRUTH", "penalty", config_mut(lambda p: p["reader"].__setitem__("penalty", p["reader"]["penalty"] + 0.1)))
    add("P11_EVALUATION_TRUTH", "estimand_orientation", jmut("evaluation_truth/estimand_table.json", lambda p: p["rows"][0].__setitem__("orientation", "right_minus_left")))
    add("P11_EVALUATION_TRUTH", "pattern", jmut("evaluation_truth/estimand_table.json", lambda p: p["patterns"].__setitem__("JOINT_PATTERN_PRESENT", not p["patterns"]["JOINT_PATTERN_PRESENT"])))
    add("P11_EVALUATION_TRUTH", "summary", jmut("evaluation_truth/diagnostic_metrics.json", lambda p: p["posteriors"]["marginal"]["hard"].__setitem__("regret", p["posteriors"]["marginal"]["hard"]["regret"] + 1e-6)))
    add("P11_EVALUATION_TRUTH", "duplication", jmut("evaluation_truth/cell_duplications.json", lambda p: p["comparisons"][0].__setitem__("actions_exact", not p["comparisons"][0]["actions_exact"])))
    add("P12_RESTART_REPLAY", "receipt_mode", jmut("replay_receipt.json", lambda p: p.__setitem__("mode", "primary")))
    add("P12_RESTART_REPLAY", "receipt_exact", jmut("replay_receipt.json", lambda p: p.__setitem__("byte_exact", False)))
    add("P12_RESTART_REPLAY", "receipt_semantic", jmut("replay_receipt.json", lambda p: p.__setitem__("semantic_exclusions_valid", False)))
    add("P12_RESTART_REPLAY", "excluded_path", jmut("replay_receipt.json", lambda p: p["excluded_paths"].append("journals")))
    add("P12_RESTART_REPLAY", "journal_semantic", jmut("journals/posterior_fit.json", lambda p: p.__setitem__("maximum_truth_materialized", "EVALUATE")))
    add("P12_RESTART_REPLAY", "worker_semantic", jmut("posterior_fit/worker_receipt.json", lambda p: p["stage_files"].__setitem__(next(iter(p["stage_files"])), "0" * 64)))
    add("P12_RESTART_REPLAY", "runtime_class", jmut("runtime.json", lambda p: p.__setitem__("execution_class", "FRESH_PROSPECTIVE")))
    add("P12_RESTART_REPLAY", "runtime_phase", jmut("runtime.json", lambda p: p["phases_executed"][0].__setitem__("phase", "evaluation_truth")))
    add("P12_RESTART_REPLAY", "runtime_extra", jmut("runtime.json", lambda p: p.__setitem__("undeclared", True)))
    def recovery_origin(artifact: Path, input_package: Path, config: Path, freeze: Path) -> None: write_json(artifact / "recovery_origin.json", {"schema_version": "proportional-physical-recovery-origin-v1", "archived_inventory_sha256": "0" * 64, "reason": "WRONG"})
    add("P12_RESTART_REPLAY", "recovery_origin", recovery_origin)
    add("P13_INVENTORY", "manifest_hash", jmut("artifact_manifest.json", lambda p: next(row for row in p["files_and_directories"] if row["type"] == "file").__setitem__("sha256", "0" * 64)))
    add("P13_INVENTORY", "manifest_mode", jmut("artifact_manifest.json", lambda p: next(row for row in p["files_and_directories"] if row["type"] == "file").__setitem__("mode", 511)))
    add("P13_INVENTORY", "manifest_omit", jmut("artifact_manifest.json", lambda p: p["files_and_directories"].pop()))
    add("P13_INVENTORY", "manifest_schema", jmut("artifact_manifest.json", lambda p: p.__setitem__("schema_version", "mutated")))
    add("P13_INVENTORY", "manifest_class", jmut("artifact_manifest.json", lambda p: p["files_and_directories"][0].__setitem__("class", "public_handoff")))
    add("P13_INVENTORY", "manifest_phase", jmut("artifact_manifest.json", lambda p: p["files_and_directories"][0].__setitem__("phase", "future")))
    add("P13_INVENTORY", "manifest_type", jmut("artifact_manifest.json", lambda p: next(row for row in p["files_and_directories"] if row["type"] == "file").__setitem__("type", "directory")))
    def artifact_owner(artifact: Path, input_package: Path, config: Path, freeze: Path) -> None: os.chown(artifact / "REPORT.md", 65534, 65534)
    add("P13_INVENTORY", "owner", artifact_owner)
    def pretty(artifact: Path, input_package: Path, config: Path, freeze: Path) -> None:
        path = artifact / "evaluation_truth/diagnostic_metrics.json"; path.write_text(json.dumps(read_json(path), indent=2, sort_keys=True) + "\n")
    add("P13_INVENTORY", "pretty_json", pretty)
    add("P14_SCOPE", "gpu", jmut("runtime.json", lambda p: p.__setitem__("gpu_used_or_queried", True)))
    add("P14_SCOPE", "prospective", jmut("runtime.json", lambda p: p.__setitem__("prospective_evidence", True)))
    add("P14_SCOPE", "promotion", jmut("runtime.json", lambda p: p.__setitem__("architecture_promoted", True)))
    def report(artifact: Path, input_package: Path, config: Path, freeze: Path) -> None: (artifact / "REPORT.md").write_text("mutated\n")
    add("P14_SCOPE", "report", report)
    add("P15_COST", "coordinator_rss", jmut("runtime.json", lambda p: p.__setitem__("coordinator_peak_rss_bytes", 2_000_000_000)))
    add("P15_COST", "worker_rss", jmut("runtime.json", lambda p: p["phases_executed"][0].__setitem__("peak_rss_bytes", 2_000_000_000)))
    add("P15_COST", "disk_budget", config_mut(lambda p: p["budgets"].__setitem__("primary_plus_replay_bytes", 1)))
    add("P15_COST", "single_file_budget", config_mut(lambda p: p["budgets"].__setitem__("single_file_bytes", 1)))
    return rows


def main() -> int:
    args = parse_args(); artifact = args.artifact.resolve(strict=True); input_package = args.input_package.resolve(strict=True); reference = args.reference.resolve(strict=True); work = args.work_root.resolve()
    if work.exists(): shutil.rmtree(work)
    work.mkdir(parents=True)
    canonical_config = ROOT / "experiments/geometria_proporcional/configs/proportional_set_valued_physical_preflight_v1.json"
    canonical_freeze = ROOT / "experiments/geometria_proporcional/configs/proportional_set_valued_physical_source_freeze_v1.json"
    catalogue_contract = read_json(canonical_config)["mutation_catalogue"]
    reverse_coverage = {case_id: sorted(requirement for requirement, ids in catalogue_contract["coverage"].items() if case_id in ids) for case_id, _, _ in cases()}
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}
    results = []; started = time.monotonic(); peak_temporary_bytes = 0
    baseline_command = [str(ROOT / "venv/bin/python"), str(CHECKER), "--artifact", str(artifact), "--input-package", str(input_package), "--reference", str(reference)]
    baseline_result = subprocess.run(baseline_command, cwd=ROOT, env=environment, text=True, capture_output=True)
    baseline_payload = json.loads(baseline_result.stdout.strip().splitlines()[-1])
    baseline_checks = baseline_payload.get("checks")
    if baseline_result.returncode != 0 or baseline_payload.get("status") != "PASS" or baseline_payload.get("passed") != len(REASONS) or baseline_payload.get("total") != len(REASONS) or not isinstance(baseline_checks, list) or [row.get("id") for row in baseline_checks if isinstance(row, dict)] != list(REASONS) or any(set(row) != {"id", "status", "reason_code"} or row["status"] != "PASS" or row["reason_code"] is not None for row in baseline_checks):
        raise RuntimeError(f"unmutated baseline failed: {baseline_payload}")
    baseline = {"argv": baseline_command, "exit": baseline_result.returncode, "status": baseline_payload["status"], "passed": baseline_payload["passed"], "total": baseline_payload["total"], "checks": baseline_checks}
    for case_id, predicate, mutate in cases():
        case = work / case_id; art = case / "artifact"; inp = case / "input"; case.mkdir()
        shutil.copytree(artifact, art); shutil.copytree(input_package, inp)
        config = case / "config.json"; freeze = case / "source_freeze.json"; shutil.copyfile(canonical_config, config); shutil.copyfile(canonical_freeze, freeze)
        if predicate == "P3_PHYSICAL_BOUNDARY": rebind_probe_hashes(art, inp)
        before = object_snapshot(art, inp, config, freeze)
        mutate(art, inp, config, freeze)
        after = object_snapshot(art, inp, config, freeze)
        changed_objects = sorted(key for key in set(before) | set(after) if before.get(key) != after.get(key))
        if len(changed_objects) != 1: raise RuntimeError(f"mutation must change exactly one object: {case_id}: {changed_objects}")
        peak_temporary_bytes = max(peak_temporary_bytes, tree_bytes(case))
        use_mutated_config = changed_objects[0] == "config"
        command = [str(ROOT / "venv/bin/python"), str(CHECKER), "--artifact", str(art), "--input-package", str(inp), "--reference", str(reference), "--config", str(config if use_mutated_config else canonical_config), "--source-freeze", str(freeze if predicate == "P1_AUTHORITY" and "freeze" in case_id else canonical_freeze), "--only", predicate]
        result = subprocess.run(command, cwd=ROOT, env=environment, text=True, capture_output=True)
        payload = json.loads(result.stdout.strip().splitlines()[-1])
        observed = payload.get("reason_code"); expected = REASONS[predicate]
        passed = result.returncode != 0 and observed == expected
        results.append({"case_id": case_id, "single_mutation": True, "mutation_object": changed_objects[0], "predicate_expected": predicate, "reason_code_expected": expected, "reason_code_observed": observed, "exit": result.returncode, "passed": passed, "requirements": reverse_coverage[case_id]})
        if not passed: raise RuntimeError(f"mutation failed: {case_id}: {payload}")
        shutil.rmtree(case)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    catalogue = sorted(row[0] for row in cases()); catalogue_sha256 = hashlib.sha256(("\n".join(catalogue) + "\n").encode()).hexdigest()
    if catalogue_sha256 != catalogue_contract["case_ids_sha256"] or len(catalogue) != catalogue_contract["case_count"] or any(not ids for ids in catalogue_contract["coverage"].values()): raise RuntimeError("mutation catalogue contract drifted")
    payload = {"schema_version": "proportional-physical-mutation-receipt-v3", "catalogue_version": catalogue_contract["version"], "catalogue_sha256": catalogue_sha256, "status": "PASS", "argv": sys.argv, "source_freeze_sha256": sha256_file(canonical_freeze), "inputs": {"artifact_manifest_sha256": sha256_file(artifact / "artifact_manifest.json"), "input_preparation_sha256": sha256_file(input_package / "preparation_freeze.json"), "reference_manifest_sha256": sha256_file(reference / "artifact_manifest.json")}, "outputs": {"case_rows": len(results)}, "versions": {"python": sys.version.split()[0], "numpy": np.__version__, "scipy": __import__("scipy").__version__, "sklearn": __import__("sklearn").__version__}, "exit": 0, "baseline": baseline, "cases": results, "passed": len(results), "total": len(results), "wall_seconds": time.monotonic() - started, "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024, "children_peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss) * 1024, "peak_temporary_bytes": peak_temporary_bytes, "preserved_bytes_before_receipt": tree_bytes(work), "gpu_used_or_queried": False}
    args.receipt.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "passed": len(results), "total": len(results)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
