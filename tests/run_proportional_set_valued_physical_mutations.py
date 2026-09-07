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
    parser.add_argument("--work-root", type=Path, default=ROOT / "data/geometria_proporcional/proportional_set_valued_physical_mutations_v1")
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


def writable(root: Path) -> None:
    for path in root.rglob("*"):
        path.chmod(0o700 if path.is_dir() else 0o600)
    root.chmod(0o700)


def jmut(relative: str, change: Callable[[Any], None], *, input_side: bool = False) -> Callable[[Path, Path, Path, Path], None]:
    def apply(artifact: Path, input_package: Path, config: Path, freeze: Path) -> None:
        path = (input_package if input_side else artifact) / relative; payload = read_json(path); change(payload); write_json(path, payload)
    return apply


def nmut(relative: str, key: str, change: Callable[[np.ndarray], None], *, input_side: bool = False) -> Callable[[Path, Path, Path, Path], None]:
    def apply(artifact: Path, input_package: Path, config: Path, freeze: Path) -> None:
        path = (input_package if input_side else artifact) / relative; arrays = load_npz(path); change(arrays[key]); write_npz(path, arrays)
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
    add("P2_PREPARATION", "decision_cardinality", nmut("prepared/public/decision_select_public.npz", "cardinality", lambda a: a.__setitem__(0, a[0] + 1), input_side=True))
    add("P2_PREPARATION", "decision_token", nmut("prepared/truth/decision_select_truth.npz", "pair_token", lambda a: a.__setitem__(0, "mutated"), input_side=True))
    add("P2_PREPARATION", "evaluate_cardinality", nmut("prepared/public/evaluate_public.npz", "cardinality", lambda a: a.__setitem__(0, a[0] + 1), input_side=True))
    add("P2_PREPARATION", "evaluate_token", nmut("prepared/truth/evaluate_truth.npz", "pair_token", lambda a: a.__setitem__(0, "mutated"), input_side=True))
    add("P2_PREPARATION", "ensemble", nmut("prepared/public/evaluate_public.npz", "ensemble_logits", lambda a: a.__setitem__((0, 0), a[0, 0] + 1e-3), input_side=True))
    add("P2_PREPARATION", "stratum", nmut("prepared/public/decision_select_public.npz", "design_stratum", lambda a: a.__setitem__(0, "UNKNOWN"), input_side=True))
    for index, field in enumerate(("Uid", "Gid", "Groups", "NoNewPrivs", "CapEff")):
        value = "0\t0\t0\t0" if field in {"Uid", "Gid"} else ("1" if field in {"Groups", "NoNewPrivs"} else "0000000000000001")
        add("P3_PHYSICAL_BOUNDARY", f"identity_{index}", jmut("posterior_fit/worker_receipt.json", lambda p, f=field, v=value: p["runtime"]["identity"].__setitem__(f, v)))
    add("P3_PHYSICAL_BOUNDARY", "cuda_env", jmut("policy_fit/worker_receipt.json", lambda p: p["runtime"]["environment"].__setitem__("CUDA_VISIBLE_DEVICES", "0")))
    add("P3_PHYSICAL_BOUNDARY", "probe", jmut("selection_propose/worker_receipt.json", lambda p: p["probes"][0].__setitem__("outcome", "OPENED")))
    add("P4_STATE_MACHINE", "previous", jmut("journals/policy_fit.json", lambda p: p.__setitem__("previous_state", "PREPARED")))
    add("P4_STATE_MACHINE", "new", jmut("journals/selection_propose.json", lambda p: p.__setitem__("new_state", "COMPLETE")))
    add("P4_STATE_MACHINE", "truth_level", jmut("journals/evaluation_apply.json", lambda p: p.__setitem__("maximum_truth_materialized", "EVALUATE")))
    add("P4_STATE_MACHINE", "package", jmut("journals/posterior_fit.json", lambda p: p.__setitem__("package_id", "0" * 64)))
    add("P5_POSTERIOR", "donor", nmut("posterior_fit/target_shuffle_arrays.npz", "donor_index", lambda a: a.__setitem__(0, a[0] + 1)))
    add("P5_POSTERIOR", "permutable", nmut("posterior_fit/target_shuffle_arrays.npz", "permutable", lambda a: a.__setitem__(0, ~a[0])))
    add("P5_POSTERIOR", "target", nmut("posterior_fit/target_shuffle_arrays.npz", "target_shuffled", lambda a: a.__setitem__((0, 0), ~a[0, 0])))
    add("P5_POSTERIOR", "map", jmut("posterior_fit/target_shuffle_map.json", lambda p: p[0].__setitem__("donor_pair_token", "mutated")))
    add("P6_POLICY", "feature_count", jmut("policy_fit/feature_schema.json", lambda p: p.__setitem__("count", 16)))
    add("P6_POLICY", "feature_name", jmut("policy_fit/feature_schema.json", lambda p: p["feature_names"].__setitem__(0, "mutated")))
    add("P6_POLICY", "control_remove", jmut("policy_fit/policy_states.json", lambda p: p["marginal"]["controls"].pop()))
    add("P6_POLICY", "control_seed", jmut("policy_fit/policy_states.json", lambda p: p["marginal"]["controls"][0].__setitem__("seed", 1)))
    add("P7_SELECTION_PROPOSE", "threshold", jmut("selection_propose/apply_metadata.json", lambda p: p["posteriors"]["marginal"][0].__setitem__("proposer_threshold", p["posteriors"]["marginal"][0]["proposer_threshold"] + 1e-3)))
    add("P7_SELECTION_PROPOSE", "action", nmut("selection_propose/candidate_public.npz", "marginal__actions", lambda a: a.__setitem__((0, 0, 0), (a[0, 0, 0] + 1) % 4)))
    add("P7_SELECTION_PROPOSE", "override", nmut("selection_propose/candidate_public.npz", "joint__override", lambda a: a.__setitem__((0, 0, 0), ~a[0, 0, 0])))
    add("P7_SELECTION_PROPOSE", "key_extra", jmut("selection_propose/selection_key_metadata.json", lambda p: p["posteriors"]["joint"][0].__setitem__("threshold", 0.0)))
    add("P8_SELECTION_EVALUATE", "mean", nmut("selection_evaluate/candidate_metrics_private.npz", "marginal__mean_regret", lambda a: a.__setitem__(0, a[0] + 1e-3)))
    add("P8_SELECTION_EVALUATE", "harm", nmut("selection_evaluate/candidate_metrics_private.npz", "joint__harm_rate", lambda a: a.__setitem__(0, a[0] + 1e-3)))
    add("P8_SELECTION_EVALUATE", "selected", jmut("selection_evaluate/selection_decision.json", lambda p: p["posteriors"]["marginal"].__setitem__("selected_index", (p["posteriors"]["marginal"]["selected_index"] + 1) % 344)))
    add("P8_SELECTION_EVALUATE", "decision_extra", jmut("selection_evaluate/selection_decision.json", lambda p: p["posteriors"]["joint"].__setitem__("threshold", 0.0)))
    add("P9_SELECTION_FREEZE", "selected_action", nmut("selection_freeze/selected_actions.npz", "marginal__actions", lambda a: a.__setitem__((0, 0), (a[0, 0] + 1) % 4)))
    add("P9_SELECTION_FREEZE", "selected_override", nmut("selection_freeze/selected_actions.npz", "joint__override", lambda a: a.__setitem__((0, 0), ~a[0, 0])))
    add("P9_SELECTION_FREEZE", "match_valid", nmut("selection_freeze/selection_matches.npz", "marginal__control_53611__match_valid", lambda a: a.__setitem__(0, ~a[0])))
    add("P9_SELECTION_FREEZE", "control_threshold", jmut("selection_freeze/selection_policy.json", lambda p: p["posteriors"]["joint"]["controls"][0]["thresholds"].__setitem__("proposer_threshold", p["posteriors"]["joint"]["controls"][0]["thresholds"]["proposer_threshold"] + 1e-3)))
    add("P10_EVALUATION_APPLY", "metadata_token", nmut("evaluation_apply/evaluation_metadata.npz", "pair_token", lambda a: a.__setitem__(0, "mutated")))
    add("P10_EVALUATION_APPLY", "metadata_card", nmut("evaluation_apply/evaluation_metadata.npz", "cardinality", lambda a: a.__setitem__(0, a[0] + 1)))
    add("P10_EVALUATION_APPLY", "mass", nmut("evaluation_apply/evaluation_masses.npz", "joint__real", lambda a: a.__setitem__((0, 0), a[0, 0] + 1e-3)))
    add("P10_EVALUATION_APPLY", "action", nmut("evaluation_apply/evaluation_actions.npz", "marginal__contextual_actions", lambda a: a.__setitem__((0, 0), (a[0, 0] + 1) % 4)))
    add("P10_EVALUATION_APPLY", "sensitivity", nmut("evaluation_apply/evaluation_sensitivities.npz", "checkpoint_17__joint__hard_actions", lambda a: a.__setitem__((0, 0), (a[0, 0] + 1) % 4)))
    add("P11_EVALUATION_TRUTH", "raw", nmut("evaluation_truth/diagnostic_arrays.npz", "marginal__hard__regret", lambda a: a.__setitem__(0, a[0] + 1e-3)))
    add("P11_EVALUATION_TRUTH", "bootstrap", nmut("evaluation_truth/bootstrap_indices.npz", "global_pair_token_index", lambda a: a.__setitem__((0, 0), (a[0, 0] + 1) % a.shape[1])))
    add("P11_EVALUATION_TRUTH", "decision", jmut("evaluation_truth/estimand_table.json", lambda p: p.__setitem__("scientific_decision", "GO")))
    add("P11_EVALUATION_TRUTH", "promotion", jmut("evaluation_truth/estimand_table.json", lambda p: p.__setitem__("architecture_promoted", True)))
    add("P12_RESTART_REPLAY", "receipt_mode", jmut("replay_receipt.json", lambda p: p.__setitem__("mode", "primary")))
    add("P12_RESTART_REPLAY", "receipt_exact", jmut("replay_receipt.json", lambda p: p.__setitem__("byte_exact", False)))
    add("P12_RESTART_REPLAY", "receipt_semantic", jmut("replay_receipt.json", lambda p: p.__setitem__("semantic_exclusions_valid", False)))
    add("P13_INVENTORY", "manifest_hash", jmut("artifact_manifest.json", lambda p: next(row for row in p["files_and_directories"] if row["type"] == "file").__setitem__("sha256", "0" * 64)))
    add("P13_INVENTORY", "manifest_mode", jmut("artifact_manifest.json", lambda p: next(row for row in p["files_and_directories"] if row["type"] == "file").__setitem__("mode", 511)))
    add("P13_INVENTORY", "manifest_omit", jmut("artifact_manifest.json", lambda p: p["files_and_directories"].pop()))
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
    return rows


def main() -> int:
    args = parse_args(); artifact = args.artifact.resolve(strict=True); input_package = args.input_package.resolve(strict=True); reference = args.reference.resolve(strict=True); work = args.work_root.resolve()
    if work.exists(): shutil.rmtree(work)
    work.mkdir(parents=True)
    canonical_config = ROOT / "experiments/geometria_proporcional/configs/proportional_set_valued_physical_preflight_v1.json"
    canonical_freeze = ROOT / "experiments/geometria_proporcional/configs/proportional_set_valued_physical_source_freeze_v1.json"
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}
    results = []; started = time.monotonic(); peak_temporary_bytes = 0
    for case_id, predicate, mutate in cases():
        case = work / case_id; art = case / "artifact"; inp = case / "input"; case.mkdir()
        shutil.copytree(artifact, art); shutil.copytree(input_package, inp); writable(art); writable(inp)
        config = case / "config.json"; freeze = case / "source_freeze.json"; shutil.copyfile(canonical_config, config); shutil.copyfile(canonical_freeze, freeze)
        mutate(art, inp, config, freeze)
        peak_temporary_bytes = max(peak_temporary_bytes, tree_bytes(case))
        command = [str(ROOT / "venv/bin/python"), str(CHECKER), "--artifact", str(art), "--input-package", str(inp), "--reference", str(reference), "--config", str(config if predicate == "P1_AUTHORITY" and "config" in case_id else canonical_config), "--source-freeze", str(freeze if predicate == "P1_AUTHORITY" and "freeze" in case_id else canonical_freeze), "--only", predicate]
        result = subprocess.run(command, cwd=ROOT, env=environment, text=True, capture_output=True)
        payload = json.loads(result.stdout.strip().splitlines()[-1])
        observed = payload.get("reason_code"); expected = REASONS[predicate]
        passed = result.returncode != 0 and observed == expected
        results.append({"case_id": case_id, "single_mutation": True, "predicate_expected": predicate, "reason_code_expected": expected, "reason_code_observed": observed, "exit": result.returncode, "passed": passed})
        if not passed: raise RuntimeError(f"mutation failed: {case_id}: {payload}")
        shutil.rmtree(case)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": "proportional-physical-mutation-receipt-v1", "status": "PASS", "argv": sys.argv, "source_freeze_sha256": sha256_file(canonical_freeze), "inputs": {"artifact_manifest_sha256": sha256_file(artifact / "artifact_manifest.json"), "input_preparation_sha256": sha256_file(input_package / "preparation_freeze.json"), "reference_manifest_sha256": sha256_file(reference / "artifact_manifest.json")}, "outputs": {"case_rows": len(results)}, "versions": {"python": sys.version.split()[0], "numpy": np.__version__, "scipy": __import__("scipy").__version__, "sklearn": __import__("sklearn").__version__}, "exit": 0, "cases": results, "passed": len(results), "total": len(results), "wall_seconds": time.monotonic() - started, "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024, "children_peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss) * 1024, "peak_temporary_bytes": peak_temporary_bytes, "preserved_bytes_before_receipt": tree_bytes(work), "gpu_used_or_queried": False}
    args.receipt.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "passed": len(results), "total": len(results)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
