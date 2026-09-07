#!/usr/bin/env python3
"""One-change mutation campaign for the independent set-valued checker."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import resource
import zipfile
from typing import Any, Callable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKER = REPO_ROOT / "experiments/geometria_proporcional/check_proportional_set_valued_native_preflight.py"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n",
        encoding="utf-8",
    )


def mutate_json(relative: str, change: Callable[[Any], None]) -> Callable[[Path], None]:
    def apply(root: Path) -> None:
        path = root / relative
        value = read_json(path); change(value); write_json(path, value)

    return apply


def mutate_npz(relative: str, change: Callable[[dict[str, np.ndarray]], None]) -> Callable[[Path], None]:
    def apply(root: Path) -> None:
        path = root / relative
        with np.load(path, allow_pickle=False) as payload:
            arrays = {key: payload[key].copy() for key in payload.files}
        change(arrays)
        np.savez_compressed(path, **arrays)

    return apply


def flip_first(array: np.ndarray) -> None:
    flat = array.reshape(-1)
    if array.dtype == bool:
        flat[0] = ~flat[0]
    elif np.issubdtype(array.dtype, np.integer):
        flat[0] += 1
    else:
        flat[0] += 0.125


def create_phase_overlap(root: Path) -> None:
    with np.load(root / "prepared/posterior_fit_truth.npz", allow_pickle=False) as source:
        token = str(source["pair_token"][0])
    path = root / "prepared/policy_fit_truth.npz"
    with np.load(path, allow_pickle=False) as payload:
        arrays = {key: payload[key].copy() for key in payload.files}
    arrays["pair_token"][0] = token
    np.savez_compressed(path, **arrays)


def refresh_manifest_entry(root: Path, relative: str) -> None:
    import hashlib

    path = root / relative
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path = root / "artifact_manifest.json"
    manifest = read_json(manifest_path)
    row = next(item for item in manifest["files"] if item["path"] == relative)
    row.update(bytes=path.stat().st_size, sha256=digest)
    write_json(manifest_path, manifest)


def semantic_report_mutation(root: Path) -> None:
    path = root / "REPORT.md"
    path.write_text(path.read_text(encoding="utf-8") + "\nArchitecture promoted.\n", encoding="utf-8")
    refresh_manifest_entry(root, "REPORT.md")


def runtime_budget_mutation(root: Path) -> None:
    path = root / "runtime.json"
    value = read_json(path); value.update(peak_rss_bytes=2_000_000_000); write_json(path, value)
    refresh_manifest_entry(root, "runtime.json")


def mutate_and_refresh(relative: str, mutation: Callable[[Path], None]) -> Callable[[Path], None]:
    def apply(root: Path) -> None:
        mutation(root)
        refresh_manifest_entry(root, relative)

    return apply


def rewrite_npz_noncanonical(relative: str) -> Callable[[Path], None]:
    def rewrite(root: Path) -> None:
        path = root / relative
        temporary = path.with_suffix(".noncanonical")
        with zipfile.ZipFile(path) as source, zipfile.ZipFile(
            temporary, "w", compression=zipfile.ZIP_DEFLATED
        ) as target:
            for index, info in enumerate(source.infolist()):
                replacement = zipfile.ZipInfo(
                    info.filename,
                    date_time=(2020, 1, 1, 0, 0, 0) if index == 0 else info.date_time,
                )
                replacement.compress_type = zipfile.ZIP_DEFLATED
                replacement.external_attr = info.external_attr
                target.writestr(replacement, source.read(info.filename))
        temporary.replace(path)
        refresh_manifest_entry(root, relative)

    return rewrite


def cases() -> list[tuple[str, str, Callable[[Path], None], dict[str, str]]]:
    return [
        ("source_hash", "SOURCE_OR_SCOPE_INVALID", mutate_json("source_bindings.json", lambda x: x["sources"][0].update(sha256="0" * 64)), {}),
        ("cuda_visible", "SOURCE_OR_SCOPE_INVALID", lambda root: None, {"CUDA_VISIBLE_DEVICES": "0"}),
        ("forbidden_path_contract", "SOURCE_OR_SCOPE_INVALID", mutate_json("config.snapshot.json", lambda x: x["forbidden_input_path_fragments"].append("wave54")), {}),
        ("penalty_recipe", "SOURCE_OR_SCOPE_INVALID", mutate_json("config.snapshot.json", lambda x: x["reader"].update(penalty=2.0)), {}),
        ("public_target", "PHASE_BUNDLE_INVALID", mutate_npz("prepared/decision_select_public.npz", lambda x: x.update(target=np.ones((len(x["pair_token"]), 4), dtype=bool))), {}),
        ("posterior_utility_leak", "PHASE_BUNDLE_INVALID", mutate_npz("prepared/posterior_fit_truth.npz", lambda x: x.update(utility=np.zeros((len(x["pair_token"]), 4)))), {}),
        ("phase_overlap", "PHASE_BUNDLE_INVALID", create_phase_overlap, {}),
        ("public_duplicate", "PHASE_BUNDLE_INVALID", mutate_npz("prepared/decision_select_public.npz", lambda x: x["pair_token"].__setitem__(1, x["pair_token"][0])), {}),
        ("prepared_provenance", "PHASE_BUNDLE_INVALID", mutate_and_refresh("prepared/posterior_fit_truth.npz", mutate_npz("prepared/posterior_fit_truth.npz", lambda x: x["cluster_id"].__setitem__(0, "corrupt"))), {}),
        ("marginal_coefficient", "MARGINAL_RECIPE_INVALID", mutate_json("posterior_fit/states.json", lambda x: x["marginal"]["real"].update(coefficient=x["marginal"]["real"]["coefficient"] + 0.1)), {}),
        ("marginal_recipe", "MARGINAL_RECIPE_INVALID", mutate_json("posterior_fit/states.json", lambda x: x["marginal"]["real"]["contract"].update(C=2.0)), {}),
        ("marginal_class_weight", "MARGINAL_RECIPE_INVALID", mutate_json("posterior_fit/states.json", lambda x: x["marginal"]["real"]["contract"].update(class_weight="balanced")), {}),
        ("joint_grid_missing", "JOINT_RECIPE_INVALID", mutate_json("posterior_fit/states.json", lambda x: x["joint"]["real"]["regularization_grid"].pop()), {}),
        ("joint_lambda_mismatch", "JOINT_RECIPE_INVALID", mutate_json("posterior_fit/states.json", lambda x: x["joint"]["target_shuffled"].update(selected_index=5, selected_regularization=10.0)), {}),
        ("joint_fold_theta", "JOINT_RECIPE_INVALID", mutate_and_refresh("posterior_fit/oof_arrays.npz", mutate_npz("posterior_fit/oof_arrays.npz", lambda x: flip_first(x["joint_real__fold_theta"]))), {}),
        ("joint_fold_assignment", "JOINT_RECIPE_INVALID", mutate_npz("posterior_fit/oof_arrays.npz", lambda x: flip_first(x["joint_real__fold_id"])), {}),
        ("shuffle_donor", "TARGET_SHUFFLE_INVALID", mutate_npz("posterior_fit/target_shuffle_arrays.npz", lambda x: x["donor_index"].__setitem__(0, 0)), {}),
        ("shuffle_semantic_map", "TARGET_SHUFFLE_INVALID", mutate_json("posterior_fit/target_shuffle_map.json", lambda x: x[0].update(donor=x[0]["receiver"])), {}),
        ("shuffle_cross_stratum", "TARGET_SHUFFLE_INVALID", mutate_json("posterior_fit/target_shuffle_map.json", lambda x: x[0].update(design_stratum="CORRUPT")), {}),
        ("hard_binding", "HARD_POSTERIOR_BINDING_INVALID", mutate_npz("decision_select/scores.npz", lambda x: flip_first(x["marginal__hard_actions"])), {}),
        ("contextual_design", "CONTEXTUAL_DESIGN_INVALID", mutate_npz("decision_select/scores.npz", lambda x: flip_first(x["joint__design"])), {}),
        ("feature_order", "CONTEXTUAL_DESIGN_INVALID", mutate_json("policy_fit/feature_schema.json", lambda x: x["feature_names"].reverse()), {}),
        ("feature_weight", "CONTEXTUAL_DESIGN_INVALID", mutate_npz("decision_select/scores.npz", lambda x: flip_first(x["marginal__weights"])), {}),
        ("model_coefficient", "CONTEXTUAL_STATE_INVALID", mutate_json("policy_fit/states.json", lambda x: x["marginal"]["true"]["states"]["proposer"]["coef"].__setitem__(0, x["marginal"]["true"]["states"]["proposer"]["coef"][0] + 0.2)), {}),
        ("ridge_alpha", "CONTEXTUAL_STATE_INVALID", mutate_json("policy_fit/states.json", lambda x: x["joint"]["true"]["states"]["proposer"].update(alpha=2.0)), {}),
        ("ridge_pseudoinverse", "CONTEXTUAL_STATE_INVALID", mutate_json("policy_fit/states.json", lambda x: x["joint"]["true"]["states"]["proposer"].update(solver="numpy.linalg.pinv")), {}),
        ("ridge_intercept_penalized", "CONTEXTUAL_STATE_INVALID", mutate_json("policy_fit/states.json", lambda x: x["marginal"]["true"]["states"]["proposer"].update(intercept_penalized=True)), {}),
        ("guard_target_sign", "CONTEXTUAL_STATE_INVALID", mutate_json("policy_fit/states.json", lambda x: x["marginal"]["true"]["states"]["harm"].update(target_name="not_harm")), {}),
        ("guard_class", "CONTEXTUAL_STATE_INVALID", mutate_json("policy_fit/states.json", lambda x: x["joint"]["true"]["states"]["incompatibility"].update(classes=[1, 0])), {}),
        ("selection_index", "SELECTION_PROTOCOL_INVALID", mutate_json("decision_select/selection_freeze.json", lambda x: x["marginal"].update(selected_index=(x["marginal"]["selected_index"] + 1) % 344)), {}),
        ("candidate_override", "SELECTION_PROTOCOL_INVALID", mutate_npz("decision_select/candidate_metrics.npz", lambda x: flip_first(x["joint__override"])), {}),
        ("candidate_grid_missing", "SELECTION_PROTOCOL_INVALID", mutate_npz("decision_select/candidate_metrics.npz", lambda x: x.update(marginal__actions=x["marginal__actions"][:-1])), {}),
        ("hard_only_missing", "SELECTION_PROTOCOL_INVALID", mutate_json("decision_select/selection_freeze.json", lambda x: x["joint"]["candidate_metadata"].pop()), {}),
        ("control_mapping", "MATCHED_CONTROL_INVALID", mutate_npz("policy_fit/control_arrays.npz", lambda x: flip_first(x["marginal__control_53611__mapping"])), {}),
        ("control_seed", "MATCHED_CONTROL_INVALID", mutate_json("policy_fit/states.json", lambda x: x["joint"]["controls"][0].update(seed=999)), {}),
        ("control_declared_digest", "MATCHED_CONTROL_INVALID", mutate_and_refresh("policy_fit/states.json", mutate_json("policy_fit/states.json", lambda x: x["marginal"]["controls"][0]["diagnostics"].update(mapping_sha256="0" * 64))), {}),
        ("matching_k", "MATCHED_CONTROL_INVALID", mutate_npz("apply_fixture/actions_and_matches.npz", lambda x: flip_first(x["marginal__control_53611__requested_k"])), {}),
        ("matching_sort", "MATCHED_CONTROL_INVALID", mutate_npz("apply_fixture/actions_and_matches.npz", lambda x: flip_first(x["joint__control_53617__selected"])), {}),
        ("common_support_union", "MATCHED_CONTROL_INVALID", mutate_npz("apply_fixture/actions_and_matches.npz", lambda x: flip_first(x["joint__u_common"])), {}),
        ("estimand_mean", "CELL_ESTIMAND_MISMATCH", mutate_json("evaluate_fixture/estimand_table.json", lambda x: x["rows"][0].update(mean_diff=x["rows"][0]["mean_diff"] + 0.1)), {}),
        ("estimand_orientation", "CELL_ESTIMAND_MISMATCH", mutate_json("evaluate_fixture/estimand_table.json", lambda x: x["rows"][1].update(orientation="right_minus_left")), {}),
        ("estimand_missing", "CELL_ESTIMAND_MISMATCH", mutate_json("evaluate_fixture/estimand_table.json", lambda x: x["rows"].pop()), {}),
        ("bootstrap_index", "CELL_ESTIMAND_MISMATCH", mutate_npz("evaluate_fixture/bootstrap_indices.npz", lambda x: flip_first(x["global_pair_token_index"])), {}),
        ("raw_missing", "CELL_ESTIMAND_MISMATCH", mutate_npz("evaluate_fixture/diagnostic_arrays.npz", lambda x: x.pop("marginal__hard__regret")), {}),
        ("pattern_logic", "CELL_ESTIMAND_MISMATCH", mutate_json("evaluate_fixture/estimand_table.json", lambda x: x["patterns"].update(JOINT_PATTERN_PRESENT=not x["patterns"]["JOINT_PATTERN_PRESENT"])), {}),
        ("sensitivity_value", "CELL_ESTIMAND_MISMATCH", mutate_and_refresh("evaluate_fixture/sensitivity_arrays.npz", mutate_npz("evaluate_fixture/sensitivity_arrays.npz", lambda x: flip_first(x["checkpoint_17__marginal__hard__actions"]))), {}),
        ("duplication_value", "CELL_ESTIMAND_MISMATCH", mutate_and_refresh("evaluate_fixture/cell_duplications.json", mutate_json("evaluate_fixture/cell_duplications.json", lambda x: x["comparisons"][0].update(actions_exact=not x["comparisons"][0]["actions_exact"]))), {}),
        ("manifest_hash", "RAW_OR_REPLAY_INVALID", mutate_json("artifact_manifest.json", lambda x: x["files"][0].update(sha256="f" * 64)), {}),
        ("npz_noncanonical", "RAW_OR_REPLAY_INVALID", rewrite_npz_noncanonical("prepared/decision_select_truth.npz"), {}),
        ("promotion_language", "CLAIM_BOUNDARY_INVALID", semantic_report_mutation, {}),
        ("equivalent_promotion_language", "CLAIM_BOUNDARY_INVALID", mutate_and_refresh("REPORT.md", lambda root: (root / "REPORT.md").write_text((root / "REPORT.md").read_text(encoding="utf-8") + "\nThe JOINT architecture is recommended for promotion.\n", encoding="utf-8")), {}),
        ("applier_truth_receipt", "CLAIM_BOUNDARY_INVALID", mutate_and_refresh("apply_fixture/action_freeze.json", mutate_json("apply_fixture/action_freeze.json", lambda x: x.update(truth_keys_received_by_applier=["target"]))), {}),
        ("runtime_rss", "COST_CONTRACT_INVALID", runtime_budget_mutation, {}),
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--case", action="append", dest="selected_cases")
    args = parser.parse_args()
    artifact = args.artifact.resolve(strict=True)
    started = time.monotonic()
    results = []
    valid_checks = []
    base_environment = dict(os.environ)
    base_environment.update({
        "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1", "BLIS_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
    })
    commands = [[sys.executable, str(CHECKER), str(artifact)]]
    if args.replay is not None:
        commands.append([sys.executable, str(CHECKER), str(args.replay.resolve(strict=True)), "--reference", str(artifact)])
    for command in commands:
        completed = subprocess.run(command, cwd=REPO_ROOT, env=base_environment, text=True, capture_output=True)
        payload = json.loads(completed.stdout.strip().splitlines()[-1])
        valid_checks.append({"command_role": "primary" if len(valid_checks) == 0 else "replay", "status": payload["status"], "returncode": completed.returncode})
    selected = cases()
    if args.selected_cases:
        wanted = set(args.selected_cases)
        selected = [row for row in selected if row[0] in wanted]
        if {row[0] for row in selected} != wanted:
            raise SystemExit(f"unknown mutation case: {sorted(wanted - {row[0] for row in selected})}")
    for name, expected, mutation, environment_change in selected:
        with tempfile.TemporaryDirectory(prefix=f"set-valued-mutation-{name}-") as directory:
            mutated = Path(directory) / "artifact"
            shutil.copytree(artifact, mutated)
            mutation(mutated)
            environment = {**base_environment, **environment_change}
            completed = subprocess.run(
                [sys.executable, str(CHECKER), str(mutated)], cwd=REPO_ROOT,
                env=environment, text=True, capture_output=True,
            )
            payload = json.loads(completed.stdout.strip().splitlines()[-1])
            observed = payload.get("reason_code")
            passed = completed.returncode != 0 and observed == expected
            results.append({"case": name, "expected": expected, "observed": observed, "pass": passed})
            print(json.dumps(results[-1], sort_keys=True), flush=True)
    elapsed = time.monotonic() - started
    peak_rss = int(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss) * 1024
    summary = {
        "schema_version": "proportional-mutation-suite-v2", "valid_checks": valid_checks,
        "cases": results, "passed": sum(row["pass"] for row in results), "total": len(results),
        "wall_seconds": elapsed, "peak_child_rss_bytes": peak_rss,
        "wall_budget_seconds": 900.0, "rss_budget_bytes": 1610612736,
        "within_budget": elapsed <= 900.0 and peak_rss <= 1610612736,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.output, summary)
    print(json.dumps(summary, sort_keys=True))
    valid_ok = all(row["status"] == "PASS" and row["returncode"] == 0 for row in valid_checks)
    return 0 if summary["passed"] == summary["total"] and summary["within_budget"] and valid_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
