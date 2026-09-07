#!/usr/bin/env python3
"""Independent CPU checker for the two native proportional design freezes."""

from __future__ import annotations

import argparse
import copy
from dataclasses import fields, replace
import hashlib
import io
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any, Callable

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "experiments/geometria_proporcional/configs"
DEFAULT_COORDINATOR = CONFIG_DIR / "proportional_dual_native_freeze_v1.json"
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

COORDINATION_PREDICATES = [
    "C1_SOURCE_BINDING",
    "C2_BRANCH_SEPARATION",
    "C3_PHASE_DAG",
    "C4_REPLAY_CONTRACT",
    "C5_NO_GPU_NO_PROMOTION",
    "C6_READINESS_SEMANTICS",
]
RELATIONAL_PREDICATES = [
    "R1_PUBLIC_PRIVATE_SCHEMA",
    "R2_MASTER_SPLIT",
    "R3_ARM_PARITY",
    "R4_DUAL_SOLVER_LOSS",
    "R5_EXECUTOR_PARITY",
    "R6_SOLVER_DENOMINATORS",
    "R7_PATH_CONTROL_ROSTER",
    "R8_WEIGHT_CONTROLS",
    "R9_TOTAL_TARGET_SHUFFLE",
    "R10_SEED_ESTIMAND",
    "R11_BASE_WEIGHTED_K192_CONFORMANCE",
    "R12_DECISION_TABLE",
]
SET_PREDICATES = [
    "S1_PHASE_SUPPORT",
    "S2_LOGIT_TARGET_SEPARATION",
    "S3_POSTERIOR_PARITY",
    "S4_NATIVE_FIT_RECIPES",
    "S5_HARD_POSTERIOR_BINDING",
    "S6_CONTEXTUAL_RECIPE",
    "S7_PROPOSER_GUARD_CREDIT",
    "S8_TARGET_SHUFFLE_FOLDS",
    "S9_MATCHED_CONTROL_TARGET_BLIND",
    "S10_CELL_DUPLICATION",
    "S11_ESTIMAND_ORDER",
    "S12_DECISION_TABLE",
]
ALL_PREDICATES = COORDINATION_PREDICATES + RELATIONAL_PREDICATES + SET_PREDICATES

EXPECTED_PHASE_DAG = [
    "source_verifier",
    "public_private_preparer",
    "train_or_fit",
    "calibration_selection",
    "freeze_hash_read_only",
    "monitor_apply_without_target",
    "monitor_evaluate_with_target",
    "independent_checker",
    "replay_second_root",
]
EXPECTED_REL_ROWS = [
    "REL_RELATION",
    "REL_PATH",
    "REL_WLS",
    "REL_IRLS",
    "REL_FAILURE",
    "REL_TARGET",
    "REL_INTERACTION",
]
EXPECTED_SET_ROWS = [
    "SET_JOINT_NLL",
    "SET_JOINT_BRIER",
    "SET_SHUFFLE",
    "READER_REGRET",
    "READER_COMPAT",
    "READER_WORST",
    "READER_CONTROL",
    "FACTOR_INTERACTION",
]
EXPECTED_PHASE_ACCESS = {
    "posterior_fit": ["logits", "target"],
    "policy_fit": ["posterior", "utility", "target"],
    "decision_select": ["frozen_states", "utility", "target"],
    "monitor_apply": ["public_inputs", "frozen_states", "thresholds"],
    "monitor_evaluate": ["frozen_actions", "target", "utility"],
}
CRITICAL_IMPLEMENTATION_PATHS = [
    "src/geometria_proporcional/proportional_graph_neural.py",
    "experiments/geometria_proporcional/check_proportional_dual_native_freeze.py",
    "experiments/geometria_proporcional/run_proportional_dual_native_preflight.py",
    "experiments/geometria_proporcional/run_proportional_base_weighted_depth_scan.py",
    "tests/test_proportional_dual_native_freeze.py",
    "tests/test_proportional_graph_irls_surrogate_fidelity.py",
]
REQUIRED_BINDING_PATHS = {
    "experiments/geometria_proporcional/PLAN_PROPORTIONAL_DUAL_NATIVE_FREEZES_CPU.md",
    "Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/547_proportional_dual_native_freeze_plan_audit.md",
    "Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/549_proportional_dual_native_freeze_plan_final_reaudit.md",
    "Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/551_proportional_k192_amendment_audit.md",
}
EXPECTED_BRANCH_PATHS = {
    "relational": "experiments/geometria_proporcional/configs/proportional_relational_native_freeze_v1.json",
    "set_valued": "experiments/geometria_proporcional/configs/proportional_set_valued_native_freeze_v1.json",
}
ARTIFACT_FILES = [
    "fixtures.json",
    "fixed_depth_raw.npz",
    "mutation_results.json",
    "scientific_report.json",
]
FIXED_CLAIMS = {
    "gpu_used_or_queried": False,
    "architecture_promoted": False,
    "scientific_decision": None,
    "decision_authority": "user",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def config_triplet(
    coordinator_path: Path = DEFAULT_COORDINATOR,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    coordinator = load_json(coordinator_path)
    relational = load_json(ROOT / coordinator["branches"]["relational"])
    set_valued = load_json(ROOT / coordinator["branches"]["set_valued"])
    return coordinator, relational, set_valued


def result(ok: bool, reason: str = "") -> dict[str, Any]:
    return {"status": "PASS" if ok else "FAIL", "reasons": [] if ok else [reason]}


def git_blob_bytes(commit: str, relative: str) -> bytes | None:
    completed = subprocess.run(
        ["git", "show", f"{commit}:{relative}"],
        cwd=ROOT,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    return completed.stdout if completed.returncode == 0 else None


def source_bindings_valid(
    coordinator: dict[str, Any], configs: list[dict[str, Any]]
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    seen: dict[str, str] = {}
    for config in configs:
        commit = str(config.get("source_commit", ""))
        if len(commit) != 40:
            reasons.append("SOURCE_COMMIT_INVALID")
        for row in config.get("source_bindings", []):
            if not isinstance(row, list) or len(row) != 3:
                reasons.append("SOURCE_BINDING_SCHEMA_INVALID")
                continue
            _, relative, expected = row
            path = ROOT / relative
            if not path.is_file():
                reasons.append(f"SOURCE_MISSING:{relative}")
                continue
            actual = sha256_file(path)
            if actual != expected:
                reasons.append(f"SOURCE_HASH_MISMATCH:{relative}")
            if relative in seen and seen[relative] != expected:
                reasons.append(f"SOURCE_HASH_CONFLICT:{relative}")
            seen[relative] = expected
    bound_paths = set(seen)
    for required in sorted(REQUIRED_BINDING_PATHS - bound_paths):
        reasons.append(f"SOURCE_BINDING_REQUIRED:{required}")
    commits = {
        str(config.get("source_commit")) for config in [coordinator, *configs]
    }
    if len(commits) != 1:
        reasons.append("SOURCE_COMMIT_DIVERGENCE")
    elif commits:
        commit = next(iter(commits))
        completed = subprocess.run(
            ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
            cwd=ROOT,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if completed.returncode != 0:
            reasons.append("SOURCE_COMMIT_UNRESOLVED")
        else:
            for relative in CRITICAL_IMPLEMENTATION_PATHS:
                historical = git_blob_bytes(commit, relative)
                current_path = ROOT / relative
                if historical is None:
                    reasons.append(f"SOURCE_COMMIT_PATH_MISSING:{relative}")
                elif not current_path.is_file() or sha256_bytes(historical) != sha256_file(
                    current_path
                ):
                    reasons.append(f"SOURCE_COMMIT_CONTENT_MISMATCH:{relative}")
    if coordinator.get("branches") != EXPECTED_BRANCH_PATHS:
        reasons.append("BRANCH_PATHS_INVALID")
    expected_branch_hashes = coordinator.get("branch_config_sha256", {})
    if set(expected_branch_hashes) != set(EXPECTED_BRANCH_PATHS):
        reasons.append("BRANCH_CONFIG_HASH_ROSTER_INVALID")
    else:
        for branch, relative in EXPECTED_BRANCH_PATHS.items():
            if expected_branch_hashes[branch] != sha256_file(ROOT / relative):
                reasons.append(f"BRANCH_CONFIG_HASH_MISMATCH:{branch}")
    return not reasons, reasons


def recursively_forbidden_claims(value: Any) -> list[str]:
    reasons: list[str] = []
    forbidden_keys = {
        "gpu_allowed",
        "go_no_go",
        "combined_score",
        "cross_branch_rank",
        "architecture_winner",
    }

    def visit(node: Any, path: str) -> None:
        if isinstance(node, dict):
            for key, child in node.items():
                normalized = str(key).casefold()
                child_path = f"{path}.{key}" if path else str(key)
                if normalized in forbidden_keys:
                    reasons.append(f"FORBIDDEN_FIELD:{child_path}")
                if normalized == "architecture_promoted" and child is not False:
                    reasons.append(f"ARCHITECTURE_PROMOTION:{child_path}")
                visit(child, child_path)
        elif isinstance(node, list):
            for index, child in enumerate(node):
                visit(child, f"{path}[{index}]")
        elif isinstance(node, str) and "READY_FOR_EXECUTION" in node.upper():
            reasons.append(f"EXECUTION_READINESS_FORBIDDEN:{path}")

    visit(value, "")
    return reasons


def evaluate_predicates(
    coordinator: dict[str, Any],
    relational: dict[str, Any],
    set_valued: dict[str, Any],
    numeric_status: str,
) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    sources_ok, source_reasons = source_bindings_valid(
        coordinator, [relational, set_valued]
    )
    rows["C1_SOURCE_BINDING"] = {
        "status": "PASS" if sources_ok else "FAIL",
        "reasons": source_reasons,
    }
    forbidden = set(coordinator.get("forbidden_fields", []))
    branch_ids = {relational.get("branch_id"), set_valued.get("branch_id")}
    separation_reasons = recursively_forbidden_claims(
        {"coordinator": coordinator, "relational": relational, "set_valued": set_valued}
    )
    rows["C2_BRANCH_SEPARATION"] = result(
        branch_ids == {"relational", "set_valued"}
        and "combined_score" in forbidden
        and "cross_branch_rank" in forbidden,
        "BRANCH_SEPARATION_INVALID",
    )
    if separation_reasons:
        rows["C2_BRANCH_SEPARATION"] = {
            "status": "FAIL",
            "reasons": separation_reasons,
        }
    rows["C3_PHASE_DAG"] = result(
        coordinator.get("phase_dag") == EXPECTED_PHASE_DAG,
        "PHASE_DAG_INVALID",
    )
    replay = coordinator.get("replay", {})
    rows["C4_REPLAY_CONTRACT"] = result(
        replay.get("roots") == ["run_a", "run_b"]
        and replay.get("scientific_files_byte_exact") is True
        and replay.get("runtime_observation_excluded") is True
        and replay.get("timestamps_in_scientific_files") is False,
        "REPLAY_CONTRACT_INVALID",
    )
    rows["C5_NO_GPU_NO_PROMOTION"] = result(
        coordinator.get("fixed_claims") == FIXED_CLAIMS
        and relational.get("fixed_claims") == FIXED_CLAIMS
        and set_valued.get("fixed_claims") == FIXED_CLAIMS
        and coordinator.get("runtime", {}).get("device") == "cpu"
        and coordinator.get("runtime", {}).get("cuda_visible_devices") == ""
        and coordinator.get("runtime", {}).get("query_accelerator_devices") is False
        and not separation_reasons,
        "GPU_OR_PROMOTION_POLICY_INVALID",
    )
    allowed = coordinator.get("allowed_design_states", [])
    rows["C6_READINESS_SEMANTICS"] = result(
        coordinator.get("execution_claim_allowed") is False
        and all("READY_FOR_EXECUTION" not in state for state in allowed)
        and relational.get("design_status")
        == "READY_FOR_RUNNER_IMPLEMENTATION_IF_PREFLIGHT_PASS"
        and set_valued.get("design_status")
        == "READY_FOR_RUNNER_IMPLEMENTATION_IF_PREFLIGHT_PASS",
        "READINESS_SEMANTICS_INVALID",
    )

    interface = relational.get("interface", {})
    arrays = interface.get("arrays", {})
    public_fields = set(interface.get("public_model_fields", []))
    private_fields = set(interface.get("private_sidecar_fields", []))
    rows["R1_PUBLIC_PRIVATE_SCHEMA"] = result(
        public_fields.isdisjoint(private_fields)
        and {"master_id", "mechanism", "x_true", "clean_log_ratio"}.issubset(
            private_fields
        )
        and {"observed_log_ratio", "path_index", "edge_variance"}.issubset(
            public_fields
        )
        and arrays.get("raw_reliability")
        == "edge_float64_in_[weight_floor,1]"
        and arrays.get("normalized_wls_weight")
        == "edge_float64_positive_valid_mean_1"
        and arrays.get("normalized_irls_base_weight")
        == "edge_float64_positive_valid_mean_1"
        and arrays.get("normalized_irls_base_weight")
        != arrays.get("raw_reliability"),
        "RELATIONAL_SCHEMA_INVALID",
    )
    generator = relational.get("generator", {})
    split_masters = generator.get("split_masters", {})
    minimum = generator.get("minimum_effective_masters", {})
    rows["R2_MASTER_SPLIT"] = result(
        generator.get("masters") == 1024
        and sum(split_masters.values()) == 1024
        and generator.get("split_unit") == "master_id"
        and all(minimum.get(key, 0) <= value for key, value in split_masters.items()),
        "MASTER_SPLIT_INVALID",
    )
    arms = relational.get("arms", {})
    rows["R3_ARM_PARITY"] = result(
        arms.get("primary") == ["RAW_GENERIC", "RAW_TYPED"]
        and arms.get("trained_control") == "RAW_TYPED_PATH_SHUFFLE"
        and len(arms.get("training_seeds", [])) == 3
        and "parameter_count" in arms.get("parity_fields", [])
        and arms.get("expected_primary_parameters") == 50435,
        "ARM_PARITY_INVALID",
    )
    loss = relational.get("training", {}).get("loss", {})
    surrogate = relational.get("executors", {}).get("training_surrogate", {})
    rows["R4_DUAL_SOLVER_LOSS"] = result(
        loss
        == {
            "relation_mse": 1.0,
            "local_closure_l1": 0.05,
            "quotient_wls_mse": 0.5,
            "quotient_fixed_k192_irls_mse": 0.5,
        }
        and surrogate.get("steps") == 192
        and surrogate.get("base_weights") == "raw_reliability"
        and surrogate.get("unit_base_evidence_is_sufficient") is False
        and surrogate.get("retune_after_confirmation_failure") is False,
        "DUAL_SOLVER_LOSS_INVALID",
    )
    executors = relational.get("executors", {})
    rows["R5_EXECUTOR_PARITY"] = result(
        executors.get("wls", {}).get("weight_floor")
        == executors.get("irls", {}).get("weight_floor")
        == interface.get("weight_floor")
        and executors.get("irls", {}).get("delta") == 1.5
        and executors.get("irls", {}).get("damping") == 1.0
        and executors.get("irls", {}).get("max_iterations") == 7500,
        "EXECUTOR_PARITY_INVALID",
    )
    rows["R6_SOLVER_DENOMINATORS"] = result(
        relational.get("solver_denominators")
        == {
            "wls": "all_valid_roster_masters",
            "irls": "common_converged_typed_generic",
            "interaction": "four_cell_common",
            "failure_rate": "all_roster_masters",
        }
        and relational.get("inference", {}).get("irls_minimum_common_coverage")
        == 0.99,
        "SOLVER_DENOMINATORS_INVALID",
    )
    path_control = relational.get("path_control", {})
    rows["R7_PATH_CONTROL_ROSTER"] = result(
        path_control.get("function") == "shuffled_path_tensors"
        and path_control.get("public_structure_only") is True
        and path_control.get("private_mutation_byte_invariant") is True
        and path_control.get("common_roster") is True
        and path_control.get("identity_allowed") is False
        and "master_id" not in path_control.get("seed_function", "")
        and "mechanism" not in path_control.get("seed_function", ""),
        "PATH_CONTROL_ROSTER_INVALID",
    )
    controls = set(relational.get("controls", []))
    rows["R8_WEIGHT_CONTROLS"] = result(
        {"UNIT_WEIGHT", "WEIGHT_LOCATION_SHUFFLE"}.issubset(controls),
        "WEIGHT_CONTROLS_INVALID",
    )
    rows["R9_TOTAL_TARGET_SHUFFLE"] = result(
        "TOTAL_TARGET_SHUFFLE" in controls,
        "TOTAL_TARGET_SHUFFLE_MISSING",
    )
    inference = relational.get("inference", {})
    rows["R10_SEED_ESTIMAND"] = result(
        inference.get("estimand")
        == "mean_effect_conditional_on_three_fixed_training_seeds"
        and inference.get("bootstrap_unit") == "master_id"
        and inference.get("seed_population_claim") is False,
        "SEED_ESTIMAND_INVALID",
    )
    rows["R11_BASE_WEIGHTED_K192_CONFORMANCE"] = result(
        numeric_status == "PASS"
        and relational.get("fixed_depth_conformance", {}).get("steps") == 192
        and relational.get("fixed_depth_conformance", {}).get("seed") == 2026090731
        and relational.get("fixed_depth_conformance", {}).get("calibration_seed")
        == 2026090723
        and relational.get("fixed_depth_conformance", {}).get(
            "calibration_seed_reuse_allowed"
        )
        is False
        and relational.get("fixed_depth_conformance", {}).get("gradient_families")
        == ["corrected_log_ratio", "raw_reliability"],
        "BASE_WEIGHTED_K192_NOT_CONFORMANT",
    )
    rows["R12_DECISION_TABLE"] = result(
        relational.get("required_decision_rows") == EXPECTED_REL_ROWS,
        "RELATIONAL_DECISION_TABLE_INVALID",
    )

    fresh = set_valued.get("fresh_draw", {})
    roles = fresh.get("roles", {})
    minima = fresh.get("minimum_roles", {})
    rows["S1_PHASE_SUPPORT"] = result(
        roles
        == {
            "posterior_fit": 384,
            "policy_fit": 384,
            "decision_select": 768,
            "monitor": 768,
        }
        and all(minima.get(key, 0) <= value for key, value in roles.items())
        and fresh.get("redraw_on_low_support") is False
        and set_valued.get("phase_access") == EXPECTED_PHASE_ACCESS,
        "SET_PHASE_SUPPORT_INVALID",
    )
    observation = set_valued.get("observation", {})
    target = set_valued.get("target", {})
    utility = set_valued.get("utility", {})
    rows["S2_LOGIT_TARGET_SEPARATION"] = result(
        observation.get("ensemble_logits") == 4
        and observation.get("per_seed_logits") == [3, 4]
        and target.get("families") == 4
        and target.get("empty_allowed") is False
        and utility.get("external_to_posterior") is True
        and set(observation)
        == {"ensemble_logits", "per_seed_logits", "checkpoint_epochs"}
        and set(target) == {"families", "nonempty_sets", "empty_allowed"}
        and set(utility) == {"policies", "incompatible_penalty", "external_to_posterior"},
        "LOGIT_TARGET_SEPARATION_INVALID",
    )
    representations = set_valued.get("representations", {})
    target_shuffle = set_valued.get("target_shuffle", {})
    rows["S3_POSTERIOR_PARITY"] = result(
        set(representations) == {"marginal", "joint"}
        and target.get("nonempty_sets") == 15
        and target_shuffle.get("same_map_for_representations") is True,
        "POSTERIOR_PARITY_INVALID",
    )
    marginal = representations.get("marginal", {})
    joint = representations.get("joint", {})
    rows["S4_NATIVE_FIT_RECIPES"] = result(
        marginal.get("recipe") == "pooled_platt"
        and marginal.get("C") == 1.0
        and marginal.get("hyperparameter_selection") is False
        and joint.get("recipe") == "joint_full"
        and joint.get("regularization_grid")
        == [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]
        and joint.get("folds") == 4
        and "cluster_id" not in joint.get("fold_key", ""),
        "NATIVE_FIT_RECIPES_INVALID",
    )
    hard = set_valued.get("hard_reader", {})
    rows["S5_HARD_POSTERIOR_BINDING"] = result(
        hard.get("recipe") == "HARD_MAP_SET"
        and hard.get("posterior_bound") is True
        and hard.get("set_tie") == "lowest_binary_index",
        "HARD_POSTERIOR_BINDING_INVALID",
    )
    contextual = set_valued.get("contextual_reader", {})
    feature_names = contextual.get("feature_names", [])
    rows["S6_CONTEXTUAL_RECIPE"] = result(
        contextual.get("candidate") == "minimum_posterior_risk"
        and contextual.get("feature_count") == len(feature_names) == 17
        and "baseline_map_cardinality" in feature_names
        and "posterior_mass_baseline_map_set" in feature_names
        and contextual.get("same_recipe_for_posteriors") is True,
        "CONTEXTUAL_RECIPE_INVALID",
    )
    rows["S7_PROPOSER_GUARD_CREDIT"] = result(
        contextual.get("fit_role") == "policy_fit"
        and contextual.get("selection_role") == "decision_select"
        and contextual.get("proposer", {}).get("model") == "Ridge"
        and contextual.get("harm_guard", {}).get("target") == "gain < -1e-12"
        and contextual.get("incompatibility_guard", {}).get("target")
        == "not target[candidate]",
        "PROPOSER_GUARD_CREDIT_INVALID",
    )
    rows["S8_TARGET_SHUFFLE_FOLDS"] = result(
        target_shuffle.get("function") == "target_derangement_v1"
        and target_shuffle.get("seed") == 53602
        and target_shuffle.get("strata")
        == ["fold_id", "design_stratum", "cardinality"]
        and target_shuffle.get("algorithm")
        == "pcg64_random_raw_order_nonzero_cyclic_shift"
        and target_shuffle.get("canonical_json") is True
        and target_shuffle.get("fixture_sha256")
        == "d7aa2f128b6d42dbe7448415dcd8d4d69ca0ad8311394a5b1209ca9579e03904"
        and target_shuffle.get("cross_fold_allowed") is False
        and target_shuffle.get("minimum_permutable_fraction") == 0.8
        and target_shuffle.get("identity_allowed") is False,
        "TARGET_SHUFFLE_FOLDS_INVALID",
    )
    matched = set_valued.get("matched_controls", {})
    rows["S9_MATCHED_CONTROL_TARGET_BLIND"] = result(
        matched.get("seeds") == [53611, 53617, 53623, 53629, 53633]
        and matched.get("application_is_target_blind") is True
        and matched.get("common_support")
        == "intersection_of_true_override_and_all_five_match_masks"
        and matched.get("missing_control_averaging") is False
        and matched.get("minimum_common_coverage") == 0.8
        and matched.get("matching_reads_target") is False
        and set(matched)
        == {
            "seeds",
            "target_shuffle_strata",
            "minimum_permutable_fraction",
            "application_is_target_blind",
            "matching_reads_target",
            "match_count_per_token",
            "common_support",
            "minimum_common_coverage",
            "missing_control_averaging",
        },
        "MATCHED_CONTROL_TARGET_BLIND_INVALID",
    )
    rows["S10_CELL_DUPLICATION"] = result(
        set_valued.get("cell_duplication_policy") == "declare_and_preserve",
        "CELL_DUPLICATION_POLICY_INVALID",
    )
    rows["S11_ESTIMAND_ORDER"] = result(
        set_valued.get("estimand_order")
        == [
            "integrity_phase_support_normalization",
            "joint_minus_marginal_and_shuffle",
            "contextual_minus_hard",
            "representation_within_reader",
            "factor_interaction",
            "matched_controls_and_duplicates",
            "utility_checkpoint_cardinality_sensitivity",
        ],
        "SET_ESTIMAND_ORDER_INVALID",
    )
    rows["S12_DECISION_TABLE"] = result(
        set_valued.get("required_decision_rows") == EXPECTED_SET_ROWS,
        "SET_DECISION_TABLE_INVALID",
    )
    if set(rows) != set(ALL_PREDICATES):
        raise AssertionError("predicate roster mismatch")
    return rows


def numpy_fixed_weighted_irls(
    observation: Any,
    values: np.ndarray,
    base_weights: np.ndarray,
    *,
    steps: int,
    delta: float,
    damping: float,
    weight_floor: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Independent NumPy reference; it imports no Torch implementation helper."""
    from geometria_proporcional.proportional_graph_contract import incidence_matrix

    valid = np.asarray(observation.edge_valid, dtype=bool)
    matrix = incidence_matrix(observation.n_nodes, observation.edge_index)[valid]
    y = np.asarray(values, dtype=np.float64)[valid]
    base = np.clip(np.asarray(base_weights, dtype=np.float64)[valid], weight_floor, None)
    base /= base.mean()
    variance = np.asarray(observation.edge_variance, dtype=np.float64)[valid]
    scale = max(float(np.sqrt(np.median(variance))), 1e-8)
    threshold = delta * scale
    weights = base.copy()

    def solve(local: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        normalized = np.clip(local, weight_floor, None)
        normalized /= normalized.mean()
        laplacian = matrix.T @ (normalized[:, None] * matrix)
        rhs = matrix.T @ (normalized * y)
        ones = np.ones((observation.n_nodes, 1), dtype=np.float64)
        kkt = np.block([[laplacian, ones], [ones.T, np.zeros((1, 1))]])
        solution = np.linalg.solve(kkt, np.concatenate((rhs, np.zeros(1))))
        return solution[:-1], normalized

    for _ in range(steps):
        x_hat, _ = solve(weights)
        residual = matrix @ x_hat - y
        candidate = np.ones_like(residual)
        large = np.abs(residual) > threshold
        candidate[large] = threshold / np.abs(residual[large])
        candidate = np.clip(candidate, weight_floor, 1.0)
        weights = (1.0 - damping) * weights + damping * (base * candidate)
    x_hat, normalized = solve(weights)
    normalized_residual = (matrix @ x_hat - y) / scale
    magnitude = np.abs(normalized_residual)
    huber = np.where(
        magnitude <= delta,
        0.5 * normalized_residual**2,
        delta * (magnitude - 0.5 * delta),
    )
    full_weights = np.zeros(len(values), dtype=np.float64)
    full_weights[valid] = normalized
    return x_hat, full_weights, float(np.sum(base * huber))


def weight_patterns(values: np.ndarray) -> list[np.ndarray]:
    size = len(values)
    order = np.argsort(np.abs(values), kind="stable")
    first = np.linspace(0.1, 1.0, size, dtype=np.float64)
    second = np.geomspace(0.01, 0.9, size, dtype=np.float64)
    second = second[np.argsort(order, kind="stable")]
    scale = max(float(np.std(values)), 1e-3)
    third = 0.05 + 0.95 / (1.0 + np.exp(np.abs(values) / scale))
    return [first, second, third.astype(np.float64)]


def select_views_by_size(views: list[Any], count: int) -> list[Any]:
    """Deterministic round-robin over every node count available."""
    groups = {
        size: sorted(
            [view for view in views if view.public.n_nodes == size],
            key=lambda view: view.private.view_id,
        )
        for size in sorted({view.public.n_nodes for view in views})
    }
    selected: list[Any] = []
    offset = 0
    while len(selected) < count:
        advanced = False
        for size in groups:
            if offset < len(groups[size]) and len(selected) < count:
                selected.append(groups[size][offset])
                advanced = True
        if not advanced:
            break
        offset += 1
    if len(selected) != count:
        raise RuntimeError("insufficient views for stratified fixed-depth sample")
    return selected


def _gradient_summary(
    records: list[tuple[np.ndarray, np.ndarray]], stable_magnitude: float
) -> dict[str, Any]:
    cosines: list[float] = []
    relative: list[float] = []
    inversions = 0
    coordinates = 0
    for analytic, numeric in records:
        if not len(analytic):
            continue
        denom = float(np.linalg.norm(analytic) * np.linalg.norm(numeric))
        cosine = 1.0 if denom == 0.0 and np.allclose(analytic, numeric) else float(
            np.dot(analytic, numeric) / max(denom, np.finfo(np.float64).tiny)
        )
        cosines.append(cosine)
        relative.append(
            float(
                np.linalg.norm(analytic - numeric)
                / max(np.linalg.norm(numeric), np.finfo(np.float64).tiny)
            )
        )
        stable = (np.abs(analytic) >= stable_magnitude) & (
            np.abs(numeric) >= stable_magnitude
        )
        inversions += int(np.sum(np.signbit(analytic[stable]) != np.signbit(numeric[stable])))
        coordinates += len(analytic)
    return {
        "probes": len(cosines),
        "coordinates": coordinates,
        "median_cosine": float(np.median(cosines)) if cosines else 0.0,
        "p95_relative_error": float(np.quantile(relative, 0.95)) if relative else float("inf"),
        "sign_inversions": inversions,
    }


def run_fixed_depth_conformance(
    config: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be the empty string")
    import torch

    from geometria_proporcional.proportional_graph_contract import (
        ProportionalGraphConfig,
        generate_graph_views,
        solve_huber_irls,
    )
    from geometria_proporcional.proportional_graph_neural import (
        differentiable_huber_irls_fixed,
    )

    torch.set_num_threads(1)
    recipe = config["fixed_depth_conformance"]
    steps = int(recipe["steps"])
    executor = config["executors"]["irls"]
    requested_graphs = int(recipe["graphs"])
    graph_config = ProportionalGraphConfig(
        masters=requested_graphs * 2,
        train_fraction=0.5,
        calibration_fraction=0.125,
        validation_fraction=0.125,
        n_min=8,
        n_max=16,
        noise_sigma=0.04,
        corruption_rate=0.15,
        corruption_amplitude_min=0.6,
        corruption_amplitude_max=1.4,
        seed=int(recipe["seed"]),
    )
    all_views = generate_graph_views(graph_config)
    grouped = [
        view for view in all_views if view.private.corruption_mechanism == "grouped"
    ]
    iid = [
        view
        for view in all_views
        if view.private.corruption_mechanism == "iid" and view.private.split != "test"
    ]
    grouped_count = requested_graphs // 2
    views = select_views_by_size(grouped, grouped_count) + select_views_by_size(
        iid, requested_graphs - grouped_count
    )
    if len(views) != requested_graphs:
        raise RuntimeError("insufficient synthetic views for fixed-depth conformance")

    fixed_errors: list[float] = []
    canonical_rmse: list[float] = []
    convergence: list[bool] = []
    canonical_iterations: list[int] = []
    state_ids: list[str] = []
    relation_gradient_records: list[tuple[np.ndarray, np.ndarray]] = []
    weight_gradient_records: list[tuple[np.ndarray, np.ndarray]] = []
    excluded_relation = 0
    excluded_weight = 0
    gradient_state_indices = set(
        np.linspace(
            0,
            len(views) * 3 - 1,
            int(recipe["gradient_probe_states"]),
            dtype=int,
        ).tolist()
    )
    state_index = 0
    for view in views:
        values = np.asarray(view.public.observed_log_ratio, dtype=np.float64)
        for pattern_index, base in enumerate(weight_patterns(values)):
            nx, nw, no = numpy_fixed_weighted_irls(
                view.public,
                values,
                base,
                steps=steps,
                delta=float(executor["delta"]),
                damping=float(executor["damping"]),
                weight_floor=float(executor["weight_floor"]),
            )
            value_tensor = torch.tensor(values, dtype=torch.float64)
            base_tensor = torch.tensor(base, dtype=torch.float64)
            output = differentiable_huber_irls_fixed(
                view.public,
                value_tensor,
                base_weights=base_tensor,
                steps=steps,
                delta=float(executor["delta"]),
                damping=float(executor["damping"]),
                weight_floor=float(executor["weight_floor"]),
            )
            fixed_errors.append(
                max(
                    float(np.max(np.abs(output.x_hat.detach().numpy() - nx))),
                    float(np.max(np.abs(output.normalized_weights.detach().numpy() - nw))),
                    abs(float(output.huber_objective.detach()) - no),
                )
            )
            canonical = solve_huber_irls(
                view.public,
                values=values,
                base_weights=base,
                delta=float(executor["delta"]),
                damping=float(executor["damping"]),
                tolerance=float(executor["tolerance"]),
                max_iterations=int(executor["max_iterations"]),
                weight_floor=float(executor["weight_floor"]),
            )
            convergence.append(bool(canonical.converged))
            canonical_iterations.append(int(canonical.iterations))
            canonical_rmse.append(float(np.sqrt(np.mean((nx - canonical.x_hat) ** 2))))
            state_ids.append(
                f"{view.private.view_id}|{view.private.corruption_mechanism}|w{pattern_index}"
            )

            if state_index in gradient_state_indices:
                vt = torch.tensor(values, dtype=torch.float64, requires_grad=True)
                bt = torch.tensor(base, dtype=torch.float64, requires_grad=True)
                grad_output = differentiable_huber_irls_fixed(
                    view.public,
                    vt,
                    base_weights=bt,
                    steps=steps,
                    delta=float(executor["delta"]),
                    damping=float(executor["damping"]),
                    weight_floor=float(executor["weight_floor"]),
                )
                target = torch.tensor(view.private.x_true, dtype=torch.float64)
                torch.mean((grad_output.x_hat - target) ** 2).backward()
                valid_edges = np.flatnonzero(view.public.edge_valid)
                chosen = valid_edges[
                    np.linspace(0, len(valid_edges) - 1, min(8, len(valid_edges)), dtype=int)
                ]
                analytic_relation: list[float] = []
                numeric_relation: list[float] = []
                analytic_weight: list[float] = []
                numeric_weight: list[float] = []
                margins = grad_output.min_huber_margin.detach().numpy()
                for edge in chosen:
                    h_value = 1e-5 * max(1.0, abs(float(values[edge])))
                    if margins[edge] <= 10.0 * h_value:
                        excluded_relation += 1
                    else:
                        losses = []
                        for sign in (-1.0, 1.0):
                            perturbed = values.copy()
                            perturbed[edge] += sign * h_value
                            px, _, _ = numpy_fixed_weighted_irls(
                                view.public,
                                perturbed,
                                base,
                                steps=steps,
                                delta=float(executor["delta"]),
                                damping=float(executor["damping"]),
                                weight_floor=float(executor["weight_floor"]),
                            )
                            losses.append(float(np.mean((px - view.private.x_true) ** 2)))
                        analytic_relation.append(float(vt.grad.detach().numpy()[edge]))
                        numeric_relation.append((losses[1] - losses[0]) / (2.0 * h_value))
                    h_weight = 1e-5 * max(1.0, abs(float(base[edge])))
                    if (
                        base[edge] - h_weight <= float(executor["weight_floor"])
                        or float(np.min(margins[view.public.edge_valid])) <= 10.0 * h_weight
                    ):
                        excluded_weight += 1
                    else:
                        losses = []
                        for sign in (-1.0, 1.0):
                            perturbed = base.copy()
                            perturbed[edge] += sign * h_weight
                            px, _, _ = numpy_fixed_weighted_irls(
                                view.public,
                                values,
                                perturbed,
                                steps=steps,
                                delta=float(executor["delta"]),
                                damping=float(executor["damping"]),
                                weight_floor=float(executor["weight_floor"]),
                            )
                            losses.append(float(np.mean((px - view.private.x_true) ** 2)))
                        analytic_weight.append(float(bt.grad.detach().numpy()[edge]))
                        numeric_weight.append((losses[1] - losses[0]) / (2.0 * h_weight))
                relation_gradient_records.append(
                    (np.asarray(analytic_relation), np.asarray(numeric_relation))
                )
                weight_gradient_records.append(
                    (np.asarray(analytic_weight), np.asarray(numeric_weight))
                )
            state_index += 1

    relation_grad = _gradient_summary(
        relation_gradient_records, float(recipe["stable_gradient_magnitude"])
    )
    weight_grad = _gradient_summary(
        weight_gradient_records, float(recipe["stable_gradient_magnitude"])
    )
    relation_grad["excluded"] = excluded_relation
    weight_grad["excluded"] = excluded_weight
    max_fixed = float(np.max(fixed_errors))
    converged_rmse = np.asarray(canonical_rmse, dtype=np.float64)[
        np.asarray(convergence, dtype=bool)
    ]
    p99_canonical = (
        float(np.quantile(converged_rmse, 0.99)) if len(converged_rmse) else float("inf")
    )
    max_canonical = float(np.max(converged_rmse)) if len(converged_rmse) else float("inf")
    gradients_ok = all(
        summary["probes"] > 0
        and summary["coordinates"] > 0
        and summary["median_cosine"] >= float(recipe["gradient_median_cosine"])
        and summary["p95_relative_error"]
        <= float(recipe["gradient_p95_relative_error"])
        and summary["sign_inversions"] <= int(recipe["allowed_sign_inversions"])
        for summary in (relation_grad, weight_grad)
    )
    node_counts = sorted({view.public.n_nodes for view in views})
    range_covered = node_counts == list(range(8, 17))
    passed = (
        max_fixed <= float(recipe["max_torch_numpy_error"])
        and all(convergence)
        and range_covered
        and p99_canonical <= float(recipe["canonical_p99_rmse"])
        and max_canonical <= float(recipe["canonical_max_rmse"])
        and gradients_ok
    )
    summary = {
        "status": "PASS" if passed else "FAIL",
        "steps": steps,
        "graphs": len(views),
        "states": len(state_ids),
        "mechanisms": sorted({view.private.corruption_mechanism for view in views}),
        "node_counts": node_counts,
        "node_range_covered": range_covered,
        "canonical_converged": int(np.sum(convergence)),
        "canonical_failed": int(len(convergence) - np.sum(convergence)),
        "all_canonical_converged": bool(all(convergence)),
        "canonical_iterations": {
            "min": int(np.min(canonical_iterations)),
            "median": float(np.median(canonical_iterations)),
            "p99": float(np.quantile(canonical_iterations, 0.99)),
            "max": int(np.max(canonical_iterations)),
        },
        "max_torch_numpy_error": max_fixed,
        "canonical_p99_rmse": p99_canonical,
        "canonical_max_rmse": max_canonical,
        "relation_gradient": relation_grad,
        "weight_gradient": weight_grad,
    }
    raw = {
        "state_id": np.asarray(state_ids),
        "fixed_error": np.asarray(fixed_errors, dtype=np.float64),
        "canonical_rmse": np.asarray(canonical_rmse, dtype=np.float64),
        "canonical_converged": np.asarray(convergence, dtype=bool),
        "canonical_iterations": np.asarray(canonical_iterations, dtype=np.int64),
    }
    return summary, raw


def target_derangement_v1(rows: list[dict[str, Any]], seed: int = 53602) -> list[dict[str, Any]]:
    rng = np.random.Generator(np.random.PCG64(seed))
    grouped: dict[tuple[int, str, int], list[str]] = {}
    metadata: dict[str, tuple[int, str, int]] = {}
    for row in rows:
        token = str(row["pair_token"])
        key = (int(row["fold_id"]), str(row["design_stratum"]), int(row["cardinality"]))
        grouped.setdefault(key, []).append(token)
        metadata[token] = key
    mapping: dict[str, str] = {}
    for stratum in sorted(grouped):
        tokens = sorted(grouped[stratum])
        if len(tokens) == 1:
            mapping[tokens[0]] = tokens[0]
            continue
        random_keys = [int(rng.bit_generator.random_raw()) for _ in tokens]
        order = [
            token
            for _, token in sorted(
                zip(random_keys, tokens, strict=True), key=lambda item: (item[0], item[1])
            )
        ]
        shift = 1 + int(rng.integers(0, len(order) - 1, endpoint=False))
        for index, receiver in enumerate(order):
            mapping[receiver] = order[(index + shift) % len(order)]
    output = []
    for receiver in sorted(mapping):
        fold, stratum, cardinality = metadata[receiver]
        output.append(
            {
                "receiver": receiver,
                "donor": mapping[receiver],
                "fold_id": fold,
                "design_stratum": stratum,
                "cardinality": cardinality,
            }
        )
    return output


def run_fixtures(expected_target_digest: str | None = None) -> dict[str, Any]:
    shuffle_rows = [
        {"pair_token": "a", "fold_id": 0, "design_stratum": "FAR", "cardinality": 2},
        {"pair_token": "b", "fold_id": 0, "design_stratum": "FAR", "cardinality": 2},
        {"pair_token": "c", "fold_id": 0, "design_stratum": "FAR", "cardinality": 2},
        {"pair_token": "d", "fold_id": 0, "design_stratum": "NEAR", "cardinality": 1},
        {"pair_token": "e", "fold_id": 1, "design_stratum": "FAR", "cardinality": 2},
        {"pair_token": "f", "fold_id": 1, "design_stratum": "FAR", "cardinality": 2},
    ]
    mapping = target_derangement_v1(shuffle_rows)
    mapping_digest = sha256_bytes(canonical_bytes(mapping))
    non_singleton = [row for row in mapping if row["receiver"] != "d"]
    derangement_ok = all(row["receiver"] != row["donor"] for row in non_singleton)
    strata_ok = all(
        next(item for item in shuffle_rows if item["pair_token"] == row["receiver"])[
            "fold_id"
        ]
        == next(item for item in shuffle_rows if item["pair_token"] == row["donor"])[
            "fold_id"
        ]
        for row in mapping
    )

    true_override = np.asarray([True, True, True, True, True, False])
    masks = [
        np.asarray(values, dtype=bool)
        for values in (
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 0, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
        )
    ]
    common = true_override.copy()
    for mask in masks:
        common &= mask
    coverage = float(np.sum(common) / np.sum(true_override))
    posterior = np.full(15, 0.005, dtype=np.float64)
    posterior[[2, 7, 8, 11]] = [0.30, 0.25, 0.175, 0.20]
    posterior /= posterior.sum()
    set_codes = np.arange(1, 16, dtype=np.int64)
    hard_map_code = int(set_codes[int(np.argmax(posterior))])
    marginals = np.asarray(
        [np.sum(posterior[(set_codes & (1 << bit)) != 0]) for bit in range(4)]
    )
    threshold_code = int(sum((value >= 0.5) << bit for bit, value in enumerate(marginals)))

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be the empty string")
    import torch

    from geometria_proporcional.proportional_graph_contract import (
        ProportionalGraphConfig,
        generate_graph_views,
    )
    from geometria_proporcional.proportional_graph_neural import (
        observation_tensors,
        shuffled_path_tensors,
    )

    def structure_seed(view: Any) -> int:
        digest = hashlib.sha256()
        for name, value in sorted(view.public.arrays().items()):
            if name == "observed_log_ratio":
                continue
            array = np.ascontiguousarray(value)
            digest.update(name.encode("utf-8"))
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(canonical_bytes(list(array.shape)))
            digest.update(array.tobytes())
        return int.from_bytes(digest.digest()[:8], "little") % (2**63 - 1)

    candidate_views = generate_graph_views(
        ProportionalGraphConfig(masters=24, n_min=8, n_max=10, seed=2026090753)
    )
    reference_view = None
    reference_shuffle = None
    for candidate in candidate_views:
        tensors = observation_tensors(candidate.public, device="cpu", dtype=torch.float64)
        shuffled = shuffled_path_tensors(tensors, seed=structure_seed(candidate))
        if bool(shuffled["path_shuffle_eligible"]):
            reference_view, reference_shuffle = candidate, shuffled
            break
    if reference_view is None or reference_shuffle is None:
        raise RuntimeError("path-control fixture found no eligible public graph")

    mutated_fields: list[str] = []
    private_invariant = True
    for field in fields(reference_view.private):
        original = getattr(reference_view.private, field.name)
        if isinstance(original, np.ndarray):
            if original.dtype == np.bool_:
                mutated = np.logical_not(original)
            else:
                mutated = original + np.ones_like(original)
        elif isinstance(original, int):
            mutated = original + 1
        else:
            mutated = f"private-mutated-{field.name}"
        variant = replace(
            reference_view,
            private=replace(reference_view.private, **{field.name: mutated}),
        )
        tensors = observation_tensors(variant.public, device="cpu", dtype=torch.float64)
        shuffled = shuffled_path_tensors(tensors, seed=structure_seed(variant))
        same = (
            bool(shuffled["path_shuffle_eligible"])
            == bool(reference_shuffle["path_shuffle_eligible"])
            and torch.equal(shuffled["path_index"], reference_shuffle["path_index"])
            and torch.equal(shuffled["path_sign"], reference_shuffle["path_sign"])
            and torch.equal(shuffled["path_valid"], reference_shuffle["path_valid"])
        )
        private_invariant &= same
        mutated_fields.append(field.name)
    fixture = {
        "target_derangement": {
            "status": "PASS"
            if derangement_ok
            and strata_ok
            and (expected_target_digest is None or mapping_digest == expected_target_digest)
            else "FAIL",
            "sha256": mapping_digest,
            "rows": len(mapping),
            "permutable_fraction": len(non_singleton) / len(mapping),
            "singleton_count": 1,
        },
        "matched_common_support": {
            "status": "PASS" if np.array_equal(common, np.asarray([1, 1, 1, 1, 0, 0], dtype=bool)) else "FAIL",
            "true_override_tokens": int(np.sum(true_override)),
            "common_tokens": int(np.sum(common)),
            "coverage": coverage,
            "uses_intersection": True,
        },
        "posterior_hard_map": {
            "status": "PASS"
            if np.isclose(posterior.sum(), 1.0)
            and np.all(posterior > 0)
            and hard_map_code != threshold_code
            else "FAIL",
            "posterior_sum": float(posterior.sum()),
            "hard_map_code": hard_map_code,
            "threshold_code": threshold_code,
            "recipes_diverge": hard_map_code != threshold_code,
        },
        "path_private_invariance": {
            "status": "PASS" if private_invariant else "FAIL",
            "mutated_fields": mutated_fields,
            "mutated_field_count": len(mutated_fields),
            "public_seed_unchanged": private_invariant,
            "path_tensors_byte_identical": private_invariant,
            "eligible": bool(reference_shuffle["path_shuffle_eligible"]),
        },
    }
    return fixture


Mutation = Callable[[dict[str, Any], dict[str, Any], dict[str, Any]], None]


def mutation_functions() -> dict[str, Mutation]:
    def set_path(container: dict[str, Any], path: list[str], value: Any) -> None:
        cursor = container
        for key in path[:-1]:
            cursor = cursor[key]
        cursor[path[-1]] = value

    mutations: dict[str, Mutation] = {
        "C1_SOURCE_BINDING": lambda c, r, s: r["source_bindings"][0].__setitem__(2, "0" * 64),
        "C2_BRANCH_SEPARATION": lambda c, r, s: c["forbidden_fields"].remove("combined_score"),
        "C3_PHASE_DAG": lambda c, r, s: c["phase_dag"].reverse(),
        "C4_REPLAY_CONTRACT": lambda c, r, s: set_path(c, ["replay", "scientific_files_byte_exact"], False),
        "C5_NO_GPU_NO_PROMOTION": lambda c, r, s: set_path(c, ["fixed_claims", "gpu_used_or_queried"], True),
        "C6_READINESS_SEMANTICS": lambda c, r, s: set_path(c, ["execution_claim_allowed"], True),
        "R1_PUBLIC_PRIVATE_SCHEMA": lambda c, r, s: r["interface"]["public_model_fields"].append("x_true"),
        "R2_MASTER_SPLIT": lambda c, r, s: set_path(r, ["generator", "split_masters", "test"], 255),
        "R3_ARM_PARITY": lambda c, r, s: set_path(r, ["arms", "expected_primary_parameters"], 50436),
        "R4_DUAL_SOLVER_LOSS": lambda c, r, s: set_path(r, ["training", "loss", "quotient_wls_mse"], 1.0),
        "R5_EXECUTOR_PARITY": lambda c, r, s: set_path(r, ["executors", "irls", "weight_floor"], 0.002),
        "R6_SOLVER_DENOMINATORS": lambda c, r, s: set_path(r, ["solver_denominators", "wls"], "four_cell_common"),
        "R7_PATH_CONTROL_ROSTER": lambda c, r, s: set_path(r, ["path_control", "seed_function"], "master_id"),
        "R8_WEIGHT_CONTROLS": lambda c, r, s: r["controls"].remove("UNIT_WEIGHT"),
        "R9_TOTAL_TARGET_SHUFFLE": lambda c, r, s: r["controls"].remove("TOTAL_TARGET_SHUFFLE"),
        "R10_SEED_ESTIMAND": lambda c, r, s: set_path(r, ["inference", "seed_population_claim"], True),
        "R11_BASE_WEIGHTED_K192_CONFORMANCE": lambda c, r, s: set_path(r, ["fixed_depth_conformance", "seed"], 2026090723),
        "R12_DECISION_TABLE": lambda c, r, s: r["required_decision_rows"].pop(),
        "S1_PHASE_SUPPORT": lambda c, r, s: set_path(s, ["fresh_draw", "redraw_on_low_support"], True),
        "S2_LOGIT_TARGET_SEPARATION": lambda c, r, s: set_path(s, ["utility", "external_to_posterior"], False),
        "S3_POSTERIOR_PARITY": lambda c, r, s: set_path(s, ["target_shuffle", "same_map_for_representations"], False),
        "S4_NATIVE_FIT_RECIPES": lambda c, r, s: set_path(s, ["representations", "joint", "fold_key"], "cluster_id"),
        "S5_HARD_POSTERIOR_BINDING": lambda c, r, s: set_path(s, ["hard_reader", "recipe"], "THRESHOLD_0_5"),
        "S6_CONTEXTUAL_RECIPE": lambda c, r, s: s["contextual_reader"]["feature_names"].remove("baseline_map_cardinality"),
        "S7_PROPOSER_GUARD_CREDIT": lambda c, r, s: set_path(s, ["contextual_reader", "fit_role"], "decision_select"),
        "S8_TARGET_SHUFFLE_FOLDS": lambda c, r, s: set_path(s, ["target_shuffle", "strata"], ["cluster_id", "cardinality"]),
        "S9_MATCHED_CONTROL_TARGET_BLIND": lambda c, r, s: set_path(s, ["matched_controls", "missing_control_averaging"], True),
        "S10_CELL_DUPLICATION": lambda c, r, s: set_path(s, ["cell_duplication_policy"], "drop"),
        "S11_ESTIMAND_ORDER": lambda c, r, s: s["estimand_order"].reverse(),
        "S12_DECISION_TABLE": lambda c, r, s: s["required_decision_rows"].pop(),
    }
    return mutations


def run_mutation_suite(
    coordinator: dict[str, Any], relational: dict[str, Any], set_valued: dict[str, Any]
) -> dict[str, Any]:
    mutations = mutation_functions()
    if set(mutations) != set(ALL_PREDICATES):
        raise AssertionError("mutation roster does not cover every predicate")
    rows = []
    for predicate in ALL_PREDICATES:
        c, r, s = copy.deepcopy((coordinator, relational, set_valued))
        mutations[predicate](c, r, s)
        numeric = "FAIL" if predicate == "R11_BASE_WEIGHTED_K192_CONFORMANCE" else "PASS"
        adjudication = evaluate_predicates(c, r, s, numeric)
        caught = adjudication[predicate]["status"] == "FAIL"
        rows.append(
            {
                "predicate": predicate,
                "caught": caught,
                "reasons": adjudication[predicate]["reasons"],
            }
        )
    def set_all_source_commits(
        c: dict[str, Any], r: dict[str, Any], s: dict[str, Any]
    ) -> None:
        unrelated = "333fa3686553e6babeefdcf4922090f6e13c67d8"
        c["source_commit"] = unrelated
        r["source_commit"] = unrelated
        s["source_commit"] = unrelated

    supplemental: list[tuple[str, str, Mutation]] = [
        (
            "C1_UNRELATED_EXISTING_COMMIT",
            "C1_SOURCE_BINDING",
            set_all_source_commits,
        ),
        (
            "C2_CROSS_BRANCH_RANK_FIELD",
            "C2_BRANCH_SEPARATION",
            lambda c, r, s: s.__setitem__("cross_branch_rank", 1),
        ),
        (
            "C5_GPU_ALLOWED_FIELD",
            "C5_NO_GPU_NO_PROMOTION",
            lambda c, r, s: c.__setitem__("gpu_allowed", True),
        ),
        (
            "C5_GO_NOGO_FIELD",
            "C5_NO_GPU_NO_PROMOTION",
            lambda c, r, s: s.__setitem__("go_no_go", "GO"),
        ),
        (
            "C6_READY_FOR_EXECUTION",
            "C6_READINESS_SEMANTICS",
            lambda c, r, s: c["allowed_design_states"].append(
                "READY_FOR_EXECUTION"
            ),
        ),
        (
            "R1_NORMALIZED_IRLS_CONFUSED_WITH_RAW",
            "R1_PUBLIC_PRIVATE_SCHEMA",
            lambda c, r, s: r["interface"]["arrays"].__setitem__(
                "normalized_irls_base_weight", "edge_float64_in_[weight_floor,1]"
            ),
        ),
        (
            "R2_SPLIT_BY_VIEW",
            "R2_MASTER_SPLIT",
            lambda c, r, s: r["generator"].__setitem__("split_unit", "view_id"),
        ),
        (
            "R4_K64_UNIT_BASE_CLAIMED_SUFFICIENT",
            "R4_DUAL_SOLVER_LOSS",
            lambda c, r, s: r["executors"]["training_surrogate"].__setitem__(
                "unit_base_evidence_is_sufficient", True
            ),
        ),
        (
            "R11_K_NOT_192",
            "R11_BASE_WEIGHTED_K192_CONFORMANCE",
            lambda c, r, s: r["fixed_depth_conformance"].__setitem__("steps", 191),
        ),
        (
            "R4_RETUNE_AFTER_CONFIRMATION_FAILURE",
            "R4_DUAL_SOLVER_LOSS",
            lambda c, r, s: r["executors"]["training_surrogate"].__setitem__(
                "retune_after_confirmation_failure", True
            ),
        ),
        (
            "R7_PATH_CONTROL_NO_COMMON_ROSTER",
            "R7_PATH_CONTROL_ROSTER",
            lambda c, r, s: r["path_control"].__setitem__("common_roster", False),
        ),
        (
            "R11_REUSE_CALIBRATION_DRAW",
            "R11_BASE_WEIGHTED_K192_CONFORMANCE",
            lambda c, r, s: r["fixed_depth_conformance"].__setitem__(
                "calibration_seed_reuse_allowed", True
            ),
        ),
        (
            "S1_MONITOR_APPLY_READS_TARGET",
            "S1_PHASE_SUPPORT",
            lambda c, r, s: s["phase_access"]["monitor_apply"].append("target"),
        ),
        (
            "S1_POSTERIOR_FIT_READS_UTILITY",
            "S1_PHASE_SUPPORT",
            lambda c, r, s: s["phase_access"]["posterior_fit"].append("utility"),
        ),
        (
            "S2_TARGET_AS_OBSERVATION",
            "S2_LOGIT_TARGET_SEPARATION",
            lambda c, r, s: s["observation"].__setitem__("target", True),
        ),
        (
            "S6_DIFFERENT_RECIPE_BY_POSTERIOR",
            "S6_CONTEXTUAL_RECIPE",
            lambda c, r, s: s["contextual_reader"].__setitem__(
                "same_recipe_for_posteriors", False
            ),
        ),
        (
            "S8_CROSS_FOLD_TARGET_SHUFFLE",
            "S8_TARGET_SHUFFLE_FOLDS",
            lambda c, r, s: s["target_shuffle"].__setitem__(
                "cross_fold_allowed", True
            ),
        ),
        (
            "S8_TARGET_SHUFFLE_IDENTITY",
            "S8_TARGET_SHUFFLE_FOLDS",
            lambda c, r, s: s["target_shuffle"].__setitem__(
                "identity_allowed", True
            ),
        ),
        (
            "S8_NONCANONICAL_TARGET_MAP",
            "S8_TARGET_SHUFFLE_FOLDS",
            lambda c, r, s: s["target_shuffle"].__setitem__(
                "canonical_json", False
            ),
        ),
        (
            "S9_MATCHING_READS_TARGET",
            "S9_MATCHED_CONTROL_TARGET_BLIND",
            lambda c, r, s: s["matched_controls"].__setitem__(
                "matching_reads_target", True
            ),
        ),
        (
            "S9_SUPPORT_UNION",
            "S9_MATCHED_CONTROL_TARGET_BLIND",
            lambda c, r, s: s["matched_controls"].__setitem__(
                "common_support", "union_of_true_override_and_all_five_match_masks"
            ),
        ),
    ]
    for case_id, predicate, mutate in supplemental:
        c, r, s = copy.deepcopy((coordinator, relational, set_valued))
        mutate(c, r, s)
        adjudication = evaluate_predicates(c, r, s, "PASS")
        caught = adjudication[predicate]["status"] == "FAIL"
        rows.append(
            {
                "case_id": case_id,
                "predicate": predicate,
                "caught": caught,
                "reasons": adjudication[predicate]["reasons"],
            }
        )
    return {
        "status": "PASS" if all(row["caught"] for row in rows) else "FAIL",
        "caught": sum(row["caught"] for row in rows),
        "total": len(rows),
        "rows": rows,
    }


def adjudicate_design_state(
    predicates: dict[str, dict[str, Any]], fixtures: dict[str, Any]
) -> str:
    coordination_ok = all(
        predicates[key]["status"] == "PASS" for key in COORDINATION_PREDICATES
    )
    relational_ok = coordination_ok and all(
        predicates[key]["status"] == "PASS" for key in RELATIONAL_PREDICATES
    )
    set_ok = coordination_ok and all(
        predicates[key]["status"] == "PASS" for key in SET_PREDICATES
    ) and all(row["status"] == "PASS" for row in fixtures.values())
    if not coordination_ok:
        return "TECHNICAL_FAILURE"
    if relational_ok and set_ok:
        return "BOTH_DESIGN_FREEZES_VALID"
    if relational_ok:
        return "RELATIONAL_FREEZE_ONLY_VALID"
    if set_ok:
        return "SET_VALUED_FREEZE_ONLY_VALID"
    return "NEITHER_DESIGN_FREEZE_VALID"


def scientific_payload(
    coordinator: dict[str, Any],
    relational: dict[str, Any],
    set_valued: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    numeric, raw = run_fixed_depth_conformance(relational)
    predicates = evaluate_predicates(coordinator, relational, set_valued, numeric["status"])
    fixtures = run_fixtures(set_valued["target_shuffle"]["fixture_sha256"])
    mutations = run_mutation_suite(coordinator, relational, set_valued)
    design_state = adjudicate_design_state(predicates, fixtures)
    report = {
        "schema_version": "proportional-dual-native-preflight-report-v1",
        "experiment_id": coordinator["experiment_id"],
        "source_commit": coordinator["source_commit"],
        "config_sha256": {
            "coordinator": sha256_file(DEFAULT_COORDINATOR),
            "relational": sha256_file(ROOT / coordinator["branches"]["relational"]),
            "set_valued": sha256_file(ROOT / coordinator["branches"]["set_valued"]),
        },
        "design_state": design_state,
        "execution_claim": "READY_FOR_RUNNER_IMPLEMENTATION_ONLY",
        "predicates": predicates,
        "predicate_counts": {
            "pass": sum(row["status"] == "PASS" for row in predicates.values()),
            "fail": sum(row["status"] == "FAIL" for row in predicates.values()),
            "total": len(predicates),
        },
        "fixed_depth_conformance": numeric,
        "fixtures": fixtures,
        "mutation_suite": mutations,
        "projected_cost": {
            "relational": relational["projected_cost"],
            "set_valued": set_valued["projected_cost"],
        },
        "fixed_claims": FIXED_CLAIMS,
    }
    return report, raw


def write_json(path: Path, value: Any) -> None:
    path.write_bytes(canonical_bytes(value))


def write_deterministic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, array in sorted(arrays.items()):
            payload = io.BytesIO()
            np.lib.format.write_array(payload, np.asarray(array), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, payload.getvalue(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def write_artifact(output: Path, report: dict[str, Any], raw: dict[str, np.ndarray]) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    write_json(output / "scientific_report.json", report)
    write_json(output / "fixtures.json", report["fixtures"])
    write_json(output / "mutation_results.json", report["mutation_suite"])
    write_deterministic_npz(output / "fixed_depth_raw.npz", raw)
    manifest = {
        "schema_version": "proportional-dual-native-preflight-manifest-v1",
        "files": [
            {"path": name, "sha256": sha256_file(output / name), "bytes": (output / name).stat().st_size}
            for name in ARTIFACT_FILES
        ],
        "fixed_claims": FIXED_CLAIMS,
    }
    write_json(output / "manifest.json", manifest)


def check_artifact(output: Path, *, recompute: bool = True) -> dict[str, Any]:
    reasons: list[str] = []
    expected_roster = set(ARTIFACT_FILES + ["manifest.json"])
    actual_roster = {path.name for path in output.iterdir()} if output.is_dir() else set()
    if actual_roster != expected_roster:
        reasons.append(
            "ARTIFACT_ROSTER_MISMATCH:"
            f"missing={sorted(expected_roster - actual_roster)},"
            f"extra={sorted(actual_roster - expected_roster)}"
        )
    try:
        report = load_json(output / "scientific_report.json")
        manifest = load_json(output / "manifest.json")
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        return {"status": "FAIL", "reasons": [*reasons, f"ARTIFACT_UNREADABLE:{type(exc).__name__}"]}
    rows = manifest.get("files")
    if (
        manifest.get("schema_version")
        != "proportional-dual-native-preflight-manifest-v1"
        or manifest.get("fixed_claims") != FIXED_CLAIMS
        or not isinstance(rows, list)
        or [row.get("path") for row in rows if isinstance(row, dict)] != ARTIFACT_FILES
        or len(rows) != len(ARTIFACT_FILES)
    ):
        reasons.append("MANIFEST_CONTRACT_INVALID")
        rows = rows if isinstance(rows, list) else []
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"path", "sha256", "bytes"}:
            reasons.append("MANIFEST_ROW_INVALID")
            continue
        path = output / str(row["path"])
        if (
            not path.is_file()
            or sha256_file(path) != row["sha256"]
            or path.stat().st_size != row["bytes"]
        ):
            reasons.append(f"ARTIFACT_MISMATCH:{row['path']}")
    if report.get("fixed_claims") != FIXED_CLAIMS:
        reasons.append("FIXED_CLAIMS_INVALID")
    if report.get("execution_claim") != "READY_FOR_RUNNER_IMPLEMENTATION_ONLY":
        reasons.append("EXECUTION_OVERCLAIM")
    if not recompute or reasons:
        return {"status": "PASS" if not reasons else "FAIL", "reasons": reasons}

    coordinator, relational, set_valued = config_triplet()
    expected_hashes = {
        "coordinator": sha256_file(DEFAULT_COORDINATOR),
        "relational": sha256_file(ROOT / EXPECTED_BRANCH_PATHS["relational"]),
        "set_valued": sha256_file(ROOT / EXPECTED_BRANCH_PATHS["set_valued"]),
    }
    if report.get("config_sha256") != expected_hashes:
        reasons.append("REPORT_CONFIG_HASH_MISMATCH")
    if report.get("source_commit") != coordinator.get("source_commit"):
        reasons.append("REPORT_SOURCE_COMMIT_MISMATCH")
    numeric_status = report.get("fixed_depth_conformance", {}).get("status")
    expected_predicates = evaluate_predicates(
        coordinator, relational, set_valued, str(numeric_status)
    )
    expected_fixtures = run_fixtures(set_valued["target_shuffle"]["fixture_sha256"])
    expected_mutations = run_mutation_suite(coordinator, relational, set_valued)
    if report.get("predicates") != expected_predicates:
        reasons.append("REPORT_PREDICATES_NOT_RECOMPOSED")
    if report.get("fixtures") != expected_fixtures:
        reasons.append("REPORT_FIXTURES_NOT_RECOMPOSED")
    if report.get("mutation_suite") != expected_mutations:
        reasons.append("REPORT_MUTATIONS_NOT_RECOMPOSED")
    counts = {
        "pass": sum(row["status"] == "PASS" for row in expected_predicates.values()),
        "fail": sum(row["status"] == "FAIL" for row in expected_predicates.values()),
        "total": len(expected_predicates),
    }
    if report.get("predicate_counts") != counts:
        reasons.append("REPORT_PREDICATE_COUNTS_INVALID")
    if report.get("design_state") != adjudicate_design_state(
        expected_predicates, expected_fixtures
    ):
        reasons.append("REPORT_DESIGN_STATE_INVALID")
    try:
        with np.load(output / "fixed_depth_raw.npz", allow_pickle=False) as raw:
            if set(raw.files) != {
                "state_id",
                "fixed_error",
                "canonical_rmse",
                "canonical_converged",
                "canonical_iterations",
            }:
                reasons.append("RAW_STATE_ROSTER_INVALID")
            else:
                summary = report["fixed_depth_conformance"]
                states = len(raw["state_id"])
                converged = np.asarray(raw["canonical_converged"], dtype=bool)
                canonical = np.asarray(raw["canonical_rmse"], dtype=np.float64)
                if any(len(raw[name]) != states for name in raw.files):
                    reasons.append("RAW_STATE_LENGTH_MISMATCH")
                if summary.get("states") != states:
                    reasons.append("RAW_STATE_COUNT_MISMATCH")
                if summary.get("canonical_converged") != int(converged.sum()):
                    reasons.append("RAW_CONVERGENCE_COUNT_MISMATCH")
                if summary.get("canonical_failed") != int((~converged).sum()):
                    reasons.append("RAW_FAILURE_COUNT_MISMATCH")
                if not np.isclose(
                    summary.get("max_torch_numpy_error", np.nan),
                    float(np.max(raw["fixed_error"])),
                    rtol=0,
                    atol=0,
                ):
                    reasons.append("RAW_FIXED_ERROR_MISMATCH")
                if converged.any() and not np.isclose(
                    summary.get("canonical_max_rmse", np.nan),
                    float(np.max(canonical[converged])),
                    rtol=0,
                    atol=0,
                ):
                    reasons.append("RAW_CANONICAL_ERROR_MISMATCH")
    except (OSError, KeyError, ValueError) as exc:
        reasons.append(f"RAW_STATE_INVALID:{type(exc).__name__}")
    return {"status": "PASS" if not reasons else "FAIL", "reasons": reasons}


def check_root_artifact(output: Path) -> dict[str, Any]:
    reasons: list[str] = []
    expected_roster = {
        "run_a",
        "run_b",
        "manifest.json",
        "replay_comparison.json",
        "runtime_observation.json",
    }
    actual_roster = {path.name for path in output.iterdir()} if output.is_dir() else set()
    if actual_roster != expected_roster:
        reasons.append("ROOT_ROSTER_MISMATCH")
    try:
        manifest = load_json(output / "manifest.json")
        replay = load_json(output / "replay_comparison.json")
        runtime = load_json(output / "runtime_observation.json")
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        return {"status": "FAIL", "reasons": [*reasons, f"ROOT_UNREADABLE:{type(exc).__name__}"]}
    expected_paths = ["replay_comparison.json", "run_a/manifest.json", "run_b/manifest.json"]
    rows = manifest.get("scientific")
    if (
        manifest.get("schema_version") != "proportional-dual-native-root-manifest-v1"
        or manifest.get("fixed_claims") != FIXED_CLAIMS
        or manifest.get("runtime_observation") != "runtime_observation.json"
        or not isinstance(rows, list)
        or [row.get("path") for row in rows if isinstance(row, dict)] != expected_paths
        or len(rows) != len(expected_paths)
    ):
        reasons.append("ROOT_MANIFEST_CONTRACT_INVALID")
        rows = rows if isinstance(rows, list) else []
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"path", "sha256", "bytes"}:
            reasons.append("ROOT_MANIFEST_ROW_INVALID")
            continue
        path = output / str(row["path"])
        if not path.is_file() or sha256_file(path) != row["sha256"] or path.stat().st_size != row["bytes"]:
            reasons.append(f"ROOT_ARTIFACT_MISMATCH:{row['path']}")
    for name in ("run_a", "run_b"):
        checked = check_artifact(output / name)
        reasons.extend(f"{name}:{reason}" for reason in checked["reasons"])
    if replay.get("status") != "PASS" or replay.get("fixed_claims") != FIXED_CLAIMS:
        reasons.append("REPLAY_CLAIMS_INVALID")
    replay_rows = replay.get("scientific_files", [])
    expected_replay_paths = ARTIFACT_FILES + ["manifest.json"]
    if [row.get("path") for row in replay_rows] != expected_replay_paths:
        reasons.append("REPLAY_ROSTER_INVALID")
    else:
        for row in replay_rows:
            name = row["path"]
            left = output / "run_a" / name
            right = output / "run_b" / name
            if (
                row.get("byte_exact") is not True
                or row.get("left_sha256") != sha256_file(left)
                or row.get("right_sha256") != sha256_file(right)
                or left.read_bytes() != right.read_bytes()
            ):
                reasons.append(f"REPLAY_MISMATCH:{name}")
    if runtime.get("gpu_used_or_queried") is not False or runtime.get(
        "cuda_visible_devices"
    ) != "":
        reasons.append("RUNTIME_GPU_CLAIM_INVALID")
    return {"status": "PASS" if not reasons else "FAIL", "reasons": reasons}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_COORDINATOR)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check-artifact", type=Path)
    parser.add_argument("--check-root", type=Path)
    args = parser.parse_args()
    if args.check_root:
        checked = check_root_artifact(args.check_root)
        print(json.dumps(checked, sort_keys=True))
        return 0 if checked["status"] == "PASS" else 1
    if args.check_artifact:
        checked = check_artifact(args.check_artifact)
        print(json.dumps(checked, sort_keys=True))
        return 0 if checked["status"] == "PASS" else 1
    coordinator, relational, set_valued = config_triplet(args.config)
    report, raw = scientific_payload(coordinator, relational, set_valued)
    if args.output:
        write_artifact(args.output, report, raw)
    print(
        json.dumps(
            {
                "design_state": report["design_state"],
                "predicate_counts": report["predicate_counts"],
                "fixed_depth": report["fixed_depth_conformance"]["status"],
                "mutations": report["mutation_suite"]["status"],
            },
            sort_keys=True,
        )
    )
    return 0 if report["predicate_counts"]["fail"] == 0 and report["mutation_suite"]["status"] == "PASS" and all(row["status"] == "PASS" for row in report["fixtures"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())
