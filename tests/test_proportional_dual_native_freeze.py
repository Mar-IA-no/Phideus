from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments/geometria_proporcional/check_proportional_dual_native_freeze.py"
SPEC = importlib.util.spec_from_file_location("proportional_dual_native_checker_test", SCRIPT)
assert SPEC and SPEC.loader
CHECKER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CHECKER
SPEC.loader.exec_module(CHECKER)


def test_structural_positive_fixture_passes_all_predicates() -> None:
    coordinator, relational, set_valued = CHECKER.config_triplet()
    rows = CHECKER.evaluate_predicates(coordinator, relational, set_valued, "PASS")
    assert len(rows) == 30
    assert all(row["status"] == "PASS" for row in rows.values())


def test_every_predicate_has_a_caught_negative_mutation() -> None:
    coordinator, relational, set_valued = CHECKER.config_triplet()
    result = CHECKER.run_mutation_suite(coordinator, relational, set_valued)
    assert result["status"] == "PASS"
    assert result["caught"] == result["total"]
    assert result["total"] >= 60
    assert {row["predicate"] for row in result["rows"]} == set(CHECKER.ALL_PREDICATES)
    required_cases = {
        "C1_UNRELATED_EXISTING_COMMIT",
        "C5_GPU_ALLOWED_FIELD",
        "C5_GO_NOGO_FIELD",
        "R1_NORMALIZED_IRLS_CONFUSED_WITH_RAW",
        "R7_PATH_CONTROL_NO_COMMON_ROSTER",
        "S1_MONITOR_APPLY_READS_TARGET",
        "S6_DIFFERENT_RECIPE_BY_POSTERIOR",
        "S8_CROSS_FOLD_TARGET_SHUFFLE",
        "S8_TARGET_SHUFFLE_IDENTITY",
        "S8_NONCANONICAL_TARGET_MAP",
        "S9_MATCHING_READS_TARGET",
        "S9_SUPPORT_UNION",
        "R5_TOLERANCE_CHANGED",
        "S4_MARGINAL_SOLVER_CHANGED",
        "S5_UTILITY_TIE_CHANGED",
        "S7_RIDGE_ALPHA_CHANGED",
        "S7_SELECTION_KEY_CHANGED",
        "S1_FRESHNESS_DISABLED",
        "C6_SET_COST_MISCLASSIFIED",
        "C1_RELATIONAL_SMOKE_CONFIG_MISSING",
        "C1_SET_W49_SCHEMA_MISSING",
        "C1_SET_W59_MANIFEST_MISSING",
    }
    assert required_cases <= {row.get("case_id") for row in result["rows"]}


def test_target_derangement_fixture_is_byte_fixed_and_stratum_safe() -> None:
    _, _, set_valued = CHECKER.config_triplet()
    fixtures = CHECKER.run_fixtures(set_valued["target_shuffle"]["fixture_sha256"])
    assert fixtures["target_derangement"] == {
        "status": "PASS",
        "sha256": "d7aa2f128b6d42dbe7448415dcd8d4d69ca0ad8311394a5b1209ca9579e03904",
        "rows": 6,
        "permutable_fraction": 5 / 6,
        "singleton_count": 1,
    }


def test_matched_control_fixture_uses_five_way_intersection() -> None:
    fixture = CHECKER.run_fixtures()["matched_common_support"]
    assert fixture["status"] == "PASS"
    assert fixture["true_override_tokens"] == 5
    assert fixture["common_tokens"] == 4
    assert np.isclose(fixture["coverage"], 0.8)
    assert fixture["uses_intersection"] is True


def test_material_fixtures_cover_posterior_and_every_private_path_field(
    monkeypatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    fixtures = CHECKER.run_fixtures()
    posterior = fixtures["posterior_hard_map"]
    assert posterior["status"] == "PASS"
    assert posterior["recipes_diverge"] is True
    assert np.isclose(posterior["posterior_sum"], 1.0)
    assert posterior["utility_tie_candidates"] == [0, 1]
    assert posterior["selected_family"] == 0
    assert posterior["utility_tie_rule"] == "lowest_family_index"
    path = fixtures["path_private_invariance"]
    assert path["status"] == "PASS"
    assert path["eligible"] is True
    assert path["path_tensors_byte_identical"] is True
    assert {"master_id", "corruption_mechanism", "seed"} <= set(
        path["mutated_fields"]
    )
    assert path["mutated_field_count"] == 12


def test_k64_base_weighted_design_diagnostic_is_rejected(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    _, relational, _ = CHECKER.config_triplet()
    relational["fixed_depth_conformance"]["steps"] = 64
    relational["fixed_depth_conformance"]["graphs"] = 32
    relational["fixed_depth_conformance"]["seed"] = 2026090723
    relational["fixed_depth_conformance"]["gradient_probe_states"] = 16
    summary, raw = CHECKER.run_fixed_depth_conformance(relational)
    assert summary["status"] == "FAIL"
    assert summary["max_torch_numpy_error"] <= 1e-9
    assert summary["canonical_max_rmse"] > relational["fixed_depth_conformance"][
        "canonical_max_rmse"
    ]
    assert len(raw["state_id"]) == 96


def test_k192_fresh_confirmation_rejects_relational_freeze(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    _, relational, _ = CHECKER.config_triplet()
    summary, raw = CHECKER.run_fixed_depth_conformance(relational)
    assert summary["status"] == "FAIL"
    assert summary["steps"] == 192
    assert summary["graphs"] == 64
    assert summary["node_range_covered"] is True
    assert summary["node_counts"] == list(range(8, 17))
    assert summary["max_torch_numpy_error"] <= 1e-9
    assert summary["canonical_failed"] > 0
    assert summary["relation_gradient"]["sign_inversions"] == 0
    assert summary["weight_gradient"]["sign_inversions"] == 0
    assert len(raw["state_id"]) == 192


def test_artifact_manifest_detects_tampering(tmp_path: Path) -> None:
    report = {
        "execution_claim": "READY_FOR_RUNNER_IMPLEMENTATION_ONLY",
        "fixed_claims": CHECKER.FIXED_CLAIMS,
        "fixtures": {},
        "mutation_suite": {},
    }
    raw = {"value": np.asarray([1.0, 2.0], dtype=np.float64)}
    output = tmp_path / "artifact"
    CHECKER.write_artifact(output, report, raw)
    assert CHECKER.check_artifact(output, recompute=False)["status"] == "PASS"
    (output / "fixtures.json").write_text('{"tampered":true}\n', encoding="utf-8")
    checked = CHECKER.check_artifact(output, recompute=False)
    assert checked["status"] == "FAIL"
    assert "ARTIFACT_MISMATCH:fixtures.json" in checked["reasons"]


def test_artifact_manifest_requires_exact_roster(tmp_path: Path) -> None:
    report = {
        "execution_claim": "READY_FOR_RUNNER_IMPLEMENTATION_ONLY",
        "fixed_claims": CHECKER.FIXED_CLAIMS,
        "fixtures": {},
        "mutation_suite": {},
    }
    output = tmp_path / "artifact"
    CHECKER.write_artifact(output, report, {"value": np.asarray([1.0])})
    manifest_path = output / "manifest.json"
    manifest = CHECKER.load_json(manifest_path)
    manifest["files"].pop(0)
    CHECKER.write_json(manifest_path, manifest)
    checked = CHECKER.check_artifact(output, recompute=False)
    assert checked["status"] == "FAIL"
    assert "MANIFEST_CONTRACT_INVALID" in checked["reasons"]
    (output / "unexpected.json").write_text("{}\n", encoding="utf-8")
    checked = CHECKER.check_artifact(output, recompute=False)
    assert checked["status"] == "FAIL"
    assert any(reason.startswith("ARTIFACT_ROSTER_MISMATCH") for reason in checked["reasons"])


def test_artifact_checker_recomposes_report_from_bound_configs(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    coordinator, relational, set_valued = CHECKER.config_triplet()
    report, raw = CHECKER.scientific_payload(coordinator, relational, set_valued)
    output = tmp_path / "artifact"
    CHECKER.write_artifact(output, report, raw)
    assert CHECKER.check_artifact(output)["status"] == "PASS"

    report_path = output / "scientific_report.json"
    tampered = CHECKER.load_json(report_path)
    tampered["predicates"]["S9_MATCHED_CONTROL_TARGET_BLIND"] = {
        "status": "FAIL",
        "reasons": ["forged"],
    }
    CHECKER.write_json(report_path, tampered)
    manifest_path = output / "manifest.json"
    manifest = CHECKER.load_json(manifest_path)
    for row in manifest["files"]:
        if row["path"] == "scientific_report.json":
            row["sha256"] = CHECKER.sha256_file(report_path)
            row["bytes"] = report_path.stat().st_size
    CHECKER.write_json(manifest_path, manifest)
    checked = CHECKER.check_artifact(output)
    assert checked["status"] == "FAIL"
    assert "REPORT_PREDICATES_NOT_RECOMPOSED" in checked["reasons"]


def test_artifact_checker_derives_k192_failure_from_raw_state(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    coordinator, relational, set_valued = CHECKER.config_triplet()
    report, raw = CHECKER.scientific_payload(coordinator, relational, set_valued)
    output = tmp_path / "artifact"
    CHECKER.write_artifact(output, report, raw)

    report_path = output / "scientific_report.json"
    forged = CHECKER.load_json(report_path)
    forged["fixed_depth_conformance"]["status"] = "PASS"
    forged["predicates"]["R11_BASE_WEIGHTED_K192_CONFORMANCE"] = {
        "status": "PASS",
        "reasons": [],
    }
    forged["predicate_counts"] = {"pass": 30, "fail": 0, "total": 30}
    forged["design_state"] = "BOTH_DESIGN_FREEZES_VALID"
    CHECKER.write_json(report_path, forged)
    manifest_path = output / "manifest.json"
    manifest = CHECKER.load_json(manifest_path)
    for row in manifest["files"]:
        if row["path"] == "scientific_report.json":
            row["sha256"] = CHECKER.sha256_file(report_path)
            row["bytes"] = report_path.stat().st_size
    CHECKER.write_json(manifest_path, manifest)
    checked = CHECKER.check_artifact(output)
    assert checked["status"] == "FAIL"
    assert "REPORT_NUMERIC_STATUS_NOT_RECOMPOSED" in checked["reasons"]
    assert "REPORT_PREDICATES_NOT_RECOMPOSED" in checked["reasons"]
    assert "REPORT_DESIGN_STATE_INVALID" in checked["reasons"]
