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
    assert result["caught"] == result["total"] == 32
    assert {row["predicate"] for row in result["rows"]} == set(CHECKER.ALL_PREDICATES)


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
    assert CHECKER.check_artifact(output)["status"] == "PASS"
    (output / "fixtures.json").write_text('{"tampered":true}\n', encoding="utf-8")
    checked = CHECKER.check_artifact(output)
    assert checked["status"] == "FAIL"
    assert "ARTIFACT_MISMATCH:fixtures.json" in checked["reasons"]
