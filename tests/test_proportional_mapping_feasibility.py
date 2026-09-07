from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments/geometria_proporcional"


def load_script(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, EXP / filename)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


PREPARE = load_script("mapping_prepare_test", "prepare_proportional_mapping_feasibility.py")
BUILDER = load_script("mapping_builder_test", "build_proportional_mapping_candidate.py")
EVALUATOR = load_script("mapping_evaluator_test", "evaluate_proportional_mapping_feasibility.py")
CHECKER = load_script("mapping_checker_test", "check_proportional_mapping_feasibility.py")
RUNNER = load_script("mapping_runner_test", "run_proportional_mapping_feasibility.py")


def test_frozen_config_binds_plan_and_29_sources() -> None:
    config = json.loads((EXP / "configs/proportional_mapping_feasibility_v1.json").read_text())
    assert len(config["source_bindings"]) == 29
    assert hashlib.sha256((ROOT / config["plan"]["path"]).read_bytes()).hexdigest() == config["plan"]["sha256"]
    for _, relative, expected in config["source_bindings"]:
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected
    assert config["fixed_claims"] == {
        "gpu_used_or_queried": False,
        "architecture_promoted": False,
        "scientific_decision": None,
        "decision_authority": "user",
    }
    assert config["controls"]["solver_replay_atol"] == 1e-8


def test_total_adjudication_algebra_exercises_three_leaves() -> None:
    config = json.loads((EXP / "configs/proportional_mapping_feasibility_v1.json").read_text())
    all_ids = sum((config["predicates"][key] for key in ("mapping", "relational", "set_valued")), [])
    technical = {"source_status": "PASS", "artifact_status": "PASS", "checker_status": "PASS", "replay_status": "PASS"}

    def rows(failed=()):
        return [CHECKER.pred(name, name not in failed, ["test"]) for name in all_ids]

    assert CHECKER.semantic_decision(rows(), technical) == "COMMON_FACTORIAL_FEASIBLE"
    assert CHECKER.semantic_decision(rows({config["predicates"]["mapping"][0]}), technical) == "BIFURCATE_NATIVE_CONTRASTS"
    assert CHECKER.semantic_decision(rows({config["predicates"]["mapping"][0], config["predicates"]["relational"][0]}), technical) == "NO_EXECUTABLE_SUCCESSOR"
    assert CHECKER.semantic_decision(rows(), {**technical, "checker_status": "FAIL"}) is None


def test_map_baseline_features_do_not_reuse_threshold_hard() -> None:
    logits = np.asarray([[0.2, -0.1, -2.0, -2.0]], dtype=np.float64)
    per_seed = np.repeat(logits[None], 3, axis=0)
    mass = np.zeros((1, 15), dtype=np.float64)
    mass[0, 2] = 0.7  # binary 0011: MAP set contains families 0 and 1
    mass[0, 0] = 0.3
    utility = np.asarray([[1.0, 0.6, 0.2, -0.2]], dtype=np.float64)
    map_set, hard = EVALUATOR.hard_map_actions(mass, utility)
    risk, candidate = EVALUATOR.posterior_risk(mass, utility, 1.25)
    design = EVALUATOR.adapted_design(logits, per_seed, mass, risk, hard, candidate, map_set, utility)
    threshold_hard = EVALUATOR.expit(logits) >= 0.5
    assert not np.array_equal(map_set, threshold_hard)
    assert design.shape == (1, 1, 17)
    assert design[0, 0, 7] == 2.0
    assert design[0, 0, 10] == 0.7
    assert np.array_equal(design, CHECKER.design17(logits, per_seed, mass, risk, hard, candidate, map_set, utility))


def test_private_fixture_variants_leave_public_candidate_byte_exact(tmp_path: Path) -> None:
    fixture_root = ROOT / "tests/fixtures/proportional_mapping_private_invariance"
    expected = {
        "a": "32d9ce95ea083c549eccb9f017cae2bdc38e084796f064496b528e55356a1b01",
        "b": "dac2f473176ce804b37bc5d66bac6ee903336ec7445bf0ada00549d6d9b46a16",
        "w49": "1d974d93de8d70205de35ffb608939830850e23c4886ce2159ea8b199fe771c7",
        "set_target": "038d9b4a86241796fa5f99dc52ba4796574bef2bd820d06c7d73be49cb251dd7",
        "graph_x": "fdd7afd21fcecdf6e572a8ea10c50d8b8c7233764c55e00a393682f76aab3ddd",
        "graph_clean_removed": "7901f06ec66ffa9a04e7aab4d15e544ff8aff1b11c3f93af1ac1a2f1de82f930",
        "graph_mechanism": "a75e63000746a9afb336cfca5a2e5c7a1765a9ea2b772509841f093844bda4ad",
    }
    candidates = []
    public_payloads = []
    for variant in expected:
        copied = tmp_path / f"fixture_{variant}"
        copied.mkdir()
        shutil.copyfile(fixture_root / variant / "fixture_manifest.json", copied / "fixture_manifest.json")
        prepared = tmp_path / f"prepared_{variant}"
        PREPARE.prepare_test_fixture(copied, prepared, expected[variant])
        candidate = tmp_path / f"candidate_{variant}"
        BUILDER.build_test_fixture(prepared / "public", candidate)
        candidates.append((candidate / "test_candidate.json").read_bytes())
        public_payloads.append((prepared / "public/payload.json").read_bytes())
        receipt = json.loads((prepared / "test_receipt.json").read_text())
        assert receipt["fixture_mode"] == "TEST_ONLY"
        assert "mapping_decision" not in receipt
    assert len(set(public_payloads)) == 1
    assert len(set(candidates)) == 1


def test_test_only_mode_rejects_non_temporary_roots() -> None:
    fixture = ROOT / "tests/fixtures/proportional_mapping_private_invariance/a"
    with pytest.raises(ValueError, match="temporary"):
        PREPARE.prepare_test_fixture(fixture, ROOT / "data/not-allowed-test-fixture", "irrelevant")


def test_relational_shuffle_is_master_level_and_edge_total() -> None:
    keys = np.asarray(["m0", "m1"])
    mapping, singletons = EVALUATOR.deranged_indices(keys, [("test", 3), ("test", 3)], 53601)
    assert singletons == 0
    assert mapping.tolist() == [1, 0]
    donor = np.asarray([-1.0, 0.25, 0.75])
    edges_a = np.asarray([[0, 1], [1, 2]], dtype=np.int64)
    edges_b = np.asarray([[0, 1], [0, 2], [1, 2]], dtype=np.int64)
    target_a = EVALUATOR.incidence_matrix(3, edges_a) @ donor
    target_b = EVALUATOR.incidence_matrix(3, edges_b) @ donor
    assert target_a.shape == (2,)
    assert target_b.shape == (3,)


def test_checker_is_not_coupled_to_builder_or_evaluator_modules() -> None:
    source = (EXP / "check_proportional_mapping_feasibility.py").read_text()
    assert "import build_proportional_mapping_candidate" not in source
    assert "import evaluate_proportional_mapping_feasibility" not in source
    assert "from geometria_proporcional" not in source


def test_mutation_suite_covers_each_predicate_and_reason() -> None:
    config = json.loads((EXP / "configs/proportional_mapping_feasibility_v1.json").read_text())
    result = CHECKER.mutation_suite(config)
    rows = result["predicate_mutations"]
    assert len(rows) == 17
    assert all(row["status"] == "PASS" and len(row["observed_reason"]) == 1 for row in rows)
    assert result["leaf_tests"] == {
        "LEAF_COMMON": "COMMON_FACTORIAL_FEASIBLE",
        "LEAF_BIFURCATE": "BIFURCATE_NATIVE_CONTRASTS",
        "LEAF_NONE": "NO_EXECUTABLE_SUCCESSOR",
    }
    assert result["production_source_tamper"]["decision"] is None


def test_each_closed_reason_branch_is_serialized_and_invalid_reason_rejected() -> None:
    for identifier, reasons in CHECKER.REASONS.items():
        for reason in reasons:
            row = CHECKER.pred(identifier, False, {"mutation": reason}, [reason])
            assert row["status"] == "FAIL"
            assert row["reason_codes"] == [reason]
            assert set(row["evidence"][0]) == {"source_id", "locator", "observed", "digest"}
        with pytest.raises(ValueError, match="closed catalog"):
            CHECKER.pred(identifier, False, {"mutation": "invalid"}, ["NOT_A_REASON"])


def test_manifest_rejects_unlisted_and_relisted_extra_file(tmp_path: Path) -> None:
    run_path = os.environ.get("MAPPING_FEASIBILITY_RUN_A")
    if not run_path:
        pytest.skip("requires a completed runner development surface")
    original = Path(run_path) / "prepared/public"
    copied = tmp_path / "public"
    shutil.copytree(original, copied)
    np.save(copied / "truth.npy", np.asarray([1.0]))
    assert CHECKER.verify_tree_manifest(copied, "mapping-prepared-public-manifest-v1") is False
    manifest = json.loads((copied / "manifest.json").read_text())
    extra = copied / "truth.npy"
    manifest["files"]["truth.npy"] = {"bytes": extra.stat().st_size, "sha256": hashlib.sha256(extra.read_bytes()).hexdigest(), "dtype": "<f8", "shape": [1]}
    manifest["pathset"] = sorted(manifest["files"])
    manifest["pathset_sha256"] = hashlib.sha256("\n".join(manifest["pathset"]).encode()).hexdigest()
    PREPARE.write_json(copied / "manifest.json", manifest)
    assert CHECKER.verify_tree_manifest(copied, "mapping-prepared-public-manifest-v1") is False


def test_checker_recomputes_forged_common_claim_and_rejects_fake_replay(tmp_path: Path) -> None:
    run_a_value = os.environ.get("MAPPING_FEASIBILITY_RUN_A")
    run_b_value = os.environ.get("MAPPING_FEASIBILITY_RUN_B")
    if not run_a_value or not run_b_value:
        pytest.skip("requires completed paired runner surfaces")
    forged_parent = tmp_path / "paired"
    shutil.copytree(Path(run_a_value), forged_parent / "run_a")
    shutil.copytree(Path(run_b_value), forged_parent / "run_b")
    forged = forged_parent / "run_a"
    candidate_path = forged / "mapping_candidate.json"
    candidate = json.loads(candidate_path.read_text())
    candidate["query"] = "forged query"
    candidate["unit_namespaces"] = {"eiv": "fake", "set_valued": "fake", "relational": "fake"}
    candidate["declared_cross_domain_unit_bridge"] = {"kind": "renaming_only"}
    candidate["mapping_decision"] = "COMMON_FACTORIAL_FEASIBLE"
    PREPARE.write_json(candidate_path, candidate)
    evidence_path = forged / "evaluation_evidence.json"
    evidence = json.loads(evidence_path.read_text())
    evidence["candidate_sha256"] = hashlib.sha256(candidate_path.read_bytes()).hexdigest()
    evidence["common_mapping_observations"] = {"all_mapping_predicates": True}
    PREPARE.write_json(evidence_path, evidence)
    completed = subprocess.run(
        [sys.executable, str(EXP / "check_proportional_mapping_feasibility.py"), "--run", str(forged), "--phase", "pre"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    adjudication = json.loads((forged / "pre_adjudication.json").read_text())
    statuses = {row["id"]: row for row in adjudication["predicates"]}
    assert statuses["M1_QUERY_UNIT"]["status"] == "FAIL"
    assert "QUERY_MISMATCH" in statuses["M1_QUERY_UNIT"]["reason_codes"]
    assert adjudication["technical_status"]["artifact_status"] == "FAIL"
    fake_replay = forged / "fake_replay.json"
    PREPARE.write_json(fake_replay, {"schema_version": "proportional-mapping-replay-evidence-v1", "status": "PASS"})
    assert CHECKER.validate_replay(forged, fake_replay) is False


def test_production_source_tamper_is_checked_from_bytes() -> None:
    config = json.loads((EXP / "configs/proportional_mapping_feasibility_v1.json").read_text())
    config["source_bindings"][0][2] = "0" * 64
    passed, receipts = CHECKER.source_status(config)
    assert passed is False
    assert next(row for row in receipts if row["id"] == "SYNTHESIS")["status"] == "FAIL"


def test_budget_signal_and_invalidation_force_null_decision(tmp_path: Path) -> None:
    with pytest.raises(RUNNER.BudgetExceeded):
        RUNNER.run_command(
            [sys.executable, "-c", "raise MemoryError('synthetic resource exhaustion')"],
            time.monotonic() + 10.0,
            512 * 1024 * 1024,
            dict(os.environ),
        )
    output = tmp_path / "failed"
    for label in ("run_a", "run_b"):
        run = output / label
        run.mkdir(parents=True)
        PREPARE.write_json(run / "adjudication.json", {"mapping_decision": "COMMON_FACTORIAL_FEASIBLE"})
    RUNNER.invalidate_outputs(output, "BUDGET_EXCEEDED", "synthetic", "NOT_RUN")
    for label in ("run_a", "run_b"):
        adjudication = json.loads((output / label / "adjudication.json").read_text())
        assert adjudication["mapping_decision"] is None
        assert adjudication["technical_status"]["artifact_status"] == "BUDGET_EXCEEDED"
