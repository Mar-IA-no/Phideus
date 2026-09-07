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


def test_real_test_only_mutations_cover_every_predicate_and_reason(tmp_path: Path) -> None:
    fixture_source = ROOT / "tests/fixtures/proportional_mapping_private_invariance/a/fixture_manifest.json"
    fixture_hash = "32d9ce95ea083c549eccb9f017cae2bdc38e084796f064496b528e55356a1b01"
    cases = []
    for index, (identifier, reasons) in enumerate(CHECKER.REASONS.items()):
        for reason in reasons:
            root = tmp_path / f"case_{index}_{reason.lower()}"
            fixture = root / "fixture"
            fixture.mkdir(parents=True)
            shutil.copyfile(fixture_source, fixture / "fixture_manifest.json")
            prepared = root / "prepared"
            PREPARE.prepare_test_fixture(fixture, prepared, fixture_hash)
            built = root / "built"
            BUILDER.build_test_fixture(prepared / "public", built)
            candidate_path = built / "test_candidate.json"
            candidate = json.loads(candidate_path.read_text())
            candidate["predicate_contract"] = {name: name != reason for name in reasons}
            candidate["mutation_case"] = {"id": identifier, "reason": reason}
            PREPARE.write_json(candidate_path, candidate)
            cases.append({"id": identifier, "reason": reason, "candidate_path": str(candidate_path)})
    suite = tmp_path / "mutation_suite.json"
    result_path = tmp_path / "mutation_results.json"
    PREPARE.write_json(suite, {"schema_version": "mapping-test-only-mutation-suite-v1", "cases": cases})
    subprocess.run(
        [sys.executable, str(EXP / "check_proportional_mapping_feasibility.py"), "--test-mutation-suite", str(suite), "--test-output", str(result_path)],
        cwd=ROOT,
        check=True,
    )
    result = json.loads(result_path.read_text())
    assert result["status"] == "PASS"
    assert len(result["predicate_mutations"]) == sum(len(reasons) for reasons in CHECKER.REASONS.values())
    assert {row["id"] for row in result["predicate_mutations"]} == set(CHECKER.REASONS)
    assert all(row["status"] == "REJECTED" and row["observed_reason_codes"] == [row["reason"]] for row in result["predicate_mutations"])
    assert all("mapping_decision" not in row for row in result["predicate_mutations"])
    for variable in ("MAPPING_FEASIBILITY_RUN_A", "MAPPING_FEASIBILITY_RUN_B"):
        run_path = os.environ.get(variable)
        if run_path:
            PREPARE.write_json(Path(run_path) / "mutation_results.json", result)


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


def test_checker_rejects_each_candidate_corruption_class_and_fake_replay(tmp_path: Path) -> None:
    run_a_value = os.environ.get("MAPPING_FEASIBILITY_RUN_A")
    run_b_value = os.environ.get("MAPPING_FEASIBILITY_RUN_B")
    if not run_a_value or not run_b_value:
        pytest.skip("requires completed paired runner surfaces")
    results = {}

    def rewrite_candidate_hash(run: Path) -> None:
        candidate_path = run / "mapping_candidate.json"
        evidence_path = run / "evaluation_evidence.json"
        evidence = json.loads(evidence_path.read_text())
        evidence["candidate_sha256"] = hashlib.sha256(candidate_path.read_bytes()).hexdigest()
        PREPARE.write_json(evidence_path, evidence)

    def rewrite_private_manifest(run: Path, relative: str) -> None:
        root = run / "prepared/private_dev"
        manifest_path = root / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        target = root / relative
        manifest["files"][relative]["bytes"] = target.stat().st_size
        manifest["files"][relative]["sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
        PREPARE.write_json(manifest_path, manifest)

    for corruption in ("hash", "keyset", "shape", "join", "predicate", "decision", "synthetic_renaming"):
        parent = tmp_path / corruption
        shutil.copytree(Path(run_a_value), parent / "run_a")
        shutil.copytree(Path(run_b_value), parent / "run_b")
        run = parent / "run_a"
        candidate_path = run / "mapping_candidate.json"
        evidence_path = run / "evaluation_evidence.json"
        candidate = json.loads(candidate_path.read_text())
        evidence = json.loads(evidence_path.read_text())
        if corruption == "hash":
            evidence["candidate_sha256"] = "0" * 64
            PREPARE.write_json(evidence_path, evidence)
        elif corruption == "keyset":
            candidate.pop("lines")
            PREPARE.write_json(candidate_path, candidate)
            rewrite_candidate_hash(run)
        elif corruption == "shape":
            candidate["public_facts"]["w54_shapes"]["ensemble_logits"]["shape"] = [1, 4]
            PREPARE.write_json(candidate_path, candidate)
            rewrite_candidate_hash(run)
        elif corruption == "join":
            path = run / "prepared/private_dev/w54/unit_key.npy"
            values = np.load(path, allow_pickle=False).astype(str)
            values[0] = "0" * 64
            with path.open("wb") as handle:
                np.lib.format.write_array(handle, values, allow_pickle=False)
            rewrite_private_manifest(run, "w54/unit_key.npy")
        elif corruption == "predicate":
            evidence["common_mapping_observations"]["candidate_query_sha256"] = "0" * 64
            PREPARE.write_json(evidence_path, evidence)
        elif corruption == "decision":
            candidate["mapping_decision"] = "COMMON_FACTORIAL_FEASIBLE"
            PREPARE.write_json(candidate_path, candidate)
            rewrite_candidate_hash(run)
        else:
            candidate["unit_namespaces"] = {"eiv": "renamed", "set_valued": "renamed", "relational": "renamed"}
            candidate["declared_cross_domain_unit_bridge"] = {"kind": "renaming_only"}
            PREPARE.write_json(candidate_path, candidate)
            rewrite_candidate_hash(run)
            evidence = json.loads(evidence_path.read_text())
            evidence["common_mapping_observations"]["candidate_unit_namespaces_sha256"] = hashlib.sha256(
                json.dumps(candidate["unit_namespaces"], sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            evidence["common_mapping_observations"]["candidate_bridge"] = candidate["declared_cross_domain_unit_bridge"]
            PREPARE.write_json(evidence_path, evidence)
        completed = subprocess.run(
            [sys.executable, str(EXP / "check_proportional_mapping_feasibility.py"), "--run", str(run), "--phase", "pre"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            env={**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"},
        )
        adjudication = json.loads((run / "pre_adjudication.json").read_text())
        assert completed.returncode == 0
        assert adjudication["mapping_decision"] is None
        if corruption == "synthetic_renaming":
            m1 = next(row for row in adjudication["predicates"] if row["id"] == "M1_QUERY_UNIT")
            assert adjudication["technical_status"]["artifact_status"] == "PASS"
            assert m1["status"] == "FAIL" and m1["reason_codes"] == ["SYNTHETIC_ID_EQUIVALENCE"]
        else:
            assert adjudication["technical_status"]["artifact_status"] == "FAIL"
            results[corruption] = "REJECTED"
    fake_replay = tmp_path / "hash/run_a/fake_replay.json"
    PREPARE.write_json(fake_replay, {"schema_version": "proportional-mapping-replay-evidence-v1", "status": "PASS"})
    assert CHECKER.validate_replay(tmp_path / "hash/run_a", fake_replay) is False
    for variable in ("MAPPING_FEASIBILITY_RUN_A", "MAPPING_FEASIBILITY_RUN_B"):
        run_path = os.environ.get(variable)
        if run_path:
            receipt_path = Path(run_path) / "mutation_results.json"
            receipt = json.loads(receipt_path.read_text())
            receipt["candidate_corruptions"] = results
            receipt["m1_synthetic_renaming"] = {"status": "REJECTED", "reason_code": "SYNTHETIC_ID_EQUIVALENCE"}
            PREPARE.write_json(receipt_path, receipt)


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
