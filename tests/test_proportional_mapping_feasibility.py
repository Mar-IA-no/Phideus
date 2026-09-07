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


def test_common_bridge_requires_material_frozen_authority() -> None:
    config = json.loads((EXP / "configs/proportional_mapping_feasibility_v1.json").read_text())
    fabricated = {
        "kind": "authority_bijection",
        "authority_source_id": "SYNTHESIS",
        "total": True,
        "synthetic": False,
        "unit_count": 1,
        "eiv_count": 1,
        "set_valued_count": 1,
        "relational_count": 1,
        "bijection_sha256": "z" * 64,
    }
    valid, diagnostic = CHECKER.verify_common_unit_authority(config, fabricated)
    assert valid is False
    assert diagnostic == {"contract_declared": False, "materialized": False}


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


def test_real_mutations_execute_production_condition_functions(tmp_path: Path) -> None:
    rows = []

    def record(identifier: str, reason: str, passed: bool, reasons: list[str], observed: object, function: str, mutated_field: str) -> None:
        assert passed is False
        assert reasons == [reason]
        row = CHECKER.pred(identifier, passed, observed, reasons)
        rows.append({"id": identifier, "status": "REJECTED", "reason_codes": row["reason_codes"], "evidence": row["evidence"], "execution_function": function, "mutated_field": mutated_field, "mapping_decision": None})

    config = json.loads((EXP / "configs/proportional_mapping_feasibility_v1.json").read_text())
    triples = [{"eiv": "e0", "set_valued": "s0", "relational": "r0"}, {"eiv": "e1", "set_valued": "s1", "relational": "r1"}]
    authority_path = tmp_path / "common_unit_authority.json"
    PREPARE.write_json(authority_path, {"triples": triples})
    authority_sha = hashlib.sha256(authority_path.read_bytes()).hexdigest()
    digest = hashlib.sha256(json.dumps(triples, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()
    common_config = json.loads(json.dumps(config))
    common_config["source_bindings"].append(["COMMON_UNIT_AUTHORITY", str(authority_path), authority_sha])
    common_config["common_unit_authority"] = {"source_id": "COMMON_UNIT_AUTHORITY", "path": str(authority_path), "sha256": authority_sha}
    namespace = {name: "common" for name in ("eiv", "set_valued", "relational")}
    bridge = {"kind": "authority_bijection", "authority_source_id": "COMMON_UNIT_AUTHORITY", "total": True, "synthetic": False, "unit_count": 2, "eiv_count": 2, "set_valued_count": 2, "relational_count": 2, "bijection_sha256": digest}
    for reason, query, spaces, mutated_bridge, field in (
        ("QUERY_MISMATCH", "different", namespace, bridge, "query"),
        ("NO_COMMON_UNIT_NAMESPACE", config["query"], {"eiv": "a", "set_valued": "b", "relational": "c"}, bridge, "unit_namespaces"),
        ("UNIT_BIJECTION_INCOMPLETE", config["query"], namespace, {**bridge, "unit_count": 1}, "declared_cross_domain_unit_bridge.unit_count"),
        ("SYNTHETIC_ID_EQUIVALENCE", config["query"], namespace, {**bridge, "kind": "renaming_only"}, "declared_cross_domain_unit_bridge.kind"),
    ):
        passed, reasons, observed = CHECKER.common_unit_contract(common_config, query, config["query"], spaces, mutated_bridge)
        record("M1_QUERY_UNIT", reason, passed, reasons, observed, "common_unit_contract", field)

    common = {
        "lines": {name: {"observation": "same", "target_authority": "same", "output": "same"} for name in ("eiv", "set_valued", "relational")},
        "adapters": {name: ["same"] for name in ("eiv", "set_valued", "relational")},
        "declared_cross_domain_target_bridge": {"total_roundtrip_exact": True, "roundtrip_lossless": True, "learned": False, "monitor_used": False},
    }
    m2_cases = []
    mutated = json.loads(json.dumps(common)); mutated["lines"]["relational"]["observation"] = "different"; m2_cases.append(("OBSERVATION_SOURCE_MISMATCH", mutated, False, "lines.relational.observation"))
    mutated = json.loads(json.dumps(common)); mutated["declared_observation_projections"] = {"roundtrip_exact": False}; m2_cases.append(("PROJECTION_NOT_INVERTIBLE", mutated, False, "declared_observation_projections.roundtrip_exact"))
    mutated = json.loads(json.dumps(common)); [mutated["lines"][name].update({"information": "same"}) for name in mutated["lines"]]; mutated["lines"]["relational"]["information"] = "different"; m2_cases.append(("INFORMATION_ASYMMETRY", mutated, False, "lines.relational.information"))
    m2_cases.append(("PRIVATE_FIELD_EXPOSED", json.loads(json.dumps(common)), True, "prepared.public.private_field"))
    for reason, candidate, leak, field in m2_cases:
        passed, reasons, observed = CHECKER.common_observation_contract(candidate, leak)
        record("M2_OBSERVATION_PARITY", reason, passed, reasons, observed, "common_observation_contract", field)

    m3_cases = []
    mutated = json.loads(json.dumps(common)); mutated["lines"]["relational"]["target_authority"] = "different"; m3_cases.append(("TARGET_SCHEMA_MISMATCH", mutated, "lines.relational.target_authority"))
    mutated = json.loads(json.dumps(common)); mutated["declared_cross_domain_target_bridge"]["total_roundtrip_exact"] = False; m3_cases.append(("TARGET_MAP_PARTIAL", mutated, "declared_cross_domain_target_bridge.total_roundtrip_exact"))
    mutated = json.loads(json.dumps(common)); mutated["declared_cross_domain_target_bridge"]["roundtrip_lossless"] = False; m3_cases.append(("TARGET_ROUNDTRIP_LOSS", mutated, "declared_cross_domain_target_bridge.roundtrip_lossless"))
    mutated = json.loads(json.dumps(common)); mutated["declared_cross_domain_target_bridge"]["learned"] = True; m3_cases.append(("LEARNED_TARGET_BRIDGE", mutated, "declared_cross_domain_target_bridge.learned"))
    mutated = json.loads(json.dumps(common)); mutated["declared_cross_domain_target_bridge"]["monitor_used"] = True; m3_cases.append(("MONITOR_TARGET_USED", mutated, "declared_cross_domain_target_bridge.monitor_used"))
    for reason, candidate, field in m3_cases:
        passed, reasons, observed = CHECKER.common_target_contract(candidate)
        record("M3_TARGET_CONSERVATION", reason, passed, reasons, observed, "common_target_contract", field)

    decision = json.loads(json.dumps(common))
    decision["decision_stack"] = {name: {"executor": "same", "reader": "same"} for name in decision["lines"]}
    decision["external_operations"] = {name: "same" for name in decision["lines"]}
    m4_cases = []
    mutated = json.loads(json.dumps(decision)); mutated["lines"]["relational"]["output"] = "different"; m4_cases.append(("SCORE_SEMANTICS_MISMATCH", mutated, "lines.relational.output"))
    mutated = json.loads(json.dumps(decision)); mutated["decision_stack"]["relational"]["executor"] = "different"; m4_cases.append(("EXECUTOR_CLASS_MISMATCH", mutated, "decision_stack.relational.executor"))
    mutated = json.loads(json.dumps(decision)); mutated["decision_stack"]["relational"]["reader"] = "different"; m4_cases.append(("READER_CLASS_MISMATCH", mutated, "decision_stack.relational.reader"))
    mutated = json.loads(json.dumps(decision)); mutated["calibration_entangled"] = True; m4_cases.append(("CALIBRATION_ENTANGLED", mutated, "calibration_entangled"))
    mutated = json.loads(json.dumps(decision)); mutated["external_operations"]["relational"] = "different"; m4_cases.append(("EXTERNAL_OPERATION_ASYMMETRY", mutated, "external_operations.relational"))
    for reason, candidate, field in m4_cases:
        passed, reasons, observed = CHECKER.common_decision_contract(candidate)
        record("M4_DECISION_STACK_PARITY", reason, passed, reasons, observed, "common_decision_contract", field)

    phase_base = json.loads(json.dumps(config["source_policy"]["phase_access"]))
    authority_base = {"utility": "SYNTHETIC_EXTERNAL", "monitor_or_lockbox_opened": False}
    m5_cases = [
        ("UNBOUND_AUTHORITY", phase_base, authority_base, False, "clean", "", "", "clean", "candidate_valid"),
        ("UTILITY_LEAKAGE", phase_base, {**authority_base, "utility": "OBSERVED"}, True, "clean", "", "", "clean", "authority.utility"),
        ("PHASE_VIOLATION", {**phase_base, "E_EVALUATOR": ["W52_POLICY"]}, authority_base, True, "clean", "", "", "clean", "phase_access.E_EVALUATOR"),
        ("MONITOR_OR_LOCKBOX_OPENED", phase_base, {**authority_base, "monitor_or_lockbox_opened": True}, True, "clean", "", "", "clean", "authority.monitor_or_lockbox_opened"),
        ("CHECKER_NOT_INDEPENDENT", phase_base, authority_base, True, "clean", "", "evaluate_proportional_mapping_feasibility", "clean", "checker_imports"),
    ]
    for reason, phase, authority, valid, evaluator_source, evaluator_imports, checker_imports, builder_source, field in m5_cases:
        passed, reasons, observed = CHECKER.authority_phase_contract(phase, authority, valid, evaluator_source, evaluator_imports, checker_imports, builder_source)
        record("M5_AUTHORITY_PHASES", reason, passed, reasons, observed, "authority_phase_contract", field)

    good_receipt = {"id": "SOURCE", "path": "source.json", "expected": "a" * 64, "actual": "a" * 64, "status": "PASS"}
    for identifier, cases in {
        "R1_SOURCE_COMPLETE": [("GRAPH_SOURCE_MISSING", "source_receipts", [{**good_receipt, "actual": None, "status": "FAIL"}]), ("GRAPH_HASH_MISMATCH", "source_receipts", [{**good_receipt, "actual": "b" * 64, "status": "FAIL"}]), ("GRAPH_SCHEMA_INVALID", "schema_valid", False)],
        "S1_SOURCE_COMPLETE": [("SET_SOURCE_MISSING", "source_receipts", [{**good_receipt, "actual": None, "status": "FAIL"}]), ("SET_HASH_MISMATCH", "source_receipts", [{**good_receipt, "actual": "b" * 64, "status": "FAIL"}]), ("SET_SCHEMA_INVALID", "schema_valid", False), ("SET_ROLE_COUNTS_INVALID", "role_counts_valid", False)],
    }.items():
        base = {"source_receipts": [good_receipt], "schema_valid": True, **({"role_counts_valid": True} if identifier.startswith("S") else {})}
        for reason, field, value in cases:
            facts = json.loads(json.dumps(base)); facts[field] = value
            reasons = CHECKER.native_reason_codes(identifier, facts)
            record(identifier, reason, not reasons, reasons, facts, "source_receipt_contract+native_reason_codes", field)

    native_cases = {
        "R2_PUBLIC_PARITY": ({"unit_keys_equal": True, "public_inputs_equal": True, "private_field_exposed": False}, [("GRAPH_UNIT_MISMATCH", "unit_keys_equal", False), ("GRAPH_INPUT_MISMATCH", "public_inputs_equal", False), ("GRAPH_PRIVATE_LEAKAGE", "private_field_exposed", True)]),
        "R3_REPRESENTATION_OUTPUT": ({"outputs_present": True, "outputs_finite": True, "topology_preserved": True}, [("REPRESENTATION_OUTPUT_MISSING", "outputs_present", False), ("REPRESENTATION_OUTPUT_NONFINITE", "outputs_finite", False), ("TOPOLOGY_CHANGED", "topology_preserved", False)]),
        "R4_EXECUTOR_FACTORIAL": ({"inputs_equal": True, "recipe_equal": True, "cells_replayed": True, "truth_used": False}, [("EXECUTOR_INPUT_MISMATCH", "inputs_equal", False), ("EXECUTOR_RECIPE_MISMATCH", "recipe_equal", False), ("EXECUTOR_CELL_MISSING", "cells_replayed", False), ("TRUTH_USED_BY_EXECUTOR", "truth_used", True)]),
        "R5_TARGET_AUTHORITY": ({"target_join_exact": True, "gauge_canonical": True, "mechanism_used": False}, [("GRAPH_TARGET_JOIN_INVALID", "target_join_exact", False), ("GAUGE_NOT_CANONICAL", "gauge_canonical", False), ("MECHANISM_LEAKAGE", "mechanism_used", True)]),
        "R6_ESTIMAND_CONTROLS": ({"estimands_equal": True, "controls_exact": True, "support_positive": True}, [("RELATIONAL_ESTIMAND_MISMATCH", "estimands_equal", False), ("RELATIONAL_CONTROL_MISMATCH", "controls_exact", False), ("RELATIONAL_SUPPORT_EMPTY", "support_positive", False)]),
        "S2_POSTERIOR_PARITY": ({"cells_present": True, "mass_valid": True, "alignment_exact": True, "utility_used": False}, [("POSTERIOR_CELL_MISSING", "cells_present", False), ("POSTERIOR_MASS_INVALID", "mass_valid", False), ("POSTERIOR_ALIGNMENT_MISMATCH", "alignment_exact", False), ("UTILITY_IN_POSTERIOR", "utility_used", True)]),
        "S3_FOUR_CELLS_EXECUTABLE": ({"cells_present": True, "hard_posterior_bound": True, "contextual_recipe_exact": True, "duplication_declared": True}, [("SET_DECISION_CELL_MISSING", "cells_present", False), ("HARD_READER_NOT_POSTERIOR_BOUND", "hard_posterior_bound", False), ("CONTEXTUAL_RECIPE_MISMATCH", "contextual_recipe_exact", False), ("CELL_DUPLICATION_UNDECLARED", "duplication_declared", False)]),
        "S4_FIT_SUPPORT_FREEZE": ({"proposer_support_positive": True, "harm_classes_present": True, "incompatibility_classes_present": True, "selection_support_positive": True, "phase_closed": True}, [("PROPOSER_SUPPORT_EMPTY", "proposer_support_positive", False), ("HARM_CLASS_MISSING", "harm_classes_present", False), ("INCOMPATIBILITY_CLASS_MISSING", "incompatibility_classes_present", False), ("SELECTION_SUPPORT_EMPTY", "selection_support_positive", False), ("SET_PHASE_VIOLATION", "phase_closed", False)]),
        "S5_TARGET_UTILITY_AUTHORITY": ({"target_join_exact": True, "target_nonempty": True, "target_used_as_input": False, "utility_contract_valid": True}, [("SET_TARGET_JOIN_INVALID", "target_join_exact", False), ("EMPTY_TARGET_SET", "target_nonempty", False), ("TARGET_LEAKAGE", "target_used_as_input", True), ("UTILITY_CONTRACT_MISMATCH", "utility_contract_valid", False)]),
        "S6_ESTIMAND_CONTROLS": ({"estimands_equal": True, "posterior_reader_entangled": False, "controls_exact": True, "support_positive": True}, [("SET_ESTIMAND_MISMATCH", "estimands_equal", False), ("POSTERIOR_READER_ENTANGLED", "posterior_reader_entangled", True), ("SET_CONTROL_MISMATCH", "controls_exact", False), ("SET_SUPPORT_EMPTY", "support_positive", False)]),
    }
    for identifier, (base, cases) in native_cases.items():
        for reason, field, value in cases:
            facts = json.loads(json.dumps(base)); facts[field] = value
            reasons = CHECKER.native_reason_codes(identifier, facts)
            record(identifier, reason, not reasons, reasons, facts, "native_reason_codes", field)

    result = {"schema_version": "proportional-mapping-mutation-execution-v3", "status": "PASS", "predicate_mutations": rows, "candidate_corruptions": {}, **CHECKER.FIXED}
    assert len(rows) == sum(len(reasons) for reasons in CHECKER.REASONS.values())
    assert {(row["id"], row["reason_codes"][0]) for row in rows} == {(identifier, reason) for identifier, reasons in CHECKER.REASONS.items() for reason in reasons}
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
