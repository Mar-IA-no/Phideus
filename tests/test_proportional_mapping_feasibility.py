from __future__ import annotations

import copy
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

    def material_case(identifier: str, reason: str, base: dict, mutate, field: str) -> None:
        material = copy.deepcopy(base)
        mutate(material)
        facts = CHECKER.derive_native_facts(identifier, material)
        reasons = CHECKER.native_reason_codes(identifier, facts)
        record(identifier, reason, not reasons, reasons, facts, "derive_native_facts+native_reason_codes", field)

    good_receipt = {"id": "SOURCE", "path": "source.json", "expected": "a" * 64, "actual": "a" * 64, "status": "PASS"}
    graph_states = [{"state": f"raw_{arm}|seed={seed}"} for arm in ("generic", "typed") for seed in (104729, 130363)]
    r1 = {"source_receipts": [good_receipt], "state_rows": graph_states, "public_names": [set(CHECKER.PUBLIC_GRAPH_NAMES) for _ in range(4)], "private_names": [set(CHECKER.PRIVATE_GRAPH_NAMES) for _ in range(4)], "private_fields": {Path(name).stem for name in CHECKER.PRIVATE_GRAPH_NAMES}, "source_digests": [("a", "a")]}
    material_case("R1_SOURCE_COMPLETE", "GRAPH_SOURCE_MISSING", r1, lambda m: m["source_receipts"][0].update(actual=None, status="FAIL"), "source_receipts[0].actual")
    material_case("R1_SOURCE_COMPLETE", "GRAPH_HASH_MISMATCH", r1, lambda m: m["source_receipts"][0].update(actual="b" * 64, status="FAIL"), "source_receipts[0].actual")
    material_case("R1_SOURCE_COMPLETE", "GRAPH_SCHEMA_INVALID", r1, lambda m: m["state_rows"].pop(), "state_rows")

    parity_left = {"unit_key": np.asarray(["u0", "u1"]), "observed_log_ratio": np.asarray([0.1, 0.2])}
    r2 = {"pairs": [{"left": parity_left, "right": copy.deepcopy(parity_left)}], "parity_fields": ("unit_key", "observed_log_ratio"), "public_names": [{"unit_key.npy", "observed_log_ratio.npy"}], "forbidden_public_stems": {"mechanism", "x_true"}}
    material_case("R2_PUBLIC_PARITY", "GRAPH_UNIT_MISMATCH", r2, lambda m: m["pairs"][0]["right"].update(unit_key=np.asarray(["u0", "other"])), "pairs[0].right.unit_key")
    material_case("R2_PUBLIC_PARITY", "GRAPH_INPUT_MISMATCH", r2, lambda m: m["pairs"][0]["right"].update(observed_log_ratio=np.asarray([0.1, 9.0])), "pairs[0].right.observed_log_ratio")
    material_case("R2_PUBLIC_PARITY", "GRAPH_PRIVATE_LEAKAGE", r2, lambda m: m["public_names"][0].add("mechanism.npy"), "public_names[0]")

    r3 = {"arrays": [{"corrected_log_ratio": np.asarray([0.1, 0.2]), "reliability": np.ones(2), "observed_log_ratio": np.asarray([0.1, 0.2])}]}
    material_case("R3_REPRESENTATION_OUTPUT", "REPRESENTATION_OUTPUT_MISSING", r3, lambda m: m["arrays"][0].pop("reliability"), "arrays[0].reliability")
    material_case("R3_REPRESENTATION_OUTPUT", "REPRESENTATION_OUTPUT_NONFINITE", r3, lambda m: m["arrays"][0].update(reliability=np.asarray([1.0, np.nan])), "arrays[0].reliability")
    material_case("R3_REPRESENTATION_OUTPUT", "TOPOLOGY_CHANGED", r3, lambda m: m["arrays"][0].update(corrected_log_ratio=np.asarray([0.1])), "arrays[0].corrected_log_ratio")

    solver_source = "def f():\n    solve_wls(n, edges, valid, corrected, reliability, floor)\n    solve_irls(n, edges, valid, variance, corrected, reliability, config)\n"
    r4 = {"recorded_recipe": {"floor": 1e-6}, "expected_recipe": {"floor": 1e-6}, "replayed_pairs": [(np.asarray([0.0, 1.0]), np.asarray([0.0, 1.0]))], "atol": 1e-8, "executor_source": solver_source, "forbidden_truth_tokens": ("x_true", "target", "mechanism")}
    material_case("R4_EXECUTOR_FACTORIAL", "EXECUTOR_INPUT_MISMATCH", r4, lambda m: m.update(executor_source="def f():\n    solve_wls(n, edges, valid, corrected, reliability, floor)\n    solve_irls(n, edges, valid, variance, typed_corrected, reliability, config)\n"), "executor_source")
    material_case("R4_EXECUTOR_FACTORIAL", "EXECUTOR_RECIPE_MISMATCH", r4, lambda m: m.update(recorded_recipe={"floor": 1e-3}), "recorded_recipe")
    material_case("R4_EXECUTOR_FACTORIAL", "EXECUTOR_CELL_MISSING", r4, lambda m: m.update(replayed_pairs=[(np.asarray([0.0, 1.0]), np.asarray([0.0, 2.0]))]), "replayed_pairs")
    material_case("R4_EXECUTOR_FACTORIAL", "TRUTH_USED_BY_EXECUTOR", r4, lambda m: m.update(executor_source="def f():\n    solve_wls(n, edges, valid, x_true, reliability, floor)\n    solve_irls(n, edges, valid, variance, x_true, reliability, config)\n"), "executor_source")

    r5 = {"views": [{"n_nodes": 2, "edges": np.asarray([[0, 1]]), "x_true": np.asarray([-0.5, 0.5]), "clean_log_ratio": np.asarray([1.0])}], "executor_source": solver_source}
    material_case("R5_TARGET_AUTHORITY", "GRAPH_TARGET_JOIN_INVALID", r5, lambda m: m["views"][0].update(clean_log_ratio=np.asarray([2.0])), "views[0].clean_log_ratio")
    material_case("R5_TARGET_AUTHORITY", "GAUGE_NOT_CANONICAL", r5, lambda m: m["views"][0].update(x_true=np.asarray([0.5, 1.5]), clean_log_ratio=np.asarray([1.0])), "views[0].x_true")
    material_case("R5_TARGET_AUTHORITY", "MECHANISM_LEAKAGE", r5, lambda m: m.update(executor_source="def f():\n    solve_wls(n, edges, valid, mechanism, reliability, floor)\n"), "executor_source")

    r6 = {"estimand_pairs": [({"mean": 1.0}, {"mean": 1.0})], "control_pairs": [({"digest": "a"}, {"digest": "a"})], "support_mask": np.asarray([True]), "atol": 1e-8}
    material_case("R6_ESTIMAND_CONTROLS", "RELATIONAL_ESTIMAND_MISMATCH", r6, lambda m: m.update(estimand_pairs=[({"mean": 2.0}, {"mean": 1.0})]), "estimand_pairs")
    material_case("R6_ESTIMAND_CONTROLS", "RELATIONAL_CONTROL_MISMATCH", r6, lambda m: m.update(control_pairs=[({"digest": "b"}, {"digest": "a"})]), "control_pairs")
    material_case("R6_ESTIMAND_CONTROLS", "RELATIONAL_SUPPORT_EMPTY", r6, lambda m: m.update(support_mask=np.asarray([False])), "support_mask")

    logits = np.zeros((384, 4)); seed_logits = np.zeros((3, 384, 4)); target = np.ones((384, 4), dtype=bool); roles = np.asarray(["calibration_fit"] * 192 + ["decision_select"] * 192)
    s1 = {"source_receipts": [good_receipt], "logits": logits, "seed_logits": seed_logits, "target": target, "roles": roles, "source_pairs": [(logits, logits.copy()), (seed_logits, seed_logits.copy()), (target, target.copy()), (roles, roles.copy())]}
    material_case("S1_SOURCE_COMPLETE", "SET_SOURCE_MISSING", s1, lambda m: m["source_receipts"][0].update(actual=None, status="FAIL"), "source_receipts[0].actual")
    material_case("S1_SOURCE_COMPLETE", "SET_HASH_MISMATCH", s1, lambda m: m["source_receipts"][0].update(actual="b" * 64, status="FAIL"), "source_receipts[0].actual")
    material_case("S1_SOURCE_COMPLETE", "SET_SCHEMA_INVALID", s1, lambda m: m.update(logits=np.zeros((383, 4))), "logits")
    material_case("S1_SOURCE_COMPLETE", "SET_ROLE_COUNTS_INVALID", s1, lambda m: m.update(roles=np.asarray(["calibration_fit"] * 191 + ["decision_select"] * 193)), "roles")

    mass = np.full((384, 15), 1.0 / 15.0); keys = np.asarray([f"k{i}" for i in range(384)])
    s2 = {"masses": {"MARGINAL": mass, "JOINT": mass.copy()}, "rows": 384, "alignment_pairs": [(keys, keys.copy())], "posterior_source": "def posterior(logits, theta): return logits + theta"}
    material_case("S2_POSTERIOR_PARITY", "POSTERIOR_CELL_MISSING", s2, lambda m: m["masses"].pop("JOINT"), "masses.JOINT")
    material_case("S2_POSTERIOR_PARITY", "POSTERIOR_MASS_INVALID", s2, lambda m: m["masses"].update(JOINT=np.full((384, 15), -1.0)), "masses.JOINT")
    material_case("S2_POSTERIOR_PARITY", "POSTERIOR_ALIGNMENT_MISMATCH", s2, lambda m: m.update(alignment_pairs=[(keys, keys[::-1])]), "alignment_pairs")
    material_case("S2_POSTERIOR_PARITY", "UTILITY_IN_POSTERIOR", s2, lambda m: m.update(posterior_source="def posterior(logits, utility): return logits + utility"), "posterior_source")

    zeros = np.zeros((384, 24), dtype=np.int64); ones = np.ones((384, 24), dtype=np.int64)
    actions = {"MARGINAL_HARD": zeros, "JOINT_HARD": zeros.copy(), "MARGINAL_CONTEXTUAL": ones, "JOINT_CONTEXTUAL": ones.copy()}
    declared_cells = {name: {"shape": [384, 24], "sha256": CHECKER.array_digest(value)} for name, value in actions.items()}
    s3 = {"actions": actions, "rows": 384, "declared_cells": declared_cells, "expected_hard_digests": {"MARGINAL": CHECKER.array_digest(zeros), "JOINT": CHECKER.array_digest(zeros)}, "expected_contextual_digests": {"MARGINAL": CHECKER.array_digest(ones), "JOINT": CHECKER.array_digest(ones)}, "declared_hard_duplication": True}
    material_case("S3_FOUR_CELLS_EXECUTABLE", "SET_DECISION_CELL_MISSING", s3, lambda m: m["actions"].pop("JOINT_CONTEXTUAL"), "actions.JOINT_CONTEXTUAL")
    material_case("S3_FOUR_CELLS_EXECUTABLE", "HARD_READER_NOT_POSTERIOR_BOUND", s3, lambda m: m["expected_hard_digests"].update(JOINT="0" * 64), "expected_hard_digests.JOINT")
    material_case("S3_FOUR_CELLS_EXECUTABLE", "CONTEXTUAL_RECIPE_MISMATCH", s3, lambda m: m["expected_contextual_digests"].update(JOINT="0" * 64), "expected_contextual_digests.JOINT")
    material_case("S3_FOUR_CELLS_EXECUTABLE", "CELL_DUPLICATION_UNDECLARED", s3, lambda m: m.update(declared_hard_duplication=False), "declared_hard_duplication")

    context = {"fit_rows": 10, "harm_0_1": [5, 5], "incompatibility_0_1": [5, 5], "candidate_count": 2}
    s4 = {"contexts": [context, copy.deepcopy(context)], "roles": roles, "monitor_or_lockbox_opened": False}
    material_case("S4_FIT_SUPPORT_FREEZE", "PROPOSER_SUPPORT_EMPTY", s4, lambda m: m["contexts"][0].update(fit_rows=0), "contexts[0].fit_rows")
    material_case("S4_FIT_SUPPORT_FREEZE", "HARM_CLASS_MISSING", s4, lambda m: m["contexts"][0].update(harm_0_1=[10, 0]), "contexts[0].harm_0_1")
    material_case("S4_FIT_SUPPORT_FREEZE", "INCOMPATIBILITY_CLASS_MISSING", s4, lambda m: m["contexts"][0].update(incompatibility_0_1=[10, 0]), "contexts[0].incompatibility_0_1")
    material_case("S4_FIT_SUPPORT_FREEZE", "SELECTION_SUPPORT_EMPTY", s4, lambda m: m.update(roles=np.asarray(["calibration_fit"] * 384)), "roles")
    material_case("S4_FIT_SUPPORT_FREEZE", "SET_PHASE_VIOLATION", s4, lambda m: m.update(monitor_or_lockbox_opened=True), "monitor_or_lockbox_opened")

    utility = np.tile(np.asarray([0.0, 1.0, 2.0, 3.0]), (24, 1))
    s5 = {"public_keys": keys, "private_keys": keys.copy(), "target": target, "utility": utility, "posterior_source": "def posterior(logits, theta): return logits + theta"}
    material_case("S5_TARGET_UTILITY_AUTHORITY", "SET_TARGET_JOIN_INVALID", s5, lambda m: m.update(private_keys=keys[::-1]), "private_keys")
    material_case("S5_TARGET_UTILITY_AUTHORITY", "EMPTY_TARGET_SET", s5, lambda m: m["target"].__setitem__(0, False), "target[0]")
    material_case("S5_TARGET_UTILITY_AUTHORITY", "TARGET_LEAKAGE", s5, lambda m: m.update(posterior_source="def posterior(logits, target): return logits + target"), "posterior_source")
    material_case("S5_TARGET_UTILITY_AUTHORITY", "UTILITY_CONTRACT_MISMATCH", s5, lambda m: m.update(utility=np.ones((24, 4))), "utility")

    s6 = {"estimand_pairs": [({"nll": 1.0}, {"nll": 1.0})], "posterior_metric_source": "def metric(mass, target): return mass.sum()", "reader_tokens": ("hard_actions", "contextual_actions", "reader"), "control_pairs": [({"digest": "a"}, {"digest": "a"})], "support_mask": np.asarray([True])}
    material_case("S6_ESTIMAND_CONTROLS", "SET_ESTIMAND_MISMATCH", s6, lambda m: m.update(estimand_pairs=[({"nll": 2.0}, {"nll": 1.0})]), "estimand_pairs")
    material_case("S6_ESTIMAND_CONTROLS", "POSTERIOR_READER_ENTANGLED", s6, lambda m: m.update(posterior_metric_source="def metric(mass, reader): return reader(mass)"), "posterior_metric_source")
    material_case("S6_ESTIMAND_CONTROLS", "SET_CONTROL_MISMATCH", s6, lambda m: m.update(control_pairs=[({"digest": "b"}, {"digest": "a"})]), "control_pairs")
    material_case("S6_ESTIMAND_CONTROLS", "SET_SUPPORT_EMPTY", s6, lambda m: m.update(support_mask=np.asarray([False])), "support_mask")

    result = {"schema_version": "proportional-mapping-mutation-execution-v4", "status": "PASS", "predicate_mutations": rows, "candidate_corruptions": {}, **CHECKER.FIXED}
    assert len(rows) == sum(len(reasons) for reasons in CHECKER.REASONS.values())
    assert {(row["id"], row["reason_codes"][0]) for row in rows} == {(identifier, reason) for identifier, reasons in CHECKER.REASONS.items() for reason in reasons}
    for variable in ("MAPPING_FEASIBILITY_RUN_A", "MAPPING_FEASIBILITY_RUN_B"):
        run_path = os.environ.get(variable)
        if run_path:
            PREPARE.write_json(Path(run_path) / "mutation_results.json", result)


def test_public_graph_private_field_reaches_r2_production_path(tmp_path: Path) -> None:
    run_value = os.environ.get("MAPPING_FEASIBILITY_RUN_A")
    if not run_value:
        pytest.skip("requires a completed runner development surface")
    run = tmp_path / "run"
    shutil.copytree(Path(run_value), run)
    state = "raw_generic__seed=104729"
    shutil.copyfile(
        run / f"prepared/private_dev/graph/{state}/mechanism.npy",
        run / f"prepared/public/graph/{state}/mechanism.npy",
    )
    config = json.loads((EXP / "configs/proportional_mapping_feasibility_v1.json").read_text())
    predicates, technical, _ = CHECKER.compute(run, config)
    r1 = next(row for row in predicates if row["id"] == "R1_SOURCE_COMPLETE")
    r2 = next(row for row in predicates if row["id"] == "R2_PUBLIC_PARITY")
    assert technical["artifact_status"] == "FAIL"
    assert r1["status"] == "FAIL" and "GRAPH_SCHEMA_INVALID" in r1["reason_codes"]
    assert r2["status"] == "FAIL" and r2["reason_codes"] == ["GRAPH_PRIVATE_LEAKAGE"]


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
