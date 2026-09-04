#!/usr/bin/env python3
"""Exercise the external BudgetPath utility port on synthetic fixtures only."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import subprocess
import sys
import time
from itertools import combinations
from pathlib import Path
from typing import Any, Callable

import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from geometria_proporcional.budget_path_schema import (  # noqa: E402
    BUDGET_FRACTIONS,
    PAIR_STATES,
    POLICY_IDS,
    SCHEMA_VERSION,
    BudgetPathArtifact,
    artifact_id,
    canonical_json,
)
from geometria_proporcional.budget_path_utility import (  # noqa: E402
    ExternalUtilityViolation,
    declaration_id,
    digest,
    evaluate_external_utility,
    make_declaration,
)

DEFAULT_CONFIG = ROOT / (
    "experiments/geometria_proporcional/configs/"
    "proportional_budget_path_external_utility_port_v1.json"
)
SOURCE_FILES = (
    "experiments/geometria_proporcional/PLAN_PROPORTIONAL_BUDGET_PATH_EXTERNAL_UTILITY_PORT_CPU.md",
    "experiments/geometria_proporcional/configs/proportional_budget_path_external_utility_port_v1.json",
    "experiments/geometria_proporcional/run_proportional_budget_path_external_utility_port.py",
    "src/geometria_proporcional/budget_path_schema.py",
    "src/geometria_proporcional/budget_path_utility.py",
)
AXES = [
    {
        "axis_id": "iid_delta",
        "population": "IID",
        "estimator": "mean_quotient_rmse_delta",
        "unit": "quotient_rmse",
        "direction": "minimize",
    },
    {
        "axis_id": "grouped_delta",
        "population": "grouped",
        "estimator": "mean_quotient_rmse_delta",
        "unit": "quotient_rmse",
        "direction": "minimize",
    },
]
TRADEOFF_COORDINATES = (
    (0.0, 0.0),
    (-1.0, 2.0),
    (-2.0, 1.0),
    (-3.0, 2.0),
    (-1.0, -1.0),
    (0.0, -2.0),
    (1.0, -3.0),
)
SINGLETON_COORDINATES = (
    (0.0, 0.0),
    (1.0, 2.0),
    (2.0, 1.0),
    (1.0, 1.0),
    (-5.0, -5.0),
    (0.0, 2.0),
    (2.0, 0.0),
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")


def load_config(path: Path) -> dict[str, Any]:
    cfg = json.loads(path.read_text())
    expected = {
        "schema_version",
        "source_budget_path_interface",
        "source_interface_manifest_sha256",
        "expected_fixtures",
        "expected_decisions",
        "expected_metamorphic_checks",
        "expected_invalid_cases",
        "numerical_zero_tolerance",
        "execution",
    }
    if set(cfg) != expected:
        raise ValueError("invalid utility-port run keys")
    if cfg["schema_version"] != "proportional-budget-path-external-utility-port-run-v1":
        raise ValueError("invalid utility-port run schema")
    if (
        cfg["expected_fixtures"] != 2
        or cfg["expected_decisions"] != 7
        or cfg["expected_metamorphic_checks"] != 3
        or cfg["expected_invalid_cases"] != 12
    ):
        raise ValueError("utility-port suite counts changed")
    if cfg["numerical_zero_tolerance"] != 1e-12:
        raise ValueError("utility-port tolerance changed")
    if cfg["execution"] != {"max_seconds": 120, "max_rss_gib": 2.0}:
        raise ValueError("utility-port execution contract changed")
    return cfg


def verify_source(cfg: dict[str, Any]) -> None:
    root = ROOT / cfg["source_budget_path_interface"]
    if sha(root / "manifest.json") != cfg["source_interface_manifest_sha256"]:
        raise AssertionError("R373 source manifest mismatch")
    manifest = json.loads((root / "manifest.json").read_text())
    for rel, expected in manifest["deterministic_files"].items():
        if sha(root / rel) != expected:
            raise AssertionError(f"R373 deterministic mismatch: {rel}")


def pareto_mask(points: np.ndarray, tolerance: float = 1e-12) -> np.ndarray:
    keep = np.ones(len(points), dtype=bool)
    for index, point in enumerate(points):
        weak = np.all(points <= point + tolerance, axis=1)
        strict = np.any(points < point - tolerance, axis=1)
        if np.any(weak & strict):
            keep[index] = False
    return keep


def pair_status(delta: np.ndarray, tolerance: float = 1e-12) -> str:
    iid, grouped = delta
    if abs(iid) <= tolerance and abs(grouped) <= tolerance:
        return "EQUIVALENT"
    if iid <= tolerance and grouped <= tolerance:
        return "EXPANSION_DOMINATES"
    if iid >= -tolerance and grouped >= -tolerance:
        return "BASE_DOMINATES"
    if iid > tolerance and grouped < -tolerance:
        return "IID_COST_GROUPED_GAIN"
    if iid < -tolerance and grouped > tolerance:
        return "IID_GAIN_GROUPED_COST"
    raise AssertionError("unclassified synthetic pair")


def build_fixture(
    fixture_id: str,
    coordinates: tuple[tuple[float, float], ...],
    *,
    reverse_reader: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    points = np.asarray(coordinates, dtype=float)
    front = pareto_mask(points)
    lineage = {"scope": "SYNTHETIC_FIXTURE", "fixture_id": fixture_id}
    policies = []
    for index, policy_id in enumerate(POLICY_IDS):
        policies.append(
            {
                "policy_id": policy_id,
                "budget_fraction": BUDGET_FRACTIONS[index],
                "action_count": index,
                "action_sha256": digest([fixture_id, "action", index]),
                "objective_master_count": 1,
                "objective_sha256": digest([fixture_id, "objective", *points[index]]),
                "objective_mean": {
                    "iid_delta": float(points[index, 0]),
                    "grouped_delta": float(points[index, 1]),
                },
                "front_member": bool(front[index]),
                "bootstrap_front_frequency": float(front[index]),
            }
        )
    pairs = []
    for base, expansion in combinations(range(len(POLICY_IDS)), 2):
        delta = points[expansion] - points[base]
        status = pair_status(delta)
        pairs.append(
            {
                "pair_id": f"{POLICY_IDS[base]}->{POLICY_IDS[expansion]}",
                "base_policy": POLICY_IDS[base],
                "expansion_policy": POLICY_IDS[expansion],
                "adjacent": expansion == base + 1,
                "mean_increment": {"iid": float(delta[0]), "grouped": float(delta[1])},
                "increment_sha256": digest(
                    [fixture_id, "increment", base, expansion, *delta]
                ),
                "point_status": status,
                "bootstrap_status_frequency": {
                    state: float(state == status) for state in PAIR_STATES
                },
            }
        )
    front_ids = [POLICY_IDS[index] for index in np.flatnonzero(front)]
    if reverse_reader:
        front_ids.reverse()
    artifact = BudgetPathArtifact(
        schema_version=SCHEMA_VERSION,
        artifact_id=artifact_id(lineage),
        lineage=lineage,
        axes=copy.deepcopy(AXES),
        policies=policies,
        pairs=pairs,
        reader={"kind": "PARETO_SET", "policy_ids": front_ids},
        authority={
            "artifact_status": "SYNTHETIC_FIXTURE",
            "formal_claim_status": "STRUCTURE_ONLY",
            "empirical_claim_status": "NOT_APPLICABLE",
            "physical_authority_status": "NOT_CLAIMED",
            "decision_status": "UNRESOLVED",
        },
        utility_boundary={
            "status": "ABSENT_EXTERNAL_REQUIRED",
            "embedded_utility_fields": [],
        },
    ).to_dict()
    receipt = synthetic_receipt(artifact)
    return artifact, receipt


def synthetic_receipt(artifact: dict[str, Any]) -> dict[str, Any]:
    parsed = BudgetPathArtifact.from_dict(artifact)
    points = np.asarray(
        [
            [row["objective_mean"]["iid_delta"], row["objective_mean"]["grouped_delta"]]
            for row in parsed.policies
        ],
        dtype=float,
    )
    if not np.all(np.isfinite(points)):
        raise ValueError("synthetic fixture has nonfinite coordinate")
    expected = {
        parsed.policies[index]["policy_id"]
        for index in np.flatnonzero(pareto_mask(points))
    }
    if set(parsed.reader["policy_ids"]) != expected:
        raise ValueError("synthetic fixture reader mismatch")
    return {
        "schema_version": "budget-path-synthetic-receipt-v1",
        "artifact_id": parsed.artifact_id,
        "artifact_sha256": digest(artifact),
        "checker_status": "VALID",
        "scope": "SYNTHETIC_FIXTURE",
        "checker_kind": "SYNTHETIC_FIXTURE_CHECKER",
    }


def positive_cases(
    tradeoff: tuple[dict[str, Any], dict[str, Any]],
    singleton: tuple[dict[str, Any], dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "case_id": "P01_WEIGHTED_UNIQUE",
            "fixture": "tradeoff",
            "utility_kind": "WEIGHTED_SUM_MINIMIZE",
            "parameters": {"weights": {"iid_delta": 0.8, "grouped_delta": 0.2}},
            "expected_selected": ["budget_0.05"],
            "expected_status": "SYNTHETIC_FIXTURE_SELECTED",
        },
        {
            "case_id": "P02_WEIGHTED_TIE",
            "fixture": "tradeoff",
            "utility_kind": "WEIGHTED_SUM_MINIMIZE",
            "parameters": {"weights": {"iid_delta": 0.5, "grouped_delta": 0.5}},
            "expected_selected": ["budget_0.10", "budget_0.20", "budget_0.40"],
            "expected_status": "SYNTHETIC_FIXTURE_SELECTED",
        },
        {
            "case_id": "P03_LEXICOGRAPHIC_IID",
            "fixture": "tradeoff",
            "utility_kind": "LEXICOGRAPHIC_MINIMIZE",
            "parameters": {"axis_priority": ["iid_delta", "grouped_delta"]},
            "expected_selected": ["budget_0.05"],
            "expected_status": "SYNTHETIC_FIXTURE_SELECTED",
        },
        {
            "case_id": "P04_LEXICOGRAPHIC_GROUPED",
            "fixture": "tradeoff",
            "utility_kind": "LEXICOGRAPHIC_MINIMIZE",
            "parameters": {"axis_priority": ["grouped_delta", "iid_delta"]},
            "expected_selected": ["budget_0.40"],
            "expected_status": "SYNTHETIC_FIXTURE_SELECTED",
        },
        {
            "case_id": "P05_EPSILON_FEASIBLE",
            "fixture": "tradeoff",
            "utility_kind": "EPSILON_CONSTRAINT_MINIMIZE",
            "parameters": {
                "primary_axis": "iid_delta",
                "constraint_axis": "grouped_delta",
                "max_value": -1.5,
            },
            "expected_selected": ["budget_0.20"],
            "expected_status": "SYNTHETIC_FIXTURE_SELECTED",
        },
        {
            "case_id": "P06_EPSILON_ABSTAIN",
            "fixture": "tradeoff",
            "utility_kind": "EPSILON_CONSTRAINT_MINIMIZE",
            "parameters": {
                "primary_axis": "iid_delta",
                "constraint_axis": "grouped_delta",
                "max_value": -4.0,
            },
            "expected_selected": [],
            "expected_status": "ABSTAIN_NO_FEASIBLE_POLICY",
        },
        {
            "case_id": "P07_SINGLETON_FRONT",
            "fixture": "singleton",
            "utility_kind": "WEIGHTED_SUM_MINIMIZE",
            "parameters": {"weights": {"iid_delta": 0.5, "grouped_delta": 0.5}},
            "expected_selected": ["budget_0.10"],
            "expected_status": "SYNTHETIC_FIXTURE_SELECTED",
        },
    ]


def execute_positive(
    fixtures: dict[str, tuple[dict[str, Any], dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    cases = positive_cases(fixtures["tradeoff"], fixtures["singleton"])
    declarations, decisions = [], []
    for case in cases:
        artifact, receipt = fixtures[case["fixture"]]
        declaration = make_declaration(
            artifact=artifact,
            checker_receipt=receipt,
            utility_kind=case["utility_kind"],
            parameters=case["parameters"],
        )
        before = (canonical_json(artifact), canonical_json(receipt))
        decision = evaluate_external_utility(artifact, receipt, declaration)
        after = (canonical_json(artifact), canonical_json(receipt))
        if before != after:
            raise AssertionError(f"{case['case_id']}: utility port mutated input")
        if decision["selected_policy_ids"] != case["expected_selected"]:
            raise AssertionError(f"{case['case_id']}: wrong selected set")
        if decision["decision_status"] != case["expected_status"]:
            raise AssertionError(f"{case['case_id']}: wrong decision status")
        declarations.append(declaration)
        decisions.append(decision)
    return cases, declarations, decisions


def metamorphic_checks(
    fixtures: dict[str, tuple[dict[str, Any], dict[str, Any]]],
) -> list[dict[str, Any]]:
    base_artifact, base_receipt = fixtures["tradeoff"]
    base_decl = make_declaration(
        artifact=base_artifact,
        checker_receipt=base_receipt,
        utility_kind="WEIGHTED_SUM_MINIMIZE",
        parameters={"weights": {"iid_delta": 0.5, "grouped_delta": 0.5}},
    )
    base_decision = evaluate_external_utility(base_artifact, base_receipt, base_decl)

    reversed_artifact, reversed_receipt = build_fixture(
        "tradeoff_reader_reversed", TRADEOFF_COORDINATES, reverse_reader=True
    )
    reversed_decl = make_declaration(
        artifact=reversed_artifact,
        checker_receipt=reversed_receipt,
        utility_kind="WEIGHTED_SUM_MINIMIZE",
        parameters={"weights": {"iid_delta": 0.5, "grouped_delta": 0.5}},
    )
    reversed_decision = evaluate_external_utility(
        reversed_artifact, reversed_receipt, reversed_decl
    )

    decoy_coordinates = list(TRADEOFF_COORDINATES)
    decoy_coordinates[0] = (100.0, 100.0)
    decoy_artifact, decoy_receipt = build_fixture(
        "tradeoff_decoy_changed", tuple(decoy_coordinates)
    )
    decoy_decl = make_declaration(
        artifact=decoy_artifact,
        checker_receipt=decoy_receipt,
        utility_kind="WEIGHTED_SUM_MINIMIZE",
        parameters={"weights": {"iid_delta": 0.5, "grouped_delta": 0.5}},
    )
    decoy_decision = evaluate_external_utility(
        decoy_artifact, decoy_receipt, decoy_decl
    )

    checks = [
        {
            "check_id": "X01_READER_ORDER_INVARIANCE",
            "status": (
                "PASS"
                if reversed_decision["selected_policy_ids"]
                == base_decision["selected_policy_ids"]
                else "FAIL"
            ),
        },
        {
            "check_id": "X02_DOMINATED_DECOY_INVARIANCE",
            "status": (
                "PASS"
                if decoy_decision["selected_policy_ids"]
                == base_decision["selected_policy_ids"]
                else "FAIL"
            ),
        },
        {
            "check_id": "X03_INPUT_BYTE_IMMUTABILITY",
            "status": (
                "PASS"
                if digest(base_artifact) == base_decl["artifact_sha256"]
                and digest(base_receipt) == base_decl["checker_receipt_sha256"]
                else "FAIL"
            ),
        },
    ]
    if any(row["status"] != "PASS" for row in checks):
        raise AssertionError("metamorphic utility-port check failed")
    return checks


def reidentify(declaration: dict[str, Any]) -> None:
    payload = copy.deepcopy(declaration)
    payload.pop("declaration_id")
    declaration["declaration_id"] = declaration_id(payload)


def forged_receipt(artifact: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "budget-path-synthetic-receipt-v1",
        "artifact_id": artifact["artifact_id"],
        "artifact_sha256": digest(artifact),
        "checker_status": "VALID",
        "scope": "SYNTHETIC_FIXTURE",
        "checker_kind": "SYNTHETIC_FIXTURE_CHECKER",
    }


def expect_invalid(
    case_id: str,
    artifact: dict[str, Any],
    receipt: dict[str, Any],
    declaration: dict[str, Any],
    mutate: Callable[[dict[str, Any], dict[str, Any], dict[str, Any]], None],
    expected: str,
) -> dict[str, str]:
    a, r, d = (
        copy.deepcopy(artifact),
        copy.deepcopy(receipt),
        copy.deepcopy(declaration),
    )
    mutate(a, r, d)
    try:
        evaluate_external_utility(a, r, d)
    except (ExternalUtilityViolation, ValueError) as exc:
        message = str(exc)
        if expected not in message:
            raise AssertionError(f"{case_id}: wrong rejection {message!r}") from exc
        return {
            "case_id": case_id,
            "design_region": "PROTOCOL_INVALID",
            "status": "REJECTED",
            "expected_signal": expected,
            "observed_signal": message,
        }
    raise AssertionError(f"{case_id}: invalid utility composition accepted")


def invalid_suite(
    artifact: dict[str, Any], receipt: dict[str, Any]
) -> list[dict[str, str]]:
    weighted = make_declaration(
        artifact=artifact,
        checker_receipt=receipt,
        utility_kind="WEIGHTED_SUM_MINIMIZE",
        parameters={"weights": {"iid_delta": 0.5, "grouped_delta": 0.5}},
    )

    def wrong_artifact_hash(a, r, d):
        d["artifact_sha256"] = "0" * 64
        reidentify(d)

    def wrong_receipt_hash(a, r, d):
        d["checker_receipt_sha256"] = "0" * 64
        reidentify(d)

    def invalid_receipt(a, r, d):
        r["checker_status"] = "INVALID"
        d["checker_receipt_sha256"] = digest(r)
        reidentify(d)

    def fake_user_scope(a, r, d):
        d["scope"] = "USER_DECLARED"
        reidentify(d)

    def zero_weight(a, r, d):
        d["parameters"]["weights"] = {"iid_delta": 1.0, "grouped_delta": 0.0}
        reidentify(d)

    def incomplete_weights(a, r, d):
        d["parameters"]["weights"] = {"iid_delta": 1.0}
        reidentify(d)

    def duplicate_priority(a, r, d):
        replacement = make_declaration(
            artifact=a,
            checker_receipt=r,
            utility_kind="LEXICOGRAPHIC_MINIMIZE",
            parameters={"axis_priority": ["iid_delta", "iid_delta"]},
        )
        d.clear()
        d.update(replacement)

    def same_epsilon_axis(a, r, d):
        replacement = make_declaration(
            artifact=a,
            checker_receipt=r,
            utility_kind="EPSILON_CONSTRAINT_MINIMIZE",
            parameters={
                "primary_axis": "iid_delta",
                "constraint_axis": "iid_delta",
                "max_value": 0.0,
            },
        )
        d.clear()
        d.update(replacement)

    def hidden_tie_break(a, r, d):
        d["tie_policy"] = "FIRST_POLICY"
        reidentify(d)

    def selected_in_declaration(a, r, d):
        d["selected_policy_id"] = "budget_0.40"

    def unknown_reader(a, r, d):
        a["reader"]["policy_ids"].append("unknown_policy")
        r.clear()
        r.update(forged_receipt(a))
        replacement = make_declaration(
            artifact=a,
            checker_receipt=r,
            utility_kind="WEIGHTED_SUM_MINIMIZE",
            parameters={"weights": {"iid_delta": 0.5, "grouped_delta": 0.5}},
        )
        d.clear()
        d.update(replacement)

    def nonfinite_coordinate(a, r, d):
        a["policies"][0]["objective_mean"]["iid_delta"] = float("inf")
        r.clear()
        r.update(forged_receipt(a))
        replacement = make_declaration(
            artifact=a,
            checker_receipt=r,
            utility_kind="WEIGHTED_SUM_MINIMIZE",
            parameters={"weights": {"iid_delta": 0.5, "grouped_delta": 0.5}},
        )
        d.clear()
        d.update(replacement)

    definitions = (
        ("N01_ARTIFACT_HASH", wrong_artifact_hash, "utility artifact hash mismatch"),
        (
            "N02_RECEIPT_HASH",
            wrong_receipt_hash,
            "utility checker receipt hash mismatch",
        ),
        ("N03_INVALID_RECEIPT", invalid_receipt, "checker receipt is not valid"),
        ("N04_FAKE_USER_SCOPE", fake_user_scope, "scope must be SYNTHETIC_FIXTURE"),
        ("N05_ZERO_WEIGHT", zero_weight, "strictly positive weights"),
        (
            "N06_INCOMPLETE_WEIGHTS",
            incomplete_weights,
            "weighted utility axes mismatch",
        ),
        ("N07_DUPLICATE_PRIORITY", duplicate_priority, "axis permutation"),
        ("N08_SAME_EPSILON_AXIS", same_epsilon_axis, "distinct known axes"),
        ("N09_HIDDEN_TIE_BREAK", hidden_tie_break, "tie policy must return all optima"),
        (
            "N10_SELECTED_IN_DECLARATION",
            selected_in_declaration,
            "utility declaration fields mismatch",
        ),
        ("N11_UNKNOWN_READER", unknown_reader, "unknown or duplicate candidate"),
        (
            "N12_NONFINITE_COORDINATE",
            nonfinite_coordinate,
            "coordinate iid_delta must be finite",
        ),
    )
    return [
        expect_invalid(case_id, artifact, receipt, weighted, mutate, expected)
        for case_id, mutate, expected in definitions
    ]


def require_clean(development: bool) -> None:
    if development:
        return
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    if status:
        raise RuntimeError("official utility-port suite requires clean worktree")


def git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def write_replay(output: Path, cfg: dict[str, Any], head: str) -> None:
    checks = "\n".join(
        f"printf '%s  %s\\n' '{sha(ROOT / rel)}' \"$repo/{rel}\" | sha256sum -c -"
        for rel in SOURCE_FILES
    )
    checks += (
        f"\nprintf '%s  %s\\n' '{cfg['source_interface_manifest_sha256']}' "
        f'"$repo/{cfg["source_budget_path_interface"]}/manifest.json" | sha256sum -c -'
    )
    text = f"""#!/bin/sh
set -eu
repo=$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd)
[ "$(git -C "$repo" rev-parse HEAD)" = "{head}" ] || exit 2
[ -z "$(git -C "$repo" status --porcelain)" ] || exit 2
{checks}
: "${{OUTPUT_DIR:?set OUTPUT_DIR}}"
exec env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 "$repo/venv/bin/python" "$repo/experiments/geometria_proporcional/run_proportional_budget_path_external_utility_port.py" --config "$repo/experiments/geometria_proporcional/configs/proportional_budget_path_external_utility_port_v1.json" --output "$OUTPUT_DIR"
"""
    (output / "replay.sh").write_text(text)
    (output / "replay.sh").chmod(0o755)


def run(cfg: dict[str, Any], output: Path, development: bool) -> None:
    require_clean(development)
    if output.exists():
        raise FileExistsError(output)
    if (
        not development
        and (ROOT / "data/geometria_proporcional").resolve()
        not in output.resolve().parents
    ):
        raise ValueError("official path invalid")
    started = time.monotonic()
    verify_source(cfg)
    fixtures = {
        "tradeoff": build_fixture("tradeoff", TRADEOFF_COORDINATES),
        "singleton": build_fixture("singleton", SINGLETON_COORDINATES),
    }
    cases, declarations, decisions = execute_positive(fixtures)
    metamorphic = metamorphic_checks(fixtures)
    invalid = invalid_suite(*fixtures["tradeoff"])
    if (
        len(fixtures) != cfg["expected_fixtures"]
        or len(decisions) != cfg["expected_decisions"]
        or len(metamorphic) != cfg["expected_metamorphic_checks"]
        or len(invalid) != cfg["expected_invalid_cases"]
    ):
        raise AssertionError("utility-port suite count mismatch")
    summary = {
        "status": "SYNTHETIC_EXTERNAL_UTILITY_PORT_CHECKED",
        "architecture_status": "CANDIDATE_NOT_PROMOTED",
        "fixture_count": len(fixtures),
        "positive_decisions": len(decisions),
        "positive_matches_expected": len(decisions),
        "metamorphic_checks": len(metamorphic),
        "metamorphic_passed": sum(row["status"] == "PASS" for row in metamorphic),
        "invalid_cases": len(invalid),
        "invalid_rejected": sum(row["status"] == "REJECTED" for row in invalid),
        "accepted_scope": "SYNTHETIC_FIXTURE_ONLY",
        "user_utility_status": "NOT_DECLARED",
        "historical_budget_paths_evaluated": 0,
        "decision_authority": "SYNTHETIC_TEST_ONLY",
        "gpu_queried": False,
    }
    output.mkdir(parents=True)
    fixture_rows = [
        {"fixture_id": name, "artifact": artifact, "receipt": receipt}
        for name, (artifact, receipt) in fixtures.items()
    ]
    write_jsonl(output / "fixtures.jsonl", fixture_rows)
    write_jsonl(output / "declarations.jsonl", declarations)
    write_jsonl(output / "decisions.jsonl", decisions)
    write_json(output / "case_manifest.json", cases)
    write_json(output / "metamorphic_results.json", metamorphic)
    write_json(output / "invalid_results.json", invalid)
    write_json(output / "summary.json", summary)
    write_json(output / "resolved_config.json", cfg)
    write_json(
        output / "environment.json",
        {
            "python": platform.python_version(),
            "numpy": importlib.metadata.version("numpy"),
            "scipy": importlib.metadata.version("scipy"),
            "scikit_learn": importlib.metadata.version("scikit-learn"),
            "threads": 1,
            "cuda_visible_devices": "",
            "refit": False,
            "new_views": 0,
            "new_solves": 0,
            "historical_budget_paths_evaluated": 0,
            "gpu_queried": False,
        },
    )
    write_replay(output, cfg, git_head())
    runtime = {
        "elapsed_seconds": time.monotonic() - started,
        "max_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
    }
    if (
        runtime["elapsed_seconds"] > cfg["execution"]["max_seconds"]
        or runtime["max_rss_gib"] > cfg["execution"]["max_rss_gib"]
    ):
        raise RuntimeError("resource budget exceeded")
    write_json(output / "runtime_observation.json", runtime)
    deterministic = sorted(
        path
        for path in output.iterdir()
        if path.is_file()
        and path.name not in {"manifest.json", "runtime_observation.json"}
    )
    write_json(
        output / "manifest.json",
        {
            "schema_version": cfg["schema_version"],
            "git_head": git_head(),
            "source_interface_manifest_sha256": cfg["source_interface_manifest_sha256"],
            "source_hashes": {rel: sha(ROOT / rel) for rel in SOURCE_FILES},
            "deterministic_files": {path.name: sha(path) for path in deterministic},
            "runtime_exclusions": ["runtime_observation.json"],
        },
    )
    print(json.dumps(runtime, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--development", action="store_true")
    args = parser.parse_args()
    run(load_config(args.config.resolve()), args.output.resolve(), args.development)


if __name__ == "__main__":
    main()
