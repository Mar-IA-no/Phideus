#!/usr/bin/env python3
"""Materialize and independently check the candidate BudgetPath interface."""

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
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any, Callable

import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from geometria_proporcional.budget_path_checker import (  # noqa: E402
    BudgetPathChecker,
    BudgetPathViolation,
)
from geometria_proporcional.budget_path_schema import (  # noqa: E402
    BUDGET_FRACTIONS,
    POLICY_IDS,
    SCHEMA_VERSION,
    BudgetPathArtifact,
    array_digest,
    artifact_id,
    canonical_json,
)

DEFAULT_CONFIG = ROOT / (
    "experiments/geometria_proporcional/configs/"
    "proportional_budget_path_typed_interface_v1.json"
)
SOURCE_FILES = (
    "experiments/geometria_proporcional/PLAN_PROPORTIONAL_BUDGET_PATH_TYPED_INTERFACE_CPU.md",
    "experiments/geometria_proporcional/configs/proportional_budget_path_typed_interface_v1.json",
    "experiments/geometria_proporcional/run_proportional_budget_path_typed_interface.py",
    "src/geometria_proporcional/budget_path_schema.py",
    "src/geometria_proporcional/budget_path_checker.py",
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
AUTHORITY = {
    "artifact_status": "CHECKABLE_CANDIDATE",
    "formal_claim_status": "STRUCTURE_ONLY",
    "empirical_claim_status": "OPENED_POSTHOC",
    "physical_authority_status": "NOT_CLAIMED",
    "decision_status": "UNRESOLVED",
}
UTILITY_BOUNDARY = {
    "status": "ABSENT_EXTERNAL_REQUIRED",
    "embedded_utility_fields": [],
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


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
        "source_mean_ranking_attribution",
        "source_attribution_manifest_sha256",
        "source_pareto_transport",
        "source_pareto_manifest_sha256",
        "source_pairwise_dominance",
        "source_pairwise_manifest_sha256",
        "arms",
        "proposal_regimes",
        "roles",
        "main_families",
        "control_family",
        "control_replicates",
        "expected_artifacts",
        "expected_mutations",
        "bootstrap_replicates",
        "numerical_zero_tolerance",
        "execution",
    }
    if set(cfg) != expected:
        raise ValueError("invalid BudgetPath run schema keys")
    if cfg["schema_version"] != "proportional-budget-path-typed-interface-run-v1":
        raise ValueError("invalid BudgetPath run schema")
    if cfg["arms"] != ["raw_generic", "raw_typed", "closure_generic", "closure_typed"]:
        raise ValueError("arms changed")
    if cfg["proposal_regimes"] != ["fixed_alpha_0.25", "r368_common"]:
        raise ValueError("proposal regimes changed")
    if cfg["roles"] != ["policy_selection", "adjudication"]:
        raise ValueError("roles changed")
    if cfg["main_families"] != ["reduced_mean", "public_base_mean", "topology_mean"]:
        raise ValueError("main families changed")
    if cfg["control_family"] != "topology_permuted_mean":
        raise ValueError("control family changed")
    if cfg["control_replicates"] != 16 or cfg["expected_artifacts"] != 608:
        raise ValueError("materialization count changed")
    if cfg["expected_mutations"] != 8 or cfg["bootstrap_replicates"] != 2000:
        raise ValueError("audit count changed")
    if cfg["numerical_zero_tolerance"] != 1e-12:
        raise ValueError("numerical tolerance changed")
    if cfg["execution"] != {"max_seconds": 300, "max_rss_gib": 4.0}:
        raise ValueError("execution contract changed")
    return cfg


def action_key(regime: str, arm: str, fraction: float, role: str, family: str) -> str:
    middle = "action" if role == "policy_selection" else "ranked|action"
    return f"{regime}|{arm}|budget={fraction:.2f}|{middle}|{family}"


def objective_key(
    regime: str,
    arm: str,
    role: str,
    family: str,
    replicate: int | None,
) -> str:
    suffix = family if replicate is None else f"{family}={replicate}"
    return f"{regime}|{arm}|{role}|{suffix}|objectives"


def source_front(
    report: dict[str, Any],
    regime: str,
    arm: str,
    role: str,
    family: str,
    replicate: int | None,
) -> dict[str, Any]:
    role_record = report["proposals"][regime]["arms"][arm]["roles"][role]
    if replicate is None:
        return role_record["families"][family]
    row = role_record[family][replicate]
    if row["replicate"] != replicate:
        raise AssertionError("R371 control ordering changed")
    return row


def source_pairs(
    report: dict[str, Any],
    regime: str,
    arm: str,
    role: str,
    family: str,
    replicate: int | None,
) -> list[dict[str, Any]]:
    role_record = report["proposals"][regime]["arms"][arm]["roles"][role]
    if replicate is None:
        return role_record["families"][family]
    row = role_record[family][replicate]
    if row["replicate"] != replicate:
        raise AssertionError("R372 control ordering changed")
    return row["pairs"]


def build_artifact(
    cfg: dict[str, Any],
    cohort: str,
    regime: str,
    arm: str,
    role: str,
    family: str,
    replicate: int | None,
    actions_pack: dict[str, np.ndarray],
    objectives_pack: dict[str, np.ndarray],
    pareto_report: dict[str, Any],
    pairwise_report: dict[str, Any],
) -> dict[str, Any]:
    action_keys = [
        action_key(regime, arm, fraction, role, family)
        for fraction in BUDGET_FRACTIONS[1:]
    ]
    raw_actions = [actions_pack[key] for key in action_keys]
    if replicate is not None:
        raw_actions = [value[replicate] for value in raw_actions]
    actions = [np.zeros_like(raw_actions[0]), *raw_actions]
    key = objective_key(regime, arm, role, family, replicate)
    values = objectives_pack[key]
    front = source_front(pareto_report, regime, arm, role, family, replicate)
    pair_rows = source_pairs(pairwise_report, regime, arm, role, family, replicate)
    lineage = {
        "cohort": cohort,
        "role": role,
        "proposal_regime": regime,
        "arm": arm,
        "family": family,
        "control_replicate": replicate,
        "source_attribution_manifest_sha256": cfg["source_attribution_manifest_sha256"],
        "source_pareto_manifest_sha256": cfg["source_pareto_manifest_sha256"],
        "source_pairwise_manifest_sha256": cfg["source_pairwise_manifest_sha256"],
        "objective_key": key,
        "action_keys": action_keys,
    }
    front_ids = front["front_policy_ids"]
    policies = []
    means = values.mean(axis=2)
    for index, policy_id in enumerate(POLICY_IDS):
        coordinate = front["coordinates"][policy_id]
        policies.append(
            {
                "policy_id": policy_id,
                "budget_fraction": BUDGET_FRACTIONS[index],
                "action_count": int(np.count_nonzero(actions[index])),
                "action_sha256": array_digest(actions[index]),
                "objective_master_count": int(values.shape[2]),
                "objective_sha256": array_digest(values[index]),
                "objective_mean": {
                    "iid_delta": float(means[index, 0]),
                    "grouped_delta": float(means[index, 1]),
                },
                "front_member": policy_id in front_ids,
                "bootstrap_front_frequency": coordinate["bootstrap_front_frequency"],
            }
        )
    pairs = []
    if len(pair_rows) != 21:
        raise AssertionError("R372 pair count changed")
    for row, (base, expansion) in zip(
        pair_rows, combinations(range(len(POLICY_IDS)), 2), strict=True
    ):
        increment = values[expansion] - values[base]
        pairs.append(
            {
                "pair_id": row["pair_id"],
                "base_policy": row["base_policy"],
                "expansion_policy": row["expansion_policy"],
                "adjacent": row["adjacent"],
                "mean_increment": row["mean_increment"],
                "increment_sha256": array_digest(increment),
                "point_status": row["point_status"],
                "bootstrap_status_frequency": row["bootstrap_status_frequency"],
            }
        )
    artifact = BudgetPathArtifact(
        schema_version=SCHEMA_VERSION,
        artifact_id=artifact_id(lineage),
        lineage=lineage,
        axes=copy.deepcopy(AXES),
        policies=policies,
        pairs=pairs,
        reader={"kind": "PARETO_SET", "policy_ids": front_ids},
        authority=copy.deepcopy(AUTHORITY),
        utility_boundary=copy.deepcopy(UTILITY_BOUNDARY),
    )
    return artifact.to_dict()


def materialize(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    attribution_root = ROOT / cfg["source_mean_ranking_attribution"]
    pareto_root = ROOT / cfg["source_pareto_transport"]
    pairwise_root = ROOT / cfg["source_pairwise_dominance"]
    artifacts: list[dict[str, Any]] = []
    for cohort in ("A", "B"):
        objectives = read_npz(pareto_root / f"cohort_{cohort.lower()}_objectives.npz")
        pareto_report = json.loads(
            (pareto_root / f"cohort_{cohort.lower()}_pareto.json").read_text()
        )
        pairwise_report = json.loads(
            (pairwise_root / f"cohort_{cohort.lower()}_pairwise.json").read_text()
        )
        action_packs = {
            "policy_selection": read_npz(
                attribution_root / f"cohort_{cohort.lower()}_selection.npz"
            ),
            "adjudication": read_npz(
                attribution_root / f"cohort_{cohort.lower()}_adjudication.npz"
            ),
        }
        for regime in cfg["proposal_regimes"]:
            for arm in cfg["arms"]:
                for role in cfg["roles"]:
                    for family in cfg["main_families"]:
                        artifacts.append(
                            build_artifact(
                                cfg,
                                cohort,
                                regime,
                                arm,
                                role,
                                family,
                                None,
                                action_packs[role],
                                objectives,
                                pareto_report,
                                pairwise_report,
                            )
                        )
                    for replicate in range(cfg["control_replicates"]):
                        artifacts.append(
                            build_artifact(
                                cfg,
                                cohort,
                                regime,
                                arm,
                                role,
                                cfg["control_family"],
                                replicate,
                                action_packs[role],
                                objectives,
                                pareto_report,
                                pairwise_report,
                            )
                        )
    if len(artifacts) != cfg["expected_artifacts"]:
        raise AssertionError("BudgetPath materialization count mismatch")
    if len({row["artifact_id"] for row in artifacts}) != len(artifacts):
        raise AssertionError("BudgetPath artifact ids are not unique")
    return artifacts


def expect_rejection(
    mutation_id: str,
    checker: BudgetPathChecker,
    artifact: dict[str, Any],
    mutate: Callable[[dict[str, Any]], None],
    expected: str,
) -> dict[str, str]:
    changed = copy.deepcopy(artifact)
    mutate(changed)
    try:
        checker.check(changed)
    except BudgetPathViolation as exc:
        message = str(exc)
        if expected not in message:
            raise AssertionError(
                f"{mutation_id}: wrong rejection {message!r}, expected {expected!r}"
            ) from exc
        return {
            "mutation_id": mutation_id,
            "design_region": "PROTOCOL_INVALID",
            "status": "REJECTED",
            "expected_signal": expected,
            "observed_signal": message,
        }
    raise AssertionError(f"{mutation_id}: checker accepted protocol-invalid artifact")


def mutation_suite(
    checker: BudgetPathChecker, artifact: dict[str, Any]
) -> list[dict[str, str]]:
    def add_selected(row: dict[str, Any]) -> None:
        row["selected_policy_id"] = "budget_0.40"

    def add_utility(row: dict[str, Any]) -> None:
        row["utility_boundary"]["iid_weight"] = 0.5

    def swap_axis(row: dict[str, Any]) -> None:
        row["axes"][0]["population"] = "grouped"

    def alter_action_hash(row: dict[str, Any]) -> None:
        row["policies"][0]["action_sha256"] = "0" * 64

    def alter_objective(row: dict[str, Any]) -> None:
        row["policies"][0]["objective_mean"]["iid_delta"] += 1.0

    def alter_front(row: dict[str, Any]) -> None:
        row["policies"][0]["front_member"] = not row["policies"][0]["front_member"]

    def alter_pair(row: dict[str, Any]) -> None:
        current = row["pairs"][0]["point_status"]
        row["pairs"][0]["point_status"] = (
            "BASE_DOMINATES" if current != "BASE_DOMINATES" else "EXPANSION_DOMINATES"
        )

    def alter_manifest(row: dict[str, Any]) -> None:
        row["lineage"]["source_pareto_manifest_sha256"] = "f" * 64

    definitions = (
        ("M01_SELECTED_POLICY_INJECTION", add_selected, "artifact schema mismatch"),
        ("M02_EMBEDDED_UTILITY", add_utility, "utility boundary mismatch"),
        ("M03_AXIS_SEMANTICS_SWAP", swap_axis, "axis contract mismatch"),
        ("M04_ACTION_HASH_TAMPER", alter_action_hash, "action hash mismatch"),
        ("M05_OBJECTIVE_MEAN_TAMPER", alter_objective, "objective mean mismatch"),
        ("M06_FRONT_MEMBERSHIP_TAMPER", alter_front, "front membership mismatch"),
        ("M07_PAIR_STATUS_TAMPER", alter_pair, "pair status mismatch"),
        (
            "M08_SOURCE_MANIFEST_TAMPER",
            alter_manifest,
            "source manifest lineage mismatch",
        ),
    )
    return [
        expect_rejection(mutation_id, checker, artifact, mutate, expected)
        for mutation_id, mutate, expected in definitions
    ]


def recursive_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        return set(value) | set().union(
            *(recursive_keys(item) for item in value.values())
        )
    if isinstance(value, list):
        return (
            set().union(*(recursive_keys(item) for item in value)) if value else set()
        )
    return set()


def build_summary(
    cfg: dict[str, Any],
    artifacts: list[dict[str, Any]],
    receipts: list[dict[str, Any]],
    mutations: list[dict[str, str]],
) -> dict[str, Any]:
    forbidden = {
        "selected_policy_id",
        "utility_weight",
        "scalar_score",
        "recommendation",
    }
    leaked = sorted(
        forbidden & set().union(*(recursive_keys(row) for row in artifacts))
    )
    if leaked:
        raise AssertionError(f"canonical BudgetPath leaks decision fields: {leaked}")
    front_sizes = Counter(receipt["front_size"] for receipt in receipts)
    family_counts = Counter(row["lineage"]["family"] for row in artifacts)
    return {
        "status": "BUDGET_PATH_STRUCTURALLY_CHECKED",
        "architecture_status": "CANDIDATE_NOT_PROMOTED",
        "artifact_count": len(artifacts),
        "checker_valid_count": sum(
            receipt["checker_status"] == "VALID" for receipt in receipts
        ),
        "mutation_count": len(mutations),
        "mutation_rejected_count": sum(
            row["status"] == "REJECTED" for row in mutations
        ),
        "nested_path_policy_count": len(POLICY_IDS),
        "pair_count_per_artifact": 21,
        "family_counts": dict(sorted(family_counts.items())),
        "front_size_distribution": {
            str(key): value for key, value in sorted(front_sizes.items())
        },
        "canonical_forbidden_decision_fields": leaked,
        "utility_status": "ABSENT_EXTERNAL_REQUIRED",
        "empirical_authority": "OPENED_POSTHOC_ONLY",
        "physical_authority": "NOT_CLAIMED",
        "decision_status": "UNRESOLVED",
        "gpu_queried": False,
        "expected_artifacts": cfg["expected_artifacts"],
    }


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
        raise RuntimeError("official BudgetPath audit requires clean worktree")


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
    for hash_key, path_key in (
        ("source_attribution_manifest_sha256", "source_mean_ranking_attribution"),
        ("source_pareto_manifest_sha256", "source_pareto_transport"),
        ("source_pairwise_manifest_sha256", "source_pairwise_dominance"),
    ):
        checks += (
            f"\nprintf '%s  %s\\n' '{cfg[hash_key]}' "
            f'"$repo/{cfg[path_key]}/manifest.json" | sha256sum -c -'
        )
    text = f"""#!/bin/sh
set -eu
repo=$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd)
[ "$(git -C "$repo" rev-parse HEAD)" = "{head}" ] || exit 2
[ -z "$(git -C "$repo" status --porcelain)" ] || exit 2
{checks}
: "${{OUTPUT_DIR:?set OUTPUT_DIR}}"
exec env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 "$repo/venv/bin/python" "$repo/experiments/geometria_proporcional/run_proportional_budget_path_typed_interface.py" --config "$repo/experiments/geometria_proporcional/configs/proportional_budget_path_typed_interface_v1.json" --output "$OUTPUT_DIR"
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
    checker = BudgetPathChecker(ROOT, cfg)
    artifacts = materialize(cfg)
    receipts = [checker.check(artifact) for artifact in artifacts]
    mutations = mutation_suite(checker, artifacts[0])
    if len(mutations) != cfg["expected_mutations"]:
        raise AssertionError("mutation suite count mismatch")
    summary = build_summary(cfg, artifacts, receipts, mutations)
    output.mkdir(parents=True)
    write_jsonl(output / "budget_paths.jsonl", artifacts)
    write_jsonl(output / "checker_receipts.jsonl", receipts)
    write_json(output / "mutation_results.json", mutations)
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
            "source_attribution_manifest_sha256": cfg[
                "source_attribution_manifest_sha256"
            ],
            "source_pareto_manifest_sha256": cfg["source_pareto_manifest_sha256"],
            "source_pairwise_manifest_sha256": cfg["source_pairwise_manifest_sha256"],
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
