#!/usr/bin/env python3
"""Audit pairwise budget dominance and transport on frozen R371 objectives."""

from __future__ import annotations

import argparse
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
from typing import Any

import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments/geometria_proporcional")]
import run_proportional_graph_frozen_adapters as frozen  # noqa: E402
import run_proportional_graph_mean_ranking_attribution as attribution  # noqa: E402
import run_proportional_graph_mean_ranking_pareto_transport as pareto  # noqa: E402

DEFAULT_CONFIG = ROOT / (
    "experiments/geometria_proporcional/configs/"
    "proportional_graph_mean_ranking_pairwise_dominance_v1.json"
)
SOURCE_FILES = (
    "experiments/geometria_proporcional/PLAN_PROPORTIONAL_MEAN_RANKING_PAIRWISE_DOMINANCE_CPU.md",
    "experiments/geometria_proporcional/configs/proportional_graph_mean_ranking_pairwise_dominance_v1.json",
    "experiments/geometria_proporcional/run_proportional_graph_mean_ranking_pairwise_dominance.py",
    "experiments/geometria_proporcional/run_proportional_graph_mean_ranking_pareto_transport.py",
    "experiments/geometria_proporcional/run_proportional_graph_mean_ranking_attribution.py",
    "experiments/geometria_proporcional/run_proportional_graph_frozen_adapters.py",
)
MAIN = ("reduced_mean", "public_base_mean", "topology_mean")
CONTROL = "topology_permuted_mean"
STATES = (
    "EXPANSION_DOMINATES",
    "BASE_DOMINATES",
    "IID_COST_GROUPED_GAIN",
    "IID_GAIN_GROUPED_COST",
    "EQUIVALENT",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def read_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def load_config(path: Path) -> dict[str, Any]:
    cfg = json.loads(path.read_text())
    expected = {
        "schema_version",
        "source_pareto_transport",
        "source_pareto_manifest_sha256",
        "source_mean_ranking_attribution",
        "source_attribution_manifest_sha256",
        "arms",
        "proposal_regimes",
        "primary_proposal",
        "policy_ids",
        "control_replicates",
        "bootstrap_replicates",
        "numerical_zero_tolerance",
        "execution",
    }
    if set(cfg) != expected:
        raise ValueError("invalid pairwise schema keys")
    if cfg["schema_version"] != "proportional-graph-mean-ranking-pairwise-dominance-v1":
        raise ValueError("invalid pairwise schema")
    if cfg["arms"] != ["raw_generic", "raw_typed", "closure_generic", "closure_typed"]:
        raise ValueError("arms changed")
    if cfg["proposal_regimes"] != ["fixed_alpha_0.25", "r368_common"]:
        raise ValueError("proposal regimes changed")
    if cfg["primary_proposal"] != "fixed_alpha_0.25":
        raise ValueError("primary proposal changed")
    if cfg["policy_ids"] != list(pareto.POLICY_IDS):
        raise ValueError("policy path changed")
    if cfg["control_replicates"] != 16 or cfg["bootstrap_replicates"] != 2000:
        raise ValueError("replicate contract changed")
    if cfg["numerical_zero_tolerance"] != 1e-12:
        raise ValueError("numerical tolerance changed")
    if cfg["execution"] != {"max_seconds": 300, "max_rss_gib": 4.0}:
        raise ValueError("execution contract changed")
    return cfg


def verify_manifest(root: Path, expected: str, label: str) -> dict[str, Any]:
    if sha(root / "manifest.json") != expected:
        raise AssertionError(f"{label} manifest mismatch")
    manifest = json.loads((root / "manifest.json").read_text())
    for rel, digest in manifest["deterministic_files"].items():
        if sha(root / rel) != digest:
            raise AssertionError(f"{label} deterministic mismatch: {rel}")
    return manifest


def verify_sources(cfg: dict[str, Any]) -> dict[str, Any]:
    pareto_root = ROOT / cfg["source_pareto_transport"]
    attribution_root = ROOT / cfg["source_mean_ranking_attribution"]
    verify_manifest(pareto_root, cfg["source_pareto_manifest_sha256"], "R371")
    verify_manifest(attribution_root, cfg["source_attribution_manifest_sha256"], "R369")
    pareto_cfg = pareto.load_config(pareto_root / "resolved_config.json")
    pareto.verify_sources(pareto_cfg)
    if (
        pareto_cfg["arms"] != cfg["arms"]
        or pareto_cfg["proposal_regimes"] != cfg["proposal_regimes"]
    ):
        raise AssertionError("R371 factorial changed")
    return pareto_cfg


def action_key(regime: str, arm: str, fraction: float, role: str, family: str) -> str:
    middle = "action" if role == "policy_selection" else "ranked|action"
    return f"{regime}|{arm}|budget={fraction:.2f}|{middle}|{family}"


def assert_nested_actions(
    cfg: dict[str, Any], role: str, actions: dict[str, np.ndarray]
) -> int:
    checks = 0
    fractions = (0.01, 0.02, 0.05, 0.10, 0.20, 0.40)
    for regime in cfg["proposal_regimes"]:
        for arm in cfg["arms"]:
            for family in MAIN:
                path = [
                    actions[action_key(regime, arm, f, role, family)] for f in fractions
                ]
                for smaller, larger in zip(path, path[1:]):
                    if np.any((smaller != 0) & (larger != smaller)):
                        raise AssertionError(
                            f"non-nested ranked path: {regime}/{arm}/{role}/{family}"
                        )
                    checks += 1
            path = [
                actions[action_key(regime, arm, f, role, CONTROL)] for f in fractions
            ]
            for smaller, larger in zip(path, path[1:]):
                for replicate in range(cfg["control_replicates"]):
                    if np.any(
                        (smaller[replicate] != 0)
                        & (larger[replicate] != smaller[replicate])
                    ):
                        raise AssertionError(
                            f"non-nested control path: {regime}/{arm}/{role}/{replicate}"
                        )
                    checks += 1
    return checks


def classify_one(delta: np.ndarray, tolerance: float) -> str:
    iid, grouped = np.asarray(delta, dtype=float)
    iid_zero = abs(iid) <= tolerance
    grouped_zero = abs(grouped) <= tolerance
    if iid_zero and grouped_zero:
        return "EQUIVALENT"
    if iid <= tolerance and grouped <= tolerance:
        return "EXPANSION_DOMINATES"
    if iid >= -tolerance and grouped >= -tolerance:
        return "BASE_DOMINATES"
    if iid > tolerance and grouped < -tolerance:
        return "IID_COST_GROUPED_GAIN"
    if iid < -tolerance and grouped > tolerance:
        return "IID_GAIN_GROUPED_COST"
    raise AssertionError(f"unclassified pairwise delta: {delta}")


def classify_many(delta: np.ndarray, tolerance: float) -> list[str]:
    return [classify_one(row, tolerance) for row in np.asarray(delta, dtype=float)]


def pairwise_records(
    values: np.ndarray, bootstrap: np.ndarray, tolerance: float
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    records: list[dict[str, Any]] = []
    increments: dict[str, np.ndarray] = {}
    for base, expansion in combinations(range(len(pareto.POLICY_IDS)), 2):
        delta = values[expansion] - values[base]
        mean = delta.mean(axis=1)
        bootstrap_means = delta[:, bootstrap].mean(axis=2).T
        counts = Counter(classify_many(bootstrap_means, tolerance))
        pair_id = f"{pareto.POLICY_IDS[base]}->{pareto.POLICY_IDS[expansion]}"
        records.append(
            {
                "pair_id": pair_id,
                "base_policy": pareto.POLICY_IDS[base],
                "expansion_policy": pareto.POLICY_IDS[expansion],
                "adjacent": expansion == base + 1,
                "mean_increment": {
                    "iid": float(mean[0]),
                    "grouped": float(mean[1]),
                },
                "point_status": classify_one(mean, tolerance),
                "bootstrap_status_frequency": {
                    state: counts[state] / len(bootstrap) for state in STATES
                },
            }
        )
        increments[pair_id] = delta
    return records, increments


def objective_key(
    regime: str, arm: str, role: str, family: str, replicate: int | None = None
) -> str:
    suffix = family if replicate is None else f"{family}={replicate}"
    return f"{regime}|{arm}|{role}|{suffix}|objectives"


def audit_cohort(
    cfg: dict[str, Any], name: str
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    pareto_root = ROOT / cfg["source_pareto_transport"]
    attribution_root = ROOT / cfg["source_mean_ranking_attribution"]
    objectives = read_npz(pareto_root / f"cohort_{name.lower()}_objectives.npz")
    action_packs = {
        "policy_selection": read_npz(
            attribution_root / f"cohort_{name.lower()}_selection.npz"
        ),
        "adjudication": read_npz(
            attribution_root / f"cohort_{name.lower()}_adjudication.npz"
        ),
    }
    nested_checks = {
        role: assert_nested_actions(cfg, role, pack)
        for role, pack in action_packs.items()
    }
    output: dict[str, Any] = {
        "cohort": name,
        "nested_action_checks": nested_checks,
        "proposals": {},
    }
    saved: dict[str, np.ndarray] = {}
    for regime in cfg["proposal_regimes"]:
        regime_record: dict[str, Any] = {"arms": {}}
        for arm in cfg["arms"]:
            arm_record: dict[str, Any] = {"roles": {}}
            for role in ("policy_selection", "adjudication"):
                bootstrap = objectives[f"{role}|bootstrap"]
                role_record: dict[str, Any] = {"families": {}, CONTROL: []}
                for family in MAIN:
                    values = objectives[objective_key(regime, arm, role, family)]
                    records, increments = pairwise_records(
                        values, bootstrap, cfg["numerical_zero_tolerance"]
                    )
                    role_record["families"][family] = records
                    for pair_id, delta in increments.items():
                        saved[f"{regime}|{arm}|{role}|{family}|{pair_id}"] = delta
                for replicate in range(cfg["control_replicates"]):
                    values = objectives[
                        objective_key(regime, arm, role, CONTROL, replicate)
                    ]
                    records, increments = pairwise_records(
                        values, bootstrap, cfg["numerical_zero_tolerance"]
                    )
                    role_record[CONTROL].append(
                        {"replicate": replicate, "pairs": records}
                    )
                    for pair_id, delta in increments.items():
                        saved[
                            f"{regime}|{arm}|{role}|{CONTROL}={replicate}|{pair_id}"
                        ] = delta
                arm_record["roles"][role] = role_record
            regime_record["arms"][arm] = arm_record
        output["proposals"][regime] = regime_record
    return output, saved


def index_pairs(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {record["pair_id"]: record for record in records}


def transition_record(
    selection: dict[str, Any], adjudication: dict[str, Any]
) -> dict[str, Any]:
    left = np.asarray(
        [selection["bootstrap_status_frequency"][state] for state in STATES]
    )
    right = np.asarray(
        [adjudication["bootstrap_status_frequency"][state] for state in STATES]
    )
    return {
        "pair_id": selection["pair_id"],
        "adjacent": selection["adjacent"],
        "selection_status": selection["point_status"],
        "adjudication_status": adjudication["point_status"],
        "point_status_agrees": selection["point_status"]
        == adjudication["point_status"],
        "bootstrap_frequency_l1": float(np.abs(left - right).sum()),
        "mean_increment_change": {
            axis: adjudication["mean_increment"][axis]
            - selection["mean_increment"][axis]
            for axis in ("iid", "grouped")
        },
    }


def summarize_values(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "min": float(array.min()),
        "max": float(array.max()),
        "values": array.tolist(),
    }


def empty_counts() -> dict[str, int]:
    return {state: 0 for state in STATES}


def build_analysis(cfg: dict[str, Any], reports: dict[str, Any]) -> dict[str, Any]:
    analysis: dict[str, Any] = {
        "status": "OPENED_PAIRWISE_DOMINANCE_AUDIT",
        "authority": "post hoc pairwise path diagnostic; no utility, cutoff, policy selection, promotion, or GO/NO-GO",
        "proposals": {},
    }
    for regime in cfg["proposal_regimes"]:
        aggregate = {
            "topology_point_status": {
                role: empty_counts() for role in ("policy_selection", "adjudication")
            },
            "topology_adjacent_point_status": {
                role: empty_counts() for role in ("policy_selection", "adjudication")
            },
            "topology_transition": {
                left: {right: 0 for right in STATES} for left in STATES
            },
            "topology_adjacent_transition": {
                left: {right: 0 for right in STATES} for left in STATES
            },
            "topology_cross_cohort_agreement": {
                role: [] for role in ("policy_selection", "adjudication")
            },
            "topology_bootstrap_l1": [],
            "topology_adjacent_bootstrap_l1": [],
            "control_transition": {
                left: {right: 0 for right in STATES} for left in STATES
            },
            "control_bootstrap_l1": [],
        }
        arms: dict[str, Any] = {}
        for arm in cfg["arms"]:
            topology_by_cohort: dict[str, Any] = {}
            for cohort in ("A", "B"):
                row = reports[cohort]["proposals"][regime]["arms"][arm]["roles"]
                selection = index_pairs(
                    row["policy_selection"]["families"]["topology_mean"]
                )
                adjudication = index_pairs(
                    row["adjudication"]["families"]["topology_mean"]
                )
                transitions = [
                    transition_record(selection[pair_id], adjudication[pair_id])
                    for pair_id in selection
                ]
                topology_by_cohort[cohort] = {
                    "policy_selection": list(selection.values()),
                    "adjudication": list(adjudication.values()),
                    "transport": transitions,
                }
                for role, records in (
                    ("policy_selection", selection.values()),
                    ("adjudication", adjudication.values()),
                ):
                    for record in records:
                        aggregate["topology_point_status"][role][
                            record["point_status"]
                        ] += 1
                        if record["adjacent"]:
                            aggregate["topology_adjacent_point_status"][role][
                                record["point_status"]
                            ] += 1
                for transition in transitions:
                    aggregate["topology_transition"][transition["selection_status"]][
                        transition["adjudication_status"]
                    ] += 1
                    aggregate["topology_bootstrap_l1"].append(
                        transition["bootstrap_frequency_l1"]
                    )
                    if transition["adjacent"]:
                        aggregate["topology_adjacent_transition"][
                            transition["selection_status"]
                        ][transition["adjudication_status"]] += 1
                        aggregate["topology_adjacent_bootstrap_l1"].append(
                            transition["bootstrap_frequency_l1"]
                        )
                for replicate in range(cfg["control_replicates"]):
                    control_selection = index_pairs(
                        row["policy_selection"][CONTROL][replicate]["pairs"]
                    )
                    control_adjudication = index_pairs(
                        row["adjudication"][CONTROL][replicate]["pairs"]
                    )
                    for pair_id in control_selection:
                        transition = transition_record(
                            control_selection[pair_id], control_adjudication[pair_id]
                        )
                        aggregate["control_transition"][transition["selection_status"]][
                            transition["adjudication_status"]
                        ] += 1
                        aggregate["control_bootstrap_l1"].append(
                            transition["bootstrap_frequency_l1"]
                        )
            for role in ("policy_selection", "adjudication"):
                left = index_pairs(topology_by_cohort["A"][role])
                right = index_pairs(topology_by_cohort["B"][role])
                agreements = [
                    left[pair_id]["point_status"] == right[pair_id]["point_status"]
                    for pair_id in left
                ]
                aggregate["topology_cross_cohort_agreement"][role].append(
                    float(np.mean(agreements))
                )
            arms[arm] = topology_by_cohort
        for key in (
            "topology_bootstrap_l1",
            "topology_adjacent_bootstrap_l1",
            "control_bootstrap_l1",
        ):
            aggregate[key] = summarize_values(aggregate[key])
        for role in ("policy_selection", "adjudication"):
            aggregate["topology_cross_cohort_agreement"][role] = summarize_values(
                aggregate["topology_cross_cohort_agreement"][role]
            )
        analysis["proposals"][regime] = {"aggregate": aggregate, "arms": arms}
    return analysis


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
        raise RuntimeError("official pairwise audit requires clean worktree")


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
        f"\nprintf '%s  %s\\n' '{cfg['source_pareto_manifest_sha256']}' "
        f'"$repo/{cfg["source_pareto_transport"]}/manifest.json" | sha256sum -c -'
    )
    checks += (
        f"\nprintf '%s  %s\\n' '{cfg['source_attribution_manifest_sha256']}' "
        f'"$repo/{cfg["source_mean_ranking_attribution"]}/manifest.json" | sha256sum -c -'
    )
    text = f"""#!/bin/sh
set -eu
repo=$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd)
[ "$(git -C "$repo" rev-parse HEAD)" = "{head}" ] || exit 2
[ -z "$(git -C "$repo" status --porcelain)" ] || exit 2
{checks}
: "${{OUTPUT_DIR:?set OUTPUT_DIR}}"
exec env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 "$repo/venv/bin/python" "$repo/experiments/geometria_proporcional/run_proportional_graph_mean_ranking_pairwise_dominance.py" --config "$repo/experiments/geometria_proporcional/configs/proportional_graph_mean_ranking_pairwise_dominance_v1.json" --output "$OUTPUT_DIR"
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
    verify_sources(cfg)
    output.mkdir(parents=True)
    reports: dict[str, Any] = {}
    for cohort in ("A", "B"):
        report, increments = audit_cohort(cfg, cohort)
        reports[cohort] = report
        write_json(output / f"cohort_{cohort.lower()}_pairwise.json", report)
        frozen.save_npz(output / f"cohort_{cohort.lower()}_increments.npz", increments)
    write_json(output / "analysis.json", build_analysis(cfg, reports))
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
    write_json(output / "resolved_config.json", cfg)
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
            "source_pareto_manifest_sha256": cfg["source_pareto_manifest_sha256"],
            "source_attribution_manifest_sha256": cfg[
                "source_attribution_manifest_sha256"
            ],
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
