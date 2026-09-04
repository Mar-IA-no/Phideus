#!/usr/bin/env python3
"""Audit power and point-effect transport for frozen abstention thresholds."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments/geometria_proporcional"))

import run_proportional_graph_safe_abstention_gate as safe  # noqa: E402

DEFAULT_CONFIG = ROOT / "experiments/geometria_proporcional/configs/proportional_graph_safe_abstention_power_audit_v1.json"
SOURCE_FILES = (
    "experiments/geometria_proporcional/PLAN_PROPORTIONAL_SAFE_ABSTENTION_POWER_AUDIT_CPU.md",
    "experiments/geometria_proporcional/configs/proportional_graph_safe_abstention_power_audit_v1.json",
    "experiments/geometria_proporcional/run_proportional_graph_safe_abstention_power_audit.py",
    "experiments/geometria_proporcional/run_proportional_graph_safe_abstention_gate.py",
)
REQUIRED_INPUTS = {
    "fresh_mixed_gate": (
        "calibration/gate_models.json",
        "adjudication/alpha_metrics.npz",
        "adjudication/features.npz",
        "adjudication/view_index.json",
    ),
    "safe_abstention_gate": (
        "selection/alpha_metrics.npz",
        "selection/threshold_statistics.npz",
        "selection/thresholds.json",
        "selection/view_index.json",
        "adjudication/alpha_metrics.npz",
        "adjudication/advantages.npz",
        "adjudication/decisions.npz",
        "adjudication/features.npz",
        "adjudication/view_index.json",
    ),
}
DATASETS = ("selection", "fresh_adjudication", "safe_adjudication")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def save_npz(path: Path, values: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{key: values[key] for key in sorted(values)})


def load_config(path: Path) -> dict[str, Any]:
    cfg = json.loads(path.read_text())
    if set(cfg) != {
        "schema_version", "sources", "arms", "families", "bootstrap_replicates",
        "bootstrap_seed", "execution",
    } or cfg["schema_version"] != "proportional-graph-safe-abstention-power-audit-v1":
        raise ValueError("invalid power audit schema")
    if cfg["arms"] != ["raw_generic", "raw_typed", "closure_generic", "closure_typed"]:
        raise ValueError("arms changed")
    if cfg["families"] != list(safe.FAMILIES):
        raise ValueError("families changed")
    if cfg["bootstrap_replicates"] != 4000 or cfg["bootstrap_seed"] != 2026090559:
        raise ValueError("bootstrap contract changed")
    if cfg["execution"] != {"max_seconds": 120, "max_rss_gib": 2.0, "threads": 1}:
        raise ValueError("execution contract changed")
    if set(cfg["sources"]) != set(REQUIRED_INPUTS):
        raise ValueError("source set changed")
    return cfg


def source_root(cfg: dict[str, Any], name: str) -> Path:
    return ROOT / cfg["sources"][name]["path"]


def verify_inputs(cfg: dict[str, Any]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for name, relatives in REQUIRED_INPUTS.items():
        root = source_root(cfg, name)
        manifest_path = root / "manifest.json"
        expected_manifest = cfg["sources"][name]["manifest_sha256"]
        if sha(manifest_path) != expected_manifest:
            raise AssertionError(f"{name} manifest hash mismatch")
        manifest = json.loads(manifest_path.read_text())
        hashes[str(manifest_path.relative_to(ROOT))] = expected_manifest
        for relative in relatives:
            path = root / relative
            actual = sha(path)
            if manifest["deterministic_files"].get(relative) != actual:
                raise AssertionError(f"{name} deterministic input mismatch: {relative}")
            hashes[str(path.relative_to(ROOT))] = actual
    return hashes


def require_clean(development: bool) -> None:
    if not development and subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout:
        raise RuntimeError("official audit requires clean worktree")


def git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout.strip()


def paired_effect_vectors(quotient: np.ndarray, action: np.ndarray) -> dict[str, np.ndarray]:
    """Average neural seeds, then return paired-master deltas against identity."""
    quotient_mean = np.asarray(quotient, dtype=np.float64).mean(axis=0)
    if quotient_mean.shape[1] != len(action) or len(action) % 2:
        raise ValueError("action and paired quotient shapes differ")
    delta = quotient_mean[action, np.arange(len(action))] - quotient_mean[0]
    iid, grouped = delta[0::2], delta[1::2]
    return {"iid": iid, "grouped": grouped, "balanced": 0.5 * (iid + grouped)}


def summarize_vector(values: np.ndarray, bootstrap: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    draws = values[bootstrap].mean(axis=1)
    return {
        "n_masters": int(len(values)),
        "mean": float(values.mean()),
        "ci95": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
    }


def action_fractions(action: np.ndarray) -> dict[str, float | bool]:
    action = np.asarray(action)
    iid_active = float(np.mean(action[0::2] != 0))
    grouped_active = float(np.mean(action[1::2] != 0))
    total_active = float(np.mean(action != 0))
    return {
        "active_total": total_active,
        "active_iid": iid_active,
        "active_grouped": grouped_active,
        "empirical_identity": bool(total_active == 0.0),
    }


def projected_sample_size(n: int, mean_iid: float, simultaneous_q95: float, active_iid: float) -> float | None:
    """Fixed-effect 1/sqrt(n) planning projection, never a confirmatory guarantee."""
    if mean_iid >= 0.0 or simultaneous_q95 <= 0.0 or active_iid <= 0.0:
        return None
    return float(n * (simultaneous_q95 / -mean_iid) ** 2)


def bootstrap_indices(cfg: dict[str, Any], counts: dict[str, int]) -> dict[str, np.ndarray]:
    packed: dict[str, np.ndarray] = {}
    for offset, dataset in enumerate(DATASETS):
        n = counts[dataset]
        rng = np.random.default_rng(cfg["bootstrap_seed"] + offset)
        packed[dataset] = rng.integers(0, n, size=(cfg["bootstrap_replicates"], n))
    return packed


def load_adjudication(root: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(root / "adjudication/alpha_metrics.npz", allow_pickle=False) as saved:
        quotient = saved["quotient_rmse"]
    with np.load(root / "adjudication/features.npz", allow_pickle=False) as saved:
        features = saved["features"].mean(axis=1)
    return quotient, features


def analyze(cfg: dict[str, Any]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    fresh_root = source_root(cfg, "fresh_mixed_gate")
    safe_root = source_root(cfg, "safe_abstention_gate")
    thresholds = json.loads((safe_root / "selection/thresholds.json").read_text())
    gate_models = json.loads((fresh_root / "calibration/gate_models.json").read_text())
    with np.load(safe_root / "selection/threshold_statistics.npz", allow_pickle=False) as saved:
        selection = {key: saved[key] for key in saved.files}
    with np.load(safe_root / "selection/alpha_metrics.npz", allow_pickle=False) as saved:
        selection_quotient = saved["quotient_rmse"]
    fresh_quotient, fresh_features = load_adjudication(fresh_root)
    safe_quotient, safe_features = load_adjudication(safe_root)
    with np.load(safe_root / "adjudication/decisions.npz", allow_pickle=False) as saved:
        stored_decisions = {key: saved[key] for key in saved.files}
    with np.load(safe_root / "adjudication/advantages.npz", allow_pickle=False) as saved:
        stored_advantages = {key: saved[key] for key in saved.files}

    counts = {
        "selection": selection_quotient.shape[-1] // 2,
        "fresh_adjudication": fresh_quotient.shape[-1] // 2,
        "safe_adjudication": safe_quotient.shape[-1] // 2,
    }
    boots = bootstrap_indices(cfg, counts)
    result: dict[str, Any] = {
        "schema_version": cfg["schema_version"],
        "regime": "posthoc_planning_diagnostic_on_opened_artifacts",
        "claims_boundary": {
            "confirmatory": False,
            "new_realization_materialized": False,
            "threshold_refit": False,
            "gpu_used": False,
            "projection_assumption": "fixed_point_effect_and_inverse_sqrt_n_simultaneous_width",
        },
        "dataset_master_counts": counts,
        "arms": {},
    }
    family_aggregates = {
        family: {
            "finite_planning_candidates": 0,
            "planning_candidates_grouped_favorable_both_opened": 0,
            "planning_candidates_balanced_favorable_both_opened": 0,
            "planning_candidates_iid_nonpositive_all_three": 0,
            "planning_candidates_iid_positive_latest": 0,
        }
        for family in cfg["families"]
    }

    for arm_index, arm in enumerate(cfg["arms"]):
        arm_record: dict[str, Any] = {"families": {}}
        for family in cfg["families"]:
            key = f"{arm}|{family}"
            family_record = thresholds["arms"][arm]["families"][family]
            threshold_values = family_record["thresholds"]
            if threshold_values[-1] is not None:
                raise AssertionError("threshold grid lost identity sentinel")
            columns = safe.feature_columns(family)
            report = gate_models["arms"][arm]["models"][family]
            _, fresh_base, fresh_advantage = safe.predicted_action_and_advantage(
                report, fresh_features[arm_index][:, columns]
            )
            _, safe_base, safe_advantage = safe.predicted_action_and_advantage(
                report, safe_features[arm_index][:, columns]
            )
            stored_name = "unconstrained_full" if family == "public_mixed_ridge_gate" else "unconstrained_reduced"
            advantage_name = "full" if family == "public_mixed_ridge_gate" else "reduced"
            if not np.array_equal(safe_base, stored_decisions[stored_name][arm_index]):
                raise AssertionError(f"stored decision mismatch: {key}")
            np.testing.assert_allclose(
                safe_advantage, stored_advantages[advantage_name][arm_index], rtol=0.0, atol=0.0
            )

            candidate_rows = []
            for index, threshold in enumerate(threshold_values[:-1]):
                selection_action = selection[f"{key}|candidate_actions"][index]
                selection_vectors = {
                    "iid": selection[f"{key}|iid_delta_by_candidate"][index],
                    "balanced": selection[f"{key}|balanced_delta_by_candidate"][index],
                }
                selection_vectors["grouped"] = 2.0 * selection_vectors["balanced"] - selection_vectors["iid"]
                fresh_action = safe.threshold_action(fresh_base, fresh_advantage, threshold)
                safe_action = safe.threshold_action(safe_base, safe_advantage, threshold)
                fresh_vectors = paired_effect_vectors(fresh_quotient[arm_index], fresh_action)
                safe_vectors = paired_effect_vectors(safe_quotient[arm_index], safe_action)
                vector_sets = {
                    "selection": selection_vectors,
                    "fresh_adjudication": fresh_vectors,
                    "safe_adjudication": safe_vectors,
                }
                action_sets = {
                    "selection": selection_action,
                    "fresh_adjudication": fresh_action,
                    "safe_adjudication": safe_action,
                }
                effects = {
                    dataset: {
                        cell: summarize_vector(vector_sets[dataset][cell], boots[dataset])
                        for cell in ("iid", "grouped", "balanced")
                    }
                    for dataset in DATASETS
                }
                fractions = {dataset: action_fractions(action_sets[dataset]) for dataset in DATASETS}
                mean_iid = effects["selection"]["iid"]["mean"]
                projection = projected_sample_size(
                    counts["selection"], mean_iid,
                    float(family_record["simultaneous_q95"]),
                    float(fractions["selection"]["active_iid"]),
                )
                row = {
                    "candidate_index": index,
                    "threshold": float(threshold),
                    "selection_simultaneous_q95": float(family_record["simultaneous_q95"]),
                    "projected_masters_fixed_effect": projection,
                    "projected_masters_ceiling": None if projection is None else int(math.ceil(projection)),
                    "effects": effects,
                    "action_fractions": fractions,
                    "point_sign_transport": {
                        cell: [float(effects[dataset][cell]["mean"]) for dataset in DATASETS]
                        for cell in ("iid", "grouped", "balanced")
                    },
                }
                candidate_rows.append(row)

            finite = [row for row in candidate_rows if row["projected_masters_fixed_effect"] is not None]
            planning = min(finite, key=lambda row: (row["projected_masters_fixed_effect"], row["candidate_index"])) if finite else None
            summary = {"planning_candidate_index": None, "planning_candidate": None}
            if planning is not None:
                signs = planning["point_sign_transport"]
                summary = {
                    "planning_candidate_index": planning["candidate_index"],
                    "planning_candidate": planning,
                    "iid_nonpositive_all_three": bool(all(value <= 0.0 for value in signs["iid"])),
                    "iid_positive_in_latest_adjudication": bool(signs["iid"][-1] > 0.0),
                    "grouped_favorable_both_opened": bool(all(value < 0.0 for value in signs["grouped"][1:])),
                    "balanced_favorable_both_opened": bool(all(value < 0.0 for value in signs["balanced"][1:])),
                }
                aggregate = family_aggregates[family]
                aggregate["finite_planning_candidates"] += 1
                aggregate["planning_candidates_grouped_favorable_both_opened"] += int(summary["grouped_favorable_both_opened"])
                aggregate["planning_candidates_balanced_favorable_both_opened"] += int(summary["balanced_favorable_both_opened"])
                aggregate["planning_candidates_iid_nonpositive_all_three"] += int(summary["iid_nonpositive_all_three"])
                aggregate["planning_candidates_iid_positive_latest"] += int(summary["iid_positive_in_latest_adjudication"])
            arm_record["families"][family] = {
                "candidate_count_excluding_identity": len(candidate_rows),
                "simultaneous_q95": float(family_record["simultaneous_q95"]),
                "summary": summary,
                "candidates": candidate_rows,
            }
        result["arms"][arm] = arm_record
    result["family_aggregates"] = family_aggregates
    result["interpretation"] = {
        "power_only_explanation_supported": False,
        "reason": (
            "Full-family candidates have finite fixed-effect projections and stable grouped/balanced point benefit, "
            "but none keeps a nonpositive IID point effect across selection and both opened adjudications; all four "
            "are positive in the latest adjudication. Reduced-family selection means yield no finite candidate."
        ),
        "next_discriminant": (
            "Do not enlarge selection under a fixed-effect assumption alone; design a fresh topology-localization-aware "
            "gate with a scale-matched control and retain prospective IID no-harm adjudication."
        ),
        "architecture_promoted": False,
        "go_no_go": None,
    }
    return result, boots


def write_replay(output: Path, cfg: dict[str, Any], input_hashes: dict[str, str], head: str) -> None:
    checks = "\n".join(
        f"printf '%s  %s\\n' '{digest}' \"$repo/{relative}\" | sha256sum -c -"
        for relative, digest in sorted(input_hashes.items())
    )
    source_checks = "\n".join(
        f"printf '%s  %s\\n' '{sha(ROOT / relative)}' \"$repo/{relative}\" | sha256sum -c -"
        for relative in SOURCE_FILES
    )
    replay = f'''#!/bin/sh
set -eu
repo=$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd)
[ "$(git -C "$repo" rev-parse HEAD)" = "{head}" ] || {{ echo "wrong git HEAD" >&2; exit 2; }}
[ -z "$(git -C "$repo" status --porcelain)" ] || {{ echo "dirty worktree" >&2; exit 2; }}
{source_checks}
{checks}
: "${{OUTPUT_DIR:?set OUTPUT_DIR to a new path}}"
exec env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 "$repo/venv/bin/python" "$repo/experiments/geometria_proporcional/run_proportional_graph_safe_abstention_power_audit.py" --config "$repo/experiments/geometria_proporcional/configs/proportional_graph_safe_abstention_power_audit_v1.json" --output "$OUTPUT_DIR"
'''
    (output / "replay.sh").write_text(replay)
    (output / "replay.sh").chmod(0o755)


def run(cfg: dict[str, Any], output: Path, development: bool) -> None:
    require_clean(development)
    if output.exists():
        raise FileExistsError(output)
    if not development and (ROOT / "data/geometria_proporcional").resolve() not in output.resolve().parents:
        raise ValueError("official output must be below data/geometria_proporcional")
    input_hashes = verify_inputs(cfg)
    output.mkdir(parents=True)
    started = time.monotonic()
    result, boots = analyze(cfg)
    write_json(output / "resolved_config.json", cfg)
    write_json(output / "analysis.json", result)
    save_npz(output / "bootstrap_indices.npz", boots)
    head = git_head()
    write_replay(output, cfg, input_hashes, head)
    deterministic = sorted(path for path in output.iterdir() if path.name not in {"manifest.json", "runtime_observation.json"})
    manifest = {
        "schema_version": cfg["schema_version"],
        "git_head": head,
        "source_hashes": {relative: sha(ROOT / relative) for relative in SOURCE_FILES},
        "input_hashes": input_hashes,
        "deterministic_files": {path.name: sha(path) for path in deterministic},
        "runtime_exclusions": ["runtime_observation.json"],
    }
    write_json(output / "manifest.json", manifest)
    runtime = {
        "elapsed_seconds": time.monotonic() - started,
        "max_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0**2),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "threads": cfg["execution"]["threads"],
    }
    if runtime["elapsed_seconds"] > cfg["execution"]["max_seconds"] or runtime["max_rss_gib"] > cfg["execution"]["max_rss_gib"]:
        raise RuntimeError(f"resource contract exceeded: {runtime}")
    write_json(output / "runtime_observation.json", runtime)
    print(json.dumps(runtime, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--development", action="store_true")
    args = parser.parse_args()
    run(load_config(args.config), args.output.resolve(), args.development)


if __name__ == "__main__":
    main()
