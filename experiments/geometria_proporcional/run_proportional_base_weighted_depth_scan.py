#!/usr/bin/env python3
"""CPU-only depth calibration for the learned-base-weight IRLS surrogate."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import resource
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
CHECKER_PATH = ROOT / "experiments/geometria_proporcional/check_proportional_dual_native_freeze.py"
DEFAULT_OUTPUT = ROOT / "data/geometria_proporcional/proportional_dual_native_freeze_v1/depth_scan_v1"
DEPTHS = [64, 96, 128, 160, 192, 256]


def load_checker() -> Any:
    spec = importlib.util.spec_from_file_location("dual_checker_depth_scan", CHECKER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load checker")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def peak_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value * 1024 if sys.platform != "darwin" else value)


def calculate(checker: Any, relational: dict[str, Any]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    from geometria_proporcional.proportional_graph_contract import (
        ProportionalGraphConfig,
        generate_graph_views,
        solve_huber_irls,
    )

    recipe = relational["k64_conformance"]
    executor = relational["executors"]["irls"]
    graph_config = ProportionalGraphConfig(
        masters=64,
        train_fraction=0.5,
        calibration_fraction=0.125,
        validation_fraction=0.125,
        n_min=8,
        n_max=16,
        noise_sigma=0.04,
        corruption_rate=0.15,
        corruption_amplitude_min=0.6,
        corruption_amplitude_max=1.4,
        seed=int(recipe["seed"]),
    )
    all_views = generate_graph_views(graph_config)
    grouped = sorted(
        [view for view in all_views if view.private.corruption_mechanism == "grouped"],
        key=lambda view: view.private.view_id,
    )
    iid = sorted(
        [view for view in all_views if view.private.corruption_mechanism == "iid" and view.private.split != "test"],
        key=lambda view: (view.public.n_nodes, view.private.view_id),
    )
    views = (grouped[:16] + iid[:16])[:32]
    state_ids: list[str] = []
    canonical_iterations: list[int] = []
    error_by_depth: dict[int, list[float]] = {depth: [] for depth in DEPTHS}
    for view in views:
        values = np.asarray(view.public.observed_log_ratio, dtype=np.float64)
        for pattern_index, base in enumerate(checker.weight_patterns(values)):
            canonical = solve_huber_irls(
                view.public,
                values=values,
                base_weights=base,
                delta=float(executor["delta"]),
                damping=float(executor["damping"]),
                tolerance=float(executor["tolerance"]),
                max_iterations=int(executor["max_iterations"]),
                weight_floor=float(executor["weight_floor"]),
            )
            if not canonical.converged:
                raise RuntimeError("canonical executor did not converge")
            state_ids.append(f"{view.private.view_id}|{view.private.corruption_mechanism}|w{pattern_index}")
            canonical_iterations.append(int(canonical.iterations))
            for depth in DEPTHS:
                x_hat, _, _ = checker.numpy_fixed_weighted_irls(
                    view.public,
                    values,
                    base,
                    steps=depth,
                    delta=float(executor["delta"]),
                    damping=float(executor["damping"]),
                    weight_floor=float(executor["weight_floor"]),
                )
                error_by_depth[depth].append(
                    float(np.sqrt(np.mean((x_hat - canonical.x_hat) ** 2)))
                )
    threshold_p99 = float(recipe["canonical_p99_rmse"])
    threshold_max = float(recipe["canonical_max_rmse"])
    rows = []
    for depth in DEPTHS:
        values = np.asarray(error_by_depth[depth], dtype=np.float64)
        p99, maximum = float(np.quantile(values, 0.99)), float(np.max(values))
        rows.append(
            {
                "steps": depth,
                "p99_rmse": p99,
                "max_rmse": maximum,
                "value_threshold_pass": p99 <= threshold_p99 and maximum <= threshold_max,
            }
        )
    passing = [row["steps"] for row in rows if row["value_threshold_pass"]]
    summary = {
        "schema_version": "proportional-base-weighted-depth-scan-v1",
        "source_commit": "03144284fc1d9c9236641983016f9393a0fc1841",
        "states": len(state_ids),
        "graphs": len(views),
        "depths": DEPTHS,
        "thresholds": {"p99_rmse": threshold_p99, "max_rmse": threshold_max},
        "canonical_iterations": {
            "min": int(np.min(canonical_iterations)),
            "median": float(np.median(canonical_iterations)),
            "p99": float(np.quantile(canonical_iterations, 0.99)),
            "max": int(np.max(canonical_iterations)),
        },
        "rows": rows,
        "first_tested_passing_depth": min(passing) if passing else None,
        "proposed_confirmatory_depth": 192 if 192 in passing else None,
        "selection_scope": "numeric_value_calibration_only",
        "gradient_confirmation_required": True,
        "gpu_used_or_queried": False,
        "architecture_promoted": False,
        "scientific_decision": None,
        "decision_authority": "user",
    }
    raw = {
        "state_id": np.asarray(state_ids),
        "canonical_iterations": np.asarray(canonical_iterations, dtype=np.int64),
        **{
            f"rmse_k{depth}": np.asarray(error_by_depth[depth], dtype=np.float64)
            for depth in DEPTHS
        },
    }
    return summary, raw


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be empty")
    output = args.output_root.resolve()
    allowed = (ROOT / "data/geometria_proporcional").resolve()
    if allowed not in output.parents:
        raise ValueError(f"output must be below {allowed}")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    checker = load_checker()
    _, relational, _ = checker.config_triplet()
    runtimes = []
    for run_name in ("run_a", "run_b"):
        started = time.perf_counter()
        summary, raw = calculate(checker, relational)
        run = output / run_name
        run.mkdir()
        checker.write_json(run / "summary.json", summary)
        checker.write_deterministic_npz(run / "raw.npz", raw)
        runtimes.append(
            {"run": run_name, "wall_seconds": time.perf_counter() - started, "peak_rss_bytes": peak_rss_bytes()}
        )
    comparison = {
        "schema_version": "proportional-base-weighted-depth-scan-replay-v1",
        "summary_byte_exact": (output / "run_a/summary.json").read_bytes() == (output / "run_b/summary.json").read_bytes(),
        "raw_byte_exact": (output / "run_a/raw.npz").read_bytes() == (output / "run_b/raw.npz").read_bytes(),
        "gpu_used_or_queried": False,
    }
    comparison["status"] = "PASS" if comparison["summary_byte_exact"] and comparison["raw_byte_exact"] else "FAIL"
    checker.write_json(output / "replay_comparison.json", comparison)
    checker.write_json(output / "runtime_observation.json", {"runs": runtimes, "gpu_used_or_queried": False})
    manifest = {
        "schema_version": "proportional-base-weighted-depth-scan-manifest-v1",
        "files": [
            {"path": relative, "sha256": checker.sha256_file(output / relative), "bytes": (output / relative).stat().st_size}
            for relative in ["replay_comparison.json", "run_a/raw.npz", "run_a/summary.json", "run_b/raw.npz", "run_b/summary.json"]
        ],
        "runtime_observation": "runtime_observation.json",
        "gpu_used_or_queried": False,
    }
    checker.write_json(output / "manifest.json", manifest)
    print(json.dumps({"replay": comparison["status"], "summary": summary, "runtime": runtimes}, sort_keys=True))
    return 0 if comparison["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
