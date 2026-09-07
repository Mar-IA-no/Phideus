#!/usr/bin/env python3
"""Run the dual-native design preflight twice and compare scientific bytes."""

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


ROOT = Path(__file__).resolve().parents[2]
CHECKER_PATH = ROOT / "experiments/geometria_proporcional/check_proportional_dual_native_freeze.py"
DEFAULT_OUTPUT = ROOT / "data/geometria_proporcional/proportional_dual_native_freeze_v1"


def load_checker() -> Any:
    spec = importlib.util.spec_from_file_location("proportional_dual_native_checker", CHECKER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load independent checker")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def peak_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value * 1024 if sys.platform != "darwin" else value)


def compare_runs(checker: Any, left: Path, right: Path) -> dict[str, Any]:
    left_manifest = checker.load_json(left / "manifest.json")
    right_manifest = checker.load_json(right / "manifest.json")
    names = sorted({row["path"] for row in left_manifest["files"]})
    rows = []
    for name in names:
        left_path, right_path = left / name, right / name
        rows.append(
            {
                "path": name,
                "left_sha256": checker.sha256_file(left_path),
                "right_sha256": checker.sha256_file(right_path),
                "byte_exact": left_path.read_bytes() == right_path.read_bytes(),
            }
        )
    rows.append(
        {
            "path": "manifest.json",
            "left_sha256": checker.sha256_file(left / "manifest.json"),
            "right_sha256": checker.sha256_file(right / "manifest.json"),
            "byte_exact": (left / "manifest.json").read_bytes()
            == (right / "manifest.json").read_bytes(),
        }
    )
    return {
        "schema_version": "proportional-dual-native-replay-comparison-v1",
        "status": "PASS" if all(row["byte_exact"] for row in rows) else "FAIL",
        "scientific_files": rows,
        "runtime_observation_compared": False,
        "fixed_claims": checker.FIXED_CLAIMS,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be the empty string")
    output_root = args.output_root.resolve()
    allowed = (ROOT / "data/geometria_proporcional").resolve()
    if allowed not in output_root.parents:
        raise ValueError(f"output must be below {allowed}")
    if output_root.exists():
        reserved = [
            output_root / name
            for name in (
                "run_a",
                "run_b",
                "manifest.json",
                "replay_comparison.json",
                "runtime_observation.json",
            )
        ]
        if any(path.exists() for path in reserved):
            raise FileExistsError(f"refusing to overwrite final preflight in {output_root}")
    else:
        output_root.mkdir(parents=True)
    checker = load_checker()
    config_path = args.config or checker.DEFAULT_COORDINATOR
    coordinator, relational, set_valued = checker.config_triplet(config_path)
    runtime_rows = []
    for name in ("run_a", "run_b"):
        started = time.perf_counter()
        report, raw = checker.scientific_payload(coordinator, relational, set_valued)
        checker.write_artifact(output_root / name, report, raw)
        checked = checker.check_artifact(output_root / name)
        runtime_rows.append(
            {
                "run": name,
                "wall_seconds": time.perf_counter() - started,
                "peak_rss_bytes": peak_rss_bytes(),
                "artifact_check": checked["status"],
            }
        )
    comparison = compare_runs(checker, output_root / "run_a", output_root / "run_b")
    checker.write_json(output_root / "replay_comparison.json", comparison)
    runtime = {
        "schema_version": "proportional-dual-native-runtime-observation-v1",
        "runs": runtime_rows,
        "gpu_used_or_queried": False,
        "cuda_visible_devices": "",
    }
    checker.write_json(output_root / "runtime_observation.json", runtime)
    root_manifest = {
        "schema_version": "proportional-dual-native-root-manifest-v1",
        "scientific": [
            {
                "path": "replay_comparison.json",
                "sha256": checker.sha256_file(output_root / "replay_comparison.json"),
                "bytes": (output_root / "replay_comparison.json").stat().st_size,
            },
            {
                "path": "run_a/manifest.json",
                "sha256": checker.sha256_file(output_root / "run_a/manifest.json"),
                "bytes": (output_root / "run_a/manifest.json").stat().st_size,
            },
            {
                "path": "run_b/manifest.json",
                "sha256": checker.sha256_file(output_root / "run_b/manifest.json"),
                "bytes": (output_root / "run_b/manifest.json").stat().st_size,
            },
        ],
        "runtime_observation": "runtime_observation.json",
        "historical_exclusions": checker.KNOWN_HISTORICAL_ROOTS,
        "fixed_claims": checker.FIXED_CLAIMS,
    }
    checker.write_json(output_root / "manifest.json", root_manifest)
    root_check = checker.check_root_artifact(output_root)
    summary = {
        "replay": comparison["status"],
        "root_artifact_check": root_check["status"],
        "design_state": checker.load_json(output_root / "run_a/scientific_report.json")[
            "design_state"
        ],
        "wall_seconds": sum(row["wall_seconds"] for row in runtime_rows),
        "peak_rss_bytes": max(row["peak_rss_bytes"] for row in runtime_rows),
        "gpu_used_or_queried": False,
    }
    print(json.dumps(summary, sort_keys=True))
    return 0 if comparison["status"] == "PASS" and root_check["status"] == "PASS" and all(row["artifact_check"] == "PASS" for row in runtime_rows) else 1


if __name__ == "__main__":
    sys.exit(main())
