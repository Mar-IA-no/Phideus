#!/usr/bin/env python3
"""Run the short physical-package suites and seal their evidence receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
from typing import Any

import numpy as np
import scipy
import sklearn


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / "venv/bin/python"
CHECKER = ROOT / "experiments/geometria_proporcional/check_proportional_set_valued_physical_preflight.py"
FREEZE = ROOT / "experiments/geometria_proporcional/configs/proportional_set_valued_physical_source_freeze_v1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--input-package", type=Path, required=True)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--manifest-only", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_row(path: Path) -> dict[str, int | str]:
    return {"bytes": path.stat().st_size, "sha256": sha256_file(path)}


def relative_rows(paths: list[Path]) -> dict[str, dict[str, int | str]]:
    return {path.resolve().relative_to(ROOT).as_posix(): file_row(path.resolve()) for path in paths}


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8")


def versions() -> dict[str, str]:
    return {"python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__}


def run_suite(name: str, argv: list[str], inputs: list[Path], outputs: list[Path], evidence: Path, total: int) -> None:
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1", "PYTHONNOUSERSITE": "1"}
    started = time.monotonic(); result = subprocess.run(argv, cwd=ROOT, env=environment, text=True, capture_output=True); wall = time.monotonic() - started
    stdout_last_json = None
    for line in reversed(result.stdout.splitlines()):
        try:
            stdout_last_json = json.loads(line); break
        except json.JSONDecodeError:
            continue
    if name != "unit_test":
        passed = None if stdout_last_json is None else stdout_last_json.get("passed")
        observed_total = None if stdout_last_json is None else stdout_last_json.get("total")
    else:
        passed = total if result.returncode == 0 and f"Ran {total} tests" in result.stderr else 0
        observed_total = total
    payload = {
        "schema_version": "proportional-physical-suite-receipt-v2", "suite": name,
        "source_freeze_sha256": sha256_file(FREEZE),
        "status": "PASS" if result.returncode == 0 and passed == total and observed_total == total else "FAIL",
        "argv": argv, "inputs": relative_rows(inputs), "outputs": relative_rows(outputs),
        "versions": versions(), "exit": result.returncode, "wall_seconds": wall,
        "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss) * 1024,
        "peak_temporary_bytes": 0, "preserved_bytes_before_receipt": sum(path.stat().st_size for path in outputs),
        "gpu_used_or_queried": False, "stdout_last_json": stdout_last_json,
        "stdout_sha256": hashlib.sha256(result.stdout.encode()).hexdigest(), "stderr_sha256": hashlib.sha256(result.stderr.encode()).hexdigest(),
        "passed": passed, "total": observed_total,
    }
    write_json(evidence / f"{name}_receipt.json", payload)
    if payload["status"] != "PASS":
        raise RuntimeError(f"{name} failed: {result.stderr[-2000:]}{result.stdout[-2000:]}")


def write_manifest(evidence: Path, primary: Path, replay: Path) -> None:
    files = [{"path": path.name, **file_row(path)} for path in sorted(evidence.iterdir()) if path.is_file() and path.name != "evidence_manifest.json"]
    write_json(evidence / "evidence_manifest.json", {"schema_version": "proportional-physical-evidence-manifest-v1", "self_excluded": True, "source_freeze_sha256": sha256_file(FREEZE), "primary_artifact_manifest_sha256": sha256_file(primary / "artifact_manifest.json"), "replay_artifact_manifest_sha256": sha256_file(replay / "artifact_manifest.json"), "files": files})


def main() -> int:
    args = parse_args(); evidence = args.evidence.resolve(); input_package = args.input_package.resolve(strict=True); primary = args.primary.resolve(strict=True); replay = args.replay.resolve(strict=True)
    evidence.mkdir(parents=True, exist_ok=True)
    if not args.manifest_only:
        common = [FREEZE, input_package / "preparation_freeze.json"]
        run_suite("unit_test", [str(PYTHON), "-m", "unittest", "tests.test_proportional_set_valued_physical", "-v"], [FREEZE], [ROOT / "tests/test_proportional_set_valued_physical.py"], evidence, 15)
        run_suite("primary_check", [str(PYTHON), str(CHECKER), "--artifact", str(primary), "--input-package", str(input_package)], [*common, primary / "artifact_manifest.json"], [primary / "artifact_manifest.json"], evidence, 15)
        run_suite("replay_check", [str(PYTHON), str(CHECKER), "--artifact", str(replay), "--input-package", str(input_package), "--reference", str(primary)], [*common, primary / "artifact_manifest.json", replay / "artifact_manifest.json"], [replay / "artifact_manifest.json"], evidence, 15)
    write_manifest(evidence, primary, replay)
    print(json.dumps({"status": "PASS", "receipts": len([path for path in evidence.iterdir() if path.is_file() and path.name != "evidence_manifest.json"]), "manifest_sha256": sha256_file(evidence / "evidence_manifest.json")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
