#!/usr/bin/env python3
"""Exercise every promotion/journal crash point of the physical package."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "experiments/geometria_proporcional/run_proportional_set_valued_physical_preflight.py"
PHASES = (
    "posterior_fit", "policy_fit", "selection_propose", "selection_evaluate",
    "selection_freeze", "evaluation_apply", "evaluation_truth",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--input-package", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, default=ROOT / "data/geometria_proporcional/proportional_set_valued_physical_recovery_v1")
    parser.add_argument("--receipt", type=Path, required=True)
    return parser.parse_args()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def main() -> int:
    args = parse_args(); reference = args.reference.resolve(strict=True); input_package = args.input_package.resolve(strict=True)
    work = args.work_root.resolve()
    if work.exists(): shutil.rmtree(work)
    work.mkdir(parents=True)
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}
    rows = []; started = time.monotonic(); peak_temporary_bytes = 0
    for phase in PHASES:
        output = work / phase
        crash_command = [str(ROOT / "venv/bin/python"), str(RUNNER), "--input-package", str(input_package), "--output-dir", str(output), "--inject-crash-after-promotion", phase]
        crash = subprocess.run(crash_command, cwd=ROOT, env=environment, text=True, capture_output=True)
        if crash.returncode == 0 or not (output / phase).exists() or (output / "journals" / f"{phase}.json").exists():
            raise RuntimeError(f"crash point did not leave the required orphan: {phase}")
        resume_command = [str(ROOT / "venv/bin/python"), str(RUNNER), "--input-package", str(input_package), "--output-dir", str(output), "--resume", "--reference-dir", str(reference)]
        resumed = subprocess.run(resume_command, cwd=ROOT, env=environment, text=True, capture_output=True)
        if resumed.returncode:
            raise RuntimeError(f"recovery failed for {phase}: {resumed.stderr[-3000:]}")
        payload = json.loads(resumed.stdout.strip().splitlines()[-1])
        if payload["replay"] is not True or not (output / "recovery_origin.json").exists():
            raise RuntimeError(f"recovery comparison failed for {phase}")
        rows.append({"crash_point": phase, "crash_exit": crash.returncode, "orphan_phase_present": True, "journal_absent": True, "recovery_action": "ARCHIVE_OUTPUT_AND_RESTART_FROM_SAME_PACKAGE", "reference_manifest_sha256": sha(reference / "artifact_manifest.json"), "recovered_manifest_sha256": sha(output / "artifact_manifest.json"), "scientific_byte_exact": True})
        peak_temporary_bytes = max(peak_temporary_bytes, tree_bytes(work))
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    freeze = ROOT / "experiments/geometria_proporcional/configs/proportional_set_valued_physical_source_freeze_v1.json"
    payload = {"schema_version": "proportional-physical-recovery-receipt-v1", "status": "PASS", "argv": sys.argv, "source_freeze_sha256": sha(freeze), "inputs": {"reference_manifest_sha256": sha(reference / "artifact_manifest.json"), "input_preparation_sha256": sha(input_package / "preparation_freeze.json")}, "outputs": {"recovered_runs": len(rows)}, "versions": {"python": sys.version.split()[0], "numpy": np.__version__, "scipy": __import__("scipy").__version__, "sklearn": __import__("sklearn").__version__}, "exit": 0, "cases": rows, "passed": len(rows), "total": len(PHASES), "wall_seconds": time.monotonic() - started, "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024, "children_peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss) * 1024, "peak_temporary_bytes": peak_temporary_bytes, "preserved_bytes_before_receipt": tree_bytes(work), "gpu_used_or_queried": False}
    args.receipt.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "passed": len(rows), "total": len(PHASES)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
