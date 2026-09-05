#!/usr/bin/env python3
"""Coordinate the physically separated Wave 59 analytical phases.

This runner consumes a previously prepared package.  It never generates a draw
or opens a sealed truth file outside the phase transition that authorizes it.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path
import pwd
import shutil
import subprocess
import sys
import tempfile
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from geometria_proporcional.wave49_schema import sha256_file  # noqa: E402
from geometria_proporcional.wave59_hgb_guard_bracket import (  # noqa: E402
    validate_pre_draw_config,
)


CONFIG_DEFAULT = REPO_ROOT / "experiments/geometria_proporcional/configs/wave59_fresh_hgb_guard_bracket.json"
WORKER_SOURCE = REPO_ROOT / "experiments/geometria_proporcional/_wave59_phase_worker.py"
RUNTIME_MODULES = (
    "__init__.py",
    "wave49_schema.py",
    "wave52_policy.py",
    "wave53_uncertainty.py",
    "wave55_policy_bridge.py",
    "wave56_contextual_gate.py",
    "wave58_open_diagnostic.py",
    "wave59_hgb_guard_bracket.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--policy-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=CONFIG_DEFAULT)
    parser.add_argument("--reference-dir", type=Path)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _copy(source: Path, destination: Path) -> None:
    source = source.resolve(strict=True)
    if source.is_symlink() or not source.is_file():
        raise RuntimeError(f"source is not a regular file: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def load_utilities(policy_manifest: Path) -> np.ndarray:
    payload = read_json(policy_manifest.resolve(strict=True))
    levels = np.asarray(payload["levels"], dtype=np.float64)
    permutations = np.asarray(payload["rank_permutations"], dtype=np.int64)
    if permutations.shape != (24, 4) or set(map(tuple, permutations)) != set(
        itertools.permutations(range(4))
    ):
        raise ValueError("policy manifest utility_matrix is invalid")
    utilities = levels[permutations]
    if utilities.shape != (24, 4) or not np.all(np.isfinite(utilities)):
        raise ValueError("policy manifest utilities are invalid")
    return utilities


def build_runtime(root: Path) -> Path:
    source = root / "source"
    package = source / "geometria_proporcional"
    package.mkdir(parents=True)
    for name in RUNTIME_MODULES:
        _copy(SRC_ROOT / "geometria_proporcional" / name, package / name)
    _copy(WORKER_SOURCE, source / WORKER_SOURCE.name)
    for path in source.rglob("*"):
        path.chmod(0o755 if path.is_dir() else 0o444)
    source.chmod(0o755)
    return source / WORKER_SOURCE.name


def _stage_request(stage: Path, phase: str) -> None:
    files = sorted(path.name for path in stage.iterdir() if path.is_file())
    hashes = {name: sha256_file(stage / name) for name in files}
    request = {
        "phase": phase,
        "allowed_files": sorted([*files, "phase_request.json"]),
        "sha256": hashes,
    }
    write_json(stage / "phase_request.json", request)
    for path in stage.iterdir():
        path.chmod(0o444)
    stage.chmod(0o555)


def run_worker(
    temporary: Path,
    stage: Path,
    phase: str,
    probes: list[Path],
) -> Path:
    worker = build_runtime(temporary)
    _stage_request(stage, phase)
    account = pwd.getpwnam("nobody")
    output = temporary / "worker-output"
    output.mkdir(mode=0o700)
    os.chown(output, account.pw_uid, account.pw_gid)
    command = [
        "setpriv",
        "--reuid",
        str(account.pw_uid),
        "--regid",
        str(account.pw_gid),
        "--clear-groups",
        "--no-new-privs",
        sys.executable,
        str(worker),
        "--stage",
        str(stage),
        "--output",
        str(output),
    ]
    for probe in probes:
        command.extend(("--forbidden-probe", str(probe)))
    env = dict(os.environ)
    env.update(
        {
            "PYTHONPATH": str(temporary / "source"),
            "WAVE59_STAGED_RUNTIME": "1",
            "CUDA_VISIBLE_DEVICES": "",
            "OMP_NUM_THREADS": "4",
            "OPENBLAS_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
            "NUMEXPR_NUM_THREADS": "4",
            "PYTHONHASHSEED": "0",
        }
    )
    completed = subprocess.run(
        command, cwd=stage, env=env, text=True, capture_output=True, check=False
    )
    if completed.returncode:
        raise RuntimeError(f"Wave 59 {phase} worker failed: {completed.stderr.strip()}")
    receipt = read_json(output / "access_receipt.json")
    if receipt["effective_uid"] != 65534 or receipt["effective_gid"] != 65534:
        raise RuntimeError("Wave 59 worker identity drifted")
    security = receipt["process_security"]
    if security != {
        "effective_capabilities_hex": "0000000000000000",
        "no_new_privileges": 1,
        "supplementary_groups": [],
    }:
        raise RuntimeError("Wave 59 worker privilege boundary failed")
    if not all(row["denied"] for row in receipt["forbidden_probes"]):
        raise RuntimeError("Wave 59 worker could access a forbidden truth probe")
    return output


def _publish(worker_output: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(destination)
    staging = destination.with_name(destination.name + ".pending")
    if staging.exists():
        raise FileExistsError(staging)
    shutil.copytree(worker_output, staging, symlinks=False)
    for path in staging.rglob("*"):
        if path.is_file():
            path.chmod(0o444)
        elif path.is_dir():
            path.chmod(0o555)
    staging.chmod(0o555)
    os.replace(staging, destination)


def _run_phase(
    run_dir: Path,
    phase: str,
    inputs: dict[str, Path],
    probes: list[Path],
) -> Path:
    with tempfile.TemporaryDirectory(prefix=f"wave59-{phase}-", dir="/tmp") as raw:
        temporary = Path(raw)
        temporary.chmod(0o711)
        stage = temporary / "stage"
        stage.mkdir()
        for name, source in inputs.items():
            _copy(source, stage / name)
        output = run_worker(temporary, stage, phase, probes)
        destination = run_dir / phase
        _publish(output, destination)
    return destination


def execute(
    prepared: Path,
    policy_manifest: Path,
    output: Path,
    config_path: Path,
) -> Path:
    config = read_json(config_path)
    validate_pre_draw_config(config)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    _copy(config_path, output / "config.snapshot.json")
    utilities = load_utilities(policy_manifest)
    utilities_path = output / ".utilities.npy"
    np.save(utilities_path, utilities)
    fit = _run_phase(
        output,
        "fit",
        {
            "config.json": config_path,
            "bundle.npz": prepared / "gate_fit_bundle.npz",
            "utilities.npy": utilities_path,
        },
        [prepared / "gate_select_truth_bundle.npz", prepared / "sealed_monitor_truth_bundle.npz"],
    )
    calibration = _run_phase(
        output,
        "calibrate_scores",
        {
            "config.json": config_path,
            "inference_bundle.npz": prepared / "gate_select_inference_bundle.npz",
            "model_states.json": fit / "model_states.json",
            "model_state_arrays.npz": fit / "model_state_arrays.npz",
            "fit_freeze.json": fit / "fit_freeze.json",
        },
        [prepared / "gate_select_truth_bundle.npz", prepared / "sealed_monitor_truth_bundle.npz"],
    )
    validation = _run_phase(
        output,
        "validate",
        {
            "config.json": config_path,
            "truth_bundle.npz": prepared / "gate_select_truth_bundle.npz",
            "validation_policy_arrays.npz": calibration / "validation_policy_arrays.npz",
            "calibration_freeze.json": calibration / "calibration_freeze.json",
            "utilities.npy": utilities_path,
        },
        [prepared / "sealed_monitor_truth_bundle.npz"],
    )
    monitor_apply = _run_phase(
        output,
        "monitor_apply",
        {
            "config.json": config_path,
            "inference_bundle.npz": prepared / "sealed_monitor_inference_bundle.npz",
            "model_states.json": fit / "model_states.json",
            "model_state_arrays.npz": fit / "model_state_arrays.npz",
            "fit_freeze.json": fit / "fit_freeze.json",
            "calibration_freeze.json": calibration / "calibration_freeze.json",
        },
        [prepared / "sealed_monitor_truth_bundle.npz"],
    )
    _run_phase(
        output,
        "monitor_evaluate",
        {
            "config.json": config_path,
            "truth_bundle.npz": prepared / "sealed_monitor_truth_bundle.npz",
            "monitor_policy_arrays.npz": monitor_apply / "monitor_policy_arrays.npz",
            "monitor_action_freeze.json": monitor_apply / "monitor_action_freeze.json",
            "utilities.npy": utilities_path,
        },
        [],
    )
    utilities_path.unlink()
    write_json(
        output / "runtime.json",
        {
            "status": "COMPLETE",
            "device": "cpu",
            "cuda_visible_devices": "",
            "phases": ["fit", "calibrate_scores", "validate", "monitor_apply", "monitor_evaluate"],
        },
    )
    return output


def main() -> None:
    args = parse_args()
    execute(
        args.prepared_dir.resolve(strict=True),
        args.policy_manifest.resolve(strict=True),
        args.output_dir.resolve(strict=False),
        args.config.resolve(strict=True),
    )


if __name__ == "__main__":
    main()
