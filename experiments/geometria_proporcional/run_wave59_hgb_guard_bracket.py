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
import time
from typing import Any
from datetime import UTC, datetime
import hashlib
import signal

import joblib
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from geometria_proporcional.wave49_schema import sha256_file  # noqa: E402
from geometria_proporcional.wave49_attestation import (  # noqa: E402
    sign_attestation,
    verify_attestation,
)
from geometria_proporcional.wave59_hgb_guard_bracket import (  # noqa: E402
    HARM_CONTROL_SEEDS,
    INCOMPATIBILITY_CONTROL_SEEDS,
    CONFIG_SOURCE_SUFFIX,
    FROZEN_STATUS,
    config_self_binding_sha256,
    inference_safe_view,
    model_id,
    validate_pre_draw_config,
)


CONFIG_DEFAULT = REPO_ROOT / "experiments/geometria_proporcional/configs/wave59_fresh_hgb_guard_bracket.json"
DEFAULT_PRIVATE_KEY = Path.home() / ".config/phideus/wave49_attestation_private.pem"
TRUSTED_PUBLIC_KEY = REPO_ROOT / "experiments/geometria_proporcional/keys/wave49_attestation_public.pem"
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
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument(
        "--attestation-private-key", type=Path, default=DEFAULT_PRIVATE_KEY
    )
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_json(path: Path, payload: Any, *, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
        + "\n"
    ).encode()
    descriptor, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(raw)
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def write_npz(path: Path, arrays: dict[str, np.ndarray], *, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npz", dir=path.parent)
    os.close(descriptor)
    temporary = Path(raw)
    try:
        np.savez(temporary, **{key: np.asarray(value) for key, value in sorted(arrays.items())})
        temporary.chmod(mode)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def materialize_prepared_bundles(
    run_dir: Path,
    config: dict[str, Any],
    *,
    policy_manifest: Path,
    wave54_selection_freeze: Path,
) -> dict[str, str]:
    """Create Wave 59 safe/truth bundles during PREPARE under root control.

    The shared preparer has already sealed the benchmark and completed blind
    inference.  This root-only step derives the frozen ``primary`` indicator
    and all result-affecting arrays once, publishes only the allowlisted safe
    projections for calibration/monitor application, and keeps full truth
    bundles mode 0600 for phase-scoped coordinator access.
    """
    if os.geteuid() != 0 or os.getegid() != 0:
        raise PermissionError("Wave 59 bundle preparation requires root")
    if run_dir.is_symlink():
        raise RuntimeError("Wave 59 preparation root cannot be a symlink")

    # These imports are deliberately local: the analytical workers never stage
    # either the oracle materializer or the legacy bundle constructor.
    from geometria_proporcional.wave49_oracle import compute_oracle_splits
    from geometria_proporcional.wave49_schema import ProtocolConfig
    import _wave56_phase_worker as wave56_worker
    import run_wave56_retrospective as retrospective

    benchmark = run_dir / "benchmark"
    protocol_path = benchmark / "protocol_config.json"
    protocol = ProtocolConfig.from_dict(read_json(protocol_path))
    utilities, _ = retrospective.load_utilities(policy_manifest.resolve(strict=True))
    selection = read_json(wave54_selection_freeze.resolve(strict=True))
    theta = np.asarray(
        selection["selected_models"]["joint_full"]["theta"], dtype=np.float64
    )
    seeds = [int(seed) for seed in config["seeds"]]
    prepared = run_dir / "prepared"
    if prepared.exists():
        raise FileExistsError(prepared)
    prepared.mkdir(mode=0o700)

    specs = {
        "train": ("gate_fit_bundle.npz", None),
        "val": ("gate_select_truth_bundle.npz", "gate_select_inference_bundle.npz"),
        "lockbox": (
            "sealed_monitor_truth_bundle.npz",
            "sealed_monitor_inference_bundle.npz",
        ),
    }
    hashes: dict[str, str] = {}
    try:
        for split, (truth_name, safe_name) in specs.items():
            with tempfile.TemporaryDirectory(prefix=f"wave59-prepare-{split}-", dir="/tmp") as raw:
                stage = Path(raw) / "stage"
                (stage / "visible").mkdir(parents=True)
                (stage / "labels").mkdir()
                (stage / "inference/logits").mkdir(parents=True)
                shutil.copyfile(benchmark / "visible" / f"{split}.jsonl", stage / "visible" / f"{split}.jsonl")
                shutil.copyfile(protocol_path, stage / "protocol_config.json")
                compute_oracle_splits(benchmark, protocol, (split,), stage / "labels")
                for seed in seeds:
                    source = run_dir / "inference/logits" / f"seed{seed}__{split}.npz"
                    shutil.copyfile(source, stage / "inference/logits" / source.name)

                role = config["physical_splits"][split]
                bundle = wave56_worker.build_bundle(stage, split, role, seeds)
                dataset_config = {
                    "hard_set_tau": float(config["hard_set_tau"]),
                    "incompatible_regret_penalty": float(config["penalty"]),
                    "primary_population": {
                        "design_stratum": "NEAR_RIVAL",
                        "minimum_true_cardinality": 2,
                    },
                }
                data = retrospective.make_dataset(bundle, theta, utilities, dataset_config)
                arrays = {
                    key: np.asarray(value)
                    for key, value in data.items()
                    if isinstance(value, np.ndarray)
                }
                truth_path = prepared / truth_name
                write_npz(truth_path, arrays, mode=0o600)
                hashes[f"prepared/{truth_name}"] = sha256_file(truth_path)
                if safe_name is not None:
                    safe_path = prepared / safe_name
                    write_npz(safe_path, inference_safe_view(arrays), mode=0o644)
                    hashes[f"prepared/{safe_name}"] = sha256_file(safe_path)
    except BaseException:
        shutil.rmtree(prepared, ignore_errors=True)
        raise
    _fsync_directory(prepared)
    return hashes


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


def validate_execution_bindings(config: dict[str, Any], config_path: Path) -> None:
    """Authenticate the frozen worktree and dependency contract before each phase."""
    if config.get("status") != FROZEN_STATUS:
        return
    validate_pre_draw_config(config)
    expected = config["source_sha256"]
    relative_config = str(config_path.resolve(strict=True).relative_to(REPO_ROOT))
    if not relative_config.endswith(CONFIG_SOURCE_SUFFIX):
        raise RuntimeError("Wave 59 execution config path drifted")
    for relative in config["required_execution_sources"]:
        path = (REPO_ROOT / relative).resolve(strict=True)
        if path.relative_to(REPO_ROOT) != Path(relative):
            raise RuntimeError(f"Wave 59 non-canonical execution source: {relative}")
        observed = (
            config_self_binding_sha256(config, relative)
            if relative == relative_config
            else sha256_file(path)
        )
        if observed != expected[relative]:
            raise RuntimeError(f"Wave 59 execution source drifted: {relative}")
        require_clean_head_source(REPO_ROOT, relative, path)
    commit = config["implementation_binding"]["commit"]
    subprocess.run(
        ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    )
    audit_path = REPO_ROOT / config["implementation_binding"]["audit_path"]
    if sha256_file(audit_path) != config["implementation_binding"]["audit_sha256"]:
        raise RuntimeError("Wave 59 implementation audit hash drifted")
    audit_text = audit_path.read_text(encoding="utf-8")
    if commit not in audit_text or "## Dictamen: PASS" not in audit_text:
        raise RuntimeError("Wave 59 implementation audit does not accept the bound commit")


def require_clean_head_source(repo_root: Path, relative: str, path: Path) -> None:
    changed = subprocess.check_output(
        ["git", "status", "--porcelain", "--", relative],
        cwd=repo_root,
        text=True,
    ).strip()
    if changed:
        raise RuntimeError(f"Wave 59 execution source is dirty: {relative}")
    head_bytes = subprocess.check_output(
        ["git", "show", f"HEAD:{relative}"], cwd=repo_root
    )
    if hashlib.sha256(head_bytes).hexdigest() != sha256_file(path):
        raise RuntimeError(f"Wave 59 execution source is not the HEAD blob: {relative}")


def validate_freeze_bindings(
    freeze_path: Path,
    *,
    phase: str,
    bindings: dict[str, Path],
    require_all: bool = False,
) -> dict[str, Any]:
    freeze = read_json(freeze_path.resolve(strict=True))
    if freeze.get("schema_version") != "wave59-phase-freeze-v1" or freeze.get(
        "phase"
    ) != phase:
        raise RuntimeError(f"Wave 59 {freeze_path.name} identity drifted")
    files = freeze.get("files")
    if not isinstance(files, dict) or not files:
        raise RuntimeError(f"Wave 59 {freeze_path.name} has no file bindings")
    if require_all and set(files) != set(bindings):
        raise RuntimeError(f"Wave 59 {freeze_path.name} coverage drifted")
    if not set(bindings).issubset(files):
        raise RuntimeError(f"Wave 59 {freeze_path.name} lacks required files")
    for name, path in sorted(bindings.items()):
        if files[name] != sha256_file(path.resolve(strict=True)):
            raise RuntimeError(f"Wave 59 {freeze_path.name} hash mismatch: {name}")
    return freeze


def validate_freeze_provenance(
    freeze: dict[str, Any], *, config_path: Path, source_bindings: Path, preparation: Path
) -> None:
    expected = {
        "config_sha256": sha256_file(config_path.resolve(strict=True)),
        "source_bindings_sha256": sha256_file(source_bindings.resolve(strict=True)),
        "preparation_freeze_sha256": sha256_file(preparation.resolve(strict=True)),
    }
    for field, digest in expected.items():
        if freeze.get(field) != digest:
            raise RuntimeError(f"Wave 59 freeze provenance mismatch: {field}")


def build_runtime(root: Path) -> Path:
    source = root / "source"
    package = source / "geometria_proporcional"
    package.mkdir(parents=True)
    for name in RUNTIME_MODULES:
        origin = SRC_ROOT / "geometria_proporcional" / name
        destination = package / name
        _copy(origin, destination)
        if sha256_file(destination) != sha256_file(origin):
            raise RuntimeError(f"Wave 59 staged runtime copy drifted: {name}")
    _copy(WORKER_SOURCE, source / WORKER_SOURCE.name)
    if sha256_file(source / WORKER_SOURCE.name) != sha256_file(WORKER_SOURCE):
        raise RuntimeError("Wave 59 staged worker copy drifted")
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


def _process_rss_bytes(pid: int) -> int:
    try:
        for line in Path(f"/proc/{pid}/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (FileNotFoundError, ProcessLookupError):
        return 0
    return 0


def _terminate_process_group(process: subprocess.Popen[str]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=5)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def run_worker(
    temporary: Path,
    stage: Path,
    phase: str,
    probes: list[Path],
    *,
    deadline: float | None = None,
) -> tuple[Path, dict[str, Any], float, int]:
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
    env = {
        "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PYTHONPATH": str(temporary / "source"),
        "WAVE59_STAGED_RUNTIME": "1",
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "4",
        "OPENBLAS_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4",
        "NUMEXPR_NUM_THREADS": "4",
        "PYTHONHASHSEED": "0",
    }
    budget = read_json(stage / "config.json")["runtime_budget"]
    started = time.monotonic()
    effective_deadline = min(
        deadline if deadline is not None else float("inf"),
        started + float(budget["max_seconds_per_run"]),
    )
    max_rss_allowed = int(budget["max_rss_bytes"])
    process = subprocess.Popen(
        command,
        cwd=stage,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    peak_rss = 0
    stdout = ""
    stderr = ""
    while True:
        peak_rss = max(peak_rss, _process_rss_bytes(process.pid))
        if peak_rss > max_rss_allowed:
            _terminate_process_group(process)
            raise RuntimeError(
                f"Wave 59 {phase} worker exceeded RSS budget: {peak_rss}>{max_rss_allowed}"
            )
        remaining = effective_deadline - time.monotonic()
        if remaining <= 0:
            _terminate_process_group(process)
            raise RuntimeError(f"Wave 59 {phase} worker exceeded wall-time budget")
        try:
            stdout, stderr = process.communicate(timeout=min(0.2, remaining))
            break
        except subprocess.TimeoutExpired:
            continue
    duration = time.monotonic() - started
    peak_rss = max(peak_rss, _process_rss_bytes(process.pid))
    if process.returncode:
        raise RuntimeError(f"Wave 59 {phase} worker failed: {stderr.strip()}")
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
    threadpools = receipt.get("threadpools", [])
    if any(int(row.get("num_threads", 0)) > 4 for row in threadpools):
        raise RuntimeError("Wave 59 worker exceeded the four-thread contract")
    return output, receipt, duration, peak_rss


def _publish(worker_output: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(destination)
    staging = destination.with_name(destination.name + ".pending")
    if staging.exists():
        raise FileExistsError(staging)
    shutil.copytree(
        worker_output,
        staging,
        symlinks=False,
        ignore=shutil.ignore_patterns("access_receipt.json"),
    )
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
    *,
    destination_name: str | None = None,
    deadline: float | None = None,
) -> tuple[Path, dict[str, Any]]:
    with tempfile.TemporaryDirectory(prefix=f"wave59-{phase}-", dir="/tmp") as raw:
        temporary = Path(raw)
        temporary.chmod(0o711)
        stage = temporary / "stage"
        stage.mkdir()
        for name, source in inputs.items():
            _copy(source, stage / name)
        output, receipt, duration, max_rss_bytes = run_worker(
            temporary, stage, phase, probes, deadline=deadline
        )
        destination = run_dir / (destination_name or phase)
        _publish(output, destination)
    journal = {
        "schema_version": "wave59-phase-journal-v1",
        "phase": phase,
        "status": receipt["status"],
        "input_sha256": {
            name: sha256_file(path.resolve(strict=True)) for name, path in sorted(inputs.items())
        },
        "output_sha256": {
            str(path.relative_to(destination)): sha256_file(path)
            for path in sorted(destination.rglob("*"))
            if path.is_file()
        },
        "access_receipt": receipt,
        "duration_seconds": duration,
        "max_rss_bytes": max_rss_bytes,
        "maximum_truth_materialized": {
            "fit": "train",
            "calibrate_scores": "train",
            "validate": "validation",
            "monitor_apply": "validation",
            "monitor_evaluate": "monitor",
        }[phase],
    }
    write_json(run_dir / "journals" / f"{phase}.json", journal)
    return destination, journal


def _reuse_or_run_phase(
    run_dir: Path,
    phase: str,
    inputs: dict[str, Path],
    probes: list[Path],
    *,
    destination_name: str,
    deadline: float | None = None,
) -> tuple[Path, dict[str, Any]]:
    destination = run_dir / destination_name
    journal_path = run_dir / "journals" / f"{phase}.json"
    if journal_path.is_file():
        journal = read_json(journal_path)
        if journal.get("phase") != phase:
            raise RuntimeError(f"Wave 59 {phase} resume journal drifted")
        expected_inputs = {
            name: sha256_file(path.resolve(strict=True)) for name, path in sorted(inputs.items())
        }
        if journal.get("input_sha256") != expected_inputs:
            raise RuntimeError(f"Wave 59 {phase} resume inputs differ")
        for relative, expected in journal.get("output_sha256", {}).items():
            path = destination / relative
            if not path.is_file() or sha256_file(path) != expected:
                raise RuntimeError(f"Wave 59 {phase} resumed output differs: {relative}")
        return destination, journal
    if destination.exists():
        raise RuntimeError(f"Wave 59 {phase} output exists without a durable journal")
    return _run_phase(
        run_dir,
        phase,
        inputs,
        probes,
        destination_name=destination_name,
        deadline=deadline,
    )


def restore_identical_hash_attempt(
    archived: Path, output: Path, config_path: Path
) -> Path:
    archived = archived.resolve(strict=True)
    output = output.resolve(strict=False)
    if output.exists():
        raise FileExistsError(output)
    if not (archived / "FAILURE.json").is_file():
        raise RuntimeError("resume source is not a Wave 59 failed attempt")
    failure = read_json(archived / "FAILURE.json")
    if failure.get("schema_version") != "wave59-failed-attempt-v1":
        raise RuntimeError("resume source failure schema drifted")
    if Path(str(failure.get("original_path", ""))).resolve() != output:
        raise RuntimeError("identical-hash resume must restore the original canonical path")
    _validate_failure_inventory(archived)
    _validate_resumed_journals(archived)
    snapshot = archived / "config.snapshot.json"
    if sha256_file(snapshot) != sha256_file(config_path.resolve(strict=True)):
        raise RuntimeError("identical-hash resume config differs")
    shutil.copytree(
        archived,
        output,
        symlinks=False,
        ignore=shutil.ignore_patterns(
            "FAILURE.json",
            "failure_inventory.json",
            "artifact_manifest.json",
        ),
    )
    _fsync_directory(output.parent)
    return output


def _validate_failure_inventory(archived: Path) -> dict[str, Any]:
    inventory_path = archived / "failure_inventory.json"
    attestation_path = archived / "failure_attestation.json"
    if not inventory_path.is_file() or inventory_path.is_symlink():
        raise RuntimeError("Wave 59 failed attempt lacks a physical failure inventory")
    if not attestation_path.is_file() or attestation_path.is_symlink():
        raise RuntimeError("Wave 59 failed attempt lacks its external-trust attestation")
    attestation = read_json(attestation_path)
    verify_attestation(attestation, TRUSTED_PUBLIC_KEY)
    failure_path = archived / "FAILURE.json"
    expected_anchor = {
        "schema_version": "wave59-failure-anchor-v1",
        "failure_inventory_sha256": sha256_file(inventory_path),
        "failure_sha256": sha256_file(failure_path),
        "archived_path": str(archived),
    }
    if attestation.get("payload") != expected_anchor:
        raise RuntimeError("Wave 59 failure attestation payload drifted")
    inventory = read_json(inventory_path)
    if inventory.get("schema_version") != "wave59-failure-inventory-v1":
        raise RuntimeError("Wave 59 failure inventory schema drifted")
    records = inventory.get("records")
    if not isinstance(records, list):
        raise RuntimeError("Wave 59 failure inventory records are absent")
    by_path = {row.get("path"): row for row in records if isinstance(row, dict)}
    if len(by_path) != len(records) or None in by_path:
        raise RuntimeError("Wave 59 failure inventory paths are not unique")
    actual = {
        str(path.relative_to(archived))
        for path in archived.rglob("*")
        if path.is_file()
        and path.name not in {"failure_inventory.json", "failure_attestation.json"}
    }
    if set(by_path) != actual:
        raise RuntimeError("Wave 59 failed-attempt inventory coverage drifted")
    for relative, row in sorted(by_path.items()):
        path = archived / relative
        if row.get("bytes") != path.stat().st_size or row.get("sha256") != sha256_file(path):
            raise RuntimeError(f"Wave 59 failed-attempt hash drifted: {relative}")
    for field in ("missing_required_through_last_journal", "extra", "overlap", "unclassified"):
        if inventory.get(field) != []:
            raise RuntimeError(f"Wave 59 failed-attempt {field} is not empty")
    return inventory


def _validate_resumed_journals(run_dir: Path) -> None:
    destinations = {
        "fit": run_dir / "fit",
        "calibrate_scores": run_dir / "calibration",
        "validate": run_dir / "validation",
        "monitor_apply": run_dir / "adjudication",
    }
    for phase, destination in destinations.items():
        journal_path = run_dir / "journals" / f"{phase}.json"
        if not journal_path.is_file():
            continue
        journal = read_json(journal_path)
        for relative, expected in journal.get("output_sha256", {}).items():
            path = destination / relative
            if not path.is_file() or sha256_file(path) != expected:
                raise RuntimeError(f"Wave 59 resumed {phase} output drifted: {relative}")
    monitor_journal = run_dir / "journals/monitor_evaluate.json"
    if monitor_journal.is_file():
        _validate_promoted_evaluation_outputs(run_dir, read_json(monitor_journal))


def _merge_evaluation(adjudication: Path, evaluation: Path, run_dir: Path) -> None:
    mapping = {
        evaluation / "bootstrap_indices.npz": adjudication / "bootstrap_indices.npz",
        evaluation / "analysis_arrays.npz": adjudication / "analysis_arrays.npz",
        evaluation / "analysis.json": run_dir / "analysis.json",
    }
    for source, destination in mapping.items():
        if not source.is_file() or source.is_symlink() or destination.exists():
            raise RuntimeError(f"invalid monitor-evaluate promotion: {source}")
        temporary = destination.with_name(f".{destination.name}.pending")
        shutil.copyfile(source, temporary)
        temporary.chmod(0o444)
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    shutil.rmtree(evaluation)


def _validate_promoted_evaluation_outputs(
    run_dir: Path, journal: dict[str, Any]
) -> None:
    promoted = {
        "bootstrap_indices.npz": run_dir / "adjudication/bootstrap_indices.npz",
        "analysis_arrays.npz": run_dir / "adjudication/analysis_arrays.npz",
        "analysis.json": run_dir / "analysis.json",
    }
    frozen = journal.get("output_sha256", {})
    if set(frozen) != set(promoted):
        raise RuntimeError("Wave 59 monitor-evaluate journal output coverage drifted")
    for relative, path in promoted.items():
        if not path.is_file() or sha256_file(path) != frozen[relative]:
            raise RuntimeError(f"Wave 59 promoted monitor output drifted: {relative}")


def _write_report(run_dir: Path) -> None:
    analysis = read_json(run_dir / "analysis.json")
    patterns = analysis["prospective_patterns"]
    lines = [
        "# Wave 59 — fresh HGB guard bracket",
        "",
        "Estado operativo: `COMPLETE`. Decisión científica: reservada al usuario.",
        "",
        "Los dos patrones se informan por separado y no seleccionan una arquitectura:",
        "",
        f"- incompatibility: `{patterns['incompatibility']['aggregate_with_replay']}`",
        f"- harm: `{patterns['harm']['aggregate_with_replay']}`",
        "",
    ]
    path = run_dir / "REPORT.md"
    encoded = "\n".join(lines).encode("utf-8")
    descriptor, raw = tempfile.mkstemp(prefix=".REPORT.md.", dir=run_dir)
    temporary = Path(raw)
    try:
        os.fchmod(descriptor, 0o444)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(run_dir)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _array_exact(left: Path, right: Path) -> bool:
    with np.load(left, allow_pickle=False) as lhs, np.load(right, allow_pickle=False) as rhs:
        if set(lhs.files) != set(rhs.files):
            return False
        for key in lhs.files:
            a = lhs[key]
            b = rhs[key]
            if a.dtype != b.dtype or a.shape != b.shape:
                return False
            if a.dtype.kind in "fc":
                if not np.array_equal(a, b, equal_nan=True):
                    return False
            elif not np.array_equal(a, b):
                return False
    return True


def _scientific_paths(canonical: bool) -> tuple[list[str], list[str]]:
    exact = [
        "config.snapshot.json",
        "source_bindings.json",
        "preparation_freeze.json",
        "fit/feature_schema.json",
        "fit/fit_freeze.json",
        "fit/max_displacement_diagnostics.json",
        "calibration/calibration_freeze.json",
        "validation/validation_freeze.json",
        "validation/validation_summary.json",
        "adjudication/monitor_action_freeze.json",
        "analysis.json",
        "REPORT.md",
    ]
    arrays = [
        "fit/model_state_arrays.npz",
        "fit/train_scores.npz",
        "fit/max_displacement_mappings.npz",
        "calibration/validation_scores.npz",
        "calibration/validation_policy_arrays.npz",
        "validation/validation_metrics.npz",
        "adjudication/monitor_scores.npz",
        "adjudication/monitor_policy_arrays.npz",
        "adjudication/bootstrap_indices.npz",
        "adjudication/analysis_arrays.npz",
    ]
    if canonical:
        exact.extend(
            [
                "pre_generation_freeze.json",
                "benchmark/manifest.json",
                "benchmark/protocol_config.json",
                "benchmark/attestations/semantic_root.json",
                "benchmark/commitments/semantic.jsonl",
                *(f"benchmark/visible/{split}.jsonl" for split in ("calibration_null", "train", "val", "lockbox")),
            ]
        )
        arrays.extend(
            [
                *(f"inference/logits/seed{seed}__{split}.npz" for seed in (17, 29, 43) for split in ("train", "val", "lockbox")),
                "prepared/gate_select_inference_bundle.npz",
                "prepared/sealed_monitor_inference_bundle.npz",
            ]
        )
    return sorted(exact), sorted(arrays)


def _authenticated_recovery_context(run_dir: Path) -> bool:
    preparation = read_json((run_dir / "preparation_freeze.json").resolve(strict=True))
    recovery = isinstance(preparation.get("recovery_provenance"), dict)
    amendment_present = (run_dir / "recovery_amendment.json").is_file()
    if recovery != amendment_present:
        raise RuntimeError("Wave 59 recovery metadata and amendment presence differ")
    if recovery:
        expected = preparation["recovery_provenance"].get("amendment_sha256")
        if expected != sha256_file(run_dir / "recovery_amendment.json"):
            raise RuntimeError("Wave 59 recovery amendment differs from authenticated metadata")
    return recovery


def _portable_joblib_check(run_dir: Path) -> dict[str, bool]:
    manifest = read_json(run_dir / "fit/model_states/manifest.json")
    states = manifest["portable_states"]
    bundle_root = run_dir / "prepared" if (run_dir / "prepared").is_dir() else None
    if bundle_root is None:
        return {identifier: True for identifier in states}
    split_specs = (
        (bundle_root / "gate_fit_bundle.npz", run_dir / "fit/train_scores.npz"),
        (bundle_root / "gate_select_inference_bundle.npz", run_dir / "calibration/validation_scores.npz"),
        (bundle_root / "sealed_monitor_inference_bundle.npz", run_dir / "adjudication/monitor_scores.npz"),
    )
    checks: dict[str, bool] = {}
    for identifier, state in states.items():
        model_path = run_dir / "fit" / manifest["models"][identifier]["path"]
        if sha256_file(model_path) != manifest["models"][identifier]["sha256"]:
            checks[identifier] = False
            continue
        model = joblib.load(model_path)
        exact = True
        for bundle_path, scores_path in split_specs:
            with np.load(bundle_path, allow_pickle=False) as bundle, np.load(
                scores_path, allow_pickle=False
            ) as scores:
                active = np.asarray(bundle["disagreement"], dtype=bool)
                design = np.asarray(bundle["design"], dtype=np.float64)[active]
                if state["kind"] in {"ridge", "logistic"} and not getattr(
                    model, "_wave59_portable", False
                ):
                    design = (design - np.asarray(state["mean"])) / np.asarray(state["scale"])
                direct = (
                    model.predict_proba(design)[:, 1]
                    if state["kind"] in {"logistic", "hgb_classifier"}
                    else model.predict(design)
                )
                expected = np.asarray(scores[identifier], dtype=np.float64)[active]
                if not np.array_equal(np.asarray(direct, dtype=np.float64), expected):
                    exact = False
                    break
        checks[identifier] = exact
    return checks


def _normalize_operational(value: Any) -> Any:
    omitted = {
        "timestamp_utc",
        "execution_mode",
        "replay_exact",
        "superseded_output",
        "duration_seconds",
        "max_rss_bytes",
        "output_sha256",
        "output_inventory_before_receipt",
        "path_sha256",
        "primary_plus_replay_seconds",
        "combined_budget_seconds",
        "preparation_duration_seconds",
        "total_run_seconds",
    }
    if isinstance(value, dict):
        return {
            key: _normalize_operational(item)
            for key, item in value.items()
            if key not in omitted
        }
    if isinstance(value, list):
        return [_normalize_operational(item) for item in value]
    return value


def compare_runs(replay: Path, primary: Path) -> dict[str, Any]:
    replay = replay.resolve(strict=True)
    primary = primary.resolve(strict=True)
    if replay == primary:
        raise ValueError("Wave 59 replay cannot compare itself")
    canonical = (replay / "benchmark").is_dir() or (primary / "benchmark").is_dir()
    if canonical and not ((replay / "benchmark").is_dir() and (primary / "benchmark").is_dir()):
        raise RuntimeError("Wave 59 primary/replay canonical scope differs")
    replay_recovery = _authenticated_recovery_context(replay)
    primary_recovery = _authenticated_recovery_context(primary)
    if replay_recovery != primary_recovery:
        raise RuntimeError("Wave 59 primary/replay recovery context differs")
    exact_paths, array_paths = _scientific_paths(canonical)
    if replay_recovery:
        exact_paths.append("recovery_amendment.json")
    exact = {
        path: (replay / path).is_file()
        and (primary / path).is_file()
        and sha256_file(replay / path) == sha256_file(primary / path)
        for path in exact_paths
    }
    arrays = {
        path: (replay / path).is_file()
        and (primary / path).is_file()
        and _array_exact(replay / path, primary / path)
        for path in array_paths
    }
    secrets: dict[str, bool] = {}
    if canonical:
        for path in [
            "generation_escrow.json",
            *(f"benchmark/sealed/{name}" for name in (
                "calibration_null.jsonl",
                "train.jsonl",
                "val.jsonl",
                "lockbox.jsonl",
                "generation_secret.json",
                "identity_secret.json",
                "semantic_commitment_secret.json",
            )),
            "prepared/gate_fit_bundle.npz",
            "prepared/gate_select_truth_bundle.npz",
            "prepared/sealed_monitor_truth_bundle.npz",
        ]:
            secrets[path] = (replay / path).is_file() and (primary / path).is_file() and sha256_file(replay / path) == sha256_file(primary / path)
    functional = {
        "primary": _portable_joblib_check(primary),
        "replay": _portable_joblib_check(replay),
    }
    operational_paths = [
        "runtime.json",
        *(f"journals/{phase}.json" for phase in (
            "fit",
            "calibrate_scores",
            "validate",
            "monitor_apply",
            "monitor_evaluate",
        )),
    ]
    if canonical:
        operational_paths.extend(
            [
                "generation_receipt.json",
                "preparation_receipt.json",
                "inference/access_receipt.json",
                "journals/prepare.json",
            ]
        )
    operational = {
        path: (replay / path).is_file()
        and (primary / path).is_file()
        and _normalize_operational(read_json(replay / path))
        == _normalize_operational(read_json(primary / path))
        for path in operational_paths
    }
    all_exact = (
        all(exact.values())
        and all(arrays.values())
        and all(secrets.values())
        and all(functional["primary"].values())
        and all(functional["replay"].values())
        and all(operational.values())
    )
    result = {
        "schema_version": "wave59-replay-comparison-v1",
        "scientific_exact": exact,
        "scientific_array_exact": arrays,
        "secret_sha256_exact": secrets,
        "functional_state_exact": functional,
        "operational_semantic": operational,
        "all_exact": bool(all_exact),
    }
    if not all_exact:
        raise RuntimeError("Wave 59 replay differs from primary")
    return result


def _finalize_replay_condition(run_dir: Path, replay_exact: bool) -> None:
    analysis = read_json(run_dir / "analysis.json")
    for payload in analysis["prospective_patterns"].values():
        payload["conditions"]["replay_exact"] = bool(replay_exact)
        values = list(payload["conditions"].values())
        payload["replay_exact"] = bool(replay_exact)
        payload["aggregate_with_replay"] = (
            "NOT_EVALUABLE"
            if "NOT_EVALUABLE" in values or "PENDING" in values
            else bool(all(values))
        )
    write_json(run_dir / "analysis.json", analysis, mode=0o444)
    _write_report(run_dir)
    journal_path = run_dir / "journals/monitor_evaluate.json"
    journal = read_json(journal_path)
    if set(journal.get("output_sha256", {})) != {
        "analysis.json",
        "analysis_arrays.npz",
        "bootstrap_indices.npz",
    }:
        raise RuntimeError("Wave 59 monitor journal output coverage drifted")
    journal["output_sha256"]["analysis.json"] = sha256_file(run_dir / "analysis.json")
    write_json(journal_path, journal, mode=0o444)


def _artifact_classes(
    run_dir: Path, *, run_role: str, recovery_context: bool
) -> dict[str, list[str]]:
    exact, arrays = _scientific_paths(canonical=True)
    if recovery_context:
        exact.append("recovery_amendment.json")
    model_ids = [
        model_id("ridge"),
        model_id("hgb"),
        *(model_id(guard, target) for guard in ("logistic", "hgb") for target in ("harm", "posterior_incompatibility")),
        *(model_id("control-hgb", "harm", seed) for seed in HARM_CONTROL_SEEDS),
        *(model_id("control-hgb", "posterior_incompatibility", seed) for seed in INCOMPATIBILITY_CONTROL_SEEDS),
    ]
    operational = [
        "generation_receipt.json",
        "preparation_receipt.json",
        "inference/access_receipt.json",
        *(f"journals/{phase}.json" for phase in (
            "prepare",
            "fit",
            "calibrate_scores",
            "validate",
            "monitor_apply",
            "monitor_evaluate",
        )),
        "runtime.json",
    ]
    if run_role == "replay":
        operational.append("preparation_replay.json")
    secret = [
        "generation_escrow.json",
        *(f"benchmark/sealed/{name}" for name in (
            "calibration_null.jsonl",
            "train.jsonl",
            "val.jsonl",
            "lockbox.jsonl",
            "generation_secret.json",
            "identity_secret.json",
            "semantic_commitment_secret.json",
        )),
        "prepared/gate_fit_bundle.npz",
        "prepared/gate_select_truth_bundle.npz",
        "prepared/sealed_monitor_truth_bundle.npz",
    ]
    self_reference = ["artifact_manifest.json"]
    if run_role == "replay":
        self_reference.append("replay_comparison.json")
    return {
        "scientific_exact": sorted(exact),
        "scientific_array_exact": sorted(arrays),
        "functional_state": sorted(
            ["fit/model_states/manifest.json", *(f"fit/model_states/{identifier}.joblib" for identifier in model_ids)]
        ),
        "operational_semantic": sorted(operational),
        "secret_excluded_from_public_manifest": sorted(secret),
        "self_reference": sorted(self_reference),
    }


def _artifact_phase(relative: str) -> int | None:
    if relative.startswith("fit/") or relative == "journals/fit.json":
        return 1
    if relative.startswith("calibration/") or relative == "journals/calibrate_scores.json":
        return 2
    if relative.startswith("validation/") or relative == "journals/validate.json":
        return 3
    if relative in {
        "adjudication/monitor_scores.npz",
        "adjudication/monitor_policy_arrays.npz",
        "adjudication/monitor_action_freeze.json",
        "adjudication/monitor_not_evaluable.json",
        "journals/monitor_apply.json",
    }:
        return 4
    if relative in {
        "adjudication/bootstrap_indices.npz",
        "adjudication/analysis_arrays.npz",
        "analysis.json",
        "REPORT.md",
        "journals/monitor_evaluate.json",
        "replay_comparison.json",
    }:
        return 5
    if relative in {"runtime.json", "artifact_manifest.json"}:
        return None
    return 0


def _terminal_artifact_classes(
    classes: dict[str, list[str]], terminal_phase: str
) -> dict[str, list[str]]:
    phase_index = {
        "fit": 1,
        "calibrate_scores": 2,
        "monitor_apply": 4,
    }
    terminal_output = {
        "fit": "fit/fit_not_evaluable.json",
        "calibrate_scores": "calibration/calibration_not_evaluable.json",
        "monitor_apply": "adjudication/monitor_not_evaluable.json",
    }
    if terminal_phase not in phase_index:
        raise RuntimeError(f"Wave 59 unsupported NOT_EVALUABLE phase: {terminal_phase}")
    stop = phase_index[terminal_phase]
    required_at_stop = {
        terminal_output[terminal_phase],
        f"journals/{terminal_phase}.json",
    }
    result: dict[str, list[str]] = {}
    for class_name, paths in classes.items():
        selected = []
        for relative in paths:
            index = _artifact_phase(relative)
            if index is None or index < stop or (index == stop and relative in required_at_stop):
                selected.append(relative)
        result[class_name] = sorted(selected)
    return result


def write_artifact_manifest(run_dir: Path, *, run_role: str) -> dict[str, Any]:
    recovery_context = _authenticated_recovery_context(run_dir)
    classes = _artifact_classes(
        run_dir, run_role=run_role, recovery_context=recovery_context
    )
    runtime = read_json(run_dir / "runtime.json") if (run_dir / "runtime.json").is_file() else {}
    terminal_status = str(runtime.get("status", "COMPLETE"))
    missing_through_terminal: set[str] = set()
    future: set[str] = set()
    if terminal_status == "NOT_EVALUABLE":
        terminal_outputs = {
            "fit/fit_not_evaluable.json",
            "calibration/calibration_not_evaluable.json",
            "adjudication/monitor_not_evaluable.json",
        }
        classes["scientific_exact"] = sorted(
            set(classes["scientific_exact"]) | terminal_outputs
        )
        journals = []
        for phase in ("prepare", "fit", "calibrate_scores", "validate", "monitor_apply"):
            path = run_dir / "journals" / f"{phase}.json"
            if path.is_file():
                journals.append(read_json(path))
        analytical = [row for row in journals if row.get("phase") != "prepare"]
        if not analytical or analytical[-1].get("status") != "NOT_EVALUABLE":
            raise RuntimeError("Wave 59 terminal manifest lacks its terminal journal")
        terminal_phase = str(analytical[-1]["phase"])
        missing_through_terminal, future = _failure_coverage(run_dir, journals, classes)
        classes = _terminal_artifact_classes(classes, terminal_phase)
    flattened = [path for values in classes.values() for path in values]
    if len(flattened) != len(set(flattened)):
        raise RuntimeError("Wave 59 artifact classes overlap")
    actual = {
        str(path.relative_to(run_dir))
        for path in run_dir.rglob("*")
        if path.is_file()
    }
    expected = set(flattened)
    missing = (expected - (actual | {"artifact_manifest.json"})) | missing_through_terminal
    extra = (actual - expected) | future
    if missing or extra:
        raise RuntimeError(
            f"Wave 59 closed artifact inventory mismatch: missing={sorted(missing)}, extra={sorted(extra)}"
        )
    public: dict[str, Any] = {}
    for class_name, paths in classes.items():
        if class_name == "secret_excluded_from_public_manifest":
            public[class_name] = {"count": len(paths), "paths_redacted": True}
            continue
        if class_name == "self_reference":
            public[class_name] = {"paths": paths, "hashes_omitted": True}
            continue
        public[class_name] = {
            path: {"sha256": sha256_file(run_dir / path), "bytes": (run_dir / path).stat().st_size}
            for path in paths
        }
    config = read_json(run_dir / "config.snapshot.json")
    manifest = {
        "schema_version": "wave59-artifact-manifest-v1",
        "closed_world": True,
        "run_role": run_role,
        "recovery_context": recovery_context,
        "terminal_status": terminal_status,
        "plan_sha256": config["plan"]["sha256"],
        "accepted_plan_audit_sha256": config["accepted_plan_audit"]["sha256"],
        "config_sha256": sha256_file(run_dir / "config.snapshot.json"),
        "classes": public,
        "coverage": {
            "missing": [],
            "extra": [],
            "overlap": [],
            "unclassified": [],
            "file_count": len(expected),
        },
    }
    write_json(run_dir / "artifact_manifest.json", manifest, mode=0o444)
    return manifest


def archive_failed_attempt(
    run_dir: Path,
    error: BaseException,
    *,
    run_role: str,
    recovery_context: bool,
    attestation_private_key: Path = DEFAULT_PRIVATE_KEY,
    trusted_public_key: Path = TRUSTED_PUBLIC_KEY,
) -> Path:
    """Preserve a failed canonical attempt without exposing escrow contents."""
    run_dir = run_dir.resolve(strict=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    archived = run_dir.with_name(f"{run_dir.name}.failed_{stamp}")
    journals = []
    journal_root = run_dir / "journals"
    if journal_root.is_dir():
        for phase in (
            "prepare",
            "fit",
            "calibrate_scores",
            "validate",
            "monitor_apply",
            "monitor_evaluate",
        ):
            path = journal_root / f"{phase}.json"
            if path.is_file():
                journals.append(read_json(path))
    last = journals[-1] if journals else None
    message_hash = hashlib.sha256(str(error).encode("utf-8")).hexdigest()
    write_json(
        run_dir / "FAILURE.json",
        {
            "schema_version": "wave59-failed-attempt-v1",
            "error_type": type(error).__name__,
            "error_message_sha256": message_hash,
            "last_state": last.get("status") if last else None,
            "maximum_truth_materialized": (
                last.get("maximum_truth_materialized") if last else "none"
            ),
            "run_role": run_role,
            "recovery_context": bool(recovery_context),
            "original_path": str(run_dir),
            "archived_path": str(archived),
        },
        mode=0o600,
    )
    classes = _artifact_classes(
        run_dir, run_role=run_role, recovery_context=recovery_context
    )
    terminal_outputs = {
        "fit/fit_not_evaluable.json",
        "calibration/calibration_not_evaluable.json",
        "adjudication/monitor_not_evaluable.json",
    }
    classes["scientific_exact"] = sorted(
        set(classes["scientific_exact"]) | terminal_outputs
    )
    class_by_path = {
        path: class_name for class_name, paths in classes.items() for path in paths
    }
    records = []
    unknown = []
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file() or path.name in {
            "failure_inventory.json",
            "failure_attestation.json",
        }:
            continue
        relative = str(path.relative_to(run_dir))
        class_name = (
            "failure_record"
            if relative == "FAILURE.json"
            else class_by_path.get(relative)
        )
        if class_name is None:
            unknown.append(relative)
            continue
        record: dict[str, Any] = {
            "path": relative,
            "class": class_name,
            "bytes": path.stat().st_size,
        }
        record["sha256"] = sha256_file(path)
        records.append(record)
    if unknown:
        raise RuntimeError(f"failed-attempt inventory has unclassified paths: {unknown}")
    missing_required, future = _failure_coverage(run_dir, journals, classes)
    write_json(
        run_dir / "failure_inventory.json",
        {
            "schema_version": "wave59-failure-inventory-v1",
            "records": records,
            "failure_records": [
                "FAILURE.json",
                "failure_inventory.json",
                "failure_attestation.json",
            ],
            "missing_required_through_last_journal": sorted(missing_required),
            "extra": sorted(future),
            "overlap": [],
            "unclassified": [],
        },
        mode=0o600,
    )
    attestation = sign_attestation(
        {
            "schema_version": "wave59-failure-anchor-v1",
            "failure_inventory_sha256": sha256_file(
                run_dir / "failure_inventory.json"
            ),
            "failure_sha256": sha256_file(run_dir / "FAILURE.json"),
            "archived_path": str(archived),
        },
        attestation_private_key.resolve(strict=True),
        trusted_public_key.resolve(strict=True),
    )
    write_json(run_dir / "failure_attestation.json", attestation, mode=0o600)
    os.replace(run_dir, archived)
    _fsync_directory(archived.parent)
    return archived


def _failure_coverage(
    run_dir: Path,
    journals: list[dict[str, Any]],
    classes: dict[str, list[str]],
) -> tuple[set[str], set[str]]:
    order = [
        "prepare",
        "fit",
        "calibrate_scores",
        "validate",
        "monitor_apply",
        "monitor_evaluate",
    ]
    phases = [str(row.get("phase")) for row in journals]
    analytical = [phase for phase in phases if phase != "prepare"]
    if analytical != order[1 : 1 + len(analytical)]:
        return {"journals:non_monotonic"}, set()
    actual = {
        str(path.relative_to(run_dir))
        for path in run_dir.rglob("*")
        if path.is_file()
    }
    required: set[str] = set()
    allowed: set[str] = {
        "config.snapshot.json",
        "source_bindings.json",
        "preparation_freeze.json",
        "recovery_amendment.json",
        "FAILURE.json",
        "failure_inventory.json",
    }
    if "prepare" in phases:
        prepare_prefixes = (
            "benchmark/",
            "inference/",
            "prepared/",
        )
        prepare_names = {
            "pre_generation_freeze.json",
            "generation_escrow.json",
            "generation_receipt.json",
            "preparation_receipt.json",
            "preparation_replay.json",
            "journals/prepare.json",
        }
        preparation_paths = {
            path
            for values in classes.values()
            for path in values
            if path in prepare_names or path.startswith(prepare_prefixes)
        }
        required |= preparation_paths
        required |= {
            "config.snapshot.json",
            "source_bindings.json",
            "preparation_freeze.json",
            "pre_generation_freeze.json",
            "generation_escrow.json",
        }
        allowed |= preparation_paths
    destination_by_phase = {
        "fit": "fit",
        "calibrate_scores": "calibration",
        "validate": "validation",
        "monitor_apply": "adjudication",
        "monitor_evaluate": "adjudication",
    }
    for journal in journals:
        phase = str(journal.get("phase"))
        allowed.add(f"journals/{phase}.json")
        required.add(f"journals/{phase}.json")
        if phase == "prepare":
            continue
        destination = destination_by_phase[phase]
        for relative in journal.get("output_sha256", {}):
            promoted = (
                "analysis.json"
                if phase == "monitor_evaluate" and relative == "analysis.json"
                else f"{destination}/{relative}"
            )
            required.add(promoted)
        phase_prefix = f"{destination}/"
        allowed |= {
            path
            for values in classes.values()
            for path in values
            if path.startswith(phase_prefix)
        }
        if phase == "monitor_evaluate":
            allowed |= {"analysis.json", "REPORT.md", "runtime.json", "artifact_manifest.json", "replay_comparison.json"}
    last_index = max((order.index(phase) for phase in phases), default=-1)
    path_phase = {
        "fit/": 1,
        "calibration/": 2,
        "validation/": 3,
        "adjudication/monitor_scores.npz": 4,
        "adjudication/monitor_policy_arrays.npz": 4,
        "adjudication/monitor_action_freeze.json": 4,
        "adjudication/monitor_not_evaluable.json": 4,
        "adjudication/bootstrap_indices.npz": 5,
        "adjudication/analysis_arrays.npz": 5,
        "analysis.json": 5,
        "REPORT.md": 5,
        "replay_comparison.json": 5,
    }
    future = set()
    for relative in actual:
        for prefix, index in path_phase.items():
            if relative == prefix or relative.startswith(prefix):
                if index > last_index:
                    future.add(relative)
                break
    return required - actual, future


def _terminal_runtime(output: Path, status: str, started: float) -> None:
    config = read_json(output / "config.snapshot.json")
    preparation_duration = 0.0
    preparation_receipt = output / "preparation_receipt.json"
    if preparation_receipt.is_file():
        preparation_duration = float(
            read_json(preparation_receipt)
            .get("coordinator_budget", {})
            .get("duration_seconds", 0.0)
        )
    phase_rss = []
    for phase in ("fit", "calibrate_scores", "validate", "monitor_apply", "monitor_evaluate"):
        journal = output / "journals" / f"{phase}.json"
        if journal.is_file():
            phase_rss.append(int(read_json(journal).get("max_rss_bytes", 0)))
    analytical_duration = time.monotonic() - started
    write_json(
        output / "runtime.json",
        {
            "status": status,
            "device": "cpu",
            "cuda_visible_devices": "",
            "duration_seconds": analytical_duration,
            "preparation_duration_seconds": preparation_duration,
            "total_run_seconds": preparation_duration + analytical_duration,
            "max_rss_bytes": max(phase_rss, default=0),
            "budget": config["runtime_budget"],
            "budget_enforced": True,
            "phases": [
                "fit",
                "calibrate_scores",
                "validate",
                "monitor_apply",
                "monitor_evaluate",
            ],
        },
    )


def _ensure_preparation_authority(
    output: Path, bundle_root: Path, config: dict[str, Any], config_path: Path
) -> Path:
    path = output / "preparation_freeze.json"
    bundle_names = (
        "gate_fit_bundle.npz",
        "gate_select_truth_bundle.npz",
        "gate_select_inference_bundle.npz",
        "sealed_monitor_truth_bundle.npz",
        "sealed_monitor_inference_bundle.npz",
    )
    observed = {
        f"prepared/{name}": sha256_file((bundle_root / name).resolve(strict=True))
        for name in bundle_names
    }
    if not path.exists():
        if config.get("status") == FROZEN_STATUS:
            raise RuntimeError("frozen Wave 59 execution lacks preparation_freeze.json")
        write_json(
            path,
            {
                "schema_version": "wave59-isolated-preparation-freeze-v1",
                "status": "TEST_ONLY_PREPARED_BUNDLES",
                "prepared_bundle_hashes": observed,
            },
            mode=0o444,
        )
    freeze = read_json(path.resolve(strict=True))
    if freeze.get("prepared_bundle_hashes") != observed:
        raise RuntimeError("Wave 59 prepared bundles differ from preparation freeze")
    if config.get("status") == FROZEN_STATUS:
        raw_sources = {
            relative: sha256_file((REPO_ROOT / relative).resolve(strict=True))
            for relative in config["required_execution_sources"]
        }
        if (
            freeze.get("config_sha256")
            != sha256_file(config_path.resolve(strict=True))
            or freeze.get("prospective_config") != config
            or freeze.get("sources") != raw_sources
            or freeze.get("source_bindings") != config["source_binding"]
        ):
            raise RuntimeError("Wave 59 preparation authority differs from frozen execution")
    return path


def _validate_phase_provenance(
    freeze: dict[str, Any], output: Path, config_path: Path, preparation: Path
) -> None:
    validate_freeze_provenance(
        freeze,
        config_path=config_path,
        source_bindings=output / "source_bindings.json",
        preparation=preparation,
    )


def _finalize_terminal(
    output: Path, status: str, started: float, *, run_role: str, canonical: bool
) -> Path:
    _terminal_runtime(output, status, started)
    if canonical:
        write_artifact_manifest(output, run_role=run_role)
    return output


def execute(
    prepared: Path,
    policy_manifest: Path,
    output: Path,
    config_path: Path,
    reference_dir: Path | None = None,
) -> Path:
    started = time.monotonic()
    config = read_json(config_path)
    validate_pre_draw_config(config)
    validate_execution_bindings(config, config_path)
    run_role = "replay" if reference_dir is not None else "primary"
    prepared_resolved = prepared.resolve(strict=True)
    preparation_duration = 0.0
    preparation_receipt_path = prepared_resolved / "preparation_receipt.json"
    if preparation_receipt_path.is_file():
        preparation_duration = float(
            read_json(preparation_receipt_path)
            .get("coordinator_budget", {})
            .get("duration_seconds", 0.0)
        )
    max_run_seconds = (
        float(config["runtime_budget"]["max_seconds_per_run"])
        - preparation_duration
    )
    if max_run_seconds <= 0:
        raise RuntimeError("Wave 59 preparation exhausted per-run wall-time budget")
    if reference_dir is not None:
        reference_root = reference_dir.resolve(strict=True)
        reference_runtime_path = reference_root / "runtime.json"
        reference_duration = float(read_json(reference_runtime_path)["duration_seconds"])
        reference_preparation = reference_root / "preparation_receipt.json"
        if reference_preparation.is_file():
            reference_duration += float(
                read_json(reference_preparation)
                .get("coordinator_budget", {})
                .get("duration_seconds", 0.0)
            )
        combined_remaining = float(
            config["runtime_budget"]["max_seconds_primary_plus_replay"]
        ) - reference_duration - preparation_duration
        if combined_remaining <= 0:
            raise RuntimeError("Wave 59 primary already exhausted combined runtime budget")
        max_run_seconds = min(max_run_seconds, combined_remaining)
    deadline = started + max_run_seconds
    prepared = prepared_resolved
    output = output.resolve(strict=False)
    canonical_existing = output == prepared and (prepared / "prepared").is_dir()
    bundle_root = prepared / "prepared" if canonical_existing else prepared
    if output.exists() and not canonical_existing:
        raise FileExistsError(output)
    if not output.exists():
        output.mkdir(parents=True)
    snapshot = output / "config.snapshot.json"
    if snapshot.exists():
        if sha256_file(snapshot) != sha256_file(config_path):
            raise RuntimeError("Wave 59 config snapshot differs from execution config")
    else:
        _copy(config_path, snapshot)
    if not (output / "source_bindings.json").exists():
        write_json(output / "source_bindings.json", config["source_binding"])
    elif read_json(output / "source_bindings.json") != config["source_binding"]:
        raise RuntimeError("Wave 59 source bindings differ from execution config")
    preparation = _ensure_preparation_authority(output, bundle_root, config, config_path)
    if config.get("status") == FROZEN_STATUS:
        if sha256_file(policy_manifest.resolve(strict=True)) != config["source_binding"][
            "wave52_policy_manifest_sha256"
        ]:
            raise RuntimeError("Wave 59 policy manifest differs from frozen upstream")
    utilities = load_utilities(policy_manifest)
    with tempfile.TemporaryDirectory(prefix="wave59-utilities-", dir="/tmp") as raw:
        utilities_path = Path(raw) / "utilities.npy"
        np.save(utilities_path, utilities)
        fit, fit_journal = _reuse_or_run_phase(
            output,
            "fit",
            {
                "config.json": config_path,
                "source_bindings.json": output / "source_bindings.json",
                "preparation_freeze.json": preparation,
                "bundle.npz": bundle_root / "gate_fit_bundle.npz",
                "utilities.npy": utilities_path,
            },
            [bundle_root / "gate_select_truth_bundle.npz", bundle_root / "sealed_monitor_truth_bundle.npz"],
            destination_name="fit",
            deadline=deadline,
        )
        if fit_journal["status"] != "FIT_COMPLETE":
            return _finalize_terminal(
                output, fit_journal["status"], started, run_role=run_role,
                canonical=canonical_existing,
            )
        validate_execution_bindings(config, config_path)
        calibration, calibration_journal = _reuse_or_run_phase(
            output,
            "calibrate_scores",
            {
                "config.json": config_path,
                "source_bindings.json": output / "source_bindings.json",
                "preparation_freeze.json": preparation,
                "inference_bundle.npz": bundle_root / "gate_select_inference_bundle.npz",
                "model_states_manifest.json": fit / "model_states/manifest.json",
                "model_state_arrays.npz": fit / "model_state_arrays.npz",
                "fit_freeze.json": fit / "fit_freeze.json",
            },
            [bundle_root / "gate_select_truth_bundle.npz", bundle_root / "sealed_monitor_truth_bundle.npz"],
            destination_name="calibration",
            deadline=deadline,
        )
        if calibration_journal["status"] != "CALIBRATION_FROZEN":
            return _finalize_terminal(
                output, calibration_journal["status"], started, run_role=run_role,
                canonical=canonical_existing,
            )
        validate_execution_bindings(config, config_path)
        calibration_freeze = validate_freeze_bindings(
            calibration / "calibration_freeze.json",
            phase="calibrate_scores",
            bindings={
                "validation_scores.npz": calibration / "validation_scores.npz",
                "validation_policy_arrays.npz": calibration
                / "validation_policy_arrays.npz",
            },
            require_all=True,
        )
        _validate_phase_provenance(
            calibration_freeze, output, config_path, preparation
        )
        validation, validation_journal = _reuse_or_run_phase(
            output,
            "validate",
            {
                "config.json": config_path,
                "source_bindings.json": output / "source_bindings.json",
                "preparation_freeze.json": preparation,
                "inference_bundle.npz": bundle_root / "gate_select_inference_bundle.npz",
                "truth_bundle.npz": bundle_root / "gate_select_truth_bundle.npz",
                "validation_scores.npz": calibration / "validation_scores.npz",
                "validation_policy_arrays.npz": calibration / "validation_policy_arrays.npz",
                "calibration_freeze.json": calibration / "calibration_freeze.json",
                "utilities.npy": utilities_path,
            },
            [bundle_root / "sealed_monitor_truth_bundle.npz"],
            destination_name="validation",
            deadline=deadline,
        )
        if validation_journal["status"] != "VALIDATION_COMPLETE":
            return _finalize_terminal(
                output, validation_journal["status"], started, run_role=run_role,
                canonical=canonical_existing,
            )
        validate_execution_bindings(config, config_path)
        validation_freeze = validate_freeze_bindings(
            validation / "validation_freeze.json",
            phase="validate",
            bindings={
                "validation_metrics.npz": validation / "validation_metrics.npz",
                "validation_summary.json": validation / "validation_summary.json",
            },
            require_all=True,
        )
        _validate_phase_provenance(validation_freeze, output, config_path, preparation)
        adjudication, apply_journal = _reuse_or_run_phase(
            output,
            "monitor_apply",
            {
                "config.json": config_path,
                "source_bindings.json": output / "source_bindings.json",
                "preparation_freeze.json": preparation,
                "inference_bundle.npz": bundle_root / "sealed_monitor_inference_bundle.npz",
                "model_states_manifest.json": fit / "model_states/manifest.json",
                "model_state_arrays.npz": fit / "model_state_arrays.npz",
                "fit_freeze.json": fit / "fit_freeze.json",
                "calibration_freeze.json": calibration / "calibration_freeze.json",
                "validation_freeze.json": validation / "validation_freeze.json",
            },
            [bundle_root / "sealed_monitor_truth_bundle.npz"],
            destination_name="adjudication",
            deadline=deadline,
        )
        if apply_journal["status"] != "MONITOR_ACTIONS_FROZEN":
            return _finalize_terminal(
                output, apply_journal["status"], started, run_role=run_role,
                canonical=canonical_existing,
            )
        validate_execution_bindings(config, config_path)
        action_freeze = validate_freeze_bindings(
            adjudication / "monitor_action_freeze.json",
            phase="monitor_apply",
            bindings={
                "monitor_scores.npz": adjudication / "monitor_scores.npz",
                "monitor_policy_arrays.npz": adjudication
                / "monitor_policy_arrays.npz",
            },
            require_all=True,
        )
        _validate_phase_provenance(action_freeze, output, config_path, preparation)
        if action_freeze.get("fit_freeze_sha256") != sha256_file(
            fit / "fit_freeze.json"
        ) or action_freeze.get("calibration_freeze_sha256") != sha256_file(
            calibration / "calibration_freeze.json"
        ) or action_freeze.get("validation_freeze_sha256") != sha256_file(
            validation / "validation_freeze.json"
        ):
            raise RuntimeError("Wave 59 monitor action freeze chain drifted")
        evaluate_inputs = {
            "config.json": config_path,
            "source_bindings.json": output / "source_bindings.json",
            "preparation_freeze.json": preparation,
            "truth_bundle.npz": bundle_root / "sealed_monitor_truth_bundle.npz",
            "monitor_policy_arrays.npz": adjudication / "monitor_policy_arrays.npz",
            "monitor_action_freeze.json": adjudication / "monitor_action_freeze.json",
            "utilities.npy": utilities_path,
        }
        evaluate_journal_path = output / "journals/monitor_evaluate.json"
        evaluation_already_promoted = (
            evaluate_journal_path.is_file()
            and (output / "analysis.json").is_file()
            and (adjudication / "bootstrap_indices.npz").is_file()
            and (adjudication / "analysis_arrays.npz").is_file()
        )
        if evaluation_already_promoted:
            evaluate_journal = read_json(evaluate_journal_path)
            if evaluate_journal.get("input_sha256") != {
                name: sha256_file(path.resolve(strict=True))
                for name, path in sorted(evaluate_inputs.items())
            }:
                raise RuntimeError("Wave 59 promoted monitor evaluation inputs differ")
            _validate_promoted_evaluation_outputs(output, evaluate_journal)
            evaluation = None
        else:
            evaluation, evaluate_journal = _reuse_or_run_phase(
                output,
                "monitor_evaluate",
                evaluate_inputs,
                [],
                destination_name=".monitor_evaluate.complete",
                deadline=deadline,
            )
        if evaluate_journal["status"] != "COMPLETE":
            return _finalize_terminal(
                output, evaluate_journal["status"], started, run_role=run_role,
                canonical=canonical_existing,
            )
        if evaluation is not None:
            _merge_evaluation(adjudication, evaluation, output)
    if time.monotonic() > deadline:
        raise RuntimeError("Wave 59 run exceeded wall-time budget")
    _terminal_runtime(output, "COMPLETE", started)
    if reference_dir is not None:
        runtime = read_json(output / "runtime.json")
        reference_runtime = read_json(reference_dir.resolve(strict=True) / "runtime.json")
        reference_preparation_duration = 0.0
        reference_preparation = reference_dir.resolve(strict=True) / "preparation_receipt.json"
        if reference_preparation.is_file():
            reference_preparation_duration = float(
                read_json(reference_preparation)
                .get("coordinator_budget", {})
                .get("duration_seconds", 0.0)
            )
        runtime["primary_plus_replay_seconds"] = (
            float(reference_runtime["duration_seconds"])
            + reference_preparation_duration
            + float(runtime["duration_seconds"])
            + preparation_duration
        )
        runtime["preparation_duration_seconds"] = preparation_duration
        runtime["combined_budget_seconds"] = int(
            config["runtime_budget"]["max_seconds_primary_plus_replay"]
        )
        if runtime["primary_plus_replay_seconds"] > runtime["combined_budget_seconds"]:
            raise RuntimeError("Wave 59 primary plus replay exceeded combined runtime budget")
        write_json(output / "runtime.json", runtime)
    _write_report(output)
    canonical = (output / "benchmark").is_dir()
    if reference_dir is not None:
        reference = reference_dir.resolve(strict=True)
        comparison = compare_runs(output, reference)
        write_json(output / "replay_comparison.json", comparison, mode=0o444)
        _finalize_replay_condition(reference, comparison["all_exact"])
        _finalize_replay_condition(output, comparison["all_exact"])
        if canonical:
            write_artifact_manifest(reference, run_role="primary")
            write_artifact_manifest(output, run_role="replay")
    elif canonical:
        write_artifact_manifest(output, run_role="primary")
    return output


def main() -> None:
    args = parse_args()
    if args.attestation_private_key.is_symlink():
        raise RuntimeError("Wave 59 failure-attestation private key cannot be a symlink")
    sign_attestation(
        {"schema_version": "wave59-failure-key-preflight-v1"},
        args.attestation_private_key.resolve(strict=True),
        TRUSTED_PUBLIC_KEY.resolve(strict=True),
    )
    output = args.output_dir.resolve(strict=False)
    prepared_arg = args.prepared_dir.resolve(strict=True)
    if args.resume_from is not None:
        restore_identical_hash_attempt(
            args.resume_from.resolve(strict=True),
            output,
            args.config.resolve(strict=True),
        )
        prepared_arg = output.resolve(strict=True)
    try:
        execute(
            prepared_arg,
            args.policy_manifest.resolve(strict=True),
            output,
            args.config.resolve(strict=True),
            args.reference_dir.resolve(strict=True) if args.reference_dir else None,
        )
    except BaseException as error:
        if output.is_dir() and (output / "config.snapshot.json").is_file():
            recovery_context = (output / "recovery_amendment.json").is_file()
            run_role = "replay" if args.reference_dir else "primary"
            archived = archive_failed_attempt(
                output,
                error,
                run_role=run_role,
                recovery_context=recovery_context,
                attestation_private_key=args.attestation_private_key,
            )
            print(
                json.dumps(
                    {"status": "FAILED", "archived_attempt": str(archived)},
                    sort_keys=True,
                ),
                file=sys.stderr,
            )
        raise


if __name__ == "__main__":
    main()
