#!/usr/bin/env python3
"""Coordinate Wave 60 source-law, score/apply, evaluation, and pair sealing.

The coordinator may copy and sign artifacts but never computes scientific
scores or metrics.  Those operations occur in the unprivileged phase worker.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import itertools
import json
import os
from pathlib import Path
import pwd
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterable, Mapping

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_name] = "4"

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from geometria_proporcional.wave49_attestation import (  # noqa: E402
    sign_attestation,
    verify_attestation,
)
from geometria_proporcional.wave60_frozen_policy_transport import (  # noqa: E402
    EVALUATE_SCHEMA,
    PHASE_FILES,
    SCORE_APPLY_SCHEMA,
    SOURCE_HASHES,
    SOURCE_LAW_SCHEMA,
    file_sha256,
    finalize_patterns,
    require_exact_keys,
    validate_pre_draw_config,
)

WORKER_SOURCE = REPO_ROOT / "experiments/geometria_proporcional/_wave60_phase_worker.py"
DEFAULT_PRIVATE_KEY = Path("/root/.config/phideus/wave49_attestation_private.pem")
TRUSTED_PUBLIC_KEY = (
    REPO_ROOT / "experiments/geometria_proporcional/keys/wave49_attestation_public.pem"
)
SOURCE_AUTHORITY_DEFAULT = (
    REPO_ROOT
    / "data/geometria_proporcional/wave60_frozen_policy_transport_source_law_v1"
)
ATTEMPT_DEFAULT = (
    REPO_ROOT / "data/geometria_proporcional/wave60_frozen_policy_transport_attempt_v1"
)
ANTECEDENT_DRAW_ROOTS = tuple(
    REPO_ROOT / "data/geometria_proporcional" / name
    for name in (
        "wave59_fresh_hgb_guard_bracket_replay_normalized_v1",
        "wave59_fresh_hgb_guard_bracket_replay_normalized_v1_replay",
        "wave59_fresh_hgb_guard_bracket_v1",
        "wave59_fresh_hgb_guard_bracket_v1.failed_20260905T071003529517Z",
        "wave59_fresh_hgb_guard_bracket_v1_replay.failed_20260905T102929791142Z",
    )
)
SOURCE_ROOT = (
    REPO_ROOT
    / "data/geometria_proporcional/wave59_fresh_hgb_guard_bracket_replay_normalized_v1"
)
POLICY_MANIFEST = (
    REPO_ROOT
    / "data/geometria_proporcional/wave52_policy_transport_v1/policy_manifest.json"
)
SOURCE_ALIASES = {
    "wave59_fit_freeze.json": SOURCE_ROOT / "fit/fit_freeze.json",
    "wave59_model_states_manifest.json": SOURCE_ROOT / "fit/model_states/manifest.json",
    "wave59_model_state_arrays.npz": SOURCE_ROOT / "fit/model_state_arrays.npz",
    "wave59_calibration_freeze.json": SOURCE_ROOT
    / "calibration/calibration_freeze.json",
    "wave59_monitor_inference_bundle.npz": SOURCE_ROOT
    / "prepared/sealed_monitor_inference_bundle.npz",
    "wave59_monitor_scores.npz": SOURCE_ROOT / "adjudication/monitor_scores.npz",
    "wave59_monitor_policy_arrays.npz": SOURCE_ROOT
    / "adjudication/monitor_policy_arrays.npz",
    "wave59_monitor_action_freeze.json": SOURCE_ROOT
    / "adjudication/monitor_action_freeze.json",
    "wave59_artifact_manifest.json": SOURCE_ROOT / "artifact_manifest.json",
    "wave59_config_snapshot.json": SOURCE_ROOT / "config.snapshot.json",
    "r454_audit.md": REPO_ROOT
    / "Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/454_wave59_successor_draw_final_audit.md",
}
RUNTIME_MODULES = (
    "__init__.py",
    "wave49_schema.py",
    "wave52_policy.py",
    "wave53_uncertainty.py",
    "wave55_policy_bridge.py",
    "wave56_contextual_gate.py",
    "wave58_open_diagnostic.py",
    "wave59_hgb_guard_bracket.py",
    "wave60_frozen_policy_transport.py",
)
RECEIPT_KEYS = {
    "schema_version",
    "phase",
    "status",
    "uid",
    "gid",
    "capabilities",
    "no_new_privs",
    "inputs",
    "outputs",
    "opened_paths",
    "denied_path_probes",
    "started_at",
    "completed_at",
}
SOURCE_SCIENTIFIC = (
    "source_law_freeze.json",
    "transport_law_manifest.json",
    "transport_law_arrays.npz",
    "frozen_policy_spec.json",
    "feature_schema.json",
)
SOURCE_COPY_FILES = (
    *SOURCE_SCIENTIFIC,
    "verify_source_law_receipt.json",
    "source_law_attestation.json",
)
SCORE_FILES = (
    "monitor_scores.npz",
    "monitor_policy_arrays.npz",
    "evaluation_index.npz",
    "monitor_action_freeze.json",
    "score_apply_receipt.json",
    "score_apply_attestation.json",
)
EVALUATION_FILES = (
    "bootstrap_indices.npz",
    "analysis_arrays.npz",
    "analysis.json",
    "evaluation_freeze.json",
    "evaluate_receipt.json",
    "evaluation_attestation.json",
)
PREPARED_BUNDLES = (
    "gate_fit_bundle.npz",
    "gate_select_inference_bundle.npz",
    "gate_select_truth_bundle.npz",
    "sealed_monitor_inference_bundle.npz",
    "sealed_monitor_truth_bundle.npz",
)


def now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_json(path: Path, payload: Any, *, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(
            payload, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False
        )
        + "\n"
    ).encode("utf-8")
    descriptor, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(raw)
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def ensure_json(path: Path, payload: Any, *, mode: int = 0o444) -> None:
    if path.exists():
        if path.is_symlink() or not path.is_file() or read_json(path) != payload:
            raise RuntimeError(f"Wave 60 staged JSON drifted: {path.name}")
        return
    write_json(path, payload, mode=mode)


def ensure_text(path: Path, payload: str, *, mode: int = 0o444) -> None:
    if path.exists():
        if (
            path.is_symlink()
            or not path.is_file()
            or path.read_text(encoding="utf-8") != payload
        ):
            raise RuntimeError(f"Wave 60 staged text drifted: {path.name}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    path.chmod(mode)
    fsync_directory(path.parent)


def copy_regular(source: Path, destination: Path, *, mode: int = 0o444) -> None:
    metadata = source.lstat()
    if source.is_symlink() or not source.is_file():
        raise RuntimeError(f"Wave 60 source is not a regular non-symlink: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as reader, destination.open("xb") as writer:
        shutil.copyfileobj(reader, writer, length=1024 * 1024)
        writer.flush()
        os.fsync(writer.fileno())
    destination.chmod(mode)
    if (
        file_sha256(source) != file_sha256(destination)
        or metadata.st_size != destination.stat().st_size
    ):
        raise RuntimeError(f"Wave 60 copy changed bytes: {source}")
    if (metadata.st_dev, metadata.st_ino) == (
        destination.stat().st_dev,
        destination.stat().st_ino,
    ):
        raise RuntimeError("Wave 60 copy unexpectedly created an inode alias")


def git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    return completed.stdout.strip()


def build_runtime(root: Path) -> Path:
    source = root / "runtime"
    package = source / "geometria_proporcional"
    package.mkdir(parents=True)
    for name in RUNTIME_MODULES:
        copy_regular(SRC_ROOT / "geometria_proporcional" / name, package / name)
    copy_regular(WORKER_SOURCE, source / WORKER_SOURCE.name)
    for path in source.rglob("*"):
        path.chmod(0o555 if path.is_dir() else 0o444)
    source.chmod(0o555)
    return source / WORKER_SOURCE.name


def _process_rss(pid: int) -> int:
    try:
        for line in (
            Path(f"/proc/{pid}/status").read_text(encoding="utf-8").splitlines()
        ):
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (FileNotFoundError, ProcessLookupError):
        return 0
    return 0


def _process_tree_rss(pid: int) -> int:
    pending = [pid]
    seen: set[int] = set()
    total = 0
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        total += _process_rss(current)
        try:
            children = Path(f"/proc/{current}/task/{current}/children").read_text(
                encoding="utf-8"
            )
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        pending.extend(int(value) for value in children.split())
    return total


def _terminate(process: subprocess.Popen[str]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=5)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def _validate_receipt(receipt: Mapping[str, Any], phase: str) -> None:
    require_exact_keys(receipt, RECEIPT_KEYS, f"{phase} receipt")
    if receipt["phase"] != phase or receipt["uid"] != 65534 or receipt["gid"] != 65534:
        raise RuntimeError("Wave 60 worker receipt identity drifted")
    if receipt["capabilities"] != "0000000000000000" or receipt["no_new_privs"] != 1:
        raise RuntimeError("Wave 60 worker privilege boundary failed")
    if not all(row.get("denied") is True for row in receipt["denied_path_probes"]):
        raise RuntimeError("Wave 60 worker accessed a forbidden path")


def stage_phase_request(stage: Path, phase: str, run_role: str) -> None:
    actual = {path.name for path in stage.iterdir() if path.is_file()}
    expected_without_request = set(PHASE_FILES[phase]) - {"phase_request.json"}
    if actual != expected_without_request:
        raise RuntimeError("Wave 60 coordinator phase staging drifted")
    request = {
        "schema_version": SOURCE_LAW_SCHEMA,
        "phase": phase,
        "allowed_files": sorted(PHASE_FILES[phase]),
        "sha256": {name: file_sha256(stage / name) for name in sorted(actual)},
        "run_role": run_role,
    }
    write_json(stage / "phase_request.json", request, mode=0o444)


def run_worker(
    workspace: Path,
    stage: Path,
    phase: str,
    probes: Iterable[Path],
    *,
    max_seconds: float = 900.0,
    max_rss: int = 1610612736,
) -> tuple[Path, dict[str, Any], float, int]:
    workspace.chmod(0o711)
    worker = build_runtime(workspace)
    output = workspace / "worker-output"
    output.mkdir(mode=0o700)
    account = pwd.getpwnam("nobody")
    for path in stage.iterdir():
        path.chmod(0o444)
    stage.chmod(0o555)
    os.chown(output, account.pw_uid, account.pw_gid)
    # The mapped uid inside bwrap must be able to write through the bind even
    # when the host user namespace maps ownership differently.
    output.chmod(0o777)
    bwrap = shutil.which("bwrap")
    setpriv = shutil.which("setpriv")
    if bwrap is None or setpriv is None:
        raise RuntimeError("Wave 60 requires bwrap and setpriv")
    command = [
        bwrap,
        "--die-with-parent",
        "--unshare-all",
        "--new-session",
        "--ro-bind",
        "/usr",
        "/usr",
        "--ro-bind",
        "/lib",
        "/lib",
        "--ro-bind",
        "/lib64",
        "/lib64",
        "--ro-bind",
        str((REPO_ROOT / "venv").resolve(strict=True)),
        "/venv",
        "--ro-bind",
        str((workspace / "runtime").resolve(strict=True)),
        "/runtime",
        "--ro-bind",
        str(stage.resolve(strict=True)),
        "/stage",
        "--bind",
        str(output.resolve(strict=True)),
        "/output",
        "--tmpfs",
        "/tmp",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--chdir",
        "/tmp",
        "--uid",
        str(account.pw_uid),
        "--gid",
        str(account.pw_gid),
        setpriv,
        "--no-new-privs",
        "/venv/bin/python",
        f"/runtime/{worker.name}",
        "--stage",
        "/stage",
        "--output",
        "/output",
        "--phase",
        phase,
    ]
    for probe in probes:
        command.extend(("--forbidden-probe", str(probe)))
    env = {
        "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PYTHONPATH": "/runtime",
        "HOME": "/tmp",
        "WAVE60_STAGED_RUNTIME": "1",
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "4",
        "OPENBLAS_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4",
        "NUMEXPR_NUM_THREADS": "4",
        "PYTHONHASHSEED": "0",
    }
    started = time.monotonic()
    process = subprocess.Popen(
        command,
        cwd=stage,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    peak = 0
    while True:
        peak = max(peak, _process_tree_rss(process.pid))
        if peak > max_rss or time.monotonic() - started > max_seconds:
            _terminate(process)
            raise RuntimeError(f"Wave 60 {phase} worker exceeded its CPU budget")
        try:
            stdout, stderr = process.communicate(timeout=0.2)
            break
        except subprocess.TimeoutExpired:
            continue
    if process.returncode:
        raise RuntimeError(f"Wave 60 {phase} worker failed: {stderr.strip()}")
    receipt_name = {
        "verify_source_law": "verify_source_law_receipt.json",
        "score_apply": "score_apply_receipt.json",
        "evaluate": "evaluate_receipt.json",
    }[phase]
    receipt = read_json(output / receipt_name)
    _validate_receipt(receipt, phase)
    return output, receipt, time.monotonic() - started, peak


def make_attestation(
    phase: str,
    payload: dict[str, Any],
    private_key: Path,
    public_key: Path = TRUSTED_PUBLIC_KEY,
) -> dict[str, Any]:
    signed = sign_attestation(
        payload, private_key.resolve(strict=True), public_key.resolve(strict=True)
    )
    verify_attestation(signed, public_key.resolve(strict=True))
    return {
        "schema_version": (
            SOURCE_LAW_SCHEMA
            if phase == "verify_source_law"
            else (SCORE_APPLY_SCHEMA if phase == "score_apply" else EVALUATE_SCHEMA)
        ),
        "phase": phase,
        "payload": payload,
        "public_key_fingerprint": signed["trusted_public_key_sha256"],
        "signature_base64": signed["signature_base64"],
    }


def verify_wave60_attestation(
    attestation: Mapping[str, Any], public_key: Path = TRUSTED_PUBLIC_KEY
) -> None:
    require_exact_keys(
        attestation,
        {
            "schema_version",
            "phase",
            "payload",
            "public_key_fingerprint",
            "signature_base64",
        },
        "attestation",
    )
    verify_attestation(
        {
            "algorithm": "Ed25519",
            "payload": attestation["payload"],
            "signature_base64": attestation["signature_base64"],
            "trusted_public_key_sha256": attestation["public_key_fingerprint"],
        },
        public_key.resolve(strict=True),
    )


def inventory(root: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError("Wave 60 package contains a symlink")
        if path.is_file():
            metadata = path.stat()
            result[str(path.relative_to(root))] = {
                "bytes": metadata.st_size,
                "sha256": file_sha256(path),
                "owner": metadata.st_uid,
                "group": metadata.st_gid,
                "mode": f"{metadata.st_mode & 0o777:04o}",
            }
    return result


def root_artifact_class(relative: str) -> str:
    """Assign every root-level path to the frozen Wave 60 artifact taxonomy."""
    name = Path(relative).name
    if relative == "artifact_manifest.json":
        return "SELF_REFERENCE"
    if relative in {
        "FAILURE.json",
        "failure_inventory.json",
        "failure_attestation.json",
    }:
        return "FAILURE_CONDITIONAL"
    if relative.startswith("failed_preparation/"):
        return "FAILURE_CONDITIONAL"
    if relative in {
        "config.snapshot.json",
        "source_bindings.json",
        "pre_generation_freeze.json",
    }:
        return "PRE_GENERATION_PUBLIC"
    if relative == "generation_escrow.json" or relative.startswith("benchmark/sealed/"):
        return "BENCHMARK_SEALED_SECRET"
    if relative.startswith("benchmark/"):
        return "BENCHMARK_PUBLIC"
    if relative.startswith("prepared/"):
        if name in {
            "gate_select_inference_bundle.npz",
            "sealed_monitor_inference_bundle.npz",
        }:
            return "PREPARED_INFERENCE_SAFE"
        if name in {
            "gate_fit_bundle.npz",
            "gate_select_truth_bundle.npz",
            "sealed_monitor_truth_bundle.npz",
        }:
            return "PREPARED_TRUTH_SECRET"
    if relative.startswith("inference/logits/"):
        return "PREPARED_INFERENCE_SAFE"
    if relative.startswith("source_law/"):
        return "SOURCE_LAW_FROZEN"
    if relative.startswith("score/"):
        if name.endswith("_receipt.json") or name.endswith("_attestation.json"):
            return "OPERATIONAL_JOURNAL"
        return "SCORE_APPLY_SCIENTIFIC"
    if relative.startswith("evaluation/"):
        if name.endswith("_receipt.json") or name.endswith("_attestation.json"):
            return "OPERATIONAL_JOURNAL"
        return "EVALUATION_SCIENTIFIC"
    if relative.startswith("journals/") or relative in {
        "generation_receipt.json",
        "preparation_receipt.json",
        "preparation_attestation.json",
        "preparation_replay.json",
        "inference/access_receipt.json",
    }:
        return "OPERATIONAL_JOURNAL"
    if relative in {"preparation_freeze.json", "runtime.json"}:
        return "FINAL_PUBLIC"
    raise RuntimeError(f"Wave 60 unclassified root artifact: {relative}")


def pair_artifact_class(relative: str) -> str:
    if relative == "artifact_manifest.json":
        return "SELF_REFERENCE"
    if relative in {
        "FAILURE.json",
        "failure_inventory.json",
        "failure_attestation.json",
    }:
        return "FAILURE_CONDITIONAL"
    if relative == "replay_comparison.json":
        return "REPLAY_COMPARISON_CONDITIONAL"
    if relative.startswith("journals/") or relative in {
        "replay_finalize_receipt.json",
        "replay_finalize_attestation.json",
    }:
        return "OPERATIONAL_JOURNAL"
    if relative in {
        "pair_status.json",
        "final_analysis.json",
        "replay_finalize_freeze.json",
        "REPORT.md",
        "runtime.json",
    }:
        return "FINAL_PUBLIC"
    raise RuntimeError(f"Wave 60 unclassified pair artifact: {relative}")


def validate_prepared_root(
    root: Path, role: str, expected_config: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate the complete COMMON boundary before inspecting draw identity."""
    required = (
        "config.snapshot.json",
        "source_bindings.json",
        "pre_generation_freeze.json",
        "generation_escrow.json",
        "generation_receipt.json",
        "preparation_freeze.json",
        "preparation_receipt.json",
        "preparation_attestation.json",
        "journals/prepare.json",
        "benchmark/manifest.json",
        "benchmark/protocol_config.json",
        *(f"prepared/{name}" for name in PREPARED_BUNDLES),
    )
    missing = [relative for relative in required if not (root / relative).is_file()]
    if missing or (root / "failed_preparation").exists():
        raise RuntimeError("INVALID_PREPARATION")
    if read_json(root / "config.snapshot.json") != dict(expected_config):
        raise RuntimeError("INVALID_PREPARATION")
    if (
        "source_binding" in expected_config
        and read_json(root / "source_bindings.json")
        != expected_config["source_binding"]
    ):
        raise RuntimeError("INVALID_PREPARATION")
    freeze = read_json(root / "preparation_freeze.json")
    if (
        freeze.get("schema_version") != "wave60-frozen-policy-transport-v1"
        or freeze.get("bundles_present") is not True
        or freeze.get("fit_operations") is not False
        or freeze.get("oracle_materialized") is not False
        or freeze.get("authorized_labels_present") is not False
    ):
        raise RuntimeError("INVALID_PREPARATION")
    if freeze.get("config_sha256") != file_sha256(root / "config.snapshot.json"):
        raise RuntimeError("INVALID_PREPARATION")
    bundle_hashes = freeze.get("prepared_bundle_hashes")
    expected_bundles = {f"prepared/{name}" for name in PREPARED_BUNDLES}
    if not isinstance(bundle_hashes, dict) or set(bundle_hashes) != expected_bundles:
        raise RuntimeError("INVALID_PREPARATION")
    if any(
        file_sha256(root / relative) != expected
        for relative, expected in bundle_hashes.items()
    ):
        raise RuntimeError("INVALID_PREPARATION")
    manifest = read_json(root / "benchmark/manifest.json")
    benchmark_files = manifest.get("files")
    if not isinstance(benchmark_files, dict) or not benchmark_files:
        raise RuntimeError("INVALID_PREPARATION")
    for relative, record in benchmark_files.items():
        path = root / "benchmark" / relative
        if (
            not isinstance(record, dict)
            or not path.is_file()
            or path.is_symlink()
            or record.get("sha256") != file_sha256(path)
            or record.get("bytes") != path.stat().st_size
        ):
            raise RuntimeError("INVALID_PREPARATION")
    actual_benchmark = {
        str(path.relative_to(root / "benchmark"))
        for path in (root / "benchmark").rglob("*")
        if path.is_file()
    }
    if actual_benchmark != {"manifest.json", *benchmark_files}:
        raise RuntimeError("INVALID_PREPARATION")
    actual_prepared = {
        path.name for path in (root / "prepared").iterdir() if path.is_file()
    }
    if actual_prepared != set(PREPARED_BUNDLES):
        raise RuntimeError("INVALID_PREPARATION")
    expected_logits = {
        f"seed{seed}__{split}.npz"
        for seed in (17, 29, 43)
        for split in ("train", "val", "lockbox")
    }
    logits = root / "inference/logits"
    if (
        not logits.is_dir()
        or {path.name for path in logits.iterdir() if path.is_file()} != expected_logits
    ):
        raise RuntimeError("INVALID_PREPARATION")
    expected_root_files = {
        *required,
        *(f"benchmark/{relative}" for relative in benchmark_files),
        *(f"inference/logits/{name}" for name in expected_logits),
        "inference/access_receipt.json",
    }
    if role == "replay":
        expected_root_files.add("preparation_replay.json")
    actual_root_files = {
        str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()
    }
    if actual_root_files != expected_root_files:
        raise RuntimeError("INVALID_PREPARATION")
    attestation = read_json(root / "preparation_attestation.json")
    verify_wave60_attestation(attestation)
    payload = attestation.get("payload", {})
    expected_payload_keys = {
        "schema_version",
        "phase",
        "run_role",
        "git_commit",
        "records",
        "truth_accessed",
        "fit_operations",
    }
    if (
        attestation.get("schema_version") != "wave60-signed-preparation-authority-v1"
        or attestation.get("phase") != "prepare"
        or not isinstance(payload, dict)
        or set(payload) != expected_payload_keys
        or payload.get("run_role") != role
        or payload.get("truth_accessed") is not False
        or payload.get("fit_operations") is not False
    ):
        raise RuntimeError("INVALID_PREPARATION")
    records = payload.get("records")
    if not isinstance(records, dict) or set(records) != {
        "pre_generation_freeze.json",
        "generation_escrow.json",
        "generation_receipt.json",
        "preparation_freeze.json",
        "preparation_receipt.json",
        "config.snapshot.json",
        "source_bindings.json",
        "journals/prepare.json",
        "benchmark/manifest.json",
        "benchmark/protocol_config.json",
        "prepared/sealed_monitor_inference_bundle.npz",
        "prepared/sealed_monitor_truth_bundle.npz",
    }:
        raise RuntimeError("INVALID_PREPARATION")
    for relative, record in records.items():
        path = root / relative
        if record != {
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }:
            raise RuntimeError("INVALID_PREPARATION")
    return {
        "preparation_freeze_sha256": file_sha256(root / "preparation_freeze.json"),
        "benchmark_manifest_sha256": file_sha256(root / "benchmark/manifest.json"),
        "prepared_bundle_hashes": dict(bundle_hashes),
    }


def write_failure_journal(
    root: Path,
    phase: str,
    error: BaseException,
    *,
    input_sha256: str,
    truth_accessed: bool,
    duration_seconds: float,
    max_rss_bytes: int = 0,
) -> Path:
    schema = {
        "source_bind": "wave60-source-binding-v1",
        "score_apply": SCORE_APPLY_SCHEMA,
        "evaluate": EVALUATE_SCHEMA,
    }[phase]
    path = root / f"journals/{phase}.json"
    write_json(
        path,
        {
            "schema_version": schema,
            "phase": phase,
            "status": "FAILED",
            "input_sha256": input_sha256,
            "error_type": type(error).__name__,
            "error_message_sha256": hashlib.sha256(
                str(error).encode("utf-8")
            ).hexdigest(),
            "truth_accessed": bool(truth_accessed),
            "duration_seconds": float(duration_seconds),
            "max_rss_bytes": int(max_rss_bytes),
        },
        mode=0o444,
    )
    return path


def discard_unpromoted_phase(root: Path, phase_directory: str) -> None:
    """Remove only a Wave 60 phase output that may not survive a failed phase."""
    for candidate in (
        root / phase_directory,
        root / f"{phase_directory}.pending",
    ):
        if candidate.exists():
            if candidate.is_symlink() or not candidate.is_dir():
                raise RuntimeError(f"Wave 60 unsafe failed-phase path: {candidate}")
            shutil.rmtree(candidate)


def discard_unsealed_root_outputs(root: Path) -> None:
    for candidate in (root / "runtime.json", root / "artifact_manifest.json"):
        if candidate.exists():
            if candidate.is_symlink() or not candidate.is_file():
                raise RuntimeError(f"Wave 60 unsafe root-seal path: {candidate}")
            candidate.unlink()


def normalize_invalid_preparation(root: Path, error: BaseException) -> None:
    failed = root / "failed_preparation"
    if failed.exists():
        if failed.is_symlink() or not failed.is_dir():
            raise RuntimeError("Wave 60 invalid failed-preparation path")
    else:
        failed.mkdir(mode=0o700)
    for child in list(root.iterdir()):
        if child.name in {
            "config.snapshot.json",
            "source_bindings.json",
            "failed_preparation",
        }:
            continue
        destination = failed / child.name
        if destination.exists():
            raise RuntimeError("Wave 60 failed-preparation collision")
        os.replace(child, destination)
    error_path = failed / "preparation_error.json"
    if not error_path.exists():
        write_json(
            error_path,
            {
                "schema_version": "wave60-preparation-failure-v1",
                "error_type": type(error).__name__,
                "error_message_sha256": hashlib.sha256(
                    str(error).encode("utf-8")
                ).hexdigest(),
            },
            mode=0o600,
        )


def _regular_identity(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"Wave 60 identity input is not a regular file: {path}")
    resolved = path.resolve(strict=True)
    metadata = resolved.stat()
    return {
        "sha256": file_sha256(resolved),
        "device": metadata.st_dev,
        "inode": metadata.st_ino,
        "resolved_path_sha256": hashlib.sha256(str(resolved).encode()).hexdigest(),
    }


def opaque_draw_fingerprint(root: Path) -> dict[str, Any]:
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError("Wave 60 draw root is not a physical directory")
    root = root.resolve(strict=True)
    files: dict[str, dict[str, Any]] = {}
    fixed = (
        "generation_escrow.json",
        "benchmark/manifest.json",
        "benchmark/attestations/semantic_root.json",
        "prepared/gate_fit_bundle.npz",
        "prepared/gate_select_inference_bundle.npz",
        "prepared/gate_select_truth_bundle.npz",
        "prepared/sealed_monitor_inference_bundle.npz",
        "prepared/sealed_monitor_truth_bundle.npz",
    )
    for relative in fixed:
        path = root / relative
        if path.exists():
            files[relative] = _regular_identity(path)
    commitment_root = root / "benchmark/commitments"
    if commitment_root.is_dir():
        for path in sorted(commitment_root.rglob("*")):
            if path.is_file():
                relative = str(path.relative_to(root))
                files[relative] = _regular_identity(path)
    commitments: dict[str, Any] = {}
    manifest_path = root / "benchmark/manifest.json"
    if manifest_path.is_file():
        manifest = read_json(manifest_path)
        for field in (
            "generation_key_commitment",
            "identity_key_commitment",
            "semantic_commitment_key_commitment",
        ):
            if field in manifest:
                commitments[field] = manifest[field]
        if isinstance(manifest.get("files"), dict):
            commitments["benchmark_file_commitments"] = manifest["files"]
    public_freeze = root / "pre_generation_freeze.json"
    if public_freeze.is_file():
        freeze = read_json(public_freeze)
        if isinstance(freeze.get("key_commitments"), dict):
            commitments["escrow_key_commitments"] = freeze["key_commitments"]
        files["pre_generation_freeze.json"] = _regular_identity(public_freeze)
    return {
        "root_path_sha256": hashlib.sha256(str(root).encode()).hexdigest(),
        "files": files,
        "commitments": commitments,
    }


def validate_new_draw_pair(
    primary: Path,
    replay: Path,
    antecedents: Iterable[Path] = ANTECEDENT_DRAW_ROOTS,
) -> dict[str, Any]:
    if primary.resolve(strict=True) == replay.resolve(strict=True):
        raise RuntimeError("INVALID_NEW_DRAW_IDENTITY")
    current = {
        "primary": opaque_draw_fingerprint(primary),
        "replay": opaque_draw_fingerprint(replay),
    }
    required_files = {
        "generation_escrow.json",
        "pre_generation_freeze.json",
        "benchmark/manifest.json",
        "benchmark/attestations/semantic_root.json",
        "prepared/gate_fit_bundle.npz",
        "prepared/gate_select_inference_bundle.npz",
        "prepared/gate_select_truth_bundle.npz",
        "prepared/sealed_monitor_inference_bundle.npz",
        "prepared/sealed_monitor_truth_bundle.npz",
    }
    required_commitments = {
        "generation_key_commitment",
        "identity_key_commitment",
        "semantic_commitment_key_commitment",
        "benchmark_file_commitments",
        "escrow_key_commitments",
    }
    if any(
        not required_files.issubset(payload["files"])
        or set(payload["commitments"]) != required_commitments
        for payload in current.values()
    ):
        raise RuntimeError("INVALID_NEW_DRAW_IDENTITY")
    if current["primary"]["commitments"] != current["replay"]["commitments"]:
        raise RuntimeError("INVALID_NEW_DRAW_IDENTITY")
    if set(current["primary"]["files"]) != set(current["replay"]["files"]):
        raise RuntimeError("INVALID_NEW_DRAW_IDENTITY")
    for relative, left in current["primary"]["files"].items():
        right = current["replay"]["files"][relative]
        if left["sha256"] != right["sha256"]:
            raise RuntimeError("INVALID_NEW_DRAW_IDENTITY")
        if (left["device"], left["inode"]) == (right["device"], right["inode"]):
            raise RuntimeError("INVALID_NEW_DRAW_IDENTITY")
    antecedent_rows: dict[str, Any] = {}
    for root in antecedents:
        if not root.exists():
            continue
        old = opaque_draw_fingerprint(root)
        collisions: list[str] = []
        for field, value in current["primary"]["commitments"].items():
            if field in old["commitments"] and value == old["commitments"][field]:
                collisions.append(f"commitment:{field}")
        for relative, record in current["primary"]["files"].items():
            if relative not in old["files"]:
                continue
            prior = old["files"][relative]
            if record["sha256"] == prior["sha256"]:
                collisions.append(f"bytes:{relative}")
            if (record["device"], record["inode"]) == (
                prior["device"],
                prior["inode"],
            ):
                collisions.append(f"inode:{relative}")
        if collisions:
            raise RuntimeError("INVALID_NEW_DRAW_IDENTITY")
        antecedent_rows[old["root_path_sha256"]] = {"collisions": []}
    return {
        "schema_version": "wave60-new-draw-identity-v1",
        "status": "PASS",
        "primary_root_path_sha256": current["primary"]["root_path_sha256"],
        "replay_root_path_sha256": current["replay"]["root_path_sha256"],
        "shared_bytes_without_hardlinks": sorted(current["primary"]["files"]),
        "antecedents": antecedent_rows,
    }


def seal_source_law_invalid(
    staging: Path,
    output: Path,
    request_path: Path,
    error: BaseException,
    *,
    private_key: Path,
    started: float,
) -> Path:
    """Publish the closed source-authority failure terminal atomically."""
    if staging.exists():
        if staging.is_symlink() or not staging.is_dir():
            raise RuntimeError("Wave 60 unsafe source-law staging path")
        shutil.rmtree(staging)
    staging.mkdir(mode=0o700)
    copy_regular(request_path.resolve(strict=True), staging / "source_law_request.json")
    request_sha256 = file_sha256(staging / "source_law_request.json")
    journal = {
        "schema_version": SOURCE_LAW_SCHEMA,
        "phase": "verify_source_law",
        "status": "FAILED",
        "input_sha256": request_sha256,
        "error_type": type(error).__name__,
        "error_message_sha256": hashlib.sha256(str(error).encode("utf-8")).hexdigest(),
        "truth_accessed": False,
        "duration_seconds": float(time.monotonic() - started),
        "max_rss_bytes": 0,
    }
    write_json(staging / "journals/verify_source_law.json", journal, mode=0o444)
    failure = {
        "schema_version": "wave60-root-failure-v1",
        "status": "FAILED",
        "terminal": "SOURCE_LAW_INVALID",
        "phase": "verify_source_law",
        "run_role": "source",
        "truth_accessed": False,
        "recovery_allowed": True,
        "error_type": type(error).__name__,
        "error_message_sha256": journal["error_message_sha256"],
        "authority_binding_sha256": request_sha256,
        "git_commit": git_commit(),
        "peer_terminal": None,
        "peer_terminal_binding_sha256": None,
        "created_at": now(),
    }
    write_json(staging / "FAILURE.json", failure, mode=0o444)
    current = inventory(staging)
    failure_inventory = {
        "schema_version": "wave60-root-failure-inventory-v1",
        "terminal": "SOURCE_LAW_INVALID",
        "last_complete_phase": "REQUEST_FROZEN",
        "files": {
            **{relative: record["sha256"] for relative, record in current.items()},
            "failure_inventory.json": "SELF_REFERENCE",
            "failure_attestation.json": "FUTURE_ATTESTATION",
        },
        "classes": {
            "source_law_request.json": "SOURCE_LAW_FROZEN",
            "journals/verify_source_law.json": "OPERATIONAL_JOURNAL",
            "FAILURE.json": "FAILURE_CONDITIONAL",
            "failure_inventory.json": "SELF_REFERENCE",
            "failure_attestation.json": "FAILURE_CONDITIONAL",
        },
        "missing_expected": [],
        "forbidden_present": [],
        "created_at": now(),
    }
    write_json(staging / "failure_inventory.json", failure_inventory, mode=0o444)
    attestation = make_attestation(
        "verify_source_law",
        {
            "scope": "pre-draw-source-law",
            "terminal": "SOURCE_LAW_INVALID",
            "failure_sha256": file_sha256(staging / "FAILURE.json"),
            "failure_inventory_sha256": file_sha256(staging / "failure_inventory.json"),
        },
        private_key,
    )
    attestation["schema_version"] = "wave60-root-failure-attestation-v1"
    write_json(staging / "failure_attestation.json", attestation, mode=0o444)
    verify_wave60_attestation(attestation)
    os.replace(staging, output)
    fsync_directory(output.parent)
    return output


def publish_source_law_authority(
    request_path: Path,
    output: Path = SOURCE_AUTHORITY_DEFAULT,
    *,
    private_key: Path = DEFAULT_PRIVATE_KEY,
    source_aliases: Mapping[str, Path] = SOURCE_ALIASES,
) -> Path:
    if output.exists():
        raise FileExistsError(output)
    parent = output.parent
    parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(output.name + ".initializing")
    if staging.exists():
        raise FileExistsError(staging)
    staging.mkdir(mode=0o700)
    started = time.monotonic()
    try:
        request = read_json(request_path.resolve(strict=True))
        if request.get("output_path") != str(output.relative_to(REPO_ROOT)):
            raise RuntimeError("Wave 60 source-law request output path drifted")
        if set(source_aliases) != set(PHASE_FILES["verify_source_law"]) - {
            "source_law_request.json"
        }:
            raise RuntimeError("Wave 60 source-law alias roster drifted")
        with tempfile.TemporaryDirectory(
            prefix="wave60-source-worker-", dir=parent
        ) as raw:
            workspace = Path(raw)
            stage = workspace / "stage"
            stage.mkdir(mode=0o755)
            copy_regular(request_path, stage / "source_law_request.json")
            for alias, source in source_aliases.items():
                copy_regular(source.resolve(strict=True), stage / alias)
            worker_output, receipt, duration, peak = run_worker(
                workspace,
                stage,
                "verify_source_law",
                [output, ATTEMPT_DEFAULT],
                max_seconds=float(request["runtime_budget"]["max_seconds"]),
                max_rss=int(request["runtime_budget"]["max_rss_bytes"]),
            )
            for name in (*SOURCE_SCIENTIFIC, "verify_source_law_receipt.json"):
                copy_regular(worker_output / name, staging / name, mode=0o444)
        copy_regular(request_path, staging / "source_law_request.json", mode=0o444)
        journal = {
            "schema_version": SOURCE_LAW_SCHEMA,
            "phase": "verify_source_law",
            "status": "SOURCE_LAW_VERIFIED",
            "input_sha256": file_sha256(staging / "source_law_request.json"),
            "duration_seconds": duration,
            "max_rss_bytes": peak,
            "truth_accessed": False,
        }
        write_json(staging / "journals/verify_source_law.json", journal, mode=0o444)
        payload = {
            "scope": "pre-draw-source-law",
            "request_sha256": file_sha256(staging / "source_law_request.json"),
            "implementation_commit": request["implementation_commit"],
            "freeze_sha256": file_sha256(staging / "source_law_freeze.json"),
            "receipt_sha256": file_sha256(staging / "verify_source_law_receipt.json"),
            "journal_sha256": file_sha256(staging / "journals/verify_source_law.json"),
        }
        attestation = make_attestation("verify_source_law", payload, private_key)
        write_json(staging / "source_law_attestation.json", attestation, mode=0o444)
        verify_wave60_attestation(attestation)
        files = inventory(staging)
        manifest = {
            "schema_version": SOURCE_LAW_SCHEMA,
            "terminal": "SOURCE_LAW_VERIFIED",
            "files": files,
            "classes": {
                relative: (
                    "OPERATIONAL_JOURNAL"
                    if relative.startswith("journals/")
                    or relative
                    in {
                        "verify_source_law_receipt.json",
                        "source_law_attestation.json",
                    }
                    else "SOURCE_LAW_FROZEN"
                )
                for relative in files
            },
            "self_reference": {
                "path": "source_authority_manifest.json",
                "hashes_omitted": True,
            },
        }
        write_json(staging / "source_authority_manifest.json", manifest, mode=0o444)
        os.replace(staging, output)
        fsync_directory(parent)
        return output
    except BaseException as error:
        return seal_source_law_invalid(
            staging,
            output,
            request_path,
            error,
            private_key=private_key,
            started=started,
        )


def initialize_attempt_container(
    attempt: Path, config_path: Path, source_bindings: Mapping[str, Any]
) -> Path:
    if attempt.exists():
        raise FileExistsError(attempt)
    staging = attempt.with_name(attempt.name + ".initializing")
    if staging.exists():
        raise FileExistsError(staging)
    staging.mkdir(parents=True, mode=0o700)
    try:
        for role in ("primary", "replay"):
            root = staging / role
            root.mkdir(mode=0o700)
            copy_regular(config_path, root / "config.snapshot.json", mode=0o644)
            write_json(root / "source_bindings.json", dict(source_bindings), mode=0o644)
        os.replace(staging, attempt)
        fsync_directory(attempt.parent)
    except BaseException:
        raise
    if not all((attempt / role).is_dir() for role in ("primary", "replay")):
        raise RuntimeError("Wave 60 atomic attempt initialization failed")
    return attempt


def validate_source_authority(
    authority: Path,
    authority_binding: Mapping[str, Any],
    config: Mapping[str, Any],
) -> None:
    authority = authority.resolve(strict=True)
    manifest_path = authority / "source_authority_manifest.json"
    if (
        file_sha256(manifest_path)
        != authority_binding["source_authority_manifest_sha256"]
    ):
        raise RuntimeError("SOURCE_BINDING_FAILED_PRE_TRUTH")
    manifest = read_json(manifest_path)
    require_exact_keys(
        manifest,
        {"schema_version", "terminal", "files", "classes", "self_reference"},
        "source authority manifest",
    )
    if (
        manifest["schema_version"] != SOURCE_LAW_SCHEMA
        or manifest["terminal"] != "SOURCE_LAW_VERIFIED"
        or manifest["self_reference"]
        != {"path": "source_authority_manifest.json", "hashes_omitted": True}
    ):
        raise RuntimeError("SOURCE_BINDING_FAILED_PRE_TRUTH")
    actual = inventory(authority)
    actual.pop("source_authority_manifest.json")
    expected_files = {
        "source_law_request.json",
        *SOURCE_SCIENTIFIC,
        "verify_source_law_receipt.json",
        "source_law_attestation.json",
        "journals/verify_source_law.json",
    }
    if (
        set(actual) != expected_files
        or manifest["files"] != actual
        or set(manifest["classes"]) != expected_files
    ):
        raise RuntimeError("SOURCE_BINDING_FAILED_PRE_TRUTH")
    freeze = read_json(authority / "source_law_freeze.json")
    require_exact_keys(
        freeze,
        {
            "schema_version",
            "phase",
            "source_law_request_sha256",
            "source_commit",
            "implementation_commit",
            "implementation_audit_sha256",
            "source_hashes",
            "roster",
            "feature_schema_sha256",
            "transport_law_manifest_sha256",
            "transport_law_arrays_sha256",
            "frozen_policy_spec_sha256",
        },
        "source law freeze",
    )
    request_sha256 = file_sha256(authority / "source_law_request.json")
    if (
        freeze["schema_version"] != SOURCE_LAW_SCHEMA
        or freeze["phase"] != "verify_source_law"
        or freeze["source_law_request_sha256"] != request_sha256
        or freeze["implementation_commit"] != config["implementation_binding"]["commit"]
        or freeze["implementation_audit_sha256"]
        != config["implementation_binding"]["audit_sha256"]
        or freeze["source_hashes"] != dict(SOURCE_HASHES)
    ):
        raise RuntimeError("SOURCE_BINDING_FAILED_PRE_TRUTH")
    frozen_outputs = {
        "source_law_freeze_sha256": file_sha256(authority / "source_law_freeze.json"),
        "transport_law_manifest_sha256": file_sha256(
            authority / "transport_law_manifest.json"
        ),
        "transport_law_arrays_sha256": file_sha256(
            authority / "transport_law_arrays.npz"
        ),
        "frozen_policy_spec_sha256": file_sha256(authority / "frozen_policy_spec.json"),
        "feature_schema_sha256": file_sha256(authority / "feature_schema.json"),
        "source_law_attestation_sha256": file_sha256(
            authority / "source_law_attestation.json"
        ),
    }
    if any(
        authority_binding[name] != observed for name, observed in frozen_outputs.items()
    ):
        raise RuntimeError("SOURCE_BINDING_FAILED_PRE_TRUTH")
    if (
        freeze["feature_schema_sha256"] != frozen_outputs["feature_schema_sha256"]
        or freeze["transport_law_manifest_sha256"]
        != frozen_outputs["transport_law_manifest_sha256"]
        or freeze["transport_law_arrays_sha256"]
        != frozen_outputs["transport_law_arrays_sha256"]
        or freeze["frozen_policy_spec_sha256"]
        != frozen_outputs["frozen_policy_spec_sha256"]
    ):
        raise RuntimeError("SOURCE_BINDING_FAILED_PRE_TRUTH")
    attestation = read_json(authority / "source_law_attestation.json")
    verify_wave60_attestation(attestation)
    payload = attestation.get("payload", {})
    if (
        attestation.get("schema_version") != SOURCE_LAW_SCHEMA
        or attestation.get("phase") != "verify_source_law"
        or not isinstance(payload, dict)
        or set(payload)
        != {
            "scope",
            "request_sha256",
            "implementation_commit",
            "freeze_sha256",
            "receipt_sha256",
            "journal_sha256",
        }
        or payload.get("scope") != "pre-draw-source-law"
        or payload.get("request_sha256") != request_sha256
        or payload.get("implementation_commit") != freeze["implementation_commit"]
        or payload.get("freeze_sha256") != frozen_outputs["source_law_freeze_sha256"]
        or payload.get("receipt_sha256")
        != file_sha256(authority / "verify_source_law_receipt.json")
        or payload.get("journal_sha256")
        != file_sha256(authority / "journals/verify_source_law.json")
    ):
        raise RuntimeError("SOURCE_BINDING_FAILED_PRE_TRUTH")


def bind_source_law(
    root: Path,
    role: str,
    config: Mapping[str, Any],
    authority: Path = SOURCE_AUTHORITY_DEFAULT,
) -> dict[str, Any]:
    if role not in {"primary", "replay"}:
        raise ValueError(role)
    authority_binding = config["source_law_authority"]
    validate_source_authority(authority, authority_binding, config)
    binding_dir = root / "source_law"
    if binding_dir.exists():
        raise FileExistsError(binding_dir)
    binding_dir.mkdir(mode=0o755)
    copied: dict[str, str] = {}
    hardlinks: dict[str, bool] = {}
    for name in SOURCE_COPY_FILES:
        source = authority / name
        copy_regular(source, binding_dir / name, mode=0o444)
        copied[name] = file_sha256(binding_dir / name)
        hardlinks[name] = (source.stat().st_dev, source.stat().st_ino) == (
            (binding_dir / name).stat().st_dev,
            (binding_dir / name).stat().st_ino,
        )
    expected = {
        "source_law_freeze.json": authority_binding["source_law_freeze_sha256"],
        "source_law_attestation.json": authority_binding[
            "source_law_attestation_sha256"
        ],
        "transport_law_manifest.json": authority_binding[
            "transport_law_manifest_sha256"
        ],
        "transport_law_arrays.npz": authority_binding["transport_law_arrays_sha256"],
        "frozen_policy_spec.json": authority_binding["frozen_policy_spec_sha256"],
        "feature_schema.json": authority_binding["feature_schema_sha256"],
    }
    if any(copied[name] != digest for name, digest in expected.items()) or any(
        hardlinks.values()
    ):
        raise RuntimeError("SOURCE_BINDING_FAILED_PRE_TRUTH")
    binding = {
        "schema_version": "wave60-source-binding-v1",
        "run_role": role,
        "config_sha256": file_sha256(root / "config.snapshot.json"),
        "source_authority_path_sha256": hashlib.sha256(
            str(authority.resolve()).encode()
        ).hexdigest(),
        "source_law_freeze_sha256": copied["source_law_freeze.json"],
        "source_law_attestation_sha256": copied["source_law_attestation.json"],
        "copied_output_hashes": copied,
        "hardlink_checks": hardlinks,
    }
    write_json(binding_dir / "source_law_binding.json", binding, mode=0o444)
    write_json(
        root / "journals/source_bind.json",
        {
            "schema_version": SOURCE_LAW_SCHEMA,
            "phase": "source_bind",
            "status": "SOURCE_LAW_BOUND",
            "input_sha256": file_sha256(root / "config.snapshot.json"),
            "truth_accessed": False,
        },
        mode=0o444,
    )
    return binding


def _stage_root_score(root: Path, stage: Path, role: str) -> None:
    sources = {
        "config.snapshot.json": root / "config.snapshot.json",
        "source_bindings.json": root / "source_law/source_law_binding.json",
        "transport_law_manifest.json": root / "source_law/transport_law_manifest.json",
        "transport_law_arrays.npz": root / "source_law/transport_law_arrays.npz",
        "frozen_policy_spec.json": root / "source_law/frozen_policy_spec.json",
        "feature_schema.json": root / "source_law/feature_schema.json",
        "sealed_monitor_inference_bundle.npz": root
        / "prepared/sealed_monitor_inference_bundle.npz",
    }
    for name, source in sources.items():
        copy_regular(source, stage / name)
    stage_phase_request(stage, "score_apply", role)


def _stage_root_evaluate(root: Path, stage: Path, role: str) -> None:
    sources = {
        "config.snapshot.json": root / "config.snapshot.json",
        "source_bindings.json": root / "source_law/source_law_binding.json",
        "evaluation_index.npz": root / "score/evaluation_index.npz",
        "monitor_policy_arrays.npz": root / "score/monitor_policy_arrays.npz",
        "monitor_action_freeze.json": root / "score/monitor_action_freeze.json",
        "sealed_monitor_truth_bundle.npz": root
        / "prepared/sealed_monitor_truth_bundle.npz",
    }
    for name, source in sources.items():
        copy_regular(source, stage / name)
    upstream = read_json(root / "source_bindings.json")
    if file_sha256(POLICY_MANIFEST) != upstream["wave52_policy_manifest_sha256"]:
        raise RuntimeError("Wave 60 policy manifest binding drifted")
    payload = read_json(POLICY_MANIFEST)
    levels = np.asarray(payload["levels"], dtype=np.float64)
    permutations = np.asarray(payload["rank_permutations"], dtype=np.int64)
    if permutations.shape != (24, 4) or set(map(tuple, permutations)) != set(
        itertools.permutations(range(4))
    ):
        raise RuntimeError("Wave 60 utility matrix source drifted")
    utilities = levels[permutations]
    if utilities.shape != (24, 4) or not np.all(np.isfinite(utilities)):
        raise RuntimeError("Wave 60 utility matrix is invalid")
    np.save(stage / "utilities.npy", utilities)
    stage_phase_request(stage, "evaluate", role)


def _promote_worker_phase(
    root: Path,
    role: str,
    phase: str,
    worker_output: Path,
    receipt: Mapping[str, Any],
    duration: float,
    peak: int,
    private_key: Path,
) -> Path:
    destination = root / ("score" if phase == "score_apply" else "evaluation")
    if destination.exists():
        raise FileExistsError(destination)
    staging = destination.with_name(destination.name + ".pending")
    staging.mkdir(mode=0o700)
    scientific = (
        (
            "monitor_scores.npz",
            "monitor_policy_arrays.npz",
            "evaluation_index.npz",
            "monitor_action_freeze.json",
        )
        if phase == "score_apply"
        else (
            "bootstrap_indices.npz",
            "analysis_arrays.npz",
            "analysis.json",
            "evaluation_freeze.json",
        )
    )
    receipt_name = (
        "score_apply_receipt.json"
        if phase == "score_apply"
        else "evaluate_receipt.json"
    )
    for name in (*scientific, receipt_name):
        copy_regular(worker_output / name, staging / name, mode=0o444)
    journal_path = root / f"journals/{phase}.json"
    if journal_path.exists():
        raise FileExistsError(journal_path)
    temporary_journal = staging / f".{phase}.journal.json"
    freeze_name = (
        "monitor_action_freeze.json"
        if phase == "score_apply"
        else "evaluation_freeze.json"
    )
    status = (
        "LOCKBOX_ACTIONS_FROZEN" if phase == "score_apply" else "EVALUATED_IMMUTABLE"
    )
    journal = {
        "schema_version": (
            SCORE_APPLY_SCHEMA if phase == "score_apply" else EVALUATE_SCHEMA
        ),
        "phase": phase,
        "status": status,
        "input_sha256": hashlib.sha256(
            json.dumps(receipt["inputs"], sort_keys=True).encode()
        ).hexdigest(),
        "truth_accessed": phase == "evaluate",
        "duration_seconds": duration,
        "max_rss_bytes": peak,
    }
    write_json(temporary_journal, journal, mode=0o444)
    payload = {
        "scope": role,
        "config_sha256": file_sha256(root / "config.snapshot.json"),
        "commit": git_commit(),
        "freeze_sha256": file_sha256(staging / freeze_name),
        "receipt_sha256": file_sha256(staging / receipt_name),
        "journal_sha256": file_sha256(temporary_journal),
    }
    attestation_name = (
        "score_apply_attestation.json"
        if phase == "score_apply"
        else "evaluation_attestation.json"
    )
    attestation = make_attestation(phase, payload, private_key)
    write_json(staging / attestation_name, attestation, mode=0o444)
    verify_wave60_attestation(attestation)
    copy_regular(temporary_journal, journal_path, mode=0o444)
    temporary_journal.unlink()
    os.replace(staging, destination)
    fsync_directory(root)
    return destination


def score_root(
    root: Path,
    role: str,
    *,
    private_key: Path = DEFAULT_PRIVATE_KEY,
    forbidden_probes: Iterable[Path] = (),
    max_seconds: float = 900.0,
    max_rss: int = 1610612736,
) -> Path:
    with tempfile.TemporaryDirectory(prefix=f"wave60-{role}-score-") as raw:
        workspace = Path(raw)
        stage = workspace / "stage"
        stage.mkdir()
        _stage_root_score(root, stage, role)
        output, receipt, duration, peak = run_worker(
            workspace,
            stage,
            "score_apply",
            forbidden_probes,
            max_seconds=max_seconds,
            max_rss=max_rss,
        )
        return _promote_worker_phase(
            root, role, "score_apply", output, receipt, duration, peak, private_key
        )


def evaluate_root(
    root: Path,
    role: str,
    *,
    private_key: Path = DEFAULT_PRIVATE_KEY,
    forbidden_probes: Iterable[Path] = (),
    max_seconds: float = 900.0,
    max_rss: int = 1610612736,
) -> Path:
    with tempfile.TemporaryDirectory(prefix=f"wave60-{role}-evaluate-") as raw:
        workspace = Path(raw)
        stage = workspace / "stage"
        stage.mkdir()
        _stage_root_evaluate(root, stage, role)
        output, receipt, duration, peak = run_worker(
            workspace,
            stage,
            "evaluate",
            forbidden_probes,
            max_seconds=max_seconds,
            max_rss=max_rss,
        )
        return _promote_worker_phase(
            root, role, "evaluate", output, receipt, duration, peak, private_key
        )


def seal_evaluated_root(root: Path, role: str) -> str:
    if (root / "FAILURE.json").exists():
        raise RuntimeError("Wave 60 evaluated root already failed")
    required = [
        *(root / "score" / name for name in SCORE_FILES),
        *(root / "evaluation" / name for name in EVALUATION_FILES),
        root / "journals/score_apply.json",
        root / "journals/evaluate.json",
    ]
    if any(not path.is_file() for path in required):
        raise RuntimeError("Wave 60 evaluated root is incomplete")
    write_json(
        root / "runtime.json",
        {
            "schema_version": "wave60-evaluated-root-v1",
            "terminal": "EVALUATED_IMMUTABLE",
            "run_role": role,
            "cuda_visible_devices": "",
            "cpu_threads": 4,
        },
        mode=0o444,
    )
    files = inventory(root)
    manifest = {
        "schema_version": "wave60-evaluated-root-v1",
        "terminal": "EVALUATED_IMMUTABLE",
        "run_role": role,
        "files": files,
        "classes": {relative: root_artifact_class(relative) for relative in files},
        "self_reference": {"path": "artifact_manifest.json", "hashes_omitted": True},
    }
    write_json(root / "artifact_manifest.json", manifest, mode=0o444)
    return file_sha256(root / "artifact_manifest.json")


def validate_evaluated_root(root: Path, role: str) -> str:
    manifest_path = root / "artifact_manifest.json"
    manifest = read_json(manifest_path)
    require_exact_keys(
        manifest,
        {
            "schema_version",
            "terminal",
            "run_role",
            "files",
            "classes",
            "self_reference",
        },
        "evaluated root manifest",
    )
    if (
        manifest["schema_version"] != "wave60-evaluated-root-v1"
        or manifest["terminal"] != "EVALUATED_IMMUTABLE"
        or manifest["run_role"] != role
        or manifest["self_reference"]
        != {"path": "artifact_manifest.json", "hashes_omitted": True}
        or (root / "FAILURE.json").exists()
    ):
        raise RuntimeError("Wave 60 evaluated root identity drifted")
    actual = inventory(root)
    actual.pop("artifact_manifest.json")
    if manifest["files"] != actual or manifest["classes"] != {
        relative: root_artifact_class(relative) for relative in actual
    }:
        raise RuntimeError("Wave 60 evaluated root inventory drifted")
    phase_specs = {
        "score_apply": (
            root / "score/score_apply_attestation.json",
            root / "score/monitor_action_freeze.json",
            root / "score/score_apply_receipt.json",
            root / "journals/score_apply.json",
        ),
        "evaluate": (
            root / "evaluation/evaluation_attestation.json",
            root / "evaluation/evaluation_freeze.json",
            root / "evaluation/evaluate_receipt.json",
            root / "journals/evaluate.json",
        ),
    }
    for phase, (
        attestation_path,
        freeze_path,
        receipt_path,
        journal_path,
    ) in phase_specs.items():
        attestation = read_json(attestation_path)
        verify_wave60_attestation(attestation)
        payload = attestation.get("payload", {})
        if attestation.get("phase") != phase or payload != {
            "scope": role,
            "config_sha256": file_sha256(root / "config.snapshot.json"),
            "commit": git_commit(),
            "freeze_sha256": file_sha256(freeze_path),
            "receipt_sha256": file_sha256(receipt_path),
            "journal_sha256": file_sha256(journal_path),
        }:
            raise RuntimeError(f"Wave 60 {phase} attestation drifted")
    return file_sha256(manifest_path)


def array_exact(left: Path, right: Path) -> bool:
    with np.load(left, allow_pickle=False) as a, np.load(
        right, allow_pickle=False
    ) as b:
        if set(a.files) != set(b.files):
            return False
        return all(
            a[key].dtype == b[key].dtype
            and a[key].shape == b[key].shape
            and np.array_equal(a[key], b[key], equal_nan=a[key].dtype.kind in "fc")
            for key in a.files
        )


def compare_evaluated_roots(primary: Path, replay: Path) -> dict[str, Any]:
    scientific_json = (
        "score/monitor_action_freeze.json",
        "evaluation/analysis.json",
        "evaluation/evaluation_freeze.json",
    )
    scientific_npz = (
        "score/monitor_scores.npz",
        "score/monitor_policy_arrays.npz",
        "score/evaluation_index.npz",
        "evaluation/bootstrap_indices.npz",
        "evaluation/analysis_arrays.npz",
    )
    json_checks = {
        name: file_sha256(primary / name) == file_sha256(replay / name)
        for name in scientific_json
    }
    npz_checks = {
        name: array_exact(primary / name, replay / name) for name in scientific_npz
    }
    functional_checks = {
        "source_law_freeze.json": file_sha256(
            primary / "source_law/source_law_freeze.json"
        )
        == file_sha256(replay / "source_law/source_law_freeze.json"),
        "transport_law_manifest.json": file_sha256(
            primary / "source_law/transport_law_manifest.json"
        )
        == file_sha256(replay / "source_law/transport_law_manifest.json"),
        "transport_law_arrays.npz": array_exact(
            primary / "source_law/transport_law_arrays.npz",
            replay / "source_law/transport_law_arrays.npz",
        ),
        "frozen_policy_spec.json": file_sha256(
            primary / "source_law/frozen_policy_spec.json"
        )
        == file_sha256(replay / "source_law/frozen_policy_spec.json"),
        "feature_schema.json": file_sha256(primary / "source_law/feature_schema.json")
        == file_sha256(replay / "source_law/feature_schema.json"),
        "source_law_attestation.json": file_sha256(
            primary / "source_law/source_law_attestation.json"
        )
        == file_sha256(replay / "source_law/source_law_attestation.json"),
    }
    secret_relatives = [
        "generation_escrow.json",
        "prepared/gate_fit_bundle.npz",
        "prepared/gate_select_truth_bundle.npz",
        "prepared/sealed_monitor_truth_bundle.npz",
    ]
    primary_sealed = primary / "benchmark/sealed"
    replay_sealed = replay / "benchmark/sealed"
    if primary_sealed.is_dir() and replay_sealed.is_dir():
        primary_secret_files = sorted(
            str(path.relative_to(primary))
            for path in primary_sealed.rglob("*")
            if path.is_file()
        )
        replay_secret_files = sorted(
            str(path.relative_to(replay))
            for path in replay_sealed.rglob("*")
            if path.is_file()
        )
        if primary_secret_files != replay_secret_files:
            secret_checks: dict[str, bool] = {"secret_inventory": False}
        else:
            secret_relatives.extend(primary_secret_files)
            secret_checks = {
                relative: file_sha256(primary / relative)
                == file_sha256(replay / relative)
                for relative in secret_relatives
            }
    else:
        secret_checks = {"secret_inventory": False}
    operational_checks: dict[str, bool] = {}
    for relative, fields in {
        "preparation_freeze.json": (
            "key_commitments",
            "benchmark_manifest_sha256",
            "prepared_bundle_hashes",
            "physical_splits",
            "oracle_materialized",
            "fit_operations",
        ),
        "generation_receipt.json": (
            "key_commitments",
            "manifest_sha256",
            "sealed_population_counts",
            "sealed_pair_token_counts_total",
            "sealed_eligible_pair_token_counts",
            "oracle_materialized",
        ),
    }.items():
        left = read_json(primary / relative)
        right = read_json(replay / relative)
        operational_checks[relative] = all(
            left.get(field) == right.get(field) for field in fields
        )
    left_binding = read_json(primary / "source_law/source_law_binding.json")
    right_binding = read_json(replay / "source_law/source_law_binding.json")
    operational_checks["source_law_binding"] = all(
        left_binding.get(field) == right_binding.get(field)
        for field in (
            "source_law_freeze_sha256",
            "source_law_attestation_sha256",
            "copied_output_hashes",
            "hardlink_checks",
        )
    )
    for relative, fields in {
        "preparation_receipt.json": (
            "preparation_freeze_sha256",
            "generation_receipt_sha256",
            "next_state",
        ),
        "score/score_apply_receipt.json": (
            "schema_version",
            "phase",
            "status",
            "uid",
            "gid",
            "capabilities",
            "no_new_privs",
            "outputs",
        ),
        "evaluation/evaluate_receipt.json": (
            "schema_version",
            "phase",
            "status",
            "uid",
            "gid",
            "capabilities",
            "no_new_privs",
            "outputs",
        ),
        "journals/score_apply.json": (
            "schema_version",
            "phase",
            "status",
            "truth_accessed",
        ),
        "journals/evaluate.json": (
            "schema_version",
            "phase",
            "status",
            "truth_accessed",
        ),
    }.items():
        left = read_json(primary / relative)
        right = read_json(replay / relative)
        operational_checks[relative] = all(
            left.get(field) == right.get(field) for field in fields
        )
    for relative in (
        "score/score_apply_attestation.json",
        "evaluation/evaluation_attestation.json",
    ):
        left = read_json(primary / relative)["payload"]
        right = read_json(replay / relative)["payload"]
        operational_checks[relative] = all(
            left.get(field) == right.get(field)
            for field in ("config_sha256", "commit", "freeze_sha256")
        )
    all_checks = {
        **json_checks,
        **npz_checks,
        **{f"functional:{key}": value for key, value in functional_checks.items()},
        **{f"secret:{key}": value for key, value in secret_checks.items()},
        **{f"operational:{key}": value for key, value in operational_checks.items()},
    }
    mismatches = sorted([name for name, exact in all_checks.items() if not exact])
    return {
        "schema_version": "wave60-replay-finalize-v1",
        "status": "EXACT" if not mismatches else "MISMATCH",
        "primary_scientific_hashes": {
            name: file_sha256(primary / name)
            for name in (*scientific_json, *scientific_npz)
        },
        "replay_scientific_hashes": {
            name: file_sha256(replay / name)
            for name in (*scientific_json, *scientific_npz)
        },
        "exact_json_md": json_checks,
        "exact_npz": npz_checks,
        "functional_states": functional_checks,
        "secret_hashes": secret_checks,
        "operational_semantic": operational_checks,
        "mismatches": mismatches,
    }


def pair_status(
    primary_terminal: str,
    replay_terminal: str,
    primary_binding: str,
    replay_binding: str,
    *,
    any_truth_accessed: bool,
    pair_failed: bool = False,
) -> dict[str, Any]:
    if pair_failed:
        terminal = (
            "PAIR_ABORTED_POST_TRUTH"
            if any_truth_accessed
            else "PAIR_ABORTED_PRE_TRUTH"
        )
        recovery = not any_truth_accessed
    elif primary_terminal == replay_terminal == "EVALUATED_IMMUTABLE":
        terminal = "COMPLETE"
        recovery = False
    elif any_truth_accessed:
        terminal = "PAIR_ABORTED_POST_TRUTH"
        recovery = False
    else:
        terminal = "PAIR_ABORTED_PRE_TRUTH"
        recovery = True
    return {
        "schema_version": "wave60-pair-status-v1",
        "terminal": terminal,
        "primary_terminal": primary_terminal,
        "replay_terminal": replay_terminal,
        "primary_terminal_binding_sha256": primary_binding,
        "replay_terminal_binding_sha256": replay_binding,
        "any_truth_accessed": bool(any_truth_accessed),
        "recovery_allowed": recovery,
        "created_at": now(),
    }


ROOT_FAILURE_TERMINALS = {
    "INVALID_NEW_DRAW_IDENTITY",
    "INVALID_PREPARATION",
    "SOURCE_BINDING_FAILED_PRE_TRUTH",
    "SCORE_APPLY_FAILED_PRE_TRUTH",
    "PEER_ABORTED_PRE_TRUTH",
    "EVALUATION_FAILED_POST_TRUTH",
}


def seal_root_failure(
    root: Path,
    *,
    terminal: str,
    phase: str,
    role: str,
    truth_accessed: bool,
    error: BaseException,
    authority_binding_sha256: str,
    last_complete_phase: str,
    private_key: Path = DEFAULT_PRIVATE_KEY,
    peer_terminal: str | None = None,
    peer_terminal_binding_sha256: str | None = None,
) -> str:
    if terminal not in ROOT_FAILURE_TERMINALS or role not in {"primary", "replay"}:
        raise ValueError("invalid Wave 60 root failure terminal")
    if (root / "FAILURE.json").exists() or (root / "artifact_manifest.json").exists():
        raise RuntimeError("Wave 60 root is already terminal")
    if terminal == "PEER_ABORTED_PRE_TRUTH":
        if (
            phase != "peer_abort"
            or peer_terminal is None
            or peer_terminal_binding_sha256 is None
        ):
            raise RuntimeError("Wave 60 peer abort lacks its directional binding")
    elif peer_terminal is not None or peer_terminal_binding_sha256 is not None:
        raise RuntimeError("Wave 60 own failure may not bind a peer")
    expected_last = {
        "INVALID_PREPARATION": "INITIALIZED",
        "INVALID_NEW_DRAW_IDENTITY": "PREPARED",
        "SOURCE_BINDING_FAILED_PRE_TRUTH": "PREPARED",
        "SCORE_APPLY_FAILED_PRE_TRUTH": "SOURCE_LAW_BOUND",
        "EVALUATION_FAILED_POST_TRUTH": "LOCKBOX_ACTIONS_FROZEN",
    }
    if terminal != "PEER_ABORTED_PRE_TRUTH" and expected_last[terminal] != (
        last_complete_phase
    ):
        raise RuntimeError("Wave 60 failure last-complete phase drifted")
    if terminal == "PEER_ABORTED_PRE_TRUTH" and last_complete_phase not in {
        "INITIALIZED",
        "PREPARED",
        "SOURCE_LAW_BOUND",
        "LOCKBOX_ACTIONS_FROZEN",
    }:
        raise RuntimeError("Wave 60 peer-abort phase drifted")
    phase_rank = {
        "INITIALIZED": 0,
        "PREPARED": 1,
        "SOURCE_LAW_BOUND": 2,
        "LOCKBOX_ACTIONS_FROZEN": 3,
    }[last_complete_phase]
    forbidden = []
    if phase_rank < 1:
        forbidden.extend(
            path
            for path in (
                root / "benchmark",
                root / "prepared",
                root / "preparation_freeze.json",
            )
            if path.exists()
        )
    if phase_rank < 2 and (root / "source_law").exists():
        forbidden.append(root / "source_law")
    if phase_rank < 3 and (root / "score").exists():
        forbidden.append(root / "score")
    if (root / "evaluation").exists() or (root / "runtime.json").exists():
        forbidden.append(root / "evaluation")
    required_failure_journal = {
        "SOURCE_BINDING_FAILED_PRE_TRUTH": root / "journals/source_bind.json",
        "SCORE_APPLY_FAILED_PRE_TRUTH": root / "journals/score_apply.json",
        "EVALUATION_FAILED_POST_TRUTH": root / "journals/evaluate.json",
    }.get(terminal)
    if (
        forbidden
        or (
            required_failure_journal is not None
            and not required_failure_journal.is_file()
        )
        or (
            terminal == "INVALID_PREPARATION"
            and not (root / "failed_preparation").is_dir()
        )
    ):
        raise RuntimeError("Wave 60 failure presence matrix drifted")
    failure = {
        "schema_version": "wave60-root-failure-v1",
        "status": "FAILED",
        "terminal": terminal,
        "phase": phase,
        "run_role": role,
        "truth_accessed": bool(truth_accessed),
        "recovery_allowed": not truth_accessed,
        "error_type": type(error).__name__,
        "error_message_sha256": hashlib.sha256(str(error).encode("utf-8")).hexdigest(),
        "authority_binding_sha256": authority_binding_sha256,
        "git_commit": git_commit(),
        "peer_terminal": peer_terminal,
        "peer_terminal_binding_sha256": peer_terminal_binding_sha256,
        "created_at": now(),
    }
    write_json(root / "FAILURE.json", failure, mode=0o444)
    current = inventory(root)
    failure_inventory = {
        "schema_version": "wave60-root-failure-inventory-v1",
        "terminal": terminal,
        "last_complete_phase": last_complete_phase,
        "files": {
            **{relative: record["sha256"] for relative, record in current.items()},
            "failure_inventory.json": "SELF_REFERENCE",
            "failure_attestation.json": "FUTURE_ATTESTATION",
        },
        "classes": {
            **{relative: root_artifact_class(relative) for relative in current},
            "failure_inventory.json": "SELF_REFERENCE",
            "failure_attestation.json": "FAILURE_CONDITIONAL",
        },
        "missing_expected": [],
        "forbidden_present": [],
        "created_at": now(),
    }
    write_json(root / "failure_inventory.json", failure_inventory, mode=0o444)
    payload = {
        "scope": role,
        "terminal": terminal,
        "failure_sha256": file_sha256(root / "FAILURE.json"),
        "failure_inventory_sha256": file_sha256(root / "failure_inventory.json"),
    }
    attestation = make_attestation(
        "evaluate" if truth_accessed else "score_apply", payload, private_key
    )
    attestation["schema_version"] = "wave60-root-failure-attestation-v1"
    attestation["phase"] = phase
    verify_wave60_attestation(attestation)
    write_json(root / "failure_attestation.json", attestation, mode=0o444)
    return file_sha256(root / "failure_attestation.json")


def publish_pair_failure(
    attempt: Path,
    status: Mapping[str, Any],
    *,
    error: BaseException,
    private_key: Path = DEFAULT_PRIVATE_KEY,
) -> Path:
    if status["terminal"] not in {"PAIR_ABORTED_PRE_TRUTH", "PAIR_ABORTED_POST_TRUTH"}:
        raise ValueError("invalid Wave 60 pair failure terminal")
    target = attempt / "pair"
    if target.exists():
        raise FileExistsError(target)
    staging = attempt / "pair.initializing"
    if staging.exists():
        raise FileExistsError(staging)
    staging.mkdir(mode=0o700)
    try:
        write_json(staging / "pair_status.json", dict(status), mode=0o444)
        failure = {
            "schema_version": "wave60-pair-failure-v1",
            "status": "FAILED",
            "terminal": status["terminal"],
            "phase": "pair_finalize",
            "run_role": "pair",
            "truth_accessed": status["any_truth_accessed"],
            "recovery_allowed": status["recovery_allowed"],
            "error_type": type(error).__name__,
            "error_message_sha256": hashlib.sha256(
                str(error).encode("utf-8")
            ).hexdigest(),
            "authority_binding_sha256": file_sha256(staging / "pair_status.json"),
            "git_commit": git_commit(),
            "created_at": now(),
        }
        write_json(staging / "FAILURE.json", failure, mode=0o444)
        current = inventory(staging)
        failure_inventory = {
            "schema_version": "wave60-pair-failure-inventory-v1",
            "terminal": status["terminal"],
            "root_terminal_bindings": {
                "primary": status["primary_terminal_binding_sha256"],
                "replay": status["replay_terminal_binding_sha256"],
            },
            "files": {
                **{relative: row["sha256"] for relative, row in current.items()},
                "failure_inventory.json": "SELF_REFERENCE",
                "failure_attestation.json": "FUTURE_ATTESTATION",
                "artifact_manifest.json": "SELF_REFERENCE",
            },
            "classes": {
                **{relative: "FAILURE_CONDITIONAL" for relative in current},
                "failure_inventory.json": "SELF_REFERENCE",
                "failure_attestation.json": "FAILURE_CONDITIONAL",
                "artifact_manifest.json": "SELF_REFERENCE",
            },
            "missing_expected": [],
            "forbidden_present": [],
            "created_at": now(),
        }
        write_json(staging / "failure_inventory.json", failure_inventory, mode=0o444)
        payload = {
            "scope": "pair",
            "pair_status_sha256": file_sha256(staging / "pair_status.json"),
            "failure_sha256": file_sha256(staging / "FAILURE.json"),
            "failure_inventory_sha256": file_sha256(staging / "failure_inventory.json"),
        }
        attestation = make_attestation(
            "evaluate" if status["any_truth_accessed"] else "score_apply",
            payload,
            private_key,
        )
        attestation["schema_version"] = "wave60-pair-failure-attestation-v1"
        attestation["phase"] = "pair_finalize"
        verify_wave60_attestation(attestation)
        write_json(staging / "failure_attestation.json", attestation, mode=0o444)
        files = inventory(staging)
        manifest = {
            "schema_version": "wave60-pair-final-v1",
            "terminal": status["terminal"],
            "files": files,
            "classes": {relative: pair_artifact_class(relative) for relative in files},
            "self_reference": {
                "path": "artifact_manifest.json",
                "hashes_omitted": True,
            },
        }
        write_json(staging / "artifact_manifest.json", manifest, mode=0o444)
        os.replace(staging, target)
        fsync_directory(attempt)
        return target
    except BaseException:
        raise


def execute_prepared_pair(
    attempt: Path,
    config: Mapping[str, Any],
    *,
    private_key: Path = DEFAULT_PRIVATE_KEY,
    authority: Path = SOURCE_AUTHORITY_DEFAULT,
) -> Path:
    """Drive both prepared roots through the pre-truth barrier and terminals."""
    roots = {role: attempt / role for role in ("primary", "replay")}
    execution_started = time.monotonic()
    runtime_budget = config.get("runtime_budget", {})
    maximum_seconds = float(runtime_budget.get("max_seconds_total", 900.0))
    maximum_rss = int(runtime_budget.get("max_rss_bytes_per_process", 1610612736))

    def remaining_seconds() -> float:
        remaining = maximum_seconds - (time.monotonic() - execution_started)
        if remaining <= 0:
            raise RuntimeError("Wave 60 pair exceeded its combined CPU budget")
        return remaining

    root_bindings: dict[str, str] = {}
    completed_phase = {role: "PREPARED" for role in roots}
    preparation_errors: dict[str, BaseException] = {}
    for role, root in roots.items():
        try:
            validate_prepared_root(root, role, config)
        except BaseException as error:
            preparation_errors[role] = error
    if preparation_errors:
        for role, error in preparation_errors.items():
            root = roots[role]
            normalize_invalid_preparation(root, error)
            root_bindings[role] = seal_root_failure(
                root,
                terminal="INVALID_PREPARATION",
                phase="prepare",
                role=role,
                truth_accessed=False,
                error=error,
                authority_binding_sha256=file_sha256(root / "config.snapshot.json"),
                last_complete_phase="INITIALIZED",
                private_key=private_key,
            )
        for role in set(roots) - set(preparation_errors):
            failed_role = next(iter(preparation_errors))
            root = roots[role]
            root_bindings[role] = seal_root_failure(
                root,
                terminal="PEER_ABORTED_PRE_TRUTH",
                phase="peer_abort",
                role=role,
                truth_accessed=False,
                error=RuntimeError("peer preparation failed"),
                authority_binding_sha256=file_sha256(root / "config.snapshot.json"),
                last_complete_phase="PREPARED",
                private_key=private_key,
                peer_terminal="INVALID_PREPARATION",
                peer_terminal_binding_sha256=root_bindings[failed_role],
            )
        status = pair_status(
            (
                "INVALID_PREPARATION"
                if "primary" in preparation_errors
                else "PEER_ABORTED_PRE_TRUTH"
            ),
            (
                "INVALID_PREPARATION"
                if "replay" in preparation_errors
                else "PEER_ABORTED_PRE_TRUTH"
            ),
            root_bindings["primary"],
            root_bindings["replay"],
            any_truth_accessed=False,
        )
        first = preparation_errors[
            "primary" if "primary" in preparation_errors else "replay"
        ]
        return publish_pair_failure(
            attempt, status, error=first, private_key=private_key
        )
    try:
        validate_new_draw_pair(roots["primary"], roots["replay"])
    except BaseException as error:
        root_bindings["primary"] = seal_root_failure(
            roots["primary"],
            terminal="INVALID_NEW_DRAW_IDENTITY",
            phase="new_draw_identity",
            role="primary",
            truth_accessed=False,
            error=error,
            authority_binding_sha256=file_sha256(
                roots["primary"] / "config.snapshot.json"
            ),
            last_complete_phase="PREPARED",
            private_key=private_key,
        )
        root_bindings["replay"] = seal_root_failure(
            roots["replay"],
            terminal="PEER_ABORTED_PRE_TRUTH",
            phase="peer_abort",
            role="replay",
            truth_accessed=False,
            error=RuntimeError("peer draw identity failed"),
            authority_binding_sha256=file_sha256(
                roots["replay"] / "config.snapshot.json"
            ),
            last_complete_phase="PREPARED",
            private_key=private_key,
            peer_terminal="INVALID_NEW_DRAW_IDENTITY",
            peer_terminal_binding_sha256=root_bindings["primary"],
        )
        status = pair_status(
            "INVALID_NEW_DRAW_IDENTITY",
            "PEER_ABORTED_PRE_TRUTH",
            root_bindings["primary"],
            root_bindings["replay"],
            any_truth_accessed=False,
        )
        return publish_pair_failure(
            attempt, status, error=error, private_key=private_key
        )
    for role, root in roots.items():
        phase_started = time.monotonic()
        try:
            bind_source_law(root, role, config, authority)
            completed_phase[role] = "SOURCE_LAW_BOUND"
        except BaseException as error:
            discard_unpromoted_phase(root, "source_law")
            write_failure_journal(
                root,
                "source_bind",
                error,
                input_sha256=file_sha256(root / "config.snapshot.json"),
                truth_accessed=False,
                duration_seconds=time.monotonic() - phase_started,
            )
            root_bindings[role] = seal_root_failure(
                root,
                terminal="SOURCE_BINDING_FAILED_PRE_TRUTH",
                phase="source_bind",
                role=role,
                truth_accessed=False,
                error=error,
                authority_binding_sha256=file_sha256(root / "config.snapshot.json"),
                last_complete_phase=completed_phase[role],
                private_key=private_key,
            )
            peer = "replay" if role == "primary" else "primary"
            root_bindings[peer] = seal_root_failure(
                roots[peer],
                terminal="PEER_ABORTED_PRE_TRUTH",
                phase="peer_abort",
                role=peer,
                truth_accessed=False,
                error=RuntimeError("peer source binding failed"),
                authority_binding_sha256=file_sha256(
                    roots[peer] / "config.snapshot.json"
                ),
                last_complete_phase=completed_phase[peer],
                private_key=private_key,
                peer_terminal="SOURCE_BINDING_FAILED_PRE_TRUTH",
                peer_terminal_binding_sha256=root_bindings[role],
            )
            status = pair_status(
                (
                    "SOURCE_BINDING_FAILED_PRE_TRUTH"
                    if role == "primary"
                    else "PEER_ABORTED_PRE_TRUTH"
                ),
                (
                    "SOURCE_BINDING_FAILED_PRE_TRUTH"
                    if role == "replay"
                    else "PEER_ABORTED_PRE_TRUTH"
                ),
                root_bindings["primary"],
                root_bindings["replay"],
                any_truth_accessed=False,
            )
            return publish_pair_failure(
                attempt, status, error=error, private_key=private_key
            )
    for role, root in roots.items():
        phase_started = time.monotonic()
        try:
            score_root(
                root,
                role,
                private_key=private_key,
                forbidden_probes=[
                    root / "prepared/sealed_monitor_truth_bundle.npz",
                    root / "prepared/gate_fit_bundle.npz",
                    root / "prepared/gate_select_truth_bundle.npz",
                ],
                max_seconds=remaining_seconds(),
                max_rss=maximum_rss,
            )
            completed_phase[role] = "LOCKBOX_ACTIONS_FROZEN"
        except BaseException as error:
            discard_unpromoted_phase(root, "score")
            write_failure_journal(
                root,
                "score_apply",
                error,
                input_sha256=file_sha256(root / "config.snapshot.json"),
                truth_accessed=False,
                duration_seconds=time.monotonic() - phase_started,
            )
            root_bindings[role] = seal_root_failure(
                root,
                terminal="SCORE_APPLY_FAILED_PRE_TRUTH",
                phase="score_apply",
                role=role,
                truth_accessed=False,
                error=error,
                authority_binding_sha256=file_sha256(root / "config.snapshot.json"),
                last_complete_phase=completed_phase[role],
                private_key=private_key,
            )
            peer = "replay" if role == "primary" else "primary"
            root_bindings[peer] = seal_root_failure(
                roots[peer],
                terminal="PEER_ABORTED_PRE_TRUTH",
                phase="peer_abort",
                role=peer,
                truth_accessed=False,
                error=RuntimeError("peer failed before truth"),
                authority_binding_sha256=file_sha256(
                    roots[peer] / "config.snapshot.json"
                ),
                last_complete_phase=completed_phase[peer],
                private_key=private_key,
                peer_terminal="SCORE_APPLY_FAILED_PRE_TRUTH",
                peer_terminal_binding_sha256=root_bindings[role],
            )
            status = pair_status(
                (
                    "SCORE_APPLY_FAILED_PRE_TRUTH"
                    if role == "primary"
                    else "PEER_ABORTED_PRE_TRUTH"
                ),
                (
                    "SCORE_APPLY_FAILED_PRE_TRUTH"
                    if role == "replay"
                    else "PEER_ABORTED_PRE_TRUTH"
                ),
                root_bindings["primary"],
                root_bindings["replay"],
                any_truth_accessed=False,
            )
            return publish_pair_failure(
                attempt, status, error=error, private_key=private_key
            )

    evaluation_errors: dict[str, BaseException] = {}
    terminals: dict[str, str] = {}
    # Truth is now authorized for both.  Each root is driven to its own terminal
    # even if the other evaluation fails first.
    for role, root in roots.items():
        phase_started = time.monotonic()
        try:
            evaluate_root(
                root,
                role,
                private_key=private_key,
                forbidden_probes=[
                    root / "prepared/sealed_monitor_inference_bundle.npz",
                    root / "source_law/transport_law_arrays.npz",
                    root / "score/monitor_scores.npz",
                ],
                max_seconds=remaining_seconds(),
                max_rss=maximum_rss,
            )
            root_bindings[role] = seal_evaluated_root(root, role)
            terminals[role] = "EVALUATED_IMMUTABLE"
        except BaseException as error:
            evaluation_errors[role] = error
            discard_unpromoted_phase(root, "evaluation")
            discard_unsealed_root_outputs(root)
            write_failure_journal(
                root,
                "evaluate",
                error,
                input_sha256=file_sha256(root / "score/monitor_action_freeze.json"),
                truth_accessed=True,
                duration_seconds=time.monotonic() - phase_started,
            )
            root_bindings[role] = seal_root_failure(
                root,
                terminal="EVALUATION_FAILED_POST_TRUTH",
                phase="evaluate",
                role=role,
                truth_accessed=True,
                error=error,
                authority_binding_sha256=file_sha256(
                    root / "score/monitor_action_freeze.json"
                ),
                last_complete_phase="LOCKBOX_ACTIONS_FROZEN",
                private_key=private_key,
            )
            terminals[role] = "EVALUATION_FAILED_POST_TRUTH"
    if evaluation_errors:
        status = pair_status(
            terminals["primary"],
            terminals["replay"],
            root_bindings["primary"],
            root_bindings["replay"],
            any_truth_accessed=True,
            pair_failed=True,
        )
        first = (
            evaluation_errors["primary"]
            if "primary" in evaluation_errors
            else evaluation_errors["replay"]
        )
        return publish_pair_failure(
            attempt, status, error=first, private_key=private_key
        )
    try:
        remaining_seconds()
        return finalize_pair(attempt, private_key=private_key)
    except BaseException as error:
        staging = attempt / "pair.initializing"
        if staging.exists():
            if staging.is_symlink() or not staging.is_dir():
                raise
            shutil.rmtree(staging)
        status = pair_status(
            "EVALUATED_IMMUTABLE",
            "EVALUATED_IMMUTABLE",
            root_bindings["primary"],
            root_bindings["replay"],
            any_truth_accessed=True,
        )
        return publish_pair_failure(
            attempt, status, error=error, private_key=private_key
        )


def finalize_pair(
    attempt: Path,
    *,
    private_key: Path = DEFAULT_PRIVATE_KEY,
) -> Path:
    primary = attempt / "primary"
    replay = attempt / "replay"
    primary_binding = validate_evaluated_root(primary, "primary")
    replay_binding = validate_evaluated_root(replay, "replay")
    target = attempt / "pair"
    if target.exists():
        raise FileExistsError(target)
    staging = attempt / "pair.initializing"
    if staging.exists():
        if staging.is_symlink() or not staging.is_dir():
            raise RuntimeError("Wave 60 pair staging is not a physical directory")
    else:
        staging.mkdir(mode=0o700)
    comparison = compare_evaluated_roots(primary, replay)
    expected_status = pair_status(
        "EVALUATED_IMMUTABLE",
        "EVALUATED_IMMUTABLE",
        primary_binding,
        replay_binding,
        any_truth_accessed=True,
    )
    status_path = staging / "pair_status.json"
    if status_path.exists():
        status = read_json(status_path)
        require_exact_keys(status, expected_status, "pair status")
        if any(
            status[key] != value
            for key, value in expected_status.items()
            if key != "created_at"
        ):
            raise RuntimeError("Wave 60 staged pair status binds different roots")
    else:
        status = expected_status
        write_json(status_path, status, mode=0o444)
    ensure_json(staging / "replay_comparison.json", comparison, mode=0o444)
    primary_analysis = read_json(primary / "evaluation/analysis.json")
    replay_analysis = read_json(replay / "evaluation/analysis.json")
    replay_exact = comparison["status"] == "EXACT"
    conditions, patterns = finalize_patterns(primary_analysis, replay_exact)
    final_analysis = {
        "schema_version": "wave60-replay-finalize-v1",
        "primary_analysis_sha256": file_sha256(primary / "evaluation/analysis.json"),
        "replay_analysis_sha256": file_sha256(replay / "evaluation/analysis.json"),
        "replay_comparison_sha256": file_sha256(staging / "replay_comparison.json"),
        "conditions": conditions,
        "patterns": patterns,
        "scientific_decision": None,
        "decision_authority": "user",
        "limitations": primary_analysis["limitations"],
    }
    if primary_analysis != replay_analysis and replay_exact:
        raise AssertionError("Wave 60 replay normalization is internally inconsistent")
    ensure_json(staging / "final_analysis.json", final_analysis, mode=0o444)
    freeze = {
        "schema_version": "wave60-replay-finalize-v1",
        "phase": "replay_finalize",
        "primary_evaluation_attestation_sha256": file_sha256(
            primary / "evaluation/evaluation_attestation.json"
        ),
        "replay_evaluation_attestation_sha256": file_sha256(
            replay / "evaluation/evaluation_attestation.json"
        ),
        "primary_root_manifest_sha256": primary_binding,
        "replay_root_manifest_sha256": replay_binding,
        "replay_comparison_sha256": file_sha256(staging / "replay_comparison.json"),
        "final_analysis_sha256": file_sha256(staging / "final_analysis.json"),
        "pair_status_sha256": file_sha256(staging / "pair_status.json"),
    }
    ensure_json(staging / "replay_finalize_freeze.json", freeze, mode=0o444)
    journal = {
        "schema_version": "wave60-replay-finalize-v1",
        "phase": "replay_finalize",
        "status": "COMPLETE",
        "input_sha256": hashlib.sha256(
            (primary_binding + replay_binding).encode()
        ).hexdigest(),
        "truth_accessed": False,
    }
    ensure_json(staging / "journals/replay_finalize.json", journal, mode=0o444)
    receipt_static = {
        "schema_version": "wave60-replay-finalize-v1",
        "phase": "replay_finalize",
        "status": "COMPLETE",
        "coordinator_uid": os.geteuid(),
        "coordinator_gid": os.getegid(),
        "inputs": {"primary": primary_binding, "replay": replay_binding},
        "outputs_before_receipt": {
            name: file_sha256(staging / name)
            for name in (
                "pair_status.json",
                "replay_comparison.json",
                "final_analysis.json",
                "replay_finalize_freeze.json",
                "journals/replay_finalize.json",
            )
        },
        "staging_path_sha256": hashlib.sha256(
            str(staging.resolve()).encode()
        ).hexdigest(),
        "publish_target_path_sha256": hashlib.sha256(
            str(target.resolve()).encode()
        ).hexdigest(),
    }
    receipt_path = staging / "replay_finalize_receipt.json"
    if receipt_path.exists():
        receipt = read_json(receipt_path)
        require_exact_keys(
            receipt,
            {*receipt_static, "started_at", "completed_at"},
            "replay finalize receipt",
        )
        if any(receipt[key] != value for key, value in receipt_static.items()):
            raise RuntimeError("Wave 60 staged replay receipt drifted")
    else:
        receipt = {
            **receipt_static,
            "started_at": now(),
            "completed_at": now(),
        }
        write_json(receipt_path, receipt, mode=0o444)
    attestation_payload = {
        "scope": "pair",
        "freeze_sha256": file_sha256(staging / "replay_finalize_freeze.json"),
        "receipt_sha256": file_sha256(staging / "replay_finalize_receipt.json"),
        "journal_sha256": file_sha256(staging / "journals/replay_finalize.json"),
        "primary_evaluation_attestation_sha256": freeze[
            "primary_evaluation_attestation_sha256"
        ],
        "replay_evaluation_attestation_sha256": freeze[
            "replay_evaluation_attestation_sha256"
        ],
    }
    attestation_path = staging / "replay_finalize_attestation.json"
    if attestation_path.exists():
        attestation = read_json(attestation_path)
        if attestation.get("payload") != attestation_payload:
            raise RuntimeError("Wave 60 staged replay attestation drifted")
    else:
        attestation = make_attestation(
            "evaluate",
            attestation_payload,
            private_key,
        )
        attestation["schema_version"] = "wave60-replay-finalize-v1"
        attestation["phase"] = "replay_finalize"
        write_json(attestation_path, attestation, mode=0o444)
    verify_wave60_attestation(attestation)
    ensure_json(
        staging / "runtime.json",
        {
            "schema_version": "wave60-pair-final-v1",
            "terminal": "COMPLETE",
            "cuda_visible_devices": "",
            "cpu_threads": 4,
        },
        mode=0o444,
    )
    report = (
        "# Wave 60 — frozen policy transport\n\n"
        f"Pair terminal: `COMPLETE`. Replay exact: `{str(replay_exact).lower()}`.\n\n"
        "Scientific decision remains with the user. The result is conditional on the new "
        "synthetic draw and the frozen Wave 59 law.\n"
    )
    ensure_text(staging / "REPORT.md", report, mode=0o444)
    files = inventory(staging)
    files.pop("artifact_manifest.json", None)
    manifest = {
        "schema_version": "wave60-pair-final-v1",
        "terminal": "COMPLETE",
        "files": files,
        "classes": {relative: pair_artifact_class(relative) for relative in files},
        "self_reference": {"path": "artifact_manifest.json", "hashes_omitted": True},
    }
    ensure_json(staging / "artifact_manifest.json", manifest, mode=0o444)
    os.replace(staging, target)
    fsync_directory(attempt)
    return target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=(
            "verify-source-law",
            "initialize-attempt",
            "execute-prepared-pair",
            "finalize-pair",
        ),
    )
    parser.add_argument("--request", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--attempt", type=Path, default=ATTEMPT_DEFAULT)
    parser.add_argument(
        "--attestation-private-key", type=Path, default=DEFAULT_PRIVATE_KEY
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Wave 60 coordinator can see CUDA")
    if args.mode == "verify-source-law":
        if args.request is None:
            raise ValueError("--request is required")
        output = args.output or SOURCE_AUTHORITY_DEFAULT
        result = publish_source_law_authority(
            args.request, output, private_key=args.attestation_private_key
        )
        status = (
            read_json(result / "FAILURE.json")["terminal"]
            if (result / "FAILURE.json").is_file()
            else "SOURCE_LAW_VERIFIED"
        )
    elif args.mode == "initialize-attempt":
        if args.config is None:
            raise ValueError("--config is required")
        config = read_json(args.config.resolve(strict=True))
        validate_pre_draw_config(config)
        result = initialize_attempt_container(
            args.attempt, args.config, config["source_binding"]
        )
        status = "INITIALIZED"
    elif args.mode == "execute-prepared-pair":
        if args.config is None:
            raise ValueError("--config is required")
        config = read_json(args.config.resolve(strict=True))
        validate_pre_draw_config(config)
        result = execute_prepared_pair(
            args.attempt, config, private_key=args.attestation_private_key
        )
        status = read_json(result / "pair_status.json")["terminal"]
    else:
        result = finalize_pair(args.attempt, private_key=args.attestation_private_key)
        status = "COMPLETE"
    print(json.dumps({"status": status, "path": str(result)}, sort_keys=True))


if __name__ == "__main__":
    main()
