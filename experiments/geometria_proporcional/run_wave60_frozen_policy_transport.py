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
import re
import shutil
import signal
import stat
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


class IntegrityDriftError(RuntimeError):
    """A durable Wave 60 binding is inconsistent with its physical artifact."""


class PresenceMatrixError(IntegrityDriftError):
    """A root does not contain exactly the artifacts authorized by its state."""


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


def fsync_tree(root: Path) -> None:
    """Durably flush a closed staging tree before its atomic publication."""
    files = sorted(path for path in root.rglob("*") if path.is_file())
    directories = sorted(
        (path for path in root.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    )
    for path in files:
        with path.open("rb") as handle:
            os.fsync(handle.fileno())
    for path in directories:
        fsync_directory(path)
    fsync_directory(root)


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
            raise IntegrityDriftError(f"Wave 60 staged JSON drifted: {path.name}")
        return
    write_json(path, payload, mode=mode)


def ensure_text(path: Path, payload: str, *, mode: int = 0o444) -> None:
    if path.exists():
        if (
            path.is_symlink()
            or not path.is_file()
            or path.read_text(encoding="utf-8") != payload
        ):
            raise IntegrityDriftError(f"Wave 60 staged text drifted: {path.name}")
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


def validate_implementation_audit_authority(
    implementation_commit: str, audit_commit: str, audit_sha256: str
) -> Path:
    """Require the exclusive direct-child PASS report before source-law work."""
    parent = subprocess.run(
        ["git", "rev-parse", f"{audit_commit}^"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    if parent != implementation_commit:
        raise RuntimeError("Wave 60 implementation audit is not a direct child")
    changed = subprocess.run(
        [
            "git",
            "diff-tree",
            "--no-commit-id",
            "--name-only",
            "-r",
            audit_commit,
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.splitlines()
    if len(changed) != 1:
        raise RuntimeError("Wave 60 implementation audit commit is not exclusive")
    relative = Path(changed[0])
    if (
        relative.parent
        != Path("Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports")
        or relative.suffix != ".md"
    ):
        raise RuntimeError("Wave 60 implementation audit path drifted")
    path = REPO_ROOT / relative
    if file_sha256(path) != audit_sha256:
        raise RuntimeError("Wave 60 implementation audit hash drifted")
    committed = subprocess.run(
        ["git", "show", f"{audit_commit}:{relative.as_posix()}"],
        cwd=REPO_ROOT,
        capture_output=True,
        check=True,
    ).stdout
    if hashlib.sha256(committed).hexdigest() != audit_sha256:
        raise RuntimeError("Wave 60 audit file differs from its exclusive commit")
    blocks = re.findall(
        r"```json\s*\n(.*?)\n```", path.read_text(encoding="utf-8"), re.DOTALL
    )
    if len(blocks) != 1:
        raise RuntimeError("Wave 60 implementation audit authority is absent")
    authority = json.loads(blocks[0])
    if (
        authority
        != {
            "schema_version": "wave60-audit-authority-v1",
            "audit_id": authority.get("audit_id"),
            "scope": "IMPLEMENTATION",
            "target": {"implementation_commit": implementation_commit},
            "technical_verdict": "PASS",
            "findings": {"high": 0, "medium": 0, "low": 0},
            "files_modified": False,
            "gpu_used_or_queried": False,
        }
        or re.fullmatch(r"R[0-9]+", str(authority.get("audit_id"))) is None
    ):
        raise RuntimeError("Wave 60 implementation audit does not grant PASS")
    return path


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


def require_physical_directory(path: Path, label: str) -> None:
    try:
        metadata = path.lstat()
    except FileNotFoundError as exc:
        raise IntegrityDriftError(f"Wave 60 {label} is absent") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise IntegrityDriftError(
            f"Wave 60 {label} is not one physical directory"
        )


def inventory_metadata(root: Path) -> dict[str, dict[str, Any]]:
    """Inventory physical identity without reopening file contents."""
    result: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*")):
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise RuntimeError("Wave 60 package contains a symlink")
        if stat.S_ISREG(metadata.st_mode):
            result[str(path.relative_to(root))] = {
                "bytes": metadata.st_size,
                "owner": metadata.st_uid,
                "group": metadata.st_gid,
                "mode": f"{metadata.st_mode & 0o777:04o}",
            }
        elif not stat.S_ISDIR(metadata.st_mode):
            raise RuntimeError("Wave 60 package contains a special node")
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
        "recovery_amendment.json",
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


def _json_document_sha256(payload: Any) -> str:
    encoded = (
        json.dumps(
            payload, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False
        )
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def prepared_common_paths(root: Path, role: str) -> set[str]:
    """Derive the exact COMMON file set from its two closed manifests."""
    manifest = read_json(root / "benchmark/manifest.json")
    benchmark_files = manifest.get("files")
    if not isinstance(benchmark_files, dict) or not benchmark_files:
        raise PresenceMatrixError("Wave 60 benchmark manifest has no closed file map")
    freeze = read_json(root / "preparation_freeze.json")
    bundle_hashes = freeze.get("prepared_bundle_hashes")
    if not isinstance(bundle_hashes, dict) or set(bundle_hashes) != {
        f"prepared/{name}" for name in PREPARED_BUNDLES
    }:
        raise PresenceMatrixError("Wave 60 preparation bundle map drifted")
    logits = {
        f"inference/logits/seed{seed}__{split}.npz"
        for seed in (17, 29, 43)
        for split in ("train", "val", "lockbox")
    }
    paths = {
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
        "inference/access_receipt.json",
        *(f"benchmark/{relative}" for relative in benchmark_files),
        *bundle_hashes,
        *logits,
    }
    if role == "replay":
        paths.add("preparation_replay.json")
    config = read_json(root / "config.snapshot.json")
    if config.get("attempt", {}).get("recovery") is not None:
        paths.add("recovery_amendment.json")
    return paths


def preparation_budget_record(root: Path, role: str) -> dict[str, Any]:
    receipt = read_json(root / "preparation_receipt.json")
    budget = receipt.get("coordinator_budget")
    expected = {
        "duration_seconds",
        "cumulative_duration_seconds",
        "prior_elapsed_seconds",
        "max_rss_bytes",
        "max_seconds",
        "max_seconds_total",
        "max_rss_allowed_bytes",
        "cuda_visible_devices",
        "budget_enforced",
    }
    if not isinstance(budget, dict) or set(budget) != expected:
        raise IntegrityDriftError("Wave 60 preparation budget ledger drifted")
    duration = float(budget["duration_seconds"])
    prior = float(budget["prior_elapsed_seconds"])
    cumulative = float(budget["cumulative_duration_seconds"])
    if (
        duration < 0.0
        or prior < 0.0
        or cumulative != prior + duration
        or float(budget["max_seconds_total"]) != 900.0
        or float(budget["max_seconds"]) != 900.0 - prior
        or int(budget["max_rss_allowed_bytes"]) != 1610612736
        or int(budget["max_rss_bytes"]) < 0
        or int(budget["max_rss_bytes"]) > 1610612736
        or budget["cuda_visible_devices"] != ""
        or budget["budget_enforced"] is not True
        or (role == "primary" and prior != 0.0)
    ):
        raise IntegrityDriftError("Wave 60 preparation budget values drifted")
    return budget


def pair_preparation_elapsed(primary: Path, replay: Path) -> float:
    left = preparation_budget_record(primary, "primary")
    right = preparation_budget_record(replay, "replay")
    if float(right["prior_elapsed_seconds"]) != float(left["duration_seconds"]):
        raise IntegrityDriftError("Wave 60 replay budget does not continue primary")
    cumulative = float(right["cumulative_duration_seconds"])
    if cumulative >= 900.0:
        raise RuntimeError("Wave 60 combined preparation budget is exhausted")
    return cumulative


def pair_durable_elapsed(primary: Path, replay: Path) -> dict[str, float]:
    """Reconstruct signed preparation and worker time after process loss."""
    preparation = pair_preparation_elapsed(primary, replay)
    worker = 0.0
    for root in (primary, replay):
        for phase in ("score_apply", "evaluate"):
            journal = read_json(root / f"journals/{phase}.json")
            duration = float(journal.get("duration_seconds", -1.0))
            if duration < 0.0:
                raise IntegrityDriftError("Wave 60 durable phase duration drifted")
            worker += duration
    coordinator = 0.0
    for root in (primary, replay):
        runtime_path = root / "runtime.json"
        if runtime_path.is_file():
            runtime = read_json(runtime_path)
            coordinator = max(
                coordinator,
                float(runtime.get("coordinator_elapsed_seconds", 0.0)),
            )
    if coordinator < 0.0:
        raise IntegrityDriftError("Wave 60 coordinator duration drifted")
    total = preparation + max(worker, coordinator)
    if total >= 900.0:
        raise IntegrityDriftError("Wave 60 durable CPU budget is exhausted")
    return {
        "preparation_seconds": preparation,
        "worker_seconds": worker,
        "coordinator_seconds": coordinator,
        "durable_seconds": total,
    }


def source_phase_paths() -> set[str]:
    return {
        *(f"source_law/{name}" for name in SOURCE_COPY_FILES),
        "source_law/source_law_binding.json",
        "journals/source_bind.json",
    }


def score_phase_paths() -> set[str]:
    return {*(f"score/{name}" for name in SCORE_FILES), "journals/score_apply.json"}


def evaluation_phase_paths() -> set[str]:
    return {
        *(f"evaluation/{name}" for name in EVALUATION_FILES),
        "journals/evaluate.json",
    }


def validate_prepared_root(
    root: Path,
    role: str,
    expected_config: Mapping[str, Any],
    *,
    allow_later_phases: bool = False,
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
    preparation_budget_record(root, role)
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
    expected_root_files = prepared_common_paths(root, role)
    actual_root_files = {
        str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()
    }
    if (not allow_later_phases and actual_root_files != expected_root_files) or (
        allow_later_phases and not expected_root_files.issubset(actual_root_files)
    ):
        raise RuntimeError("INVALID_PREPARATION")
    attestation = read_json(root / "preparation_attestation.json")
    verify_wave60_attestation(attestation)
    payload = attestation.get("payload", {})
    expected_payload_keys = {
        "schema_version",
        "phase",
        "run_role",
        "execution_mode",
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
        or payload.get("execution_mode")
        not in (
            {"primary", "recovery"} if role == "primary" else {"replay"}
        )
        or payload.get("truth_accessed") is not False
        or payload.get("fit_operations") is not False
    ):
        raise RuntimeError("INVALID_PREPARATION")
    records = payload.get("records")
    expected_attested_records = {
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
    }
    if expected_config.get("attempt", {}).get("recovery") is not None:
        expected_attested_records.add("recovery_amendment.json")
    if not isinstance(records, dict) or set(records) != expected_attested_records:
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
        declared = manifest.get("files")
        if isinstance(declared, dict) and declared:
            commitments["benchmark_file_commitments"] = declared
            for relative, record in declared.items():
                candidate = Path(relative)
                if (
                    not isinstance(relative, str)
                    or candidate.is_absolute()
                    or candidate.as_posix() != relative
                    or ".." in candidate.parts
                    or not isinstance(record, dict)
                ):
                    raise RuntimeError("INVALID_NEW_DRAW_IDENTITY")
                path = root / "benchmark" / candidate
                identity = _regular_identity(path)
                if identity["sha256"] != record.get("sha256") or (
                    "bytes" in record and path.stat().st_size != record["bytes"]
                ):
                    raise RuntimeError("INVALID_NEW_DRAW_IDENTITY")
                files[f"benchmark/{relative}"] = identity
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
        validate_implementation_audit_authority(
            request["implementation_commit"],
            request["implementation_audit_commit"],
            request["implementation_audit_sha256"],
        )
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


def _phase_request_inputs(
    root: Path,
    role: str,
    phase: str,
    *,
    sealed_hashes: Mapping[str, str] | None = None,
) -> dict[str, str]:
    if phase == "score_apply":
        physical = {
            "config.snapshot.json": root / "config.snapshot.json",
            "source_bindings.json": root / "source_law/source_law_binding.json",
            "transport_law_manifest.json": root
            / "source_law/transport_law_manifest.json",
            "transport_law_arrays.npz": root / "source_law/transport_law_arrays.npz",
            "frozen_policy_spec.json": root / "source_law/frozen_policy_spec.json",
            "feature_schema.json": root / "source_law/feature_schema.json",
            "sealed_monitor_inference_bundle.npz": root
            / "prepared/sealed_monitor_inference_bundle.npz",
        }
    elif phase == "evaluate":
        physical = {
            "config.snapshot.json": root / "config.snapshot.json",
            "source_bindings.json": root / "source_law/source_law_binding.json",
            "evaluation_index.npz": root / "score/evaluation_index.npz",
            "monitor_policy_arrays.npz": root / "score/monitor_policy_arrays.npz",
            "monitor_action_freeze.json": root / "score/monitor_action_freeze.json",
            "sealed_monitor_truth_bundle.npz": root
            / "prepared/sealed_monitor_truth_bundle.npz",
        }
    else:
        raise ValueError(phase)
    inputs = {}
    for name, path in physical.items():
        relative = str(path.relative_to(root))
        if sealed_hashes is not None and relative in sealed_hashes:
            inputs[name] = sealed_hashes[relative]
        else:
            inputs[name] = file_sha256(path)
    if phase == "evaluate":
        evaluation_freeze = read_json(root / "evaluation/evaluation_freeze.json")
        inputs["utilities.npy"] = evaluation_freeze["utilities_sha256"]
    request = {
        "schema_version": SOURCE_LAW_SCHEMA,
        "phase": phase,
        "allowed_files": sorted(PHASE_FILES[phase]),
        "sha256": dict(sorted(inputs.items())),
        "run_role": role,
    }
    inputs["phase_request.json"] = _json_document_sha256(request)
    return dict(sorted(inputs.items()))


def _validate_success_journal(
    path: Path,
    *,
    phase: str,
    schema: str,
    status: str,
    truth_accessed: bool,
    inputs: Mapping[str, str],
) -> None:
    journal = read_json(path)
    require_exact_keys(
        journal,
        {
            "schema_version",
            "phase",
            "status",
            "input_sha256",
            "truth_accessed",
            "duration_seconds",
            "max_rss_bytes",
        },
        f"{phase} journal",
    )
    expected_input = hashlib.sha256(
        json.dumps(dict(inputs), sort_keys=True).encode()
    ).hexdigest()
    if (
        journal["schema_version"] != schema
        or journal["phase"] != phase
        or journal["status"] != status
        or journal["truth_accessed"] is not truth_accessed
        or journal["input_sha256"] != expected_input
        or float(journal["duration_seconds"]) < 0.0
        or int(journal["max_rss_bytes"]) < 0
    ):
        raise IntegrityDriftError(f"Wave 60 {phase} journal drifted")


def _validate_worker_phase(
    root: Path,
    role: str,
    phase: str,
    *,
    sealed_hashes: Mapping[str, str] | None = None,
) -> None:
    directory = root / ("score" if phase == "score_apply" else "evaluation")
    if phase == "score_apply":
        freeze_name = "monitor_action_freeze.json"
        receipt_name = "score_apply_receipt.json"
        attestation_name = "score_apply_attestation.json"
        schema = SCORE_APPLY_SCHEMA
        status = "LOCKBOX_ACTIONS_FROZEN"
        output_names = {
            "monitor_scores.npz",
            "monitor_policy_arrays.npz",
            "evaluation_index.npz",
            freeze_name,
        }
        freeze = read_json(directory / freeze_name)
        require_exact_keys(
            freeze,
            {
                "schema_version",
                "phase",
                "source_law_freeze_sha256",
                "inference_bundle_sha256",
                "scores_sha256",
                "policy_arrays_sha256",
                "evaluation_index_sha256",
            },
            "score freeze",
        )
        expected_freeze = {
            "schema_version": schema,
            "phase": phase,
            "source_law_freeze_sha256": file_sha256(
                root / "source_law/source_law_freeze.json"
            ),
            "inference_bundle_sha256": file_sha256(
                root / "prepared/sealed_monitor_inference_bundle.npz"
            ),
            "scores_sha256": file_sha256(directory / "monitor_scores.npz"),
            "policy_arrays_sha256": file_sha256(
                directory / "monitor_policy_arrays.npz"
            ),
            "evaluation_index_sha256": file_sha256(directory / "evaluation_index.npz"),
        }
    else:
        freeze_name = "evaluation_freeze.json"
        receipt_name = "evaluate_receipt.json"
        attestation_name = "evaluation_attestation.json"
        schema = EVALUATE_SCHEMA
        status = "EVALUATED_IMMUTABLE"
        output_names = {
            "bootstrap_indices.npz",
            "analysis_arrays.npz",
            "analysis.json",
            freeze_name,
        }
        freeze = read_json(directory / freeze_name)
        require_exact_keys(
            freeze,
            {
                "schema_version",
                "phase",
                "truth_bundle_sha256",
                "action_freeze_sha256",
                "policy_arrays_sha256",
                "evaluation_index_sha256",
                "utilities_sha256",
                "bootstrap_sha256",
                "analysis_arrays_sha256",
                "analysis_sha256",
            },
            "evaluation freeze",
        )
        truth_relative = "prepared/sealed_monitor_truth_bundle.npz"
        truth_sha256 = (
            sealed_hashes[truth_relative]
            if sealed_hashes is not None
            else file_sha256(root / truth_relative)
        )
        expected_freeze = {
            "schema_version": schema,
            "phase": phase,
            "truth_bundle_sha256": truth_sha256,
            "action_freeze_sha256": file_sha256(
                root / "score/monitor_action_freeze.json"
            ),
            "policy_arrays_sha256": file_sha256(
                root / "score/monitor_policy_arrays.npz"
            ),
            "evaluation_index_sha256": file_sha256(root / "score/evaluation_index.npz"),
            "utilities_sha256": freeze["utilities_sha256"],
            "bootstrap_sha256": file_sha256(directory / "bootstrap_indices.npz"),
            "analysis_arrays_sha256": file_sha256(directory / "analysis_arrays.npz"),
            "analysis_sha256": file_sha256(directory / "analysis.json"),
        }
    if freeze != expected_freeze:
        raise IntegrityDriftError(f"Wave 60 {phase} freeze/output chain drifted")
    inputs = _phase_request_inputs(
        root, role, phase, sealed_hashes=sealed_hashes
    )
    receipt_path = directory / receipt_name
    receipt = read_json(receipt_path)
    _validate_receipt(receipt, phase)
    if (
        receipt["schema_version"] != schema
        or receipt["status"] != status
        or receipt["inputs"] != inputs
        or receipt["outputs"]
        != {name: file_sha256(directory / name) for name in sorted(output_names)}
        or len(receipt["opened_paths"]) != len(PHASE_FILES[phase])
    ):
        raise IntegrityDriftError(f"Wave 60 {phase} receipt drifted")
    journal_path = root / f"journals/{phase}.json"
    _validate_success_journal(
        journal_path,
        phase=phase,
        schema=schema,
        status=status,
        truth_accessed=phase == "evaluate",
        inputs=inputs,
    )
    attestation = read_json(directory / attestation_name)
    verify_wave60_attestation(attestation)
    expected_payload = {
        "scope": role,
        "config_sha256": file_sha256(root / "config.snapshot.json"),
        "commit": git_commit(),
        "freeze_sha256": file_sha256(directory / freeze_name),
        "receipt_sha256": file_sha256(receipt_path),
        "journal_sha256": file_sha256(journal_path),
    }
    if attestation["phase"] != phase or attestation["payload"] != expected_payload:
        raise IntegrityDriftError(f"Wave 60 {phase} attestation drifted")


def _validate_source_phase(
    root: Path,
    role: str,
    *,
    committed_hashes: Mapping[str, str] | None = None,
) -> None:
    binding = read_json(root / "source_law/source_law_binding.json")
    require_exact_keys(
        binding,
        {
            "schema_version",
            "run_role",
            "config_sha256",
            "source_authority_path_sha256",
            "source_law_freeze_sha256",
            "source_law_attestation_sha256",
            "copied_output_hashes",
            "hardlink_checks",
        },
        "source binding",
    )
    copied = {}
    for name in SOURCE_COPY_FILES:
        relative = f"source_law/{name}"
        copied[name] = (
            committed_hashes[relative]
            if committed_hashes is not None and relative in committed_hashes
            else file_sha256(root / relative)
        )
    if (
        binding["schema_version"] != "wave60-source-binding-v1"
        or binding["run_role"] != role
        or binding["config_sha256"] != file_sha256(root / "config.snapshot.json")
        or binding["copied_output_hashes"] != copied
        or set(binding["hardlink_checks"]) != set(SOURCE_COPY_FILES)
        or any(binding["hardlink_checks"].values())
        or binding["source_law_freeze_sha256"] != copied["source_law_freeze.json"]
        or binding["source_law_attestation_sha256"]
        != copied["source_law_attestation.json"]
    ):
        raise IntegrityDriftError("Wave 60 source binding drifted")
    freeze = read_json(root / "source_law/source_law_freeze.json")
    for field, name in {
        "feature_schema_sha256": "feature_schema.json",
        "transport_law_manifest_sha256": "transport_law_manifest.json",
        "transport_law_arrays_sha256": "transport_law_arrays.npz",
        "frozen_policy_spec_sha256": "frozen_policy_spec.json",
    }.items():
        if freeze.get(field) != copied[name]:
            raise IntegrityDriftError("Wave 60 source freeze/output chain drifted")
    source_attestation = read_json(root / "source_law/source_law_attestation.json")
    verify_wave60_attestation(source_attestation)
    if (
        source_attestation.get("payload", {}).get("freeze_sha256")
        != copied["source_law_freeze.json"]
    ):
        raise IntegrityDriftError("Wave 60 source attestation drifted")
    journal = read_json(root / "journals/source_bind.json")
    if journal != {
        "schema_version": SOURCE_LAW_SCHEMA,
        "phase": "source_bind",
        "status": "SOURCE_LAW_BOUND",
        "input_sha256": file_sha256(root / "config.snapshot.json"),
        "truth_accessed": False,
    }:
        raise IntegrityDriftError("Wave 60 source journal drifted")


def validate_completed_root_phases(root: Path, role: str) -> set[str]:
    config = read_json(root / "config.snapshot.json")
    validate_prepared_root(root, role, config, allow_later_phases=True)
    _validate_source_phase(root, role)
    _validate_worker_phase(root, role, "score_apply")
    _validate_worker_phase(root, role, "evaluate")
    return (
        prepared_common_paths(root, role)
        | source_phase_paths()
        | score_phase_paths()
        | evaluation_phase_paths()
    )


def seal_evaluated_root(
    root: Path, role: str, *, coordinator_elapsed_seconds: float = 0.0
) -> str:
    if (root / "FAILURE.json").exists():
        raise RuntimeError("Wave 60 evaluated root already failed")
    expected = validate_completed_root_phases(root, role)
    actual = set(inventory(root))
    if actual != expected:
        raise PresenceMatrixError(
            f"Wave 60 evaluated root presence drifted: "
            f"missing={sorted(expected - actual)} extra={sorted(actual - expected)}"
        )
    write_json(
        root / "runtime.json",
        {
            "schema_version": "wave60-evaluated-root-v1",
            "terminal": "EVALUATED_IMMUTABLE",
            "run_role": role,
            "cuda_visible_devices": "",
            "cpu_threads": 4,
            "coordinator_elapsed_seconds": float(coordinator_elapsed_seconds),
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


def _validate_sealed_preparation_chain(
    root: Path,
    role: str,
    manifest_files: Mapping[str, Mapping[str, Any]],
) -> None:
    """Validate signed preparation commitments without reopening secret bytes."""
    config = read_json(root / "config.snapshot.json")
    validate_pre_draw_config(config)
    preparation_budget_record(root, role)
    attestation = read_json(root / "preparation_attestation.json")
    verify_wave60_attestation(attestation)
    records = attestation.get("payload", {}).get("records")
    expected_records = {
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
    }
    if config.get("attempt", {}).get("recovery") is not None:
        expected_records.add("recovery_amendment.json")
    payload = attestation["payload"]
    if (
        attestation["schema_version"]
        != "wave60-signed-preparation-authority-v1"
        or attestation["phase"] != "prepare"
        or payload.get("schema_version")
        != "wave60-signed-preparation-authority-v1"
        or payload.get("phase")
        != "wave60-preparation-finalized-before-source-binding"
        or payload.get("run_role") != role
        or payload.get("truth_accessed") is not False
        or payload.get("fit_operations") is not False
        or not isinstance(records, dict)
        or set(records) != expected_records
    ):
        raise IntegrityDriftError("Wave 60 preparation attestation drifted")
    secret_records = {
        "generation_escrow.json",
        "prepared/sealed_monitor_truth_bundle.npz",
    }
    for relative, record in records.items():
        manifest_record = manifest_files.get(relative)
        if record != {
            "path": relative,
            "bytes": manifest_record.get("bytes") if manifest_record else None,
            "sha256": manifest_record.get("sha256") if manifest_record else None,
        }:
            raise IntegrityDriftError(
                f"Wave 60 signed preparation record drifted: {relative}"
            )
        if relative not in secret_records and file_sha256(root / relative) != record[
            "sha256"
        ]:
            raise IntegrityDriftError(
                f"Wave 60 public preparation record drifted: {relative}"
            )
    freeze = read_json(root / "preparation_freeze.json")
    if payload.get("git_commit") != freeze.get("git_commit"):
        raise IntegrityDriftError("Wave 60 preparation commit drifted")
    bundle_hashes = freeze.get("prepared_bundle_hashes")
    if not isinstance(bundle_hashes, dict) or set(bundle_hashes) != {
        f"prepared/{name}" for name in PREPARED_BUNDLES
    }:
        raise IntegrityDriftError("Wave 60 signed prepared bundle map drifted")
    for relative, expected in bundle_hashes.items():
        if manifest_files.get(relative, {}).get("sha256") != expected:
            raise IntegrityDriftError(
                f"Wave 60 prepared commitment drifted: {relative}"
            )
    benchmark_manifest = read_json(root / "benchmark/manifest.json")
    benchmark_files = benchmark_manifest.get("files")
    if not isinstance(benchmark_files, dict):
        raise IntegrityDriftError("Wave 60 benchmark commitment map drifted")
    for relative, record in benchmark_files.items():
        root_record = manifest_files.get(f"benchmark/{relative}")
        if (
            not isinstance(record, dict)
            or root_record is None
            or record.get("bytes") != root_record.get("bytes")
            or record.get("sha256") != root_record.get("sha256")
        ):
            raise IntegrityDriftError(
                f"Wave 60 benchmark commitment drifted: {relative}"
            )


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
    expected = (
        prepared_common_paths(root, role)
        | source_phase_paths()
        | score_phase_paths()
        | evaluation_phase_paths()
        | {"runtime.json"}
    )
    actual = inventory_metadata(root)
    actual.pop("artifact_manifest.json")
    if (
        set(actual) != expected
        or not isinstance(manifest["files"], dict)
        or set(manifest["files"]) != expected
        or manifest["classes"]
        != {relative: root_artifact_class(relative) for relative in actual}
    ):
        raise RuntimeError("Wave 60 evaluated root inventory drifted")
    for relative, metadata in actual.items():
        recorded = manifest["files"].get(relative)
        if not isinstance(recorded, dict) or {
            key: recorded.get(key) for key in ("bytes", "owner", "group", "mode")
        } != metadata:
            raise RuntimeError(
                f"Wave 60 evaluated root metadata drifted: {relative}"
            )
        if root_artifact_class(relative) not in {
            "BENCHMARK_SEALED_SECRET",
            "PREPARED_TRUTH_SECRET",
        } and relative != "source_law/transport_law_arrays.npz" and file_sha256(
            root / relative
        ) != recorded.get("sha256"):
            raise IntegrityDriftError(
                f"Wave 60 evaluated public artifact drifted: {relative}"
            )
    runtime = read_json(root / "runtime.json")
    if runtime != {
        "schema_version": "wave60-evaluated-root-v1",
        "terminal": "EVALUATED_IMMUTABLE",
        "run_role": role,
        "cuda_visible_devices": "",
        "cpu_threads": 4,
        "coordinator_elapsed_seconds": runtime.get(
            "coordinator_elapsed_seconds"
        ),
    }:
        raise RuntimeError("Wave 60 evaluated runtime drifted")
    if float(runtime["coordinator_elapsed_seconds"]) < 0.0:
        raise RuntimeError("Wave 60 evaluated runtime duration drifted")
    _validate_sealed_preparation_chain(root, role, manifest["files"])
    committed_hashes = {
        relative: record["sha256"]
        for relative, record in manifest["files"].items()
        if root_artifact_class(relative)
        in {"BENCHMARK_SEALED_SECRET", "PREPARED_TRUTH_SECRET"}
        or relative == "source_law/transport_law_arrays.npz"
    }
    _validate_source_phase(root, role, committed_hashes=committed_hashes)
    _validate_worker_phase(
        root, role, "score_apply", sealed_hashes=committed_hashes
    )
    _validate_worker_phase(
        root, role, "evaluate", sealed_hashes=committed_hashes
    )
    binding = file_sha256(manifest_path)
    pair_root = root.parent / "pair"
    pair_status_path = pair_root / "pair_status.json"
    if pair_status_path.is_file():
        pair_state = read_json(pair_status_path)
        if pair_state.get(f"{role}_terminal_binding_sha256") != binding:
            raise IntegrityDriftError(
                "Wave 60 root manifest differs from its pair-status binding"
            )
    pair_freeze_path = pair_root / "replay_finalize_freeze.json"
    if pair_freeze_path.is_file():
        pair_freeze = read_json(pair_freeze_path)
        field = f"{role}_root_manifest_sha256"
        if pair_freeze.get(field) != binding:
            raise IntegrityDriftError(
                "Wave 60 root manifest differs from its external pair binding"
            )
    return binding


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
    primary_manifest = read_json(primary / "artifact_manifest.json")["files"]
    replay_manifest = read_json(replay / "artifact_manifest.json")["files"]
    functional_checks = {
        "source_law_freeze.json": file_sha256(
            primary / "source_law/source_law_freeze.json"
        )
        == file_sha256(replay / "source_law/source_law_freeze.json"),
        "transport_law_manifest.json": file_sha256(
            primary / "source_law/transport_law_manifest.json"
        )
        == file_sha256(replay / "source_law/transport_law_manifest.json"),
        "transport_law_arrays.npz": (
            primary_manifest["source_law/transport_law_arrays.npz"]["sha256"]
            == replay_manifest["source_law/transport_law_arrays.npz"]["sha256"]
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
    primary_secret_hashes = {
        relative: record["sha256"]
        for relative, record in primary_manifest.items()
        if root_artifact_class(relative)
        in {"BENCHMARK_SEALED_SECRET", "PREPARED_TRUTH_SECRET"}
    }
    replay_secret_hashes = {
        relative: record["sha256"]
        for relative, record in replay_manifest.items()
        if root_artifact_class(relative)
        in {"BENCHMARK_SEALED_SECRET", "PREPARED_TRUTH_SECRET"}
    }
    secret_names = sorted(set(primary_secret_hashes) | set(replay_secret_hashes))
    secret_checks = {
        relative: primary_secret_hashes.get(relative)
        == replay_secret_hashes.get(relative)
        for relative in secret_names
    }
    secret_checks["secret_inventory"] = set(primary_secret_hashes) == set(
        replay_secret_hashes
    )
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

ROOT_FAILURE_SEMANTICS = {
    "INVALID_PREPARATION": {
        "phase": "prepare",
        "last_complete_phase": "INITIALIZED",
        "truth_accessed": False,
        "recovery_allowed": True,
    },
    "INVALID_NEW_DRAW_IDENTITY": {
        "phase": "new_draw_identity",
        "last_complete_phase": "PREPARED",
        "truth_accessed": False,
        "recovery_allowed": True,
    },
    "SOURCE_BINDING_FAILED_PRE_TRUTH": {
        "phase": "source_bind",
        "last_complete_phase": "PREPARED",
        "truth_accessed": False,
        "recovery_allowed": True,
    },
    "SCORE_APPLY_FAILED_PRE_TRUTH": {
        "phase": "score_apply",
        "last_complete_phase": "SOURCE_LAW_BOUND",
        "truth_accessed": False,
        "recovery_allowed": True,
    },
    "EVALUATION_FAILED_POST_TRUTH": {
        "phase": "evaluate",
        "last_complete_phase": "LOCKBOX_ACTIONS_FROZEN",
        "truth_accessed": True,
        "recovery_allowed": False,
    },
}


def _expected_failure_prefix(
    root: Path, role: str, terminal: str, last_complete_phase: str
) -> tuple[set[str], list[str], list[str]]:
    initialized = {"config.snapshot.json", "source_bindings.json"}
    actual = set(inventory(root)) - {
        "FAILURE.json",
        "failure_inventory.json",
        "failure_attestation.json",
    }
    if terminal == "INVALID_PREPARATION":
        failure_files = {
            relative
            for relative in actual
            if relative.startswith("failed_preparation/")
        }
        expected = initialized | failure_files
        if "failed_preparation/preparation_error.json" not in failure_files:
            expected.add("failed_preparation/preparation_error.json")
    else:
        config = read_json(root / "config.snapshot.json")
        if last_complete_phase == "INITIALIZED":
            expected = initialized
        else:
            validate_prepared_root(root, role, config, allow_later_phases=True)
            expected = prepared_common_paths(root, role)
        if last_complete_phase in {"SOURCE_LAW_BOUND", "LOCKBOX_ACTIONS_FROZEN"}:
            _validate_source_phase(root, role)
            expected |= source_phase_paths()
        if last_complete_phase == "LOCKBOX_ACTIONS_FROZEN":
            _validate_worker_phase(root, role, "score_apply")
            expected |= score_phase_paths()
        failure_journal = {
            "SOURCE_BINDING_FAILED_PRE_TRUTH": "journals/source_bind.json",
            "SCORE_APPLY_FAILED_PRE_TRUTH": "journals/score_apply.json",
            "EVALUATION_FAILED_POST_TRUTH": "journals/evaluate.json",
        }.get(terminal)
        if failure_journal is not None:
            expected.add(failure_journal)
            journal = read_json(root / failure_journal)
            require_exact_keys(
                journal,
                {
                    "schema_version",
                    "phase",
                    "status",
                    "input_sha256",
                    "error_type",
                    "error_message_sha256",
                    "truth_accessed",
                    "duration_seconds",
                    "max_rss_bytes",
                },
                "failed phase journal",
            )
            expected_phase = {
                "SOURCE_BINDING_FAILED_PRE_TRUTH": "source_bind",
                "SCORE_APPLY_FAILED_PRE_TRUTH": "score_apply",
                "EVALUATION_FAILED_POST_TRUTH": "evaluate",
            }[terminal]
            if (
                journal["phase"] != expected_phase
                or journal["status"] != "FAILED"
                or journal["truth_accessed"]
                is not (terminal == "EVALUATION_FAILED_POST_TRUTH")
                or float(journal["duration_seconds"]) < 0.0
                or int(journal["max_rss_bytes"]) < 0
            ):
                raise PresenceMatrixError("Wave 60 failed journal drifted")
    return expected, sorted(expected - actual), sorted(actual - expected)


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
        if (
            truth_accessed is not False
            or peer_terminal not in ROOT_FAILURE_SEMANTICS
            or ROOT_FAILURE_SEMANTICS[peer_terminal]["truth_accessed"] is not False
            or last_complete_phase
            not in {
                "INITIALIZED",
                "PREPARED",
                "SOURCE_LAW_BOUND",
                "LOCKBOX_ACTIONS_FROZEN",
            }
        ):
            raise RuntimeError("Wave 60 peer-abort semantics drifted")
    else:
        if peer_terminal is not None or peer_terminal_binding_sha256 is not None:
            raise RuntimeError("Wave 60 own failure may not bind a peer")
        semantics = ROOT_FAILURE_SEMANTICS[terminal]
        if (
            phase != semantics["phase"]
            or last_complete_phase != semantics["last_complete_phase"]
            or truth_accessed is not semantics["truth_accessed"]
        ):
            raise RuntimeError("Wave 60 failure terminal semantics drifted")
    _, missing_expected, forbidden_present = _expected_failure_prefix(
        root, role, terminal, last_complete_phase
    )
    if missing_expected or forbidden_present:
        raise PresenceMatrixError(
            "Wave 60 failure presence matrix drifted: "
            f"missing={missing_expected} forbidden={forbidden_present}"
        )
    failure = {
        "schema_version": "wave60-root-failure-v1",
        "status": "FAILED",
        "terminal": terminal,
        "phase": phase,
        "run_role": role,
        "truth_accessed": bool(truth_accessed),
        "recovery_allowed": (
            True
            if terminal == "PEER_ABORTED_PRE_TRUTH"
            else ROOT_FAILURE_SEMANTICS[terminal]["recovery_allowed"]
        ),
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
        "missing_expected": missing_expected,
        "forbidden_present": forbidden_present,
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


def validate_root_terminal(
    root: Path,
    role: str,
    *,
    expected_terminal: str | None = None,
    public_key: Path = TRUSTED_PUBLIC_KEY,
) -> tuple[str, str, bool, dict[str, Any] | None]:
    """Validate one physical root terminal and return its attested binding."""
    if role not in {"primary", "replay"}:
        raise ValueError("invalid Wave 60 root role")
    require_physical_directory(root, f"{role} root")
    terminal_files = {
        "FAILURE.json",
        "failure_inventory.json",
        "failure_attestation.json",
    }
    has_manifest = (root / "artifact_manifest.json").is_file()
    present_failure = {name for name in terminal_files if (root / name).is_file()}
    if has_manifest:
        if present_failure:
            raise IntegrityDriftError("Wave 60 root has contradictory terminals")
        terminal = "EVALUATED_IMMUTABLE"
        binding = validate_evaluated_root(root, role)
        if expected_terminal is not None and terminal != expected_terminal:
            raise IntegrityDriftError("Wave 60 root terminal differs from pair status")
        return terminal, binding, True, None
    if present_failure != terminal_files:
        raise IntegrityDriftError("Wave 60 root failure terminal is incomplete")

    failure = read_json(root / "FAILURE.json")
    require_exact_keys(
        failure,
        {
            "schema_version",
            "status",
            "terminal",
            "phase",
            "run_role",
            "truth_accessed",
            "recovery_allowed",
            "error_type",
            "error_message_sha256",
            "authority_binding_sha256",
            "git_commit",
            "peer_terminal",
            "peer_terminal_binding_sha256",
            "created_at",
        },
        "root failure",
    )
    terminal = failure["terminal"]
    if (
        failure["schema_version"] != "wave60-root-failure-v1"
        or failure["status"] != "FAILED"
        or terminal not in ROOT_FAILURE_TERMINALS
        or failure["run_role"] != role
        or not isinstance(failure["error_type"], str)
        or not isinstance(failure["created_at"], str)
        or re.fullmatch(r"[0-9a-f]{64}", failure["error_message_sha256"] or "")
        is None
        or re.fullmatch(r"[0-9a-f]{64}", failure["authority_binding_sha256"] or "")
        is None
        or re.fullmatch(r"[0-9a-f]{40}", failure["git_commit"] or "") is None
        or (expected_terminal is not None and terminal != expected_terminal)
    ):
        raise IntegrityDriftError("Wave 60 root failure identity drifted")

    failure_inventory = read_json(root / "failure_inventory.json")
    require_exact_keys(
        failure_inventory,
        {
            "schema_version",
            "terminal",
            "last_complete_phase",
            "files",
            "classes",
            "missing_expected",
            "forbidden_present",
            "created_at",
        },
        "root failure inventory",
    )
    last_complete = failure_inventory["last_complete_phase"]
    if terminal == "PEER_ABORTED_PRE_TRUTH":
        peer_terminal = failure["peer_terminal"]
        peer_binding = failure["peer_terminal_binding_sha256"]
        semantic_ok = (
            failure["phase"] == "peer_abort"
            and failure["truth_accessed"] is False
            and failure["recovery_allowed"] is True
            and peer_terminal in ROOT_FAILURE_SEMANTICS
            and ROOT_FAILURE_SEMANTICS[peer_terminal]["truth_accessed"] is False
            and re.fullmatch(r"[0-9a-f]{64}", peer_binding or "") is not None
            and last_complete
            in {
                "INITIALIZED",
                "PREPARED",
                "SOURCE_LAW_BOUND",
                "LOCKBOX_ACTIONS_FROZEN",
            }
        )
    else:
        semantics = ROOT_FAILURE_SEMANTICS[terminal]
        semantic_ok = (
            failure["phase"] == semantics["phase"]
            and last_complete == semantics["last_complete_phase"]
            and failure["truth_accessed"] is semantics["truth_accessed"]
            and failure["recovery_allowed"] is semantics["recovery_allowed"]
            and failure["peer_terminal"] is None
            and failure["peer_terminal_binding_sha256"] is None
        )
    if not semantic_ok:
        raise IntegrityDriftError("Wave 60 root failure semantics drifted")

    authority_path = (
        root / "score/monitor_action_freeze.json"
        if terminal == "EVALUATION_FAILED_POST_TRUTH"
        else root / "config.snapshot.json"
    )
    if (
        not authority_path.is_file()
        or file_sha256(authority_path) != failure["authority_binding_sha256"]
    ):
        raise IntegrityDriftError("Wave 60 root failure authority binding drifted")
    expected_prefix, missing, forbidden = _expected_failure_prefix(
        root, role, terminal, last_complete
    )
    if missing or forbidden:
        raise PresenceMatrixError(
            "Wave 60 sealed failure presence matrix drifted: "
            f"missing={missing} forbidden={forbidden}"
        )
    physical = inventory(root)
    if set(physical) != expected_prefix | terminal_files:
        raise PresenceMatrixError("Wave 60 sealed root failure is not closed-world")
    before_inventory = {
        relative: record
        for relative, record in physical.items()
        if relative not in {"failure_inventory.json", "failure_attestation.json"}
    }
    expected_files = {
        **{relative: record["sha256"] for relative, record in before_inventory.items()},
        "failure_inventory.json": "SELF_REFERENCE",
        "failure_attestation.json": "FUTURE_ATTESTATION",
    }
    expected_classes = {
        **{relative: root_artifact_class(relative) for relative in before_inventory},
        "failure_inventory.json": "SELF_REFERENCE",
        "failure_attestation.json": "FAILURE_CONDITIONAL",
    }
    if (
        failure_inventory["schema_version"]
        != "wave60-root-failure-inventory-v1"
        or failure_inventory["terminal"] != terminal
        or failure_inventory["files"] != expected_files
        or failure_inventory["classes"] != expected_classes
        or failure_inventory["missing_expected"] != []
        or failure_inventory["forbidden_present"] != []
        or not isinstance(failure_inventory["created_at"], str)
    ):
        raise IntegrityDriftError("Wave 60 root failure inventory drifted")
    attestation = read_json(root / "failure_attestation.json")
    verify_wave60_attestation(attestation, public_key)
    expected_payload = {
        "scope": role,
        "terminal": terminal,
        "failure_sha256": file_sha256(root / "FAILURE.json"),
        "failure_inventory_sha256": file_sha256(root / "failure_inventory.json"),
    }
    if (
        attestation["schema_version"]
        != "wave60-root-failure-attestation-v1"
        or attestation["phase"] != failure["phase"]
        or attestation["payload"] != expected_payload
    ):
        raise IntegrityDriftError("Wave 60 root failure attestation drifted")
    return (
        terminal,
        file_sha256(root / "failure_attestation.json"),
        bool(failure["truth_accessed"]),
        failure,
    )


def validate_pair_status_against_roots(
    attempt: Path,
    status: Mapping[str, Any],
    *,
    public_key: Path = TRUSTED_PUBLIC_KEY,
) -> dict[str, tuple[str, str, bool, dict[str, Any] | None]]:
    require_physical_directory(attempt, "attempt container")
    require_exact_keys(
        status,
        {
            "schema_version",
            "terminal",
            "primary_terminal",
            "replay_terminal",
            "primary_terminal_binding_sha256",
            "replay_terminal_binding_sha256",
            "any_truth_accessed",
            "recovery_allowed",
            "created_at",
        },
        "pair status",
    )
    if status["schema_version"] != "wave60-pair-status-v1" or not isinstance(
        status["created_at"], str
    ):
        raise IntegrityDriftError("Wave 60 pair status identity drifted")
    roots = {
        role: validate_root_terminal(
            attempt / role,
            role,
            expected_terminal=status[f"{role}_terminal"],
            public_key=public_key,
        )
        for role in ("primary", "replay")
    }
    for role, (_, binding, _, _) in roots.items():
        if binding != status[f"{role}_terminal_binding_sha256"]:
            raise IntegrityDriftError("Wave 60 pair status root binding drifted")
    any_truth = any(root[2] for root in roots.values())
    both_evaluated = all(
        root[0] == "EVALUATED_IMMUTABLE" for root in roots.values()
    )
    expected_pair_terminal = (
        status["terminal"]
        if both_evaluated
        and status["terminal"] in {"COMPLETE", "PAIR_ABORTED_POST_TRUTH"}
        else ("PAIR_ABORTED_POST_TRUTH" if any_truth else "PAIR_ABORTED_PRE_TRUTH")
    )
    if (
        status["terminal"] != expected_pair_terminal
        or status["any_truth_accessed"] is not any_truth
        or status["recovery_allowed"] is not (
            expected_pair_terminal == "PAIR_ABORTED_PRE_TRUTH"
        )
    ):
        raise IntegrityDriftError("Wave 60 pair terminal semantics drifted")
    for role, root_state in roots.items():
        failure = root_state[3]
        if failure is None or failure["terminal"] != "PEER_ABORTED_PRE_TRUTH":
            continue
        peer = "replay" if role == "primary" else "primary"
        if (
            failure["peer_terminal"] != roots[peer][0]
            or failure["peer_terminal_binding_sha256"] != roots[peer][1]
        ):
            raise IntegrityDriftError("Wave 60 peer-abort directional binding drifted")
    return roots


def validate_pair_failure_package(
    attempt: Path,
    *,
    pair_path: Path | None = None,
    public_key: Path = TRUSTED_PUBLIC_KEY,
) -> dict[str, Any]:
    """Validate a closed pair failure against both physical root terminals."""
    pair = attempt / "pair" if pair_path is None else pair_path
    require_physical_directory(pair, "pair failure package")
    status = read_json(pair / "pair_status.json")
    validate_pair_status_against_roots(attempt, status, public_key=public_key)
    if status["terminal"] not in {
        "PAIR_ABORTED_PRE_TRUTH",
        "PAIR_ABORTED_POST_TRUTH",
    }:
        raise IntegrityDriftError("Wave 60 pair failure claims a non-failure terminal")
    expected_paths = {
        "pair_status.json",
        "FAILURE.json",
        "failure_inventory.json",
        "failure_attestation.json",
        "artifact_manifest.json",
    }
    physical = inventory(pair)
    if set(physical) != expected_paths:
        raise PresenceMatrixError("Wave 60 pair failure package is not closed-world")
    failure = read_json(pair / "FAILURE.json")
    require_exact_keys(
        failure,
        {
            "schema_version",
            "status",
            "terminal",
            "phase",
            "run_role",
            "truth_accessed",
            "recovery_allowed",
            "error_type",
            "error_message_sha256",
            "authority_binding_sha256",
            "git_commit",
            "created_at",
        },
        "pair failure",
    )
    if (
        failure["schema_version"] != "wave60-pair-failure-v1"
        or failure["status"] != "FAILED"
        or failure["terminal"] != status["terminal"]
        or failure["phase"] != "pair_finalize"
        or failure["run_role"] != "pair"
        or failure["truth_accessed"] is not status["any_truth_accessed"]
        or failure["recovery_allowed"] is not status["recovery_allowed"]
        or failure["authority_binding_sha256"]
        != file_sha256(pair / "pair_status.json")
        or re.fullmatch(r"[0-9a-f]{64}", failure["error_message_sha256"] or "")
        is None
        or re.fullmatch(r"[0-9a-f]{40}", failure["git_commit"] or "") is None
    ):
        raise IntegrityDriftError("Wave 60 pair failure identity drifted")
    failure_inventory = read_json(pair / "failure_inventory.json")
    require_exact_keys(
        failure_inventory,
        {
            "schema_version",
            "terminal",
            "root_terminal_bindings",
            "files",
            "classes",
            "missing_expected",
            "forbidden_present",
            "created_at",
        },
        "pair failure inventory",
    )
    before_inventory = {
        relative: record
        for relative, record in physical.items()
        if relative
        not in {
            "failure_inventory.json",
            "failure_attestation.json",
            "artifact_manifest.json",
        }
    }
    expected_files = {
        **{relative: record["sha256"] for relative, record in before_inventory.items()},
        "failure_inventory.json": "SELF_REFERENCE",
        "failure_attestation.json": "FUTURE_ATTESTATION",
        "artifact_manifest.json": "SELF_REFERENCE",
    }
    expected_classes = {
        **{relative: "FAILURE_CONDITIONAL" for relative in before_inventory},
        "failure_inventory.json": "SELF_REFERENCE",
        "failure_attestation.json": "FAILURE_CONDITIONAL",
        "artifact_manifest.json": "SELF_REFERENCE",
    }
    if (
        failure_inventory["schema_version"]
        != "wave60-pair-failure-inventory-v1"
        or failure_inventory["terminal"] != status["terminal"]
        or failure_inventory["root_terminal_bindings"]
        != {
            "primary": status["primary_terminal_binding_sha256"],
            "replay": status["replay_terminal_binding_sha256"],
        }
        or failure_inventory["files"] != expected_files
        or failure_inventory["classes"] != expected_classes
        or failure_inventory["missing_expected"] != []
        or failure_inventory["forbidden_present"] != []
    ):
        raise IntegrityDriftError("Wave 60 pair failure inventory drifted")
    attestation = read_json(pair / "failure_attestation.json")
    verify_wave60_attestation(attestation, public_key)
    expected_payload = {
        "scope": "pair",
        "pair_status_sha256": file_sha256(pair / "pair_status.json"),
        "failure_sha256": file_sha256(pair / "FAILURE.json"),
        "failure_inventory_sha256": file_sha256(pair / "failure_inventory.json"),
    }
    if (
        attestation["schema_version"]
        != "wave60-pair-failure-attestation-v1"
        or attestation["phase"] != "pair_finalize"
        or attestation["payload"] != expected_payload
    ):
        raise IntegrityDriftError("Wave 60 pair failure attestation drifted")
    manifest = read_json(pair / "artifact_manifest.json")
    require_exact_keys(
        manifest,
        {"schema_version", "terminal", "files", "classes", "self_reference"},
        "pair failure manifest",
    )
    manifest_files = dict(physical)
    manifest_files.pop("artifact_manifest.json")
    if (
        manifest["schema_version"] != "wave60-pair-final-v1"
        or manifest["terminal"] != status["terminal"]
        or manifest["files"] != manifest_files
        or manifest["classes"]
        != {relative: pair_artifact_class(relative) for relative in manifest_files}
        or manifest["self_reference"]
        != {"path": "artifact_manifest.json", "hashes_omitted": True}
    ):
        raise IntegrityDriftError("Wave 60 pair failure manifest drifted")
    return status


def publish_pair_failure(
    attempt: Path,
    status: Mapping[str, Any],
    *,
    error: BaseException,
    private_key: Path = DEFAULT_PRIVATE_KEY,
) -> Path:
    if status["terminal"] not in {"PAIR_ABORTED_PRE_TRUTH", "PAIR_ABORTED_POST_TRUTH"}:
        raise ValueError("invalid Wave 60 pair failure terminal")
    validate_pair_status_against_roots(attempt, status)
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
        fsync_tree(staging)
        validate_pair_failure_package(attempt, pair_path=staging)
        os.replace(staging, target)
        fsync_directory(attempt)
        validate_pair_failure_package(attempt)
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
    preparation_elapsed = 0.0

    def remaining_seconds() -> float:
        remaining = (
            maximum_seconds
            - preparation_elapsed
            - (time.monotonic() - execution_started)
        )
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
    preparation_elapsed = pair_preparation_elapsed(roots["primary"], roots["replay"])
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
            root_bindings[role] = seal_evaluated_root(
                root,
                role,
                coordinator_elapsed_seconds=time.monotonic() - execution_started,
            )
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
        return finalize_pair(
            attempt,
            private_key=private_key,
            elapsed_before_finalize=(
                preparation_elapsed + (time.monotonic() - execution_started)
            ),
            maximum_seconds=maximum_seconds,
        )
    except IntegrityDriftError as error:
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
            pair_failed=True,
        )
        return publish_pair_failure(
            attempt, status, error=error, private_key=private_key
        )


def finalize_pair(
    attempt: Path,
    *,
    private_key: Path = DEFAULT_PRIVATE_KEY,
    elapsed_before_finalize: float | None = None,
    maximum_seconds: float = 900.0,
) -> Path:
    finalize_started = time.monotonic()
    primary = attempt / "primary"
    replay = attempt / "replay"
    durable_budget = pair_durable_elapsed(primary, replay)
    before_finalize = max(
        durable_budget["durable_seconds"],
        float(elapsed_before_finalize or 0.0),
    )
    if maximum_seconds != 900.0 or before_finalize >= maximum_seconds:
        raise IntegrityDriftError("Wave 60 combined CPU budget is exhausted")
    try:
        primary_binding = validate_evaluated_root(primary, "primary")
        replay_binding = validate_evaluated_root(replay, "replay")
    except RuntimeError as error:
        if isinstance(error, IntegrityDriftError):
            raise
        raise IntegrityDriftError(str(error)) from error
    target = attempt / "pair"
    if target.exists():
        raise FileExistsError(target)
    staging = attempt / "pair.initializing"
    if staging.exists():
        if staging.is_symlink() or not staging.is_dir():
            raise IntegrityDriftError(
                "Wave 60 pair staging is not a physical directory"
            )
    else:
        staging.mkdir(mode=0o700)
    runtime_path = staging / "runtime.json"
    prior_finalize_seconds = 0.0
    prior_finalize_invocations = 0
    prior_runtime_sha256: str | None = None
    if runtime_path.exists():
        prior_runtime = read_json(runtime_path)
        prior_budget = prior_runtime.get("budget", {})
        if (
            prior_runtime.get("schema_version") != "wave60-pair-final-v1"
            or prior_runtime.get("terminal") != "COMPLETE"
            or prior_runtime.get("cuda_visible_devices") != ""
            or prior_runtime.get("cpu_threads") != 4
            or set(prior_budget)
            != {
                "max_seconds_total",
                "preparation_seconds",
                "worker_seconds",
                "coordinator_seconds",
                "durable_seconds",
                "elapsed_before_finalize_seconds",
                "finalize_seconds",
                "finalize_invocations",
                "observed_total_seconds",
                "budget_enforced",
            }
            or float(prior_budget["max_seconds_total"]) != maximum_seconds
            or float(prior_budget["preparation_seconds"])
            != durable_budget["preparation_seconds"]
            or float(prior_budget["worker_seconds"])
            != durable_budget["worker_seconds"]
            or float(prior_budget["coordinator_seconds"])
            != durable_budget["coordinator_seconds"]
            or float(prior_budget["durable_seconds"])
            != durable_budget["durable_seconds"]
            or float(prior_budget["elapsed_before_finalize_seconds"])
            < durable_budget["durable_seconds"]
            or float(prior_budget["finalize_seconds"]) < 0.0
            or not isinstance(prior_budget["finalize_invocations"], int)
            or int(prior_budget["finalize_invocations"]) < 1
            or float(prior_budget["observed_total_seconds"])
            != float(prior_budget["elapsed_before_finalize_seconds"])
            + float(prior_budget["finalize_seconds"])
            or float(prior_budget["observed_total_seconds"]) >= maximum_seconds
            or prior_budget["budget_enforced"] is not True
        ):
            raise IntegrityDriftError("Wave 60 staged runtime budget drifted")
        before_finalize = max(
            before_finalize,
            float(prior_budget["elapsed_before_finalize_seconds"]),
        )
        prior_finalize_seconds = float(prior_budget["finalize_seconds"])
        prior_finalize_invocations = int(prior_budget["finalize_invocations"])
        prior_runtime_sha256 = file_sha256(runtime_path)
    prior_manifest_path = staging / "artifact_manifest.json"
    if prior_manifest_path.exists():
        prior_manifest = read_json(prior_manifest_path)
        prior_files = inventory(staging)
        prior_files.pop("artifact_manifest.json")
        expected_prior_paths = {
            "pair_status.json",
            "replay_comparison.json",
            "final_analysis.json",
            "replay_finalize_freeze.json",
            "journals/replay_finalize.json",
            "replay_finalize_receipt.json",
            "REPORT.md",
            "runtime.json",
            "replay_finalize_attestation.json",
        }
        if (
            prior_runtime_sha256 is None
            or set(prior_files) != expected_prior_paths
            or prior_manifest
            != {
                "schema_version": "wave60-pair-final-v1",
                "terminal": "COMPLETE",
                "files": prior_files,
                "classes": {
                    relative: pair_artifact_class(relative)
                    for relative in prior_files
                },
                "self_reference": {
                    "path": "artifact_manifest.json",
                    "hashes_omitted": True,
                },
            }
        ):
            raise IntegrityDriftError("Wave 60 staged pair manifest drifted")
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
            raise IntegrityDriftError(
                "Wave 60 staged pair status binds different roots"
            )
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
        raise IntegrityDriftError(
            "Wave 60 replay normalization is internally inconsistent"
        )
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
            raise IntegrityDriftError("Wave 60 staged replay receipt drifted")
    else:
        receipt = {
            **receipt_static,
            "started_at": now(),
            "completed_at": now(),
        }
        write_json(receipt_path, receipt, mode=0o444)
    report = (
        "# Wave 60 — frozen policy transport\n\n"
        f"Pair terminal: `COMPLETE`. Replay exact: `{str(replay_exact).lower()}`.\n\n"
        "Scientific decision remains with the user. The result is conditional on the new "
        "synthetic draw and the frozen Wave 59 law.\n"
    )
    ensure_text(staging / "REPORT.md", report, mode=0o444)
    current_finalize_seconds = time.monotonic() - finalize_started
    finalize_seconds = prior_finalize_seconds + current_finalize_seconds
    observed_total = before_finalize + finalize_seconds
    if observed_total >= maximum_seconds:
        raise IntegrityDriftError("Wave 60 finalize exceeded its CPU budget")
    prior_attestation_path = staging / "replay_finalize_attestation.json"
    if prior_attestation_path.exists():
        if prior_runtime_sha256 is None:
            raise IntegrityDriftError("Wave 60 staged attestation lacks prior runtime")
        prior_attestation = read_json(prior_attestation_path)
        verify_wave60_attestation(prior_attestation)
        prior_payload = {
            "scope": "pair",
            "freeze_sha256": file_sha256(staging / "replay_finalize_freeze.json"),
            "receipt_sha256": file_sha256(staging / "replay_finalize_receipt.json"),
            "journal_sha256": file_sha256(staging / "journals/replay_finalize.json"),
            "runtime_sha256": prior_runtime_sha256,
            "primary_evaluation_attestation_sha256": freeze[
                "primary_evaluation_attestation_sha256"
            ],
            "replay_evaluation_attestation_sha256": freeze[
                "replay_evaluation_attestation_sha256"
            ],
        }
        if (
            prior_attestation["schema_version"]
            != "wave60-replay-finalize-v1"
            or prior_attestation["phase"] != "replay_finalize"
            or prior_attestation["payload"] != prior_payload
        ):
            raise IntegrityDriftError("Wave 60 staged replay attestation drifted")
    runtime = {
        "schema_version": "wave60-pair-final-v1",
        "terminal": "COMPLETE",
        "cuda_visible_devices": "",
        "cpu_threads": 4,
        "budget": {
            "max_seconds_total": maximum_seconds,
            **durable_budget,
            "elapsed_before_finalize_seconds": before_finalize,
            "finalize_seconds": finalize_seconds,
            "finalize_invocations": prior_finalize_invocations + 1,
            "observed_total_seconds": observed_total,
            "budget_enforced": True,
        },
    }
    write_json(runtime_path, runtime, mode=0o444)
    attestation_payload = {
        "scope": "pair",
        "freeze_sha256": file_sha256(staging / "replay_finalize_freeze.json"),
        "receipt_sha256": file_sha256(staging / "replay_finalize_receipt.json"),
        "journal_sha256": file_sha256(staging / "journals/replay_finalize.json"),
        "runtime_sha256": file_sha256(runtime_path),
        "primary_evaluation_attestation_sha256": freeze[
            "primary_evaluation_attestation_sha256"
        ],
        "replay_evaluation_attestation_sha256": freeze[
            "replay_evaluation_attestation_sha256"
        ],
    }
    attestation_path = staging / "replay_finalize_attestation.json"
    attestation = make_attestation(
        "evaluate",
        attestation_payload,
        private_key,
    )
    attestation["schema_version"] = "wave60-replay-finalize-v1"
    attestation["phase"] = "replay_finalize"
    write_json(attestation_path, attestation, mode=0o444)
    verify_wave60_attestation(attestation)
    files = inventory(staging)
    files.pop("artifact_manifest.json", None)
    manifest = {
        "schema_version": "wave60-pair-final-v1",
        "terminal": "COMPLETE",
        "files": files,
        "classes": {relative: pair_artifact_class(relative) for relative in files},
        "self_reference": {"path": "artifact_manifest.json", "hashes_omitted": True},
    }
    write_json(staging / "artifact_manifest.json", manifest, mode=0o444)
    fsync_tree(staging)
    if (
        before_finalize
        + prior_finalize_seconds
        + (time.monotonic() - finalize_started)
        >= maximum_seconds
    ):
        raise IntegrityDriftError("Wave 60 finalize exceeded its CPU budget")
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
