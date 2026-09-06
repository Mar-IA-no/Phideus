#!/usr/bin/env python3
"""Escrow, generate, and blindly infer the fresh Wave 56 Stage 1 benchmark."""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
from copy import deepcopy
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import pwd
import grp
import re
import secrets
import signal
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Callable

# The shared preparer is CPU-only in every supported prospective contract.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = "4"

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from geometria_proporcional.wave49_checker import (  # noqa: E402
    validate_manifest,
    validate_semantic_attestation,
    validate_visible_package,
)
from geometria_proporcional.wave49_attestation import (  # noqa: E402
    AttestationError,
    sign_attestation,
    verify_attestation,
)
from geometria_proporcional.wave49_generator import generate_benchmark  # noqa: E402
from geometria_proporcional.wave49_schema import (  # noqa: E402
    ProtocolConfig,
    canonical_json,
    default_protocol_config,
    read_jsonl,
    sha256_bytes,
    sha256_file,
)
from geometria_proporcional.wave50_model import FeatureNormalizer  # noqa: E402
from geometria_proporcional.wave50_neural import (  # noqa: E402
    load_labeled_records,
    prepare_examples,
    split_tokens,
    stratified_token_subset,
)
from geometria_proporcional.wave51_factored import (  # noqa: E402
    DualHeadDeepSet,
    predict_dual_logits,
)
from geometria_proporcional.wave56_contextual_gate import FEATURE_NAMES  # noqa: E402

CONFIG_DEFAULT = REPO_ROOT / "experiments/geometria_proporcional/configs/wave56_contextual_gate_fresh.json"
WORKER_PATH = REPO_ROOT / "experiments/geometria_proporcional/_wave56_infer_worker.py"
PUBLIC_KEY = REPO_ROOT / "experiments/geometria_proporcional/keys/wave49_attestation_public.pem"
ESCROW_NAME = "generation_escrow.json"
FREEZE_NAME = "pre_generation_freeze.json"
RECOVERY_AMENDMENT_COPY_NAME = "recovery_amendment.json"
WAVE59_PREPARATION_ATTESTATION_NAME = "preparation_attestation.json"
WAVE59_PREPARATION_ATTESTATION_SCHEMA = "wave59-signed-preparation-authority-v1"
WAVE59_FRESH_PREPARATION_ATTESTATION_SCHEMA = (
    "wave59-signed-fresh-preparation-authority-v1"
)
RECOVERY_AMENDMENT_SCHEMA = "wave56-stage1-authority-matrix-finalization-amendment-v1"
RECOVERY_AMENDMENT_RELATIVE = (
    "experiments/geometria_proporcional/configs/"
    "wave56_stage1_authority_matrix_finalization_amendment_v8.json"
)
RECOVERY_PLAN_RELATIVE = (
    "Biblioteca/Geometria_Proporcional_Ground_Truth/waves/"
    "WAVE_56_STAGE1_AUTHORITY_MATRIX_FINALIZATION_PLAN.md"
)
AUDIT_REPORTS_RELATIVE_DIR = (
    "Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports"
)
PREPARER_RELATIVE = "experiments/geometria_proporcional/prepare_wave56_fresh.py"
RUNNER_RELATIVE = "experiments/geometria_proporcional/run_wave56_contextual_gate.py"
RECOVERY_TEST_RELATIVE = "tests/test_wave56_preoracle_recovery.py"
WAVE57_RECOVERY_AMENDMENT_SCHEMA = (
    "wave57-contextual-tail-guard-preoracle-recovery-amendment-v1"
)
WAVE57_RECOVERY_AMENDMENT_RELATIVE = (
    "experiments/geometria_proporcional/configs/"
    "wave57_preoracle_pair_token_recovery_amendment_v1.json"
)
WAVE57_RECOVERY_PLAN_RELATIVE = (
    "Biblioteca/Geometria_Proporcional_Ground_Truth/waves/"
    "WAVE_57_PREORACLE_PAIR_TOKEN_RECOVERY_PLAN.md"
)
WAVE57_RECOVERY_TEST_RELATIVE = "tests/test_wave57_prospective.py"
WAVE57_CONFIG_SCHEMA = "wave57-contextual-harm-guard-v1"
WAVE59_CONFIG_SCHEMA = "wave59-fresh-hgb-guard-bracket-v1"
WAVE60_CONFIG_SCHEMA = "wave60-frozen-policy-transport-v1"
WAVE60_RECOVERY_AMENDMENT_SCHEMA = "wave60-pretruth-recovery-amendment-v1"
WAVE59_RECOVERY_AMENDMENT_SCHEMA = (
    "wave59-hgb-guard-bracket-preoracle-recovery-amendment-v1"
)
WAVE59_RECOVERY_AMENDMENT_RELATIVE = (
    "experiments/geometria_proporcional/configs/"
    "wave59_preoracle_pair_token_recovery_amendment_v1.json"
)
WAVE59_RECOVERY_PLAN_RELATIVE = (
    "Biblioteca/Geometria_Proporcional_Ground_Truth/waves/"
    "WAVE_59_PREORACLE_PAIR_TOKEN_RECOVERY_PLAN.md"
)
WAVE59_RUNNER_RELATIVE = (
    "experiments/geometria_proporcional/run_wave59_hgb_guard_bracket.py"
)
WAVE59_PROSPECTIVE_TEST_RELATIVE = "tests/test_wave59_prospective.py"
WAVE59_RECOVERY_TEST_RELATIVE = "tests/test_wave59_preoracle_recovery.py"
WAVE59_SUCCESSOR_PLAN_RELATIVE = (
    "Biblioteca/Geometria_Proporcional_Ground_Truth/waves/"
    "WAVE_59_REPLAY_NORMALIZATION_SUCCESSOR_PLAN.md"
)
WAVE59_SUCCESSOR_IMPLEMENTATION_PATHS = frozenset(
    {
        "src/geometria_proporcional/wave59_hgb_guard_bracket.py",
        PREPARER_RELATIVE,
        WAVE59_RUNNER_RELATIVE,
        WAVE59_PROSPECTIVE_TEST_RELATIVE,
        WAVE59_RECOVERY_TEST_RELATIVE,
    }
)
WAVE59_ANTECEDENT_PRIMARY_RELATIVE = (
    "data/geometria_proporcional/wave59_fresh_hgb_guard_bracket_v1"
)
WAVE59_ANTECEDENT_REPLAY_FAILURE_RELATIVE = (
    "data/geometria_proporcional/"
    "wave59_fresh_hgb_guard_bracket_v1_replay.failed_20260905T102929791142Z"
)
WAVE59_ANTECEDENT_SENTINELS = {
    "analysis.json": "79a4c1eca78497d9fd6ea49177508ac9f879cabe0824b566b437d8b9f8e7eb96",
    "artifact_manifest.json": "ce30675744b27d656ff7eb177145ae5fac77d9f092ccc483d329425864185770",
    "FAILURE.json": "4967c6108ffd07e1b8cf6c0f301e702722be592f0d678594dd89e659053aa3c4",
    "failure_inventory.json": "353afe6f61c476a82d5c061fadcdfa13bad5c27787629189cb80094d4769b9a9",
    "failure_attestation.json": "8cb19ce7d741a0caaa73c88d44c96fe10557dbba6283290f90e98bbec0cb93e3",
}
PHASE_ENTRY_IMPLEMENTATION_COMMIT = "7b37b5381b0c7540e86de2d53001903475d321ab"
AUTHORITY_IMPLEMENTATION_COMMIT = "3f404103111a67721fa7a3d15cbf4ec392025e5f"
COVERAGE_IMPLEMENTATION_COMMIT = "68316175067419c914af584e14ec2bafa4ff550b"
SPLITS = ("train", "val", "lockbox")
INFERENCE_RUNTIME_SOURCES = (
    "__init__.py",
    "wave49_schema.py",
    "wave50_model.py",
    "wave50_neural.py",
    "wave51_factored.py",
)
SECRET_FILES = (
    "generation_secret.json",
    "identity_secret.json",
    "semantic_commitment_secret.json",
)


def preparation_phase_prefix(config: dict[str, Any]) -> str:
    """Return the wave owning preparation metadata without changing Wave 56 labels."""
    if config.get("schema_version") == WAVE57_CONFIG_SCHEMA:
        return "wave57"
    if config.get("schema_version") == WAVE59_CONFIG_SCHEMA:
        return "wave59"
    if config.get("schema_version") == WAVE60_CONFIG_SCHEMA:
        return "wave60"
    return "wave56"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wave50-dir", type=Path, required=True)
    parser.add_argument("--wave51-dir", type=Path, required=True)
    parser.add_argument("--wave52-dir", type=Path, required=True)
    parser.add_argument("--wave54-dir", type=Path, required=True)
    parser.add_argument("--wave55-dir", type=Path, required=True)
    parser.add_argument("--stage0-dir", type=Path, required=True)
    parser.add_argument("--wave56-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=CONFIG_DEFAULT)
    parser.add_argument("--attestation-private-key", type=Path, required=True)
    parser.add_argument("--replay-secrets-from", type=Path)
    parser.add_argument("--recovery-secrets-from", type=Path)
    parser.add_argument("--recovery-amendment", type=Path)
    parser.add_argument("--reference-dir", type=Path)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def digest(path: Path) -> str:
    return sha256_file(path.resolve(strict=True))


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def fsync_tree(root: Path) -> None:
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


def atomic_write_json(path: Path, payload: Any, mode: int = 0o600) -> str:
    """Publish canonical JSON durably and verify bytes, payload, and permissions."""
    encoded = (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode()
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        fsync_directory(path.parent)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    actual = path.read_bytes()
    if actual != encoded or json.loads(actual) != payload:
        raise RuntimeError(f"atomic publication verification failed: {path}")
    if stat.S_IMODE(path.stat().st_mode) != mode:
        raise PermissionError(f"unexpected mode for {path}")
    return hashlib.sha256(actual).hexdigest()


def canonical_json_sha256(payload: Any) -> str:
    encoded = (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode()
    return hashlib.sha256(encoded).hexdigest()


def compact_json_sha256(payload: Any) -> str:
    return sha256_bytes(canonical_json(payload).encode())


def _git_output(repo_root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo_root, text=True).strip()


def git_blob_sha256(repo_root: Path, commit: str, relative: str) -> str:
    payload = subprocess.check_output(
        ["git", "show", f"{commit}:{relative}"], cwd=repo_root
    )
    return sha256_bytes(payload)


def git_changed_paths(repo_root: Path, commit: str) -> set[str]:
    output = _git_output(
        repo_root,
        "diff-tree",
        "--root",
        "--no-commit-id",
        "--name-only",
        "-r",
        commit,
    )
    return {line for line in output.splitlines() if line}


def git_introduction_commit(repo_root: Path, relative: str) -> str:
    output = _git_output(
        repo_root,
        "log",
        "--diff-filter=A",
        "--format=%H",
        "--reverse",
        "--",
        relative,
    )
    commits = [line for line in output.splitlines() if line]
    if len(commits) != 1:
        raise RuntimeError(f"expected one Git introduction commit for {relative}")
    return commits[0]


def require_ancestor(repo_root: Path, ancestor: str, descendant: str) -> None:
    completed = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=repo_root,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Git provenance is not ancestral: {ancestor} -> {descendant}")


def require_direct_parent(repo_root: Path, child: str, expected_parent: str, label: str) -> None:
    lineage = _git_output(repo_root, "rev-list", "--parents", "-n", "1", child).split()
    if len(lineage) != 2 or lineage[1] != expected_parent:
        raise RuntimeError(f"{label} commit must directly descend from its frozen predecessor")


def require_repo_artifact(
    repo_root: Path,
    relative: str,
    expected_sha256: str | None = None,
) -> tuple[Path, str]:
    candidate = Path(relative)
    if candidate.is_absolute() or candidate.as_posix() != relative or ".." in candidate.parts:
        raise RuntimeError(f"non-canonical repository artifact path: {relative}")
    path = repo_root / candidate
    try:
        metadata = path.lstat()
    except FileNotFoundError as exc:
        raise RuntimeError(f"repository artifact is not one regular file: {relative}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise RuntimeError(f"repository artifact is not one regular file: {relative}")
    subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", relative],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )
    if _git_output(repo_root, "status", "--porcelain", "--", relative):
        raise RuntimeError(f"repository artifact differs from HEAD: {relative}")
    actual = sha256_file(path)
    if expected_sha256 is not None and actual != expected_sha256:
        raise RuntimeError(f"repository artifact hash mismatch: {relative}")
    head = _git_output(repo_root, "rev-parse", "HEAD")
    if git_blob_sha256(repo_root, head, relative) != actual:
        raise RuntimeError(f"repository artifact is not identical to HEAD: {relative}")
    return path, actual


def require_audit_report_path(relative: str, label: str) -> None:
    candidate = Path(relative)
    if (
        candidate.parent != Path(AUDIT_REPORTS_RELATIVE_DIR)
        or candidate.suffix != ".md"
        or candidate.name in {"", ".", ".."}
    ):
        raise RuntimeError(
            f"{label} path must be one Markdown report directly under "
            f"{AUDIT_REPORTS_RELATIVE_DIR}"
        )


def parse_wave60_audit_report(
    path: Path, *, scope: str, target: dict[str, str]
) -> dict[str, Any]:
    """Parse the sole normative JSON block; prose can never grant authority."""
    text = path.read_text(encoding="utf-8")
    blocks = re.findall(r"```json\s*\n(.*?)\n```", text, flags=re.DOTALL)
    if len(blocks) != 1:
        raise RuntimeError("Wave 60 audit must contain one canonical JSON block")
    try:
        authority = json.loads(blocks[0])
    except json.JSONDecodeError as exc:
        raise RuntimeError("Wave 60 audit authority JSON is invalid") from exc
    expected_keys = {
        "schema_version", "audit_id", "scope", "target", "technical_verdict",
        "findings", "files_modified", "gpu_used_or_queried",
    }
    if not isinstance(authority, dict) or set(authority) != expected_keys:
        raise RuntimeError("Wave 60 audit authority keyset drifted")
    if (
        authority["schema_version"] != "wave60-audit-authority-v1"
        or not isinstance(authority["audit_id"], str)
        or re.fullmatch(r"R[0-9]+", authority["audit_id"]) is None
        or authority["scope"] != scope
        or authority["target"] != target
        or authority["technical_verdict"] != "PASS"
        or authority["findings"] != {"high": 0, "medium": 0, "low": 0}
        or authority["files_modified"] is not False
        or authority["gpu_used_or_queried"] is not False
    ):
        raise RuntimeError("Wave 60 audit does not grant exact PASS authority")
    return authority


def validate_wave60_audit_commit(
    repo_root: Path,
    binding: dict[str, Any],
    *,
    scope: str,
    target: dict[str, str],
    expected_parent: str,
) -> None:
    relative = binding["audit_path"]
    require_audit_report_path(relative, f"Wave 60 {scope} audit")
    path, _ = require_repo_artifact(repo_root, relative, binding["audit_sha256"])
    introduction = git_introduction_commit(repo_root, relative)
    if introduction != binding["audit_commit"]:
        raise RuntimeError("Wave 60 audit commit differs from report introduction")
    if git_changed_paths(repo_root, introduction) != {relative}:
        raise RuntimeError("Wave 60 audit commit contains unrelated paths")
    if git_blob_sha256(repo_root, introduction, relative) != binding["audit_sha256"]:
        raise RuntimeError("Wave 60 audit report differs from its exclusive commit")
    require_direct_parent(
        repo_root, introduction, expected_parent, f"Wave 60 {scope} audit"
    )
    parse_wave60_audit_report(path, scope=scope, target=target)


def validate_wave60_final_config_authority(
    repo_root: Path,
    config_path: Path,
    config: dict[str, Any],
    head: str,
) -> None:
    """Require config-only commit followed by its exclusive PASS audit at HEAD."""
    relative_config = str(config_path.relative_to(repo_root))
    audit_binding = config["final_audit"]
    relative_audit = audit_binding["audit_path"]
    require_audit_report_path(relative_audit, "Wave 60 final config audit")
    if git_introduction_commit(repo_root, relative_audit) != head:
        raise RuntimeError("Wave 60 final config audit is not HEAD")
    if git_changed_paths(repo_root, head) != {relative_audit}:
        raise RuntimeError("Wave 60 final config audit commit is not exclusive")
    config_commit = _git_output(repo_root, "rev-parse", f"{head}^")
    if git_changed_paths(repo_root, config_commit) != {relative_config}:
        raise RuntimeError("Wave 60 config commit is not exclusive")
    recovery = config["attempt"]["recovery"]
    frozen_predecessor = (
        config["source_law_authority"]["audit_commit"]
        if recovery is None
        else recovery["amendment_audit_commit"]
    )
    require_direct_parent(
        repo_root,
        config_commit,
        frozen_predecessor,
        "Wave 60 config",
    )
    config_sha256 = digest(config_path)
    if git_blob_sha256(repo_root, config_commit, relative_config) != config_sha256:
        raise RuntimeError("Wave 60 config differs from its exclusive commit")
    audit_path, audit_sha256 = require_repo_artifact(
        repo_root, relative_audit
    )
    authority = parse_wave60_audit_report(
        audit_path,
        scope="CONFIG",
        target={
            "config_commit": config_commit,
            "config_sha256": config_sha256,
        },
    )
    if authority["audit_id"] != audit_binding["audit_id"]:
        raise RuntimeError("Wave 60 final config audit id drifted")
    if git_blob_sha256(repo_root, head, relative_audit) != audit_sha256:
        raise RuntimeError("Wave 60 final config audit blob drifted")
    implementation_commit = config["implementation_binding"]["commit"]
    implementation_sources = {
        "src/geometria_proporcional/wave60_frozen_policy_transport.py",
        "experiments/geometria_proporcional/run_wave60_frozen_policy_transport.py",
        "experiments/geometria_proporcional/_wave60_phase_worker.py",
        PREPARER_RELATIVE,
        "tests/test_wave60_frozen_policy_transport.py",
    }
    for relative in implementation_sources:
        expected = git_blob_sha256(repo_root, implementation_commit, relative)
        if config["source_sha256"].get(relative) != expected or digest(
            repo_root / relative
        ) != expected:
            raise RuntimeError(
                f"Wave 60 executed blob differs from audited implementation: {relative}"
            )


def require_canonical_repo_directory(
    repo_root: Path, relative: str, label: str
) -> Path:
    """Resolve one literal repository directory without traversal or symlink aliases."""
    repository = repo_root.resolve(strict=True)
    candidate = Path(relative)
    if (
        candidate.is_absolute()
        or candidate.as_posix() != relative
        or ".." in candidate.parts
    ):
        raise RuntimeError(f"{label} path is not canonical")
    lexical = repository / candidate
    resolved = lexical.resolve(strict=True)
    if resolved != lexical or not resolved.is_relative_to(repository):
        raise RuntimeError(f"{label} path escaped its namespace")
    metadata = resolved.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise RuntimeError(f"{label} is not one physical directory")
    return resolved


def _secure_file_record(path: Path, relative: str) -> dict[str, Any]:
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode):
        raise RuntimeError(f"recovery source contains a non-regular file: {relative}")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        identity = (before.st_dev, before.st_ino, before.st_mode, before.st_uid, before.st_gid, before.st_size)
        opened_identity = (
            opened.st_dev,
            opened.st_ino,
            opened.st_mode,
            opened.st_uid,
            opened.st_gid,
            opened.st_size,
        )
        if identity != opened_identity:
            raise RuntimeError(f"recovery source changed while opening: {relative}")
        hasher = hashlib.sha256()
        while block := os.read(descriptor, 1024 * 1024):
            hasher.update(block)
    finally:
        os.close(descriptor)
    after = path.lstat()
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_uid,
        after.st_gid,
        after.st_size,
    )
    if identity != after_identity:
        raise RuntimeError(f"recovery source changed while hashing: {relative}")
    return {
        "path": relative,
        "type": "file",
        "mode": f"{stat.S_IMODE(before.st_mode):04o}",
        "uid": before.st_uid,
        "gid": before.st_gid,
        "bytes": before.st_size,
        "sha256": hasher.hexdigest(),
    }


def physical_tree_inventory(root: Path) -> list[dict[str, Any]]:
    """Return a closed lstat inventory without following links or special files."""
    root_metadata = root.lstat()
    if stat.S_ISLNK(root_metadata.st_mode):
        raise RuntimeError("recovery source root cannot be a symlink")
    root = root.resolve(strict=True)
    records: list[dict[str, Any]] = []

    def walk(path: Path, relative: str) -> None:
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise RuntimeError(f"recovery source contains a symlink: {relative}")
        if not stat.S_ISDIR(metadata.st_mode):
            raise RuntimeError(f"recovery source root/member is not a directory: {relative}")
        records.append(
            {
                "path": relative,
                "type": "directory",
                "mode": f"{stat.S_IMODE(metadata.st_mode):04o}",
                "uid": metadata.st_uid,
                "gid": metadata.st_gid,
            }
        )
        with os.scandir(path) as entries:
            children = sorted(entries, key=lambda entry: entry.name)
        for entry in children:
            child_relative = entry.name if relative == "." else f"{relative}/{entry.name}"
            child_path = path / entry.name
            child_stat = entry.stat(follow_symlinks=False)
            if stat.S_ISLNK(child_stat.st_mode):
                raise RuntimeError(f"recovery source contains a symlink: {child_relative}")
            if stat.S_ISDIR(child_stat.st_mode):
                walk(child_path, child_relative)
            elif stat.S_ISREG(child_stat.st_mode):
                records.append(_secure_file_record(child_path, child_relative))
            else:
                raise RuntimeError(f"recovery source contains a special file: {child_relative}")

    walk(root, ".")
    return sorted(records, key=lambda row: row["path"])


def sealed_population_counts(path: Path) -> dict[str, int]:
    rows = read_jsonl(path)
    all_tokens = {str(row["pair_token"]) for row in rows}
    eligible = {
        str(row["pair_token"])
        for row in rows
        if not row["is_out_of_catalog"]
        and row["calibration_population"] == "canonical_preserving"
    }
    out_of_catalog = {
        str(row["pair_token"]) for row in rows if row["is_out_of_catalog"]
    }
    noncanonical = {
        str(row["pair_token"])
        for row in rows
        if row["calibration_population"] != "canonical_preserving"
    }
    return {
        "rows": len(rows),
        "total_unique_pair_tokens": len(all_tokens),
        "eligible_unique_pair_tokens": len(eligible),
        "out_of_catalog_unique_pair_tokens": len(out_of_catalog),
        "noncanonical_unique_pair_tokens": len(noncanonical),
        "eligible_intersection_noncanonical_unique_pair_tokens": len(eligible & noncanonical),
    }


def validate_wave59_population_contract(
    config: dict[str, Any],
    population_counts: dict[str, dict[str, int]],
    *,
    recovery_context: dict[str, Any] | None,
) -> str:
    """Validate rows and the typed pair-token population before inference."""
    expected_rows = int(config["fresh_benchmark"]["expected_visible_fixtures_per_split"])
    expected_tokens = int(
        config["fresh_benchmark"]["expected_eligible_pair_tokens_per_split"]
    )
    if any(counts["rows"] != expected_rows for counts in population_counts.values()):
        raise RuntimeError("fresh benchmark sealed row count differs from prospective freeze")
    successor_count_basis = None
    if config.get("schema_version") == WAVE59_CONFIG_SCHEMA:
        from geometria_proporcional.wave59_hgb_guard_bracket import (
            SUCCESSOR_PAIR_TOKEN_COUNT_BASIS,
            is_successor_config,
        )

        if is_successor_config(config):
            successor_count_basis = config["fresh_benchmark"].get(
                "pair_token_count_basis"
            )
            if successor_count_basis != SUCCESSOR_PAIR_TOKEN_COUNT_BASIS:
                raise RuntimeError("Wave 59 successor pair-token basis drifted")
    elif config.get("schema_version") == WAVE60_CONFIG_SCHEMA:
        successor_count_basis = config["fresh_benchmark"].get("pair_token_count_basis")
        if successor_count_basis != "eligible_unique_pair_tokens":
            raise RuntimeError("Wave 60 pair-token basis drifted")
    count_field = (
        "eligible_unique_pair_tokens"
        if recovery_context is not None or successor_count_basis is not None
        else "total_unique_pair_tokens"
    )
    if any(counts[count_field] != expected_tokens for counts in population_counts.values()):
        qualifier = "eligible " if count_field == "eligible_unique_pair_tokens" else ""
        raise RuntimeError(
            f"fresh benchmark {qualifier}pair-token count differs from prospective freeze"
        )
    return count_field


def require_hash(path: Path, expected: str) -> dict[str, Any]:
    actual = digest(path)
    if actual != expected:
        raise RuntimeError(f"hash mismatch for {path}: {actual} != {expected}")
    return {"path": str(path.resolve()), "sha256": actual, "bytes": path.stat().st_size}


def require_sources_at_head(relative_paths: list[str]) -> tuple[str, dict[str, str]]:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    hashes: dict[str, str] = {}
    for relative in relative_paths:
        path = (REPO_ROOT / relative).resolve(strict=True)
        if path.relative_to(REPO_ROOT) != Path(relative):
            raise RuntimeError(f"non-canonical execution source path: {relative}")
        subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", relative],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
        )
        changed = subprocess.check_output(
            ["git", "status", "--porcelain", "--", relative], cwd=REPO_ROOT, text=True
        ).strip()
        if changed:
            raise RuntimeError(f"execution source differs from HEAD: {relative}")
        hashes[relative] = digest(path)
    return commit, hashes


def validate_prospective_config(config: dict[str, Any]) -> None:
    """Fail before escrow if the canonical prospective contract is incomplete or drifted."""
    schema = config.get("schema_version")
    supported = {
        "wave56-contextual-residual-gate-stage1-v1",
        "wave57-contextual-harm-guard-v1",
        WAVE59_CONFIG_SCHEMA,
        WAVE60_CONFIG_SCHEMA,
    }
    if schema not in supported:
        raise RuntimeError("prospective schema version drifted")
    if schema == WAVE59_CONFIG_SCHEMA:
        from geometria_proporcional.wave59_hgb_guard_bracket import (
            SUCCESSOR_PAIR_TOKEN_COUNT_BASIS,
            is_successor_config,
            validate_pre_draw_config,
        )

        validate_pre_draw_config(config)
        if config.get("status") != "FROZEN_PROSPECTIVE_PROTOCOL_PRE_KEY_DRAW":
            raise RuntimeError("Wave 59 config is not frozen for a pre-key draw")
        sources = config.get("required_execution_sources")
        if not isinstance(sources, list) or not sources or len(sources) != len(set(sources)):
            raise RuntimeError("Wave 59 execution-source manifest is incomplete")
        if config.get("physical_splits") != {
            "train": "gate_fit",
            "val": "gate_select",
            "lockbox": "sealed_monitor",
        }:
            raise RuntimeError("Wave 59 physical split roles drifted")
        if config.get("seeds") != [17, 29, 43]:
            raise RuntimeError("Wave 59 inference seeds drifted")
        if int(config.get("inference_batch_size", -1)) != 256:
            raise RuntimeError("Wave 59 inference batch size drifted")
        expected_fresh = {
            "protocol": "wave49-relational-benchmark-v2",
            "expected_visible_fixtures_per_split": 4992,
            "expected_eligible_pair_tokens_per_split": 768,
            "no_redraw_after_escrow": True,
            "sealed_directory_mode": "0700",
            "escrow_file_mode": "0600",
            "inference_uid": 65534,
            "inference_gid": 65534,
            "inference_user": "nobody",
            "staging_parent": "/tmp",
        }
        if is_successor_config(config):
            expected_fresh["pair_token_count_basis"] = (
                SUCCESSOR_PAIR_TOKEN_COUNT_BASIS
            )
        if config.get("fresh_benchmark", {}) != expected_fresh:
            raise RuntimeError("Wave 59 fresh benchmark contract drifted")
        if not isinstance(config.get("source_binding"), dict):
            raise RuntimeError("Wave 59 source binding is absent")
        return
    if schema == WAVE60_CONFIG_SCHEMA:
        from geometria_proporcional.wave60_frozen_policy_transport import (
            validate_pre_draw_config as validate_wave60_pre_draw_config,
        )

        validate_wave60_pre_draw_config(config)
        expected_fresh = {
            "protocol": "wave49-relational-benchmark-v2",
            "expected_visible_fixtures_per_split": 4992,
            "expected_eligible_pair_tokens_per_split": 768,
            "pair_token_count_basis": "eligible_unique_pair_tokens",
            "no_redraw_after_escrow": True,
            "sealed_directory_mode": "0700",
            "escrow_file_mode": "0600",
            "inference_uid": 65534,
            "inference_gid": 65534,
            "inference_user": "nobody",
            "staging_parent": "/tmp",
        }
        if config.get("fresh_benchmark") != expected_fresh:
            raise RuntimeError("Wave 60 fresh benchmark contract drifted")
        if config.get("seeds") != [17, 29, 43] or int(
            config.get("inference_batch_size", -1)
        ) != 256:
            raise RuntimeError("Wave 60 inference contract drifted")
        return
    if config.get("status") != "FROZEN_PROSPECTIVE_PROTOCOL_PRE_KEY_DRAW":
        raise RuntimeError("prospective config is not frozen for a pre-key draw")
    if config.get("device") != "cpu" or config.get("seeds") != [17, 29, 43]:
        raise RuntimeError("prospective device or inference seeds drifted")
    if tuple(config.get("feature_names", ())) != FEATURE_NAMES:
        raise RuntimeError("prospective feature schema differs from frozen primitives")
    if config.get("physical_splits") != {
        "train": "gate_fit",
        "val": "gate_select",
        "lockbox": "sealed_monitor",
    }:
        raise RuntimeError("prospective physical split roles drifted")
    fresh = config.get("fresh_benchmark", {})
    expected_fresh = {
        "protocol": "wave49-relational-benchmark-v2",
        "expected_visible_fixtures_per_split": 4992,
        "expected_eligible_pair_tokens_per_split": 768,
        "no_redraw_after_escrow": True,
        "sealed_directory_mode": "0700",
        "escrow_file_mode": "0600",
        "inference_uid": 65534,
        "inference_gid": 65534,
        "inference_user": "nobody",
        "staging_parent": "/tmp",
    }
    if fresh != expected_fresh:
        raise RuntimeError("fresh benchmark contract drifted")
    absent = config.get("absent_support", {})
    if absent != {
        "source": "wave54_calibration_fit_unseen_sets",
        "set_indices": [0, 4, 8, 10, 12],
    }:
        raise RuntimeError("absent-support contract drifted")
    absent_key = (
        "absent_support_tokens_per_set"
        if schema == "wave57-contextual-harm-guard-v1"
        else "absent_support_tokens"
    )
    if int(config.get("minimums", {}).get(absent_key, -1)) != 30:
        raise RuntimeError("absent-support minimum drifted")
    sources = config.get("required_execution_sources")
    if not isinstance(sources, list) or not sources or len(sources) != len(set(sources)):
        raise RuntimeError("execution-source manifest is absent or contains duplicates")
    if schema == "wave57-contextual-harm-guard-v1":
        from geometria_proporcional.wave57_tail_guard import (
            EXPECTED_SKLEARN_VERSION,
            HARM_MODEL_CONTRACT,
            validate_wave57_frozen_config,
        )

        validate_wave57_frozen_config(config)
        harm = dict(config.get("harm_model", {}))
        observed_contract = {
            key: harm.get(key) for key in HARM_MODEL_CONTRACT
        }
        if observed_contract != HARM_MODEL_CONTRACT:
            raise RuntimeError("Wave 57 harm-model contract drifted")
        if harm.get("positive_class") != "gain_lt_negative_1e-12":
            raise RuntimeError("Wave 57 positive class drifted")
        if harm.get("sklearn_version") != EXPECTED_SKLEARN_VERSION:
            raise RuntimeError("Wave 57 scikit-learn contract drifted")
        required_criteria = {
            "regret_reduction_vs_hard_min",
            "regret_vs_hard_ci95_upper_below",
            "accuracy_vs_hard_ci95_lower_at_least",
            "compatibility_vs_hard_ci95_lower_at_least",
            "worst_regret_vs_hard_mean_at_most",
            "worst_regret_vs_hard_ci95_upper_at_most",
            "accuracy_vs_proposer_ci95_lower_above",
            "worst_regret_vs_proposer_ci95_upper_below",
            "regret_vs_proposer_ci95_upper_at_most",
            "regret_vs_shuffled_ci95_upper_below",
            "worst_regret_vs_shuffled_ci95_upper_below",
            "shard_nonidentity_and_sign_stability_required",
            "replay_exact_required",
        }
    else:
        required_criteria = {
            "regret_reduction_vs_hard_min",
            "regret_vs_hard_ci95_upper_below",
            "accuracy_vs_hard_ci95_lower_at_least",
            "compatibility_vs_hard_ci95_lower_at_least",
            "regret_reduction_vs_scalar_min",
            "regret_vs_scalar_ci95_upper_below",
            "regret_reduction_vs_advantage_only_min",
            "regret_vs_advantage_only_ci95_upper_below",
            "regret_reduction_vs_shuffled_min",
            "regret_vs_shuffled_ci95_upper_below",
            "accuracy_vs_pure_joint_ci95_lower_above",
            "regret_vs_pure_joint_ci95_upper_at_most",
            "selector_sensitive_required",
            "replay_exact_required",
        }
    if set(config.get("diagnostic_criteria", {})) != required_criteria:
        raise RuntimeError("diagnostic criteria are incomplete or contain undeclared keys")


def has_escrow(path: Path) -> bool:
    return (path / ESCROW_NAME).is_file()


def archived_attempts(output: Path, primary_name: str) -> list[Path]:
    return sorted(output.parent.glob(f"{primary_name}.failed_*")) + sorted(
        output.parent.glob(f"{primary_name}.superseded_*")
    )


def validate_invocation(
    args: argparse.Namespace,
    output: Path,
    config: dict[str, Any],
    *,
    repo_root: Path = REPO_ROOT,
) -> str:
    """Enforce one primary escrow; replay and recovery may never redraw."""
    primary_name = str(config["primary_output_name"])
    replay_name = str(config["replay_output_name"])
    canonical_parent = (repo_root / config["output_parent_relative"]).resolve()
    wave60_initialized = False
    if output.exists() and config.get("schema_version") == WAVE60_CONFIG_SCHEMA:
        wave60_initialized = {
            str(path.relative_to(output))
            for path in output.rglob("*")
            if path.is_file()
        } == {"config.snapshot.json", "source_bindings.json"}
    if output.parent != canonical_parent:
        raise ValueError(f"prospective outputs must live directly under {canonical_parent}")
    if output.name not in {primary_name, replay_name}:
        raise ValueError(f"output name must be {primary_name!r} or {replay_name!r}")
    if args.replay_secrets_from and args.recovery_secrets_from:
        raise ValueError("replay and recovery modes are mutually exclusive")
    amendment_arg = getattr(args, "recovery_amendment", None)
    if amendment_arg and not (args.replay_secrets_from or args.recovery_secrets_from):
        raise ValueError("recovery amendment cannot authorize a fresh primary")

    if output.name == replay_name:
        if not args.replay_secrets_from or not args.reference_dir:
            raise ValueError("replay requires --replay-secrets-from and --reference-dir")
        source = args.replay_secrets_from.resolve(strict=True)
        reference = args.reference_dir.resolve(strict=True)
        if source != reference or source == output or source.name != primary_name:
            raise ValueError("replay must use the distinct canonical primary as reference/key source")
        if not has_escrow(source):
            raise RuntimeError("replay source lacks a durable escrow")
        if output.exists() and not args.force and not wave60_initialized:
            raise FileExistsError("existing replay requires --force archival")
        return "replay"

    if args.replay_secrets_from or args.reference_dir:
        raise ValueError("primary output cannot use replay arguments")
    prior = archived_attempts(output, primary_name)
    key_bearing = [path for path in prior if has_escrow(path)]
    if args.recovery_secrets_from:
        source = args.recovery_secrets_from.resolve(strict=True)
        allowed = [path.resolve() for path in prior]
        if config.get("schema_version") == WAVE60_CONFIG_SCHEMA:
            recovery = config.get("attempt", {}).get("recovery")
            if not isinstance(recovery, dict):
                raise ValueError("Wave 60 recovery requires a versioned authority")
            prior_primary = (
                repo_root
                / recovery["prior_attempt_container"]
                / config["primary_output_name"]
            ).resolve(strict=True)
            allowed.append(prior_primary)
        if source != output and source not in allowed:
            raise ValueError("recovery escrow must come from this primary or one of its archives")
        if not has_escrow(source):
            raise RuntimeError("recovery source lacks a durable escrow")
        if output.exists() and not args.force and not wave60_initialized:
            raise ValueError("recovering an existing primary requires --force archival")
        return "recovery"
    if output.exists() and not wave60_initialized:
        raise FileExistsError("the unique primary already exists; fresh redraw is forbidden")
    if key_bearing:
        raise RuntimeError("a prior primary escrow exists; use --recovery-secrets-from")
    return "primary"


def token_logits(examples: list[dict[str, Any]], logits: np.ndarray) -> tuple[list[str], np.ndarray]:
    grouped: dict[str, list[np.ndarray]] = defaultdict(list)
    for example, row in zip(examples, logits, strict=True):
        grouped[str(example["pair_token"])].append(np.asarray(row, dtype=np.float64))
    tokens = sorted(grouped)
    return tokens, np.stack([np.mean(grouped[token], axis=0) for token in tokens])


def historical_preflight(
    wave50: Path,
    wave51: Path,
    wave52: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Re-forward the frozen historical monitor exactly before any output exists."""
    torch.set_num_threads(int(config["cpu_threads"]))
    torch.use_deterministic_algorithms(True)
    binding = config["source_binding"]
    inputs = [
        require_hash(wave50 / "benchmark/visible/val.jsonl", binding["wave50_visible_val_sha256"]),
        require_hash(wave50 / "authorized_labels/val.jsonl", binding["wave50_authorized_val_sha256"]),
        require_hash(wave50 / "benchmark/protocol_config.json", binding["wave50_protocol_sha256"]),
        require_hash(wave51 / "normalizer.npz", binding["wave51_normalizer_sha256"]),
        require_hash(wave51 / "split_manifest.json", binding["wave51_split_manifest_sha256"]),
    ]
    records, _ = load_labeled_records(
        wave50 / "benchmark/visible/val.jsonl",
        wave50 / "authorized_labels/val.jsonl",
        wave50 / "benchmark/protocol_config.json",
        "val",
    )
    records = stratified_token_subset(records, 3072, 5102)
    _, monitor = split_tokens(records, 0.5, 5102)
    manifest = json.loads((wave51 / "split_manifest.json").read_text(encoding="utf-8"))
    actual_tokens = sorted({str(row["pair_token"]) for row in monitor})
    if actual_tokens != manifest["val_monitor"]:
        raise RuntimeError("historical val-monitor split does not reproduce Wave 51")
    with np.load(wave51 / "normalizer.npz", allow_pickle=False) as data:
        normalizer = FeatureNormalizer(data["mean"], data["std"])
    examples = prepare_examples(monitor, normalizer)
    checks: list[dict[str, Any]] = []
    for seed in config["seeds"]:
        checkpoint_path = wave51 / "checkpoints" / f"seed{seed}__sigmoid_only.pt"
        inputs.append(require_hash(checkpoint_path, binding["wave51_checkpoints_sha256"][str(seed)]))
        reference_path = wave52 / "raw_eval/frozen_set" / f"seed{seed}__val_monitor.npz"
        inputs.append(require_hash(reference_path, binding["wave52_val_monitor_sha256"][str(seed)]))
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model = DualHeadDeepSet()
        model.load_state_dict(checkpoint["model_state"])
        set_logits, choice_logits = predict_dual_logits(
            model, examples, int(config["inference_batch_size"])
        )
        tokens, token_set = token_logits(examples, set_logits)
        choice_tokens, token_choice = token_logits(examples, choice_logits)
        target_by_token = {
            str(row["pair_token"]): np.asarray(row["target"], dtype=np.float32) for row in monitor
        }
        token_target = np.stack([target_by_token[token] for token in tokens])
        with np.load(reference_path, allow_pickle=False) as reference:
            exact = {
                "pair_token": np.array_equal(np.asarray(tokens), reference["pair_token"]),
                "choice_pair_token": tokens == choice_tokens,
                "target": np.array_equal(token_target, reference["target"]),
                "set_logits": np.array_equal(token_set, reference["set_logits"]),
                "choice_logits": np.array_equal(token_choice, reference["choice_logits"]),
            }
        if not all(exact.values()):
            raise RuntimeError(f"Wave 52 historical re-forward mismatch for seed {seed}: {exact}")
        checks.append({"seed": seed, "array_exact": exact})
    return {"status": "PASS", "checks": checks, "n_tokens": len(actual_tokens), "inputs": inputs}


def preparation_preflight(args: argparse.Namespace, config_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Validate every source and historical invariant without creating output."""
    if os.geteuid() != 0:
        raise PermissionError("fresh preparation requires root to enforce the sealed boundary")
    nobody = pwd.getpwnam(config["fresh_benchmark"]["inference_user"])
    nogroup = grp.getgrgid(int(config["fresh_benchmark"]["inference_gid"]))
    if nobody.pw_uid != int(config["fresh_benchmark"]["inference_uid"]) or nogroup.gr_gid != nobody.pw_gid:
        raise RuntimeError("frozen nobody/nogroup identity does not match this host")
    if shutil.which("setpriv") is None:
        raise RuntimeError("setpriv is required for the inference boundary")
    validate_prospective_config(config)
    config_relative = str(config_path.relative_to(REPO_ROOT))
    if config_relative not in config.get("required_execution_sources", []):
        raise ValueError("fresh run config is not its canonical bound execution source")
    if config.get("schema_version") == WAVE59_CONFIG_SCHEMA:
        validate_wave59_successor_final_authority(config, config_path)
    if args.attestation_private_key.is_symlink():
        raise RuntimeError("attestation private key cannot be a symlink")
    private_key = args.attestation_private_key.resolve(strict=True)
    if not private_key.is_file():
        raise RuntimeError("attestation private key must be one existing regular file")

    commit, source_hashes = require_sources_at_head(config["required_execution_sources"])
    if config.get("schema_version") == WAVE59_CONFIG_SCHEMA:
        from geometria_proporcional.wave59_hgb_guard_bracket import (
            config_self_binding_sha256,
        )

        expected_sources = config["source_sha256"]
        config_key = str(config_path.relative_to(REPO_ROOT))
        observed_bound = dict(source_hashes)
        observed_bound[config_key] = config_self_binding_sha256(config, config_key)
        if observed_bound != expected_sources:
            amendment_path = getattr(args, "recovery_amendment", None)
            if amendment_path is None or not (
                getattr(args, "recovery_secrets_from", None)
                or getattr(args, "replay_secrets_from", None)
            ):
                differing = sorted(
                    key
                    for key in set(observed_bound) | set(expected_sources)
                    if observed_bound.get(key) != expected_sources.get(key)
                )
                raise RuntimeError(f"Wave 59 execution source binding drifted: {differing}")
            validate_wave59_repository_recovery_authority(
                amendment_path,
                expected_sources,
                observed_bound,
                repo_root=REPO_ROOT,
            )
        implementation_commit = config["implementation_binding"]["commit"]
        require_ancestor(REPO_ROOT, implementation_commit, commit)
        audit_path = REPO_ROOT / config["implementation_binding"]["audit_path"]
        if digest(audit_path) != config["implementation_binding"]["audit_sha256"]:
            raise RuntimeError("Wave 59 implementation audit hash drifted")
        audit_text = audit_path.read_text(encoding="utf-8")
        if (
            implementation_commit not in audit_text
            or "## Dictamen: PASS" not in audit_text
        ):
            raise RuntimeError(
                "Wave 59 implementation audit does not accept the bound commit"
            )
    elif config.get("schema_version") == WAVE60_CONFIG_SCHEMA:
        from geometria_proporcional.wave60_frozen_policy_transport import (
            config_self_binding_sha256,
        )

        expected_sources = config["source_sha256"]
        config_key = str(config_path.relative_to(REPO_ROOT))
        observed_bound = dict(source_hashes)
        observed_bound[config_key] = config_self_binding_sha256(config, config_key)
        if observed_bound != expected_sources:
            differing = sorted(
                key
                for key in set(observed_bound) | set(expected_sources)
                if observed_bound.get(key) != expected_sources.get(key)
            )
            raise RuntimeError(f"Wave 60 execution source binding drifted: {differing}")
        implementation = config["implementation_binding"]
        require_ancestor(REPO_ROOT, implementation["commit"], commit)
        require_ancestor(REPO_ROOT, implementation["audit_commit"], commit)
        validate_wave60_audit_commit(
            REPO_ROOT,
            implementation,
            scope="IMPLEMENTATION",
            target={"implementation_commit": implementation["commit"]},
            expected_parent=implementation["commit"],
        )
        source_authority = config["source_law_authority"]
        require_ancestor(REPO_ROOT, source_authority["audit_commit"], commit)
        validate_wave60_audit_commit(
            REPO_ROOT,
            source_authority,
            scope="SOURCE_LAW",
            target={
                "source_authority_manifest_sha256": source_authority[
                    "source_authority_manifest_sha256"
                ]
            },
            expected_parent=implementation["audit_commit"],
        )
        authority_path = (REPO_ROOT / source_authority["path"]).resolve(strict=True)
        if digest(authority_path / "source_authority_manifest.json") != source_authority[
            "source_authority_manifest_sha256"
        ]:
            raise RuntimeError("Wave 60 source-law authority manifest drifted")
        validate_wave60_final_config_authority(
            REPO_ROOT, config_path, config, commit
        )
    binding = config["source_binding"]
    wave50 = args.wave50_dir.resolve(strict=True)
    wave51 = args.wave51_dir.resolve(strict=True)
    wave52 = args.wave52_dir.resolve(strict=True)
    wave54 = args.wave54_dir.resolve(strict=True)
    wave55 = args.wave55_dir.resolve(strict=True)
    stage0 = args.stage0_dir.resolve(strict=True)
    wave56 = args.wave56_dir.resolve(strict=True) if args.wave56_dir else None
    upstream = [
        require_hash(PUBLIC_KEY, binding["wave49_attestation_public_key_sha256"]),
        require_hash(wave52 / "policy_manifest.json", binding["wave52_policy_manifest_sha256"]),
        require_hash(wave54 / "selection_freeze.json", binding["wave54_selection_freeze_sha256"]),
        require_hash(wave54 / "posterior_state.npz", binding["wave54_posterior_state_sha256"]),
        require_hash(wave54 / "summary.json", binding["wave54_summary_sha256"]),
        require_hash(wave55 / "bundles/decision_select.npz", binding["wave55_decision_select_sha256"]),
        require_hash(wave55 / "bundles/sealed_monitor.npz", binding["wave55_sealed_monitor_sha256"]),
        require_hash(stage0 / "selection_freeze.json", binding["wave56_stage0_selection_freeze_sha256"]),
        require_hash(stage0 / "analysis_core.json", binding["wave56_stage0_analysis_core_sha256"]),
    ]
    schema = config.get("schema_version")
    if schema == "wave57-contextual-harm-guard-v1":
        if wave56 is None:
            raise ValueError("Wave 57 preparation requires --wave56-dir")
        upstream.extend(
            [
                require_hash(
                    wave56 / "phases/adjudicate.complete/analytics.complete/result_arrays.npz",
                    binding["wave56_stage1_result_arrays_sha256"],
                ),
                require_hash(
                    wave56 / "phases/adjudicate.complete/analytics.complete/REPORT_WAVE56_STAGE1.json",
                    binding["wave56_stage1_report_sha256"],
                ),
            ]
        )
    selection = json.loads((stage0 / "selection_freeze.json").read_text(encoding="utf-8"))
    analysis = json.loads((stage0 / "analysis_core.json").read_text(encoding="utf-8"))
    if tuple(analysis.get("feature_names", ())) != FEATURE_NAMES:
        raise RuntimeError("Stage 0 feature schema differs from prospective freeze")
    if schema not in {WAVE59_CONFIG_SCHEMA, WAVE60_CONFIG_SCHEMA}:
        expected_model = config["primary_model"]
        if selection.get("selected_family") != expected_model["family"]:
            raise RuntimeError("Stage 0 selected family differs from prospective freeze")
        if float(selection.get("selected_params", {}).get("alpha")) != float(expected_model["alpha"]):
            raise RuntimeError("Stage 0 selected alpha differs from prospective freeze")
        wave54_summary = json.loads((wave54 / "summary.json").read_text(encoding="utf-8"))
        historical_absent = wave54_summary.get("unseen_mass", {}).get("unseen_set_indices")
        frozen_absent = config.get("absent_support", {}).get("set_indices")
        if historical_absent != frozen_absent:
            raise RuntimeError(
                "prospective absent-support indices differ from the bound Wave 54 summary"
            )
    historical = historical_preflight(wave50, wave51, wave52, config)
    return {
        "git_commit": commit,
        "config_sha256": digest(config_path),
        "prospective_config": config,
        "sources": source_hashes,
        "upstream": upstream,
        "historical_preflight": historical,
        "source_bindings": binding,
    }


def read_escrow(path: Path) -> dict[str, Any]:
    escrow_path = path / ESCROW_NAME
    escrow_stat = escrow_path.lstat()
    if stat.S_ISLNK(escrow_stat.st_mode) or not stat.S_ISREG(escrow_stat.st_mode):
        raise PermissionError("escrow must be one physical regular file")
    if stat.S_IMODE(escrow_stat.st_mode) != 0o600 or escrow_stat.st_uid != 0:
        raise PermissionError("escrow must remain root-owned mode 0600")
    payload = json.loads(escrow_path.read_text(encoding="utf-8"))
    keys = payload.get("keys", {})
    if set(keys) != set(SECRET_FILES):
        raise RuntimeError("escrow key set is incomplete")
    values = [bytes.fromhex(keys[name]) for name in SECRET_FILES]
    if any(len(value) != 32 for value in values) or len(set(values)) != 3:
        raise RuntimeError("escrow requires three distinct 32-byte keys")
    expected = {name: sha256_bytes(value) for name, value in zip(SECRET_FILES, values, strict=True)}
    if payload.get("key_commitments") != expected:
        raise RuntimeError("escrow commitments do not verify")
    return payload


def public_freeze_from_escrow(escrow: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": escrow["schema_version"],
        "phase": "keys-escrowed-and-contract-frozen-before-generation",
        "contract": escrow["contract"],
        "key_commitments": escrow["key_commitments"],
        "contains_secrets": False,
        "generator_invoked": False,
    }


def verify_escrow_and_freeze(output: Path, expected_escrow: dict[str, Any]) -> None:
    actual_escrow = read_escrow(output)
    if actual_escrow != expected_escrow:
        raise RuntimeError("published escrow differs from in-memory payload")
    if digest(output / ESCROW_NAME) != canonical_json_sha256(expected_escrow):
        raise RuntimeError("published escrow hash differs from canonical payload hash")
    freeze_path = output / FREEZE_NAME
    freeze_stat = freeze_path.stat()
    if stat.S_IMODE(freeze_stat.st_mode) != 0o644 or freeze_stat.st_uid != 0:
        raise PermissionError("public pre-generation freeze must be root-owned mode 0644")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze != public_freeze_from_escrow(actual_escrow):
        raise RuntimeError("public freeze is not the secret-free projection of escrow")
    if digest(freeze_path) != canonical_json_sha256(freeze):
        raise RuntimeError("public freeze hash differs from canonical payload hash")


def keys_from_escrow(escrow: dict[str, Any]) -> tuple[bytes, bytes, bytes]:
    return tuple(bytes.fromhex(escrow["keys"][name]) for name in SECRET_FILES)  # type: ignore[return-value]


def make_escrow(contract: dict[str, Any], keys: tuple[bytes, bytes, bytes]) -> dict[str, Any]:
    key_hex = {name: value.hex() for name, value in zip(SECRET_FILES, keys, strict=True)}
    commitments = {name: sha256_bytes(value) for name, value in zip(SECRET_FILES, keys, strict=True)}
    return {
        "schema_version": "wave56-key-escrow-v1",
        "phase": "durable-key-escrow-before-generation",
        "contract": contract,
        "keys": key_hex,
        "key_commitments": commitments,
    }


def _require_keys(payload: dict[str, Any], expected: set[str], label: str) -> None:
    if set(payload) != expected:
        raise RuntimeError(f"{label} keys differ from the recovery schema")


def _require_report_fields(
    path: Path,
    fields: list[str],
    label: str,
    *,
    allow_one_terminal_blank: bool = False,
) -> None:
    payload = path.read_bytes()
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeError(f"{label} is not canonical UTF-8") from exc
    if (
        allow_one_terminal_blank
        and text.endswith("\n\n")
        and not text.endswith("\n\n\n")
    ):
        text = text[:-1]
    invalid_separator = any(
        (ord(character) < 32 and character not in {"\n", "\t"})
        or character in {"\x7f", "\x85", "\u2028", "\u2029"}
        for character in text
    )
    lines = text.split("\n")
    header_end = 2 + len(fields)
    result_fields = [field for field in fields if field.startswith("**Result:** `")]
    expected_result = (
        result_fields[0].removeprefix("**Result:** `").removesuffix("`")
        if len(result_fields) == 1
        else None
    )
    final_block = [
        "## Machine-verifiable decision",
        "",
        f"**Final decision:** `{expected_result}`",
        "",
    ]
    invalid_layout = (
        invalid_separator
        or not text.endswith("\n")
        or len(lines) <= header_end + 1
        or not lines[0].startswith("# ")
        or any(token in lines[0] for token in ("<", ">", "```", "~~~"))
        or lines[1] != ""
        or lines[2:header_end] != fields
        or lines[header_end] != ""
        or not lines[header_end + 1].startswith("## ")
        or "<!--" in text
        or "-->" in text
        or "```" in text
        or "~~~" in text
        or expected_result not in {"PASS", "REVISE"}
        or lines[-4:] != final_block
    )
    if invalid_layout:
        raise RuntimeError(
            f"{label} does not contain one unique canonical attestation block"
        )
    for field in fields:
        prefix = field.partition("`")[0]
        matches = [
            line
            for line in lines
            if line.startswith(prefix)
        ]
        if matches != [field]:
            raise RuntimeError(
                f"{label} does not contain one unique canonical attestation block"
            )
    if [line for line in lines if line.startswith("**Final decision:** ")] != [
        final_block[2]
    ]:
        raise RuntimeError(
            f"{label} does not contain one unique canonical terminal decision"
        )


def _require_successor_pass_report(
    path: Path, fields: list[str], label: str
) -> None:
    """Require one canonical PASS block without contradictory dictamen headings."""
    _require_report_fields(path, fields, label)
    lines = path.read_text(encoding="utf-8").splitlines()
    dictamen = [line for line in lines if line.startswith("## Dictamen:")]
    if dictamen != ["## Dictamen: PASS"]:
        raise RuntimeError(f"{label} does not contain one coherent PASS dictamen")


def _require_unique_report_lines(
    path: Path, expected: dict[str, str], label: str
) -> None:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise RuntimeError(f"{label} is not canonical UTF-8") from exc
    for prefix, line in expected.items():
        matches = [candidate for candidate in lines if candidate.startswith(prefix)]
        if matches != [line]:
            raise RuntimeError(f"{label} field is absent, duplicated, or divergent: {prefix}")


def _wave59_successor_source_delta(
    config: dict[str, Any], config_path: Path, repo_root: Path
) -> tuple[dict[str, Any], dict[str, Any], str, str]:
    """Validate the closed successor delta and return its Git chain anchors."""
    from geometria_proporcional.wave59_hgb_guard_bracket import (
        LEGACY_CONFIG_SOURCE_RELATIVE,
        SUCCESSOR_IMPLEMENTATION_AUDIT_SOURCES,
        SUCCESSOR_CONFIG_SOURCE_RELATIVE,
        SUCCESSOR_PAIR_TOKEN_COUNT_BASIS,
        SUCCESSOR_R449_AUTHORITY,
        SUCCESSOR_R450_AUTHORITY,
        SUCCESSOR_REJECTED_IMPLEMENTATION_COMMIT,
        config_self_binding_sha256,
        config_source_relative,
    )

    if config_source_relative(config) != SUCCESSOR_CONFIG_SOURCE_RELATIVE:
        raise RuntimeError("Wave 59 successor validator received the legacy config")
    canonical_config = repo_root / SUCCESSOR_CONFIG_SOURCE_RELATIVE
    if (
        config_path.is_symlink()
        or config_path.absolute() != canonical_config.absolute()
        or config_path.resolve(strict=True) != canonical_config.resolve(strict=True)
    ):
        raise RuntimeError("Wave 59 successor config path is not canonical")

    authority = config.get("successor_authority")
    if not isinstance(authority, dict) or set(authority) != {
        "schema_version",
        "base_config",
        "plan",
        "plan_audit",
        "final_config_audit_path",
    }:
        raise RuntimeError("Wave 59 successor authority shape drifted")
    if authority["schema_version"] != "wave59-replay-normalized-successor-authority-v1":
        raise RuntimeError("Wave 59 successor authority schema drifted")
    base_record = authority["base_config"]
    if not isinstance(base_record, dict) or set(base_record) != {"path", "sha256"}:
        raise RuntimeError("Wave 59 successor base-config binding drifted")
    if base_record["path"] != LEGACY_CONFIG_SOURCE_RELATIVE:
        raise RuntimeError("Wave 59 successor base-config path drifted")
    base_path, _ = require_repo_artifact(
        repo_root, LEGACY_CONFIG_SOURCE_RELATIVE, base_record["sha256"]
    )
    base = json.loads(base_path.read_text(encoding="utf-8"))

    plan = authority["plan"]
    plan_audit = authority["plan_audit"]
    for record, label in ((plan, "plan"), (plan_audit, "plan audit")):
        if not isinstance(record, dict) or set(record) != {"commit", "path", "sha256"}:
            raise RuntimeError(f"Wave 59 successor {label} binding drifted")
    if plan["path"] != WAVE59_SUCCESSOR_PLAN_RELATIVE:
        raise RuntimeError("Wave 59 successor plan path drifted")
    require_repo_artifact(repo_root, plan["path"], plan["sha256"])
    if git_changed_paths(repo_root, plan["commit"]) != {plan["path"]}:
        raise RuntimeError("Wave 59 successor plan commit contains unrelated paths")
    if git_blob_sha256(repo_root, plan["commit"], plan["path"]) != plan["sha256"]:
        raise RuntimeError("Wave 59 successor plan blob drifted")
    require_audit_report_path(plan_audit["path"], "Wave 59 successor plan audit")
    plan_audit_path, _ = require_repo_artifact(
        repo_root, plan_audit["path"], plan_audit["sha256"]
    )
    plan_audit_commit = git_introduction_commit(repo_root, plan_audit["path"])
    if plan_audit_commit != plan_audit["commit"]:
        raise RuntimeError("Wave 59 successor plan-audit introduction drifted")
    if git_changed_paths(repo_root, plan_audit_commit) != {plan_audit["path"]}:
        raise RuntimeError("Wave 59 successor plan-audit commit contains unrelated paths")
    require_direct_parent(
        repo_root, plan_audit_commit, plan["commit"], "Wave 59 successor plan audit"
    )
    for historical, label in (
        (SUCCESSOR_R449_AUTHORITY, "R449"),
        (SUCCESSOR_R450_AUTHORITY, "R450"),
    ):
        historical_path, _ = require_repo_artifact(
            repo_root, historical["path"], historical["sha256"]
        )
        if (
            git_introduction_commit(repo_root, historical["path"])
            != historical["commit"]
        ):
            raise RuntimeError(f"Wave 59 successor {label} introduction drifted")
        if git_changed_paths(repo_root, historical["commit"]) != {
            historical["path"]
        }:
            raise RuntimeError(
                f"Wave 59 successor {label} commit contains unrelated paths"
            )
        if digest(historical_path) != historical["sha256"]:
            raise RuntimeError(f"Wave 59 successor {label} bytes drifted")
    require_direct_parent(
        repo_root,
        plan["commit"],
        SUCCESSOR_R450_AUTHORITY["commit"],
        "Wave 59 successor corrected plan",
    )
    _require_successor_pass_report(
        plan_audit_path,
        [
            f"**Plan commit:** `{plan['commit']}`  ",
            f"**Plan SHA-256:** `{plan['sha256']}`  ",
            f"**R449 commit:** `{SUCCESSOR_R449_AUTHORITY['commit']}`  ",
            f"**R449 report SHA-256:** `{SUCCESSOR_R449_AUTHORITY['sha256']}`  ",
            f"**R450 commit:** `{SUCCESSOR_R450_AUTHORITY['commit']}`  ",
            f"**R450 report SHA-256:** `{SUCCESSOR_R450_AUTHORITY['sha256']}`  ",
            "**Rejected implementation commit:** "
            f"`{SUCCESSOR_REJECTED_IMPLEMENTATION_COMMIT}`  ",
            "**Result:** `PASS`",
        ],
        "Wave 59 successor plan audit",
    )

    binding = config.get("implementation_binding")
    if not isinstance(binding, dict) or set(binding) != {
        "status", "commit", "audit_path", "audit_sha256"
    }:
        raise RuntimeError("Wave 59 successor implementation binding drifted")
    implementation_commit = binding["commit"]
    if binding["status"] != "ACCEPTED_IMPLEMENTATION_AUDIT":
        raise RuntimeError("Wave 59 successor implementation is not accepted")
    require_direct_parent(
        repo_root,
        implementation_commit,
        plan_audit_commit,
        "Wave 59 successor implementation",
    )
    if git_changed_paths(repo_root, implementation_commit) != set(
        WAVE59_SUCCESSOR_IMPLEMENTATION_PATHS
    ):
        raise RuntimeError("Wave 59 successor implementation paths drifted")
    require_audit_report_path(binding["audit_path"], "Wave 59 successor implementation audit")
    implementation_audit_path, _ = require_repo_artifact(
        repo_root, binding["audit_path"], binding["audit_sha256"]
    )
    implementation_audit_commit = git_introduction_commit(
        repo_root, binding["audit_path"]
    )
    if git_changed_paths(repo_root, implementation_audit_commit) != {binding["audit_path"]}:
        raise RuntimeError("Wave 59 successor implementation-audit paths drifted")
    require_direct_parent(
        repo_root,
        implementation_audit_commit,
        implementation_commit,
        "Wave 59 successor implementation audit",
    )
    implementation_fields = [
        f"**Implementation commit:** `{implementation_commit}`  "
    ]
    for field_label, relative in SUCCESSOR_IMPLEMENTATION_AUDIT_SOURCES:
        implementation_fields.append(
            f"**{field_label}:** "
            f"`{git_blob_sha256(repo_root, implementation_commit, relative)}`  "
        )
    implementation_fields.append("**Result:** `PASS`")
    _require_successor_pass_report(
        implementation_audit_path,
        implementation_fields,
        "Wave 59 successor implementation audit",
    )

    expected_required = []
    old_audit = base["implementation_binding"]["audit_path"]
    for relative in base["required_execution_sources"]:
        if relative == LEGACY_CONFIG_SOURCE_RELATIVE:
            expected_required.append(SUCCESSOR_CONFIG_SOURCE_RELATIVE)
        elif relative == old_audit:
            expected_required.append(binding["audit_path"])
        else:
            expected_required.append(relative)
    expected_required.extend(
        [plan["path"], plan_audit["path"], WAVE59_RECOVERY_TEST_RELATIVE]
    )
    if config.get("required_execution_sources") != expected_required:
        raise RuntimeError("Wave 59 successor execution-source list drifted")
    hashes = config.get("source_sha256")
    if not isinstance(hashes, dict) or set(hashes) != set(expected_required):
        raise RuntimeError("Wave 59 successor execution-source map drifted")

    normalized_base = deepcopy(base)
    normalized_successor = deepcopy(config)
    for payload in (normalized_base, normalized_successor):
        for key in (
            "primary_output", "primary_output_name", "replay_output",
            "replay_output_name", "implementation_binding",
            "required_execution_sources", "source_sha256",
        ):
            payload.pop(key, None)
    normalized_successor.pop("successor_authority", None)
    if normalized_successor.get("fresh_benchmark", {}).pop(
        "pair_token_count_basis", None
    ) != SUCCESSOR_PAIR_TOKEN_COUNT_BASIS:
        raise RuntimeError("Wave 59 successor pair-token basis drifted")
    if normalized_successor != normalized_base:
        raise RuntimeError("Wave 59 successor changed the frozen scientific config")
    if {
        "primary_output": config.get("primary_output"),
        "primary_output_name": config.get("primary_output_name"),
        "replay_output": config.get("replay_output"),
        "replay_output_name": config.get("replay_output_name"),
    } != {
        "primary_output": "data/geometria_proporcional/wave59_fresh_hgb_guard_bracket_replay_normalized_v1",
        "primary_output_name": "wave59_fresh_hgb_guard_bracket_replay_normalized_v1",
        "replay_output": "data/geometria_proporcional/wave59_fresh_hgb_guard_bracket_replay_normalized_v1_replay",
        "replay_output_name": "wave59_fresh_hgb_guard_bracket_replay_normalized_v1_replay",
    }:
        raise RuntimeError("Wave 59 successor output identity drifted")

    existing_implementation = set(WAVE59_SUCCESSOR_IMPLEMENTATION_PATHS) - {
        WAVE59_RECOVERY_TEST_RELATIVE
    }
    removed = {LEGACY_CONFIG_SOURCE_RELATIVE, old_audit}
    changed_existing = existing_implementation
    # Diagnose mutations to invariant sources before checking whether a later
    # wave legitimately changed a shared implementation path in the worktree.
    # The physical implementation binding is still enforced immediately below.
    for relative in set(base["required_execution_sources"]) - removed - changed_existing:
        if hashes.get(relative) != base["source_sha256"][relative]:
            raise RuntimeError(
                f"Wave 59 successor changed an unauthorized source hash: {relative}"
            )
    for relative in existing_implementation:
        if base["source_sha256"].get(relative) is None:
            raise RuntimeError("Wave 59 base config lacks an implementation source")
        expected_blob = git_blob_sha256(repo_root, implementation_commit, relative)
        if hashes.get(relative) != expected_blob or digest(repo_root / relative) != expected_blob:
            raise RuntimeError(f"Wave 59 successor implementation hash drifted: {relative}")
    recovery_test_blob = git_blob_sha256(
        repo_root, implementation_commit, WAVE59_RECOVERY_TEST_RELATIVE
    )
    if (
        hashes.get(WAVE59_RECOVERY_TEST_RELATIVE) != recovery_test_blob
        or digest(repo_root / WAVE59_RECOVERY_TEST_RELATIVE) != recovery_test_blob
    ):
        raise RuntimeError("Wave 59 successor recovery-test hash drifted")
    for relative in (plan["path"], plan_audit["path"], binding["audit_path"]):
        if hashes.get(relative) != digest(repo_root / relative):
            raise RuntimeError(f"Wave 59 successor authority hash drifted: {relative}")
    if hashes.get(SUCCESSOR_CONFIG_SOURCE_RELATIVE) != config_self_binding_sha256(
        config, SUCCESSOR_CONFIG_SOURCE_RELATIVE
    ):
        raise RuntimeError("Wave 59 successor config self-binding drifted")
    return base, authority, implementation_commit, implementation_audit_commit


def _validate_wave59_antecedent_sentinels(repo_root: Path) -> None:
    primary = repo_root / WAVE59_ANTECEDENT_PRIMARY_RELATIVE
    failure = repo_root / WAVE59_ANTECEDENT_REPLAY_FAILURE_RELATIVE
    for relative, expected in WAVE59_ANTECEDENT_SENTINELS.items():
        root = failure if relative.startswith("failure_") or relative == "FAILURE.json" else primary
        if digest(root / relative) != expected:
            raise RuntimeError(f"Wave 59 antecedent sentinel drifted: {relative}")
    analysis = json.loads((primary / "analysis.json").read_text(encoding="utf-8"))
    if analysis.get("scientific_decision") is not None:
        raise RuntimeError("Wave 59 antecedent acquired a scientific decision")
    for label in ("harm", "incompatibility"):
        pattern = analysis.get("prospective_patterns", {}).get(label, {})
        if pattern.get("replay_exact") != "PENDING" or pattern.get("aggregate_with_replay") is not None:
            raise RuntimeError(f"Wave 59 antecedent replay state drifted: {label}")
    failure_record = json.loads((failure / "FAILURE.json").read_text(encoding="utf-8"))
    if {
        "run_role": failure_record.get("run_role"),
        "last_state": failure_record.get("last_state"),
        "recovery_context": failure_record.get("recovery_context"),
    } != {"run_role": "replay", "last_state": "COMPLETE", "recovery_context": True}:
        raise RuntimeError("Wave 59 antecedent failure state drifted")


def validate_wave59_successor_final_authority(
    config: dict[str, Any], config_path: Path, *, repo_root: Path = REPO_ROOT
) -> None:
    """Require the audited config commit to be the exact clean execution HEAD."""
    from geometria_proporcional.wave59_hgb_guard_bracket import (
        SUCCESSOR_CONFIG_SOURCE_RELATIVE,
        is_successor_config,
    )

    if not is_successor_config(config):
        return
    _, authority, _, implementation_audit_commit = _wave59_successor_source_delta(
        config, config_path, repo_root
    )
    config_commit = git_introduction_commit(repo_root, SUCCESSOR_CONFIG_SOURCE_RELATIVE)
    if git_changed_paths(repo_root, config_commit) != {SUCCESSOR_CONFIG_SOURCE_RELATIVE}:
        raise RuntimeError("Wave 59 successor config commit contains unrelated paths")
    require_direct_parent(
        repo_root, config_commit, implementation_audit_commit, "Wave 59 successor config"
    )
    final_relative = authority["final_config_audit_path"]
    require_audit_report_path(final_relative, "Wave 59 successor final config audit")
    final_path, _ = require_repo_artifact(repo_root, final_relative)
    final_commit = git_introduction_commit(repo_root, final_relative)
    head = _git_output(repo_root, "rev-parse", "HEAD")
    if final_commit != head or git_changed_paths(repo_root, final_commit) != {final_relative}:
        raise RuntimeError("Wave 59 successor final audit is not the exclusive HEAD")
    require_direct_parent(
        repo_root, final_commit, config_commit, "Wave 59 successor final config audit"
    )
    _require_successor_pass_report(
        final_path,
        [
            f"**Config commit:** `{config_commit}`  ",
            f"**Config SHA-256:** `{digest(config_path)}`  ",
            "**Result:** `PASS`",
        ],
        "Wave 59 successor final config audit",
    )
    if _git_output(repo_root, "status", "--porcelain"):
        raise RuntimeError("Wave 59 successor requires a globally clean worktree")
    _validate_wave59_antecedent_sentinels(repo_root)


def _validate_wave56_contract_delta(
    escrow_contract: dict[str, Any],
    execution_contract: dict[str, Any],
    amendment: dict[str, Any],
    repo_root: Path,
) -> None:
    origin = amendment["escrow_origin"]
    implementation = amendment["implementation"]
    preparer = implementation["preparer"]
    runner = implementation["runner"]
    if compact_json_sha256(escrow_contract) != origin["contract_sha256"]:
        raise RuntimeError("escrow-origin contract hash differs from amendment")
    if escrow_contract.get("git_commit") != origin["contract_git_commit"]:
        raise RuntimeError("escrow-origin commit differs from amendment")
    if set(escrow_contract) != set(execution_contract):
        raise RuntimeError("execution contract fields differ from escrow-origin contract")
    for key in escrow_contract:
        if key not in {"git_commit", "sources"} and escrow_contract[key] != execution_contract[key]:
            raise RuntimeError(f"execution contract changed frozen field: {key}")
    old_sources = escrow_contract.get("sources", {})
    new_sources = execution_contract.get("sources", {})
    if set(old_sources) != set(new_sources):
        raise RuntimeError("execution source set differs from escrow-origin source set")
    changed = {name for name in old_sources if old_sources[name] != new_sources[name]}
    if changed != {PREPARER_RELATIVE, RUNNER_RELATIVE}:
        raise RuntimeError(
            "recovery permits only the preparer and runner source deltas, "
            f"got {sorted(changed)}"
        )
    if preparer != {
        "path": PREPARER_RELATIVE,
        "old_sha256": old_sources[PREPARER_RELATIVE],
        "new_sha256": new_sources[PREPARER_RELATIVE],
    }:
        raise RuntimeError("preparer source delta differs from amendment")
    if runner != {
        "path": RUNNER_RELATIVE,
        "old_sha256": old_sources[RUNNER_RELATIVE],
        "new_sha256": new_sources[RUNNER_RELATIVE],
    }:
        raise RuntimeError("runner source delta differs from amendment")
    head = _git_output(repo_root, "rev-parse", "HEAD")
    if execution_contract.get("git_commit") != head:
        raise RuntimeError("execution contract is not bound to current HEAD")


def _validate_wave57_contract_delta(
    origin_contract: dict[str, Any],
    execution_contract: dict[str, Any],
    amendment: dict[str, Any],
    repo_root: Path,
) -> None:
    """Accept only the two source-hash changes frozen by the Wave 57 amendment."""
    origin = amendment["escrow_origin"]
    implementation = amendment["implementation"]
    if compact_json_sha256(origin_contract) != origin["contract_sha256"]:
        raise RuntimeError("Wave 57 escrow-origin contract hash differs from amendment")
    if origin_contract.get("git_commit") != origin["contract_git_commit"]:
        raise RuntimeError("Wave 57 escrow-origin commit differs from amendment")
    if set(origin_contract) != set(execution_contract):
        raise RuntimeError("Wave 57 execution contract fields differ from escrow origin")
    for key in origin_contract:
        if key not in {"git_commit", "sources"} and origin_contract[key] != execution_contract[key]:
            raise RuntimeError(f"Wave 57 execution contract changed frozen field: {key}")
    old_sources = origin_contract.get("sources", {})
    new_sources = execution_contract.get("sources", {})
    if set(old_sources) != set(new_sources):
        raise RuntimeError("Wave 57 execution source set differs from escrow origin")
    allowed = {PREPARER_RELATIVE, WAVE57_RECOVERY_TEST_RELATIVE}
    changed = {name for name in old_sources if old_sources[name] != new_sources[name]}
    if changed != allowed:
        raise RuntimeError(
            "Wave 57 recovery permits only preparer and recovery-test source deltas, "
            f"got {sorted(changed)}"
        )
    for label, relative in (
        ("preparer", PREPARER_RELATIVE),
        ("test", WAVE57_RECOVERY_TEST_RELATIVE),
    ):
        expected = {
            "path": relative,
            "old_sha256": old_sources[relative],
            "new_sha256": new_sources[relative],
        }
        if implementation[label] != expected:
            raise RuntimeError(f"Wave 57 {label} source delta differs from amendment")
    head = _git_output(repo_root, "rev-parse", "HEAD")
    if execution_contract.get("git_commit") != head:
        raise RuntimeError("Wave 57 execution contract is not bound to current HEAD")


def _validate_wave59_contract_delta(
    origin_contract: dict[str, Any],
    execution_contract: dict[str, Any],
    amendment: dict[str, Any],
    repo_root: Path,
) -> None:
    """Accept only the three source deltas authorized for Wave 59 recovery."""
    origin = amendment["escrow_origin"]
    implementation = amendment["implementation"]
    if compact_json_sha256(origin_contract) != origin["contract_sha256"]:
        raise RuntimeError("Wave 59 escrow-origin contract hash differs from amendment")
    if origin_contract.get("git_commit") != origin["contract_git_commit"]:
        raise RuntimeError("Wave 59 escrow-origin commit differs from amendment")
    if set(origin_contract) != set(execution_contract):
        raise RuntimeError("Wave 59 execution contract fields differ from escrow origin")
    for key in origin_contract:
        if key not in {"git_commit", "sources"} and origin_contract[key] != execution_contract[key]:
            raise RuntimeError(f"Wave 59 execution contract changed frozen field: {key}")
    old_sources = origin_contract.get("sources", {})
    new_sources = execution_contract.get("sources", {})
    if set(old_sources) != set(new_sources):
        raise RuntimeError("Wave 59 execution source set differs from escrow origin")
    allowed = {
        PREPARER_RELATIVE,
        WAVE59_RUNNER_RELATIVE,
        WAVE59_PROSPECTIVE_TEST_RELATIVE,
    }
    changed = {name for name in old_sources if old_sources[name] != new_sources[name]}
    if changed != allowed:
        raise RuntimeError(
            "Wave 59 recovery permits only preparer, runner, and prospective-test "
            f"source deltas, got {sorted(changed)}"
        )
    for label, relative in (
        ("preparer", PREPARER_RELATIVE),
        ("runner", WAVE59_RUNNER_RELATIVE),
        ("prospective_test", WAVE59_PROSPECTIVE_TEST_RELATIVE),
    ):
        expected = {
            "path": relative,
            "old_sha256": old_sources[relative],
            "new_sha256": new_sources[relative],
        }
        if implementation[label] != expected:
            raise RuntimeError(f"Wave 59 {label} source delta differs from amendment")
    head = _git_output(repo_root, "rev-parse", "HEAD")
    if execution_contract.get("git_commit") != head:
        raise RuntimeError("Wave 59 execution contract is not bound to current HEAD")


def _validate_contract_delta(
    origin_contract: dict[str, Any],
    execution_contract: dict[str, Any],
    amendment: dict[str, Any],
    repo_root: Path,
) -> None:
    if amendment.get("schema_version") == WAVE57_RECOVERY_AMENDMENT_SCHEMA:
        _validate_wave57_contract_delta(
            origin_contract, execution_contract, amendment, repo_root
        )
        return
    if amendment.get("schema_version") == WAVE59_RECOVERY_AMENDMENT_SCHEMA:
        _validate_wave59_contract_delta(
            origin_contract, execution_contract, amendment, repo_root
        )
        return
    if amendment.get("schema_version") == WAVE60_RECOVERY_AMENDMENT_SCHEMA:
        _validate_wave60_contract_delta(
            origin_contract, execution_contract, amendment, repo_root
        )
        return
    _validate_wave56_contract_delta(
        origin_contract, execution_contract, amendment, repo_root
    )


def _validate_wave60_contract_delta(
    origin_contract: dict[str, Any],
    execution_contract: dict[str, Any],
    amendment: dict[str, Any],
    repo_root: Path,
) -> None:
    """Allow only the versioned namespace/authority delta for Wave 60 recovery."""
    origin = origin_contract.get("prospective_config")
    current = execution_contract.get("prospective_config")
    if not isinstance(origin, dict) or not isinstance(current, dict):
        raise RuntimeError("Wave 60 recovery contracts lack prospective configs")
    validate_prospective_config(origin)
    validate_prospective_config(current)
    immutable = {
        "schema_version",
        "status",
        "device",
        "cpu_threads",
        "penalty",
        "bootstrap",
        "runtime_budget",
        "main_policies",
        "source_law_authority",
        "implementation_binding",
        "plan_binding",
        "fresh_benchmark",
        "physical_splits",
        "seeds",
        "inference_batch_size",
        "feature_names",
        "source_binding",
    }
    if any(origin.get(key) != current.get(key) for key in immutable):
        raise RuntimeError("Wave 60 recovery changed the frozen scientific contract")
    if int(origin["attempt"]["version"]) >= int(current["attempt"]["version"]):
        raise RuntimeError("Wave 60 recovery attempt version did not advance")
    recovery = current["attempt"]["recovery"]
    if (
        recovery["prior_attempt_container"] != origin["attempt"]["container"]
        or recovery["amendment_sha256"]
        != sha256_file(repo_root / recovery["amendment_path"])
        or amendment["prior_attempt_container"]
        != recovery["prior_attempt_container"]
    ):
        raise RuntimeError("Wave 60 recovery namespace is not amendment-bound")
    origin_sources = origin_contract.get("sources", {})
    current_sources = execution_contract.get("sources", {})
    shared = set(origin_sources) & set(current_sources)
    config_paths = {
        relative
        for relative in set(origin_sources) | set(current_sources)
        if relative.endswith("wave60_frozen_policy_transport.json")
    }
    if any(
        origin_sources[path] != current_sources[path]
        for path in shared - config_paths
    ):
        raise RuntimeError("Wave 60 recovery changed an execution-source blob")


def _inventory_records_by_path(
    inventory: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for record in inventory:
        relative = record.get("path")
        if not isinstance(relative, str) or relative in records:
            raise RuntimeError("recovery inventory contains an invalid or duplicate path")
        records[relative] = record
    return records


def validate_wave59_repository_recovery_authority(
    amendment_path: Path,
    expected_sources: dict[str, str],
    observed_sources: dict[str, str],
    *,
    repo_root: Path = REPO_ROOT,
) -> tuple[dict[str, Any], str]:
    """Authenticate the public Git/source chain without touching draw secrets."""
    canonical_path = repo_root / WAVE59_RECOVERY_AMENDMENT_RELATIVE
    if (
        amendment_path.is_symlink()
        or amendment_path.absolute() != canonical_path.absolute()
        or amendment_path.resolve(strict=True) != canonical_path.resolve(strict=True)
    ):
        raise RuntimeError("Wave 59 amendment must use its canonical repository path")
    path, amendment_sha256 = require_repo_artifact(
        repo_root, WAVE59_RECOVERY_AMENDMENT_RELATIVE
    )
    amendment = json.loads(path.read_text(encoding="utf-8"))
    if sha256_file(path) != canonical_json_sha256(amendment):
        raise RuntimeError("Wave 59 recovery amendment is not canonical pretty JSON")
    _require_keys(
        amendment,
        {
            "schema_version",
            "status",
            "plan",
            "plan_audit",
            "implementation",
            "implementation_audit",
            "final_audit_path",
            "escrow_origin",
            "population_contract",
            "assertions",
        },
        "Wave 59 recovery amendment",
    )
    if amendment["schema_version"] != WAVE59_RECOVERY_AMENDMENT_SCHEMA:
        raise RuntimeError("Wave 59 recovery amendment schema differs")
    if amendment["status"] != "APPROVED_PREORACLE_RECOVERY":
        raise RuntimeError("Wave 59 recovery amendment is not approved")
    if amendment["assertions"] != {
        "no_redraw": True,
        "no_inference_in_origin": True,
        "no_materialized_oracle_in_origin": True,
        "no_authorized_labels_in_origin": True,
    }:
        raise RuntimeError("Wave 59 recovery amendment assertions differ")

    if set(expected_sources) != set(observed_sources):
        raise RuntimeError("Wave 59 recovery execution-source set differs")
    allowed = {
        PREPARER_RELATIVE,
        WAVE59_RUNNER_RELATIVE,
        WAVE59_PROSPECTIVE_TEST_RELATIVE,
    }
    changed = {
        relative
        for relative in expected_sources
        if expected_sources[relative] != observed_sources[relative]
    }
    if changed != allowed:
        raise RuntimeError(
            "Wave 59 recovery source authority permits exactly preparer, runner, "
            f"and prospective-test deltas, got {sorted(changed)}"
        )

    plan = amendment["plan"]
    _require_keys(plan, {"commit", "path", "sha256"}, "Wave 59 recovery plan")
    if plan["path"] != WAVE59_RECOVERY_PLAN_RELATIVE:
        raise RuntimeError("Wave 59 recovery plan path differs")
    require_repo_artifact(repo_root, plan["path"], plan["sha256"])
    plan_commit = plan["commit"]
    if git_changed_paths(repo_root, plan_commit) != {plan["path"]}:
        raise RuntimeError("Wave 59 recovery-plan commit contains unrelated paths")
    if git_blob_sha256(repo_root, plan_commit, plan["path"]) != plan["sha256"]:
        raise RuntimeError("Wave 59 recovery-plan blob differs from amendment")

    plan_audit = amendment["plan_audit"]
    _require_keys(plan_audit, {"commit", "path", "sha256"}, "Wave 59 plan audit")
    require_audit_report_path(plan_audit["path"], "Wave 59 plan audit")
    plan_audit_path, _ = require_repo_artifact(
        repo_root, plan_audit["path"], plan_audit["sha256"]
    )
    _require_report_fields(
        plan_audit_path,
        [
            f"**Plan commit:** `{plan_commit}`",
            f"**Plan SHA:** `{plan['sha256']}`",
            "**Result:** `PASS`",
        ],
        "Wave 59 plan audit",
        allow_one_terminal_blank=True,
    )
    plan_audit_commit = plan_audit["commit"]
    if git_introduction_commit(repo_root, plan_audit["path"]) != plan_audit_commit:
        raise RuntimeError("Wave 59 plan-audit commit differs from its introduction")
    if git_changed_paths(repo_root, plan_audit_commit) != {plan_audit["path"]}:
        raise RuntimeError("Wave 59 plan-audit commit contains unrelated paths")
    require_direct_parent(
        repo_root, plan_audit_commit, plan_commit, "Wave 59 plan audit"
    )

    implementation = amendment["implementation"]
    _require_keys(
        implementation,
        {"commit", "preparer", "runner", "prospective_test", "recovery_test"},
        "Wave 59 recovery implementation",
    )
    implementation_commit = implementation["commit"]
    require_direct_parent(
        repo_root,
        implementation_commit,
        plan_audit_commit,
        "Wave 59 recovery implementation",
    )
    implementation_paths = {
        PREPARER_RELATIVE,
        WAVE59_RECOVERY_TEST_RELATIVE,
    }
    if git_changed_paths(repo_root, implementation_commit) != implementation_paths:
        raise RuntimeError(
            "Wave 59 authority-chain correction commit changed unauthorized paths"
        )
    for label, relative in (
        ("preparer", PREPARER_RELATIVE),
        ("runner", WAVE59_RUNNER_RELATIVE),
        ("prospective_test", WAVE59_PROSPECTIVE_TEST_RELATIVE),
    ):
        delta = implementation[label]
        _require_keys(
            delta, {"path", "old_sha256", "new_sha256"}, f"Wave 59 {label}"
        )
        expected = {
            "path": relative,
            "old_sha256": expected_sources[relative],
            "new_sha256": observed_sources[relative],
        }
        if delta != expected:
            raise RuntimeError(f"Wave 59 {label} source delta differs")
        if git_blob_sha256(repo_root, implementation_commit, relative) != delta["new_sha256"]:
            raise RuntimeError(f"Wave 59 {label} implementation blob differs")
        require_repo_artifact(repo_root, relative, delta["new_sha256"])
    recovery_test = implementation["recovery_test"]
    _require_keys(
        recovery_test,
        {"path", "sha256", "introduced_commit"},
        "Wave 59 recovery test",
    )
    if recovery_test["path"] != WAVE59_RECOVERY_TEST_RELATIVE:
        raise RuntimeError("Wave 59 recovery-test path differs")
    if (
        git_introduction_commit(repo_root, WAVE59_RECOVERY_TEST_RELATIVE)
        != recovery_test["introduced_commit"]
    ):
        raise RuntimeError("Wave 59 recovery-test historical introduction drifted")
    require_ancestor(
        repo_root,
        recovery_test["introduced_commit"],
        implementation_commit,
    )
    if git_blob_sha256(
        repo_root, implementation_commit, WAVE59_RECOVERY_TEST_RELATIVE
    ) != recovery_test["sha256"]:
        raise RuntimeError("Wave 59 recovery-test implementation blob differs")
    require_repo_artifact(
        repo_root, WAVE59_RECOVERY_TEST_RELATIVE, recovery_test["sha256"]
    )

    implementation_audit = amendment["implementation_audit"]
    _require_keys(
        implementation_audit,
        {"commit", "path", "sha256"},
        "Wave 59 implementation audit",
    )
    require_audit_report_path(
        implementation_audit["path"], "Wave 59 implementation audit"
    )
    implementation_audit_path, _ = require_repo_artifact(
        repo_root, implementation_audit["path"], implementation_audit["sha256"]
    )
    _require_report_fields(
        implementation_audit_path,
        [
            f"**Implementation commit:** `{implementation_commit}`",
            f"**Preparer SHA-256:** `{implementation['preparer']['new_sha256']}`",
            f"**Runner SHA-256:** `{implementation['runner']['new_sha256']}`",
            "**Prospective test SHA-256:** "
            f"`{implementation['prospective_test']['new_sha256']}`",
            f"**Recovery test SHA-256:** `{recovery_test['sha256']}`",
            "**Result:** `PASS`",
        ],
        "Wave 59 implementation audit",
        allow_one_terminal_blank=True,
    )
    implementation_audit_commit = implementation_audit["commit"]
    if git_introduction_commit(
        repo_root, implementation_audit["path"]
    ) != implementation_audit_commit:
        raise RuntimeError("Wave 59 implementation-audit commit differs")
    if git_changed_paths(repo_root, implementation_audit_commit) != {
        implementation_audit["path"]
    }:
        raise RuntimeError("Wave 59 implementation-audit commit contains unrelated paths")
    require_direct_parent(
        repo_root,
        implementation_audit_commit,
        implementation_commit,
        "Wave 59 implementation audit",
    )

    amendment_commit = git_introduction_commit(
        repo_root, WAVE59_RECOVERY_AMENDMENT_RELATIVE
    )
    if git_changed_paths(repo_root, amendment_commit) != {
        WAVE59_RECOVERY_AMENDMENT_RELATIVE
    }:
        raise RuntimeError("Wave 59 amendment commit contains unrelated paths")
    require_direct_parent(
        repo_root,
        amendment_commit,
        implementation_audit_commit,
        "Wave 59 amendment",
    )
    if git_blob_sha256(
        repo_root, amendment_commit, WAVE59_RECOVERY_AMENDMENT_RELATIVE
    ) != amendment_sha256:
        raise RuntimeError("Wave 59 amendment blob changed after its commit")

    final_relative = amendment["final_audit_path"]
    require_audit_report_path(final_relative, "Wave 59 final audit")
    if final_relative in {plan_audit["path"], implementation_audit["path"]}:
        raise RuntimeError("Wave 59 audits must use distinct reports")
    final_path, _ = require_repo_artifact(repo_root, final_relative)
    final_commit = git_introduction_commit(repo_root, final_relative)
    if git_changed_paths(repo_root, final_commit) != {final_relative}:
        raise RuntimeError("Wave 59 final-audit commit contains unrelated paths")
    require_direct_parent(repo_root, final_commit, amendment_commit, "Wave 59 final audit")
    head = _git_output(repo_root, "rev-parse", "HEAD")
    if final_commit != head:
        raise RuntimeError("Wave 59 recovery execution HEAD must be final-audit commit")
    _require_report_fields(
        final_path,
        [
            f"**Audited package commit:** `{amendment_commit}`",
            f"**Amendment SHA-256:** `{amendment_sha256}`",
            "**Result:** `PASS`",
        ],
        "Wave 59 final audit",
        allow_one_terminal_blank=True,
    )
    allowed_after_implementation = {
        implementation_audit["path"],
        WAVE59_RECOVERY_AMENDMENT_RELATIVE,
        final_relative,
    }
    changed_after_implementation = {
        line
        for line in _git_output(
            repo_root, "diff", "--name-only", f"{implementation_commit}..{head}"
        ).splitlines()
        if line
    }
    if changed_after_implementation != allowed_after_implementation:
        raise RuntimeError("Wave 59 post-implementation commits changed unauthorized paths")
    if _git_output(repo_root, "status", "--porcelain"):
        raise RuntimeError("Wave 59 recovery requires a globally clean worktree")
    return amendment, amendment_sha256


def _wave57_public_json(failed: Path, relative: str) -> dict[str, Any]:
    allowed = {
        "FAILURE.json",
        FREEZE_NAME,
        "benchmark/manifest.json",
        "benchmark/attestations/semantic_root.json",
    }
    if relative not in allowed:
        raise RuntimeError(f"Wave 57 content-blind preflight forbids JSON parsing: {relative}")
    path = failed / relative
    if path.resolve(strict=True).relative_to(failed.resolve(strict=True)) != Path(relative):
        raise RuntimeError(f"Wave 57 public JSON path is non-canonical: {relative}")
    return json.loads(path.read_text(encoding="utf-8"))


def _wave57_manifest_record(
    records: dict[str, dict[str, Any]], relative: str
) -> dict[str, Any]:
    record = records.get(f"benchmark/{relative}")
    if record is None or record.get("type") != "file":
        raise RuntimeError(f"Wave 57 manifest file is absent from inventory: {relative}")
    return {"bytes": record["bytes"], "sha256": record["sha256"]}


def validate_wave57_failed_origin_content_blind(
    amendment: dict[str, Any],
    source_parent: Path,
    execution_contract: dict[str, Any],
    trusted_public_key_path: Path,
    *,
    repo_root: Path = REPO_ROOT,
) -> tuple[Path, list[dict[str, Any]], dict[str, Any]]:
    """Authenticate the Wave 57 origin without parsing escrow or sealed content."""
    origin = amendment["escrow_origin"]
    failed = source_parent / origin["failed_attempt_basename"]
    if failed.parent.resolve() != source_parent.resolve() or failed.name != origin["failed_attempt_basename"]:
        raise RuntimeError("Wave 57 recovery-origin basename escapes its canonical parent")
    observed = physical_tree_inventory(failed)
    if observed != origin["inventory"]:
        raise RuntimeError("Wave 57 recovery-origin physical whitelist differs from amendment")
    if len(observed) != 24:
        raise RuntimeError("Wave 57 recovery-origin inventory must contain exactly 24 entries")
    records = _inventory_records_by_path(observed)
    if sum(row.get("type") == "directory" for row in observed) != 6:
        raise RuntimeError("Wave 57 recovery-origin directory count differs")
    if sum(row.get("type") == "file" for row in observed) != 18:
        raise RuntimeError("Wave 57 recovery-origin file count differs")
    forbidden = (
        "inference",
        "authorized_labels",
        "bundles",
        "phases",
        "preparation_freeze.json",
        "generation_receipt.json",
        "benchmark/sealed/oracle",
    )
    for relative in records:
        if any(relative == item or relative.startswith(f"{item}/") for item in forbidden):
            raise RuntimeError(f"Wave 57 pre-oracle origin contains forbidden material: {relative}")
    for record in observed:
        if record["uid"] != 0 or record["gid"] != 0:
            raise PermissionError("Wave 57 recovery-origin entries must remain root-owned")
        expected_mode = "0700" if record["type"] == "directory" else (
            "0644" if record["path"] == FREEZE_NAME else "0600"
        )
        if record["mode"] != expected_mode:
            raise PermissionError(
                f"Wave 57 recovery-origin mode differs for {record['path']}"
            )
    required_hashes = {
        ESCROW_NAME: origin["escrow_sha256"],
        FREEZE_NAME: origin["pre_generation_freeze_sha256"],
        "FAILURE.json": origin["failure_sha256"],
        "benchmark/manifest.json": origin["benchmark_manifest_sha256"],
    }
    for relative, expected in required_hashes.items():
        record = records.get(relative)
        if record is None or record.get("sha256") != expected:
            raise RuntimeError(f"Wave 57 recovery-origin hash differs for {relative}")

    failure = _wave57_public_json(failed, "FAILURE.json")
    if failure != {
        "error_type": "RuntimeError",
        "escrow_present": True,
        "message": "fresh benchmark pair-token count differs from prospective freeze",
        "redraw_forbidden_if_escrow_present": True,
    }:
        raise RuntimeError("Wave 57 FAILURE.json is not the authorized pre-oracle failure")
    freeze = _wave57_public_json(failed, FREEZE_NAME)
    _require_keys(
        freeze,
        {
            "schema_version",
            "phase",
            "contract",
            "key_commitments",
            "contains_secrets",
            "generator_invoked",
        },
        "Wave 57 public pre-generation freeze",
    )
    if (
        freeze["schema_version"] != "wave56-key-escrow-v1"
        or freeze["phase"] != "keys-escrowed-and-contract-frozen-before-generation"
        or freeze["contains_secrets"] is not False
        or freeze["generator_invoked"] is not False
    ):
        raise RuntimeError("Wave 57 public pre-generation freeze semantics differ")
    commitments = freeze["key_commitments"]
    if set(commitments) != set(SECRET_FILES) or any(
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        for value in commitments.values()
    ):
        raise RuntimeError("Wave 57 public key commitments are malformed")
    _validate_wave57_contract_delta(
        freeze["contract"], execution_contract, amendment, repo_root
    )

    manifest = _wave57_public_json(failed, "benchmark/manifest.json")
    if manifest.get("schema_version") != "wave49-relational-benchmark-v2":
        raise RuntimeError("Wave 57 benchmark manifest schema differs")
    manifest_files = manifest.get("files")
    if not isinstance(manifest_files, dict):
        raise RuntimeError("Wave 57 benchmark manifest lacks its file map")
    physical_benchmark_files = {
        relative.removeprefix("benchmark/")
        for relative, record in records.items()
        if relative.startswith("benchmark/")
        and record["type"] == "file"
        and relative != "benchmark/manifest.json"
    }
    if set(manifest_files) != physical_benchmark_files:
        raise RuntimeError("Wave 57 manifest file map differs from physical inventory")
    for relative, expected in manifest_files.items():
        if expected != _wave57_manifest_record(records, relative):
            raise RuntimeError(f"Wave 57 manifest record differs for {relative}")
    if manifest.get("counts") != {
        "calibration_null": 3072,
        "train": 4992,
        "val": 4992,
        "lockbox": 4992,
    }:
        raise RuntimeError("Wave 57 public manifest counts differ")
    manifest_commitments = {
        "generation_secret.json": manifest.get("generation_key_commitment"),
        "identity_secret.json": manifest.get("identity_key_commitment"),
        "semantic_commitment_secret.json": manifest.get(
            "semantic_commitment_key_commitment"
        ),
    }
    if manifest_commitments != commitments:
        raise RuntimeError("Wave 57 manifest commitments differ from public freeze")

    protocol = default_protocol_config(smoke=False)
    protocol_record = _wave57_manifest_record(records, "protocol_config.json")
    if protocol_record["sha256"] != execution_contract["source_bindings"][
        "wave50_protocol_sha256"
    ]:
        raise RuntimeError("Wave 57 opaque protocol config differs from frozen protocol")
    validate_visible_package(failed / "benchmark", protocol)

    attestation = _wave57_public_json(
        failed, "benchmark/attestations/semantic_root.json"
    )
    try:
        verify_attestation(attestation, trusted_public_key_path)
    except AttestationError as exc:
        raise RuntimeError("Wave 57 detached semantic attestation is invalid") from exc
    payload = attestation.get("payload", {})
    expected_attested_commitments = {
        "generation": commitments["generation_secret.json"],
        "identity": commitments["identity_secret.json"],
        "semantic_hmac": commitments["semantic_commitment_secret.json"],
    }
    expected_sealed = {
        split: _wave57_manifest_record(records, f"sealed/{split}.jsonl")
        for split in (*SPLITS, "calibration_null")
    }
    if payload.get("phase") != "sealed-semantics-committed-before-selector":
        raise RuntimeError("Wave 57 detached attestation phase differs")
    if payload.get("schema_version") != "wave49-relational-benchmark-v2":
        raise RuntimeError("Wave 57 detached attestation schema differs")
    if payload.get("protocol_config") != _wave57_manifest_record(
        records, "protocol_config.json"
    ):
        raise RuntimeError("Wave 57 detached protocol record differs")
    if payload.get("semantic_commitments") != _wave57_manifest_record(
        records, "commitments/semantic.jsonl"
    ):
        raise RuntimeError("Wave 57 detached commitment record differs")
    if payload.get("sealed_truth") != expected_sealed:
        raise RuntimeError("Wave 57 detached sealed-truth records differ")
    if payload.get("counts") != manifest["counts"]:
        raise RuntimeError("Wave 57 detached counts differ from public manifest")
    if payload.get("key_commitments") != expected_attested_commitments:
        raise RuntimeError("Wave 57 detached commitments differ from public freeze")
    if manifest.get("semantic_attestation") != {
        "path": "attestations/semantic_root.json",
        "phase": payload["phase"],
        "trusted_public_key_sha256": attestation.get("trusted_public_key_sha256"),
    }:
        raise RuntimeError("Wave 57 manifest attestation binding differs")
    return failed, observed, freeze["contract"]


def validate_wave57_failed_origin_semantic(
    amendment: dict[str, Any],
    failed: Path,
    trusted_public_key_path: Path,
    expected_inventory: list[dict[str, Any]],
) -> dict[str, Any]:
    """Validate secrets and sealed truth only after content-blind authority passes."""
    observed = physical_tree_inventory(failed)
    if observed != expected_inventory:
        raise RuntimeError("Wave 57 recovery origin changed before semantic validation")
    escrow = read_escrow(failed)
    freeze = json.loads((failed / FREEZE_NAME).read_text(encoding="utf-8"))
    if freeze != public_freeze_from_escrow(escrow):
        raise RuntimeError("Wave 57 public freeze differs from authorized escrow")
    benchmark = failed / "benchmark"
    validate_manifest(benchmark)
    protocol = ProtocolConfig.from_dict(
        json.loads((benchmark / "protocol_config.json").read_text(encoding="utf-8"))
    )
    validate_visible_package(benchmark, protocol)
    validate_semantic_attestation(benchmark, trusted_public_key_path)
    expected_counts = amendment["population_contract"]["counts_by_split"]
    actual_counts = {
        split: sealed_population_counts(benchmark / "sealed" / f"{split}.jsonl")
        for split in SPLITS
    }
    if actual_counts != expected_counts:
        raise RuntimeError("Wave 57 recovery-origin token populations differ from amendment")
    return escrow


def _wave59_public_json(failed: Path, relative: str) -> dict[str, Any]:
    allowed = {
        "FAILURE.json",
        "failure_inventory.json",
        "failure_attestation.json",
        FREEZE_NAME,
        "benchmark/manifest.json",
        "benchmark/attestations/semantic_root.json",
    }
    if relative not in allowed:
        raise RuntimeError(f"Wave 59 content-blind preflight forbids JSON parsing: {relative}")
    path = failed / relative
    if path.resolve(strict=True).relative_to(failed.resolve(strict=True)) != Path(relative):
        raise RuntimeError(f"Wave 59 public JSON path is non-canonical: {relative}")
    return json.loads(path.read_text(encoding="utf-8"))


def _wave59_manifest_record(
    records: dict[str, dict[str, Any]], relative: str
) -> dict[str, Any]:
    record = records.get(f"benchmark/{relative}")
    if record is None or record.get("type") != "file":
        raise RuntimeError(f"Wave 59 manifest file is absent from inventory: {relative}")
    return {"bytes": record["bytes"], "sha256": record["sha256"]}


def validate_wave59_failed_origin_content_blind(
    amendment: dict[str, Any],
    source_parent: Path,
    execution_contract: dict[str, Any],
    trusted_public_key_path: Path,
    *,
    repo_root: Path = REPO_ROOT,
) -> tuple[Path, list[dict[str, Any]], dict[str, Any]]:
    """Authenticate the Wave 59 failure without parsing escrow or sealed material."""
    origin = amendment["escrow_origin"]
    failed = source_parent / origin["failed_attempt_basename"]
    if failed.parent.resolve() != source_parent.resolve() or failed.name != origin["failed_attempt_basename"]:
        raise RuntimeError("Wave 59 recovery-origin basename escapes its canonical parent")
    observed = physical_tree_inventory(failed)
    if observed != origin["inventory"]:
        raise RuntimeError("Wave 59 recovery-origin physical whitelist differs from amendment")
    if len(observed) != 26:
        raise RuntimeError("Wave 59 recovery-origin inventory must contain exactly 26 entries")
    records = _inventory_records_by_path(observed)
    if sum(row.get("type") == "directory" for row in observed) != 6:
        raise RuntimeError("Wave 59 recovery-origin directory count differs")
    if sum(row.get("type") == "file" for row in observed) != 20:
        raise RuntimeError("Wave 59 recovery-origin file count differs")
    forbidden = (
        "inference",
        "authorized_labels",
        "prepared",
        "bundles",
        "phases",
        "journals",
        "preparation_freeze.json",
        "generation_receipt.json",
        "benchmark/sealed/oracle",
    )
    for relative in records:
        if any(relative == item or relative.startswith(f"{item}/") for item in forbidden):
            raise RuntimeError(f"Wave 59 pre-oracle origin contains forbidden material: {relative}")
    for record in observed:
        if record["uid"] != 0 or record["gid"] != 0:
            raise PermissionError("Wave 59 recovery-origin entries must remain root-owned")
        expected_mode = "0700" if record["type"] == "directory" else (
            "0644" if record["path"] == FREEZE_NAME else "0600"
        )
        if record["mode"] != expected_mode:
            raise PermissionError(
                f"Wave 59 recovery-origin mode differs for {record['path']}"
            )
    required_hashes = {
        ESCROW_NAME: origin["escrow_sha256"],
        FREEZE_NAME: origin["pre_generation_freeze_sha256"],
        "FAILURE.json": origin["failure_sha256"],
        "failure_inventory.json": origin["failure_inventory_sha256"],
        "failure_attestation.json": origin["failure_attestation_sha256"],
        "benchmark/manifest.json": origin["benchmark_manifest_sha256"],
    }
    for relative, expected in required_hashes.items():
        record = records.get(relative)
        if record is None or record.get("sha256") != expected:
            raise RuntimeError(f"Wave 59 recovery-origin hash differs for {relative}")

    failure = _wave59_public_json(failed, "FAILURE.json")
    expected_primary = source_parent / execution_contract["prospective_config"][
        "primary_output_name"
    ]
    expected_failure = {
        "archived_path": str(failed.resolve()),
        "error_message_sha256": hashlib.sha256(
            b"fresh benchmark pair-token count differs from prospective freeze"
        ).hexdigest(),
        "error_type": "RuntimeError",
        "last_state": None,
        "maximum_truth_materialized": "none",
        "original_path": str(expected_primary.resolve()),
        "recovery_context": False,
        "run_role": "primary",
        "schema_version": "wave59-failed-attempt-v1",
    }
    if failure != expected_failure:
        raise RuntimeError("Wave 59 FAILURE.json is not the authorized pre-oracle failure")

    failure_inventory = _wave59_public_json(failed, "failure_inventory.json")
    _require_keys(
        failure_inventory,
        {
            "schema_version",
            "records",
            "failure_records",
            "missing_required_through_last_journal",
            "extra",
            "overlap",
            "unclassified",
        },
        "Wave 59 failure inventory",
    )
    if failure_inventory["schema_version"] != "wave59-failure-inventory-v1":
        raise RuntimeError("Wave 59 failure inventory schema differs")
    if failure_inventory["failure_records"] != [
        "FAILURE.json",
        "failure_inventory.json",
        "failure_attestation.json",
    ]:
        raise RuntimeError("Wave 59 failure inventory metadata list differs")
    if any(
        failure_inventory[field] != []
        for field in ("missing_required_through_last_journal", "overlap", "unclassified")
    ):
        raise RuntimeError("Wave 59 failure inventory reports unresolved coverage")
    inventory_rows = failure_inventory["records"]
    if not isinstance(inventory_rows, list):
        raise RuntimeError("Wave 59 failure inventory records are absent")
    inventory_by_path: dict[str, dict[str, Any]] = {}
    for row in inventory_rows:
        _require_keys(
            row, {"path", "class", "bytes", "sha256"}, "Wave 59 failure record"
        )
        relative = row["path"]
        if not isinstance(relative, str) or relative in inventory_by_path:
            raise RuntimeError("Wave 59 failure inventory path is invalid or duplicated")
        inventory_by_path[relative] = row
    expected_inventory_paths = {
        relative
        for relative, record in records.items()
        if record["type"] == "file"
        and relative not in {"failure_inventory.json", "failure_attestation.json"}
    }
    if set(inventory_by_path) != expected_inventory_paths:
        raise RuntimeError("Wave 59 failure inventory file coverage differs")
    for relative, row in inventory_by_path.items():
        physical = records[relative]
        if row["bytes"] != physical["bytes"] or row["sha256"] != physical["sha256"]:
            raise RuntimeError(f"Wave 59 failure inventory record differs for {relative}")
        expected_class = (
            "failure_record"
            if relative == "FAILURE.json"
            else "secret_excluded_from_public_manifest"
            if relative == ESCROW_NAME or relative.startswith("benchmark/sealed/")
            else "scientific_exact"
        )
        if row["class"] != expected_class:
            raise RuntimeError(f"Wave 59 failure inventory class differs for {relative}")
    expected_extra = sorted(expected_inventory_paths - {"FAILURE.json"})
    if failure_inventory["extra"] != expected_extra:
        raise RuntimeError("Wave 59 failure inventory extra set differs")

    failure_attestation = _wave59_public_json(failed, "failure_attestation.json")
    try:
        verify_attestation(failure_attestation, trusted_public_key_path)
    except AttestationError as exc:
        raise RuntimeError("Wave 59 failure attestation is invalid") from exc
    expected_anchor = {
        "schema_version": "wave59-failure-anchor-v1",
        "failure_inventory_sha256": origin["failure_inventory_sha256"],
        "failure_sha256": origin["failure_sha256"],
        "archived_path": str(failed.resolve()),
    }
    if failure_attestation.get("payload") != expected_anchor:
        raise RuntimeError("Wave 59 failure attestation payload differs")

    freeze = _wave59_public_json(failed, FREEZE_NAME)
    _require_keys(
        freeze,
        {
            "schema_version",
            "phase",
            "contract",
            "key_commitments",
            "contains_secrets",
            "generator_invoked",
        },
        "Wave 59 public pre-generation freeze",
    )
    if (
        freeze["schema_version"] != "wave56-key-escrow-v1"
        or freeze["phase"] != "keys-escrowed-and-contract-frozen-before-generation"
        or freeze["contains_secrets"] is not False
        or freeze["generator_invoked"] is not False
    ):
        raise RuntimeError("Wave 59 public pre-generation freeze semantics differ")
    commitments = freeze["key_commitments"]
    if set(commitments) != set(SECRET_FILES) or any(
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        for value in commitments.values()
    ):
        raise RuntimeError("Wave 59 public key commitments are malformed")
    _validate_wave59_contract_delta(
        freeze["contract"], execution_contract, amendment, repo_root
    )

    manifest = _wave59_public_json(failed, "benchmark/manifest.json")
    if manifest.get("schema_version") != "wave49-relational-benchmark-v2":
        raise RuntimeError("Wave 59 benchmark manifest schema differs")
    manifest_files = manifest.get("files")
    if not isinstance(manifest_files, dict):
        raise RuntimeError("Wave 59 benchmark manifest lacks its file map")
    physical_benchmark_files = {
        relative.removeprefix("benchmark/")
        for relative, record in records.items()
        if relative.startswith("benchmark/")
        and record["type"] == "file"
        and relative != "benchmark/manifest.json"
    }
    if set(manifest_files) != physical_benchmark_files:
        raise RuntimeError("Wave 59 manifest file map differs from physical inventory")
    for relative, expected in manifest_files.items():
        if expected != _wave59_manifest_record(records, relative):
            raise RuntimeError(f"Wave 59 manifest record differs for {relative}")
    if manifest.get("counts") != {
        "calibration_null": 3072,
        "train": 4992,
        "val": 4992,
        "lockbox": 4992,
    }:
        raise RuntimeError("Wave 59 public manifest counts differ")
    manifest_commitments = {
        "generation_secret.json": manifest.get("generation_key_commitment"),
        "identity_secret.json": manifest.get("identity_key_commitment"),
        "semantic_commitment_secret.json": manifest.get(
            "semantic_commitment_key_commitment"
        ),
    }
    if manifest_commitments != commitments:
        raise RuntimeError("Wave 59 manifest commitments differ from public freeze")
    protocol_record = _wave59_manifest_record(records, "protocol_config.json")
    if protocol_record["sha256"] != execution_contract["source_bindings"][
        "wave50_protocol_sha256"
    ]:
        raise RuntimeError("Wave 59 protocol config differs from frozen protocol")
    validate_visible_package(failed / "benchmark", default_protocol_config(smoke=False))

    attestation = _wave59_public_json(
        failed, "benchmark/attestations/semantic_root.json"
    )
    try:
        verify_attestation(attestation, trusted_public_key_path)
    except AttestationError as exc:
        raise RuntimeError("Wave 59 detached semantic attestation is invalid") from exc
    payload = attestation.get("payload", {})
    expected_attested_commitments = {
        "generation": commitments["generation_secret.json"],
        "identity": commitments["identity_secret.json"],
        "semantic_hmac": commitments["semantic_commitment_secret.json"],
    }
    expected_sealed = {
        split: _wave59_manifest_record(records, f"sealed/{split}.jsonl")
        for split in (*SPLITS, "calibration_null")
    }
    if payload.get("phase") != "sealed-semantics-committed-before-selector":
        raise RuntimeError("Wave 59 detached semantic attestation phase differs")
    if payload.get("schema_version") != "wave49-relational-benchmark-v2":
        raise RuntimeError("Wave 59 detached semantic attestation schema differs")
    if payload.get("protocol_config") != protocol_record:
        raise RuntimeError("Wave 59 detached protocol record differs")
    if payload.get("semantic_commitments") != _wave59_manifest_record(
        records, "commitments/semantic.jsonl"
    ):
        raise RuntimeError("Wave 59 detached commitment record differs")
    if payload.get("sealed_truth") != expected_sealed:
        raise RuntimeError("Wave 59 detached sealed-truth records differ")
    if payload.get("counts") != manifest["counts"]:
        raise RuntimeError("Wave 59 detached counts differ from public manifest")
    if payload.get("key_commitments") != expected_attested_commitments:
        raise RuntimeError("Wave 59 detached commitments differ from public freeze")
    if manifest.get("semantic_attestation") != {
        "path": "attestations/semantic_root.json",
        "phase": payload["phase"],
        "trusted_public_key_sha256": attestation.get("trusted_public_key_sha256"),
    }:
        raise RuntimeError("Wave 59 manifest attestation binding differs")
    return failed, observed, freeze["contract"]


def validate_wave59_failed_origin_semantic(
    amendment: dict[str, Any],
    failed: Path,
    trusted_public_key_path: Path,
    expected_inventory: list[dict[str, Any]],
) -> dict[str, Any]:
    """Open Wave 59 secrets only after repeating the complete opaque inventory."""
    observed = physical_tree_inventory(failed)
    if observed != expected_inventory:
        raise RuntimeError("Wave 59 recovery origin changed before semantic validation")
    escrow = read_escrow(failed)
    freeze = json.loads((failed / FREEZE_NAME).read_text(encoding="utf-8"))
    if freeze != public_freeze_from_escrow(escrow):
        raise RuntimeError("Wave 59 public freeze differs from authorized escrow")
    benchmark = failed / "benchmark"
    validate_manifest(benchmark)
    protocol = ProtocolConfig.from_dict(
        json.loads((benchmark / "protocol_config.json").read_text(encoding="utf-8"))
    )
    validate_visible_package(benchmark, protocol)
    validate_semantic_attestation(benchmark, trusted_public_key_path)
    expected_counts = amendment["population_contract"]["counts_by_split"]
    actual_counts = {
        split: sealed_population_counts(benchmark / "sealed" / f"{split}.jsonl")
        for split in SPLITS
    }
    if actual_counts != expected_counts:
        raise RuntimeError("Wave 59 recovery-origin token populations differ from amendment")
    return escrow


def validate_failed_recovery_origin(
    amendment: dict[str, Any],
    source_parent: Path,
    trusted_public_key_path: Path,
) -> tuple[Path, list[dict[str, Any]]]:
    origin = amendment["escrow_origin"]
    failed = source_parent / origin["failed_attempt_basename"]
    if failed.parent.resolve() != source_parent.resolve() or failed.name != origin["failed_attempt_basename"]:
        raise RuntimeError("recovery-origin basename escapes its canonical parent")
    observed = physical_tree_inventory(failed)
    if observed != origin["inventory"]:
        raise RuntimeError("recovery-origin physical whitelist differs from amendment")
    required_hashes = {
        ESCROW_NAME: origin["escrow_sha256"],
        FREEZE_NAME: origin["pre_generation_freeze_sha256"],
        "FAILURE.json": origin["failure_sha256"],
        "benchmark/manifest.json": origin["benchmark_manifest_sha256"],
    }
    for relative, expected in required_hashes.items():
        if sha256_file(failed / relative) != expected:
            raise RuntimeError(f"recovery-origin hash differs for {relative}")
    failure = json.loads((failed / "FAILURE.json").read_text(encoding="utf-8"))
    if failure != {
        "error_type": "RuntimeError",
        "escrow_present": True,
        "message": "fresh benchmark pair-token count differs from prospective freeze",
        "redraw_forbidden_if_escrow_present": True,
    }:
        raise RuntimeError("recovery-origin FAILURE.json is not the authorized pre-oracle failure")
    escrow = read_escrow(failed)
    freeze = json.loads((failed / FREEZE_NAME).read_text(encoding="utf-8"))
    if freeze != public_freeze_from_escrow(escrow):
        raise RuntimeError("recovery-origin public freeze differs from escrow")
    benchmark = failed / "benchmark"
    validate_manifest(benchmark)
    protocol = ProtocolConfig.from_dict(
        json.loads((benchmark / "protocol_config.json").read_text(encoding="utf-8"))
    )
    validate_visible_package(benchmark, protocol)
    validate_semantic_attestation(benchmark, trusted_public_key_path)
    expected_counts = amendment["population_contract"]["counts_by_split"]
    actual_counts = {
        split: sealed_population_counts(benchmark / "sealed" / f"{split}.jsonl")
        for split in SPLITS
    }
    if actual_counts != expected_counts:
        raise RuntimeError("recovery-origin token populations differ from amendment")
    return failed, observed


def _validate_wave56_recovery_amendment(
    amendment_path: Path,
    source: Path,
    execution_contract: dict[str, Any],
    mode: str,
    *,
    repo_root: Path = REPO_ROOT,
    trusted_public_key_path: Path = PUBLIC_KEY,
) -> dict[str, Any]:
    if mode not in {"recovery", "replay"}:
        raise RuntimeError("pre-oracle amendment is valid only for recovery or replay")
    source_metadata = source.lstat()
    if stat.S_ISLNK(source_metadata.st_mode) or not stat.S_ISDIR(source_metadata.st_mode):
        raise RuntimeError("recovery source must be one physical directory, not a symlink")
    canonical_path = repo_root / RECOVERY_AMENDMENT_RELATIVE
    if amendment_path.resolve(strict=True) != canonical_path.resolve(strict=True):
        raise RuntimeError("recovery amendment must use its canonical repository path")
    path, amendment_sha256 = require_repo_artifact(
        repo_root, RECOVERY_AMENDMENT_RELATIVE
    )
    amendment = json.loads(path.read_text(encoding="utf-8"))
    if sha256_file(path) != canonical_json_sha256(amendment):
        raise RuntimeError("recovery amendment is not canonical pretty JSON")
    _require_keys(
        amendment,
        {
            "schema_version",
            "status",
            "plan",
            "plan_audit",
            "implementation",
            "implementation_audit",
            "final_audit_path",
            "escrow_origin",
            "population_contract",
            "assertions",
        },
        "recovery amendment",
    )
    if amendment["schema_version"] != RECOVERY_AMENDMENT_SCHEMA:
        raise RuntimeError("recovery amendment schema differs")
    if amendment["status"] != "APPROVED_PREORACLE_RECOVERY":
        raise RuntimeError("recovery amendment is not approved")
    if amendment["assertions"] != {
        "no_redraw": True,
        "no_inference_in_origin": True,
        "no_oracle_in_origin": True,
        "no_labels_in_origin": True,
    }:
        raise RuntimeError("recovery amendment assertions differ")
    population_contract = amendment["population_contract"]
    _require_keys(
        population_contract,
        {"eligibility_predicate", "counts_by_split"},
        "population contract",
    )
    if population_contract["eligibility_predicate"] != {
        "is_out_of_catalog": False,
        "calibration_population": "canonical_preserving",
        "filter_rows_before_deduplicating_pair_tokens": True,
    }:
        raise RuntimeError("recovery eligibility predicate differs")
    counts_by_split = population_contract["counts_by_split"]
    if set(counts_by_split) != set(SPLITS):
        raise RuntimeError("recovery population split set differs")
    population_count_keys = {
        "rows",
        "total_unique_pair_tokens",
        "eligible_unique_pair_tokens",
        "out_of_catalog_unique_pair_tokens",
        "noncanonical_unique_pair_tokens",
        "eligible_intersection_noncanonical_unique_pair_tokens",
    }
    for split in SPLITS:
        _require_keys(
            counts_by_split[split], population_count_keys, f"population counts for {split}"
        )

    plan = amendment["plan"]
    _require_keys(plan, {"commit", "path", "sha256"}, "recovery plan")
    if plan["path"] != RECOVERY_PLAN_RELATIVE:
        raise RuntimeError("recovery plan path differs from the frozen canonical plan")
    require_repo_artifact(repo_root, plan["path"], plan["sha256"])
    plan_commit = git_introduction_commit(repo_root, plan["path"])
    if plan["commit"] != plan_commit:
        raise RuntimeError("recovery plan commit differs from its introduction")
    if git_changed_paths(repo_root, plan_commit) != {plan["path"]}:
        raise RuntimeError("recovery plan commit contains unrelated paths")

    plan_audit = amendment["plan_audit"]
    _require_keys(plan_audit, {"commit", "path", "sha256"}, "recovery plan audit")
    require_audit_report_path(plan_audit["path"], "recovery plan audit")
    plan_audit_path, _ = require_repo_artifact(
        repo_root, plan_audit["path"], plan_audit["sha256"]
    )
    _require_report_fields(
        plan_audit_path,
        [
            f"**Plan commit:** `{plan_commit}`",
            f"**Plan SHA-256:** `{plan['sha256']}`",
            "**Result:** `PASS`",
        ],
        "recovery plan audit",
    )
    plan_audit_commit = git_introduction_commit(repo_root, plan_audit["path"])
    if plan_audit["commit"] != plan_audit_commit:
        raise RuntimeError("recovery plan-audit commit differs from its introduction")
    if git_changed_paths(repo_root, plan_audit_commit) != {plan_audit["path"]}:
        raise RuntimeError("recovery plan-audit commit contains unrelated paths")
    require_direct_parent(repo_root, plan_audit_commit, plan_commit, "recovery plan audit")

    implementation = amendment["implementation"]
    _require_keys(
        implementation,
        {
            "runner_commit",
            "authority_commit",
            "coverage_commit",
            "commit",
            "preparer",
            "runner",
            "test",
        },
        "recovery implementation",
    )
    runner_commit = implementation["runner_commit"]
    if runner_commit != PHASE_ENTRY_IMPLEMENTATION_COMMIT:
        raise RuntimeError("runner implementation commit differs from the frozen plan")
    authority_commit = implementation["authority_commit"]
    if authority_commit != AUTHORITY_IMPLEMENTATION_COMMIT:
        raise RuntimeError("authority implementation commit differs from the frozen plan")
    coverage_commit = implementation["coverage_commit"]
    if coverage_commit != COVERAGE_IMPLEMENTATION_COMMIT:
        raise RuntimeError("coverage implementation commit differs from the frozen plan")
    implementation_commit = implementation["commit"]
    head = _git_output(repo_root, "rev-parse", "HEAD")
    require_ancestor(repo_root, implementation_commit, head)
    require_ancestor(repo_root, runner_commit, authority_commit)
    require_ancestor(repo_root, authority_commit, coverage_commit)
    require_ancestor(repo_root, coverage_commit, plan_commit)
    runner_lineage = _git_output(
        repo_root, "rev-list", "--parents", "-n", "1", runner_commit
    ).split()
    if len(runner_lineage) != 2:
        raise RuntimeError("runner implementation commit must have exactly one parent")
    if git_changed_paths(repo_root, runner_commit) != {
        PREPARER_RELATIVE,
        RUNNER_RELATIVE,
        RECOVERY_TEST_RELATIVE,
    }:
        raise RuntimeError("runner implementation commit changed unexpected paths")
    if git_changed_paths(repo_root, authority_commit) != {
        PREPARER_RELATIVE,
        RECOVERY_TEST_RELATIVE,
    }:
        raise RuntimeError("authority implementation commit changed unexpected paths")
    if git_changed_paths(repo_root, coverage_commit) != {
        PREPARER_RELATIVE,
        RECOVERY_TEST_RELATIVE,
    }:
        raise RuntimeError("coverage implementation commit changed unexpected paths")
    if git_blob_sha256(repo_root, runner_commit, RUNNER_RELATIVE) != git_blob_sha256(
        repo_root, authority_commit, RUNNER_RELATIVE
    ):
        raise RuntimeError("runner changed in authority implementation commit")
    if git_blob_sha256(repo_root, authority_commit, RUNNER_RELATIVE) != git_blob_sha256(
        repo_root, coverage_commit, RUNNER_RELATIVE
    ):
        raise RuntimeError("runner changed in coverage implementation commit")
    require_direct_parent(
        repo_root, implementation_commit, plan_audit_commit, "recovery implementation"
    )
    preparer = implementation["preparer"]
    runner = implementation["runner"]
    test = implementation["test"]
    _require_keys(preparer, {"path", "old_sha256", "new_sha256"}, "preparer delta")
    _require_keys(runner, {"path", "old_sha256", "new_sha256"}, "runner delta")
    _require_keys(test, {"path", "sha256"}, "recovery test")
    if test["path"] != RECOVERY_TEST_RELATIVE:
        raise RuntimeError("recovery test path differs")
    if git_changed_paths(repo_root, implementation_commit) != {
        PREPARER_RELATIVE,
        RECOVERY_TEST_RELATIVE,
    }:
        raise RuntimeError(
            "implementation commit contains files outside preparer and recovery test"
        )
    if git_blob_sha256(repo_root, implementation_commit, PREPARER_RELATIVE) != preparer["new_sha256"]:
        raise RuntimeError("implementation commit preparer blob differs from amendment")
    if git_blob_sha256(repo_root, implementation_commit, RECOVERY_TEST_RELATIVE) != test["sha256"]:
        raise RuntimeError("implementation commit test blob differs from amendment")
    if git_blob_sha256(repo_root, runner_commit, RUNNER_RELATIVE) != runner["new_sha256"]:
        raise RuntimeError("runner implementation blob differs from amendment")
    if git_blob_sha256(repo_root, implementation_commit, RUNNER_RELATIVE) != runner["new_sha256"]:
        raise RuntimeError("runner changed after its audited implementation commit")
    require_repo_artifact(repo_root, PREPARER_RELATIVE, preparer["new_sha256"])
    require_repo_artifact(repo_root, RUNNER_RELATIVE, runner["new_sha256"])
    require_repo_artifact(repo_root, RECOVERY_TEST_RELATIVE, test["sha256"])

    implementation_audit = amendment["implementation_audit"]
    _require_keys(implementation_audit, {"path", "sha256"}, "implementation audit")
    require_audit_report_path(implementation_audit["path"], "implementation audit")
    audit_path, _ = require_repo_artifact(
        repo_root, implementation_audit["path"], implementation_audit["sha256"]
    )
    _require_report_fields(
        audit_path,
        [
            f"**Implementation commit:** `{implementation_commit}`",
            f"**Runner commit:** `{runner_commit}`",
            f"**Authority commit:** `{authority_commit}`",
            f"**Coverage commit:** `{coverage_commit}`",
            f"**Preparer SHA-256:** `{preparer['new_sha256']}`",
            f"**Runner SHA-256:** `{runner['new_sha256']}`",
            f"**Test SHA-256:** `{test['sha256']}`",
            "**Result:** `PASS`",
        ],
        "implementation audit",
    )
    audit_commit = git_introduction_commit(repo_root, implementation_audit["path"])
    amendment_commit = git_introduction_commit(repo_root, RECOVERY_AMENDMENT_RELATIVE)
    final_path_relative = amendment["final_audit_path"]
    require_audit_report_path(final_path_relative, "final audit")
    if final_path_relative == implementation_audit["path"]:
        raise RuntimeError("implementation and final audits must use distinct reports")
    final_path, final_sha256 = require_repo_artifact(repo_root, final_path_relative)
    final_commit = git_introduction_commit(repo_root, final_path_relative)
    if git_changed_paths(repo_root, audit_commit) != {implementation_audit["path"]}:
        raise RuntimeError("implementation-audit commit contains unrelated paths")
    if git_changed_paths(repo_root, amendment_commit) != {RECOVERY_AMENDMENT_RELATIVE}:
        raise RuntimeError("amendment commit contains unrelated paths")
    if git_changed_paths(repo_root, final_commit) != {final_path_relative}:
        raise RuntimeError("final-audit commit contains unrelated paths")
    require_direct_parent(
        repo_root, audit_commit, implementation_commit, "implementation-audit"
    )
    require_direct_parent(repo_root, amendment_commit, audit_commit, "amendment")
    require_direct_parent(repo_root, final_commit, amendment_commit, "final-audit")
    if final_commit != head:
        raise RuntimeError("execution HEAD must be exactly the final-audit commit")
    if git_blob_sha256(repo_root, amendment_commit, RECOVERY_AMENDMENT_RELATIVE) != amendment_sha256:
        raise RuntimeError("amendment blob changed after its canonical commit")
    _require_report_fields(
        final_path,
        [
            f"**Audited package commit:** `{amendment_commit}`",
            f"**Amendment SHA-256:** `{amendment_sha256}`",
            "**Result:** `PASS`",
        ],
        "final audit",
    )
    allowed_after_implementation = {
        implementation_audit["path"],
        RECOVERY_AMENDMENT_RELATIVE,
        final_path_relative,
    }
    changed_after_implementation = {
        line
        for line in _git_output(
            repo_root, "diff", "--name-only", f"{implementation_commit}..{head}"
        ).splitlines()
        if line
    }
    if changed_after_implementation != allowed_after_implementation:
        raise RuntimeError("post-implementation commits changed unauthorized paths")
    if _git_output(repo_root, "status", "--porcelain"):
        raise RuntimeError("recovery requires a globally clean worktree")

    origin = amendment["escrow_origin"]
    _require_keys(
        origin,
        {
            "failed_attempt_basename",
            "contract_git_commit",
            "contract_sha256",
            "escrow_sha256",
            "pre_generation_freeze_sha256",
            "failure_sha256",
            "benchmark_manifest_sha256",
            "inventory",
        },
        "escrow origin",
    )
    require_ancestor(repo_root, origin["contract_git_commit"], implementation_commit)
    if (
        git_blob_sha256(repo_root, origin["contract_git_commit"], PREPARER_RELATIVE)
        != implementation["preparer"]["old_sha256"]
    ):
        raise RuntimeError("escrow-origin preparer blob differs from approved old hash")
    failed, inventory = validate_failed_recovery_origin(
        amendment, source.parent.resolve(), trusted_public_key_path
    )
    if mode == "recovery" and source.resolve(strict=True) != failed.resolve(strict=True):
        raise RuntimeError("recovery must source the exact failed attempt in the amendment")
    if mode == "replay":
        copied = source / RECOVERY_AMENDMENT_COPY_NAME
        if copied.is_symlink() or not copied.is_file() or sha256_file(copied) != amendment_sha256:
            raise RuntimeError("replay primary lacks the exact approved amendment copy")
        copied_stat = copied.stat()
        if stat.S_IMODE(copied_stat.st_mode) != 0o644 or copied_stat.st_uid != 0:
            raise PermissionError("replay primary amendment copy must be root-owned mode 0644")
        source_freeze_path = source / "preparation_freeze.json"
        if source_freeze_path.is_symlink() or not source_freeze_path.is_file():
            raise RuntimeError("replay primary preparation freeze is not one regular file")
        source_freeze_stat = source_freeze_path.stat()
        if stat.S_IMODE(source_freeze_stat.st_mode) != 0o644 or source_freeze_stat.st_uid != 0:
            raise PermissionError("replay primary preparation freeze must be root-owned mode 0644")
        source_freeze = json.loads(source_freeze_path.read_text(encoding="utf-8"))
        if source_freeze.get("recovery_provenance", {}).get("amendment_sha256") != amendment_sha256:
            raise RuntimeError("replay primary is not bound to the approved amendment")

    escrow = read_escrow(source)
    _validate_contract_delta(escrow["contract"], execution_contract, amendment, repo_root)
    if sha256_file(failed / ESCROW_NAME) != origin["escrow_sha256"]:
        raise RuntimeError("failed-attempt escrow changed after origin validation")
    return {
        "amendment": amendment,
        "amendment_sha256": amendment_sha256,
        "amendment_path": RECOVERY_AMENDMENT_RELATIVE,
        "implementation_commit": implementation_commit,
        "implementation_audit": implementation_audit,
        "final_audit": {"path": final_path_relative, "sha256": final_sha256},
        "escrow_origin_contract_sha256": origin["contract_sha256"],
        "failed_attempt": failed,
        "failed_attempt_basename": failed.name,
        "benchmark_manifest_sha256": origin["benchmark_manifest_sha256"],
        "origin_inventory": inventory,
        "repo_root": repo_root,
    }


def _validate_wave57_recovery_amendment(
    amendment_path: Path,
    source: Path,
    execution_contract: dict[str, Any],
    mode: str,
    *,
    repo_root: Path = REPO_ROOT,
    trusted_public_key_path: Path = PUBLIC_KEY,
) -> dict[str, Any]:
    if mode not in {"recovery", "replay"}:
        raise RuntimeError("Wave 57 amendment is valid only for recovery or replay")
    source_metadata = source.lstat()
    if stat.S_ISLNK(source_metadata.st_mode) or not stat.S_ISDIR(source_metadata.st_mode):
        raise RuntimeError("Wave 57 recovery source must be one physical directory")
    canonical_path = repo_root / WAVE57_RECOVERY_AMENDMENT_RELATIVE
    if amendment_path.resolve(strict=True) != canonical_path.resolve(strict=True):
        raise RuntimeError("Wave 57 amendment must use its canonical repository path")
    path, amendment_sha256 = require_repo_artifact(
        repo_root, WAVE57_RECOVERY_AMENDMENT_RELATIVE
    )
    amendment = json.loads(path.read_text(encoding="utf-8"))
    if sha256_file(path) != canonical_json_sha256(amendment):
        raise RuntimeError("Wave 57 recovery amendment is not canonical pretty JSON")
    _require_keys(
        amendment,
        {
            "schema_version",
            "status",
            "plan",
            "plan_audit",
            "implementation",
            "implementation_audit",
            "final_audit_path",
            "escrow_origin",
            "population_contract",
            "assertions",
        },
        "Wave 57 recovery amendment",
    )
    if amendment["schema_version"] != WAVE57_RECOVERY_AMENDMENT_SCHEMA:
        raise RuntimeError("Wave 57 recovery amendment schema differs")
    if amendment["status"] != "APPROVED_PREORACLE_RECOVERY":
        raise RuntimeError("Wave 57 recovery amendment is not approved")
    if amendment["assertions"] != {
        "no_redraw": True,
        "no_inference_in_origin": True,
        "no_materialized_oracle_in_origin": True,
        "no_authorized_labels_in_origin": True,
    }:
        raise RuntimeError("Wave 57 recovery amendment assertions differ")
    population_contract = amendment["population_contract"]
    _require_keys(
        population_contract,
        {"eligibility_predicate", "counts_by_split"},
        "Wave 57 population contract",
    )
    if population_contract["eligibility_predicate"] != {
        "is_out_of_catalog": False,
        "calibration_population": "canonical_preserving",
        "filter_rows_before_deduplicating_pair_tokens": True,
    }:
        raise RuntimeError("Wave 57 recovery eligibility predicate differs")
    expected_population = {
        "rows": 4992,
        "total_unique_pair_tokens": 1152,
        "eligible_unique_pair_tokens": 768,
        "out_of_catalog_unique_pair_tokens": 384,
        "noncanonical_unique_pair_tokens": 192,
        "eligible_intersection_noncanonical_unique_pair_tokens": 192,
    }
    counts_by_split = population_contract["counts_by_split"]
    if counts_by_split != {split: expected_population for split in SPLITS}:
        raise RuntimeError("Wave 57 recovery populations differ from the frozen draw")

    plan = amendment["plan"]
    _require_keys(plan, {"commit", "path", "sha256"}, "Wave 57 recovery plan")
    if plan["path"] != WAVE57_RECOVERY_PLAN_RELATIVE:
        raise RuntimeError("Wave 57 recovery plan path differs")
    require_repo_artifact(repo_root, plan["path"], plan["sha256"])
    plan_commit = plan["commit"]
    if git_changed_paths(repo_root, plan_commit) != {plan["path"]}:
        raise RuntimeError("Wave 57 recovery-plan commit contains unrelated paths")
    if git_blob_sha256(repo_root, plan_commit, plan["path"]) != plan["sha256"]:
        raise RuntimeError("Wave 57 recovery-plan blob differs from amendment")

    plan_audit = amendment["plan_audit"]
    _require_keys(plan_audit, {"commit", "path", "sha256"}, "Wave 57 plan audit")
    require_audit_report_path(plan_audit["path"], "Wave 57 plan audit")
    plan_audit_path, _ = require_repo_artifact(
        repo_root, plan_audit["path"], plan_audit["sha256"]
    )
    _require_report_fields(
        plan_audit_path,
        [
            f"**Plan commit:** `{plan_commit}`",
            f"**Plan SHA:** `{plan['sha256']}`",
            "**Result:** `PASS`",
        ],
        "Wave 57 plan audit",
        allow_one_terminal_blank=True,
    )
    plan_audit_commit = plan_audit["commit"]
    if git_introduction_commit(repo_root, plan_audit["path"]) != plan_audit_commit:
        raise RuntimeError("Wave 57 plan-audit commit differs from its introduction")
    if git_changed_paths(repo_root, plan_audit_commit) != {plan_audit["path"]}:
        raise RuntimeError("Wave 57 plan-audit commit contains unrelated paths")
    require_direct_parent(
        repo_root, plan_audit_commit, plan_commit, "Wave 57 plan audit"
    )

    implementation = amendment["implementation"]
    _require_keys(
        implementation,
        {"commit", "preparer", "test"},
        "Wave 57 recovery implementation",
    )
    implementation_commit = implementation["commit"]
    require_direct_parent(
        repo_root,
        implementation_commit,
        plan_audit_commit,
        "Wave 57 recovery implementation",
    )
    if git_changed_paths(repo_root, implementation_commit) != {
        PREPARER_RELATIVE,
        WAVE57_RECOVERY_TEST_RELATIVE,
    }:
        raise RuntimeError("Wave 57 implementation commit changed unauthorized paths")
    for label, relative in (
        ("preparer", PREPARER_RELATIVE),
        ("test", WAVE57_RECOVERY_TEST_RELATIVE),
    ):
        delta = implementation[label]
        _require_keys(delta, {"path", "old_sha256", "new_sha256"}, f"Wave 57 {label}")
        if delta["path"] != relative:
            raise RuntimeError(f"Wave 57 {label} path differs")
        if git_blob_sha256(repo_root, implementation_commit, relative) != delta["new_sha256"]:
            raise RuntimeError(f"Wave 57 {label} implementation blob differs")
        require_repo_artifact(repo_root, relative, delta["new_sha256"])

    implementation_audit = amendment["implementation_audit"]
    _require_keys(
        implementation_audit,
        {"commit", "path", "sha256"},
        "Wave 57 implementation audit",
    )
    require_audit_report_path(
        implementation_audit["path"], "Wave 57 implementation audit"
    )
    implementation_audit_path, _ = require_repo_artifact(
        repo_root, implementation_audit["path"], implementation_audit["sha256"]
    )
    _require_report_fields(
        implementation_audit_path,
        [
            f"**Implementation commit:** `{implementation_commit}`",
            f"**Preparer SHA-256:** `{implementation['preparer']['new_sha256']}`",
            f"**Test SHA-256:** `{implementation['test']['new_sha256']}`",
            "**Result:** `PASS`",
        ],
        "Wave 57 implementation audit",
        allow_one_terminal_blank=True,
    )
    implementation_audit_commit = implementation_audit["commit"]
    if (
        git_introduction_commit(repo_root, implementation_audit["path"])
        != implementation_audit_commit
    ):
        raise RuntimeError("Wave 57 implementation-audit commit differs")
    if git_changed_paths(repo_root, implementation_audit_commit) != {
        implementation_audit["path"]
    }:
        raise RuntimeError("Wave 57 implementation-audit commit contains unrelated paths")
    require_direct_parent(
        repo_root,
        implementation_audit_commit,
        implementation_commit,
        "Wave 57 implementation audit",
    )

    amendment_commit = git_introduction_commit(
        repo_root, WAVE57_RECOVERY_AMENDMENT_RELATIVE
    )
    if git_changed_paths(repo_root, amendment_commit) != {
        WAVE57_RECOVERY_AMENDMENT_RELATIVE
    }:
        raise RuntimeError("Wave 57 amendment commit contains unrelated paths")
    require_direct_parent(
        repo_root,
        amendment_commit,
        implementation_audit_commit,
        "Wave 57 amendment",
    )
    if (
        git_blob_sha256(repo_root, amendment_commit, WAVE57_RECOVERY_AMENDMENT_RELATIVE)
        != amendment_sha256
    ):
        raise RuntimeError("Wave 57 amendment blob changed after its commit")

    final_path_relative = amendment["final_audit_path"]
    require_audit_report_path(final_path_relative, "Wave 57 final audit")
    if final_path_relative in {plan_audit["path"], implementation_audit["path"]}:
        raise RuntimeError("Wave 57 audits must use distinct reports")
    final_path, final_sha256 = require_repo_artifact(repo_root, final_path_relative)
    final_commit = git_introduction_commit(repo_root, final_path_relative)
    if git_changed_paths(repo_root, final_commit) != {final_path_relative}:
        raise RuntimeError("Wave 57 final-audit commit contains unrelated paths")
    require_direct_parent(
        repo_root, final_commit, amendment_commit, "Wave 57 final audit"
    )
    head = _git_output(repo_root, "rev-parse", "HEAD")
    if final_commit != head:
        raise RuntimeError("Wave 57 execution HEAD must be exactly the final-audit commit")
    _require_report_fields(
        final_path,
        [
            f"**Audited package commit:** `{amendment_commit}`",
            f"**Amendment SHA-256:** `{amendment_sha256}`",
            "**Result:** `PASS`",
        ],
        "Wave 57 final audit",
        allow_one_terminal_blank=True,
    )
    allowed_after_implementation = {
        implementation_audit["path"],
        WAVE57_RECOVERY_AMENDMENT_RELATIVE,
        final_path_relative,
    }
    changed_after_implementation = {
        line
        for line in _git_output(
            repo_root, "diff", "--name-only", f"{implementation_commit}..{head}"
        ).splitlines()
        if line
    }
    if changed_after_implementation != allowed_after_implementation:
        raise RuntimeError("Wave 57 post-implementation commits changed unauthorized paths")
    if _git_output(repo_root, "status", "--porcelain"):
        raise RuntimeError("Wave 57 recovery requires a globally clean worktree")

    origin = amendment["escrow_origin"]
    _require_keys(
        origin,
        {
            "failed_attempt_basename",
            "contract_git_commit",
            "contract_sha256",
            "escrow_sha256",
            "pre_generation_freeze_sha256",
            "failure_sha256",
            "benchmark_manifest_sha256",
            "inventory",
        },
        "Wave 57 escrow origin",
    )
    require_ancestor(repo_root, origin["contract_git_commit"], implementation_commit)
    for label, relative in (
        ("preparer", PREPARER_RELATIVE),
        ("test", WAVE57_RECOVERY_TEST_RELATIVE),
    ):
        if (
            git_blob_sha256(repo_root, origin["contract_git_commit"], relative)
            != implementation[label]["old_sha256"]
        ):
            raise RuntimeError(f"Wave 57 escrow-origin {label} blob differs")

    failed, inventory, public_contract = validate_wave57_failed_origin_content_blind(
        amendment,
        source.parent.resolve(),
        execution_contract,
        trusted_public_key_path,
        repo_root=repo_root,
    )
    if mode == "recovery" and source.resolve(strict=True) != failed.resolve(strict=True):
        raise RuntimeError("Wave 57 recovery must source the exact failed attempt")
    if mode == "replay":
        copied = source / RECOVERY_AMENDMENT_COPY_NAME
        if copied.is_symlink() or not copied.is_file() or sha256_file(copied) != amendment_sha256:
            raise RuntimeError("Wave 57 replay primary lacks the approved amendment copy")
        copied_stat = copied.stat()
        if stat.S_IMODE(copied_stat.st_mode) != 0o644 or copied_stat.st_uid != 0:
            raise PermissionError("Wave 57 replay amendment copy must be root-owned mode 0644")
        source_freeze_path = source / "preparation_freeze.json"
        if source_freeze_path.is_symlink() or not source_freeze_path.is_file():
            raise RuntimeError("Wave 57 replay primary preparation freeze is invalid")
        source_freeze_stat = source_freeze_path.stat()
        if stat.S_IMODE(source_freeze_stat.st_mode) != 0o644 or source_freeze_stat.st_uid != 0:
            raise PermissionError("Wave 57 replay freeze must be root-owned mode 0644")
        source_freeze = json.loads(source_freeze_path.read_text(encoding="utf-8"))
        if source_freeze.get("recovery_provenance", {}).get("amendment_sha256") != amendment_sha256:
            raise RuntimeError("Wave 57 replay primary is not bound to the amendment")

    validate_wave57_failed_origin_semantic(
        amendment, failed, trusted_public_key_path, inventory
    )
    source_escrow = read_escrow(source)
    _validate_wave57_contract_delta(
        source_escrow["contract"] if mode == "replay" else public_contract,
        execution_contract,
        amendment,
        repo_root,
    )
    if sha256_file(failed / ESCROW_NAME) != origin["escrow_sha256"]:
        raise RuntimeError("Wave 57 failed-attempt escrow changed after validation")
    return {
        "amendment": amendment,
        "amendment_sha256": amendment_sha256,
        "amendment_path": WAVE57_RECOVERY_AMENDMENT_RELATIVE,
        "implementation_commit": implementation_commit,
        "implementation_audit": implementation_audit,
        "final_audit": {"path": final_path_relative, "sha256": final_sha256},
        "escrow_origin_contract_sha256": origin["contract_sha256"],
        "failed_attempt": failed,
        "failed_attempt_basename": failed.name,
        "benchmark_manifest_sha256": origin["benchmark_manifest_sha256"],
        "origin_inventory": inventory,
        "repo_root": repo_root,
    }


def _validate_wave59_recovery_amendment(
    amendment_path: Path,
    source: Path,
    execution_contract: dict[str, Any],
    mode: str,
    *,
    repo_root: Path = REPO_ROOT,
    trusted_public_key_path: Path = PUBLIC_KEY,
) -> dict[str, Any]:
    if mode not in {"recovery", "replay"}:
        raise RuntimeError("Wave 59 amendment is valid only for recovery or replay")
    source_metadata = source.lstat()
    if stat.S_ISLNK(source_metadata.st_mode) or not stat.S_ISDIR(source_metadata.st_mode):
        raise RuntimeError("Wave 59 recovery source must be one physical directory")
    config = execution_contract["prospective_config"]
    required_sources = config["required_execution_sources"]
    config_relatives = [
        relative
        for relative in required_sources
        if relative.endswith("wave59_fresh_hgb_guard_bracket.json")
    ]
    if len(config_relatives) != 1:
        raise RuntimeError("Wave 59 recovery cannot identify its frozen config source")
    from geometria_proporcional.wave59_hgb_guard_bracket import (
        config_self_binding_sha256,
    )

    observed_bound = dict(execution_contract["sources"])
    config_relative = config_relatives[0]
    observed_bound[config_relative] = config_self_binding_sha256(
        config, config_relative
    )
    amendment, amendment_sha256 = validate_wave59_repository_recovery_authority(
        amendment_path,
        config["source_sha256"],
        observed_bound,
        repo_root=repo_root,
    )
    population_contract = amendment["population_contract"]
    _require_keys(
        population_contract,
        {"eligibility_predicate", "counts_by_split"},
        "Wave 59 population contract",
    )
    if population_contract["eligibility_predicate"] != {
        "is_out_of_catalog": False,
        "calibration_population": "canonical_preserving",
        "filter_rows_before_deduplicating_pair_tokens": True,
    }:
        raise RuntimeError("Wave 59 recovery eligibility predicate differs")
    expected_population = {
        "rows": 4992,
        "total_unique_pair_tokens": 1152,
        "eligible_unique_pair_tokens": 768,
        "out_of_catalog_unique_pair_tokens": 384,
        "noncanonical_unique_pair_tokens": 192,
        "eligible_intersection_noncanonical_unique_pair_tokens": 192,
    }
    if population_contract["counts_by_split"] != {
        split: expected_population for split in SPLITS
    }:
        raise RuntimeError("Wave 59 recovery populations differ from the frozen law")

    origin = amendment["escrow_origin"]
    _require_keys(
        origin,
        {
            "failed_attempt_basename",
            "contract_git_commit",
            "contract_sha256",
            "escrow_sha256",
            "pre_generation_freeze_sha256",
            "failure_sha256",
            "failure_inventory_sha256",
            "failure_attestation_sha256",
            "benchmark_manifest_sha256",
            "inventory",
        },
        "Wave 59 escrow origin",
    )
    implementation = amendment["implementation"]
    require_ancestor(repo_root, origin["contract_git_commit"], implementation["commit"])
    for label, relative in (
        ("preparer", PREPARER_RELATIVE),
        ("runner", WAVE59_RUNNER_RELATIVE),
        ("prospective_test", WAVE59_PROSPECTIVE_TEST_RELATIVE),
    ):
        if git_blob_sha256(
            repo_root, origin["contract_git_commit"], relative
        ) != implementation[label]["old_sha256"]:
            raise RuntimeError(f"Wave 59 escrow-origin {label} blob differs")

    failed, inventory, public_contract = validate_wave59_failed_origin_content_blind(
        amendment,
        source.parent.resolve(),
        execution_contract,
        trusted_public_key_path,
        repo_root=repo_root,
    )
    if mode == "recovery" and source.resolve(strict=True) != failed.resolve(strict=True):
        raise RuntimeError("Wave 59 recovery must source the exact failed attempt")
    if mode == "replay":
        copied = source / RECOVERY_AMENDMENT_COPY_NAME
        if copied.is_symlink() or not copied.is_file() or sha256_file(copied) != amendment_sha256:
            raise RuntimeError("Wave 59 replay primary lacks the approved amendment copy")
        copied_stat = copied.stat()
        if stat.S_IMODE(copied_stat.st_mode) != 0o644 or copied_stat.st_uid != 0:
            raise PermissionError("Wave 59 replay amendment copy must be root-owned mode 0644")
        source_freeze_path = source / "preparation_freeze.json"
        if source_freeze_path.is_symlink() or not source_freeze_path.is_file():
            raise RuntimeError("Wave 59 replay primary preparation freeze is invalid")
        source_freeze_stat = source_freeze_path.stat()
        if stat.S_IMODE(source_freeze_stat.st_mode) != 0o644 or source_freeze_stat.st_uid != 0:
            raise PermissionError("Wave 59 replay freeze must be root-owned mode 0644")
        source_freeze = json.loads(source_freeze_path.read_text(encoding="utf-8"))
        if source_freeze.get("recovery_provenance", {}).get("amendment_sha256") != amendment_sha256:
            raise RuntimeError("Wave 59 replay primary is not bound to the amendment")

    validate_wave59_failed_origin_semantic(
        amendment, failed, trusted_public_key_path, inventory
    )
    source_escrow = read_escrow(source)
    _validate_wave59_contract_delta(
        source_escrow["contract"] if mode == "replay" else public_contract,
        execution_contract,
        amendment,
        repo_root,
    )
    if sha256_file(failed / ESCROW_NAME) != origin["escrow_sha256"]:
        raise RuntimeError("Wave 59 failed-attempt escrow changed after validation")
    return {
        "amendment": amendment,
        "amendment_sha256": amendment_sha256,
        "amendment_path": WAVE59_RECOVERY_AMENDMENT_RELATIVE,
        "implementation_commit": implementation["commit"],
        "implementation_audit": amendment["implementation_audit"],
        "final_audit": {
            "path": amendment["final_audit_path"],
            "sha256": sha256_file(repo_root / amendment["final_audit_path"]),
        },
        "escrow_origin_contract_sha256": origin["contract_sha256"],
        "failed_attempt": failed,
        "failed_attempt_basename": failed.name,
        "benchmark_manifest_sha256": origin["benchmark_manifest_sha256"],
        "origin_inventory": inventory,
        "repo_root": repo_root,
    }


def _validate_wave60_recovery_origin(
    amendment: dict[str, Any],
    config: dict[str, Any],
    *,
    repo_root: Path,
    trusted_public_key_path: Path = PUBLIC_KEY,
) -> tuple[Path, list[dict[str, Any]]]:
    prior = require_canonical_repo_directory(
        repo_root,
        amendment["prior_attempt_container"],
        "Wave 60 prior attempt",
    )
    pair = prior / "pair"
    primary = prior / "primary"
    from run_wave60_frozen_policy_transport import validate_pair_failure_package

    pair_status = validate_pair_failure_package(
        prior, public_key=trusted_public_key_path
    )
    pair_failure = pair / "FAILURE.json"
    failure = json.loads(pair_failure.read_text(encoding="utf-8"))
    if (
        pair_status.get("terminal") != "PAIR_ABORTED_PRE_TRUTH"
        or pair_status.get("any_truth_accessed") is not False
        or pair_status.get("recovery_allowed") is not True
        or sha256_file(pair_failure) != amendment["prior_pair_failure_sha256"]
        or failure.get("schema_version") != "wave60-pair-failure-v1"
        or failure.get("terminal") != "PAIR_ABORTED_PRE_TRUTH"
        or failure.get("truth_accessed") is not False
        or failure.get("recovery_allowed") is not True
        or failure.get("authority_binding_sha256")
        != sha256_file(pair / "pair_status.json")
    ):
        raise RuntimeError("Wave 60 recovery origin is not an authorized pre-truth abort")
    pair_attestation = json.loads(
        (pair / "failure_attestation.json").read_text(encoding="utf-8")
    )
    verify_attestation(
        {
            "algorithm": "Ed25519",
            "payload": pair_attestation["payload"],
            "signature_base64": pair_attestation["signature_base64"],
            "trusted_public_key_sha256": pair_attestation[
                "public_key_fingerprint"
            ],
        },
        trusted_public_key_path.resolve(strict=True),
    )
    if (
        pair_attestation.get("schema_version")
        != "wave60-pair-failure-attestation-v1"
        or pair_attestation.get("payload", {}).get("pair_status_sha256")
        != sha256_file(pair / "pair_status.json")
        or pair_attestation.get("payload", {}).get("failure_sha256")
        != amendment["prior_pair_failure_sha256"]
    ):
        raise RuntimeError("Wave 60 pair failure attestation drifted")
    observed = physical_tree_inventory(primary)
    if observed != amendment["origin_inventory"]:
        raise RuntimeError("Wave 60 recovery origin inventory drifted")
    records = _inventory_records_by_path(observed)
    preserved = amendment["preserved_draw_sha256"]
    for relative, expected in preserved.items():
        record = records.get(relative)
        if record is None or record.get("sha256") != expected:
            raise RuntimeError(f"Wave 60 preserved draw drifted: {relative}")
    escrow = read_escrow(primary)
    if (
        records[ESCROW_NAME]["sha256"] != amendment["escrow_origin"]["escrow_sha256"]
        or records[FREEZE_NAME]["sha256"]
        != amendment["escrow_origin"]["pre_generation_freeze_sha256"]
        or records["benchmark/manifest.json"]["sha256"]
        != amendment["escrow_origin"]["benchmark_manifest_sha256"]
        or compact_json_sha256(escrow["contract"])
        != amendment["escrow_origin"]["contract_sha256"]
    ):
        raise RuntimeError("Wave 60 recovery escrow origin drifted")
    benchmark_manifest = json.loads(
        (primary / "benchmark/manifest.json").read_text(encoding="utf-8")
    )
    expected_draw = {
        ESCROW_NAME,
        FREEZE_NAME,
        "benchmark/manifest.json",
        *(f"benchmark/{relative}" for relative in benchmark_manifest["files"]),
    }
    if set(preserved) != expected_draw:
        raise RuntimeError("Wave 60 preserved draw map is not closed-world")
    if config["attempt"]["recovery"]["preserved_draw_sha256"] != preserved:
        raise RuntimeError("Wave 60 config and amendment draw maps differ")
    root_failure = json.loads((primary / "FAILURE.json").read_text(encoding="utf-8"))
    root_attestation_path = primary / "failure_attestation.json"
    root_attestation = json.loads(root_attestation_path.read_text(encoding="utf-8"))
    verify_attestation(
        {
            "algorithm": "Ed25519",
            "payload": root_attestation["payload"],
            "signature_base64": root_attestation["signature_base64"],
            "trusted_public_key_sha256": root_attestation[
                "public_key_fingerprint"
            ],
        },
        trusted_public_key_path.resolve(strict=True),
    )
    if (
        root_failure.get("truth_accessed") is not False
        or root_failure.get("recovery_allowed") is not True
        or root_attestation.get("payload", {}).get("failure_sha256")
        != sha256_file(primary / "FAILURE.json")
        or pair_status.get("primary_terminal_binding_sha256")
        != sha256_file(root_attestation_path)
    ):
        raise RuntimeError("Wave 60 primary failure binding drifted")
    return primary, observed


def _validate_wave60_recovery_amendment(
    amendment_path: Path,
    source: Path,
    execution_contract: dict[str, Any],
    mode: str,
    *,
    repo_root: Path = REPO_ROOT,
    trusted_public_key_path: Path = PUBLIC_KEY,
) -> dict[str, Any]:
    if mode not in {"recovery", "replay"}:
        raise RuntimeError("Wave 60 amendment is valid only for recovery or replay")
    config = execution_contract["prospective_config"]
    recovery = config["attempt"]["recovery"]
    canonical = (repo_root / recovery["amendment_path"]).resolve(strict=True)
    if amendment_path.resolve(strict=True) != canonical:
        raise RuntimeError("Wave 60 recovery amendment path drifted")
    if sha256_file(canonical) != recovery["amendment_sha256"]:
        raise RuntimeError("Wave 60 recovery amendment hash drifted")
    amendment = json.loads(canonical.read_text(encoding="utf-8"))
    _require_keys(
        amendment,
        {
            "schema_version",
            "status",
            "prior_attempt_container",
            "prior_pair_failure_sha256",
            "prior_config_audit",
            "escrow_origin",
            "preserved_draw_sha256",
            "population_contract",
            "origin_inventory",
        },
        "Wave 60 recovery amendment",
    )
    if (
        amendment["schema_version"] != WAVE60_RECOVERY_AMENDMENT_SCHEMA
        or amendment["status"] != "APPROVED"
        or amendment["prior_attempt_container"]
        != recovery["prior_attempt_container"]
        or amendment["prior_pair_failure_sha256"]
        != recovery["prior_pair_failure_sha256"]
        or amendment["prior_config_audit"]
        != {
            "commit": recovery["prior_config_audit_commit"],
            "path": recovery["prior_config_audit_path"],
            "sha256": recovery["prior_config_audit_sha256"],
        }
        or amendment["preserved_draw_sha256"]
        != recovery["preserved_draw_sha256"]
    ):
        raise RuntimeError("Wave 60 recovery amendment authority drifted")
    _require_keys(
        amendment["escrow_origin"],
        {
            "contract_sha256",
            "escrow_sha256",
            "pre_generation_freeze_sha256",
            "benchmark_manifest_sha256",
        },
        "Wave 60 escrow origin",
    )
    expected_population = {
        "rows": 4992,
        "total_unique_pair_tokens": 1152,
        "eligible_unique_pair_tokens": 768,
        "out_of_catalog_unique_pair_tokens": 384,
        "noncanonical_unique_pair_tokens": 192,
        "eligible_intersection_noncanonical_unique_pair_tokens": 192,
    }
    if amendment["population_contract"] != {
        "eligibility_predicate": {
            "is_out_of_catalog": False,
            "calibration_population": "canonical_preserving",
            "filter_rows_before_deduplicating_pair_tokens": True,
        },
        "counts_by_split": {split: expected_population for split in SPLITS},
    }:
        raise RuntimeError("Wave 60 recovery population contract drifted")
    config_relatives = [
        relative
        for relative in config["required_execution_sources"]
        if relative.endswith("wave60_frozen_policy_transport.json")
    ]
    if len(config_relatives) != 1:
        raise RuntimeError("Wave 60 recovery cannot identify its frozen config source")
    config_relative = config_relatives[0]
    prior_audit_binding = {
        "audit_commit": recovery["prior_config_audit_commit"],
        "audit_path": recovery["prior_config_audit_path"],
        "audit_sha256": recovery["prior_config_audit_sha256"],
    }
    prior_config_commit = _git_output(
        repo_root, "rev-parse", f"{recovery['prior_config_audit_commit']}^"
    )
    if git_changed_paths(repo_root, prior_config_commit) != {config_relative}:
        raise RuntimeError("Wave 60 prior config commit is not exclusive")
    prior_config_bytes = subprocess.check_output(
        ["git", "show", f"{prior_config_commit}:{config_relative}"],
        cwd=repo_root,
    )
    prior_config_sha256 = sha256_bytes(prior_config_bytes)
    try:
        prior_config = json.loads(prior_config_bytes)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Wave 60 prior config blob is not canonical JSON") from exc
    validate_prospective_config(prior_config)
    require_direct_parent(
        repo_root,
        prior_config_commit,
        prior_config["source_law_authority"]["audit_commit"],
        "Wave 60 prior config",
    )
    if (
        prior_config["attempt"]["container"]
        != recovery["prior_attempt_container"]
        or prior_config["final_audit"]["audit_path"]
        != recovery["prior_config_audit_path"]
    ):
        raise RuntimeError("Wave 60 prior config audit does not bind the recovered attempt")
    validate_wave60_audit_commit(
        repo_root,
        prior_audit_binding,
        scope="CONFIG",
        target={
            "config_commit": prior_config_commit,
            "config_sha256": prior_config_sha256,
        },
        expected_parent=prior_config_commit,
    )
    prior_audit_path, _ = require_repo_artifact(
        repo_root,
        recovery["prior_config_audit_path"],
        recovery["prior_config_audit_sha256"],
    )
    prior_audit_authority = parse_wave60_audit_report(
        prior_audit_path,
        scope="CONFIG",
        target={
            "config_commit": prior_config_commit,
            "config_sha256": prior_config_sha256,
        },
    )
    if prior_audit_authority["audit_id"] != prior_config["final_audit"]["audit_id"]:
        raise RuntimeError("Wave 60 prior config audit id drifted")

    amendment_commit = git_introduction_commit(
        repo_root, recovery["amendment_path"]
    )
    if git_changed_paths(repo_root, amendment_commit) != {
        recovery["amendment_path"]
    }:
        raise RuntimeError("Wave 60 recovery amendment commit is not exclusive")
    require_direct_parent(
        repo_root,
        amendment_commit,
        recovery["prior_config_audit_commit"],
        "Wave 60 recovery amendment",
    )
    audit_binding = {
        "audit_commit": recovery["amendment_audit_commit"],
        "audit_path": recovery["amendment_audit_path"],
        "audit_sha256": recovery["amendment_audit_sha256"],
    }
    validate_wave60_audit_commit(
        repo_root,
        audit_binding,
        scope="RECOVERY_AMENDMENT",
        target={"amendment_sha256": recovery["amendment_sha256"]},
        expected_parent=amendment_commit,
    )
    require_ancestor(
        repo_root,
        recovery["amendment_audit_commit"],
        execution_contract["git_commit"],
    )
    failed, observed = _validate_wave60_recovery_origin(
        amendment,
        config,
        repo_root=repo_root,
        trusted_public_key_path=trusted_public_key_path,
    )
    expected_source = (
        failed
        if mode == "recovery"
        else (repo_root / config["primary_output"]).resolve(strict=True)
    )
    if source.resolve(strict=True) != expected_source:
        raise RuntimeError("Wave 60 recovery/replay source is not canonical")
    if mode == "replay":
        source_freeze = json.loads(
            (source / "preparation_freeze.json").read_text(encoding="utf-8")
        )
        if (
            source_freeze.get("recovery_provenance", {}).get("amendment_sha256")
            != recovery["amendment_sha256"]
        ):
            raise RuntimeError("Wave 60 replay primary lacks recovery provenance")
    origin_contract = read_escrow(failed)["contract"]
    if (
        origin_contract.get("prospective_config") != prior_config
        or origin_contract.get("config_sha256") != prior_config_sha256
    ):
        raise RuntimeError("Wave 60 recovered escrow is not bound to the prior config")
    _validate_wave60_contract_delta(
        origin_contract, execution_contract, amendment, repo_root
    )
    return {
        "amendment": amendment,
        "amendment_sha256": recovery["amendment_sha256"],
        "amendment_path": recovery["amendment_path"],
        "implementation_commit": config["implementation_binding"]["commit"],
        "implementation_audit": config["implementation_binding"],
        "final_audit": config.get("final_audit"),
        "escrow_origin_contract_sha256": amendment["escrow_origin"][
            "contract_sha256"
        ],
        "failed_attempt": failed,
        "failed_attempt_basename": failed.name,
        "benchmark_manifest_sha256": amendment["escrow_origin"][
            "benchmark_manifest_sha256"
        ],
        "origin_inventory": observed,
        "reuse_source": source.resolve(strict=True),
        "repo_root": repo_root,
    }


def validate_recovery_amendment(
    amendment_path: Path,
    source: Path,
    execution_contract: dict[str, Any],
    mode: str,
    *,
    repo_root: Path = REPO_ROOT,
    trusted_public_key_path: Path = PUBLIC_KEY,
) -> dict[str, Any]:
    schema = execution_contract.get("prospective_config", {}).get("schema_version")
    if schema == WAVE57_CONFIG_SCHEMA:
        return _validate_wave57_recovery_amendment(
            amendment_path,
            source,
            execution_contract,
            mode,
            repo_root=repo_root,
            trusted_public_key_path=trusted_public_key_path,
        )
    if schema == WAVE59_CONFIG_SCHEMA:
        return _validate_wave59_recovery_amendment(
            amendment_path,
            source,
            execution_contract,
            mode,
            repo_root=repo_root,
            trusted_public_key_path=trusted_public_key_path,
        )
    if schema == WAVE60_CONFIG_SCHEMA:
        return _validate_wave60_recovery_amendment(
            amendment_path,
            source,
            execution_contract,
            mode,
            repo_root=repo_root,
            trusted_public_key_path=trusted_public_key_path,
        )
    return _validate_wave56_recovery_amendment(
        amendment_path,
        source,
        execution_contract,
        mode,
        repo_root=repo_root,
        trusted_public_key_path=trusted_public_key_path,
    )


def revalidate_authorized_recovery_origin(
    context: dict[str, Any],
    execution_contract: dict[str, Any],
    trusted_public_key_path: Path,
) -> tuple[Path, list[dict[str, Any]]]:
    amendment = context["amendment"]
    schema = amendment.get("schema_version")
    if schema == WAVE60_RECOVERY_AMENDMENT_SCHEMA:
        return _validate_wave60_recovery_origin(
            amendment,
            execution_contract["prospective_config"],
            repo_root=context["repo_root"],
            trusted_public_key_path=trusted_public_key_path,
        )
    if schema == WAVE59_RECOVERY_AMENDMENT_SCHEMA:
        failed, inventory, _ = validate_wave59_failed_origin_content_blind(
            amendment,
            context["failed_attempt"].parent,
            execution_contract,
            trusted_public_key_path,
            repo_root=context["repo_root"],
        )
        validate_wave59_failed_origin_semantic(
            amendment, failed, trusted_public_key_path, inventory
        )
        return failed, inventory
    if schema != WAVE57_RECOVERY_AMENDMENT_SCHEMA:
        return validate_failed_recovery_origin(
            amendment,
            context["failed_attempt"].parent,
            trusted_public_key_path,
        )
    failed, inventory, _ = validate_wave57_failed_origin_content_blind(
        amendment,
        context["failed_attempt"].parent,
        execution_contract,
        trusted_public_key_path,
        repo_root=context["repo_root"],
    )
    validate_wave57_failed_origin_semantic(
        amendment, failed, trusted_public_key_path, inventory
    )
    return failed, inventory


def validate_reused_escrow(
    source: Path,
    contract: dict[str, Any],
    recovery_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    escrow = read_escrow(source)
    if escrow.get("contract") != contract:
        if recovery_context is None:
            raise RuntimeError("replay/recovery commit, config, sources, or bindings differ from escrow")
        amendment = recovery_context["amendment"]
        _validate_contract_delta(
            escrow["contract"], contract, amendment, recovery_context["repo_root"]
        )
    freeze_path = source / FREEZE_NAME
    if freeze_path.exists():
        if freeze_path.is_symlink() or not freeze_path.is_file():
            raise RuntimeError("source pre-generation freeze is not one regular file")
        freeze_stat = freeze_path.stat()
        if stat.S_IMODE(freeze_stat.st_mode) != 0o644 or freeze_stat.st_uid != 0:
            raise PermissionError("source pre-generation freeze must be root-owned mode 0644")
        freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
        if freeze != public_freeze_from_escrow(escrow):
            raise RuntimeError("source pre-generation freeze does not verify against escrow")
    elif recovery_context is not None:
        raise RuntimeError("amended recovery source lacks the public pre-generation freeze")
    return escrow


def archive_output(output: Path, reason: str) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    archived = output.with_name(f"{output.name}.{reason}_{stamp}")
    os.replace(output, archived)
    fsync_directory(output.parent)
    return archived


def prepare_output(
    output: Path, force: bool, config: dict[str, Any] | None = None
) -> Path | None:
    if output.exists() and (config or {}).get("schema_version") == WAVE60_CONFIG_SCHEMA:
        if force:
            raise ValueError("Wave 60 initialized roots cannot be superseded in place")
        actual = {str(path.relative_to(output)) for path in output.rglob("*") if path.is_file()}
        if actual != {"config.snapshot.json", "source_bindings.json"}:
            raise RuntimeError("Wave 60 initialized root inventory drifted before preparation")
        return None
    archived = archive_output(output, "superseded") if output.exists() and force else None
    output.mkdir(mode=0o700, parents=False, exist_ok=False)
    fsync_directory(output.parent)
    return archived


def seal_tree(sealed: Path) -> None:
    if sealed.is_symlink():
        raise RuntimeError("sealed root cannot be a symlink")
    for path in sealed.rglob("*"):
        if path.is_symlink():
            raise RuntimeError("sealed tree contains a symlink")
        os.chown(path, 0, 0)
        path.chmod(0o700 if path.is_dir() else 0o600)
    os.chown(sealed, 0, 0)
    sealed.chmod(0o700)
    fsync_tree(sealed)


def verify_generator_keys(benchmark: Path, keys: tuple[bytes, bytes, bytes]) -> None:
    for name, expected in zip(SECRET_FILES, keys, strict=True):
        payload = json.loads((benchmark / "sealed" / name).read_text(encoding="utf-8"))
        if bytes.fromhex(payload["key_hex"]) != expected:
            raise RuntimeError(f"generator persisted a key inconsistent with escrow: {name}")


def copy_regular(source: Path, destination: Path) -> None:
    if source.is_symlink() or not source.is_file():
        raise RuntimeError(f"staging source is not one regular file: {source}")
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with source.open("rb") as reader, destination.open("xb") as writer:
        shutil.copyfileobj(reader, writer, length=1024 * 1024)


def copy_wave60_preserved_benchmark(
    source_root: Path,
    benchmark: Path,
    preserved_draw_sha256: dict[str, str],
) -> None:
    """Materialize the authorized draw byte-exactly with fresh inodes."""
    source_benchmark = source_root / "benchmark"
    if source_benchmark.is_symlink() or not source_benchmark.is_dir():
        raise RuntimeError("Wave 60 recovery benchmark source is not physical")
    benchmark.mkdir(mode=0o700)
    for source in sorted(source_benchmark.rglob("*")):
        relative = source.relative_to(source_benchmark)
        destination = benchmark / relative
        metadata = source.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise RuntimeError("Wave 60 recovery benchmark contains a symlink")
        if stat.S_ISDIR(metadata.st_mode):
            destination.mkdir(mode=stat.S_IMODE(metadata.st_mode))
        elif stat.S_ISREG(metadata.st_mode):
            copy_regular(source, destination)
            destination.chmod(stat.S_IMODE(metadata.st_mode))
            copied = destination.stat()
            if (metadata.st_dev, metadata.st_ino) == (copied.st_dev, copied.st_ino):
                raise RuntimeError("Wave 60 recovery created a benchmark hardlink")
        else:
            raise RuntimeError("Wave 60 recovery benchmark contains a special node")
    for relative, expected in preserved_draw_sha256.items():
        if not relative.startswith("benchmark/"):
            continue
        destination = benchmark / Path(relative).relative_to("benchmark")
        if sha256_file(destination) != expected:
            raise RuntimeError(f"Wave 60 copied draw drifted: {relative}")
    fsync_tree(benchmark)


def strip_checkpoints(wave51: Path, destination: Path, config: dict[str, Any]) -> list[dict[str, Any]]:
    destination.mkdir(mode=0o700, parents=True, exist_ok=False)
    receipts: list[dict[str, Any]] = []
    for seed in config["seeds"]:
        source = wave51 / "checkpoints" / f"seed{seed}__sigmoid_only.pt"
        checkpoint = torch.load(source, map_location="cpu", weights_only=False)
        if int(checkpoint.get("seed", -1)) != seed or checkpoint.get("output") != "sigmoid_only":
            raise RuntimeError(f"checkpoint identity mismatch for seed {seed}")
        model_state = checkpoint.get("model_state")
        if not isinstance(model_state, dict) or not model_state:
            raise RuntimeError(f"checkpoint lacks model state for seed {seed}")
        payload: dict[str, np.ndarray] = {
            "seed": np.asarray(seed, dtype=np.int64),
            "output": np.asarray("sigmoid_only"),
        }
        for name, tensor in model_state.items():
            if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
                raise RuntimeError(f"checkpoint state is not tensor-only for seed {seed}")
            payload[f"state::{name}"] = tensor.detach().cpu().numpy()
        target = destination / f"seed{seed}__sigmoid_only.npz"
        # NumPy archives are byte-stable for identical ordered arrays, unlike a
        # repeated torch.save of the same state_dict on current PyTorch.
        np.savez_compressed(target, **payload)
        receipts.append(
            {
                "seed": seed,
                "source_sha256": digest(source),
                "staged_sha256": digest(target),
                "payload_keys": sorted(payload),
                "staged_format": "deterministic_npz_tensor_state_v1",
            }
        )
    return receipts


def chown_stage(stage: Path, uid: int, gid: int) -> None:
    for path in stage.rglob("*"):
        if path.is_symlink():
            raise RuntimeError("staging contains a symlink")
        os.chown(path, uid, gid)
        path.chmod(0o500 if path.is_dir() else 0o400)
    os.chown(stage, uid, gid)
    stage.chmod(0o700)


def build_inference_runtime(runtime: Path, uid: int, gid: int) -> dict[str, str]:
    package = runtime / "geometria_proporcional"
    package.mkdir(mode=0o700, parents=True, exist_ok=False)
    for name in INFERENCE_RUNTIME_SOURCES:
        copy_regular(SRC_ROOT / "geometria_proporcional" / name, package / name)
    copy_regular(WORKER_PATH, runtime / "wave56_infer_worker.py")
    hashes = inventory_hashes(runtime)
    for path in runtime.rglob("*"):
        os.chown(path, uid, gid)
        path.chmod(0o500 if path.is_dir() else 0o400)
    os.chown(runtime, uid, gid)
    runtime.chmod(0o500)
    return hashes


def inventory_hashes(root: Path) -> dict[str, str]:
    if any(path.is_symlink() for path in root.rglob("*")):
        raise RuntimeError(f"symlink found while inventorying {root}")
    return {
        str(path.relative_to(root)): digest(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def promote_inference(stage_output: Path, output: Path) -> dict[str, str]:
    pending = output / "inference.pending"
    final = output / "inference"
    pending.mkdir(mode=0o700, exist_ok=False)
    try:
        for source in sorted(stage_output.rglob("*")):
            if source.is_file():
                target = pending / source.relative_to(stage_output)
                copy_regular(source, target)
                target.chmod(0o600)
        hashes = inventory_hashes(pending)
        fsync_tree(pending)
        os.replace(pending, final)
        fsync_directory(output)
        if inventory_hashes(final) != hashes:
            raise RuntimeError("inference promotion changed content")
        return hashes
    except BaseException:
        shutil.rmtree(pending, ignore_errors=True)
        raise


def stage_and_infer(
    output: Path,
    benchmark: Path,
    wave51: Path,
    config_path: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    staging_parent = Path(config["fresh_benchmark"]["staging_parent"]).resolve(strict=True)
    workspace = Path(tempfile.mkdtemp(prefix="wave56-inference-", dir=staging_parent))
    stage = workspace / "stage"
    runtime = workspace / "runtime"
    stage.mkdir(mode=0o700)
    if workspace.is_relative_to(output) or output.is_relative_to(workspace):
        raise RuntimeError("inference staging must be disjoint from the output package")
    try:
        for split in SPLITS:
            copy_regular(benchmark / "visible" / f"{split}.jsonl", stage / "visible" / f"{split}.jsonl")
        copy_regular(benchmark / "protocol_config.json", stage / "protocol_config.json")
        copy_regular(config_path, stage / "wave56_config.json")
        copy_regular(wave51 / "normalizer.npz", stage / "frozen/normalizer.npz")
        checkpoint_receipts = strip_checkpoints(wave51, stage / "frozen/checkpoints", config)
        stage_hashes = inventory_hashes(stage)
        uid = int(config["fresh_benchmark"]["inference_uid"])
        gid = int(config["fresh_benchmark"]["inference_gid"])
        runtime_hashes = build_inference_runtime(runtime, uid, gid)
        chown_stage(stage, uid, gid)
        os.chown(workspace, uid, gid)
        workspace.chmod(0o700)
        env = {
            "HOME": "/nonexistent",
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "PYTHONPATH": str(runtime),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "CUDA_VISIBLE_DEVICES": "",
            "OMP_NUM_THREADS": str(config.get("cpu_threads", 4)),
            "OPENBLAS_NUM_THREADS": str(config.get("cpu_threads", 4)),
            "MKL_NUM_THREADS": str(config.get("cpu_threads", 4)),
            "NUMEXPR_NUM_THREADS": str(config.get("cpu_threads", 4)),
        }
        command = [
            shutil.which("setpriv") or "setpriv",
            f"--reuid={uid}",
            f"--regid={gid}",
            "--clear-groups",
            "--no-new-privs",
            sys.executable,
            str(runtime / "wave56_infer_worker.py"),
            "--stage",
            ".",
            "--output",
            "inference",
            "--sealed-probe",
            str(benchmark / "sealed/train.jsonl"),
        ]
        process = subprocess.Popen(
            command, cwd=stage, env=env, start_new_session=True
        )
        runtime_budget = config.get("runtime_budget", {})
        deadline = time.monotonic() + float(
            runtime_budget.get(
                "max_seconds_per_run", runtime_budget.get("max_seconds_total", 1800)
            )
        )
        rss_limit = int(
            runtime_budget.get(
                "max_rss_bytes",
                runtime_budget.get("max_rss_bytes_per_process", 8 * 1024**3),
            )
        )
        peak_rss = 0
        while process.poll() is None:
            try:
                status = Path(f"/proc/{process.pid}/status").read_text(encoding="utf-8")
                for line in status.splitlines():
                    if line.startswith("VmRSS:"):
                        peak_rss = max(peak_rss, int(line.split()[1]) * 1024)
                        break
            except FileNotFoundError:
                pass
            if peak_rss > rss_limit or time.monotonic() >= deadline:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=5)
                except (ProcessLookupError, subprocess.TimeoutExpired):
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()
                kind = "RSS" if peak_rss > rss_limit else "wall-time"
                raise RuntimeError(f"blind inference exceeded Wave 59 {kind} budget")
            time.sleep(0.1)
        if process.returncode:
            raise subprocess.CalledProcessError(process.returncode, command)
        worker_output = stage / "inference"
        receipt = json.loads((worker_output / "access_receipt.json").read_text(encoding="utf-8"))
        if receipt.get("effective_uid") != uid or receipt.get("effective_gid") != gid:
            raise RuntimeError("inference worker did not retain the frozen unprivileged identity")
        if receipt.get("sealed_truth_probe", {}).get("passed") is not True:
            raise RuntimeError("sealed truth access probe did not pass")
        inference_hashes = promote_inference(worker_output, output)
        return {
            "phase": "blind-inference-all-splits-before-any-oracle",
            "staging_parent": str(staging_parent),
            "staging_disjoint": True,
            "staging_input_hashes": stage_hashes,
            "runtime_hashes": runtime_hashes,
            "checkpoint_receipts": checkpoint_receipts,
            "inference_hashes": inference_hashes,
            "effective_uid": uid,
            "effective_gid": gid,
            "negative_truth_probe": "PermissionError",
            "fit_operations": False,
        }
    finally:
        shutil.rmtree(workspace, ignore_errors=False)


def assert_prepared_boundary(output: Path, benchmark: Path) -> None:
    forbidden = (
        output / "authorized_labels",
        output / "bundles",
        output / "fit_freeze.json",
        output / "selection_freeze.json",
        benchmark / "sealed/oracle",
    )
    present = [str(path) for path in forbidden if path.exists()]
    if present:
        raise RuntimeError(f"preparation opened future material: {present}")


def array_exact(left_path: Path, right_path: Path) -> bool:
    with np.load(left_path, allow_pickle=False) as left, np.load(right_path, allow_pickle=False) as right:
        if set(left.files) != set(right.files):
            return False
        for name in left.files:
            left_array = left[name]
            right_array = right[name]
            if left_array.dtype != right_array.dtype or left_array.shape != right_array.shape:
                return False
            equal_nan = left_array.dtype.kind in "fc"
            if not np.array_equal(left_array, right_array, equal_nan=equal_nan):
                return False
        return True


def recovery_provenance(
    context: dict[str, Any], execution_contract: dict[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": context["amendment"].get(
            "schema_version", RECOVERY_AMENDMENT_SCHEMA
        ),
        "amendment_path": context["amendment_path"],
        "amendment_sha256": context["amendment_sha256"],
        "implementation_commit": context["implementation_commit"],
        "implementation_audit": context["implementation_audit"],
        "final_audit": context["final_audit"],
        "execution_contract_sha256": compact_json_sha256(execution_contract),
        "escrow_origin": {
            "failed_attempt_basename": context["failed_attempt_basename"],
            "contract_sha256": context["escrow_origin_contract_sha256"],
            "benchmark_manifest_sha256": context["benchmark_manifest_sha256"],
        },
    }


def _wave59_attested_file_record(root: Path, relative: str) -> dict[str, Any]:
    path = root / relative
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise RuntimeError(f"Wave 59 attested path is not one regular file: {relative}")
    return {
        "path": relative,
        "bytes": metadata.st_size,
        "sha256": digest(path),
    }


def validate_wave59_closed_benchmark_inventory(benchmark_root: Path) -> dict[str, Any]:
    """Match every physical benchmark node to the public manifest, content-blind."""
    root_metadata = benchmark_root.lstat()
    if stat.S_ISLNK(root_metadata.st_mode) or not stat.S_ISDIR(root_metadata.st_mode):
        raise RuntimeError("Wave 59 benchmark root must be one physical directory")
    benchmark_root = benchmark_root.resolve(strict=True)
    manifest = json.loads((benchmark_root / "manifest.json").read_text(encoding="utf-8"))
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise RuntimeError("Wave 59 benchmark manifest file map drifted")
    declared = set(files)
    expected_directories: set[str] = set()
    for relative, record in files.items():
        candidate = Path(relative)
        if candidate.is_absolute() or ".." in candidate.parts or candidate == Path("manifest.json"):
            raise RuntimeError("Wave 59 benchmark manifest path is non-canonical")
        if not isinstance(record, dict) or set(record) != {"bytes", "sha256"}:
            raise RuntimeError("Wave 59 benchmark manifest record drifted")
        for parent in candidate.parents:
            if parent != Path("."):
                expected_directories.add(str(parent))
    physical_files: set[str] = set()
    physical_directories: set[str] = set()
    for path in sorted(benchmark_root.rglob("*")):
        relative = str(path.relative_to(benchmark_root))
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise RuntimeError("Wave 59 signed benchmark contains a symlink")
        if stat.S_ISREG(metadata.st_mode):
            physical_files.add(relative)
        elif stat.S_ISDIR(metadata.st_mode):
            physical_directories.add(relative)
        else:
            raise RuntimeError("Wave 59 signed benchmark contains a special node")
    if physical_files != declared | {"manifest.json"}:
        raise RuntimeError("Wave 59 signed benchmark closed file inventory drifted")
    if physical_directories != expected_directories:
        raise RuntimeError("Wave 59 signed benchmark closed directory inventory drifted")
    for relative, expected in files.items():
        path = benchmark_root / relative
        metadata = path.lstat()
        if metadata.st_size != expected["bytes"] or digest(path) != expected["sha256"]:
            raise RuntimeError(f"Wave 59 benchmark manifest binding drifted: {relative}")
    return manifest


def wave59_preparation_attestation_payload(
    root: Path, execution_mode: str
) -> dict[str, Any]:
    """Reconstruct the closed signed authority without reading sealed material."""
    root = root.resolve(strict=True)
    config = json.loads((root / "config.snapshot.json").read_text(encoding="utf-8"))
    from geometria_proporcional.wave59_hgb_guard_bracket import is_successor_config

    fresh_successor = is_successor_config(config)
    allowed_modes = {"primary", "replay"} if fresh_successor else {"recovery", "replay"}
    if execution_mode not in allowed_modes:
        raise RuntimeError("Wave 59 signed preparation mode differs from its config form")
    freeze = json.loads((root / "preparation_freeze.json").read_text(encoding="utf-8"))
    generation = json.loads((root / "generation_receipt.json").read_text(encoding="utf-8"))
    preparation = json.loads((root / "preparation_receipt.json").read_text(encoding="utf-8"))
    journal = json.loads((root / "journals/prepare.json").read_text(encoding="utf-8"))
    provenance = freeze.get("recovery_provenance")
    amendment_path = root / RECOVERY_AMENDMENT_COPY_NAME
    if fresh_successor:
        if (
            "recovery_provenance" in freeze
            or "recovery_provenance" in generation
            or "recovery_provenance" in preparation
            or amendment_path.exists()
        ):
            raise RuntimeError("Wave 59 fresh preparation mixed recovery authority")
    elif (
        not isinstance(provenance, dict)
        or generation.get("recovery_provenance") != provenance
        or preparation.get("recovery_provenance") != provenance
        or not amendment_path.is_file()
    ):
        raise RuntimeError("Wave 59 signed recovery preparation lacks authority")
    if any(
        row.get("execution_mode") != execution_mode
        for row in (generation, preparation, journal)
    ):
        raise RuntimeError("Wave 59 signed preparation execution mode drifted")
    inference_hashes = inventory_hashes(root / "inference")
    bundle_names = (
        "gate_fit_bundle.npz",
        "gate_select_truth_bundle.npz",
        "gate_select_inference_bundle.npz",
        "sealed_monitor_truth_bundle.npz",
        "sealed_monitor_inference_bundle.npz",
    )
    prepared_bundle_hashes = {
        f"prepared/{name}": digest(root / "prepared" / name) for name in bundle_names
    }
    if freeze.get("inference_hashes") != inference_hashes:
        raise RuntimeError("Wave 59 signed inference map differs from preparation freeze")
    if freeze.get("prepared_bundle_hashes") != prepared_bundle_hashes:
        raise RuntimeError("Wave 59 signed bundle map differs from preparation freeze")
    record_paths = (
        FREEZE_NAME,
        "benchmark/manifest.json",
        "generation_receipt.json",
        "preparation_freeze.json",
        "preparation_receipt.json",
        "config.snapshot.json",
        "source_bindings.json",
        "journals/prepare.json",
    )
    if not fresh_successor:
        record_paths = (RECOVERY_AMENDMENT_COPY_NAME, *record_paths)
    payload = {
        "schema_version": (
            WAVE59_FRESH_PREPARATION_ATTESTATION_SCHEMA
            if fresh_successor
            else WAVE59_PREPARATION_ATTESTATION_SCHEMA
        ),
        "phase": "wave59-preparation-finalized-before-analysis",
        "run_role": "replay" if execution_mode == "replay" else "primary",
        "execution_mode": execution_mode,
        "git_commit": freeze.get("git_commit"),
        "records": {
            relative: _wave59_attested_file_record(root, relative)
            for relative in record_paths
        },
        "inference_hashes": inference_hashes,
        "prepared_bundle_hashes": prepared_bundle_hashes,
    }
    if not fresh_successor:
        payload["recovery_provenance"] = provenance
    return payload


def publish_wave59_preparation_attestation(
    root: Path,
    execution_mode: str,
    private_key_path: Path,
    trusted_public_key_path: Path = PUBLIC_KEY,
) -> dict[str, Any]:
    root_metadata = root.lstat()
    if stat.S_ISLNK(root_metadata.st_mode) or not stat.S_ISDIR(root_metadata.st_mode):
        raise RuntimeError("Wave 59 preparation-attestation root must be physical")
    validate_wave59_closed_benchmark_inventory(root / "benchmark")
    payload = wave59_preparation_attestation_payload(root, execution_mode)
    receipt = sign_attestation(
        payload,
        private_key_path.resolve(strict=True),
        trusted_public_key_path.resolve(strict=True),
    )
    atomic_write_json(
        root / WAVE59_PREPARATION_ATTESTATION_NAME, receipt, mode=0o644
    )
    verify_attestation(receipt, trusted_public_key_path.resolve(strict=True))
    return receipt


def publish_wave60_preparation_attestation(
    root: Path,
    execution_mode: str,
    private_key_path: Path,
    trusted_public_key_path: Path = PUBLIC_KEY,
) -> dict[str, Any]:
    """Sign the immutable Wave 60 preparation boundary without opening truth."""
    if execution_mode not in {"primary", "recovery", "replay"}:
        raise RuntimeError("Wave 60 preparation role drifted")
    config = json.loads((root / "config.snapshot.json").read_text(encoding="utf-8"))
    record_paths = [
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
    ]
    if config.get("attempt", {}).get("recovery") is not None:
        record_paths.append(RECOVERY_AMENDMENT_COPY_NAME)
    records = {}
    for relative in record_paths:
        records[relative] = _wave59_attested_file_record(root, relative)
    payload = {
        "schema_version": "wave60-signed-preparation-authority-v1",
        "phase": "wave60-preparation-finalized-before-source-binding",
        "run_role": "replay" if execution_mode == "replay" else "primary",
        "execution_mode": execution_mode,
        "git_commit": json.loads(
            (root / "preparation_freeze.json").read_text(encoding="utf-8")
        )["git_commit"],
        "records": records,
        "truth_accessed": False,
        "fit_operations": False,
    }
    signed = sign_attestation(
        payload,
        private_key_path.resolve(strict=True),
        trusted_public_key_path.resolve(strict=True),
    )
    receipt = {
        "schema_version": "wave60-signed-preparation-authority-v1",
        "phase": "prepare",
        "payload": payload,
        "public_key_fingerprint": signed["trusted_public_key_sha256"],
        "signature_base64": signed["signature_base64"],
    }
    atomic_write_json(root / WAVE59_PREPARATION_ATTESTATION_NAME, receipt, mode=0o644)
    verify_attestation(signed, trusted_public_key_path.resolve(strict=True))
    return receipt


def compare_preparation(replay: Path, primary: Path, config: dict[str, Any]) -> dict[str, bool]:
    if replay.resolve() == primary.resolve():
        raise ValueError("replay cannot reference itself")
    validate_manifest(replay / "benchmark")
    validate_manifest(primary / "benchmark")
    checks: dict[str, bool] = {}
    replay_manifest = json.loads((replay / "benchmark/manifest.json").read_text(encoding="utf-8"))
    primary_manifest = json.loads((primary / "benchmark/manifest.json").read_text(encoding="utf-8"))
    commitment_names = (
        "generation_key_commitment",
        "identity_key_commitment",
        "semantic_commitment_key_commitment",
    )
    checks["key_commitments"] = all(replay_manifest[name] == primary_manifest[name] for name in commitment_names)
    for relative in [
        "benchmark/manifest.json",
        "benchmark/protocol_config.json",
        *(f"benchmark/visible/{split}.jsonl" for split in SPLITS),
    ]:
        checks[f"content:{relative}"] = digest(replay / relative) == digest(primary / relative)
    for relative in (ESCROW_NAME, FREEZE_NAME):
        replay_artifact = replay / relative
        primary_artifact = primary / relative
        if replay_artifact.exists() or primary_artifact.exists():
            checks[f"content:{relative}"] = (
                replay_artifact.is_file()
                and not replay_artifact.is_symlink()
                and primary_artifact.is_file()
                and not primary_artifact.is_symlink()
                and digest(replay_artifact) == digest(primary_artifact)
            )
    primary_amendment = primary / RECOVERY_AMENDMENT_COPY_NAME
    replay_amendment = replay / RECOVERY_AMENDMENT_COPY_NAME
    if primary_amendment.exists() or replay_amendment.exists():
        checks[f"content:{RECOVERY_AMENDMENT_COPY_NAME}"] = (
            primary_amendment.is_file()
            and not primary_amendment.is_symlink()
            and replay_amendment.is_file()
            and not replay_amendment.is_symlink()
            and digest(replay_amendment) == digest(primary_amendment)
        )
    for split in SPLITS:
        for seed in config["seeds"]:
            relative = Path("inference/logits") / f"seed{seed}__{split}.npz"
            checks[f"array:{relative}"] = array_exact(replay / relative, primary / relative)
    if config.get("schema_version") in {WAVE59_CONFIG_SCHEMA, WAVE60_CONFIG_SCHEMA}:
        for name in (
            "gate_fit_bundle.npz",
            "gate_select_truth_bundle.npz",
            "gate_select_inference_bundle.npz",
            "sealed_monitor_truth_bundle.npz",
            "sealed_monitor_inference_bundle.npz",
        ):
            relative = Path("prepared") / name
            checks[f"array:{relative}"] = array_exact(replay / relative, primary / relative)
    replay_freeze = json.loads((replay / "preparation_freeze.json").read_text(encoding="utf-8"))
    primary_freeze = json.loads((primary / "preparation_freeze.json").read_text(encoding="utf-8"))
    checks["preparation_freeze"] = replay_freeze == primary_freeze
    replay_generation = json.loads((replay / "generation_receipt.json").read_text(encoding="utf-8"))
    primary_generation = json.loads((primary / "generation_receipt.json").read_text(encoding="utf-8"))
    for field in (
        "sealed_population_counts",
        "sealed_pair_token_counts_total",
        "sealed_eligible_pair_token_counts",
        "recovery_provenance",
    ):
        if field in primary_generation or field in replay_generation:
            checks[f"generation_receipt:{field}"] = (
                replay_generation.get(field) == primary_generation.get(field)
            )
    if not all(checks.values()):
        raise RuntimeError(f"preparation replay mismatch: {checks}")
    return checks


def execute_preparation(
    args: argparse.Namespace,
    output: Path,
    config_path: Path,
    config: dict[str, Any],
    mode: str,
    contract: dict[str, Any],
    reused_escrow: dict[str, Any] | None,
    *,
    recovery_context: dict[str, Any] | None = None,
    keys_override: tuple[bytes, bytes, bytes] | None = None,
    protocol_override: Any | None = None,
    trusted_public_key_path: Path = PUBLIC_KEY,
    generation_fn: Callable[..., dict[str, Any]] = generate_benchmark,
    crash_hook: Callable[[str, Path], None] | None = None,
) -> None:
    if reused_escrow is not None and keys_override is not None:
        raise ValueError("recovery/replay keys come only from the durable escrow")
    if recovery_context is not None and (reused_escrow is None or mode not in {"recovery", "replay"}):
        raise ValueError("amended execution requires recovery/replay with a reused escrow")
    if recovery_context is not None:
        failed, inventory = revalidate_authorized_recovery_origin(
            recovery_context,
            contract,
            trusted_public_key_path,
        )
        if failed != recovery_context["failed_attempt"] or inventory != recovery_context["origin_inventory"]:
            raise RuntimeError("recovery origin changed before key extraction")
    keys = keys_from_escrow(reused_escrow) if reused_escrow else (
        keys_override
        if keys_override is not None
        else (
            secrets.token_bytes(32),
            secrets.token_bytes(32),
            secrets.token_bytes(32),
        )
    )
    if len(set(keys)) != 3:
        raise RuntimeError("fresh keys must be distinct")
    escrow = reused_escrow if reused_escrow is not None else make_escrow(contract, keys)
    if crash_hook:
        crash_hook("before_escrow", output)
    escrow_sha256 = atomic_write_json(output / ESCROW_NAME, escrow, mode=0o600)
    if recovery_context is not None:
        expected_escrow_sha256 = recovery_context["amendment"]["escrow_origin"]["escrow_sha256"]
        if escrow_sha256 != expected_escrow_sha256:
            raise RuntimeError("republished escrow bytes differ from the failed pre-oracle origin")
    if crash_hook:
        crash_hook("after_escrow", output)
    freeze_sha256 = atomic_write_json(
        output / FREEZE_NAME, public_freeze_from_escrow(escrow), mode=0o644
    )
    if recovery_context is not None:
        expected_freeze_sha256 = recovery_context["amendment"]["escrow_origin"][
            "pre_generation_freeze_sha256"
        ]
        if freeze_sha256 != expected_freeze_sha256:
            raise RuntimeError("republished public freeze differs from the failed pre-oracle origin")
    verify_escrow_and_freeze(output, escrow)
    provenance = (
        recovery_provenance(recovery_context, contract)
        if recovery_context is not None
        else None
    )
    if recovery_context is not None:
        copied_sha256 = atomic_write_json(
            output / RECOVERY_AMENDMENT_COPY_NAME,
            recovery_context["amendment"],
            mode=0o644,
        )
        if copied_sha256 != recovery_context["amendment_sha256"]:
            raise RuntimeError("published recovery amendment differs from approved repository artifact")
    if crash_hook:
        crash_hook("after_pre_generation_freeze", output)

    protocol = protocol_override if protocol_override is not None else default_protocol_config(smoke=False)
    benchmark = output / "benchmark"
    if (
        recovery_context is not None
        and config.get("schema_version") == WAVE60_CONFIG_SCHEMA
    ):
        copy_wave60_preserved_benchmark(
            recovery_context["reuse_source"],
            benchmark,
            recovery_context["amendment"]["preserved_draw_sha256"],
        )
    else:
        generation_fn(
            benchmark,
            protocol,
            generation_key=keys[0],
            identity_key=keys[1],
            commitment_key=keys[2],
            attestation_private_key_path=args.attestation_private_key.resolve(strict=True),
            trusted_public_key_path=trusted_public_key_path,
        )
    seal_tree(benchmark / "sealed")
    verify_generator_keys(benchmark, keys)
    validate_manifest(benchmark)
    validate_visible_package(benchmark, protocol)
    validate_semantic_attestation(benchmark, trusted_public_key_path)
    manifest = json.loads((benchmark / "manifest.json").read_text(encoding="utf-8"))
    manifest_sha256 = digest(benchmark / "manifest.json")
    if recovery_context is not None:
        failed, inventory = revalidate_authorized_recovery_origin(
            recovery_context,
            contract,
            trusted_public_key_path,
        )
        if failed != recovery_context["failed_attempt"] or inventory != recovery_context["origin_inventory"]:
            raise RuntimeError("recovery origin changed during benchmark regeneration")
        if manifest_sha256 != recovery_context["benchmark_manifest_sha256"]:
            raise RuntimeError("regenerated benchmark manifest differs from failed pre-oracle origin")
    binding = config["source_binding"]
    if manifest["generation_key_commitment"] == binding["wave50_generation_key_commitment"]:
        raise RuntimeError("fresh generation commitment unexpectedly equals Wave 50")
    visible_hashes = {
        split: digest(benchmark / "visible" / f"{split}.jsonl") for split in SPLITS
    }
    expected_rows = int(config["fresh_benchmark"]["expected_visible_fixtures_per_split"])
    if any(int(manifest["counts"][split]) != expected_rows for split in SPLITS):
        raise RuntimeError("fresh benchmark visible split count differs from prospective freeze")
    population_counts = {
        split: sealed_population_counts(benchmark / "sealed" / f"{split}.jsonl")
        for split in SPLITS
    }
    validate_wave59_population_contract(
        config, population_counts, recovery_context=recovery_context
    )
    if recovery_context is not None:
        expected_population = recovery_context["amendment"]["population_contract"]["counts_by_split"]
        if population_counts != expected_population:
            raise RuntimeError("regenerated benchmark populations differ from recovery amendment")
    total_pair_token_counts = {
        split: counts["total_unique_pair_tokens"] for split, counts in population_counts.items()
    }
    eligible_pair_token_counts = {
        split: counts["eligible_unique_pair_tokens"] for split, counts in population_counts.items()
    }
    if binding["wave50_visible_val_sha256"] in visible_hashes.values():
        raise RuntimeError("fresh visible observations equal historical Wave 50 val")
    fsync_tree(benchmark)
    if crash_hook:
        crash_hook("after_generation", output)
    generation_receipt = {
        "phase": "fresh-benchmark-generated-after-verified-escrow-freeze",
        "execution_mode": mode,
        "escrow_sha256": escrow_sha256,
        "key_commitments": escrow["key_commitments"],
        "manifest_sha256": manifest_sha256,
        "visible_sha256": visible_hashes,
        "sealed_population_counts": population_counts,
        "sealed_pair_token_counts_total": total_pair_token_counts,
        "sealed_eligible_pair_token_counts": eligible_pair_token_counts,
        "sealed_root_owner": 0,
        "sealed_root_mode": "0700",
        "oracle_materialized": False,
    }
    if provenance is not None:
        generation_receipt["recovery_provenance"] = provenance
    atomic_write_json(output / "generation_receipt.json", generation_receipt, mode=0o644)

    inference = stage_and_infer(
        output,
        benchmark,
        args.wave51_dir.resolve(strict=True),
        config_path,
        config,
    )
    prepared_bundle_hashes: dict[str, str] = {}
    if config.get("schema_version") in {WAVE59_CONFIG_SCHEMA, WAVE60_CONFIG_SCHEMA}:
        from run_wave59_hgb_guard_bracket import materialize_prepared_bundles

        prepared_bundle_hashes = materialize_prepared_bundles(
            output,
            config,
            policy_manifest=args.wave52_dir.resolve(strict=True) / "policy_manifest.json",
            wave54_selection_freeze=(
                args.wave54_dir.resolve(strict=True) / "selection_freeze.json"
            ),
        )
    if crash_hook:
        crash_hook("after_inference", output)
    assert_prepared_boundary(output, benchmark)
    preparation_freeze = {
        "schema_version": config["schema_version"],
        "phase": "prepared-with-blind-inference-before-any-oracle",
        "git_commit": contract["git_commit"],
        "config_sha256": contract["config_sha256"],
        "prospective_config": contract["prospective_config"],
        "sources": contract["sources"],
        "upstream": contract["upstream"],
        "historical_preflight": contract["historical_preflight"],
        "source_bindings": contract["source_bindings"],
        "key_commitments": escrow["key_commitments"],
        "benchmark_manifest_sha256": digest(benchmark / "manifest.json"),
        "protocol_config_sha256": digest(benchmark / "protocol_config.json"),
        "visible_sha256": visible_hashes,
        "staging_input_hashes": inference["staging_input_hashes"],
        "inference_runtime_hashes": inference["runtime_hashes"],
        "checkpoint_receipts": inference["checkpoint_receipts"],
        "inference_hashes": inference["inference_hashes"],
        "inference_uid": inference["effective_uid"],
        "inference_gid": inference["effective_gid"],
        "negative_truth_probe": inference["negative_truth_probe"],
        "oracle_materialized": False,
        "authorized_labels_present": False,
        "bundles_present": config.get("schema_version") in {
            WAVE59_CONFIG_SCHEMA,
            WAVE60_CONFIG_SCHEMA,
        },
        "fit_operations": False,
        "physical_splits": config["physical_splits"],
    }
    if config.get("schema_version") in {WAVE59_CONFIG_SCHEMA, WAVE60_CONFIG_SCHEMA}:
        preparation_freeze["prepared_bundle_hashes"] = prepared_bundle_hashes
    if provenance is not None:
        preparation_freeze["recovery_provenance"] = provenance
    atomic_write_json(output / "preparation_freeze.json", preparation_freeze, mode=0o644)
    verify = json.loads((output / "preparation_freeze.json").read_text(encoding="utf-8"))
    if verify != preparation_freeze:
        raise RuntimeError("preparation freeze failed post-publication verification")
    if crash_hook:
        crash_hook("after_preparation_freeze", output)

    replay_checks = None
    phase_prefix = preparation_phase_prefix(config)
    if mode == "replay":
        replay_checks = compare_preparation(output, args.reference_dir.resolve(strict=True), config)
        atomic_write_json(
            output / "preparation_replay.json",
            {
                "phase": f"{phase_prefix}-preparation-exact-replay",
                "checks": replay_checks,
                "all_exact": all(replay_checks.values()),
            },
            mode=0o644,
        )
    preparation_receipt = {
        "phase": f"{phase_prefix}-stage1-preparation-complete",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "execution_mode": mode,
        "preparation_freeze_sha256": digest(output / "preparation_freeze.json"),
        "generation_receipt_sha256": digest(output / "generation_receipt.json"),
        "replay_exact": all(replay_checks.values()) if replay_checks else None,
        "next_state": "PREPARED",
    }
    if provenance is not None:
        preparation_receipt["recovery_provenance"] = provenance
    atomic_write_json(
        output / "preparation_receipt.json",
        preparation_receipt,
        mode=0o644,
    )
    if config.get("schema_version") in {WAVE59_CONFIG_SCHEMA, WAVE60_CONFIG_SCHEMA}:
        config_snapshot = output / "config.snapshot.json"
        bindings_snapshot = output / "source_bindings.json"
        if config.get("schema_version") == WAVE60_CONFIG_SCHEMA:
            if digest(config_snapshot) != digest(config_path):
                raise RuntimeError("Wave 60 initialized config snapshot drifted")
            if json.loads(bindings_snapshot.read_text(encoding="utf-8")) != contract[
                "source_bindings"
            ]:
                raise RuntimeError("Wave 60 initialized source bindings drifted")
        else:
            copy_regular(config_path, config_snapshot)
            config_snapshot.chmod(0o644)
            atomic_write_json(
                bindings_snapshot, contract["source_bindings"], mode=0o644
            )
        journals = output / "journals"
        journals.mkdir(mode=0o755)
        journals.chmod(0o755)
        atomic_write_json(
            journals / "prepare.json",
            {
                "schema_version": (
                    "wave60-preparation-journal-v1"
                    if config.get("schema_version") == WAVE60_CONFIG_SCHEMA
                    else "wave59-phase-journal-v1"
                ),
                "phase": "prepare",
                "status": "PREPARED",
                "execution_mode": mode,
                "preparation_freeze_sha256": digest(output / "preparation_freeze.json"),
                "prepared_bundle_hashes": prepared_bundle_hashes,
                "maximum_truth_materialized": "prepared_all_splits_root_only",
                "next_state": "PREPARED",
            },
            mode=0o644,
        )


def run_preparation_transaction(
    args: argparse.Namespace,
    output: Path,
    config_path: Path,
    config: dict[str, Any],
    mode: str,
    contract: dict[str, Any],
    reused_escrow: dict[str, Any] | None,
    *,
    force: bool,
    recovery_context: dict[str, Any] | None = None,
    keys_override: tuple[bytes, bytes, bytes] | None = None,
    protocol_override: Any | None = None,
    trusted_public_key_path: Path = PUBLIC_KEY,
    generation_fn: Callable[..., dict[str, Any]] = generate_benchmark,
    crash_hook: Callable[[str, Path], None] | None = None,
) -> Path | None:
    """Execute one preparation attempt and archive every failed physical state."""
    archived = prepare_output(output, force, config)
    try:
        execute_preparation(
            args,
            output,
            config_path,
            config,
            mode,
            contract,
            reused_escrow,
            recovery_context=recovery_context,
            keys_override=keys_override,
            protocol_override=protocol_override,
            trusted_public_key_path=trusted_public_key_path,
            generation_fn=generation_fn,
            crash_hook=crash_hook,
        )
        receipt_path = output / "preparation_receipt.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["superseded_output"] = str(archived) if archived else None
        atomic_write_json(receipt_path, receipt, mode=0o644)
    except BaseException as error:
        if output.exists():
            if config.get("schema_version") == WAVE59_CONFIG_SCHEMA:
                from run_wave59_hgb_guard_bracket import archive_failed_attempt

                archive_failed_attempt(
                    output,
                    error,
                    run_role="replay" if mode == "replay" else "primary",
                    recovery_context=recovery_context is not None,
                    attestation_private_key=args.attestation_private_key,
                    trusted_public_key=trusted_public_key_path,
                )
            elif config.get("schema_version") == WAVE60_CONFIG_SCHEMA:
                failed = output / "failed_preparation"
                failed.mkdir(mode=0o700, exist_ok=False)
                for child in list(output.iterdir()):
                    if child.name in {
                        "config.snapshot.json",
                        "source_bindings.json",
                        "failed_preparation",
                    }:
                        continue
                    os.replace(child, failed / child.name)
                atomic_write_json(
                    failed / "preparation_error.json",
                    {
                        "schema_version": "wave60-preparation-failure-v1",
                        "error_type": type(error).__name__,
                        "error_message_sha256": hashlib.sha256(
                            str(error).encode("utf-8")
                        ).hexdigest(),
                    },
                    mode=0o600,
                )
            else:
                try:
                    atomic_write_json(
                        output / "FAILURE.json",
                        {
                            "error_type": type(error).__name__,
                            "message": str(error),
                            "escrow_present": has_escrow(output),
                            "redraw_forbidden_if_escrow_present": True,
                        },
                        mode=0o600,
                    )
                finally:
                    archive_output(output, "failed")
        raise
    return archived


def wave60_prior_preparation_elapsed(
    args: argparse.Namespace, config: dict[str, Any], mode: str
) -> float:
    """Load the signed predecessor duration used by the shared 900 s ledger."""
    if config.get("schema_version") != WAVE60_CONFIG_SCHEMA or mode == "primary":
        return 0.0
    source_arg = (
        args.replay_secrets_from if mode == "replay" else args.recovery_secrets_from
    )
    source = source_arg.resolve(strict=True)
    receipt_path = source / "preparation_receipt.json"
    attestation_path = source / WAVE59_PREPARATION_ATTESTATION_NAME
    if not receipt_path.is_file() or not attestation_path.is_file():
        raise RuntimeError("Wave 60 prior preparation budget authority is absent")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
    if set(attestation) != {
        "schema_version", "phase", "payload", "public_key_fingerprint",
        "signature_base64",
    } or attestation.get("schema_version") != "wave60-signed-preparation-authority-v1":
        raise RuntimeError("Wave 60 prior preparation attestation drifted")
    verify_attestation(
        {
            "algorithm": "Ed25519",
            "payload": attestation["payload"],
            "signature_base64": attestation["signature_base64"],
            "trusted_public_key_sha256": attestation["public_key_fingerprint"],
        },
        PUBLIC_KEY.resolve(strict=True),
    )
    record = attestation.get("payload", {}).get("records", {}).get(
        "preparation_receipt.json"
    )
    if record != {
        "path": "preparation_receipt.json",
        "bytes": receipt_path.stat().st_size,
        "sha256": sha256_file(receipt_path),
    } or attestation.get("payload", {}).get("run_role") != "primary":
        raise RuntimeError("Wave 60 prior preparation receipt is not signed")
    budget = receipt.get("coordinator_budget")
    if not isinstance(budget, dict) or set(budget) != {
        "duration_seconds", "cumulative_duration_seconds", "prior_elapsed_seconds",
        "max_rss_bytes", "max_seconds", "max_seconds_total",
        "max_rss_allowed_bytes", "cuda_visible_devices", "budget_enforced",
    }:
        raise RuntimeError("Wave 60 prior preparation budget ledger drifted")
    duration = float(budget["duration_seconds"])
    if (
        float(budget["prior_elapsed_seconds"]) != 0.0
        or float(budget["cumulative_duration_seconds"]) != duration
        or float(budget["max_seconds_total"]) != 900.0
        or duration < 0.0
        or duration >= 900.0
    ):
        raise RuntimeError("Wave 60 prior preparation budget ledger is invalid")
    return duration


@contextmanager
def wave59_coordinator_budget(
    config: dict[str, Any], *, prior_elapsed_seconds: float = 0.0
) -> Any:
    """Apply the Wave 59 wall/RSS envelope to preflight and preparation itself."""
    if config.get("schema_version") not in {WAVE59_CONFIG_SCHEMA, WAVE60_CONFIG_SCHEMA}:
        yield None
        return
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Wave 59 preparation coordinator can see CUDA")
    budget = config["runtime_budget"]
    total_seconds = float(
        budget.get("max_seconds_per_run", budget.get("max_seconds_total"))
    )
    wave60_budget = config.get("schema_version") == WAVE60_CONFIG_SCHEMA
    if not wave60_budget and prior_elapsed_seconds != 0.0:
        raise RuntimeError("prior elapsed budget is Wave 60-only")
    if prior_elapsed_seconds < 0.0 or prior_elapsed_seconds >= total_seconds:
        raise RuntimeError("Wave 60 combined preparation budget is exhausted")
    seconds = total_seconds - prior_elapsed_seconds
    rss_limit = int(
        budget.get("max_rss_bytes", budget.get("max_rss_bytes_per_process"))
    )
    started = time.monotonic()
    state: dict[str, Any] = {
        "duration_seconds": 0.0,
        "max_rss_bytes": 0,
        "max_seconds": seconds,
        "max_rss_allowed_bytes": rss_limit,
        "cuda_visible_devices": "",
        "budget_enforced": True,
    }
    if wave60_budget:
        state.update(
            {
                "max_seconds_total": total_seconds,
                "prior_elapsed_seconds": float(prior_elapsed_seconds),
                "cumulative_duration_seconds": float(prior_elapsed_seconds),
            }
        )
    stop = threading.Event()

    def raise_budget(signum: int, frame: Any) -> None:
        kind = "RSS" if signum == signal.SIGUSR1 else "wall-time"
        raise RuntimeError(f"Wave 59 preparation coordinator exceeded {kind} budget")

    def watch_rss() -> None:
        while not stop.wait(0.1):
            try:
                status = Path("/proc/self/status").read_text(encoding="utf-8")
                rss = next(
                    int(line.split()[1]) * 1024
                    for line in status.splitlines()
                    if line.startswith("VmRSS:")
                )
                state["max_rss_bytes"] = max(int(state["max_rss_bytes"]), rss)
                if rss > rss_limit:
                    os.kill(os.getpid(), signal.SIGUSR1)
                    return
            except (FileNotFoundError, StopIteration):
                continue

    previous_alarm = signal.getsignal(signal.SIGALRM)
    previous_rss = signal.getsignal(signal.SIGUSR1)
    signal.signal(signal.SIGALRM, raise_budget)
    signal.signal(signal.SIGUSR1, raise_budget)
    watcher = threading.Thread(target=watch_rss, name="wave59-rss-budget", daemon=True)
    watcher.start()
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield state
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        stop.set()
        watcher.join(timeout=1.0)
        state["duration_seconds"] = time.monotonic() - started
        if wave60_budget:
            state["cumulative_duration_seconds"] = (
                float(prior_elapsed_seconds) + float(state["duration_seconds"])
            )
        signal.signal(signal.SIGALRM, previous_alarm)
        signal.signal(signal.SIGUSR1, previous_rss)


def main() -> None:
    os.umask(0o077)
    args = parse_args()
    config_path = args.config.resolve(strict=True)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output = args.output_dir.resolve()
    mode = validate_invocation(args, output, config)
    recovery_context = None
    prior_elapsed_seconds = wave60_prior_preparation_elapsed(args, config, mode)
    budget_context = (
        wave59_coordinator_budget(
            config, prior_elapsed_seconds=prior_elapsed_seconds
        )
        if config.get("schema_version") == WAVE60_CONFIG_SCHEMA
        else wave59_coordinator_budget(config)
    )

    try:
        with budget_context as coordinator_budget:
            # This entire preflight is intentionally before output creation or archival.
            contract = preparation_preflight(args, config_path, config)
            reused_escrow = None
            if mode in {"replay", "recovery"}:
                source_arg = (
                    args.replay_secrets_from
                    if mode == "replay"
                    else args.recovery_secrets_from
                )
                source_metadata = source_arg.lstat()
                if stat.S_ISLNK(source_metadata.st_mode) or not stat.S_ISDIR(
                    source_metadata.st_mode
                ):
                    raise RuntimeError(
                        "replay/recovery source must be one physical directory"
                    )
                source = source_arg.resolve(strict=True)
                if args.recovery_amendment is not None:
                    recovery_context = validate_recovery_amendment(
                        args.recovery_amendment,
                        source,
                        contract,
                        mode,
                    )
                reused_escrow = validate_reused_escrow(
                    source, contract, recovery_context
                )

            run_preparation_transaction(
                args,
                output,
                config_path,
                config,
                mode,
                contract,
                reused_escrow,
                force=args.force,
                recovery_context=recovery_context,
            )
        if coordinator_budget is not None:
            receipt_path = output / "preparation_receipt.json"
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt["coordinator_budget"] = coordinator_budget
            atomic_write_json(receipt_path, receipt, mode=0o644)
            if config.get("schema_version") == WAVE60_CONFIG_SCHEMA:
                publish_wave60_preparation_attestation(
                    output,
                    mode,
                    args.attestation_private_key,
                    PUBLIC_KEY,
                )
            else:
                publish_wave59_preparation_attestation(
                    output,
                    mode,
                    args.attestation_private_key,
                    PUBLIC_KEY,
                )
    except BaseException as error:
        if output.exists() and config.get("schema_version") == WAVE59_CONFIG_SCHEMA:
            from run_wave59_hgb_guard_bracket import archive_failed_attempt

            archive_failed_attempt(
                output,
                error,
                run_role="replay" if mode == "replay" else "primary",
                recovery_context=recovery_context is not None,
                attestation_private_key=args.attestation_private_key,
                trusted_public_key=PUBLIC_KEY,
            )
        raise
    print(json.dumps({"state": "PREPARED", "execution_mode": mode}, sort_keys=True))


if __name__ == "__main__":
    main()
