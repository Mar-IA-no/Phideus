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
import stat
import threading
from contextlib import contextmanager

# The coordinator is CPU-only too.  Set this before importing libraries that
# may inspect CUDA or initialize threaded numerical runtimes.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = "4"

import joblib
import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

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
    PHASE_FILES,
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

HEX64 = frozenset("0123456789abcdef")
FAILURE_KEYS = frozenset(
    {
        "schema_version",
        "error_type",
        "error_message_sha256",
        "last_state",
        "maximum_truth_materialized",
        "run_role",
        "recovery_context",
        "original_path",
        "archived_path",
    }
)
FAILURE_METADATA = frozenset(
    {
        "FAILURE.json",
        "failure_inventory.json",
        "failure_attestation.json",
        "failure_attestation_error.json",
        "artifact_manifest.json",
    }
)
FAILURE_RECORD_CLASSES = frozenset(
    {
        "scientific_exact",
        "scientific_array_exact",
        "functional_state",
        "operational_semantic",
        "secret_excluded_from_public_manifest",
        "self_reference",
        "failure_record",
    }
)
PHASE_PROBE_RELATIVES = {
    "fit": (
        "prepared/gate_select_truth_bundle.npz",
        "prepared/sealed_monitor_truth_bundle.npz",
    ),
    "calibrate_scores": (
        "prepared/gate_select_truth_bundle.npz",
        "prepared/sealed_monitor_truth_bundle.npz",
    ),
    "validate": ("prepared/sealed_monitor_truth_bundle.npz",),
    "monitor_apply": ("prepared/sealed_monitor_truth_bundle.npz",),
    "monitor_evaluate": (),
}


def _require_exact_keys(payload: Any, expected: set[str] | frozenset[str], label: str) -> dict[str, Any]:
    if not isinstance(payload, dict) or set(payload) != set(expected):
        observed = sorted(payload) if isinstance(payload, dict) else type(payload).__name__
        raise RuntimeError(
            f"Wave 59 {label} keys drifted: expected={sorted(expected)}, observed={observed}"
        )
    return payload


def _require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in HEX64 for character in value)
    ):
        raise RuntimeError(f"Wave 59 {label} is not one lowercase SHA-256 digest")
    return value


def _require_safe_relative_path(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"Wave 59 {label} is not a nonempty relative path")
    path = Path(value)
    if path.is_absolute() or value != path.as_posix() or ".." in path.parts:
        raise RuntimeError(f"Wave 59 {label} is not a canonical relative path")
    return value


def _json_payload_sha256(payload: Any) -> str:
    encoded = (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
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


def validate_execution_bindings(
    config: dict[str, Any],
    config_path: Path,
    recovery_root: Path | None = None,
) -> None:
    """Authenticate the frozen worktree and dependency contract before each phase."""
    if config.get("status") != FROZEN_STATUS:
        return
    validate_pre_draw_config(config)
    expected = config["source_sha256"]
    relative_config = str(config_path.resolve(strict=True).relative_to(REPO_ROOT))
    if not relative_config.endswith(CONFIG_SOURCE_SUFFIX):
        raise RuntimeError("Wave 59 execution config path drifted")
    raw_observed: dict[str, str] = {}
    for relative in config["required_execution_sources"]:
        path = (REPO_ROOT / relative).resolve(strict=True)
        if path.relative_to(REPO_ROOT) != Path(relative):
            raise RuntimeError(f"Wave 59 non-canonical execution source: {relative}")
        raw_observed[relative] = sha256_file(path)
        require_clean_head_source(REPO_ROOT, relative, path)
    observed = dict(raw_observed)
    observed[relative_config] = config_self_binding_sha256(config, relative_config)
    if observed != expected:
        if recovery_root is None:
            differing = sorted(
                relative
                for relative in set(observed) | set(expected)
                if observed.get(relative) != expected.get(relative)
            )
            raise RuntimeError(f"Wave 59 execution source drifted: {differing}")
        copied = recovery_root / "recovery_amendment.json"
        if copied.is_symlink() or not copied.is_file():
            raise RuntimeError("Wave 59 recovered execution lacks its amendment copy")
        from prepare_wave56_fresh import (
            WAVE59_RECOVERY_AMENDMENT_RELATIVE,
            validate_wave59_repository_recovery_authority,
        )

        amendment, amendment_sha256 = validate_wave59_repository_recovery_authority(
            REPO_ROOT / WAVE59_RECOVERY_AMENDMENT_RELATIVE,
            expected,
            observed,
            repo_root=REPO_ROOT,
        )
        if sha256_file(copied) != amendment_sha256 or read_json(copied) != amendment:
            raise RuntimeError("Wave 59 recovered amendment copy differs from authority")
        validate_signed_preparation_package(
            recovery_root, config, amendment, amendment_sha256
        )
        preparation = read_json(recovery_root / "preparation_freeze.json")
        if (
            preparation["git_commit"]
            != subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
            ).strip()
            or preparation["prospective_config"] != config
            or preparation["sources"] != raw_observed
        ):
            raise RuntimeError("Wave 59 recovered preparation source authority differs")
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


def _require_root_owned_mode(path: Path, expected_mode: int, label: str, *, directory: bool) -> None:
    metadata = path.lstat()
    expected_type = stat.S_ISDIR if directory else stat.S_ISREG
    if stat.S_ISLNK(metadata.st_mode) or not expected_type(metadata.st_mode):
        raise RuntimeError(f"Wave 59 {label} physical type drifted")
    if metadata.st_uid != 0 or metadata.st_gid != 0:
        raise PermissionError(f"Wave 59 {label} must be root-owned")
    if stat.S_IMODE(metadata.st_mode) != expected_mode:
        raise PermissionError(f"Wave 59 {label} mode drifted")


def validate_signed_preparation_package(
    root: Path,
    config: dict[str, Any],
    amendment: dict[str, Any],
    amendment_sha256: str,
) -> str:
    """Authenticate one finalized recovery package without opening sealed truth."""
    candidates = {
        (REPO_ROOT / config["primary_output"]).resolve(strict=False): "recovery",
        (REPO_ROOT / config["replay_output"]).resolve(strict=False): "replay",
    }
    unresolved = root.absolute()
    _require_root_owned_mode(unresolved, 0o700, "preparation root", directory=True)
    resolved = unresolved.resolve(strict=True)
    if resolved not in candidates:
        raise RuntimeError("Wave 59 recovered package path is not canonical")
    execution_mode = candidates[resolved]
    for relative, mode in {
        "benchmark": 0o700,
        "inference": 0o700,
        "prepared": 0o700,
        "journals": 0o755,
    }.items():
        _require_root_owned_mode(resolved / relative, mode, relative, directory=True)
    public_modes = {
        "recovery_amendment.json": 0o644,
        "pre_generation_freeze.json": 0o644,
        "benchmark/manifest.json": 0o600,
        "generation_receipt.json": 0o644,
        "preparation_freeze.json": 0o644,
        "preparation_receipt.json": 0o644,
        "config.snapshot.json": 0o644,
        "source_bindings.json": 0o644,
        "journals/prepare.json": 0o644,
        "preparation_attestation.json": 0o644,
    }
    for relative, mode in public_modes.items():
        _require_root_owned_mode(resolved / relative, mode, relative, directory=False)
    for path in sorted((resolved / "inference").rglob("*")):
        if path.is_file() or path.is_symlink():
            _require_root_owned_mode(path, 0o600, str(path.relative_to(resolved)), directory=False)
    bundle_modes = {
        "gate_fit_bundle.npz": 0o600,
        "gate_select_truth_bundle.npz": 0o600,
        "gate_select_inference_bundle.npz": 0o644,
        "sealed_monitor_truth_bundle.npz": 0o600,
        "sealed_monitor_inference_bundle.npz": 0o644,
    }
    if {path.name for path in (resolved / "prepared").iterdir()} != set(bundle_modes):
        raise RuntimeError("Wave 59 prepared bundle inventory drifted")
    for name, mode in bundle_modes.items():
        _require_root_owned_mode(resolved / "prepared" / name, mode, f"prepared/{name}", directory=False)

    freeze = _require_exact_keys(
        read_json(resolved / "preparation_freeze.json"),
        {
            "schema_version", "phase", "git_commit", "config_sha256",
            "prospective_config", "sources", "upstream", "historical_preflight",
            "source_bindings", "key_commitments", "benchmark_manifest_sha256",
            "protocol_config_sha256", "visible_sha256", "staging_input_hashes",
            "inference_runtime_hashes", "checkpoint_receipts", "inference_hashes",
            "inference_uid", "inference_gid", "negative_truth_probe",
            "oracle_materialized", "authorized_labels_present", "bundles_present",
            "fit_operations", "physical_splits", "prepared_bundle_hashes",
            "recovery_provenance",
        },
        "signed preparation freeze",
    )
    if freeze["schema_version"] != config["schema_version"] or freeze["phase"] != "prepared-with-blind-inference-before-any-oracle":
        raise RuntimeError("Wave 59 signed preparation freeze identity drifted")
    if any(
        freeze[field] is not expected
        for field, expected in {
            "oracle_materialized": False,
            "authorized_labels_present": False,
            "bundles_present": True,
            "fit_operations": False,
        }.items()
    ):
        raise RuntimeError("Wave 59 signed preparation boundary drifted")
    generation = _require_exact_keys(
        read_json(resolved / "generation_receipt.json"),
        {
            "phase", "execution_mode", "escrow_sha256", "key_commitments",
            "manifest_sha256", "visible_sha256", "sealed_population_counts",
            "sealed_pair_token_counts_total", "sealed_eligible_pair_token_counts",
            "sealed_root_owner", "sealed_root_mode", "oracle_materialized",
            "recovery_provenance",
        },
        "signed generation receipt",
    )
    if generation["phase"] != "fresh-benchmark-generated-after-verified-escrow-freeze" or generation["execution_mode"] != execution_mode:
        raise RuntimeError("Wave 59 signed generation receipt identity drifted")
    if generation["oracle_materialized"] is not False or generation["sealed_root_owner"] != 0 or generation["sealed_root_mode"] != "0700":
        raise RuntimeError("Wave 59 signed generation boundary drifted")
    preparation = _require_exact_keys(
        read_json(resolved / "preparation_receipt.json"),
        {
            "phase", "timestamp_utc", "execution_mode", "preparation_freeze_sha256",
            "generation_receipt_sha256", "replay_exact", "next_state",
            "recovery_provenance", "superseded_output", "coordinator_budget",
        },
        "signed preparation receipt",
    )
    if preparation["phase"] != "wave59-stage1-preparation-complete" or preparation["execution_mode"] != execution_mode or preparation["next_state"] != "PREPARED":
        raise RuntimeError("Wave 59 signed preparation receipt identity drifted")
    journal = _validate_resumed_journal(read_json(resolved / "journals/prepare.json"), "prepare")
    if journal["execution_mode"] != execution_mode:
        raise RuntimeError("Wave 59 signed prepare journal mode drifted")
    provenance = freeze["recovery_provenance"]
    if any(payload.get("recovery_provenance") != provenance for payload in (generation, preparation)):
        raise RuntimeError("Wave 59 signed recovery provenance differs")
    if provenance.get("amendment_sha256") != amendment_sha256:
        raise RuntimeError("Wave 59 signed recovery amendment provenance differs")
    if read_json(resolved / "recovery_amendment.json") != amendment:
        raise RuntimeError("Wave 59 signed recovery amendment content differs")
    if read_json(resolved / "config.snapshot.json") != config or read_json(resolved / "source_bindings.json") != config["source_binding"]:
        raise RuntimeError("Wave 59 signed config or source bindings differ")
    pre_generation = _require_exact_keys(
        read_json(resolved / "pre_generation_freeze.json"),
        {"schema_version", "phase", "contains_secrets", "generator_invoked", "contract", "key_commitments"},
        "signed pre-generation freeze",
    )
    if (
        pre_generation["schema_version"] != "wave56-key-escrow-v1"
        or pre_generation["phase"] != "keys-escrowed-and-contract-frozen-before-generation"
        or pre_generation["contains_secrets"] is not False
        or pre_generation["generator_invoked"] is not False
    ):
        raise RuntimeError("Wave 59 signed pre-generation freeze identity drifted")
    benchmark_root = resolved / "benchmark"
    manifest = _require_exact_keys(
        read_json(benchmark_root / "manifest.json"),
        {
            "schema_version", "generator", "software", "files", "counts",
            "catalog_families", "out_of_catalog_families", "generation_key_commitment",
            "identity_key_commitment", "semantic_commitment_key_commitment",
            "calibration_contract", "semantic_attestation",
        },
        "signed benchmark manifest",
    )
    if manifest["schema_version"] != "wave49-relational-benchmark-v2" or manifest["generator"] != "wave49_generator":
        raise RuntimeError("Wave 59 signed benchmark manifest identity drifted")
    from prepare_wave56_fresh import validate_wave59_closed_benchmark_inventory

    if validate_wave59_closed_benchmark_inventory(benchmark_root) != manifest:
        raise RuntimeError("Wave 59 signed benchmark manifest changed during validation")
    if (
        freeze["config_sha256"] != sha256_file(resolved / "config.snapshot.json")
        or freeze["source_bindings"] != config["source_binding"]
        or freeze["benchmark_manifest_sha256"] != sha256_file(resolved / "benchmark/manifest.json")
        or generation["manifest_sha256"] != freeze["benchmark_manifest_sha256"]
        or preparation["preparation_freeze_sha256"] != sha256_file(resolved / "preparation_freeze.json")
        or preparation["generation_receipt_sha256"] != sha256_file(resolved / "generation_receipt.json")
        or journal["preparation_freeze_sha256"] != preparation["preparation_freeze_sha256"]
        or journal["prepared_bundle_hashes"] != freeze["prepared_bundle_hashes"]
        or generation["key_commitments"] != freeze["key_commitments"]
        or pre_generation["key_commitments"] != freeze["key_commitments"]
    ):
        raise RuntimeError("Wave 59 signed preparation hash chain drifted")

    from prepare_wave56_fresh import wave59_preparation_attestation_payload

    attestation = _require_exact_keys(
        read_json(resolved / "preparation_attestation.json"),
        {"algorithm", "payload", "signature_base64", "trusted_public_key_sha256"},
        "signed preparation attestation",
    )
    expected_payload = wave59_preparation_attestation_payload(resolved, execution_mode)
    if attestation["payload"] != expected_payload:
        raise RuntimeError("Wave 59 preparation attestation payload differs from bytes")
    verify_attestation(attestation, TRUSTED_PUBLIC_KEY.resolve(strict=True))
    return execution_mode


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
    archived: Path,
    output: Path,
    config_path: Path,
    policy_manifest: Path | None = None,
) -> Path:
    archived = archived.resolve(strict=True)
    output = output.resolve(strict=False)
    if output.exists():
        raise FileExistsError(output)
    if not (archived / "FAILURE.json").is_file():
        raise RuntimeError("resume source is not a Wave 59 failed attempt")
    failure = read_json(archived / "FAILURE.json")
    _validate_failure_record(failure, archived)
    if Path(failure["original_path"]).resolve() != output:
        raise RuntimeError("identical-hash resume must restore the original canonical path")
    _validate_failure_inventory(archived)
    snapshot = archived / "config.snapshot.json"
    if sha256_file(snapshot) != sha256_file(config_path.resolve(strict=True)):
        raise RuntimeError("identical-hash resume config differs")
    _validate_resumed_journals(
        archived,
        failure,
        config_path,
        policy_manifest,
    )
    _validate_failure_journal_alignment(failure, archived)
    def ignore_root_failure_metadata(directory: str, names: list[str]) -> list[str]:
        if Path(directory).resolve() != archived:
            return []
        return sorted(set(names) & FAILURE_METADATA)

    shutil.copytree(
        archived,
        output,
        symlinks=False,
        ignore=ignore_root_failure_metadata,
    )
    residue = sorted(name for name in FAILURE_METADATA if (output / name).exists())
    if residue:
        raise RuntimeError(f"Wave 59 restore retained failure metadata: {residue}")
    _fsync_directory(output.parent)
    return output


def _validate_failure_record(failure: Any, archived: Path) -> dict[str, Any]:
    failure = _require_exact_keys(failure, FAILURE_KEYS, "FAILURE.json")
    if failure["schema_version"] != "wave59-failed-attempt-v1":
        raise RuntimeError("resume source failure schema drifted")
    if not isinstance(failure["error_type"], str) or not failure["error_type"]:
        raise RuntimeError("Wave 59 failure error_type drifted")
    _require_sha256(failure["error_message_sha256"], "failure error_message_sha256")
    if failure["last_state"] is not None and not isinstance(failure["last_state"], str):
        raise RuntimeError("Wave 59 failure last_state drifted")
    if not isinstance(failure["maximum_truth_materialized"], str):
        raise RuntimeError("Wave 59 failure maximum truth drifted")
    if failure["run_role"] not in {"primary", "replay"}:
        raise RuntimeError("Wave 59 failure run_role drifted")
    if type(failure["recovery_context"]) is not bool:
        raise RuntimeError("Wave 59 failure recovery_context drifted")
    if not isinstance(failure["original_path"], str) or not Path(
        failure["original_path"]
    ).is_absolute():
        raise RuntimeError("Wave 59 failure original_path drifted")
    if failure["archived_path"] != str(archived):
        raise RuntimeError("Wave 59 failure archived_path drifted")
    return failure


def _validate_failure_inventory(archived: Path) -> dict[str, Any]:
    inventory_path = archived / "failure_inventory.json"
    attestation_path = archived / "failure_attestation.json"
    if not inventory_path.is_file() or inventory_path.is_symlink():
        raise RuntimeError("Wave 59 failed attempt lacks a physical failure inventory")
    if not attestation_path.is_file() or attestation_path.is_symlink():
        raise RuntimeError("Wave 59 failed attempt lacks its external-trust attestation")
    failure_path = archived / "FAILURE.json"
    if not failure_path.is_file() or failure_path.is_symlink():
        raise RuntimeError("Wave 59 failed attempt lacks physical FAILURE.json")
    failure = _validate_failure_record(read_json(failure_path), archived)
    attestation = _require_exact_keys(
        read_json(attestation_path),
        {"algorithm", "payload", "signature_base64", "trusted_public_key_sha256"},
        "failure attestation",
    )
    if attestation["algorithm"] != "Ed25519":
        raise RuntimeError("Wave 59 failure attestation algorithm drifted")
    if not isinstance(attestation["signature_base64"], str):
        raise RuntimeError("Wave 59 failure attestation signature drifted")
    _require_sha256(
        attestation["trusted_public_key_sha256"],
        "failure attestation trust-root fingerprint",
    )
    anchor = _require_exact_keys(
        attestation["payload"],
        {
            "schema_version",
            "failure_inventory_sha256",
            "failure_sha256",
            "archived_path",
        },
        "failure attestation payload",
    )
    _require_sha256(anchor["failure_inventory_sha256"], "attested inventory hash")
    _require_sha256(anchor["failure_sha256"], "attested failure hash")
    verify_attestation(attestation, TRUSTED_PUBLIC_KEY)
    expected_anchor = {
        "schema_version": "wave59-failure-anchor-v1",
        "failure_inventory_sha256": sha256_file(inventory_path),
        "failure_sha256": sha256_file(failure_path),
        "archived_path": str(archived),
    }
    if attestation.get("payload") != expected_anchor:
        raise RuntimeError("Wave 59 failure attestation payload drifted")
    inventory = _require_exact_keys(
        read_json(inventory_path),
        {
            "schema_version",
            "records",
            "failure_records",
            "missing_required_through_last_journal",
            "extra",
            "overlap",
            "unclassified",
        },
        "failure inventory",
    )
    if inventory["schema_version"] != "wave59-failure-inventory-v1":
        raise RuntimeError("Wave 59 failure inventory schema drifted")
    if inventory["failure_records"] != [
        "FAILURE.json",
        "failure_inventory.json",
        "failure_attestation.json",
    ]:
        raise RuntimeError("Wave 59 failure inventory metadata list drifted")
    records = inventory["records"]
    if not isinstance(records, list):
        raise RuntimeError("Wave 59 failure inventory records are absent")
    validated_records = []
    for index, raw_row in enumerate(records):
        row = _require_exact_keys(
            raw_row, {"path", "class", "bytes", "sha256"}, f"failure inventory record {index}"
        )
        _require_safe_relative_path(row["path"], f"failure inventory record {index} path")
        if row["class"] not in FAILURE_RECORD_CLASSES:
            raise RuntimeError(f"Wave 59 failure inventory record {index} class drifted")
        if type(row["bytes"]) is not int or row["bytes"] < 0:
            raise RuntimeError(f"Wave 59 failure inventory record {index} bytes drifted")
        _require_sha256(row["sha256"], f"failure inventory record {index} hash")
        validated_records.append(row)
    by_path = {row["path"]: row for row in validated_records}
    if len(by_path) != len(records) or None in by_path:
        raise RuntimeError("Wave 59 failure inventory paths are not unique")
    symlinks = sorted(
        str(path.relative_to(archived))
        for path in archived.rglob("*")
        if path.is_symlink()
    )
    if symlinks:
        raise RuntimeError(f"Wave 59 failed attempt contains symlinks: {symlinks}")
    actual = {
        str(path.relative_to(archived))
        for path in archived.rglob("*")
        if path.is_file()
        and path not in {inventory_path, attestation_path}
    }
    if set(by_path) != actual:
        raise RuntimeError("Wave 59 failed-attempt inventory coverage drifted")
    classes = _artifact_classes(
        archived,
        run_role=failure["run_role"],
        recovery_context=failure["recovery_context"],
    )
    classes["scientific_exact"] = sorted(
        set(classes["scientific_exact"])
        | {
            "fit/fit_not_evaluable.json",
            "calibration/calibration_not_evaluable.json",
            "adjudication/monitor_not_evaluable.json",
        }
    )
    expected_class = {
        relative: class_name
        for class_name, paths in classes.items()
        for relative in paths
    }
    expected_class["FAILURE.json"] = "failure_record"
    for relative, row in sorted(by_path.items()):
        path = archived / relative
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"Wave 59 failed-attempt path is not physical: {relative}")
        if row["class"] != expected_class.get(relative):
            raise RuntimeError(
                f"Wave 59 failed-attempt class drifted: {relative}"
            )
        if row["bytes"] != path.stat().st_size or row["sha256"] != sha256_file(path):
            raise RuntimeError(f"Wave 59 failed-attempt hash drifted: {relative}")
    for field in ("missing_required_through_last_journal", "extra", "overlap", "unclassified"):
        if inventory.get(field) != []:
            raise RuntimeError(f"Wave 59 failed-attempt {field} is not empty")
    journals = []
    for phase in (
        "prepare",
        "fit",
        "calibrate_scores",
        "validate",
        "monitor_apply",
        "monitor_evaluate",
    ):
        journal_path = archived / "journals" / f"{phase}.json"
        if journal_path.is_file():
            journals.append(read_json(journal_path))
    missing, future = _failure_coverage(archived, journals, classes)
    if missing or future:
        raise RuntimeError(
            "Wave 59 failed-attempt derived coverage drifted: "
            f"missing={sorted(missing)}, future={sorted(future)}"
        )
    return inventory


def _validate_hash_map(payload: Any, label: str) -> dict[str, str]:
    if not isinstance(payload, dict):
        raise RuntimeError(f"Wave 59 {label} is not a hash map")
    result: dict[str, str] = {}
    for relative, digest in payload.items():
        _require_safe_relative_path(relative, f"{label} path")
        result[relative] = _require_sha256(digest, f"{label} hash for {relative}")
    return result


def _validate_access_receipt(
    receipt: Any,
    phase: str,
    status: str,
    journal_inputs: dict[str, str],
    original_root: Path,
) -> None:
    receipt = _require_exact_keys(
        receipt,
        {
            "phase",
            "status",
            "effective_uid",
            "effective_gid",
            "process_security",
            "threadpools",
            "stage_hashes",
            "forbidden_probes",
            "output_inventory_before_receipt",
            "benchmark_root_received",
        },
        f"{phase} access receipt",
    )
    if receipt["phase"] != phase or receipt["status"] != status:
        raise RuntimeError(f"Wave 59 {phase} access receipt state drifted")
    if receipt["effective_uid"] != 65534 or receipt["effective_gid"] != 65534:
        raise RuntimeError(f"Wave 59 {phase} resumed worker identity drifted")
    security = _require_exact_keys(
        receipt["process_security"],
        {"effective_capabilities_hex", "no_new_privileges", "supplementary_groups"},
        f"{phase} process security",
    )
    if security != {
        "effective_capabilities_hex": "0000000000000000",
        "no_new_privileges": 1,
        "supplementary_groups": [],
    }:
        raise RuntimeError(f"Wave 59 {phase} resumed worker security drifted")
    if type(receipt["benchmark_root_received"]) is not bool or receipt[
        "benchmark_root_received"
    ]:
        raise RuntimeError(f"Wave 59 {phase} benchmark-root receipt drifted")
    stage_hashes = _validate_hash_map(
        receipt["stage_hashes"], f"{phase} stage hashes"
    )
    expected_stage_files = set(PHASE_FILES[phase])
    if set(stage_hashes) != expected_stage_files:
        raise RuntimeError(f"Wave 59 {phase} stage coverage drifted")
    if {
        name: stage_hashes[name]
        for name in expected_stage_files - {"phase_request.json"}
    } != journal_inputs:
        raise RuntimeError(f"Wave 59 {phase} stage/input hashes drifted")
    request = {
        "phase": phase,
        "allowed_files": sorted(expected_stage_files),
        "sha256": journal_inputs,
    }
    if stage_hashes["phase_request.json"] != _json_payload_sha256(request):
        raise RuntimeError(f"Wave 59 {phase} request hash drifted")
    _validate_hash_map(
        receipt["output_inventory_before_receipt"], f"{phase} output inventory"
    )
    probes = receipt["forbidden_probes"]
    if not isinstance(probes, list):
        raise RuntimeError(f"Wave 59 {phase} forbidden-probe receipt drifted")
    expected_probes = [
        {
            "path_sha256": hashlib.sha256(
                str((original_root / relative).resolve(strict=False)).encode("utf-8")
            ).hexdigest(),
            "denied": True,
            "error_type": "PermissionError",
        }
        for relative in PHASE_PROBE_RELATIVES[phase]
    ]
    if probes != expected_probes:
        raise RuntimeError(f"Wave 59 {phase} forbidden-probe coverage drifted")
    for index, probe in enumerate(probes):
        probe = _require_exact_keys(
            probe, {"path_sha256", "denied", "error_type"}, f"{phase} probe {index}"
        )
        _require_sha256(probe["path_sha256"], f"{phase} probe {index} path hash")
        if probe["denied"] is not True or probe["error_type"] not in {
            "PermissionError",
            "FileNotFoundError",
        }:
            raise RuntimeError(f"Wave 59 {phase} forbidden probe was not denied")
    threadpools = receipt["threadpools"]
    if not isinstance(threadpools, list) or not threadpools:
        raise RuntimeError(f"Wave 59 {phase} threadpool receipt drifted")
    for index, pool in enumerate(threadpools):
        if not isinstance(pool, dict):
            raise RuntimeError(f"Wave 59 {phase} threadpool {index} drifted")
        api = pool.get("internal_api")
        expected = {
            "user_api", "internal_api", "num_threads", "prefix", "filepath", "version"
        }
        if api in {"openblas", "mkl", "blis"}:
            expected |= {"threading_layer", "architecture"}
        _require_exact_keys(pool, expected, f"{phase} threadpool {index}")
        if type(pool["num_threads"]) is not int or not 0 < pool["num_threads"] <= 4:
            raise RuntimeError(f"Wave 59 {phase} threadpool {index} exceeds contract")


def _validate_resumed_journal(
    journal: Any, phase: str, original_root: Path | None = None
) -> dict[str, Any]:
    if phase == "prepare":
        journal = _require_exact_keys(
            journal,
            {
                "schema_version",
                "phase",
                "status",
                "execution_mode",
                "preparation_freeze_sha256",
                "prepared_bundle_hashes",
                "maximum_truth_materialized",
                "next_state",
            },
            "prepare journal",
        )
        if (
            journal["schema_version"] != "wave59-phase-journal-v1"
            or journal["phase"] != "prepare"
            or journal["status"] != "PREPARED"
            or journal["next_state"] != "PREPARED"
            or journal["execution_mode"] not in {"fresh", "replay", "recovery"}
            or journal["maximum_truth_materialized"]
            != "prepared_all_splits_root_only"
        ):
            raise RuntimeError("Wave 59 prepare journal values drifted")
        _require_sha256(
            journal["preparation_freeze_sha256"], "prepare journal freeze hash"
        )
        bundles = _validate_hash_map(
            journal["prepared_bundle_hashes"], "prepared bundle hashes"
        )
        if set(bundles) != {
            "prepared/gate_fit_bundle.npz",
            "prepared/gate_select_truth_bundle.npz",
            "prepared/gate_select_inference_bundle.npz",
            "prepared/sealed_monitor_truth_bundle.npz",
            "prepared/sealed_monitor_inference_bundle.npz",
        }:
            raise RuntimeError("Wave 59 prepare journal bundle coverage drifted")
        return journal
    journal = _require_exact_keys(
        journal,
        {
            "schema_version",
            "phase",
            "status",
            "input_sha256",
            "output_sha256",
            "access_receipt",
            "duration_seconds",
            "max_rss_bytes",
            "maximum_truth_materialized",
        },
        f"{phase} journal",
    )
    expected_status = {
        "fit": {"FIT_COMPLETE", "NOT_EVALUABLE"},
        "calibrate_scores": {"CALIBRATION_FROZEN", "NOT_EVALUABLE"},
        "validate": {"VALIDATION_COMPLETE"},
        "monitor_apply": {"MONITOR_ACTIONS_FROZEN", "NOT_EVALUABLE"},
        "monitor_evaluate": {"COMPLETE"},
    }[phase]
    expected_truth = {
        "fit": "train",
        "calibrate_scores": "train",
        "validate": "validation",
        "monitor_apply": "validation",
        "monitor_evaluate": "monitor",
    }[phase]
    if (
        journal["schema_version"] != "wave59-phase-journal-v1"
        or journal["phase"] != phase
        or journal["status"] not in expected_status
        or journal["maximum_truth_materialized"] != expected_truth
    ):
        raise RuntimeError(f"Wave 59 {phase} journal values drifted")
    inputs = _validate_hash_map(journal["input_sha256"], f"{phase} journal inputs")
    outputs = _validate_hash_map(
        journal["output_sha256"], f"{phase} journal outputs"
    )
    expected_inputs = set(PHASE_FILES[phase]) - {"phase_request.json"}
    if set(inputs) != expected_inputs:
        raise RuntimeError(f"Wave 59 {phase} journal input coverage drifted")
    if not isinstance(journal["duration_seconds"], (int, float)) or isinstance(
        journal["duration_seconds"], bool
    ) or journal["duration_seconds"] < 0:
        raise RuntimeError(f"Wave 59 {phase} journal duration drifted")
    if type(journal["max_rss_bytes"]) is not int or journal["max_rss_bytes"] < 0:
        raise RuntimeError(f"Wave 59 {phase} journal RSS drifted")
    if original_root is None:
        raise RuntimeError(f"Wave 59 {phase} journal lacks its original root")
    _validate_access_receipt(
        journal["access_receipt"],
        phase,
        journal["status"],
        inputs,
        original_root,
    )
    receipt_outputs = journal["access_receipt"]["output_inventory_before_receipt"]
    if set(receipt_outputs) != set(outputs) or any(
        receipt_outputs[relative] != digest
        for relative, digest in outputs.items()
        if not (phase == "monitor_evaluate" and relative == "analysis.json")
    ):
        raise RuntimeError(f"Wave 59 {phase} journal output receipt drifted")
    return journal


def _utilities_npy_sha256(policy_manifest: Path | None) -> str:
    if policy_manifest is None:
        raise RuntimeError(
            "Wave 59 recovery requires the frozen policy manifest before copy"
        )
    utilities = load_utilities(policy_manifest.resolve(strict=True))
    with tempfile.SpooledTemporaryFile(max_size=1024 * 1024) as handle:
        np.save(handle, utilities)
        handle.seek(0)
        return hashlib.sha256(handle.read()).hexdigest()


def _expected_resumed_input_hashes(
    run_dir: Path,
    phase: str,
    config_path: Path,
    policy_manifest: Path | None,
) -> dict[str, str]:
    config = read_json(config_path.resolve(strict=True))
    if policy_manifest is not None and sha256_file(
        policy_manifest.resolve(strict=True)
    ) != config["source_binding"]["wave52_policy_manifest_sha256"]:
        raise RuntimeError(
            "Wave 59 recovery policy manifest differs from the bound upstream"
        )
    common = {
        "config.json": config_path.resolve(strict=True),
        "source_bindings.json": run_dir / "source_bindings.json",
        "preparation_freeze.json": run_dir / "preparation_freeze.json",
    }
    phase_paths: dict[str, Path] = {
        "fit": {
            "bundle.npz": run_dir / "prepared/gate_fit_bundle.npz",
        },
        "calibrate_scores": {
            "inference_bundle.npz": run_dir
            / "prepared/gate_select_inference_bundle.npz",
            "model_states_manifest.json": run_dir / "fit/model_states/manifest.json",
            "model_state_arrays.npz": run_dir / "fit/model_state_arrays.npz",
            "fit_freeze.json": run_dir / "fit/fit_freeze.json",
        },
        "validate": {
            "inference_bundle.npz": run_dir
            / "prepared/gate_select_inference_bundle.npz",
            "truth_bundle.npz": run_dir / "prepared/gate_select_truth_bundle.npz",
            "validation_scores.npz": run_dir / "calibration/validation_scores.npz",
            "validation_policy_arrays.npz": run_dir
            / "calibration/validation_policy_arrays.npz",
            "calibration_freeze.json": run_dir
            / "calibration/calibration_freeze.json",
        },
        "monitor_apply": {
            "inference_bundle.npz": run_dir
            / "prepared/sealed_monitor_inference_bundle.npz",
            "model_states_manifest.json": run_dir / "fit/model_states/manifest.json",
            "model_state_arrays.npz": run_dir / "fit/model_state_arrays.npz",
            "fit_freeze.json": run_dir / "fit/fit_freeze.json",
            "calibration_freeze.json": run_dir
            / "calibration/calibration_freeze.json",
            "validation_freeze.json": run_dir / "validation/validation_freeze.json",
        },
        "monitor_evaluate": {
            "truth_bundle.npz": run_dir / "prepared/sealed_monitor_truth_bundle.npz",
            "monitor_policy_arrays.npz": run_dir
            / "adjudication/monitor_policy_arrays.npz",
            "monitor_action_freeze.json": run_dir
            / "adjudication/monitor_action_freeze.json",
        },
    }[phase]
    paths = {**common, **phase_paths}
    expected = {
        name: sha256_file(path.resolve(strict=True))
        for name, path in sorted(paths.items())
    }
    if "utilities.npy" in PHASE_FILES[phase]:
        expected["utilities.npy"] = _utilities_npy_sha256(policy_manifest)
    if set(expected) != set(PHASE_FILES[phase]) - {"phase_request.json"}:
        raise RuntimeError(f"Wave 59 {phase} durable input contract drifted")
    return dict(sorted(expected.items()))


def _validate_resumed_journals(
    run_dir: Path,
    failure: dict[str, Any],
    config_path: Path,
    policy_manifest: Path | None,
) -> None:
    original_root = Path(failure["original_path"])
    analytical_journal_paths = [
        run_dir / "journals" / f"{phase}.json"
        for phase in (
            "fit",
            "calibrate_scores",
            "validate",
            "monitor_apply",
            "monitor_evaluate",
        )
    ]
    if any(path.is_file() for path in analytical_journal_paths):
        config = read_json(config_path.resolve(strict=True))
        if read_json(run_dir / "source_bindings.json") != config["source_binding"]:
            raise RuntimeError("Wave 59 recovery source bindings drifted")
        _ensure_preparation_authority(
            run_dir,
            run_dir / "prepared",
            config,
            config_path,
        )
    prepare_path = run_dir / "journals/prepare.json"
    if prepare_path.is_file():
        prepare = _validate_resumed_journal(read_json(prepare_path), "prepare")
        if prepare["preparation_freeze_sha256"] != sha256_file(
            run_dir / "preparation_freeze.json"
        ):
            raise RuntimeError("Wave 59 prepare journal freeze drifted")
        observed_bundles = {
            relative: sha256_file(run_dir / relative)
            for relative in prepare["prepared_bundle_hashes"]
        }
        if prepare["prepared_bundle_hashes"] != observed_bundles:
            raise RuntimeError("Wave 59 prepare journal bundles drifted")
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
        journal = _validate_resumed_journal(
            read_json(journal_path), phase, original_root
        )
        expected_inputs = _expected_resumed_input_hashes(
            run_dir,
            phase,
            config_path,
            policy_manifest,
        )
        if journal["input_sha256"] != expected_inputs:
            raise RuntimeError(f"Wave 59 resumed {phase} input values drifted")
        expected_outputs = _expected_phase_output_names(
            run_dir, phase, journal["status"]
        )
        if set(journal["output_sha256"]) != expected_outputs:
            raise RuntimeError(f"Wave 59 resumed {phase} output contract drifted")
        phase_number = {
            "fit": 1,
            "calibrate_scores": 2,
            "validate": 3,
            "monitor_apply": 4,
        }[phase]
        observed_outputs = {
            str(path.relative_to(destination)): sha256_file(path)
            for path in destination.rglob("*")
            if path.is_file()
            and _artifact_phase(str(path.relative_to(run_dir))) == phase_number
        }
        if journal["output_sha256"] != observed_outputs:
            raise RuntimeError(f"Wave 59 resumed {phase} output coverage drifted")
        for relative, expected in journal["output_sha256"].items():
            path = destination / relative
            if not path.is_file() or sha256_file(path) != expected:
                raise RuntimeError(f"Wave 59 resumed {phase} output drifted: {relative}")
    monitor_journal = run_dir / "journals/monitor_evaluate.json"
    if monitor_journal.is_file():
        journal = _validate_resumed_journal(
            read_json(monitor_journal), "monitor_evaluate", original_root
        )
        expected_inputs = _expected_resumed_input_hashes(
            run_dir,
            "monitor_evaluate",
            config_path,
            policy_manifest,
        )
        if journal["input_sha256"] != expected_inputs:
            raise RuntimeError(
                "Wave 59 resumed monitor_evaluate input values drifted"
            )
        if set(journal["output_sha256"]) != _expected_phase_output_names(
            run_dir, "monitor_evaluate", journal["status"]
        ):
            raise RuntimeError(
                "Wave 59 resumed monitor_evaluate output contract drifted"
            )
        _validate_promoted_evaluation_outputs(run_dir, journal)


def _validate_failure_journal_alignment(
    failure: dict[str, Any], run_dir: Path
) -> None:
    journals = []
    for phase in (
        "prepare",
        "fit",
        "calibrate_scores",
        "validate",
        "monitor_apply",
        "monitor_evaluate",
    ):
        path = run_dir / "journals" / f"{phase}.json"
        if path.is_file():
            journals.append(read_json(path))
    last = journals[-1] if journals else None
    expected_state = last["status"] if last else None
    expected_truth = last["maximum_truth_materialized"] if last else "none"
    if (
        failure["last_state"] != expected_state
        or failure["maximum_truth_materialized"] != expected_truth
    ):
        raise RuntimeError("Wave 59 failure record and durable journals differ")


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
        "coordinator_budget",
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


def _verified_preparation_attestation_invariants(run_dir: Path) -> dict[str, Any]:
    receipt = _require_exact_keys(
        read_json(run_dir / "preparation_attestation.json"),
        {"algorithm", "payload", "signature_base64", "trusted_public_key_sha256"},
        "replay preparation attestation",
    )
    verify_attestation(receipt, TRUSTED_PUBLIC_KEY.resolve(strict=True))
    payload = dict(receipt["payload"])
    records = dict(payload["records"])
    for relative in (
        "generation_receipt.json",
        "preparation_receipt.json",
        "journals/prepare.json",
    ):
        records.pop(relative)
    payload["records"] = records
    payload.pop("run_role")
    payload.pop("execution_mode")
    return payload


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
    if canonical:
        frozen_comparison = (
            read_json(replay / "config.snapshot.json").get("status") == FROZEN_STATUS
            and read_json(primary / "config.snapshot.json").get("status") == FROZEN_STATUS
        )
        operational["preparation_attestation.json"] = (
            _verified_preparation_attestation_invariants(replay)
            == _verified_preparation_attestation_invariants(primary)
            if frozen_comparison
            else _normalize_operational(read_json(replay / "preparation_attestation.json"))
            == _normalize_operational(read_json(primary / "preparation_attestation.json"))
        )
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
        "preparation_attestation.json",
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


def _expected_phase_output_names(
    run_dir: Path, phase: str, status: str
) -> set[str]:
    not_evaluable = {
        "fit": "fit_not_evaluable.json",
        "calibrate_scores": "calibration_not_evaluable.json",
        "monitor_apply": "monitor_not_evaluable.json",
    }
    if status == "NOT_EVALUABLE":
        if phase not in not_evaluable:
            raise RuntimeError(f"Wave 59 {phase} cannot terminate NOT_EVALUABLE")
        return {not_evaluable[phase]}
    if phase == "monitor_evaluate":
        return {"analysis.json", "analysis_arrays.npz", "bootstrap_indices.npz"}
    phase_number = {
        "fit": 1,
        "calibrate_scores": 2,
        "validate": 3,
        "monitor_apply": 4,
    }[phase]
    destination = {
        "fit": "fit",
        "calibrate_scores": "calibration",
        "validate": "validation",
        "monitor_apply": "adjudication",
    }[phase]
    classes = _artifact_classes(
        run_dir, run_role="primary", recovery_context=False
    )
    relative_prefix = destination + "/"
    return {
        relative.removeprefix(relative_prefix)
        for paths in classes.values()
        for relative in paths
        if relative.startswith(relative_prefix)
        and _artifact_phase(relative) == phase_number
    }
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
    all_terminal_outputs = set(terminal_output.values())
    required_at_stop = {
        terminal_output[terminal_phase],
        f"journals/{terminal_phase}.json",
    }
    result: dict[str, list[str]] = {}
    for class_name, paths in classes.items():
        selected = []
        for relative in paths:
            if (
                relative in all_terminal_outputs
                and relative != terminal_output[terminal_phase]
            ):
                continue
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
        if not path.is_file():
            continue
        relative = str(path.relative_to(run_dir))
        if relative in {"failure_inventory.json", "failure_attestation.json"}:
            continue
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
    try:
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
    except BaseException as signing_error:
        marker = {
            "schema_version": "wave59-unattested-failure-anchor-v1",
            "status": "UNATTESTED_SIGNING_FAILURE",
            "authoritative": False,
            "signing_error_type": type(signing_error).__name__,
            "signing_error_message_sha256": hashlib.sha256(
                str(signing_error).encode("utf-8")
            ).hexdigest(),
            "failure_sha256": sha256_file(run_dir / "FAILURE.json"),
        }
        try:
            write_json(
                run_dir / "failure_attestation_error.json", marker, mode=0o600
            )
            inventory_path = run_dir / "failure_inventory.json"
            inventory = read_json(inventory_path)
            inventory["failure_records"] = [
                "FAILURE.json",
                "failure_inventory.json",
                "failure_attestation_error.json",
            ]
            write_json(inventory_path, inventory, mode=0o600)
        finally:
            os.replace(run_dir, archived)
            _fsync_directory(archived.parent)
        return archived
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
    statuses = [
        str(row.get("status"))
        for row in journals
        if row.get("phase") != "prepare"
    ]
    if "NOT_EVALUABLE" in statuses and (
        statuses[-1] != "NOT_EVALUABLE" or statuses.count("NOT_EVALUABLE") != 1
    ):
        return {"journals:continued_after_terminal"}, set()
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
        "failure_attestation.json",
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
            "preparation_attestation.json",
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
    elif (
        (run_dir / "config.snapshot.json").is_file()
        and read_json(run_dir / "config.snapshot.json").get("status") != FROZEN_STATUS
        and (run_dir / "preparation_freeze.json").is_file()
        and read_json(run_dir / "preparation_freeze.json").get("status")
        == "TEST_ONLY_PREPARED_BUNDLES"
    ):
        test_bundle_paths = {
            path
            for values in classes.values()
            for path in values
            if path.startswith("prepared/")
        }
        required |= test_bundle_paths
        allowed |= test_bundle_paths
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
        expected_outputs = _expected_phase_output_names(
            run_dir, phase, str(journal.get("status"))
        )
        for relative in expected_outputs:
            promoted = (
                "analysis.json"
                if phase == "monitor_evaluate" and relative == "analysis.json"
                else f"{destination}/{relative}"
            )
            required.add(promoted)
        for relative in expected_outputs:
            promoted = (
                "analysis.json"
                if phase == "monitor_evaluate" and relative == "analysis.json"
                else f"{destination}/{relative}"
            )
            allowed.add(promoted)
        if str(journal.get("status")) == "NOT_EVALUABLE":
            allowed |= {"runtime.json", "artifact_manifest.json"}
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
    return required - actual, future | (actual - allowed)


def _coordinator_rss_bytes() -> int:
    try:
        status = Path("/proc/self/status").read_text(encoding="utf-8")
        return next(
            int(line.split()[1]) * 1024
            for line in status.splitlines()
            if line.startswith("VmRSS:")
        )
    except (FileNotFoundError, StopIteration):
        return 0


@contextmanager
def analytical_coordinator_budget(
    config: dict[str, Any], seconds: float
) -> Any:
    """Keep the analytical coordinator inside wall/RSS limits through fsync."""
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Wave 59 analytical coordinator can see CUDA")
    if seconds <= 0:
        raise RuntimeError("Wave 59 analytical coordinator has no wall-time budget")
    thread_limiter = threadpool_limits(limits=4)
    pools = threadpool_info()
    if not pools or any(
        not 0 < int(pool.get("num_threads", 0)) <= 4 for pool in pools
    ):
        thread_limiter.restore_original_limits()
        raise RuntimeError(
            "Wave 59 analytical coordinator exceeded the four-thread contract"
        )
    rss_limit = int(config["runtime_budget"]["max_rss_bytes"])
    started = time.monotonic()
    state: dict[str, Any] = {
        "duration_seconds": 0.0,
        "max_rss_bytes": _coordinator_rss_bytes(),
        "max_seconds": float(seconds),
        "max_rss_allowed_bytes": rss_limit,
        "cuda_visible_devices": "",
        "threadpools": [
            {
                "internal_api": pool.get("internal_api"),
                "prefix": pool.get("prefix"),
                "num_threads": int(pool.get("num_threads", 0)),
            }
            for pool in pools
        ],
        "budget_enforced": True,
        "_started_monotonic": started,
    }
    if int(state["max_rss_bytes"]) > rss_limit:
        thread_limiter.restore_original_limits()
        raise RuntimeError("Wave 59 analytical coordinator exceeded RSS budget")
    stop = threading.Event()

    def raise_budget(signum: int, frame: Any) -> None:
        kind = "RSS" if signum == signal.SIGUSR1 else "wall-time"
        raise RuntimeError(
            f"Wave 59 analytical coordinator exceeded {kind} budget"
        )

    def watch_rss() -> None:
        while not stop.wait(0.1):
            rss = _coordinator_rss_bytes()
            state["duration_seconds"] = time.monotonic() - started
            state["max_rss_bytes"] = max(int(state["max_rss_bytes"]), rss)
            if rss > rss_limit:
                os.kill(os.getpid(), signal.SIGUSR1)
                return

    previous_alarm = signal.getsignal(signal.SIGALRM)
    previous_rss = signal.getsignal(signal.SIGUSR1)
    signal.signal(signal.SIGALRM, raise_budget)
    signal.signal(signal.SIGUSR1, raise_budget)
    watcher = threading.Thread(
        target=watch_rss, name="wave59-analytical-rss-budget", daemon=True
    )
    watcher.start()
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield state
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        stop.set()
        watcher.join(timeout=1.0)
        state["duration_seconds"] = time.monotonic() - started
        state["max_rss_bytes"] = max(
            int(state["max_rss_bytes"]), _coordinator_rss_bytes()
        )
        signal.signal(signal.SIGALRM, previous_alarm)
        signal.signal(signal.SIGUSR1, previous_rss)
        thread_limiter.restore_original_limits()


def _checkpoint_analytical_budget(state: dict[str, Any]) -> tuple[float, int]:
    duration = time.monotonic() - float(state["_started_monotonic"])
    rss = max(int(state["max_rss_bytes"]), _coordinator_rss_bytes())
    state["duration_seconds"] = duration
    state["max_rss_bytes"] = rss
    if duration > float(state["max_seconds"]):
        raise RuntimeError("Wave 59 analytical coordinator exceeded wall-time budget")
    if rss > int(state["max_rss_allowed_bytes"]):
        raise RuntimeError("Wave 59 analytical coordinator exceeded RSS budget")
    return duration, rss


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


def _execute_once(
    prepared: Path,
    policy_manifest: Path,
    output: Path,
    config_path: Path,
    reference_dir: Path | None = None,
) -> Path:
    started = time.monotonic()
    config = read_json(config_path)
    validate_pre_draw_config(config)
    run_role = "replay" if reference_dir is not None else "primary"
    prepared_input = prepared.absolute()
    validate_execution_bindings(config, config_path, prepared_input)
    prepared_resolved = prepared_input.resolve(strict=True)
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
        validate_execution_bindings(config, config_path, prepared_resolved)
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
        validate_execution_bindings(config, config_path, prepared_resolved)
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
        validate_execution_bindings(config, config_path, prepared_resolved)
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
        validate_execution_bindings(config, config_path, prepared_resolved)
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


def _preparation_duration(root: Path) -> float:
    receipt = root / "preparation_receipt.json"
    if not receipt.is_file():
        return 0.0
    return float(
        read_json(receipt).get("coordinator_budget", {}).get("duration_seconds", 0.0)
    )


def _finalize_accounted_runtime(
    output: Path,
    config: dict[str, Any],
    state: dict[str, Any],
    preparation_duration: float,
    reference_dir: Path | None,
    *,
    run_role: str,
) -> None:
    """Publish conservative final accounting while the watchdog remains active."""
    runtime_path = output / "runtime.json"
    if not runtime_path.is_file():
        raise RuntimeError("Wave 59 analytical run ended without runtime accounting")
    canonical = (output / "benchmark").is_dir()
    reference_total = 0.0
    if reference_dir is not None:
        reference = reference_dir.resolve(strict=True)
        reference_runtime = read_json(reference / "runtime.json")
        reference_total = float(reference_runtime["duration_seconds"]) + _preparation_duration(
            reference
        )
    recorded_duration = -1.0
    recorded_rss = -1
    for _ in range(5):
        duration, rss = _checkpoint_analytical_budget(state)
        # A small conservative margin accounts for the final runtime/manifest
        # fsyncs instead of reporting a timestamp taken before they happen.
        recorded_duration = duration + 0.25
        recorded_rss = rss
        if recorded_duration > float(state["max_seconds"]):
            raise RuntimeError(
                "Wave 59 analytical finalization exceeded wall-time budget"
            )
        runtime = read_json(runtime_path)
        runtime["duration_seconds"] = recorded_duration
        runtime["preparation_duration_seconds"] = preparation_duration
        runtime["total_run_seconds"] = preparation_duration + recorded_duration
        runtime["max_rss_bytes"] = max(int(runtime.get("max_rss_bytes", 0)), rss)
        runtime["coordinator_budget"] = {
            key: state[key]
            for key in (
                "max_seconds",
                "max_rss_allowed_bytes",
                "cuda_visible_devices",
                "threadpools",
                "budget_enforced",
            )
        }
        runtime["coordinator_budget"].update(
            {"duration_seconds": recorded_duration, "max_rss_bytes": rss}
        )
        if reference_dir is not None:
            combined = reference_total + preparation_duration + recorded_duration
            combined_limit = float(
                config["runtime_budget"]["max_seconds_primary_plus_replay"]
            )
            if combined > combined_limit:
                raise RuntimeError(
                    "Wave 59 primary plus replay exceeded combined runtime budget"
                )
            runtime["primary_plus_replay_seconds"] = combined
            runtime["combined_budget_seconds"] = int(combined_limit)
        write_json(runtime_path, runtime)
        if canonical:
            write_artifact_manifest(output, run_role=run_role)
        final_duration, final_rss = _checkpoint_analytical_budget(state)
        if final_duration <= recorded_duration and final_rss <= recorded_rss:
            return
    raise RuntimeError("Wave 59 final resource accounting did not stabilize")


def execute(
    prepared: Path,
    policy_manifest: Path,
    output: Path,
    config_path: Path,
    reference_dir: Path | None = None,
) -> Path:
    """Run and finalize Wave 59 under one coordinator wall/RSS envelope."""
    config_path = config_path.resolve(strict=True)
    config = read_json(config_path)
    prepared_input = prepared.absolute()
    if config.get("status") == FROZEN_STATUS:
        _require_root_owned_mode(
            prepared_input, 0o700, "caller preparation root", directory=True
        )
    prepared_root = prepared_input.resolve(strict=True)
    preparation_duration = _preparation_duration(prepared_root)
    allowed = float(config["runtime_budget"]["max_seconds_per_run"]) - preparation_duration
    if reference_dir is not None:
        reference = reference_dir.resolve(strict=True)
        combined_allowed = (
            float(config["runtime_budget"]["max_seconds_primary_plus_replay"])
            - _preparation_duration(reference)
            - float(read_json(reference / "runtime.json")["duration_seconds"])
            - preparation_duration
        )
        allowed = min(allowed, combined_allowed)
    run_role = "replay" if reference_dir is not None else "primary"
    with analytical_coordinator_budget(config, allowed) as budget_state:
        result = _execute_once(
            prepared_input,
            policy_manifest,
            output,
            config_path,
            reference_dir,
        )
        _finalize_accounted_runtime(
            result,
            config,
            budget_state,
            preparation_duration,
            reference_dir,
            run_role=run_role,
        )
        return result


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
    prepared_arg = args.prepared_dir.absolute()
    if args.resume_from is not None:
        restore_identical_hash_attempt(
            args.resume_from.resolve(strict=True),
            output,
            args.config.resolve(strict=True),
            args.policy_manifest.resolve(strict=True),
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
