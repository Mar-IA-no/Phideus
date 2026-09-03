#!/usr/bin/env python3
"""Escrow, generate, and blindly infer the fresh Wave 56 Stage 1 benchmark."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import pwd
import grp
import secrets
import shutil
import stat
import subprocess
import sys
import tempfile
from typing import Any, Callable

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
from geometria_proporcional.wave49_generator import generate_benchmark  # noqa: E402
from geometria_proporcional.wave49_schema import (  # noqa: E402
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wave50-dir", type=Path, required=True)
    parser.add_argument("--wave51-dir", type=Path, required=True)
    parser.add_argument("--wave52-dir", type=Path, required=True)
    parser.add_argument("--wave54-dir", type=Path, required=True)
    parser.add_argument("--wave55-dir", type=Path, required=True)
    parser.add_argument("--stage0-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=CONFIG_DEFAULT)
    parser.add_argument("--attestation-private-key", type=Path, required=True)
    parser.add_argument("--replay-secrets-from", type=Path)
    parser.add_argument("--recovery-secrets-from", type=Path)
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
    if config.get("schema_version") != "wave56-contextual-residual-gate-stage1-v1":
        raise RuntimeError("prospective schema version drifted")
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
    if int(config.get("minimums", {}).get("absent_support_tokens", -1)) != 30:
        raise RuntimeError("absent-support minimum drifted")
    sources = config.get("required_execution_sources")
    if not isinstance(sources, list) or not sources or len(sources) != len(set(sources)):
        raise RuntimeError("execution-source manifest is absent or contains duplicates")
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
    if output.parent != canonical_parent:
        raise ValueError(f"Wave 56 outputs must live directly under {canonical_parent}")
    if output.name not in {primary_name, replay_name}:
        raise ValueError(f"output name must be {primary_name!r} or {replay_name!r}")
    if args.replay_secrets_from and args.recovery_secrets_from:
        raise ValueError("replay and recovery modes are mutually exclusive")

    if output.name == replay_name:
        if not args.replay_secrets_from or not args.reference_dir:
            raise ValueError("replay requires --replay-secrets-from and --reference-dir")
        source = args.replay_secrets_from.resolve(strict=True)
        reference = args.reference_dir.resolve(strict=True)
        if source != reference or source == output or source.name != primary_name:
            raise ValueError("replay must use the distinct canonical primary as reference/key source")
        if not has_escrow(source):
            raise RuntimeError("replay source lacks a durable escrow")
        if output.exists() and not args.force:
            raise FileExistsError("existing replay requires --force archival")
        return "replay"

    if args.replay_secrets_from or args.reference_dir:
        raise ValueError("primary output cannot use replay arguments")
    prior = archived_attempts(output, primary_name)
    key_bearing = [path for path in prior if has_escrow(path)]
    if args.recovery_secrets_from:
        source = args.recovery_secrets_from.resolve(strict=True)
        allowed = [path.resolve() for path in prior]
        if source != output and source not in allowed:
            raise ValueError("recovery escrow must come from this primary or one of its archives")
        if not has_escrow(source):
            raise RuntimeError("recovery source lacks a durable escrow")
        if output.exists() and not args.force:
            raise ValueError("recovering an existing primary requires --force archival")
        return "recovery"
    if output.exists():
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
    if config_path != CONFIG_DEFAULT.resolve():
        raise ValueError("fresh run must use the canonical prospective config path")
    if args.attestation_private_key.is_symlink():
        raise RuntimeError("attestation private key cannot be a symlink")
    private_key = args.attestation_private_key.resolve(strict=True)
    if not private_key.is_file():
        raise RuntimeError("attestation private key must be one existing regular file")

    commit, source_hashes = require_sources_at_head(config["required_execution_sources"])
    binding = config["source_binding"]
    wave50 = args.wave50_dir.resolve(strict=True)
    wave51 = args.wave51_dir.resolve(strict=True)
    wave52 = args.wave52_dir.resolve(strict=True)
    wave54 = args.wave54_dir.resolve(strict=True)
    wave55 = args.wave55_dir.resolve(strict=True)
    stage0 = args.stage0_dir.resolve(strict=True)
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
    selection = json.loads((stage0 / "selection_freeze.json").read_text(encoding="utf-8"))
    analysis = json.loads((stage0 / "analysis_core.json").read_text(encoding="utf-8"))
    expected_model = config["primary_model"]
    if selection.get("selected_family") != expected_model["family"]:
        raise RuntimeError("Stage 0 selected family differs from prospective freeze")
    if float(selection.get("selected_params", {}).get("alpha")) != float(expected_model["alpha"]):
        raise RuntimeError("Stage 0 selected alpha differs from prospective freeze")
    if tuple(analysis.get("feature_names", ())) != FEATURE_NAMES:
        raise RuntimeError("Stage 0 feature schema differs from prospective freeze")
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
    escrow_stat = escrow_path.stat()
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


def validate_reused_escrow(source: Path, contract: dict[str, Any]) -> dict[str, Any]:
    escrow = read_escrow(source)
    if escrow.get("contract") != contract:
        raise RuntimeError("replay/recovery commit, config, sources, or bindings differ from escrow")
    freeze_path = source / FREEZE_NAME
    if freeze_path.exists():
        freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
        if freeze != public_freeze_from_escrow(escrow):
            raise RuntimeError("source pre-generation freeze does not verify against escrow")
    return escrow


def archive_output(output: Path, reason: str) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    archived = output.with_name(f"{output.name}.{reason}_{stamp}")
    os.replace(output, archived)
    fsync_directory(output.parent)
    return archived


def prepare_output(output: Path, force: bool) -> Path | None:
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
        subprocess.run(command, check=True, cwd=stage, env=env)
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


def compare_preparation(replay: Path, primary: Path, config: dict[str, Any]) -> dict[str, bool]:
    if replay.resolve() == primary.resolve():
        raise ValueError("replay cannot reference itself")
    checks: dict[str, bool] = {}
    replay_manifest = json.loads((replay / "benchmark/manifest.json").read_text(encoding="utf-8"))
    primary_manifest = json.loads((primary / "benchmark/manifest.json").read_text(encoding="utf-8"))
    commitment_names = (
        "generation_key_commitment",
        "identity_key_commitment",
        "semantic_commitment_key_commitment",
    )
    checks["key_commitments"] = all(replay_manifest[name] == primary_manifest[name] for name in commitment_names)
    for relative in ["benchmark/protocol_config.json", *(f"benchmark/visible/{split}.jsonl" for split in SPLITS)]:
        checks[f"content:{relative}"] = digest(replay / relative) == digest(primary / relative)
    for split in SPLITS:
        for seed in config["seeds"]:
            relative = Path("inference/logits") / f"seed{seed}__{split}.npz"
            checks[f"array:{relative}"] = array_exact(replay / relative, primary / relative)
    replay_freeze = json.loads((replay / "preparation_freeze.json").read_text(encoding="utf-8"))
    primary_freeze = json.loads((primary / "preparation_freeze.json").read_text(encoding="utf-8"))
    checks["preparation_freeze"] = replay_freeze == primary_freeze
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
    keys_override: tuple[bytes, bytes, bytes] | None = None,
    protocol_override: Any | None = None,
    trusted_public_key_path: Path = PUBLIC_KEY,
    generation_fn: Callable[..., dict[str, Any]] = generate_benchmark,
    crash_hook: Callable[[str, Path], None] | None = None,
) -> None:
    if reused_escrow is not None and keys_override is not None:
        raise ValueError("recovery/replay keys come only from the durable escrow")
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
    escrow = make_escrow(contract, keys)
    if reused_escrow is not None and escrow != reused_escrow:
        raise RuntimeError("reused escrow was not preserved byte-semantically")
    if crash_hook:
        crash_hook("before_escrow", output)
    escrow_sha256 = atomic_write_json(output / ESCROW_NAME, escrow, mode=0o600)
    if crash_hook:
        crash_hook("after_escrow", output)
    atomic_write_json(output / FREEZE_NAME, public_freeze_from_escrow(escrow), mode=0o644)
    verify_escrow_and_freeze(output, escrow)
    if crash_hook:
        crash_hook("after_pre_generation_freeze", output)

    protocol = protocol_override if protocol_override is not None else default_protocol_config(smoke=False)
    benchmark = output / "benchmark"
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
    binding = config["source_binding"]
    if manifest["generation_key_commitment"] == binding["wave50_generation_key_commitment"]:
        raise RuntimeError("fresh generation commitment unexpectedly equals Wave 50")
    visible_hashes = {
        split: digest(benchmark / "visible" / f"{split}.jsonl") for split in SPLITS
    }
    expected_rows = int(config["fresh_benchmark"]["expected_visible_fixtures_per_split"])
    if any(int(manifest["counts"][split]) != expected_rows for split in SPLITS):
        raise RuntimeError("fresh benchmark visible split count differs from prospective freeze")
    expected_tokens = int(
        config["fresh_benchmark"]["expected_eligible_pair_tokens_per_split"]
    )
    sealed_pair_token_counts = {
        split: len(
            {
                str(row["pair_token"])
                for row in read_jsonl(benchmark / "sealed" / f"{split}.jsonl")
            }
        )
        for split in SPLITS
    }
    if any(count != expected_tokens for count in sealed_pair_token_counts.values()):
        raise RuntimeError("fresh benchmark pair-token count differs from prospective freeze")
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
        "manifest_sha256": digest(benchmark / "manifest.json"),
        "visible_sha256": visible_hashes,
        "sealed_pair_token_counts": sealed_pair_token_counts,
        "sealed_root_owner": 0,
        "sealed_root_mode": "0700",
        "oracle_materialized": False,
    }
    atomic_write_json(output / "generation_receipt.json", generation_receipt, mode=0o644)

    inference = stage_and_infer(
        output,
        benchmark,
        args.wave51_dir.resolve(strict=True),
        config_path,
        config,
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
        "bundles_present": False,
        "fit_operations": False,
        "physical_splits": config["physical_splits"],
    }
    atomic_write_json(output / "preparation_freeze.json", preparation_freeze, mode=0o644)
    verify = json.loads((output / "preparation_freeze.json").read_text(encoding="utf-8"))
    if verify != preparation_freeze:
        raise RuntimeError("preparation freeze failed post-publication verification")
    if crash_hook:
        crash_hook("after_preparation_freeze", output)

    replay_checks = None
    if mode == "replay":
        replay_checks = compare_preparation(output, args.reference_dir.resolve(strict=True), config)
        atomic_write_json(
            output / "preparation_replay.json",
            {
                "phase": "wave56-preparation-exact-replay",
                "checks": replay_checks,
                "all_exact": all(replay_checks.values()),
            },
            mode=0o644,
        )
    atomic_write_json(
        output / "preparation_receipt.json",
        {
            "phase": "wave56-stage1-preparation-complete",
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "execution_mode": mode,
            "preparation_freeze_sha256": digest(output / "preparation_freeze.json"),
            "generation_receipt_sha256": digest(output / "generation_receipt.json"),
            "replay_exact": all(replay_checks.values()) if replay_checks else None,
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
    keys_override: tuple[bytes, bytes, bytes] | None = None,
    protocol_override: Any | None = None,
    trusted_public_key_path: Path = PUBLIC_KEY,
    generation_fn: Callable[..., dict[str, Any]] = generate_benchmark,
    crash_hook: Callable[[str, Path], None] | None = None,
) -> Path | None:
    """Execute one preparation attempt and archive every failed physical state."""
    archived = prepare_output(output, force)
    try:
        execute_preparation(
            args,
            output,
            config_path,
            config,
            mode,
            contract,
            reused_escrow,
            keys_override=keys_override,
            protocol_override=protocol_override,
            trusted_public_key_path=trusted_public_key_path,
            generation_fn=generation_fn,
            crash_hook=crash_hook,
        )
    except BaseException as error:
        if output.exists():
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
    receipt_path = output / "preparation_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["superseded_output"] = str(archived) if archived else None
    atomic_write_json(receipt_path, receipt, mode=0o644)
    return archived


def main() -> None:
    os.umask(0o077)
    args = parse_args()
    config_path = args.config.resolve(strict=True)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output = args.output_dir.resolve()
    mode = validate_invocation(args, output, config)

    # This entire preflight is intentionally before output creation or archival.
    contract = preparation_preflight(args, config_path, config)
    reused_escrow = None
    if mode in {"replay", "recovery"}:
        source_arg = args.replay_secrets_from if mode == "replay" else args.recovery_secrets_from
        reused_escrow = validate_reused_escrow(source_arg.resolve(strict=True), contract)

    run_preparation_transaction(
        args,
        output,
        config_path,
        config,
        mode,
        contract,
        reused_escrow,
        force=args.force,
    )
    print(json.dumps({"state": "PREPARED", "execution_mode": mode}, sort_keys=True))


if __name__ == "__main__":
    main()
