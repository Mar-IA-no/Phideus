#!/usr/bin/env python3
"""Generate, execute, and audit the Wave 49 classical benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from geometria_proporcional.wave49_checker import (  # noqa: E402
    freeze_execution_manifest,
    freeze_prediction_manifest,
    validate_all,
    validate_manifest,
    validate_oracle_consistency,
    validate_prediction_manifest,
    validate_sealed_alignment,
    validate_semantic_attestation,
    validate_visible_package,
)
from geometria_proporcional.wave49_evaluator import evaluate_benchmark  # noqa: E402
from geometria_proporcional.wave49_executor import run_restricted_executor  # noqa: E402
from geometria_proporcional.wave49_generator import generate_benchmark  # noqa: E402
from geometria_proporcional.wave49_mutations import run_mutation_suite  # noqa: E402
from geometria_proporcional.wave49_oracle import compute_oracle  # noqa: E402
from geometria_proporcional.wave49_schema import ProtocolConfig, default_protocol_config  # noqa: E402
from geometria_proporcional.wave49_selector import SELECTORS  # noqa: E402

WORKER = REPO_ROOT / "experiments" / "geometria_proporcional" / "_wave49_executor_worker.py"
DEFAULT_PUBLIC_KEY = (
    REPO_ROOT / "experiments" / "geometria_proporcional" / "keys"
    / "wave49_attestation_public.pem"
)


def _load_config(output_dir: Path) -> ProtocolConfig:
    data = json.loads((output_dir / "protocol_config.json").read_text(encoding="utf-8"))
    return ProtocolConfig.from_dict(data)


def _source_paths(
    config_json: Path | None = None,
    trusted_public_key_path: Path = DEFAULT_PUBLIC_KEY,
) -> dict[str, Path]:
    package = REPO_ROOT / "src" / "geometria_proporcional"
    paths = {
        "runner": Path(__file__).resolve(),
        "worker": WORKER,
        "trusted_attestation_public_key": Path(trusted_public_key_path),
        **{
            path.name: path
            for path in sorted(package.glob("wave49_*.py"))
        },
    }
    test_path = REPO_ROOT / "tests" / "test_wave49_relational_benchmark.py"
    if test_path.exists():
        paths["protocol_tests"] = test_path
    if config_json is not None:
        paths["requested_config"] = config_json.resolve()
    return paths


def _git_provenance() -> dict[str, object]:
    def git(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args], cwd=REPO_ROOT, text=True, capture_output=True, check=False
        )
        return completed.stdout.strip() if completed.returncode == 0 else "UNAVAILABLE"

    status = git("status", "--porcelain=v1", "--untracked-files=all")
    return {
        "head": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status and status != "UNAVAILABLE"),
        "status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
    }


def _invocation(command: str, output_dir: Path, smoke: bool, config_json: Path | None = None) -> dict[str, object]:
    return {
        "command": command,
        "argv": sys.argv,
        "output_dir": str(output_dir),
        "smoke": smoke,
        "config_json": str(config_json.resolve()) if config_json else None,
        "cwd": str(Path.cwd()),
        "git": _git_provenance(),
    }


def _execute_and_freeze(
    output_dir: Path,
    config: ProtocolConfig,
    invocation: dict[str, object],
    config_json: Path | None = None,
    trusted_public_key_path: Path = DEFAULT_PUBLIC_KEY,
) -> None:
    validate_semantic_attestation(output_dir, trusted_public_key_path)
    receipt = run_restricted_executor(output_dir, config, WORKER, REPO_ROOT)
    if receipt["prediction_row_counts"] != {
        split: count * len(SELECTORS)
        for split, count in receipt["visible_fixture_counts"].items()
    }:
        raise RuntimeError("restricted executor prediction counts are inconsistent")
    freeze_prediction_manifest(
        output_dir,
        invocation,
        _source_paths(config_json, trusted_public_key_path),
    )
    validate_prediction_manifest(output_dir)


def run_all(
    output_dir: Path,
    config: ProtocolConfig,
    smoke: bool,
    force: bool = False,
    config_json: Path | None = None,
    generation_key: bytes | None = None,
    identity_key: bytes | None = None,
    commitment_key: bytes | None = None,
    attestation_private_key_path: Path | None = None,
    trusted_public_key_path: Path = DEFAULT_PUBLIC_KEY,
) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not force:
            raise RuntimeError(f"output directory is not empty: {output_dir}; pass --force to replace it")
        shutil.rmtree(output_dir)
    invocation = _invocation("all", output_dir, smoke, config_json)
    manifest = generate_benchmark(
        output_dir,
        config,
        generation_key=generation_key,
        identity_key=identity_key,
        commitment_key=commitment_key,
        attestation_private_key_path=attestation_private_key_path,
        trusted_public_key_path=trusted_public_key_path,
    )
    validate_manifest(output_dir)
    validate_semantic_attestation(output_dir, trusted_public_key_path)
    validate_visible_package(output_dir, config)
    validate_sealed_alignment(output_dir, config)

    _execute_and_freeze(
        output_dir, config, invocation, config_json, trusted_public_key_path
    )
    compute_oracle(output_dir, config)
    validate_oracle_consistency(output_dir)
    evaluate_benchmark(output_dir, config)
    mutations = run_mutation_suite(output_dir, config, trusted_public_key_path)
    freeze_execution_manifest(
        output_dir,
        invocation,
        _source_paths(config_json, trusted_public_key_path),
    )
    checks = validate_all(
        output_dir, config, {spec.name for spec in SELECTORS}, trusted_public_key_path
    )
    print(json.dumps({
        "generated": manifest["counts"],
        "checks": {
            "visible": checks["visible_counts"],
            "predictions": checks["prediction_counts"],
            "ledger_events": checks["ledger_events"],
            "mutations_rejected": sum(row["status"] == "REJECTED" for row in mutations),
        },
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("all", "generate", "execute", "oracle", "evaluate", "check"))
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "data" / "geometria_proporcional" / "wave49")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--config-json", type=Path)
    parser.add_argument(
        "--attestation-private-key",
        type=Path,
        default=Path(os.environ["WAVE49_ATTESTATION_PRIVATE_KEY"])
        if "WAVE49_ATTESTATION_PRIVATE_KEY" in os.environ else None,
        help="external Ed25519 private key; required for all/generate and never copied into the artifact",
    )
    parser.add_argument(
        "--attestation-public-key",
        type=Path,
        default=DEFAULT_PUBLIC_KEY,
        help="trusted Ed25519 public key outside the run artifact",
    )
    parser.add_argument(
        "--replay-secrets-from",
        type=Path,
        help="reuse prior sealed generation, identity, and commitment keys for an exact replay",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    trusted_public_key_path = args.attestation_public_key.resolve()

    if args.smoke and args.config_json:
        raise ValueError("--smoke and --config-json are mutually exclusive")
    replay_root = args.replay_secrets_from.resolve() if args.replay_secrets_from else None
    replay_config = (
        ProtocolConfig.from_dict(json.loads((replay_root / "protocol_config.json").read_text(encoding="utf-8")))
        if replay_root else None
    )
    requested_config = (
        ProtocolConfig.from_dict(json.loads(args.config_json.read_text(encoding="utf-8")))
        if args.config_json
        else default_protocol_config(smoke=args.smoke) if args.smoke or replay_config is None
        else replay_config
    )
    if replay_config is not None and replay_config != requested_config:
        raise ValueError("replay protocol config does not match the requested config")
    generation_key = identity_key = commitment_key = None
    if replay_root is not None:
        generation_key = bytes.fromhex(json.loads(
            (replay_root / "sealed" / "generation_secret.json").read_text(encoding="utf-8")
        )["key_hex"])
        identity_key = bytes.fromhex(json.loads(
            (replay_root / "sealed" / "identity_secret.json").read_text(encoding="utf-8")
        )["key_hex"])
        commitment_key = bytes.fromhex(json.loads(
            (replay_root / "sealed" / "semantic_commitment_secret.json").read_text(encoding="utf-8")
        )["key_hex"])

    if args.command == "all":
        run_all(
            output_dir, requested_config, args.smoke, args.force, args.config_json,
            generation_key=generation_key, identity_key=identity_key,
            commitment_key=commitment_key,
            attestation_private_key_path=args.attestation_private_key,
            trusted_public_key_path=trusted_public_key_path,
        )
        return
    if args.command == "generate":
        if output_dir.exists() and any(output_dir.iterdir()) and not args.force:
            raise RuntimeError("output directory is not empty; pass --force to replace it")
        if output_dir.exists() and args.force:
            shutil.rmtree(output_dir)
        generate_benchmark(
            output_dir,
            requested_config,
            generation_key=generation_key,
            identity_key=identity_key,
            commitment_key=commitment_key,
            attestation_private_key_path=args.attestation_private_key,
            trusted_public_key_path=trusted_public_key_path,
        )
        return

    config = _load_config(output_dir)
    invocation = _invocation(args.command, output_dir, args.smoke, args.config_json)
    if args.command == "execute":
        _execute_and_freeze(
            output_dir,
            config,
            invocation,
            args.config_json,
            trusted_public_key_path,
        )
    elif args.command == "oracle":
        validate_prediction_manifest(output_dir)
        validate_semantic_attestation(output_dir, trusted_public_key_path)
        compute_oracle(output_dir, config)
        validate_oracle_consistency(output_dir)
    elif args.command == "evaluate":
        validate_prediction_manifest(output_dir)
        validate_semantic_attestation(output_dir, trusted_public_key_path)
        validate_oracle_consistency(output_dir)
        evaluate_benchmark(output_dir, config)
        run_mutation_suite(output_dir, config, trusted_public_key_path)
        freeze_execution_manifest(
            output_dir,
            invocation,
            _source_paths(args.config_json, trusted_public_key_path),
        )
    elif args.command == "check":
        validate_all(
            output_dir, config, {spec.name for spec in SELECTORS}, trusted_public_key_path
        )


if __name__ == "__main__":
    main()
