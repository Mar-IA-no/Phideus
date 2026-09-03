#!/usr/bin/env python3
"""Run the prospectively frozen Wave 50 matched neural experiment on CPU."""

from __future__ import annotations

import argparse
import json
import os
import pwd
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from geometria_proporcional.wave49_checker import (  # noqa: E402
    freeze_prediction_manifest as freeze_classical_predictions,
    validate_manifest,
    validate_prediction_manifest,
    validate_sealed_alignment,
    validate_semantic_attestation,
    validate_visible_package,
    validate_oracle_rows,
)
from geometria_proporcional.wave49_executor import run_restricted_executor  # noqa: E402
from geometria_proporcional.wave49_generator import generate_benchmark  # noqa: E402
from geometria_proporcional.wave49_oracle import compute_oracle_splits  # noqa: E402
from geometria_proporcional.wave49_schema import (  # noqa: E402
    ProtocolConfig,
    default_protocol_config,
    read_jsonl,
    sha256_file,
    write_json,
)
from geometria_proporcional.wave50_neural import load_labeled_records, prepare_examples, FeatureNormalizer  # noqa: E402
from geometria_proporcional.wave50_mutations import run_mutation_suite  # noqa: E402
from geometria_proporcional.wave50_protocol import (  # noqa: E402
    ARMS,
    CHECKPOINT_VARIANTS,
    align_fixture_subset,
    assert_oracle_absent,
    issue_authorized_targets,
    mean_metrics,
    paired_bootstrap_difference,
    realized_target_inventory,
    token_metric_rows,
    validate_authorized_targets,
    validate_frozen_files,
    validate_pair_token_alignment,
    validate_restricted_receipt,
    validate_stage,
    freeze_files,
)


TRAIN_WORKER = REPO_ROOT / "experiments/geometria_proporcional/_wave50_train_worker.py"
INFER_WORKER = REPO_ROOT / "experiments/geometria_proporcional/_wave50_infer_worker.py"
CLASSICAL_WORKER = REPO_ROOT / "experiments/geometria_proporcional/_wave49_executor_worker.py"
PUBLIC_KEY = REPO_ROOT / "experiments/geometria_proporcional/keys/wave49_attestation_public.pem"
DEFAULT_PRIVATE_KEY = Path.home() / ".config/phideus/wave49_attestation_private.pem"
PRIMARY_OUTPUT_NAME = "wave50_prospective_v1"
REPLAY_OUTPUT_NAME = "wave50_prospective_v1_replay"
DEFAULT_RECOVERY_AUTHORIZATION = (
    REPO_ROOT / "experiments/geometria_proporcional/configs/wave50_recovery_v1.json"
)
TRAIN_PACKAGE_SOURCES = (
    "__init__.py",
    "wave49_schema.py",
    "wave49_attestation.py",
    "wave50_model.py",
    "wave50_neural.py",
    "wave50_protocol.py",
)
INFER_PACKAGE_SOURCES = ("__init__.py", "wave49_schema.py", "wave50_model.py")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "experiments/geometria_proporcional/configs/wave50_prospective_v1.json",
    )
    parser.add_argument("--attestation-private-key", type=Path, default=DEFAULT_PRIVATE_KEY)
    parser.add_argument(
        "--replay-secrets-from",
        type=Path,
        help="reuse a prior prospective run's sealed generator keys for exact replay",
    )
    parser.add_argument(
        "--compare-to",
        type=Path,
        help="compare this completed replay exactly against a prior prospective run",
    )
    parser.add_argument(
        "--recovery-secrets-from",
        type=Path,
        help="reuse generator keys from a recorded technical failure without changing the lockbox",
    )
    parser.add_argument(
        "--recovery-authorization",
        type=Path,
        default=DEFAULT_RECOVERY_AUTHORIZATION,
        help="versioned identity and allowed-code-delta attestation for technical recovery",
    )
    return parser.parse_args()


def _git(*args: str) -> str:
    completed = subprocess.run(["git", *args], cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or f"git {' '.join(args)} failed")
    return completed.stdout.strip()


def _execution_sources(config: Path, recovery_authorization: Path | None = None) -> list[Path]:
    paths = [
        Path(__file__).resolve(),
        TRAIN_WORKER.resolve(),
        INFER_WORKER.resolve(),
        CLASSICAL_WORKER.resolve(),
        PUBLIC_KEY.resolve(),
        *(REPO_ROOT / "src/geometria_proporcional" / name for name in TRAIN_PACKAGE_SOURCES),
        *(REPO_ROOT / "src/geometria_proporcional" / name for name in INFER_PACKAGE_SOURCES),
        *(REPO_ROOT / "src/geometria_proporcional").glob("wave49_*.py"),
        REPO_ROOT / "src/geometria_proporcional/wave50_mutations.py",
        config.resolve(),
    ]
    if recovery_authorization is not None:
        paths.append(recovery_authorization.resolve())
    paths = list({path.resolve(): None for path in paths})
    for path in paths:
        relative = path.relative_to(REPO_ROOT)
        if not _git("ls-files", "--error-unmatch", str(relative)):
            raise RuntimeError(f"execution source is not tracked: {relative}")
        if subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(relative)], cwd=REPO_ROOT).returncode:
            raise RuntimeError(f"execution source differs from HEAD: {relative}")
    return paths


def _snapshot_sources(output_dir: Path, sources: list[Path]) -> Path:
    snapshot_root = output_dir / "source_snapshots/preexecution"
    copied = []
    for source in sources:
        relative = source.relative_to(REPO_ROOT)
        destination = snapshot_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied.append(destination)
    manifest = output_dir / "source_snapshot_manifest.json"
    freeze_files(
        output_dir,
        manifest,
        "clean-commit-source-state-snapshotted-before-generation",
        copied,
        extra={
            "git_commit": _git("rev-parse", "HEAD"),
            "execution_semantics": {
                "root_modules": "loaded_from_hash-verified_clean_commit_before_snapshot_and_immutable_in_process",
                "restricted_workers": "executed_from_staged_snapshot_copies",
            },
        },
    )
    validate_frozen_files(
        output_dir, manifest, "clean-commit-source-state-snapshotted-before-generation"
    )
    return manifest


def _replay_keys(replay_root: Path | None) -> tuple[bytes | None, bytes | None, bytes | None]:
    if replay_root is None:
        return None, None, None
    benchmark = replay_root.resolve()
    if (benchmark / "benchmark").is_dir():
        benchmark = benchmark / "benchmark"
    sealed = benchmark / "sealed"
    names = (
        "generation_secret.json",
        "identity_secret.json",
        "semantic_commitment_secret.json",
    )
    keys = tuple(bytes.fromhex(json.loads((sealed / name).read_text(encoding="utf-8"))["key_hex"]) for name in names)
    if any(len(key) != 32 for key in keys) or len(set(keys)) != 3:
        raise RuntimeError("replay generator keys are invalid or not distinct")
    return keys


def _validate_recovery_source(
    recovery_root: Path,
    config_path: Path,
    authorization_path: Path,
) -> dict[str, object]:
    root = recovery_root.resolve()
    authorization_path = authorization_path.resolve(strict=True)
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    if authorization.get("status") != "TECHNICAL_RECOVERY_AUTHORIZED_AFTER_CODE_AUDIT":
        raise RuntimeError("recovery authorization is not audited and frozen")
    if root.name != authorization.get("failed_attempt_directory"):
        raise RuntimeError("recovery source is not the authorized failed attempt")
    failure_path = root / "technical_failure.json"
    if not failure_path.is_file():
        raise RuntimeError("recovery source lacks technical_failure.json")
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    if failure.get("status") != "TECHNICAL_FAILURE_BEFORE_CANONICAL_ADJUDICATION":
        raise RuntimeError("recovery source is not a recorded technical failure")
    if failure.get("frozen_predictions_completed") is not True:
        raise RuntimeError("recovery source did not freeze predictions before failure")
    if failure.get("architecture_or_protocol_changed") is not False:
        raise RuntimeError("recovery source does not attest an unchanged scientific protocol")
    if failure.get("source_commit") != authorization.get("source_commit"):
        raise RuntimeError("recovery source commit differs from authorization")
    if (root / "execution_manifest.json").exists() or (root / "run_status.json").exists():
        raise RuntimeError("completed prospective runs cannot be used as recovery sources")
    source_config = root / "prospective_config.json"
    if not source_config.is_file() or sha256_file(source_config) != sha256_file(config_path):
        raise RuntimeError("recovery source prospective config differs from current config")
    identity_paths = {
        "technical_failure_sha256": failure_path,
        "lockbox_visible_sha256": root / "benchmark/visible/lockbox.jsonl",
        "benchmark_manifest_sha256": root / "benchmark/manifest.json",
        "training_manifest_sha256": root / "training_manifest.json",
        "neural_prediction_manifest_sha256": root / "neural_prediction_manifest.json",
        "classical_prediction_manifest_sha256": root / "benchmark/prediction_manifest.json",
    }
    for field, path in identity_paths.items():
        if sha256_file(path) != authorization.get(field):
            raise RuntimeError(f"recovery source identity mismatch: {field}")
    source_manifest = json.loads(
        (root / "source_snapshot_manifest.json").read_text(encoding="utf-8")
    )
    if source_manifest.get("git_commit") != authorization.get("source_commit"):
        raise RuntimeError("recovery source snapshot commit differs from authorization")
    prefix = "source_snapshots/preexecution/"
    changed = {}
    for snapshot_relative, before_record in source_manifest["files"].items():
        if not snapshot_relative.startswith(prefix):
            raise RuntimeError("unexpected recovery source snapshot path")
        repo_relative = snapshot_relative.removeprefix(prefix)
        current_path = REPO_ROOT / repo_relative
        if not current_path.is_file():
            raise RuntimeError(f"recovery source file disappeared: {repo_relative}")
        after_sha = sha256_file(current_path)
        if before_record["sha256"] != after_sha:
            changed[repo_relative] = {
                "before_sha256": before_record["sha256"],
                "after_sha256": after_sha,
            }
    allowed = {
        name: {
            "before_sha256": row["before_sha256"],
            "after_sha256": row["after_sha256"],
        }
        for name, row in authorization["allowed_source_deltas"].items()
    }
    if changed != allowed:
        raise RuntimeError("current source deltas differ from recovery authorization")
    validate_manifest(root / "benchmark")
    validate_prediction_manifest(root / "benchmark")
    validate_frozen_files(
        root, root / "training_manifest.json", "training-frozen-before-lockbox-mount"
    )
    validate_frozen_files(
        root,
        root / "neural_prediction_manifest.json",
        "lockbox-predictions-frozen-before-oracle",
    )
    validate_frozen_files(
        root,
        root / "source_snapshot_manifest.json",
        "clean-commit-source-state-snapshotted-before-generation",
    )
    _replay_keys(root)
    return {
        "status": "TECHNICAL_RECOVERY_REUSES_FROZEN_LOCKBOX_KEYS",
        "source_attempt": str(root),
        "source_failure_sha256": sha256_file(failure_path),
        "source_benchmark_manifest_sha256": sha256_file(root / "benchmark/manifest.json"),
        "prospective_config_sha256": sha256_file(config_path),
        "recovery_authorization_sha256": sha256_file(authorization_path),
        "allowed_source_deltas": authorization["allowed_source_deltas"],
        "architecture_or_protocol_changed": False,
        "scientific_go_nogo_decision": "USER_ONLY",
    }


def _validate_output_path(output_dir: Path) -> None:
    output = output_dir.resolve()
    allowed = (REPO_ROOT / "data/geometria_proporcional").resolve()
    if allowed not in output.parents or output == allowed:
        raise ValueError(f"output must be a child of {allowed}")
    if not output.name.startswith("wave50_prospective_"):
        raise ValueError("prospective output directory name must start with wave50_prospective_")


def _copy_worker_sources(
    stage: Path,
    worker: Path,
    names: tuple[str, ...],
    snapshot_root: Path,
) -> Path:
    package = stage / "source/geometria_proporcional"
    package.mkdir(parents=True)
    for name in names:
        shutil.copy2(snapshot_root / "src/geometria_proporcional" / name, package / name)
    target = stage / "worker.py"
    shutil.copy2(snapshot_root / worker.relative_to(REPO_ROOT), target)
    return target


def _make_readable(root: Path) -> None:
    for path in [root, *root.rglob("*")]:
        path.chmod(0o755 if path.is_dir() else 0o644)


def _run_restricted(stage: Path, output_stage: Path, worker: Path, probes: list[Path]) -> None:
    account = pwd.getpwnam("nobody")
    _make_readable(stage)
    output_stage.mkdir()
    # The worker UID lives inside bwrap's user namespace; the root-only temp
    # parent keeps this host path private while 0777 permits the mapped UID to write.
    output_stage.chmod(0o777)
    os.chown(output_stage, account.pw_uid, account.pw_gid)
    command = [
        "bwrap",
        "--die-with-parent",
        "--unshare-all",
        "--new-session",
        "--ro-bind", "/usr", "/usr",
        "--ro-bind", "/lib", "/lib",
        "--ro-bind", "/lib64", "/lib64",
        "--ro-bind", str((REPO_ROOT / "venv").resolve()), "/venv",
        "--ro-bind", str(stage), "/work",
        "--bind", str(output_stage), "/output",
        "--tmpfs", "/tmp",
        "--proc", "/proc",
        "--dev", "/dev",
        "--chdir", "/tmp",
        "--setenv", "PYTHONPATH", "/work/source",
        "--setenv", "HOME", "/tmp",
        "--setenv", "OMP_NUM_THREADS", "1",
        "--setenv", "OPENBLAS_NUM_THREADS", "1",
        "--setenv", "MKL_NUM_THREADS", "1",
        "--setenv", "PYTHONHASHSEED", "0",
        "--uid", str(account.pw_uid),
        "--gid", str(account.pw_gid),
        "/venv/bin/python", "/work/worker.py",
        "--stage", "/work", "--output", "/output",
    ]
    for probe in probes:
        command.extend(["--forbidden-probe", str(probe)])
    completed = subprocess.run(
        command,
        cwd="/tmp",
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"restricted worker failed ({completed.returncode}): {completed.stderr.strip()}")


def _run_training(
    output_dir: Path,
    benchmark: Path,
    labels: Path,
    config_path: Path,
    snapshot_root: Path,
) -> Path:
    with tempfile.TemporaryDirectory(prefix="wave50-training-", dir="/tmp") as raw:
        root = Path(raw)
        stage = root / "stage"
        stage.mkdir()
        for subdir in ("visible", "labels"):
            (stage / subdir).mkdir()
        for split in ("train", "val"):
            shutil.copy2(benchmark / "visible" / f"{split}.jsonl", stage / "visible" / f"{split}.jsonl")
            shutil.copy2(labels / f"{split}.jsonl", stage / "labels" / f"{split}.jsonl")
        shutil.copy2(
            snapshot_root / config_path.relative_to(REPO_ROOT),
            stage / "prospective_config.json",
        )
        worker = _copy_worker_sources(
            stage, TRAIN_WORKER, TRAIN_PACKAGE_SOURCES, snapshot_root
        )
        expected_inventory = validate_stage(stage, "training")
        worker_out = root / "worker-output"
        _run_restricted(
            stage,
            worker_out,
            worker,
            [benchmark / "visible/lockbox.jsonl", benchmark / "sealed/lockbox.jsonl"],
        )
        validate_restricted_receipt(
            worker_out / "access_receipt.json",
            "restricted-training-complete",
            worker_out,
            input_root=stage,
            expected_allowlist=expected_inventory,
            expected_command_prefix=[
                "/work/worker.py", "--stage", "/work", "--output", "/output",
            ],
        )
        destination = output_dir / "training"
        shutil.copytree(worker_out, destination)
    return destination


def _strip_checkpoints(training: Path, destination: Path) -> None:
    destination.mkdir(parents=True)
    for source in sorted((training / "checkpoints").glob("*.pt")):
        payload = torch.load(source, map_location="cpu", weights_only=False)
        torch.save({
            "model_state": payload["model_state"],
            "seed": payload["seed"],
            "arm": payload["arm"],
            "variant": payload["variant"],
        }, destination / source.name)


def _split_identity_inventory(benchmark: Path, config: dict) -> dict[str, object]:
    tokens = {}
    counts = {}
    for split in ("train", "val", "lockbox"):
        rows = read_jsonl(benchmark / "sealed" / f"{split}.jsonl")
        tokens[split] = {row["pair_token"] for row in rows}
        primary = [
            row for row in rows
            if not row["is_out_of_catalog"]
            and row["calibration_population"] == "canonical_preserving"
        ]
        primary_tokens = {}
        for row in primary:
            primary_tokens.setdefault(row["pair_token"], row["design_stratum"])
        by_stratum = Counter(primary_tokens.values())
        counts[split] = {
            "visible_fixtures": len(read_jsonl(benchmark / "visible" / f"{split}.jsonl")),
            "sealed_fixtures": len(rows),
            "pair_tokens": len(tokens[split]),
            "primary_fixtures": len(primary),
            "primary_pair_tokens": len(primary_tokens),
            "primary_pair_tokens_by_design_stratum": dict(sorted(by_stratum.items())),
        }
    overlaps = {
        f"{left}|{right}": len(tokens[left] & tokens[right])
        for index, left in enumerate(tokens)
        for right in list(tokens)[index + 1:]
    }
    if any(overlaps.values()):
        raise RuntimeError(f"pair_token overlap across generator splits: {overlaps}")
    expected_strata = config["benchmark"]["expected_primary_pair_tokens_by_design_stratum_per_split"]
    for split, row in counts.items():
        expected = {
            "visible_fixtures": config["benchmark"]["expected_visible_fixtures_per_split"],
            "sealed_fixtures": config["benchmark"]["expected_visible_fixtures_per_split"],
            "primary_fixtures": config["benchmark"]["expected_primary_fixtures_per_split"],
            "primary_pair_tokens": config["benchmark"]["expected_primary_pair_tokens_per_split"],
        }
        for field, value in expected.items():
            if row[field] != value:
                raise RuntimeError(f"preregistered {field} mismatch in {split}: {row[field]} != {value}")
        if row["primary_pair_tokens_by_design_stratum"] != expected_strata:
            raise RuntimeError(f"preregistered stratum counts mismatch in {split}")
    return {"counts": counts, "cross_split_pair_token_overlaps": overlaps}


def _run_inference(
    output_dir: Path,
    benchmark: Path,
    config_path: Path,
    snapshot_root: Path,
) -> Path:
    training = output_dir / "training"
    inference_only = output_dir / "frozen_inference_checkpoints"
    with tempfile.TemporaryDirectory(prefix="wave50-inference-", dir="/tmp") as raw:
        root = Path(raw)
        stage = root / "stage"
        stage.mkdir()
        (stage / "visible").mkdir()
        (stage / "frozen/checkpoints").mkdir(parents=True)
        shutil.copy2(benchmark / "visible/lockbox.jsonl", stage / "visible/lockbox.jsonl")
        shutil.copy2(
            snapshot_root / config_path.relative_to(REPO_ROOT),
            stage / "prospective_config.json",
        )
        shutil.copy2(training / "normalizer.npz", stage / "frozen/normalizer.npz")
        shutil.copy2(training / "thresholds.json", stage / "frozen/thresholds.json")
        for checkpoint in inference_only.glob("*.pt"):
            shutil.copy2(checkpoint, stage / "frozen/checkpoints" / checkpoint.name)
        worker = _copy_worker_sources(
            stage, INFER_WORKER, INFER_PACKAGE_SOURCES, snapshot_root
        )
        expected_inventory = validate_stage(stage, "inference")
        worker_out = root / "worker-output"
        _run_restricted(
            stage,
            worker_out,
            worker,
            [benchmark / "sealed/lockbox.jsonl", output_dir / "authorized_labels/train.jsonl"],
        )
        validate_restricted_receipt(
            worker_out / "access_receipt.json",
            "restricted-single-lockbox-inference-complete",
            worker_out,
            input_root=stage,
            expected_allowlist=expected_inventory,
            expected_command_prefix=[
                "/work/worker.py", "--stage", "/work", "--output", "/output",
            ],
        )
        destination = output_dir / "inference"
        shutil.copytree(worker_out, destination)
    return destination


def _load_ensemble_logits(inference: Path, arm: str, variant: str, seeds: list[int]) -> tuple[np.ndarray, np.ndarray]:
    arrays = []
    fixture_ids = None
    for seed in seeds:
        path = inference / "logits" / f"seed{seed}__{arm}__{variant}__lockbox.npz"
        with np.load(path) as data:
            current = data["fixture_id"].copy()
            if fixture_ids is None:
                fixture_ids = current
            elif not np.array_equal(fixture_ids, current):
                raise RuntimeError("lockbox fixture order differs between seeds")
            arrays.append(data["logits"].astype(np.float64))
    return fixture_ids, np.mean(np.stack(arrays), axis=0)


def _load_seed_logits(
    inference: Path,
    arm: str,
    variant: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    with np.load(
        inference / "logits" / f"seed{seed}__{arm}__{variant}__lockbox.npz"
    ) as data:
        return data["fixture_id"].copy(), data["logits"].astype(np.float64)


def _subset_examples_and_logits(
    examples: list[dict],
    logits: np.ndarray,
    stratum: str | None,
) -> tuple[list[dict], np.ndarray]:
    indices = [
        index for index, example in enumerate(examples)
        if stratum is None or example["design_stratum"] == stratum
    ]
    return [examples[index] for index in indices], logits[indices]


def _secondary_slice_value(example: dict, dimension: str) -> object:
    if dimension == "target_cardinality":
        return len(example["target_families"])
    if dimension == "family":
        return example["family_id"]
    return example[dimension]


def _evaluate(output_dir: Path, benchmark: Path, config: dict) -> dict:
    records, _ = load_labeled_records(
        benchmark / "visible/lockbox.jsonl",
        benchmark / "sealed/oracle/lockbox.jsonl",
        benchmark / "protocol_config.json",
        "lockbox",
    )
    normalizer_data = np.load(output_dir / "training/normalizer.npz")
    normalizer = FeatureNormalizer(normalizer_data["mean"], normalizer_data["std"])
    examples = prepare_examples(records, normalizer)
    thresholds = json.loads((output_dir / "training/thresholds.json").read_text(encoding="utf-8"))
    seeds = config["training"]["seeds"]
    visible_fixture_ids = [
        str(row["fixture_id"])
        for row in read_jsonl(benchmark / "visible/lockbox.jsonl")
    ]
    expected_ids = [str(example["fixture_id"]) for example in examples]
    results = {}
    metric_rows = {}
    per_seed_sensitivity = {}
    for arm in ARMS:
        tau = thresholds[arm]["selected"]["tau"]
        for variant in CHECKPOINT_VARIANTS:
            fixture_ids, logits = _load_ensemble_logits(output_dir / "inference", arm, variant, seeds)
            validate_pair_token_alignment(
                visible_fixture_ids,
                [str(value) for value in fixture_ids],
            )
            aligned = align_fixture_subset(expected_ids, fixture_ids, logits)
            key = f"{arm}|{variant}"
            results[key] = {}
            metric_rows[key] = {}
            for label, stratum in (("ALL_ELIGIBLE", None), ("NEAR_RIVAL", "NEAR_RIVAL"), ("FAR_RIVAL", "FAR_RIVAL")):
                subset_examples, subset_logits = _subset_examples_and_logits(examples, aligned, stratum)
                rows = token_metric_rows(subset_examples, subset_logits, arm, tau)
                metric_rows[key][label] = rows
                results[key][label] = {
                    "n_pair_tokens": len(rows),
                    "metrics": mean_metrics(rows),
                }
            nuisance = {}
            for dimension in config["controls"]["nuisance_slice_probes"]["dimensions"]:
                nuisance[dimension] = {}
                values = sorted({str(_secondary_slice_value(example, dimension)) for example in examples})
                for value in values:
                    indices = [
                        i for i, example in enumerate(examples)
                        if str(_secondary_slice_value(example, dimension)) == value
                    ]
                    rows = token_metric_rows([examples[i] for i in indices], aligned[indices], arm, tau)
                    nuisance[dimension][value] = {"n_pair_tokens": len(rows), "metrics": mean_metrics(rows)}
            results[key]["nuisance_slices"] = nuisance
            secondary = {}
            for dimension in ("target_cardinality", "family"):
                secondary[dimension] = {}
                values = sorted({str(_secondary_slice_value(example, dimension)) for example in examples})
                for value in values:
                    indices = [
                        i for i, example in enumerate(examples)
                        if str(_secondary_slice_value(example, dimension)) == value
                    ]
                    rows = token_metric_rows(
                        [examples[i] for i in indices], aligned[indices], arm, tau
                    )
                    secondary[dimension][value] = {
                        "n_pair_tokens": len(rows),
                        "metrics": mean_metrics(rows),
                    }
            results[key]["secondary_slices"] = secondary

    for arm in ARMS:
        tau = thresholds[arm]["selected"]["tau"]
        per_seed_sensitivity[arm] = {}
        for seed in seeds:
            fixture_ids, logits = _load_seed_logits(
                output_dir / "inference", arm, "main", int(seed)
            )
            validate_pair_token_alignment(
                visible_fixture_ids,
                [str(value) for value in fixture_ids],
            )
            logits = align_fixture_subset(expected_ids, fixture_ids, logits)
            seed_rows = {}
            for label, stratum in (
                ("ALL_ELIGIBLE", None),
                ("NEAR_RIVAL", "NEAR_RIVAL"),
                ("FAR_RIVAL", "FAR_RIVAL"),
            ):
                subset_examples, subset_logits = _subset_examples_and_logits(
                    examples, logits, stratum
                )
                rows = token_metric_rows(subset_examples, subset_logits, arm, tau)
                seed_rows[label] = {
                    "n_pair_tokens": len(rows),
                    "metrics": mean_metrics(rows),
                }
            per_seed_sensitivity[arm][str(seed)] = seed_rows

    all_lockbox_labels = read_jsonl(benchmark / "sealed/oracle/lockbox.jsonl")
    diagnostic_population_inventory = {}
    for population_name, predicate in {
        "primary_canonical_in_catalog": lambda row: (
            not row["is_out_of_catalog"]
            and row["calibration_population"] == "canonical_preserving"
        ),
        "translation_in_catalog": lambda row: (
            not row["is_out_of_catalog"]
            and row["calibration_population"] != "canonical_preserving"
        ),
        "out_of_catalog": lambda row: bool(row["is_out_of_catalog"]),
    }.items():
        selected = [row for row in all_lockbox_labels if predicate(row)]
        diagnostic_population_inventory[population_name] = {
            "n_fixtures": len(selected),
            "n_pair_tokens": len({row["pair_token"] for row in selected}),
            "role": (
                "confirmatory_eligible"
                if population_name == "primary_canonical_in_catalog"
                else "inventory_only_no_competence_claim"
            ),
        }

    uncertainty = config["uncertainty"]
    left = metric_rows["sigmoid_set|main"]["NEAR_RIVAL"]
    right = metric_rows["softmax_partial|main"]["NEAR_RIVAL"]
    contrasts = {
        metric: paired_bootstrap_difference(
            left,
            right,
            metric,
            uncertainty["replicates"],
            uncertainty["seed"] + offset,
            uncertainty["confirmatory_interval_level_each"],
        )
        for offset, metric in enumerate(("set_recall", "top1_compatible"))
    }
    criteria = config["confirmatory_criteria_for_interpretation"]
    pattern = {
        "set_recall_superiority": contrasts["set_recall"]["ci_lo"] > criteria["set_recall_material_difference"],
        "top1_noninferiority": contrasts["top1_compatible"]["ci_lo"] > criteria["top1_compatible_noninferiority_margin"],
    }
    pattern["joint_evidential_pattern"] = all(pattern.values())
    lockbox_constraint_transfer = {}
    for arm in ARMS:
        metrics = results[f"{arm}|main"]["ALL_ELIGIBLE"]["metrics"]
        target_cardinality = float(np.mean([
            len({family for family, selected in zip(
                ("PROP", "AFFINE_OFFSET", "POWER_NONUNIT", "SATURATING"),
                example["target"], strict=True,
            ) if selected})
            for example in examples
        ]))
        lockbox_constraint_transfer[arm] = {
            "mean_width_minus_mean_target_cardinality": metrics["width"] - target_cardinality,
            "any_incompatible_rate": metrics["any_incompatible"],
        }
    classical_predictions = {
        row["fixture_id"]: row
        for row in read_jsonl(benchmark / "predictions/lockbox.jsonl")
        if row["selector"] == "catalog_eiv"
    }
    classical_fixture_rows = []
    for example in examples:
        predicted = set(classical_predictions[example["fixture_id"]]["structural_compatible_set"])
        target = set(example["target_families"])
        incompatible = len(predicted - target)
        classical_fixture_rows.append({
            "pair_token": example["pair_token"],
            "design_stratum": example["design_stratum"],
            "set_recall": len(predicted & target) / len(target),
            "complete_coverage": float(target <= predicted),
            "width": float(len(predicted)),
            "any_incompatible": float(incompatible > 0),
            "incompatible_fraction": incompatible / max(len(predicted), 1),
        })
    classical_reference = {}
    for label, stratum in (("ALL_ELIGIBLE", None), ("NEAR_RIVAL", "NEAR_RIVAL"), ("FAR_RIVAL", "FAR_RIVAL")):
        selected = [row for row in classical_fixture_rows if stratum is None or row["design_stratum"] == stratum]
        grouped: dict[str, list[dict]] = {}
        for row in selected:
            grouped.setdefault(row["pair_token"], []).append(row)
        token_rows = [
            {
                name: float(np.mean([row[name] for row in rows]))
                for name in ("set_recall", "complete_coverage", "width", "any_incompatible", "incompatible_fraction")
            }
            for rows in grouped.values()
        ]
        classical_reference[label] = {
            "n_pair_tokens": len(token_rows),
            "metrics": {
                name: float(np.mean([row[name] for row in token_rows]))
                for name in token_rows[0]
            },
        }
    return {
        "scope": "same-generator prospective internal adjudication",
        "decision_authority": "user",
        "automatic_go": False,
        "results": results,
        "confirmatory_contrasts_NEAR_RIVAL": contrasts,
        "predeclared_pattern": pattern,
        "lockbox_constraint_transfer_report_only": lockbox_constraint_transfer,
        "catalog_eiv_external_reference": classical_reference,
        "per_seed_sensitivity": per_seed_sensitivity,
        "diagnostic_population_inventory": diagnostic_population_inventory,
    }


def _write_report(output_dir: Path, summary: dict) -> None:
    contrasts = summary["confirmatory_contrasts_NEAR_RIVAL"]
    pattern = summary["predeclared_pattern"]
    lines = [
        "# Wave 50 — prospective matched neural result",
        "",
        "> Same-generator prospective internal adjudication. The user retains architecture promotion and scientific GO/NO-GO.",
        "",
        "## Confirmatory contrasts (NEAR_RIVAL; sigmoid minus softmax)",
        "",
        "| Metric | Difference | 97.5% CI | Pair tokens |",
        "|---|---:|---:|---:|",
    ]
    for metric in ("set_recall", "top1_compatible"):
        row = contrasts[metric]
        lines.append(
            f"| {metric} | {row['left_minus_right']:+.4f} | [{row['ci_lo']:+.4f}, {row['ci_hi']:+.4f}] | {row['n_pair_tokens']} |"
        )
    lines.extend([
        "",
        "## Predeclared evidential pattern",
        "",
        f"- Set-recall superiority condition: `{pattern['set_recall_superiority']}`",
        f"- Top-1 non-inferiority condition: `{pattern['top1_noninferiority']}`",
        f"- Joint pattern: `{pattern['joint_evidential_pattern']}`",
        "",
        "This pattern is evidence for the narrow output/loss contrast only. It is not an automatic GO and does not validate a full proportional architecture.",
        "",
    ])
    lines.extend([
        "## Secondary diagnostic slices",
        "",
        "These slices are descriptive and were not used to select the model, threshold, or confirmatory claim.",
        "",
        "| Arm | Dimension | Value | Pair tokens | Set recall | Top-1 compatible | Width |",
        "|---|---|---|---:|---:|---:|---:|",
    ])
    for arm in ARMS:
        slices = summary["results"][f"{arm}|main"]["secondary_slices"]
        for dimension in ("target_cardinality", "family"):
            for value, row in slices[dimension].items():
                metrics = row["metrics"]
                lines.append(
                    f"| {arm} | {dimension} | {value} | {row['n_pair_tokens']} | "
                    f"{metrics['set_recall']:.4f} | {metrics['top1_compatible']:.4f} | "
                    f"{metrics['width']:.4f} |"
                )
    lines.append("")
    lines.extend([
        "## Population scope",
        "",
        "| Population | Fixtures | Pair tokens | Role |",
        "|---|---:|---:|---|",
    ])
    for name, row in summary["diagnostic_population_inventory"].items():
        lines.append(
            f"| {name} | {row['n_fixtures']} | {row['n_pair_tokens']} | {row['role']} |"
        )
    lines.extend([
        "",
        "Per-seed sensitivity is preserved in `prospective_summary.json`; the confirmatory system remains the frozen raw-logit ensemble.",
        "",
    ])
    (output_dir / "REPORT_WAVE50_PROSPECTIVE.md").write_text("\n".join(lines), encoding="utf-8")


def _assert_exact_value(left: object, right: object, label: str) -> None:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        if left.dtype != right.dtype or left.shape != right.shape or not torch.equal(left, right):
            raise RuntimeError(f"exact replay tensor mismatch: {label}")
        return
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        if left.dtype != right.dtype or left.shape != right.shape or not np.array_equal(left, right):
            raise RuntimeError(f"exact replay array mismatch: {label}")
        return
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            raise RuntimeError(f"exact replay mapping keys mismatch: {label}")
        for key in sorted(left, key=str):
            _assert_exact_value(left[key], right[key], f"{label}.{key}")
        return
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if type(left) is not type(right) or len(left) != len(right):
            raise RuntimeError(f"exact replay sequence mismatch: {label}")
        for index, (lvalue, rvalue) in enumerate(zip(left, right, strict=True)):
            _assert_exact_value(lvalue, rvalue, f"{label}[{index}]")
        return
    if left != right:
        raise RuntimeError(f"exact replay value mismatch: {label}")


def _compare_named_files(reference: Path, replay: Path, relatives: list[Path]) -> int:
    for relative in relatives:
        left = reference / relative
        right = replay / relative
        if not left.is_file() or not right.is_file() or sha256_file(left) != sha256_file(right):
            raise RuntimeError(f"exact replay file mismatch: {relative}")
    return len(relatives)


def _compare_npz_tree(reference: Path, replay: Path, relative_dir: Path) -> int:
    left_root = reference / relative_dir
    right_root = replay / relative_dir
    left_files = sorted(path.relative_to(left_root) for path in left_root.rglob("*.npz"))
    right_files = sorted(path.relative_to(right_root) for path in right_root.rglob("*.npz"))
    if left_files != right_files:
        raise RuntimeError(f"exact replay npz inventory mismatch: {relative_dir}")
    for relative in left_files:
        with np.load(left_root / relative, allow_pickle=False) as left, np.load(
            right_root / relative, allow_pickle=False
        ) as right:
            if left.files != right.files:
                raise RuntimeError(f"exact replay npz keys mismatch: {relative_dir / relative}")
            for key in left.files:
                _assert_exact_value(left[key], right[key], f"{relative_dir / relative}:{key}")
    return len(left_files)


def _compare_npz_file(reference: Path, replay: Path, relative: Path) -> None:
    with np.load(reference / relative, allow_pickle=False) as left, np.load(
        replay / relative, allow_pickle=False
    ) as right:
        if left.files != right.files:
            raise RuntimeError(f"exact replay npz keys mismatch: {relative}")
        for key in left.files:
            _assert_exact_value(left[key], right[key], f"{relative}:{key}")


def _compare_checkpoint_tree(reference: Path, replay: Path, relative_dir: Path) -> int:
    left_root = reference / relative_dir
    right_root = replay / relative_dir
    left_files = sorted(path.relative_to(left_root) for path in left_root.rglob("*.pt"))
    right_files = sorted(path.relative_to(right_root) for path in right_root.rglob("*.pt"))
    if left_files != right_files:
        raise RuntimeError(f"exact replay checkpoint inventory mismatch: {relative_dir}")
    for relative in left_files:
        left = torch.load(left_root / relative, map_location="cpu", weights_only=False)
        right = torch.load(right_root / relative, map_location="cpu", weights_only=False)
        _assert_exact_value(left, right, str(relative_dir / relative))
    return len(left_files)


def _compare_receipt(
    reference: Path,
    replay: Path,
    relative: Path,
    ignored_keys: set[str],
) -> None:
    left = json.loads((reference / relative).read_text(encoding="utf-8"))
    right = json.loads((replay / relative).read_text(encoding="utf-8"))
    for key in ignored_keys:
        left.pop(key, None)
        right.pop(key, None)
    _assert_exact_value(left, right, str(relative))


def _compare_classical_prediction_manifest(reference: Path, replay: Path) -> None:
    relative = Path("benchmark/prediction_manifest.json")
    left = json.loads((reference / relative).read_text(encoding="utf-8"))
    right = json.loads((replay / relative).read_text(encoding="utf-8"))
    for payload in (left, right):
        payload["files"].pop("predictions/access_receipt.json", None)
    _assert_exact_value(left, right, str(relative))


def _compare_recovery_pre_oracle(reference: Path, recovered: Path) -> dict[str, object]:
    """Prove that a technical recovery regenerated the exposed experiment exactly."""
    reference = reference.resolve()
    recovered = recovered.resolve()
    recovery_provenance = json.loads(
        (recovered / "recovery_provenance.json").read_text(encoding="utf-8")
    )
    for root in (reference, recovered):
        validate_manifest(root / "benchmark")
        validate_prediction_manifest(root / "benchmark")
        validate_frozen_files(
            root, root / "training_manifest.json", "training-frozen-before-lockbox-mount"
        )
        validate_frozen_files(
            root,
            root / "neural_prediction_manifest.json",
            "lockbox-predictions-frozen-before-oracle",
        )

    benchmark_manifest = json.loads(
        (reference / "benchmark/manifest.json").read_text(encoding="utf-8")
    )
    exact_files = [
        Path("prospective_config.json"),
        Path("realized_target_inventory.json"),
        Path("training/thresholds.json"),
        Path("training/order_permutation.json"),
        Path("training/shuffle_manifest.json"),
        Path("training/validation_split_manifest.json"),
        Path("training/training_summary.json"),
        Path("mutation_results.json"),
        *(
            Path("benchmark") / relative
            for relative in benchmark_manifest["files"]
        ),
        *(
            path.relative_to(reference)
            for path in sorted((reference / "authorized_labels").rglob("*"))
            if path.is_file()
        ),
        *(
            path.relative_to(reference)
            for path in sorted((reference / "benchmark/predictions").glob("*.jsonl"))
        ),
        Path("benchmark/predictions/abstention_calibration.json"),
    ]
    exact_files = sorted(set(exact_files))

    reference_prediction_manifest = json.loads(
        (reference / "benchmark/prediction_manifest.json").read_text(encoding="utf-8")
    )
    recovered_prediction_manifest = json.loads(
        (recovered / "benchmark/prediction_manifest.json").read_text(encoding="utf-8")
    )
    for payload in (reference_prediction_manifest, recovered_prediction_manifest):
        payload["files"].pop("predictions/access_receipt.json", None)
        payload.pop("sources", None)
        payload.pop("invocation", None)
    _assert_exact_value(
        reference_prediction_manifest,
        recovered_prediction_manifest,
        "recovery classical prediction manifest semantics",
    )

    reference_training = json.loads((reference / "training_manifest.json").read_text())
    recovered_training = json.loads((recovered / "training_manifest.json").read_text())
    excluded_training = (
        "source_snapshots/",
        "source_snapshot_manifest.json",
        "recovery_provenance.json",
        "training/access_receipt.json",
    )
    reference_training_files = {
        key: value for key, value in reference_training["files"].items()
        if not key.startswith(excluded_training[0]) and key not in excluded_training[1:]
    }
    recovered_training_files = {
        key: value for key, value in recovered_training["files"].items()
        if not key.startswith(excluded_training[0]) and key not in excluded_training[1:]
    }
    _assert_exact_value(
        reference_training_files,
        recovered_training_files,
        "recovery training manifest scientific inventory",
    )

    reference_neural = json.loads((reference / "neural_prediction_manifest.json").read_text())
    recovered_neural = json.loads((recovered / "neural_prediction_manifest.json").read_text())
    reference_neural_files = {
        key: value for key, value in reference_neural["files"].items()
        if key != "training_manifest.json"
    }
    recovered_neural_files = {
        key: value for key, value in recovered_neural["files"].items()
        if key not in {"training_manifest.json", "benchmark/visible/lockbox.jsonl"}
    }
    _assert_exact_value(
        reference_neural_files,
        recovered_neural_files,
        "recovery neural manifest scientific inventory",
    )
    _compare_npz_file(reference, recovered, Path("training/normalizer.npz"))
    for relative in (Path("training/access_receipt.json"), Path("inference/access_receipt.json")):
        left = json.loads((reference / relative).read_text(encoding="utf-8"))
        right = json.loads((recovered / relative).read_text(encoding="utf-8"))
        left.pop("command", None)
        right.pop("command", None)
        for repo_relative, delta in recovery_provenance["allowed_source_deltas"].items():
            if not repo_relative.startswith("src/"):
                continue
            staged_relative = f"source/{repo_relative.removeprefix('src/')}"
            if staged_relative not in left.get("input_hashes", {}):
                continue
            if left["input_hashes"][staged_relative] != delta["before_sha256"]:
                raise RuntimeError(f"recovery receipt before-hash mismatch: {staged_relative}")
            if right["input_hashes"].get(staged_relative) != delta["after_sha256"]:
                raise RuntimeError(f"recovery receipt after-hash mismatch: {staged_relative}")
            left["input_hashes"].pop(staged_relative)
            right["input_hashes"].pop(staged_relative)
        _assert_exact_value(left, right, f"recovery {relative}")
    _compare_receipt(
        reference,
        recovered,
        Path("benchmark/predictions/access_receipt.json"),
        {"timestamp_utc"},
    )
    return {
        "status": "RECOVERY_PREORACLE_EXACT_EQUIVALENCE_PASS",
        "files_byte_exact": _compare_named_files(reference, recovered, exact_files),
        "npz_files_array_exact": 1 + sum(
            _compare_npz_tree(reference, recovered, Path(relative))
            for relative in ("training/logits", "inference/logits")
        ),
        "checkpoints_semantically_exact": sum(
            _compare_checkpoint_tree(reference, recovered, Path(relative))
            for relative in ("training/checkpoints", "frozen_inference_checkpoints")
        ),
        "runtime_receipts_semantically_exact": 3,
        "comparison_completed_before_oracle_open": True,
        "manifest_semantics_compared": ["classical_prediction", "training", "neural"],
    }


def _compare_exact_replay(reference: Path, replay: Path) -> dict[str, object]:
    reference = reference.resolve()
    if not (reference / "execution_manifest.json").is_file():
        raise RuntimeError("replay reference is incomplete: execution_manifest.json missing")
    validate_frozen_files(
        reference,
        reference / "execution_manifest.json",
        "oracle-opened-after-neural-and-classical-prediction-freeze",
    )
    for root in (reference, replay):
        validate_frozen_files(
            root, root / "training_manifest.json", "training-frozen-before-lockbox-mount"
        )
        validate_frozen_files(
            root,
            root / "neural_prediction_manifest.json",
            "lockbox-predictions-frozen-before-oracle",
        )
    exact_files = [
        Path("prospective_config.json"),
        Path("split_identity_inventory.json"),
        Path("realized_target_inventory.json"),
        Path("realized_target_inventory_complete.json"),
        Path("training/thresholds.json"),
        Path("training/order_permutation.json"),
        Path("training/shuffle_manifest.json"),
        Path("training/validation_split_manifest.json"),
        Path("training_manifest.json"),
        Path("neural_prediction_manifest.json"),
        Path("mutation_results.json"),
        Path("prospective_summary.json"),
        Path("REPORT_WAVE50_PROSPECTIVE.md"),
        Path("source_snapshot_manifest.json"),
    ]
    if (reference / "recovery_provenance.json").is_file():
        exact_files.append(Path("recovery_provenance.json"))
        exact_files.append(Path("recovery_equivalence.json"))
    exact_files.extend(
        path.relative_to(reference)
        for root in ("benchmark", "authorized_labels", "source_snapshots")
        for path in sorted((reference / root).rglob("*"))
        if path.is_file()
        and path.name != "access_receipt.json"
        and path.relative_to(reference) != Path("benchmark/prediction_manifest.json")
    )
    exact_files = sorted(set(exact_files))
    _compare_npz_file(reference, replay, Path("training/normalizer.npz"))
    _compare_receipt(
        reference,
        replay,
        Path("benchmark/predictions/access_receipt.json"),
        {"timestamp_utc"},
    )
    _compare_classical_prediction_manifest(reference, replay)
    for relative in (
        Path("training/access_receipt.json"),
        Path("inference/access_receipt.json"),
    ):
        _compare_receipt(
            reference,
            replay,
            relative,
            {"command"},
        )
    result = {
        "status": "EXACT_REPLAY_PASS",
        "reference": str(reference),
        "runtime_receipts_semantically_exact": 3,
        "classical_prediction_manifest_semantically_exact": True,
        "runtime_only_receipt_fields_excluded": [
            "benchmark classical timestamp_utc",
            "restricted worker command paths",
        ],
        "stage_manifests_independently_validated": 4,
        "files_byte_exact": _compare_named_files(reference, replay, exact_files),
        "npz_files_array_exact": 1 + sum(
            _compare_npz_tree(reference, replay, Path(relative))
            for relative in ("training/logits", "inference/logits")
        ),
        "checkpoints_semantically_exact": sum(
            _compare_checkpoint_tree(reference, replay, Path(relative))
            for relative in ("training/checkpoints", "frozen_inference_checkpoints")
        ),
    }
    return result


def main() -> None:
    args = _parse_args()
    if os.geteuid() != 0:
        raise RuntimeError("prospective orchestrator requires root for OS-separated identities")
    output_dir = args.output_dir.resolve()
    config_path = args.config.resolve()
    _validate_output_path(output_dir)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["status"] != "PREREGISTRATION_AUDITED_AND_FROZEN_BEFORE_GENERATION":
        raise RuntimeError("prospective config is not audited and frozen")
    if bool(args.compare_to) != bool(args.replay_secrets_from):
        raise ValueError("exact replay requires both --replay-secrets-from and --compare-to")
    if args.recovery_secrets_from and (args.compare_to or args.replay_secrets_from):
        raise ValueError("technical recovery cannot be combined with exact replay mode")
    expected_output_name = REPLAY_OUTPUT_NAME if args.compare_to else PRIMARY_OUTPUT_NAME
    if output_dir.name != expected_output_name:
        raise ValueError(
            f"prospective role requires canonical output directory {expected_output_name}"
        )
    if args.compare_to:
        reference = args.compare_to.resolve()
        replay_source = args.replay_secrets_from.resolve()
        replay_run_root = replay_source.parent if replay_source.name == "benchmark" else replay_source
        if reference != replay_run_root or reference.name != PRIMARY_OUTPUT_NAME:
            raise ValueError("replay secrets and comparison must reference the canonical primary run")
    recovery_authorization = (
        args.recovery_authorization.resolve()
        if args.recovery_secrets_from
        or (args.compare_to and (args.compare_to / "recovery_provenance.json").is_file())
        else None
    )
    sources = _execution_sources(config_path, recovery_authorization)
    recovery = (
        _validate_recovery_source(
            args.recovery_secrets_from,
            config_path,
            args.recovery_authorization,
        )
        if args.recovery_secrets_from
        else None
    )
    if output_dir.exists():
        raise RuntimeError("prospective output path already exists; it is immutable")
    output_dir.mkdir(parents=True)
    output_dir.chmod(0o700)
    shutil.copy2(config_path, output_dir / "prospective_config.json")
    recovery_context = recovery
    if recovery_context is not None:
        write_json(output_dir / "recovery_provenance.json", recovery_context)
    elif args.compare_to and (args.compare_to / "recovery_provenance.json").is_file():
        shutil.copy2(
            args.compare_to / "recovery_provenance.json",
            output_dir / "recovery_provenance.json",
        )
        recovery_context = json.loads(
            (output_dir / "recovery_provenance.json").read_text(encoding="utf-8")
        )
    source_snapshot_manifest = _snapshot_sources(output_dir, sources)
    snapshot_root = output_dir / "source_snapshots/preexecution"
    benchmark = output_dir / "benchmark"
    protocol = default_protocol_config(smoke=False)
    key_source = args.recovery_secrets_from or args.replay_secrets_from
    generation_key, identity_key, commitment_key = _replay_keys(key_source)
    generate_benchmark(
        benchmark,
        protocol,
        generation_key=generation_key,
        identity_key=identity_key,
        commitment_key=commitment_key,
        attestation_private_key_path=args.attestation_private_key,
        trusted_public_key_path=PUBLIC_KEY,
    )
    validate_manifest(benchmark)
    validate_visible_package(benchmark, protocol)
    assert_oracle_absent(benchmark)
    benchmark.chmod(0o700)

    labels = output_dir / "authorized_labels"
    issue_authorized_targets(
        benchmark, protocol, labels, args.attestation_private_key, PUBLIC_KEY, config_path
    )
    validate_authorized_targets(labels, PUBLIC_KEY, config_path)
    inventory = realized_target_inventory(labels)
    write_json(output_dir / "realized_target_inventory.json", inventory)
    validate_frozen_files(
        output_dir, source_snapshot_manifest, "clean-commit-source-state-snapshotted-before-generation"
    )
    _run_training(output_dir, benchmark, labels, config_path, snapshot_root)
    validate_restricted_receipt(
        output_dir / "training/access_receipt.json",
        "restricted-training-complete",
        output_dir / "training",
    )
    if (output_dir / "training/TRAINING_ABORT.json").exists():
        raise RuntimeError("CALIBRATION_INADMISSIBLE: lockbox was not mounted")

    inference_only = output_dir / "frozen_inference_checkpoints"
    _strip_checkpoints(output_dir / "training", inference_only)
    training_files = [path for path in (output_dir / "training").rglob("*") if path.is_file()]
    inference_checkpoint_files = [path for path in inference_only.rglob("*") if path.is_file()]
    authorized_label_files = [
        path for path in (output_dir / "authorized_labels").rglob("*") if path.is_file()
    ]
    source_snapshot_files = [
        path for path in (output_dir / "source_snapshots").rglob("*") if path.is_file()
    ]
    training_manifest = output_dir / "training_manifest.json"
    freeze_files(
        output_dir,
        training_manifest,
        "training-frozen-before-lockbox-mount",
        [
            *training_files,
            *inference_checkpoint_files,
            *authorized_label_files,
            output_dir / "prospective_config.json",
            output_dir / "realized_target_inventory.json",
            source_snapshot_manifest,
            *(
                [output_dir / "recovery_provenance.json"]
                if (output_dir / "recovery_provenance.json").is_file()
                else []
            ),
            *source_snapshot_files,
        ],
        extra={
            "git_commit": _git("rev-parse", "HEAD"),
            "config_sha256": sha256_file(config_path),
            "sources": {str(path.relative_to(REPO_ROOT)): sha256_file(path) for path in sources},
            "inference_inputs": {
                "normalizer.npz": {
                    "sha256": sha256_file(output_dir / "training/normalizer.npz"),
                    "bytes": (output_dir / "training/normalizer.npz").stat().st_size,
                },
                "thresholds.json": {
                    "sha256": sha256_file(output_dir / "training/thresholds.json"),
                    "bytes": (output_dir / "training/thresholds.json").stat().st_size,
                },
                "checkpoints": {
                    path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
                    for path in inference_checkpoint_files
                },
            },
        },
    )
    validate_frozen_files(output_dir, training_manifest, "training-frozen-before-lockbox-mount")
    assert_oracle_absent(benchmark)
    validate_frozen_files(
        output_dir, source_snapshot_manifest, "clean-commit-source-state-snapshotted-before-generation"
    )
    _run_inference(output_dir, benchmark, config_path, snapshot_root)
    validate_restricted_receipt(
        output_dir / "inference/access_receipt.json",
        "restricted-single-lockbox-inference-complete",
        output_dir / "inference",
    )

    inference_files = [path for path in (output_dir / "inference").rglob("*") if path.is_file()]
    neural_prediction_manifest = output_dir / "neural_prediction_manifest.json"
    freeze_files(
        output_dir,
        neural_prediction_manifest,
        "lockbox-predictions-frozen-before-oracle",
        [
            *inference_files,
            *inference_checkpoint_files,
            benchmark / "visible/lockbox.jsonl",
            training_manifest,
            output_dir / "training/normalizer.npz",
            output_dir / "training/thresholds.json",
            output_dir / "prospective_config.json",
        ],
        extra={
            "lockbox_visible": {
                "sha256": sha256_file(benchmark / "visible/lockbox.jsonl"),
                "bytes": (benchmark / "visible/lockbox.jsonl").stat().st_size,
            },
        },
    )
    validate_frozen_files(output_dir, neural_prediction_manifest, "lockbox-predictions-frozen-before-oracle")
    validate_frozen_files(output_dir, training_manifest, "training-frozen-before-lockbox-mount")

    mutations = run_mutation_suite(output_dir, benchmark, PUBLIC_KEY, config_path)
    if len(mutations) != len(config["inference"]["mutation_failures_required"]):
        raise RuntimeError("mutation suite count differs from preregistered inventory")

    assert_oracle_absent(benchmark)
    validate_frozen_files(
        output_dir, source_snapshot_manifest, "clean-commit-source-state-snapshotted-before-generation"
    )
    snapshot_classical_worker = snapshot_root / CLASSICAL_WORKER.relative_to(REPO_ROOT)
    with tempfile.TemporaryDirectory(prefix="wave50-classical-source-", dir="/tmp") as raw:
        classical_stage = Path(raw)
        shutil.copytree(snapshot_root / "src", classical_stage / "src")
        staged_worker = classical_stage / "worker.py"
        shutil.copy2(snapshot_classical_worker, staged_worker)
        _make_readable(classical_stage)
        classical_receipt = run_restricted_executor(
            benchmark, protocol, staged_worker, classical_stage
        )
    classical_sources = {
        path.name: path
        for path in (snapshot_root / "src/geometria_proporcional").glob("wave49_*.py")
    }
    classical_sources.update({
        "worker": snapshot_classical_worker,
        "runner": snapshot_root / Path(__file__).resolve().relative_to(REPO_ROOT),
    })
    freeze_classical_predictions(
        benchmark,
        {"command": "wave50 classical reference before oracle", "git_commit": _git("rev-parse", "HEAD")},
        classical_sources,
    )
    write_json(output_dir / "classical_reference_receipt.json", classical_receipt)

    validate_frozen_files(output_dir, neural_prediction_manifest, "lockbox-predictions-frozen-before-oracle")
    validate_frozen_files(output_dir, training_manifest, "training-frozen-before-lockbox-mount")
    if recovery_context is not None:
        write_json(
            output_dir / "recovery_equivalence.json",
            _compare_recovery_pre_oracle(
                Path(str(recovery_context["source_attempt"])), output_dir
            ),
        )
    validate_semantic_attestation(benchmark, PUBLIC_KEY)
    validate_sealed_alignment(benchmark, protocol)
    write_json(
        output_dir / "split_identity_inventory.json",
        _split_identity_inventory(benchmark, config),
    )
    compute_oracle_splits(benchmark, protocol, ("lockbox",))
    lockbox_oracle = read_jsonl(benchmark / "sealed/oracle/lockbox.jsonl")
    validate_oracle_rows(
        lockbox_oracle,
        protocol.oracle_compatibility_distance,
        protocol.oracle_ood_distance,
    )
    if {row["fixture_id"] for row in lockbox_oracle} != {
        row["fixture_id"] for row in read_jsonl(benchmark / "sealed/lockbox.jsonl")
    }:
        raise RuntimeError("lockbox oracle coverage differs from sealed truth")
    complete_inventory = dict(inventory)
    complete_inventory.update(
        realized_target_inventory(benchmark / "sealed/oracle", splits=("lockbox",))
    )
    write_json(output_dir / "realized_target_inventory_complete.json", complete_inventory)
    summary = _evaluate(output_dir, benchmark, config)
    write_json(output_dir / "prospective_summary.json", summary)
    _write_report(output_dir, summary)
    if args.compare_to:
        write_json(
            output_dir / "exact_replay_comparison.json",
            _compare_exact_replay(args.compare_to, output_dir),
        )
    run_status = (
        "PROSPECTIVE_COMPLETE_EXACT_REPLAY_PASS"
        if args.compare_to
        else "PRIMARY_RUN_COMPLETE_EXACT_REPLAY_PENDING"
    )
    write_json(output_dir / "run_status.json", {
        "status": run_status,
        "technical_recovery": recovery_context is not None,
        "scientific_go_nogo_decision": "USER_ONLY",
    })
    final_paths = [
        path for path in output_dir.rglob("*")
        if path.is_file() and path.name != "execution_manifest.json"
    ]
    freeze_files(
        output_dir,
        output_dir / "execution_manifest.json",
        "oracle-opened-after-neural-and-classical-prediction-freeze",
        final_paths,
        extra={"scientific_go_nogo_decision": "USER_ONLY"},
    )
    validate_frozen_files(
        output_dir,
        output_dir / "execution_manifest.json",
        "oracle-opened-after-neural-and-classical-prediction-freeze",
    )
    print(json.dumps({
        "status": run_status,
        "joint_evidential_pattern": summary["predeclared_pattern"]["joint_evidential_pattern"],
        "output_dir": str(output_dir),
    }, indent=2))


if __name__ == "__main__":
    main()
