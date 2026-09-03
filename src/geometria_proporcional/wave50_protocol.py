"""Prospective protocol helpers for the Wave 50 matched neural experiment."""

from __future__ import annotations

import json
import os
import platform
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from .wave49_attestation import file_record, sign_attestation, verify_attestation
from .wave49_schema import CATALOG_FAMILIES, ProtocolConfig, read_jsonl, sha256_file, write_json, write_jsonl


TRAIN_SPLITS = ("train", "val")
CHECKPOINT_VARIANTS = ("main", "no_eiv", "true_target_control", "shuffled_target")
ARMS = ("softmax_partial", "sigmoid_set")
TRAIN_STAGE_SOURCES = {
    "worker.py",
    "source/geometria_proporcional/__init__.py",
    "source/geometria_proporcional/wave49_schema.py",
    "source/geometria_proporcional/wave49_attestation.py",
    "source/geometria_proporcional/wave50_model.py",
    "source/geometria_proporcional/wave50_neural.py",
    "source/geometria_proporcional/wave50_protocol.py",
}
INFERENCE_STAGE_SOURCES = {
    "worker.py",
    "source/geometria_proporcional/__init__.py",
    "source/geometria_proporcional/wave49_schema.py",
    "source/geometria_proporcional/wave50_model.py",
}


def issue_authorized_targets(
    benchmark_dir: Path,
    protocol: ProtocolConfig,
    destination: Path,
    private_key: Path,
    public_key: Path,
    prospective_config: Path,
) -> dict[str, Any]:
    """Issue only train/val targets and sign their exact content."""
    from .wave49_oracle import compute_oracle_splits

    benchmark_dir = Path(benchmark_dir).resolve()
    destination = Path(destination).resolve()
    if (benchmark_dir / "sealed" / "oracle" / "lockbox.jsonl").exists():
        raise RuntimeError("lockbox oracle already exists before prospective training")
    counts = compute_oracle_splits(benchmark_dir, protocol, TRAIN_SPLITS, destination)
    files = {split: file_record(destination / f"{split}.jsonl") for split in TRAIN_SPLITS}
    payload = {
        "phase": "train-val-targets-issued-before-training",
        "schema": "oracle_compatible_set_v1",
        "authorized_splits": list(TRAIN_SPLITS),
        "forbidden_split": "lockbox",
        "oracle_compatibility_distance": protocol.oracle_compatibility_distance,
        "prospective_config": file_record(Path(prospective_config)),
        "files": files,
        "counts": counts,
    }
    receipt = sign_attestation(payload, private_key, public_key)
    write_json(destination / "target_attestation.json", receipt)
    return receipt


def validate_authorized_targets(
    destination: Path,
    public_key: Path,
    prospective_config: Path,
) -> dict[str, Any]:
    destination = Path(destination)
    if (destination / "lockbox.jsonl").exists():
        raise RuntimeError("authorized target package contains lockbox labels")
    receipt = json.loads((destination / "target_attestation.json").read_text(encoding="utf-8"))
    verify_attestation(receipt, public_key)
    payload = receipt["payload"]
    if payload.get("authorized_splits") != list(TRAIN_SPLITS):
        raise RuntimeError("target attestation split scope mismatch")
    if payload.get("prospective_config") != file_record(Path(prospective_config)):
        raise RuntimeError("target attestation prospective config mismatch")
    for split in TRAIN_SPLITS:
        path = destination / f"{split}.jsonl"
        if payload["files"].get(split) != file_record(path):
            raise RuntimeError(f"target attestation hash mismatch for {split}")
        rows = read_jsonl(path)
        if any(row.get("split") != split for row in rows):
            raise RuntimeError(f"target row split mismatch in {split}")
    return receipt


def stage_inventory(root: Path) -> list[str]:
    return sorted(str(path.relative_to(root)) for path in Path(root).rglob("*") if path.is_file())


def validate_stage(root: Path, phase: str) -> list[str]:
    """Enforce exact semantic boundaries before launching a restricted worker."""
    inventory = stage_inventory(root)
    lowered = [item.lower() for item in inventory]
    if phase == "training":
        required = {
            "visible/train.jsonl", "visible/val.jsonl", "labels/train.jsonl",
            "labels/val.jsonl", "prospective_config.json",
        }
        forbidden_terms = ("lockbox", "sealed", "oracle/", "historical")
        allowed = required | TRAIN_STAGE_SOURCES
    elif phase == "inference":
        required = {
            "visible/lockbox.jsonl", "prospective_config.json",
            "frozen/normalizer.npz", "frozen/thresholds.json",
        }
        forbidden_terms = ("labels", "sealed", "optimizer", "train.jsonl", "val.jsonl")
        allowed = required | INFERENCE_STAGE_SOURCES
    else:
        raise ValueError(f"unknown stage phase: {phase}")
    missing = required - set(inventory)
    if missing:
        raise RuntimeError(f"{phase} stage missing required files: {sorted(missing)}")
    offenders = [item for item, low in zip(inventory, lowered, strict=True) if any(term in low for term in forbidden_terms)]
    if phase == "inference":
        offenders = [item for item in offenders if item != "visible/lockbox.jsonl"]
    if offenders:
        raise RuntimeError(f"{phase} stage contains forbidden files: {offenders}")
    if phase == "inference":
        allowed.update(
            item for item in inventory
            if item.startswith("frozen/checkpoints/") and item.endswith(".pt")
        )
    unexpected = set(inventory) - allowed
    if unexpected:
        raise RuntimeError(f"{phase} stage contains files outside allowlist: {sorted(unexpected)}")
    return inventory


def file_manifest(root: Path, paths: Iterable[Path]) -> dict[str, dict[str, Any]]:
    root = Path(root).resolve()
    return {
        str(Path(path).resolve().relative_to(root)): file_record(Path(path))
        for path in sorted(Path(path).resolve() for path in paths)
    }


def freeze_files(
    root: Path,
    manifest_path: Path,
    phase: str,
    paths: Iterable[Path],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    root = Path(root).resolve()
    manifest_path = Path(manifest_path)
    payload = {
        "phase": phase,
        "files": file_manifest(root, paths),
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
        **(extra or {}),
    }
    write_json(manifest_path, payload)
    return payload


def validate_frozen_files(root: Path, manifest_path: Path, expected_phase: str) -> dict[str, Any]:
    root = Path(root).resolve()
    payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if payload.get("phase") != expected_phase:
        raise RuntimeError(f"freeze phase mismatch: expected {expected_phase}")
    for relative, expected in payload["files"].items():
        path = root / relative
        if not path.exists() or file_record(path) != expected:
            raise RuntimeError(f"frozen artifact mismatch: {relative}")
    return payload


def assert_oracle_absent(benchmark_dir: Path) -> None:
    oracle = Path(benchmark_dir) / "sealed/oracle"
    if oracle.exists() and any(oracle.iterdir()):
        raise RuntimeError("oracle exists before prediction freeze")


def validate_restricted_receipt(
    path: Path,
    expected_phase: str,
    output_root: Path | None = None,
    input_root: Path | None = None,
    expected_allowlist: Iterable[str] | None = None,
    expected_command_prefix: list[str] | None = None,
) -> dict[str, Any]:
    receipt = json.loads(Path(path).read_text(encoding="utf-8"))
    if receipt.get("phase") != expected_phase:
        raise RuntimeError("restricted receipt phase mismatch")
    if receipt.get("effective_uid") in (None, 0):
        raise RuntimeError("restricted receipt does not prove non-root execution")
    if receipt.get("forbidden_probe_denied") is not True:
        raise RuntimeError("restricted receipt does not prove forbidden probes were denied")
    required = {"command", "input_hashes", "output_hashes", "operations", "allowlist"}
    if not required <= set(receipt):
        raise RuntimeError(f"restricted receipt missing fields: {sorted(required - set(receipt))}")
    if expected_phase.startswith("restricted-single-lockbox"):
        if receipt.get("single_lockbox_pass") is not True:
            raise RuntimeError("inference receipt does not assert a single lockbox pass")
        if any(operation.get("fit") is not False for operation in receipt["operations"]):
            raise RuntimeError("fit operation present in frozen inference receipt")
        expected_operations = (
            "load_frozen_normalizer", "load_inference_only_checkpoints", "predict"
        )
    elif expected_phase == "restricted-training-complete":
        expected_operations = ("fit_normalizer", "fit_models", "select_threshold", "monitor")
    else:
        expected_operations = None
    if expected_operations is not None and tuple(
        operation.get("operation") for operation in receipt["operations"]
    ) != expected_operations:
        raise RuntimeError("restricted receipt operation sequence mismatch")
    if expected_allowlist is not None:
        expected_inventory = sorted(expected_allowlist)
        if receipt["allowlist"] != expected_inventory:
            raise RuntimeError("restricted receipt allowlist mismatch")
    if set(receipt["input_hashes"]) != set(receipt["allowlist"]):
        raise RuntimeError("restricted receipt input hashes do not cover its allowlist")
    if expected_command_prefix is not None:
        command = receipt["command"]
        if command[:len(expected_command_prefix)] != expected_command_prefix:
            raise RuntimeError("restricted receipt command mismatch")
    if input_root is not None:
        root = Path(input_root)
        for relative, expected_hash in receipt["input_hashes"].items():
            artifact = root / relative
            if not artifact.exists() or sha256_file(artifact) != expected_hash:
                raise RuntimeError(f"restricted input hash mismatch: {relative}")
    if output_root is not None:
        root = Path(output_root)
        for relative, expected_hash in receipt["output_hashes"].items():
            artifact = root / relative
            if not artifact.exists() or sha256_file(artifact) != expected_hash:
                raise RuntimeError(f"restricted output hash mismatch: {relative}")
    return receipt


def validate_pair_token_alignment(reference: Iterable[str], candidate: Iterable[str]) -> None:
    if list(reference) != list(candidate):
        raise RuntimeError("pair_token alignment changed")


def compatible_probabilities(logits: np.ndarray, arm: str) -> np.ndarray:
    tensor = torch.from_numpy(np.asarray(logits))
    if arm == "softmax_partial":
        return torch.softmax(tensor, dim=-1).numpy()
    if arm == "sigmoid_set":
        return torch.sigmoid(tensor).numpy()
    raise ValueError(f"unknown arm: {arm}")


def token_metric_rows(
    examples: list[dict[str, Any]],
    logits: np.ndarray,
    arm: str,
    tau: float,
) -> list[dict[str, Any]]:
    """Reduce correlated fixture views before returning one metric row per token."""
    probabilities = compatible_probabilities(logits, arm)
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, example in enumerate(examples):
        grouped[example["pair_token"]].append(index)
    rows = []
    for token in sorted(grouped):
        indices = grouped[token]
        target = examples[indices[0]]["target"].astype(bool)
        if not all(np.array_equal(examples[index]["target"], target) for index in indices):
            raise RuntimeError(f"target mismatch within pair_token {token}")
        per_view = []
        for index in indices:
            probability = probabilities[index]
            predicted = probability >= tau
            width = int(predicted.sum())
            intersection = int(np.logical_and(predicted, target).sum())
            incompatible = int(np.logical_and(predicted, ~target).sum())
            generating = examples[index]["family_id"]
            per_view.append({
                "set_recall": intersection / int(target.sum()),
                "complete_coverage": float(np.all(~target | predicted)),
                "generating_family_coverage": float(predicted[CATALOG_FAMILIES.index(generating)]),
                "exact_set": float(np.array_equal(predicted, target)),
                "incompatible_fraction": incompatible / max(width, 1),
                "any_incompatible": float(incompatible > 0),
                "width": float(width),
                "empty": float(width == 0),
                "top1_compatible": float(target[int(np.argmax(probability))]),
            })
        first = examples[indices[0]]
        names = tuple(per_view[0])
        rows.append({
            "pair_token": token,
            "design_stratum": first["design_stratum"],
            "cardinality": int(target.sum()),
            "family_id": first["family_id"],
            "n": int(first["n"]),
            "noise_mode": first["noise_mode"],
            "covariance_mode": first["covariance_mode"],
            "range_mode": first["range_mode"],
            **{name: float(np.mean([row[name] for row in per_view])) for name in names},
        })
    return rows


def mean_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    names = (
        "set_recall", "complete_coverage", "generating_family_coverage", "exact_set",
        "incompatible_fraction", "any_incompatible", "width", "empty", "top1_compatible",
    )
    return {name: float(np.mean([row[name] for row in rows])) for name in names}


def select_prospective_tau(
    examples: list[dict[str, Any]],
    logits: np.ndarray,
    arm: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    grid = config["validation"]["threshold_grid"]
    taus = np.round(np.arange(grid["start"], grid["stop"] + grid["step"] / 2, grid["step"]), 10)
    constraints = config["validation"]["threshold_constraints"]
    target_cardinality = float(np.mean([len(example["target_families"]) for example in examples]))
    candidates = []
    for tau in taus:
        rows = token_metric_rows(examples, logits, arm, float(tau))
        metrics = mean_metrics(rows)
        admissible = (
            metrics["width"] - target_cardinality
            <= constraints["mean_width_minus_mean_target_cardinality_max"] + 1e-12
            and metrics["any_incompatible"]
            <= constraints["fixture_any_incompatible_rate_max"] + 1e-12
        )
        candidates.append({"tau": float(tau), "admissible": bool(admissible), **metrics})
    eligible = [row for row in candidates if row["admissible"]]
    if not eligible:
        return {
            "status": "CALIBRATION_INADMISSIBLE",
            "arm": arm,
            "target_cardinality": target_cardinality,
            "candidates": candidates,
        }
    selected = max(
        eligible,
        key=lambda row: (
            row["set_recall"], row["complete_coverage"], -row["width"], row["tau"]
        ),
    )
    return {
        "status": "ADMISSIBLE",
        "arm": arm,
        "target_cardinality": target_cardinality,
        "selected": selected,
        "candidates": candidates,
    }


def paired_bootstrap_difference(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
    metric: str,
    replicates: int,
    seed: int,
    interval_level: float,
) -> dict[str, Any]:
    by_left = {row["pair_token"]: row for row in left}
    by_right = {row["pair_token"]: row for row in right}
    tokens = sorted(set(by_left) & set(by_right))
    if set(by_left) != set(by_right) or not tokens:
        raise RuntimeError("paired bootstrap token alignment mismatch")
    delta = np.asarray([by_left[token][metric] - by_right[token][metric] for token in tokens])
    rng = np.random.default_rng(seed)
    sampled = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        sampled[index] = delta[rng.integers(0, len(delta), len(delta))].mean()
    tail = (1.0 - interval_level) / 2.0
    lo, hi = np.quantile(sampled, [tail, 1.0 - tail])
    return {
        "metric": metric,
        "left_minus_right": float(delta.mean()),
        "interval_level": interval_level,
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "n_pair_tokens": len(tokens),
        "replicates": replicates,
    }


def realized_target_inventory(
    label_dir: Path,
    splits: tuple[str, ...] = TRAIN_SPLITS,
) -> dict[str, Any]:
    inventory = {}
    all_tokens: dict[str, set[str]] = {}
    for split in splits:
        rows = [
            row for row in read_jsonl(Path(label_dir) / f"{split}.jsonl")
            if not row["is_out_of_catalog"] and row["calibration_population"] == "canonical_preserving"
        ]
        by_token: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_token[row["pair_token"]].append(row)
        all_tokens[split] = set(by_token)
        counts = Counter(
            (group[0]["design_stratum"], len(group[0]["oracle_compatible_set"]))
            for group in by_token.values()
        )
        inventory[split] = {
            "pair_tokens": len(by_token),
            "by_design_stratum_and_cardinality": {
                f"{stratum}|{cardinality}": count
                for (stratum, cardinality), count in sorted(counts.items())
            },
        }
    for index, left in enumerate(splits):
        for right in splits[index + 1:]:
            if all_tokens[left] & all_tokens[right]:
                raise RuntimeError(f"pair_token overlap between {left} and {right}")
    return inventory


def restricted_receipt(phase: str, inventory: list[str], operations: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "phase": phase,
        "effective_uid": os.geteuid(),
        "effective_gid": os.getegid(),
        "allowlist": inventory,
        "operations": operations,
    }
