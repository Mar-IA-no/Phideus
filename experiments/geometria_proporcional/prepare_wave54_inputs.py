#!/usr/bin/env python3
"""Prepare physically separated fit/select and sealed-monitor bundles for Wave 54."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
PLAN_PATH = (
    REPO_ROOT
    / "Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_54_JOINT_SET_POSTERIOR_PLAN.md"
)
PRIMITIVES_PATH = REPO_ROOT / "src/geometria_proporcional/wave54_joint_set.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wave50-dir", type=Path, required=True)
    parser.add_argument("--wave52-dir", type=Path, required=True)
    parser.add_argument("--wave53-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT
        / "experiments/geometria_proporcional/configs/wave54_joint_set.json",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_hash(path: Path, expected: str) -> dict[str, str]:
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"hash mismatch for {path}: expected {expected}, got {actual}")
    return {"path": str(path), "sha256": actual}


def file_receipt(path: Path) -> dict[str, str]:
    path = path.resolve(strict=True)
    return {"path": str(path), "sha256": sha256_file(path)}


def paths_overlap(left: Path, right: Path) -> bool:
    left = left.resolve()
    right = right.resolve()
    return left == right or left in right.parents or right in left.parents


def reject_lockbox_paths(paths: list[Path]) -> None:
    for path in paths:
        resolved = path.resolve()
        if any("lockbox" in part.lower() for part in resolved.parts):
            raise ValueError(f"Wave 54 must not access a lockbox path: {resolved}")


def require_sources_at_head(paths: list[Path]) -> None:
    for path in paths:
        resolved = path.resolve(strict=True)
        try:
            relative = resolved.relative_to(REPO_ROOT)
        except ValueError as error:
            raise RuntimeError(f"execution source is outside the repository: {resolved}") from error
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", str(relative)],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        if tracked.returncode != 0:
            raise RuntimeError(f"execution source is not tracked at HEAD: {relative}")
        changed = subprocess.run(
            ["git", "status", "--porcelain", "--", str(relative)],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        if changed:
            raise RuntimeError(f"execution source differs from HEAD: {relative}")


def prepare_output_directory(output: Path, force: bool) -> Path | None:
    archived = None
    if output.exists():
        if not force:
            raise FileExistsError(f"output exists: {output}; use --force")
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        archived = output.with_name(f"{output.name}.superseded_{stamp}")
        if archived.exists():
            raise FileExistsError(f"archive target already exists: {archived}")
        output.rename(archived)
    output.mkdir(parents=True)
    return archived


def validate_token_separation(
    threshold: dict[str, np.ndarray], monitor: dict[str, np.ndarray]
) -> None:
    threshold_tokens = threshold["pair_token"].astype(str)
    monitor_tokens = monitor["pair_token"].astype(str)
    if len(set(threshold_tokens)) != len(threshold_tokens):
        raise RuntimeError("duplicate pair_token in fit/select source")
    if len(set(monitor_tokens)) != len(monitor_tokens):
        raise RuntimeError("duplicate pair_token in sealed monitor source")
    overlap = sorted(set(threshold_tokens) & set(monitor_tokens))
    if overlap:
        raise RuntimeError(f"fit/select and monitor overlap on {len(overlap)} pair_token values")


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def git_state() -> tuple[str, str]:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=REPO_ROOT,
        text=True,
    ).strip()
    return commit, dirty


def load_metadata(path: Path, allowed_tokens: set[str]) -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            token = str(row.get("pair_token", ""))
            if token not in allowed_tokens:
                continue
            if row.get("calibration_population") != "canonical_preserving":
                continue
            current = {
                "design_stratum": str(row["design_stratum"]),
                "cardinality": len(row["oracle_compatible_set"]),
            }
            if token in metadata and metadata[token] != current:
                raise RuntimeError(f"conflicting canonical metadata for {token}")
            metadata[token] = current
    missing = sorted(allowed_tokens - set(metadata))
    if missing:
        raise RuntimeError(f"missing canonical metadata for {len(missing)} tokens")
    return metadata


def load_split_ensemble(
    raw_dir: Path,
    split: str,
    seeds: list[int],
    expected_hashes: dict[str, str],
) -> tuple[dict[str, np.ndarray], list[dict[str, str]]]:
    arrays = []
    receipts = []
    for seed in seeds:
        path = raw_dir / f"seed{seed}__{split}.npz"
        receipts.append(require_hash(path, expected_hashes[str(seed)]))
        with np.load(path, allow_pickle=False) as data:
            arrays.append({key: data[key] for key in data.files})
    base = arrays[0]
    for other in arrays[1:]:
        for key in ("pair_token", "cluster_id", "target"):
            if not np.array_equal(base[key], other[key]):
                raise RuntimeError(f"{split}: {key} differs across seeds")
    per_seed_logits = np.stack([row["set_logits"].astype(np.float64) for row in arrays])
    if not np.all(np.isfinite(per_seed_logits)):
        raise RuntimeError(f"{split}: non-finite logits")
    return {
        "pair_token": base["pair_token"].astype(str),
        "cluster_id": base["cluster_id"].astype(str),
        "target": base["target"].astype(bool),
        "per_seed_logits": per_seed_logits,
        "ensemble_logits": per_seed_logits.mean(axis=0),
    }, receipts


def attach_metadata(bundle: dict[str, np.ndarray], metadata: dict[str, dict[str, Any]]) -> None:
    bundle["design_stratum"] = np.asarray(
        [metadata[str(token)]["design_stratum"] for token in bundle["pair_token"]]
    )
    bundle["cardinality"] = np.asarray(
        [metadata[str(token)]["cardinality"] for token in bundle["pair_token"]],
        dtype=np.int64,
    )
    if not np.array_equal(bundle["target"].sum(axis=1), bundle["cardinality"]):
        raise RuntimeError("target cardinality differs from canonical metadata")


def main() -> None:
    args = parse_args()
    commit, dirty = git_state()
    if dirty:
        raise RuntimeError("tracked worktree must be clean before preparing Wave 54 inputs")
    output = args.output_dir.resolve()
    sources = [
        args.wave50_dir.resolve(strict=True),
        args.wave52_dir.resolve(strict=True),
        args.wave53_dir.resolve(strict=True),
    ]
    config_path = args.config.resolve(strict=True)
    reject_lockbox_paths([*sources, config_path, output])
    execution_sources = [Path(__file__), PRIMITIVES_PATH, PLAN_PATH, config_path]
    require_sources_at_head(execution_sources)
    for source in sources:
        if paths_overlap(output, source):
            raise ValueError("output cannot overlap an input directory")
    if output == REPO_ROOT or output in REPO_ROOT.parents:
        raise ValueError("output cannot be the repository or one of its ancestors")
    if any(output == path.resolve() or output in path.resolve().parents for path in execution_sources):
        raise ValueError("output cannot contain an execution source")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    binding = config["source_binding"]
    wave50 = args.wave50_dir.resolve()
    wave52 = args.wave52_dir.resolve()
    wave53 = args.wave53_dir.resolve()
    inputs = [
        require_hash(
            wave50 / "authorized_labels/val.jsonl",
            binding["wave50_val_labels_sha256"],
        ),
        require_hash(
            wave52 / "policy_manifest.json", binding["wave52_policy_manifest_sha256"]
        ),
        require_hash(wave52 / "summary.json", binding["wave52_summary_sha256"]),
        require_hash(
            wave53 / "split_manifest.json", binding["wave53_split_manifest_sha256"]
        ),
        require_hash(
            wave53 / "platt_calibrator.json",
            binding["wave53_platt_calibrator_sha256"],
        ),
    ]
    raw_dir = wave52 / "raw_eval/frozen_set"
    threshold, threshold_receipts = load_split_ensemble(
        raw_dir, "val_threshold", config["seeds"], binding["wave52_eval_sha256"]["val_threshold"]
    )
    monitor, monitor_receipts = load_split_ensemble(
        raw_dir, "val_monitor", config["seeds"], binding["wave52_eval_sha256"]["val_monitor"]
    )
    inputs.extend(threshold_receipts + monitor_receipts)
    validate_token_separation(threshold, monitor)

    all_tokens = set(threshold["pair_token"].tolist()) | set(monitor["pair_token"].tolist())
    metadata = load_metadata(wave50 / "authorized_labels/val.jsonl", all_tokens)
    attach_metadata(threshold, metadata)
    attach_metadata(monitor, metadata)

    split_manifest = json.loads((wave53 / "split_manifest.json").read_text(encoding="utf-8"))
    calibration = set(split_manifest["calibration_fit"])
    decision = set(split_manifest["decision_select"])
    threshold_tokens = set(threshold["pair_token"].astype(str).tolist())
    if calibration & decision or calibration | decision != threshold_tokens:
        raise RuntimeError("Wave 53 threshold split is not a disjoint complete partition")
    threshold["split_role"] = np.asarray(
        ["calibration_fit" if str(token) in calibration else "decision_select" for token in threshold["pair_token"]]
    )

    archived_output = prepare_output_directory(output, args.force)
    fit_path = output / "fit_select_bundle.npz"
    monitor_path = output / "sealed_monitor_bundle.npz"
    np.savez_compressed(fit_path, **threshold)
    np.savez_compressed(monitor_path, **monitor)
    manifest = {
        "status": "PHYSICALLY_SEPARATED_INPUT_BUNDLE",
        "git_commit": commit,
        "config_sha256": sha256_file(config_path),
        "execution_sources": [
            file_receipt(path) for path in execution_sources
        ],
        "inputs": inputs,
        "outputs": {
            "fit_select_bundle.npz": sha256_file(fit_path),
            "sealed_monitor_bundle.npz": sha256_file(monitor_path),
        },
        "counts": {
            "calibration_fit": int(np.sum(threshold["split_role"] == "calibration_fit")),
            "decision_select": int(np.sum(threshold["split_role"] == "decision_select")),
            "val_monitor": len(monitor["pair_token"]),
        },
        "no_statistical_fit_or_evaluation": True,
        "superseded_output": str(archived_output) if archived_output else None,
        "lockbox_accessed": False,
    }
    write_json(output / "input_bundle_manifest.json", manifest)
    print(json.dumps(manifest["counts"], sort_keys=True))


if __name__ == "__main__":
    sys.exit(main())
