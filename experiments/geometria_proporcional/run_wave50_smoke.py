#!/usr/bin/env python3
"""Run the Wave 50 CPU-only development smoke on opened Wave 49 train/val data."""

from __future__ import annotations

import argparse
import copy
import json
import os
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from geometria_proporcional.wave49_schema import sha256_file, write_json  # noqa: E402
from geometria_proporcional.wave50_neural import (  # noqa: E402
    DeepSetClassifier,
    FAMILY_TO_INDEX,
    TARGET_COMPATIBILITY_DISTANCE,
    TARGET_SCHEMA_VERSION,
    balanced_target_derangement,
    fit_normalizer,
    load_smoke_records,
    logits_payload,
    parameter_count,
    predict_logits,
    prepare_examples,
    smoke_metrics,
    select_smoke_tau,
    split_tokens,
    stable_hash,
    stratified_token_subset,
    train_fixed_recipe,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "experiments/geometria_proporcional/configs/wave50_smoke.json",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _validate_paths(benchmark_dir: Path, output_dir: Path) -> None:
    benchmark_dir = benchmark_dir.resolve()
    output_dir = output_dir.resolve()
    repo = REPO_ROOT.resolve()
    allowed_output_root = (repo / "data/geometria_proporcional").resolve()
    if output_dir == repo or output_dir in repo.parents:
        raise ValueError("refusing output path at or above repository root")
    if (
        output_dir == benchmark_dir
        or output_dir in benchmark_dir.parents
        or benchmark_dir in output_dir.parents
    ):
        raise ValueError("output path must be disjoint from benchmark input")
    if allowed_output_root not in output_dir.parents:
        raise ValueError(f"output path must be a child of {allowed_output_root}")


def _execution_sources(config_path: Path) -> list[dict[str, str]]:
    paths = [
        Path(__file__).resolve(),
        (REPO_ROOT / "src/geometria_proporcional/__init__.py").resolve(),
        (REPO_ROOT / "src/geometria_proporcional/wave49_schema.py").resolve(),
        (REPO_ROOT / "src/geometria_proporcional/wave50_neural.py").resolve(),
        config_path.resolve(),
    ]
    records = []
    for path in paths:
        relative = path.relative_to(REPO_ROOT)
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", str(relative)],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        if tracked.returncode != 0:
            raise RuntimeError(f"execution source is not tracked: {relative}")
        clean = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(relative)], cwd=REPO_ROOT)
        if clean.returncode != 0:
            raise RuntimeError(f"execution source differs from HEAD: {relative}")
        records.append({"path": str(relative), "sha256": sha256_file(path)})
    return records


def _save_logits(path: Path, examples: list[dict], logits: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **logits_payload(examples, logits))


def _checkpoint_payload(
    model: DeepSetClassifier,
    optimizer_state: dict,
    config: dict,
    seed: int,
    arm: str,
    variant: str,
) -> dict:
    return {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer_state,
        "torch_rng_state": torch.get_rng_state(),
        "numpy_seed": seed,
        "seed": seed,
        "arm": arm,
        "variant": variant,
        "config": config,
    }


def _write_report(output_dir: Path, summary: dict) -> None:
    lines = [
        "# Wave 50 matched neural development smoke",
        "",
        "> `SMOKE_ONLY`: opened historical train/val data; lockbox was not loaded. "
        "This artifact cannot adjudicate an architecture or scientific GO/NO-GO.",
        "",
        f"- CPU wall time: `{summary['runtime']['wall_seconds']:.2f}s`",
        f"- Parameters per model: `{summary['parameter_count']}`",
        f"- Train fixtures/tokens: `{summary['data']['train_fixtures']}` / `{summary['data']['train_tokens']}`",
        f"- Monitor fixtures/tokens: `{summary['data']['monitor_fixtures']}` / `{summary['data']['monitor_tokens']}`",
        f"- Maximum order-permutation logit delta: `{summary['max_permutation_delta']:.3e}`",
        f"- Peak process RSS: `{summary['runtime']['max_rss_kib']} KiB`",
        "",
        "## Monitor metrics",
        "",
        "Each row uses an arm-specific `SMOKE_DIAGNOSTIC_ONLY` threshold selected on the disjoint "
        "`val_threshold` subset; the prospective calibration rule is not exercised here.",
        "",
        "| seed | arm | variant | tau diagnostic | first loss | last loss | membership AUC | top1 compatible | set recall | width |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["runs"]:
        metrics = row["monitor_metrics"]["overall"]
        lines.append(
            f"| {row['seed']} | {row['arm']} | {row['variant']} | "
            f"{row['diagnostic_tau_selection']['tau']:.2f} | "
            f"{row['history'][0]['mean_loss']:.4f} | {row['history'][-1]['mean_loss']:.4f} | "
            f"{metrics['membership_macro_auc']:.3f} | {metrics['top1_compatible']:.3f} | "
            f"{metrics['set_recall']:.3f} | {metrics['width']:.3f} |"
        )
    lines.extend([
        "",
        "## Scope",
        "",
        "The smoke checks implementation, learning signal, target-shuffle behavior, matched parameter "
        "counts, permutation invariance, and CPU cost. A fresh prospectively sealed package remains "
        "mandatory for any evidential comparison.",
        "",
    ])
    (output_dir / "REPORT_WAVE50_SMOKE.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = _parse_args()
    benchmark_dir = args.benchmark_dir.resolve()
    output_dir = args.output_dir.resolve()
    _validate_paths(benchmark_dir, output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.force:
            raise SystemExit(f"non-empty output exists: {output_dir}; pass --force to replace")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = json.loads(args.config.read_text(encoding="utf-8"))
    execution_sources = _execution_sources(args.config)
    if config.get("status") != "SMOKE_ONLY":
        raise ValueError("Wave 50 development config must declare status=SMOKE_ONLY")
    if torch.cuda.is_initialized():
        raise RuntimeError("development smoke must start before any CUDA context is initialized")
    torch.set_num_threads(int(config["cpu_threads"]))
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()

    train_all, train_access = load_smoke_records(benchmark_dir, "train")
    val_all, val_access = load_smoke_records(benchmark_dir, "val")
    train_records = stratified_token_subset(
        train_all, int(config["max_train_fixtures"]), int(config["subset_seed"])
    )
    val_records = stratified_token_subset(
        val_all, int(config["max_val_fixtures"]), int(config["subset_seed"]) + 1
    )
    threshold_records, monitor_records = split_tokens(
        val_records, float(config["val_threshold_fraction"]), int(config["split_seed"])
    )
    train_tokens = {record["pair_token"] for record in train_records}
    threshold_tokens = {record["pair_token"] for record in threshold_records}
    monitor_tokens = {record["pair_token"] for record in monitor_records}
    if train_tokens & threshold_tokens or train_tokens & monitor_tokens or threshold_tokens & monitor_tokens:
        raise RuntimeError("pair_token leakage across smoke partitions")

    normalizer = fit_normalizer(train_records)
    np.savez(output_dir / "normalizer.npz", mean=normalizer.mean, std=normalizer.std)
    train_examples = prepare_examples(train_records, normalizer)
    threshold_examples = prepare_examples(threshold_records, normalizer)
    monitor_examples = prepare_examples(monitor_records, normalizer)

    eligible_tokens, _, eligibility_report = balanced_target_derangement(
        train_all, int(config["shuffle_seed"])
    )
    eligible_records = [record for record in train_all if record["pair_token"] in eligible_tokens]
    control_records = stratified_token_subset(
        eligible_records,
        int(config["max_control_fixtures"]),
        int(config["subset_seed"]) + 2,
        include_target_hash=True,
    )
    control_tokens, shuffled_targets, shuffle_report = balanced_target_derangement(
        control_records, int(config["shuffle_seed"]) + 1
    )
    control_records = [
        record for record in control_records if record["pair_token"] in control_tokens
    ]
    shuffle_report["full_pool_eligibility"] = eligibility_report
    shuffle_report["n_control_tokens_after_subset_and_revalidation"] = len(control_tokens)
    true_control_examples = prepare_examples(control_records, normalizer)
    shuffled_examples = prepare_examples(control_records, normalizer, target_by_token=shuffled_targets)
    original_targets = {record["pair_token"]: record["target_families"] for record in control_records}
    write_json(output_dir / "shuffle_manifest.json", {
        "schema": "wave50-smoke-target-derangement-v1",
        "report": shuffle_report,
        "rows": [
            {
                "pair_token": token,
                "original": list(original_targets[token]),
                "replacement": [
                    family for family, selected in zip(
                        ("PROP", "AFFINE_OFFSET", "POWER_NONUNIT", "SATURATING"),
                        shuffled_targets[token],
                        strict=True,
                    ) if selected
                ],
            }
            for token in sorted(control_tokens)
        ],
    })
    write_json(output_dir / "target_manifest.json", {
        "target_schema_version": TARGET_SCHEMA_VERSION,
        "oracle_compatibility_distance": TARGET_COMPATIBILITY_DISTANCE,
        "scope": "opened_historical_train_val_in_catalog_canonical_preserving",
        "lockbox_labels_loaded": False,
        "family_to_index": FAMILY_TO_INDEX,
        "train_target_sha256": stable_hash([
            [record["fixture_id"], list(record["target_families"])] for record in train_records
        ]),
        "val_target_sha256": stable_hash([
            [record["fixture_id"], list(record["target_families"])] for record in val_records
        ]),
    })

    arms = ("softmax_partial", "sigmoid_set")
    variants = {
        "main": train_examples,
        "true_target_control": true_control_examples,
        "shuffled_target": shuffled_examples,
    }
    run_rows = []
    parameter_counts: set[int] = set()
    max_permutation_delta = 0.0
    for seed in config["seeds"]:
        torch.manual_seed(int(seed))
        initial = DeepSetClassifier()
        initial_state = copy.deepcopy(initial.state_dict())
        for arm in arms:
            for variant, examples in variants.items():
                model = DeepSetClassifier()
                model.load_state_dict(initial_state)
                parameter_counts.add(parameter_count(model))
                history, optimizer_state = train_fixed_recipe(
                    model=model,
                    examples=examples,
                    arm=arm,
                    seed=int(seed),
                    epochs=int(config["epochs"]),
                    batch_tokens=int(config["batch_tokens"]),
                    learning_rate=float(config["learning_rate"]),
                    weight_decay=float(config["weight_decay"]),
                )
                threshold_logits = predict_logits(
                    model, threshold_examples, int(config["inference_batch_size"])
                )
                diagnostic_tau, tau_selection = select_smoke_tau(
                    threshold_examples,
                    threshold_logits,
                    arm,
                    config["diagnostic_tau_grid"],
                )
                monitor_logits = predict_logits(
                    model, monitor_examples, int(config["inference_batch_size"])
                )
                monitor_metrics = smoke_metrics(
                    monitor_examples, monitor_logits, arm, tau=diagnostic_tau
                )
                stem = f"seed{seed}__{arm}__{variant}"
                _save_logits(output_dir / "logits" / f"{stem}__monitor.npz", monitor_examples, monitor_logits)
                _save_logits(
                    output_dir / "logits" / f"{stem}__val_threshold.npz",
                    threshold_examples,
                    threshold_logits,
                )
                train_logits = predict_logits(
                    model, examples, int(config["inference_batch_size"])
                )
                _save_logits(output_dir / "logits" / f"{stem}__train.npz", examples, train_logits)
                if variant == "main":
                    probe = monitor_examples[: min(16, len(monitor_examples))]
                    original = predict_logits(model, probe, int(config["inference_batch_size"]))
                    permuted = []
                    for index, example in enumerate(probe):
                        clone = dict(example)
                        order = np.random.default_rng(int(seed) + index + 100_000).permutation(
                            len(example["features"])
                        )
                        clone["features"] = example["features"][order]
                        permuted.append(clone)
                    shuffled_logits = predict_logits(
                        model, permuted, int(config["inference_batch_size"])
                    )
                    max_permutation_delta = max(
                        max_permutation_delta, float(np.max(np.abs(original - shuffled_logits)))
                    )
                checkpoint_path = output_dir / "checkpoints" / f"{stem}.pt"
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    _checkpoint_payload(model, optimizer_state, config, int(seed), arm, variant),
                    checkpoint_path,
                )
                run_rows.append({
                    "seed": int(seed),
                    "arm": arm,
                    "variant": variant,
                    "history": history,
                    "diagnostic_tau_selection": tau_selection,
                    "monitor_metrics": monitor_metrics,
                    "checkpoint": str(checkpoint_path.relative_to(output_dir)),
                })

    if len(parameter_counts) != 1:
        raise RuntimeError(f"parameter mismatch across arms: {sorted(parameter_counts)}")
    if max_permutation_delta > float(config["permutation_atol"]):
        raise RuntimeError(f"permutation invariance failed: max delta {max_permutation_delta}")

    accessed = sorted({Path(path).resolve() for path in train_access + val_access})
    if any("lockbox" in path.name for path in accessed):
        raise RuntimeError("smoke accessed lockbox")
    receipt = {
        "status": "SMOKE_ONLY",
        "allowed_splits": ["train", "val"],
        "lockbox_accessed": False,
        "files_read": [
            {"path": str(path), "sha256": sha256_file(path)} for path in accessed
        ] + [{"path": str(args.config.resolve()), "sha256": sha256_file(args.config.resolve())}],
    }
    write_json(output_dir / "access_receipt.json", receipt)

    artifact_files = sorted(
        path for path in output_dir.rglob("*")
        if path.is_file() and path.name not in {"artifact_manifest.json", "summary.json", "REPORT_WAVE50_SMOKE.md"}
    )
    artifact_manifest = {
        "execution_sources": execution_sources,
        "files": [
            {"path": str(path.relative_to(output_dir)), "sha256": sha256_file(path)}
            for path in artifact_files
        ],
    }
    write_json(output_dir / "artifact_manifest.json", artifact_manifest)

    summary = {
        "status": "SMOKE_ONLY",
        "scientific_claim_allowed": False,
        "git_commit": _git_commit(),
        "execution_sources": execution_sources,
        "artifact_manifest_sha256": sha256_file(output_dir / "artifact_manifest.json"),
        "config_sha256": sha256_file(args.config.resolve()),
        "config": config,
        "parameter_count": next(iter(parameter_counts)),
        "max_permutation_delta": max_permutation_delta,
        "data": {
            "train_fixtures": len(train_examples),
            "train_tokens": len(train_tokens),
            "threshold_fixtures": len(threshold_examples),
            "threshold_tokens": len(threshold_tokens),
            "monitor_fixtures": len(monitor_examples),
            "monitor_tokens": len(monitor_tokens),
            "target_cardinality_train": {
                str(cardinality): sum(len(record["target_families"]) == cardinality for record in train_records)
                for cardinality in range(1, 5)
            },
            "theoretical_softmax_zero_gradient_fixtures": sum(
                len(record["target_families"]) == 4 for record in train_records
            ),
            "shuffle_control": shuffle_report,
        },
        "runtime": {
            "device": "cpu",
            "cpu_threads": int(config["cpu_threads"]),
            "wall_seconds": time.perf_counter() - started,
            "cuda_initialized": torch.cuda.is_initialized(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "max_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
        },
        "runs": run_rows,
        "artifact_hash": stable_hash({
            "config": config,
            "runs": run_rows,
            "shuffle_report": shuffle_report,
        }),
    }
    write_json(output_dir / "summary.json", summary)
    _write_report(output_dir, summary)


if __name__ == "__main__":
    main()
