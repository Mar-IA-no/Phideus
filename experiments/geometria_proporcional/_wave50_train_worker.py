#!/usr/bin/env python3
"""Restricted Wave 50 trainer; consumes only staged train/val observations and labels."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

from geometria_proporcional.wave49_schema import sha256_file, write_json
from geometria_proporcional.wave50_neural import (
    DeepSetClassifier,
    balanced_target_derangement,
    fit_normalizer,
    load_labeled_records,
    logits_payload,
    parameter_count,
    permute_example_points,
    predict_logits,
    prepare_examples,
    split_tokens,
    stratified_token_subset,
    train_fixed_recipe,
)
from geometria_proporcional.wave50_protocol import (
    ARMS,
    CHECKPOINT_VARIANTS,
    restricted_receipt,
    select_prospective_tau,
    stage_inventory,
    validate_stage,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--forbidden-probe", action="append", type=Path, default=[])
    return parser.parse_args()


def _probe_forbidden(paths: list[Path]) -> list[dict[str, object]]:
    results = []
    for path in paths:
        denied = False
        try:
            path.read_bytes()
        except (PermissionError, FileNotFoundError):
            denied = True
        if not denied:
            raise RuntimeError(f"restricted trainer could read forbidden path: {path.name}")
        results.append({"name": path.name, "denied": True})
    return results


def _save_logits(path: Path, examples: list[dict], logits: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **logits_payload(examples, logits))


def main() -> None:
    args = _parse_args()
    stage = args.stage.resolve()
    output = args.output.resolve()
    if os.geteuid() == 0:
        raise RuntimeError("restricted trainer must not run as root")
    inventory = validate_stage(stage, "training")
    probes = _probe_forbidden(args.forbidden_probe)
    config = json.loads((stage / "prospective_config.json").read_text(encoding="utf-8"))
    torch.set_num_threads(int(config["training"]["cpu_threads"]))
    torch.use_deterministic_algorithms(True)

    train_records, train_reads = load_labeled_records(
        stage / "visible/train.jsonl",
        stage / "labels/train.jsonl",
        None,
        "train",
        expected_schema=config["benchmark"]["source_protocol_schema"],
        expected_compatibility_distance=config["target"]["oracle_compatibility_distance"],
    )
    val_records, val_reads = load_labeled_records(
        stage / "visible/val.jsonl",
        stage / "labels/val.jsonl",
        None,
        "val",
        expected_schema=config["benchmark"]["source_protocol_schema"],
        expected_compatibility_distance=config["target"]["oracle_compatibility_distance"],
    )
    expected = config["validation"]
    threshold_records, monitor_records = split_tokens(
        val_records,
        0.5,
        expected["split_seed"],
        exact_first_tokens=expected["val_threshold_pair_tokens"],
    )
    if len({row["pair_token"] for row in threshold_records}) != expected["val_threshold_pair_tokens"]:
        raise RuntimeError("val-threshold token count mismatch")
    if len({row["pair_token"] for row in monitor_records}) != expected["val_monitor_pair_tokens"]:
        raise RuntimeError("val-monitor token count mismatch")
    write_json(output / "validation_split_manifest.json", {
        "unit": "pair_token",
        "split_seed": expected["split_seed"],
        "val_threshold": sorted({row["pair_token"] for row in threshold_records}),
        "val_monitor": sorted({row["pair_token"] for row in monitor_records}),
    })

    normalizer = fit_normalizer(train_records)
    output.mkdir(parents=True, exist_ok=True)
    np.savez(output / "normalizer.npz", mean=normalizer.mean, std=normalizer.std)
    main_examples = prepare_examples(train_records, normalizer)
    no_eiv_examples = prepare_examples(train_records, normalizer, no_eiv=True)
    threshold_examples = prepare_examples(threshold_records, normalizer)
    monitor_examples = prepare_examples(monitor_records, normalizer)
    threshold_no_eiv = prepare_examples(threshold_records, normalizer, no_eiv=True)
    monitor_no_eiv = prepare_examples(monitor_records, normalizer, no_eiv=True)

    eligible_tokens, _, eligibility = balanced_target_derangement(
        train_records, config["controls"]["target_shuffle"]["seed"]
    )
    eligible = [row for row in train_records if row["pair_token"] in eligible_tokens]
    control_records = stratified_token_subset(
        eligible,
        config["controls"]["target_shuffle"]["max_control_fixtures"],
        config["controls"]["target_shuffle"]["seed"],
        include_target_hash=True,
    )
    selected, shuffled_targets, shuffle_report = balanced_target_derangement(
        control_records, config["controls"]["target_shuffle"]["seed"]
    )
    control_records = [row for row in control_records if row["pair_token"] in selected]
    if not control_records:
        raise RuntimeError("target-shuffle control has no eligible records")
    expected_replacements = config["controls"]["target_shuffle"]["minimum_replacements_per_original_hash"]
    if shuffle_report["minimum_replacements_per_original_hash"] < expected_replacements:
        raise RuntimeError("target-shuffle replacement diversity is below the frozen minimum")
    true_control = prepare_examples(control_records, normalizer)
    shuffled_control = prepare_examples(control_records, normalizer, target_by_token=shuffled_targets)
    shuffle_report["full_pool_eligibility"] = eligibility
    write_json(output / "shuffle_manifest.json", {
        "report": shuffle_report,
        "rows": [
            {
                "pair_token": token,
                "replacement": shuffled_targets[token].astype(int).tolist(),
            }
            for token in sorted(selected)
        ],
    })

    variants = {
        "main": main_examples,
        "no_eiv": no_eiv_examples,
        "true_target_control": true_control,
        "shuffled_target": shuffled_control,
    }
    threshold_logits: dict[str, list[np.ndarray]] = {arm: [] for arm in ARMS}
    order_permutation_checks = []
    runs = []
    counts = set()
    for seed in config["training"]["seeds"]:
        torch.manual_seed(int(seed))
        initial = DeepSetClassifier()
        initial_state = copy.deepcopy(initial.state_dict())
        for arm in ARMS:
            for variant in CHECKPOINT_VARIANTS:
                model = DeepSetClassifier()
                model.load_state_dict(initial_state)
                counts.add(parameter_count(model))
                history, optimizer_state = train_fixed_recipe(
                    model,
                    variants[variant],
                    arm,
                    int(seed),
                    config["training"]["epochs"],
                    config["training"]["batch_pair_tokens"],
                    config["training"]["learning_rate"],
                    config["training"]["weight_decay"],
                )
                stem = f"seed{seed}__{arm}__{variant}"
                checkpoint = output / "checkpoints" / f"{stem}.pt"
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer_state,
                    "torch_rng_state": torch.get_rng_state(),
                    "sampler_recipe": "pair_token_groups_sorted_then_rng_permutation(seed_plus_epoch)",
                    "point_permutation_recipe": "seed_x_1e6_plus_epoch_x_1e4_plus_batch_index",
                    "seed": int(seed),
                    "arm": arm,
                    "variant": variant,
                    "prospective_config_sha256": sha256_file(stage / "prospective_config.json"),
                    "history": history,
                }, checkpoint)
                for split_name, examples in (
                    ("train", variants[variant]),
                    (
                        "val_threshold",
                        threshold_no_eiv if variant == "no_eiv" else threshold_examples,
                    ),
                    (
                        "val_monitor",
                        monitor_no_eiv if variant == "no_eiv" else monitor_examples,
                    ),
                ):
                    logits = predict_logits(model, examples, batch_size=256)
                    _save_logits(output / "logits" / f"{stem}__{split_name}.npz", examples, logits)
                    if variant == "main" and split_name == "val_threshold":
                        threshold_logits[arm].append(logits)
                if variant == "main":
                    reference = predict_logits(model, monitor_examples, batch_size=256)
                    tolerance = float(config["controls"]["order_permutation"]["absolute_logit_tolerance"])
                    for permutation_index in range(3):
                        permutation_seed = (
                            config["validation"]["split_seed"]
                            + int(seed)
                            + 100_000 * permutation_index
                        )
                        permuted_examples = permute_example_points(
                            monitor_examples, seed=permutation_seed
                        )
                        permuted = predict_logits(model, permuted_examples, batch_size=256)
                        max_delta = float(np.max(np.abs(reference - permuted)))
                        order_permutation_checks.append({
                            "seed": int(seed),
                            "arm": arm,
                            "variant": variant,
                            "permutation_index": permutation_index,
                            "permutation_seed": permutation_seed,
                            "n_fixtures": len(monitor_examples),
                            "max_abs_logit_delta": max_delta,
                            "absolute_logit_tolerance": tolerance,
                            "status": "PASS" if max_delta <= tolerance else "FAIL",
                        })
                        if max_delta > tolerance:
                            raise RuntimeError(
                                f"order-permutation invariance failed for {arm}/seed{seed}/"
                                f"perm{permutation_index}: {max_delta} > {tolerance}"
                            )
                runs.append({
                    "seed": int(seed), "arm": arm, "variant": variant,
                    "checkpoint": str(checkpoint.relative_to(output)),
                    "checkpoint_sha256": sha256_file(checkpoint),
                    "history": history,
                })
    if counts != {config["model"]["expected_parameters"]}:
        raise RuntimeError(f"parameter-count mismatch: {sorted(counts)}")
    if len(runs) != config["inference"]["expected_last_epoch_checkpoints"]:
        raise RuntimeError("checkpoint inventory count mismatch")

    thresholds = {}
    for arm in ARMS:
        ensemble = np.mean(np.stack(threshold_logits[arm]), axis=0)
        selection = select_prospective_tau(threshold_examples, ensemble, arm, config)
        thresholds[arm] = selection
    write_json(output / "thresholds.json", thresholds)
    write_json(output / "order_permutation.json", {
        "status": "PASS",
        "checks": order_permutation_checks,
        "max_abs_logit_delta": max(
            row["max_abs_logit_delta"] for row in order_permutation_checks
        ),
    })
    if any(row["status"] != "ADMISSIBLE" for row in thresholds.values()):
        write_json(output / "TRAINING_ABORT.json", {
            "status": "CALIBRATION_INADMISSIBLE",
            "thresholds": thresholds,
        })

    write_json(output / "training_summary.json", {
        "status": "CALIBRATION_INADMISSIBLE" if (output / "TRAINING_ABORT.json").exists() else "TRAINING_COMPLETE",
        "runs": runs,
        "train_fixtures": len(train_records),
        "train_pair_tokens": len({row["pair_token"] for row in train_records}),
        "val_threshold_pair_tokens": len({row["pair_token"] for row in threshold_records}),
        "val_monitor_pair_tokens": len({row["pair_token"] for row in monitor_records}),
    })
    receipt = restricted_receipt("restricted-training-complete", inventory, [
        {"operation": "fit_normalizer", "split": "train"},
        {"operation": "fit_models", "split": "train"},
        {"operation": "select_threshold", "split": "val_threshold"},
        {"operation": "monitor", "split": "val_monitor"},
    ])
    receipt.update({
        "command": sys.argv,
        "input_hashes": {
            relative: sha256_file(stage / relative)
            for relative in inventory
        },
        "output_hashes": {
            relative: sha256_file(output / relative)
            for relative in stage_inventory(output)
        },
        "forbidden_probe_denied": all(row["denied"] for row in probes),
        "forbidden_probes": probes,
        "files_read": sorted(set(train_reads + val_reads + [str(stage / "prospective_config.json")])),
        "output_inventory": stage_inventory(output),
    })
    write_json(output / "access_receipt.json", receipt)


if __name__ == "__main__":
    main()
