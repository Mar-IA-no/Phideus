#!/usr/bin/env python3
"""Inference-only worker for fresh Wave 55 train/val observations."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from geometria_proporcional.wave49_schema import read_jsonl, sha256_file, write_json
from geometria_proporcional.wave50_model import FeatureNormalizer, canonical_point_features
from geometria_proporcional.wave50_neural import prepare_examples
from geometria_proporcional.wave51_factored import DualHeadDeepSet, predict_dual_logits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def inventory(root: Path) -> list[str]:
    return sorted(str(path.relative_to(root)) for path in root.rglob("*") if path.is_file())


def validate_stage(stage: Path, seeds: list[int]) -> list[str]:
    files = inventory(stage)
    required = {
        "visible/train.jsonl",
        "visible/val.jsonl",
        "protocol_config.json",
        "wave55_config.json",
        "frozen/normalizer.npz",
        *(f"frozen/checkpoints/seed{seed}__sigmoid_only.pt" for seed in seeds),
    }
    missing = required - set(files)
    if missing:
        raise RuntimeError(f"inference stage missing files: {sorted(missing)}")
    forbidden_terms = ("oracle", "label", "sealed", "lockbox", "optimizer", "history")
    offenders = [name for name in files if any(term in name.lower() for term in forbidden_terms)]
    if offenders:
        raise RuntimeError(f"inference stage contains forbidden material: {offenders}")
    if set(files) != required:
        raise RuntimeError(f"inference stage has unexpected files: {sorted(set(files) - required)}")
    return files


def load_visible(path: Path, split: str, schema: str) -> list[dict]:
    rows = read_jsonl(path)
    if any(row.get("split") != split or row.get("schema_version") != schema for row in rows):
        raise RuntimeError(f"visible {split} scope/schema mismatch")
    if len({row["fixture_id"] for row in rows}) != len(rows):
        raise RuntimeError(f"duplicate fixture_id in visible {split}")
    return [
        {
            "fixture_id": row["fixture_id"],
            "features": canonical_point_features(row),
            "target": np.zeros(4, dtype=np.float32),
        }
        for row in rows
    ]


def main() -> None:
    args = parse_args()
    stage = args.stage.resolve(strict=True)
    output = args.output.resolve()
    config = json.loads((stage / "wave55_config.json").read_text(encoding="utf-8"))
    seeds = [int(seed) for seed in config["seeds"]]
    files = validate_stage(stage, seeds)
    torch.set_num_threads(int(config["cpu_threads"]))
    torch.use_deterministic_algorithms(True)
    normalizer_data = np.load(stage / "frozen/normalizer.npz", allow_pickle=False)
    normalizer = FeatureNormalizer(normalizer_data["mean"], normalizer_data["std"])
    protocol = json.loads((stage / "protocol_config.json").read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=False)

    result_files = []
    for split in ("train", "val"):
        records = load_visible(
            stage / "visible" / f"{split}.jsonl", split, protocol["schema_version"]
        )
        examples = prepare_examples(records, normalizer)
        fixture_id = np.asarray([record["fixture_id"] for record in records])
        for seed in seeds:
            checkpoint_path = stage / "frozen/checkpoints" / f"seed{seed}__sigmoid_only.pt"
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if set(checkpoint) != {"model_state", "seed", "output"}:
                raise RuntimeError("worker accepts inference-only checkpoints")
            if int(checkpoint["seed"]) != seed or checkpoint["output"] != "sigmoid_only":
                raise RuntimeError("checkpoint identity mismatch")
            model = DualHeadDeepSet()
            model.load_state_dict(checkpoint["model_state"])
            set_logits, choice_logits = predict_dual_logits(
                model, examples, int(config["inference_batch_size"])
            )
            path = output / "logits" / f"seed{seed}__{split}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                path,
                fixture_id=fixture_id,
                set_logits=set_logits,
                choice_logits=choice_logits,
                seed=np.asarray(seed),
                split=np.asarray(split),
            )
            result_files.append({
                "path": str(path.relative_to(output)),
                "sha256": sha256_file(path),
                "rows": len(records),
            })

    write_json(
        output / "access_receipt.json",
        {
            "phase": "wave55-visible-inference-before-oracle",
            "effective_uid": os.geteuid(),
            "command": list(os.sys.argv),
            "fit_operations": False,
            "allowed_splits": ["train", "val"],
            "stage_inventory": files,
            "stage_hashes": {name: sha256_file(stage / name) for name in files},
            "outputs": result_files,
            "oracle_or_labels_available": False,
        },
    )


if __name__ == "__main__":
    main()
