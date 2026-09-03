#!/usr/bin/env python3
"""Restricted Wave 50 inference worker; it exposes no fitting entrypoint."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

from geometria_proporcional.wave49_schema import sha256_file, write_json
from geometria_proporcional.wave50_model import (
    DeepSetClassifier,
    FeatureNormalizer,
    predict_logits,
    canonical_point_features,
)
from geometria_proporcional.wave49_schema import read_jsonl


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--forbidden-probe", action="append", type=Path, default=[])
    return parser.parse_args()


def _probe_forbidden(paths: list[Path]) -> list[dict[str, object]]:
    rows = []
    for path in paths:
        denied = False
        try:
            path.read_bytes()
        except (PermissionError, FileNotFoundError):
            denied = True
        if not denied:
            raise RuntimeError(f"restricted inference could read forbidden path: {path.name}")
        rows.append({"name": path.name, "denied": True})
    return rows


def _inventory(root: Path) -> list[str]:
    return sorted(str(path.relative_to(root)) for path in root.rglob("*") if path.is_file())


def _validate_inference_stage(root: Path) -> list[str]:
    inventory = _inventory(root)
    required = {
        "visible/lockbox.jsonl", "prospective_config.json",
        "frozen/normalizer.npz", "frozen/thresholds.json",
    }
    missing = required - set(inventory)
    if missing:
        raise RuntimeError(f"inference stage missing required files: {sorted(missing)}")
    forbidden = ("labels", "sealed", "optimizer", "train.jsonl", "val.jsonl", "wave50_neural")
    offenders = [item for item in inventory if any(term in item.lower() for term in forbidden)]
    if offenders:
        raise RuntimeError(f"inference stage contains forbidden files: {offenders}")
    allowed = required | {
        "worker.py",
        "source/geometria_proporcional/__init__.py",
        "source/geometria_proporcional/wave49_schema.py",
        "source/geometria_proporcional/wave50_model.py",
    }
    allowed.update(
        item for item in inventory
        if item.startswith("frozen/checkpoints/") and item.endswith(".pt")
    )
    unexpected = set(inventory) - allowed
    if unexpected:
        raise RuntimeError(f"inference stage contains files outside allowlist: {sorted(unexpected)}")
    return inventory


def _load_records(visible_path: Path, expected_schema: str) -> tuple[list[dict], list[str]]:
    rows = read_jsonl(visible_path)
    if any(row.get("split") != "lockbox" or row.get("schema_version") != expected_schema for row in rows):
        raise RuntimeError("lockbox visible row scope/schema mismatch")
    if len({row["fixture_id"] for row in rows}) != len(rows):
        raise RuntimeError("duplicate lockbox fixture_id")
    records = [{
        "fixture_id": row["fixture_id"],
        "features": canonical_point_features(row),
        "target": np.zeros(4, dtype=np.float32),
    } for row in rows]
    return records, [str(visible_path)]


def _prepare(records: list[dict], normalizer: FeatureNormalizer, no_eiv: bool) -> list[dict]:
    return [{**row, "features": normalizer.transform(row["features"], no_eiv=no_eiv)} for row in records]


def main() -> None:
    args = _parse_args()
    stage = args.stage.resolve()
    output = args.output.resolve()
    if os.geteuid() == 0:
        raise RuntimeError("restricted inference must not run as root")
    inventory = _validate_inference_stage(stage)
    probes = _probe_forbidden(args.forbidden_probe)
    config = json.loads((stage / "prospective_config.json").read_text(encoding="utf-8"))
    torch.set_num_threads(int(config["training"]["cpu_threads"]))
    torch.use_deterministic_algorithms(True)
    normalizer_data = np.load(stage / "frozen/normalizer.npz")
    normalizer = FeatureNormalizer(normalizer_data["mean"], normalizer_data["std"])
    records, reads = _load_records(
        stage / "visible/lockbox.jsonl", config["benchmark"]["source_protocol_schema"]
    )
    output.mkdir(parents=True, exist_ok=True)
    prediction_files = []
    for checkpoint_path in sorted((stage / "frozen/checkpoints").glob("*.pt")):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if "optimizer_state" in checkpoint:
            raise RuntimeError("inference staging must contain inference-only checkpoints")
        model = DeepSetClassifier()
        model.load_state_dict(checkpoint["model_state"])
        examples = _prepare(records, normalizer, checkpoint["variant"] == "no_eiv")
        logits = predict_logits(model, examples, batch_size=256)
        path = output / "logits" / f"{checkpoint_path.stem}__lockbox.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            fixture_id=np.asarray([row["fixture_id"] for row in records]),
            logits=logits.astype(np.float32),
            seed=np.asarray(checkpoint["seed"]),
            arm=np.asarray(checkpoint["arm"]),
            variant=np.asarray(checkpoint["variant"]),
        )
        prediction_files.append({
            "path": str(path.relative_to(output)),
            "sha256": sha256_file(path),
            "rows": len(records),
        })
    expected = config["inference"]["expected_last_epoch_checkpoints"]
    if len(prediction_files) != expected:
        raise RuntimeError(f"inference checkpoint count mismatch: {len(prediction_files)} != {expected}")
    receipt = {
        "phase": "restricted-single-lockbox-inference-complete",
        "effective_uid": os.geteuid(),
        "effective_gid": os.getegid(),
        "allowlist": inventory,
        "operations": [
        {"operation": "load_frozen_normalizer", "fit": False},
        {"operation": "load_inference_only_checkpoints", "fit": False},
        {"operation": "predict", "split": "lockbox", "fit": False},
        ],
    }
    receipt.update({
        "command": sys.argv,
        "input_hashes": {
            relative: sha256_file(stage / relative)
            for relative in inventory
        },
        "output_hashes": {
            relative: sha256_file(output / relative)
            for relative in _inventory(output)
        },
        "forbidden_probe_denied": all(row["denied"] for row in probes),
        "single_lockbox_pass": True,
        "forbidden_probes": probes,
        "files_read": reads,
        "predictions": prediction_files,
    })
    write_json(output / "access_receipt.json", receipt)


if __name__ == "__main__":
    main()
