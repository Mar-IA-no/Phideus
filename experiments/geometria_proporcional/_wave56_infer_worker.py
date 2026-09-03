#!/usr/bin/env python3
"""Unprivileged inference-only worker for all fresh Wave 56 visible splits."""

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

SPLITS = ("train", "val", "lockbox")
EXPECTED_UID = 65534
EXPECTED_GID = 65534


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sealed-probe", type=Path, required=True)
    return parser.parse_args()


def inventory(root: Path) -> list[str]:
    return sorted(str(path.relative_to(root)) for path in root.rglob("*") if path.is_file())


def validate_stage(stage: Path, seeds: list[int]) -> list[str]:
    if any(path.is_symlink() for path in stage.rglob("*")):
        raise RuntimeError("inference stage contains a symlink")
    files = inventory(stage)
    required = {
        *(f"visible/{split}.jsonl" for split in SPLITS),
        "protocol_config.json",
        "wave56_config.json",
        "frozen/normalizer.npz",
        *(f"frozen/checkpoints/seed{seed}__sigmoid_only.npz" for seed in seeds),
    }
    missing = required - set(files)
    if missing:
        raise RuntimeError(f"inference stage missing files: {sorted(missing)}")
    forbidden_terms = ("oracle", "label", "sealed", "truth", "optimizer", "history")
    offenders = [name for name in files if any(term in name.lower() for term in forbidden_terms)]
    if offenders:
        raise RuntimeError(f"inference stage contains forbidden material: {offenders}")
    unexpected = set(files) - required
    if unexpected:
        raise RuntimeError(f"inference stage has unexpected files: {sorted(unexpected)}")
    return files


def negative_truth_probe(path: Path) -> dict[str, object]:
    try:
        with path.open("rb") as handle:
            handle.read(1)
    except PermissionError:
        return {"attempted": True, "denied_with": "PermissionError", "passed": True}
    except FileNotFoundError as error:
        raise RuntimeError("sealed truth probe target is missing") from error
    raise RuntimeError("unprivileged worker could read sealed truth")


def process_security_state() -> dict[str, object]:
    status: dict[str, str] = {}
    for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            status[key] = value.strip()
    effective_capabilities = int(status.get("CapEff", "-1"), 16)
    no_new_privileges = int(status.get("NoNewPrivs", "0"))
    supplementary_groups = os.getgroups()
    if effective_capabilities != 0 or no_new_privileges != 1 or supplementary_groups:
        raise PermissionError("worker privilege drop is incomplete")
    return {
        "effective_capabilities_hex": status["CapEff"],
        "no_new_privileges": no_new_privileges,
        "supplementary_groups": supplementary_groups,
    }


def load_visible(path: Path, split: str, schema: str) -> list[dict[str, object]]:
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
    if Path.cwd().resolve() != stage:
        raise RuntimeError("worker cwd must be the isolated staging directory")
    if os.geteuid() != EXPECTED_UID or os.getegid() != EXPECTED_GID:
        raise RuntimeError("worker must run as nobody/nogroup")
    security = process_security_state()
    output = (stage / args.output).resolve()
    if output.parent != stage or output.name != "inference":
        raise ValueError("worker output must be stage/inference")

    config = json.loads((stage / "wave56_config.json").read_text(encoding="utf-8"))
    seeds = [int(seed) for seed in config["seeds"]]
    files = validate_stage(stage, seeds)
    probe = negative_truth_probe(args.sealed_probe)
    torch.set_num_threads(int(config["cpu_threads"]))
    torch.use_deterministic_algorithms(True)
    with np.load(stage / "frozen/normalizer.npz", allow_pickle=False) as normalizer_data:
        normalizer = FeatureNormalizer(normalizer_data["mean"], normalizer_data["std"])
    protocol = json.loads((stage / "protocol_config.json").read_text(encoding="utf-8"))
    output.mkdir(mode=0o700, parents=False, exist_ok=False)

    result_files: list[dict[str, object]] = []
    for split in SPLITS:
        records = load_visible(
            stage / "visible" / f"{split}.jsonl", split, protocol["schema_version"]
        )
        examples = prepare_examples(records, normalizer)
        fixture_id = np.asarray([record["fixture_id"] for record in records])
        for seed in seeds:
            model = DualHeadDeepSet()
            checkpoint_path = stage / "frozen/checkpoints" / f"seed{seed}__sigmoid_only.npz"
            with np.load(checkpoint_path, allow_pickle=False) as checkpoint:
                expected_state = set(model.state_dict())
                observed_state = {
                    name.removeprefix("state::")
                    for name in checkpoint.files
                    if name.startswith("state::")
                }
                if set(checkpoint.files) != {
                    "seed",
                    "output",
                    *(f"state::{name}" for name in expected_state),
                }:
                    raise RuntimeError("worker accepts only the canonical inference state")
                if int(checkpoint["seed"]) != seed or str(checkpoint["output"]) != "sigmoid_only":
                    raise RuntimeError("checkpoint identity mismatch")
                if observed_state != expected_state:
                    raise RuntimeError("checkpoint model-state schema mismatch")
                model_state = {
                    name: torch.from_numpy(checkpoint[f"state::{name}"].copy())
                    for name in model.state_dict()
                }
            model.load_state_dict(model_state)
            set_logits, choice_logits = predict_dual_logits(
                model, examples, int(config["inference_batch_size"])
            )
            path = output / "logits" / f"seed{seed}__{split}.npz"
            path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            np.savez_compressed(
                path,
                fixture_id=fixture_id,
                set_logits=set_logits,
                choice_logits=choice_logits,
                seed=np.asarray(seed),
                split=np.asarray(split),
            )
            result_files.append(
                {
                    "path": str(path.relative_to(output)),
                    "sha256": sha256_file(path),
                    "rows": len(records),
                }
            )

    write_json(
        output / "access_receipt.json",
        {
            "phase": "wave56-visible-inference-before-any-oracle",
            "effective_uid": os.geteuid(),
            "effective_gid": os.getegid(),
            "process_security": security,
            "working_directory_is_stage": True,
            "fit_operations": False,
            "allowed_splits": list(SPLITS),
            "stage_inventory": files,
            "stage_hashes": {name: sha256_file(stage / name) for name in files},
            "sealed_truth_probe": probe,
            "outputs": result_files,
            "oracle_or_labels_available": False,
        },
    )


if __name__ == "__main__":
    main()
