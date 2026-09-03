"""Mutation suite for the Wave 50 prospective execution boundary."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Callable

from .wave49_schema import write_json
from .wave50_protocol import (
    assert_oracle_absent,
    freeze_files,
    validate_authorized_targets,
    validate_frozen_files,
    validate_pair_token_alignment,
    validate_restricted_receipt,
    validate_stage,
)


def _required_stage(root: Path, phase: str) -> None:
    names = (
        (
            "visible/train.jsonl", "visible/val.jsonl", "labels/train.jsonl",
            "labels/val.jsonl", "prospective_config.json",
        )
        if phase == "training"
        else (
            "visible/lockbox.jsonl", "prospective_config.json",
            "frozen/normalizer.npz", "frozen/thresholds.json",
        )
    )
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")


def _expect_rejection(name: str, fn: Callable[[], None]) -> dict[str, str]:
    try:
        fn()
    except Exception as exc:  # the mutation contract is rejection, not an exception class
        return {"mutation": name, "status": "REJECTED", "reason": str(exc)}
    return {"mutation": name, "status": "ACCEPTED_INVALID"}


def _frozen_mutation(name: str, original: Path) -> dict[str, str]:
    with tempfile.TemporaryDirectory(prefix="wave50-mutation-freeze-") as raw:
        root = Path(raw)
        artifact = root / original.name
        shutil.copy2(original, artifact)
        manifest = root / "manifest.json"
        freeze_files(root, manifest, "test-freeze", [artifact])
        artifact.write_bytes(artifact.read_bytes() + b"mutation")
        return _expect_rejection(
            name, lambda: validate_frozen_files(root, manifest, "test-freeze")
        )


def run_mutation_suite(
    output_dir: Path,
    benchmark_dir: Path,
    public_key: Path,
    prospective_config: Path,
) -> list[dict[str, str]]:
    output_dir = Path(output_dir)
    rows: list[dict[str, str]] = []
    with tempfile.TemporaryDirectory(prefix="wave50-mutation-stage-") as raw:
        root = Path(raw)
        training = root / "training"
        _required_stage(training, "training")
        injected = training / "visible/lockbox.jsonl"
        injected.write_bytes(b"mutation")
        rows.append(_expect_rejection(
            "training_lockbox_visible_injected", lambda: validate_stage(training, "training")
        ))
        injected.unlink()
        (training / "sealed").mkdir()
        (training / "sealed/truth.jsonl").write_bytes(b"mutation")
        rows.append(_expect_rejection(
            "training_sealed_path_injected", lambda: validate_stage(training, "training")
        ))

        labels = root / "labels"
        shutil.copytree(output_dir / "authorized_labels", labels)
        (labels / "lockbox.jsonl").write_bytes(b"mutation")
        rows.append(_expect_rejection(
            "lockbox_label_precreated",
            lambda: validate_authorized_targets(labels, public_key, prospective_config),
        ))

        receipt = json.loads((output_dir / "inference/access_receipt.json").read_text(encoding="utf-8"))
        receipt["operations"].append({"operation": "fit", "fit": True})
        receipt_path = root / "receipt.json"
        write_json(receipt_path, receipt)
        rows.append(_expect_rejection(
            "inference_fit_operation_requested",
            lambda: validate_restricted_receipt(
                receipt_path, "restricted-single-lockbox-inference-complete"
            ),
        ))

    checkpoint = next((output_dir / "frozen_inference_checkpoints").glob("*.pt"))
    prediction = next((output_dir / "inference/logits").glob("*.npz"))
    rows.extend([
        _frozen_mutation("checkpoint_changed_after_freeze", checkpoint),
        _frozen_mutation("normalizer_changed_after_freeze", output_dir / "training/normalizer.npz"),
        _frozen_mutation("threshold_changed_after_freeze", output_dir / "training/thresholds.json"),
        _expect_rejection(
            "pair_token_alignment_changed",
            lambda: validate_pair_token_alignment(["a", "b", "c"], ["a", "c", "b"]),
        ),
        _frozen_mutation("prediction_changed_after_freeze", prediction),
    ])
    with tempfile.TemporaryDirectory(prefix="wave50-mutation-oracle-") as raw:
        root = Path(raw)
        (root / "sealed/oracle").mkdir(parents=True)
        (root / "sealed/oracle/lockbox.jsonl").write_bytes(b"mutation")
        rows.append(_expect_rejection(
            "oracle_created_before_prediction_receipt", lambda: assert_oracle_absent(root)
        ))
    if any(row["status"] != "REJECTED" for row in rows):
        raise RuntimeError(f"mutation suite accepted an invalid package: {rows}")
    write_json(output_dir / "mutation_results.json", {
        "status": "PASS",
        "rejected": len(rows),
        "rows": rows,
    })
    return rows
