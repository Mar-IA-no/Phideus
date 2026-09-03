"""Protocol-level tests for prospective Wave 50 boundaries and readout."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from geometria_proporcional.wave49_schema import write_json  # noqa: E402
from geometria_proporcional.wave50_model import (  # noqa: E402
    DeepSetClassifier,
    permute_example_points,
    predict_logits,
)
from geometria_proporcional.wave50_protocol import (  # noqa: E402
    assert_oracle_absent,
    freeze_files,
    paired_bootstrap_difference,
    select_prospective_tau,
    validate_frozen_files,
    validate_restricted_receipt,
    validate_stage,
)


def _example(token: str, target: list[int], logit_hint: float = 0.0) -> dict:
    families = ("PROP", "AFFINE_OFFSET", "POWER_NONUNIT", "SATURATING")
    return {
        "fixture_id": f"{token}-{logit_hint}",
        "pair_token": token,
        "target": np.asarray(target, dtype=np.float32),
        "target_families": tuple(f for f, active in zip(families, target, strict=True) if active),
        "family_id": next(f for f, active in zip(families, target, strict=True) if active),
        "design_stratum": "NEAR_RIVAL",
        "n": 8,
        "noise_mode": "low_balanced",
        "covariance_mode": "homoscedastic",
        "range_mode": "wide",
    }


def _tau_config(width_excess: float = 0.25, incompatible: float = 0.4) -> dict:
    return {
        "validation": {
            "threshold_grid": {"start": 0.05, "stop": 0.95, "step": 0.05},
            "threshold_constraints": {
                "mean_width_minus_mean_target_cardinality_max": width_excess,
                "fixture_any_incompatible_rate_max": incompatible,
            },
        }
    }


def _minimal_stage(root: Path, phase: str) -> None:
    if phase == "training":
        names = (
            "visible/train.jsonl", "visible/val.jsonl", "labels/train.jsonl",
            "labels/val.jsonl", "prospective_config.json",
        )
    else:
        names = (
            "visible/lockbox.jsonl", "prospective_config.json",
            "frozen/normalizer.npz", "frozen/thresholds.json",
        )
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")


def test_training_stage_rejects_lockbox_and_sealed_content(tmp_path: Path):
    _minimal_stage(tmp_path, "training")
    assert "visible/train.jsonl" in validate_stage(tmp_path, "training")
    (tmp_path / "visible/lockbox.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(RuntimeError, match="forbidden"):
        validate_stage(tmp_path, "training")
    (tmp_path / "visible/lockbox.jsonl").unlink()
    (tmp_path / "sealed").mkdir()
    (tmp_path / "sealed/truth.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(RuntimeError, match="forbidden"):
        validate_stage(tmp_path, "training")


def test_stage_rejects_arbitrary_files_outside_allowlist(tmp_path: Path):
    _minimal_stage(tmp_path, "training")
    (tmp_path / "extra-readable.txt").write_text("not allowed", encoding="utf-8")
    with pytest.raises(RuntimeError, match="outside allowlist"):
        validate_stage(tmp_path, "training")


def test_inference_stage_rejects_labels_and_optimizer(tmp_path: Path):
    _minimal_stage(tmp_path, "inference")
    assert "visible/lockbox.jsonl" in validate_stage(tmp_path, "inference")
    (tmp_path / "labels").mkdir()
    (tmp_path / "labels/lockbox.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(RuntimeError, match="forbidden"):
        validate_stage(tmp_path, "inference")


def test_frozen_manifest_detects_post_freeze_mutation(tmp_path: Path):
    artifact = tmp_path / "checkpoint.pt"
    artifact.write_bytes(b"original")
    manifest = tmp_path / "manifest.json"
    freeze_files(tmp_path, manifest, "frozen", [artifact])
    validate_frozen_files(tmp_path, manifest, "frozen")
    artifact.write_bytes(b"mutated")
    with pytest.raises(RuntimeError, match="mismatch"):
        validate_frozen_files(tmp_path, manifest, "frozen")


def test_tau_selection_aborts_when_constraints_are_impossible():
    examples = [_example("a", [1, 0, 0, 0]), _example("b", [0, 1, 0, 0])]
    logits = np.zeros((2, 4), dtype=np.float32)
    result = select_prospective_tau(examples, logits, "sigmoid_set", _tau_config(-1.0, -1.0))
    assert result["status"] == "CALIBRATION_INADMISSIBLE"


def test_tau_selection_uses_recall_then_declared_tiebreaks():
    examples = [_example("a", [1, 1, 0, 0]), _example("b", [0, 1, 1, 0])]
    logits = np.asarray([[3.0, 2.0, -3.0, -3.0], [-3.0, 3.0, 2.0, -3.0]])
    result = select_prospective_tau(examples, logits, "sigmoid_set", _tau_config(0.5, 0.5))
    assert result["status"] == "ADMISSIBLE"
    assert result["selected"]["set_recall"] == pytest.approx(1.0)
    assert result["selected"]["width"] == pytest.approx(2.0)


def test_bootstrap_is_paired_and_rejects_alignment_mismatch():
    left = [{"pair_token": f"t{i}", "score": float(i + 1)} for i in range(8)]
    right = [{"pair_token": f"t{i}", "score": float(i)} for i in range(8)]
    result = paired_bootstrap_difference(left, right, "score", 500, 9, 0.95)
    assert result["left_minus_right"] == pytest.approx(1.0)
    assert result["ci_lo"] == pytest.approx(1.0)
    with pytest.raises(RuntimeError, match="alignment"):
        paired_bootstrap_difference(left, right[:-1], "score", 10, 9, 0.95)


def test_inference_receipt_rejects_fit_and_early_oracle(tmp_path: Path):
    receipt = tmp_path / "receipt.json"
    payload = {
        "phase": "restricted-single-lockbox-inference-complete",
        "effective_uid": 65534,
        "forbidden_probe_denied": True,
        "command": ["worker"],
        "input_hashes": {},
        "output_hashes": {},
        "operations": [
            {"operation": "load_frozen_normalizer", "fit": False},
            {"operation": "load_inference_only_checkpoints", "fit": False},
            {"operation": "predict", "fit": False},
        ],
        "allowlist": [],
        "single_lockbox_pass": True,
    }
    write_json(receipt, payload)
    validate_restricted_receipt(receipt, payload["phase"])
    payload["operations"].append({"operation": "fit", "fit": True})
    write_json(receipt, payload)
    with pytest.raises(RuntimeError, match="fit operation"):
        validate_restricted_receipt(receipt, payload["phase"])

    benchmark = tmp_path / "benchmark"
    assert_oracle_absent(benchmark)
    (benchmark / "sealed/oracle").mkdir(parents=True)
    (benchmark / "sealed/oracle/lockbox.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="oracle exists"):
        assert_oracle_absent(benchmark)


def test_deepset_logits_are_invariant_to_point_permutation():
    torch.manual_seed(17)
    model = DeepSetClassifier()
    examples = [
        {
            "features": np.arange(n * 6, dtype=np.float32).reshape(n, 6) / 10.0,
            "target": np.asarray([1, 0, 1, 0], dtype=np.float32),
        }
        for n in (3, 8, 13)
    ]
    reference = predict_logits(model, examples, batch_size=3)
    permuted = predict_logits(model, permute_example_points(examples, seed=91), batch_size=3)
    assert np.max(np.abs(reference - permuted)) <= 1e-6
