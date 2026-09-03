"""Unit tests for the Wave 50 matched neural development smoke."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from geometria_proporcional.wave50_neural import (  # noqa: E402
    DeepSetClassifier,
    balanced_target_derangement,
    load_smoke_records,
    parameter_count,
    partial_label_softmax_loss,
    split_tokens,
    smoke_metrics,
    token_batch_indices,
)


def _record(token: str, target: tuple[str, ...], stratum: str = "NEAR_RIVAL") -> dict:
    families = ("PROP", "AFFINE_OFFSET", "POWER_NONUNIT", "SATURATING")
    vector = np.asarray([family in target for family in families], dtype=np.float32)
    return {
        "fixture_id": f"{token}-view",
        "pair_token": token,
        "features": np.ones((3, 6), dtype=np.float32),
        "target": vector,
        "target_families": target,
        "family_id": target[0],
        "design_stratum": stratum,
        "representation": "original",
        "covariance_knowledge": "full",
        "n": 3,
    }


def test_matched_models_have_identical_parameters_and_initial_state():
    torch.manual_seed(17)
    first = DeepSetClassifier()
    torch.manual_seed(17)
    second = DeepSetClassifier()
    assert parameter_count(first) == parameter_count(second)
    for left, right in zip(first.parameters(), second.parameters(), strict=True):
        assert torch.equal(left, right)


def test_deepset_is_permutation_invariant():
    torch.manual_seed(19)
    model = DeepSetClassifier().eval()
    points = torch.randn(2, 7, 6)
    mask = torch.ones(2, 7, dtype=torch.bool)
    permutation = torch.tensor([4, 0, 6, 2, 1, 5, 3])
    with torch.no_grad():
        original = model(points, mask)
        shuffled = model(points[:, permutation], mask[:, permutation])
    assert torch.allclose(original, shuffled, atol=1e-7, rtol=0.0)


def test_partial_label_loss_uses_the_same_nonempty_target_set():
    logits = torch.tensor([[0.1, 0.2, 0.3, 0.4], [1.0, -1.0, 0.0, 0.5]])
    all_compatible = torch.ones_like(logits)
    singleton = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    assert partial_label_softmax_loss(logits, all_compatible).item() == pytest.approx(0.0)
    expected = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 3]))
    assert partial_label_softmax_loss(logits, singleton).item() == pytest.approx(expected.item())
    with pytest.raises(ValueError, match="non-empty"):
        partial_label_softmax_loss(logits, torch.zeros_like(logits))


def test_smoke_loader_refuses_lockbox_before_reading(tmp_path: Path):
    with pytest.raises(ValueError, match="smoke split"):
        load_smoke_records(tmp_path, "lockbox")


def test_smoke_loader_rejects_train_symlinked_to_lockbox(tmp_path: Path):
    visible = tmp_path / "visible"
    oracle = tmp_path / "sealed" / "oracle"
    visible.mkdir(parents=True)
    oracle.mkdir(parents=True)
    (visible / "lockbox.jsonl").write_text("", encoding="utf-8")
    (oracle / "lockbox.jsonl").write_text("", encoding="utf-8")
    (visible / "train.jsonl").symlink_to(visible / "lockbox.jsonl")
    (oracle / "train.jsonl").symlink_to(oracle / "lockbox.jsonl")
    with pytest.raises(ValueError, match="resolves to lockbox"):
        load_smoke_records(tmp_path, "train")


def test_force_output_guard_rejects_repository_and_benchmark_tree():
    script = REPO_ROOT / "experiments/geometria_proporcional/run_wave50_smoke.py"
    spec = importlib.util.spec_from_file_location("wave50_smoke_runner", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    benchmark = REPO_ROOT / "data/geometria_proporcional/wave49"
    with pytest.raises(ValueError, match="repository root"):
        module._validate_paths(benchmark, REPO_ROOT)
    with pytest.raises(ValueError, match="disjoint"):
        module._validate_paths(benchmark, benchmark / "nested-output")
    with pytest.raises(ValueError, match="must be a child"):
        module._validate_paths(benchmark, REPO_ROOT.parent / "other-project/output")


def test_token_split_never_separates_correlated_views():
    records = []
    for index in range(12):
        target = ("PROP",) if index % 2 == 0 else ("AFFINE_OFFSET",)
        records.extend([
            _record(f"token-{index}", target),
            {**_record(f"token-{index}", target), "fixture_id": f"token-{index}-scaled"},
        ])
    first, second = split_tokens(records, 0.5, seed=7)
    first_tokens = {row["pair_token"] for row in first}
    second_tokens = {row["pair_token"] for row in second}
    assert first_tokens.isdisjoint(second_tokens)
    assert first_tokens | second_tokens == {f"token-{index}" for index in range(12)}


def test_token_split_honors_exact_global_count_without_breaking_strata():
    records = []
    for index in range(23):
        target = ("PROP",) if index < 11 else ("PROP", "AFFINE_OFFSET")
        stratum = "NEAR_RIVAL" if index % 2 else "FAR_RIVAL"
        records.append(_record(f"exact-{index}", target, stratum))
    first, second = split_tokens(records, 0.5, seed=13, exact_first_tokens=11)
    assert len({row["pair_token"] for row in first}) == 11
    assert len({row["pair_token"] for row in second}) == 12
    assert {
        (row["design_stratum"], len(row["target_families"])) for row in first
    } == {
        (row["design_stratum"], len(row["target_families"])) for row in second
    }


def test_balanced_derangement_has_zero_target_matches_and_excludes_single_hash():
    records = []
    for index in range(4):
        records.append(_record(f"prop-{index}", ("PROP",)))
        records.append(_record(f"affine-{index}", ("AFFINE_OFFSET",)))
        records.append(_record(f"power-{index}", ("POWER_NONUNIT",)))
    for index in range(3):
        records.append(_record(f"all-{index}", (
            "AFFINE_OFFSET", "POWER_NONUNIT", "PROP", "SATURATING"
        )))
    selected, mapping, report = balanced_target_derangement(records, seed=11)
    original = {row["pair_token"]: row["target"] for row in records}
    assert selected
    assert report["residual_target_matches"] == 0
    assert report["minimum_replacements_per_original_hash"] >= 2
    assert all(not np.array_equal(original[token], mapping[token]) for token in selected)
    assert all(not token.startswith("all-") for token in selected)
    assert any(row["reason"] == "fewer_than_three_target_hashes" for row in report["excluded_strata"])


def test_training_batches_keep_all_views_of_each_token_together():
    records = []
    for index in range(9):
        target = ("PROP",) if index % 2 == 0 else ("AFFINE_OFFSET",)
        records.extend([
            _record(f"token-{index}", target),
            {**_record(f"token-{index}", target), "fixture_id": f"token-{index}-scaled"},
        ])
    batches = token_batch_indices(records, batch_tokens=3, seed=17, epoch=0)
    seen = set()
    for batch in batches:
        batch_tokens = {records[int(index)]["pair_token"] for index in batch}
        for token in batch_tokens:
            all_indices = {i for i, row in enumerate(records) if row["pair_token"] == token}
            assert all_indices <= set(batch.tolist())
        assert seen.isdisjoint(batch_tokens)
        seen.update(batch_tokens)
    assert len(seen) == 9


def test_membership_auc_is_computed_after_token_level_view_averaging():
    examples = []
    logits = []
    for index in range(8):
        target = ("PROP",) if index < 4 else ("AFFINE_OFFSET",)
        for view in range(2):
            record = _record(f"token-{index}", target)
            record["fixture_id"] = f"token-{index}-view-{view}"
            examples.append(record)
            logits.append([2.0 if index < 4 else -2.0, -2.0 if index < 4 else 2.0, 0.0, 0.0])
    metrics = smoke_metrics(examples, np.asarray(logits), "sigmoid_set")
    assert metrics["n_pair_tokens"] == 8
    assert metrics["overall"]["membership_auc_by_family"]["PROP"] == pytest.approx(1.0)
    assert metrics["overall"]["membership_auc_by_family"]["AFFINE_OFFSET"] == pytest.approx(1.0)
