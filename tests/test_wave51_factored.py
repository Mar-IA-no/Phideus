"""Tests for the Wave 51 factored set/choice development smoke."""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from geometria_proporcional.wave51_factored import (  # noqa: E402
    DualHeadDeepSet,
    choice_metrics,
    make_optimizer,
    parameter_count,
    state_dict_digest,
    token_metric_rows,
    train_epochs,
)


def _examples() -> list[dict]:
    families = np.eye(4, dtype=np.float32)
    rows = []
    for token_index in range(8):
        target = families[token_index % 4]
        for view in range(2):
            rows.append({
                "fixture_id": f"t{token_index}-v{view}",
                "pair_token": f"t{token_index}",
                "design_stratum": "NEAR_RIVAL" if token_index < 4 else "FAR_RIVAL",
                "features": np.asarray([
                    [token_index / 8, view, 0.1, 0.2, 0.0, 1.0],
                    [token_index / 8 + 0.1, view, 0.1, 0.2, 0.0, 1.0],
                ], dtype=np.float32),
                "target": target.copy(),
            })
    return rows


def test_dual_head_is_permutation_invariant_and_parameter_matched():
    torch.manual_seed(17)
    first = DualHeadDeepSet().eval()
    torch.manual_seed(17)
    second = DualHeadDeepSet().eval()
    assert parameter_count(first) == parameter_count(second) == 13_384
    assert state_dict_digest(first) == state_dict_digest(second)

    points = torch.randn(3, 7, 6)
    mask = torch.ones(3, 7, dtype=torch.bool)
    permutation = torch.tensor([4, 0, 6, 2, 1, 5, 3])
    with torch.no_grad():
        original = first(points, mask)
        shuffled = first(points[:, permutation], mask[:, permutation])
    for left, right in zip(original, shuffled, strict=True):
        assert torch.allclose(left, right, atol=1e-7, rtol=0.0)


def test_factored_phase_a_matches_sigmoid_and_phase_b_preserves_set_path():
    torch.manual_seed(23)
    initial = DualHeadDeepSet().state_dict()
    sigmoid = DualHeadDeepSet()
    factored = DualHeadDeepSet()
    sigmoid.load_state_dict(copy.deepcopy(initial))
    factored.load_state_dict(copy.deepcopy(initial))
    examples = _examples()

    for model in (sigmoid, factored):
        optimizer = make_optimizer(model.parameters(), 1e-3, 1e-4)
        train_epochs(model, examples, "set_bce", 23, 0, 2, 4, optimizer)
    assert state_dict_digest(sigmoid) == state_dict_digest(factored)

    frozen_before = state_dict_digest(
        factored, prefixes=("point_mlp.", "set_mlp.", "set_head.")
    )
    choice_before = state_dict_digest(factored, prefixes=("choice_head.",))
    for name, parameter in factored.named_parameters():
        parameter.requires_grad_(name.startswith("choice_head."))
    optimizer = make_optimizer(factored.choice_head.parameters(), 1e-3, 1e-4)
    train_epochs(factored, examples, "choice_partial", 23, 2, 2, 4, optimizer)
    assert state_dict_digest(
        factored, prefixes=("point_mlp.", "set_mlp.", "set_head.")
    ) == frozen_before
    assert state_dict_digest(factored, prefixes=("choice_head.",)) != choice_before


def test_joint_objective_updates_both_heads():
    torch.manual_seed(31)
    model = DualHeadDeepSet()
    set_before = state_dict_digest(model, prefixes=("set_head.",))
    choice_before = state_dict_digest(model, prefixes=("choice_head.",))
    optimizer = make_optimizer(model.parameters(), 1e-3, 1e-4)
    train_epochs(model, _examples(), "joint_equal", 31, 0, 1, 4, optimizer)
    assert state_dict_digest(model, prefixes=("set_head.",)) != set_before
    assert state_dict_digest(model, prefixes=("choice_head.",)) != choice_before


def test_staged_unfrozen_updates_set_path_and_choice_after_shared_phase_a():
    torch.manual_seed(37)
    phase_a = DualHeadDeepSet()
    optimizer = make_optimizer(phase_a.parameters(), 1e-3, 1e-4)
    train_epochs(phase_a, _examples(), "set_bce", 37, 0, 1, 4, optimizer)
    staged = copy.deepcopy(phase_a)
    set_before = state_dict_digest(
        staged, prefixes=("point_mlp.", "set_mlp.", "set_head.")
    )
    choice_before = state_dict_digest(staged, prefixes=("choice_head.",))
    optimizer = make_optimizer(staged.parameters(), 1e-3, 1e-4)
    train_epochs(staged, _examples(), "choice_partial", 37, 1, 1, 4, optimizer)
    assert state_dict_digest(
        staged, prefixes=("point_mlp.", "set_mlp.", "set_head.")
    ) != set_before
    assert state_dict_digest(staged, prefixes=("choice_head.",)) != choice_before


def test_choice_metric_gates_selection_to_predicted_set():
    examples = [{
        "pair_token": "a",
        "target": np.asarray([1, 0, 0, 0], dtype=np.float32),
    }]
    set_logits = np.asarray([[2.0, 2.0, -2.0, -2.0]])
    choice_logits = np.asarray([[1.0, 3.0, 9.0, 0.0]])
    result = choice_metrics(examples, set_logits, choice_logits, "sigmoid", tau=0.5)
    assert result["choice_top1_compatible"] == 0.0
    assert result["choice_top1_gated_compatible"] == 0.0

    choice_logits = np.asarray([[3.0, 1.0, 9.0, 0.0]])
    result = choice_metrics(examples, set_logits, choice_logits, "sigmoid", tau=0.5)
    assert result["choice_top1_compatible"] == 0.0
    assert result["choice_top1_gated_compatible"] == 1.0


def test_choice_metric_falls_back_when_predicted_set_is_empty():
    examples = [{
        "pair_token": "a",
        "target": np.asarray([0, 1, 0, 0], dtype=np.float32),
    }]
    result = choice_metrics(
        examples,
        np.full((1, 4), -10.0),
        np.asarray([[0.0, 2.0, 1.0, -1.0]]),
        "sigmoid",
        tau=0.9,
    )
    assert result["choice_top1_compatible"] == pytest.approx(1.0)
    assert result["choice_top1_gated_compatible"] == pytest.approx(1.0)


def test_token_metric_rows_average_canonical_views():
    examples = _examples()[:2]
    set_logits = np.asarray([[3.0, -3.0, -3.0, -3.0], [-3.0, -3.0, -3.0, -3.0]])
    choice_logits = np.asarray([[3.0, 2.0, 1.0, 0.0], [0.0, 3.0, 2.0, 1.0]])
    rows = token_metric_rows(examples, set_logits, choice_logits, "sigmoid", tau=0.5)
    assert len(rows) == 1
    assert rows[0]["pair_token"] == "t0"
    assert rows[0]["n_canonical_views"] == 2
    assert rows[0]["set_recall"] == pytest.approx(0.5)
    assert rows[0]["choice_top1_compatible"] == pytest.approx(0.5)
    assert rows[0]["choice_top1_gated_compatible"] == pytest.approx(0.5)


def _load_runner_module():
    script = REPO_ROOT / "experiments/geometria_proporcional/run_wave51_factored_smoke.py"
    spec = importlib.util.spec_from_file_location("wave51_factored_runner", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runner_refuses_lockbox_symlink_before_read(tmp_path: Path):
    module = _load_runner_module()
    lockbox = tmp_path / "lockbox.jsonl"
    lockbox.write_text("", encoding="utf-8")
    disguised = tmp_path / "train.jsonl"
    disguised.symlink_to(lockbox)
    with pytest.raises(ValueError, match="resolves to lockbox"):
        module._assert_no_lockbox_inputs([disguised])


def test_source_binding_rejects_mutated_wave50_input(tmp_path: Path):
    module = _load_runner_module()
    source = tmp_path / "wave50"
    relatives = (
        "benchmark/visible/train.jsonl",
        "authorized_labels/train.jsonl",
        "benchmark/visible/val.jsonl",
        "authorized_labels/val.jsonl",
        "benchmark/protocol_config.json",
    )
    file_hashes = {}
    for index, relative in enumerate(relatives):
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"fixture-{index}\n", encoding="utf-8")
        file_hashes[relative] = module.sha256_file(path)
    entries = {relative: {"sha256": digest} for relative, digest in file_hashes.items()}
    execution_path = source / "execution_manifest.json"
    training_path = source / "training_manifest.json"
    execution_path.write_text(json.dumps({"files": entries}), encoding="utf-8")
    training_path.write_text(json.dumps({
        "git_commit": "canonical-commit",
        "phase": "training-frozen-before-lockbox-mount",
        "files": entries,
    }), encoding="utf-8")
    config = {"source_binding": {
        "canonical_wave50_commit": "canonical-commit",
        "training_manifest_phase": "training-frozen-before-lockbox-mount",
        "execution_manifest_sha256": module.sha256_file(execution_path),
        "training_manifest_sha256": module.sha256_file(training_path),
        "files": file_hashes,
    }}
    checked = module._validate_source_binding(source, config)
    assert len(checked["files"]) == 5
    missing = copy.deepcopy(config)
    missing["source_binding"]["files"].pop(relatives[0])
    with pytest.raises(ValueError, match="exactly the authorized"):
        module._validate_source_binding(source, missing)
    (source / relatives[0]).write_text("mutated\n", encoding="utf-8")
    with pytest.raises(ValueError, match=relatives[0]):
        module._validate_source_binding(source, config)
