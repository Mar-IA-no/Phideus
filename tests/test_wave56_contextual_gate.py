from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
from pathlib import Path

import numpy as np
import pytest

from geometria_proporcional.wave53_uncertainty import nonempty_sets
from geometria_proporcional.wave56_contextual_gate import (
    FEATURE_NAMES,
    apply_gate,
    contextual_design,
    disagreement_weights,
    fit_weighted_scaler,
    stratified_gain_shuffle,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_runner():
    spec = importlib.util.spec_from_file_location(
        "wave56_retrospective_test",
        REPO_ROOT / "experiments/geometria_proporcional/run_wave56_retrospective.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def synthetic_design_inputs():
    logits = np.asarray([[1.0, -0.5, 0.2, -1.0], [-2.0, -1.0, -0.5, -0.2]])
    per_seed = np.stack([logits - 0.1, logits, logits + 0.1])
    mass = np.full((2, 15), 1.0 / 15.0)
    risk = np.asarray(
        [
            np.tile([0.4, 0.1, 0.7, 0.8], (3, 1)),
            np.tile([0.2, 0.3, 0.1, 0.9], (3, 1)),
        ]
    )
    hard = np.asarray([[0, 0, 0], [3, 3, 3]])
    posterior = np.argmin(risk, axis=-1)
    utilities = np.asarray(
        [[1.0, 0.6, 0.2, -0.2], [0.6, 1.0, -0.2, 0.2], [0.2, -0.2, 1.0, 0.6]]
    )
    return logits, per_seed, mass, risk, hard, posterior, utilities


def test_contextual_design_has_frozen_schema_and_is_finite() -> None:
    logits, per_seed, mass, risk, hard, posterior, utilities = synthetic_design_inputs()
    out = contextual_design(
        ensemble_logits=logits,
        per_seed_logits=per_seed,
        set_mass=mass,
        action_risk=risk,
        hard_actions=hard,
        posterior_actions=posterior,
        utilities=utilities,
    )
    assert out["design"].shape == (2, 3, len(FEATURE_NAMES))
    assert np.isfinite(out["design"]).all()
    assert out["hard_set"].any(axis=1).all()
    assert np.allclose(mass @ nonempty_sets(4).sum(axis=1), 2.1333333333333333)


def test_disagreement_weights_give_each_active_token_unit_mass() -> None:
    disagreement = np.asarray([[True, True, False], [False, False, False], [True, False, False]])
    weights = disagreement_weights(disagreement)
    assert np.allclose(weights.sum(axis=1), [1.0, 0.0, 1.0])
    assert np.array_equal(weights > 0.0, disagreement)


def test_weighted_scaler_centers_and_handles_constant_columns() -> None:
    values = np.asarray([[1.0, 5.0], [3.0, 5.0], [7.0, 5.0]])
    weights = np.asarray([0.5, 0.25, 0.25])
    scaler = fit_weighted_scaler(values, weights)
    transformed = scaler.transform(values)
    assert np.allclose(np.average(transformed, axis=0, weights=weights), 0.0)
    assert scaler.scale[1] == 1.0
    assert np.allclose(transformed[:, 1], 0.0)


def test_stratified_shuffle_preserves_weighted_target_measure() -> None:
    disagreement = np.asarray(
        [
            [True, True, False],
            [True, True, False],
            [True, False, False],
            [True, False, False],
        ]
    )
    gain = np.asarray(
        [
            [1.0, 2.0, 99.0],
            [3.0, 4.0, 98.0],
            [5.0, 97.0, 96.0],
            [7.0, 95.0, 94.0],
        ]
    )
    out = stratified_gain_shuffle(gain, disagreement, seed=56031)
    counts = disagreement.sum(axis=1)
    for policy in range(3):
        for count in np.unique(counts[disagreement[:, policy]]):
            rows = disagreement[:, policy] & (counts == count)
            assert sorted(out["target"][rows, policy]) == sorted(gain[rows, policy])
            before = np.sum(gain[rows, policy] / count)
            after = np.sum(out["target"][rows, policy] / count)
            assert after == before
    assert np.array_equal(out["target"][~disagreement], gain[~disagreement])


def test_gate_is_strict_and_hard_only_is_exact() -> None:
    scores = np.asarray([[0.5, 0.6, 0.7]])
    hard = np.asarray([[0, 0, 1]])
    posterior = np.asarray([[1, 1, 1]])
    gated = apply_gate(scores, hard, posterior, 0.5)
    assert gated["override"].tolist() == [[False, True, False]]
    identity = apply_gate(scores, hard, posterior, "hard_only")
    assert np.array_equal(identity["actions"], hard)
    assert not identity["override"].any()


def test_bootstrap_is_single_deterministic_paired_index_matrix() -> None:
    runner = load_runner()
    config = {"bootstrap": {"seed": 5607, "replicates": 7}}
    first = runner.paired_bootstrap_indices(5, config)
    second = runner.paired_bootstrap_indices(5, config)
    assert first.shape == (7, 5)
    assert np.array_equal(first, second)
    left = np.arange(5, dtype=float)
    result = runner.paired_delta_ci(left, left - 1.0, first)
    assert result["mean_diff"] == pytest.approx(1.0)
    assert result["ci95_low"] == pytest.approx(1.0)
    assert result["ci95_high"] == pytest.approx(1.0)


def test_feature_schema_is_ordered_and_complete() -> None:
    runner = load_runner()
    schema = runner.feature_schema()
    assert [row["index"] for row in schema] == list(range(len(FEATURE_NAMES)))
    assert [row["name"] for row in schema] == list(FEATURE_NAMES)
    assert all(row["dtype"] == "float64" for row in schema)
    assert all(row["domain"] and row["normalization"] for row in schema)


def test_replay_compares_feature_schema_as_well_as_arrays(tmp_path: Path) -> None:
    runner = load_runner()
    primary = tmp_path / "primary"
    replay = tmp_path / "replay"
    primary.mkdir()
    replay.mkdir()
    for name in ("analysis_core.json", "selection_freeze.json", "feature_schema.json"):
        (primary / name).write_text('{"same": true}\n', encoding="utf-8")
        (replay / name).write_text('{"same": true}\n', encoding="utf-8")
    np.savez_compressed(
        primary / "result_arrays.npz",
        values=np.asarray([0.0, np.nan, 2.0]),
        labels=np.asarray(["a", "b", "c"]),
    )
    np.savez_compressed(
        replay / "result_arrays.npz",
        values=np.asarray([0.0, np.nan, 2.0]),
        labels=np.asarray(["a", "b", "c"]),
    )
    assert runner.compare_reference(replay, primary)["all_exact"] is True
    (replay / "feature_schema.json").write_text('{"same": false}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="replay mismatch"):
        runner.compare_reference(replay, primary)


def test_source_preflight_happens_before_output_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = load_runner()
    config = tmp_path / "config.json"
    config.write_text("{}", encoding="utf-8")
    output = tmp_path / "must_not_exist"
    monkeypatch.setattr(
        runner,
        "parse_args",
        lambda: argparse.Namespace(
            wave55_dir=tmp_path / "wave55",
            wave54_dir=tmp_path / "wave54",
            wave52_dir=tmp_path / "wave52",
            output_dir=output,
            config=config,
            reference_dir=None,
            force=False,
        ),
    )
    monkeypatch.setattr(
        runner,
        "require_sources_at_head",
        lambda paths: (_ for _ in ()).throw(RuntimeError("preflight failed")),
    )
    with pytest.raises(RuntimeError, match="preflight failed"):
        runner.main()
    assert not output.exists()


def test_token_folds_are_stratified_without_policy_expansion() -> None:
    runner = load_runner()
    cards = np.tile(np.asarray([2, 3, 4]), 10)
    data = {
        "primary": np.ones(len(cards), dtype=bool),
        "pair_token": np.asarray([f"t{i}" for i in range(len(cards))]),
        "cardinality": cards,
    }
    config = {
        "folds": {"n_splits": 5, "random_state": 5601},
        "minimums": {"tokens_per_fold": 1},
    }
    folds = runner.make_folds(data, config)
    assert set(folds.tolist()) == set(range(5))
    for fold in range(5):
        assert set(cards[folds == fold].tolist()) == {2, 3, 4}


def test_policy_groups_must_be_a_disjoint_three_by_eight_partition(tmp_path: Path) -> None:
    runner = load_runner()
    payload = {
        "levels": [1.0, 0.6, 0.2, -0.2],
        "rank_permutations": list(itertools.permutations(range(4))),
        "groups": [list(range(8)), list(range(8, 16)), list(range(15, 23))],
    }
    path = tmp_path / "policy_manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="disjoint 3x8 partition"):
        runner.load_utilities(path)


def test_strict_json_rejects_nonfinite_metrics(tmp_path: Path) -> None:
    runner = load_runner()
    path = tmp_path / "bad.json"
    with pytest.raises(ValueError, match="non-finite JSON value"):
        runner.write_json_strict(path, {"metric": [1.0, float("nan")]})
    assert not path.exists()
