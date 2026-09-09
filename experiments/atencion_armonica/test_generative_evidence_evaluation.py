"""Arithmetic tests of readout/estimands, not new campaign observations."""
from copy import deepcopy

import numpy as np
import pytest

from experiments.atencion_armonica.test_generative_evidence import fixture
from src.atencion_armonica import generative_evidence as ge
from src.atencion_armonica import generative_evidence_evaluation as ev
from src.atencion_armonica.generative_evidence_cache import SPLITS
from src.atencion_armonica.generative_evidence_supervision import candidate_supervision


def calibration():
    identities = [f"{i:064x}" for i in range(512)]
    records = []
    for arm in ge.ARMS:
        for cp in ge.CHECKPOINTS:
            for seed in ge.READER_SEEDS:
                for epoch in ev.EPOCHS:
                    v = 0.
                    if arm == "local":
                        if epoch == 5 and cp == ge.CHECKPOINTS[0] and seed == ge.READER_SEEDS[0]:
                            v = 1.
                        elif epoch == 10:
                            v = .2
                    elif arm == "generative" and epoch in (15, 20):
                        v = .3
                    records.append({"arm": arm, "checkpoint_seed": cp, "reader_seed": seed, "epoch": epoch,
                        "split": "calibration", "split_seed": SPLITS["calibration"][1],
                        "identities": identities.copy(), "ari": [v if i in (0, 9, 17, 511) else None for i in range(512)]})
    return records


def test_readout_canonical_tie_and_empty_denominator():
    ps = fixture()[4]
    metrics = candidate_supervision(ps, np.repeat(np.arange(2), 4))["metrics"]
    partitions, ms = [[] for _ in range(512)], [[] for _ in range(512)]
    for i in (0, 511):
        partitions[i], ms[i] = ps, metrics
    pred = np.array([[.2, .3], [.1, .4], [.8, .7], [.1, .4]], np.float32)
    offsets = np.r_[np.int64(0), np.cumsum([len(p) for p in partitions], dtype=np.int64)]
    result = ev.chosen_metrics(partitions, ms, pred, offsets)
    assert result["decisions"][0]["candidate_index"] == 0
    assert result["decisions"][0]["co_minimum_count"] == 2
    assert result["decisions"][511]["candidate_index"] == 1
    assert np.flatnonzero(result["eligible"]).tolist() == [0, 511]
    assert result["decisions"][9] is None and np.isnan(result["metrics"][9]).all()
    np.testing.assert_array_equal(result["metrics"][511], [metrics[1][m] for m in ev.METRICS])
    with pytest.raises(ValueError):
        ev.chosen_metrics(partitions, ms, pred.astype(np.float64), offsets)
    wrong = offsets.copy()
    wrong[2] += 1
    with pytest.raises(ValueError):
        ev.chosen_metrics(partitions, ms, pred, wrong)


def test_selection_270_records_scene_average_not_best_cell_and_earliest_tie():
    result = ev.select_epochs(calibration())
    assert result["count"] == 512 and result["eligible_scene_ids"] == [0, 9, 17, 511]
    assert len(result["excluded_scene_ids"]) == 508
    assert {a: v["epoch"] for a, v in result["selected"].items()} == {"local": 10, "generative": 15, "decoupled": 5}
    assert result["selected"]["local"]["mean_ari_by_epoch"][0] == pytest.approx(1/9)
    assert result["selected"]["local"]["cell_mean_ari_by_epoch"][0][0][0] == 1.


@pytest.mark.parametrize("kind", ["missing", "duplicate", "identity", "mask", "nan", "test_role", "empty"])
def test_selection_rejects_incomplete_or_changed_support(kind):
    records = calibration()
    if kind == "missing":
        records.pop()
    elif kind == "duplicate":
        records.append(deepcopy(records[0]))
    elif kind == "identity":
        records[0]["identities"].reverse()
    elif kind == "mask":
        records[0]["ari"][0] = None
    elif kind == "nan":
        records[0]["ari"][0] = float("nan")
    elif kind == "test_role":
        records[0]["split"] = "iid"
    else:
        for row in records:
            row["ari"] = [None]*512
    with pytest.raises(ValueError):
        ev.select_epochs(records)


def test_test_estimand_exact_scene_pairs_and_primary_intervals():
    mask = np.zeros(512, bool)
    mask[[1, 99, 511]] = True
    values = {a: np.full((512, 3, 3, len(ev.METRICS)), np.nan, np.float64) for a in ge.ARMS}
    for a in ge.ARMS:
        values[a][mask] = 0.
    # Heterogeneous scene differences; 9 seeds are cells, not bootstrap units.
    for i, delta in zip(np.flatnonzero(mask), (-.3, .1, .8)):
        values["generative"][i, :, :, 0] = delta
    values["local"][1, 0, 0, 0] = .9
    result = ev.summarize_learned(values, split="deformed_family", eligible=mask)
    indices = np.random.default_rng(np.random.SeedSequence([2026090994, 2026090985])).integers(0, 3, (2000, 3), dtype=np.int64)
    np.testing.assert_array_equal(result["bootstrap_indices"], indices)
    delta = np.array([-.4, .1, .8])
    expected = np.percentile(delta[indices].mean(axis=1), [1.25, 98.75], method="linear")
    actual = result["contrasts"]["generative-minus-local"]["ari"]
    assert actual["mean"] == pytest.approx(delta.mean())
    np.testing.assert_allclose(actual["interval"], expected, rtol=0, atol=1e-15)
    assert actual["nominal_percent"] == 97.5
    assert result["contrasts"]["generative-minus-local"]["vi"]["nominal_percent"] == 95
    assert result["output_count"] == 3 and result["coverage"] == 3/512
    assert result["cell_means"]["local"][0][0][0] == pytest.approx(.3)
    iid = ev.summarize_learned(values, split="iid", eligible=mask)
    assert iid["contrasts"]["generative-minus-local"]["ari"]["nominal_percent"] == 95


@pytest.mark.parametrize("count", [0, 1])
def test_no_output_null_and_single_scene_degenerate(count):
    mask = np.arange(512) < count
    values = {a: np.full((512, 3, 3, len(ev.METRICS)), np.nan, np.float64) for a in ge.ARMS}
    for v in values.values():
        v[mask] = .25
    result = ev.summarize_learned(values, split="deformed_family", eligible=mask)
    stat = result["contrasts"]["generative-minus-local"]["ari"]
    if count:
        assert stat["mean"] == 0. and stat["interval"] == [0., 0.]
        assert result["interval_status"] == "DEGENERATE_SINGLE_SCENE"
    else:
        assert stat["mean"] is None and stat["interval"] is None
        assert result["cell_means"]["local"] is None
        assert result["bootstrap_indices"].shape == (2000, 0)


def test_imputation_and_bad_metric_shape_rejected():
    mask = np.zeros(512, bool)
    values = {a: np.zeros((512, 3, 3, len(ev.METRICS)), np.float64) for a in ge.ARMS}
    with pytest.raises(ValueError):
        ev.summarize_learned(values, split="iid", eligible=mask)
    with pytest.raises(ValueError):
        ev.bootstrap_indices("train", mask)
