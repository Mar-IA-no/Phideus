import pytest

from src.atencion_armonica.geometric_decision_work import training_work, head_forecast


def test_actual_roster_work_includes_partial_batch_and_snapshot_union():
    w = training_work(4036, 512)
    assert w == {"cells": 72, "epochs_per_cell": 50, "updates_per_epoch": 127,
        "last_batch_scenes": 4, "updates": 457200, "snapshots": 17856,
        "warmup_updates": 360, "steady_updates": 456840,
        "noninitial_snapshots": 17784, "calibration_batches": 12672, "checkpoint_corpus_loads": 3}


def test_forecast_is_not_reduced_to_forward_or_steady_update_cost():
    cases = {(kind, objective): {"status": "MECHANICAL_PROFILE_NOT_TRAINED_CELL", "objective": objective,
        "device": "cpu", "all_update_seconds": [2.]+[1.]*24, "snapshot_io_seconds": [3., 4.],
        "evaluation_batch_io_seconds": [1., 2., 3.], "setup_seconds": 5., "exact_recovery_digest": "fixture"}
        for kind in ("envelope", "first_train_batch") for objective in ("mse", "decision")}
    report = head_forecast(cases, eligible_train=4036, eligible_calibration=512, corpus_load_seconds=7.)
    assert report["measured_unit_seconds"]["warmup_updates"] == 6/5
    assert report["measured_unit_seconds"]["steady_updates"] == 1.
    assert report["projected_cost_components"]["checkpoint_corpus_loads"] == 21.
    assert report["projected_seconds_before_margin"] == sum(report["projected_cost_components"].values())
    with pytest.raises(ValueError, match="measured full-checkpoint"):
        head_forecast(cases, eligible_train=4036, eligible_calibration=512, corpus_load_seconds=0.)
    cases["envelope", "mse"]["device"] = "cuda:0"
    with pytest.raises(ValueError, match="one backend"):
        head_forecast(cases, eligible_train=4036, eligible_calibration=512, corpus_load_seconds=7.)
