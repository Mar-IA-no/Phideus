"""Abstract partition fixtures; no observational q generation or test access."""
import numpy as np
import pytest

from src.atencion_armonica.geometric_decision_reporting import event_universe, roundtrip_comparison, coordinate_comparison


def test_partitions_correspond_by_events_not_rank_or_group_order():
    first = event_universe([[[0, 1], [2, 3]]], [2, 0, 3, 1])
    second = event_universe([[[1, 0], [3, 2]]], [0, 2, 1, 3])
    assert first == second == [((0, 2), (1, 3))]
    with pytest.raises(ValueError, match="bijection"):
        event_universe([], [0, 0, 1, 3])
    with pytest.raises(ValueError, match="cover"):
        event_universe([[[0, 1], [1, 3]]], [0, 1, 2, 3])


def test_roundtrip_does_not_match_different_candidates_by_array_position():
    before = {"partitions": [[[0, 1], [2, 3]], [[0, 2], [1, 3]]], "canonical_to_observed": [0, 1, 2, 3]}
    after = {"partitions": [[[0, 3], [1, 2]], [[0, 1], [2, 3]]], "canonical_to_observed": [0, 1, 2, 3]}
    a = np.zeros((2, 8), np.float32)
    b = np.stack([np.ones(8, np.float32), np.zeros(8, np.float32)])
    result = roundtrip_comparison(before, after, energy_before=np.array([1., 9.]), energy_after=np.array([7., 1.]),
        evidence_before=a, evidence_after=b, choice_before=0, choice_after=1)
    assert result["common_candidates"] == 1
    assert result["before_indices"] == [0] and result["after_indices"] == [1]
    assert result["max_energy_error_common"] == 0 and result["max_channel_error_common"] == [0.]*8
    assert len(result["lost_candidates"]) == len(result["added_candidates"]) == 1
    assert result["same_exact_choice_by_event"] is True


def test_empty_probe_keeps_support_loss_and_undefined_choice():
    before = {"partitions": [[[0, 1], [2, 3]]], "canonical_to_observed": [0, 1, 2, 3]}
    after = {"partitions": [], "canonical_to_observed": [0, 1, 2, 3]}
    result = roundtrip_comparison(before, after, energy_before=np.array([1.]), energy_after=np.empty(0),
        evidence_before=np.zeros((1, 8), np.float32), evidence_after=np.empty((0, 8), np.float32),
        choice_before=0, choice_after=None)
    assert result["after_empty"] and result["max_energy_error_common"] is None
    assert result["same_exact_choice_by_event"] is None and len(result["lost_candidates"]) == 1


def test_coordinate_statistics_on_abstract_two_point_vectors():
    # Two points are outside the observational domain (8..32 events): these
    # are arithmetic fixtures, not new scene coordinates or sampler outputs.
    values = {"original": np.array([0., 1.], np.float32),
        "shifted64": np.array([.25, 1.25], np.float64),
        "shifted32": np.array([.25, 1.25], np.float32),
        "q_center": np.array([-.5, .5], np.float32), "q_probe": np.array([-.5, .5], np.float32)}
    report = coordinate_comparison(values)
    assert set(report["max_pairwise_log_ratio_change"].values()) == {0.}
    assert set(report["tie_counts"].values()) == {0}
    values["q_probe"] = np.array([0., 0.], np.float32)
    report = coordinate_comparison(values)
    assert report["tie_counts"]["q_probe"] == 1
    assert report["max_pairwise_log_ratio_change"]["original_to_q_probe"] == 1.
