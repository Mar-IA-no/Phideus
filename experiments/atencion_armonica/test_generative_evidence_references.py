"""Arithmetic tests for system references, without fits, forwards or files."""
from copy import deepcopy

import numpy as np
import pytest
from scipy.special import expit

from experiments.atencion_armonica.test_generative_evidence import fixture
from src.atencion_armonica import generative_evidence as ge
from src.atencion_armonica import generative_evidence_references as r
from src.atencion_armonica.learned_partition_metrics import METRICS
from src.atencion_armonica.partial_compatibility_evaluation import read_partition


def case():
    q, z, _, _, ps, fits = fixture()
    logits = {cp: z.copy() for cp in ge.CHECKPOINTS}
    inventory = {"candidates": [{"partition": p, "origin": "pool", "status": "SUPPORTED"} for p in ps]}
    return q, logits, ps, fits, inventory


def test_system_minimum_and_canonical_ties_do_not_select_by_labels():
    q, z, ps, fits, inventory = case()
    for f in fits:
        for b in f["branches"].values():
            b.update(LB=0., UB=2.)
    fits[1]["branches"]["deformed-low"]["UB"] = 1.
    choices = r.observable_choices(ps, fits, z, np.argsort(q))
    assert choices["base"]["candidate_index"] == 0
    assert choices["extended"]["candidate_index"] == 1
    assert choices["extended"]["branch"] == "deformed-low"
    assert set(choices["historical"]) == {str(c) for c in ge.CHECKPOINTS}
    result = r.evaluate_references(ps, inventory, choices, np.repeat(np.arange(2), 4))
    assert all(set(m) == set(METRICS) for m in result["candidate_metrics"])
    assert result["coverage"]["planted"] == "pool"
    assert result["oracle"]["maximum_ari"] == 1.
    assert result["oracle"]["minimum_vi"] == 0.
    assert result["references"]["base"]["metrics"] == result["candidate_metrics"][0]


def test_historical_mapping_is_in_canonical_coordinates():
    q, z, ps, fits, _ = case()
    permutation = np.array([7, 0, 6, 1, 5, 2, 4, 3])
    reordered = {cp: v[np.ix_(permutation, permutation)] for cp, v in z.items()}
    order = np.argsort(q[permutation])
    choices = r.observable_choices(ps, fits, reordered, order)
    inverse = np.argsort(order)
    for cp, threshold in r.THRESHOLDS.items():
        partition = read_partition(expit(reordered[cp].astype(np.float64)), threshold)
        expected = ge.law.signature([inverse[list(g)].tolist() for g in partition])
        assert choices["historical"][str(cp)] == {"partition": expected, "threshold": threshold}


def test_empty_candidates_keep_historical_and_null_oracle():
    q, z, _, _, _ = case()
    choices = r.observable_choices([], [], z, np.argsort(q))
    result = r.evaluate_references([], {"candidates": []}, choices, np.repeat(np.arange(2), 4))
    assert result["references"]["base"] is None and result["references"]["extended"] is None
    assert result["oracle"]["maximum_ari"] is None and result["oracle"]["minimum_vi"] is None
    assert result["raw_entropies"].shape == (0, 2)
    assert result["coverage"] == {"has_output": False, "candidate_count": 0, "planted": "absent"}
    assert len(result["references"]["historical"]) == 3
    assert all(set(x["metrics"]) == set(METRICS) for x in result["references"]["historical"].values())


def test_reject_inconsistent_universe_and_neural_roster():
    q, z, ps, fits, inventory = case()
    choices = r.observable_choices(ps, fits, z, np.argsort(q))
    with pytest.raises(ValueError, match="checkpoint"):
        r.observable_choices(ps, fits, {ge.CHECKPOINTS[0]: z[ge.CHECKPOINTS[0]]}, np.argsort(q))
    with pytest.raises(ValueError, match="inventory"):
        r.evaluate_references(ps, {"candidates": []}, choices, np.repeat(np.arange(2), 4))
    changed = deepcopy(choices)
    changed["historical"][str(ge.CHECKPOINTS[0])]["threshold"] = .5
    with pytest.raises(ValueError, match="threshold"):
        r.evaluate_references(ps, inventory, changed, np.repeat(np.arange(2), 4))


def test_historical_not_constrained_to_groups_of_at_least_four():
    q, z, ps, fits, inventory = case()
    # Very negative equal logits leave singleton clusters at all frozen cuts.
    z = {cp: np.full_like(v, -20) for cp, v in z.items()}
    choices = r.observable_choices(ps, fits, z, np.argsort(q))
    result = r.evaluate_references(ps, inventory, choices, np.repeat(np.arange(2), 4))
    assert all(m["sub3_member_fraction"] == 0 for m in result["candidate_metrics"])
    assert all(x["metrics"]["sub3_member_fraction"] == 1 for x in result["references"]["historical"].values())
