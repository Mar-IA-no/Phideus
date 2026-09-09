"""CPU mechanical fixtures, not observations from the confirmatory campaign."""
from copy import deepcopy
from itertools import combinations

import numpy as np
import pytest
import torch

from src.atencion_armonica import generative_evidence as ge
from src.atencion_armonica import observable_source_rivals as law
from src.atencion_armonica.generative_evidence_model import EvidenceHead, collate, partition_cost_loss
from src.atencion_armonica.learned_partition_core import partition_errors
from src.atencion_armonica.learned_partition_readout import choose_costs


def fixture(n=8):
    """Explicit arithmetic q32 fixture; bounds are stubbed, not claimed as fits."""
    q = np.arange(n, dtype=np.float32)/8
    z = (q[:, None]+q[None, :]).astype(np.float32)
    triples = np.asarray(list(combinations(range(n), 3)), np.int64)
    residual = q[triples].sum(1).astype(np.float64)
    first = tuple(tuple(range(i, i+4)) for i in range(0, n, 4))
    second = [list(g) for g in first]
    second[0][-1], second[1][0] = second[1][0], second[0][-1]
    ps = sorted({law.signature(first), law.signature(second)})
    fits = [{"partition": p, "status": "FITTED", "branches": {
        b: {"LB": float(i+j+1), "UB": float(i+j+2)}
        for j, b in enumerate(ge.BRANCHES) if len(p) in law.BRANCHES[b][2]}}
        for i, p in enumerate(ps)]
    return q, z, triples, residual, ps, fits


def norms(row):
    rows = lambda: iter([row])
    return ge.fit_common_normalizer(rows), ge.fit_evidence_normalizer(rows)


def test_joint_channel_and_common_metadata():
    row = ge.observable_rows(*fixture())
    assert row["globals"].shape == (2, 17)
    assert row["evidence"].shape == (2, 6)
    np.testing.assert_array_equal(row["globals"][:, 6:11], [[1, 0, 0, 0, 0]]*2)
    np.testing.assert_array_equal(row["globals"][:, 11:14], np.ones((2, 3)))
    np.testing.assert_array_equal(row["globals"][:, 14:], [[3/8, 3/8, 5/8]]*2)
    np.testing.assert_array_equal(row["evidence"][0], np.log1p(np.array([1, 2, 2, 3, 3, 4])/8))
    np.testing.assert_allclose(row["incidence"].sum(1), 1, rtol=0, atol=2e-7)


def test_permutation_remaps_logits_and_triples():
    q, z, t, r, ps, fits = fixture()
    original = ge.observable_rows(q, z, t, r, ps, fits)
    p = np.array([7, 2, 0, 5, 3, 1, 6, 4])
    qp = q[p]
    changed = ge.observable_rows(qp, z[np.ix_(p, p)], t, qp[t].sum(1).astype(np.float64), ps, fits)
    for key in ("groups", "globals", "incidence", "evidence", "available"):
        np.testing.assert_array_equal(original[key], changed[key])


def test_exact_q32_gauge_shift_of_preserved_observables():
    q, z, t, r, ps, fits = fixture()
    shifted = q+np.float32(2.)
    assert np.array_equal(shifted.astype(np.float64)-q.astype(np.float64), np.full(8, 2.))
    # Conditional row invariance with preserved relative features/fits. This is
    # not a test of the observation producer, backbone, or grid fitter.
    original = ge.observable_rows(q, z, t, r, ps, fits)
    changed = ge.observable_rows(shifted, z, t, r, ps, fits)
    for key in ("groups", "globals", "incidence", "evidence", "available", "canonical_to_observed"):
        np.testing.assert_array_equal(original[key], changed[key])


def test_q32_ties_report_stable_order_not_event_identity():
    _, z, t, r, ps, fits = fixture()
    q = np.array([.5, 0., .5, .25, .125, .75, .625, .875], np.float32)
    row = ge.observable_rows(q, z, t, r, ps, fits)
    assert row["q_tie_count"] == 1
    assert row["canonical_to_observed"].tolist() == [1, 4, 3, 0, 2, 6, 5, 7]
    # Swapping the two equal-frequency events preserves q but not their order
    # identity in the delivered array. Stable sort must not manufacture it.
    p = np.array([2, 1, 0, 3, 4, 5, 6, 7])
    residual_lookup = {tuple(triple): value for triple, value in zip(t, r)}
    rp = np.asarray([residual_lookup[tuple(sorted(p[triple]))] for triple in t], np.float64)
    changed = ge.observable_rows(q[p], z[np.ix_(p, p)], t, rp, ps, fits)
    assert changed["q_tie_count"] == 1
    assert changed["canonical_to_observed"].tolist() == row["canonical_to_observed"].tolist()
    assert not np.array_equal(p[changed["canonical_to_observed"]], row["canonical_to_observed"])


def test_branch_mask_after_normalization_and_empty_output():
    q, z, t, r, ps, fits = fixture(16)
    row = ge.observable_rows(q, z, t, r, ps, fits)
    assert row["available"].tolist() == [[True, False, False]]*2
    # Four-source-only TRAIN cannot fit coordinates for absent branches.
    with pytest.raises(ValueError, match="no training support"):
        ge.fit_evidence_normalizer(lambda: iter([row]))
    common = ge.fit_common_normalizer(lambda: iter([row]))
    em = {"mean": [2.]*6, "scale": [1.]*6}
    inputs = ge.model_inputs(row, common, em, "generative", split_seed=10, scene_id=0)
    assert np.all(inputs["evidence"][:, 2:] == 0)
    empty = ge.observable_rows(q, z, t, r, [], [])
    assert empty["groups"].shape == (0, 9) and empty["evidence"].shape == (0, 6)
    assert ge.model_inputs(empty, common, em, "local", split_seed=10, scene_id=0)["globals"].shape == (0, 17)
    with pytest.raises(ValueError, match="ragged"):
        collate([ge.model_inputs(empty, common, em, "local", split_seed=10, scene_id=0)])


def test_normalizer_equal_scene_mass_not_candidate_mass():
    a = {"groups": np.zeros((1, 9)), "globals": np.zeros((1, 17)),
         "evidence": np.zeros((1, 6)), "available": np.ones((1, 3), bool)}
    b = {"groups": np.full((5, 9), 10.), "globals": np.full((3, 17), 10.),
         "evidence": np.full((3, 6), 10.), "available": np.ones((3, 3), bool)}
    for norm in (ge.fit_common_normalizer(lambda: iter([a, b])),
                 ge.fit_evidence_normalizer(lambda: iter([a, b]))):
        assert norm["mean"] == [5.]*len(norm["mean"])
        assert norm["scale"] == [5.]*len(norm["scale"])
        assert norm["scene_count"] == [2]*len(norm["mean"])


def test_sham_joint_multiset_and_common_inputs():
    row = ge.observable_rows(*fixture())
    cn, en = norms(row)
    inputs = {arm: ge.model_inputs(row, cn, en, arm, split_seed=10, scene_id=0) for arm in ge.ARMS}
    for key in ("groups", "globals", "incidence"):
        for arm in ge.ARMS:
            np.testing.assert_array_equal(inputs[arm][key], inputs["local"][key])
    assert np.all(inputs["local"]["evidence"] == 0)
    np.testing.assert_array_equal(inputs["decoupled"]["evidence"], inputs["generative"]["evidence"][::-1])
    changed, receipt = ge.decouple(row["partitions"], inputs["generative"]["evidence"], split_seed=10, scene_id=0)
    assert receipt["changed_mask"] == [True, True]
    np.testing.assert_array_equal(changed, inputs["decoupled"]["evidence"])
    _, same = ge.decouple(row["partitions"], np.zeros((2, 6), np.float32), split_seed=10, scene_id=0)
    assert same["status"] == "INPUT_UNCHANGED"
    _, singleton = ge.decouple(row["partitions"][:1], np.ones((1, 6), np.float32), split_seed=10, scene_id=0)
    assert singleton["strata"][0]["status"] == "NO_PERMUTATION"


def test_one_topology_paired_initialization_without_global_rng_effect():
    before = torch.random.get_rng_state().clone()
    a, b = EvidenceHead(ge.READER_SEEDS[0]), EvidenceHead(ge.READER_SEEDS[0])
    assert torch.equal(before, torch.random.get_rng_state())
    assert sum(p.numel() for p in a.parameters()) == 2194
    assert all(torch.equal(v, b.state_dict()[k]) for k, v in a.state_dict().items())
    row = ge.observable_rows(*fixture())
    cn, en = norms(row)
    batch = collate([ge.model_inputs(row, cn, en, "generative", split_seed=10, scene_id=0)])
    assert torch.equal(a(batch), b(batch))


def test_partial_batch_padding_loss_and_supervision_firewall():
    row = ge.observable_rows(*fixture())
    cn, en = norms(row)
    inp = ge.model_inputs(row, cn, en, "generative", split_seed=10, scene_id=0)
    smaller = {k: v.copy() for k, v in inp.items()}
    for key in ("globals", "incidence", "evidence"):
        smaller[key] = smaller[key][:1]
    batch = collate([inp, smaller])
    model = EvidenceHead(ge.READER_SEEDS[0])
    prediction = model(batch)
    target = torch.zeros_like(prediction)
    loss = partition_cost_loss(prediction, target, batch["candidate_mask"])
    # Match the specified component -> candidate -> scene reductions. Flattening
    # the first scene changes float32 addition order, not the loss estimand.
    component_means = prediction.square().mean(dim=-1)
    expected = (component_means[0].mean()+component_means[1, 0])/2
    assert torch.equal(loss, expected)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    polluted = {**batch, "source_ids": torch.zeros(2, 8)}
    with pytest.raises(ValueError, match="supervision"):
        model(polluted)
    broken = {k: v.clone() for k, v in batch.items()}
    broken["evidence"][1, 1, 0] = 1
    with pytest.raises(ValueError, match="padding"):
        model(broken)


def test_unchanged_supervised_target_and_readout():
    row = ge.observable_rows(*fixture())
    y = np.repeat(np.arange(2), 4)
    costs = np.stack([partition_errors(p, y)["normalized"] for p in row["partitions"]]).astype(np.float32)
    choice = choose_costs(costs, row["partitions"])
    assert choice["cost"] == 0.
    assert choice["signature"] == law.signature([list(range(4)), list(range(4, 8))])


def test_sham_never_crosses_size_strata_and_is_repeatable():
    ps = sorted(law.signature(p) for p in [
        [range(4), range(4, 12)],
        [[0, 1, 2, 4], [3, 5, 6, 7, 8, 9, 10, 11]],
        [range(6), range(6, 12)],
        [[0, 1, 2, 3, 4, 6], [5, 7, 8, 9, 10, 11]],
        [range(4), range(4, 8), range(8, 12)],
    ])
    values = np.arange(30, dtype=np.float32).reshape(5, 6)
    a, receipt = ge.decouple(ps, values, split_seed=10, scene_id=3)
    b, repeated = ge.decouple(ps, values, split_seed=10, scene_id=3)
    np.testing.assert_array_equal(a, b)
    assert receipt == repeated
    for i, donor in enumerate(receipt["donors"]):
        assert sorted(map(len, ps[i])) == sorted(map(len, ps[donor]))
        assert (donor == i) == (len(ps[i]) == 3)
        np.testing.assert_array_equal(a[i], values[donor])
    assert sorted(map(tuple, a)) == sorted(map(tuple, values))


def test_head_group_and_candidate_permutation_equivariance():
    row = ge.observable_rows(*fixture())
    cn, en = norms(row)
    inp = ge.model_inputs(row, cn, en, "generative", split_seed=10, scene_id=0)
    gp, cp = np.arange(len(inp["groups"]))[::-1], np.array([1, 0])
    permuted = {"groups": inp["groups"][gp].copy(),
                "globals": inp["globals"][cp].copy(),
                "evidence": inp["evidence"][cp].copy(),
                "incidence": inp["incidence"][np.ix_(cp, gp)].copy()}
    model = EvidenceHead(ge.READER_SEEDS[0])
    # Matrix reductions may differ at roundoff when groups are reordered.
    torch.testing.assert_close(model(collate([permuted])),
                               model(collate([inp]))[:, cp], rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("corruption", ["bounds", "branch", "q", "triples", "duplicate"])
def test_invalid_observable_inputs_rejected(corruption):
    q, z, t, r, ps, fits = deepcopy(fixture())
    if corruption == "bounds":
        fits[0]["branches"]["base-low"]["LB"] = float("nan")
    elif corruption == "branch":
        fits[0]["branches"].pop("deformed-low")
    elif corruption == "q":
        q = q.astype(np.float64)
    elif corruption == "triples":
        t = t[::-1]
    else:
        ps = [ps[0], ps[0]]
    with pytest.raises(ValueError):
        ge.observable_rows(q, z, t, r, ps, fits)
