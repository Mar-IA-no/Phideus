"""Arithmetic fixtures only; no sampler, corpus, reserved test seed or CUDA."""
import numpy as np
import pytest
import torch

from experiments.atencion_armonica.test_generative_evidence import fixture
from src.atencion_armonica import generative_evidence as ge
from src.atencion_armonica import geometric_decision_core as core
from src.atencion_armonica.geometric_decision_model import (
    GeometricDecisionHead, collate, component_mse, decision_loss, decision_losses,
)


def interface(n=8):
    raw = ge.observable_rows(*fixture(n))
    cn = ge.fit_common_normalizer(lambda: iter([raw]))
    en = {"mean": [0.]*6, "scale": [1.]*6}
    delivered = ge.model_inputs(raw, cn, en, "generative", split_seed=11, scene_id=0)
    inputs, record = core.delivered_interface(delivered, raw, scale=2., split_seed=11, scene_id=0)
    return inputs, raw, record


def tensors(energies, costs):
    energy = np.asarray(energies, np.float64)
    cost = np.asarray(costs, np.float32)
    h = torch.tensor(np.stack((energy/2, energy/2), -1), dtype=torch.float64, requires_grad=True)
    target = torch.tensor(np.stack((cost/2, cost/2), -1), dtype=torch.float32)
    mask = torch.ones(energy.shape, dtype=torch.bool)
    return h, target, mask


def test_available_minimum_and_scene_weighted_scale():
    evidence = np.array([[0., 4., 0., 0., 0., 0.], [0., 3., 0., 2., 0., 0.]], np.float64)
    available = np.array([[1, 0, 0], [1, 1, 0]], bool)
    np.testing.assert_array_equal(core.geometric_log_cost(evidence, available), [4., 2.])
    scale = core.fit_scale([np.array([2.]), np.array([4.]*7), np.array([], np.float64)])
    assert scale == {"scale": np.sqrt(10.), "zero_rms": False, "eligible_scenes": 2, "empty_scenes": 1}
    assert core.fit_scale([np.zeros(3)]) == {"scale": 1., "zero_rms": True, "eligible_scenes": 1, "empty_scenes": 0}
    with pytest.raises(ValueError):
        core.fit_scale([np.array([], np.float64)])
    with pytest.raises(ValueError):
        core.geometric_log_cost(evidence, np.zeros((2, 3), bool))
    with pytest.raises(ValueError):
        core.geometric_log_cost(evidence.astype(np.float32), available)
    assert core.geometric_log_cost(np.zeros((0, 6)), np.zeros((0, 3), bool)).shape == (0,)


def test_equal_information_and_scalar_donor():
    inputs, raw, record = interface()
    donor = record["six_channel_sham"]["donors"]
    expected = (core.geometric_log_cost(raw["evidence"], raw["available"])/2).astype(np.float32)
    assert inputs["evidence"].shape == (2, 8)
    np.testing.assert_array_equal(inputs["evidence"][:, 6], expected)
    np.testing.assert_array_equal(inputs["evidence"][:, 7], expected[donor])
    for route in core.ROUTES[:3]:
        routed = core.route_inputs(inputs, route)
        for k in inputs:
            assert routed[k].dtype == np.float32
            assert routed[k].tobytes() == inputs[k].tobytes()
    local = core.route_inputs(inputs, "local")
    assert not local["evidence"].any()
    assert inputs["evidence"].any()
    for k in ("groups", "globals", "incidence"):
        np.testing.assert_array_equal(local[k], inputs[k])


def test_singleton_donor_strata_preserve_scalar_exactly():
    q, z, triples, residual, _, _ = fixture(12)
    partitions = sorted([ge.law.signature([list(range(4)), list(range(4, 12))]),
                         ge.law.signature([list(range(6)), list(range(6, 12))])])
    fits = [{"partition": p, "status": "FITTED", "branches":
             {b: {"LB": float(i+1), "UB": float(i+2)} for b in ge.BRANCHES}}
            for i, p in enumerate(partitions)]
    raw = ge.observable_rows(q, z, triples, residual, partitions, fits)
    cn = ge.fit_common_normalizer(lambda: iter([raw]))
    en = {"mean": [0.]*6, "scale": [1.]*6}
    delivered = ge.model_inputs(raw, cn, en, "generative", split_seed=11, scene_id=1)
    inputs, record = core.delivered_interface(delivered, raw, scale=2., split_seed=11, scene_id=1)
    sham = record["six_channel_sham"]
    assert sham["donors"] == [0, 1]
    assert all(r["shift"] is None and r["status"] == "NO_PERMUTATION" for r in sham["strata"])
    assert record["scalar_changed_mask"] == [False, False]
    assert inputs["evidence"][:, 6].tobytes() == inputs["evidence"][:, 7].tobytes()


def test_empty_interface_is_preserved_but_not_collated():
    q, z, triples, residual, _, _ = fixture()
    empty = ge.observable_rows(q, z, triples, residual, [], [])
    cn = {"mean": [0.]*5, "scale": [1.]*5}
    en = {"mean": [0.]*6, "scale": [1.]*6}
    delivered = ge.model_inputs(empty, cn, en, "generative", split_seed=11, scene_id=2)
    inputs, record = core.delivered_interface(delivered, empty, scale=2., split_seed=11, scene_id=2)
    assert inputs["evidence"].shape == (0, 8)
    assert record["scalar_changed_mask"] == []
    assert record["six_channel_sham"]["donors"] == []
    with pytest.raises(ValueError, match="eligible"):
        collate([inputs])


@pytest.mark.parametrize("route", core.ROUTES)
def test_common_initial_state_and_exact_bypass(route):
    inputs, _, _ = interface()
    rng_before = torch.random.get_rng_state()
    model = GeometricDecisionHead(core.READER_SEEDS[0], route)
    assert torch.equal(rng_before, torch.random.get_rng_state())
    reference = GeometricDecisionHead(core.READER_SEEDS[0], "injection")
    assert sum(p.numel() for p in model.parameters()) == 2258
    for k, p in reference.state_dict().items():
        assert torch.equal(p, model.state_dict()[k])
    h = model(collate([core.route_inputs(inputs, route)]))
    expected = inputs["evidence"][:, 6 if route == "geometric" else 7].astype(np.float64)
    if route in ("injection", "local"):
        expected = np.zeros(2)
    assert h.dtype == torch.float64
    np.testing.assert_array_equal(h.detach().numpy().sum(-1)[0], expected)


def test_signed_output_and_local_guard():
    inputs, _, _ = interface()
    model = GeometricDecisionHead(core.READER_SEEDS[0], "local")
    with pytest.raises(ValueError, match="Local"):
        model(collate([inputs]))
    with torch.no_grad():
        model.partition2.bias.copy_(torch.tensor([-3., 1.]))
    h = model(collate([core.route_inputs(inputs, "local")]))
    assert h[0, 0].tolist() == [-3., 1.]
    target = torch.zeros_like(h, dtype=torch.float32)
    assert component_mse(h, target, torch.ones((1, 2), dtype=torch.bool)).item() == 5.


def test_ties_follow_signature_not_position():
    _, raw, _ = interface()
    ps = raw["partitions"]
    assert core.choose_energy(np.array([0., 0.]), ps) == 0
    assert core.choose_energy(np.array([0., 0.]), ps[::-1]) == 1
    assert core.choose_energy(np.array([1e-16, 0.]), ps) == 1
    assert core.choose_energy(np.empty(0), []) is None


@pytest.mark.parametrize("energies,costs", [
    ([[1.]], [[.5]]),
    ([[0., 0.]], [[0., 1.]]),
    ([[-2., 1., 3.]], [[0., 0., .5]]),
    ([[2., -1., 1.]], [[0., 0., 0.]]),
])
def test_surrogate_cooptimal_and_bound(energies, costs):
    h, target, mask = tensors(energies, costs)
    losses = decision_losses(h, target, mask)
    e, t = h.detach().numpy().sum(-1), target.numpy().astype(np.float64).sum(-1)
    for i in range(len(e)):
        oracle = t[i] == min(t[i])
        expected = max(t[i]-min(t[i])+np.mean(e[i, oracle])-e[i])
        assert losses[i].item() == pytest.approx(expected, abs=1e-12, rel=0)
        selected_regrets = t[i, e[i] == min(e[i])]-min(t[i])
        assert losses[i].item()+1e-12 >= max(selected_regrets)
    losses.mean().backward()
    assert torch.isfinite(h.grad).all()
    if len(e[0]) == 1:
        assert losses.item() == 0.
        assert not h.grad.any()


def test_bound_random_arrays_independent_numpy_oracle():
    rng = np.random.default_rng(711)
    for n in (1, 2, 3, 17, 82):
        h, target, mask = tensors(rng.uniform(-10, 10, (16, n)), rng.integers(0, 9, (16, n))/4)
        actual = decision_losses(h, target, mask).detach().numpy()
        for i in range(16):
            e = h[i].detach().numpy().sum(axis=1)
            t = target[i].numpy().astype(np.float64).sum(axis=1)
            optimum = [j for j in range(n) if t[j] == min(t)]
            ref = sum(float(e[j]) for j in optimum)/len(optimum)
            expected = max(float(t[j]-min(t)+ref-e[j]) for j in range(n))
            assert abs(actual[i]-expected) <= 1e-12
            assert actual[i]+1e-12 >= t[int(np.argmin(e))]-min(t)


def test_amax_subgradient_shared_at_exact_tie():
    h, target, mask = tensors([[0., 1., 1.]], [[0., 1., 1.]])
    loss = decision_loss(h, target, mask)
    assert loss.item() == 0.
    loss.backward()
    # reference contributes +1 to optimal; tied violations each contribute -1/3.
    np.testing.assert_allclose(h.grad.numpy()[0, :, 0], [2/3, -1/3, -1/3], atol=1e-15, rtol=0)
    np.testing.assert_array_equal(h.grad.numpy()[..., 0], h.grad.numpy()[..., 1])


def test_objective_uses_double_sum_of_delivered_components():
    # Both float32 sums are 1, but the sums of separately promoted components differ.
    target = torch.tensor([[[1., 2**-25], [1., 0.]]], dtype=torch.float32)
    assert torch.equal(target.sum(-1), torch.ones((1, 2)))
    h = torch.tensor([[[0., 0.], [.5, .5]]], dtype=torch.float64)
    mask = torch.ones((1, 2), dtype=torch.bool)
    assert decision_loss(h, target, mask).item() == 1+2**-25


def test_padding_and_scene_weighting_not_candidate_weighting():
    h = torch.tensor([[[1., 1.], [999., -999.]], [[2., 2.], [2., 2.]]], dtype=torch.float64)
    t = torch.zeros((2, 2, 2), dtype=torch.float32)
    m = torch.tensor([[True, False], [True, True]])
    assert component_mse(h, t, m).item() == 2.5
    assert decision_loss(h, t, m).item() == 0.
    for mask in (torch.zeros_like(m), torch.ones_like(m, dtype=torch.float32)):
        with pytest.raises(ValueError):
            decision_loss(h, t, mask)
    with pytest.raises(ValueError):
        decision_loss(h.float(), t, m)
    t[0, 1, 0] = .1
    with pytest.raises(ValueError):
        decision_loss(h, t, m)


def test_model_transport_and_mask_validation():
    inputs, _, _ = interface()
    model = GeometricDecisionHead(core.READER_SEEDS[0], "geometric")
    with torch.no_grad():
        model.partition2.weight.fill_(.125)
        model.partition2.bias.fill_(.25)
    original = model(collate([inputs])).detach().numpy()[0]
    moved = {"groups": inputs["groups"][::-1].copy(), "globals": inputs["globals"][::-1].copy(),
             "incidence": inputs["incidence"][::-1, ::-1].copy(), "evidence": inputs["evidence"][::-1].copy()}
    changed = model(collate([moved])).detach().numpy()[0, ::-1]
    np.testing.assert_allclose(original, changed, atol=1e-6, rtol=1e-5)
    bad = collate([inputs])
    bad["incidence"][0, 0] = 0
    with pytest.raises(ValueError, match="incidence"):
        model(bad)
    bad = collate([inputs])
    bad["targets"] = torch.zeros((1, 2, 2))
    with pytest.raises(ValueError, match="schema"):
        model(bad)


def test_roundtrip_has_fixed_dtype_and_order_not_a_sampler():
    q = np.array([-.51, -.44, -.23, -.12, .09, .31, .40, .50], np.float32)
    out = core.recentered_roundtrip(q)
    shifted = (q.astype(np.float64)+np.log(2.)).astype(np.float32).astype(np.float64)
    np.testing.assert_array_equal(out["q_probe"], (shifted-np.mean(shifted, dtype=np.float64)).astype(np.float32))
    assert out["q_probe"].dtype == np.float32
    np.testing.assert_array_equal(out["original"], q)


def test_no_empty_scene_or_privileged_input_in_head():
    x, _, _ = interface()
    empty = {k: v[:0].copy() for k, v in x.items()}
    with pytest.raises(ValueError):
        collate([empty])
    x["target"] = np.zeros((2, 2), np.float32)
    with pytest.raises(ValueError):
        collate([x])


def test_padding_cannot_create_overflow_in_mse_gradient():
    h = torch.tensor([[[1., 1.], [1e300, 1e300]]], dtype=torch.float64, requires_grad=True)
    target = torch.zeros((1, 2, 2), dtype=torch.float32)
    mask = torch.tensor([[True, False]])
    loss = component_mse(h, target, mask)
    loss.backward()
    assert loss.item() == 1.
    assert torch.isfinite(h.grad).all()
    assert not h.grad[0, 1].any()


@pytest.mark.parametrize("loss_function", [component_mse, decision_loss])
def test_dense_envelope_forward_backward(loss_function):
    # Deliberately not a realizable partition dataset: maximum tensor envelope.
    x = {"groups": np.sin(np.arange(328*9, dtype=np.float32)).reshape(328, 9),
         "globals": np.cos(np.arange(82*17, dtype=np.float32)).reshape(82, 17),
         "incidence": np.full((82, 328), 1/328, np.float32),
         "evidence": np.linspace(0, 1, 82*8, dtype=np.float32).reshape(82, 8)}
    batch = collate([x]*32)
    model = GeometricDecisionHead(core.READER_SEEDS[1], "geometric")
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, foreach=False, fused=False)
    # Reverse candidate cost so the initial increasing bypass violates the margin.
    target = torch.from_numpy(np.linspace(0, 1, 32*82*2, dtype=np.float32).reshape(32, 82, 2)).flip(1)
    output = model(batch)
    loss = loss_function(output, target, batch["candidate_mask"])
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer.step()
    assert torch.isfinite(model(batch)).all()
    assert model.partition2.weight.count_nonzero() > 0
