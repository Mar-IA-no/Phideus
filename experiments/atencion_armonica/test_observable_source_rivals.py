"""Mechanical tests only: no campaign data, networks, or implicit CUDA."""
from itertools import combinations, product
import inspect
import json

import numpy as np
import pytest
from scipy.optimize import minimize

from src.atencion_armonica import observable_source_rivals as r


def fixture():
    raw = np.concatenate([np.log(f0)+r.template(range(1, 5), beta, 0.)
                          for f0, beta in ((120., 1e-4), (410., 1e-3))])
    order = np.argsort(raw, kind="stable")
    q = r.centered(raw).astype(np.float32)[order].astype(np.float64)
    labels = np.repeat([0, 1], 4)[order]
    p = r.signature([np.flatnonzero(labels == i).tolist() for i in range(2)])
    return q, p


def test_projection_against_independent_constrained_optimizer():
    rng = np.random.default_rng(2026090902)
    for _ in range(20):
        k = int(rng.integers(2, 5))
        a, m = rng.normal(0., 2., k), rng.integers(4, 9, k)
        b = r.project_offsets(a, m)
        constraints = [{"type": "ineq", "fun": lambda x, i=i, j=j: r.WIDTH-x[i]+x[j]}
                       for i, j in product(range(k), repeat=2) if i != j]
        ref = minimize(lambda x: np.sum(m*(x-a)**2), np.full(k, np.average(a, weights=m)),
                       method="SLSQP", constraints=constraints,
                       options={"ftol": 1e-11, "maxiter": 300})
        assert abs(np.sum(m*(b-a)**2)-ref.fun) < 1e-7
        assert abs(np.sum(m*(b-a))) < 1e-12
        assert np.ptp(b) <= r.WIDTH+1e-12
    a = np.array([.1, .2])
    assert np.array_equal(a, r.project_offsets(a, [4, 8]))


def test_noise_projection_and_quantization_bound():
    rng = np.random.default_rng(713)
    n = 13
    p = np.eye(n)-np.ones((n, n))/n
    assert np.linalg.matrix_rank(p) == n-1
    np.testing.assert_allclose(p@p.T, p, atol=1e-14)
    for _ in range(50):
        observed = r.centered(rng.normal(size=n))
        q = observed.astype(np.float32).astype(np.float64)
        pred = r.centered(rng.normal(size=n))
        metrics = r.witness_metrics(q, pred)
        original_j = np.sum((observed-pred)**2)/(2*r.SIGMA**2)
        assert abs(original_j-metrics["J"]) <= metrics["quantization_objective_bound"]+1e-7
        assert metrics["noiseless_collision_status"] == "NOT_TESTED"
    q = np.array([-1., 1.])
    assert r.witness_metrics(q, q)["sample_fit_status"] == "EXACT_SAMPLE_FIT"


def test_inventory_determinism_no_truth_and_cardinality():
    q, p = fixture()
    inv = r.candidate_inventory({"a": [p], "b": [p]}, len(q))
    # All 4x4 swaps; no move can preserve minimum size four here.
    assert inv["neighbor_count_before_cap"] == 16
    assert len(inv["candidates"]) == 17
    assert all(r.supported(row["partition"]) for row in inv["candidates"])
    assert inv == r.candidate_inventory({"b": [p[::-1]], "a": [p]}, len(q))
    singleton = tuple((i,) for i in range(len(q)))
    empty = r.candidate_inventory({"a": [singleton]}, len(q))
    assert empty["neighbor_count_retained"] == 0
    assert empty["candidates"][0]["status"] == "OUTSIDE_GENERATIVE_CARDINALITY"
    with pytest.raises(ValueError):
        r.candidate_inventory({"bad": [[(0, 1), (1, 2)]]}, 3)
    assert tuple(inspect.signature(r.fit_candidates).parameters) == ("q", "partitions", "fitter")


def test_group_grid_minima_against_scalar_exhaustion():
    grid = r.Grid(5, 5)
    y = r.template([1, 3, 5, 8], 3e-4, 1e-5)+np.array([0, 1, -1, 2])*r.CENTS_TO_LOG
    row = r.GroupFitter(grid).fit([y], "deformed-low")[0]
    beta, gamma = grid.values("deformed-low")
    for res, step in (("fine", 1), ("coarse", 4)):
        for ns, actual in zip(combinations(range(1, 9), 4), row[res]):
            expected = [(np.sum((r.centered(y)-r.centered(r.template(ns, b, g)))**2), bi, gi)
                        for bi, b in enumerate(beta) if bi % step == 0
                        for gi, g in enumerate(gamma) if gi % step == 0]
            cost, bi, gi = min(expected)
            assert abs(cost-actual["sse"]) < 1e-13
            assert (bi, gi) == (actual["beta_index"], actual["gamma_index"])


def test_fits_support_nesting_replay_and_gauge():
    q, p = fixture()
    inv = r.candidate_inventory({"a": [p]}, len(q))
    ps = [row["partition"] for row in inv["candidates"][:3]]
    out = r.fit_candidates(q, ps, r.GroupFitter(r.Grid(5, 5)))
    for fit in out["fits"]:
        assert fit["families"]["extended"]["UB"] <= fit["families"]["base"]["UB"]
        for b in fit["branches"].values():
            assert b["LB"] <= b["UB"]+r.TOL_J
            assert b["UB"] <= b["coarse"]["J"]
            assert min(b["witness"]["f0"]) >= 100.-1e-10
            assert max(b["witness"]["f0"]) <= 500.+1e-10
    serialized = json.loads(json.dumps(out))
    replay = r.replay_fits(q, serialized["group_factors"], ps)
    assert json.dumps(replay, sort_keys=True) == json.dumps(out["fits"], sort_keys=True)
    order = np.array([3, 1, 5, 0, 7, 4, 2, 6])
    # A scene permutation canonicalizes to exactly the same observable input.
    np.testing.assert_array_equal(np.sort(q[order], kind="stable"), q)
    # Common translation vanishes in the mathematical objective before q32 rounding.
    fs = {(x["branch"], tuple(x["group"])): x["factor"] for x in out["group_factors"]}
    factors = [fs[("base-low", g)] for g in p]
    a = r.joint_search(r.centered(q), p, factors, "fine")
    b = r.joint_search(r.centered(q+4.), p, factors, "fine")
    assert abs(a["J"]-b["J"]) < r.TOL_J


def test_small_complete_grid_brackets_true_discrete_minimum():
    # Size eight has only one index assignment; enumerate all grid products.
    grid = r.Grid(5, 5)
    beta, _ = grid.values("base-low")
    raw = np.concatenate([np.log(f0)+r.template(range(1, 9), b, 0.)
                          for f0, b in ((100., beta[1]), (500., beta[3]))])
    order = np.argsort(raw, kind="stable")
    q = r.centered(raw).astype(np.float32)[order].astype(np.float64)
    labels = np.repeat([0, 1], 8)[order]
    p = r.signature([np.flatnonzero(labels == i).tolist() for i in range(2)])
    out = r.fit_candidates(q, [p], r.GroupFitter(grid))["fits"][0]["branches"]["base-low"]
    y = r.centered(q)
    costs = []
    for b1, b2 in product(beta, repeat=2):
        ts = [r.template(range(1, 9), b, 0.) for b in (b1, b2)]
        offsets = [np.mean(y[list(g)]-t) for g, t in zip(p, ts)]
        b = r.project_offsets(offsets, [8, 8])
        mu = np.empty(len(q))
        for g, offset, t in zip(p, b, ts):
            mu[list(g)] = offset+t
        costs.append(np.sum((y-r.centered(mu))**2)/(2*r.SIGMA**2))
    optimum = min(costs)
    assert out["LB"] <= optimum+r.TOL_J
    assert optimum <= out["UB"]+r.TOL_J


def test_independent_rival_argmins_and_tolerance():
    def fake(p, lo, hi):
        return {"partition": p, "status": "FITTED", "families": {"base": {"LB": lo, "UB": hi}}}
    truth = fake(((0, 1), (2, 3)), 5., 6.)
    low = fake(((0, 2), (1, 3)), 1., 12.)
    upper = fake(((0, 3), (1, 2)), 4., 4.5)
    result = r.rival_margin([truth, low, upper], truth, "base")
    assert result["interval"] == [-5., -.5]
    assert result["lower_partition"] == low["partition"]
    assert result["upper_partition"] == upper["partition"]
    assert result["status"] == "RIVAL_BETTER_ON_GRID"
    assert r.rival_margin([truth], truth, "base")["delta_UB"] is None
    near = fake(upper["partition"], 5., 5.-r.TOL_J/2)
    assert r.rival_margin([near], truth, "base")["status"] == "UNRESOLVED_GRID_COMPARISON"


def test_branch_cardinality_and_invalid_inputs():
    q = np.arange(16, dtype=np.float32).astype(np.float64)
    p = tuple(tuple(range(i, i+4)) for i in range(0, 16, 4))
    fit = r.fit_candidates(q, [p], r.GroupFitter(r.Grid(5, 5)))["fits"][0]
    assert set(fit["branches"]) == {"base-low"}
    with pytest.raises(ValueError):
        r.GroupFitter().fit([[4., 3., 2., 1.]], "base-low")
    with pytest.raises(ValueError):
        r.Grid(6, 5)
    with pytest.raises(ValueError):
        r.witness_metrics([.1, .2], [.1, .2])


def test_q32_port_is_closed_even_without_supported_candidates():
    outside = tuple((i,) for i in range(8))
    invalid = (np.array([.1, .2]), np.linspace(-1, 1, 8), np.ones((2, 4)),
               np.array([0, 1, 2, 3, 4, 5, 6, np.nan]), np.arange(8.)[::-1])
    for q in invalid:
        for partitions in ([], [outside]):
            with pytest.raises(ValueError):
                r.fit_candidates(q, partitions, r.GroupFitter(r.Grid(5, 5)))
            with pytest.raises(ValueError):
                r.replay_fits(q, [], partitions)
    q = np.arange(8, dtype=np.float32)
    assert r.fit_candidates(q, [], r.GroupFitter())["fits"] == []
    assert r.replay_fits(q, [], []) == []
    assert r.fit_candidates(q, [outside], r.GroupFitter())["fits"][0]["status"] == "OUTSIDE_GENERATIVE_CARDINALITY"
