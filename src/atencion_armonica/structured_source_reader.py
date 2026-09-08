"""Observable-only partition pool and shared-source energy reader.

No truth, calibration selection, dataset generation, Torch or checkpoint IO.
All partition/group signatures use stable observed-frequency ranks.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from scipy.special import expit

from .source_coherence import GroupFitCache

GAMMAS = (0., .01, .03, .1, .3, 1.)
FACTORS = ("pairs", "shared_source", "decoupled_source", "local_compatibility")


def _q_vector(q):
    q = np.asarray(q)
    if q.dtype != np.float32 or q.ndim != 1 or not 3 <= len(q) <= 32 or not np.isfinite(q).all():
        raise ValueError("expected 3..32 finite observed float32 coordinates")
    return q


def _matrix(value, n, *, probability=False, logits=False):
    a = np.asarray(value)
    if (a.shape != (n, n) or not np.issubdtype(a.dtype, np.floating)
            or not np.isfinite(a).all() or not np.array_equal(a, a.T)
            or (logits and a.dtype != np.float32)
            or (probability and (np.any(a < 0) or np.any(a > 1)))):
        raise ValueError("invalid symmetric observed matrix")
    return a.astype(np.float64)


def signature(groups):
    return tuple(sorted(tuple(sorted(int(i) for i in g)) for g in groups))


def tree_cuts(distance):
    """Historical scipy average-linkage, every merge retained (also ties)."""
    n = len(distance)
    condensed = squareform(distance, checks=True)
    tree = linkage(condensed, method="average", optimal_ordering=False)
    clusters = {i: (i,) for i in range(n)}
    cuts = [signature(clusters.values())]
    for step, (left, right, _, _) in enumerate(tree):
        clusters[n+step] = tuple(sorted(clusters.pop(int(left))+clusters.pop(int(right))))
        cuts.append(signature(clusters.values()))
    return {"tree": tree.tolist(), "cuts": cuts,
            "input_distance_repetitions": int(len(condensed)-len(np.unique(condensed))),
            "merge_height_repetitions": int(len(tree)-len(np.unique(tree[:, 2])))}


def build_pool(q, logits, pair_support):
    q = _q_vector(q)
    z = _matrix(logits, len(q), logits=True)
    support = _matrix(pair_support, len(q), probability=True)
    order = np.argsort(q, kind="stable")
    trees = {}
    for name, p in (("neural", expit(z)), ("analytic", support)):
        distance = 1-p[np.ix_(order, order)]
        np.fill_diagonal(distance, 0.)
        trees[name] = tree_cuts(distance)
    unfiltered = sorted(set(trees["neural"]["cuts"]) | set(trees["analytic"]["cuts"]))
    partitions = [p for p in unfiltered if max(map(len, p)) <= 8]
    return {"canonical_to_observed": order.tolist(), "canonical_q32": q[order].tolist(),
            "q_tie_count": int(len(q)-len(np.unique(q))), "trees": trees,
            "unfiltered_partitions": unfiltered, "partitions": partitions,
            "unfiltered_count": len(unfiltered), "filtered_count": len(unfiltered)-len(partitions),
            "retained_count": len(partitions), "max_group_size_prior": 8}


def bounded_cost(residual_cents):
    r = np.asarray(residual_cents, dtype=np.float64)
    if not np.isfinite(r).all() or np.any(r < 0):
        raise ValueError("invalid nonnegative residual")
    # All finite float32 observations yield residuals far below float64 overflow.
    with np.errstate(over="raise", invalid="raise"):
        return r*r/(r*r+4.)


def decouple_costs(groups, costs, *, split_seed, scene_id):
    if any(type(x) is not int or x < 0 for x in (split_seed, scene_id)):
        raise ValueError("invalid scene metadata")
    costs = np.asarray(costs, dtype=np.float64)
    if (costs.shape != (len(groups),) or not np.isfinite(costs).all()
            or np.any(costs < 0) or np.any(costs > 1)
            or list(groups) != sorted(set(groups))):
        raise ValueError("expected canonical unique groups and bounded costs")
    result, strata = costs.copy(), []
    for m in sorted(set(map(len, groups))):
        ids = np.array([i for i, g in enumerate(groups) if len(g) == m], dtype=np.int64)
        length = len(ids)
        rng = np.random.default_rng(np.random.SeedSequence([2026090785, split_seed, scene_id, m]))
        shift = int(rng.integers(1, length)) if length > 1 else None
        assigned = ids if shift is None else ids[(np.arange(length)+shift) % length]
        result[ids] = costs[assigned]
        delta = np.abs(result[ids]-costs[ids])
        strata.append({"size": m, "group_ids": ids.tolist(), "length": length,
                       "shift": shift, "donor_group_ids": assigned.tolist(),
                       "changed_fraction": float(np.mean(delta != 0)),
                       "mean_abs_cost_change": float(delta.mean()), "max_abs_cost_change": float(delta.max()),
                       "status": "NO_SHAM_CONTRAST" if not np.any(delta) else "COSTS_CHANGED"})
    return result, strata


def partition_energies(z, partitions, groups, costs):
    """Common extensive normalization; constants retained for verification."""
    n = len(z)
    lookup = {group: i for i, group in enumerate(groups)}
    constant = float(np.logaddexp(0., z[np.triu_indices(n, 1)]).sum())
    rows = []
    for partition in partitions:
        members = [lookup[g] for g in partition]
        within = sum(float(z[a, b]) for g in partition for a, b in combinations(g, 2))
        factor_values = {name: sum(len(groups[i])*float(values[i]) for i in members)/n
                         for name, values in costs.items()}
        small = [g for g in partition if len(g) < 3]
        rows.append({"signature": partition, "group_ids": members,
                     "bce_sum": constant-within, "bce_constant": constant,
                     "pair_dependent_sum": -within, "pair_energy": -within/n,
                     "factor_energies": factor_values, "group_count": len(partition),
                     "sub3_group_count": len(small), "sub3_member_fraction": sum(map(len, small))/n,
                     "zero_cost_member_fraction": {
                         name: sum(len(groups[i]) for i in members if values[i] == 0)/n
                         for name, values in costs.items()}})
    return rows


def sham_energy_support(shared, decoupled):
    shared, decoupled = np.asarray(shared), np.asarray(decoupled)
    delta = decoupled-shared
    if delta.ndim != 1 or not len(delta) or not np.isfinite(delta).all():
        raise ValueError("invalid candidate energy vectors")
    return {"shared_energies": shared.tolist(), "decoupled_energies": decoupled.tolist(),
            "changed_fraction": float(np.mean(delta != 0)),
            "mean_abs_energy_change": float(np.abs(delta).mean()),
            "max_abs_energy_change": float(np.abs(delta).max()),
            "status": "NOT_EVALUABLE_SHAM_SUPPORT" if np.all(delta == delta[0]) else "RANKING_CONTRAST_AVAILABLE"}


def score_scene(q, logits, pair_support, triples, residual_cents, *, split_seed, scene_id, fit_cache=None):
    """Fit canonical groups once across checkpoint pools; no supervision port."""
    pool = build_pool(q, logits, pair_support)
    n, order = len(q), np.asarray(pool["canonical_to_observed"])
    triples, residuals = np.asarray(triples), np.asarray(residual_cents)
    expected = np.array(list(combinations(range(n), 3)), dtype=np.int64)
    if (triples.dtype != np.int64 or not np.array_equal(triples, expected)
            or residuals.shape != (len(expected),) or not np.isfinite(residuals).all()
            or np.any(residuals < 0)):
        raise ValueError("expected complete ordered observable triple cache")
    inverse = np.argsort(order)
    endpoint = {tuple(sorted(inverse[t])): float(r) for t, r in zip(triples, residuals)}
    q_canonical = np.asarray(q)[order]
    z = np.asarray(logits, dtype=np.float64)[np.ix_(order, order)]
    groups = sorted({g for p in pool["partitions"] for g in p})
    cache = fit_cache if fit_cache is not None else GroupFitCache()
    start_hits, start_misses = cache.hits, cache.misses
    records, global_costs, local_costs = [], [], []
    for group in groups:
        witness = cache.fit(str(split_seed), scene_id, q_canonical, group)
        local = [endpoint[t] for t in combinations(group, 3)]
        global_costs.append(float(bounded_cost(witness["fine"]["minimum_cents"])) if len(group) >= 3 else 0.)
        local_costs.append(float(bounded_cost(local).mean()) if local else 0.)
        records.append({"members": group, "size": len(group), "witness": witness,
                        "constraint_status": "ABSENT_UNDERCONSTRAINED" if len(group) < 3 else "GRID_FACTOR_NOT_CERTIFICATE",
                        "endpoint_residual_cents": local,
                        "pool_occurrences": sum(group in p for p in pool["partitions"])})
    sham, strata = decouple_costs(groups, global_costs, split_seed=split_seed, scene_id=scene_id)
    costs = {"shared_source": global_costs, "decoupled_source": sham, "local_compatibility": local_costs}
    for i, record in enumerate(records):
        record["costs"] = {name: float(values[i]) for name, values in costs.items()}
    rows = partition_energies(z, pool["partitions"], groups, costs)
    return {"pool": pool, "groups": records, "candidates": rows, "sham_strata": strata,
            "sham_support": sham_energy_support(
                [r["factor_energies"]["shared_source"] for r in rows],
                [r["factor_energies"]["decoupled_source"] for r in rows]),
            "fit_cache_hits": cache.hits-start_hits, "fit_cache_misses": cache.misses-start_misses}


def choose_partition(scored, factor, gamma=0.):
    if factor not in FACTORS or gamma not in GAMMAS or (factor == "pairs" and gamma != 0):
        raise ValueError("reader or gamma outside frozen design")
    rows = scored["candidates"]
    energies = np.array([r["pair_energy"] if gamma == 0 else
                         r["pair_energy"]+gamma*r["factor_energies"][factor] for r in rows], dtype=np.float64)
    if not len(energies) or not np.isfinite(energies).all():
        raise ValueError("invalid candidate energies")
    minimum = float(energies.min())
    ties = np.flatnonzero(energies == minimum)
    chosen = min(ties.tolist(), key=lambda i: signature(rows[i]["signature"]))
    other = energies[energies > minimum]
    partition = rows[chosen]["signature"]
    mapping = scored["pool"]["canonical_to_observed"]
    return {"candidate_index": chosen, "canonical_signature": partition,
            "partition": signature([[mapping[i] for i in group] for group in partition]),
            "energy": minimum, "candidate_energies": energies.tolist(),
            "co_minimum_count": len(ties), "next_level_energy": float(other.min()) if len(other) else None,
            "next_level_gap": float(other.min()-minimum) if len(other) else None,
            "authority": "TIED_CANONICAL_CHOICE" if len(ties) > 1 else "UNIQUE_WITHIN_POOL"}
