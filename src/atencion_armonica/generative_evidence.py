"""Observable candidate-level evidence, shared features and stratified controls.

Pure NumPy: no supervision, model loading, data generation or CUDA entry point.
Campaign provenance and permission to consume a split belong to the caller.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np

from . import observable_source_rivals as law
from .structured_source_reader import bounded_cost

ARMS = ("local", "generative", "decoupled")
CHECKPOINTS = (2026090721, 2026090722, 2026090723)
READER_SEEDS = (2026090991, 2026090992, 2026090993)
BRANCHES = ("base-low", "base-high", "deformed-low")
MAX_CANDIDATES, MAX_GROUPS = 82, 328


def partitions_checked(partitions, n):
    ps = [law.validate_partition(p, n) for p in partitions]
    if ps != sorted(set(ps)) or len(ps) > MAX_CANDIDATES or any(not law.supported(p) for p in ps):
        raise ValueError("expected canonical unique supported candidate list")
    return ps


def candidate_channel(partitions, fits, n):
    """Six branch-specific grid costs; availability is common metadata."""
    ps = partitions_checked(partitions, n)
    if len(fits) != len(ps):
        raise ValueError("fit roster differs from candidate roster")
    values = np.zeros((len(ps), 6), np.float64)
    available = np.zeros((len(ps), 3), bool)
    dimensions = np.zeros((len(ps), 3), np.float64)
    for i, (p, fit) in enumerate(zip(ps, fits)):
        expected = {b for b in BRANCHES if len(p) in law.BRANCHES[b][2]}
        if (fit["status"] != "FITTED" or law.signature(fit["partition"]) != p
                or set(fit["branches"]) != expected):
            raise ValueError("candidate fit identity, status or branch support differs")
        for j, branch in enumerate(BRANCHES):
            if branch not in expected:
                continue
            f = fit["branches"][branch]
            pair = np.asarray([f["LB"], f["UB"]], np.float64)
            if not np.isfinite(pair).all() or np.any(pair < 0) or pair[0] > pair[1]+law.TOL_J:
                raise ValueError("invalid discrete fit bounds")
            values[i, 2*j:2*j+2] = np.log1p(pair/n)
            available[i, j] = True
            dimensions[i, j] = ((3 if branch == "deformed-low" else 2)*len(p)-1)/n
    return values, available, dimensions


def observable_rows(q, logits, triples, residual_cents, partitions, fits):
    """q/logits/triples use delivered order; partitions/fits use frequency ranks."""
    q, z = np.asarray(q), np.asarray(logits)
    if q.dtype != np.float32 or q.ndim != 1:
        raise ValueError("expected delivered float32 frequency coordinates")
    order = np.argsort(q, kind="stable")
    law.observable_q32(q[order])
    n = len(q)
    if (z.dtype != np.float32 or z.shape != (n, n) or not np.isfinite(z).all()
            or not np.array_equal(z, z.T)):
        raise ValueError("expected finite symmetric preserved float32 logits")
    ts, rs = np.asarray(triples), np.asarray(residual_cents)
    expected = np.asarray(list(combinations(range(n), 3)), np.int64)
    if (ts.dtype != np.int64 or not np.array_equal(ts, expected)
            or rs.shape != (len(expected),) or not np.isfinite(rs).all() or np.any(rs < 0)):
        raise ValueError("expected the complete ordered observable triple residuals")
    ps = partitions_checked(partitions, n)
    channel, available, dimensions = candidate_channel(ps, fits, n)
    groups = sorted({g for p in ps for g in p})
    if len(groups) > MAX_GROUPS:
        raise ValueError("group envelope exceeded")
    inverse = np.argsort(order)
    endpoint = {tuple(sorted(inverse[t])): float(r) for t, r in zip(ts, rs)}
    z = z.astype(np.float64)[np.ix_(order, order)]
    common = np.empty((len(groups), 9), np.float64)
    for i, group in enumerate(groups):
        m = len(group)
        within = np.asarray([z[a, b] for a, b in combinations(group, 2)])
        local = bounded_cost([endpoint[t] for t in combinations(group, 3)])
        common[i] = [m/8, m/n, m*(m-1)/(n*(n-1)), within.mean(), within.std(),
                     within.min(), within.max(), float(m < 3), local.mean()]
    global_rows = np.empty((len(ps), 17), np.float64)
    incidence = np.zeros((len(ps), len(groups)), np.float32)
    lookup = {g: i for i, g in enumerate(groups)}
    for i, p in enumerate(ps):
        k = len(p)
        energy = -sum(float(z[a, b]) for g in p for a, b in combinations(g, 2))/n
        global_rows[i] = [n/32, k/n, energy, sum((len(g)/n)**2 for g in p),
                         sum(len(g) for g in p if len(g) == 1)/n,
                         sum(len(g) for g in p if len(g) < 3)/n,
                         *[sum(len(g) == m for g in p)/k for m in range(4, 9)],
                         *available[i].astype(float), *dimensions[i]]
        for g in p:
            incidence[i, lookup[g]] = len(g)/n
    return {"n": n, "partitions": ps, "group_ids": groups,
            "canonical_to_observed": order, "q_tie_count": n-len(np.unique(q)),
            "groups": common, "globals": global_rows, "incidence": incidence,
            "evidence": channel, "available": available}


def _scene_moments(rows, selector, dim):
    """Two streaming passes, same scene then row weighting for mean and variance."""
    count = np.zeros(dim, np.int64)
    total = np.zeros(dim, np.float64)
    for row in rows():
        for j, values in enumerate(selector(row)):
            if len(values):
                total[j] += np.mean(values, dtype=np.float64)
                count[j] += 1
    if np.any(count == 0):
        raise ValueError("normalizer coordinate has no training support")
    mean = total/count
    variance = np.zeros(dim, np.float64)
    second_count = np.zeros(dim, np.int64)
    for row in rows():
        for j, values in enumerate(selector(row)):
            if len(values):
                variance[j] += np.mean(np.square(values-mean[j]), dtype=np.float64)
                second_count[j] += 1
    if not np.array_equal(count, second_count):
        raise ValueError("normalizer source changed between passes")
    variance /= count
    zero = variance == 0
    scale = np.where(zero, 1., np.sqrt(variance))
    if not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("invalid training moments")
    return {"mean": mean.tolist(), "scale": scale.tolist(),
            "zero_variance": zero.tolist(), "scene_count": count.tolist()}


def fit_common_normalizer(rows):
    """Caller supplies a restartable, bound TRAIN-only row iterator."""
    return _scene_moments(rows, lambda r: [*[r["groups"][:, j] for j in range(3, 7)],
                                         r["globals"][:, 2]], 5)


def fit_evidence_normalizer(rows):
    return _scene_moments(rows, lambda r: [r["evidence"][r["available"][:, j//2], j]
                                         for j in range(6)], 6)


def _normalizer(record, dimension):
    mean, scale = np.asarray(record["mean"], np.float64), np.asarray(record["scale"], np.float64)
    if (mean.shape != (dimension,) or scale.shape != mean.shape
            or not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0)):
        raise ValueError("wrong frozen normalizer")
    return mean, scale


def decouple(partitions, values, *, split_seed, scene_id):
    """Preserve joint vector multiset within scene and sorted group-size tuple."""
    values = np.asarray(values)
    if (values.dtype != np.float32 or values.shape != (len(partitions), 6)
            or not np.isfinite(values).all() or any(type(x) is not int or x < 0 for x in (split_seed, scene_id))):
        raise ValueError("invalid delivered evidence or scene identity")
    ps = [law.signature(p) for p in partitions]
    if ps != sorted(set(ps)):
        raise ValueError("sham requires canonical candidate order")
    strata = sorted({tuple(sorted(map(len, p))) for p in ps})
    donors = np.arange(len(ps), dtype=np.int64)
    records = []
    for sizes in strata:
        ids = np.asarray([i for i, p in enumerate(ps) if tuple(sorted(map(len, p))) == sizes], np.int64)
        rng = np.random.default_rng(np.random.SeedSequence([2026090995, split_seed, scene_id, *sizes]))
        shift = int(rng.integers(1, len(ids))) if len(ids) > 1 else None
        assigned = ids if shift is None else ids[(np.arange(len(ids))+shift) % len(ids)]
        donors[ids] = assigned
        changed = np.any(values[ids] != values[assigned], axis=1)
        records.append({"sizes": sizes, "candidate_ids": ids.tolist(), "donors": assigned.tolist(),
                        "shift": shift, "changed_fraction": float(changed.mean()),
                        "status": "NO_PERMUTATION" if shift is None else
                            "INPUT_CHANGED" if changed.any() else "INPUT_UNCHANGED"})
    result = values[donors].copy()
    changed = np.any(result != values, axis=1)
    return result, {"strata": records, "donors": donors.tolist(), "changed_mask": changed.tolist(),
                    "status": "INPUT_CHANGED" if changed.any() else "INPUT_UNCHANGED"}


def model_inputs(row, common_norm, evidence_norm, arm, *, split_seed, scene_id):
    if arm not in ARMS:
        raise ValueError("unknown evidence intervention")
    mean, scale = _normalizer(common_norm, 5)
    groups, globals_ = row["groups"].copy(), row["globals"].copy()
    groups[:, 3:7] = (groups[:, 3:7]-mean[:4])/scale[:4]
    globals_[:, 2] = (globals_[:, 2]-mean[4])/scale[4]
    emean, escale = _normalizer(evidence_norm, 6)
    evidence = ((row["evidence"]-emean)/escale).astype(np.float32)
    evidence[~np.repeat(row["available"], 2, axis=1)] = 0.
    if arm == "local":
        evidence[:] = 0.
    elif arm == "decoupled":
        evidence, _ = decouple(row["partitions"], evidence, split_seed=split_seed, scene_id=scene_id)
    result = {"groups": groups.astype(np.float32), "globals": globals_.astype(np.float32),
              "incidence": row["incidence"].copy(), "evidence": evidence}
    if any(not np.isfinite(v).all() for v in result.values()):
        raise ValueError("nonfinite normalized observable input")
    return result
