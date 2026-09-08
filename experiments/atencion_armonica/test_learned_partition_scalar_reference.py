"""Frozen pre-optimization validator for differential tests only.

Copied verbatim from commit66ed7a8 learned_partition_cache.validate_rows;
function renamed. Support reference copied from e84108d input_support, renamed.
Not a production fast path or scientific ground truth.
"""
import numpy as np
from src.atencion_armonica.learned_partition_core import ARMS
from src.atencion_armonica.structured_source_reader import signature


def scalar_validation_reference(row):
    if type(row.n) is not int or not 3 <= row.n <= 32:
        raise ValueError("invalid observable event count")
    n, u, c = row.n, len(row.groups), len(row.candidates)
    if (not 1 <= u <= 94 or not 1 <= c <= 64 or tuple(sorted(set(row.groups))) != row.groups
            or tuple(sorted(set(row.candidates))) != row.candidates or set(row.costs) != set(ARMS[1:])):
        raise ValueError("invalid canonical group/candidate roster")
    specifications = ((row.group_features, (u, 8), np.float64),
                      (row.global_features, (c, 6), np.float64),
                      (row.incidence, (c, u), np.float32),
                      *[(row.costs[a], (u,), np.float64) for a in ARMS[1:]])
    for value, shape, dtype in specifications:
        if value.shape != shape or value.dtype != dtype or not np.isfinite(value).all():
            raise ValueError("feature row shape/dtype/value differs")
    for i, group in enumerate(row.groups):
        if (not 1 <= len(group) <= 8 or tuple(sorted(set(group))) != group
                or any(type(x) is not int or not 0 <= x < n for x in group)):
            raise ValueError("invalid canonical group members")
        m = len(group)
        expected = [m/8, m/n, m*(m-1)/(n*(n-1)), float(m < 3)]
        if not np.array_equal(row.group_features[i, [0, 1, 2, 7]], expected):
            raise ValueError("group cardinality features differ")
        stats = row.group_features[i, 3:7]
        if (stats[1] < 0 or stats[2] > stats[0] or stats[0] > stats[3]
                or (m == 1 and np.any(stats != 0))):
            raise ValueError("invalid within-group logit statistics")
        if any(not 0 <= row.costs[a][i] <= 1 or (m < 3 and row.costs[a][i] != 0) for a in ARMS[1:]):
            raise ValueError("invalid bounded factor or underconstrained group")
    lookup = {g: i for i, g in enumerate(row.groups)}
    incidence = np.zeros_like(row.incidence)
    for i, candidate in enumerate(row.candidates):
        if signature(candidate) != candidate or sorted(x for g in candidate for x in g) != list(range(n)):
            raise ValueError("invalid complete partition")
        try:
            ids = [lookup[g] for g in candidate]
        except KeyError as exc:
            raise ValueError("candidate group absent from roster") from exc
        incidence[i, ids] = [len(g)/n for g in candidate]
        expected = [n/32, len(candidate)/n, sum((len(g)/n)**2 for g in candidate),
                    sum(len(g) for g in candidate if len(g) == 1)/n,
                    sum(len(g) for g in candidate if len(g) < 3)/n]
        if not np.array_equal(row.global_features[i, [0, 1, 3, 4, 5]], expected):
            raise ValueError("partition cardinality features differ")
    if not np.array_equal(row.incidence, incidence) or np.any(incidence.sum(0) == 0):
        raise ValueError("incidence differs or group is unused")
    return row


def scalar_support_reference(original, changed):
    """Same canonical group rows, common features/globals and incidence required.

    Aligned changes preserve provenance; only multiset changes count as new
    effective input. Neither status establishes physical semantics.
    """
    if set(original) != {"groups", "globals", "incidence"} or set(changed) != set(original):
        raise ValueError("expected observable-only model inputs")
    a, b = np.asarray(original["groups"]), np.asarray(changed["groups"])
    incidence = np.asarray(original["incidence"])
    if (a.ndim != 2 or a.shape[1] != 9 or b.shape != a.shape
            or a.dtype != np.float32 or b.dtype != np.float32
            or incidence.ndim != 2 or incidence.shape[1] != len(a) or not len(incidence)
            or incidence.dtype != np.float32
            or any(not np.isfinite(v).all() for x in (original, changed) for v in x.values())
            or not np.array_equal(a[:, :8], b[:, :8])
            or not np.array_equal(original["globals"], changed["globals"])
            or not np.array_equal(incidence, changed["incidence"])
            or np.any(incidence < 0) or np.any(incidence > 1)
            or not np.allclose(incidence.sum(axis=1), 1, rtol=0, atol=2e-7)):
        raise ValueError("support comparison changes the common input or incidence")
    group_changed = a[:, 8] != b[:, 8]
    aligned, effective = [], []
    for weights in incidence:
        ids = np.flatnonzero(weights > 0)
        aligned.append(bool(np.any(group_changed[ids])))
        left = sorted(tuple(float(v) for v in np.r_[a[i], weights[i]]) for i in ids)
        right = sorted(tuple(float(v) for v in np.r_[b[i], weights[i]]) for i in ids)
        effective.append(left != right)
    return {"status": "INPUT_CHANGED" if any(effective) else "INPUT_UNCHANGED",
            "group_count": len(a), "candidate_count": len(incidence),
            "changed_group_mask": group_changed.tolist(), "aligned_candidate_mask": aligned,
            "effective_candidate_mask": effective,
            "changed_group_fraction": float(group_changed.mean()),
            "aligned_candidate_fraction": float(np.mean(aligned)),
            "effective_candidate_fraction": float(np.mean(effective))}
