"""Frozen pre-optimization validator for differential tests only.

Copied verbatim from commit66ed7a8 learned_partition_cache.validate_rows;
function renamed. Not a production fast path or scientific ground truth.
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

