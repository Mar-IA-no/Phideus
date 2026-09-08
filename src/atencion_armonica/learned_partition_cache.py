"""Lossless observable feature-row storage; no truth loading or Torch.

Integrity of the containing sealed bundle and dataset authorization must be
checked by its caller. This module rejects malformed numerical structure.
"""
from __future__ import annotations

import numpy as np

from .learned_partition_core import ARMS, FeatureRows
from .structured_source_artifacts import write_npz
from .structured_source_reader import signature


def validate_rows(row):
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
    cardinalities = []
    for group in row.groups:
        if (not 1 <= len(group) <= 8 or tuple(sorted(set(group))) != group
                or any(type(x) is not int or not 0 <= x < n for x in group)):
            raise ValueError("invalid canonical group members")
        m = len(group)
        cardinalities.append([m/8, m/n, m*(m-1)/(n*(n-1)), float(m < 3)])
    if not np.array_equal(row.group_features[:, [0, 1, 2, 7]], cardinalities):
        raise ValueError("group cardinality features differ")
    sizes = np.asarray([len(g) for g in row.groups])
    stats = row.group_features[:, 3:7]
    if (np.any(stats[:, 1] < 0) or np.any(stats[:, 2] > stats[:, 0])
            or np.any(stats[:, 0] > stats[:, 3]) or np.any(stats[sizes == 1] != 0)):
        raise ValueError("invalid within-group logit statistics")
    for arm in ARMS[1:]:
        cost = row.costs[arm]
        if np.any(cost < 0) or np.any(cost > 1) or np.any(cost[sizes < 3] != 0):
            raise ValueError("invalid bounded factor or underconstrained group")
    lookup = {g: i for i, g in enumerate(row.groups)}
    incidence = np.zeros_like(row.incidence)
    partition_cardinalities = []
    for i, candidate in enumerate(row.candidates):
        if signature(candidate) != candidate or sorted(x for g in candidate for x in g) != list(range(n)):
            raise ValueError("invalid complete partition")
        try:
            ids = [lookup[g] for g in candidate]
        except KeyError as exc:
            raise ValueError("candidate group absent from roster") from exc
        incidence[i, ids] = [len(g)/n for g in candidate]
        partition_cardinalities.append([n/32, len(candidate)/n, sum((len(g)/n)**2 for g in candidate),
                    sum(len(g) for g in candidate if len(g) == 1)/n,
                    sum(len(g) for g in candidate if len(g) < 3)/n])
    if not np.array_equal(row.global_features[:, [0, 1, 3, 4, 5]], partition_cardinalities):
        raise ValueError("partition cardinality features differ")
    if not np.array_equal(row.incidence, incidence) or np.any(incidence.sum(0) == 0):
        raise ValueError("incidence differs or group is unused")
    return row


def save_rows(path, row):
    validate_rows(row)
    lookup = {g: i for i, g in enumerate(row.groups)}
    group_sizes = np.array([len(g) for g in row.groups], dtype=np.int64)
    candidate_sizes = np.array([len(c) for c in row.candidates], dtype=np.int64)
    write_npz(path, n=np.asarray(row.n, np.int64),
              group_members=np.array([i for g in row.groups for i in g], np.int64),
              group_offsets=np.r_[np.int64(0), group_sizes.cumsum()],
              candidate_group_ids=np.array([lookup[g] for c in row.candidates for g in c], np.int64),
              candidate_offsets=np.r_[np.int64(0), candidate_sizes.cumsum()],
              group_features=row.group_features, global_features=row.global_features,
              incidence=row.incidence, **{a: row.costs[a] for a in ARMS[1:]})


def load_rows(path):
    with np.load(path, allow_pickle=False) as raw:
        expected = {"n", "group_members", "group_offsets", "candidate_group_ids", "candidate_offsets",
                    "group_features", "global_features", "incidence", *ARMS[1:]}
        if set(raw.files) != expected:
            raise ValueError("observable cache schema differs or includes supervision")
        a = {k: raw[k] for k in raw.files}
    if a["n"].shape != () or a["n"].dtype != np.int64:
        raise ValueError("invalid event-count schema")
    for key in ("group_members", "group_offsets", "candidate_group_ids", "candidate_offsets"):
        if a[key].ndim != 1 or a[key].dtype != np.int64:
            raise ValueError("invalid ragged index schema")
    for offset, value in (("group_offsets", "group_members"), ("candidate_offsets", "candidate_group_ids")):
        indices = a[offset]
        if len(indices) < 2 or indices[0] != 0 or indices[-1] != len(a[value]) or np.any(np.diff(indices) <= 0):
            raise ValueError("invalid ragged offsets")
    groups = tuple(tuple(int(v) for v in a["group_members"][lo:hi])
                   for lo, hi in zip(a["group_offsets"][:-1], a["group_offsets"][1:]))
    if np.any(a["candidate_group_ids"] < 0) or np.any(a["candidate_group_ids"] >= len(groups)):
        raise ValueError("candidate group index outside roster")
    candidates = tuple(tuple(groups[int(v)] for v in a["candidate_group_ids"][lo:hi])
                       for lo, hi in zip(a["candidate_offsets"][:-1], a["candidate_offsets"][1:]))
    return validate_rows(FeatureRows(int(a["n"]), groups, candidates, a["group_features"],
                                    a["global_features"], a["incidence"], {k: a[k] for k in ARMS[1:]}))
