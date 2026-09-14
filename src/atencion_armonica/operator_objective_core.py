"""Retrospective operator/target diagnostics; pure NumPy, no file or model access.

Candidate indices always refer to a caller-validated canonical partition roster.
These kernels provide no authority to open data, fit, train, or select a model.
"""
from __future__ import annotations

import math

import numpy as np

NUMERICAL_NEIGHBORHOOD = 1e-12


def _vector(value, *, size=None):
    result = np.asarray(value)
    if (result.ndim != 1 or result.dtype.kind not in "fiu"
            or (size is not None and len(result) != size)
            or not np.isfinite(result).all()):
        raise ValueError("expected a finite numeric vector of the declared length")
    return result


def _indices(indices, size):
    ids = np.arange(size, dtype=np.int64) if indices is None else np.asarray(indices)
    if ids.ndim == 1 and not len(ids):
        ids = ids.astype(np.int64)
    if (ids.ndim != 1 or ids.dtype.kind not in "iu" or np.any(ids < 0)
            or np.any(ids >= size) or not np.array_equal(ids, np.unique(ids))):
        raise ValueError("indices must be sorted, unique canonical candidate indices")
    return ids.astype(np.int64)


def partition_targets(partitions, labels):
    """Recompute mathematical entropies and their delivered float32 representation.

    This general discrete kernel also accepts small unit-test partitions. The
    production adapter separately enforces the inherited 2..4/4..8 support.
    Labels must already be in canonical frequency order; no sorting happens here.
    """
    y = np.asarray(labels)
    if y.ndim != 1 or y.dtype.kind not in "iu" or not 2 <= len(y) <= 32:
        raise ValueError("expected canonical integer labels on 2..32 events")
    n = len(y)
    ps = []
    for partition in partitions:
        p = tuple(tuple(g) for g in partition)
        if (not p or any(not g for g in p)
                or any(type(i) not in (int, np.int64) for g in p for i in g)
                or p != tuple(sorted(tuple(sorted(g)) for g in p))
                or sorted(i for g in p for i in g) != list(range(n))):
            raise ValueError("partition is not canonical, complete and disjoint")
        ps.append(p)
    if ps != sorted(set(ps)):
        raise ValueError("candidate roster must be canonical and unique")
    _, source = np.unique(y, return_inverse=True)
    source_count = int(source.max()) + 1
    raw = np.empty((len(ps), 2), np.float64)
    ari = np.empty(len(ps), np.float64)
    exact = np.empty(len(ps), bool)
    k_error = np.empty(len(ps), np.int64)
    for i, p in enumerate(ps):
        counts = np.asarray([np.bincount(source[list(g)], minlength=source_count)
                             for g in p], np.float64)
        group, truth = counts.sum(axis=1), counts.sum(axis=0)
        a, b = np.nonzero(counts)
        mass = counts[a, b]/n
        raw[i] = [-np.sum(mass*np.log(counts[a, b]/truth[b])),
                  -np.sum(mass*np.log(counts[a, b]/group[a]))]
        tp = int(np.sum(counts*(counts-1)))
        fp = int(np.sum(group*(group-1)))-tp
        fn = int(np.sum(truth*(truth-1)))-tp
        tn = n*(n-1)-tp-fp-fn
        denominator = (tp+fn)*(fn+tn)+(tp+fp)*(fp+tn)
        ari[i] = 1. if fp == fn == 0 else 2.*(tp*tn-fn*fp)/denominator
        exact[i] = bool(np.all(np.count_nonzero(counts, axis=0) == 1)
                        and np.all(np.count_nonzero(counts, axis=1) == 1))
        k_error[i] = len(p)-source_count
    raw[raw == 0] = 0.
    u64 = raw/np.log(n)
    u32 = u64.astype(np.float32)
    return {"raw": raw, "u64": u64, "u32": u32,
            "t64": u64.sum(axis=1, dtype=np.float64),
            "t32": u32.sum(axis=1, dtype=np.float32),
            "ari": ari, "exact": exact, "k_error": k_error}


def order_record(values, indices=None):
    """Minimize exact numeric values, breaking ties by canonical candidate index."""
    x = _vector(values)
    ids = _indices(indices, len(x))
    if not len(ids):
        return {"status": "NO_CANDIDATE", "chosen": None, "order": [],
                "tie_blocks": [], "optima": [], "minimum": None, "next_gap": None}
    order = ids[np.argsort(x[ids], kind="stable")]
    groups = np.split(order, np.flatnonzero(x[order][1:] != x[order][:-1])+1)
    minimum = x[order[0]]
    gap = float(np.float64(x[groups[1][0]])-np.float64(minimum)) if len(groups) > 1 else None
    return {"status": "DEFINED", "chosen": int(order[0]), "order": order.tolist(),
            "tie_blocks": [g.tolist() for g in groups], "optima": groups[0].tolist(),
            "minimum": float(minimum), "next_gap": gap}


def kendall_tau_b(score, target):
    """Exact ties, descriptive tau-b; no p-value or imputation for constants."""
    x = _vector(score)
    y = _vector(target, size=len(x))
    a, b = np.triu_indices(len(x), k=1)
    # Comparisons avoid overflow from subtracting otherwise finite values.
    dx = (x[a] > x[b]).astype(np.int8)-(x[a] < x[b]).astype(np.int8)
    dy = (y[a] > y[b]).astype(np.int8)-(y[a] < y[b]).astype(np.int8)
    c = int(np.sum(dx*dy == 1))
    d = int(np.sum(dx*dy == -1))
    tx = int(np.sum((dx == 0) & (dy != 0)))
    ty = int(np.sum((dy == 0) & (dx != 0)))
    both = int(np.sum((dx == 0) & (dy == 0)))
    denominator = math.sqrt((c+d+tx)*(c+d+ty))
    if not len(x):
        status = "NO_CANDIDATE"
    elif len(x) == 1:
        status = "INSUFFICIENT_PAIRS"
    elif denominator:
        status = "DEFINED"
    elif np.all(dx == 0) and np.all(dy == 0):
        status = "BOTH_CONSTANT"
    elif np.all(dx == 0):
        status = "CONSTANT_SCORE"
    else:
        status = "CONSTANT_TARGET"
    return {"status": status, "value": (c-d)/denominator if denominator else None,
            "concordant": c, "discordant": d, "score_only_ties": tx,
            "target_only_ties": ty, "double_ties": both, "pair_count": len(a)}


def regression_errors(prediction, target):
    """Keep candidate/component errors; sum in float32 before error in float64."""
    h, u = np.asarray(prediction), np.asarray(target)
    if (h.ndim != 2 or h.shape != u.shape or h.shape[1] != 2
            or h.dtype != np.float32 or u.dtype != np.float32
            or not np.isfinite(h).all() or not np.isfinite(u).all()
            or np.any(h < 0) or np.any(u < 0) or np.any(u > 1)):
        raise ValueError("expected finite nonnegative float32 prediction and bounded targets")
    s32 = h.sum(axis=1, dtype=np.float32)
    t32 = u.sum(axis=1, dtype=np.float32)
    if not np.isfinite(s32).all():
        raise ValueError("prediction sum overflowed float32")
    bias = h.astype(np.float64)-u.astype(np.float64)
    squared = np.square(bias)
    sum_squared = np.square(s32.astype(np.float64)-t32.astype(np.float64))
    return {"s32": s32, "component_bias": bias, "component_squared": squared,
            "sum_squared": sum_squared,
            "mean_component_bias": bias.mean(axis=0).tolist() if len(h) else None,
            "mean_component_squared": squared.mean(axis=0).tolist() if len(h) else None,
            "loss_like": float(squared.mean(axis=1).mean()) if len(h) else None,
            "mean_sum_squared": float(sum_squared.mean()) if len(h) else None}


def oracle_report(t64, t32, ari, indices=None):
    t = _vector(t64)
    numeric = _vector(t32, size=len(t))
    a = _vector(ari, size=len(t))
    ids = _indices(indices, len(t))
    truth, rounded, quality = order_record(t, ids), order_record(numeric, ids), order_record(-a, ids)
    out = {"target64": truth, "target32": rounded, "ari": quality}
    if not len(ids):
        return {**out, "comparisons": None}
    x, y, z = truth["chosen"], rounded["chosen"], quality["chosen"]
    near = ids[t[ids]-t[x] <= NUMERICAL_NEIGHBORHOOD].tolist()
    return {**out, "comparisons": {
        "target64_target32_same_choice": x == y,
        "target32_in_target64_optima": y in truth["optima"],
        "target32_target64_regret": float(t[y]-t[x]),
        "target64_target32_regret": float(np.float64(numeric[x])-np.float64(numeric[y])),
        "near_target64_optima": near, "target32_in_near_target64_optima": y in near,
        "target64_ari_same_choice": x == z,
        "target64_ari_optima_intersect": bool(set(truth["optima"]) & set(quality["optima"])),
        "ari_gap_of_target64_choice": float(a[z]-a[x]),
        "target64_regret_of_ari_choice": float(t[z]-t[x])}}


def describe_score(score, targets, indices=None):
    """Decision diagnostics on one full universe or a declared observable stratum."""
    x = _vector(score, size=len(targets["t64"]))
    ids = _indices(indices, len(x))
    choice = order_record(x, ids)
    oracle = oracle_report(targets["t64"], targets["t32"], targets["ari"], ids)
    tau = kendall_tau_b(x[ids], targets["t64"][ids])
    out = {"candidate_ids": ids.tolist(), "choice": choice, "tau": tau}
    if not len(ids):
        return {**out, "decision": None}
    i = choice["chosen"]
    t, u, a = oracle["target64"], oracle["target32"], oracle["ari"]
    return {**out, "decision": {
        "target64_regret": float(targets["t64"][i]-t["minimum"]),
        "target32_regret": float(np.float64(targets["t32"][i])-u["minimum"]),
        "same_target64_choice": i == t["chosen"], "target64_optimal": i in t["optima"],
        "ari": float(targets["ari"][i]),
        "ari_gap": float(targets["ari"][a["chosen"]]-targets["ari"][i]),
        "k_absolute_error": int(abs(targets["k_error"][i])), "exact": bool(targets["exact"][i])}}


def observable_strata(partitions, available, extended_branch):
    """Four separate schemes, never pooled into a replacement for full-universe."""
    mask = np.asarray(available)
    if (mask.dtype != bool or mask.shape != (len(partitions), 3)
            or len(extended_branch) != len(partitions)):
        raise ValueError("candidate branch metadata extent differs")
    schemes = {name: {} for name in ("full", "k", "sizes_available", "sizes_available_branch")}
    schemes["full"]["all"] = list(range(len(partitions)))
    for i, p in enumerate(partitions):
        sizes = ",".join(map(str, sorted(map(len, p))))
        bits = "".join("1" if b else "0" for b in mask[i])
        key = f"sizes={sizes};available={bits}"
        for name, label in (("k", f"k={len(p)}"), ("sizes_available", key),
                            ("sizes_available_branch", f"{key};branch={extended_branch[i]}")):
            schemes[name].setdefault(label, []).append(i)
    return {name: dict(sorted(groups.items())) for name, groups in schemes.items()}


def paired_scene_difference(left_cells, right_cells):
    """Pair strata within each cell, then cells within scene; complete-nine sensitivity.

    Inputs map all nine cell IDs to {stratum_id: scalar or None}. For a classic
    reference the caller supplies the same map under each matched cell ID.
    """
    if set(left_cells) != set(right_cells) or len(left_cells) != 9:
        raise ValueError("paired scene requires the same nine canonical cell IDs")
    cell_results, common = {}, None
    for cell in sorted(left_cells):
        left, right = left_cells[cell], right_cells[cell]
        if set(left) != set(right):
            raise ValueError("paired comparators must describe identical strata")
        valid = []
        for key in sorted(left):
            for value in (left[key], right[key]):
                if value is not None and (not isinstance(value, (int, float, np.integer, np.floating))
                                         or isinstance(value, (bool, np.bool_))
                                         or not math.isfinite(float(value))):
                    raise ValueError("paired metric must be finite or explicitly undefined")
            if left[key] is not None and right[key] is not None:
                valid.append(key)
        delta = [float(left[key])-float(right[key]) for key in valid]
        cell_results[cell] = {"strata": valid, "count": len(valid),
                              "mean": float(np.mean(delta)) if delta else None}
        common = set(valid) if common is None else common & set(valid)
    values = [row["mean"] for row in cell_results.values() if row["mean"] is not None]
    complete = [float(np.mean([float(left_cells[cell][key])-float(right_cells[cell][key])
                               for key in sorted(common)])) for cell in sorted(left_cells)] if common else []
    return {"cells": cell_results, "valid_cell_count": len(values),
            "mean": float(np.mean(values)) if values else None,
            "complete_nine": {"strata": sorted(common), "count": len(common),
                              "mean": float(np.mean(complete)) if complete else None}}
