"""Pure post-seal metrics; no IO, truth authority, fit or model execution.

Caller authenticates labels by event identity and the complete global seal.
Unlike the historical reader, signed components and delivered targets sum64.
"""
from __future__ import annotations

import numpy as np

from . import generative_evidence as ge
from .geometric_decision_core import ARMS, choose_energy
from .operator_objective_core import partition_targets, kendall_tau_b, order_record

BOOTSTRAP_SEED = 2026091494
BOOTSTRAP_COUNT = 10000
CONTRASTS = ("geometric_minus_injection_mse", "geometric_minus_injection_decision",
             "interaction_decision_minus_mse", "geometric_minus_decoupled_decision")


def targets(partitions, labels):
    """One canonical candidate universe; labels already in its event order."""
    ps = ge.partitions_checked(partitions, len(labels))
    value = partition_targets(ps, labels)
    return {k: value[k] for k in ("raw", "u64", "u32", "ari", "exact", "k_error")} | {
        "tM": value["t64"], "tD": value["u32"].astype(np.float64).sum(axis=1, dtype=np.float64)}


def describe(partitions, target, energy, *, components=None):
    """Full-universe decision and optional signed component errors, scene weight1."""
    count = len(partitions)
    required = {"raw", "u64", "u32", "ari", "exact", "k_error", "tM", "tD"}
    if set(target) != required:
        raise ValueError("target schema differs from mathematical/delivered contract")
    for key in ("raw", "u64", "u32", "ari", "exact", "k_error", "tM", "tD"):
        array = target[key]
        shape = (count, 2) if key in ("raw", "u64", "u32") else (count,)
        dtype = np.float32 if key == "u32" else np.bool_ if key == "exact" else np.int64 if key == "k_error" else np.float64
        if not isinstance(array, np.ndarray) or array.dtype != dtype or array.shape != shape or not np.isfinite(array).all():
            raise ValueError("target array dtype, shape or finiteness differs")
    if (not np.array_equal(target["tM"], target["u64"].sum(axis=1, dtype=np.float64))
            or not np.array_equal(target["u32"], target["u64"].astype(np.float32))
            or not np.array_equal(target["tD"], target["u32"].astype(np.float64).sum(axis=1, dtype=np.float64))):
        raise ValueError("target delivered/math sum arithmetic differs")
    # order_record ties by index; the authenticated partition roster is canonical.
    if count:
        ge.partitions_checked(partitions, sum(map(len, partitions[0])))
    chosen = choose_energy(energy, partitions)
    errors = None
    if components is not None:
        if (not isinstance(components, np.ndarray) or components.dtype != np.float64
                or components.shape != (count, 2) or not np.isfinite(components).all()
                or not np.array_equal(energy, components.sum(axis=1, dtype=np.float64))):
            raise ValueError("signed components must reproduce preserved energy exactly")
        bias = components-target["u32"].astype(np.float64)
        squared = np.square(bias)
        if not np.isfinite(squared).all():
            raise ValueError("component squared error overflow")
        errors = {"bias": bias, "squared": squared,
                  "mse": float(squared.mean(axis=1).mean()) if count else None}
    orders = {name: order_record(target[name]) for name in ("tM", "tD")}
    result = {"candidate_count": count, "chosen": chosen, "score_order": order_record(energy),
              "oracles": orders, "tau": kendall_tau_b(energy, target["tM"]), "errors": errors}
    if chosen is None:
        return {**result, "decision": None}
    return {**result, "decision": {
        "regret_tM": float(target["tM"][chosen]-orders["tM"]["minimum"]),
        "regret_tD": float(target["tD"][chosen]-orders["tD"]["minimum"]),
        "optimal_tM": chosen in orders["tM"]["optima"],
        "optimal_tD": chosen in orders["tD"]["optima"],
        "ari": float(target["ari"][chosen]), "exact": bool(target["exact"][chosen]),
        "k_absolute_error": int(abs(target["k_error"][chosen]))}}


def primary(regret, eligible, *, check=lambda: None):
    """Scene-first four contrasts; caller enforces512scenes and deformed_family.

Input axes: scene, ARMS, ge.CHECKPOINTS, READER_SEEDS. Ineligible rows contain
only NaNs; eligible rows contain all72 finite nonnegative regrets. No nanmean.
Returned arrays must be preserved by the publication adapter, not discarded.
"""
    if (not isinstance(regret, np.ndarray) or regret.dtype != np.float64
            or regret.ndim != 4 or not 1 <= len(regret) <= 512 or regret.shape[1:] != (8, 3, 3)
            or not isinstance(eligible, np.ndarray) or eligible.dtype != np.bool_
            or eligible.shape != (len(regret),) or not np.isfinite(regret[eligible]).all()
            or np.any(regret[eligible] < 0) or not np.isnan(regret[~eligible]).all()):
        raise ValueError("primary requires complete72-cell eligible scenes and explicit empty support")
    scene_ids = np.flatnonzero(eligible).astype(np.int64)
    count = len(scene_ids)
    means = regret[eligible].reshape(count, 8, 9).mean(axis=2, dtype=np.float64)
    def arm(name):
        return means[:, ARMS.index(name)]
    first = arm("geometric_mse")-arm("injection_mse")
    second = arm("geometric_decision")-arm("injection_decision")
    delta = np.column_stack((first, second, second-first,
                            arm("geometric_decision")-arm("decoupled_decision")))
    indices = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED)).integers(
        0, count, size=(BOOTSTRAP_COUNT, count), dtype=np.int64) if count else np.empty((BOOTSTRAP_COUNT, 0), np.int64)
    distribution = np.empty((BOOTSTRAP_COUNT, 4), np.float64)
    if count:
        for begin in range(0, BOOTSTRAP_COUNT, 128):
            check()
            distribution[begin:begin+128] = delta[indices[begin:begin+128]].mean(axis=1, dtype=np.float64)
        interval = np.quantile(distribution, [.00625, .99375], axis=0, method="linear")
    else:
        distribution[:] = np.nan
        interval = None
    summary = {"schema": "geometric-decision-primary-v1", "total_scenes": len(regret),
        "eligible_scenes": count, "empty_scenes": len(regret)-count, "conditioned_cells_per_arm": 9,
        "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_count": BOOTSTRAP_COUNT,
        "confidence_level": .9875, "unit": "scene", "contrasts": {
            name: {"mean": float(delta[:, i].mean()) if count else None,
                   "interval": interval[:, i].tolist() if count else None}
            for i, name in enumerate(CONTRASTS)}}
    return {"summary": summary, "arrays": {"eligible_scene_ids": scene_ids,
        "arm_scene_means": means, "scene_contrasts": delta, "bootstrap_indices": indices,
        "bootstrap_distribution": distribution}}
