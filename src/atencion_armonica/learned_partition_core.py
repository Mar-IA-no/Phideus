"""NumPy partition supervision and observable tensorization; no draws or Torch.

The supervision port is explicit and separate from observable_features.
Stage authorization and provenance belong to the future campaign gate.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from .structured_source_reader import signature

ARMS = ("pairs_structure", "local_compatibility", "shared_source", "decoupled_source")
READER_SEEDS = (2026090891, 2026090892, 2026090893)
BLOCK_IDS = (10, 11, 20, 21, 30, 40)


def initialization_seeds():
    rows = [{"reader_seed": s, "block_id": b,
             "torch_seed": int(np.random.SeedSequence([s, b]).generate_state(1, dtype=np.uint64)[0])}
            for s in READER_SEEDS for b in BLOCK_IDS]
    if len({r["torch_seed"] for r in rows}) != len(rows):
        raise ValueError("initialization stream collision")
    return rows


def partition_errors(partition, source_ids):
    """Supervision-only H(P|Y), H(Y|P), in nats and divided by log N."""
    y = np.asarray(source_ids)
    if y.ndim != 1 or not 3 <= len(y) <= 32 or y.dtype.kind not in "iu":
        raise ValueError("expected integer supervision on 3..32 events")
    flat = [i for g in partition for i in g]
    if (not partition or any(not g for g in partition)
            or any(type(i) not in (int, np.int64) for i in flat)
            or sorted(flat) != list(range(len(y)))):
        raise ValueError("partition must cover every event exactly once")
    _, labels = np.unique(y, return_inverse=True)
    n = len(y)
    counts = np.asarray([np.bincount(labels[list(g)], minlength=labels.max()+1)
                         for g in partition], dtype=np.float64)
    group, source = counts.sum(axis=1), counts.sum(axis=0)
    a, b = np.nonzero(counts)
    mass = counts[a, b]/n
    raw = np.array([-np.sum(mass*np.log(counts[a, b]/source[b])),
                    -np.sum(mass*np.log(counts[a, b]/group[a]))], dtype=np.float64)
    raw[raw == 0] = 0.  # Canonicalize signed zero, not small positive errors.
    return {"raw": raw, "normalized": raw/np.log(n)}


@dataclass(frozen=True)
class FeatureRows:
    n: int
    groups: tuple
    candidates: tuple
    group_features: np.ndarray
    global_features: np.ndarray
    incidence: np.ndarray
    costs: dict


def observable_features(scored, logits):
    """Consume only a scored observable pool and its original-order logits."""
    order = np.asarray(scored["pool"]["canonical_to_observed"])
    z = np.asarray(logits)
    n = len(order)
    if (not 3 <= n <= 32 or order.dtype.kind not in "iu"
            or not np.array_equal(np.sort(order), np.arange(n))
            or z.shape != (n, n) or z.dtype != np.float32
            or not np.isfinite(z).all() or not np.array_equal(z, z.T)):
        raise ValueError("invalid canonical mapping or observed symmetric float32 logits")
    z = z.astype(np.float64)[np.ix_(order, order)]
    groups = tuple(tuple(g["members"]) for g in scored["groups"])
    candidates = tuple(signature(p["signature"]) for p in scored["candidates"])
    if not groups or list(groups) != sorted(set(groups)) or not candidates or list(candidates) != sorted(set(candidates)):
        raise ValueError("expected ordered unique groups and candidates")
    common, costs = [], {a: [] for a in ARMS[1:]}
    for group, record in zip(groups, scored["groups"]):
        if (not 1 <= len(group) <= 8 or tuple(sorted(set(group))) != group
                or any(type(i) not in (int, np.int64) or not 0 <= i < n for i in group)
                or record["size"] != len(group)):
            raise ValueError("invalid group or size")
        m = len(group)
        within = np.asarray([z[i, j] for i, j in combinations(group, 2)], dtype=np.float64)
        stats = [within.mean(), within.std(ddof=0), within.min(), within.max()] if len(within) else [0.]*4
        common.append([m/8, m/n, m*(m-1)/(n*(n-1)), *stats, float(m < 3)])
        for arm in costs:
            value = float(record["costs"][arm])
            if not np.isfinite(value) or not 0 <= value <= 1 or (m < 3 and value != 0):
                raise ValueError("invalid bounded group factor")
            costs[arm].append(value)
    incidence = np.zeros((len(candidates), len(groups)), np.float32)
    global_rows, used = [], set()
    lookup = {g: i for i, g in enumerate(groups)}
    for c, (partition, record) in enumerate(zip(candidates, scored["candidates"])):
        if sorted(i for g in partition for i in g) != list(range(n)):
            raise ValueError("candidate is not a partition")
        ids = [lookup[g] for g in partition]
        if record["group_ids"] != ids:
            raise ValueError("candidate group incidence differs")
        used.update(ids)
        incidence[c, ids] = np.asarray([len(g)/n for g in partition], np.float32)
        energy = -sum(float(z[i, j]) for g in partition for i, j in combinations(g, 2))/n
        if energy != record["pair_energy"]:
            raise ValueError("pair energy differs from actual logits")
        global_rows.append([n/32, len(partition)/n, energy,
                            sum((len(g)/n)**2 for g in partition),
                            sum(len(g) for g in partition if len(g) == 1)/n,
                            sum(len(g) for g in partition if len(g) < 3)/n])
    if used != set(range(len(groups))):
        raise ValueError("unused group outside the common candidate pool")
    return FeatureRows(n, groups, candidates, np.asarray(common, np.float64),
                       np.asarray(global_rows, np.float64), incidence,
                       {a: np.asarray(v, np.float64) for a, v in costs.items()})


def fit_normalizer(rows, *, expected_count):
    """Train-only caller supplies exact roster; equal scene mass, not group mass."""
    if not rows or len(rows) != expected_count:
        raise ValueError("incomplete train roster")
    group_mean = np.mean([r.group_features[:, 3:7].mean(axis=0) for r in rows], axis=0)
    group_var = np.mean([((r.group_features[:, 3:7]-group_mean)**2).mean(axis=0) for r in rows], axis=0)
    energy_mean = float(np.mean([r.global_features[:, 2].mean() for r in rows]))
    energy_var = float(np.mean([((r.global_features[:, 2]-energy_mean)**2).mean() for r in rows]))
    mean = np.r_[group_mean, energy_mean]
    scale = np.sqrt(np.r_[group_var, energy_var])
    zero = scale == 0
    scale[zero] = 1.
    if not np.isfinite(mean).all() or not np.isfinite(scale).all():
        raise ValueError("nonfinite normalizer")
    return {"mean": mean, "scale": scale, "zero_variance": zero, "scene_count": expected_count}


def model_inputs(row, normalizer, arm):
    if arm not in ARMS:
        raise ValueError("unknown reader arm")
    mean, scale = np.asarray(normalizer["mean"]), np.asarray(normalizer["scale"])
    if mean.shape != (5,) or scale.shape != (5,) or not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("invalid frozen normalizer")
    groups, global_rows = row.group_features.copy(), row.global_features.copy()
    groups[:, 3:7] = (groups[:, 3:7]-mean[:4])/scale[:4]
    global_rows[:, 2] = (global_rows[:, 2]-mean[4])/scale[4]
    if arm != ARMS[0]:
        groups = np.column_stack((groups, row.costs[arm]))
    result = {"groups": groups.astype(np.float32), "globals": global_rows.astype(np.float32),
              "incidence": row.incidence.copy()}
    if any(not np.isfinite(v).all() for v in result.values()):
        raise ValueError("nonfinite reader input")
    return result
