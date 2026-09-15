"""Pure summaries for post-replay reporting; no IO, model or truth capability."""
from __future__ import annotations

import numpy as np


def event_universe(partitions, rank_to_event):
    """Compare topology by persistent event identity, never by frequency/rank."""
    order = np.asarray(rank_to_event)
    if (order.ndim != 1 or order.dtype.kind not in "iu" or not len(order)
            or sorted(order.tolist()) != list(range(len(order)))):
        raise ValueError("event identity requires a complete bijection")
    result = []
    for partition in partitions:
        if (not partition or any(not group for group in partition)
                or any(type(i) not in (int, np.int64, np.int32) for group in partition for i in group)
                or sorted(i for group in partition for i in group) != list(range(len(order)))):
            raise ValueError("partition must cover every event exactly once")
        result.append(tuple(sorted(tuple(sorted(int(order[i]) for i in group)) for group in partition)))
    if len(set(result)) != len(result):
        raise ValueError("duplicate candidate topology")
    return result


def roundtrip_comparison(before, after, *, energy_before, energy_after,
                         evidence_before, evidence_after, choice_before, choice_after):
    """Use only paired candidates for numeric differences; retain support loss."""
    first = event_universe(before["partitions"], before["canonical_to_observed"])
    second = event_universe(after["partitions"], after["canonical_to_observed"])
    if len(before["canonical_to_observed"]) != len(after["canonical_to_observed"]):
        raise ValueError("roundtrip cannot replace the event universe")
    for keys, energy, evidence, chosen in ((first, energy_before, evidence_before, choice_before),
                                         (second, energy_after, evidence_after, choice_after)):
        if (not isinstance(energy, np.ndarray) or energy.dtype != np.float64 or energy.shape != (len(keys),)
                or not np.isfinite(energy).all() or not isinstance(evidence, np.ndarray)
                or evidence.dtype != np.float32 or evidence.shape != (len(keys), 8) or not np.isfinite(evidence).all()):
            raise ValueError("paired report requires delivered channels and preserved energy")
        if (not keys and chosen is not None) or (keys and (type(chosen) is not int or not 0 <= chosen < len(keys))):
            raise ValueError("choice does not belong to its candidate support")
        if keys and energy[chosen] != np.min(energy):
            raise ValueError("reported choice is not a preserved minimum")
    a, b = {key: i for i, key in enumerate(first)}, {key: i for i, key in enumerate(second)}
    common = sorted(a.keys() & b.keys())
    left = np.asarray([a[key] for key in common], np.int64)
    right = np.asarray([b[key] for key in common], np.int64)
    channel_error = (np.abs(evidence_before[left].astype(np.float64)-evidence_after[right].astype(np.float64))
                     if common else np.empty((0, 8), np.float64))
    return {"before_candidates": len(first), "after_candidates": len(second), "common_candidates": len(common),
        "lost_candidates": sorted(a.keys()-b.keys()), "added_candidates": sorted(b.keys()-a.keys()),
        "common_event_partitions": common, "before_indices": left.tolist(), "after_indices": right.tolist(),
        "before_empty": not first, "after_empty": not second,
        "max_energy_error_common": float(np.max(np.abs(energy_before[left]-energy_after[right]))) if common else None,
        "max_channel_error_common": channel_error.max(axis=0).tolist() if common else [None]*8,
        "same_exact_choice_by_event": None if choice_before is None or choice_after is None else first[choice_before] == second[choice_after],
        "before_choice_by_event": None if choice_before is None else first[choice_before],
        "after_choice_by_event": None if choice_after is None else second[choice_after],
        "scope": "joint quantization/recentering effect; common-candidate differences are conditioned on support"}


def coordinate_comparison(coordinates):
    """Pairwise log-ratio differences in delivered event order, no new q values."""
    dtypes = {"original": np.float32, "shifted64": np.float64, "shifted32": np.float32,
              "q_center": np.float32, "q_probe": np.float32}
    if set(coordinates) != set(dtypes):
        raise ValueError("coordinate report requires the preserved roundtrip fields")
    n = len(coordinates["original"])
    for key, dtype in dtypes.items():
        value = coordinates[key]
        if not isinstance(value, np.ndarray) or value.dtype != dtype or value.shape != (n,) or n < 2 or not np.isfinite(value).all():
            raise ValueError("coordinate report dtype, support or finiteness differs")
    def relations(q):
        q = q.astype(np.float64)
        return q[:, None]-q[None, :]
    pairs = (("original", "shifted64"), ("shifted64", "shifted32"),
             ("shifted32", "q_probe"), ("original", "q_probe"), ("original", "q_center"))
    return {"event_count": n,
        "tie_counts": {name: n-len(np.unique(value)) for name, value in coordinates.items()},
        "max_pairwise_log_ratio_change": {a+"_to_"+b: float(np.max(np.abs(relations(coordinates[a])-relations(coordinates[b]))))
                                          for a, b in pairs},
        "scope": "numerical coordinate changes; q_center is diagnostic only, not another pipeline"}
