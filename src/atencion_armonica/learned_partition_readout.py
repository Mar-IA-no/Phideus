"""Observable input support and readout of preserved float32 learned costs."""
from __future__ import annotations

import numpy as np

from .structured_source_reader import signature


def choose_costs(predictions, candidates):
    values = np.asarray(predictions)
    canonical = [signature(p) for p in candidates]
    if (values.dtype != np.float32 or values.shape != (len(canonical), 2)
            or not len(canonical) or not np.isfinite(values).all() or np.any(values < 0)
            or len(set(canonical)) != len(canonical)):
        raise ValueError("expected finite nonnegative float32 costs for unique candidates")
    sums = values.sum(axis=1, dtype=np.float32)
    if not np.isfinite(sums).all():
        raise ValueError("nonfinite total predicted cost")
    minimum = sums.min()
    ties = np.flatnonzero(sums == minimum)
    chosen = min(ties.tolist(), key=lambda i: canonical[i])
    larger = sums[sums > minimum]
    return {"candidate_index": chosen, "signature": canonical[chosen],
            "predicted_components": values[chosen].tolist(), "cost": float(minimum),
            "co_minimum_count": len(ties),
            "next_level_gap": float(np.float64(larger.min())-minimum) if len(larger) else None}


def input_support(original, changed):
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
    # Float32 row values are identical to np.r_[row, float32_weight], without
    # allocating that array again for every incidence entry of every candidate.
    left_rows = [tuple(float(v) for v in row) for row in a]
    right_rows = [tuple(float(v) for v in row) for row in b]
    aligned, effective = [], []
    for weights in incidence:
        ids = np.flatnonzero(weights > 0)
        aligned.append(bool(np.any(group_changed[ids])))
        left = sorted(left_rows[i]+(float(weights[i]),) for i in ids)
        right = sorted(right_rows[i]+(float(weights[i]),) for i in ids)
        effective.append(left != right)
    return {"status": "INPUT_CHANGED" if any(effective) else "INPUT_UNCHANGED",
            "group_count": len(a), "candidate_count": len(incidence),
            "changed_group_mask": group_changed.tolist(), "aligned_candidate_mask": aligned,
            "effective_candidate_mask": effective,
            "changed_group_fraction": float(group_changed.mean()),
            "aligned_candidate_fraction": float(np.mean(aligned)),
            "effective_candidate_fraction": float(np.mean(effective))}


def prediction_dependence(original, changed, candidates):
    first, second = choose_costs(original, candidates), choose_costs(changed, candidates)
    delta = changed.astype(np.float64)-original.astype(np.float64)
    delta_sum = (changed.sum(axis=1, dtype=np.float32).astype(np.float64)
                 -original.sum(axis=1, dtype=np.float32).astype(np.float64))
    return {"component_delta": delta.tolist(), "sum_delta": delta_sum.tolist(),
            "sum_delta_min": float(delta_sum.min()), "sum_delta_max": float(delta_sum.max()),
            "constant_sum_delta": bool(np.all(delta_sum == delta_sum[0])),
            "maximum_absolute_component_delta": float(np.abs(delta).max()),
            "decision_changed": first["signature"] != second["signature"],
            "original_choice": first, "changed_choice": second}
