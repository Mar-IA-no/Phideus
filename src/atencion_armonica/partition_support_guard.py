"""Explicit diagnostic-only float64 mass guard; all scientific helpers remain frozen."""
from __future__ import annotations
import json
import numpy as np
from .learned_partition_core import model_inputs
from .learned_partition_readout import prediction_dependence
from .learned_partition_inference import intervention_inputs


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
            or not np.allclose(incidence.sum(axis=1, dtype=np.float64), 1, rtol=0, atol=2e-7)):
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


def support_by_size(row, original, changed):
    support = input_support(original, changed)
    mask = np.asarray(support["changed_group_mask"])
    sizes = np.asarray([len(g) for g in row.groups])
    support["by_group_size"] = {str(size): {"group_count": int(np.sum(sizes == size)),
        "changed_group_count": int(mask[sizes == size].sum()),
        "changed_group_fraction": float(mask[sizes == size].mean())} for size in np.unique(sizes)}
    return support


def intervention_report(row, original_input, changed_input, original_prediction, changed_prediction):
    """No truth: input support and empirical dependence, not semantic validity."""
    return {"input": support_by_size(row, original_input, changed_input),
            "prediction": prediction_dependence(original_prediction, changed_prediction, row.candidates)}


def replay_support(rows, normalizer, arm, intervention, original_predictions, changed_predictions):
    """Recompute diagnostics independently of stored reports, without training."""
    if not rows or len(rows) != len(original_predictions) or len(rows) != len(changed_predictions):
        raise ValueError("support replay requires the complete aligned scene roster")
    reports = [intervention_report(row, model_inputs(row, normalizer, arm),
        intervention_inputs(row, normalizer, arm, intervention), original, changed)
        for row, original, changed in zip(rows, original_predictions, changed_predictions)]
    # Candidate signatures are tuples in memory and lists in the JSON artifact.
    return json.loads(json.dumps(reports, allow_nan=False))
