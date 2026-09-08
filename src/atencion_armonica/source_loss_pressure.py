"""Post-hoc derivatives in symmetric edge-logit coordinates, NumPy only."""
from __future__ import annotations

from itertools import combinations

import numpy as np


def edge_pressure(logits, triples, physical_weights, sham_weights, *, sham_evaluable):
    z = np.asarray(logits, dtype=np.float64)
    if (z.ndim != 2 or z.shape[0] != z.shape[1] or not 3 <= len(z) <= 32
            or not np.isfinite(z).all() or not np.array_equal(z, z.T)):
        raise ValueError("expected finite exactly symmetric logits for 3..32 events")
    n = len(z)
    expected = np.array(list(combinations(range(n), 3)), dtype=np.int64)
    if not np.array_equal(triples, expected):
        raise ValueError("triples must be the complete canonical unordered roster")
    weights = [np.asarray(w, dtype=np.float32).astype(np.float64) for w in (physical_weights, sham_weights)]
    if any(w.shape != (len(expected),) or not np.isfinite(w).all() or np.any((w < 0)|(w > 1)) for w in weights):
        raise ValueError("invalid training-quantized weights")
    if not isinstance(sham_evaluable, (bool, np.bool_)) or (not sham_evaluable and np.any(weights[1] != 0)):
        raise ValueError("non-evaluable sham must have zero weights")
    edges = np.array(list(combinations(range(n), 2)), dtype=np.int64)
    edge_index = np.full((n, n), -1, dtype=np.int64)
    edge_index[edges[:, 0], edges[:, 1]] = np.arange(len(edges))
    i, j, k = expected.T
    triple_edges = np.stack((edge_index[i, j], edge_index[i, k], edge_index[j, k]), axis=1)
    edge_logits = z[edges[:, 0], edges[:, 1]]
    p = np.exp(-np.logaddexp(0., -edge_logits))
    pt = p[triple_edges]
    clique = pt.prod(axis=1)
    raw = {"edges": edges, "triples": expected, "edge_logits": edge_logits, "probabilities": p,
           "clique_probability": clique, "sham_evaluable": bool(sham_evaluable)}
    for name, w in zip(("physical", "sham"), weights):
        derivative = np.zeros(len(edges), dtype=np.float64)
        for coordinate, others in enumerate(((1, 2), (0, 2), (0, 1))):
            e = triple_edges[:, coordinate]
            contribution = w*p[e]*(1-p[e])*pt[:, others[0]]*pt[:, others[1]]/len(expected)
            np.add.at(derivative, e, contribution)
        raw[name+"_weights"] = w
        raw[name+"_derivative"] = derivative
        raw[name+"_contributions"] = w*clique
    return raw


def evaluate_pressure(raw, source_ids):
    """Truth is used here for evaluation only, after observable derivatives."""
    labels = np.asarray(source_ids)
    n = int(raw["edges"].max())+1
    if labels.shape != (n,) or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("expected one integer source ID per observation")
    edges, triples = raw["edges"], raw["triples"]
    y = labels[edges[:, 0]] == labels[edges[:, 1]]
    true_triple = (labels[triples[:, 0]] == labels[triples[:, 1]]) & (labels[triples[:, 1]] == labels[triples[:, 2]])
    bce = (raw["probabilities"]-y)/len(edges)
    result = {"n_edges": len(edges), "n_triples": len(triples), "sham_evaluable": raw["sham_evaluable"],
              "coordinate_authority": "SYMMETRIC_EDGE_LOGIT_NOT_NETWORK_PARAMETER", "classes": {}, "penalties": {}}
    for c in (0, 1):
        mask = y == c
        B = float(np.abs(bce[mask]).sum())
        P = float(.1*raw["physical_derivative"][mask].sum())
        S = float(.1*raw["sham_derivative"][mask].sum())
        status = "NO_CLASS" if not mask.any() else "ZERO_BCE_DENOMINATOR" if B == 0 else "EVALUABLE"
        result["classes"][str(c)] = {"edge_count": int(mask.sum()), "B": B, "P": P, "S": S,
                                     "P_over_B": P/B if B > 0 else None,
                                     "S_over_B": S/B if B > 0 and raw["sham_evaluable"] else None,
                                     "P_ratio_status": status,
                                     "S_ratio_status": status if raw["sham_evaluable"] else "SHAM_NOT_EVALUABLE"}
    for name in ("physical", "sham"):
        weights, contribution = raw[name+"_weights"], raw[name+"_contributions"]
        total = float(contribution.sum())
        enabled = name != "sham" or raw["sham_evaluable"]
        result["penalties"][name] = {
            "status": "EVALUABLE" if enabled else "SHAM_NOT_EVALUABLE",
            "L": total/len(triples),
            "F_true": float(contribution[true_triple].sum()/total) if total > 0 else None,
            "F_true_status": ("SHAM_NOT_EVALUABLE" if not enabled else
                              "EVALUABLE" if total > 0 else "ZERO_TOTAL_CONTRIBUTION"),
            "L_truth": float(weights[true_triple].sum()/len(triples)),
            "true_triple_count": int(true_triple.sum()), "mixed_triple_count": int((~true_triple).sum()),
            "mean_weight_true": float(weights[true_triple].mean()) if true_triple.any() else None,
            "mean_weight_mixed": float(weights[~true_triple].mean()) if (~true_triple).any() else None}
    # Preserve evaluation arrays separately from the observable derivative API.
    evaluation = {"edge_targets": y, "true_triple": true_triple, "bce_derivative": bce}
    return result, evaluation
