"""Scene-level metrics and validation-only reader selection for the frozen study."""
from __future__ import annotations

import numpy as np
from scipy.special import expit, xlogy
from sklearn.metrics import adjusted_rand_score, average_precision_score, roc_auc_score

from .partial_compatibility_evaluation import read_partition

THRESHOLDS = tuple(i/20 for i in range(1, 20))


def partition_labels(partition, n):
    labels = np.full(n, -1, dtype=np.int64)
    for label, members in enumerate(partition):
        for member in members:
            if not 0 <= member < n or labels[member] != -1:
                raise ValueError("not a partition of this scene")
            labels[member] = label
    if np.any(labels < 0):
        raise ValueError("partition omits observations")
    return labels


def pair_metrics(p, y, logits=None):
    p, y = np.asarray(p, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if p.shape != y.shape or p.ndim != 1 or not np.isfinite(p).all() or np.any((p < 0)|(p > 1)):
        raise ValueError("invalid pair probabilities or targets")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("targets must be binary")
    if not len(p):
        return {"n_pairs": 0, "status": "NO_PAIRS"}
    positive = y == 1
    predicted = p >= .5
    both = len(np.unique(y)) == 2
    # The analytic reference may produce exact 0/1. Its BCE alone uses this
    # explicit numerical clip; neural BCE uses preserved logits without clipping.
    if logits is None:
        clipped = np.clip(p, 1e-12, 1-1e-12)
        bce = -np.mean(y*np.log(clipped)+(1-y)*np.log1p(-clipped))
    else:
        z = np.asarray(logits, dtype=np.float64)
        # The preserved logits/probabilities can be float32; promoting them does
        # not undo sigmoid rounding. Permit that representation error only.
        precision = np.finfo(np.float32)
        if (z.shape != p.shape or not np.isfinite(z).all()
                or not np.allclose(expit(z), p, atol=precision.tiny, rtol=4*precision.eps)):
            raise ValueError("logits and probabilities disagree")
        bce = np.mean(np.logaddexp(0, z)-y*z)
    return {"n_pairs": len(p), "status": "EVALUABLE", "brier": float(np.mean((p-y)**2)),
            "bce": float(bce), "target_prevalence": float(y.mean()),
            "mean_probability": float(p.mean()), "predicted_prevalence_at_half": float(predicted.mean()),
            "positive_recall_at_half": float(predicted[positive].mean()) if positive.any() else None,
            "constant_hard_prediction_at_half": bool(np.all(predicted == predicted[0])),
            "constant_probability": bool(np.all(p == p[0])),
            "ap": float(average_precision_score(y, p)) if both else None,
            "auc": float(roc_auc_score(y, p)) if both else None,
            "ranking_status": "EVALUABLE" if both else "ONE_TARGET_CLASS",
            "binary_entropy_nats": float(np.mean(-xlogy(p, p)-xlogy(1-p, 1-p)))}


def scene_metrics(probabilities, source_ids, q, threshold, *, logits=None):
    p = np.asarray(probabilities, dtype=np.float64)
    truth, q = np.asarray(source_ids), np.asarray(q, dtype=np.float32).astype(np.float64)
    n = len(q)
    if p.shape != (n, n) or truth.shape != (n,) or not np.isfinite(q).all():
        raise ValueError("scene shapes or coordinates invalid")
    partition = read_partition(p, threshold)
    labels = partition_labels(partition, n)
    mask = ~np.eye(n, dtype=bool)  # Both orientations, exactly as in training.
    y = truth[:, None] == truth[None, :]
    near = (np.abs(q[:, None]-q[None, :])*1200/np.log(2) <= 10) & mask
    z = None if logits is None else np.asarray(logits, dtype=np.float64)
    if z is not None and z.shape != p.shape:
        raise ValueError("wrong logit shape")
    return {"pairs": pair_metrics(p[mask], y[mask], None if z is None else z[mask]),
            "near_collision_pairs": pair_metrics(p[near], y[near], None if z is None else z[near]),
            "ari": float(adjusted_rand_score(truth, labels)),
            "exact_partition": bool(np.array_equal(labels[:, None] == labels[None, :], y)),
            "k_inferred": len(partition), "k_true_for_evaluation": len(np.unique(truth)),
            "k_error": len(partition)-len(np.unique(truth)),
            "k_absolute_error": abs(len(partition)-len(np.unique(truth))),
            "partition": [list(group) for group in partition],
            "global_ambiguity_status": "UNADJUDICATED"}


def select_reader(probability_matrices, source_id_lists, *, split):
    if split != "validation":
        raise PermissionError("reader selection is validation-only")
    if not probability_matrices or len(probability_matrices) != len(source_id_lists):
        raise ValueError("missing or misaligned validation scenes")
    scores = []
    for threshold in THRESHOLDS:
        values = [adjusted_rand_score(truth, partition_labels(read_partition(p, threshold), len(truth)))
                  for p, truth in zip(probability_matrices, source_id_lists)]
        scores.append(float(np.mean(values)))
    chosen = int(np.argmax(scores))  # Ordered grid gives the lowest exact tie.
    return {"threshold": THRESHOLDS[chosen], "threshold_grid": list(THRESHOLDS),
            "mean_ari_grid": scores, "selection_split": split,
            "scene_count": len(probability_matrices)}


def paired_bootstrap(candidate, control, *, control_index, seed):
    a, b = np.asarray(candidate, np.float64), np.asarray(control, np.float64)
    if a.shape != (1024,) or b.shape != a.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("primary bootstrap requires all 1024 paired scenes")
    if control_index not in (0, 1, 2) or seed not in (0, 2026090721, 2026090722, 2026090723):
        raise ValueError("undeclared comparison or training seed")
    delta = a-b
    rng = np.random.default_rng(np.random.SeedSequence([2026090731, control_index, seed]))
    indices = rng.integers(0, 1024, size=(2000, 1024))
    samples = delta[indices].mean(axis=1)
    low, high = np.percentile(samples, [2.5, 97.5])
    return {"mean_delta": float(delta.mean()), "percentile_95": [float(low), float(high)],
            "n_scenes": 1024, "resamples": 2000, "control_index": control_index, "seed": seed,
            "interval_status": "DESCRIPTIVE_NOT_MULTIPLICITY_ADJUSTED",
            "delta_direction": "compatibility_minus_control"}


def seed_disagreement(probability_matrices):
    """Per-scene population SD of probabilities plus hard-pair disagreement."""
    values = np.asarray(probability_matrices, dtype=np.float64)
    if (values.ndim != 3 or values.shape[0] != 3 or values.shape[1] != values.shape[2]
            or values.shape[1] < 2 or not np.isfinite(values).all()
            or np.any((values < 0)|(values > 1))):
        raise ValueError("expected three seed matrices for the same scene")
    mask = ~np.eye(values.shape[1], dtype=bool)
    flat = values[:, mask]
    hard = flat >= .5
    return {"mean_probability_sd_across_three_seeds": float(flat.std(axis=0, ddof=0).mean()),
            "fraction_pairs_with_hard_seed_disagreement": float(np.any(hard != hard[0], axis=0).mean()),
            "n_pairs": int(mask.sum()), "seed_count": 3}
