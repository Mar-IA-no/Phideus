"""Post-seal evaluation only: indexed-event correspondence and pair recovery.

No sampler, models, fitter, file IO or inference entry points. The campaign must
enforce the global prediction seal before supplying truth to these functions.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score

from .measurement_sensor import vector


def cents_distance(emitted, detected):
    a = vector(emitted, "emitted frequencies", positive=True)
    b = vector(detected, "detected frequencies", positive=True)
    return 1200*np.abs(np.log2(a)[:, None]-np.log2(b)[None, :])


def correspondence(emitted, detected, *, tolerance=20.):
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("invalid tolerance")
    distance = cents_distance(emitted, detected)
    edges = distance <= tolerance
    n, m = edges.shape
    left_degree, right_degree = edges.sum(axis=1), edges.sum(axis=0)
    match = np.full(m, -1, dtype=np.int64)
    left_status = np.where(left_degree == 0, "missing", "ambiguous").astype("U24")
    right_status = np.where(right_degree == 0, "spurious/unassigned", "ambiguous").astype("U24")
    components, seen_left, seen_right = [], set(), set()
    for start in range(n):
        if start in seen_left or not left_degree[start]:
            continue
        left, right, queue = set(), set(), [(0, start)]
        while queue:
            side, node = queue.pop()
            if side == 0:
                if node in left:
                    continue
                left.add(node)
                queue.extend((1, int(j)) for j in np.flatnonzero(edges[node]))
            else:
                if node in right:
                    continue
                right.add(node)
                queue.extend((0, int(i)) for i in np.flatnonzero(edges[:, node]))
        seen_left.update(left)
        seen_right.update(right)
        unique = len(left) == len(right) == 1
        if unique:
            i, j = next(iter(left)), next(iter(right))
            match[j] = i
            left_status[i] = right_status[j] = "unique_tolerance_match"
        components.append({"emitted": sorted(left), "detected": sorted(right),
                           "status": "unique_tolerance_match" if unique else "ambiguous"})
    # Isolated vertices are components too; preserve both sides explicitly.
    components.extend({"emitted": [int(i)], "detected": [], "status": "missing"}
                      for i in np.flatnonzero(left_degree == 0))
    components.extend({"emitted": [], "detected": [int(j)], "status": "spurious/unassigned"}
                      for j in np.flatnonzero(right_degree == 0))
    return {"distance_cents": distance, "edges": edges, "detected_to_emitted": match,
            "emitted_degree": left_degree, "detected_degree": right_degree,
            "emitted_status": left_status, "detected_status": right_status,
            "components": components}


def detection_cost(emitted, detected, *, cutoff=20.):
    """GOSPA-style alpha=2,p=1 on indexed lists, not semantic correspondence.

Rectangular savings assignment avoids a quadratic detected×detected matrix.
Real edges at or above cutoff have zero savings and are never counted as matches.
"""
    if not np.isfinite(cutoff) or cutoff <= 0:
        raise ValueError("invalid cutoff")
    distance = cents_distance(emitted, detected)
    n, m = distance.shape
    savings = np.minimum(distance-cutoff, 0.)
    rows, cols = linear_sum_assignment(np.concatenate((savings, np.zeros((n, n))), axis=1))
    pairs = [(int(i), int(j)) for i, j in zip(rows, cols)
             if j < m and distance[i, j] < cutoff]
    localization = float(sum(distance[i, j] for i, j in pairs))
    missing, spurious = n-len(pairs), m-len(pairs)
    return {"cost": localization + cutoff/2*(missing+spurious),
            "localization": localization, "missing": missing, "spurious": spurious,
            "assignment": pairs}


def labels(values, n):
    x = np.asarray(values)
    if x.shape != (n,) or x.dtype.kind not in "iu":
        raise ValueError("expected integer labels on the declared event set")
    return x


def checked_match(values, n):
    x = np.asarray(values)
    if x.ndim != 1 or x.dtype.kind not in "iu" or np.any((x < -1) | (x >= n)):
        raise ValueError("invalid correspondence indices")
    positive = x[x >= 0]
    if len(np.unique(positive)) != len(positive):
        raise ValueError("correspondence must be one-to-one")
    return x


def pair_count(x):
    _, counts = np.unique(x, return_counts=True)
    return int(np.sum(counts*(counts-1)//2))


def pair_score(truth, match, prediction):
    truth = labels(truth, len(truth))
    match = checked_match(match, len(truth))
    prediction = labels(prediction, len(match))
    total, proposed = pair_count(truth), pair_count(prediction)
    good = np.flatnonzero(match >= 0)
    joint = np.column_stack((truth[match[good]], prediction[good]))
    _, counts = np.unique(joint, axis=0, return_counts=True)
    tp = int(np.sum(counts*(counts-1)//2))
    f = 1. if total+proposed == 0 else 2.*tp/(total+proposed)
    return {"T": total, "P": proposed, "TP": tp, "F": f}


def evaluate_candidates(truth, match, candidates, selected):
    """Score the actual shared universe; requires nonzero truth-pair support.

The campaign sampler has T>0. pair_score separately defines the T=P=0 fixture.
Empty/unsupported universes pass [] and None, never a fabricated partition.
"""
    truth = labels(truth, len(truth))
    match = checked_match(match, len(truth))
    total = pair_count(truth)
    if total <= 0:
        raise ValueError("campaign decomposition requires T>0")
    recoverable = pair_count(truth[match[match >= 0]])
    upper = 2.*recoverable/(total+recoverable)
    scores = [pair_score(truth, match, p) for p in candidates]
    if not scores:
        if selected is not None:
            raise ValueError("selected index in an empty universe")
        coverage, f = 0., 0.
    else:
        if type(selected) is not int or not 0 <= selected < len(scores):
            raise ValueError("selection must index the evaluated universe")
        coverage, f = max(s["F"] for s in scores), scores[selected]["F"]
    if not 0 <= f <= coverage <= upper+1e-14 or upper > 1:
        raise ValueError("inconsistent F/C/U bounds")
    return {"T": total, "R": recoverable, "U": upper, "C": coverage, "F": f,
            "candidate_scores": scores,
            "oracle_indices": [i for i, s in enumerate(scores) if s["F"] == coverage],
            "correspondence_loss": 1-upper, "coverage_loss": upper-coverage,
            "choice_loss": coverage-f}


def common_support_metrics(truth, match, prediction):
    truth = labels(truth, len(truth))
    match = checked_match(match, len(truth))
    prediction = labels(prediction, len(match))
    detected = np.flatnonzero(match >= 0)
    detected = detected[np.argsort(match[detected], kind="stable")]
    emitted = match[detected]
    n = len(emitted)
    def recode(x):
        codes = {}
        return np.array([codes.setdefault(int(v), len(codes)) for v in x], dtype=np.int64)
    a, b = recode(truth[emitted]), recode(prediction[detected])
    result = {"n": n, "emitted_indices": emitted, "detected_indices": detected,
              "truth_labels": a, "prediction_labels": b, "ARI": None, "VI_bits": None}
    if n < 2:
        return result
    table = np.zeros((int(a.max())+1, int(b.max())+1), dtype=np.int64)
    np.add.at(table, (a, b), 1)
    row, col = table.sum(axis=1), table.sum(axis=0)
    i, j = np.nonzero(table)
    cells = table[i, j].astype(np.float64)
    vi = np.sum(cells/n*(np.log2(row[i]/cells)+np.log2(col[j]/cells)))
    result.update(ARI=float(adjusted_rand_score(a, b)), VI_bits=float(vi))
    return result
