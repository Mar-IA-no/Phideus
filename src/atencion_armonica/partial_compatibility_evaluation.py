"""Fixed reader without true cardinality; metrics will retain per-scene units."""
from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform


def read_partition(probabilities, threshold: float):
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim != 2 or p.shape[0] != p.shape[1] or len(p) < 2:
        raise ValueError("expected a square probability matrix")
    if not np.isfinite(p).all() or np.any(p < 0) or np.any(p > 1):
        raise ValueError("invalid probabilities")
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("invalid distance threshold")
    distance = 1-(p+p.T)/2
    np.fill_diagonal(distance, 0)
    tree = linkage(squareform(distance, checks=True), method="average", optimal_ordering=False)
    labels = fcluster(tree, t=threshold, criterion="distance")
    return tuple(sorted(tuple(np.flatnonzero(labels == label).tolist()) for label in set(labels)))
