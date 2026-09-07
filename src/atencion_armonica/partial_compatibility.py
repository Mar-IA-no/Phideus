"""Observable triple residual, matched sham, and frequency-only descriptors.

The residual is an endpoint-conditioned fit, not an exact manifold distance or
an identifiability certificate. This module imports no torch or dataset labels.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np

from .peak_tokens import _simple_ratio_grid, _harmonic_log_grid

INDEX_TRIPLES = np.asarray(list(combinations(range(1, 9), 3)), dtype=np.float64)
BETA_LO, BETA_HI = 1e-5, 2e-2
LOG_TO_CENTS = 1200.0 / np.log(2.0)


def observed_vector(log_f):
    q = np.asarray(log_f)
    if q.dtype != np.float32 or q.ndim != 1 or not 3 <= len(q) <= 32:
        raise ValueError("expected 3..32 float32 observed log frequencies")
    if not np.all(np.isfinite(q)):
        raise ValueError("nonfinite log frequencies")
    return q.astype(np.float64)


def _difference(u, a, beta):
    return np.log(u/a) + 0.5*(np.log1p(beta*u*u)-np.log1p(beta*a*a))


def triple_geometry(log_f):
    q = observed_vector(log_f)
    triples = np.asarray(list(combinations(range(len(q)), 3)), dtype=np.int64)
    values = np.sort(q[triples], axis=1)
    middle = (values[:, 1]-values[:, 0])[:, None]
    outer = (values[:, 2]-values[:, 0])[:, None]
    a, b, c = INDEX_TRIPLES.T
    lower, upper = _difference(c, a, BETA_LO), _difference(c, a, BETA_HI)
    low, high = outer <= lower, outer >= upper
    interior = ~(low | high)
    beta = np.where(low, BETA_LO, BETA_HI)*np.ones((len(triples), 56))
    # Never evaluate the endpoint quotient outside its representable interval.
    rows, cols = np.nonzero(interior)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        t = np.exp(2*outer[rows, 0])*(a[cols]/c[cols])**2
        beta[rows, cols] = np.clip((t-1)/(c[cols]**2-t*a[cols]**2), BETA_LO, BETA_HI)
        e_b = middle-_difference(b, a, beta)
        e_c = outer-_difference(c, a, beta)
        residuals = LOG_TO_CENTS*np.sqrt((e_b**2+e_c**2)/2)
        best = residuals.argmin(axis=1)
        residual = residuals[np.arange(len(triples)), best]
        weight = residual**2/(residual**2+100.0)
    if not np.all(np.isfinite(residuals)) or not np.all(np.isfinite(beta)):
        raise FloatingPointError("nonfinite triple geometry")
    support = np.zeros((len(q), len(q)), dtype=np.float64)
    for x, y in ((0, 1), (0, 2), (1, 2)):
        np.add.at(support, (triples[:, x], triples[:, y]), 1-weight)
    support = (support+support.T)/(len(q)-2)
    return {"triples": triples, "residual_cents": residual, "weights": weight,
            "argmin_index_triple": best, "pair_support": support}


def sham_geometry(log_f, geometry, *, split_seed: int, scene_id: int):
    q = observed_vector(log_f)
    for value in (split_seed, scene_id):
        if type(value) is not int or value < 0:
            raise ValueError("sham metadata must be nonnegative integers")
    order = np.argsort(q, kind="stable")
    triples = geometry["triples"]
    weights = geometry["weights"]
    if len(np.unique(q)) != len(q) or len(triples) < 2:
        return {"weights": np.zeros_like(weights), "evaluable": False,
                "shift": None, "canonical_to_delivered": None}
    lookup = {tuple(t): i for i, t in enumerate(triples.tolist())}
    mapping = np.asarray([lookup[tuple(sorted(order[t]))]
                          for t in triples], dtype=np.int64)
    canonical_weights = weights[mapping]
    rng = np.random.default_rng(np.random.SeedSequence([2026090730, split_seed, scene_id]))
    shift = int(rng.integers(1, len(triples)))
    result = np.empty_like(weights)
    result[mapping] = canonical_weights[(np.arange(len(triples))+shift) % len(triples)]
    return {"weights": result, "evaluable": True, "shift": shift,
            "canonical_to_delivered": mapping}


def frequency_features(log_f):
    """Only the exact float32 observation enters the model's feature builder."""
    q = observed_vector(log_f)
    geometry = triple_geometry(log_f)
    dlogf = np.abs(q[:, None]-q[None, :])
    grid, _ = _simple_ratio_grid()
    differences = np.abs(dlogf[:, :, None]-grid)
    classes = differences.argmin(axis=2).astype(np.int64)
    ratio_residual = differences.min(axis=2)
    h = q[:, None]-_harmonic_log_grid()
    common = np.abs(h[:, None, :, None]-h[None, :, None, :]).min(axis=(2, 3))
    pair_cont = np.stack((dlogf, ratio_residual, common, geometry["pair_support"]), axis=-1)
    tokens = np.stack((q, np.zeros_like(q)), axis=-1).astype(np.float32)
    return {"tokens": tokens, "pair_cont": pair_cont.astype(np.float32),
            "ratio_class_id": classes, "geometry": geometry}
