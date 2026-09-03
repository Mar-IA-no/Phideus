"""Numerically independent long-double cross-check for Wave 49 distances."""

from __future__ import annotations

import numpy as np


def _curve(family: str, x: np.ndarray, params: dict[str, float]) -> np.ndarray:
    p = {key: np.longdouble(value) for key, value in params.items()}
    if family == "PROP":
        return p["k"] * x
    if family == "AFFINE_OFFSET":
        return p["a"] + p["b"] * x
    if family == "POWER_NONUNIT":
        return p["a"] * np.power(x, p["p"])
    if family == "SATURATING":
        return p["L"] * x / (p["K"] + x)
    raise ValueError(f"Unknown reference family: {family}")


def reference_family_distances(
    latent_x: np.ndarray,
    clean_y: np.ndarray,
    covariance: np.ndarray,
    catalog: dict[str, list[dict[str, float]]],
) -> dict[str, float]:
    """Recompute D_EIV using long double and a manual 2x2 inverse."""
    x = np.asarray(latent_x, dtype=np.longdouble)
    y = np.asarray(clean_y, dtype=np.longdouble)
    cov = np.asarray(covariance, dtype=np.longdouble)
    determinant = cov[:, 0, 0] * cov[:, 1, 1] - cov[:, 0, 1] * cov[:, 1, 0]
    if np.any(determinant <= 0):
        raise ValueError("reference oracle requires SPD covariance")
    weight_y = cov[:, 0, 0] / determinant
    distances: dict[str, float] = {}
    for family, candidates in catalog.items():
        best = np.longdouble(np.inf)
        for params in candidates:
            delta = y - _curve(family, x, params)
            value = np.longdouble("0.5") * np.sum(weight_y * delta * delta, dtype=np.longdouble)
            if value < best:
                best = value
        distances[family] = float(best)
    return distances
