"""Predeclared scene-first primary contrasts; arrays must come from sealed eval.

No decision thresholds or architecture promotion. Uncertainty is conditional
on the nine frozen backbone/reader cells, not uncertainty over new trainings.
"""
from __future__ import annotations

import numpy as np

PRIMARY_NAMES = (
    "extended_nominal_minus_canonical",
    "geometric_nominal_minus_canonical",
    "geometric_injection_advantage_nominal_minus_canonical",
    "geometric_minus_extended_nominal",
)
ROUTES = ("injection", "geometric", "decoupled", "local")


def primary_contrasts(neural_f, extended_f):
    """Axes: scenario, scene, condition, route, backbone, reader seed.

Scenario/condition orders are measurement_contract.SCENARIOS/CONDITIONS.
All 128 scenes per scenario are present, including scientific abstentions F=0.
These shape checks cannot prove roster provenance; the caller authenticates it.
"""
    neural, extended = np.asarray(neural_f), np.asarray(extended_f)
    for a, shape in ((neural, (4, 128, 4, 4, 3, 3)), (extended, (4, 128, 4))):
        if (a.dtype != np.float64 or a.shape != shape or not np.isfinite(a).all()
                or np.any((a < 0) | (a > 1))):
            raise ValueError("expected complete finite scene-first F arrays")
    means = neural.mean(axis=(4, 5), dtype=np.float64)
    deformed = means[3]
    injection, geometric = deformed[:, :, 0], deformed[:, :, 1]
    reference = extended[3]
    return np.column_stack((reference[:, 1]-reference[:, 0],
                            geometric[:, 1]-geometric[:, 0],
                            (geometric[:, 1]-injection[:, 1])-(geometric[:, 0]-injection[:, 0]),
                            geometric[:, 1]-reference[:, 1]))


def bootstrap_primaries(contrasts):
    values = np.asarray(contrasts)
    if (values.dtype != np.float64 or values.shape != (128, 4)
            or not np.isfinite(values).all()):
        raise ValueError("expected four paired contrasts on all128deformed scenes")
    rng = np.random.Generator(np.random.PCG64(2026091549))
    indices = rng.integers(0, 128, size=(10000, 128), dtype=np.int64)
    distribution = values[indices].mean(axis=1, dtype=np.float64)
    bounds = np.quantile(distribution, [.00625, .99375], axis=0, method="linear")
    estimates = values.mean(axis=0, dtype=np.float64)
    return {"indices": indices, "distribution": distribution,
            "report": {name: {"mean": float(estimates[i]), "lower": float(bounds[0, i]),
                              "upper": float(bounds[1, i]), "scene_count": 128,
                              "confidence_level": .9875}
                       for i, name in enumerate(PRIMARY_NAMES)}}
