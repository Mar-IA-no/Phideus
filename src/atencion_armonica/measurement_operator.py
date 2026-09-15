"""Observable adapter into frozen kernels; no training, drawing, fitting or IO.

The campaign owns saved observations/logits, source authentication and global
seals. Absolute detected frequencies stop here; kernels receive only q32.
"""
from __future__ import annotations

import numpy as np

from .measurement_contract import validate_identity
from .measurement_sensor import operator_observation

PREPARED_KEYS = {"unit", "status", "observation"}


def prepare_input(unit, coordinates):
    unit = validate_identity(unit)
    if unit["condition"] == "canonical":
        original = np.asarray(coordinates)
        if (original.ndim != 1 or original.dtype.kind not in "fiu"
                or not 8 <= len(original) <= 32 or not np.isfinite(original).all()):
            raise ValueError("invalid canonical coordinates")
        q = original.astype(np.float32)
        if not np.array_equal(q.astype(np.float64), original):
            raise ValueError("canonical coordinates must preserve exact old q32")
    else:
        state = operator_observation(coordinates)
        if state["status"] != "ELIGIBLE":
            return {"unit": unit, "status": state["status"], "observation": None}
        q = state["q32"]
    return {"unit": unit, "status": "ELIGIBLE",
            "observation": {"scene_id": unit["scene_id"], "split_seed": unit["split_seed"],
                            "log_f": q.astype(np.float64).tolist()}}


def validate_prepared(prepared):
    if not isinstance(prepared, dict) or set(prepared) != PREPARED_KEYS:
        raise ValueError("unexpected fields at observable adapter")
    unit = validate_identity(prepared["unit"])
    if prepared["status"] == "OUTSIDE_OPERATOR_DOMAIN":
        if unit["condition"] == "canonical" or prepared["observation"] is not None:
            raise ValueError("invalid out-of-domain input")
        return None
    if prepared["status"] != "ELIGIBLE":
        raise ValueError("unknown input status")
    from .geometric_decision_observables import validate_observation
    return validate_observation(prepared["observation"], scene_id=unit["scene_id"],
                                split_seed=unit["split_seed"])


def features_for_input(prepared):
    if validate_prepared(prepared) is None:
        return None
    from .partial_compatibility_cache import feature_record
    return feature_record(prepared["observation"])


def assemble_observable(prepared, features, logits):
    """Consume already preserved features/forward, never launch callbacks here."""
    q = validate_prepared(prepared)
    if q is None:
        if features is not None or logits is not None:
            raise ValueError("out-of-domain observations must not enter feature/forward path")
        return {"unit": prepared["unit"], "status": "OUTSIDE_OPERATOR_DOMAIN", "scene": None}
    from .geometric_decision_observables import scene_from_sources
    scene = scene_from_sources(prepared["unit"]["scenario"], prepared["observation"],
                               features, logits, expected_seed=prepared["unit"]["split_seed"])
    status = "NO_CANDIDATE" if scene["status"] == "NO_OBSERVABLE_CANDIDATE" else scene["status"]
    return {"unit": prepared["unit"], "status": status, "scene": scene}


def candidate_origins(scene):
    """Descriptive membership in the joint proposer, without candidate selection."""
    def signature(p):
        return tuple(sorted(tuple(sorted(g)) for g in p))
    native, analytic = {}, {}
    for cp, pool in scene["pools"].items():
        native[cp] = {signature(p) for p in pool["trees"]["neural"]["cuts"]}
        analytic[cp] = {signature(p) for p in pool["trees"]["analytic"]["cuts"]}
    lookup = {signature(row["partition"]): row for row in scene["inventory"]["candidates"]}
    result = []
    for partition in scene["partitions"]:
        p = signature(partition)
        row = lookup[p]
        result.append({"partition": p, "origin": row["origin"],
                       "checkpoints": row.get("checkpoints", []), "parents": row.get("parents", []),
                       "native_membership": {cp: p in pool for cp, pool in native.items()},
                       "analytic_membership": {cp: p in pool for cp, pool in analytic.items()}})
    return result
