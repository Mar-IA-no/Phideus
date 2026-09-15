"""Finite identities, calibration and rank mapping; no scene generation or IO.

An identity is not permission to generate its scene. The executable campaign
must authenticate its freeze and stage before opening any prospective data.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import PurePosixPath

import numpy as np

from .measurement_sensor import DETECTOR_GRID

SCENARIOS = ("iid", "ood_beta", "ood_polyphony", "deformed_family")
CONDITIONS = ("canonical", "nominal", "short", "noisy")
ROLES = {
    "development": (8, (2026091501, 2026091502, 2026091503, 2026091504)),
    "calibration": (16, (2026091511, 2026091512, 2026091513, 2026091514)),
    "test": (128, (2026091531, 2026091532, 2026091533, 2026091534)),
}
IDENTITY_KEYS = {"role", "scenario", "condition", "scene_id", "split_seed"}


def identity(role, scenario, condition, scene_id):
    if (role not in ROLES or scenario not in SCENARIOS or condition not in CONDITIONS
            or type(scene_id) is not int or not 0 <= scene_id < ROLES[role][0]):
        raise ValueError("unit outside the fixed paired roster")
    return {"role": role, "scenario": scenario, "condition": condition,
            "scene_id": scene_id, "split_seed": ROLES[role][1][SCENARIOS.index(scenario)]}


def validate_identity(unit):
    if not isinstance(unit, dict) or set(unit) != IDENTITY_KEYS:
        raise ValueError("invalid paired identity schema")
    expected = identity(unit["role"], unit["scenario"], unit["condition"], unit["scene_id"])
    if type(unit["split_seed"]) is not int or unit != expected:
        raise ValueError("paired seed differs from the fixed roster")
    return expected


def unit_roster(role, *, audio_only=False):
    if role not in ROLES or type(audio_only) is not bool:
        raise ValueError("unknown role or condition selector")
    return [identity(role, scenario, condition, i)
            for scenario in SCENARIOS for i in range(ROLES[role][0])
            for condition in (CONDITIONS[1:] if audio_only else CONDITIONS)]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=True, allow_nan=False).encode("ascii")).hexdigest()


def reference(ref):
    if not isinstance(ref, dict) or set(ref) != {"path", "sha256", "bytes"}:
        raise ValueError("expected relative artifact reference")
    path, sha, size = ref["path"], ref["sha256"], ref["bytes"]
    if (not isinstance(path, str) or not path or path == "." or "\\" in path
            or PurePosixPath(path).is_absolute() or ".." in PurePosixPath(path).parts
            or str(PurePosixPath(path)) != path
            or not isinstance(sha, str) or len(sha) != 64
            or any(c not in "0123456789abcdef" for c in sha)
            or type(size) is not int or size < 0):
        raise ValueError("invalid relative reference or hash")
    return dict(ref)


def envelope(unit, *, emission, waveform, detection, kernel_identity):
    unit = validate_identity(unit)
    if unit["condition"] == "canonical":
        if waveform is not None or detection is not None:
            raise ValueError("canonical input has no waveform or detection reference")
    elif waveform is None or detection is None:
        raise ValueError("audio unit requires waveform and detector provenance")
    if kernel_identity is not None and (not isinstance(kernel_identity, str)
            or len(kernel_identity) != 64 or any(c not in "0123456789abcdef" for c in kernel_identity)):
        raise ValueError("invalid kernel identity")
    row = {"unit": unit, "emission": reference(emission),
           "waveform": None if waveform is None else reference(waveform),
           "detection": None if detection is None else reference(detection),
           "kernel_identity": kernel_identity}
    return {**row, "envelope_sha256": digest(row)}


def calibrate_detector(costs, units):
    """Nine fixed detectors × 192 equally weighted scene-condition costs.

Units and raw costs are retained in the result. No F, coverage or reader score
port. The caller authenticates detection-cost construction and seals this result.
"""
    expected = unit_roster("calibration", audio_only=True)
    if units != expected:
        raise ValueError("calibration unit order or completeness differs")
    values = np.asarray(costs)
    if (values.dtype != np.float64 or values.shape != (9, 192)
            or not np.isfinite(values).all() or np.any(values < 0)):
        raise ValueError("expected nine complete finite nonnegative cost vectors")
    means = values.mean(axis=1, dtype=np.float64)
    selected = min(range(9), key=lambda i: (means[i], *DETECTOR_GRID[i]))
    result = {"units": expected, "grid": [list(x) for x in DETECTOR_GRID],
              "costs": values.tolist(), "means": means.tolist(),
              "selected_index": selected, "height": DETECTOR_GRID[selected][0],
              "prominence": DETECTOR_GRID[selected][1], "unit_count": 192}
    return {**result, "calibration_sha256": digest(result)}


def partition_to_observed_labels(partition, canonical_to_observed):
    """Translate frequency-rank partitions to the actual delivered event order."""
    order = np.asarray(canonical_to_observed)
    n = len(order)
    if (order.ndim != 1 or order.dtype.kind not in "iu"
            or sorted(order.tolist()) != list(range(n))):
        raise ValueError("rank map must be a complete permutation")
    flat = [x for g in partition for x in g]
    if (any(not len(g) for g in partition) or any(type(x) is not int for x in flat)
            or sorted(flat) != list(range(n))):
        raise ValueError("partition must cover every event exactly once")
    result = np.empty(n, dtype=np.int64)
    for label, group in enumerate(partition):
        result[order[np.asarray(group, dtype=np.int64)]] = label
    return result
