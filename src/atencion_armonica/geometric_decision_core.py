"""Observable geometric interface and arithmetic; no datasets, Torch or file IO.

The caller authenticates TRAIN provenance for the scale and the old delivered
six channels. These pure functions cannot authorize a campaign or open truth.
"""
from __future__ import annotations

import math

import numpy as np

from . import generative_evidence as ge

ROUTES = ("injection", "geometric", "decoupled", "local")
LOSSES = ("mse", "decision")
ARMS = tuple(f"{route}_{loss}" for route in ROUTES for loss in LOSSES)
READER_SEEDS = (2026091491, 2026091492, 2026091493)
INPUT_KEYS = frozenset(("groups", "globals", "incidence", "evidence"))


def geometric_log_cost(evidence, available):
    """Minimum available log1p(UB/N), directly in the preserved raw precision."""
    if (not isinstance(evidence, np.ndarray) or evidence.dtype != np.float64
            or evidence.ndim != 2 or evidence.shape[1] != 6
            or len(evidence) > ge.MAX_CANDIDATES
            or not isinstance(available, np.ndarray) or available.dtype != np.bool_
            or available.shape != (len(evidence), 3)
            or not np.isfinite(evidence).all() or np.any(evidence < 0)
            or np.any(evidence[~np.repeat(available, 2, axis=1)] != 0)
            or not available.any(axis=1).all()):
        raise ValueError("invalid raw evidence or available branch support")
    return np.min(np.where(available, evidence[:, 1::2], np.inf), axis=1)


def fit_scale(scene_costs):
    """Scene then candidate RMS; provenance and full TRAIN roster are external.

    Empty scenes contribute no mass and are counted explicitly. Iteration order
    is significant: the authenticated caller supplies increasing scene IDs.
    """
    total, eligible, empty = np.float64(0), 0, 0
    for costs in scene_costs:
        if (not isinstance(costs, np.ndarray) or costs.dtype != np.float64
                or costs.ndim != 1 or len(costs) > ge.MAX_CANDIDATES
                or not np.isfinite(costs).all() or np.any(costs < 0)):
            raise ValueError("invalid geometric costs for scale")
        if len(costs):
            total += np.mean(np.square(costs), dtype=np.float64)
            eligible += 1
        else:
            empty += 1
    if not eligible or not np.isfinite(total):
        raise ValueError("scale requires finite nonempty TRAIN support")
    rms = float(np.sqrt(total/eligible))
    return {"scale": 1. if rms == 0 else rms, "zero_rms": rms == 0,
            "eligible_scenes": eligible, "empty_scenes": empty}


def delivered_interface(delivered, raw_row, *, scale, split_seed, scene_id):
    """Identical eight channels for all nonlocal routes, including donor scalar.

    No hidden float64 bypass: its sole source is these delivered float32 bytes.
    The raw row is observable metadata, never a supervision record.
    """
    if (set(delivered) != INPUT_KEYS or type(scale) not in (int, float)
            or not math.isfinite(scale) or scale <= 0):
        raise ValueError("invalid delivered schema or frozen scale")
    evidence = delivered["evidence"]
    costs = geometric_log_cost(raw_row["evidence"], raw_row["available"])
    if (evidence.dtype != np.float32 or evidence.shape != (len(costs), 6)
            or not np.isfinite(evidence).all()
            or len(raw_row["partitions"]) != len(costs)):
        raise ValueError("delivered/raw candidate extent differs")
    _, sham = ge.decouple(raw_row["partitions"], evidence,
                          split_seed=split_seed, scene_id=scene_id)
    z = (costs/scale).astype(np.float32)
    donor = np.asarray(sham["donors"], np.int64)
    extra = np.column_stack((z, z[donor]))
    result = {k: v.copy() for k, v in delivered.items()}
    result["evidence"] = np.concatenate((evidence, extra), axis=1)
    if any(v.dtype != np.float32 or not np.isfinite(v).all() for v in result.values()):
        raise ValueError("nonfinite or nondelivered precision at interface")
    return result, {"six_channel_sham": sham, "scalar_changed_mask": (z != z[donor]).tolist()}


def route_inputs(inputs, route):
    if route not in ROUTES or set(inputs) != INPUT_KEYS:
        raise ValueError("unknown route or input schema")
    result = {k: v.copy() for k, v in inputs.items()}
    if route == "local":
        result["evidence"][:] = 0
    return result


def choose_energy(energy, partitions):
    """Exact energy ties by canonical signature, independently of array order."""
    if (not isinstance(energy, np.ndarray) or energy.dtype != np.float64
            or energy.shape != (len(partitions),) or not np.isfinite(energy).all()
            or len(partitions) > ge.MAX_CANDIDATES):
        raise ValueError("invalid finite decision energy")
    if not len(partitions):
        return None
    n = sum(map(len, partitions[0]))
    ps = [ge.law.validate_partition(p, n) for p in partitions]
    if len(set(ps)) != len(ps) or any(not ge.law.supported(p) for p in ps):
        raise ValueError("invalid candidate universe")
    return min(range(len(ps)), key=lambda i: (energy[i], ps[i]))


def recentered_roundtrip(q):
    """Delivered-coordinate numerical stability probe, NOT a physical redraw.

    New protocol revision fixes this order explicitly. q_center is an input
    diagnostic only; q_probe is the sole new pipeline input.
    """
    if (not isinstance(q, np.ndarray) or q.dtype != np.float32 or q.ndim != 1
            or not 8 <= len(q) <= 32 or not np.isfinite(q).all()):
        raise ValueError("expected finite delivered q32")
    q64 = q.astype(np.float64)
    shifted64 = q64+np.log(2.)
    shifted32 = shifted64.astype(np.float32)
    shifted_delivered64 = shifted32.astype(np.float64)
    probe = (shifted_delivered64-shifted_delivered64.mean(dtype=np.float64)).astype(np.float32)
    return {"original": q.copy(), "shifted64": shifted64, "shifted32": shifted32,
            "q_center": (q64-q64.mean(dtype=np.float64)).astype(np.float32), "q_probe": probe}
