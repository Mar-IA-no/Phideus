"""Seed-explicit observable assembly without historical split-table mutation.

The caller supplies an authenticated expected seed, observed features/logits,
fits and TRAIN normalizers. This module cannot draw, forward, fit or open truth.
It is also usable on declared OPEN fixtures for resource measurement.
"""
from __future__ import annotations

import hashlib

import numpy as np

from . import generative_evidence as ge
from .geometric_decision_core import delivered_interface, recentered_roundtrip
from .partial_compatibility_cache import encoded
from .structured_source_data import validate_record
from .structured_source_reader import build_pool

ROLES = ("train", "calibration", "iid", "ood_beta", "ood_polyphony", "deformed_family")
SCENE_KEYS = {"split", "observation", "features", "logits", "pools", "inventory", "partitions",
              "canonical_to_observed", "q32", "identity", "status"}


def validate_observation(observation, *, scene_id, split_seed):
    if (not isinstance(observation, dict) or set(observation) != {"scene_id", "split_seed", "log_f"}
            or type(scene_id) is not int or not 0 <= scene_id < 4096
            or type(split_seed) is not int or split_seed < 0
            or type(observation["scene_id"]) is not int or observation["scene_id"] != scene_id
            or type(observation["split_seed"]) is not int or observation["split_seed"] != split_seed):
        raise ValueError("observable schema or explicit expected identity differs")
    original = np.asarray(observation["log_f"])
    if not isinstance(observation["log_f"], list):
        raise ValueError("observable coordinates require a JSON list")
    if original.dtype.kind not in "fiu":
        raise ValueError("observable coordinates must be numeric")
    q = original.astype(np.float32)
    ge.law.observable_q32(np.sort(q, kind="stable"))
    if not np.array_equal(original, q.astype(np.float64)):
        raise ValueError("observation must preserve exact delivered q32")
    return q


def canonical_observation(observation, q):
    return {"scene_id": observation["scene_id"], "split_seed": observation["split_seed"],
            "log_f": q.astype(np.float64).tolist()}


def scene_from_sources(split, observation, features, logits, *, expected_seed):
    if split not in ROLES:
        raise ValueError("unknown observable role")
    scene_id = observation.get("scene_id") if isinstance(observation, dict) else None
    q = validate_observation(observation, scene_id=scene_id, split_seed=expected_seed)
    observation = canonical_observation(observation, q)
    if scene_id >= (4096 if split == "train" else 512):
        raise ValueError("scene ID outside the full role roster")
    validate_record(features, q)
    if (not isinstance(logits, dict) or set(logits) != set(ge.CHECKPOINTS)
            or any(not isinstance(z, np.ndarray) or z.dtype != np.float32 or z.shape != (len(q), len(q))
                   or not np.isfinite(z).all() or not np.array_equal(z, z.T) for z in logits.values())):
        raise ValueError("observable source requires three finite symmetric preserved logits")
    pools = {str(cp): build_pool(q, logits[cp], features["pair_support"]) for cp in ge.CHECKPOINTS}
    inventory = ge.law.candidate_inventory({cp: pool["partitions"] for cp, pool in pools.items()}, len(q))
    partitions = ge.partitions_checked(sorted(ge.law.signature(row["partition"])
        for row in inventory["candidates"] if row["status"] == "SUPPORTED"), len(q))
    order = np.argsort(q, kind="stable")
    identity = hashlib.sha256(encoded({"split": split, "observation": observation, "partitions": partitions})).hexdigest()
    return {"split": split, "observation": observation,
        "features": {k: v.copy() for k, v in features.items()}, "logits": {k: v.copy() for k, v in logits.items()},
        "pools": pools, "inventory": inventory, "partitions": partitions,
        "canonical_to_observed": order, "q32": q[order].copy(), "identity": identity,
        "status": "ELIGIBLE" if partitions else "NO_OBSERVABLE_CANDIDATE"}


def roundtrip_observation(observation, *, expected_seed):
    """One derived observation, plus diagnostic coordinates and event lineage.

    Event identity is the original delivered vector position, not a frequency
    match. No sidecar is supplied, reconstructed or parsed here. The original
    sampler identity alone does not authorize this transformed observation.
    """
    q = validate_observation(observation, scene_id=observation["scene_id"], split_seed=expected_seed)
    observation = canonical_observation(observation, q)
    coordinates = recentered_roundtrip(q)
    probe = coordinates["q_probe"]
    changed = {**observation, "log_f": probe.astype(np.float64).tolist()}
    validate_observation(changed, scene_id=observation["scene_id"], split_seed=expected_seed)
    original_order, probe_order = (np.argsort(v, kind="stable") for v in (q, probe))
    def ties(value):
        return [np.flatnonzero(value == x).tolist() for x in np.unique(value) if np.count_nonzero(value == x) > 1]
    return {"observation": changed, "coordinates": coordinates,
        "lineage": {"kind": "delivered-coordinate-roundtrip-not-physical-redraw",
            "parent_observation_sha256": hashlib.sha256(encoded(observation)).hexdigest(),
            "derived_observation_sha256": hashlib.sha256(encoded(changed)).hexdigest(),
            "event_ids": list(range(len(q))), "original_rank_to_event": original_order.tolist(),
            "probe_rank_to_event": probe_order.tolist(), "original_tied_event_groups": ties(q),
            "probe_tied_event_groups": ties(probe),
            "correspondence": "event position preserved; tied frequencies do not identify a unique rank match",
            "q_center": "diagnostic only, not a third pipeline", "sidecar_access": False}}


def inputs_from_fits(scene, fits, normalizers, *, scale, expected_seed):
    if not isinstance(scene, dict) or set(scene) != SCENE_KEYS:
        raise ValueError("observable scene contains unexpected fields or supervision")
    observation = scene["observation"]
    q = validate_observation(observation, scene_id=observation["scene_id"], split_seed=expected_seed)
    if scene["split"] not in ROLES or observation["scene_id"] >= (4096 if scene["split"] == "train" else 512):
        raise ValueError("observable role or scene extent differs")
    validate_record(scene["features"], q)
    ps = ge.partitions_checked(scene["partitions"], len(q))
    identity = hashlib.sha256(encoded({"split": scene["split"], "observation": observation, "partitions": ps})).hexdigest()
    if (identity != scene["identity"] or not np.array_equal(scene["canonical_to_observed"], np.argsort(q, kind="stable"))
            or not np.array_equal(scene["q32"], q[scene["canonical_to_observed"]])
            or scene["status"] != ("ELIGIBLE" if ps else "NO_OBSERVABLE_CANDIDATE")
            or set(scene["logits"]) != set(ge.CHECKPOINTS)
            or set(normalizers) != {"common", "evidence"}
            or set(normalizers["common"]) != {str(cp) for cp in ge.CHECKPOINTS}):
        raise ValueError("observable universe or TRAIN normalizer roster differs")
    raw, inputs, diagnostics = {}, {}, {}
    for cp in ge.CHECKPOINTS:
        raw[cp] = ge.observable_rows(q, scene["logits"][cp], scene["features"]["triples"],
            scene["features"]["residual_cents"], ps, fits)
        six = ge.model_inputs(raw[cp], normalizers["common"][str(cp)], normalizers["evidence"], "generative",
                              split_seed=expected_seed, scene_id=observation["scene_id"])
        inputs[cp], diagnostics[cp] = delivered_interface(six, raw[cp], scale=scale,
            split_seed=expected_seed, scene_id=observation["scene_id"])
    return {"identity": identity, "raw": raw, "inputs": inputs, "diagnostics": diagnostics}
