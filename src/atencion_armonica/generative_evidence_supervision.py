"""Explicit supervision port; never imported by observable preparation or models.

The only file reader here opens pinned OPEN train/calibration sidecars.
Test supervision needs its own sealed-output gate before these pure helpers.
"""
from __future__ import annotations

import json

import numpy as np

from .generative_evidence import partitions_checked
from .generative_evidence_reuse import OpenShard, validate_observation
from .learned_partition_core import partition_errors
from .learned_partition_metrics import _augment, METRICS
from .observable_rival_evaluation import validate_truth
from .structured_source_metrics import partition_metrics

TRUTH_KEYS = {"scene_id", "split_seed", "sources", "sigma_cents", "mean_log_f_observed",
              "log_f_ideal", "sensor_log_noise", "source_ids", "partial_indices", "permutation"}


def reconstruct_truth(observation, truth, split):
    """Pure semantic reconstruction, not authorization to open a sidecar file."""
    if (set(observation) != {"scene_id", "split_seed", "log_f"} or set(truth) != TRUTH_KEYS
            or any(type(truth[k]) is not int or truth[k] != observation[k] for k in ("scene_id", "split_seed"))
            or split not in ("train", "calibration", "iid", "ood_beta", "ood_polyphony", "deformed_family")
            or type(truth["sigma_cents"]) not in (int, float) or truth["sigma_cents"] != 2.):
        raise ValueError("supervision schema, identity or source law differs")
    q = np.asarray(observation["log_f"], np.float64)
    if (q.ndim != 1 or not 8 <= len(q) <= 32 or not np.isfinite(q).all()
            or not np.array_equal(q, q.astype(np.float32).astype(np.float64))):
        raise ValueError("supervision requires an exact q32 observation")
    for key in ("source_ids", "partial_indices", "permutation"):
        if not isinstance(truth[key], list) or len(truth[key]) != len(q) or any(type(i) is not int for i in truth[key]):
            raise ValueError("supervision membership/permutation schema differs")
    for source in truth["sources"]:
        if (set(source) != {"f0", "beta", "gamma", "indices"}
                or any(type(source[k]) not in (int, float) or not np.isfinite(source[k])
                       for k in ("f0", "beta", "gamma"))):
            raise ValueError("supervision source parameter schema differs")
    order = np.argsort(q, kind="stable")
    scene = {"q32": q[order], "canonical_to_observed": order,
             "scene_id": observation["scene_id"], "split": split}
    return validate_truth(scene, truth)


def open_truths(shard):
    if not isinstance(shard, OpenShard):
        raise PermissionError("only verified OPEN shards may enter this sidecar reader")
    raw = shard.data.read("sidecars.jsonl")
    truths = [json.loads(line) for line in raw.splitlines()]
    if len(truths) != len(shard.observations):
        raise ValueError("supervision scene roster differs")
    result = []
    for i, obs, truth in zip(shard.ids, shard.observations, truths):
        validate_observation(obs, i, shard.split)
        result.append(reconstruct_truth(obs, truth, shard.split))
    return result


def candidate_supervision(partitions, canonical_labels):
    """New candidate roster, inherited entropies/metrics; no old target padding."""
    y = np.asarray(canonical_labels)
    if y.ndim != 1 or y.dtype.kind not in "iu" or not 8 <= len(y) <= 32:
        raise ValueError("expected canonical integer source labels")
    ps = partitions_checked(partitions, len(y))
    errors = [partition_errors(p, y) for p in ps]
    raw = np.asarray([e["raw"] for e in errors], np.float64).reshape(-1, 2)
    targets = np.asarray([e["normalized"] for e in errors], np.float32).reshape(-1, 2)
    metrics = [_augment(partition_metrics(p, y), e["raw"], len(y), len(np.unique(y)))
               for p, e in zip(ps, errors)]
    if any(set(m) != set(METRICS) for m in metrics):
        raise ValueError("candidate metrics differ from the inherited thirteen")
    return {"raw": raw, "targets": targets, "metrics": metrics}
