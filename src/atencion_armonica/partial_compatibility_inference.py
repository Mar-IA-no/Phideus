"""Observation-only raw inference and lossless ragged float32 logit storage."""
from __future__ import annotations

import hashlib
import numpy as np
import torch

from .partial_compatibility_learning import collate_cached_observations


def collect_logits(model, records, *, device, batch_size=128):
    """Caller owns checkpoint/device authorization; this API accepts no sidecar."""
    if not records or batch_size != 128:
        raise ValueError("inference uses the fixed batch size on nonempty observations")
    model.eval()
    matrices = []
    with torch.inference_mode():
        for start in range(0, len(records), batch_size):
            selected = records[start:start+batch_size]
            batch = {k: v.to(device) for k, v in collate_cached_observations(selected).items()}
            logits = model(batch)
            if logits.dtype != torch.float32 or not torch.isfinite(logits).all():
                raise ValueError("inference did not produce finite float32 logits")
            expected = batch["pair_valid"].shape
            if logits.shape != expected:
                raise ValueError("unexpected logit shape")
            raw = logits.detach().cpu().numpy()
            for row, record in enumerate(selected):
                n = len(record["tokens"])
                matrices.append(raw[row, :n, :n].copy())
    return matrices


def observation_identity(observations):
    if not observations or any(set(o) != {"scene_id", "split_seed", "log_f"} for o in observations):
        raise ValueError("logit identity requires observations without privileged fields")
    if any(type(o["scene_id"]) is not int or type(o["split_seed"]) is not int for o in observations):
        raise ValueError("invalid observation IDs")
    ids = np.array([o["scene_id"] for o in observations], dtype=np.int64)
    seeds = np.array([o["split_seed"] for o in observations], dtype=np.int64)
    # Unlike cross-split deduplication, logit identity must preserve node order.
    fingerprints = np.array([hashlib.sha256(np.asarray(o["log_f"], dtype="<f4").tobytes()).hexdigest()
                             for o in observations], dtype="U64")
    return ids, seeds, fingerprints


def save_logits(path, matrices, observations):
    if not matrices or any(m.dtype != np.float32 or m.ndim != 2 or m.shape[0] != m.shape[1]
                           or len(m) < 2 or not np.isfinite(m).all() for m in matrices):
        raise ValueError("raw logits must be finite square float32 matrices")
    ids, seeds, fingerprints = observation_identity(observations)
    sizes = np.array([len(m) for m in matrices], dtype=np.int64)
    if len(matrices) != len(observations) or not np.array_equal(sizes, [len(o["log_f"]) for o in observations]):
        raise ValueError("logit sizes do not match observed identities")
    offsets = np.r_[np.int64(0), np.cumsum(sizes*sizes)]
    with path.open("xb") as handle:
        np.savez_compressed(handle, logits=np.concatenate([m.ravel() for m in matrices]),
                            sizes=sizes, offsets=offsets, scene_ids=ids, split_seeds=seeds,
                            observation_fingerprints=fingerprints)


def load_logits(path, expected_observations):
    expected_ids, expected_seeds, expected_fingerprints = observation_identity(expected_observations)
    with np.load(path, allow_pickle=False) as arrays:
        if set(arrays.files) != {"logits", "sizes", "offsets", "scene_ids", "split_seeds", "observation_fingerprints"}:
            raise ValueError("wrong raw logit schema")
        flat, sizes, offsets, ids = [arrays[k] for k in ("logits", "sizes", "offsets", "scene_ids")]
        seeds, fingerprints = arrays["split_seeds"], arrays["observation_fingerprints"]
    expected = np.asarray([len(o["log_f"]) for o in expected_observations], dtype=np.int64)
    if (flat.dtype != np.float32 or flat.ndim != 1 or not np.isfinite(flat).all()
            or any(a.dtype != np.int64 for a in (sizes, offsets, ids, seeds))
            or expected.ndim != 1 or not len(expected) or np.any(expected < 2)
            or not np.array_equal(sizes, expected) or not np.array_equal(ids, expected_ids)
            or not np.array_equal(seeds, expected_seeds) or not np.array_equal(fingerprints, expected_fingerprints)
            or not np.array_equal(offsets, np.r_[np.int64(0), np.cumsum(expected*expected)])
            or len(flat) != offsets[-1]):
        raise ValueError("raw logits do not match the ordered observation split")
    return [flat[offsets[i]:offsets[i+1]].reshape(int(n), int(n)).copy() for i, n in enumerate(sizes)]
