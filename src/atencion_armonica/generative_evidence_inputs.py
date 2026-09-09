"""Persist the exact float32 inputs for all three arms without tripling common arrays.

Pure codec, no filesystem or resource authority. The caller authenticates raw
and train-only normalizer references before packing, then pins this archive's
bytes before training. Targets and candidate metrics never enter this port.
"""
from __future__ import annotations

import hashlib

import numpy as np

from . import generative_evidence as ge
from . import generative_evidence_cache as cache
from .partial_compatibility_cache import encoded

FIELDS = {"groups": 9, "globals": 17, "local": 6, "generative": 6, "decoupled": 6}
OFFSETS = {"group_offsets", "candidate_offsets", "incidence_offsets"}


def pack_inputs(raw, common_norm, evidence_norm, *, binding, split, checkpoint_seed, raw_ref, normalizer_ref):
    decoded = cache.unpack_rows(raw, binding=binding, split=split, checkpoint_seed=checkpoint_seed)
    cache._binding(raw_ref)
    cache._binding(normalizer_ref)
    rows, shams = [], []
    for scene_id, row in zip(decoded["scene_ids"], decoded["rows"]):
        value = ge.model_inputs(row, common_norm, evidence_norm, "generative",
                               split_seed=cache.SPLITS[split][1], scene_id=scene_id)
        sham, record = ge.decouple(row["partitions"], value["evidence"],
                                  split_seed=cache.SPLITS[split][1], scene_id=scene_id)
        rows.append({"groups": value["groups"], "globals": value["globals"],
                     "generative": value["evidence"], "decoupled": sham,
                     "local": np.zeros_like(value["evidence"])})
        shams.append(record)
    metadata = {"schema": "generative-evidence-delivered-shard-v1", "binding": binding,
                "split": split, "checkpoint_seed": checkpoint_seed, "raw": raw_ref,
                "normalizers": normalizer_ref, "scenes": cache._decode(raw["metadata"])["scenes"], "sham": shams}
    return {**{name: np.concatenate([r[name] for r in rows]) for name in FIELDS},
            **{name: raw[name].copy() for name in OFFSETS | {"incidence"}}, "metadata": cache._metadata(metadata)}


def unpack_inputs(arrays, *, binding, split, checkpoint_seed, raw_ref, normalizer_ref, identities):
    cache._binding(binding)
    if set(arrays) != set(FIELDS) | OFFSETS | {"incidence", "metadata"}:
        raise ValueError("delivered input schema differs or contains supervision")
    meta = cache._decode(arrays["metadata"])
    if (set(meta) != {"schema", "binding", "split", "checkpoint_seed", "raw", "normalizers", "scenes", "sham"}
            or meta["schema"] != "generative-evidence-delivered-shard-v1" or meta["binding"] != binding
            or split not in cache.SPLITS or meta["split"] != split or checkpoint_seed not in ge.CHECKPOINTS
            or meta["checkpoint_seed"] != checkpoint_seed or meta["raw"] != raw_ref
            or meta["normalizers"] != normalizer_ref or not 1 <= len(meta["scenes"]) <= 512
            or len(meta["sham"]) != len(meta["scenes"])
            or [s["identity_sha256"] for s in meta["scenes"]] != identities or len(set(identities)) != len(identities)):
        raise ValueError("delivered inputs provenance or identity order differs")
    count = len(identities)
    for name, maximum in (("group_offsets", ge.MAX_GROUPS), ("candidate_offsets", ge.MAX_CANDIDATES),
                          ("incidence_offsets", ge.MAX_GROUPS*ge.MAX_CANDIDATES)):
        off = arrays[name]
        if (off.dtype != np.int64 or off.shape != (count+1,) or off[0] != 0
                or np.any(np.diff(off) < 0) or np.any(np.diff(off) > maximum)):
            raise ValueError("delivered input offsets differ")
    go, co, io = (arrays[k] for k in ("group_offsets", "candidate_offsets", "incidence_offsets"))
    for name, width in FIELDS.items():
        value = arrays[name]
        end = go[-1] if name == "groups" else co[-1]
        if value.dtype != np.float32 or value.shape != (end, width) or not np.isfinite(value).all():
            raise ValueError("delivered input dtype, extent or finiteness differs")
    if (arrays["incidence"].dtype != np.float32 or arrays["incidence"].shape != (io[-1],)
            or not np.isfinite(arrays["incidence"]).all() or np.any(arrays["local"] != 0)):
        raise ValueError("delivered incidence/local channel differs")
    result, ids = {arm: [] for arm in ge.ARMS}, []
    for i, scene in enumerate(meta["scenes"]):
        if set(scene) != {"observation", "partitions", "identity_sha256"}:
            raise ValueError("delivered scene schema differs")
        obs = scene["observation"]
        if (set(obs) != {"scene_id", "split_seed", "log_f"} or type(obs["scene_id"]) is not int
                or not 0 <= obs["scene_id"] < cache.SPLITS[split][0] or obs["split_seed"] != cache.SPLITS[split][1]):
            raise ValueError("delivered observation identity differs")
        q = np.asarray(obs["log_f"])
        if not np.array_equal(q, q.astype(np.float32).astype(np.float64)):
            raise ValueError("delivered observation must retain exact q32")
        ge.law.observable_q32(np.sort(q.astype(np.float32)))
        ps = ge.partitions_checked(scene["partitions"], len(q))
        expected_identity = hashlib.sha256(encoded({"split": split, "observation": obs, "partitions": ps})).hexdigest()
        gs = sorted({g for p in ps for g in p})
        if (expected_identity != identities[i] or go[i+1]-go[i] != len(gs) or co[i+1]-co[i] != len(ps)
                or io[i+1]-io[i] != len(ps)*len(gs)):
            raise ValueError("delivered candidate geometry or identity differs")
        incidence = arrays["incidence"][io[i]:io[i+1]].reshape(len(ps), len(gs))
        expected = np.array([[len(g)/len(q) if g in p else 0 for g in gs] for p in ps], np.float32).reshape(len(ps), len(gs))
        if not np.array_equal(incidence, expected):
            raise ValueError("delivered incidence no longer expresses candidate membership")
        gen = arrays["generative"][co[i]:co[i+1]]
        available = np.array([[len(p) in ge.law.BRANCHES[b][2] for b in ge.BRANCHES]
                              for p in ps], bool).reshape(len(ps), 3)
        if np.any(gen[~np.repeat(available, 2, axis=1)] != 0):
            raise ValueError("unavailable generative branches must remain zero")
        sham, record = ge.decouple(ps, gen, split_seed=cache.SPLITS[split][1], scene_id=obs["scene_id"])
        if not np.array_equal(sham, arrays["decoupled"][co[i]:co[i+1]]) or encoded(record) != encoded(meta["sham"][i]):
            raise ValueError("delivered sham values or attribution differ")
        for arm in ge.ARMS:
            result[arm].append({"groups": arrays["groups"][go[i]:go[i+1]],
                "globals": arrays["globals"][co[i]:co[i+1]], "incidence": incidence,
                "evidence": arrays[arm][co[i]:co[i+1]]})
        ids.append(obs["scene_id"])
    if ids != sorted(set(ids)):
        raise ValueError("delivered scene order differs")
    return {"scene_ids": ids, "identities": identities.copy(), "inputs": result, "sham": meta["sham"]}
