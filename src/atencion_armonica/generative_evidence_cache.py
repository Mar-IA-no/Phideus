"""Ragged compact arrays with explicit observation and candidate identities.

Pure codecs, not authorization to draw, fit, train or open a test sidecar.
The stage runner authenticates file bytes and supplies the expected binding.
No fixed 512-scene padding or inherited 64-candidate/94-group limits.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

from . import generative_evidence as ge
from .partial_compatibility_cache import encoded

SPLITS = {"train": (4096, 2026090880), "calibration": (512, 2026090881),
          "iid": (512, 2026090982), "ood_beta": (512, 2026090983),
          "ood_polyphony": (512, 2026090984), "deformed_family": (512, 2026090985)}
FIELDS = {"groups": (9, np.float64), "globals": (17, np.float64),
          "evidence": (6, np.float64), "available": (3, np.bool_)}
ROW_KEYS = {"n", "partitions", "group_ids", "canonical_to_observed", "q_tie_count",
            "groups", "globals", "incidence", "evidence", "available"}


def _metadata(value):
    return np.frombuffer(encoded(value), np.uint8).copy()


def _decode(value):
    if not isinstance(value, np.ndarray) or value.dtype != np.uint8 or value.ndim != 1:
        raise ValueError("metadata must be canonical JSON bytes, never pickle")
    raw = value.tobytes()
    decoded = json.loads(raw)
    if encoded(decoded) != raw:
        raise ValueError("noncanonical cache metadata")
    return decoded


def _binding(binding):
    if not isinstance(binding, dict) or not binding or json.loads(encoded(binding)) != binding:
        raise ValueError("cache requires a nonempty canonical provenance binding")


def scene_metadata(split, observation, row):
    if (split not in SPLITS or set(observation) != {"scene_id", "split_seed", "log_f"}
            or type(observation["scene_id"]) is not int or not 0 <= observation["scene_id"] < SPLITS[split][0]
            or type(observation["split_seed"]) is not int or observation["split_seed"] != SPLITS[split][1]
            or set(row) != ROW_KEYS):
        raise ValueError("observation/row role or schema differs")
    original = np.asarray(observation["log_f"])
    q = original.astype(np.float32)
    ge.law.observable_q32(np.sort(q))
    if not np.array_equal(original, q.astype(np.float64)) or type(row["n"]) is not int or row["n"] != len(q):
        raise ValueError("row does not match exact delivered q32")
    ps = ge.partitions_checked(row["partitions"], len(q))
    gs = sorted({g for p in ps for g in p})
    if ([tuple(g) for g in row["group_ids"]] != gs or len(gs) > ge.MAX_GROUPS
            or not np.array_equal(row["canonical_to_observed"], np.argsort(q, kind="stable"))
            or type(row["q_tie_count"]) is not int or row["q_tie_count"] != len(q)-len(np.unique(q))):
        raise ValueError("group roster, canonical mapping or ties differ")
    for name, (width, dtype) in FIELDS.items():
        v, count = row[name], len(gs) if name == "groups" else len(ps)
        if not isinstance(v, np.ndarray) or v.dtype != dtype or v.shape != (count, width) or not np.isfinite(v).all():
            raise ValueError("compact raw feature shape, dtype or finiteness differs")
    expected = np.zeros((len(ps), len(gs)), np.float32)
    lookup = {g: i for i, g in enumerate(gs)}
    for i, p in enumerate(ps):
        for g in p:
            expected[i, lookup[g]] = len(g)/len(q)
    inc = row["incidence"]
    if not isinstance(inc, np.ndarray) or inc.dtype != np.float32 or not np.array_equal(inc, expected):
        raise ValueError("candidate/group incidence identity differs")
    available = np.array([[len(p) in ge.law.BRANCHES[b][2] for b in ge.BRANCHES] for p in ps], bool).reshape(-1, 3)
    if (not np.array_equal(row["available"], available)
            or np.any(row["evidence"][~np.repeat(available, 2, axis=1)] != 0)):
        raise ValueError("candidate branch support differs")
    identity = {"split": split, "observation": observation, "partitions": ps}
    return json.loads(encoded({"observation": observation, "partitions": ps,
        "identity_sha256": hashlib.sha256(encoded(identity)).hexdigest()}))


def pack_rows(split, checkpoint_seed, observations, rows, *, binding):
    _binding(binding)
    if (type(checkpoint_seed) is not int or checkpoint_seed not in ge.CHECKPOINTS
            or not 1 <= len(rows) <= 512 or len(observations) != len(rows)):
        raise ValueError("wrong compact shard count or checkpoint")
    scenes = [scene_metadata(split, obs, row) for obs, row in zip(observations, rows)]
    ids = [s["observation"]["scene_id"] for s in scenes]
    if ids != sorted(set(ids)):
        raise ValueError("compact scene roster must be sorted and unique")
    gc, cc = [len(r["groups"]) for r in rows], [len(r["globals"]) for r in rows]
    result = {name: np.concatenate([r[name] for r in rows]) for name in FIELDS}
    result.update({"group_offsets": np.r_[np.int64(0), np.cumsum(gc, dtype=np.int64)],
                   "candidate_offsets": np.r_[np.int64(0), np.cumsum(cc, dtype=np.int64)],
                   "incidence_offsets": np.r_[np.int64(0), np.cumsum(np.asarray(gc)*cc, dtype=np.int64)],
                   "incidence": np.concatenate([r["incidence"].ravel() for r in rows]),
                   "metadata": _metadata({"schema": "generative-evidence-raw-shard-v1", "binding": binding,
                       "split": split, "checkpoint_seed": checkpoint_seed, "scenes": scenes})})
    return result


def unpack_rows(arrays, *, binding, split, checkpoint_seed):
    _binding(binding)
    if set(arrays) != set(FIELDS) | {"incidence", "group_offsets", "candidate_offsets", "incidence_offsets", "metadata"}:
        raise ValueError("raw cache schema differs or contains supervision")
    meta = _decode(arrays["metadata"])
    if (set(meta) != {"schema", "binding", "split", "checkpoint_seed", "scenes"}
            or meta["schema"] != "generative-evidence-raw-shard-v1" or meta["binding"] != binding
            or meta["split"] != split or meta["checkpoint_seed"] != checkpoint_seed
            or type(checkpoint_seed) is not int or checkpoint_seed not in ge.CHECKPOINTS
            or not 1 <= len(meta["scenes"]) <= 512):
        raise ValueError("raw cache provenance or role differs")
    count = len(meta["scenes"])
    for name, maximum in (("group_offsets", ge.MAX_GROUPS), ("candidate_offsets", ge.MAX_CANDIDATES),
                          ("incidence_offsets", ge.MAX_GROUPS*ge.MAX_CANDIDATES)):
        off = arrays[name]
        if (off.dtype != np.int64 or off.shape != (count+1,) or off[0] != 0
                or np.any(np.diff(off) < 0) or np.any(np.diff(off) > maximum)):
            raise ValueError("invalid ragged offsets")
    go, co, io = (arrays[k] for k in ("group_offsets", "candidate_offsets", "incidence_offsets"))
    for name, (width, dtype) in FIELDS.items():
        end = go[-1] if name == "groups" else co[-1]
        if arrays[name].dtype != dtype or arrays[name].shape != (end, width):
            raise ValueError("raw compact extent or dtype differs")
    if arrays["incidence"].dtype != np.float32 or arrays["incidence"].shape != (io[-1],):
        raise ValueError("incidence compact extent differs")
    rows, identities = [], []
    for i, s in enumerate(meta["scenes"]):
        obs, ps = s["observation"], s["partitions"]
        q = np.asarray(obs["log_f"], np.float32)
        gs = sorted({tuple(g) for p in ps for g in p})
        if (go[i+1]-go[i] != len(gs) or co[i+1]-co[i] != len(ps)
                or io[i+1]-io[i] != len(ps)*len(gs)):
            raise ValueError("candidate signatures do not match array extents")
        row = {"n": len(q), "partitions": ps, "group_ids": gs,
               "canonical_to_observed": np.argsort(q, kind="stable"), "q_tie_count": len(q)-len(np.unique(q)),
               "incidence": arrays["incidence"][io[i]:io[i+1]].reshape(len(ps), len(gs))}
        for name in FIELDS:
            off = go if name == "groups" else co
            row[name] = arrays[name][off[i]:off[i+1]]
        if scene_metadata(split, obs, row) != s:
            raise ValueError("scene/candidate identity digest differs")
        rows.append(row)
        identities.append(s["identity_sha256"])
    ids = [s["observation"]["scene_id"] for s in meta["scenes"]]
    if ids != sorted(set(ids)):
        raise ValueError("raw cache scene ordering differs")
    return {"scene_ids": ids, "identities": identities, "observations": [s["observation"] for s in meta["scenes"]], "rows": rows}


def pack_targets(identities, supervision, *, binding):
    """Separate archive; identity order must equal the corresponding raw cache."""
    _binding(binding)
    if (not 1 <= len(identities) <= 512 or len(supervision) != len(identities)
            or len(set(identities)) != len(identities)
            or any(not isinstance(s, str) or len(s) != 64 for s in identities)):
        raise ValueError("target identity roster differs")
    for row in supervision:
        raw, target = row["raw"], row["targets"]
        if (raw.dtype != np.float64 or target.dtype != np.float32 or raw.ndim != 2 or raw.shape[1] != 2
                or raw.shape != target.shape or not 0 <= len(raw) <= ge.MAX_CANDIDATES
                or not np.isfinite(raw).all() or not np.isfinite(target).all()
                or np.any(raw < 0) or np.any(target < 0) or np.any(target > 1)):
            raise ValueError("invalid compact supervision")
    return {"raw": np.concatenate([r["raw"] for r in supervision]),
            "targets": np.concatenate([r["targets"] for r in supervision]),
            "offsets": np.r_[np.int64(0), np.cumsum([len(r["raw"]) for r in supervision], dtype=np.int64)],
            "metadata": _metadata({"schema": "generative-evidence-target-shard-v1", "binding": binding,
                                    "identities": list(identities)})}


def unpack_targets(arrays, decoded_rows, *, binding):
    _binding(binding)
    if set(arrays) != {"raw", "targets", "offsets", "metadata"}:
        raise ValueError("target cache schema differs")
    meta = _decode(arrays["metadata"])
    if meta != {"schema": "generative-evidence-target-shard-v1", "binding": binding, "identities": decoded_rows["identities"]}:
        raise ValueError("target cache identity/order/binding differs")
    counts = [len(r["partitions"]) for r in decoded_rows["rows"]]
    offsets = np.r_[np.int64(0), np.cumsum(counts, dtype=np.int64)]
    if arrays["offsets"].dtype != np.int64 or not np.array_equal(arrays["offsets"], offsets):
        raise ValueError("target candidate offsets differ")
    for key, dtype in (("raw", np.float64), ("targets", np.float32)):
        a = arrays[key]
        if a.dtype != dtype or a.shape != (offsets[-1], 2) or not np.isfinite(a).all() or np.any(a < 0):
            raise ValueError("target shape/dtype/domain differs")
    result = []
    for i, row in enumerate(decoded_rows["rows"]):
        raw = arrays["raw"][offsets[i]:offsets[i+1]]
        target = arrays["targets"][offsets[i]:offsets[i+1]]
        if np.any(target > 1) or not np.array_equal((raw/np.log(row["n"])).astype(np.float32), target):
            raise ValueError("delivered targets differ from raw normalized entropies")
        result.append({"raw": raw, "targets": target})
    return result
