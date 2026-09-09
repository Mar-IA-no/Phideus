"""Full TRAIN roster and checkpoint alignment around the pure moment kernels.

The caller authenticates each loaded NPZ against its immutable preparation
receipt. This layer additionally detects changes between normalization passes.
It never accepts calibration, a reduced count, or a supervision loader.
"""
from __future__ import annotations

import hashlib

import numpy as np

from . import generative_evidence as ge
from . import generative_evidence_cache as cache
from .partial_compatibility_cache import encoded


def arrays_digest(arrays):
    digest = hashlib.sha256()
    for key, array in sorted(arrays.items()):
        digest.update(encoded([key, array.dtype.str, list(array.shape)]))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def fit_training_normalizers(load_shard, *, binding):
    """load_shard(checkpoint_seed, shard_index) returns authenticated raw arrays.

    Exactly eight shards of 512 scenes for each of the three checkpoints.
    Empty candidate sets remain in the roster but contribute no moments.
    """
    digests, eligible, identities, excluded = {}, [], [], []
    branch_counts = np.zeros(3, np.int64)

    def decode(cp, shard):
        arrays = load_shard(cp, shard)
        digest = arrays_digest(arrays)
        key = (cp, shard)
        if digests.setdefault(key, digest) != digest:
            raise ValueError("training arrays changed between normalization passes")
        result = cache.unpack_rows(arrays, binding=binding, split="train", checkpoint_seed=cp)
        if result["scene_ids"] != list(range(shard*512, (shard+1)*512)):
            raise ValueError("normalizer requires the complete ordered 4096-scene TRAIN roster")
        return result

    for shard in range(8):
        baseline = decode(ge.CHECKPOINTS[0], shard)
        identities.extend(baseline["identities"])
        for scene_id, row in zip(baseline["scene_ids"], baseline["rows"]):
            (eligible if len(row["partitions"]) else excluded).append(scene_id)
            branch_counts += row["available"].any(axis=0)
        for cp in ge.CHECKPOINTS[1:]:
            other = decode(cp, shard)
            if (other["identities"] != baseline["identities"]
                    or other["observations"] != baseline["observations"]):
                raise ValueError("checkpoint observation/candidate identity differs")
            for a, b in zip(baseline["rows"], other["rows"]):
                # Only group logit moments and global logit energy may differ.
                for key in ("evidence", "available", "incidence", "canonical_to_observed"):
                    if not np.array_equal(a[key], b[key]):
                        raise ValueError("checkpoint shared generative evidence or structure differs")
                if (not np.array_equal(a["groups"][:, [0, 1, 2, 7, 8]], b["groups"][:, [0, 1, 2, 7, 8]])
                        or not np.array_equal(np.delete(a["globals"], 2, axis=1), np.delete(b["globals"], 2, axis=1))):
                    raise ValueError("checkpoint non-logit features differ")
        del baseline, other
    if len(identities) != 4096 or len(set(identities)) != 4096 or np.any(branch_counts == 0):
        raise ValueError("incomplete TRAIN identity roster or absent branch support")

    def rows(cp):
        for shard in range(8):
            yield from decode(cp, shard)["rows"]

    common = {str(cp): ge.fit_common_normalizer(lambda cp=cp: rows(cp)) for cp in ge.CHECKPOINTS}
    evidence = ge.fit_evidence_normalizer(lambda: rows(ge.CHECKPOINTS[0]))
    if (any(n["scene_count"] != [len(eligible)]*5 for n in common.values())
            or evidence["scene_count"] != np.repeat(branch_counts, 2).tolist()):
        raise ValueError("normalizer support differs from the observable training roster")
    return {"schema": "generative-evidence-train-normalizers-v1", "binding": binding,
            "split": "train", "split_seed": cache.SPLITS["train"][1], "scene_count": 4096,
            "eligible_scene_ids": eligible, "excluded_scene_ids": excluded,
            "scene_identities": identities, "common": common, "evidence": evidence,
            "array_digests": [{"checkpoint_seed": cp, "shard": shard, "sha256": digest}
                              for (cp, shard), digest in sorted(digests.items())]}
