"""Lossless packed normalized inputs: one file per checkpoint/arm/shard."""
from __future__ import annotations

import numpy as np

from .structured_source_artifacts import write_npz

COUNT, GROUPS, CANDIDATES = 512, 94, 64
FIELDS = {"groups", "globals", "incidence", "group_mask", "candidate_mask", "scene_ids"}


def validate_arrays(arrays, *, scene_ids, dim):
    ids = np.asarray(scene_ids)
    if (dim not in (8, 9) or ids.dtype != np.int64 or ids.shape != (COUNT,)
            or ids[0] < 0 or not np.array_equal(ids, np.arange(ids[0], ids[0]+COUNT)) or set(arrays) != FIELDS):
        raise ValueError("packed input roster, dimension or schema differs")
    shapes = {"groups": (COUNT, GROUPS, dim), "globals": (COUNT, CANDIDATES, 6),
        "incidence": (COUNT, CANDIDATES, GROUPS), "group_mask": (COUNT, GROUPS),
        "candidate_mask": (COUNT, CANDIDATES), "scene_ids": (COUNT,)}
    for name, shape in shapes.items():
        dtype = np.int64 if name == "scene_ids" else np.bool_ if name.endswith("_mask") else np.float32
        if arrays[name].dtype != dtype or arrays[name].shape != shape or not np.isfinite(arrays[name]).all():
            raise ValueError("packed input shapes, precision or finite values differ")
    if not np.array_equal(arrays["scene_ids"], ids):
        raise ValueError("packed input belongs to another ordered shard")
    gmask, cmask = arrays["group_mask"], arrays["candidate_mask"]
    ng, nc = gmask.sum(axis=1), cmask.sum(axis=1)
    if (np.any(ng == 0) or np.any(nc == 0)
            or not np.array_equal(gmask, np.arange(GROUPS)[None, :] < ng[:, None])
            or not np.array_equal(cmask, np.arange(CANDIDATES)[None, :] < nc[:, None])):
        raise ValueError("packed ragged masks must be nonempty contiguous prefixes")
    if (np.any(arrays["groups"][~gmask] != 0) or np.any(arrays["globals"][~cmask] != 0)
            or np.any(arrays["incidence"][~cmask] != 0)
            or np.any(np.where(gmask[:, None, :], 0., arrays["incidence"]) != 0)):
        raise ValueError("packed padding must be exactly zero, not an extra feature")
    return ng, nc


def pack_inputs(path, inputs, *, scene_ids, dim):
    if len(inputs) != COUNT or dim not in (8, 9):
        raise ValueError("packed inputs require one complete512-scene shard")
    arrays = {"groups": np.zeros((COUNT, GROUPS, dim), np.float32),
        "globals": np.zeros((COUNT, CANDIDATES, 6), np.float32),
        "incidence": np.zeros((COUNT, CANDIDATES, GROUPS), np.float32),
        "group_mask": np.zeros((COUNT, GROUPS), np.bool_),
        "candidate_mask": np.zeros((COUNT, CANDIDATES), np.bool_), "scene_ids": np.asarray(scene_ids)}
    for i, row in enumerate(inputs):
        if set(row) != {"groups", "globals", "incidence"}:
            raise ValueError("packed producer accepts observable inputs only")
        g, c, w = (row[k] for k in ("groups", "globals", "incidence"))
        if (g.ndim != 2 or c.ndim != 2 or not 1 <= len(g) <= GROUPS or not 1 <= len(c) <= CANDIDATES
                or g.shape[1] != dim or c.shape[1] != 6 or w.shape != (len(c), len(g))
                or any(v.dtype != np.float32 or not np.isfinite(v).all() for v in (g, c, w))):
            raise ValueError("invalid ragged normalized input")
        arrays["groups"][i, :len(g)] = g
        arrays["globals"][i, :len(c)] = c
        arrays["incidence"][i, :len(c), :len(g)] = w
        arrays["group_mask"][i, :len(g)] = True
        arrays["candidate_mask"][i, :len(c)] = True
    validate_arrays(arrays, scene_ids=scene_ids, dim=dim)
    write_npz(path, **arrays)


def read_inputs(path, *, scene_ids, dim):
    with np.load(path, allow_pickle=False) as raw:
        arrays = {k: raw[k] for k in raw.files}
    ng, nc = validate_arrays(arrays, scene_ids=scene_ids, dim=dim)
    return [{"groups": arrays["groups"][i, :g], "globals": arrays["globals"][i, :c],
             "incidence": arrays["incidence"][i, :c, :g]} for i, (g, c) in enumerate(zip(ng, nc))]
