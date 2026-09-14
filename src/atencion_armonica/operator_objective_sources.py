"""Read-only authenticated source ports for the retrospective diagnostic.

No producer/store constructor, fitter, forward, training, or random draw is
called. The campaign adapter must first authenticate the closed test roster;
these low-level readers alone do not establish that authority.
"""
from __future__ import annotations

import gzip
import hashlib
from io import BytesIO
import json
import math
import os
from pathlib import Path, PurePosixPath
import stat
import zipfile

import numpy as np

from . import observable_source_rivals as law

BRANCHES = ("base-low", "base-high", "deformed-low")
MAX_CANDIDATES = 82


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)+"\n").encode()


def _json(raw):
    value = json.loads(raw)
    if encoded(value) != raw:
        raise ValueError("source JSON is not canonical")
    return value


def _digest(value):
    if (not isinstance(value, str) or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)):
        raise ValueError("invalid SHA256 reference")
    return value


def _size(value, maximum):
    if type(value) is not int or not 0 <= value <= maximum:
        raise ValueError("source size exceeds its declared envelope")
    return value


class AuthenticatedReader:
    """Explicit relative bases; no symlinks, traversal, unbounded reads or writes.

    Directory file descriptors keep the no-symlink condition valid during
    traversal, not only at a preceding pathname check. The trusted root itself
    is absolute and must have no symlink component when this port is created.
    """

    def __init__(self, root, *, maximum_bytes=256*1024**2):
        root = Path(root)
        if not root.is_absolute() or root != Path(os.path.abspath(root)):
            raise ValueError("reader requires a canonical absolute root")
        if any(p.is_symlink() for p in [root, *root.parents]) or not root.is_dir():
            raise ValueError("reader root must be an existing nonsymlink directory")
        self.root = root
        self.maximum_bytes = _size(maximum_bytes, 1024**3)

    @staticmethod
    def parts(relative):
        if (not isinstance(relative, str) or not relative or "\\" in relative
                or "\x00" in relative or PurePosixPath(relative).is_absolute()
                or any(p in ("", ".", "..") for p in relative.split("/"))):
            raise ValueError("reference must be an explicit safe relative path")
        return relative.split("/")

    def bytes(self, ref, *, base=None):
        if (not isinstance(ref, dict) or set(ref) not in
                ({"path", "sha256"}, {"path", "sha256", "bytes"})):
            raise ValueError("invalid source reference schema")
        digest = _digest(ref["sha256"])
        declared = _size(ref["bytes"], self.maximum_bytes) if "bytes" in ref else None
        parts = ([] if base is None else self.parts(base))+self.parts(ref["path"])
        directory = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            for part in parts[:-1]:
                child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
                os.close(directory)
                directory = child
            fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
            with os.fdopen(fd, "rb") as handle:
                info = os.fstat(handle.fileno())
                if not stat.S_ISREG(info.st_mode) or info.st_size > self.maximum_bytes:
                    raise ValueError("source is not a bounded regular file")
                if declared is not None and info.st_size != declared:
                    raise ValueError("source size differs from receipt")
                raw = handle.read((declared if declared is not None else self.maximum_bytes)+1)
        finally:
            os.close(directory)
        if ((declared is not None and len(raw) != declared) or len(raw) > self.maximum_bytes
                or hashlib.sha256(raw).hexdigest() != digest):
            raise ValueError("source bytes differ from authenticated receipt")
        return raw

    def json(self, ref, *, base=None):
        return _json(self.bytes(ref, base=base))

    def arrays(self, ref, *, base=None, maximum_decoded_bytes=512*1024**2):
        raw = self.bytes(ref, base=base)
        maximum = _size(maximum_decoded_bytes, 1024**3)
        with zipfile.ZipFile(BytesIO(raw)) as archive:
            members = archive.infolist()
            names = [m.filename for m in members]
            if (not members or len(names) != len(set(names))
                    or any(not n.endswith(".npy") or len(self.parts(n)) != 1 for n in names)
                    or sum(m.file_size for m in members) > maximum):
                raise ValueError("array archive names or decoded envelope differ")
            # Bound header-declared allocation before np.load, not after it.
            for member in members:
                with archive.open(member) as handle:
                    version = np.lib.format.read_magic(handle)
                    header_readers = {(1, 0): np.lib.format.read_array_header_1_0,
                                      (2, 0): np.lib.format.read_array_header_2_0}
                    if version not in header_readers:
                        raise ValueError("unsupported diagnostic array header version")
                    shape, _, dtype = header_readers[version](handle)
                    if (dtype.kind not in "fiub" or len(shape) > 2
                            or any(type(d) is not int or d < 0 for d in shape)
                            or math.prod(shape)*dtype.itemsize != member.file_size-handle.tell()):
                        raise ValueError("array header differs from bounded numeric payload")
        with np.load(BytesIO(raw), allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
        if any(a.dtype.hasobject for a in arrays.values()) or sum(a.nbytes for a in arrays.values()) > maximum:
            raise ValueError("array content exceeds non-object envelope")
        return arrays

    def fit_json(self, receipt, *, base, maximum_decoded_bytes=16*1024**2):
        if (not isinstance(receipt, dict) or set(receipt) !=
                {"path", "bytes", "sha256", "codec", "decoded_bytes", "decoded_sha256"}
                or receipt["codec"] != "canonical-json-gzip3-mtime0"):
            raise ValueError("invalid factor codec receipt")
        count = _size(receipt["decoded_bytes"], maximum_decoded_bytes)
        digest = _digest(receipt["decoded_sha256"])
        raw = self.bytes({k: receipt[k] for k in ("path", "bytes", "sha256")}, base=base)
        with gzip.GzipFile(fileobj=BytesIO(raw), mode="rb") as stream:
            decoded = stream.read(count+1)
        if len(decoded) != count or hashlib.sha256(decoded).hexdigest() != digest:
            raise ValueError("decoded factors differ from receipt")
        # Match the historical codec without replaying any numerical fit.
        if gzip.compress(decoded, compresslevel=3, mtime=0) != raw:
            raise ValueError("factor compression differs from declared canonical codec")
        return _json(decoded)


def compact_fit(value, observation):
    """Extract supported canonical partitions and six costs, not factor trees."""
    if (set(value) != {"observation", "inventory", "fits", "group_factors"}
            or encoded(value["observation"]) != encoded(observation)):
        raise ValueError("fit observation or source schema differs")
    q = np.asarray(observation["log_f"])
    if (q.ndim != 1 or q.dtype.kind not in "fiu" or not 8 <= len(q) <= 32
            or not np.isfinite(q).all() or not np.array_equal(q, q.astype(np.float32).astype(np.float64))):
        raise ValueError("fit observation must retain exact q32")
    law.observable_q32(np.sort(q.astype(np.float32)))
    inventory = value["inventory"]
    rows = inventory["candidates"]
    all_partitions, supported = [], []
    for row in rows:
        p = law.validate_partition(row["partition"], len(q))
        if tuple(map(tuple, row["partition"])) != p or row["origin"] not in ("pool", "neighbor"):
            raise ValueError("inventory signature or origin differs")
        expected = "SUPPORTED" if law.supported(p) else "OUTSIDE_GENERATIVE_CARDINALITY"
        if row["status"] != expected:
            raise ValueError("inventory support differs from frozen cardinality law")
        all_partitions.append(p)
        if row["status"] == "SUPPORTED":
            supported.append(p)
    if len(all_partitions) != len(set(all_partitions)) or len(supported) > MAX_CANDIDATES:
        raise ValueError("inventory has duplicate candidates or exceeds envelope")
    ps = sorted(supported)
    if len(ps) != len(value["fits"]):
        raise ValueError("fit roster differs from supported inventory")
    bounds = np.zeros((len(ps), 6), np.float64)
    mask = np.zeros((len(ps), 3), bool)
    for i, (p, fit) in enumerate(zip(ps, value["fits"])):
        expected = {b for b in BRANCHES if len(p) in law.BRANCHES[b][2]}
        if (fit["status"] != "FITTED" or tuple(map(tuple, fit["partition"])) != p
                or set(fit["branches"]) != expected):
            raise ValueError("fit signature, status or available branches differ")
        for j, name in enumerate(BRANCHES):
            if name not in expected:
                continue
            pair = [fit["branches"][name][key] for key in ("LB", "UB")]
            if any(type(x) not in (int, float) for x in pair):
                raise ValueError("fit bounds must be numeric, not coerced strings or booleans")
            if not np.isfinite(pair).all() or min(pair) < 0 or pair[0] > pair[1]+law.TOL_J:
                raise ValueError("invalid discrete fit bounds")
            bounds[i, 2*j:2*j+2] = pair
            mask[i, j] = True
    scores, winners = {}, {}
    for family, allowed in law.FAMILIES.items():
        for offset, kind in enumerate(("LB", "UB")):
            key = f"{family}_{kind.lower()}"
            picked = [min((b for j, b in enumerate(BRANCHES) if b in allowed and mask[i, j]),
                          key=lambda b: (bounds[i, 2*BRANCHES.index(b)+offset], b)) for i in range(len(ps))]
            scores[key] = np.asarray([bounds[i, 2*BRANCHES.index(b)+offset] for i, b in enumerate(picked)], np.float64)
            winners[key] = picked
    return {"n": len(q), "partitions": ps, "inventory": inventory,
            "bounds": bounds, "available": mask, "log_channel": np.log1p(bounds/len(q)),
            "scores": scores, "winning_branches": winners}


def verify_delivered_channel(compact, normalizer, delivered, sham):
    """Recompose delivered float32 values using preserved donors, without RNG.

    Caller authenticates normalizer TRAIN lineage and the delivered shard.
    This verifies values and donor attribution, not that a new sham was drawn.
    """
    n = len(compact["partitions"])
    mean, scale = (np.asarray(normalizer[k]) for k in ("mean", "scale"))
    if (mean.dtype.kind not in "fiu" or scale.dtype.kind not in "fiu"
            or mean.shape != (6,) or scale.shape != (6,) or not np.isfinite(mean).all()
            or not np.isfinite(scale).all() or np.any(scale <= 0)):
        raise ValueError("invalid frozen TRAIN normalizer")
    expected = ((compact["log_channel"]-mean)/scale).astype(np.float32)
    expected[~np.repeat(compact["available"], 2, axis=1)] = 0.
    if set(delivered) != {"local", "generative", "decoupled"}:
        raise ValueError("delivered arm roster differs")
    for a in delivered.values():
        if not isinstance(a, np.ndarray) or a.dtype != np.float32 or a.shape != (n, 6) or not np.isfinite(a).all():
            raise ValueError("delivered channel extent, dtype or finiteness differs")
    if not np.array_equal(delivered["generative"], expected) or np.any(delivered["local"] != 0):
        raise ValueError("delivered generative/local values differ")
    if set(sham) != {"strata", "donors", "changed_mask", "status"}:
        raise ValueError("sham schema differs")
    if (not isinstance(sham["donors"], list) or len(sham["donors"]) != n
            or any(type(x) is not int for x in sham["donors"])):
        raise ValueError("saved donors must be explicit integer indices")
    donors = np.asarray(sham["donors"], np.int64)
    if not np.array_equal(np.sort(donors), np.arange(n)):
        raise ValueError("saved donors are not a permutation")
    ps = compact["partitions"]
    sizes = [tuple(sorted(map(len, p))) for p in ps]
    if any(sizes[i] != sizes[d] for i, d in enumerate(donors)):
        raise ValueError("donor crosses the fixed size stratum")
    decoupled = expected[donors]
    changed = np.any(decoupled != expected, axis=1)
    records = []
    for size in sorted(set(sizes)):
        ids = np.asarray([i for i, s in enumerate(sizes) if s == size], np.int64)
        assigned = donors[ids]
        shift = None if len(ids) == 1 else int(np.flatnonzero(ids == assigned[0])[0])
        if len(ids) > 1 and (shift == 0 or not np.array_equal(assigned, np.roll(ids, -shift))):
            raise ValueError("saved stratum donors differ from a nontrivial cyclic shift")
        record_changed = changed[ids]
        records.append({"sizes": size, "candidate_ids": ids.tolist(), "donors": assigned.tolist(),
                        "shift": shift, "changed_fraction": float(record_changed.mean()),
                        "status": "NO_PERMUTATION" if shift is None else
                        "INPUT_CHANGED" if record_changed.any() else "INPUT_UNCHANGED"})
    expected_sham = {"strata": records, "donors": donors.tolist(), "changed_mask": changed.tolist(),
                     "status": "INPUT_CHANGED" if changed.any() else "INPUT_UNCHANGED"}
    if encoded(expected_sham) != encoded(sham) or not np.array_equal(delivered["decoupled"], decoupled):
        raise ValueError("delivered sham values or preserved attribution differ")
    return {"generative": expected, "decoupled": decoupled, "donors": donors.copy()}
