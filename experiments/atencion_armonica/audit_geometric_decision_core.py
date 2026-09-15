"""Independent, CPU-only kernels and authenticated readers for the final audit.

This module deliberately does not import the geometric-decision metric,
evaluation, reporting, selection, model, fitter, or sampling modules.  It owns
the arithmetic used as the audit oracle.  Importing it performs no filesystem
access.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from io import BytesIO
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np


CHECKPOINTS = (2026090721, 2026090722, 2026090723)
READER_SEEDS = (2026091491, 2026091492, 2026091493)
ROUTES = ("injection", "geometric", "decoupled", "local")
LOSSES = ("mse", "decision")
ARMS = tuple(f"{route}_{loss}" for route in ROUTES for loss in LOSSES)
STAGES = ("initial", "selected")
METRICS = ("regret_tM", "regret_tD", "ari", "optimal_tM", "optimal_tD",
           "exact", "k_absolute_error", "mse_components", "tau_b")
CLASSICAL = ("base", "extended", "z", "d")
TESTS = (("iid", 2026091582), ("ood_beta", 2026091483),
         ("ood_polyphony", 2026091484), ("deformed_family", 2026091485))
BOOTSTRAP_SEED = 2026091494
BOOTSTRAP_COUNT = 10000
CONTRASTS = ("geometric_minus_injection_mse", "geometric_minus_injection_decision",
             "interaction_decision_minus_mse", "geometric_minus_decoupled_decision")
ATOL = 1e-12


def encoded(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def sha256_path(path: Path, *, check=lambda: None) -> tuple[str, int]:
    digest, size = hashlib.sha256(), 0
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            check()
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def receipt(value: Any, *, sizes: bool = True) -> dict[str, Any]:
    keys = {"path", "sha256", "bytes"} if sizes else {"path", "sha256"}
    if (not isinstance(value, dict) or set(value) != keys or type(value["path"]) is not str
            or not value["path"] or Path(value["path"]).is_absolute() or ".." in Path(value["path"]).parts
            or Path(value["path"]).as_posix() != value["path"]
            or type(value["sha256"]) is not str or len(value["sha256"]) != 64
            or any(c not in "0123456789abcdef" for c in value["sha256"])
            or (sizes and (type(value["bytes"]) is not int or value["bytes"] < 0))):
        raise ValueError("invalid explicit artifact receipt")
    return value


@dataclass
class Coverage:
    identities: dict[tuple[str, str], tuple[int, str]] = field(default_factory=dict)
    files_by_store: dict[str, int] = field(default_factory=dict)
    bytes_by_store: dict[str, int] = field(default_factory=dict)
    types: dict[str, int] = field(default_factory=dict)
    counted: set[tuple[str, str]] = field(default_factory=set)

    def record(self, store: str, ref: dict[str, Any], kind: str) -> None:
        ident = (ref["bytes"], ref["sha256"])
        key = (store, ref["path"])
        previous = self.identities.setdefault(key, ident)
        if previous != ident:
            raise ValueError(f"conflicting identities for {store}:{ref['path']}")
        if key in self.counted:
            return
        self.counted.add(key)
        self.files_by_store[store] = self.files_by_store.get(store, 0) + 1
        self.bytes_by_store[store] = self.bytes_by_store.get(store, 0) + ref["bytes"]
        self.types[kind] = self.types.get(kind, 0) + 1

    def summary(self) -> dict[str, Any]:
        return {"files_by_store": dict(sorted(self.files_by_store.items())),
                "bytes_by_store": dict(sorted(self.bytes_by_store.items())),
                "artifacts_by_type": dict(sorted(self.types.items()))}


class AuditStore:
    """Read authenticated bytes from one explicitly owned root.

    The caller supplies the store label and root; no ownership is inferred from
    a receipt's `path` field.  Symlinks are rejected at every in-root component.
    """

    def __init__(self, label: str, root: Path, coverage: Coverage):
        if type(label) is not str or not label:
            raise ValueError("explicit store label required")
        candidate = Path(root)
        if not candidate.is_dir() or candidate.is_symlink():
            raise ValueError(f"missing or symlinked store root: {label}")
        self.label, self.root, self.coverage = label, candidate.resolve(), coverage

    def path(self, relative: str) -> Path:
        if (type(relative) is not str or not relative or Path(relative).is_absolute()
                or ".." in Path(relative).parts or Path(relative).as_posix() != relative):
            raise ValueError("artifact path escapes its declared store")
        path = self.root / relative
        if any(p.is_symlink() for p in (path, *path.parents) if p.is_relative_to(self.root)):
            raise ValueError("artifact path traverses a symlink")
        return path

    def current_reference(self, relative: str, *, check=lambda: None) -> dict[str, Any]:
        path = self.path(relative)
        digest, size = sha256_path(path, check=check)
        return {"path": relative, "sha256": digest, "bytes": size}

    def _authenticated(self, ref: dict[str, Any], *, kind: str, retain: bool,
                       check=lambda: None) -> bytes:
        ref = receipt(ref)
        path = self.path(ref["path"])
        digest, size, chunks = hashlib.sha256(), 0, []
        with path.open("rb") as stream:
            while block := stream.read(1024 * 1024):
                check()
                digest.update(block)
                size += len(block)
                if retain:
                    chunks.append(block)
        if size != ref["bytes"] or digest.hexdigest() != ref["sha256"]:
            raise ValueError(f"authenticated bytes changed: {self.label}:{ref['path']}")
        self.coverage.record(self.label, ref, kind)
        return b"".join(chunks)

    def authenticate(self, ref: dict[str, Any], *, kind: str = "opaque", check=lambda: None) -> None:
        """Hash an artifact without retaining it (used for opaque checkpoints)."""
        self._authenticated(ref, kind=kind, retain=False, check=check)

    def read(self, ref: dict[str, Any], *, kind: str = "opaque", check=lambda: None) -> bytes:
        """Return exactly the byte sequence whose digest was authenticated."""
        return self._authenticated(ref, kind=kind, retain=True, check=check)

    def json(self, ref: dict[str, Any], *, check=lambda: None) -> Any:
        raw = self.read(ref, kind="json", check=check)
        value = json.loads(raw)
        if encoded(value) != raw:
            raise ValueError(f"noncanonical JSON: {self.label}:{ref['path']}")
        return value

    def arrays(self, ref: dict[str, Any], *, check=lambda: None) -> dict[str, np.ndarray]:
        raw = self.read(ref, kind="npz", check=check)
        with np.load(BytesIO(raw), allow_pickle=False) as archive:
            if len(archive.files) != len(set(archive.files)):
                raise ValueError("duplicate array names")
            return {name: archive[name] for name in archive.files}


def project_reference(root: Path, ref: dict[str, Any], *, check=lambda: None) -> dict[str, Any]:
    """Authenticate a source/input declared relative to the repository."""
    ref = receipt(ref, sizes="bytes" in ref)
    path = (Path(root) / ref["path"])
    resolved_root = Path(root).resolve()
    if (not path.resolve().is_relative_to(resolved_root)
            or any(part.is_symlink() for part in (path, *path.parents) if part.is_relative_to(resolved_root))):
        raise ValueError("project reference escapes or traverses a symlink")
    digest, size = sha256_path(path, check=check)
    if digest != ref["sha256"] or ("bytes" in ref and size != ref["bytes"]):
        raise ValueError(f"project source changed: {ref['path']}")
    return {"path": ref["path"], "sha256": digest, "bytes": size}


def canonical_partition(partition: Any, n: int) -> tuple[tuple[int, ...], ...]:
    try:
        raw = tuple(tuple(group) for group in partition)
    except TypeError:
        raise ValueError("partition must be a nested integer sequence") from None
    if any(type(i) is bool or not isinstance(i, (int, np.integer)) for group in raw for i in group):
        raise ValueError("partition identities must be integers, never coerced")
    groups = tuple(tuple(int(i) for i in group) for group in raw)
    if (not groups or any(not group for group in groups)
            or any(tuple(sorted(group)) != group for group in groups)
            or tuple(sorted(groups)) != groups
            or sorted(i for group in groups for i in group) != list(range(n))):
        raise ValueError("partition is not canonical, complete and disjoint")
    return groups


def partitions_checked(partitions: Any, n: int) -> list[tuple[tuple[int, ...], ...]]:
    if not isinstance(partitions, (list, tuple)):
        raise ValueError("candidate roster must be explicit")
    result = [canonical_partition(partition, n) for partition in partitions]
    if result != sorted(set(result)):
        raise ValueError("candidate roster is not canonical and unique")
    return result


def independent_targets(partitions: Any, labels: np.ndarray) -> dict[str, np.ndarray]:
    y = np.asarray(labels)
    if y.ndim != 1 or y.dtype.kind not in "iu" or not 2 <= len(y) <= 32:
        raise ValueError("labels must identify 2..32 events")
    n = len(y)
    ps = partitions_checked(partitions, n)
    _, truth = np.unique(y, return_inverse=True)
    truth_count = int(truth.max()) + 1
    u64 = np.empty((len(ps), 2), np.float64)
    ari = np.empty(len(ps), np.float64)
    exact = np.empty(len(ps), np.bool_)
    k_error = np.empty(len(ps), np.int64)
    for index, partition in enumerate(ps):
        table = [[sum(1 for event in group if truth[event] == truth_id)
                  for truth_id in range(truth_count)] for group in partition]
        group_mass = [sum(row) for row in table]
        truth_mass = [sum(row[j] for row in table) for j in range(truth_count)]
        # H(A,B)-H(B) and H(A,B)-H(A), using scalar fsum rather than the
        # production conditional-entropy vectorization.
        joint_h = -math.fsum((count / n) * math.log(count / n)
                             for row in table for count in row if count)
        group_h = -math.fsum((count / n) * math.log(count / n) for count in group_mass if count)
        truth_h = -math.fsum((count / n) * math.log(count / n) for count in truth_mass if count)
        u64[index] = ((joint_h - truth_h) / math.log(n),
                      (joint_h - group_h) / math.log(n))
        choose2 = lambda count: count * (count - 1) // 2
        observed = sum(choose2(count) for row in table for count in row)
        row_pairs = sum(choose2(count) for count in group_mass)
        column_pairs = sum(choose2(count) for count in truth_mass)
        total_pairs = choose2(n)
        expected = row_pairs * column_pairs / total_pairs
        maximum = 0.5 * (row_pairs + column_pairs)
        ari[index] = 1.0 if maximum == expected else (observed - expected) / (maximum - expected)
        exact[index] = bool(np.all(np.count_nonzero(table, axis=0) == 1)
                            and np.all(np.count_nonzero(table, axis=1) == 1))
        k_error[index] = len(partition) - truth_count
    u64[u64 == 0] = 0.0
    u32 = u64.astype(np.float32)
    return {"u64": u64, "u32": u32, "tM": u64.sum(1, dtype=np.float64),
            "tD": u32.astype(np.float64).sum(1, dtype=np.float64),
            "ari": ari, "exact": exact, "k_error": k_error}


def order_record(values: np.ndarray, indices: Iterable[int] | None = None) -> dict[str, Any]:
    x = np.asarray(values)
    if x.ndim != 1 or x.dtype.kind not in "fiu" or not np.isfinite(x).all():
        raise ValueError("finite one-dimensional score required")
    raw_ids = list(range(len(x))) if indices is None else list(indices)
    if any(type(i) is bool or not isinstance(i, (int, np.integer)) for i in raw_ids):
        raise ValueError("candidate indices must be integer identities")
    ids = np.asarray(raw_ids, np.int64)
    if (ids.ndim != 1 or np.any(ids < 0) or np.any(ids >= len(x))
            or not np.array_equal(ids, np.unique(ids))):
        raise ValueError("candidate indices must be sorted and unique")
    if not len(ids):
        return {"status": "NO_CANDIDATE", "chosen": None, "order": [],
                "tie_blocks": [], "optima": [], "minimum": None, "next_gap": None}
    ordered = sorted((int(i) for i in ids), key=lambda i: (x[i], i))
    blocks: list[list[int]] = []
    for candidate in ordered:
        if not blocks or x[candidate] != x[blocks[-1][0]]:
            blocks.append([])
        blocks[-1].append(candidate)
    minimum = x[ordered[0]]
    gap = float(np.float64(x[blocks[1][0]]) - np.float64(minimum)) if len(blocks) > 1 else None
    return {"status": "DEFINED", "chosen": ordered[0], "order": ordered,
            "tie_blocks": blocks, "optima": blocks[0],
            "minimum": float(minimum), "next_gap": gap}


def choose_energy(energy: np.ndarray, partitions: Any) -> int | None:
    values = np.asarray(energy)
    if values.dtype != np.float64 or values.shape != (len(partitions),) or not np.isfinite(values).all():
        raise ValueError("invalid decision energy")
    if not len(partitions):
        return None
    n = sum(len(group) for group in partitions[0])
    ps = [canonical_partition(partition, n) for partition in partitions]
    if len(ps) != len(set(ps)):
        raise ValueError("transport candidate roster contains duplicates")
    return min(range(len(ps)), key=lambda i: (values[i], ps[i]))


def kendall_tau_b(score: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    x, y = np.asarray(score), np.asarray(target)
    if x.shape != y.shape or x.ndim != 1 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("paired finite score/target vectors required")
    c = d = tx = ty = both = 0
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            sx = 1 if x[i] > x[j] else -1 if x[i] < x[j] else 0
            sy = 1 if y[i] > y[j] else -1 if y[i] < y[j] else 0
            if sx == sy != 0: c += 1
            elif sx and sy: d += 1
            elif not sx and sy: tx += 1
            elif sx and not sy: ty += 1
            else: both += 1
    denominator = math.sqrt((c + d + tx) * (c + d + ty))
    if not len(x):
        status = "NO_CANDIDATE"
    elif len(x) == 1:
        status = "INSUFFICIENT_PAIRS"
    elif denominator:
        status = "DEFINED"
    elif c == d == tx == ty == 0:
        status = "BOTH_CONSTANT"
    elif c == d == ty == 0:
        status = "CONSTANT_SCORE"
    else:
        status = "CONSTANT_TARGET"
    return {"status": status, "value": (c - d) / denominator if denominator else None,
            "concordant": c, "discordant": d, "score_only_ties": tx,
            "target_only_ties": ty, "double_ties": both, "pair_count": len(x) * (len(x) - 1) // 2}


def describe(partitions: Any, target: dict[str, np.ndarray], energy: np.ndarray,
             components: np.ndarray | None = None) -> dict[str, Any]:
    count = len(partitions)
    chosen = choose_energy(np.asarray(energy), partitions)
    orders = {name: order_record(target[name]) for name in ("tM", "tD")}
    errors = None
    if components is not None:
        h = np.asarray(components)
        if (h.dtype != np.float64 or h.shape != (count, 2)
                or not np.array_equal(np.asarray(energy), h.sum(1, dtype=np.float64))):
            raise ValueError("components do not reproduce energy exactly")
        squared = np.square(h - target["u32"].astype(np.float64))
        errors = {"mse": float(squared.mean(axis=1).mean()) if count else None}
    result = {"candidate_count": count, "chosen": chosen, "score_order": order_record(energy),
              "oracles": orders, "tau": kendall_tau_b(energy, target["tM"]), "errors": errors}
    if chosen is None:
        return {**result, "decision": None}
    return {**result, "decision": {
        "regret_tM": float(target["tM"][chosen] - orders["tM"]["minimum"]),
        "regret_tD": float(target["tD"][chosen] - orders["tD"]["minimum"]),
        "optimal_tM": chosen in orders["tM"]["optima"],
        "optimal_tD": chosen in orders["tD"]["optima"],
        "ari": float(target["ari"][chosen]), "exact": bool(target["exact"][chosen]),
        "k_absolute_error": int(abs(target["k_error"][chosen]))}}


def canonical_target_arithmetic(target: dict[str, np.ndarray]) -> None:
    """Enforce delivered arithmetic without consulting the independent oracle."""
    u64, u32 = target["u64"], target["u32"]
    if (u64.dtype != np.float64 or u32.dtype != np.float32
            or target["tM"].dtype != np.float64 or target["tD"].dtype != np.float64
            or u64.shape != u32.shape
            or u64.ndim != 2 or u64.shape[1] != 2
            or not np.array_equal(u32, u64.astype(np.float32))
            or not np.array_equal(target["tM"], np.asarray(
                [math.fsum(row.tolist()) for row in u64], np.float64))
            or not np.array_equal(target["tD"], np.asarray(
                [math.fsum(row.astype(np.float64).tolist()) for row in u32], np.float64))):
        raise ValueError("canonical u32/tM/tD arithmetic differs")


def stratum_metrics(partitions: Any, target: dict[str, np.ndarray], energy: np.ndarray,
                    definitions: dict[str, Any], components: np.ndarray | None = None) -> dict[str, Any]:
    """Recompute each declared stratum directly over its explicit candidate IDs."""
    count = len(partitions)
    result = {}
    covered: list[int] = []
    for name, raw_ids in definitions.items():
        if type(name) is not str:
            raise ValueError("stratum name must be explicit")
        ids = list(raw_ids)
        if (not ids or any(type(i) is not int or not 0 <= i < count for i in ids)
                or ids != sorted(set(ids))):
            raise ValueError("stratum candidate support differs")
        covered.extend(ids)
        score_order = order_record(energy, ids)
        t_m = order_record(target["tM"], ids)
        t_d = order_record(target["tD"], ids)
        chosen = score_order["chosen"]
        metrics = {key: None for key in METRICS}
        if chosen is not None:
            metrics.update(regret_tM=float(target["tM"][chosen] - t_m["minimum"]),
                regret_tD=float(target["tD"][chosen] - t_d["minimum"]),
                ari=float(target["ari"][chosen]), optimal_tM=chosen in t_m["optima"],
                optimal_tD=chosen in t_d["optima"], exact=bool(target["exact"][chosen]),
                k_absolute_error=int(abs(target["k_error"][chosen])),
                tau_b=kendall_tau_b(energy[ids], target["tM"][ids])["value"])
            if components is not None:
                errors = np.square(components[ids] - target["u32"][ids].astype(np.float64))
                metrics["mse_components"] = float(math.fsum(errors.ravel().tolist()) / errors.size)
        result[name] = {"candidate_ids": ids, "chosen": chosen, "metrics": metrics,
            "tau_status": kendall_tau_b(energy[ids], target["tM"][ids])["status"]}
    if sorted(covered) != list(range(count)) or len(covered) != len(set(covered)):
        raise ValueError("strata are not an exact partition of candidate support")
    return result


def observable_strata(partitions: Any, available: np.ndarray, extended_branch: list[str]) -> dict[str, Any]:
    count = len(partitions)
    if (available.dtype != np.bool_ or available.shape != (count, 3)
            or not isinstance(extended_branch, list) or len(extended_branch) != count
            or any(type(value) is not str for value in extended_branch)):
        raise ValueError("stratum source metadata differs")
    groups = {name: {} for name in ("full", "k", "sizes_available", "sizes_available_branch")}
    groups["full"]["all"] = list(range(count))
    for i, partition in enumerate(partitions):
        sizes = ",".join(str(size) for size in sorted(len(group) for group in partition))
        bits = "".join("1" if value else "0" for value in available[i])
        base = f"sizes={sizes};available={bits}"
        labels = {"k": f"k={len(partition)}", "sizes_available": base,
                  "sizes_available_branch": f"{base};branch={extended_branch[i]}"}
        for scheme, label in labels.items(): groups[scheme].setdefault(label, []).append(i)
    return {scheme: dict(sorted(values.items())) for scheme, values in groups.items()}


def validate_sham(partitions: Any, evidence: np.ndarray, sham: dict[str, Any]) -> None:
    count = len(partitions)
    if evidence.dtype != np.float32 or evidence.shape != (count, 6) or not np.isfinite(evidence).all():
        raise ValueError("sham evidence support differs")
    donors = sham.get("donors")
    if (not isinstance(donors, list) or any(type(i) is not int for i in donors)
            or sorted(donors) != list(range(count))):
        raise ValueError("sham donor permutation differs")
    by_size: dict[tuple[int, ...], list[int]] = {}
    for i, partition in enumerate(partitions):
        by_size.setdefault(tuple(sorted(len(group) for group in partition)), []).append(i)
    expected_sizes = sorted(by_size)
    records = sham.get("strata")
    if not isinstance(records, list) or [tuple(row.get("sizes", ())) for row in records] != expected_sizes:
        raise ValueError("sham size strata differ")
    changed_all = []
    for row, sizes in zip(records, expected_sizes):
        ids, assigned = by_size[sizes], [donors[i] for i in by_size[sizes]]
        changed = [bool(np.any(evidence[i] != evidence[j])) for i, j in zip(ids, assigned)]
        shift = row.get("shift")
        changed_fraction = row.get("changed_fraction")
        expected_assigned = ids if shift is None else [ids[(i + shift) % len(ids)] for i in range(len(ids))]
        expected_status = "NO_PERMUTATION" if shift is None else "INPUT_CHANGED" if any(changed) else "INPUT_UNCHANGED"
        if (row.get("candidate_ids") != ids or row.get("donors") != assigned or assigned != expected_assigned
                or (shift is None) != (len(ids) == 1) or (shift is not None and (type(shift) is not int or not 1 <= shift < len(ids)))
                or type(changed_fraction) is not float or not math.isfinite(changed_fraction)
                or not math.isclose(changed_fraction, math.fsum(changed) / len(changed), abs_tol=ATOL, rel_tol=0)
                or row.get("status") != expected_status):
            raise ValueError("sham stratum arithmetic differs")
        changed_all.extend(changed)
    expected_mask = [bool(np.any(evidence[i] != evidence[donors[i]])) for i in range(count)]
    expected_status = "INPUT_CHANGED" if any(expected_mask) else "INPUT_UNCHANGED"
    if sham.get("changed_mask") != expected_mask or sham.get("status") != expected_status:
        raise ValueError("sham global mask/status differs")


def margin(values: np.ndarray) -> float | None:
    ordered = sorted(float(value) for value in values)
    return None if len(ordered) < 2 else ordered[1] - ordered[0]


def transport_diagnostic(arrays: dict[str, np.ndarray], partitions: Any,
                         routed_inputs: dict[str, np.ndarray], bypass_channel: int | None,
                         parent_components: np.ndarray) -> dict[str, Any]:
    """Derive transport permutations, restored values, margins and choices."""
    count = len(partitions)
    ps = [canonical_partition(p, sum(map(len, partitions[0]))) for p in partitions]
    expected_orders = {
        "candidate_order": np.arange(count - 1, -1, -1, dtype=np.int64),
        "group_order": np.arange(len(routed_inputs["groups"]) - 1, -1, -1, dtype=np.int64),
        "channel_order": np.arange(7, -1, -1, dtype=np.int64),
        "weight_column_order": np.r_[np.arange(33), 33 + np.arange(7, -1, -1)].astype(np.int64),
    }
    required = set(expected_orders) | {"transported_inputs/" + key for key in routed_inputs} | {
        "baseline/components", "baseline/energy", "baseline/offsets", "transported_components",
        "restored_components", "transported_energy", "restored_energy", "batched_components"}
    if set(arrays) != required:
        raise ValueError("transport array schema differs")
    for key, expected in expected_orders.items():
        assert_exact(arrays[key], expected, "transport " + key)
    co = arrays["candidate_order"]
    numeric_shapes = {"baseline/components": (count, 2), "baseline/energy": (count,),
        "transported_components": (count, 2), "restored_components": (count, 2),
        "transported_energy": (count,), "restored_energy": (count,), "batched_components": (count, 2)}
    if (arrays["baseline/offsets"].dtype != np.int64
            or not np.array_equal(arrays["baseline/offsets"], np.array([0, count], np.int64))
            or any(arrays[key].dtype != np.float64 or arrays[key].shape != shape
                   or not np.isfinite(arrays[key]).all() for key, shape in numeric_shapes.items())):
        raise ValueError("transport numeric dtype, support or finiteness differs")
    parent = np.asarray(parent_components)
    if parent.dtype != np.float64 or parent.shape != (count, 2):
        raise ValueError("transport parent prediction slice differs")
    assert_exact(arrays["batched_components"], parent,
                 "transport batched components vs parent prediction")
    for key, expected in {
        "transported_inputs/groups": routed_inputs["groups"][arrays["group_order"]],
        "transported_inputs/globals": routed_inputs["globals"][co],
        "transported_inputs/incidence": routed_inputs["incidence"][np.ix_(co, arrays["group_order"])],
        "transported_inputs/evidence": routed_inputs["evidence"][np.ix_(co, arrays["channel_order"])],
    }.items():
        assert_exact(arrays[key], expected, key)
    assert_exact(arrays["restored_components"], arrays["transported_components"][np.argsort(co)], "transport inverse")
    assert_exact(arrays["baseline/energy"], arrays["baseline/components"].sum(1, dtype=np.float64), "baseline energy")
    assert_exact(arrays["transported_energy"], arrays["transported_components"].sum(1, dtype=np.float64), "transport energy")
    assert_exact(arrays["restored_energy"], arrays["restored_components"].sum(1, dtype=np.float64), "restored energy")
    moved_ps = [ps[int(i)] for i in co]
    before = choose_energy(arrays["baseline/energy"], ps)
    after = choose_energy(arrays["transported_energy"], moved_ps)
    restored, baseline = arrays["restored_components"], arrays["baseline/components"]
    singleton_energy = arrays["batched_components"].sum(1, dtype=np.float64)
    result = {"atol": 1e-6, "rtol": 1e-5,
        "within_numeric_tolerance": bool(np.allclose(restored, baseline, atol=1e-6, rtol=1e-5)),
        "max_component_error": float(np.max(np.abs(restored - baseline))),
        "max_energy_error": float(np.max(np.abs(arrays["restored_energy"] - arrays["baseline/energy"]))),
        "same_exact_choice": ps[before] == moved_ps[after],
        "original_margin": margin(arrays["baseline/energy"]),
        "transported_margin": margin(arrays["transported_energy"]),
        "original_choice": ps[before], "transported_choice": moved_ps[after],
        "scope": "coordinate transport and float32 reduction stability, not learned physical invariance",
        "bypass_channel": bypass_channel,
        "singleton_vs_batch_max_error": float(np.max(np.abs(baseline - arrays["batched_components"]))),
        "singleton_vs_batch_same_exact_choice": before == choose_energy(singleton_energy, ps)}
    return result


def primary(regret: np.ndarray, eligible: np.ndarray, *, check=lambda: None) -> dict[str, Any]:
    values, mask = np.asarray(regret), np.asarray(eligible)
    if (values.dtype != np.float64 or values.ndim != 4 or values.shape[1:] != (8, 3, 3)
            or len(values) != len(mask) or mask.dtype != np.bool_
            or not np.isfinite(values[mask]).all() or np.any(values[mask] < 0)
            or not np.isnan(values[~mask]).all()):
        raise ValueError("primary requires complete scene/arm/cell support")
    scene_ids = np.flatnonzero(mask).astype(np.int64)
    cells = values[mask].reshape(len(scene_ids), 8, 9)
    means = np.asarray([[math.fsum(cells[s, a].tolist()) / 9 for a in range(8)]
                        for s in range(len(scene_ids))], np.float64).reshape(len(scene_ids), 8)
    arm = lambda name: means[:, ARMS.index(name)]
    first = arm("geometric_mse") - arm("injection_mse")
    second = arm("geometric_decision") - arm("injection_decision")
    deltas = np.column_stack((first, second, second - first,
                              arm("geometric_decision") - arm("decoupled_decision")))
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    indices = (rng.integers(0, len(scene_ids), size=(BOOTSTRAP_COUNT, len(scene_ids)), dtype=np.int64)
               if len(scene_ids) else np.empty((BOOTSTRAP_COUNT, 0), np.int64))
    distribution = np.empty((BOOTSTRAP_COUNT, 4), np.float64)
    if len(scene_ids):
        for begin in range(BOOTSTRAP_COUNT):
            check()
            weights = np.bincount(indices[begin], minlength=len(scene_ids))
            for contrast in range(4):
                distribution[begin, contrast] = math.fsum(
                    float(weights[i]) * float(deltas[i, contrast]) for i in range(len(scene_ids))) / len(scene_ids)
        def linear_quantile(column: np.ndarray, probability: float) -> float:
            ordered = sorted(float(value) for value in column)
            position = (len(ordered) - 1) * probability
            lower, upper = math.floor(position), math.ceil(position)
            return ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower])
        interval = np.asarray([[linear_quantile(distribution[:, j], p) for j in range(4)]
                               for p in (0.00625, 0.99375)], np.float64)
    else:
        distribution[:] = np.nan
        interval = None
    summary = {"schema": "geometric-decision-primary-v1", "total_scenes": len(values),
        "eligible_scenes": len(scene_ids), "empty_scenes": len(values) - len(scene_ids),
        "conditioned_cells_per_arm": 9, "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_count": BOOTSTRAP_COUNT, "confidence_level": 0.9875, "unit": "scene",
        "contrasts": {name: {"mean": float(deltas[:, i].mean()) if len(scene_ids) else None,
                              "interval": interval[:, i].tolist() if len(scene_ids) else None}
                      for i, name in enumerate(CONTRASTS)}}
    return {"summary": summary, "arrays": {"eligible_scene_ids": scene_ids,
        "arm_scene_means": means, "scene_contrasts": deltas,
        "bootstrap_indices": indices, "bootstrap_distribution": distribution}}


def _supported_mean(values: np.ndarray, axes: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    axes = tuple(sorted(axis % values.ndim for axis in axes))
    retained = tuple(i for i in range(values.ndim) if i not in axes)
    shape = tuple(values.shape[i] for i in retained)
    means, counts = np.full(shape, np.nan, np.float64), np.zeros(shape, np.int64)
    for out_index in np.ndindex(shape):
        selector: list[Any] = [slice(None)] * values.ndim
        for axis, coordinate in zip(retained, out_index): selector[axis] = coordinate
        vector = np.asarray(values[tuple(selector)]).ravel()
        finite = [float(x) for x in vector if math.isfinite(float(x))]
        counts[out_index] = len(finite)
        if finite: means[out_index] = math.fsum(finite) / len(finite)
    return means, counts


def _plain(value: Any) -> Any:
    if isinstance(value, np.ndarray): return _plain(value.tolist())
    if isinstance(value, np.generic): return _plain(value.item())
    if isinstance(value, float) and math.isnan(value): return None
    if isinstance(value, dict): return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)): return [_plain(item) for item in value]
    return value


def aggregate_summary(learned: np.ndarray, classical: np.ndarray, eligible: np.ndarray,
                      presence: list[str], *, _nested: bool = False) -> dict[str, Any]:
    """Rebuild all descriptive report tables from complete preserved matrices."""
    if (learned.dtype != np.float64 or learned.ndim != 6 or learned.shape[1:] != (2, 8, 3, 3, 9)
            or classical.dtype != np.float64 or classical.shape != (len(learned), 4, 9)
            or eligible.dtype != np.bool_ or eligible.shape != (len(learned),)
            or len(presence) != len(learned) or any(value not in ("pool", "neighbor", "absent") for value in presence)):
        raise ValueError("summary matrices or presence roster differ")
    def summary(values: np.ndarray) -> dict[str, Any]:
        mean, count = _supported_mean(values, (0,))
        return {"mean": mean, "defined_scenes": count, "total_scenes": len(values)}
    scene_means, cell_counts = _supported_mean(learned, (3, 4))
    gap, _ = _supported_mean(learned[:, 1] - learned[:, 0], (2, 3))
    loss_effect = np.stack([learned[:, 1, 2 * i + 1] - learned[:, 1, 2 * i] for i in range(4)], axis=1)
    loss_means, _ = _supported_mean(loss_effect, (2, 3))
    result = {"metric_order": METRICS, "arm_order": ARMS, "stage_order": STAGES,
        "checkpoint_order": CHECKPOINTS, "reader_order": READER_SEEDS, "classical_order": CLASSICAL,
        "eligible_scenes": int(eligible.sum()), "total_scenes": len(eligible),
        "arm_scene_first": summary(scene_means), "cell_descriptive": summary(learned),
        "defined_cells_by_scene": cell_counts, "selected_minus_initial": summary(gap),
        "decision_minus_mse_by_route": summary(loss_means), "classical": summary(classical),
        "selected_minus_classical": {name: summary(_supported_mean(
            learned[:, 1] - classical[:, j, None, None, None, :], (2, 3))[0])
            for j, name in enumerate(CLASSICAL)},
        "authority": "descriptive; conditioned cells are not independent scenes"}
    if not _nested:
        result["presence"] = {name: aggregate_summary(learned[np.asarray([p == name for p in presence])],
            classical[np.asarray([p == name for p in presence])], eligible[np.asarray([p == name for p in presence])],
            [p for p in presence if p == name], _nested=True) for name in ("pool", "neighbor", "absent")}
    return _plain(result)


def labels_by_event(original: dict[str, Any], derived: dict[str, Any], labels: np.ndarray) -> np.ndarray:
    raw_first, raw_second = original["canonical_to_observed"], derived["canonical_to_observed"]
    if any(type(i) is bool or not isinstance(i, (int, np.integer)) for seq in (raw_first, raw_second) for i in seq):
        raise ValueError("event correspondence identities must be integers")
    first = np.asarray(raw_first, np.int64)
    second = np.asarray(raw_second, np.int64)
    labels = np.asarray(labels)
    if labels.ndim != 1 or labels.dtype.kind not in "iu":
        raise ValueError("canonical labels must retain integer identity")
    n = len(labels)
    if (first.shape != (n,) or second.shape != (n,)
            or not np.array_equal(np.sort(first), np.arange(n))
            or not np.array_equal(np.sort(second), np.arange(n))):
        raise ValueError("event correspondence is not a bijection")
    event_labels = np.empty(n, np.int64)
    event_labels[first] = labels
    return event_labels[second]


def event_universe(partitions: Any, rank_to_event: Any) -> list[tuple[tuple[int, ...], ...]]:
    order = np.asarray(rank_to_event)
    if (order.ndim != 1 or not len(order) or order.dtype.kind not in "iu"
            or sorted(order.tolist()) != list(range(len(order)))):
        raise ValueError("event identity requires a bijection")
    result = []
    for partition in partitions:
        p = canonical_partition(partition, len(order))
        result.append(tuple(sorted(tuple(sorted(int(order[i]) for i in group)) for group in p)))
    if len(result) != len(set(result)):
        raise ValueError("duplicate event partition")
    return result


def roundtrip_comparison(before: dict[str, Any], after: dict[str, Any], *,
                         energy_before: np.ndarray, energy_after: np.ndarray,
                         evidence_before: np.ndarray, evidence_after: np.ndarray,
                         choice_before: int | None, choice_after: int | None) -> dict[str, Any]:
    first = event_universe(before["partitions"], before["canonical_to_observed"])
    second = event_universe(after["partitions"], after["canonical_to_observed"])
    if len(before["canonical_to_observed"]) != len(after["canonical_to_observed"]):
        raise ValueError("roundtrip replaced the event universe")
    for keys, energy, evidence, choice in ((first, energy_before, evidence_before, choice_before),
                                           (second, energy_after, evidence_after, choice_after)):
        if (energy.dtype != np.float64 or energy.shape != (len(keys),)
                or evidence.dtype != np.float32 or evidence.shape != (len(keys), 8)
                or not np.isfinite(energy).all() or not np.isfinite(evidence).all()):
            raise ValueError("roundtrip arrays have wrong extent")
        if (not keys and choice is not None) or (keys and (type(choice) is not int or not 0 <= choice < len(keys))):
            raise ValueError("roundtrip choice is outside support")
        if keys and energy[choice] != np.min(energy):
            raise ValueError("roundtrip choice is not a minimum")
    a, b = {key: i for i, key in enumerate(first)}, {key: i for i, key in enumerate(second)}
    common = sorted(a.keys() & b.keys())
    left = np.asarray([a[key] for key in common], np.int64)
    right = np.asarray([b[key] for key in common], np.int64)
    error = (np.abs(evidence_before[left].astype(np.float64) - evidence_after[right].astype(np.float64))
             if common else np.empty((0, 8), np.float64))
    return {"before_candidates": len(first), "after_candidates": len(second),
        "common_candidates": len(common), "lost_candidates": sorted(a.keys() - b.keys()),
        "added_candidates": sorted(b.keys() - a.keys()), "common_event_partitions": common,
        "before_indices": left.tolist(), "after_indices": right.tolist(),
        "before_empty": not first, "after_empty": not second,
        "max_energy_error_common": float(np.max(np.abs(energy_before[left] - energy_after[right]))) if common else None,
        "max_channel_error_common": error.max(0).tolist() if common else [None] * 8,
        "same_exact_choice_by_event": None if choice_before is None or choice_after is None else first[choice_before] == second[choice_after],
        "before_choice_by_event": None if choice_before is None else first[choice_before],
        "after_choice_by_event": None if choice_after is None else second[choice_after],
        "scope": "joint quantization/recentering effect; common-candidate differences are conditioned on support"}


def coordinate_comparison(coordinates: dict[str, np.ndarray]) -> dict[str, Any]:
    dtypes = {"original": np.float32, "shifted64": np.float64, "shifted32": np.float32,
              "q_center": np.float32, "q_probe": np.float32}
    if set(coordinates) != set(dtypes):
        raise ValueError("coordinate fields differ")
    n = len(coordinates["original"])
    for key, dtype in dtypes.items():
        if (coordinates[key].dtype != dtype or coordinates[key].shape != (n,) or n < 2
                or not np.isfinite(coordinates[key]).all()):
            raise ValueError("coordinate dtype or extent differs")
    relation = lambda q: q.astype(np.float64)[:, None] - q.astype(np.float64)[None, :]
    pairs = (("original", "shifted64"), ("shifted64", "shifted32"),
             ("shifted32", "q_probe"), ("original", "q_probe"), ("original", "q_center"))
    return {"event_count": n,
        "tie_counts": {key: n - len(np.unique(value)) for key, value in coordinates.items()},
        "max_pairwise_log_ratio_change": {a + "_to_" + b: float(np.max(np.abs(relation(coordinates[a]) - relation(coordinates[b]))))
                                           for a, b in pairs},
        "scope": "numerical coordinate changes; q_center is diagnostic only, not another pipeline"}


def select_structural_cut(scene_rows: list[dict[str, Any]], probe_ids: list[int]) -> dict[str, Any]:
    """Pure selection from sealed observable metadata, never metric payloads."""
    if len(scene_rows) != 512 or [row.get("scene_id") for row in scene_rows] != list(range(512)):
        raise ValueError("structural selection requires ordered 512-scene metadata")
    counts = []
    for row in scene_rows:
        partitions = row.get("partitions")
        if not isinstance(partitions, list):
            raise ValueError("scene metadata lacks explicit candidate roster")
        counts.append(len(partitions))
    eligible = [i for i, count in enumerate(counts) if count]
    if probe_ids != eligible[:4] or len(probe_ids) > 4:
        raise ValueError("probe roster is not the first min(4, eligible) scenes")
    reasons: dict[int, list[str]] = {0: ["boundary_zero"], 511: ["boundary_last"]}
    empty = next((i for i, count in enumerate(counts) if count == 0), None)
    if empty is not None:
        reasons.setdefault(empty, []).append("first_empty")
    maximum = max(counts)
    max_id = counts.index(maximum)
    reasons.setdefault(max_id, []).append("first_max_candidates")
    for sid in probe_ids:
        reasons.setdefault(sid, []).append("sealed_roundtrip_original")
    return {"scene_ids": sorted(reasons), "reasons": {str(k): reasons[k] for k in sorted(reasons)},
            "probe_scene_ids": probe_ids, "probe_count": len(probe_ids),
            "first_empty": empty, "maximum_candidate_count": maximum, "first_max_scene_id": max_id}


def assert_exact(actual: Any, expected: Any, label: str) -> None:
    if isinstance(actual, np.ndarray) or isinstance(expected, np.ndarray):
        if not (isinstance(actual, np.ndarray) and isinstance(expected, np.ndarray)
                and actual.dtype == expected.dtype and actual.shape == expected.shape
                and np.array_equal(actual, expected, equal_nan=True)):
            raise ValueError(label + " differs exactly")
    elif actual != expected:
        raise ValueError(label + " differs exactly")


def assert_close(actual: Any, expected: Any, label: str, *, atol: float = ATOL) -> None:
    a, b = np.asarray(actual), np.asarray(expected)
    if a.shape != b.shape or not np.allclose(a, b, atol=atol, rtol=0, equal_nan=True):
        raise ValueError(label + " exceeds declared tolerance")
