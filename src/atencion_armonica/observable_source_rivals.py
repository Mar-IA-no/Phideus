"""Finite generative fits of observable partitions, without supervision ports.

See PROTOCOL_OBSERVABLE_SOURCE_RIVALS.md. Bounds concern the discrete grid,
not continuous identifiability. Saved group factors support replay without
another grid sweep. This module does not import the frozen learned pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product
import hashlib
import json

import numpy as np

CENTS_TO_LOG = float(np.log(2.) / 1200.)
SIGMA = 2 * CENTS_TO_LOG
TOL_J = 1e-7
WIDTH = float(np.log(5.))
BRANCHES = {
    "base-low": ((1e-4, 1e-3), (0., 0.), (2, 3, 4)),
    "base-high": ((3e-3, 1e-2), (0., 0.), (2, 3)),
    "deformed-low": ((1e-4, 1e-3), (5e-6, 5e-5), (2, 3)),
}
FAMILIES = {"base": ("base-low", "base-high"), "extended": tuple(BRANCHES)}


def signature(groups):
    return tuple(sorted(tuple(sorted(g)) for g in groups))


def signature_key(groups):
    return json.dumps(signature(groups), separators=(",", ":"), ensure_ascii=True)


def validate_partition(groups, n):
    s = signature(groups)
    if any(type(i) is not int for g in s for i in g):
        raise ValueError("event indices must be integers")
    if any(not g for g in s) or sorted(i for g in s for i in g) != list(range(n)):
        raise ValueError("not a complete disjoint partition")
    return s


def supported(groups):
    return len(groups) in (2, 3, 4) and all(4 <= len(g) <= 8 for g in groups)


def candidate_inventory(pools, n):
    """Exact union, then hash-ordered one-step neighbors; never cost-selected."""
    provenance = {}
    for origin, pool in sorted(pools.items()):
        for p in pool:
            s = validate_partition(p, n)
            provenance.setdefault(s, set()).add(origin)
    neighbors = {}
    for p in sorted(provenance):
        if not supported(p):
            continue
        for a, b in combinations(range(len(p)), 2):
            for x, y in product(p[a], p[b]):
                q = [list(g) for g in p]
                q[a].remove(x); q[a].append(y)
                q[b].remove(y); q[b].append(x)
                neighbors.setdefault(signature(q), set()).add(p)
        for a, b in product(range(len(p)), repeat=2):
            if a == b or len(p[a]) <= 4 or len(p[b]) >= 8:
                continue
            for x in p[a]:
                q = [list(g) for g in p]
                q[a].remove(x); q[b].append(x)
                neighbors.setdefault(signature(q), set()).add(p)
    for p in provenance:
        neighbors.pop(p, None)
    ordered = sorted(neighbors, key=lambda p: (
        hashlib.sha256(signature_key(p).encode("ascii")).hexdigest(), p))[:64]
    rows = [{"partition": p, "origin": "pool", "checkpoints": sorted(origins),
             "status": "SUPPORTED" if supported(p) else "OUTSIDE_GENERATIVE_CARDINALITY"}
            for p, origins in sorted(provenance.items())]
    rows.extend({"partition": p, "origin": "neighbor", "parents": sorted(neighbors[p]),
                 "status": "SUPPORTED"} for p in ordered)
    return {"candidates": rows, "neighbor_count_before_cap": len(neighbors),
            "neighbor_count_retained": len(ordered), "pool_count": len(provenance)}


@dataclass(frozen=True)
class Grid:
    beta_count: int = 257
    gamma_count: int = 65
    stride: int = 4

    def __post_init__(self):
        if self.stride != 4 or any(type(n) is not int or n < 5 or (n-1) % 4
                                   for n in (self.beta_count, self.gamma_count)):
            raise ValueError("nested grids require 4j+1 points and stride four")

    def values(self, branch):
        b, g, _ = BRANCHES[branch]
        return (np.geomspace(*b, self.beta_count),
                np.array([0.]) if g[1] == 0 else np.geomspace(*g, self.gamma_count))


def template(indices, beta, gamma):
    n = np.asarray(indices, dtype=np.float64)
    return np.log(n) + .5 * np.log1p(beta*n**2 + gamma*n**4)


def centered(q):
    q = np.asarray(q, dtype=np.float64)
    if q.ndim != 1 or not len(q) or not np.isfinite(q).all():
        raise ValueError("expected a finite nonempty vector")
    return q - q.mean()


def observable_q32(q):
    """Validate the observable port even when no supported partition exists."""
    q = np.asarray(q, dtype=np.float64)
    if (q.ndim != 1 or not 8 <= len(q) <= 32 or not np.isfinite(q).all()
            or not np.array_equal(q, q.astype(np.float32).astype(np.float64))
            or np.any(np.diff(q) < 0)):
        raise ValueError("expected a canonical, finite, exact q32 vector of length 8..32")
    return q


def project_offsets(a, weights):
    """Weighted projection on range <= log(5), with batches on axis zero."""
    a = np.asarray(a, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if (a.ndim not in (1, 2) or weights.shape != (a.shape[-1],)
            or not np.isfinite(a).all() or not np.isfinite(weights).all()
            or np.any(weights <= 0)):
        raise ValueError("invalid projection inputs")
    one = a.ndim == 1
    a = np.atleast_2d(a)
    lo = a.min(axis=1) - WIDTH
    hi = a.max(axis=1)
    for _ in range(64):
        mid = (lo + hi) / 2
        trial = np.clip(a, mid[:, None], mid[:, None] + WIDTH)
        derivative = ((trial-a)*weights).sum(axis=1)
        hi = np.where(derivative > 0, mid, hi)
        lo = np.where(derivative > 0, lo, mid)
    b = np.clip(a, ((lo+hi)/2)[:, None], ((lo+hi)/2+WIDTH)[:, None])
    b = np.where((np.ptp(a, axis=1) <= WIDTH)[:, None], a, b)
    return b[0] if one else b


def witness_metrics(q, prediction):
    q32 = np.asarray(q, dtype=np.float32)
    q64 = q32.astype(np.float64)
    if not np.array_equal(np.asarray(q), q64):
        raise ValueError("quantization diagnostics require exact q32 input")
    pred = np.asarray(prediction, dtype=np.float64)
    if pred.shape != q64.shape or not np.isfinite(pred).all():
        raise ValueError("invalid predicted vector")
    residual = centered(q64) - centered(pred)
    sse = float(np.square(residual).sum())
    eps = .5*np.maximum(q64-np.nextafter(q32, np.float32(-np.inf)).astype(np.float64),
                        np.nextafter(q32, np.float32(np.inf)).astype(np.float64)-q64)
    e = float(np.linalg.norm(eps))
    return {"J": sse/(2*SIGMA**2), "rms_cents": float(np.sqrt(sse/len(q64))/CENTS_TO_LOG),
            "quantization_objective_bound": (2*np.sqrt(sse)*e+e**2)/(2*SIGMA**2),
            "sample_fit_status": "EXACT_SAMPLE_FIT" if np.array_equal(
                centered(pred).astype(np.float32), q32) else "INEXACT_SAMPLE_FIT",
            "noiseless_collision_status": "NOT_TESTED"}


class GroupFitter:
    """Batched exhaustive group sweeps. CUDA is opt-in, not auto-selected."""
    def __init__(self, grid=Grid(), device="cpu", assignment_batch=8):
        if device not in ("cpu", "cuda") or not 1 <= assignment_batch <= 8:
            raise ValueError("invalid backend or assignment batch")
        self.grid, self.device, self.assignment_batch = grid, device, assignment_batch
        self.banks = {}
        self.torch = None
        if device == "cuda":
            import torch
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            self.torch = torch

    def _bank(self, m, branch):
        key = m, branch
        if key not in self.banks:
            ns = np.asarray(list(combinations(range(1, 9), m)), dtype=np.int64)
            beta, gamma = self.grid.values(branch)
            t = template(ns[:, None, None, :], beta[None, :, None, None],
                         gamma[None, None, :, None])
            tc = t-t.mean(axis=-1, keepdims=True)
            if self.torch is not None:
                tc = self.torch.as_tensor(tc, dtype=self.torch.float64, device="cuda")
            self.banks[key] = ns, beta, gamma, tc
        return self.banks[key]

    def fit(self, observations, branch):
        """Same-sized, ascending groups, <=8 per batch. Results are JSON-ready."""
        ys = np.asarray(observations, dtype=np.float64)
        if (ys.ndim != 2 or not 1 <= len(ys) <= 8 or not 4 <= ys.shape[1] <= 8
                or not np.isfinite(ys).all() or np.any(np.diff(ys, axis=1) < 0)):
            raise ValueError("expected one to eight sorted supported groups")
        ns, beta, gamma, bank = self._bank(ys.shape[1], branch)
        yc = ys-ys.mean(axis=-1, keepdims=True)
        if self.torch is not None:
            yc = self.torch.as_tensor(yc, dtype=self.torch.float64, device="cuda")
        output = [{"branch": branch, "size": ys.shape[1], "mean_observed": float(y.mean()),
                   "fine": [], "coarse": []} for y in ys]
        for start in range(0, len(ns), self.assignment_batch):
            tc = bank[start:start+self.assignment_batch]
            error = yc[:, None, None, None, :]-tc[None, ...]
            if self.torch is None:
                costs = np.square(error).sum(axis=-1)
            else:
                costs = error.square().sum(dim=-1)
            for resolution, step in (("fine", 1), ("coarse", self.grid.stride)):
                c = costs[:, :, ::step, ::step].reshape(len(ys), len(tc), -1)
                if self.torch is None:
                    ix = c.argmin(axis=-1)
                    values = np.take_along_axis(c, ix[..., None], axis=-1)[..., 0]
                else:
                    values, ix = c.min(dim=-1)
                    values, ix = values.cpu().numpy(), ix.cpu().numpy()
                for row, record in enumerate(output):
                    for local, assignment in enumerate(ns[start:start+len(tc)]):
                        bi, gi = np.unravel_index(int(ix[row, local]),
                                                (len(beta[::step]), len(gamma[::step])))
                        bi, gi = int(bi*step), int(gi*step)
                        t = template(assignment, beta[bi], gamma[gi])
                        record[resolution].append({"indices": assignment.tolist(),
                            "beta_index": bi, "gamma_index": gi, "beta": float(beta[bi]),
                            "gamma": float(gamma[gi]), "template": t.tolist(),
                            "template_mean": float(t.mean()), "sse": float(values[row, local])})
        return output


def _factor_key(factor):
    return tuple(factor["indices"]), factor["beta_index"], factor["gamma_index"]


def joint_search(y, partition, factors, resolution):
    """Reconstruct all shortlisted products from saved factors, no grid sweep."""
    shortlist = [sorted(f[resolution], key=lambda x: (x["sse"], _factor_key(x)))[:4]
                 for f in factors]
    choices = list(product(*shortlist))
    # Exact numerical ties use the full parameter lexicographic key, not local rank.
    choices.sort(key=lambda fs: tuple(_factor_key(f) for f in fs))
    offsets = np.array([[float(np.mean(y[list(g)]))-f["template_mean"]
                         for g, f in zip(partition, fs)] for fs in choices])
    b = project_offsets(offsets, np.array([len(g) for g in partition]))
    log_f0 = b-b.min(axis=1, keepdims=True)+np.log(100.)
    prediction = np.empty((len(choices), len(y)), dtype=np.float64)
    for j, g in enumerate(partition):
        prediction[:, list(g)] = (log_f0[:, j, None]
            +np.array([fs[j]["template"] for fs in choices]))
    prediction -= prediction.mean(axis=1, keepdims=True)
    costs = np.square(y[None, :]-prediction).sum(axis=1)/(2*SIGMA**2)
    best = int(np.argmin(costs))
    return {"J": float(costs[best]), "resolution": resolution,
            "evaluated_products": len(choices), "sources": list(choices[best]),
            "f0": np.exp(log_f0[best]).tolist(), "prediction": prediction[best].tolist()}


def partition_fit(q, partition, group_factors):
    """Compose saved group factors with scene-level branch/support constraints."""
    q = observable_q32(q)
    y = centered(q)
    p = validate_partition(partition, len(y))
    if not supported(p):
        return {"partition": p, "status": "OUTSIDE_GENERATIVE_CARDINALITY"}
    branches = {}
    for branch, (_, _, ks) in BRANCHES.items():
        if len(p) not in ks:
            continue
        fs = [group_factors[(branch, g)] for g in p]
        coarse = joint_search(y, p, fs, "coarse")
        fine = joint_search(y, p, fs, "fine")
        # Coarse parameter indices already name the identical fine-grid points.
        winner = min((coarse, fine), key=lambda w: (
            w["J"], tuple(_factor_key(f) for f in w["sources"])))
        lb = sum(min(f["sse"] for f in row["fine"]) for row in fs)/(2*SIGMA**2)
        if lb > winner["J"]+TOL_J or winner["J"] > coarse["J"]+TOL_J:
            raise ArithmeticError("discrete bounds or coarse nesting violated")
        metrics = witness_metrics(q, winner["prediction"])
        if abs(metrics["J"]-winner["J"]) > TOL_J:
            raise ArithmeticError("witness reconstruction differs")
        branches[branch] = {"LB": float(lb), "UB": winner["J"], "witness": winner,
                            "coarse": coarse, "fine_shortlist": fine, **metrics}
    families = {}
    for family, names in FAMILIES.items():
        allowed = [b for b in names if b in branches]
        low = min(allowed, key=lambda b: (branches[b]["LB"], b))
        upper = min(allowed, key=lambda b: (branches[b]["UB"], b))
        families[family] = {"LB": branches[low]["LB"], "UB": branches[upper]["UB"],
                            "lower_branch": low, "upper_branch": upper}
    return {"partition": p, "status": "FITTED", "branches": branches, "families": families}


def fit_candidates(q, partitions, fitter):
    """Observable-only entry point; return factors for deterministic replay."""
    q = observable_q32(q)
    y = centered(q)
    ps = sorted(set(validate_partition(p, len(q)) for p in partitions))
    valid = [p for p in ps if supported(p)]
    factors = {}
    for branch, (_, _, ks) in BRANCHES.items():
        groups = sorted({g for p in valid if len(p) in ks for g in p})
        for m in range(4, 9):
            gm = [g for g in groups if len(g) == m]
            for start in range(0, len(gm), 8):
                batch = gm[start:start+8]
                rows = fitter.fit([y[list(g)] for g in batch], branch)
                factors.update({(branch, g): row for g, row in zip(batch, rows)})
    return {"fits": [partition_fit(q, p, factors) for p in ps],
            "group_factors": [{"branch": b, "group": g, "factor": f}
                              for (b, g), f in sorted(factors.items())]}


def replay_fits(q, saved_factors, partitions):
    q = observable_q32(q)
    fs = {(row["branch"], tuple(row["group"])): row["factor"] for row in saved_factors}
    return [partition_fit(q, p, fs) for p in sorted(set(signature(p) for p in partitions))]


def rival_margin(observable_fits, planted_fit, family):
    """Evaluation helper with explicit privileged reference, not a search input."""
    truth = signature(planted_fit["partition"])
    rivals = [f for f in observable_fits if f["status"] == "FITTED"
              and signature(f["partition"]) != truth]
    if not rivals:
        return {"status": "NO_OBSERVABLE_RIVAL", "delta_UB": None, "interval": None,
                "lower_partition": None, "upper_partition": None}
    low = min(rivals, key=lambda f: (f["families"][family]["LB"], signature(f["partition"])))
    upper = min(rivals, key=lambda f: (f["families"][family]["UB"], signature(f["partition"])))
    t = planted_fit["families"][family]
    dl = low["families"][family]["LB"]-t["UB"]
    du = upper["families"][family]["UB"]-t["LB"]
    status = ("RIVAL_BETTER_ON_GRID" if du < -TOL_J else
              "PLANTED_BETTER_ON_GRID" if dl > TOL_J else "UNRESOLVED_GRID_COMPARISON")
    return {"status": status, "delta_UB": upper["families"][family]["UB"]-t["UB"],
            "interval": [dl, du], "lower_partition": low["partition"],
            "upper_partition": upper["partition"]}
