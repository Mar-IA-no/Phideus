"""Fixed cumulative operator limits, charged across resumable attempts.

The caller holds an exclusive process lock before opening an attempt. A lost
finish receipt consumes the entire reservation; no silent crash-time refund.
Resource samples do not claim an OS-enforced memory or disk quota.
"""
from __future__ import annotations

import math
import resource
import shutil
import time

from .geometric_decision_store import BASES

STAGES = {"profile": 600., "open": 1800., "training": 14400.,
          "fresh": 21600., "evaluation": 7200., "audit": 3600.}
LIMITS = {"stages": STAGES, "total_seconds": 49200., "rss_bytes": 8*1024**3,
          "vram_bytes": 8*1024**3, "new_bytes": 100*1024**3, "free_bytes": 30*1024**3}


class BudgetExceeded(RuntimeError):
    pass


def _seconds(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def owned_bytes(roots):
    total = 0
    for root in roots:
        for path in root.rglob("*"):
            if path.is_symlink():
                # pytest uses relative "current" links inside the owned audit
                # tree. Count the link itself, never read/follow its target.
                total += path.lstat().st_size
            elif path.is_file():
                total += path.stat().st_size
    return total


class StageBudget:
    def __init__(self, store, stage, *, manifest_ref, reservation_seconds, prior_charges, output_roots,
                 clock=time.monotonic, started_at=None, rss=None, disk_free=None, bytes_used=None, vram=None):
        if store.binding.get("limits") != LIMITS or stage not in STAGES:
            raise ValueError("fixed protocol limits and stage must be bound before launch")
        if not isinstance(store.json(manifest_ref), dict):
            raise ValueError("attempt requires an authenticated stage manifest")
        self.manifest_ref = manifest_ref.copy()
        if not _seconds(reservation_seconds) or reservation_seconds <= 0:
            raise ValueError("positive bounded attempt reservation required")
        if (not isinstance(prior_charges, list)
                or any(set(r) != {"stage", "seconds", "source"} or r["stage"] not in STAGES
                       or not _seconds(r["seconds"]) or not isinstance(r["source"], dict) or not r["source"]
                       for r in prior_charges)
                or store.binding.get("prior_charges") != prior_charges):
            raise ValueError("prior measured costs must be source-bound without reset")
        roots = [p.resolve() for p in output_roots]
        if (not roots or len(set(roots)) != len(roots)
                or any(not p.is_dir() or not any(p.is_relative_to(base) for base in BASES) for p in roots)
                or store.binding.get("output_roots") != [str(p) for p in roots]
                or any(a != b and (a.is_relative_to(b) or b.is_relative_to(a)) for a in roots for b in roots)):
            raise ValueError("output accounting roots must be disjoint, owned and bound")
        self.store, self.stage, self.clock = store, stage, clock
        self.started, self.closed = clock() if started_at is None else started_at, False
        if not _seconds(self.started) or self.started > clock():
            raise ValueError("attempt must include a valid monotonic launch time")
        self.rss = rss or (lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        self.disk_free = disk_free or (lambda: min(shutil.disk_usage(p).free for p in roots))
        self.bytes_used = bytes_used or (lambda: owned_bytes(roots))
        self.vram = vram or (lambda: 0)  # Caller supplies CUDA allocator sample only for its admitted backend.
        self.charged = {name: 0. for name in STAGES}
        for row in prior_charges:
            self.charged[row["stage"]] += row["seconds"]
        starts = sorted(store.path("attempts").glob("*/start.json"))
        for i, path in enumerate(starts):
            if path.parent.name != f"{i:04d}":
                raise ValueError("attempt ledger sequence is not contiguous")
            ref = store.reference(path)
            start = store.json(ref)
            if (set(start) != {"schema", "binding", "manifest", "stage", "reservation_seconds", "charged_before"}
                    or start["schema"] != "geometric-decision-attempt-v1" or start["binding"] != store.binding
                    or start["stage"] not in STAGES or not _seconds(start["reservation_seconds"])
                    or start["reservation_seconds"] <= 0 or start["charged_before"] != self.charged):
                raise ValueError("attempt provenance or cumulative prefix differs")
            store.json(start["manifest"])
            finish_path = store.path(f"attempts/{i:04d}/finish.json")
            charge = start["reservation_seconds"]
            if finish_path.exists():
                finish = store.json(store.reference(finish_path))
                if (set(finish) != {"schema", "start", "status", "seconds", "charged_after", "completion"}
                        or finish["schema"] != "geometric-decision-attempt-finish-v1" or finish["start"] != ref
                        or finish["status"] not in ("COMPLETE", "PAUSED", "FAILED", "LIMIT_REACHED")
                        or not _seconds(finish["seconds"])):
                    raise ValueError("attempt finish differs from reservation")
                charge = finish["seconds"]
                expected = {**self.charged, start["stage"]: self.charged[start["stage"]]+charge}
                if finish["charged_after"] != expected:
                    raise ValueError("attempt finish loses cumulative costs")
                overrun = (charge >= start["reservation_seconds"] or expected[start["stage"]] > STAGES[start["stage"]]
                           or sum(expected.values()) > LIMITS["total_seconds"])
                if overrun and finish["status"] != "LIMIT_REACHED":
                    raise ValueError("overrun cannot be replayed as a normal attempt finish")
                if finish["completion"] is not None and finish["status"] != "COMPLETE":
                    raise ValueError("noncomplete attempt cannot authorize a completion")
            self.charged[start["stage"]] += charge
        remaining = min(STAGES[stage]-self.charged[stage], LIMITS["total_seconds"]-sum(self.charged.values()))
        if reservation_seconds > remaining:
            raise BudgetExceeded("reservation exceeds remaining fixed stage/total budget")
        self.reservation = reservation_seconds
        self.folder = f"attempts/{len(starts):04d}"
        self.start_ref = store.publish_json(self.folder+"/start.json",
            {"schema": "geometric-decision-attempt-v1", "binding": store.binding,
             "manifest": self.manifest_ref, "stage": stage,
             "reservation_seconds": reservation_seconds, "charged_before": self.charged})
        self.last_heavy_check = -math.inf
        try:
            self.check(force_resources=True)
        except BaseException:
            self.finish("FAILED")
            raise

    def check(self, *, force_resources=False):
        if self.closed:
            raise ValueError("attempt is already closed")
        now = self.clock()
        if now-self.started >= self.reservation:
            raise BudgetExceeded("attempt duration reservation exhausted")
        if self.rss() > LIMITS["rss_bytes"] or self.vram() > LIMITS["vram_bytes"]:
            raise BudgetExceeded("RSS/VRAM sampled guard exceeded")
        # Avoid walking an increasing checkpoint tree for every optimizer step.
        if force_resources or now-self.last_heavy_check >= 5.:
            self.store.json(self.manifest_ref)
            if self.disk_free() < LIMITS["free_bytes"] or self.bytes_used() > LIMITS["new_bytes"]:
                raise BudgetExceeded("disk-space sampled guard exceeded")
            self.last_heavy_check = now

    def finish(self, status, *, completion=None):
        if self.closed or status not in ("COMPLETE", "PAUSED", "FAILED", "LIMIT_REACHED"):
            raise ValueError("invalid or duplicate attempt finish")
        if completion is not None and status != "COMPLETE":
            raise ValueError("only a completed operation can bind a completion artifact")
        limit_error = None
        if status == "COMPLETE":
            try:
                self.check(force_resources=True)
            except BudgetExceeded as exc:
                limit_error = exc
        elapsed = self.clock()-self.started
        if not _seconds(elapsed):
            raise ValueError("monotonic clock moved backward or became nonfinite")
        after = {**self.charged, self.stage: self.charged[self.stage]+elapsed}
        if (elapsed >= self.reservation or after[self.stage] > STAGES[self.stage]
                or sum(after.values()) > LIMITS["total_seconds"]):
            limit_error = BudgetExceeded("attempt exceeded its reservation or cumulative cap at finish")
        if limit_error is not None:
            status, completion = "LIMIT_REACHED", None
        ref = self.store.publish_json(self.folder+"/finish.json",
            {"schema": "geometric-decision-attempt-finish-v1", "start": self.start_ref,
             "status": status, "seconds": elapsed, "charged_after": after, "completion": completion})
        self.closed = True
        if limit_error is not None:
            raise limit_error
        return ref


def admit_projection(stage, *, measured_seconds, units_measured, remaining_units, charged_seconds):
    if (stage not in STAGES or not _seconds(measured_seconds) or measured_seconds <= 0
            or type(units_measured) is not int or units_measured <= 0
            or type(remaining_units) is not int or remaining_units < 0 or not _seconds(charged_seconds)):
        raise ValueError("finite measured projection with explicit remaining work required")
    projected = 1.25*measured_seconds/units_measured*remaining_units
    if charged_seconds+projected > STAGES[stage]:
        raise BudgetExceeded("profile plus 25 percent exceeds fixed stage budget; do not shrink the roster")
    return {"stage": stage, "measured_seconds": measured_seconds, "units_measured": units_measured,
            "remaining_units": remaining_units, "margin": 1.25, "charged_seconds": charged_seconds,
            "projected_remaining_seconds": projected}
