"""Cumulative operator budget and conservative fixed-roster forecast.

No GPU or scheduler hooks. Failed and interrupted attempts remain charged;
an attempt without a finish receipt consumes its full reserved duration.
"""
from __future__ import annotations

import math
import resource
import shutil
import time

from .operator_objective_artifacts import encoded

LIMITS = {"seconds": 1800., "profile_seconds": 180., "audit_reserve_seconds": 600.,
          "rss_bytes": 6*1024**3, "new_bytes": 4*1024**3, "free_bytes": 80*1024**3}


class BudgetExceeded(RuntimeError):
    pass


def forecast(profile, inventory, *, charged_seconds):
    """Frozen four-profile/max-cost/82-candidate forecast, with margin x2."""
    if len(profile) != 4 or {r["split"] for r in profile} != {"iid", "ood_beta", "ood_polyphony", "deformed_family"}:
        raise ValueError("forecast requires the four fixed scene-zero profiles")
    for row in profile:
        if (row["scene_id"] != 0 or type(row["candidate_count"]) is not int
                or not 2 <= row["candidate_count"] <= 82
                or any(type(row[k]) is not int or row[k] <= 0 for k in ("decoded_bytes", "bundle_bytes"))
                or any(type(row.get(k, 0.)) not in (float, int) or not math.isfinite(row.get(k, 0.))
                       or row.get(k, 0.) < 0 for k in ("extraction_seconds", "diagnostic_seconds", "setup_seconds"))):
            raise ValueError("profile lacks finite costs or sufficient candidate pairs")
    if inventory["scene_count"] != 2048 or not math.isfinite(charged_seconds) or charged_seconds < 0:
        raise ValueError("forecast requires the complete fixed inventory and charged time")
    extraction_per_byte = max(r["extraction_seconds"]/r["decoded_bytes"] for r in profile)
    diagnostic_per_pair_cell_scheme = max(r["diagnostic_seconds"]/(math.comb(r["candidate_count"], 2)*27*4)
                                          for r in profile)
    extraction = extraction_per_byte*inventory["decoded_bytes"]
    diagnostic = diagnostic_per_pair_cell_scheme*math.comb(82, 2)*27*4*2048
    # Replay recomputes the diagnosis; original extraction is reused from compact bundles.
    # IO/codec time is included in the measured diagnostic leg of each profile.
    setup = 2*sum(r.get("setup_seconds", 0.) for r in profile)
    remaining = 2*(extraction+2*diagnostic+setup)+LIMITS["audit_reserve_seconds"]
    total = charged_seconds+remaining
    bytes_per_candidate = max(r["bundle_bytes"]/r["candidate_count"] for r in profile)
    projected_bytes = 2*bytes_per_candidate*82*2048
    other_outputs_reserve = 64*1024**2
    return {"extraction_seconds_per_decoded_byte": extraction_per_byte,
            "diagnostic_seconds_per_pair_cell_scheme": diagnostic_per_pair_cell_scheme,
            "projected_extraction_seconds": extraction, "projected_diagnostic_seconds": diagnostic,
            "projected_remaining_setup_seconds": setup,
            "replay_diagnostic_factor": 2, "remaining_safety_factor": 2,
            "audit_reserve_seconds": LIMITS["audit_reserve_seconds"], "charged_seconds": charged_seconds,
            "projected_total_seconds": total, "projected_scene_bundle_bytes": projected_bytes,
            "other_outputs_reserve_bytes": other_outputs_reserve,
            "projected_new_bytes": projected_bytes+other_outputs_reserve,
            "time_fits": total <= LIMITS["seconds"],
            "outputs_fit": projected_bytes+other_outputs_reserve <= LIMITS["new_bytes"]}


def _own_bytes(root):
    total = 0
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("unexpected symlink in owned diagnostic output")
        if path.is_file():
            total += path.stat().st_size
    return total


class AttemptBudget:
    """One non-concurrent operator attempt with immutable start/finish receipts.

    The runner must hold a process lock before constructing this object and
    release it only after finish. Receipt existence never proves a live process.
    """

    def __init__(self, store, operation, *, limits=None, clock=time.monotonic):
        if operation not in ("prepare", "profile", "run", "replay", "audit"):
            raise ValueError("unknown diagnostic operation")
        self.limits = dict(LIMITS if limits is None else limits)
        if set(self.limits) != set(LIMITS) or any(not math.isfinite(v) or v <= 0 for v in self.limits.values()):
            raise ValueError("invalid operational budget")
        self.store, self.operation, self.clock = store, operation, clock
        manifest, self.manifest_ref = store.manifest()
        if manifest.get("limits") != self.limits:
            raise ValueError("attempt limits differ from the fixed manifest budget")
        self.started = clock()
        self.charged_before = 0.
        self.profile_before = 0.
        attempts = store.root/"attempts"
        existing = sorted(attempts.glob("*/start.json")) if attempts.exists() else []
        for i, path in enumerate(existing):
            if path.parent.name != f"{i:04d}":
                raise ValueError("attempt roster is not a contiguous immutable sequence")
            start = self._local_json(path)
            if start["manifest"] != self.manifest_ref:
                raise ValueError("attempt belongs to another diagnostic manifest")
            finish = path.parent/"finish.json"
            if finish.exists():
                end = self._local_json(finish)
                if (end["start"] != start or not math.isfinite(end["seconds"])
                        or end["seconds"] < 0):
                    raise ValueError("attempt finish differs from its reservation")
                if end["status"] == "BUDGET_EXHAUSTED":
                    raise BudgetExceeded("terminal exhausted budget requires explicit redesign")
                charged = end["seconds"]
            else:
                charged = start["allocated_seconds"]  # No unobserved-crash time refund.
            self.charged_before += charged
            if start["operation"] == "profile":
                self.profile_before += charged
        reserve = 0 if operation == "audit" else self.limits["audit_reserve_seconds"]
        available = self.limits["seconds"]-self.charged_before-reserve
        if operation == "profile":
            available = min(available, self.limits["profile_seconds"]-self.profile_before)
        if available <= 0:
            raise BudgetExceeded("cumulative operator budget exhausted before launch")
        self.allocation = available
        self.relative = f"attempts/{len(existing):04d}"
        self.start_record = {"manifest": self.manifest_ref, "operation": operation,
                             "charged_before": self.charged_before, "allocated_seconds": available}
        self.closed = False
        store.publish_json(self.relative+"/start.json", self.start_record)
        try:
            self.check()
        except BudgetExceeded:
            self.finish("BUDGET_EXHAUSTED")
            raise
        except Exception:
            self.finish("FAILED")
            raise

    @staticmethod
    def _local_json(path):
        import json
        if path.is_symlink():
            raise ValueError("attempt receipts cannot be symlinks")
        raw = path.read_bytes()
        value = json.loads(raw)
        if encoded(value) != raw:
            raise ValueError("attempt receipt is not canonical")
        return value

    def check(self, *, additional_bytes=0):
        if self.closed:
            raise ValueError("attempt is already terminal")
        _, ref = self.store.manifest()
        if ref != self.manifest_ref:
            raise ValueError("diagnostic manifest changed during operation")
        if self.clock()-self.started >= self.allocation:
            raise BudgetExceeded("cumulative diagnostic time budget exceeded")
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 > self.limits["rss_bytes"]:
            raise BudgetExceeded("diagnostic RSS envelope exceeded")
        if shutil.disk_usage(self.store.root).free < self.limits["free_bytes"]:
            raise BudgetExceeded("diagnostic free-space guard reached")
        if type(additional_bytes) is not int or additional_bytes < 0:
            raise ValueError("additional output bytes must be a nonnegative integer")
        staging = self.store.project/".agent-work/phideus-operator-objective-20260914"
        roots = [staging] if self.store.root.is_relative_to(staging) else [self.store.root, staging]
        if sum(_own_bytes(root) for root in roots if root.exists())+additional_bytes > self.limits["new_bytes"]:
            raise BudgetExceeded("diagnostic new-output envelope exceeded")

    def finish(self, status, *, completion=None):
        if self.closed or status not in ("COMPLETE", "FAILED", "PAUSED", "BUDGET_EXHAUSTED"):
            raise ValueError("invalid or duplicate attempt terminal state")
        elapsed = self.clock()-self.started
        record = {"start": self.start_record, "status": status,
                  "seconds": max(elapsed, 0.),
                  "observed_seconds": elapsed, "charged_total": self.charged_before+max(elapsed, 0.),
                  "rss_peak_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
        if completion is not None:
            if status != "COMPLETE":
                raise ValueError("only a successful attempt can seal a completion")
            record["completion"] = completion
        self.store.publish_json(self.relative+"/finish.json", record)
        self.closed = True
        return record
