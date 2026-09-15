"""Accumulated-cost phase control and explicit interrupted-attempt recovery.

This is not a generic task service: one synchronous producer runs under the
measurement store's process lock. Phase-specific roster/seal validation belongs
to the concrete runner. No sources, scenes, models or CUDA are opened here.
"""
from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
import time

from .partial_compatibility_cache import encoded

PHASES = ("profile_cpu", "profile_gpu", "calibration", "freeze", "test_observation",
          "test_prediction", "seal", "evaluation", "replay", "audit")
AUDIT_PHASES = frozenset(("replay", "audit"))
LIMITS = {"total_seconds": 28800., "gpu_seconds": 21600., "reserve_seconds": 7200.}


def finite_nonnegative(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def host_boot_id():
    return Path("/proc/sys/kernel/random/boot_id").read_text().strip()


class PhaseController:
    """The caller owns store.exclusive() throughout each synchronous phase.

Admission/validation callbacks are required, not default-open. They must check
the pinned source snapshot, stage prerequisite artifacts and complete roster.
"""
    def __init__(self, store, *, limits, clock=time.monotonic, boot=host_boot_id):
        if (not isinstance(limits, dict) or set(limits) != set(LIMITS)
                or any(not finite_nonnegative(v) for v in limits.values())
                or limits["reserve_seconds"] > limits["total_seconds"]
                or limits["gpu_seconds"] > limits["total_seconds"]):
            raise ValueError("invalid accumulated resource budget")
        self.store, self._limits, self.clock, self.boot = store, encoded(limits), clock, boot
        self._active = None

    @property
    def limits(self):
        import json
        return json.loads(self._limits)

    def _starts(self):
        self.store._require_lock()
        base = self.store.path("control/attempts")
        directories = sorted(base.iterdir()) if base.exists() else []
        if [p.name for p in directories] != [f"{i:05d}" for i in range(len(directories))]:
            raise ValueError("attempt ledger is not contiguous")
        result = []
        for i, directory in enumerate(directories):
            name = f"control/attempts/{i:05d}/start.json"
            ref = self.store.reference(name)
            row = self.store.json(ref)
            if (set(row) != {"schema", "binding", "limits", "phase", "identity", "reservation", "gpu", "boot", "started", "previous"}
                    or row["schema"] != "measurement-control-start-v1"
                    or encoded(row["binding"]) != encoded(self.store.binding)
                    or encoded(row["limits"]) != self._limits or row["phase"] not in PHASES
                    or type(row["gpu"]) is not bool or not finite_nonnegative(row["reservation"])
                    or not finite_nonnegative(row["started"]) or not isinstance(row["boot"], str)
                    or not isinstance(row["identity"], dict) or not row["identity"]
                    or encoded(row["previous"]) != encoded(None if not result else result[-1][1])):
                raise ValueError("attempt identity, limits or chain differs")
            result.append((row, ref))
        return result

    def _end(self, i, start_ref):
        name = f"control/attempts/{i:05d}/finish.json"
        if not self.store.path(name).exists():
            name = f"control/attempts/{i:05d}/reconciled.json"
            if not self.store.path(name).exists():
                return None
        ref = self.store.reference(name)
        row = self.store.json(ref)
        fields = {"schema", "start", "status", "seconds", "cost_kind", "error", "payload"}
        external = row.get("cost_kind") == "external_evidence"
        if (set(row) != (fields | {"evidence"} if external else fields)
                or row["schema"] != "measurement-control-finish-v1"
                or encoded(row["start"]) != encoded(start_ref)
                or row["status"] not in ("COMPLETE", "FAILED", "INTERRUPTED")
                or row["cost_kind"] not in ("measured", "elapsed_upper_bound", "external_evidence")
                or not finite_nonnegative(row["seconds"])):
            raise ValueError("invalid attempt finish or cost")
        if external:
            evidence = self._duration_evidence(row["evidence"], start_ref)
            if row["status"] != "INTERRUPTED" or row["seconds"] != evidence["seconds"]:
                raise ValueError("reconciled cost differs from duration evidence")
        return row

    def costs(self, *, reserve_active=False):
        totals = {"total_seconds": 0., "gpu_seconds": 0., "work_seconds": 0.}
        for i, (start, ref) in enumerate(self._starts()):
            end = self._end(i, ref)
            if end is None:
                if not reserve_active or self._active != ref:
                    raise RuntimeError("unreconciled interrupted attempt; budget cannot reset")
                seconds = max(start["reservation"], self.clock()-start["started"])
            else:
                seconds = end["seconds"]
            totals["total_seconds"] += seconds
            totals["gpu_seconds"] += seconds if start["gpu"] else 0.
            totals["work_seconds"] += seconds if start["phase"] not in AUDIT_PHASES else 0.
        return totals

    def _duration_evidence(self, evidence_ref, start_ref):
        evidence = self.store.json(evidence_ref)
        if (set(evidence) != {"schema", "start", "seconds", "kind", "provenance", "sources"}
                or evidence["schema"] != "measurement-duration-evidence-v1"
                or encoded(evidence["start"]) != encoded(start_ref)
                or not finite_nonnegative(evidence["seconds"])
                or evidence["kind"] not in ("measured", "upper_bound")
                or not isinstance(evidence["provenance"], str) or not evidence["provenance"].strip()
                or not isinstance(evidence["sources"], list) or not evidence["sources"]):
            raise ValueError("invalid external duration evidence")
        for source in evidence["sources"]:
            self.store.read(source)
        return evidence

    def reconcile_external(self, start_ref, evidence_ref):
        """Caller must first establish the duration from independently observed logs.

This authenticates provenance, not the scientific truth of a caller's duration.
Never substitute the reservation for missing elapsed-time evidence.
"""
        starts = self._starts()
        if self._active is not None or not starts or encoded(starts[-1][1]) != encoded(start_ref):
            raise ValueError("external recovery requires the final inactive attempt")
        i, (start, _) = len(starts)-1, starts[-1]
        if self._end(i, start_ref) is not None or start["boot"] == self.boot():
            raise ValueError("external recovery requires an unclosed cross-boot attempt")
        evidence = self._duration_evidence(evidence_ref, start_ref)
        self.store.publish_json(f"control/attempts/{i:05d}/reconciled.json", {
            "schema": "measurement-control-finish-v1", "start": start_ref,
            "status": "INTERRUPTED", "seconds": evidence["seconds"],
            "cost_kind": "external_evidence", "error": "cross-boot duration evidence",
            "payload": None, "evidence": evidence_ref})
        return self.costs()

    def reconcile(self):
        """Called only after lock acquisition proves no synchronous producer owns it.

Same-boot time until now bounds the missing elapsed duration including idle.
Cross-boot unknown cost stays blocked, not reset or guessed from reservation.
"""
        starts = self._starts()
        for i, (start, ref) in enumerate(starts):
            if self._end(i, ref) is not None:
                continue
            if i != len(starts)-1:
                raise ValueError("unclosed attempt before a later attempt")
            if start["boot"] != self.boot():
                raise RuntimeError("cross-boot interrupted cost requires external duration evidence")
            duration = self.clock()-start["started"]
            if not finite_nonnegative(duration):
                raise ValueError("monotonic recovery clock moved backwards")
            self.store.publish_json(f"control/attempts/{i:05d}/reconciled.json", {
                "schema": "measurement-control-finish-v1", "start": ref,
                "status": "INTERRUPTED", "seconds": duration,
                "cost_kind": "elapsed_upper_bound", "error": "missing finish; same-boot elapsed bound",
                "payload": None})
        return self.costs()

    def run(self, phase, identity, *, reservation, gpu, admit, produce, validate, resources):
        if (phase not in PHASES or not isinstance(identity, dict) or not identity
                or not finite_nonnegative(reservation) or reservation <= 0 or type(gpu) is not bool
                or not all(callable(f) for f in (admit, produce, validate, resources))):
            raise ValueError("phase requires explicit identity, reservation and validation callbacks")
        self.store._require_lock()
        if self._active is not None:
            raise RuntimeError("nested measured phases are not allowed")
        identity = deepcopy(identity)
        costs = self.reconcile()
        starts = self._starts()
        recover_folder = None
        for i, (start, ref) in enumerate(starts):
            if start["phase"] == phase and encoded(start["identity"]) == encoded(identity):
                # A complete payload outlives FAILED/INTERRUPTED accounting.
                # Recovery never calls produce; semantic validation still runs.
                candidate = f"control/attempts/{i:05d}/artifact"
                if self.store.path(candidate+"/payload.npz").exists():
                    recover_folder = candidate
                    break
        limits = self.limits
        if (costs["total_seconds"]+reservation > limits["total_seconds"]
                or (gpu and costs["gpu_seconds"]+reservation > limits["gpu_seconds"])
                or (phase not in AUDIT_PHASES and costs["work_seconds"]+reservation >
                    limits["total_seconds"]-limits["reserve_seconds"])):
            raise RuntimeError("accumulated budget/reserve exhausted before phase")
        i = len(starts)
        folder = f"control/attempts/{i:05d}"
        started, boot = self.clock(), self.boot()
        start_ref = self.store.publish_json(folder+"/start.json", {
            "schema": "measurement-control-start-v1", "binding": self.store.binding,
            "limits": limits, "phase": phase, "identity": identity,
            "reservation": reservation, "gpu": gpu, "boot": boot, "started": started,
            "previous": None if not starts else starts[-1][1]})
        self._active = start_ref
        status, error, payload = "FAILED", None, None
        def check():
            if self.boot() != boot or self.clock()-started >= reservation:
                raise TimeoutError("phase time reservation exhausted")
            resources()
            if self.boot() != boot or self.clock()-started >= reservation:
                raise TimeoutError("phase time reservation exhausted during resource verification")
        try:
            check()
            admit(phase, deepcopy(identity))
            check()
            if recover_folder is None:
                result, arrays = produce(check)
            else:
                restored = self.store.completed(recover_folder, identity)
                payload, result, arrays = restored
            check()
            validate(result, arrays)
            if recover_folder is None:
                payload = self.store.publish_stage(folder+"/artifact", identity, result, arrays)
            restored = (self.store.completed(folder+"/artifact", identity)
                        if recover_folder is None else restored)
            check()
            status = "COMPLETE"
            return restored
        except BaseException as exception:
            error = f"{type(exception).__name__}: {exception}"
            raise
        finally:
            self._active = None
            # The tiny finish receipt is administrative overhead; all material
            # preflight, source I/O, validation and payload reload precede this.
            self.store.publish_json(folder+"/finish.json", {
                "schema": "measurement-control-finish-v1", "start": start_ref,
                "status": status, "seconds": self.clock()-started,
                "cost_kind": "measured", "error": error, "payload": payload})
