"""Concrete phase ordering, frozen detector and global prediction seal.

Phase payload builders still must validate their scientific contents. This gate
authenticates their exact references and full unit identities, not mere counts.
The source verifier is a required runner service that checks the frozen code and
reuse inventory against current bytes; there is no default-open implementation.
"""
from __future__ import annotations

import numpy as np

from .measurement_contract import unit_roster, calibrate_detector, digest
from .partial_compatibility_cache import encoded


class MeasurementAdmission:
    def __init__(self, controller, *, verify_sources):
        if not callable(verify_sources):
            raise ValueError("source verification service required")
        self.controller, self.store, self.verify_sources = controller, controller.store, verify_sources

    def phase_result(self, phase, *, identity=None):
        self.store._require_lock()
        for i, (start, start_ref) in reversed(list(enumerate(self.controller._starts()))):
            if start["phase"] != phase:
                continue
            end = self.controller._end(i, start_ref)
            if end is None or end["status"] != "COMPLETE":
                continue
            if identity is not None and encoded(start["identity"]) != encoded(identity):
                raise ValueError("completed prerequisite source identity differs")
            ref = end["payload"]
            receipt = self.store.json(ref)
            if (receipt.get("schema") != "measurement-stage-complete-v1"
                    or encoded(receipt.get("binding")) != encoded(self.store.binding)
                    or encoded(receipt.get("identity")) != encoded(start["identity"])):
                raise ValueError("phase completion receipt identity differs")
            result, arrays = self.store._payload(receipt["payload"], start["identity"])
            return ref, result, arrays
        raise PermissionError(f"required phase not complete: {phase}")

    def _fixed(self, name):
        ref = self.store.reference(name)
        return ref, self.store.json(ref)

    def _freeze_artifacts(self):
        """A staged freeze is immutable even if its final cost receipt was lost."""
        return [start for i, (start, _) in enumerate(self.controller._starts())
                if start["phase"] == "freeze" and self.store.path(
                    f"control/attempts/{i:05d}/artifact/payload.npz").exists()]

    def freeze(self, *, source_snapshot, exclusions, projected_work_seconds, projected_replay_seconds,
               projected_gpu_seconds, reservation, resources):
        """The complete phase artifact is the canonical freeze, never a side file."""
        identity = {"phase": "freeze", "source_snapshot": source_snapshot, "exclusions": exclusions,
                    "projected_work_seconds": projected_work_seconds,
                    "projected_replay_seconds": projected_replay_seconds,
                    "projected_gpu_seconds": projected_gpu_seconds}
        def produce(check):
            result = self._build_freeze(**{k: v for k, v in identity.items() if k != "phase"})
            check()
            return result, {}
        def validate(result, arrays):
            if arrays or any(encoded(result.get(k)) != encoded(v) for k, v in identity.items() if k != "phase"):
                raise ValueError("freeze result differs from its declared inputs")
        return self.controller.run("freeze", identity, reservation=reservation, gpu=False,
            admit=self.admit, produce=produce, validate=validate, resources=resources)[0]

    def _build_freeze(self, *, source_snapshot, exclusions, projected_work_seconds,
                      projected_replay_seconds, projected_gpu_seconds):
        self.store._require_lock()
        self.verify_sources(source_snapshot)
        profiles = {p: self.phase_result(p, identity={"phase": p, "source_snapshot": source_snapshot})[0]
                    for p in ("profile_cpu", "profile_gpu")}
        calibration_ref, calibration, _ = self.phase_result("calibration", identity={
            "phase": "calibration", "source_snapshot": source_snapshot})
        replayed = calibrate_detector(np.asarray(calibration["costs"], np.float64), calibration["units"])
        if encoded(replayed) != encoded(calibration):
            raise ValueError("calibration does not replay the declared detector selection")
        exclusion = self.store.json(exclusions)
        if (set(exclusion) != {"schema", "fingerprints", "sources"}
                or exclusion["schema"] != "measurement-exclusions-v1"
                or not isinstance(exclusion["fingerprints"], list)
                or exclusion["fingerprints"] != sorted(set(exclusion["fingerprints"]))
                or any(not isinstance(x, str) or len(x) != 64 or any(c not in "0123456789abcdef" for c in x)
                       for x in exclusion["fingerprints"])
                or not isinstance(exclusion["sources"], list) or not exclusion["sources"]):
            raise ValueError("invalid predeclared prior-observation exclusion inventory")
        for ref in exclusion["sources"]:
            self.store.read(ref)
        estimates = (projected_work_seconds, projected_replay_seconds)
        if any(type(x) not in (int, float) or not np.isfinite(x) or x <= 0 for x in estimates):
            raise ValueError("profile projection requires finite positive work and replay costs")
        if (type(projected_gpu_seconds) not in (int, float) or not np.isfinite(projected_gpu_seconds)
                or projected_gpu_seconds < 0 or projected_gpu_seconds > sum(estimates)):
            raise ValueError("invalid projected GPU occupancy")
        costs, limits = self.controller.costs(reserve_active=True), self.controller.limits
        work, replay = 1.5*projected_work_seconds, 1.5*projected_replay_seconds
        if (costs["total_seconds"]+work+max(replay, limits["reserve_seconds"]) > limits["total_seconds"]
                or costs["work_seconds"]+work > limits["total_seconds"]-limits["reserve_seconds"]
                or costs["gpu_seconds"]+1.5*projected_gpu_seconds > limits["gpu_seconds"]):
            raise RuntimeError("profiled projection plus audit reserve does not fit")
        freeze = {"schema": "measurement-freeze-v1", "binding": self.store.binding,
            "source_snapshot": source_snapshot, "profiles": profiles, "calibration": calibration_ref,
            "detector": {k: calibration[k] for k in ("height", "prominence", "calibration_sha256")},
            "exclusions": exclusions, "units": unit_roster("test"), "limits": limits,
            "projected_work_seconds": projected_work_seconds,
            "projected_replay_seconds": projected_replay_seconds, "margin": 1.5,
            "projected_gpu_seconds": projected_gpu_seconds,
            "costs_with_freeze_reservation": costs}
        return freeze

    def verify_freeze(self):
        ref, freeze, _ = self.phase_result("freeze")
        if (freeze.get("schema") != "measurement-freeze-v1"
                or encoded(freeze.get("binding")) != encoded(self.store.binding)
                or encoded(freeze.get("limits")) != encoded(self.controller.limits)
                or digest(freeze.get("units")) != digest(unit_roster("test"))):
            raise ValueError("frozen binding, limits or test roster differs")
        self.verify_sources(freeze["source_snapshot"])
        self.store.read(freeze["exclusions"])
        for phase, expected in freeze["profiles"].items():
            if encoded(self.phase_result(phase, identity={"phase": phase,
                    "source_snapshot": freeze["source_snapshot"]})[0]) != encoded(expected):
                raise ValueError("profile changed after freeze")
        if encoded(self.phase_result("calibration", identity={"phase": "calibration",
                "source_snapshot": freeze["source_snapshot"]})[0]) != encoded(freeze["calibration"]):
            raise ValueError("calibration changed after freeze")
        return ref, freeze

    def _unit_index(self, phase):
        ref, result, _ = self.phase_result(phase)
        if (set(result) != {"schema", "freeze", "units", "records"}
                or result["schema"] != f"measurement-{phase}-index-v1"
                or encoded(result["freeze"]) != encoded(self.verify_freeze()[0])
                or digest(result["units"]) != digest(unit_roster("test"))
                or not isinstance(result["records"], list) or len(result["records"]) != 2048):
            raise ValueError("phase index does not cover the exact paired unit roster")
        # Unit receipts, not heavy arrays, are authenticated here. The runner's
        # per-unit validator authenticates all referenced scientific payloads.
        for unit, record_ref in zip(result["units"], result["records"]):
            record = self.store.json(record_ref)
            if (set(record) != {"schema", "unit", "artifact"}
                    or record["schema"] != f"measurement-{phase}-unit-v1"
                    or digest(record.get("unit")) != digest(unit)):
                raise ValueError("unit receipt differs from its paired identity")
            artifact = self.store.json(record["artifact"])
            expected_identity = {"phase": phase, "unit": unit, "freeze": result["freeze"]}
            if (artifact.get("schema") != "measurement-stage-complete-v1"
                    or encoded(artifact.get("binding")) != encoded(self.store.binding)
                    or digest(artifact.get("identity")) != digest(expected_identity)):
                raise ValueError("unit has no matching complete artifact")
            self.store._payload(artifact["payload"], expected_identity)
        return ref

    def seal_predictions(self, freeze, *, reservation, resources):
        identity = {"phase": "seal", "freeze": freeze}
        def produce(check):
            observed = self._unit_index("test_observation")
            check()
            predicted = self._unit_index("test_prediction")
            check()
            return {"schema": "measurement-prediction-seal-v1", "freeze": freeze,
                    "observations": observed, "predictions": predicted,
                    "units_sha256": digest(unit_roster("test"))}, {}
        def validate(result, arrays):
            if arrays or result.get("schema") != "measurement-prediction-seal-v1" or encoded(result.get("freeze")) != encoded(freeze):
                raise ValueError("invalid prediction seal")
        return self.controller.run("seal", identity, reservation=reservation, gpu=False,
            admit=self.admit, produce=produce, validate=validate, resources=resources)[0]

    def admit(self, phase, identity):
        self.store._require_lock()
        starts = self.controller._starts()
        if phase == "freeze":
            for start, _ in starts:
                if start["phase"] not in ("profile_cpu", "profile_gpu", "calibration", "freeze"):
                    raise PermissionError("cannot freeze or revise after test execution began")
            for start in self._freeze_artifacts():
                if encoded(start["identity"]) != encoded(identity):
                    raise PermissionError("freeze identity cannot change once attempted")
            self.verify_sources(identity["source_snapshot"])
            return
        if phase in ("profile_cpu", "profile_gpu", "calibration"):
            if set(identity) != {"phase", "source_snapshot"} or identity["phase"] != phase:
                raise ValueError("open phase identity differs")
            if self._freeze_artifacts():
                raise PermissionError("open development/calibration is closed after freeze")
            self.verify_sources(identity["source_snapshot"])
            if phase != "profile_cpu":
                self.phase_result("profile_cpu", identity={"phase": "profile_cpu",
                    "source_snapshot": identity["source_snapshot"]})
            return
        freeze_ref, _ = self.verify_freeze()
        if (set(identity) != {"phase", "freeze"} or identity["phase"] != phase
                or encoded(identity.get("freeze")) != encoded(freeze_ref)):
            raise PermissionError("phase does not bind the frozen test contract")
        if phase == "test_observation":
            return
        if phase == "test_prediction":
            self._unit_index("test_observation")
            return
        if phase == "seal":
            return  # Both complete indices are authenticated by the measured producer.
        if phase not in ("evaluation", "replay", "audit"):
            raise ValueError("unknown measurement phase")
        _, seal, _ = self.phase_result("seal", identity={"phase": "seal", "freeze": freeze_ref})
        expected = {"schema": "measurement-prediction-seal-v1", "freeze": freeze_ref,
            "observations": self._unit_index("test_observation"),
            "predictions": self._unit_index("test_prediction"),
            "units_sha256": digest(unit_roster("test"))}
        if encoded(seal) != encoded(expected):
            raise ValueError("global prediction seal differs from the completed paired indices")
        if phase in ("replay", "audit"):
            self.phase_result("evaluation")
