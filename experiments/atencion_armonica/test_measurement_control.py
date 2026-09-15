import numpy as np
import pytest

from src.atencion_armonica.measurement_control import PhaseController
from src.atencion_armonica.measurement_store import MeasurementStore


class Clock:
    def __init__(self):
        self.value = 0.
    def __call__(self):
        return self.value
    def advance(self, n):
        self.value += n


def setup(tmp_path):
    store = MeasurementStore(tmp_path/"store", binding={"fixture": 1})
    clock = Clock()
    control = PhaseController(store, limits={"total_seconds": 100., "gpu_seconds": 60., "reserve_seconds": 25.},
                              clock=clock, boot=lambda: "fixture-boot")
    return store, clock, control


def run(control, produce, *, phase="calibration", reservation=10., gpu=False, validate=lambda r, a: None):
    return control.run(phase, {"unit": 1}, reservation=reservation, gpu=gpu,
                       admit=lambda p, i: None, produce=produce, validate=validate, resources=lambda: None)


def test_costs_and_replay_are_accumulated_without_recompute(tmp_path):
    store, clock, control = setup(tmp_path)
    calls = []
    def produce(check):
        calls.append(1)
        clock.advance(4)
        check()
        return {"ok": True}, {"data": np.arange(2)}
    with store.exclusive():
        run(control, produce, gpu=True, validate=lambda r, a: clock.advance(1))
        run(control, produce, gpu=False, validate=lambda r, a: clock.advance(2))
        assert len(calls) == 1
        assert control.costs() == {"total_seconds": 7., "gpu_seconds": 5., "work_seconds": 7.}


def test_failure_charged_and_explicit_new_attempt_does_not_reset(tmp_path):
    store, clock, control = setup(tmp_path)
    def fail(check):
        clock.advance(3)
        raise ValueError("fixture failure")
    with store.exclusive():
        with pytest.raises(ValueError):
            run(control, fail)
        assert control.costs()["total_seconds"] == 3
        run(control, lambda check: ({}, {"x": np.zeros(1)}))
        assert control.costs()["total_seconds"] == 3
        assert len(control._starts()) == 2


def test_reserve_blocks_work_but_allows_audit(tmp_path):
    store, clock, control = setup(tmp_path)
    def produce(check):
        clock.advance(70)
        return {}, {}
    with store.exclusive():
        run(control, produce, reservation=75)
        with pytest.raises(RuntimeError, match="budget/reserve"):
            run(control, lambda check: ({}, {}), reservation=10)
        run(control, lambda check: ({}, {}), phase="audit", reservation=10)


def test_gpu_budget_and_deadline_are_real_preconditions(tmp_path):
    store, clock, control = setup(tmp_path)
    with store.exclusive():
        with pytest.raises(RuntimeError, match="budget"):
            run(control, lambda check: ({}, {}), gpu=True, reservation=61)
        assert not control._starts()
        def overrun(check):
            clock.advance(11)
            check()
            raise AssertionError("deadline not enforced")
        with pytest.raises(TimeoutError):
            run(control, overrun)
        assert control.costs()["total_seconds"] == 11


def test_missing_finish_has_same_boot_upper_bound(tmp_path, monkeypatch):
    store, clock, control = setup(tmp_path)
    original = store.publish_json
    def interrupt(name, value):
        if name.endswith("finish.json"):
            raise InterruptedError("missing cost receipt")
        return original(name, value)
    with store.exclusive():
        monkeypatch.setattr(store, "publish_json", interrupt)
        with pytest.raises(InterruptedError):
            run(control, lambda check: ({"once": True}, {"x": np.arange(2)}))
        monkeypatch.setattr(store, "publish_json", original)
        clock.advance(8)
        assert control.reconcile()["total_seconds"] == 8
        result = run(control, lambda check: (_ for _ in ()).throw(AssertionError("must not repeat")))
        assert result[1] == {"once": True}
        assert control.costs()["total_seconds"] == 8


def test_cross_boot_unknown_cost_is_not_reset(tmp_path, monkeypatch):
    store, clock, control = setup(tmp_path)
    original = store.publish_json
    def interrupt(name, value):
        if name.endswith("finish.json"):
            raise InterruptedError()
        return original(name, value)
    with store.exclusive():
        monkeypatch.setattr(store, "publish_json", interrupt)
        with pytest.raises(InterruptedError):
            run(control, lambda check: ({}, {}))
        monkeypatch.setattr(store, "publish_json", original)
        control.boot = lambda: "different-boot"
        with pytest.raises(RuntimeError, match="cross-boot"):
            control.reconcile()


def test_first_resource_check_is_charged_and_deadline_checks_its_cost(tmp_path):
    store, clock, control = setup(tmp_path)
    calls = []
    def resources():
        calls.append(1)
        clock.advance(2)
    with store.exclusive():
        control.run("calibration", {"unit": 1}, reservation=20, gpu=False,
                    admit=lambda p, i: None, produce=lambda check: ({}, {}),
                    validate=lambda r, a: None, resources=resources)
        assert control.costs()["total_seconds"] == len(calls)*2 == clock.value
        with pytest.raises(TimeoutError, match="resource verification"):
            control.run("calibration", {"unit": 2}, reservation=1, gpu=False,
                        admit=lambda p, i: None,
                        produce=lambda check: (_ for _ in ()).throw(AssertionError("producer reached")),
                        validate=lambda r, a: None, resources=resources)
        assert control.costs()["total_seconds"] == clock.value


def test_final_reload_overrun_cannot_close_complete(tmp_path, monkeypatch):
    store, clock, control = setup(tmp_path)
    original = store.completed
    calls = []
    def delayed(*args, **kwargs):
        calls.append(1)
        result = original(*args, **kwargs)
        # publish_stage calls completed twice; this is the final reload.
        if len(calls) == 3:
            clock.advance(6)
        return result
    with store.exclusive():
        monkeypatch.setattr(store, "completed", delayed)
        with pytest.raises(TimeoutError):
            run(control, lambda check: ({}, {}), reservation=5)
        assert control._end(0, control._starts()[0][1])["status"] == "FAILED"
        assert control.costs()["total_seconds"] == 6


def test_external_duration_recovery_binds_attempt_and_sources(tmp_path, monkeypatch):
    store, clock, control = setup(tmp_path)
    original = store.publish_json
    def interrupt(name, value):
        if name.endswith("finish.json"):
            raise InterruptedError()
        return original(name, value)
    with store.exclusive():
        monkeypatch.setattr(store, "publish_json", interrupt)
        with pytest.raises(InterruptedError):
            run(control, lambda check: ({"once": True}, {}))
        monkeypatch.setattr(store, "publish_json", original)
        control.boot = lambda: "different-boot"
        start = control._starts()[0][1]
        source = store.publish_json("fixture-time-log.json", {"elapsed": 7., "fixture": True})
        evidence = {"schema": "measurement-duration-evidence-v1", "start": start, "seconds": 7.,
                    "kind": "upper_bound", "provenance": "manual clock fixture, not real campaign",
                    "sources": [source]}
        bad = store.publish_json("bad-evidence.json", {**evidence, "start": source})
        with pytest.raises(ValueError, match="duration evidence"):
            control.reconcile_external(start, bad)
        ref = store.publish_json("duration-evidence.json", evidence)
        assert control.reconcile_external(start, ref)["total_seconds"] == 7
        result = run(control, lambda check: (_ for _ in ()).throw(AssertionError("recomputed")))
        assert result[1] == {"once": True}
        assert control.costs()["total_seconds"] == 7
