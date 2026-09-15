import numpy as np
import pytest

from src.atencion_armonica.measurement_stage import run_stage
from src.atencion_armonica.measurement_store import MeasurementStore


def test_admission_precedes_producer_and_recovery_does_not_repeat(tmp_path):
    store = MeasurementStore(tmp_path/"store", binding={"fixture": "stage"})
    calls = []
    def authorize(identity):
        calls.append("admission")
        assert identity == {"operation": "fixture"}
    def produce():
        calls.append("produce")
        return {"status": "OK"}, {"x": np.arange(3)}
    with store.exclusive():
        for _ in range(2):
            result = run_stage(store, "stage", {"operation": "fixture"},
                               authorize=authorize, produce=produce, check=lambda: calls.append("check"))
        assert calls[0] == "admission" and calls[1] == "check"
        assert calls.count("produce") == 1
        assert result[1] == {"status": "OK"}
        finish = store.json(store.reference("stage/attempt-finish.json"))
        assert finish["status"] == "COMPLETE" and finish["seconds"] >= 0


def test_admission_failure_never_starts_attempt(tmp_path):
    store = MeasurementStore(tmp_path/"store", binding={"fixture": "denied"})
    def deny(identity):
        raise PermissionError("test remains closed")
    with store.exclusive():
        with pytest.raises(PermissionError):
            run_stage(store, "stage", {"operation": "fixture"}, authorize=deny,
                      produce=lambda: (_ for _ in ()).throw(AssertionError()), check=lambda: None)
        assert not store.path("stage/attempt-start.json").exists()


def test_failed_attempt_preserves_cost_and_cannot_silently_restart(tmp_path):
    store = MeasurementStore(tmp_path/"store", binding={"fixture": "failed"})
    calls = []
    def fail():
        calls.append(1)
        raise ValueError("fixture failure")
    with store.exclusive():
        with pytest.raises(ValueError, match="fixture failure"):
            run_stage(store, "stage", {"operation": "fixture"}, authorize=lambda _: None,
                      produce=fail, check=lambda: None)
        finish_ref = store.reference("stage/attempt-finish.json")
        assert store.json(finish_ref)["status"] == "FAILED"
        with pytest.raises(RuntimeError, match="accounted recovery"):
            run_stage(store, "stage", {"operation": "fixture"}, authorize=lambda _: None,
                      produce=fail, check=lambda: None)
        assert len(calls) == 1 and store.reference("stage/attempt-finish.json") == finish_ref
