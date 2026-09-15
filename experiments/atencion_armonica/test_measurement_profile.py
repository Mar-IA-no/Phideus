"""Profile wiring on manual audio with a stub readout: no GPU or new sampler."""
from types import SimpleNamespace

import pytest

from experiments.atencion_armonica.test_measurement_open import manual_draw
from src.atencion_armonica.measurement_open import cpu_profile
from src.atencion_armonica.measurement_store import MeasurementStore
from src.atencion_armonica import measurement_profile as module


def test_profile_keeps_all_sixteen_units_and_requires_same_source(tmp_path, monkeypatch):
    calls = []
    def manual_result(store, folder, prepared, **kwargs):
        calls.append(prepared["unit"])
        return {"schema": "measurement-inference-unit-v1", "unit": prepared["unit"],
                "status": ("NO_CANDIDATE" if prepared["status"] == "ELIGIBLE" else prepared["status"]),
                "candidate_count": 0, "stages": {}, "choices": {}, "seconds": {}}
    monkeypatch.setattr(module, "predict_unit", manual_result)
    store = MeasurementStore(tmp_path/"store", binding={"fixture": "profile-wiring"})
    backend = SimpleNamespace(runtime={"fixture": "no CUDA"}, ownership={"fixture": "no device"})
    with store.exclusive():
        cpu, arrays = cpu_profile(store, "manual", lambda: None, draw=manual_draw)
        ref = store.publish_stage("cpu-profile", {"phase": "profile_cpu", "source_snapshot": "manual"}, cpu, arrays)
        result, arrays = module.gpu_profile(store, "manual", ref, {}, backend, lambda: None)
        module.validate_gpu_profile(store, "manual", ref, result, arrays, lambda: None)
        assert len(calls) == len(result["records"]) == 16
        assert [r["unit"] for r in result["records"]] == [r["unit"] for r in cpu["rows"]]
        replay, arrays = module.gpu_profile(store, "manual", ref, {}, backend, lambda: None)
        module.validate_gpu_profile(store, "manual", ref, replay, arrays, lambda: None)
        assert len(calls) == 16
        with pytest.raises(ValueError, match="exact admitted CPU"):
            module.gpu_profile(store, "different", ref, {}, backend, lambda: None)
