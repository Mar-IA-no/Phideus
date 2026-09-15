"""Manual observable fixtures and tiny CPU fitting, never prospective samples."""
from copy import deepcopy

import numpy as np
import pytest

from src.atencion_armonica.measurement_contract import identity
from src.atencion_armonica.measurement_operator import prepare_input
from src.atencion_armonica.measurement_payload import pack, unpack, stage
from src.atencion_armonica.measurement_prediction import predict_unit
from src.atencion_armonica.measurement_reuse import CHECKPOINTS, READERS, EPOCHS
from src.atencion_armonica.measurement_store import MeasurementStore


def test_typed_payload_preserves_integer_keys_tuples_dtypes_and_refuses_extra_arrays(tmp_path):
    value = {CHECKPOINTS[0]: {"partition": ((0, 1), (2, 3)),
                            "matrix": np.arange(4, dtype=np.float32).reshape(2, 2)},
             "other": [None, True, 1, 1., "1", np.zeros(0, np.int64)]}
    metadata, arrays = pack(value)
    decoded = unpack(metadata, arrays)
    assert type(next(iter(decoded))) is int
    assert type(decoded[CHECKPOINTS[0]]["partition"]) is tuple
    assert decoded[CHECKPOINTS[0]]["matrix"].dtype == np.float32
    assert decoded["other"][:-1] == value["other"][:-1]
    with pytest.raises(ValueError, match="unexpected"):
        unpack(metadata, {**arrays, "extra": np.zeros(0)})
    with pytest.raises(ValueError, match="finite"):
        pack({"bad": np.array([np.nan])})
    store = MeasurementStore(tmp_path/"store", binding={"fixture": "tree"})
    with store.exclusive():
        first, _ = stage(store, "tree", {"fixture": 1}, produce=lambda: value,
                         validate=lambda r: None, check=lambda: None)
        second, saved = stage(store, "tree", {"fixture": 1},
            produce=lambda: (_ for _ in ()).throw(AssertionError("recomputed")),
            validate=lambda r: None, check=lambda: None)
        assert first == second
        np.testing.assert_array_equal(saved[CHECKPOINTS[0]]["matrix"], value[CHECKPOINTS[0]]["matrix"])


class ManualBackend:
    def __init__(self):
        self.calls = {"forward": 0, "fit": 0, "predict": 0}
    def execution(self, check):
        return {"fixture": "CPU manual backend, not a GPU execution"}
    def validate_execution(self, ref):
        assert ref == {"fixture": "CPU manual backend, not a GPU execution"}
    def forward(self, cp, features, check):
        self.calls["forward"] += 1
        n = len(features["tokens"])
        z = np.full((n, n), -3., np.float32)
        z[:4, :4] = z[4:, 4:] = 3.
        return z
    def fit(self, q, partitions, check):
        self.calls["fit"] += 1
        from src.atencion_armonica.observable_source_rivals import fit_candidates, GroupFitter, Grid
        return fit_candidates(q, partitions, GroupFitter(Grid(5, 5, 4), device="cpu"))
    def predict(self, head, row, check):
        self.calls["predict"] += 1
        n = len(row["globals"])
        return {"components": np.zeros((n, 2), np.float64), "energy": np.zeros(n, np.float64),
                "offsets": np.array([0, n], np.int64)}


def manual_reuse():
    from experiments.atencion_armonica.test_geometric_decision_observables import norms
    return {"checkpoints": [{"seed": cp, "fixture": True} for cp in CHECKPOINTS],
            "normalizers": {**norms(), "unused_provenance": "not reestimated"}, "scale": 2.,
            "references": [{"fixture": "manual CPU"}],
            "heads": [{"arm": arm, "checkpoint_seed": cp, "reader_seed": seed,
                       "record": {"fixture": [arm, cp, seed]}}
                      for arm in EPOCHS for cp in CHECKPOINTS for seed in READERS]}


def test_observable_pipeline_saves_all_heads_and_replays_without_backend(tmp_path):
    unit = identity("development", "iid", "canonical", 0)
    q = np.array([-1.1, -.8, -.5, -.2, .2, .5, .8, 1.1], np.float32)
    prepared = prepare_input(unit, q)
    reused, backend = manual_reuse(), ManualBackend()
    store = MeasurementStore(tmp_path/"store", binding={"fixture": "prediction"})
    with store.exclusive():
        source = store.publish_json("manual-observation.json", prepared)
        result = predict_unit(store, "unit", prepared, observation_ref=source,
            source_snapshot="manual-fixture", reused=reused, backend=backend, check=lambda: None)
        assert result["status"] == "ELIGIBLE"
        assert result["candidate_count"] > 0
        assert backend.calls == {"forward": 3, "fit": 1, "predict": 36}
        assert len(result["choices"]) == 40  # Four raw/delivered classics plus 36 heads.
        backend.execution = lambda check: (_ for _ in ()).throw(AssertionError("new execution on cache hit"))
        second = predict_unit(store, "unit", prepared, observation_ref=source,
            source_snapshot="manual-fixture", reused=reused, backend=backend, check=lambda: None)
        assert backend.calls == {"forward": 3, "fit": 1, "predict": 36}
        assert result["stages"] == second["stages"]
        assert result["choices"] == second["choices"]


def test_outside_domain_never_reaches_features_or_backend(tmp_path, monkeypatch):
    import src.atencion_armonica.measurement_prediction as module
    def forbidden(*args, **kwargs):
        raise AssertionError("out-of-domain observation reached expensive code")
    monkeypatch.setattr(module, "features_for_input", forbidden)
    prepared = prepare_input(identity("development", "iid", "nominal", 0), np.array([100., 200.]))
    store = MeasurementStore(tmp_path/"store", binding={"fixture": "outside"})
    with store.exclusive():
        result = predict_unit(store, "unit", prepared, observation_ref="manual", source_snapshot="manual",
                              reused={}, backend=None, check=lambda: None)
        assert result["status"] == "OUTSIDE_OPERATOR_DOMAIN" and result["stages"] == {}


def test_cuda_service_rejects_unmeasured_use_without_querying_device(tmp_path):
    from src.atencion_armonica.measurement_cuda import CUDABackend
    from src.atencion_armonica.measurement_control import PhaseController, LIMITS
    store = MeasurementStore(tmp_path/"store", binding={"fixture": "no-CUDA"})
    control = PhaseController(store, limits=LIMITS)
    def forbidden():
        raise AssertionError("ownership/device queried without active GPU phase")
    backend = CUDABackend(control, {}, source_snapshot="manual", verify_gpu_ownership=forbidden)
    with store.exclusive():
        with pytest.raises(PermissionError, match="active measured"):
            backend.forward(CHECKPOINTS[0], {}, lambda: None)
        assert backend.peak_reserved() == 0
        backend.close()
        with pytest.raises(PermissionError, match="closed CUDA"):
            backend.forward(CHECKPOINTS[0], {}, lambda: None)


@pytest.mark.parametrize("receipt_available", [False, True])
def test_cuda_measurement_and_cleanup_fail_closed_but_allow_active_timeout_cleanup(tmp_path, receipt_available):
    from types import SimpleNamespace
    from src.atencion_armonica.measurement_cuda import CUDABackend
    from src.atencion_armonica.measurement_control import PhaseController, LIMITS
    from experiments.atencion_armonica.test_measurement_control import Clock
    store = MeasurementStore(tmp_path/"store", binding={"fixture": "CUDA-lifecycle"})
    clock, calls = Clock(), []
    control = PhaseController(store, limits=LIMITS, clock=clock, boot=lambda: "manual")
    backend = CUDABackend(control, {}, source_snapshot="manual", verify_gpu_ownership=lambda: {"fixture": True})
    backend.torch = SimpleNamespace(cuda=SimpleNamespace(
        max_memory_reserved=lambda i: calls.append("peak") or 1,
        synchronize=lambda i: calls.append("sync"), empty_cache=lambda: calls.append("empty")))
    with store.exclusive():
        for method in (backend.peak_reserved, backend.close):
            with pytest.raises(PermissionError, match="active measured"):
                method()
        assert not calls
        def produce(check):
            backend._phase = control._active
            backend.ownership = {"fixture": True}
            backend.runtime_ref = {"fixture": "no actual CUDA"} if receipt_available else None
            if receipt_available:
                assert backend.peak_reserved() == 1
            else:
                with pytest.raises(PermissionError, match="ownership receipt"):
                    backend.peak_reserved()
            clock.advance(11)
            backend.close()  # Cleanup still works before control closes its failed attempt.
            return {}, {}
        with pytest.raises(TimeoutError):
            control.run("profile_gpu", {"phase": "profile_gpu", "source_snapshot": "manual"},
                        reservation=10, gpu=True, admit=lambda p, i: None, produce=produce,
                        validate=lambda r, a: None, resources=lambda: None)
        expected = (["peak"] if receipt_available else [])+["sync", "peak", "empty"]
        assert calls == expected
        assert control.costs()["gpu_seconds"] == 11
        assert backend.peak_reserved() == 1  # Final guard reads cached peak, not CUDA.
        assert calls == expected


def test_old_execution_receipt_can_be_authenticated_without_starting_cuda(tmp_path):
    from src.atencion_armonica.measurement_cuda import CUDABackend
    from src.atencion_armonica.measurement_control import PhaseController, LIMITS
    store = MeasurementStore(tmp_path/"store", binding={"fixture": "execution-provenance"})
    control = PhaseController(store, limits=LIMITS)
    reused = {"runtime": {"torch": "manual", "numpy": "manual"}}
    backend = CUDABackend(control, reused, source_snapshot="manual",
                          verify_gpu_ownership=lambda: (_ for _ in ()).throw(AssertionError("queried GPU")))
    with store.exclusive():
        def produce(check):
            ref = store.publish_json("manual-execution.json", {
                "schema": "measurement-cuda-execution-v1", "start": control._active,
                "source_snapshot": "manual", "runtime": {"torch": "manual", "numpy": "manual",
                    "cuda": "manual", "cudnn": "manual", "device": "NVIDIA GeForce RTX 3090"},
                "ownership": {"fixture": "not a real execution"}, "settings": backend.SETTINGS})
            return {"execution": ref}, {}
        _, result, _ = control.run("profile_gpu", {"phase": "profile_gpu", "source_snapshot": "manual"},
            reservation=10, gpu=True, admit=lambda p, i: None, produce=produce,
            validate=lambda r, a: None, resources=lambda: None)
        assert control._active is None
        row = backend.validate_execution(result["execution"])
        assert row["source_snapshot"] == "manual" and backend.torch is None
        wrong = store.publish_json("wrong-execution.json", {**row, "source_snapshot": "changed"})
        with pytest.raises(ValueError, match="provenance"):
            backend.validate_execution(wrong)
