import numpy as np
import pytest

from src.atencion_armonica.measurement_admission import MeasurementAdmission
from src.atencion_armonica.measurement_contract import calibrate_detector, unit_roster
from src.atencion_armonica.measurement_control import PhaseController
from src.atencion_armonica.measurement_store import MeasurementStore


def setup(tmp_path):
    store = MeasurementStore(tmp_path/"store", binding={"fixture": "admission"})
    control = PhaseController(store, limits={"total_seconds": 100., "gpu_seconds": 60., "reserve_seconds": 25.},
                              clock=lambda: 0., boot=lambda: "fixture")
    def sources(ref):
        if ref != "pinned-fixture":
            raise ValueError("source snapshot changed")
    return store, control, MeasurementAdmission(control, verify_sources=sources)


def open_phases(control, admission):
    for phase in ("profile_cpu", "profile_gpu", "calibration"):
        result = (calibrate_detector(np.ones((9, 192), dtype=np.float64),
                                     unit_roster("calibration", audio_only=True))
                  if phase == "calibration" else {"fixture_profile": True})
        control.run(phase, {"phase": phase, "source_snapshot": "pinned-fixture"}, reservation=5., gpu=False,
                    admit=admission.admit, produce=lambda check, result=result: (result, {}),
                    validate=lambda r, a: None, resources=lambda: None)


def freeze(store, admission, **changes):
    prior = store.publish_json("prior.json", {"mechanical_fixture_only": True})
    exclusions = store.publish_json("exclusions.json", {"schema": "measurement-exclusions-v1",
                                                        "fingerprints": [], "sources": [prior]})
    args = dict(source_snapshot="pinned-fixture", exclusions=exclusions,
                projected_work_seconds=10., projected_replay_seconds=10., projected_gpu_seconds=5.,
                reservation=5., resources=lambda: None)
    return admission.freeze(**{**args, **changes})


def test_test_and_evaluation_stay_closed_until_prerequisites(tmp_path):
    store, control, admission = setup(tmp_path)
    with store.exclusive():
        with pytest.raises(PermissionError):
            admission.admit("test_observation", {"phase": "test_observation", "freeze": None})
        open_phases(control, admission)
        ref = freeze(store, admission)
        admission.admit("test_observation", {"phase": "test_observation", "freeze": ref})
        with pytest.raises(PermissionError):
            admission.admit("test_prediction", {"phase": "test_prediction", "freeze": ref})
        with pytest.raises(PermissionError):
            admission.admit("evaluation", {"phase": "evaluation", "freeze": ref})
        with pytest.raises(PermissionError, match="closed after freeze"):
            admission.admit("calibration", {"phase": "calibration", "source_snapshot": "pinned-fixture"})


def test_projection_and_gpu_reserve_are_checked_before_freeze(tmp_path):
    store, control, admission = setup(tmp_path)
    with store.exclusive():
        open_phases(control, admission)
        with pytest.raises(RuntimeError, match="does not fit"):
            freeze(store, admission, projected_work_seconds=55.)
        with pytest.raises(PermissionError, match="required phase"):
            admission.verify_freeze()
        # No freeze payload exists: an explicit revised projection is still legal.
        with pytest.raises(RuntimeError, match="does not fit"):
            freeze(store, admission, projected_work_seconds=40., projected_gpu_seconds=41.)


def test_incomplete_paired_roster_cannot_seal(tmp_path):
    store, control, admission = setup(tmp_path)
    with store.exclusive():
        open_phases(control, admission)
        ref = freeze(store, admission)
        malformed = {"schema": "measurement-test_observation-index-v1", "freeze": ref,
                     "units": unit_roster("test")[:-1], "records": []}
        control.run("test_observation", {"phase": "test_observation", "freeze": ref}, reservation=5., gpu=False,
                    admit=admission.admit, produce=lambda check: (malformed, {}),
                    validate=lambda r, a: None, resources=lambda: None)
        with pytest.raises(ValueError, match="exact paired"):
            admission.seal_predictions(ref, reservation=5., resources=lambda: None)


def test_new_sources_do_not_enter_existing_freeze(tmp_path):
    store, control, admission = setup(tmp_path)
    with store.exclusive():
        open_phases(control, admission)
        with pytest.raises(ValueError, match="snapshot changed"):
            freeze(store, admission, source_snapshot="different")


def test_different_valid_snapshots_cannot_mix_prerequisites(tmp_path):
    store, control, admission = setup(tmp_path)
    admission.verify_sources = lambda ref: None
    with store.exclusive():
        open_phases(control, admission)
        with pytest.raises(ValueError, match="source identity"):
            control.run("profile_gpu", {"phase": "profile_gpu", "source_snapshot": "snapshot-b"},
                        reservation=5., gpu=False, admit=admission.admit,
                        produce=lambda check: ({}, {}), validate=lambda r, a: None, resources=lambda: None)
        with pytest.raises(ValueError, match="source identity"):
            freeze(store, admission, source_snapshot="snapshot-b")
        with pytest.raises(PermissionError, match="required phase"):
            admission.verify_freeze()


def test_freeze_and_seal_verification_costs_are_charged(tmp_path, monkeypatch):
    store, control, admission = setup(tmp_path)
    from experiments.atencion_armonica.test_measurement_control import Clock
    clock = Clock()
    control.clock = clock
    with store.exclusive():
        open_phases(control, admission)
        original = admission.verify_sources
        def sources(ref):
            clock.advance(.1)
            original(ref)
        admission.verify_sources = sources
        frozen = freeze(store, admission)
        assert control.costs()["total_seconds"] == clock.value > 0
        measured = control.costs()["total_seconds"]
        def index(phase):
            clock.advance(.2)
            raise ValueError("fixture index rejection")
        monkeypatch.setattr(admission, "_unit_index", index)
        with pytest.raises(ValueError, match="index rejection"):
            admission.seal_predictions(frozen, reservation=5., resources=lambda: None)
        assert control.costs()["total_seconds"] == clock.value > measured
        with pytest.raises(PermissionError, match="required phase"):
            admission.phase_result("seal")
