"""CPU budget/provenance fixtures; no real TRAIN, sampler or profile main."""
import time

import pytest

from experiments.atencion_armonica import profile_geometric_decision_closing as module
from experiments.atencion_armonica.test_geometric_decision_selection_operation import stores
from src.atencion_armonica.geometric_decision_store import ArtifactStore


def test_projection_covers_preseal_and_postseal_recovery_and_all_remaining_work():
    timings = dict.fromkeys(module.UNITS, 1.)
    observed = {"observed_path_with_closing_seconds": 100.,
        "observable_recovery_with_closing_seconds": 50., "projected_profile_bytes": 100}
    result = module.projection(timings, observed, overhead=2., profile_bytes=100)
    assert result["fresh_seconds"] == 965.
    assert result["evaluation_seconds"] == 1357.5
    assert result["evaluation_inventory_count"] == 5
    assert result["projected_new_bytes"] == 16100
    assert result["preseal_recovery_count"] == result["postseal_recovery_count"] == 1
    assert result["test_authority"] is False and result["not_a_worst_case_bound"]
    with pytest.raises(ValueError, match="all finite"):
        module.projection({**timings, "open-truth": 0.}, observed, overhead=2., profile_bytes=100)


def test_failed_admission_precedes_any_cpu_profile_data_access(tmp_path, monkeypatch):
    control, _ = stores(tmp_path)
    store = ArtifactStore(tmp_path/"profile", binding={"fixture": "no-real-data"})
    manifest = control.publish_json("manifest.json", {"operation": "profile-closing",
        "binding": store.binding, "root": str(store.root)})
    def forbidden(*args, **kwargs):
        raise AssertionError("no sampler or evaluation after failed source admission")
    monkeypatch.setattr(module, "_draw_scene", forbidden)
    monkeypatch.setattr(module, "evaluate_batch", forbidden)
    def interrupted():
        raise InterruptedError("fixture interrupted before data")
    with pytest.raises(InterruptedError):
        module.execute(control, store, manifest, started_at=time.monotonic(), verify=interrupted)
    finish = control.json(control.reference(control.path("attempts/0000/finish.json")))
    assert finish["status"] == "PAUSED" and finish["completion"] is None
    assert not store.path("known-draw/0/intent.json").exists()


def test_material_admission_runs_inside_budget_and_phase_alarm(tmp_path, monkeypatch):
    control, _ = stores(tmp_path)
    store = ArtifactStore(tmp_path/"profile", binding={"fixture": "no-real-data"})
    manifest = control.publish_json("manifest.json", {"operation": "profile-closing",
        "binding": store.binding, "root": str(store.root)})
    checks = []
    def admission(ctrl, *, check):
        assert ctrl is control
        assert control.path("attempts/0000/start.json").exists()
        assert not control.path("attempts/0000/finish.json").exists()
        assert 0 < module.signal.getitimer(module.signal.ITIMER_REAL)[0] <= 120.
        checks.append(check.__self__.stage)
        check(force_resources=True)
        raise module.BudgetExceeded("fixture material admission limit")
    with pytest.raises(module.BudgetExceeded, match="material admission limit"):
        module.execute(control, store, manifest, started_at=time.monotonic(), verify=lambda: None, admit=admission)
    assert checks == ["profile"]
    finish = control.json(control.reference(control.path("attempts/0000/finish.json")))
    assert finish["status"] == "LIMIT_REACHED" and finish["completion"] is None
    assert module.signal.getitimer(module.signal.ITIMER_REAL)[0] == 0
    assert not store.path("admission.json").exists()


def test_inputs_forward_the_live_guard_to_profile_admission(tmp_path, monkeypatch):
    control, _ = stores(tmp_path)
    calls = []
    def check():
        calls.append("check")
    monkeypatch.setattr(module, "admitted_open", lambda: (control, None, None))
    monkeypatch.setattr(module, "admitted_training", lambda *a, **k: (1, 2, 3))
    monkeypatch.setattr(module, "admitted_selection", lambda *a, **k: (4, 5, 6))
    monkeypatch.setattr(module, "admitted_archive", lambda *a, **k: (7, 8, 9))
    def profile(*args, **kwargs):
        assert kwargs["check"] is check
        kwargs["check"]()
        raise InterruptedError("fixture before data")
    monkeypatch.setattr(module, "admit_profile", profile)
    with pytest.raises(InterruptedError):
        module.admit_inputs(control, check=check)
    assert len(calls) == 6


def test_main_opens_material_open_only_inside_live_admission(tmp_path, monkeypatch):
    control, _ = stores(tmp_path)
    monkeypatch.setattr(module, "ROOT", module.Path.cwd().resolve())
    monkeypatch.setattr(module, "BASES", [tmp_path])
    monkeypatch.setattr(module, "CONTROL_BINDING", control.reference(control.path("binding.json")))
    monkeypatch.setattr(module, "sources", lambda: [])
    monkeypatch.setattr(module, "reference", lambda p: {"sha256": module.PROTOCOL_SHA})
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        monkeypatch.setenv(key, "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(module, "LAUNCH_STARTED", time.monotonic())
    calls = []
    def material_open():
        calls.append("open")
        assert control.path("attempts/0000/start.json").exists()
        assert 0 < module.signal.getitimer(module.signal.ITIMER_REAL)[0] <= 120.
        raise InterruptedError("fixture inside material OPEN")
    monkeypatch.setattr(module, "admitted_open", material_open)
    with pytest.raises(InterruptedError, match="inside material OPEN"):
        module.main()
    assert calls == ["open"]
    finish = control.json(control.reference(control.path("attempts/0000/finish.json")))
    assert finish["status"] == "PAUSED" and finish["completion"] is None
    assert not (tmp_path/"profiles/closing-cpu-0/admission.json").exists()
