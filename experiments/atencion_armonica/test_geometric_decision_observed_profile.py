"""Resource projection and failed profile receipts; CPU-only fixtures."""
import time

import pytest

from experiments.atencion_armonica import profile_geometric_decision_observed as module
from experiments.atencion_armonica.test_geometric_decision_selection_operation import stores
from src.atencion_armonica.geometric_decision_store import ArtifactStore


def test_projection_preserves_full_roster_and_marks_unmeasured_costs():
    timings = dict.fromkeys(("archive", "original-observable", "original-classical", "original-readout",
        "roundtrip-observable", "roundtrip-classical", "roundtrip-readout", "observable-recovery"), 1.)
    result = module.projection(timings, profile_bytes=100, overhead_seconds=2.)
    assert result["fresh_scenes"] == 2048
    assert result["observed_path_seconds"] == 510.
    assert result["observable_recovery_seconds"] == 170.
    assert result["projected_bytes"] == 16000
    assert set(result["phase_units"]) == set(timings)
    assert result["phase_units"]["original-readout"]["head_states"] == 144
    assert result["phase_units"]["observable-recovery"]["units"] == 20
    assert result["not_measured"] and result["test_authority"] is False
    with pytest.raises(ValueError, match="full nonempty"):
        module.projection({**timings, "roundtrip-readout": 0.}, profile_bytes=100, overhead_seconds=2.)


def test_profile_failure_before_gpu_preserves_attempt_and_no_completion(tmp_path, monkeypatch):
    control, _ = stores(tmp_path)
    store = ArtifactStore(tmp_path/"profile", binding={"fixture": "no-GPU"})
    manifest = control.publish_json("manifests/observed.json", {"operation": "profile-observed",
        "root": str(store.root), "binding": store.binding})
    def forbidden(*args, **kwargs):
        raise AssertionError("no CUDA after failed source admission")
    monkeypatch.setattr(module, "gpu_runtime", forbidden)
    def interrupted():
        raise InterruptedError("fixture interrupted before GPU")
    with pytest.raises(InterruptedError):
        module.execute(control, store, manifest, preparation=None, open_ref=None, reuse=None,
            observations=None, archive=None, archive_complete=None, selection=None, selection_ref=None,
            verify=interrupted, started_at=time.monotonic(), reservation=30.)
    finish = control.json(control.reference(control.path("attempts/0000/finish.json")))
    assert finish["status"] == "PAUSED" and finish["completion"] is None and finish["seconds"] > 0
    assert not control.path("outputs/profile-observed.json").exists()
