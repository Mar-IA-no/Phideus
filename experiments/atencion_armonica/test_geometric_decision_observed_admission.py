"""Admission receipts only; no profile execution, draws, arrays or truth."""
import pytest

from src.atencion_armonica import geometric_decision_observed_admission as module
from src.atencion_armonica.geometric_decision_store import ArtifactStore
from src.atencion_armonica.generative_evidence_reuse import VerifiedBytes


def fixture(tmp_path, monkeypatch, *, status="COMPLETE", wrong_archive=False, bad_tail=False):
    control = ArtifactStore(tmp_path/"control", binding={"protocol": {"fixture": "protocol"}})
    archive = ArtifactStore(tmp_path/"archive", binding={"fixture": "archive"})
    prior = archive.publish_json("exclusions.json", {"fixture": "prior"})
    archive_complete = {"binding": archive.binding, "test_authority": False, "exclusions": prior}
    archive_evidence = {"fixture": "archive-complete"}
    binding = {"schema": "geometric-decision-observed-profile-binding-v1",
        "archive": {"fixture": "wrong"} if wrong_archive else archive_evidence,
        "protocol": control.binding["protocol"], "test_authority": False,
        "scene_ids": list(range(16)), "code": [{**prior, "path": "archive/exclusions.json"}]}
    profile = ArtifactStore(tmp_path/"profiles/observed-cuda-0", binding=binding)
    observable = profile.publish_json("observed.json", {"fixture": "observable-complete"})
    report = profile.publish_json("result.json", {"schema": "geometric-decision-observed-profile-v1",
        "binding": binding, "result": observable, "recovery": observable,
        "new_test_observations": 0, "elapsed_to_forecast_seconds": 11. if bad_tail else 8.,
        "forecast": {"test_authority": False, "margin": 1.25,
            "observed_path_seconds": 100., "observable_recovery_seconds": 50., "projected_bytes": 1000}})
    manifest = control.publish_json("manifest.json", {"operation": "profile-observed",
        "root": str(profile.root), "binding": binding})
    start = control.publish_json("start.json", {"binding": control.binding, "stage": "profile",
        "manifest": manifest, "reservation_seconds": 30.})
    output = control.publish_json("output.json", {"root": str(profile.root), "manifest": manifest,
        "binding": profile.reference(profile.path("binding.json")), "result": report})
    finish = control.publish_json("finish.json", {"status": status, "start": start,
        "completion": output if status == "COMPLETE" else None, "seconds": 10.})
    monkeypatch.setattr(module, "PROFILE_FINISH", finish)
    calls = []
    def extension(reader, prior_ref, view, report_ref, *, check):
        # Real extension semantics covered by test_geometric_decision_profile_exclusions.
        assert reader.json(prior_ref) == archive.json(prior)
        assert view.root == profile.root and report_ref == report
        calls.append(prior_ref)
        return {"fixture": "extended", "test_access": False}
    monkeypatch.setattr(module, "extend_profile_exclusions", extension)
    return (control, archive, archive_complete, archive_evidence, VerifiedBytes(tmp_path)), calls


def test_complete_profile_extends_only_archive_prior_and_counts_tail(tmp_path, monkeypatch):
    args, calls = fixture(tmp_path, monkeypatch)
    result = module.admit_profile(*args, check=lambda: None)
    assert len(calls) == 1
    assert result["forecast"]["closing_tail_seconds"] == 2.
    assert result["forecast"]["observed_path_with_closing_seconds"] == 110.
    assert result["forecast"]["observable_recovery_with_closing_seconds"] == 60.
    assert result["forecast"]["still_required"] and result["forecast"]["test_authority"] is False
    assert not hasattr(result["profile"], "publish_json")


@pytest.mark.parametrize("kwargs,match", [({"status": "PAUSED"}, "not COMPLETE"),
    ({"wrong_archive": True}, "another archive"), ({"bad_tail": True}, "closing time")])
def test_admission_rejects_wrong_authority_before_extending(tmp_path, monkeypatch, kwargs, match):
    args, calls = fixture(tmp_path, monkeypatch, **kwargs)
    with pytest.raises(ValueError, match=match):
        module.admit_profile(*args, check=lambda: None)
    assert not calls


def test_changed_profile_source_stops_before_extending(tmp_path, monkeypatch):
    args, calls = fixture(tmp_path, monkeypatch)
    reader = args[-1]
    def changed(ref):
        raise ValueError("fixture source no longer matches frozen bytes")
    monkeypatch.setattr(reader, "read", changed)
    with pytest.raises(ValueError, match="frozen bytes"):
        module.admit_profile(*args, check=lambda: None)
    assert not calls
