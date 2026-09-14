"""Budgeted controller fixtures: main and CUDA are never invoked."""
import time
from types import SimpleNamespace

import pytest

from experiments.atencion_armonica import train_geometric_decision as op
from src.atencion_armonica.geometric_decision_store import ArtifactStore
from src.atencion_armonica.geometric_decision_budget import LIMITS


def stores(tmp_path):
    control = ArtifactStore(tmp_path/"control", binding={"limits": LIMITS, "prior_charges": [], "output_roots": [str(tmp_path)]})
    campaign = ArtifactStore(tmp_path/"campaign", binding={"device": "cpu", "fixture": "operator-only"})
    manifest = control.publish_json("manifest.json", {"operation": "training", "campaign_binding": campaign.binding, "root": str(campaign.root)})
    return control, campaign, manifest


def test_finish_complete_requires_runner_success_and_keeps_cost(tmp_path, monkeypatch):
    control, campaign, manifest = stores(tmp_path)
    def complete(prep, ref, store, **kwargs):
        kwargs["check"]()
        return store.publish_json("complete.json", {"fixture": "operator-success-not-a-scientific-campaign"})
    monkeypatch.setattr(op, "run_campaign", complete)
    result = op.execute(SimpleNamespace(), {}, campaign, control, manifest, started_at=time.monotonic(),
                        reservation=10., verify=lambda: None, progress=lambda row: None)
    finish = control.json(result["finish"])
    assert finish["status"] == "COMPLETE" and finish["charged_after"]["training"] > 0
    assert op.training_charged(control) == finish["seconds"]
    owner = control.json(control.reference(control.path("attempts/0000/owner.json")))
    assert owner["device"] == "cpu" and owner["availability"] is None and owner["pid"] > 0


def test_pause_is_not_complete_and_next_attempt_retains_charge(tmp_path, monkeypatch):
    control, campaign, manifest = stores(tmp_path)
    def pause(*args, **kwargs):
        raise InterruptedError("explicit arithmetic pause")
    monkeypatch.setattr(op, "run_campaign", pause)
    with pytest.raises(InterruptedError):
        op.execute(SimpleNamespace(), {}, campaign, control, manifest, started_at=time.monotonic(),
                   reservation=10., verify=lambda: None, progress=lambda row: None)
    finish = control.json(control.reference(control.path("attempts/0000/finish.json")))
    assert finish["status"] == "PAUSED" and finish["completion"] is None
    assert not control.path("outputs/training.json").exists()
    assert op.training_charged(control) > 0
