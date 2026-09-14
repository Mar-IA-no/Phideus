"""CPU operator/provenance fixtures, never main, real data, models or samplers."""
import time

import pytest

from experiments.atencion_armonica import select_geometric_decision as op
from src.atencion_armonica.geometric_decision_budget import LIMITS, StageBudget
from src.atencion_armonica.geometric_decision_store import ArtifactStore
from src.atencion_armonica.geometric_decision_selection_profile import (
    profile_selection, selection_forecast, admitted_selection_profile,
)


def stores(tmp_path):
    control = ArtifactStore(tmp_path/"control", binding={"limits": LIMITS, "prior_charges": [],
        "output_roots": [str(tmp_path)]})
    campaign = ArtifactStore(tmp_path/"campaign", binding={"fixture": "not-trained"})
    return control, campaign


def training_receipt(control, campaign, *, status="COMPLETE", omit_cell=False, wrong_root=False):
    manifest = control.publish_json("manifests/training.json", {"operation": "training",
        "campaign_binding": campaign.binding, "root": str(campaign.root)})
    budget = StageBudget(control, "training", manifest_ref=manifest, reservation_seconds=30.,
        prior_charges=[], output_roots=[campaign.root.parent])
    if status != "COMPLETE":
        return budget.finish(status)
    entries = [{"checkpoint_seed": cp, "arm": arm, "reader_seed": seed} for cp, arm, seed in op.ROSTER]
    complete = campaign.publish_json("complete.json", {"schema": "geometric-decision-campaign-complete-v1",
        "binding": campaign.binding, "cells": entries[:-1] if omit_cell else entries})
    output = control.publish_json("outputs/training.json", {"manifest": manifest,
        "root": str(campaign.root/"wrong") if wrong_root else str(campaign.root), "complete": complete})
    return budget.finish("COMPLETE", completion=output)


def test_training_admission_requires_authorized_complete_not_output_presence(tmp_path):
    control, campaign = stores(tmp_path)
    with pytest.raises(ValueError, match="no COMPLETE"):
        op.admitted_training(control, root=campaign.root)
    training_receipt(control, campaign, status="PAUSED")
    with pytest.raises(ValueError, match="no COMPLETE"):
        op.admitted_training(control, root=campaign.root)


@pytest.mark.parametrize("omit_cell,wrong_root,match", [(True, False, "72 cells"), (False, True, "provenance")])
def test_training_admission_rejects_partial_roster_and_wrong_root(tmp_path, omit_cell, wrong_root, match):
    control, campaign = stores(tmp_path)
    training_receipt(control, campaign, omit_cell=omit_cell, wrong_root=wrong_root)
    with pytest.raises(ValueError, match=match):
        op.admitted_training(control, root=campaign.root)


def test_training_admission_returns_authenticated_readonly_campaign(tmp_path):
    control, campaign = stores(tmp_path)
    finish = training_receipt(control, campaign)
    view, complete, evidence = op.admitted_training(control, root=campaign.root)
    assert view.binding == campaign.binding and len(view.json(complete)["cells"]) == 72
    assert evidence["finish"] == finish and not hasattr(view, "publish_json")


def test_bounded_selection_pauses_without_authority_and_keeps_charge(tmp_path, monkeypatch):
    control, campaign = stores(tmp_path)
    output = ArtifactStore(tmp_path/"selection", binding={"fixture": "selection"})
    manifest = control.publish_json("manifests/selection.json", {"operation": "calibration-selection",
        "binding": output.binding, "root": str(output.root)})
    def pause(*args, **kwargs):
        raise InterruptedError("fixture pause")
    monkeypatch.setattr(op, "select_campaign", pause)
    with pytest.raises(InterruptedError):
        op.execute(None, {}, campaign, {}, output, control, manifest, started_at=time.monotonic(),
            verify=lambda: None, reservation=30.)
    first = control.json(control.reference(control.path("attempts/0000/finish.json")))
    assert first["status"] == "PAUSED" and first["completion"] is None
    assert not control.path("outputs/selection.json").exists()
    def completed(*args, **kwargs):
        kwargs["check"]()
        return output.publish_json("selection.json", {"fixture": "success-not-scientific"})
    monkeypatch.setattr(op, "select_campaign", completed)
    result = op.execute(None, {}, campaign, {}, output, control, manifest, started_at=time.monotonic(),
        verify=lambda: None, reservation=30.)
    last = control.json(result["finish"])
    assert last["status"] == "COMPLETE"
    assert last["charged_after"]["evaluation"] == first["seconds"]+last["seconds"]


def test_verification_failure_cannot_authorize_an_output(tmp_path):
    control, _ = stores(tmp_path)
    manifest = control.publish_json("manifests/fixture.json", {"fixture": "verification"})
    calls = []
    def verify():
        calls.append(1)
        if len(calls) == 2:
            raise ValueError("frozen code changed")
    with pytest.raises(ValueError, match="frozen code"):
        op.bounded_operation(control, manifest, stage="evaluation", started_at=time.monotonic(),
            verify=verify, reservation=30., operation=lambda check: control.publish_json("fixture-output.json", {"fixture": 1}))
    finish = control.json(control.reference(control.path("attempts/0000/finish.json")))
    assert finish["status"] == "FAILED" and finish["completion"] is None


def test_full_arithmetic_profile_and_admission_do_not_select_real_epochs(tmp_path):
    control, _ = stores(tmp_path)
    profile = ArtifactStore(tmp_path/"profile", binding={"fixture": "arithmetic-envelope"})
    manifest = control.publish_json("manifests/profile.json", {"operation": "profile-selection",
        "binding": profile.binding, "root": str(profile.root)})
    def operation(check):
        ref = profile_selection(profile, check=check)
        return control.publish_json("outputs/profile.json", {"manifest": manifest, "root": str(profile.root),
            "binding": profile.reference(profile.path("binding.json")), "profile": ref})
    completed = op.bounded_operation(control, manifest, stage="profile", started_at=time.monotonic(),
        verify=lambda: None, reservation=120., operation=operation)
    admission = admitted_selection_profile(control, root=profile.root, corpus_load_seconds=29.)
    assert admission["finish"] == completed["finish"]
    forecast = admission["forecast"]
    assert forecast["corpus_loads"] == 3 and forecast["calibration_reads"] == 1584
    assert forecast["final_state_loads"] == 72 and forecast["state_blob_reads"] == 792
    assert forecast["projected_seconds"] > 1.25*87
    assert not control.path("outputs/selection.json").exists()
    report = profile.json(admission["output"]["profile"])
    report["selection_seconds"] = float("nan")
    with pytest.raises(ValueError, match="positive finite"):
        selection_forecast(report, corpus_load_seconds=29.)
