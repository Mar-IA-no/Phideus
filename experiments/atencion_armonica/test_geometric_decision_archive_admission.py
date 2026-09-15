"""Closed-operator and archive membership fixtures, not scientific training."""
from copy import deepcopy
import time

import numpy as np
import pytest

from experiments.atencion_armonica import prepare_geometric_decision_archive as op
from experiments.atencion_armonica.test_geometric_decision_selection_operation import stores, training_receipt
from src.atencion_armonica import geometric_decision_archive_admission as admission
from src.atencion_armonica.geometric_decision_budget import StageBudget
from src.atencion_armonica.geometric_decision_head_archive import SHAPES, ARMS, ROSTER
from src.atencion_armonica.geometric_decision_store import ArtifactStore


def selection_receipt(control, campaign, *, status="COMPLETE", wrong_training=False):
    _, campaign_ref, training = op.admitted_training(control, root=campaign.root)
    selection = ArtifactStore(campaign.root.parent/"selection", binding={
        "schema": "geometric-decision-selection-binding-v1", "campaign": campaign_ref,
        "campaign_binding": campaign.binding, "training": {} if wrong_training else training})
    manifest = control.publish_json("manifests/selection.json", {"operation": "calibration-selection",
        "binding": selection.binding, "root": str(selection.root)})
    budget = StageBudget(control, "evaluation", manifest_ref=manifest, reservation_seconds=30.,
        prior_charges=[], output_roots=[campaign.root.parent])
    targets = selection.publish_arrays("targets.npz", {"fixture": np.zeros(1)})
    record = {"binding": selection.binding, "campaign": campaign_ref,
        "fresh_tests": "not opened", "calibration_identities": [f"fixture-{i}" for i in range(512)],
        "targets": targets, "calibrations": [{"checkpoint_seed": cp, "arm": arm, "reader_seed": seed,
            "epoch": e, "calibration": {"fixture": e}, "state": {"fixture": e}}
            for cp, arm, seed in ROSTER for e in range(5, 51, 5)],
        "result": {"schema": "geometric-decision-calibration-selection-v1", "arms": {
            arm: {"selected_epoch": 5, "epochs": [{"epoch": e, "mean_regret_tD": .1} for e in range(5, 51, 5)]}
            for arm in ARMS}}}
    ref = selection.publish_json("selection.json", record)
    output = control.publish_json("outputs/selection.json", {"manifest": manifest,
        "root": str(selection.root), "selection": ref})
    finish = budget.finish(status, completion=output if status == "COMPLETE" else None)
    return selection, ref, finish, campaign_ref, training


@pytest.mark.parametrize("status", ["PAUSED", "FAILED", "COMPLETE"])
def test_only_complete_selection_is_admitted(tmp_path, status):
    control, campaign = stores(tmp_path)
    training_receipt(control, campaign)
    selection, ref, finish, campaign_ref, training = selection_receipt(control, campaign, status=status)
    kwargs = {"root": selection.root, "campaign": campaign, "campaign_ref": campaign_ref, "training": training}
    if status != "COMPLETE":
        with pytest.raises(ValueError, match="no COMPLETE"):
            admission.admitted_selection(control, **kwargs)
    else:
        view, actual, evidence = admission.admitted_selection(control, **kwargs)
        assert actual == ref and evidence["finish"] == finish and view.binding == selection.binding
        assert not hasattr(view, "publish_json")


def test_selection_cannot_belong_to_another_training_authority(tmp_path):
    control, campaign = stores(tmp_path)
    training_receipt(control, campaign)
    selection, _, _, campaign_ref, training = selection_receipt(control, campaign, wrong_training=True)
    with pytest.raises(ValueError, match="another admitted training"):
        admission.admitted_selection(control, root=selection.root, campaign=campaign, campaign_ref=campaign_ref, training=training)


def test_archive_admission_has_no_target_array_capability(tmp_path, monkeypatch):
    control, campaign = stores(tmp_path)
    training_receipt(control, campaign)
    selection, ref, _, campaign_ref, training = selection_receipt(control, campaign)
    def forbidden(*args, **kwargs):
        raise AssertionError("archive admission cannot parse calibration target arrays")
    monkeypatch.setattr(admission.ReadOnlyStore, "arrays", forbidden)
    _, actual, _ = admission.admitted_selection(control, root=selection.root,
        campaign=campaign, campaign_ref=campaign_ref, training=training)
    assert actual == ref


def test_archive_membership_all_144_and_wrong_member_rejected(tmp_path):
    control, campaign = stores(tmp_path)
    training_receipt(control, campaign)
    selection, ref, _, campaign_ref, _ = selection_receipt(control, campaign)
    output = ArtifactStore(tmp_path/"archive", binding={"selection": ref, "selection_binding": selection.binding})
    arrays = output.publish_arrays("numeric.npz", {k: np.zeros(s, np.float32) for k, s in SHAPES.items()})
    records = []
    for cp, arm, seed in ROSTER:
        for stage in ("initial", "selected"):
            e = 0 if stage == "initial" else 5
            records.append(output.publish_json(f"heads/cp_{cp}/{arm}/seed_{seed}/{stage}.json", {
                "schema": "geometric-decision-frozen-head-v1", "binding": output.binding,
                "checkpoint_seed": cp, "arm": arm, "reader_seed": seed, "stage": stage, "epoch": e,
                "arrays": arrays, "source": {"campaign": campaign_ref, "root": f"cells/cp_{cp}/{arm}/seed_{seed}",
                    "calibration": {"fixture": e}, "state": {"fixture": e}}}))
    value = {"schema": "geometric-decision-head-archive-v1", "binding": output.binding,
        "selection": ref, "selected_epochs": {arm: 5 for arm in ARMS}, "count": 144,
        "records": records, "new_initializations": False, "forward": False}
    archive_ref = output.publish_json("heads.json", value)
    assert len(admission.verify_heads(output, archive_ref, selection, ref, check=lambda: None)) == 144
    changed = deepcopy(value)
    changed["records"][0], changed["records"][1] = changed["records"][1], changed["records"][0]
    bad = output.publish_json("swapped.json", changed)
    with pytest.raises(ValueError, match="identity/order"):
        admission.verify_heads(output, bad, selection, ref, check=lambda: None)


def test_archive_operator_keeps_failed_attempt_and_no_completion(tmp_path, monkeypatch):
    control, campaign = stores(tmp_path)
    output = ArtifactStore(tmp_path/"archive", binding={"fixture": "archive"})
    manifest = control.publish_json("manifests/archive.json", {"operation": "archive-and-exclusions",
        "binding": output.binding, "root": str(output.root)})
    def interrupted(*args, **kwargs):
        raise InterruptedError("fixture archive pause")
    monkeypatch.setattr(op, "preserve_heads", interrupted)
    with pytest.raises(InterruptedError):
        op.execute(control, output, None, {}, campaign, {}, None, manifest, source_refs=[],
            started_at=time.monotonic(), verify=lambda: None, reservation=30.)
    finish = control.json(control.reference(control.path("attempts/0000/finish.json")))
    assert finish["status"] == "PAUSED" and finish["seconds"] > 0 and finish["completion"] is None
    assert not control.path("outputs/archive.json").exists()
