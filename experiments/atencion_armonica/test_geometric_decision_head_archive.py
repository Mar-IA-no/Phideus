"""144 arithmetic serialized head states, not 72 scientific trainings."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.atencion_armonica import geometric_decision_head_archive as module
from src.atencion_armonica.geometric_decision_store import ArtifactStore


def test_archive_keeps_all_initial_selected_states_and_exact_recovery(tmp_path):
    campaign = ArtifactStore(tmp_path/"training", binding={"fixture": "head-export", "device": "cpu"})
    entries, provenance = [], []
    for cp, arm, seed in module.ROSTER:
        data = {"binding": {"fixture": "data", "cp": cp}, "eligible": {"train": list(range(33))}}
        data_ref = campaign.publish_json(f"data/{cp}.json", data)
        relative = f"cells/cp_{cp}/{arm}/seed_{seed}"
        cell = ArtifactStore(campaign.root/relative, binding={"schema": "geometric-decision-cell-binding-v1",
            "data": data["binding"], "arm": arm, "checkpoint_seed": cp, "reader_seed": seed, "device": "cpu"})
        previous, calibrations = None, []
        for epoch in (0, 5):
            params = {k: torch.full(shape, float(epoch)/100, dtype=torch.float32) for k, shape in module.SHAPES.items()}
            state = {"schema": "geometric-decision-training-state-v1", "binding": cell.binding,
                "checkpoint_seed": cp, "arm": arm, "reader_seed": seed, "epoch": epoch,
                "steps": 2*epoch, "next_batch": 0, "scene_ids": list(range(33)), "model": params}
            state_ref = cell.save_state(SimpleNamespace(state=lambda: state), previous)
            previous = state_ref
            calibrations.append(cell.publish_json(f"cal-{epoch}.json", {"binding": cell.binding,
                "epoch": epoch, "state": state_ref}))
        # Unselected epochs and completion are explicit metadata stubs. The
        # audited training/selection admission is an external prerequisite.
        closed = cell.publish_json("complete.json", {"schema": "geometric-decision-cell-complete-v1",
            "binding": cell.binding, "last_epoch": 50, "steps": 100,
            "history": [{"fixture": "not-trained"}]*50, "calibration": [calibrations[0]]+[calibrations[1]]*10})
        entries.append({"checkpoint_seed": cp, "arm": arm, "reader_seed": seed, "root": relative,
                        "complete": closed, "data": data_ref})
        provenance.extend({"checkpoint_seed": cp, "arm": arm, "reader_seed": seed, "epoch": epoch,
            "calibration": calibrations[1], "state": state_ref} for epoch in range(5, 51, 5))
    complete = campaign.publish_json("complete.json", {"schema": "geometric-decision-campaign-complete-v1",
        "binding": campaign.binding, "cells": entries})
    selection_store = ArtifactStore(tmp_path/"selection", binding={"campaign_binding": campaign.binding})
    selection = selection_store.publish_json("selection.json", {"binding": selection_store.binding,
        "campaign": complete, "calibrations": provenance, "result": {
            "schema": "geometric-decision-calibration-selection-v1", "arms": {arm: {
                "selected_epoch": 5, "epochs": [{"epoch": e, "mean_regret_tD": .1} for e in range(5, 51, 5)]}
                for arm in module.ARMS}}})
    output = ArtifactStore(tmp_path/"heads", binding={"selection": selection, "selection_binding": selection_store.binding})
    ref = module.preserve_heads(selection_store, selection, campaign, complete, output, check=lambda: None)
    report = output.json(ref)
    assert report["count"] == len(report["records"]) == 144
    assert report["new_initializations"] is False
    for i, head_ref in enumerate(report["records"]):
        record, arrays = module.read_head_arrays(output, head_ref)
        assert record["stage"] == ("initial" if i % 2 == 0 else "selected")
        assert all(np.array_equal(a, np.full(a.shape, .0 if i % 2 == 0 else .05, np.float32)) for a in arrays.values())
    assert module.preserve_heads(selection_store, selection, campaign, complete, output, check=lambda: None) == ref
    record = output.json(report["records"][0])
    arrays = output.arrays(record["arrays"])
    arrays["partition2.bias"][0] = .1
    record["arrays"] = output.publish_arrays("bad-initial.npz", arrays)
    bad = output.publish_json("bad-initial.json", record)
    with pytest.raises(ValueError, match="zero residual"):
        module.read_head_arrays(output, bad)


def test_selected_epoch_cannot_choose_later_exact_tie():
    selection = {"result": {"schema": "geometric-decision-calibration-selection-v1", "arms": {
        arm: {"selected_epoch": 10, "epochs": [{"epoch": e, "mean_regret_tD": .2} for e in range(5, 51, 5)]}
        for arm in module.ARMS}}}
    with pytest.raises(ValueError, match="criterion"):
        module.selected_epochs(selection)


@pytest.mark.parametrize("mutation", ["shape", "dtype", "extra", "nan"])
def test_parameter_export_rejects_invalid_numeric_state(mutation):
    params = {k: torch.zeros(shape) for k, shape in module.SHAPES.items()}
    if mutation == "shape":
        params["group1.weight"] = torch.zeros((9, 32))
    elif mutation == "dtype":
        params["group1.weight"] = params["group1.weight"].double()
    elif mutation == "extra":
        params["targets"] = torch.zeros(1)
    else:
        params["group1.weight"][0, 0] = float("nan")
    with pytest.raises(ValueError):
        module.model_arrays({"model": params, "epoch": 5})
