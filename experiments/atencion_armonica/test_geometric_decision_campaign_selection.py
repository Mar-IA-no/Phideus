"""Selection wiring over arithmetic outputs; these are not trained cells."""
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.atencion_armonica.test_geometric_decision_cell import dataset
from src.atencion_armonica.geometric_decision_campaign import ROSTER, cell_identity
from src.atencion_armonica.geometric_decision_campaign_selection import select_campaign
from src.atencion_armonica.geometric_decision_cell import EPOCHS
from src.atencion_armonica.geometric_decision_store import ArtifactStore


def test_complete_selection_wiring_keeps_all_cells_and_one_epoch_per_arm(tmp_path, dataset):
    preparation = SimpleNamespace(store=SimpleNamespace(binding={"fixture": "preparation"}),
                                  load_checkpoint=lambda *args, **kwargs: dataset)
    open_ref = {"fixture": "open-complete"}
    campaign = ArtifactStore(tmp_path/"campaign", binding={"preparation": preparation.store.binding,
                            "open_complete": open_ref, "device": "cpu"})
    entries = []
    targets = [row["targets"] for row in dataset.rows["calibration"]]
    offsets = np.r_[np.int64(0), np.cumsum([len(t) for t in targets], dtype=np.int64)]
    for cp, arm, seed in ROSTER:
        relative = f"cells/cp_{cp}/{arm}/seed_{seed}"
        cell = ArtifactStore(campaign.root/relative, binding=cell_identity(dataset, cp, arm, seed, "cpu"))
        refs, previous = [], None
        for epoch in EPOCHS:
            numeric = {"binding": cell.binding, "steps": 2*epoch, "epoch": epoch, "next_batch": 0,
                "history": [{"fixture_epoch": i+1} for i in range(epoch)],
                "fixture": "arithmetic-serialization-not-trained"}
            state = cell.save_state(SimpleNamespace(state=lambda: numeric), previous)
            previous = state
            preferred = arm != "geometric_decision" or epoch == 10
            energy = np.tile([1., 0.] if preferred else [0., 1.], int(offsets[-1])//2)
            arrays = cell.publish_arrays(f"fixture-pred-{epoch}.npz", {
                "energy": energy, "components": np.column_stack((energy/2, energy/2)), "offsets": offsets})
            refs.append(cell.publish_json(f"fixture-cal-{epoch}.json", {"schema": "geometric-decision-calibration-v1",
                "binding": cell.binding, "epoch": epoch, "state": state,
                "identities": [r["identity"] for r in dataset.rows["calibration"]],
                "eligible_scene_ids": dataset.eligible["calibration"], "predictions": arrays}))
        ref = cell.publish_json("complete.json", {"schema": "geometric-decision-cell-complete-v1",
            "binding": cell.binding, "last_epoch": 50, "steps": 100, "calibration": refs,
            "last_state": state, "history": numeric["history"]})
        entries.append({"checkpoint_seed": cp, "arm": arm, "reader_seed": seed, "root": relative, "complete": ref})
    ref = campaign.publish_json("complete.json", {"schema": "geometric-decision-campaign-complete-v1",
        "binding": campaign.binding, "cells": entries})
    output = ArtifactStore(tmp_path/"selection", binding={"campaign": ref, "campaign_binding": campaign.binding})
    selected = output.json(select_campaign(preparation, open_ref, campaign, ref, output, check=lambda: None))
    assert len(selected["calibrations"]) == 720
    assert selected["fresh_tests"] == "not opened"
    for arm, result in selected["result"]["arms"].items():
        assert result["selected_epoch"] == (10 if arm == "geometric_decision" else 5)
    saved_targets = output.arrays(selected["targets"])
    np.testing.assert_array_equal(saved_targets["targets"], np.concatenate(targets))
    np.testing.assert_array_equal(saved_targets["offsets"], offsets)
    # A recovery after publication must reuse the exact NPZ bytes and JSON,
    # not serialize a fresh zip timestamp or silently replace a selection.
    second = select_campaign(preparation, open_ref, campaign, ref, output, check=lambda: None)
    assert output.json(second) == selected
    assert len(list(output.root.glob("*.npz"))) == 1
    # Missing numeric state cannot be hidden behind still-valid JSON receipts.
    cp, arm, seed = ROSTER[0]
    first_cell = ArtifactStore(campaign.root/entries[0]["root"], binding=cell_identity(dataset, cp, arm, seed, "cpu"))
    first_complete = first_cell.json(entries[0]["complete"])
    for name, mutation in (("extra", {"unexpected": True}), ("history", {"history": [{}]*50}),
                           ("last_state", {"last_state": first_cell.json(first_complete["calibration"][0])["state"]})):
        bad_cell = first_cell.publish_json(f"bad-{name}.json", {**first_complete, **mutation})
        bad_entries = [{**entries[0], "complete": bad_cell}, *entries[1:]]
        bad_ref = campaign.publish_json(f"bad-{name}.json", {"schema": "geometric-decision-campaign-complete-v1",
            "binding": campaign.binding, "cells": bad_entries})
        bad_output = ArtifactStore(tmp_path/f"selection-bad-{name}", binding={"campaign": bad_ref,
            "campaign_binding": campaign.binding})
        with pytest.raises(ValueError):
            select_campaign(preparation, open_ref, campaign, bad_ref, bad_output, check=lambda: None)
        assert not bad_output.path("selection.json").exists()
    first_calibration = first_cell.json(first_complete["calibration"][0])
    blob = first_cell.json(first_calibration["state"])["state"]
    first_cell.path(blob["path"]).unlink()  # Only this test's own arithmetic snapshot.
    with pytest.raises(FileNotFoundError):
        select_campaign(preparation, open_ref, campaign, ref, output, check=lambda: None)
