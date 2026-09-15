"""Full 144-head mechanical traversal; no sampler or scientific checkpoints."""
from copy import deepcopy

import numpy as np
import pytest

from experiments.atencion_armonica.test_geometric_decision_archive_admission import (
    stores, training_receipt, selection_receipt, ARMS, ROSTER, SHAPES)
from experiments.atencion_armonica.test_geometric_decision_pipeline import inputs, fitted_scene
from experiments.atencion_armonica.test_geometric_decision_scene_store import norms
from src.atencion_armonica import geometric_decision_observed_run as module
from src.atencion_armonica import geometric_decision_predictions as predictions
from src.atencion_armonica import generative_evidence as ge
from src.atencion_armonica.geometric_decision_store import ArtifactStore


@pytest.fixture
def archive(tmp_path):
    control, campaign = stores(tmp_path)
    training_receipt(control, campaign)
    selection, ref, _, campaign_ref, _ = selection_receipt(control, campaign)
    store = ArtifactStore(tmp_path/"archive", binding={"selection": ref, "selection_binding": selection.binding})
    arrays = store.publish_arrays("numeric.npz", {k: np.zeros(s, np.float32) for k, s in SHAPES.items()})
    heads = []
    for cp, arm, seed in ROSTER:
        for stage in ("initial", "selected"):
            epoch = 0 if stage == "initial" else 5
            heads.append(store.publish_json(f"heads/cp_{cp}/{arm}/seed_{seed}/{stage}.json", {
                "schema": "geometric-decision-frozen-head-v1", "binding": store.binding,
                "checkpoint_seed": cp, "arm": arm, "reader_seed": seed, "stage": stage, "epoch": epoch,
                "arrays": arrays, "source": {"campaign": campaign_ref, "root": f"cells/cp_{cp}/{arm}/seed_{seed}",
                    "calibration": {"fixture": epoch}, "state": {"fixture": epoch}}}))
    archive_ref = store.publish_json("heads.json", {"schema": "geometric-decision-head-archive-v1",
        "binding": store.binding, "selection": ref, "selected_epochs": {arm: 5 for arm in ARMS},
        "count": 144, "records": heads, "new_initializations": False, "forward": False})
    return dict(head_store=store, archive_ref=archive_ref, selection_store=selection, selection_ref=ref)


def test_complete_observable_traversal_and_recovery(inputs, archive, monkeypatch):
    store, scene, _, observations, args = inputs
    forwards, fits = [], []
    def forward(cp, records):
        forwards.append(cp["seed"])
        return [scene["logits"][cp["seed"]].copy() for _ in records]
    fitter = ge.law.GroupFitter(ge.law.Grid(5, 5), device="cpu")
    def fit(q, partitions):
        fits.append(q.copy())
        return ge.law.fit_candidates(q, partitions, fitter)
    kwargs = {**args, **archive, "forward": forward, "fit_candidates": fit, "device": "cpu",
        "normalizers": norms(), "normalization_ref": {"fixture": "unit-norms"}, "scale": 2.,
        "fit_origin": {"fixture": "5x5-not-scientific-grid"}}
    ref = module.run_observed(store, "batch", "iid", observations, **kwargs)
    record = store.json(ref)
    assert record["truth_access"] is False and record["global_seal"] is False
    assert record["roundtrip_scene_ids"] == [0]
    assert len(forwards) == 6 and len(fits) == 2  # Original/derived, shared by all heads.
    expected = archive["head_store"].json(archive["archive_ref"])["records"]
    assert [r["head"] for r in record["records"]] == expected
    assert [r["head"] for r in record["roundtrip"]["records"]] == expected
    assert all(r["transport"] is not None for r in record["records"])
    assert all(r["transport"] is None for r in record["roundtrip"]["records"])
    assert len(store.json(record["classical"])["records"]) == 1
    assert len(store.json(record["roundtrip"]["classical"])["records"]) == 1
    def forbidden(*args, **kwargs):
        raise AssertionError("complete traversal recovery must not execute models or fit")
    monkeypatch.setattr(predictions, "archived_model", forbidden)
    assert module.run_observed(store, "batch", "iid", observations,
        **{**kwargs, "forward": forbidden, "fit_candidates": forbidden}, recovery_only=True) == ref


def test_incomplete_head_roster_rejected_before_any_forward(inputs, archive):
    store, _, _, observations, args = inputs
    heads = archive["head_store"]
    value = deepcopy(heads.json(archive["archive_ref"]))
    value["records"].pop()
    bad = heads.publish_json("incomplete.json", value)
    def forbidden(*args, **kwargs):
        raise AssertionError("roster admission precedes all experimental computation")
    with pytest.raises(ValueError, match="archive"):
        module.run_observed(store, "bad", "iid", observations,
            **{**args, **archive, "archive_ref": bad, "forward": forbidden, "fit_candidates": forbidden,
               "device": "cpu", "normalizers": norms(), "normalization_ref": {}, "scale": 2., "fit_origin": {}})
    assert not store.path("bad/original/features.json").exists()
