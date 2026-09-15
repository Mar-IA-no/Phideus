"""Evaluation on known mechanical observations, not prospective scientific data."""
from copy import deepcopy

import numpy as np
import pytest

from experiments.atencion_armonica.test_geometric_decision_observed_run import archive
from experiments.atencion_armonica.test_geometric_decision_pipeline import inputs, fitted_scene
from experiments.atencion_armonica.test_geometric_decision_scene_store import norms
from experiments.atencion_armonica.test_geometric_decision_metrics import PARTITIONS, LABELS
from src.atencion_armonica import geometric_decision_evaluation as module
from src.atencion_armonica import geometric_decision_predictions as predictions
from src.atencion_armonica import geometric_decision_metrics as metrics
from src.atencion_armonica.geometric_decision_observed_run import run_observed
from src.atencion_armonica.geometric_decision_store import ArtifactStore
from src.atencion_armonica import generative_evidence as ge


def test_144_head_evaluation_and_exact_replay_without_models(inputs, archive, monkeypatch):
    store, scene, _, observations, args = inputs
    fitter = ge.law.GroupFitter(ge.law.Grid(5, 5), device="cpu")
    kwargs = {**args, **archive, "device": "cpu", "normalizers": norms(),
        "normalization_ref": {"fixture": "unit"}, "scale": 2., "fit_origin": {"fixture": "5x5"},
        "forward": lambda cp, records: [scene["logits"][cp["seed"]].copy() for _ in records],
        "fit_candidates": lambda q, partitions: ge.law.fit_candidates(q, partitions, fitter)}
    ref = run_observed(store, "batch", "iid", observations, **kwargs)
    batch = store.json(ref)
    output = ArtifactStore(store.root.parent/"evaluation", binding={"fixture": "explicit-OPEN-labels"})
    def forbidden(*args, **kwargs):
        raise AssertionError("evaluation cannot run a model or fitter")
    monkeypatch.setattr(predictions, "archived_model", forbidden)
    monkeypatch.setattr(ge.law, "fit_candidates", forbidden)
    labels = {0: np.repeat([0, 1], 4).astype(np.int64)}
    result, learned, eligible = module.evaluate_batch(store, batch, labels, output, "original", check=lambda: None)
    assert learned.shape == (1, 2, 8, 3, 3, len(module.NAMES))
    assert eligible.tolist() == [True]
    saved = output.json(result)
    assert len(saved["heads"]) == 144
    replay, again, mask = module.evaluate_batch(store, batch, labels, output, "original", check=lambda: None, replay=True)
    assert replay == result
    assert learned.tobytes() == again.tobytes() and np.array_equal(mask, eligible)
    original = store.json(store.json(batch["sources"])["sources"][0])["scene"]
    probe = store.json(store.json(batch["roundtrip"]["sources"])["sources"][0])["scene"]
    moved = {0: module.labels_by_event(original, probe, labels[0])}
    derived, _, _ = module.evaluate_batch(store, batch["roundtrip"], moved, output, "probe", check=lambda: None)
    assert module.evaluate_batch(store, batch["roundtrip"], moved, output,
        "probe", check=lambda: None, replay=True)[0] == derived
    bad = deepcopy(batch)
    bad["records"] = bad["records"][:-1]
    with pytest.raises(ValueError, match="every initial"):
        module.evaluate_batch(store, bad, labels, output, "bad", check=lambda: None)
    with pytest.raises(ValueError, match="cannot repair"):
        module.evaluate_batch(store, batch, labels, output, "missing", check=lambda: None, replay=True)


def test_rank_mapping_uses_event_identity_not_frequency_match():
    original = {"canonical_to_observed": [2, 0, 3, 1]}
    probe = {"canonical_to_observed": [0, 2, 1, 3]}
    labels = np.array([7, 8, 9, 10], np.int64)
    np.testing.assert_array_equal(module.labels_by_event(original, probe, labels), [8, 7, 10, 9])
    with pytest.raises(ValueError, match="correspondence"):
        module.labels_by_event(original, {"canonical_to_observed": [0, 0, 1, 3]}, labels)


def test_strata_are_conditioned_not_replacement_primary():
    target = metrics.targets(PARTITIONS, LABELS)
    h = np.array([[2., 2.], [-1., -1.]])
    energy = h.sum(axis=1)
    result = module.strata_metrics(energy, target, {"full": {"all": [0, 1]}, "k": {"both": [0, 1], "only_first": [0]}}, h)
    assert "full" not in result
    full = metrics.describe(PARTITIONS, target, energy, components=h)
    assert result["k"]["both"]["metrics"]["regret_tM"] == full["decision"]["regret_tM"]
    assert result["k"]["both"]["metrics"]["regret_tD"] == full["decision"]["regret_tD"]
    assert result["k"]["only_first"]["metrics"]["regret_tM"] == 0.
    assert result["k"]["only_first"]["metrics"]["tau_b"] is None


def test_summary_preserves_scene_first_gaps_and_undefined_support():
    learned = np.zeros((2, 2, 8, 3, 3, len(module.NAMES)))
    learned[0, 1, 0, 0, 0, 0] = 9.
    learned[1, 1, 0, :, :, 0] = 3.
    learned[..., -1] = np.nan
    classical = np.zeros((2, 4, len(module.NAMES)))
    result = module.summarize(learned, classical, np.ones(2, bool))
    assert result["selected_minus_initial"]["mean"][0][0] == 2.
    assert result["arm_scene_first"]["mean"][1][0][-1] is None
    assert result["arm_scene_first"]["defined_scenes"][1][0][-1] == 0
    empty = module.summarize(learned[:0], classical[:0], np.zeros(0, bool))
    assert empty["total_scenes"] == 0 and empty["arm_scene_first"]["mean"][1][0][0] is None
