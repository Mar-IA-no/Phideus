"""Recoverable source/fitting fixtures with no real forward or sampler."""
from copy import deepcopy

import pytest
import numpy as np

from experiments.atencion_armonica.test_geometric_decision_scene_store import fitted_scene, norms
from src.atencion_armonica import geometric_decision_pipeline as module
from src.atencion_armonica.geometric_decision_store import ArtifactStore


@pytest.fixture
def inputs(tmp_path, fitted_scene):
    scene, fitted = fitted_scene
    store = ArtifactStore(tmp_path/"pipeline", binding={"fixture": "q8-recoverable-pipeline"})
    observations = [scene["observation"]]
    origin = {"kind": "original", "records": [store.publish_json("observation.json", observations[0])]}
    checkpoints = [{"seed": cp, "checkpoint": {"fixture": cp}} for cp in module.ge.CHECKPOINTS]
    arguments = {"expected_seed": 11, "observation_origin": origin, "checkpoints": checkpoints,
                 "runtime": {"fixture": "no-model-or-GPU"}, "check": lambda: None}
    return store, scene, fitted, observations, arguments


def test_source_and_fit_resume_without_forward_or_fitting(inputs):
    store, scene, fitted, observations, args = inputs
    forwards, fits = [], []
    def forward(cp, records):
        forwards.append(cp["seed"])
        assert len(records) == 1
        return [scene["logits"][cp["seed"]].copy()]
    source = module.prepare_sources(store, "original", "iid", observations, forward=forward, **args)
    assert forwards == list(module.ge.CHECKPOINTS)
    def fit(q, partitions):
        fits.append(1)
        assert partitions == scene["partitions"]
        return deepcopy(fitted)
    kwargs = {"normalizers": norms(), "normalization_ref": {"fixture": "unit-norms"}, "scale": 2.,
              "fit_origin": {"fixture": "5x5-cached-fit"}, "check": lambda: None}
    result = module.prepare_inputs(store, "original", source, fit_candidates=fit, **kwargs)
    assert len(fits) == 1
    def forbidden(*args):
        raise AssertionError("completed computation must be reused")
    assert module.prepare_sources(store, "original", "iid", observations, forward=forbidden, **args) == source
    assert module.prepare_inputs(store, "original", source, fit_candidates=forbidden, **kwargs) == result
    with pytest.raises(ValueError, match="runtime"):
        module.prepare_inputs(store, "original", source, fit_candidates=forbidden,
            **{**kwargs, "fit_origin": {"fixture": "another-runtime"}})


def test_resume_preserves_first_backbone_after_later_forward_interrupt(inputs):
    store, scene, _, observations, args = inputs
    calls = []
    def interrupted(cp, records):
        calls.append(cp["seed"])
        if len(calls) == 2:
            raise InterruptedError("fixture interruption")
        return [scene["logits"][cp["seed"]].copy()]
    with pytest.raises(InterruptedError):
        module.prepare_sources(store, "original", "iid", observations, forward=interrupted, **args)
    resumed = []
    def forward(cp, records):
        resumed.append(cp["seed"])
        return [scene["logits"][cp["seed"]].copy()]
    module.prepare_sources(store, "original", "iid", observations, forward=forward, **args)
    assert resumed == list(module.ge.CHECKPOINTS[1:])


def test_orphan_logits_stop_before_forward(inputs):
    store, scene, _, observations, args = inputs
    cp = module.ge.CHECKPOINTS[0]
    store.publish_arrays(f"original/logits-{cp}.npz", module.pack_logits([scene["logits"][cp]], observations))
    def forbidden(*args):
        raise AssertionError("orphan logits must stop before forward")
    with pytest.raises(RuntimeError, match="reconciliation"):
        module.prepare_sources(store, "original", "iid", observations, forward=forbidden, **args)


def test_wrong_original_receipt_stops_before_forward(inputs):
    store, _, _, observations, args = inputs
    bad_observation = {**observations[0], "scene_id": 1}
    bad_ref = store.publish_json("another-observation.json", bad_observation)
    def forbidden(*args):
        raise AssertionError("wrong identity must stop before forward")
    with pytest.raises(ValueError, match="authenticated observation"):
        module.prepare_sources(store, "original", "iid", observations, forward=forbidden,
            **{**args, "observation_origin": {"kind": "original", "records": [bad_ref]}})


def test_roundtrip_pipeline_recomputes_observables_and_keeps_original_event_parent(inputs):
    from src.atencion_armonica.geometric_decision_observables import roundtrip_observation
    store, scene, _, observations, args = inputs
    seen_tokens = []
    def forward(cp, records):
        seen_tokens.append(records[0]["tokens"].copy())
        return [scene["logits"][cp["seed"]].copy()]
    source = module.prepare_sources(store, "original", "iid", observations, forward=forward, **args)
    parent = store.json(source)["sources"][0]
    transformed = roundtrip_observation(observations[0], expected_seed=11)
    probe_ref = module.prepare_sources(store, "probe", "iid", [transformed["observation"]], forward=forward,
        **{**args, "observation_origin": {"kind": "roundtrip", "parents": [parent]}})
    probe_source = store.json(probe_ref)["sources"][0]
    record = store.json(probe_source)
    assert record["derivation"]["parent"] == parent
    assert record["derivation"]["lineage"]["event_ids"] == list(range(8))
    assert len(seen_tokens) == 6
    assert not (seen_tokens[0] == seen_tokens[-1]).all()
    fitter = module.ge.law.GroupFitter(module.ge.law.Grid(5, 5), device="cpu")
    result = module.prepare_inputs(store, "probe", probe_ref, normalizers=norms(),
        normalization_ref={"fixture": "unit-norms"}, scale=2., fit_origin={"fixture": "CPU-5x5"},
        fit_candidates=lambda q, ps: module.ge.law.fit_candidates(q, ps, fitter), check=lambda: None)
    assert store.json(result)["scene_ids"] == [0]


def test_noncanonical_integer_observation_rejected_before_any_forward(inputs):
    store, _, _, observations, args = inputs
    obs = {**observations[0], "log_f": list(range(8))}
    origin = {"kind": "original", "records": [store.publish_json("integer-observation.json", obs)]}
    calls = []
    def forward(cp, records):
        calls.append(cp["seed"])
        return [np.zeros((8, 8), np.float32)]
    with pytest.raises(ValueError, match="canonical JSON"):
        module.prepare_sources(store, "integer", "iid", [obs], forward=forward,
            **{**args, "observation_origin": origin})
    assert calls == []
    assert not store.path("integer").exists()


def test_mutating_checkpoint_callback_cannot_change_receipt_or_caller_identity(inputs):
    store, scene, _, observations, args = inputs
    before = deepcopy(args)
    def mutated(cp, records):
        cp["checkpoint"]["fixture"] = "changed"
        return [scene["logits"][cp["seed"]].copy()]
    with pytest.raises(ValueError, match="mutated admitted checkpoint"):
        module.prepare_sources(store, "mutation", "iid", observations, forward=mutated, **args)
    assert args == before
    cp = module.ge.CHECKPOINTS[0]
    assert not store.path(f"mutation/logits-{cp}.npz").exists()
    assert not store.path(f"mutation/logits-{cp}.json").exists()
    def valid(cp, records):
        return [scene["logits"][cp["seed"]].copy()]
    ref = module.prepare_sources(store, "mutation", "iid", observations, forward=valid, **args)
    assert module.prepare_sources(store, "mutation", "iid", observations, forward=valid, **args) == ref
