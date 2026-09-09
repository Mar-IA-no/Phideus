"""Real OPEN reuse through the operator, without fitting, CUDA or new draws."""
import json

import pytest

from src.atencion_armonica import generative_evidence as ge
from src.atencion_armonica import generative_evidence_supervision as supervision
from src.atencion_armonica.generative_evidence_prepared import PreparedStore
from src.atencion_armonica.generative_evidence_preparation import OpenPreparation, ProfileFits

BINDING = {"purpose": "OPEN_OPERATOR_FIXTURE_NOT_CAMPAIGN"}


@pytest.fixture(autouse=True)
def forbid_compute_and_labels(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("observable recovery must not fit or open labels")
    monkeypatch.setattr(ge.law, "fit_candidates", forbidden)
    monkeypatch.setattr(ge.law.GroupFitter, "__init__", forbidden)
    monkeypatch.setattr(supervision, "open_truths", forbidden)


def test_32_profile_fits_reused_and_second_pass_preserves_all_receipts(tmp_path):
    store = PreparedStore(tmp_path/"store", binding=BINDING)
    progress = []
    operator = OpenPreparation(store, device="cuda:0", check=lambda: None, progress=progress.append)
    shard = operator.reuse.shard("train", 0)
    before = {}
    for i in range(32):
        operator._observable(shard, i)
        before[i] = [store._record("train", i, k)[1] for k in ("fit", "observable")]
    recovered = OpenPreparation(PreparedStore(store.root, binding=BINDING), device="cuda:0",
                                check=lambda: None, progress=progress.append)
    for i in range(32):
        recovered._observable(shard, i)
        assert before[i] == [store._record("train", i, k)[1] for k in ("fit", "observable")]
    assert operator.fitter is None and recovered.fitter is None
    assert len(progress) == 64 and all(json.loads(p)["recovered_fit"] for p in progress[32:])
    assert store._record("train", 26, "observable")[0]["candidate_count"] == 0
    assert not list(store.root.rglob("supervision.json"))


def test_stop_after_fit_resumes_without_fit_or_labels(tmp_path):
    store = PreparedStore(tmp_path/"store", binding=BINDING)
    calls = []
    def stop():
        calls.append(1)
        if len(calls) == 2:
            raise InterruptedError("stop at durable fit")
    operator = OpenPreparation(store, device="cuda:0", check=stop, progress=lambda x: None)
    with pytest.raises(InterruptedError, match="durable fit"):
        operator.prepare_shard("train", 0)
    # prepare_shard checks once before loading; stop at first scene starts
    # before fitting, so no fit should have appeared yet.
    assert not list(store.root.rglob("fit.json"))
    calls.clear()
    shard = operator.reuse.shard("train", 0)
    with pytest.raises(InterruptedError):
        operator._observable(shard, 0)
    fit = store._record("train", 0, "fit")[1]
    assert not (store.root/"train/00000/observable.json").exists()
    recovered = OpenPreparation(store, device="cuda:0", check=lambda: None, progress=lambda x: None)
    recovered._observable(shard, 0)
    assert store._record("train", 0, "fit")[1] == fit
    assert (store.root/"train/00000/observable.json").is_file()


def test_no_test_or_unprofiled_fit_substitution(tmp_path):
    store = PreparedStore(tmp_path/"store", binding=BINDING)
    operator = OpenPreparation(store, device="cuda:0", check=lambda: None)
    with pytest.raises(PermissionError):
        operator.prepare_shard("iid", 0)
    with pytest.raises(ValueError, match="not fitted"):
        operator.profile.fitted("train", 32, {})
    scene = operator.reuse.shard("train", 0).scene(0)
    scene["inventory"] = {"candidates": []}
    with pytest.raises(ValueError, match="universe"):
        operator.profile.fitted("train", 0, scene)


def test_profile_source_change_rejected_before_reuse(monkeypatch):
    from src.atencion_armonica.generative_evidence_reuse import VerifiedBytes
    original = VerifiedBytes.read
    def reject(self, ref):
        if ref["path"].endswith("observable_source_rivals.py"):
            raise ValueError("source changed")
        return original(self, ref)
    monkeypatch.setattr(VerifiedBytes, "read", reject)
    with pytest.raises(ValueError, match="source changed"):
        ProfileFits()
