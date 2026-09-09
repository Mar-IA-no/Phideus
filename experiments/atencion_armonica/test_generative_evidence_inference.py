"""Full-count arithmetic inference fixture, not fresh test access."""
from copy import deepcopy

import numpy as np
import pytest
import torch

from experiments.atencion_armonica.test_generative_evidence import fixture, norms
from src.atencion_armonica import generative_evidence as ge
from src.atencion_armonica import generative_evidence_inference as inf
from src.atencion_armonica.generative_evidence_model import EvidenceHead, collate


def delivered():
    f = fixture()
    row, empty = ge.observable_rows(*f), ge.observable_rows(*f[:4], [], [])
    common, evidence = norms(row)
    return {a: [ge.model_inputs(row if i < 35 else empty, common, evidence, a,
                    split_seed=2026090982, scene_id=i) for i in range(512)] for a in ge.ARMS}


def test_exact_45_roster_and_18_interventions():
    rows = inf.inference_roster()
    assert len(rows) == 45 and len({tuple(r.values()) for r in rows}) == 45
    assert sum(r["intervention"] == "original" for r in rows) == 27
    assert all(r["arm"] == "generative" for r in rows if r["intervention"] != "original")


def test_partial_batch_preserves_empty_roster_and_model():
    torch.set_num_threads(1)
    rows = delivered()["generative"]
    model = EvidenceHead(ge.READER_SEEDS[0])
    before = deepcopy(model.state_dict())
    calls = []
    result = inf.predict(model, rows, device="cpu", check=lambda: calls.append(True))
    assert result["components"].shape == (70, 2) and result["offsets"].shape == (513,)
    assert np.all(result["offsets"][35:] == 70) and len(calls) == 4
    for start, stop in ((0, 32), (32, 35)):
        with torch.no_grad():
            expected = model(collate(rows[start:stop])).numpy().reshape(-1, 2)
        np.testing.assert_array_equal(result["components"][start*2:stop*2], expected)
    for key, value in model.state_dict().items():
        assert torch.equal(value, before[key])


def test_interventions_only_change_channel_and_replay_dependence():
    values = delivered()
    for intervention, arm in (("original", "generative"), ("zero", "local"), ("decoupled", "decoupled")):
        assert inf.intervention_inputs(values, intervention) is values[arm]
    wrong = deepcopy(values)
    wrong["local"][0]["globals"][0, 0] += 1
    with pytest.raises(ValueError):
        inf.intervention_inputs(wrong, "zero")
    p = fixture()[4]
    original = np.array([[1., 1.], [2., 2.]], np.float32)
    changed = original+np.float32(1.)
    report = inf.dependence(original, changed, p)["dependence"]
    assert report["constant_sum_delta"] and not report["decision_changed"]
    assert inf.dependence(np.empty((0, 2), np.float32), np.empty((0, 2), np.float32), [])["dependence"] is None


def test_no_output_no_forward_and_early_stop():
    rows = delivered()["local"]
    empty = [rows[-1]]*512
    model = EvidenceHead(ge.READER_SEEDS[0])
    def forbidden(*args, **kwargs):
        raise AssertionError("no forward for absent candidates")
    handle = model.register_forward_pre_hook(forbidden)
    result = inf.predict(model, empty, device="cpu", check=lambda: None)
    assert result["components"].shape == (0, 2)
    def stop():
        raise InterruptedError("stop before inference")
    with pytest.raises(InterruptedError):
        inf.predict(model, rows, device="cpu", check=stop)
    handle.remove()
    with pytest.raises(ValueError):
        inf.predict(model, rows[:-1], device="cpu", check=lambda: None)
    contaminated = rows.copy()
    contaminated[0] = {**rows[0], "targets": np.zeros((2, 2), np.float32)}
    with pytest.raises(ValueError):
        inf.predict(model, contaminated, device="cpu", check=lambda: None)
