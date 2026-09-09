"""Mechanical profile plumbing with an explicit stub fitter; no new grid sweep."""
from copy import deepcopy
import json

import numpy as np
import pytest

from experiments.atencion_armonica.test_generative_evidence import fixture
from src.atencion_armonica import generative_evidence as ge
from src.atencion_armonica import generative_evidence_profile as profile
from src.atencion_armonica import generative_evidence_supervision as supervision
from src.atencion_armonica.generative_evidence_model import EvidenceHead, collate


def test_maximum_tensor_envelope_is_accepted_by_actual_head():
    x = profile.envelope_inputs()
    batch = collate([x]*32)
    output = EvidenceHead(ge.READER_SEEDS[0])(batch)
    assert output.shape == (32, 82, 2)
    assert batch["groups"].shape == (32, 328, 9)


@pytest.mark.parametrize("device", ["cuda", "auto", "cuda:1"])
def test_profile_never_chooses_an_implicit_device(tmp_path, device):
    with pytest.raises(ValueError, match="explicit CPU"):
        profile.run_profile(tmp_path/"invalid", device=device)


def test_busy_gpu_fails_before_loading_cuda(monkeypatch):
    def output(command, **kwargs):
        return "123, other_project, 1000 MiB" if "--query-compute-apps=pid,process_name,used_memory" in command else "NVIDIA GeForce RTX 3090, uuid, 24576 MiB, 1000 MiB"
    monkeypatch.setattr(profile.subprocess, "check_output", output)
    with pytest.raises(RuntimeError, match="another compute owner"):
        profile.gpu_availability()


def test_complete_profile_plumbing_with_stub_fits_and_no_cuda(tmp_path, monkeypatch):
    q, z, triples, residual, partitions, fits = fixture()
    class OpenFixture:
        def shard(self, split, index):
            assert (split, index) == ("train", 0)
            return self
        def scene(self, scene_id):
            return {"observation": {"scene_id": scene_id, "split_seed": 2026090880, "log_f": q.tolist()},
                    "features": {"triples": triples, "residual_cents": residual},
                    "logits": {cp: z for cp in ge.CHECKPOINTS}, "q32": q,
                    "partitions": partitions, "inventory": {"purpose": "STUB_TEST_NOT_REAL_FIT"},
                    "canonical_to_observed": np.arange(8)}
        def receipt(self):
            return {"purpose": "STUB_TEST_NOT_OPEN_REUSE"}
    monkeypatch.setattr(profile, "OpenReuse", OpenFixture)
    monkeypatch.setattr(supervision, "open_truths", lambda shard: [{"labels": np.repeat([0, 1], 4)}]*32)
    monkeypatch.setattr(ge.law, "fit_candidates", lambda q, ps, fitter: {"fits": deepcopy(fits), "group_factors": []})
    def forbidden(*a, **kw):
        raise AssertionError("profile CPU reached a CUDA or real fit path")
    monkeypatch.setattr(ge.law.GroupFitter, "fit", forbidden)
    monkeypatch.setattr(profile, "gpu_availability", forbidden)
    import torch
    monkeypatch.setattr(torch.cuda, "set_device", forbidden)
    report = profile.run_profile(tmp_path/"mechanical-profile", device="cpu")
    assert report["status"] == "MEASURED" and len(report["fit_scenes"]) == 32
    assert set(report["heads"]) == {"envelope", "observed_train"}
    assert all(len(h["all_update_seconds"]) == 25 for h in report["heads"].values())
    saved = json.loads((tmp_path/"mechanical-profile/report.json").read_bytes())
    assert saved == report
    assert saved["reuse"] == {"purpose": "STUB_TEST_NOT_OPEN_REUSE"}


def test_source_binding_pins_frozen_protocol_and_fitter():
    binding = profile.source_binding()
    assert binding["sources"][profile.PROTOCOL] == profile.PROTOCOL_SHA
    assert binding["sources"]["src/atencion_armonica/observable_source_rivals.py"] == profile.CORE_SHA
    assert binding["scene_ids"] == list(range(32))
