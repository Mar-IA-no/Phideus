"""Arithmetic-only profile tests: no source corpus, sampler or CUDA calls."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.atencion_armonica.test_geometric_decision_core import interface
from experiments.atencion_armonica.test_geometric_decision_training import runtime
from src.atencion_armonica.geometric_decision_profile import (
    envelope_rows, extract_first_batch, save_batch, read_batch, head_case, fitter_workloads, fit_profile,
    assert_same_arrays,
)
from src.atencion_armonica.geometric_decision_store import ArtifactStore
from src.atencion_armonica.geometric_decision_core import READER_SEEDS
from src.atencion_armonica.geometric_decision_training import epoch_batches
from src.atencion_armonica.geometric_decision_model import GeometricDecisionHead, collate
from src.atencion_armonica.generative_evidence import CHECKPOINTS
from src.atencion_armonica import geometric_decision_profile as module


def rows():
    x, raw, _ = interface()
    return [{"scene_id": i, "identity": str(i), "inputs": deepcopy(x),
             "targets": np.array([[.7, .6], [.1, .2]], np.float32),
             "partitions": raw["partitions"], "q32": np.linspace(0, 1, 8, dtype=np.float32)} for i in range(32)]


def test_envelope_matches_full_shape_and_has_valid_eight_channel_support(runtime):
    r = envelope_rows()
    b = collate([v["inputs"] for v in r])
    assert b["groups"].shape == (32, 328, 9)
    assert b["globals"].shape == (32, 82, 17)
    assert b["evidence"].shape == (32, 82, 8)
    result = GeometricDecisionHead(READER_SEEDS[0], "geometric")(b)
    assert result.shape == (32, 82, 2)


def test_array_identity_rejects_signed_zero_and_dtype_changes():
    positive = {"targets": np.array([0.], np.float32)}
    for other in ({"targets": np.array([-0.], np.float32)}, {"targets": np.array([0.], np.float64)}):
        with pytest.raises(ValueError, match="bytes differ"):
            assert_same_arrays(positive, other)
    assert len(assert_same_arrays(positive, positive)) == 64


@pytest.mark.parametrize("objective", ["mse", "decision"])
def test_head_uses_real_kernel_and_exact_recovery(tmp_path, runtime, objective):
    store = ArtifactStore(tmp_path/objective, binding={"fixture": "arithmetic-mechanical-profile", "objective": objective})
    ref = head_case(store, rows(), objective, "cpu", check=lambda: None)
    result = store.json(ref)
    assert len(result["all_update_seconds"]) == 25
    assert len(result["snapshot_io_seconds"]) == 2
    assert len(result["evaluation_batch_io_seconds"]) == 3
    assert store.load_state(result["last"])["steps"] == 25
    assert store.load_state(result["middle"])["steps"] == 10
    assert result["exact_recovery_digest"] == store.json(result["last"])["state_digest"]
    assert result["status"] == "MECHANICAL_PROFILE_NOT_TRAINED_CELL"


def test_batch_roundtrip_preserves_arrays_and_candidate_order(tmp_path):
    r = rows()
    store = ArtifactStore(tmp_path/"subset", binding={"fixture": "profile-subset"})
    provenance = {"scene_ids": list(range(32)), "fixture": "not-real-train"}
    ref = save_batch(store, r, provenance)
    restored, actual = read_batch(store, ref)
    assert actual == provenance
    for a, b in zip(r, restored):
        for key in a["inputs"]:
            np.testing.assert_array_equal(a["inputs"][key], b["inputs"][key])
        np.testing.assert_array_equal(a["targets"], b["targets"])
        np.testing.assert_array_equal(a["q32"], b["q32"])
        assert a["partitions"] == b["partitions"]
    record = store.json(ref)
    record["rows"][0]["partitions"].reverse()
    wrong = store.publish_json("wrong-order.json", record)
    with pytest.raises(ValueError, match="candidate order"):
        read_batch(store, wrong)


def test_fitter_workloads_ignore_targets_and_are_paired_by_shape():
    r = rows()
    expected = fitter_workloads(r)
    assert len(expected) == 30
    for row in r:
        row["targets"] = object()
        row["inputs"] = object()
    assert fitter_workloads(r) == expected
    for item in expected:
        assert len(item["values"]) == len(item["origins"]) <= 8
        assert all(len(v) == item["size"] for v in item["values"])
        assert all(np.all(np.diff(v) >= 0) for v in item["values"])
        if item["kind"] == "arithmetic":
            assert len(item["values"]) == 8


def test_fitter_operator_preserves_full_factors_and_skips_absent_shapes(tmp_path, monkeypatch):
    calls = []
    class Fitter:
        def __init__(self, grid, device, assignment_batch):
            assert (grid.beta_count, grid.gamma_count, grid.stride) == (257, 65, 4)
            assert device == "cpu" and assignment_batch == 8
        def fit(self, values, branch):
            calls.append((values, branch))
            return [{"full_factor_fixture": v, "branch": branch} for v in values]
    monkeypatch.setattr(module.law, "GroupFitter", Fitter)
    workloads = fitter_workloads(rows())
    store = ArtifactStore(tmp_path/"fits", binding={"fixture": "fitter-profile"})
    result = store.json(fit_profile(store, workloads, "cpu", check=lambda: None))
    assert len(result["results"]) == 30
    assert len(calls) == sum(bool(w["values"]) for w in workloads)
    for value in result["results"]:
        if value["status"] == "MEASURED":
            saved = store.json(value["factors"])
            assert len(saved["factors"]) == value["groups"]


def test_extract_uses_scheduled_ids_train_only_and_existing_scale():
    template = rows()[0]
    calls, verified = [], []
    eligible = [i for i in range(4096) if i % 17]
    cp = CHECKPOINTS[0]
    class Source:
        corpus = SimpleNamespace(norm={"eligible_scene_ids": eligible})
        def observable_shard(self, split, shard, *, check):
            assert split == "train"
            calls.append((split, shard))
            return {"scene_ids": list(range(shard*512, (shard+1)*512)), "identities": [str(i) for i in range(shard*512, (shard+1)*512)],
                "observations": [{"log_f": template["q32"].tolist()}]*512, "raw": {cp: [template]*512},
                "targets_ref": {"fixture_target": shard}}
        def interface_shard(self, obs, scale):
            assert scale == {"scale": "sealed"}
            return {cp: [{"inputs": template["inputs"]}]*512}
        def supervision_shard(self, obs):
            return [{"targets": template["targets"]}]*512
    class Preparation:
        source = Source()
        store = SimpleNamespace(binding={"fixture": "pinned"}, json=lambda ref: {"scale": "sealed"})
        def completion(self, ref):
            return {"scale": {"fixture_scale": 1}, "entries": [
                {"split": "train", "checkpoint_seed": cp, "shard": s, "index": {"fixture_index": s}} for s in range(8)]}
        def _identity(self, *args):
            return "authenticated-identity"
        def _arrays(self, rows):
            return "authenticated-arrays"
        def _verify_entry(self, ref, identity, arrays):
            assert identity == "authenticated-identity" and arrays == "authenticated-arrays"
            verified.append(ref)
    r, provenance = extract_first_batch(Preparation(), {"fixture_complete": 1}, check=lambda: None)
    expected = epoch_batches(READER_SEEDS[0], 0, eligible)[0].tolist()
    assert [row["scene_id"] for row in r] == provenance["scene_ids"] == expected
    assert len(verified) == len(calls) == len({i//512 for i in expected})
    assert all(row["scene_id"] % 17 for row in r)


def test_operator_charges_fixture_preparation_and_refuses_repetition(tmp_path, monkeypatch, runtime):
    import time
    from experiments.atencion_armonica import profile_geometric_decision as op
    from src.atencion_armonica.geometric_decision_budget import LIMITS
    control = ArtifactStore(tmp_path/"control", binding={"limits": LIMITS, "prior_charges": [],
                           "output_roots": [str(tmp_path)]})
    r = rows()
    monkeypatch.setattr(op, "BASES", (tmp_path,))
    monkeypatch.setattr(op, "LAUNCH_STARTED", time.monotonic())
    data = SimpleNamespace(binding={"fixture": "loaded"}, eligible={"train": list(range(32)), "calibration": list(range(32))}, rows={"train": r})
    preparation = SimpleNamespace(load_checkpoint=lambda *args, **kwargs: data)
    monkeypatch.setattr(op, "admitted_open", lambda: (control, preparation, {"fixture": "complete"}))
    monkeypatch.setattr(op, "sources", lambda: [{"fixture": "code"}])
    monkeypatch.setattr(op, "reference", lambda path: {"sha256": op.PROTOCOL_SHA})
    monkeypatch.setattr(op.profile, "extract_first_batch", lambda *args, **kwargs: (r, {"scene_ids": list(range(32))}))
    monkeypatch.setattr("sys.argv", ["profile", "--task", "inputs", "--device", "cpu"])
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        monkeypatch.setenv(key, "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    op.main()
    finish = control.json(control.reference(control.path("attempts/0000/finish.json")))
    assert finish["status"] == "COMPLETE" and finish["charged_after"]["profile"] > 0
    restored, provenance = op.admitted_profile_inputs(control)
    assert len(restored) == 32 and provenance["finish"]["path"] == "attempts/0000/finish.json"
    with pytest.raises(ValueError, match="no silent repetition"):
        op.main()
    assert not control.path("attempts/0001/start.json").exists()
    monkeypatch.setattr("sys.argv", ["profile", "--task", "head", "--device", "cpu"])
    monkeypatch.setattr(op, "LAUNCH_STARTED", time.monotonic())
    def fake_case(store, rows, objective, device, *, check):
        check()
        assert store.binding["input_provenance"]["finish"]["path"] == "attempts/0000/finish.json"
        return store.publish_json("result.json", {"fixture": "not-executed-head"})
    monkeypatch.setattr(op.profile, "head_case", fake_case)
    op.main()
    start = control.json(control.reference(control.path("attempts/0001/start.json")))
    manifest = control.json(start["manifest"])
    binding = manifest["binding"]
    assert binding["input_provenance"]["finish"]["path"] == "attempts/0000/finish.json"
    assert binding["input_provenance"]["checkpoint_load"]["data_binding"] == {"fixture": "loaded"}
    assert binding["runtime"]["cuda_visible_devices"] == ""
    assert "cublas_workspace_config" in binding["runtime"] and "cuda_build" in binding["runtime"]
