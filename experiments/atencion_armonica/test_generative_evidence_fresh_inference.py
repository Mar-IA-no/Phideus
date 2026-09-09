"""CPU mechanics for fresh inference; no draw, fit, real model forward or CUDA."""
from copy import deepcopy
import os

import numpy as np
import pytest
import torch

from src.atencion_armonica import generative_evidence as ge
from src.atencion_armonica import generative_evidence_fresh_inference as fresh
from src.atencion_armonica import generative_evidence_fresh_store as fresh_store


TEST_ROOT = fresh.ROOT/".agent-work/phideus-r726-fresh-inference-tests-20260909"/f"run-{os.getpid()}"
FREEZE_REF = {"path": "fixture/test-freeze.json", "sha256": "f"*64}
BINDING = {"test_freeze": FREEZE_REF}


def _store(monkeypatch, name):
    root = TEST_ROOT/name
    root.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(fresh_store, "TEMPORARY", TEST_ROOT.parent)
    return fresh.FreshObservableStore(root, binding=BINDING)


def _cells():
    return [{"arm": arm, "checkpoint_seed": cp, "reader_seed": seed,
             "cell_id": f"{arm}-cp{cp}-seed{seed}"}
            for arm in ge.ARMS for cp in ge.CHECKPOINTS for seed in ge.READER_SEEDS]


def frozen_fixture():
    states = [{"cell": cell, "epoch": 5,
               "complete": {"path": cell["cell_id"]+"/complete.json", "sha256": "a"*64},
               "state": {"path": cell["cell_id"]+"/state.json", "sha256": "b"*64}}
              for cell in _cells()]
    roster = [{"kind": "original", **cell} for cell in _cells()]
    roster += [{"kind": "intervention", **cell, "intervention": intervention}
               for cell in _cells() if cell["arm"] == "generative"
               for intervention in ("zero", "decoupled")]
    return {"freeze": FREEZE_REF, "manifest": {"selected_states": states,
        "prediction_roster": roster, "normalizers": {"path": "n", "sha256": "c"*64},
        "runtime": {"python": "x", "numpy": "x", "torch": "x"}}, "exclusions": {}}


def rows_fixture():
    groups = np.zeros((1, 9), np.float32)
    globals_ = np.zeros((1, 17), np.float32)
    incidence = np.ones((1, 1), np.float32)
    def row(evidence):
        return {"groups": groups.copy(), "globals": globals_.copy(),
                "incidence": incidence.copy(), "evidence": np.full((1, 6), evidence, np.float32)}
    return {arm: [row({"local": 0., "generative": 1., "decoupled": 2.}[arm]) for _ in range(512)]
            for arm in ge.ARMS}


def test_roster_maps_freeze_and_kernel_orders_bijectively():
    roster, selected = fresh._roster(frozen_fixture())
    assert len(roster) == 45 and len(selected) == 27
    assert [r["inference_index"] for r in roster] == list(range(45))
    assert sorted(r["freeze_index"] for r in roster) == list(range(45))
    assert sum(r["intervention"] == "original" for r in roster) == 27
    assert sum(r["intervention"] != "original" for r in roster) == 18
    assert all(r["arm"] == "generative" for r in roster if r["intervention"] != "original")
    bad = frozen_fixture()
    bad["manifest"]["prediction_roster"].pop()
    with pytest.raises(ValueError, match="45"):
        fresh._roster(bad)


def test_intervention_mapping_preserves_each_original_arm_and_only_changes_evidence():
    delivered = {"inputs": rows_fixture()}
    for arm in ge.ARMS:
        rows = fresh._intervention_rows(delivered, {"arm": arm, "intervention": "original"})
        assert rows is delivered["inputs"][arm]
    assert fresh._intervention_rows(delivered, {"arm": "generative", "intervention": "zero"}) \
           is delivered["inputs"]["local"]
    assert fresh._intervention_rows(delivered, {"arm": "generative", "intervention": "decoupled"}) \
           is delivered["inputs"]["decoupled"]
    with pytest.raises(ValueError, match="only the Generativa"):
        fresh._intervention_rows(delivered, {"arm": "local", "intervention": "zero"})


def test_ragged_logit_archive_binds_ordered_observations_and_sizes():
    observations = [{"scene_id": i, "split_seed": 9, "log_f": np.arange(n, dtype=np.float32).tolist()}
                    for i, n in enumerate((8, 10))]
    matrices = [np.eye(8, dtype=np.float32), np.eye(10, dtype=np.float32)]
    arrays = fresh._logit_arrays(matrices, observations)
    assert arrays["sizes"].tolist() == [8, 10]
    assert arrays["offsets"].tolist() == [0, 64, 164]
    with pytest.raises(ValueError, match="complete observable roster"):
        fresh._logit_arrays([np.eye(9, dtype=np.float32), matrices[1]], observations)
    asymmetric = matrices[0].copy()
    asymmetric[0, 1] = 1
    with pytest.raises(ValueError, match="backbone logits"):
        fresh._logit_arrays([asymmetric, matrices[1]], observations)


def test_source_scene_uses_exact_three_checkpoint_pool_union_on_known_q8():
    observation = {"scene_id": 0, "split_seed": fresh.cache.SPLITS["iid"][1],
                   "log_f": (np.arange(8, dtype=np.float32)/8).astype(np.float64).tolist()}
    feature = fresh.feature_record(observation)
    matrices = {cp: [np.zeros((8, 8), np.float32)] for cp in ge.CHECKPOINTS}
    scene = fresh._scene_from_sources("iid", 0, observation, feature, matrices)
    assert list(scene["pools"]) == [str(cp) for cp in ge.CHECKPOINTS]
    assert scene["partitions"] == sorted(set(scene["partitions"]))
    assert all(ge.law.supported(p) for p in scene["partitions"])
    assert scene["canonical_to_observed"].tolist() == list(range(8))


def test_fixed_payloads_are_idempotent_and_never_replaced(monkeypatch):
    store = _store(monkeypatch, "fixed")
    arrays = {"x": np.arange(8, dtype=np.float32)}
    ref = fresh._fixed_arrays(store, "iid/mechanical.npz", arrays)
    assert fresh._fixed_arrays(store, "iid/mechanical.npz", arrays) == ref
    with pytest.raises(ValueError, match="cannot replace"):
        fresh._fixed_arrays(store, "iid/mechanical.npz", {"x": arrays["x"]+1})
    ref = fresh._fixed_json(store, "iid/mechanical.json", {"status": "COMPLETE"})
    assert fresh._fixed_json(store, "iid/mechanical.json", {"status": "COMPLETE"}) == ref
    with pytest.raises(ValueError, match="cannot replace"):
        fresh._fixed_json(store, "iid/mechanical.json", {"status": "CHANGED"})


class FakeModel:
    def __init__(self):
        self.weight = torch.zeros(1)

    def parameters(self):
        yield self.weight


def test_prediction_receipt_reuses_complete_payload_and_rejects_ambiguity(monkeypatch):
    store = _store(monkeypatch, "prediction")
    delivered = {"inputs": rows_fixture()}
    row = fresh._roster(frozen_fixture())[0][0]
    delivered_ref = {"path": "fixture/delivered.json", "sha256": "d"*64}
    calls = []
    def fake_predict(model, rows, *, device, check):
        calls.append((device, len(rows)))
        return {"components": np.ones((512, 2), np.float32),
                "offsets": np.arange(513, dtype=np.int64)}
    monkeypatch.setattr(fresh.inference, "predict", fake_predict)
    ref, record = fresh._prediction_record(store, "iid", row, delivered_ref,
                                            delivered, FakeModel(), lambda: None)
    assert calls == [("cpu", 512)] and record["prediction"]["path"].endswith(".npz")
    again, reopened = fresh._prediction_record(store, "iid", row, delivered_ref,
                                               delivered, None, lambda: None)
    assert again == ref and reopened == record and len(calls) == 1
    another = deepcopy(row)
    another["reader_seed"] = ge.READER_SEEDS[1]
    payload, _, _ = fresh._prediction_paths(store, "iid", another)
    payload.parent.mkdir(parents=True, exist_ok=True)
    fresh.storage.write_arrays(payload, {"components": np.ones((512, 2), np.float32),
                                         "offsets": np.arange(513, dtype=np.int64)})
    with pytest.raises(RuntimeError, match="ambiguous"):
        fresh._prediction_record(store, "iid", another, delivered_ref,
                                 delivered, FakeModel(), lambda: None)


def test_real_checkpoint_kernel_refuses_cpu_before_importing_forward():
    with pytest.raises(ValueError, match="explicit cuda:0"):
        fresh._checkpoint_forward({}, [], {}, "cpu")


def test_package_runtime_is_not_accepted_as_cuda_runtime():
    package = {"python": "3", "numpy": "2", "torch": "2", "scipy": "1",
               "scikit-learn": "1", "torch_distribution": "cpu"}
    with pytest.raises(ValueError, match="CUDA runtime differs"):
        fresh._validated_gpu_runtime(package, package)
    runtime = {"torch": "2", "numpy": "2", "cuda": "12.8", "cudnn": 90701,
               "device": "NVIDIA GeForce RTX 3090"}
    assert fresh._validated_gpu_runtime(runtime, package) is runtime


def test_preseal_failure_never_publishes_seal_and_success_verifies_first(monkeypatch):
    store = _store(monkeypatch, "preseal")
    index_value = {"status": fresh.STATUS}
    fresh._fixed_json(store, "iid/predictions.json", index_value)
    events, fail = [], {"active": True}
    original_fixed = fresh._fixed_json

    def fake_verify(split, *, freeze_ref, check, root, require_seal=True):
        events.append("verify-seal" if require_seal else "verify-preseal")
        if fail["active"]:
            raise ValueError("synthetic preseal rejection")
        if not require_seal:
            return {"seal": None, "index": index_value, "choices": {}, "records": []}
        return {"seal": fresh._root_ref(store.path("iid/prediction_seal.json")),
                "index": index_value, "choices": {}, "records": []}

    def tracked_fixed(target, relative, value):
        if relative == "iid/prediction_seal.json":
            events.append("write-seal")
        return original_fixed(target, relative, value)

    monkeypatch.setattr(fresh, "_verify", fake_verify)
    monkeypatch.setattr(fresh, "_fixed_json", tracked_fixed)
    with pytest.raises(ValueError, match="synthetic preseal rejection"):
        fresh._seal_verified_index(store, "iid", index_value, frozen_fixture(), lambda: None, store.root)
    assert events == ["verify-preseal"]
    assert not store.path("iid/prediction_seal.json").exists()

    events.clear()
    fail["active"] = False
    result = fresh._seal_verified_index(
        store, "iid", index_value, frozen_fixture(), lambda: None, store.root)
    assert events == ["verify-preseal", "write-seal", "verify-seal"]
    assert result == fresh._root_ref(store.path("iid/prediction_seal.json"))


def test_verifier_does_not_create_a_missing_output_store(monkeypatch):
    root = TEST_ROOT/"missing-verification-root"
    monkeypatch.setattr(fresh, "_freeze_authority", lambda ref, check: frozen_fixture())
    with pytest.raises(FileNotFoundError, match="does not exist"):
        fresh._verify("iid", freeze_ref=FREEZE_REF, check=lambda: None, root=root)
    assert not root.exists()


def test_read_prediction_requires_verified_seal_membership(monkeypatch):
    store = _store(monkeypatch, "read")
    local = fresh._fixed_arrays(store, "iid/pred.npz",
                                {"components": np.ones((1, 2), np.float32),
                                 "offsets": np.array([0, 1], np.int64)})
    record = {"prediction": fresh._root_artifact(store, local)}
    index = {"status": fresh.STATUS}
    index_local = fresh._fixed_json(store, "iid/read-index.json", index)
    index_ref = fresh._root_ref(store.path(index_local["path"]))
    seal = {"status": fresh.SEAL_STATUS, "prediction_index": index_ref}
    seal_local = fresh._fixed_json(store, "iid/read-seal.json", seal)
    seal_ref = fresh._root_ref(store.path(seal_local["path"]))
    verified = {"seal": seal_ref, "index": index, "choices": {}, "records": [record]}
    loaded = fresh.read_prediction(verified, record)
    assert loaded["components"].shape == (1, 2)
    changed = deepcopy(record)
    changed["prediction"]["sha256"] = "0"*64
    with pytest.raises(ValueError, match="verified complete seal"):
        fresh.read_prediction(verified, changed)
    forged = deepcopy(verified)
    forged["seal"] = {"path": seal_ref["path"], "sha256": "0"*64}
    with pytest.raises(ValueError, match="authenticated complete seal"):
        fresh.read_prediction(forged, record)
