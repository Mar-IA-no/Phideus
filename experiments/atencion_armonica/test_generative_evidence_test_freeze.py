"""Mechanical freeze tests; no fresh draw, model, forward, CUDA or truth port."""
from copy import deepcopy
import hashlib
import os
from pathlib import Path

import pytest

from src.atencion_armonica import generative_evidence_test_freeze as freeze
from src.atencion_armonica.generative_evidence_storage import write_json


TEST_ROOT = freeze.ROOT/".agent-work/phideus-r723-test-freeze-tests-20260909"/f"run-{os.getpid()}"


def root_ref(path):
    path = Path(path)
    raw = path.read_bytes()
    return {"path": path.relative_to(freeze.ROOT).as_posix(), "sha256": hashlib.sha256(raw).hexdigest()}


def local_ref(path, root):
    path = Path(path)
    raw = path.read_bytes()
    return {"path": path.relative_to(root).as_posix(), "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw)}


def roster():
    return [{"arm": arm, "checkpoint_seed": cp, "reader_seed": seed,
             "cell_id": f"{arm}-cp{cp}-seed{seed}"}
            for arm in ("local", "generative", "decoupled") for cp in (21, 22, 23)
            for seed in (2026090991, 2026090992, 2026090993)]


def test_missing_future_source_fails_before_selection_eligibility(monkeypatch):
    called = []
    monkeypatch.setattr(freeze, "_source_hashes",
                        lambda check: (_ for _ in ()).throw(FileNotFoundError("future source missing")))
    monkeypatch.setattr(freeze, "_selection_context", lambda check: called.append("selection"))
    with pytest.raises(FileNotFoundError, match="future source"):
        freeze._expected({"path": "unused", "sha256": "0"*64}, lambda: None)
    assert called == []


def test_source_inventory_is_explicit_and_rejects_absence(monkeypatch):
    assert "src/atencion_armonica/generative_evidence_inference.py" in freeze.REQUIRED_SOURCES
    folder = TEST_ROOT/"sources"
    folder.mkdir(parents=True, exist_ok=False)
    first, second = folder/"first.py", folder/"second.py"
    first.write_bytes(b"first\n")
    rel = lambda path: path.relative_to(freeze.ROOT).as_posix()
    monkeypatch.setattr(freeze, "REQUIRED_SOURCES", (rel(first), rel(second)))
    with pytest.raises(FileNotFoundError, match="second.py"):
        freeze._source_hashes(lambda: None)
    second.write_bytes(b"second\n")
    original_hashes = freeze._source_hashes(lambda: None)
    assert original_hashes == {
        rel(first): hashlib.sha256(b"first\n").hexdigest(),
        rel(second): hashlib.sha256(b"second\n").hexdigest()}
    first.rename(first.with_suffix(".preserved"))
    first.write_bytes(b"changed kernel\n")
    assert freeze._source_hashes(lambda: None) != original_hashes


def exclusion_fixture(folder):
    folder.mkdir(parents=True, exist_ok=False)
    source = folder/"observations.jsonl"
    source.write_bytes(b'{"scene_id":0}\n')
    source_ref = root_ref(source)
    catalog = folder/"mechanical_catalog.json"
    catalog_value = {"schema": "fixture-mechanical-catalog", "records": ["complete"]}
    write_json(catalog, catalog_value)
    catalog_ref = root_ref(catalog)
    a, b = hashlib.sha256(b"a").hexdigest(), hashlib.sha256(b"b").hexdigest()
    value = {"schema": "generative-evidence-observed-exclusions-v1",
        "status": "EXCLUSIONS_ONLY_NOT_TEST_AUTHORIZED",
        "fingerprint": "SHA256 sorted absolute little-endian float32 log_f bytes",
        "groups": [
            {"name": "original", "source": source_ref, "fingerprints": [a, b], "count": 2,
             "unique_count": 2, "additional_unique": 2, "declared_alias": False},
            {"name": "alias", "source": source_ref, "fingerprints": [a], "count": 1,
             "unique_count": 1, "additional_unique": 0, "declared_alias": True}],
        "fingerprints": sorted([a, b]), "unique_count": 2,
        "consumed_sha256": {catalog_ref["path"]: catalog_ref["sha256"],
                            source_ref["path"]: source_ref["sha256"]}, "test_access": False}
    path = folder/"inventory.json"
    write_json(path, value)
    return path, value, catalog_value


def test_exclusion_inventory_reopens_sources_and_rejects_alias_drift(monkeypatch):
    path, value, catalog = exclusion_fixture(TEST_ROOT/"exclusions")
    builder = type("Builder", (), {"mechanical_catalog": staticmethod(lambda: catalog)})
    kernel = type("Kernel", (), {"build_exclusions": staticmethod(lambda ref: value)})
    monkeypatch.setattr(freeze, "_exclusion_builder", lambda: builder)
    monkeypatch.setattr(freeze, "_exclusion_kernel", lambda: kernel)
    assert freeze._verify_exclusions(root_ref(path), lambda: None) == value
    bad = deepcopy(value)
    bad["groups"][1]["fingerprints"] = [hashlib.sha256(b"new").hexdigest()]
    bad["groups"][1]["additional_unique"] = 1
    bad_path = path.with_name("bad-inventory.json")
    write_json(bad_path, bad)
    with pytest.raises(ValueError, match="alias"):
        freeze._verify_exclusions(root_ref(bad_path), lambda: None)


def test_exclusion_inventory_rejects_self_consistent_subset(monkeypatch):
    path, complete, catalog = exclusion_fixture(TEST_ROOT/"exclusion-subset")
    subset = deepcopy(complete)
    subset["groups"] = [deepcopy(subset["groups"][0])]
    subset["groups"][0].update(fingerprints=[subset["fingerprints"][0]], count=1,
                                unique_count=1, additional_unique=1)
    subset["fingerprints"] = [subset["fingerprints"][0]]
    subset["unique_count"] = 1
    subset_path = path.with_name("subset-inventory.json")
    write_json(subset_path, subset)
    builder = type("Builder", (), {"mechanical_catalog": staticmethod(lambda: catalog)})
    kernel = type("Kernel", (), {"build_exclusions": staticmethod(lambda ref: complete)})
    monkeypatch.setattr(freeze, "_exclusion_builder", lambda: builder)
    monkeypatch.setattr(freeze, "_exclusion_kernel", lambda: kernel)
    with pytest.raises(ValueError, match="complete exact reconstruction"):
        freeze._verify_exclusions(root_ref(subset_path), lambda: None)


class FakeStore:
    def __init__(self, root, normalizer):
        self.root, self.normalizer = root, normalizer

    def path(self, name):
        return self.root/name

    def reference(self, path):
        return local_ref(path, self.root)

    def json(self, ref):
        assert ref == self.normalizer
        return {"schema": "fixture-normalizer"}


class FakeCellArtifacts:
    def __init__(self, root, *, binding):
        self.root, self.binding = Path(root), binding

    def path(self, name):
        return self.root/name

    def reference(self, path):
        return local_ref(path, self.root)

    def load_state(self, ref):
        return {"binding": self.binding, "epoch": 5, "next_batch": 0,
                "arm": self.binding["arm"], "checkpoint_seed": self.binding["checkpoint_seed"],
                "reader_seed": self.binding["reader_seed"]}


def selection_fixture(folder):
    folder.mkdir(parents=True, exist_ok=False)
    manifest_path = folder/"selection-manifest.json"
    write_json(manifest_path, {"fixture": "selection stage"})
    selection_manifest_ref = root_ref(manifest_path)
    training_manifest = {"path": "fixture/training-manifest.json", "sha256": "1"*64}
    normal_path = folder/"normalizers.json"
    write_json(normal_path, {"fixture": "normalizers"})
    normal_root = root_ref(normal_path)
    normal_local = local_ref(normal_path, folder)
    cells, selected_states = [], {arm: {"epoch": 5, "cells": []}
                                  for arm in ("local", "generative", "decoupled")}
    for cell in roster():
        root = folder/"training"/cell["cell_id"]
        root.mkdir(parents=True)
        binding = {"training_manifest": training_manifest,
            "data": {"normalizers": normal_local}, "arm": cell["arm"],
            "checkpoint_seed": cell["checkpoint_seed"], "reader_seed": cell["reader_seed"]}
        write_json(root/"binding.json", binding)
        state_path = root/"state-5.pt"
        state_path.write_bytes(cell["cell_id"].encode())
        state_local = local_ref(state_path, root)
        snapshots = [state_local for _ in range(51)]
        complete = {"status": "TRAINED_NOT_SELECTED", "binding": binding, "arm": cell["arm"],
            "checkpoint_seed": cell["checkpoint_seed"], "reader_seed": cell["reader_seed"],
            "snapshots": snapshots}
        complete_path = root/"complete.json"
        write_json(complete_path, complete)
        complete_ref = root_ref(complete_path)
        cells.append({"cell": cell, "complete": complete_ref})
        selected_states[cell["arm"]]["cells"].append({"cell": cell, "cell_complete": complete_ref,
            "epoch": 5, "state": root_ref(state_path), "calibration": {"path": "cal", "sha256": "2"*64},
            "predictions": {"path": "pred", "sha256": "3"*64}, "calibration_readout": {}})
    training_index_path = folder/"training-index.json"
    write_json(training_index_path, {"schema": "generative-evidence-training-complete-v1",
        "status": "TRAINED_NOT_SELECTED", "manifest": training_manifest, "cells": cells,
        "cell_count": 27, "accumulated_seconds": 1., "test_access": False})
    training_index_ref = root_ref(training_index_path)
    selection = {"schema": "generative-evidence-calibration-selection-v1",
        "binding": {"selection_manifest": selection_manifest_ref, "training_manifest": training_manifest,
                    "training_complete": training_index_ref},
        "status": "CALIBRATION_SELECTED_NOT_TEST_AUTHORIZED", "training_accumulated_seconds": 1.,
        "cell_count": 27, "calibration_record_count": 270,
        "calibration_records": [{"fixture": i} for i in range(270)],
        "initial_models": {}, "selection": {"selected": {arm: {"epoch": 5}
            for arm in selected_states}}, "selected_states": selected_states, "test_access": False}
    selection_path = folder/"selection.json"
    write_json(selection_path, selection)
    store = FakeStore(folder, normal_local)
    delivery = {"store": store, "corpus": type("Corpus", (), {"normalizer_ref": normal_local})()}
    stage = {"training": {"manifest": training_manifest, "index": training_index_ref},
             "output": {"path": root_ref(selection_path)["path"],
                        "status": "CALIBRATION_SELECTED_NOT_TEST_AUTHORIZED"},
             "runtime": {"python": "fixture"}}
    return root_ref(selection_path), selection_manifest_ref, stage, delivery, normal_root


def test_selection_context_reopens_all_27_states_and_common_normalizer(monkeypatch):
    selected_ref, stage_ref, stage, delivery, normalizer = selection_fixture(TEST_ROOT/"selection")
    class SelectionRunner:
        @staticmethod
        def verified_selection(): return selected_ref
        @staticmethod
        def read_manifest(ref):
            assert ref == stage_ref
            return stage
    class Training:
        @staticmethod
        def read_training_manifest(ref): return {"roster": roster()}, delivery
        @staticmethod
        def cell_roster(): return roster()
    monkeypatch.setattr(freeze, "_selection_runner", lambda: SelectionRunner)
    monkeypatch.setattr(freeze, "_training_runner", lambda: Training)
    from src.atencion_armonica import generative_evidence_cell
    monkeypatch.setattr(generative_evidence_cell, "CellArtifacts", FakeCellArtifacts)
    checks = []
    result = freeze._selection_context(lambda: checks.append(1))
    assert len(result["selected_states"]) == 27 and result["normalizers"] == normalizer
    assert len(checks) >= 29
    bad = deepcopy(stage)
    bad["training"]["index"] = {"path": "wrong", "sha256": "0"*64}
    monkeypatch.setattr(SelectionRunner, "read_manifest", staticmethod(lambda ref: bad))
    with pytest.raises(ValueError, match="manifest"):
        freeze._selection_context(lambda: None)


def context_fixture():
    states = [{"cell": cell, "epoch": 5, "complete": {"path": cell["cell_id"]+"/complete", "sha256": "4"*64},
               "state": {"path": cell["cell_id"]+"/state", "sha256": "5"*64}} for cell in roster()]
    return {"selection": {"path": "selection", "sha256": "6"*64},
        "selection_manifest": {"path": "selection-manifest", "sha256": "7"*64},
        "training_manifest": {"path": "training-manifest", "sha256": "8"*64},
        "training_complete": {"path": "training-index", "sha256": "9"*64},
        "normalizers": {"path": "normalizers", "sha256": "a"*64},
        "selected_states": states, "runtime": {"python": "fixture"},
        "arms": ["local", "generative", "decoupled"]}


def test_prediction_roster_is_27_original_plus_18_generative_interventions():
    context = context_fixture()
    value = freeze._prediction_roster(context["selected_states"], context["arms"])
    assert len(value) == 45
    assert sum(row["kind"] == "original" for row in value) == 27
    assert sum(row["kind"] == "intervention" for row in value) == 18
    assert {row["intervention"] for row in value if row["kind"] == "intervention"} == {"zero", "decoupled"}


def test_freeze_and_verify_are_immutable_nonauthorized_ports(monkeypatch):
    folder = TEST_ROOT/"freeze"
    folder.mkdir(parents=True, exist_ok=False)
    path, exclusions, _ = exclusion_fixture(folder/"exclusions")
    exclusions_ref = root_ref(path)
    monkeypatch.setattr(freeze, "FREEZE", folder/"test_freeze.json")
    monkeypatch.setattr(freeze, "_source_hashes", lambda check: {"future.py": "b"*64})
    context = context_fixture()
    monkeypatch.setattr(freeze, "_selection_context", lambda check: context)
    monkeypatch.setattr(freeze, "_verify_exclusions", lambda ref, check: exclusions)
    result = freeze.freeze_tests(exclusions_ref=exclusions_ref, check=lambda: None)
    assert result["manifest"]["draw_count"] == 0
    assert result["manifest"]["test_truth_access"] is False
    assert result["manifest"]["sidecar_access"] is False
    assert len(result["manifest"]["tests"]) == 4
    assert len(result["manifest"]["prediction_roster"]) == 45
    assert freeze.verify_freeze(result["freeze"], check=lambda: None) == result
    monkeypatch.setattr(freeze, "_source_hashes", lambda check: {"future.py": "c"*64})
    with pytest.raises(ValueError, match="dependencies"):
        freeze.verify_freeze(result["freeze"], check=lambda: None)
    monkeypatch.setattr(freeze, "_source_hashes", lambda check: {"future.py": "b"*64})
    changed = deepcopy(context)
    changed["runtime"] = {"python": "changed"}
    monkeypatch.setattr(freeze, "_selection_context", lambda check: changed)
    with pytest.raises(ValueError, match="dependencies"):
        freeze.verify_freeze(result["freeze"], check=lambda: None)
