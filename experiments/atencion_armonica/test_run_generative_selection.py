"""Mechanical CPU tests for the selection supervisor; no real cells or selection."""
from contextlib import nullcontext
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import signal

import pytest

from experiments.atencion_armonica import run_generative_selection as runner
from src.atencion_armonica.generative_evidence_storage import write_json


TEST_ROOT = runner.ROOT/".agent-work/phideus-r718-selection-supervisor-tests-20260909"/f"run-{os.getpid()}"


def ref(path):
    return {"path": str(path), "sha256": hashlib.sha256(str(path).encode()).hexdigest()}


def roster():
    arms = ("local", "generative", "decoupled")
    return [{"arm": arm, "checkpoint_seed": cp, "reader_seed": seed,
             "cell_id": f"{arm}-cp{cp}-seed{seed}"}
            for arm in arms for cp in (21, 22, 23) for seed in (2026090991, 2026090992, 2026090993)]


def training_fixture():
    manifest_ref, index_ref, last = ref("training-manifest"), ref("training-index"), ref("last-exit")
    cells = []
    for cell in roster():
        path = (f"data/atencion_armonica/generative_evidence_reader_v1/training/{cell['arm']}"
                f"/cp_{cell['checkpoint_seed']}/seed_{cell['reader_seed']}/complete.json")
        cells.append({"cell": cell, "complete": ref(path)})
    manifest = {"roster": roster()}
    index = {"schema": "generative-evidence-training-complete-v1", "status": "TRAINED_NOT_SELECTED",
        "manifest": manifest_ref, "cells": cells, "cell_count": 27,
        "accumulated_seconds": 120., "test_access": False}
    ids = [c["cell_id"] for c in roster()]
    accounting = 120., last, {i: 4. for i in ids}, {i: ref(i+"-exit") for i in ids}

    class Training:
        TRAINING_MANIFEST = Path("training-manifest")
        TRAINING_ROOT = Path("training")
        def reference(self, path):
            return manifest_ref if path == self.TRAINING_MANIFEST else index_ref
        def read_training_manifest(self, value):
            assert value == manifest_ref
            return manifest, object()
        def training_accumulated(self, value):
            assert value == manifest_ref
            return accounting
    return Training(), index_ref, index, {"manifest": manifest_ref, "index": index_ref,
        "last_exit": last, "accumulated_seconds": 120.}


def test_training_boundary_reads_only_complete_metadata(monkeypatch):
    training, index_ref, index, expected = training_fixture()
    monkeypatch.setattr(runner, "_training", lambda: training)
    monkeypatch.setattr(runner, "_read", lambda value: index if value == index_ref else None)
    assert runner._training_boundary() == expected
    bad = deepcopy(index)
    bad["cells"].pop()
    monkeypatch.setattr(runner, "_read", lambda value: bad)
    with pytest.raises(ValueError, match="27-cell"):
        runner._training_boundary()


def test_manifest_binds_sources_runtime_training_budget_and_nonauthorized_output(monkeypatch):
    control = TEST_ROOT/"manifest"
    control.mkdir(parents=True, exist_ok=False)
    monkeypatch.setattr(runner, "CONTROL", control)
    monkeypatch.setattr(runner, "MANIFEST", control/"manifest.json")
    monkeypatch.setattr(runner, "PREPARATION_LOCK", control/"preparation.lock")
    monkeypatch.setattr(runner, "_lock", lambda path: nullcontext())
    training = {"manifest": ref("training-manifest"), "index": ref("training-index"),
                "last_exit": ref("last-exit"), "accumulated_seconds": 120.}
    source_map, runtime = {"selection.py": "a"*64}, {"python": "fixture"}
    monkeypatch.setattr(runner, "_training_boundary", lambda: training)
    monkeypatch.setattr(runner, "sources", lambda: source_map)
    monkeypatch.setattr(runner, "runtime", lambda: runtime)
    manifest_ref = runner.initialize_selection()
    value = runner.read_manifest(manifest_ref)
    assert value["training"] == training and value["limit_seconds"] == 12*3600
    assert value["device"] == "cpu" and value["test_access"] is False
    assert value["output"]["status"] == "CALIBRATION_SELECTED_NOT_TEST_AUTHORIZED"
    monkeypatch.setattr(runner, "sources", lambda: {"selection.py": "b"*64})
    with pytest.raises(ValueError, match="sources"):
        runner.read_manifest(manifest_ref)


def test_output_contract_uses_canonical_arm_ids_from_r714(monkeypatch):
    control = TEST_ROOT/"output-contract"
    control.mkdir(parents=True, exist_ok=False)
    manifest_path, output = control/"manifest.json", control/"index.json"
    write_json(manifest_path, {"fixture": True})
    monkeypatch.setattr(runner, "MANIFEST", manifest_path)
    monkeypatch.setattr(runner, "OUTPUT", output)
    class Training:
        @staticmethod
        def cell_roster(): return roster()
    monkeypatch.setattr(runner, "_training", lambda: Training)
    training = {"manifest": ref("training-manifest"), "index": ref("training-index"),
                "accumulated_seconds": 120.}
    binding = {"selection_manifest": runner.reference(manifest_path),
               "training_manifest": training["manifest"], "training_complete": training["index"]}
    arms = ("local", "generative", "decoupled")
    value = {"schema": "generative-evidence-calibration-selection-v1", "binding": binding,
        "status": "CALIBRATION_SELECTED_NOT_TEST_AUTHORIZED", "training_accumulated_seconds": 120.,
        "cell_count": 27, "calibration_record_count": 270,
        "calibration_records": [{} for _ in range(270)], "initial_models": {}, "selection": {},
        "selected_states": {arm: {"cells": [{} for _ in range(9)]} for arm in arms},
        "test_access": False}
    write_json(output, value)
    manifest = {"training": training,
                "output": {"status": "CALIBRATION_SELECTED_NOT_TEST_AUTHORIZED"}}
    assert runner._output_reference(manifest) == runner.reference(output)


def test_selection_prefix_inherits_training_elapsed_and_requires_typed_receipts(monkeypatch):
    control = TEST_ROOT/"prefix"
    control.mkdir(parents=True, exist_ok=False)
    monkeypatch.setattr(runner, "CONTROL", control)
    training = {"accumulated_seconds": 120., "last_exit": ref("training-last")}
    manifest_ref = ref("selection-manifest")
    monkeypatch.setattr(runner, "read_manifest", lambda value: {"training": training})
    attempt, unit = "attempt-0000", "phideus-generative-selection-"+"a"*16
    remaining = runner.LIMIT_SECONDS-120
    command = runner._service_command(unit, remaining-runner.GRACE, attempt)
    launch_path = control/f"{attempt}.launch.json"
    write_json(launch_path, {"schema": "generative-evidence-selection-launch-v1",
        "manifest": manifest_ref, "previous_exit": training["last_exit"], "unit": unit,
        "command": command, "used_seconds": 120., "remaining_seconds": remaining,
        "runtime_seconds": remaining-runner.GRACE})
    launch_ref = runner.reference(launch_path)
    write_json(control/f"{attempt}.worker.json", {"launch": launch_ref,
        "status": "PAUSED_RECOVERABLE", "reason": "fixture stop", "seconds": 2.,
        "peak_rss_bytes": 1234, "cuda_initialized": False})
    write_json(control/f"{attempt}.exit.json", {"launch": launch_ref, "process_returncode": 75,
        "seconds": 3., "terminal": True})
    total, previous = runner.selection_accumulated(manifest_ref)
    assert total == 123. and previous == runner.reference(control/f"{attempt}.exit.json")
    worker = runner._read(runner.reference(control/f"{attempt}.worker.json"))
    assert not runner._worker_schema({**worker, "cuda_initialized": True}, 75)
    assert not runner._worker_schema({**worker, "status": runner.STATUS, "selection": ref("x")}, 75)


def test_worker_is_cpu_only_typed_and_calls_selection_inside_guard(monkeypatch):
    control = TEST_ROOT/"worker"
    control.mkdir(parents=True, exist_ok=False)
    output = control/"index.json"
    output.write_bytes(b"fixture")
    monkeypatch.setattr(runner, "CONTROL", control)
    monkeypatch.setattr(runner, "OUTPUT", output)
    manifest_ref = ref("manifest")
    attempt, unit = "attempt-0000", "phideus-generative-selection-"+"b"*16
    runtime_seconds = runner.LIMIT_SECONDS-100-runner.GRACE
    command = runner._service_command(unit, runtime_seconds, attempt)
    write_json(control/f"{attempt}.launch.json", {"schema": "generative-evidence-selection-launch-v1",
        "manifest": manifest_ref, "previous_exit": ref("training-last"), "unit": unit,
        "command": command, "used_seconds": 100., "remaining_seconds": runner.LIMIT_SECONDS-100,
        "runtime_seconds": runtime_seconds})
    monkeypatch.setattr(runner, "_prefix", lambda value, paths: (100., ref("training-last")))
    monkeypatch.setattr(runner, "_verify_service", lambda got, seconds: None)
    monkeypatch.setattr(runner, "read_manifest", lambda value: {"output": {}, "training": {}})
    monkeypatch.setattr(runner, "_worker_check", lambda *args: None)
    selected = runner.reference(output)
    monkeypatch.setattr(runner, "_output_reference", lambda manifest: selected)
    from src.atencion_armonica import generative_evidence_selection as selection
    monkeypatch.setattr(selection, "run_selection", lambda *, check, stage_manifest: {
        "path": "index.json", "sha256": selected["sha256"], "bytes": output.stat().st_size})
    import torch
    monkeypatch.setattr(torch, "set_num_threads", lambda value: None)
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda value: None)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setenv("PHIDEUS_GENERATIVE_SELECTION_UNIT", unit)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        monkeypatch.setenv(name, "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert runner.worker_selection(attempt) == 0
    receipt = runner._read(runner.reference(control/f"{attempt}.worker.json"))
    assert receipt["status"] == runner.STATUS and receipt["selection"] == selected
    assert receipt["cuda_initialized"] is False


def test_worker_sigterm_during_heavy_consumer_is_typed_recoverable(monkeypatch):
    control = TEST_ROOT/"signal"
    control.mkdir(parents=True, exist_ok=False)
    monkeypatch.setattr(runner, "CONTROL", control)
    manifest_ref = ref("manifest")
    attempt, unit = "attempt-0000", "phideus-generative-selection-"+"b"*16
    runtime_seconds = runner.LIMIT_SECONDS-100-runner.GRACE
    command = runner._service_command(unit, runtime_seconds, attempt)
    write_json(control/f"{attempt}.launch.json", {"schema": "generative-evidence-selection-launch-v1",
        "manifest": manifest_ref, "previous_exit": ref("training-last"), "unit": unit,
        "command": command, "used_seconds": 100., "remaining_seconds": runner.LIMIT_SECONDS-100,
        "runtime_seconds": runtime_seconds})
    monkeypatch.setattr(runner, "_prefix", lambda value, paths: (100., ref("training-last")))
    monkeypatch.setattr(runner, "_verify_service", lambda got, seconds: None)
    monkeypatch.setattr(runner, "read_manifest", lambda value: {"output": {}, "training": {}})
    monkeypatch.setattr(runner, "_worker_check", lambda *args: None)
    from src.atencion_armonica import generative_evidence_selection as selection
    def interrupt(*, check, stage_manifest):
        os.kill(os.getpid(), signal.SIGTERM)
        raise AssertionError("the installed worker handler did not interrupt")
    monkeypatch.setattr(selection, "run_selection", interrupt)
    import torch
    monkeypatch.setattr(torch, "set_num_threads", lambda value: None)
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda value: None)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setenv("PHIDEUS_GENERATIVE_SELECTION_UNIT", unit)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        monkeypatch.setenv(name, "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert runner.worker_selection(attempt) == 75
    receipt = runner._read(runner.reference(control/f"{attempt}.worker.json"))
    assert receipt["status"] == "PAUSED_RECOVERABLE"
    assert "stop requested" in receipt["reason"] and receipt["cuda_initialized"] is False


def test_parent_launch_is_cpu_bounded_and_exit0_still_uses_typed_verifier(monkeypatch):
    control = TEST_ROOT/"parent"
    control.mkdir(parents=True, exist_ok=False)
    manifest = control/"manifest.json"
    manifest.write_bytes(b"fixture")
    monkeypatch.setattr(runner, "CONTROL", control)
    monkeypatch.setattr(runner, "MANIFEST", manifest)
    monkeypatch.setattr(runner, "PREPARATION_LOCK", control/"preparation.lock")
    monkeypatch.setattr(runner, "_lock", lambda path: nullcontext())
    monkeypatch.setattr(runner, "read_manifest", lambda value: {"training": {}})
    monkeypatch.setattr(runner, "selection_accumulated", lambda value: (100., ref("training-last")))
    class Training:
        @staticmethod
        def check_disk(): return {"new_bytes": 0, "free_bytes": 10**12}
    monkeypatch.setattr(runner, "_training", lambda: Training)
    observed = {}
    def execute(command, unit, launch_ref, exit_path):
        observed.update(command=command, unit=unit, launch=runner._read(launch_ref))
        return {"process_returncode": 0}
    monkeypatch.setattr(runner, "_execute_command", execute)
    expected = ref("selected")
    monkeypatch.setattr(runner, "verified_selection", lambda: expected)
    assert runner.run_selection() == expected
    command = observed["command"]
    assert "CUDA_VISIBLE_DEVICES=" in command and "--property=MemoryMax=6G" in command
    assert "--property=MemorySwapMax=0" in command
    assert observed["launch"]["runtime_seconds"] == runner.LIMIT_SECONDS-100-runner.GRACE
    assert observed["launch"]["previous_exit"] == ref("training-last")


def test_public_cli_has_only_initialize_run_and_hidden_worker():
    source = Path(runner.__file__).read_text()
    assert 'mode.add_argument("--initialize"' in source
    assert 'mode.add_argument("--run"' in source
    assert 'mode.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)' in source
    assert "draw" not in source and "forward" not in source
