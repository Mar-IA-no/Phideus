"""Mechanical CPU tests for the finite fresh-test supervisor; no real stages."""
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import signal
from types import SimpleNamespace

import pytest

from experiments.atencion_armonica import run_generative_tests as runner
from src.atencion_armonica.generative_evidence_storage import write_json


TEST_ROOT = runner.ROOT/".agent-work/phideus-r728-test-supervisor-tests-20260909"/f"run-{os.getpid()}"


def fake_ref(name):
    return {"path": name, "sha256": hashlib.sha256(name.encode()).hexdigest()}


def root_ref(path):
    path = Path(path)
    return {"path": path.relative_to(runner.ROOT).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def fake_torch():
    cuda = SimpleNamespace(is_initialized=lambda: False, max_memory_reserved=lambda *a: 0)
    return SimpleNamespace(cuda=cuda, set_num_threads=lambda value: None,
        set_num_interop_threads=lambda value: None, use_deterministic_algorithms=lambda value: None)


def test_stage_roster_is_freeze_then_four_fixed_prediction_evaluation_replays():
    stages = runner.stage_roster()
    assert len(stages) == 13 and stages[0] == {
        "index": 0, "kind": "freeze", "split": None, "device": "cpu"}
    assert [row["split"] for row in stages[1::3]] == [
        "iid", "ood_beta", "ood_polyphony", "deformed_family"]
    for start in range(1, 13, 3):
        assert [(row["kind"], row["device"]) for row in stages[start:start+3]] == [
            ("predict", "cuda:0"), ("evaluate", "cpu"), ("replay", "cpu")]
    assert [row["index"] for row in stages] == list(range(13))


def test_source_closure_names_supervisor_inference_evaluation_and_resource_helpers():
    paths = {path.relative_to(runner.ROOT).as_posix() for path in runner._source_paths()}
    required = {
        "experiments/atencion_armonica/run_generative_tests.py",
        "experiments/atencion_armonica/run_generative_training.py",
        "experiments/atencion_armonica/run_generative_selection.py",
        "src/atencion_armonica/generative_evidence_test_freeze.py",
        "src/atencion_armonica/generative_evidence_fresh_data.py",
        "src/atencion_armonica/generative_evidence_fresh_inference.py",
        "src/atencion_armonica/generative_evidence_inference.py",
        "src/atencion_armonica/generative_evidence_fresh_evaluation.py",
    }
    assert required <= paths
    assert required <= set(runner.sources())


def test_initialize_is_metadata_only_and_revalidates_selection_sources_and_exclusions(monkeypatch):
    folder = TEST_ROOT/"initialize"
    folder.mkdir(parents=True, exist_ok=False)
    inventory = folder/"inventory.json"
    write_json(inventory, {"schema": "generative-evidence-observed-exclusions-v1",
        "status": "EXCLUSIONS_ONLY_NOT_TEST_AUTHORIZED", "test_access": False})
    monkeypatch.setattr(runner, "CONTROL", folder/"control")
    monkeypatch.setattr(runner, "MANIFEST", folder/"control/manifest.json")
    monkeypatch.setattr(runner, "COMMON_LOCK", folder/"common.lock")
    monkeypatch.setattr(runner, "FINAL", folder/"final.json")
    class Training:
        @staticmethod
        def _lock(path): return nullcontext()
    monkeypatch.setattr(runner, "_training", lambda: Training)
    authority = {"selection": fake_ref("selection"), "selection_manifest": fake_ref("selection-manifest"),
                 "training_manifest": fake_ref("training-manifest"),
                 "training_complete": fake_ref("training-complete")}
    monkeypatch.setattr(runner, "_selection_authority", lambda: authority)
    monkeypatch.setattr(runner, "sources", lambda: {"future.py": "a"*64})
    monkeypatch.setattr(runner, "runtime", lambda: {"python": "fixture"})
    ref = runner.initialize_tests(inventory.relative_to(runner.ROOT).as_posix())
    value = runner.read_manifest(ref)
    assert value["selection"] == authority and value["exclusions"] == root_ref(inventory)
    assert value["policy"] == {"freeze_before_draw": True, "seal_before_truth": True,
                               "adaptive_test_reuse": False, "promotion": False}
    assert value["output"]["status"] == runner.FINAL_STATUS
    monkeypatch.setattr(runner, "sources", lambda: {"future.py": "b"*64})
    with pytest.raises(ValueError, match="sources"):
        runner.read_manifest(ref)


def test_service_commands_are_own_module_cpu_or_exact_gpu_and_bounded():
    cpu, gpu = runner.stage_roster()[0], runner.stage_roster()[1]
    unit = "phideus-generative-test-"+"a"*16
    first = runner._service_command(unit, 123, cpu, "attempt-0000")
    second = runner._service_command(unit, 123, gpu, "attempt-0000")
    for command in (first, second):
        assert "--property=MemoryMax=6G" in command
        assert "--property=MemorySwapMax=0" in command
        assert "--property=RuntimeMaxSec=123s" in command
        assert "--property=TimeoutStopSec=30s" in command
        assert "experiments.atencion_armonica.run_generative_tests" in command
        assert "run_generative_training" not in command
    assert "CUDA_VISIBLE_DEVICES=" in first
    assert "CUDA_VISIBLE_DEVICES=0" in second


def _receipt(control, manifest_ref, index, *, seconds, status, output=None, availability=None):
    stage = runner.stage_roster()[index]
    attempt = f"attempt-{len(list(control.glob('attempt-*.launch.json'))):04d}"
    prior_exits = sorted(control.glob("attempt-*.exit.json"))
    previous = root_ref(prior_exits[-1]) if prior_exits else None
    used = sum(json.loads(path.read_text())["seconds"] for path in prior_exits)
    remaining = int(runner.LIMIT_SECONDS-used)
    unit = "phideus-generative-test-"+f"{len(prior_exits):016x}"
    command = runner._service_command(unit, remaining-runner.GRACE, stage, attempt)
    launch = control/f"{attempt}.launch.json"
    write_json(launch, {"schema": "generative-evidence-fresh-test-launch-v1",
        "manifest": manifest_ref, "stage": stage, "previous_exit": previous, "unit": unit,
        "command": command, "availability": availability, "used_seconds": float(used),
        "remaining_seconds": remaining, "runtime_seconds": remaining-runner.GRACE})
    launch_ref = root_ref(launch)
    worker = {"launch": launch_ref, "stage": stage, "status": status, "seconds": seconds,
        "peak_rss_bytes": 1000, "peak_reserved_bytes": 0, "cuda_initialized": False,
        "availability": availability}
    code = 0 if status == runner._success_status(stage) else 75
    if output is not None:
        worker["output"] = output
    else:
        worker["reason"] = "fixture pause"
    write_json(control/f"{attempt}.worker.json", worker)
    write_json(control/f"{attempt}.exit.json", {"launch": launch_ref, "process_returncode": code,
        "seconds": seconds, "terminal": True})
    return launch


def test_ledger_charges_pause_and_retry_and_advances_only_typed_exact_output(monkeypatch):
    control = TEST_ROOT/"ledger"
    control.mkdir(parents=True, exist_ok=False)
    monkeypatch.setattr(runner, "CONTROL", control)
    manifest_ref = fake_ref("manifest")
    monkeypatch.setattr(runner, "read_manifest", lambda ref: {"stages": runner.stage_roster()})
    monkeypatch.setattr(runner, "_availability_ok", lambda value: value == {"gpu": "free"})
    outputs = {0: control/"freeze.json", 1: control/"seal.json"}
    for path in outputs.values():
        path.write_bytes(path.name.encode())
    monkeypatch.setattr(runner, "_stage_path", lambda stage: outputs[stage["index"]])
    first = _receipt(control, manifest_ref, 0, seconds=3., status="FREEZE_COMPLETE",
                     output=root_ref(outputs[0]))
    _receipt(control, manifest_ref, 1, seconds=5., status="PAUSED_RECOVERABLE",
             availability={"gpu": "free"})
    # A retry remains on stage 1 and inherits both elapsed receipts.
    _receipt(control, manifest_ref, 1, seconds=7., status="PREDICTIONS_SEALED",
             output=root_ref(outputs[1]), availability={"gpu": "free"})
    state = runner.test_accumulated(manifest_ref)
    assert first.name == "attempt-0000.launch.json"
    assert state["seconds"] == 15. and state["next_stage"] == 2
    assert len(state["completions"]) == 2 and state["last_status"] == "COMPLETE"
    bad = control/"attempt-0002.worker.json"
    value = json.loads(bad.read_text())
    value["output"] = fake_ref("other")
    bad.write_bytes((json.dumps(value, sort_keys=True, separators=(",", ":"))+"\n").encode())
    with pytest.raises(ValueError, match="another output"):
        runner.test_accumulated(manifest_ref)


def test_missing_worker_receipt_is_ambiguous_and_never_skipped(monkeypatch):
    control = TEST_ROOT/"missing-worker"
    control.mkdir(parents=True, exist_ok=False)
    monkeypatch.setattr(runner, "CONTROL", control)
    monkeypatch.setattr(runner, "read_manifest", lambda ref: {"stages": runner.stage_roster()})
    manifest_ref = fake_ref("manifest")
    stage, attempt = runner.stage_roster()[0], "attempt-0000"
    unit = "phideus-generative-test-"+"b"*16
    command = runner._service_command(unit, runner.LIMIT_SECONDS-runner.GRACE, stage, attempt)
    path = control/f"{attempt}.launch.json"
    write_json(path, {"schema": "generative-evidence-fresh-test-launch-v1", "manifest": manifest_ref,
        "stage": stage, "previous_exit": None, "unit": unit, "command": command,
        "availability": None, "used_seconds": 0., "remaining_seconds": runner.LIMIT_SECONDS,
        "runtime_seconds": runner.LIMIT_SECONDS-runner.GRACE})
    with pytest.raises(RuntimeError, match="unreconciled"):
        runner.test_accumulated(manifest_ref)


@pytest.mark.parametrize("interrupt", [False, True])
def test_cpu_worker_publishes_typed_success_or_recoverable_signal(interrupt, monkeypatch):
    control = TEST_ROOT/("worker-pause" if interrupt else "worker-success")
    control.mkdir(parents=True, exist_ok=False)
    output = control/"freeze.json"
    output.write_bytes(b"freeze")
    monkeypatch.setattr(runner, "CONTROL", control)
    monkeypatch.setattr(runner, "_stage_path", lambda stage: output)
    manifest_ref = fake_ref("manifest")
    stage, attempt = runner.stage_roster()[0], "attempt-0000"
    unit = "phideus-generative-test-"+"c"*16
    runtime_seconds = runner.LIMIT_SECONDS-runner.GRACE
    command = runner._service_command(unit, runtime_seconds, stage, attempt)
    launch = control/f"{attempt}.launch.json"
    write_json(launch, {"schema": "generative-evidence-fresh-test-launch-v1", "manifest": manifest_ref,
        "stage": stage, "previous_exit": None, "unit": unit, "command": command,
        "availability": None, "used_seconds": 0., "remaining_seconds": runner.LIMIT_SECONDS,
        "runtime_seconds": runtime_seconds})
    monkeypatch.setattr(runner, "_prefix", lambda ref, paths: {"seconds": 0., "previous_exit": None,
        "next_stage": 0, "completions": [], "last_status": None})
    monkeypatch.setattr(runner, "read_manifest", lambda ref: {"fixture": True})
    class Training:
        @staticmethod
        def _verify_service(unit, seconds): return None
        @staticmethod
        def _worker_check(*args, **kwargs):
            if args[2]:
                raise InterruptedError("fixture stop")
    monkeypatch.setattr(runner, "_training", lambda: Training)
    monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch())
    calls = []
    def execute(stage, manifest, check, progress):
        calls.append(stage)
        if interrupt:
            os.kill(os.getpid(), signal.SIGTERM)
            check()
        return root_ref(output)
    monkeypatch.setattr(runner, "_execute_stage", execute)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        monkeypatch.setenv(name, "1")
    code = runner.worker_stage(attempt)
    assert code == (75 if interrupt else 0)
    receipt = json.loads((control/f"{attempt}.worker.json").read_text())
    assert receipt["status"] == ("PAUSED_RECOVERABLE" if interrupt else "FREEZE_COMPLETE")
    assert receipt["cuda_initialized"] is False and receipt["availability"] is None
    assert calls == [stage]


def test_prediction_stage_draws_before_inference_and_returns_exact_seal(monkeypatch):
    folder = TEST_ROOT/"stage-order"
    folder.mkdir(parents=True, exist_ok=False)
    freeze_path, seal = folder/"freeze.json", folder/"seal.json"
    freeze_path.write_bytes(b"freeze")
    seal.write_bytes(b"seal")
    frozen = runner._freeze()
    monkeypatch.setattr(frozen, "FREEZE", freeze_path)
    from src.atencion_armonica import generative_evidence_fresh_data as data
    from src.atencion_armonica import generative_evidence_fresh_inference as inference
    events = []
    monkeypatch.setattr(data, "produce_test", lambda *a, **kw: events.append("draw"))
    def predict(*args, **kwargs):
        events.append("predict")
        return root_ref(seal)
    monkeypatch.setattr(inference, "prepare_and_predict", predict)
    result = runner._execute_stage(runner.stage_roster()[1], {}, lambda: None, lambda value: None)
    assert events == ["draw", "predict"] and result == root_ref(seal)


def test_final_index_requires_all_thirteen_and_replay_same_evaluation():
    stages, completions = runner.stage_roster(), []
    for stage in stages:
        output = fake_ref(f"output-{stage['index']}")
        if stage["kind"] == "replay":
            output = completions[-1]["output"]
        completions.append({"stage": stage, "launch": fake_ref(f"launch-{stage['index']}"),
            "exit": fake_ref(f"exit-{stage['index']}"), "worker": fake_ref(f"worker-{stage['index']}"),
            "output": output})
    state = {"seconds": 123., "previous_exit": completions[-1]["exit"], "next_stage": 13,
             "completions": completions, "last_status": "COMPLETE"}
    value = runner._final_value(fake_ref("manifest"), state)
    assert value["stage_count"] == 13 and value["promotion"] is False
    assert set(value["tests"]) == {"iid", "ood_beta", "ood_polyphony", "deformed_family"}
    changed = json.loads(json.dumps(state))
    changed["completions"][-1]["output"] = fake_ref("different-replay")
    with pytest.raises(ValueError, match="exact evaluation"):
        runner._final_value(fake_ref("manifest"), changed)
