"""Finite supervisor for freeze, four fresh tests, evaluation and replay.

The public parent performs metadata checks and launches one bounded worker per
durable stage.  It never draws, loads a model, opens truth or evaluates outside
the supervised service.  Completion is thirteen typed worker/terminal pairs,
not a successful process exit or the mere presence of an artifact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import signal
import sys
import time
import uuid

from src.atencion_armonica.generative_evidence_reuse import ROOT, VerifiedBytes
from src.atencion_armonica.generative_evidence_storage import write_json
from src.atencion_armonica.structured_source_artifacts import safe_member


TEMP = ROOT/".agent-work/phideus-generative-evidence-20260909"
CONTROL = TEMP/"test-control"
MANIFEST = CONTROL/"manifest.json"
COMMON_LOCK = TEMP/"preparation-control/operator.lock"
FRESH = ROOT/"data/atencion_armonica/generative_evidence_reader_v1/fresh"
FINAL = FRESH/"test_completion.json"
PROTOCOL = ROOT/"experiments/atencion_armonica/PROTOCOL_GENERATIVE_EVIDENCE_READER.md"
LIMIT_SECONDS = 4*3600
MEMORY_LIMIT = 6*1024**3
STORAGE_LIMIT = 60*1024**3
FREE_MIN = 80*1024**3
GRACE = 30
FINAL_STATUS = "FRESH_TESTS_EVALUATED_REPLAYED_NOT_PROMOTED"


def _training():
    from experiments.atencion_armonica import run_generative_training
    return run_generative_training


def _selection():
    from experiments.atencion_armonica import run_generative_selection
    return run_generative_selection


def _freeze():
    from src.atencion_armonica import generative_evidence_test_freeze
    return generative_evidence_test_freeze


def _profile():
    from src.atencion_armonica import generative_evidence_profile
    return generative_evidence_profile


def _number(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _valid_ref(value):
    return (isinstance(value, dict) and set(value) == {"path", "sha256"}
            and isinstance(value["path"], str) and bool(value["path"])
            and isinstance(value["sha256"], str) and len(value["sha256"]) == 64
            and all(c in "0123456789abcdef" for c in value["sha256"]))


def reference(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    return {"path": path.relative_to(ROOT).as_posix(),
            "sha256": hashlib.sha256(raw).hexdigest()}


def _read(ref):
    return VerifiedBytes(ROOT).json(ref)


def stage_roster():
    stages = [{"index": 0, "kind": "freeze", "split": None, "device": "cpu"}]
    index = 1
    for split, _, _ in _freeze().TESTS:
        for kind, device in (("predict", "cuda:0"), ("evaluate", "cpu"), ("replay", "cpu")):
            stages.append({"index": index, "kind": kind, "split": split, "device": device})
            index += 1
    if len(stages) != 13:
        raise RuntimeError("fresh-test supervisor requires exactly thirteen stages")
    return stages


def _source_paths():
    inherited = [ROOT/name for name in _freeze().REQUIRED_SOURCES]
    explicit = [Path(__file__).resolve(), PROTOCOL,
        ROOT/"experiments/atencion_armonica/run_generative_training.py",
        ROOT/"experiments/atencion_armonica/run_generative_selection.py",
        ROOT/"experiments/atencion_armonica/prepare_generative_evidence.py",
        ROOT/"src/atencion_armonica/generative_evidence_profile.py"]
    return sorted(set(inherited+explicit))


def sources():
    result = {}
    for path in _source_paths():
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"fresh-test execution source missing: {path.relative_to(ROOT)}")
        result[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def runtime():
    return _training().runtime()


def _selection_authority():
    selection_ref = _selection().verified_selection()
    if not _valid_ref(selection_ref):
        raise ValueError("selection authority returned an invalid root reference")
    value = _read(selection_ref)
    binding = value.get("binding") if isinstance(value, dict) else None
    if (not isinstance(value, dict)
            or value.get("status") != "CALIBRATION_SELECTED_NOT_TEST_AUTHORIZED"
            or value.get("test_access") is not False or not isinstance(binding, dict)
            or set(binding) != {"selection_manifest", "training_manifest", "training_complete"}):
        raise ValueError("fresh tests require the typed non-test-authorized selection")
    if any(not _valid_ref(binding[name]) for name in binding):
        raise ValueError("selection authority lacks its exact parent references")
    return {"selection": selection_ref, **binding}


def _exclusion_reference(relative):
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("--exclusions requires a root-relative inventory path")
    path = safe_member(ROOT, relative)
    if path.name != "inventory.json" or not path.is_file() or path.is_symlink():
        raise ValueError("exclusions must name an existing regular inventory.json")
    ref = reference(path)
    value = _read(ref)
    if (not isinstance(value, dict)
            or value.get("schema") != "generative-evidence-observed-exclusions-v1"
            or value.get("status") != "EXCLUSIONS_ONLY_NOT_TEST_AUTHORIZED"
            or value.get("test_access") is not False):
        raise ValueError("exclusion metadata has wrong schema or authority")
    return ref


def _manifest_value(exclusions):
    return {"schema": "generative-evidence-fresh-test-stage-v1",
        "stage": "FIXED_FOUR_TESTS_NOT_PROMOTED", "protocol": reference(PROTOCOL),
        "sources": sources(), "runtime": runtime(), "selection": _selection_authority(),
        "exclusions": exclusions, "stages": stage_roster(),
        "limits": {"total_seconds": LIMIT_SECONDS, "memory_max_bytes": MEMORY_LIMIT,
                   "storage_max_bytes": STORAGE_LIMIT, "free_min_bytes": FREE_MIN,
                   "termination_grace_seconds": GRACE},
        "output": {"path": FINAL.relative_to(ROOT).as_posix(), "status": FINAL_STATUS},
        "policy": {"freeze_before_draw": True, "seal_before_truth": True,
                   "adaptive_test_reuse": False, "promotion": False}}


def initialize_tests(exclusions_path):
    training = _training()
    with training._lock(COMMON_LOCK), training._lock(CONTROL/"operator.lock"):
        if MANIFEST.exists():
            raise FileExistsError("fresh-test manifest already exists; use --run")
        exclusions = _exclusion_reference(exclusions_path)
        value = _manifest_value(exclusions)
        CONTROL.mkdir(parents=True, exist_ok=True)
        write_json(MANIFEST, value)
        return reference(MANIFEST)


def read_manifest(ref):
    if not _valid_ref(ref) or ref["path"] != MANIFEST.relative_to(ROOT).as_posix():
        raise ValueError("unexpected fresh-test manifest path")
    value = _read(ref)
    keys = {"schema", "stage", "protocol", "sources", "runtime", "selection", "exclusions",
            "stages", "limits", "output", "policy"}
    if (set(value) != keys or value.get("schema") != "generative-evidence-fresh-test-stage-v1"
            or value.get("stage") != "FIXED_FOUR_TESTS_NOT_PROMOTED"
            or not _valid_ref(value.get("exclusions"))):
        raise ValueError("fresh-test manifest schema or exclusion reference differs")
    expected = _manifest_value(_exclusion_reference(value["exclusions"]["path"]))
    if value != expected:
        raise ValueError("fresh-test manifest sources, authority, roster or limits differ")
    return value


def _attempt_name(value):
    if (not isinstance(value, str) or len(value) != 12 or not value.startswith("attempt-")
            or not value[8:].isdigit()):
        raise ValueError("invalid fresh-test attempt identifier")
    return value


def _unit_name(value):
    prefix = "phideus-generative-test-"
    if (not isinstance(value, str) or not value.startswith(prefix) or len(value) != len(prefix)+16
            or any(c not in "0123456789abcdef" for c in value[len(prefix):])):
        raise ValueError("invalid fresh-test unit")
    return value


def _service_command(unit, runtime_seconds, stage, attempt):
    _unit_name(unit)
    _attempt_name(attempt)
    if stage not in stage_roster() or type(runtime_seconds) is not int or runtime_seconds <= 0:
        raise ValueError("invalid fresh-test worker stage or runtime")
    gpu = stage["device"] == "cuda:0"
    env = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1", "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "CUDA_VISIBLE_DEVICES": "0" if gpu else "", "PHIDEUS_GENERATIVE_UNIT": unit,
        "TMPDIR": str(TEMP)}
    return ["systemd-run", "--quiet", "--collect", "--wait", "--pipe", f"--unit={unit}",
        "--property=MemoryMax=6G", "--property=MemorySwapMax=0", "--property=OOMPolicy=kill",
        f"--property=RuntimeMaxSec={runtime_seconds}s", f"--property=TimeoutStopSec={GRACE}s",
        "--property=KillMode=control-group", f"--working-directory={ROOT}", "/usr/bin/env",
        *[f"{key}={value}" for key, value in env.items()], str(ROOT/"venv/bin/python"), "-m",
        "experiments.atencion_armonica.run_generative_tests", "--worker", "--attempt", attempt]


def _launches():
    return sorted(CONTROL.glob("attempt-*.launch.json"))


def _stage_path(stage):
    if stage["kind"] == "freeze":
        return _freeze().FREEZE
    if stage["kind"] == "predict":
        return FRESH/stage["split"]/"prediction_seal.json"
    return FRESH/stage["split"]/"evaluation/index.json"


def _stage_reference(stage):
    path = _stage_path(stage)
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"typed stage output missing: {path.relative_to(ROOT)}")
    return reference(path)


def _success_status(stage):
    return {"freeze": "FREEZE_COMPLETE", "predict": "PREDICTIONS_SEALED",
            "evaluate": "EVALUATED", "replay": "REPLAY_VERIFIED"}[stage["kind"]]


def _availability_ok(value):
    return _training()._valid_gpu_availability(value)


def _worker_schema(worker, returncode, stage, output):
    common = {"launch", "stage", "status", "seconds", "peak_rss_bytes",
              "peak_reserved_bytes", "cuda_initialized", "availability"}
    if (type(returncode) is not int or not isinstance(worker, dict) or worker.get("stage") != stage
            or not _number(worker.get("seconds")) or not _number(worker.get("peak_rss_bytes"))
            or not _number(worker.get("peak_reserved_bytes"))
            or type(worker.get("cuda_initialized")) is not bool
            or (stage["device"] == "cpu" and (worker.get("availability") is not None
                                               or worker.get("cuda_initialized") is not False))
            or (stage["device"] == "cuda:0" and not _availability_ok(worker.get("availability")))):
        return False
    if worker.get("status") == _success_status(stage):
        return set(worker) == common | {"output"} and returncode == 0 and worker.get("output") == output
    if worker.get("status") == "PAUSED_RECOVERABLE":
        return set(worker) == common | {"reason"} and returncode == 75
    if worker.get("status") == "FAILED":
        return set(worker) == common | {"reason"} and returncode not in (0, 75)
    return False


def _prefix(manifest_ref, paths):
    manifest = read_manifest(manifest_ref)
    roster = manifest["stages"]
    all_paths = _launches()
    if [path.name for path in all_paths] != [f"attempt-{i:04d}.launch.json" for i in range(len(all_paths))]:
        raise ValueError("fresh-test attempt indices are not contiguous")
    if all_paths[:len(paths)] != list(paths):
        raise ValueError("fresh-test accounting is not an exact ledger prefix")
    total, previous, next_stage, completions, last_status = 0., None, 0, [], None
    for path in paths:
        if next_stage >= len(roster):
            raise ValueError("fresh-test ledger contains attempts after the final stage")
        if last_status == "FAILED":
            raise ValueError("fresh-test ledger continued after a failed attempt")
        launch_ref, launch = reference(path), _read(reference(path))
        stage = roster[next_stage]
        remaining = math.floor(LIMIT_SECONDS-total)
        attempt = path.name.removesuffix(".launch.json")
        keys = {"schema", "manifest", "stage", "previous_exit", "unit", "command",
                "availability", "used_seconds", "remaining_seconds", "runtime_seconds"}
        availability = launch.get("availability")
        if (set(launch) != keys or launch.get("schema") != "generative-evidence-fresh-test-launch-v1"
                or launch.get("manifest") != manifest_ref or launch.get("stage") != stage
                or launch.get("previous_exit") != previous or launch.get("used_seconds") != total
                or launch.get("remaining_seconds") != remaining
                or launch.get("runtime_seconds") != remaining-GRACE
                or launch.get("command") != _service_command(
                    launch.get("unit"), launch.get("runtime_seconds"), stage, attempt)
                or (stage["device"] == "cpu" and availability is not None)
                or (stage["device"] == "cuda:0" and not _availability_ok(availability))):
            raise ValueError("fresh-test launch chain, stage, resources or command differs")
        exit_path = path.with_name(path.name.replace(".launch.json", ".exit.json"))
        worker_path = path.with_name(path.name.replace(".launch.json", ".worker.json"))
        if not exit_path.exists() or not worker_path.exists():
            raise RuntimeError(f"unreconciled fresh-test attempt: {launch.get('unit')}")
        exit_ref, worker_ref = reference(exit_path), reference(worker_path)
        end, worker = _read(exit_ref), _read(worker_ref)
        if (end.get("launch") != launch_ref or end.get("terminal") is not True
                or not _number(end.get("seconds")) or worker.get("launch") != launch_ref):
            raise ValueError("fresh-test attempt lacks matching terminal receipts")
        status = worker.get("status")
        output = _stage_reference(stage) if status == _success_status(stage) else None
        if not _worker_schema(worker, end.get("process_returncode"), stage, output):
            raise ValueError("fresh-test worker receipt is untyped or names another output")
        total += end["seconds"]
        previous = exit_ref
        last_status = status
        if status == _success_status(stage):
            completions.append({"stage": stage, "launch": launch_ref, "exit": exit_ref,
                                "worker": worker_ref, "output": output})
            next_stage += 1
            last_status = "COMPLETE"
    return {"seconds": float(total), "previous_exit": previous, "next_stage": next_stage,
            "completions": completions, "last_status": last_status}


def test_accumulated(manifest_ref):
    return _prefix(manifest_ref, _launches())


def _execute_command(command, unit, launch_ref, exit_path):
    return _training()._execute_command(command, unit, launch_ref, exit_path)


def _execute_stage(stage, manifest, check, progress):
    freeze_ref = reference(_freeze().FREEZE) if _freeze().FREEZE.is_file() else None
    if stage["kind"] == "freeze":
        result = _freeze().freeze_tests(exclusions_ref=manifest["exclusions"], check=check)
        return result["freeze"]
    if freeze_ref is None:
        raise RuntimeError("a completed freeze is required before every fresh-test stage")
    if stage["kind"] == "predict":
        from src.atencion_armonica import generative_evidence_fresh_data as fresh_data
        from src.atencion_armonica import generative_evidence_fresh_inference as fresh_inference
        fresh_data.produce_test(stage["split"], freeze_ref=freeze_ref, check=check)
        return fresh_inference.prepare_and_predict(stage["split"], freeze_ref=freeze_ref,
            device=stage["device"], check=check, progress=progress)
    from src.atencion_armonica import generative_evidence_fresh_evaluation as fresh_evaluation
    if stage["kind"] == "evaluate":
        return fresh_evaluation.evaluate_test(stage["split"], freeze_ref=freeze_ref, check=check)
    if stage["kind"] == "replay":
        return fresh_evaluation.replay_test(stage["split"], freeze_ref=freeze_ref, check=check)
    raise ValueError("unknown fresh-test stage")


def worker_stage(attempt):
    _attempt_name(attempt)
    launch_path = CONTROL/f"{attempt}.launch.json"
    launch_ref, launch = reference(launch_path), _read(reference(launch_path))
    paths = _launches()
    if not paths or paths[-1] != launch_path:
        raise ValueError("fresh-test worker launch is not the unique ledger tail")
    prefix = _prefix(launch["manifest"], paths[:-1])
    roster = stage_roster()
    if prefix["next_stage"] >= len(roster) or prefix["last_status"] == "FAILED":
        raise ValueError("fresh-test worker follows no eligible stage")
    stage = roster[prefix["next_stage"]]
    remaining = math.floor(LIMIT_SECONDS-prefix["seconds"])
    keys = {"schema", "manifest", "stage", "previous_exit", "unit", "command",
            "availability", "used_seconds", "remaining_seconds", "runtime_seconds"}
    if (set(launch) != keys or launch.get("schema") != "generative-evidence-fresh-test-launch-v1"
            or launch.get("stage") != stage or launch.get("previous_exit") != prefix["previous_exit"]
            or launch.get("used_seconds") != prefix["seconds"]
            or launch.get("remaining_seconds") != remaining
            or launch.get("runtime_seconds") != remaining-GRACE
            or launch.get("command") != _service_command(
                launch.get("unit"), launch.get("runtime_seconds"), stage, attempt)
            or (stage["device"] == "cpu" and launch.get("availability") is not None)
            or (stage["device"] == "cuda:0" and not _availability_ok(launch.get("availability")))):
        raise ValueError("fresh-test worker launch, budget or stage differs")

    started, stopped, last_disk = time.monotonic(), [], [0.]

    def stop(signum, frame):
        stopped.append(signum)

    handlers = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    torch = None
    availability = None
    report = {"launch": launch_ref, "stage": stage, "status": "INCOMPLETE"}
    code = 1
    try:
        _training()._verify_service(launch["unit"], launch["runtime_seconds"])
        manifest = read_manifest(launch["manifest"])
        import torch as torch_module
        torch = torch_module
        if torch.cuda.is_initialized():
            raise RuntimeError("CUDA initialized before fresh-test worker device checks")
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        if (any(os.environ.get(name) != "1" for name in
                ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"))
                or os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8"):
            raise RuntimeError("fresh-test worker deterministic environment differs")
        if stage["device"] == "cuda:0":
            if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
                raise RuntimeError("GPU prediction stage lacks its exact visible device")
            availability = _profile().gpu_availability()
            if not _availability_ok(availability):
                raise RuntimeError("GPU is unavailable before fresh-test prediction")
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.cuda.set_device(0)
            fraction = MEMORY_LIMIT/torch.cuda.get_device_properties(0).total_memory
            if not 0 < fraction <= 1:
                raise RuntimeError("fresh-test GPU memory fraction is invalid")
            torch.cuda.set_per_process_memory_fraction(fraction, 0)
            torch.cuda.reset_peak_memory_stats(0)
        elif os.environ.get("CUDA_VISIBLE_DEVICES") != "":
            raise RuntimeError("CPU fresh-test stage must hide CUDA")

        def check():
            now = time.monotonic()
            due = now-last_disk[0] >= 30
            active_torch = torch if stage["device"] == "cuda:0" and torch.cuda.is_initialized() else None
            _training()._worker_check(started, launch["runtime_seconds"], stopped,
                                      torch=active_torch, check_storage=due)
            if due:
                last_disk[0] = now

        def progress(value):
            print(value if isinstance(value, str) else json.dumps(value, sort_keys=True), flush=True)

        check()
        output = _execute_stage(stage, manifest, check, progress)
        expected = _stage_reference(stage)
        if output != expected:
            raise ValueError("fresh-test stage returned another canonical output")
        check()
        report.update(status=_success_status(stage), output=expected)
        code = 0
    except InterruptedError as exc:
        report.update(status="PAUSED_RECOVERABLE", reason=str(exc))
        code = 75
    except BaseException as exc:
        report.update(status="FAILED", reason=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        initialized = bool(torch is not None and torch.cuda.is_initialized())
        peak_reserved = (torch.cuda.max_memory_reserved(0)
                         if initialized and stage["device"] == "cuda:0" else 0)
        report.update(seconds=time.monotonic()-started,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            peak_reserved_bytes=peak_reserved, cuda_initialized=initialized,
            availability=availability)
        write_json(CONTROL/f"{attempt}.worker.json", report)
    return code


def _final_value(manifest_ref, state):
    if state["next_stage"] != 13 or len(state["completions"]) != 13:
        raise RuntimeError("all thirteen typed stages are required for final closure")
    by_split = {}
    for split, _, _ in _freeze().TESTS:
        rows = [row for row in state["completions"] if row["stage"]["split"] == split]
        if [row["stage"]["kind"] for row in rows] != ["predict", "evaluate", "replay"]:
            raise ValueError("final split lacks prediction, evaluation or replay authority")
        if rows[1]["output"] != rows[2]["output"]:
            raise ValueError("read-only replay did not return the exact evaluation index")
        by_split[split] = {"prediction_seal": rows[0]["output"],
            "evaluation": rows[1]["output"], "replay_worker": rows[2]["worker"]}
    return {"schema": "generative-evidence-fresh-tests-complete-v1", "manifest": manifest_ref,
        "status": FINAL_STATUS, "stage_count": 13, "stages": state["completions"],
        "tests": by_split, "accumulated_seconds": state["seconds"],
        "promotion": False, "adaptive_test_reuse": False}


def _publish_final(manifest_ref, state):
    value = _final_value(manifest_ref, state)
    if FINAL.exists():
        ref = reference(FINAL)
        if _read(ref) != value:
            raise ValueError("cannot replace a different fresh-test completion")
        return ref
    if not FRESH.is_dir() or FRESH.is_symlink():
        raise ValueError("fresh-test completion requires the bound canonical fresh root")
    write_json(FINAL, value)
    ref = reference(FINAL)
    if _read(ref) != value:
        raise ValueError("fresh-test completion changed after publication")
    return ref


def verified_tests():
    manifest_ref = reference(MANIFEST)
    read_manifest(manifest_ref)
    state = test_accumulated(manifest_ref)
    if state["seconds"] > LIMIT_SECONDS:
        raise RuntimeError("fresh-test completion exceeded its four-hour budget")
    expected = _final_value(manifest_ref, state)
    ref = reference(FINAL)
    if _read(ref) != expected:
        raise ValueError("fresh-test final index differs from thirteen typed stages")
    return ref


def run_tests():
    training = _training()
    with training._lock(COMMON_LOCK), training._lock(CONTROL/"operator.lock"):
        manifest_ref = reference(MANIFEST)
        manifest = read_manifest(manifest_ref)
        while True:
            state = test_accumulated(manifest_ref)
            if state["seconds"] > LIMIT_SECONDS:
                raise RuntimeError("fresh-test cumulative time exceeded four hours")
            if state["next_stage"] == len(manifest["stages"]):
                _publish_final(manifest_ref, state)
                return verified_tests()
            if state["last_status"] == "FAILED":
                raise RuntimeError("failed fresh-test stage cannot be relaunched automatically")
            remaining = math.floor(LIMIT_SECONDS-state["seconds"])
            if remaining <= GRACE+10:
                raise RuntimeError("fresh-test cumulative time exhausted")
            training.check_disk()
            stage = manifest["stages"][state["next_stage"]]
            availability = None
            if stage["device"] == "cuda:0":
                availability = _profile().gpu_availability()
                if not _availability_ok(availability):
                    raise RuntimeError("GPU is unavailable before fresh-test launch")
            attempt = f"attempt-{len(_launches()):04d}"
            unit = "phideus-generative-test-"+uuid.uuid4().hex[:16]
            runtime_seconds = remaining-GRACE
            command = _service_command(unit, runtime_seconds, stage, attempt)
            launch_path = CONTROL/f"{attempt}.launch.json"
            write_json(launch_path, {"schema": "generative-evidence-fresh-test-launch-v1",
                "manifest": manifest_ref, "stage": stage, "previous_exit": state["previous_exit"],
                "unit": unit, "command": command, "availability": availability,
                "used_seconds": state["seconds"], "remaining_seconds": remaining,
                "runtime_seconds": runtime_seconds})
            launch_ref = reference(launch_path)
            end = _execute_command(command, unit, launch_ref,
                                   CONTROL/f"{attempt}.exit.json")
            if end.get("process_returncode") == 75:
                raise InterruptedError("fresh-test stage paused at a recoverable boundary")
            if end.get("process_returncode") != 0:
                raise RuntimeError("fresh-test stage did not complete")
            # Parse the new receipt and exact canonical output before advancing.
            updated = test_accumulated(manifest_ref)
            if updated["next_stage"] != state["next_stage"]+1:
                raise RuntimeError("fresh-test worker exit did not complete its declared stage")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--initialize", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--exclusions", help="root-relative reviewed inventory.json")
    parser.add_argument("--attempt", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.initialize:
        if args.exclusions is None or args.attempt is not None:
            parser.error("--initialize requires --exclusions and rejects --attempt")
        result = initialize_tests(args.exclusions)
    elif args.run:
        if args.exclusions is not None or args.attempt is not None:
            parser.error("--run does not accept --exclusions or --attempt")
        result = run_tests()
    else:
        if args.attempt is None or args.exclusions is not None:
            parser.error("internal worker requires --attempt only")
        sys.exit(worker_stage(args.attempt))
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
