"""Bounded CPU calibration selection after the 27-cell training closure.

Public modes only initialize or run the recoverable selection stage.  The
parent handles small metadata, hashes, accounting and process supervision;
all 27-cell reopening and selection work stays inside the bounded worker.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
import uuid

from src.atencion_armonica.generative_evidence_reuse import ROOT, VerifiedBytes
from src.atencion_armonica.generative_evidence_storage import write_json


TEMP = ROOT/".agent-work/phideus-generative-evidence-20260909"
CONTROL = TEMP/"selection-control"
MANIFEST = CONTROL/"manifest.json"
PREPARATION_LOCK = TEMP/"preparation-control/operator.lock"
OUTPUT = ROOT/"data/atencion_armonica/generative_evidence_reader_v1/selection/index.json"
PROTOCOL = ROOT/"experiments/atencion_armonica/PROTOCOL_GENERATIVE_EVIDENCE_READER.md"
TRAINING_CLI = ROOT/"experiments/atencion_armonica/run_generative_training.py"
SELECTION_SOURCE = ROOT/"src/atencion_armonica/generative_evidence_selection.py"
EVALUATION_SOURCE = ROOT/"src/atencion_armonica/generative_evidence_evaluation.py"
PREPARATION_CLI = ROOT/"experiments/atencion_armonica/prepare_generative_evidence.py"
REUSE_SOURCE = ROOT/"src/atencion_armonica/generative_evidence_reuse.py"
STORAGE_SOURCE = ROOT/"src/atencion_armonica/generative_evidence_storage.py"
LIMIT_SECONDS = 12*3600
GIB = 1024**3
MEMORY_LIMIT = 6*GIB
STORAGE_LIMIT = 60*GIB
FREE_MIN = 80*GIB
GRACE = 30
STATUS = "CALIBRATION_SELECTED"


def _training():
    from experiments.atencion_armonica import run_generative_training
    return run_generative_training


def _preparation():
    from experiments.atencion_armonica import prepare_generative_evidence
    return prepare_generative_evidence


def reference(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": hashlib.sha256(raw).hexdigest()}


def _valid_ref(value):
    return (isinstance(value, dict) and set(value) == {"path", "sha256"}
            and isinstance(value["path"], str) and bool(value["path"])
            and isinstance(value["sha256"], str) and len(value["sha256"]) == 64
            and all(c in "0123456789abcdef" for c in value["sha256"]))


def _number(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _read(ref):
    return VerifiedBytes(ROOT).json(ref)


def _source_paths():
    return sorted({Path(__file__).resolve(), TRAINING_CLI, SELECTION_SOURCE, EVALUATION_SOURCE,
                   PROTOCOL, PREPARATION_CLI, REUSE_SOURCE, STORAGE_SOURCE})


def sources():
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in _source_paths()}


def runtime():
    return _training().runtime()


def _lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BaseException:
        handle.close()
        raise
    return handle


def _attempt_name(value):
    if (not isinstance(value, str) or len(value) != 12 or not value.startswith("attempt-")
            or not value[8:].isdigit()):
        raise ValueError("invalid selection attempt identifier")
    return value


def _unit_name(value):
    prefix = "phideus-generative-selection-"
    suffix = value[len(prefix):] if isinstance(value, str) and value.startswith(prefix) else ""
    if len(suffix) != 16 or any(c not in "0123456789abcdef" for c in suffix):
        raise ValueError("invalid owned selection unit")
    return value


def _launches():
    return sorted(CONTROL.glob("attempt-*.launch.json"))


def _training_boundary():
    """Authenticate only training metadata/index; never reopen cell payloads."""
    training = _training()
    manifest_ref = training.reference(training.TRAINING_MANIFEST)
    manifest, _ = training.read_training_manifest(manifest_ref)
    total, last_exit, per_cell, cell_parents = training.training_accumulated(manifest_ref)
    index_ref = training.reference(training.TRAINING_ROOT/"index.json")
    index = _read(index_ref)
    expected_keys = {"schema", "status", "manifest", "cells", "cell_count",
                     "accumulated_seconds", "test_access"}
    roster = manifest["roster"]
    expected_paths = [
        f"data/atencion_armonica/generative_evidence_reader_v1/training/{c['arm']}"
        f"/cp_{c['checkpoint_seed']}/seed_{c['reader_seed']}/complete.json" for c in roster]
    cells = index.get("cells")
    if (set(index) != expected_keys or index.get("schema") != "generative-evidence-training-complete-v1"
            or index.get("status") != "TRAINED_NOT_SELECTED" or index.get("manifest") != manifest_ref
            or index.get("cell_count") != 27 or index.get("accumulated_seconds") != total
            or index.get("test_access") is not False or not isinstance(cells, list) or len(cells) != 27
            or [row.get("cell") for row in cells] != roster
            or any(set(row) != {"cell", "complete"} or not _valid_ref(row["complete"]) for row in cells)
            or [row["complete"]["path"] for row in cells] != expected_paths
            or len({json.dumps(row["complete"], sort_keys=True) for row in cells}) != 27):
        raise ValueError("training aggregate metadata is not the exact 27-cell closure")
    ids = [c["cell_id"] for c in roster]
    if (not _number(total) or total >= LIMIT_SECONDS or not _valid_ref(last_exit)
            or set(per_cell) != set(ids) or set(cell_parents) != set(ids)
            or any(not _number(per_cell[i]) or not _valid_ref(cell_parents[i]) for i in ids)):
        raise RuntimeError("training accounting is not a completed prefix with selection budget")
    return {"manifest": manifest_ref, "index": index_ref, "last_exit": last_exit,
            "accumulated_seconds": float(total)}


def initialize_selection():
    with _lock(PREPARATION_LOCK), _lock(CONTROL/"operator.lock"):
        if MANIFEST.exists():
            raise FileExistsError("selection manifest already exists; use --run")
        training = _training_boundary()
        value = {"schema": "generative-evidence-selection-stage-v1",
            "stage": "CALIBRATION_SELECTION_NOT_TEST_AUTHORIZED", "device": "cpu",
            "sources": sources(), "runtime": runtime(), "training": training,
            "limit_seconds": LIMIT_SECONDS, "memory_max_bytes": MEMORY_LIMIT,
            "storage_max_bytes": STORAGE_LIMIT, "free_min_bytes": FREE_MIN,
            "output": {"path": OUTPUT.relative_to(ROOT).as_posix(),
                       "status": "CALIBRATION_SELECTED_NOT_TEST_AUTHORIZED", "test_access": False},
            "test_access": False}
        write_json(MANIFEST, value)
        return reference(MANIFEST)


def read_manifest(ref):
    if ref.get("path") != MANIFEST.relative_to(ROOT).as_posix():
        raise ValueError("unexpected selection manifest path")
    value = _read(ref)
    keys = {"schema", "stage", "device", "sources", "runtime", "training", "limit_seconds",
            "memory_max_bytes", "storage_max_bytes", "free_min_bytes", "output", "test_access"}
    expected_output = {"path": OUTPUT.relative_to(ROOT).as_posix(),
                       "status": "CALIBRATION_SELECTED_NOT_TEST_AUTHORIZED", "test_access": False}
    if (set(value) != keys or value["schema"] != "generative-evidence-selection-stage-v1"
            or value["stage"] != "CALIBRATION_SELECTION_NOT_TEST_AUTHORIZED"
            or value["device"] != "cpu" or value["sources"] != sources()
            or value["runtime"] != runtime() or value["training"] != _training_boundary()
            or value["limit_seconds"] != LIMIT_SECONDS or value["memory_max_bytes"] != MEMORY_LIMIT
            or value["storage_max_bytes"] != STORAGE_LIMIT or value["free_min_bytes"] != FREE_MIN
            or value["output"] != expected_output or value["test_access"] is not False):
        raise ValueError("selection stage identity, sources, runtime or training boundary differs")
    return value


def _service_command(unit, runtime_seconds, attempt):
    _attempt_name(attempt)
    _unit_name(unit)
    if type(runtime_seconds) is not int or runtime_seconds <= 0:
        raise ValueError("selection runtime must be a positive integer")
    env = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "NUMEXPR_NUM_THREADS": "1", "CUDA_VISIBLE_DEVICES": "",
           "PHIDEUS_GENERATIVE_SELECTION_UNIT": unit, "TMPDIR": str(TEMP)}
    return ["systemd-run", "--quiet", "--collect", "--wait", "--pipe", f"--unit={unit}",
        "--property=MemoryMax=6G", "--property=MemorySwapMax=0", "--property=OOMPolicy=kill",
        f"--property=RuntimeMaxSec={runtime_seconds}s", f"--property=TimeoutStopSec={GRACE}s",
        "--property=KillMode=control-group", f"--working-directory={ROOT}", "/usr/bin/env",
        *[f"{k}={v}" for k, v in env.items()], str(ROOT/"venv/bin/python"), "-m",
        "experiments.atencion_armonica.run_generative_selection", "--worker", "--attempt", attempt]


def _execute_command(command, unit, launch_ref, exit_path):
    started, requested = time.monotonic(), []
    handlers = {s: signal.signal(s, lambda signum, frame: requested.append(signum))
                for s in (signal.SIGINT, signal.SIGTERM)}
    result = {"launch": launch_ref}
    try:
        result["process_returncode"] = _preparation().run_command(command, unit, requested)
    finally:
        probe = _preparation().terminal_state(unit)
        result.update(seconds=time.monotonic()-started, terminal=probe["terminal"],
                      systemd_state=probe["state"], systemd_probe_returncode=probe["returncode"],
                      systemd_stderr=probe["stderr"])
        write_json(exit_path, result)
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
    if result["terminal"] is not True:
        raise RuntimeError("selection worker has not reached an authoritative terminal state")
    return result


def _verify_service(unit, runtime_seconds):
    group = Path("/proc/self/cgroup").read_text().strip().split("::")[-1]
    if (os.environ.get("PHIDEUS_GENERATIVE_SELECTION_UNIT") != unit
            or Path(group).name != unit+".service"):
        raise RuntimeError("selection worker requires its exact supervised service")
    cgroup = Path("/sys/fs/cgroup")/group.lstrip("/")
    if ((cgroup/"memory.max").read_text().strip() != str(MEMORY_LIMIT)
            or (cgroup/"memory.swap.max").read_text().strip() != "0"):
        raise RuntimeError("selection worker memory guard differs")
    properties = subprocess.check_output(["systemctl", "show", unit+".service", "--property=KillMode",
        "--property=OOMPolicy"], text=True, timeout=5)
    if dict(line.split("=", 1) for line in properties.splitlines()) != {
            "KillMode": "control-group", "OOMPolicy": "kill"}:
        raise RuntimeError("selection worker kill/oom guard differs")
    _preparation().verify_deadline(unit, runtime_seconds)


def _worker_check(started, runtime_seconds, stopped, last_disk):
    now = time.monotonic()
    if stopped or now-started >= runtime_seconds-GRACE:
        raise InterruptedError("recoverable selection stop at an immutable boundary")
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 > MEMORY_LIMIT:
        raise RuntimeError("selection worker RSS envelope exceeded")
    if now-last_disk[0] >= 30:
        _training().check_disk()
        last_disk[0] = now


def _worker_schema(worker, process_returncode):
    base = {"launch", "status", "seconds", "peak_rss_bytes", "cuda_initialized"}
    if (type(process_returncode) is not int or not isinstance(worker, dict)
            or not _number(worker.get("seconds"))
            or not _number(worker.get("peak_rss_bytes")) or worker.get("cuda_initialized") is not False):
        return False
    if worker.get("status") == STATUS:
        return set(worker) == base | {"selection"} and process_returncode == 0 and _valid_ref(worker["selection"])
    if worker.get("status") == "PAUSED_RECOVERABLE":
        return set(worker) == base | {"reason"} and process_returncode == 75
    if worker.get("status") == "FAILED":
        return set(worker) == base | {"reason"} and process_returncode not in (0, 75)
    return False


def _prefix(manifest_ref, paths):
    manifest = read_manifest(manifest_ref)
    total = manifest["training"]["accumulated_seconds"]
    previous = manifest["training"]["last_exit"]
    all_paths = _launches()
    if [p.name for p in all_paths] != [f"attempt-{i:04d}.launch.json" for i in range(len(all_paths))]:
        raise ValueError("selection attempt indices are not contiguous")
    if all_paths[:len(paths)] != list(paths):
        raise ValueError("selection accounting is not an exact ledger prefix")
    for path in paths:
        launch_ref, launch = reference(path), _read(reference(path))
        remaining = math.floor(LIMIT_SECONDS-total)
        attempt = path.name.removesuffix(".launch.json")
        keys = {"schema", "manifest", "previous_exit", "unit", "command", "used_seconds",
                "remaining_seconds", "runtime_seconds"}
        if (set(launch) != keys or launch.get("schema") != "generative-evidence-selection-launch-v1"
                or launch.get("manifest") != manifest_ref or launch.get("previous_exit") != previous
                or launch.get("used_seconds") != total or launch.get("remaining_seconds") != remaining
                or launch.get("runtime_seconds") != remaining-GRACE
                or launch.get("command") != _service_command(
                    launch.get("unit"), launch.get("runtime_seconds"), attempt)):
            raise ValueError("selection launch chain, budget or command differs")
        exit_path = path.with_name(path.name.replace(".launch.json", ".exit.json"))
        worker_path = path.with_name(path.name.replace(".launch.json", ".worker.json"))
        if not exit_path.exists() or not worker_path.exists():
            raise RuntimeError(f"unreconciled selection attempt: {launch.get('unit')}")
        exit_ref, end = reference(exit_path), _read(reference(exit_path))
        worker = _read(reference(worker_path))
        if (end.get("launch") != launch_ref or end.get("terminal") is not True
                or not _number(end.get("seconds")) or not _worker_schema(worker, end.get("process_returncode"))
                or worker.get("launch") != launch_ref):
            raise ValueError("selection attempt lacks matching terminal and typed receipts")
        total += end["seconds"]
        previous = exit_ref
    return float(total), previous


def selection_accumulated(manifest_ref):
    return _prefix(manifest_ref, _launches())


def _output_reference(manifest):
    ref = reference(OUTPUT)
    value = _read(ref)
    keys = {"schema", "binding", "status", "training_accumulated_seconds", "cell_count",
            "calibration_record_count", "calibration_records", "initial_models", "selection",
            "selected_states", "test_access"}
    binding = {"selection_manifest": reference(MANIFEST),
               "training_manifest": manifest["training"]["manifest"],
               "training_complete": manifest["training"]["index"]}
    arm_roster = [cell["arm"] for cell in _training().cell_roster()]
    arms = list(dict.fromkeys(arm_roster))
    if len(arm_roster) != 27 or len(arms) != 3:
        raise ValueError("training arm roster differs before selection verification")
    if (set(value) != keys or value["schema"] != "generative-evidence-calibration-selection-v1"
            or value["binding"] != binding or value["status"] != manifest["output"]["status"]
            or value["training_accumulated_seconds"] != manifest["training"]["accumulated_seconds"]
            or value["cell_count"] != 27 or value["calibration_record_count"] != 270
            or not isinstance(value["calibration_records"], list) or len(value["calibration_records"]) != 270
            or set(value["selected_states"]) != set(arms)
            or any(len(v.get("cells", [])) != 9 for v in value["selected_states"].values())
            or value["test_access"] is not False):
        raise ValueError("selection output is not the exact non-test-authorized closure")
    return ref


def worker_selection(attempt):
    _attempt_name(attempt)
    started, stopped, last_disk = time.monotonic(), [], [0.]
    launch_path = CONTROL/f"{attempt}.launch.json"
    launch_ref, report, code, torch = None, {"launch": None, "status": "INCOMPLETE"}, 1, None
    def stop(signum, frame):
        stopped.append(signum)
        raise InterruptedError("recoverable selection stop requested")
    handlers = {sig: signal.signal(sig, stop)
                for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        launch_ref = reference(launch_path)
        report["launch"] = launch_ref
        launch = _read(launch_ref)
        keys = {"schema", "manifest", "previous_exit", "unit", "command", "used_seconds",
                "remaining_seconds", "runtime_seconds"}
        if (set(launch) != keys or launch["schema"] != "generative-evidence-selection-launch-v1"
                or launch["command"] != _service_command(launch["unit"], launch["runtime_seconds"], attempt)
                or not _launches() or _launches()[-1] != launch_path):
            raise ValueError("selection worker launch is not the exact ledger tail")
        used, previous = _prefix(launch["manifest"], _launches()[:-1])
        remaining = math.floor(LIMIT_SECONDS-used)
        if (launch["previous_exit"] != previous or launch["used_seconds"] != used
                or launch["remaining_seconds"] != remaining
                or launch["runtime_seconds"] != remaining-GRACE):
            raise ValueError("selection worker budget prefix differs")
        _verify_service(launch["unit"], launch["runtime_seconds"])
        if (os.environ.get("CUDA_VISIBLE_DEVICES") != ""
                or any(os.environ.get(k) != "1" for k in
                       ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"))):
            raise RuntimeError("selection worker must remain CPU-only and single-threaded")
        manifest = read_manifest(launch["manifest"])
        check = lambda: _worker_check(started, launch["runtime_seconds"], stopped, last_disk)
        check()
        import torch as torch_module
        torch = torch_module
        if torch.cuda.is_initialized():
            raise RuntimeError("CUDA initialized before CPU selection")
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        from src.atencion_armonica import generative_evidence_selection as selection
        local = selection.run_selection(check=check, stage_manifest=launch["manifest"])
        selected = _output_reference(manifest)
        if (local.get("path") != "index.json" or local.get("sha256") != selected["sha256"]
                or local.get("bytes") != OUTPUT.stat().st_size or torch.cuda.is_initialized()):
            raise ValueError("selection worker output reference or CPU-only state differs")
        check()
        report.update(status=STATUS, selection=selected)
        code = 0
    except InterruptedError as exc:
        if launch_ref is None:
            launch_ref = reference(launch_path)
            report["launch"] = launch_ref
        report.update(status="PAUSED_RECOVERABLE", reason=str(exc))
        code = 75
    except BaseException as exc:
        report.update(status="FAILED", reason=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        report.update(seconds=time.monotonic()-started,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            cuda_initialized=bool(torch is not None and torch.cuda.is_initialized()))
        write_json(CONTROL/f"{attempt}.worker.json", report)
    return code


def _attempt_state(path):
    launch_ref, launch = reference(path), _read(reference(path))
    exit_path = path.with_name(path.name.replace(".launch.json", ".exit.json"))
    worker_path = path.with_name(path.name.replace(".launch.json", ".worker.json"))
    if not exit_path.exists() or not worker_path.exists():
        raise RuntimeError("selection attempt lacks terminal/worker receipts; do not relaunch ambiguously")
    end, worker = _read(reference(exit_path)), _read(reference(worker_path))
    if (end.get("launch") != launch_ref or worker.get("launch") != launch_ref
            or end.get("terminal") is not True or not _worker_schema(worker, end.get("process_returncode"))):
        raise ValueError("selection attempt terminal and typed states differ")
    return launch, end, worker


def verified_selection():
    manifest_ref = reference(MANIFEST)
    manifest = read_manifest(manifest_ref)
    total, _ = selection_accumulated(manifest_ref)
    if total > LIMIT_SECONDS:
        raise RuntimeError("completed selection exceeded the shared training/selection budget")
    launches = _launches()
    if not launches:
        raise RuntimeError("selection has no supervised attempt")
    _, end, worker = _attempt_state(launches[-1])
    selected = _output_reference(manifest)
    if end["process_returncode"] != 0 or worker["status"] != STATUS or worker["selection"] != selected:
        raise RuntimeError("selection is not typed terminal CALIBRATION_SELECTED")
    return selected


def run_selection():
    with _lock(PREPARATION_LOCK), _lock(CONTROL/"operator.lock"):
        manifest_ref = reference(MANIFEST)
        read_manifest(manifest_ref)
        used, parent = selection_accumulated(manifest_ref)
        launches = _launches()
        if launches:
            _, end, worker = _attempt_state(launches[-1])
            if worker["status"] == STATUS:
                return verified_selection()
            if worker["status"] != "PAUSED_RECOVERABLE" or end["process_returncode"] != 75:
                raise RuntimeError("failed or ambiguous selection cannot be relaunched automatically")
        remaining = math.floor(LIMIT_SECONDS-used)
        if remaining <= GRACE+10:
            raise RuntimeError("shared training/selection cumulative time exhausted")
        _training().check_disk()
        attempt = f"attempt-{len(launches):04d}"
        unit = "phideus-generative-selection-"+uuid.uuid4().hex[:16]
        runtime_seconds = remaining-GRACE
        command = _service_command(unit, runtime_seconds, attempt)
        launch_path = CONTROL/f"{attempt}.launch.json"
        write_json(launch_path, {"schema": "generative-evidence-selection-launch-v1",
            "manifest": manifest_ref, "previous_exit": parent, "unit": unit, "command": command,
            "used_seconds": used, "remaining_seconds": remaining, "runtime_seconds": runtime_seconds})
        launch_ref = reference(launch_path)
        end = _execute_command(command, unit, launch_ref,
                               CONTROL/f"{attempt}.exit.json")
        if end["process_returncode"] != 0:
            raise RuntimeError("calibration selection did not complete")
        return verified_selection()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--initialize", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--attempt", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.initialize:
        result = initialize_selection()
    elif args.run:
        result = run_selection()
    else:
        sys.exit(worker_selection(args.attempt))
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
