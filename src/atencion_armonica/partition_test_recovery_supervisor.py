"""Owned CPU-only supervision for the closed, versioned test recovery."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from . import learned_partition_provenance as p
from . import partition_test_recovery_gate as gate
from .learned_partition_metrics import TESTS
from .learned_partition_supervisor import process_measurement, process_identity, _terminate
from .structured_source_artifacts import safe_member, write_json, mark_failure

INPUTS = {"split", "authorization", "data", "logits", "scored", "recovery_authorization"}
ARGUMENTS = {"normalized": INPUTS | {"shard", "normalizers", "train"},
             "inference": INPUTS | {"normalized", "gpu_grant"},
             "evaluation": INPUTS | {"normalized", "predictions"}}


def validate_request(ref):
    request = p.read_reference(ref)
    root = p.ROOT/gate.RECOVERY
    if (set(request) != {"schema", "request_id", "operation", "output", "arguments", "execution_contract"}
            or request["schema"] != "partition-test-memory-recovery-request-v1"
            or not isinstance(request["request_id"], str) or not request["request_id"].strip()
            or request["operation"] not in ARGUMENTS or not isinstance(request["arguments"], dict)
            or set(request["arguments"]) != ARGUMENTS[request["operation"]]):
        raise ValueError("closed recovery request schema differs")
    arguments = request["arguments"]
    if arguments["split"] not in TESTS or arguments["authorization"] != gate.FROZEN["test_authorization"]:
        raise PermissionError("recovery is only for the four declared frozen tests")
    if request["operation"] == "normalized" and (type(arguments["shard"]) is not int or arguments["shard"] != 0):
        raise ValueError("test normalization must use shard zero")
    if request["operation"] == "inference" and arguments["gpu_grant"] is not None:
        raise PermissionError("the versioned recovery heads are CPU-only")
    auth, _ = gate.verify_authorization(arguments["recovery_authorization"])
    if request["execution_contract"] != auth["contract"]:
        raise ValueError("request uses a different recovery executor")
    output_root = root/"outputs"/arguments["split"]
    output = gate.canonical_path(request["output"], output_root)
    if output.exists() or output.with_name(output.name+".FAILURE.json").exists():
        raise FileExistsError("recovery never overwrites or completes another attempt")
    gate.canonical_path(ref["path"], root/"requests")
    return request, output


def execute(ref):
    request, output = validate_request(ref)
    from .partition_test_recovery import normalize_shard, infer_test, evaluate_test
    fn = {"normalized": normalize_shard, "inference": infer_test, "evaluation": evaluate_test}[request["operation"]]
    result = fn(output, **request["arguments"])
    if p.read_reference(ref) != request:
        raise ValueError("recovery request changed during execution")
    gate.verify_contract(request["execution_contract"])
    return result


def supervise_process(command, *, env, control, wall, rss_limit, payload):
    """One directly owned child and inherited pipe; also exercised by fixtures."""
    started = time.monotonic()
    read_fd, write_fd = os.pipe()
    child, result, failure, peak = None, None, None, 0
    terminal, exit_code = True, None
    try:
        with (control/"stdout.log").open("xb") as stdout, (control/"stderr.log").open("xb") as stderr:
            child = subprocess.Popen([*command, str(read_fd)], env=env, pass_fds=(read_fd,),
                stdout=stdout, stderr=stderr, cwd=p.ROOT, start_new_session=True)
            os.close(read_fd)
            read_fd = None
            write_json(control/"started.json", {**process_identity(child.pid), "parent_pid": os.getpid(),
                "wall_limit_seconds": wall, "rss_limit_bytes": rss_limit, "gpu_limit_bytes": 0})
            with os.fdopen(write_fd, "w") as pipe:
                write_fd = None
                json.dump({**payload, "supervisor_pid": os.getpid()}, pipe)
            while child.poll() is None:
                rss, _ = process_measurement(child.pid, gpu=False)
                peak = max(peak, rss)
                if time.monotonic()-started > wall or peak >= rss_limit:
                    raise TimeoutError("recovery worker exceeded its original CPU envelope")
                time.sleep(.2)
            if child.wait() != 0:
                raise RuntimeError(f"recovery worker exit {child.returncode}")
        envelope = json.loads((control/"stdout.log").read_bytes())
        if (set(envelope) != {"result", "peak_rss_bytes"} or type(envelope["peak_rss_bytes"]) is not int
                or envelope["peak_rss_bytes"] <= 0):
            raise ValueError("worker omitted its final high-water RSS")
        peak = max(peak, envelope["peak_rss_bytes"])
        if time.monotonic()-started > wall or peak >= rss_limit:
            raise TimeoutError("worker final HWM or elapsed time exceeded the envelope")
        result = envelope["result"]
    except BaseException as exc:
        failure = exc
    finally:
        for fd in (read_fd, write_fd):
            if fd is not None:
                os.close(fd)
        if child is not None:
            terminal = False
            try:
                exit_code = _terminate(child)
                terminal = True
            except BaseException as exc:
                failure = exc
    return {"worker_pid": None if child is None else child.pid,
        "worker_terminal_confirmed": terminal, "worker_exit_code": exit_code,
        "result": result, "seconds": time.monotonic()-started,
        "observed_peak_rss_bytes": peak, "observed_peak_gpu_bytes": 0,
        "error": None if failure is None else repr(failure)}


def supervised(ref):
    request, output = validate_request(ref)
    root = p.ROOT/gate.RECOVERY
    staging = root/"supervision"
    gate.canonical_path(staging, root)
    staging.mkdir(parents=True, exist_ok=True)
    control = Path(tempfile.mkdtemp(prefix="supervisor-", dir=staging))
    wall, rss_limit = gate.LIMITS[request["operation"]]
    write_json(control/"request.json", {"reference": ref, "request": request, "owner": "/root"})
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1", TMPDIR=str(p.ROOT/gate.LOCAL))
    for key in ("PYTHONMALLOC", "MALLOC_TRIM_THRESHOLD_", "MALLOC_MMAP_THRESHOLD_", "MALLOC_ARENA_MAX"):
        env.pop(key, None)
    command = [sys.executable, str(p.ROOT/"experiments/atencion_armonica/run_partition_test_recovery.py"), "--worker-fd"]
    validate_request(ref)
    outcome = supervise_process(command, env=env, control=control, wall=wall, rss_limit=rss_limit,
                                payload={"reference": ref})
    try:
        if outcome["error"] is not None:
            raise RuntimeError(outcome["error"])
        if not outcome["worker_terminal_confirmed"] or outcome["worker_exit_code"] != 0:
            raise RuntimeError("recovery worker completion is not confirmed")
        if p.verify_reference(outcome["result"]) != output/"manifest.json":
            raise ValueError("recovery worker returned another output")
        if p.read_reference(ref) != request:
            raise ValueError("recovery request changed after worker completion")
        gate.verify_contract(request["execution_contract"])
        gate.checked_bundle(outcome["result"], request["operation"], request["arguments"]["recovery_authorization"])
        if request["operation"] == "evaluation":
            gate.checked_evaluation(outcome["result"], **request["arguments"])
    except BaseException as exc:
        outcome["error"] = repr(exc)
        if output.is_dir():
            mark_failure(output, exc)
        mark_failure(control, exc)
    receipt = {"schema": "partition-test-memory-recovery-terminal-v1", **outcome,
        "status": "COMPLETE" if outcome["error"] is None else "FAILED",
        "request": ref, "request_id": request["request_id"], "operation": request["operation"],
        "output": request["output"], "execution_contract": request["execution_contract"],
        "budget": None, "recovery_status": "NOT_TRAINING"}
    write_json(control/"terminal.json", receipt)
    if receipt["status"] != "COMPLETE":
        raise RuntimeError(f"recovery stopped; inspect {control.relative_to(p.ROOT)}/terminal.json")
    return {"result": outcome["result"], "supervisor": p.reference(control/"terminal.json")}
