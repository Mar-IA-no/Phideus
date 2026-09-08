"""Supervised structured-reader stages from an immutable JSON request.

No default action, no training, and no CLI switch that opens tests without
their hash-bound audited freeze. Supervisor logs live in project-owned staging.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.atencion_armonica import structured_source_gate as gate
from src.atencion_armonica.structured_source_artifacts import mark_failure, safe_member, write_json

OPERATIONS = {"prepare", "forward", "analyze", "cpu_preflight", "forward_profile",
              "authorize_calibration", "freeze", "authorize_test"}
FILE_OPERATIONS = {"authorize_calibration", "freeze", "authorize_test"}


def validate_request(ref):
    request = gate.read_reference(ref)
    if (set(request) != {"operation", "output", "arguments"} or request["operation"] not in OPERATIONS
            or not isinstance(request["arguments"], dict)):
        raise ValueError("invalid stage request schema")
    output = safe_member(ROOT, request["output"])
    if not output.is_relative_to(ROOT/"data/atencion_armonica"):
        raise ValueError("scientific output must belong to the project's harmonic data tree")
    if output.exists():
        raise FileExistsError("stage output already exists; never replace an earlier attempt")
    return request, output


def execute(ref):
    request, output = validate_request(ref)
    from src.atencion_armonica.structured_source_data import prepare_split
    from src.atencion_armonica.structured_source_runner import forward_split, analyze_split
    from src.atencion_armonica.structured_source_profile import cpu_preflight, forward_profile
    functions = {"prepare": prepare_split, "forward": forward_split, "analyze": analyze_split,
                 "cpu_preflight": cpu_preflight, "forward_profile": forward_profile,
                 "authorize_calibration": gate.create_calibration_authorization,
                 "freeze": gate.create_freeze, "authorize_test": gate.create_test_authorization}
    result = functions[request["operation"]](output, **request["arguments"])
    if gate.read_reference(ref) != request:
        raise ValueError("stage request changed during execution")
    return result if result is not None else gate.reference(output)


def limits(operation):
    if operation == "cpu_preflight":
        return 120., 1024**3
    if operation in {"forward", "forward_profile"}:
        return 600., 4*1024**3  # Host RSS; separate worker checks <2GiB VRAM.
    return 1200., 2*1024**3


def supervised(ref):
    request, output = validate_request(ref)
    ceiling, rss_ceiling = limits(request["operation"])
    staging = ROOT/".agent-work/phideus-structured-reader-20260908"
    if not staging.is_dir():
        raise ValueError("project-owned request staging must already exist")
    control = Path(tempfile.mkdtemp(prefix="supervisor-", dir=staging))
    write_json(control/"request.json", {"reference": ref, "request": request,
                                       "wall_limit_seconds": ceiling, "rss_limit_bytes": rss_ceiling})
    read_fd, write_fd = os.pipe()
    child, started, peak = None, time.monotonic(), 0
    try:
        environment = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        if request["operation"] not in {"forward", "forward_profile"}:
            environment["CUDA_VISIBLE_DEVICES"] = ""  # CPU stages cannot initialize a GPU.
        with (control/"stdout.log").open("xb") as stdout, (control/"stderr.log").open("xb") as stderr:
            child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--worker-fd", str(read_fd)],
                                     env=environment, pass_fds=(read_fd,), stdout=stdout, stderr=stderr)
            os.close(read_fd)
            read_fd = None
            with os.fdopen(write_fd, "w") as pipe:
                write_fd = None
                json.dump(ref, pipe)
            while child.poll() is None:
                try:
                    status = Path(f"/proc/{child.pid}/status").read_text()
                    hwm = next((int(line.split()[1])*1024 for line in status.splitlines() if line.startswith("VmHWM:")), 0)
                    peak = max(peak, hwm)
                except FileNotFoundError:
                    pass  # Child may have just exited; wait() below is authoritative.
                if time.monotonic()-started > ceiling or peak >= rss_ceiling:
                    raise TimeoutError("supervised stage exhausted wall time or host RSS envelope")
                time.sleep(.1)
            if child.wait() != 0:
                raise RuntimeError(f"stage worker exit {child.returncode}; see {control.relative_to(ROOT)}")
        elapsed = time.monotonic()-started
        # The standalone supervisor owns one worker. Child HWM also covers a
        # peak between polls or a worker already reaped by poll()/wait().
        peak = max(peak, resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss*1024)
        if elapsed > ceiling or peak >= rss_ceiling:
            raise TimeoutError("finished stage exceeded wall time or host RSS envelope")
        result = json.loads((control/"stdout.log").read_text())
        result_path = gate.verify_reference(result)
        expected_path = output if request["operation"] in FILE_OPERATIONS else output/"manifest.json"
        if result_path.resolve() != expected_path.resolve():
            raise ValueError("worker returned an artifact other than the requested stage output")
        write_json(control/"result.json", {"status": "COMPLETE", "result": result,
                                           "seconds": elapsed, "observed_peak_rss_bytes": peak})
        return {"result": result, "supervisor": gate.reference(control/"result.json")}
    except BaseException as exc:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        if request["operation"] not in FILE_OPERATIONS and output.is_dir():
            mark_failure(output, exc)
        elif request["operation"] in FILE_OPERATIONS and output.is_file():
            # Revoke only this receipt, never every receipt sharing its parent.
            marker = output.with_name(f"{output.name}.FAILURE.json")
            if not marker.exists():
                write_json(marker, {"status": "INCOMPLETE", "error": repr(exc), "supervisor": control.relative_to(ROOT).as_posix()})
        mark_failure(control, exc)
        raise
    finally:
        for fd in (read_fd, write_fd):
            if fd is not None:
                os.close(fd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path)
    parser.add_argument("--sha256")
    parser.add_argument("--worker-fd", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_fd is not None:
        if args.request is not None or args.sha256 is not None:
            parser.error("worker uses only its inherited request pipe")
        with os.fdopen(args.worker_fd) as handle:
            ref = json.load(handle)
        print(json.dumps(execute(ref), sort_keys=True))
    else:
        if args.request is None or args.sha256 is None:
            parser.error("explicit request path and expected SHA256 are required")
        ref = {"path": args.request.resolve().relative_to(ROOT).as_posix(), "sha256": args.sha256}
        print(json.dumps(supervised(ref), sort_keys=True))
