"""Owned process supervision with immutable terminal receipts.

The worker receives a request through an inherited pipe. Receipts distinguish
confirmed process termination from an experiment's scientific artifact state.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from contextlib import ExitStack

from . import learned_partition_provenance as p
from .structured_source_artifacts import mark_failure, safe_member, write_json

OUTPUTS = p.ROOT/"data/atencion_armonica/learned_partition_reader_v1"
STAGING = OUTPUTS/"supervision"
TEMPORARIES = p.ROOT/".agent-work/phideus-learned-reader-20260908"
ARGUMENTS = {
    "geometry_profile": {"audit"},
    "training_cpu_profile": {"audit", "geometry"},
    "training_gpu_profile": {"audit", "geometry", "gpu_grant"},
    "authorize_data": {"implementation_audit", "profiles"},
    "authorize_reuse": {"implementation_audit", "profiles", "reuse_plan_audit"},
    "import_prepared": {"authorization"},
    "prepare": {"split", "shard", "authorization", "previous", "earlier_shards"},
    "aggregate_data": {"split", "authorization", "previous", "shards"},
    "forward": {"split", "shard", "authorization", "data", "gpu_grant"},
    "score": {"split", "shard", "authorization", "data", "logits"},
    "targets": {"split", "shard", "authorization", "data", "logits", "scored"},
    "training_corpus": {"split", "authorization", "data", "shards"},
    "normalizers": {"authorization", "train"},
    "normalized": {"split", "shard", "authorization", "data", "logits", "scored", "normalizers", "train"},
    "train_cell": {"authorization", "train", "calibration", "normalizers", "normalized_train",
                   "normalized_calibration", "arm", "checkpoint_seed", "reader_seed", "gpu_grant", "resume", "reuse_audit"},
    "freeze": {"data_authorization", "train", "calibration", "normalizers", "normalized_train", "normalized_calibration", "cells", "reuse_audit"},
    "authorize_test": {"freeze", "freeze_audit"},
    "test_inference": {"split", "authorization", "data", "logits", "scored", "normalized", "gpu_grant"},
    "test_evaluation": {"split", "authorization", "data", "logits", "scored", "normalized", "predictions"},
}
FILE_OPERATIONS = {"authorize_data", "authorize_reuse", "freeze", "authorize_test"}


def validate_request(ref):
    request = p.read_reference(ref)
    if (not isinstance(request, dict) or set(request) != {"request_id", "operation", "output", "arguments", "common"}
            or request["operation"] not in ARGUMENTS or not isinstance(request["arguments"], dict)
            or set(request["arguments"]) != ARGUMENTS[request["operation"]]
            or not isinstance(request["request_id"], str) or not request["request_id"].strip()):
        raise ValueError("stage request identity, operation or argument schema differs")
    output = safe_member(p.ROOT, request["output"])
    if not output.is_relative_to(OUTPUTS) or output == OUTPUTS:
        raise ValueError("stage output must be a child of the learned-reader campaign tree")
    if output.is_relative_to(OUTPUTS/"requests") or output.is_relative_to(STAGING):
        raise ValueError("scientific stages cannot write into request or supervision authority")
    if output.exists() or output.with_name(output.name+".FAILURE.json").exists():
        raise FileExistsError("attempt output already exists; no overwrite or reuse")
    from .learned_partition_gate import common_binding
    if request["common"] != common_binding():
        raise ValueError("request sources/runtime/checkpoints changed")
    request_path = safe_member(p.ROOT, ref["path"])
    if not request_path.is_relative_to(OUTPUTS/"requests"):
        raise ValueError("campaign requests must be durable in the canonical requests tree")
    return request, output


def limits(operation):
    if operation == "geometry_profile":
        return 120., 1024**3, 0
    if operation in {"training_cpu_profile", "training_gpu_profile"}:
        return 120., 4*1024**3, 2*1024**3 if operation == "training_gpu_profile" else 0
    if operation == "forward":
        return 1200., 4*1024**3, 2*1024**3
    if operation in {"train_cell", "test_inference"}:
        return 1200., 4*1024**3, 0  # Frozen data authorization selects the actual device below.
    if operation == "score":
        return 2400., 2*1024**3, 0
    if operation not in ARGUMENTS:
        raise ValueError("unknown resource envelope")
    return 1200., 2*1024**3, 0


def execute(ref, *, permit=None):
    request, output = validate_request(ref)
    op, kwargs = request["operation"], dict(request["arguments"])
    if op == "train_cell":
        from .learned_partition_budget import verify_permit
        from .learned_partition_campaign import train_cell
        verify_permit(permit, ref)
        kwargs.update(request_ref=ref, permit=permit)
        fn = train_cell
    elif op in {"authorize_reuse", "import_prepared"}:
        from .learned_partition_reuse import create_reuse_authorization, import_prepared
        fn = {"authorize_reuse": create_reuse_authorization, "import_prepared": import_prepared}[op]
    elif op in {"freeze", "authorize_test"}:
        from .learned_partition_selection import create_freeze, create_test_authorization
        fn = {"freeze": create_freeze, "authorize_test": create_test_authorization}[op]
    elif op in {"test_inference", "test_evaluation"}:
        from .learned_partition_test import infer_test, evaluate_test
        fn = {"test_inference": infer_test, "test_evaluation": evaluate_test}[op]
    # Deferred imports keep the geometry producer independent of Torch.
    elif op.endswith("profile"):
        from .learned_partition_profile import geometry_profile, training_profile
        fn = geometry_profile if op == "geometry_profile" else training_profile
        if op != "geometry_profile":
            kwargs["device"] = "cpu" if op == "training_cpu_profile" else "cuda:0"
    elif op in {"prepare", "aggregate_data", "authorize_data"}:
        from .learned_partition_data import prepare_shard
        from .learned_partition_gate import aggregate_split, create_data_authorization
        fn = {"prepare": prepare_shard, "aggregate_data": aggregate_split,
              "authorize_data": create_data_authorization}[op]
    else:
        from .learned_partition_runner import (forward_shard, score_shard, supervised_targets_shard,
            aggregate_training_corpus, fit_train_normalizers, normalize_shard)
        fn = {"forward": forward_shard, "score": score_shard, "targets": supervised_targets_shard,
              "training_corpus": aggregate_training_corpus, "normalizers": fit_train_normalizers,
              "normalized": normalize_shard}[op]
    result = fn(output, **kwargs)
    from .learned_partition_gate import common_binding
    if p.read_reference(ref) != request or common_binding() != request["common"]:
        raise ValueError("request or source binding changed during stage")
    return result


def process_measurement(pid, *, gpu):
    """Inspect only the owned worker PID. This function never initializes CUDA."""
    try:
        status = Path(f"/proc/{pid}/status").read_text()
        rss = max((int(line.split()[1])*1024 for line in status.splitlines()
                   if line.startswith(("VmHWM:", "VmRSS:"))), default=0)
    except FileNotFoundError:
        rss = 0
    used = 0
    if gpu:
        result = subprocess.run(["nvidia-smi", "--id=0", "--query-compute-apps=pid,used_gpu_memory",
            "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True, timeout=5)
        for line in result.stdout.splitlines():
            fields = [s.strip() for s in line.split(",")]
            if len(fields) != 2:
                raise ValueError("GPU process measurement schema differs")
            if int(fields[0]) == pid:
                used += int(fields[1])*1024**2
    return rss, used


def _terminate(child):
    """Stop our separately created session; wait confirms direct child death."""
    if child.poll() is None:
        if os.getpgid(child.pid) != child.pid:
            raise RuntimeError("owned worker unexpectedly changed process group")
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=5)
    return child.wait(timeout=5)


def process_identity(pid):
    return {"pid": pid, "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            "process_start_ticks": Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]}


def arm_parent_death(expected_pid):
    """Linux worker guard, installed before any CUDA or optimizer operation."""
    import ctypes
    if os.getppid() != expected_pid or expected_pid <= 1:
        raise PermissionError("worker lost its supervisor during bootstrap")
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, int(signal.SIGKILL), 0, 0, 0) != 0:  # PR_SET_PDEATHSIG; cannot be swallowed by cooperative stop.
        raise OSError(ctypes.get_errno(), "cannot install parent-death guard")
    if os.getppid() != expected_pid:
        raise PermissionError("supervisor died while arming the worker")


def supervised(ref):
    request, output = validate_request(ref)
    wall, rss_limit, gpu_limit = limits(request["operation"])
    if request["operation"] in {"train_cell", "test_inference"}:
        from .learned_partition_gate import verify_authorization
        auth = verify_authorization(request["arguments"]["authorization"],
            "train" if request["operation"] == "train_cell" else request["arguments"]["split"])
        if request["operation"] == "test_inference":
            auth = p.read_reference(p.read_reference(auth["freeze"])["data_authorization"])
        gpu_limit = 2*1024**3 if auth["training_device"] == "cuda:0" else 0
    STAGING.mkdir(parents=True, exist_ok=True)
    control = Path(tempfile.mkdtemp(prefix="supervisor-", dir=STAGING))
    started = time.monotonic()
    write_json(control/"request.json", {"reference": ref, "request": request,
        "owner": "/root", "wall_limit_seconds": wall, "rss_limit_bytes": rss_limit, "gpu_limit_bytes": gpu_limit,
        "monotonic_started": started})
    read_fd, write_fd = os.pipe()
    child, peak_rss, peak_gpu, failure, result = None, 0, 0, None, None
    leases, permit = ExitStack(), None
    try:
        if request["operation"] == "train_cell":
            from .learned_partition_budget import reserve
            permit, wall = leases.enter_context(reserve(ref, request, control))
        env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                   CUBLAS_WORKSPACE_CONFIG=":4096:8", TMPDIR=str(TEMPORARIES))
        if not TEMPORARIES.is_dir():
            raise ValueError("project-owned temporary directory must exist")
        env["CUDA_VISIBLE_DEVICES"] = "0" if gpu_limit else ""
        cli = p.ROOT/"experiments/atencion_armonica/run_learned_partition.py"
        with (control/"stdout.log").open("xb") as stdout, (control/"stderr.log").open("xb") as stderr:
            child = subprocess.Popen([sys.executable, str(cli), "--worker-fd", str(read_fd)], env=env,
                pass_fds=(read_fd,), stdout=stdout, stderr=stderr, start_new_session=True, cwd=p.ROOT)
            os.close(read_fd)
            read_fd = None
            write_json(control/"started.json", {"request": ref, "budget": permit,
                **process_identity(child.pid), "monotonic_started": started})
            with os.fdopen(write_fd, "w") as pipe:
                write_fd = None
                json.dump({"reference": ref, "permit": permit, "supervisor_pid": os.getpid()}, pipe)
            while child.poll() is None:
                rss, gpu = process_measurement(child.pid, gpu=bool(gpu_limit))
                peak_rss, peak_gpu = max(peak_rss, rss), max(peak_gpu, gpu)
                if time.monotonic()-started > wall or peak_rss >= rss_limit or (gpu_limit and peak_gpu >= gpu_limit):
                    raise TimeoutError("owned worker exceeded its wall/RSS/GPU envelope")
                time.sleep(.2)
            if child.wait() != 0:
                raise RuntimeError(f"worker exit {child.returncode}; logs in {control.relative_to(p.ROOT)}")
        if time.monotonic()-started > wall:
            raise TimeoutError("worker finished beyond its wall envelope")
        envelope = json.loads((control/"stdout.log").read_text())
        if (set(envelope) != {"result", "peak_rss_bytes"} or type(envelope["peak_rss_bytes"]) is not int
                or envelope["peak_rss_bytes"] <= 0):
            raise ValueError("missing complete worker high-water memory measurement")
        peak_rss = max(peak_rss, envelope["peak_rss_bytes"])
        if peak_rss >= rss_limit:
            raise RuntimeError("worker high-water RSS exceeded its envelope between supervisor polls")
        result = envelope["result"]
        expected = output if request["operation"] in FILE_OPERATIONS else output/"manifest.json"
        if p.verify_reference(result).resolve() != expected.resolve():
            raise ValueError("worker returned an artifact outside its requested output")
        from .learned_partition_gate import common_binding
        if p.read_reference(ref) != request or common_binding() != request["common"]:
            raise ValueError("supervised request binding changed")
    except BaseException as exc:
        failure = exc
        if request["operation"] == "train_cell" and permit is None:
            # reserve may have published its immutable entry but failed before
            # yielding (for example during journal fsync). Preserve that debt.
            from .learned_partition_budget import REGISTRY
            reserved = REGISTRY/(ref["sha256"]+".json")
            if reserved.is_file():
                candidate = p.reference(reserved)
                entry = p.read_reference(candidate)
                if entry["request"] == ref and entry["control"] == control.relative_to(p.ROOT).as_posix():
                    permit = candidate
    finally:
        for fd in (read_fd, write_fd):
            if fd is not None:
                os.close(fd)
        terminal, exit_code = child is None, None
        if child is not None:
            try:
                exit_code = _terminate(child)
                terminal = True
            except BaseException as stop_error:
                failure = stop_error
        if failure is not None:
            if output.is_dir():
                mark_failure(output, failure)
            elif output.is_file():
                write_json(output.with_name(output.name+".FAILURE.json"), {"status": "INCOMPLETE", "error": repr(failure)})
            mark_failure(control, failure)
        receipt = {"status": "COMPLETE" if failure is None else "FAILED",
            "request": ref, "request_id": request["request_id"], "operation": request["operation"],
            "output": request["output"], "worker_pid": None if child is None else child.pid,
            "worker_terminal_confirmed": terminal, "worker_exit_code": exit_code,
            "result": result, "seconds": time.monotonic()-started,
            "observed_peak_rss_bytes": peak_rss, "observed_peak_gpu_bytes": peak_gpu,
            "error": None if failure is None else repr(failure), "budget": permit}
        if request["operation"] == "train_cell":
            from .learned_partition_budget import recovery_status
            receipt["recovery_status"] = recovery_status(output, request) if terminal else "UNCONFIRMED"
        else:
            receipt["recovery_status"] = "NOT_TRAINING"
        write_json(control/"terminal.json", receipt)
        leases.close()
    if failure is not None:
        raise RuntimeError(f"stage failed; terminal receipt: {control.relative_to(p.ROOT)}/terminal.json") from failure
    return {"result": result, "supervisor": p.reference(control/"terminal.json")}
