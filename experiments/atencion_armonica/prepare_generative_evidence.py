"""Externally bounded OPEN preparation, never training or fresh-test access.

Initialize a stage manifest after review, then run its recoverable fixed roster.
The immutable attempt ledger charges actual wall time, including failed runs.
An attempt lacking an exit receipt must be reconciled before another launch.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import shlex
import shutil
import signal
import subprocess
import sys
import time
import uuid

from src.atencion_armonica.generative_evidence_preparation import OpenPreparation, PROFILE
from src.atencion_armonica.generative_evidence_prepared import PreparedStore
from src.atencion_armonica.generative_evidence_profile import PROTOCOL, PROTOCOL_SHA, CORE_SHA, gpu_availability
from src.atencion_armonica.generative_evidence_reuse import AUTHORIZATION, IMPORT, ROOT, OpenReuse, VerifiedBytes
from src.atencion_armonica.generative_evidence_storage import write_json

TEMP = ROOT/".agent-work/phideus-generative-evidence-20260909"
CONTROL = TEMP/"preparation-control"
DESTINATION = ROOT/"data/atencion_armonica/generative_evidence_reader_v1"
MANIFEST = CONTROL/"manifest.json"
REFINEMENT = {"path": str((TEMP/"profile-preparation-io-01/comparison.json").relative_to(ROOT)),
              "sha256": "45555269fd4215fd9529d5db03760b4511763e9c5b036aba50ff1d4c0a1e48f3"}
GIB = 1024**3
LIMIT_SECONDS = 6*3600
GRACE = 30


def reference(path):
    raw = path.read_bytes()
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": hashlib.sha256(raw).hexdigest()}


def sources():
    paths = [ROOT/PROTOCOL, ROOT/"experiments/atencion_armonica/profile_generative_evidence.py",
             Path(__file__).resolve(), *(ROOT/"src/atencion_armonica").glob("*.py")]
    result = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
    if result[PROTOCOL] != PROTOCOL_SHA or result["src/atencion_armonica/observable_source_rivals.py"] != CORE_SHA:
        raise ValueError("frozen protocol or fitting core changed")
    return result


def runtime():
    import numpy as np
    import torch
    return {"python": platform.python_version(), "numpy": np.__version__, "torch": str(torch.__version__),
            "cuda": torch.version.cuda, "cpu_threads": 1, "deterministic": True, "tf32": False}


def disk_usage():
    # Metadata only. Include owned intermediate/profile artifacts, not old data.
    total = 0
    for root in (TEMP, DESTINATION):
        if root.exists():
            for base, folders, files in os.walk(root, followlinks=False):
                for name in files:
                    path = Path(base)/name
                    if path.is_symlink():
                        raise ValueError("new artifact tree contains a symlink")
                    total += path.stat().st_size
    return {"new_bytes": total, "free_bytes": shutil.disk_usage(ROOT).free}


def check_disk():
    value = disk_usage()
    if value["new_bytes"] >= 60*GIB or value["free_bytes"] < 80*GIB:
        raise RuntimeError("OPEN preparation storage envelope reached")
    return value


def resource_plan():
    reader = VerifiedBytes(ROOT)
    profile, measured = reader.json(PROFILE), reader.json(REFINEMENT)
    for path, sha in profile["binding"]["sources"].items():
        reader.read({"path": path, "sha256": sha})
    # The profile is the measured cost base, not a worst-case guarantee. Allow
    # an additional hour for full-roster sealing/normalization and new receipt
    # overhead. This reserve is explicit planning allowance, NOT measurement.
    projected = measured["costs"]["cuda:0"]["estimated_open_preparation_seconds"]+3600
    prior = measured["seconds"]
    for ref in measured["supervisor_exits"]:
        prior += reader.json({k: ref[k] for k in ("path", "sha256")})["seconds"]
    if projected+prior >= LIMIT_SECONDS:
        raise RuntimeError("complete OPEN projection exceeds preparation budget")
    if runtime() != profile["runtime"]:
        raise ValueError("preparation runtime differs from the measured GPU runtime")
    return profile, {"projection_seconds": projected, "unmeasured_overhead_reserve_seconds": 3600,
                     "prior_profile_seconds": prior}


def initialize():
    if MANIFEST.exists():
        raise FileExistsError("stage manifest already exists; use --run to resume")
    _, plan = resource_plan()
    reuse = OpenReuse()
    value = {"schema": "generative-evidence-open-stage-v1", "stage": "OPEN_PREPARATION_ONLY",
             "device": "cuda:0", "sources": sources(), "runtime": runtime(),
             "authorization": AUTHORIZATION, "import": IMPORT, "reuse": reuse.receipt(),
             "profile": PROFILE, "resource_measurements": REFINEMENT,
             **plan, "limit_seconds": LIMIT_SECONDS,
             "memory_max_bytes": 6*GIB, "storage_max_bytes": 60*GIB,
             "free_min_bytes": 80*GIB, "disk_at_initialization": check_disk(),
             "count": {"train": 4096, "calibration": 512}, "test_access": False}
    CONTROL.mkdir(exist_ok=True)
    write_json(MANIFEST, value)
    return reference(MANIFEST)


def read_manifest(ref):
    if ref["path"] != MANIFEST.relative_to(ROOT).as_posix():
        raise ValueError("unexpected OPEN manifest path")
    value = VerifiedBytes(ROOT).json(ref)
    current_sources = sources()
    profile, plan = resource_plan()
    required_sources = set(profile["binding"]["sources"]) | {
        "src/atencion_armonica/generative_evidence_prepared.py",
        "src/atencion_armonica/generative_evidence_preparation.py",
        Path(__file__).resolve().relative_to(ROOT).as_posix()}
    keys = {"schema", "stage", "device", "sources", "runtime", "authorization", "import", "reuse", "profile",
            "resource_measurements", "projection_seconds", "unmeasured_overhead_reserve_seconds", "prior_profile_seconds",
            "limit_seconds", "memory_max_bytes", "storage_max_bytes", "free_min_bytes", "disk_at_initialization",
            "count", "test_access"}
    if (set(value) != keys or value["schema"] != "generative-evidence-open-stage-v1" or value["stage"] != "OPEN_PREPARATION_ONLY"
            or value["device"] != "cuda:0"
            or not isinstance(value["sources"], dict) or not required_sources.issubset(value["sources"])
            or any(current_sources.get(path) != sha for path, sha in value["sources"].items())
            or value["runtime"] != runtime()
            or value["count"] != {"train": 4096, "calibration": 512} or value["test_access"] is not False
            or value["limit_seconds"] != LIMIT_SECONDS or value["memory_max_bytes"] != 6*GIB
            or value["storage_max_bytes"] != 60*GIB or value["free_min_bytes"] != 80*GIB
            or value["authorization"] != AUTHORIZATION or value["import"] != IMPORT
            or value["profile"] != PROFILE or value["resource_measurements"] != REFINEMENT
            or any(value[k] != v for k, v in plan.items())
            or value["reuse"] != OpenReuse().receipt()):
        raise ValueError("OPEN stage identity, sources, runtime or limits differ")
    return value


def accumulated(manifest_ref, initial):
    if type(initial) not in (int, float) or not math.isfinite(initial) or not 0 <= initial < LIMIT_SECONDS:
        raise ValueError("invalid initial elapsed budget")
    total, previous = initial, None
    for path in sorted(CONTROL.glob("attempt-*.launch.json")):
        launch_ref = reference(path)
        launch = VerifiedBytes(ROOT).json(launch_ref)
        if launch["manifest"] != manifest_ref or launch["previous_exit"] != previous:
            raise ValueError("attempt ledger parent or manifest differs")
        end_path = path.with_name(path.name.replace(".launch.json", ".exit.json"))
        if not end_path.exists():
            raise RuntimeError(f"unreconciled attempt: {launch['unit']}; inspect its authoritative state, do not relaunch")
        end_ref = reference(end_path)
        end = VerifiedBytes(ROOT).json(end_ref)
        if (end["launch"] != launch_ref or end["terminal"] is not True
                or type(end["seconds"]) not in (int, float) or not math.isfinite(end["seconds"]) or end["seconds"] < 0):
            raise ValueError("attempt has no valid terminal elapsed receipt")
        total += end["seconds"]
        previous = end_ref
    return total, previous


def terminal_state(unit):
    result = subprocess.run(["systemctl", "show", unit+".service", "--property=LoadState",
                             "--property=ActiveState", "--property=SubState"], capture_output=True, text=True, timeout=5)
    state = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    terminal = state.get("LoadState") == "not-found" or state.get("ActiveState") in ("inactive", "failed")
    return {"terminal": terminal, "state": state, "returncode": result.returncode, "stderr": result.stderr}


def verify_deadline(unit, runtime_seconds):
    object_result = shlex.split(subprocess.check_output([
        "busctl", "call", "org.freedesktop.systemd1", "/org/freedesktop/systemd1",
        "org.freedesktop.systemd1.Manager", "GetUnit", "s", unit+".service"], text=True, timeout=5))
    if len(object_result) != 2 or object_result[0] != "o":
        raise RuntimeError("could not resolve the worker systemd object")
    for prop, expected in (("RuntimeMaxUSec", runtime_seconds*1_000_000), ("TimeoutStopUSec", GRACE*1_000_000)):
        value = subprocess.check_output(["busctl", "get-property", "org.freedesktop.systemd1", object_result[1],
            "org.freedesktop.systemd1.Service", prop], text=True, timeout=5).strip()
        if value != f"t {expected}":
            raise RuntimeError(f"worker external deadline differs: {prop}")


def request_stop(unit):
    # Starts systemd's stop job, including TimeoutStopSec and eventual SIGKILL.
    # A raw `systemctl kill` alone would not start that external grace timer.
    return subprocess.run(["systemctl", "stop", "--no-block", unit+".service"],
                          capture_output=True, text=True, timeout=5).returncode == 0


def run_command(command, unit, stop_requested):
    # Terminal Ctrl-C reaches the supervisor, not its systemd-run client. The
    # supervisor owns forwarding and remains able to observe the service exit.
    process = subprocess.Popen(command, start_new_session=True)
    forwarded = False
    while process.poll() is None:
        if stop_requested and not forwarded:
            # The signal can arrive before systemd has created the service.
            # Retry this same stop request until creation or parent termination;
            # never launch a replacement job because observation is incomplete.
            forwarded = request_stop(unit)
        time.sleep(.25)
    if stop_requested:
        if not forwarded:
            request_stop(unit)  # Also cover an already-terminal command client.
        deadline = time.monotonic()+GRACE+5
        while not terminal_state(unit)["terminal"]:
            if time.monotonic() >= deadline:
                raise RuntimeError("own service stop has not reached a terminal state")
            time.sleep(.25)
    return process.returncode


def run_supervised():
    CONTROL.mkdir(exist_ok=True)
    with (CONTROL/"operator.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest_ref = reference(MANIFEST)
        manifest = read_manifest(manifest_ref)
        used, parent = accumulated(manifest_ref, manifest["prior_profile_seconds"])
        remaining = math.floor(LIMIT_SECONDS-used)
        if remaining <= GRACE+10:
            raise RuntimeError("preparation cumulative time exhausted")
        check_disk()
        availability = gpu_availability()  # No CUDA until both parent/worker check.
        count = len(list(CONTROL.glob("attempt-*.launch.json")))
        stem = f"attempt-{count:04d}"
        unit = "phideus-generative-prepare-"+uuid.uuid4().hex[:16]
        env = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
               "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "CUDA_VISIBLE_DEVICES": "0",
               "PHIDEUS_PREPARATION_UNIT": unit, "TMPDIR": str(TEMP)}
        command = ["systemd-run", "--quiet", "--collect", "--wait", "--pipe", f"--unit={unit}",
                   "--property=MemoryMax=6G", "--property=MemorySwapMax=0", "--property=OOMPolicy=kill",
                   f"--property=RuntimeMaxSec={remaining-GRACE}s", f"--property=TimeoutStopSec={GRACE}s",
                   "--property=KillMode=control-group", f"--working-directory={ROOT}", "/usr/bin/env",
                   *[f"{k}={v}" for k, v in env.items()], str(ROOT/"venv/bin/python"), "-m",
                   "experiments.atencion_armonica.prepare_generative_evidence", "--worker", "--attempt", stem]
        launch_path = CONTROL/f"{stem}.launch.json"
        write_json(launch_path, {"manifest": manifest_ref, "previous_exit": parent, "unit": unit,
            "owner": "Phideus Codex", "command": command, "availability": availability,
            "used_seconds": used, "remaining_seconds": remaining, "runtime_seconds": remaining-GRACE})
        launch_ref = reference(launch_path)
        started = time.monotonic()
        # A user stop forwards SIGTERM to this exact service. It does not kill
        # another process or interpret an observation timeout as completion.
        stop_requested = []
        def stop(signum, frame):
            stop_requested.append(signum)
        handlers = {s: signal.signal(s, stop) for s in (signal.SIGINT, signal.SIGTERM)}
        result = {"launch": launch_ref}
        try:
            result["returncode"] = run_command(command, unit, stop_requested)
        finally:
            result["seconds"] = time.monotonic()-started
            result.update(terminal_state(unit))
            write_json(CONTROL/f"{stem}.exit.json", result)
            for s, handler in handlers.items():
                signal.signal(s, handler)
        if result["terminal"] is not True:
            raise RuntimeError("worker has not reached an authoritative terminal state")
        return result["returncode"]


def worker(stem):
    if not isinstance(stem, str) or len(stem) != 12 or not stem.startswith("attempt-") or not stem[8:].isdigit():
        raise ValueError("invalid attempt identifier")
    launch_ref = reference(CONTROL/f"{stem}.launch.json")
    launch = VerifiedBytes(ROOT).json(launch_ref)
    unit = os.environ.get("PHIDEUS_PREPARATION_UNIT")
    group = Path("/proc/self/cgroup").read_text().strip().split("::")[-1]
    if unit != launch["unit"] or Path(group).name != unit+".service":
        raise RuntimeError("worker requires its exact supervised service")
    cg = Path("/sys/fs/cgroup")/group.lstrip("/")
    if (cg/"memory.max").read_text().strip() != str(6*GIB) or (cg/"memory.swap.max").read_text().strip() != "0":
        raise RuntimeError("worker memory guard differs")
    # systemd exposes times in a display format: read numeric usec via busctl
    # to verify the external deadline without a fragile duration string parser.
    props = subprocess.check_output(["systemctl", "show", unit+".service", "--property=KillMode",
                                    "--property=OOMPolicy"], text=True, timeout=5)
    if dict(line.split("=", 1) for line in props.splitlines()) != {"KillMode": "control-group", "OOMPolicy": "kill"}:
        raise RuntimeError("worker kill/oom guard differs")
    verify_deadline(unit, launch["runtime_seconds"])
    manifest = read_manifest(launch["manifest"])
    check_disk()
    availability = gpu_availability()
    import torch
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if (any(os.environ.get(k) != "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"))
            or os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8"):
        raise RuntimeError("worker deterministic environment differs")
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(6*GIB/torch.cuda.get_device_properties(0).total_memory, 0)
    torch.cuda.reset_peak_memory_stats(0)
    started, disk_checked, stopped = time.monotonic(), 0., []
    def stop(signum, frame):
        stopped.append(signum)
    handlers = {s: signal.signal(s, stop) for s in (signal.SIGINT, signal.SIGTERM)}
    def check():
        nonlocal disk_checked
        now = time.monotonic()
        if stopped or now-started >= launch["runtime_seconds"]-GRACE:
            raise InterruptedError("recoverable stop at preparation boundary")
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 > 6*GIB or torch.cuda.max_memory_reserved(0) > 6*GIB:
            raise RuntimeError("preparation memory envelope exceeded")
        if now-disk_checked >= 30:
            check_disk()
            disk_checked = now
    report = {"launch": launch_ref, "availability": availability, "status": "INCOMPLETE"}
    code = 1
    try:
        store = PreparedStore(DESTINATION, binding={"stage_manifest": launch["manifest"]})
        operator = OpenPreparation(store, device=manifest["device"], check=check,
                                   progress=lambda text: print(text, flush=True))
        report["prepared"] = operator.run()
        read_manifest(launch["manifest"])
        check_disk()
        report["status"], code = "OPEN_PREPARED", 0
    except InterruptedError as exc:
        report["status"], report["reason"], code = "PAUSED_RECOVERABLE", str(exc), 75
    except BaseException as exc:
        report["reason"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report.update(seconds=time.monotonic()-started,
                      peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                      peak_reserved_bytes=torch.cuda.max_memory_reserved(0))
        write_json(CONTROL/f"{stem}.worker.json", report)
        for s, handler in handlers.items():
            signal.signal(s, handler)
    return code


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--initialize", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--attempt", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.initialize:
        print(json.dumps(initialize()))
    else:
        sys.exit(worker(args.attempt) if args.worker else run_supervised())
