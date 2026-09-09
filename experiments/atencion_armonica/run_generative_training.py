"""Bounded delivery and 27-cell OPEN training; no fresh-test access.

Public modes initialize immutable manifests and run recoverable supervised
attempts.  A process exit is never stage completion: typed worker receipts and
all referenced payloads are reopened before advancing.
"""
from __future__ import annotations

import argparse
import fcntl
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import signal
import shutil
import stat
import subprocess
import sys
import time
import uuid

from experiments.atencion_armonica import prepare_generative_evidence as preparation
from src.atencion_armonica import generative_evidence as ge
from src.atencion_armonica.generative_evidence_profile import gpu_availability
from src.atencion_armonica.generative_evidence_reuse import ROOT, VerifiedBytes
from src.atencion_armonica.generative_evidence_storage import write_json

TEMP = ROOT/".agent-work/phideus-generative-evidence-20260909"
PREPARATION_CONTROL = TEMP/"preparation-control"
PREPARATION_LOCK = PREPARATION_CONTROL/"operator.lock"
DESTINATION = ROOT/"data/atencion_armonica/generative_evidence_reader_v1"
DELIVERY_CONTROL = TEMP/"delivery-control"
DELIVERY_MANIFEST = DELIVERY_CONTROL/"manifest.json"
TRAINING_CONTROL = TEMP/"training-control"
TRAINING_MANIFEST = TRAINING_CONTROL/"manifest.json"
TRAINING_ROOT = DESTINATION/"training"
PROTOCOL = ROOT/"experiments/atencion_armonica/PROTOCOL_GENERATIVE_EVIDENCE_READER.md"
COMPARISON = {"path": ".agent-work/phideus-generative-evidence-20260909/profile-preparation-io-01/comparison.json",
              "sha256": "45555269fd4215fd9529d5db03760b4511763e9c5b036aba50ff1d4c0a1e48f3"}
GPU_PROFILE = {"path": ".agent-work/phideus-generative-evidence-20260909/profile-gpu-01/report.json",
               "sha256": "c3144164274e154f2dee4ca93eea81410d68558da623545006d751ee473051ea"}
CPU_PROFILE = {"path": ".agent-work/phideus-generative-evidence-20260909/profile-cpu-01/report.json",
               "sha256": "f27b7bb1391057f55f220189ae136f96701cae5c31ceaab4283ced308b1e6402"}
GIB = 1024**3
MEMORY_LIMIT = 6*GIB
PREPARATION_LIMIT = 6*3600
TRAINING_LIMIT = 12*3600
CELL_LIMIT = 1800
GRACE = 30
UNMEASURED_SECONDS_PER_CELL = 300
LOAD_PROFILE_METHOD = "one sequential full load_cell per arm/checkpoint; measured, not a training authorization"


def reference(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": hashlib.sha256(raw).hexdigest()}


def _source_paths():
    modules = ("generative_evidence.py", "generative_evidence_cache.py", "generative_evidence_cell.py",
        "generative_evidence_corpus.py", "generative_evidence_inputs.py", "generative_evidence_model.py",
        "generative_evidence_normalization.py", "generative_evidence_preparation.py",
        "generative_evidence_prepared.py", "generative_evidence_profile.py", "generative_evidence_reuse.py",
        "generative_evidence_storage.py", "generative_evidence_supervision.py", "generative_evidence_training.py")
    return sorted({PROTOCOL, Path(__file__).resolve(), ROOT/"experiments/atencion_armonica/prepare_generative_evidence.py",
                   *(ROOT/"src/atencion_armonica"/name for name in modules)})


def sources():
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in _source_paths()}


def runtime():
    return preparation.runtime()


def disk_usage():
    """Count TEMP links physically; reject links/special nodes in canonical data."""
    total = 0
    for owned in (TEMP, DESTINATION):
        if not owned.exists():
            continue
        for base, folders, files in os.walk(owned, followlinks=False):
            for name in [*folders, *files]:
                path = Path(base)/name
                mode = path.lstat().st_mode
                if stat.S_ISLNK(mode):
                    if owned == DESTINATION:
                        raise ValueError("canonical artifact tree contains a symlink")
                    total += path.lstat().st_size
                    continue
                if name in files:
                    if not stat.S_ISREG(mode):
                        raise ValueError("new artifact tree contains a non-regular payload")
                    total += path.lstat().st_size
    return {"new_bytes": total, "free_bytes": shutil.disk_usage(ROOT).free}


def check_disk():
    value = disk_usage()
    if value["new_bytes"] >= 60*GIB or value["free_bytes"] < 80*GIB:
        raise RuntimeError("generative campaign storage envelope reached")
    return value


def cell_roster():
    return [{"arm": arm, "checkpoint_seed": cp, "reader_seed": seed,
             "cell_id": f"{arm}-cp{cp}-seed{seed}"}
            for arm in ge.ARMS for cp in ge.CHECKPOINTS for seed in ge.READER_SEEDS]


def _number(value, *, minimum=0):
    return type(value) in (int, float) and math.isfinite(value) and value >= minimum


def _read(ref):
    return VerifiedBytes(ROOT).json(ref)


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
        raise ValueError("invalid attempt identifier")
    return value


def _standard_preparation_launches():
    result = []
    for path in PREPARATION_CONTROL.glob("attempt-*.launch.json"):
        name = path.name
        if len(name) == 24 and name.startswith("attempt-") and name[8:12].isdigit():
            result.append(path)
    return sorted(result)


def _preparation_base_accumulated(manifest_ref, initial):
    total, previous = initial, None
    paths = _standard_preparation_launches()
    if [p.name for p in paths] != [f"attempt-{i:04d}.launch.json" for i in range(len(paths))]:
        raise ValueError("OPEN attempt indices are not contiguous")
    for path in paths:
        launch_ref, launch = reference(path), _read(reference(path))
        if launch.get("manifest") != manifest_ref or launch.get("previous_exit") != previous:
            raise ValueError("OPEN attempt parent or manifest differs")
        exit_path = path.with_name(path.name.replace(".launch.json", ".exit.json"))
        if not exit_path.exists():
            raise RuntimeError(f"unreconciled OPEN attempt: {launch.get('unit')}")
        exit_ref, end = reference(exit_path), _read(reference(exit_path))
        if end.get("launch") != launch_ref or end.get("terminal") is not True or not _number(end.get("seconds")):
            raise ValueError("OPEN attempt lacks a terminal elapsed receipt")
        total += end["seconds"]
        previous = exit_ref
    return total, previous


def verified_open_completion():
    """Require the last OPEN attempt to have typed success, not merely exit0."""
    manifest_ref = reference(preparation.MANIFEST)
    manifest = preparation.read_manifest(manifest_ref)
    used, parent = _preparation_base_accumulated(manifest_ref, manifest["prior_profile_seconds"])
    if used >= PREPARATION_LIMIT:
        raise RuntimeError("typed OPEN completion exhausted the shared preparation/delivery budget")
    launches = _standard_preparation_launches()
    if not launches or parent is None:
        raise RuntimeError("OPEN preparation has no terminal attempt")
    launch_path = launches[-1]
    launch_ref = reference(launch_path)
    exit_path = launch_path.with_name(launch_path.name.replace(".launch.json", ".exit.json"))
    exit_ref, end = reference(exit_path), _read(reference(exit_path))
    if (parent != exit_ref or end.get("launch") != launch_ref or end.get("terminal") is not True
            or end.get("returncode") != 0):
        raise RuntimeError("last OPEN attempt is not authoritatively terminal")
    stem = launch_path.name.removesuffix(".launch.json")
    worker_path = PREPARATION_CONTROL/f"{stem}.worker.json"
    if not worker_path.exists():
        raise RuntimeError("last OPEN exit has no typed worker completion")
    worker_ref = reference(worker_path)
    worker = _read(worker_ref)
    required = {"launch", "availability", "status", "prepared", "seconds", "peak_rss_bytes",
                "peak_reserved_bytes"}
    if (set(worker) != required or worker["launch"] != launch_ref or worker["status"] != "OPEN_PREPARED"
            or not _number(worker["seconds"]) or not _number(worker["peak_rss_bytes"])
            or not _number(worker["peak_reserved_bytes"]) or not _valid_gpu_availability(worker["availability"])):
        raise RuntimeError("last OPEN worker is not typed OPEN_PREPARED")
    from src.atencion_armonica.generative_evidence_prepared import PreparedStore
    from src.atencion_armonica.generative_evidence_corpus import TrainingCorpus
    store = PreparedStore(DESTINATION, binding={"stage_manifest": manifest_ref})
    expected = store.reference(store.path("open_prepared.json"))
    if worker["prepared"] != expected:
        raise ValueError("OPEN worker does not reference the canonical prepared completion")
    TrainingCorpus(store, expected)
    return {"manifest": manifest_ref, "manifest_value": manifest, "exit": exit_ref,
            "worker": worker_ref, "prepared": expected, "used_seconds": used}


def initialize_delivery():
    with _lock(PREPARATION_LOCK):
        if DELIVERY_MANIFEST.exists():
            raise FileExistsError("delivery manifest already exists; use --run-delivery")
        opened = verified_open_completion()
        value = {"schema": "generative-evidence-delivery-stage-v1", "stage": "FLOAT32_DELIVERY_AND_LOAD_PROFILE",
            "device": "cpu", "sources": sources(), "runtime": runtime(), "open": {k: opened[k] for k in
                ("manifest", "exit", "worker", "prepared")}, "preparation_used_seconds": opened["used_seconds"],
            "limit_seconds": PREPARATION_LIMIT, "memory_max_bytes": MEMORY_LIMIT,
            "storage_max_bytes": 60*GIB, "free_min_bytes": 80*GIB,
            "delivered_shards": 27, "load_profile_cases": 9, "test_access": False,
            "resource_measurements": COMPARISON}
        DELIVERY_CONTROL.mkdir(parents=True, exist_ok=True)
        write_json(DELIVERY_MANIFEST, value)
        return reference(DELIVERY_MANIFEST)


def read_delivery_manifest(ref):
    if ref["path"] != DELIVERY_MANIFEST.relative_to(ROOT).as_posix():
        raise ValueError("unexpected delivery manifest path")
    value = _read(ref)
    opened = verified_open_completion()
    keys = {"schema", "stage", "device", "sources", "runtime", "open", "preparation_used_seconds",
            "limit_seconds", "memory_max_bytes", "storage_max_bytes", "free_min_bytes", "delivered_shards",
            "load_profile_cases", "test_access", "resource_measurements"}
    if (set(value) != keys or value["schema"] != "generative-evidence-delivery-stage-v1"
            or value["stage"] != "FLOAT32_DELIVERY_AND_LOAD_PROFILE" or value["device"] != "cpu"
            or value["runtime"] != runtime() or value["sources"] != sources()
            or value["open"] != {k: opened[k] for k in ("manifest", "exit", "worker", "prepared")}
            or value["preparation_used_seconds"] != opened["used_seconds"]
            or value["limit_seconds"] != PREPARATION_LIMIT or value["memory_max_bytes"] != MEMORY_LIMIT
            or value["storage_max_bytes"] != 60*GIB or value["free_min_bytes"] != 80*GIB
            or value["delivered_shards"] != 27 or value["load_profile_cases"] != 9
            or value["test_access"] is not False or value["resource_measurements"] != COMPARISON):
        raise ValueError("delivery stage identity, sources, runtime or limits differ")
    return value


def _delivery_launches():
    return sorted(DELIVERY_CONTROL.glob("attempt-*.launch.json"))


def _delivery_prefix(delivery_ref, paths):
    manifest = read_delivery_manifest(delivery_ref)
    # Separate receipts preserve the frozen OPEN ledger, while the numeric
    # origin is its fully charged typed completion: delivery cannot reset 6 h.
    total = manifest["preparation_used_seconds"]
    previous = manifest["open"]["exit"]
    all_paths = _delivery_launches()
    if [p.name for p in all_paths] != [f"attempt-{i:04d}.launch.json" for i in range(len(all_paths))]:
        raise ValueError("delivery attempt indices are not contiguous")
    if all_paths[:len(paths)] != list(paths):
        raise ValueError("delivery accounting is not an exact ledger prefix")
    for path in paths:
        launch_ref, launch = reference(path), _read(reference(path))
        keys = {"schema", "phase", "manifest", "delivery_manifest", "previous_exit", "unit", "command",
                "used_seconds", "remaining_seconds", "runtime_seconds"}
        remaining = math.floor(PREPARATION_LIMIT-total)
        if (set(launch) != keys or launch.get("schema") != "generative-evidence-delivery-launch-v1"
                or launch.get("delivery_manifest") != delivery_ref or launch.get("previous_exit") != previous
                or launch.get("manifest") != manifest["open"]["manifest"] or launch.get("phase") != "delivery"):
            raise ValueError("delivery attempt chain or binding differs")
        if (launch["used_seconds"] != total or launch["remaining_seconds"] != remaining
                or launch["runtime_seconds"] != remaining-GRACE
                or launch["command"] != _service_command(
                    launch["unit"], launch["runtime_seconds"], "--delivery-worker",
                    path.name.removesuffix(".launch.json"), gpu=False)):
            raise ValueError("delivery attempt accounting or command differs")
        exit_path = path.with_name(path.name.replace(".launch.json", ".exit.json"))
        if not exit_path.exists():
            raise RuntimeError(f"unreconciled delivery attempt: {launch.get('unit')}")
        exit_ref, end = reference(exit_path), _read(reference(exit_path))
        if end.get("launch") != launch_ref or end.get("terminal") is not True or not _number(end.get("seconds")):
            raise ValueError("delivery attempt lacks a terminal elapsed receipt")
        total += end["seconds"]
        previous = exit_ref
    return total, previous


def delivery_accumulated(delivery_ref):
    return _delivery_prefix(delivery_ref, _delivery_launches())


def _service_command(unit, runtime_seconds, mode, attempt, *, gpu):
    _attempt_name(attempt)
    if type(runtime_seconds) is not int or runtime_seconds <= 0:
        raise ValueError("worker runtime must be a positive integer")
    env = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "CUDA_VISIBLE_DEVICES": "0" if gpu else "",
           "PHIDEUS_GENERATIVE_UNIT": unit, "TMPDIR": str(TEMP)}
    return ["systemd-run", "--quiet", "--collect", "--wait", "--pipe", f"--unit={unit}",
        "--property=MemoryMax=6G", "--property=MemorySwapMax=0", "--property=OOMPolicy=kill",
        f"--property=RuntimeMaxSec={runtime_seconds}s", f"--property=TimeoutStopSec={GRACE}s",
        "--property=KillMode=control-group", f"--working-directory={ROOT}", "/usr/bin/env",
        *[f"{k}={v}" for k, v in env.items()], str(ROOT/"venv/bin/python"), "-m",
        "experiments.atencion_armonica.run_generative_training", mode, "--attempt", attempt]


def _execute_command(command, unit, launch_ref, exit_path):
    started, requested = time.monotonic(), []
    def stop(signum, frame):
        requested.append(signum)
    handlers = {s: signal.signal(s, stop) for s in (signal.SIGINT, signal.SIGTERM)}
    result = {"launch": launch_ref}
    try:
        result["process_returncode"] = preparation.run_command(command, unit, requested)
    finally:
        probe = preparation.terminal_state(unit)
        result.update(seconds=time.monotonic()-started, terminal=probe["terminal"],
                      systemd_state=probe["state"], systemd_probe_returncode=probe["returncode"],
                      systemd_stderr=probe["stderr"])
        write_json(exit_path, result)
        for s, handler in handlers.items():
            signal.signal(s, handler)
    if result["terminal"] is not True:
        raise RuntimeError("worker has not reached an authoritative terminal state")
    return result


def _verify_service(unit, runtime_seconds):
    group = Path("/proc/self/cgroup").read_text().strip().split("::")[-1]
    if os.environ.get("PHIDEUS_GENERATIVE_UNIT") != unit or Path(group).name != unit+".service":
        raise RuntimeError("worker requires its exact supervised service")
    cg = Path("/sys/fs/cgroup")/group.lstrip("/")
    if (cg/"memory.max").read_text().strip() != str(MEMORY_LIMIT) or (cg/"memory.swap.max").read_text().strip() != "0":
        raise RuntimeError("worker memory guard differs")
    props = subprocess.check_output(["systemctl", "show", unit+".service", "--property=KillMode",
        "--property=OOMPolicy"], text=True, timeout=5)
    if dict(line.split("=", 1) for line in props.splitlines()) != {"KillMode": "control-group", "OOMPolicy": "kill"}:
        raise RuntimeError("worker kill/oom guard differs")
    preparation.verify_deadline(unit, runtime_seconds)


def _worker_check(started, runtime_seconds, stopped, *, torch=None, check_storage=True):
    if stopped or time.monotonic()-started >= runtime_seconds-GRACE:
        raise InterruptedError("recoverable stop at a durable boundary")
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 > MEMORY_LIMIT:
        raise RuntimeError("worker RSS envelope exceeded")
    if torch is not None and torch.cuda.max_memory_reserved(0) > MEMORY_LIMIT:
        raise RuntimeError("worker GPU envelope exceeded")
    if check_storage:
        check_disk()


def _load_profile(store, corpus, delivered, check):
    path = store.path("delivered/load_profile.json")
    if path.exists():
        value = store.json(store.reference(path))
        _validate_load_profile(value, delivered)
        return store.reference(path)
    cases = []
    for arm in ge.ARMS:
        for cp in ge.CHECKPOINTS:
            check()
            started = time.monotonic()
            data = corpus.load_cell(delivered, arm=arm, checkpoint_seed=cp, check=check)
            cases.append({"arm": arm, "checkpoint_seed": cp, "seconds": time.monotonic()-started,
                "train_count": len(data.rows["train"]), "calibration_count": len(data.rows["calibration"]),
                "train_eligible": len(data.eligible["train"]),
                "calibration_eligible": len(data.eligible["calibration"])})
            del data
            gc.collect()
    value = {"schema": "generative-evidence-real-cell-loader-profile-v1", "device": "cpu",
        "delivered": delivered, "cases": cases, "case_count": 9,
        "method": LOAD_PROFILE_METHOD,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
    _validate_load_profile(value, delivered)
    write_json(path, value)
    return store.reference(path)


def _validate_load_profile(value, delivered):
    keys = {"schema", "device", "delivered", "cases", "case_count", "method", "peak_rss_bytes"}
    expected = [(a, cp) for a in ge.ARMS for cp in ge.CHECKPOINTS]
    if (set(value) != keys or value["schema"] != "generative-evidence-real-cell-loader-profile-v1"
            or value["device"] != "cpu" or value["delivered"] != delivered or value["case_count"] != 9
            or value["method"] != LOAD_PROFILE_METHOD or not _number(value["peak_rss_bytes"])
            or len(value["cases"]) != 9
            or [(c.get("arm"), c.get("checkpoint_seed")) for c in value["cases"]] != expected):
        raise ValueError("real loader profile identity or roster differs")
    for case in value["cases"]:
        if (set(case) != {"arm", "checkpoint_seed", "seconds", "train_count", "calibration_count",
                          "train_eligible", "calibration_eligible"}
                or not _number(case["seconds"]) or case["train_count"] != 4096 or case["calibration_count"] != 512
                or type(case["train_eligible"]) is not int or not 0 < case["train_eligible"] <= 4096
                or type(case["calibration_eligible"]) is not int or not 0 < case["calibration_eligible"] <= 512):
            raise ValueError("real loader profile case differs")


def worker_delivery(stem):
    _attempt_name(stem)
    launch_path = DELIVERY_CONTROL/f"{stem}.launch.json"
    launch_ref, launch = reference(launch_path), _read(reference(launch_path))
    launch_keys = {"schema", "phase", "manifest", "delivery_manifest", "previous_exit", "unit", "command",
                   "used_seconds", "remaining_seconds", "runtime_seconds"}
    if (set(launch) != launch_keys or launch["schema"] != "generative-evidence-delivery-launch-v1"
            or launch["phase"] != "delivery" or launch["command"] != _service_command(
                launch["unit"], launch["runtime_seconds"], "--delivery-worker", stem, gpu=False)):
        raise ValueError("delivery launch schema or command differs")
    paths = _delivery_launches()
    if not paths or paths[-1] != launch_path:
        raise ValueError("delivery worker launch is not the unique ledger tail")
    used, previous = _delivery_prefix(launch["delivery_manifest"], paths[:-1])
    remaining = math.floor(PREPARATION_LIMIT-used)
    if (launch["previous_exit"] != previous or launch["used_seconds"] != used
            or launch["remaining_seconds"] != remaining or launch["runtime_seconds"] != remaining-GRACE):
        raise ValueError("delivery worker budget prefix differs")
    _verify_service(launch["unit"], launch["runtime_seconds"])
    if (os.environ.get("CUDA_VISIBLE_DEVICES") != ""
            or any(os.environ.get(k) != "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"))):
        raise RuntimeError("delivery worker must remain CPU-only and single-threaded")
    manifest = read_delivery_manifest(launch["delivery_manifest"])
    started, stopped, disk_checked = time.monotonic(), [], 0.
    handlers = {s: signal.signal(s, lambda signum, frame: stopped.append(signum)) for s in (signal.SIGINT, signal.SIGTERM)}
    report, code = {"launch": launch_ref, "status": "INCOMPLETE"}, 1
    def check():
        nonlocal disk_checked
        now = time.monotonic()
        due = now-disk_checked >= 30
        _worker_check(started, launch["runtime_seconds"], stopped, check_storage=due)
        if due:
            disk_checked = now
    try:
        from src.atencion_armonica.generative_evidence_prepared import PreparedStore
        from src.atencion_armonica.generative_evidence_corpus import TrainingCorpus
        store = PreparedStore(DESTINATION, binding={"stage_manifest": manifest["open"]["manifest"]})
        corpus = TrainingCorpus(store, manifest["open"]["prepared"])
        report["delivered"] = corpus.materialize(check=check, progress=lambda row: print(json.dumps(row), flush=True))
        report["load_profile"] = _load_profile(store, corpus, report["delivered"], check)
        read_delivery_manifest(launch["delivery_manifest"])
        check()
        report["status"], code = "DELIVERED_PROFILED", 0
    except InterruptedError as exc:
        report.update(status="PAUSED_RECOVERABLE", reason=str(exc))
        code = 75
    except BaseException as exc:
        report.update(status="FAILED", reason=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report.update(seconds=time.monotonic()-started,
                      peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        write_json(DELIVERY_CONTROL/f"{stem}.worker.json", report)
        for s, handler in handlers.items(): signal.signal(s, handler)
    return code


def _delivery_attempt_state(path):
    launch_ref, launch = reference(path), _read(reference(path))
    exit_path = path.with_name(path.name.replace(".launch.json", ".exit.json"))
    end = _read(reference(exit_path))
    worker_path = path.with_name(path.name.replace(".launch.json", ".worker.json"))
    if not worker_path.exists():
        raise RuntimeError("terminal delivery lacks a typed worker receipt; do not relaunch ambiguously")
    worker_ref, worker = reference(worker_path), _read(reference(worker_path))
    if worker.get("launch") != launch_ref or end.get("launch") != launch_ref or end.get("terminal") is not True:
        raise ValueError("delivery worker/exit parent differs")
    return launch, end, worker_ref, worker


def verified_delivery():
    delivery_ref = reference(DELIVERY_MANIFEST)
    manifest = read_delivery_manifest(delivery_ref)
    used, _ = delivery_accumulated(delivery_ref)
    if used > PREPARATION_LIMIT:
        raise RuntimeError("typed delivery exceeded the shared preparation/delivery budget")
    launches = _delivery_launches()
    if not launches:
        raise RuntimeError("delivery has no attempt")
    launch, end, worker_ref, worker = _delivery_attempt_state(launches[-1])
    if end.get("process_returncode") != 0 or worker.get("status") != "DELIVERED_PROFILED":
        raise RuntimeError("delivery is not typed DELIVERED_PROFILED")
    from src.atencion_armonica.generative_evidence_prepared import PreparedStore
    from src.atencion_armonica.generative_evidence_corpus import TrainingCorpus
    store = PreparedStore(DESTINATION, binding={"stage_manifest": manifest["open"]["manifest"]})
    corpus = TrainingCorpus(store, manifest["open"]["prepared"])
    delivered = store.reference(store.path("delivered/index.json"))
    profile = store.reference(store.path("delivered/load_profile.json"))
    if worker.get("delivered") != delivered or worker.get("load_profile") != profile:
        raise ValueError("delivery worker references another output")
    _validate_load_profile(store.json(profile), delivered)
    return {"manifest": delivery_ref, "worker": worker_ref, "delivered": delivered,
            "load_profile": profile, "store": store, "corpus": corpus}


def run_delivery():
    with _lock(PREPARATION_LOCK):
        delivery_ref = reference(DELIVERY_MANIFEST)
        manifest = read_delivery_manifest(delivery_ref)
        used, parent = delivery_accumulated(delivery_ref)
        launches = _delivery_launches()
        if launches:
            _, end, _, worker = _delivery_attempt_state(launches[-1])
            if worker["status"] == "DELIVERED_PROFILED":
                return {k: v for k, v in verified_delivery().items() if k not in {"store", "corpus"}}
            if worker["status"] != "PAUSED_RECOVERABLE" or end["process_returncode"] != 75:
                raise RuntimeError("failed or ambiguous delivery cannot be relaunched automatically")
        remaining = math.floor(PREPARATION_LIMIT-used)
        if remaining <= GRACE+10:
            raise RuntimeError("shared preparation/delivery cumulative time exhausted")
        check_disk()
        index = len(_delivery_launches())
        stem = f"attempt-{index:04d}"
        unit = "phideus-generative-delivery-"+uuid.uuid4().hex[:16]
        runtime_seconds = remaining-GRACE
        command = _service_command(unit, runtime_seconds, "--delivery-worker", stem, gpu=False)
        launch_path = DELIVERY_CONTROL/f"{stem}.launch.json"
        write_json(launch_path, {"schema": "generative-evidence-delivery-launch-v1", "phase": "delivery",
            "manifest": manifest["open"]["manifest"], "delivery_manifest": delivery_ref,
            "previous_exit": parent, "unit": unit, "command": command, "used_seconds": used,
            "remaining_seconds": remaining, "runtime_seconds": runtime_seconds})
        launch_ref = reference(launch_path)
        end = _execute_command(command, unit, launch_ref,
                               DELIVERY_CONTROL/f"{stem}.exit.json")
        if end["process_returncode"] != 0:
            raise RuntimeError("delivery attempt did not complete")
        return {k: v for k, v in verified_delivery().items() if k not in {"store", "corpus"}}


def verify_delivered_payloads(delivery):
    """Read all 27 compact shards without creating or repairing any payload."""
    corpus, store, delivered = delivery["corpus"], delivery["store"], delivery["delivered"]
    complete = store.json(delivered)
    expected = [(split, cp, shard) for split in ("train", "calibration")
                for shard in range((4096 if split == "train" else 512)//512) for cp in ge.CHECKPOINTS]
    if [(e.get("split"), e.get("checkpoint_seed"), e.get("shard")) for e in complete["entries"]] != expected:
        raise ValueError("delivered corpus does not contain the exact 27-shard roster")
    for entry in complete["entries"]:
        index, shard_ref, _, decoded = corpus.raw(entry["split"], entry["checkpoint_seed"], entry["shard"])
        value = store.json(entry["index"])
        corpus._decode(value, store.arrays(value["inputs"]), decoded,
                       corpus._identity(entry["split"], entry["checkpoint_seed"], entry["shard"], index, shard_ref))
    return True


def training_resource_plan(delivery):
    reader = VerifiedBytes(ROOT)
    comparison, gpu, cpu = reader.json(COMPARISON), reader.json(GPU_PROFILE), reader.json(CPU_PROFILE)
    store, profile_ref = delivery["store"], delivery["load_profile"]
    load = store.json(profile_ref)
    _validate_load_profile(load, delivery["delivered"])
    if (comparison.get("status") != "RESOURCE_ESTIMATE_NOT_CAMPAIGN_AUTHORIZATION"
            or gpu.get("status") != "MEASURED" or gpu.get("device") != "cuda:0"
            or cpu.get("status") != "MEASURED" or cpu.get("device") != "cpu"
            or gpu.get("runtime") != runtime() or load["peak_rss_bytes"] > MEMORY_LIMIT
            or gpu["peak_rss_bytes"] > MEMORY_LIMIT or gpu.get("peak_reserved_bytes", 0) > MEMORY_LIMIT):
        raise ValueError("resource evidence identity, runtime or envelope differs")
    slim_profiles = [{k: ref[k] for k in ("path", "sha256")} for ref in comparison.get("profiles", [])]
    if slim_profiles != [CPU_PROFILE, GPU_PROFILE]:
        raise ValueError("comparison does not bind the exact CPU/GPU reports")
    heads = gpu["heads"]
    if set(heads) != {"envelope", "observed_train"}:
        raise ValueError("GPU profile lacks envelope and observed head cases")
    measured = {"setup_seconds": max(h["setup_seconds"] for h in heads.values()),
        "update_seconds": max(max(h["steady_update_seconds"]) for h in heads.values()),
        "calibration_batch_io_seconds": max(max(h["evaluation_batch_io_seconds"]) for h in heads.values()),
        "snapshot_save_validation_seconds": max(h["snapshot_io_seconds"] for h in heads.values())}
    by_case = {(c["arm"], c["checkpoint_seed"]): c for c in load["cases"]}
    cells, total = [], 0.
    for cell in cell_roster():
        case = by_case[cell["arm"], cell["checkpoint_seed"]]
        train_batches = math.ceil(case["train_eligible"]/32)
        calibration_batches = math.ceil(case["calibration_eligible"]/32)
        base = (measured["setup_seconds"]+case["seconds"]
            + 50*train_batches*measured["update_seconds"]
            + 10*calibration_batches*measured["calibration_batch_io_seconds"]
            + 51*measured["snapshot_save_validation_seconds"]
            + 51*measured["snapshot_save_validation_seconds"]
            + 10*measured["calibration_batch_io_seconds"])
        projected = 2*base+UNMEASURED_SECONDS_PER_CELL
        if projected >= CELL_LIMIT:
            raise RuntimeError("a projected cell exceeds its 1800-second allowance")
        cells.append({**cell, "loader_seconds": case["seconds"], "train_batches": train_batches,
            "calibration_batches": calibration_batches, "measured_base_seconds": base,
            "factor_two_seconds": 2*base, "unmeasured_reserve_seconds": UNMEASURED_SECONDS_PER_CELL,
            "projected_seconds": projected})
        total += projected
    if total >= TRAINING_LIMIT:
        raise RuntimeError("complete 27-cell projection exceeds the 12-hour budget")
    return {"schema": "generative-evidence-training-resource-plan-v1", "comparison": COMPARISON,
        "gpu_profile": GPU_PROFILE, "cpu_profile": CPU_PROFILE, "real_loader_profile": profile_ref,
        "measured_gpu_maxima": measured, "snapshot_count_per_cell": 51,
        "final_snapshot_reads_per_cell": 51, "final_calibration_reads_per_cell": 10,
        "allowance_factor": 2, "unmeasured_reserve_seconds_per_cell": UNMEASURED_SECONDS_PER_CELL,
        "method": "GPU maxima across envelope/observed; real CPU load_cell; 50 epochs; 10 calibrations; 51 snapshot saves/validations plus 51 final reads and 10 calibration reads; factor2 plus explicit unmeasured reserve",
        "cells": cells, "projected_total_seconds": total,
        "observed_peak_rss_bytes": max(load["peak_rss_bytes"], gpu["peak_rss_bytes"]),
        "observed_peak_gpu_reserved_bytes": gpu.get("peak_reserved_bytes", 0)}


def initialize_training():
    with _lock(PREPARATION_LOCK), _lock(TRAINING_CONTROL/"operator.lock"):
        if TRAINING_MANIFEST.exists():
            raise FileExistsError("training manifest already exists; use --run-training")
        delivery = verified_delivery()
        verify_delivered_payloads(delivery)
        plan = training_resource_plan(delivery)
        value = {"schema": "generative-evidence-training-stage-v1", "stage": "TRAIN_27_NOT_SELECTED",
            "device": "cuda:0", "sources": sources(), "runtime": runtime(),
            "delivery": {k: delivery[k] for k in ("manifest", "worker", "delivered", "load_profile")},
            "resource_plan": plan, "roster": cell_roster(), "epochs": 50,
            "limit_seconds": TRAINING_LIMIT, "cell_limit_seconds": CELL_LIMIT,
            "memory_max_bytes": MEMORY_LIMIT, "storage_max_bytes": 60*GIB,
            "free_min_bytes": 80*GIB, "test_access": False, "selection": False}
        TRAINING_CONTROL.mkdir(parents=True, exist_ok=True)
        write_json(TRAINING_MANIFEST, value)
        return reference(TRAINING_MANIFEST)


def read_training_manifest(ref):
    if ref["path"] != TRAINING_MANIFEST.relative_to(ROOT).as_posix():
        raise ValueError("unexpected training manifest path")
    value = _read(ref)
    delivery = verified_delivery()
    plan = training_resource_plan(delivery)
    keys = {"schema", "stage", "device", "sources", "runtime", "delivery", "resource_plan", "roster",
            "epochs", "limit_seconds", "cell_limit_seconds", "memory_max_bytes", "storage_max_bytes",
            "free_min_bytes", "test_access", "selection"}
    if (set(value) != keys or value["schema"] != "generative-evidence-training-stage-v1"
            or value["stage"] != "TRAIN_27_NOT_SELECTED" or value["device"] != "cuda:0"
            or value["runtime"] != runtime() or value["sources"] != sources()
            or value["delivery"] != {k: delivery[k] for k in ("manifest", "worker", "delivered", "load_profile")}
            or value["resource_plan"] != plan or value["roster"] != cell_roster() or value["epochs"] != 50
            or value["limit_seconds"] != TRAINING_LIMIT or value["cell_limit_seconds"] != CELL_LIMIT
            or value["memory_max_bytes"] != MEMORY_LIMIT or value["storage_max_bytes"] != 60*GIB
            or value["free_min_bytes"] != 80*GIB or value["test_access"] is not False
            or value["selection"] is not False):
        raise ValueError("training stage identity, sources, resources or roster differ")
    return value, delivery


def _valid_gpu_availability(value):
    return (isinstance(value, dict) and set(value) == {"processes", "inventory"}
            and value["processes"] == "" and isinstance(value["inventory"], str)
            and "RTX 3090" in value["inventory"] and len(value["inventory"].strip().splitlines()) == 1)


def _training_prefix(manifest_ref, paths):
    total, previous, per_cell, cell_parent = 0., None, {c["cell_id"]: 0. for c in cell_roster()}, {}
    all_paths = sorted(TRAINING_CONTROL.glob("attempt-*.launch.json"))
    if [p.name for p in all_paths] != [f"attempt-{i:04d}.launch.json" for i in range(len(all_paths))]:
        raise ValueError("training attempt indices are not contiguous")
    if all_paths[:len(paths)] != list(paths):
        raise ValueError("training accounting is not an exact ledger prefix")
    roster = {c["cell_id"]: c for c in cell_roster()}
    for path in paths:
        launch_ref, launch = reference(path), _read(reference(path))
        cell_id = launch.get("cell", {}).get("cell_id")
        keys = {"schema", "manifest", "previous_exit", "previous_cell_exit", "cell", "unit", "command",
                "availability", "used_total_seconds", "used_cell_seconds", "remaining_allowance_seconds",
                "runtime_seconds"}
        remaining = math.floor(min(TRAINING_LIMIT-total, CELL_LIMIT-per_cell.get(cell_id, math.inf)))
        if (set(launch) != keys or launch.get("schema") != "generative-evidence-training-launch-v1"
                or launch.get("manifest") != manifest_ref or launch.get("previous_exit") != previous
                or cell_id not in roster or launch.get("cell") != roster[cell_id]
                or launch.get("previous_cell_exit") != cell_parent.get(cell_id)):
            raise ValueError("training attempt global/cell parent or binding differs")
        if (launch["used_total_seconds"] != total or launch["used_cell_seconds"] != per_cell[cell_id]
                or launch["remaining_allowance_seconds"] != remaining
                or launch["runtime_seconds"] != remaining-GRACE or not _valid_gpu_availability(launch["availability"])
                or launch["command"] != _service_command(
                    launch["unit"], launch["runtime_seconds"], "--training-worker",
                    path.name.removesuffix(".launch.json"), gpu=True)):
            raise ValueError("training attempt accounting, availability or command differs")
        exit_path = path.with_name(path.name.replace(".launch.json", ".exit.json"))
        if not exit_path.exists():
            raise RuntimeError(f"unreconciled training attempt: {launch.get('unit')}")
        exit_ref, end = reference(exit_path), _read(reference(exit_path))
        if end.get("launch") != launch_ref or end.get("terminal") is not True or not _number(end.get("seconds")):
            raise ValueError("training attempt lacks a terminal elapsed receipt")
        total += end["seconds"]
        per_cell[cell_id] += end["seconds"]
        previous, cell_parent[cell_id] = exit_ref, exit_ref
    return total, previous, per_cell, cell_parent


def training_accumulated(manifest_ref):
    return _training_prefix(manifest_ref, sorted(TRAINING_CONTROL.glob("attempt-*.launch.json")))


def _cell_root(cell):
    return TRAINING_ROOT/cell["arm"]/f"cp_{cell['checkpoint_seed']}"/f"seed_{cell['reader_seed']}"


def _cell_binding(manifest_ref, data, cell):
    return {"training_manifest": manifest_ref, "data": data.binding,
            "arm": cell["arm"], "checkpoint_seed": cell["checkpoint_seed"],
            "reader_seed": cell["reader_seed"]}


def verify_cell_complete(manifest_ref, delivery, cell, *, check=lambda: None):
    from src.atencion_armonica.generative_evidence_cell import CellArtifacts, read_calibration
    root = _cell_root(cell)
    if not (root/"complete.json").exists():
        return None
    data = delivery["corpus"].load_cell(delivery["delivered"], arm=cell["arm"],
                                        checkpoint_seed=cell["checkpoint_seed"], check=check)
    store = CellArtifacts(root, binding=_cell_binding(manifest_ref, data, cell))
    local_ref = store.reference(root/"complete.json")
    complete = store.json(local_ref)
    keys = {"schema", "binding", "status", "arm", "checkpoint_seed", "reader_seed", "epochs", "steps",
            "train_eligible", "calibration_eligible", "snapshots", "calibrations", "last_epoch"}
    batches = math.ceil(len(data.eligible["train"])/32)
    if (set(complete) != keys or complete["schema"] != "generative-evidence-cell-complete-v1"
            or complete["binding"] != store.binding or complete["status"] != "TRAINED_NOT_SELECTED"
            or any(complete[k] != cell[k] for k in ("arm", "checkpoint_seed", "reader_seed"))
            or complete["epochs"] != 50 or complete["steps"] != 50*batches
            or complete["train_eligible"] != data.eligible["train"]
            or complete["calibration_eligible"] != data.eligible["calibration"]
            or len(complete["snapshots"]) != 51 or len(complete["calibrations"]) != 10
            or complete["last_epoch"] != complete["snapshots"][-1]):
        raise ValueError("cell completion identity or full roster differs")
    previous = None
    # Validate every physical boundary and its parent chain, including optional
    # mid-epoch recovery snapshots, before accepting the 51 required epochs.
    for path in sorted(store.path("snapshots").glob("step_*.json")):
        row_ref, row = store.reference(path), store.json(store.reference(path))
        if row["previous"] != previous:
            raise ValueError("cell snapshot chain lost a parent")
        store.load_state(row_ref)
        previous = row_ref
    for epoch, snap_ref in enumerate(complete["snapshots"]):
        expected = store.reference(store.path(f"snapshots/step_{epoch*batches:06d}.json"))
        state = store.load_state(snap_ref)
        if snap_ref != expected or state["epoch"] != epoch or state["next_batch"] != 0:
            raise ValueError("required epoch snapshot differs")
    for ref_cal, epoch in zip(complete["calibrations"], range(5, 51, 5)):
        expected = store.reference(store.path(f"calibration/epoch_{epoch:02d}/index.json"))
        if ref_cal != expected:
            raise ValueError("required calibration reference differs")
        read_calibration(data, store, ref_cal, complete["snapshots"][epoch], epoch)
    return reference(root/"complete.json")


def worker_training(attempt):
    _attempt_name(attempt)
    launch_path = TRAINING_CONTROL/f"{attempt}.launch.json"
    launch_ref, launch = reference(launch_path), _read(reference(launch_path))
    launch_keys = {"schema", "manifest", "previous_exit", "previous_cell_exit", "cell", "unit", "command",
                   "availability", "used_total_seconds", "used_cell_seconds", "remaining_allowance_seconds",
                   "runtime_seconds"}
    if (set(launch) != launch_keys or launch["schema"] != "generative-evidence-training-launch-v1"
            or launch["command"] != _service_command(
                launch["unit"], launch["runtime_seconds"], "--training-worker", attempt, gpu=True)):
        raise ValueError("training launch schema or command differs")
    paths = sorted(TRAINING_CONTROL.glob("attempt-*.launch.json"))
    if not paths or paths[-1] != launch_path:
        raise ValueError("training worker launch is not the unique ledger tail")
    total, previous, per_cell, cell_parents = _training_prefix(launch["manifest"], paths[:-1])
    cell_id = launch["cell"].get("cell_id")
    remaining = math.floor(min(TRAINING_LIMIT-total, CELL_LIMIT-per_cell.get(cell_id, math.inf)))
    if (cell_id not in per_cell or launch["previous_exit"] != previous
            or launch["previous_cell_exit"] != cell_parents.get(cell_id)
            or launch["used_total_seconds"] != total or launch["used_cell_seconds"] != per_cell[cell_id]
            or launch["remaining_allowance_seconds"] != remaining or launch["runtime_seconds"] != remaining-GRACE
            or not _valid_gpu_availability(launch["availability"])):
        raise ValueError("training worker budget prefix or availability differs")
    _verify_service(launch["unit"], launch["runtime_seconds"])
    manifest, delivery = read_training_manifest(launch["manifest"])
    cell = launch["cell"]
    import torch
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA initialized before worker availability check")
    availability = gpu_availability()
    if not _valid_gpu_availability(availability):
        raise RuntimeError("worker GPU availability identity differs")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if any(os.environ.get(k) != "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")):
        raise RuntimeError("worker CPU thread environment differs")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("worker CUDA deterministic workspace differs")
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(MEMORY_LIMIT/torch.cuda.get_device_properties(0).total_memory, 0)
    torch.cuda.reset_peak_memory_stats(0)
    started, stopped, disk_checked = time.monotonic(), [], 0.
    handlers = {s: signal.signal(s, lambda signum, frame: stopped.append(signum)) for s in (signal.SIGINT, signal.SIGTERM)}
    report, code = {"launch": launch_ref, "availability": availability, "status": "INCOMPLETE"}, 1
    def check():
        nonlocal disk_checked
        now = time.monotonic()
        due = now-disk_checked >= 30
        _worker_check(started, launch["runtime_seconds"], stopped, torch=torch, check_storage=due)
        if due:
            disk_checked = now
    try:
        from src.atencion_armonica.generative_evidence_cell import CellArtifacts, run_cell
        data = delivery["corpus"].load_cell(delivery["delivered"], arm=cell["arm"],
                                            checkpoint_seed=cell["checkpoint_seed"], check=check)
        store = CellArtifacts(_cell_root(cell), binding=_cell_binding(launch["manifest"], data, cell))
        report["complete"] = run_cell(data, store, arm=cell["arm"], checkpoint_seed=cell["checkpoint_seed"],
            reader_seed=cell["reader_seed"], device=manifest["device"], check=check,
            progress=lambda row: print(json.dumps(row), flush=True))
        check()
        report["status"], code = "TRAINED", 0
    except InterruptedError as exc:
        report.update(status="PAUSED_RECOVERABLE", reason=str(exc))
        code = 75
    except BaseException as exc:
        report.update(status="FAILED", reason=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report.update(seconds=time.monotonic()-started,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            peak_reserved_bytes=torch.cuda.max_memory_reserved(0))
        write_json(TRAINING_CONTROL/f"{attempt}.worker.json", report)
        for s, handler in handlers.items(): signal.signal(s, handler)
    return code


def _latest_cell_worker(cell_id):
    found = []
    for path in sorted(TRAINING_CONTROL.glob("attempt-*.launch.json")):
        launch_ref, launch = reference(path), _read(reference(path))
        if launch.get("cell", {}).get("cell_id") == cell_id:
            worker_path = path.with_name(path.name.replace(".launch.json", ".worker.json"))
            exit_path = path.with_name(path.name.replace(".launch.json", ".exit.json"))
            if not worker_path.exists():
                raise RuntimeError("terminal cell attempt lacks a typed worker receipt; do not relaunch ambiguously")
            found.append({"launch_ref": launch_ref, "launch": launch,
                          "exit": _read(reference(exit_path)), "worker": _read(reference(worker_path))})
    return found[-1] if found else None


def _require_trained_authority(manifest_ref, cell, complete, latest):
    if latest is None:
        raise RuntimeError("cell payload exists without a supervised attempt authority")
    launch, end, worker = latest["launch"], latest["exit"], latest["worker"]
    if (launch.get("manifest") != manifest_ref or launch.get("cell") != cell
            or end.get("launch") != latest["launch_ref"] or end.get("terminal") is not True
            or end.get("process_returncode") != 0 or worker.get("launch") != latest["launch_ref"]
            or worker.get("status") != "TRAINED"):
        raise RuntimeError("latest cell attempt is not typed terminal TRAINED")
    local = worker.get("complete")
    if (not isinstance(local, dict) or set(local) != {"path", "sha256", "bytes"}
            or local["path"] != "complete.json" or local["sha256"] != complete["sha256"]
            or not _number(local["bytes"])):
        raise ValueError("TRAINED worker does not bind the exact complete payload")


def _recoverable_cell_attempt(manifest_ref, cell, latest):
    """A durable payload can precede a pause; only its typed attempt may resume."""
    if latest is None:
        return False
    launch, end, worker = latest["launch"], latest["exit"], latest["worker"]
    return (launch.get("manifest") == manifest_ref and launch.get("cell") == cell
            and end.get("launch") == latest["launch_ref"] and end.get("terminal") is True
            and end.get("process_returncode") == 75
            and worker.get("launch") == latest["launch_ref"]
            and worker.get("status") == "PAUSED_RECOVERABLE")


def verify_cell_completion_authority(manifest_ref, delivery, cell):
    """Reopen payloads and require the latest attempt's exact typed authority."""
    training_accumulated(manifest_ref)  # Reject any missing/invalid terminal first.
    complete = verify_cell_complete(manifest_ref, delivery, cell)
    if complete is None:
        raise RuntimeError("cell is not complete")
    _require_trained_authority(manifest_ref, cell, complete, _latest_cell_worker(cell["cell_id"]))
    return complete


def verified_training():
    """Closed downstream port: all 27 cells plus the aggregate index."""
    manifest_ref = reference(TRAINING_MANIFEST)
    manifest, delivery = read_training_manifest(manifest_ref)
    cells = [{"cell": cell, "complete": verify_cell_completion_authority(manifest_ref, delivery, cell)}
             for cell in manifest["roster"]]
    total, _, per_cell, _ = training_accumulated(manifest_ref)
    if total > TRAINING_LIMIT or any(v > CELL_LIMIT for v in per_cell.values()):
        raise RuntimeError("training completion exceeds its accumulated time envelope")
    expected = {"schema": "generative-evidence-training-complete-v1", "status": "TRAINED_NOT_SELECTED",
        "manifest": manifest_ref, "cells": cells, "cell_count": 27,
        "accumulated_seconds": total, "test_access": False}
    path = TRAINING_ROOT/"index.json"
    index = reference(path)
    if _read(index) != expected:
        raise ValueError("aggregate training index differs from 27 typed completions")
    return {"manifest": manifest_ref, "index": index, "cells": cells, "delivery": delivery}


def run_training():
    with _lock(PREPARATION_LOCK), _lock(TRAINING_CONTROL/"operator.lock"):
        manifest_ref = reference(TRAINING_MANIFEST)
        manifest, delivery = read_training_manifest(manifest_ref)
        check_disk()
        completes = []
        for cell in manifest["roster"]:
            total, parent, per_cell, cell_parents = training_accumulated(manifest_ref)
            latest = _latest_cell_worker(cell["cell_id"])
            complete = verify_cell_complete(manifest_ref, delivery, cell)
            recoverable = _recoverable_cell_attempt(manifest_ref, cell, latest)
            if complete is not None and not recoverable:
                _require_trained_authority(manifest_ref, cell, complete, latest)
                completes.append({"cell": cell, "complete": complete})
                continue
            if latest is not None and not recoverable:
                raise RuntimeError("failed or ambiguous cell cannot be relaunched automatically")
            remaining = math.floor(min(TRAINING_LIMIT-total, CELL_LIMIT-per_cell[cell["cell_id"]]))
            if remaining <= GRACE+10:
                raise RuntimeError("global or per-cell training time exhausted")
            availability = gpu_availability()
            attempt = f"attempt-{len(list(TRAINING_CONTROL.glob('attempt-*.launch.json'))):04d}"
            unit = "phideus-generative-cell-"+uuid.uuid4().hex[:16]
            runtime_seconds = remaining-GRACE
            command = _service_command(unit, runtime_seconds, "--training-worker", attempt, gpu=True)
            launch_path = TRAINING_CONTROL/f"{attempt}.launch.json"
            write_json(launch_path, {"schema": "generative-evidence-training-launch-v1", "manifest": manifest_ref,
                "previous_exit": parent, "previous_cell_exit": cell_parents.get(cell["cell_id"]),
                "cell": cell, "unit": unit, "command": command, "availability": availability,
                "used_total_seconds": total, "used_cell_seconds": per_cell[cell["cell_id"]],
                "remaining_allowance_seconds": remaining, "runtime_seconds": runtime_seconds})
            launch_ref = reference(launch_path)
            end = _execute_command(command, unit, launch_ref, TRAINING_CONTROL/f"{attempt}.exit.json")
            if end["process_returncode"] != 0:
                raise RuntimeError(f"cell {cell['cell_id']} did not complete")
            latest = _latest_cell_worker(cell["cell_id"])
            worker = latest["worker"]
            if worker.get("status") != "TRAINED":
                raise RuntimeError("exit0 cell lacks typed TRAINED completion")
            complete = verify_cell_complete(manifest_ref, delivery, cell)
            if complete is None:
                raise RuntimeError("typed TRAINED worker lacks a revalidated complete cell")
            _require_trained_authority(manifest_ref, cell, complete, latest)
            completes.append({"cell": cell, "complete": complete})
        # Account for the last attempt and refuse a completion beyond either cap.
        total, _, per_cell, _ = training_accumulated(manifest_ref)
        if total > TRAINING_LIMIT or any(v > CELL_LIMIT for v in per_cell.values()):
            raise RuntimeError("completed roster exceeded its accumulated time envelope")
        value = {"schema": "generative-evidence-training-complete-v1", "status": "TRAINED_NOT_SELECTED",
                 "manifest": manifest_ref, "cells": completes, "cell_count": 27,
                 "accumulated_seconds": total, "test_access": False}
        path = TRAINING_ROOT/"index.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if _read(reference(path)) != value:
                raise ValueError("training completion changed on revalidation")
        else:
            write_json(path, value)
        return reference(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--initialize-delivery", action="store_true")
    mode.add_argument("--run-delivery", action="store_true")
    mode.add_argument("--initialize-training", action="store_true")
    mode.add_argument("--run-training", action="store_true")
    mode.add_argument("--delivery-worker", action="store_true", help=argparse.SUPPRESS)
    mode.add_argument("--training-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--attempt", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.initialize_delivery:
        result = initialize_delivery()
    elif args.run_delivery:
        result = run_delivery()
    elif args.initialize_training:
        result = initialize_training()
    elif args.run_training:
        result = run_training()
    elif args.delivery_worker:
        sys.exit(worker_delivery(args.attempt))
    else:
        sys.exit(worker_training(args.attempt))
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
