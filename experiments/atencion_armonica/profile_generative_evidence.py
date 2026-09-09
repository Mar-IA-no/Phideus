"""OS-supervised resource profile; never starts the27-cell campaign.

Only this process's unique transient service is stopped. The service enforces
6GiB memory (no swap) and116s runtime with2s termination grace externally to
Python/CUDA. Internal timers are a cooperative early exit, not the hard guard.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

from src.atencion_armonica.generative_evidence_profile import ROOT, run_profile, source_binding
from src.atencion_armonica.generative_evidence_storage import write_json


def verify_worker_guard():
    unit = os.environ.get("PHIDEUS_PROFILE_UNIT", "")
    group = Path("/proc/self/cgroup").read_text().strip().split("::")[-1]
    if not unit.startswith("phideus-generative-profile-") or Path(group).name != unit+".service":
        raise RuntimeError("profile worker requires its own supervised transient service")
    cg = Path("/sys/fs/cgroup")/group.lstrip("/")
    maximum = (cg/"memory.max").read_text().strip()
    if maximum == "max" or not 0 < int(maximum) <= 6*1024**3 or (cg/"memory.swap.max").read_text().strip() != "0":
        raise RuntimeError("worker memory guard absent")
    text = subprocess.check_output(["systemctl", "show", unit+".service", "--property=RuntimeMaxUSec",
                                    "--property=TimeoutStopUSec", "--property=KillMode"], text=True, timeout=5)
    if dict(line.split("=", 1) for line in text.splitlines()) != {
            "RuntimeMaxUSec": "1min 56s", "TimeoutStopUSec": "2s", "KillMode": "control-group"}:
        raise RuntimeError("worker external runtime/termination guard differs")


def supervised_profile(output, device):
    output = Path(output).resolve()
    base = ROOT/".agent-work/phideus-generative-evidence-20260909"
    if not output.is_relative_to(base) or output == base or output.exists() or not output.parent.is_dir():
        raise ValueError("profile needs a fresh owned temporary output")
    unit = "phideus-generative-profile-"+uuid.uuid4().hex[:16]
    env = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "CUDA_VISIBLE_DEVICES": "0" if device == "cuda:0" else "",
           "PHIDEUS_PROFILE_UNIT": unit, "TMPDIR": str(base)}
    command = ["systemd-run", "--quiet", "--collect", "--wait", "--pipe", f"--unit={unit}",
               "--property=MemoryMax=6G", "--property=MemorySwapMax=0", "--property=OOMPolicy=kill",
               "--property=RuntimeMaxSec=116s", "--property=TimeoutStopSec=2s", "--property=KillMode=control-group",
               f"--working-directory={ROOT}", "/usr/bin/env", *[f"{k}={v}" for k, v in env.items()],
               str(ROOT/"venv/bin/python"), "-m", "experiments.atencion_armonica.profile_generative_evidence",
               "--worker", "--device", device, "--output", str(output)]
    launch = {"schema": "generative-evidence-profile-supervisor-v1", "owner": "Phideus Codex", "unit": unit,
              "command": command, "binding": source_binding(), "memory_max_bytes": 6*1024**3,
              "runtime_seconds": 116, "termination_grace_seconds": 2}
    launch_path = output.with_name(output.name+".launch.json")
    launch_ref = write_json(launch_path, launch)
    def interrupt(signum, frame):
        raise KeyboardInterrupt(f"profile supervisor received signal{signum}")
    handlers = {s: signal.signal(s, interrupt) for s in (signal.SIGTERM, signal.SIGINT)}
    started = time.monotonic()
    result = {"launch": {"path": str(launch_path.relative_to(ROOT)), **launch_ref}, "unit": unit}
    try:
        completed = subprocess.run(command, timeout=123)
        result["returncode"] = completed.returncode
    except BaseException as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        subprocess.run(["systemctl", "stop", unit+".service"], timeout=5, check=False)
        raise
    finally:
        result["seconds"] = time.monotonic()-started
        state = subprocess.run(["systemctl", "show", unit+".service", "--property=ActiveState",
                                "--property=SubState", "--property=Result"], capture_output=True, text=True, timeout=5)
        result["terminal_state"] = {"stdout": state.stdout, "stderr": state.stderr, "returncode": state.returncode}
        write_json(output.with_name(output.name+".exit.json"), result)
        for s, handler in handlers.items():
            signal.signal(s, handler)
    return completed.returncode

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        verify_worker_guard()
        report = run_profile(args.output, device=args.device)
        print({"status": report["status"], "seconds": report["seconds"], "device": report["device"]})
    else:
        sys.exit(supervised_profile(args.output, args.device))
