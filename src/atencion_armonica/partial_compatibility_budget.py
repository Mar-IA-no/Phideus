"""Exclusive, fail-closed wall-time budget for this study's sequential GPU jobs."""
from __future__ import annotations

import fcntl
import json
import math
import os
from pathlib import Path
import time

LIMIT_SECONDS = 24*3600


def atomic_json(path, value):
    temporary = path.with_name(path.name+f".{os.getpid()}.tmp")
    with temporary.open("x") as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


class CampaignBudget:
    """Unsettled reservations require explicit recovery, never an automatic reset.

    Hold the advisory lock throughout child execution and pass its FD to the
    child. If the parent dies, the child still holds the lock; if both die, the
    persisted reservation prevents a fresh process from assuming unused time.
    Charged time includes CPU work while the GPU worker is alive, conservatively.
    """

    def __init__(self, root: Path, *, profile_manifest_sha256, profile_seconds):
        if not profile_manifest_sha256 or not math.isfinite(profile_seconds) or not 0 <= profile_seconds < LIMIT_SECONDS:
            raise ValueError("invalid prior profile cost")
        root.mkdir(parents=True, exist_ok=True)
        self.path = root/"GPU_BUDGET.json"
        self.lock = (root/"GPU_BUDGET.lock").open("a+")
        self.started = None
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if self.path.exists():
                self.state = json.loads(self.path.read_text())
                if (self.state["limit_seconds"] != LIMIT_SECONDS
                        or self.state["profile_manifest_sha256"] != profile_manifest_sha256
                        or self.state["profile_seconds"] != profile_seconds
                        or not math.isfinite(self.state["charged_seconds"])
                        or self.state["charged_seconds"] < profile_seconds):
                    raise ValueError("budget identity or charge mismatch")
                if self.state["reservation"] is not None:
                    raise RuntimeError("unsettled previous GPU reservation: reconcile evidence before resuming")
            else:
                self.state = {"limit_seconds": LIMIT_SECONDS,
                              "profile_manifest_sha256": profile_manifest_sha256,
                              "profile_seconds": profile_seconds, "charged_seconds": profile_seconds,
                              "reservation": None, "attempts": []}
                atomic_json(self.path, self.state)
        except BaseException:
            self.lock.close()
            raise

    @property
    def remaining(self):
        return max(0., LIMIT_SECONDS-self.state["charged_seconds"])

    def reserve(self, output, request_sha256):
        if self.started is not None or self.state["reservation"] is not None or self.remaining < 1:
            raise RuntimeError("no available GPU reservation")
        self.state["reservation"] = {"output": str(Path(output).resolve()),
                                     "request_sha256": request_sha256,
                                     "reserved_seconds": self.remaining, "parent_pid": os.getpid(),
                                     "started_unix": time.time()}
        atomic_json(self.path, self.state)
        self.started = time.monotonic()
        return self.remaining

    def execution_deadline(self, termination_margin=15.):
        if self.started is None or self.remaining <= termination_margin:
            raise RuntimeError("insufficient budget including termination margin")
        return self.started+self.remaining-termination_margin

    def settle(self, status):
        if self.started is None:
            raise RuntimeError("no live reservation")
        elapsed = time.monotonic()-self.started
        self.state["attempts"].append({**self.state["reservation"], "seconds": elapsed, "status": status})
        self.state["charged_seconds"] += elapsed
        self.state["overrun_seconds"] = max(0., self.state["charged_seconds"]-LIMIT_SECONDS)
        self.state["reservation"] = None
        atomic_json(self.path, self.state)
        self.started = None

    def close(self):
        self.lock.close()  # Does not settle an unknown child state.


def verify_inherited_lease(root, output, request_sha256, fd, deadline):
    """Operational ancestry check, not a security boundary against arbitrary code."""
    if type(fd) is not int or fd < 0 or not math.isfinite(deadline):
        raise ValueError("worker requires a live inherited lease")
    actual, expected = os.fstat(fd), (root/"GPU_BUDGET.lock").stat()
    if (actual.st_dev, actual.st_ino) != (expected.st_dev, expected.st_ino):
        raise ValueError("worker inherited the wrong lock descriptor")
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = json.loads((root/"GPU_BUDGET.json").read_text())
    reservation = state["reservation"]
    if (reservation is None or reservation["parent_pid"] != os.getppid()
            or reservation["output"] != str(Path(output).resolve())
            or reservation["request_sha256"] != request_sha256
            or deadline <= time.monotonic()
            or deadline-time.monotonic() > reservation["reserved_seconds"]):
        raise ValueError("worker has no matching live parent reservation")
