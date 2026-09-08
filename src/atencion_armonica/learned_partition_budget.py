"""One append-only training budget across successful and failed attempts.

The parent holds the campaign lock throughout a worker's lifetime. A missing
or unconfirmed terminal receipt blocks the next lease; absence is not death.
No GPU access, model construction or dataset generation occurs here.
"""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import json
import math
import os
import time
from pathlib import Path

from . import learned_partition_provenance as p
from .learned_partition_core import ARMS, READER_SEEDS
from .learned_partition_metrics import SEEDS
from .partial_compatibility_cache import sha_file
from .structured_source_artifacts import safe_member, write_json, mark_failure

STAGING = p.ROOT/"data/atencion_armonica/learned_partition_reader_v1/supervision"
REGISTRY = STAGING/"training-budget"
CELL_SECONDS = 600.
CAMPAIGN_SECONDS = 21600.
TERMINAL_KEYS = {"status", "request", "request_id", "operation", "output", "worker_pid",
    "worker_terminal_confirmed", "worker_exit_code", "result", "seconds", "observed_peak_rss_bytes",
    "observed_peak_gpu_bytes", "error", "budget", "recovery_status"}


def cell_identity(request):
    args = request["arguments"]
    if (request["operation"] != "train_cell" or args.get("arm") not in ARMS
            or type(args.get("checkpoint_seed")) is not int or args["checkpoint_seed"] not in SEEDS
            or type(args.get("reader_seed")) is not int or args["reader_seed"] not in READER_SEEDS):
        raise ValueError("budget request does not identify a declared training cell")
    return {key: args[key] for key in ("arm", "checkpoint_seed", "reader_seed")}


def terminal_receipt(ref, *, request_ref):
    """Specialized read of a terminal receipt, including failed stage parents.

This never makes a failed scientific bundle valid. The exception to ordinary
failure-marker rejection applies only to this owned supervisor receipt.
"""
    if not isinstance(ref, dict) or set(ref) != {"path", "sha256"}:
        raise ValueError("terminal receipt needs exact path/hash")
    path = safe_member(p.ROOT, ref["path"])
    if (path.name != "terminal.json" or path.parent.parent != STAGING
            or not path.parent.name.startswith("supervisor-") or path.is_symlink()
            or sha_file(path) != ref["sha256"]):
        raise ValueError("terminal receipt is outside the owned supervisor or changed")
    record = json.loads(path.read_bytes())
    request = p.read_reference(request_ref)
    if (set(record) != TERMINAL_KEYS or record["request"] != request_ref
            or record["request_id"] != request["request_id"] or record["output"] != request["output"]
            or record["operation"] != "train_cell" or request["operation"] != "train_cell"
            or record["status"] not in ("COMPLETE", "FAILED") or record["worker_terminal_confirmed"] is not True
            or record["recovery_status"] not in ("NO_UPDATES_NO_SNAPSHOT", "SNAPSHOT_REQUIRED")
            or type(record["seconds"]) not in (int, float) or not math.isfinite(record["seconds"])
            or record["seconds"] <= 0):
        raise ValueError("training parent is not a verified terminal attempt")
    if isinstance(record["error"], dict) and set(record["error"]) == {"reconciliation"}:
        verify_reconciliation(record, path.parent)
    elif record["worker_pid"] is not None and type(record["worker_exit_code"]) is not int:
        raise ValueError("worker death is not backed by its exit status")
    if record["status"] == "COMPLETE" and (record["worker_exit_code"] != 0 or record["error"] is not None):
        raise ValueError("successful supervisor receipt has contradictory terminal state")
    if sha_file(path) != ref["sha256"]:
        raise ValueError("terminal receipt changed while reading")
    return record


def recovery_status(output, request):
    """Classify only a terminal worker; initialization precedes all updates."""
    root = Path(output)
    resume = request["arguments"].get("resume")
    inherited = resume is not None and resume.get("snapshot") is not None
    published = [f for f in root.glob("snapshots/*/manifest.json") if not f.parent.name.startswith("snapshot_staging_")]
    return "SNAPSHOT_REQUIRED" if inherited or published or (root/"training_ready.json").exists() else "NO_UPDATES_NO_SNAPSHOT"


def accounting():
    """Closed registry, all attempts and terminal states, no caller-picked subset."""
    entries, total, cells = [], 0., {}
    origin_path = STAGING/"budget_origin.json"
    journal_path = STAGING/"budget_reservations.jsonl"
    controls = {}
    for control in STAGING.glob("supervisor-*"):
        descriptor = control/"request.json"
        if not descriptor.is_file():
            raise ValueError("supervisor lost its immutable request descriptor")
        owner = json.loads(descriptor.read_bytes())
        if owner["request"]["operation"] != "train_cell":
            continue
        for name in ("started.json", "terminal.json"):
            file = control/name
            if file.is_file():
                record = json.loads(file.read_bytes())
                if record.get("budget") is not None:
                    permit = record["budget"]
                    if record["request"] != owner["reference"]:
                        raise ValueError("supervisor budget and request identity differ")
                    controls.setdefault(permit["path"], []).append((permit, control))
    if not REGISTRY.is_dir():
        if origin_path.exists() or journal_path.exists() or controls:
            raise ValueError("canonical training budget registry was lost; never restart at zero")
        return entries, total, cells
    files = sorted(REGISTRY.iterdir())
    if any(f.is_symlink() or not f.is_file() or (f.name != "lease.lock" and not f.name.endswith(".json")) for f in files):
        raise ValueError("training budget registry inventory differs")
    seen_requests, seen_outputs = set(), set()
    for path in files:
        if path.name == "lease.lock":
            continue
        entry = json.loads(path.read_bytes())
        if (set(entry) != {"request", "control", "cell", "supervisor", "remaining_seconds", "prior_terminals", "reserved_monotonic"}
                or path.name != entry["request"]["sha256"]+".json"):
            raise ValueError("training attempt registry schema or request identity differs")
        request = p.read_reference(entry["request"])
        cell = cell_identity(request)
        if cell != entry["cell"] or request["output"] in seen_outputs or entry["request"]["sha256"] in seen_requests:
            raise ValueError("duplicate attempt output/request or changed cell")
        seen_outputs.add(request["output"])
        seen_requests.add(entry["request"]["sha256"])
        control = safe_member(p.ROOT, entry["control"])
        terminal = control/"terminal.json"
        if not terminal.is_file():
            raise PermissionError("an earlier training attempt has no terminal receipt; do not reset its budget")
        ref = p.reference(terminal)
        record = terminal_receipt(ref, request_ref=entry["request"])
        total += record["seconds"]
        key = tuple(cell.values())
        cells[key] = cells.get(key, 0.)+record["seconds"]
        entry_ref = p.reference(path)
        if record["budget"] != entry_ref:
            raise ValueError("terminal receipt does not bind its original budget reservation")
        entries.append({"entry": entry_ref, "terminal": ref, "record": record, "request": request, "reservation": entry})
    entries.sort(key=lambda row: len(row["reservation"]["prior_terminals"]))
    for i, row in enumerate(entries):
        if row["reservation"]["prior_terminals"] != [earlier["terminal"] for earlier in entries[:i]]:
            raise ValueError("training budget reservations do not form one complete chronological chain")
    if entries:
        if not origin_path.is_file() or json.loads(origin_path.read_bytes()) != {
                "schema": "learned_training_budget_origin_v1", "first_request": entries[0]["record"]["request"]}:
            raise ValueError("canonical training budget origin is absent or changed")
    elif origin_path.exists():
        raise ValueError("canonical budget origin survived but all attempts were lost")
    indexed = {row["entry"]["path"]: row for row in entries}
    for path, pointers in controls.items():
        if path not in indexed or any(ref != indexed[path]["entry"] or control.relative_to(p.ROOT).as_posix()
                != indexed[path]["reservation"]["control"] for ref, control in pointers):
            raise ValueError("training registry omits or changes a supervised attempt, including its latest suffix")
    if set(indexed) != set(controls):
        raise ValueError("training registry entry has no matching durable supervisor evidence")
    verify_journal([row["entry"] for row in entries])
    return entries, total, cells


def verify_journal(expected):
    """Independent append-only reservation inventory, outside entries/controls."""
    path = STAGING/"budget_reservations.jsonl"
    if not path.exists():
        if expected:
            raise ValueError("training reservation journal is missing")
        return
    if path.is_symlink():
        raise ValueError("reservation journal cannot be a symlink")
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise ValueError("interrupted reservation journal append requires reconciliation")
    records = [json.loads(line) for line in raw.splitlines()]
    wanted = [{"sequence": i, "entry": ref} for i, ref in enumerate(expected)]
    if records != wanted:
        raise ValueError("reservation journal differs from the complete attempt chain")


def append_journal(ref, sequence):
    payload = (json.dumps({"sequence": sequence, "entry": ref}, sort_keys=True)+"\n").encode()
    _append_journal_bytes(payload)


def _append_journal_bytes(payload):
    path = STAGING/"budget_reservations.jsonl"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(fd, "ab", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(fd)
    finally:
        os.close(fd)
    fd = os.open(STAGING, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def repair_journal_tail(entries):
    """Complete only an undelivered final reservation, never remove a byte."""
    if not entries:
        raise ValueError("journal repair requires an exact final reservation")
    refs = [ref for ref, _ in entries]
    expected = [(json.dumps({"sequence": i, "entry": ref}, sort_keys=True)+"\n").encode()
                for i, ref in enumerate(refs)]
    prefix, complete = b"".join(expected[:-1]), b"".join(expected)
    journal = STAGING/"budget_reservations.jsonl"
    if journal.is_symlink():
        raise ValueError("journal repair cannot follow a symlink")
    raw = journal.read_bytes() if journal.exists() else b""
    if not raw.startswith(prefix) or not complete.startswith(raw) or len(raw) >= len(complete):
        raise ValueError("journal is not the unique missing literal tail; no truncation or reset")
    if json.loads((STAGING/"budget_origin.json").read_bytes()) != {
            "schema": "learned_training_budget_origin_v1", "first_request": entries[0][1]["request"]}:
        raise ValueError("journal repair requires the unchanged campaign origin")
    prior = []
    for ref, entry in entries:
        if p.read_reference(ref) != entry or entry["prior_terminals"] != prior:
            raise ValueError("journal repair requires the complete exact predecessor chain")
        control = safe_member(p.ROOT, entry["control"])
        if control.parent != STAGING or not control.name.startswith("supervisor-"):
            raise ValueError("journal repair control is outside the campaign")
        descriptor = json.loads((control/"request.json").read_bytes())
        if descriptor["reference"] != entry["request"] or descriptor["request"] != p.read_reference(entry["request"]):
            raise ValueError("journal repair request or supervisor descriptor changed")
        if ref != refs[-1]:
            terminal = p.reference(control/"terminal.json")
            terminal_receipt(terminal, request_ref=entry["request"])
            prior.append(terminal)
    process_terminal(entries[-1][1]["supervisor"])
    if (control/"started.json").exists():
        raise ValueError("cannot repair an allegedly undelivered reservation with a started worker")
    _append_journal_bytes(complete[len(raw):])
    verify_journal(refs)


def process_terminal(identity):
    """No signalling: distinguish absence/reuse/zombie from a live exact PID."""
    from .learned_partition_supervisor import process_identity
    if (not isinstance(identity, dict) or set(identity) != {"pid", "boot_id", "process_start_ticks"}
            or type(identity["pid"]) is not int or identity["pid"] <= 1):
        raise ValueError("missing complete process identity")
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    if identity["boot_id"] != boot:
        return "PRIOR_BOOT"
    try:
        current = process_identity(identity["pid"])
        if current != identity:
            return "PID_REUSED"
        state = Path(f"/proc/{identity['pid']}/stat").read_text().rsplit(")", 1)[1].split()[0]
        if state in ("Z", "X"):
            return "ZOMBIE_OR_DEAD"
    except FileNotFoundError:
        return "PID_ABSENT"
    raise PermissionError("exact owned process is still live; no reconciliation or signalling")


def verify_reconciliation(record, control):
    ref = record["error"]["reconciliation"]
    path = p.verify_reference(ref)
    if path != control/"reconciliation.json":
        raise ValueError("reconciliation evidence is outside the original supervisor")
    evidence = p.read_reference(ref)
    keys = {"status", "request", "budget", "supervisor", "worker", "started", "seconds", "clock", "checked_monotonic"}
    if (set(evidence) != keys or evidence["status"] != "CONFIRMED_TERMINAL_UNKNOWN_EXIT"
            or evidence["request"] != record["request"] or evidence["budget"] != record["budget"]
            or evidence["seconds"] != record["seconds"] or record["status"] != "FAILED"
            or record["worker_exit_code"] is not None or record["result"] is not None):
        raise ValueError("reconciled receipt differs from its exact terminal evidence")
    entry = p.read_reference(record["budget"])
    allowed = {"PRIOR_BOOT", "PID_REUSED", "ZOMBIE_OR_DEAD", "PID_ABSENT"}
    if evidence["supervisor"] not in allowed:
        raise ValueError("reconciliation lacks supervisor death evidence")
    if evidence["started"] is None:
        if evidence["worker"] != "REQUEST_PIPE_NOT_DELIVERED" or record["worker_pid"] is not None or (control/"started.json").exists():
            raise ValueError("missing bootstrap receipt cannot hide a started worker")
    else:
        if p.verify_reference(evidence["started"]) != control/"started.json":
            raise ValueError("reconciliation uses another worker bootstrap")
        started = p.read_reference(evidence["started"])
        if (started["request"] != record["request"] or started["budget"] != record["budget"]
                or started["pid"] != record["worker_pid"] or evidence["worker"] not in allowed):
            raise ValueError("worker identity does not match reconciliation")
    expected = (evidence["checked_monotonic"]-entry["reserved_monotonic"]
                if evidence["clock"] == "SAME_BOOT_ELAPSED_UPPER_BOUND" else CAMPAIGN_SECONDS+CELL_SECONDS)
    if (evidence["clock"] not in {"SAME_BOOT_ELAPSED_UPPER_BOUND", "CLOCK_LOST_EXHAUST_CAMPAIGN"}
            or expected <= 0 or evidence["seconds"] != expected):
        raise ValueError("reconciled time debit is not conservative or reproducible")


def reconcile(permit):
    """Explicit CPU recovery after a dead supervisor, never a fresh allowance.

    Same-boot elapsed time includes the detection delay. A lost monotonic clock
    exhausts the campaign rather than inventing elapsed training time.
    """
    path = p.verify_reference(permit)
    if path.parent != REGISTRY:
        raise ValueError("reconciliation requires a canonical reservation")
    with (REGISTRY/"lease.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        entry = p.read_reference(permit)
        entries = [(p.reference(f), json.loads(f.read_bytes())) for f in REGISTRY.glob("*.json")]
        entries.sort(key=lambda r: len(r[1]["prior_terminals"]))
        if not entries or entries[-1][0] != permit:
            raise ValueError("only the latest interrupted reservation can need reconciliation")
        try:
            verify_journal([r[0] for r in entries])
        except ValueError:
            repair_journal_tail(entries)
        control = safe_member(p.ROOT, entry["control"])
        if control.parent != STAGING or not control.name.startswith("supervisor-"):
            raise ValueError("reconciliation control is not campaign-owned")
        if (control/"terminal.json").exists():
            ref = p.reference(control/"terminal.json")
            terminal_receipt(ref, request_ref=entry["request"])
            return ref
        request = p.read_reference(entry["request"])
        descriptor = json.loads((control/"request.json").read_bytes())
        if descriptor["reference"] != entry["request"] or descriptor["request"] != request:
            raise ValueError("interrupted supervisor request changed")
        supervisor_state = process_terminal(entry["supervisor"])
        start_path = control/"started.json"
        start_ref, worker_pid, worker_state = None, None, "REQUEST_PIPE_NOT_DELIVERED"
        if start_path.exists():
            start_ref = p.reference(start_path)
            start = p.read_reference(start_ref)
            if start["request"] != entry["request"] or start["budget"] != permit:
                raise ValueError("interrupted worker belongs to another reservation")
            worker_pid = start["pid"]
            worker_state = process_terminal({k: start[k] for k in ("pid", "boot_id", "process_start_ticks")})
        now = time.monotonic()
        same_boot = entry["supervisor"]["boot_id"] == Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        seconds = now-entry["reserved_monotonic"] if same_boot else CAMPAIGN_SECONDS+CELL_SECONDS
        if seconds <= 0:
            raise ValueError("cannot recover a positive conservative time debit")
        evidence = {"status": "CONFIRMED_TERMINAL_UNKNOWN_EXIT", "request": entry["request"], "budget": permit,
            "supervisor": supervisor_state, "worker": worker_state, "started": start_ref,
            "seconds": seconds, "checked_monotonic": now,
            "clock": "SAME_BOOT_ELAPSED_UPPER_BOUND" if same_boot else "CLOCK_LOST_EXHAUST_CAMPAIGN"}
        evidence_path = control/"reconciliation.json"
        # A reconciler itself may stop after publishing evidence: preserve it.
        if evidence_path.exists():
            evidence = json.loads(evidence_path.read_bytes())
        else:
            write_json(evidence_path, evidence)
        output = safe_member(p.ROOT, request["output"])
        record = {"status": "FAILED", "request": entry["request"], "request_id": request["request_id"],
            "operation": "train_cell", "output": request["output"], "worker_pid": worker_pid,
            "worker_terminal_confirmed": True, "worker_exit_code": None, "result": None,
            "seconds": evidence["seconds"], "observed_peak_rss_bytes": None, "observed_peak_gpu_bytes": None,
            "error": {"reconciliation": p.reference(evidence_path)}, "budget": permit,
            "recovery_status": recovery_status(output, request)}
        verify_reconciliation(record, control)
        if output.is_dir():
            mark_failure(output, RuntimeError("supervisor lost; reconciled FAILED with unknown exit status"))
        write_json(control/"terminal.json", record)
        return p.reference(control/"terminal.json")


@contextmanager
def reserve(request_ref, request, control):
    """Reserve remaining limits while holding the one campaign-owned lock."""
    from .learned_partition_supervisor import process_identity
    cell = cell_identity(request)
    if p.read_reference(request_ref) != request:
        raise ValueError("training request changed before its budget reservation")
    control = Path(control)
    if control.parent != STAGING or not control.name.startswith("supervisor-"):
        raise ValueError("budget reservation requires the owned supervisor directory")
    REGISTRY.mkdir(exist_ok=True)
    with (REGISTRY/"lease.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        entries, total, cells = accounting()
        prior = [row for row in entries if cell_identity(row["request"]) == cell]
        if any(row["record"]["status"] == "COMPLETE" for row in prior):
            raise PermissionError("cell already complete; reuse it instead of retraining")
        resume = request["arguments"].get("resume")
        if prior:
            if (not isinstance(resume, dict) or set(resume) != {"request", "terminal", "snapshot"}
                    or resume["request"] != prior[-1]["record"]["request"]
                    or resume["terminal"] != prior[-1]["terminal"]):
                raise PermissionError("failed cell must resume an exact registered parent, not reset initialization")
        elif resume is not None:
            raise ValueError("resume parent does not belong to this campaign registry")
        remaining = min(CELL_SECONDS-cells.get(tuple(cell.values()), 0.), CAMPAIGN_SECONDS-total)
        if remaining <= 0:
            raise PermissionError("cumulative training budget exhausted; no fresh attempt allowance")
        entry = {"request": request_ref, "control": control.relative_to(p.ROOT).as_posix(), "cell": cell,
            "supervisor": process_identity(os.getpid()), "remaining_seconds": remaining,
            "reserved_monotonic": json.loads((control/"request.json").read_bytes()).get("monotonic_started", time.monotonic()),
            "prior_terminals": [row["terminal"] for row in entries]}
        path = REGISTRY/(request_ref["sha256"]+".json")
        if not entries:
            write_json(STAGING/"budget_origin.json", {"schema": "learned_training_budget_origin_v1", "first_request": request_ref})
        write_json(path, entry)
        ref = p.reference(path)
        append_journal(ref, len(entries))
        yield ref, remaining


def verify_permit(ref, request_ref):
    """Worker validates the live supervising parent and its immutable allowance."""
    from .learned_partition_supervisor import process_identity
    path = p.verify_reference(ref)
    if path.parent != REGISTRY or path.name != request_ref["sha256"]+".json":
        raise PermissionError("training requires its canonical budget reservation")
    entry = p.read_reference(ref)
    if entry["request"] != request_ref or entry["supervisor"] != process_identity(os.getppid()):
        raise PermissionError("training budget parent is absent or differs from the live supervisor")
    if not 0 < entry["remaining_seconds"] <= CELL_SECONDS:
        raise ValueError("invalid remaining training allowance")
    return entry
