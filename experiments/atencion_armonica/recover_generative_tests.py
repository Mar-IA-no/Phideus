"""One-off recovery supervisor for the preserved fresh namespace failure.

The parent is metadata-only.  The bounded worker authenticates and adopts the
already drawn IID namespace, publishes an immutable reconciliation receipt,
then delegates every scientific stage to the frozen original supervisor.
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
from src.atencion_armonica.partial_compatibility_cache import encoded
from experiments.atencion_armonica import run_generative_tests as old


CONTROL = old.TEMP/"test-recovery-control"
MANIFEST = CONTROL/"manifest.json"
RECEIPT = CONTROL/"namespace-reconciliation.json"
BINDING = old.FRESH/"binding.json"
FINAL = old.FRESH/"test_completion.json"
PLAN = ROOT/"Biblioteca/Geometria_Proporcional_Ground_Truth/notes/20260909_fresh_namespace_recovery_plan.md"
AUDIT = ROOT/"Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/732_generative_fresh_namespace_recovery_plan_audit.md"
PLAN_SHA = "4c7699eebc817d3fd99968237def067907040f2a304012dda6ea0e7542271efe"
AUDIT_SHA = "aed1943a4a4ff04edd49620b4bff425d30688e98348a29bbba008ad9559e9d9e"
ORIGINAL_MANIFEST = old.CONTROL/"manifest.json"
ORIGINAL_SECONDS = 120.09713516899501
ORIGINAL_MANIFEST_SHA = "f7cfa78612e841cbc544b96e4fc636cf03d11329a6cb78c849e22e0cc8505634"
FREEZE_SHA = "8b1030ca466ef289eeef738149b878886a640fbc598774cb6fabd01147ee7937"
IID_SHA = "3472ed7a63f64d6597ca73a66628658f606ef14323a3039e179c57eec95ddefe"
FAILURE = "ValueError: cannot adopt an unbound existing fresh store"
ORIGINAL_FILES = {
    "attempt-0000.launch.json": "5c213680b34187196ea9f1865ece995c29843a9101182cb679edfd142543d71c",
    "attempt-0000.exit.json": "4799fea22d6d95f71ed08236132adb3cf184783ea90fa7cfc41c96455277f7a8",
    "attempt-0000.worker.json": "ea030f8719f551bb926b5ce5408fdf9495658dbd23387837edeaf96bc7942997",
    "attempt-0001.launch.json": "34ed99fc7b88e6041a49304ca547dd7bb3622ea30912d553d83d99bd676a6957",
    "attempt-0001.exit.json": "ba330c9e802b99c50248c7cdff254ab5d0a3a5d3d3ef6e04361e1579f401ead2",
    "attempt-0001.worker.json": "b8eb43c3658917a8c380306f17eb519d0f7237078aec6043cef66e33fe11854f",
}


def reference(path):
    path = Path(path).absolute()
    if not path.is_relative_to(ROOT):
        raise ValueError("recovery reference escapes project")
    for ancestor in (path, *path.parents):
        if ancestor == ROOT:
            break
        if ancestor.is_symlink():
            raise ValueError("recovery references must not traverse symlinks")
    raw = path.read_bytes()
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": hashlib.sha256(raw).hexdigest()}


def _read(ref):
    return VerifiedBytes(ROOT).json(ref)


def _valid_ref(value):
    return (isinstance(value, dict) and set(value) == {"path", "sha256"}
            and isinstance(value["path"], str) and bool(value["path"])
            and isinstance(value["sha256"], str) and len(value["sha256"]) == 64
            and all(c in "0123456789abcdef" for c in value["sha256"]))


def sources():
    paths = [Path(__file__).resolve(), PLAN, AUDIT]
    result = {}
    for path in paths:
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"recovery source missing: {path.relative_to(ROOT)}")
        result[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    if (result[PLAN.relative_to(ROOT).as_posix()] != PLAN_SHA
            or result[AUDIT.relative_to(ROOT).as_posix()] != AUDIT_SHA):
        raise ValueError("recovery plan or independent approval differs from the audited version")
    return result


def verify_original():
    manifest_ref = reference(ORIGINAL_MANIFEST)
    if manifest_ref["sha256"] != ORIGINAL_MANIFEST_SHA:
        raise ValueError("original manifest differs from the failed execution")
    manifest = old.read_manifest(manifest_ref)
    paths = old._launches()
    if [p.name for p in paths] != ["attempt-0000.launch.json", "attempt-0001.launch.json"]:
        raise ValueError("original attempt prefix is not the exact two terminal attempts")
    for name, digest in ORIGINAL_FILES.items():
        if reference(old.CONTROL/name)["sha256"] != digest:
            raise ValueError(f"original receipt changed: {name}")
    state = old.test_accumulated(manifest_ref)
    failed = _read(reference(old.CONTROL/"attempt-0001.worker.json"))
    if (state["next_stage"] != 1 or len(state["completions"]) != 1
            or state["last_status"] != "FAILED" or state["seconds"] != ORIGINAL_SECONDS
            or failed.get("status") != "FAILED" or failed.get("reason") != FAILURE):
        raise ValueError("original terminal state is not the fixed namespace failure")
    freeze_ref = reference(old._freeze().FREEZE)
    iid_ref = reference(old.FRESH/"draws/iid/index.json")
    if freeze_ref["sha256"] != FREEZE_SHA or iid_ref["sha256"] != IID_SHA:
        raise ValueError("preserved freeze or IID index differs")
    return {"manifest": manifest_ref, "value": manifest, "state": state,
            "freeze": freeze_ref, "iid": iid_ref,
            "last_exit": reference(old.CONTROL/"attempt-0001.exit.json")}


def _manifest_value(original):
    return {"schema": "generative-evidence-fresh-test-recovery-v1",
        "stage": "EXACT_NAMESPACE_RECOVERY_NOT_PROMOTED",
        "plan": reference(PLAN), "plan_audit": reference(AUDIT), "sources": sources(),
        "original_manifest": original["manifest"], "original_last_exit": original["last_exit"],
        "original_seconds": ORIGINAL_SECONDS, "freeze": original["freeze"],
        "iid_index": original["iid"], "stages": original["value"]["stages"][1:],
        "limits": original["value"]["limits"], "output": original["value"]["output"],
        "policy": {"one_off_exact_failure": True, "no_redraw": True,
                   "scientific_executor": original["manifest"], "promotion": False}}


def initialize_recovery():
    training = old._training()
    with training._lock(old.COMMON_LOCK), training._lock(CONTROL/"operator.lock"):
        if MANIFEST.exists():
            raise FileExistsError("recovery manifest already exists; use --run")
        original = verify_original()
        CONTROL.mkdir(parents=True, exist_ok=True)
        write_json(MANIFEST, _manifest_value(original))
        return reference(MANIFEST)


def read_manifest(ref):
    if not _valid_ref(ref) or ref["path"] != MANIFEST.relative_to(ROOT).as_posix():
        raise ValueError("unexpected recovery manifest")
    value = _read(ref)
    expected = _manifest_value(verify_original())
    if value != expected:
        raise ValueError("recovery manifest or exact authority differs")
    return value


def _regular(path):
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"recovery requires regular file: {path}")


def _directory(path):
    if not path.is_dir() or path.is_symlink():
        raise ValueError(f"recovery requires real directory: {path}")


def _strict_bootstrap_tree(binding_present):
    _directory(old.FRESH)
    expected = {"draws"} | ({"binding.json"} if binding_present else set())
    if {p.name for p in old.FRESH.iterdir()} != expected:
        raise ValueError("unreceipted fresh root contains files beyond the exact bootstrap tree")
    draws = old.FRESH/"draws"
    _directory(draws)
    if {p.name for p in draws.iterdir()} != {"iid"}:
        raise ValueError("unreceipted draws contain another split")
    iid = draws/"iid"
    _directory(iid)
    if {p.name for p in iid.iterdir()} != {"index.json", *[f"{i:05d}" for i in range(512)]}:
        raise ValueError("IID bootstrap roster differs")
    _regular(iid/"index.json")
    for scene_id in range(512):
        folder = iid/f"{scene_id:05d}"
        _directory(folder)
        if {p.name for p in folder.iterdir()} != {"intent.json", "observation.json", "sidecar.json", "draw.json"}:
            raise ValueError("IID scene bootstrap inventory differs")
        for path in folder.iterdir():
            _regular(path)


def _binding_value(original):
    return {"test_freeze": original["freeze"]}


def _receipt_value(manifest_ref, original, binding_ref):
    return {"schema": "generative-evidence-fresh-namespace-reconciliation-v1",
        "manifest": manifest_ref, "original_manifest": original["manifest"],
        "original_last_exit": original["last_exit"], "failure": FAILURE,
        "freeze": original["freeze"], "iid_before": original["iid"],
        "iid_after": original["iid"], "binding": binding_ref,
        "status": "EXACT_IID_NAMESPACE_RECONCILED_NO_SCIENTIFIC_OUTPUT"}


def _namespace_state(manifest_ref):
    original = verify_original()
    if RECEIPT.exists() or RECEIPT.is_symlink():
        return "C", verify_reconciliation(manifest_ref, deep=False, check=None)
    if BINDING.exists() or BINDING.is_symlink():
        _regular(BINDING)
        if VerifiedBytes(ROOT).read(reference(BINDING)) != encoded(_binding_value(original)):
            raise ValueError("unreceipted binding differs")
        return "B", None
    return "A", None


def verify_reconciliation(manifest_ref, *, deep, check):
    original = verify_original()
    _regular(BINDING)
    _regular(RECEIPT)
    binding_ref = reference(BINDING)
    if VerifiedBytes(ROOT).read(binding_ref) != encoded(_binding_value(original)):
        raise ValueError("recovery binding differs")
    receipt_ref = reference(RECEIPT)
    if _read(receipt_ref) != _receipt_value(manifest_ref, original, binding_ref):
        raise ValueError("namespace reconciliation receipt differs")
    if reference(old.FRESH/"draws/iid/index.json") != original["iid"]:
        raise ValueError("IID index changed after reconciliation")
    if deep:
        _authenticate_iid(original, check)
        from src.atencion_armonica.generative_evidence_fresh_store import FreshObservableStore
        FreshObservableStore(old.FRESH, binding=_binding_value(original))
    return receipt_ref


def _authenticate_iid(original, check):
    if not callable(check):
        raise TypeError("IID authentication requires a bounded worker callback")
    from src.atencion_armonica.generative_evidence_fresh_data import FreshObservations
    observed = FreshObservations("iid", freeze_ref=original["freeze"], check=check)
    # The data port returns a DRAW_ROOT-relative reference, not a ROOT reference.
    if (observed.files.root != old.FRESH/"draws"
            or observed.reference != {"path": "iid/index.json", "sha256": original["iid"]["sha256"]}):
        raise ValueError("authenticated IID observations differ")


def reconcile_namespace(manifest_ref, check):
    read_manifest(manifest_ref)
    check()
    state, receipt = _namespace_state(manifest_ref)
    if state == "C":
        return verify_reconciliation(manifest_ref, deep=True, check=check)
    original = verify_original()
    _strict_bootstrap_tree(state == "B")
    _authenticate_iid(original, check)
    check()
    before = reference(old.FRESH/"draws/iid/index.json")
    if state == "A":
        write_json(BINDING, _binding_value(original))
    check()  # A cooperative pause here leaves the exact state B.
    binding_ref = reference(BINDING)
    from src.atencion_armonica.generative_evidence_fresh_store import FreshObservableStore
    FreshObservableStore(old.FRESH, binding=_binding_value(original))
    after = reference(old.FRESH/"draws/iid/index.json")
    if before != original["iid"] or after != before:
        raise ValueError("reconciliation changed the IID index")
    write_json(RECEIPT, _receipt_value(manifest_ref, original, binding_ref))
    check()  # The receipt exists before any delegated scientific output.
    return verify_reconciliation(manifest_ref, deep=True, check=check)


def _service_command(unit, runtime_seconds, stage, attempt):
    command = old._service_command(unit, runtime_seconds, stage, attempt)
    token = "experiments.atencion_armonica.run_generative_tests"
    positions = [i for i, value in enumerate(command) if value == token]
    if len(positions) != 1:
        raise ValueError("original service command has no unique worker module token")
    command[positions[0]] = "experiments.atencion_armonica.recover_generative_tests"
    return command


def _launches():
    return sorted(CONTROL.glob("attempt-*.launch.json"))


def _attempt_name(value):
    return old._attempt_name(value)


def _launch_state(manifest_ref):
    state, receipt = _namespace_state(manifest_ref)
    return state, receipt if state == "C" else None


def _prefix(manifest_ref, paths):
    manifest = read_manifest(manifest_ref)
    original = verify_original()
    roster = old.stage_roster()
    all_paths = _launches()
    if [p.name for p in all_paths] != [f"attempt-{i:04d}.launch.json" for i in range(len(all_paths))]:
        raise ValueError("recovery attempt indices are not contiguous")
    if all_paths[:len(paths)] != list(paths):
        raise ValueError("recovery accounting is not an exact ledger prefix")
    total = ORIGINAL_SECONDS
    previous = original["last_exit"]
    next_stage = 1
    completions = list(original["state"]["completions"])
    last_status = "ORIGINAL_FAILURE_ACCEPTED_BY_EXACT_RECOVERY"
    current_receipt = (verify_reconciliation(manifest_ref, deep=False, check=None)
                       if RECEIPT.exists() or RECEIPT.is_symlink() else None)
    for path in paths:
        if next_stage >= len(roster) or last_status == "FAILED":
            raise ValueError("recovery ledger continued beyond an eligible stage")
        launch_ref = reference(path)
        launch = _read(launch_ref)
        stage = roster[next_stage]
        remaining = math.floor(old.LIMIT_SECONDS-total)
        attempt = path.name.removesuffix(".launch.json")
        keys = {"schema", "manifest", "stage", "previous_exit", "unit", "command",
                "availability", "used_seconds", "remaining_seconds", "runtime_seconds",
                "bootstrap_state", "reconciliation"}
        bootstrap = launch.get("bootstrap_state")
        reconciliation = launch.get("reconciliation")
        if bootstrap in ("A", "B"):
            bootstrap_ok = stage["index"] == 1 and reconciliation is None
        else:
            bootstrap_ok = (bootstrap == "C" and _valid_ref(reconciliation)
                            and reconciliation == current_receipt)
        availability = launch.get("availability")
        if (set(launch) != keys or launch.get("schema") != "generative-evidence-fresh-test-recovery-launch-v1"
                or launch.get("manifest") != manifest_ref or launch.get("stage") != stage
                or launch.get("previous_exit") != previous or launch.get("used_seconds") != total
                or launch.get("remaining_seconds") != remaining
                or launch.get("runtime_seconds") != remaining-old.GRACE or not bootstrap_ok
                or launch.get("command") != _service_command(launch.get("unit"),
                    launch.get("runtime_seconds"), stage, attempt)
                or (stage["device"] == "cpu" and availability is not None)
                or (stage["device"] == "cuda:0" and not old._availability_ok(availability))):
            raise ValueError("recovery launch chain, authority, resources or command differs")
        exit_path = path.with_name(path.name.replace(".launch.json", ".exit.json"))
        worker_path = path.with_name(path.name.replace(".launch.json", ".worker.json"))
        if not exit_path.exists() or not worker_path.exists():
            raise RuntimeError(f"unreconciled recovery attempt: {launch.get('unit')}")
        exit_ref, worker_ref = reference(exit_path), reference(worker_path)
        end, worker = _read(exit_ref), _read(worker_ref)
        if (end.get("launch") != launch_ref or end.get("terminal") is not True
                or not old._number(end.get("seconds")) or worker.get("launch") != launch_ref):
            raise ValueError("recovery attempt lacks matching terminal receipts")
        status = worker.get("status")
        output = old._stage_reference(stage) if status == old._success_status(stage) else None
        if not old._worker_schema(worker, end.get("process_returncode"), stage, output):
            raise ValueError("recovery worker receipt is untyped or names another output")
        total += end["seconds"]
        previous = exit_ref
        last_status = status
        if status == old._success_status(stage):
            if current_receipt is None:
                raise ValueError("scientific completion lacks namespace reconciliation")
            completions.append({"stage": stage, "launch": launch_ref, "exit": exit_ref,
                                "worker": worker_ref, "output": output})
            next_stage += 1
            last_status = "COMPLETE"
    return {"seconds": float(total), "previous_exit": previous, "next_stage": next_stage,
            "completions": completions, "last_status": last_status}


def recovery_accumulated(manifest_ref):
    return _prefix(manifest_ref, _launches())


def _validate_tail(attempt):
    _attempt_name(attempt)
    launch_path = CONTROL/f"{attempt}.launch.json"
    paths = _launches()
    if not paths or paths[-1] != launch_path:
        raise ValueError("recovery worker launch is not the unique ledger tail")
    launch_ref, launch = reference(launch_path), _read(reference(launch_path))
    prefix = _prefix(launch["manifest"], paths[:-1])
    if prefix["next_stage"] >= 13 or prefix["last_status"] == "FAILED":
        raise ValueError("recovery worker follows no eligible stage")
    stage = old.stage_roster()[prefix["next_stage"]]
    remaining = math.floor(old.LIMIT_SECONDS-prefix["seconds"])
    keys = {"schema", "manifest", "stage", "previous_exit", "unit", "command",
            "availability", "used_seconds", "remaining_seconds", "runtime_seconds",
            "bootstrap_state", "reconciliation"}
    if (set(launch) != keys or launch.get("schema") != "generative-evidence-fresh-test-recovery-launch-v1"
            or launch.get("stage") != stage or launch.get("previous_exit") != prefix["previous_exit"]
            or launch.get("used_seconds") != prefix["seconds"]
            or launch.get("remaining_seconds") != remaining
            or launch.get("runtime_seconds") != remaining-old.GRACE
            or launch.get("command") != _service_command(launch.get("unit"),
                launch.get("runtime_seconds"), stage, attempt)
            or (stage["device"] == "cpu" and launch.get("availability") is not None)
            or (stage["device"] == "cuda:0" and not old._availability_ok(launch.get("availability")))):
        raise ValueError("recovery worker launch, budget or stage differs")
    live_state, live_receipt = _launch_state(launch["manifest"])
    if launch.get("bootstrap_state") != live_state or launch.get("reconciliation") != live_receipt:
        raise ValueError("recovery namespace changed between parent and worker")
    if stage["index"] != 1 and live_state != "C":
        raise ValueError("later scientific stages require prior reconciliation")
    return launch_ref, launch, stage


def worker_stage(attempt):
    launch_ref, launch, stage = _validate_tail(attempt)
    started, stopped, last_disk = time.monotonic(), [], [0.]
    def stop(signum, frame):
        stopped.append(signum)
    handlers = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    torch = None
    availability = None
    report = {"launch": launch_ref, "stage": stage, "status": "INCOMPLETE"}
    code = 1
    try:
        old._training()._verify_service(launch["unit"], launch["runtime_seconds"])
        manifest = read_manifest(launch["manifest"])
        import torch as torch_module
        torch = torch_module
        if torch.cuda.is_initialized():
            raise RuntimeError("CUDA initialized before recovery worker device checks")
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        if (any(os.environ.get(name) != "1" for name in
                ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"))
                or os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8"):
            raise RuntimeError("recovery worker deterministic environment differs")
        if stage["device"] == "cuda:0":
            if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
                raise RuntimeError("GPU recovery stage lacks its exact visible device")
            availability = old._profile().gpu_availability()
            if not old._availability_ok(availability):
                raise RuntimeError("GPU unavailable before recovery")
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.cuda.set_device(0)
            fraction = old.MEMORY_LIMIT/torch.cuda.get_device_properties(0).total_memory
            if not 0 < fraction <= 1:
                raise RuntimeError("recovery GPU fraction is invalid")
            torch.cuda.set_per_process_memory_fraction(fraction, 0)
            torch.cuda.reset_peak_memory_stats(0)
        elif os.environ.get("CUDA_VISIBLE_DEVICES") != "":
            raise RuntimeError("CPU recovery stage must hide CUDA")
        def check():
            now = time.monotonic()
            due = now-last_disk[0] >= 30
            active = torch if stage["device"] == "cuda:0" and torch.cuda.is_initialized() else None
            old._training()._worker_check(started, launch["runtime_seconds"], stopped,
                                          torch=active, check_storage=due)
            if due:
                last_disk[0] = now
        def progress(value):
            print(value if isinstance(value, str) else json.dumps(value, sort_keys=True), flush=True)
        check()
        if stage["index"] == 1:
            reconcile_namespace(launch["manifest"], check)
        else:
            verify_reconciliation(launch["manifest"], deep=False, check=check)
        output = old._execute_stage(stage, verify_original()["value"], check, progress)
        expected = old._stage_reference(stage)
        if output != expected:
            raise ValueError("recovered stage returned another output")
        check()
        report.update(status=old._success_status(stage), output=expected)
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
        peak = torch.cuda.max_memory_reserved(0) if initialized and stage["device"] == "cuda:0" else 0
        report.update(seconds=time.monotonic()-started,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            peak_reserved_bytes=peak, cuda_initialized=initialized, availability=availability)
        write_json(CONTROL/f"{attempt}.worker.json", report)
    return code


def verified_tests():
    manifest_ref = reference(MANIFEST)
    state = recovery_accumulated(manifest_ref)
    if state["seconds"] > old.LIMIT_SECONDS:
        raise RuntimeError("recovered tests exceeded the inherited four-hour budget")
    verify_reconciliation(manifest_ref, deep=False, check=None)
    expected = old._final_value(manifest_ref, state)
    ref = reference(FINAL)
    if _read(ref) != expected:
        raise ValueError("recovered final index differs from both typed execution chains")
    return ref


def run_recovery():
    training = old._training()
    with training._lock(old.COMMON_LOCK), training._lock(CONTROL/"operator.lock"):
        manifest_ref = reference(MANIFEST)
        read_manifest(manifest_ref)
        while True:
            state = recovery_accumulated(manifest_ref)
            if state["seconds"] > old.LIMIT_SECONDS:
                raise RuntimeError("recovery exceeded the inherited four-hour budget")
            if state["next_stage"] == 13:
                verify_reconciliation(manifest_ref, deep=False, check=None)
                old._publish_final(manifest_ref, state)
                return verified_tests()
            if state["last_status"] == "FAILED":
                raise RuntimeError("new failed recovery attempt cannot be retried automatically")
            remaining = math.floor(old.LIMIT_SECONDS-state["seconds"])
            if remaining <= old.GRACE+10:
                raise RuntimeError("inherited fresh-test time exhausted")
            training.check_disk()
            stage = old.stage_roster()[state["next_stage"]]
            bootstrap, reconciliation = _launch_state(manifest_ref)
            if bootstrap != "C":
                if stage["index"] != 1:
                    raise ValueError("only IID may bootstrap the namespace")
                _strict_bootstrap_tree(bootstrap == "B")
            availability = None
            if stage["device"] == "cuda:0":
                availability = old._profile().gpu_availability()
                if not old._availability_ok(availability):
                    raise RuntimeError("GPU unavailable before recovery launch")
            attempt = f"attempt-{len(_launches()):04d}"
            unit = "phideus-generative-test-"+uuid.uuid4().hex[:16]
            command = _service_command(unit, remaining-old.GRACE, stage, attempt)
            launch_path = CONTROL/f"{attempt}.launch.json"
            write_json(launch_path, {
                "schema": "generative-evidence-fresh-test-recovery-launch-v1",
                "manifest": manifest_ref, "stage": stage, "previous_exit": state["previous_exit"],
                "unit": unit, "command": command, "availability": availability,
                "used_seconds": state["seconds"], "remaining_seconds": remaining,
                "runtime_seconds": remaining-old.GRACE, "bootstrap_state": bootstrap,
                "reconciliation": reconciliation})
            end = old._execute_command(command, unit, reference(launch_path),
                                       CONTROL/f"{attempt}.exit.json")
            if end.get("process_returncode") == 75:
                raise InterruptedError("recovery paused at a durable boundary")
            if end.get("process_returncode") != 0:
                raise RuntimeError("recovery stage did not complete")
            updated = recovery_accumulated(manifest_ref)
            if updated["next_stage"] != state["next_stage"]+1:
                raise RuntimeError("recovery exit did not complete its typed stage")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--initialize", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--attempt", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        if args.attempt is None:
            parser.error("worker requires --attempt")
        sys.exit(worker_stage(args.attempt))
    if args.attempt is not None:
        parser.error("--attempt is worker-only")
    print(json.dumps(initialize_recovery() if args.initialize else run_recovery(), sort_keys=True))


if __name__ == "__main__":
    main()
