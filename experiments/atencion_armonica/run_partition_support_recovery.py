"""Finite support recovery: owned CPU children and a typed four-test roster."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import resource
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.atencion_armonica import learned_partition_provenance as p
from src.atencion_armonica import partition_support_recovery as gate
from src.atencion_armonica import partition_test_recovery_gate as old_gate
from src.atencion_armonica.partition_test_recovery_supervisor import supervise_process
from src.atencion_armonica.structured_source_artifacts import write_json, mark_failure
from experiments.atencion_armonica import operate_partition_test_recovery as old_operator

INPUTS = {"split", "authorization", "data", "logits", "scored", "normalized",
          "recovery_authorization", "support_recovery_authorization"}
ARGUMENTS = {"inference": INPUTS | {"gpu_grant"}, "evaluation": INPUTS | {"predictions"}}


def validate_request(ref):
    request = p.read_reference(ref)
    root = p.ROOT/gate.RECOVERY
    if (set(request) != {"schema", "request_id", "operation", "output", "arguments", "execution_contract"}
            or request["schema"] != "partition-support-recovery-request-v1"
            or not isinstance(request["request_id"], str) or not request["request_id"].strip()
            or request["operation"] not in ARGUMENTS or not isinstance(request["arguments"], dict)
            or set(request["arguments"]) != ARGUMENTS[request["operation"]]):
        raise ValueError("closed recovery request schema differs")
    arguments = request["arguments"]
    if arguments["split"] not in gate.SPLITS_NEW or arguments["authorization"] != gate.FROZEN["test_authorization"]:
        raise PermissionError("recovery is only for the four declared frozen tests")
    if request["operation"] == "inference" and arguments["gpu_grant"] is not None:
        raise PermissionError("the versioned recovery heads are CPU-only")
    if arguments["recovery_authorization"] != gate.OLD_AUTH:
        raise PermissionError("upstream recovery authority differs")
    auth, _ = gate.verify_authorization(arguments["support_recovery_authorization"])
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
    from src.atencion_armonica.partition_support_recovery import infer_test, evaluate_test
    fn = {"inference": infer_test, "evaluation": evaluate_test}[request["operation"]]
    result = fn(output, **request["arguments"])
    if p.read_reference(ref) != request:
        raise ValueError("recovery request changed during execution")
    gate.verify_contract(request["execution_contract"])
    return result

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
    command = [sys.executable, str(p.ROOT/"experiments/atencion_armonica/run_partition_support_recovery.py"), "--worker-fd"]
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
        gate.checked_bundle(outcome["result"], request["operation"], request["arguments"]["support_recovery_authorization"])
        if request["operation"] == "evaluation":
            gate.checked_evaluation(outcome["result"], **request["arguments"])
    except BaseException as exc:
        outcome["error"] = repr(exc)
        if output.is_dir():
            mark_failure(output, exc)
        mark_failure(control, exc)
    receipt = {"schema": "partition-support-recovery-terminal-v1", **outcome,
        "status": "COMPLETE" if outcome["error"] is None else "FAILED",
        "request": ref, "request_id": request["request_id"], "operation": request["operation"],
        "output": request["output"], "execution_contract": request["execution_contract"],
        "budget": None, "recovery_status": "NOT_TRAINING"}
    write_json(control/"terminal.json", receipt)
    if receipt["status"] != "COMPLETE":
        raise RuntimeError(f"recovery stopped; inspect {control.relative_to(p.ROOT)}/terminal.json")
    return {"result": outcome["result"], "supervisor": p.reference(control/"terminal.json")}


def step(name, operation, output, arguments):
    root = p.ROOT/gate.RECOVERY
    request_path = gate.canonical_path(root/"requests"/(name+"_01.json"), root/"requests")
    saved = gate.canonical_path(root/"receipts"/(name+"_01.json"), root/"receipts")
    output = gate.canonical_path(output, root/"outputs"/arguments["split"])
    auth, _ = gate.verify_authorization(arguments["support_recovery_authorization"])
    request = {"schema": "partition-support-recovery-request-v1",
        "request_id": "phideus-support-"+name+"-20260908-01", "operation": operation,
        "output": output.relative_to(p.ROOT).as_posix(), "arguments": arguments,
        "execution_contract": auth["contract"]}
    if saved.exists():
        value = old_operator.completed(saved, request_path, request, output)
        gate.checked_bundle(value["result"], operation, arguments["support_recovery_authorization"])
        if operation == "evaluation":
            gate.checked_evaluation(value["result"], **arguments)
        print(json.dumps({"event": "REUSED", "name": name}), flush=True)
        return value
    if request_path.exists():
        raise RuntimeError("request exists without receipt; explicit recovery required")
    if output.exists() or output.with_name(output.name+".FAILURE.json").exists():
        raise FileExistsError("support recovery output already exists")
    for parent in (request_path.parent, saved.parent, output.parent):
        parent.mkdir(parents=True, exist_ok=True)
    write_json(request_path, request)
    print(json.dumps({"event": "STARTED", "name": name, "operator_pid": os.getpid(),
                      "request": p.reference(request_path)}), flush=True)
    value = supervised(p.reference(request_path))
    write_json(saved, value)
    print(json.dumps({"event": "COMPLETE", "name": name, **value}), flush=True)
    return value


NORMALIZATION_AUDIT = {"path": old_gate.RECOVERY + "/authorization/normalization_audit_02.json",
    "sha256": "c7f1b3f52638a7289dab3543a0d4feffd2caeb2d9ca5303a67c4842ab847e484"}


def check_roster(record, support_authorization):
    """Typed consumers retain IID/beta02 and support recovery as distinct producers."""
    if (set(record) != {"schema", "support_authorization", "normalization_audit", "splits"}
            or record["schema"] != "partition-support-mixed-test-roster-v1"
            or record["support_authorization"] != support_authorization
            or record["normalization_audit"] != NORMALIZATION_AUDIT
            or set(record["splits"]) != set(gate.TESTS)):
        raise ValueError("mixed test roster identity or exact four splits differ")
    gate.verify_authorization(support_authorization)
    for split in gate.TESTS:
        case = record["splits"][split]
        if set(case) != {"executor", "data", "aggregate", "logits", "scored", "normalized",
                         "predictions", "evaluation", "replay"}:
            raise ValueError("mixed test entry schema differs")
        new = split in gate.SPLITS_NEW
        if case["executor"] != ("support_v1" if new else "memory02"):
            raise ValueError("test split was reassigned to another executor")
        arguments = {k: case[k] for k in ("data", "logits", "scored", "normalized", "predictions")}
        arguments.update(authorization=gate.FROZEN["test_authorization"], recovery_authorization=gate.OLD_AUTH)
        consumer = gate if new else old_gate
        if new:
            arguments["support_recovery_authorization"] = support_authorization
        for key in ("evaluation", "replay"):
            envelope = case[key]
            if set(envelope) != {"result", "supervisor"}:
                raise ValueError("test completion envelope differs")
            terminal = p.read_reference(envelope["supervisor"])
            request = p.read_reference(terminal["request"])
            if (terminal["status"] != "COMPLETE" or terminal["worker_terminal_confirmed"] is not True
                    or terminal["worker_exit_code"] != 0 or terminal["result"] != envelope["result"]
                    or terminal["operation"] != "evaluation" or request["operation"] != "evaluation"
                    or request["arguments"] != {"split": split, **arguments}
                    or request["output"] != terminal["output"]
                    or p.verify_reference(envelope["result"]) != p.ROOT/terminal["output"]/"manifest.json"):
                raise ValueError("evaluation terminal/request/result binding differs")
        consumer.compare_evaluation_replay(case["evaluation"]["result"], case["replay"]["result"], split, **arguments)
    old_operator.verify_normalization_audit(NORMALIZATION_AUDIT, contract=gate.OLD_CONTRACT,
        target=record["splits"]["iid"]["normalized"])
    return record


def run(support_authorization):
    gate.verify_authorization(support_authorization)
    _, contract = old_gate.verify_authorization(gate.OLD_AUTH)
    frozen = p.read_reference(gate.FROZEN["freeze"])
    test_auth = gate.FROZEN["test_authorization"]
    previous = {"train": frozen["train_data"], "calibration": frozen["calibration_data"]}
    grant = p.reference(p.ROOT/gate.LOCAL/"gpu_grant.json")
    roster = {}
    for split in gate.TESTS:
        def upstream(name, operation, path, **arguments):
            return old_operator.step(name, operation, p.ROOT/gate.TREE/path, arguments,
                recovery_auth=gate.OLD_AUTH, contract=contract, canonical=True)["result"]
        data = upstream(split+"_00_data", "prepare", f"{split}/shard_00/data", split=split,
            shard=0, authorization=test_auth, previous=previous.copy(), earlier_shards=[])
        aggregate = upstream(split+"_data", "aggregate_data", f"{split}/aggregate", split=split,
            authorization=test_auth, previous=previous.copy(), shards=[data])
        logits = upstream(split+"_00_logits", "forward", f"{split}/shard_00/logits", split=split,
            shard=0, authorization=test_auth, data=data, gpu_grant=grant)
        scored = upstream(split+"_00_scored", "score", f"{split}/shard_00/scored", split=split,
            shard=0, authorization=test_auth, data=data, logits=logits)
        def memory(name, operation, **arguments):
            return old_operator.step(split+"_"+name, operation,
                p.ROOT/old_gate.RECOVERY/"outputs"/split/(name+"_01"),
                {**arguments, "recovery_authorization": gate.OLD_AUTH},
                recovery_auth=gate.OLD_AUTH, contract=contract)
        normalized = memory("normalized", "normalized", split=split, shard=0, authorization=test_auth,
            data=data, logits=logits, scored=scored, normalizers=frozen["normalizers"], train=frozen["train"])["result"]
        if split == "iid":
            old_operator.verify_normalization_audit(NORMALIZATION_AUDIT, contract=gate.OLD_CONTRACT, target=normalized)
        inputs = dict(split=split, authorization=test_auth, data=data, logits=logits, scored=scored, normalized=normalized)
        if split in gate.SPLITS_NEW:
            def stage(name, operation, **arguments):
                return step(split+"_"+name, operation, p.ROOT/gate.RECOVERY/"outputs"/split/(name+"_01"),
                    {**arguments, "recovery_authorization": gate.OLD_AUTH,
                     "support_recovery_authorization": support_authorization})
            consumer = gate
            extra = {"support_recovery_authorization": support_authorization}
        else:
            # These requests already completed: never produce new IID/beta work.
            for name in ("predictions", "evaluation", "replay"):
                if not (p.ROOT/old_gate.RECOVERY/"receipts"/(split+"_"+name+"_01.json")).is_file():
                    raise PermissionError("old completed tests must be preserved, not regenerated")
            stage, consumer, extra = memory, old_gate, {}
        predictions = stage("predictions", "inference", **inputs, gpu_grant=None)["result"]
        evaluation = stage("evaluation", "evaluation", **inputs, predictions=predictions)
        replay = stage("replay", "evaluation", **inputs, predictions=predictions)
        consumer.compare_evaluation_replay(evaluation["result"], replay["result"], **inputs,
            predictions=predictions, recovery_authorization=gate.OLD_AUTH, **extra)
        roster[split] = dict(executor="support_v1" if split in gate.SPLITS_NEW else "memory02",
            data=data, aggregate=aggregate, logits=logits, scored=scored, normalized=normalized,
            predictions=predictions, evaluation=evaluation, replay=replay)
        previous[split] = aggregate
    record = {"schema": "partition-support-mixed-test-roster-v1", "support_authorization": support_authorization,
              "normalization_audit": NORMALIZATION_AUDIT, "splits": roster}
    check_roster(record, support_authorization)
    path = gate.canonical_path(p.ROOT/gate.RECOVERY/"tests_01.json", p.ROOT/gate.RECOVERY)
    write_json(path, record)
    return p.reference(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path)
    parser.add_argument("--sha256")
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--worker-fd", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_fd is not None:
        if args.request is not None or args.sha256 is not None or args.authorization is not None:
            parser.error("worker accepts only the inherited request pipe")
        with os.fdopen(args.worker_fd) as handle:
            record = json.load(handle)
        if set(record) != {"reference", "supervisor_pid"}:
            raise ValueError("support worker envelope differs")
        from src.atencion_armonica.learned_partition_supervisor import arm_parent_death
        arm_parent_death(record["supervisor_pid"])
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
            raise PermissionError("support workers must not expose CUDA")
        result = execute(record["reference"])
        print(json.dumps({"result": result, "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}))
    else:
        if args.sha256 is None or (args.request is None) == (args.authorization is None):
            parser.error("provide exactly one request or authorization, plus its immutable SHA256")
        path = args.request if args.request is not None else args.authorization
        gate.canonical_path(path, p.ROOT/gate.RECOVERY)
        ref = {"path": path.absolute().relative_to(p.ROOT).as_posix(), "sha256": args.sha256}
        print(json.dumps(supervised(ref) if args.request is not None else run(ref)))


if __name__ == "__main__":
    main()
