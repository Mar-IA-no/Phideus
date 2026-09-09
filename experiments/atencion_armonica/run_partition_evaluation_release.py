"""Finite evaluation-only memory recovery; upstream producers retain their own authority."""
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
from src.atencion_armonica import partition_evaluation_release as gate
from src.atencion_armonica.partition_test_recovery_supervisor import supervise_process
from src.atencion_armonica.structured_source_artifacts import write_json, mark_failure
from experiments.atencion_armonica import run_partition_support_recovery as sop
old_gate, old_operator = sop.old_gate, sop.old_operator
NORMALIZATION_AUDIT = sop.NORMALIZATION_AUDIT
ARGUMENTS = {"evaluation": {"split", "authorization", "data", "logits", "scored", "normalized", "predictions",
    "recovery_authorization", "support_recovery_authorization", "evaluation_release_authorization"}}


def validate_request(ref):
    request = p.read_reference(ref)
    root = p.ROOT/gate.RECOVERY
    if (set(request) != {"schema", "request_id", "operation", "output", "arguments", "execution_contract"}
            or request["schema"] != "partition-evaluation-release-request-v1"
            or not isinstance(request["request_id"], str) or not request["request_id"].strip()
            or request["operation"] not in ARGUMENTS or not isinstance(request["arguments"], dict)
            or set(request["arguments"]) != ARGUMENTS[request["operation"]]):
        raise ValueError("closed recovery request schema differs")
    arguments = request["arguments"]
    if arguments["split"] not in gate.SPLITS_NEW or arguments["authorization"] != gate.FROZEN["test_authorization"]:
        raise PermissionError("release evaluation only accepts the two pending support tests")
    if arguments["recovery_authorization"] != gate.old.OLD_AUTH or arguments["support_recovery_authorization"] != gate.SUPPORT_AUTH:
        raise PermissionError("upstream recovery authority differs")
    auth, _ = gate.verify_authorization(arguments["evaluation_release_authorization"])
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
    from src.atencion_armonica.partition_evaluation_release import evaluate_test
    fn = evaluate_test
    result = fn(output, **request["arguments"])
    if p.read_reference(ref) != request:
        raise ValueError("recovery request changed during execution")
    gate.verify_contract(request["execution_contract"])
    return result

def supervised(ref):
    request, output = validate_request(ref)
    memory = {line.split(":")[0]: int(line.split()[1])*1024
        for line in Path("/proc/meminfo").read_text().splitlines() if line.startswith("MemAvailable:")}
    if memory.get("MemAvailable", 0) < 8*1024**3:
        raise RuntimeError("evaluation RAM amendment requires at least 8 GiB available before launch")
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
    command = [sys.executable, str(p.ROOT/"experiments/atencion_armonica/run_partition_evaluation_release.py"), "--worker-fd"]
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
        gate.checked_bundle(outcome["result"], request["operation"], request["arguments"]["evaluation_release_authorization"])
        if request["operation"] == "evaluation":
            gate.checked_evaluation(outcome["result"], **request["arguments"])
    except BaseException as exc:
        outcome["error"] = repr(exc)
        if output.is_dir():
            mark_failure(output, exc)
        mark_failure(control, exc)
    receipt = {"schema": "partition-evaluation-release-terminal-v1", **outcome,
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
    auth, _ = gate.verify_authorization(arguments["evaluation_release_authorization"])
    request = {"schema": "partition-evaluation-release-request-v1",
        "request_id": "phideus-evaluation-release-v2-"+name+"-20260908-01", "operation": operation,
        "output": output.relative_to(p.ROOT).as_posix(), "arguments": arguments,
        "execution_contract": auth["contract"]}
    if saved.exists():
        value = old_operator.completed(saved, request_path, request, output)
        gate.checked_bundle(value["result"], operation, arguments["evaluation_release_authorization"])
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

def check_roster(record, evaluation_release_authorization):
    if (set(record) != {"schema", "evaluation_release_authorization", "normalization_audit", "splits"}
            or record["schema"] != "partition-evaluation-release-mixed-roster-v1"
            or record["evaluation_release_authorization"] != evaluation_release_authorization
            or record["normalization_audit"] != NORMALIZATION_AUDIT or set(record["splits"]) != set(gate.TESTS)):
        raise ValueError("evaluation release requires its exact four-test typed roster")
    auth, contract = gate.verify_authorization(evaluation_release_authorization)
    frozen = p.read_reference(gate.FROZEN["freeze"])
    previous = {"train": frozen["train_data"], "calibration": frozen["calibration_data"]}
    for split in gate.TESTS:
        case = record["splits"][split]
        new = split in gate.SPLITS_NEW
        executor = {"predictions": "support03", "evaluation": "release02"} if new else {
            "predictions": "memory02", "evaluation": "memory02"}
        if (set(case) != {"executor", "data", "aggregate", "logits", "scored", "normalized",
                         "predictions", "evaluation", "replay"} or case["executor"] != executor):
            raise ValueError("test input and evaluation executors were reassigned")
        sop.checked_aggregate(case["aggregate"], split, case["data"], previous, contract["base_common"])
        arguments = {k: case[k] for k in ("data", "logits", "scored", "normalized", "predictions")}
        arguments.update(authorization=gate.FROZEN["test_authorization"], recovery_authorization=gate.old.OLD_AUTH)
        consumer = gate if new else old_gate
        if new:
            arguments.update(support_recovery_authorization=gate.SUPPORT_AUTH,
                             evaluation_release_authorization=evaluation_release_authorization)
        sop.checked_predictions(case["predictions"], split, arguments, gate.SUPPORT_AUTH)
        contract_ref = auth["contract"] if new else gate.old.OLD_CONTRACT
        schema = "partition-evaluation-release" if new else "partition-test-memory-recovery"
        for key in ("evaluation", "replay"):
            envelope = case[key]
            if set(envelope) != {"result", "supervisor"}:
                raise ValueError("test completion envelope differs")
            terminal = p.read_reference(envelope["supervisor"])
            request = p.read_reference(terminal["request"])
            if (terminal["schema"] != schema+"-terminal-v1" or request["schema"] != schema+"-request-v1"
                    or terminal["status"] != "COMPLETE" or terminal["worker_terminal_confirmed"] is not True
                    or terminal["worker_exit_code"] != 0 or terminal["result"] != envelope["result"]
                    or terminal["operation"] != "evaluation" or request["operation"] != "evaluation"
                    or request["arguments"] != {"split": split, **arguments}
                    or terminal["execution_contract"] != contract_ref or request["execution_contract"] != contract_ref
                    or request["output"] != terminal["output"]
                    or p.verify_reference(envelope["result"]) != p.ROOT/terminal["output"]/"manifest.json"):
                raise ValueError("evaluation request/terminal/result identity differs")
        consumer.compare_evaluation_replay(case["evaluation"]["result"], case["replay"]["result"], split, **arguments)
        previous[split] = case["aggregate"]
    old_operator.verify_normalization_audit(NORMALIZATION_AUDIT, contract=gate.old.OLD_CONTRACT,
        target=record["splits"]["iid"]["normalized"])
    return record


def require_preserved(split, name, producer):
    """The only upstream production allowed in this continuation is deformed_family."""
    if split == "deformed_family":
        return
    if split not in ("iid", "ood_beta", "ood_polyphony"):
        raise PermissionError("unknown preservation scope")
    roots = {"canonical": (p.ROOT/gate.LOCAL/"operation_receipts_06", name+".json"),
        "memory": (p.ROOT/old_gate.RECOVERY/"receipts", name+"_01.json"),
        "support": (p.ROOT/gate.old.RECOVERY/"receipts", name+"_01.json")}
    if producer not in roots:
        raise PermissionError("unknown preserved producer")
    root, filename = roots[producer]
    receipt = gate.canonical_path(root/filename, root)
    if not receipt.is_file():
        raise PermissionError("completed upstream receipt missing; regeneration is forbidden")


def run(evaluation_release_authorization):
    gate.verify_authorization(evaluation_release_authorization)
    _, contract = old_gate.verify_authorization(gate.old.OLD_AUTH)
    frozen = p.read_reference(gate.FROZEN["freeze"])
    test_auth = gate.FROZEN["test_authorization"]
    previous = {"train": frozen["train_data"], "calibration": frozen["calibration_data"]}
    grant = p.reference(p.ROOT/gate.LOCAL/"gpu_grant.json")
    roster = {}
    for split in gate.TESTS:
        def upstream(name, operation, path, **arguments):
            require_preserved(split, name, "canonical")
            return old_operator.step(name, operation, p.ROOT/gate.TREE/path, arguments,
                recovery_auth=gate.old.OLD_AUTH, contract=contract, canonical=True)["result"]
        data = upstream(split+"_00_data", "prepare", f"{split}/shard_00/data", split=split,
            shard=0, authorization=test_auth, previous=previous.copy(), earlier_shards=[])
        aggregate = upstream(split+"_data", "aggregate_data", f"{split}/aggregate", split=split,
            authorization=test_auth, previous=previous.copy(), shards=[data])
        logits = upstream(split+"_00_logits", "forward", f"{split}/shard_00/logits", split=split,
            shard=0, authorization=test_auth, data=data, gpu_grant=grant)
        scored = upstream(split+"_00_scored", "score", f"{split}/shard_00/scored", split=split,
            shard=0, authorization=test_auth, data=data, logits=logits)
        def memory(name, operation, **arguments):
            require_preserved(split, split+"_"+name, "memory")
            return old_operator.step(split+"_"+name, operation,
                p.ROOT/old_gate.RECOVERY/"outputs"/split/(name+"_01"),
                {**arguments, "recovery_authorization": gate.old.OLD_AUTH},
                recovery_auth=gate.old.OLD_AUTH, contract=contract)
        normalized = memory("normalized", "normalized", split=split, shard=0, authorization=test_auth,
            data=data, logits=logits, scored=scored, normalizers=frozen["normalizers"], train=frozen["train"])["result"]
        if split == "iid":
            old_operator.verify_normalization_audit(NORMALIZATION_AUDIT, contract=gate.old.OLD_CONTRACT, target=normalized)
        inputs = dict(split=split, authorization=test_auth, data=data, logits=logits, scored=scored, normalized=normalized)
        if split in gate.SPLITS_NEW:
            old_arguments = {**inputs, "gpu_grant": None, "recovery_authorization": gate.old.OLD_AUTH,
                             "support_recovery_authorization": gate.SUPPORT_AUTH}
            require_preserved(split, split+"_predictions", "support")
            predictions = sop.step(split+"_predictions", "inference",
                p.ROOT/gate.old.RECOVERY/"outputs"/split/"predictions_01", old_arguments)["result"]
            arguments = {**inputs, "predictions": predictions, "recovery_authorization": gate.old.OLD_AUTH,
                "support_recovery_authorization": gate.SUPPORT_AUTH,
                "evaluation_release_authorization": evaluation_release_authorization}
            evaluation = step(split+"_evaluation", "evaluation",
                p.ROOT/gate.RECOVERY/"outputs"/split/"evaluation_01", arguments)
            replay = step(split+"_replay", "evaluation", p.ROOT/gate.RECOVERY/"outputs"/split/"replay_01", arguments)
            consumer, executor = gate, {"predictions": "support03", "evaluation": "release02"}
        else:
            for name in ("predictions", "evaluation", "replay"):
                if not (p.ROOT/old_gate.RECOVERY/"receipts"/(split+"_"+name+"_01.json")).is_file():
                    raise PermissionError("completed IID/beta must be preserved, not regenerated")
            predictions = memory("predictions", "inference", **inputs, gpu_grant=None)["result"]
            evaluation = memory("evaluation", "evaluation", **inputs, predictions=predictions)
            replay = memory("replay", "evaluation", **inputs, predictions=predictions)
            arguments = {**inputs, "predictions": predictions, "recovery_authorization": gate.old.OLD_AUTH}
            consumer, executor = old_gate, {"predictions": "memory02", "evaluation": "memory02"}
        consumer.compare_evaluation_replay(evaluation["result"], replay["result"], **arguments)
        roster[split] = dict(executor=executor, data=data, aggregate=aggregate, logits=logits, scored=scored,
            normalized=normalized, predictions=predictions, evaluation=evaluation, replay=replay)
        previous[split] = aggregate
    record = {"schema": "partition-evaluation-release-mixed-roster-v1",
        "evaluation_release_authorization": evaluation_release_authorization,
        "normalization_audit": NORMALIZATION_AUDIT, "splits": roster}
    check_roster(record, evaluation_release_authorization)
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
