"""Resume the fixed test roster through audited versioned consumers, no retry.

First invocation stops after IID normalization. Completing the remaining tests
requires a separate integrated audit of that measured recovery stage.
"""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.atencion_armonica import learned_partition_provenance as p
from src.atencion_armonica import partition_test_recovery_gate as gate
from src.atencion_armonica.learned_partition_supervisor import supervised as canonical_supervised
from src.atencion_armonica.partition_test_recovery_supervisor import supervised
from src.atencion_armonica.structured_source_artifacts import write_json


def completed(saved, request_path, expected_request, output):
    envelope = json.loads(saved.read_bytes())
    if set(envelope) != {"result", "supervisor"}:
        raise ValueError("saved operator receipt schema differs")
    request_ref = p.reference(request_path)
    terminal = p.read_reference(envelope["supervisor"])
    if (p.read_reference(request_ref) != expected_request or terminal["request"] != request_ref
            or terminal["status"] != "COMPLETE" or terminal["result"] != envelope["result"]
            or terminal["worker_terminal_confirmed"] is not True or terminal["worker_exit_code"] != 0
            or p.verify_reference(envelope["result"]) != output/"manifest.json"):
        raise ValueError("saved stage is not the exact completed request")
    return envelope


def step(name, operation, output, arguments, *, recovery_auth, contract, canonical=False):
    tree = ROOT/gate.TREE
    if canonical:
        request_path = tree/"requests"/f"{name}_06.json"
        saved = ROOT/gate.LOCAL/"operation_receipts_06"/f"{name}.json"
        request = {"request_id": f"phideus-learned-{name}-20260908-06", "operation": operation,
                   "output": output.relative_to(ROOT).as_posix(), "arguments": arguments,
                   "common": contract["base_common"]}
    else:
        request_path = ROOT/gate.RECOVERY/"requests"/f"{name}_01.json"
        saved = ROOT/gate.RECOVERY/"receipts"/f"{name}_01.json"
        auth, _ = gate.verify_authorization(recovery_auth)
        request = {"schema": "partition-test-memory-recovery-request-v1",
                   "request_id": f"phideus-test-memory-{name}-20260908-01", "operation": operation,
                   "output": output.relative_to(ROOT).as_posix(), "arguments": arguments,
                   "execution_contract": auth["contract"]}
        gate.canonical_path(request_path, ROOT/gate.RECOVERY/"requests")
        gate.canonical_path(saved, ROOT/gate.RECOVERY/"receipts")
        gate.canonical_path(output, ROOT/gate.RECOVERY/"outputs"/arguments["split"])
    if saved.exists():
        value = completed(saved, request_path, request, output)
        if not canonical and operation == "evaluation":
            gate.checked_evaluation(value["result"], **arguments)
        print(json.dumps({"event": "REUSED", "name": name}), flush=True)
        return value
    if request_path.exists():
        raise RuntimeError("request exists without completed receipt; stop for explicit recovery")
    for parent in (request_path.parent, saved.parent, output.parent):
        parent.mkdir(parents=True, exist_ok=True)
    write_json(request_path, request)
    print(json.dumps({"event": "STARTED", "name": name, "operator_pid": os.getpid(),
                      "request": p.reference(request_path)}), flush=True)
    value = (canonical_supervised if canonical else supervised)(p.reference(request_path))
    if not canonical and operation == "evaluation":
        gate.checked_evaluation(value["result"], **arguments)
    write_json(saved, value)
    print(json.dumps({"event": "COMPLETE", "name": name, **value}), flush=True)
    return value


def verify_normalization_audit(ref, *, contract, target):
    record = p.read_reference(ref)
    if (set(record) != {"status", "scope", "contract", "target", "reports"}
            or record["status"] != "PASS" or record["scope"] != "NORMALIZED_MEMORY_RECOVERY"
            or record["contract"] != contract or record["target"] != target
            or not isinstance(record["reports"], list) or not record["reports"]):
        raise PermissionError("tests require an integrated audit of the completed IID normalization")
    if len({r["path"] for r in record["reports"]}) != len(record["reports"]):
        raise ValueError("duplicated stage audit report")
    for report in record["reports"]:
        if not p.verify_reference(report).read_text().strip():
            raise ValueError("empty normalization recovery audit")


def run(phase, recovery_auth, normalization_audit):
    auth, contract = gate.verify_authorization(recovery_auth)
    frozen = p.read_reference(gate.FROZEN["freeze"])
    test_auth = gate.FROZEN["test_authorization"]
    previous = {"train": frozen["train_data"], "calibration": frozen["calibration_data"]}
    grant = p.reference(ROOT/gate.LOCAL/"gpu_grant.json")
    tree = ROOT/gate.TREE
    outputs = ROOT/gate.RECOVERY/"outputs"
    roster = {}
    for split in gate.TESTS:
        def old(name, operation, path, **arguments):
            return step(name, operation, tree/path, arguments,
                        recovery_auth=recovery_auth, contract=contract, canonical=True)["result"]
        data = old(split+"_00_data", "prepare", f"{split}/shard_00/data", split=split,
                   shard=0, authorization=test_auth, previous=previous.copy(), earlier_shards=[])
        aggregate = old(split+"_data", "aggregate_data", f"{split}/aggregate", split=split,
                        authorization=test_auth, previous=previous.copy(), shards=[data])
        logits = old(split+"_00_logits", "forward", f"{split}/shard_00/logits", split=split,
                     shard=0, authorization=test_auth, data=data, gpu_grant=grant)
        scored = old(split+"_00_scored", "score", f"{split}/shard_00/scored", split=split,
                     shard=0, authorization=test_auth, data=data, logits=logits)
        def new(name, operation, **arguments):
            return step(split+"_"+name, operation, outputs/split/(name+"_01"),
                        {**arguments, "recovery_authorization": recovery_auth},
                        recovery_auth=recovery_auth, contract=contract)
        normalized = new("normalized", "normalized", split=split, shard=0, authorization=test_auth,
            data=data, logits=logits, scored=scored, normalizers=frozen["normalizers"], train=frozen["train"])["result"]
        if split == "iid":
            if phase == "normalize-iid":
                return normalized
            if normalization_audit is None:
                raise PermissionError("independent normalized-stage acceptance is missing")
            verify_normalization_audit(normalization_audit, contract=auth["contract"], target=normalized)
        inputs = dict(split=split, authorization=test_auth, data=data, logits=logits,
                      scored=scored, normalized=normalized)
        predictions = new("predictions", "inference", **inputs, gpu_grant=None)["result"]
        evaluation = new("evaluation", "evaluation", **inputs, predictions=predictions)
        replay = new("replay", "evaluation", **inputs, predictions=predictions)
        gate.compare_evaluation_replay(evaluation["result"], replay["result"], **inputs,
            predictions=predictions, recovery_authorization=recovery_auth)
        roster[split] = dict(data=data, aggregate=aggregate, logits=logits, scored=scored, normalized=normalized,
                            predictions=predictions, evaluation=evaluation, replay=replay)
        previous[split] = aggregate
    path = ROOT/gate.RECOVERY/"tests_01.json"
    write_json(path, {"recovery_authorization": recovery_auth, "normalization_audit": normalization_audit,
                      "splits": roster})
    return p.reference(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("normalize-iid", "tests"))
    parser.add_argument("--authorization", required=True, type=Path)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--normalization-audit", type=Path)
    parser.add_argument("--normalization-audit-sha256")
    args = parser.parse_args()
    if args.phase == "tests" and (args.normalization_audit is None or args.normalization_audit_sha256 is None):
        parser.error("tests require the measured normalization audit path and SHA")
    if args.phase == "normalize-iid" and (args.normalization_audit is not None or args.normalization_audit_sha256 is not None):
        parser.error("the first normalization does not accept a future stage audit")
    authorization = {"path": args.authorization.resolve().relative_to(ROOT).as_posix(), "sha256": args.sha256}
    audit = None if args.normalization_audit is None else {
        "path": args.normalization_audit.resolve().relative_to(ROOT).as_posix(), "sha256": args.normalization_audit_sha256}
    print(json.dumps(run(args.phase, authorization, audit)))


if __name__ == "__main__":
    main()
