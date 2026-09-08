"""Closed post-selection recovery contract; never reassign a producer's identity.

The original learned-reader code and its source inventory remain untouched.
This executor accepts precisely one scientific base and three new test roles.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from . import learned_partition_gate as base_gate
from . import learned_partition_provenance as p
from .learned_partition_core import ARMS
from .learned_partition_metrics import SEEDS, TESTS
from .learned_partition_validation import boundary, fresh_pass, memoized
from .partial_compatibility_cache import encoded, sha_file
from .structured_source_artifacts import safe_member, verify_bundle, write_json

BASE_SHA = "1219fef94b8f770e7a4b81713ff0ef3e9d67a65f4d276487bc10d0dc6eaa67dd"
TREE = "data/atencion_armonica/learned_partition_reader_v1"
RECOVERY = TREE + "/test_memory_recovery_v1"
LOCAL = ".agent-work/phideus-learned-reader-20260908"
LIBRARY = "Biblioteca/Geometria_Proporcional_Ground_Truth"
PLAN = {"path": LOCAL + "/PLAN_VERSIONED_TEST_MEMORY_RECOVERY_V2.md",
        "sha256": "d05dcb94e69ffcf1f2ed5c65ccc2f23e3fcf284f22d30a1698efcfbda4a092ff"}
DIAGNOSTIC = {"path": LIBRARY + "/experiment_notes/normalization_memory_evidence.json",
              "sha256": "dc356bda6f47ee65d4f9c49b3aa20db8658919fddca081df9ca62f9de5450a1a"}
FROZEN = {
    "freeze": {"path": TREE + "/selection/freeze_06.json", "sha256": "08bef072cfce37b3cda30be7e0278d5452f460cd194a265e783144cdcf9644cc"},
    "test_authorization": {"path": TREE + "/authorization/test_06.json", "sha256": "d48e5f0b758965dd673f2d152501de2f885fb9be8b5f4087ae0dd9c76900da36"},
    "selection_audit": {"path": LOCAL + "/selection_audit_06.json", "sha256": "83141b2b9c2289a1ba5a9df3182df599ce57b1490a06672a04276cc0bc9f4a69"},
    "selection_report": {"path": LIBRARY + "/agent_reports/671_learned_partition_selection06_audit.md", "sha256": "2f61dd3d846f0ed013590de6b601c9462e98f1d97ca9552aaed6f1550473982a"},
    "plan_audit": {"path": LIBRARY + "/agent_reports/673_versioned_test_memory_recovery_plan_audit.md", "sha256": "5630331eae6a02efe481bee80fe318464df0487fde2de9793f2c26ffdb83069e"},
}
FAILED_REQUEST = {"path": TREE + "/requests/iid_00_normalized_06.json", "sha256": "dc0791fb2bea40f6cde9a867b126e5adbca97b46e7e6e4207946b9fe2ab6d092"}
FAILED_TERMINAL = {"path": TREE + "/supervision/supervisor-vi6ekkdu/terminal.json", "sha256": "95bd60679810638bbbb941334d06c05a1091becc28a2db650cc2dd95dc318fbc"}
FAILED_MARKER = {"path": TREE + "/iid/shard_00/normalized/FAILURE.json", "sha256": "41186a605287dc77eb385884ee1f2011bc694bd3e75176c616c09cea964b21b6"}
SOURCES = (
    "src/atencion_armonica/partition_test_recovery_gate.py",
    "src/atencion_armonica/partition_test_recovery.py",
    "src/atencion_armonica/partition_test_recovery_supervisor.py",
    "experiments/atencion_armonica/run_partition_test_recovery.py",
    "experiments/atencion_armonica/operate_partition_test_recovery.py",
    "experiments/atencion_armonica/test_partition_test_recovery.py",
)
LIMITS = {"normalized": [1200, 2*1024**3], "inference": [1200, 4*1024**3],
          "evaluation": [1200, 2*1024**3]}
ROLES = {"normalized": "learned_normalized_memory_recovery_v1",
         "inference": "learned_predictions_memory_recovery_v1",
         "evaluation": "learned_evaluation_memory_recovery_v1"}
EVALUATION_FILES = {"bootstrap.npz", "metrics.json", "candidate_evidence.json", "summary.json",
    "intervention_metrics.json", "intervention_summary.json", "support.json", "support_summary.json"}


def canonical_path(path, anchor):
    """Confine operational paths before any mkdir; reject every symlink ancestor."""
    path, anchor = Path(path), Path(anchor)
    if not path.is_absolute():
        path = p.ROOT/path
    if not anchor.is_absolute():
        anchor = p.ROOT/anchor
    relative = path.relative_to(p.ROOT)
    if ".." in relative.parts or not path.is_relative_to(anchor) or path == anchor:
        raise ValueError("operational path escapes its canonical subtree")
    current = p.ROOT
    for part in relative.parts:
        current = current/part
        if current.is_symlink():
            raise ValueError("operational path contains a symlink")
    if not path.resolve().is_relative_to(anchor.resolve()):
        raise ValueError("resolved operational path escapes its canonical subtree")
    return path


def hash_only(ref):
    """Evidence bytes, NOT a COMPLETE-stage authority. Used for known failures."""
    if not isinstance(ref, dict) or set(ref) != {"path", "sha256"}:
        raise ValueError("evidence requires an exact path/hash reference")
    path = safe_member(p.ROOT, ref["path"])
    if path.is_symlink() or sha_file(path) != ref["sha256"]:
        raise ValueError("evidence bytes changed")
    return path


def failed_evidence():
    record = p.read_reference(DIAGNOSTIC)
    if (record.get("schema") != "normalization-memory-diagnostic-evidence-v1"
            or record.get("status") != "DIAGNOSTIC_ONLY_NOT_CAMPAIGN_COMPLETION"
            or record.get("base_common_sha256") != BASE_SHA
            or record.get("failed_terminal") != FAILED_TERMINAL):
        raise ValueError("diagnostic identity or scope differs")
    terminal = json.loads(hash_only(FAILED_TERMINAL).read_bytes())
    if (terminal["status"] != "FAILED" or terminal["request"] != FAILED_REQUEST
            or terminal["worker_terminal_confirmed"] is not True
            or terminal["worker_exit_code"] != -15 or terminal["result"] is not None):
        raise ValueError("failed attempt is not the declared confirmed terminal")
    hash_only(FAILED_MARKER)
    p.verify_reference(FAILED_REQUEST)
    expected = {TREE + f"/iid/shard_00/normalized/seed_{s}/{a}.npz" for s in SEEDS for a in ARMS}
    candidates = record["failed_candidate_roster"]
    if len(candidates) != 12 or {r["path"] for r in candidates} != expected or len(record["sources"]) != 7:
        raise ValueError("diagnostic source or failed-candidate roster differs")
    root = p.ROOT/TREE/"iid/shard_00/normalized"
    if {f.relative_to(p.ROOT).as_posix() for f in root.rglob("*") if f.is_file()} != expected | {FAILED_MARKER["path"]}:
        raise ValueError("failed output changed or was retroactively sealed")
    for ref in [*record["sources"], *candidates]:
        hash_only(ref)
    return candidates


def executor_sources():
    return {name: sha_file(p.ROOT/name) for name in SOURCES}


def contract_candidate():
    return {"schema": "partition-test-memory-recovery-contract-v1", "base_common": base_gate.common_binding(),
            "executor_sources": executor_sources(), "plan": PLAN, "frozen": FROZEN,
            "diagnostic": DIAGNOSTIC, "failed_request": FAILED_REQUEST, "failed_terminal": FAILED_TERMINAL,
            "failed_marker": FAILED_MARKER, "limits": LIMITS, "roles": ROLES,
            "test_splits": list(TESTS), "device": "cpu"}


@boundary
@memoized
def verify_contract(ref):
    record = p.read_reference(ref)
    expected = contract_candidate()
    if (record != expected or hashlib.sha256(encoded(record["base_common"])).hexdigest() != BASE_SHA):
        raise PermissionError("executor, base, contract inventory or recovery scope changed")
    p.verify_reference(PLAN)
    for value in FROZEN.values():
        p.verify_reference(value)
    failed_evidence()
    auth = p.read_reference(FROZEN["test_authorization"])
    if auth != {"status": "TEST_READY", "common": record["base_common"],
                "freeze": FROZEN["freeze"], "freeze_audit": FROZEN["selection_audit"]}:
        raise ValueError("original test authority changed")
    base_gate.verify_audit(FROZEN["selection_audit"], record["base_common"],
                           scope="SELECTION_FREEZE", target=FROZEN["freeze"])
    if FROZEN["selection_report"] not in p.read_reference(FROZEN["selection_audit"])["reports"]:
        raise ValueError("selection authority omitted its independently pinned report")
    return record


def create_contract(output):
    write_json(output, contract_candidate())
    ref = p.reference(output)
    verify_contract(ref)
    return ref


def verify_implementation_audit(ref, contract):
    audit = p.read_reference(ref)
    if (set(audit) != {"status", "scope", "contract", "reports", "resolved_plan_findings"}
            or audit["status"] != "PASS" or audit["scope"] != "TEST_MEMORY_RECOVERY_IMPLEMENTATION"
            or audit["contract"] != contract or audit["resolved_plan_findings"] != ["R673-F1", "R673-F2"]
            or not isinstance(audit["reports"], list) or not audit["reports"]):
        raise PermissionError("missing complete independent recovery implementation audit")
    if len({r["path"] for r in audit["reports"]}) != len(audit["reports"]):
        raise ValueError("duplicate implementation report")
    for report in audit["reports"]:
        if not p.verify_reference(report).read_text().strip():
            raise ValueError("empty recovery audit report")


@boundary
@memoized
def verify_authorization(ref):
    auth = p.read_reference(ref)
    if (set(auth) != {"schema", "status", "contract", "implementation_audit", "test_authorization"}
            or auth["schema"] != "partition-test-memory-recovery-authorization-v1"
            or auth["status"] != "READY" or auth["test_authorization"] != FROZEN["test_authorization"]):
        raise PermissionError("new stages require their own downstream authorization")
    contract = verify_contract(auth["contract"])
    verify_implementation_audit(auth["implementation_audit"], auth["contract"])
    return auth, contract


def create_authorization(output, *, contract, implementation_audit):
    verify_contract(contract)
    verify_implementation_audit(implementation_audit, contract)
    write_json(output, {"schema": "partition-test-memory-recovery-authorization-v1", "status": "READY",
        "contract": contract, "implementation_audit": implementation_audit,
        "test_authorization": FROZEN["test_authorization"]})
    ref = p.reference(output)
    verify_authorization(ref)
    return ref


def execution_binding(recovery_authorization):
    auth, contract = verify_authorization(recovery_authorization)
    return {"base_common": contract["base_common"], "execution_contract": auth["contract"],
            "recovery_authorization": recovery_authorization}


def checked_bundle(ref, operation, recovery_authorization):
    expected = execution_binding(recovery_authorization)
    path = p.verify_reference(ref)
    if path.name != "manifest.json":
        raise ValueError("expected versioned bundle manifest")
    manifest = verify_bundle(path.parent, ref["sha256"], role=ROLES[operation])
    if any(manifest["binding"].get(k) != v for k, v in expected.items()):
        raise ValueError("versioned producer/executor contract differs")
    return path.parent, manifest


def assert_fresh(recovery_authorization, expected):
    if fresh_pass(execution_binding)(recovery_authorization) != expected:
        raise ValueError("recovery identity changed during stage")


def checked_evaluation(ref, split, *, authorization, data, logits, scored, normalized,
                       predictions, recovery_authorization):
    if split not in TESTS or authorization != FROZEN["test_authorization"]:
        raise PermissionError("evaluation requires its declared frozen test")
    expected = {**execution_binding(recovery_authorization), "split": split,
        "authorization": authorization, "data": data, "logits": logits, "scored": scored,
        "normalized": normalized, "predictions": predictions, "freeze": FROZEN["freeze"]}
    root, manifest = checked_bundle(ref, "evaluation", recovery_authorization)
    if manifest["binding"] != expected or set(manifest["artifacts_sha256"]) != EVALUATION_FILES:
        raise ValueError("evaluation binding or eight-payload inventory differs")
    return root, manifest


def compare_evaluation_replay(evaluation, replay, split, **arguments):
    _, original = checked_evaluation(evaluation, split, **arguments)
    _, repeated = checked_evaluation(replay, split, **arguments)
    if original["artifacts_sha256"] != repeated["artifacts_sha256"]:
        raise ValueError("evaluation/replay payloads are not byte-exact")
