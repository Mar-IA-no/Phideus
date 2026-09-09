"""Evaluation-only recovery: release serialized objects before support reconstruction."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import time
import numpy as np

from . import partition_support_recovery as old
from . import learned_partition_provenance as p
from . import learned_partition_runner as runner
from .learned_partition_core import ARMS, READER_SEEDS
from .learned_partition_data import load_supervision
from .learned_partition_metrics import (SEEDS, TESTS, SPLITS, METRICS, REFERENCES,
    evaluate_preserved_predictions, summarize_test, bootstrap_indices)
from .learned_partition_test import (inference_roster, prefix, intervention_metrics_for_scene,
    summarize_interventions, summarize_support)
from .partition_support_guard import replay_support
from .partition_support_recovery import preserved_test
from .learned_partition_validation import boundary, memoized, fresh_pass
from .partial_compatibility_cache import encoded, sha_file
from .structured_source_artifacts import write_json, write_npz, seal_bundle, mark_failure, verify_bundle

TREE, LOCAL, FROZEN = old.TREE, old.LOCAL, old.FROZEN
RECOVERY = TREE + "/evaluation_release_recovery_v1"
SPLITS_NEW = old.SPLITS_NEW
LIMITS = {"evaluation": old.LIMITS["evaluation"]}
ROLES = {"evaluation": "learned_evaluation_release_v1"}
EVALUATION_FILES = old.EVALUATION_FILES
canonical_path, evidence_only = old.canonical_path, old.evidence_only
SUPPORT_AUTH = {"path": old.RECOVERY + "/authorization/recovery_03.json",
    "sha256": "4fd9016ec43ca1507421d0efdca3332375110da8d5990d208d2f754dcdd01157"}
SUPPORT_CONTRACT = {"path": old.RECOVERY + "/contracts/contract_03.json",
    "sha256": "db79401ab560b443415899a0adaae1809c31eb5b280c60e6620131c2ca994aff"}
PLAN = {"path": LOCAL + "/PLAN_EVALUATION_RELEASE_RECOVERY.md",
    "sha256": "7c9e6cb08531bbdee7a9556a9091fd047576f92151367baaf57bfcaa87c20ca4"}
PLAN_AUDIT = {"path": LOCAL + "/evaluation_release_plan_audit.json",
    "sha256": "814e060a9ddeb37ce27651a89d878cbd5f3e9ca9896662d90f91a5559e5e983c"}
PLAN_REPORT = {"path": old.gate02.LIBRARY + "/agent_reports/684_partition_evaluation_release_plan_audit.md",
    "sha256": "73ed826a5a2ad8d62237ab4c2c7410f61da2d20d6a7f335fb6ecd37d60889728"}
DIAGNOSTIC_REPORT = {"path": old.gate02.LIBRARY + "/agent_reports/683_partition_evaluation_support_memory_diagnosis.md",
    "sha256": "2972dc4c2fcc10611f1a48949aa8cca61678de31f8ac507688c7a22bde87d425"}
DIAGNOSTIC = {"path": old.gate02.LIBRARY + "/experiment_notes/partition_evaluation_support_memory_diagnostic.json",
    "sha256": "083d7d5262095a3f65a16184d3b5449cd0a7b31a45db26ac59624456a519e124"}
FAILED_ROOT = old.RECOVERY + "/outputs/ood_polyphony/evaluation_01"
FAILED_REQUEST = {"path": old.RECOVERY + "/requests/ood_polyphony_evaluation_01.json",
    "sha256": "e8c1f39664f7799f4f9d569d9bc385839fcf1c7563416b158bf789dfb1f94dbd"}
FAILED_TERMINAL = {"path": old.RECOVERY + "/supervision/supervisor-fdjeu3ar/terminal.json",
    "sha256": "33dd6f6d8d17a436f4ecc36e630c4dbd2a6920f9ba929027655121f8c682f6d9"}
FAILED_FILES = {"bootstrap.npz", "metrics.json", "candidate_evidence.json", "summary.json",
                "intervention_metrics.json", "intervention_summary.json"}
SOURCES = ("src/atencion_armonica/partition_evaluation_release.py",
    "experiments/atencion_armonica/run_partition_evaluation_release.py",
    "experiments/atencion_armonica/test_partition_evaluation_release.py")


def failed_evidence():
    record = json.loads(evidence_only(DIAGNOSTIC).read_bytes())
    if (record.get("schema") != "partition-evaluation-support-memory-diagnostic-v1"
            or record.get("status") != "DIAGNOSTIC_ONLY_NOT_CAMPAIGN_COMPLETION"
            or record.get("terminal") != FAILED_TERMINAL or record.get("request") != FAILED_REQUEST
            or record.get("contract") != SUPPORT_CONTRACT):
        raise ValueError("evaluation memory diagnostic identity differs")
    terminal = json.loads(evidence_only(FAILED_TERMINAL).read_bytes())
    started = json.loads(evidence_only(record["started"]).read_bytes())
    request = json.loads(evidence_only(FAILED_REQUEST).read_bytes())
    if (terminal["status"] != "FAILED" or terminal["worker_terminal_confirmed"] is not True
            or terminal["worker_exit_code"] != -15 or terminal["result"] is not None
            or terminal["request"] != FAILED_REQUEST or terminal["output"] != FAILED_ROOT
            or terminal["operation"] != "evaluation" or terminal["execution_contract"] != SUPPORT_CONTRACT
            or terminal["observed_peak_gpu_bytes"] != 0
            or started["rss_limit_bytes"] != LIMITS["evaluation"][1]
            or started["wall_limit_seconds"] != LIMITS["evaluation"][0]
            or terminal["seconds"] >= started["wall_limit_seconds"]
            or terminal["observed_peak_rss_bytes"] <= started["rss_limit_bytes"]
            or request["operation"] != "evaluation" or request["output"] != FAILED_ROOT
            or request["execution_contract"] != SUPPORT_CONTRACT
            or request["arguments"]["support_recovery_authorization"] != SUPPORT_AUTH
            or request["arguments"]["recovery_authorization"] != old.OLD_AUTH):
        raise ValueError("not the declared confirmed evaluation RSS failure")
    inventory = [{k: v[k] for k in ("path", "sha256")} for v in record["partial_inventory"]]
    expected = {FAILED_ROOT + "/" + name for name in FAILED_FILES | {"FAILURE.json"}}
    root = canonical_path(FAILED_ROOT, p.ROOT/old.RECOVERY/"outputs"/"ood_polyphony")
    paths = list(root.rglob("*"))
    if (len(inventory) != 7 or {r["path"] for r in inventory} != expected
            or any(path.is_symlink() for path in paths)
            or {path.relative_to(p.ROOT).as_posix() for path in paths if path.is_file()} != expected):
        raise ValueError("evaluation failure inventory changed or was retroactively completed")
    for ref in inventory:
        evidence_only(ref)
    marker = json.loads((root/"FAILURE.json").read_bytes())
    if marker != {"status": "INCOMPLETE",
            "error": "RuntimeError(\"TimeoutError('recovery worker exceeded its original CPU envelope')\")"}:
        raise ValueError("evaluation failure marker differs")
    return [r for r in inventory if Path(r["path"]).name in FAILED_FILES]


def verify_plan_audit():
    p.verify_reference(PLAN)
    if p.read_reference(PLAN_AUDIT) != {"schema": "partition-evaluation-release-plan-audit-v1",
            "status": "PASS", "scope": "EVALUATION_RELEASE_PLAN", "target": PLAN, "reports": [PLAN_REPORT]}:
        raise PermissionError("exact independent evaluation-release plan acceptance missing")
    for report in (PLAN_REPORT, DIAGNOSTIC_REPORT):
        if not p.verify_reference(report).read_text().strip():
            raise ValueError("empty release report")


def contract_candidate():
    return {"schema": "partition-evaluation-release-contract-v1",
        "base_common": old.gate02.base_gate.common_binding(),
        "support_contract": SUPPORT_CONTRACT, "support_authorization": SUPPORT_AUTH,
        "executor_sources": {name: sha_file(p.ROOT/name) for name in SOURCES},
        "plan": PLAN, "plan_audit": PLAN_AUDIT, "diagnostic": DIAGNOSTIC,
        "diagnostic_report": DIAGNOSTIC_REPORT, "failed_request": FAILED_REQUEST,
        "failed_terminal": FAILED_TERMINAL, "freeze": FROZEN["freeze"],
        "test_authorization": FROZEN["test_authorization"], "roles": ROLES,
        "test_splits": list(SPLITS_NEW), "limits": LIMITS, "device": "cpu"}


@boundary
@memoized
def verify_contract(ref):
    record = p.read_reference(ref)
    if record != contract_candidate() or hashlib.sha256(encoded(record["base_common"])).hexdigest() != old.gate02.BASE_SHA:
        raise PermissionError("evaluation release scope, sources or base changed")
    auth, _ = old.verify_authorization(SUPPORT_AUTH)
    if auth["contract"] != SUPPORT_CONTRACT:
        raise PermissionError("support producer identity changed")
    verify_plan_audit()
    failed_evidence()
    return record


def create_contract(output):
    output = canonical_path(output, p.ROOT/RECOVERY/"contracts")
    verify_plan_audit()
    failed_evidence()
    old.verify_authorization(SUPPORT_AUTH)
    write_json(output, contract_candidate())
    ref = p.reference(output)
    verify_contract(ref)
    return ref


def verify_implementation_audit(ref, contract):
    record = p.read_reference(ref)
    if (set(record) != {"status", "scope", "contract", "reports"} or record["status"] != "PASS"
            or record["scope"] != "EVALUATION_RELEASE_IMPLEMENTATION" or record["contract"] != contract
            or not isinstance(record["reports"], list) or not record["reports"]
            or len({r["path"] for r in record["reports"]}) != len(record["reports"])):
        raise PermissionError("independent evaluation implementation acceptance missing")
    for report in record["reports"]:
        if not p.verify_reference(report).read_text().strip():
            raise ValueError("empty implementation report")


@boundary
@memoized
def verify_authorization(ref):
    auth = p.read_reference(ref)
    if (set(auth) != {"schema", "status", "contract", "implementation_audit", "support_authorization"}
            or auth["schema"] != "partition-evaluation-release-authorization-v1"
            or auth["status"] != "READY" or auth["support_authorization"] != SUPPORT_AUTH):
        raise PermissionError("release evaluation requires its own authority")
    contract = verify_contract(auth["contract"])
    verify_implementation_audit(auth["implementation_audit"], auth["contract"])
    return auth, contract


def create_authorization(output, *, contract, implementation_audit):
    output = canonical_path(output, p.ROOT/RECOVERY/"authorization")
    verify_contract(contract)
    verify_implementation_audit(implementation_audit, contract)
    write_json(output, {"schema": "partition-evaluation-release-authorization-v1", "status": "READY",
        "contract": contract, "implementation_audit": implementation_audit, "support_authorization": SUPPORT_AUTH})
    ref = p.reference(output)
    verify_authorization(ref)
    return ref


def execution_binding(evaluation_release_authorization):
    auth, _ = verify_authorization(evaluation_release_authorization)
    return {**old.execution_binding(SUPPORT_AUTH), "evaluation_release_contract": auth["contract"],
            "evaluation_release_authorization": evaluation_release_authorization}


def release_entry(recovery_authorization, authorization, support_recovery_authorization, evaluation_release_authorization):
    if (recovery_authorization != old.OLD_AUTH or authorization != FROZEN["test_authorization"]
            or support_recovery_authorization != SUPPORT_AUTH):
        raise PermissionError("release evaluation requires exact upstream 02/03 identities")
    return execution_binding(evaluation_release_authorization)


def checked_bundle(ref, operation, evaluation_release_authorization):
    expected = execution_binding(evaluation_release_authorization)
    if operation != "evaluation":
        raise PermissionError("release executor has no inference authority")
    path = p.verify_reference(ref)
    if path.name != "manifest.json":
        raise ValueError("expected release evaluation manifest")
    manifest = verify_bundle(path.parent, ref["sha256"], role=ROLES[operation])
    if any(manifest["binding"].get(k) != v for k, v in expected.items()):
        raise ValueError("release execution identity differs")
    return path.parent, manifest


def checked_evaluation(ref, split, *, authorization, data, logits, scored, normalized, predictions,
                       recovery_authorization, support_recovery_authorization, evaluation_release_authorization):
    execution = release_entry(recovery_authorization, authorization, support_recovery_authorization, evaluation_release_authorization)
    if split not in SPLITS_NEW:
        raise PermissionError("release evaluation cannot reinterpret completed IID/beta")
    expected = {**execution, "split": split, "authorization": authorization, "data": data, "logits": logits,
        "scored": scored, "normalized": normalized, "predictions": predictions, "freeze": FROZEN["freeze"]}
    root, manifest = checked_bundle(ref, "evaluation", evaluation_release_authorization)
    if manifest["binding"] != expected or set(manifest["artifacts_sha256"]) != EVALUATION_FILES:
        raise ValueError("evaluation binding or exact eight-payload inventory differs")
    return root, manifest


def assert_fresh(evaluation_release_authorization, expected):
    if fresh_pass(execution_binding)(evaluation_release_authorization) != expected:
        raise ValueError("evaluation release identity changed during stage")


def compare_evaluation_replay(evaluation, replay, split, **arguments):
    _, a = checked_evaluation(evaluation, split, **arguments)
    _, b = checked_evaluation(replay, split, **arguments)
    if a["artifacts_sha256"] != b["artifacts_sha256"]:
        raise ValueError("evaluation/replay payloads are not byte-exact")


def compare_failed(output, split, arguments):
    if split != "ood_polyphony":
        return
    refs = failed_evidence()
    if json.loads(evidence_only(FAILED_REQUEST).read_bytes())["arguments"] != arguments:
        raise ValueError("failed-control comparison requires the exact original inputs/predictions")
    for ref in refs:
        if sha_file(output/Path(ref["path"]).name) != ref["sha256"]:
            raise ValueError("recalculated evaluation differs from the six preserved payloads")
    failed_evidence()


def evaluate_test(output, split, *, authorization, data, logits, scored, normalized, predictions, recovery_authorization, support_recovery_authorization, evaluation_release_authorization):
    """Identical primary/replay entrypoint; no new forward, training or observations."""
    execution = release_entry(recovery_authorization, authorization, support_recovery_authorization, evaluation_release_authorization)
    started = time.monotonic()
    kwargs = dict(authorization=authorization, data=data, logits=logits, scored=scored, normalized=normalized,
                  recovery_authorization=recovery_authorization, support_recovery_authorization=support_recovery_authorization)
    auth, frozen, cache, raw, rows, values, prediction_root = preserved_test(predictions, split, **kwargs)
    # All99 prediction passes are sealed and validated before this truth port.
    truths = load_supervision(cache)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    try:
        metrics, evidence, interventions = [], [], []
        for i, truth in enumerate(truths):
            for seed in SEEDS:
                original = {(a, s): values[a, seed, s, "original"][i] for a in ARMS for s in READER_SEEDS}
                evaluated = evaluate_preserved_predictions(rows[seed][i][0], truth["source_ids"], raw[seed][i], seed, original)
                evidence.append({"scene_id": i, "checkpoint_seed": seed,
                    "oracle": evaluated["oracle"], "neural_brier": evaluated["neural_brier"],
                    "candidate_metrics": evaluated["candidate_metrics"], "references": evaluated["references"]})
                interventions.extend(intervention_metrics_for_scene(i, seed, rows[seed][i][1], values, evaluated))
                for reader in READER_SEEDS:
                    metric = {a: {k: evaluated["learned"][a, reader]["metrics"][k] for k in METRICS} for a in ARMS}
                    metric.update({a: {k: evaluated["references"][a]["metrics"][k] for k in METRICS} for a in REFERENCES})
                    metrics.append({"scene_id": i, "checkpoint_seed": seed, "reader_seed": reader,
                        "split": split, "split_seed": SPLITS[split][1], "metrics": metric})
        indices = bootstrap_indices()
        write_npz(output/"bootstrap.npz", indices=indices)
        write_json(output/"metrics.json", metrics)
        write_json(output/"candidate_evidence.json", evidence)
        write_json(output/"summary.json", summarize_test(metrics, split=split, indices=indices))
        write_json(output/"intervention_metrics.json", interventions)
        write_json(output/"intervention_summary.json", summarize_interventions(interventions, split=split, indices=indices))
        del raw, metrics, evidence, interventions, original, evaluated, metric, indices
        # Reconstruct dependence from observable inputs and preserved predictions,
        # not from the diagnostic JSON being checked. No model forward is needed.
        norms = runner.read_normalizers(frozen["normalizers"], auth["common"],
            authorization=frozen["data_authorization"], train=frozen["train"])
        support = []
        for entry in inference_roster():
            arm, seed, reader, intervention = (entry[k] for k in
                ("arm", "checkpoint_seed", "reader_seed", "intervention"))
            if intervention == "original":
                continue
            reconstructed = replay_support([row for _, row in rows[seed]], norms[seed], arm, intervention,
                values[arm, seed, reader, "original"], values[arm, seed, reader, intervention])
            if reconstructed != json.loads((prediction_root/(prefix(entry)+"_support.json")).read_bytes()):
                raise ValueError("preserved support diagnostics differ from input/prediction replay")
            support.append({**entry, "scenes": reconstructed})
        write_json(output/"support.json", support)
        write_json(output/"support_summary.json", summarize_support(support))
        del rows, values, support, reconstructed, norms
        if preserved_test(predictions, split, **kwargs)[0] != auth or load_supervision(cache) != truths:
            raise ValueError("test evaluation sources changed")
        compare_failed(output, split, {"split": split, **kwargs, "predictions": predictions})
        assert_fresh(evaluation_release_authorization, execution)
        seal_bundle(output, role=ROLES["evaluation"], binding={**execution, "split": split,
            **kwargs, "predictions": predictions, "freeze": auth["freeze"]}, resources={"seconds": time.monotonic()-started})
        ref = p.reference(output/"manifest.json")
        checked_evaluation(ref, split, **kwargs, predictions=predictions, evaluation_release_authorization=evaluation_release_authorization)
        return ref
    except BaseException as exc:
        mark_failure(output, exc)
        raise
