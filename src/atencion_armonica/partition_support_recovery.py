"""Closed support-validator recovery, retaining the frozen 06/02 scientific path."""
from __future__ import annotations

from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import time
import numpy as np

from . import learned_partition_provenance as p
from . import learned_partition_runner as runner
from . import partition_test_recovery as old
from . import partition_test_recovery_gate as gate02
from .learned_partition_core import ARMS, READER_SEEDS, model_inputs
from .learned_partition_data import load_supervision
from .learned_partition_metrics import (SEEDS, TESTS, SPLITS, METRICS, REFERENCES,
    evaluate_preserved_predictions, summarize_test, bootstrap_indices)
from .learned_partition_test import (inference_roster, prefix, write_predictions,
    read_predictions, intervention_metrics_for_scene, summarize_interventions, summarize_support)
from .partition_support_guard import replay_support
from .learned_partition_validation import boundary, memoized, fresh_pass
from .partial_compatibility_cache import encoded, sha_file
from .structured_source_artifacts import mark_failure, seal_bundle, write_json, write_npz, verify_bundle

TREE = gate02.TREE
RECOVERY = TREE + "/support_validation_recovery_v1"
LOCAL = gate02.LOCAL
FROZEN = gate02.FROZEN
LIMITS = {k: gate02.LIMITS[k] for k in ("inference", "evaluation")}
SPLITS_NEW = ("ood_polyphony", "deformed_family")
ROLES = {"inference": "learned_predictions_support_recovery_v1",
         "evaluation": "learned_evaluation_support_recovery_v1"}
EVALUATION_FILES = gate02.EVALUATION_FILES
SOURCES = (
    "src/atencion_armonica/partition_support_guard.py",
    "src/atencion_armonica/partition_support_recovery.py",
    "experiments/atencion_armonica/run_partition_support_recovery.py",
    "experiments/atencion_armonica/test_partition_support_recovery.py",
)
PLAN = {"path": LOCAL + "/PLAN_SUPPORT_VALIDATION_RECOVERY.md",
        "sha256": "4ac6c27d4c52f4a90d2102eea4f2646b47d1f7458970a06c26bb9e91d48e6df0"}
PLAN_AUDIT = {"path": LOCAL + "/support_plan_audit.json",
              "sha256": "e91383618d4aa4c91c474d89791818cb273811f57080c58aad75d7be191ed28f"}
PLAN_REPORTS = [
    {"path": gate02.LIBRARY + "/agent_reports/680_partition_support_validation_recovery_plan_audit.md",
     "sha256": "ca65c5ec43da71cef09dcc02e8b05c9fae4f28211fcc8a555656992265faad7a"},
    {"path": gate02.LIBRARY + "/agent_reports/681_partition_support_validation_plan_resolution.md",
     "sha256": "e78b3e9a544ae5d9b9325a74a8bfbf2a247e567cb01c88b632a7ab170872a474"},
]
OLD_AUTH = {"path": gate02.RECOVERY + "/authorization/recovery_02.json",
            "sha256": "fb6c9229329a158a5b43d2622a017bdc53314b3d1a047475f860ae1334a9c7a3"}
OLD_CONTRACT = {"path": gate02.RECOVERY + "/contracts/contract_02.json",
                "sha256": "75242f36a2b2b29610e8944f55eb5b1eba9ccf141e90e5930f0a059bc711de7b"}
DIAGNOSTIC = {"path": gate02.LIBRARY + "/experiment_notes/partition_support_reduction_diagnostic.json",
              "sha256": "aaaf6572fe78d55dc0d4f7cb4247a0f66aca3bc0d5957e5ed4aeaa550e9adfdf"}
FAILED_ROOT = gate02.RECOVERY + "/outputs/ood_polyphony/predictions_01"
FAILED_REQUEST = {"path": gate02.RECOVERY + "/requests/ood_polyphony_predictions_01.json",
                  "sha256": "9b7faf17bc0051ffca4b33e08b7a8a4be5fddc8c22b052879dd9924a570787f9"}
FAILED_TERMINAL = {"path": gate02.RECOVERY + "/supervision/supervisor-zqijy7bc/terminal.json",
                   "sha256": "fed094a5a614dad6445e4672917effe075b30a502354620ea0a05a34685d3f40"}
FAILED_MARKER = {"path": FAILED_ROOT + "/FAILURE.json",
                 "sha256": "7473cd7f803954e12d9f56e2b8d4a8c320bad27c70d8a09fbc9987065cb7e819"}
canonical_path = gate02.canonical_path


def evidence_only(ref):
    """Confinement + bytes, never permission to consume an incomplete stage."""
    if not isinstance(ref, dict) or set(ref) != {"path", "sha256"}:
        raise ValueError("evidence requires exact path/hash")
    canonical_path(ref["path"], p.ROOT)
    return gate02.hash_only(ref)


def failed_evidence():
    record = json.loads(evidence_only(DIAGNOSTIC).read_bytes())
    if (record.get("schema") != "support-reduction-diagnostic-v1"
            or record.get("status") != "DIAGNOSTIC_NOT_CAMPAIGN"
            or record.get("request") != FAILED_REQUEST or record.get("failed_terminal") != FAILED_TERMINAL):
        raise ValueError("support diagnostic identity differs")
    terminal = json.loads(evidence_only(FAILED_TERMINAL).read_bytes())
    request = json.loads(evidence_only(FAILED_REQUEST).read_bytes())
    if (terminal["status"] != "FAILED" or terminal["request"] != FAILED_REQUEST
            or terminal["worker_terminal_confirmed"] is not True
            or terminal["worker_exit_code"] != 1 or terminal["result"] is not None
            or terminal["output"] != FAILED_ROOT or terminal["operation"] != "inference"
            or request["output"] != FAILED_ROOT or request["execution_contract"] != OLD_CONTRACT
            or request["arguments"]["recovery_authorization"] != OLD_AUTH
            or request["arguments"]["normalized"] != record["normalized"]):
        raise ValueError("not the declared failed inference attempt")
    if json.loads(evidence_only(FAILED_MARKER).read_bytes()) != {
            "status": "INCOMPLETE", "error": "ValueError('support comparison changes the common input or incidence')"}:
        raise ValueError("failure marker differs")
    expected = {FAILED_ROOT + f"/seed_{s}/pairs_structure_{r}_original.npz"
                for s in SEEDS for r in READER_SEEDS}
    expected |= {FAILED_ROOT + f"/seed_{SEEDS[0]}/local_compatibility_{READER_SEEDS[0]}_{kind}.npz"
                 for kind in ("original", "zero")}
    inventory = record["failed_artifact_inventory"]
    if len(inventory) != 12 or {r["path"] for r in inventory} != expected | {FAILED_MARKER["path"]}:
        raise ValueError("diagnostic failed inventory differs")
    root = canonical_path(FAILED_ROOT, p.ROOT/gate02.RECOVERY/"outputs"/"ood_polyphony")
    paths = list(root.rglob("*"))
    if any(path.is_symlink() for path in paths) or {
            f.relative_to(p.ROOT).as_posix() for f in paths if f.is_file()} != expected | {FAILED_MARKER["path"]}:
        raise ValueError("failed tree changed or was retroactively sealed")
    for ref in inventory:
        evidence_only(ref)
    evidence_only(record["source"])
    p.verify_reference(record["normalized"])
    for result in record["results"]:
        evidence_only(result["pack"])
        for case in result["cases"]:
            evidence_only(case["canonical_rows"])
    return [r for r in inventory if r["path"] in expected]


def verify_plan_audit():
    p.verify_reference(PLAN)
    if p.read_reference(PLAN_AUDIT) != {
            "schema": "partition-support-plan-audit-v1", "status": "PASS",
            "scope": "SUPPORT_VALIDATION_RECOVERY_PLAN", "target": PLAN,
            "reports": PLAN_REPORTS, "resolved_findings": ["R680-F1", "R680-F2"]}:
        raise PermissionError("exact independent plan resolution missing")
    for ref in PLAN_REPORTS:
        if not p.verify_reference(ref).read_text().strip():
            raise ValueError("empty plan audit")


def contract_candidate():
    return {"schema": "partition-support-recovery-contract-v1", "base_common": gate02.base_gate.common_binding(),
        "old_contract": OLD_CONTRACT, "old_authorization": OLD_AUTH,
        "executor_sources": {name: sha_file(p.ROOT/name) for name in SOURCES},
        "plan": PLAN, "plan_audit": PLAN_AUDIT, "diagnostic": DIAGNOSTIC,
        "failed_request": FAILED_REQUEST, "failed_terminal": FAILED_TERMINAL,
        "failed_marker": FAILED_MARKER, "freeze": FROZEN["freeze"],
        "test_authorization": FROZEN["test_authorization"], "roles": ROLES,
        "test_splits": list(SPLITS_NEW), "limits": LIMITS, "device": "cpu"}


@boundary
@memoized
def verify_contract(ref):
    record = p.read_reference(ref)
    if record != contract_candidate() or hashlib.sha256(encoded(record["base_common"])).hexdigest() != gate02.BASE_SHA:
        raise PermissionError("support recovery sources, scope or base changed")
    old_auth, _ = gate02.verify_authorization(OLD_AUTH)
    if old_auth["contract"] != OLD_CONTRACT:
        raise PermissionError("old recovery authority changed")
    verify_plan_audit()
    failed_evidence()
    return record


def create_contract(output):
    output = canonical_path(output, p.ROOT/RECOVERY/"contracts")
    verify_plan_audit()
    failed_evidence()
    gate02.verify_authorization(OLD_AUTH)
    write_json(output, contract_candidate())
    ref = p.reference(output)
    verify_contract(ref)
    return ref


def verify_implementation_audit(ref, contract):
    record = p.read_reference(ref)
    if (set(record) != {"status", "scope", "contract", "reports", "resolved_plan_findings"}
            or record["status"] != "PASS" or record["scope"] != "SUPPORT_VALIDATION_RECOVERY_IMPLEMENTATION"
            or record["contract"] != contract or record["resolved_plan_findings"] != ["R680-F1", "R680-F2"]
            or not isinstance(record["reports"], list) or not record["reports"]
            or len({r["path"] for r in record["reports"]}) != len(record["reports"])):
        raise PermissionError("independent support implementation acceptance missing")
    for report in record["reports"]:
        if not p.verify_reference(report).read_text().strip():
            raise ValueError("empty implementation audit")


@boundary
@memoized
def verify_authorization(ref):
    auth = p.read_reference(ref)
    if (set(auth) != {"schema", "status", "contract", "implementation_audit", "old_authorization"}
            or auth["schema"] != "partition-support-recovery-authorization-v1"
            or auth["status"] != "READY" or auth["old_authorization"] != OLD_AUTH):
        raise PermissionError("support recovery requires its own authorization")
    contract = verify_contract(auth["contract"])
    verify_implementation_audit(auth["implementation_audit"], auth["contract"])
    return auth, contract


def create_authorization(output, *, contract, implementation_audit):
    output = canonical_path(output, p.ROOT/RECOVERY/"authorization")
    verify_contract(contract)
    verify_implementation_audit(implementation_audit, contract)
    write_json(output, {"schema": "partition-support-recovery-authorization-v1", "status": "READY",
        "contract": contract, "implementation_audit": implementation_audit, "old_authorization": OLD_AUTH})
    ref = p.reference(output)
    verify_authorization(ref)
    return ref


def execution_binding(support_recovery_authorization):
    auth, _ = verify_authorization(support_recovery_authorization)
    return {**gate02.execution_binding(OLD_AUTH), "support_execution_contract": auth["contract"],
            "support_recovery_authorization": support_recovery_authorization}


def recovery_entry(recovery_authorization, authorization, support_recovery_authorization):
    if recovery_authorization != OLD_AUTH or authorization != FROZEN["test_authorization"]:
        raise PermissionError("support recovery requires the exact frozen upstream")
    return execution_binding(support_recovery_authorization)


def checked_bundle(ref, operation, support_recovery_authorization):
    expected = execution_binding(support_recovery_authorization)
    path = p.verify_reference(ref)
    if path.name != "manifest.json":
        raise ValueError("expected support-recovery bundle manifest")
    manifest = verify_bundle(path.parent, ref["sha256"], role=ROLES[operation])
    if any(manifest["binding"].get(k) != v for k, v in expected.items()):
        raise ValueError("support recovery producer identity differs")
    return path.parent, manifest


def assert_fresh(support_recovery_authorization, expected):
    if fresh_pass(execution_binding)(support_recovery_authorization) != expected:
        raise ValueError("support recovery identity changed during stage")


def checked_evaluation(ref, split, *, authorization, data, logits, scored, normalized,
                       predictions, recovery_authorization, support_recovery_authorization):
    execution = recovery_entry(recovery_authorization, authorization, support_recovery_authorization)
    if split not in SPLITS_NEW:
        raise PermissionError("support recovery does not reinterpret IID or beta")
    expected = {**execution, "split": split, "authorization": authorization, "data": data,
        "logits": logits, "scored": scored, "normalized": normalized, "predictions": predictions,
        "freeze": FROZEN["freeze"]}
    root, manifest = checked_bundle(ref, "evaluation", support_recovery_authorization)
    if manifest["binding"] != expected or set(manifest["artifacts_sha256"]) != EVALUATION_FILES:
        raise ValueError("evaluation binding or exact eight-payload inventory differs")
    return root, manifest


def compare_evaluation_replay(evaluation, replay, split, **arguments):
    _, a = checked_evaluation(evaluation, split, **arguments)
    _, b = checked_evaluation(replay, split, **arguments)
    if a["artifacts_sha256"] != b["artifacts_sha256"]:
        raise ValueError("evaluation/replay payloads are not byte-exact")


@boundary
def test_inputs(split, *, authorization, data, logits, scored, normalized,
                recovery_authorization, support_recovery_authorization):
    recovery_entry(recovery_authorization, authorization, support_recovery_authorization)
    if split not in SPLITS_NEW:
        raise PermissionError("only the two unfinished test splits may use support recovery")
    return old.test_inputs(split, authorization=authorization, data=data, logits=logits,
        scored=scored, normalized=normalized, recovery_authorization=recovery_authorization)


def compare_failed_polyphony(output, split, arguments):
    if split != "ood_polyphony":
        return
    refs = failed_evidence()
    original_arguments = json.loads(evidence_only(FAILED_REQUEST).read_bytes())["arguments"]
    if {k: v for k, v in arguments.items() if k != "support_recovery_authorization"} != {
            k: v for k, v in original_arguments.items() if k != "gpu_grant"}:
        raise ValueError("failed-control comparison requires the exact original inputs")
    for ref in refs:
        original = evidence_only(ref)
        replacement = output/original.relative_to(p.ROOT/FAILED_ROOT)
        if sha_file(replacement) != ref["sha256"]:
            raise ValueError("recovered prediction bytes differ from the failed control")
        old.compare_npz_exact(original, replacement)
    failed_evidence()


def infer_test(output, split, *, authorization, data, logits, scored, normalized, gpu_grant, recovery_authorization, support_recovery_authorization):
    execution = recovery_entry(recovery_authorization, authorization, support_recovery_authorization)
    started = time.monotonic()
    kwargs = dict(authorization=authorization, data=data, logits=logits, scored=scored, normalized=normalized,
                  recovery_authorization=recovery_authorization, support_recovery_authorization=support_recovery_authorization)
    auth, frozen, cache, _, rows, normal_root, norms = test_inputs(split, **kwargs)
    training_auth = p.read_reference(frozen["data_authorization"])
    device = training_auth["training_device"]
    if device != "cpu":
        raise PermissionError("this recovery is restricted to the selected CPU heads")
    if device == "cpu" and gpu_grant is not None:
        raise ValueError("CPU head inference cannot occupy a GPU lease")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    try:
        from .structured_source_profile import gpu_lease
        lease = gpu_lease(gpu_grant) if device == "cuda:0" else nullcontext(None)
        with lease as availability:
            import torch
            from .learned_partition_model import PartitionCostHead
            from .learned_partition_snapshots import read_snapshot
            from .learned_partition_inference import predict_inputs, intervention_inputs
            from .partition_support_guard import intervention_report
            torch.set_num_threads(1)
            torch.use_deterministic_algorithms(True)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            if device == "cuda:0":
                torch.cuda.reset_peak_memory_stats(0)
            for seed in SEEDS:
                (output/f"seed_{seed}").mkdir()
            for cell in frozen["cells"]:
                seed, reader, arm = (cell[k] for k in ("checkpoint_seed", "reader_seed", "arm"))
                cell_root = p.verify_reference(cell["result"]).parent
                cell_manifest = p.read_reference(cell["result"])
                binding = cell_manifest["binding"]["cell"]
                chain = json.loads((cell_root/"chain.json").read_bytes())
                epoch = frozen["selection"]["selected"][arm]["epoch"]
                selected = None
                for ref in chain["snapshots"]:
                    manifest = p.read_reference(ref)
                    if manifest["position"] == {"epoch": epoch, "next_batch": 0, "steps": epoch*128}:
                        selected = ref
                if selected is None:
                    raise ValueError("selected epoch is absent from the completed training chain")
                state, _ = read_snapshot(selected, expected_binding=binding)
                model = PartitionCostHead(arm, reader).to(device)
                model.load_state_dict(state["model"], strict=True)
                del state
                original = runner.read_inputs(normal_root/f"seed_{seed}/{arm}.npz", scene_ids=list(range(512)),
                                              dim=8 if arm == ARMS[0] else 9)
                for inputs, (_, row) in zip(original, rows[seed]):
                    expected = model_inputs(row, norms[seed], arm)
                    if any(not np.array_equal(inputs[k], expected[k]) for k in expected):
                        raise ValueError("test input differs from its frozen train-only transform")
                predictions = predict_inputs(model, original)
                record = {"arm": arm, "checkpoint_seed": seed, "reader_seed": reader, "intervention": "original"}
                write_predictions(output/f"{prefix(record)}.npz", predictions)
                interventions = [] if arm == ARMS[0] else ["zero", "rotate_one"]
                if arm == "shared_source":
                    interventions.append("original_sham")
                for intervention in interventions:
                    changed = [intervention_inputs(row, norms[seed], arm, intervention) for _, row in rows[seed]]
                    altered = predict_inputs(model, changed)
                    entry = {**record, "intervention": intervention}
                    write_predictions(output/f"{prefix(entry)}.npz", altered)
                    support = [intervention_report(row, before, after, pv, av)
                        for (_, row), before, after, pv, av in zip(rows[seed], original, changed, predictions, altered)]
                    write_json(output/f"{prefix(entry)}_support.json", support)
                del model
                if time.monotonic()-started > 1200:
                    raise TimeoutError("test inference exceeded its forward envelope")
            peak = torch.cuda.max_memory_reserved(0) if device == "cuda:0" else 0
            if peak >= 2*1024**3:
                raise RuntimeError("test inference exceeded its VRAM envelope")
        write_json(output/"index.json", inference_roster())
        del rows, norms, original, inputs, row, expected, predictions, changed, altered, support
        if test_inputs(split, **kwargs)[0] != auth:
            raise ValueError("test inference provenance changed")
        compare_failed_polyphony(output, split, {"split": split, **kwargs})
        assert_fresh(support_recovery_authorization, execution)
        seal_bundle(output, role=ROLES["inference"], binding={**execution, "split": split,
            **kwargs, "freeze": auth["freeze"], "gpu_grant": gpu_grant},
            resources={"seconds": time.monotonic()-started, "peak_reserved_bytes": peak, "availability": availability})
        ref = p.reference(output/"manifest.json")
        checked_bundle(ref, "inference", support_recovery_authorization)
        return ref
    except BaseException as exc:
        mark_failure(output, exc)
        raise

@boundary
def preserved_test(ref, split, *, authorization, data, logits, scored, normalized, recovery_authorization, support_recovery_authorization):
    execution = recovery_entry(recovery_authorization, authorization, support_recovery_authorization)
    kwargs = dict(authorization=authorization, data=data, logits=logits, scored=scored, normalized=normalized,
                  recovery_authorization=recovery_authorization, support_recovery_authorization=support_recovery_authorization)
    auth, frozen, cache, raw, rows, _, _ = test_inputs(split, **kwargs)
    root, m = checked_bundle(ref, "inference", support_recovery_authorization)
    binding = m["binding"]
    expected = {**execution, "split": split, **kwargs, "freeze": auth["freeze"]}
    if set(binding) != set(expected)|{"gpu_grant"} or any(binding[k] != v for k, v in expected.items()):
        raise ValueError("test predictions use a different selected model or input chain")
    roster = inference_roster()
    files = {"index.json", *[prefix(r)+".npz" for r in roster],
             *[prefix(r)+"_support.json" for r in roster if r["intervention"] != "original"]}
    if json.loads((root/"index.json").read_bytes()) != roster or set(m["artifacts_sha256"]) != files:
        raise ValueError("test prediction/intervention inventory is incomplete")
    values = {}
    for entry in roster:
        key = tuple(entry[k] for k in ("arm", "checkpoint_seed", "reader_seed", "intervention"))
        values[key] = read_predictions(root/(prefix(entry)+".npz"), [r[1] for r in rows[entry["checkpoint_seed"]]])
    return auth, frozen, cache, raw, rows, values, root

def evaluate_test(output, split, *, authorization, data, logits, scored, normalized, predictions, recovery_authorization, support_recovery_authorization):
    """Identical primary/replay entrypoint; no new forward, training or observations."""
    execution = recovery_entry(recovery_authorization, authorization, support_recovery_authorization)
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
        del raw, rows, values, metrics, evidence, interventions, support, original, evaluated, metric, reconstructed, norms, indices
        if preserved_test(predictions, split, **kwargs)[0] != auth or load_supervision(cache) != truths:
            raise ValueError("test evaluation sources changed")
        assert_fresh(support_recovery_authorization, execution)
        seal_bundle(output, role=ROLES["evaluation"], binding={**execution, "split": split,
            **kwargs, "predictions": predictions, "freeze": auth["freeze"]}, resources={"seconds": time.monotonic()-started})
        ref = p.reference(output/"manifest.json")
        checked_evaluation(ref, split, **kwargs, predictions=predictions)
        return ref
    except BaseException as exc:
        mark_failure(output, exc)
        raise
