"""Post-replay descriptive report, under the shared audit-stage budget.

No truth parser, model, sampler or fitting is called. Never run while the
prospective/evaluation/replay operators are incomplete.
"""
from __future__ import annotations

import time
LAUNCH_STARTED = time.monotonic()

import fcntl
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from experiments.atencion_armonica.run_geometric_decision_fresh import completed_operation, bounded_operation
from experiments.atencion_armonica import run_geometric_decision_postseal as postseal
from experiments.atencion_armonica.prepare_geometric_decision_open import ROOT, PROTOCOL, PROTOCOL_SHA, reference, code_snapshot
from experiments.atencion_armonica.profile_geometric_decision import CONTROL_BINDING
from src.atencion_armonica.geometric_decision_open import ReadOnlyStore
from src.atencion_armonica.geometric_decision_store import ArtifactStore, BASES
from src.atencion_armonica.geometric_decision_release import admitted_seal, TEST_ROSTER
from src.atencion_armonica.geometric_decision_reporting import roundtrip_comparison, coordinate_comparison
from src.atencion_armonica.geometric_decision_evaluation import plain, NAMES
from src.atencion_armonica.partial_compatibility_cache import encoded

OPERATOR = "experiments/atencion_armonica/report_geometric_decision.py"
PLAN = "experiments/atencion_armonica/PLAN_GEOMETRIC_DECISION_FINAL_REPORT.md"


def sources():
    return sorted([*code_snapshot(), reference(OPERATOR), reference(PLAN),
        reference(postseal.OPERATOR), reference(postseal.PLAN),
        reference("experiments/atencion_armonica/run_geometric_decision_fresh.py")], key=lambda r: r["path"])


def completed_inputs(control):
    """Refuse before reading any metric payload unless both passes completed."""
    records = [completed_operation(control, operation) for operation in ("prospective-observables", "evaluate", "replay")]
    fresh, evaluate, replay = records
    freeze_ref, seal_ref = fresh[4]["test_freeze"], fresh[4]["seal"]
    frozen = control.json(freeze_ref)
    for source in frozen["code"]:
        if reference(source["path"]) != source:
            raise ValueError("scientific source changed after freeze")
    for row in (evaluate, replay):
        _, _, start, manifest, output = row
        if (start["stage"] != "evaluation" or start["binding"] != control.binding
                or manifest["observable_finish"] != fresh[0] or manifest["test_freeze"] != freeze_ref
                or output["manifest"] != start["manifest"] or output["test_freeze"] != freeze_ref
                or output["root"] != str(BASES[0]/"evaluation")):
            raise ValueError("report requires the same completed frozen evaluation")
    execution_contract = evaluate[3]["execution_contract"]
    if any(row[3]["execution_contract"] != execution_contract or row[4]["execution_contract"] != execution_contract
            for row in (evaluate, replay)):
        raise ValueError("evaluation and replay use different execution contracts")
    postseal.validate_contract(control, execution_contract, freeze_ref)
    recovery = postseal.frozen_op.admitted_observable_replay(control, fresh[0], freeze_ref, seal_ref)
    if evaluate[3]["observable_replay_finish"] is not None or replay[3]["observable_replay_finish"] != recovery:
        raise ValueError("metric report lacks the exact observable recovery")
    if evaluate[4]["complete"] != replay[4]["complete"]:
        raise ValueError("metric replay differs from original completion")
    binding = {"test_freeze": freeze_ref, "prediction_seal": seal_ref}
    data = encoded(binding)
    store = ReadOnlyStore(BASES[0]/"evaluation", binding_ref={"path": "binding.json", "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest()})
    complete = store.json(evaluate[4]["complete"])
    if (complete["schema"] != "geometric-decision-evaluation-complete-v1" or complete["binding"] != binding
            or complete["original_scenes"] != 2048 or len(complete["results"]) != 4
            or [r["split"] for r in complete["results"]] != [r["split"] for r in TEST_ROSTER]):
        raise ValueError("report requires four complete scenario evaluations")
    return store, complete, {"fresh_finish": fresh[0], "evaluation_finish": evaluate[0],
        "replay_finish": replay[0], "evaluation_complete": evaluate[4]["complete"],
        "test_freeze": freeze_ref, "prediction_seal": seal_ref, "execution_contract": execution_contract,
        "observable_replay_finish": recovery}


def sham_summary(evaluation, targets, *, check):
    cells, scenes = {}, []
    for ref in targets:
        check()
        target = evaluation.json(ref)
        for cp, diagnosis in target["diagnostics"].items():
            row = cells.setdefault(cp, {"scenes": 0, "eligible_scenes": 0, "candidates": 0,
                "scalar_changed": 0, "six_channels_changed": 0, "scalar_fractions": [], "six_channel_fractions": []})
            first, second = diagnosis["scalar_changed_mask"], diagnosis["six_channel_sham"]["changed_mask"]
            if (len(first) != target["candidate_count"] or len(second) != len(first)
                    or any(type(v) is not bool for v in first+second)):
                raise ValueError("sham support differs from evaluated candidate universe")
            donors = diagnosis["six_channel_sham"]["donors"]
            if (any(type(v) is not int for v in donors) or sorted(donors) != list(range(len(first)))
                    or any((first[i] or second[i]) and donor == i for i, donor in enumerate(donors))):
                raise ValueError("sham donors must permute the candidate universe consistently")
            groups = diagnosis["six_channel_sham"]["strata"]
            covered = []
            for group in groups:
                ids = group["candidate_ids"]
                if (not ids or any(type(i) is not int or not 0 <= i < len(first) for i in ids)
                        or len(set(ids)) != len(ids) or sorted(group["donors"]) != sorted(ids)
                        or group["donors"] != [donors[i] for i in ids]):
                    raise ValueError("sham strata must preserve donor membership")
                covered.extend(ids)
            if sorted(covered) != list(range(len(first))):
                raise ValueError("sham strata do not cover the candidate universe")
            scenes.append({"scene_id": target["scene_id"], "checkpoint_seed": cp, "source": ref,
                "candidate_denominator": len(first), "diagnostics": diagnosis,
                "stratum_support": [{"sizes": g["sizes"], "candidate_denominator": len(g["candidate_ids"]),
                    "singleton": len(g["candidate_ids"]) == 1,
                    "scalar_changed": sum(first[i] for i in g["candidate_ids"]),
                    "six_channels_changed": sum(second[i] for i in g["candidate_ids"])} for g in groups]})
            row["scenes"] += 1
            row["candidates"] += len(first)
            row["scalar_changed"] += sum(first)
            row["six_channels_changed"] += sum(second)
            if first:
                row["eligible_scenes"] += 1
                row["scalar_fractions"].append(sum(first)/len(first))
                row["six_channel_fractions"].append(sum(second)/len(second))
    for row in cells.values():
        for source, name in (("scalar_fractions", "scalar_scene_mean"), ("six_channel_fractions", "six_channel_scene_mean")):
            values = row.pop(source)
            row[name] = float(np.mean(values)) if values else None
    return {"checkpoint_cells": cells, "scene_checkpoint_rows": scenes,
        "authority": "descriptive within-scene correspondence change, not independent replications"}


def stratum_rows(rows, targets, *, check):
    """Extract preserved metrics and support; never refit or pool strata."""
    if [r["scene_id"] for r in rows] != list(targets):
        raise ValueError("stratified rows differ from the target scene roster")
    result = []
    for row in rows:
        check()
        target = targets[row["scene_id"]]
        full = row["full"]
        if (set(full) != {"candidate_count", "chosen", "score_order", "oracles", "tau", "errors", "decision"}
                or full["candidate_count"] != target["candidate_count"]
                or set(full["oracles"]) != {"tM", "tD"}
                or not isinstance(full["score_order"], dict)):
            raise ValueError("full-scene diagnostic schema or support differs")
        expected = {k: v for k, v in target["strata"].items() if k != "full"}
        if set(row["strata"]) != set(expected):
            raise ValueError("stratum schemes differ from evaluated definitions")
        support = {}
        for scheme, definitions in expected.items():
            groups = row["strata"][scheme]
            if set(groups) != set(definitions):
                raise ValueError("stratum names differ from evaluated definitions")
            for name, group in groups.items():
                if group["candidate_ids"] != definitions[name] or set(group["metrics"]) != set(NAMES):
                    raise ValueError("stratum candidates or metric roster differ")
            support[scheme] = {"strata": len(groups),
                "candidate_denominator": sum(len(g["candidate_ids"]) for g in groups.values()),
                "singleton_strata": sum(len(g["candidate_ids"]) == 1 for g in groups.values()),
                "defined_strata": {k: sum(g["metrics"][k] is not None for g in groups.values()) for k in NAMES},
                "undefined_strata": {k: sum(g["metrics"][k] is None for g in groups.values()) for k in NAMES}}
        result.append({"scene_id": row["scene_id"], "candidate_denominator": target["candidate_count"],
            "eligible_scene": target["candidate_count"] > 0, "full": full,
            "strata": row["strata"], "support": support})
    return result


def stratified_report(evaluation, current, output, folder, *, check):
    targets = {}
    for ref in current["targets"]:
        check()
        target = evaluation.json(ref)
        if target["scene_id"] in targets:
            raise ValueError("duplicate stratified target scene")
        targets[target["scene_id"]] = target
    if list(targets) != current["scene_ids"] or len(current["heads"]) != 144:
        raise ValueError("stratified report needs every scene and head")
    heads = []
    for i, ref in enumerate(current["heads"]):
        check()
        head = evaluation.json(ref)
        if head["scene_ids"] != current["scene_ids"]:
            raise ValueError("stratified head roster differs")
        value = {"source": ref, "identity": head["identity"],
            "rows": stratum_rows(head["rows"], targets, check=check)}
        heads.append(output.publish_json(f"{folder}/heads/{i:03d}.json", value))
    classical = evaluation.json(current["classical"])
    classics = {}
    for name in current["summary"]["classical_order"]:
        values = [{"scene_id": row["scene_id"], **row["values"][name]} for row in classical]
        classics[name] = output.publish_json(f"{folder}/classical/{name}.json", {
            "source": current["classical"], "name": name, "rows": stratum_rows(values, targets, check=check)})
    return output.publish_json(folder+"/index.json", {"scene_ids": current["scene_ids"],
        "total_scenes": len(targets), "eligible_scenes": sum(t["candidate_count"] > 0 for t in targets.values()),
        "heads": heads, "classical": classics,
        "authority": "preserved per-stratum metrics, not a pooled or replacement primary; cells are conditioned"})


def probe_report(fresh, batch, *, check):
    original = fresh.json(batch["sources"])
    inputs = fresh.json(batch["inputs"])
    index = {sid: i for i, sid in enumerate(original["scene_ids"])}
    derived = batch["roundtrip"]
    if derived is None:
        return {"scene_ids": [], "coordinates": [], "heads": []}
    after = fresh.json(derived["sources"])
    after_inputs = fresh.json(derived["inputs"])
    scenes = []
    for j, sid in enumerate(after["scene_ids"]):
        check()
        i = index[sid]
        a, b = fresh.json(original["sources"][i]), fresh.json(after["sources"][j])
        if b["derivation"]["parent"] != original["sources"][i]:
            raise ValueError("probe report lacks original event lineage")
        before_input, after_input = fresh.json(inputs["records"][i]), fresh.json(after_inputs["records"][j])
        scenes.append({"scene_id": sid, "original_index": i, "before": a["scene"], "after": b["scene"],
            "before_input": fresh.arrays(before_input["arrays"]), "after_input": fresh.arrays(after_input["arrays"]),
            "coordinate_source": b["coordinates"], "coordinates": coordinate_comparison(fresh.arrays(b["coordinates"]))})
    heads = []
    if len(batch["records"]) != 144 or len(derived["records"]) != 144:
        raise ValueError("probe report requires the full head roster")
    for first, second in zip(batch["records"], derived["records"]):
        check()
        a, b = fresh.json(first["prediction"]), fresh.json(second["prediction"])
        if a["head"] != b["head"] or first["head"] != second["head"]:
            raise ValueError("roundtrip paired different heads")
        before_values, after_values = fresh.arrays(a["arrays"]), fresh.arrays(b["arrays"])
        transports = fresh.json(first["transport"])
        if transports["scene_ids"] != after["scene_ids"]:
            raise ValueError("transport and roundtrip use different probes")
        cp, rows = a["checkpoint_seed"], []
        for j, scene in enumerate(scenes):
            check()
            i = scene["original_index"]
            x, y = before_values["offsets"][i:i+2]
            z, w = after_values["offsets"][j:j+2]
            transport = fresh.json(transports["records"][j])
            if transport["scene_id"] != scene["scene_id"] or transport["prediction"] != first["prediction"]:
                raise ValueError("transport belongs to a different readout")
            rows.append({"scene_id": scene["scene_id"], "transport_source": transports["records"][j],
                "transport": transport["diagnostic"], "roundtrip": roundtrip_comparison(scene["before"], scene["after"],
                    energy_before=before_values["energy"][x:y], energy_after=after_values["energy"][z:w],
                    evidence_before=scene["before_input"][f"inputs/{cp}/evidence"],
                    evidence_after=scene["after_input"][f"inputs/{cp}/evidence"],
                    choice_before=a["choices"][i], choice_after=b["choices"][j])})
        heads.append({"identity": {k: a[k] for k in ("checkpoint_seed", "reader_seed", "arm", "stage")},
            "before_prediction": first["prediction"], "after_prediction": second["prediction"], "rows": rows})
    return {"scene_ids": after["scene_ids"], "coordinates": [
        {k: s[k] for k in ("scene_id", "coordinate_source", "coordinates")} for s in scenes], "heads": heads}


def collect(control, output, *, check):
    evaluation, complete, origins = completed_inputs(control)
    fresh, seal, seal_ref = admitted_seal(control, origins["fresh_finish"], origins["test_freeze"], check=check)
    if seal_ref != origins["prediction_seal"]:
        raise ValueError("report references a different prediction seal")
    results = []
    for row, batch in zip(complete["results"], seal["batches"]):
        check()
        current = evaluation.json(row["original"])
        observed = fresh.json(batch["observed"])
        if (current["scene_ids"] != list(range(512)) or current["observed_sources"] != observed["sources"]
                or row["split"] != batch["split"] or current["summary"]["metric_order"] != list(NAMES)):
            raise ValueError("report summary scope differs from sealed observations")
        probes = probe_report(fresh, observed, check=check)
        probe_ref = output.publish_json(row["split"]+"/probes.json", plain(probes))
        refs = output.publish_json(row["split"]+"/summary.json", {
            "source": row, "summary": current["summary"], "probe_report": probe_ref,
            "stratified_report": stratified_report(evaluation, current, output, row["split"]+"/strata", check=check),
            "sham": sham_summary(evaluation, current["targets"], check=check)})
        results.append({"split": row["split"], "summary": refs})
    primary = evaluation.json(complete["primary"])
    if primary["split"] != "deformed_family" or primary["metric"] != "regret_tM" or primary["stage"] != "selected":
        raise ValueError("report primary differs from the frozen estimand")
    return output.publish_json("complete.json", {"schema": "geometric-decision-report-v1", "binding": output.binding,
        "origins": origins, "scenarios": results, "primary_source": complete["primary"], "primary": primary["summary"],
        "authority": "descriptive extraction; independent technical and horizon review still required"})


def main():
    if (Path.cwd().resolve() != ROOT or os.environ.get("CUDA_VISIBLE_DEVICES") != ""
            or any(os.environ.get(k) != "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"))):
        raise ValueError("report requires one-thread CPU-only project environment")
    view = ReadOnlyStore(BASES[0]/"control", binding_ref=CONTROL_BINDING)
    control = ArtifactStore(view.root, binding=view.binding)
    with control.path("operator.lock").open("ab") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Cheap prerequisite check, not metric/observable access or admission.
        completed_operation(control, "replay")
        if control.path("manifests/report.json").exists():
            raise ValueError("report already attempted; preserve receipts instead of retrying")
        code, protocol = sources(), reference(PROTOCOL)
        output = ArtifactStore(BASES[0]/"report", binding={"code": code, "protocol": protocol})
        manifest = control.publish_json("manifests/report.json", {"operation": "post-replay-report",
            "code": code, "protocol": protocol, "root": str(output.root)})
        def verify():
            if sources() != code or reference(PROTOCOL) != protocol or protocol["sha256"] != PROTOCOL_SHA:
                raise ValueError("report source changed during extraction")
        def operation(check):
            result = collect(control, output, check=check)
            return control.publish_json("outputs/report.json", {"manifest": manifest, "root": str(output.root),
                "binding": output.reference(output.path("binding.json")), "result": result})
        result = bounded_operation(control, manifest, stage="audit", started_at=LAUNCH_STARTED,
            verify=verify, reservation=600., operation=operation)
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
