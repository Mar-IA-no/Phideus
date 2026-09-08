"""Seal all test predictions before opening truth; replay without a new forward."""
from __future__ import annotations

from contextlib import nullcontext
import json
from pathlib import Path
import time

import numpy as np

from . import learned_partition_gate as gate
from . import learned_partition_provenance as p
from . import learned_partition_runner as runner
from .learned_partition_core import ARMS, READER_SEEDS, model_inputs
from .learned_partition_data import _bundle, load_supervision
from .learned_partition_metrics import (SEEDS, TESTS, SPLITS, METRICS, REFERENCES,
    evaluate_preserved_predictions, summarize_test, bootstrap_indices)
from .structured_source_artifacts import mark_failure, seal_bundle, write_json, write_npz


def inference_roster():
    return [{"arm": arm, "checkpoint_seed": seed, "reader_seed": reader, "intervention": intervention}
        for seed in SEEDS for reader in READER_SEEDS for arm in ARMS
        for intervention in (("original",) if arm == ARMS[0] else
            (("original", "zero", "rotate_one", "original_sham") if arm == "shared_source"
             else ("original", "zero", "rotate_one")))]


def prefix(row):
    return f"seed_{row['checkpoint_seed']}/{row['arm']}_{row['reader_seed']}_{row['intervention']}"


def write_predictions(path, values):
    if len(values) != 512 or any(v.dtype != np.float32 or v.ndim != 2 or v.shape[1] != 2
                               or not 1 <= len(v) <= 64 or not np.isfinite(v).all() or np.any(v < 0) for v in values):
        raise ValueError("test predictions must preserve all finite candidate components")
    offsets = np.r_[np.int64(0), np.cumsum([len(v) for v in values], dtype=np.int64)]
    write_npz(path, components=np.concatenate(values), offsets=offsets, scene_ids=np.arange(512, dtype=np.int64))


def read_predictions(path, pools):
    with np.load(path, allow_pickle=False) as raw:
        if set(raw.files) != {"components", "offsets", "scene_ids"}:
            raise ValueError("prediction schema differs or contains truth")
        values, offsets, ids = (raw[k] for k in ("components", "offsets", "scene_ids"))
    expected = np.r_[np.int64(0), np.cumsum([len(row.candidates) for row in pools], dtype=np.int64)]
    if (len(pools) != 512 or values.dtype != np.float32 or values.shape != (expected[-1], 2)
            or not np.isfinite(values).all() or np.any(values < 0) or offsets.dtype != np.int64
            or not np.array_equal(offsets, expected) or ids.dtype != np.int64 or not np.array_equal(ids, np.arange(512))):
        raise ValueError("preserved prediction candidate/scene order differs")
    return [values[lo:hi] for lo, hi in zip(offsets[:-1], offsets[1:])]


def test_inputs(split, *, authorization, data, logits, scored, normalized):
    if split not in TESTS:
        raise PermissionError("test inference requires a declared fresh test role")
    auth, cache = runner.stage_inputs(authorization, split, 0, data)
    frozen = p.read_reference(auth["freeze"])
    raw = runner.ordered_forward(logits, cache, auth["common"], authorization=authorization, data=data)
    rows = runner.scored_shard(scored, cache, auth["common"], authorization=authorization, data=data, logits=logits)
    normal_root = runner.normalized_shard(normalized, cache, auth["common"], authorization=authorization,
        data=data, logits=logits, scored=scored, normalizers=frozen["normalizers"], train=frozen["train"])
    norms = runner.read_normalizers(frozen["normalizers"], auth["common"],
                                   authorization=frozen["data_authorization"], train=frozen["train"])
    return auth, frozen, cache, raw, rows, normal_root, norms


def infer_test(output, split, *, authorization, data, logits, scored, normalized, gpu_grant):
    started = time.monotonic()
    kwargs = dict(authorization=authorization, data=data, logits=logits, scored=scored, normalized=normalized)
    auth, frozen, cache, _, rows, normal_root, norms = test_inputs(split, **kwargs)
    training_auth = p.read_reference(frozen["data_authorization"])
    device = training_auth["training_device"]
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
            from .learned_partition_inference import predict_inputs, intervention_inputs, intervention_report
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
                if time.monotonic()-started > 600:
                    raise TimeoutError("test inference exceeded its forward envelope")
            peak = torch.cuda.max_memory_reserved(0) if device == "cuda:0" else 0
            if peak >= 2*1024**3:
                raise RuntimeError("test inference exceeded its VRAM envelope")
        write_json(output/"index.json", inference_roster())
        if test_inputs(split, **kwargs)[0] != auth:
            raise ValueError("test inference provenance changed")
        seal_bundle(output, role="learned_test_predictions", binding={"common": auth["common"], "split": split,
            **kwargs, "freeze": auth["freeze"], "gpu_grant": gpu_grant},
            resources={"seconds": time.monotonic()-started, "peak_reserved_bytes": peak, "availability": availability})
        return p.reference(output/"manifest.json")
    except BaseException as exc:
        mark_failure(output, exc)
        raise


def preserved_test(ref, split, *, authorization, data, logits, scored, normalized):
    kwargs = dict(authorization=authorization, data=data, logits=logits, scored=scored, normalized=normalized)
    auth, frozen, cache, raw, rows, _, _ = test_inputs(split, **kwargs)
    root, m = _bundle(ref, "learned_test_predictions", auth["common"])
    binding = m["binding"]
    expected = {"common": auth["common"], "split": split, **kwargs, "freeze": auth["freeze"]}
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


def evaluate_test(output, split, *, authorization, data, logits, scored, normalized, predictions):
    """Identical primary/replay entrypoint; no new forward, training or observations."""
    started = time.monotonic()
    kwargs = dict(authorization=authorization, data=data, logits=logits, scored=scored, normalized=normalized)
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
        if preserved_test(predictions, split, **kwargs)[0] != auth or load_supervision(cache) != truths:
            raise ValueError("test evaluation sources changed")
        seal_bundle(output, role="learned_test_evaluation", binding={"common": auth["common"], "split": split,
            **kwargs, "predictions": predictions, "freeze": auth["freeze"]}, resources={"seconds": time.monotonic()-started})
        return p.reference(output/"manifest.json")
    except BaseException as exc:
        mark_failure(output, exc)
        raise


def intervention_metrics_for_scene(scene_id, checkpoint_seed, row, predictions, evaluated):
    """Truth-only post-hoc metrics: reuse the already evaluated candidate pool."""
    from .learned_partition_readout import choose_costs
    records = []
    for entry in inference_roster():
        if entry["checkpoint_seed"] != checkpoint_seed or entry["intervention"] == "original":
            continue
        arm, reader, intervention = (entry[k] for k in ("arm", "reader_seed", "intervention"))
        chosen = choose_costs(predictions[arm, checkpoint_seed, reader, intervention][scene_id], row.candidates)
        metrics = {k: evaluated["candidate_metrics"][chosen["candidate_index"]][k] for k in METRICS}
        original = {k: evaluated["learned"][arm, reader]["metrics"][k] for k in METRICS}
        records.append({**entry, "scene_id": scene_id, "candidate_index": chosen["candidate_index"],
            "original_candidate_index": evaluated["learned"][arm, reader]["choice"]["candidate_index"],
            "metrics": metrics, "original_metrics": original,
            "delta_vs_original": {k: float(metrics[k])-float(original[k]) for k in METRICS}})
    return records


def summarize_interventions(records, *, split, indices):
    """Paired scene-first descriptive intervals, including k and fragmentation."""
    if split not in TESTS or np.asarray(indices).dtype != np.int64 or not np.array_equal(indices, bootstrap_indices()):
        raise ValueError("intervention summary requires the declared test/bootstrap")
    roster = [r for r in inference_roster() if r["intervention"] != "original"]
    expected = {(r["arm"], r["checkpoint_seed"], r["reader_seed"], r["intervention"], i)
                for r in roster for i in range(512)}
    indexed = {}
    fields = {*roster[0], "scene_id", "candidate_index", "original_candidate_index",
              "metrics", "original_metrics", "delta_vs_original"}
    for record in records:
        if set(record) != fields:
            raise ValueError("intervention metrics schema differs")
        key = tuple(record[k] for k in ("arm", "checkpoint_seed", "reader_seed", "intervention", "scene_id"))
        if key not in expected or key in indexed or any(type(record[k]) is not int for k in
                ("checkpoint_seed", "reader_seed", "scene_id", "candidate_index", "original_candidate_index")):
            raise ValueError("intervention cases are duplicated, missing or out of roster")
        for name in ("metrics", "original_metrics", "delta_vs_original"):
            if set(record[name]) != set(METRICS) or not np.isfinite(list(record[name].values())).all():
                raise ValueError("intervention omitted finite partition metrics")
        if any(record["delta_vs_original"][k] != float(record["metrics"][k])-float(record["original_metrics"][k]) for k in METRICS):
            raise ValueError("intervention deltas differ from the same original head")
        indexed[key] = record
    if set(indexed) != expected:
        raise ValueError("intervention metrics require every case, regardless of input support")
    result = {}
    for arm, intervention in dict.fromkeys((r["arm"], r["intervention"]) for r in roster):
        values = {}
        for field in ("metrics", "original_metrics", "delta_vs_original"):
            values[field] = np.array([[[[indexed[arm, seed, reader, intervention, i][field][k] for k in METRICS]
                                     for reader in READER_SEEDS] for seed in SEEDS] for i in range(512)], np.float64)
        scene_delta = values["delta_vs_original"].mean(axis=(1, 2))
        samples = scene_delta[indices].mean(axis=1)
        intervals = np.percentile(samples, [2.5, 97.5], axis=0)
        result[arm+"/"+intervention] = {
            "means": dict(zip(METRICS, values["metrics"].mean(axis=(1, 2)).mean(axis=0).tolist())),
            "original_means": dict(zip(METRICS, values["original_metrics"].mean(axis=(1, 2)).mean(axis=0).tolist())),
            "delta_vs_original": {k: {"delta": float(scene_delta[:, j].mean()),
                "interval": intervals[:, j].tolist(), "nominal_coverage": .95, "family": "descriptive"}
                for j, k in enumerate(METRICS)},
            "per_cell_means": values["metrics"].mean(axis=0).tolist(),
            "cases": 4608, "excluded_cases": 0}
    return {"split": split, "split_seed": SPLITS[split][1], "scene_count": 512,
        "conditional_on_checkpoint_and_reader_seeds": True, "metric_order": list(METRICS),
        "interventions": result, "bootstrap": {"unit": "scene", "resamples": 2000,
            "seed": 2026090894, "method": "percentile", "use": "post_hoc_descriptive_not_selection"}}


def replay_support(rows, normalizer, arm, intervention, original_predictions, changed_predictions):
    """Recompute diagnostics independently of stored reports, without training."""
    from .learned_partition_inference import intervention_inputs, intervention_report
    if not rows or len(rows) != len(original_predictions) or len(rows) != len(changed_predictions):
        raise ValueError("support replay requires the complete aligned scene roster")
    reports = [intervention_report(row, model_inputs(row, normalizer, arm),
        intervention_inputs(row, normalizer, arm, intervention), original, changed)
        for row, original, changed in zip(rows, original_predictions, changed_predictions)]
    # Candidate signatures are tuples in memory and lists in the JSON artifact.
    return json.loads(json.dumps(reports, allow_nan=False))


def summarize_support(records):
    expected = [r for r in inference_roster() if r["intervention"] != "original"]
    if (not isinstance(records, list) or len(records) != len(expected)
            or any(set(r) != {*e, "scenes"} or any(r[k] != v for k, v in e.items())
                   or not isinstance(r["scenes"], list) or len(r["scenes"]) != 512 for r, e in zip(records, expected))):
        raise ValueError("support requires every declared scene/head intervention, without exclusion")
    groups = {}
    for record in records:
        key = record["arm"]+"/"+record["intervention"]
        groups.setdefault(key, {})[record["checkpoint_seed"], record["reader_seed"]] = record["scenes"]
    result = {}
    for key, cases in groups.items():
        input_count, changed_predictions, changed_decisions, constant, magnitudes = 0, 0, 0, 0, []
        coverage = []
        for i in range(512):
            supported_inputs, supported_heads = 0, 0
            for seed in SEEDS:
                inputs = [cases[seed, reader][i]["input"] for reader in READER_SEEDS]
                if any(value != inputs[0] for value in inputs[1:]):
                    raise ValueError("input support cannot change across reader initialization")
                supported = inputs[0]["status"] == "INPUT_CHANGED"
                if inputs[0]["status"] not in ("INPUT_CHANGED", "INPUT_UNCHANGED"):
                    raise ValueError("unknown observable support status")
                supported_inputs += supported
                for reader in READER_SEEDS:
                    prediction = cases[seed, reader][i]["prediction"]
                    magnitude = prediction["maximum_absolute_component_delta"]
                    changed_predictions += magnitude > 0
                    changed_decisions += prediction["decision_changed"]
                    constant += prediction["constant_sum_delta"]
                    magnitudes.append(magnitude)
                    supported_heads += supported
            input_count += supported_inputs
            coverage.append({"scene_id": i, "input_supported": supported_inputs, "input_denominator": 3,
                             "trained_supported": supported_heads, "trained_denominator": 9})
        result[key] = {"input_cases": 1536, "trained_cases": 4608, "input_changed_cases": input_count,
            "component_changed_cases": int(changed_predictions), "decision_changed_cases": int(changed_decisions),
            "constant_sum_delta_cases": int(constant), "maximum_absolute_component_delta": float(max(magnitudes)),
            "mean_maximum_absolute_component_delta": float(np.mean(magnitudes)),
            "coverage_by_scene": coverage, "excluded_from_primary_metrics": 0}
    return result
