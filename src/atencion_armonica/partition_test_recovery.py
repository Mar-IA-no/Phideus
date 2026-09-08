"""Versioned test-only execution; scientific helpers and selected weights stay 06.

Copied traversals deliberately retain their original math and check order.
The only lifetime changes release completed objects before revalidation.
"""
from __future__ import annotations

from contextlib import nullcontext
import json
from pathlib import Path
import time
import numpy as np

from . import learned_partition_provenance as p
from . import learned_partition_runner as runner
from . import partition_test_recovery_gate as rg
from .learned_partition_core import ARMS, READER_SEEDS, model_inputs
from .learned_partition_data import load_supervision
from .learned_partition_metrics import (SEEDS, TESTS, SPLITS, METRICS, REFERENCES,
    evaluate_preserved_predictions, summarize_test, bootstrap_indices)
from .learned_partition_runner import (stage_inputs, training_corpus, read_normalizers,
    ordered_forward, scored_shard, pack_inputs, cpu_resources, _identity)
from .learned_partition_test import (inference_roster, prefix, write_predictions,
    read_predictions, intervention_metrics_for_scene, summarize_interventions,
    replay_support, summarize_support)
from .structured_source_artifacts import mark_failure, seal_bundle, write_json, write_npz
from .learned_partition_validation import boundary


def recovery_entry(recovery_authorization, authorization):
    execution = rg.execution_binding(recovery_authorization)
    if authorization != rg.FROZEN["test_authorization"]:
        raise PermissionError("recovery accepts only the original frozen test authorization")
    return execution


def recovery_identity(execution, authorization, data, split, shard):
    identity = _identity(execution["base_common"], authorization, data, split, shard)
    del identity["common"]
    return {**execution, **identity}


def compare_npz_exact(left, right):
    with np.load(left, allow_pickle=False) as a, np.load(right, allow_pickle=False) as b:
        if set(a.files) != set(b.files):
            raise ValueError("normalization array inventory differs")
        for key in a.files:
            x, y = a[key], b[key]
            if x.dtype != y.dtype or not np.array_equal(x, y):
                raise ValueError("normalized arrays differ from the failed candidate control")


def compare_failed_iid(output, split):
    if split != "iid":
        return
    candidates = rg.failed_evidence()
    for ref in candidates:
        original = rg.hash_only(ref)
        relative = original.relative_to(p.ROOT/rg.TREE/"iid/shard_00/normalized")
        compare_npz_exact(output/relative, original)
    rg.failed_evidence()


def normalize_shard(output, split, shard, *, authorization, data, logits, scored, normalizers, train, recovery_authorization):
    execution = recovery_entry(recovery_authorization, authorization)
    started = time.monotonic()
    auth, cache = stage_inputs(authorization, split, shard, data)
    # For test data, the training authorization is embedded in the audited freeze.
    training_auth = authorization if split in ("train", "calibration") else p.read_reference(auth["freeze"])["data_authorization"]
    training_corpus(train, "train", authorization=training_auth)
    norms = read_normalizers(normalizers, auth["common"], authorization=training_auth, train=train)
    ordered_forward(logits, cache, auth["common"], authorization=authorization, data=data)
    rows = scored_shard(scored, cache, auth["common"], authorization=authorization, data=data, logits=logits)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    try:
        for seed in SEEDS:
            (output/f"seed_{seed}").mkdir()
            for arm in ARMS:
                inputs = [model_inputs(row, norms[seed], arm) for _, row in rows[seed]]
                pack_inputs(output/f"seed_{seed}/{arm}.npz", inputs,
                            scene_ids=cache.scene_ids, dim=8 if arm == ARMS[0] else 9)
                cpu_resources(started)
        del rows, inputs
        if stage_inputs(authorization, split, shard, data)[0] != auth:
            raise ValueError("normalization authorization changed")
        read_normalizers(normalizers, auth["common"], authorization=training_auth, train=train)
        ordered_forward(logits, cache, auth["common"], authorization=authorization, data=data)
        scored_shard(scored, cache, auth["common"], authorization=authorization, data=data, logits=logits)
        compare_failed_iid(output, split)
        rg.assert_fresh(recovery_authorization, execution)
        seal_bundle(output, role=rg.ROLES["normalized"], binding={**recovery_identity(execution, authorization, data, split, shard),
            "logits": logits, "scored": scored, "normalizers": normalizers, "train": train}, resources=cpu_resources(started))
        ref = p.reference(output/"manifest.json")
        normalized_shard(ref, cache, auth["common"], authorization=authorization, data=data,
            logits=logits, scored=scored, normalizers=normalizers, train=train,
            recovery_authorization=recovery_authorization)
        return ref
    except BaseException as exc:
        mark_failure(output, exc)
        raise

def normalized_shard(ref, cache, common, *, authorization, data, logits, scored, normalizers, train, recovery_authorization):
    execution = recovery_entry(recovery_authorization, authorization)
    if execution["base_common"] != common:
        raise ValueError("normalized scientific base differs")
    root, manifest = rg.checked_bundle(ref, "normalized", recovery_authorization)
    if manifest["binding"] != {**recovery_identity(execution, authorization, data, cache.split, cache.shard),
            "logits": logits, "scored": scored, "normalizers": normalizers, "train": train}:
        raise ValueError("normalized inputs have different source identities")
    files = {f"seed_{seed}/{arm}.npz" for seed in SEEDS for arm in ARMS}
    if set(manifest["artifacts_sha256"]) != files:
        raise ValueError("normalized input roster differs")
    return root

@boundary
def test_inputs(split, *, authorization, data, logits, scored, normalized, recovery_authorization):
    recovery_entry(recovery_authorization, authorization)
    if split not in TESTS:
        raise PermissionError("test inference requires a declared fresh test role")
    auth, cache = runner.stage_inputs(authorization, split, 0, data)
    frozen = p.read_reference(auth["freeze"])
    raw = runner.ordered_forward(logits, cache, auth["common"], authorization=authorization, data=data)
    rows = runner.scored_shard(scored, cache, auth["common"], authorization=authorization, data=data, logits=logits)
    normal_root = normalized_shard(normalized, cache, auth["common"], authorization=authorization,
        data=data, logits=logits, scored=scored, normalizers=frozen["normalizers"], train=frozen["train"],
        recovery_authorization=recovery_authorization)
    norms = runner.read_normalizers(frozen["normalizers"], auth["common"],
                                   authorization=frozen["data_authorization"], train=frozen["train"])
    return auth, frozen, cache, raw, rows, normal_root, norms

def infer_test(output, split, *, authorization, data, logits, scored, normalized, gpu_grant, recovery_authorization):
    execution = recovery_entry(recovery_authorization, authorization)
    started = time.monotonic()
    kwargs = dict(authorization=authorization, data=data, logits=logits, scored=scored, normalized=normalized,
                  recovery_authorization=recovery_authorization)
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
                if time.monotonic()-started > 1200:
                    raise TimeoutError("test inference exceeded its forward envelope")
            peak = torch.cuda.max_memory_reserved(0) if device == "cuda:0" else 0
            if peak >= 2*1024**3:
                raise RuntimeError("test inference exceeded its VRAM envelope")
        write_json(output/"index.json", inference_roster())
        del rows, norms, original, inputs, row, expected, predictions, changed, altered, support
        if test_inputs(split, **kwargs)[0] != auth:
            raise ValueError("test inference provenance changed")
        rg.assert_fresh(recovery_authorization, execution)
        seal_bundle(output, role=rg.ROLES["inference"], binding={**execution, "split": split,
            **kwargs, "freeze": auth["freeze"], "gpu_grant": gpu_grant},
            resources={"seconds": time.monotonic()-started, "peak_reserved_bytes": peak, "availability": availability})
        ref = p.reference(output/"manifest.json")
        rg.checked_bundle(ref, "inference", recovery_authorization)
        return ref
    except BaseException as exc:
        mark_failure(output, exc)
        raise

@boundary
def preserved_test(ref, split, *, authorization, data, logits, scored, normalized, recovery_authorization):
    execution = recovery_entry(recovery_authorization, authorization)
    kwargs = dict(authorization=authorization, data=data, logits=logits, scored=scored, normalized=normalized,
                  recovery_authorization=recovery_authorization)
    auth, frozen, cache, raw, rows, _, _ = test_inputs(split, **kwargs)
    root, m = rg.checked_bundle(ref, "inference", recovery_authorization)
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

def evaluate_test(output, split, *, authorization, data, logits, scored, normalized, predictions, recovery_authorization):
    """Identical primary/replay entrypoint; no new forward, training or observations."""
    execution = recovery_entry(recovery_authorization, authorization)
    started = time.monotonic()
    kwargs = dict(authorization=authorization, data=data, logits=logits, scored=scored, normalized=normalized,
                  recovery_authorization=recovery_authorization)
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
        rg.assert_fresh(recovery_authorization, execution)
        seal_bundle(output, role=rg.ROLES["evaluation"], binding={**execution, "split": split,
            **kwargs, "predictions": predictions, "freeze": auth["freeze"]}, resources={"seconds": time.monotonic()-started})
        ref = p.reference(output/"manifest.json")
        rg.checked_evaluation(ref, split, **kwargs, predictions=predictions)
        return ref
    except BaseException as exc:
        mark_failure(output, exc)
        raise
