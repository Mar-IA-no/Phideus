"""Observable shard execution for the learned partition-reader campaign.

No supervision parsing in forward or scoring. Training and evaluation use
separate downstream ports after these raw outputs have been sealed.
"""
from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np

from . import learned_partition_gate as gate
from . import learned_partition_provenance as p
from .learned_partition_cache import load_rows, save_rows
from .learned_partition_core import observable_features, fit_normalizer, model_inputs, ARMS
from .learned_partition_inputs import pack_inputs, read_inputs
from .learned_partition_data import ObservationShard, SHARD_SIZE, _bundle, scene_ids, load_supervision
from .learned_partition_metrics import SPLITS, SEEDS, candidate_targets
from .source_artifacts import load_ordered_logits
from .source_coherence import GroupFitCache, SourceFitter
from .structured_source_artifacts import mark_failure, seal_bundle, write_json, write_npz
from .structured_source_data import cpu_resources
from .structured_source_reader import score_scene
from .structured_source_metrics import evaluate_scene
from .learned_partition_validation import boundary, memoized


@boundary
def stage_inputs(authorization, split, shard, data):
    auth = gate.verify_authorization(authorization, split)
    cache = ObservationShard(data, split, shard, auth["common"])
    binding = cache.manifest["binding"]
    if binding.get("authorization") != authorization:
        raise ValueError("observations were produced under another authorization")
    after, prior = gate.verify_data_stage(authorization, split, shard, binding["previous"], binding["earlier_shards"])
    if after != auth or cache.fingerprints & prior:
        raise ValueError("observation authorization changed or overlaps previous corpus")
    return auth, cache


def _identity(common, authorization, data, split, shard):
    return {"common": common, "authorization": authorization, "data": data, "split": split,
            "split_seed": SPLITS[split][1], "shard": shard, "scene_ids": scene_ids(split, shard), "count": SHARD_SIZE}


def ordered_forward(ref, cache, common, *, authorization, data):
    root, m = _bundle(ref, "learned_logits_shard", common)
    b = m["binding"]
    expected = _identity(common, authorization, data, cache.split, cache.shard)
    if set(b) != set(expected)|{"gpu_grant"} or any(b[k] != v for k, v in expected.items()):
        raise ValueError("forward identity or data source differs")
    if set(m["artifacts_sha256"]) != {"forward.json", *[f"seed_{s}.npz" for s in SEEDS]}:
        raise ValueError("forward inventory differs")
    metadata = json.loads((root/"forward.json").read_bytes())
    expected_rows = [{"seed": row["seed"], "checkpoint": row["checkpoint"], "path": f"seed_{row['seed']}.npz",
                      "count": SHARD_SIZE} for row in common["checkpoints"]]
    if metadata["rows"] != expected_rows:
        raise ValueError("forward checkpoint roster differs")
    return {s: load_ordered_logits(root/f"seed_{s}.npz", cache.observations) for s in SEEDS}


def forward_shard(output, split, shard, *, authorization, data, gpu_grant):
    started = time.monotonic()
    auth, cache = stage_inputs(authorization, split, shard, data)
    from .structured_source_profile import gpu_lease
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    try:
        with gpu_lease(gpu_grant) as availability:
            from .structured_source_runner import checkpoint_forward, gpu_runtime
            from .partial_compatibility_inference import save_logits
            import torch
            runtime = gpu_runtime()
            rows = []
            for checkpoint in auth["common"]["checkpoints"]:
                if time.monotonic()-started > 600:
                    raise TimeoutError("forward shard budget exhausted")
                matrices = checkpoint_forward(checkpoint, cache.records, runtime)
                path = f"seed_{checkpoint['seed']}.npz"
                save_logits(output/path, matrices, cache.observations)
                rows.append({"seed": checkpoint["seed"], "checkpoint": checkpoint["checkpoint"],
                             "path": path, "count": SHARD_SIZE})
                del matrices
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_reserved(0)
        write_json(output/"forward.json", {"rows": rows, "runtime": runtime})
        if stage_inputs(authorization, split, shard, data)[0] != auth:
            raise ValueError("forward authorization changed")
        seconds = time.monotonic()-started
        if seconds > 600 or peak >= 2*1024**3:
            raise RuntimeError("forward exceeds wall time or VRAM envelope")
        binding = {**_identity(auth["common"], authorization, data, split, shard), "gpu_grant": gpu_grant}
        seal_bundle(output, role="learned_logits_shard", binding=binding,
                    resources={"seconds": seconds, "peak_reserved_bytes": peak, "runtime": runtime,
                               "availability": availability})
        ref = p.reference(output/"manifest.json")
        ordered_forward(ref, cache, auth["common"], authorization=authorization, data=data)
        return ref
    except BaseException as exc:
        mark_failure(output, exc)
        raise


def score_shard(output, split, shard, *, authorization, data, logits):
    started = time.monotonic()
    auth, cache = stage_inputs(authorization, split, shard, data)
    raw = ordered_forward(logits, cache, auth["common"], authorization=authorization, data=data)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    try:
        fitter = SourceFitter()
        for seed in SEEDS:
            (output/f"seed_{seed}").mkdir()
        index = []
        for position, (obs, record) in enumerate(zip(cache.observations, cache.records)):
            fit_cache = GroupFitCache(fitter)
            i = obs["scene_id"]
            for seed in SEEDS:
                z = raw[seed][position]
                scored = score_scene(np.asarray(obs["log_f"], np.float32), z, record["pair_support"],
                    record["triples"], record["residual_cents"], split_seed=obs["split_seed"], scene_id=i, fit_cache=fit_cache)
                row = observable_features(scored, z)
                prefix = f"seed_{seed}/{i:05d}"
                write_json(output/f"{prefix}_pool.json", scored)
                save_rows(output/f"{prefix}_rows.npz", row)
                index.append({"scene_id": i, "checkpoint_seed": seed,
                              "pool": f"{prefix}_pool.json", "rows": f"{prefix}_rows.npz"})
            cpu_resources(started)
        write_json(output/"index.json", index)
        if stage_inputs(authorization, split, shard, data)[0] != auth:
            raise ValueError("scoring authorization changed")
        ordered_forward(logits, cache, auth["common"], authorization=authorization, data=data)
        binding = {**_identity(auth["common"], authorization, data, split, shard), "logits": logits}
        seal_bundle(output, role="learned_scored_shard", binding=binding, resources=cpu_resources(started))
        ref = p.reference(output/"manifest.json")
        scored_shard(ref, cache, auth["common"], authorization=authorization, data=data, logits=logits)
        return ref
    except BaseException as exc:
        mark_failure(output, exc)
        raise


def scored_shard(ref, cache, common, *, authorization, data, logits):
    root, m = _bundle(ref, "learned_scored_shard", common)
    if m["binding"] != {**_identity(common, authorization, data, cache.split, cache.shard), "logits": logits}:
        raise ValueError("scored shard has different raw data, logits or authorization")
    expected = [{"scene_id": i, "checkpoint_seed": s, "pool": f"seed_{s}/{i:05d}_pool.json",
                 "rows": f"seed_{s}/{i:05d}_rows.npz"} for i in cache.scene_ids for s in SEEDS]
    index = json.loads((root/"index.json").read_bytes())
    if (index != expected or set(m["artifacts_sha256"]) !=
            {"index.json", *[r[k] for r in expected for k in ("pool", "rows")]}):
        raise ValueError("scored candidate or feature inventory differs")
    result = {s: [] for s in SEEDS}
    for r in index:
        scored = json.loads((root/r["pool"]).read_bytes())
        rows = load_rows(root/r["rows"])
        if rows.candidates != tuple(tuple(tuple(g) for g in c["signature"]) for c in scored["candidates"]):
            raise ValueError("scored pool and cached feature candidates differ")
        result[r["checkpoint_seed"]].append((scored, rows))
    return result


def supervised_targets_shard(output, split, shard, *, authorization, data, logits, scored):
    """Train/calibration-only target port; test truth requires sealed inference."""
    if split not in ("train", "calibration"):
        raise PermissionError("training target producer cannot open a test sidecar")
    started = time.monotonic()
    auth, cache = stage_inputs(authorization, split, shard, data)
    raw = ordered_forward(logits, cache, auth["common"], authorization=authorization, data=data)
    rows = scored_shard(scored, cache, auth["common"], authorization=authorization, data=data, logits=logits)
    truths = load_supervision(cache)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    try:
        for seed in SEEDS:
            (output/f"seed_{seed}").mkdir()
        for position, truth in enumerate(truths):
            i = cache.scene_ids[position]
            for seed in SEEDS:
                pool, _ = rows[seed][position]
                values = candidate_targets(pool, truth["source_ids"])
                write_npz(output/f"seed_{seed}/{i:05d}_targets.npz", **values)
                metrics = evaluate_scene(pool, truth["source_ids"], raw[seed][position],
                    {SEEDS[0]: .55, SEEDS[1]: .65, SEEDS[2]: .6}[seed])
                write_json(output/f"seed_{seed}/{i:05d}_metrics.json", metrics)
            cpu_resources(started)
        if stage_inputs(authorization, split, shard, data)[0] != auth:
            raise ValueError("target authorization changed")
        load_supervision(cache)
        ordered_forward(logits, cache, auth["common"], authorization=authorization, data=data)
        scored_shard(scored, cache, auth["common"], authorization=authorization, data=data, logits=logits)
        binding = {**_identity(auth["common"], authorization, data, split, shard), "logits": logits, "scored": scored}
        seal_bundle(output, role="learned_supervised_targets_shard", binding=binding, resources=cpu_resources(started))
        return p.reference(output/"manifest.json")
    except BaseException as exc:
        mark_failure(output, exc)
        raise


def targets_shard(ref, cache, common, *, authorization, data, logits, scored, rows):
    if cache.split not in ("train", "calibration"):
        raise PermissionError("training target loader cannot read test truth")
    root, m = _bundle(ref, "learned_supervised_targets_shard", common)
    if m["binding"] != {**_identity(common, authorization, data, cache.split, cache.shard), "logits": logits, "scored": scored}:
        raise ValueError("target provenance differs")
    expected = {f"seed_{s}/{i:05d}_{suffix}" for s in SEEDS for i in cache.scene_ids
                for suffix in ("targets.npz", "metrics.json")}
    if set(m["artifacts_sha256"]) != expected:
        raise ValueError("supervised inventory differs")
    result = {s: [] for s in SEEDS}
    for s in SEEDS:
        for position, i in enumerate(cache.scene_ids):
            n, candidates = rows[s][position][1].n, rows[s][position][1].candidates
            with np.load(root/f"seed_{s}/{i:05d}_targets.npz", allow_pickle=False) as arrays:
                if set(arrays.files) != {"raw", "targets"}:
                    raise ValueError("target array schema differs")
                raw, target = arrays["raw"], arrays["targets"]
            if (raw.shape != (len(candidates), 2) or raw.dtype != np.float64 or target.dtype != np.float32
                    or not np.isfinite(raw).all() or np.any(raw < 0)
                    or not np.array_equal(target, (raw/np.log(n)).astype(np.float32))
                    or np.any(target > 1)):
                raise ValueError("raw and normalized targets differ")
            metrics = json.loads((root/f"seed_{s}/{i:05d}_metrics.json").read_bytes())
            if len(metrics["candidate_metrics"]) != len(candidates):
                raise ValueError("metric candidate roster differs")
            result[s].append({"raw": raw, "targets": target, "metrics": metrics})
    return result


@boundary
@memoized
def training_corpus(ref, split, *, authorization):
    """Validate the complete train/calibration dependency chain, not test data."""
    if split not in ("train", "calibration"):
        raise PermissionError("learned training corpus cannot contain a test split")
    auth = gate.verify_authorization(authorization, split)
    root, manifest = _bundle(ref, "learned_training_corpus", auth["common"])
    b = manifest["binding"]
    if (set(b) != {"common", "authorization", "split", "data"} or b["authorization"] != authorization
            or b["split"] != split or set(manifest["artifacts_sha256"]) != {"shards.json"}):
        raise ValueError("training corpus identity or inventory differs")
    data_root, data_manifest = _bundle(b["data"], "learned_observation_split", auth["common"])
    gate.split_fingerprints(b["data"], split, auth["common"], authorization=authorization,
                           previous=data_manifest["binding"]["previous"])
    data_shards = json.loads((data_root/"shards.json").read_bytes())
    shards = json.loads((root/"shards.json").read_bytes())
    if not isinstance(shards, list) or len(shards) != SPLITS[split][0]//SHARD_SIZE:
        raise ValueError("incomplete corpus shard roster")
    for i, entry in enumerate(shards):
        if set(entry) != {"data", "logits", "scored", "targets"} or entry["data"] != data_shards[i]:
            raise ValueError("corpus data shard differs from the complete aggregate")
        _, cache = stage_inputs(authorization, split, i, entry["data"])
        ordered_forward(entry["logits"], cache, auth["common"], authorization=authorization, data=entry["data"])
        rows = scored_shard(entry["scored"], cache, auth["common"], authorization=authorization,
                            data=entry["data"], logits=entry["logits"])
        targets_shard(entry["targets"], cache, auth["common"], authorization=authorization,
                       data=entry["data"], logits=entry["logits"], scored=entry["scored"], rows=rows)
    return auth, manifest, shards


def aggregate_training_corpus(output, split, *, authorization, data, shards):
    if split not in ("train", "calibration"):
        raise PermissionError("no tests in training corpus")
    auth = gate.verify_authorization(authorization, split)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    try:
        write_json(output/"shards.json", shards)
        seal_bundle(output, role="learned_training_corpus", binding={"common": auth["common"],
            "authorization": authorization, "split": split, "data": data}, resources={"operation": "aggregate_no_draws"})
        ref = p.reference(output/"manifest.json")
        training_corpus(ref, split, authorization=authorization)
        return ref
    except BaseException as exc:
        mark_failure(output, exc)
        raise


class _DiskRows:
    """Bounded-memory, ordered repeated passes for the exact normalizer kernel."""
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __iter__(self):
        return (load_rows(path) for path in self.paths)


def fit_train_normalizers(output, *, authorization, train):
    started = time.monotonic()
    auth, _, shards = training_corpus(train, "train", authorization=authorization)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    try:
        for seed in SEEDS:
            paths = [p.verify_reference(shard["scored"]).parent/f"seed_{seed}/{i:05d}_rows.npz"
                     for index, shard in enumerate(shards) for i in scene_ids("train", index)]
            normalizer = fit_normalizer(_DiskRows(paths), expected_count=4096)
            write_npz(output/f"seed_{seed}.npz", **normalizer)
            cpu_resources(started)
        if training_corpus(train, "train", authorization=authorization)[0] != auth:
            raise ValueError("normalizer training provenance changed")
        seal_bundle(output, role="learned_train_normalizers", binding={"common": auth["common"],
            "authorization": authorization, "train": train, "count": 4096}, resources=cpu_resources(started))
        return p.reference(output/"manifest.json")
    except BaseException as exc:
        mark_failure(output, exc)
        raise


def read_normalizers(ref, common, *, authorization, train):
    root, manifest = _bundle(ref, "learned_train_normalizers", common)
    if (manifest["binding"] != {"common": common, "authorization": authorization, "train": train, "count": 4096}
            or set(manifest["artifacts_sha256"]) != {f"seed_{s}.npz" for s in SEEDS}):
        raise ValueError("normalizer is not bound to the complete train corpus")
    result = {}
    for seed in SEEDS:
        with np.load(root/f"seed_{seed}.npz", allow_pickle=False) as arrays:
            if set(arrays.files) != {"mean", "scale", "zero_variance", "scene_count"}:
                raise ValueError("normalizer array inventory differs")
            values = {k: arrays[k] for k in arrays.files}
        mean, scale, zero, count = (values[k] for k in ("mean", "scale", "zero_variance", "scene_count"))
        if (mean.shape != (5,) or scale.shape != (5,) or zero.shape != (5,)
                or mean.dtype != np.float64 or scale.dtype != np.float64 or zero.dtype != np.bool_
                or count.shape != () or count.dtype != np.int64 or count.item() != 4096
                or not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0)
                or np.any(scale[zero] != 1.)):
            raise ValueError("invalid train-only normalizer values")
        result[seed] = values
    return result


def normalize_shard(output, split, shard, *, authorization, data, logits, scored, normalizers, train):
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
        if stage_inputs(authorization, split, shard, data)[0] != auth:
            raise ValueError("normalization authorization changed")
        read_normalizers(normalizers, auth["common"], authorization=training_auth, train=train)
        ordered_forward(logits, cache, auth["common"], authorization=authorization, data=data)
        scored_shard(scored, cache, auth["common"], authorization=authorization, data=data, logits=logits)
        seal_bundle(output, role="learned_normalized_shard", binding={**_identity(auth["common"], authorization, data, split, shard),
            "logits": logits, "scored": scored, "normalizers": normalizers, "train": train}, resources=cpu_resources(started))
        return p.reference(output/"manifest.json")
    except BaseException as exc:
        mark_failure(output, exc)
        raise


def normalized_shard(ref, cache, common, *, authorization, data, logits, scored, normalizers, train):
    root, manifest = _bundle(ref, "learned_normalized_shard", common)
    if manifest["binding"] != {**_identity(common, authorization, data, cache.split, cache.shard),
            "logits": logits, "scored": scored, "normalizers": normalizers, "train": train}:
        raise ValueError("normalized inputs have different source identities")
    files = {f"seed_{seed}/{arm}.npz" for seed in SEEDS for arm in ARMS}
    if set(manifest["artifacts_sha256"]) != files:
        raise ValueError("normalized input roster differs")
    return root
