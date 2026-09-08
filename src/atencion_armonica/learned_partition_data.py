"""Fixed 512-scene shards with separate observable and supervision ports.

The prospective producer is reachable only after the campaign gate validates
authorization. Pure helpers support mechanical checks with previously seen
observations; they are not permission to materialize campaign data.
"""
from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np

from . import learned_partition_provenance as provenance
from .learned_partition_metrics import SPLITS
from .partial_compatibility_cache import encoded, feature_record, observation_fingerprint
from .shared_partial_data import _observe
from .structured_source_artifacts import mark_failure, seal_bundle, verify_bundle, write_json, write_npz
from .structured_source_data import cpu_resources, validate_record

SHARD_SIZE = 512
ROLES = tuple(SPLITS)


def scene_ids(split, shard):
    if split not in SPLITS or type(shard) is not int or not 0 <= shard < SPLITS[split][0]//SHARD_SIZE:
        raise ValueError("split or shard outside the fixed prospective roster")
    return list(range(shard*SHARD_SIZE, (shard+1)*SHARD_SIZE))


def _draw_scene(split, scene_id, seed):
    """Copied historical law and draw order; new counts, no historical mutation."""
    if (split not in SPLITS or type(scene_id) is not int or not 0 <= scene_id < SPLITS[split][0]
            or type(seed) is not int or seed < 0):
        raise ValueError("invalid scene or seed")
    rng = np.random.default_rng(np.random.SeedSequence([seed, scene_id]))
    k = 4 if split == "ood_polyphony" else int(rng.integers(2, 4))
    lo, hi = (3e-3, 1e-2) if split == "ood_beta" else (1e-4, 1e-3)
    sources = []
    for _ in range(k):
        f0 = float(np.exp(rng.uniform(np.log(100.), np.log(500.))))
        beta = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        size = int(rng.integers(4, 9))
        indices = sorted(rng.choice(np.arange(1, 9), size, replace=False).tolist())
        gamma = (float(np.exp(rng.uniform(np.log(5e-6), np.log(5e-5)))) if split == "deformed_family" else 0.)
        sources.append({"f0": f0, "beta": beta, "gamma": gamma, "indices": indices})
    return _observe(sources, rng, 2., scene_id, seed)


def validate_observation(obs, scene_id, split):
    if (split not in SPLITS or set(obs) != {"scene_id", "split_seed", "log_f"}
            or type(scene_id) is not int or not 0 <= scene_id < SPLITS[split][0]
            or type(obs["scene_id"]) is not int or obs["scene_id"] != scene_id
            or type(obs["split_seed"]) is not int or obs["split_seed"] != SPLITS[split][1]):
        raise ValueError("observation schema or split identity differs")
    original = np.asarray(obs["log_f"])
    q = original.astype(np.float32)
    if (q.ndim != 1 or not 8 <= len(q) <= 32 or not np.isfinite(q).all()
            or not np.array_equal(original, q.astype(np.float64))):
        raise ValueError("observation is not an exact finite q32 vector")
    return q


from .learned_partition_validation import memoized, claim_reference, fresh_pass


def _bundle(ref, role, common):
    claim_reference(ref, role=role, binding=common)
    return _verified_bundle(ref, role, common)


@memoized
def _verified_bundle(ref, role, common):
    path = provenance.verify_reference(ref)
    if path.name != "manifest.json":
        raise ValueError("expected bundle manifest reference")
    manifest = verify_bundle(path.parent, ref["sha256"], role=role)
    if manifest["binding"].get("common") != common:
        raise ValueError("bundle common binding differs")
    return path.parent, manifest


class ObservationShard:
    """Hash sidecar bytes for integrity, but never parse labels in this port."""
    def __init__(self, ref, split, shard, common):
        ids = scene_ids(split, shard)
        self.root, self.manifest = _bundle(ref, "learned_observation_shard", common)
        b = self.manifest["binding"]
        expected = {"split": split, "split_seed": SPLITS[split][1], "shard": shard,
                    "count": SHARD_SIZE, "split_count": SPLITS[split][0], "scene_ids": ids}
        if any(b.get(k) != v for k, v in expected.items()):
            raise ValueError("shard is not the requested complete ordered role")
        files = {"observations.jsonl", "sidecars.jsonl", "rows.json", "deduplication.json"}
        files.update(f"features/{i:05d}.npz" for i in ids)
        if set(self.manifest["artifacts_sha256"]) != files:
            raise ValueError("shard scientific inventory differs")
        self.observations = [json.loads(line) for line in (self.root/"observations.jsonl").read_bytes().splitlines()]
        rows = json.loads((self.root/"rows.json").read_bytes())
        if len(rows) != SHARD_SIZE or len(self.observations) != SHARD_SIZE:
            raise ValueError("shard observation roster is incomplete")
        self.records, self.fingerprints = [], set()
        for i, obs, row in zip(ids, self.observations, rows):
            q = validate_observation(obs, i, split)
            fp = observation_fingerprint(obs)
            if row != {"scene_id": i, "n": len(q), "path": f"features/{i:05d}.npz", "fingerprint": fp}:
                raise ValueError("shard feature row has a different observation identity")
            if fp in self.fingerprints:
                raise ValueError("duplicate observation; no replacement")
            self.fingerprints.add(fp)
            with np.load(self.root/row["path"], allow_pickle=False) as raw:
                record = {k: raw[k] for k in raw.files}
            validate_record(record, q)
            self.records.append(record)
        self.split, self.shard, self.scene_ids = split, shard, ids
        self.reference, self.common = ref, common


def load_supervision(cache):
    """Explicit train/metric-only port; validate reconstruction and source law."""
    root, manifest = fresh_pass(_bundle)(cache.reference, "learned_observation_shard", cache.common)
    if root != cache.root or manifest != cache.manifest:
        raise ValueError("supervision cache changed since observable loading")
    truths = [json.loads(line) for line in (cache.root/"sidecars.jsonl").read_bytes().splitlines()]
    _validate_supervision(cache, truths)
    after_root, after_manifest = fresh_pass(_bundle)(cache.reference, "learned_observation_shard", cache.common)
    if after_root != root or after_manifest != manifest:
        raise ValueError("supervision bundle changed during parsing")
    return truths


def _validate_supervision(cache, truths):
    """Pure semantic checks; not permission to consume an unsealed sidecar."""
    if len(truths) != len(cache.observations):
        raise ValueError("supervision roster differs")
    keys = {"scene_id", "split_seed", "sources", "sigma_cents", "mean_log_f_observed", "log_f_ideal",
            "sensor_log_noise", "source_ids", "partial_indices", "permutation"}
    for obs, truth in zip(cache.observations, truths):
        n = len(obs["log_f"])
        if (set(truth) != keys or type(truth["scene_id"]) is not int or truth["scene_id"] != obs["scene_id"]
                or type(truth["split_seed"]) is not int or truth["split_seed"] != obs["split_seed"]
                or truth["sigma_cents"] != 2.):
            raise ValueError("supervision identity or schema differs")
        labels, indices, permutation = [truth[k] for k in ("source_ids", "partial_indices", "permutation")]
        allowed_counts = (4,) if cache.split == "ood_polyphony" else (2, 3)
        if (any(len(v) != n or any(type(x) is not int for x in v) for v in (labels, indices, permutation))
                or len(truth["sources"]) not in allowed_counts
                or sorted(permutation) != list(range(n))
                or sorted(set(labels)) != list(range(len(truth["sources"])))
                or any(not 1 <= x <= 8 for x in indices)):
            raise ValueError("supervision membership or permutation differs")
        ideal, source_labels, source_indices = [], [], []
        lo, hi = (3e-3, 1e-2) if cache.split == "ood_beta" else (1e-4, 1e-3)
        for sid, source in enumerate(truth["sources"]):
            if (set(source) != {"f0", "beta", "gamma", "indices"}
                    or any(type(source[k]) not in (int, float) or not np.isfinite(source[k]) for k in ("f0", "beta", "gamma"))
                    or not 100 <= source["f0"] <= 500 or not lo <= source["beta"] <= hi
                    or not (5e-6 <= source["gamma"] <= 5e-5 if cache.split == "deformed_family" else source["gamma"] == 0)
                    or not 4 <= len(source["indices"]) <= 8
                    or any(type(x) is not int or not 1 <= x <= 8 for x in source["indices"])
                    or source["indices"] != sorted(set(source["indices"]))):
                raise ValueError("source law schema differs")
            ns = np.asarray(source["indices"], np.float64)
            ideal.extend((np.log(source["f0"])+np.log(ns)+.5*np.log1p(source["beta"]*ns**2+source["gamma"]*ns**4)).tolist())
            source_labels.extend([sid]*len(ns))
            source_indices.extend(source["indices"])
        observed_ideal = np.asarray(truth["log_f_ideal"], np.float64)
        noise = np.asarray(truth["sensor_log_noise"], np.float64)
        if (len(ideal) != n or observed_ideal.shape != (n,) or noise.shape != (n,)
                or not np.isfinite(noise).all() or not np.isfinite(truth["mean_log_f_observed"])
                or not np.array_equal(np.asarray(ideal)[permutation], observed_ideal)
                or not np.array_equal(np.asarray(source_labels)[permutation], labels)
                or not np.array_equal(np.asarray(source_indices)[permutation], indices)
                or float((observed_ideal+noise)[np.argsort(permutation)].mean(dtype=np.float64)) != truth["mean_log_f_observed"]
                or not np.array_equal((observed_ideal+noise-truth["mean_log_f_observed"]).astype(np.float32),
                                      np.asarray(obs["log_f"], np.float32))):
            raise ValueError("sidecar does not reconstruct its source law and delivered observation")
        expected_obs, expected_truth = _draw_scene(cache.split, obs["scene_id"], obs["split_seed"])
        if expected_obs != obs or expected_truth != truth:
            raise ValueError("sidecar is not the exact seeded draw belonging to this observation")
    return truths


def prepare_shard(output, split, shard, *, authorization, previous, earlier_shards):
    # Import deferred: incomplete campaign integration must fail before draws.
    from .learned_partition_gate import verify_data_stage
    started = time.monotonic()
    auth, prior = verify_data_stage(authorization, split, shard, previous, earlier_shards)
    ids = scene_ids(split, shard)
    cpu_resources(started)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    try:
        (output/"features").mkdir()
        write_json(output/"deduplication.json", {"prior_fingerprints": sorted(prior), "policy": "reject_no_replacement"})
        seen, rows = set(prior), []
        with (output/"observations.jsonl").open("xb") as observed, (output/"sidecars.jsonl").open("xb") as truth:
            for i in ids:
                obs, sidecar = _draw_scene(split, i, SPLITS[split][1])
                q = validate_observation(obs, i, split)
                observed.write(encoded(obs))
                truth.write(encoded(sidecar))  # Keep the offending draw before rejecting it.
                fp = observation_fingerprint(obs)
                if fp in seen:
                    raise ValueError("duplicate observation; no draw may be replaced")
                seen.add(fp)
                record = feature_record(obs)
                validate_record(record, q)
                name = f"features/{i:05d}.npz"
                write_npz(output/name, **record)
                rows.append({"scene_id": i, "n": len(q), "path": name, "fingerprint": fp})
                cpu_resources(started)
        write_json(output/"rows.json", rows)
        after_auth, after_prior = verify_data_stage(authorization, split, shard, previous, earlier_shards)
        if after_auth != auth or set(after_prior) != set(prior):
            raise ValueError("data authorization or exclusion corpus changed during generation")
        binding = {"common": auth["common"], "authorization": authorization, "previous": previous,
                   "earlier_shards": earlier_shards, "split": split, "split_seed": SPLITS[split][1],
                   "shard": shard, "count": SHARD_SIZE, "split_count": SPLITS[split][0], "scene_ids": ids}
        seal_bundle(output, role="learned_observation_shard", binding=binding, resources=cpu_resources(started))
        return provenance.reference(output/"manifest.json")
    except BaseException as exc:
        mark_failure(output, exc)
        raise
