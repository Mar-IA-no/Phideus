"""Prospective data stages and observable-only cache, separate from supervision.

Only prepare_split draws the declared roster, after its audited authorization.
The pure private law accepts a seed for mechanical tests; it is not a CLI gate.
No historical producer, seed table or artifact is modified.
"""
from __future__ import annotations

from itertools import combinations
import json
from pathlib import Path
import resource
import time

import numpy as np

from .partial_compatibility_cache import RECORD_KEYS, encoded, feature_record, observation_fingerprint
from .shared_partial_data import _observe
from .structured_source_artifacts import mark_failure, seal_bundle, write_json, write_npz
from .structured_source_metrics import SPLIT_SEEDS
from . import structured_source_gate as gate

SPLIT_ORDER = tuple(SPLIT_SEEDS)
COUNT = 256


def _draw_scene(split, scene_id, seed):
    """Same source law and draw ordering as the frozen historical producer."""
    if split not in SPLIT_SEEDS or type(scene_id) is not int or not 0 <= scene_id < COUNT:
        raise ValueError("unknown split or scene outside the declared roster")
    if type(seed) is not int or seed < 0:
        raise ValueError("invalid split seed")
    rng = np.random.default_rng(np.random.SeedSequence([seed, scene_id]))
    k = 4 if split == "ood_polyphony" else int(rng.integers(2, 4))
    lo, hi = (3e-3, 1e-2) if split == "ood_beta" else (1e-4, 1e-3)
    sources = []
    for _ in range(k):
        f0 = float(np.exp(rng.uniform(np.log(100.), np.log(500.))))
        beta = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        size = int(rng.integers(4, 9))
        indices = sorted(rng.choice(np.arange(1, 9), size, replace=False).tolist())
        gamma = (float(np.exp(rng.uniform(np.log(5e-6), np.log(5e-5))))
                 if split == "deformed_family" else 0.)
        sources.append({"f0": f0, "beta": beta, "gamma": gamma, "indices": indices})
    return _observe(sources, rng, 2., scene_id, seed)


def validate_observation(obs, scene_id, split):
    if (set(obs) != {"scene_id", "split_seed", "log_f"}
            or type(obs["scene_id"]) is not int or obs["scene_id"] != scene_id
            or type(obs["split_seed"]) is not int or obs["split_seed"] != SPLIT_SEEDS[split]):
        raise ValueError("observable schema or ordered split identity differs")
    original = np.asarray(obs["log_f"])
    q = original.astype(np.float32)
    if (q.ndim != 1 or not 8 <= len(q) <= 32 or not np.isfinite(q).all()
            or not np.array_equal(original, q.astype(np.float64))):
        raise ValueError("observation is not an exact finite delivered float32 vector")
    return q


def validate_record(record, q):
    n = len(q)
    triples = np.asarray(list(combinations(range(n), 3)), dtype=np.int64)
    t = len(triples)
    shapes = {"tokens": ((n, 2), np.float32), "pair_cont": ((n, n, 4), np.float32),
              "ratio_class_id": ((n, n), np.int64), "triples": ((t, 3), np.int64),
              "residual_cents": ((t,), np.float64), "weights": ((t,), np.float64),
              "argmin_index_triple": ((t,), np.int64), "pair_support": ((n, n), np.float64),
              "sham_weights": ((t,), np.float64), "sham_evaluable": ((), np.bool_),
              "sham_shift": ((), np.int64)}
    if set(record) != RECORD_KEYS:
        raise ValueError("feature schema changed or contains supervision")
    for key, (shape, dtype) in shapes.items():
        a = record[key]
        if a.shape != shape or a.dtype != dtype or not np.isfinite(a).all():
            raise ValueError(f"invalid feature shape/dtype/value: {key}")
    if (not np.array_equal(record["tokens"][:, 0], q) or np.any(record["tokens"][:, 1] != 0)
            or not np.array_equal(record["triples"], triples)
            or np.any(record["residual_cents"] < 0)
            or np.any(record["argmin_index_triple"] < 0) or np.any(record["argmin_index_triple"] >= 56)):
        raise ValueError("features differ from their observable identity or index domain")
    for key in ("weights", "pair_support", "sham_weights"):
        if np.any(record[key] < 0) or np.any(record[key] > 1):
            raise ValueError("compatibility values outside [0,1]")
    for key in ("pair_support", "ratio_class_id"):
        if not np.array_equal(record[key], record[key].T):
            raise ValueError("asymmetric pair feature")
    if not np.array_equal(record["pair_cont"], record["pair_cont"].transpose(1, 0, 2)):
        raise ValueError("asymmetric continuous pair feature")
    mapping = record["canonical_to_delivered"]
    if mapping.dtype != np.int64:
        raise ValueError("invalid sham mapping dtype")
    if bool(record["sham_evaluable"]):
        if (mapping.shape != (t,) or not np.array_equal(np.sort(mapping), np.arange(t))
                or not 1 <= int(record["sham_shift"]) < t):
            raise ValueError("invalid sham permutation")
    elif mapping.shape != (0,) or int(record["sham_shift"]) != -1:
        raise ValueError("unevaluable sham requires explicit empty mapping")


class StructuredObservations:
    """Integrity/identity validation only; caller must verify stage authorization.

    Reading the inventory hashes the sidecar bytes but never parses or exposes
    labels. Its explicit evaluation loader below is the only truth entry point.
    """
    def __init__(self, ref, split, common):
        self.root, self.manifest = gate.bundle_reference(ref, f"{split}_data", common)
        binding = self.manifest["binding"]
        if (binding.get("split") != split or binding.get("split_seed") != SPLIT_SEEDS[split]
                or binding.get("count") != COUNT):
            raise ValueError("data bundle has a different split identity")
        self.split = split
        self.observations = [json.loads(line) for line in (self.root/"observations.jsonl").read_bytes().splitlines()]
        rows = json.loads((self.root/"rows.json").read_bytes())
        expected_files = {"observations.jsonl", "sidecars.jsonl", "rows.json", "deduplication.json"}
        expected_files.update(f"features/{i:05d}.npz" for i in range(COUNT))
        if (len(self.observations) != COUNT or len(rows) != COUNT
                or set(self.manifest["artifacts_sha256"]) != expected_files):
            raise ValueError("wrong data roster or scientific inventory")
        self.records, self.fingerprints = [], set()
        for i, (obs, row) in enumerate(zip(self.observations, rows)):
            q = validate_observation(obs, i, split)
            fingerprint = observation_fingerprint(obs)
            if row != {"scene_id": i, "n": len(q), "path": f"features/{i:05d}.npz", "fingerprint": fingerprint}:
                raise ValueError("feature row is misaligned")
            if fingerprint in self.fingerprints:
                raise ValueError("duplicate observation; no silent replacement")
            self.fingerprints.add(fingerprint)
            with np.load(self.root/row["path"], allow_pickle=False) as raw:
                record = {k: raw[k] for k in raw.files}
            validate_record(record, q)
            self.records.append(record)

    def __len__(self):
        return len(self.observations)


def load_supervision(cache):
    """Evaluation-only sidecars, kept separate from forward and geometry inputs."""
    truths = [json.loads(line) for line in (cache.root/"sidecars.jsonl").read_bytes().splitlines()]
    if len(truths) != COUNT:
        raise ValueError("wrong supervision roster")
    for i, (obs, truth) in enumerate(zip(cache.observations, truths)):
        n = len(obs["log_f"])
        labels, indices = truth["source_ids"], truth["partial_indices"]
        if (truth["scene_id"] != i or truth["split_seed"] != obs["split_seed"]
                or len(labels) != n or any(type(x) is not int for x in labels)
                or sorted(set(labels)) != list(range(len(truth["sources"])))
                or len(indices) != n or any(type(x) is not int or not 1 <= x <= 8 for x in indices)
                or sorted(truth["permutation"]) != list(range(n))):
            raise ValueError("supervision identity or membership schema differs")
        ideal = np.asarray(truth["log_f_ideal"], np.float64)
        noise = np.asarray(truth["sensor_log_noise"], np.float64)
        if (ideal.shape != (n,) or noise.shape != (n,) or not np.isfinite(ideal).all()
                or not np.isfinite(noise).all()
                or not np.array_equal((ideal+noise-truth["mean_log_f_observed"]).astype(np.float32),
                                      np.asarray(obs["log_f"], np.float32))):
            raise ValueError("supervision does not reconstruct the delivered observation")
    return truths


def prior_corpus(authorization_record, split, previous):
    """Require every earlier fresh role in fixed operational order; no omission."""
    expected = SPLIT_ORDER[:SPLIT_ORDER.index(split)]
    if set(previous) != set(expected):
        raise ValueError("all previous fresh split references are required")
    common = authorization_record["common"]
    if split == "calibration":
        calibration_auth = authorization_record
        frozen = None
    else:
        frozen = gate.read_reference(authorization_record["freeze"])
        calibration_auth = gate.read_reference(frozen["calibration_authorization"])
        if previous["calibration"] != frozen["calibration_data"]:
            raise ValueError("previous calibration differs from the frozen dataset")
    refs = calibration_auth["historical_observations"]
    if refs != gate.historical_observations():
        raise ValueError("historical corpus omitted")
    seen = set()
    for ref in refs:
        for line in gate.verify_reference(ref).read_bytes().splitlines():
            seen.add(observation_fingerprint(json.loads(line)))
    for role in expected:
        root, manifest = gate.bundle_reference(previous[role], f"{role}_data", common)
        binding = manifest["binding"]
        if (binding.get("split") != role or binding.get("split_seed") != SPLIT_SEEDS[role]
                or binding.get("count") != COUNT):
            raise ValueError("previous fresh corpus role differs")
        if role != "calibration" and gate.read_reference(binding["authorization"]) != authorization_record:
            raise ValueError("previous test was not produced under this freeze authorization")
        if binding.get("previous") != {r: previous[r] for r in SPLIT_ORDER[:SPLIT_ORDER.index(role)]}:
            raise ValueError("previous corpus chain was branched or omitted")
        observations = [json.loads(line) for line in (root/"observations.jsonl").read_bytes().splitlines()]
        if len(observations) != COUNT:
            raise ValueError("previous corpus is incomplete")
        for i, obs in enumerate(observations):
            validate_observation(obs, i, role)
            fingerprint = observation_fingerprint(obs)
            if fingerprint in seen:
                raise ValueError("previous fresh corpus already contains a duplicate")
            seen.add(fingerprint)
    return seen


def cpu_resources(started):
    result = {"seconds": time.monotonic()-started,
              "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
    if result["seconds"] > 1200 or result["peak_rss_bytes"] >= 2*1024**3:
        raise RuntimeError("CPU phase budget exceeded; keep incomplete output")
    return result


def prepare_split(output, split, *, authorization, previous):
    started = time.monotonic()
    auth = gate.verify_authorization(authorization, split)  # Before mkdir or RNG.
    seen = set(prior_corpus(auth, split, previous))
    cpu_resources(started)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    try:
        (output/"features").mkdir()
        write_json(output/"deduplication.json", {"historical_observations": gate.historical_observations(),
                                               "previous": previous, "prior_unique_observations": len(seen),
                                               "policy": "fail_on_sorted_q32_duplicate_no_replacement"})
        rows = []
        with (output/"observations.jsonl").open("xb") as observed, (output/"sidecars.jsonl").open("xb") as truth:
            for i in range(COUNT):
                obs, sidecar = _draw_scene(split, i, SPLIT_SEEDS[split])
                q = validate_observation(obs, i, split)
                # Preserve the exact failing draw too; never silently discard it.
                observed.write(encoded(obs))
                truth.write(encoded(sidecar))
                fingerprint = observation_fingerprint(obs)
                if fingerprint in seen:
                    raise ValueError("duplicate observation; no draw may be replaced")
                seen.add(fingerprint)
                record = feature_record(obs)  # No sidecar argument.
                validate_record(record, q)
                name = f"features/{i:05d}.npz"
                write_npz(output/name, **record)
                rows.append({"scene_id": i, "n": len(q), "path": name, "fingerprint": fingerprint})
                cpu_resources(started)
        write_json(output/"rows.json", rows)
        if gate.verify_authorization(authorization, split) != auth:
            raise ValueError("authorization changed during data preparation")
        if prior_corpus(auth, split, previous) != seen.difference(r["fingerprint"] for r in rows):
            raise ValueError("deduplication inputs changed during generation")
        binding = {"common": auth["common"], "authorization": authorization, "previous": previous,
                   "split": split, "split_seed": SPLIT_SEEDS[split], "count": COUNT}
        seal_bundle(output, role=f"{split}_data", binding=binding, resources=cpu_resources(started))
        return gate.reference(output/"manifest.json")
    except BaseException as exc:
        mark_failure(output, exc)
        raise
