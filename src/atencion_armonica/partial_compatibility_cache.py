"""Immutable, NumPy-only feature cache; inference never reads supervision.

This preparer deliberately has NO test-unlock option. The prospective evaluator
must implement the separately audited freeze gate before materializing tests.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import resource
import time

import numpy as np

from .partial_compatibility import frequency_features, sham_geometry
from .shared_partial_data import OPEN_SPLITS, SPLITS, generate_scene

ROOT = Path(__file__).resolve().parents[2]
SOURCE_NAMES = ("partial_compatibility_cache.py", "partial_compatibility.py",
                "shared_partial_data.py", "peak_tokens.py")
RECORD_KEYS = frozenset(("tokens", "pair_cont", "ratio_class_id", "triples", "weights",
                        "residual_cents", "argmin_index_triple", "pair_support",
                        "sham_weights", "sham_evaluable", "sham_shift", "canonical_to_delivered"))


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)+"\n").encode()


def sha_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cache_sources():
    return {f"src/atencion_armonica/{name}": sha_file(Path(__file__).with_name(name))
            for name in SOURCE_NAMES}


def observation_fingerprint(obs):
    q = np.asarray(obs["log_f"], dtype="<f4")
    return hashlib.sha256(np.sort(q).tobytes()).hexdigest()


def feature_record(obs):
    if set(obs) != {"scene_id", "split_seed", "log_f"}:
        raise ValueError("observation must not contain supervision")
    q = np.asarray(obs["log_f"], dtype=np.float32)
    features = frequency_features(q)
    g = features.pop("geometry")
    sham = sham_geometry(q, g, split_seed=obs["split_seed"], scene_id=obs["scene_id"])
    return {**features, **g, "sham_weights": sham["weights"],
            "sham_evaluable": np.asarray(sham["evaluable"], dtype=bool),
            "sham_shift": np.asarray(-1 if sham["shift"] is None else sham["shift"], dtype=np.int64),
            "canonical_to_delivered": np.asarray([] if sham["canonical_to_delivered"] is None
                                                  else sham["canonical_to_delivered"], dtype=np.int64)}


def prepare_open_split(output: Path, split: str):
    if split not in OPEN_SPLITS:
        raise PermissionError("this preparer cannot open test splits")
    count, seed = SPLITS[split]
    output.mkdir(parents=True, exist_ok=False)
    (output/"features").mkdir()
    started, sources, rows, seen = time.monotonic(), cache_sources(), [], set()
    try:
        with (output/"observations.jsonl").open("xb") as obs_file, \
             (output/"sidecars.jsonl").open("xb") as truth_file:
            for scene_id in range(count):
                obs, truth = generate_scene(split, scene_id)
                fingerprint = observation_fingerprint(obs)
                if fingerprint in seen:
                    raise ValueError("duplicate observation; no scene may be silently rejected")
                seen.add(fingerprint)
                record = feature_record(obs)  # No truth argument or sidecar lookup.
                name = f"features/{scene_id:05d}.npz"
                with (output/name).open("xb") as handle:
                    np.savez_compressed(handle, **record)
                obs_file.write(encoded(obs))
                truth_file.write(encoded(truth))
                rows.append({"scene_id": scene_id, "n": len(obs["log_f"]), "path": name,
                             "sha256": sha_file(output/name), "fingerprint": fingerprint})
                if time.monotonic()-started > 120 or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 >= 1024**3:
                    raise RuntimeError("CPU preparation budget exceeded; retain incomplete output")
        if cache_sources() != sources:
            raise RuntimeError("cache source changed during preparation")
        report = {"schema_version": 1, "status": "COMPLETE", "split": split,
                  "count": count, "split_seed": seed, "rows": rows, "source_sha256": sources,
                  "observations_sha256": sha_file(output/"observations.jsonl"),
                  "sidecars_sha256": sha_file(output/"sidecars.jsonl"),
                  "seconds": time.monotonic()-started,
                  "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                  "numpy_version": np.__version__}
        if report["seconds"] > 120 or report["peak_rss_bytes"] >= 1024**3:
            raise RuntimeError("CPU preparation budget exceeded before completion")
        with (output/"manifest.json").open("xb") as handle:
            handle.write(encoded(report))
    except BaseException as exc:
        with (output/"FAILURE.json").open("xb") as handle:
            handle.write(encoded({"status": "INCOMPLETE", "error": repr(exc)}))
        raise


class ObservationCache:
    """Load and verify observable cache once, without opening any sidecar."""

    def __init__(self, root: Path, expected_split: str):
        if expected_split not in OPEN_SPLITS:
            raise PermissionError("test cache requires the future frozen evaluation gate")
        self.root = root
        self.manifest = json.loads((root/"manifest.json").read_bytes())
        m = self.manifest
        if (m["schema_version"] != 1 or m["status"] != "COMPLETE"
                or m["split"] != expected_split or (m["count"], m["split_seed"]) != SPLITS[expected_split]
                or (root/"FAILURE.json").exists()):
            raise ValueError("wrong, incomplete or stale split")
        if m["source_sha256"] != cache_sources():
            raise ValueError("cache belongs to a different feature implementation")
        if sha_file(root/"observations.jsonl") != m["observations_sha256"]:
            raise ValueError("observation hash mismatch")
        self.observations = [json.loads(line) for line in (root/"observations.jsonl").read_bytes().splitlines()]
        if len(self.observations) != m["count"] or len(m["rows"]) != m["count"]:
            raise ValueError("split size mismatch")
        self.records, self.fingerprints = [], set()
        for scene_id, (row, obs) in enumerate(zip(m["rows"], self.observations)):
            if (set(obs) != {"scene_id", "split_seed", "log_f"} or obs["scene_id"] != scene_id
                    or obs["split_seed"] != m["split_seed"] or row["scene_id"] != scene_id
                    or row["path"] != f"features/{scene_id:05d}.npz"
                    or row["n"] != len(obs["log_f"])):
                raise ValueError("observation schema or ordered IDs mismatch")
            fingerprint = observation_fingerprint(obs)
            if fingerprint != row["fingerprint"] or fingerprint in self.fingerprints:
                raise ValueError("observation fingerprint or uniqueness mismatch")
            self.fingerprints.add(fingerprint)
            path = root/row["path"]
            if sha_file(path) != row["sha256"]:
                raise ValueError("feature hash mismatch")
            with np.load(path, allow_pickle=False) as arrays:
                record = {key: arrays[key] for key in arrays.files}
            if set(record) != RECORD_KEYS or any(not np.isfinite(a).all() for a in record.values()):
                raise ValueError("invalid cache feature schema")
            if not np.array_equal(record["tokens"][:, 0], np.asarray(obs["log_f"], np.float32)):
                raise ValueError("tokens do not match serialized observation")
            self.records.append(record)

    def __len__(self):
        return len(self.records)


def load_supervision(cache: ObservationCache):
    """Explicit training/evaluation authority, separate from ObservationCache."""
    path = cache.root/"sidecars.jsonl"
    if sha_file(path) != cache.manifest["sidecars_sha256"]:
        raise ValueError("supervision hash mismatch")
    truths = [json.loads(line) for line in path.read_bytes().splitlines()]
    if len(truths) != len(cache):
        raise ValueError("supervision count mismatch")
    for scene_id, (truth, obs) in enumerate(zip(truths, cache.observations)):
        if (truth["scene_id"] != scene_id or truth["split_seed"] != obs["split_seed"]
                or len(truth["source_ids"]) != len(obs["log_f"])
                or any(type(label) is not int for label in truth["source_ids"])):
            raise ValueError("supervision alignment mismatch")
    return truths


def assert_disjoint(*caches):
    seen = set()
    for cache in caches:
        if seen & cache.fingerprints:
            raise ValueError("identical observation shared across splits")
        seen.update(cache.fingerprints)
