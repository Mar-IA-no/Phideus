"""Held-out cache, accessible only through a verified prospective evaluation freeze."""
from __future__ import annotations

import json
from pathlib import Path
import resource
import time

import numpy as np

from .partial_compatibility_cache import (
    RECORD_KEYS, cache_sources, encoded, feature_record, observation_fingerprint, sha_file,
)
from .partial_compatibility_test_gate import TEST_SPLITS, verify_freeze
from .shared_partial_data import SPLITS, generate_scene


def prepare_test_split(output, split, freeze_path):
    freeze = verify_freeze(freeze_path)
    if split not in TEST_SPLITS:
        raise PermissionError("only the five frozen test splits belong here")
    count, seed = SPLITS[split]
    output.mkdir(parents=True, exist_ok=False)
    (output/"features").mkdir()
    started, rows, seen = time.monotonic(), [], set()
    try:
        with (output/"observations.jsonl").open("xb") as obs_file, (output/"sidecars.jsonl").open("xb") as truth_file:
            for scene_id in range(count):
                obs, truth = generate_scene(split, scene_id, allow_test=True)
                fingerprint = observation_fingerprint(obs)
                if fingerprint in seen:
                    raise ValueError("duplicate test observation; never filter scenes")
                seen.add(fingerprint)
                record = feature_record(obs)
                name = f"features/{scene_id:05d}.npz"
                with (output/name).open("xb") as handle:
                    np.savez_compressed(handle, **record)
                obs_file.write(encoded(obs))
                truth_file.write(encoded(truth))
                rows.append({"scene_id": scene_id, "n": len(obs["log_f"]), "path": name,
                             "sha256": sha_file(output/name), "fingerprint": fingerprint})
                if time.monotonic()-started > 120 or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 >= 1024**3:
                    raise RuntimeError("held-out CPU preparation budget exceeded")
        if verify_freeze(freeze_path) != freeze:
            raise ValueError("test freeze changed during preparation")
        manifest = {"schema_version": 1, "status": "COMPLETE", "split": split, "count": count, "split_seed": seed,
                    "evaluation_freeze_sha256": sha_file(freeze_path), "source_sha256": cache_sources(), "rows": rows,
                    "observations_sha256": sha_file(output/"observations.jsonl"),
                    "sidecars_sha256": sha_file(output/"sidecars.jsonl"), "seconds": time.monotonic()-started,
                    "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                    "numpy_version": np.__version__}
        if manifest["seconds"] > 120 or manifest["peak_rss_bytes"] >= 1024**3:
            raise RuntimeError("held-out preparation budget exceeded before close")
        with (output/"manifest.json").open("xb") as handle:
            handle.write(encoded(manifest))
    except BaseException as exc:
        with (output/"FAILURE.json").open("xb") as handle:
            handle.write(encoded({"status": "INCOMPLETE", "error": repr(exc)}))
        raise


class FrozenTestCache:
    """Observation-only reader, preserving the open cache format without bypassing it."""

    def __init__(self, root, expected_split, freeze_path):
        verify_freeze(freeze_path)
        if expected_split not in TEST_SPLITS:
            raise PermissionError("not a frozen held-out split")
        self.root = root
        self.manifest = json.loads((root/"manifest.json").read_text())
        m = self.manifest
        if (m["status"] != "COMPLETE" or m["schema_version"] != 1 or m["split"] != expected_split
                or (m["count"], m["split_seed"]) != SPLITS[expected_split]
                or m["evaluation_freeze_sha256"] != sha_file(freeze_path)
                or m["source_sha256"] != cache_sources() or (root/"FAILURE.json").exists()
                or sha_file(root/"observations.jsonl") != m["observations_sha256"]):
            raise ValueError("test cache identity, freeze or hash mismatch")
        self.observations = [json.loads(line) for line in (root/"observations.jsonl").read_bytes().splitlines()]
        if len(self.observations) != m["count"] or len(m["rows"]) != m["count"]:
            raise ValueError("test cache count mismatch")
        self.records, self.fingerprints = [], set()
        for scene_id, (row, obs) in enumerate(zip(m["rows"], self.observations)):
            if (set(obs) != {"scene_id", "split_seed", "log_f"} or obs["scene_id"] != scene_id
                    or obs["split_seed"] != m["split_seed"] or row["scene_id"] != scene_id
                    or row["path"] != f"features/{scene_id:05d}.npz" or row["n"] != len(obs["log_f"])):
                raise ValueError("test cache ordered observation schema mismatch")
            fingerprint = observation_fingerprint(obs)
            if fingerprint != row["fingerprint"] or fingerprint in self.fingerprints:
                raise ValueError("duplicate or changed test observation")
            self.fingerprints.add(fingerprint)
            if sha_file(root/row["path"]) != row["sha256"]:
                raise ValueError("test feature hash mismatch")
            with np.load(root/row["path"], allow_pickle=False) as arrays:
                record = {key: arrays[key] for key in arrays.files}
            if set(record) != RECORD_KEYS or any(not np.isfinite(a).all() for a in record.values()):
                raise ValueError("test feature schema invalid")
            if not np.array_equal(record["tokens"][:, 0], np.asarray(obs["log_f"], dtype=np.float32)):
                raise ValueError("test tokens/observation mismatch")
            self.records.append(record)

    def __len__(self):
        return len(self.records)


def verify_disjoint_manifests(roots):
    """Compare identities across all eight manifests without retaining all feature tensors."""
    seen, split_names = set(), set()
    for root in roots:
        m = json.loads((root/"manifest.json").read_text())
        if (m["status"] != "COMPLETE" or m["split"] not in SPLITS or m["split"] in split_names
                or m.get("schema_version") != 1 or (m["count"], m["split_seed"]) != SPLITS[m["split"]]
                or len(m["rows"]) != m["count"]
                or [row["scene_id"] for row in m["rows"]] != list(range(m["count"]))
                or (root/"FAILURE.json").exists()):
            raise ValueError("incomplete or repeated cache split")
        split_names.add(m["split"])
        identities = [row["fingerprint"] for row in m["rows"]]
        if len(identities) != m["count"] or len(set(identities)) != len(identities) or seen.intersection(identities):
            raise ValueError("repeated observable scene within or across splits")
        seen.update(identities)
    if split_names != set(SPLITS):
        raise ValueError("cross-split verification requires all eight declared splits")
