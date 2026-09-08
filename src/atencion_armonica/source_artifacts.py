"""Read-only NumPy access to the closed campaign, with pinned provenance."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .partial_compatibility_cache import RECORD_KEYS, sha_file

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT/"data/atencion_armonica"
SPLITS = ("validation", "ood_beta", "ood_polyphony")
ARMS = ("pairs_descriptors", "pairs_compatibility", "pairs_sham", "pairs_transitivity")
SEEDS = (2026090721, 2026090722, 2026090723)
ANCHORS = {
    "shared_partial_evaluation_freeze_v1.json": "0fa76af360ca97c56190453030daa62a8145447441228fcc7113c2b8808cba1f",
    "shared_partial_validation_logits_v1/manifest.json": "7a5ea2c5ec857f4de31164bc7e813480daaa7067dbe2b4d52f2ce7fb3fa6b4a5",
    "shared_partial_validation_selection_v1/manifest.json": "813c842aaf05929709f187f2fe7bc5d54c3ac751974812f34f4d73bd2e477db5",
    "shared_partial_test_logits_v1/manifest.json": "d1acf22c9f6452cc4404c009184be6618517da520123a235e33b6ccdafda82f9",
    "shared_partial_test_analysis_v1/manifest.json": "62eb10e502bfd32b99ec4a5acd13f6777999b08f4266fedf4f8744bd98a4c05b"}


def reader_key(reader):
    return reader["arm"] if reader["seed"] is None else f'{reader["arm"]}__seed_{reader["seed"]}'


def load_ordered_logits(path, observations):
    """Same archived schema as load_logits, without importing its Torch module."""
    if not observations or any(set(o) != {"scene_id", "split_seed", "log_f"} for o in observations):
        raise ValueError("expected observable-only ordered identities")
    if any(type(o[k]) is not int for o in observations for k in ("scene_id", "split_seed")):
        raise ValueError("invalid identity metadata")
    q = [np.asarray(o["log_f"], dtype="<f4") for o in observations]
    if any(a.ndim != 1 or not 2 <= len(a) <= 32 or not np.isfinite(a).all() for a in q):
        raise ValueError("invalid observed coordinates")
    sizes = np.array([len(a) for a in q], dtype=np.int64)
    expected = {"sizes": sizes, "offsets": np.r_[np.int64(0), np.cumsum(sizes*sizes)],
                "scene_ids": np.array([o["scene_id"] for o in observations], dtype=np.int64),
                "split_seeds": np.array([o["split_seed"] for o in observations], dtype=np.int64),
                "observation_fingerprints": np.array([hashlib.sha256(a.tobytes()).hexdigest() for a in q], dtype="U64")}
    with np.load(path, allow_pickle=False) as arrays:
        if set(arrays.files) != set(expected)|{"logits"}:
            raise ValueError("wrong raw logit schema")
        for key, value in expected.items():
            if arrays[key].dtype != value.dtype or not np.array_equal(arrays[key], value):
                raise ValueError("raw logits do not match ordered observation identities")
        flat = arrays["logits"]
    if flat.dtype != np.float32 or flat.shape != (expected["offsets"][-1],) or not np.isfinite(flat).all():
        raise ValueError("invalid raw float32 logits")
    result = [flat[start:end].reshape(int(n), int(n)).copy()
              for start, end, n in zip(expected["offsets"][:-1], expected["offsets"][1:], sizes)]
    if any(not np.array_equal(a, a.T) for a in result):
        raise ValueError("asymmetric archived logits")
    return result


def validate_partition(partition, n):
    if (not isinstance(partition, list) or not partition
            or any(not isinstance(group, list) or not group for group in partition)):
        raise ValueError("partition must have nonempty groups")
    flat = [member for group in partition for member in group]
    if any(type(i) is not int for i in flat) or sorted(flat) != list(range(n)):
        raise ValueError("partition must cover each observed event exactly once")
    return partition


class ClosedArtifacts:
    """Validate full identity rosters but materialize only fixed IDs 0..31."""
    def __init__(self):
        self.inputs = {}
        pinned = {name: self.json(DATA/name, digest) for name, digest in ANCHORS.items()}
        self.freeze = pinned["shared_partial_evaluation_freeze_v1.json"]
        self.verify_sources(self.freeze["source_sha256"])
        if self.freeze["status"] != "FROZEN_BEFORE_TEST" or len(self.freeze["source_sha256"]) != 25:
            raise ValueError("wrong prior freeze")
        selection_root = DATA/"shared_partial_validation_selection_v1"
        selection = pinned["shared_partial_validation_selection_v1/manifest.json"]
        self.reader_manifest = self.json(selection_root/"readers.json", selection["readers_sha256"])
        frozen = {(r["arm"], r["seed"]): r for r in self.freeze["readers"]}
        self.readers = [r for r in self.reader_manifest["readers"] if r["arm"] in ARMS or r["arm"] == "analytic_support"]
        roster = {(a, s) for a in ARMS for s in SEEDS}|{("analytic_support", None)}
        if len(self.readers) != 13 or {(r["arm"], r["seed"]) for r in self.readers} != roster:
            raise ValueError("wrong reader roster")
        for reader in self.readers:
            if any(reader[k] != v for k, v in frozen[(reader["arm"], reader["seed"])].items()):
                raise ValueError("reader differs from frozen threshold/checkpoint")
        self.raw = {}
        for kind in ("validation", "test"):
            root = DATA/f"shared_partial_{kind}_logits_v1"
            manifest = pinned[f"shared_partial_{kind}_logits_v1/manifest.json"]
            if manifest["status"] != f"RAW_{kind.upper()}_COMPLETE":
                raise ValueError("incomplete raw forward")
            request = self.json(root/"request.json", manifest["request_sha256"])
            forward = self.json(root/"forward.json", manifest["forward_sha256"])
            self.raw[kind] = (root, request, forward)
        self.analysis = pinned["shared_partial_test_analysis_v1/manifest.json"]

    def verify_sources(self, sources):
        for name, digest in sources.items():
            self.verify(ROOT/name, digest)

    def verify(self, path, digest):
        path = path.resolve()
        if sha_file(path) != digest:
            raise ValueError(f"hash mismatch: {path}")
        if (path.parent/"FAILURE.json").exists():
            raise ValueError(f"incomplete artifact: {path}")
        self.inputs[str(path.relative_to(ROOT))] = digest

    def json(self, path, digest):
        self.verify(path, digest)
        return json.loads(path.read_bytes())

    def jsonl(self, path, digest):
        self.verify(path, digest)
        return [json.loads(line) for line in path.read_bytes().splitlines()]

    def load_split(self, split):
        if split not in SPLITS:
            raise ValueError("split outside fixed diagnostic")
        kind = "validation" if split == "validation" else "test"
        root = DATA/("shared_partial_cache_v1" if kind == "validation" else "shared_partial_test_cache_v1")/split
        digest = (self.freeze["validation_manifest_sha256"] if kind == "validation" else
                  self.raw[kind][1]["test_manifest_sha256"][split])
        manifest = self.json(root/"manifest.json", digest)
        self.verify_sources(manifest["source_sha256"])
        if manifest["status"] != "COMPLETE" or manifest["split"] != split or manifest["count"] != 1024:
            raise ValueError("wrong cached split")
        observations = self.jsonl(root/"observations.jsonl", manifest["observations_sha256"])
        truths = self.jsonl(root/"sidecars.jsonl", manifest["sidecars_sha256"])
        if len(observations) != 1024 or len(truths) != 1024 or len(manifest["rows"]) != 1024:
            raise ValueError("wrong scene roster")
        records = []
        for scene_id, (obs, truth, row) in enumerate(zip(observations, truths, manifest["rows"])):
            if (set(obs) != {"scene_id", "split_seed", "log_f"} or obs["scene_id"] != scene_id
                    or obs["split_seed"] != manifest["split_seed"] or truth["scene_id"] != scene_id
                    or truth["split_seed"] != obs["split_seed"] or row["scene_id"] != scene_id
                    or row["n"] != len(obs["log_f"]) or len(truth["source_ids"]) != row["n"]
                    or any(type(x) is not int for x in truth["source_ids"])
                    or row["path"] != f"features/{scene_id:05d}.npz"):
                raise ValueError("observation/truth/feature identity mismatch")
            q = np.asarray(obs["log_f"], dtype="<f4")
            if hashlib.sha256(np.sort(q).tobytes()).hexdigest() != row["fingerprint"]:
                raise ValueError("feature fingerprint mismatch")
            if scene_id < 32:
                self.verify(root/row["path"], row["sha256"])
                with np.load(root/row["path"], allow_pickle=False) as arrays:
                    record = {k: arrays[k] for k in arrays.files}
                if (set(record) != RECORD_KEYS or any(not np.isfinite(a).all() for a in record.values())
                        or not np.array_equal(record["tokens"][:, 0], q)):
                    raise ValueError("feature schema/order mismatch")
                records.append(record)
        partitions, logits = {}, {}
        for reader in self.readers:
            key = reader_key(reader)
            if kind == "validation":
                path = DATA/"shared_partial_validation_selection_v1"/reader["metrics_path"]
                partition_digest = reader["metrics_sha256"]
            else:
                name = f"{split}/{key}.jsonl"
                path = DATA/"shared_partial_test_analysis_v1"/name
                partition_digest = self.analysis["artifacts_sha256"][name]
            rows = self.jsonl(path, partition_digest)
            if len(rows) != 1024 or [r["scene_id"] for r in rows] != list(range(1024)):
                raise ValueError("partition scene identity mismatch")
            partitions[key] = [validate_partition(r["partition"], len(o["log_f"])) for r, o in zip(rows, observations)][:32]
            if reader["seed"] is not None:
                raw_root, _, forward = self.raw[kind]
                selected = [r for r in forward["rows"] if r["arm"] == reader["arm"] and r["seed"] == reader["seed"]
                            and (kind == "validation" or r["split"] == split)]
                if len(selected) != 1:
                    raise ValueError("missing/duplicate raw reader")
                row = selected[0]
                expected_path = f"{key}.npz" if kind == "validation" else f"{split}/{key}.npz"
                if row["path"] != expected_path or row["checkpoint_sha256"] != reader["checkpoint_sha256"] or row["scene_count"] != 1024:
                    raise ValueError("raw reader/checkpoint mismatch")
                self.verify(raw_root/row["path"], row["sha256"])
                logits[key] = load_ordered_logits(raw_root/row["path"], observations)[:32]
        return observations[:32], records, truths[:32], partitions, logits
