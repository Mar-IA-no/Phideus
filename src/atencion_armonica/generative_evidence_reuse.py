"""Read-only train/calibration adapter for the enriched candidate universe.

Pinned import 06, not an import of old training wrappers or old tensor caps.
No draws, fits, forwards, sidecar parsing, model loading or CUDA entry points.
Payload hashes are checked on the exact bytes consumed and retained in a ledger.
"""
from __future__ import annotations

from io import BytesIO
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from .generative_evidence import CHECKPOINTS, partitions_checked
from . import observable_source_rivals as law
from .partial_compatibility_cache import observation_fingerprint
from .source_artifacts import load_ordered_logits
from .structured_source_artifacts import safe_member
from .structured_source_data import validate_record

ROOT = Path(__file__).resolve().parents[2]
OLD = "data/atencion_armonica/learned_partition_reader_v1"
AUTHORIZATION = {"path": f"{OLD}/authorization/train_calibration_06.json",
                 "sha256": "7740d564c748389f47ce7a818675ed5331e81ae4e0c7e50e6576a2e8df8568b5"}
IMPORT = {"path": f"{OLD}/reuse_06/manifest.json",
          "sha256": "f09c96b700accb834bb99233f9b01278e3b30d99d4fc3e5ea3ae74c5b9813ceb"}
OPEN_SPLITS = {"train": (4096, 2026090880), "calibration": (512, 2026090881)}


class VerifiedBytes:
    """Hash and parse the same read; ledger is evidence, never a permission gate."""
    def __init__(self, root):
        self.root, self.consumed = Path(root).resolve(), {}

    def read(self, ref):
        if (not isinstance(ref, dict) or set(ref) != {"path", "sha256"}
                or not isinstance(ref["sha256"], str) or len(ref["sha256"]) != 64):
            raise ValueError("invalid immutable file reference")
        path = safe_member(self.root, ref["path"])
        if path.is_symlink():
            raise ValueError("historical input must not be a symlink")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != ref["sha256"]:
            raise ValueError(f"consumed payload hash differs: {ref['path']}")
        previous = self.consumed.setdefault(ref["path"], ref["sha256"])
        if previous != ref["sha256"]:
            raise ValueError("one path claimed with conflicting hashes")
        return raw

    def json(self, ref):
        return json.loads(self.read(ref))


class HistoricalBundle:
    """Validate the manifest now, payload bytes individually when consumed."""
    def __init__(self, reader, ref, role, common):
        self.reader, self.ref = reader, ref.copy()
        self.manifest = reader.json(ref)
        m = self.manifest
        if (set(m) != {"schema", "status", "role", "binding", "artifacts_sha256", "resources_sha256"}
                or m["schema"] != "structured-source-bundle-v1" or m["status"] != "COMPLETE"
                or m["role"] != role or m["binding"].get("common") != common
                or Path(ref["path"]).name != "manifest.json"):
            raise ValueError("historical bundle role, common binding or state differs")
        self.prefix = Path(ref["path"]).parent.as_posix()
        for marker in ("FAILURE.json", "INCOMPLETE.json"):
            if safe_member(reader.root, (Path(self.prefix)/marker).as_posix()).exists():
                raise ValueError("historical bundle has an incomplete marker")

    def reference(self, name):
        safe_member(self.reader.root/self.prefix, name)
        return {"path": (Path(self.prefix)/name).as_posix(), "sha256": self.manifest["artifacts_sha256"][name]}

    def read(self, name):
        return self.reader.read(self.reference(name))

    def json(self, name):
        return json.loads(self.read(name))


class OpenReuse:
    """Only the two predeclared, open development splits; tests are impossible here."""
    def __init__(self):
        self.reader = VerifiedBytes(ROOT)
        auth = self.reader.json(AUTHORIZATION)
        if auth["status"] != "TRAIN_CALIBRATION_READY":
            raise ValueError("wrong historical authorization")
        self.common = auth["common"]
        imported = HistoricalBundle(self.reader, IMPORT, "learned_prepared_import", self.common)
        if imported.manifest["binding"]["authorization"] != AUTHORIZATION:
            raise ValueError("import belongs to another authorization")
        self.prepared = imported.json("prepared.json")
        if self.prepared["authorization"] != AUTHORIZATION:
            raise ValueError("prepared reuse authority differs")
        self.corpora, self.shards = {}, {}
        for split, (count, seed) in OPEN_SPLITS.items():
            corpus = self.bundle(self.prepared[split], "learned_training_corpus")
            b = corpus.manifest["binding"]
            if b["authorization"] != AUTHORIZATION or b["split"] != split:
                raise ValueError("corpus identity differs")
            aggregate = self.bundle(b["data"], "learned_observation_split")
            ab = aggregate.manifest["binding"]
            if any(ab[k] != v for k, v in {"authorization": AUTHORIZATION, "split": split,
                                            "count": count, "split_seed": seed}.items()):
                raise ValueError("aggregate identity differs")
            refs = corpus.json("shards.json")
            if (len(refs) != count//512 or any(set(r) != {"data", "logits", "scored", "targets"} for r in refs)
                    or [r["data"] for r in refs] != aggregate.json("shards.json")):
                raise ValueError("open shard roster differs")
            self.corpora[split], self.shards[split] = corpus.ref, refs

    def bundle(self, ref, role):
        if not ref["path"].startswith(f"{OLD}/reuse_06/"):
            raise ValueError("source is not the pinned imported cohort")
        return HistoricalBundle(self.reader, ref, role, self.common)

    def shard(self, split, shard):
        if split not in OPEN_SPLITS:
            raise PermissionError("reuse port exposes only open train/calibration, never tests")
        if type(shard) is not int or not 0 <= shard < len(self.shards[split]):
            raise ValueError("shard outside the fixed open roster")
        return OpenShard(self, split, shard)

    def receipt(self):
        return {"operation": "READ_ONLY_OPEN_REUSE_NO_DRAWS_NO_FORWARDS",
                "authorization": AUTHORIZATION.copy(), "import": IMPORT.copy(),
                "corpora": copy.deepcopy(self.corpora), "consumed_sha256": dict(sorted(self.reader.consumed.items()))}


class OpenShard:
    """Read one 512-scene observation/logit shard; stream features and pools by scene."""
    def __init__(self, reuse, split, shard):
        self.reuse, self.split, self.shard = reuse, split, shard
        self.count, self.seed = OPEN_SPLITS[split]
        self.ids = list(range(shard*512, (shard+1)*512))
        refs = reuse.shards[split][shard]
        self.data = reuse.bundle(refs["data"], "learned_observation_shard")
        self.logits = reuse.bundle(refs["logits"], "learned_logits_shard")
        self.scored = reuse.bundle(refs["scored"], "learned_scored_shard")
        shared = {"authorization": AUTHORIZATION, "split": split, "split_seed": self.seed,
                  "shard": shard, "scene_ids": self.ids, "count": 512}
        for bundle in (self.data, self.logits, self.scored):
            b = bundle.manifest["binding"]
            if any(b.get(k) != v for k, v in shared.items()):
                raise ValueError("shard manifest identity differs")
        if (self.data.manifest["binding"]["split_count"] != self.count
                or self.logits.manifest["binding"]["data"] != self.data.ref
                or self.scored.manifest["binding"]["data"] != self.data.ref
                or self.scored.manifest["binding"]["logits"] != self.logits.ref):
            raise ValueError("cross-stage data/logit references differ")
        self.observations = [json.loads(line) for line in self.data.read("observations.jsonl").splitlines()]
        rows = self.data.json("rows.json")
        if len(rows) != 512 or len(self.observations) != 512:
            raise ValueError("incomplete ordered observation shard")
        seen = set()
        for i, obs, row in zip(self.ids, self.observations, rows):
            q = validate_observation(obs, i, split)
            fp = observation_fingerprint(obs)
            if row != {"scene_id": i, "n": len(q), "path": f"features/{i:05d}.npz", "fingerprint": fp}:
                raise ValueError("feature row identity differs")
            if fp in seen:
                raise ValueError("duplicate observation in open shard")
            seen.add(fp)
        forward = self.logits.json("forward.json")
        expected_rows = [{"seed": r["seed"], "checkpoint": r["checkpoint"],
                          "path": f"seed_{r['seed']}.npz", "count": 512} for r in reuse.common["checkpoints"]]
        # Producer records CPU/package identity and CUDA-forward identity in
        # different schemas. Compare their shared versions, not whole dicts.
        runtime = forward["runtime"]
        if ([r["seed"] for r in expected_rows] != list(CHECKPOINTS)
                or forward["rows"] != expected_rows
                or set(runtime) != {"torch", "numpy", "cuda", "cudnn", "device"}
                or any(runtime[k] != reuse.common["runtime"][k] for k in ("torch", "numpy"))):
            raise ValueError("preserved forward checkpoint roster/runtime differs")
        self.matrices = {cp: load_ordered_logits(BytesIO(self.logits.read(f"seed_{cp}.npz")), self.observations)
                         for cp in CHECKPOINTS}

    def scene(self, scene_id):
        if type(scene_id) is not int or scene_id not in self.ids:
            raise ValueError("scene outside this shard")
        j = scene_id-self.ids[0]
        obs = self.observations[j]
        q = validate_observation(obs, scene_id, self.split)
        with np.load(BytesIO(self.data.read(f"features/{scene_id:05d}.npz")), allow_pickle=False) as saved:
            record = {k: saved[k] for k in saved.files}
        validate_record(record, q)
        order = np.argsort(q, kind="stable")
        pools = {}
        for cp in CHECKPOINTS:
            pool = self.scored.json(f"seed_{cp}/{scene_id:05d}_pool.json")["pool"]
            if (pool["canonical_to_observed"] != order.tolist() or pool["canonical_q32"] != q[order].tolist()
                    or pool["q_tie_count"] != len(q)-len(np.unique(q)) or pool["max_group_size_prior"] != 8):
                raise ValueError("preserved pool is not aligned to delivered q32")
            ps = [law.validate_partition(p, len(q)) for p in pool["partitions"]]
            if ps != sorted(set(ps)) or len(ps) != pool["retained_count"]:
                raise ValueError("preserved pool roster differs")
            pools[str(cp)] = ps
        inventory = law.candidate_inventory(pools, len(q))
        ps = partitions_checked(sorted(law.signature(r["partition"]) for r in inventory["candidates"]
                                       if r["status"] == "SUPPORTED"), len(q))
        return {"observation": copy.deepcopy(obs), "features": record,
                "logits": {cp: self.matrices[cp][j].copy() for cp in CHECKPOINTS},
                "pools": pools, "inventory": inventory, "partitions": ps,
                "canonical_to_observed": order, "q32": q[order],
                "status": "ELIGIBLE" if ps else "NO_OBSERVABLE_CANDIDATE"}


def validate_observation(obs, scene_id, split):
    if (split not in OPEN_SPLITS or type(scene_id) is not int
            or not 0 <= scene_id < OPEN_SPLITS[split][0]
            or set(obs) != {"scene_id", "split_seed", "log_f"}
            or type(obs["scene_id"]) is not int or obs["scene_id"] != scene_id
            or type(obs["split_seed"]) is not int or obs["split_seed"] != OPEN_SPLITS[split][1]):
        raise ValueError("open observation schema or identity differs")
    original = np.asarray(obs["log_f"])
    q = original.astype(np.float32)
    law.observable_q32(np.sort(q))
    if not np.array_equal(original, q.astype(np.float64)):
        raise ValueError("open observation is not exact q32")
    return q
