"""Authenticated complete OPEN corpus -> durable float32 inputs -> cell data.

Only compact aggregate arrays are consumed: no factors, draws, CUDA, historical
forwards or fresh-test access. The stage caller pins the OPEN completion and
this implementation, owns the resource budget, and freezes delivered receipts
before training. A receipt here is data provenance, not execution permission.
"""
from __future__ import annotations

import json

from . import generative_evidence as ge
from . import generative_evidence_cache as cache
from . import generative_evidence_inputs as inputs
from . import generative_evidence_storage as storage
from .generative_evidence_cell import CellData
from .generative_evidence_normalization import arrays_digest
from .generative_evidence_reuse import OPEN_SPLITS
from .partial_compatibility_cache import encoded


class TrainingCorpus:
    def __init__(self, store, prepared_ref):
        self.store = store
        self.prepared_ref = json.loads(encoded(prepared_ref))
        complete = store.json(self.prepared_ref)
        if (set(complete) != {"schema", "binding", "status", "splits", "normalizers", "reuse", "profile"}
                or complete["schema"] != "generative-evidence-open-preparation-v1"
                or complete["binding"] != store.binding
                or complete["status"] != "OPEN_PREPARED_NOT_TRAINED_NO_TEST_ACCESS"
                or set(complete["splits"]) != set(OPEN_SPLITS)):
            raise ValueError("a complete bound OPEN preparation is required")
        self.splits = {}
        for split, (count, seed) in OPEN_SPLITS.items():
            value = store.json(complete["splits"][split])
            if (set(value) != {"schema", "binding", "split", "split_seed", "count", "shards"}
                    or value["schema"] != "generative-evidence-prepared-split-v1"
                    or value["binding"] != store.binding or value["split"] != split
                    or value["split_seed"] != seed or value["count"] != count
                    or len(value["shards"]) != count//512):
                raise ValueError("OPEN split requires its full ordered shard roster")
            self.splits[split] = value
        self.normalizer_ref = complete["normalizers"]
        norm = store.json(self.normalizer_ref)
        if set(norm) != {"prepared_train", "normalizers"} or norm["prepared_train"] != complete["splits"]["train"]:
            raise ValueError("normalizers must bind this complete TRAIN split")
        self.norm = n = norm["normalizers"]
        if (set(n) != {"schema", "binding", "split", "split_seed", "scene_count", "eligible_scene_ids",
                       "excluded_scene_ids", "scene_identities", "common", "evidence", "array_digests"}
                or n["schema"] != "generative-evidence-train-normalizers-v1" or n["binding"] != store.binding
                or n["split"] != "train" or n["split_seed"] != OPEN_SPLITS["train"][1]
                or n["scene_count"] != 4096 or len(n["scene_identities"]) != 4096
                or len(set(n["scene_identities"])) != 4096
                or set(n["common"]) != {str(cp) for cp in ge.CHECKPOINTS}
                or not n["eligible_scene_ids"]):
            raise ValueError("normalizer provenance or full TRAIN identity differs")
        eligible, excluded = n["eligible_scene_ids"], n["excluded_scene_ids"]
        if (eligible != sorted(set(eligible)) or excluded != sorted(set(excluded))
                or sorted(eligible+excluded) != list(range(4096))):
            raise ValueError("normalizer eligible/excluded roster does not partition TRAIN")
        for value in n["common"].values():
            ge._normalizer(value, 5)
            if value["scene_count"] != [len(eligible)]*5:
                raise ValueError("common normalizer support differs")
        ge._normalizer(n["evidence"], 6)
        counts = n["evidence"]["scene_count"]
        if (len(counts) != 6 or any(type(c) is not int or not 0 < c <= len(eligible) for c in counts)
                or counts[::2] != counts[1::2]):
            raise ValueError("evidence normalizer lacks valid TRAIN branch support")
        self.digests = {}
        for row in n["array_digests"]:
            if set(row) != {"checkpoint_seed", "shard", "sha256"}:
                raise ValueError("normalizer array digest schema differs")
            key = row["checkpoint_seed"], row["shard"]
            if key in self.digests:
                raise ValueError("duplicate normalizer array digest")
            self.digests[key] = row["sha256"]
        if set(self.digests) != {(cp, s) for cp in ge.CHECKPOINTS for s in range(8)}:
            raise ValueError("normalizer requires all 24 TRAIN array digests")

    def raw(self, split, checkpoint_seed, shard):
        if split not in OPEN_SPLITS:
            raise PermissionError("training corpus has no fresh-test port")
        if (checkpoint_seed not in ge.CHECKPOINTS or type(shard) is not int
                or not 0 <= shard < OPEN_SPLITS[split][0]//512):
            raise ValueError("unknown OPEN checkpoint/shard")
        shard_ref = self.splits[split]["shards"][shard]
        index = self.store.json(shard_ref)
        ids = list(range(shard*512, (shard+1)*512))
        if (set(index) != {"schema", "binding", "split", "shard", "scene_ids", "records", "raw", "targets", "metrics"}
                or index["schema"] != "generative-evidence-prepared-shard-v1"
                or index["binding"] != self.store.binding or index["split"] != split
                or index["shard"] != shard or index["scene_ids"] != ids
                or set(index["raw"]) != {str(cp) for cp in ge.CHECKPOINTS}
                or len(index["records"]) != 512
                or any(set(r) != {"fit", "observable", "supervision"} for r in index["records"])):
            raise ValueError("prepared aggregate shard identity or roster differs")
        ref = index["raw"][str(checkpoint_seed)]
        arrays = self.store.arrays(ref)
        decoded = cache.unpack_rows(arrays, binding=self.store.binding, split=split, checkpoint_seed=checkpoint_seed)
        if decoded["scene_ids"] != ids:
            raise ValueError("raw corpus shard does not contain all 512 scenes in order")
        if split == "train":
            eligible = [i for i, r in zip(ids, decoded["rows"]) if r["partitions"]]
            if (arrays_digest(arrays) != self.digests[checkpoint_seed, shard]
                    or decoded["identities"] != self.norm["scene_identities"][shard*512:(shard+1)*512]
                    or eligible != [i for i in self.norm["eligible_scene_ids"] if i in range(shard*512, (shard+1)*512)]):
                raise ValueError("TRAIN arrays no longer match frozen normalization inputs")
        return index, shard_ref, arrays, decoded

    def _identity(self, split, cp, shard, index, shard_ref):
        return {"schema": "generative-evidence-delivered-index-v1", "binding": self.store.binding,
                "prepared": self.prepared_ref, "normalizers": self.normalizer_ref,
                "split": split, "checkpoint_seed": cp, "shard": shard,
                "prepared_shard": shard_ref, "raw": index["raw"][str(cp)]}

    def _decode(self, value, arrays, decoded, identity):
        if set(value) != set(identity) | {"inputs", "array_digest"} or any(value[k] != v for k, v in identity.items()):
            raise ValueError("delivered index provenance differs")
        if arrays_digest(arrays) != value["array_digest"]:
            raise ValueError("delivered array content differs")
        result = inputs.unpack_inputs(arrays, binding=self.store.binding, split=identity["split"],
            checkpoint_seed=identity["checkpoint_seed"], raw_ref=identity["raw"],
            normalizer_ref=self.normalizer_ref, identities=decoded["identities"])
        if result["scene_ids"] != decoded["scene_ids"]:
            raise ValueError("delivered shard scene roster differs")
        return result

    def materialize(self, *, check, progress=print):
        """Caller charges preparation budget; this does not launch training."""
        entries = []
        for split in OPEN_SPLITS:
            for shard in range(OPEN_SPLITS[split][0]//512):
                shared_identities = None
                for cp in ge.CHECKPOINTS:
                    check()
                    index, shard_ref, raw, decoded = self.raw(split, cp, shard)
                    if shared_identities is not None and decoded["identities"] != shared_identities:
                        raise ValueError("cross-checkpoint candidate identities differ")
                    shared_identities = decoded["identities"]
                    expected = inputs.pack_inputs(raw, self.norm["common"][str(cp)], self.norm["evidence"],
                        binding=self.store.binding, split=split, checkpoint_seed=cp,
                        raw_ref=index["raw"][str(cp)], normalizer_ref=self.normalizer_ref)
                    identity = self._identity(split, cp, shard, index, shard_ref)
                    path = self.store.path(f"delivered/{split}/cp_{cp}/shard_{shard}/index.json")
                    if path.exists():
                        ref = self.store.reference(path)
                        value = self.store.json(ref)
                        saved = self.store.arrays(value["inputs"])
                        self._decode(value, saved, decoded, identity)
                        if arrays_digest(saved) != arrays_digest(expected):
                            raise ValueError("saved float32 inputs differ from raw TRAIN-normalized values")
                    else:
                        check()
                        blob = self.store._new(path.parent, "inputs", ".npz")
                        storage.write_arrays(blob, expected)
                        value = {**identity, "inputs": self.store.reference(blob), "array_digest": arrays_digest(expected)}
                        self._decode(value, self.store.arrays(value["inputs"]), decoded, identity)
                        storage.write_json(path, value)
                        ref = self.store.reference(path)
                    entries.append({"split": split, "checkpoint_seed": cp, "shard": shard, "index": ref})
                    progress({"stage": "delivered32", "split": split, "checkpoint_seed": cp, "shard": shard})
        check()
        value = {"schema": "generative-evidence-delivered-open-v1", "binding": self.store.binding,
                 "prepared": self.prepared_ref, "normalizers": self.normalizer_ref, "entries": entries}
        path = self.store.path("delivered/index.json")
        if path.exists():
            if self.store.json(self.store.reference(path)) != value:
                raise ValueError("cannot replace completed delivered corpus")
        else:
            storage.write_json(path, value)
        return self.store.reference(path)

    def load_cell(self, delivered_ref, *, arm, checkpoint_seed, check):
        if arm not in ge.ARMS or checkpoint_seed not in ge.CHECKPOINTS:
            raise ValueError("unknown declared arm/checkpoint")
        complete = self.store.json(delivered_ref)
        if (set(complete) != {"schema", "binding", "prepared", "normalizers", "entries"}
                or complete["schema"] != "generative-evidence-delivered-open-v1"
                or complete["binding"] != self.store.binding or complete["prepared"] != self.prepared_ref
                or complete["normalizers"] != self.normalizer_ref):
            raise ValueError("complete delivered corpus provenance differs")
        roster = [(s, cp, i) for s in OPEN_SPLITS for i in range(OPEN_SPLITS[s][0]//512) for cp in ge.CHECKPOINTS]
        entries = complete["entries"]
        if (any(set(e) != {"split", "checkpoint_seed", "shard", "index"} for e in entries)
                or [(e["split"], e["checkpoint_seed"], e["shard"]) for e in entries] != roster):
            raise ValueError("delivered corpus must retain all 27 ordered shards")
        rows, targets = {s: [] for s in OPEN_SPLITS}, []
        for entry in entries:
            if entry["checkpoint_seed"] != checkpoint_seed:
                continue
            check()
            split, shard = entry["split"], entry["shard"]
            index, shard_ref, raw, decoded = self.raw(split, checkpoint_seed, shard)
            value = self.store.json(entry["index"])
            loaded = self._decode(value, self.store.arrays(value["inputs"]), decoded,
                                  self._identity(split, checkpoint_seed, shard, index, shard_ref))
            supervised = cache.unpack_targets(self.store.arrays(index["targets"]), decoded, binding=self.store.binding)
            targets.append({"split": split, "shard": shard, "targets": index["targets"]})
            for i, scene_id in enumerate(decoded["scene_ids"]):
                rows[split].append({"scene_id": scene_id, "identity": decoded["identities"][i],
                    "inputs": loaded["inputs"][arm][i], "partitions": decoded["rows"][i]["partitions"],
                    "targets": supervised[i]["targets"]})
        check()
        binding = {"prepared_binding": self.store.binding, "prepared": self.prepared_ref,
                   "delivered": delivered_ref, "normalizers": self.normalizer_ref,
                   "arm": arm, "checkpoint_seed": checkpoint_seed, "targets": targets}
        return CellData(rows["train"], rows["calibration"], binding=binding)
