"""Read-only adapter of authenticated, previously prepared OPEN aggregates.

No mkdir, source writes, fit, forward, sampler or fresh-test port. The caller
pins all three root receipts before construction. Existing codecs validate
the old six-channel input; the new interface is built separately in memory.
"""
from __future__ import annotations

import hashlib
from io import BytesIO
import json
from pathlib import Path

import numpy as np

from . import generative_evidence as ge
from . import generative_evidence_cache as cache
from .generative_evidence_corpus import TrainingCorpus
from .generative_evidence_reuse import OPEN_SPLITS
from .geometric_decision_core import delivered_interface, fit_scale, geometric_log_cost
from .partial_compatibility_cache import encoded


class ReadOnlyStore:
    def __init__(self, root, *, binding_ref):
        candidate = Path(root)
        if not candidate.is_dir() or candidate.is_symlink():
            raise ValueError("read-only source root must already exist without a symlink")
        self.root = candidate.resolve()
        self.consumed = {}
        self.binding = self.json(binding_ref)

    def path(self, relative):
        value = Path(relative)
        if (type(relative) is not str or not relative or value.is_absolute()
                or ".." in value.parts or value.as_posix() != relative
                or value.suffix not in (".json", ".npz")):
            raise ValueError("invalid source aggregate member")
        result = self.root/value
        if any(p.is_symlink() for p in (result, *result.parents) if p.is_relative_to(self.root)):
            raise ValueError("source aggregate cannot traverse a symlink")
        return result

    def read(self, ref):
        if (not isinstance(ref, dict) or set(ref) != {"path", "sha256", "bytes"}
                or type(ref["sha256"]) is not str or len(ref["sha256"]) != 64
                or any(c not in "0123456789abcdef" for c in ref["sha256"])
                or type(ref["bytes"]) is not int or ref["bytes"] < 0):
            raise ValueError("explicit immutable source receipt required")
        raw = self.path(ref["path"]).read_bytes()
        if len(raw) != ref["bytes"] or hashlib.sha256(raw).hexdigest() != ref["sha256"]:
            raise ValueError("read-only source bytes differ from pinned receipt")
        if self.consumed.setdefault(ref["path"], ref.copy()) != ref:
            raise ValueError("one source path claimed with different identities")
        return raw

    def json(self, ref):
        raw = self.read(ref)
        value = json.loads(raw)
        if encoded(value) != raw:
            raise ValueError("source JSON is not canonical")
        return value

    def arrays(self, ref):
        raw = self.read(ref)
        with np.load(BytesIO(raw), allow_pickle=False) as archive:
            if len(archive.files) != len(set(archive.files)):
                raise ValueError("duplicate aggregate array names")
            return {key: archive[key] for key in archive.files}


class OpenSource:
    def __init__(self, root, *, binding_ref, prepared_ref, delivered_ref):
        self.store = ReadOnlyStore(root, binding_ref=binding_ref)
        self.corpus = TrainingCorpus(self.store, prepared_ref)
        self.delivered_ref = json.loads(encoded(delivered_ref))
        complete = self.store.json(delivered_ref)
        if (set(complete) != {"schema", "binding", "prepared", "normalizers", "entries"}
                or complete["schema"] != "generative-evidence-delivered-open-v1"
                or complete["binding"] != self.store.binding
                or complete["prepared"] != self.corpus.prepared_ref
                or complete["normalizers"] != self.corpus.normalizer_ref):
            raise ValueError("OPEN delivered completion has another source binding")
        roster = [(s, cp, shard) for s in OPEN_SPLITS for shard in range(OPEN_SPLITS[s][0]//512)
                  for cp in ge.CHECKPOINTS]
        entries = complete["entries"]
        if (any(set(e) != {"split", "checkpoint_seed", "shard", "index"} for e in entries)
                or [(e["split"], e["checkpoint_seed"], e["shard"]) for e in entries] != roster):
            raise ValueError("OPEN delivered completion requires all 27 ordered entries")
        self.entries = {(e["split"], e["checkpoint_seed"], e["shard"]): e["index"] for e in entries}

    @staticmethod
    def _role(split, shard):
        if split not in OPEN_SPLITS:
            raise PermissionError("this adapter exposes only OPEN train/calibration")
        if type(shard) is not int or not 0 <= shard < OPEN_SPLITS[split][0]//512:
            raise ValueError("OPEN shard outside complete roster")

    def observable_shard(self, split, shard, *, check):
        self._role(split, shard)
        result = {"split": split, "shard": shard, "raw": {}, "inputs": {}, "source_refs": {}}
        first = None
        for cp in ge.CHECKPOINTS:
            check()
            index, shard_ref, _, decoded = self.corpus.raw(split, cp, shard)
            entry_ref = self.entries[split, cp, shard]
            entry = self.store.json(entry_ref)
            delivered = self.corpus._decode(entry, self.store.arrays(entry["inputs"]), decoded,
                self.corpus._identity(split, cp, shard, index, shard_ref))
            if first is None:
                first = decoded
                result.update({k: decoded[k] for k in ("scene_ids", "identities", "observations")})
                result["targets_ref"] = index["targets"].copy()  # Identity only, NOT parsed targets.
            else:
                if any(decoded[k] != first[k] for k in ("scene_ids", "identities", "observations")):
                    raise ValueError("OPEN backbones do not share observation/candidate identities")
                for a, b in zip(first["rows"], decoded["rows"]):
                    for key in ("evidence", "available", "incidence", "canonical_to_observed"):
                        if not np.array_equal(a[key], b[key]):
                            raise ValueError("OPEN geometric evidence differs across backbones")
                if index["targets"] != result["targets_ref"]:
                    raise ValueError("OPEN backbones do not share supervision identity")
            result["raw"][cp] = decoded["rows"]
            result["inputs"][cp] = delivered["inputs"]["generative"]
            result["source_refs"][str(cp)] = {"prepared_shard": shard_ref, "raw": index["raw"][str(cp)],
                "delivered_index": entry_ref, "delivered_arrays": entry["inputs"]}
        check()
        return result

    def supervision_shard(self, observable):
        """Explicit OPEN target port, separate from scale and observable delivery."""
        split, shard = observable["split"], observable["shard"]
        self._role(split, shard)
        index = self.store.json(self.corpus.splits[split]["shards"][shard])
        if index["targets"] != observable["targets_ref"]:
            raise ValueError("supervision reference differs from authenticated OPEN shard")
        decoded = {"identities": observable["identities"], "rows": observable["raw"][ge.CHECKPOINTS[0]]}
        return cache.unpack_targets(self.store.arrays(index["targets"]), decoded, binding=self.store.binding)

    def geometric_scale(self, *, check):
        """Full 4096-scene TRAIN pass, each scene once after cross-backbone check."""
        identities = []
        def costs():
            for shard in range(8):
                observable = self.observable_shard("train", shard, check=check)
                identities.extend(observable["identities"])
                for row in observable["raw"][ge.CHECKPOINTS[0]]:
                    yield geometric_log_cost(row["evidence"], row["available"])
        scale = fit_scale(costs())
        if (len(identities) != 4096 or identities != self.corpus.norm["scene_identities"]
                or scale["eligible_scenes"] != len(self.corpus.norm["eligible_scene_ids"])
                or scale["empty_scenes"] != len(self.corpus.norm["excluded_scene_ids"])):
            raise ValueError("geometric scale does not cover authenticated TRAIN exactly once")
        return {"schema": "geometric-decision-open-scale-v1", "prepared": self.corpus.prepared_ref,
                "delivered": self.delivered_ref, "normalizers": self.corpus.normalizer_ref,
                "scene_identities": identities, "rule": "scene_then_candidate_rms_log_min_available_ub_v1",
                **scale}

    def interface_shard(self, observable, scale):
        if (scale["schema"] != "geometric-decision-open-scale-v1"
                or scale["prepared"] != self.corpus.prepared_ref or scale["delivered"] != self.delivered_ref
                or scale["normalizers"] != self.corpus.normalizer_ref
                or scale["scene_identities"] != self.corpus.norm["scene_identities"]):
            raise ValueError("interface scale does not belong to authenticated TRAIN")
        split, shard = observable["split"], observable["shard"]
        self._role(split, shard)
        result = {}
        for cp in ge.CHECKPOINTS:
            rows = []
            for scene_id, delivered, raw in zip(observable["scene_ids"], observable["inputs"][cp], observable["raw"][cp]):
                inputs, sham = delivered_interface(delivered, raw, scale=scale["scale"],
                    split_seed=OPEN_SPLITS[split][1], scene_id=scene_id)
                rows.append({"scene_id": scene_id, "inputs": inputs, "sham": sham})
            if len(rows) != 512:
                raise ValueError("interface requires the complete observable shard")
            result[cp] = rows
        return result
