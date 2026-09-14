"""Durable eight-channel OPEN adaptation without refitting or backbone forwards.

The new store holds only added/changed evidence and its diagnostic attribution;
common inputs and supervision retain authenticated references to the old store.
Resource admission, locks and the final frozen completion receipt are external.
"""
from __future__ import annotations

import uuid

import numpy as np

from .generative_evidence import CHECKPOINTS
from .generative_evidence_normalization import arrays_digest
from .generative_evidence_reuse import OPEN_SPLITS
from .geometric_decision_cell import CellData
from .partial_compatibility_cache import encoded

ROSTER = tuple((split, cp, shard) for split in OPEN_SPLITS
               for shard in range(OPEN_SPLITS[split][0]//512) for cp in CHECKPOINTS)


def source_identity(source):
    return {"root": str(source.store.root), "binding": source.store.binding,
            "prepared": source.corpus.prepared_ref, "delivered": source.delivered_ref,
            "normalizers": source.corpus.normalizer_ref}


class OpenPreparation:
    def __init__(self, source, store):
        if store.binding.get("source") != source_identity(source):
            raise ValueError("preparation store must bind the exact authenticated OPEN source")
        self.source, self.store = source, store

    def _identity(self, observable, cp, scale_ref, interface):
        return {"schema": "geometric-decision-delivered-shard-v1", "binding": self.store.binding,
                "split": observable["split"], "shard": observable["shard"], "checkpoint_seed": cp,
                "scale": scale_ref, "scene_ids": observable["scene_ids"],
                "identities": observable["identities"], "source": observable["source_refs"][str(cp)],
                "sham": [r["sham"] for r in interface]}

    @staticmethod
    def _arrays(interface):
        counts = [len(r["inputs"]["evidence"]) for r in interface]
        return {"offsets": np.r_[np.int64(0), np.cumsum(counts, dtype=np.int64)],
                "evidence": np.concatenate([r["inputs"]["evidence"] for r in interface]),
                "decoupled_six": np.concatenate([r["diagnostics"]["decoupled_six"] for r in interface])}

    def _verify_entry(self, ref, identity, expected):
        record = self.store.json(ref)
        if (set(record) != set(identity) | {"arrays", "array_digest"}
                or any(encoded(record[k]) != encoded(v) for k, v in identity.items())):
            raise ValueError("prepared shard differs from exact source, scale or scene identities")
        arrays = self.store.arrays(record["arrays"])
        if (arrays_digest(arrays) != record["array_digest"]
                or arrays_digest(arrays) != arrays_digest(expected)):
            raise ValueError("prepared evidence differs from recomputed authenticated interface")
        return arrays

    def prepare(self, *, check, progress=print):
        """Publish complete OPEN adaptation; this does not train or parse targets."""
        source, store = self.source, self.store
        check()
        # The scale record is immutable and source-bound. On recovery its stored
        # bytes are reused; the completed manifest will pin its exact receipt.
        path = store.path("scale.json")
        if path.exists():
            scale_ref = store.reference(path)
            scale = store.json(scale_ref)
        else:
            scale = source.geometric_scale(check=check)
            scale_ref = store.publish_json("scale.json", scale)
        entries = []
        for split in OPEN_SPLITS:
            for shard in range(OPEN_SPLITS[split][0]//512):
                check()
                observable = source.observable_shard(split, shard, check=check)
                interfaces = source.interface_shard(observable, scale)
                for cp in CHECKPOINTS:
                    check()
                    interface = interfaces[cp]
                    identity = self._identity(observable, cp, scale_ref, interface)
                    arrays = self._arrays(interface)
                    folder = f"delivered/{split}/cp_{cp}/shard_{shard}"
                    path = store.path(f"{folder}/index.json")
                    if path.exists():
                        ref = store.reference(path)
                    else:
                        blob = store.publish_arrays(f"{folder}/inputs-{uuid.uuid4().hex}.npz", arrays)
                        ref = store.publish_json(f"{folder}/index.json",
                            {**identity, "arrays": blob, "array_digest": arrays_digest(arrays)})
                    self._verify_entry(ref, identity, arrays)
                    entries.append({"split": split, "checkpoint_seed": cp, "shard": shard, "index": ref})
                    progress({"stage": "open-adaptation", "split": split, "checkpoint_seed": cp, "shard": shard})
        check()
        return store.publish_json("complete.json", {"schema": "geometric-decision-open-prepared-v1",
            "binding": store.binding, "scale": scale_ref, "entries": entries,
            "supervision": "not parsed; authenticated source target port remains separate"})

    def completion(self, ref):
        record = self.store.json(ref)
        if (set(record) != {"schema", "binding", "scale", "entries", "supervision"}
                or record["schema"] != "geometric-decision-open-prepared-v1" or record["binding"] != self.store.binding
                or record["supervision"] != "not parsed; authenticated source target port remains separate"
                or any(set(e) != {"split", "checkpoint_seed", "shard", "index"} for e in record["entries"])
                or [(e["split"], e["checkpoint_seed"], e["shard"]) for e in record["entries"]] != list(ROSTER)):
            raise ValueError("preparation completion requires all 27 exact ordered entries")
        return record

    def load_checkpoint(self, complete_ref, checkpoint_seed, *, check):
        """Explicit OPEN supervision port, full 4096/512, one backbone at a time."""
        if type(checkpoint_seed) is not int or checkpoint_seed not in CHECKPOINTS:
            raise ValueError("checkpoint outside the declared backbone roster")
        complete = self.completion(complete_ref)
        scale = self.store.json(complete["scale"])
        entries = {(e["split"], e["checkpoint_seed"], e["shard"]): e["index"] for e in complete["entries"]}
        rows, targets = {split: [] for split in OPEN_SPLITS}, []
        for split in OPEN_SPLITS:
            for shard in range(OPEN_SPLITS[split][0]//512):
                check()
                observable = self.source.observable_shard(split, shard, check=check)
                interface = self.source.interface_shard(observable, scale)[checkpoint_seed]
                arrays = self._verify_entry(entries[split, checkpoint_seed, shard],
                    self._identity(observable, checkpoint_seed, complete["scale"], interface), self._arrays(interface))
                supervision = self.source.supervision_shard(observable)
                targets.append({"split": split, "shard": shard, "targets": observable["targets_ref"]})
                for i, scene_id in enumerate(observable["scene_ids"]):
                    first, last = arrays["offsets"][i:i+2]
                    inputs = {**interface[i]["inputs"], "evidence": arrays["evidence"][first:last]}
                    rows[split].append({"scene_id": scene_id, "identity": observable["identities"][i],
                        "inputs": inputs, "partitions": observable["raw"][checkpoint_seed][i]["partitions"],
                        "targets": supervision[i]["targets"]})
        check()
        return CellData(rows["train"], rows["calibration"], binding={"prepared_binding": self.store.binding,
            "completion": complete_ref, "scale": complete["scale"], "checkpoint_seed": checkpoint_seed,
            "supervision": targets})
