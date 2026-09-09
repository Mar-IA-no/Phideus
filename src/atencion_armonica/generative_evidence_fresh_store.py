"""Fresh-test observable storage, with no producer or truth-opening authority.

Explicitly reuse the frozen OPEN codecs without inheriting its supervision
ports or modifying its split policy. The caller authenticates test_freeze,
owns resources and seals all 45 predictions before any later truth access.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import generative_evidence as ge
from . import generative_evidence_cache as cache
from . import generative_evidence_storage as storage
from .generative_evidence_prepared import PreparedStore, SCENE_KEYS
from .generative_evidence_reuse import ROOT
from .partial_compatibility_cache import encoded

TESTS = ("iid", "ood_beta", "ood_polyphony", "deformed_family")
CANONICAL = ROOT/"data/atencion_armonica/generative_evidence_reader_v1/fresh"
TEMPORARY = ROOT/".agent-work/phideus-generative-fresh-store-tests-20260909"
STATUS = "OBSERVABLE_PREPARED_NO_TRUTH_ACCESS"


def validate_observation(observation, scene_id, split):
    if split not in TESTS:
        raise PermissionError("fresh observable storage rejects OPEN or unknown roles")
    if (type(scene_id) is not int or not 0 <= scene_id < 512
            or set(observation) != {"scene_id", "split_seed", "log_f"}
            or type(observation["scene_id"]) is not int or observation["scene_id"] != scene_id
            or type(observation["split_seed"]) is not int
            or observation["split_seed"] != cache.SPLITS[split][1]):
        raise ValueError("fresh observation role or exact identity differs")
    original = np.asarray(observation["log_f"])
    q = original.astype(np.float32)
    ge.law.observable_q32(np.sort(q))
    if not np.array_equal(original, q.astype(np.float64)):
        raise ValueError("fresh observation must preserve exact delivered q32")
    return q


class FreshObservableStore:
    # Only these generic serialization methods are shared. No OPEN producer,
    # supervisor, normalizer, target reader or privileged aggregate is exposed.
    path = PreparedStore.path
    reference = PreparedStore.reference
    json = PreparedStore.json
    arrays = PreparedStore.arrays
    _record = PreparedStore._record
    _publish = PreparedStore._publish
    _new = PreparedStore._new
    save_fit = PreparedStore.save_fit
    save_observables = PreparedStore.save_observables
    load_row = PreparedStore.load_row

    def __init__(self, root, *, binding):
        cache._binding(binding)
        ref = binding.get("test_freeze")
        if (set(binding) != {"test_freeze"} or not isinstance(ref, dict)
                or set(ref) != {"path", "sha256"} or not isinstance(ref["path"], str)
                or not ref["path"] or not isinstance(ref["sha256"], str)
                or len(ref["sha256"]) != 64 or any(c not in "0123456789abcdef" for c in ref["sha256"])):
            raise ValueError("observable store requires its caller-authenticated test freeze reference")
        self.binding = json.loads(encoded(binding))
        self.root = Path(root).resolve()
        if self.root != CANONICAL and not (self.root.is_relative_to(TEMPORARY) and self.root != TEMPORARY):
            raise ValueError("fresh store requires its canonical or dedicated fixture root")
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root/"binding.json"
        if path.exists():
            if path.is_symlink() or path.read_bytes() != encoded(self.binding):
                raise ValueError("fresh observable store binding differs")
        else:
            if any(self.root.iterdir()):
                raise ValueError("cannot adopt an unbound existing fresh store")
            storage.write_json(path, self.binding)

    def _folder(self, split, scene_id):
        if split not in TESTS:
            raise PermissionError("fresh observable storage rejects OPEN or unknown roles")
        if type(scene_id) is not int or not 0 <= scene_id < 512:
            raise ValueError("fresh scene outside its fixed 512-scene roster")
        return self.path(f"{split}/{scene_id:05d}")

    def _scene(self, split, scene_id, scene):
        self._folder(split, scene_id)
        if set(scene) != SCENE_KEYS:
            raise ValueError("fresh observable scene schema differs or includes privileged data")
        q = validate_observation(scene["observation"], scene_id, split)
        ps = ge.partitions_checked(scene["partitions"], len(q))
        order = np.argsort(q, kind="stable")
        if (not np.array_equal(q[order], scene["q32"])
                or not np.array_equal(order, scene["canonical_to_observed"])
                or scene["status"] != ("ELIGIBLE" if ps else "NO_OBSERVABLE_CANDIDATE")
                or set(scene["logits"]) != set(ge.CHECKPOINTS)):
            raise ValueError("fresh observable universe or coordinate system differs")
        return q, ps

    def load_fit(self, split, scene_id):
        record, ref = self._record(split, scene_id, "fit")
        artifact = record["artifact"]
        value = storage.read_scene(self.path(artifact["path"]), {k: v for k, v in artifact.items() if k != "path"})
        validate_observation(value["observation"], scene_id, split)
        return value, ref

    def _records(self, split):
        records = []
        for scene_id in range(512):
            fitted, fit_ref = self._record(split, scene_id, "fit")
            artifact = fitted["artifact"]
            # Reauthenticate the complete compressed bytes without repeatedly
            # decoding every factor tree for each checkpoint's aggregate.
            actual = self.reference(self.path(artifact["path"]))
            if actual != {k: artifact[k] for k in ("path", "sha256", "bytes")}:
                raise ValueError("fresh aggregate lost its complete fit artifact")
            observed, observed_ref = self._record(split, scene_id, "observable")
            if observed["fit"] != fit_ref:
                raise ValueError("fresh observable receipt no longer binds its fit")
            records.append({"fit": fit_ref, "observable": observed_ref})
        return records

    def _index(self, split):
        self._folder(split, 0)
        ref = self.reference(self.path(f"{split}/observables.json"))
        value = self.json(ref)
        if (set(value) != {"schema", "binding", "status", "split", "split_seed", "scene_ids", "records", "raw"}
                or value["schema"] != "generative-evidence-fresh-observables-v1"
                or value["binding"] != self.binding or value["status"] != STATUS
                or value["split"] != split or value["split_seed"] != cache.SPLITS[split][1]
                or value["scene_ids"] != list(range(512))
                or value["records"] != self._records(split)
                or set(value["raw"]) != {str(cp) for cp in ge.CHECKPOINTS}):
            raise ValueError("fresh observable aggregate identity or full roster differs")
        return value, ref

    def load_observables(self, split, checkpoint_seed):
        if checkpoint_seed not in ge.CHECKPOINTS:
            raise ValueError("unknown fresh checkpoint")
        index, _ = self._index(split)
        decoded = cache.unpack_rows(self.arrays(index["raw"][str(checkpoint_seed)]),
                                    binding=self.binding, split=split, checkpoint_seed=checkpoint_seed)
        if decoded["scene_ids"] != list(range(512)):
            raise ValueError("fresh aggregate lacks the fixed scene roster")
        for scene_id, observation, identity in zip(decoded["scene_ids"], decoded["observations"], decoded["identities"]):
            validate_observation(observation, scene_id, split)
            record, _ = self._record(split, scene_id, "observable")
            if identity != record["identity"]:
                raise ValueError("fresh aggregate changed a scene candidate identity")
        return decoded

    def seal_observables(self, split, *, check):
        if not callable(check):
            raise TypeError("fresh observable aggregation needs a resource callback")
        self._folder(split, 0)
        path = self.path(f"{split}/observables.json")
        if path.exists():
            for cp in ge.CHECKPOINTS:
                check()
                self.load_observables(split, cp)
            return self.reference(path)
        check()
        records = self._records(split)  # Missing scenes fail before publishing aggregate arrays.
        raw, identities = {}, None
        for cp in ge.CHECKPOINTS:
            check()
            values = []
            for scene_id in range(512):
                check()
                values.append(self.load_row(split, scene_id, cp))
            current = [v["identities"][0] for v in values]
            if identities is not None and current != identities:
                raise ValueError("fresh checkpoint candidate identities differ")
            identities = current
            arrays = cache.pack_rows(split, cp, [v["observations"][0] for v in values],
                                     [v["rows"][0] for v in values], binding=self.binding)
            member = self._new(path.parent, f"raw-{cp}", ".npz")
            receipt = storage.write_arrays(member, arrays)
            cache.unpack_rows(storage.read_arrays(member, receipt), binding=self.binding,
                              split=split, checkpoint_seed=cp)
            raw[str(cp)] = {"path": member.relative_to(self.root).as_posix(), **receipt}
        check()
        if self._records(split) != records:
            raise ValueError("fresh scene receipts changed during aggregation")
        storage.write_json(path, {"schema": "generative-evidence-fresh-observables-v1",
            "binding": self.binding, "status": STATUS, "split": split,
            "split_seed": cache.SPLITS[split][1], "scene_ids": list(range(512)),
            "records": records, "raw": raw})
        return self.reference(path)
