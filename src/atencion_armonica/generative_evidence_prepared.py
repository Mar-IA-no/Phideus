"""Recoverable OPEN preparation artifacts and complete compact split readers.

No fitter, model, generator, CUDA or sidecar reader is invoked here. The stage
operator owns resource limits and authenticates input/sidecar provenance. This
store preserves completed fit/observable/supervision boundaries separately;
unreferenced files after interruption remain recoverable, never overwritten.
"""
from __future__ import annotations

import hashlib
import json
from math import comb
from pathlib import Path
import uuid

import numpy as np

from . import generative_evidence as ge
from . import generative_evidence_cache as cache
from . import generative_evidence_storage as storage
from .generative_evidence_reuse import OPEN_SPLITS, ROOT, validate_observation
from .generative_evidence_normalization import arrays_digest
from .learned_partition_metrics import METRICS
from .partial_compatibility_cache import encoded
from .structured_source_artifacts import safe_member

SCENE_KEYS = {"observation", "features", "logits", "pools", "inventory", "partitions",
              "canonical_to_observed", "q32", "status"}


class PreparedStore:
    """An artifact store, not permission to produce data or train a cell."""
    def __init__(self, root, *, binding):
        cache._binding(binding)
        self.root = Path(root).resolve()
        canonical = ROOT/"data/atencion_armonica/generative_evidence_reader_v1"
        temporary = ROOT/".agent-work/phideus-generative-evidence-20260909"
        if self.root != canonical and not (self.root.is_relative_to(temporary) and self.root != temporary):
            raise ValueError("preparation store must use its canonical or owned temporary root")
        self.binding = json.loads(encoded(binding))
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root/"binding.json"
        if path.exists():
            if path.is_symlink() or json.loads(path.read_bytes()) != self.binding:
                raise ValueError("preparation store binding differs")
        else:
            if any(self.root.iterdir()):
                raise ValueError("cannot adopt an unbound existing directory")
            storage.write_json(path, self.binding)

    def path(self, relative):
        path = safe_member(self.root, relative)
        if any(p.is_symlink() for p in [path, *path.parents] if p.is_relative_to(self.root)):
            raise ValueError("prepared artifacts cannot traverse symlinks")
        return path

    def reference(self, path):
        path = self.path(Path(path).relative_to(self.root).as_posix())
        raw = path.read_bytes()
        return {"path": path.relative_to(self.root).as_posix(), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}

    def json(self, ref):
        if set(ref) != {"path", "sha256", "bytes"}:
            raise ValueError("invalid prepared JSON reference")
        raw = self.path(ref["path"]).read_bytes()
        if len(raw) != ref["bytes"] or hashlib.sha256(raw).hexdigest() != ref["sha256"]:
            raise ValueError("prepared JSON bytes differ")
        value = json.loads(raw)
        if encoded(value) != raw:
            raise ValueError("prepared JSON is not canonical")
        return value

    def arrays(self, ref):
        if set(ref) != {"path", "sha256", "bytes"}:
            raise ValueError("invalid prepared array reference")
        return storage.read_arrays(self.path(ref["path"]), {k: ref[k] for k in ("sha256", "bytes")})

    def _folder(self, split, scene_id):
        if split not in OPEN_SPLITS:
            raise PermissionError("OPEN preparation cannot generate or admit a fresh test")
        if type(scene_id) is not int or not 0 <= scene_id < OPEN_SPLITS[split][0]:
            raise ValueError("scene outside the declared OPEN split")
        return self.path(f"{split}/{scene_id:05d}")

    def _record(self, split, scene_id, kind):
        path = self._folder(split, scene_id)/f"{kind}.json"
        ref = self.reference(path)
        value = self.json(ref)
        if (value.get("schema") != f"generative-evidence-{kind}-v1" or value.get("binding") != self.binding
                or value.get("split") != split or value.get("scene_id") != scene_id):
            raise ValueError("scene receipt identity/binding differs")
        return value, ref

    def _publish(self, split, scene_id, kind, value):
        path = self._folder(split, scene_id)/f"{kind}.json"
        storage.write_json(path, {"schema": f"generative-evidence-{kind}-v1", "binding": self.binding,
                                 "split": split, "scene_id": scene_id, **value})
        return self.reference(path)

    def _new(self, folder, kind, suffix):
        folder.mkdir(parents=True, exist_ok=True)
        return folder/f"{kind}-{uuid.uuid4().hex}{suffix}"

    def _scene(self, split, scene_id, scene):
        self._folder(split, scene_id)
        if set(scene) != SCENE_KEYS:
            raise ValueError("observable scene schema differs or includes privileged data")
        q = validate_observation(scene["observation"], scene_id, split)
        ps = ge.partitions_checked(scene["partitions"], len(q))
        if (not np.array_equal(q[np.argsort(q, kind="stable")], scene["q32"])
                or not np.array_equal(np.argsort(q, kind="stable"), scene["canonical_to_observed"])
                or scene["status"] != ("ELIGIBLE" if ps else "NO_OBSERVABLE_CANDIDATE")
                or set(scene["logits"]) != set(ge.CHECKPOINTS)):
            raise ValueError("observable scene universe or coordinate system differs")
        return q, ps

    def save_fit(self, split, scene_id, scene, fitted, *, origin):
        q, ps = self._scene(split, scene_id, scene)
        cache._binding(origin)
        if set(fitted) != {"fits", "group_factors"}:
            raise ValueError("fit archive must preserve fits and complete group factors")
        ge.candidate_channel(ps, fitted["fits"], len(q))
        expected = {(b, g) for p in ps for b in ge.BRANCHES if len(p) in ge.law.BRANCHES[b][2] for g in p}
        seen = set()
        for row in fitted["group_factors"]:
            key = row["branch"], tuple(row["group"])
            factor = row["factor"]
            if (set(row) != {"branch", "group", "factor"} or key in seen or key not in expected
                    or factor["branch"] != key[0] or factor["size"] != len(key[1])
                    or any(len(factor[r]) != comb(8, len(key[1])) for r in ("coarse", "fine"))):
                raise ValueError("fit factors lost branches, groups or assignments")
            seen.add(key)
        if seen != expected:
            raise ValueError("fit factor roster is incomplete")
        folder = self._folder(split, scene_id)
        if (folder/"fit.json").exists():
            saved, ref = self.load_fit(split, scene_id)
            expected_value = {"observation": scene["observation"], "inventory": scene["inventory"], **fitted}
            old = self.json(ref)
            if encoded(saved) != encoded(expected_value) or old["origin"] != origin:
                raise ValueError("cannot replace a completed fit")
            return ref
        value = {"observation": scene["observation"], "inventory": scene["inventory"], **fitted}
        path = self._new(folder, "factors", ".json.gz")
        receipt = storage.write_scene(path, value)
        storage.read_scene(path, receipt)
        return self._publish(split, scene_id, "fit", {"origin": origin,
            "artifact": {"path": path.relative_to(self.root).as_posix(), **receipt}})

    def load_fit(self, split, scene_id):
        record, ref = self._record(split, scene_id, "fit")
        artifact = record["artifact"]
        value = storage.read_scene(self.path(artifact["path"]), {k: v for k, v in artifact.items() if k != "path"})
        validate_observation(value["observation"], scene_id, split)
        return value, ref

    def save_observables(self, split, scene_id, scene):
        q, ps = self._scene(split, scene_id, scene)
        fitted, fit_ref = self.load_fit(split, scene_id)
        if (fitted["observation"] != scene["observation"]
                or encoded(fitted["inventory"]) != encoded(scene["inventory"])):
            raise ValueError("preserved fit belongs to another observation or universe")
        folder = self._folder(split, scene_id)
        f = scene["features"]
        input_digest = arrays_digest({"q": q, "triples": f["triples"], "residual_cents": f["residual_cents"],
                                      **{str(cp): scene["logits"][cp] for cp in ge.CHECKPOINTS}})
        if (folder/"observable.json").exists():
            record, ref = self._record(split, scene_id, "observable")
            if record["fit"] != fit_ref or record["input_digest"] != input_digest:
                raise ValueError("observable receipt points to another fit or input")
            for cp in ge.CHECKPOINTS:
                self.load_row(split, scene_id, cp)
            return ref
        refs, identity = {}, None
        for cp in ge.CHECKPOINTS:
            row = ge.observable_rows(q, scene["logits"][cp], f["triples"], f["residual_cents"], ps, fitted["fits"])
            arrays = cache.pack_rows(split, cp, [scene["observation"]], [row], binding=self.binding)
            path = self._new(folder, f"raw-{cp}", ".npz")
            receipt = storage.write_arrays(path, arrays)
            decoded = cache.unpack_rows(storage.read_arrays(path, receipt), binding=self.binding, split=split, checkpoint_seed=cp)
            current = decoded["identities"][0]
            if identity is not None and current != identity:
                raise ValueError("checkpoint candidate identities differ")
            identity = current
            refs[str(cp)] = {"path": path.relative_to(self.root).as_posix(), **receipt}
        return self._publish(split, scene_id, "observable", {"fit": fit_ref, "raw": refs,
            "identity": identity, "candidate_count": len(ps), "status": scene["status"], "input_digest": input_digest})

    def load_row(self, split, scene_id, checkpoint_seed):
        if checkpoint_seed not in ge.CHECKPOINTS:
            raise ValueError("unknown checkpoint")
        record, _ = self._record(split, scene_id, "observable")
        if set(record["raw"]) != {str(cp) for cp in ge.CHECKPOINTS}:
            raise ValueError("observable checkpoint roster differs")
        result = cache.unpack_rows(self.arrays(record["raw"][str(checkpoint_seed)]), binding=self.binding,
                                   split=split, checkpoint_seed=checkpoint_seed)
        if (result["scene_ids"] != [scene_id] or result["identities"] != [record["identity"]]
                or len(result["rows"][0]["partitions"]) != record["candidate_count"]):
            raise ValueError("compact row does not match its scene receipt")
        return result

    def save_supervision(self, split, scene_id, supervision, *, identity, source):
        """Caller supplies supervision computed from a verified OPEN sidecar."""
        cache._binding(source)
        decoded = self.load_row(split, scene_id, ge.CHECKPOINTS[0])
        _, observable_ref = self._record(split, scene_id, "observable")
        if decoded["identities"] != [identity] or set(supervision) != {"raw", "targets", "metrics"}:
            raise ValueError("supervision identity/schema differs")
        arrays = cache.pack_targets([identity], [supervision], binding=self.binding)
        cache.unpack_targets(arrays, decoded, binding=self.binding)
        if (len(supervision["metrics"]) != len(decoded["rows"][0]["partitions"])
                or any(set(m) != set(METRICS) or not np.isfinite(list(m.values())).all() for m in supervision["metrics"])):
            raise ValueError("supervised candidate metric roster differs")
        folder = self._folder(split, scene_id)
        if (folder/"supervision.json").exists():
            old, ref = self._record(split, scene_id, "supervision")
            if old["observable"] != observable_ref or old["source"] != source:
                raise ValueError("supervision source changed on recovery")
            loaded = self.load_supervision(split, scene_id)
            if (any(not np.array_equal(loaded[k], supervision[k]) for k in ("raw", "targets"))
                    or encoded(loaded["metrics"]) != encoded(supervision["metrics"])):
                raise ValueError("cannot replace completed supervision")
            return ref
        path = self._new(folder, "targets", ".npz")
        receipt = storage.write_arrays(path, arrays)
        cache.unpack_targets(storage.read_arrays(path, receipt), decoded, binding=self.binding)
        metrics = self._new(folder, "metrics", ".json")
        storage.write_json(metrics, {"identity": identity, "metrics": supervision["metrics"]})
        return self._publish(split, scene_id, "supervision", {"observable": observable_ref, "source": source,
            "targets": {"path": path.relative_to(self.root).as_posix(), **receipt}, "metrics": self.reference(metrics)})

    def load_supervision(self, split, scene_id):
        record, _ = self._record(split, scene_id, "supervision")
        _, observable_ref = self._record(split, scene_id, "observable")
        if record["observable"] != observable_ref:
            raise ValueError("supervision/observable parent changed")
        decoded = self.load_row(split, scene_id, ge.CHECKPOINTS[0])
        target = cache.unpack_targets(self.arrays(record["targets"]), decoded, binding=self.binding)[0]
        metrics = self.json(record["metrics"])
        if metrics["identity"] != decoded["identities"][0] or len(metrics["metrics"]) != len(target["targets"]):
            raise ValueError("candidate metrics identity differs")
        return {**target, "metrics": metrics["metrics"]}

    def _shard_path(self, split, shard):
        self._folder(split, 0)
        if type(shard) is not int or not 0 <= shard < OPEN_SPLITS[split][0]//512:
            raise ValueError("shard outside the complete OPEN roster")
        return self.path(f"{split}/shard_{shard}/index.json")

    def shard_index(self, split, shard):
        path = self._shard_path(split, shard)
        ref = self.reference(path)
        index = self.json(ref)
        ids = list(range(shard*512, (shard+1)*512))
        if (set(index) != {"schema", "binding", "split", "shard", "scene_ids", "records", "raw", "targets", "metrics"}
                or index["schema"] != "generative-evidence-prepared-shard-v1" or index["binding"] != self.binding
                or index["split"] != split or index["shard"] != shard or index["scene_ids"] != ids
                or set(index["raw"]) != {str(cp) for cp in ge.CHECKPOINTS}
                or len(index["records"]) != 512):
            raise ValueError("prepared shard index identity or roster differs")
        for i, records in zip(ids, index["records"]):
            if set(records) != {"fit", "observable", "supervision"}:
                raise ValueError("scene preparation boundary missing from shard")
            for kind, r in records.items():
                v = self.json(r)
                if (v["binding"] != self.binding or v["split"] != split or v["scene_id"] != i
                        or v["schema"] != f"generative-evidence-{kind}-v1"):
                    raise ValueError("prepared shard scene parent differs")
        return index, ref

    def raw_shard(self, split, checkpoint_seed, shard):
        """Observable-only port for full TRAIN normalization; no target parsing."""
        if checkpoint_seed not in ge.CHECKPOINTS:
            raise ValueError("unknown checkpoint")
        index, _ = self.shard_index(split, shard)
        arrays = self.arrays(index["raw"][str(checkpoint_seed)])
        decoded = cache.unpack_rows(arrays, binding=self.binding, split=split, checkpoint_seed=checkpoint_seed)
        if decoded["scene_ids"] != index["scene_ids"]:
            raise ValueError("raw shard scene order differs")
        return arrays

    def load_shard(self, split, checkpoint_seed, shard):
        index, _ = self.shard_index(split, shard)
        raw = self.raw_shard(split, checkpoint_seed, shard)
        decoded = cache.unpack_rows(raw, binding=self.binding, split=split, checkpoint_seed=checkpoint_seed)
        targets = cache.unpack_targets(self.arrays(index["targets"]), decoded, binding=self.binding)
        metrics = self.json(index["metrics"])
        if (set(metrics) != {"identities", "metrics"} or metrics["identities"] != decoded["identities"]
                or len(metrics["metrics"]) != 512
                or any(len(ms) != len(row["partitions"]) for ms, row in zip(metrics["metrics"], decoded["rows"]))):
            raise ValueError("packed candidate metrics roster differs")
        return {**decoded, "targets": targets, "metrics": metrics["metrics"]}

    def seal_shard(self, split, shard):
        path = self._shard_path(split, shard)
        if path.exists():
            for cp in ge.CHECKPOINTS:
                self.load_shard(split, cp, shard)
            return self.reference(path)
        ids = list(range(shard*512, (shard+1)*512))
        records = [{k: self._record(split, i, k)[1] for k in ("fit", "observable", "supervision")} for i in ids]
        raw_refs, identities, decoded_first = {}, None, None
        for cp in ge.CHECKPOINTS:
            values = [self.load_row(split, i, cp) for i in ids]
            current = [v["identities"][0] for v in values]
            if identities is not None and current != identities:
                raise ValueError("packed checkpoint candidate identities differ")
            identities = current
            arrays = cache.pack_rows(split, cp, [v["observations"][0] for v in values],
                                     [v["rows"][0] for v in values], binding=self.binding)
            member = self._new(path.parent, f"raw-{cp}", ".npz")
            receipt = storage.write_arrays(member, arrays)
            decoded = cache.unpack_rows(storage.read_arrays(member, receipt), binding=self.binding,
                                         split=split, checkpoint_seed=cp)
            if cp == ge.CHECKPOINTS[0]:
                decoded_first = decoded
            raw_refs[str(cp)] = {"path": member.relative_to(self.root).as_posix(), **receipt}
        supervision = [self.load_supervision(split, i) for i in ids]
        member = self._new(path.parent, "targets", ".npz")
        receipt = storage.write_arrays(member, cache.pack_targets(identities, supervision, binding=self.binding))
        cache.unpack_targets(storage.read_arrays(member, receipt), decoded_first, binding=self.binding)
        targets = {"path": member.relative_to(self.root).as_posix(), **receipt}
        member = self._new(path.parent, "metrics", ".json")
        storage.write_json(member, {"identities": identities, "metrics": [s["metrics"] for s in supervision]})
        # Bind every completed scene receipt before publishing the aggregate.
        for scene_records in records:
            for ref in scene_records.values():
                self.json(ref)
        storage.write_json(path, {"schema": "generative-evidence-prepared-shard-v1", "binding": self.binding,
            "split": split, "shard": shard, "scene_ids": ids, "records": records,
            "raw": raw_refs, "targets": targets, "metrics": self.reference(member)})
        return self.reference(path)

    def seal_split(self, split):
        self._folder(split, 0)
        # A scene-receipt roster alone does not prove the aggregate payloads
        # still exist and match. Reuse the full shard validation at this seal.
        refs = [self.seal_shard(split, shard) for shard in range(OPEN_SPLITS[split][0]//512)]
        value = {"schema": "generative-evidence-prepared-split-v1", "binding": self.binding,
                 "split": split, "split_seed": OPEN_SPLITS[split][1], "count": OPEN_SPLITS[split][0], "shards": refs}
        path = self.path(f"{split}/index.json")
        if path.exists():
            if self.json(self.reference(path)) != value:
                raise ValueError("cannot replace a completed split")
        else:
            storage.write_json(path, value)
        return self.reference(path)
