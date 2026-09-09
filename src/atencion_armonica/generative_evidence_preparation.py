"""Fixed OPEN preparation workflow, with no test, training or model port.

The supervisor supplies the immutable stage binding and a resource/stop check.
Only an authenticated historical profile can replace a new fit. Completed
scene boundaries survive interruption; labels enter after a shard's complete
observable pass. Public entry points do not accept a reduced scene roster.
"""
from __future__ import annotations

import json
import time

from . import generative_evidence as ge
from . import generative_evidence_storage as storage
from .generative_evidence_reuse import AUTHORIZATION, IMPORT, OPEN_SPLITS, ROOT, OpenReuse, VerifiedBytes
from .generative_evidence_normalization import fit_training_normalizers
from .partial_compatibility_cache import encoded

PROFILE = {"path": ".agent-work/phideus-generative-evidence-20260909/profile-gpu-01/report.json",
           "sha256": "c3144164274e154f2dee4ca93eea81410d68558da623545006d751ee473051ea"}


class ProfileFits:
    """Read the same pinned 32 GPU fits; added source files do not invalidate history."""
    def __init__(self):
        self.reader = VerifiedBytes(ROOT)
        self.report = self.reader.json(PROFILE)
        r = self.report
        if (r["status"] != "MEASURED" or r["device"] != "cuda:0"
                or r["binding"]["scene_ids"] != list(range(32))
                or r["binding"]["split"] != "train" or r["binding"]["split_seed"] != OPEN_SPLITS["train"][1]
                or [s["scene_id"] for s in r["fit_scenes"]] != list(range(32))
                or r["reuse"]["authorization"] != AUTHORIZATION or r["reuse"]["import"] != IMPORT):
            raise ValueError("historical profile identity differs")
        for path, sha in r["binding"]["sources"].items():
            self.reader.read({"path": path, "sha256": sha})

    def origin(self, split, scene_id):
        if split != "train" or scene_id not in range(32):
            return None
        ref = self.report["fit_scenes"][scene_id]["scene"]
        path = (ROOT/PROFILE["path"]).parent/f"scene_{scene_id:05d}.json.gz"
        return {"kind": "PRESERVED_PROFILE_FIT", "profile": PROFILE,
                "artifact": {"path": path.relative_to(ROOT).as_posix(), **ref}}

    def fitted(self, split, scene_id, scene):
        origin = self.origin(split, scene_id)
        if origin is None:
            raise ValueError("scene was not fitted by the historical profile")
        ref = origin["artifact"]
        value = storage.read_scene(ROOT/ref["path"], {k: v for k, v in ref.items() if k != "path"})
        if (value["observation"] != scene["observation"]
                or encoded(value["inventory"]) != encoded(scene["inventory"])):
            raise ValueError("profile fit observation/candidate universe differs")
        return {k: value[k] for k in ("fits", "group_factors")}


class OpenPreparation:
    """One shard in memory; caller guards resources before each scene boundary."""
    def __init__(self, store, *, device, check, progress=print):
        if device not in ("cpu", "cuda:0"):
            raise ValueError("explicit fitting backend required")
        self.store, self.device, self.check, self.progress = store, device, check, progress
        self.reuse, self.profile = OpenReuse(), ProfileFits()
        self.fitter = None  # No CUDA initialization for recovery/profile reuse.

    def _new_origin(self, shard):
        return {"kind": "NEW_OPEN_FIT", "device": self.device,
                "authorization": AUTHORIZATION, "import": IMPORT,
                "data": shard.data.ref, "scored": shard.scored.ref,
                "grid": {"beta_count": 257, "gamma_count": 65, "stride": 4}}

    def _observable(self, shard, scene_id):
        """Private single-scene boundary; the public method fixes all 512 IDs."""
        self.check()
        started = time.monotonic()
        split = shard.split
        scene = shard.scene(scene_id)
        origin = self.profile.origin(split, scene_id) or self._new_origin(shard)
        path = self.store.path(f"{split}/{scene_id:05d}/fit.json")
        reused = path.exists()
        if reused:
            value, ref = self.store.load_fit(split, scene_id)
            if self.store.json(ref)["origin"] != origin:
                raise ValueError("completed fit has another provenance/backend")
            fitted = {k: value[k] for k in ("fits", "group_factors")}
            if value["observation"] != scene["observation"] or encoded(value["inventory"]) != encoded(scene["inventory"]):
                raise ValueError("completed fit differs from authenticated OPEN inputs")
        elif origin["kind"] == "PRESERVED_PROFILE_FIT":
            fitted = self.profile.fitted(split, scene_id, scene)
        else:
            if self.fitter is None:
                self.fitter = ge.law.GroupFitter(ge.law.Grid(257, 65, 4),
                                                device="cuda" if self.device == "cuda:0" else "cpu")
            fitted = ge.law.fit_candidates(scene["q32"], scene["partitions"], self.fitter)
        self.store.save_fit(split, scene_id, scene, fitted, origin=origin)
        # If stopping here, the fit is durable and need never be swept again.
        self.check()
        self.store.save_observables(split, scene_id, scene)
        self.progress(json.dumps({"stage": "observable", "split": split, "scene_id": scene_id,
            "candidates": len(scene["partitions"]), "recovered_fit": reused,
            "fit_origin": origin["kind"], "seconds": time.monotonic()-started}))
        self.check()

    def prepare_shard(self, split, shard_index):
        self.check()
        shard = self.reuse.shard(split, shard_index)  # Reject non-OPEN before writes.
        for scene_id in shard.ids:
            self._observable(shard, scene_id)
        # This is the sole sidecar opening point; observable preparation above
        # never receives these labels. Both splits are declared OPEN development.
        from .generative_evidence_supervision import open_truths, candidate_supervision
        self.check()
        truths = open_truths(shard)
        source = {"authorization": AUTHORIZATION, "import": IMPORT, "data": shard.data.ref,
                  "sidecars": shard.data.reference("sidecars.jsonl")}
        for scene_id, truth in zip(shard.ids, truths):
            self.check()
            row = self.store.load_row(split, scene_id, ge.CHECKPOINTS[0])
            value = candidate_supervision(row["rows"][0]["partitions"], truth["labels"])
            self.store.save_supervision(split, scene_id, value, identity=row["identities"][0], source=source)
        self.check()
        ref = self.store.seal_shard(split, shard_index)
        self.progress(json.dumps({"stage": "shard_complete", "split": split, "shard": shard_index, "index": ref}))
        return ref

    def run(self):
        splits = {}
        for split, (count, _) in OPEN_SPLITS.items():
            for shard in range(count//512):
                self.prepare_shard(split, shard)
            self.check()
            splits[split] = self.store.seal_split(split)
        self.check()
        norm_path = self.store.path("normalizers.json")
        if norm_path.exists():
            normalizers = self.store.json(self.store.reference(norm_path))
            if normalizers["prepared_train"] != splits["train"]:
                raise ValueError("normalizers refer to another complete TRAIN split")
        else:
            def load(cp, shard):
                self.check()
                return self.store.raw_shard("train", cp, shard)
            normalizers = {"prepared_train": splits["train"], "normalizers":
                           fit_training_normalizers(load, binding=self.store.binding)}
            storage.write_json(norm_path, normalizers)
        self.check()
        value = {"schema": "generative-evidence-open-preparation-v1", "binding": self.store.binding,
                 "status": "OPEN_PREPARED_NOT_TRAINED_NO_TEST_ACCESS", "splits": splits,
                 "normalizers": self.store.reference(norm_path), "reuse": self.reuse.receipt(),
                 "profile": PROFILE}
        # Reuse ledger on recovery records the same authenticated inputs; there
        # are no sidecar skips hidden behind previously completed shards.
        path = self.store.path("open_prepared.json")
        if path.exists():
            if self.store.json(self.store.reference(path)) != value:
                raise ValueError("OPEN preparation completion changed on recovery")
        else:
            storage.write_json(path, value)
        return self.store.reference(path)
