"""Fixed fresh draws and an observable-only port, behind the complete freeze.

The sampler writes observation and privileged sidecar separately. Duplicates
are preserved and stop the run. Recovery never repeats an attempted draw;
an incomplete pair requires reconciliation, not a replacement observation.
The caller owns the single-operator lock and remaining test-stage resources.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from . import generative_evidence_cache as cache
from . import generative_evidence_exclusions as exclusions
from . import generative_evidence_storage as storage
from .generative_evidence_fresh_store import TESTS, CANONICAL, validate_observation
from .generative_evidence_reuse import ROOT, VerifiedBytes
from .partial_compatibility_cache import encoded

DRAW_ROOT = CANONICAL/"draws"
TEMPORARY = ROOT/".agent-work/phideus-generative-fresh-data-tests-20260909"
DRAW_SCHEMA = "generative-evidence-fresh-draw-v1"
INDEX_SCHEMA = "generative-evidence-fresh-draw-index-v1"


def _authority(ref, check):
    if not callable(check):
        raise TypeError("fresh data requires a resource callback")
    check()
    from .generative_evidence_test_freeze import verify_freeze
    return verify_freeze(ref, check=check)


class _DrawFiles:
    """File mechanics only; construction is not authority to sample or read truth."""
    def __init__(self, root, freeze_ref):
        self.root = Path(root).resolve()
        if self.root != DRAW_ROOT and not (self.root.is_relative_to(TEMPORARY) and self.root != TEMPORARY):
            raise ValueError("fresh draw root must be canonical or a dedicated fixture")
        self.freeze = json.loads(encoded(freeze_ref))
        self.reader = VerifiedBytes(self.root)

    def path(self, name):
        from .structured_source_artifacts import safe_member
        path = safe_member(self.root, name)
        if path.is_symlink():
            raise ValueError("fresh data must not use symlinks")
        return path

    def reference(self, path):
        path = self.path(Path(path).relative_to(self.root).as_posix())
        return {"path": path.relative_to(self.root).as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    def folder(self, split, scene_id):
        if split not in TESTS or type(scene_id) is not int or not 0 <= scene_id < 512:
            raise ValueError("fresh draw outside fixed role and scene roster")
        return self.path(f"{split}/{scene_id:05d}")

    def intent(self, split, scene_id):
        self.folder(split, scene_id)
        return {"schema": DRAW_SCHEMA, "test_freeze": self.freeze, "split": split,
                "split_seed": cache.SPLITS[split][1], "scene_id": scene_id}

    def reopen(self, split, scene_id, frozen, produced):
        folder = self.folder(split, scene_id)
        intent = self.intent(split, scene_id)
        if self.reader.json(self.reference(folder/"intent.json")) != intent:
            raise ValueError("draw intent differs from frozen identity")
        # Byte authentication only for the privileged port: never json(sidecar).
        obs_ref, truth_ref = [self.reference(folder/name) for name in ("observation.json", "sidecar.json")]
        observation = self.reader.json(obs_ref)
        validate_observation(observation, scene_id, split)
        self.reader.read(truth_ref)
        match = exclusions.duplicate_matches(observation["log_f"], frozen, produced)
        expected = {**intent, "observation": obs_ref, "sidecar": truth_ref, "duplicate_check": match,
                    "status": "DUPLICATE_PRESERVED" if match["duplicate"] else "DRAW_PRESERVED"}
        receipt_path = folder/"draw.json"
        if receipt_path.exists():
            if self.reader.json(self.reference(receipt_path)) != expected:
                raise ValueError("draw receipt differs from preserved payloads or duplicate history")
        return observation, expected, receipt_path

    def index(self, split, frozen, produced, check):
        self.folder(split, 0)
        path = self.path(f"{split}/index.json")
        ref = self.reference(path)
        value = self.reader.json(ref)
        records, hashes = [], list(produced)
        for scene_id in range(512):
            check()
            _, expected, receipt_path = self.reopen(split, scene_id, frozen, hashes)
            if expected["status"] != "DRAW_PRESERVED":
                raise RuntimeError("duplicate-preserving stop is not a complete test")
            records.append(self.reference(receipt_path))
            hashes.append(expected["duplicate_check"]["fingerprint"])
        wanted = {"schema": INDEX_SCHEMA, "test_freeze": self.freeze, "split": split,
                  "split_seed": cache.SPLITS[split][1], "scene_ids": list(range(512)),
                  "records": records, "status": "DRAWN_NO_TRUTH_ACCESS"}
        if value != wanted:
            raise ValueError("fresh data index differs from its complete immutable roster")
        return ref, value, hashes

    def earlier(self, split, frozen, check):
        self.folder(split, 0)
        hashes = []
        for previous in TESTS[:TESTS.index(split)]:
            _, _, hashes = self.index(previous, frozen, hashes, check)
        return hashes


def _produce_verified(files, split, frozen, *, check, draw):
    """Mechanics shared by the public gated runner and no-sampler fixtures."""
    produced = files.earlier(split, frozen, check)
    index_path = files.path(f"{split}/index.json")
    if index_path.exists():
        return files.index(split, frozen, produced, check)[0]
    records = []
    for scene_id in range(512):
        check()
        folder = files.folder(split, scene_id)
        intent_path = folder/"intent.json"
        if not intent_path.exists():
            if folder.exists() and any(folder.iterdir()):
                raise RuntimeError("unbound draw files require reconciliation; no draw attempted")
            # No hole may be filled with a new draw after later IDs exist.
            if any(files.folder(split, i).exists() for i in range(scene_id+1, 512)):
                raise RuntimeError("noncontiguous draw prefix requires reconciliation")
            folder.mkdir(parents=True, exist_ok=True)
            storage.write_json(intent_path, files.intent(split, scene_id))
            observation, sidecar = draw(split, scene_id, cache.SPLITS[split][1])
            storage.write_json(folder/"observation.json", observation)
            storage.write_json(folder/"sidecar.json", sidecar)
        if not all((folder/name).is_file() for name in ("observation.json", "sidecar.json")):
            raise RuntimeError("attempted draw has an incomplete pair; never redraw or replace")
        _, receipt, receipt_path = files.reopen(split, scene_id, frozen, produced)
        if not receipt_path.exists():
            storage.write_json(receipt_path, receipt)
        if receipt["status"] == "DUPLICATE_PRESERVED":
            raise RuntimeError(f"duplicate preserved at {split}/{scene_id}; no replacement draw")
        records.append(files.reference(receipt_path))
        produced.append(receipt["duplicate_check"]["fingerprint"])
    check()
    storage.write_json(index_path, {"schema": INDEX_SCHEMA, "test_freeze": files.freeze,
        "split": split, "split_seed": cache.SPLITS[split][1], "scene_ids": list(range(512)),
        "records": records, "status": "DRAWN_NO_TRUTH_ACCESS"})
    # Reopen every recorded byte before advertising a completed observable port.
    return files.index(split, frozen, files.earlier(split, frozen, check), check)[0]


def produce_test(split, *, freeze_ref, check):
    verified = _authority(freeze_ref, check)
    from .learned_partition_data import _draw_scene
    files = _DrawFiles(DRAW_ROOT, verified["freeze"])
    return _produce_verified(files, split, verified["exclusions"], check=check, draw=_draw_scene)


class FreshObservations:
    """Authenticate complete draws; expose observations but no parsed sidecars."""
    def __init__(self, split, *, freeze_ref, check):
        verified = _authority(freeze_ref, check)
        self.files = _DrawFiles(DRAW_ROOT, verified["freeze"])
        previous = self.files.earlier(split, verified["exclusions"], check)
        self.reference, self.index, _ = self.files.index(split, verified["exclusions"], previous, check)
        self.split = split

    def observation(self, scene_id):
        self.files.folder(self.split, scene_id)
        receipt = self.files.reader.json(self.index["records"][scene_id])
        self.files.reader.read(receipt["sidecar"])
        observation = self.files.reader.json(receipt["observation"])
        validate_observation(observation, scene_id, self.split)
        return observation
