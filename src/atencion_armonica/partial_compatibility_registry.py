"""Verify the complete training roster before checkpoint inference, without Torch."""
from __future__ import annotations

import json
from pathlib import Path

from .partial_compatibility_cache import sha_file

ARM_NAMES = ("pairs_descriptors", "pairs_compatibility", "pairs_sham", "pairs_transitivity", "tokens_descriptors")
TRAINING_SEEDS = (2026090721, 2026090722, 2026090723)
ARTIFACTS = {"config.json", "curve.jsonl", "last_epoch.pt", "epoch_10.pt", "epoch_25.pt", "epoch_50.pt"}


def training_registry(root: Path):
    """All hashes and identities are checked; no inference on partial campaigns."""
    manifest = json.loads((root/"manifest.json").read_text())
    if (manifest["status"] != "TRAINED_NOT_EVALUATED" or manifest["test_status"] != "CLOSED"
            or (root/"FAILURE.json").exists()):
        raise ValueError("training campaign is incomplete")
    for name, key in (("request.json", "request_sha256"), ("training.json", "training_sha256"),
                      ("worker.log", "worker_log_sha256")):
        if sha_file(root/name) != manifest[key]:
            raise ValueError("campaign artifact hash mismatch")
    request = json.loads((root/"request.json").read_text())
    if request.get("scope") != "15_trainings_no_test_access" or request.get("test_status") != "CLOSED":
        raise ValueError("training request scope/test status mismatch")
    trained = json.loads((root/"training.json").read_text())
    if (trained["status"] != "TRAINED_NOT_EVALUATED" or trained["test_status"] != "CLOSED"
            or manifest["source_sha256"] != request["source_sha256"]
            or trained["source_sha256"] != request["source_sha256"]):
        raise ValueError("training/source status mismatch")
    expected = {(arm, seed) for arm in ARM_NAMES for seed in TRAINING_SEEDS}
    cells, initial = {}, {}
    for row in trained["cells"]:
        key = (row["arm"], row["seed"])
        relative = f"cells/{row['arm']}__seed_{row['seed']}"
        if key not in expected or key in cells or row["path"] != relative:
            raise ValueError("wrong, duplicate or unsafe training cell")
        cell = root/relative
        if (cell/"FAILURE.json").exists() or sha_file(cell/"result.json") != row["result_sha256"]:
            raise ValueError("cell incomplete or hash mismatch")
        result = json.loads((cell/"result.json").read_text())
        binding = result["binding"]
        if (result["status"] != "TRAINED_NOT_EVALUATED" or result["steps"] != 3200
                or (binding["arm"], binding["seed"]) != key
                or binding["source_sha256"] != request["source_sha256"]
                or binding["train_manifest_sha256"] != request["cache_manifests"]["train"]
                or set(result["artifacts_sha256"]) != ARTIFACTS):
            raise ValueError("cell binding or artifact roster mismatch")
        for name, digest in result["artifacts_sha256"].items():
            if sha_file(cell/name) != digest:
                raise ValueError("cell artifact changed")
        if result["artifacts_sha256"]["last_epoch.pt"] != row["last_epoch_sha256"] or result["last_epoch_sha256"] != row["last_epoch_sha256"]:
            raise ValueError("checkpoint hashes disagree")
        if row["arm"] != "tokens_descriptors":
            seed = row["seed"]
            digest = binding["initial_model_sha256"]
            if seed in initial and initial[seed] != digest:
                raise ValueError("pair models did not share initialization")
            initial[seed] = digest
        cells[key] = {"checkpoint": cell/"last_epoch.pt", "checkpoint_sha256": row["last_epoch_sha256"],
                      "binding": binding, "cell_path": cell}
    if set(cells) != expected:
        raise ValueError("the fifteen-cell roster is incomplete")
    return {"cells": cells, "manifest_sha256": sha_file(root/"manifest.json"),
            "request": request, "manifest": manifest}
