"""Tiny text stand-ins test provenance without deserializing models or using GPU."""
from pathlib import Path
import tempfile
import unittest

from src.atencion_armonica.partial_compatibility_cache import encoded, sha_file
from src.atencion_armonica.partial_compatibility_registry import (
    ARM_NAMES, ARTIFACTS, TRAINING_SEEDS, training_registry,
)


def fixture(root, *, omit=False, mismatch=False, wrong_scope=False):
    sources = {"fixture": "digest"}
    request = {"source_sha256": sources, "cache_manifests": {"train": "cache fixture"},
               "scope": "wrong" if wrong_scope else "15_trainings_no_test_access", "test_status": "CLOSED"}
    (root/"request.json").write_bytes(encoded(request))
    rows = []
    for seed in TRAINING_SEEDS:
        for arm in ARM_NAMES:
            if omit and seed == TRAINING_SEEDS[-1] and arm == ARM_NAMES[-1]:
                continue
            relative = f"cells/{arm}__seed_{seed}"
            cell = root/relative
            cell.mkdir(parents=True)
            for name in ARTIFACTS:
                (cell/name).write_bytes(b"fixture not a torch checkpoint")
            checkpoint_hash = sha_file(cell/"last_epoch.pt")
            binding = {"arm": arm, "seed": seed, "source_sha256": sources,
                       "train_manifest_sha256": "cache fixture",
                       "initial_model_sha256": "different" if mismatch and arm == ARM_NAMES[1] else "same"}
            result = {"status": "TRAINED_NOT_EVALUATED", "steps": 3200, "binding": binding,
                      "artifacts_sha256": {name: sha_file(cell/name) for name in ARTIFACTS},
                      "last_epoch_sha256": checkpoint_hash}
            (cell/"result.json").write_bytes(encoded(result))
            rows.append({"arm": arm, "seed": seed, "path": relative,
                         "result_sha256": sha_file(cell/"result.json"), "last_epoch_sha256": checkpoint_hash})
    (root/"training.json").write_bytes(encoded({"status": "TRAINED_NOT_EVALUATED", "test_status": "CLOSED",
                                               "source_sha256": sources, "cells": rows}))
    (root/"worker.log").write_bytes(b"fixture")
    (root/"manifest.json").write_bytes(encoded({"status": "TRAINED_NOT_EVALUATED", "test_status": "CLOSED",
             "source_sha256": sources, "request_sha256": sha_file(root/"request.json"),
             "training_sha256": sha_file(root/"training.json"), "worker_log_sha256": sha_file(root/"worker.log")}))


class RegistryTests(unittest.TestCase):
    def test_all_fifteen_and_hash_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            self.assertEqual(len(training_registry(root)["cells"]), 15)
            path = root/"cells"/f"{ARM_NAMES[0]}__seed_{TRAINING_SEEDS[0]}"/"epoch_10.pt"
            path.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                training_registry(root)

    def test_incomplete_roster_and_different_initializations(self):
        for kwargs, message in (({"omit": True}, "roster is incomplete"),
                                ({"mismatch": True}, "share initialization"),
                                ({"wrong_scope": True}, "request scope/test status")):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                fixture(root, **kwargs)
                with self.assertRaisesRegex(ValueError, message):
                    training_registry(root)


if __name__ == "__main__":
    unittest.main()
