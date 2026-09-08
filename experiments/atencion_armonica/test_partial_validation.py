"""CPU-only launch guards and a two-scene stub of all sixteen reader selections."""
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from experiments.atencion_armonica import analyze_partial_validation as analysis
from experiments.atencion_armonica import collect_partial_validation_logits as forward
from src.atencion_armonica.partial_compatibility_registry import ARM_NAMES, TRAINING_SEEDS


class ValidationStageTests(unittest.TestCase):
    def test_forward_missing_authority_or_lease_never_launches(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(forward.subprocess, "Popen") as spawn:
            root = Path(tmp)
            with self.assertRaises(ValueError):
                forward.launch(root/"out", root/"training", root/"missing")
            self.assertFalse((root/"out").exists())
            spawn.assert_not_called()
            (root/"request.json").write_bytes(b"fixture")
            with self.assertRaisesRegex(ValueError, "inherited lease"):
                forward.worker(root, time.monotonic()+30, -1)

    def test_cpu_analysis_writes_all_fifteen_plus_analytic(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_root, raw_root = root/"cache", root/"raw"
            (cache_root/"validation").mkdir(parents=True)
            raw_root.mkdir()
            (cache_root/"validation"/"manifest.json").write_bytes(b"cache fixture")
            (raw_root/"manifest.json").write_bytes(b"raw fixture")
            observations = [{"scene_id": i, "split_seed": 123, "log_f": q}
                            for i, q in enumerate(([0., .001, .8, 1.], [0., .2, .3, .9]))]
            truths = [{"source_ids": [0, 0, 1, 1]} for _ in observations]
            logits = [np.where(np.equal.outer(t["source_ids"], t["source_ids"]), 2., -2.).astype(np.float32)
                      for t in truths]
            cache = MagicMock()
            cache.observations = observations
            cache.records = [{"pair_support": np.where(z > 0, .9, .1)} for z in logits]
            rows = {(arm, seed): {"path": f"{arm}_{seed}.npz", "checkpoint_sha256": "fixture"}
                    for arm in ARM_NAMES for seed in TRAINING_SEEDS}
            registry = {"manifest_sha256": "training fixture", "cells": rows}
            with patch.object(analysis, "training_registry", return_value=registry), \
                 patch.object(analysis, "validated_raw_forward", return_value=rows), \
                 patch.object(analysis, "ObservationCache", return_value=cache), \
                 patch.object(analysis, "load_supervision", return_value=truths), \
                 patch.object(analysis, "load_logits", return_value=logits), \
                 patch.object(analysis, "sources", return_value={"fixture": "hash"}), \
                 patch.object(analysis, "CACHE_ROOT", cache_root), patch("builtins.print"), \
                 patch("torch.cuda.is_available", side_effect=AssertionError("GPU forbidden")):
                analysis.analyze(root/"output", raw_root, root/"training")
            readers = json.loads((root/"output"/"readers.json").read_text())
            self.assertEqual(len(readers["readers"]), 16)
            self.assertEqual(readers["test_status"], "CLOSED")
            self.assertEqual(readers["runtime"], analysis.cpu_runtime())
            manifest = json.loads((root/"output"/"manifest.json").read_text())
            self.assertEqual(manifest["runtime"], readers["runtime"])
            self.assertEqual(readers["readers"][-1]["arm"], "analytic_support")
            for row in readers["readers"]:
                summary = json.loads((root/"output"/row["summary_path"]).read_text())
                self.assertEqual(summary["scene_count"], 2)
                self.assertEqual(summary["partition"]["ari"]["mean"], 1.)
                self.assertEqual(summary["pairs"]["brier"]["eligible_scenes"], 2)
                self.assertEqual(len((root/"output"/row["metrics_path"]).read_text().splitlines()), 2)

    def test_checkpoint_identity_is_rechecked_at_use(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"checkpoint"
            path.write_bytes(b"fixture")
            cell = {"checkpoint": path, "checkpoint_sha256": forward.sha_file(path)}
            forward.verify_checkpoint_bytes(cell)
            path.write_bytes(b"changed during inference")
            with self.assertRaisesRegex(ValueError, "changed at inference"):
                forward.verify_checkpoint_bytes(cell)


if __name__ == "__main__":
    unittest.main()
