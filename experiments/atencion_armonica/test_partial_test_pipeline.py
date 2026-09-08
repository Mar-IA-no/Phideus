"""CPU-only synthetic arrays/files, never draws from the five prospective test splits."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from experiments.atencion_armonica import analyze_partial_test as analysis
from experiments.atencion_armonica import collect_partial_test_logits as forward
from src.atencion_armonica.partial_compatibility_cache import encoded, sha_file
from src.atencion_armonica.partial_compatibility_metrics import paired_bootstrap, pair_metrics
from src.atencion_armonica.partial_compatibility_registry import ARM_NAMES, TRAINING_SEEDS
from src.atencion_armonica.partial_compatibility_test_gate import TEST_SPLITS


class TestPipelineTests(unittest.TestCase):
    def test_near_summary_preserves_scene_denominators_and_missing_classes(self):
        near = [pair_metrics([], []), pair_metrics([.2, .4], [0, 0]),
                pair_metrics([.8, .6], [1, 1]), pair_metrics([.2, .8], [0, 1])]
        rows = [{"pairs": pair_metrics([.2, .8], [0, 1]), "near_collision_pairs": value}
                for value in near]
        result = analysis.summarize_test(rows)
        self.assertEqual(result["pairs"]["brier"]["eligible_scenes"], 4)
        self.assertEqual(result["near_collision_scene_count"], 3)
        self.assertEqual(result["near_collision_directed_pair_count"], 6)
        summary = result["near_collision_pairs"]
        self.assertEqual(summary["brier"]["eligible_scenes"], 3)
        self.assertAlmostEqual(summary["brier"]["mean"], .08)
        self.assertEqual(summary["ap"]["eligible_scenes"], 1)
        self.assertEqual(summary["auc"]["eligible_scenes"], 1)
        self.assertEqual(summary["positive_recall_at_half"]["eligible_scenes"], 2)
        self.assertTrue(all(value["total_scenes"] == 4 for value in summary.values()))
        empty = analysis.summarize_test(rows[:1])["near_collision_pairs"]
        self.assertTrue(all(value == {"mean": None, "eligible_scenes": 0, "total_scenes": 1}
                            for value in empty.values()))

    def test_gpu_launch_needs_authority_and_freeze_before_creating_output(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(forward.subprocess, "Popen") as spawn:
            root = Path(tmp)
            with self.assertRaises(ValueError):
                forward.launch(root/"out", root/"freeze", root/"missing")
            (root/"receipt").write_bytes(b"fixture")
            with self.assertRaises(FileNotFoundError):
                forward.launch(root/"out", root/"missing_freeze", root/"receipt")
            spawn.assert_not_called()
            self.assertFalse((root/"out").exists())

    def test_all_three_comparisons_and_within_scene_seed_mean(self):
        candidate = {seed: np.linspace(.1, .2, 1024)+i*.01 for i, seed in enumerate(TRAINING_SEEDS)}
        offsets = {"pairs_descriptors": (.01, -.005, .003), "pairs_sham": (.02, .01, .015),
                   "pairs_transitivity": (-.01, -.02, -.005)}
        scores = {"pairs_compatibility": candidate}
        for control, values in offsets.items():
            scores[control] = {seed: candidate[seed]+values[i] for i, seed in enumerate(TRAINING_SEEDS)}
        report, deltas = analysis.primary_comparisons(scores)
        self.assertEqual(set(report), set(offsets))
        self.assertEqual(len(deltas), 12)
        self.assertEqual(report["pairs_transitivity"]["role"], "mandatory_attribution")
        self.assertGreater(report["pairs_transitivity"]["mean_training_seed_delta"], 0)
        for control_index, control in enumerate(offsets):
            averaged = np.stack([candidate[s]-scores[control][s] for s in TRAINING_SEEDS]).mean(axis=0)
            np.testing.assert_array_equal(deltas[f"{control}__mean_three_seeds"], averaged)
            expected = paired_bootstrap(averaged, np.zeros(1024), control_index=control_index, seed=0)
            self.assertEqual(report[control]["within_scene_mean_three_seeds"], expected)

    def test_raw_roster_requires_all_seventy_five_identities(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw, cache = root/"raw", root/"cache"
            raw.mkdir()
            cache.mkdir()
            freeze_path = root/"freeze.json"
            freeze_path.write_bytes(b"fixture freeze")
            freeze = {"source_sha256": {"fixture": "source"}}
            registry = {"cells": {(a, s): {"checkpoint_sha256": f"fixture:{a}:{s}"} for a in ARM_NAMES for s in TRAINING_SEEDS}}
            rows, manifests = [], {}
            for split in TEST_SPLITS:
                (cache/split).mkdir()
                (cache/split/"manifest.json").write_bytes(b"cache fixture")
                manifests[split] = sha_file(cache/split/"manifest.json")
                (raw/split).mkdir()
                for arm in ARM_NAMES:
                    for seed in TRAINING_SEEDS:
                        name = f"{split}/{arm}__seed_{seed}.npz"
                        (raw/name).write_bytes(b"fixture raw identity, not an array payload")
                        rows.append({"split": split, "arm": arm, "seed": seed, "path": name,
                                     "sha256": sha_file(raw/name), "scene_count": 1024,
                                     "checkpoint_sha256": registry["cells"][(arm, seed)]["checkpoint_sha256"]})
            (raw/"request.json").write_bytes(encoded({"freeze_sha256": sha_file(freeze_path),
                 "source_sha256": freeze["source_sha256"], "test_manifest_sha256": manifests}))
            (raw/"worker.log").write_bytes(b"fixture")

            def close_roster(selected):
                (raw/"forward.json").write_bytes(encoded({"status": "RAW_TEST_COMPLETE", "freeze_sha256": sha_file(freeze_path), "rows": selected}))
                (raw/"manifest.json").write_bytes(encoded({"status": "RAW_TEST_COMPLETE", "freeze_sha256": sha_file(freeze_path),
                    "source_sha256": freeze["source_sha256"], "request_sha256": sha_file(raw/"request.json"),
                    "forward_sha256": sha_file(raw/"forward.json"), "worker_log_sha256": sha_file(raw/"worker.log")}))

            close_roster(rows)
            with patch.object(analysis, "TEST_CACHE_ROOT", cache):
                self.assertEqual(len(analysis.validated_raw(raw, freeze_path, freeze, registry)), 75)
                close_roster(rows[:-1])
                with self.assertRaisesRegex(ValueError, "all 75"):
                    analysis.validated_raw(raw, freeze_path, freeze, registry)
                close_roster(rows[:-1]+[rows[0]])
                with self.assertRaisesRegex(ValueError, "identity mismatch"):
                    analysis.validated_raw(raw, freeze_path, freeze, registry)


if __name__ == "__main__":
    unittest.main()
