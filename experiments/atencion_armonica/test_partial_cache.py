"""NumPy cache fixtures; production train/validation/test are never generated."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from src.atencion_armonica import partial_compatibility_cache as cache
from src.atencion_armonica.shared_partial_data import mechanical_fixture


def fixture_scene(split, scene_id):
    obs, truth = mechanical_fixture(3)
    obs["scene_id"] = truth["scene_id"] = scene_id
    obs["split_seed"] = truth["split_seed"] = cache.SPLITS[split][1]
    # Distinct mechanical cache fixtures, not draws from a prospective split.
    obs["log_f"][0] += scene_id * .002
    return obs, truth


class CacheTests(unittest.TestCase):
    def setUp(self):
        base = cache.ROOT/".agent-work/phideus-geometric-rebase-20260907/cache-tests"
        base.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=base)
        self.addCleanup(self.tmp.cleanup)
        self.output = Path(self.tmp.name)/"cache"

    def prepare(self):
        with patch.dict(cache.SPLITS, {"development": (2, 2026090710)}), \
             patch.object(cache, "generate_scene", side_effect=fixture_scene) as generate:
            cache.prepare_open_split(self.output, "development")
            self.assertEqual(generate.call_count, 2)
            return cache.ObservationCache(self.output, "development")

    def test_cache_roundtrip_and_inference_does_not_read_truth(self):
        observed = self.prepare()
        self.assertEqual(len(observed), 2)
        expected = cache.feature_record(fixture_scene("development", 1)[0])
        for key in cache.RECORD_KEYS:
            np.testing.assert_array_equal(observed.records[1][key], expected[key])
        self.assertEqual(len(cache.load_supervision(observed)), 2)
        original = Path.open
        def guarded(path, *args, **kwargs):
            if path.name == "sidecars.jsonl":
                raise AssertionError("inference accessed truth")
            return original(path, *args, **kwargs)
        with patch.dict(cache.SPLITS, {"development": (2, 2026090710)}), patch.object(Path, "open", guarded):
            self.assertEqual(len(cache.ObservationCache(self.output, "development")), 2)
        self.assertNotIn("torch", sys.modules)

    def test_closed_test_before_creation_and_existing_output(self):
        for split in ("iid", "ood_beta", "ood_polyphony", "ood_noise", "deformed_family"):
            with self.assertRaises(PermissionError):
                cache.prepare_open_split(self.output, split)
            self.assertFalse(self.output.exists())
        self.prepare()
        with self.assertRaises(FileExistsError):
            cache.prepare_open_split(self.output, "development")

    def test_feature_and_supervision_corruption(self):
        observed = self.prepare()
        with (self.output/"features/00000.npz").open("ab") as handle:
            handle.write(b"fixture-corruption")
        with patch.dict(cache.SPLITS, {"development": (2, 2026090710)}):
            with self.assertRaisesRegex(ValueError, "feature hash"):
                cache.ObservationCache(self.output, "development")
        with (self.output/"sidecars.jsonl").open("ab") as handle:
            handle.write(b"fixture-corruption")
        with self.assertRaisesRegex(ValueError, "supervision hash"):
            cache.load_supervision(observed)

    def test_duplicate_is_incomplete_not_filtered(self):
        with patch.dict(cache.SPLITS, {"development": (2, 2026090710)}), \
             patch.object(cache, "generate_scene", side_effect=lambda *args: fixture_scene("development", 0)):
            with self.assertRaisesRegex(ValueError, "duplicate"):
                cache.prepare_open_split(self.output, "development")
        self.assertFalse((self.output/"manifest.json").exists())
        self.assertEqual(json.loads((self.output/"FAILURE.json").read_bytes())["status"], "INCOMPLETE")

    def test_fingerprint_ignores_delivered_order_and_overlap_rejected(self):
        observed = self.prepare()
        obs = observed.observations[0]
        reverse = {**obs, "log_f": obs["log_f"][::-1]}
        self.assertEqual(cache.observation_fingerprint(obs), cache.observation_fingerprint(reverse))
        with self.assertRaisesRegex(ValueError, "across splits"):
            cache.assert_disjoint(observed, observed)


if __name__ == "__main__":
    unittest.main()
