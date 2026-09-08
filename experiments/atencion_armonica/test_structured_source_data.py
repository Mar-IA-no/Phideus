"""Mechanical namespaces only: never draws a prospective split seed."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from src.atencion_armonica import structured_source_data as data
from src.atencion_armonica import structured_source_gate as gate
from src.atencion_armonica.partial_compatibility_cache import feature_record, observation_fingerprint
from src.atencion_armonica.structured_source_artifacts import write_json

FIXTURE_SEED = 2026090790


def fixture_observation(split, i):
    obs, truth = data._draw_scene(split, i, FIXTURE_SEED)
    # Relabel mechanical observations to exercise serialization identities;
    # this is never a draw from the evaluation RNG namespace.
    obs["split_seed"] = truth["split_seed"] = data.SPLIT_SEEDS[split]
    return obs, truth


class DataTests(unittest.TestCase):
    def test_law_uses_declared_family_and_reconstructs_observation(self):
        for split in data.SPLIT_ORDER:
            obs, truth = data._draw_scene(split, 0, FIXTURE_SEED)
            self.assertEqual(obs["split_seed"], FIXTURE_SEED)
            self.assertEqual((obs, truth), data._draw_scene(split, 0, FIXTURE_SEED))
            k = len(truth["sources"])
            self.assertIn(k, (4,) if split == "ood_polyphony" else (2, 3))
            self.assertEqual(truth["sigma_cents"], 2.)
            ideal, labels, indices = [], [], []
            lo, hi = (3e-3, 1e-2) if split == "ood_beta" else (1e-4, 1e-3)
            for sid, s in enumerate(truth["sources"]):
                self.assertTrue(100 <= s["f0"] <= 500 and lo <= s["beta"] <= hi)
                self.assertTrue(4 <= len(s["indices"]) <= 8)
                self.assertEqual(s["indices"], sorted(set(s["indices"])))
                self.assertTrue(5e-6 <= s["gamma"] <= 5e-5 if split == "deformed_family" else s["gamma"] == 0)
                ns = np.asarray(s["indices"])
                ideal.extend(np.log(s["f0"])+np.log(ns)+.5*np.log1p(s["beta"]*ns**2+s["gamma"]*ns**4))
                labels.extend([sid]*len(ns))
                indices.extend(ns)
            perm = truth["permutation"]
            np.testing.assert_array_equal(np.array(ideal)[perm], truth["log_f_ideal"])
            np.testing.assert_array_equal(np.array(labels)[perm], truth["source_ids"])
            np.testing.assert_array_equal(np.array(indices)[perm], truth["partial_indices"])
            reconstructed = (np.array(ideal)[perm]+truth["sensor_log_noise"]-truth["mean_log_f_observed"]).astype(np.float32)
            np.testing.assert_array_equal(reconstructed, obs["log_f"])
        for i in (-1, 256, True):
            with self.assertRaises(ValueError):
                data._draw_scene("calibration", i, FIXTURE_SEED)

    def test_feature_shape_identity_and_truth_rejection(self):
        obs, _ = fixture_observation("calibration", 0)
        q = data.validate_observation(obs, 0, "calibration")
        record = feature_record(obs)
        data.validate_record(record, q)
        for mutation in (lambda r: r.update(source_ids=np.zeros(len(q))),
                         lambda r: r.update(triples=r["triples"][::-1]),
                         lambda r: r.update(tokens=r["tokens"].astype(np.float64)),
                         lambda r: r["tokens"].__setitem__((0, 1), 1),
                         lambda r: r["pair_support"].__setitem__((0, 1), 2)):
            bad = copy.deepcopy(record)
            mutation(bad)
            with self.assertRaises(ValueError):
                data.validate_record(bad, q)
        for change in ({"source_ids": []}, {"split_seed": 2026090781}, {"scene_id": True}):
            with self.assertRaises(ValueError):
                data.validate_observation({**obs, **change}, 0, "calibration")

    def test_denied_gate_precedes_rng_and_output_creation(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(gate, "verify_authorization", side_effect=PermissionError), \
                patch.object(data, "_draw_scene") as draw:
            output = Path(tmp)/"denied"
            with self.assertRaises(PermissionError):
                data.prepare_split(output, "iid", authorization={}, previous={})
            draw.assert_not_called()
            self.assertFalse(output.exists())

    def test_bundle_roundtrip_and_duplicate_draw_preservation(self):
        scenes = [fixture_observation("calibration", i) for i in range(3)]
        auth = {"common": {"mechanical_fixture": True}}
        with tempfile.TemporaryDirectory() as tmp, patch.object(gate, "ROOT", Path(tmp)), \
                patch.object(data, "COUNT", 3), patch.object(gate, "verify_authorization", return_value=auth), \
                patch.object(gate, "historical_observations", return_value=[]), \
                patch.object(data, "prior_corpus", return_value=set()), \
                patch.object(data, "_draw_scene", side_effect=lambda split, i, seed: scenes[i]):
            output = Path(tmp)/"fixture"
            ref = data.prepare_split(output, "calibration", authorization={"fixture": True}, previous={})
            cache = data.StructuredObservations(ref, "calibration", auth["common"])
            self.assertEqual(cache.observations, [s[0] for s in scenes])
            self.assertEqual(data.load_supervision(cache), [s[1] for s in scenes])
            self.assertEqual(len(cache.fingerprints), 3)
            with self.assertRaises(ValueError):
                data.StructuredObservations(ref, "iid", auth["common"])
            with patch.object(data, "prior_corpus", return_value={observation_fingerprint(scenes[0][0])}), \
                    patch.object(data, "_draw_scene", return_value=scenes[0]) as draw:
                failed = Path(tmp)/"failed"
                with self.assertRaisesRegex(ValueError, "duplicate"):
                    data.prepare_split(failed, "calibration", authorization={}, previous={})
                self.assertEqual(draw.call_count, 1)
                self.assertTrue((failed/"FAILURE.json").is_file())
                self.assertFalse((failed/"manifest.json").exists())
                self.assertEqual(len((failed/"observations.jsonl").read_text().splitlines()), 1)
                self.assertEqual(len((failed/"sidecars.jsonl").read_text().splitlines()), 1)

    def test_all_prior_roles_mandatory_and_frozen_calibration_exact(self):
        with self.assertRaisesRegex(ValueError, "all previous"):
            data.prior_corpus({}, "ood_polyphony", {"calibration": {}, "iid": {}})
        with self.assertRaisesRegex(ValueError, "all previous"):
            data.prior_corpus({}, "calibration", {"iid": {}})
        with patch.object(gate, "read_reference", side_effect=[{"calibration_authorization": {}, "calibration_data": {"pin": 1}}, {}]):
            with self.assertRaisesRegex(ValueError, "frozen dataset"):
                data.prior_corpus({"common": {}, "freeze": {}}, "iid", {"calibration": {"pin": 2}})
        with patch.object(gate, "historical_observations", return_value=[{"required": True}]):
            with self.assertRaisesRegex(ValueError, "historical corpus omitted"):
                data.prior_corpus({"common": {}, "historical_observations": []}, "calibration", {})


if __name__ == "__main__":
    unittest.main()
