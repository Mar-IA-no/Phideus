"""Small CPU fixtures; never generates a held-out evaluation split."""
from itertools import combinations
import json
import math
import sys
import unittest

import numpy as np

from src.atencion_armonica.partial_compatibility import (
    frequency_features, triple_geometry, sham_geometry,
)
from src.atencion_armonica.shared_partial_data import (
    generate_scene, mechanical_fixture, SPLITS, OPEN_SPLITS,
)
from src.atencion_armonica.peak_tokens import compute_pair_features


def scalar_residual(values):
    """Separate scalar implementation of the frozen equation, no vector core."""
    x, y, z = sorted(float(v) for v in values)
    candidates = []
    for a, b, c in combinations(range(1, 9), 3):
        def difference(u, beta):
            return math.log(u/a)+0.5*(math.log1p(beta*u*u)-math.log1p(beta*a*a))
        if z-x <= difference(c, 1e-5):
            beta = 1e-5
        elif z-x >= difference(c, 2e-2):
            beta = 2e-2
        else:
            t = math.exp(2*(z-x))*(a/c)**2
            beta = min(2e-2, max(1e-5, (t-1)/(c*c-t*a*a)))
        candidates.append(1200/math.log(2)*math.sqrt(
            ((y-x-difference(b, beta))**2+(z-x-difference(c, beta))**2)/2))
    return min(candidates)


def restored_weights(q, weights, permutation):
    return {tuple(sorted(permutation[list(t)])): float(w)
            for t, w in zip(combinations(range(len(q)), 3), weights)}


class GeometryTests(unittest.TestCase):
    def test_scalar_reference_and_clamps(self):
        # Known lower/outside and upper/outside branches plus interior.
        for q in ([-.1, 0., .1], [-3., 0., 3.], [-1., -.2, .5], [0., 0., 0.]):
            q = np.asarray(q, dtype=np.float32)
            g = triple_geometry(q)
            self.assertAlmostEqual(g["residual_cents"][0], scalar_residual(q), places=9)
        observation, _ = mechanical_fixture(3)
        q = np.asarray(observation["log_f"][:7], dtype=np.float32)
        g = triple_geometry(q)
        np.testing.assert_allclose(g["residual_cents"],
                                   [scalar_residual(q[t]) for t in g["triples"]], atol=1e-9, rtol=1e-10)

    def test_true_family_quantized_residual(self):
        n = np.arange(1, 9, dtype=np.float64)
        q = np.log(173*n*np.sqrt(1+5e-4*n*n))
        q = (q-q.mean()).astype(np.float32)
        g = triple_geometry(q)
        self.assertEqual(len(g["triples"]), 56)
        self.assertLess(g["residual_cents"].max(), .001)

    def test_pair_support_is_mean_over_thirds(self):
        q = np.asarray([-.7, -.1, .2, .6], dtype=np.float32)
        g = triple_geometry(q)
        for i in range(len(q)):
            self.assertEqual(g["pair_support"][i, i], 0)
            for j in range(i+1, len(q)):
                relevant = [1-w for t, w in zip(g["triples"], g["weights"]) if i in t and j in t]
                self.assertAlmostEqual(g["pair_support"][i, j], sum(relevant)/len(relevant))

    def test_sham_exact_derivation_and_equivariance(self):
        q = np.asarray([-.7, .6, -.1, .2, 1.1], dtype=np.float32)
        g = triple_geometry(q)
        s = sham_geometry(q, g, split_seed=2026090710, scene_id=0)
        expected_shift = int(np.random.default_rng(np.random.SeedSequence(
            [2026090730, 2026090710, 0])).integers(1, 10))
        self.assertTrue(s["evaluable"])
        self.assertEqual(s["shift"], expected_shift)
        np.testing.assert_array_equal(np.sort(g["weights"]), np.sort(s["weights"]))
        mapping = s["canonical_to_delivered"]
        np.testing.assert_array_equal(s["weights"][mapping],
            g["weights"][mapping][(np.arange(10)+expected_shift) % 10])
        perm = np.array([2, 4, 0, 3, 1])
        gp = triple_geometry(q[perm])
        sp = sham_geometry(q[perm], gp, split_seed=2026090710, scene_id=0)
        self.assertEqual(restored_weights(q, s["weights"], np.arange(5)),
                         restored_weights(q, sp["weights"], perm))
        np.testing.assert_array_equal(g["pair_support"][np.ix_(perm, perm)], gp["pair_support"])

    def test_ties_and_single_triple_do_not_choose_labels(self):
        for q in ([0., 0., .2, .6], [0., .2, .6]):
            q = np.asarray(q, dtype=np.float32)
            g = triple_geometry(q)
            sham = sham_geometry(q, g, split_seed=1, scene_id=0)
            self.assertFalse(sham["evaluable"])
            self.assertTrue(np.all(sham["weights"] == 0))

    def test_frequency_only_features_and_legacy_parity(self):
        q = np.asarray([-.5, -.2, 0., .8], dtype=np.float32)
        features = frequency_features(q)
        historical, classes = compute_pair_features(np.exp(q.astype(np.float64)), np.ones(len(q)))
        np.testing.assert_allclose(features["pair_cont"][..., :3], historical[..., :3], atol=1e-7)
        np.testing.assert_array_equal(features["ratio_class_id"], classes)
        np.testing.assert_array_equal(features["tokens"][:, 0], q)
        self.assertTrue(np.all(features["tokens"][:, 1] == 0))
        np.testing.assert_array_equal(features["pair_cont"][..., 3],
                                      features["geometry"]["pair_support"].astype(np.float32))

    def test_input_contract_and_maximum_fixture(self):
        for bad in (np.zeros(4, dtype=np.float64), np.zeros(2, dtype=np.float32),
                    np.zeros(33, dtype=np.float32), np.array([0, 1, np.nan], dtype=np.float32)):
            with self.assertRaises(ValueError):
                frequency_features(bad)
        observation, _ = mechanical_fixture(4)
        q = np.asarray(observation["log_f"], dtype=np.float32)
        g = frequency_features(q)
        self.assertEqual(g["pair_cont"].shape, (32, 32, 4))
        self.assertEqual(len(g["geometry"]["triples"]), math.comb(32, 3))
        self.assertTrue(np.isfinite(g["pair_cont"]).all())


class ObservationTests(unittest.TestCase):
    def test_determinism_separation_and_reconstruction(self):
        obs, side = generate_scene("development", 0)
        self.assertEqual((obs, side), generate_scene("development", 0))
        self.assertEqual(set(obs), {"scene_id", "split_seed", "log_f"})
        recovered = (np.array(side["log_f_ideal"])+np.array(side["sensor_log_noise"])
                     - side["mean_log_f_observed"]).astype(np.float32)
        np.testing.assert_array_equal(recovered, np.asarray(obs["log_f"], dtype=np.float32))
        self.assertEqual(obs, json.loads(json.dumps(obs, allow_nan=False)))
        for source in side["sources"]:
            self.assertTrue(4 <= len(source["indices"]) <= 8)
            self.assertEqual(len(source["indices"]), len(set(source["indices"])))
            self.assertTrue(100 <= source["f0"] <= 500)
            self.assertTrue(1e-4 <= source["beta"] <= 1e-3)
        self.assertNotEqual(obs["log_f"], generate_scene("development", 1)[0]["log_f"])

    def test_heldout_closed_and_invalid_ids(self):
        for split in SPLITS.keys()-OPEN_SPLITS:
            with self.assertRaises(PermissionError):
                generate_scene(split, 0)
        for bad in (-1, 64, True, 0.5):
            with self.assertRaises(ValueError):
                generate_scene("development", bad)

    def test_mechanical_deformation_is_not_sensor_change(self):
        plain, truth = mechanical_fixture(4)
        changed, altered = mechanical_fixture(4, deformed=True)
        self.assertEqual(truth["sensor_log_noise"], altered["sensor_log_noise"])
        self.assertEqual(truth["source_ids"], altered["source_ids"])
        self.assertEqual(truth["permutation"], altered["permutation"])
        self.assertNotEqual(truth["log_f_ideal"], altered["log_f_ideal"])
        self.assertNotEqual(plain["log_f"], changed["log_f"])

    def test_no_torch_import(self):
        self.assertNotIn("torch", sys.modules)


if __name__ == "__main__":
    unittest.main()
