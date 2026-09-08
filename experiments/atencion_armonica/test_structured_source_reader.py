"""Mechanical fixtures only: no prospective generator or checkpoint access."""
import inspect
from itertools import combinations
import unittest

import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

from src.atencion_armonica.partial_compatibility import triple_geometry
from src.atencion_armonica.partial_compatibility_cache import encoded
from src.atencion_armonica.source_coherence import GroupFitCache
from src.atencion_armonica.structured_source_reader import (
    FACTORS, bounded_cost, build_pool, choose_partition, decouple_costs,
    partition_energies, score_scene, sham_energy_support, signature, tree_cuts)


def fixture(n=10):
    q = (np.arange(n, dtype=np.float32)*.03125)[::-1].copy()
    rng = np.random.default_rng(178)
    raw = rng.normal(size=(n, n)).astype(np.float32)
    z = (raw+raw.T)*np.float32(.5)
    g = triple_geometry(q)
    return q, z, g


def score(q, z, g, cache=None):
    return score_scene(q, z, g["pair_support"], g["triples"], g["residual_cents"],
                       split_seed=10, scene_id=20, fit_cache=cache)


class PoolTests(unittest.TestCase):
    def test_all_merges_even_ties_and_common_cardinality_filter(self):
        distance = np.ones((5, 5))-np.eye(5)
        result = tree_cuts(distance)
        self.assertEqual([len(p) for p in result["cuts"]], [5, 4, 3, 2, 1])
        self.assertEqual(result["merge_height_repetitions"], 3)
        np.testing.assert_array_equal(result["tree"], linkage(squareform(distance), method="average"))
        q, z, g = fixture(12)
        pool = build_pool(q, z, g["pair_support"])
        self.assertLessEqual(pool["unfiltered_count"], 24)
        self.assertGreater(pool["filtered_count"], 0)
        self.assertIn(signature([[i] for i in range(12)]), pool["partitions"])
        self.assertEqual(pool["partitions"], sorted(set(pool["unfiltered_partitions"]) & set(pool["partitions"])))
        for p in pool["partitions"]:
            self.assertEqual(sorted(i for group in p for i in group), list(range(12)))
            self.assertLessEqual(max(map(len, p)), 8)
        self.assertEqual(set(pool["partitions"]), {p for p in pool["unfiltered_partitions"] if max(map(len, p)) <= 8})

    def test_permutation_and_exact_offset_no_input_mutation(self):
        q, z, g = fixture()
        original_q, original_z = q.copy(), z.copy()
        baseline = score(q, z, g)
        perm = np.random.default_rng(7).permutation(len(q))
        pgeom = triple_geometry(q[perm])
        # Keep the exact same analytic pair matrix, just reindex its vertices.
        pgeom["pair_support"] = g["pair_support"][np.ix_(perm, perm)]
        permuted = score(q[perm], z[np.ix_(perm, perm)], pgeom)
        self.assertEqual(baseline["pool"]["partitions"], permuted["pool"]["partitions"])
        self.assertEqual(baseline["candidates"], permuted["candidates"])
        for factor in FACTORS:
            gamma = 0 if factor == "pairs" else .3
            a, b = choose_partition(baseline, factor, gamma), choose_partition(permuted, factor, gamma)
            self.assertEqual(a["canonical_signature"], b["canonical_signature"])
            self.assertEqual(a["partition"], signature([[perm[i] for i in group] for group in b["partition"]]))
        shifted = score(q+np.float32(.125), z, g)
        self.assertEqual(baseline["candidates"], shifted["candidates"])
        np.testing.assert_array_equal(q, original_q)
        np.testing.assert_array_equal(z, original_z)

    def test_duplicate_q_and_invalid_inputs(self):
        q, z, g = fixture(4)
        q[1] = q[0]
        self.assertEqual(build_pool(q, z, g["pair_support"])["q_tie_count"], 1)
        with self.assertRaises(ValueError):
            build_pool(q.astype(np.float64), z, g["pair_support"])
        z[0, 1] += 1
        with self.assertRaises(ValueError):
            build_pool(q, z, g["pair_support"])
        with self.assertRaises(ValueError):
            score_scene(q, z+np.nan, g["pair_support"], g["triples"], g["residual_cents"], split_seed=10, scene_id=20)


class EnergyTests(unittest.TestCase):
    def test_direct_enumeration_of_bce_and_group_factors(self):
        q, z, g = fixture(6)
        out = score(q, z, g)
        order = out["pool"]["canonical_to_observed"]
        canonical_z = z[np.ix_(order, order)].astype(np.float64)
        for row in out["candidates"]:
            p = row["signature"]
            truth_pi = {(a, b) for group in p for a, b in combinations(group, 2)}
            loss = sum(np.logaddexp(0., canonical_z[a, b]) - ((a, b) in truth_pi)*canonical_z[a, b]
                       for a, b in combinations(range(6), 2))
            self.assertAlmostEqual(row["bce_sum"], loss, places=12)
            self.assertAlmostEqual(row["pair_energy"]*6, loss-row["bce_constant"], places=12)
            for name in FACTORS[1:]:
                expected = sum(out["groups"][i]["size"]*out["groups"][i]["costs"][name] for i in row["group_ids"])/6
                self.assertEqual(row["factor_energies"][name], expected)
        for group in out["groups"]:
            if group["size"] < 3:
                self.assertEqual(group["constraint_status"], "ABSENT_UNDERCONSTRAINED")
                self.assertEqual(group["costs"]["shared_source"], 0.)
            else:
                local = group["endpoint_residual_cents"]
                self.assertEqual(group["costs"]["local_compatibility"], float(np.mean([r*r/(r*r+4) for r in local])))
        self.assertEqual(float(bounded_cost(2)), .5)

    def test_extensive_local_change_not_reweighted_by_unrelated_events(self):
        z = np.ones((6, 6), dtype=np.float64)
        partitions = [((0, 1, 2),), ((0,), (1,), (2,))]
        def run(n):
            extra = ((3, 4, 5),) if n == 6 else ()
            ps = [signature(p+extra) for p in partitions]
            gs = sorted({g for p in ps for g in p})
            costs = {"shared_source": [.5 if len(g) == 3 else 0. for g in gs]}
            rs = partition_energies(z[:n, :n], ps, gs, costs)
            return ((rs[0]["pair_energy"]-rs[1]["pair_energy"])*n,
                    (rs[0]["factor_energies"]["shared_source"]-rs[1]["factor_energies"]["shared_source"])*n)
        self.assertEqual(run(3), run(6))

    def test_gamma_zero_identity_and_exact_ties(self):
        q, z, g = fixture(5)
        z.fill(0)
        out = score(q, z, g)
        baseline = choose_partition(out, "pairs")
        self.assertEqual(baseline["co_minimum_count"], len(out["candidates"]))
        self.assertEqual(baseline["authority"], "TIED_CANONICAL_CHOICE")
        self.assertIsNone(baseline["next_level_gap"])
        for factor in FACTORS[1:]:
            self.assertEqual(encoded(baseline), encoded(choose_partition(out, factor, 0)))
        out["candidates"].reverse()
        self.assertEqual(baseline["canonical_signature"], choose_partition(out, "pairs")["canonical_signature"])
        with self.assertRaises(ValueError):
            choose_partition(out, "shared_source", .2)


class ShamCacheTests(unittest.TestCase):
    def test_exact_rotation_multisets_and_constant_energy_support(self):
        groups = [(0,), (0, 1, 2), (0, 1, 3), (0, 2, 3)]
        costs = np.array([0., .125, .5, .875])
        sham, strata = decouple_costs(groups, costs, split_seed=10, scene_id=20)
        shift = int(np.random.default_rng(np.random.SeedSequence([2026090785, 10, 20, 3])).integers(1, 3))
        np.testing.assert_array_equal(sham[1:], costs[1:][(np.arange(3)+shift) % 3])
        np.testing.assert_array_equal(np.sort(sham), np.sort(costs))
        self.assertEqual(strata[0]["status"], "NO_SHAM_CONTRAST")
        self.assertEqual(sham_energy_support([.25, .5], [.5, .75])["status"], "NOT_EVALUABLE_SHAM_SUPPORT")
        self.assertEqual(sham_energy_support([.25, .5], [.5, .5])["status"], "RANKING_CONTRAST_AVAILABLE")

    def test_cache_reuse_and_complete_triples_and_no_truth_port(self):
        q, z, g = fixture(6)
        cache = GroupFitCache()
        a = score(q, z, g, cache)
        b = score(q, z, g, cache)
        self.assertEqual(b["fit_cache_misses"], 0)
        self.assertEqual(b["fit_cache_hits"], len(a["groups"]))
        self.assertEqual(a["candidates"], b["candidates"])
        with self.assertRaises(ValueError):
            score_scene(q, z, g["pair_support"], g["triples"][:-1], g["residual_cents"][:-1], split_seed=10, scene_id=20)
        with self.assertRaises(ValueError):
            score(q+np.float32(.125), z, g, cache)
        self.assertNotIn("truth", inspect.signature(score_scene).parameters)
        self.assertNotIn("source_ids", inspect.signature(score_scene).parameters)


if __name__ == "__main__":
    unittest.main()
