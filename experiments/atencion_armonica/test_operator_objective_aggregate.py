"""Scene-unit, missing-support and presence-slice aggregation fixtures."""
from copy import deepcopy
import unittest

from src.atencion_armonica import operator_objective_aggregate as aggregate
from src.atencion_armonica import operator_objective_scene as scene
from experiments.atencion_armonica.test_operator_objective_scene import example


class AggregateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.complete = scene.diagnose_scene(*example())
        cls.empty = scene.diagnose_scene(*example([]))

    def test_distribution_defined_scenes_only_and_quantiles(self):
        result = aggregate.distribution([0., None, 2.])
        self.assertEqual(result, {"mean": 1., "defined": 2, "total": 3, "undefined": 1,
                                  "q10": .2, "q50": 1., "q90": 1.8})
        self.assertIsNone(aggregate.distribution([])["mean"])
        for bad in (["1"], [True], [float("nan")]):
            with self.assertRaises(ValueError):
                aggregate.distribution(bad)

    def test_scene_unit_empty_scene_and_presence_slices(self):
        acc = aggregate.ScenarioAccumulator("deformed_family", expected_scenes=2)
        acc.add(0, "pool", self.complete)
        acc.add(1, "absent", self.empty)
        result = acc.finalize()
        full = result["slices"]["all"]
        self.assertEqual(full["scene_count"], 2)
        metric = full["metrics"]["arm/generative/full/ari"]
        self.assertEqual(metric["defined"], 1)
        self.assertEqual(metric["total"], 2)
        self.assertEqual(metric["support_totals"], {"cells_defined": 9, "cells_total": 18})
        self.assertEqual(result["slices"]["pool"]["scene_ids"], [0])
        self.assertEqual(result["slices"]["absent"]["scene_ids"], [1])
        self.assertEqual(result["slices"]["neighbor"]["scene_count"], 0)
        self.assertEqual(result["slices"]["neighbor"]["metrics"]["arm/generative/full/ari"]["total"], 0)
        counts = full["tau_status_counts"]["classical/extended_ub/full"]
        self.assertEqual(counts, {"DEFINED": 1, "NO_CANDIDATE": 1})
        self.assertEqual(full["metrics"]["oracle/full/target64_ari_same_choice"]["mean"], 1.)

    def test_pairing_support_and_complete_nine_remain_separate(self):
        compact, labels, predictions = example()
        predictions["generative"][scene.CELLS[0]][:] = .1
        result = scene.diagnose_scene(compact, labels, predictions)
        acc = aggregate.ScenarioAccumulator("iid", expected_scenes=1)
        acc.add(0, "neighbor", result)
        metrics = acc.finalize()["slices"]["all"]["metrics"]
        partial = metrics["paired/generative-local/full/tau"]
        self.assertEqual(partial["defined"], 1)
        self.assertEqual(partial["support_totals"]["cells_defined"], 8)
        complete = metrics["paired_complete_nine/generative-local/full/tau"]
        self.assertEqual(complete["defined"], 0)
        self.assertIsNone(complete["mean"])

    def test_incomplete_duplicate_wrong_order_and_mutation_rejected(self):
        acc = aggregate.ScenarioAccumulator("ood_beta", expected_scenes=2)
        with self.assertRaises(ValueError):
            acc.finalize()
        with self.assertRaises(ValueError):
            acc.add(1, "pool", self.complete)
        acc.add(0, "pool", self.complete)
        with self.assertRaises(ValueError):
            acc.add(0, "pool", self.complete)
        bad = deepcopy(self.complete)
        del bad["arm_summary"]["generative"]["full"]["ari"]
        with self.assertRaises(ValueError):
            acc.add(1, "pool", bad)
        self.assertEqual(acc.count, 1)
        acc.add(1, "absent", self.empty)
        self.assertEqual(acc.finalize()["scene_count"], 2)


if __name__ == "__main__":
    unittest.main()
