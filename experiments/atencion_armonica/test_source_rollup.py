import unittest

from src.atencion_armonica.source_artifacts import ARMS, SEEDS
from src.atencion_armonica.source_rollup import GROUP_METRICS, aggregate_split, mean_coverage, pressure_metrics, scene_group_strata
from src.atencion_armonica.source_loss_pressure import edge_pressure, evaluate_pressure
import numpy as np


class RollupTests(unittest.TestCase):
    def test_null_coverage_and_ineligible_groups(self):
        self.assertEqual(mean_coverage([2., None, 4.]), {"mean": 3., "eligible": 2, "total": 3})
        group = {"size": 2, "category": "pure", "fit": {"status": "UNDERCONSTRAINED"}}
        row = scene_group_strata([group, group])[0]
        self.assertEqual((row["groups"], row["members"]), (2, 4))
        self.assertEqual(row["metrics"]["fine_rms_cents"], {"mean": None, "eligible": 0, "total": 2})

    def test_seed_complete_then_equal_scene_not_pooled_groups(self):
        readers = [{"arm": a, "seed": s} for a in ARMS for s in SEEDS]+[{"arm": "analytic_support", "seed": None}]
        names = [f"{a}__seed_{s}" for a in ARMS for s in SEEDS]+["analytic_support", "privileged_truth_groups"]
        rows = []
        for i in range(3):
            for name in names:
                value = 2.+2*i
                if name == f"{ARMS[0]}__seed_{SEEDS[0]}" and i == 2:
                    value = None
                metrics = {m: mean_coverage([value]) for m in GROUP_METRICS}
                rows.append({"scene_id": i, "reader": name, "strata": [{"size": 3, "category": "pure",
                    "groups": 100 if i == 1 else 1, "members": 300 if i == 1 else 3,
                    "statuses": {}, "metrics": metrics}], "pressure_metrics": {"fixture": value} if "__seed_" in name else None})
        result = aggregate_split(rows, readers, scene_count=3)
        metric = result["arms"][ARMS[0]]["group_strata"][0]["metrics"]["fine_rms_cents"]
        self.assertEqual((metric["mean"], metric["eligible"], metric["total"]), (3., 2, 3))
        self.assertEqual(metric["eligible_readers_by_scene"], [3, 3, 2])
        self.assertEqual(result["arms"][ARMS[0]]["pressure"]["fixture"]["mean"], 3.)
        self.assertEqual(result["readers"]["analytic_support"]["group_strata"][0]["metrics"]["fine_rms_cents"]["mean"], 4.)
        with self.assertRaises(ValueError):
            aggregate_split(rows[:-1], readers, scene_count=3)

    def test_disabled_sham_zero_is_not_aggregated(self):
        raw = edge_pressure(np.zeros((3, 3)), np.array([[0, 1, 2]]), [.5], [0.], sham_evaluable=False)
        summary, _ = evaluate_pressure(raw, [0, 0, 0])
        metrics = pressure_metrics(summary)
        self.assertIsNone(metrics["sham.L"])
        self.assertIsNone(metrics["class_1.S"])
        self.assertIsNone(metrics["class_0.B"])
        self.assertGreater(metrics["physical.L"], 0.)


if __name__ == "__main__":
    unittest.main()
