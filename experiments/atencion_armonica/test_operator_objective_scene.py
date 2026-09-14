"""Complete method roster on tiny deterministic mathematical fixtures."""
import unittest

import numpy as np

from src.atencion_armonica import operator_objective_core as core
from src.atencion_armonica import operator_objective_scene as scene
from src.atencion_armonica import operator_objective_sources as sources
from experiments.atencion_armonica.test_operator_objective_sources import fitted_fixture


def example(partitions=None, n=8):
    fitted = fitted_fixture(partitions, n)
    compact = sources.compact_fit(fitted, fitted["observation"])
    labels = np.repeat(np.arange(n//4), 4)
    target = core.partition_targets(compact["partitions"], labels)["u32"]
    predictions = {arm: {cell: target.copy() for cell in scene.CELLS} for arm in scene.ARMS}
    return compact, labels, predictions


class SceneTests(unittest.TestCase):
    def test_complete_roster_perfect_predictions_and_preserved_errors(self):
        result = scene.diagnose_scene(*example())
        self.assertEqual(set(result["classical"]), set(scene.CLASSICAL))
        self.assertEqual(sum(map(len, result["learned"].values())), 27)
        for arm in scene.ARMS:
            summary = result["arm_summary"][arm]["full"]
            self.assertEqual(summary["tau"], {"mean": 1., "defined": 9, "total": 9})
            self.assertEqual(summary["loss_like"]["mean"], 0.)
            self.assertEqual(summary["target64_regret"]["mean"], 0.)
            for cell in scene.CELLS:
                error = result["arrays"]["errors"][arm][cell]
                self.assertEqual(error["s32"].dtype, np.float32)
                np.testing.assert_array_equal(error["component_squared"], np.zeros((2, 2)))
        for pair in ("generative-local", "generative-decoupled"):
            self.assertEqual(result["paired"][pair]["full"]["ari"]["mean"], 0.)
        self.assertEqual(result["oracles"]["full"]["all"]["target64"]["chosen"], 0)

    def test_constant_cell_excluded_from_tau_not_other_metrics(self):
        compact, labels, predictions = example()
        predictions["generative"][scene.CELLS[0]][:] = .1
        result = scene.diagnose_scene(compact, labels, predictions)
        summary = result["arm_summary"]["generative"]["full"]
        self.assertEqual(summary["tau"]["defined"], 8)
        self.assertEqual(summary["ari"]["defined"], 9)
        pair = result["paired"]["generative-local"]["full"]["tau"]
        self.assertEqual(pair["valid_cell_count"], 8)
        self.assertEqual(pair["mean"], 0.)
        self.assertIsNone(pair["complete_nine"]["mean"])
        self.assertEqual(result["learned"]["generative"][scene.CELLS[0]]["schemes"]["full"]
                         ["tau_status_counts"], {"CONSTANT_SCORE": 1})

    def test_scene_mean_cells_not_replication_or_candidate_weight(self):
        compact, labels, predictions = example()
        for i, cell in enumerate(scene.CELLS):
            predictions["generative"][cell] += np.float32(i/10)
        result = scene.diagnose_scene(compact, labels, predictions)
        expected = np.mean([np.mean((predictions["generative"][cell].astype(float)
                                    -result["arrays"]["targets"]["u32"].astype(float))**2)
                            for cell in scene.CELLS])
        self.assertAlmostEqual(result["arm_summary"]["generative"]["full"]["loss_like"]["mean"], expected)
        self.assertEqual(result["paired"]["generative-local"]["full"]["loss_like"]["mean"], expected)

    def test_strata_are_not_substituted_for_full_and_singletons_remain(self):
        ps = sorted([((0, 1, 2, 3), (4, 5, 6, 7), tuple(range(8, 16))),
                     (tuple(range(8)), tuple(range(8, 16))),
                     tuple(tuple(range(i, i+4)) for i in range(0, 16, 4))])
        result = scene.diagnose_scene(*example(ps, 16))
        self.assertEqual(result["arm_summary"]["generative"]["full"]["tau"]["mean"], 1.)
        self.assertIsNone(result["arm_summary"]["generative"]["k"]["tau"]["mean"])
        method = result["learned"]["generative"][scene.CELLS[0]]
        self.assertEqual(method["schemes"]["k"]["tau_status_counts"], {"INSUFFICIENT_PAIRS": 3})
        self.assertEqual(method["summary"]["k"]["ari"]["defined"], 3)
        self.assertEqual(len(result["oracles"]["k"]), 3)

    def test_empty_scene_retains_all_methods_without_imputation(self):
        result = scene.diagnose_scene(*example([]))
        self.assertEqual(result["candidate_count"], 0)
        self.assertEqual(result["strata"]["full"], {"all": []})
        self.assertEqual(len(result["learned"]["local"]), 9)
        for arm in scene.ARMS:
            for scheme in result["arm_summary"][arm].values():
                for metric in scheme.values():
                    self.assertIsNone(metric["mean"])
                    self.assertEqual(metric["defined"], 0)
        self.assertEqual(result["arrays"]["targets"]["u32"].shape, (0, 2))
        self.assertIsNone(result["paired"]["generative-local"]["full"]["tau"]["mean"])

    def test_reject_missing_cell_extra_arm_wrong_dtype_or_extent(self):
        compact, labels, predictions = example()
        del predictions["local"][scene.CELLS[0]]
        with self.assertRaises(ValueError):
            scene.diagnose_scene(compact, labels, predictions)
        compact, labels, predictions = example()
        predictions["other"] = predictions["local"]
        with self.assertRaises(ValueError):
            scene.diagnose_scene(compact, labels, predictions)
        for array in (np.zeros((2, 2), np.float64), np.zeros((1, 2), np.float32)):
            compact, labels, predictions = example()
            predictions["local"][scene.CELLS[0]] = array
            with self.assertRaises(ValueError):
                scene.diagnose_scene(compact, labels, predictions)


if __name__ == "__main__":
    unittest.main()
