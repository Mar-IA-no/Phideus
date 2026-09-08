"""Fixed feature tensors only; no scene producer or campaign optimization."""
import copy
import unittest

import numpy as np
import torch

from experiments.atencion_armonica.test_learned_partition_cache import fixture
from src.atencion_armonica.learned_partition_core import fit_normalizer, model_inputs
from src.atencion_armonica.learned_partition_model import PartitionCostHead
from src.atencion_armonica.learned_partition_inference import (
    intervention_inputs, intervention_report, predict_inputs)


class InferenceTests(unittest.TestCase):
    def test_support_matches_scalar_reference_for_dense_sparse_and_permuted_inputs(self):
        from src.atencion_armonica.learned_partition_profile import mechanical_inputs
        from src.atencion_armonica.learned_partition_readout import input_support
        from experiments.atencion_armonica.test_learned_partition_scalar_reference import scalar_support_reference
        for mode in ("dense", "sparse", "weighted"):
            original = mechanical_inputs(9)
            original["groups"][:, :8] = 0
            original["groups"][:, 8] = np.arange(94, dtype=np.float32)/94
            if mode == "sparse":
                original["incidence"][:] = 0
                for c, weights in enumerate(original["incidence"]):
                    weights[np.array([0, 1, 2])+c] = [.25, .25, .5]
            elif mode == "weighted":
                weights = np.arange(1, 95, dtype=np.float32)
                original["incidence"][:] = weights/weights.sum(dtype=np.float32)
            for transform in ("identity", "zero", "reverse", "signed_zero"):
                changed = copy.deepcopy(original)
                if transform == "zero":
                    changed["groups"][:, 8] = 0
                elif transform == "reverse":
                    changed["groups"][:, 8] = changed["groups"][::-1, 8]
                elif transform == "signed_zero":
                    changed["groups"][0, 8] = -0.
                expected = scalar_support_reference(original, changed)
                self.assertEqual(input_support(original, changed), expected)
                if mode == "dense" and transform == "reverse":
                    self.assertEqual(expected["status"], "INPUT_UNCHANGED")
                    self.assertTrue(all(expected["aligned_candidate_mask"]))
            for mutate in (lambda a: a["groups"].__setitem__((0, 1), 1.),
                           lambda a: a["groups"].__setitem__((0, 8), float("nan")),
                           lambda a: a["incidence"].__setitem__((0, 0), -1.),
                           lambda a: a.update(groups=a["groups"].astype(np.float64))):
                changed = copy.deepcopy(original)
                mutate(changed)
                for fn in (scalar_support_reference, input_support):
                    with self.assertRaises(ValueError):
                        fn(original, changed)

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_prediction_preserves_rng_parameters_and_mode(self):
        row = fixture()
        normalizer = fit_normalizer([row], expected_count=1)
        inputs = model_inputs(row, normalizer, "shared_source")
        model = PartitionCostHead("shared_source", 2026090891)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        rng = torch.get_rng_state().clone()
        predicted = predict_inputs(model, [inputs]*33)
        self.assertTrue(model.training)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, before[key]))
        self.assertEqual(len(predicted), 33)
        for a in predicted:
            self.assertEqual(a.shape, (len(row.candidates), 2))
            self.assertEqual(a.dtype, np.float32)
            self.assertTrue(np.isfinite(a).all())
        model.eval()
        with self.assertRaises(ValueError):
            predict_inputs(model, [{**inputs, "truth": np.arange(8)}])
        self.assertFalse(model.training)

    def test_interventions_modify_only_declared_channel(self):
        row = fixture()
        ids = [i for i, g in enumerate(row.groups) if len(g) == 4]
        row.costs["shared_source"][ids] = [.2, .8]
        row.costs["decoupled_source"][ids] = [.8, .2]
        normalizer = fit_normalizer([row], expected_count=1)
        original = model_inputs(row, normalizer, "shared_source")
        model = PartitionCostHead("shared_source", 2026090891)
        for intervention in ("zero", "rotate_one", "original_sham"):
            changed = intervention_inputs(row, normalizer, "shared_source", intervention)
            for key in ("globals", "incidence"):
                np.testing.assert_array_equal(original[key], changed[key])
            np.testing.assert_array_equal(original["groups"][:, :8], changed["groups"][:, :8])
            if intervention == "zero":
                np.testing.assert_array_equal(changed["groups"][:, 8], np.zeros(len(row.groups)))
            else:
                np.testing.assert_array_equal(changed["groups"][ids, 8], np.asarray([.8, .2], np.float32))
            predictions = predict_inputs(model, [original, changed])
            report = intervention_report(row, original, changed, *predictions)
            self.assertEqual(report["input"]["by_group_size"]["4"]["changed_group_count"], 2)
            self.assertEqual(len(report["prediction"]["component_delta"]), len(row.candidates))
        np.testing.assert_array_equal(row.costs["shared_source"][ids], [.2, .8])
        with self.assertRaises(ValueError):
            intervention_inputs(row, normalizer, "pairs_structure", "zero")
        with self.assertRaises(ValueError):
            intervention_inputs(row, normalizer, "local_compatibility", "original_sham")


if __name__ == "__main__":
    unittest.main()
