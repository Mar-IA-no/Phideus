"""Mechanical checks only: no scene producer, campaign observations or CUDA."""
import unittest

import numpy as np
import torch

from src.atencion_armonica.learned_partition_core import (ARMS, initialization_seeds, partition_errors,
                                                         observable_features, fit_normalizer, model_inputs)
from src.atencion_armonica.learned_partition_model import PartitionCostHead, partition_cost_loss
from src.atencion_armonica.structured_source_reader import signature, partition_energies


def batch():
    # Two scenes, each with one singleton partition and one two-group partition.
    x = torch.arange(2*6*9, dtype=torch.float32).reshape(2, 6, 9)/100
    incidence = torch.zeros((2, 2, 6), dtype=torch.float32)
    incidence[:, 0, :4] = .25
    incidence[:, 1, 4:] = .5
    return {"groups": x, "globals": torch.ones(2, 2, 6), "incidence": incidence,
            "group_mask": torch.ones(2, 6, dtype=torch.bool),
            "candidate_mask": torch.ones(2, 2, dtype=torch.bool)}


class CoreTests(unittest.TestCase):
    def test_observable_features_and_normalizer(self):
        n = 8
        a = np.arange(n*n, dtype=np.float32).reshape(n, n)/10
        z = (a+a.T).astype(np.float32)
        partitions = sorted([signature([[i] for i in range(n)]),
                             signature([list(range(4)), list(range(4, 8))])])
        groups = sorted({g for p in partitions for g in p})
        costs = {arm: [.25 if len(g) >= 3 else 0. for g in groups] for arm in ARMS[1:]}
        rows = partition_energies(z.astype(np.float64), partitions, groups, costs)
        scored = {"pool": {"canonical_to_observed": list(range(n))},
                  "groups": [{"members": list(g), "size": len(g),
                              "costs": {arm: costs[arm][i] for arm in costs}}
                             for i, g in enumerate(groups)], "candidates": rows}
        features = observable_features(scored, z)
        np.testing.assert_allclose(features.incidence.sum(axis=1), 1.)
        np.testing.assert_array_equal(features.group_features[0, 3:7], np.zeros(4))
        normalizer = fit_normalizer([features], expected_count=1)
        inputs = model_inputs(features, normalizer, "shared_source")
        np.testing.assert_allclose(inputs["groups"][:, 3:7].mean(axis=0), np.zeros(4), atol=2e-7)
        self.assertEqual(inputs["groups"].shape, (len(groups), 9))
        self.assertEqual(model_inputs(features, normalizer, "pairs_structure")["groups"].shape[1], 8)
        before = features.group_features.copy()
        inputs["groups"][:] = -123
        np.testing.assert_array_equal(before, features.group_features)
        with self.assertRaises(ValueError):
            fit_normalizer([features], expected_count=2)
        scored["candidates"][0]["pair_energy"] += 1
        with self.assertRaises(ValueError):
            observable_features(scored, z)

    def test_valid_padding_does_not_change_outputs(self):
        model = PartitionCostHead("shared_source", 2026090891)
        b = batch()
        expected = model(b)
        padded = {"groups": torch.nn.functional.pad(b["groups"], (0, 0, 0, 2)),
                  "globals": torch.nn.functional.pad(b["globals"], (0, 0, 0, 1)),
                  "incidence": torch.nn.functional.pad(b["incidence"], (0, 2, 0, 1)),
                  "group_mask": torch.nn.functional.pad(b["group_mask"], (0, 2)),
                  "candidate_mask": torch.nn.functional.pad(b["candidate_mask"], (0, 1))}
        torch.testing.assert_close(model(padded)[:, :2], expected, rtol=0, atol=2e-7)

    def test_partition_errors_and_relabeling(self):
        y = np.array([0, 0, 1, 1])
        np.testing.assert_array_equal(partition_errors([[0, 1], [2, 3]], y)["raw"], [0, 0])
        np.testing.assert_allclose(partition_errors([[0], [1], [2], [3]], y)["normalized"], [.5, 0])
        np.testing.assert_allclose(partition_errors([[0, 1, 2, 3]], y)["normalized"], [0, .5])
        np.testing.assert_array_equal(partition_errors([[0, 2], [1, 3]], y)["raw"],
                                      partition_errors([[3, 1], [2, 0]], np.array([7, 7, 19, 19]))["raw"])
        with self.assertRaises(ValueError):
            partition_errors([[0, 1], [1, 3]], y)

    def test_initialization_and_active_shapes(self):
        self.assertEqual(len({r["torch_seed"] for r in initialization_seeds()}), 18)
        models = [PartitionCostHead(a, 2026090891) for a in ARMS]
        self.assertEqual([sum(p.numel() for p in m.parameters()) for m in models], [1643, 1650, 1650, 1650])
        for other in models[2:]:
            for name, value in models[1].state_dict().items():
                self.assertTrue(torch.equal(value, other.state_dict()[name]))
        baseline, physical = models[:2]
        self.assertTrue(torch.equal(baseline.group1.weight[:32], physical.group1.weight[:, :8]))
        self.assertTrue(torch.equal(baseline.group2.weight[:, :32], physical.group2.weight))
        self.assertTrue(torch.equal(baseline.partition1.weight, physical.partition1.weight))
        self.assertTrue(torch.equal(baseline.partition2.weight, physical.partition2.weight))

    def test_head_group_invariance_candidate_equivariance(self):
        model = PartitionCostHead("shared_source", 2026090891)
        original = batch()
        output = model(original)
        perm = torch.tensor([5, 0, 4, 2, 1, 3])
        changed = {k: v.clone() for k, v in original.items()}
        changed["groups"] = original["groups"][:, perm]
        changed["group_mask"] = original["group_mask"][:, perm]
        changed["incidence"] = original["incidence"][:, :, perm]
        torch.testing.assert_close(model(changed), output, rtol=0, atol=2e-7)
        for k in ("globals", "candidate_mask", "incidence"):
            changed[k] = changed[k][:, [1, 0]]
        torch.testing.assert_close(model(changed), output[:, [1, 0]], rtol=0, atol=2e-7)

    def test_loss_equal_scene_mass_and_padding(self):
        pred = torch.tensor([[[2., 2.], [99., 99.]], [[0., 0.], [0., 0.]]], requires_grad=True)
        mask = torch.tensor([[True, False], [True, True]])
        loss = partition_cost_loss(pred, torch.zeros_like(pred), mask)
        self.assertEqual(loss.item(), 2.)
        loss.backward()
        self.assertEqual(pred.grad[0, 1].abs().sum().item(), 0.)

    def test_no_truth_and_no_invalid_masks(self):
        model = PartitionCostHead("shared_source", 2026090891)
        b = batch()
        with self.assertRaises(ValueError):
            model({**b, "target": torch.zeros(2, 2, 2)})
        b["group_mask"][0, 0] = False
        with self.assertRaises(ValueError):
            model(b)

    def test_gradient_paths(self):
        for arm in ARMS:
            model = PartitionCostHead(arm, 2026090891)
            b = batch()
            if arm == ARMS[0]:
                b["groups"] = b["groups"][:, :, :8]
            out = model(b)
            partition_cost_loss(out, torch.zeros_like(out), b["candidate_mask"]).backward()
            for name, parameter in model.named_parameters():
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertGreater(model.group1.weight.grad.abs().sum().item(), 0.)
            self.assertGreater(model.group2.weight.grad.abs().sum().item(), 0.)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
