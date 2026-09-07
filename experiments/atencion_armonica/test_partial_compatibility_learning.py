"""Small CPU forward/gradient fixtures only, not the development preflight."""
import unittest
from unittest.mock import patch

import numpy as np
import torch

from src.atencion_armonica.partial_compatibility_learning import (
    ARMS, collate_observations, collate_targets, loss_components, objective, seed_cpu,
)
from src.atencion_armonica.pairformer import build_model


def observation(values, scene_id=0):
    return {"log_f": np.asarray(values, dtype=np.float32).tolist(),
            "scene_id": scene_id, "split_seed": 2026090710}


class LearningTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.obs = [observation([-.5, -.2, .3]), observation([-.7, -.3, 0., .2, .8], 1)]
        self.labels = [[0, 0, 1], [0, 1, 0, 1, 1]]

    def test_cpu_seed_does_not_call_accelerator_seeding(self):
        with patch("torch.cuda.manual_seed_all", side_effect=AssertionError("CUDA seed called")):
            with patch("torch.cuda.is_available", side_effect=AssertionError("CUDA queried")):
                seed_cpu(17)
                a = torch.rand(4, device="cpu")
                seed_cpu(17)
                torch.testing.assert_close(a, torch.rand(4, device="cpu"), rtol=0, atol=0)

    def test_supervision_does_not_enter_observations(self):
        batch = collate_observations(self.obs)
        self.assertNotIn("target", batch)
        for forbidden in ("source_id", "source_ids", "amp", "f0", "beta", "k"):
            bad = dict(self.obs[0], **{forbidden: [0]})
            with self.assertRaises(ValueError):
                collate_observations([bad])
        a = collate_targets(self.labels, batch)
        b = collate_targets([[7, 7, -4], [7, -4, 7, -4, -4]], batch)
        torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_scene_means_not_global_pair_counts(self):
        batch = collate_observations(self.obs)
        targets = collate_targets(self.labels, batch)
        logits = torch.arange(50, dtype=torch.float32).reshape(2, 5, 5)/30-1
        logits = (logits+logits.transpose(1, 2))/2
        both = loss_components(logits, batch, targets)
        for row in range(2):
            single = collate_observations([self.obs[row]])
            target = collate_targets([self.labels[row]], single)
            n = len(self.obs[row]["log_f"])
            part = loss_components(logits[row:row+1, :n, :n], single, target)
            for key in both:
                torch.testing.assert_close(both[key][row], part[key][0])
        for arm in ARMS:
            penalty = ARMS[arm][1]
            expected = both["bce"]+(0 if penalty == "bce" else .1*both[penalty])
            torch.testing.assert_close(objective(both, arm), expected.mean())

    def test_penalties_against_scalar_formula(self):
        batch = collate_observations([self.obs[1]])
        target = collate_targets([self.labels[1]], batch)
        seed_cpu(21)
        logits = torch.randn(1, 5, 5, requires_grad=True)
        parts = loss_components(logits, batch, target)
        p = torch.sigmoid(logits).detach().numpy()[0]
        physical, sham, trans = [], [], []
        for t, w, ws in zip(batch["triples"][0].tolist(), batch["physical_weights"][0].tolist(),
                            batch["sham_weights"][0].tolist()):
            i, j, k = t
            x, y, z = p[i, j], p[i, k], p[j, k]
            physical.append(w*x*y*z)
            sham.append(ws*x*y*z)
            trans.append((max(0,x*z-y)**2+max(0,x*y-z)**2+max(0,y*z-x)**2)/3)
        for key, expected in (("physical", physical), ("sham", sham), ("transitivity", trans)):
            self.assertAlmostEqual(parts[key].item(), float(np.mean(expected)), places=7)
        parts["physical"].sum().backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(logits.grad.norm().item(), 0)

    def test_transitivity_inactive_then_violated(self):
        batch = collate_observations([self.obs[0]])
        target = collate_targets([self.labels[0]], batch)
        zero = torch.zeros(1, 3, 3, requires_grad=True)
        self.assertEqual(loss_components(zero, batch, target)["transitivity"].item(), 0)
        probabilities = torch.tensor([[[.5, .95, .05], [.95, .5, .95], [.05, .95, .5]]])
        logits = torch.logit(probabilities).requires_grad_()
        trans = loss_components(logits, batch, target)["transitivity"].sum()
        trans.backward()
        self.assertGreater(trans.item(), 0)
        self.assertGreater(logits.grad.norm().item(), 0)

    def test_sham_tie_keeps_bce_scene(self):
        obs = [observation([0., 0., .2, .3])]
        batch = collate_observations(obs)
        target = collate_targets([[0, 1, 1, 0]], batch)
        parts = loss_components(torch.zeros(1, 4, 4), batch, target)
        self.assertFalse(batch["sham_evaluable"][0])
        self.assertEqual(parts["sham"].item(), 0)
        torch.testing.assert_close(objective(parts, "pairs_sham"), parts["bce"].mean())

    def test_historical_configs_forward_gradient_and_padding(self):
        with patch("torch.cuda.manual_seed_all", side_effect=AssertionError("CUDA seed called")):
            with patch("torch.cuda.is_available", side_effect=AssertionError("CUDA queried")):
                batch = collate_observations(self.obs)
                target = collate_targets(self.labels, batch)
                for name, arm in (("B-local", "pairs_compatibility"), ("A-rich", "tokens_descriptors")):
                    seed_cpu(23)
                    model = build_model(name).cpu()
                    logits = model(batch)
                    self.assertEqual(logits.shape, (2, 5, 5))
                    torch.testing.assert_close(logits, logits.transpose(1, 2), rtol=0, atol=0)
                    single = collate_observations([self.obs[0]])
                    torch.testing.assert_close(logits[0, :3, :3], model(single)[0], atol=2e-6, rtol=2e-5)
                    loss = objective(loss_components(logits, batch, target), arm)
                    loss.backward()
                    grads = [p.grad for p in model.parameters() if p.grad is not None]
                    self.assertTrue(all(torch.isfinite(g).all() for g in grads))
                    self.assertGreater(sum(g.norm().item() for g in grads), 0)


if __name__ == "__main__":
    unittest.main()
