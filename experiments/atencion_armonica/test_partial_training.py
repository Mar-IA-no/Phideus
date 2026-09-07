"""CPU-only recipe/persistence tests, not a substitute for the GPU campaign."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from src.atencion_armonica.partial_compatibility_training import (
    RECIPE, SEEDS, atomic_checkpoint, epoch_order, lr_factor, model_digest, training_cell, validate_resume,
)


class TrainingUtilityTests(unittest.TestCase):
    def test_order_is_a_full_reproducible_permutation_shared_by_arms(self):
        for seed in SEEDS:
            a = epoch_order(seed, 0)
            np.testing.assert_array_equal(np.sort(a), np.arange(8192))
            np.testing.assert_array_equal(a, epoch_order(seed, 0))
            self.assertFalse(np.array_equal(a, epoch_order(seed, 1)))
        self.assertFalse(np.array_equal(epoch_order(SEEDS[0], 0), epoch_order(SEEDS[1], 0)))
        for args in ((0, 0), (SEEDS[0], 50), (SEEDS[0], 0, 0)):
            with self.assertRaises(ValueError):
                epoch_order(*args)

    def test_schedule_exact_recipe_and_restore(self):
        self.assertEqual(lr_factor(0), 0)
        self.assertEqual(lr_factor(160), 1)
        self.assertEqual(lr_factor(3200), 0)
        self.assertTrue(all(lr_factor(s) <= lr_factor(s+1) for s in range(160)))
        self.assertTrue(all(lr_factor(s) >= lr_factor(s+1) for s in range(160, 3200)))
        parameter = torch.nn.Parameter(torch.zeros(1))
        optimizer = torch.optim.AdamW([parameter], lr=RECIPE["lr"])
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
        self.assertEqual(optimizer.param_groups[0]["lr"], 0)
        for _ in range(160):
            optimizer.step()
            scheduler.step()
        self.assertEqual(scheduler.last_epoch, 160)
        self.assertEqual(optimizer.param_groups[0]["lr"], RECIPE["lr"])
        for step, total in ((-1, 3200), (3201, 3200), (0, 0), (0, 10)):
            with self.assertRaises(ValueError):
                lr_factor(step, total)

    def test_atomic_checkpoint_and_digest_without_cuda(self):
        with patch("torch.cuda.is_available", side_effect=AssertionError("GPU forbidden")):
            model = torch.nn.Linear(2, 1)
            before = model_digest(model)
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp)/"last_epoch.pt"
                atomic_checkpoint(path, {"model": model.state_dict(), "step": 0})
                atomic_checkpoint(path, {"model": model.state_dict(), "step": 64})
                saved = torch.load(path, weights_only=True)
                self.assertEqual(saved["step"], 64)
                self.assertEqual(list(Path(tmp).iterdir()), [path])
                model.load_state_dict(saved["model"])
                self.assertEqual(before, model_digest(model))
            with torch.no_grad():
                model.weight.add_(1)
            self.assertNotEqual(before, model_digest(model))

    def test_only_whole_epoch_resume(self):
        binding = {"recipe": RECIPE, "seed": SEEDS[0]}
        old = {"binding": binding, "resumable": True, "next_epoch": 10,
               "next_batch": 0, "steps": 640, "scheduler": {"last_epoch": 640},
               "elapsed_total_seconds": 10.0}
        validate_resume(old, binding)
        for changes in ({"resumable": False}, {"next_batch": 1}, {"steps": 641},
                        {"next_epoch": 50}, {"scheduler": {"last_epoch": 639}},
                        {"binding": {}}, {"elapsed_total_seconds": float("nan")}):
            with self.assertRaises(ValueError):
                validate_resume({**old, **changes}, binding)
        # JSON serialization of the binding cannot alter its meaning.
        validate_resume({**old, "binding": json.loads(json.dumps(binding))}, binding)

    def test_supervision_is_loaded_and_setup_failure_is_recorded(self):
        module = "src.atencion_armonica.partial_compatibility_training"
        cache = MagicMock()
        cache.manifest = {"split": "train"}
        cache.__len__.return_value = 8192
        truths = object()
        with tempfile.TemporaryDirectory() as tmp, \
             patch(module+".load_supervision", return_value=truths) as supervision, \
             patch(module+"._training_cell", side_effect=RuntimeError("fixture setup failure")) as inner:
            output = Path(tmp)/"attempt"
            with self.assertRaisesRegex(RuntimeError, "fixture setup failure"):
                training_cell(output, cache, "pairs_descriptors", SEEDS[0], {}, remaining_seconds=10)
            supervision.assert_called_once_with(cache)
            self.assertIs(inner.call_args.args[2], truths)
            self.assertEqual(json.loads((output/"FAILURE.json").read_text())["status"], "INCOMPLETE")
        with tempfile.TemporaryDirectory() as tmp, \
             patch(module+".load_supervision", side_effect=ValueError("corrupt truth")):
            output = Path(tmp)/"attempt"
            with self.assertRaisesRegex(ValueError, "corrupt truth"):
                training_cell(output, cache, "pairs_descriptors", SEEDS[0], {}, remaining_seconds=10)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
