"""Small CPU architecture sanity; not evidence of learned geometric generalization."""
import unittest
from unittest.mock import patch

import numpy as np
import torch

from src.atencion_armonica.partial_compatibility_cache import feature_record
from src.atencion_armonica.partial_compatibility_learning import collate_cached_observations, seed_cpu
from src.atencion_armonica.pairformer import build_model
from src.atencion_armonica.shared_partial_data import mechanical_fixture


class ModelGeometryTests(unittest.TestCase):
    def test_both_fixed_architectures_respect_observed_permutation_and_synthetic_scale(self):
        torch.set_num_threads(1)
        original = mechanical_fixture(3)[0]
        q = np.asarray(original["log_f"], dtype=np.float32)
        permutation = np.arange(len(q))[::-1].copy()
        shifted = q.astype(np.float64)+np.log(1.7)
        scaled = (shifted-shifted.mean()).astype(np.float32)
        observations = [original, {**original, "log_f": q[permutation].tolist()},
                        {**original, "log_f": scaled.tolist()}]
        batch = collate_cached_observations([feature_record(o) for o in observations])
        with patch("torch.cuda.is_available", side_effect=AssertionError("GPU forbidden")):
            for name in ("B-local", "A-rich"):
                seed_cpu(2026090721)
                model = build_model(name).eval()
                with torch.inference_mode():
                    logits = model(batch).numpy()
                np.testing.assert_allclose(logits[1], logits[0][np.ix_(permutation, permutation)], atol=1e-5, rtol=1e-5)
                np.testing.assert_allclose(logits[2], logits[0], atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
