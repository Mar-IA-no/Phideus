"""CPU tensor parity: cache must not change the frozen observation pathway."""
import unittest
from unittest.mock import patch

import torch

from src.atencion_armonica.partial_compatibility_cache import feature_record
from src.atencion_armonica.partial_compatibility_learning import (
    collate_cached_observations, collate_observations,
)
from src.atencion_armonica.shared_partial_data import mechanical_fixture


class CachedLearningTests(unittest.TestCase):
    def test_bitwise_parity_no_recompute_and_no_accelerator_query(self):
        observations = [mechanical_fixture(k)[0] for k in (3, 4)]
        records = [feature_record(obs) for obs in observations]
        direct = collate_observations(observations)
        with patch("torch.cuda.is_available", side_effect=AssertionError("GPU forbidden")), \
             patch("src.atencion_armonica.partial_compatibility_learning.frequency_features",
                   side_effect=AssertionError("cache must not recompute")):
            cached = collate_cached_observations(records)
        self.assertEqual(set(direct), set(cached))
        for key in direct:
            self.assertTrue(torch.equal(direct[key], cached[key]), key)
        with self.assertRaises(ValueError):
            collate_cached_observations([{**records[0], "source_ids": [0]*24}])


if __name__ == "__main__":
    unittest.main()
