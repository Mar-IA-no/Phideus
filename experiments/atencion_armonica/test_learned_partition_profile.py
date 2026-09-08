"""Profile shape and authorization checks, not actual resource profiling."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from experiments.atencion_armonica.test_learned_partition_data import BASE
from src.atencion_armonica import learned_partition_profile as profile


class ProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        BASE.mkdir(parents=True, exist_ok=True)

    def test_dense_padding_tensor_has_no_observation_or_truth(self):
        for dim in (8, 9):
            rows = profile.mechanical_inputs(dim)
            self.assertEqual(set(rows), {"groups", "globals", "incidence"})
            self.assertEqual(rows["groups"].shape, (94, dim))
            self.assertEqual(rows["globals"].shape, (64, 6))
            self.assertEqual(rows["incidence"].shape, (64, 94))
            self.assertTrue((rows["incidence"] > 0).all())
            np.testing.assert_allclose(rows["incidence"].sum(1), 1, rtol=0, atol=2e-7)
            self.assertTrue(all(a.dtype == np.float32 and np.isfinite(a).all() for a in rows.values()))

    def test_profile_denial_precedes_outputs_and_gpu_lease(self):
        with tempfile.TemporaryDirectory(dir=BASE) as folder:
            output = Path(folder)/"not_created"
            with patch.object(profile.gate, "common_binding", return_value={"fixture": 1}), \
                 patch.object(profile.gate, "verify_audit", side_effect=PermissionError), \
                 patch("src.atencion_armonica.structured_source_profile.gpu_lease") as lease:
                with self.assertRaises(PermissionError):
                    profile.geometry_profile(output, audit={})
                with self.assertRaises(PermissionError):
                    profile.training_profile(output, audit={}, device="cuda:0", gpu_grant={})
                lease.assert_not_called()
                self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
