"""Synthetic archive identity fixtures; no Torch or campaign data."""
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

from src.atencion_armonica.source_artifacts import load_ordered_logits, validate_partition


class ArtifactTests(unittest.TestCase):
    def fixture(self, path):
        obs = [{"scene_id": i, "split_seed": 11, "log_f": [i+.1, i+.2, i+.3]} for i in range(2)]
        matrices = [np.array([[0., .2+i, .3], [.2+i, 0., -.5], [.3, -.5, 0.]], np.float32) for i in range(2)]
        arrays = dict(logits=np.concatenate([a.ravel() for a in matrices]), sizes=np.array([3, 3], np.int64),
                      offsets=np.array([0, 9, 18], np.int64), scene_ids=np.array([0, 1], np.int64),
                      split_seeds=np.array([11, 11], np.int64), observation_fingerprints=np.array([
                          hashlib.sha256(np.asarray(o["log_f"], dtype="<f4").tobytes()).hexdigest() for o in obs], dtype="U64"))
        np.savez(path, **arrays)
        return obs, matrices, arrays

    def test_bitwise_roundtrip_and_ordered_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"fixture.npz"
            obs, expected, _ = self.fixture(path)
            actual = load_ordered_logits(path, obs)
            for a, b in zip(actual, expected):
                self.assertEqual(a.tobytes(), b.tobytes())
            for changed in (obs[::-1], [{**o, "log_f": o["log_f"][::-1]} for o in obs],
                            [{**o, "split_seed": 12} for o in obs], [{**o, "source_ids": [0, 0, 0]} for o in obs]):
                with self.assertRaises(ValueError):
                    load_ordered_logits(path, changed)
        self.assertNotIn("torch", sys.modules)

    def test_corrupt_shape_dtype_or_symmetry(self):
        with tempfile.TemporaryDirectory() as tmp:
            obs, _, arrays = self.fixture(Path(tmp)/"base.npz")
            changes = [{"sizes": arrays["sizes"].astype(np.int32)}, {"offsets": np.array([0, 9, 17])},
                       {"logits": arrays["logits"].astype(float)}, {"extra": np.array(1)},
                       {"logits": np.arange(18, dtype=np.float32)},
                       {"observation_fingerprints": arrays["observation_fingerprints"].astype("U65")}]
            for i, change in enumerate(changes):
                path = Path(tmp)/f"corrupt_{i}.npz"
                np.savez(path, **(arrays|change))
                with self.assertRaises(ValueError):
                    load_ordered_logits(path, obs)

    def test_partition_exact_cover(self):
        self.assertEqual(validate_partition([[2, 0], [1]], 3), [[2, 0], [1]])
        for bad in ([], [[0], []], [[0, 1], [1, 2]], [[0, 1]], [[0, 1, 3]], [[0., 1, 2]], [[False, 1, 2]]):
            with self.assertRaises(ValueError):
                validate_partition(bad, 3)


if __name__ == "__main__":
    unittest.main()
