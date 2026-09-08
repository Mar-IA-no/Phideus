"""Packed input parity and corruption fixtures; no prospective observations."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from experiments.atencion_armonica.test_learned_partition_cache import BASE, fixture
from src.atencion_armonica.learned_partition_core import model_inputs, READER_SEEDS
from src.atencion_armonica.learned_partition_inputs import pack_inputs, read_inputs, validate_arrays
from src.atencion_armonica.learned_partition_inference import predict_inputs
from src.atencion_armonica.learned_partition_model import PartitionCostHead


class PackedInputTests(unittest.TestCase):
    def test_exact_ragged_roundtrip_forward_parity_and_closed_masks(self):
        BASE.mkdir(parents=True, exist_ok=True)
        torch.set_num_threads(1)
        full = model_inputs(fixture(), {"mean": np.zeros(5), "scale": np.ones(5)}, "shared_source")
        used = np.flatnonzero(full["incidence"][0] > 0)
        small = {"groups": full["groups"][used].copy(), "globals": full["globals"][:1].copy(),
                 "incidence": full["incidence"][:1, used].copy()}
        original = [full, small]*256
        ids = list(range(512, 1024))
        with tempfile.TemporaryDirectory(dir=BASE) as folder:
            path = Path(folder)/"inputs.npz"
            pack_inputs(path, original, scene_ids=ids, dim=9)
            restored = read_inputs(path, scene_ids=ids, dim=9)
            for before, after in zip(original, restored):
                for key in before:
                    np.testing.assert_array_equal(before[key], after[key])
            model = PartitionCostHead("shared_source", READER_SEEDS[0])
            for before, after in zip(predict_inputs(model, original), predict_inputs(model, restored)):
                np.testing.assert_array_equal(before, after)
            with self.assertRaises(FileExistsError):
                pack_inputs(path, original, scene_ids=ids, dim=9)
            with self.assertRaises(ValueError):
                read_inputs(path, scene_ids=list(range(512)), dim=9)
            with np.load(path, allow_pickle=False) as raw:
                baseline = {k: raw[k] for k in raw.files}
            for mutate in (
                lambda a: a.update(truth=np.zeros(512)),
                lambda a: a["groups"].__setitem__((0, -1, 0), 1.),
                lambda a: a["incidence"].__setitem__((0, 0, -1), .1),
                lambda a: a["group_mask"].__setitem__((0, 0), False),
                lambda a: a["candidate_mask"].__setitem__((0, -1), True),
                lambda a: a["globals"].__setitem__((0, 0, 0), np.nan),
            ):
                arrays = {k: v.copy() for k, v in baseline.items()}
                mutate(arrays)
                with self.assertRaises(ValueError):
                    validate_arrays(arrays, scene_ids=ids, dim=9)


if __name__ == "__main__":
    unittest.main()
