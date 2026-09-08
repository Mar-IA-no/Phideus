"""No training: deterministic CPU stub tests raw inference and exact persistence."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from src.atencion_armonica.partial_compatibility_cache import feature_record
from src.atencion_armonica.partial_compatibility_inference import collect_logits, load_logits, save_logits
from src.atencion_armonica.shared_partial_data import mechanical_fixture


class Stub(torch.nn.Module):
    def forward(self, batch):
        q = batch["tokens"][..., 0]
        return q[:, :, None]+q[:, None, :]


class InferenceTests(unittest.TestCase):
    def test_observation_only_padding_and_exact_roundtrip(self):
        observations = [mechanical_fixture(k)[0] for k in (3, 4)]
        records = [feature_record(o) for o in observations]
        with patch("torch.cuda.is_available", side_effect=AssertionError("GPU forbidden")):
            raw = collect_logits(Stub(), records, device="cpu")
        self.assertEqual([m.shape for m in raw], [(24, 24), (32, 32)])
        for result, record in zip(raw, records):
            q = record["tokens"][:, 0]
            np.testing.assert_array_equal(result, q[:, None]+q[None, :])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"logits.npz"
            save_logits(path, raw, observations)
            restored = load_logits(path, observations)
            for a, b in zip(raw, restored):
                np.testing.assert_array_equal(a, b)
            with self.assertRaises(FileExistsError):
                save_logits(path, raw, observations)
            with self.assertRaises(ValueError):
                load_logits(path, observations[::-1])
        with self.assertRaises(ValueError):
            collect_logits(Stub(), [{**records[0], "source_ids": [0]*24}], device="cpu")

    def test_same_size_swapped_scene_and_split_id_are_rejected(self):
        first = mechanical_fixture(3)[0]
        second = {**first, "scene_id": first["scene_id"]+1, "log_f": [q+.001 for q in first["log_f"]]}
        observations = [first, second]
        raw = collect_logits(Stub(), [feature_record(o) for o in observations], device="cpu")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"logits.npz"
            save_logits(path, raw, observations)
            for expected in (observations[::-1], [{**first, "split_seed": first["split_seed"]+1}, second],
                             [{**first, "log_f": second["log_f"]}, second],
                             [{**first, "log_f": first["log_f"][::-1]}, second]):
                with self.assertRaises(ValueError):
                    load_logits(path, expected)


if __name__ == "__main__":
    unittest.main()
