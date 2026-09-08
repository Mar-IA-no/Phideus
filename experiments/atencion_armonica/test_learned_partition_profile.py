"""Profile shape and authorization checks, not actual resource profiling."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from experiments.atencion_armonica.test_learned_partition_data import BASE
from src.atencion_armonica import learned_partition_profile as profile


class ProfileTests(unittest.TestCase):
    def test_timing_blocks_execute_every_operation_and_keep_total_seconds(self):
        calls = []
        with patch.object(profile.time, "monotonic", side_effect=[0., 32., 32., 64., 64., 96.]):
            measured = profile.timed_repeats(lambda: calls.append(1), operations_per_repeat=32)
        self.assertEqual(len(calls), 96)
        self.assertEqual(measured, [32., 32., 32.])

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

    def test_io_primitives_measure_real_files_with_closed_denominators(self):
        # A helper check, not a FULL_IMPLEMENTATION receipt or campaign profile.
        with tempfile.TemporaryDirectory(dir=BASE) as folder:
            result = profile.validation_io_measurements(Path(folder)/"io")
            profile.resources.validate_validation_io(result)
            self.assertEqual(result["bundle"]["files"], 3075)
            self.assertEqual(result["inventory"]["entries"], 3078)
            for dim in (8, 9):
                packed = result["packed"][str(dim)]
                self.assertLess(packed["compressed_bytes"], packed["uncompressed_upper_bytes"])
                self.assertEqual(len(packed["seconds"]), 3)

    def test_snapshot_and_calibration_primitives_use_actual_loaders(self):
        import torch
        from experiments.atencion_armonica.test_learned_partition_training import advance
        from src.atencion_armonica.learned_partition_training import TrainingKernel
        from src.atencion_armonica.learned_partition_snapshots import write_snapshot
        torch.set_num_threads(1)
        kernel = TrainingKernel("shared_source", 2026090891, binding={"mechanical": "profile_helper"}, count=32)
        with tempfile.TemporaryDirectory(dir=BASE) as folder:
            refs = [write_snapshot(folder, "initial", kernel)]
            for i in range(10):
                advance(kernel)
                refs.append(write_snapshot(folder, f"step_{i+1}", kernel, parents=[refs[-1]]))
            result = profile.snapshot_and_calibration_measurements(folder, kernel, refs, profile.mechanical_inputs(9))
            self.assertEqual(result["snapshot_unique_counts"], [11]*3)
            self.assertEqual(result["calibration_scene_count"], 512)
            self.assertEqual(result["calibration_candidate_count"], 64)
            self.assertEqual(len(result["calibration_read_seconds"]), 3)

    def test_semantic_primitives_reuse_fixed_observation_and_production_target_reader(self):
        from src.atencion_armonica.shared_partial_data import mechanical_fixture
        from src.atencion_armonica.structured_source_artifacts import write_json, write_npz
        from src.atencion_armonica.learned_partition_runner import read_target_metrics
        obs, truth = mechanical_fixture(4)
        record = profile.feature_record(obs)
        matrix = (4*record["pair_support"]-2).astype(np.float32)
        scored = profile.score_scene(np.asarray(obs["log_f"], np.float32), matrix, record["pair_support"],
            record["triples"], record["residual_cents"], split_seed=obs["split_seed"], scene_id=obs["scene_id"],
            fit_cache=profile.GroupFitCache(profile.SourceFitter()))
        row = profile.observable_features(scored, matrix)
        with tempfile.TemporaryDirectory(dir=BASE) as folder:
            case = Path(folder)
            write_json(case/"observation.json", obs)
            write_npz(case/"features.npz", **record)
            for i in range(3):
                write_json(case/f"pool_{i}.json", scored)
                profile.save_rows(case/f"rows_{i}.npz", row)
            result = profile.semantic_measurements(case, obs, truth, [row]*3, [matrix]*3)
            profile.resources.validate_semantic_measurements([result, result])
            read_target_metrics(case/"targets_0.npz", case/"metrics_0.json", n=32, candidates=row.candidates)
            with self.assertRaises(ValueError):
                read_target_metrics(case/"targets_0.npz", case/"metrics_0.json", n=32, candidates=row.candidates[:-1])


if __name__ == "__main__":
    unittest.main()
