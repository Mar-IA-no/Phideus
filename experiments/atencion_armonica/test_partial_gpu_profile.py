"""CPU-only orchestration fixtures; never enters the GPU worker."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("partial_gpu_profile_fixture",
        Path(__file__).with_name("profile_partial_compatibility_gpu.py"))
profile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profile)


class ProfileTests(unittest.TestCase):
    def test_projection_all_fifteen_runs_and_evaluation(self):
        rows = [{"arm": arm, "train_step_seconds": [.02,.03,.025],
                 "eval_step_seconds": [.01,.012,.008], "peak_reserved_bytes": 1024**3}
                for arm in profile.ARMS]
        result = profile.project_resource_cost(rows)
        self.assertEqual(result["training_seconds"], 1440)
        self.assertAlmostEqual(result["evaluation_seconds"], 8.64)
        self.assertEqual(result["status"], "WITHIN_ESTIMATE")
        rows[0]["peak_reserved_bytes"] = 8*1024**3
        self.assertEqual(profile.project_resource_cost(rows)["status"], "REVIEW_REQUIRED")
        with self.assertRaises(ValueError):
            profile.project_resource_cost(rows[:-1])

    def test_missing_receipt_does_not_spawn_or_import_torch(self):
        root = profile.ROOT/".agent-work/phideus-geometric-rebase-20260907/profile-tests"
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as tmp:
            output = Path(tmp)/"output"
            with patch.object(profile.subprocess, "run") as run:
                with self.assertRaises(ValueError):
                    profile.launch(output, Path(tmp)/"absent")
                run.assert_not_called()
            self.assertFalse(output.exists())
        self.assertNotIn("torch", sys.modules)

    def test_receipt_mutation_blocks_completion_and_worker_before_torch(self):
        root = profile.ROOT/".agent-work/phideus-geometric-rebase-20260907/profile-tests"
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as tmp:
            output, receipt = Path(tmp)/"output", Path(tmp)/"receipt.json"
            profile.write_json(receipt, {"fixture": "original"})
            def fake_child(*args, **kwargs):
                # Deliberate mutation of this test-owned temporary receipt only.
                with receipt.open("w") as handle:
                    json.dump({"fixture": "changed"}, handle)
            with patch.object(profile.subprocess, "run", side_effect=fake_child):
                with self.assertRaisesRegex(RuntimeError, "receipt changed"):
                    profile.launch(output, receipt)
            self.assertTrue((output/"FAILURE.json").is_file())
            self.assertFalse((output/"manifest.json").exists())
            with self.assertRaisesRegex(RuntimeError, "receipt changed"):
                profile.worker(output, receipt)
        self.assertNotIn("torch", sys.modules)


if __name__ == "__main__":
    unittest.main()
