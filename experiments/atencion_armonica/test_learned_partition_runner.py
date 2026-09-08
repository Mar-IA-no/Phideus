"""Observable stage/forward identity fixtures; no model loading or GPU."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from experiments.atencion_armonica.test_learned_partition_data import BASE, ROOT
from src.atencion_armonica import learned_partition_runner as runner
from src.atencion_armonica.structured_source_artifacts import write_json


class RunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        BASE.mkdir(parents=True, exist_ok=True)

    def test_score_envelope_and_unchanged_historical_cpu_envelope(self):
        for seconds in (1200.001, 2399.999, 2400.):
            with patch.object(runner.time, "monotonic", return_value=seconds), \
                 patch.object(runner.resource, "getrusage", return_value=SimpleNamespace(ru_maxrss=1000)):
                self.assertEqual(runner.score_resources(0.), {"seconds": seconds, "peak_rss_bytes": 1024000})
                with self.assertRaises(RuntimeError):
                    runner.cpu_resources(0.)
        for seconds, kib in ((2400.001, 1000), (1., 2*1024**2)):
            with patch.object(runner.time, "monotonic", return_value=seconds), \
                 patch.object(runner.resource, "getrusage", return_value=SimpleNamespace(ru_maxrss=kib)):
                with self.assertRaises(RuntimeError):
                    runner.score_resources(0.)

    def test_denial_precedes_forward_device_or_outputs(self):
        with tempfile.TemporaryDirectory(dir=BASE) as folder:
            output = Path(folder)/"not_created"
            with patch.object(runner, "stage_inputs", side_effect=PermissionError), \
                 patch("src.atencion_armonica.structured_source_profile.gpu_lease") as lease:
                with self.assertRaises(PermissionError):
                    runner.forward_shard(output, "iid", 0, authorization={}, data={}, gpu_grant={})
                lease.assert_not_called()
                self.assertFalse(output.exists())
                with self.assertRaises(PermissionError):
                    runner.score_shard(output, "iid", 0, authorization={}, data={}, logits={})
                self.assertFalse(output.exists())

    def test_training_ports_reject_tests_before_loading_truth(self):
        with patch.object(runner, "load_supervision") as truth, patch.object(runner, "stage_inputs") as inputs:
            with self.assertRaises(PermissionError):
                runner.supervised_targets_shard("unused", "iid", 0, authorization={}, data={}, logits={}, scored={})
            with self.assertRaises(PermissionError):
                runner.training_corpus({}, "ood_polyphony", authorization={})
            truth.assert_not_called()
            inputs.assert_not_called()

    def test_disk_normalizer_preserves_exact_equations_and_multiple_passes(self):
        from experiments.atencion_armonica.test_learned_partition_cache import fixture
        rows = [fixture(), fixture()]
        with tempfile.TemporaryDirectory(dir=BASE) as folder:
            paths = [Path(folder)/f"row_{i}.npz" for i in range(2)]
            for path, row in zip(paths, rows):
                runner.save_rows(path, row)
            disk = runner.fit_normalizer(runner._DiskRows(paths), expected_count=2)
            memory = runner.fit_normalizer(rows, expected_count=2)
            for key in disk:
                np.testing.assert_array_equal(disk[key], memory[key])

    def test_forward_preserved_identity_and_order_without_model(self):
        # Read already observed q32 vectors; no new observations are drawn.
        path = ROOT/"data/atencion_armonica/shared_partial_cache_v1/train/observations.jsonl"
        with path.open() as f:
            observations = [json.loads(next(f)) for _ in range(512)]
        for i, obs in enumerate(observations):
            obs["scene_id"], obs["split_seed"] = i, runner.SPLITS["iid"][1]
        checkpoints = [{"seed": s, "checkpoint": {"path": f"mechanical_{s}", "sha256": "0"*64}}
                       for s in runner.SEEDS]
        common, authorization, data = {"checkpoints": checkpoints}, {"fixture": "auth"}, {"fixture": "data"}
        cache = SimpleNamespace(split="iid", shard=0, observations=observations)
        from src.atencion_armonica.partial_compatibility_inference import save_logits
        with tempfile.TemporaryDirectory(dir=BASE) as folder:
            root = Path(folder)
            matrices = [np.zeros((len(o["log_f"]), len(o["log_f"])), np.float32) for o in observations]
            rows = []
            for cp in checkpoints:
                file = f"seed_{cp['seed']}.npz"
                save_logits(root/file, matrices, observations)
                rows.append({"seed": cp["seed"], "checkpoint": cp["checkpoint"], "path": file, "count": 512})
            write_json(root/"forward.json", {"rows": rows, "runtime": {"fixture": True}})
            m = {"binding": {**runner._identity(common, authorization, data, "iid", 0), "gpu_grant": {}},
                 "artifacts_sha256": dict.fromkeys(["forward.json", *[r["path"] for r in rows]], "fixture")}
            # Generic bundle hashes are independently covered; this test isolates
            # semantic identity and the real lossless raw-logit decoder.
            with patch.object(runner, "_bundle", return_value=(root, m)):
                result = runner.ordered_forward({}, cache, common, authorization=authorization, data=data)
                self.assertEqual(set(result), set(runner.SEEDS))
                self.assertEqual(len(result[runner.SEEDS[0]]), 512)
                cache.observations = observations[::-1]
                with self.assertRaises(ValueError):
                    runner.ordered_forward({}, cache, common, authorization=authorization, data=data)
                cache.observations = observations
                m["binding"]["count"] = 256
                with self.assertRaises(ValueError):
                    runner.ordered_forward({}, cache, common, authorization=authorization, data=data)


if __name__ == "__main__":
    unittest.main()
