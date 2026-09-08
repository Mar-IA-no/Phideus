"""CPU-only mechanical wiring/replay tests; mocks never authorize real data."""
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from src.atencion_armonica import structured_source_gate as gate
from src.atencion_armonica import structured_source_runner as runner
from src.atencion_armonica import structured_source_profile as profile
from src.atencion_armonica.partial_compatibility_cache import encoded, feature_record
from src.atencion_armonica.source_coherence import SourceFitter
from src.atencion_armonica.structured_source_metrics import SEEDS
from src.atencion_armonica.structured_source_artifacts import write_json


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.common = {"mechanical_fixture": True, "checkpoints": [
            {"seed": s, "threshold": t} for s, t in zip(SEEDS, (.55, .65, .6))]}
        self.obs = {"scene_id": 0, "split_seed": 2026090790, "log_f": np.linspace(-.8, .8, 8, dtype=np.float32).tolist()}
        self.features = feature_record(self.obs)
        a = np.arange(64, dtype=np.float32).reshape(8, 8)/64
        self.raw = {s: a+a.T for s in SEEDS}
        self.truth = [0]*4+[1]*4

    def payload(self, labels):
        return runner.scene_payload(self.obs, self.features, self.raw, self.common["checkpoints"], labels, fitter=SourceFitter())

    def test_truth_changes_evaluation_not_pool_fit_costs_or_observable_cache(self):
        original = encoded(self.obs)
        a = self.payload(self.truth)
        b = self.payload(list(range(8)))
        self.assertEqual(original, encoded(self.obs))
        for seed in SEEDS:
            self.assertEqual(encoded(a[seed]["scored"]), encoded(b[seed]["scored"]))
        self.assertNotEqual(encoded(a[SEEDS[0]]["evaluation"]), encoded(b[SEEDS[0]]["evaluation"]))
        self.assertGreater(a[SEEDS[1]]["scored"]["fit_cache_hits"], 0)

    def test_forward_denial_precedes_device_functions(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(runner, "stage_inputs", side_effect=PermissionError), \
                patch.object(runner, "gpu_runtime") as runtime, patch.object(profile, "gpu_lease") as lease:
            with self.assertRaises(PermissionError):
                runner.forward_split(Path(tmp)/"denied", "iid", authorization={}, data={}, gpu_grant={})
            runtime.assert_not_called()
            lease.assert_not_called()
            self.assertFalse((Path(tmp)/"denied").exists())

    def test_calibration_analysis_and_replay_exact_bytes_without_forward(self):
        payloads = self.payload(self.truth)
        observations = [{**self.obs, "scene_id": i, "split_seed": 2026090780} for i in range(256)]
        cache = SimpleNamespace(observations=observations, records=[self.features]*256)
        raw = {s: [self.raw[s]]*256 for s in SEEDS}
        auth = {"common": self.common}
        with tempfile.TemporaryDirectory() as tmp, patch.object(gate, "ROOT", Path(tmp)), \
                patch.object(runner, "stage_inputs", return_value=(auth, cache, raw)), \
                patch.object(runner, "load_supervision", return_value=[{"source_ids": self.truth}]*256), \
                patch.object(runner, "scene_payload", return_value=payloads), \
                patch.object(runner, "checkpoint_forward") as forward:
            primary = runner.analyze_split(Path(tmp)/"primary", "calibration", authorization={}, data={}, logits={})
            replay = runner.analyze_split(Path(tmp)/"replay", "calibration", authorization={}, data={}, logits={}, replay_of=primary)
            _, a = gate.bundle_reference(primary, "calibration_analysis", self.common)
            _, b = gate.bundle_reference(replay, "calibration_analysis", self.common)
            self.assertEqual(a["artifacts_sha256"], b["artifacts_sha256"])
            self.assertEqual(len(a["artifacts_sha256"]), 770)  #768fullscene-seed states+grid+selection.
            self.assertEqual(len((Path(tmp)/"primary/gamma_grid.jsonl").read_text().splitlines()), 768)
            selection = json.loads((Path(tmp)/"primary/selection.json").read_text())
            self.assertEqual(selection["scene_count"], 256)
            self.assertEqual(selection["seeds"], list(SEEDS))
            self.assertEqual(json.loads((Path(tmp)/"replay/resources.json").read_text())["replay_status"], "EXACT_SCIENTIFIC_BYTES")
            forward.assert_not_called()

    def test_gpu_receipt_and_live_conflict_checked_without_torch(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(gate, "ROOT", Path(tmp)):
            (Path(tmp)/".agent-work/phideus-structured-reader-20260908").mkdir(parents=True)
            path = Path(tmp)/"grant.json"
            write_json(path, {"status": "AUTHORIZED", "project": "Phideus", "device": "NVIDIA GeForce RTX 3090",
                              "user_directive": "MECHANICAL FIXTURE; not a production grant"})
            ref = gate.reference(path)
            info = SimpleNamespace(stdout="NVIDIA GeForce RTX 3090, GPU-fixture, 100, 24576\n")
            with patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "0"}), \
                    patch.object(profile.subprocess, "run", side_effect=[info, SimpleNamespace(stdout="")]):
                with profile.gpu_lease(ref) as availability:
                    self.assertEqual(availability["compute_processes_before"], [])
            with patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "0"}), \
                    patch.object(profile.subprocess, "run", side_effect=[info, SimpleNamespace(stdout="123, 500\n")]):
                with self.assertRaisesRegex(RuntimeError, "existing compute"):
                    with profile.gpu_lease(ref):
                        self.fail("conflicting device must never be acquired")
            with patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": ""}), \
                    patch.object(profile.subprocess, "run", return_value=info):
                with self.assertRaisesRegex(RuntimeError, "visibility"):
                    with profile.gpu_lease(ref):
                        self.fail("CPU-only environment must not acquire CUDA")


if __name__ == "__main__":
    unittest.main()
