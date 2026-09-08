"""End-to-end synthetic fixtures; no closed-campaign artifact access."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from experiments.atencion_armonica import run_source_coherence as runner
from src.atencion_armonica.source_artifacts import ARMS, SEEDS, reader_key


class MechanicalArtifacts:
    def __init__(self):
        self.inputs = {}
        self.readers = [{"arm": a, "seed": s, "threshold": .5} for a in ARMS for s in SEEDS]
        self.readers.append({"arm": "analytic_support", "seed": None, "threshold": .5})

    def load_split(self, split):
        q = np.log(np.array([1., 2., 3.])).astype(np.float32)
        observations = [{"scene_id": i, "split_seed": 11, "log_f": q.tolist()} for i in range(32)]
        records = [{"triples": np.array([[0, 1, 2]]), "residual_cents": np.array([.3]),
                    "weights": np.array([.2]), "sham_weights": np.array([0.]), "sham_evaluable": np.array(False)} for _ in range(32)]
        truths = [{"source_ids": [0, 0, 0]} for _ in range(32)]
        partitions = {reader_key(r): [[[0, 1, 2]] for _ in range(32)] for r in self.readers}
        logits = {reader_key(r): [np.zeros((3, 3), np.float32) for _ in range(32)] for r in self.readers if r["seed"] is not None}
        return observations, records, truths, partitions, logits


class RunnerTests(unittest.TestCase):
    def test_complete_fixture_and_scientific_replay(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(runner, "ClosedArtifacts", MechanicalArtifacts), \
                patch.object(runner, "verify_preflight", return_value="MECHANICAL_FIXTURE_NOT_REAL_PREFLIGHT"), \
                patch.dict(os.environ, {k: "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}):
            manifests = []
            for name in ("primary", "replay"):
                output = Path(tmp)/name
                output.mkdir()
                runner.run(output)
                manifests.append(json.loads((output/"manifest.json").read_text()))
                summary = json.loads((output/"summary.json").read_text())
                self.assertEqual(summary["total_groups"], 96*14)
                self.assertEqual(summary["total_pressure_records"], 1152)
                self.assertEqual(summary["cache_misses"], 96)
                self.assertEqual(summary["cache_hits"], 96*13)
                self.assertFalse((output/"FAILURE.json").exists())
            self.assertEqual(manifests[0]["scientific_sha256"], manifests[1]["scientific_sha256"])

    def test_timeout_marker_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(runner.subprocess, "run", side_effect=subprocess.TimeoutExpired("fixture", 305)) as child:
            output = Path(tmp)/"run"
            with self.assertRaises(subprocess.TimeoutExpired):
                runner.supervised_run(output)
            self.assertEqual(json.loads((output/"FAILURE.json").read_text())["status"], "INCOMPLETE")
            with self.assertRaises(FileExistsError):
                runner.supervised_run(output)
            self.assertEqual(child.call_count, 1)

    def test_bad_thread_environment_fails_before_artifact_access(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"OMP_NUM_THREADS": "2"}), \
                patch.object(runner, "ClosedArtifacts") as artifacts:
            with self.assertRaises(ValueError):
                runner.run(Path(tmp))
            artifacts.assert_not_called()
            self.assertEqual(json.loads((Path(tmp)/"FAILURE.json").read_text())["status"], "INCOMPLETE")


if __name__ == "__main__":
    unittest.main()
