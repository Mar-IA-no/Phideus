"""CPU-only resource accounting fixtures, without an official preflight run."""
from pathlib import Path
import json
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from experiments.atencion_armonica import preflight_source_coherence as preflight


class PreflightTests(unittest.TestCase):
    def test_projection_uses_worst_size_and_full_roster(self):
        p = preflight.resource_projection({"3": .001, "4": .002, "8": .0001}, .003, .004, 2.)
        expected = 13440*.002+1152*.003+13440*.004+2.
        self.assertAlmostEqual(p["per_run_seconds"], expected)
        self.assertAlmostEqual(p["primary_plus_replay_seconds"], 2*expected)
        self.assertEqual(p["projected_fit_count"], 13440)
        self.assertEqual(p["status"], "ESTIMATE_NOT_GUARANTEED_BOUND")

    def test_existing_output_is_not_overwritten_or_executed(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(preflight, "SourceFitter") as fitter:
            with self.assertRaises(FileExistsError):
                preflight.run(Path(tmp))
            fitter.assert_not_called()
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_walltime_and_memory_guards(self):
        with patch.object(preflight.time, "monotonic", return_value=121.):
            with self.assertRaisesRegex(RuntimeError, "120s"):
                preflight.guard(0.)

    def test_projection_over_budget_is_incomplete(self):
        self.assertEqual(preflight.projection_status({"per_run_seconds": 301}), "INCOMPLETE")
        self.assertEqual(preflight.projection_status({"per_run_seconds": 300}), "READY_FOR_AUDITED_RUN")

    def test_supervisor_timeout_preserves_failure_and_rejects_existing_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/"run"
            with patch.object(preflight.subprocess, "run", side_effect=subprocess.TimeoutExpired("fixture", 125)) as child:
                with self.assertRaises(subprocess.TimeoutExpired):
                    preflight.supervised_run(output)
                marker = (output/"FAILURE.json").read_bytes()
                self.assertEqual(json.loads(marker)["status"], "INCOMPLETE")
                with self.assertRaises(FileExistsError):
                    preflight.supervised_run(output)
                self.assertEqual((output/"FAILURE.json").read_bytes(), marker)
                self.assertEqual(child.call_count, 1)

    def test_supervisor_preserves_child_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/"run"
            def child(*args, **kwargs):
                with (output/"FAILURE.json").open("xb") as handle:
                    handle.write(b'{"status":"INCOMPLETE","owner":"child"}')
                raise subprocess.TimeoutExpired("fixture", 125)
            with patch.object(preflight.subprocess, "run", side_effect=child):
                with self.assertRaises(subprocess.TimeoutExpired):
                    preflight.supervised_run(output)
            self.assertEqual(json.loads((output/"FAILURE.json").read_text())["owner"], "child")
        with patch.object(preflight.resource, "getrusage") as usage, patch.object(preflight.time, "monotonic", return_value=1.):
            usage.return_value.ru_maxrss = 1024**2
            with self.assertRaisesRegex(RuntimeError, "1GiB"):
                preflight.guard(0.)


if __name__ == "__main__":
    unittest.main()
