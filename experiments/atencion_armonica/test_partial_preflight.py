"""Preflight harness fixtures; actual development generation remains mocked."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from src.atencion_armonica.partial_compatibility_evaluation import read_partition
from src.atencion_armonica.shared_partial_data import mechanical_fixture

SCRIPT = Path(__file__).with_name("preflight_partial_compatibility.py")
spec = importlib.util.spec_from_file_location("preflight_fixture_runner", SCRIPT)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class PreflightTests(unittest.TestCase):
    def test_reader_no_cardinality_and_permutation(self):
        p = np.array([[1,.9,.1,.2],[.9,1,.2,.1],[.1,.2,1,.8],[.2,.1,.8,1]])
        expected = ((0,1),(2,3))
        self.assertEqual(read_partition(p,.5), expected)
        perm = np.array([2,0,3,1])
        result = read_partition(p[np.ix_(perm,perm)],.5)
        restored = tuple(sorted(tuple(sorted(perm[list(block)])) for block in result))
        self.assertEqual(restored, expected)
        self.assertEqual(len(read_partition(p,0)), 4)
        self.assertEqual(len(read_partition(p,1)), 1)
        with self.assertRaises(ValueError):
            read_partition(np.array([[1, np.nan], [0, 1]]), .5)

    def test_geometry_stage_closed_generation_and_artifacts(self):
        base = runner.ROOT/".agent-work/phideus-geometric-rebase-20260907/preflight-tests"
        base.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as tmp:
            output = Path(tmp)
            calls = []
            def fake_scene(split, scene_id):
                calls.append((split, scene_id))
                obs, truth = mechanical_fixture(3)
                obs["scene_id"] = truth["scene_id"] = scene_id
                return obs, truth
            with patch.object(runner, "generate_scene", side_effect=fake_scene):
                runner.geometry_stage(output)
            self.assertEqual(calls, [("development", i) for i in range(64)])
            report = json.loads((output/"geometry_report.json").read_bytes())
            self.assertEqual(report["status"], "PASS")
            self.assertFalse(report["torch_imported"])
            self.assertEqual(len(report["rows"]), 67)
            self.assertEqual(report["rows"][-2]["n"], 32)
            with np.load(output/"features.npz", allow_pickle=False) as cache:
                self.assertEqual(cache["max_eval/triples"].shape, (4960,3))
                self.assertEqual(cache["development_000/tokens"].shape, (24,2))
            observations = [json.loads(x) for x in (output/"observations.jsonl").read_bytes().splitlines()]
            self.assertEqual(len(observations),64)
            self.assertTrue(all(set(o) == {"log_f","scene_id","split_seed"} for o in observations))
            with self.assertRaises(FileExistsError):
                runner.write_new(output/"geometry_report.json", b"overwrite")

    def test_campaign_children_sequential_bounded_and_receipt(self):
        base = runner.ROOT/".agent-work/phideus-geometric-rebase-20260907/preflight-tests"
        base.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as tmp:
            output = Path(tmp)/"campaign"
            calls = []
            def fake_run(command, *, check, timeout):
                stage = command[command.index("--stage")+1]
                self.assertTrue(check)
                calls.append((stage, timeout))
                runner.write_new(output/f"{stage}_fixture.json", b"{}\n")
            with patch.object(runner.subprocess, "run", side_effect=fake_run):
                runner.campaign(output)
            self.assertEqual(calls, [("geometry",120),("gradients",30)])
            receipt = json.loads((output/"manifest.json").read_bytes())
            self.assertEqual(receipt["status"],"PASS")
            self.assertEqual(set(receipt["artifacts_sha256"]),
                             {"source_freeze.json","geometry_fixture.json","gradients_fixture.json"})
            with self.assertRaises(FileExistsError):
                runner.campaign(output)


if __name__ == "__main__":
    unittest.main()
