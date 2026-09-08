"""Supervisor failure mechanics without a real worker, corpus or CUDA."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.atencion_armonica import run_structured_source as cli
from src.atencion_armonica import structured_source_gate as gate
from src.atencion_armonica.structured_source_artifacts import write_json


class SupervisorTests(unittest.TestCase):
    def fake_terminal_worker(self, result, *, code=0, on_start=None):
        class FakeWorker:
            def __init__(self, *args, **kwargs):
                self.pid, self.returncode = 99999999, code
                self.pipe = os.dup(kwargs["pass_fds"][0])
                if on_start is not None:
                    on_start()
                kwargs["stdout"].write(json.dumps(result() if callable(result) else result).encode())

            def poll(self):
                return self.returncode

            def wait(self, timeout=None):
                os.close(self.pipe)
                return self.returncode
        return FakeWorker

    def test_successful_worker_cannot_return_an_unrelated_existing_artifact(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(cli, "ROOT", Path(tmp)), patch.object(gate, "ROOT", Path(tmp)):
            root = Path(tmp)
            (root/".agent-work/phideus-structured-reader-20260908").mkdir(parents=True)
            path = root/"request.json"
            write_json(path, {"operation": "authorize_test", "output": "data/atencion_armonica/expected.json", "arguments": {}})
            write_json(root/"unrelated.json", {"fixture": True})
            worker = self.fake_terminal_worker(gate.reference(root/"unrelated.json"))
            with patch.object(cli.subprocess, "Popen", worker):
                with self.assertRaises(ValueError):
                    cli.supervised(gate.reference(path))

    def test_failed_worker_revokes_receipt_written_before_its_failure(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(cli, "ROOT", Path(tmp)), patch.object(gate, "ROOT", Path(tmp)):
            root = Path(tmp)
            (root/".agent-work/phideus-structured-reader-20260908").mkdir(parents=True)
            (root/"data/atencion_armonica").mkdir(parents=True)
            output = root/"data/atencion_armonica/receipt.json"
            path = root/"request.json"
            write_json(path, {"operation": "authorize_test", "output": output.relative_to(root).as_posix(), "arguments": {}})
            worker = self.fake_terminal_worker({}, code=1, on_start=lambda: write_json(output, {"fixture": True}))
            with patch.object(cli.subprocess, "Popen", worker):
                with self.assertRaises(RuntimeError):
                    cli.supervised(gate.reference(path))
            self.assertTrue(output.is_file())  # Retain evidence, but revoke its authority.
            with self.assertRaises(ValueError):
                gate.verify_reference(gate.reference(output))

    def test_exit_observed_after_deadline_cannot_be_reported_complete(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(cli, "ROOT", Path(tmp)), patch.object(gate, "ROOT", Path(tmp)):
            root = Path(tmp)
            (root/".agent-work/phideus-structured-reader-20260908").mkdir(parents=True)
            (root/"data/atencion_armonica").mkdir(parents=True)
            output = root/"data/atencion_armonica/receipt.json"
            path = root/"request.json"
            write_json(path, {"operation": "authorize_test", "output": output.relative_to(root).as_posix(), "arguments": {}})
            worker = self.fake_terminal_worker(lambda: gate.reference(output),
                                               on_start=lambda: write_json(output, {"fixture": True}))
            with patch.object(cli.subprocess, "Popen", worker), patch.object(cli.time, "monotonic", side_effect=[0., 1201.]):
                with self.assertRaises(TimeoutError):
                    cli.supervised(gate.reference(path))

    def test_request_path_schema_and_existing_output_rejected(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(cli, "ROOT", Path(tmp)), patch.object(gate, "ROOT", Path(tmp)):
            for i, request in enumerate((
                    {"operation": "train", "output": "data/atencion_armonica/x", "arguments": {}},
                    {"operation": "prepare", "output": "../outside", "arguments": {}},
                    {"operation": "prepare", "output": "arbitrary/x", "arguments": {}})):
                path = Path(tmp)/f"bad_{i}.json"
                write_json(path, request)
                with self.assertRaises(ValueError):
                    cli.validate_request(gate.reference(path))
            output = Path(tmp)/"data/atencion_armonica/existing"
            output.mkdir(parents=True)
            path = Path(tmp)/"existing.json"
            write_json(path, {"operation": "prepare", "output": "data/atencion_armonica/existing", "arguments": {}})
            with self.assertRaises(FileExistsError):
                cli.validate_request(gate.reference(path))

    def test_timeout_terminates_only_own_worker_and_preserves_failure(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(cli, "ROOT", Path(tmp)), patch.object(gate, "ROOT", Path(tmp)):
            root = Path(tmp)
            staging = root/".agent-work/phideus-structured-reader-20260908"
            staging.mkdir(parents=True)
            output = root/"data/atencion_armonica/mechanical_failure"
            path = root/"request.json"
            write_json(path, {"operation": "cpu_preflight", "output": output.relative_to(root).as_posix(), "arguments": {}})
            created = []

            class FakeWorker:
                def __init__(self, *args, **kwargs):
                    self.pid, self.returncode, self.terminated = 99999999, None, False
                    self.pipe = os.dup(kwargs["pass_fds"][0])
                    output.mkdir(parents=True)
                    write_json(output/"partial.json", {"fixture": True})
                    created.append(self)

                def poll(self):
                    return self.returncode

                def terminate(self):
                    self.terminated = True
                    self.returncode = -15

                def wait(self, timeout=None):
                    os.close(self.pipe)
                    return self.returncode

            with patch.object(cli.subprocess, "Popen", FakeWorker), patch.object(cli.time, "monotonic", side_effect=[0., 121.]):
                with self.assertRaises(TimeoutError):
                    cli.supervised(gate.reference(path))
            self.assertTrue(created[0].terminated)
            self.assertTrue((output/"partial.json").is_file())
            self.assertTrue((output/"FAILURE.json").is_file())
            self.assertFalse((output/"manifest.json").exists())
            controls = list(staging.iterdir())
            self.assertEqual(len(controls), 1)
            self.assertTrue((controls[0]/"FAILURE.json").is_file())
            self.assertEqual(json.loads((controls[0]/"request.json").read_text())["wall_limit_seconds"], 120.)


if __name__ == "__main__":
    unittest.main()
