"""Owned-handle lifecycle tests; no campaign worker, draws, CUDA or process kill."""
import json
import os
import signal
import subprocess
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.atencion_armonica.test_learned_partition_data import BASE
from src.atencion_armonica import learned_partition_supervisor as s
from src.atencion_armonica import learned_partition_gate as gate
from src.atencion_armonica.structured_source_artifacts import write_json


class SupervisorTests(unittest.TestCase):
    def test_closed_request_and_no_overwrite(self):
        request = {"request_id": "mechanical", "operation": "geometry_profile", "arguments": {"audit": {}},
                   "output": "data/atencion_armonica/learned_partition_reader_v1/not_launched", "common": {}}
        ref = {"path": "data/atencion_armonica/learned_partition_reader_v1/requests/mechanical.json", "sha256": "0"*64}
        with patch.object(s.p, "read_reference", return_value=request), patch.object(gate, "common_binding", return_value={}):
            self.assertEqual(s.validate_request(ref)[0], request)
            request["arguments"]["arbitrary"] = True
            with self.assertRaises(ValueError):
                s.validate_request(ref)
            del request["arguments"]["arbitrary"]
            request["output"] = "../outside"
            with self.assertRaises(ValueError):
                s.validate_request(ref)

    def test_parent_death_guard_is_armed_in_own_short_cpu_child(self):
        code = ("import ctypes,os; from src.atencion_armonica.learned_partition_supervisor import arm_parent_death; "
                "arm_parent_death(os.getppid()); value=ctypes.c_int(); "
                "assert ctypes.CDLL(None).prctl(2,ctypes.byref(value),0,0,0)==0; print(value.value)")
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, timeout=10,
                                env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
        self.assertEqual(int(result.stdout.strip()), signal.SIGKILL)
        with patch.object(s.os, "getppid", return_value=1):
            with self.assertRaises(PermissionError):
                s.arm_parent_death(123456)

    def test_limit_table_and_unknown_operation(self):
        self.assertEqual(s.limits("geometry_profile"), (120., 1024**3, 0))
        self.assertEqual(s.limits("training_gpu_profile"), (120., 4*1024**3, 2*1024**3))
        self.assertEqual(s.limits("forward"), (1200., 4*1024**3, 2*1024**3))
        self.assertEqual(s.limits("training_cpu_profile"), (120., 4*1024**3, 0))
        self.assertEqual(s.limits("train_cell"), (1200., 4*1024**3, 0))
        self.assertEqual(s.limits("test_inference"), (1200., 4*1024**3, 0))
        self.assertEqual(s.limits("score"), (2400., 2*1024**3, 0))
        special = {"geometry_profile", "training_cpu_profile", "training_gpu_profile",
                   "forward", "train_cell", "test_inference", "score"}
        for operation in set(s.ARGUMENTS)-special:
            self.assertEqual(s.limits(operation), (1200., 2*1024**3, 0))
        with self.assertRaises(ValueError):
            s.limits("unbounded")

    def test_terminal_receipt_preserves_failed_output_and_rejects_high_water(self):
        BASE.mkdir(parents=True, exist_ok=True)
        for variant in ("ok", "exit", "rss", "wrong_output"):
            with self.subTest(variant=variant), tempfile.TemporaryDirectory(dir=BASE) as folder:
                staging = Path(folder)
                output = staging/"output.json"
                request = {"request_id": "mechanical", "operation": "authorize_data", "output": str(output), "common": {}}
                created = []
                class Worker:
                    def __init__(self, *args, **kwargs):
                        self.pid, self.returncode = 99999999, 1 if variant == "exit" else 0
                        self.pipe = os.dup(kwargs["pass_fds"][0])
                        write_json(output, {"mechanical": True})
                        kwargs["stdout"].write(json.dumps({"result": {"path": str(output), "sha256": "fake"},
                            "peak_rss_bytes": 3*1024**3 if variant == "rss" else 1000}).encode())
                        created.append(self)
                    def poll(self):
                        return self.returncode
                    def wait(self, timeout=None):
                        if self.pipe is not None:
                            os.close(self.pipe)
                            self.pipe = None
                        return self.returncode
                with patch.object(s, "STAGING", staging), patch.object(s, "validate_request", return_value=(request, output)), \
                     patch.object(s.subprocess, "Popen", Worker), patch.object(s, "process_identity", return_value={"pid": 99999999}), \
                     patch.object(s.p, "verify_reference", return_value=staging/"other" if variant == "wrong_output" else output), \
                     patch.object(s.p, "reference", side_effect=lambda path: {"path": str(path), "sha256": "mechanical"}), \
                     patch.object(s.p, "read_reference", return_value=request), patch.object(gate, "common_binding", return_value={}):
                    if variant == "ok":
                        s.supervised({})
                    else:
                        with self.assertRaises(RuntimeError):
                            s.supervised({})
                receipt_path, = staging.glob("supervisor-*/terminal.json")
                receipt = json.loads(receipt_path.read_text())
                self.assertTrue(receipt["worker_terminal_confirmed"])
                self.assertEqual(receipt["status"], "COMPLETE" if variant == "ok" else "FAILED")
                self.assertTrue(output.exists())
                self.assertEqual(output.with_name("output.json.FAILURE.json").exists(), variant != "ok")
                self.assertIsNone(created[0].pipe)

    def test_termination_checks_owned_process_group_before_signal(self):
        class Worker:
            pid = 123456
            def poll(self):
                return None
        with patch.object(s.os, "getpgid", return_value=123), patch.object(s.os, "killpg") as kill:
            with self.assertRaises(RuntimeError):
                s._terminate(Worker())
            kill.assert_not_called()


if __name__ == "__main__":
    unittest.main()
