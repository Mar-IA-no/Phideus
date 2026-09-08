"""Calibration serialization and training-port denials using fixed tensors."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.atencion_armonica.test_learned_partition_data import BASE
from experiments.atencion_armonica.test_learned_partition_cache import fixture
from src.atencion_armonica import learned_partition_campaign as c
from src.atencion_armonica.learned_partition_core import fit_normalizer, model_inputs
from src.atencion_armonica.learned_partition_model import PartitionCostHead
from src.atencion_armonica.learned_partition_training import TrainingKernel
from src.atencion_armonica.learned_partition_snapshots import write_snapshot
from src.atencion_armonica import learned_partition_budget as budget
from src.atencion_armonica.structured_source_artifacts import write_json, mark_failure


class CampaignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        BASE.mkdir(parents=True, exist_ok=True)
        torch.set_num_threads(1)

    def assert_state_equal(self, left, right):
        if isinstance(left, torch.Tensor):
            self.assertTrue(torch.equal(left, right))
        elif isinstance(left, np.ndarray):
            np.testing.assert_array_equal(left, right)
        elif isinstance(left, dict):
            self.assertEqual(set(left), set(right))
            for key in left:
                self.assert_state_equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assert_state_equal(a, b)
        else:
            self.assertEqual(left, right)

    def failed_parent(self, root, name, binding, snapshot, *, recovery_status="SNAPSHOT_REQUIRED"):
        output = root/name
        args = {k: binding[k] for k in ("authorization", "train", "calibration", "normalizers", "normalized_train",
                "normalized_calibration", "arm", "checkpoint_seed", "reader_seed")}
        args["resume"] = None
        request = {"request_id": name, "operation": "train_cell", "output": output.relative_to(c.p.ROOT).as_posix(), "arguments": args}
        request_path = root/f"request_{name}.json"
        write_json(request_path, request)
        ref = c.p.reference(request_path)
        control = root/f"supervisor-{name}"
        control.mkdir()
        terminal = {k: None for k in budget.TERMINAL_KEYS}
        terminal.update(status="FAILED", request=ref, request_id=name, operation="train_cell", output=request["output"],
            worker_terminal_confirmed=True, seconds=7., error="mechanical interruption", budget={"fixture": True},
            observed_peak_rss_bytes=1000, observed_peak_gpu_bytes=0, recovery_status=recovery_status)
        write_json(control/"terminal.json", terminal)
        return {"request": ref, "terminal": c.p.reference(control/"terminal.json"), "snapshot": snapshot}

    def test_real_schedule_cooperative_and_hard_resume_match_continuous_with_prefix(self):
        row = fixture()
        inputs = model_inputs(row, fit_normalizer([row], expected_count=1), "shared_source")
        data = {"train": {"inputs": [inputs]*32, "targets": [np.array([[.1, .2], [.3, .1]], np.float32)]*32},
                "calibration": {"inputs": [inputs]*512, "candidates": [row.candidates]*512,
                                "ari": [np.array([.2, .8], np.float64)]*512}}
        binding = {"namespace": "MECHANICAL_TENSORS_ONLY", "count": 32, "device": "cpu", "arm": "shared_source",
            "checkpoint_seed": c.SEEDS[0], "reader_seed": c.READER_SEEDS[0],
            **{k: {} for k in ("authorization", "train", "calibration", "normalizers")},
            "normalized_train": [], "normalized_calibration": []}
        rng = torch.get_rng_state().clone()
        numpy_rng = np.random.get_state()
        def initial(root):
            (root/"snapshots").mkdir(parents=True)
            write_json(root/"ancestry.json", {"resume": None, "snapshots": [], "calibrations": []})
            torch.set_rng_state(rng)
            np.random.set_state(numpy_rng)
            kernel = TrainingKernel("shared_source", c.READER_SEEDS[0], binding=binding, count=32)
            ref = write_snapshot(root/"snapshots", "initial", kernel)
            write_json(root/"training_ready.json", {"snapshot": ref, "steps": 0, "binding": binding})
            return kernel, [ref], []
        with tempfile.TemporaryDirectory(dir=BASE) as folder:
            root = Path(folder)
            continuous, full_snaps, full_cal = initial(root/"continuous")
            c.run_schedule(root/"continuous", continuous, data, full_snaps, full_cal, should_stop=lambda: False)
            c.snapshot_chain(full_snaps, binding, complete=True)
            expected_state = continuous.state()
            for mode in ("cooperative", "hard"):
                parent = root/mode
                kernel, snapshots, calibrations = initial(parent)
                def stop():
                    if kernel.steps != 7:
                        return False
                    if mode == "hard":
                        raise RuntimeError("simulated abrupt loss without a new checkpoint")
                    return True
                with self.assertRaises((InterruptedError, RuntimeError)):
                    c.run_schedule(parent, kernel, data, snapshots, calibrations, should_stop=stop)
                mark_failure(parent, RuntimeError("mechanical interruption"))
                resume = self.failed_parent(root, mode, binding, snapshots[-1])
                with patch.object(budget, "STAGING", root):
                    state, prefix_snaps, prefix_cal = c.resume_state(resume, binding, data["calibration"])
                self.assertEqual(state["steps"], 7 if mode == "cooperative" else 5)
                self.assertEqual(prefix_cal, calibrations)
                self.assertEqual(len(prefix_cal), 1)
                preserved_prefix = c.p.verify_reference(prefix_cal[0]).read_bytes()
                suffix = root/f"resumed_{mode}"
                (suffix/"snapshots").mkdir(parents=True)
                restored = TrainingKernel("shared_source", c.READER_SEEDS[0], binding=binding, count=32)
                restored.restore(state)
                c.run_schedule(suffix, restored, data, prefix_snaps, prefix_cal, should_stop=lambda: False)
                self.assert_state_equal(restored.state(), expected_state)
                c.snapshot_chain(prefix_snaps, binding, complete=True)
                self.assertEqual(c.p.verify_reference(prefix_cal[0]).read_bytes(), preserved_prefix)
                self.assertEqual(len(prefix_cal), 10)
                for left, right in zip(full_cal, prefix_cal):
                    a = c.p.verify_reference(left).parent/"predictions.npz"
                    b = c.p.verify_reference(right).parent/"predictions.npz"
                    self.assertEqual(a.read_bytes(), b.read_bytes())
                wrong = {**resume, "snapshot": snapshots[0]}
                with patch.object(budget, "STAGING", root), self.assertRaises(ValueError):
                    c.resume_state(wrong, binding, data["calibration"])

    def test_pre_initialization_failure_has_explicit_no_snapshot_recovery(self):
        binding = {"arm": "shared_source", "checkpoint_seed": c.SEEDS[0], "reader_seed": c.READER_SEEDS[0],
            **{k: {} for k in ("authorization", "train", "calibration", "normalizers")},
            "normalized_train": [], "normalized_calibration": []}
        with tempfile.TemporaryDirectory(dir=BASE) as folder:
            root = Path(folder)
            resume = self.failed_parent(root, "absent_output", binding, None, recovery_status="NO_UPDATES_NO_SNAPSHOT")
            with patch.object(budget, "STAGING", root):
                self.assertEqual(c.resume_state(resume, binding, {}), (None, [], []))
                parent = root/"absent_output"
                parent.mkdir()
                write_json(parent/"ancestry.json", {"resume": None, "snapshots": [], "calibrations": []})
                write_json(parent/"training_ready.json", {"fixture": "already initialized"})
                with self.assertRaises(ValueError):
                    c.resume_state(resume, binding, {})

    def test_calibration_roundtrip_512_tensor_slots_without_observations(self):
        row = fixture()
        inputs = model_inputs(row, fit_normalizer([row], expected_count=1), "shared_source")
        model = PartitionCostHead("shared_source", c.READER_SEEDS[0])
        data = {"inputs": [inputs]*512, "candidates": [row.candidates]*512,
                "ari": [np.array([.2, .8], np.float64)]*512}
        binding = {"arm": "shared_source", "checkpoint_seed": c.SEEDS[0], "reader_seed": c.READER_SEEDS[0]}
        snapshot = {"fixture": "not_a_campaign_checkpoint"}
        with tempfile.TemporaryDirectory(dir=BASE) as folder:
            ref = c.save_calibration(folder, 5, snapshot, model, data, binding)
            result = c.calibration_record(ref, binding, data, snapshot=snapshot)
            self.assertEqual(result["epoch"], 5)
            self.assertEqual(len(result["ari"]), 512)
            with self.assertRaises(ValueError):
                c.calibration_record(ref, binding, data, snapshot={"different": True})
            with self.assertRaises(FileExistsError):
                c.save_calibration(folder, 5, snapshot, model, data, binding)

    def test_train_denial_before_data_model_or_output(self):
        with tempfile.TemporaryDirectory(dir=BASE) as folder:
            output = Path(folder)/"not_created"
            with patch.object(c, "verify_permit", side_effect=PermissionError), patch.object(c, "load_cell_data") as load:
                with self.assertRaises(PermissionError):
                    c.train_cell(output, authorization={}, train={}, calibration={}, normalizers={}, normalized_train=[],
                        normalized_calibration=[], arm="shared_source", checkpoint_seed=c.SEEDS[0], reader_seed=c.READER_SEEDS[0],
                        gpu_grant=None, resume=None, request_ref={}, permit=None)
                load.assert_not_called()
                self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
