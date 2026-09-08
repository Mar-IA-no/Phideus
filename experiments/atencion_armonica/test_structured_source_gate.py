"""Authorization fixtures are synthetic receipts, never production approvals."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from src.atencion_armonica import structured_source_gate as gate
from src.atencion_armonica.structured_source_artifacts import seal_bundle, write_json, write_npz


class GateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.patch = patch.object(gate, "ROOT", self.root)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.common = {"mechanical_fixture": True, "checkpoints": [
            {"seed": seed, "checkpoint": {"path": f"fake_{seed}.pt", "sha256": "f"*64}}
            for seed in gate.SEEDS]}

    def json_ref(self, name, value):
        write_json(self.root/name, value)
        return gate.reference(self.root/name)

    def profile(self, name, role, report):
        root = self.root/name
        root.mkdir()
        write_json(root/"report.json", report)
        seal_bundle(root, role=role, binding={"common": self.common}, resources={"fixture": True})
        return gate.reference(root/"manifest.json")

    def calibration(self):
        raw = self.json_ref("raw_mechanical_audit.json", {"fixture": "not a scientific audit"})
        audit = self.json_ref("audit.json", {"status": "PASS", "common": self.common, "reports": [raw], "freeze_sha256": None})
        cpu = self.profile("cpu", "cpu_preflight", {"status": "READY", "seconds": 1, "peak_rss_bytes": 100000,
                                                   "projected_cpu_phase_seconds": 10, "projected_peak_rss_bytes": 200000})
        gpu = self.profile("gpu", "forward_profile", {"status": "READY", "device": "NVIDIA GeForce RTX 3090",
                                                       "projected_forward_seconds": 5, "peak_reserved_bytes": 100000})
        return {"status": "CALIBRATION_READY", "common": self.common, "implementation_audit": audit,
                "cpu_preflight": cpu, "gpu_profile": gpu, "historical_observations": []}

    def test_calibration_requires_current_audit_resources_and_entire_corpus(self):
        record = self.calibration()
        with patch.object(gate, "historical_observations", return_value=[]):
            gate._verify_calibration(record, self.common)
            for field, value in (("status", "TEST_READY"), ("common", {}), ("historical_observations", [{}]),
                                 ("implementation_audit", record["gpu_profile"]), ("gpu_profile", record["cpu_preflight"])):
                with self.assertRaises((ValueError, KeyError)):
                    gate._verify_calibration({**record, field: value}, self.common)
        with patch.object(gate, "historical_observations", return_value=[{"required": True}]):
            with self.assertRaisesRegex(ValueError, "incomplete"):
                gate._verify_calibration(record, self.common)

    def test_profile_over_budget_or_wrong_device_cannot_authorize_calibration(self):
        record = self.calibration()
        for i, change in enumerate(({"device": "CPU"}, {"projected_forward_seconds": 601},
                                    {"peak_reserved_bytes": 2*1024**3}, {"projected_forward_seconds": -1})):
            ref = self.profile(f"bad_gpu_{i}", "forward_profile", {"status": "READY", "device": "NVIDIA GeForce RTX 3090",
                                                                  "projected_forward_seconds": 5, "peak_reserved_bytes": 100000, **change})
            with self.assertRaises(ValueError):
                gate._verify_calibration({**record, "gpu_profile": ref}, self.common)

    def test_audit_is_exact_to_sources_and_freeze_and_failure_revokes(self):
        report = self.json_ref("raw.json", {"fixture": True})
        receipt = self.json_ref("receipt.json", {"status": "PASS", "common": self.common, "reports": [report], "freeze_sha256": "a"*64})
        gate.verify_audit(receipt, self.common, freeze_sha="a"*64)
        for common, digest in (({}, "a"*64), (self.common, "b"*64), (self.common, None)):
            with self.assertRaises(ValueError):
                gate.verify_audit(receipt, common, freeze_sha=digest)
        write_json(self.root/"FAILURE.json", {"fixture": True})
        with self.assertRaisesRegex(ValueError, "incomplete"):
            gate.verify_reference(report)

    def test_test_access_requires_distinct_frozen_and_independently_audited_chain(self):
        cal = self.json_ref("calibration.json", {"status": "CALIBRATION_READY", "common": self.common})
        with patch.object(gate, "common_binding", return_value=self.common):
            with self.assertRaises(PermissionError):
                gate.verify_authorization(cal, "iid")
            bad = self.json_ref("test_bad.json", {"status": "TEST_READY", "common": self.common,
                                                  "freeze": cal, "freeze_audit": {}})
            with self.assertRaises(ValueError):
                gate.verify_authorization(bad, "iid")
            frozen = self.json_ref("frozen.json", {"status": "FROZEN_BEFORE_TEST", "common": self.common})
            no_audit = self.json_ref("test_no_audit.json", {"status": "TEST_READY", "common": self.common,
                                                            "freeze": frozen, "freeze_audit": cal})
            with self.assertRaisesRegex(ValueError, "audit"):
                gate.verify_authorization(no_audit, "iid")

    def test_individual_failure_revokes_only_its_own_receipt(self):
        failed = self.json_ref("failed_receipt.json", {"fixture": True})
        retained = self.json_ref("retained_receipt.json", {"fixture": True})
        self.json_ref("failed_receipt.json.FAILURE.json", {"status": "INCOMPLETE"})
        with self.assertRaisesRegex(ValueError, "incomplete"):
            gate.verify_reference(failed)
        self.assertEqual(gate.verify_reference(retained), self.root/"retained_receipt.json")

    def test_raw_forward_full_roster_order_fingerprints_and_checkpoint(self):
        observations = [{"scene_id": i, "split_seed": 2026090790, "log_f": np.array([i/256, 1, 2], np.float32).tolist()}
                        for i in range(256)]
        fingerprints = np.asarray([hashlib.sha256(np.asarray(o["log_f"], "<f4").tobytes()).hexdigest()
                                   for o in observations], dtype="U64")
        rows = []
        for c in self.common["checkpoints"]:
            seed = c["seed"]
            write_npz(self.root/f"seed_{seed}.npz", logits=np.zeros(256*9, np.float32),
                      sizes=np.full(256, 3, np.int64), offsets=np.arange(257, dtype=np.int64)*9,
                      scene_ids=np.arange(256, dtype=np.int64), split_seeds=np.full(256, 2026090790, np.int64),
                      observation_fingerprints=fingerprints)
            rows.append({"seed": seed, "checkpoint": c["checkpoint"], "path": f"seed_{seed}.npz", "count": 256})
        write_json(self.root/"forward.json", {"rows": rows})
        manifest = seal_bundle(self.root, role="mechanical_forward", binding={}, resources={})
        result = gate.ordered_forward(self.root, manifest, observations, self.common)
        self.assertEqual(set(result), set(gate.SEEDS))
        with self.assertRaises(ValueError):
            gate.ordered_forward(self.root, manifest, observations[::-1], self.common)
        changed = copy.deepcopy(self.common)
        changed["checkpoints"][0]["checkpoint"]["sha256"] = "0"*64
        with self.assertRaises(ValueError):
            gate.ordered_forward(self.root, manifest, observations, changed)
        with self.assertRaises(ValueError):
            gate.ordered_forward(self.root, {"artifacts_sha256": {}}, observations, self.common)


if __name__ == "__main__":
    unittest.main()
