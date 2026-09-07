"""Budget/lock/crash accounting without importing Torch or querying GPU."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.atencion_armonica.partial_compatibility_budget import CampaignBudget, LIMIT_SECONDS, verify_inherited_lease


class BudgetTests(unittest.TestCase):
    def test_exclusive_lock_and_cumulative_charge(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = CampaignBudget(root, profile_manifest_sha256="fixture", profile_seconds=3)
            with self.assertRaises(BlockingIOError):
                CampaignBudget(root, profile_manifest_sha256="fixture", profile_seconds=3)
            with patch("time.monotonic", side_effect=[100., 110.]):
                self.assertEqual(first.reserve(root/"run1", "fixture request"), LIMIT_SECONDS-3)
                self.assertEqual(first.execution_deadline(), 100.+LIMIT_SECONDS-3-15)
                first.settle("COMPLETE")
            first.close()
            second = CampaignBudget(root, profile_manifest_sha256="fixture", profile_seconds=3)
            self.assertEqual(second.remaining, LIMIT_SECONDS-13)
            second.close()
            stored = json.loads((root/"GPU_BUDGET.json").read_text())
            self.assertEqual(len(stored["attempts"]), 1)

    def test_parent_death_does_not_reset_reservation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = CampaignBudget(root, profile_manifest_sha256="fixture", profile_seconds=3)
            first.reserve(root/"run1", "fixture request")
            first.close()
            with self.assertRaisesRegex(RuntimeError, "unsettled"):
                CampaignBudget(root, profile_manifest_sha256="fixture", profile_seconds=3)
            self.assertIsNotNone(json.loads((root/"GPU_BUDGET.json").read_text())["reservation"])

    def test_changed_profile_identity_is_not_a_new_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = CampaignBudget(root, profile_manifest_sha256="fixture", profile_seconds=3)
            first.close()
            with self.assertRaises(ValueError):
                CampaignBudget(root, profile_manifest_sha256="different", profile_seconds=3)

    def test_worker_lease_binds_descriptor_parent_request_and_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            budget = CampaignBudget(root, profile_manifest_sha256="fixture", profile_seconds=3)
            output = root/"run1"
            budget.reserve(output, "fixture request")
            deadline = budget.execution_deadline()
            with patch("os.getppid", return_value=os.getpid()):
                verify_inherited_lease(root, output, "fixture request", budget.lock.fileno(), deadline)
                for changed_output, digest in ((root/"other", "fixture request"), (output, "wrong")):
                    with self.assertRaises(ValueError):
                        verify_inherited_lease(root, changed_output, digest, budget.lock.fileno(), deadline)
                with self.assertRaises(ValueError):
                    verify_inherited_lease(root, output, "fixture request", None, deadline)
                with (root/"unrelated").open("w") as handle:
                    with self.assertRaisesRegex(ValueError, "wrong lock"):
                        verify_inherited_lease(root, output, "fixture request", handle.fileno(), deadline)
            with patch("os.getppid", return_value=-1):
                with self.assertRaises(ValueError):
                    verify_inherited_lease(root, output, "fixture request", budget.lock.fileno(), deadline)
            budget.settle("FIXTURE")
            budget.close()

    def test_overrun_is_recorded_not_hidden(self):
        with tempfile.TemporaryDirectory() as tmp:
            budget = CampaignBudget(Path(tmp), profile_manifest_sha256="fixture", profile_seconds=3)
            with patch("time.monotonic", side_effect=[100., 100.+LIMIT_SECONDS]):
                budget.reserve(Path(tmp)/"run1", "fixture request")
                budget.settle("INCOMPLETE")
            self.assertEqual(budget.state["overrun_seconds"], 3)
            self.assertEqual(budget.remaining, 0)
            budget.close()


if __name__ == "__main__":
    unittest.main()
