"""Launcher failure gates only, using temporary CPU fixture manifests."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.atencion_armonica import train_partial_compatibility_campaign as campaign


class CampaignGateTests(unittest.TestCase):
    def test_authorization_required_before_output_or_child(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(campaign.subprocess, "Popen") as spawn:
            root = Path(tmp)
            with self.assertRaises(ValueError):
                campaign.launch(root/"out", root/"missing")
            self.assertFalse((root/"out").exists())
            spawn.assert_not_called()

    def test_request_detects_source_receipt_and_manifest_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            receipt = root/"receipt"
            receipt.write_bytes(b"fixture")
            request = {"source_sha256": {"fixture": "source"}, "authorization_path": str(receipt),
                       "authorization_sha256": campaign.sha_file(receipt), "cache_manifests": {}}
            (root/"manifest.json").write_bytes(b"fixture profile")
            compat = {"path": "receipt", "sha256": campaign.sha_file(receipt),
                      "parity_test": "receipt", "parity_test_sha256": campaign.sha_file(receipt)}
            request.update(profile_manifest_sha256=campaign.sha_file(root/"manifest.json"), profile_compatibility=compat)
            with patch.object(campaign, "sources", return_value={"fixture": "source"}), \
                 patch.object(campaign, "PROFILE_ROOT", root), patch.object(campaign, "ROOT", root), \
                 patch.object(campaign, "COMPATIBILITY", compat), \
                 patch.object(campaign, "APPROVED_PROFILE_SHA256", request["profile_manifest_sha256"]):
                campaign.verify_request(request)
                receipt.write_bytes(b"changed")
                with self.assertRaisesRegex(RuntimeError, "receipt changed"):
                    campaign.verify_request(request)
            with patch.object(campaign, "sources", return_value={}):
                with self.assertRaisesRegex(RuntimeError, "source changed"):
                    campaign.verify_request(request)
            receipt.write_bytes(b"fixture")
            (root/"train").mkdir()
            (root/"train"/"manifest.json").write_bytes(b"fixture manifest")
            request["cache_manifests"] = {"train": "wrong digest"}
            with patch.object(campaign, "sources", return_value={"fixture": "source"}), \
                 patch.object(campaign, "CACHE_ROOT", root):
                with self.assertRaisesRegex(RuntimeError, "manifest changed"):
                    campaign.verify_request(request)

    def test_bad_profile_never_launches_or_creates_output(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(campaign.subprocess, "Popen") as spawn:
            root = Path(tmp)
            receipt = root/"receipt"
            receipt.write_bytes(b"fixture")
            with patch.object(campaign, "verified_profile", side_effect=ValueError("bad profile")):
                with self.assertRaisesRegex(ValueError, "bad profile"):
                    campaign.launch(root/"out", receipt)
            self.assertFalse((root/"out").exists())
            spawn.assert_not_called()

    def test_profile_checks_artifact_hash_before_trusting_projection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/"profile.json").write_bytes(b"corrupt fixture")
            campaign.write_new(root/"manifest.json", {"status": "COMPLETE",
                               "scope": "GPU_profile_not_full_experiment",
                               "artifacts_sha256": {"profile.json": "wrong"}})
            with patch.object(campaign, "PROFILE_ROOT", root), \
                 patch.object(campaign, "APPROVED_PROFILE_SHA256", campaign.sha_file(root/"manifest.json")):
                with self.assertRaisesRegex(ValueError, "hash mismatch"):
                    campaign.verified_profile()

    def test_postchild_failure_settles_incomplete_not_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            receipt = root/"receipt"
            receipt.write_bytes(b"fixture")
            (root/"manifest.json").write_bytes(b"profile fixture")
            for split in ("development", "train", "validation"):
                (root/split).mkdir()
                (root/split/"manifest.json").write_bytes(b"cache fixture")
            with patch.object(campaign, "verified_profile", return_value={"seconds": 3}), \
                 patch.object(campaign, "PROFILE_ROOT", root), patch.object(campaign, "CACHE_ROOT", root), \
                 patch.object(campaign, "BUDGET_ROOT", root), patch.object(campaign, "sources", return_value={}), \
                 patch.object(campaign, "verify_request", side_effect=RuntimeError("postchild drift")), \
                 patch.object(campaign.subprocess, "Popen") as spawn:
                spawn.return_value.wait.return_value = 0
                spawn.return_value.poll.return_value = 0
                with self.assertRaisesRegex(RuntimeError, "postchild drift"):
                    campaign.launch(root/"out", receipt)
            budget = json.loads((root/"GPU_BUDGET.json").read_text())
            self.assertEqual(budget["attempts"][0]["status"], "INCOMPLETE")
            self.assertIsNone(budget["reservation"])
            self.assertTrue((root/"out"/"FAILURE.json").exists())
            self.assertFalse((root/"out"/"manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
