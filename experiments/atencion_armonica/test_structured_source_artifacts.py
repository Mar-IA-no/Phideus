import tempfile
from pathlib import Path
import unittest

import numpy as np

from src.atencion_armonica.partial_compatibility_cache import sha_file
from src.atencion_armonica.structured_source_artifacts import (
    mark_failure, safe_member, seal_bundle, verify_bundle, write_json, write_npz)


class ArtifactTests(unittest.TestCase):
    def test_complete_inventory_pinned_identity_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_json(root/"config.json", {"fixture": True})
            write_npz(root/"raw.npz", values=np.arange(5, dtype=np.float32))
            m = seal_bundle(root, role="mechanical_fixture", binding={"source": "fixture"}, resources={"seconds": 1.})
            digest = sha_file(root/"manifest.json")
            self.assertEqual(m, verify_bundle(root, digest, role="mechanical_fixture"))
            with self.assertRaises(ValueError):
                verify_bundle(root, "0"*64, role="mechanical_fixture")
            with self.assertRaises(ValueError):
                verify_bundle(root, digest, role="test_data")
            with self.assertRaises(ValueError):
                seal_bundle(root, role="mechanical_fixture", binding={}, resources={})
            with self.assertRaises(FileExistsError):
                write_json(root/"config.json", {"overwritten": True})
            write_json(root/"extra.json", {})
            with self.assertRaises(ValueError):
                verify_bundle(root, digest, role="mechanical_fixture")

    def test_failure_marker_revokes_completeness_and_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_json(root/"raw.json", {})
            seal_bundle(root, role="mechanical_fixture", binding={}, resources={})
            digest = sha_file(root/"manifest.json")
            mark_failure(root, ValueError("first"))
            failure = sha_file(root/"FAILURE.json")
            mark_failure(root, ValueError("second"))
            self.assertEqual(failure, sha_file(root/"FAILURE.json"))
            with self.assertRaises(ValueError):
                verify_bundle(root, digest, role="mechanical_fixture")

    def test_unsealed_failure_and_unsafe_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mark_failure(root, RuntimeError("unfinished"))
            with self.assertRaises(ValueError):
                seal_bundle(root, role="fixture", binding={}, resources={})
            for name in ("../outside", "/absolute", "./ambiguous", "a/../b", "a//b", "a\\b", ""):
                with self.assertRaises(ValueError):
                    safe_member(root, name)
            with self.assertRaises(ValueError):
                write_npz(root/"unsafe.npz", value=np.array([{}], dtype=object))


if __name__ == "__main__":
    unittest.main()
