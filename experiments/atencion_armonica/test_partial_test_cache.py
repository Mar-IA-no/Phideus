"""All test draws are mocked mechanical fixtures; prospective splits stay unopened."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from src.atencion_armonica import partial_compatibility_test_cache as cache_module
from src.atencion_armonica.partial_compatibility_cache import feature_record, load_supervision
from src.atencion_armonica.shared_partial_data import mechanical_fixture


class ClosedTestCacheTests(unittest.TestCase):
    @staticmethod
    def fixture_scene(split, scene_id, *, allow_test):
        assert allow_test is True
        obs, truth = mechanical_fixture(3)
        obs["scene_id"] = truth["scene_id"] = scene_id
        obs["split_seed"] = truth["split_seed"] = 123
        obs["log_f"][0] += scene_id*.001
        return obs, truth

    def test_no_freeze_means_no_draw_and_no_output(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(cache_module, "generate_scene") as draw:
            root = Path(tmp)
            with self.assertRaises(FileNotFoundError):
                cache_module.prepare_test_split(root/"out", "iid", root/"missing_freeze")
            self.assertFalse((root/"out").exists())
            draw.assert_not_called()

    def test_frozen_fixture_roundtrip_matches_open_features_and_never_reads_truth(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            freeze = root/"freeze"
            freeze.write_bytes(b"fixture no actual test authority")
            with patch.object(cache_module, "verify_freeze", return_value={"fixture": True}), \
                 patch.dict(cache_module.SPLITS, {"iid": (2, 123)}), \
                 patch.object(cache_module, "generate_scene", side_effect=self.fixture_scene):
                output = root/"out"
                cache_module.prepare_test_split(output, "iid", freeze)
                original_open = Path.open

                def observed_only(path, *args, **kwargs):
                    if path.name == "sidecars.jsonl":
                        raise AssertionError("inference must not open truth")
                    return original_open(path, *args, **kwargs)

                with patch.object(Path, "open", observed_only):
                    cache = cache_module.FrozenTestCache(output, "iid", freeze)
                self.assertEqual(len(load_supervision(cache)), 2)
                for obs, record in zip(cache.observations, cache.records):
                    direct = feature_record(obs)
                    for key in direct:
                        np.testing.assert_array_equal(record[key], direct[key])
                freeze.write_bytes(b"different fixture freeze")
                with self.assertRaisesRegex(ValueError, "identity, freeze or hash"):
                    cache_module.FrozenTestCache(output, "iid", freeze)

    def test_eight_manifests_require_exact_declared_counts_seeds_schema_and_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            specs = {name: (2, 100+i) for i, name in enumerate(cache_module.SPLITS)}
            roots = []
            for name, (count, seed) in specs.items():
                path = root/name
                path.mkdir()
                manifest = {"schema_version": 1, "status": "COMPLETE", "split": name, "count": count,
                            "split_seed": seed, "rows": [{"scene_id": i, "fingerprint": f"fixture:{name}:{i}"} for i in range(count)]}
                (path/"manifest.json").write_text(json.dumps(manifest))
                roots.append(path)
            with patch.dict(cache_module.SPLITS, specs):
                cache_module.verify_disjoint_manifests(roots)
                target = roots[0]/"manifest.json"
                original = json.loads(target.read_text())
                for changes in ({"count": 1, "rows": original["rows"][:1]}, {"split_seed": -1},
                                {"schema_version": 2}, {"rows": original["rows"][::-1]}):
                    target.write_text(json.dumps({**original, **changes}))
                    with self.assertRaises(ValueError):
                        cache_module.verify_disjoint_manifests(roots)
                target.write_text(json.dumps(original))
                with self.assertRaises(ValueError):
                    cache_module.verify_disjoint_manifests(roots[:-1])


if __name__ == "__main__":
    unittest.main()
