"""Data-port checks using already observed historical scenes, never fresh seeds."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from src.atencion_armonica import learned_partition_data as data
from src.atencion_armonica import learned_partition_provenance as provenance
from src.atencion_armonica.structured_source_artifacts import write_json

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/".agent-work/phideus-learned-reader-20260908/tests"
OLD = ROOT/"data/atencion_armonica/structured_source_reader_v1"


def historical_first(split):
    with (OLD/f"{split}_data/observations.jsonl").open() as f:
        obs = json.loads(next(f))
    with (OLD/f"{split}_data/sidecars.jsonl").open() as f:
        truth = json.loads(next(f))
    return obs, truth


class DataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        BASE.mkdir(parents=True, exist_ok=True)

    def test_copied_law_matches_already_observed_four_regimes_exactly(self):
        for split in ("iid", "ood_beta", "ood_polyphony", "deformed_family"):
            obs, truth = historical_first(split)
            reproduced = data._draw_scene(split, obs["scene_id"], obs["split_seed"])
            self.assertEqual(reproduced, (obs, truth))

    def test_complete_fixed_shard_roster_and_bounds(self):
        ids = sum([data.scene_ids("train", i) for i in range(8)], [])
        self.assertEqual(ids, list(range(4096)))
        self.assertEqual(data.scene_ids("ood_polyphony", 0), list(range(512)))
        for args in (("train", 8), ("iid", 1), ("test", 0), ("train", 0.)):
            with self.assertRaises(ValueError):
                data.scene_ids(*args)

    def test_observation_identity_truth_and_quantization_rejection(self):
        obs, _ = historical_first("iid")
        obs["split_seed"] = data.SPLITS["iid"][1]  # Identity-only fixture, same historical q32.
        data.validate_observation(obs, 0, "iid")
        for mutate in (lambda r: r.update(source_ids=[0]*8), lambda r: r.update(scene_id=1),
                       lambda r: r.update(split_seed=0), lambda r: r["log_f"].__setitem__(0, .1)):
            r = copy.deepcopy(obs)
            mutate(r)
            with self.assertRaises(ValueError):
                data.validate_observation(r, 0, "iid")

    def test_supervision_reconstructs_sources_and_rejects_semantic_mismatch(self):
        for split in ("iid", "ood_beta", "ood_polyphony", "deformed_family"):
            obs, truth = historical_first(split)
            with tempfile.TemporaryDirectory(dir=BASE) as folder:
                write_json(Path(folder)/"sidecars.jsonl", truth)
                cache = SimpleNamespace(root=Path(folder), observations=[obs], split=split)
                self.assertEqual(data._validate_supervision(cache, [truth]), [truth])
        obs, truth = historical_first("iid")
        for mutate in (lambda t: t["sources"][0].update(beta=.5),
                       lambda t: t["source_ids"].__setitem__(0, 99),
                       lambda t: t["log_f_ideal"].__setitem__(0, 0.),
                       lambda t: t.update(mean_log_f_observed=0.),
                       lambda t: t.update(extra=True)):
            changed = copy.deepcopy(truth)
            mutate(changed)
            with tempfile.TemporaryDirectory(dir=BASE) as folder:
                write_json(Path(folder)/"sidecars.jsonl", changed)
                with self.assertRaises(ValueError):
                    data._validate_supervision(SimpleNamespace(root=Path(folder), observations=[obs], split="iid"), [changed])

    def test_internally_consistent_forged_sidecar_is_not_the_seeded_draw(self):
        obs, truth = historical_first("iid")
        forged = copy.deepcopy(truth)
        shift = np.roll(np.arange(len(obs["log_f"])), 1)
        for key in ("permutation", "log_f_ideal", "source_ids", "partial_indices"):
            forged[key] = np.asarray(truth[key])[shift].tolist()
        # Preserve exactly the old measured vector while reassigning sources.
        measured = np.asarray(truth["log_f_ideal"])+np.asarray(truth["sensor_log_noise"])
        forged["sensor_log_noise"] = (measured-np.asarray(forged["log_f_ideal"])).tolist()
        self.assertNotEqual(forged["source_ids"], truth["source_ids"])
        with self.assertRaises(ValueError):
            data._validate_supervision(SimpleNamespace(observations=[obs], split="iid"), [forged])

    def test_supervision_revalidates_bundle_before_and_after_parsing(self):
        obs, truth = historical_first("iid")
        with tempfile.TemporaryDirectory(dir=BASE) as folder:
            root = Path(folder)
            write_json(root/"sidecars.jsonl", truth)
            cache = SimpleNamespace(root=root, observations=[obs], split="iid", reference={}, common={}, manifest={"fixture": 1})
            with patch.object(data, "_bundle", return_value=(root, cache.manifest)) as verify:
                self.assertEqual(data.load_supervision(cache), [truth])
                self.assertEqual(verify.call_count, 2)
            with patch.object(data, "_bundle", side_effect=[(root, cache.manifest), (root, {"changed": True})]):
                with self.assertRaises(ValueError):
                    data.load_supervision(cache)

    def test_authorization_denial_precedes_rng_and_output(self):
        with tempfile.TemporaryDirectory(dir=BASE) as folder:
            output = Path(folder)/"must_not_exist"
            with patch("src.atencion_armonica.learned_partition_gate.verify_data_stage", side_effect=PermissionError), \
                 patch.object(data, "_draw_scene") as draw:
                with self.assertRaises(PermissionError):
                    data.prepare_shard(output, "iid", 0, authorization={}, previous={}, earlier_shards=[])
                draw.assert_not_called()
                self.assertFalse(output.exists())

    def test_exact_prior_roster_and_mechanical_alias(self):
        corpus = provenance.prior_corpus()
        self.assertEqual(len(corpus["observation_references"]), 13)
        self.assertEqual(len(corpus["fingerprints"]), 15684)
        self.assertEqual(len(provenance.mechanical_fingerprints()), 4)
        self.assertEqual(corpus["fingerprints"], sorted(set(corpus["fingerprints"])))


if __name__ == "__main__":
    unittest.main()
