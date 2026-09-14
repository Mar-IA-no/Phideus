"""Frozen-corpus binding checks and tiny diagnostic-result mutation fixtures.

The optional real-corpus tests read only closed JSON and prediction bytes, not
factor archives, truth sidecars, models, or a scientific diagnostic roster.
"""
from copy import deepcopy
from pathlib import Path
import unittest

import numpy as np

from src.atencion_armonica import operator_objective_corpus as corpus
from src.atencion_armonica import operator_objective_scene as scene
from src.atencion_armonica import operator_objective_sources as sources
from src.atencion_armonica.generative_evidence_supervision import candidate_supervision
from src.atencion_armonica.learned_partition_readout import choose_costs
from experiments.atencion_armonica.test_operator_objective_scene import example

ROOT = Path(__file__).resolve().parents[2]


def extracted_fixture(empty=False):
    compact, labels, predictions = example([] if empty else None)
    ps = compact["partitions"]
    target = candidate_supervision(ps, labels)
    choices = {}
    for family in ("base", "extended"):
        key = family+"_ub"
        i = min(range(len(ps)), key=lambda i: (compact["scores"][key][i], ps[i])) if ps else None
        choices[family] = None if i is None else {"candidate_index": i, "partition": ps[i],
            "UB": float(compact["scores"][key][i]), "branch": compact["winning_branches"][key][i]}
    old = {"raw_entropies": target["raw"].tolist(), "normalized_targets": target["targets"].tolist(),
           "candidate_metrics": target["metrics"],
           "coverage": {"has_output": bool(ps), "candidate_count": len(ps), "planted": "pool" if ps else "absent"},
           "references": {k: {"choice": v} if v is not None else None for k, v in choices.items()}}
    extracted = {"compact": compact, "canonical_labels": labels, "predictions": predictions,
                 "archived_result": old, "archived_choices": choices,
                 "archived_decisions": {a: {c: choose_costs(h, ps) if ps else None for c, h in rows.items()}
                                        for a, rows in predictions.items()}}
    return extracted, scene.diagnose_scene(compact, labels, predictions)


class CotejoTests(unittest.TestCase):
    def test_inherited_metrics_choices_and_float32_readout_match(self):
        for empty in (False, True):
            extracted, result = extracted_fixture(empty)
            coverage = corpus.verify_scene_result(extracted, result)
            self.assertEqual(coverage["has_output"], not empty)

    def test_target_float32_one_bit_mutation_is_rejected(self):
        extracted, result = extracted_fixture()
        old = extracted["archived_result"]
        old["normalized_targets"][1][0] = float(np.nextafter(np.float32(old["normalized_targets"][1][0]), np.float32(1)))
        with self.assertRaisesRegex(ValueError, "targets"):
            corpus.verify_scene_result(extracted, result)

    def test_candidate_metric_and_coverage_mutations_rejected(self):
        for field in ("ari", "split_normalized", "k_error", "exact_partition"):
            extracted, result = extracted_fixture()
            extracted["archived_result"]["candidate_metrics"][0][field] = 10.
            with self.assertRaisesRegex(ValueError, "metric"):
                corpus.verify_scene_result(extracted, result)
        extracted, result = extracted_fixture()
        extracted["archived_result"]["coverage"]["planted"] = "neighbor"
        with self.assertRaisesRegex(ValueError, "coverage"):
            corpus.verify_scene_result(extracted, result)

    def test_ub_branch_and_learned_sum_mutations_rejected(self):
        extracted, result = extracted_fixture()
        extracted["archived_choices"]["extended"]["branch"] = "deformed-low"
        with self.assertRaisesRegex(ValueError, "choice"):
            corpus.verify_scene_result(extracted, result)
        extracted, result = extracted_fixture()
        extracted["archived_decisions"]["local"][scene.CELLS[0]]["cost"] += .001
        with self.assertRaisesRegex(ValueError, "float32 decision"):
            corpus.verify_scene_result(extracted, result)

    def test_prediction_offsets_and_dtype_envelope(self):
        arrays = {"components": np.zeros((2, 2), np.float32),
                  "offsets": np.array([0, 2]+[2]*511, np.int64)}
        self.assertIs(corpus.prediction_array(arrays), arrays)
        for field, replacement in (("components", arrays["components"].astype(np.float64)),
                                    ("offsets", np.arange(513, dtype=np.int64)),
                                    ("offsets", np.array([0, 83]+[83]*511, np.int64))):
            bad = {**arrays, field: replacement}
            with self.assertRaises(ValueError):
                corpus.prediction_array(bad)


class CachedReadOnly:
    """Mutations only in returned copies; real authenticated sources unchanged."""
    def __init__(self):
        self.reader = sources.AuthenticatedReader(ROOT)
        self.json_cache, self.byte_cache = {}, {}
        self.mutation = None

    def json(self, ref, *, base=None):
        key = (base, sources.encoded(ref))
        if key not in self.json_cache:
            self.json_cache[key] = self.reader.json(ref, base=base)
        value = deepcopy(self.json_cache[key])
        if self.mutation:
            self.mutation(ref["path"], value)
        return value

    def bytes(self, ref, *, base=None):
        key = (base, sources.encoded(ref))
        if key not in self.byte_cache:
            self.byte_cache[key] = self.reader.bytes(ref, base=base)
        return self.byte_cache[key]

    def fit_json(self, *args, **kwargs):
        raise AssertionError("no factor decode in receipt-binding development tests")


@unittest.skipUnless((ROOT/corpus.COMPLETION["path"]).is_file(), "optional closed local corpus not present")
class ClosedHeaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reader = CachedReadOnly()
        cls.root = corpus.ClosedCorpus(cls.reader)
        cls.header = cls.root.header("iid")

    def setUp(self):
        self.reader.mutation = None
        self.addCleanup(setattr, self.reader, "mutation", None)

    def test_all_45_authenticated_before_27_originals(self):
        self.assertEqual(len(self.header["originals"]), 27)
        self.assertEqual(len(self.reader.byte_cache), 45)
        self.assertEqual(len(self.root.selected), 27)
        self.assertIsNone(self.root.receipts)
        with self.assertRaisesRegex(ValueError, "inventory"):
            self.root.load_split("iid")

    def test_distinct_freeze_and_inference_orders_resolve_by_explicit_map(self):
        rows = self.header["index"]["roster"]
        self.assertEqual(rows[1]["inference_index"], 1)
        self.assertEqual(rows[1]["freeze_index"], 9)
        self.assertEqual(sorted(r["freeze_index"] for r in rows), list(range(45)))

    def test_duplicate_frozen_index_rejected(self):
        root = corpus.ClosedCorpus(self.reader)
        def mutate(path, value):
            if path.endswith("/iid/predictions.json"):
                value["roster"][1]["freeze_index"] = value["roster"][0]["freeze_index"]
        self.reader.mutation = mutate
        with self.assertRaisesRegex(ValueError, "bijectively"):
            root.header("iid")

    def test_intervention_metadata_not_ignored(self):
        root = corpus.ClosedCorpus(self.reader)
        def mutate(path, value):
            if path.endswith("generative-zero.json"):
                value["truth_access"] = True
        self.reader.mutation = mutate
        with self.assertRaisesRegex(ValueError, "role differs"):
            root.header("iid")

    def test_readout_not_bound_to_another_prediction(self):
        root = corpus.ClosedCorpus(self.reader)
        def mutate(path, value):
            if path == "readouts/00.json":
                value["prediction"]["sha256"] = "0"*64
        self.reader.mutation = mutate
        with self.assertRaisesRegex(ValueError, "exact prediction"):
            root.header("iid")

    def test_closed_replay_worker_must_match_its_stage(self):
        def mutate(path, value):
            if path.endswith("attempt-0012.worker.json"):
                value["stage"]["split"] = "iid"
        self.reader.mutation = mutate
        with self.assertRaisesRegex(ValueError, "closed stage"):
            corpus.ClosedCorpus(self.reader)


if __name__ == "__main__":
    unittest.main()
