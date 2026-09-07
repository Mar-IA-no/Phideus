"""Harness tests on fixtures only; no historical pool or accelerator access."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import run_energy_partition_audit as h


SCRATCH = h.ROOT / ".agent-work/phideus-geometric-rebase-20260907/harness-tests"


def fixture_record(mid=0, polyphony=2, regime="easy"):
    weights = ([1/11, 2/11, 3/11, 5/11], [1/38, 7/38, 13/38, 17/38],
               [1/67, 11/67, 23/67, 32/67])
    peaks = [{"amp": float(np.sqrt(weight)), "freq": 100*(i+1)*(s+1),
              "source_id": s, "harmonic": i+1}
             for s in range(polyphony) for i, weight in enumerate(weights[s])]
    return {"mixture_id": mid, "polyphony": polyphony, "regime": regime, "peaks": peaks}


def fixture_pool(path):
    path.mkdir()
    records, cells = [], {}
    for k in (1, 2, 3):
        for regime in ("easy", "hard"):
            start = len(records)
            records.extend(fixture_record(start+i, k, regime) for i in range(4))
            cells[f"poly{k}_{regime}"] = [start, start+4]
    h.write_new(path / "pool_meta.json", h.encoded({"cell_ranges": cells}))
    h.write_new(path / "mixtures.jsonl", b"".join(h.encoded(r) for r in records))


class EnergyHarnessTests(unittest.TestCase):
    def setUp(self):
        SCRATCH.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=SCRATCH)
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_selected_stream_and_integrity(self):
        pool = self.root/"pool"
        fixture_pool(pool)
        records, provenance = h.select_records(pool)
        self.assertEqual(len(records), 24)
        self.assertEqual(provenance["pool_sha256"], h.digest((pool/"mixtures.jsonl").read_bytes()))
        with (pool/"mixtures.jsonl").open("ab") as stream:
            stream.write(records[0][1])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            h.select_records(pool)

    def test_missing_id(self):
        pool = self.root/"pool"
        fixture_pool(pool)
        path = pool/"mixtures.jsonl"
        lines = path.read_bytes().splitlines(keepends=True)
        path.write_bytes(b"".join(lines[:-1]))
        with self.assertRaisesRegex(ValueError, "missing"):
            h.select_records(pool)

    def test_input_boundary_and_relabeling(self):
        record = fixture_record()
        amplitudes = np.array([p["amp"] for p in record["peaks"]])
        with patch.object(h, "solve_energy_partition", wraps=h.solve_energy_partition) as solver:
            pred = h.infer(amplitudes, 123, "observed_logamp_float32", h.SEEDS[0])
        args, kwargs = solver.call_args
        self.assertEqual(len(args), 1)
        self.assertEqual(set(kwargs), {"tolerance"})
        decoded = h.accessible_amplitudes(amplitudes)
        np.testing.assert_array_equal(args[0], decoded[pred["permutation"]])
        labels = [p["source_id"] for p in record["peaks"]]
        self.assertEqual(h.evaluate(pred, labels), h.evaluate(pred, [99 if s == 0 else -8 for s in labels]))
        self.assertTrue(h.evaluate(pred, labels)["unique_exact_match"])

    def test_equivariance_semantics(self):
        a = np.full(8, .5)
        preds = [h.infer(a, 1, "pool_float64", seed) for seed in h.SEEDS]
        self.assertEqual(h.equivariance(*preds), "PASS")
        self.assertEqual(preds[0]["solver"]["status"], "MULTIPLE")
        partial = copy.deepcopy(preds[0])
        partial["solver"]["status"] = "LIMIT_NODES"
        self.assertEqual(h.equivariance(partial, preds[1]), "NOT_EVALUABLE_LIMIT")

    def test_complete_fixture_run_replay_and_denominators(self):
        pool = self.root/"pool"
        fixture_pool(pool)
        primary, replay = self.root/"primary", self.root/"replay"
        h.run(primary, pool)
        h.run(replay, pool)
        for name in ("selected_records.jsonl", "predictions.json", "summary.json", "manifest.json"):
            self.assertEqual((primary/name).read_bytes(), (replay/name).read_bytes())
        rows = json.loads((primary/"predictions.json").read_bytes())
        self.assertEqual(len(rows), 96)
        self.assertTrue(all(len(row["predictions"]) == 2 for row in rows))
        summary = json.loads((primary/"summary.json").read_bytes())["summary"]
        self.assertEqual(len(summary), 12)
        self.assertTrue(all(s["n"] == 8 for s in summary))
        for row in summary:
            self.assertEqual(sum(row["status_counts"].values()), 8)
            if row["view"] == "source_gain":
                self.assertEqual(row["status_counts"], {"PRIOR_VIOLATION": 8})
        with self.assertRaises(FileExistsError):
            h.run(primary, pool)


if __name__ == "__main__":
    unittest.main()
