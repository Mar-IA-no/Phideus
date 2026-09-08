"""Fixed numerical cache fixtures, not prospective scene draws."""
import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np

from src.atencion_armonica.learned_partition_core import ARMS, observable_features
from src.atencion_armonica.learned_partition_cache import load_rows, save_rows, validate_rows
from src.atencion_armonica.structured_source_reader import signature, partition_energies

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/".agent-work/phideus-learned-reader-20260908/tests"


def fixture(n=8):
    a = np.arange(n*n, dtype=np.float32).reshape(n, n)/10
    z = (a+a.T).astype(np.float32)
    partitions = sorted([signature([[i] for i in range(n)]),
                         signature([list(range(i, i+4)) for i in range(0, n, 4)])])
    groups = sorted({g for p in partitions for g in p})
    costs = {arm: [.25 if len(g) >= 3 else 0. for g in groups] for arm in ARMS[1:]}
    scored = {"pool": {"canonical_to_observed": list(range(n))},
              "groups": [{"members": list(g), "size": len(g),
                          "costs": {arm: costs[arm][i] for arm in costs}}
                         for i, g in enumerate(groups)],
              "candidates": partition_energies(z.astype(np.float64), partitions, groups, costs)}
    return observable_features(scored, z)


class CacheTests(unittest.TestCase):
    def test_vectorized_validator_matches_frozen_scalar_reference(self):
        from dataclasses import replace
        from experiments.atencion_armonica.test_learned_partition_scalar_reference import scalar_validation_reference
        def outcome(fn, row):
            try:
                self.assertIs(fn(row), row)
                return True
            except ValueError:
                return False
        for n in (8, 32):
            row = fixture(n)
            self.assertTrue(outcome(scalar_validation_reference, row))
            self.assertTrue(outcome(validate_rows, row))
            variants = [replace(row, n=2), replace(row, groups=row.groups[::-1]),
                replace(row, candidates=row.candidates[::-1]), replace(row, groups=row.groups[:-1]),
                replace(row, group_features=row.group_features.astype(np.float32)),
                replace(row, incidence=row.incidence.astype(np.float64)),
                replace(row, costs={}), replace(row, candidates=(row.candidates[0][:-1],))]
            for name in ("group_features", "global_features", "incidence", *ARMS[1:]):
                original = row.costs[name] if name in ARMS[1:] else getattr(row, name)
                for index in np.ndindex(original.shape):
                    for value in (-1., 0., .125, 1., 2., float("nan")):
                        changed = copy.deepcopy(row)
                        a = changed.costs[name] if name in ARMS[1:] else getattr(changed, name)
                        a[index] = value
                        self.assertEqual(outcome(scalar_validation_reference, changed), outcome(validate_rows, changed),
                                         (n, name, index, value))
            for changed in variants:
                self.assertEqual(outcome(scalar_validation_reference, changed), outcome(validate_rows, changed))

    @classmethod
    def setUpClass(cls):
        BASE.mkdir(parents=True, exist_ok=True)

    def test_exact_roundtrip_and_no_overwrite(self):
        original = fixture()
        with tempfile.TemporaryDirectory(dir=BASE) as folder:
            path = Path(folder)/"features.npz"
            save_rows(path, original)
            restored = load_rows(path)
            for key in ("n", "groups", "candidates"):
                self.assertEqual(getattr(original, key), getattr(restored, key))
            for key in ("group_features", "global_features", "incidence"):
                np.testing.assert_array_equal(getattr(original, key), getattr(restored, key))
                self.assertEqual(getattr(original, key).dtype, getattr(restored, key).dtype)
            for arm in ARMS[1:]:
                np.testing.assert_array_equal(original.costs[arm], restored.costs[arm])
            with self.assertRaises(FileExistsError):
                save_rows(path, original)

    def test_invalid_semantics(self):
        for mutate in (
            lambda r: r.group_features.__setitem__((0, 0), 5),
            lambda r: r.group_features.__setitem__((0, 4), -1),
            lambda r: r.global_features.__setitem__((0, 1), 0),
            lambda r: r.incidence.__setitem__((0, 0), .75),
            lambda r: r.costs[ARMS[1]].__setitem__(0, .1),
            lambda r: r.costs[ARMS[2]].__setitem__(0, float("nan")),
        ):
            value = copy.deepcopy(fixture())
            mutate(value)
            with self.assertRaises(ValueError):
                validate_rows(value)

    def test_malformed_indices_dtype_or_truth_are_rejected(self):
        with tempfile.TemporaryDirectory(dir=BASE) as folder:
            path = Path(folder)/"original.npz"
            save_rows(path, fixture())
            with np.load(path, allow_pickle=False) as saved:
                original = {k: saved[k] for k in saved.files}
            for i, mutate in enumerate((
                lambda a: a.update(truth=np.zeros(8)),
                lambda a: a["group_offsets"].__setitem__(1, 0),
                lambda a: a["candidate_group_ids"].__setitem__(0, 999),
                lambda a: a.update(n=np.array(8, np.int32)),
                lambda a: a.update(incidence=a["incidence"].astype(np.float64)),
                lambda a: a["group_members"].__setitem__(0, -1),
            )):
                data = copy.deepcopy(original)
                mutate(data)
                bad = Path(folder)/f"invalid_{i}.npz"
                np.savez(bad, **data)
                with self.assertRaises(ValueError):
                    load_rows(bad)


if __name__ == "__main__":
    unittest.main()
