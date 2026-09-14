"""Exact fast-path differential tests; fixtures only, no campaign operators."""
from copy import deepcopy
import itertools
import unittest
from unittest.mock import patch

import numpy as np

from src.atencion_armonica import operator_diagnostic_fast as fast
from src.atencion_armonica import operator_diagnostic_cached as cached
from src.atencion_armonica import operator_objective_artifacts as artifacts
from src.atencion_armonica import operator_objective_core as core
from src.atencion_armonica import operator_objective_scene as scene
from experiments.atencion_armonica.test_operator_objective_scene import example


class CodecTests(unittest.TestCase):
    def equal(self, value):
        expected = artifacts.bundle_bytes(value)
        actual = fast.bundle_bytes(value)
        self.assertEqual(actual, expected)
        self.assertEqual(artifacts.bundle_bytes(artifacts.load_bundle(*actual)), expected)

    def test_layout_endian_dtype_shape_and_empty_arrays(self):
        for dtype in ("bool", "i1", "<i8", ">u4", "<f4", ">f8"):
            array = np.arange(24).reshape(4, 6).astype(dtype)
            for value in (array, array[:, ::2], array.T, array[:0], array.reshape(2, 2, 2, 3)):
                with self.subTest(dtype=dtype, shape=value.shape):
                    self.equal({"array": value})
        self.equal(np.array(-0., np.float32))

    def test_scalars_strings_aliases_and_signed_zero(self):
        shared = {"a": [0., -0., None, True, np.float32(.1)], "z": np.arange(3)}
        self.equal({"ñ": shared, "a": (shared, shared), "escapes": "\n\t\\\"☃"})
        for value in (np.int64(2), np.uint64(2**63), np.float16(.1), np.float64(-0.),
                      np.bool_(True), np.str_("☃"), 2**80, -0., None):
            with self.subTest(value=repr(value)):
                self.equal(value)

    def test_no_stale_cache_between_calls_or_mutation(self):
        value = {"x": np.array([1., -0.], np.float32), "list": [1, 2]}
        first = fast.bundle_bytes(value)
        self.assertEqual(first, artifacts.bundle_bytes(value))
        value["x"][0] = 2
        value["list"].append(3)
        second = fast.bundle_bytes(value)
        self.assertNotEqual(first, second)
        self.assertEqual(second, artifacts.bundle_bytes(value))

    def test_invalid_values_and_cycles_rejected(self):
        cyclic = []
        cyclic.append(cyclic)
        invalid = ({"__ndarray__": {}}, {1: 0}, np.array([{}], object),
                   np.array([np.nan]), np.float32(np.inf), np.complex64(1j),
                   {"x": [float("nan")]}, {"x": {1}}, b"bytes", cyclic)
        for value in invalid:
            for encoder in (artifacts.bundle_bytes, fast.bundle_bytes):
                with self.subTest(encoder=encoder.__module__, value=type(value)):
                    with self.assertRaises((TypeError, ValueError, RecursionError)):
                        encoder(value)
        valid_alias = [1, 2]
        with self.assertRaises(ValueError):
            fast.bundle_bytes([valid_alias, valid_alias, {"__ndarray__": {}}])

    def test_custom_container_views_delegate_once_to_reference(self):
        class DifferentViews(dict):
            def values(self):
                return [1]
            def items(self):
                return [("x", {"__ndarray__": {}})]
        for encoder in (artifacts.bundle_bytes, fast.bundle_bytes):
            with self.assertRaises(ValueError):
                encoder({"nested": DifferentViews()})
        class Counting(dict):
            def __init__(self):
                super().__init__()
                self.calls = 0
            def items(self):
                self.calls += 1
                return [("x", self.calls)]
        left, right = Counting(), Counting()
        self.assertEqual(artifacts.bundle_bytes({"a": left}), fast.bundle_bytes({"a": right}))
        self.assertEqual((left.calls, right.calls), (1, 1))


class MeanAndPairTests(unittest.TestCase):
    def test_sequence_bits_missing_order_and_cache_hits(self):
        cache = fast.MeanCache()
        sequences = ([0.], [-0.], [None], [], [0., None], [None, 0.],
                     [1e16, 1., -1e16], [1e16, -1e16, 1.],
                     [np.nextafter(0., 1.), -0.], [np.float32(.1), np.int64(2)])
        for values in sequences:
            expected = artifacts.bundle_bytes(scene.mean_record(values))
            self.assertEqual(artifacts.bundle_bytes(cache(values)), expected)
            with patch.object(scene, "mean_record", side_effect=AssertionError("cache miss")):
                self.assertEqual(artifacts.bundle_bytes(cache(iter(values))), expected)
        self.assertEqual(len(cache.records), len(sequences))

    def test_invalid_values_rejected_even_after_valid_cache_entry(self):
        cache = fast.MeanCache()
        cache([0.])
        cache([1.])
        for values in ([False], [np.bool_(True)], ["0"], [np.nan], [np.inf], [object()]):
            for function in (cache, scene.mean_record):
                with self.assertRaises(ValueError):
                    function(values)

    def test_pairing_partial_support_and_cancellation_exact(self):
        left = {c: {"z": 1e16, "a": 1., "b": -1e16} for c in scene.CELLS}
        right = {c: {"z": 0., "a": -0., "b": 0.} for c in scene.CELLS}
        for i in range(10):
            if i:
                left[scene.CELLS[i-1]]["a"] = None
                right[scene.CELLS[i-1]]["b"] = None
            expected = core.paired_scene_difference(left, right)
            actual = fast._paired_scene_difference(left, right, means=fast.MeanCache())
            self.assertEqual(artifacts.bundle_bytes(actual), artifacts.bundle_bytes(expected))


class DiagnosticTests(unittest.TestCase):
    def equivalent(self, args):
        before = artifacts.bundle_bytes(args)
        reference = scene.diagnose_scene(*args)
        expected = artifacts.bundle_bytes(reference)
        result = fast.diagnose_scene(*args)
        self.assertEqual(artifacts.bundle_bytes(result), expected)
        self.assertEqual(fast.bundle_bytes(result), expected)
        self.assertEqual(fast.bundle_bytes(cached.diagnose_scene(*args)), expected)
        self.assertEqual(artifacts.bundle_bytes(args), before)
        return result

    def test_empty_singleton_distinct_strata_and_full35_fixture(self):
        for parts in ([], [((0, 1, 2, 3), (4, 5, 6, 7))], None):
            self.equivalent(example(parts))
        parts = sorted([((0, 1, 2, 3), (4, 5, 6, 7), tuple(range(8, 16))),
                        (tuple(range(8)), tuple(range(8, 16))),
                        tuple(tuple(range(i, i+4)) for i in range(0, 16, 4))])
        args = example(parts, 16)
        args[0]["winning_branches"]["extended_ub"] = ["base-low", "base-high", "deformed-low"]
        self.equivalent(args)
        all_events = set(range(8))
        parts = sorted((tuple(g), tuple(sorted(all_events-set(g))))
                       for t in itertools.combinations(range(1, 8), 3) for g in [(0, *t)])
        args = example(parts)
        for ai, arm in enumerate(scene.ARMS):
            for ci, cell in enumerate(scene.CELLS):
                x = np.arange(70).reshape(35, 2)
                args[2][arm][cell] = ((x*(ci+1)+ai*13) % 37).astype(np.float32)/np.float32(37)
        self.equivalent(args)

    def test_partial_support_ties_and_no_cross_call_leakage(self):
        args = example()
        for i, cell in enumerate(scene.CELLS):
            if i % 2:
                args[2]["generative"][cell][:] = .125
            args[2]["local"][cell] += np.float32(i/20)
        self.equivalent(args)
        args[1][:] = 0
        args[0]["scores"]["base_ub"][:] = 0
        self.equivalent(args)

    def test_invalid_roster_and_inputs_still_rejected(self):
        mutations = (lambda a: a[2]["local"].pop(scene.CELLS[0]),
                     lambda a: a[2]["local"].__setitem__(scene.CELLS[0], np.zeros((2, 2), np.float64)),
                     lambda a: a[2]["local"][scene.CELLS[0]].fill(-1),
                     lambda a: a[0]["scores"]["base_ub"].fill(np.nan))
        for mutate in mutations:
            args = example()
            mutate(args)
            for implementation in (scene.diagnose_scene, fast.diagnose_scene):
                with self.assertRaises(ValueError):
                    implementation(*deepcopy(args))


if __name__ == "__main__":
    unittest.main()
