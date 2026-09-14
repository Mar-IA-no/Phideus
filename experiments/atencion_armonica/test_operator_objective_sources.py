"""Tiny source-port fixtures, not scenes or new sham draws for the campaign."""
from copy import deepcopy
import gzip
import hashlib
from io import BytesIO
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np

from src.atencion_armonica import generative_evidence as ge
from src.atencion_armonica import operator_objective_sources as sources

FIXTURES = Path(__file__).resolve().parents[2]/".agent-work/phideus-operator-objective-20260914/source-tests"


def fitted_fixture(partitions=None, n=8):
    ps = partitions if partitions is not None else [((0, 1, 2, 3), (4, 5, 6, 7)),
                                                  ((0, 1, 2, 4), (3, 5, 6, 7))]
    observation = {"scene_id": 0, "split_seed": 2026090982,
                   "log_f": np.arange(n, dtype=np.float32).tolist()}
    rows = [{"partition": p, "origin": "pool", "status": "SUPPORTED"} for p in ps]
    fits = [{"partition": p, "status": "FITTED", "branches": {
        b: {"LB": float(i+j), "UB": float(i+j+1)}
        for j, b in enumerate(sources.BRANCHES) if len(p) in sources.law.BRANCHES[b][2]}}
        for i, p in enumerate(ps)]
    return {"observation": observation, "inventory": {"candidates": rows},
            "fits": fits, "group_factors": []}


class FilePortTests(unittest.TestCase):
    def setUp(self):
        FIXTURES.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="case-", dir=FIXTURES)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.reader = sources.AuthenticatedReader(self.root, maximum_bytes=8192)

    def write(self, name, raw):
        path = self.root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        return {"path": name, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}

    def test_explicit_base_and_hash_only_reference(self):
        ref = self.write("base/item.json", sources.encoded({"x": [1, 2]}))
        relative = {**ref, "path": "item.json"}
        self.assertEqual(self.reader.json(relative, base="base"), {"x": [1, 2]})
        self.assertEqual(self.reader.json({k: v for k, v in ref.items() if k != "bytes"}), {"x": [1, 2]})
        with self.assertRaises(FileNotFoundError):
            self.reader.json(relative)  # No filename search or fallback base.

    def test_reject_hash_size_and_noncanonical_json(self):
        ref = self.write("item.json", sources.encoded({"x": 1}))
        for bad in ({**ref, "sha256": "0"*64}, {**ref, "bytes": ref["bytes"]+1},
                    {**ref, "bytes": True}, {**ref, "sha256": "xyz"}, {**ref, "extra": 0}):
            with self.assertRaises(ValueError):
                self.reader.json(bad)
        ref = self.write("spaced.json", b'{"x": 1}')
        with self.assertRaises(ValueError):
            self.reader.json(ref)

    def test_reject_traversal_absolute_and_symlink_components(self):
        ref = self.write("actual/item.json", sources.encoded({"x": 1}))
        for path in ("../item.json", "/etc/passwd", "actual//item.json", "actual/./item.json",
                     "actual/../actual/item.json", "", "actual\\item.json"):
            with self.assertRaises(ValueError):
                self.reader.json({**ref, "path": path})
        (self.root/"alias").symlink_to(self.root/"actual", target_is_directory=True)
        (self.root/"linked.json").symlink_to(self.root/"actual/item.json")
        for path in ("alias/item.json", "linked.json"):
            with self.assertRaises(OSError):
                self.reader.json({**ref, "path": path})
        with self.assertRaises(ValueError):
            sources.AuthenticatedReader(self.root/"alias")

    def test_regular_file_and_byte_limit(self):
        ref = self.write("large.bin", b"a"*8193)
        for item in (ref, {k: v for k, v in ref.items() if k != "bytes"}):
            with self.assertRaises(ValueError):
                self.reader.bytes(item)
        with self.assertRaises(ValueError):
            self.reader.bytes({"path": "large.bin", "bytes": -1, "sha256": ref["sha256"]})

    def test_arrays_retain_dtype_and_bound_decoded_size(self):
        stream = BytesIO()
        np.savez_compressed(stream, components=np.zeros((4, 2), np.float32), offsets=np.array([0, 4], np.int64))
        ref = self.write("inputs.npz", stream.getvalue())
        arrays = self.reader.arrays(ref)
        self.assertEqual(arrays["components"].dtype, np.float32)
        np.testing.assert_array_equal(arrays["offsets"], [0, 4])
        with self.assertRaises(ValueError):
            self.reader.arrays(ref, maximum_decoded_bytes=8)

    def test_arrays_reject_pickle_and_invalid_members(self):
        stream = BytesIO()
        np.savez_compressed(stream, obj=np.asarray([{}], object))
        with self.assertRaises(ValueError):
            self.reader.arrays(self.write("object.npz", stream.getvalue()))
        for member in ("../x.npy", "x.txt", "folder/x.npy"):
            stream = BytesIO()
            with zipfile.ZipFile(stream, "w") as archive:
                archive.writestr(member, b"x")
            with self.assertRaises(ValueError):
                self.reader.arrays(self.write("invalid.npz", stream.getvalue()))

    def test_array_header_cannot_request_allocation_larger_than_payload(self):
        from unittest.mock import patch
        member = BytesIO()
        np.lib.format.write_array_header_1_0(member, {"descr": "<f4", "fortran_order": False,
                                                    "shape": (2**40,)})
        stream = BytesIO()
        with zipfile.ZipFile(stream, "w") as archive:
            archive.writestr("oversized.npy", member.getvalue())
        ref = self.write("oversized.npz", stream.getvalue())
        with patch.object(np, "load", side_effect=AssertionError("must fail before allocation")):
            with self.assertRaises(ValueError):
                self.reader.arrays(ref)

    def compressed(self, value):
        raw = sources.encoded(value)
        ref = self.write("factors/fit.gz", gzip.compress(raw, compresslevel=3, mtime=0))
        return {**ref, "path": "fit.gz", "codec": "canonical-json-gzip3-mtime0",
                "decoded_bytes": len(raw), "decoded_sha256": hashlib.sha256(raw).hexdigest()}

    def test_factor_decode_receipts_and_limit(self):
        value = fitted_fixture()
        ref = self.compressed(value)
        self.assertEqual(sources.encoded(self.reader.fit_json(ref, base="factors")), sources.encoded(value))
        for bad in ({**ref, "decoded_bytes": ref["decoded_bytes"]-1},
                    {**ref, "decoded_sha256": "0"*64}, {**ref, "codec": "gzip"}):
            with self.assertRaises(ValueError):
                self.reader.fit_json(bad, base="factors")
        with self.assertRaises(ValueError):
            self.reader.fit_json(ref, base="factors", maximum_decoded_bytes=4)

    def test_factor_codec_not_merely_decodable(self):
        value = fitted_fixture()
        ref = self.compressed(value)
        changed = self.write("factors/fit.gz", gzip.compress(sources.encoded(value), compresslevel=9, mtime=0))
        ref.update({k: changed[k] for k in ("sha256", "bytes")})
        with self.assertRaises(ValueError):
            self.reader.fit_json(ref, base="factors")


class CompactPortTests(unittest.TestCase):
    def test_branch_channels_equal_inherited_formula_and_tie_names(self):
        value = fitted_fixture()
        # UB tie goes to base-high lexicographically, not declared branch order.
        value["fits"][0]["branches"]["base-high"]["UB"] = 1.
        result = sources.compact_fit(value, value["observation"])
        channel, mask, _ = ge.candidate_channel(result["partitions"], value["fits"], 8)
        np.testing.assert_array_equal(result["log_channel"], channel)
        np.testing.assert_array_equal(result["available"], mask)
        self.assertEqual(result["winning_branches"]["extended_ub"][0], "base-high")
        self.assertEqual(result["scores"]["base_lb"].tolist(), [0., 1.])
        self.assertNotIn("group_factors", result)

    def test_inventory_filter_not_first_n_rows_and_canonical_order(self):
        value = fitted_fixture()
        excluded = {"partition": (tuple(range(8)),), "origin": "pool",
                    "status": "OUTSIDE_GENERATIVE_CARDINALITY"}
        value["inventory"]["candidates"] = [excluded, *reversed(value["inventory"]["candidates"])]
        result = sources.compact_fit(value, value["observation"])
        self.assertEqual(result["partitions"], [f["partition"] for f in value["fits"]])
        self.assertEqual(len(result["inventory"]["candidates"]), 3)

    def test_k_four_only_base_low_and_empty_universe(self):
        value = fitted_fixture([tuple(tuple(range(i, i+4)) for i in range(0, 16, 4))], n=16)
        result = sources.compact_fit(value, value["observation"])
        np.testing.assert_array_equal(result["available"], [[True, False, False]])
        for winners in result["winning_branches"].values():
            self.assertEqual(winners, ["base-low"])
        value = fitted_fixture([])
        result = sources.compact_fit(value, value["observation"])
        self.assertEqual(result["bounds"].shape, (0, 6))
        self.assertEqual(result["scores"]["extended_ub"].shape, (0,))

    def test_identity_support_and_bounds_fail_closed(self):
        variants = []
        value = fitted_fixture(); value["fits"].reverse(); variants.append(value)
        value = fitted_fixture(); value["inventory"]["candidates"] *= 2; variants.append(value)
        value = fitted_fixture(); value["inventory"]["candidates"][0]["status"] = "EXCLUDED"; variants.append(value)
        value = fitted_fixture(); del value["fits"][0]["branches"]["base-high"]; variants.append(value)
        for bound in ("1", True, -1., float("nan"), 100.):
            value = fitted_fixture(); value["fits"][0]["branches"]["base-low"]["LB"] = bound; variants.append(value)
        for value in variants:
            with self.assertRaises(ValueError):
                sources.compact_fit(value, value["observation"])
        value = fitted_fixture()
        with self.assertRaises(ValueError):
            sources.compact_fit(value, {**value["observation"], "scene_id": 2})

    def delivered(self, partitions=None, n=8):
        value = fitted_fixture(partitions, n)
        compact = sources.compact_fit(value, value["observation"])
        norm = {"mean": [1.]*6, "scale": [2.]*6}
        expected = ((compact["log_channel"]-1)/2).astype(np.float32)
        expected[~np.repeat(compact["available"], 2, axis=1)] = 0.
        # Unit-test reference generator only; production verification never calls RNG.
        decoupled, sham = ge.decouple(compact["partitions"], expected, split_seed=3, scene_id=0)
        return compact, norm, {"local": np.zeros_like(expected), "generative": expected, "decoupled": decoupled}, sham

    def test_frozen_normalization_and_saved_donors_without_rng(self):
        from unittest.mock import patch
        compact, norm, delivered, sham = self.delivered()
        with patch.object(np.random, "default_rng", side_effect=AssertionError("no new draw")):
            result = sources.verify_delivered_channel(compact, norm, delivered, sham)
        np.testing.assert_array_equal(result["donors"], [1, 0])
        np.testing.assert_array_equal(result["decoupled"], delivered["generative"][[1, 0]])

    def test_mask_zero_after_normalization_and_empty_channels(self):
        ps = [tuple(tuple(range(i, i+4)) for i in range(0, 16, 4))]
        for compact, norm, delivered, sham in (self.delivered(ps, 16), self.delivered([])):
            sources.verify_delivered_channel(compact, norm, delivered, sham)
        self.assertTrue(np.all(self.delivered(ps, 16)[2]["generative"][:, 2:] == 0))

    def test_reject_changed_channels_donors_attribution_and_normalizer(self):
        compact, norm, delivered, sham = self.delivered()
        for arm in delivered:
            bad = deepcopy(delivered); bad[arm][0, 0] += 1
            with self.assertRaises(ValueError):
                sources.verify_delivered_channel(compact, norm, bad, sham)
        for donors in ([0, 0], [True, 0], [0, 1]):
            bad = {**sham, "donors": donors}
            with self.assertRaises(ValueError):
                sources.verify_delivered_channel(compact, norm, delivered, bad)
        bad = deepcopy(sham); bad["strata"][0]["changed_fraction"] = .123
        with self.assertRaises(ValueError):
            sources.verify_delivered_channel(compact, norm, delivered, bad)
        for scale in ([0.]*6, ["2"]*6, [float("nan")]*6):
            with self.assertRaises(ValueError):
                sources.verify_delivered_channel(compact, {**norm, "scale": scale}, delivered, sham)


if __name__ == "__main__":
    unittest.main()
