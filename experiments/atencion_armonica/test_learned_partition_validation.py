"""Ephemeral DAG boundaries using fixed tensors and owned temporary bundles."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from experiments.atencion_armonica.test_learned_partition_training import BASE, advance
from src.atencion_armonica import learned_partition_data as data
from src.atencion_armonica import learned_partition_snapshots as snapshots
from src.atencion_armonica import learned_partition_selection as selection
from src.atencion_armonica.learned_partition_training import TrainingKernel
from src.atencion_armonica.learned_partition_validation import boundary, memoized, validation_pass
from src.atencion_armonica.structured_source_artifacts import seal_bundle, write_json


class ValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        BASE.mkdir(parents=True, exist_ok=True)
        torch.set_num_threads(1)

    def test_scopes_copy_results_and_drop_success_or_exception(self):
        seen = []
        @memoized
        def node(x=1):
            seen.append(x)
            return {"values": [x]}
        @boundary
        def consumer():
            node()["values"].append(99)
            return node(x=1)
        self.assertEqual(consumer(), {"values": [1]})
        self.assertEqual(seen, [1])
        consumer()
        self.assertEqual(seen, [1, 1])
        with self.assertRaises(RuntimeError), validation_pass() as session:
            consumer()
            raise RuntimeError("work failed")
        self.assertEqual(session.results, {})
        consumer()
        self.assertEqual(seen, [1]*4)
        node(); node()  # No session, no implicit persistent cache.
        self.assertEqual(seen, [1]*6)

    def test_failed_nodes_and_cycles_never_publish_results(self):
        @memoized
        def cyclic():
            return cyclic()
        with validation_pass() as session:
            for _ in range(2):
                with self.assertRaisesRegex(ValueError, "cyclic"):
                    cyclic()
                self.assertEqual(session.results, {})
                self.assertEqual(session.active, set())

    def test_snapshot_cache_preserves_exact_binding_types(self):
        binding = {"fixture": [1, 2]}
        kernel = TrainingKernel("shared_source", 2026090891, binding=binding, count=32)
        with tempfile.TemporaryDirectory(dir=BASE) as folder:
            ref = snapshots.write_snapshot(folder, "initial", kernel)
            with validation_pass():
                snapshots.read_snapshot(ref, expected_binding=binding)
                with self.assertRaises(ValueError):
                    snapshots.read_snapshot(ref, expected_binding={"fixture": (1, 2)})
            with self.assertRaises(ValueError):
                snapshots.read_snapshot(ref, expected_binding={"fixture": (1, 2)})
        @memoized
        def identity(value):
            return value
        with validation_pass():
            for value in (True, 1, 1., "1", None, [1], (1,)):
                self.assertIs(type(identity(value)), type(value))
            with self.assertRaises(ValueError):
                identity({1: "non-string-key"})
            with self.assertRaises(ValueError):
                identity({1, 2})

    def test_direct_selection_reference_cannot_change_hash_inside_pass(self):
        with tempfile.TemporaryDirectory(dir=BASE) as folder:
            path = Path(folder)/"freeze.json"
            write_json(path, {"selection": {}, "version": 1})
            first = selection.p.reference(path)
            with patch.object(selection, "validate_freeze"), patch.object(selection.gate, "common_binding", return_value={}):
                with validation_pass():
                    self.assertEqual(selection.verify_selection_chain(first, {})["version"], 1)
                    replacement = Path(folder)/"replacement.json"
                    write_json(replacement, {"selection": {}, "version": 2})
                    replacement.replace(path)  # Deliberate mutation of this test's own fixture.
                    second = selection.p.reference(path)
                    with self.assertRaisesRegex(ValueError, "conflicting hashes"):
                        selection.verify_selection_chain(second, {})
                self.assertEqual(selection.verify_selection_chain(second, {})["version"], 2)

    def bundle(self, folder, *, sidecar=False):
        root = Path(folder)
        write_json(root/("sidecars.jsonl" if sidecar else "payload.json"), {"mechanical": True})
        seal_bundle(root, role="learned_observation_shard", binding={"common": {"fixture": True}}, resources={})
        return data.provenance.reference(root/"manifest.json")

    def test_bundle_dedup_conflicting_references_and_fresh_after_mutation(self):
        with tempfile.TemporaryDirectory(dir=BASE) as folder:
            ref = self.bundle(folder)
            with patch.object(data, "verify_bundle", wraps=data.verify_bundle) as verifier:
                with validation_pass() as session:
                    for _ in range(4):
                        data._bundle(ref, "learned_observation_shard", {"fixture": True})
                    self.assertEqual(verifier.call_count, 1)
                    for changed, role, binding in (({**ref, "sha256": "0"*64}, "learned_observation_shard", {"fixture": True}),
                            (ref, "another_role", {"fixture": True}), (ref, "learned_observation_shard", {"fixture": False})):
                        with self.assertRaisesRegex(ValueError, "conflicting"):
                            data._bundle(changed, role, binding)
                self.assertEqual(session.results, {})
                with validation_pass():
                    data._bundle(ref, "learned_observation_shard", {"fixture": True})
                self.assertEqual(verifier.call_count, 2)
                write_json(Path(folder)/"unexpected.json", {"mutation": True})
                with validation_pass(), self.assertRaises(ValueError):
                    data._bundle(ref, "learned_observation_shard", {"fixture": True})

    def test_supervision_internal_after_check_does_not_join_outer_cache(self):
        with tempfile.TemporaryDirectory(dir=BASE) as folder:
            ref = self.bundle(folder, sidecar=True)
            root, manifest = data._bundle(ref, "learned_observation_shard", {"fixture": True})
            cache = SimpleNamespace(reference=ref, common={"fixture": True}, root=root, manifest=manifest)
            def mutate(_cache, _truths):
                write_json(root/"unexpected.json", {"mutation_during_parse": True})
            with validation_pass(), patch.object(data, "_validate_supervision", side_effect=mutate):
                data._bundle(ref, "learned_observation_shard", cache.common)
                with self.assertRaises(ValueError):
                    data.load_supervision(cache)

    def test_eleven_snapshots_are_loaded_once_per_pass_and_not_returned_by_alias(self):
        kernel = TrainingKernel("shared_source", 2026090891, binding={"fixture": "validation_only"}, count=32)
        with tempfile.TemporaryDirectory(dir=BASE) as folder:
            refs = [snapshots.write_snapshot(folder, "initial", kernel)]
            for index in range(10):
                advance(kernel)
                refs.append(snapshots.write_snapshot(folder, f"step_{index+1}", kernel, parents=[refs[-1]]))
            with patch.object(torch, "load", wraps=torch.load) as loader:
                for pass_index in range(2):
                    with validation_pass():
                        for ref in refs:
                            state, _ = snapshots.read_snapshot(ref, expected_binding=kernel.binding)
                            state["binding"]["fixture"] = "caller mutation"
                        state, _ = snapshots.read_snapshot(refs[-1], expected_binding=kernel.binding)
                        self.assertEqual(state["binding"], kernel.binding)
                    self.assertEqual(loader.call_count, 11*(pass_index+1))
            first = snapshots.ROOT/refs[0]["path"]
            write_json(first.parent/"INCOMPLETE.json", {"mutation_between_passes": True})
            with self.assertRaises(ValueError):
                snapshots.read_snapshot(refs[-1], expected_binding=kernel.binding)


if __name__ == "__main__":
    unittest.main()
