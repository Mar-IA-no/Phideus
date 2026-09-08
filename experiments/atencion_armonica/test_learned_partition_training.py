"""Short mechanical optimization/recovery checks, not campaign training."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from src.atencion_armonica.learned_partition_readout import choose_costs, input_support, prediction_dependence
from src.atencion_armonica.learned_partition_training import TrainingKernel, collate_inputs, collate_targets, epoch_batches
from src.atencion_armonica.learned_partition_snapshots import ROOT, read_snapshot, write_snapshot
from src.atencion_armonica.partial_compatibility_cache import sha_file

BASE = ROOT/".agent-work/phideus-learned-reader-20260908/tests"


def row(index=0):
    return {"groups": (np.arange(54, dtype=np.float32).reshape(6, 9)+index)/100,
            "globals": np.ones((2, 6), np.float32),
            "incidence": np.array([[.25, .25, .25, .25, 0, 0], [0, 0, 0, 0, .5, .5]], np.float32)}


def advance(kernel):
    ids = kernel.expected_scene_ids()
    batch = collate_inputs([row(int(i)) for i in ids])
    target = collate_targets([np.array([[.4, 0.], [.1, .2]], np.float32) for _ in ids], batch["candidate_mask"])
    return kernel.step(batch, target, ids)


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        BASE.mkdir(parents=True, exist_ok=True)

    def assert_state_equal(self, a, b):
        self.assertEqual(type(a), type(b))
        if isinstance(a, dict):
            self.assertEqual(set(a), set(b))
            for key in a:
                self.assert_state_equal(a[key], b[key])
        elif isinstance(a, (list, tuple)):
            self.assertEqual(len(a), len(b))
            for x, y in zip(a, b):
                self.assert_state_equal(x, y)
        elif isinstance(a, torch.Tensor):
            self.assertTrue(torch.equal(a, b))
        elif isinstance(a, np.ndarray):
            np.testing.assert_array_equal(a, b)
        else:
            self.assertEqual(a, b)

    def test_epoch_order_complete_paired_and_rejected(self):
        a = epoch_batches(2026090891, 0)
        self.assertEqual(a.shape, (128, 32))
        np.testing.assert_array_equal(np.sort(a.ravel()), np.arange(4096))
        np.testing.assert_array_equal(a, epoch_batches(2026090891, 0))
        self.assertFalse(np.array_equal(a, epoch_batches(2026090891, 1)))
        for args in ((0, 0), (2026090891, 50)):
            with self.assertRaises(ValueError):
                epoch_batches(*args)

    def test_continuous_equals_mid_epoch_atomic_resume(self):
        binding = {"namespace": "MECHANICAL_ONLY", "fixture": "analytic-tensor64", "version": 1}
        continuous = TrainingKernel("shared_source", 2026090891, binding=binding, count=64)
        expected_losses = [advance(continuous) for _ in range(4)]
        expected = continuous.state()
        interrupted = TrainingKernel("shared_source", 2026090891, binding=binding, count=64)
        first = advance(interrupted)
        with tempfile.TemporaryDirectory(dir=BASE) as folder:
            ref = write_snapshot(folder, "step_00000001", interrupted)
            saved, m = read_snapshot(ref, expected_binding=binding)
            self.assertEqual(m["position"], {"epoch": 0, "next_batch": 1, "steps": 1})
            # Parent failure does not corrupt an independently sealed snapshot.
            (Path(folder)/"INCOMPLETE.json").write_text('{"status":"mechanical interruption"}')
            resumed = TrainingKernel("shared_source", 2026090891, binding=binding, count=64)
            resumed.restore(saved)
            losses = [first]+[advance(resumed) for _ in range(3)]
            self.assertEqual(losses, expected_losses)
            self.assert_state_equal(resumed.state(), expected)
            with self.assertRaises(FileExistsError):
                write_snapshot(folder, "step_00000001", resumed)
            with self.assertRaises(ValueError):
                read_snapshot(ref, expected_binding={**binding, "version": 2})
            with (ROOT/ref["path"]).with_name("state.pt").open("ab") as handle:
                handle.write(b"tampered")
            with self.assertRaises(ValueError):
                read_snapshot(ref, expected_binding=binding)

    def test_snapshot_binding_and_position_rejected(self):
        kernel = TrainingKernel("shared_source", 2026090891, binding={"fixture": 1}, count=64)
        state = kernel.state()
        for field, value in (("device", "cuda:0"), ("batch_hash", "wrong"), ("steps", 1), ("count", 4096)):
            changed = copy.deepcopy(state)
            changed[field] = value
            with self.assertRaises(ValueError):
                kernel.restore(changed)

    def test_snapshot_rejects_incomplete_extra_and_invalid_states_before_mutation(self):
        kernel = TrainingKernel("shared_source", 2026090891, binding={"fixture": 1}, count=64)
        advance(kernel)
        original = kernel.state()
        bad = []
        bad.append({k: original[k] for k in ("binding", "epoch", "next_batch", "steps")})
        bad.append({**original, "extra": True})
        for mutate in (
            lambda s: s["model"].pop("group1.bias"),
            lambda s: s["model"]["group1.weight"].fill_(float("nan")),
            lambda s: s["optimizer"]["param_groups"][0].update(lr=.2),
            lambda s: s["optimizer"]["state"][0].pop("exp_avg"),
            lambda s: s["optimizer"]["state"][0]["step"].fill_(0),
            lambda s: s.update(torch_rng=torch.zeros(3, dtype=torch.uint8)),
            lambda s: s["accumulator"]["moments"]["targets"].update(count=0),
            lambda s: s.update(binding={"roles": ("train", "calibration")}),
            lambda s: s.update(binding={"nested": {1: "train"}}),
        ):
            changed = copy.deepcopy(original)
            mutate(changed)
            bad.append(changed)
        with tempfile.TemporaryDirectory(dir=BASE) as folder:
            for i, state in enumerate(bad):
                class Payload:
                    def state(self):
                        return state
                with self.assertRaises(ValueError):
                    write_snapshot(folder, f"invalid_{i}", Payload())
                self.assertFalse((Path(folder)/f"invalid_{i}").exists())
                with self.assertRaises(ValueError):
                    kernel.restore(state)
                self.assert_state_equal(kernel.state(), original)

    def test_reader_rejects_rehashed_but_invalid_payload(self):
        kernel = TrainingKernel("shared_source", 2026090891, binding={"fixture": 1}, count=64)
        with tempfile.TemporaryDirectory(dir=BASE) as folder:
            ref = write_snapshot(folder, "initial", kernel)
            manifest_path = ROOT/ref["path"]
            state_path = manifest_path.with_name("state.pt")
            state = kernel.state()
            state.pop("optimizer")
            torch.save(state, state_path)  # Deliberate corruption of this test-owned fixture.
            m = json.loads(manifest_path.read_bytes())
            m["state_sha256"] = sha_file(state_path)
            manifest_path.write_text(json.dumps(m))
            changed = {**ref, "sha256": sha_file(manifest_path)}
            with self.assertRaises(ValueError):
                read_snapshot(changed, expected_binding=kernel.binding)

    def test_reader_rejects_noncanonical_binding_even_with_matching_hashes(self):
        from src.atencion_armonica.partial_compatibility_cache import encoded
        import hashlib
        for binding in ({"roles": ("train", "calibration")}, {"nested": {1: "train"}}):
            kernel = TrainingKernel("shared_source", 2026090891, binding=binding, count=64)
            with tempfile.TemporaryDirectory(dir=BASE) as folder:
                ref = write_snapshot(folder, "initial", kernel)
                manifest_path = ROOT/ref["path"]
                state_path = manifest_path.with_name("state.pt")
                state = kernel.state()
                state["binding"] = binding
                torch.save(state, state_path)
                m = json.loads(manifest_path.read_bytes())
                m["state_sha256"] = sha_file(state_path)
                m["binding_sha256"] = hashlib.sha256(encoded(binding)).hexdigest()
                manifest_path.write_text(json.dumps(m))
                with self.assertRaises(ValueError):
                    read_snapshot({**ref, "sha256": sha_file(manifest_path)}, expected_binding=binding)

    def test_parents_require_intact_monotone_same_cell_chain(self):
        kernel = TrainingKernel("shared_source", 2026090891, binding={"fixture": 1}, count=64)
        with tempfile.TemporaryDirectory(dir=BASE) as folder:
            initial = write_snapshot(folder, "initial", kernel)
            with self.assertRaises(ValueError):
                write_snapshot(folder, "same_step", kernel, parents=[initial])
            foreign = TrainingKernel("local_compatibility", 2026090891, binding=kernel.binding, count=64)
            advance(foreign)
            with self.assertRaises(ValueError):
                write_snapshot(folder, "foreign_arm", foreign, parents=[initial])
            foreign = TrainingKernel("shared_source", 2026090891, binding={"fixture": 2}, count=64)
            advance(foreign)
            with self.assertRaises(ValueError):
                write_snapshot(folder, "foreign_binding", foreign, parents=[initial])
            arbitrary = Path(folder)/"arbitrary.json"
            arbitrary.write_text('{"status":"NOT_A_SNAPSHOT"}')
            arbitrary_ref = {"path": arbitrary.relative_to(ROOT).as_posix(), "sha256": sha_file(arbitrary)}
            advance(kernel)
            with self.assertRaises(ValueError):
                write_snapshot(folder, "arbitrary_parent", kernel, parents=[arbitrary_ref])
            first = write_snapshot(folder, "first", kernel, parents=[initial])
            advance(kernel)
            second = write_snapshot(folder, "second", kernel, parents=[first])
            restored, _ = read_snapshot(second, expected_binding=kernel.binding)
            self.assert_state_equal(restored, kernel.state())
            # A corrupt grandparent invalidates existing descendants and new children.
            with (ROOT/initial["path"]).with_name("state.pt").open("ab") as handle:
                handle.write(b"corrupt ancestor")
            with self.assertRaises(ValueError):
                read_snapshot(second, expected_binding=kernel.binding)
            advance(kernel)
            with self.assertRaises(ValueError):
                write_snapshot(folder, "third", kernel, parents=[second])

    def test_incomplete_update_cannot_be_snapshotted(self):
        kernel = TrainingKernel("shared_source", 2026090891, binding={"fixture": 1}, count=64)
        b = collate_inputs([row() for _ in range(32)])
        target = torch.full((32, 2, 2), float("nan"))
        with self.assertRaises(ValueError):
            kernel.step(b, target, kernel.expected_scene_ids())
        with self.assertRaises(ValueError):
            kernel.state()

    def test_diagnostics_use_real_rows_and_separate_parameter_slices(self):
        kernel = TrainingKernel("pairs_structure", 2026090891, binding={"fixture": 1}, count=32)
        ids = kernel.expected_scene_ids()
        rows = [row(int(i)) for i in ids]
        for r in rows:
            r["groups"] = r["groups"][:, :8]
        b = collate_inputs(rows)
        targets = collate_targets([np.zeros((2, 2), np.float32) for _ in ids], b["candidate_mask"])
        kernel.step(b, targets, ids)
        self.assertEqual(len(kernel.history), 1)
        d = kernel.history[0]
        self.assertEqual(d["moments"]["group_inputs"]["count"], 32*6)
        self.assertEqual(d["activity"]["partition1"]["count"], 32*2)
        self.assertIn("baseline_extra_row", d["gradient_norms"])
        self.assertIn("baseline_extra_column", d["update_norms"])

    def test_collation_rejects_truth_and_variable_shapes(self):
        with self.assertRaises(ValueError):
            collate_inputs([{**row(), "truth": [0]*8}])
        with self.assertRaises(ValueError):
            collate_inputs([row(), {**row(), "groups": np.zeros((6, 8), np.float32)}])
        with self.assertRaises(ValueError):
            collate_targets([np.zeros((3, 2), np.float32)], torch.ones(1, 2, dtype=torch.bool))
        with self.assertRaises(ValueError):
            collate_targets([np.zeros((1, 2), np.float32)], torch.tensor([[False, True]]))

    def test_input_support_not_old_energy_and_not_indistinguishable_groups(self):
        a = {"groups": np.zeros((2, 9), np.float32), "globals": np.zeros((1, 6), np.float32),
             "incidence": np.array([[.5, .5]], np.float32)}
        a["groups"][:, 8] = [0, 1]
        b = copy.deepcopy(a)
        b["groups"][:, 8] = [1, 0]
        support = input_support(a, b)
        self.assertEqual(support["status"], "INPUT_UNCHANGED")
        self.assertTrue(support["aligned_candidate_mask"][0])
        a["groups"][1, 0] = b["groups"][1, 0] = 1
        self.assertEqual(input_support(a, b)["status"], "INPUT_CHANGED")
        self.assertEqual(float(a["groups"][:, 8].mean()), float(b["groups"][:, 8].mean()))

    def test_readout_ties_and_dependence(self):
        candidates = [((0, 1), (2, 3)), ((0,), (1,), (2,), (3,))]
        cost = np.array([[.1, .2], [.2, .1]], np.float32)
        chosen = choose_costs(cost, candidates)
        self.assertEqual(chosen["candidate_index"], 1)
        self.assertEqual(chosen["co_minimum_count"], 2)
        changed = np.array([[0., 0.], [.2, .1]], np.float32)
        d = prediction_dependence(cost, changed, candidates)
        self.assertTrue(d["decision_changed"])
        self.assertFalse(d["constant_sum_delta"])


if __name__ == "__main__":
    unittest.main()
