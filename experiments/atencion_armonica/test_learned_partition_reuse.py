"""Mechanical guard/import checks; no campaign draws, GPU or training jobs."""
import copy
from contextlib import ExitStack
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from src.atencion_armonica import learned_partition_reuse as r
from src.atencion_armonica import learned_partition_gate as gate
from src.atencion_armonica.learned_partition_model import PartitionCostHead, partition_cost_loss
from src.atencion_armonica.learned_partition_core import ARMS
from src.atencion_armonica.structured_source_artifacts import write_json, write_npz, seal_bundle
from experiments.atencion_armonica.test_learned_partition_core import batch
from src.atencion_armonica import learned_partition_budget as budget
from src.atencion_armonica.learned_partition_training import TrainingKernel
from src.atencion_armonica.learned_partition_snapshots import write_snapshot

BASE = r.p.ROOT/".agent-work/phideus-learned-reader-20260908/reuse_tests"


class GuardTests(unittest.TestCase):
    def test_support_one_to_32_with_padding_to_94(self):
        model = PartitionCostHead("shared_source", 2026090891)
        for support in range(1, 33):
            for width in sorted({support, support+1, 32, 47, 64, 94}):
                w = torch.zeros(1, 1, width, dtype=torch.float32)
                w[0, 0, :support] = 1/support
                b = {"groups": torch.zeros(1, width, 9), "globals": torch.zeros(1, 1, 6),
                    "incidence": w, "group_mask": torch.arange(width)[None, :] < support,
                    "candidate_mask": torch.ones(1, 1, dtype=torch.bool)}
                with torch.no_grad():
                    self.assertTrue(torch.isfinite(model(b)).all())
                self.assertLessEqual(float(abs(w.double().sum()-1)), 2e-7)

    def test_actual_float32_false_rejection_is_not_tolerance_relaxation(self):
        # At least one reduction width reproduces the historical f32 false negative.
        errors = []
        for width in range(34, 95):
            values = torch.zeros(32, 40, width)
            values[24, 0, [0, 6, 7, 9, 17, 19, 21, 23, 24, 26, 28, 31, 32, 33]] = 1/14
            errors.append(abs(float(values.sum(-1)[24, 0])-1))
            self.assertLess(abs(float(values.sum(-1, dtype=torch.float64)[24, 0])-1), 2e-7)
        self.assertGreater(max(errors), 2e-7)
        model = PartitionCostHead("shared_source", 2026090891)
        for variant in ("sum", "nan", "negative", "padding", "mask"):
            b = batch()
            if variant == "sum":
                b["incidence"][0, 0, 0] += 1e-5
            elif variant == "nan":
                b["incidence"][0, 0, 0] = float("nan")
            elif variant == "negative":
                b["incidence"][0, 0, 0] = -.25
                b["incidence"][0, 0, 1] += .5
            elif variant == "padding":
                b["candidate_mask"][0, 0] = False
            else:
                b["group_mask"] = b["group_mask"].float()
            with self.assertRaises(ValueError, msg=variant):
                model(b)

    def test_six_observed_failure_incidence_patterns_without_campaign_access(self):
        # Numeric-only regression fixtures from the declared failed train cohort.
        # No observations, labels, model outputs or candidate quality are loaded.
        patterns = [
            (14, [0, 6, 7, 9, 17, 19, 21, 23, 24, 26, 28, 31, 32, 33], [1]*14),
            (14, [1, 9, 17, 18, 20, 22, 25, 29, 30, 32, 33, 35], [2, 1, 2]+[1]*9),
            (18, [0, 9, 15, 17, 23, 24, 30, 33, 34, 35, 36, 37, 41, 42, 43], [1, 1, 4]+[1]*12),
            (15, [2, 5, 7, 12, 14, 22, 27, 28, 29, 32], [2, 2, 1, 2, 2, 1, 2, 1, 1, 1]),
            (15, [5, 6, 14, 20, 22, 24, 26, 29, 31, 32, 33, 34, 36], [2, 1, 1, 2]+[1]*9),
            (14, [8, 9, 15, 17, 22, 23, 28, 29, 32, 33, 34, 35], [2, 1, 2]+[1]*9)]
        model = PartitionCostHead("shared_source", 2026090891)
        for n, positions, sizes in patterns:
            for width in range(max(positions)+1, 95):
                incidence = torch.zeros(1, 1, width)
                incidence[0, 0, positions] = torch.tensor(np.asarray(sizes, np.float64)/n, dtype=torch.float32)
                b = {"groups": torch.zeros(1, width, 9), "globals": torch.zeros(1, 1, 6),
                    "incidence": incidence, "group_mask": torch.arange(width)[None, :] <= max(positions),
                    "candidate_mask": torch.ones(1, 1, dtype=torch.bool)}
                with torch.no_grad():
                    self.assertTrue(torch.isfinite(model(b)).all())

    def test_forward_gradient_and_update_are_bit_exact_on_common_domain(self):
        def historical_forward(model, b):
            self.assertTrue(torch.allclose(b["incidence"].sum(-1)[b["candidate_mask"]],
                torch.ones_like(b["incidence"].sum(-1)[b["candidate_mask"]]), rtol=0, atol=2e-7))
            first = torch.relu(model.group1(b["groups"]))
            second = torch.relu(model.group2(first))
            aggregate = torch.bmm(b["incidence"], second)
            hidden = torch.relu(model.partition1(torch.cat((aggregate, b["globals"]), dim=-1)))
            return torch.nn.functional.softplus(model.partition2(hidden), beta=1, threshold=20)
        for arm in ARMS:
            old = PartitionCostHead(arm, 2026090891)
            new = copy.deepcopy(old)
            b = batch()
            if arm == "pairs_structure":
                b["groups"] = b["groups"][:, :, :8]
            expected, actual = historical_forward(old, b), new(b)
            self.assertEqual(actual.dtype, torch.float32)
            self.assertTrue(torch.equal(expected, actual))
            target = torch.zeros_like(expected)
            for model, values in ((old, expected), (new, actual)):
                opt = torch.optim.AdamW(model.parameters(), lr=.001, betas=(.9, .999), eps=1e-8,
                                       weight_decay=1e-4, foreach=False, fused=False)
                partition_cost_loss(values, target, b["candidate_mask"]).backward()
                opt.step()
            for a, b_parameter in zip(old.parameters(), new.parameters()):
                self.assertEqual(b_parameter.dtype, torch.float32)
                self.assertEqual(b_parameter.grad.dtype, torch.float32)
                self.assertTrue(torch.equal(a.grad, b_parameter.grad))
                self.assertTrue(torch.equal(a, b_parameter))


class ImportTests(unittest.TestCase):
    def setUp(self):
        BASE.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=BASE)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.old, self.new = {"fixture": "producer"}, {"fixture": "executor"}
        old_path, new_path = self.root/"old_auth.json", self.root/"new_auth.json"
        write_json(old_path, {"common": self.old})
        write_json(new_path, {"common": self.new, "reuse": {"mechanical": True}})
        self.old_ref, self.new_ref = r.p.reference(old_path), r.p.reference(new_path)
        self.files = []
        def bundle(relative, role, refs=None):
            folder = self.root/relative
            folder.mkdir(parents=True)
            binding = {"common": self.old, "authorization": self.old_ref}
            if refs:
                binding["dependencies"] = refs
            if role in r.INDEX_ROLES:
                write_json(folder/"shards.json", refs or [])
            else:
                write_npz(folder/"scientific.npz", values=np.arange(7, dtype=np.float32))
            seal_bundle(folder, role=role, binding=binding, resources={"operation": "MECHANICAL_FIXTURE"})
            ref = r.p.reference(folder/"manifest.json")
            self.files.append(ref)
            return ref
        d1 = bundle("train/data", "learned_observation_shard")
        d2 = bundle("calibration/data", "learned_observation_shard", [d1])
        a1 = bundle("train/aggregate", "learned_observation_split", [d1])
        a2 = bundle("calibration/aggregate", "learned_observation_split", [d2, a1])
        c1 = bundle("train/corpus", "learned_training_corpus", [d1, a1])
        c2 = bundle("calibration/corpus", "learned_training_corpus", [d2, a2])
        normalizer = bundle("normalizers", "learned_train_normalizers", [c1])
        n1 = bundle("train/normalized", "learned_normalized_shard", [c1, normalizer])
        n2 = bundle("calibration/normalized", "learned_normalized_shard", [c2, normalizer])
        self.prepared = {"authorization": self.old_ref, "train": c1, "calibration": c2,
            "normalizers": normalizer, "normalized_train": [n1], "normalized_calibration": [n2]}
        self.counts, count, size = {}, 0, 0
        for ref in self.files:
            path = r.p.verify_reference(ref)
            m = r.p.read_reference(ref)
            self.counts[m["role"]] = self.counts.get(m["role"], 0)+1
            count += len(m["artifacts_sha256"])
            size += sum((path.parent/n).stat().st_size for n in m["artifacts_sha256"])
        self.decl = {"destination": (self.root/"imported").relative_to(r.p.ROOT).as_posix(),
            "expected_bundles": len(self.files), "expected_payloads": count, "expected_payload_bytes": size}
        for context in (patch.object(r, "ROOT", self.root), patch.object(r, "ROLES", self.counts),
                patch.object(r, "origin", return_value=(self.decl, self.prepared, {"common": self.old})),
                patch.object(gate, "verify_authorization", return_value={"common": self.new, "reuse": {}})):
            context.start()
            self.addCleanup(context.stop)

    def test_complete_import_preserves_bytes_and_changes_only_declared_indices(self):
        result = r.import_prepared(self.root/"imported", authorization=self.new_ref)
        prepared = r.verify_import(result, self.new, full_bytes=True)
        self.assertEqual(prepared["authorization"], self.new_ref)
        for ref in self.files:
            old_path = r.p.verify_reference(ref)
            new_path = self.root/"imported"/old_path.relative_to(self.root)
            old, new = json.loads(old_path.read_bytes()), json.loads(new_path.read_bytes())
            if old["role"] not in r.INDEX_ROLES:
                self.assertEqual(old["artifacts_sha256"], new["artifacts_sha256"])
                self.assertNotEqual((old_path.parent/"scientific.npz").stat().st_ino,
                                    (new_path.parent/"scientific.npz").stat().st_ino)
            self.assertEqual(new["binding"]["common"], self.new)
        with self.assertRaises(FileExistsError):
            r.import_prepared(self.root/"imported", authorization=self.new_ref)

    def test_incomplete_declared_roster_and_other_destination_are_rejected(self):
        with self.assertRaises(ValueError):
            r.import_prepared(self.root/"other", authorization=self.new_ref)
        self.decl["expected_bundles"] += 1
        with self.assertRaises(ValueError):
            r.import_prepared(self.root/"imported", authorization=self.new_ref)
        self.assertTrue((self.root/"imported/FAILURE.json").is_file())

    def test_reuse_requires_exact_completed_import_audit(self):
        with self.assertRaises(PermissionError):
            r.verify_completion(None, self.new_ref, self.prepared)
        result = r.import_prepared(self.root/"imported", authorization=self.new_ref)
        prepared = r.verify_import(result, self.new)
        target = {"import": result, "prepared": r.p.reference(self.root/"imported/prepared.json")}
        report = self.root/"mechanical_report.txt"
        report.write_text("Mechanical fixture only; not an actual independent campaign audit.\n")
        audit_path = self.root/"audit.json"
        write_json(audit_path, {"status": "PASS", "scope": "PREPARED_REUSE_COMPLETE", "common": self.new,
            "target": target, "reports": [r.p.reference(report)]})
        audit = r.p.reference(audit_path)
        self.assertEqual(r.verify_completion(audit, self.new_ref, prepared), target)
        changed = copy.deepcopy(prepared)
        changed["normalized_train"] = []
        with self.assertRaises(ValueError):
            r.verify_completion(audit, self.new_ref, changed)

    def test_changed_scientific_copy_is_rejected_without_changing_original(self):
        result = r.import_prepared(self.root/"imported", authorization=self.new_ref)
        copied = self.root/"imported/train/data/scientific.npz"
        copied.write_bytes(b"mechanical corruption fixture")
        with self.assertRaises(ValueError):
            r.verify_import(result, self.new, full_bytes=True)
        # The source stays intact despite corruption of the independent copy.
        original = self.root/"train/data/scientific.npz"
        self.assertNotEqual(original.read_bytes(), copied.read_bytes())

    def test_reuse_authorization_cannot_reach_draws(self):
        from src.atencion_armonica import learned_partition_data as data
        with patch.object(gate, "verify_data_stage", return_value=({"reuse": {}}, set())), \
             patch.object(data, "_draw_scene", side_effect=AssertionError("no draw")) as draw:
            with self.assertRaises(PermissionError):
                data.prepare_shard(self.root/"never", "train", 0, authorization={}, previous={}, earlier_shards=[])
            draw.assert_not_called()
            self.assertFalse((self.root/"never").exists())

    def test_source_symlink_is_rejected_before_importing_it(self):
        source = self.root/"train/data/scientific.npz"
        preserved = source.with_name("preserved.bin")
        source.rename(preserved)
        source.symlink_to(preserved.name)
        with self.assertRaises(ValueError):
            r.import_prepared(self.root/"imported", authorization=self.new_ref)


class SourceAmendmentTests(unittest.TestCase):
    def test_only_declared_source_delta_and_unchanged_protocol_runtime_are_allowed(self):
        old = {k: {"fixture": k} for k in ("plan", "protocol", "runtime", "checkpoints")}
        old.update(source_sha256={**dict.fromkeys(r.EDITED, "old"), "budget.py": "unchanged"},
                   test_sha256={"old_test.py": "old"})
        new = copy.deepcopy(old)
        new["source_sha256"].update(dict.fromkeys(r.EDITED, "new"))
        new["source_sha256"].update(dict.fromkeys(("src/atencion_armonica/learned_partition_reuse.py",
            r.p.AMENDMENT["path"], r.p.REUSE_DECLARATION["path"]), "new"))
        new["test_sha256"]["experiments/atencion_armonica/test_learned_partition_reuse.py"] = "new"
        r.validate_source_change(new, old)
        for field in ("protocol", "runtime", "checkpoints", "plan"):
            changed = copy.deepcopy(new)
            changed[field] = {"changed": True}
            with self.assertRaises(ValueError):
                r.validate_source_change(changed, old)
        changed = copy.deepcopy(new)
        changed["source_sha256"]["budget.py"] = "changed"
        with self.assertRaises(ValueError):
            r.validate_source_change(changed, old)
        changed = copy.deepcopy(new)
        changed["test_sha256"] = {}
        with self.assertRaises(ValueError):
            r.validate_source_change(changed, old)
        changed = copy.deepcopy(new)
        changed["test_sha256"]["old_test.py"] = "silently changed"
        with self.assertRaises(ValueError):
            r.validate_source_change(changed, old)
        changed = copy.deepcopy(new)
        changed["test_sha256"]["extra_test.py"] = "undeclared"
        with self.assertRaises(ValueError):
            r.validate_source_change(changed, old)


class InitialContinuityTests(unittest.TestCase):
    def test_complete_initial_state_preserved_and_later_learning_rejected(self):
        from experiments.atencion_armonica.test_learned_partition_budget import BudgetTests
        from src.atencion_armonica.learned_partition_campaign import resume_state
        BASE.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=BASE) as folder, ExitStack() as stack:
            base = Path(folder)
            root = base/"original"
            root.mkdir()
            staging = base/"supervision"
            staging.mkdir()
            stack.enter_context(patch.object(budget, "STAGING", staging))
            stack.enter_context(patch.object(budget, "REGISTRY", staging/"registry"))
            helper = BudgetTests()
            original_ref, new_ref = {"fixture": "producer"}, {"fixture": "executor"}
            prepared = {"authorization": original_ref, "train": {}, "calibration": {},
                "normalizers": {}, "normalized_train": [], "normalized_calibration": []}
            cell = {"arm": "pairs_structure", "checkpoint_seed": 2026090721, "reader_seed": 2026090891}
            old_auth = {"common": {"fixture": "old"}, "training_device": "cpu"}
            old_binding = {"common": old_auth["common"], **prepared, **cell, "device": "cpu", "count": 4096}
            kernel = TrainingKernel(cell["arm"], cell["reader_seed"], binding=old_binding)
            (root/"snapshots").mkdir()
            snapshot = write_snapshot(root/"snapshots", "initial", kernel)
            write_json(root/"ancestry.json", {"resume": None, "snapshots": [], "calibrations": []})
            request_path = root/"request.json"
            request = {"request_id": "mechanical-original", "operation": "train_cell",
                "output": root.relative_to(r.p.ROOT).as_posix(),
                "arguments": {**prepared, **cell, "resume": None}}
            write_json(request_path, request)
            request_ref = r.p.reference(request_path)
            control = staging/"supervisor-original"
            control.mkdir()
            write_json(control/"request.json", {"reference": request_ref, "request": request})
            spent = 53.077151232995675
            with budget.reserve(request_ref, request, control) as (old_permit, _):
                terminal_ref = helper.terminal(request_ref, request, control, old_permit, spent)
            resume = {"request": request_ref, "terminal": terminal_ref, "snapshot": snapshot}
            new_request = {**request, "request_id": "mechanical-current",
                "output": (base/"current").relative_to(r.p.ROOT).as_posix(),
                "arguments": {**request["arguments"], "resume": resume}}
            current_path = base/"current.json"
            write_json(current_path, new_request)
            current_ref = r.p.reference(current_path)
            current_control = staging/"supervisor-current"
            current_control.mkdir()
            write_json(current_control/"request.json", {"reference": current_ref, "request": new_request})
            new_binding = {**old_binding, "common": {"fixture": "new"}, "authorization": new_ref,
                           "reuse_audit": {"mechanical_audit_fixture": True}}
            self.assertEqual(budget.accounting()[1], spent)
            with patch.object(r, "origin", return_value=({"failed_parent": resume}, prepared, old_auth)), \
                 patch.object(r, "verify_completion", return_value={}) as completion, \
                 budget.reserve(current_ref, new_request, current_control) as (permit, remaining):
                self.assertEqual(remaining, 1200.-spent)
                # The active reservation is real. Never weaken accounting or
                # fabricate its terminal to make the worker bootstrap pass.
                with self.assertRaises(PermissionError):
                    budget.accounting()
                self.assertFalse((current_control/"terminal.json").exists())
                # A real child verifies the live parent permit and crosses the
                # bootstrap boundary; no mocking of reserve/accounting/permit.
                context = dict(staging=str(staging), permit=permit, request=current_ref,
                    resume=resume, prepared=prepared, old_auth=old_auth, binding=new_binding)
                child_code = """
import json,sys
from pathlib import Path
from unittest.mock import patch
from src.atencion_armonica import learned_partition_budget as b, learned_partition_reuse as r
from src.atencion_armonica.learned_partition_campaign import resume_state
c=json.loads(sys.argv[1])
with patch.object(b,'STAGING',Path(c['staging'])), patch.object(b,'REGISTRY',Path(c['staging'])/'registry'), patch.object(r,'origin',return_value=({'failed_parent':c['resume']},c['prepared'],c['old_auth'])), patch.object(r,'verify_completion',return_value={}):
    allowance=b.verify_permit(c['permit'],c['request'])
    state,parents,calibrations=resume_state(c['resume'],c['binding'],None)
    print(json.dumps({'steps':state['steps'],'epoch':state['epoch'],'next_batch':state['next_batch'],'parents':parents,'calibrations':calibrations,'remaining':allowance['remaining_seconds']}))
"""
                result = subprocess.run([sys.executable, "-c", child_code, json.dumps(context)],
                    capture_output=True, text=True, check=True, timeout=20,
                    env=dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1"))
                self.assertEqual(json.loads(result.stdout), dict(steps=0, epoch=0, next_batch=0,
                    parents=[], calibrations=[], remaining=1200.-spent))
                state, inherited, calibrations = resume_state(resume, new_binding, None)
                self.assertEqual(inherited, [])
                self.assertEqual(calibrations, [])
                completion.assert_called_once_with(new_binding["reuse_audit"], new_ref,
                                                    {k: new_binding[k] for k in r.PREPARED_FIELDS})
                restored = TrainingKernel(cell["arm"], cell["reader_seed"], binding=new_binding)
                restored.restore(state)
                def equal(a, b):
                    if isinstance(a, torch.Tensor):
                        self.assertTrue(torch.equal(a, b))
                    elif isinstance(a, np.ndarray):
                        np.testing.assert_array_equal(a, b)
                    elif isinstance(a, dict):
                        self.assertEqual(set(a), set(b))
                        for k in a:
                            equal(a[k], b[k])
                    elif isinstance(a, (tuple, list)):
                        self.assertEqual(len(a), len(b))
                        for x, y in zip(a, b):
                            equal(x, y)
                    else:
                        self.assertEqual(a, b)
                def states_equal_except_binding():
                    a, b = kernel.state(), restored.state()
                    self.assertEqual(a.pop("binding"), old_binding)
                    self.assertEqual(b.pop("binding"), new_binding)
                    equal(a, b)
                states_equal_except_binding()
                b = batch()
                inputs = {k: v.repeat((16,)+(1,)*(v.ndim-1)) for k, v in b.items()}
                inputs["groups"] = inputs["groups"][:, :, :8]
                target = torch.zeros(32, 2, 2)
                for _ in range(3):
                    for candidate in (kernel, restored):
                        candidate.step(inputs, target, candidate.expected_scene_ids())
                    states_equal_except_binding()
                for key, value in (("reader_seed", 2026090892), ("device", "cuda:0")):
                    with self.assertRaises(ValueError):
                        r.initial_continuity(resume, {**new_binding, key: value})
                debit_path = r.p.verify_reference(old_permit)
                preserved = base/"preserved_debit.json"
                debit_path.rename(preserved)
                try:
                    with self.assertRaises(FileNotFoundError):
                        r.initial_continuity(resume, new_binding)
                finally:
                    preserved.rename(debit_path)
                old_terminal = budget.terminal_receipt(terminal_ref, request_ref=request_ref)
                original_debit = r.p.read_reference(old_permit)
                for field, replacement in (("cell", {**cell, "reader_seed": 2026090892}),
                        ("request", current_ref), ("control", current_control.relative_to(r.p.ROOT).as_posix()),
                        ("extra", "undeclared")):
                    debit_path.rename(preserved)
                    try:
                        write_json(debit_path, {**original_debit, field: replacement})
                        mutated_terminal = {**old_terminal, "budget": r.p.reference(debit_path)}
                        with patch.object(budget, "terminal_receipt", return_value=mutated_terminal):
                            with self.assertRaises(ValueError, msg=field):
                                r.initial_continuity(resume, new_binding)
                    finally:
                        debit_path.unlink()  # Only this test-created malformed fixture.
                        preserved.rename(debit_path)
                wrong_location = base/"debit_copy.json"
                write_json(wrong_location, original_debit)
                with patch.object(budget, "terminal_receipt", return_value={
                        **old_terminal, "budget": r.p.reference(wrong_location)}):
                    with self.assertRaises(ValueError):
                        r.initial_continuity(resume, new_binding)
                write_snapshot(root/"snapshots", "learned", kernel, parents=[snapshot])
                with self.assertRaises(ValueError):
                    r.initial_continuity(resume, new_binding)
                helper.terminal(current_ref, new_request, current_control, permit, 7.)
            self.assertEqual(budget.accounting()[1], spent+7.)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
