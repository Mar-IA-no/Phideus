"""Mechanical regression tests; no campaign data, test truth or CUDA execution."""
import ast
from contextlib import ExitStack
import copy
import inspect
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from experiments.atencion_armonica.test_learned_partition_cache import fixture
from src.atencion_armonica import learned_partition_runner as old_runner
from src.atencion_armonica import learned_partition_test as old_test
from src.atencion_armonica import partition_test_recovery as recovery
from src.atencion_armonica import partition_test_recovery_gate as gate
from src.atencion_armonica import partition_test_recovery_supervisor as supervisor
from src.atencion_armonica.structured_source_artifacts import write_json, verify_bundle, write_npz, seal_bundle

BASE = gate.p.ROOT/gate.LOCAL/"recovery_test_fixtures"
DELETIONS = {
    "normalize_shard": ["rows", "inputs"],
    "infer_test": ["rows", "norms", "original", "inputs", "row", "expected", "predictions", "changed", "altered", "support"],
    "evaluate_test": ["raw", "rows", "values", "metrics", "evidence", "interventions", "support", "original", "evaluated", "metric", "reconstructed", "norms", "indices"],
}
ROLE = {"normalize_shard": "learned_normalized_shard", "infer_test": "learned_test_predictions",
        "evaluate_test": "learned_test_evaluation"}


def parsed(fn):
    return ast.parse(inspect.getsource(fn)).body[0]


class DeclaredDelta(ast.NodeTransformer):
    """Undo only the declared interface/provenance delta for a structural check."""
    def __init__(self, name):
        self.name = name

    def visit_FunctionDef(self, node):
        pairs = [(a, default) for a, default in zip(node.args.kwonlyargs, node.args.kw_defaults)
                 if a.arg != "recovery_authorization"]
        node.args.kwonlyargs, node.args.kw_defaults = map(list, zip(*pairs))
        return self.generic_visit(node)

    def visit_Assign(self, node):
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id in {"execution", "ref"}:
            return None
        return self.generic_visit(node)

    def visit_Delete(self, node):
        if [n.id for n in node.targets if isinstance(n, ast.Name)] == DELETIONS[self.name]:
            return None
        return node

    def visit_If(self, node):
        if ast.dump(node.test) == ast.dump(ast.parse('device != "cpu"', mode="eval").body):
            return None
        return self.generic_visit(node)

    def visit_Expr(self, node):
        if isinstance(node.value, ast.Call):
            name = ast.unparse(node.value.func)
            if name in {"compare_failed_iid", "rg.assert_fresh", "rg.checked_bundle", "rg.checked_evaluation", "normalized_shard"}:
                return None
        return self.generic_visit(node)

    def visit_Return(self, node):
        if isinstance(node.value, ast.Name) and node.value.id == "ref":
            return ast.parse('return p.reference(output/"manifest.json")').body[0]
        return self.generic_visit(node)

    def visit_Call(self, node):
        node.keywords = [k for k in node.keywords if k.arg != "recovery_authorization"]
        if isinstance(node.func, ast.Name) and node.func.id == "recovery_identity":
            node.func.id = "_identity"
            node.args[0] = ast.parse('auth["common"]', mode="eval").body
        if isinstance(node.func, ast.Name) and node.func.id == "seal_bundle":
            for key in node.keywords:
                if key.arg == "role":
                    key.value = ast.Constant(ROLE[self.name])
        return self.generic_visit(node)

    def visit_Dict(self, node):
        for i, (key, value) in enumerate(zip(node.keys, node.values)):
            if key is None and isinstance(value, ast.Name) and value.id == "execution":
                node.keys[i] = ast.Constant("common")
                node.values[i] = ast.parse('auth["common"]', mode="eval").body
        return self.generic_visit(node)


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        BASE.mkdir(parents=True, exist_ok=True)

    def test_scientific_traversals_and_check_order_match_original(self):
        for name, original in (("normalize_shard", old_runner.normalize_shard),
                               ("infer_test", old_test.infer_test), ("evaluate_test", old_test.evaluate_test)):
            with self.subTest(entrypoint=name):
                new = parsed(getattr(recovery, name))
                deletes = [n for n in ast.walk(new) if isinstance(n, ast.Delete)
                           and [ast.unparse(t) for t in n.targets] == DELETIONS[name]]
                self.assertEqual(len(deletes), 1)
                body = next(n for n in new.body if isinstance(n, ast.Try)).body
                index = body.index(deletes[0])
                self.assertIsInstance(body[index+1], ast.If)
                self.assertNotIn("cache", DELETIONS[name])
                self.assertNotIn("truths", DELETIONS[name])
                transformed = DeclaredDelta(name).visit(copy.deepcopy(new))
                self.assertEqual(ast.dump(transformed), ast.dump(parsed(original)))

    def test_twelve_packed_files_match_original_producer(self):
        auth = {"common": {"fixture": "mechanical"}, "freeze": {"path": "fixture-freeze", "sha256": "0"*64}}
        cache = SimpleNamespace(scene_ids=list(range(512)), split="ood_beta", shard=0)
        rows = {s: [(None, fixture())]*512 for s in recovery.SEEDS}
        norms = {s: {"mean": np.zeros(5), "scale": np.ones(5)} for s in recovery.SEEDS}
        execution = {"base_common": auth["common"], "execution_contract": "fixture-contract", "recovery_authorization": "fixture-recovery"}
        actual_read = gate.p.read_reference
        def read(ref):
            return {"data_authorization": "fixture-train"} if ref == auth["freeze"] else actual_read(ref)
        def checked(ref, operation, authorization):
            path = gate.p.verify_reference(ref)
            return path.parent, verify_bundle(path.parent, ref["sha256"], role=gate.ROLES[operation])
        with tempfile.TemporaryDirectory(dir=BASE) as folder, ExitStack() as stack:
            stack.enter_context(patch.object(gate.p, "read_reference", side_effect=read))
            stack.enter_context(patch.object(recovery, "recovery_entry", return_value=execution))
            stack.enter_context(patch.object(gate, "checked_bundle", side_effect=checked))
            stack.enter_context(patch.object(gate, "assert_fresh"))
            for module in (old_runner, recovery):
                for name, value in (("stage_inputs", (auth, cache)), ("training_corpus", None),
                                    ("read_normalizers", norms), ("ordered_forward", None), ("scored_shard", rows)):
                    stack.enter_context(patch.object(module, name, return_value=value))
            kwargs = dict(authorization=gate.FROZEN["test_authorization"], data="fixture-data", logits="fixture-logits",
                          scored="fixture-scored", normalizers="fixture-normalizers", train="fixture-train")
            root = Path(folder)
            old_runner.normalize_shard(root/"old", "ood_beta", 0, **kwargs)
            ref = recovery.normalize_shard(root/"new", "ood_beta", 0, **kwargs, recovery_authorization="fixture-recovery")
            self.assertEqual(actual_read(ref)["role"], gate.ROLES["normalized"])
            paths = list((root/"old").glob("seed_*/*.npz"))
            self.assertEqual(len(paths), 12)
            for path in paths:
                recovery.compare_npz_exact(path, root/"new"/path.relative_to(root/"old"))
            with self.assertRaises(FileExistsError):
                recovery.normalize_shard(root/"new", "ood_beta", 0, **kwargs, recovery_authorization="fixture-recovery")

    def test_exact_comparison_rejects_dtype_order_and_inventory(self):
        with tempfile.TemporaryDirectory(dir=BASE) as folder:
            root = Path(folder)
            write_npz(root/"original.npz", values=np.array([1, 2], dtype=np.float32))
            variants = [{"values": np.array([1, 2], dtype=np.float64)},
                        {"values": np.array([2, 1], dtype=np.float32)},
                        {"other": np.array([1, 2], dtype=np.float32)}]
            for i, arrays in enumerate(variants):
                path = root/f"changed{i}.npz"
                write_npz(path, **arrays)
                with self.assertRaises(ValueError):
                    recovery.compare_npz_exact(root/"original.npz", path)

    def test_all_99_inference_passes_match_original_on_mechanical_fixture(self):
        from src.atencion_armonica.learned_partition_model import PartitionCostHead
        from src.atencion_armonica import learned_partition_snapshots as snapshots
        from src.atencion_armonica.learned_partition_inputs import pack_inputs
        row = fixture()
        rows = {s: [(None, row)]*512 for s in recovery.SEEDS}
        norms = {s: {"mean": np.zeros(5), "scale": np.ones(5)} for s in recovery.SEEDS}
        auth = {"common": {"fixture": "mechanical"}, "freeze": "fixture-freeze"}
        execution = {"base_common": auth["common"], "execution_contract": "fixture-contract",
                     "recovery_authorization": "fixture-recovery"}
        with tempfile.TemporaryDirectory(dir=BASE) as folder, ExitStack() as stack:
            root = Path(folder)
            train_auth = root/"train_auth.json"
            write_json(train_auth, {"training_device": "cpu"})
            frozen = {"data_authorization": gate.p.reference(train_auth), "cells": [],
                      "selection": {"selected": {a: {"epoch": 40} for a in recovery.ARMS}}}
            for seed in recovery.SEEDS:
                (root/"normalized"/f"seed_{seed}").mkdir(parents=True)
                for arm in recovery.ARMS:
                    values = [recovery.model_inputs(row, norms[seed], arm)]*512
                    pack_inputs(root/"normalized"/f"seed_{seed}/{arm}.npz", values,
                                scene_ids=list(range(512)), dim=8 if arm == recovery.ARMS[0] else 9)
                    for reader in recovery.READER_SEEDS:
                        cell = root/f"cell_{seed}_{arm}_{reader}"
                        cell.mkdir()
                        write_json(cell/"snapshot.json", {"position": {"epoch": 40, "next_batch": 0, "steps": 5120}})
                        write_json(cell/"chain.json", {"snapshots": [gate.p.reference(cell/"snapshot.json")]})
                        write_json(cell/"manifest.json", {"binding": {"cell": {"arm": arm, "reader_seed": reader}}})
                        frozen["cells"].append({"checkpoint_seed": seed, "reader_seed": reader, "arm": arm,
                                                "result": gate.p.reference(cell/"manifest.json")})
            def state(ref, *, expected_binding):
                model = PartitionCostHead(expected_binding["arm"], expected_binding["reader_seed"])
                return {"model": model.state_dict()}, {}
            def inputs(*args, **kwargs):
                return auth, frozen, SimpleNamespace(), None, rows, root/"normalized", norms
            def checked(ref, operation, authorization):
                path = gate.p.verify_reference(ref)
                return path.parent, verify_bundle(path.parent, ref["sha256"], role=gate.ROLES[operation])
            stack.enter_context(patch.object(snapshots, "read_snapshot", side_effect=state))
            stack.enter_context(patch.object(old_test, "test_inputs", side_effect=inputs))
            stack.enter_context(patch.object(recovery, "test_inputs", side_effect=inputs))
            stack.enter_context(patch.object(recovery, "recovery_entry", return_value=execution))
            stack.enter_context(patch.object(gate, "assert_fresh"))
            stack.enter_context(patch.object(gate, "checked_bundle", side_effect=checked))
            kwargs = dict(authorization="fixture-auth", data="fixture-data", logits="fixture-logits",
                          scored="fixture-scored", normalized="fixture-normalized", gpu_grant=None)
            before = old_test.infer_test(root/"original", "ood_beta", **kwargs)
            after = recovery.infer_test(root/"versioned", "ood_beta", **kwargs, recovery_authorization="fixture-recovery")
            old = gate.p.read_reference(before)
            new = gate.p.read_reference(after)
            self.assertEqual(old["artifacts_sha256"], new["artifacts_sha256"])
            self.assertEqual(sum(name.endswith(".npz") for name in new["artifacts_sha256"]), 99)

    def test_closed_contract_rejects_source_base_role_and_omission(self):
        expected = gate.contract_candidate()
        for key in ("executor_sources", "base_common", "roles", "plan", "limits", "frozen"):
            with self.subTest(field=key), tempfile.TemporaryDirectory(dir=BASE) as folder:
                value = copy.deepcopy(expected)
                value[key] = {}
                path = Path(folder)/"contract.json"
                write_json(path, value)
                with self.assertRaises(PermissionError):
                    gate.verify_contract(gate.p.reference(path))

    def test_missing_audit_and_foreign_authorization_are_rejected(self):
        with tempfile.TemporaryDirectory(dir=BASE) as folder:
            path = Path(folder)/"audit.json"
            write_json(path, {"status": "PASS", "scope": "TEST_MEMORY_RECOVERY_IMPLEMENTATION", "contract": "x",
                              "reports": [], "resolved_plan_findings": ["R673-F1", "R673-F2"]})
            with self.assertRaises(PermissionError):
                gate.verify_implementation_audit(gate.p.reference(path), "x")
            path = Path(folder)/"authorization.json"
            write_json(path, {"status": "TEST_READY", "common": {}, "freeze": {}, "freeze_audit": {}})
            with self.assertRaises(PermissionError):
                gate.verify_authorization(gate.p.reference(path))

    def test_request_and_output_symlink_ancestors_rejected_before_writes(self):
        with tempfile.TemporaryDirectory(dir=BASE) as folder, ExitStack() as stack:
            root = Path(folder)
            tree = root/"versioned"
            outside = root/"outside"
            outside.mkdir()
            (tree/"outputs"/"iid").mkdir(parents=True)
            (tree/"requests").mkdir()
            recovery_path = tree.relative_to(gate.p.ROOT).as_posix()
            stack.enter_context(patch.object(gate, "RECOVERY", recovery_path))
            stack.enter_context(patch.object(gate, "verify_authorization", return_value=({"contract": "fixture"}, {})))
            args = dict(split="iid", shard=0, authorization=gate.FROZEN["test_authorization"],
                data="fixture", logits="fixture", scored="fixture", normalizers="fixture",
                train="fixture", recovery_authorization="fixture")
            base = dict(schema="partition-test-memory-recovery-request-v1", request_id="fixture",
                operation="normalized", arguments=args, execution_contract="fixture")
            link = tree/"outputs"/"iid"/"escape"
            link.symlink_to(outside, target_is_directory=True)
            request = {**base, "output": (link/"new").relative_to(gate.p.ROOT).as_posix()}
            path = tree/"requests"/"output_escape.json"
            write_json(path, request)
            with self.assertRaises(ValueError):
                supervisor.validate_request(gate.p.reference(path))
            self.assertEqual(list(outside.iterdir()), [])
            request_link = tree/"requests"/"escape"
            request_link.symlink_to(outside, target_is_directory=True)
            request = {**base, "output": (tree/"outputs"/"iid"/"new").relative_to(gate.p.ROOT).as_posix()}
            outside_request = outside/"request.json"
            write_json(outside_request, request)
            ref = {"path": (request_link/"request.json").relative_to(gate.p.ROOT).as_posix(),
                   "sha256": gate.sha_file(outside_request)}
            with self.assertRaises(ValueError):
                supervisor.validate_request(ref)
            self.assertFalse((tree/"outputs"/"iid"/"new").exists())
            self.assertEqual(list(outside.iterdir()), [outside_request])
            valid = tree/"outputs"/"iid"/"new"
            self.assertEqual(gate.canonical_path(valid, tree/"outputs"/"iid"), valid)

    def test_closed_evaluation_binding_inventory_and_mandatory_replay(self):
        execution = dict(base_common={"fixture": True}, execution_contract="fixture-contract",
                         recovery_authorization="fixture-recovery")
        args = dict(authorization=gate.FROZEN["test_authorization"], data="fixture-data",
            logits="fixture-logits", scored="fixture-scored", normalized="fixture-normalized",
            predictions="fixture-predictions", recovery_authorization="fixture-recovery")
        binding = {**execution, **args, "split": "iid", "freeze": gate.FROZEN["freeze"]}
        with tempfile.TemporaryDirectory(dir=BASE) as folder, patch.object(gate, "execution_binding", return_value=execution):
            root = Path(folder)
            def bundle(name, *, identity=None, files=None, changed=False):
                output = root/name
                output.mkdir()
                for file in gate.EVALUATION_FILES if files is None else files:
                    if file.endswith(".npz"):
                        write_npz(output/file, indices=np.arange(4))
                    else:
                        write_json(output/file, {"fixture": 2 if changed and file == "metrics.json" else 1})
                seal_bundle(output, role=gate.ROLES["evaluation"],
                    binding=binding if identity is None else identity, resources={"seconds": len(name)})
                return gate.p.reference(output/"manifest.json")
            original, replay = bundle("original"), bundle("replay")
            gate.checked_evaluation(original, "iid", **args)
            gate.compare_evaluation_replay(original, replay, "iid", **args)
            variants = [bundle("wrong_split", identity={**binding, "split": "wrong"}),
                bundle("wrong_source", identity={**binding, "predictions": "other"}),
                bundle("missing", files=gate.EVALUATION_FILES-{"summary.json"}),
                bundle("extra", files=gate.EVALUATION_FILES|{"junk.json"}),
                bundle("junk", files={"junk.json"})]
            for ref in variants:
                with self.assertRaises(ValueError):
                    gate.checked_evaluation(ref, "iid", **args)
            altered = bundle("altered", changed=True)
            with self.assertRaises(ValueError):
                gate.compare_evaluation_replay(original, altered, "iid", **args)
            from experiments.atencion_armonica import operate_partition_test_recovery as operator
            actual_read, actual_reference = gate.p.read_reference, gate.p.reference
            frozen = {"train_data": "fixture", "calibration_data": "fixture",
                      "normalizers": "fixture", "train": "fixture"}
            def read(ref):
                return frozen if ref == gate.FROZEN["freeze"] else actual_read(ref)
            def reference(path):
                return "fixture-grant" if Path(path).name == "gpu_grant.json" else actual_reference(path)
            def step(name, operation, output, arguments, **kwargs):
                if operation == "evaluation":
                    return {"result": altered if name.endswith("replay") else original}
                return {"result": {"prepare": args["data"], "aggregate_data": "fixture-aggregate",
                    "forward": args["logits"], "score": args["scored"], "normalized": args["normalized"],
                    "inference": args["predictions"]}[operation]}
            with ExitStack() as stack:
                stack.enter_context(patch.object(gate, "verify_authorization", return_value=({"contract": "fixture-contract"}, {})))
                stack.enter_context(patch.object(gate.p, "read_reference", side_effect=read))
                stack.enter_context(patch.object(gate.p, "reference", side_effect=reference))
                stack.enter_context(patch.object(operator, "step", side_effect=step))
                stack.enter_context(patch.object(operator, "verify_normalization_audit"))
                writer = stack.enter_context(patch.object(operator, "write_json"))
                with self.assertRaisesRegex(ValueError, "not byte-exact"):
                    operator.run("tests", "fixture-recovery", "fixture-audit")
                writer.assert_not_called()

    def test_real_owned_child_success_failure_timeout_and_final_hwm(self):
        code = ('import json,os,sys; from src.atencion_armonica.learned_partition_supervisor import arm_parent_death; '
                'p=json.load(os.fdopen(int(sys.argv[1]))); arm_parent_death(p["supervisor_pid"]); ')
        cases = [
            (code+'print(json.dumps({"result": "fixture", "peak_rss_bytes": 1}))', 10, 1024**3, True),
            (code+'raise SystemExit(7)', 10, 1024**3, False),
            (code+'import time; time.sleep(10)', .05, 1024**3, False),
            (code+'print(json.dumps({"result": "fixture", "peak_rss_bytes": 2*1024**3}))', 10, 1024**3, False),
            (code+'print(json.dumps({"result": "fixture"}))', 10, 1024**3, False),
        ]
        for source, wall, cap, success in cases:
            with self.subTest(success=success, wall=wall), tempfile.TemporaryDirectory(dir=BASE) as folder:
                result = supervisor.supervise_process([sys.executable, "-c", source],
                    env=dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1"),
                    control=Path(folder), wall=wall, rss_limit=cap, payload={})
                self.assertEqual(result["error"] is None, success)
                self.assertTrue(result["worker_terminal_confirmed"])
                self.assertIsNotNone(result["worker_exit_code"])
                self.assertFalse(Path(f'/proc/{result["worker_pid"]}').exists())


if __name__ == "__main__":
    unittest.main()
