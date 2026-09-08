"""Support-only numerical regression, closed authority and owned-child checks."""
import ast
from contextlib import ExitStack
import copy
import inspect
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from src.atencion_armonica import learned_partition_readout as old_guard
from src.atencion_armonica import learned_partition_inference as old_inference
from src.atencion_armonica import learned_partition_test as old_test
from src.atencion_armonica import partition_test_recovery as old
from src.atencion_armonica import partition_support_guard as guard
from src.atencion_armonica import partition_support_recovery as recovery
from experiments.atencion_armonica import run_partition_support_recovery as operator
from src.atencion_armonica.structured_source_artifacts import write_json, write_npz, seal_bundle

BASE = recovery.p.ROOT/recovery.LOCAL/"support_test_fixtures"


def parsed(fn):
    return ast.parse(inspect.getsource(fn)).body[0]


class InterfaceDelta(ast.NodeTransformer):
    """Undo only support authority/import deltas; retain the full scientific AST."""
    def visit_FunctionDef(self, node):
        pairs = [(a, d) for a, d in zip(node.args.kwonlyargs, node.args.kw_defaults)
                 if a.arg != "support_recovery_authorization"]
        node.args.kwonlyargs, node.args.kw_defaults = map(list, zip(*pairs))
        return self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if node.module == "partition_support_guard":
            return None
        if node.module == "learned_partition_inference":
            node.names.append(ast.alias(name="intervention_report"))
        return node

    def visit_Expr(self, node):
        if isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == "compare_failed_polyphony":
            return None
        return self.generic_visit(node)

    def visit_Call(self, node):
        node.keywords = [k for k in node.keywords if k.arg != "support_recovery_authorization"]
        name = ast.unparse(node.func)
        if name == "recovery_entry":
            node.args = node.args[:2]
        if name in {"assert_fresh", "checked_bundle", "checked_evaluation"}:
            node.func = ast.Attribute(value=ast.Name(id="rg", ctx=ast.Load()), attr=name, ctx=ast.Load())
            node.args = [ast.Name(id="recovery_authorization", ctx=ast.Load())
                         if isinstance(n, ast.Name) and n.id == "support_recovery_authorization" else n for n in node.args]
        return self.generic_visit(node)

    def visit_Name(self, node):
        if node.id == "ROLES":
            return ast.Attribute(value=ast.Name(id="rg", ctx=ast.Load()), attr="ROLES", ctx=ast.Load())
        return node


class SupportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        BASE.mkdir(parents=True, exist_ok=True)

    def test_only_mass_accumulation_changes_in_guard(self):
        tree = parsed(guard.input_support)
        hits = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and ast.unparse(node.func) == "incidence.sum":
                self.assertEqual([k.arg for k in node.keywords], ["axis", "dtype"])
                self.assertEqual(ast.unparse(node.keywords[1].value), "np.float64")
                node.keywords.pop()
                hits += 1
        self.assertEqual(hits, 1)
        self.assertEqual(ast.dump(tree), ast.dump(parsed(old_guard.input_support)))
        for name in ("support_by_size", "intervention_report"):
            self.assertEqual(ast.dump(parsed(getattr(guard, name))), ast.dump(parsed(getattr(old_inference, name))))
        tree = parsed(old_test.replay_support)
        tree.body = [n for n in tree.body if not isinstance(n, ast.ImportFrom)]
        self.assertEqual(ast.dump(tree), ast.dump(parsed(guard.replay_support)))

    def test_scientific_traversals_exact_modulo_declared_delta(self):
        for name in ("infer_test", "preserved_test", "evaluate_test"):
            with self.subTest(name=name):
                tree = InterfaceDelta().visit(parsed(getattr(recovery, name)))
                self.assertEqual(ast.dump(tree), ast.dump(parsed(getattr(old, name))))

    def inputs(self, n=26, width=94, strided=False):
        array = np.zeros((2, width), dtype=np.float32)
        array[:, :n] = np.float32(1/n)
        if strided:
            storage = np.zeros((2, width*2), dtype=np.float32)
            storage[:, ::2] = array
            array = storage[:, ::2]
        return {"groups": np.zeros((width, 9), dtype=np.float32),
                "globals": np.zeros((2, 3), dtype=np.float32), "incidence": array}

    def test_cardinalities_padding_layout_and_old_valid_outputs(self):
        rejected = 0
        for n in range(1, 33):
            for width in sorted({n, 32, 63, 94}):
                for strided in (False, True):
                    a = self.inputs(n, width, strided)
                    b = {k: v.copy() for k, v in a.items()}
                    b["groups"][:n, 8] = 1
                    result = guard.input_support(a, b)
                    self.assertEqual(result["status"], "INPUT_CHANGED")
                    self.assertEqual(guard.input_support(a, a)["status"], "INPUT_UNCHANGED")
                    try:
                        previous = old_guard.input_support(a, b)
                    except ValueError:
                        rejected += 1
                    else:
                        self.assertEqual(previous, result)
        self.assertGreater(rejected, 0)

    def test_invalid_common_inputs_still_rejected(self):
        a = self.inputs()
        for field, index, value in (("groups", (0, 0), 1), ("globals", (0, 0), 1),
                                   ("incidence", (0, 0), 0), ("groups", (0, 8), np.nan),
                                   ("globals", (0, 0), np.inf)):
            b = {k: v.copy() for k, v in a.items()}
            b[field][index] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                guard.input_support(a, b)
        for value in (-.1, 1.1, .5, np.nan, np.inf):
            b = {k: v.copy() for k, v in a.items()}
            b["incidence"][0, 0] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                guard.input_support(b, b)

    def test_plan_receipt_is_exact_and_closed(self):
        valid = recovery.p.read_reference(recovery.PLAN_AUDIT)
        real_read = recovery.p.read_reference
        for key, value in (("scope", "WRONG"), ("status", "NEEDS_REVISION"),
                           ("target", recovery.OLD_CONTRACT), ("reports", []), ("resolved_findings", [])):
            altered = {**valid, key: value}
            with patch.object(recovery.p, "read_reference", side_effect=lambda r: altered if r == recovery.PLAN_AUDIT else real_read(r)):
                with self.assertRaises(PermissionError):
                    recovery.verify_plan_audit()

    def test_symlink_confinement_and_overwrite(self):
        with tempfile.TemporaryDirectory(dir=BASE) as folder:
            root = Path(folder)
            (root/"real").mkdir()
            (root/"link").symlink_to(root/"real", target_is_directory=True)
            with self.assertRaises(ValueError):
                recovery.canonical_path(root/"link"/"child.json", root)
            output = root/"record.json"
            write_json(output, {"preserved": True})
            with self.assertRaises(FileExistsError):
                write_json(output, {"preserved": False})
            with self.assertRaises(ValueError):
                recovery.evidence_only({"path": (root/"link"/"missing").relative_to(recovery.p.ROOT).as_posix(), "sha256": "0"*64})

    def failed_fixture(self, root):
        failed = root/"failed"
        failed.mkdir()
        rel = failed.relative_to(recovery.p.ROOT).as_posix()
        files = [f"seed_{s}/pairs_structure_{r}_original.npz" for s in recovery.SEEDS for r in recovery.READER_SEEDS]
        files += [f"seed_{recovery.SEEDS[0]}/local_compatibility_{recovery.READER_SEEDS[0]}_{k}.npz" for k in ("original", "zero")]
        for name in files:
            (failed/name).parent.mkdir(exist_ok=True)
            write_npz(failed/name, costs=np.ones((1, 2), dtype=np.float32))
        write_json(failed/"FAILURE.json", {"status": "INCOMPLETE", "error": "ValueError('support comparison changes the common input or incidence')"})
        write_json(root/"normalized.json", {"fixture": True})
        normal = recovery.p.reference(root/"normalized.json")
        write_json(root/"request.json", {"output": rel, "execution_contract": recovery.OLD_CONTRACT,
            "arguments": {"recovery_authorization": recovery.OLD_AUTH, "normalized": normal}})
        request = recovery.p.reference(root/"request.json")
        terminal = {"status": "FAILED", "request": request, "worker_terminal_confirmed": True,
            "worker_exit_code": 1, "result": None, "output": rel, "operation": "inference"}
        write_json(root/"terminal.json", terminal)
        termref = recovery.p.reference(root/"terminal.json")
        inventory = [recovery.p.reference(failed/name) for name in [*files, "FAILURE.json"]]
        write_json(root/"diagnostic.json", {"schema": "support-reduction-diagnostic-v1",
            "status": "DIAGNOSTIC_NOT_CAMPAIGN", "request": request, "failed_terminal": termref,
            "normalized": normal, "source": normal, "results": [], "failed_artifact_inventory": inventory})
        return {"FAILED_ROOT": rel, "FAILED_REQUEST": request, "FAILED_TERMINAL": termref,
            "FAILED_MARKER": recovery.p.reference(failed/"FAILURE.json"),
            "DIAGNOSTIC": recovery.p.reference(root/"diagnostic.json")}, failed, terminal

    def test_failed_evidence_is_never_complete_authority(self):
        with tempfile.TemporaryDirectory(dir=BASE) as folder, ExitStack() as stack:
            root = Path(folder)
            constants, failed, terminal = self.failed_fixture(root)
            for key, value in constants.items():
                stack.enter_context(patch.object(recovery, key, value))
            # Fixture anchor substitutes only the operational tree; no live data are edited.
            stack.enter_context(patch.object(recovery.gate02, "RECOVERY", root.relative_to(recovery.p.ROOT).as_posix()))
            actual_canonical = recovery.canonical_path
            stack.enter_context(patch.object(recovery, "canonical_path", side_effect=lambda path, anchor:
                actual_canonical(path, root if str(path) == constants["FAILED_ROOT"] else anchor)))
            self.assertEqual(len(recovery.failed_evidence()), 11)
            write_json(failed/"manifest.json", {"fake": "COMPLETE"})
            with self.assertRaises(ValueError):
                recovery.failed_evidence()

    def test_evidence_hash_mutation_rejected(self):
        with tempfile.TemporaryDirectory(dir=BASE) as folder:
            path = Path(folder)/"bytes.json"
            write_json(path, {"immutable": 1})
            ref = recovery.p.reference(path)
            with self.assertRaises(ValueError):
                recovery.evidence_only({**ref, "sha256": "0"*64})

    def test_request_rejects_wrong_scope_and_cpu_gpu_grant(self):
        base = {"schema": "partition-support-recovery-request-v1", "request_id": "fixture",
                "operation": "inference", "output": "unused", "execution_contract": "fixture"}
        arguments = {k: None for k in operator.ARGUMENTS["inference"]}
        arguments.update(split="ood_polyphony", authorization=recovery.FROZEN["test_authorization"],
                         recovery_authorization=recovery.OLD_AUTH)
        for update in ({"split": "iid"}, {"gpu_grant": "cuda"}, {"recovery_authorization": "other"}):
            record = {**base, "arguments": {**arguments, **update}}
            with patch.object(recovery.p, "read_reference", return_value=record), self.assertRaises(PermissionError):
                operator.validate_request({"path": "fixture", "sha256": "0"*64})

    def test_replay_payload_divergence_rejected(self):
        with patch.object(recovery, "checked_evaluation", side_effect=[
                (None, {"artifacts_sha256": {"metrics.json": "a"}}),
                (None, {"artifacts_sha256": {"metrics.json": "b"}})]):
            with self.assertRaises(ValueError):
                recovery.compare_evaluation_replay("first", "second", "ood_polyphony")

    def test_evaluation_rejects_mixed_binding_and_payload_roster(self):
        kwargs = dict(authorization=recovery.FROZEN["test_authorization"], data="data", logits="logits",
            scored="scored", normalized="normalized02", predictions="predictions-new",
            recovery_authorization=recovery.OLD_AUTH, support_recovery_authorization="support-auth")
        execution = {"base_common": "base", "execution_contract": recovery.OLD_CONTRACT,
            "recovery_authorization": recovery.OLD_AUTH, "support_execution_contract": "support-contract",
            "support_recovery_authorization": "support-auth"}
        binding = {**execution, "split": "ood_polyphony", "freeze": recovery.FROZEN["freeze"],
                   **{k: v for k, v in kwargs.items() if k not in execution}}
        valid = {"binding": binding, "artifacts_sha256": {n: "digest" for n in recovery.EVALUATION_FILES}}
        with patch.object(recovery, "recovery_entry", return_value=execution):
            for manifest in ({**valid, "binding": {**binding, "normalized": "retagged"}},
                             {**valid, "binding": {**binding, "support_execution_contract": recovery.OLD_CONTRACT}},
                             {**valid, "artifacts_sha256": {"metrics.json": "digest"}}):
                with patch.object(recovery, "checked_bundle", return_value=(None, manifest)), self.assertRaises(ValueError):
                    recovery.checked_evaluation("manifest", "ood_polyphony", **kwargs)
            with patch.object(recovery, "checked_bundle", return_value=(None, valid)):
                recovery.checked_evaluation("manifest", "ood_polyphony", **kwargs)

    def test_implementation_audit_targets_exact_new_contract(self):
        record = {"status": "PASS", "scope": "SUPPORT_VALIDATION_RECOVERY_IMPLEMENTATION",
            "contract": "new-contract", "reports": [recovery.PLAN_REPORTS[1]],
            "resolved_plan_findings": ["R680-F1", "R680-F2"]}
        for key, value in (("contract", recovery.OLD_CONTRACT), ("status", "NEEDS_REVISION"),
                           ("scope", "TEST_MEMORY_RECOVERY_IMPLEMENTATION"), ("reports", [])):
            with patch.object(recovery.p, "read_reference", return_value={**record, key: value}), self.assertRaises(PermissionError):
                recovery.verify_implementation_audit("fixture", "new-contract")

    def test_contract_rejects_changed_source_or_plan_inventory(self):
        record = recovery.contract_candidate()
        for key, value in (("executor_sources", {}), ("plan", recovery.OLD_CONTRACT),
                           ("old_authorization", "changed"), ("roles", recovery.gate02.ROLES)):
            with patch.object(recovery.p, "read_reference", return_value={**record, key: value}), self.assertRaises(PermissionError):
                recovery.verify_contract({"path": "fixture", "sha256": "0"*64})

    def test_real_child_success_fail_timeout_and_cap(self):
        code = ('import json,os,sys,resource,time; '
                'json.load(os.fdopen(int(sys.argv[-1]))); '
                'mode=sys.argv[1]; '
                'time.sleep(1 if mode=="timeout" else 0); '
                'sys.exit(1) if mode=="fail" else None; '
                'print(json.dumps({"result":"fixture", "peak_rss_bytes": '
                'resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}))')
        for mode in ("success", "fail", "timeout", "cap"):
            with tempfile.TemporaryDirectory(dir=BASE) as folder:
                result = operator.supervise_process([sys.executable, "-c", code, mode],
                    env=dict(os.environ, CUDA_VISIBLE_DEVICES=""), control=Path(folder),
                    # Fork/exec can inherit the parent's pre-exec RSS high-water mark.
                    wall=.05 if mode == "timeout" else 5, rss_limit=1 if mode == "cap" else 2*1024**3, payload={})
                self.assertTrue(result["worker_terminal_confirmed"])
                if mode == "success":
                    self.assertIsNone(result["error"])
                    self.assertEqual(result["worker_exit_code"], 0)
                else:
                    self.assertIsNotNone(result["error"])


if __name__ == "__main__":
    unittest.main()
