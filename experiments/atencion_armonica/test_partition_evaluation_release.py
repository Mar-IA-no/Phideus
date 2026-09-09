"""Evaluation lifetime/identity fixtures; no campaign producers or model execution."""
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

from experiments.atencion_armonica.test_learned_partition_cache import fixture
from src.atencion_armonica import partition_evaluation_release as release
from src.atencion_armonica import partition_support_recovery as old
from experiments.atencion_armonica import run_partition_evaluation_release as operator
from src.atencion_armonica.structured_source_artifacts import write_json, write_npz, seal_bundle
from src.atencion_armonica.structured_source_reader import partition_energies
from src.atencion_armonica.partition_support_guard import replay_support

BASE = release.p.ROOT/release.LOCAL/"evaluation_release_tests"
EARLY = ["raw", "metrics", "evidence", "interventions", "original", "evaluated", "metric", "indices"]
LATE = ["rows", "values", "support", "reconstructed", "norms"]
OLD_DEL = ["raw", "rows", "values", "metrics", "evidence", "interventions", "support",
           "original", "evaluated", "metric", "reconstructed", "norms", "indices"]


class DeclaredDelta(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        pairs = [(a, d) for a, d in zip(node.args.kwonlyargs, node.args.kw_defaults)
                 if a.arg != "evaluation_release_authorization"]
        node.args.kwonlyargs, node.args.kw_defaults = map(list, zip(*pairs))
        return self.generic_visit(node)

    def visit_Delete(self, node):
        names = [ast.unparse(n) for n in node.targets]
        if names == EARLY:
            return None
        if names == LATE:
            return ast.parse("del "+", ".join(OLD_DEL)).body[0]
        return node

    def visit_Expr(self, node):
        if isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == "compare_failed":
            return None
        return self.generic_visit(node)

    def visit_Call(self, node):
        node.keywords = [k for k in node.keywords if k.arg != "evaluation_release_authorization"]
        if ast.unparse(node.func) == "release_entry":
            node.func.id = "recovery_entry"
            node.args = node.args[:3]
        if ast.unparse(node.func) == "assert_fresh":
            node.args[0].id = "support_recovery_authorization"
        return self.generic_visit(node)


class ReleaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        BASE.mkdir(parents=True, exist_ok=True)

    def test_exact_scientific_ast_and_lifetime_boundary(self):
        new = ast.parse(inspect.getsource(release.evaluate_test)).body[0]
        old_tree = ast.parse(inspect.getsource(old.evaluate_test)).body[0]
        deletes = [n for n in ast.walk(new) if isinstance(n, ast.Delete)]
        self.assertEqual([[ast.unparse(t) for t in n.targets] for n in deletes], [EARLY, LATE])
        body = next(n for n in new.body if isinstance(n, ast.Try)).body
        index = body.index(deletes[0])
        self.assertIn("intervention_summary.json", [n.value for n in ast.walk(body[index-1])
            if isinstance(n, ast.Constant)])
        self.assertEqual(ast.unparse(body[index+1].targets[0]), "norms")
        self.assertEqual(ast.dump(DeclaredDelta().visit(copy.deepcopy(new))), ast.dump(old_tree))

    def test_eight_payloads_equal_and_dead_locals_absent_before_support(self):
        # One numerical scene, three checkpoints/readers. Aggregators are reduced
        # fixture summaries; scalar partition metrics and support run unchanged.
        row = fixture()
        n = row.n
        a = np.arange(n*n, dtype=np.float32).reshape(n, n)/10
        z = (a+a.T).astype(np.float32)
        scored = {"pool": {"canonical_to_observed": list(range(n))},
            "candidates": partition_energies(z.astype(np.float64), row.candidates, row.groups, row.costs)}
        rows = {s: [(scored, row)] for s in release.SEEDS}
        raw = {s: [z] for s in release.SEEDS}
        values = {tuple(e[k] for k in ("arm", "checkpoint_seed", "reader_seed", "intervention")):
            [np.array([[.4, .1], [.1, .2]], np.float32)] for e in release.inference_roster()}
        norms = {s: {"mean": np.zeros(5), "scale": np.ones(5)} for s in release.SEEDS}
        truths = [{"source_ids": [0]*4+[1]*4}]
        frozen = {"normalizers": "fixture-norm", "data_authorization": "fixture-data-auth", "train": "fixture-train"}
        auth = {"common": {"fixture": True}, "freeze": release.FROZEN["freeze"]}
        cache = object()
        checkpoints = []
        with tempfile.TemporaryDirectory(dir=BASE) as folder, ExitStack() as stack:
            root = Path(folder)
            prediction_root = root/"predictions"
            prediction_root.mkdir()
            for entry in release.inference_roster():
                arm, seed, reader, kind = (entry[k] for k in ("arm", "checkpoint_seed", "reader_seed", "intervention"))
                if kind != "original":
                    path = prediction_root/(release.prefix(entry)+"_support.json")
                    path.parent.mkdir(exist_ok=True)
                    report = replay_support([row], norms[seed], arm, kind,
                        values[arm, seed, reader, "original"], values[arm, seed, reader, kind])
                    write_json(path, report)
            def read_normalizers(*args, **kwargs):
                frame = inspect.currentframe().f_back
                try:
                    if frame.f_globals["__name__"] == release.__name__ and frame.f_code.co_name == "evaluate_test":
                        self.assertFalse(set(EARLY) & set(frame.f_locals))
                        self.assertTrue({"rows", "values", "cache", "truths", "prediction_root"} <= set(frame.f_locals))
                        checkpoints.append("released")
                finally:
                    del frame
                return norms
            stack.enter_context(patch.object(release.runner, "read_normalizers", new=read_normalizers))
            for module in (old, release):
                stack.enter_context(patch.object(module, "preserved_test",
                    return_value=(auth, frozen, cache, raw, rows, values, prediction_root)))
                stack.enter_context(patch.object(module, "load_supervision", return_value=truths))
                stack.enter_context(patch.object(module, "assert_fresh"))
                stack.enter_context(patch.object(module, "checked_evaluation"))
                stack.enter_context(patch.object(module, "summarize_test", side_effect=lambda records, **kw:
                    {"count": len(records), "first": records[0], "split": kw["split"]}))
                stack.enter_context(patch.object(module, "summarize_interventions", side_effect=lambda records, **kw:
                    {"count": len(records), "first": records[0], "split": kw["split"]}))
                stack.enter_context(patch.object(module, "summarize_support", side_effect=lambda records:
                    {"count": len(records), "first": records[0]}))
            stack.enter_context(patch.object(old, "recovery_entry", return_value={"fixture": "old"}))
            stack.enter_context(patch.object(release, "release_entry", return_value={"fixture": "new"}))
            stack.enter_context(patch.object(release, "compare_failed"))
            kwargs = dict(authorization=release.FROZEN["test_authorization"], data="fixture-data", logits="fixture-logits",
                scored="fixture-scored", normalized="fixture-normalized", predictions="fixture-predictions",
                recovery_authorization=old.OLD_AUTH, support_recovery_authorization=release.SUPPORT_AUTH)
            left = old.evaluate_test(root/"old", "ood_polyphony", **kwargs)
            right = release.evaluate_test(root/"new", "ood_polyphony", **kwargs, evaluation_release_authorization="fixture")
            first, second = release.p.read_reference(left), release.p.read_reference(right)
            self.assertEqual(set(first["artifacts_sha256"]), release.EVALUATION_FILES)
            self.assertEqual(first["artifacts_sha256"], second["artifacts_sha256"])
            self.assertEqual(checkpoints, ["released"])

    def test_exact_plan_and_implementation_authority(self):
        valid = release.p.read_reference(release.PLAN_AUDIT)
        for key, value in (("target", release.SUPPORT_CONTRACT), ("status", "NEEDS_REVISION"),
                           ("reports", []), ("scope", "SUPPORT_VALIDATION_RECOVERY_PLAN")):
            with patch.object(release.p, "read_reference", return_value={**valid, key: value}), self.assertRaises(PermissionError):
                release.verify_plan_audit()
        impl = {"status": "PASS", "scope": "EVALUATION_RELEASE_IMPLEMENTATION", "contract": "new", "reports": [release.PLAN_REPORT]}
        for key, value in (("contract", release.SUPPORT_CONTRACT), ("scope", "OTHER"), ("reports", [])):
            with patch.object(release.p, "read_reference", return_value={**impl, key: value}), self.assertRaises(PermissionError):
                release.verify_implementation_audit("audit", "new")

    def test_reject_source_contract_and_role_changes(self):
        valid = release.contract_candidate()
        for key, value in (("executor_sources", {}), ("support_contract", "other"), ("roles", old.ROLES), ("limits", {})):
            with patch.object(release.p, "read_reference", return_value={**valid, key: value}), self.assertRaises(PermissionError):
                release.verify_contract({"path": "fixture", "sha256": "0"*64})

    def test_request_is_evaluation_only_and_rejects_cuda_or_mixed_inputs(self):
        arguments = {k: None for k in operator.ARGUMENTS["evaluation"]}
        arguments.update(split="ood_polyphony", authorization=release.FROZEN["test_authorization"],
            recovery_authorization=old.OLD_AUTH, support_recovery_authorization=release.SUPPORT_AUTH)
        base = {"schema": "partition-evaluation-release-request-v1", "request_id": "fixture",
            "operation": "evaluation", "output": "unused", "arguments": arguments, "execution_contract": "fixture"}
        variants = [{**base, "operation": "inference"}, {**base, "arguments": {**arguments, "gpu_grant": "cuda"}},
                    {**base, "arguments": {**arguments, "split": "iid"}},
                    {**base, "arguments": {**arguments, "support_recovery_authorization": "other"}}]
        for record in variants:
            with patch.object(release.p, "read_reference", return_value=record), self.assertRaises((ValueError, PermissionError)):
                operator.validate_request({"path": "fixture", "sha256": "0"*64})

    def test_closed_evaluation_and_eight_payload_replay(self):
        execution = {"base_common": "base", "evaluation_release_contract": "contract",
            "evaluation_release_authorization": "auth", "recovery_authorization": old.OLD_AUTH,
            "support_recovery_authorization": release.SUPPORT_AUTH}
        args = dict(authorization=release.FROZEN["test_authorization"], data="data", logits="logits", scored="scored",
            normalized="normalized", predictions="predictions", recovery_authorization=old.OLD_AUTH,
            support_recovery_authorization=release.SUPPORT_AUTH, evaluation_release_authorization="auth")
        binding = {**execution, **args, "split": "ood_polyphony", "freeze": release.FROZEN["freeze"]}
        with tempfile.TemporaryDirectory(dir=BASE) as folder, \
                patch.object(release, "release_entry", return_value=execution), \
                patch.object(release, "execution_binding", return_value=execution):
            root = Path(folder)
            def bundle(name, *, identity=None, changed=False, files=None):
                out = root/name
                out.mkdir()
                for file in release.EVALUATION_FILES if files is None else files:
                    if file.endswith('.npz'):
                        write_npz(out/file, indices=np.arange(3))
                    else:
                        write_json(out/file, {"fixture": 2 if changed and file == "metrics.json" else 1})
                seal_bundle(out, role=release.ROLES["evaluation"], binding=binding if identity is None else identity, resources={})
                return release.p.reference(out/"manifest.json")
            a, b = bundle("a"), bundle("b")
            release.compare_evaluation_replay(a, b, "ood_polyphony", **args)
            for ref in (bundle("different", changed=True), bundle("wrong", identity={**binding, "predictions": "other"}),
                        bundle("missing", files=release.EVALUATION_FILES-{"support.json"})):
                with self.assertRaises(ValueError):
                    release.compare_evaluation_replay(a, ref, "ood_polyphony", **args)

    def test_six_failed_payloads_require_identical_inputs_and_bytes(self):
        with tempfile.TemporaryDirectory(dir=BASE) as folder:
            root = Path(folder)
            out = root/"new"
            out.mkdir()
            refs = []
            for name in release.FAILED_FILES:
                if name.endswith('.npz'):
                    write_npz(out/name, indices=np.arange(3))
                else:
                    write_json(out/name, {"fixture": True})
                refs.append(release.p.reference(out/name))
            request = root/"request.json"
            write_json(request, {"arguments": {"fixed": True}})
            with patch.object(release, "FAILED_REQUEST", release.p.reference(request)), \
                    patch.object(release, "failed_evidence", return_value=refs) as verify:
                release.compare_failed(out, "ood_polyphony", {"fixed": True})
                self.assertEqual(verify.call_count, 2)
                with self.assertRaises(ValueError):
                    release.compare_failed(out, "ood_polyphony", {"fixed": False})
                with (out/"metrics.json").open("ab") as handle:
                    handle.write(b"mutation")
                with self.assertRaises(ValueError):
                    release.compare_failed(out, "ood_polyphony", {"fixed": True})

    def test_failure_inventory_terminal_and_marker_are_not_completion(self):
        for mode in ("valid", "extra", "missing", "manifest", "symlink", "marker", "terminal", "hash"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory(dir=BASE) as folder, ExitStack() as stack:
                root = Path(folder)
                failed = root/"outputs"/"ood_polyphony"/"failed"
                failed.mkdir(parents=True)
                rel = failed.relative_to(release.p.ROOT).as_posix()
                for name in release.FAILED_FILES:
                    if mode == "missing" and name == "summary.json":
                        continue
                    write_json(failed/name, {"fixture": name})
                marker = {"status": "INCOMPLETE",
                    "error": "RuntimeError(\"TimeoutError('recovery worker exceeded its original CPU envelope')\")"}
                write_json(failed/"FAILURE.json", marker if mode != "marker" else {"status": "COMPLETE"})
                write_json(root/"request.json", {"operation": "evaluation", "output": rel,
                    "execution_contract": release.SUPPORT_CONTRACT, "arguments": {
                        "support_recovery_authorization": release.SUPPORT_AUTH, "recovery_authorization": old.OLD_AUTH}})
                request = release.p.reference(root/"request.json")
                write_json(root/"started.json", {"rss_limit_bytes": release.LIMITS["evaluation"][1],
                    "wall_limit_seconds": release.LIMITS["evaluation"][0]})
                terminal = {"status": "FAILED", "worker_terminal_confirmed": True, "worker_exit_code": -15,
                    "result": None, "request": request, "output": rel, "operation": "evaluation",
                    "execution_contract": release.SUPPORT_CONTRACT, "observed_peak_gpu_bytes": 0,
                    "seconds": 100, "observed_peak_rss_bytes": release.LIMITS["evaluation"][1]+1}
                if mode == "terminal":
                    terminal["worker_terminal_confirmed"] = False
                write_json(root/"terminal.json", terminal)
                term = release.p.reference(root/"terminal.json")
                inventory = [release.p.reference(path) for path in failed.iterdir()]
                if mode == "hash":
                    inventory[0]["sha256"] = "0"*64
                write_json(root/"diagnostic.json", {
                    "schema": "partition-evaluation-support-memory-diagnostic-v1",
                    "status": "DIAGNOSTIC_ONLY_NOT_CAMPAIGN_COMPLETION",
                    "terminal": term, "request": request, "contract": release.SUPPORT_CONTRACT,
                    "started": release.p.reference(root/"started.json"), "partial_inventory": inventory})
                if mode in ("extra", "manifest"):
                    write_json(failed/("manifest.json" if mode == "manifest" else "extra.json"), {"fake": True})
                if mode == "symlink":
                    (failed/"link").symlink_to(root/"request.json")
                for key, value in {"FAILED_ROOT": rel, "FAILED_REQUEST": request, "FAILED_TERMINAL": term,
                        "DIAGNOSTIC": release.p.reference(root/"diagnostic.json")}.items():
                    stack.enter_context(patch.object(release, key, value))
                stack.enter_context(patch.object(old, "RECOVERY", root.relative_to(release.p.ROOT).as_posix()))
                if mode == "valid":
                    self.assertEqual(len(release.failed_evidence()), 6)
                else:
                    with self.assertRaises(ValueError):
                        release.failed_evidence()

    def test_new_request_confinement_symlinks_and_no_overwrite(self):
        with tempfile.TemporaryDirectory(dir=BASE) as folder, ExitStack() as stack:
            root = Path(folder)
            stack.enter_context(patch.object(release, "RECOVERY", root.relative_to(release.p.ROOT).as_posix()))
            stack.enter_context(patch.object(release, "verify_authorization", return_value=({"contract": "fixture"}, {})))
            args = {k: "fixture" for k in operator.ARGUMENTS["evaluation"]}
            args.update(split="ood_polyphony", authorization=release.FROZEN["test_authorization"],
                recovery_authorization=old.OLD_AUTH, support_recovery_authorization=release.SUPPORT_AUTH)
            output = root/"outputs"/"ood_polyphony"/"evaluation"
            output.parent.mkdir(parents=True)
            requests = root/"requests"
            requests.mkdir()
            request = {"schema": "partition-evaluation-release-request-v1", "request_id": "fixture",
                "operation": "evaluation", "output": output.relative_to(release.p.ROOT).as_posix(),
                "arguments": args, "execution_contract": "fixture"}
            path = requests/"fixture.json"
            write_json(path, request)
            ref = release.p.reference(path)
            self.assertEqual(operator.validate_request(ref)[1], output)
            output.mkdir()
            with self.assertRaises(FileExistsError):
                operator.validate_request(ref)
            with self.assertRaises(FileExistsError):
                write_json(path, request)
            (output.parent/"link").symlink_to(root, target_is_directory=True)
            for bad in (root/"escape", output.parent/"link"/"new"):
                with patch.object(release.p, "read_reference",
                        return_value={**request, "output": bad.relative_to(release.p.ROOT).as_posix()}):
                    with self.assertRaises(ValueError):
                        operator.validate_request(ref)

    def test_roster_calls_upstream_closure_before_evaluation(self):
        splits = {s: dict(executor={"predictions": "support03", "evaluation": "release01"} if s in release.SPLITS_NEW
            else {"predictions": "memory02", "evaluation": "memory02"}, data="data", aggregate="aggregate",
            logits="logits", scored="scored", normalized="normalized", predictions="predictions",
            evaluation="evaluation", replay="replay") for s in release.TESTS}
        record = {"schema": "partition-evaluation-release-mixed-roster-v1", "evaluation_release_authorization": "auth",
            "normalization_audit": operator.NORMALIZATION_AUDIT, "splits": splits}
        for name in ("checked_aggregate", "checked_predictions"):
            with patch.object(release, "verify_authorization", return_value=({"contract": "fixture"}, {"base_common": {}})), \
                    patch.object(release.p, "read_reference", return_value={"train_data": "train", "calibration_data": "cal"}), \
                    patch.object(operator.sop, "checked_aggregate") as aggregate, \
                    patch.object(operator.sop, "checked_predictions") as predictions, \
                    patch.object(operator.old_gate, "compare_evaluation_replay") as evaluation:
                (aggregate if name == "checked_aggregate" else predictions).side_effect = ValueError("corrupt "+name)
                with self.assertRaisesRegex(ValueError, "corrupt "+name):
                    operator.check_roster(record, "auth")
                evaluation.assert_not_called()

    def test_new_worker_cli_rejects_cuda_and_supervisor_retains_caps(self):
        self.assertEqual(release.LIMITS, {"evaluation": [1200, 2147483648]})
        self.assertIs(operator.supervise_process, operator.sop.supervise_process)
        with tempfile.TemporaryDirectory(dir=BASE) as folder:
            # Exercise the actual new CLI through the already-tested process supervisor.
            result = operator.supervise_process(
                [sys.executable, str(release.p.ROOT/release.SOURCES[1]), "--worker-fd"],
                env=dict(os.environ, CUDA_VISIBLE_DEVICES="fixture-forbidden", OMP_NUM_THREADS="1",
                    OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1"), control=Path(folder),
                wall=10, rss_limit=2147483648, payload={"reference": "must-not-be-read"})
            self.assertTrue(result["worker_terminal_confirmed"])
            self.assertNotEqual(result["worker_exit_code"], 0)
            self.assertIsNotNone(result["error"])
            logs = "\n".join(path.read_text() for path in Path(folder).iterdir() if path.suffix == ".log")
            self.assertIn("must not expose CUDA", logs)


if __name__ == "__main__":
    unittest.main()
