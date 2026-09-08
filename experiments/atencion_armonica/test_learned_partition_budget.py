"""Immutable budget/terminal fixtures, no training process or device usage."""
from contextlib import ExitStack
import json
import os
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest
from unittest.mock import patch

from experiments.atencion_armonica.test_learned_partition_data import BASE
from src.atencion_armonica import learned_partition_budget as b
from src.atencion_armonica.structured_source_artifacts import write_json, seal_bundle, verify_bundle


class BudgetTests(unittest.TestCase):
    def fixture(self, root, name, *, resume=None, arm="shared_source"):
        request = {"request_id": name, "operation": "train_cell", "output": (root/name).relative_to(b.p.ROOT).as_posix(),
                   "arguments": {"arm": arm, "checkpoint_seed": b.SEEDS[0], "reader_seed": b.READER_SEEDS[0], "resume": resume}}
        request["arguments"]["authorization"] = None
        path = root/f"{name}.json"
        write_json(path, request)
        control = root/f"supervisor-{name}"
        control.mkdir()
        ref = b.p.reference(path)
        write_json(control/"request.json", {"reference": ref, "request": request})
        return ref, request, control

    def terminal(self, ref, request, control, permit, seconds, status="FAILED"):
        record = {k: None for k in b.TERMINAL_KEYS}
        record.update(status=status, request=ref, request_id=request["request_id"], operation="train_cell",
            output=request["output"], worker_pid=99999999, worker_terminal_confirmed=True,
            worker_exit_code=0 if status == "COMPLETE" else 1, seconds=seconds,
            observed_peak_rss_bytes=1000, observed_peak_gpu_bytes=0,
            error=None if status == "COMPLETE" else "mechanical failure", budget=permit, recovery_status="SNAPSHOT_REQUIRED")
        write_json(control/"started.json", {"request": ref, "budget": permit})
        write_json(control/"terminal.json", record)
        return b.p.reference(control/"terminal.json")

    def test_failed_time_is_accumulated_and_reset_or_old_parent_rejected(self):
        BASE.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=BASE) as folder, ExitStack() as stack:
            root = Path(folder)
            stack.enter_context(patch.object(b, "STAGING", root))
            stack.enter_context(patch.object(b, "REGISTRY", root/"registry"))
            ref, request, control = self.fixture(root, "first")
            with b.reserve(ref, request, control) as (permit, remaining):
                self.assertEqual(remaining, 600.)
                with self.assertRaises(PermissionError):
                    b.accounting()
                terminal = self.terminal(ref, request, control, permit, 17.)
            entries, total, cells = b.accounting()
            self.assertEqual((len(entries), total, list(cells.values())), (1, 17., [17.]))
            newref, newrequest, newcontrol = self.fixture(root, "reset")
            with self.assertRaises(PermissionError), b.reserve(newref, newrequest, newcontrol):
                pass
            resume = {"request": ref, "terminal": terminal, "snapshot": {"fixture": "snapshot"}}
            secondref, second, secondcontrol = self.fixture(root, "second", resume=resume)
            with b.reserve(secondref, second, secondcontrol) as (permit, remaining):
                self.assertEqual(remaining, 583.)
                self.terminal(secondref, second, secondcontrol, permit, 21.)
            self.assertEqual(b.accounting()[1], 38.)
            thirdref, third, thirdcontrol = self.fixture(root, "wrong_parent", resume=resume)
            with self.assertRaises(PermissionError), b.reserve(thirdref, third, thirdcontrol):
                pass
            last_entry = b.REGISTRY/(secondref["sha256"]+".json")
            displaced = root/"preserved_entry.json"
            last_entry.rename(displaced)
            with self.assertRaises(ValueError):
                b.accounting()
            displaced.rename(last_entry)
            self.assertEqual(b.accounting()[1], 38.)
            last_entry.rename(displaced)
            secondcontrol.rename(root/"preserved_control")
            with self.assertRaises(ValueError):
                b.accounting()
            with self.assertRaises(ValueError), b.reserve(thirdref, third, thirdcontrol):
                pass
            displaced.rename(last_entry)
            (root/"preserved_control").rename(secondcontrol)
            self.assertEqual(b.accounting()[1], 38.)
            b.REGISTRY.rename(root/"preserved_registry")
            with self.assertRaises(ValueError):
                b.accounting()

    def test_complete_cell_is_reused_and_campaign_total_includes_other_cells(self):
        with tempfile.TemporaryDirectory(dir=BASE) as folder, ExitStack() as stack:
            root = Path(folder)
            stack.enter_context(patch.object(b, "STAGING", root))
            stack.enter_context(patch.object(b, "REGISTRY", root/"registry"))
            ref, request, control = self.fixture(root, "complete")
            with b.reserve(ref, request, control) as (permit, _):
                self.terminal(ref, request, control, permit, 100., status="COMPLETE")
            ref, request, control = self.fixture(root, "duplicate")
            with self.assertRaises(PermissionError), b.reserve(ref, request, control):
                pass
            stack.enter_context(patch.object(b, "CAMPAIGN_SECONDS", 150.))
            ref, request, control = self.fixture(root, "other", arm="pairs_structure")
            with b.reserve(ref, request, control) as (permit, remaining):
                self.assertEqual(remaining, 50.)
                self.terminal(ref, request, control, permit, 51.)
            ref, request, control = self.fixture(root, "exhausted", arm="local_compatibility")
            with self.assertRaises(PermissionError), b.reserve(ref, request, control):
                pass

    def test_worker_without_live_parent_permit_is_rejected(self):
        with self.assertRaises((ValueError, TypeError, PermissionError)):
            b.verify_permit(None, {"sha256": "0"*64})

    def test_reconcile_checks_both_processes_and_charges_elapsed_time(self):
        with tempfile.TemporaryDirectory(dir=BASE) as folder, ExitStack() as stack:
            root = Path(folder)
            stack.enter_context(patch.object(b, "STAGING", root))
            stack.enter_context(patch.object(b, "REGISTRY", root/"registry"))
            ref, request, control = self.fixture(root, "abrupt")
            with b.reserve(ref, request, control) as (permit, _):
                pass  # Mechanical lost-supervisor fixture, not a launched worker.
            with self.assertRaises(PermissionError):
                b.reconcile(permit)  # Our actual supervisor identity is still live.
            entry = b.p.read_reference(permit)
            write_json(control/"started.json", {"request": ref, "budget": permit, **entry["supervisor"],
                                                "monotonic_started": entry["reserved_monotonic"]})
            with patch.object(b, "process_terminal", side_effect=["PID_ABSENT", PermissionError("worker live")]):
                with self.assertRaises(PermissionError):
                    b.reconcile(permit)
            self.assertFalse((control/"terminal.json").exists())
            with patch.object(b, "process_terminal", return_value="PID_ABSENT"), \
                 patch.object(b.time, "monotonic", return_value=entry["reserved_monotonic"]+19.):
                terminal = b.reconcile(permit)
            record = b.terminal_receipt(terminal, request_ref=ref)
            self.assertIsNone(record["worker_exit_code"])
            self.assertEqual(record["seconds"], 19.)
            self.assertEqual(b.accounting()[1], 19.)
            self.assertEqual(b.reconcile(permit), terminal)
            resume = {"request": ref, "terminal": terminal, "snapshot": None}
            newref, newrequest, newcontrol = self.fixture(root, "resumed", resume=resume)
            with b.reserve(newref, newrequest, newcontrol) as (newpermit, remaining):
                self.assertEqual(remaining, 581.)
                self.terminal(newref, newrequest, newcontrol, newpermit, 7.)
            self.assertEqual(b.accounting()[1], 26.)

    def test_reconcile_real_departed_cpu_identity_before_worker_bootstrap(self):
        result = subprocess.run([sys.executable, "-c",
            "import json,os; from src.atencion_armonica.learned_partition_supervisor import process_identity; "
            "print(json.dumps(process_identity(os.getpid())))"], check=True, capture_output=True, text=True,
            timeout=10, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
        departed = json.loads(result.stdout)
        self.assertIn(b.process_terminal(departed), ("PID_ABSENT", "PID_REUSED"))
        with tempfile.TemporaryDirectory(dir=BASE) as folder, ExitStack() as stack:
            root = Path(folder)
            stack.enter_context(patch.object(b, "STAGING", root))
            stack.enter_context(patch.object(b, "REGISTRY", root/"registry"))
            ref, request, control = self.fixture(root, "bootstrap")
            with patch("src.atencion_armonica.learned_partition_supervisor.process_identity", return_value=departed):
                with b.reserve(ref, request, control) as (permit, _):
                    pass
            record = b.terminal_receipt(b.reconcile(permit), request_ref=ref)
            self.assertIsNone(record["worker_pid"])
            self.assertEqual(record["recovery_status"], "NO_UPDATES_NO_SNAPSHOT")
            self.assertGreater(b.accounting()[1], 0)

    def test_lost_boot_clock_exhausts_budget_instead_of_inventing_elapsed_time(self):
        with tempfile.TemporaryDirectory(dir=BASE) as folder, ExitStack() as stack:
            root = Path(folder)
            stack.enter_context(patch.object(b, "STAGING", root))
            stack.enter_context(patch.object(b, "REGISTRY", root/"registry"))
            ref, request, control = self.fixture(root, "prior_boot")
            identity = {"pid": 99999999, "boot_id": "mechanical-prior-boot", "process_start_ticks": "1"}
            with patch("src.atencion_armonica.learned_partition_supervisor.process_identity", return_value=identity):
                with b.reserve(ref, request, control) as (permit, _):
                    pass
            terminal = b.reconcile(permit)
            self.assertEqual(b.accounting()[1], b.CAMPAIGN_SECONDS+b.CELL_SECONDS)
            newref, newrequest, newcontrol = self.fixture(root, "resume_lost_clock",
                resume={"request": ref, "terminal": terminal, "snapshot": None})
            with self.assertRaises(PermissionError), b.reserve(newref, newrequest, newcontrol):
                pass

    def test_reconcile_completes_only_exact_unpublished_journal_tail(self):
        for fragment in ("missing", "partial", "invalid"):
            with self.subTest(fragment=fragment), tempfile.TemporaryDirectory(dir=BASE) as folder, ExitStack() as stack:
                root = Path(folder)
                stack.enter_context(patch.object(b, "STAGING", root))
                stack.enter_context(patch.object(b, "REGISTRY", root/"registry"))
                ref, request, control = self.fixture(root, "journal_crash")
                with patch.object(b, "append_journal", side_effect=RuntimeError("mechanical I/O failure")):
                    with self.assertRaises(RuntimeError), b.reserve(ref, request, control):
                        pass
                permit = b.p.reference(b.REGISTRY/(ref["sha256"]+".json"))
                line = (json.dumps({"sequence": 0, "entry": permit}, sort_keys=True)+"\n").encode()
                journal = root/"budget_reservations.jsonl"
                if fragment != "missing":
                    journal.write_bytes(line[:23] if fragment == "partial" else b"invalid")
                with patch.object(b, "process_terminal", return_value="PID_ABSENT"):
                    if fragment == "invalid":
                        with self.assertRaises(ValueError):
                            b.reconcile(permit)
                        self.assertEqual(journal.read_bytes(), b"invalid")
                        continue
                    terminal = b.reconcile(permit)
                self.assertEqual(journal.read_bytes(), line)
                b.terminal_receipt(terminal, request_ref=ref)
                self.assertGreater(b.accounting()[1], 0)

    def test_reconciled_failure_invalidates_sealed_output_but_preserves_files(self):
        with tempfile.TemporaryDirectory(dir=BASE) as folder, ExitStack() as stack:
            root = Path(folder)
            stack.enter_context(patch.object(b, "STAGING", root))
            stack.enter_context(patch.object(b, "REGISTRY", root/"registry"))
            ref, request, control = self.fixture(root, "sealed")
            with b.reserve(ref, request, control) as (permit, _):
                pass
            output = b.p.ROOT/request["output"]
            output.mkdir()
            write_json(output/"training_ready.json", {"mechanical": True})
            seal_bundle(output, role="learned_training_cell", binding={}, resources={})
            result = b.p.reference(output/"manifest.json")
            verify_bundle(output, result["sha256"], role="learned_training_cell")
            with patch.object(b, "process_terminal", return_value="PID_ABSENT"):
                terminal = b.reconcile(permit)
            self.assertEqual(b.terminal_receipt(terminal, request_ref=ref)["recovery_status"], "SNAPSHOT_REQUIRED")
            self.assertTrue((output/"training_ready.json").is_file())
            with self.assertRaises(ValueError):
                verify_bundle(output, result["sha256"], role="learned_training_cell")

    def test_supervisor_preserves_entry_when_reserve_fails_before_yield(self):
        from src.atencion_armonica import learned_partition_supervisor as s
        from src.atencion_armonica import learned_partition_gate as gate
        with tempfile.TemporaryDirectory(dir=BASE) as folder, ExitStack() as stack:
            root = Path(folder)
            stack.enter_context(patch.object(b, "STAGING", root))
            stack.enter_context(patch.object(b, "REGISTRY", root/"registry"))
            stack.enter_context(patch.object(s, "STAGING", root))
            ref, request, _ = self.fixture(root, "captured")
            with patch.object(s, "validate_request", return_value=(request, b.p.ROOT/request["output"])), \
                 patch.object(gate, "verify_authorization", return_value={"training_device": "cpu"}), \
                 patch.object(b, "append_journal", side_effect=RuntimeError("mechanical I/O failure")), \
                 patch.object(s.subprocess, "Popen") as launch:
                with self.assertRaises(RuntimeError):
                    s.supervised(ref)
                launch.assert_not_called()
            permit = b.p.reference(b.REGISTRY/(ref["sha256"]+".json"))
            terminal_path, = root.glob("supervisor-*/terminal.json")
            terminal = b.p.reference(terminal_path)
            self.assertEqual(b.terminal_receipt(terminal, request_ref=ref)["budget"], permit)
            with patch.object(b, "process_terminal", return_value="PID_ABSENT"):
                self.assertEqual(b.reconcile(permit), terminal)
            self.assertGreater(b.accounting()[1], 0)


if __name__ == "__main__":
    unittest.main()
