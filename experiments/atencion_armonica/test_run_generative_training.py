"""CPU-only supervisor mechanics; no corpus production, fit, forward or training."""
from copy import deepcopy
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.atencion_armonica import run_generative_training as cli
from src.atencion_armonica.generative_evidence_storage import write_json


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, value)
    return cli.reference(path)


@pytest.fixture
def controls(tmp_path, monkeypatch):
    prep = tmp_path/"preparation-control"
    delivery = tmp_path/"delivery-control"
    train = tmp_path/"training-control"
    prep.mkdir(); delivery.mkdir(); train.mkdir()
    monkeypatch.setattr(cli, "PREPARATION_CONTROL", prep)
    monkeypatch.setattr(cli, "PREPARATION_LOCK", prep/"operator.lock")
    monkeypatch.setattr(cli, "DELIVERY_CONTROL", delivery)
    monkeypatch.setattr(cli, "DELIVERY_MANIFEST", delivery/"manifest.json")
    monkeypatch.setattr(cli, "TRAINING_CONTROL", train)
    monkeypatch.setattr(cli, "TRAINING_MANIFEST", train/"manifest.json")
    monkeypatch.setattr(cli.preparation, "CONTROL", prep)
    return prep, delivery, train


def test_fixed_roster_and_systemd_caps_leave_stop_grace_outside_runtime():
    roster = cli.cell_roster()
    assert len(roster) == 27 and len({c["cell_id"] for c in roster}) == 27
    assert [(c["arm"], c["checkpoint_seed"], c["reader_seed"]) for c in roster] == [
        (a, cp, seed) for a in cli.ge.ARMS for cp in cli.ge.CHECKPOINTS for seed in cli.ge.READER_SEEDS]
    command = cli._service_command("own-unit", 1770, "--training-worker", "attempt-0000", gpu=True)
    assert "--property=MemoryMax=6G" in command and "--property=MemorySwapMax=0" in command
    assert "--property=RuntimeMaxSec=1770s" in command and "--property=TimeoutStopSec=30s" in command
    assert "--property=KillMode=control-group" in command and "CUDA_VISIBLE_DEVICES=0" in command
    with pytest.raises(ValueError, match="identifier"):
        cli._service_command("u", 10, "--training-worker", "../escape", gpu=True)


def test_physical_disk_accounting_rejects_links_without_following_them(tmp_path, monkeypatch):
    temporary, destination = tmp_path/"temp", tmp_path/"data"
    temporary.mkdir(); destination.mkdir()
    (temporary/"payload").write_bytes(b"1234")
    monkeypatch.setattr(cli, "TEMP", temporary)
    monkeypatch.setattr(cli, "DESTINATION", destination)
    monkeypatch.setattr(cli.shutil, "disk_usage", lambda path: SimpleNamespace(free=100*cli.GIB))
    assert cli.disk_usage() == {"new_bytes": 4, "free_bytes": 100*cli.GIB}
    (temporary/"dangling").symlink_to(temporary/"absent")
    assert cli.disk_usage()["new_bytes"] == 4+len(str(temporary/"absent"))
    (destination/"bad").symlink_to(destination/"absent")
    with pytest.raises(ValueError, match="canonical.*symlink"):
        cli.disk_usage()


def test_open_exit_without_typed_worker_never_authorizes_delivery(controls, monkeypatch):
    prep, _, _ = controls
    monkeypatch.setattr(cli.preparation, "MANIFEST", prep/"manifest.json")
    manifest_ref = put(cli.preparation.MANIFEST, {"fixture": True})
    monkeypatch.setattr(cli.preparation, "read_manifest", lambda ref: {"prior_profile_seconds": 3.})
    launch = prep/"attempt-0000.launch.json"
    write_json(launch, {"manifest": manifest_ref, "previous_exit": None, "unit": "u"})
    lref = cli.reference(launch)
    write_json(prep/"attempt-0000.exit.json", {"launch": lref, "terminal": True, "seconds": 2., "returncode": 0})
    with pytest.raises(RuntimeError, match="typed worker"):
        cli.verified_open_completion()


def test_shared_preparation_delivery_ledger_charges_failures_and_parents(controls, monkeypatch):
    prep, delivery, _ = controls
    open_ref = {"path": "fixture/open.json", "sha256": "0"*64}
    delivery_ref = {"path": "fixture/delivery.json", "sha256": "2"*64}
    previous = None
    launch0 = prep/"attempt-0000.launch.json"
    put(launch0, {"manifest": open_ref, "previous_exit": previous, "unit": "open"})
    exit0 = prep/"attempt-0000.exit.json"
    previous = put(exit0, {"launch": cli.reference(launch0), "terminal": True, "seconds": 7.})
    launch1 = delivery/"attempt-0000.launch.json"
    remaining = cli.PREPARATION_LIMIT-17
    runtime = remaining-cli.GRACE
    put(launch1, {"schema": "generative-evidence-delivery-launch-v1", "phase": "delivery",
                  "manifest": open_ref, "delivery_manifest": delivery_ref,
                  "previous_exit": previous, "unit": "delivery",
                  "command": cli._service_command("delivery", runtime, "--delivery-worker",
                                                  "attempt-0000", gpu=False),
                  "used_seconds": 17., "remaining_seconds": remaining, "runtime_seconds": runtime})
    exit1 = delivery/"attempt-0000.exit.json"
    final = put(exit1, {"launch": cli.reference(launch1), "terminal": True, "seconds": 5.})
    monkeypatch.setattr(cli, "read_delivery_manifest", lambda ref: {
        "open": {"manifest": open_ref, "exit": previous}, "preparation_used_seconds": 17.})
    assert cli.delivery_accumulated(delivery_ref) == (22., final)
    exit1.rename(delivery/"preserved-exit")
    with pytest.raises(RuntimeError, match="unreconciled"):
        cli.delivery_accumulated(delivery_ref)


def test_training_ledger_has_global_and_per_cell_chains(controls):
    _, _, train = controls
    manifest = {"path": "fixture/training.json", "sha256": "3"*64}
    cells = cli.cell_roster()
    global_parent, cell_parent = None, {}
    total, per_cell = 0., {c["cell_id"]: 0. for c in cells}
    availability = {"processes": "", "inventory": "NVIDIA GeForce RTX 3090, fixture\n"}
    for i, (cell, seconds, code) in enumerate(((cells[0], 19., 1), (cells[1], 7., 75), (cells[0], 5., 0))):
        launch = train/f"attempt-{i:04d}.launch.json"
        remaining = int(min(cli.TRAINING_LIMIT-total, cli.CELL_LIMIT-per_cell[cell["cell_id"]]))
        runtime = remaining-cli.GRACE
        put(launch, {"schema": "generative-evidence-training-launch-v1", "manifest": manifest,
            "previous_exit": global_parent, "previous_cell_exit": cell_parent.get(cell["cell_id"]),
            "cell": cell, "unit": f"u{i}",
            "command": cli._service_command(f"u{i}", runtime, "--training-worker",
                                            f"attempt-{i:04d}", gpu=True),
            "availability": availability, "used_total_seconds": total,
            "used_cell_seconds": per_cell[cell["cell_id"]],
            "remaining_allowance_seconds": remaining, "runtime_seconds": runtime})
        end = train/f"attempt-{i:04d}.exit.json"
        ref = put(end, {"launch": cli.reference(launch), "terminal": True,
                        "seconds": seconds, "process_returncode": code})
        global_parent, cell_parent[cell["cell_id"]] = ref, ref
        total += seconds
        per_cell[cell["cell_id"]] += seconds
    total, parent, per_cell, parents = cli.training_accumulated(manifest)
    assert total == 31 and parent == global_parent
    assert per_cell[cells[0]["cell_id"]] == 24 and per_cell[cells[1]["cell_id"]] == 7
    assert parents == cell_parent
    (train/"attempt-0002.exit.json").rename(train/"preserved-exit")
    with pytest.raises(RuntimeError, match="unreconciled"):
        cli.training_accumulated(manifest)


def test_payload_alone_or_failed_latest_attempt_cannot_authorize_cell():
    cell = cli.cell_roster()[0]
    manifest = {"path": "fixture/training.json", "sha256": "4"*64}
    complete = {"path": "data/cell/complete.json", "sha256": "5"*64}
    with pytest.raises(RuntimeError, match="without a supervised"):
        cli._require_trained_authority(manifest, cell, complete, None)
    launch_ref = {"path": "fixture/launch.json", "sha256": "6"*64}
    latest = {"launch_ref": launch_ref, "launch": {"manifest": manifest, "cell": cell},
              "exit": {"launch": launch_ref, "terminal": True, "process_returncode": 1},
              "worker": {"launch": launch_ref, "status": "TRAINED",
                         "complete": {"path": "complete.json", "sha256": complete["sha256"], "bytes": 1}}}
    with pytest.raises(RuntimeError, match="not typed"):
        cli._require_trained_authority(manifest, cell, complete, latest)
    latest["exit"]["process_returncode"] = 0
    cli._require_trained_authority(manifest, cell, complete, latest)
    latest["worker"]["complete"]["sha256"] = "7"*64
    with pytest.raises(ValueError, match="exact complete"):
        cli._require_trained_authority(manifest, cell, complete, latest)


class FakeStore:
    def __init__(self, profile):
        self.profile = profile
    def json(self, ref):
        return deepcopy(self.profile)


def test_projection_uses_real_nine_loads_and_51_saves_plus_51_final_reads(monkeypatch):
    delivered = {"path": "delivered/index.json", "sha256": "8"*64, "bytes": 10}
    cases = [{"arm": arm, "checkpoint_seed": cp, "seconds": 12., "train_count": 4096,
              "calibration_count": 512, "train_eligible": 4095, "calibration_eligible": 511}
             for arm in cli.ge.ARMS for cp in cli.ge.CHECKPOINTS]
    load = {"schema": "generative-evidence-real-cell-loader-profile-v1", "device": "cpu",
            "delivered": delivered, "cases": cases, "case_count": 9, "method": cli.LOAD_PROFILE_METHOD,
            "peak_rss_bytes": 2*cli.GIB}
    head = {"setup_seconds": 1., "steady_update_seconds": [0.01],
            "evaluation_batch_io_seconds": [0.02], "snapshot_io_seconds": 0.03}
    gpu = {"status": "MEASURED", "device": "cuda:0", "runtime": {"runtime": True},
           "heads": {"envelope": head, "observed_train": deepcopy(head)},
           "peak_rss_bytes": 3*cli.GIB, "peak_reserved_bytes": 1*cli.GIB}
    cpu = {"status": "MEASURED", "device": "cpu"}
    comparison = {"status": "RESOURCE_ESTIMATE_NOT_CAMPAIGN_AUTHORIZATION",
                  "profiles": [{**cli.CPU_PROFILE, "bytes": 1}, {**cli.GPU_PROFILE, "bytes": 1}]}
    class Reader:
        def __init__(self, root): pass
        def json(self, ref):
            return deepcopy({cli.COMPARISON["path"]: comparison, cli.GPU_PROFILE["path"]: gpu,
                             cli.CPU_PROFILE["path"]: cpu}[ref["path"]])
    monkeypatch.setattr(cli, "VerifiedBytes", Reader)
    monkeypatch.setattr(cli, "runtime", lambda: {"runtime": True})
    delivery = {"store": FakeStore(load), "load_profile": {"path": "load.json"}, "delivered": delivered}
    plan = cli.training_resource_plan(delivery)
    assert len(plan["cells"]) == 27 and plan["snapshot_count_per_cell"] == 51
    assert plan["final_snapshot_reads_per_cell"] == 51 and plan["final_calibration_reads_per_cell"] == 10
    assert plan["unmeasured_reserve_seconds_per_cell"] == 300
    expected_base = 1+12+50*128*.01+10*16*.02+51*.03+51*.03+10*.02
    assert plan["cells"][0]["measured_base_seconds"] == pytest.approx(expected_base)
    assert plan["projected_total_seconds"] < cli.TRAINING_LIMIT


def test_load_profile_requires_exact_nine_case_roster():
    delivered = {"path": "delivered/index.json", "sha256": "8"*64, "bytes": 10}
    cases = [{"arm": arm, "checkpoint_seed": cp, "seconds": 1., "train_count": 4096,
              "calibration_count": 512, "train_eligible": 4000, "calibration_eligible": 500}
             for arm in cli.ge.ARMS for cp in cli.ge.CHECKPOINTS]
    value = {"schema": "generative-evidence-real-cell-loader-profile-v1", "device": "cpu",
             "delivered": delivered, "cases": cases, "case_count": 9, "method": cli.LOAD_PROFILE_METHOD,
             "peak_rss_bytes": 1}
    cli._validate_load_profile(value, delivered)
    value["cases"] = value["cases"][:-1]
    with pytest.raises(ValueError, match="roster"):
        cli._validate_load_profile(value, delivered)


def paused_attempt(manifest, cell):
    ref = {"path": "fixture/launch.json", "sha256": "6"*64}
    return {"launch_ref": ref, "launch": {"manifest": manifest, "cell": cell},
            "exit": {"launch": ref, "terminal": True, "process_returncode": 75},
            "worker": {"launch": ref, "status": "PAUSED_RECOVERABLE"}}


def test_recoverable_authority_requires_exact_parents_and_terminal_pause():
    manifest, cell = {"fixture": True}, cli.cell_roster()[0]
    latest = paused_attempt(manifest, cell)
    assert cli._recoverable_cell_attempt(manifest, cell, latest)
    assert not cli._recoverable_cell_attempt(manifest, cell, None)
    for section, key, value in (("launch", "manifest", {}), ("launch", "cell", {}),
            ("exit", "launch", {}), ("worker", "launch", {}), ("exit", "terminal", False),
            ("exit", "process_returncode", 0), ("worker", "status", "FAILED")):
        changed = deepcopy(latest)
        changed[section][key] = value
        assert not cli._recoverable_cell_attempt(manifest, cell, changed)


@pytest.mark.parametrize("payload_exists", [False, True])
def test_pause_after_completion_relaunches_instead_of_authorizing(controls, monkeypatch, payload_exists):
    _, _, train = controls
    manifest = put(train/"manifest.json", {"fixture": True})
    cell = cli.cell_roster()[0]
    latest = paused_attempt(manifest, cell)
    complete = {"path": "fixture/complete.json", "sha256": "5"*64} if payload_exists else None
    monkeypatch.setattr(cli, "read_training_manifest", lambda ref: ({"roster": [cell]}, {}))
    monkeypatch.setattr(cli, "check_disk", lambda: None)
    monkeypatch.setattr(cli, "training_accumulated", lambda ref: (
        3., None, {cell["cell_id"]: 3.}, {}))
    monkeypatch.setattr(cli, "_latest_cell_worker", lambda cell_id: latest)
    monkeypatch.setattr(cli, "verify_cell_complete", lambda *a, **k: complete)
    monkeypatch.setattr(cli, "gpu_availability", lambda: {
        "processes": "", "inventory": "NVIDIA GeForce RTX 3090, fixture\n"})
    def reject_promotion(*args):
        raise AssertionError("paused payload was promoted without reopening the worker")
    monkeypatch.setattr(cli, "_require_trained_authority", reject_promotion)
    class RelaunchReached(Exception):
        pass
    def execute(command, unit, launch_ref, exit_path):
        launch = cli._read(launch_ref)
        assert launch["cell"] == cell and launch["used_cell_seconds"] == 3.
        assert "--training-worker" in command
        raise RelaunchReached
    monkeypatch.setattr(cli, "_execute_command", execute)
    with pytest.raises(RelaunchReached):
        cli.run_training()
