"""CPU-only tests of stage bindings, elapsed ledgers and the external guard."""
import json
from types import SimpleNamespace

import pytest

from experiments.atencion_armonica import prepare_generative_evidence as cli
from src.atencion_armonica.generative_evidence_storage import write_json


@pytest.fixture
def control(tmp_path, monkeypatch):
    path = tmp_path/"control"
    path.mkdir()
    monkeypatch.setattr(cli, "CONTROL", path)
    monkeypatch.setattr(cli, "MANIFEST", path/"manifest.json")
    return path


def test_initialize_binds_real_profiles_without_cuda_or_dataset_writes(control, monkeypatch):
    import torch
    before = torch.cuda.is_initialized()
    monkeypatch.setattr(cli, "check_disk", lambda: {"new_bytes": 0, "free_bytes": 100*cli.GIB})
    ref = cli.initialize()
    manifest = cli.read_manifest(ref)
    assert manifest["stage"] == "OPEN_PREPARATION_ONLY" and not manifest["test_access"]
    assert manifest["projection_seconds"]+manifest["prior_profile_seconds"] < cli.LIMIT_SECONDS
    assert manifest["count"] == {"train": 4096, "calibration": 512}
    assert manifest["prior_profile_seconds"] > 110
    assert torch.cuda.is_initialized() == before
    with pytest.raises(FileExistsError):
        cli.initialize()
    original = cli.sources
    monkeypatch.setattr(cli, "sources", lambda: {**original(), "unexpected.py": "0"*64})
    assert cli.read_manifest(ref) == manifest  # Later, unused modules do not change this stage.
    changed = next(iter(manifest["sources"]))
    monkeypatch.setattr(cli, "sources", lambda: {**original(), changed: "0"*64})
    with pytest.raises(ValueError, match="sources"):
        cli.read_manifest(ref)


def test_ledger_accumulates_failure_and_pause_and_pins_parents(control):
    manifest = {"path": "fixture/manifest.json", "sha256": "0"*64}
    assert cli.accumulated(manifest, 12.) == (12., None)
    previous = None
    for i, seconds in enumerate((7., 11.)):
        path = control/f"attempt-{i:04d}.launch.json"
        write_json(path, {"manifest": manifest, "previous_exit": previous, "unit": f"fixture-{i}"})
        with pytest.raises(RuntimeError, match="unreconciled"):
            cli.accumulated(manifest, 12.)
        end = control/f"attempt-{i:04d}.exit.json"
        write_json(end, {"launch": cli.reference(path), "terminal": True,
                         "seconds": seconds, "returncode": 1 if i == 0 else 75})
        previous = cli.reference(end)
    assert cli.accumulated(manifest, 12.) == (30., previous)
    with pytest.raises(ValueError, match="manifest"):
        cli.accumulated({"path": "another", "sha256": "1"*64}, 12.)
    with pytest.raises(ValueError, match="initial"):
        cli.accumulated(manifest, -1.)


@pytest.mark.parametrize("field, value", [("sources", {}), ("prior_profile_seconds", -10),
    ("projection_seconds", 0), ("profile", {}), ("resource_measurements", {}), ("unmeasured_overhead_reserve_seconds", 0)])
def test_preexisting_manifest_cannot_drop_binding_or_change_budget(control, monkeypatch, field, value):
    monkeypatch.setattr(cli, "check_disk", lambda: {"new_bytes": 0, "free_bytes": 100*cli.GIB})
    cli.initialize()
    manifest = json.loads(cli.MANIFEST.read_bytes())
    manifest[field] = value
    # A fabricated, independently hashed JSON is not valid stage evidence.
    fake = control/"fabricated.json"
    write_json(fake, manifest)
    monkeypatch.setattr(cli, "MANIFEST", fake)
    with pytest.raises(ValueError, match="OPEN stage"):
        cli.read_manifest(cli.reference(fake))


def test_ledger_never_treats_running_or_ambiguous_exit_as_terminal(control):
    manifest = {"path": "fixture", "sha256": "0"*64}
    launch = control/"attempt-0000.launch.json"
    write_json(launch, {"manifest": manifest, "previous_exit": None, "unit": "fixture"})
    write_json(control/"attempt-0000.exit.json", {"launch": cli.reference(launch), "terminal": False, "seconds": 5})
    with pytest.raises(ValueError, match="terminal"):
        cli.accumulated(manifest, 0)


@pytest.mark.parametrize("text, expected", [
    ("LoadState=loaded\nActiveState=active\nSubState=running\n", False),
    ("", False),
    ("LoadState=not-found\nActiveState=inactive\nSubState=dead\n", True),
    ("LoadState=loaded\nActiveState=failed\nSubState=failed\n", True),
])
def test_authoritative_terminal_states(monkeypatch, text, expected):
    monkeypatch.setattr(cli.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=text, stderr="", returncode=0))
    assert cli.terminal_state("fixture")["terminal"] is expected


def test_numeric_external_deadline_verification(monkeypatch):
    def output(command, **kwargs):
        if "GetUnit" in command:
            return 'o "/org/freedesktop/systemd1/unit/fixture"\n'
        return "t 60000000\n" if command[-1] == "RuntimeMaxUSec" else "t 30000000\n"
    monkeypatch.setattr(cli.subprocess, "check_output", output)
    cli.verify_deadline("fixture", 60)
    with pytest.raises(RuntimeError, match="deadline"):
        cli.verify_deadline("fixture", 61)


def test_stop_activates_external_grace_and_retries_startup_race(monkeypatch):
    commands = []
    def run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=4 if len(commands) == 1 else 0)
    monkeypatch.setattr(cli.subprocess, "run", run)
    class Process:
        returncode = 75
        states = iter([None, None, None, 75])
        def poll(self):
            return next(self.states)
    monkeypatch.setattr(cli.subprocess, "Popen", lambda command, **kw: Process())
    monkeypatch.setattr(cli, "terminal_state", lambda unit: {"terminal": True})
    monkeypatch.setattr(cli.time, "sleep", lambda seconds: None)
    assert cli.run_command(["fixture-no-launch"], "own-uuid", [15]) == 75
    assert commands == [["systemctl", "stop", "--no-block", "own-uuid.service"]]*2


def test_stop_is_forwarded_even_when_command_client_already_exited(monkeypatch):
    calls = []
    def popen(command, **kwargs):
        assert kwargs == {"start_new_session": True}
        return SimpleNamespace(poll=lambda: 130, returncode=130)
    monkeypatch.setattr(cli.subprocess, "Popen", popen)
    monkeypatch.setattr(cli, "request_stop", lambda unit: calls.append(unit) or True)
    monkeypatch.setattr(cli, "terminal_state", lambda unit: {"terminal": True})
    assert cli.run_command(["no-launch"], "own-uuid", [2]) == 130
    assert calls == ["own-uuid"]


def test_disk_limit_and_worker_outside_supervision_rejected(control, monkeypatch):
    monkeypatch.setattr(cli, "disk_usage", lambda: {"new_bytes": 60*cli.GIB, "free_bytes": 100*cli.GIB})
    with pytest.raises(RuntimeError, match="storage"):
        cli.check_disk()
    monkeypatch.setattr(cli, "disk_usage", lambda: {"new_bytes": 0, "free_bytes": 79*cli.GIB})
    with pytest.raises(RuntimeError, match="storage"):
        cli.check_disk()
    write_json(control/"attempt-0000.launch.json", {"unit": "fixture"})
    monkeypatch.setenv("PHIDEUS_PREPARATION_UNIT", "fixture")
    with pytest.raises(RuntimeError, match="supervised"):
        cli.worker("attempt-0000")
    with pytest.raises(ValueError, match="identifier"):
        cli.worker("../escape")
