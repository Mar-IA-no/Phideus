"""OS-isolated executor for the Wave 49 public selector package."""

from __future__ import annotations

import json
import os
import pwd
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from .wave49_schema import ProtocolConfig, SPLITS, read_jsonl, sha256_file, write_json


def _drop_to(uid: int, gid: int):
    def demote() -> None:
        os.setgroups([])
        os.setgid(gid)
        os.setuid(uid)

    return demote


def run_restricted_executor(
    output_dir: Path,
    config: ProtocolConfig,
    worker_path: Path,
    repo_root: Path,
) -> dict[str, object]:
    """Execute selectors as ``nobody`` with only a staged public package.

    The real sealed tree remains root-owned mode 0700. The worker receives its
    path solely as a negative access probe and must record ``PermissionError``.
    """
    if os.geteuid() != 0:
        raise RuntimeError("Wave 49 sealed execution requires root to enforce a distinct OS identity")
    output_dir = Path(output_dir).resolve()
    sealed_dir = output_dir / "sealed"
    sealed_dir.chmod(0o700)
    account = pwd.getpwnam("nobody")

    with tempfile.TemporaryDirectory(prefix="wave49-public-", dir="/tmp") as raw_stage:
        stage = Path(raw_stage)
        visible_stage = stage / "visible"
        predictions_stage = stage / "predictions"
        shutil.copytree(output_dir / "visible", visible_stage)
        shutil.copy2(output_dir / "protocol_config.json", stage / "protocol_config.json")
        predictions_stage.mkdir()
        for path in (stage, visible_stage, predictions_stage):
            path.chmod(0o755 if path != predictions_stage else 0o700)
        (stage / "protocol_config.json").chmod(0o644)
        os.chown(predictions_stage, account.pw_uid, account.pw_gid)
        for path in visible_stage.rglob("*"):
            path.chmod(0o644 if path.is_file() else 0o755)

        command = [
            sys.executable,
            str(Path(worker_path).resolve()),
            "--visible-dir",
            str(visible_stage),
            "--predictions-dir",
            str(predictions_stage),
            "--config",
            str(stage / "protocol_config.json"),
            "--sealed-probe",
            str(sealed_dir / "identity_secret.json"),
        ]
        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path(repo_root).resolve() / "src")
        completed = subprocess.run(
            command,
            cwd="/tmp",
            env=env,
            text=True,
            capture_output=True,
            check=False,
            preexec_fn=_drop_to(account.pw_uid, account.pw_gid),
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"restricted executor failed ({completed.returncode}): {completed.stderr.strip()}"
            )
        receipt_path = predictions_stage / "access_receipt.json"
        if not receipt_path.exists():
            raise RuntimeError("restricted executor did not emit an access receipt")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("effective_uid") == 0 or not receipt.get("sealed_probe_denied"):
            raise RuntimeError("restricted executor boundary was not demonstrated")

        predictions_dir = output_dir / "predictions"
        if predictions_dir.exists():
            shutil.rmtree(predictions_dir)
        shutil.copytree(predictions_stage, predictions_dir)

    expected = {split: len(read_jsonl(output_dir / "visible" / f"{split}.jsonl")) for split in SPLITS}
    actual = {
        split: len(read_jsonl(output_dir / "predictions" / f"{split}.jsonl"))
        for split in SPLITS
    }
    receipt["visible_fixture_counts"] = expected
    receipt["prediction_row_counts"] = actual
    receipt["worker_sha256"] = sha256_file(Path(worker_path))
    receipt["boundary_method"] = "setuid-nobody-over-public-only-staging"
    receipt["orchestrator_verified"] = True
    write_json(output_dir / "predictions" / "access_receipt.json", receipt)
    return receipt
