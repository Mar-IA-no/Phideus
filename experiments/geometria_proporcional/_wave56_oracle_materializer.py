#!/usr/bin/env python3
"""Root-only, split-scoped oracle materializer for Wave 56 Stage 1."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import shutil
import stat
import tempfile

from geometria_proporcional.wave49_oracle import compute_oracle_splits
from geometria_proporcional.wave49_schema import ProtocolConfig, read_jsonl, sha256_file

SPLIT_ROLES = {
    "train": "gate_fit",
    "val": "gate_select",
    "lockbox": "sealed_monitor",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "lockbox"), required=True)
    parser.add_argument("--role", choices=("gate_fit", "gate_select", "sealed_monitor"), required=True)
    parser.add_argument("--destination", type=Path, required=True)
    return parser.parse_args()


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_json(path: Path, payload: object, mode: int = 0o600) -> None:
    encoded = (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode()
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        fsync_directory(path.parent)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def validate_scope(benchmark: Path, config: dict[str, object], split: str, role: str) -> Path:
    if os.geteuid() != 0 or os.getegid() != 0:
        raise PermissionError("oracle materializer must run as root")
    if config.get("physical_splits") != SPLIT_ROLES:
        raise RuntimeError("prospective config split-role table differs from materializer freeze")
    expected_role = SPLIT_ROLES[split]
    if role != expected_role:
        raise ValueError(f"split {split!r} is frozen to role {expected_role!r}")
    sealed = benchmark / "sealed"
    sealed_stat = sealed.stat()
    if stat.S_IMODE(sealed_stat.st_mode) != 0o700 or sealed_stat.st_uid != 0:
        raise PermissionError("benchmark sealed directory must be root-owned mode 0700")
    truth = sealed / f"{split}.jsonl"
    if truth.is_symlink() or not truth.is_file():
        raise RuntimeError("requested sealed truth must be one regular file")
    return truth


def main() -> None:
    os.umask(0o077)
    args = parse_args()
    benchmark = args.benchmark_root.resolve(strict=True)
    config_path = args.config.resolve(strict=True)
    destination = args.destination.resolve()
    if destination.exists():
        raise FileExistsError(destination)
    if destination.is_relative_to(benchmark):
        raise ValueError("authorized labels must be staged outside the benchmark root")
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if destination.parent.is_symlink():
        raise RuntimeError("materialization destination parent cannot be a symlink")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    truth = validate_scope(benchmark, config, args.split, args.role)
    protocol_path = benchmark / "protocol_config.json"
    protocol = ProtocolConfig.from_dict(json.loads(protocol_path.read_text(encoding="utf-8")))
    manifest_path = benchmark / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    truth_relative = f"sealed/{args.split}.jsonl"
    frozen_truth = manifest.get("files", {}).get(truth_relative)
    observed_truth = {"sha256": sha256_file(truth), "bytes": truth.stat().st_size}
    if not isinstance(frozen_truth, dict) or observed_truth != frozen_truth:
        raise RuntimeError("requested sealed truth differs from the frozen benchmark manifest")

    pending = Path(tempfile.mkdtemp(prefix=f".{destination.name}.pending.", dir=destination.parent))
    try:
        counts = compute_oracle_splits(benchmark, protocol, (args.split,), pending)
        label = pending / f"{args.split}.jsonl"
        if set(path.name for path in pending.iterdir()) != {label.name}:
            raise RuntimeError("materializer produced material outside its split scope")
        rows = read_jsonl(label)
        if any(row.get("split") != args.split for row in rows):
            raise RuntimeError("materialized oracle escaped requested split")
        label.chmod(0o600)
        with label.open("rb") as handle:
            os.fsync(handle.fileno())
        receipt = {
            "phase": "wave56-split-scoped-oracle-materialization",
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "effective_uid": os.geteuid(),
            "effective_gid": os.getegid(),
            "split": args.split,
            "role": args.role,
            "count": counts[args.split],
            "benchmark_manifest_sha256": sha256_file(manifest_path),
            "sealed_truth_expected": frozen_truth,
            "sealed_truth_observed": observed_truth,
            "sealed_truth_sha256": observed_truth["sha256"],
            "protocol_config_sha256": sha256_file(protocol_path),
            "prospective_config_sha256": sha256_file(config_path),
            "authorized_labels_sha256": sha256_file(label),
            "other_splits_materialized": False,
        }
        atomic_json(pending / "materialization_receipt.json", receipt)
        fsync_directory(pending)
        os.replace(pending, destination)
        fsync_directory(destination.parent)
        published_label = destination / f"{args.split}.jsonl"
        published_receipt = destination / "materialization_receipt.json"
        if sha256_file(published_label) != receipt["authorized_labels_sha256"]:
            raise RuntimeError("published label hash differs after atomic promotion")
        if not published_receipt.is_file():
            raise RuntimeError("materialization receipt missing after atomic promotion")
    except BaseException:
        shutil.rmtree(pending, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
