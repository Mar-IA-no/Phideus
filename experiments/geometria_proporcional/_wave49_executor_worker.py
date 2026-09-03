#!/usr/bin/env python3
"""Restricted Wave 49 worker. It intentionally imports no generator or oracle."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from geometria_proporcional.wave49_schema import ProtocolConfig, SPLITS, sha256_file, write_json
from geometria_proporcional.wave49_selector import execute_selectors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visible-dir", type=Path, required=True)
    parser.add_argument("--predictions-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--sealed-probe", type=Path, required=True)
    args = parser.parse_args()

    config = ProtocolConfig.from_dict(json.loads(args.config.read_text(encoding="utf-8")))
    denied = False
    try:
        args.sealed_probe.read_bytes()
    except PermissionError:
        denied = True
    if not denied:
        raise RuntimeError("sealed probe was readable by the restricted executor")

    counts = execute_selectors(args.visible_dir, args.predictions_dir, config)
    write_json(args.predictions_dir / "access_receipt.json", {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "actor": "wave49_restricted_executor",
        "effective_uid": os.geteuid(),
        "effective_gid": os.getegid(),
        "sealed_probe_denied": denied,
        "sealed_probe_name": args.sealed_probe.name,
        "operations": [
            {
                "operation": "calibrate",
                "split": "calibration_null",
                "input_scope": "visible/calibration_null.jsonl",
                "sealed_access": False,
            },
            *[
                {
                    "operation": "predict",
                    "split": split,
                    "input_scope": f"visible/{split}.jsonl",
                    "sealed_access": False,
                }
                for split in SPLITS
            ],
        ],
        "input_hashes": {
            path.name: sha256_file(path)
            for path in sorted(args.visible_dir.glob("*.jsonl"))
        },
        "prediction_counts": counts,
    })


if __name__ == "__main__":
    main()
