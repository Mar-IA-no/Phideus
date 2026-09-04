#!/usr/bin/env python3
"""Canonical Wave 57 entrypoint over the audited transactional coordinator."""

from __future__ import annotations

from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_wave56_contextual_gate as shared  # noqa: E402


CONFIG = HERE / "configs/wave57_contextual_tail_guard_fresh.json"


def main() -> None:
    if "--config" not in sys.argv:
        sys.argv.extend(("--config", str(CONFIG)))
    shared.main()


if __name__ == "__main__":
    main()
