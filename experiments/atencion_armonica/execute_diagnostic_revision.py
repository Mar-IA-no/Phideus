"""Reviewed complete diagnostic/replay; no budget override or additional profile."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.atencion_armonica import diagnostic_execution_revision as revised


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("snapshot", "run", "replay"))
    parser.add_argument("--review-path")
    parser.add_argument("--review-sha256")
    args = parser.parse_args()
    if args.operation == "snapshot":
        store = revised.base.DiagnosticStore(ROOT/revised.OUTPUT, project_root=ROOT)
        print(revised.encoded(revised.snapshot(store)).decode(), end="")
        return
    if not args.review_path or not args.review_sha256:
        parser.error("a reviewed snapshot path and SHA256 are required")
    try:
        result = revised.execute(args.operation, {"path": args.review_path, "sha256": args.review_sha256})
        print(json.dumps(result), flush=True)
    except revised.base.Paused as exc:
        print(str(exc), flush=True)
        raise SystemExit(75)
    except revised.base.BudgetExceeded as exc:
        print(str(exc), flush=True)
        raise SystemExit(76)


if __name__ == "__main__":
    main()
