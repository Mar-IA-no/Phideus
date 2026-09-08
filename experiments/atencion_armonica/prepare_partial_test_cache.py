"""CPU preparation of one test split, only after the prospective evaluation freeze."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.atencion_armonica.partial_compatibility_test_cache import prepare_test_split
from src.atencion_armonica.partial_compatibility_test_gate import TEST_SPLITS


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=TEST_SPLITS, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        prepare_test_split(args.output, args.split, args.freeze)
    else:
        subprocess.run([sys.executable, __file__, "--output", str(args.output), "--split", args.split,
                        "--freeze", str(args.freeze), "--worker"], check=True, timeout=125)
