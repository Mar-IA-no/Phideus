"""Prepare one open split in a bounded CPU subprocess; no held-out test option."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.atencion_armonica.partial_compatibility_cache import prepare_open_split


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True, choices=("development", "train", "validation"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        prepare_open_split(args.output, args.split)
    else:
        subprocess.run([sys.executable, __file__, "--split", args.split, "--output", str(args.output),
                        "--worker"], timeout=125, check=True)
