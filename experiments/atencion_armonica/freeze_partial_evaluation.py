"""Create the prospective test freeze after all readers and implementation audit close."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.atencion_armonica.partial_compatibility_test_gate import create_freeze


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "training", "selection", "raw-validation", "audit"):
        parser.add_argument("--"+name, required=True, type=Path)
    args = parser.parse_args()
    create_freeze(args.output, args.training, args.selection, args.raw_validation, args.audit)
