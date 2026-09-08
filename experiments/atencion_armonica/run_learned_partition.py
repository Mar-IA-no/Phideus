"""Explicit hash-bound learned-reader stages; no default operation."""
from __future__ import annotations

import argparse
import json
import os
import resource
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.atencion_armonica.learned_partition_supervisor import execute, supervised, arm_parent_death


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path)
    parser.add_argument("--sha256")
    parser.add_argument("--reconcile-permit", type=Path,
                        help="CPU-only recovery of a dead supervisor's canonical budget reservation")
    parser.add_argument("--worker-fd", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_fd is not None:
        if args.request is not None or args.sha256 is not None or args.reconcile_permit is not None:
            parser.error("worker reads only its inherited request pipe")
        with os.fdopen(args.worker_fd) as handle:
            envelope = json.load(handle)
        if set(envelope) != {"reference", "permit", "supervisor_pid"}:
            raise ValueError("invalid inherited request envelope")
        arm_parent_death(envelope["supervisor_pid"])
        result = execute(envelope["reference"], permit=envelope["permit"])
        print(json.dumps({"result": result,
                          "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}, sort_keys=True))
    elif args.reconcile_permit is not None:
        if args.request is not None or args.sha256 is None:
            parser.error("reconciliation requires only --reconcile-permit and --sha256")
        from src.atencion_armonica.learned_partition_budget import reconcile
        ref = {"path": args.reconcile_permit.resolve().relative_to(ROOT).as_posix(), "sha256": args.sha256}
        print(json.dumps({"supervisor": reconcile(ref)}, sort_keys=True))
    else:
        if args.request is None or args.sha256 is None:
            parser.error("request and expected SHA256 are both required")
        ref = {"path": args.request.resolve().relative_to(ROOT).as_posix(), "sha256": args.sha256}
        print(json.dumps(supervised(ref), sort_keys=True))


if __name__ == "__main__":
    main()
