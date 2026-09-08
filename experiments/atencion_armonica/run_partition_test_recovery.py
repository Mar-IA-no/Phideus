"""Explicit CPU-only entrypoint for audited versioned test recovery."""
import argparse
import json
import os
from pathlib import Path
import resource
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path)
    parser.add_argument("--sha256")
    parser.add_argument("--worker-fd", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    from src.atencion_armonica.partition_test_recovery_supervisor import execute, supervised
    if args.worker_fd is not None:
        if args.request is not None or args.sha256 is not None:
            parser.error("worker only accepts its inherited request pipe")
        with os.fdopen(args.worker_fd) as handle:
            record = json.load(handle)
        if set(record) != {"reference", "supervisor_pid"}:
            raise ValueError("recovery worker envelope differs")
        from src.atencion_armonica.learned_partition_supervisor import arm_parent_death
        arm_parent_death(record["supervisor_pid"])
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
            raise PermissionError("recovery worker must not expose CUDA")
        result = execute(record["reference"])
        print(json.dumps({"result": result, "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}))
    else:
        if args.request is None or args.sha256 is None:
            parser.error("both immutable request path and SHA256 are required")
        ref = {"path": args.request.resolve().relative_to(ROOT).as_posix(), "sha256": args.sha256}
        print(json.dumps(supervised(ref)))


if __name__ == "__main__":
    main()
