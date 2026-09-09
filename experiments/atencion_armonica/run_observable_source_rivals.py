"""Explicit stages; invoking the CLI never chooses CUDA automatically."""
import argparse
import json
import os
from pathlib import Path
import sys

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.atencion_armonica import observable_rival_campaign as campaign


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage", choices=("profile", "inventory", "initialize", "run", "evaluate",
                                          "replay", "evaluation-replay", "profile-compare"))
    parser.add_argument("--root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--second-profile", type=Path)
    parser.add_argument("--review", type=Path, action="append", default=[])
    args = parser.parse_args()
    if args.stage in ("profile", "inventory", "profile-compare") and args.output is None:
        parser.error("this stage requires --output (immutable new artifact)")
    if args.stage not in ("profile", "inventory", "profile-compare") and args.root is None:
        parser.error("this stage requires --root")
    if args.stage == "profile":
        result = campaign.write_once(args.output, campaign.profile(args.device))
    elif args.stage == "inventory":
        result = campaign.write_once(args.output, {"scenes": campaign.load_observable_inventory()})
    elif args.stage == "profile-compare":
        if args.profile is None or args.second_profile is None:
            parser.error("profile-compare requires both profile paths")
        result = campaign.write_once(args.output, campaign.compare_profiles(
            campaign.reference(args.profile), campaign.reference(args.second_profile)))
    elif args.stage == "initialize":
        if args.profile is None or not args.review:
            parser.error("initialize requires a mechanical --profile and --review evidence")
        result = campaign.initialize(args.root, args.device, campaign.reference(args.profile),
                                     [campaign.reference(p) for p in args.review])
    elif args.stage == "run":
        result = campaign.run(args.root)
    elif args.stage == "replay":
        result = campaign.replay(args.root)
        if args.output:
            campaign.write_once(args.output, result)
    else:
        from src.atencion_armonica import observable_rival_evaluation as evaluation
        result = evaluation.run(args.root, replay=args.stage == "evaluation-replay")
        if args.output:
            campaign.write_once(args.output, result)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
