#!/usr/bin/env python3
"""
Gate 5B — Test 2: Parameter-Matched Ablations.

Controls that d4a4's improvement is NOT merely from additional parameters.
Trains 3 models with ~66.5M params (= d4a4) but with crippled descriptors:

  a. random_injection:   descriptor replaced by Gaussian noise
  b. shuffled_injection: real descriptor but shuffled across batch samples
  c. zero_injection:     descriptor zeroed (params exist but no signal)

Each is trained for 30 epochs, seed=42, same config as d4a4.
If d4a4's advantage is causal (from ratio info), ablations should
perform significantly worse (~D0 level: 73%).

NOTE: This test requires training. Intended for UNC (SLURM).
See gate5b_slurm/ for job scripts.

Usage (local, for reference):
    python experiments/bias_control/gate5b/test02_param_matched.py \
        --mode random --epochs 30 --output data/gate5b_results/param_matched/random
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Gate 5B — Test 2: Parameter-Matched Ablations')
    parser.add_argument('--mode', type=str, required=True,
                        choices=['random', 'shuffled', 'zero'],
                        help='Ablation mode: random/shuffled/zero injection')
    parser.add_argument('--maestro-dir', type=str, default='data/maestro_v3/maestro-v3.0.0')
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num-workers', type=int, default=8)

    args = parser.parse_args()

    logger.info(f"Test 2: Parameter-Matched Ablation mode={args.mode}")
    logger.info("This test requires training (30 epochs). Use SLURM for UNC execution.")
    logger.info("SLURM script: experiments/bias_control/slurm/gate5b_param_matched.sh")

    # The actual training would monkey-patch descriptor functions similar to test01,
    # but DURING training rather than evaluation.
    # Implementation deferred to SLURM submission.

    logger.info("Stub: training not implemented in this script.")
    logger.info("Use gate43_scratch_training.py with modified descriptor functions.")


if __name__ == '__main__':
    main()
