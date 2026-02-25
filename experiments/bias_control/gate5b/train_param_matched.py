#!/usr/bin/env python3
"""
Gate 5B — Test 2: Parameter-Matched Training with Ablated Descriptors.

Trains a d4a4-architecture model (~66.5M params) with crippled descriptors
to control that d4a4's improvement is causal (from ratio info, not just params).

Ablation modes:
  - random:   Compute real descriptor, replace with randn_like (matched mean/std)
  - shuffled: Compute real descriptor, permute across batch dimension
  - zero:     Return zeros_like (params exist but no signal)

Strategy: monkey-patch compute_audio_descriptor_a4 and
compute_local_interval_features in the gate43_scratch_training module
namespace BEFORE calling main(). All callers within that module
(including _encode_audio_with_descriptor, _encode_midi_with_intervals,
and the model classes) resolve the functions from the module namespace
at call time, so the patches persist through the entire training run.

Usage (via SLURM):
    python experiments/bias_control/gate5b/train_param_matched.py \
        --ablation-mode random \
        --output /home/mfmendez/results/gate5b_param_matched/random \
        --maestro-dir /scratch/12345/maestro-v3.0.0 \
        -- --epochs 30 --batch-size 16 --from-scratch ...

Everything after '--' is passed through to gate43_scratch_training.main().
"""

import logging
import sys
from pathlib import Path

# Ensure repo root is importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('train_param_matched')


import torch


def make_ablation_wrapper(original_fn, mode):
    """
    Create a wrapper that computes the real descriptor then ablates it.

    Args:
        original_fn: the original compute_* function
        mode: 'random', 'shuffled', or 'zero'
    """
    call_count = [0]

    def wrapper(*args, **kwargs):
        result = original_fn(*args, **kwargs)
        call_count[0] += 1

        if mode == 'zero':
            return torch.zeros_like(result)
        elif mode == 'random':
            # Match per-batch mean/std so the noise has similar scale
            std = result.std()
            mean = result.mean()
            return torch.randn_like(result) * std + mean
        elif mode == 'shuffled':
            B = result.size(0)
            if B > 1:
                perm = torch.randperm(B, device=result.device)
                return result[perm]
            return result
        else:
            raise ValueError(f"Unknown ablation mode: {mode}")

    wrapper._ablation_mode = mode
    wrapper._original_fn = original_fn
    wrapper._call_count = call_count
    return wrapper


def main():
    # Split argv at '--': our args before, training args after
    our_argv = []
    training_argv = []

    if '--' in sys.argv[1:]:
        split_idx = sys.argv.index('--')
        our_argv = sys.argv[1:split_idx]
        training_argv = sys.argv[split_idx + 1:]
    else:
        our_argv = sys.argv[1:]

    # Parse our args manually (simple)
    import argparse
    parser = argparse.ArgumentParser(
        description='Gate 5B — Train with ablated descriptors'
    )
    parser.add_argument(
        '--ablation-mode', type=str, required=True,
        choices=['random', 'shuffled', 'zero'],
        help='How to cripple the descriptors'
    )
    args = parser.parse_args(our_argv)

    mode = args.ablation_mode
    logger.info(f"=== Parameter-Matched Training: ablation_mode={mode} ===")
    logger.info(f"Training args: {training_argv}")

    # Import training module
    import experiments.bias_control.gate43_scratch.gate43_scratch_training as g43

    # Save originals
    orig_a4 = g43.compute_audio_descriptor_a4
    orig_d4 = g43.compute_local_interval_features

    # Patch with ablation wrappers
    g43.compute_audio_descriptor_a4 = make_ablation_wrapper(orig_a4, mode)
    logger.info(f"Patched compute_audio_descriptor_a4 → {mode} ablation")

    g43.compute_local_interval_features = make_ablation_wrapper(orig_d4, mode)
    logger.info(f"Patched compute_local_interval_features → {mode} ablation")

    # Build sys.argv for the training script's argparse
    sys.argv = ['gate43_scratch_training.py'] + training_argv

    # Run training
    logger.info("Calling gate43_scratch_training.main()...")
    g43.main()

    # Report call counts
    a4_calls = g43.compute_audio_descriptor_a4._call_count[0]
    d4_calls = g43.compute_local_interval_features._call_count[0]
    logger.info(f"Ablation wrapper call counts: a4={a4_calls}, d4={d4_calls}")
    logger.info(f"=== Parameter-Matched Training DONE: mode={mode} ===")


if __name__ == '__main__':
    main()
