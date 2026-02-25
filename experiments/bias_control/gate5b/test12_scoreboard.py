#!/usr/bin/env python3
"""
Gate 5B — Test 12: Gate Scoreboard.

Re-evaluates all 4 Gate 5B checkpoints with canonical pinned configuration.
Primary purpose: validate that the universal checkpoint loader correctly
reconstructs each model (S values must match historical records).

Canonical config:
    pool_size=256, n_queries=500, n_hard=64, n_semi_hard=32, seed=42

Expected values:
    D0:     ~73.4%  (ctail e50)
    d4a4:   ~83.8%  (60ep cosine e50, ALL-TIME RECORD)
    a4r:    ~82.0%  (30ep e29)
    d4-a4r: ~79.8%  (30ep e30)

Usage:
    # Single model
    python experiments/bias_control/gate5b/test12_scoreboard.py \
        --model models/gate5b/d4a4/best_model.pt

    # All 4 models
    python experiments/bias_control/gate5b/test12_scoreboard.py --all
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Pinned canonical evaluation config — never change without versioning
CANONICAL_EVAL_CONFIG = {
    'pool_size': 256,
    'n_queries': 500,
    'n_hard_negatives': 64,
    'n_semi_hard_negatives': 32,
    'seed': 42,
}

# Expected S values for validation (± tolerance)
EXPECTED_S = {
    'd0': 0.734,
    'd4a4': 0.838,
    'a4r': 0.820,
    'd4-a4r': 0.798,
}
S_TOLERANCE = 0.02  # ±2pp acceptable from eval variance

# Default checkpoint paths
DEFAULT_CHECKPOINTS = {
    'D0': 'models/gate5b/D0/best_model.pt',
    'd4a4': 'models/gate5b/d4a4/best_model.pt',
    'a4r': 'models/gate5b/a4r/best_model.pt',
    'd4-a4r': 'models/gate5b/d4-a4r/best_model.pt',
}


def evaluate_single(
    model_path: str,
    maestro_dir: str,
    device: str,
    num_workers: int,
) -> Dict[str, Any]:
    """Evaluate a single checkpoint with canonical config."""
    from experiments.bias_control.gate5b.harness import (
        setup_gate5b_test, get_normal_embeddings, get_output_dir, save_test_result,
    )
    from experiments.bias_control.evaluate_structured_pool import (
        evaluate_with_precomputed_embeddings,
        analyze_hard_negatives_fast,
        PoolConfig,
    )

    # Setup
    class Args:
        pass
    args = Args()
    args.model = model_path
    args.maestro_dir = maestro_dir
    args.device = device
    args.seed = CANONICAL_EVAL_CONFIG['seed']
    args.num_workers = num_workers
    args.output = None

    model, meta, dataset, index = setup_gate5b_test(args)
    output_dir = get_output_dir(args, meta)

    # Extract embeddings (and cache for reuse by other tests)
    audio_embs, midi_embs = get_normal_embeddings(
        model, dataset, device, meta['descriptor'], output_dir,
        num_workers=num_workers,
    )

    # Evaluate with canonical config
    config = PoolConfig(
        pool_size=CANONICAL_EVAL_CONFIG['pool_size'],
        n_hard_negatives=CANONICAL_EVAL_CONFIG['n_hard_negatives'],
        n_semi_hard_negatives=CANONICAL_EVAL_CONFIG['n_semi_hard_negatives'],
        n_queries=CANONICAL_EVAL_CONFIG['n_queries'],
    )

    logger.info("Evaluating A2M...")
    a2m = evaluate_with_precomputed_embeddings(
        audio_embs, midi_embs, dataset, index, config,
        direction='a2m', seed=CANONICAL_EVAL_CONFIG['seed'],
    )

    logger.info("Evaluating M2A...")
    m2a = evaluate_with_precomputed_embeddings(
        audio_embs, midi_embs, dataset, index, config,
        direction='m2a', seed=CANONICAL_EVAL_CONFIG['seed'],
    )

    logger.info("Analyzing hard negatives...")
    hard_neg = analyze_hard_negatives_fast(
        audio_embs, midi_embs, dataset, index,
        n_samples=500, seed=CANONICAL_EVAL_CONFIG['seed'],
    )

    # Compute S = min(A2M_R@10, M2A_R@10)
    S = min(a2m['mean_recall@10'], m2a['mean_recall@10'])

    result = {
        'S': S,
        'a2m': a2m,
        'm2a': m2a,
        'hard_negatives': hard_neg,
        'canonical_config': CANONICAL_EVAL_CONFIG,
    }

    # Save
    output_dir = get_output_dir(args, meta)
    save_test_result(result, output_dir, 'test12_scoreboard', meta, seed=args.seed)

    return result, meta


def main():
    parser = argparse.ArgumentParser(description='Gate 5B — Test 12: Scoreboard')
    parser.add_argument('--model', type=str, default=None, help='Single checkpoint path')
    parser.add_argument('--all', action='store_true', help='Evaluate all 4 Gate 5B checkpoints')
    parser.add_argument('--maestro-dir', type=str, default='data/maestro_v3/maestro-v3.0.0')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num-workers', type=int, default=8)

    args = parser.parse_args()

    if not args.model and not args.all:
        parser.error("Provide --model PATH or --all")

    models = {}
    if args.all:
        models = dict(DEFAULT_CHECKPOINTS)
    else:
        models = {'custom': args.model}

    scoreboard = {}
    total_start = time.time()

    for arm, path in models.items():
        if not Path(path).exists():
            logger.warning(f"Checkpoint not found: {path} — skipping {arm}")
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"EVALUATING: {arm}")
        logger.info(f"{'='*60}")

        start = time.time()
        result, meta = evaluate_single(path, args.maestro_dir, args.device, args.num_workers)
        elapsed = time.time() - start

        descriptor = meta['descriptor']
        S = result['S']

        # Validation against expected
        expected = EXPECTED_S.get(descriptor)
        status = ''
        if expected is not None:
            delta = abs(S - expected)
            if delta <= S_TOLERANCE:
                status = 'PASS'
            else:
                status = f'WARN (delta={delta:.3f})'

        scoreboard[arm] = {
            'descriptor': descriptor,
            'epoch': meta['epoch'],
            'S': S,
            'a2m_r10': result['a2m']['mean_recall@10'],
            'm2a_r10': result['m2a']['mean_recall@10'],
            'hard_neg_acc': result['hard_negatives'].get('accuracy_vs_random'),
            'expected_S': expected,
            'validation': status,
            'eval_time_s': elapsed,
        }

        logger.info(f"\n  {arm}: S={S:.1%}, A2M={result['a2m']['mean_recall@10']:.1%}, "
                     f"M2A={result['m2a']['mean_recall@10']:.1%}, "
                     f"hard_neg={result['hard_negatives'].get('accuracy_vs_random', 0):.1%} "
                     f"[{status}] ({elapsed:.0f}s)")

    # Summary
    total_elapsed = time.time() - total_start
    logger.info(f"\n{'='*60}")
    logger.info("GATE 5B SCOREBOARD")
    logger.info(f"{'='*60}")
    logger.info(f"{'Arm':<10} {'S':>8} {'A2M':>8} {'M2A':>8} {'HardNeg':>8} {'Valid':>10} {'Time':>6}")
    logger.info("-" * 60)
    for arm, data in sorted(scoreboard.items(), key=lambda x: -x[1]['S']):
        logger.info(
            f"{arm:<10} {data['S']:>7.1%} "
            f"{data['a2m_r10']:>7.1%} {data['m2a_r10']:>7.1%} "
            f"{data['hard_neg_acc'] or 0:>7.1%} "
            f"{data['validation']:>10} {data['eval_time_s']:>5.0f}s"
        )
    logger.info(f"\nTotal time: {total_elapsed/60:.1f} min")

    # Save consolidated scoreboard
    scoreboard_path = Path('data/gate5b_results/scoreboard.json')
    scoreboard_path.parent.mkdir(parents=True, exist_ok=True)
    with open(scoreboard_path, 'w') as f:
        json.dump(scoreboard, f, indent=2)
    logger.info(f"Scoreboard saved: {scoreboard_path}")


if __name__ == '__main__':
    main()
