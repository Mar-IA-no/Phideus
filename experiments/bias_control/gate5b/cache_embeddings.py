#!/usr/bin/env python3
"""
Gate 5B — Cache normal embeddings for all models.

Extracts and saves embeddings once so all tests can reuse them.
Skip models that already have cached embeddings (use --force to re-extract).

Usage:
    python experiments/bias_control/gate5b/cache_embeddings.py --all
    python experiments/bias_control/gate5b/cache_embeddings.py --model models/gate5b/d4a4/best_model.pt
"""

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINTS = {
    'D0': 'models/gate5b/D0/best_model.pt',
    'd4a4': 'models/gate5b/d4a4/best_model.pt',
    'a4r': 'models/gate5b/a4r/best_model.pt',
    'd4-a4r': 'models/gate5b/d4-a4r/best_model.pt',
}


def main():
    parser = argparse.ArgumentParser(description='Gate 5B — Cache normal embeddings')
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--maestro-dir', type=str, default='data/maestro_v3/maestro-v3.0.0')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--force', action='store_true', help='Re-extract even if cache exists')
    args = parser.parse_args()

    if not args.model and not args.all:
        parser.error("Provide --model PATH or --all")

    from experiments.bias_control.gate5b.harness import (
        setup_gate5b_test, get_normal_embeddings, get_output_dir,
        _embeddings_cache_path,
    )

    models = {}
    if args.all:
        models = dict(DEFAULT_CHECKPOINTS)
    else:
        models = {'custom': args.model}

    for arm, path in models.items():
        if not Path(path).exists():
            logger.warning(f"Checkpoint not found: {path} — skipping {arm}")
            continue

        # Setup
        class Args:
            pass
        a = Args()
        a.model = path
        a.maestro_dir = args.maestro_dir
        a.device = args.device
        a.seed = args.seed
        a.num_workers = args.num_workers
        a.output = None

        model, meta, dataset, index = setup_gate5b_test(a)
        output_dir = get_output_dir(a, meta)

        # Check cache
        cache_path = _embeddings_cache_path(output_dir)
        if cache_path.exists() and not args.force:
            logger.info(f"SKIP {arm}: cache exists at {cache_path}")
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"CACHING: {arm} (descriptor={meta['descriptor']})")
        logger.info(f"{'='*60}")

        start = time.time()
        audio_embs, midi_embs = get_normal_embeddings(
            model, dataset, args.device, meta['descriptor'],
            output_dir, num_workers=args.num_workers,
            force_extract=True,
        )
        elapsed = time.time() - start
        logger.info(f"  Cached {audio_embs.shape[0]} segments in {elapsed:.1f}s")

    logger.info("Done caching embeddings.")


if __name__ == '__main__':
    main()
