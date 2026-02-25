#!/usr/bin/env python3
"""
Gate 5B — Test 4: Transposition Invariance.

If the model learns from pitch ratios (relative intervals), transposing MIDI
by ±N semitones should NOT change the audio-MIDI matching quality much.
For D0 (no ratio injection), transposition should degrade performance more.

Method:
  1. Extract audio embeddings normally (unchanged)
  2. For each transposition [-6, -3, -1, +1, +3, +6] semitones:
     - Transpose midi_pitch (valid positions only) + clamp to [0, 127]
     - Extract MIDI embeddings with transposed input
     - Evaluate Audio→MIDI_transposed retrieval
  3. Compare S(transposed) vs S(original) per model

Usage:
    python experiments/bias_control/gate5b/test04_transposition.py \
        --model models/gate5b/d4a4/best_model.pt

    python experiments/bias_control/gate5b/test04_transposition.py --all
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any, Tuple, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

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

TRANSPOSITIONS = [-6, -3, -1, 0, +1, +3, +6]


@torch.no_grad()
def extract_audio_embeddings(model, dataset, device, batch_size, num_workers):
    """Extract audio embeddings only (no MIDI)."""
    from src.bias_control.datasets.maestro_segments import collate_segments

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_segments,
        pin_memory=True, prefetch_factor=2,
    )

    audio_embs = []
    for batch in tqdm(loader, desc="Extracting audio embeddings"):
        audio = batch['audio'].to(device, non_blocking=True)
        midi_pitch = batch['midi_pitch'].to(device, non_blocking=True)
        midi_velocity = batch['midi_velocity'].to(device, non_blocking=True)
        midi_duration = batch['midi_duration'].to(device, non_blocking=True)
        midi_mask = batch['midi_mask'].to(device, non_blocking=True)

        z_audio, _ = model(audio, midi_pitch, midi_velocity, midi_duration, midi_mask)
        audio_embs.append(z_audio.cpu())

    return torch.cat(audio_embs, dim=0)


@torch.no_grad()
def extract_midi_embeddings_transposed(model, dataset, device, batch_size, num_workers, shift):
    """Extract MIDI embeddings with transposed pitch."""
    from src.bias_control.datasets.maestro_segments import collate_segments

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_segments,
        pin_memory=True, prefetch_factor=2,
    )

    midi_embs = []
    for batch in tqdm(loader, desc=f"Extracting MIDI (shift={shift:+d})"):
        audio = batch['audio'].to(device, non_blocking=True)
        midi_pitch = batch['midi_pitch'].to(device, non_blocking=True)
        midi_velocity = batch['midi_velocity'].to(device, non_blocking=True)
        midi_duration = batch['midi_duration'].to(device, non_blocking=True)
        midi_mask = batch['midi_mask'].to(device, non_blocking=True)

        # Transpose: clone, shift valid positions, clamp
        pitch_t = midi_pitch.clone()
        valid = ~midi_mask  # True = valid (mask is True=padding)
        pitch_t[valid] = (pitch_t[valid] + shift).clamp(0, 127)

        _, z_midi = model(audio, pitch_t, midi_velocity, midi_duration, midi_mask)
        midi_embs.append(z_midi.cpu())

    return torch.cat(midi_embs, dim=0)


def evaluate_retrieval_from_embeddings(audio_embs, midi_embs, dataset, index, seed=42):
    """Run evaluation with pre-computed embeddings."""
    from experiments.bias_control.evaluate_structured_pool import (
        evaluate_with_precomputed_embeddings, PoolConfig,
    )

    config = PoolConfig(pool_size=256, n_hard_negatives=64,
                        n_semi_hard_negatives=32, n_queries=500)

    a2m = evaluate_with_precomputed_embeddings(
        audio_embs, midi_embs, dataset, index, config,
        direction='a2m', seed=seed,
    )
    m2a = evaluate_with_precomputed_embeddings(
        audio_embs, midi_embs, dataset, index, config,
        direction='m2a', seed=seed,
    )

    S = min(a2m['mean_recall@10'], m2a['mean_recall@10'])
    return S, a2m['mean_recall@10'], m2a['mean_recall@10']


def evaluate_single(model_path, maestro_dir, device, num_workers, seed):
    """Run transposition invariance test on one checkpoint."""
    from experiments.bias_control.gate5b.harness import (
        setup_gate5b_test, get_output_dir, save_test_result,
        load_cached_embeddings,
    )
    from experiments.bias_control.gate5b.checkpoint_loader import get_eval_batch_size

    class Args:
        pass
    args = Args()
    args.model = model_path
    args.maestro_dir = maestro_dir
    args.device = device
    args.seed = seed
    args.num_workers = num_workers
    args.output = None

    model, meta, dataset, index = setup_gate5b_test(args)
    descriptor = meta['descriptor']
    batch_size = get_eval_batch_size(descriptor)
    output_dir = get_output_dir(args, meta)

    dev = torch.device(device if torch.cuda.is_available() else 'cpu')

    # Try to load cached normal embeddings (audio + midi for shift=0)
    cached_audio, cached_midi = None, None
    try:
        cached_audio, cached_midi = load_cached_embeddings(output_dir)
        logger.info("Using cached audio embeddings (skip extraction)")
    except FileNotFoundError:
        pass

    # Audio embeddings: from cache or extract
    if cached_audio is not None:
        audio_embs = cached_audio
    else:
        logger.info("Extracting audio embeddings (once)...")
        audio_embs = extract_audio_embeddings(model, dataset, dev, batch_size, num_workers)

    # Evaluate each transposition
    results_by_shift = {}
    for shift in TRANSPOSITIONS:
        logger.info(f"\nTransposition: {shift:+d} semitones")

        # shift=0: use cached midi embeddings if available
        if shift == 0 and cached_midi is not None:
            logger.info("  Using cached MIDI embeddings for shift=0")
            midi_embs = cached_midi
        else:
            midi_embs = extract_midi_embeddings_transposed(
                model, dataset, dev, batch_size, num_workers, shift,
            )

        S, a2m, m2a = evaluate_retrieval_from_embeddings(
            audio_embs, midi_embs, dataset, index, seed,
        )
        results_by_shift[shift] = {
            'S': S,
            'a2m_r10': a2m,
            'm2a_r10': m2a,
        }
        logger.info(f"  shift={shift:+d}: S={S:.1%}")

    # Compute degradation relative to shift=0
    S_baseline = results_by_shift[0]['S']
    for shift, data in results_by_shift.items():
        data['delta_S'] = data['S'] - S_baseline

    result = {
        'S_baseline': S_baseline,
        'transpositions': {str(k): v for k, v in results_by_shift.items()},
        'degradation_curve': {
            str(k): v['delta_S'] for k, v in results_by_shift.items()
        },
    }

    output_dir = get_output_dir(args, meta)
    save_test_result(result, output_dir, 'test04_transposition', meta, seed=seed)

    return result, meta


def main():
    parser = argparse.ArgumentParser(description='Gate 5B — Test 4: Transposition Invariance')
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--maestro-dir', type=str, default='data/maestro_v3/maestro-v3.0.0')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num-workers', type=int, default=8)

    args = parser.parse_args()

    if not args.model and not args.all:
        parser.error("Provide --model PATH or --all")

    models = {}
    if args.all:
        models = dict(DEFAULT_CHECKPOINTS)
    else:
        models = {'custom': args.model}

    all_results = {}

    for arm, path in models.items():
        if not Path(path).exists():
            logger.warning(f"Checkpoint not found: {path} — skipping {arm}")
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"TRANSPOSITION INVARIANCE: {arm}")
        logger.info(f"{'='*60}")

        start = time.time()
        result, meta = evaluate_single(
            path, args.maestro_dir, args.device, args.num_workers, args.seed,
        )
        elapsed = time.time() - start

        all_results[arm] = {
            'descriptor': meta['descriptor'],
            'S_baseline': result['S_baseline'],
            'curve': result['degradation_curve'],
            'time_s': elapsed,
        }

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("TRANSPOSITION INVARIANCE SUMMARY")
    logger.info(f"{'='*60}")

    header = f"{'Arm':<10} {'S(0)':>6}"
    for shift in TRANSPOSITIONS:
        if shift != 0:
            header += f" {'d(' + str(shift) + ')':>8}"
    logger.info(header)
    logger.info("-" * (10 + 6 + 8 * (len(TRANSPOSITIONS) - 1)))

    for arm, data in all_results.items():
        line = f"{arm:<10} {data['S_baseline']:>5.1%}"
        for shift in TRANSPOSITIONS:
            if shift != 0:
                delta = data['curve'].get(str(shift), 0)
                line += f" {delta:>+7.1%}"
        logger.info(line)


if __name__ == '__main__':
    main()
