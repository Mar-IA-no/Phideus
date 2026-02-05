#!/usr/bin/env python3
"""
Structured Pool Evaluation for BIAS_CONTROL (OPTIMIZED)

Pre-extracts ALL embeddings once (~10-15 min), then runs fast in-memory evaluation.

Pool structure per query:
- 1 positive (same piece, same time)
- 64 hard negatives (same piece, different time)
- 32 semi-hard negatives (same composer, if available)
- Rest: random negatives from different pieces

Usage:
    python experiments/bias_control/evaluate_structured_pool.py \
        --model data/bias_control_medium/training_outputs/gate2/checkpoint_epoch45.pt \
        --maestro-dir data/maestro_v3/maestro-v3.0.0 \
        --output data/bias_control_medium/evaluations/structured_pool.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class PoolConfig:
    """Configuration for structured evaluation pool."""
    pool_size: int = 256
    n_hard_negatives: int = 64      # Same piece, different time
    n_semi_hard_negatives: int = 32  # Same composer
    n_queries: int = 500             # Number of queries to evaluate


def build_segment_index(dataset) -> Dict:
    """
    Build index for efficient negative sampling.

    Returns:
        {
            'by_piece': {piece_idx: [segment_indices]},
            'by_composer': {composer: [segment_indices]},
            'piece_to_composer': {piece_idx: composer}
        }
    """
    by_piece = {}
    by_composer = {}
    piece_to_composer = {}

    for i, seg in enumerate(dataset.segments):
        piece_idx = seg.piece_idx
        composer = getattr(seg, 'canonical_composer', 'Unknown')

        if piece_idx not in by_piece:
            by_piece[piece_idx] = []
        by_piece[piece_idx].append(i)

        if composer not in by_composer:
            by_composer[composer] = []
        by_composer[composer].append(i)

        piece_to_composer[piece_idx] = composer

    return {
        'by_piece': by_piece,
        'by_composer': by_composer,
        'piece_to_composer': piece_to_composer
    }


@torch.no_grad()
def extract_all_embeddings(
    model,
    dataset,
    device: torch.device,
    batch_size: int = 64,  # Optimized for RTX 3090 (24GB VRAM)
    num_workers: int = 8   # Optimized for i5-12600K (16 cores)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pre-extract ALL embeddings from the dataset.

    This is the key optimization: extract once, use many times.

    Returns:
        (audio_embeddings, midi_embeddings): Both [N, D] on CPU
    """
    from src.bias_control.datasets.maestro_segments import collate_segments

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_segments,
        pin_memory=True,
        prefetch_factor=2
    )

    audio_embs = []
    midi_embs = []

    n_batches = len(loader)
    logger.info(f"Extracting embeddings: {len(dataset)} segments, {n_batches} batches")

    start_time = time.time()

    for batch in tqdm(loader, desc="Extracting embeddings"):
        audio = batch['audio'].to(device, non_blocking=True)
        midi_pitch = batch['midi_pitch'].to(device, non_blocking=True)
        midi_velocity = batch['midi_velocity'].to(device, non_blocking=True)
        midi_duration = batch['midi_duration'].to(device, non_blocking=True)
        midi_mask = batch['midi_mask'].to(device, non_blocking=True)

        z_audio, z_midi = model(
            audio, midi_pitch, midi_velocity, midi_duration, midi_mask
        )

        audio_embs.append(z_audio.cpu())
        midi_embs.append(z_midi.cpu())

    elapsed = time.time() - start_time
    logger.info(f"Extraction completed in {elapsed/60:.1f} minutes ({elapsed/len(dataset)*1000:.1f} ms/segment)")

    return torch.cat(audio_embs, dim=0), torch.cat(midi_embs, dim=0)


def sample_structured_pool(
    query_idx: int,
    index: Dict,
    n_segments: int,
    config: PoolConfig,
    rng: random.Random,
    segments  # dataset.segments for piece lookup
) -> Tuple[List[int], int]:
    """
    Sample a structured pool of candidates for a query.

    Returns:
        (candidate_indices, positive_position)
    """
    query_piece = segments[query_idx].piece_idx
    query_composer = index['piece_to_composer'].get(query_piece, 'Unknown')

    candidates = []

    # 1. Hard negatives: same piece, different time
    same_piece_indices = [i for i in index['by_piece'].get(query_piece, []) if i != query_idx]
    n_hard = min(config.n_hard_negatives, len(same_piece_indices))
    if n_hard > 0:
        candidates.extend(rng.sample(same_piece_indices, n_hard))

    # 2. Semi-hard negatives: same composer, different piece
    same_composer_indices = [
        i for i in index['by_composer'].get(query_composer, [])
        if segments[i].piece_idx != query_piece
    ]
    n_semi = min(config.n_semi_hard_negatives, len(same_composer_indices))
    if n_semi > 0:
        candidates.extend(rng.sample(same_composer_indices, n_semi))

    # 3. Random negatives: different pieces
    same_piece_set = set(index['by_piece'].get(query_piece, []))
    random_pool = [i for i in range(n_segments) if i not in same_piece_set and i not in candidates]

    n_random = config.pool_size - len(candidates) - 1
    if n_random > 0 and random_pool:
        n_random = min(n_random, len(random_pool))
        candidates.extend(rng.sample(random_pool, n_random))

    # 4. Insert positive at random position
    positive_position = rng.randint(0, len(candidates))
    candidates.insert(positive_position, query_idx)

    return candidates, positive_position


def evaluate_with_precomputed_embeddings(
    audio_embs: torch.Tensor,
    midi_embs: torch.Tensor,
    dataset,
    index: Dict,
    config: PoolConfig,
    direction: str = 'a2m',
    seed: int = 42
) -> Dict:
    """
    Fast evaluation using pre-computed embeddings.

    All operations are in-memory tensor ops - very fast.
    """
    rng = random.Random(seed)
    n_segments = len(audio_embs)

    # Normalize embeddings once
    audio_norm = F.normalize(audio_embs, dim=1)
    midi_norm = F.normalize(midi_embs, dim=1)

    # Sample query indices
    query_indices = rng.sample(range(n_segments), min(config.n_queries, n_segments))

    all_metrics = []

    for query_idx in tqdm(query_indices, desc=f"Evaluating {direction}"):
        # Sample pool
        candidate_indices, positive_position = sample_structured_pool(
            query_idx, index, n_segments, config, rng, dataset.segments
        )

        # Get embeddings (instant - just indexing)
        if direction == 'a2m':
            query_emb = audio_norm[query_idx]
            candidate_embs = midi_norm[candidate_indices]
        else:
            query_emb = midi_norm[query_idx]
            candidate_embs = audio_norm[candidate_indices]

        # Compute similarities
        sims = (query_emb @ candidate_embs.T)

        # Get rank of positive
        sorted_indices = sims.argsort(descending=True)
        rank = (sorted_indices == positive_position).nonzero().item()

        metrics = {
            'rank': rank,
            'reciprocal_rank': 1.0 / (rank + 1),
            'recall@1': 1.0 if rank < 1 else 0.0,
            'recall@5': 1.0 if rank < 5 else 0.0,
            'recall@10': 1.0 if rank < 10 else 0.0,
            'recall@20': 1.0 if rank < 20 else 0.0,
        }
        all_metrics.append(metrics)

    # Aggregate
    agg = {}
    for key in all_metrics[0].keys():
        values = [m[key] for m in all_metrics]
        agg[f'mean_{key}'] = float(np.mean(values))
        if key == 'rank':
            agg['median_rank'] = float(np.median(values))

    agg['pool_size'] = config.pool_size
    agg['n_hard_negatives'] = config.n_hard_negatives
    agg['n_semi_hard_negatives'] = config.n_semi_hard_negatives
    agg['n_queries'] = len(query_indices)
    agg['random_baseline'] = 1.0 / config.pool_size

    for k in [1, 5, 10]:
        recall_key = f'mean_recall@{k}'
        if recall_key in agg:
            random_recall = k / config.pool_size
            agg[f'vs_random@{k}'] = agg[recall_key] / random_recall if random_recall > 0 else 0

    return agg


def analyze_hard_negatives_fast(
    audio_embs: torch.Tensor,
    midi_embs: torch.Tensor,
    dataset,
    index: Dict,
    n_samples: int = 500,  # Increased since it's fast now
    seed: int = 42
) -> Dict:
    """
    Fast hard negative analysis using pre-computed embeddings.
    """
    rng = random.Random(seed)

    audio_norm = F.normalize(audio_embs, dim=1)
    midi_norm = F.normalize(midi_embs, dim=1)

    # Find pieces with enough segments
    pieces_with_many = [
        piece_idx for piece_idx, indices in index['by_piece'].items()
        if len(indices) >= 10
    ]

    if not pieces_with_many:
        return {'error': 'No pieces with enough segments'}

    correct_vs_same_piece = 0
    correct_vs_random = 0
    total = 0

    all_indices = list(range(len(audio_embs)))

    for _ in tqdm(range(n_samples), desc="Hard negative analysis"):
        piece_idx = rng.choice(pieces_with_many)
        piece_segments = index['by_piece'][piece_idx]

        # Query segment
        query_idx = rng.choice(piece_segments)

        # Hard negative (same piece, different time)
        hard_neg_candidates = [i for i in piece_segments if i != query_idx]
        hard_neg_idx = rng.choice(hard_neg_candidates)

        # Random negative (different piece)
        random_candidates = [i for i in all_indices if i not in piece_segments]
        random_neg_idx = rng.choice(random_candidates)

        # Compute similarities (instant)
        query_audio = audio_norm[query_idx]

        sim_pos = (query_audio @ midi_norm[query_idx]).item()
        sim_hard = (query_audio @ midi_norm[hard_neg_idx]).item()
        sim_random = (query_audio @ midi_norm[random_neg_idx]).item()

        if sim_pos > sim_hard:
            correct_vs_same_piece += 1
        if sim_pos > sim_random:
            correct_vs_random += 1

        total += 1

    return {
        'n_samples': total,
        'accuracy_vs_same_piece': correct_vs_same_piece / total if total > 0 else 0,
        'accuracy_vs_random': correct_vs_random / total if total > 0 else 0,
        'interpretation': {
            'vs_same_piece': 'Can distinguish same segment from different time in same piece',
            'vs_random': 'Can distinguish same segment from random different piece'
        }
    }


def main():
    parser = argparse.ArgumentParser(description='Structured pool evaluation (optimized)')
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--maestro-dir', type=str, default='data/maestro_v3/maestro-v3.0.0')
    parser.add_argument('--output', type=str, default=None, help='Output JSON file')
    parser.add_argument('--pool-size', type=int, default=256)
    parser.add_argument('--n-hard', type=int, default=64)
    parser.add_argument('--n-semi-hard', type=int, default=32)
    parser.add_argument('--n-queries', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for extraction (RTX 3090: 64)')
    parser.add_argument('--num-workers', type=int, default=8, help='DataLoader workers (i5-12600K: 8)')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    maestro_dir = Path(args.maestro_dir)

    # Load model
    logger.info(f"Loading model from {args.model}")
    from src.bias_control.architectures.cross_modal_model import CrossModalModel

    checkpoint = torch.load(args.model, map_location=device)
    model = CrossModalModel(audio_encoder='lite')
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Load dataset
    logger.info("Loading validation dataset...")
    from src.bias_control.datasets.maestro_segments import MaestroSegmentDataset

    dataset = MaestroSegmentDataset(
        maestro_dir=maestro_dir,
        segment_len=4.0,
        hop=1.0,
        split='validation'
    )

    # Build index
    logger.info("Building segment index...")
    index = build_segment_index(dataset)
    logger.info(f"Index: {len(index['by_piece'])} pieces, {len(index['by_composer'])} composers")

    # ========================================
    # KEY OPTIMIZATION: Extract ALL embeddings ONCE
    # ========================================
    logger.info("\n" + "="*60)
    logger.info("PRE-EXTRACTING ALL EMBEDDINGS (one-time cost)")
    logger.info("="*60)

    audio_embs, midi_embs = extract_all_embeddings(
        model, dataset, device,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    logger.info(f"Embeddings shape: audio={audio_embs.shape}, midi={midi_embs.shape}")

    # Configure pool
    config = PoolConfig(
        pool_size=args.pool_size,
        n_hard_negatives=args.n_hard,
        n_semi_hard_negatives=args.n_semi_hard,
        n_queries=args.n_queries
    )

    results = {
        'config': {
            'model': args.model,
            'pool_size': args.pool_size,
            'n_hard_negatives': args.n_hard,
            'n_semi_hard_negatives': args.n_semi_hard,
            'n_queries': args.n_queries,
            'seed': args.seed
        }
    }

    # ========================================
    # FAST EVALUATION (uses pre-computed embeddings)
    # ========================================
    logger.info("\n" + "="*60)
    logger.info("STRUCTURED POOL EVALUATION (fast, in-memory)")
    logger.info("="*60)

    # Audio → MIDI
    logger.info("\nDirection: Audio → MIDI")
    start = time.time()
    results['a2m'] = evaluate_with_precomputed_embeddings(
        audio_embs, midi_embs, dataset, index, config,
        direction='a2m', seed=args.seed
    )
    logger.info(f"A2M completed in {time.time()-start:.1f}s")

    # MIDI → Audio
    logger.info("\nDirection: MIDI → Audio")
    start = time.time()
    results['m2a'] = evaluate_with_precomputed_embeddings(
        audio_embs, midi_embs, dataset, index, config,
        direction='m2a', seed=args.seed
    )
    logger.info(f"M2A completed in {time.time()-start:.1f}s")

    # Hard negative analysis
    logger.info("\n" + "="*60)
    logger.info("HARD NEGATIVE ANALYSIS (fast)")
    logger.info("="*60)
    start = time.time()
    results['hard_negative_analysis'] = analyze_hard_negatives_fast(
        audio_embs, midi_embs, dataset, index,
        n_samples=500, seed=args.seed
    )
    logger.info(f"Hard negative analysis completed in {time.time()-start:.1f}s")

    # Print summary
    logger.info("\n" + "="*60)
    logger.info("SUMMARY")
    logger.info("="*60)

    for direction in ['a2m', 'm2a']:
        r = results[direction]
        logger.info(f"\n{direction.upper()}:")
        logger.info(f"  Recall@1:  {r['mean_recall@1']:.1%} (vs random: {r['vs_random@1']:.1f}x)")
        logger.info(f"  Recall@5:  {r['mean_recall@5']:.1%} (vs random: {r['vs_random@5']:.1f}x)")
        logger.info(f"  Recall@10: {r['mean_recall@10']:.1%} (vs random: {r['vs_random@10']:.1f}x)")
        logger.info(f"  MRR:       {r['mean_reciprocal_rank']:.3f}")
        logger.info(f"  Mean Rank: {r['mean_rank']:.1f} / {config.pool_size}")

    hn = results['hard_negative_analysis']
    logger.info(f"\nHard Negative Analysis:")
    logger.info(f"  Accuracy vs same-piece-diff-time: {hn['accuracy_vs_same_piece']:.1%}")
    logger.info(f"  Accuracy vs random:               {hn['accuracy_vs_random']:.1%}")

    # GO/NO-GO decision
    logger.info("\n" + "="*60)
    logger.info("GO/NO-GO ASSESSMENT")
    logger.info("="*60)

    r1_a2m = results['a2m']['mean_recall@1']
    r1_m2a = results['m2a']['mean_recall@1']
    hn_acc = hn['accuracy_vs_same_piece']

    # Thresholds
    threshold_r1 = 0.05  # 5% recall@1 = 12.8x random
    threshold_hn = 0.55  # Better than chance on hard negatives

    pass_r1 = (r1_a2m > threshold_r1) or (r1_m2a > threshold_r1)
    pass_hn = hn_acc > threshold_hn

    if pass_r1 and pass_hn:
        decision = "GO"
        logger.info(f"Decision: {decision}")
        logger.info("  ✓ Model shows cross-modal learning with temporal discrimination")
    elif pass_r1 or pass_hn:
        decision = "WEAK-GO"
        logger.info(f"Decision: {decision}")
        logger.info("  ~ Partial signal detected, more training may help")
    else:
        decision = "NO-GO"
        logger.info(f"Decision: {decision}")
        logger.info("  ✗ No clear cross-modal temporal learning detected")

    results['decision'] = decision
    results['thresholds'] = {
        'recall@1': threshold_r1,
        'hard_negative_accuracy': threshold_hn
    }

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"\nResults saved to {args.output}")

    return results


if __name__ == '__main__':
    main()
