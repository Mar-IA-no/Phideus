#!/usr/bin/env python3
"""
Structured Pool Evaluation for BIAS_CONTROL

Instead of ranking against the entire validation set (13K+ candidates),
uses a structured pool of 256 candidates per query:

- 1 positive (same piece, same time)
- 64 hard negatives (same piece, different time)
- 32 semi-hard negatives (same composer, if available)
- Rest: random negatives from different pieces

This gives much more sensitive metrics for detecting temporal identity learning.

Usage:
    python experiments/bias_control/evaluate_structured_pool.py \
        --model data/bias_control_medium/training_outputs/gate2/best_model.pt \
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

import numpy as np
import torch
import torch.nn.functional as F
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

        # Index by piece
        if piece_idx not in by_piece:
            by_piece[piece_idx] = []
        by_piece[piece_idx].append(i)

        # Index by composer
        if composer not in by_composer:
            by_composer[composer] = []
        by_composer[composer].append(i)

        # Map piece to composer
        piece_to_composer[piece_idx] = composer

    return {
        'by_piece': by_piece,
        'by_composer': by_composer,
        'piece_to_composer': piece_to_composer
    }


def sample_structured_pool(
    query_idx: int,
    dataset,
    index: Dict,
    config: PoolConfig,
    rng: random.Random
) -> Tuple[List[int], int]:
    """
    Sample a structured pool of candidates for a query.

    Args:
        query_idx: Index of the query segment
        dataset: MaestroSegmentDataset
        index: Pre-built index from build_segment_index
        config: Pool configuration
        rng: Random number generator

    Returns:
        (candidate_indices, positive_position): List of indices and position of positive
    """
    query_seg = dataset.segments[query_idx]
    query_piece = query_seg.piece_idx
    query_composer = index['piece_to_composer'].get(query_piece, 'Unknown')

    candidates = []

    # 1. Hard negatives: same piece, different time
    same_piece_indices = [i for i in index['by_piece'].get(query_piece, []) if i != query_idx]
    n_hard = min(config.n_hard_negatives, len(same_piece_indices))
    if n_hard > 0:
        hard_negs = rng.sample(same_piece_indices, n_hard)
        candidates.extend(hard_negs)

    # 2. Semi-hard negatives: same composer, different piece
    same_composer_indices = [
        i for i in index['by_composer'].get(query_composer, [])
        if dataset.segments[i].piece_idx != query_piece
    ]
    n_semi = min(config.n_semi_hard_negatives, len(same_composer_indices))
    if n_semi > 0:
        semi_hard_negs = rng.sample(same_composer_indices, n_semi)
        candidates.extend(semi_hard_negs)

    # 3. Random negatives: different pieces
    all_indices = set(range(len(dataset)))
    same_piece_set = set(index['by_piece'].get(query_piece, []))
    random_pool = list(all_indices - same_piece_set - set(candidates))

    n_random = config.pool_size - len(candidates) - 1  # -1 for positive
    if n_random > 0 and random_pool:
        n_random = min(n_random, len(random_pool))
        random_negs = rng.sample(random_pool, n_random)
        candidates.extend(random_negs)

    # 4. Insert positive (the query itself represents the positive pair)
    # The positive is the MIDI corresponding to this audio query
    positive_position = rng.randint(0, len(candidates))
    candidates.insert(positive_position, query_idx)

    return candidates, positive_position


@torch.no_grad()
def extract_embeddings_batch(
    model,
    dataset,
    indices: List[int],
    device: torch.device,
    batch_size: int = 32
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract audio and MIDI embeddings for given indices.

    Returns:
        (audio_embeddings, midi_embeddings): Both [N, D]
    """
    from src.bias_control.datasets.maestro_segments import collate_segments

    # Create subset
    subset_data = [dataset[i] for i in indices]

    audio_embs = []
    midi_embs = []

    for i in range(0, len(subset_data), batch_size):
        batch_data = subset_data[i:i+batch_size]
        batch = collate_segments(batch_data)

        audio = batch['audio'].to(device)
        midi = batch['midi'].to(device)

        # Forward through model
        z_audio, z_midi = model(audio, midi)

        audio_embs.append(z_audio.cpu())
        midi_embs.append(z_midi.cpu())

    return torch.cat(audio_embs, dim=0), torch.cat(midi_embs, dim=0)


def compute_retrieval_metrics(
    query_emb: torch.Tensor,
    candidate_embs: torch.Tensor,
    positive_idx: int,
    k_values: Tuple[int, ...] = (1, 5, 10, 20)
) -> Dict[str, float]:
    """
    Compute retrieval metrics for a single query.

    Args:
        query_emb: [D] query embedding
        candidate_embs: [N, D] candidate embeddings
        positive_idx: Index of positive in candidates
        k_values: K values for Recall@K

    Returns:
        Dict with rank, reciprocal_rank, and recall@k
    """
    # Normalize
    query_norm = F.normalize(query_emb.unsqueeze(0), dim=1)
    cand_norm = F.normalize(candidate_embs, dim=1)

    # Compute similarities
    sims = (query_norm @ cand_norm.T).squeeze(0)  # [N]

    # Get rank of positive
    sorted_indices = sims.argsort(descending=True)
    rank = (sorted_indices == positive_idx).nonzero().item()

    metrics = {
        'rank': rank,
        'reciprocal_rank': 1.0 / (rank + 1),
    }

    for k in k_values:
        metrics[f'recall@{k}'] = 1.0 if rank < k else 0.0

    return metrics


def evaluate_structured_pool(
    model,
    dataset,
    config: PoolConfig,
    device: torch.device,
    direction: str = 'a2m',  # 'a2m' or 'm2a'
    seed: int = 42
) -> Dict:
    """
    Run structured pool evaluation.

    Args:
        model: Trained cross-modal model
        dataset: MaestroSegmentDataset
        config: Pool configuration
        device: torch device
        direction: 'a2m' (audio→MIDI) or 'm2a' (MIDI→audio)
        seed: Random seed

    Returns:
        Dict with aggregated metrics
    """
    rng = random.Random(seed)
    np.random.seed(seed)

    # Build index
    logger.info("Building segment index...")
    index = build_segment_index(dataset)

    # Sample queries
    all_indices = list(range(len(dataset)))
    query_indices = rng.sample(all_indices, min(config.n_queries, len(all_indices)))

    all_metrics = []

    logger.info(f"Evaluating {len(query_indices)} queries with pool size {config.pool_size}...")

    for query_idx in tqdm(query_indices, desc=f"Evaluating {direction}"):
        # Sample pool
        candidate_indices, positive_position = sample_structured_pool(
            query_idx, dataset, index, config, rng
        )

        # Extract embeddings
        audio_embs, midi_embs = extract_embeddings_batch(
            model, dataset, candidate_indices, device
        )

        # Get query embedding
        if direction == 'a2m':
            query_emb = audio_embs[positive_position]  # Audio query
            candidate_embs = midi_embs  # MIDI candidates
        else:
            query_emb = midi_embs[positive_position]  # MIDI query
            candidate_embs = audio_embs  # Audio candidates

        # Compute metrics
        metrics = compute_retrieval_metrics(
            query_emb, candidate_embs, positive_position
        )
        all_metrics.append(metrics)

    # Aggregate
    agg = {}
    for key in all_metrics[0].keys():
        values = [m[key] for m in all_metrics]
        agg[f'mean_{key}'] = float(np.mean(values))
        if key == 'rank':
            agg['median_rank'] = float(np.median(values))

    # Add pool composition info
    agg['pool_size'] = config.pool_size
    agg['n_hard_negatives'] = config.n_hard_negatives
    agg['n_semi_hard_negatives'] = config.n_semi_hard_negatives
    agg['n_queries'] = len(query_indices)
    agg['random_baseline'] = 1.0 / config.pool_size

    # Compute vs random multiplier
    for k in [1, 5, 10]:
        recall_key = f'mean_recall@{k}'
        if recall_key in agg:
            random_recall = k / config.pool_size
            agg[f'vs_random@{k}'] = agg[recall_key] / random_recall if random_recall > 0 else 0

    return agg


def analyze_hard_negative_performance(
    model,
    dataset,
    device: torch.device,
    n_samples: int = 200,
    seed: int = 42
) -> Dict:
    """
    Specifically analyze performance on hard negatives (same piece, different time).

    This is the key metric for temporal identity learning.
    """
    from src.bias_control.datasets.maestro_segments import collate_segments

    rng = random.Random(seed)
    index = build_segment_index(dataset)

    # Find pieces with enough segments for comparison
    pieces_with_many_segments = [
        piece_idx for piece_idx, indices in index['by_piece'].items()
        if len(indices) >= 10
    ]

    if not pieces_with_many_segments:
        return {'error': 'No pieces with enough segments'}

    correct_vs_same_piece = 0
    correct_vs_random = 0
    total = 0

    logger.info(f"Analyzing hard negative performance on {n_samples} samples...")

    for _ in tqdm(range(n_samples), desc="Hard negative analysis"):
        # Pick a random piece
        piece_idx = rng.choice(pieces_with_many_segments)
        piece_segments = index['by_piece'][piece_idx]

        if len(piece_segments) < 3:
            continue

        # Pick query and positive (same segment)
        query_idx = rng.choice(piece_segments)

        # Pick hard negative (same piece, different time)
        hard_neg_candidates = [i for i in piece_segments if i != query_idx]
        hard_neg_idx = rng.choice(hard_neg_candidates)

        # Pick random negative (different piece)
        all_indices = list(range(len(dataset)))
        random_neg_candidates = [i for i in all_indices if i not in piece_segments]
        random_neg_idx = rng.choice(random_neg_candidates)

        # Extract embeddings
        indices = [query_idx, hard_neg_idx, random_neg_idx]
        audio_embs, midi_embs = extract_embeddings_batch(model, dataset, indices, device)

        # Audio query, MIDI candidates
        query_audio = F.normalize(audio_embs[0:1], dim=1)
        pos_midi = F.normalize(midi_embs[0:1], dim=1)  # Positive = same segment
        hard_midi = F.normalize(midi_embs[1:2], dim=1)  # Same piece, different time
        random_midi = F.normalize(midi_embs[2:3], dim=1)  # Different piece

        sim_pos = (query_audio @ pos_midi.T).item()
        sim_hard = (query_audio @ hard_midi.T).item()
        sim_random = (query_audio @ random_midi.T).item()

        # Check if positive beats hard negative
        if sim_pos > sim_hard:
            correct_vs_same_piece += 1

        # Check if positive beats random
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
    parser = argparse.ArgumentParser(description='Structured pool evaluation')
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--maestro-dir', type=str, default='data/maestro_v3/maestro-v3.0.0')
    parser.add_argument('--output', type=str, default=None, help='Output JSON file')
    parser.add_argument('--pool-size', type=int, default=256)
    parser.add_argument('--n-hard', type=int, default=64)
    parser.add_argument('--n-semi-hard', type=int, default=32)
    parser.add_argument('--n-queries', type=int, default=500)
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

    # Run structured pool evaluation
    logger.info("\n" + "="*60)
    logger.info("STRUCTURED POOL EVALUATION")
    logger.info("="*60)

    # Audio → MIDI
    logger.info("\nDirection: Audio → MIDI")
    results['a2m'] = evaluate_structured_pool(
        model, dataset, config, device, direction='a2m', seed=args.seed
    )

    # MIDI → Audio
    logger.info("\nDirection: MIDI → Audio")
    results['m2a'] = evaluate_structured_pool(
        model, dataset, config, device, direction='m2a', seed=args.seed
    )

    # Hard negative analysis
    logger.info("\n" + "="*60)
    logger.info("HARD NEGATIVE ANALYSIS")
    logger.info("="*60)
    results['hard_negative_analysis'] = analyze_hard_negative_performance(
        model, dataset, device, n_samples=200, seed=args.seed
    )

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

    # Criteria for structured pool
    r1_a2m = results['a2m']['mean_recall@1']
    r1_m2a = results['m2a']['mean_recall@1']
    hn_acc = hn['accuracy_vs_same_piece']

    # Thresholds (more reasonable for pool of 256)
    threshold_r1 = 0.05  # 5% recall@1 = 12.8x random
    threshold_hn = 0.55  # Slightly better than chance on hard negatives

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
