#!/usr/bin/env python3
"""
Gate 1: Intra-Modal Baselines

Objectives:
1. Audio→Audio retrieval with MERT embeddings
2. MIDI→MIDI retrieval with Transformer embeddings
3. Establish floor for cross-modal performance

Criteria GO:
- Audio→Audio Recall@10: > 50%
- MIDI→MIDI Recall@10: > 50%
- Gap aligned vs random: > 0.3

Usage:
    python experiments/bias_control/gate1_intra_modal.py \
        --maestro-dir data/maestro_v3/maestro-v3.0.0 \
        --output data/evaluations/bias_control/gate1 \
        --n-samples 500
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.bias_control.datasets.maestro_segments import (
    MaestroSegmentDataset,
    collate_segments,
)
from src.bias_control.encoders.mert_encoder import MERTEncoder, MERTEncoderLite
from src.bias_control.encoders.midi_encoder import MIDIEncoder

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def compute_retrieval_metrics(
    embeddings: torch.Tensor,
    piece_indices: torch.Tensor,
    k_values: Tuple[int, ...] = (1, 5, 10, 20),
) -> Dict[str, float]:
    """
    Compute intra-modal retrieval metrics.

    For intra-modal, positive = same piece, negative = different piece.

    Args:
        embeddings: [N, D] embeddings
        piece_indices: [N] piece index for each sample
        k_values: K values for Recall@K

    Returns:
        Dict with metrics
    """
    N = embeddings.size(0)
    device = embeddings.device

    # Normalize
    emb_norm = F.normalize(embeddings, dim=1)

    # Compute similarity matrix
    sim_matrix = emb_norm @ emb_norm.T  # [N, N]

    # Build ground truth: positives are same piece (excluding self)
    piece_matrix = piece_indices.unsqueeze(0) == piece_indices.unsqueeze(1)
    same_piece = piece_matrix.float()

    # Exclude self-similarity
    mask_self = ~torch.eye(N, dtype=torch.bool, device=device)
    same_piece = same_piece * mask_self

    # For each query, find rank of first same-piece match
    metrics = {}

    # Sort by similarity
    _, sorted_indices = sim_matrix.sort(dim=1, descending=True)

    # For each sample, find rank of nearest same-piece sample
    ranks = []
    for i in range(N):
        sorted_i = sorted_indices[i]
        same_piece_mask = piece_matrix[i, sorted_i]
        same_piece_mask[sorted_i == i] = False  # Exclude self

        # Find first True (same piece)
        same_piece_positions = same_piece_mask.nonzero()
        if len(same_piece_positions) > 0:
            rank = same_piece_positions[0].item()
            ranks.append(rank)

    if len(ranks) == 0:
        logger.warning("No same-piece matches found!")
        return {f'recall@{k}': 0.0 for k in k_values}

    ranks = torch.tensor(ranks, dtype=torch.float32)

    # Recall@K
    for k in k_values:
        recall = (ranks < k).float().mean().item()
        metrics[f'recall@{k}'] = recall

    # MRR
    metrics['mrr'] = (1.0 / (ranks + 1)).mean().item()

    # Mean rank
    metrics['mean_rank'] = ranks.mean().item()

    # Compute gap
    # Aligned = same piece pairs
    same_piece_sim = sim_matrix[piece_matrix & mask_self].mean().item() if (piece_matrix & mask_self).any() else 0
    # Random = different piece pairs
    diff_piece_sim = sim_matrix[~piece_matrix & mask_self].mean().item() if (~piece_matrix & mask_self).any() else 0

    metrics['same_piece_similarity'] = same_piece_sim
    metrics['diff_piece_similarity'] = diff_piece_sim
    metrics['gap'] = same_piece_sim - diff_piece_sim

    return metrics


def extract_audio_embeddings(
    dataset: MaestroSegmentDataset,
    encoder: torch.nn.Module,
    device: torch.device,
    batch_size: int = 8,
    max_samples: int = 500,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract audio embeddings.

    Returns:
        embeddings: [N, D]
        piece_indices: [N]
    """
    # Subset dataset
    indices = list(range(min(len(dataset), max_samples)))
    subset = Subset(dataset, indices)

    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_segments,
    )

    embeddings_list = []
    piece_indices_list = []

    encoder.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting audio embeddings"):
            audio = batch['audio'].to(device)
            emb = encoder(audio)
            embeddings_list.append(emb.cpu())
            piece_indices_list.append(batch['piece_idx'])

    embeddings = torch.cat(embeddings_list, dim=0)
    piece_indices = torch.cat(piece_indices_list, dim=0)

    return embeddings, piece_indices


def extract_midi_embeddings(
    dataset: MaestroSegmentDataset,
    encoder: torch.nn.Module,
    device: torch.device,
    batch_size: int = 32,
    max_samples: int = 500,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract MIDI embeddings.

    Returns:
        embeddings: [N, D]
        piece_indices: [N]
    """
    indices = list(range(min(len(dataset), max_samples)))
    subset = Subset(dataset, indices)

    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_segments,
    )

    embeddings_list = []
    piece_indices_list = []

    encoder.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting MIDI embeddings"):
            pitch = batch['midi_pitch'].to(device)
            velocity = batch['midi_velocity'].to(device)
            duration = batch['midi_duration'].to(device)
            mask = batch['midi_mask'].to(device) if batch['midi_mask'] is not None else None

            emb = encoder(pitch, velocity, duration, padding_mask=mask)
            embeddings_list.append(emb.cpu())
            piece_indices_list.append(batch['piece_idx'])

    embeddings = torch.cat(embeddings_list, dim=0)
    piece_indices = torch.cat(piece_indices_list, dim=0)

    return embeddings, piece_indices


def run_gate1(
    maestro_dir: Path,
    output_dir: Path,
    segment_len: float = 8.0,
    hop: float = 2.0,
    n_samples: int = 500,
    use_mert_lite: bool = True,  # Use lite version for faster testing
    device: Optional[str] = None,
) -> Dict:
    """
    Run Gate 1: Intra-Modal Baselines.

    Returns:
        Results dict with GO/NO-GO decision
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    results = {
        'gate': 1,
        'name': 'Intra-Modal Baselines',
        'config': {
            'segment_len': segment_len,
            'hop': hop,
            'n_samples': n_samples,
            'use_mert_lite': use_mert_lite,
            'device': str(device),
        },
    }

    # Create dataset (validation split)
    logger.info("Loading dataset...")
    dataset = MaestroSegmentDataset(
        maestro_dir=maestro_dir,
        segment_len=segment_len,
        hop=hop,
        split='validation',
        load_audio=True,
        load_midi=True,
    )
    logger.info(f"Dataset size: {len(dataset)} segments")

    # 1. Audio→Audio with MERT
    logger.info("Step 1: Audio→Audio retrieval...")
    if use_mert_lite:
        audio_encoder = MERTEncoderLite(output_dim=1024).to(device)
    else:
        audio_encoder = MERTEncoder(freeze=True, device=str(device))

    audio_emb, audio_pieces = extract_audio_embeddings(
        dataset, audio_encoder, device,
        batch_size=8 if not use_mert_lite else 16,
        max_samples=n_samples,
    )

    audio_metrics = compute_retrieval_metrics(
        audio_emb, audio_pieces,
        k_values=(1, 5, 10, 20),
    )
    results['audio_retrieval'] = {
        **audio_metrics,
        'threshold_recall@10': 0.50,
        'pass': audio_metrics['recall@10'] >= 0.50,
    }

    # Free memory
    del audio_encoder
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # 2. MIDI→MIDI with Transformer
    logger.info("Step 2: MIDI→MIDI retrieval...")
    midi_encoder = MIDIEncoder(
        embed_dim=512,
        n_layers=4,
        n_heads=8,
    ).to(device)

    midi_emb, midi_pieces = extract_midi_embeddings(
        dataset, midi_encoder, device,
        batch_size=32,
        max_samples=n_samples,
    )

    midi_metrics = compute_retrieval_metrics(
        midi_emb, midi_pieces,
        k_values=(1, 5, 10, 20),
    )
    results['midi_retrieval'] = {
        **midi_metrics,
        'threshold_recall@10': 0.50,
        'pass': midi_metrics['recall@10'] >= 0.50,
    }

    # 3. Gap check
    gap_audio = results['audio_retrieval']['gap']
    gap_midi = results['midi_retrieval']['gap']
    results['gap'] = {
        'audio': gap_audio,
        'midi': gap_midi,
        'threshold': 0.30,
        'pass': gap_audio >= 0.30 and gap_midi >= 0.30,
    }

    # Overall decision
    all_pass = (
        results['audio_retrieval']['pass'] and
        results['midi_retrieval']['pass'] and
        results['gap']['pass']
    )
    results['decision'] = 'GO' if all_pass else 'NO-GO'
    results['status'] = 'PASS' if all_pass else 'FAIL'

    return results


def main():
    parser = argparse.ArgumentParser(description="Gate 1: Intra-Modal Baselines")
    parser.add_argument(
        '--maestro-dir',
        type=str,
        required=True,
        help='Path to MAESTRO dataset'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/evaluations/bias_control/gate1',
        help='Output directory'
    )
    parser.add_argument(
        '--segment-len',
        type=float,
        default=8.0,
        help='Segment length in seconds'
    )
    parser.add_argument(
        '--hop',
        type=float,
        default=2.0,
        help='Hop between segments'
    )
    parser.add_argument(
        '--n-samples',
        type=int,
        default=500,
        help='Number of samples to evaluate'
    )
    parser.add_argument(
        '--use-full-mert',
        action='store_true',
        help='Use full MERT model (slow, needs more VRAM)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device (cuda/cpu)'
    )

    args = parser.parse_args()

    maestro_dir = Path(args.maestro_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not maestro_dir.exists():
        logger.error(f"MAESTRO directory not found: {maestro_dir}")
        sys.exit(1)

    # Run Gate 1
    results = run_gate1(
        maestro_dir=maestro_dir,
        output_dir=output_dir,
        segment_len=args.segment_len,
        hop=args.hop,
        n_samples=args.n_samples,
        use_mert_lite=not args.use_full_mert,
        device=args.device,
    )

    # Save results
    results_path = output_dir / "gate1_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("GATE 1: INTRA-MODAL BASELINES - RESULTS")
    print("=" * 60)

    print(f"\n1. Audio→Audio Retrieval:")
    print(f"   Recall@10: {results['audio_retrieval']['recall@10']:.1%} (threshold: 50%)")
    print(f"   MRR: {results['audio_retrieval']['mrr']:.3f}")
    print(f"   Gap: {results['audio_retrieval']['gap']:.3f}")
    print(f"   Status: {'PASS' if results['audio_retrieval']['pass'] else 'FAIL'}")

    print(f"\n2. MIDI→MIDI Retrieval:")
    print(f"   Recall@10: {results['midi_retrieval']['recall@10']:.1%} (threshold: 50%)")
    print(f"   MRR: {results['midi_retrieval']['mrr']:.3f}")
    print(f"   Gap: {results['midi_retrieval']['gap']:.3f}")
    print(f"   Status: {'PASS' if results['midi_retrieval']['pass'] else 'FAIL'}")

    print(f"\n3. Gap Check:")
    print(f"   Audio gap: {results['gap']['audio']:.3f}")
    print(f"   MIDI gap: {results['gap']['midi']:.3f}")
    print(f"   Threshold: 0.30")
    print(f"   Status: {'PASS' if results['gap']['pass'] else 'FAIL'}")

    print(f"\n" + "=" * 60)
    print(f"DECISION: {results['decision']}")
    print("=" * 60)
    print(f"\nResults saved to: {results_path}")

    return 0 if results['decision'] == 'GO' else 1


if __name__ == '__main__':
    sys.exit(main())
