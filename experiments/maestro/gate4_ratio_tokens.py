#!/usr/bin/env python3
"""
GATE 4: Ratio Token Training (The Phideus Test)
================================================

This is where "Rosetta" becomes Phideus: we test if constellation/ratio
tokens can achieve cross-modal alignment comparable to dense representations.

Two phases:
1. Baseline WITHOUT networks: Direct constellation matching
2. Training WITH networks:
   - VICReg/Barlow: Token encoders + VICReg or Barlow loss (as per original plan)
   - ConstellationVAE: With decoder and reconstruction loss (UOEMD models)
   - JEPA-lite: Latent prediction without decoder (UOEMD models)

GO Criteria:
- Baseline matching > random (proves signal in ratio representation)
- Trained model comparable to dense (Gate 3) performance

Token format: [log_ratio, delta_t, weight, anchor_band, target_band]
Shape: [B, T, max_tokens, 5] with mask [B, T, max_tokens]

Usage:
------
# Option 1: VICReg (as per original plan)
python experiments/maestro/gate4_ratio_tokens.py \
    --data data/maestro_v3/constellations/tokens.npz \
    --model vicreg --encoder-type mlp

# Option 2: Barlow Twins (as per original plan)
python experiments/maestro/gate4_ratio_tokens.py \
    --data data/maestro_v3/constellations/tokens.npz \
    --model barlow --encoder-type transformer

# Option 3: ConstellationVAE (UOEMD model)
python experiments/maestro/gate4_ratio_tokens.py \
    --data data/maestro_v3/constellations/tokens.npz \
    --model constellation --encoder-type mlp --decoder-type histogram

# Option 4: JEPA-lite (UOEMD model)
python experiments/maestro/gate4_ratio_tokens.py \
    --data data/maestro_v3/constellations/tokens.npz \
    --model jepa-lite --encoder-type mlp
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.datasets.maestro_dataset import (
    MAESTROConstellationDataset,
    create_maestro_dataloaders,
    collate_maestro_constellation,
)
from src.RNA.constellation_vae import (
    ConstellationVAE,
    MLPConstellationEncoder,
    TransformerConstellationEncoder,
)
from src.RNA.jepa_lite import JEPALite
from src.RNA.vicreg import VICRegLoss
from src.RNA.barlow_twins import BarlowTwinsLoss


# ═══════════════════════════════════════════════════════════════════════════════
# 0. DUAL TOKEN ENCODER (VICReg/Barlow - as per original plan)
# ═══════════════════════════════════════════════════════════════════════════════

class DualTokenEncoder(nn.Module):
    """
    Dual encoder for constellation tokens using VICReg or Barlow loss.

    This is the model architecture specified in the original plan:
    - Token encoders (MLP or Transformer) for both audio and MIDI
    - VICReg or Barlow Twins loss for cross-modal alignment
    - NO decoder, NO reconstruction loss
    """

    def __init__(
        self,
        encoder_type: str = 'mlp',
        token_dim: int = 5,
        max_tokens: int = 64,
        hidden_dim: int = 128,
        z_dim: int = 64,
        loss_type: str = 'vicreg',
        dropout: float = 0.1,
    ):
        super().__init__()

        self.z_dim = z_dim
        self.loss_type = loss_type

        # Create encoders (same architecture for both modalities)
        if encoder_type == 'mlp':
            self.encoder_audio = MLPConstellationEncoder(
                token_dim=token_dim,
                max_tokens=max_tokens,
                hidden_dim=hidden_dim,
                z_shared_dim=z_dim,
                z_private_dim=0,  # No private latent for VICReg/Barlow
                dropout=dropout,
            )
            self.encoder_midi = MLPConstellationEncoder(
                token_dim=token_dim,
                max_tokens=max_tokens,
                hidden_dim=hidden_dim,
                z_shared_dim=z_dim,
                z_private_dim=0,
                dropout=dropout,
            )
        else:  # transformer
            self.encoder_audio = TransformerConstellationEncoder(
                token_dim=token_dim,
                max_tokens=max_tokens,
                hidden_dim=hidden_dim,
                z_shared_dim=z_dim,
                z_private_dim=0,
                dropout=dropout,
            )
            self.encoder_midi = TransformerConstellationEncoder(
                token_dim=token_dim,
                max_tokens=max_tokens,
                hidden_dim=hidden_dim,
                z_shared_dim=z_dim,
                z_private_dim=0,
                dropout=dropout,
            )

        # Loss function
        if loss_type == 'vicreg':
            self.loss_fn = VICRegLoss(
                lambda_inv=25.0,
                lambda_var=25.0,
                lambda_cov=1.0,
            )
        else:  # barlow
            self.loss_fn = BarlowTwinsLoss(
                lambda_off_diag=1.0 / z_dim,  # Common default: 1/z_dim
            )

    def encode(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        lengths: Optional[torch.Tensor],
        modality: str,
    ) -> torch.Tensor:
        """Encode tokens to latent space."""
        encoder = self.encoder_audio if modality == 'audio' else self.encoder_midi
        z_mean, _, _, _ = encoder(tokens, mask, lengths)
        # Pool over time dimension
        z = z_mean.mean(dim=1)  # [B, z_dim]
        return z

    def forward(
        self,
        audio_tokens: torch.Tensor,
        audio_mask: torch.Tensor,
        midi_tokens: torch.Tensor,
        midi_mask: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        z_audio = self.encode(audio_tokens, audio_mask, lengths, 'audio')
        z_midi = self.encode(midi_tokens, midi_mask, lengths, 'midi')

        return {
            'z_audio': z_audio,
            'z_midi': z_midi,
        }

    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Compute VICReg or Barlow loss."""
        z_audio = outputs['z_audio']
        z_midi = outputs['z_midi']

        losses = self.loss_fn(z_audio, z_midi)

        return losses


# ═══════════════════════════════════════════════════════════════════════════════
# 1. DIRECT MATCHING BASELINE (NO NEURAL NETWORK)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_histogram_from_tokens(
    tokens: np.ndarray,
    mask: np.ndarray,
    n_ratio_bins: int = 64,
    n_dt_bins: int = 32,
    ratio_range: Tuple[float, float] = (-4.0, 4.0),  # log2 ratio
    dt_range: Tuple[float, float] = (0.0, 2.0),
) -> np.ndarray:
    """
    Convert constellation tokens to 2D histogram (ratio x delta_t).

    Args:
        tokens: [T, K, 5] constellation tokens
        mask: [T, K] validity mask
        n_ratio_bins: Number of bins for log ratio
        n_dt_bins: Number of bins for delta_t
        ratio_range: Range for log ratio
        dt_range: Range for delta_t

    Returns:
        histogram: [n_ratio_bins, n_dt_bins] normalized histogram
    """
    # Extract valid tokens
    valid = mask > 0
    if not valid.any():
        return np.zeros((n_ratio_bins, n_dt_bins), dtype=np.float32)

    log_ratios = tokens[:, :, 0][valid]  # log_ratio
    delta_ts = tokens[:, :, 1][valid]    # delta_t
    weights = tokens[:, :, 2][valid]     # weight

    # Normalize weights
    weights = np.abs(weights)
    weights = weights / (weights.sum() + 1e-8)

    # Create histogram
    hist, _, _ = np.histogram2d(
        log_ratios,
        delta_ts,
        bins=[n_ratio_bins, n_dt_bins],
        range=[ratio_range, dt_range],
        weights=weights,
    )

    # Normalize
    hist = hist / (hist.sum() + 1e-8)

    return hist.astype(np.float32)


def compute_histogram_similarity(hist1: np.ndarray, hist2: np.ndarray) -> float:
    """
    Compute histogram intersection similarity.
    """
    return float(np.minimum(hist1, hist2).sum())


def direct_matching_baseline(
    dataset: MAESTROConstellationDataset,
    n_samples: int = 100,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Baseline evaluation WITHOUT neural networks.

    Uses direct histogram matching between audio and MIDI constellations.

    Args:
        dataset: MAESTRO constellation dataset
        n_samples: Number of samples to evaluate
        seed: Random seed

    Returns:
        Dict with metrics
    """
    np.random.seed(seed)

    n = min(n_samples, len(dataset))
    indices = np.random.choice(len(dataset), n, replace=False)

    # Extract histograms
    audio_hists = []
    midi_hists = []

    print(f"Computing histograms for {n} samples...")
    for idx in tqdm(indices):
        audio_tokens, audio_mask, midi_tokens, midi_mask, meta = dataset[idx]

        audio_hist = compute_histogram_from_tokens(
            audio_tokens.numpy(), audio_mask.numpy()
        )
        midi_hist = compute_histogram_from_tokens(
            midi_tokens.numpy(), midi_mask.numpy()
        )

        audio_hists.append(audio_hist)
        midi_hists.append(midi_hist)

    audio_hists = np.array(audio_hists)  # [N, bins, bins]
    midi_hists = np.array(midi_hists)

    # Flatten for similarity
    audio_flat = audio_hists.reshape(n, -1)
    midi_flat = midi_hists.reshape(n, -1)

    # Compute similarity matrix
    print("Computing similarity matrix...")
    sim = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            sim[i, j] = compute_histogram_similarity(audio_hists[i], midi_hists[j])

    # Metrics
    labels = np.arange(n)

    # Recall@K
    results = {}
    for k in [1, 5, 10]:
        topk = np.argsort(-sim, axis=1)[:, :k]
        correct = np.any(topk == labels[:, None], axis=1)
        results[f'recall@{k}'] = float(correct.mean())

    # MRR
    sorted_idx = np.argsort(-sim, axis=1)
    ranks = np.where(sorted_idx == labels[:, None])[1] + 1
    results['mrr'] = float((1.0 / ranks).mean())
    results['mean_rank'] = float(ranks.mean())

    # Aligned vs shuffled gap
    aligned_scores = np.diag(sim)
    shuffled_scores = sim[np.arange(n), np.roll(np.arange(n), 1)]
    results['aligned_mean'] = float(aligned_scores.mean())
    results['shuffled_mean'] = float(shuffled_scores.mean())
    results['gap'] = results['aligned_mean'] - results['shuffled_mean']

    # Random baseline
    results['random_baseline'] = 1.0 / n
    results['ratio_vs_random'] = results['recall@1'] / results['random_baseline']

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 2. TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    device: str,
    model_type: str,
    beta_kl_shared: float = 0.001,
    beta_kl_private: float = 0.01,
    lambda_infonce: float = 1.0,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_infonce = 0
    n_batches = 0

    for batch in tqdm(train_loader, desc="Training", leave=False):
        audio_tok, audio_msk, midi_tok, midi_msk, metas, lengths = batch
        audio_tok = audio_tok.to(device)
        audio_msk = audio_msk.to(device)
        midi_tok = midi_tok.to(device)
        midi_msk = midi_msk.to(device)
        lengths = lengths.to(device)

        optimizer.zero_grad()

        if model_type in ['vicreg', 'barlow']:
            # VICReg/Barlow: simple forward + loss
            outputs = model(audio_tok, audio_msk, midi_tok, midi_msk, lengths)
            losses = model.compute_loss(outputs)
            loss = losses['total']
            infonce = losses.get('invariance', losses['total']).item()  # Use invariance as proxy
        elif model_type == 'constellation':
            outputs = model(audio_tok, audio_msk, midi_tok, midi_msk, lengths)
            losses = model.compute_loss(
                outputs, audio_tok, midi_tok, audio_msk, midi_msk, lengths,
                beta_kl_shared=beta_kl_shared,
                beta_kl_private=beta_kl_private,
                lambda_infonce=lambda_infonce,
            )
            loss = losses['total']
            infonce = losses['infonce'].item()
        else:  # jepa-lite
            outputs = model(audio_tok, audio_msk, midi_tok, midi_msk, lengths)
            losses = model.compute_loss(outputs)
            loss = losses['total']
            infonce = losses['infonce'].item()
            # Update EMA if using
            model.update_target_encoder()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        total_infonce += infonce
        n_batches += 1

    return {
        'loss': total_loss / n_batches,
        'infonce': total_infonce / n_batches,
    }


@torch.no_grad()
def validate_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    device: str,
    model_type: str,
) -> Dict[str, float]:
    """Validate for one epoch."""
    model.eval()
    total_loss = 0
    all_z_audio = []
    all_z_midi = []
    n_batches = 0

    for batch in tqdm(val_loader, desc="Validation", leave=False):
        audio_tok, audio_msk, midi_tok, midi_msk, metas, lengths = batch
        audio_tok = audio_tok.to(device)
        audio_msk = audio_msk.to(device)
        midi_tok = midi_tok.to(device)
        midi_msk = midi_msk.to(device)
        lengths = lengths.to(device)

        if model_type in ['vicreg', 'barlow']:
            # VICReg/Barlow: embeddings are already pooled
            outputs = model(audio_tok, audio_msk, midi_tok, midi_msk, lengths)
            losses = model.compute_loss(outputs)
            total_loss += losses['total'].item()

            z_audio = outputs['z_audio']  # Already [B, z_dim]
            z_midi = outputs['z_midi']
        elif model_type == 'constellation':
            outputs = model(audio_tok, audio_msk, midi_tok, midi_msk, lengths)
            losses = model.compute_loss(
                outputs, audio_tok, midi_tok, audio_msk, midi_msk, lengths
            )
            total_loss += losses['total'].item()

            # Get z_shared for retrieval
            z_audio = outputs['audio_z_shared'].mean(dim=1)  # Pool over time
            z_midi = outputs['vib_z_shared'].mean(dim=1)
        else:  # jepa-lite
            outputs = model(audio_tok, audio_msk, midi_tok, midi_msk, lengths)
            losses = model.compute_loss(outputs)
            total_loss += losses['total'].item()

            z_audio = outputs['z_audio'].mean(dim=1)
            z_midi = outputs['z_vib'].mean(dim=1)

        all_z_audio.append(z_audio.cpu())
        all_z_midi.append(z_midi.cpu())
        n_batches += 1

    # Compute retrieval metrics
    z_audio = torch.cat(all_z_audio, dim=0).numpy()
    z_midi = torch.cat(all_z_midi, dim=0).numpy()

    # Cosine similarity
    z_a_norm = z_audio / (np.linalg.norm(z_audio, axis=1, keepdims=True) + 1e-8)
    z_m_norm = z_midi / (np.linalg.norm(z_midi, axis=1, keepdims=True) + 1e-8)
    sim = z_a_norm @ z_m_norm.T

    N = len(z_audio)
    labels = np.arange(N)

    # Recall@1
    top1 = np.argmax(sim, axis=1)
    recall1 = float((top1 == labels).mean())

    # MRR
    sorted_idx = np.argsort(-sim, axis=1)
    ranks = np.where(sorted_idx == labels[:, None])[1] + 1
    mrr = float((1.0 / ranks).mean())

    # Variance (collapse check)
    var_audio = float(z_audio.var(axis=0).mean())
    var_midi = float(z_midi.var(axis=0).mean())

    # Aligned vs shuffled gap
    aligned = np.diag(sim)
    shuffled = sim[np.arange(N), np.roll(np.arange(N), 1)]
    gap = float(aligned.mean() - shuffled.mean())

    return {
        'loss': total_loss / n_batches,
        'recall@1': recall1,
        'mrr': mrr,
        'var_audio': var_audio,
        'var_midi': var_midi,
        'gap': gap,
        'random_baseline': 1.0 / N,
    }


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    epochs: int,
    output_dir: Path,
    lr: float = 1e-3,
    model_type: str = 'constellation',
) -> Dict:
    """Full training loop."""
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = []
    best_recall = 0
    best_epoch = 0

    print(f"\nTraining {model_type.upper()} model for {epochs} epochs")
    print("=" * 60)

    for epoch in range(1, epochs + 1):
        train_metrics = train_epoch(model, train_loader, optimizer, device, model_type)
        val_metrics = validate_epoch(model, val_loader, device, model_type)
        scheduler.step()

        history.append({
            'epoch': epoch,
            'train': train_metrics,
            'val': val_metrics,
        })

        # Print progress
        print(f"Epoch {epoch}/{epochs}: "
              f"train_loss={train_metrics['loss']:.4f}, "
              f"val_loss={val_metrics['loss']:.4f}, "
              f"R@1={val_metrics['recall@1']:.4f}, "
              f"gap={val_metrics['gap']:.4f}")

        # Check collapse
        if val_metrics['var_audio'] < 0.01 or val_metrics['var_midi'] < 0.01:
            print(f"  WARNING: Low variance detected (audio={val_metrics['var_audio']:.4f}, midi={val_metrics['var_midi']:.4f})")

        # Save best
        if val_metrics['recall@1'] > best_recall:
            best_recall = val_metrics['recall@1']
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
            }, output_dir / 'best_model.pt')
            print(f"  Best model saved (R@1={best_recall:.4f})")

    # Save final
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'history': history,
    }, output_dir / 'final_model.pt')

    return {
        'history': history,
        'best_epoch': best_epoch,
        'best_recall': best_recall,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 3. EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: str,
    output_dir: Path,
    model_type: str,
) -> Dict:
    """Full evaluation on test set."""
    model.eval()

    all_z_audio = []
    all_z_midi = []
    all_piece_ids = []

    for batch in tqdm(test_loader, desc="Extracting embeddings"):
        audio_tok, audio_msk, midi_tok, midi_msk, metas, lengths = batch
        audio_tok = audio_tok.to(device)
        audio_msk = audio_msk.to(device)
        midi_tok = midi_tok.to(device)
        midi_msk = midi_msk.to(device)
        lengths = lengths.to(device)

        if model_type in ['vicreg', 'barlow']:
            outputs = model(audio_tok, audio_msk, midi_tok, midi_msk, lengths)
            z_audio = outputs['z_audio']  # Already [B, z_dim]
            z_midi = outputs['z_midi']
        elif model_type == 'constellation':
            outputs = model(audio_tok, audio_msk, midi_tok, midi_msk, lengths)
            z_audio = outputs['audio_z_shared'].mean(dim=1)
            z_midi = outputs['vib_z_shared'].mean(dim=1)
        else:  # jepa-lite
            outputs = model(audio_tok, audio_msk, midi_tok, midi_msk, lengths)
            z_audio = outputs['z_audio'].mean(dim=1)
            z_midi = outputs['z_vib'].mean(dim=1)

        all_z_audio.append(z_audio.cpu().numpy())
        all_z_midi.append(z_midi.cpu().numpy())

        for m in metas:
            all_piece_ids.append(m['piece_id'])

    z_audio = np.concatenate(all_z_audio, axis=0)
    z_midi = np.concatenate(all_z_midi, axis=0)

    # Save embeddings
    np.savez(
        output_dir / 'embeddings.npz',
        z_audio=z_audio,
        z_midi=z_midi,
        piece_ids=all_piece_ids,
    )

    # Compute metrics
    N = len(z_audio)

    z_a_norm = z_audio / (np.linalg.norm(z_audio, axis=1, keepdims=True) + 1e-8)
    z_m_norm = z_midi / (np.linalg.norm(z_midi, axis=1, keepdims=True) + 1e-8)
    sim = z_a_norm @ z_m_norm.T

    labels = np.arange(N)

    results = {}

    # Recall@K
    for k in [1, 5, 10, 20]:
        topk = np.argsort(-sim, axis=1)[:, :k]
        correct = np.any(topk == labels[:, None], axis=1)
        results[f'recall@{k}'] = float(correct.mean())

    # MRR
    sorted_idx = np.argsort(-sim, axis=1)
    ranks = np.where(sorted_idx == labels[:, None])[1] + 1
    results['mrr'] = float((1.0 / ranks).mean())
    results['mean_rank'] = float(ranks.mean())

    # Gap
    aligned = np.diag(sim)
    shuffled = sim[np.arange(N), np.roll(np.arange(N), 1)]
    results['gap'] = float(aligned.mean() - shuffled.mean())

    # Variance
    results['var_audio'] = float(z_audio.var(axis=0).mean())
    results['var_midi'] = float(z_midi.var(axis=0).mean())

    # Random baseline
    results['random_baseline'] = 1.0 / N
    results['ratio_vs_random'] = results['recall@1'] / results['random_baseline']

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 4. MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description='GATE 4: Ratio Token Training')
    parser.add_argument('--data', type=Path, required=True,
                        help='Path to constellation tokens NPZ')
    parser.add_argument('--output', type=Path, default=Path('data/training_outputs/maestro_constellation'),
                        help='Output directory')
    parser.add_argument('--model', type=str, default='vicreg',
                        choices=['vicreg', 'barlow', 'constellation', 'jepa-lite'],
                        help='Model type: vicreg/barlow (original plan) or constellation/jepa-lite (UOEMD)')
    parser.add_argument('--encoder-type', type=str, default='mlp',
                        choices=['mlp', 'transformer'],
                        help='Encoder architecture')
    parser.add_argument('--decoder-type', type=str, default='histogram',
                        choices=['histogram', 'token'],
                        help='Decoder type (only for constellation)')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--z-dim', type=int, default=64,
                        help='Latent dimension')
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--max-tokens', type=int, default=48,
                        help='Max tokens per frame')
    parser.add_argument('--max-frames', type=int, default=100,
                        help='Max frames per sequence')
    parser.add_argument('--dropout-shared', type=float, default=0.5,
                        help='Dropout on shared latent')
    parser.add_argument('--beta-kl-private', type=float, default=0.01,
                        help='KL weight for private latent')
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--baseline-only', action='store_true',
                        help='Only run direct matching baseline')
    parser.add_argument('--baseline-samples', type=int, default=200,
                        help='Number of samples for baseline')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    args.output.mkdir(parents=True, exist_ok=True)

    print("GATE 4: Ratio Token Training")
    print("=" * 60)
    print(f"Data: {args.data}")
    print(f"Model: {args.model}")
    print(f"Encoder: {args.encoder_type}")
    if args.model == 'constellation':
        print(f"Decoder: {args.decoder_type}")
    print(f"Device: {device}")
    print(f"Output: {args.output}")

    # ═══════════════════════════════════════════════════════════════════════════
    # PHASE 1: Direct Matching Baseline (No Neural Network)
    # ═══════════════════════════════════════════════════════════════════════════

    print("\n" + "=" * 60)
    print("PHASE 1: Direct Matching Baseline")
    print("=" * 60)

    # Create dataset for baseline
    baseline_dataset = MAESTROConstellationDataset(
        npz_path=args.data,
        split='test',
        max_frames=args.max_frames,
        frame_sample_mode='start',
    )

    print(f"\nDataset: {len(baseline_dataset)} test segments")
    stats = baseline_dataset.get_stats()
    print(f"  Max tokens/frame: {stats['max_tokens']}")
    print(f"  Avg audio tokens: {stats['avg_audio_tokens']:.1f}")
    print(f"  Avg midi tokens: {stats['avg_midi_tokens']:.1f}")

    # Run baseline
    baseline_results = direct_matching_baseline(
        baseline_dataset,
        n_samples=args.baseline_samples,
        seed=args.seed,
    )

    print(f"\nBaseline Results (Direct Histogram Matching):")
    print(f"  Recall@1: {baseline_results['recall@1']:.4f}")
    print(f"  Recall@5: {baseline_results['recall@5']:.4f}")
    print(f"  MRR: {baseline_results['mrr']:.4f}")
    print(f"  Gap (aligned-shuffled): {baseline_results['gap']:.4f}")
    print(f"  vs Random: {baseline_results['ratio_vs_random']:.1f}x")

    # Check GO criterion for baseline
    baseline_pass = baseline_results['recall@1'] > baseline_results['random_baseline'] * 2

    print(f"\nBaseline GO criterion (R@1 > 2x random): {'PASS' if baseline_pass else 'FAIL'}")

    # Save baseline results
    with open(args.output / 'baseline_results.json', 'w') as f:
        json.dump(baseline_results, f, indent=2)

    if args.baseline_only:
        print("\nBaseline-only mode, exiting.")
        return

    # ═══════════════════════════════════════════════════════════════════════════
    # PHASE 2: Neural Network Training
    # ═══════════════════════════════════════════════════════════════════════════

    print("\n" + "=" * 60)
    print("PHASE 2: Neural Network Training")
    print("=" * 60)

    # Create dataloaders
    train_loader, val_loader, test_loader = create_maestro_dataloaders(
        npz_path=args.data,
        batch_size=args.batch_size,
        max_frames=args.max_frames,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    # Read max_tokens from dataset (NPZ contains the correct value from analizador)
    max_tokens = train_loader.dataset.max_tokens
    print(f"Using max_tokens={max_tokens} from dataset")

    # Create model
    if args.model in ['vicreg', 'barlow']:
        # Original plan: Token encoders + VICReg/Barlow loss
        model = DualTokenEncoder(
            encoder_type=args.encoder_type,
            token_dim=5,
            max_tokens=max_tokens,
            hidden_dim=args.hidden_dim,
            z_dim=args.z_dim,
            loss_type=args.model,  # 'vicreg' or 'barlow'
            dropout=0.1,
        ).to(device)
        model_type = args.model  # 'vicreg' or 'barlow'
    elif args.model == 'constellation':
        # UOEMD model: ConstellationVAE with decoder
        model = ConstellationVAE(
            encoder_type=args.encoder_type,
            decoder_type=args.decoder_type,
            token_dim=5,
            max_tokens=max_tokens,
            hidden_dim=args.hidden_dim,
            z_shared_dim=args.z_dim // 2,
            z_private_dim=args.z_dim // 2,
            dropout=0.1,
            dropout_shared=args.dropout_shared,
        ).to(device)
        model_type = 'constellation'
    else:  # jepa-lite
        # UOEMD model: JEPA-lite without decoder
        model = JEPALite(
            encoder_type=args.encoder_type,
            token_dim=5,
            max_tokens=max_tokens,
            hidden_dim=args.hidden_dim,
            z_dim=args.z_dim,
            use_ema_target=True,
        ).to(device)
        model_type = 'jepa-lite'

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {n_params:,}")

    # Train
    train_results = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        output_dir=args.output,
        lr=args.lr,
        model_type=args.model,
    )

    # Load best and evaluate
    print("\nEvaluating best model on test set...")
    checkpoint = torch.load(args.output / 'best_model.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_results = evaluate_model(model, test_loader, device, args.output, args.model)

    # ═══════════════════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════════════════

    print("\n" + "=" * 60)
    print("GATE 4 SUMMARY")
    print("=" * 60)

    print(f"\nBaseline (Direct Matching):")
    print(f"  Recall@1: {baseline_results['recall@1']:.4f}")
    print(f"  Gap: {baseline_results['gap']:.4f}")

    print(f"\nNeural Network ({args.model}):")
    print(f"  Recall@1: {test_results['recall@1']:.4f}")
    print(f"  Recall@5: {test_results['recall@5']:.4f}")
    print(f"  MRR: {test_results['mrr']:.4f}")
    print(f"  Gap: {test_results['gap']:.4f}")
    print(f"  vs Random: {test_results['ratio_vs_random']:.1f}x")
    print(f"  Variance: audio={test_results['var_audio']:.4f}, midi={test_results['var_midi']:.4f}")

    # GO criteria
    baseline_signal = baseline_results['recall@1'] > baseline_results['random_baseline'] * 2
    model_comparable = test_results['recall@1'] >= baseline_results['recall@1'] * 0.8  # Within 80% of baseline
    no_collapse = test_results['var_audio'] > 0.01 and test_results['var_midi'] > 0.01

    print(f"\nGO Criteria:")
    print(f"  Baseline shows signal (R@1 > 2x random): {'PASS' if baseline_signal else 'FAIL'}")
    print(f"  Model comparable to baseline (within 80%): {'PASS' if model_comparable else 'FAIL'}")
    print(f"  No collapse (var > 0.01): {'PASS' if no_collapse else 'FAIL'}")

    go = baseline_signal and model_comparable and no_collapse
    print(f"\n{'GATE 4 PASS' if go else 'GATE 4 FAIL'}")

    # Save results
    all_results = {
        'baseline': baseline_results,
        'train': train_results,
        'test': test_results,
        'config': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        'go_criteria': {
            'baseline_signal': baseline_signal,
            'model_comparable': model_comparable,
            'no_collapse': no_collapse,
            'pass': go,
        },
    }

    with open(args.output / 'gate4_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    # Generate report
    with open(args.output / 'GATE4_REPORT.md', 'w') as f:
        f.write("# GATE 4: Ratio Token Training Report\n\n")
        f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"**Model**: {args.model} ({args.encoder_type})\n")
        f.write(f"**Epochs**: {args.epochs}\n\n")

        f.write("## Phase 1: Direct Matching Baseline\n\n")
        f.write("| Metric | Value |\n|--------|-------|\n")
        for k, v in baseline_results.items():
            if isinstance(v, float):
                f.write(f"| {k} | {v:.4f} |\n")

        f.write("\n## Phase 2: Neural Network\n\n")
        f.write("| Metric | Value |\n|--------|-------|\n")
        for k, v in test_results.items():
            if isinstance(v, float):
                f.write(f"| {k} | {v:.4f} |\n")

        f.write(f"\n## GO/NO-GO\n\n")
        f.write(f"- Baseline shows signal: {'PASS' if baseline_signal else 'FAIL'}\n")
        f.write(f"- Model comparable to baseline: {'PASS' if model_comparable else 'FAIL'}\n")
        f.write(f"- No collapse: {'PASS' if no_collapse else 'FAIL'}\n")
        f.write(f"\n**Result**: {'**PASS**' if go else '**FAIL**'}\n")

    print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
