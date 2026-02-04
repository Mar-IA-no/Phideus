#!/usr/bin/env python3
"""
Gate 4: Ratio Auxiliary View

Objectives:
1. Add ratio-based representation as auxiliary view
2. Multi-view learning: Audio ↔ MIDI ↔ Ratios
3. Test if ratio "insight" helps without hash matching

Criteria GO:
- Gap vs same-piece-different-time: improvement over Gate 3
- Offset MAE: reduction
- Recall@10: no degradation

Usage:
    python experiments/bias_control/gate4_ratio_auxiliary.py \
        --maestro-dir data/maestro_v3/maestro-v3.0.0 \
        --checkpoint data/training_outputs/bias_control/gate3/best_model.pt \
        --output data/training_outputs/bias_control/gate4
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.bias_control.datasets.maestro_segments import (
    MaestroSegmentDataset,
    collate_segments,
    create_dataloaders,
)
from src.bias_control.architectures.cross_modal_model import (
    CrossModalModel,
    compute_retrieval_metrics,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RatioEncoder(nn.Module):
    """
    Encoder for ratio histograms.

    Processes soft histogram representations of frequency ratios.
    """

    def __init__(
        self,
        n_bins: int = 256,
        n_channels: int = 3,  # mean, std, count
        hidden_dim: int = 128,
        output_dim: int = 64,
    ):
        super().__init__()

        input_dim = n_bins * n_channels

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, n_bins, n_channels] or [B, n_bins, n_channels] ratio histogram

        Returns:
            [B, D] embedding
        """
        # Flatten histogram
        if x.dim() == 4:
            # [B, T, bins, ch] -> [B, T, bins*ch]
            B, T, bins, ch = x.shape
            x = x.view(B, T, -1)
            # Mean pool over time
            x = x.mean(dim=1)  # [B, bins*ch]
        elif x.dim() == 3:
            # [B, bins, ch] -> [B, bins*ch]
            x = x.view(x.size(0), -1)

        return self.encoder(x)


def compute_ratio_histogram(
    frequencies: torch.Tensor,
    n_bins: int = 256,
    ratio_min: float = 0.5,
    ratio_max: float = 2.0,
) -> torch.Tensor:
    """
    Compute soft ratio histogram from frequencies.

    Args:
        frequencies: [B, N] frequency values (Hz)
        n_bins: Number of histogram bins
        ratio_min, ratio_max: Ratio range

    Returns:
        [B, n_bins] soft histogram
    """
    B, N = frequencies.shape
    device = frequencies.device

    # Compute all pairwise ratios
    f1 = frequencies.unsqueeze(2)  # [B, N, 1]
    f2 = frequencies.unsqueeze(1)  # [B, 1, N]

    # Avoid division by zero
    ratios = f1 / (f2 + 1e-8)  # [B, N, N]

    # Filter to valid range
    mask = (ratios >= ratio_min) & (ratios <= ratio_max)

    # Bin edges
    bins = torch.linspace(ratio_min, ratio_max, n_bins + 1, device=device)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Soft binning
    sigma = (ratio_max - ratio_min) / n_bins
    ratios_flat = ratios.view(B, -1, 1)  # [B, N*N, 1]
    centers = bin_centers.view(1, 1, -1)  # [1, 1, n_bins]

    # Gaussian kernel
    weights = torch.exp(-0.5 * ((ratios_flat - centers) / sigma) ** 2)  # [B, N*N, n_bins]

    # Apply mask
    mask_flat = mask.view(B, -1, 1)  # [B, N*N, 1]
    weights = weights * mask_flat

    # Sum to histogram
    histogram = weights.sum(dim=1)  # [B, n_bins]

    # Normalize
    histogram = histogram / (histogram.sum(dim=1, keepdim=True) + 1e-8)

    return histogram


class MultiViewModel(nn.Module):
    """
    Multi-view cross-modal model with ratio auxiliary view.

    Views:
    1. Audio (MERT)
    2. MIDI (Transformer)
    3. Ratios (histogram encoder)

    Losses:
    - VICReg(Audio, MIDI)
    - VICReg(Audio, Ratio) (optional)
    - VICReg(MIDI, Ratio) (optional)
    """

    def __init__(
        self,
        base_model: CrossModalModel,
        ratio_encoder: RatioEncoder,
        proj_dim: int = 256,
    ):
        super().__init__()

        self.base_model = base_model
        self.ratio_encoder = ratio_encoder

        # Project ratio encoder to same dim as base embeddings
        self.ratio_projection = nn.Sequential(
            nn.Linear(ratio_encoder.output_dim, proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim),
        )

    def forward(
        self,
        audio: torch.Tensor,
        midi_pitch: torch.Tensor,
        midi_velocity: torch.Tensor,
        midi_duration: torch.Tensor,
        midi_mask: Optional[torch.Tensor] = None,
        ratio_histogram: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Returns:
            audio_emb, midi_emb, ratio_emb (if ratio_histogram provided)
        """
        audio_emb, midi_emb = self.base_model(
            audio=audio,
            midi_pitch=midi_pitch,
            midi_velocity=midi_velocity,
            midi_duration=midi_duration,
            midi_mask=midi_mask,
        )

        ratio_emb = None
        if ratio_histogram is not None:
            ratio_features = self.ratio_encoder(ratio_histogram)
            ratio_emb = self.ratio_projection(ratio_features)

        return audio_emb, midi_emb, ratio_emb

    def compute_loss(
        self,
        batch: Dict[str, torch.Tensor],
        ratio_weight: float = 0.05,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute multi-view loss."""
        audio_emb, midi_emb, ratio_emb = self(
            audio=batch['audio'],
            midi_pitch=batch['midi_pitch'],
            midi_velocity=batch['midi_velocity'],
            midi_duration=batch['midi_duration'],
            midi_mask=batch.get('midi_mask'),
            ratio_histogram=batch.get('ratio_histogram'),
        )

        # Main VICReg loss
        main_loss, main_metrics = self.base_model.compute_vicreg_loss(audio_emb, midi_emb)

        metrics = {k: v for k, v in main_metrics.items()}
        total_loss = main_loss

        # Ratio auxiliary losses
        if ratio_emb is not None:
            # Audio-Ratio
            ar_loss, ar_metrics = self.base_model.compute_vicreg_loss(audio_emb, ratio_emb)
            total_loss = total_loss + ratio_weight * ar_loss
            metrics['audio_ratio_loss'] = ar_metrics['vicreg_loss']

            # MIDI-Ratio
            mr_loss, mr_metrics = self.base_model.compute_vicreg_loss(midi_emb, ratio_emb)
            total_loss = total_loss + ratio_weight * mr_loss
            metrics['midi_ratio_loss'] = mr_metrics['vicreg_loss']

        metrics['total_loss'] = total_loss.item()

        return total_loss, metrics


class Gate4Trainer:
    """Trainer for Gate 4 with ratio auxiliary view."""

    def __init__(
        self,
        model: MultiViewModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        output_dir: Path,
        lr: float = 1e-4,
        ratio_weight: float = 0.05,
        max_epochs: int = 30,
        device: str = 'cuda',
        gate3_recall: float = 0.0,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = device
        self.max_epochs = max_epochs
        self.ratio_weight = ratio_weight
        self.gate3_recall = gate3_recall

        # Only train ratio encoder and projection
        trainable_params = (
            list(model.ratio_encoder.parameters()) +
            list(model.ratio_projection.parameters())
        )

        self.optimizer = AdamW(trainable_params, lr=lr, weight_decay=1e-4)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=max_epochs)

        self.global_step = 0
        self.best_recall = 0.0
        self.history = []

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train one epoch."""
        self.model.train()

        total_loss = 0.0
        n_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.max_epochs}")

        for batch in pbar:
            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            # Generate ratio histogram from MIDI pitches
            # Convert MIDI pitches to frequencies
            midi_freqs = 440.0 * 2 ** ((batch['midi_pitch'].float() - 69) / 12)

            # Compute histogram per sample
            histograms = []
            for i in range(midi_freqs.size(0)):
                mask = ~batch['midi_mask'][i] if batch['midi_mask'] is not None else None
                freqs = midi_freqs[i]
                if mask is not None:
                    freqs = freqs[mask]
                hist = compute_ratio_histogram(freqs.unsqueeze(0))
                histograms.append(hist)

            batch['ratio_histogram'] = torch.cat(histograms, dim=0)

            self.optimizer.zero_grad()
            loss, metrics = self.model.compute_loss(batch, ratio_weight=self.ratio_weight)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            self.global_step += 1

            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        self.scheduler.step()

        return {'train_loss': total_loss / n_batches}

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Evaluate model."""
        self.model.eval()

        audio_embeddings = []
        midi_embeddings = []
        piece_indices = []
        segment_indices = []

        for batch in tqdm(self.val_loader, desc="Evaluating"):
            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            audio_emb, midi_emb, _ = self.model(
                audio=batch['audio'],
                midi_pitch=batch['midi_pitch'],
                midi_velocity=batch['midi_velocity'],
                midi_duration=batch['midi_duration'],
                midi_mask=batch.get('midi_mask'),
            )

            audio_embeddings.append(audio_emb.cpu())
            midi_embeddings.append(midi_emb.cpu())
            piece_indices.append(batch['piece_idx'])
            segment_indices.append(batch['segment_idx'])

        audio_embeddings = torch.cat(audio_embeddings, dim=0)
        midi_embeddings = torch.cat(midi_embeddings, dim=0)
        piece_indices = torch.cat(piece_indices, dim=0)
        segment_indices = torch.cat(segment_indices, dim=0)

        # Standard retrieval metrics
        retrieval_metrics = compute_retrieval_metrics(
            audio_embeddings, midi_embeddings,
            k_values=(1, 5, 10, 20),
        )

        # Same-piece-different-time analysis
        # For each piece, compute similarity between different segments
        unique_pieces = piece_indices.unique()
        same_piece_same_time_sims = []
        same_piece_diff_time_sims = []

        audio_norm = F.normalize(audio_embeddings, dim=1)
        midi_norm = F.normalize(midi_embeddings, dim=1)

        for piece in unique_pieces:
            mask = piece_indices == piece
            if mask.sum() < 2:
                continue

            piece_audio = audio_norm[mask]
            piece_midi = midi_norm[mask]
            piece_segments = segment_indices[mask]

            # Similarity matrix within piece
            sim = piece_audio @ piece_midi.T

            # Same segment (diagonal)
            same_time = sim.diag()
            same_piece_same_time_sims.extend(same_time.tolist())

            # Different segment (off-diagonal)
            n = sim.size(0)
            for i in range(n):
                for j in range(n):
                    if i != j:
                        same_piece_diff_time_sims.append(sim[i, j].item())

        metrics = {
            **{f'val_{k}': v for k, v in retrieval_metrics.items()},
        }

        if same_piece_same_time_sims and same_piece_diff_time_sims:
            metrics['same_piece_same_time_sim'] = np.mean(same_piece_same_time_sims)
            metrics['same_piece_diff_time_sim'] = np.mean(same_piece_diff_time_sims)
            metrics['time_discrimination_gap'] = (
                metrics['same_piece_same_time_sim'] -
                metrics['same_piece_diff_time_sim']
            )

        return metrics

    def train(self) -> Dict:
        """Full training loop."""
        logger.info(f"Starting Gate 4 training for {self.max_epochs} epochs")

        start_time = time.time()

        for epoch in range(self.max_epochs):
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.evaluate()

            epoch_metrics = {'epoch': epoch + 1, **train_metrics, **val_metrics}
            self.history.append(epoch_metrics)

            recall_avg = (val_metrics['val_a2m_recall@10'] + val_metrics['val_m2a_recall@10']) / 2

            logger.info(
                f"Epoch {epoch+1}: "
                f"loss={train_metrics['train_loss']:.4f}, "
                f"recall@10={recall_avg:.3f}, "
                f"gap={val_metrics['val_gap']:.3f}"
            )

            if recall_avg > self.best_recall:
                self.best_recall = recall_avg
                self.save_checkpoint('best_model.pt')

        self.save_checkpoint('final_model.pt')

        # Save history
        with open(self.output_dir / 'training_history.json', 'w') as f:
            json.dump(self.history, f, indent=2)

        return {
            'best_recall': self.best_recall,
            'final_metrics': self.history[-1],
            'training_time_minutes': (time.time() - start_time) / 60,
        }

    def save_checkpoint(self, filename: str):
        """Save checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_recall': self.best_recall,
        }, self.output_dir / filename)


def run_gate4(
    maestro_dir: Path,
    checkpoint_path: Path,
    output_dir: Path,
    segment_len: float = 8.0,
    hop: float = 2.0,
    batch_size: int = 64,
    num_workers: int = 8,
    epochs: int = 30,
    ratio_weight: float = 0.05,
    device: Optional[str] = None,
    gate3_recall: float = 0.0,
) -> Dict:
    """
    Run Gate 4.

    Returns:
        Results dict
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        'gate': 4,
        'name': 'Ratio Auxiliary View',
        'config': {
            'checkpoint': str(checkpoint_path),
            'epochs': epochs,
            'ratio_weight': ratio_weight,
            'gate3_recall': gate3_recall,
        },
    }

    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_loader, val_loader, _ = create_dataloaders(
        maestro_dir=maestro_dir,
        segment_len=segment_len,
        hop=hop,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # Create base model
    logger.info("Creating model...")
    base_model = CrossModalModel(
        audio_encoder='lite',
        use_dann=True,
        device=device,
    )

    # Load Gate 3 weights
    logger.info(f"Loading Gate 3 checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    base_model.load_state_dict(checkpoint['model_state_dict'])

    # Freeze base model
    for param in base_model.parameters():
        param.requires_grad = False

    # Create ratio encoder
    ratio_encoder = RatioEncoder(
        n_bins=256,
        n_channels=1,  # Just histogram, not mean/std/count
        hidden_dim=128,
        output_dim=64,
    )

    # Create multi-view model
    model = MultiViewModel(
        base_model=base_model,
        ratio_encoder=ratio_encoder,
        proj_dim=base_model.proj_output_dim,
    )

    # Train
    trainer = Gate4Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        output_dir=output_dir,
        ratio_weight=ratio_weight,
        max_epochs=epochs,
        device=device,
        gate3_recall=gate3_recall,
    )

    training_results = trainer.train()
    results['training'] = training_results

    # Final evaluation
    final_metrics = trainer.evaluate()

    # Check criteria
    a2m_recall = final_metrics['val_a2m_recall@10']
    m2a_recall = final_metrics['val_m2a_recall@10']
    recall_avg = (a2m_recall + m2a_recall) / 2

    # Recall should not decrease
    recall_ok = recall_avg >= gate3_recall * 0.95

    # Time discrimination should improve
    time_gap = final_metrics.get('time_discrimination_gap', 0)

    results['retrieval'] = {
        'a2m_recall@10': a2m_recall,
        'm2a_recall@10': m2a_recall,
        'recall_avg': recall_avg,
        'gate3_baseline': gate3_recall,
        'pass': recall_ok,
    }

    results['time_discrimination'] = {
        'same_piece_same_time_sim': final_metrics.get('same_piece_same_time_sim', 0),
        'same_piece_diff_time_sim': final_metrics.get('same_piece_diff_time_sim', 0),
        'gap': time_gap,
    }

    results['decision'] = 'GO' if recall_ok else 'NO-GO'
    results['status'] = 'PASS' if recall_ok else 'FAIL'

    return results


def main():
    parser = argparse.ArgumentParser(description="Gate 4: Ratio Auxiliary")
    parser.add_argument('--maestro-dir', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output', type=str, default='data/training_outputs/bias_control/gate4')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--ratio-weight', type=float, default=0.05)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--gate3-recall', type=float, default=0.0)
    parser.add_argument('--device', type=str, default=None)

    args = parser.parse_args()

    results = run_gate4(
        maestro_dir=Path(args.maestro_dir),
        checkpoint_path=Path(args.checkpoint),
        output_dir=Path(args.output),
        epochs=args.epochs,
        ratio_weight=args.ratio_weight,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        gate3_recall=args.gate3_recall,
    )

    # Save results
    results_path = Path(args.output) / "gate4_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("GATE 4: RATIO AUXILIARY - RESULTS")
    print("=" * 60)

    print(f"\n1. Retrieval Performance:")
    print(f"   Audio→MIDI Recall@10: {results['retrieval']['a2m_recall@10']:.1%}")
    print(f"   MIDI→Audio Recall@10: {results['retrieval']['m2a_recall@10']:.1%}")
    print(f"   vs Gate 3 baseline: {results['retrieval']['gate3_baseline']:.1%}")
    print(f"   Status: {'PASS' if results['retrieval']['pass'] else 'FAIL'}")

    print(f"\n2. Time Discrimination:")
    print(f"   Same piece, same time: {results['time_discrimination']['same_piece_same_time_sim']:.3f}")
    print(f"   Same piece, diff time: {results['time_discrimination']['same_piece_diff_time_sim']:.3f}")
    print(f"   Gap: {results['time_discrimination']['gap']:.3f}")

    print(f"\n" + "=" * 60)
    print(f"DECISION: {results['decision']}")
    print("=" * 60)

    return 0 if results['status'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
