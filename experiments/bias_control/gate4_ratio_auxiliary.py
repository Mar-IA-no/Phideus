#!/usr/bin/env python3
"""
Gate 4: Ratio Auxiliary View (Plan v2 — consolidated with Codex, DEC-002)

Objectives:
1. Add ratio-based representation as auxiliary VICReg view
2. Multi-view learning: Audio <-> MIDI <-> Ratios
3. Test if ratio "insight" helps cross-modal retrieval

Key design decisions (DEC-002):
- Base model unfrozen (except MERT) — freeze-all invalidates the test
- Dual checkpoint save: full MultiViewModel + base-only for eval compatibility
- Data regime aligned to Gate 2: segment_len=4.0, hop=1.0, batch_size=16
- Training metrics are monitoring only — GO/NO-GO by structured pool
- Control run with ratio_weight=0.0 for causal attribution

Usage:
    # Run A: ratio auxiliary
    python experiments/bias_control/gate4_ratio_auxiliary.py \
        --maestro-dir data/maestro_v3/maestro-v3.0.0 \
        --checkpoint data/bias_control_medium/training_outputs/gate2/checkpoint_epoch45.pt \
        --output data/bias_control_medium/training_outputs/gate4_runA \
        --epochs 30 --ratio-weight 0.1 --batch-size 16 \
        --segment-len 4.0 --hop 1.0 --num-workers 8

    # Run B: control (no ratio)
    python experiments/bias_control/gate4_ratio_auxiliary.py \
        --maestro-dir data/maestro_v3/maestro-v3.0.0 \
        --checkpoint data/bias_control_medium/training_outputs/gate2/checkpoint_epoch45.pt \
        --output data/bias_control_medium/training_outputs/gate4_runB \
        --epochs 30 --ratio-weight 0.0 --batch-size 16 \
        --segment-len 4.0 --hop 1.0 --num-workers 8
"""

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys
import time
import random

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


# ---------------------------------------------------------------------------
# Ratio Encoder
# ---------------------------------------------------------------------------

class RatioEncoder(nn.Module):
    """Encoder for soft ratio histograms."""

    def __init__(
        self,
        n_bins: int = 256,
        n_channels: int = 1,
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
            x: [B, n_bins] or [B, n_bins, n_channels] ratio histogram
        Returns:
            [B, output_dim] embedding
        """
        if x.dim() == 3:
            x = x.view(x.size(0), -1)
        return self.encoder(x)


# ---------------------------------------------------------------------------
# Ratio Histogram Computation (vectorized with memory safety)
# ---------------------------------------------------------------------------

def compute_ratio_histogram(
    frequencies: torch.Tensor,
    n_bins: int = 256,
    ratio_min: float = 0.5,
    ratio_max: float = 2.0,
    max_notes: int = 128,
) -> torch.Tensor:
    """
    Compute soft ratio histogram from frequencies (vectorized, memory-safe).

    Args:
        frequencies: [B, N] frequency values (Hz), may contain zeros (padding)
        n_bins: Number of histogram bins
        ratio_min, ratio_max: Ratio range
        max_notes: Cap per sample to limit memory (N^2 pairs)

    Returns:
        [B, n_bins] soft histogram (normalized)
    """
    B, N = frequencies.shape
    device = frequencies.device

    # Cap notes for memory safety
    if N > max_notes:
        frequencies = frequencies[:, :max_notes]
        N = max_notes

    # Mask out zero/padding frequencies
    valid = frequencies > 1.0  # [B, N] — frequencies below 1 Hz are padding

    # Compute all pairwise ratios: [B, N, N]
    f1 = frequencies.unsqueeze(2)  # [B, N, 1]
    f2 = frequencies.unsqueeze(1)  # [B, 1, N]
    ratios = f1 / (f2 + 1e-8)

    # Mask: both frequencies must be valid, and ratio in range
    pair_valid = valid.unsqueeze(2) & valid.unsqueeze(1)  # [B, N, N]
    in_range = (ratios >= ratio_min) & (ratios <= ratio_max)
    mask = pair_valid & in_range  # [B, N, N]

    # Bin centers
    bins = torch.linspace(ratio_min, ratio_max, n_bins + 1, device=device)
    bin_centers = (bins[:-1] + bins[1:]) / 2  # [n_bins]
    sigma = (ratio_max - ratio_min) / n_bins

    # Soft binning — process in chunks if N is large
    ratios_flat = ratios.reshape(B, -1)  # [B, N*N]
    mask_flat = mask.reshape(B, -1)  # [B, N*N]

    # Gaussian kernel: [B, N*N, n_bins]
    chunk_size = 4096
    n_pairs = ratios_flat.size(1)
    histogram = torch.zeros(B, n_bins, device=device)

    for start in range(0, n_pairs, chunk_size):
        end = min(start + chunk_size, n_pairs)
        r_chunk = ratios_flat[:, start:end].unsqueeze(2)  # [B, chunk, 1]
        m_chunk = mask_flat[:, start:end].unsqueeze(2)  # [B, chunk, 1]
        centers = bin_centers.view(1, 1, -1)  # [1, 1, n_bins]
        weights = torch.exp(-0.5 * ((r_chunk - centers) / sigma) ** 2)  # [B, chunk, n_bins]
        weights = weights * m_chunk
        histogram += weights.sum(dim=1)  # [B, n_bins]

    # Normalize to probability distribution
    histogram = histogram / (histogram.sum(dim=1, keepdim=True) + 1e-8)

    return histogram


def compute_batch_ratio_histograms(
    midi_pitch: torch.Tensor,
    midi_mask: Optional[torch.Tensor],
    n_bins: int = 256,
    max_notes: int = 128,
) -> torch.Tensor:
    """
    Compute ratio histograms for a full batch from MIDI pitches (baseline descriptor).

    Args:
        midi_pitch: [B, N] MIDI pitch values (0-127)
        midi_mask: [B, N] True = padding
        n_bins: Histogram bins
        max_notes: Cap per sample

    Returns:
        [B, n_bins] histograms
    """
    # Convert MIDI pitch to frequency (Hz)
    midi_freqs = 440.0 * 2 ** ((midi_pitch.float() - 69) / 12)

    # Zero out padding
    if midi_mask is not None:
        midi_freqs = midi_freqs * (~midi_mask).float()

    return compute_ratio_histogram(midi_freqs, n_bins=n_bins, max_notes=max_notes)


def compute_batch_ratio_histograms_enriched(
    midi_pitch: torch.Tensor,
    midi_velocity: torch.Tensor,
    midi_duration: torch.Tensor,
    midi_mask: Optional[torch.Tensor],
    n_bins: int = 256,
    max_notes: int = 128,
) -> torch.Tensor:
    """
    Compute enriched 3-channel ratio histograms weighted by velocity and duration.

    Channel 0: Velocity-weighted histogram (forte notes contribute more)
    Channel 1: Duration-weighted histogram (sustained notes contribute more)
    Channel 2: Unweighted histogram (same as baseline, for reference)

    Args:
        midi_pitch: [B, N] MIDI pitch values (0-127)
        midi_velocity: [B, N] MIDI velocity values (0-127)
        midi_duration: [B, N] duration bucket indices (0-31)
        midi_mask: [B, N] True = padding
        n_bins: Histogram bins
        max_notes: Cap per sample

    Returns:
        [B, n_bins, 3] enriched histograms
    """
    B, N = midi_pitch.shape
    device = midi_pitch.device

    # Convert MIDI pitch to frequency (Hz)
    midi_freqs = 440.0 * 2 ** ((midi_pitch.float() - 69) / 12)

    # Zero out padding
    if midi_mask is not None:
        pad_mask = (~midi_mask).float()
        midi_freqs = midi_freqs * pad_mask
    else:
        pad_mask = torch.ones(B, N, device=device)

    # Cap notes for memory safety
    if N > max_notes:
        midi_freqs = midi_freqs[:, :max_notes]
        midi_velocity = midi_velocity[:, :max_notes]
        midi_duration = midi_duration[:, :max_notes]
        pad_mask = pad_mask[:, :max_notes]
        N = max_notes

    # Mask out zero/padding frequencies
    valid = midi_freqs > 1.0  # [B, N]

    # Compute all pairwise ratios: [B, N, N]
    f1 = midi_freqs.unsqueeze(2)  # [B, N, 1]
    f2 = midi_freqs.unsqueeze(1)  # [B, 1, N]
    ratio_min, ratio_max = 0.5, 2.0
    ratios = f1 / (f2 + 1e-8)

    # Mask: both frequencies must be valid, and ratio in range
    pair_valid = valid.unsqueeze(2) & valid.unsqueeze(1)  # [B, N, N]
    in_range = (ratios >= ratio_min) & (ratios <= ratio_max)
    mask = pair_valid & in_range  # [B, N, N]

    # Compute pairwise weights for each channel
    # Velocity weights: geometric mean of the two notes' velocities, normalized to [0, 1]
    vel_float = midi_velocity.float() / 127.0  # [B, N] in [0, 1]
    vel_i = vel_float.unsqueeze(2)  # [B, N, 1]
    vel_j = vel_float.unsqueeze(1)  # [B, 1, N]
    vel_weights = torch.sqrt(vel_i * vel_j + 1e-8)  # [B, N, N] geometric mean

    # Duration weights: geometric mean of durations (use bucket as proxy, normalize)
    # +1 floor so bucket 0 (short notes) still contributes nonzero weight
    dur_float = (midi_duration.float() + 1.0) / 32.0  # [B, N] in [1/32, 1]
    dur_i = dur_float.unsqueeze(2)  # [B, N, 1]
    dur_j = dur_float.unsqueeze(1)  # [B, 1, N]
    dur_weights = torch.sqrt(dur_i * dur_j + 1e-8)  # [B, N, N]

    # Bin centers
    bins = torch.linspace(ratio_min, ratio_max, n_bins + 1, device=device)
    bin_centers = (bins[:-1] + bins[1:]) / 2  # [n_bins]
    sigma = (ratio_max - ratio_min) / n_bins

    # Flatten for chunked processing
    ratios_flat = ratios.reshape(B, -1)      # [B, N*N]
    mask_flat = mask.reshape(B, -1)          # [B, N*N]
    vel_flat = vel_weights.reshape(B, -1)    # [B, N*N]
    dur_flat = dur_weights.reshape(B, -1)    # [B, N*N]

    # 3-channel histogram
    histogram = torch.zeros(B, n_bins, 3, device=device)

    chunk_size = 4096
    n_pairs = ratios_flat.size(1)

    for start in range(0, n_pairs, chunk_size):
        end = min(start + chunk_size, n_pairs)
        r_chunk = ratios_flat[:, start:end].unsqueeze(2)   # [B, chunk, 1]
        m_chunk = mask_flat[:, start:end].unsqueeze(2)      # [B, chunk, 1]
        v_chunk = vel_flat[:, start:end].unsqueeze(2)       # [B, chunk, 1]
        d_chunk = dur_flat[:, start:end].unsqueeze(2)       # [B, chunk, 1]
        centers = bin_centers.view(1, 1, -1)                # [1, 1, n_bins]

        kernel = torch.exp(-0.5 * ((r_chunk - centers) / sigma) ** 2)  # [B, chunk, n_bins]
        kernel_masked = kernel * m_chunk  # [B, chunk, n_bins]

        # Channel 0: velocity-weighted
        histogram[:, :, 0] += (kernel_masked * v_chunk).sum(dim=1)
        # Channel 1: duration-weighted
        histogram[:, :, 1] += (kernel_masked * d_chunk).sum(dim=1)
        # Channel 2: unweighted (same as baseline)
        histogram[:, :, 2] += kernel_masked.sum(dim=1)

    # Normalize each channel independently
    for c in range(3):
        channel_sum = histogram[:, :, c].sum(dim=1, keepdim=True) + 1e-8
        histogram[:, :, c] = histogram[:, :, c] / channel_sum

    return histogram


# ---------------------------------------------------------------------------
# Multi-View Model
# ---------------------------------------------------------------------------

class MultiViewModel(nn.Module):
    """
    Multi-view cross-modal model with ratio auxiliary view.

    Views:
    1. Audio (MERT) -> embedding
    2. MIDI (Transformer) -> embedding
    3. Ratios (histogram encoder) -> embedding

    Losses:
    - VICReg(Audio, MIDI) — main
    - VICReg(Audio, Ratio) * ratio_weight — auxiliary
    - VICReg(MIDI, Ratio) * ratio_weight — auxiliary
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

        # Project ratio embedding to same dim as base embeddings
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
        Returns:
            audio_emb [B, D], midi_emb [B, D], ratio_emb [B, D] or None
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
        ratio_weight: float = 0.1,
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

        # Main VICReg loss (audio <-> midi)
        main_loss, main_metrics = self.base_model.compute_vicreg_loss(audio_emb, midi_emb)

        metrics = {k: v for k, v in main_metrics.items()}
        total_loss = main_loss

        # Ratio auxiliary losses
        if ratio_emb is not None and ratio_weight > 0:
            ar_loss, ar_metrics = self.base_model.compute_vicreg_loss(audio_emb, ratio_emb)
            total_loss = total_loss + ratio_weight * ar_loss
            metrics['audio_ratio_loss'] = ar_metrics['vicreg_loss']

            mr_loss, mr_metrics = self.base_model.compute_vicreg_loss(midi_emb, ratio_emb)
            total_loss = total_loss + ratio_weight * mr_loss
            metrics['midi_ratio_loss'] = mr_metrics['vicreg_loss']

        metrics['total_loss'] = total_loss.item()

        return total_loss, metrics


# ---------------------------------------------------------------------------
# Linear Warmup Scheduler
# ---------------------------------------------------------------------------

class LinearWarmupCosineScheduler:
    """Linear warmup followed by cosine annealing."""

    def __init__(self, optimizer, warmup_steps: int, total_steps: int):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.base_lrs = [pg['lr'] for pg in optimizer.param_groups]
        self.step_count = 0

    def step(self):
        self.step_count += 1
        if self.step_count <= self.warmup_steps:
            # Linear warmup
            scale = self.step_count / max(1, self.warmup_steps)
        else:
            # Cosine annealing
            progress = (self.step_count - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps
            )
            scale = 0.5 * (1 + math.cos(math.pi * progress))

        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg['lr'] = base_lr * scale

    def get_last_lr(self) -> List[float]:
        return [pg['lr'] for pg in self.optimizer.param_groups]

    def state_dict(self):
        return {
            'step_count': self.step_count,
            'base_lrs': self.base_lrs,
        }

    def load_state_dict(self, state_dict):
        self.step_count = state_dict['step_count']
        self.base_lrs = state_dict['base_lrs']


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Gate4Trainer:
    """Trainer for Gate 4 with ratio auxiliary view."""

    def __init__(
        self,
        model: MultiViewModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        output_dir: Path,
        lr_midi: float = 5e-5,
        lr_proj: float = 1e-4,
        lr_ratio: float = 5e-4,
        ratio_weight: float = 0.1,
        max_epochs: int = 30,
        warmup_steps: int = 500,
        max_batches_per_epoch: Optional[int] = None,
        max_val_batches: Optional[int] = None,
        descriptor: str = 'baseline',
        device: str = 'cuda',
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = device
        self.max_epochs = max_epochs
        self.ratio_weight = ratio_weight
        self.max_batches_per_epoch = max_batches_per_epoch
        self.max_val_batches = max_val_batches
        self.descriptor = descriptor

        # --- Multi-group optimizer (Plan v2 section 2) ---
        # Group 1: MIDI encoder — conservative LR
        midi_params = list(model.base_model.midi_encoder.parameters())
        # Group 2: Projection heads — moderate LR
        proj_params = (
            list(model.base_model.audio_projection.parameters()) +
            list(model.base_model.midi_projection.parameters())
        )
        # Group 3: Ratio encoder + projection — higher LR (new modules)
        ratio_params = (
            list(model.ratio_encoder.parameters()) +
            list(model.ratio_projection.parameters())
        )

        self.optimizer = AdamW([
            {'params': midi_params, 'lr': lr_midi, 'name': 'midi_encoder'},
            {'params': proj_params, 'lr': lr_proj, 'name': 'projections'},
            {'params': ratio_params, 'lr': lr_ratio, 'name': 'ratio'},
        ], weight_decay=1e-4)

        # Warmup + cosine scheduler
        effective_train_batches = len(train_loader)
        if self.max_batches_per_epoch is not None:
            effective_train_batches = min(self.max_batches_per_epoch, len(train_loader))
        total_steps = max_epochs * max(effective_train_batches, 1)
        self.scheduler = LinearWarmupCosineScheduler(
            self.optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
        )

        self.global_step = 0
        self.best_recall = 0.0
        self.history = []

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train one epoch."""
        self.model.train()

        total_loss = 0.0
        total_ar_loss = 0.0
        total_mr_loss = 0.0
        n_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.max_epochs}")

        for batch_idx, batch in enumerate(pbar):
            if self.max_batches_per_epoch is not None and batch_idx >= self.max_batches_per_epoch:
                break
            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            # Compute ratio histograms on-the-fly from MIDI pitches
            if self.ratio_weight > 0:
                if self.descriptor == 'enriched':
                    batch['ratio_histogram'] = compute_batch_ratio_histograms_enriched(
                        midi_pitch=batch['midi_pitch'],
                        midi_velocity=batch['midi_velocity'],
                        midi_duration=batch['midi_duration'],
                        midi_mask=batch.get('midi_mask'),
                    )
                else:
                    batch['ratio_histogram'] = compute_batch_ratio_histograms(
                        midi_pitch=batch['midi_pitch'],
                        midi_mask=batch.get('midi_mask'),
                    )

            self.optimizer.zero_grad()
            loss, metrics = self.model.compute_loss(batch, ratio_weight=self.ratio_weight)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()

            total_loss += metrics['total_loss']
            total_ar_loss += metrics.get('audio_ratio_loss', 0)
            total_mr_loss += metrics.get('midi_ratio_loss', 0)
            n_batches += 1
            self.global_step += 1

            # Show LR info periodically
            lrs = self.scheduler.get_last_lr()
            pbar.set_postfix({
                'loss': f"{metrics['total_loss']:.4f}",
                'gap': f"{metrics.get('gap', 0):.3f}",
                'lr_midi': f"{lrs[0]:.1e}",
            })

        result = {
            'train_loss': total_loss / max(n_batches, 1),
            'train_ar_loss': total_ar_loss / max(n_batches, 1),
            'train_mr_loss': total_mr_loss / max(n_batches, 1),
        }

        # Log LR per group
        lrs = self.scheduler.get_last_lr()
        for i, pg in enumerate(self.optimizer.param_groups):
            name = pg.get('name', f'group_{i}')
            result[f'lr_{name}'] = lrs[i]

        return result

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Evaluate model on validation set."""
        self.model.eval()

        audio_embeddings = []
        midi_embeddings = []
        piece_indices = []
        segment_indices = []

        for batch_idx, batch in enumerate(tqdm(self.val_loader, desc="Evaluating")):
            if self.max_val_batches is not None and batch_idx >= self.max_val_batches:
                break
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
            piece_indices.append(batch['piece_idx'].cpu())
            segment_indices.append(batch['segment_idx'].cpu())

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
        effective_train_batches = len(self.train_loader)
        if self.max_batches_per_epoch is not None:
            effective_train_batches = min(self.max_batches_per_epoch, len(self.train_loader))
        effective_val_batches = len(self.val_loader)
        if self.max_val_batches is not None:
            effective_val_batches = min(self.max_val_batches, len(self.val_loader))

        logger.info(f"Starting Gate 4 training for {self.max_epochs} epochs")
        logger.info(f"  ratio_weight: {self.ratio_weight}")
        logger.info(f"  LR groups: midi={self.optimizer.param_groups[0]['lr']}, "
                     f"proj={self.optimizer.param_groups[1]['lr']}, "
                     f"ratio={self.optimizer.param_groups[2]['lr']}")
        logger.info(f"  Warmup: {self.scheduler.warmup_steps} steps")
        logger.info(
            f"  Train batches/epoch: {effective_train_batches} "
            f"(loader={len(self.train_loader)}, limit={self.max_batches_per_epoch})"
        )
        logger.info(
            f"  Val batches: {effective_val_batches} "
            f"(loader={len(self.val_loader)}, limit={self.max_val_batches})"
        )

        start_time = time.time()

        for epoch in range(self.max_epochs):
            epoch_start = time.time()
            train_metrics = self.train_epoch(epoch)

            # Save checkpoint BEFORE eval (so weights survive if eval crashes)
            self.save_checkpoint(f'checkpoint_epoch{epoch+1}.pt', epoch)

            val_metrics = self.evaluate()

            epoch_time = (time.time() - epoch_start) / 60

            epoch_metrics = {
                'epoch': epoch + 1,
                'epoch_time_min': round(epoch_time, 1),
                **train_metrics,
                **val_metrics,
            }
            self.history.append(epoch_metrics)

            recall_a2m = val_metrics.get('val_a2m_recall@10', 0)
            recall_m2a = val_metrics.get('val_m2a_recall@10', 0)
            recall_avg = (recall_a2m + recall_m2a) / 2
            gap = val_metrics.get('val_gap', 0)
            time_gap = val_metrics.get('time_discrimination_gap', 0)

            logger.info(
                f"Epoch {epoch+1}/{self.max_epochs} ({epoch_time:.1f}min): "
                f"loss={train_metrics['train_loss']:.4f}, "
                f"R@10 a2m={recall_a2m:.3f}, m2a={recall_m2a:.3f}, "
                f"gap={gap:.3f}, time_gap={time_gap:.3f}"
            )
            if self.ratio_weight > 0:
                logger.info(
                    f"  ratio losses: ar={train_metrics['train_ar_loss']:.4f}, "
                    f"mr={train_metrics['train_mr_loss']:.4f}"
                )

            if recall_avg > self.best_recall:
                self.best_recall = recall_avg
                self.save_checkpoint('best_model.pt', epoch)
                logger.info(f"  -> New best: R@10 avg={recall_avg:.3f}")

        self.save_checkpoint('final_model.pt', self.max_epochs - 1)

        # Save history
        with open(self.output_dir / 'training_history.json', 'w') as f:
            json.dump(self.history, f, indent=2)

        total_time = (time.time() - start_time) / 60
        logger.info(f"Training complete: {total_time:.1f} minutes")

        return {
            'best_recall': self.best_recall,
            'final_metrics': self.history[-1],
            'training_time_minutes': round(total_time, 1),
        }

    def save_checkpoint(self, filename: str, epoch: int):
        """
        Save checkpoint — dual format for eval compatibility (DEC-002 Bloqueante A).

        Saves:
        1. Full MultiViewModel state (for resume/continuation)
        2. Base-only CrossModalModel state (for evaluate_structured_pool.py)
        """
        path = self.output_dir / filename

        # Full checkpoint (for resume)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'epoch': epoch + 1,
            'best_recall': self.best_recall,
            'ratio_weight': self.ratio_weight,
        }, path)

        # Base-only checkpoint (for evaluation — compatible with CrossModalModel)
        base_path = path.with_name(path.stem + '_base.pt')
        torch.save({
            'model_state_dict': self.model.base_model.state_dict(),
            'epoch': epoch + 1,
            'ratio_weight': self.ratio_weight,
        }, base_path)


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_gate4(
    maestro_dir: Path,
    checkpoint_path: Path,
    output_dir: Path,
    segment_len: float = 4.0,
    hop: float = 1.0,
    batch_size: int = 16,
    num_workers: int = 8,
    epochs: int = 30,
    ratio_weight: float = 0.1,
    lr_midi: float = 5e-5,
    lr_proj: float = 1e-4,
    lr_ratio: float = 5e-4,
    warmup_steps: int = 500,
    max_batches_per_epoch: Optional[int] = None,
    max_val_batches: Optional[int] = None,
    seed: Optional[int] = 42,
    device: Optional[str] = None,
    descriptor: str = 'baseline',
) -> Dict:
    """Run Gate 4 training."""
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = {
        'gate': 4,
        'name': 'Ratio Auxiliary View',
        'checkpoint': str(checkpoint_path),
        'epochs': epochs,
        'ratio_weight': ratio_weight,
        'descriptor': descriptor,
        'segment_len': segment_len,
        'hop': hop,
        'batch_size': batch_size,
        'lr_midi': lr_midi,
        'lr_proj': lr_proj,
        'lr_ratio': lr_ratio,
        'warmup_steps': warmup_steps,
        'max_batches_per_epoch': max_batches_per_epoch,
        'max_val_batches': max_val_batches,
        'seed': seed,
    }

    # Save config
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    logger.info(f"Gate 4 config: {json.dumps(config, indent=2)}")

    if max_batches_per_epoch is not None and max_batches_per_epoch <= 0:
        raise ValueError("max_batches_per_epoch must be > 0")
    if max_val_batches is not None and max_val_batches <= 0:
        raise ValueError("max_val_batches must be > 0")

    # Reproducibility for data order and init between runs (A/B causal comparison)
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        logger.info(f"Using seed={seed}")

    # Create dataloaders (aligned to Gate 2: 4.0/1.0)
    logger.info("Creating dataloaders...")
    train_loader, val_loader, _ = create_dataloaders(
        maestro_dir=maestro_dir,
        segment_len=segment_len,
        hop=hop,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # Create base model — NO DANN (Gate 2 checkpoint doesn't have it)
    logger.info("Creating model...")
    base_model = CrossModalModel(
        audio_encoder='lite',
        use_dann=False,
        device=device,
    )

    # Load Gate 2 checkpoint
    logger.info(f"Loading Gate 2 checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    base_model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    # Freeze ONLY MERT encoder — everything else trainable (Plan v2 section 1)
    for param in base_model.audio_encoder.parameters():
        param.requires_grad = False

    # Count trainable vs frozen params
    trainable = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in base_model.parameters() if not p.requires_grad)
    logger.info(f"Base model: {trainable:,} trainable, {frozen:,} frozen (MERT)")

    # Create ratio encoder
    n_channels = 3 if descriptor == 'enriched' else 1
    ratio_encoder = RatioEncoder(
        n_bins=256,
        n_channels=n_channels,
        hidden_dim=128,
        output_dim=64,
    )

    # Create multi-view model
    model = MultiViewModel(
        base_model=base_model,
        ratio_encoder=ratio_encoder,
        proj_dim=base_model.proj_output_dim,
    )

    ratio_params = sum(p.numel() for p in ratio_encoder.parameters())
    ratio_proj_params = sum(p.numel() for p in model.ratio_projection.parameters())
    logger.info(f"Ratio modules: {ratio_params + ratio_proj_params:,} params (descriptor={descriptor}, channels={n_channels})")

    # Train
    trainer = Gate4Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        output_dir=output_dir,
        lr_midi=lr_midi,
        lr_proj=lr_proj,
        lr_ratio=lr_ratio,
        ratio_weight=ratio_weight,
        max_epochs=epochs,
        warmup_steps=warmup_steps,
        max_batches_per_epoch=max_batches_per_epoch,
        max_val_batches=max_val_batches,
        descriptor=descriptor,
        device=device,
    )

    training_results = trainer.train()

    # Final evaluation
    final_metrics = trainer.evaluate()

    # Build results
    a2m_recall = final_metrics.get('val_a2m_recall@10', 0)
    m2a_recall = final_metrics.get('val_m2a_recall@10', 0)
    recall_avg = (a2m_recall + m2a_recall) / 2
    time_gap = final_metrics.get('time_discrimination_gap', 0)

    results = {
        **config,
        'training': training_results,
        'retrieval': {
            'a2m_recall@10': a2m_recall,
            'm2a_recall@10': m2a_recall,
            'recall_avg': recall_avg,
        },
        'time_discrimination': {
            'same_piece_same_time_sim': final_metrics.get('same_piece_same_time_sim', 0),
            'same_piece_diff_time_sim': final_metrics.get('same_piece_diff_time_sim', 0),
            'gap': time_gap,
        },
        'note': 'GO/NO-GO decision by structured pool evaluation only (DEC-002)',
    }

    # Save results
    results_path = output_dir / 'gate4_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("GATE 4: RATIO AUXILIARY VIEW - RESULTS")
    print("=" * 60)
    print(f"\nConfig: ratio_weight={ratio_weight}, epochs={epochs}")
    print(f"\n1. Retrieval (val set — monitoring only):")
    print(f"   Audio->MIDI R@10: {a2m_recall:.1%}")
    print(f"   MIDI->Audio R@10: {m2a_recall:.1%}")
    print(f"\n2. Time Discrimination:")
    print(f"   Same piece, same time: {final_metrics.get('same_piece_same_time_sim', 0):.3f}")
    print(f"   Same piece, diff time: {final_metrics.get('same_piece_diff_time_sim', 0):.3f}")
    print(f"   Gap: {time_gap:.3f}")
    print(f"\n3. Training time: {training_results['training_time_minutes']:.1f} min")
    print(f"\nNOTE: Final GO/NO-GO by structured pool evaluation.")
    print("=" * 60)

    return results


def main():
    parser = argparse.ArgumentParser(description="Gate 4: Ratio Auxiliary View (Plan v2)")
    parser.add_argument('--maestro-dir', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to Gate 2 checkpoint')
    parser.add_argument('--output', type=str,
                        default='data/bias_control_medium/training_outputs/gate4')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--ratio-weight', type=float, default=0.1,
                        help='Weight for ratio auxiliary losses (0.0 = control run)')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size (Gate 2 used 16 due to MERT VRAM)')
    parser.add_argument('--segment-len', type=float, default=4.0,
                        help='Segment length in seconds (Gate 2 used 4.0)')
    parser.add_argument('--hop', type=float, default=1.0,
                        help='Hop between segments in seconds (Gate 2 used 1.0)')
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--lr-midi', type=float, default=5e-5)
    parser.add_argument('--lr-proj', type=float, default=1e-4)
    parser.add_argument('--lr-ratio', type=float, default=5e-4)
    parser.add_argument('--warmup-steps', type=int, default=500)
    parser.add_argument(
        '--max-batches-per-epoch',
        type=int,
        default=None,
        help='Max training batches per epoch (default: all)'
    )
    parser.add_argument(
        '--max-val-batches',
        type=int,
        default=None,
        help='Max validation batches per epoch (default: all)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducible run ordering'
    )
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument(
        '--descriptor',
        type=str,
        default='baseline',
        choices=['baseline', 'enriched'],
        help='Ratio descriptor variant: baseline (pitch-only, 1ch) or enriched (vel+dur weighted, 3ch)'
    )

    args = parser.parse_args()

    results = run_gate4(
        maestro_dir=Path(args.maestro_dir),
        checkpoint_path=Path(args.checkpoint),
        output_dir=Path(args.output),
        segment_len=args.segment_len,
        hop=args.hop,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        epochs=args.epochs,
        ratio_weight=args.ratio_weight,
        lr_midi=args.lr_midi,
        lr_proj=args.lr_proj,
        lr_ratio=args.lr_ratio,
        warmup_steps=args.warmup_steps,
        max_batches_per_epoch=args.max_batches_per_epoch,
        max_val_batches=args.max_val_batches,
        seed=args.seed,
        device=args.device,
        descriptor=args.descriptor,
    )


if __name__ == '__main__':
    main()
