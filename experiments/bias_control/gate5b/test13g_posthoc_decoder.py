#!/usr/bin/env python3
"""Test 13G Phase B — Post-Hoc Pre-Pooling Decoder.

Scientific question: "Given pre-pooling representations, how decodable
is the piano roll?"

Phase A showed decoding piano rolls from z[256] (post-pooling) gives
identical poor F1 (~0.114) regardless of λ. Mean-pooling (750:1)
destroys temporal info. This phase probes pre-pooling features directly.

Interpretive constraint: If a4r wins, the claim is "the pre-pooling
representation of a4r is more decodable musically." NOT attributable
to "ratios" alone — a4r changes mechanism (reverse cross-att) AND
compression regime (2400→188).

Usage:
    # Step 0: Precompute PR targets (once, ~15 min)
    python experiments/bias_control/gate5b/test13g_posthoc_decoder.py --phase precompute

    # Step 1: Train each arm
    python experiments/bias_control/gate5b/test13g_posthoc_decoder.py --descriptor d0 --phase train
    python experiments/bias_control/gate5b/test13g_posthoc_decoder.py --descriptor a4r --phase train
    python experiments/bias_control/gate5b/test13g_posthoc_decoder.py --descriptor d4a4 --phase train

    # Optional: D0 pooled to 188 (control for sequence length)
    python experiments/bias_control/gate5b/test13g_posthoc_decoder.py --descriptor d0 --pool-to 188 --phase train

    # Qualitative generation
    python experiments/bias_control/gate5b/test13g_posthoc_decoder.py --descriptor a4r --phase generate
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# --- Project imports ---
from experiments.bias_control.gate5b.decoder_model import sinusoidal_positional_encoding
from experiments.bias_control.gate5b.test13g_generative_encoder import (
    build_pr_targets, N_FRAMES, N_PITCHES, SR, HOP_LENGTH, MIN_PITCH,
)
from experiments.bias_control.gate5b.test11_decoder_suite import _compute_onset_f1
from experiments.bias_control.gate5b.checkpoint_loader import load_model_from_checkpoint
from experiments.bias_control.gate5b.render_midi_audio import (
    write_notes_to_midi, render_midi_to_audio,
)
from src.bias_control.datasets.maestro_segments import (
    MaestroSegmentDataset, collate_segments,
)
from experiments.bias_control.gate43_scratch.gate43_scratch_training import seed_everything

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════════════

POSTHOC_CONFIG = {
    'lr': 1e-4,
    'weight_decay': 1e-4,
    'epochs': 40,
    'batch_size': 16,
    'patience_rounds': 4,       # eval rounds (not epochs) without improvement → stop
    'grad_clip': 1.0,
    'pr_pos_weight': 50.0,      # from Test 11
    'pr_threshold': 0.1,        # from Test 11
    'onset_tolerance': 2,       # from Test 11
    'eval_every': 5,            # eval every N epochs
    'train_segments': 8000,
    'val_subsample': 2000,
    'seed': 42,
}

CHECKPOINT_MAP = {
    'd0':     'models/gate5b/D0/best_model.pt',
    'd4a4':   'models/gate5b/d4a4/best_model.pt',
    'a4r':    'models/gate5b/a4r/best_model.pt',
    'd4-a4r': 'models/gate5b/d4-a4r/best_model.pt',
}

# Map CLI descriptor names to checkpoint_loader descriptor names
DESCRIPTOR_MAP = {
    'd0':     'D0',
    'd4a4':   'd4a4',
    'a4r':    'a4r',
    'd4-a4r': 'd4-a4r',
}

MAESTRO_DIR = Path('data/maestro_v3/maestro-v3.0.0')
RESULTS_BASE = Path('data/gate5b_results')
SEGMENT_LEN = 4.0


# ══════════════════════════════════════════════════════════════════════════════
# PostHocPRDecoder (~2.1M params)
# ══════════════════════════════════════════════════════════════════════════════

class PostHocPRDecoder(nn.Module):
    """Cross-attention decoder for piano roll from pre-pooling encoder features.

    Architecture:
        encoder_feats [B, N, 1024]   N=2400 (D0/d4a4) or 188 (a4r)
          ├── k_proj: Linear(1024, d_model) → K [B, N, d_model]
          ├── v_proj: Linear(1024, d_model) → V [B, N, d_model]
          └── frame_queries [n_frames, d_model] (learned) + sinusoidal PE
               ├── 1× CrossAttention(Q=queries, K, V) + residual + LN
               ├── 2× SelfAttention layers (norm_first=True)
               └── Linear(d_model, n_pitches) → logits [B, 188, 88]
    """

    def __init__(
        self,
        feat_dim: int = 1024,
        d_model: int = 256,
        n_heads: int = 4,
        n_self_layers: int = 2,
        d_ff: int = 1024,
        n_frames: int = N_FRAMES,    # 188
        n_pitches: int = N_PITCHES,  # 88
        dropout: float = 0.1,
    ):
        super().__init__()

        # Project encoder features to d_model
        self.k_proj = nn.Linear(feat_dim, d_model)
        self.v_proj = nn.Linear(feat_dim, d_model)

        # Learned frame queries + sinusoidal PE
        self.frame_queries = nn.Parameter(torch.randn(n_frames, d_model) * 0.02)
        pe = sinusoidal_positional_encoding(n_frames, d_model)
        self.register_buffer('pos_encoding', pe)

        # Cross-attention: Q=queries, K/V=projected encoder features
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )
        self.cross_attn_norm = nn.LayerNorm(d_model)

        # Self-attention layers
        self_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_ff, dropout=dropout,
            activation='gelu', batch_first=True, norm_first=True,
        )
        self.self_attn = nn.TransformerEncoder(self_layer, num_layers=n_self_layers)

        # Output projection → raw logits (no sigmoid)
        self.output_head = nn.Linear(d_model, n_pitches)

        self._n_params = sum(p.numel() for p in self.parameters())

    def forward(self, encoder_feats: torch.Tensor) -> torch.Tensor:
        """
        Args:
            encoder_feats: [B, N, feat_dim] pre-pooling encoder features.
        Returns:
            logits: [B, n_frames, n_pitches] raw logits.
        """
        B = encoder_feats.shape[0]

        # Project K, V from encoder features
        K = self.k_proj(encoder_feats)   # [B, N, d_model]
        V = self.v_proj(encoder_feats)   # [B, N, d_model]

        # Expand queries: [n_frames, d_model] → [B, n_frames, d_model]
        Q = (self.frame_queries + self.pos_encoding).unsqueeze(0).expand(B, -1, -1)

        # Cross-attention + residual + LN
        attn_out, _ = self.cross_attn(Q, K, V)
        x = self.cross_attn_norm(Q + attn_out)

        # Self-attention layers
        x = self.self_attn(x)

        # Output logits
        return self.output_head(x)   # [B, n_frames, n_pitches]


# ══════════════════════════════════════════════════════════════════════════════
# EncoderFeatureExtractor (hook context manager)
# ══════════════════════════════════════════════════════════════════════════════

class EncoderFeatureExtractor:
    """Context manager that hooks audio_encoder.transformer to capture
    pre-pooling features. Features are detached but stay on GPU.

    Usage:
        extractor = EncoderFeatureExtractor(model)
        with extractor:
            model(audio, ...)
            feats = extractor.pop_features()  # [B, N, 1024]
    """

    def __init__(self, model: nn.Module):
        base = model
        while hasattr(base, 'base_model'):
            base = base.base_model
        self.transformer = base.audio_encoder.transformer
        self._captured: List[torch.Tensor] = []
        self._hook_handle = None

    def __enter__(self):
        def hook_fn(module, inp, out):
            self._captured.append(out.detach())
        self._hook_handle = self.transformer.register_forward_hook(hook_fn)
        return self

    def __exit__(self, *args):
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    def pop_features(self) -> torch.Tensor:
        """Return and remove the last captured tensor."""
        assert len(self._captured) == 1, \
            f"Expected 1 captured tensor, got {len(self._captured)}"
        return self._captured.pop()


# ══════════════════════════════════════════════════════════════════════════════
# PRProbeDataset + collate
# ══════════════════════════════════════════════════════════════════════════════

class PRProbeDataset(Dataset):
    """Wraps a base dataset, attaching precomputed PR targets per item.

    With shuffle=True in DataLoader, each item carries its own target
    so there is no index confusion.
    """

    def __init__(self, base_dataset, pr_targets: np.ndarray):
        self.base = base_dataset
        self.targets = pr_targets   # [N, 188, 88]
        assert len(self.base) == len(self.targets), \
            f"Dataset ({len(self.base)}) and targets ({len(self.targets)}) mismatch"

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        item['pr_target'] = torch.from_numpy(self.targets[idx].astype(np.float32))
        return item


def collate_with_pr(batch: list) -> dict:
    """Extend collate_segments with PR target stacking."""
    pr_targets = torch.stack([b.pop('pr_target') for b in batch])
    result = collate_segments(batch)
    result['pr_target'] = pr_targets
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Precompute PR targets
# ══════════════════════════════════════════════════════════════════════════════

def precompute_pr_targets(
    dataset,
    output_path: Path,
    batch_size: int = 32,
    num_workers: int = 8,
    desc: str = "Building PR targets",
) -> np.ndarray:
    """Build PR targets for all segments in dataset and save to NPZ."""

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_segments,
        pin_memory=False,
    )

    all_targets = []
    n_skipped = 0

    for batch in tqdm(loader, desc=desc):
        midi_onset = batch.get('midi_onset')
        midi_duration_sec = batch.get('midi_duration_sec')

        if midi_onset is None or midi_duration_sec is None:
            B = batch['midi_pitch'].shape[0]
            n_skipped += B
            all_targets.append(
                np.zeros((B, N_FRAMES, N_PITCHES), dtype=np.float32)
            )
            continue

        pr = build_pr_targets(
            batch['midi_pitch'], batch['midi_velocity'],
            batch['midi_onset'], batch['midi_duration_sec'],
            batch['midi_mask'],
        )   # [B, 188, 88]
        all_targets.append(pr.numpy())

    targets = np.concatenate(all_targets, axis=0)   # [N, 188, 88]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(output_path), targets=targets)

    size_mb = output_path.stat().st_size / 1e6
    if n_skipped > 0:
        logger.warning(f"Skipped {n_skipped} segments (no onset data)")
    logger.info(
        f"Saved PR targets: {targets.shape} → {output_path} ({size_mb:.0f} MB)"
    )
    return targets


# ══════════════════════════════════════════════════════════════════════════════
# Pool features utility
# ══════════════════════════════════════════════════════════════════════════════

def pool_features_to_length(
    feats: torch.Tensor, target_len: int,
) -> torch.Tensor:
    """Adaptive average pool [B, N, D] → [B, target_len, D]."""
    B, N, D = feats.shape
    if N == target_len:
        return feats
    feats_t = feats.transpose(1, 2)                        # [B, D, N]
    feats_t = F.adaptive_avg_pool1d(feats_t, target_len)   # [B, D, target_len]
    return feats_t.transpose(1, 2)                          # [B, target_len, D]


# ══════════════════════════════════════════════════════════════════════════════
# Metrics
# ══════════════════════════════════════════════════════════════════════════════

def compute_all_metrics(
    pred_logits: np.ndarray,    # [N, 188, 88]
    true_pr: np.ndarray,        # [N, 188, 88]
    threshold: float = 0.1,
    onset_tolerance: int = 2,
    pos_weight: float = 50.0,
    max_onset_samples: int = 500,
) -> Dict[str, float]:
    """Compute frame-F1, onset-F1, BCE, cosine."""

    pred_prob = 1.0 / (1.0 + np.exp(-np.clip(pred_logits, -20, 20)))

    # --- Frame F1 ---
    pred_bin = (pred_prob > threshold).astype(np.float32)
    true_bin = (true_pr > threshold).astype(np.float32)
    tp = float((pred_bin * true_bin).sum())
    fp = float((pred_bin * (1 - true_bin)).sum())
    fn = float(((1 - pred_bin) * true_bin).sum())
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    frame_f1 = 2 * precision * recall / (precision + recall + 1e-8)

    # --- BCE (chunked to avoid OOM on CPU) ---
    bce_sum = 0.0
    n_elements = 0
    pw = torch.tensor([pos_weight])
    chunk_size = 2000
    for i in range(0, len(pred_logits), chunk_size):
        chunk_l = torch.from_numpy(pred_logits[i:i + chunk_size].copy())
        chunk_t = torch.from_numpy(true_pr[i:i + chunk_size].copy())
        bce_sum += F.binary_cross_entropy_with_logits(
            chunk_l, chunk_t, pos_weight=pw, reduction='sum',
        ).item()
        n_elements += chunk_l.numel()
    bce = bce_sum / max(n_elements, 1)

    # --- Cosine similarity ---
    pred_flat = pred_prob.reshape(-1)
    true_flat = true_pr.reshape(-1)
    dot = np.dot(pred_flat, true_flat)
    norm_p = np.linalg.norm(pred_flat) + 1e-8
    norm_t = np.linalg.norm(true_flat) + 1e-8
    cosine = float(dot / (norm_p * norm_t))

    # --- Onset F1 (subset) ---
    n_onset = min(len(pred_prob), max_onset_samples)
    onset_f1_values = []
    for i in range(n_onset):
        result = _compute_onset_f1(
            pred_prob[i], true_pr[i],
            threshold=threshold,
            tolerance=onset_tolerance,
        )
        onset_f1_values.append(result['onset_f1'])
    onset_f1 = float(np.mean(onset_f1_values)) if onset_f1_values else 0.0

    return {
        'frame_f1': frame_f1,
        'frame_precision': precision,
        'frame_recall': recall,
        'onset_f1': onset_f1,
        'bce': bce,
        'cosine': cosine,
        'n_samples': len(pred_logits),
        'n_onset_samples': n_onset,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Verify hook (pre-flight)
# ══════════════════════════════════════════════════════════════════════════════

def verify_hook(
    model: nn.Module,
    dataset,
    device: torch.device,
    descriptor: str,
) -> dict:
    """Pre-flight check: verify hook captures correct shapes."""

    loader = DataLoader(
        Subset(dataset, list(range(min(4, len(dataset))))),
        batch_size=2,
        shuffle=False,
        collate_fn=collate_segments,
    )

    model.eval()
    extractor = EncoderFeatureExtractor(model)

    with extractor:
        batch = next(iter(loader))
        audio = batch['audio'].to(device)
        midi_pitch = batch['midi_pitch'].to(device)
        midi_velocity = batch['midi_velocity'].to(device)
        midi_duration = batch['midi_duration'].to(device)
        midi_mask = batch['midi_mask'].to(device)

        with torch.no_grad():
            model(audio, midi_pitch, midi_velocity, midi_duration, midi_mask)

        feats = extractor.pop_features()
        B, N, D = feats.shape

    expected_D = 1024
    expected_N_ranges = {
        'd0':     (2300, 2500),
        'd4a4':   (2300, 2500),
        'a4r':    (180, 200),
        'd4-a4r': (180, 200),
    }
    lo, hi = expected_N_ranges.get(descriptor, (1, 10000))

    ok = D == expected_D and lo <= N <= hi
    status = "PASS" if ok else "FAIL"
    logger.info(
        f"Hook verify [{status}]: descriptor={descriptor}, "
        f"shape=[B={B}, N={N}, D={D}]"
    )

    if not ok:
        raise RuntimeError(
            f"Hook shape mismatch for {descriptor}: got [{B}, {N}, {D}], "
            f"expected N in [{lo}, {hi}], D={expected_D}"
        )

    # Verify no gradient on encoder (sanity)
    has_encoder_grad = any(p.grad is not None for p in model.parameters())
    logger.info(f"  Encoder has gradients: {has_encoder_grad} (should be False)")

    return {'shape': [B, N, D], 'ok': ok}


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation (standalone — for full-val and generate phases)
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_posthoc(
    model: nn.Module,
    decoder: PostHocPRDecoder,
    val_loader: DataLoader,
    device: torch.device,
    pool_to: Optional[int] = None,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """Run evaluation, return (metrics, all_logits, all_targets)."""

    model.eval()
    decoder.eval()

    all_logits = []
    all_targets = []

    extractor = EncoderFeatureExtractor(model)
    with extractor:
        for batch in tqdm(val_loader, desc="  Evaluating", leave=False):
            audio = batch['audio'].to(device, non_blocking=True)
            midi_pitch = batch['midi_pitch'].to(device, non_blocking=True)
            midi_velocity = batch['midi_velocity'].to(device, non_blocking=True)
            midi_duration = batch['midi_duration'].to(device, non_blocking=True)
            midi_mask = batch['midi_mask'].to(device, non_blocking=True)
            pr_target = batch['pr_target']   # stays CPU for metrics

            model(audio, midi_pitch, midi_velocity, midi_duration, midi_mask)
            feats = extractor.pop_features()

            if pool_to is not None:
                feats = pool_features_to_length(feats, pool_to)

            logits = decoder(feats)   # [B, 188, 88]
            all_logits.append(logits.cpu().numpy())
            all_targets.append(pr_target.numpy())

    all_logits_np = np.concatenate(all_logits, axis=0)
    all_targets_np = np.concatenate(all_targets, axis=0)

    metrics = compute_all_metrics(
        all_logits_np, all_targets_np,
        threshold=POSTHOC_CONFIG['pr_threshold'],
        onset_tolerance=POSTHOC_CONFIG['onset_tolerance'],
        pos_weight=POSTHOC_CONFIG['pr_pos_weight'],
    )

    return metrics, all_logits_np, all_targets_np


# ══════════════════════════════════════════════════════════════════════════════
# Training loop
# ══════════════════════════════════════════════════════════════════════════════

def train_posthoc_decoder(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    descriptor: str,
    output_dir: Path,
    device: torch.device,
    pool_to: Optional[int] = None,
    config: Optional[dict] = None,
) -> dict:
    """Main training loop for PostHocPRDecoder.

    Patience semantics: patience_rounds counts eval rounds, not epochs.
    With eval_every=5 and patience_rounds=4, earliest stop at epoch 25
    (if no improvement from epoch 5 through epoch 20).
    """

    cfg = {**POSTHOC_CONFIG, **(config or {})}

    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / 'checkpoints'
    ckpt_dir.mkdir(exist_ok=True)
    eval_dir = output_dir / 'eval_per_epoch'
    eval_dir.mkdir(exist_ok=True)

    # Save config
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(
            {**cfg, 'descriptor': descriptor, 'pool_to': pool_to},
            f, indent=2,
        )

    # Create decoder
    decoder = PostHocPRDecoder(
        feat_dim=1024, d_model=256, n_heads=4,
        n_self_layers=2, d_ff=1024,
        n_frames=N_FRAMES, n_pitches=N_PITCHES,
        dropout=0.1,
    ).to(device)

    logger.info(f"PostHocPRDecoder: {decoder._n_params / 1e6:.2f}M params")

    optimizer = AdamW(
        decoder.parameters(),
        lr=cfg['lr'],
        weight_decay=cfg['weight_decay'],
    )

    pos_weight = torch.tensor([cfg['pr_pos_weight']], device=device)

    # Freeze encoder completely
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    best_f1 = 0.0
    best_epoch = 0
    patience_counter = 0
    history = []

    total_start = time.time()
    pool_label = f" pool-to={pool_to}" if pool_to else ""
    logger.info(
        f"Starting PostHoc training: descriptor={descriptor}{pool_label}, "
        f"epochs={cfg['epochs']}, eval_every={cfg['eval_every']}, "
        f"patience={cfg['patience_rounds']} rounds"
    )

    extractor = EncoderFeatureExtractor(model)
    stopped_early = False

    with extractor:
        for epoch in range(1, cfg['epochs'] + 1):
            epoch_start = time.time()
            decoder.train()

            total_loss = 0.0
            n_batches = 0

            pbar = tqdm(
                train_loader,
                desc=f"Epoch {epoch}/{cfg['epochs']}",
                dynamic_ncols=True,
            )
            for batch in pbar:
                audio = batch['audio'].to(device, non_blocking=True)
                midi_pitch = batch['midi_pitch'].to(device, non_blocking=True)
                midi_velocity = batch['midi_velocity'].to(device, non_blocking=True)
                midi_duration = batch['midi_duration'].to(device, non_blocking=True)
                midi_mask = batch['midi_mask'].to(device, non_blocking=True)
                pr_target = batch['pr_target'].to(device, non_blocking=True)

                # Forward through frozen encoder (hook captures transformer output)
                with torch.no_grad():
                    model(audio, midi_pitch, midi_velocity, midi_duration, midi_mask)
                feats = extractor.pop_features()   # [B, N, 1024] detached

                if pool_to is not None:
                    feats = pool_features_to_length(feats, pool_to)

                # Decoder forward + loss (gradients flow through decoder only)
                logits = decoder(feats)   # [B, 188, 88]
                loss = F.binary_cross_entropy_with_logits(
                    logits, pr_target, pos_weight=pos_weight,
                )

                loss.backward()
                clip_grad_norm_(decoder.parameters(), max_norm=cfg['grad_clip'])
                optimizer.step()
                optimizer.zero_grad()

                total_loss += loss.item()
                n_batches += 1

                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'avg': f"{total_loss / n_batches:.4f}",
                })

            avg_loss = total_loss / max(n_batches, 1)
            epoch_time = (time.time() - epoch_start) / 60

            epoch_record = {
                'epoch': epoch,
                'avg_loss': avg_loss,
                'epoch_time_min': round(epoch_time, 2),
            }

            # Save checkpoint every epoch
            ckpt_data = {
                'decoder_state_dict': decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'best_f1': best_f1,
                'config': cfg,
                'descriptor': descriptor,
                'pool_to': pool_to,
            }
            torch.save(ckpt_data, ckpt_dir / f'checkpoint_epoch{epoch}.pt')

            # --- Evaluation every N epochs ---
            if epoch % cfg['eval_every'] == 0:
                logger.info(f"Evaluating at epoch {epoch}...")
                decoder.eval()
                eval_logits = []
                eval_targets = []

                for val_batch in tqdm(
                    val_loader, desc=f"  Val epoch {epoch}", leave=False,
                ):
                    v_audio = val_batch['audio'].to(device, non_blocking=True)
                    v_mp = val_batch['midi_pitch'].to(device, non_blocking=True)
                    v_mv = val_batch['midi_velocity'].to(device, non_blocking=True)
                    v_md = val_batch['midi_duration'].to(device, non_blocking=True)
                    v_mm = val_batch['midi_mask'].to(device, non_blocking=True)
                    v_pr = val_batch['pr_target']   # CPU

                    with torch.no_grad():
                        model(v_audio, v_mp, v_mv, v_md, v_mm)
                    v_feats = extractor.pop_features()

                    if pool_to is not None:
                        v_feats = pool_features_to_length(v_feats, pool_to)

                    with torch.no_grad():
                        v_logits = decoder(v_feats)

                    eval_logits.append(v_logits.cpu().numpy())
                    eval_targets.append(v_pr.numpy())

                eval_logits_np = np.concatenate(eval_logits, axis=0)
                eval_targets_np = np.concatenate(eval_targets, axis=0)

                metrics = compute_all_metrics(
                    eval_logits_np, eval_targets_np,
                    threshold=cfg['pr_threshold'],
                    onset_tolerance=cfg['onset_tolerance'],
                    pos_weight=cfg['pr_pos_weight'],
                )

                epoch_record['eval'] = metrics

                with open(eval_dir / f'eval_epoch{epoch}.json', 'w') as f:
                    json.dump(metrics, f, indent=2)

                logger.info(
                    f"  Epoch {epoch}: frame_f1={metrics['frame_f1']:.4f}, "
                    f"onset_f1={metrics['onset_f1']:.4f}, "
                    f"bce={metrics['bce']:.4f}, cosine={metrics['cosine']:.4f}"
                )

                # Best tracking + patience
                if metrics['frame_f1'] > best_f1:
                    best_f1 = metrics['frame_f1']
                    best_epoch = epoch
                    patience_counter = 0
                    # Update best_f1 in ckpt before saving
                    ckpt_data['best_f1'] = best_f1
                    torch.save(ckpt_data, ckpt_dir / 'best_f1.pt')
                    logger.info(
                        f"  ★ New best frame_f1={best_f1:.4f} at epoch {epoch}"
                    )
                else:
                    patience_counter += 1
                    logger.info(
                        f"  No improvement "
                        f"({patience_counter}/{cfg['patience_rounds']} rounds)"
                    )
                    if patience_counter >= cfg['patience_rounds']:
                        logger.info(f"  Early stopping at epoch {epoch}")
                        history.append(epoch_record)
                        stopped_early = True
                        break

            else:
                logger.info(
                    f"Epoch {epoch}: avg_loss={avg_loss:.4f}, "
                    f"time={epoch_time:.1f}min"
                )

            history.append(epoch_record)

    total_time = (time.time() - total_start) / 60

    results = {
        'descriptor': descriptor,
        'pool_to': pool_to,
        'best_f1': best_f1,
        'best_epoch': best_epoch,
        'stopped_early': stopped_early,
        'total_time_min': round(total_time, 1),
        'decoder_params': decoder._n_params,
        'history': history,
        'config': cfg,
    }

    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Final summary JSON
    with open(output_dir / 'test13g_posthoc_results.json', 'w') as f:
        json.dump({
            'descriptor': descriptor,
            'pool_to': pool_to,
            'best_f1': best_f1,
            'best_epoch': best_epoch,
            'stopped_early': stopped_early,
            'total_time_min': round(total_time, 1),
        }, f, indent=2)

    logger.info(
        f"Training complete: best_f1={best_f1:.4f} at epoch {best_epoch}, "
        f"total time={total_time:.1f}min"
    )

    return results


# ══════════════════════════════════════════════════════════════════════════════
# PR → notes (qualitative)
# ══════════════════════════════════════════════════════════════════════════════

def pr_to_notes(
    pr: np.ndarray,
    threshold: float = 0.1,
    hop_sec: float = SEGMENT_LEN / N_FRAMES,
) -> List[Tuple[int, float, float, int]]:
    """Convert piano roll [188, 88] → list of (pitch, onset, offset, velocity).

    Simple threshold-based onset detection. Qualitative only — not a
    validated transcription pipeline.
    """
    notes = []
    T, P = pr.shape
    pr_binary = pr > threshold

    for p in range(P):
        in_note = False
        onset_t = 0
        for t in range(T):
            if pr_binary[t, p] and not in_note:
                onset_t = t
                in_note = True
            elif not pr_binary[t, p] and in_note:
                pitch = p + MIN_PITCH
                vel = min(int(pr[onset_t:t, p].max() * 127), 127)
                vel = max(vel, 1)
                notes.append((pitch, onset_t * hop_sec, t * hop_sec, vel))
                in_note = False
        if in_note:
            pitch = p + MIN_PITCH
            vel = min(int(pr[onset_t:T, p].max() * 127), 127)
            vel = max(vel, 1)
            notes.append((pitch, onset_t * hop_sec, T * hop_sec, vel))

    return notes


# ══════════════════════════════════════════════════════════════════════════════
# Generate samples (qualitative MIDI/WAV/PNG)
# ══════════════════════════════════════════════════════════════════════════════

def generate_samples(
    model: nn.Module,
    decoder: PostHocPRDecoder,
    val_dataset,
    device: torch.device,
    output_dir: Path,
    descriptor: str,
    pool_to: Optional[int] = None,
    n_samples: int = 8,
    seed: int = 42,
):
    """Generate qualitative samples: MIDI, WAV, and PNG comparison."""

    gen_dir = output_dir / 'generation_samples'
    gen_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(seed)
    indices = sorted(rng.choice(len(val_dataset), min(n_samples, len(val_dataset)), replace=False))

    subset = Subset(val_dataset, indices)
    loader = DataLoader(
        subset, batch_size=1, shuffle=False, collate_fn=collate_segments,
    )

    model.eval()
    decoder.eval()

    extractor = EncoderFeatureExtractor(model)
    with extractor:
        for i, batch in enumerate(tqdm(loader, desc="Generating samples")):
            audio = batch['audio'].to(device, non_blocking=True)
            midi_pitch = batch['midi_pitch'].to(device, non_blocking=True)
            midi_velocity = batch['midi_velocity'].to(device, non_blocking=True)
            midi_duration = batch['midi_duration'].to(device, non_blocking=True)
            midi_mask = batch['midi_mask'].to(device, non_blocking=True)

            with torch.no_grad():
                model(audio, midi_pitch, midi_velocity, midi_duration, midi_mask)
            feats = extractor.pop_features()

            if pool_to is not None:
                feats = pool_features_to_length(feats, pool_to)

            with torch.no_grad():
                logits = decoder(feats)   # [1, 188, 88]

            pred_pr = torch.sigmoid(logits[0]).cpu().numpy()   # [188, 88]

            # Build ground truth PR for comparison
            if batch.get('midi_onset') is not None:
                gt_pr = build_pr_targets(
                    batch['midi_pitch'].cpu(),
                    batch['midi_velocity'].cpu(),
                    batch['midi_onset'].cpu(),
                    batch['midi_duration_sec'].cpu(),
                    batch['midi_mask'].cpu(),
                )[0].numpy()   # [188, 88]
            else:
                gt_pr = np.zeros_like(pred_pr)

            sample_id = f"sample_{i:03d}_idx{indices[i]}"

            # Save predicted MIDI
            pred_notes = pr_to_notes(pred_pr)
            if pred_notes:
                midi_path = gen_dir / f'{sample_id}_pred.mid'
                write_notes_to_midi(pred_notes, midi_path)
                wav_path = gen_dir / f'{sample_id}_pred.wav'
                render_midi_to_audio(midi_path, wav_path)

            # Save ground truth MIDI
            gt_notes = pr_to_notes(gt_pr)
            if gt_notes:
                gt_midi_path = gen_dir / f'{sample_id}_gt.mid'
                write_notes_to_midi(gt_notes, gt_midi_path)
                gt_wav_path = gen_dir / f'{sample_id}_gt.wav'
                render_midi_to_audio(gt_midi_path, gt_wav_path)

            # Save PNG comparison (pred vs GT piano roll)
            try:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt

                fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
                axes[0].imshow(
                    gt_pr.T, aspect='auto', origin='lower',
                    cmap='hot', vmin=0, vmax=1,
                )
                axes[0].set_title(f'Ground Truth — {descriptor}')
                axes[0].set_ylabel('Pitch')

                axes[1].imshow(
                    pred_pr.T, aspect='auto', origin='lower',
                    cmap='hot', vmin=0, vmax=1,
                )
                axes[1].set_title(f'Predicted — {descriptor}')
                axes[1].set_ylabel('Pitch')
                axes[1].set_xlabel('Frame')

                plt.tight_layout()
                plt.savefig(gen_dir / f'{sample_id}_comparison.png', dpi=120)
                plt.close()
            except ImportError:
                pass

    logger.info(f"Generated {n_samples} samples in {gen_dir}")


# ══════════════════════════════════════════════════════════════════════════════
# Main + CLI
# ══════════════════════════════════════════════════════════════════════════════

def _setup_datasets(
    maestro_dir: Path, cfg: dict, seed: int,
) -> Tuple:
    """Create train/val datasets with precomputed PR targets."""

    # Load precomputed targets
    train_targets_path = RESULTS_BASE / 'pr_targets_train_8k.npz'
    val_targets_path = RESULTS_BASE / 'pr_targets_val.npz'

    if not train_targets_path.exists() or not val_targets_path.exists():
        raise FileNotFoundError(
            f"PR targets not found. Run --phase precompute first.\n"
            f"  Expected: {train_targets_path}\n"
            f"  Expected: {val_targets_path}"
        )

    train_targets = np.load(str(train_targets_path))['targets']
    val_targets = np.load(str(val_targets_path))['targets']

    logger.info(
        f"Loaded PR targets: train={train_targets.shape}, val={val_targets.shape}"
    )

    # Create base datasets
    train_dataset = MaestroSegmentDataset(
        maestro_dir=str(maestro_dir), split='train',
        segment_len=SEGMENT_LEN,
    )
    val_dataset = MaestroSegmentDataset(
        maestro_dir=str(maestro_dir), split='validation',
        segment_len=SEGMENT_LEN,
    )

    # Subsample train
    rng = np.random.RandomState(seed)
    n_train = min(cfg['train_segments'], len(train_dataset))
    train_indices = sorted(rng.choice(len(train_dataset), n_train, replace=False))
    train_subset = Subset(train_dataset, train_indices)

    # Verify target count matches
    assert len(train_indices) == len(train_targets), \
        f"Train indices ({len(train_indices)}) != targets ({len(train_targets)})"

    # Wrap with PR targets
    probe_train = PRProbeDataset(train_subset, train_targets)

    # Val quick (subsample for eval during training)
    n_val_sub = min(cfg['val_subsample'], len(val_dataset))
    val_sub_indices = sorted(rng.choice(len(val_dataset), n_val_sub, replace=False))
    val_sub_targets = val_targets[val_sub_indices]
    probe_val_quick = PRProbeDataset(
        Subset(val_dataset, val_sub_indices), val_sub_targets,
    )

    # Val full (for final evaluation)
    probe_val_full = PRProbeDataset(val_dataset, val_targets)

    train_loader = DataLoader(
        probe_train,
        batch_size=cfg['batch_size'],
        shuffle=True,
        num_workers=8,
        collate_fn=collate_with_pr,
        pin_memory=True,
        prefetch_factor=2,
    )
    val_quick_loader = DataLoader(
        probe_val_quick,
        batch_size=cfg['batch_size'],
        shuffle=False,
        num_workers=8,
        collate_fn=collate_with_pr,
        pin_memory=True,
        prefetch_factor=2,
    )
    val_full_loader = DataLoader(
        probe_val_full,
        batch_size=cfg['batch_size'],
        shuffle=False,
        num_workers=8,
        collate_fn=collate_with_pr,
        pin_memory=True,
        prefetch_factor=2,
    )

    return (
        train_loader, val_quick_loader, val_full_loader,
        val_dataset,   # raw (for generate_samples)
    )


def main():
    parser = argparse.ArgumentParser(
        description='Test 13G Phase B — Post-Hoc Pre-Pooling Decoder',
    )
    parser.add_argument(
        '--descriptor', type=str, choices=['d0', 'd4a4', 'a4r', 'd4-a4r'],
        help='Encoder arm to probe',
    )
    parser.add_argument(
        '--phase', type=str, required=True,
        choices=['precompute', 'train', 'generate', 'fullval'],
        help='Phase: precompute targets, train decoder, generate samples, '
             'or run full-val evaluation',
    )
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument(
        '--pool-to', type=int, default=None,
        help='Pool encoder features to this length (e.g., 188 for D0 control)',
    )
    parser.add_argument(
        '--maestro-dir', type=str, default=str(MAESTRO_DIR),
    )
    parser.add_argument('--seed', type=int, default=POSTHOC_CONFIG['seed'])
    parser.add_argument(
        '--output-dir', type=str, default=None,
        help='Override output directory',
    )

    args = parser.parse_args()

    # Logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(name)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
    )

    seed_everything(args.seed)
    maestro_dir = Path(args.maestro_dir)
    device = torch.device(
        args.device if torch.cuda.is_available() else 'cpu'
    )

    os.environ.setdefault(
        'PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True'
    )

    # ── Phase: precompute ──────────────────────────────────────────────────
    if args.phase == 'precompute':
        logger.info("Phase: precompute PR targets")

        train_dataset = MaestroSegmentDataset(
            maestro_dir=str(maestro_dir), split='train',
            segment_len=SEGMENT_LEN,
        )
        val_dataset = MaestroSegmentDataset(
            maestro_dir=str(maestro_dir), split='validation',
            segment_len=SEGMENT_LEN,
        )

        # Subsample train to 8k (deterministic)
        rng = np.random.RandomState(args.seed)
        n_train = min(POSTHOC_CONFIG['train_segments'], len(train_dataset))
        train_indices = sorted(
            rng.choice(len(train_dataset), n_train, replace=False)
        )
        train_subset = Subset(train_dataset, train_indices)

        logger.info(
            f"Train: {len(train_subset)} segments (subsampled from "
            f"{len(train_dataset)})"
        )
        logger.info(f"Val: {len(val_dataset)} segments (full)")

        # Save train indices for reproducibility
        indices_path = RESULTS_BASE / 'pr_train_8k_indices.npy'
        indices_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(indices_path), np.array(train_indices))
        logger.info(f"Saved train indices: {indices_path}")

        precompute_pr_targets(
            train_subset,
            RESULTS_BASE / 'pr_targets_train_8k.npz',
            batch_size=32, num_workers=8,
            desc="PR targets (train 8k)",
        )

        precompute_pr_targets(
            val_dataset,
            RESULTS_BASE / 'pr_targets_val.npz',
            batch_size=32, num_workers=8,
            desc="PR targets (val full)",
        )

        logger.info("Precompute complete.")
        return

    # ── Phases requiring a descriptor ──────────────────────────────────────
    if args.descriptor is None:
        parser.error("--descriptor is required for train/generate/fullval")

    descriptor = args.descriptor
    desc_for_loader = DESCRIPTOR_MAP[descriptor]

    # Load encoder checkpoint
    ckpt_path = CHECKPOINT_MAP[descriptor]
    logger.info(f"Loading encoder: {ckpt_path}")
    model, metadata = load_model_from_checkpoint(ckpt_path, device=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    logger.info(
        f"Loaded {desc_for_loader}: epoch={metadata.get('epoch', '?')}, "
        f"best_S={metadata.get('best_S', '?')}"
    )

    # Output directory
    pool_suffix = f"_pool{args.pool_to}" if args.pool_to else ""
    arm_dir = desc_for_loader   # D0, d4a4, a4r — matches existing dirs
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = RESULTS_BASE / arm_dir / f'test13g_posthoc{pool_suffix}'

    # ── Phase: train ───────────────────────────────────────────────────────
    if args.phase == 'train':
        cfg = {**POSTHOC_CONFIG, 'seed': args.seed}

        (
            train_loader, val_quick_loader, val_full_loader,
            val_dataset_raw,
        ) = _setup_datasets(maestro_dir, cfg, args.seed)

        # Verify hook before training
        verify_hook(model, val_dataset_raw, device, descriptor)

        # Train
        results = train_posthoc_decoder(
            model=model,
            train_loader=train_loader,
            val_loader=val_quick_loader,
            descriptor=descriptor,
            output_dir=output_dir,
            device=device,
            pool_to=args.pool_to,
            config=cfg,
        )

        # Full-val evaluation with best checkpoint
        best_ckpt_path = output_dir / 'checkpoints' / 'best_f1.pt'
        if best_ckpt_path.exists():
            logger.info("Running full-val evaluation with best checkpoint...")
            ckpt = torch.load(str(best_ckpt_path), map_location=device)
            decoder = PostHocPRDecoder().to(device)
            decoder.load_state_dict(ckpt['decoder_state_dict'])

            full_metrics, _, _ = evaluate_posthoc(
                model, decoder, val_full_loader, device,
                pool_to=args.pool_to,
            )

            results['full_val'] = full_metrics

            with open(output_dir / 'test13g_posthoc_results.json', 'w') as f:
                json.dump({
                    'descriptor': descriptor,
                    'pool_to': args.pool_to,
                    'best_f1': results['best_f1'],
                    'best_epoch': results['best_epoch'],
                    'stopped_early': results['stopped_early'],
                    'total_time_min': results['total_time_min'],
                    'full_val': full_metrics,
                }, f, indent=2)

            logger.info(
                f"Full-val: frame_f1={full_metrics['frame_f1']:.4f}, "
                f"onset_f1={full_metrics['onset_f1']:.4f}"
            )

            # Auto-generate qualitative samples after training
            logger.info("Generating qualitative samples...")
            generate_samples(
                model=model,
                decoder=decoder,
                val_dataset=val_dataset_raw,
                device=device,
                output_dir=output_dir,
                descriptor=descriptor,
                pool_to=args.pool_to,
            )
        else:
            logger.warning("No best_f1.pt checkpoint found — skipping full-val and generation")

    # ── Phase: fullval ─────────────────────────────────────────────────────
    elif args.phase == 'fullval':
        cfg = {**POSTHOC_CONFIG, 'seed': args.seed}
        _, _, val_full_loader, _ = _setup_datasets(maestro_dir, cfg, args.seed)

        best_ckpt_path = output_dir / 'checkpoints' / 'best_f1.pt'
        if not best_ckpt_path.exists():
            logger.error(f"No checkpoint at {best_ckpt_path}")
            sys.exit(1)

        ckpt = torch.load(str(best_ckpt_path), map_location=device)
        decoder = PostHocPRDecoder().to(device)
        decoder.load_state_dict(ckpt['decoder_state_dict'])

        logger.info(f"Full-val evaluation: {descriptor}")
        full_metrics, _, _ = evaluate_posthoc(
            model, decoder, val_full_loader, device,
            pool_to=args.pool_to,
        )

        with open(output_dir / 'test13g_posthoc_fullval.json', 'w') as f:
            json.dump(full_metrics, f, indent=2)

        logger.info(
            f"Full-val: frame_f1={full_metrics['frame_f1']:.4f}, "
            f"onset_f1={full_metrics['onset_f1']:.4f}, "
            f"bce={full_metrics['bce']:.4f}, cosine={full_metrics['cosine']:.4f}"
        )

    # ── Phase: generate ────────────────────────────────────────────────────
    elif args.phase == 'generate':
        val_dataset = MaestroSegmentDataset(
            maestro_dir=str(maestro_dir), split='validation',
            segment_len=SEGMENT_LEN,
        )

        best_ckpt_path = output_dir / 'checkpoints' / 'best_f1.pt'
        if not best_ckpt_path.exists():
            logger.error(f"No checkpoint at {best_ckpt_path}")
            sys.exit(1)

        ckpt = torch.load(str(best_ckpt_path), map_location=device)
        decoder = PostHocPRDecoder().to(device)
        decoder.load_state_dict(ckpt['decoder_state_dict'])

        generate_samples(
            model=model,
            decoder=decoder,
            val_dataset=val_dataset,
            device=device,
            output_dir=output_dir,
            descriptor=descriptor,
            pool_to=args.pool_to,
        )


if __name__ == '__main__':
    main()
