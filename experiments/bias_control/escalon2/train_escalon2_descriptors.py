#!/usr/bin/env python3
"""
S2-P2-main: Descriptor-Augmented Cross-Modal Training (Speech <-> EGG)

Extends D0 (train_escalon2.py) with vocal descriptor injection.
Tests 3 hypotheses via separate descriptor arms:
  - V4-lin: Temporal F0 dynamics, LINEAR ratios (Hypothesis A: natural)
  - H-series: Intra-frame harmonic structure (Hypothesis B: strong Phideus)
  - A4-16k: Spectral band energy deltas (Hypothesis C: non-ratio control)

Architecture:
  Speech -> SpeechEGGEncoderAug(d=512, desc_dim=D) -> ProjectionHead(512->256) -> z_speech
  EGG    -> SpeechEGGEncoderAug(d=512, desc_dim=D) -> ProjectionHead(512->256) -> z_egg
  Loss   = VICReg(z_speech, z_egg)

Near-identity init: W=[I|0] in injection layer -> ep0 identical to D0.
Descriptors computed per-modality (no cross-modal leakage).
H-series normalization stats frozen from train set.

Usage:
  # Full 30ep run with V4-lin
  python experiments/bias_control/escalon2/train_escalon2_descriptors.py \
      --lombard-dir data/lombard/FLombard \
      --segment-index data/lombard/segment_index.json \
      --f0-cache data/lombard/f0_cache_noise0.npz \
      --output data/lombard/v4lin_seed42 \
      --descriptor v4_lin --epochs 30

  # Smoke test (3ep x 50 batches)
  python experiments/bias_control/escalon2/train_escalon2_descriptors.py \
      --lombard-dir data/lombard/FLombard \
      --segment-index data/lombard/segment_index.json \
      --f0-cache data/lombard/f0_cache_noise0.npz \
      --output data/lombard/smoke_v4lin \
      --descriptor v4_lin --epochs 3 --max-batches 50
"""

import argparse
import json
import logging
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.bias_control.encoders.speech_egg_encoder_aug import SpeechEGGEncoderAug
from src.bias_control.encoders.projection import ProjectionHead
from src.RNA.vicreg import VICRegLoss
from src.bias_control.training.preflight import DriftSentinel
from src.bias_control.datasets.lombard_segments_aug import (
    LombardSegmentDatasetAug,
    collate_lombard_aug,
    create_lombard_dataloaders_aug,
)
from src.bias_control.vocal_descriptors import (
    compute_v4_linear, compute_v4_log, compute_h_series, compute_a4_16k,
    compute_a10a, compute_a10b, compute_a10c, compute_a10d, compute_a10e,
    get_descriptor_dim, load_h_series_norm_stats,
)
from experiments.bias_control.escalon2.eval_escalon2 import (
    extract_embeddings_lombard,
    evaluate_retrieval_lombard,
    grouped_bootstrap_ci,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ── Utilities ────────────────────────────────────────────────────────────────

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class LinearWarmupCosineScheduler:
    """Linear warmup then cosine annealing."""

    def __init__(self, optimizer, warmup_steps: int, total_steps: int):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.base_lrs = [pg['lr'] for pg in optimizer.param_groups]
        self.step_count = 0

    def step(self):
        self.step_count += 1
        if self.step_count <= self.warmup_steps:
            scale = self.step_count / max(1, self.warmup_steps)
        else:
            progress = (self.step_count - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps
            )
            scale = 0.5 * (1 + math.cos(math.pi * progress))
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg['lr'] = base_lr * scale

    def state_dict(self):
        return {'step_count': self.step_count, 'base_lrs': self.base_lrs}

    def load_state_dict(self, state_dict):
        self.step_count = state_dict['step_count']
        self.base_lrs = state_dict['base_lrs']


# ── Descriptor computation ───────────────────────────────────────────────────

class DescriptorComputer:
    """Computes descriptors for a batch, dispatching by type.

    Encapsulates descriptor logic so it can be passed to extract_embeddings_lombard
    as a callable: descriptor_fn(batch, modality, device) -> [B, T_cnn, D].
    """

    def __init__(self, descriptor_type: str, h_series_norm_stats=None):
        self.descriptor_type = descriptor_type
        self.h_series_norm_stats = h_series_norm_stats
        # T_cnn for SpeechEGGEncoder with 32000-sample input = 800
        self.t_cnn = 800

    def __call__(self, batch, modality: str, device: str) -> torch.Tensor:
        """Compute descriptor for a batch.

        Args:
            batch: dict from LombardSegmentDatasetAug collate
            modality: 'speech' or 'egg'
            device: torch device

        Returns:
            [B, T_cnn, D] descriptor tensor
        """
        waveform = batch[modality].to(device)
        f0 = batch[f'f0_{modality}'].to(device)
        voiced = batch[f'voiced_{modality}'].to(device)

        if self.descriptor_type == 'v4_lin':
            return compute_v4_linear(f0, voiced, target_length=self.t_cnn)

        elif self.descriptor_type == 'v4_log':
            return compute_v4_log(f0, voiced, target_length=self.t_cnn)

        elif self.descriptor_type == 'h_series':
            norm_stats = None
            if self.h_series_norm_stats and modality in self.h_series_norm_stats:
                norm_stats = self.h_series_norm_stats[modality]
            return compute_h_series(
                waveform, f0, voiced,
                target_length=self.t_cnn,
                norm_stats=norm_stats,
            )

        elif self.descriptor_type == 'a4_16k':
            return compute_a4_16k(waveform, target_length=self.t_cnn)

        elif self.descriptor_type == 'a10a':
            return compute_a10a(waveform, target_length=self.t_cnn)

        elif self.descriptor_type == 'a10b':
            return compute_a10b(waveform, target_length=self.t_cnn)

        elif self.descriptor_type == 'a10c':
            return compute_a10c(waveform, target_length=self.t_cnn)

        elif self.descriptor_type == 'a10d':
            return compute_a10d(waveform, target_length=self.t_cnn)

        elif self.descriptor_type == 'a10e':
            return compute_a10e(waveform, target_length=self.t_cnn)

        else:
            raise ValueError(f"Unknown descriptor: {self.descriptor_type}")


# ── H-series normalization stats precomputation ──────────────────────────────

@torch.no_grad()
def precompute_h_series_stats(train_loader, device, max_batches=100):
    """Compute mean/std of H-series features over train set, per-modality.

    These stats are frozen and used for all subsequent training and eval.
    Processes up to max_batches to get stable estimates.

    Returns:
        dict: {'speech': {'mean': [8], 'std': [8]}, 'egg': {'mean': ...}}
    """
    logger.info("Precomputing H-series normalization stats (frozen)...")
    stats = {}

    for modality in ['speech', 'egg']:
        all_descs = []
        n_batches = 0
        for batch in tqdm(train_loader, desc=f"H-series stats ({modality})", leave=False):
            if n_batches >= max_batches:
                break

            waveform = batch[modality].to(device)
            f0 = batch[f'f0_{modality}'].to(device)
            voiced = batch[f'voiced_{modality}'].to(device)

            desc = compute_h_series(
                waveform, f0, voiced,
                target_length=800,
                norm_stats=None,  # raw, unnormalized
            )  # [B, 800, 8]

            # Only include voiced frames for stats
            v_interp = voiced.float().unsqueeze(1)
            v_interp = torch.nn.functional.interpolate(
                v_interp, size=800, mode='linear', align_corners=False
            ).squeeze(1)  # [B, 800]
            v_mask = v_interp > 0.5  # [B, 800]

            for b in range(desc.size(0)):
                voiced_frames = desc[b][v_mask[b]]  # [N_voiced, 8]
                if voiced_frames.size(0) > 10:
                    all_descs.append(voiced_frames.cpu())

            n_batches += 1

        if all_descs:
            all_cat = torch.cat(all_descs, dim=0)  # [N_total, 8]
            mean = all_cat.mean(dim=0)
            std = all_cat.std(dim=0).clamp(min=1e-6)
            stats[modality] = {
                'mean': mean,
                'std': std,
            }
            logger.info(f"  {modality}: {all_cat.size(0)} voiced frames, "
                        f"mean={mean.tolist()}, std={std.tolist()}")
        else:
            logger.warning(f"  {modality}: no voiced frames found, using identity stats")
            stats[modality] = {
                'mean': torch.zeros(8),
                'std': torch.ones(8),
            }

    return stats


def save_h_series_stats(stats, output_dir):
    """Save H-series normalization stats to JSON files."""
    for modality in ['speech', 'egg']:
        path = os.path.join(output_dir, f'h_series_norm_{modality}.json')
        data = {
            'mean': stats[modality]['mean'].tolist(),
            'std': stats[modality]['std'].tolist(),
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"  Saved H-series stats: {path}")


# ── Training ─────────────────────────────────────────────────────────────────

def train_one_epoch(speech_enc, egg_enc, proj_speech, proj_egg,
                    vicreg_loss, optimizer, scheduler, train_loader,
                    device, descriptor_computer, max_batches=None):
    """Train one epoch with descriptor injection."""
    speech_enc.train()
    egg_enc.train()
    proj_speech.train()
    proj_egg.train()

    total_loss = 0
    total_inv = 0
    total_var = 0
    total_cov = 0
    n_batches = 0

    for batch in tqdm(train_loader, desc="Training", leave=False):
        if max_batches and n_batches >= max_batches:
            break

        speech = batch['speech'].to(device)
        egg = batch['egg'].to(device)

        # Compute descriptors per-modality
        desc_speech = descriptor_computer(batch, 'speech', device)
        desc_egg = descriptor_computer(batch, 'egg', device)

        # Forward with descriptor injection
        z_speech = proj_speech(speech_enc(speech, descriptor=desc_speech))
        z_egg = proj_egg(egg_enc(egg, descriptor=desc_egg))

        # VICReg loss
        loss_dict = vicreg_loss(z_speech, z_egg)
        loss = loss_dict['total']

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(speech_enc.parameters()) + list(egg_enc.parameters()) +
            list(proj_speech.parameters()) + list(proj_egg.parameters()),
            max_norm=1.0,
        )
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        total_inv += loss_dict['invariance'].item()
        total_var += loss_dict['variance'].item()
        total_cov += loss_dict['covariance'].item()
        n_batches += 1

    n = max(n_batches, 1)
    return {
        'loss': total_loss / n,
        'invariance': total_inv / n,
        'variance': total_var / n,
        'covariance': total_cov / n,
        'n_batches': n_batches,
    }


@torch.no_grad()
def quick_val(speech_enc, egg_enc, proj_speech, proj_egg,
              vicreg_loss, val_loader, device, descriptor_computer,
              max_batches=50):
    """Quick validation loss with descriptors."""
    speech_enc.eval()
    egg_enc.eval()
    proj_speech.eval()
    proj_egg.eval()

    total_loss = 0
    n = 0

    for batch in val_loader:
        if n >= max_batches:
            break

        speech = batch['speech'].to(device)
        egg = batch['egg'].to(device)

        desc_speech = descriptor_computer(batch, 'speech', device)
        desc_egg = descriptor_computer(batch, 'egg', device)

        z_speech = proj_speech(speech_enc(speech, descriptor=desc_speech))
        z_egg = proj_egg(egg_enc(egg, descriptor=desc_egg))

        loss_dict = vicreg_loss(z_speech, z_egg)
        total_loss += loss_dict['total'].item()
        n += 1

    return total_loss / max(n, 1)


def run_structured_eval(speech_enc, egg_enc, proj_speech, proj_egg,
                        test_loader, test_segments, device,
                        descriptor_computer,
                        pool_size=128, n_queries=500, seed=42):
    """Run full structured retrieval evaluation with descriptors."""
    speech_embs, egg_embs = extract_embeddings_lombard(
        speech_enc, egg_enc, proj_speech, proj_egg,
        test_loader, device=device,
        descriptor_fn=descriptor_computer,
    )

    results = evaluate_retrieval_lombard(
        speech_embs, egg_embs, test_segments,
        pool_size=pool_size, n_queries=n_queries, seed=seed,
    )

    ci = grouped_bootstrap_ci(results['per_query'])
    results['ci'] = ci

    return results


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='S2-P2-main: Descriptor Training')
    parser.add_argument('--lombard-dir', type=str, required=True)
    parser.add_argument('--segment-index', type=str, required=True)
    parser.add_argument('--f0-cache', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--descriptor', type=str, required=True,
                        choices=['v4_lin', 'v4_log', 'h_series', 'a4_16k', 'a10a', 'a10b', 'a10c', 'a10d', 'a10e'],
                        help='Descriptor type to inject')
    parser.add_argument('--condition', type=str, default='noise0')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--lr-enc', type=float, default=5e-4)
    parser.add_argument('--lr-proj', type=float, default=1e-3)
    parser.add_argument('--warmup-steps', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--max-batches', type=int, default=None,
                        help='Max batches per epoch (for smoke test)')
    parser.add_argument('--structured-eval-epochs', type=int, nargs='+',
                        default=None)
    parser.add_argument('--pool-size', type=int, default=128)
    parser.add_argument('--n-queries', type=int, default=500)
    parser.add_argument('--h-series-stats-batches', type=int, default=100,
                        help='Batches for H-series stat precomputation')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    seed_everything(args.seed)
    device = torch.device(args.device)

    if args.structured_eval_epochs is None:
        args.structured_eval_epochs = [5, 10, 15, 20, 25, 28, 29, 30]

    desc_dim = get_descriptor_dim(args.descriptor)

    logger.info("=" * 60)
    logger.info(f"S2-P2-MAIN: {args.descriptor.upper()} (dim={desc_dim})")
    logger.info("=" * 60)

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader = create_lombard_dataloaders_aug(
        lombard_dir=args.lombard_dir,
        segment_index_path=args.segment_index,
        f0_cache_path=args.f0_cache,
        noise_condition=args.condition,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    with open(args.segment_index) as f:
        si = json.load(f)
    test_segments = [
        s for s in si['segments']
        if s['split'] == 'test' and s['noise_condition'] == args.condition
    ]

    n_train = len(train_loader.dataset)
    n_val = len(val_loader.dataset)
    n_test = len(test_loader.dataset)
    batches_per_epoch = len(train_loader)
    if args.max_batches:
        batches_per_epoch = min(batches_per_epoch, args.max_batches)

    logger.info(f"  Train: {n_train} segments ({batches_per_epoch} batches/ep)")
    logger.info(f"  Val: {n_val}, Test: {n_test} segments")
    logger.info(f"  Descriptor: {args.descriptor} (dim={desc_dim})")

    # ── H-series stats (if needed) ────────────────────────────────────────────
    h_series_norm_stats = None
    if args.descriptor == 'h_series':
        h_series_norm_stats = precompute_h_series_stats(
            train_loader, device,
            max_batches=args.h_series_stats_batches,
        )
        save_h_series_stats(h_series_norm_stats, args.output)

    # ── Descriptor computer ───────────────────────────────────────────────────
    descriptor_computer = DescriptorComputer(
        descriptor_type=args.descriptor,
        h_series_norm_stats=h_series_norm_stats,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    speech_enc = SpeechEGGEncoderAug(
        descriptor_dim=desc_dim, output_dim=512, n_layers=4, n_heads=8
    ).to(device)
    egg_enc = SpeechEGGEncoderAug(
        descriptor_dim=desc_dim, output_dim=512, n_layers=4, n_heads=8
    ).to(device)
    proj_speech = ProjectionHead(input_dim=512, hidden_dim=512, output_dim=256).to(device)
    proj_egg = ProjectionHead(input_dim=512, hidden_dim=512, output_dim=256).to(device)
    vicreg_loss = VICRegLoss(lambda_inv=10.0, lambda_var=10.0, lambda_cov=1.0)

    total_params = sum(p.numel() for p in list(speech_enc.parameters()) +
                       list(egg_enc.parameters()) + list(proj_speech.parameters()) +
                       list(proj_egg.parameters()))

    logger.info(f"  Speech encoder: {sum(p.numel() for p in speech_enc.parameters()):,} params")
    logger.info(f"  EGG encoder: {sum(p.numel() for p in egg_enc.parameters()):,} params")
    logger.info(f"  Total: {total_params:,} params (all trainable)")

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer = AdamW([
        {'params': speech_enc.parameters(), 'lr': args.lr_enc},
        {'params': egg_enc.parameters(), 'lr': args.lr_enc},
        {'params': proj_speech.parameters(), 'lr': args.lr_proj},
        {'params': proj_egg.parameters(), 'lr': args.lr_proj},
    ], weight_decay=0.01)

    total_steps = batches_per_epoch * args.epochs
    scheduler = LinearWarmupCosineScheduler(optimizer, args.warmup_steps, total_steps)

    logger.info(f"  LR: enc={args.lr_enc}, proj={args.lr_proj}")
    logger.info(f"  Warmup: {args.warmup_steps} steps, total: {total_steps} steps")

    # ── DriftSentinel ─────────────────────────────────────────────────────────
    all_modules = nn.ModuleDict({
        'speech_enc': speech_enc,
        'egg_enc': egg_enc,
        'proj_speech': proj_speech,
        'proj_egg': proj_egg,
    })
    sentinel = DriftSentinel(all_modules)

    # ── Save config ───────────────────────────────────────────────────────────
    config = {
        'mode': 'train_descriptors',
        'descriptor': args.descriptor,
        'descriptor_dim': desc_dim,
        'condition': args.condition,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr_enc': args.lr_enc,
        'lr_proj': args.lr_proj,
        'warmup_steps': args.warmup_steps,
        'seed': args.seed,
        'total_params': total_params,
        'n_train': n_train,
        'n_val': n_val,
        'n_test': n_test,
        'batches_per_epoch': batches_per_epoch,
        'max_batches': args.max_batches,
        'structured_eval_epochs': args.structured_eval_epochs,
        'encoder': f'SpeechEGGEncoderAug(d=512, desc_dim={desc_dim})',
        'projection': 'ProjectionHead(512->512->256)',
        'loss': 'VICReg(inv=10, var=10, cov=1)',
        'f0_cache': args.f0_cache,
        'h_series_norm_stats': 'per_modality' if args.descriptor == 'h_series' else None,
        'injection': f'Linear(512+{desc_dim}, 512), W=[I|0], bias=0',
        'd0_baseline_S': 0.778,
    }
    with open(os.path.join(args.output, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_S = 0.0
    best_epoch = 0
    history = []
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        t_ep = time.time()

        train_metrics = train_one_epoch(
            speech_enc, egg_enc, proj_speech, proj_egg,
            vicreg_loss, optimizer, scheduler, train_loader,
            device, descriptor_computer, max_batches=args.max_batches,
        )

        val_loss = quick_val(
            speech_enc, egg_enc, proj_speech, proj_egg,
            vicreg_loss, val_loader, device, descriptor_computer,
        )

        # DriftSentinel after epoch 1
        if epoch == 1:
            drift = sentinel.check(all_modules)
            n_drifted = sum(1 for d in drift.values() if d > 0)
            logger.info(f"  DriftSentinel: {n_drifted}/{len(drift)} params drifted")
            if n_drifted == 0:
                logger.error("  GHOST TRAINING DETECTED — no parameters moved!")
                return

        # Structured eval at canonical epochs
        eval_results = None
        if epoch in args.structured_eval_epochs:
            logger.info(f"  [eval] Running structured retrieval (ep {epoch})...")
            eval_results = run_structured_eval(
                speech_enc, egg_enc, proj_speech, proj_egg,
                test_loader, test_segments, device,
                descriptor_computer,
                pool_size=args.pool_size, n_queries=args.n_queries,
                seed=args.seed,
            )
            S = eval_results['S']
            logger.info(
                f"  [eval] S2E@10={eval_results['S2E_at_k']:.1%} "
                f"E2S@10={eval_results['E2S_at_k']:.1%} "
                f"S={S:.1%} "
                f"CI=[{eval_results['ci']['S_ci_lo']:.1%}, "
                f"{eval_results['ci']['S_ci_hi']:.1%}]"
            )

            if S > best_S:
                best_S = S
                best_epoch = epoch
                torch.save({
                    'epoch': epoch,
                    'speech_enc': speech_enc.state_dict(),
                    'egg_enc': egg_enc.state_dict(),
                    'proj_speech': proj_speech.state_dict(),
                    'proj_egg': proj_egg.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'S': S,
                    'descriptor': args.descriptor,
                    'descriptor_dim': desc_dim,
                    'eval_results': {k: v for k, v in eval_results.items()
                                     if k != 'per_query'},
                }, os.path.join(args.output, 'best_model.pt'))
                logger.info(f"  >>> New best: S={S:.1%} @ epoch {epoch}")

        # Checkpoint every epoch
        torch.save({
            'epoch': epoch,
            'speech_enc': speech_enc.state_dict(),
            'egg_enc': egg_enc.state_dict(),
            'proj_speech': proj_speech.state_dict(),
            'proj_egg': proj_egg.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'descriptor': args.descriptor,
            'descriptor_dim': desc_dim,
        }, os.path.join(args.output, f'checkpoint_ep{epoch:03d}.pt'))

        ep_time = time.time() - t_ep
        log_msg = (
            f"Epoch {epoch}/{args.epochs} ({ep_time:.1f}s): "
            f"loss={train_metrics['loss']:.4f} "
            f"val={val_loss:.4f} "
            f"inv={train_metrics['invariance']:.3f} "
            f"var={train_metrics['variance']:.3f} "
            f"cov={train_metrics['covariance']:.3f}"
        )
        if eval_results:
            log_msg += f" | S={eval_results['S']:.1%}"
        logger.info(log_msg)

        entry = {
            'epoch': epoch,
            'train_loss': train_metrics['loss'],
            'val_loss': val_loss,
            'invariance': train_metrics['invariance'],
            'variance': train_metrics['variance'],
            'covariance': train_metrics['covariance'],
            'time_sec': ep_time,
        }
        if eval_results:
            entry['S2E'] = eval_results['S2E_at_k']
            entry['E2S'] = eval_results['E2S_at_k']
            entry['S'] = eval_results['S']
            entry['ci_lo'] = eval_results['ci']['S_ci_lo']
            entry['ci_hi'] = eval_results['ci']['S_ci_hi']
        history.append(entry)

    total_time = time.time() - t_start
    logger.info("=" * 60)
    logger.info(f"TRAINING COMPLETE: {total_time/60:.1f} minutes")
    logger.info(f"  Descriptor: {args.descriptor} (dim={desc_dim})")
    logger.info(f"  Best S = {best_S:.1%} @ epoch {best_epoch}")
    logger.info(f"  D0 baseline: S=77.8%")
    logger.info(f"  Delta vs D0: {(best_S - 0.778)*100:+.1f}pp")
    logger.info("=" * 60)

    with open(os.path.join(args.output, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    summary = {
        'descriptor': args.descriptor,
        'descriptor_dim': desc_dim,
        'best_S': best_S,
        'best_epoch': best_epoch,
        'total_time_min': total_time / 60,
        'total_params': total_params,
        'n_epochs': args.epochs,
        'condition': args.condition,
        'd0_baseline_S': 0.778,
        'delta_vs_d0_pp': round((best_S - 0.778) * 100, 1),
    }
    with open(os.path.join(args.output, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)


if __name__ == '__main__':
    main()
