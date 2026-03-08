#!/usr/bin/env python3
"""
S2-P2-control: D0 Neural Cross-Modal Training (Speech <-> EGG)

Two identical SpeechEGGEncoder encoders + ProjectionHeads + VICReg.
No descriptors (D0 baseline). Establishes S_control for Escalón 2.

Architecture:
  Speech -> SpeechEGGEncoder(d=512) -> ProjectionHead(512->512->256) -> z_speech
  EGG    -> SpeechEGGEncoder(d=512) -> ProjectionHead(512->512->256) -> z_egg
  Loss   = VICReg(z_speech, z_egg, λ_inv=10, λ_var=10, λ_cov=1)

Protocol (from S2-P0):
  sr=16kHz, segment=2.0s, hop=0.5s
  Pilot: noise0 only
  Epoch = full pass of train segments
  Structured eval at canonical epochs

Usage:
  python experiments/bias_control/escalon2/train_escalon2.py \
      --lombard-dir data/lombard/FLombard \
      --segment-index data/lombard/segment_index.json \
      --output data/lombard/d0_seed42 \
      --epochs 30 --batch-size 64 --seed 42

  # Mini-run (throughput check):
  python experiments/bias_control/escalon2/train_escalon2.py \
      --lombard-dir data/lombard/FLombard \
      --segment-index data/lombard/segment_index.json \
      --output data/lombard/d0_mini \
      --epochs 1 --batch-size 64 --max-batches 20 --seed 42
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

from src.bias_control.encoders.speech_egg_encoder import SpeechEGGEncoder
from src.bias_control.encoders.projection import ProjectionHead
from src.RNA.vicreg import VICRegLoss
from src.bias_control.training.preflight import DriftSentinel
from src.bias_control.datasets.lombard_segments import (
    LombardSegmentDataset,
    collate_lombard,
    create_lombard_dataloaders,
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


# ── Training ─────────────────────────────────────────────────────────────────

def train_one_epoch(speech_enc, egg_enc, proj_speech, proj_egg,
                    vicreg_loss, optimizer, scheduler, train_loader,
                    device, max_batches=None):
    """Train one epoch. Returns avg loss and component losses."""
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

        # Forward
        z_speech = proj_speech(speech_enc(speech))
        z_egg = proj_egg(egg_enc(egg))

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
              vicreg_loss, val_loader, device, max_batches=50):
    """Quick validation loss (not retrieval)."""
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

        z_speech = proj_speech(speech_enc(speech))
        z_egg = proj_egg(egg_enc(egg))

        loss_dict = vicreg_loss(z_speech, z_egg)
        total_loss += loss_dict['total'].item()
        n += 1

    return total_loss / max(n, 1)


def run_structured_eval(speech_enc, egg_enc, proj_speech, proj_egg,
                        test_loader, test_segments, device,
                        pool_size=128, n_queries=500, seed=42):
    """Run full structured retrieval evaluation."""
    speech_embs, egg_embs = extract_embeddings_lombard(
        speech_enc, egg_enc, proj_speech, proj_egg,
        test_loader, device=device,
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
    parser = argparse.ArgumentParser(description='S2-P2: D0 Neural Training')
    parser.add_argument('--lombard-dir', type=str, required=True)
    parser.add_argument('--segment-index', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
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
                        help='Max batches per epoch (for mini-run)')
    parser.add_argument('--structured-eval-epochs', type=int, nargs='+',
                        default=None,
                        help='Epochs for structured eval (default: 5,10,15,20,25,28,29,30)')
    parser.add_argument('--pool-size', type=int, default=128)
    parser.add_argument('--n-queries', type=int, default=500)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    seed_everything(args.seed)
    device = torch.device(args.device)

    if args.structured_eval_epochs is None:
        args.structured_eval_epochs = [5, 10, 15, 20, 25, 28, 29, 30]

    logger.info("=" * 60)
    logger.info("S2-P2-CONTROL: D0 Neural Speech <-> EGG")
    logger.info("=" * 60)

    # ── Data ─────────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader = create_lombard_dataloaders(
        lombard_dir=args.lombard_dir,
        segment_index_path=args.segment_index,
        noise_condition=args.condition,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Load test segments for structured eval
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
    logger.info(f"  Condition: {args.condition}")

    # ── Model ────────────────────────────────────────────────────────────────
    speech_enc = SpeechEGGEncoder(output_dim=512, n_layers=4, n_heads=8).to(device)
    egg_enc = SpeechEGGEncoder(output_dim=512, n_layers=4, n_heads=8).to(device)
    proj_speech = ProjectionHead(input_dim=512, hidden_dim=512, output_dim=256).to(device)
    proj_egg = ProjectionHead(input_dim=512, hidden_dim=512, output_dim=256).to(device)
    vicreg_loss = VICRegLoss(lambda_inv=10.0, lambda_var=10.0, lambda_cov=1.0)

    total_params = sum(p.numel() for p in list(speech_enc.parameters()) +
                       list(egg_enc.parameters()) + list(proj_speech.parameters()) +
                       list(proj_egg.parameters()))
    trainable_params = sum(p.numel() for p in list(speech_enc.parameters()) +
                           list(egg_enc.parameters()) + list(proj_speech.parameters()) +
                           list(proj_egg.parameters()) if p.requires_grad)

    logger.info(f"  Speech encoder: {sum(p.numel() for p in speech_enc.parameters()):,} params")
    logger.info(f"  EGG encoder: {sum(p.numel() for p in egg_enc.parameters()):,} params")
    logger.info(f"  Total: {total_params:,} params (all trainable)")

    # ── Optimizer ────────────────────────────────────────────────────────────
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
    logger.info(f"  Structured eval epochs: {args.structured_eval_epochs}")

    # ── DriftSentinel ────────────────────────────────────────────────────────
    all_modules = nn.ModuleDict({
        'speech_enc': speech_enc,
        'egg_enc': egg_enc,
        'proj_speech': proj_speech,
        'proj_egg': proj_egg,
    })
    sentinel = DriftSentinel(all_modules)

    # ── Save config ──────────────────────────────────────────────────────────
    config = {
        'mode': 'train',
        'condition': args.condition,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr_enc': args.lr_enc,
        'lr_proj': args.lr_proj,
        'warmup_steps': args.warmup_steps,
        'seed': args.seed,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'n_train': n_train,
        'n_val': n_val,
        'n_test': n_test,
        'batches_per_epoch': batches_per_epoch,
        'max_batches': args.max_batches,
        'structured_eval_epochs': args.structured_eval_epochs,
        'encoder': 'SpeechEGGEncoder(d=512, n_layers=4, n_heads=8)',
        'projection': 'ProjectionHead(512->512->256)',
        'loss': 'VICReg(inv=10, var=10, cov=1)',
    }
    with open(os.path.join(args.output, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    # ── Training loop ────────────────────────────────────────────────────────
    best_S = 0.0
    best_epoch = 0
    history = []
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        t_ep = time.time()

        # Train
        train_metrics = train_one_epoch(
            speech_enc, egg_enc, proj_speech, proj_egg,
            vicreg_loss, optimizer, scheduler, train_loader,
            device, max_batches=args.max_batches,
        )

        # Quick val
        val_loss = quick_val(
            speech_enc, egg_enc, proj_speech, proj_egg,
            vicreg_loss, val_loader, device,
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
                # Save best model
                torch.save({
                    'epoch': epoch,
                    'speech_enc': speech_enc.state_dict(),
                    'egg_enc': egg_enc.state_dict(),
                    'proj_speech': proj_speech.state_dict(),
                    'proj_egg': proj_egg.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'S': S,
                    'eval_results': {k: v for k, v in eval_results.items()
                                     if k != 'per_query'},
                }, os.path.join(args.output, 'best_model.pt'))
                logger.info(f"  >>> New best: S={S:.1%} @ epoch {epoch}")

        # Save checkpoint every epoch
        torch.save({
            'epoch': epoch,
            'speech_enc': speech_enc.state_dict(),
            'egg_enc': egg_enc.state_dict(),
            'proj_speech': proj_speech.state_dict(),
            'proj_egg': proj_egg.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
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
    logger.info(f"  Best S = {best_S:.1%} @ epoch {best_epoch}")
    logger.info("=" * 60)

    # Save history
    with open(os.path.join(args.output, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    # Save final summary
    summary = {
        'best_S': best_S,
        'best_epoch': best_epoch,
        'total_time_min': total_time / 60,
        'total_params': total_params,
        'n_epochs': args.epochs,
        'condition': args.condition,
        'cca_baseline_S': 0.644,  # From P1
    }
    with open(os.path.join(args.output, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)


if __name__ == '__main__':
    main()
