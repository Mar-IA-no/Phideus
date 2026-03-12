#!/usr/bin/env python3
"""
S2-P2.5b: Conditioned Projection (FiLM) Training for Speech <-> EGG

Encoder is standard D0 (SpeechEGGEncoder, no descriptor injection).
Descriptor conditions the projection head via FiLM:
    encoder(waveform) -> [B, 512]
    desc = descriptor_computer(batch, modality) -> [B, T_cnn, D]
    cond = desc.mean(dim=1) -> [B, D]
    z = ConditionedProjectionHead(features, cond=cond) -> [B, 256]

This is the lightest possible injection: the encoder learns the same
representation as D0, and only the projection to shared space is
descriptor-aware. In Escalon 1, this mechanism (Gate 8 pca) gave the
best audio-only result: 82.6% vs ctrl 79.2%.

FiLM: h' = (1 + gamma) * h + beta, zero-init -> identity at init.
At epoch 0, this is exactly D0.

Usage:
  # V4-lin + proj_cond (30ep)
  python experiments/bias_control/escalon2/train_escalon2_pca.py \\
      --lombard-dir data/lombard/FLombard \\
      --segment-index data/lombard/segment_index.json \\
      --f0-cache data/lombard/f0_cache_noise0.npz \\
      --output data/lombard/v4lin_pca_seed42 \\
      --descriptor v4_lin --epochs 30

  # H-series + proj_cond (30ep)
  python experiments/bias_control/escalon2/train_escalon2_pca.py \\
      --lombard-dir data/lombard/FLombard \\
      --segment-index data/lombard/segment_index.json \\
      --f0-cache data/lombard/f0_cache_noise0.npz \\
      --output data/lombard/hseries_pca_seed42 \\
      --descriptor h_series --epochs 30

  # A4-16k + proj_cond (30ep)
  python experiments/bias_control/escalon2/train_escalon2_pca.py \\
      --lombard-dir data/lombard/FLombard \\
      --segment-index data/lombard/segment_index.json \\
      --f0-cache data/lombard/f0_cache_noise0.npz \\
      --output data/lombard/a4_16k_pca_seed42 \\
      --descriptor a4_16k --epochs 30
"""

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.bias_control.encoders.speech_egg_encoder import SpeechEGGEncoder
from src.bias_control.encoders.projection import ConditionedProjectionHead
from src.RNA.vicreg import VICRegLoss
from src.bias_control.training.preflight import DriftSentinel
from src.bias_control.datasets.lombard_segments_aug import create_lombard_dataloaders_aug
from src.bias_control.vocal_descriptors import get_descriptor_dim

from experiments.bias_control.escalon2.train_escalon2_descriptors import (
    DescriptorComputer,
    precompute_h_series_stats,
    save_h_series_stats,
    LinearWarmupCosineScheduler,
    seed_everything,
    seed_worker,
)

from experiments.bias_control.escalon2.eval_escalon2 import (
    evaluate_retrieval_lombard,
    grouped_bootstrap_ci,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ── Embedding extraction (proj_cond variant) ────────────────────────────────

@torch.no_grad()
def extract_embeddings_pca(speech_enc, egg_enc, proj_speech, proj_egg,
                           dataloader, device, descriptor_computer):
    """Extract embeddings for proj_cond: descriptor goes to projection, not encoder.

    Args:
        proj_speech/proj_egg: ConditionedProjectionHead instances
        descriptor_computer: DescriptorComputer callable

    Returns:
        speech_embs, egg_embs: dicts (clip_id, seg_idx) -> np.array [D]
    """
    speech_enc.eval()
    egg_enc.eval()
    proj_speech.eval()
    proj_egg.eval()

    speech_embs = {}
    egg_embs = {}

    for batch in tqdm(dataloader, desc="Extracting embeddings", leave=False):
        speech_wav = batch['speech'].to(device)
        egg_wav = batch['egg'].to(device)

        # Encoder: standard D0 forward (no descriptor)
        feat_speech = speech_enc(speech_wav)  # [B, 512]
        feat_egg = egg_enc(egg_wav)  # [B, 512]

        # Descriptor -> temporal mean -> conditioning vector
        desc_speech = descriptor_computer(batch, 'speech', device)  # [B, T_cnn, D]
        desc_egg = descriptor_computer(batch, 'egg', device)
        cond_speech = desc_speech.mean(dim=1)  # [B, D]
        cond_egg = desc_egg.mean(dim=1)

        # Conditioned projection
        z_speech = proj_speech(feat_speech, cond=cond_speech)  # [B, 256]
        z_egg = proj_egg(feat_egg, cond=cond_egg)

        z_speech = z_speech.cpu().numpy()
        z_egg = z_egg.cpu().numpy()

        for i in range(len(batch['clip_id'])):
            key = (batch['clip_id'][i], batch['segment_idx'][i])
            speech_embs[key] = z_speech[i]
            egg_embs[key] = z_egg[i]

    return speech_embs, egg_embs


# ── Usage metrics ────────────────────────────────────────────────────────────

def collect_usage_metrics(proj_speech, proj_egg):
    """Collect FiLM parameter metrics."""
    metrics = {}
    for name, proj in [('speech', proj_speech), ('egg', proj_egg)]:
        for i, gen in enumerate(proj.film_generators):
            last_layer = gen[-1]
            metrics[f'{name}_film{i}_weight_norm'] = last_layer.weight.data.norm().item()
            metrics[f'{name}_film{i}_bias_norm'] = last_layer.bias.data.norm().item()
    return metrics


# ── Training ─────────────────────────────────────────────────────────────────

def train_one_epoch(speech_enc, egg_enc, proj_speech, proj_egg,
                    vicreg_loss, optimizer, scheduler, train_loader,
                    device, descriptor_computer, max_batches=None):
    """Train one epoch with FiLM-conditioned projection."""
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

        # Standard encoder forward (no descriptor)
        feat_speech = speech_enc(speech)  # [B, 512]
        feat_egg = egg_enc(egg)

        # Descriptor -> conditioning
        desc_speech = descriptor_computer(batch, 'speech', device)
        desc_egg = descriptor_computer(batch, 'egg', device)
        cond_speech = desc_speech.mean(dim=1)  # [B, D]
        cond_egg = desc_egg.mean(dim=1)

        # FiLM-conditioned projection
        z_speech = proj_speech(feat_speech, cond=cond_speech)
        z_egg = proj_egg(feat_egg, cond=cond_egg)

        # VICReg loss
        loss_dict = vicreg_loss(z_speech, z_egg)
        loss = loss_dict['total']

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
    """Quick validation loss."""
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

        feat_speech = speech_enc(speech)
        feat_egg = egg_enc(egg)

        desc_speech = descriptor_computer(batch, 'speech', device)
        desc_egg = descriptor_computer(batch, 'egg', device)
        cond_speech = desc_speech.mean(dim=1)
        cond_egg = desc_egg.mean(dim=1)

        z_speech = proj_speech(feat_speech, cond=cond_speech)
        z_egg = proj_egg(feat_egg, cond=cond_egg)

        loss_dict = vicreg_loss(z_speech, z_egg)
        total_loss += loss_dict['total'].item()
        n += 1

    return total_loss / max(n, 1)


def run_structured_eval(speech_enc, egg_enc, proj_speech, proj_egg,
                        test_loader, test_segments, device,
                        descriptor_computer,
                        pool_size=128, n_queries=500, seed=42):
    """Run full structured retrieval evaluation with proj_cond."""
    speech_embs, egg_embs = extract_embeddings_pca(
        speech_enc, egg_enc, proj_speech, proj_egg,
        test_loader, device, descriptor_computer,
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
    parser = argparse.ArgumentParser(
        description='S2-P2.5b: Conditioned Projection (FiLM) Training'
    )
    parser.add_argument('--lombard-dir', type=str, required=True)
    parser.add_argument('--segment-index', type=str, required=True)
    parser.add_argument('--f0-cache', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--descriptor', type=str, required=True,
                        choices=['v4_lin', 'v4_log', 'h_series', 'a4_16k'])
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
    parser.add_argument('--h-series-stats-batches', type=int, default=100)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    seed_everything(args.seed)
    device = torch.device(args.device)

    if args.structured_eval_epochs is None:
        args.structured_eval_epochs = [5, 10, 15, 20, 25, 28, 29, 30]

    desc_dim = get_descriptor_dim(args.descriptor)

    logger.info("=" * 60)
    logger.info(f"S2-P2.5b: PROJ_COND (FiLM) + {args.descriptor.upper()} "
                f"(dim={desc_dim})")
    logger.info("=" * 60)

    # ── Data ──────────────────────────────────────────────────────────────
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
    logger.info(f"  Injection: proj_cond (FiLM on projection)")
    logger.info(f"  Descriptor: {args.descriptor} (dim={desc_dim})")

    # ── H-series stats (if needed) ────────────────────────────────────────
    h_series_norm_stats = None
    if args.descriptor == 'h_series':
        h_series_norm_stats = precompute_h_series_stats(
            train_loader, device,
            max_batches=args.h_series_stats_batches,
        )
        save_h_series_stats(h_series_norm_stats, args.output)

    # ── Descriptor computer ───────────────────────────────────────────────
    descriptor_computer = DescriptorComputer(
        descriptor_type=args.descriptor,
        h_series_norm_stats=h_series_norm_stats,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    # Standard D0 encoders — no descriptor injection
    speech_enc = SpeechEGGEncoder(output_dim=512, n_layers=4, n_heads=8).to(device)
    egg_enc = SpeechEGGEncoder(output_dim=512, n_layers=4, n_heads=8).to(device)

    # FiLM-conditioned projection heads
    proj_speech = ConditionedProjectionHead(
        input_dim=512, hidden_dim=512, output_dim=256, cond_dim=desc_dim,
    ).to(device)
    proj_egg = ConditionedProjectionHead(
        input_dim=512, hidden_dim=512, output_dim=256, cond_dim=desc_dim,
    ).to(device)

    vicreg_loss = VICRegLoss(lambda_inv=10.0, lambda_var=10.0, lambda_cov=1.0)

    total_params = sum(
        p.numel() for p in
        list(speech_enc.parameters()) + list(egg_enc.parameters()) +
        list(proj_speech.parameters()) + list(proj_egg.parameters())
    )

    # FiLM-specific param count
    film_params = sum(p.numel() for p in proj_speech.film_generators.parameters())
    film_params *= 2  # both projections

    enc_params = sum(p.numel() for p in speech_enc.parameters())
    proj_params = sum(p.numel() for p in proj_speech.parameters())

    logger.info(f"  Speech encoder: {enc_params:,} params (standard D0)")
    logger.info(f"  EGG encoder: {enc_params:,} params (standard D0)")
    logger.info(f"  Each projection: {proj_params:,} params (ConditionedProjectionHead)")
    logger.info(f"  FiLM params (both proj): {film_params:,}")
    logger.info(f"  Total: {total_params:,} params (all trainable)")

    # ── Optimizer ─────────────────────────────────────────────────────────
    optimizer = AdamW([
        {'params': speech_enc.parameters(), 'lr': args.lr_enc},
        {'params': egg_enc.parameters(), 'lr': args.lr_enc},
        {'params': proj_speech.parameters(), 'lr': args.lr_proj},
        {'params': proj_egg.parameters(), 'lr': args.lr_proj},
    ], weight_decay=0.01)

    total_steps = batches_per_epoch * args.epochs
    scheduler = LinearWarmupCosineScheduler(
        optimizer, args.warmup_steps, total_steps
    )

    logger.info(f"  LR: enc={args.lr_enc}, proj={args.lr_proj}")
    logger.info(f"  Warmup: {args.warmup_steps} steps, total: {total_steps} steps")

    # ── DriftSentinel ─────────────────────────────────────────────────────
    all_modules = nn.ModuleDict({
        'speech_enc': speech_enc,
        'egg_enc': egg_enc,
        'proj_speech': proj_speech,
        'proj_egg': proj_egg,
    })
    sentinel = DriftSentinel(all_modules)

    # ── Save config ───────────────────────────────────────────────────────
    config = {
        'mode': 'train_proj_cond',
        'injection': 'pca',
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
        'film_params': film_params,
        'n_train': n_train,
        'n_val': n_val,
        'n_test': n_test,
        'batches_per_epoch': batches_per_epoch,
        'max_batches': args.max_batches,
        'structured_eval_epochs': args.structured_eval_epochs,
        'encoder': 'SpeechEGGEncoder(d=512) — standard D0, no descriptor',
        'projection': f'ConditionedProjectionHead(512->512->256, cond_dim={desc_dim})',
        'loss': 'VICReg(inv=10, var=10, cov=1)',
        'f0_cache': args.f0_cache,
        'h_series_norm_stats': 'per_modality' if args.descriptor == 'h_series' else None,
        'd0_baseline_S': 0.778,
    }
    with open(os.path.join(args.output, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    # ── Training loop ─────────────────────────────────────────────────────
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

        # FiLM usage metrics
        usage_metrics = collect_usage_metrics(proj_speech, proj_egg)

        # DriftSentinel after epoch 1
        if epoch == 1:
            drift = sentinel.check(all_modules)
            n_drifted = sum(1 for d in drift.values() if d > 0)
            logger.info(f"  DriftSentinel: {n_drifted}/{len(drift)} params drifted")
            if n_drifted == 0:
                logger.error("  GHOST TRAINING DETECTED — no parameters moved!")
                return

            logger.info("  FiLM metrics (ep1):")
            for k, v in usage_metrics.items():
                logger.info(f"    {k}: {v:.6f}")

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
                    'injection': 'pca',
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
            'injection': 'pca',
            'descriptor': args.descriptor,
            'descriptor_dim': desc_dim,
        }, os.path.join(args.output, f'checkpoint_ep{epoch:03d}.pt'))

        ep_time = time.time() - t_ep

        # Log
        log_msg = (
            f"Epoch {epoch}/{args.epochs} ({ep_time:.1f}s): "
            f"loss={train_metrics['loss']:.4f} "
            f"val={val_loss:.4f} "
            f"inv={train_metrics['invariance']:.3f} "
            f"var={train_metrics['variance']:.3f} "
            f"cov={train_metrics['covariance']:.3f}"
        )

        # Add FiLM norm to log
        fn = usage_metrics.get('speech_film0_weight_norm', 0)
        log_msg += f" | film0_wn={fn:.4f}"

        if eval_results:
            log_msg += f" | S={eval_results['S']:.1%}"

        logger.info(log_msg)

        # History entry
        entry = {
            'epoch': epoch,
            'train_loss': train_metrics['loss'],
            'val_loss': val_loss,
            'invariance': train_metrics['invariance'],
            'variance': train_metrics['variance'],
            'covariance': train_metrics['covariance'],
            'time_sec': ep_time,
        }
        entry.update(usage_metrics)

        if eval_results:
            entry['S2E'] = eval_results['S2E_at_k']
            entry['E2S'] = eval_results['E2S_at_k']
            entry['S'] = eval_results['S']
            entry['ci_lo'] = eval_results['ci']['S_ci_lo']
            entry['ci_hi'] = eval_results['ci']['S_ci_hi']

        history.append(entry)

    total_time = time.time() - t_start

    # ── Summary ───────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info(f"TRAINING COMPLETE: {total_time/60:.1f} minutes")
    logger.info(f"  Injection: proj_cond (FiLM)")
    logger.info(f"  Descriptor: {args.descriptor} (dim={desc_dim})")
    logger.info(f"  Best S = {best_S:.1%} @ epoch {best_epoch}")
    logger.info(f"  D0 baseline: S=77.8%")
    logger.info(f"  Delta vs D0: {(best_S - 0.778)*100:+.1f}pp")
    logger.info("=" * 60)

    # Final FiLM metrics
    final_usage = collect_usage_metrics(proj_speech, proj_egg)
    logger.info("Final FiLM metrics:")
    for k, v in final_usage.items():
        logger.info(f"  {k}: {v:.6f}")

    with open(os.path.join(args.output, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    summary = {
        'injection': 'pca',
        'descriptor': args.descriptor,
        'descriptor_dim': desc_dim,
        'best_S': best_S,
        'best_epoch': best_epoch,
        'total_time_min': total_time / 60,
        'total_params': total_params,
        'film_params': film_params,
        'n_epochs': args.epochs,
        'condition': args.condition,
        'd0_baseline_S': 0.778,
        'delta_vs_d0_pp': round((best_S - 0.778) * 100, 1),
        'final_film_metrics': final_usage,
    }
    with open(os.path.join(args.output, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)


if __name__ == '__main__':
    main()
