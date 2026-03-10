#!/usr/bin/env python3
"""
S2-P2.5: Attention-Based Descriptor Injection Training (Speech <-> EGG)

Two attention-based injection mechanisms matched to descriptor nature:
  - attn_bias: V4-lin descriptor modulates Transformer self-attention via
               factored bilinear bias (asymmetric phi/psi + per-head W)
  - xattn: H-series/A4-16k descriptor queries CNN features via residual
           cross-attention (content-only hypothesis, no pos_emb on K/V)

Shared components imported from train_escalon2_descriptors.py:
  DescriptorComputer, precompute_h_series_stats, save_h_series_stats,
  LinearWarmupCosineScheduler, seed_everything, seed_worker

Eval imported from eval_escalon2.py:
  extract_embeddings_lombard, evaluate_retrieval_lombard, grouped_bootstrap_ci

Architecture:
  Speech -> Encoder(d=512, injection) -> ProjectionHead(512->256) -> z_speech
  EGG    -> Encoder(d=512, injection) -> ProjectionHead(512->256) -> z_egg
  Loss   = VICReg(z_speech, z_egg)

Encoder variants:
  attn_bias: SpeechEGGEncoderAttnBias  (~29.1M + 2,200/enc)
  xattn:     SpeechEGGEncoderXAttn     (~29.1M + 1.06M/enc)

Usage:
  # V4-lin + attention bias (30ep)
  python experiments/bias_control/escalon2/train_escalon2_attn.py \\
      --lombard-dir data/lombard/FLombard \\
      --segment-index data/lombard/segment_index.json \\
      --f0-cache data/lombard/f0_cache_noise0.npz \\
      --output data/lombard/v4lin_attnbias_seed42 \\
      --injection attn_bias --epochs 30

  # H-series + cross-attention (30ep)
  python experiments/bias_control/escalon2/train_escalon2_attn.py \\
      --lombard-dir data/lombard/FLombard \\
      --segment-index data/lombard/segment_index.json \\
      --f0-cache data/lombard/f0_cache_noise0.npz \\
      --output data/lombard/hseries_xattn_seed42 \\
      --injection xattn --epochs 30

  # A4-16k + cross-attention (10ep control)
  python experiments/bias_control/escalon2/train_escalon2_attn.py \\
      --lombard-dir data/lombard/FLombard \\
      --segment-index data/lombard/segment_index.json \\
      --f0-cache data/lombard/f0_cache_noise0.npz \\
      --output data/lombard/a4_16k_xattn_seed42 \\
      --injection xattn --descriptor a4_16k --epochs 10

  # Smoke test (3ep x 50 batches)
  python experiments/bias_control/escalon2/train_escalon2_attn.py \\
      --lombard-dir data/lombard/FLombard \\
      --segment-index data/lombard/segment_index.json \\
      --f0-cache data/lombard/f0_cache_noise0.npz \\
      --output data/lombard/smoke_attn \\
      --injection attn_bias --epochs 3 --max-batches 50
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# ── New encoder classes ──────────────────────────────────────────────────────
from src.bias_control.encoders.speech_egg_encoder_attn_bias import (
    SpeechEGGEncoderAttnBias,
)
from src.bias_control.encoders.speech_egg_encoder_xattn import (
    SpeechEGGEncoderXAttn,
)

# ── Shared components (no reimplementation) ──────────────────────────────────
from src.bias_control.encoders.projection import ProjectionHead
from src.RNA.vicreg import VICRegLoss
from src.bias_control.training.preflight import DriftSentinel
from src.bias_control.datasets.lombard_segments_aug import (
    create_lombard_dataloaders_aug,
)
from src.bias_control.vocal_descriptors import get_descriptor_dim

# Import shared utilities from train_escalon2_descriptors.py
from experiments.bias_control.escalon2.train_escalon2_descriptors import (
    DescriptorComputer,
    precompute_h_series_stats,
    save_h_series_stats,
    LinearWarmupCosineScheduler,
    seed_everything,
    seed_worker,
)

# Import eval functions
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


# ── Injection defaults ───────────────────────────────────────────────────────

INJECTION_DEFAULT_DESCRIPTOR = {
    'attn_bias': 'v4_lin',
    'xattn': 'h_series',
}


# ── Usage metrics ────────────────────────────────────────────────────────────

def collect_usage_metrics(speech_enc, egg_enc, injection_type):
    """Collect injection-specific usage metrics (no forward pass needed).

    Returns dict with scalar metrics for logging.
    """
    metrics = {}

    if injection_type == 'attn_bias':
        for name, enc in [('speech', speech_enc), ('egg', egg_enc)]:
            bc = enc.bias_computer
            metrics[f'{name}_bias_scale'] = bc.bias_scale.item()
            metrics[f'{name}_W_norm'] = bc.W.data.norm().item()
            metrics[f'{name}_W_max'] = bc.W.data.abs().max().item()
            metrics[f'{name}_phi_w_norm'] = bc.phi_net[0].weight.data.norm().item()
            metrics[f'{name}_psi_w_norm'] = bc.psi_net[0].weight.data.norm().item()

    elif injection_type == 'xattn':
        for name, enc in [('speech', speech_enc), ('egg', egg_enc)]:
            metrics[f'{name}_xattn_scale'] = enc.xattn_scale.item()
            metrics[f'{name}_desc_proj_w_norm'] = enc.desc_proj.weight.data.norm().item()
            # MHA in_proj weight norm (combined Q/K/V projection)
            for pname, p in enc.cross_attention.named_parameters():
                if 'in_proj_weight' in pname:
                    metrics[f'{name}_mha_in_proj_norm'] = p.data.norm().item()
                    break

    return metrics


# ── Training ─────────────────────────────────────────────────────────────────

def train_one_epoch(speech_enc, egg_enc, proj_speech, proj_egg,
                    vicreg_loss, optimizer, scheduler, train_loader,
                    device, descriptor_computer, max_batches=None):
    """Train one epoch with descriptor injection via attention."""
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
    parser = argparse.ArgumentParser(
        description='S2-P2.5: Attention-Based Descriptor Injection Training'
    )
    parser.add_argument('--lombard-dir', type=str, required=True)
    parser.add_argument('--segment-index', type=str, required=True)
    parser.add_argument('--f0-cache', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--injection', type=str, required=True,
                        choices=['attn_bias', 'xattn'],
                        help='Injection mechanism: attn_bias or xattn')
    parser.add_argument('--descriptor', type=str, default=None,
                        choices=['v4_lin', 'v4_log', 'h_series', 'a4_16k'],
                        help='Descriptor type (default: auto from injection)')
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
    # AttnBias-specific
    parser.add_argument('--d-bias', type=int, default=16,
                        help='Bias embedding dim for attn_bias')
    # XAttn-specific
    parser.add_argument('--n-xattn-heads', type=int, default=4,
                        help='Number of cross-attention heads for xattn')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    seed_everything(args.seed)
    device = torch.device(args.device)

    # Auto-default descriptor from injection type
    if args.descriptor is None:
        args.descriptor = INJECTION_DEFAULT_DESCRIPTOR[args.injection]
        logger.info(f"Auto-selected descriptor: {args.descriptor} "
                    f"(default for {args.injection})")

    if args.structured_eval_epochs is None:
        args.structured_eval_epochs = [5, 10, 15, 20, 25, 28, 29, 30]

    desc_dim = get_descriptor_dim(args.descriptor)

    logger.info("=" * 60)
    logger.info(f"S2-P2.5: {args.injection.upper()} + {args.descriptor.upper()} "
                f"(dim={desc_dim})")
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
    logger.info(f"  Injection: {args.injection}")
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
    if args.injection == 'attn_bias':
        speech_enc = SpeechEGGEncoderAttnBias(
            descriptor_dim=desc_dim, d_bias=args.d_bias,
            output_dim=512, n_layers=4, n_heads=8,
        ).to(device)
        egg_enc = SpeechEGGEncoderAttnBias(
            descriptor_dim=desc_dim, d_bias=args.d_bias,
            output_dim=512, n_layers=4, n_heads=8,
        ).to(device)
        enc_desc = (f'SpeechEGGEncoderAttnBias(d=512, desc_dim={desc_dim}, '
                    f'd_bias={args.d_bias})')

    elif args.injection == 'xattn':
        speech_enc = SpeechEGGEncoderXAttn(
            descriptor_dim=desc_dim, n_xattn_heads=args.n_xattn_heads,
            output_dim=512, n_layers=4, n_heads=8,
        ).to(device)
        egg_enc = SpeechEGGEncoderXAttn(
            descriptor_dim=desc_dim, n_xattn_heads=args.n_xattn_heads,
            output_dim=512, n_layers=4, n_heads=8,
        ).to(device)
        enc_desc = (f'SpeechEGGEncoderXAttn(d=512, desc_dim={desc_dim}, '
                    f'xattn_heads={args.n_xattn_heads})')

    proj_speech = ProjectionHead(
        input_dim=512, hidden_dim=512, output_dim=256
    ).to(device)
    proj_egg = ProjectionHead(
        input_dim=512, hidden_dim=512, output_dim=256
    ).to(device)
    vicreg_loss = VICRegLoss(lambda_inv=10.0, lambda_var=10.0, lambda_cov=1.0)

    total_params = sum(
        p.numel() for p in
        list(speech_enc.parameters()) + list(egg_enc.parameters()) +
        list(proj_speech.parameters()) + list(proj_egg.parameters())
    )

    # Count injection-specific params
    base_params = sum(
        p.numel() for p in
        list(speech_enc.feature_extractor.parameters()) +
        list(speech_enc.transformer.parameters())
    ) + speech_enc.pos_embedding.numel()
    injection_params_per_enc = sum(p.numel() for p in speech_enc.parameters()) - base_params

    logger.info(f"  Speech encoder: {sum(p.numel() for p in speech_enc.parameters()):,} params")
    logger.info(f"  EGG encoder: {sum(p.numel() for p in egg_enc.parameters()):,} params")
    logger.info(f"  Injection params per encoder: {injection_params_per_enc:,}")
    logger.info(f"  Total: {total_params:,} params (all trainable)")

    # ── Optimizer ─────────────────────────────────────────────────────────────
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
        'mode': 'train_attn_injection',
        'injection': args.injection,
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
        'injection_params_per_encoder': injection_params_per_enc,
        'n_train': n_train,
        'n_val': n_val,
        'n_test': n_test,
        'batches_per_epoch': batches_per_epoch,
        'max_batches': args.max_batches,
        'structured_eval_epochs': args.structured_eval_epochs,
        'encoder': enc_desc,
        'projection': 'ProjectionHead(512->512->256)',
        'loss': 'VICReg(inv=10, var=10, cov=1)',
        'f0_cache': args.f0_cache,
        'h_series_norm_stats': 'per_modality' if args.descriptor == 'h_series' else None,
        'd0_baseline_S': 0.778,
        'concat_baseline_S': {
            'v4_lin': 0.678,
            'h_series': 0.598,
            'a4_16k': 0.778,
        }.get(args.descriptor, None),
    }
    if args.injection == 'attn_bias':
        config['d_bias'] = args.d_bias
    elif args.injection == 'xattn':
        config['n_xattn_heads'] = args.n_xattn_heads

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

        # Usage metrics (cheap, parameter-level only)
        usage_metrics = collect_usage_metrics(
            speech_enc, egg_enc, args.injection
        )

        # DriftSentinel after epoch 1
        if epoch == 1:
            drift = sentinel.check(all_modules)
            n_drifted = sum(1 for d in drift.values() if d > 0)
            logger.info(f"  DriftSentinel: {n_drifted}/{len(drift)} params drifted")
            if n_drifted == 0:
                logger.error("  GHOST TRAINING DETECTED — no parameters moved!")
                return

            # Log initial usage metrics
            logger.info("  Usage metrics (ep1):")
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
                    'injection': args.injection,
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
            'injection': args.injection,
            'descriptor': args.descriptor,
            'descriptor_dim': desc_dim,
        }, os.path.join(args.output, f'checkpoint_ep{epoch:03d}.pt'))

        ep_time = time.time() - t_ep

        # Log epoch summary
        log_msg = (
            f"Epoch {epoch}/{args.epochs} ({ep_time:.1f}s): "
            f"loss={train_metrics['loss']:.4f} "
            f"val={val_loss:.4f} "
            f"inv={train_metrics['invariance']:.3f} "
            f"var={train_metrics['variance']:.3f} "
            f"cov={train_metrics['covariance']:.3f}"
        )

        # Add key usage metric to log line
        if args.injection == 'attn_bias':
            bs = usage_metrics.get('speech_bias_scale', 0)
            wn = usage_metrics.get('speech_W_norm', 0)
            log_msg += f" | bs={bs:.4f} Wn={wn:.3f}"
        elif args.injection == 'xattn':
            xs = usage_metrics.get('speech_xattn_scale', 0)
            log_msg += f" | xs={xs:.4f}"

        if eval_results:
            log_msg += f" | S={eval_results['S']:.1%}"

        logger.info(log_msg)

        # Build history entry
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

    # ── Summary ───────────────────────────────────────────────────────────────
    concat_S = config.get('concat_baseline_S', None)
    logger.info("=" * 60)
    logger.info(f"TRAINING COMPLETE: {total_time/60:.1f} minutes")
    logger.info(f"  Injection: {args.injection}")
    logger.info(f"  Descriptor: {args.descriptor} (dim={desc_dim})")
    logger.info(f"  Best S = {best_S:.1%} @ epoch {best_epoch}")
    logger.info(f"  D0 baseline: S=77.8%")
    logger.info(f"  Delta vs D0: {(best_S - 0.778)*100:+.1f}pp")
    if concat_S is not None:
        logger.info(f"  Concat baseline ({args.descriptor}): S={concat_S:.1%}")
        logger.info(f"  Delta vs concat: {(best_S - concat_S)*100:+.1f}pp")
    logger.info("=" * 60)

    # Final usage metrics
    final_usage = collect_usage_metrics(speech_enc, egg_enc, args.injection)
    logger.info("Final usage metrics:")
    for k, v in final_usage.items():
        logger.info(f"  {k}: {v:.6f}")

    # Save history and summary
    with open(os.path.join(args.output, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    summary = {
        'injection': args.injection,
        'descriptor': args.descriptor,
        'descriptor_dim': desc_dim,
        'best_S': best_S,
        'best_epoch': best_epoch,
        'total_time_min': total_time / 60,
        'total_params': total_params,
        'injection_params_per_encoder': injection_params_per_enc,
        'n_epochs': args.epochs,
        'condition': args.condition,
        'd0_baseline_S': 0.778,
        'concat_baseline_S': concat_S,
        'delta_vs_d0_pp': round((best_S - 0.778) * 100, 1),
        'delta_vs_concat_pp': round((best_S - concat_S) * 100, 1) if concat_S else None,
        'final_usage_metrics': final_usage,
    }
    with open(os.path.join(args.output, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)


if __name__ == '__main__':
    main()
