"""
Gate 6 — Exp C: AMT Decoder over Frozen VICReg Features

Trains a powerful AMT decoder (~38M params) over pre-pooling features from
frozen VICReg encoders (Gate 5B checkpoints). Tests whether descriptor-
conditioned features are more musically decodifiable.

Regime: Phideus (24kHz, 4s segments, 188 frames)

Arms:
    D0      [B, 2400, 1024] baseline
    d4a4    [B, 2400, 1024] concat A4+D4
    a4r     [B, 188, 1024]  reverse cross-att A4
    d4-a4r  [B, 188, 1024]  mixed reverse A4+D4

Usage:
    python experiments/bias_control/gate6/vicreg_amt_decoder.py \
        --descriptor d4a4 \
        --maestro-dir data/maestro_v3/maestro-v3.0.0 \
        --output data/gate6_results/vicreg_decoder/d4a4 \
        --epochs 80 --batch-size 16 --device cuda

Reuses:
    - EncoderFeatureExtractor from test13g_posthoc_decoder.py
    - build_pr_targets from test13g_generative_encoder.py
    - MaestroSegmentDataset from maestro_segments.py
    - checkpoint_loader from gate5b
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset

# ── Project imports ────────────────────────────────────────────────────────

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.bias_control.gate5b.checkpoint_loader import (
    load_model_from_checkpoint,
    get_eval_batch_size,
)
from experiments.bias_control.gate5b.test13g_posthoc_decoder import (
    EncoderFeatureExtractor,
    pool_features_to_length,
    compute_all_metrics,
    N_FRAMES,
    N_PITCHES,
)
from experiments.bias_control.gate5b.test13g_generative_encoder import (
    build_pr_targets,
)
from src.bias_control.datasets.maestro_segments import (
    MaestroSegmentDataset,
    collate_segments,
)
from experiments.bias_control.gate6.amt_decoder_model import (
    AMTDecoder,
    FocalLoss,
    detect_onsets,
)

# ── Constants ──────────────────────────────────────────────────────────────

CHECKPOINT_MAP = {
    'd0':     'models/gate5b/D0/best_model.pt',
    'd4a4':   'models/gate5b/d4a4/best_model.pt',
    'a4r':    'models/gate5b/a4r/best_model.pt',
    'd4-a4r': 'models/gate5b/d4-a4r/best_model.pt',
}

DESCRIPTOR_MAP = {
    'd0':     'd0',
    'd4a4':   'd4a4',
    'a4r':    'a4r',
    'd4-a4r': 'd4-a4r',
}

# Expected pre-pooling feature sequence lengths
EXPECTED_N = {
    'd0':     2400,   # 2300-2500
    'd4a4':   2400,   # 2300-2500
    'a4r':    188,    # 180-200
    'd4-a4r': 188,    # 180-200
}

DEFAULT_CONFIG = {
    'lr': 3e-4,
    'weight_decay': 0.01,
    'warmup_epochs': 5,
    'epochs': 80,
    'eval_every': 5,
    'checkpoint_every': 1,
    'batch_size': 16,
    'seed': 42,
    'focal_alpha': 0.25,
    'focal_gamma': 2.0,
    'onset_weight': 5.0,
    'pool_to': 188,
    'grad_clip': 1.0,
    # AMTDecoder config
    'd_model': 512,
    'n_heads': 8,
    'n_layers': 8,
    'd_ff': 2048,
    'dropout': 0.1,
}


# ── Cosine annealing with warmup ──────────────────────────────────────────

class CosineWarmupScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr_ratio=0.01):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr_ratio = min_lr_ratio
        self.base_lrs = [pg['lr'] for pg in optimizer.param_groups]

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            # Linear warmup
            factor = (epoch + 1) / self.warmup_epochs
        else:
            # Cosine annealing
            progress = (epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
            factor = self.min_lr_ratio + 0.5 * (1 - self.min_lr_ratio) * (1 + np.cos(np.pi * progress))

        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg['lr'] = base_lr * factor


# ── Piano roll target building ─────────────────────────────────────────────

def build_targets_from_batch(batch, device):
    """Build piano roll targets from a collated batch."""
    pr = build_pr_targets(
        midi_pitch=batch['midi_pitch'].to(device),
        midi_velocity=batch['midi_velocity'].to(device),
        midi_onset=batch['midi_onset'].to(device),
        midi_duration_sec=batch['midi_duration_sec'].to(device),
        midi_mask=batch['midi_mask'].to(device),
    )
    return pr  # [B, 188, 88]


# ── Feature extraction ─────────────────────────────────────────────────────

@torch.no_grad()
def extract_features_batch(model, batch, device, descriptor):
    """Extract pre-pooling features from frozen encoder."""
    audio = batch['audio'].to(device)
    midi_pitch = batch['midi_pitch'].to(device)
    midi_velocity = batch['midi_velocity'].to(device)
    midi_duration = batch['midi_duration'].to(device)
    midi_mask = batch['midi_mask'].to(device)

    extractor = EncoderFeatureExtractor(model)
    with extractor:
        model(audio, midi_pitch, midi_velocity, midi_duration, midi_mask)
        features = extractor.pop_features()

    return features  # [B, N, 1024]


# ── Training loop ──────────────────────────────────────────────────────────

def train_epoch(
    encoder_model, decoder, optimizer, criterion,
    train_loader, device, descriptor, config,
    pool_to=None,
):
    """Train decoder for one epoch."""
    decoder.train()
    total_loss = 0
    n_batches = 0

    for batch in train_loader:
        # Extract features (frozen encoder)
        features = extract_features_batch(encoder_model, batch, device, descriptor)

        # Build piano roll targets
        targets = build_targets_from_batch(batch, device)

        # Forward decoder
        logits = decoder(features, pool_to=pool_to)

        # Compute loss
        onset_mask = detect_onsets(targets)
        loss = criterion(logits, targets, onset_mask)

        # Backward
        optimizer.zero_grad()
        loss.backward()

        if config.get('grad_clip', 0) > 0:
            nn.utils.clip_grad_norm_(decoder.parameters(), config['grad_clip'])

        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(
    encoder_model, decoder, val_loader, device, descriptor,
    pool_to=None, max_batches=200,
):
    """Evaluate decoder on validation set."""
    decoder.eval()
    all_logits = []
    all_targets = []

    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break

        features = extract_features_batch(encoder_model, batch, device, descriptor)
        targets = build_targets_from_batch(batch, device)

        logits = decoder(features, pool_to=pool_to)

        all_logits.append(logits.cpu().numpy())
        all_targets.append(targets.cpu().numpy())

    all_logits = np.concatenate(all_logits, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    metrics = compute_all_metrics(all_logits, all_targets)
    return metrics


def train_amt_decoder(
    descriptor: str,
    maestro_dir: str,
    output_dir: str,
    config: dict,
    device: str = 'cuda',
):
    """Full training pipeline for Exp C."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Save config
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # ── Load frozen encoder ────────────────────────────────────────────
    ckpt_path = CHECKPOINT_MAP[descriptor]
    desc_name = DESCRIPTOR_MAP[descriptor]

    print(f"\n{'='*60}")
    print(f"  Gate 6 Exp C: AMT Decoder — {descriptor}")
    print(f"{'='*60}")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"  Descriptor: {desc_name}")
    print(f"  Device: {device}")

    model, metadata = load_model_from_checkpoint(
        ckpt_path, device=torch.device(device),
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    print(f"  Encoder loaded: {metadata.get('descriptor', '?')} "
          f"(S={metadata.get('best_S', 0):.3f})")

    # ── Create decoder ─────────────────────────────────────────────────
    decoder = AMTDecoder(
        feat_dim=1024,
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers'],
        d_ff=config['d_ff'],
        n_frames=N_FRAMES,
        n_pitches=N_PITCHES,
        dropout=config['dropout'],
    ).to(device)

    print(f"  Decoder params: {decoder.num_params_M:.1f}M")

    # ── Data ───────────────────────────────────────────────────────────
    train_ds = MaestroSegmentDataset(
        maestro_dir, segment_len=4.0, hop=2.0,
        sample_rate=24000, split='train',
    )
    val_ds = MaestroSegmentDataset(
        maestro_dir, segment_len=4.0, hop=2.0,
        sample_rate=24000, split='validation',
    )

    # Subsample if needed
    if len(train_ds) > 10000:
        rng = np.random.RandomState(seed)
        indices = rng.choice(len(train_ds), 10000, replace=False)
        train_ds = Subset(train_ds, indices)
    if len(val_ds) > 5000:
        rng = np.random.RandomState(seed)
        indices = rng.choice(len(val_ds), 5000, replace=False)
        val_ds = Subset(val_ds, indices)

    print(f"  Train segments: {len(train_ds)}")
    print(f"  Val segments: {len(val_ds)}")

    batch_size = config['batch_size']
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_segments, num_workers=8,
        pin_memory=True, prefetch_factor=2,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_segments, num_workers=8,
        pin_memory=True,
    )

    # ── Optimizer + Scheduler ──────────────────────────────────────────
    optimizer = AdamW(
        decoder.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay'],
    )
    scheduler = CosineWarmupScheduler(
        optimizer,
        warmup_epochs=config['warmup_epochs'],
        total_epochs=config['epochs'],
    )

    criterion = FocalLoss(
        alpha=config['focal_alpha'],
        gamma=config['focal_gamma'],
        onset_weight=config['onset_weight'],
    )

    pool_to = config.get('pool_to', N_FRAMES)

    # ── Training ───────────────────────────────────────────────────────
    history = []
    best_f1 = 0
    best_epoch = 0
    t_start = time.time()

    print(f"\n  Training for {config['epochs']} epochs...")
    print(f"  {'Epoch':<6} {'Loss':<10} {'LR':<10} {'F1':<8} {'Onset_F1':<10} {'Best_F1':<10}")
    print(f"  {'-'*6} {'-'*10} {'-'*10} {'-'*8} {'-'*10} {'-'*10}")

    for epoch in range(1, config['epochs'] + 1):
        scheduler.step(epoch - 1)

        t_epoch = time.time()
        train_loss = train_epoch(
            model, decoder, optimizer, criterion,
            train_loader, device, descriptor, config,
            pool_to=pool_to,
        )

        lr = optimizer.param_groups[0]['lr']

        # Evaluate
        metrics = None
        if epoch % config['eval_every'] == 0 or epoch == config['epochs']:
            metrics = evaluate(
                model, decoder, val_loader, device, descriptor,
                pool_to=pool_to,
            )

            f1 = metrics['frame_f1']
            onset_f1 = metrics.get('onset_f1', 0)

            if f1 > best_f1:
                best_f1 = f1
                best_epoch = epoch
                torch.save({
                    'epoch': epoch,
                    'decoder_state_dict': decoder.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_f1': best_f1,
                    'metrics': metrics,
                    'config': config,
                    'descriptor': descriptor,
                }, output_dir / 'best_decoder.pt')

            print(f"  {epoch:<6} {train_loss:<10.4f} {lr:<10.6f} "
                  f"{f1:<8.4f} {onset_f1:<10.4f} {best_f1:<10.4f}")

            # Save eval results
            eval_dir = output_dir / 'eval_per_epoch'
            eval_dir.mkdir(exist_ok=True)
            with open(eval_dir / f'eval_epoch{epoch}.json', 'w') as f:
                json.dump(metrics, f, indent=2)
        else:
            print(f"  {epoch:<6} {train_loss:<10.4f} {lr:<10.6f} "
                  f"{'—':<8} {'—':<10} {best_f1:<10.4f}")

        # Save epoch entry
        entry = {
            'epoch': epoch,
            'train_loss': train_loss,
            'lr': lr,
            'elapsed_s': time.time() - t_epoch,
        }
        if metrics:
            entry.update(metrics)
        history.append(entry)

        # Checkpoint
        if config.get('checkpoint_every', 0) > 0 and epoch % config['checkpoint_every'] == 0:
            ckpt_dir = output_dir / 'checkpoints'
            ckpt_dir.mkdir(exist_ok=True)
            torch.save({
                'epoch': epoch,
                'decoder_state_dict': decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'config': config,
                'descriptor': descriptor,
            }, ckpt_dir / f'checkpoint_epoch{epoch}.pt')

    total_time = time.time() - t_start

    # ── Final results ──────────────────────────────────────────────────
    final_results = {
        'descriptor': descriptor,
        'best_f1': best_f1,
        'best_epoch': best_epoch,
        'total_time_minutes': total_time / 60,
        'decoder_params_M': decoder.num_params_M,
        'history': history,
    }

    with open(output_dir / 'training_results.json', 'w') as f:
        json.dump(final_results, f, indent=2, default=float)

    print(f"\n  {'='*60}")
    print(f"  DONE: {descriptor}")
    print(f"  Best F1: {best_f1:.4f} @ epoch {best_epoch}")
    print(f"  Time: {total_time/60:.1f} min")
    print(f"  Results: {output_dir}")
    print(f"  {'='*60}")

    return final_results


# ── CLI ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Gate 6 Exp C: AMT Decoder over VICReg Features',
    )
    parser.add_argument('--descriptor', type=str, required=True,
                        choices=list(CHECKPOINT_MAP.keys()),
                        help='Which arm to evaluate')
    parser.add_argument('--maestro-dir', type=str, required=True,
                        help='Path to MAESTRO v3.0.0 directory')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--eval-every', type=int, default=5)
    parser.add_argument('--checkpoint-every', type=int, default=1)
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint path')

    args = parser.parse_args()

    config = dict(DEFAULT_CONFIG)
    config['epochs'] = args.epochs
    config['batch_size'] = args.batch_size
    config['lr'] = args.lr
    config['seed'] = args.seed
    config['eval_every'] = args.eval_every
    config['checkpoint_every'] = args.checkpoint_every

    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    train_amt_decoder(
        descriptor=args.descriptor,
        maestro_dir=args.maestro_dir,
        output_dir=args.output,
        config=config,
        device=args.device,
    )


if __name__ == '__main__':
    main()
