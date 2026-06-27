#!/usr/bin/env python3
"""E3-P1: Parameter Recovery — validates that the Lissajous dataset is learnable.

Trains a simple classifier (encoder + linear heads) to predict ratio, phase, and
amplitude from either audio or image. Reports IID accuracy and OOD analysis.

Usage:
    python experiments/escalon3/f1_parameter_recovery.py \
        --modality audio \
        --data data/escalon3/bundled \
        --output data/escalon3/p1_audio_seed42 \
        --epochs 30 --seed 42

    python experiments/escalon3/f1_parameter_recovery.py \
        --modality image \
        --data data/escalon3/bundled \
        --output data/escalon3/p1_image_seed42 \
        --epochs 30 --seed 42
"""

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.escalon3.lissajous_dataset import (
    LissajousDataset, collate_lissajous, create_lissajous_dataloaders, load_ood_dataset
)
from src.escalon3.encoders import (
    build_lissajous_audio_encoder, build_lissajous_image_encoder,
)


# ============================================================
# Utilities
# ============================================================

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
                1, self.total_steps - self.warmup_steps)
            scale = 0.5 * (1 + math.cos(math.pi * progress))
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg['lr'] = base_lr * scale


# ============================================================
# Model
# ============================================================

class ParameterRecoveryModel(nn.Module):
    """Encoder + linear heads for ratio/phase/amp prediction."""

    def __init__(self, encoder: nn.Module, n_ratios: int = 16, n_phases: int = 8):
        super().__init__()
        self.encoder = encoder
        dim = encoder.output_dim
        self.ratio_head = nn.Linear(dim, n_ratios)
        self.phase_head = nn.Linear(dim, n_phases)
        self.amp_head = nn.Linear(dim, 1)
        self.ratio_float_head = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> dict:
        feat = self.encoder(x)
        return {
            'features': feat,
            'ratio_logits': self.ratio_head(feat),
            'phase_logits': self.phase_head(feat),
            'amp_pred': self.amp_head(feat).squeeze(-1),
            'ratio_float_pred': self.ratio_float_head(feat).squeeze(-1),
        }


# ============================================================
# Training
# ============================================================

def compute_loss(out: dict, batch: dict) -> dict:
    """Compute combined loss for all heads."""
    ratio_loss = F.cross_entropy(out['ratio_logits'], batch['ratio_id'])
    phase_loss = F.cross_entropy(out['phase_logits'], batch['phase_idx'])
    amp_loss = F.mse_loss(out['amp_pred'], batch['amp_ratio'])
    rf_loss = F.mse_loss(out['ratio_float_pred'], batch['ratio_float'])

    total = ratio_loss + phase_loss + amp_loss + rf_loss

    return {
        'total': total,
        'ratio_ce': ratio_loss.item(),
        'phase_ce': phase_loss.item(),
        'amp_mse': amp_loss.item(),
        'rf_mse': rf_loss.item(),
    }


def _resolve_input_key(modality: str, encoder) -> str:
    """Resolve batch key from modality + encoder.input_kind."""
    if modality == 'image':
        return 'image'
    # Audio modality: check if encoder needs CQT
    kind = getattr(encoder, 'input_kind', 'waveform')
    if hasattr(encoder, 'encoder'):
        kind = getattr(encoder.encoder, 'input_kind', 'waveform')
    return 'audio_cqt' if kind == 'cqt' else 'audio'


def train_one_epoch(model, loader, optimizer, scheduler, device, modality):
    model.train()
    total_loss = 0
    n_batches = 0
    input_key = _resolve_input_key(modality, model)

    for batch in tqdm(loader, desc="Train", leave=False):
        x = batch[input_key].to(device)
        batch_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

        out = model(x)
        loss_dict = compute_loss(out, batch_gpu)

        optimizer.zero_grad()
        loss_dict['total'].backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss_dict['total'].item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, device, modality) -> dict:
    model.eval()
    input_key = _resolve_input_key(modality, model)
    all_ratio_correct = 0
    all_phase_correct = 0
    all_amp_preds = []
    all_amp_targets = []
    all_rf_preds = []
    all_rf_targets = []
    n_samples = 0

    for batch in loader:
        x = batch[input_key].to(device)
        batch_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

        out = model(x)
        B = x.shape[0]

        # Ratio accuracy
        ratio_pred = out['ratio_logits'].argmax(dim=-1)
        all_ratio_correct += (ratio_pred == batch_gpu['ratio_id']).sum().item()

        # Phase accuracy
        phase_pred = out['phase_logits'].argmax(dim=-1)
        all_phase_correct += (phase_pred == batch_gpu['phase_idx']).sum().item()

        # Amplitude
        all_amp_preds.append(out['amp_pred'].cpu())
        all_amp_targets.append(batch['amp_ratio'])

        # Ratio float
        all_rf_preds.append(out['ratio_float_pred'].cpu())
        all_rf_targets.append(batch['ratio_float'])

        n_samples += B

    # Compute metrics
    ratio_acc = all_ratio_correct / max(n_samples, 1)
    phase_acc = all_phase_correct / max(n_samples, 1)

    amp_preds = torch.cat(all_amp_preds)
    amp_targets = torch.cat(all_amp_targets)
    amp_r2 = 1.0 - ((amp_preds - amp_targets)**2).mean() / max(amp_targets.var(), 1e-8)

    rf_preds = torch.cat(all_rf_preds)
    rf_targets = torch.cat(all_rf_targets)
    rf_mae = (rf_preds - rf_targets).abs().mean().item()
    rf_r2 = 1.0 - ((rf_preds - rf_targets)**2).mean() / max(rf_targets.var(), 1e-8)

    return {
        'n_samples': n_samples,
        'ratio_acc': ratio_acc,
        'phase_acc': phase_acc,
        'amp_r2': amp_r2.item(),
        'ratio_float_mae': rf_mae,
        'ratio_float_r2': rf_r2.item(),
    }


# ============================================================
# OOD Evaluation
# ============================================================

@torch.no_grad()
def evaluate_ood(model, dataset, device, modality, split_name) -> dict:
    """Evaluate on OOD split with open-set analysis."""
    model.eval()
    input_key = _resolve_input_key(modality, model)
    loader = DataLoader(dataset, batch_size=64, shuffle=False,
                        collate_fn=collate_lissajous, num_workers=4)

    all_ratio_preds = []
    all_ratio_ids = []
    all_equiv_ids = []
    all_rf_preds = []
    all_rf_targets = []
    all_features = []

    for batch in loader:
        x = batch[input_key].to(device)
        out = model(x)

        all_ratio_preds.append(out['ratio_logits'].argmax(dim=-1).cpu())
        all_ratio_ids.append(batch['ratio_id'])
        all_equiv_ids.append(batch['equiv_id'])
        all_rf_preds.append(out['ratio_float_pred'].cpu())
        all_rf_targets.append(batch['ratio_float'])
        all_features.append(out['features'].cpu())

    ratio_preds = torch.cat(all_ratio_preds)
    ratio_ids = torch.cat(all_ratio_ids)
    equiv_ids = torch.cat(all_equiv_ids)
    rf_preds = torch.cat(all_rf_preds)
    rf_targets = torch.cat(all_rf_targets)
    features = torch.cat(all_features)

    # For equiv_ood: check if predicted class matches equiv_id (reduced form)
    equiv_correct = (ratio_preds == equiv_ids).float().mean().item()

    # For ratio_ood: report which classes they map to
    rf_mae = (rf_preds - rf_targets).abs().mean().item()

    # Report predicted class distribution for each true ratio
    pred_by_true = {}
    for i in range(len(ratio_ids)):
        true_id = int(ratio_ids[i])
        pred_id = int(ratio_preds[i])
        pred_by_true.setdefault(true_id, []).append(pred_id)

    class_mapping = {}
    for true_id, preds in pred_by_true.items():
        from collections import Counter
        counts = Counter(preds)
        top = counts.most_common(3)
        class_mapping[true_id] = {
            'top_predictions': [(cls, cnt / len(preds)) for cls, cnt in top],
            'n_samples': len(preds),
        }

    return {
        'split': split_name,
        'n_samples': len(ratio_ids),
        'equiv_accuracy': equiv_correct,
        'ratio_float_mae': rf_mae,
        'class_mapping': class_mapping,
    }


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="E3-P1: Parameter Recovery")
    parser.add_argument("--modality", choices=['audio', 'image'], required=True)
    parser.add_argument("--audio-encoder", default="baseline",
                        choices=["baseline", "attpool", "resatt", "cnnxfmr",
                                 "cqtconv", "cqtshift", "cqthybrid"])
    parser.add_argument("--image-encoder", default="baseline",
                        choices=["baseline"])
    parser.add_argument("--data", default="data/escalon3/bundled")
    parser.add_argument("--output", default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.output is None:
        args.output = f"data/escalon3/p1_{args.modality}_seed{args.seed}"

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Data
    train_loader, val_loader, test_loader = create_lissajous_dataloaders(
        args.data, batch_size=args.batch_size, num_workers=8
    )
    print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, "
          f"Test: {len(test_loader.dataset)}")

    # Model
    if args.modality == 'audio':
        encoder = build_lissajous_audio_encoder(args.audio_encoder, output_dim=256)
    else:
        encoder = build_lissajous_image_encoder(args.image_encoder, output_dim=256)

    model = ParameterRecoveryModel(encoder, n_ratios=16, n_phases=8).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = LinearWarmupCosineScheduler(optimizer, args.warmup_steps, total_steps)

    # Training loop
    best_val_acc = 0.0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler,
                                     device, args.modality)
        val_metrics = evaluate(model, val_loader, device, args.modality)
        val_acc = val_metrics['ratio_acc']

        log = {
            'epoch': epoch,
            'train_loss': train_loss,
            **{f'val_{k}': v for k, v in val_metrics.items()},
        }
        history.append(log)

        print(f"E{epoch:02d} | loss={train_loss:.4f} | "
              f"val_ratio={val_acc:.3f} phase={val_metrics['phase_acc']:.3f} "
              f"amp_r2={val_metrics['amp_r2']:.3f} rf_mae={val_metrics['ratio_float_mae']:.4f}")

        # Checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_ratio_acc': val_acc,
        }
        torch.save(checkpoint, output_dir / f'checkpoint_e{epoch:02d}.pt')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(checkpoint, output_dir / 'best_model.pt')

    # Final evaluation should use the best validation checkpoint, not the last epoch.
    best_ckpt = torch.load(output_dir / 'best_model.pt', map_location=device)
    model.load_state_dict(best_ckpt['model_state_dict'])

    # Final test evaluation
    print("\n=== IID TEST EVALUATION ===")
    test_metrics = evaluate(model, test_loader, device, args.modality)
    for k, v in test_metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    # OOD evaluations
    print("\n=== RATIO-OOD EVALUATION ===")
    ratio_ood_ds = load_ood_dataset(args.data, 'ratio_ood')
    ratio_ood_results = evaluate_ood(model, ratio_ood_ds, device, args.modality, 'ratio_ood')
    print(f"  ratio_float MAE: {ratio_ood_results['ratio_float_mae']:.4f}")
    for true_id, info in ratio_ood_results['class_mapping'].items():
        top = info['top_predictions']
        top_str = ', '.join(f"class {c}: {p:.1%}" for c, p in top)
        print(f"  ratio_id={true_id} (n={info['n_samples']}): {top_str}")

    print(f"\n=== EQUIVALENCE-OOD EVALUATION ({args.modality}) ===")
    equiv_ood_ds = load_ood_dataset(args.data, 'equiv_ood')
    equiv_ood_results = evaluate_ood(model, equiv_ood_ds, device, args.modality, 'equiv_ood')
    print(f"  equiv accuracy: {equiv_ood_results['equiv_accuracy']:.4f}")
    for true_id, info in equiv_ood_results['class_mapping'].items():
        top = info['top_predictions']
        top_str = ', '.join(f"class {c}: {p:.1%}" for c, p in top)
        print(f"  equiv_id={true_id} (n={info['n_samples']}): {top_str}")

    # Save all results
    all_results = {
        'modality': args.modality,
        'config': vars(args),
        'history': history,
        'test_metrics': test_metrics,
        'ratio_ood': ratio_ood_results,
        'equiv_ood': equiv_ood_results,
    }
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    # GO/NO-GO summary (aligned with CRITERIOS_GO_NO_GO_ESCALON_3.md)
    # Hierarchy: canonical > heuristic > non-blocking diagnostic
    print("\n=== GO/NO-GO SUMMARY ===")
    print("  (See CRITERIOS_GO_NO_GO_ESCALON_3.md for canonical vs heuristic distinction)")

    # --- Canonical: ratio must be recoverable ---
    iid_ok = test_metrics['ratio_acc'] > 0.95
    print(f"\n  [CANONICAL] IID ratio acc: {test_metrics['ratio_acc']:.3f} "
          f"{'GO' if iid_ok else 'FAIL'} (heuristic >0.95)")

    # --- Canonical: OOD must be interpretable ---
    equiv_thresh = 0.90 if args.modality == 'image' else 0.50
    equiv_ok = equiv_ood_results['equiv_accuracy'] > equiv_thresh
    print(f"  [CANONICAL] equiv-OOD acc: {equiv_ood_results['equiv_accuracy']:.3f} "
          f"{'GO' if equiv_ok else 'NEEDS REVIEW'} (heuristic >{equiv_thresh})")

    # --- Heuristic: phase (partially non-identifiable by construction) ---
    print(f"\n  [HEURISTIC] IID phase acc: {test_metrics['phase_acc']:.3f} "
          f"(target >0.80 — degeneracies exist in {'image' if args.modality == 'image' else 'audio w/ global pool'})")

    # --- Non-blocking: amp in image is not identifiable by construction ---
    if args.modality == 'image':
        print(f"  [NON-BLOCKING] IID amp R²: {test_metrics['amp_r2']:.3f} "
              f"(renderer normalizes per-axis → amp not recoverable from image)")
    else:
        amp_ok = test_metrics['amp_r2'] > 0.90
        print(f"  [HEURISTIC] IID amp R²: {test_metrics['amp_r2']:.3f} "
              f"{'GO' if amp_ok else 'NEEDS REVIEW'} (heuristic >0.90)")

    # --- Overall: only canonical criteria matter ---
    canonical_go = iid_ok  # ratio recoverable is the core criterion
    print(f"\n  Best checkpoint: epoch {best_ckpt['epoch']} (val_ratio={best_ckpt['val_ratio_acc']:.3f})")
    print(f"  Canonical reading: {'GO for ratio learnability' if canonical_go else 'FAIL — ratio not recoverable'}")


if __name__ == "__main__":
    main()
