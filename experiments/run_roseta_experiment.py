#!/usr/bin/env python3
"""
Roseta Experiment - Multi-Phase Training and Evaluation
========================================================

Implements the Roseta Experiment protocol for validating cross-modal
harmonic alignment between Audio and Vibration domains.

Phases:
-------
1. TRAIN ON HEALTHY: Train VAE on HH (healthy) data only
2. INJECT FAULTS: Evaluate on fault conditions without retraining
3. CROSS-RETRIEVAL: Test Audio → z_shared → Vibration translation

Usage:
------
# Phase 1: Train on healthy
python experiments/run_roseta_experiment.py \
    --phase train \
    --data data/datasets/roseta_full.npz \
    --output data/training_outputs/roseta \
    --epochs 50

# Phase 2: Evaluate on faults
python experiments/run_roseta_experiment.py \
    --phase evaluate \
    --data data/datasets/roseta_full.npz \
    --checkpoint data/training_outputs/roseta/best_model.pt

# Phase 3: Cross-retrieval test
python experiments/run_roseta_experiment.py \
    --phase cross-retrieval \
    --data data/datasets/roseta_full.npz \
    --checkpoint data/training_outputs/roseta/best_model.pt

# Full experiment
python experiments/run_roseta_experiment.py \
    --phase full \
    --data data/datasets/roseta_full.npz \
    --output data/training_outputs/roseta \
    --epochs 50
"""

from __future__ import annotations

import argparse
import json
import os
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

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.datasets.roseta_dataset import (
    RosetaDataset,
    create_roseta_dataloaders,
    create_evaluation_loader,
)
from src.RNA.roseta_vae import RosetaVAE, compute_alignment_metrics


# ───────────────────────────────────
# Training Functions
# ───────────────────────────────────

def train_epoch(
    model: RosetaVAE,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    device: str,
    beta_kl: float = 1.0,
    lambda_infonce: float = 1.0,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_losses = {
        'total': 0, 'recon_audio': 0, 'recon_vibration': 0,
        'kl_total': 0, 'infonce': 0
    }
    n_batches = 0

    for batch in tqdm(train_loader, desc="Training", leave=False):
        audio, vibration, metas, lengths = batch
        audio = audio.to(device)
        vibration = vibration.to(device)
        lengths = lengths.to(device)

        optimizer.zero_grad()

        outputs = model(audio, vibration, lengths)
        losses = model.compute_loss(
            audio, vibration, outputs, lengths,
            beta_kl=beta_kl,
            lambda_infonce=lambda_infonce,
        )

        losses['total'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        for k in total_losses:
            total_losses[k] += losses[k].item()
        n_batches += 1

    return {k: v / n_batches for k, v in total_losses.items()}


@torch.no_grad()
def validate_epoch(
    model: RosetaVAE,
    val_loader: DataLoader,
    device: str,
    beta_kl: float = 1.0,
    lambda_infonce: float = 1.0,
) -> Dict[str, float]:
    """Validate for one epoch."""
    model.eval()
    total_losses = {
        'total': 0, 'recon_audio': 0, 'recon_vibration': 0,
        'kl_total': 0, 'infonce': 0
    }
    all_audio_z = []
    all_vibration_z = []
    n_batches = 0

    for batch in tqdm(val_loader, desc="Validation", leave=False):
        audio, vibration, metas, lengths = batch
        audio = audio.to(device)
        vibration = vibration.to(device)
        lengths = lengths.to(device)

        outputs = model(audio, vibration, lengths)
        losses = model.compute_loss(
            audio, vibration, outputs, lengths,
            beta_kl=beta_kl,
            lambda_infonce=lambda_infonce,
        )

        for k in total_losses:
            total_losses[k] += losses[k].item()

        # Collect z_shared for alignment metrics
        all_audio_z.append(outputs['audio_z_shared'].cpu())
        all_vibration_z.append(outputs['vibration_z_shared'].cpu())
        n_batches += 1

    avg_losses = {k: v / n_batches for k, v in total_losses.items()}

    # Compute alignment metrics
    all_audio_z = torch.cat(all_audio_z, dim=0)
    all_vibration_z = torch.cat(all_vibration_z, dim=0)
    alignment = compute_alignment_metrics(all_audio_z, all_vibration_z)

    avg_losses.update({f'align_{k}': v for k, v in alignment.items()})

    return avg_losses


def train_phase1(
    model: RosetaVAE,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    epochs: int,
    output_dir: Path,
    lr: float = 1e-3,
    beta_kl: float = 1.0,
    lambda_infonce: float = 1.0,
) -> Dict:
    """
    Phase 1: Train on Healthy data.

    Goal: Learn that "healthy audio" and "healthy vibration" map to
    the same z_shared region.
    """
    print("\n" + "=" * 60)
    print("PHASE 1: Training on Healthy Data")
    print("=" * 60)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {
        'train': [],
        'val': [],
    }
    best_val_loss = float('inf')
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")

        train_metrics = train_epoch(
            model, train_loader, optimizer, device,
            beta_kl=beta_kl, lambda_infonce=lambda_infonce,
        )
        val_metrics = validate_epoch(
            model, val_loader, device,
            beta_kl=beta_kl, lambda_infonce=lambda_infonce,
        )

        scheduler.step()

        history['train'].append(train_metrics)
        history['val'].append(val_metrics)

        # Print metrics
        print(f"  Train - Loss: {train_metrics['total']:.4f} "
              f"(recon_a: {train_metrics['recon_audio']:.4f}, "
              f"recon_v: {train_metrics['recon_vibration']:.4f}, "
              f"kl: {train_metrics['kl_total']:.4f}, "
              f"infonce: {train_metrics['infonce']:.4f})")
        print(f"  Val   - Loss: {val_metrics['total']:.4f} "
              f"| Align: cos={val_metrics['align_cosine_similarity']:.4f}, "
              f"acc={val_metrics['align_retrieval_accuracy']:.4f}")

        # Save best model
        if val_metrics['total'] < best_val_loss:
            best_val_loss = val_metrics['total']
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
            }, output_dir / 'best_model.pt')
            print(f"  ✓ New best model saved!")

    # Save final model
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_metrics['total'],
    }, output_dir / 'final_model.pt')

    return {
        'history': history,
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
    }


@torch.no_grad()
def evaluate_on_condition(
    model: RosetaVAE,
    loader: DataLoader,
    device: str,
    condition: str,
) -> Dict[str, float]:
    """Evaluate model on a specific condition."""
    model.eval()

    all_audio_z = []
    all_vibration_z = []
    total_recon_audio = 0
    total_recon_vibration = 0
    n_batches = 0

    for batch in loader:
        audio, vibration, metas, lengths = batch
        audio = audio.to(device)
        vibration = vibration.to(device)
        lengths = lengths.to(device)

        outputs = model(audio, vibration, lengths)

        total_recon_audio += nn.functional.mse_loss(
            outputs['audio_recon'], audio
        ).item()
        total_recon_vibration += nn.functional.mse_loss(
            outputs['vibration_recon'], vibration
        ).item()

        all_audio_z.append(outputs['audio_z_shared'].cpu())
        all_vibration_z.append(outputs['vibration_z_shared'].cpu())
        n_batches += 1

    all_audio_z = torch.cat(all_audio_z, dim=0)
    all_vibration_z = torch.cat(all_vibration_z, dim=0)

    alignment = compute_alignment_metrics(all_audio_z, all_vibration_z)

    return {
        'condition': condition,
        'recon_audio': total_recon_audio / n_batches,
        'recon_vibration': total_recon_vibration / n_batches,
        **alignment,
    }


def evaluate_phase2(
    model: RosetaVAE,
    npz_path: Path,
    device: str,
    conditions: List[str] = ['HH', 'RU', 'RM', 'FB', 'SW', 'VU', 'BR', 'KA'],
    max_frames: int = 100,
) -> Dict[str, Dict]:
    """
    Phase 2: Evaluate on fault conditions.

    Goal: Verify that audio and vibration migrate together when faults occur.
    """
    print("\n" + "=" * 60)
    print("PHASE 2: Evaluating on Fault Conditions")
    print("=" * 60)

    results = {}

    for condition in conditions:
        print(f"\nEvaluating {condition}...")
        loader = create_evaluation_loader(
            npz_path,
            condition=condition,
            batch_size=8,
            strategy='sequence',
            max_frames=max_frames,
        )

        metrics = evaluate_on_condition(model, loader, device, condition)
        results[condition] = metrics

        print(f"  Recon: audio={metrics['recon_audio']:.4f}, "
              f"vibration={metrics['recon_vibration']:.4f}")
        print(f"  Alignment: cos={metrics['cosine_similarity']:.4f}, "
              f"l2={metrics['l2_distance']:.4f}, "
              f"acc={metrics['retrieval_accuracy']:.4f}")

    return results


@torch.no_grad()
def cross_retrieval_phase3(
    model: RosetaVAE,
    npz_path: Path,
    device: str,
    conditions: List[str] = ['HH', 'RU', 'FB'],
    max_frames: int = 100,
) -> Dict[str, Dict]:
    """
    Phase 3: Cross-retrieval test.

    Goal: Given only audio, predict vibration histogram via z_shared.
    """
    print("\n" + "=" * 60)
    print("PHASE 3: Cross-Retrieval Test")
    print("=" * 60)

    model.eval()
    results = {}

    for condition in conditions:
        print(f"\nCross-retrieval for {condition}...")
        loader = create_evaluation_loader(
            npz_path,
            condition=condition,
            batch_size=4,
            strategy='sequence',
            max_frames=max_frames,
        )

        total_cross_mse = 0
        total_pearson = 0
        n_samples = 0

        for batch in loader:
            audio, vibration, metas, lengths = batch
            audio = audio.to(device)
            vibration = vibration.to(device)
            lengths = lengths.to(device)

            # Encode audio only
            audio_enc = model.encode_audio(audio, lengths)

            # Cross-decode: use audio's z_shared to predict vibration
            # We need to sample a z_private for vibration
            z_private_vibration = torch.randn(
                audio_enc['z_shared'].shape[0],
                audio_enc['z_shared'].shape[1],
                model.z_private_dim,
                device=device
            )

            cross_vibration = model.decode_vibration(
                audio_enc['z_shared'],
                z_private_vibration
            )

            # MSE between predicted and actual vibration
            mse = nn.functional.mse_loss(cross_vibration, vibration).item()
            total_cross_mse += mse

            # Pearson correlation (averaged over batch)
            for i in range(audio.size(0)):
                pred = cross_vibration[i].flatten().cpu().numpy()
                real = vibration[i].flatten().cpu().numpy()
                corr = np.corrcoef(pred, real)[0, 1]
                if not np.isnan(corr):
                    total_pearson += corr
                    n_samples += 1

        avg_mse = total_cross_mse / len(loader)
        avg_pearson = total_pearson / max(n_samples, 1)

        results[condition] = {
            'cross_mse': avg_mse,
            'pearson_correlation': avg_pearson,
        }

        print(f"  Cross-reconstruction MSE: {avg_mse:.4f}")
        print(f"  Pearson correlation: {avg_pearson:.4f}")

    return results


# ───────────────────────────────────
# Report Generation
# ───────────────────────────────────

def generate_report(
    output_dir: Path,
    phase1_results: Optional[Dict],
    phase2_results: Optional[Dict],
    phase3_results: Optional[Dict],
    model_params: int,
):
    """Generate markdown report."""
    report_path = output_dir / 'roseta_experiment_report.md'

    with open(report_path, 'w') as f:
        f.write("# Roseta Experiment Report\n\n")
        f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Model Parameters**: {model_params:,}\n\n")

        f.write("---\n\n")

        if phase1_results:
            f.write("## Phase 1: Training on Healthy Data\n\n")
            f.write(f"- Best Epoch: {phase1_results['best_epoch']}\n")
            f.write(f"- Best Val Loss: {phase1_results['best_val_loss']:.4f}\n\n")

            # Training curves summary
            if phase1_results['history']['val']:
                final_val = phase1_results['history']['val'][-1]
                f.write("### Final Validation Metrics\n\n")
                f.write(f"| Metric | Value |\n")
                f.write(f"|--------|-------|\n")
                for k, v in final_val.items():
                    f.write(f"| {k} | {v:.4f} |\n")
                f.write("\n")

        if phase2_results:
            f.write("## Phase 2: Fault Condition Evaluation\n\n")
            f.write("| Condition | Recon Audio | Recon Vibr | Cos Sim | Retrieval Acc |\n")
            f.write("|-----------|-------------|------------|---------|---------------|\n")
            for cond, metrics in phase2_results.items():
                f.write(f"| {cond} | {metrics['recon_audio']:.4f} | "
                        f"{metrics['recon_vibration']:.4f} | "
                        f"{metrics['cosine_similarity']:.4f} | "
                        f"{metrics['retrieval_accuracy']:.4f} |\n")
            f.write("\n")

            # Analysis
            healthy = phase2_results.get('HH', {})
            f.write("### Key Findings\n\n")
            f.write(f"- Healthy baseline cos_sim: {healthy.get('cosine_similarity', 'N/A')}\n")

            # Check if faults maintain alignment
            fault_sims = [v['cosine_similarity'] for k, v in phase2_results.items() if k != 'HH']
            if fault_sims:
                avg_fault_sim = sum(fault_sims) / len(fault_sims)
                f.write(f"- Average fault cos_sim: {avg_fault_sim:.4f}\n")
                f.write(f"- Alignment maintained under faults: ")
                if avg_fault_sim > 0.5:
                    f.write("**YES** ✓\n\n")
                elif avg_fault_sim > 0.3:
                    f.write("**PARTIAL**\n\n")
                else:
                    f.write("**NO** ✗\n\n")

        if phase3_results:
            f.write("## Phase 3: Cross-Retrieval Test\n\n")
            f.write("| Condition | Cross MSE | Pearson Correlation |\n")
            f.write("|-----------|-----------|---------------------|\n")
            for cond, metrics in phase3_results.items():
                f.write(f"| {cond} | {metrics['cross_mse']:.4f} | "
                        f"{metrics['pearson_correlation']:.4f} |\n")
            f.write("\n")

            # Success criteria
            healthy = phase3_results.get('HH', {})
            f.write("### Success Criteria\n\n")
            pearson = healthy.get('pearson_correlation', 0)
            f.write(f"- Target Pearson > 0.7: {pearson:.4f} ")
            if pearson > 0.7:
                f.write("**PASSED** ✓\n")
            elif pearson > 0.5:
                f.write("**PARTIAL**\n")
            else:
                f.write("**FAILED** ✗\n")

        f.write("\n---\n\n")
        f.write("## Conclusion\n\n")
        f.write("*The Roseta Experiment validates that harmonic ratios form a ")
        f.write("cross-modal language when z_shared alignment is achieved.*\n")

    print(f"\n✓ Report saved to {report_path}")


# ───────────────────────────────────
# Main
# ───────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Roseta Experiment - Cross-Modal Harmonic Alignment"
    )
    parser.add_argument(
        '--phase', type=str, default='full',
        choices=['train', 'evaluate', 'cross-retrieval', 'full'],
        help="Experiment phase to run"
    )
    parser.add_argument(
        '--data', type=Path, required=True,
        help="Path to roseta_*.npz dataset"
    )
    parser.add_argument(
        '--output', type=Path, default=Path('data/training_outputs/roseta'),
        help="Output directory"
    )
    parser.add_argument(
        '--checkpoint', type=Path, default=None,
        help="Path to checkpoint (for evaluate/cross-retrieval)"
    )
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--max-frames', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--beta-kl', type=float, default=1.0)
    parser.add_argument('--lambda-infonce', type=float, default=1.0)
    parser.add_argument('--z-shared-dim', type=int, default=32)
    parser.add_argument('--z-private-dim', type=int, default=16)
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument(
        '--all-data', action='store_true',
        help="Train on ALL conditions (not just Healthy). Recommended for more robust training."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Setup
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    args.output.mkdir(parents=True, exist_ok=True)

    print(f"Roseta Experiment")
    print(f"=" * 60)
    print(f"Phase: {args.phase}")
    print(f"Data: {args.data}")
    print(f"Output: {args.output}")
    print(f"Device: {device}")
    print(f"=" * 60)

    # Create model
    model = RosetaVAE(
        input_bins=256,
        input_channels=3,
        hidden_dim=args.hidden_dim,
        z_shared_dim=args.z_shared_dim,
        z_private_dim=args.z_private_dim,
    ).to(device)

    print(f"\nModel parameters: {model.count_parameters():,}")

    # Load checkpoint if provided
    if args.checkpoint and args.checkpoint.exists():
        print(f"Loading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

    phase1_results = None
    phase2_results = None
    phase3_results = None

    # Phase 1: Train
    if args.phase in ['train', 'full']:
        healthy_only = not args.all_data
        train_loader, val_loader = create_roseta_dataloaders(
            npz_path=args.data,
            batch_size=args.batch_size,
            healthy_only=healthy_only,
            strategy='sequence',
            max_frames=args.max_frames,
        )
        if args.all_data:
            print(f"Training on ALL conditions (128 files)")

        phase1_results = train_phase1(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=args.epochs,
            output_dir=args.output,
            lr=args.lr,
            beta_kl=args.beta_kl,
            lambda_infonce=args.lambda_infonce,
        )

        # Load best model for evaluation
        checkpoint = torch.load(args.output / 'best_model.pt', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

    # Phase 2: Evaluate
    if args.phase in ['evaluate', 'full']:
        phase2_results = evaluate_phase2(
            model=model,
            npz_path=args.data,
            device=device,
            max_frames=args.max_frames,
        )

    # Phase 3: Cross-retrieval
    if args.phase in ['cross-retrieval', 'full']:
        phase3_results = cross_retrieval_phase3(
            model=model,
            npz_path=args.data,
            device=device,
            max_frames=args.max_frames,
        )

    # Generate report
    generate_report(
        output_dir=args.output,
        phase1_results=phase1_results,
        phase2_results=phase2_results,
        phase3_results=phase3_results,
        model_params=model.count_parameters(),
    )

    # Save results as JSON
    results = {
        'phase1': phase1_results,
        'phase2': phase2_results,
        'phase3': phase3_results,
        'config': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'max_frames': args.max_frames,
            'lr': args.lr,
            'beta_kl': args.beta_kl,
            'lambda_infonce': args.lambda_infonce,
            'z_shared_dim': args.z_shared_dim,
            'z_private_dim': args.z_private_dim,
        }
    }

    # Convert numpy/tensor values to JSON-serializable
    def to_serializable(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [to_serializable(v) for v in obj]
        return obj

    with open(args.output / 'results.json', 'w') as f:
        json.dump(to_serializable(results), f, indent=2)

    print(f"\n✓ Experiment complete! Results saved to {args.output}")


if __name__ == "__main__":
    main()
