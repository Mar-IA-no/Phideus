#!/usr/bin/env python3
"""
Ablation Studies for RosetaVAE - WP6
=====================================

Systematic ablation to identify where the novelty lies:

Conditions:
- A (proposed): ratio-hist + InfoNCE + z_private fix
- B: ratio-hist + NO InfoNCE + z_private fix
- C: raw PSD + InfoNCE + z_private fix
- D: ratio-hist (no aux channels) + InfoNCE + z_private fix

Usage:
------
python experiments/run_ablations.py \
    --data data/datasets/roseta_full.npz \
    --output data/ablations \
    --epochs 50
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.RNA.roseta_vae import RosetaVAE, compute_alignment_metrics
from src.datasets.roseta_dataset import (
    RosetaDataset, create_roseta_dataloaders, collate_roseta_sequences
)


# Ablation conditions
ABLATION_CONDITIONS = {
    'A_proposed': {
        'description': 'Full model: ratio-hist + InfoNCE + z_private fix',
        'lambda_infonce': 1.0,
        'beta_kl_private': 0.01,
        'dropout_shared': 0.5,
        'lambda_diff': 0.1,
        'input_channels': 3,  # ratio + energy + entropy
    },
    'B_no_infonce': {
        'description': 'No InfoNCE: ratio-hist + z_private fix',
        'lambda_infonce': 0.0,  # Disabled
        'beta_kl_private': 0.01,
        'dropout_shared': 0.5,
        'lambda_diff': 0.1,
        'input_channels': 3,
    },
    'C_raw_psd': {
        'description': 'Raw PSD instead of ratio histogram',
        'lambda_infonce': 1.0,
        'beta_kl_private': 0.01,
        'dropout_shared': 0.5,
        'lambda_diff': 0.1,
        'input_channels': 1,  # Only raw PSD, no aux
        'use_raw_psd': True,
    },
    'D_no_aux': {
        'description': 'Ratio histogram without auxiliary channels (energy/entropy)',
        'lambda_infonce': 1.0,
        'beta_kl_private': 0.01,
        'dropout_shared': 0.5,
        'lambda_diff': 0.1,
        'input_channels': 1,  # Only ratio channel
    },
}


def train_epoch(
    model: RosetaVAE,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    device: str,
    config: Dict,
) -> Dict[str, float]:
    """Train for one epoch with ablation config."""
    model.train()
    total_losses = {'total': 0, 'recon': 0, 'kl': 0, 'infonce': 0}
    n_batches = 0

    for batch in train_loader:
        audio, vibration, metas, lengths = batch
        audio = audio.to(device)
        vibration = vibration.to(device)
        lengths = lengths.to(device)

        # Handle channel reduction for ablations C and D
        if config.get('input_channels', 3) < 3:
            audio = audio[..., :config['input_channels']]
            vibration = vibration[..., :config['input_channels']]

        optimizer.zero_grad()

        outputs = model(audio, vibration, lengths,
                        dropout_shared=config.get('dropout_shared', 0.0))

        losses = model.compute_loss(
            audio, vibration, outputs, lengths,
            beta_kl_shared=1.0,
            beta_kl_private=config.get('beta_kl_private', 0.01),
            lambda_infonce=config.get('lambda_infonce', 1.0),
            lambda_diff=config.get('lambda_diff', 0.1),
        )

        losses['total'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_losses['total'] += losses['total'].item()
        total_losses['recon'] += (losses['recon_audio'].item() +
                                   losses['recon_vibration'].item())
        total_losses['kl'] += losses['kl_total'].item()
        total_losses['infonce'] += losses['infonce'].item()
        n_batches += 1

    return {k: v / n_batches for k, v in total_losses.items()}


@torch.no_grad()
def evaluate_model(
    model: RosetaVAE,
    val_loader: DataLoader,
    device: str,
    config: Dict,
) -> Dict[str, float]:
    """Evaluate model with ablation config."""
    model.eval()
    total_losses = {'total': 0, 'recon': 0, 'kl': 0, 'infonce': 0}
    all_audio_z = []
    all_vib_z = []
    n_batches = 0

    for batch in val_loader:
        audio, vibration, metas, lengths = batch
        audio = audio.to(device)
        vibration = vibration.to(device)
        lengths = lengths.to(device)

        if config.get('input_channels', 3) < 3:
            audio = audio[..., :config['input_channels']]
            vibration = vibration[..., :config['input_channels']]

        outputs = model(audio, vibration, lengths, dropout_shared=0.0)
        losses = model.compute_loss(
            audio, vibration, outputs, lengths,
            beta_kl_shared=1.0,
            beta_kl_private=config.get('beta_kl_private', 0.01),
            lambda_infonce=config.get('lambda_infonce', 1.0),
            lambda_diff=config.get('lambda_diff', 0.1),
        )

        total_losses['total'] += losses['total'].item()
        total_losses['recon'] += (losses['recon_audio'].item() +
                                   losses['recon_vibration'].item())
        total_losses['kl'] += losses['kl_total'].item()
        total_losses['infonce'] += losses['infonce'].item()

        all_audio_z.append(outputs['audio_z_shared'].cpu())
        all_vib_z.append(outputs['vibration_z_shared'].cpu())
        n_batches += 1

    avg_losses = {k: v / n_batches for k, v in total_losses.items()}

    # Alignment metrics
    all_audio_z = torch.cat(all_audio_z, dim=0)
    all_vib_z = torch.cat(all_vib_z, dim=0)
    alignment = compute_alignment_metrics(all_audio_z, all_vib_z)

    avg_losses.update({f'align_{k}': v for k, v in alignment.items()})

    return avg_losses


def run_ablation(
    condition_name: str,
    config: Dict,
    data_path: Path,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    device: str,
) -> Dict:
    """Run a single ablation condition."""
    print(f"\n{'=' * 60}")
    print(f"Running: {condition_name}")
    print(f"Config: {config['description']}")
    print(f"{'=' * 60}")

    # Create model with appropriate input channels
    input_channels = config.get('input_channels', 3)
    model = RosetaVAE(
        input_bins=256,
        input_channels=input_channels,
        hidden_dim=128,
        z_shared_dim=32,
        z_private_dim=16,
    ).to(device)

    print(f"Model parameters: {model.count_parameters():,}")
    print(f"Input channels: {input_channels}")

    # Create dataloaders
    train_loader, val_loader, test_loader = create_roseta_dataloaders(
        npz_path=data_path,
        batch_size=batch_size,
        healthy_only=False,
        strategy='sequence',
        max_frames=100,
        return_test=True,
    )

    # Training
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {'train': [], 'val': []}
    best_val_loss = float('inf')

    for epoch in range(1, epochs + 1):
        train_metrics = train_epoch(model, train_loader, optimizer, device, config)
        val_metrics = evaluate_model(model, val_loader, device, config)

        scheduler.step()

        history['train'].append(train_metrics)
        history['val'].append(val_metrics)

        if epoch % 10 == 0 or epoch == epochs:
            print(f"  Epoch {epoch}: train_loss={train_metrics['total']:.4f}, "
                  f"val_loss={val_metrics['total']:.4f}, "
                  f"cos_sim={val_metrics['align_cosine_similarity']:.4f}")

        if val_metrics['total'] < best_val_loss:
            best_val_loss = val_metrics['total']
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': config,
                'epoch': epoch,
            }, output_dir / f'{condition_name}_best.pt')

    # Final evaluation on test set
    test_metrics = evaluate_model(model, test_loader, device, config) if test_loader else None

    return {
        'condition': condition_name,
        'config': config,
        'best_val_loss': best_val_loss,
        'final_val_metrics': history['val'][-1],
        'test_metrics': test_metrics,
        'history': history,
    }


def generate_report(results: Dict, output_path: Path):
    """Generate ablation study report."""

    report = f"""# Ablation Study Report (WP6)

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Conditions Tested

| ID | Description |
|----|-------------|
"""
    for name, config in ABLATION_CONDITIONS.items():
        report += f"| {name} | {config['description']} |\n"

    report += """
---

## Results Summary

| Condition | Val Loss | Cos Sim | Retrieval Acc | InfoNCE |
|-----------|----------|---------|---------------|---------|
"""
    for name, res in results.items():
        if isinstance(res, dict) and 'final_val_metrics' in res:
            val = res['final_val_metrics']
            report += f"| {name} | {res['best_val_loss']:.4f} | "
            report += f"{val.get('align_cosine_similarity', 0):.4f} | "
            report += f"{val.get('align_retrieval_accuracy', 0):.4f} | "
            report += f"{val.get('infonce', 0):.4f} |\n"

    report += """
---

## Analysis

### Key Questions

1. **Does InfoNCE matter?** (A vs B)
"""
    if 'A_proposed' in results and 'B_no_infonce' in results:
        a_cos = results['A_proposed']['final_val_metrics'].get('align_cosine_similarity', 0)
        b_cos = results['B_no_infonce']['final_val_metrics'].get('align_cosine_similarity', 0)
        if a_cos > b_cos + 0.1:
            report += f"   YES: A ({a_cos:.4f}) >> B ({b_cos:.4f}) - InfoNCE is critical\n"
        else:
            report += f"   MARGINAL: A ({a_cos:.4f}) ≈ B ({b_cos:.4f}) - InfoNCE may not be key\n"

    report += """
2. **Does ratio-histogram matter?** (A vs C)
"""
    if 'A_proposed' in results and 'C_raw_psd' in results:
        a_cos = results['A_proposed']['final_val_metrics'].get('align_cosine_similarity', 0)
        c_cos = results['C_raw_psd']['final_val_metrics'].get('align_cosine_similarity', 0)
        if a_cos > c_cos + 0.1:
            report += f"   YES: A ({a_cos:.4f}) >> C ({c_cos:.4f}) - ratio-hist representation adds value\n"
        elif c_cos > a_cos + 0.1:
            report += f"   NO: C ({c_cos:.4f}) > A ({a_cos:.4f}) - raw PSD is sufficient (problem!)\n"
        else:
            report += f"   MARGINAL: A ({a_cos:.4f}) ≈ C ({c_cos:.4f}) - representation may not matter\n"

    report += """
3. **Do auxiliary channels (energy/entropy) matter?** (A vs D)
"""
    if 'A_proposed' in results and 'D_no_aux' in results:
        a_cos = results['A_proposed']['final_val_metrics'].get('align_cosine_similarity', 0)
        d_cos = results['D_no_aux']['final_val_metrics'].get('align_cosine_similarity', 0)
        if a_cos > d_cos + 0.05:
            report += f"   YES: A ({a_cos:.4f}) > D ({d_cos:.4f}) - aux channels help\n"
        else:
            report += f"   MARGINAL: A ({a_cos:.4f}) ≈ D ({d_cos:.4f}) - aux channels may be redundant\n"

    report += """
---

## Conclusions

"""
    # Determine where the novelty lies
    if 'A_proposed' in results and 'B_no_infonce' in results:
        a_cos = results['A_proposed']['final_val_metrics'].get('align_cosine_similarity', 0)
        b_cos = results['B_no_infonce']['final_val_metrics'].get('align_cosine_similarity', 0)
        if a_cos > b_cos + 0.1:
            report += "- **Primary novelty**: InfoNCE contrastive loss for cross-modal alignment\n"

    if 'A_proposed' in results and 'C_raw_psd' in results:
        a_cos = results['A_proposed']['final_val_metrics'].get('align_cosine_similarity', 0)
        c_cos = results['C_raw_psd']['final_val_metrics'].get('align_cosine_similarity', 0)
        if a_cos > c_cos + 0.1:
            report += "- **Secondary novelty**: Ratio-histogram representation\n"

    report += """
---

*Generated by run_ablations.py (WP6)*
"""

    with open(output_path / 'REPORT_ABLATIONS.md', 'w') as f:
        f.write(report)

    print(f"Report saved to {output_path / 'REPORT_ABLATIONS.md'}")


def main():
    parser = argparse.ArgumentParser(description='Run ablation studies')
    parser.add_argument('--data', type=Path, required=True,
                        help='Path to roseta dataset')
    parser.add_argument('--output', type=Path,
                        default=Path('data/ablations'),
                        help='Output directory')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Epochs per condition')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--conditions', type=str, nargs='+',
                        default=list(ABLATION_CONDITIONS.keys()),
                        help='Conditions to run')
    parser.add_argument('--device', type=str, default='auto')
    args = parser.parse_args()

    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    args.output.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Ablation Studies (WP6)")
    print("=" * 60)
    print(f"Data: {args.data}")
    print(f"Output: {args.output}")
    print(f"Device: {device}")
    print(f"Epochs per condition: {args.epochs}")
    print(f"Conditions: {args.conditions}")

    results = {}

    for condition_name in args.conditions:
        if condition_name not in ABLATION_CONDITIONS:
            print(f"Unknown condition: {condition_name}, skipping...")
            continue

        config = ABLATION_CONDITIONS[condition_name]

        result = run_ablation(
            condition_name=condition_name,
            config=config,
            data_path=args.data,
            output_dir=args.output,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=device,
        )

        results[condition_name] = result

        # Save intermediate results
        with open(args.output / 'ablation_results.json', 'w') as f:
            # Convert numpy/tensor to serializable
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

            json.dump(to_serializable(results), f, indent=2)

    # Generate report
    generate_report(results, args.output)

    # Print summary
    print("\n" + "=" * 60)
    print("ABLATION SUMMARY")
    print("=" * 60)

    print(f"\n{'Condition':<20} {'Val Loss':<12} {'Cos Sim':<12}")
    print("-" * 44)

    for name, res in results.items():
        if isinstance(res, dict) and 'final_val_metrics' in res:
            val = res['final_val_metrics']
            print(f"{name:<20} {res['best_val_loss']:<12.4f} "
                  f"{val.get('align_cosine_similarity', 0):<12.4f}")


if __name__ == '__main__':
    main()
