#!/usr/bin/env python3
"""
Freeze Baseline - Rosetta1 2.0 WP1
===================================

Creates a reproducibility checkpoint for the current Rosetta1 model.

This script:
1. Copies the current best checkpoint to artifacts/baseline/
2. Extracts and saves latent representations
3. Saves current metrics
4. Generates a README for reproducibility

Usage:
------
python experiments/freeze_baseline.py \
    --checkpoint models/roseta_vae_best.pt \
    --data data/datasets/roseta_full.npz \
    --output artifacts/baseline
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import yaml
from tqdm import tqdm

from src.RNA.roseta_vae import RosetaVAE, compute_alignment_metrics
from src.datasets.roseta_dataset import RosetaDataset, collate_roseta_sequences
from torch.utils.data import DataLoader


def load_model(checkpoint_path: Path, device: str) -> tuple:
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    config = checkpoint.get('config', {})
    model = RosetaVAE(
        input_bins=config.get('input_bins', 256),
        input_channels=config.get('input_channels', 3),
        hidden_dim=config.get('hidden_dim', 128),
        z_shared_dim=config.get('z_shared_dim', 32),
        z_private_dim=config.get('z_private_dim', 16),
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, checkpoint, config


@torch.no_grad()
def extract_latents(
    model: RosetaVAE,
    dataloader: DataLoader,
    device: str,
) -> Dict[str, np.ndarray]:
    """Extract all latent representations from dataset."""
    all_audio_z_shared = []
    all_audio_z_private = []
    all_vib_z_shared = []
    all_vib_z_private = []
    all_conditions = []
    all_filenames = []

    for batch in tqdm(dataloader, desc="Extracting latents"):
        audio, vibration, metas, lengths = batch
        audio = audio.to(device)
        vibration = vibration.to(device)

        # Encode
        audio_enc = model.encode_audio(audio)
        vib_enc = model.encode_vibration(vibration)

        # Use mean (deterministic) for saved latents
        # Average over time dimension
        audio_z_shared = audio_enc['z_shared_mean'].mean(dim=1).cpu().numpy()
        audio_z_private = audio_enc['z_private_mean'].mean(dim=1).cpu().numpy()
        vib_z_shared = vib_enc['z_shared_mean'].mean(dim=1).cpu().numpy()
        vib_z_private = vib_enc['z_private_mean'].mean(dim=1).cpu().numpy()

        all_audio_z_shared.append(audio_z_shared)
        all_audio_z_private.append(audio_z_private)
        all_vib_z_shared.append(vib_z_shared)
        all_vib_z_private.append(vib_z_private)

        for meta in metas:
            all_conditions.append(meta['condition'])
            all_filenames.append(meta.get('filename', 'unknown'))

    return {
        'audio_z_shared': np.concatenate(all_audio_z_shared, axis=0),
        'audio_z_private': np.concatenate(all_audio_z_private, axis=0),
        'vibration_z_shared': np.concatenate(all_vib_z_shared, axis=0),
        'vibration_z_private': np.concatenate(all_vib_z_private, axis=0),
        'conditions': np.array(all_conditions),
        'filenames': np.array(all_filenames),
    }


@torch.no_grad()
def compute_baseline_metrics(
    model: RosetaVAE,
    dataloader: DataLoader,
    device: str,
) -> Dict[str, float]:
    """Compute baseline metrics for the model."""
    all_audio_z = []
    all_vib_z = []
    total_recon_audio = 0
    total_recon_vib = 0
    total_kl = 0
    total_infonce = 0
    n_batches = 0

    for batch in tqdm(dataloader, desc="Computing metrics"):
        audio, vibration, metas, lengths = batch
        audio = audio.to(device)
        vibration = vibration.to(device)

        outputs = model(audio, vibration, lengths.to(device) if lengths is not None else None)
        losses = model.compute_loss(audio, vibration, outputs, lengths.to(device) if lengths is not None else None)

        total_recon_audio += losses['recon_audio'].item()
        total_recon_vib += losses['recon_vibration'].item()
        total_kl += losses['kl_total'].item()
        total_infonce += losses['infonce'].item()

        # Flatten temporal dimension for alignment metrics
        all_audio_z.append(outputs['audio_z_shared'].view(-1, model.z_shared_dim).cpu())
        all_vib_z.append(outputs['vibration_z_shared'].view(-1, model.z_shared_dim).cpu())
        n_batches += 1

    # Aggregate
    all_audio_z = torch.cat(all_audio_z, dim=0)
    all_vib_z = torch.cat(all_vib_z, dim=0)

    alignment = compute_alignment_metrics(all_audio_z, all_vib_z)

    # z_private statistics
    # Re-compute to get private latents
    z_priv_audio_var = []
    z_priv_vib_var = []
    z_priv_diff = []

    for batch in dataloader:
        audio, vibration, metas, lengths = batch
        audio = audio.to(device)
        vibration = vibration.to(device)

        audio_enc = model.encode_audio(audio)
        vib_enc = model.encode_vibration(vibration)

        z_priv_a = audio_enc['z_private_mean'].mean(dim=1)  # [B, D]
        z_priv_v = vib_enc['z_private_mean'].mean(dim=1)

        z_priv_audio_var.append(z_priv_a.var(dim=0).mean().item())
        z_priv_vib_var.append(z_priv_v.var(dim=0).mean().item())
        z_priv_diff.append(torch.norm(z_priv_a - z_priv_v, dim=-1).mean().item())

    return {
        'recon_audio': total_recon_audio / n_batches,
        'recon_vibration': total_recon_vib / n_batches,
        'kl_total': total_kl / n_batches,
        'infonce': total_infonce / n_batches,
        'total_loss': (total_recon_audio + total_recon_vib + total_kl + total_infonce) / n_batches,
        'cosine_similarity': alignment['cosine_similarity'],
        'l2_distance': alignment['l2_distance'],
        'retrieval_accuracy': alignment['retrieval_accuracy'],
        'z_private_audio_var': float(np.mean(z_priv_audio_var)),
        'z_private_vib_var': float(np.mean(z_priv_vib_var)),
        'z_private_diff': float(np.mean(z_priv_diff)),
    }


def generate_readme(
    output_dir: Path,
    checkpoint_path: Path,
    data_path: Path,
    metrics: Dict,
    config: Dict,
) -> None:
    """Generate reproducibility README."""
    readme = f"""# Rosetta1 Baseline Checkpoint

## Date
{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Source Files
- **Checkpoint**: `{checkpoint_path}`
- **Dataset**: `{data_path}`
- **Config**: `config/rosetta1_baseline.yaml`

## Contents
- `checkpoint.pt` - Model weights and optimizer state
- `latents.npz` - Extracted latent representations for all samples
- `metrics.json` - Evaluation metrics
- `README.md` - This file

## Model Configuration
```yaml
{yaml.dump(config, default_flow_style=False)}
```

## Baseline Metrics

### Losses
| Metric | Value |
|--------|-------|
| Reconstruction Audio | {metrics['recon_audio']:.6f} |
| Reconstruction Vibration | {metrics['recon_vibration']:.6f} |
| KL Divergence | {metrics['kl_total']:.6f} |
| InfoNCE | {metrics['infonce']:.6f} |
| **Total Loss** | **{metrics['total_loss']:.6f}** |

### Alignment (z_shared)
| Metric | Value |
|--------|-------|
| Cosine Similarity | {metrics['cosine_similarity']:.4f} |
| L2 Distance | {metrics['l2_distance']:.4f} |
| Retrieval Accuracy | {metrics['retrieval_accuracy']:.4f} |

### z_private Statistics
| Metric | Value | Target |
|--------|-------|--------|
| Audio Variance | {metrics['z_private_audio_var']:.4f} | > 0.1 |
| Vibration Variance | {metrics['z_private_vib_var']:.4f} | > 0.1 |
| Cross-modal Diff | {metrics['z_private_diff']:.4f} | > 0.5 |

## Reproducibility Command
```bash
python experiments/run_roseta_experiment.py \\
    --phase full \\
    --data {data_path} \\
    --output data/training_outputs/roseta_baseline \\
    --epochs 100 \\
    --batch-size 8 \\
    --all-data
```

## Notes
This baseline was frozen as part of Rosetta1 2.0 (WP1) to ensure
reproducibility and enable fair comparison with subsequent improvements.

Key observations:
- z_private variance: {"OK" if metrics['z_private_audio_var'] > 0.1 else "LOW (collapse suspected)"}
- z_private differentiation: {"OK" if metrics['z_private_diff'] > 0.5 else "LOW (modalities too similar)"}
"""
    with open(output_dir / 'README.md', 'w') as f:
        f.write(readme)


def main():
    parser = argparse.ArgumentParser(description='Freeze Rosetta1 baseline')
    parser.add_argument('--checkpoint', type=Path,
                        default=Path('models/roseta_vae_best.pt'),
                        help='Path to checkpoint')
    parser.add_argument('--data', type=Path,
                        default=Path('data/datasets/roseta_full.npz'),
                        help='Path to dataset')
    parser.add_argument('--output', type=Path,
                        default=Path('artifacts/baseline'),
                        help='Output directory')
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 60)
    print("Rosetta1 Baseline Freeze (WP1)")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Data: {args.data}")
    print(f"Output: {args.output}")
    print(f"Device: {device}")

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Check if checkpoint exists
    if not args.checkpoint.exists():
        print(f"\nWARNING: Checkpoint not found at {args.checkpoint}")
        print("Creating placeholder baseline...")

        # Create placeholder files
        placeholder_metrics = {
            'status': 'placeholder',
            'message': 'No trained model found. Train with run_roseta_experiment.py first.',
            'timestamp': datetime.now().isoformat(),
        }
        with open(args.output / 'metrics.json', 'w') as f:
            json.dump(placeholder_metrics, f, indent=2)

        with open(args.output / 'README.md', 'w') as f:
            f.write("# Baseline Placeholder\n\nNo trained model found. Run training first.\n")

        print(f"Placeholder created at {args.output}")
        return

    # Load model
    print("\nLoading model...")
    model, checkpoint, config = load_model(args.checkpoint, device)
    print(f"  Parameters: {model.count_parameters():,}")

    # Copy checkpoint
    print("\nCopying checkpoint...")
    shutil.copy(args.checkpoint, args.output / 'checkpoint.pt')

    # Load dataset
    print("\nLoading dataset...")
    if not args.data.exists():
        print(f"WARNING: Dataset not found at {args.data}")
        print("Skipping latent extraction and metrics computation.")

        metrics = {'status': 'no_data', 'message': 'Dataset not found'}
        with open(args.output / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)

        generate_readme(args.output, args.checkpoint, args.data, metrics, config)
        return

    dataset = RosetaDataset(
        args.data,
        strategy='sequence',
        max_frames=100,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_roseta_sequences,
    )
    print(f"  Samples: {len(dataset)}")

    # Extract latents
    print("\nExtracting latents...")
    latents = extract_latents(model, dataloader, device)
    np.savez_compressed(
        args.output / 'latents.npz',
        **latents
    )
    print(f"  Saved latents: {list(latents.keys())}")
    print(f"  Audio z_shared shape: {latents['audio_z_shared'].shape}")

    # Compute metrics
    print("\nComputing baseline metrics...")
    metrics = compute_baseline_metrics(model, dataloader, device)

    # Add metadata
    metrics['timestamp'] = datetime.now().isoformat()
    metrics['checkpoint_path'] = str(args.checkpoint)
    metrics['data_path'] = str(args.data)
    metrics['n_samples'] = len(dataset)

    with open(args.output / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    # Generate README
    print("\nGenerating README...")
    generate_readme(args.output, args.checkpoint, args.data, metrics, config)

    # Print summary
    print("\n" + "=" * 60)
    print("Baseline Frozen Successfully!")
    print("=" * 60)
    print(f"\nArtifacts saved to: {args.output}/")
    print(f"  - checkpoint.pt")
    print(f"  - latents.npz")
    print(f"  - metrics.json")
    print(f"  - README.md")

    print(f"\nKey Metrics:")
    print(f"  Cosine Similarity: {metrics['cosine_similarity']:.4f}")
    print(f"  Retrieval Accuracy: {metrics['retrieval_accuracy']:.4f}")
    print(f"  z_private variance (audio): {metrics['z_private_audio_var']:.4f}")
    print(f"  z_private diff: {metrics['z_private_diff']:.4f}")

    # Warnings
    if metrics['z_private_audio_var'] < 0.1:
        print("\n[!] WARNING: z_private variance < 0.1 indicates collapse")
    if metrics['z_private_diff'] < 0.5:
        print("[!] WARNING: z_private diff < 0.5 indicates modalities are too similar")


if __name__ == '__main__':
    main()
