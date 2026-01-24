#!/usr/bin/env python3
"""
Extract Latent Representations from Trained RosetaVAE
======================================================

Runs inference on the full dataset and exports all latent representations
for visualization and analysis.

Output NPZ structure:
- mu_shared_audio: [N, T, 32] - Audio z_shared means
- mu_shared_vib: [N, T, 32] - Vibration z_shared means
- mu_private_audio: [N, T, 16] - Audio z_private means
- mu_private_vib: [N, T, 16] - Vibration z_private means
- logvar_shared_audio: [N, T, 32]
- logvar_shared_vib: [N, T, 32]
- condition: [N] - Condition code (HH, RU, FB, etc.)
- filename: [N] - Original filename
- is_healthy: [N] - Boolean healthy flag
- lengths: [N] - Actual sequence lengths

Usage:
------
python experiments/extract_roseta_latents.py \
    --model data/training_outputs/roseta_full/best_model.pt \
    --data data/datasets/roseta_full.npz \
    --output data/visualizations/roseta_latents.npz
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm

from src.RNA.roseta_vae import RosetaVAE
from src.datasets.roseta_dataset import RosetaDataset, collate_roseta_sequences
from torch.utils.data import DataLoader


def extract_latents(
    model: RosetaVAE,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    """
    Extract all latent representations from model.

    Returns dict with numpy arrays for all latent components.
    """
    model.eval()

    # Storage lists
    mu_shared_audio_all = []
    mu_shared_vib_all = []
    mu_private_audio_all = []
    mu_private_vib_all = []
    logvar_shared_audio_all = []
    logvar_shared_vib_all = []
    conditions = []
    filenames = []
    is_healthy = []
    lengths_all = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting latents"):
            audio, vibration, metas, lengths = batch
            audio = audio.to(device)
            vibration = vibration.to(device)

            # Forward pass to get latents
            outputs = model(audio, vibration, lengths)

            # Extract means and logvars (use means for visualization)
            B = audio.size(0)

            for i in range(B):
                T = lengths[i].item()

                # Slice to actual length
                mu_shared_audio_all.append(
                    outputs['audio_z_shared_mean'][i, :T].cpu().numpy()
                )
                mu_shared_vib_all.append(
                    outputs['vibration_z_shared_mean'][i, :T].cpu().numpy()
                )
                mu_private_audio_all.append(
                    outputs['audio_z_private_mean'][i, :T].cpu().numpy()
                )
                mu_private_vib_all.append(
                    outputs['vibration_z_private_mean'][i, :T].cpu().numpy()
                )
                logvar_shared_audio_all.append(
                    outputs['audio_z_shared_logvar'][i, :T].cpu().numpy()
                )
                logvar_shared_vib_all.append(
                    outputs['vibration_z_shared_logvar'][i, :T].cpu().numpy()
                )

                # Metadata
                conditions.append(metas[i]['condition'])
                filenames.append(metas[i].get('filename', f'file_{i}'))
                is_healthy.append(metas[i]['condition'] == 'HH')
                lengths_all.append(T)

    return {
        'mu_shared_audio': mu_shared_audio_all,
        'mu_shared_vib': mu_shared_vib_all,
        'mu_private_audio': mu_private_audio_all,
        'mu_private_vib': mu_private_vib_all,
        'logvar_shared_audio': logvar_shared_audio_all,
        'logvar_shared_vib': logvar_shared_vib_all,
        'condition': np.array(conditions),
        'filename': np.array(filenames),
        'is_healthy': np.array(is_healthy),
        'lengths': np.array(lengths_all),
    }


def flatten_temporal(latents: Dict) -> Dict[str, np.ndarray]:
    """
    Flatten temporal sequences for frame-level analysis.

    Converts [N_files, T_i, D] -> [N_frames, D] with tracking arrays.
    """
    mu_shared_audio_flat = []
    mu_shared_vib_flat = []
    file_idx_flat = []
    frame_idx_flat = []
    condition_flat = []
    is_healthy_flat = []

    for i, (mu_a, mu_v) in enumerate(zip(
        latents['mu_shared_audio'],
        latents['mu_shared_vib']
    )):
        T = mu_a.shape[0]
        for t in range(T):
            mu_shared_audio_flat.append(mu_a[t])
            mu_shared_vib_flat.append(mu_v[t])
            file_idx_flat.append(i)
            frame_idx_flat.append(t)
            condition_flat.append(latents['condition'][i])
            is_healthy_flat.append(latents['is_healthy'][i])

    return {
        'mu_shared_audio': np.array(mu_shared_audio_flat),
        'mu_shared_vib': np.array(mu_shared_vib_flat),
        'file_idx': np.array(file_idx_flat),
        'frame_idx': np.array(frame_idx_flat),
        'condition': np.array(condition_flat),
        'is_healthy': np.array(is_healthy_flat),
    }


def main():
    parser = argparse.ArgumentParser(description='Extract RosetaVAE latents')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model .pt file')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to roseta dataset .npz')
    parser.add_argument('--output', type=str, required=True,
                        help='Output path for latents .npz')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--max-frames', type=int, default=100)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load model
    print(f"\nLoading model from {args.model}")
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)

    # Initialize model with same config (use defaults if not saved)
    model_config = checkpoint.get('config', {})
    model = RosetaVAE(
        input_bins=model_config.get('input_bins', 256),
        input_channels=model_config.get('input_channels', 3),
        hidden_dim=model_config.get('hidden_dim', 128),
        z_shared_dim=model_config.get('z_shared_dim', 32),
        z_private_dim=model_config.get('z_private_dim', 16),
    ).to(device)

    # Load weights
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)

    print(f"Model loaded: {model.count_parameters():,} parameters")

    # Load full dataset (all conditions)
    print(f"\nLoading dataset from {args.data}")
    dataset = RosetaDataset(
        npz_path=args.data,
        healthy_only=False,  # All conditions
        strategy='sequence',
        max_frames=args.max_frames,
        frame_sample_mode='start',  # Deterministic
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_roseta_sequences,
        num_workers=0,
    )

    stats = dataset.get_stats()
    print(f"Dataset: {stats['n_files']} files, {stats['total_frames']} frames")
    print(f"Conditions: {stats['conditions']}")

    # Extract latents
    print("\nExtracting latent representations...")
    latents = extract_latents(model, dataloader, device)

    # Also create flattened version for frame-level analysis
    latents_flat = flatten_temporal(latents)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save hierarchical (per-file) latents
    np.savez_compressed(
        output_path,
        # Per-file arrays (variable length, stored as object arrays)
        mu_shared_audio=np.array(latents['mu_shared_audio'], dtype=object),
        mu_shared_vib=np.array(latents['mu_shared_vib'], dtype=object),
        mu_private_audio=np.array(latents['mu_private_audio'], dtype=object),
        mu_private_vib=np.array(latents['mu_private_vib'], dtype=object),
        logvar_shared_audio=np.array(latents['logvar_shared_audio'], dtype=object),
        logvar_shared_vib=np.array(latents['logvar_shared_vib'], dtype=object),
        # Metadata
        condition=latents['condition'],
        filename=latents['filename'],
        is_healthy=latents['is_healthy'],
        lengths=latents['lengths'],
        # Flattened arrays for easy visualization
        flat_mu_shared_audio=latents_flat['mu_shared_audio'],
        flat_mu_shared_vib=latents_flat['mu_shared_vib'],
        flat_file_idx=latents_flat['file_idx'],
        flat_frame_idx=latents_flat['frame_idx'],
        flat_condition=latents_flat['condition'],
        flat_is_healthy=latents_flat['is_healthy'],
    )

    print(f"\n✔ Latents saved to {output_path}")
    print(f"  Files: {len(latents['condition'])}")
    print(f"  Total frames: {len(latents_flat['condition'])}")
    print(f"  z_shared dim: {latents_flat['mu_shared_audio'].shape[1]}")

    # Print condition breakdown
    print("\nCondition breakdown:")
    for cond in np.unique(latents['condition']):
        count = (latents['condition'] == cond).sum()
        print(f"  {cond}: {count} files")


if __name__ == '__main__':
    main()
