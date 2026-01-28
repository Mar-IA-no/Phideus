#!/usr/bin/env python3
"""
Cross-Modal Reconstruction Evaluation for RosetaVAE
====================================================

The "Rosetta Stone Test": Can we reconstruct histograms from one modality
using ONLY the z_shared from the other modality?

This script evaluates:
1. Cross-reconstruction quality (audio→vib, vib→audio)
2. Cycle consistency (audio→vib→audio)
3. Retrieval accuracy (find matching pair in batch)

These metrics address the key criticism: "alignment ≠ cross-modality"

Usage:
------
python experiments/evaluate_cross_reconstruction.py \
    --model models/roseta_vae_best.pt \
    --data data/datasets/roseta_full.npz \
    --output data/evaluations/cross_reconstruction
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import jensenshannon
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.RNA.roseta_vae import RosetaVAE, compute_alignment_metrics
from src.datasets.roseta_dataset import RosetaDataset, create_roseta_dataloaders, collate_roseta_sequences


def load_model(model_path: str, device: str) -> RosetaVAE:
    """Load trained RosetaVAE model."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Get config from checkpoint or use defaults
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

    print(f"Loaded model from {model_path}")
    print(f"  z_shared_dim: {model.z_shared_dim}")
    print(f"  z_private_dim: {model.z_private_dim}")

    return model


def cross_reconstruct(
    model: RosetaVAE,
    audio: torch.Tensor,
    vibration: torch.Tensor,
    use_zero_private: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Perform cross-modal reconstruction.

    Args:
        model: trained RosetaVAE
        audio: [B, T, bins, channels]
        vibration: [B, T, bins, channels]
        use_zero_private: if True, use zeros for z_private (pure z_shared test)

    Returns:
        dict with reconstructions and latents
    """
    with torch.no_grad():
        # Encode both modalities
        audio_enc = model.encode_audio(audio)
        vib_enc = model.encode_vibration(vibration)

        # Get z_shared (using mean, not sampled, for deterministic eval)
        z_shared_audio = audio_enc['z_shared_mean']
        z_shared_vib = vib_enc['z_shared_mean']

        B, T, _ = z_shared_audio.shape

        if use_zero_private:
            # Pure z_shared test: use zeros for z_private
            z_private_zeros = torch.zeros(B, T, model.z_private_dim, device=audio.device)
            z_private_for_audio = z_private_zeros
            z_private_for_vib = z_private_zeros
        else:
            # Use the target's z_private (easier task)
            z_private_for_audio = audio_enc['z_private_mean']
            z_private_for_vib = vib_enc['z_private_mean']

        # Self-reconstruction (baseline)
        audio_self_recon = model.decode_audio(z_shared_audio, z_private_for_audio)
        vib_self_recon = model.decode_vibration(z_shared_vib, z_private_for_vib)

        # Cross-reconstruction: Audio's z_shared → Vibration decoder
        vib_from_audio = model.decode_vibration(z_shared_audio, z_private_for_vib)

        # Cross-reconstruction: Vibration's z_shared → Audio decoder
        audio_from_vib = model.decode_audio(z_shared_vib, z_private_for_audio)

        # Cycle consistency: audio → vib_pred → audio_recon
        # First: audio → z_shared_audio → vib_pred
        # Then: vib_pred → z_shared_vib_pred → audio_cycle
        vib_pred_enc = model.encode_vibration(vib_from_audio)
        z_shared_vib_pred = vib_pred_enc['z_shared_mean']
        audio_cycle = model.decode_audio(z_shared_vib_pred, z_private_for_audio)

    return {
        # Self-reconstructions
        'audio_self_recon': audio_self_recon,
        'vib_self_recon': vib_self_recon,
        # Cross-reconstructions
        'vib_from_audio': vib_from_audio,
        'audio_from_vib': audio_from_vib,
        # Cycle reconstruction
        'audio_cycle': audio_cycle,
        # Latents
        'z_shared_audio': z_shared_audio,
        'z_shared_vib': z_shared_vib,
    }


def histogram_metrics(pred: np.ndarray, real: np.ndarray) -> Dict[str, float]:
    """
    Compute metrics between predicted and real histograms.

    Args:
        pred: predicted histogram [T, bins, channels] or [bins, channels]
        real: real histogram [T, bins, channels] or [bins, channels]

    Returns:
        dict with metrics
    """
    # Flatten to 1D for correlation
    pred_flat = pred.flatten()
    real_flat = real.flatten()

    # MSE
    mse = np.mean((pred_flat - real_flat) ** 2)

    # MAE
    mae = np.mean(np.abs(pred_flat - real_flat))

    # Pearson correlation
    if np.std(pred_flat) > 1e-8 and np.std(real_flat) > 1e-8:
        pearson_corr, _ = pearsonr(pred_flat, real_flat)
    else:
        pearson_corr = 0.0

    # Spearman correlation
    if np.std(pred_flat) > 1e-8 and np.std(real_flat) > 1e-8:
        spearman_corr, _ = spearmanr(pred_flat, real_flat)
    else:
        spearman_corr = 0.0

    # Per-channel Jensen-Shannon divergence (for probability distributions)
    js_divergences = []
    if pred.ndim == 3:
        T, bins, channels = pred.shape
        for c in range(channels):
            for t in range(T):
                p = pred[t, :, c] + 1e-10
                q = real[t, :, c] + 1e-10
                p = p / p.sum()
                q = q / q.sum()
                js_divergences.append(jensenshannon(p, q))
    else:
        bins, channels = pred.shape
        for c in range(channels):
            p = pred[:, c] + 1e-10
            q = real[:, c] + 1e-10
            p = p / p.sum()
            q = q / q.sum()
            js_divergences.append(jensenshannon(p, q))

    js_div = np.mean(js_divergences)

    return {
        'mse': float(mse),
        'mae': float(mae),
        'pearson_corr': float(pearson_corr),
        'spearman_corr': float(spearman_corr),
        'js_divergence': float(js_div),
    }


def retrieval_accuracy(
    z_audio: torch.Tensor,
    z_vib: torch.Tensor,
    k_values: List[int] = [1, 5, 10],
) -> Dict[str, float]:
    """
    Compute retrieval accuracy: given z_audio, find correct z_vib in batch.

    Args:
        z_audio: [N, D] audio embeddings
        z_vib: [N, D] vibration embeddings
        k_values: list of k for top-k accuracy

    Returns:
        dict with top-k accuracies
    """
    # Normalize
    z_a = F.normalize(z_audio, dim=-1)
    z_v = F.normalize(z_vib, dim=-1)

    # Similarity matrix
    sim = torch.matmul(z_a, z_v.T)  # [N, N]

    N = z_a.size(0)
    labels = torch.arange(N, device=z_a.device)

    results = {}

    # Top-k accuracy
    for k in k_values:
        if k > N:
            k = N
        _, topk_indices = sim.topk(k, dim=-1)
        correct = (topk_indices == labels.unsqueeze(1)).any(dim=-1)
        acc = correct.float().mean().item()
        results[f'top{k}_accuracy'] = acc

    # Mean Reciprocal Rank
    sorted_indices = sim.argsort(dim=-1, descending=True)
    ranks = (sorted_indices == labels.unsqueeze(1)).nonzero(as_tuple=True)[1] + 1
    mrr = (1.0 / ranks.float()).mean().item()
    results['mrr'] = mrr

    return results


def evaluate_dataset(
    model: RosetaVAE,
    dataloader: DataLoader,
    device: str,
    use_zero_private: bool = True,
) -> Dict[str, any]:
    """
    Evaluate cross-reconstruction on entire dataset.
    """
    all_metrics = {
        'self_recon_audio': [],
        'self_recon_vib': [],
        'cross_audio_to_vib': [],
        'cross_vib_to_audio': [],
        'cycle_audio': [],
    }

    all_z_audio = []
    all_z_vib = []

    print(f"\nEvaluating cross-reconstruction (zero_private={use_zero_private})...")

    for batch in tqdm(dataloader, desc="Processing batches"):
        # Unpack batch: (audio, vibration, metadata, lengths) for sequences
        if len(batch) == 4:
            audio, vibration, metas, lengths = batch
        else:
            audio, vibration, metas = batch
            lengths = None

        audio = audio.to(device)
        vibration = vibration.to(device)

        results = cross_reconstruct(model, audio, vibration, use_zero_private)

        # Convert to numpy
        audio_np = audio.cpu().numpy()
        vib_np = vibration.cpu().numpy()

        audio_self = results['audio_self_recon'].cpu().numpy()
        vib_self = results['vib_self_recon'].cpu().numpy()
        vib_from_audio = results['vib_from_audio'].cpu().numpy()
        audio_from_vib = results['audio_from_vib'].cpu().numpy()
        audio_cycle = results['audio_cycle'].cpu().numpy()

        # Compute metrics per sample
        B = audio_np.shape[0]
        for i in range(B):
            all_metrics['self_recon_audio'].append(
                histogram_metrics(audio_self[i], audio_np[i])
            )
            all_metrics['self_recon_vib'].append(
                histogram_metrics(vib_self[i], vib_np[i])
            )
            all_metrics['cross_audio_to_vib'].append(
                histogram_metrics(vib_from_audio[i], vib_np[i])
            )
            all_metrics['cross_vib_to_audio'].append(
                histogram_metrics(audio_from_vib[i], audio_np[i])
            )
            all_metrics['cycle_audio'].append(
                histogram_metrics(audio_cycle[i], audio_np[i])
            )

        # Collect z_shared for retrieval
        z_a = results['z_shared_audio'].view(-1, model.z_shared_dim)
        z_v = results['z_shared_vib'].view(-1, model.z_shared_dim)
        all_z_audio.append(z_a.cpu())
        all_z_vib.append(z_v.cpu())

    # Aggregate metrics
    aggregated = {}
    for key, metrics_list in all_metrics.items():
        aggregated[key] = {}
        for metric_name in metrics_list[0].keys():
            values = [m[metric_name] for m in metrics_list]
            aggregated[key][metric_name] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
            }

    # Retrieval accuracy
    z_audio_all = torch.cat(all_z_audio, dim=0)
    z_vib_all = torch.cat(all_z_vib, dim=0)

    # Sample for retrieval (full matrix too large)
    N = min(1000, z_audio_all.size(0))
    idx = torch.randperm(z_audio_all.size(0))[:N]
    retrieval = retrieval_accuracy(z_audio_all[idx], z_vib_all[idx])

    return {
        'reconstruction_metrics': aggregated,
        'retrieval_metrics': retrieval,
        'n_samples': len(all_metrics['self_recon_audio']),
        'n_frames_retrieval': N,
    }


def plot_results(results: Dict, output_path: Path):
    """Generate visualization of results."""

    # Extract key metrics
    metrics_names = ['pearson_corr', 'mse', 'js_divergence']
    categories = ['self_recon_audio', 'self_recon_vib', 'cross_audio_to_vib',
                  'cross_vib_to_audio', 'cycle_audio']
    labels = ['Self-Audio', 'Self-Vib', 'Audio→Vib', 'Vib→Audio', 'Cycle']

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, metric in zip(axes, metrics_names):
        means = [results['reconstruction_metrics'][cat][metric]['mean'] for cat in categories]
        stds = [results['reconstruction_metrics'][cat][metric]['std'] for cat in categories]

        colors = ['#2ecc71', '#2ecc71', '#e74c3c', '#e74c3c', '#f39c12']
        bars = ax.bar(labels, means, yerr=stds, capsize=5, color=colors, edgecolor='black')

        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'{metric.replace("_", " ").title()}')
        ax.tick_params(axis='x', rotation=45)

        # Add value labels
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{mean:.3f}', ha='center', va='bottom', fontsize=9)

    plt.suptitle('Cross-Reconstruction Evaluation: The Rosetta Stone Test', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path / 'cross_reconstruction_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Retrieval plot
    fig, ax = plt.subplots(figsize=(8, 5))
    retrieval = results['retrieval_metrics']
    metrics = ['top1_accuracy', 'top5_accuracy', 'top10_accuracy', 'mrr']
    values = [retrieval.get(m, 0) for m in metrics]
    labels = ['Top-1', 'Top-5', 'Top-10', 'MRR']

    bars = ax.bar(labels, values, color=['#3498db', '#3498db', '#3498db', '#9b59b6'],
                  edgecolor='black')
    ax.set_ylabel('Accuracy / Score')
    ax.set_title(f'Retrieval Accuracy (N={results["n_frames_retrieval"]})')
    ax.set_ylim(0, 1)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
               f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Add random baseline
    n = results['n_frames_retrieval']
    ax.axhline(y=1/n, color='red', linestyle='--', label=f'Random (1/{n})')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path / 'retrieval_accuracy.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved plots to {output_path}")


def generate_report(results: Dict, output_path: Path, use_zero_private: bool):
    """Generate markdown report."""

    recon = results['reconstruction_metrics']
    retrieval = results['retrieval_metrics']

    report = f"""# Cross-Reconstruction Evaluation Report

## Configuration
- **z_private mode**: {'zeros (pure z_shared test)' if use_zero_private else 'target z_private'}
- **Total samples**: {results['n_samples']}
- **Retrieval batch size**: {results['n_frames_retrieval']}

---

## 1. Reconstruction Quality

### Self-Reconstruction (Baseline)

| Metric | Audio | Vibration |
|--------|-------|-----------|
| Pearson Corr | {recon['self_recon_audio']['pearson_corr']['mean']:.4f} ± {recon['self_recon_audio']['pearson_corr']['std']:.4f} | {recon['self_recon_vib']['pearson_corr']['mean']:.4f} ± {recon['self_recon_vib']['pearson_corr']['std']:.4f} |
| MSE | {recon['self_recon_audio']['mse']['mean']:.6f} | {recon['self_recon_vib']['mse']['mean']:.6f} |
| JS Divergence | {recon['self_recon_audio']['js_divergence']['mean']:.4f} | {recon['self_recon_vib']['js_divergence']['mean']:.4f} |

### Cross-Reconstruction (THE TEST)

| Metric | Audio→Vib | Vib→Audio |
|--------|-----------|-----------|
| **Pearson Corr** | **{recon['cross_audio_to_vib']['pearson_corr']['mean']:.4f}** ± {recon['cross_audio_to_vib']['pearson_corr']['std']:.4f} | **{recon['cross_vib_to_audio']['pearson_corr']['mean']:.4f}** ± {recon['cross_vib_to_audio']['pearson_corr']['std']:.4f} |
| MSE | {recon['cross_audio_to_vib']['mse']['mean']:.6f} | {recon['cross_vib_to_audio']['mse']['mean']:.6f} |
| JS Divergence | {recon['cross_audio_to_vib']['js_divergence']['mean']:.4f} | {recon['cross_vib_to_audio']['js_divergence']['mean']:.4f} |

### Cycle Consistency (Audio → Vib → Audio)

| Metric | Value |
|--------|-------|
| Pearson Corr | {recon['cycle_audio']['pearson_corr']['mean']:.4f} ± {recon['cycle_audio']['pearson_corr']['std']:.4f} |
| MSE | {recon['cycle_audio']['mse']['mean']:.6f} |
| JS Divergence | {recon['cycle_audio']['js_divergence']['mean']:.4f} |

---

## 2. Retrieval Accuracy

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Top-1 Accuracy** | **{retrieval['top1_accuracy']:.4f}** | Can find exact match |
| Top-5 Accuracy | {retrieval['top5_accuracy']:.4f} | Match in top 5 |
| Top-10 Accuracy | {retrieval.get('top10_accuracy', 0):.4f} | Match in top 10 |
| MRR | {retrieval['mrr']:.4f} | Mean Reciprocal Rank |
| Random Baseline | {1/results['n_frames_retrieval']:.6f} | (1/N) |

---

## 3. Interpretation

### Cross-Reconstruction Quality
"""
    # Add interpretation based on results
    cross_corr_avg = (recon['cross_audio_to_vib']['pearson_corr']['mean'] +
                      recon['cross_vib_to_audio']['pearson_corr']['mean']) / 2

    if cross_corr_avg > 0.7:
        report += """
**STRONG**: Cross-reconstruction correlation > 0.7 indicates that z_shared
contains transferable information. The model has learned a genuine cross-modal
representation, not just alignment.
"""
    elif cross_corr_avg > 0.5:
        report += """
**MODERATE**: Cross-reconstruction correlation 0.5-0.7 indicates partial transfer.
z_shared captures some cross-modal structure, but significant information is
lost during translation.
"""
    else:
        report += """
**WEAK**: Cross-reconstruction correlation < 0.5 indicates that z_shared may not
contain sufficient transferable information. The alignment observed in embedding
space may not translate to actual reconstruction capability.
"""

    report += f"""
### Retrieval Quality
"""
    if retrieval['top1_accuracy'] > 0.5:
        report += """
**STRONG**: Top-1 accuracy > 50% indicates that z_shared encodes enough information
to uniquely identify corresponding pairs. This supports the cross-modal hypothesis.
"""
    elif retrieval['top1_accuracy'] > 0.2:
        report += """
**MODERATE**: Top-1 accuracy 20-50% indicates partial discriminative power.
z_shared captures some unique information but may benefit from more training.
"""
    else:
        report += """
**WEAK**: Top-1 accuracy < 20% suggests z_shared may be too generic or collapsed.
Consider checking for posterior collapse or adjusting InfoNCE temperature.
"""

    report += f"""
---

## 4. Recommendations

### If results are weak:
1. Increase InfoNCE weight in training
2. Add explicit cross-reconstruction loss
3. Check for posterior collapse (KL per dimension)
4. Verify data alignment (audio-vib timestamps)

### If results are strong:
1. Proceed to ablation studies
2. Test on held-out conditions
3. Document for publication

---

*Generated by evaluate_cross_reconstruction.py*
"""

    with open(output_path / 'REPORT_CROSS_RECONSTRUCTION.md', 'w') as f:
        f.write(report)

    print(f"Saved report to {output_path / 'REPORT_CROSS_RECONSTRUCTION.md'}")


def compute_mean_baseline(dataloader, device) -> Dict[str, np.ndarray]:
    """
    Compute mean histogram baseline from training data.
    This is the simplest baseline: predict average histogram for all samples.
    """
    audio_sum = None
    vib_sum = None
    n_samples = 0

    for batch in dataloader:
        if len(batch) == 4:
            audio, vibration, metas, lengths = batch
        else:
            audio, vibration, metas = batch

        audio_np = audio.numpy()
        vib_np = vibration.numpy()

        if audio_sum is None:
            audio_sum = audio_np.sum(axis=0)
            vib_sum = vib_np.sum(axis=0)
        else:
            audio_sum += audio_np.sum(axis=0)
            vib_sum += vib_np.sum(axis=0)

        n_samples += audio_np.shape[0]

    return {
        'mean_audio': audio_sum / n_samples,
        'mean_vib': vib_sum / n_samples,
    }


def evaluate_with_controls(
    model: RosetaVAE,
    dataloader: DataLoader,
    device: str,
    use_zero_private: bool = True,
    pairing: str = 'aligned',
    use_random_zshared: bool = False,
    baseline_hists: Optional[Dict] = None,
) -> Dict[str, any]:
    """
    Evaluate cross-reconstruction with negative controls.

    Args:
        model: trained RosetaVAE
        dataloader: data loader
        device: compute device
        use_zero_private: use zeros for z_private
        pairing: 'aligned' (correct pairs) or 'shuffled' (broken pairs)
        use_random_zshared: use random N(0,1) for z_shared instead of encoded
        baseline_hists: mean histograms for baseline comparison

    Returns:
        dict with evaluation metrics
    """
    all_metrics = {
        'cross_audio_to_vib': [],
        'cross_vib_to_audio': [],
    }

    if baseline_hists is not None:
        all_metrics['baseline_vib'] = []
        all_metrics['baseline_audio'] = []

    all_z_audio = []
    all_z_vib = []

    control_desc = f"pairing={pairing}, random_z={use_random_zshared}"
    print(f"\nEvaluating ({control_desc})...")

    for batch in tqdm(dataloader, desc="Processing"):
        if len(batch) == 4:
            audio, vibration, metas, lengths = batch
        else:
            audio, vibration, metas = batch
            lengths = None

        audio = audio.to(device)
        vibration = vibration.to(device)

        # Shuffled pairing: break audio-vibration correspondence
        if pairing == 'shuffled':
            idx = torch.randperm(vibration.size(0))
            vibration = vibration[idx]

        with torch.no_grad():
            # Encode
            audio_enc = model.encode_audio(audio)
            vib_enc = model.encode_vibration(vibration)

            if use_random_zshared:
                # Random z_shared (should destroy cross-reconstruction)
                B, T, _ = audio_enc['z_shared_mean'].shape
                z_shared_audio = torch.randn(B, T, model.z_shared_dim, device=device)
                z_shared_vib = torch.randn(B, T, model.z_shared_dim, device=device)
            else:
                z_shared_audio = audio_enc['z_shared_mean']
                z_shared_vib = vib_enc['z_shared_mean']

            B, T, _ = z_shared_audio.shape

            if use_zero_private:
                z_private = torch.zeros(B, T, model.z_private_dim, device=device)
            else:
                z_private = vib_enc['z_private_mean']

            # Cross-reconstruction
            vib_from_audio = model.decode_vibration(z_shared_audio, z_private)
            audio_from_vib = model.decode_audio(z_shared_vib,
                                                 audio_enc['z_private_mean'] if not use_zero_private
                                                 else torch.zeros(B, T, model.z_private_dim, device=device))

        # Compute metrics
        audio_np = audio.cpu().numpy()
        vib_np = vibration.cpu().numpy()
        vib_from_audio_np = vib_from_audio.cpu().numpy()
        audio_from_vib_np = audio_from_vib.cpu().numpy()

        for i in range(audio_np.shape[0]):
            all_metrics['cross_audio_to_vib'].append(
                histogram_metrics(vib_from_audio_np[i], vib_np[i])
            )
            all_metrics['cross_vib_to_audio'].append(
                histogram_metrics(audio_from_vib_np[i], audio_np[i])
            )

            # Baseline comparison
            if baseline_hists is not None:
                all_metrics['baseline_vib'].append(
                    histogram_metrics(baseline_hists['mean_vib'], vib_np[i])
                )
                all_metrics['baseline_audio'].append(
                    histogram_metrics(baseline_hists['mean_audio'], audio_np[i])
                )

        # Collect z for retrieval
        z_a = z_shared_audio.view(-1, model.z_shared_dim)
        z_v = z_shared_vib.view(-1, model.z_shared_dim)
        all_z_audio.append(z_a.cpu())
        all_z_vib.append(z_v.cpu())

    # Aggregate
    aggregated = {}
    for key, metrics_list in all_metrics.items():
        if not metrics_list:
            continue
        aggregated[key] = {}
        for metric_name in metrics_list[0].keys():
            values = [m[metric_name] for m in metrics_list]
            aggregated[key][metric_name] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
            }

    # Retrieval
    z_audio_all = torch.cat(all_z_audio, dim=0)
    z_vib_all = torch.cat(all_z_vib, dim=0)
    N = min(1000, z_audio_all.size(0))
    idx = torch.randperm(z_audio_all.size(0))[:N]
    retrieval = retrieval_accuracy(z_audio_all[idx], z_vib_all[idx])

    return {
        'reconstruction_metrics': aggregated,
        'retrieval_metrics': retrieval,
        'n_samples': len(all_metrics['cross_audio_to_vib']),
        'control_settings': {
            'pairing': pairing,
            'use_random_zshared': use_random_zshared,
            'use_zero_private': use_zero_private,
        }
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate cross-reconstruction')
    parser.add_argument('--model', type=str,
                        default='models/roseta_vae_best.pt',
                        help='Path to trained model')
    parser.add_argument('--data', type=str,
                        default='data/datasets/roseta_full.npz',
                        help='Path to dataset')
    parser.add_argument('--output', type=str,
                        default='data/evaluations/cross_reconstruction',
                        help='Output directory')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--use-target-private', action='store_true',
                        help='Use target z_private instead of zeros')
    # Negative control arguments (WP2)
    parser.add_argument('--pairing', type=str, choices=['aligned', 'shuffled'],
                        default='aligned',
                        help='Pairing mode: aligned (correct) or shuffled (broken)')
    parser.add_argument('--baseline', type=str, choices=['none', 'mean_hist'],
                        default='none',
                        help='Baseline comparison: none or mean_hist')
    parser.add_argument('--zshared', type=str, choices=['real', 'random'],
                        default='real',
                        help='z_shared source: real (encoded) or random (N(0,1))')
    parser.add_argument('--run-all-controls', action='store_true',
                        help='Run all control conditions and compare')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("Cross-Reconstruction Evaluation: The Rosetta Stone Test")
    print("="*60)
    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print(f"Data: {args.data}")
    print(f"Controls: pairing={args.pairing}, baseline={args.baseline}, zshared={args.zshared}")

    # Load model
    model = load_model(args.model, device)

    # Load dataset
    print(f"\nLoading dataset...")
    dataset = RosetaDataset(
        args.data,
        strategy='sequence',
        max_frames=100,  # Truncate long sequences
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_roseta_sequences,
    )
    print(f"  Samples: {len(dataset)}")
    print(f"  Stats: {dataset.get_stats()}")

    # Compute baseline if requested
    baseline_hists = None
    if args.baseline == 'mean_hist':
        print("\nComputing mean histogram baseline...")
        baseline_hists = compute_mean_baseline(dataloader, device)

    use_zero_private = not args.use_target_private

    if args.run_all_controls:
        # Run all control conditions for comparison
        print("\n" + "="*60)
        print("Running ALL control conditions...")
        print("="*60)

        all_results = {}

        # Condition 1: Aligned (normal)
        print("\n[1/4] Aligned pairs (normal condition)...")
        all_results['aligned'] = evaluate_with_controls(
            model, dataloader, device, use_zero_private,
            pairing='aligned', use_random_zshared=False, baseline_hists=baseline_hists
        )

        # Condition 2: Shuffled pairs (negative control)
        print("\n[2/4] Shuffled pairs (negative control)...")
        all_results['shuffled'] = evaluate_with_controls(
            model, dataloader, device, use_zero_private,
            pairing='shuffled', use_random_zshared=False, baseline_hists=baseline_hists
        )

        # Condition 3: Random z_shared (negative control)
        print("\n[3/4] Random z_shared (negative control)...")
        all_results['random_z'] = evaluate_with_controls(
            model, dataloader, device, use_zero_private,
            pairing='aligned', use_random_zshared=True, baseline_hists=baseline_hists
        )

        # Condition 4: Both shuffled AND random (strongest negative)
        print("\n[4/4] Shuffled + Random z (double negative)...")
        all_results['shuffled_random'] = evaluate_with_controls(
            model, dataloader, device, use_zero_private,
            pairing='shuffled', use_random_zshared=True, baseline_hists=baseline_hists
        )

        # Save all results
        with open(output_path / 'all_controls_results.json', 'w') as f:
            json.dump(all_results, f, indent=2)

        # Generate comparison table
        print("\n" + "="*60)
        print("CONTROL CONDITIONS COMPARISON")
        print("="*60)
        print("\nCross-Reconstruction Pearson Correlation:")
        print("-" * 50)
        print(f"{'Condition':<20} {'Audio→Vib':<15} {'Vib→Audio':<15}")
        print("-" * 50)

        for cond_name, cond_results in all_results.items():
            recon = cond_results['reconstruction_metrics']
            a2v = recon['cross_audio_to_vib']['pearson_corr']['mean']
            v2a = recon['cross_vib_to_audio']['pearson_corr']['mean']
            print(f"{cond_name:<20} {a2v:<15.4f} {v2a:<15.4f}")

        if baseline_hists is not None:
            print("\nBaseline Comparison (mean histogram):")
            aligned_a2v = all_results['aligned']['reconstruction_metrics']['cross_audio_to_vib']['pearson_corr']['mean']
            baseline_a2v = all_results['aligned']['reconstruction_metrics'].get('baseline_vib', {}).get('pearson_corr', {}).get('mean', 0)
            print(f"  Model: {aligned_a2v:.4f}")
            print(f"  Mean baseline: {baseline_a2v:.4f}")
            print(f"  Delta (model - baseline): {aligned_a2v - baseline_a2v:+.4f}")

        print("\nRetrieval Top-1 Accuracy:")
        print("-" * 50)
        for cond_name, cond_results in all_results.items():
            top1 = cond_results['retrieval_metrics']['top1_accuracy']
            print(f"{cond_name:<20} {top1:.4f}")

        # Validation checks
        print("\n" + "="*60)
        print("VALIDATION CHECKS (Go/No-Go)")
        print("="*60)

        aligned_corr = (all_results['aligned']['reconstruction_metrics']['cross_audio_to_vib']['pearson_corr']['mean'] +
                        all_results['aligned']['reconstruction_metrics']['cross_vib_to_audio']['pearson_corr']['mean']) / 2
        shuffled_corr = (all_results['shuffled']['reconstruction_metrics']['cross_audio_to_vib']['pearson_corr']['mean'] +
                         all_results['shuffled']['reconstruction_metrics']['cross_vib_to_audio']['pearson_corr']['mean']) / 2

        delta_corr = aligned_corr - shuffled_corr
        print(f"\n1. Shuffled degradation: Δcorr = {delta_corr:.4f}")
        if delta_corr > 0.15:
            print("   ✓ PASS: Shuffled pairs significantly worse (Δ > 0.15)")
        else:
            print("   ✗ FAIL: Shuffled pairs not degraded enough")

        aligned_top1 = all_results['aligned']['retrieval_metrics']['top1_accuracy']
        shuffled_top1 = all_results['shuffled']['retrieval_metrics']['top1_accuracy']
        random_baseline = 1.0 / all_results['aligned']['n_samples']

        print(f"\n2. Retrieval sanity check:")
        print(f"   Aligned Top-1: {aligned_top1:.4f}")
        print(f"   Shuffled Top-1: {shuffled_top1:.4f}")
        print(f"   Random baseline: {random_baseline:.6f}")

        if shuffled_top1 < 0.05:
            print("   ✓ PASS: Shuffled retrieval near random")
        else:
            print("   ⚠ WARNING: Shuffled retrieval unexpectedly high")

        results = all_results['aligned']  # Use aligned for main report

    else:
        # Single condition evaluation
        results = evaluate_with_controls(
            model, dataloader, device, use_zero_private,
            pairing=args.pairing,
            use_random_zshared=(args.zshared == 'random'),
            baseline_hists=baseline_hists
        )

        # Also run standard evaluate_dataset for full metrics
        standard_results = evaluate_dataset(model, dataloader, device, use_zero_private)
        results['full_metrics'] = standard_results

    # Save raw results
    with open(output_path / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Generate visualizations (use standard results if available)
    if 'full_metrics' in results:
        plot_results(results['full_metrics'], output_path)
        generate_report(results['full_metrics'], output_path, use_zero_private)
        display_results = results['full_metrics']
    else:
        display_results = results

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY: Cross-Reconstruction Results")
    print("="*60)

    recon = display_results.get('reconstruction_metrics', {})

    if 'self_recon_audio' in recon:
        print(f"\nReconstruction (Pearson Correlation):")
        print(f"  Self-recon Audio:    {recon['self_recon_audio']['pearson_corr']['mean']:.4f}")
        print(f"  Self-recon Vib:      {recon['self_recon_vib']['pearson_corr']['mean']:.4f}")
        print(f"  Cross Audio→Vib:     {recon['cross_audio_to_vib']['pearson_corr']['mean']:.4f}")
        print(f"  Cross Vib→Audio:     {recon['cross_vib_to_audio']['pearson_corr']['mean']:.4f}")
        if 'cycle_audio' in recon:
            print(f"  Cycle Audio:         {recon['cycle_audio']['pearson_corr']['mean']:.4f}")
    elif 'cross_audio_to_vib' in recon:
        print(f"\nCross-Reconstruction (Pearson Correlation):")
        print(f"  Audio→Vib:     {recon['cross_audio_to_vib']['pearson_corr']['mean']:.4f}")
        print(f"  Vib→Audio:     {recon['cross_vib_to_audio']['pearson_corr']['mean']:.4f}")

    if 'retrieval_metrics' in display_results:
        print(f"\nRetrieval:")
        print(f"  Top-1 Accuracy:      {display_results['retrieval_metrics']['top1_accuracy']:.4f}")
        print(f"  Top-5 Accuracy:      {display_results['retrieval_metrics']['top5_accuracy']:.4f}")
        print(f"  MRR:                 {display_results['retrieval_metrics']['mrr']:.4f}")

    if 'cross_audio_to_vib' in recon:
        cross_corr = (recon['cross_audio_to_vib']['pearson_corr']['mean'] +
                      recon['cross_vib_to_audio']['pearson_corr']['mean']) / 2

        print(f"\n{'='*60}")
        if cross_corr > 0.7:
            print("✓ PASS: Cross-reconstruction demonstrates genuine cross-modality")
        elif cross_corr > 0.5:
            print("~ PARTIAL: Cross-reconstruction shows moderate transfer")
        else:
            print("✗ FAIL: Cross-reconstruction insufficient for H3 claim")
        print(f"{'='*60}")


if __name__ == '__main__':
    main()
