#!/usr/bin/env python3
"""
Regime Separation Evaluation for RosetaVAE - WP5
=================================================

Evaluates whether z_shared separates healthy from fault conditions.

Metrics:
1. Silhouette Score (on z_shared, NOT on UMAP)
2. Linear Probe AUC (logistic regression)
3. Normalized Centroid Distance

This addresses the concern about conflating separability claims.

Usage:
------
python experiments/evaluate_regime_separation.py \
    --model models/roseta_vae_best.pt \
    --data data/datasets/roseta_full.npz \
    --output data/evaluations/regime_separation
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import silhouette_score, roc_auc_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

from src.RNA.roseta_vae import RosetaVAE
from src.datasets.roseta_dataset import RosetaDataset, collate_roseta_sequences


def load_model(model_path: str, device: str) -> RosetaVAE:
    """Load trained RosetaVAE model."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

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

    return model


@torch.no_grad()
def extract_embeddings(
    model: RosetaVAE,
    dataloader: DataLoader,
    device: str,
) -> Dict[str, np.ndarray]:
    """Extract embeddings and conditions."""
    all_audio_z = []
    all_vib_z = []
    all_conditions = []
    all_is_healthy = []

    for batch in tqdm(dataloader, desc="Extracting"):
        audio, vibration, metas, lengths = batch
        audio = audio.to(device)
        vibration = vibration.to(device)

        audio_enc = model.encode_audio(audio)
        vib_enc = model.encode_vibration(vibration)

        # Use mean, average over time
        audio_z = audio_enc['z_shared_mean'].mean(dim=1).cpu().numpy()
        vib_z = vib_enc['z_shared_mean'].mean(dim=1).cpu().numpy()

        all_audio_z.append(audio_z)
        all_vib_z.append(vib_z)

        for meta in metas:
            cond = meta['condition']
            all_conditions.append(cond)
            all_is_healthy.append(1 if cond == 'HH' else 0)

    return {
        'audio_z': np.concatenate(all_audio_z, axis=0),
        'vib_z': np.concatenate(all_vib_z, axis=0),
        'conditions': np.array(all_conditions),
        'is_healthy': np.array(all_is_healthy),
    }


def evaluate_regime_separation(
    z_shared: np.ndarray,
    labels: np.ndarray,
    label_type: str = 'binary',
) -> Dict[str, float]:
    """
    Evaluate regime separation in z_shared.

    Args:
        z_shared: [N, D] embeddings
        labels: [N] labels (0=healthy, 1=fault for binary; or condition names)
        label_type: 'binary' or 'multiclass'

    Returns:
        dict with metrics
    """
    results = {}

    # 1. Silhouette Score (on raw z_shared, NOT on dimensionality-reduced)
    if len(np.unique(labels)) >= 2:
        silhouette = silhouette_score(z_shared, labels)
        results['silhouette'] = float(silhouette)
    else:
        results['silhouette'] = 0.0

    # 2. Linear Probe with cross-validation
    if label_type == 'binary':
        clf = LogisticRegression(max_iter=1000, random_state=42)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # AUC scores
        try:
            auc_scores = cross_val_score(clf, z_shared, labels, cv=cv, scoring='roc_auc')
            results['linear_probe_auc'] = float(np.mean(auc_scores))
            results['linear_probe_auc_std'] = float(np.std(auc_scores))
        except Exception as e:
            results['linear_probe_auc'] = 0.5
            results['linear_probe_auc_std'] = 0.0

        # Accuracy scores
        acc_scores = cross_val_score(clf, z_shared, labels, cv=cv, scoring='accuracy')
        results['linear_probe_accuracy'] = float(np.mean(acc_scores))
        results['linear_probe_accuracy_std'] = float(np.std(acc_scores))

    else:  # multiclass
        # Encode labels
        le = LabelEncoder()
        encoded_labels = le.fit_transform(labels)

        clf = LogisticRegression(max_iter=1000, multi_class='multinomial', random_state=42)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        acc_scores = cross_val_score(clf, z_shared, encoded_labels, cv=cv, scoring='accuracy')
        results['linear_probe_accuracy'] = float(np.mean(acc_scores))
        results['linear_probe_accuracy_std'] = float(np.std(acc_scores))

        # Random baseline
        results['random_baseline_accuracy'] = 1.0 / len(np.unique(labels))

    # 3. Centroid Distance (normalized)
    unique_labels = np.unique(labels)
    if len(unique_labels) == 2:
        c0 = z_shared[labels == unique_labels[0]].mean(axis=0)
        c1 = z_shared[labels == unique_labels[1]].mean(axis=0)

        pooled_std = z_shared.std()
        centroid_dist = np.linalg.norm(c0 - c1) / (pooled_std + 1e-8)
        results['centroid_distance_normalized'] = float(centroid_dist)

        # Also raw distance
        results['centroid_distance_raw'] = float(np.linalg.norm(c0 - c1))

    elif len(unique_labels) > 2:
        # Average pairwise centroid distance for multiclass
        centroids = {lbl: z_shared[labels == lbl].mean(axis=0) for lbl in unique_labels}
        pooled_std = z_shared.std()

        distances = []
        for i, l1 in enumerate(unique_labels):
            for l2 in unique_labels[i+1:]:
                dist = np.linalg.norm(centroids[l1] - centroids[l2])
                distances.append(dist)

        results['avg_centroid_distance_raw'] = float(np.mean(distances))
        results['avg_centroid_distance_normalized'] = float(np.mean(distances) / (pooled_std + 1e-8))

    # 4. Within-class variance vs between-class variance ratio (Fisher criterion)
    if len(unique_labels) >= 2:
        overall_mean = z_shared.mean(axis=0)

        # Between-class scatter
        Sb = 0
        for lbl in unique_labels:
            mask = labels == lbl
            n_k = mask.sum()
            mean_k = z_shared[mask].mean(axis=0)
            Sb += n_k * np.outer(mean_k - overall_mean, mean_k - overall_mean)

        # Within-class scatter
        Sw = 0
        for lbl in unique_labels:
            mask = labels == lbl
            centered = z_shared[mask] - z_shared[mask].mean(axis=0)
            Sw += centered.T @ centered

        # Trace ratio (higher is better)
        trace_Sb = np.trace(Sb)
        trace_Sw = np.trace(Sw)
        fisher_ratio = trace_Sb / (trace_Sw + 1e-8)
        results['fisher_ratio'] = float(fisher_ratio)

    return results


def plot_separation(
    embeddings: Dict[str, np.ndarray],
    output_path: Path,
):
    """Generate visualization of regime separation."""
    from sklearn.decomposition import PCA

    z_combined = np.vstack([embeddings['audio_z'], embeddings['vib_z']])
    conditions = np.concatenate([embeddings['conditions'], embeddings['conditions']])
    modalities = np.array(['audio'] * len(embeddings['audio_z']) +
                          ['vib'] * len(embeddings['vib_z']))

    # PCA for visualization
    pca = PCA(n_components=2)
    z_2d = pca.fit_transform(z_combined)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Colored by condition
    unique_conds = np.unique(conditions)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_conds)))
    cond_to_color = dict(zip(unique_conds, colors))

    ax = axes[0]
    for cond in unique_conds:
        mask = conditions == cond
        ax.scatter(z_2d[mask, 0], z_2d[mask, 1],
                   c=[cond_to_color[cond]], label=cond, alpha=0.7, s=30)
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    ax.set_title('z_shared by Condition')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')

    # Plot 2: Colored by healthy vs fault
    ax = axes[1]
    is_healthy = np.array([c == 'HH' for c in conditions])
    ax.scatter(z_2d[is_healthy, 0], z_2d[is_healthy, 1],
               c='green', label='Healthy (HH)', alpha=0.7, s=30)
    ax.scatter(z_2d[~is_healthy, 0], z_2d[~is_healthy, 1],
               c='red', label='Fault', alpha=0.7, s=30)
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    ax.set_title('z_shared: Healthy vs Fault')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path / 'regime_separation_pca.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved PCA plot to {output_path / 'regime_separation_pca.png'}")


def generate_report(
    binary_results: Dict,
    multiclass_results: Dict,
    audio_results: Dict,
    vib_results: Dict,
    output_path: Path,
    n_samples: int,
):
    """Generate regime separation report."""

    report = f"""# Regime Separation Evaluation Report (WP5)

## Overview

This report evaluates whether z_shared meaningfully separates operational regimes
(healthy vs fault conditions). We use metrics on the raw z_shared space, NOT on
dimensionality-reduced visualizations like UMAP/t-SNE.

**Samples**: {n_samples}

---

## 1. Binary Classification (Healthy vs Fault)

### Audio z_shared

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Silhouette Score | {audio_results['binary']['silhouette']:.4f} | {"Strong" if audio_results['binary']['silhouette'] > 0.5 else "Moderate" if audio_results['binary']['silhouette'] > 0.25 else "Weak"} separation |
| Linear Probe AUC | {audio_results['binary']['linear_probe_auc']:.4f} ± {audio_results['binary'].get('linear_probe_auc_std', 0):.4f} | {"Strong" if audio_results['binary']['linear_probe_auc'] > 0.9 else "Moderate" if audio_results['binary']['linear_probe_auc'] > 0.75 else "Weak"} |
| Linear Probe Accuracy | {audio_results['binary']['linear_probe_accuracy']:.4f} | - |
| Centroid Distance (norm) | {audio_results['binary'].get('centroid_distance_normalized', 0):.4f} | {"Large" if audio_results['binary'].get('centroid_distance_normalized', 0) > 1 else "Moderate" if audio_results['binary'].get('centroid_distance_normalized', 0) > 0.5 else "Small"} gap |
| Fisher Ratio | {audio_results['binary'].get('fisher_ratio', 0):.4f} | Higher = better |

### Vibration z_shared

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Silhouette Score | {vib_results['binary']['silhouette']:.4f} | {"Strong" if vib_results['binary']['silhouette'] > 0.5 else "Moderate" if vib_results['binary']['silhouette'] > 0.25 else "Weak"} separation |
| Linear Probe AUC | {vib_results['binary']['linear_probe_auc']:.4f} ± {vib_results['binary'].get('linear_probe_auc_std', 0):.4f} | - |
| Centroid Distance (norm) | {vib_results['binary'].get('centroid_distance_normalized', 0):.4f} | - |

---

## 2. Multi-class Classification (All Conditions)

### Audio z_shared

| Metric | Value |
|--------|-------|
| Silhouette Score | {audio_results['multiclass']['silhouette']:.4f} |
| Linear Probe Accuracy | {audio_results['multiclass']['linear_probe_accuracy']:.4f} ± {audio_results['multiclass'].get('linear_probe_accuracy_std', 0):.4f} |
| Random Baseline | {audio_results['multiclass'].get('random_baseline_accuracy', 0):.4f} |
| Avg Centroid Distance (norm) | {audio_results['multiclass'].get('avg_centroid_distance_normalized', 0):.4f} |

### Vibration z_shared

| Metric | Value |
|--------|-------|
| Silhouette Score | {vib_results['multiclass']['silhouette']:.4f} |
| Linear Probe Accuracy | {vib_results['multiclass']['linear_probe_accuracy']:.4f} |

---

## 3. Interpretation Guidelines

### Silhouette Score
- **> 0.5**: Strong structure, clear separation
- **0.25 - 0.5**: Moderate structure, some overlap
- **< 0.25**: Weak structure, significant overlap

### Linear Probe AUC
- **> 0.9**: Excellent linear separability
- **0.75 - 0.9**: Good separability
- **0.5 - 0.75**: Marginal separability
- **~0.5**: No better than random

### What This Means for Claims

"""
    # Add interpretation
    sil = audio_results['binary']['silhouette']
    auc = audio_results['binary']['linear_probe_auc']

    if sil > 0.5 and auc > 0.9:
        report += """
**STRONG SEPARATION**: z_shared clearly distinguishes healthy from fault conditions.
This supports claims about regime-specific encoding in the latent space.
"""
    elif sil > 0.25 or auc > 0.75:
        report += """
**MODERATE SEPARATION**: z_shared shows some separation between regimes, but with
significant overlap. Claims about regime separation should be made cautiously.
"""
    else:
        report += """
**WEAK SEPARATION**: z_shared does not clearly separate regimes. Claims about
regime-specific encoding are NOT supported. The model may be learning
general signal features rather than condition-specific patterns.
"""

    report += """
---

## 4. Recommendations

1. **If separation is weak**: Consider condition-aware training (e.g., condition
   prediction auxiliary loss) or curriculum learning.

2. **If multiclass is poor but binary is good**: The model may only distinguish
   "normal" from "abnormal" without differentiating fault types.

3. **Compare with raw PSD baseline**: Does z_shared outperform raw spectral
   features? If not, the model may not be learning useful abstractions.

---

*Generated by evaluate_regime_separation.py (WP5)*
"""

    with open(output_path / 'REPORT_REGIME_SEPARATION.md', 'w') as f:
        f.write(report)

    print(f"Report saved to {output_path / 'REPORT_REGIME_SEPARATION.md'}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate regime separation')
    parser.add_argument('--model', type=str,
                        default='models/roseta_vae_best.pt',
                        help='Path to trained model')
    parser.add_argument('--data', type=str,
                        default='data/datasets/roseta_full.npz',
                        help='Path to dataset')
    parser.add_argument('--output', type=str,
                        default='data/evaluations/regime_separation',
                        help='Output directory')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--max-frames', type=int, default=100)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Regime Separation Evaluation (WP5)")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print(f"Data: {args.data}")

    # Load model
    model = load_model(args.model, device)

    # Load dataset
    print("\nLoading dataset...")
    dataset = RosetaDataset(
        args.data,
        strategy='sequence',
        max_frames=args.max_frames,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_roseta_sequences,
    )
    print(f"  Samples: {len(dataset)}")

    # Extract embeddings
    print("\nExtracting embeddings...")
    embeddings = extract_embeddings(model, dataloader, device)

    # Evaluate
    print("\nEvaluating regime separation...")

    # Binary: healthy vs fault
    print("  [1/4] Audio binary (healthy vs fault)...")
    audio_binary = evaluate_regime_separation(
        embeddings['audio_z'],
        embeddings['is_healthy'],
        label_type='binary'
    )

    print("  [2/4] Vibration binary...")
    vib_binary = evaluate_regime_separation(
        embeddings['vib_z'],
        embeddings['is_healthy'],
        label_type='binary'
    )

    # Multiclass: all conditions
    print("  [3/4] Audio multiclass (all conditions)...")
    audio_multiclass = evaluate_regime_separation(
        embeddings['audio_z'],
        embeddings['conditions'],
        label_type='multiclass'
    )

    print("  [4/4] Vibration multiclass...")
    vib_multiclass = evaluate_regime_separation(
        embeddings['vib_z'],
        embeddings['conditions'],
        label_type='multiclass'
    )

    # Compile results
    results = {
        'audio': {
            'binary': audio_binary,
            'multiclass': audio_multiclass,
        },
        'vib': {
            'binary': vib_binary,
            'multiclass': vib_multiclass,
        },
        'n_samples': len(dataset),
        'conditions': list(np.unique(embeddings['conditions'])),
    }

    # Save results
    with open(output_path / 'regime_separation_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Generate visualization
    print("\nGenerating visualization...")
    plot_separation(embeddings, output_path)

    # Generate report
    generate_report(
        binary_results={'audio': audio_binary, 'vib': vib_binary},
        multiclass_results={'audio': audio_multiclass, 'vib': vib_multiclass},
        audio_results={'binary': audio_binary, 'multiclass': audio_multiclass},
        vib_results={'binary': vib_binary, 'multiclass': vib_multiclass},
        output_path=output_path,
        n_samples=len(dataset),
    )

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY: Regime Separation")
    print("=" * 60)

    print(f"\nBinary (Healthy vs Fault):")
    print(f"  Audio Silhouette: {audio_binary['silhouette']:.4f}")
    print(f"  Audio Linear Probe AUC: {audio_binary['linear_probe_auc']:.4f}")
    print(f"  Vib Silhouette: {vib_binary['silhouette']:.4f}")

    print(f"\nMulticlass (All Conditions):")
    print(f"  Audio Silhouette: {audio_multiclass['silhouette']:.4f}")
    print(f"  Audio Linear Probe Acc: {audio_multiclass['linear_probe_accuracy']:.4f}")

    # Verdict
    sil = audio_binary['silhouette']
    if sil > 0.5:
        print("\n✓ STRONG regime separation")
    elif sil > 0.25:
        print("\n~ MODERATE regime separation")
    else:
        print("\n✗ WEAK regime separation")


if __name__ == '__main__':
    main()
