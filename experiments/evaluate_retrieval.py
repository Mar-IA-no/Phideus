#!/usr/bin/env python3
"""
Retrieval Evaluation for RosetaVAE - WP4
=========================================

Extended retrieval evaluation with multiple modes:
- Global: all samples against all samples
- Intra-condition: only candidates from same condition
- Intra-file: only candidates from same file (different frames)

This addresses the critique that retrieval in a small batch is too easy.

Usage:
------
python experiments/evaluate_retrieval.py \
    --model models/roseta_vae_best.pt \
    --data data/datasets/roseta_full.npz \
    --output data/evaluations/retrieval
"""

import sys
import os

# Configure CUBLAS for deterministic behavior (must be set before importing torch)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.RNA.roseta_vae import RosetaVAE
from src.datasets.roseta_dataset import (
    RosetaDataset, collate_roseta_sequences, create_roseta_dataloaders
)


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
    """Extract all embeddings and metadata."""
    all_audio_z = []
    all_vib_z = []
    all_conditions = []
    all_file_indices = []

    for batch in tqdm(dataloader, desc="Extracting embeddings"):
        audio, vibration, metas, lengths = batch
        audio = audio.to(device)
        vibration = vibration.to(device)

        audio_enc = model.encode_audio(audio)
        vib_enc = model.encode_vibration(vibration)

        # Use mean and average over time
        audio_z = audio_enc['z_shared_mean'].mean(dim=1).cpu().numpy()
        vib_z = vib_enc['z_shared_mean'].mean(dim=1).cpu().numpy()

        all_audio_z.append(audio_z)
        all_vib_z.append(vib_z)

        for meta in metas:
            all_conditions.append(meta['condition'])
            # Use filename as file identifier
            all_file_indices.append(meta.get('filename', 'unknown'))

    return {
        'audio_z': np.concatenate(all_audio_z, axis=0),
        'vib_z': np.concatenate(all_vib_z, axis=0),
        'conditions': np.array(all_conditions),
        'file_indices': np.array(all_file_indices),
    }


def compute_retrieval_metrics(
    z_query: np.ndarray,
    z_gallery: np.ndarray,
    k_values: List[int] = [1, 5, 10],
    mask: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute retrieval metrics.

    Args:
        z_query: [N, D] query embeddings
        z_gallery: [N, D] gallery embeddings
        k_values: list of k for top-k accuracy
        mask: [N, N] boolean mask where True means valid candidate

    Returns:
        dict with metrics
    """
    # L2 normalize
    z_q = z_query / (np.linalg.norm(z_query, axis=1, keepdims=True) + 1e-8)
    z_g = z_gallery / (np.linalg.norm(z_gallery, axis=1, keepdims=True) + 1e-8)

    # Similarity matrix
    sim = z_q @ z_g.T  # [N, N]

    if mask is not None:
        # Set invalid candidates to -inf
        sim = np.where(mask, sim, -np.inf)

    N = z_q.shape[0]
    labels = np.arange(N)

    results = {}

    # Top-k accuracy
    for k in k_values:
        if k > N:
            k = N
        topk_indices = np.argsort(-sim, axis=1)[:, :k]
        correct = np.any(topk_indices == labels[:, None], axis=1)
        results[f'top{k}_accuracy'] = float(correct.mean())

    # Mean Reciprocal Rank
    sorted_indices = np.argsort(-sim, axis=1)
    ranks = np.where(sorted_indices == labels[:, None])[1] + 1
    results['mrr'] = float((1.0 / ranks).mean())

    # Mean rank
    results['mean_rank'] = float(ranks.mean())

    # Recall@k for different k
    for k in [1, 5, 10, 20, 50]:
        if k <= N:
            topk = np.argsort(-sim, axis=1)[:, :k]
            recall = np.any(topk == labels[:, None], axis=1).mean()
            results[f'recall@{k}'] = float(recall)

    return results


def evaluate_retrieval_modes(
    embeddings: Dict[str, np.ndarray],
    k_values: List[int] = [1, 5, 10],
) -> Dict[str, Dict]:
    """
    Evaluate retrieval in multiple modes.

    Modes:
    - global: all vs all
    - intra_condition: only within same condition
    - cross_condition: only across different conditions
    """
    audio_z = embeddings['audio_z']
    vib_z = embeddings['vib_z']
    conditions = embeddings['conditions']
    file_indices = embeddings['file_indices']

    N = len(audio_z)
    results = {}

    # 1. Global retrieval (audio → vib)
    print("\n[1] Global retrieval (audio → vibration)...")
    results['global_audio_to_vib'] = compute_retrieval_metrics(
        audio_z, vib_z, k_values
    )

    # 2. Global retrieval (vib → audio)
    print("[2] Global retrieval (vibration → audio)...")
    results['global_vib_to_audio'] = compute_retrieval_metrics(
        vib_z, audio_z, k_values
    )

    # 3. Intra-condition retrieval
    print("[3] Intra-condition retrieval...")
    unique_conditions = np.unique(conditions)

    intra_results = []
    for cond in unique_conditions:
        mask = conditions == cond
        if mask.sum() < 2:
            continue

        idx = np.where(mask)[0]
        audio_subset = audio_z[idx]
        vib_subset = vib_z[idx]

        cond_results = compute_retrieval_metrics(
            audio_subset, vib_subset, k_values
        )
        cond_results['condition'] = cond
        cond_results['n_samples'] = len(idx)
        intra_results.append(cond_results)

    results['intra_condition'] = intra_results

    # Average intra-condition metrics
    if intra_results:
        avg_intra = {}
        for key in ['top1_accuracy', 'top5_accuracy', 'mrr']:
            values = [r[key] for r in intra_results if key in r]
            if values:
                avg_intra[key] = float(np.mean(values))
        results['intra_condition_avg'] = avg_intra

    # 4. Cross-condition retrieval (harder: find match among other conditions)
    print("[4] Cross-condition retrieval...")
    # For each sample, candidates are only from OTHER conditions
    cross_mask = np.array([
        [conditions[i] != conditions[j] for j in range(N)]
        for i in range(N)
    ])
    # But we still want the diagonal to be findable if it's the only match
    # Actually for cross-condition, we use different approach:
    # Query from condition A, gallery from all, but exclude same condition
    cross_results = []
    for query_cond in unique_conditions:
        query_mask = conditions == query_cond
        gallery_mask = conditions != query_cond

        query_idx = np.where(query_mask)[0]
        gallery_idx = np.where(gallery_mask)[0]

        if len(query_idx) == 0 or len(gallery_idx) == 0:
            continue

        # For cross-condition, we can't have exact matches
        # So we measure how well embeddings cluster
        audio_query = audio_z[query_idx]
        vib_gallery = vib_z[gallery_idx]

        # Compute nearest neighbor (but no ground truth match exists)
        # Instead, measure consistency: audio and vib from same condition should be similar
        vib_query = vib_z[query_idx]
        audio_gallery = audio_z[gallery_idx]

        # Self-consistency: how similar is audio to its paired vib?
        sim_paired = (audio_query * vib_query).sum(axis=1) / (
            np.linalg.norm(audio_query, axis=1) * np.linalg.norm(vib_query, axis=1) + 1e-8
        )

        cross_results.append({
            'query_condition': query_cond,
            'mean_paired_similarity': float(sim_paired.mean()),
            'n_queries': len(query_idx),
        })

    results['cross_condition'] = cross_results

    # 5. Shuffled baseline (negative control)
    print("[5] Shuffled baseline (negative control)...")
    perm = np.random.permutation(N)
    vib_shuffled = vib_z[perm]
    results['shuffled_baseline'] = compute_retrieval_metrics(
        audio_z, vib_shuffled, k_values
    )

    # 6. Random embeddings baseline
    print("[6] Random embeddings baseline...")
    random_z = np.random.randn(*audio_z.shape)
    results['random_baseline'] = compute_retrieval_metrics(
        random_z, vib_z, k_values
    )

    return results


def generate_report(
    results: Dict,
    output_path: Path,
    n_samples: int,
):
    """Generate retrieval evaluation report."""

    report = f"""# Retrieval Evaluation Report (WP4)

## Configuration
- Total samples: {n_samples}
- Expected random baseline: {1/n_samples:.6f} (1/N)

---

## 1. Global Retrieval (All vs All)

### Audio → Vibration
| Metric | Value |
|--------|-------|
| Top-1 Accuracy | {results['global_audio_to_vib']['top1_accuracy']:.4f} |
| Top-5 Accuracy | {results['global_audio_to_vib']['top5_accuracy']:.4f} |
| Top-10 Accuracy | {results['global_audio_to_vib'].get('top10_accuracy', 0):.4f} |
| MRR | {results['global_audio_to_vib']['mrr']:.4f} |
| Mean Rank | {results['global_audio_to_vib']['mean_rank']:.1f} |

### Vibration → Audio
| Metric | Value |
|--------|-------|
| Top-1 Accuracy | {results['global_vib_to_audio']['top1_accuracy']:.4f} |
| Top-5 Accuracy | {results['global_vib_to_audio']['top5_accuracy']:.4f} |
| MRR | {results['global_vib_to_audio']['mrr']:.4f} |

---

## 2. Intra-Condition Retrieval

| Condition | N | Top-1 | Top-5 | MRR |
|-----------|---|-------|-------|-----|
"""
    for r in results.get('intra_condition', []):
        report += f"| {r['condition']} | {r['n_samples']} | {r['top1_accuracy']:.4f} | {r.get('top5_accuracy', 0):.4f} | {r['mrr']:.4f} |\n"

    if 'intra_condition_avg' in results:
        avg = results['intra_condition_avg']
        report += f"\n**Average**: Top-1 = {avg.get('top1_accuracy', 0):.4f}, MRR = {avg.get('mrr', 0):.4f}\n"

    report += f"""
---

## 3. Negative Controls

### Shuffled Pairing (should be near random)
| Metric | Value | Expected |
|--------|-------|----------|
| Top-1 | {results['shuffled_baseline']['top1_accuracy']:.4f} | ~{1/n_samples:.6f} |
| MRR | {results['shuffled_baseline']['mrr']:.4f} | - |

### Random Embeddings
| Metric | Value |
|--------|-------|
| Top-1 | {results['random_baseline']['top1_accuracy']:.4f} |
| MRR | {results['random_baseline']['mrr']:.4f} |

---

## 4. Validation Checks

"""
    # Check 1: Model vs random
    model_top1 = results['global_audio_to_vib']['top1_accuracy']
    random_top1 = results['random_baseline']['top1_accuracy']
    if model_top1 > random_top1 + 0.1:
        report += "✓ **PASS**: Model significantly outperforms random baseline\n"
    else:
        report += "✗ **FAIL**: Model does not outperform random baseline\n"

    # Check 2: Aligned vs shuffled
    shuffled_top1 = results['shuffled_baseline']['top1_accuracy']
    if model_top1 > shuffled_top1 + 0.1:
        report += "✓ **PASS**: Aligned pairs much better than shuffled\n"
    else:
        report += "✗ **FAIL**: Shuffled pairs unexpectedly similar to aligned\n"

    # Check 3: Top-1 > 15% (criterion from plan)
    if model_top1 > 0.15:
        report += f"✓ **PASS**: Top-1 accuracy ({model_top1:.1%}) > 15% threshold\n"
    else:
        report += f"✗ **FAIL**: Top-1 accuracy ({model_top1:.1%}) below 15% threshold\n"

    report += """
---

## Interpretation

The retrieval task tests whether z_shared encodes enough information to
uniquely identify corresponding audio-vibration pairs.

- **Strong** (Top-1 > 50%): z_shared encodes unique identifying information
- **Moderate** (Top-1 15-50%): z_shared captures partial correspondence
- **Weak** (Top-1 < 15%): z_shared may be too generic or collapsed

---

*Generated by evaluate_retrieval.py (WP4)*
"""

    with open(output_path / 'REPORT_RETRIEVAL.md', 'w') as f:
        f.write(report)

    print(f"Report saved to {output_path / 'REPORT_RETRIEVAL.md'}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate retrieval')
    parser.add_argument('--model', type=str,
                        default='models/roseta_vae_best.pt',
                        help='Path to trained model')
    parser.add_argument('--data', type=str,
                        default='data/datasets/roseta_full.npz',
                        help='Path to dataset')
    parser.add_argument('--output', type=str,
                        default='data/evaluations/retrieval',
                        help='Output directory')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--max-frames', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    args = parser.parse_args()

    # Set random seeds for full reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # Enable deterministic algorithms (may raise error if non-deterministic op used)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        print("Warning: Could not enable deterministic algorithms")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Retrieval Evaluation (WP4)")
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
    # Use generator for reproducibility
    g = torch.Generator()
    g.manual_seed(args.seed)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_roseta_sequences,
        num_workers=0,  # Deterministic: no parallel workers
        generator=g,
    )
    print(f"  Samples: {len(dataset)}")

    # Extract embeddings
    print("\nExtracting embeddings...")
    embeddings = extract_embeddings(model, dataloader, device)

    # Evaluate
    print("\nEvaluating retrieval...")
    results = evaluate_retrieval_modes(embeddings)
    results['n_samples'] = len(dataset)

    # Save results
    with open(output_path / 'retrieval_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # Generate report
    generate_report(results, output_path, len(dataset))

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print(f"\nGlobal Retrieval (Audio → Vib):")
    print(f"  Top-1: {results['global_audio_to_vib']['top1_accuracy']:.4f}")
    print(f"  Top-5: {results['global_audio_to_vib']['top5_accuracy']:.4f}")
    print(f"  MRR: {results['global_audio_to_vib']['mrr']:.4f}")

    print(f"\nBaselines:")
    print(f"  Shuffled Top-1: {results['shuffled_baseline']['top1_accuracy']:.4f}")
    print(f"  Random Top-1: {results['random_baseline']['top1_accuracy']:.4f}")
    print(f"  Theoretical random: {1/len(dataset):.6f}")

    model_top1 = results['global_audio_to_vib']['top1_accuracy']
    if model_top1 > 0.15:
        print(f"\n✓ Top-1 > 15%: PASS")
    else:
        print(f"\n✗ Top-1 < 15%: FAIL")


if __name__ == '__main__':
    main()
