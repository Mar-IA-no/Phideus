#!/usr/bin/env python3
"""
Gate 5B — Test 10: UMAP/t-SNE Visualizations.

Generates embedding visualizations for all 4 Gate 5B models:
- UMAP and t-SNE projections
- Color by modality (audio=cyan, midi=magenta)
- 2x2 comparison grid: D0 vs d4a4 vs a4r vs d4-a4r
- Per-model detail plots with piece coloring

Usage:
    python experiments/bias_control/gate5b/test10_visualizations.py --all
    python experiments/bias_control/gate5b/test10_visualizations.py \
        --model models/gate5b/d4a4/best_model.pt
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINTS = {
    'D0': 'models/gate5b/D0/best_model.pt',
    'd4a4': 'models/gate5b/d4a4/best_model.pt',
    'a4r': 'models/gate5b/a4r/best_model.pt',
    'd4-a4r': 'models/gate5b/d4-a4r/best_model.pt',
}

VIZ_DIR = Path('data/gate5b_results/visualizations')


def extract_for_viz(model_path, maestro_dir, device, num_workers, n_samples=2000, seed=42):
    """Extract embeddings for visualization (subsample for speed)."""
    from experiments.bias_control.gate5b.harness import (
        setup_gate5b_test, get_normal_embeddings, get_output_dir,
    )

    class Args:
        pass
    args = Args()
    args.model = model_path
    args.maestro_dir = maestro_dir
    args.device = device
    args.seed = seed
    args.num_workers = num_workers
    args.output = None

    model, meta, dataset, index = setup_gate5b_test(args)
    descriptor = meta['descriptor']
    output_dir = get_output_dir(args, meta)

    audio_embs, midi_embs = get_normal_embeddings(
        model, dataset, device, descriptor, output_dir,
        num_workers=num_workers,
    )

    # Subsample for visualization
    N = audio_embs.size(0)
    if N > n_samples:
        rng = np.random.RandomState(seed)
        idx = rng.choice(N, n_samples, replace=False)
        idx.sort()
        audio_embs = audio_embs[idx]
        midi_embs = midi_embs[idx]
        piece_indices = [dataset.segments[i].piece_idx for i in idx]
    else:
        piece_indices = [seg.piece_idx for seg in dataset.segments]

    return audio_embs.numpy(), midi_embs.numpy(), piece_indices, meta


def compute_projections(audio_embs, midi_embs, seed=42):
    """Compute UMAP and t-SNE projections on combined embeddings."""
    combined = np.concatenate([audio_embs, midi_embs], axis=0)  # [2N, 256]
    N = audio_embs.shape[0]

    projections = {}

    # t-SNE
    try:
        from sklearn.manifold import TSNE
        logger.info("  Computing t-SNE...")
        tsne = TSNE(n_components=2, random_state=seed, perplexity=30, max_iter=1000)
        proj_tsne = tsne.fit_transform(combined)
        projections['tsne'] = {
            'audio': proj_tsne[:N],
            'midi': proj_tsne[N:],
        }
    except ImportError:
        logger.warning("sklearn not available — skipping t-SNE")

    # UMAP
    try:
        import umap
        logger.info("  Computing UMAP...")
        reducer = umap.UMAP(n_components=2, random_state=seed, n_neighbors=15, min_dist=0.1)
        proj_umap = reducer.fit_transform(combined)
        projections['umap'] = {
            'audio': proj_umap[:N],
            'midi': proj_umap[N:],
        }
    except ImportError:
        logger.warning("umap-learn not available — skipping UMAP")

    return projections


def plot_single_model(projections, arm, piece_indices, output_dir):
    """Generate detailed plots for a single model."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available — skipping plots")
        return

    for method, data in projections.items():
        audio_xy = data['audio']
        midi_xy = data['midi']

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        # Left: by modality
        ax = axes[0]
        ax.scatter(audio_xy[:, 0], audio_xy[:, 1], c='cyan', alpha=0.3, s=3, label='Audio')
        ax.scatter(midi_xy[:, 0], midi_xy[:, 1], c='magenta', alpha=0.3, s=3, label='MIDI')
        ax.set_title(f'{arm} — {method.upper()} (by modality)')
        ax.legend(markerscale=5)
        ax.set_xticks([])
        ax.set_yticks([])

        # Right: by piece (top 10 pieces colored)
        ax = axes[1]
        unique_pieces = list(set(piece_indices))
        if len(unique_pieces) > 10:
            from collections import Counter
            piece_counts = Counter(piece_indices)
            top_pieces = [p for p, _ in piece_counts.most_common(10)]
        else:
            top_pieces = unique_pieces

        cmap = plt.cm.tab10
        # Background: gray
        ax.scatter(audio_xy[:, 0], audio_xy[:, 1], c='lightgray', alpha=0.2, s=2)
        ax.scatter(midi_xy[:, 0], midi_xy[:, 1], c='lightgray', alpha=0.2, s=2)

        for i, piece in enumerate(top_pieces):
            mask = [j for j, p in enumerate(piece_indices) if p == piece]
            if mask:
                color = cmap(i / 10)
                ax.scatter(audio_xy[mask, 0], audio_xy[mask, 1],
                           c=[color], alpha=0.6, s=8, marker='o')
                ax.scatter(midi_xy[mask, 0], midi_xy[mask, 1],
                           c=[color], alpha=0.6, s=8, marker='^')

        ax.set_title(f'{arm} — {method.upper()} (by piece, top 10)')
        ax.set_xticks([])
        ax.set_yticks([])

        plt.tight_layout()
        path = output_dir / f'{arm}_{method}_detail.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"  Saved: {path}")


def plot_comparison_grid(all_projections, output_dir):
    """Generate 2x2 comparison grid across all 4 models."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available — skipping comparison grid")
        return

    arms = ['D0', 'd4a4', 'a4r', 'd4-a4r']
    available = [a for a in arms if a in all_projections]

    for method in ['tsne', 'umap']:
        # Check all have this method
        if not all(method in all_projections[a]['projections'] for a in available):
            continue

        n = len(available)
        cols = min(n, 2)
        rows = (n + 1) // 2

        fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 7 * rows))
        if n == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes.reshape(1, -1)

        for idx, arm in enumerate(available):
            r, c = divmod(idx, cols)
            ax = axes[r, c]
            data = all_projections[arm]['projections'][method]

            ax.scatter(data['audio'][:, 0], data['audio'][:, 1],
                       c='cyan', alpha=0.3, s=2, label='Audio')
            ax.scatter(data['midi'][:, 0], data['midi'][:, 1],
                       c='magenta', alpha=0.3, s=2, label='MIDI')
            ax.set_title(f'{arm}', fontsize=14, fontweight='bold')
            ax.legend(markerscale=5, loc='upper right')
            ax.set_xticks([])
            ax.set_yticks([])

        # Hide unused axes
        for idx in range(n, rows * cols):
            r, c = divmod(idx, cols)
            axes[r, c].set_visible(False)

        fig.suptitle(f'Gate 5B — {method.upper()} Comparison', fontsize=16, y=1.02)
        plt.tight_layout()
        path = output_dir / f'comparison_{method}.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved comparison: {path}")


def main():
    parser = argparse.ArgumentParser(description='Gate 5B — Test 10: Visualizations')
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--maestro-dir', type=str, default='data/maestro_v3/maestro-v3.0.0')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--n-samples', type=int, default=2000,
                        help='Number of segments to visualize (subsampled)')

    args = parser.parse_args()

    if not args.model and not args.all:
        parser.error("Provide --model PATH or --all")

    models = {}
    if args.all:
        models = dict(DEFAULT_CHECKPOINTS)
    else:
        models = {'custom': args.model}

    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    all_projections = {}

    for arm, path in models.items():
        if not Path(path).exists():
            logger.warning(f"Checkpoint not found: {path} — skipping {arm}")
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"VISUALIZATIONS: {arm}")
        logger.info(f"{'='*60}")

        start = time.time()

        audio_embs, midi_embs, piece_indices, meta = extract_for_viz(
            path, args.maestro_dir, args.device, args.num_workers,
            n_samples=args.n_samples, seed=args.seed,
        )

        projections = compute_projections(audio_embs, midi_embs, seed=args.seed)
        plot_single_model(projections, arm, piece_indices, VIZ_DIR)

        all_projections[arm] = {
            'projections': projections,
            'piece_indices': piece_indices,
            'meta': meta,
        }

        # Save embeddings as NPZ for future use
        npz_path = VIZ_DIR / f'{arm}_embeddings.npz'
        np.savez_compressed(
            npz_path,
            audio_embs=audio_embs,
            midi_embs=midi_embs,
            piece_indices=np.array(piece_indices),
        )
        logger.info(f"  Saved embeddings: {npz_path}")

        elapsed = time.time() - start
        logger.info(f"  Completed in {elapsed:.0f}s")

    # Comparison grid
    if len(all_projections) > 1:
        logger.info("\nGenerating comparison grid...")
        plot_comparison_grid(all_projections, VIZ_DIR)

    # Save metadata
    meta_path = VIZ_DIR / 'visualization_meta.json'
    meta_out = {}
    for arm, data in all_projections.items():
        meta_out[arm] = {
            'descriptor': data['meta']['descriptor'],
            'epoch': data['meta']['epoch'],
            'best_S': data['meta']['best_S'],
            'n_samples': len(data['piece_indices']),
            'methods': list(data['projections'].keys()),
        }
    with open(meta_path, 'w') as f:
        json.dump(meta_out, f, indent=2)
    logger.info(f"\nMetadata saved: {meta_path}")


if __name__ == '__main__':
    main()
