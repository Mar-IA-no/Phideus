#!/usr/bin/env python3
"""
Gate 2.5: Embedding Analysis (Diagnostic)

Objectives:
1. Visualize embeddings with t-SNE/UMAP
2. Analyze clustering by modality vs piece
3. Detect collapse or modal shortcuts
4. Inform decision for Gate 3 (DANN)

This is a diagnostic gate - no GO/NO-GO, but informs next steps.

Usage:
    python experiments/bias_control/gate2_5_embedding_analysis.py \
        --model data/training_outputs/bias_control/gate2/best_model.pt \
        --maestro-dir data/maestro_v3/maestro-v3.0.0 \
        --output data/evaluations/bias_control/gate2_5
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.bias_control.datasets.maestro_segments import (
    MaestroSegmentDataset,
    collate_segments,
)
from src.bias_control.architectures.cross_modal_model import CrossModalModel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_embeddings(
    model: CrossModalModel,
    dataset: MaestroSegmentDataset,
    device: str,
    max_samples: int = 1000,
    batch_size: int = 32,
) -> Dict[str, np.ndarray]:
    """
    Extract embeddings from model.

    Returns:
        Dict with:
            audio_embeddings: [N, D]
            midi_embeddings: [N, D]
            piece_indices: [N]
            segment_indices: [N]
            composers: [N] (string indices)
    """
    from torch.utils.data import Subset

    indices = list(range(min(len(dataset), max_samples)))
    subset = Subset(dataset, indices)

    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_segments,
    )

    audio_embs = []
    midi_embs = []
    piece_idxs = []
    segment_idxs = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting embeddings"):
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            audio_emb, midi_emb = model.get_embeddings(batch)

            audio_embs.append(audio_emb.cpu().numpy())
            midi_embs.append(midi_emb.cpu().numpy())
            piece_idxs.append(batch['piece_idx'].numpy())
            segment_idxs.append(batch['segment_idx'].numpy())

    return {
        'audio_embeddings': np.concatenate(audio_embs, axis=0),
        'midi_embeddings': np.concatenate(midi_embs, axis=0),
        'piece_indices': np.concatenate(piece_idxs, axis=0),
        'segment_indices': np.concatenate(segment_idxs, axis=0),
    }


def compute_tsne(
    embeddings: np.ndarray,
    perplexity: int = 30,
    n_components: int = 2,
) -> np.ndarray:
    """Compute t-SNE embedding."""
    try:
        from sklearn.manifold import TSNE
        tsne = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            random_state=42,
            n_iter=1000,
        )
        return tsne.fit_transform(embeddings)
    except ImportError:
        logger.warning("sklearn not available for t-SNE")
        return None


def compute_umap(
    embeddings: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    n_components: int = 2,
) -> np.ndarray:
    """Compute UMAP embedding."""
    try:
        import umap
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=n_components,
            random_state=42,
        )
        return reducer.fit_transform(embeddings)
    except ImportError:
        logger.warning("umap-learn not available")
        return None


def compute_silhouette_scores(
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Compute silhouette score for clustering quality."""
    try:
        from sklearn.metrics import silhouette_score
        # Handle case with too few unique labels
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
            return 0.0
        return silhouette_score(embeddings, labels)
    except ImportError:
        return 0.0


def analyze_modal_separation(
    audio_embeddings: np.ndarray,
    midi_embeddings: np.ndarray,
) -> Dict[str, float]:
    """
    Analyze how separated audio and MIDI embeddings are.

    If highly separated → modal shortcut (need DANN)
    If mixed → good cross-modal alignment
    """
    # Combine embeddings
    combined = np.concatenate([audio_embeddings, midi_embeddings], axis=0)
    modality_labels = np.concatenate([
        np.zeros(len(audio_embeddings)),
        np.ones(len(midi_embeddings)),
    ])

    # Silhouette by modality
    silhouette_modality = compute_silhouette_scores(combined, modality_labels)

    # Linear separability (train simple classifier)
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score

        clf = LogisticRegression(max_iter=1000, random_state=42)
        scores = cross_val_score(clf, combined, modality_labels, cv=5)
        linear_accuracy = scores.mean()
    except ImportError:
        linear_accuracy = 0.5

    # Centroid distance
    audio_centroid = audio_embeddings.mean(axis=0)
    midi_centroid = midi_embeddings.mean(axis=0)
    centroid_distance = np.linalg.norm(audio_centroid - midi_centroid)

    # Intra-modal variance
    audio_var = np.var(audio_embeddings, axis=0).mean()
    midi_var = np.var(midi_embeddings, axis=0).mean()

    return {
        'silhouette_by_modality': float(silhouette_modality),
        'linear_separability': float(linear_accuracy),
        'centroid_distance': float(centroid_distance),
        'audio_variance': float(audio_var),
        'midi_variance': float(midi_var),
    }


def analyze_piece_clustering(
    audio_embeddings: np.ndarray,
    midi_embeddings: np.ndarray,
    piece_indices: np.ndarray,
) -> Dict[str, float]:
    """
    Analyze how well embeddings cluster by piece.

    Good piece clustering → useful signal
    Poor piece clustering → model not learning piece identity
    """
    # Silhouette by piece (audio only)
    unique_pieces = np.unique(piece_indices)
    if len(unique_pieces) < 2:
        return {'silhouette_by_piece_audio': 0.0, 'silhouette_by_piece_midi': 0.0}

    silhouette_audio = compute_silhouette_scores(audio_embeddings, piece_indices)
    silhouette_midi = compute_silhouette_scores(midi_embeddings, piece_indices)

    # Cross-modal piece matching
    # For each piece, how similar are audio and MIDI centroids?
    piece_similarities = []
    for piece_idx in unique_pieces:
        mask = piece_indices == piece_idx
        if mask.sum() < 2:
            continue

        audio_centroid = audio_embeddings[mask].mean(axis=0)
        midi_centroid = midi_embeddings[mask].mean(axis=0)

        # Cosine similarity
        sim = np.dot(audio_centroid, midi_centroid) / (
            np.linalg.norm(audio_centroid) * np.linalg.norm(midi_centroid) + 1e-8
        )
        piece_similarities.append(sim)

    mean_piece_similarity = np.mean(piece_similarities) if piece_similarities else 0.0

    return {
        'silhouette_by_piece_audio': float(silhouette_audio),
        'silhouette_by_piece_midi': float(silhouette_midi),
        'mean_piece_similarity': float(mean_piece_similarity),
        'n_pieces_analyzed': len(unique_pieces),
    }


def analyze_variance_per_dimension(
    embeddings: np.ndarray,
    name: str = "embeddings",
) -> Dict[str, float]:
    """Analyze variance distribution across dimensions."""
    var_per_dim = np.var(embeddings, axis=0)

    return {
        f'{name}_var_mean': float(var_per_dim.mean()),
        f'{name}_var_std': float(var_per_dim.std()),
        f'{name}_var_min': float(var_per_dim.min()),
        f'{name}_var_max': float(var_per_dim.max()),
        f'{name}_dead_dims': int((var_per_dim < 0.01).sum()),  # Nearly zero variance
        f'{name}_total_dims': len(var_per_dim),
    }


def create_visualizations(
    data: Dict[str, np.ndarray],
    output_dir: Path,
) -> List[str]:
    """Create and save visualizations."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available for visualizations")
        return []

    saved_plots = []
    output_dir.mkdir(parents=True, exist_ok=True)

    # Combined embeddings for visualization
    combined = np.concatenate([
        data['audio_embeddings'],
        data['midi_embeddings']
    ], axis=0)

    modality_labels = np.concatenate([
        np.zeros(len(data['audio_embeddings'])),
        np.ones(len(data['midi_embeddings'])),
    ])

    piece_labels = np.concatenate([
        data['piece_indices'],
        data['piece_indices'],
    ])

    # 1. t-SNE colored by modality
    logger.info("Computing t-SNE...")
    tsne_coords = compute_tsne(combined)

    if tsne_coords is not None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # By modality
        scatter1 = axes[0].scatter(
            tsne_coords[:, 0], tsne_coords[:, 1],
            c=modality_labels, cmap='coolwarm', alpha=0.5, s=10
        )
        axes[0].set_title('t-SNE: Colored by Modality')
        axes[0].set_xlabel('t-SNE 1')
        axes[0].set_ylabel('t-SNE 2')
        plt.colorbar(scatter1, ax=axes[0], label='Modality (0=Audio, 1=MIDI)')

        # By piece
        scatter2 = axes[1].scatter(
            tsne_coords[:, 0], tsne_coords[:, 1],
            c=piece_labels % 20, cmap='tab20', alpha=0.5, s=10
        )
        axes[1].set_title('t-SNE: Colored by Piece (mod 20)')
        axes[1].set_xlabel('t-SNE 1')
        axes[1].set_ylabel('t-SNE 2')

        plt.tight_layout()
        tsne_path = output_dir / 'tsne_visualization.png'
        plt.savefig(tsne_path, dpi=150, bbox_inches='tight')
        plt.close()
        saved_plots.append(str(tsne_path))
        logger.info(f"Saved t-SNE visualization to {tsne_path}")

    # 2. Variance per dimension histogram
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    audio_var = np.var(data['audio_embeddings'], axis=0)
    midi_var = np.var(data['midi_embeddings'], axis=0)

    axes[0].hist(audio_var, bins=50, alpha=0.7, label='Audio')
    axes[0].hist(midi_var, bins=50, alpha=0.7, label='MIDI')
    axes[0].set_xlabel('Variance')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Variance per Dimension')
    axes[0].legend()

    # Sorted variance
    axes[1].plot(sorted(audio_var, reverse=True), label='Audio')
    axes[1].plot(sorted(midi_var, reverse=True), label='MIDI')
    axes[1].set_xlabel('Dimension (sorted)')
    axes[1].set_ylabel('Variance')
    axes[1].set_title('Sorted Variance per Dimension')
    axes[1].legend()

    plt.tight_layout()
    var_path = output_dir / 'variance_analysis.png'
    plt.savefig(var_path, dpi=150, bbox_inches='tight')
    plt.close()
    saved_plots.append(str(var_path))

    return saved_plots


def run_analysis(
    model_path: Path,
    maestro_dir: Path,
    output_dir: Path,
    segment_len: float = 8.0,
    hop: float = 2.0,
    max_samples: int = 1000,
    device: Optional[str] = None,
) -> Dict:
    """
    Run embedding analysis.

    Returns:
        Analysis results with recommendations
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        'gate': '2.5',
        'name': 'Embedding Analysis',
        'config': {
            'model_path': str(model_path),
            'max_samples': max_samples,
        },
    }

    # Load model
    logger.info(f"Loading model from {model_path}")
    model = CrossModalModel(
        audio_encoder='lite',
        use_dann=False,
        device=device,
    )
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    # Load dataset
    dataset = MaestroSegmentDataset(
        maestro_dir=maestro_dir,
        segment_len=segment_len,
        hop=hop,
        split='validation',
    )

    # Extract embeddings
    logger.info("Extracting embeddings...")
    data = extract_embeddings(model, dataset, device, max_samples)

    # Save embeddings
    np.savez(
        output_dir / 'embeddings.npz',
        **data
    )

    # Analyze modal separation
    logger.info("Analyzing modal separation...")
    modal_analysis = analyze_modal_separation(
        data['audio_embeddings'],
        data['midi_embeddings'],
    )
    results['modal_separation'] = modal_analysis

    # Analyze piece clustering
    logger.info("Analyzing piece clustering...")
    piece_analysis = analyze_piece_clustering(
        data['audio_embeddings'],
        data['midi_embeddings'],
        data['piece_indices'],
    )
    results['piece_clustering'] = piece_analysis

    # Analyze variance
    logger.info("Analyzing variance...")
    audio_var = analyze_variance_per_dimension(data['audio_embeddings'], 'audio')
    midi_var = analyze_variance_per_dimension(data['midi_embeddings'], 'midi')
    results['variance'] = {**audio_var, **midi_var}

    # Create visualizations
    logger.info("Creating visualizations...")
    saved_plots = create_visualizations(data, output_dir)
    results['visualizations'] = saved_plots

    # Generate recommendations
    recommendations = []

    # Check for modal shortcut
    if modal_analysis['linear_separability'] > 0.90:
        recommendations.append({
            'issue': 'Strong modal shortcut detected',
            'evidence': f"Linear separability = {modal_analysis['linear_separability']:.1%}",
            'action': 'PROCEED TO GATE 3 (DANN) to force modal-agnostic embeddings',
        })
    elif modal_analysis['linear_separability'] > 0.70:
        recommendations.append({
            'issue': 'Moderate modal separation',
            'evidence': f"Linear separability = {modal_analysis['linear_separability']:.1%}",
            'action': 'Consider GATE 3 (DANN) for improvement',
        })
    else:
        recommendations.append({
            'issue': 'Low modal separation (good)',
            'evidence': f"Linear separability = {modal_analysis['linear_separability']:.1%}",
            'action': 'Model already modal-agnostic, may skip GATE 3',
        })

    # Check for collapse
    if results['variance']['audio_dead_dims'] > 10 or results['variance']['midi_dead_dims'] > 10:
        recommendations.append({
            'issue': 'Partial collapse detected',
            'evidence': f"Dead dims: audio={results['variance']['audio_dead_dims']}, midi={results['variance']['midi_dead_dims']}",
            'action': 'Review VICReg variance loss weight',
        })

    # Check piece clustering
    if piece_analysis['silhouette_by_piece_audio'] < 0.1:
        recommendations.append({
            'issue': 'Poor piece clustering',
            'evidence': f"Silhouette = {piece_analysis['silhouette_by_piece_audio']:.3f}",
            'action': 'Model not learning piece identity, consider architecture changes',
        })

    results['recommendations'] = recommendations

    return results


def main():
    parser = argparse.ArgumentParser(description="Gate 2.5: Embedding Analysis")
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained model checkpoint'
    )
    parser.add_argument(
        '--maestro-dir',
        type=str,
        required=True,
        help='Path to MAESTRO dataset'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/evaluations/bias_control/gate2_5',
        help='Output directory'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=1000,
        help='Maximum samples to analyze'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device'
    )

    args = parser.parse_args()

    model_path = Path(args.model)
    maestro_dir = Path(args.maestro_dir)
    output_dir = Path(args.output)

    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        sys.exit(1)

    # Run analysis
    results = run_analysis(
        model_path=model_path,
        maestro_dir=maestro_dir,
        output_dir=output_dir,
        max_samples=args.max_samples,
        device=args.device,
    )

    # Save results
    results_path = output_dir / "gate2_5_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("GATE 2.5: EMBEDDING ANALYSIS - RESULTS")
    print("=" * 60)

    print(f"\n1. Modal Separation:")
    print(f"   Linear separability: {results['modal_separation']['linear_separability']:.1%}")
    print(f"   Silhouette by modality: {results['modal_separation']['silhouette_by_modality']:.3f}")
    print(f"   Centroid distance: {results['modal_separation']['centroid_distance']:.3f}")

    print(f"\n2. Piece Clustering:")
    print(f"   Silhouette (audio): {results['piece_clustering']['silhouette_by_piece_audio']:.3f}")
    print(f"   Silhouette (midi): {results['piece_clustering']['silhouette_by_piece_midi']:.3f}")
    print(f"   Mean piece similarity: {results['piece_clustering']['mean_piece_similarity']:.3f}")

    print(f"\n3. Variance Analysis:")
    print(f"   Audio dead dims: {results['variance']['audio_dead_dims']}/{results['variance']['audio_total_dims']}")
    print(f"   MIDI dead dims: {results['variance']['midi_dead_dims']}/{results['variance']['midi_total_dims']}")

    print(f"\n4. Recommendations:")
    for rec in results['recommendations']:
        print(f"   - {rec['issue']}")
        print(f"     Evidence: {rec['evidence']}")
        print(f"     Action: {rec['action']}")

    print(f"\n" + "=" * 60)
    print("(This is a diagnostic gate - no GO/NO-GO decision)")
    print("=" * 60)
    print(f"\nResults saved to: {results_path}")
    if results['visualizations']:
        print(f"Visualizations: {results['visualizations']}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
