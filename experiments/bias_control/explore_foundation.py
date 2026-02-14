#!/usr/bin/env python3
"""
Foundation Model Explorer — Retrieval, Visualization, Similarity Analysis.

Extracts embeddings once, caches in memory, runs 6 probes:
  P1: Retrieval examples (top-5 nearest neighbors)
  P2: UMAP 2D visualization (modality, composer, pieces)
  P3: Pair alignment histogram (matched vs random)
  P4: Similarity matrix heatmap (10 pieces × 5 segments)
  P5: Per-piece accuracy ranking
  P6: Cross-modal interpolation

Usage:
    python experiments/bias_control/explore_foundation.py \
        --checkpoint data/.../checkpoint_epochN_base.pt \
        --maestro-dir data/maestro_v3/maestro-v3.0.0 \
        --output Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/resultados_compartir \
        --mode all
"""

import argparse
import json
import logging
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Building blocks (reuse from evaluate_structured_pool.py)
# ---------------------------------------------------------------------------

from experiments.bias_control.evaluate_structured_pool import (
    extract_all_embeddings,
    build_segment_index,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_model_and_dataset(args):
    """Load model from _base.pt checkpoint and validation dataset."""
    from src.bias_control.architectures.cross_modal_model import CrossModalModel
    from src.bias_control.datasets.maestro_segments import MaestroSegmentDataset

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Load model — MUST be a _base.pt (eval-compatible CrossModalModel)
    logger.info(f"Loading model from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)

    # Structural validation: reject wrapper keys (gate42., base_model.)
    wrapper_keys = [k for k in state_dict if k.startswith(('gate42.', 'base_model.', 'dann.'))]
    if wrapper_keys:
        raise RuntimeError(
            f"Checkpoint contains wrapper keys ({wrapper_keys[:3]}...). "
            f"This script requires a _base.pt checkpoint with bare CrossModalModel weights. "
            f"Got: {args.checkpoint}"
        )

    model = CrossModalModel(audio_encoder='lite')
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()

    # Load validation dataset
    logger.info("Loading validation dataset...")
    dataset = MaestroSegmentDataset(
        maestro_dir=Path(args.maestro_dir),
        segment_len=4.0,
        hop=1.0,
        split='validation',
    )
    logger.info(f"Dataset: {len(dataset)} segments")

    return model, dataset, device


def _extract_and_cache(model, dataset, device, batch_size=64, num_workers=8):
    """Extract all embeddings (one-time cost)."""
    logger.info("Extracting all embeddings (one-time cost)...")
    torch.cuda.empty_cache()
    audio_embs, midi_embs = extract_all_embeddings(
        model, dataset, device,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    logger.info(f"Embeddings: audio={audio_embs.shape}, midi={midi_embs.shape}")
    return audio_embs, midi_embs


def _piece_label(seg) -> str:
    """Human-readable label for a segment's piece."""
    return f"{seg.canonical_composer} — {seg.canonical_title}"


# ---------------------------------------------------------------------------
# P1: Retrieval examples
# ---------------------------------------------------------------------------

def probe_retrieval(audio_embs, midi_embs, dataset, index, output_dir, seed=42):
    """P1: Interactive retrieval — top-5 nearest neighbors."""
    logger.info("\n=== P1: Retrieval Examples ===")
    rng = np.random.RandomState(seed)

    audio_norm = F.normalize(audio_embs, dim=1)
    midi_norm = F.normalize(midi_embs, dim=1)

    # Pick 20 random segments (one per piece to avoid duplicates)
    pieces = list(index['by_piece'].keys())
    chosen_pieces = rng.choice(pieces, size=min(20, len(pieces)), replace=False)
    query_indices = [rng.choice(index['by_piece'][p]) for p in chosen_pieces]

    results = []

    for qi in query_indices:
        seg = dataset.segments[qi]
        label = _piece_label(seg)

        # Audio → MIDI (top-5)
        sims_a2m = (audio_norm[qi] @ midi_norm.T).numpy()
        top5_a2m = np.argsort(sims_a2m)[::-1][:5]

        # MIDI → Audio (top-5)
        sims_m2a = (midi_norm[qi] @ audio_norm.T).numpy()
        top5_m2a = np.argsort(sims_m2a)[::-1][:5]

        entry = {
            'query_idx': int(qi),
            'query_piece': label,
            'query_composer': seg.canonical_composer,
            'query_segment': int(seg.segment_idx),
            'a2m_top5': [],
            'm2a_top5': [],
        }

        for rank, idx in enumerate(top5_a2m):
            s = dataset.segments[idx]
            entry['a2m_top5'].append({
                'rank': rank + 1,
                'idx': int(idx),
                'piece': _piece_label(s),
                'similarity': float(sims_a2m[idx]),
                'correct': bool(s.piece_idx == seg.piece_idx),
            })

        for rank, idx in enumerate(top5_m2a):
            s = dataset.segments[idx]
            entry['m2a_top5'].append({
                'rank': rank + 1,
                'idx': int(idx),
                'piece': _piece_label(s),
                'similarity': float(sims_m2a[idx]),
                'correct': bool(s.piece_idx == seg.piece_idx),
            })

        results.append(entry)

    # Compute summary stats
    a2m_top1_correct = sum(1 for r in results if r['a2m_top5'][0]['correct'])
    m2a_top1_correct = sum(1 for r in results if r['m2a_top5'][0]['correct'])
    a2m_top5_correct = sum(1 for r in results if any(x['correct'] for x in r['a2m_top5']))
    m2a_top5_correct = sum(1 for r in results if any(x['correct'] for x in r['m2a_top5']))
    n = len(results)

    summary = {
        'n_queries': n,
        'a2m_top1_accuracy': a2m_top1_correct / n,
        'a2m_top5_accuracy': a2m_top5_correct / n,
        'm2a_top1_accuracy': m2a_top1_correct / n,
        'm2a_top5_accuracy': m2a_top5_correct / n,
    }

    # Save JSON
    with open(output_dir / 'retrieval_examples.json', 'w') as f:
        json.dump({'summary': summary, 'examples': results}, f, indent=2)

    # Save human-readable TXT
    with open(output_dir / 'retrieval_examples.txt', 'w') as f:
        f.write("FOUNDATION MODEL — RETRIEVAL EXAMPLES\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Summary: A2M top-1={summary['a2m_top1_accuracy']:.0%}, "
                f"A2M top-5={summary['a2m_top5_accuracy']:.0%}, "
                f"M2A top-1={summary['m2a_top1_accuracy']:.0%}, "
                f"M2A top-5={summary['m2a_top5_accuracy']:.0%}\n\n")

        for r in results:
            f.write(f"Query: {r['query_piece']} (seg {r['query_segment']})\n")
            f.write("  Audio → MIDI top-5:\n")
            for x in r['a2m_top5']:
                mark = "OK" if x['correct'] else "  "
                f.write(f"    [{mark}] #{x['rank']} sim={x['similarity']:.4f}  {x['piece']}\n")
            f.write("  MIDI → Audio top-5:\n")
            for x in r['m2a_top5']:
                mark = "OK" if x['correct'] else "  "
                f.write(f"    [{mark}] #{x['rank']} sim={x['similarity']:.4f}  {x['piece']}\n")
            f.write("\n")

    logger.info(f"P1 done — A2M top-1: {summary['a2m_top1_accuracy']:.0%}, "
                f"M2A top-1: {summary['m2a_top1_accuracy']:.0%}")
    return summary


# ---------------------------------------------------------------------------
# P2: UMAP visualization
# ---------------------------------------------------------------------------

def probe_umap(audio_embs, midi_embs, dataset, index, output_dir, seed=42):
    """P2: UMAP 2D maps — modality, composer, selected pieces."""
    logger.info("\n=== P2: UMAP Visualization ===")
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import umap

    audio_np = audio_embs.numpy()
    midi_np = midi_embs.numpy()
    combined = np.concatenate([audio_np, midi_np], axis=0)
    n_audio = len(audio_np)

    logger.info(f"Running UMAP on {combined.shape[0]} points...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=seed)
    coords = reducer.fit_transform(combined)

    coords_audio = coords[:n_audio]
    coords_midi = coords[n_audio:]

    # --- (a) Modality ---
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(coords_audio[:, 0], coords_audio[:, 1], c='cyan', s=2, alpha=0.3, label='Audio')
    ax.scatter(coords_midi[:, 0], coords_midi[:, 1], c='magenta', s=2, alpha=0.3, label='MIDI')
    ax.legend(markerscale=5)
    ax.set_title('UMAP — Modality (Audio=cyan, MIDI=magenta)')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    fig.savefig(output_dir / 'umap_modality.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info("Saved umap_modality.png")

    # --- (b) Composer (top 10) ---
    composer_list = [dataset.segments[i].canonical_composer for i in range(len(dataset))]
    composer_counts = Counter(composer_list)
    top10_composers = [c for c, _ in composer_counts.most_common(10)]
    composer_to_id = {c: i for i, c in enumerate(top10_composers)}

    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = plt.cm.get_cmap('tab10', 10)

    # Grey for "other" composers
    other_mask_a = np.array([composer_list[i] not in composer_to_id for i in range(n_audio)])
    ax.scatter(coords_audio[other_mask_a, 0], coords_audio[other_mask_a, 1],
               c='lightgrey', s=2, alpha=0.15, label='Other (audio)')
    other_mask_m = other_mask_a  # same segments
    ax.scatter(coords_midi[other_mask_m, 0], coords_midi[other_mask_m, 1],
               c='lightgrey', s=2, alpha=0.15, label='Other (midi)')

    for comp, cid in composer_to_id.items():
        mask = np.array([composer_list[i] == comp for i in range(n_audio)])
        color = cmap(cid)
        ax.scatter(coords_audio[mask, 0], coords_audio[mask, 1],
                   c=[color], s=4, alpha=0.4, marker='o')
        ax.scatter(coords_midi[mask, 0], coords_midi[mask, 1],
                   c=[color], s=4, alpha=0.4, marker='^', label=comp)

    ax.legend(markerscale=3, fontsize=7, loc='best', ncol=2)
    ax.set_title('UMAP — Top 10 Composers (o=audio, ^=midi)')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    fig.savefig(output_dir / 'umap_composer.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info("Saved umap_composer.png")

    # --- (c) Selected pieces ---
    rng = np.random.RandomState(seed)
    pieces = list(index['by_piece'].keys())
    # Pick 8 pieces with enough segments (>= 10)
    big_pieces = [p for p in pieces if len(index['by_piece'][p]) >= 10]
    chosen = rng.choice(big_pieces, size=min(8, len(big_pieces)), replace=False)

    fig, ax = plt.subplots(figsize=(10, 8))
    # Background: all points in grey
    ax.scatter(coords_audio[:, 0], coords_audio[:, 1], c='lightgrey', s=1, alpha=0.1)
    ax.scatter(coords_midi[:, 0], coords_midi[:, 1], c='lightgrey', s=1, alpha=0.1)

    piece_cmap = plt.cm.get_cmap('Set1', len(chosen))
    for i, p in enumerate(chosen):
        seg_idxs = np.array(index['by_piece'][p])
        seg = dataset.segments[seg_idxs[0]]
        short_title = seg.canonical_title[:30]
        color = piece_cmap(i)

        ax.scatter(coords_audio[seg_idxs, 0], coords_audio[seg_idxs, 1],
                   c=[color], s=8, alpha=0.6, marker='o')
        ax.scatter(coords_midi[seg_idxs, 0], coords_midi[seg_idxs, 1],
                   c=[color], s=8, alpha=0.6, marker='^', label=short_title)

    ax.legend(markerscale=2, fontsize=6, loc='best')
    ax.set_title('UMAP — Selected Pieces (o=audio, ^=midi)')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    fig.savefig(output_dir / 'umap_pieces.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info("Saved umap_pieces.png")

    return {'status': 'ok', 'n_points': combined.shape[0]}


# ---------------------------------------------------------------------------
# P3: Pair alignment histogram
# ---------------------------------------------------------------------------

def probe_pair_alignment(audio_embs, midi_embs, dataset, index, output_dir, seed=42):
    """P3: Cosine similarity histogram — matched vs random pairs."""
    logger.info("\n=== P3: Pair Alignment Histogram ===")
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    rng = np.random.RandomState(seed)
    n = len(audio_embs)

    audio_norm = F.normalize(audio_embs, dim=1).numpy()
    midi_norm = F.normalize(midi_embs, dim=1).numpy()

    n_pairs = min(200, n)

    # Matched pairs: same index = same segment
    matched_indices = rng.choice(n, size=n_pairs, replace=False)
    matched_sims = np.array([
        np.dot(audio_norm[i], midi_norm[i]) for i in matched_indices
    ])

    # Random pairs: audio_i vs midi_j (i != j)
    random_i = rng.choice(n, size=n_pairs, replace=True)
    random_j = rng.choice(n, size=n_pairs, replace=True)
    # Ensure i != j
    for k in range(n_pairs):
        while random_j[k] == random_i[k]:
            random_j[k] = rng.randint(0, n)
    random_sims = np.array([
        np.dot(audio_norm[random_i[k]], midi_norm[random_j[k]]) for k in range(n_pairs)
    ])

    # Stats
    stats = {
        'matched': {
            'mean': float(np.mean(matched_sims)),
            'std': float(np.std(matched_sims)),
            'median': float(np.median(matched_sims)),
            'min': float(np.min(matched_sims)),
            'max': float(np.max(matched_sims)),
        },
        'random': {
            'mean': float(np.mean(random_sims)),
            'std': float(np.std(random_sims)),
            'median': float(np.median(random_sims)),
            'min': float(np.min(random_sims)),
            'max': float(np.max(random_sims)),
        },
    }
    gap = stats['matched']['mean'] - stats['random']['mean']
    stats['gap_mean'] = float(gap)

    # Cohen's d
    pooled_std = np.sqrt((np.std(matched_sims)**2 + np.std(random_sims)**2) / 2)
    stats['cohens_d'] = float(gap / pooled_std) if pooled_std > 0 else 0.0

    with open(output_dir / 'pair_alignment_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)

    # Histogram
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(matched_sims, bins=40, alpha=0.6, label=f"Matched (mean={stats['matched']['mean']:.3f})", color='green')
    ax.hist(random_sims, bins=40, alpha=0.6, label=f"Random (mean={stats['random']['mean']:.3f})", color='red')
    ax.axvline(stats['matched']['mean'], color='darkgreen', linestyle='--', linewidth=1)
    ax.axvline(stats['random']['mean'], color='darkred', linestyle='--', linewidth=1)
    ax.set_xlabel('Cosine Similarity')
    ax.set_ylabel('Count')
    ax.set_title(f"Pair Alignment — gap={gap:.3f}, Cohen's d={stats['cohens_d']:.2f}")
    ax.legend()
    fig.savefig(output_dir / 'pair_alignment_histogram.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    logger.info(f"P3 done — matched mean={stats['matched']['mean']:.3f}, "
                f"random mean={stats['random']['mean']:.3f}, gap={gap:.3f}")
    return stats


# ---------------------------------------------------------------------------
# P4: Similarity matrix
# ---------------------------------------------------------------------------

def probe_similarity_matrix(audio_embs, midi_embs, dataset, index, output_dir, seed=42):
    """P4: Heatmap of cosine similarity for 10 pieces × 5 segments."""
    logger.info("\n=== P4: Similarity Matrix ===")
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    rng = np.random.RandomState(seed)

    audio_norm = F.normalize(audio_embs, dim=1)
    midi_norm = F.normalize(midi_embs, dim=1)

    # Pick 10 pieces with >= 5 segments
    pieces = [p for p in index['by_piece'] if len(index['by_piece'][p]) >= 5]
    if len(pieces) == 0:
        logger.warning("P4 skipped — no pieces with >= 5 segments")
        return {'status': 'skipped', 'reason': 'not_enough_segments_per_piece'}
    chosen_pieces = sorted(rng.choice(pieces, size=min(10, len(pieces)), replace=False))

    seg_indices = []
    labels = []
    for p in chosen_pieces:
        segs = sorted(index['by_piece'][p])[:5]  # first 5 segments
        seg_indices.extend(segs)
        seg = dataset.segments[segs[0]]
        short = seg.canonical_title[:20]
        for s_idx in segs:
            labels.append(f"{short}_{dataset.segments[s_idx].segment_idx}")

    seg_indices = np.array(seg_indices)
    n = len(seg_indices)

    # Compute cosine similarity: audio_i vs midi_j
    audio_sub = audio_norm[seg_indices]  # [n, D]
    midi_sub = midi_norm[seg_indices]    # [n, D]
    sim_matrix = (audio_sub @ midi_sub.T).numpy()  # [n, n]

    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(sim_matrix, cmap='RdYlGn', vmin=-0.2, vmax=1.0, aspect='equal')
    fig.colorbar(im, ax=ax, label='Cosine Similarity')

    # Draw piece boundaries
    pos = 0
    for p in chosen_pieces:
        count = min(5, len(index['by_piece'][p]))
        ax.axhline(pos - 0.5, color='black', linewidth=0.5, alpha=0.5)
        ax.axvline(pos - 0.5, color='black', linewidth=0.5, alpha=0.5)
        pos += count

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=90, fontsize=5)
    ax.set_yticklabels(labels, fontsize=5)
    ax.set_xlabel('MIDI segments')
    ax.set_ylabel('Audio segments')
    ax.set_title('Cross-Modal Similarity Matrix (10 pieces × 5 segments)')
    fig.savefig(output_dir / 'similarity_matrix.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Stats: mean diagonal (matched) vs off-diagonal (all cross-pairs)
    diag_sims = np.diag(sim_matrix)
    off_diag_sims = []
    for i in range(n):
        for j in range(n):
            if i != j:
                off_diag_sims.append(sim_matrix[i, j])

    stats = {
        'n_pieces': len(chosen_pieces),
        'n_segments': n,
        'diagonal_mean': float(np.mean(diag_sims)),
        'diagonal_std': float(np.std(diag_sims)),
        'off_diagonal_mean': float(np.mean(off_diag_sims)),
        'off_diagonal_std': float(np.std(off_diag_sims)),
    }

    logger.info(f"P4 done — diagonal mean={stats['diagonal_mean']:.3f}, "
                f"off-diag mean={stats['off_diagonal_mean']:.3f}")
    return stats


# ---------------------------------------------------------------------------
# P5: Per-piece accuracy
# ---------------------------------------------------------------------------

def probe_per_piece(audio_embs, midi_embs, dataset, index, output_dir):
    """P5: For each piece, does the NN of audio find its own MIDI?"""
    logger.info("\n=== P5: Per-Piece Accuracy ===")

    audio_norm = F.normalize(audio_embs, dim=1)
    midi_norm = F.normalize(midi_embs, dim=1)

    piece_results = []

    for piece_idx, seg_indices in index['by_piece'].items():
        seg = dataset.segments[seg_indices[0]]
        n_segs = len(seg_indices)

        correct = 0
        for si in seg_indices:
            # Find nearest MIDI for this audio
            sims = (audio_norm[si] @ midi_norm.T).numpy()
            nn_idx = np.argmax(sims)
            nn_piece = dataset.segments[nn_idx].piece_idx
            if nn_piece == piece_idx:
                correct += 1

        accuracy = correct / n_segs if n_segs > 0 else 0.0
        piece_results.append({
            'piece_idx': int(piece_idx),
            'composer': seg.canonical_composer,
            'title': seg.canonical_title,
            'n_segments': n_segs,
            'correct': correct,
            'accuracy': accuracy,
        })

    # Sort by accuracy descending
    piece_results.sort(key=lambda x: x['accuracy'], reverse=True)

    overall_acc = sum(r['correct'] for r in piece_results) / sum(r['n_segments'] for r in piece_results)

    # Correlation analysis: accuracy vs composer / n_segments (proxy for duration)
    correlations = {}

    # Composer breakdown: mean accuracy per composer
    composer_accs = defaultdict(list)
    for r in piece_results:
        composer_accs[r['composer']].append(r['accuracy'])
    composer_summary = {
        c: {'mean_acc': float(np.mean(accs)), 'n_pieces': len(accs)}
        for c, accs in sorted(composer_accs.items(), key=lambda x: -np.mean(x[1]))
    }
    correlations['by_composer'] = composer_summary

    # Correlation: accuracy vs n_segments (duration proxy)
    accs = np.array([r['accuracy'] for r in piece_results])
    n_segs_arr = np.array([r['n_segments'] for r in piece_results])
    if len(accs) > 2 and np.std(accs) > 0 and np.std(n_segs_arr) > 0:
        corr_duration = float(np.corrcoef(accs, n_segs_arr)[0, 1])
    else:
        corr_duration = 0.0
    correlations['acc_vs_n_segments_corr'] = corr_duration

    # Correlation: accuracy vs segment density (segments/piece, already = n_segments)
    correlations['note'] = (
        "n_segments is proportional to piece duration (4s segments, 1s hop). "
        "Positive correlation means longer pieces are easier to retrieve."
    )

    # Save JSON
    output = {
        'overall_accuracy': overall_acc,
        'n_pieces': len(piece_results),
        'correlations': correlations,
        'pieces': piece_results,
    }
    with open(output_dir / 'per_piece_accuracy.json', 'w') as f:
        json.dump(output, f, indent=2)

    # Save human-readable ranking
    with open(output_dir / 'per_piece_ranking.txt', 'w') as f:
        f.write("FOUNDATION MODEL — PER-PIECE ACCURACY RANKING\n")
        f.write("=" * 80 + "\n")
        f.write(f"Overall accuracy: {overall_acc:.1%} ({sum(r['correct'] for r in piece_results)}"
                f"/{sum(r['n_segments'] for r in piece_results)})\n\n")

        f.write("--- Correlation: accuracy vs duration (n_segments) ---\n")
        f.write(f"  Pearson r = {corr_duration:.3f}\n\n")

        f.write("--- Accuracy by Composer ---\n")
        for comp, info in composer_summary.items():
            f.write(f"  {comp[:30]:<32} mean_acc={info['mean_acc']:.0%}  ({info['n_pieces']} pieces)\n")
        f.write("\n")

        f.write(f"{'Rank':<5} {'Acc':>6} {'Corr':>5} {'Segs':>5}  {'Composer':<25} {'Title'}\n")
        f.write("-" * 80 + "\n")
        for rank, r in enumerate(piece_results, 1):
            f.write(f"{rank:<5} {r['accuracy']:>5.0%} {r['correct']:>5}/{r['n_segments']:<5}"
                    f"  {r['composer'][:25]:<25} {r['title'][:40]}\n")

    logger.info(f"P5 done — overall {overall_acc:.1%}, "
                f"best={piece_results[0]['accuracy']:.0%} ({piece_results[0]['title'][:30]}), "
                f"worst={piece_results[-1]['accuracy']:.0%} ({piece_results[-1]['title'][:30]}), "
                f"acc↔duration r={corr_duration:.3f}")
    return output


# ---------------------------------------------------------------------------
# P6: Cross-modal interpolation
# ---------------------------------------------------------------------------

def probe_interpolation(audio_embs, midi_embs, dataset, index, output_dir, seed=42):
    """P6: Interpolate between two contrastive pieces, find nearest MIDI."""
    logger.info("\n=== P6: Cross-Modal Interpolation ===")
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    rng = np.random.RandomState(seed)

    audio_norm = F.normalize(audio_embs, dim=1)
    midi_norm = F.normalize(midi_embs, dim=1)

    # Find two contrastive pieces (different composers)
    composers = list(index['by_composer'].keys())
    if len(composers) < 2:
        logger.warning("Not enough composers for interpolation")
        return {'status': 'skipped'}

    # Pick two composers with many segments
    composer_sizes = [(c, len(index['by_composer'][c])) for c in composers]
    composer_sizes.sort(key=lambda x: x[1], reverse=True)
    comp_a, comp_b = composer_sizes[0][0], composer_sizes[1][0]

    # Pick one segment from each
    idx_a = rng.choice(index['by_composer'][comp_a])
    idx_b = rng.choice(index['by_composer'][comp_b])
    seg_a = dataset.segments[idx_a]
    seg_b = dataset.segments[idx_b]

    emb_a = audio_norm[idx_a]
    emb_b = audio_norm[idx_b]

    # Interpolation: 11 steps from alpha=0 (A) to alpha=1 (B)
    alphas = np.linspace(0.0, 1.0, 11)
    interp_results = []

    for alpha in alphas:
        interp = (1 - alpha) * emb_a + alpha * emb_b
        interp = F.normalize(interp, dim=0)

        # Find nearest MIDI
        sims = (interp @ midi_norm.T).numpy()
        nn_idx = np.argmax(sims)
        nn_seg = dataset.segments[nn_idx]

        interp_results.append({
            'alpha': float(alpha),
            'nn_idx': int(nn_idx),
            'nn_piece': _piece_label(nn_seg),
            'nn_composer': nn_seg.canonical_composer,
            'nn_similarity': float(sims[nn_idx]),
            'sim_to_A': float((interp @ midi_norm[idx_a]).item()),
            'sim_to_B': float((interp @ midi_norm[idx_b]).item()),
        })

    output = {
        'piece_A': _piece_label(seg_a),
        'piece_B': _piece_label(seg_b),
        'idx_A': int(idx_a),
        'idx_B': int(idx_b),
        'steps': interp_results,
    }

    with open(output_dir / 'interpolation_results.json', 'w') as f:
        json.dump(output, f, indent=2)

    # Plot: similarity to A and B as function of alpha
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    sim_A = [r['sim_to_A'] for r in interp_results]
    sim_B = [r['sim_to_B'] for r in interp_results]
    nn_sim = [r['nn_similarity'] for r in interp_results]

    ax1.plot(alphas, sim_A, 'o-', label=f'Sim to A ({seg_a.canonical_composer[:15]})', color='blue')
    ax1.plot(alphas, sim_B, 's-', label=f'Sim to B ({seg_b.canonical_composer[:15]})', color='red')
    ax1.plot(alphas, nn_sim, 'd--', label='Sim to NN', color='green', alpha=0.7)
    ax1.set_xlabel('alpha (0=A, 1=B)')
    ax1.set_ylabel('Cosine Similarity')
    ax1.set_title('Interpolation Similarities')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Bar chart: which composer is the NN at each step
    nn_composers = [r['nn_composer'] for r in interp_results]
    unique_composers = list(set(nn_composers))
    colors_map = {comp_a: 'blue', comp_b: 'red'}
    bar_colors = [colors_map.get(c, 'grey') for c in nn_composers]

    ax2.bar(range(len(alphas)), [1]*len(alphas), color=bar_colors, alpha=0.7)
    ax2.set_xticks(range(len(alphas)))
    ax2.set_xticklabels([f'{a:.1f}' for a in alphas], fontsize=8)
    ax2.set_xlabel('alpha')
    ax2.set_title(f'NN Composer (blue={comp_a[:15]}, red={comp_b[:15]}, grey=other)')
    ax2.set_yticks([])

    fig.suptitle(f'Interpolation: {seg_a.canonical_title[:30]} ↔ {seg_b.canonical_title[:30]}', fontsize=10)
    fig.savefig(output_dir / 'interpolation_plot.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    logger.info(f"P6 done — {_piece_label(seg_a)[:40]} ↔ {_piece_label(seg_b)[:40]}")
    return output


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

PROBES = {
    'retrieval': probe_retrieval,
    'umap': probe_umap,
    'pairs': probe_pair_alignment,
    'similarity': probe_similarity_matrix,
    'per_piece': probe_per_piece,
    'interpolation': probe_interpolation,
}


def main():
    parser = argparse.ArgumentParser(description='Explore Foundation Model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to _base.pt checkpoint (CrossModalModel)')
    parser.add_argument('--maestro-dir', type=str, default='data/maestro_v3/maestro-v3.0.0')
    parser.add_argument('--output', type=str,
                        default='Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/resultados_compartir')
    parser.add_argument('--mode', type=str, default='all',
                        help=f"Probes to run: {', '.join(PROBES.keys())}, all")
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Validate checkpoint: warn if name doesn't suggest _base.pt
    # (structural validation happens inside _load_model_and_dataset via strict=True)
    ckpt_name = Path(args.checkpoint).name
    if '_base' not in ckpt_name:
        logger.warning(
            f"Checkpoint '{ckpt_name}' does not contain '_base' in its name. "
            "This script expects eval-compatible _base.pt checkpoints. "
            "Loading will fail with strict=True if the state_dict has wrapper keys."
        )

    # Determine which probes to run
    if args.mode == 'all':
        modes = list(PROBES.keys())
    else:
        modes = [m.strip() for m in args.mode.split(',')]
        for m in modes:
            if m not in PROBES:
                logger.error(f"Unknown mode: {m}. Available: {', '.join(PROBES.keys())}")
                sys.exit(1)

    # Load model and dataset
    model, dataset, device = _load_model_and_dataset(args)

    # Build index
    logger.info("Building segment index...")
    index = build_segment_index(dataset)
    logger.info(f"Index: {len(index['by_piece'])} pieces, {len(index['by_composer'])} composers")

    # Extract embeddings (one-time cost)
    audio_embs, midi_embs = _extract_and_cache(
        model, dataset, device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Free GPU memory — all probes work on CPU tensors
    del model
    torch.cuda.empty_cache()

    # Run probes
    all_results = {}
    t_start = time.time()

    for mode in modes:
        t0 = time.time()
        probe_fn = PROBES[mode]

        if mode == 'per_piece':
            result = probe_fn(audio_embs, midi_embs, dataset, index, output_dir)
        else:
            result = probe_fn(audio_embs, midi_embs, dataset, index, output_dir, seed=args.seed)

        all_results[mode] = result
        logger.info(f"  [{mode}] completed in {time.time()-t0:.1f}s")

    elapsed = time.time() - t_start
    logger.info(f"\nAll probes completed in {elapsed:.0f}s")

    # Save combined summary
    summary = {
        'checkpoint': args.checkpoint,
        'n_segments': len(dataset),
        'n_pieces': len(index['by_piece']),
        'n_composers': len(index['by_composer']),
        'embedding_dim': int(audio_embs.shape[1]),
        'probes_run': modes,
        'elapsed_seconds': elapsed,
        'results': {},
    }
    for mode, result in all_results.items():
        if isinstance(result, dict):
            summary['results'][mode] = result

    with open(output_dir / 'explore_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info(f"Summary saved to {output_dir / 'explore_summary.json'}")
    logger.info(f"All outputs in: {output_dir}/")


if __name__ == '__main__':
    main()
