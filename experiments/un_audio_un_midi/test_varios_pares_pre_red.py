#!/usr/bin/env python3
"""
Test Varios Pares - Pre-Red Validation
========================================

Runs 3 types of validation on 10 audio-MIDI pairs:
1. Token Compatibility: distributions match between audio and MIDI
2. Retrieval without NN: Shazam-style matching
3. Self vs Cross: aligned vs within-piece-different-time

Following GPT5.2Think recommendations:
- Segment each piece into 20s segments with 10s hop
- ~11 segments per piece → ~110 total queries
- Evaluate retrieval with hard negatives
"""

from __future__ import annotations

import sys
import json
import time
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import entropy

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import librosa
except ImportError:
    print("ERROR: librosa required")
    sys.exit(1)

from src.utils.midi_utils import parse_midi, Note

# Import parallel extraction functions from V2
from test_single_pair_v2_parallel import (
    extract_audio_constellation_parallel,
    extract_midi_constellation_parallel,
    compare_histograms,
    N_WORKERS,
)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Segment:
    """A segment of audio or MIDI."""
    piece_id: int
    piece_name: str
    segment_id: int
    start_time: float
    end_time: float
    tokens: List[np.ndarray]  # List of token arrays per frame
    histogram: np.ndarray  # Pre-computed histogram for retrieval

    def get_flat_tokens(self) -> np.ndarray:
        """Return all tokens concatenated."""
        valid = [t for t in self.tokens if len(t) > 0]
        if valid:
            return np.concatenate(valid)
        return np.zeros((0, 5), dtype=np.float32)


@dataclass
class Piece:
    """A piece with its segments."""
    piece_id: int
    name: str
    audio_path: Path
    midi_path: Path
    duration: float
    audio_segments: List[Segment]
    midi_segments: List[Segment]


# ═══════════════════════════════════════════════════════════════════════════════
# SEGMENTATION
# ═══════════════════════════════════════════════════════════════════════════════

def segment_tokens(
    tokens_per_frame: List[np.ndarray],
    frame_times: np.ndarray,
    piece_id: int,
    piece_name: str,
    segment_len: float = 20.0,
    hop: float = 10.0,
    sr: int = 22050,
    hop_length: int = 512,
) -> List[Segment]:
    """Segment tokens into fixed-length segments."""

    duration = frame_times[-1] if len(frame_times) > 0 else 0
    segments = []

    frame_dt = hop_length / sr
    start_time = 0.0
    seg_id = 0

    while start_time + segment_len <= duration + 0.1:
        end_time = start_time + segment_len

        # Find frame indices for this segment
        start_frame = int(start_time / frame_dt)
        end_frame = int(end_time / frame_dt)
        end_frame = min(end_frame, len(tokens_per_frame))

        if start_frame >= end_frame:
            start_time += hop
            continue

        # Extract tokens for this segment
        seg_tokens = tokens_per_frame[start_frame:end_frame]

        # Compute histogram for retrieval
        hist = compute_segment_histogram(seg_tokens)

        segments.append(Segment(
            piece_id=piece_id,
            piece_name=piece_name,
            segment_id=seg_id,
            start_time=start_time,
            end_time=end_time,
            tokens=seg_tokens,
            histogram=hist,
        ))

        seg_id += 1
        start_time += hop

    return segments


def compute_segment_histogram(tokens: List[np.ndarray], n_bins: int = 50) -> np.ndarray:
    """Compute normalized histogram for a segment."""
    flat = np.concatenate([t for t in tokens if len(t) > 0]) if any(len(t) > 0 for t in tokens) else np.zeros((0, 5))

    if len(flat) == 0:
        return np.zeros(n_bins)

    log_ratios = flat[:, 0]
    hist, _ = np.histogram(log_ratios, bins=np.linspace(0, 2.5, n_bins + 1), density=True)
    hist = hist / (hist.sum() + 1e-8)  # L1 normalize
    return hist.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# RETRIEVAL WITHOUT NN
# ═══════════════════════════════════════════════════════════════════════════════

def compute_similarity(hist1: np.ndarray, hist2: np.ndarray) -> float:
    """Compute cosine similarity between histograms."""
    norm1 = np.linalg.norm(hist1)
    norm2 = np.linalg.norm(hist2)
    if norm1 < 1e-8 or norm2 < 1e-8:
        return 0.0
    return float(np.dot(hist1, hist2) / (norm1 * norm2))


def retrieval_evaluation(
    audio_segments: List[Segment],
    midi_segments: List[Segment],
) -> Dict:
    """
    Evaluate retrieval: for each audio segment, rank all MIDI segments.

    Returns metrics including:
    - Recall@K for K=1,5,10,20
    - MRR (Mean Reciprocal Rank)
    - Gap analysis (aligned vs negatives)
    """
    n_audio = len(audio_segments)
    n_midi = len(midi_segments)

    if n_audio == 0 or n_midi == 0:
        return {'error': 'No segments'}

    # Build similarity matrix
    similarities = np.zeros((n_audio, n_midi))
    for i, a_seg in enumerate(audio_segments):
        for j, m_seg in enumerate(midi_segments):
            similarities[i, j] = compute_similarity(a_seg.histogram, m_seg.histogram)

    # For each audio segment, find rank of its aligned MIDI segment
    ranks = []
    aligned_scores = []
    neg_random_scores = []
    neg_same_piece_scores = []

    for i, a_seg in enumerate(audio_segments):
        # Find aligned MIDI segment (same piece, same segment_id)
        aligned_idx = None
        for j, m_seg in enumerate(midi_segments):
            if m_seg.piece_id == a_seg.piece_id and m_seg.segment_id == a_seg.segment_id:
                aligned_idx = j
                break

        if aligned_idx is None:
            continue

        # Get similarity scores
        scores = similarities[i]
        aligned_score = scores[aligned_idx]
        aligned_scores.append(aligned_score)

        # Rank (1-indexed)
        rank = int((scores > aligned_score).sum()) + 1
        ranks.append(rank)

        # Collect negative scores
        for j, m_seg in enumerate(midi_segments):
            if j == aligned_idx:
                continue
            if m_seg.piece_id == a_seg.piece_id:
                # Same piece, different time
                neg_same_piece_scores.append(scores[j])
            else:
                # Different piece (random)
                neg_random_scores.append(scores[j])

    if len(ranks) == 0:
        return {'error': 'No aligned pairs found'}

    ranks = np.array(ranks)

    # Compute metrics
    recall_at = {}
    for k in [1, 5, 10, 20]:
        recall_at[k] = float((ranks <= k).mean())

    mrr = float(np.mean(1.0 / ranks))

    results = {
        'n_queries': len(ranks),
        'n_candidates': n_midi,
        'recall@1': recall_at[1],
        'recall@5': recall_at[5],
        'recall@10': recall_at[10],
        'recall@20': recall_at[20],
        'mrr': mrr,
        'mean_rank': float(ranks.mean()),
        'median_rank': float(np.median(ranks)),

        # Score distributions
        'aligned_score_mean': float(np.mean(aligned_scores)),
        'aligned_score_std': float(np.std(aligned_scores)),
        'neg_random_score_mean': float(np.mean(neg_random_scores)) if neg_random_scores else 0,
        'neg_same_piece_score_mean': float(np.mean(neg_same_piece_scores)) if neg_same_piece_scores else 0,

        # Gap analysis
        'gap_aligned_vs_random': float(np.mean(aligned_scores) - np.mean(neg_random_scores)) if neg_random_scores else 0,
        'gap_aligned_vs_same_piece': float(np.mean(aligned_scores) - np.mean(neg_same_piece_scores)) if neg_same_piece_scores else 0,
    }

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# SELF VS CROSS EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

def self_vs_cross_evaluation(pieces: List[Piece]) -> Dict:
    """
    For each piece, compare:
    - Self: similarity(audio_seg_i, midi_seg_i) for aligned segments
    - Cross: similarity(audio_seg_i, midi_seg_j) for j != i within same piece
    """
    self_scores = []
    cross_scores = []

    for piece in pieces:
        n_segs = min(len(piece.audio_segments), len(piece.midi_segments))
        if n_segs < 2:
            continue

        for i in range(n_segs):
            a_seg = piece.audio_segments[i]

            for j in range(n_segs):
                m_seg = piece.midi_segments[j]
                score = compute_similarity(a_seg.histogram, m_seg.histogram)

                if i == j:
                    self_scores.append(score)
                else:
                    cross_scores.append(score)

    if not self_scores or not cross_scores:
        return {'error': 'Insufficient data'}

    return {
        'self_score_mean': float(np.mean(self_scores)),
        'self_score_std': float(np.std(self_scores)),
        'cross_score_mean': float(np.mean(cross_scores)),
        'cross_score_std': float(np.std(cross_scores)),
        'gap_self_vs_cross': float(np.mean(self_scores) - np.mean(cross_scores)),
        'n_self': len(self_scores),
        'n_cross': len(cross_scores),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

def process_pair(
    audio_path: Path,
    midi_path: Path,
    piece_id: int,
    sr: int = 22050,
    hop_length: int = 512,
    n_cqt_bins: int = 84,
    segment_len: float = 20.0,
    segment_hop: float = 10.0,
    max_duration: Optional[float] = 120.0,
) -> Tuple[Piece, Dict]:
    """Process a single audio-MIDI pair."""

    piece_name = audio_path.stem

    # Load audio
    audio, _ = librosa.load(audio_path, sr=sr, mono=True, duration=max_duration)
    rms = np.sqrt(np.mean(audio ** 2))
    if rms > 0:
        audio = audio / rms * 0.1
    duration = len(audio) / sr

    # Extract audio constellations
    audio_tokens, audio_times, audio_stats = extract_audio_constellation_parallel(
        audio, sr=sr, hop_length=hop_length, n_cqt_bins=n_cqt_bins,
        n_workers=N_WORKERS,
    )

    # Parse and extract MIDI constellations
    all_notes = parse_midi(midi_path)
    notes = [n for n in all_notes if n.onset < duration]

    midi_tokens, midi_times, midi_stats = extract_midi_constellation_parallel(
        notes, duration=duration, sr=sr, hop_length=hop_length, n_cqt_bins=n_cqt_bins,
        n_workers=N_WORKERS,
    )

    # Segment
    audio_segments = segment_tokens(
        audio_tokens, audio_times, piece_id, piece_name,
        segment_len=segment_len, hop=segment_hop, sr=sr, hop_length=hop_length,
    )
    midi_segments = segment_tokens(
        midi_tokens, midi_times, piece_id, piece_name,
        segment_len=segment_len, hop=segment_hop, sr=sr, hop_length=hop_length,
    )

    # Token compatibility metrics
    compat = compare_histograms(audio_tokens, midi_tokens)

    piece = Piece(
        piece_id=piece_id,
        name=piece_name,
        audio_path=audio_path,
        midi_path=midi_path,
        duration=duration,
        audio_segments=audio_segments,
        midi_segments=midi_segments,
    )

    stats = {
        'piece_id': piece_id,
        'name': piece_name,
        'duration': duration,
        'n_audio_segments': len(audio_segments),
        'n_midi_segments': len(midi_segments),
        'compatibility': compat,
    }

    return piece, stats


def visualize_overall_results(
    compatibility_results: List[Dict],
    retrieval_results: Dict,
    self_cross_results: Dict,
    output_path: Path,
):
    """Create summary visualization."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1. Compatibility: cosine similarity per piece
    ax = axes[0, 0]
    cosines = [r['compatibility']['cosine_similarity'] for r in compatibility_results]
    piece_names = [r['name'][:15] for r in compatibility_results]
    bars = ax.barh(range(len(cosines)), cosines, color='steelblue')
    ax.axvline(0.9, color='green', ls='--', label='GO threshold')
    ax.set_yticks(range(len(cosines)))
    ax.set_yticklabels(piece_names, fontsize=8)
    ax.set_xlabel('Cosine Similarity')
    ax.set_title('Token Compatibility per Piece')
    ax.legend()
    ax.set_xlim(0, 1)

    # 2. Compatibility: token ratio
    ax = axes[0, 1]
    ratios = [r['compatibility']['token_ratio'] for r in compatibility_results]
    ax.barh(range(len(ratios)), ratios, color='coral')
    ax.axvline(0.5, color='red', ls='--', alpha=0.5)
    ax.axvline(2.0, color='red', ls='--', alpha=0.5, label='Acceptable range')
    ax.set_yticks(range(len(ratios)))
    ax.set_yticklabels(piece_names, fontsize=8)
    ax.set_xlabel('Token Ratio (Audio/MIDI)')
    ax.set_title('Token Density Balance')
    ax.legend()

    # 3. Retrieval: Recall@K
    ax = axes[0, 2]
    ks = [1, 5, 10, 20]
    recalls = [retrieval_results.get(f'recall@{k}', 0) for k in ks]
    n_cand = retrieval_results.get('n_candidates', 1)
    random_baselines = [k/n_cand for k in ks]
    x = np.arange(len(ks))
    width = 0.35
    ax.bar(x - width/2, recalls, width, label='Model', color='steelblue')
    ax.bar(x + width/2, random_baselines, width, label='Random', color='lightgray')
    ax.set_xticks(x)
    ax.set_xticklabels([f'@{k}' for k in ks])
    ax.set_ylabel('Recall')
    ax.set_title(f'Retrieval Performance (n={retrieval_results.get("n_queries", 0)} queries)')
    ax.legend()
    ax.set_ylim(0, 1)

    # 4. Score distributions
    ax = axes[1, 0]
    scores = [
        retrieval_results.get('aligned_score_mean', 0),
        retrieval_results.get('neg_same_piece_score_mean', 0),
        retrieval_results.get('neg_random_score_mean', 0),
    ]
    labels = ['Aligned', 'Same Piece', 'Random']
    colors = ['green', 'orange', 'red']
    ax.bar(labels, scores, color=colors)
    ax.set_ylabel('Mean Similarity')
    ax.set_title('Score Distribution by Type')
    ax.set_ylim(0, 1)

    # 5. Self vs Cross
    ax = axes[1, 1]
    if 'error' not in self_cross_results:
        self_mean = self_cross_results['self_score_mean']
        cross_mean = self_cross_results['cross_score_mean']
        self_std = self_cross_results['self_score_std']
        cross_std = self_cross_results['cross_score_std']
        ax.bar(['Self (aligned)', 'Cross (diff time)'],
               [self_mean, cross_mean],
               yerr=[self_std, cross_std],
               color=['green', 'orange'], capsize=5)
        ax.set_ylabel('Mean Similarity')
        ax.set_title(f'Self vs Cross (gap={self_cross_results["gap_self_vs_cross"]:.3f})')
    else:
        ax.text(0.5, 0.5, self_cross_results['error'], ha='center', va='center')
        ax.set_title('Self vs Cross')

    # 6. Summary GO/NO-GO
    ax = axes[1, 2]
    ax.axis('off')

    # Compute GO/NO-GO criteria
    avg_cosine = np.mean(cosines)
    avg_ratio = np.mean(ratios)
    recall_1 = retrieval_results.get('recall@1', 0)
    recall_5 = retrieval_results.get('recall@5', 0)
    n_cand = retrieval_results.get('n_candidates', 1)
    gap_self_cross = self_cross_results.get('gap_self_vs_cross', 0)

    checks = [
        (f"Avg Cosine: {avg_cosine:.3f}", avg_cosine > 0.9, "> 0.9"),
        (f"Avg Token Ratio: {avg_ratio:.2f}x", 0.5 < avg_ratio < 2.0, "0.5-2.0"),
        (f"Recall@1: {recall_1:.1%}", recall_1 > 5/n_cand if n_cand > 0 else False, f"> {5/n_cand:.1%}"),
        (f"Recall@5: {recall_5:.1%}", recall_5 > 25/n_cand if n_cand > 0 else False, f"> {25/n_cand:.1%}"),
        (f"Self-Cross Gap: {gap_self_cross:.3f}", gap_self_cross > 0.05, "> 0.05"),
    ]

    text_lines = ["=" * 35, "GO/NO-GO SUMMARY", "=" * 35, ""]
    all_pass = True
    for check, passed, threshold in checks:
        status = "✓ GO" if passed else "✗ FAIL"
        text_lines.append(f"{check}")
        text_lines.append(f"  {status} (need {threshold})")
        text_lines.append("")
        if not passed:
            all_pass = False

    text_lines.append("=" * 35)
    text_lines.append(f"OVERALL: {'✓ GO' if all_pass else '✗ NO-GO'}")
    text_lines.append("=" * 35)

    ax.text(0.1, 0.95, '\n'.join(text_lines), transform=ax.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat' if all_pass else 'lightcoral', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Pre-red validation with 10 pairs")
    parser.add_argument('--input-dir', type=Path,
                        default=Path(__file__).parent / "Varios_pares")
    parser.add_argument('--output-dir', type=Path,
                        default=Path(__file__).parent / "Varios_pares" / "results")
    parser.add_argument('--segment-len', type=float, default=20.0)
    parser.add_argument('--segment-hop', type=float, default=10.0)
    parser.add_argument('--max-duration', type=float, default=120.0,
                        help="Max duration per piece (seconds)")
    parser.add_argument('--workers', type=int, default=N_WORKERS)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PRE-RED VALIDATION: 10 Pairs Test")
    print("=" * 70)
    print(f"Input:        {args.input_dir}")
    print(f"Output:       {args.output_dir}")
    print(f"Segment:      {args.segment_len}s (hop {args.segment_hop}s)")
    print(f"Max duration: {args.max_duration}s per piece")
    print(f"Workers:      {args.workers}")
    print("=" * 70)

    # Find pairs
    audio_files = sorted(args.input_dir.glob("*.wav"))
    print(f"\nFound {len(audio_files)} audio files")

    if len(audio_files) == 0:
        print("ERROR: No WAV files found")
        sys.exit(1)

    # Process all pairs
    pieces = []
    compatibility_results = []
    total_start = time.time()

    for i, audio_path in enumerate(audio_files):
        midi_path = audio_path.with_suffix('.midi')
        if not midi_path.exists():
            midi_path = audio_path.with_suffix('.mid')
        if not midi_path.exists():
            print(f"  SKIP: No MIDI for {audio_path.name}")
            continue

        print(f"\n[{i+1}/{len(audio_files)}] Processing {audio_path.name[:50]}...")
        t0 = time.time()

        try:
            piece, stats = process_pair(
                audio_path, midi_path, piece_id=i,
                max_duration=args.max_duration,
                segment_len=args.segment_len,
                segment_hop=args.segment_hop,
            )
            pieces.append(piece)
            compatibility_results.append(stats)

            print(f"    Duration: {stats['duration']:.1f}s")
            print(f"    Segments: {stats['n_audio_segments']} audio, {stats['n_midi_segments']} midi")
            print(f"    Cosine: {stats['compatibility']['cosine_similarity']:.4f}")
            print(f"    Time: {time.time() - t0:.1f}s")

        except Exception as e:
            print(f"    ERROR: {e}")
            continue

    if len(pieces) == 0:
        print("\nERROR: No pieces processed successfully")
        sys.exit(1)

    # Collect all segments
    all_audio_segments = []
    all_midi_segments = []
    for piece in pieces:
        all_audio_segments.extend(piece.audio_segments)
        all_midi_segments.extend(piece.midi_segments)

    print(f"\n{'=' * 70}")
    print(f"TOTAL: {len(pieces)} pieces, {len(all_audio_segments)} audio segments, {len(all_midi_segments)} midi segments")
    print(f"{'=' * 70}")

    # === EVALUATION 1: Token Compatibility ===
    print("\n[EVAL 1] Token Compatibility...")
    avg_cosine = np.mean([r['compatibility']['cosine_similarity'] for r in compatibility_results])
    avg_hist_int = np.mean([r['compatibility']['histogram_intersection'] for r in compatibility_results])
    avg_ratio = np.mean([r['compatibility']['token_ratio'] for r in compatibility_results])
    print(f"  Avg Cosine:       {avg_cosine:.4f}")
    print(f"  Avg Hist Int:     {avg_hist_int:.4f}")
    print(f"  Avg Token Ratio:  {avg_ratio:.2f}x")

    # === EVALUATION 2: Retrieval ===
    print("\n[EVAL 2] Retrieval (Shazam-style)...")
    retrieval_results = retrieval_evaluation(all_audio_segments, all_midi_segments)
    if 'error' not in retrieval_results:
        print(f"  Queries:    {retrieval_results['n_queries']}")
        print(f"  Candidates: {retrieval_results['n_candidates']}")
        print(f"  Recall@1:   {retrieval_results['recall@1']:.1%}")
        print(f"  Recall@5:   {retrieval_results['recall@5']:.1%}")
        print(f"  Recall@10:  {retrieval_results['recall@10']:.1%}")
        print(f"  MRR:        {retrieval_results['mrr']:.4f}")
        print(f"  Mean Rank:  {retrieval_results['mean_rank']:.1f}")
        print(f"  Gap (aligned vs random):     {retrieval_results['gap_aligned_vs_random']:.4f}")
        print(f"  Gap (aligned vs same piece): {retrieval_results['gap_aligned_vs_same_piece']:.4f}")
    else:
        print(f"  ERROR: {retrieval_results['error']}")

    # === EVALUATION 3: Self vs Cross ===
    print("\n[EVAL 3] Self vs Cross (within piece)...")
    self_cross_results = self_vs_cross_evaluation(pieces)
    if 'error' not in self_cross_results:
        print(f"  Self (aligned) mean:  {self_cross_results['self_score_mean']:.4f}")
        print(f"  Cross (diff time):    {self_cross_results['cross_score_mean']:.4f}")
        print(f"  Gap:                  {self_cross_results['gap_self_vs_cross']:.4f}")
    else:
        print(f"  ERROR: {self_cross_results['error']}")

    # === VISUALIZATION ===
    print("\n[VIS] Creating summary visualization...")
    visualize_overall_results(
        compatibility_results,
        retrieval_results,
        self_cross_results,
        args.output_dir / "summary_visualization.png",
    )

    # === SAVE RESULTS ===
    results = {
        'config': {
            'n_pieces': len(pieces),
            'n_audio_segments': len(all_audio_segments),
            'n_midi_segments': len(all_midi_segments),
            'segment_len': args.segment_len,
            'segment_hop': args.segment_hop,
            'max_duration': args.max_duration,
        },
        'compatibility': {
            'per_piece': compatibility_results,
            'avg_cosine': avg_cosine,
            'avg_histogram_intersection': avg_hist_int,
            'avg_token_ratio': avg_ratio,
        },
        'retrieval': retrieval_results,
        'self_vs_cross': self_cross_results,
        'total_time': time.time() - total_start,
    }

    with open(args.output_dir / "pre_red_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    # === GO/NO-GO DECISION ===
    print("\n" + "=" * 70)
    print("GO/NO-GO DECISION")
    print("=" * 70)

    go_criteria = [
        ("Token Compatibility (cosine > 0.9)", avg_cosine > 0.9),
        ("Token Ratio (0.5-2.0x)", 0.5 < avg_ratio < 2.0),
        ("Retrieval Recall@1 > 5x random", retrieval_results.get('recall@1', 0) > 5 / retrieval_results.get('n_candidates', 1)),
        ("Self vs Cross gap > 0.05", self_cross_results.get('gap_self_vs_cross', 0) > 0.05),
    ]

    all_pass = True
    for name, passed in go_criteria:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_pass = False

    print("\n" + "=" * 70)
    if all_pass:
        print("  ✓ GO: All criteria passed. Proceed to full MAESTRO retrieval.")
    else:
        print("  ✗ NO-GO: Some criteria failed. Review extractor/matching.")
    print("=" * 70)

    print(f"\nResults saved to: {args.output_dir}")
    print(f"Total time: {time.time() - total_start:.1f}s")


if __name__ == '__main__':
    main()
