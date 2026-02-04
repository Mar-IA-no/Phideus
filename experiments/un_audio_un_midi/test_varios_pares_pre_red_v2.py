#!/usr/bin/env python3
"""
Test Varios Pares - Pre-Red Validation V2
==========================================

Improved matching using:
1. 2D/3D Shazam-style hashes: (dt_bin, log_ratio_bin, f_anchor_coarse)
2. TF-IDF weighting: rare tokens weight more
3. 2D histogram: (log_ratio × delta_t)

Based on GPT5.2Think recommendations after V1 showed:
- Token compatibility OK (cosine > 0.95)
- But retrieval failed because 1D histograms don't capture temporal identity
"""

from __future__ import annotations

import sys
import json
import time
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass, field
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import librosa
except ImportError:
    print("ERROR: librosa required")
    sys.exit(1)

from src.utils.midi_utils import parse_midi, Note

from test_single_pair_v2_parallel import (
    extract_audio_constellation_parallel,
    extract_midi_constellation_parallel,
    compare_histograms,
    N_WORKERS,
)


# ═══════════════════════════════════════════════════════════════════════════════
# HASH CONFIGURATION (GPT5.2Think recommendations)
# ═══════════════════════════════════════════════════════════════════════════════

# Hash bins configuration
DT_BIN_SIZE = 2  # frames (~46ms)
LOG_RATIO_BIN_SIZE = 1/24  # octave (~50 cents)
N_ANCHOR_BANDS = 8  # Coarse frequency bands

# Histogram 2D configuration
N_LOG_RATIO_BINS = 25  # 0 to 2.5 in 0.1 steps
N_DT_BINS = 20  # 0 to 40 frames


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Segment:
    """A segment with multiple representations."""
    piece_id: int
    piece_name: str
    segment_id: int
    start_time: float
    end_time: float
    tokens: List[np.ndarray]

    # Multiple representations for matching
    hash_bag: Counter = field(default_factory=Counter)  # 3D hash counts
    hist_1d: np.ndarray = field(default_factory=lambda: np.zeros(50))
    hist_2d: np.ndarray = field(default_factory=lambda: np.zeros((N_LOG_RATIO_BINS, N_DT_BINS)))

    def get_flat_tokens(self) -> np.ndarray:
        valid = [t for t in self.tokens if len(t) > 0]
        if valid:
            return np.concatenate(valid)
        return np.zeros((0, 5), dtype=np.float32)


@dataclass
class Piece:
    piece_id: int
    name: str
    audio_path: Path
    midi_path: Path
    duration: float
    audio_segments: List[Segment]
    midi_segments: List[Segment]


# ═══════════════════════════════════════════════════════════════════════════════
# HASHING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def token_to_hash(token: np.ndarray) -> Tuple[int, int, int]:
    """
    Convert token [log_ratio, delta_t, weight, anchor_band, target_band] to hash.

    Returns: (dt_bin, log_ratio_bin, anchor_band_coarse)
    """
    log_ratio = token[0]
    delta_t = token[1]
    anchor_band = int(token[3])

    dt_bin = int(delta_t / DT_BIN_SIZE)
    log_ratio_bin = int(log_ratio / LOG_RATIO_BIN_SIZE)
    anchor_band_coarse = min(anchor_band * N_ANCHOR_BANDS // 16, N_ANCHOR_BANDS - 1)

    return (dt_bin, log_ratio_bin, anchor_band_coarse)


def compute_hash_bag(tokens: List[np.ndarray]) -> Counter:
    """Compute bag of hashes for a segment."""
    bag = Counter()
    for frame_tokens in tokens:
        if len(frame_tokens) == 0:
            continue
        for token in frame_tokens:
            h = token_to_hash(token)
            bag[h] += 1
    return bag


def compute_hist_2d(tokens: List[np.ndarray]) -> np.ndarray:
    """Compute 2D histogram (log_ratio × delta_t)."""
    hist = np.zeros((N_LOG_RATIO_BINS, N_DT_BINS), dtype=np.float32)

    for frame_tokens in tokens:
        if len(frame_tokens) == 0:
            continue
        for token in frame_tokens:
            log_ratio = token[0]
            delta_t = token[1]

            lr_bin = min(int(log_ratio / 0.1), N_LOG_RATIO_BINS - 1)
            dt_bin = min(int(delta_t / 2), N_DT_BINS - 1)

            if lr_bin >= 0 and dt_bin >= 0:
                hist[lr_bin, dt_bin] += 1

    # L1 normalize
    total = hist.sum()
    if total > 0:
        hist = hist / total

    return hist


def compute_hist_1d(tokens: List[np.ndarray], n_bins: int = 50) -> np.ndarray:
    """Compute 1D histogram of log_ratios."""
    flat = np.concatenate([t for t in tokens if len(t) > 0]) if any(len(t) > 0 for t in tokens) else np.zeros((0, 5))

    if len(flat) == 0:
        return np.zeros(n_bins)

    log_ratios = flat[:, 0]
    hist, _ = np.histogram(log_ratios, bins=np.linspace(0, 2.5, n_bins + 1), density=True)
    hist = hist / (hist.sum() + 1e-8)
    return hist.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# TF-IDF COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════════

class TFIDFScorer:
    """TF-IDF scorer for hash bags."""

    def __init__(self, all_segments: List[Segment]):
        """Build IDF from all segments."""
        self.n_docs = len(all_segments)
        self.doc_freq = Counter()  # How many segments contain each hash

        for seg in all_segments:
            unique_hashes = set(seg.hash_bag.keys())
            for h in unique_hashes:
                self.doc_freq[h] += 1

    def idf(self, h: tuple) -> float:
        """Inverse document frequency with smoothing."""
        df = self.doc_freq.get(h, 0)
        return np.log((self.n_docs + 1) / (df + 1)) + 1

    def tfidf_vector(self, bag: Counter) -> Dict[tuple, float]:
        """Compute TF-IDF weighted vector."""
        total = sum(bag.values())
        if total == 0:
            return {}

        result = {}
        for h, count in bag.items():
            tf = count / total
            result[h] = tf * self.idf(h)
        return result


# ═══════════════════════════════════════════════════════════════════════════════
# SIMILARITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def hash_overlap_score(bag1: Counter, bag2: Counter) -> float:
    """Simple hash overlap (intersection count)."""
    common = set(bag1.keys()) & set(bag2.keys())
    if not common:
        return 0.0

    overlap = sum(min(bag1[h], bag2[h]) for h in common)
    total = sum(bag1.values()) + sum(bag2.values())
    return 2.0 * overlap / (total + 1e-8)


def tfidf_cosine_score(vec1: Dict, vec2: Dict) -> float:
    """Cosine similarity between TF-IDF vectors."""
    common = set(vec1.keys()) & set(vec2.keys())
    if not common:
        return 0.0

    dot = sum(vec1[h] * vec2[h] for h in common)
    norm1 = np.sqrt(sum(v**2 for v in vec1.values()))
    norm2 = np.sqrt(sum(v**2 for v in vec2.values()))

    if norm1 < 1e-8 or norm2 < 1e-8:
        return 0.0

    return dot / (norm1 * norm2)


def hist_2d_similarity(hist1: np.ndarray, hist2: np.ndarray) -> float:
    """Cosine similarity between 2D histograms."""
    flat1 = hist1.flatten()
    flat2 = hist2.flatten()

    norm1 = np.linalg.norm(flat1)
    norm2 = np.linalg.norm(flat2)

    if norm1 < 1e-8 or norm2 < 1e-8:
        return 0.0

    return float(np.dot(flat1, flat2) / (norm1 * norm2))


def hist_1d_similarity(hist1: np.ndarray, hist2: np.ndarray) -> float:
    """Cosine similarity between 1D histograms."""
    norm1 = np.linalg.norm(hist1)
    norm2 = np.linalg.norm(hist2)

    if norm1 < 1e-8 or norm2 < 1e-8:
        return 0.0

    return float(np.dot(hist1, hist2) / (norm1 * norm2))


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
    """Segment tokens and compute all representations."""

    duration = frame_times[-1] if len(frame_times) > 0 else 0
    segments = []

    frame_dt = hop_length / sr
    start_time = 0.0
    seg_id = 0

    while start_time + segment_len <= duration + 0.1:
        end_time = start_time + segment_len

        start_frame = int(start_time / frame_dt)
        end_frame = int(end_time / frame_dt)
        end_frame = min(end_frame, len(tokens_per_frame))

        if start_frame >= end_frame:
            start_time += hop
            continue

        seg_tokens = tokens_per_frame[start_frame:end_frame]

        seg = Segment(
            piece_id=piece_id,
            piece_name=piece_name,
            segment_id=seg_id,
            start_time=start_time,
            end_time=end_time,
            tokens=seg_tokens,
            hash_bag=compute_hash_bag(seg_tokens),
            hist_1d=compute_hist_1d(seg_tokens),
            hist_2d=compute_hist_2d(seg_tokens),
        )

        segments.append(seg)
        seg_id += 1
        start_time += hop

    return segments


# ═══════════════════════════════════════════════════════════════════════════════
# RETRIEVAL EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

def retrieval_evaluation_multi(
    audio_segments: List[Segment],
    midi_segments: List[Segment],
) -> Dict:
    """
    Evaluate retrieval with multiple similarity methods.
    """
    n_audio = len(audio_segments)
    n_midi = len(midi_segments)

    if n_audio == 0 or n_midi == 0:
        return {'error': 'No segments'}

    # Build TF-IDF scorer from all MIDI segments
    print("    Building TF-IDF index...")
    tfidf = TFIDFScorer(midi_segments)

    # Pre-compute TF-IDF vectors for all segments
    audio_tfidf = [tfidf.tfidf_vector(seg.hash_bag) for seg in audio_segments]
    midi_tfidf = [tfidf.tfidf_vector(seg.hash_bag) for seg in midi_segments]

    methods = {
        'hash_overlap': lambda a, m: hash_overlap_score(a.hash_bag, m.hash_bag),
        'tfidf_cosine': lambda a, m: tfidf_cosine_score(audio_tfidf[audio_segments.index(a)], midi_tfidf[midi_segments.index(m)]),
        'hist_2d': lambda a, m: hist_2d_similarity(a.hist_2d, m.hist_2d),
        'hist_1d': lambda a, m: hist_1d_similarity(a.hist_1d, m.hist_1d),
    }

    all_results = {}

    for method_name, score_fn in methods.items():
        print(f"    Evaluating {method_name}...")

        ranks = []
        aligned_scores = []
        neg_random_scores = []
        neg_same_piece_scores = []

        for i, a_seg in enumerate(audio_segments):
            # Find aligned MIDI segment
            aligned_idx = None
            for j, m_seg in enumerate(midi_segments):
                if m_seg.piece_id == a_seg.piece_id and m_seg.segment_id == a_seg.segment_id:
                    aligned_idx = j
                    break

            if aligned_idx is None:
                continue

            # Compute scores
            scores = [score_fn(a_seg, m_seg) for m_seg in midi_segments]
            aligned_score = scores[aligned_idx]
            aligned_scores.append(aligned_score)

            # Rank
            rank = int(sum(1 for s in scores if s > aligned_score)) + 1
            ranks.append(rank)

            # Collect negatives
            for j, m_seg in enumerate(midi_segments):
                if j == aligned_idx:
                    continue
                if m_seg.piece_id == a_seg.piece_id:
                    neg_same_piece_scores.append(scores[j])
                else:
                    neg_random_scores.append(scores[j])

        if len(ranks) == 0:
            all_results[method_name] = {'error': 'No aligned pairs'}
            continue

        ranks = np.array(ranks)

        recall_at = {}
        for k in [1, 5, 10, 20]:
            recall_at[k] = float((ranks <= k).mean())

        mrr = float(np.mean(1.0 / ranks))

        all_results[method_name] = {
            'n_queries': len(ranks),
            'n_candidates': n_midi,
            'recall@1': recall_at[1],
            'recall@5': recall_at[5],
            'recall@10': recall_at[10],
            'recall@20': recall_at[20],
            'mrr': mrr,
            'mean_rank': float(ranks.mean()),
            'aligned_score_mean': float(np.mean(aligned_scores)),
            'neg_random_score_mean': float(np.mean(neg_random_scores)) if neg_random_scores else 0,
            'neg_same_piece_score_mean': float(np.mean(neg_same_piece_scores)) if neg_same_piece_scores else 0,
            'gap_aligned_vs_random': float(np.mean(aligned_scores) - np.mean(neg_random_scores)) if neg_random_scores else 0,
            'gap_aligned_vs_same_piece': float(np.mean(aligned_scores) - np.mean(neg_same_piece_scores)) if neg_same_piece_scores else 0,
        }

    return all_results


def self_vs_cross_evaluation_multi(pieces: List[Piece]) -> Dict:
    """Self vs Cross with multiple methods."""
    methods = {
        'hash_overlap': lambda a, m: hash_overlap_score(a.hash_bag, m.hash_bag),
        'hist_2d': lambda a, m: hist_2d_similarity(a.hist_2d, m.hist_2d),
        'hist_1d': lambda a, m: hist_1d_similarity(a.hist_1d, m.hist_1d),
    }

    all_results = {}

    for method_name, score_fn in methods.items():
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
                    score = score_fn(a_seg, m_seg)
                    if i == j:
                        self_scores.append(score)
                    else:
                        cross_scores.append(score)

        if not self_scores or not cross_scores:
            all_results[method_name] = {'error': 'Insufficient data'}
            continue

        all_results[method_name] = {
            'self_score_mean': float(np.mean(self_scores)),
            'cross_score_mean': float(np.mean(cross_scores)),
            'gap': float(np.mean(self_scores) - np.mean(cross_scores)),
        }

    return all_results


# ═══════════════════════════════════════════════════════════════════════════════
# PROCESSING
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

    audio, _ = librosa.load(audio_path, sr=sr, mono=True, duration=max_duration)
    rms = np.sqrt(np.mean(audio ** 2))
    if rms > 0:
        audio = audio / rms * 0.1
    duration = len(audio) / sr

    audio_tokens, audio_times, audio_stats = extract_audio_constellation_parallel(
        audio, sr=sr, hop_length=hop_length, n_cqt_bins=n_cqt_bins,
        n_workers=N_WORKERS,
    )

    all_notes = parse_midi(midi_path)
    notes = [n for n in all_notes if n.onset < duration]

    midi_tokens, midi_times, midi_stats = extract_midi_constellation_parallel(
        notes, duration=duration, sr=sr, hop_length=hop_length, n_cqt_bins=n_cqt_bins,
        n_workers=N_WORKERS,
    )

    audio_segments = segment_tokens(
        audio_tokens, audio_times, piece_id, piece_name,
        segment_len=segment_len, hop=segment_hop, sr=sr, hop_length=hop_length,
    )
    midi_segments = segment_tokens(
        midi_tokens, midi_times, piece_id, piece_name,
        segment_len=segment_len, hop=segment_hop, sr=sr, hop_length=hop_length,
    )

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


# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def visualize_results(
    compatibility_results: List[Dict],
    retrieval_results: Dict,
    self_cross_results: Dict,
    output_path: Path,
):
    """Create summary visualization with multiple methods."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 11))

    # 1. Compatibility per piece
    ax = axes[0, 0]
    cosines = [r['compatibility']['cosine_similarity'] for r in compatibility_results]
    piece_names = [r['name'][:12] for r in compatibility_results]
    ax.barh(range(len(cosines)), cosines, color='steelblue')
    ax.axvline(0.9, color='green', ls='--', label='GO threshold')
    ax.set_yticks(range(len(cosines)))
    ax.set_yticklabels(piece_names, fontsize=7)
    ax.set_xlabel('Cosine Similarity')
    ax.set_title('Token Compatibility')
    ax.set_xlim(0, 1)

    # 2. Retrieval comparison by method
    ax = axes[0, 1]
    methods = list(retrieval_results.keys())
    recall_1 = [retrieval_results[m].get('recall@1', 0) for m in methods]
    recall_5 = [retrieval_results[m].get('recall@5', 0) for m in methods]
    x = np.arange(len(methods))
    width = 0.35
    ax.bar(x - width/2, recall_1, width, label='Recall@1', color='steelblue')
    ax.bar(x + width/2, recall_5, width, label='Recall@5', color='coral')
    n_cand = retrieval_results.get(methods[0], {}).get('n_candidates', 110)
    ax.axhline(1/n_cand, color='gray', ls='--', alpha=0.5, label='Random@1')
    ax.axhline(5/n_cand, color='gray', ls=':', alpha=0.5, label='Random@5')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, fontsize=8)
    ax.set_ylabel('Recall')
    ax.set_title('Retrieval by Method')
    ax.legend(fontsize=7)
    ax.set_ylim(0, 1)

    # 3. Gap analysis by method
    ax = axes[0, 2]
    gaps_random = [retrieval_results[m].get('gap_aligned_vs_random', 0) for m in methods]
    gaps_same = [retrieval_results[m].get('gap_aligned_vs_same_piece', 0) for m in methods]
    ax.bar(x - width/2, gaps_random, width, label='vs Random', color='green')
    ax.bar(x + width/2, gaps_same, width, label='vs Same Piece', color='orange')
    ax.axhline(0.05, color='red', ls='--', label='Min useful gap')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, fontsize=8)
    ax.set_ylabel('Gap')
    ax.set_title('Score Gap (Aligned - Negative)')
    ax.legend(fontsize=7)

    # 4. Self vs Cross by method
    ax = axes[1, 0]
    sc_methods = list(self_cross_results.keys())
    self_means = [self_cross_results[m].get('self_score_mean', 0) for m in sc_methods]
    cross_means = [self_cross_results[m].get('cross_score_mean', 0) for m in sc_methods]
    x = np.arange(len(sc_methods))
    ax.bar(x - width/2, self_means, width, label='Self (aligned)', color='green')
    ax.bar(x + width/2, cross_means, width, label='Cross (diff time)', color='orange')
    ax.set_xticks(x)
    ax.set_xticklabels(sc_methods, rotation=15, fontsize=8)
    ax.set_ylabel('Mean Score')
    ax.set_title('Self vs Cross')
    ax.legend()

    # 5. Self-Cross Gap by method
    ax = axes[1, 1]
    sc_gaps = [self_cross_results[m].get('gap', 0) for m in sc_methods]
    colors = ['green' if g > 0.05 else 'red' for g in sc_gaps]
    ax.bar(sc_methods, sc_gaps, color=colors)
    ax.axhline(0.05, color='red', ls='--', label='GO threshold')
    ax.set_ylabel('Gap')
    ax.set_title('Self-Cross Gap by Method')
    ax.legend()

    # 6. Summary
    ax = axes[1, 2]
    ax.axis('off')

    # Find best method
    best_method = max(methods, key=lambda m: retrieval_results[m].get('recall@1', 0))
    best_recall = retrieval_results[best_method]

    avg_cosine = np.mean(cosines)
    best_r1 = best_recall.get('recall@1', 0)
    best_r5 = best_recall.get('recall@5', 0)
    best_gap = best_recall.get('gap_aligned_vs_same_piece', 0)
    n_cand = best_recall.get('n_candidates', 110)

    checks = [
        (f"Token Compat: {avg_cosine:.3f}", avg_cosine > 0.9, "> 0.9"),
        (f"Best Method: {best_method}", True, "-"),
        (f"Recall@1: {best_r1:.1%}", best_r1 > 5/n_cand, f"> {5/n_cand:.1%}"),
        (f"Recall@5: {best_r5:.1%}", best_r5 > 25/n_cand, f"> {25/n_cand:.1%}"),
        (f"Gap (same piece): {best_gap:.3f}", best_gap > 0.05, "> 0.05"),
    ]

    text = ["=" * 40, "GO/NO-GO SUMMARY (V2)", "=" * 40, ""]
    all_pass = True
    for check, passed, threshold in checks:
        status = "✓" if passed else "✗"
        text.append(f"{status} {check}")
        text.append(f"    (need {threshold})")
        text.append("")
        if not passed and threshold != "-":
            all_pass = False

    text.append("=" * 40)
    text.append(f"OVERALL: {'✓ GO' if all_pass else '✗ NO-GO'}")
    text.append("=" * 40)

    ax.text(0.05, 0.95, '\n'.join(text), transform=ax.transAxes,
            fontsize=9, va='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgreen' if all_pass else 'lightcoral', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Pre-red validation V2 with hashes")
    parser.add_argument('--input-dir', type=Path,
                        default=Path(__file__).parent / "Varios_pares")
    parser.add_argument('--output-dir', type=Path,
                        default=Path(__file__).parent / "Varios_pares" / "results_v2")
    parser.add_argument('--segment-len', type=float, default=20.0)
    parser.add_argument('--segment-hop', type=float, default=10.0)
    parser.add_argument('--max-duration', type=float, default=120.0)
    parser.add_argument('--workers', type=int, default=N_WORKERS)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PRE-RED VALIDATION V2: Hashes + TF-IDF + Hist2D")
    print("=" * 70)
    print(f"Input:        {args.input_dir}")
    print(f"Output:       {args.output_dir}")
    print(f"Segment:      {args.segment_len}s (hop {args.segment_hop}s)")
    print(f"Max duration: {args.max_duration}s per piece")
    print(f"Workers:      {args.workers}")
    print("=" * 70)

    audio_files = sorted(args.input_dir.glob("*.wav"))
    print(f"\nFound {len(audio_files)} audio files")

    if len(audio_files) == 0:
        print("ERROR: No WAV files found")
        sys.exit(1)

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

            print(f"    Duration: {stats['duration']:.1f}s, Segments: {stats['n_audio_segments']}")
            print(f"    Cosine: {stats['compatibility']['cosine_similarity']:.4f}")
            print(f"    Time: {time.time() - t0:.1f}s")

        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue

    if len(pieces) == 0:
        print("\nERROR: No pieces processed")
        sys.exit(1)

    all_audio_segments = []
    all_midi_segments = []
    for piece in pieces:
        all_audio_segments.extend(piece.audio_segments)
        all_midi_segments.extend(piece.midi_segments)

    print(f"\n{'=' * 70}")
    print(f"TOTAL: {len(pieces)} pieces, {len(all_audio_segments)} segments")
    print(f"{'=' * 70}")

    # === EVAL 1: Token Compatibility ===
    print("\n[EVAL 1] Token Compatibility...")
    avg_cosine = np.mean([r['compatibility']['cosine_similarity'] for r in compatibility_results])
    print(f"  Avg Cosine: {avg_cosine:.4f}")

    # === EVAL 2: Retrieval with multiple methods ===
    print("\n[EVAL 2] Retrieval (multiple methods)...")
    retrieval_results = retrieval_evaluation_multi(all_audio_segments, all_midi_segments)

    for method, res in retrieval_results.items():
        if 'error' in res:
            print(f"  {method}: ERROR - {res['error']}")
        else:
            print(f"  {method}:")
            print(f"    Recall@1: {res['recall@1']:.1%}, Recall@5: {res['recall@5']:.1%}")
            print(f"    Gap vs random: {res['gap_aligned_vs_random']:.4f}")
            print(f"    Gap vs same piece: {res['gap_aligned_vs_same_piece']:.4f}")

    # === EVAL 3: Self vs Cross ===
    print("\n[EVAL 3] Self vs Cross...")
    self_cross_results = self_vs_cross_evaluation_multi(pieces)

    for method, res in self_cross_results.items():
        if 'error' in res:
            print(f"  {method}: ERROR - {res['error']}")
        else:
            print(f"  {method}: gap = {res['gap']:.4f}")

    # === VISUALIZATION ===
    print("\n[VIS] Creating visualization...")
    visualize_results(
        compatibility_results,
        retrieval_results,
        self_cross_results,
        args.output_dir / "summary_v2.png",
    )

    # === SAVE ===
    results = {
        'config': {
            'n_pieces': len(pieces),
            'n_segments': len(all_audio_segments),
            'segment_len': args.segment_len,
            'hash_config': {
                'dt_bin_size': DT_BIN_SIZE,
                'log_ratio_bin_size': LOG_RATIO_BIN_SIZE,
                'n_anchor_bands': N_ANCHOR_BANDS,
            },
        },
        'compatibility': {
            'avg_cosine': avg_cosine,
            'per_piece': compatibility_results,
        },
        'retrieval': retrieval_results,
        'self_vs_cross': self_cross_results,
        'total_time': time.time() - total_start,
    }

    with open(args.output_dir / "pre_red_results_v2.json", 'w') as f:
        json.dump(results, f, indent=2)

    # === GO/NO-GO ===
    print("\n" + "=" * 70)
    print("GO/NO-GO DECISION (V2)")
    print("=" * 70)

    best_method = max(retrieval_results.keys(),
                      key=lambda m: retrieval_results[m].get('recall@1', 0))
    best = retrieval_results[best_method]
    n_cand = best.get('n_candidates', 110)

    criteria = [
        ("Token Compatibility (cosine > 0.9)", avg_cosine > 0.9),
        (f"Best Method: {best_method}", True),
        (f"Recall@1 > 5x random ({5/n_cand:.1%})", best.get('recall@1', 0) > 5/n_cand),
        (f"Recall@5 > 5x random ({25/n_cand:.1%})", best.get('recall@5', 0) > 25/n_cand),
        ("Gap (same piece) > 0.05", best.get('gap_aligned_vs_same_piece', 0) > 0.05),
    ]

    all_pass = True
    for name, passed in criteria:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")
        if not passed and "Best Method" not in name:
            all_pass = False

    print("\n" + "=" * 70)
    if all_pass:
        print("  ✓ GO: Proceed to full MAESTRO retrieval!")
    else:
        print("  ✗ NO-GO: Need further improvements.")
    print("=" * 70)

    print(f"\nResults: {args.output_dir}")
    print(f"Time: {time.time() - total_start:.1f}s")


if __name__ == '__main__':
    main()
