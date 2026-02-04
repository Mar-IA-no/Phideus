#!/usr/bin/env python3
"""
Test Varios Pares - SHAZAM-STYLE Retrieval
==========================================

Implements the REAL Shazam algorithm:
1. Hash includes t_anchor_abs (absolute time within piece)
2. DB: hash -> list of (piece_id, t_anchor_abs)
3. Query: vote for offset (tr - tq) for each match
4. Score: peak of offset histogram (time-difference consensus)
5. IDF weighting: rare hashes weight more

This should fix the "same piece, different time" hard negative problem.
"""

from __future__ import annotations

import sys
import json
import time
from pathlib import Path
from typing import List, Tuple, Dict, Optional, NamedTuple
from dataclasses import dataclass, field
from collections import Counter, defaultdict
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
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# Hash bins (balanced for speed vs accuracy)
DT_BIN_SIZE = 2  # frames (~46ms)
LOG_RATIO_BIN_SIZE = 1/24  # octave (~50 cents)
N_ANCHOR_BANDS = 4  # Fewer bands for speed

# Offset voting
OFFSET_BIN_SIZE = 4  # frames (~92ms tolerance)
MAX_OFFSET_BINS = 500  # cover up to ~11s offset range

# Subsampling for speed
MAX_HASHES_PER_SEGMENT = 5000  # Limit hashes per query segment


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

class HashEntry(NamedTuple):
    """Entry in the hash database."""
    piece_id: int
    t_abs: int  # absolute frame within piece


@dataclass
class PieceData:
    """All data for a piece."""
    piece_id: int
    name: str
    audio_path: Path
    midi_path: Path
    duration: float

    # Full piece tokens (not segmented)
    audio_tokens: List[np.ndarray] = field(default_factory=list)
    midi_tokens: List[np.ndarray] = field(default_factory=list)

    # Hashes with absolute time
    audio_hashes: List[Tuple[tuple, int]] = field(default_factory=list)  # (hash, t_abs)
    midi_hashes: List[Tuple[tuple, int]] = field(default_factory=list)


class ShazamDB:
    """Shazam-style hash database for MIDI."""

    def __init__(self):
        self.db: Dict[tuple, List[HashEntry]] = defaultdict(list)
        self.doc_freq: Counter = Counter()  # For IDF
        self.n_pieces: int = 0
        self.piece_names: Dict[int, str] = {}

    def add_piece(self, piece_id: int, name: str, hashes: List[Tuple[tuple, int]]):
        """Add a piece's hashes to the database."""
        self.piece_names[piece_id] = name
        self.n_pieces = max(self.n_pieces, piece_id + 1)

        # Track unique hashes for this piece (for IDF)
        unique_hashes = set()

        for h, t_abs in hashes:
            self.db[h].append(HashEntry(piece_id, t_abs))
            unique_hashes.add(h)

        # Update document frequency
        for h in unique_hashes:
            self.doc_freq[h] += 1

    def idf(self, h: tuple) -> float:
        """Inverse document frequency with smoothing."""
        df = self.doc_freq.get(h, 0)
        return np.log((self.n_pieces + 1) / (df + 1)) + 1

    def query(self, query_hashes: List[Tuple[tuple, int]],
              use_idf: bool = True) -> Dict[int, Tuple[float, int]]:
        """
        Query the database using Shazam offset-consensus voting.

        Returns: {piece_id: (score, best_offset)}
        """
        # Vote accumulator: piece_id -> offset_bin -> score
        votes: Dict[int, Dict[int, float]] = defaultdict(lambda: defaultdict(float))

        for h, tq in query_hashes:
            if h not in self.db:
                continue

            weight = self.idf(h) if use_idf else 1.0

            for entry in self.db[h]:
                # Compute offset: MIDI time - query time
                offset = entry.t_abs - tq
                offset_bin = offset // OFFSET_BIN_SIZE

                # Clamp to valid range
                if abs(offset_bin) < MAX_OFFSET_BINS:
                    votes[entry.piece_id][offset_bin] += weight

        # Find best offset for each piece
        results = {}
        for piece_id, offset_votes in votes.items():
            if not offset_votes:
                continue
            best_offset_bin = max(offset_votes, key=offset_votes.get)
            best_score = offset_votes[best_offset_bin]
            results[piece_id] = (best_score, best_offset_bin * OFFSET_BIN_SIZE)

        return results


# ═══════════════════════════════════════════════════════════════════════════════
# HASHING WITH ABSOLUTE TIME
# ═══════════════════════════════════════════════════════════════════════════════

def extract_hashes_with_time(tokens_per_frame: List[np.ndarray]) -> List[Tuple[tuple, int]]:
    """
    Extract hashes with absolute frame time.

    Returns: List of (hash, t_abs) tuples
    """
    hashes = []

    for t_frame, frame_tokens in enumerate(tokens_per_frame):
        if len(frame_tokens) == 0:
            continue

        for token in frame_tokens:
            log_ratio = token[0]
            delta_t = token[1]
            anchor_band = int(token[3])

            # Create hash
            dt_bin = int(delta_t / DT_BIN_SIZE)
            log_ratio_bin = int(log_ratio / LOG_RATIO_BIN_SIZE)
            anchor_band_coarse = min(anchor_band * N_ANCHOR_BANDS // 16, N_ANCHOR_BANDS - 1)

            h = (dt_bin, log_ratio_bin, anchor_band_coarse)
            hashes.append((h, t_frame))

    return hashes


# ═══════════════════════════════════════════════════════════════════════════════
# PROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

def process_piece(
    audio_path: Path,
    midi_path: Path,
    piece_id: int,
    sr: int = 22050,
    hop_length: int = 512,
    n_cqt_bins: int = 84,
    max_duration: Optional[float] = None,
) -> PieceData:
    """Process a single piece (full duration, not segmented)."""

    piece_name = audio_path.stem

    # Load audio
    audio, _ = librosa.load(audio_path, sr=sr, mono=True, duration=max_duration)
    rms = np.sqrt(np.mean(audio ** 2))
    if rms > 0:
        audio = audio / rms * 0.1
    duration = len(audio) / sr

    # Extract audio constellations
    audio_tokens, audio_times, _ = extract_audio_constellation_parallel(
        audio, sr=sr, hop_length=hop_length, n_cqt_bins=n_cqt_bins,
        n_workers=N_WORKERS,
    )

    # Parse and extract MIDI constellations
    all_notes = parse_midi(midi_path)
    notes = [n for n in all_notes if n.onset < duration]

    midi_tokens, midi_times, _ = extract_midi_constellation_parallel(
        notes, duration=duration, sr=sr, hop_length=hop_length, n_cqt_bins=n_cqt_bins,
        n_workers=N_WORKERS,
    )

    # Extract hashes with absolute time
    audio_hashes = extract_hashes_with_time(audio_tokens)
    midi_hashes = extract_hashes_with_time(midi_tokens)

    return PieceData(
        piece_id=piece_id,
        name=piece_name,
        audio_path=audio_path,
        midi_path=midi_path,
        duration=duration,
        audio_tokens=audio_tokens,
        midi_tokens=midi_tokens,
        audio_hashes=audio_hashes,
        midi_hashes=midi_hashes,
    )


def extract_segment_hashes(
    hashes: List[Tuple[tuple, int]],
    start_frame: int,
    end_frame: int,
    max_hashes: int = MAX_HASHES_PER_SEGMENT,
) -> List[Tuple[tuple, int]]:
    """Extract hashes for a segment, adjusting times relative to segment start."""
    segment_hashes = []
    for h, t_abs in hashes:
        if start_frame <= t_abs < end_frame:
            # Time relative to segment start
            t_rel = t_abs - start_frame
            segment_hashes.append((h, t_rel))

    # Subsample if too many hashes
    if len(segment_hashes) > max_hashes:
        indices = np.random.choice(len(segment_hashes), max_hashes, replace=False)
        segment_hashes = [segment_hashes[i] for i in sorted(indices)]

    return segment_hashes


# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_shazam(
    pieces: List[PieceData],
    segment_len: float = 20.0,
    segment_hop: float = 10.0,
    sr: int = 22050,
    hop_length: int = 512,
) -> Dict:
    """
    Evaluate Shazam-style retrieval:
    - Build DB from all MIDI pieces
    - Query with audio segments
    - Score by offset consensus
    """
    frame_dt = hop_length / sr
    segment_frames = int(segment_len / frame_dt)
    hop_frames = int(segment_hop / frame_dt)

    # Build MIDI database
    print("  Building Shazam DB...", flush=True)
    db = ShazamDB()
    for piece in pieces:
        db.add_piece(piece.piece_id, piece.name, piece.midi_hashes)

    print(f"  DB: {len(db.db)} unique hashes, {db.n_pieces} pieces", flush=True)

    # Generate audio query segments
    queries = []
    for piece in pieces:
        n_frames = len(piece.audio_tokens)
        start = 0
        seg_id = 0
        while start + segment_frames <= n_frames:
            end = start + segment_frames
            seg_hashes = extract_segment_hashes(piece.audio_hashes, start, end)
            queries.append({
                'piece_id': piece.piece_id,
                'segment_id': seg_id,
                'start_frame': start,
                'end_frame': end,
                'hashes': seg_hashes,
                'expected_offset': start,  # Expected MIDI offset
            })
            start += hop_frames
            seg_id += 1

    print(f"  Queries: {len(queries)}")

    # Evaluate
    results_idf = {'correct_piece': 0, 'correct_offset': 0, 'ranks': [], 'offset_errors': []}
    results_no_idf = {'correct_piece': 0, 'correct_offset': 0, 'ranks': [], 'offset_errors': []}

    print(f"  Evaluating {len(queries)} queries...", flush=True)
    for qi, q in enumerate(queries):
        if qi % 20 == 0:
            print(f"    Query {qi+1}/{len(queries)}...", flush=True)
        for use_idf, results in [(True, results_idf), (False, results_no_idf)]:
            matches = db.query(q['hashes'], use_idf=use_idf)

            if not matches:
                results['ranks'].append(len(pieces) + 1)
                continue

            # Rank by score
            ranked = sorted(matches.items(), key=lambda x: -x[1][0])

            # Find rank of true piece
            true_piece = q['piece_id']
            rank = len(pieces) + 1
            for i, (pid, (score, offset)) in enumerate(ranked):
                if pid == true_piece:
                    rank = i + 1
                    pred_offset = offset
                    break

            results['ranks'].append(rank)

            if rank == 1:
                results['correct_piece'] += 1
                # Check offset accuracy
                expected_offset = q['expected_offset']
                offset_error = abs(pred_offset - expected_offset) * frame_dt
                results['offset_errors'].append(offset_error)
                if offset_error < 1.0:  # Within 1 second
                    results['correct_offset'] += 1

    n_queries = len(queries)

    def compute_metrics(results, name):
        ranks = np.array(results['ranks'])
        metrics = {
            'name': name,
            'n_queries': n_queries,
            'n_candidates': len(pieces),
            'piece_accuracy': results['correct_piece'] / n_queries,
            'offset_accuracy_1s': results['correct_offset'] / n_queries if results['correct_piece'] > 0 else 0,
            'recall@1': float((ranks <= 1).mean()),
            'recall@3': float((ranks <= 3).mean()),
            'recall@5': float((ranks <= 5).mean()),
            'mrr': float(np.mean(1.0 / ranks)),
            'mean_rank': float(ranks.mean()),
        }
        if results['offset_errors']:
            metrics['offset_mae'] = float(np.mean(results['offset_errors']))
            metrics['offset_median'] = float(np.median(results['offset_errors']))
        return metrics

    return {
        'with_idf': compute_metrics(results_idf, 'Shazam + IDF'),
        'without_idf': compute_metrics(results_no_idf, 'Shazam (no IDF)'),
    }


def evaluate_segment_vs_segment(
    pieces: List[PieceData],
    segment_len: float = 20.0,
    segment_hop: float = 10.0,
    sr: int = 22050,
    hop_length: int = 512,
) -> Dict:
    """
    Classic segment-to-segment retrieval for comparison.
    """
    frame_dt = hop_length / sr
    segment_frames = int(segment_len / frame_dt)
    hop_frames = int(segment_hop / frame_dt)

    # Generate segments
    audio_segments = []
    midi_segments = []

    for piece in pieces:
        n_frames_a = len(piece.audio_tokens)
        n_frames_m = len(piece.midi_tokens)

        # Audio segments
        start = 0
        seg_id = 0
        while start + segment_frames <= n_frames_a:
            audio_segments.append({
                'piece_id': piece.piece_id,
                'segment_id': seg_id,
                'hashes': set(h for h, _ in extract_segment_hashes(piece.audio_hashes, start, start + segment_frames)),
            })
            start += hop_frames
            seg_id += 1

        # MIDI segments
        start = 0
        seg_id = 0
        while start + segment_frames <= n_frames_m:
            midi_segments.append({
                'piece_id': piece.piece_id,
                'segment_id': seg_id,
                'hashes': set(h for h, _ in extract_segment_hashes(piece.midi_hashes, start, start + segment_frames)),
            })
            start += hop_frames
            seg_id += 1

    # Simple Jaccard scoring
    ranks = []
    aligned_scores = []
    same_piece_scores = []
    random_scores = []

    for a_seg in audio_segments:
        a_hashes = a_seg['hashes']
        if not a_hashes:
            continue

        scores = []
        aligned_idx = None

        for j, m_seg in enumerate(midi_segments):
            m_hashes = m_seg['hashes']

            # Jaccard
            if not m_hashes:
                score = 0
            else:
                intersection = len(a_hashes & m_hashes)
                union = len(a_hashes | m_hashes)
                score = intersection / union if union > 0 else 0

            scores.append(score)

            if m_seg['piece_id'] == a_seg['piece_id'] and m_seg['segment_id'] == a_seg['segment_id']:
                aligned_idx = j
                aligned_scores.append(score)
            elif m_seg['piece_id'] == a_seg['piece_id']:
                same_piece_scores.append(score)
            else:
                random_scores.append(score)

        if aligned_idx is None:
            continue

        # Rank
        aligned_score = scores[aligned_idx]
        rank = sum(1 for s in scores if s > aligned_score) + 1
        ranks.append(rank)

    ranks = np.array(ranks)
    n_queries = len(ranks)
    n_cand = len(midi_segments)

    return {
        'name': 'Segment Jaccard',
        'n_queries': n_queries,
        'n_candidates': n_cand,
        'recall@1': float((ranks <= 1).mean()),
        'recall@5': float((ranks <= 5).mean()),
        'recall@10': float((ranks <= 10).mean()),
        'mrr': float(np.mean(1.0 / ranks)),
        'mean_rank': float(ranks.mean()),
        'aligned_score_mean': float(np.mean(aligned_scores)) if aligned_scores else 0,
        'same_piece_score_mean': float(np.mean(same_piece_scores)) if same_piece_scores else 0,
        'random_score_mean': float(np.mean(random_scores)) if random_scores else 0,
        'gap_vs_same_piece': float(np.mean(aligned_scores) - np.mean(same_piece_scores)) if same_piece_scores else 0,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def visualize_results(shazam_results: Dict, segment_results: Dict, output_path: Path):
    """Create summary visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Recall comparison
    ax = axes[0, 0]
    methods = ['Shazam+IDF', 'Shazam', 'Seg Jaccard']
    recall_1 = [
        shazam_results['with_idf']['recall@1'],
        shazam_results['without_idf']['recall@1'],
        segment_results['recall@1'],
    ]
    recall_5 = [
        shazam_results['with_idf'].get('recall@5', shazam_results['with_idf'].get('recall@3', 0)),
        shazam_results['without_idf'].get('recall@5', shazam_results['without_idf'].get('recall@3', 0)),
        segment_results['recall@5'],
    ]
    x = np.arange(len(methods))
    width = 0.35
    ax.bar(x - width/2, recall_1, width, label='Recall@1', color='steelblue')
    ax.bar(x + width/2, recall_5, width, label='Recall@5', color='coral')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel('Recall')
    ax.set_title('Retrieval Performance')
    ax.legend()
    ax.set_ylim(0, 1)

    # Add random baseline
    n_cand_shazam = shazam_results['with_idf']['n_candidates']
    n_cand_seg = segment_results['n_candidates']
    ax.axhline(1/n_cand_shazam, color='gray', ls='--', alpha=0.5)

    # 2. Piece accuracy (Shazam specialty)
    ax = axes[0, 1]
    piece_acc = [
        shazam_results['with_idf']['piece_accuracy'],
        shazam_results['without_idf']['piece_accuracy'],
    ]
    ax.bar(['Shazam+IDF', 'Shazam'], piece_acc, color=['green', 'lightgreen'])
    ax.set_ylabel('Piece Accuracy')
    ax.set_title('Piece-Level Accuracy (Shazam Offset Consensus)')
    ax.set_ylim(0, 1)

    # 3. Offset accuracy
    ax = axes[1, 0]
    offset_acc = [
        shazam_results['with_idf'].get('offset_accuracy_1s', 0),
        shazam_results['without_idf'].get('offset_accuracy_1s', 0),
    ]
    ax.bar(['Shazam+IDF', 'Shazam'], offset_acc, color=['blue', 'lightblue'])
    ax.set_ylabel('Offset Accuracy (within 1s)')
    ax.set_title('Time Localization Accuracy')
    ax.set_ylim(0, 1)

    # 4. Summary
    ax = axes[1, 1]
    ax.axis('off')

    best = shazam_results['with_idf']
    seg = segment_results

    lines = [
        "=" * 45,
        "SHAZAM EVALUATION SUMMARY",
        "=" * 45,
        "",
        f"Queries: {best['n_queries']}, Pieces: {best['n_candidates']}",
        "",
        "--- SHAZAM + IDF ---",
        f"  Piece Accuracy:  {best['piece_accuracy']:.1%}",
        f"  Offset Acc (<1s): {best.get('offset_accuracy_1s', 0):.1%}",
        f"  Recall@1:        {best['recall@1']:.1%}",
        f"  MRR:             {best['mrr']:.3f}",
        "",
        "--- Segment Jaccard (baseline) ---",
        f"  Recall@1:        {seg['recall@1']:.1%}",
        f"  Gap vs same piece: {seg['gap_vs_same_piece']:.4f}",
        "",
        "=" * 45,
    ]

    # GO/NO-GO
    piece_go = best['piece_accuracy'] > 0.8
    offset_go = best.get('offset_accuracy_1s', 0) > 0.5
    overall_go = piece_go and offset_go

    lines.append(f"Piece GO (>80%): {'✓' if piece_go else '✗'}")
    lines.append(f"Offset GO (>50%): {'✓' if offset_go else '✗'}")
    lines.append("")
    lines.append(f"OVERALL: {'✓ GO' if overall_go else '✗ NO-GO'}")
    lines.append("=" * 45)

    ax.text(0.05, 0.95, '\n'.join(lines), transform=ax.transAxes,
            fontsize=10, va='top', fontfamily='monospace',
            bbox=dict(boxstyle='round',
                     facecolor='lightgreen' if overall_go else 'lightyellow',
                     alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Shazam-style retrieval evaluation")
    parser.add_argument('--input-dir', type=Path,
                        default=Path(__file__).parent / "Varios_pares")
    parser.add_argument('--output-dir', type=Path,
                        default=Path(__file__).parent / "Varios_pares" / "results_shazam")
    parser.add_argument('--segment-len', type=float, default=20.0)
    parser.add_argument('--segment-hop', type=float, default=10.0)
    parser.add_argument('--max-duration', type=float, default=120.0)
    parser.add_argument('--workers', type=int, default=N_WORKERS)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("SHAZAM-STYLE RETRIEVAL EVALUATION")
    print("=" * 70)
    print(f"Input:        {args.input_dir}")
    print(f"Output:       {args.output_dir}")
    print(f"Segment:      {args.segment_len}s (hop {args.segment_hop}s)")
    print(f"Max duration: {args.max_duration}s per piece")
    print("=" * 70)

    audio_files = sorted(args.input_dir.glob("*.wav"))
    print(f"\nFound {len(audio_files)} audio files")

    if len(audio_files) == 0:
        print("ERROR: No WAV files found")
        sys.exit(1)

    # Process all pieces
    pieces = []
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
            piece = process_piece(
                audio_path, midi_path, piece_id=i,
                max_duration=args.max_duration,
            )
            pieces.append(piece)

            print(f"    Duration: {piece.duration:.1f}s")
            print(f"    Audio hashes: {len(piece.audio_hashes):,}")
            print(f"    MIDI hashes:  {len(piece.midi_hashes):,}")
            print(f"    Time: {time.time() - t0:.1f}s")

        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue

    if len(pieces) == 0:
        print("\nERROR: No pieces processed")
        sys.exit(1)

    print(f"\n{'=' * 70}")
    print(f"TOTAL: {len(pieces)} pieces")
    print(f"{'=' * 70}")

    # === SHAZAM EVALUATION ===
    print("\n[EVAL] Shazam-style retrieval...")
    shazam_results = evaluate_shazam(
        pieces,
        segment_len=args.segment_len,
        segment_hop=args.segment_hop,
    )

    print("\n  --- Shazam + IDF ---")
    r = shazam_results['with_idf']
    print(f"  Piece Accuracy:   {r['piece_accuracy']:.1%}")
    print(f"  Offset Acc (<1s): {r.get('offset_accuracy_1s', 0):.1%}")
    print(f"  Recall@1:         {r['recall@1']:.1%}")
    print(f"  Recall@3:         {r['recall@3']:.1%}")
    print(f"  MRR:              {r['mrr']:.4f}")
    if 'offset_mae' in r:
        print(f"  Offset MAE:       {r['offset_mae']:.2f}s")

    print("\n  --- Shazam (no IDF) ---")
    r = shazam_results['without_idf']
    print(f"  Piece Accuracy:   {r['piece_accuracy']:.1%}")
    print(f"  Recall@1:         {r['recall@1']:.1%}")

    # === SEGMENT BASELINE ===
    print("\n[EVAL] Segment-to-segment (baseline)...")
    segment_results = evaluate_segment_vs_segment(
        pieces,
        segment_len=args.segment_len,
        segment_hop=args.segment_hop,
    )

    print(f"  Recall@1:         {segment_results['recall@1']:.1%}")
    print(f"  Gap vs same piece: {segment_results['gap_vs_same_piece']:.4f}")

    # === VISUALIZATION ===
    print("\n[VIS] Creating visualization...")
    visualize_results(shazam_results, segment_results, args.output_dir / "shazam_results.png")

    # === SAVE ===
    results = {
        'config': {
            'n_pieces': len(pieces),
            'segment_len': args.segment_len,
            'hash_config': {
                'dt_bin_size': DT_BIN_SIZE,
                'log_ratio_bin_size': LOG_RATIO_BIN_SIZE,
                'n_anchor_bands': N_ANCHOR_BANDS,
                'offset_bin_size': OFFSET_BIN_SIZE,
            },
        },
        'shazam': shazam_results,
        'segment_baseline': segment_results,
        'total_time': time.time() - total_start,
    }

    with open(args.output_dir / "shazam_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    # === GO/NO-GO ===
    print("\n" + "=" * 70)
    print("GO/NO-GO DECISION (Shazam)")
    print("=" * 70)

    best = shazam_results['with_idf']

    criteria = [
        (f"Piece Accuracy: {best['piece_accuracy']:.1%}", best['piece_accuracy'] > 0.8, "> 80%"),
        (f"Offset Acc (<1s): {best.get('offset_accuracy_1s', 0):.1%}", best.get('offset_accuracy_1s', 0) > 0.5, "> 50%"),
        (f"Recall@1: {best['recall@1']:.1%}", best['recall@1'] > 0.5, "> 50%"),
    ]

    all_pass = True
    for name, passed, threshold in criteria:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name} (need {threshold})")
        if not passed:
            all_pass = False

    print("\n" + "=" * 70)
    if all_pass:
        print("  ✓ GO: Shazam-style retrieval works! Proceed to full MAESTRO.")
    else:
        print("  ✗ NO-GO: Need improvements or more data.")
    print("=" * 70)

    print(f"\nResults: {args.output_dir}")
    print(f"Time: {time.time() - total_start:.1f}s")


if __name__ == '__main__':
    main()
