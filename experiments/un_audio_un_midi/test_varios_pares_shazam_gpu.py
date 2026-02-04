#!/usr/bin/env python3
"""
Test Varios Pares - SHAZAM-STYLE Retrieval (GPU Optimized)
==========================================================

GPU-accelerated version using PyTorch for fast matching.
"""

from __future__ import annotations

import sys
import json
import time
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field
from collections import Counter, defaultdict
import multiprocessing as mp

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import librosa
    import torch
except ImportError as e:
    print(f"ERROR: {e}")
    sys.exit(1)

from src.utils.midi_utils import parse_midi, Note

from test_single_pair_v2_parallel import (
    extract_audio_constellation_parallel,
    extract_midi_constellation_parallel,
    compare_histograms,
    N_WORKERS,
)

# Check GPU
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# Hash config (balanced: ~5K-20K unique hashes)
DT_BIN_SIZE = 2  # frames
LOG_RATIO_BIN_SIZE = 1/24  # octave (~50 cents)
N_ANCHOR_BANDS = 8


@dataclass
class PieceData:
    piece_id: int
    name: str
    duration: float
    audio_hashes: List[Tuple[int, int]]  # (hash_id, t_abs)
    midi_hashes: List[Tuple[int, int]]


def token_to_hash_id(token: np.ndarray) -> int:
    """Convert token to integer hash ID for fast lookup."""
    log_ratio = token[0]
    delta_t = token[1]
    anchor_band = int(token[3])

    dt_bin = int(delta_t / DT_BIN_SIZE) % 64  # 6 bits
    log_ratio_bin = int(log_ratio / LOG_RATIO_BIN_SIZE) % 64  # 6 bits
    band = min(anchor_band * N_ANCHOR_BANDS // 16, N_ANCHOR_BANDS - 1)  # 2 bits

    # Pack into single int
    return (dt_bin << 8) | (log_ratio_bin << 2) | band


def extract_hashes_fast(tokens_per_frame: List[np.ndarray]) -> List[Tuple[int, int]]:
    """Extract hashes as (hash_id, t_abs) tuples."""
    hashes = []
    for t_frame, frame_tokens in enumerate(tokens_per_frame):
        if len(frame_tokens) == 0:
            continue
        for token in frame_tokens:
            h = token_to_hash_id(token)
            hashes.append((h, t_frame))
    return hashes


def process_piece(audio_path: Path, midi_path: Path, piece_id: int,
                  max_duration: float = 120.0) -> PieceData:
    """Process a single piece."""
    sr = 22050
    hop_length = 512

    audio, _ = librosa.load(audio_path, sr=sr, mono=True, duration=max_duration)
    rms = np.sqrt(np.mean(audio ** 2))
    if rms > 0:
        audio = audio / rms * 0.1
    duration = len(audio) / sr

    audio_tokens, _, _ = extract_audio_constellation_parallel(
        audio, sr=sr, hop_length=hop_length, n_workers=N_WORKERS,
    )

    all_notes = parse_midi(midi_path)
    notes = [n for n in all_notes if n.onset < duration]

    midi_tokens, _, _ = extract_midi_constellation_parallel(
        notes, duration=duration, sr=sr, hop_length=hop_length, n_workers=N_WORKERS,
    )

    return PieceData(
        piece_id=piece_id,
        name=audio_path.stem,
        duration=duration,
        audio_hashes=extract_hashes_fast(audio_tokens),
        midi_hashes=extract_hashes_fast(midi_tokens),
    )


def evaluate_shazam_gpu(
    pieces: List[PieceData],
    segment_len: float = 20.0,
    segment_hop: float = 10.0,
    sr: int = 22050,
    hop_length: int = 512,
) -> Dict:
    """GPU-accelerated Shazam evaluation using histogram matching."""

    frame_dt = hop_length / sr
    segment_frames = int(segment_len / frame_dt)
    hop_frames = int(segment_hop / frame_dt)

    # Collect all unique hashes to build vocabulary
    print("  Building hash vocabulary...", flush=True)
    all_hashes = set()
    for piece in pieces:
        for h, _ in piece.audio_hashes:
            all_hashes.add(h)
        for h, _ in piece.midi_hashes:
            all_hashes.add(h)

    hash_to_idx = {h: i for i, h in enumerate(all_hashes)}
    n_hashes = len(hash_to_idx)
    print(f"  Vocabulary: {n_hashes} unique hashes", flush=True)

    # Build segment histograms (count of each hash per segment)
    print("  Building segment histograms...", flush=True)

    audio_segments = []  # (piece_id, segment_id, start_frame, histogram)
    midi_segments = []

    for piece in pieces:
        n_frames_a = max(h[1] for h in piece.audio_hashes) + 1 if piece.audio_hashes else 0
        n_frames_m = max(h[1] for h in piece.midi_hashes) + 1 if piece.midi_hashes else 0

        # Build frame -> hashes lookup
        audio_frame_hashes = defaultdict(list)
        for h, t in piece.audio_hashes:
            audio_frame_hashes[t].append(h)

        midi_frame_hashes = defaultdict(list)
        for h, t in piece.midi_hashes:
            midi_frame_hashes[t].append(h)

        # Audio segments
        start = 0
        seg_id = 0
        while start + segment_frames <= n_frames_a:
            hist = np.zeros(n_hashes, dtype=np.float32)
            for t in range(start, start + segment_frames):
                for h in audio_frame_hashes[t]:
                    if h in hash_to_idx:
                        hist[hash_to_idx[h]] += 1
            # L2 normalize
            norm = np.linalg.norm(hist)
            if norm > 0:
                hist = hist / norm
            audio_segments.append((piece.piece_id, seg_id, start, hist))
            start += hop_frames
            seg_id += 1

        # MIDI segments
        start = 0
        seg_id = 0
        while start + segment_frames <= n_frames_m:
            hist = np.zeros(n_hashes, dtype=np.float32)
            for t in range(start, start + segment_frames):
                for h in midi_frame_hashes[t]:
                    if h in hash_to_idx:
                        hist[hash_to_idx[h]] += 1
            norm = np.linalg.norm(hist)
            if norm > 0:
                hist = hist / norm
            midi_segments.append((piece.piece_id, seg_id, start, hist))
            start += hop_frames
            seg_id += 1

    print(f"  Audio segments: {len(audio_segments)}", flush=True)
    print(f"  MIDI segments:  {len(midi_segments)}", flush=True)

    if len(audio_segments) == 0 or len(midi_segments) == 0:
        return {'error': 'No segments'}

    # Convert to tensors
    print("  Computing similarity matrix on GPU...", flush=True)
    audio_hists = torch.tensor(np.stack([s[3] for s in audio_segments]), device=DEVICE)
    midi_hists = torch.tensor(np.stack([s[3] for s in midi_segments]), device=DEVICE)

    # Compute cosine similarity matrix (all pairs at once)
    # similarity[i, j] = audio_segment_i · midi_segment_j
    similarities = torch.mm(audio_hists, midi_hists.T).cpu().numpy()

    print("  Evaluating retrieval...", flush=True)

    # Evaluate
    ranks = []
    aligned_scores = []
    same_piece_scores = []
    random_scores = []

    for i, (a_pid, a_sid, a_start, _) in enumerate(audio_segments):
        # Find aligned MIDI segment
        aligned_idx = None
        for j, (m_pid, m_sid, m_start, _) in enumerate(midi_segments):
            if m_pid == a_pid and m_sid == a_sid:
                aligned_idx = j
                break

        if aligned_idx is None:
            continue

        scores = similarities[i]
        aligned_score = scores[aligned_idx]
        aligned_scores.append(aligned_score)

        # Rank
        rank = int((scores > aligned_score).sum()) + 1
        ranks.append(rank)

        # Collect negative scores
        for j, (m_pid, m_sid, _, _) in enumerate(midi_segments):
            if j == aligned_idx:
                continue
            if m_pid == a_pid:
                same_piece_scores.append(scores[j])
            else:
                random_scores.append(scores[j])

    ranks = np.array(ranks)
    n_queries = len(ranks)
    n_cand = len(midi_segments)

    results = {
        'n_queries': n_queries,
        'n_candidates': n_cand,
        'recall@1': float((ranks <= 1).mean()),
        'recall@5': float((ranks <= 5).mean()),
        'recall@10': float((ranks <= 10).mean()),
        'recall@20': float((ranks <= 20).mean()),
        'mrr': float(np.mean(1.0 / ranks)),
        'mean_rank': float(ranks.mean()),
        'aligned_score_mean': float(np.mean(aligned_scores)),
        'same_piece_score_mean': float(np.mean(same_piece_scores)) if same_piece_scores else 0,
        'random_score_mean': float(np.mean(random_scores)) if random_scores else 0,
        'gap_vs_same_piece': float(np.mean(aligned_scores) - np.mean(same_piece_scores)) if same_piece_scores else 0,
        'gap_vs_random': float(np.mean(aligned_scores) - np.mean(random_scores)) if random_scores else 0,
    }

    return results


def evaluate_piece_level_shazam(
    pieces: List[PieceData],
    segment_len: float = 20.0,
    segment_hop: float = 10.0,
    sr: int = 22050,
    hop_length: int = 512,
) -> Dict:
    """
    Shazam-style piece-level evaluation with offset voting.
    Query: audio segment -> Vote for (piece, offset) -> Predict piece + offset
    """
    frame_dt = hop_length / sr
    segment_frames = int(segment_len / frame_dt)
    hop_frames = int(segment_hop / frame_dt)
    offset_bin_size = 4  # frames

    # Build MIDI hash database: hash -> [(piece_id, t_abs), ...]
    print("  Building MIDI hash DB...", flush=True)
    midi_db_raw = defaultdict(list)
    for piece in pieces:
        for h, t in piece.midi_hashes:
            midi_db_raw[h].append((piece.piece_id, t))

    # Limit matches per hash to avoid explosion
    MAX_MATCHES_PER_HASH = 100
    midi_db = {}
    for h, matches in midi_db_raw.items():
        if len(matches) > MAX_MATCHES_PER_HASH:
            # Keep random sample
            indices = np.random.choice(len(matches), MAX_MATCHES_PER_HASH, replace=False)
            midi_db[h] = [matches[i] for i in indices]
        else:
            midi_db[h] = matches

    print(f"  DB size: {len(midi_db)} unique hashes (limited to {MAX_MATCHES_PER_HASH} matches each)", flush=True)

    # Generate audio queries (segments)
    queries = []
    for piece in pieces:
        n_frames = max(h[1] for h in piece.audio_hashes) + 1 if piece.audio_hashes else 0

        # Build frame -> hashes
        frame_hashes = defaultdict(list)
        for h, t in piece.audio_hashes:
            frame_hashes[t].append((h, t))

        start = 0
        seg_id = 0
        while start + segment_frames <= n_frames:
            seg_hashes = []
            for t in range(start, start + segment_frames):
                for h, t_abs in frame_hashes[t]:
                    t_rel = t_abs - start  # Relative to segment start
                    seg_hashes.append((h, t_rel))

            # Subsample if too many
            if len(seg_hashes) > 1000:
                indices = np.random.choice(len(seg_hashes), 1000, replace=False)
                seg_hashes = [seg_hashes[i] for i in indices]

            queries.append({
                'piece_id': piece.piece_id,
                'segment_id': seg_id,
                'start_frame': start,
                'hashes': seg_hashes,
            })
            start += hop_frames
            seg_id += 1

    print(f"  Queries: {len(queries)}", flush=True)

    # Evaluate with offset voting
    print("  Evaluating with offset voting...", flush=True)
    correct_piece = 0
    correct_offset = 0
    ranks = []
    offset_errors = []

    for qi, q in enumerate(queries):
        if qi % 20 == 0:
            print(f"    Query {qi+1}/{len(queries)}...", flush=True)

        # Vote: (piece_id, offset_bin) -> count
        votes = Counter()

        for h, tq in q['hashes']:
            if h not in midi_db:
                continue
            for (pid, tr) in midi_db[h]:
                offset = tr - tq
                offset_bin = offset // offset_bin_size
                votes[(pid, offset_bin)] += 1

        if not votes:
            ranks.append(len(pieces) + 1)
            continue

        # Group by piece, take max offset score
        piece_scores = defaultdict(lambda: (0, 0))  # piece -> (best_score, best_offset_bin)
        for (pid, off_bin), score in votes.items():
            if score > piece_scores[pid][0]:
                piece_scores[pid] = (score, off_bin)

        # Rank pieces
        ranked = sorted(piece_scores.items(), key=lambda x: -x[1][0])

        # Find rank of true piece
        true_piece = q['piece_id']
        rank = len(pieces) + 1
        for i, (pid, (score, off_bin)) in enumerate(ranked):
            if pid == true_piece:
                rank = i + 1
                pred_offset_frames = off_bin * offset_bin_size
                break

        ranks.append(rank)

        if rank == 1:
            correct_piece += 1
            expected_offset = q['start_frame']
            offset_error_s = abs(pred_offset_frames - expected_offset) * frame_dt
            offset_errors.append(offset_error_s)
            if offset_error_s < 1.0:
                correct_offset += 1

    ranks = np.array(ranks)
    n_queries = len(queries)

    results = {
        'n_queries': n_queries,
        'n_pieces': len(pieces),
        'piece_accuracy': correct_piece / n_queries,
        'offset_accuracy_1s': correct_offset / n_queries,
        'recall@1': float((ranks <= 1).mean()),
        'recall@3': float((ranks <= 3).mean()),
        'recall@5': float((ranks <= 5).mean()),
        'mrr': float(np.mean(1.0 / ranks)),
        'mean_rank': float(ranks.mean()),
    }

    if offset_errors:
        results['offset_mae'] = float(np.mean(offset_errors))

    return results


def visualize(seg_results: Dict, piece_results: Dict, output_path: Path):
    """Create visualization."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. Segment retrieval
    ax = axes[0]
    metrics = ['recall@1', 'recall@5', 'recall@10', 'recall@20']
    values = [seg_results.get(m, 0) for m in metrics]
    n_cand = seg_results.get('n_candidates', 110)
    random_baselines = [1/n_cand, 5/n_cand, 10/n_cand, 20/n_cand]

    x = np.arange(len(metrics))
    width = 0.35
    ax.bar(x - width/2, values, width, label='Model', color='steelblue')
    ax.bar(x + width/2, random_baselines, width, label='Random', color='lightgray')
    ax.set_xticks(x)
    ax.set_xticklabels(['@1', '@5', '@10', '@20'])
    ax.set_ylabel('Recall')
    ax.set_title(f'Segment Retrieval (n={seg_results.get("n_queries", 0)})')
    ax.legend()
    ax.set_ylim(0, 1)

    # 2. Gap analysis
    ax = axes[1]
    gaps = [
        seg_results.get('gap_vs_same_piece', 0),
        seg_results.get('gap_vs_random', 0),
    ]
    colors = ['green' if g > 0.05 else 'red' for g in gaps]
    ax.bar(['vs Same Piece', 'vs Random'], gaps, color=colors)
    ax.axhline(0.05, color='red', ls='--', label='GO threshold')
    ax.set_ylabel('Gap')
    ax.set_title('Score Gaps')
    ax.legend()

    # 3. Piece-level (Shazam offset)
    ax = axes[2]
    piece_acc = piece_results.get('piece_accuracy', 0)
    offset_acc = piece_results.get('offset_accuracy_1s', 0)
    ax.bar(['Piece Acc', 'Offset Acc'], [piece_acc, offset_acc],
           color=['green' if piece_acc > 0.8 else 'orange',
                  'green' if offset_acc > 0.5 else 'orange'])
    ax.axhline(0.8, color='red', ls='--', alpha=0.5)
    ax.axhline(0.5, color='red', ls=':', alpha=0.5)
    ax.set_ylabel('Accuracy')
    ax.set_title('Shazam Offset Voting')
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=Path,
                        default=Path(__file__).parent / "Varios_pares")
    parser.add_argument('--output-dir', type=Path,
                        default=Path(__file__).parent / "Varios_pares" / "results_shazam_gpu")
    parser.add_argument('--max-duration', type=float, default=120.0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("SHAZAM EVALUATION (GPU)")
    print("=" * 70, flush=True)

    audio_files = sorted(args.input_dir.glob("*.wav"))
    print(f"Found {len(audio_files)} audio files", flush=True)

    # Process pieces
    pieces = []
    t0 = time.time()

    for i, audio_path in enumerate(audio_files):
        midi_path = audio_path.with_suffix('.midi')
        if not midi_path.exists():
            midi_path = audio_path.with_suffix('.mid')
        if not midi_path.exists():
            continue

        print(f"\n[{i+1}/{len(audio_files)}] {audio_path.name[:40]}...", flush=True)
        piece = process_piece(audio_path, midi_path, i, args.max_duration)
        pieces.append(piece)
        print(f"    Hashes: audio={len(piece.audio_hashes):,}, midi={len(piece.midi_hashes):,}", flush=True)

    print(f"\n{'=' * 70}")
    print(f"Processed {len(pieces)} pieces in {time.time()-t0:.1f}s")
    print("=" * 70, flush=True)

    # Evaluate segment retrieval (GPU)
    print("\n[EVAL 1] Segment retrieval (GPU)...", flush=True)
    seg_results = evaluate_shazam_gpu(pieces)

    print(f"\n  Recall@1:  {seg_results['recall@1']:.1%}")
    print(f"  Recall@5:  {seg_results['recall@5']:.1%}")
    print(f"  Recall@10: {seg_results['recall@10']:.1%}")
    print(f"  Gap vs same piece: {seg_results['gap_vs_same_piece']:.4f}")
    print(f"  Gap vs random:     {seg_results['gap_vs_random']:.4f}")

    # Evaluate piece-level (Shazam offset)
    print("\n[EVAL 2] Piece-level Shazam (offset voting)...", flush=True)
    piece_results = evaluate_piece_level_shazam(pieces)

    print(f"\n  Piece Accuracy:   {piece_results['piece_accuracy']:.1%}")
    print(f"  Offset Acc (<1s): {piece_results['offset_accuracy_1s']:.1%}")
    print(f"  Recall@1:         {piece_results['recall@1']:.1%}")
    if 'offset_mae' in piece_results:
        print(f"  Offset MAE:       {piece_results['offset_mae']:.2f}s")

    # Visualize
    visualize(seg_results, piece_results, args.output_dir / "results.png")

    # Save
    results = {
        'segment_retrieval': seg_results,
        'piece_level_shazam': piece_results,
        'total_time': time.time() - t0,
    }
    with open(args.output_dir / "results.json", 'w') as f:
        json.dump(results, f, indent=2)

    # GO/NO-GO
    print("\n" + "=" * 70)
    print("GO/NO-GO")
    print("=" * 70)

    criteria = [
        (f"Piece Accuracy: {piece_results['piece_accuracy']:.1%}",
         piece_results['piece_accuracy'] > 0.8, "> 80%"),
        (f"Segment Gap (same piece): {seg_results['gap_vs_same_piece']:.4f}",
         seg_results['gap_vs_same_piece'] > 0.05, "> 0.05"),
    ]

    all_pass = True
    for name, passed, threshold in criteria:
        status = "✓" if passed else "✗"
        print(f"  {status} {name} (need {threshold})")
        if not passed:
            all_pass = False

    print("\n" + "=" * 70)
    print(f"OVERALL: {'✓ GO' if all_pass else '✗ NO-GO'}")
    print("=" * 70)
    print(f"\nResults: {args.output_dir}")


if __name__ == '__main__':
    main()
