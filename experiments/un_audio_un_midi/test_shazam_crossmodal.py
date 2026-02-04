#!/usr/bin/env python3
"""
CROSS-MODAL TEST: Audio vs MIDI with corrected Shazam voting.

Uses the same voting logic that passed the Oracle test.
"""

from __future__ import annotations

import sys
import json
import time
from pathlib import Path
from typing import List, Tuple, Dict
from collections import Counter, defaultdict
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

from src.utils.midi_utils import parse_midi

from test_single_pair_v2_parallel import (
    extract_audio_constellation_parallel,
    extract_midi_constellation_parallel,
    N_WORKERS,
)

# Same config as Oracle
DT_BIN_SIZE = 2
LOG_RATIO_BIN_SIZE = 1/24
N_ANCHOR_BANDS = 8
OFFSET_BIN_SIZE = 4


def token_to_hash(token: np.ndarray) -> int:
    log_ratio = token[0]
    delta_t = token[1]
    anchor_band = int(token[3])

    dt_bin = int(delta_t / DT_BIN_SIZE) % 64
    log_ratio_bin = int(log_ratio / LOG_RATIO_BIN_SIZE) % 64
    band = min(anchor_band * N_ANCHOR_BANDS // 16, N_ANCHOR_BANDS - 1)

    return (dt_bin << 8) | (log_ratio_bin << 3) | band


def extract_hashes(tokens_per_frame: List[np.ndarray]) -> List[Tuple[int, int]]:
    """Extract (hash, t_abs) tuples."""
    hashes = []
    for t_frame, frame_tokens in enumerate(tokens_per_frame):
        if len(frame_tokens) == 0:
            continue
        for token in frame_tokens:
            h = token_to_hash(token)
            hashes.append((h, t_frame))
    return hashes


class ShazamDB:
    """Shazam DB indexed by PIECE (not segments)."""

    def __init__(self):
        self.index: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
        self.doc_freq: Counter = Counter()
        self.n_pieces = 0
        self.pieces_with_hash: Dict[int, set] = defaultdict(set)

    def add_piece(self, piece_id: int, hashes: List[Tuple[int, int]]):
        self.n_pieces = max(self.n_pieces, piece_id + 1)
        for h, t_abs in hashes:
            self.index[h].append((piece_id, t_abs))
            self.pieces_with_hash[h].add(piece_id)
        for h in self.pieces_with_hash:
            self.doc_freq[h] = len(self.pieces_with_hash[h])

    def idf(self, h: int) -> float:
        df = self.doc_freq.get(h, 0)
        return np.log((self.n_pieces + 1) / (df + 1)) + 1

    def query(
        self,
        query_hashes: List[Tuple[int, int]],
        query_t0_abs: int,
        use_idf: bool = True,
        stoplist_pct: float = 0.5,
        max_matches_per_hash: int = 100,
    ) -> Dict[int, Tuple[float, int]]:
        stoplist_df = int(self.n_pieces * stoplist_pct)
        votes: Dict[int, Dict[int, float]] = defaultdict(lambda: defaultdict(float))

        for h, t_rel in query_hashes:
            if h not in self.index:
                continue
            if self.doc_freq[h] > stoplist_df:
                continue

            weight = self.idf(h) if use_idf else 1.0
            t_query_abs = query_t0_abs + t_rel

            matches = self.index[h]
            if len(matches) > max_matches_per_hash:
                matches = matches[:max_matches_per_hash]

            for piece_id, t_db_abs in matches:
                offset = t_db_abs - t_query_abs
                offset_bin = offset // OFFSET_BIN_SIZE
                votes[piece_id][offset_bin] += weight

        results = {}
        for piece_id, offset_votes in votes.items():
            if not offset_votes:
                continue
            best_offset_bin = max(offset_votes, key=offset_votes.get)
            best_score = offset_votes[best_offset_bin]
            results[piece_id] = (best_score, best_offset_bin * OFFSET_BIN_SIZE)

        return results


def run_crossmodal_test(input_dir: Path, output_dir: Path, max_duration: float = 120.0):
    """Audio query -> MIDI DB."""

    print("=" * 70)
    print("CROSS-MODAL TEST: Audio vs MIDI")
    print("=" * 70, flush=True)

    sr = 22050
    hop_length = 512
    frame_dt = hop_length / sr
    segment_len = 20.0
    segment_hop = 10.0
    segment_frames = int(segment_len / frame_dt)
    hop_frames = int(segment_hop / frame_dt)

    audio_files = sorted(input_dir.glob("*.wav"))
    print(f"Found {len(audio_files)} audio files", flush=True)

    # Process all pieces
    pieces_audio = []  # (piece_id, hashes, n_frames)
    pieces_midi = []

    t0 = time.time()
    for i, audio_path in enumerate(audio_files[:10]):
        midi_path = audio_path.with_suffix('.midi')
        if not midi_path.exists():
            midi_path = audio_path.with_suffix('.mid')
        if not midi_path.exists():
            continue

        print(f"\n[{i+1}] {audio_path.name[:40]}...", flush=True)

        # Audio
        audio, _ = librosa.load(audio_path, sr=sr, mono=True, duration=max_duration)
        rms = np.sqrt(np.mean(audio ** 2))
        if rms > 0:
            audio = audio / rms * 0.1
        duration = len(audio) / sr

        audio_tokens, _, _ = extract_audio_constellation_parallel(
            audio, sr=sr, hop_length=hop_length, n_workers=N_WORKERS,
        )
        audio_hashes = extract_hashes(audio_tokens)

        # MIDI
        notes = parse_midi(midi_path)
        notes = [n for n in notes if n.onset < duration]

        midi_tokens, _, _ = extract_midi_constellation_parallel(
            notes, duration=duration, sr=sr, hop_length=hop_length, n_workers=N_WORKERS,
        )
        midi_hashes = extract_hashes(midi_tokens)

        pieces_audio.append((i, audio_hashes, len(audio_tokens)))
        pieces_midi.append((i, midi_hashes, len(midi_tokens)))

        print(f"    Audio: {len(audio_hashes):,} hashes, MIDI: {len(midi_hashes):,} hashes", flush=True)

    print(f"\n{'=' * 70}")
    print(f"Processed {len(pieces_audio)} pieces in {time.time()-t0:.1f}s")
    print("=" * 70, flush=True)

    # Build MIDI DB (full pieces)
    print("\nBuilding MIDI DB (full pieces)...", flush=True)
    db = ShazamDB()
    for piece_id, hashes, n_frames in pieces_midi:
        db.add_piece(piece_id, hashes)
    print(f"DB: {len(db.index)} unique hashes", flush=True)

    # Generate Audio queries
    queries = []
    for piece_id, hashes, n_frames in pieces_audio:
        frame_hashes = defaultdict(list)
        for h, t in hashes:
            frame_hashes[t].append(h)

        start = 0
        seg_id = 0
        while start + segment_frames <= n_frames:
            seg_hashes = []
            for t_abs in range(start, start + segment_frames):
                for h in frame_hashes[t_abs]:
                    t_rel = t_abs - start
                    seg_hashes.append((h, t_rel))

            queries.append({
                'piece_id': piece_id,
                'segment_id': seg_id,
                't0_abs': start,
                'hashes': seg_hashes,
            })
            start += hop_frames
            seg_id += 1

    print(f"Queries: {len(queries)}", flush=True)

    # Evaluate
    print("\nEvaluating...", flush=True)
    correct_piece = 0
    correct_offset = 0
    offset_errors = []
    ranks = []

    for qi, q in enumerate(queries):
        if qi % 20 == 0:
            print(f"  Query {qi+1}/{len(queries)}...", flush=True)

        results = db.query(
            q['hashes'],
            query_t0_abs=q['t0_abs'],
            use_idf=True,
        )

        if not results:
            ranks.append(len(pieces_midi) + 1)
            continue

        ranked = sorted(results.items(), key=lambda x: -x[1][0])
        true_piece = q['piece_id']
        rank = len(pieces_midi) + 1

        for i, (pid, (score, pred_offset)) in enumerate(ranked):
            if pid == true_piece:
                rank = i + 1
                break

        ranks.append(rank)

        if rank == 1:
            correct_piece += 1
            # For cross-modal with aligned audio/MIDI, expected offset ~0
            expected_offset = 0
            offset_error_frames = abs(pred_offset - expected_offset)
            offset_error_s = offset_error_frames * frame_dt
            offset_errors.append(offset_error_s)

            if offset_error_s < 1.0:
                correct_offset += 1

    # Results
    n_queries = len(queries)
    ranks = np.array(ranks)
    n_pieces = len(pieces_midi)

    print("\n" + "=" * 70)
    print("CROSS-MODAL RESULTS (Audio vs MIDI)")
    print("=" * 70)
    print(f"Queries: {n_queries}, Pieces: {n_pieces}")
    print(f"Random baseline: top-1 = {1/n_pieces:.1%}, top-5 = {5/n_pieces:.1%}")
    print()
    print(f"Piece Accuracy (top-1): {correct_piece / n_queries:.1%}")
    print(f"Recall@3:               {(ranks <= 3).mean():.1%}")
    print(f"Recall@5:               {(ranks <= 5).mean():.1%}")
    print(f"MRR:                    {np.mean(1.0 / ranks):.3f}")
    print()

    if offset_errors:
        print(f"Offset Accuracy (<1s):  {correct_offset / n_queries:.1%}")
        print(f"Offset MAE:             {np.mean(offset_errors):.2f}s")
    else:
        print("No correct predictions to measure offset")

    # Comparison with random
    print()
    print("Improvement over random:")
    print(f"  Top-1: {(correct_piece/n_queries) / (1/n_pieces):.1f}x")
    print(f"  Top-5: {(ranks <= 5).mean() / (5/n_pieces):.1f}x")

    print()
    print("=" * 70)

    # GO/NO-GO
    piece_acc = correct_piece / n_queries
    offset_acc = correct_offset / n_queries if correct_piece > 0 else 0

    criteria = [
        (f"Piece Acc: {piece_acc:.1%}", piece_acc > 0.5, "> 50%"),
        (f"Recall@5: {(ranks <= 5).mean():.1%}", (ranks <= 5).mean() > 0.8, "> 80%"),
        (f"Offset Acc: {offset_acc:.1%}", offset_acc > 0.3, "> 30%"),
    ]

    all_pass = True
    for name, passed, threshold in criteria:
        status = "✓" if passed else "✗"
        print(f"  {status} {name} (need {threshold})")
        if not passed:
            all_pass = False

    print()
    if all_pass:
        print("✓ CROSS-MODAL GO: Ratio language works for Audio↔MIDI!")
    else:
        print("✗ CROSS-MODAL NO-GO: Need improvements")
    print("=" * 70)

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {
        'n_queries': n_queries,
        'n_pieces': n_pieces,
        'piece_accuracy': correct_piece / n_queries,
        'recall@1': float((ranks <= 1).mean()),
        'recall@3': float((ranks <= 3).mean()),
        'recall@5': float((ranks <= 5).mean()),
        'mrr': float(np.mean(1.0 / ranks)),
        'offset_accuracy_1s': offset_acc,
        'offset_mae': float(np.mean(offset_errors)) if offset_errors else None,
    }
    with open(output_dir / "crossmodal_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    ax = axes[0]
    ks = [1, 3, 5]
    recalls = [(ranks <= k).mean() for k in ks]
    randoms = [k/n_pieces for k in ks]
    x = np.arange(len(ks))
    ax.bar(x - 0.2, recalls, 0.4, label='Model', color='steelblue')
    ax.bar(x + 0.2, randoms, 0.4, label='Random', color='lightgray')
    ax.set_xticks(x)
    ax.set_xticklabels([f'@{k}' for k in ks])
    ax.set_ylabel('Recall')
    ax.set_title('Piece Retrieval')
    ax.legend()

    ax = axes[1]
    ax.hist(offset_errors, bins=20, color='steelblue', edgecolor='white')
    ax.axvline(1.0, color='red', ls='--', label='1s threshold')
    ax.set_xlabel('Offset Error (s)')
    ax.set_ylabel('Count')
    ax.set_title('Offset Error Distribution')
    ax.legend()

    ax = axes[2]
    ax.text(0.5, 0.5, f"Piece Acc: {piece_acc:.1%}\nOffset Acc: {offset_acc:.1%}\n\n{'GO' if all_pass else 'NO-GO'}",
            ha='center', va='center', fontsize=16,
            transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightgreen' if all_pass else 'lightcoral'))
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_dir / "crossmodal_results.png", dpi=150)
    plt.close()

    print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=Path,
                        default=Path(__file__).parent / "Varios_pares")
    parser.add_argument('--output-dir', type=Path,
                        default=Path(__file__).parent / "Varios_pares" / "results_crossmodal")
    parser.add_argument('--max-duration', type=float, default=120.0)
    args = parser.parse_args()

    run_crossmodal_test(args.input_dir, args.output_dir, args.max_duration)
