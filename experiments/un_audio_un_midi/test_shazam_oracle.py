#!/usr/bin/env python3
"""
ORACLE TEST: Verify Shazam offset voting works correctly.

If MIDI→TF vs MIDI→TF doesn't work, the bug is in voting implementation.
If it works, the problem is cross-modal, not the algorithm.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import List, Tuple, Dict
from collections import Counter, defaultdict
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import librosa
except ImportError:
    print("ERROR: librosa required")
    sys.exit(1)

from src.utils.midi_utils import parse_midi

from test_single_pair_v2_parallel import (
    extract_midi_constellation_parallel,
    N_WORKERS,
)

# Hash config
DT_BIN_SIZE = 2  # frames (~46ms)
LOG_RATIO_BIN_SIZE = 1/24  # ~50 cents
N_ANCHOR_BANDS = 8
OFFSET_BIN_SIZE = 4  # frames (~92ms)


def token_to_hash(token: np.ndarray) -> int:
    """Convert token to hash ID."""
    log_ratio = token[0]
    delta_t = token[1]
    anchor_band = int(token[3])

    dt_bin = int(delta_t / DT_BIN_SIZE) % 64
    log_ratio_bin = int(log_ratio / LOG_RATIO_BIN_SIZE) % 64
    band = min(anchor_band * N_ANCHOR_BANDS // 16, N_ANCHOR_BANDS - 1)

    return (dt_bin << 8) | (log_ratio_bin << 3) | band


def extract_hashes_with_abs_time(
    tokens_per_frame: List[np.ndarray],
) -> List[Tuple[int, int]]:
    """Extract (hash, t_abs) tuples where t_abs is ABSOLUTE frame from piece start."""
    hashes = []
    for t_frame, frame_tokens in enumerate(tokens_per_frame):
        if len(frame_tokens) == 0:
            continue
        for token in frame_tokens:
            h = token_to_hash(token)
            hashes.append((h, t_frame))  # t_frame is absolute within piece
    return hashes


class ShazamDB:
    """Shazam DB indexed by PIECE, not segments."""

    def __init__(self):
        # hash -> [(piece_id, t_abs), ...]
        self.index: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
        self.doc_freq: Counter = Counter()  # hash -> number of pieces containing it
        self.n_pieces = 0
        self.pieces_with_hash: Dict[int, set] = defaultdict(set)

    def add_piece(self, piece_id: int, hashes: List[Tuple[int, int]]):
        """Add entire piece's hashes to DB."""
        self.n_pieces = max(self.n_pieces, piece_id + 1)

        for h, t_abs in hashes:
            self.index[h].append((piece_id, t_abs))
            self.pieces_with_hash[h].add(piece_id)

        # Update doc freq
        for h in self.pieces_with_hash:
            self.doc_freq[h] = len(self.pieces_with_hash[h])

    def idf(self, h: int) -> float:
        """IDF with smoothing."""
        df = self.doc_freq.get(h, 0)
        return np.log((self.n_pieces + 1) / (df + 1)) + 1

    def query(
        self,
        query_hashes: List[Tuple[int, int]],
        query_t0_abs: int,  # Absolute start time of query segment
        use_idf: bool = True,
        stoplist_pct: float = 0.5,  # Ignore hashes in >50% of pieces
        max_matches_per_hash: int = 100,
    ) -> Dict[int, Tuple[float, int]]:
        """
        Query with offset voting.

        Args:
            query_hashes: [(hash, t_rel)] where t_rel is relative to segment start
            query_t0_abs: Absolute start frame of the segment within the piece

        Returns: {piece_id: (score, best_offset_frames)}
        """
        # Stoplist threshold
        stoplist_df = int(self.n_pieces * stoplist_pct)

        # Vote accumulator: piece_id -> offset_bin -> score
        votes: Dict[int, Dict[int, float]] = defaultdict(lambda: defaultdict(float))

        for h, t_rel in query_hashes:
            if h not in self.index:
                continue

            # Skip common hashes (stoplist)
            if self.doc_freq[h] > stoplist_df:
                continue

            weight = self.idf(h) if use_idf else 1.0

            # CRITICAL: Convert query time to ABSOLUTE
            t_query_abs = query_t0_abs + t_rel

            matches = self.index[h]
            # Cap matches
            if len(matches) > max_matches_per_hash:
                matches = matches[:max_matches_per_hash]

            for piece_id, t_db_abs in matches:
                # Offset = DB time - Query time (both absolute)
                offset = t_db_abs - t_query_abs
                offset_bin = offset // OFFSET_BIN_SIZE

                votes[piece_id][offset_bin] += weight

        # Find best offset for each piece (PEAK, not sum)
        results = {}
        for piece_id, offset_votes in votes.items():
            if not offset_votes:
                continue
            best_offset_bin = max(offset_votes, key=offset_votes.get)
            best_score = offset_votes[best_offset_bin]
            results[piece_id] = (best_score, best_offset_bin * OFFSET_BIN_SIZE)

        return results


def run_oracle_test(input_dir: Path, max_duration: float = 120.0):
    """
    ORACLE TEST: MIDI vs MIDI (same extraction).

    If this doesn't work, voting is broken.
    """
    print("=" * 70)
    print("ORACLE TEST: MIDI vs MIDI")
    print("=" * 70)
    print("This tests if offset voting works correctly.")
    print("Expected: Piece Acc ~100%, Offset MAE ~0s")
    print("=" * 70, flush=True)

    sr = 22050
    hop_length = 512
    frame_dt = hop_length / sr
    segment_len = 20.0  # seconds
    segment_hop = 10.0
    segment_frames = int(segment_len / frame_dt)
    hop_frames = int(segment_hop / frame_dt)

    # Find MIDI files
    midi_files = sorted(input_dir.glob("*.midi")) + sorted(input_dir.glob("*.mid"))
    print(f"\nFound {len(midi_files)} MIDI files", flush=True)

    if len(midi_files) == 0:
        print("ERROR: No MIDI files")
        return

    # Process pieces (extract MIDI tokens)
    pieces_data = []  # (piece_id, name, hashes, n_frames)

    for i, midi_path in enumerate(midi_files[:10]):  # Limit to 10
        print(f"\n[{i+1}] Processing {midi_path.name[:40]}...", flush=True)

        notes = parse_midi(midi_path)
        if not notes:
            continue

        duration = min(max(n.offset for n in notes), max_duration)
        notes = [n for n in notes if n.onset < duration]

        tokens, times, _ = extract_midi_constellation_parallel(
            notes, duration=duration, sr=sr, hop_length=hop_length,
            n_workers=N_WORKERS,
        )

        hashes = extract_hashes_with_abs_time(tokens)
        n_frames = len(tokens)

        pieces_data.append((i, midi_path.stem, hashes, n_frames))
        print(f"    Frames: {n_frames}, Hashes: {len(hashes):,}", flush=True)

    print(f"\n{'=' * 70}")
    print(f"Total: {len(pieces_data)} pieces")
    print("=" * 70, flush=True)

    # Build DB from ENTIRE pieces
    print("\nBuilding Shazam DB (full pieces)...", flush=True)
    db = ShazamDB()
    for piece_id, name, hashes, n_frames in pieces_data:
        db.add_piece(piece_id, hashes)

    print(f"DB: {len(db.index)} unique hashes", flush=True)

    # Generate queries (segments from SAME MIDI)
    queries = []
    for piece_id, name, hashes, n_frames in pieces_data:
        # Build frame -> hashes lookup
        frame_hashes = defaultdict(list)
        for h, t in hashes:
            frame_hashes[t].append(h)

        start = 0
        seg_id = 0
        while start + segment_frames <= n_frames:
            # Extract segment hashes with RELATIVE times
            seg_hashes = []
            for t_abs in range(start, start + segment_frames):
                for h in frame_hashes[t_abs]:
                    t_rel = t_abs - start  # Relative to segment
                    seg_hashes.append((h, t_rel))

            queries.append({
                'piece_id': piece_id,
                'segment_id': seg_id,
                't0_abs': start,  # ABSOLUTE start frame
                'hashes': seg_hashes,
            })

            start += hop_frames
            seg_id += 1

    print(f"Queries: {len(queries)}", flush=True)

    # Evaluate
    print("\nEvaluating offset voting...", flush=True)
    correct_piece = 0
    correct_offset = 0
    offset_errors = []
    ranks = []

    for qi, q in enumerate(queries):
        if qi % 20 == 0:
            print(f"  Query {qi+1}/{len(queries)}...", flush=True)

        # Query with ABSOLUTE t0
        results = db.query(
            q['hashes'],
            query_t0_abs=q['t0_abs'],  # CRITICAL: pass absolute time
            use_idf=True,
            stoplist_pct=0.5,
            max_matches_per_hash=100,
        )

        if not results:
            ranks.append(len(pieces_data) + 1)
            continue

        # Rank by score
        ranked = sorted(results.items(), key=lambda x: -x[1][0])

        true_piece = q['piece_id']
        rank = len(pieces_data) + 1

        for i, (pid, (score, pred_offset)) in enumerate(ranked):
            if pid == true_piece:
                rank = i + 1
                break

        ranks.append(rank)

        if rank == 1:
            correct_piece += 1
            # Check offset
            # For MIDI vs MIDI: offset should be ~0 because query and DB are same source
            # pred_offset = t_db_abs - t_query_abs
            # If query is from frame 100, and matches DB at frame 100, offset = 0
            expected_offset = 0  # Same source = no offset
            offset_error_frames = abs(pred_offset - expected_offset)
            offset_error_s = offset_error_frames * frame_dt
            offset_errors.append(offset_error_s)

            # Debug first few
            if qi < 5:
                print(f"    DEBUG: query t0={q['t0_abs']}, pred_offset={pred_offset}, error={offset_error_s:.2f}s")

            if offset_error_s < 1.0:
                correct_offset += 1

    # Results
    n_queries = len(queries)
    ranks = np.array(ranks)

    print("\n" + "=" * 70)
    print("ORACLE RESULTS (MIDI vs MIDI)")
    print("=" * 70)
    print(f"Queries: {n_queries}, Pieces: {len(pieces_data)}")
    print()
    print(f"Piece Accuracy (top-1): {correct_piece / n_queries:.1%}")
    print(f"Recall@3:               {(ranks <= 3).mean():.1%}")
    print(f"Recall@5:               {(ranks <= 5).mean():.1%}")
    print()

    if offset_errors:
        print(f"Offset Accuracy (<1s):  {correct_offset / n_queries:.1%}")
        print(f"Offset MAE:             {np.mean(offset_errors):.2f}s")
        print(f"Offset Median:          {np.median(offset_errors):.2f}s")
    else:
        print("No correct piece predictions to measure offset")

    print()
    print("=" * 70)

    # GO/NO-GO for oracle
    piece_acc = correct_piece / n_queries
    offset_acc = correct_offset / n_queries if correct_piece > 0 else 0

    if piece_acc > 0.9 and offset_acc > 0.8:
        print("✓ ORACLE PASS: Voting implementation is correct!")
        print("  Problem is cross-modal, not algorithm.")
    elif piece_acc > 0.5:
        print("⚠ ORACLE PARTIAL: Voting mostly works but needs tuning")
    else:
        print("✗ ORACLE FAIL: Voting implementation has bugs!")
        print("  Fix voting before testing cross-modal.")

    print("=" * 70)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=Path,
                        default=Path(__file__).parent / "Varios_pares")
    parser.add_argument('--max-duration', type=float, default=120.0)
    args = parser.parse_args()

    run_oracle_test(args.input_dir, args.max_duration)
