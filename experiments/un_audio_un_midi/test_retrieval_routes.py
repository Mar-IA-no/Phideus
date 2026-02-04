#!/usr/bin/env python3
"""
Shazam-style Retrieval Test for Route A vs Route B.

Tests actual piece identification using offset-consensus voting.
This is the definitive test to decide which route works better.
"""

from __future__ import annotations

import sys
import json
import time
from pathlib import Path
from typing import List, Tuple, Dict, Set
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
from src.extractors.event_based_extractor import (
    extract_events_from_midi,
    extract_events_from_audio,
    align_audio_events_to_midi,
    extract_all_tokens as extract_event_tokens,
    token_to_hash as event_token_to_hash,
    EVENT_FRAME_RATE,
)
from src.extractors.improved_tf_extractor import (
    extract_improved_tokens_from_audio,
    extract_improved_tokens_from_midi_tf,
    token_to_hash_improved as tf_token_to_hash,
    SR, HOP_LENGTH,
)


# Shazam config
OFFSET_BIN_SIZE = 4  # frames


class ShazamDB:
    """Shazam-style database for offset voting."""

    def __init__(self):
        self.index: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
        self.doc_freq: Counter = Counter()
        self.n_pieces = 0
        self.pieces_with_hash: Dict[int, set] = defaultdict(set)

    def add_piece(self, piece_id: int, hashes: List[Tuple[int, int]]):
        """Add piece to DB. hashes = [(hash_value, t_abs), ...]"""
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
        stoplist_pct: float = 0.3,
        max_matches_per_hash: int = 100,
    ) -> Dict[int, Tuple[float, int]]:
        """Query with offset voting. Returns {piece_id: (score, best_offset)}"""
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


def extract_hashes_route_a(
    audio: np.ndarray,
    notes: List,
    sr: int,
    duration: float,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Extract hashes for Route A (Event-Based)."""
    # Audio events
    audio_events = extract_events_from_audio(audio, sr=sr)
    midi_events = extract_events_from_midi(notes)

    # Align
    audio_events = align_audio_events_to_midi(audio_events, midi_events)

    # Extract tokens
    audio_tokens = extract_event_tokens(audio_events)
    midi_tokens = extract_event_tokens(midi_events)

    # Convert to (hash, t_abs) - use anchor time for t_abs
    audio_hashes = []
    for t in audio_tokens:
        h = event_token_to_hash(t)
        t_abs = t.t_anchor  # Use actual anchor time
        audio_hashes.append((h, t_abs))

    midi_hashes = []
    for t in midi_tokens:
        h = event_token_to_hash(t)
        t_abs = t.t_anchor  # Use actual anchor time
        midi_hashes.append((h, t_abs))

    return audio_hashes, midi_hashes


def extract_hashes_route_b(
    audio: np.ndarray,
    notes: List,
    sr: int,
    duration: float,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Extract hashes for Route B (Improved TF)."""
    # Extract tokens
    audio_tokens, _ = extract_improved_tokens_from_audio(audio, sr=sr)
    midi_tokens, _ = extract_improved_tokens_from_midi_tf(notes, duration=duration, sr=sr)

    # Convert to (hash, t_abs) - use actual anchor time
    audio_hashes = [(tf_token_to_hash(t, use_folded=True), t.t_anchor) for t in audio_tokens]
    midi_hashes = [(tf_token_to_hash(t, use_folded=True), t.t_anchor) for t in midi_tokens]

    return audio_hashes, midi_hashes


def run_retrieval_test(input_dir: Path, output_dir: Path, max_duration: float = 120.0):
    """Run Shazam-style retrieval test."""

    print("=" * 70)
    print("SHAZAM RETRIEVAL TEST: Route A vs Route B")
    print("=" * 70, flush=True)

    segment_len = 20.0  # seconds
    segment_hop = 10.0
    segment_frames_a = int(segment_len * EVENT_FRAME_RATE)
    hop_frames_a = int(segment_hop * EVENT_FRAME_RATE)
    frame_dt_a = 1.0 / EVENT_FRAME_RATE

    frame_dt_b = HOP_LENGTH / SR
    segment_frames_b = int(segment_len / frame_dt_b)
    hop_frames_b = int(segment_hop / frame_dt_b)

    audio_files = sorted(input_dir.glob("*.wav"))
    print(f"Found {len(audio_files)} audio files", flush=True)

    # Process all pieces
    pieces_a = []  # (piece_id, audio_hashes, midi_hashes)
    pieces_b = []

    t0 = time.time()

    for i, audio_path in enumerate(audio_files[:10]):
        midi_path = audio_path.with_suffix('.midi')
        if not midi_path.exists():
            midi_path = audio_path.with_suffix('.mid')
        if not midi_path.exists():
            continue

        print(f"\n[{i+1}] {audio_path.name[:40]}...", flush=True)

        # Load audio
        audio, _ = librosa.load(audio_path, sr=SR, mono=True, duration=max_duration)
        rms = np.sqrt(np.mean(audio ** 2))
        if rms > 0:
            audio = audio / rms * 0.1
        duration = len(audio) / SR

        # Load MIDI
        notes = parse_midi(midi_path)
        notes = [n for n in notes if n.onset < duration]

        # Route A
        print("  Route A...", end=" ", flush=True)
        audio_hashes_a, midi_hashes_a = extract_hashes_route_a(audio, notes, SR, duration)
        print(f"{len(audio_hashes_a)} audio, {len(midi_hashes_a)} midi hashes")

        # Route B
        print("  Route B...", end=" ", flush=True)
        audio_hashes_b, midi_hashes_b = extract_hashes_route_b(audio, notes, SR, duration)
        print(f"{len(audio_hashes_b)} audio, {len(midi_hashes_b)} midi hashes")

        pieces_a.append((i, audio_hashes_a, midi_hashes_a))
        pieces_b.append((i, audio_hashes_b, midi_hashes_b))

    print(f"\n{'=' * 70}")
    print(f"Processed {len(pieces_a)} pieces in {time.time()-t0:.1f}s")
    print("=" * 70, flush=True)

    # Test retrieval for each route
    results = {}

    for route_name, pieces, frame_dt, segment_frames, hop_frames in [
        ("Route A (Event-Based)", pieces_a, frame_dt_a, segment_frames_a, hop_frames_a),
        ("Route B (Improved TF)", pieces_b, frame_dt_b, segment_frames_b, hop_frames_b),
    ]:
        print(f"\n=== {route_name} RETRIEVAL ===\n")

        # Build MIDI DB (full pieces)
        db = ShazamDB()
        for piece_id, audio_hashes, midi_hashes in pieces:
            db.add_piece(piece_id, midi_hashes)
        print(f"DB: {len(db.index)} unique hashes")

        # Generate Audio queries (segments)
        queries = []
        for piece_id, audio_hashes, midi_hashes in pieces:
            # Group by frame (approximate)
            max_t = max(t for h, t in audio_hashes) if audio_hashes else 0
            n_frames = max_t + 1

            start = 0
            seg_id = 0
            while start + segment_frames // 10 <= n_frames:  # Adjust for different frame rates
                seg_hashes = [(h, t - start) for h, t in audio_hashes
                              if start <= t < start + segment_frames // 10]

                if seg_hashes:
                    queries.append({
                        'piece_id': piece_id,
                        'segment_id': seg_id,
                        't0_abs': start,
                        'hashes': seg_hashes,
                    })
                start += hop_frames // 10
                seg_id += 1

        print(f"Queries: {len(queries)}")

        if not queries:
            print("No queries generated!")
            results[route_name] = {
                'piece_accuracy': 0,
                'recall_at_3': 0,
                'recall_at_5': 0,
            }
            continue

        # Evaluate
        correct_piece = 0
        correct_offset = 0
        ranks = []
        offset_errors = []

        for q in queries:
            result = db.query(
                q['hashes'],
                query_t0_abs=q['t0_abs'],
                use_idf=True,
            )

            if not result:
                ranks.append(len(pieces) + 1)
                continue

            ranked = sorted(result.items(), key=lambda x: -x[1][0])
            true_piece = q['piece_id']
            rank = len(pieces) + 1

            for r, (pid, (score, pred_offset)) in enumerate(ranked):
                if pid == true_piece:
                    rank = r + 1
                    break

            ranks.append(rank)

            if rank == 1:
                correct_piece += 1
                # For cross-modal with aligned audio/MIDI, expected offset ~0
                offset_error = abs(pred_offset) * frame_dt
                offset_errors.append(offset_error)
                if offset_error < 1.0:
                    correct_offset += 1

        n_queries = len(queries)
        ranks = np.array(ranks)

        piece_acc = correct_piece / n_queries
        recall_at_3 = (ranks <= 3).mean()
        recall_at_5 = (ranks <= 5).mean()
        offset_acc = correct_offset / n_queries if correct_piece > 0 else 0
        offset_mae = np.mean(offset_errors) if offset_errors else 0

        print(f"\nResults:")
        print(f"  Piece Accuracy (top-1): {piece_acc:.1%}")
        print(f"  Recall@3:               {recall_at_3:.1%}")
        print(f"  Recall@5:               {recall_at_5:.1%}")
        print(f"  Offset Accuracy (<1s):  {offset_acc:.1%}")
        print(f"  Offset MAE:             {offset_mae:.2f}s")

        # Random baseline
        random_baseline = 1 / len(pieces)
        print(f"\n  Random baseline: {random_baseline:.1%}")
        print(f"  Improvement: {piece_acc / random_baseline:.1f}x")

        results[route_name] = {
            'piece_accuracy': float(piece_acc),
            'recall_at_3': float(recall_at_3),
            'recall_at_5': float(recall_at_5),
            'offset_accuracy': float(offset_acc),
            'offset_mae': float(offset_mae),
            'n_queries': n_queries,
            'improvement_over_random': float(piece_acc / random_baseline) if random_baseline > 0 else 0,
        }

    # ================================================================
    # Summary and GO/NO-GO
    # ================================================================
    print("\n" + "=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)

    for route_name, res in results.items():
        print(f"\n{route_name}:")
        print(f"  Piece Accuracy: {res['piece_accuracy']:.1%}")
        print(f"  Recall@5:       {res['recall_at_5']:.1%}")
        print(f"  Improvement:    {res.get('improvement_over_random', 0):.1f}x over random")

        # GO/NO-GO
        piece_acc_pass = res['piece_accuracy'] > 0.50
        recall_pass = res['recall_at_5'] > 0.80

        if piece_acc_pass and recall_pass:
            print(f"  ✓ GO: Cross-modal identification works!")
        elif res['piece_accuracy'] > 0.20:
            print(f"  ⚠ PARTIAL: Better than random but not production-ready")
        else:
            print(f"  ✗ NO-GO: Near random performance")

    # Winner
    route_a_acc = results.get("Route A (Event-Based)", {}).get('piece_accuracy', 0)
    route_b_acc = results.get("Route B (Improved TF)", {}).get('piece_accuracy', 0)

    print("\n" + "=" * 70)
    if route_a_acc > route_b_acc:
        print("WINNER: Route A (Event-Based)")
        print(f"  Margin: +{(route_a_acc - route_b_acc):.1%}")
    elif route_b_acc > route_a_acc:
        print("WINNER: Route B (Improved TF)")
        print(f"  Margin: +{(route_b_acc - route_a_acc):.1%}")
    else:
        print("TIE: Both routes have same accuracy")
    print("=" * 70)

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "retrieval_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=Path,
                        default=Path(__file__).parent / "Varios_pares")
    parser.add_argument('--output-dir', type=Path,
                        default=Path(__file__).parent / "Varios_pares" / "retrieval_routes")
    parser.add_argument('--max-duration', type=float, default=120.0)
    args = parser.parse_args()

    run_retrieval_test(args.input_dir, args.output_dir, args.max_duration)
