#!/usr/bin/env python3
"""
Ablation: Test Route A with ONLY chord tokens (which have 72% overlap).

Hypothesis: Removing low-overlap tokens (sequential 8%, constellation 3%)
may improve accuracy by reducing noise.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import List, Tuple, Dict
from collections import Counter, defaultdict
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import librosa
from src.utils.midi_utils import parse_midi
from src.extractors.event_based_extractor import (
    extract_events_from_midi,
    extract_events_from_audio,
    align_audio_events_to_midi,
    extract_all_tokens,
    token_to_hash as event_token_to_hash,
    EVENT_FRAME_RATE,
)
from src.extractors.improved_tf_extractor import SR


OFFSET_BIN_SIZE = 4


class ShazamDB:
    """Shazam-style database for offset voting."""

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

    def query(self, query_hashes, query_t0_abs, use_idf=True, stoplist_pct=0.3):
        stoplist_df = int(self.n_pieces * stoplist_pct)
        votes = defaultdict(lambda: defaultdict(float))

        for h, t_rel in query_hashes:
            if h not in self.index or self.doc_freq[h] > stoplist_df:
                continue

            weight = self.idf(h) if use_idf else 1.0
            t_query_abs = query_t0_abs + t_rel

            for piece_id, t_db_abs in self.index[h][:100]:
                offset = t_db_abs - t_query_abs
                offset_bin = offset // OFFSET_BIN_SIZE
                votes[piece_id][offset_bin] += weight

        results = {}
        for piece_id, offset_votes in votes.items():
            if offset_votes:
                best_offset_bin = max(offset_votes, key=offset_votes.get)
                results[piece_id] = (offset_votes[best_offset_bin], best_offset_bin * OFFSET_BIN_SIZE)
        return results


def run_ablation(input_dir: Path, use_chord=True, use_sequential=True, use_constellation=True):
    """Run ablation with specific token types."""

    config_name = f"chord={use_chord}, seq={use_sequential}, const={use_constellation}"
    print(f"\n{'='*60}")
    print(f"ABLATION: {config_name}")
    print(f"{'='*60}")

    audio_files = sorted(input_dir.glob("*.wav"))

    # Process pieces
    pieces = []
    for i, audio_path in enumerate(audio_files):
        midi_path = audio_path.with_suffix('.midi')
        if not midi_path.exists():
            midi_path = audio_path.with_suffix('.mid')
        if not midi_path.exists():
            continue

        audio, _ = librosa.load(audio_path, sr=SR, mono=True, duration=120.0)
        rms = np.sqrt(np.mean(audio ** 2))
        if rms > 0:
            audio = audio / rms * 0.1
        duration = len(audio) / SR

        notes = parse_midi(midi_path)
        notes = [n for n in notes if n.onset < duration]

        audio_events = extract_events_from_audio(audio, sr=SR)
        midi_events = extract_events_from_midi(notes)
        audio_events = align_audio_events_to_midi(audio_events, midi_events)

        # Extract with specified token types
        audio_tokens = extract_all_tokens(
            audio_events,
            use_chord=use_chord,
            use_sequential=use_sequential,
            use_constellation=use_constellation
        )
        midi_tokens = extract_all_tokens(
            midi_events,
            use_chord=use_chord,
            use_sequential=use_sequential,
            use_constellation=use_constellation
        )

        audio_hashes = [(event_token_to_hash(t), t.t_anchor) for t in audio_tokens]
        midi_hashes = [(event_token_to_hash(t), t.t_anchor) for t in midi_tokens]

        pieces.append((i, audio_hashes, midi_hashes))

    print(f"Processed {len(pieces)} pieces")

    # Build DB
    db = ShazamDB()
    for piece_id, audio_hashes, midi_hashes in pieces:
        db.add_piece(piece_id, midi_hashes)
    print(f"DB: {len(db.index)} unique hashes")

    # Generate queries
    segment_len = 20.0
    segment_hop = 10.0
    segment_frames = int(segment_len * EVENT_FRAME_RATE)
    hop_frames = int(segment_hop * EVENT_FRAME_RATE)

    queries = []
    for piece_id, audio_hashes, midi_hashes in pieces:
        if not audio_hashes:
            continue
        max_t = max(t for h, t in audio_hashes)

        start = 0
        seg_id = 0
        while start + segment_frames // 10 <= max_t + 1:
            seg_hashes = [(h, t - start) for h, t in audio_hashes
                          if start <= t < start + segment_frames // 10]
            if seg_hashes:
                queries.append({
                    'piece_id': piece_id,
                    't0_abs': start,
                    'hashes': seg_hashes,
                })
            start += hop_frames // 10
            seg_id += 1

    print(f"Queries: {len(queries)}")

    # Evaluate
    correct = 0
    ranks = []

    for q in queries:
        result = db.query(q['hashes'], q['t0_abs'])
        if not result:
            ranks.append(len(pieces) + 1)
            continue

        ranked = sorted(result.items(), key=lambda x: -x[1][0])
        true_piece = q['piece_id']
        rank = len(pieces) + 1

        for r, (pid, _) in enumerate(ranked):
            if pid == true_piece:
                rank = r + 1
                break

        ranks.append(rank)
        if rank == 1:
            correct += 1

    n_queries = len(queries)
    ranks = np.array(ranks)

    piece_acc = correct / n_queries
    recall_at_3 = (ranks <= 3).mean()
    recall_at_5 = (ranks <= 5).mean()

    print(f"\nResults:")
    print(f"  Piece Accuracy: {piece_acc:.1%}")
    print(f"  Recall@3:       {recall_at_3:.1%}")
    print(f"  Recall@5:       {recall_at_5:.1%}")
    print(f"  vs Random (5%): {piece_acc / 0.05:.1f}x")

    return piece_acc, recall_at_5


def main():
    input_dir = Path(__file__).parent / "muestra_replicacion"

    print("="*60)
    print("TOKEN TYPE ABLATION STUDY")
    print("="*60)

    results = []

    # Test different combinations
    configs = [
        (True, True, True, "All tokens (baseline)"),
        (True, False, False, "Chord ONLY"),
        (True, True, False, "Chord + Sequential"),
        (True, False, True, "Chord + Constellation"),
        (False, True, True, "Sequential + Constellation (no chord)"),
    ]

    for use_chord, use_seq, use_const, name in configs:
        acc, r5 = run_ablation(input_dir, use_chord, use_seq, use_const)
        results.append((name, acc, r5))

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Configuration':<35} {'Accuracy':>10} {'Recall@5':>10}")
    print("-"*60)
    for name, acc, r5 in results:
        print(f"{name:<35} {acc:>9.1%} {r5:>9.1%}")


if __name__ == '__main__':
    main()
