#!/usr/bin/env python3
"""
Error Analysis for Escalón 1 - Audio ↔ MIDI Cross-Modal Retrieval.

Analyzes why accuracy is ~26% (N=20) and identifies improvement opportunities.

Phases:
1. Characterize errors (which pieces fail, why)
2. Analyze hash distribution (is IDF working?)
3. Identify systematic patterns

Usage:
    python analyze_errors.py --input-dir muestra_replicacion --route A
"""

from __future__ import annotations

import sys
import json
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Set
from collections import Counter, defaultdict
from dataclasses import dataclass
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


@dataclass
class PieceStats:
    """Statistics for a single piece."""
    piece_id: int
    name: str
    duration: float
    n_notes: int
    n_audio_hashes: int
    n_midi_hashes: int
    n_queries: int
    n_correct: int
    accuracy: float
    avg_rank: float
    ranks: List[int]


@dataclass
class HashStats:
    """Statistics for hash distribution."""
    n_unique_hashes: int
    n_total_hashes: int
    df_distribution: Dict[int, int]  # doc_freq -> count
    stopword_hashes: Set[int]
    high_idf_hashes: Set[int]


class ErrorAnalyzer:
    """Analyzes errors in cross-modal retrieval."""

    def __init__(self, route: str = 'A'):
        self.route = route
        self.pieces: List[PieceStats] = []
        self.hash_stats: HashStats = None
        self.confusion_matrix: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self.query_details: List[Dict] = []

        # Route-specific params
        if route == 'A':
            self.frame_rate = EVENT_FRAME_RATE
            self.extract_hashes = self._extract_hashes_route_a
        else:
            self.frame_rate = SR / HOP_LENGTH
            self.extract_hashes = self._extract_hashes_route_b

    def _extract_hashes_route_a(self, audio, notes, sr, duration):
        """Extract hashes using Route A (Event-Based)."""
        audio_events = extract_events_from_audio(audio, sr=sr)
        midi_events = extract_events_from_midi(notes)
        audio_events = align_audio_events_to_midi(audio_events, midi_events)

        audio_tokens = extract_event_tokens(audio_events)
        midi_tokens = extract_event_tokens(midi_events)

        audio_hashes = [(event_token_to_hash(t), t.t_anchor) for t in audio_tokens]
        midi_hashes = [(event_token_to_hash(t), t.t_anchor) for t in midi_tokens]

        return audio_hashes, midi_hashes

    def _extract_hashes_route_b(self, audio, notes, sr, duration):
        """Extract hashes using Route B (Improved TF)."""
        audio_tokens, _ = extract_improved_tokens_from_audio(audio, sr=sr)
        midi_tokens, _ = extract_improved_tokens_from_midi_tf(notes, duration=duration, sr=sr)

        audio_hashes = [(tf_token_to_hash(t, use_folded=True), t.t_anchor) for t in audio_tokens]
        midi_hashes = [(tf_token_to_hash(t, use_folded=True), t.t_anchor) for t in midi_tokens]

        return audio_hashes, midi_hashes

    def load_pieces(self, input_dir: Path, max_duration: float = 120.0):
        """Load and process all pieces."""
        print(f"Loading pieces from {input_dir}...")

        audio_files = sorted(input_dir.glob("*.wav"))
        print(f"Found {len(audio_files)} audio files")

        self.all_audio_hashes = []
        self.all_midi_hashes = []

        for i, audio_path in enumerate(audio_files):
            midi_path = audio_path.with_suffix('.midi')
            if not midi_path.exists():
                midi_path = audio_path.with_suffix('.mid')
            if not midi_path.exists():
                continue

            print(f"  [{i+1}] {audio_path.name[:50]}...", end=" ", flush=True)

            # Load audio
            audio, _ = librosa.load(audio_path, sr=SR, mono=True, duration=max_duration)
            rms = np.sqrt(np.mean(audio ** 2))
            if rms > 0:
                audio = audio / rms * 0.1
            duration = len(audio) / SR

            # Load MIDI
            notes = parse_midi(midi_path)
            notes = [n for n in notes if n.onset < duration]

            # Extract hashes
            audio_hashes, midi_hashes = self.extract_hashes(audio, notes, SR, duration)

            self.all_audio_hashes.append((i, audio_hashes))
            self.all_midi_hashes.append((i, midi_hashes))

            print(f"{len(audio_hashes)} audio, {len(midi_hashes)} midi hashes, {len(notes)} notes")

        print(f"\nLoaded {len(self.all_audio_hashes)} pieces")

    def analyze_hash_distribution(self):
        """Phase 2: Analyze hash distribution."""
        print("\n" + "="*70)
        print("PHASE 2: HASH DISTRIBUTION ANALYSIS")
        print("="*70)

        # Build global hash index
        all_hashes = Counter()
        piece_hashes = defaultdict(set)

        for piece_id, midi_hashes in self.all_midi_hashes:
            for h, t in midi_hashes:
                all_hashes[h] += 1
                piece_hashes[h].add(piece_id)

        n_pieces = len(self.all_midi_hashes)
        n_unique = len(all_hashes)
        n_total = sum(all_hashes.values())

        # Document frequency distribution
        df_dist = Counter()
        for h, pieces in piece_hashes.items():
            df_dist[len(pieces)] += 1

        # Identify stopwords (appear in >30% of pieces)
        stoplist_pct = 0.3
        stoplist_df = int(n_pieces * stoplist_pct)
        stopwords = {h for h, pieces in piece_hashes.items() if len(pieces) > stoplist_df}

        # High IDF hashes (appear in 1-2 pieces)
        high_idf = {h for h, pieces in piece_hashes.items() if len(pieces) <= 2}

        print(f"\n📊 Hash Statistics:")
        print(f"  Total pieces: {n_pieces}")
        print(f"  Unique hashes: {n_unique:,}")
        print(f"  Total hash occurrences: {n_total:,}")
        print(f"  Avg hashes per piece: {n_total / n_pieces:.0f}")

        print(f"\n📈 Document Frequency Distribution:")
        for df in sorted(df_dist.keys()):
            count = df_dist[df]
            pct = count / n_unique * 100
            bar = "█" * int(pct / 2)
            print(f"  DF={df:2d}: {count:5d} hashes ({pct:5.1f}%) {bar}")

        print(f"\n🚫 Stopwords (DF > {stoplist_df}):")
        print(f"  {len(stopwords):,} hashes ({len(stopwords)/n_unique*100:.1f}% of unique)")

        print(f"\n💎 High-IDF hashes (DF <= 2):")
        print(f"  {len(high_idf):,} hashes ({len(high_idf)/n_unique*100:.1f}% of unique)")

        # Analyze overlap between audio and MIDI hashes
        print(f"\n🔗 Cross-Modal Hash Overlap:")
        for piece_id, audio_hashes in self.all_audio_hashes:
            _, midi_hashes = self.all_midi_hashes[piece_id]

            audio_set = set(h for h, t in audio_hashes)
            midi_set = set(h for h, t in midi_hashes)

            overlap = audio_set & midi_set
            overlap_no_stop = overlap - stopwords

            print(f"  Piece {piece_id}: "
                  f"audio={len(audio_set):4d}, midi={len(midi_set):4d}, "
                  f"overlap={len(overlap):4d} ({len(overlap)/len(midi_set)*100:.1f}%), "
                  f"no-stop={len(overlap_no_stop):4d}")

        self.hash_stats = HashStats(
            n_unique_hashes=n_unique,
            n_total_hashes=n_total,
            df_distribution=dict(df_dist),
            stopword_hashes=stopwords,
            high_idf_hashes=high_idf,
        )

        return self.hash_stats

    def run_retrieval_with_details(self, segment_len: float = 20.0, segment_hop: float = 10.0):
        """Run retrieval and collect detailed results."""
        print("\n" + "="*70)
        print("PHASE 1: RETRIEVAL WITH DETAILED ANALYSIS")
        print("="*70)

        # Build MIDI database
        from test_retrieval_routes import ShazamDB, OFFSET_BIN_SIZE

        db = ShazamDB()
        for piece_id, midi_hashes in self.all_midi_hashes:
            db.add_piece(piece_id, midi_hashes)

        print(f"DB: {len(db.index):,} unique hashes from {len(self.all_midi_hashes)} pieces")

        # Generate queries
        frame_dt = 1.0 / self.frame_rate
        segment_frames = int(segment_len * self.frame_rate)
        hop_frames = int(segment_hop * self.frame_rate)

        queries = []
        for piece_id, audio_hashes in self.all_audio_hashes:
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
                        'segment_id': seg_id,
                        't0_abs': start,
                        'hashes': seg_hashes,
                    })
                start += hop_frames // 10
                seg_id += 1

        print(f"Generated {len(queries)} queries")

        # Run queries and collect details
        piece_results = defaultdict(lambda: {'correct': 0, 'total': 0, 'ranks': []})

        for q in queries:
            result = db.query(q['hashes'], q['t0_abs'], use_idf=True)

            if not result:
                piece_results[q['piece_id']]['total'] += 1
                piece_results[q['piece_id']]['ranks'].append(len(self.all_midi_hashes) + 1)
                continue

            ranked = sorted(result.items(), key=lambda x: -x[1][0])
            true_piece = q['piece_id']

            rank = len(self.all_midi_hashes) + 1
            predicted = ranked[0][0] if ranked else -1

            for r, (pid, (score, offset)) in enumerate(ranked):
                if pid == true_piece:
                    rank = r + 1
                    break

            piece_results[true_piece]['total'] += 1
            piece_results[true_piece]['ranks'].append(rank)
            if rank == 1:
                piece_results[true_piece]['correct'] += 1

            # Track confusion
            if rank > 1 and predicted != true_piece:
                self.confusion_matrix[true_piece][predicted] += 1

            self.query_details.append({
                'piece_id': true_piece,
                'segment_id': q['segment_id'],
                'rank': rank,
                'predicted': predicted,
                'n_hashes': len(q['hashes']),
                'top_score': ranked[0][1][0] if ranked else 0,
                'true_score': result.get(true_piece, (0, 0))[0],
            })

        # Summarize by piece
        print(f"\n📊 Per-Piece Results:")
        print(f"{'Piece':>5} {'Name':<40} {'Acc':>6} {'Avg Rank':>8} {'Queries':>7}")
        print("-" * 70)

        total_correct = 0
        total_queries = 0

        for piece_id, audio_hashes in self.all_audio_hashes:
            r = piece_results[piece_id]
            acc = r['correct'] / r['total'] if r['total'] > 0 else 0
            avg_rank = np.mean(r['ranks']) if r['ranks'] else 0

            total_correct += r['correct']
            total_queries += r['total']

            # Get piece name
            name = f"piece_{piece_id}"  # Would need to track actual names

            status = "✓" if acc > 0.5 else "✗" if acc < 0.1 else "~"
            print(f"{piece_id:>5} {name:<40} {acc:>5.1%} {avg_rank:>8.1f} {r['total']:>7} {status}")

        print("-" * 70)
        overall_acc = total_correct / total_queries if total_queries > 0 else 0
        print(f"{'TOTAL':>5} {'':<40} {overall_acc:>5.1%} {'':<8} {total_queries:>7}")

        return piece_results

    def analyze_confusion(self):
        """Analyze which pieces get confused with which."""
        print("\n" + "="*70)
        print("CONFUSION ANALYSIS")
        print("="*70)

        if not self.confusion_matrix:
            print("No confusion data (all queries correct or no data)")
            return

        print("\n🔄 Most Common Confusions:")
        print(f"{'True':>6} -> {'Pred':>6} {'Count':>6}")
        print("-" * 25)

        confusions = []
        for true_id, preds in self.confusion_matrix.items():
            for pred_id, count in preds.items():
                confusions.append((true_id, pred_id, count))

        confusions.sort(key=lambda x: -x[2])
        for true_id, pred_id, count in confusions[:20]:
            print(f"{true_id:>6} -> {pred_id:>6} {count:>6}")

    def analyze_query_details(self):
        """Analyze query-level patterns."""
        print("\n" + "="*70)
        print("QUERY-LEVEL ANALYSIS")
        print("="*70)

        if not self.query_details:
            print("No query details")
            return

        # Analyze by hash count
        hash_bins = defaultdict(lambda: {'correct': 0, 'total': 0})
        for q in self.query_details:
            bin_id = q['n_hashes'] // 50 * 50  # Bin by 50s
            hash_bins[bin_id]['total'] += 1
            if q['rank'] == 1:
                hash_bins[bin_id]['correct'] += 1

        print("\n📊 Accuracy by Query Hash Count:")
        print(f"{'Hashes':>10} {'Queries':>8} {'Correct':>8} {'Accuracy':>8}")
        print("-" * 40)

        for bin_id in sorted(hash_bins.keys()):
            b = hash_bins[bin_id]
            acc = b['correct'] / b['total'] if b['total'] > 0 else 0
            print(f"{bin_id:>7}-{bin_id+49:<3} {b['total']:>8} {b['correct']:>8} {acc:>7.1%}")

        # Analyze score gap
        print("\n📊 Score Gap Analysis (correct vs incorrect):")
        correct_gaps = []
        incorrect_gaps = []

        for q in self.query_details:
            gap = q['true_score'] - q['top_score'] if q['rank'] > 1 else 0
            if q['rank'] == 1:
                correct_gaps.append(q['top_score'])
            else:
                incorrect_gaps.append(gap)

        if correct_gaps:
            print(f"  Correct queries: avg_score = {np.mean(correct_gaps):.2f}")
        if incorrect_gaps:
            print(f"  Incorrect queries: avg_gap = {np.mean(incorrect_gaps):.2f}")

    def generate_report(self, output_dir: Path):
        """Generate analysis report."""
        output_dir.mkdir(parents=True, exist_ok=True)

        report = {
            'route': self.route,
            'n_pieces': len(self.all_audio_hashes),
            'hash_stats': {
                'n_unique': self.hash_stats.n_unique_hashes,
                'n_total': self.hash_stats.n_total_hashes,
                'df_distribution': self.hash_stats.df_distribution,
                'n_stopwords': len(self.hash_stats.stopword_hashes),
                'n_high_idf': len(self.hash_stats.high_idf_hashes),
            } if self.hash_stats else None,
            'query_details': self.query_details[:100],  # Sample
        }

        with open(output_dir / 'error_analysis.json', 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\n💾 Report saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Error Analysis for Escalón 1')
    parser.add_argument('--input-dir', type=Path,
                        default=Path(__file__).parent / "muestra_replicacion")
    parser.add_argument('--output-dir', type=Path,
                        default=Path(__file__).parent / "error_analysis")
    parser.add_argument('--route', choices=['A', 'B'], default='A')
    parser.add_argument('--max-duration', type=float, default=120.0)
    args = parser.parse_args()

    print("="*70)
    print(f"ERROR ANALYSIS - Route {args.route}")
    print("="*70)

    analyzer = ErrorAnalyzer(route=args.route)

    # Load pieces
    analyzer.load_pieces(args.input_dir, args.max_duration)

    # Phase 2: Hash distribution (high priority)
    analyzer.analyze_hash_distribution()

    # Phase 1: Retrieval with details
    analyzer.run_retrieval_with_details()

    # Confusion analysis
    analyzer.analyze_confusion()

    # Query-level patterns
    analyzer.analyze_query_details()

    # Save report
    analyzer.generate_report(args.output_dir)

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()
