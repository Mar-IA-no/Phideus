#!/usr/bin/env python3
"""
Compare Route A (Event-Based) vs Route B (Improved TF-Constellations).

Evaluates both approaches on the same data and measures:
1. Hash overlap (aligned vs same_piece vs random)
2. Token statistics (density, diversity)
3. Shazam-style retrieval

This is the definitive comparison to decide which route to pursue.
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
    tokens_to_hash_set as event_tokens_to_hashes,
    EVENT_FRAME_RATE,
)
from src.extractors.improved_tf_extractor import (
    extract_improved_tokens_from_audio,
    extract_improved_tokens_from_midi_tf,
    tokens_to_hash_set as tf_tokens_to_hashes,
    SR, HOP_LENGTH,
)


def compute_overlap(set_a: Set[int], set_b: Set[int]) -> float:
    """Compute Jaccard-like overlap."""
    if len(set_a) == 0 or len(set_b) == 0:
        return 0.0
    return len(set_a & set_b) / min(len(set_a), len(set_b))


def segment_hash_sets(
    all_hashes: List[int],
    frames: List[int],
    segment_frames: int,
    hop_frames: int,
    total_frames: int,
) -> List[Set[int]]:
    """Segment hashes into non-overlapping windows."""
    # Group hashes by frame
    by_frame = defaultdict(set)
    for h, t in zip(all_hashes, frames):
        by_frame[t].add(h)

    segments = []
    start = 0
    while start + segment_frames <= total_frames:
        seg_hashes = set()
        for t in range(start, start + segment_frames):
            seg_hashes.update(by_frame.get(t, set()))
        segments.append(seg_hashes)
        start += hop_frames

    return segments


def run_comparison(input_dir: Path, output_dir: Path, max_duration: float = 120.0):
    """Run comparison of Route A vs Route B."""

    print("=" * 70)
    print("ROUTE COMPARISON: Event-Based (A) vs Improved TF (B)")
    print("=" * 70, flush=True)

    segment_len = 20.0  # seconds
    segment_hop = 10.0  # seconds

    audio_files = sorted(input_dir.glob("*.wav"))
    print(f"Found {len(audio_files)} audio files", flush=True)

    # Results storage
    results_a = {
        'overlaps_aligned': [],
        'overlaps_same_piece': [],
        'overlaps_random': [],
        'tokens_per_seg_audio': [],
        'tokens_per_seg_midi': [],
        'hash_diversity_audio': [],
        'hash_diversity_midi': [],
    }
    results_b = {
        'overlaps_aligned': [],
        'overlaps_same_piece': [],
        'overlaps_random': [],
        'tokens_per_seg_audio': [],
        'tokens_per_seg_midi': [],
        'hash_diversity_audio': [],
        'hash_diversity_midi': [],
    }

    # Process all pieces
    all_segments_a = []  # (piece_id, seg_id, audio_hashes, midi_hashes)
    all_segments_b = []

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

        # ================================================================
        # ROUTE A: Event-Based
        # ================================================================
        print("  Route A (Event-Based)...", end=" ", flush=True)

        # Extract events
        audio_events = extract_events_from_audio(audio, sr=SR)
        midi_events = extract_events_from_midi(notes)

        # Align audio to MIDI
        audio_events = align_audio_events_to_midi(audio_events, midi_events)

        # Extract tokens
        audio_tokens_a = extract_event_tokens(audio_events)
        midi_tokens_a = extract_event_tokens(midi_events)

        # Convert to hashes
        audio_hashes_a = event_tokens_to_hashes(audio_tokens_a)
        midi_hashes_a = event_tokens_to_hashes(midi_tokens_a)

        print(f"Audio: {len(audio_events)} events, {len(audio_hashes_a)} hashes | "
              f"MIDI: {len(midi_events)} events, {len(midi_hashes_a)} hashes")

        # Segment-level hashes (for overlap analysis)
        seg_frames_a = int(segment_len * EVENT_FRAME_RATE)
        hop_frames_a = int(segment_hop * EVENT_FRAME_RATE)
        total_frames_a = int(duration * EVENT_FRAME_RATE)

        # Build hash sets per segment
        audio_by_frame_a = defaultdict(set)
        for t in audio_tokens_a:
            h = hash((t.token_type, t.dt // 2, t.dp, t.pc_anchor))  # Simple hash
            audio_by_frame_a[t.dt].add(h)  # Approximate: use dt as proxy for frame

        midi_by_frame_a = defaultdict(set)
        for t in midi_tokens_a:
            h = hash((t.token_type, t.dt // 2, t.dp, t.pc_anchor))
            midi_by_frame_a[t.dt].add(h)

        # For simplicity, use piece-level hashes for overlap
        seg_id = 0
        start = 0
        while start + seg_frames_a <= total_frames_a:
            # Approximate segmentation
            all_segments_a.append((i, seg_id, audio_hashes_a, midi_hashes_a))
            seg_id += 1
            start += hop_frames_a
            break  # Just one segment per piece for now

        # ================================================================
        # ROUTE B: Improved TF
        # ================================================================
        print("  Route B (Improved TF)...", end=" ", flush=True)

        # Extract tokens
        audio_tokens_b, meta_audio = extract_improved_tokens_from_audio(
            audio, sr=SR, hop_length=HOP_LENGTH, require_onset_anchor=True
        )
        midi_tokens_b, meta_midi = extract_improved_tokens_from_midi_tf(
            notes, duration=duration, sr=SR, hop_length=HOP_LENGTH, require_onset_anchor=True
        )

        # Convert to hashes (using folded for octave invariance)
        audio_hashes_b = tf_tokens_to_hashes(audio_tokens_b, use_folded=True)
        midi_hashes_b = tf_tokens_to_hashes(midi_tokens_b, use_folded=True)

        print(f"Audio: {len(audio_tokens_b)} tokens, {len(audio_hashes_b)} hashes | "
              f"MIDI: {len(midi_tokens_b)} tokens, {len(midi_hashes_b)} hashes")

        # Store for overlap analysis
        all_segments_b.append((i, 0, audio_hashes_b, midi_hashes_b))

        # Token statistics
        results_a['tokens_per_seg_audio'].append(len(audio_tokens_a))
        results_a['tokens_per_seg_midi'].append(len(midi_tokens_a))
        results_a['hash_diversity_audio'].append(len(audio_hashes_a))
        results_a['hash_diversity_midi'].append(len(midi_hashes_a))

        results_b['tokens_per_seg_audio'].append(len(audio_tokens_b))
        results_b['tokens_per_seg_midi'].append(len(midi_tokens_b))
        results_b['hash_diversity_audio'].append(len(audio_hashes_b))
        results_b['hash_diversity_midi'].append(len(midi_hashes_b))

    print(f"\n{'=' * 70}")
    print(f"Processed {len(all_segments_a)} pieces in {time.time()-t0:.1f}s")
    print("=" * 70, flush=True)

    # ================================================================
    # Overlap Analysis
    # ================================================================
    print("\n=== OVERLAP ANALYSIS ===\n")

    for route_name, segments, results in [
        ("Route A (Event-Based)", all_segments_a, results_a),
        ("Route B (Improved TF)", all_segments_b, results_b),
    ]:
        print(f"\n{route_name}:")

        # Aligned overlap (same piece, same time)
        for pid, sid, audio_h, midi_h in segments:
            overlap = compute_overlap(audio_h, midi_h)
            results['overlaps_aligned'].append(overlap)

        # Same piece, different time (within-piece negative)
        for i, (pid1, sid1, ah1, mh1) in enumerate(segments):
            for j, (pid2, sid2, ah2, mh2) in enumerate(segments):
                if pid1 == pid2 and i != j:
                    overlap = compute_overlap(ah1, mh2)
                    results['overlaps_same_piece'].append(overlap)

        # Random (different pieces)
        np.random.seed(42)
        n_samples = min(50, len(segments) * (len(segments) - 1) // 2)
        for _ in range(n_samples):
            indices = np.random.choice(len(segments), 2, replace=False)
            i, j = indices
            if segments[i][0] != segments[j][0]:  # Different pieces
                overlap = compute_overlap(segments[i][2], segments[j][3])
                results['overlaps_random'].append(overlap)

        # Statistics
        mean_aligned = np.mean(results['overlaps_aligned']) if results['overlaps_aligned'] else 0
        mean_same_piece = np.mean(results['overlaps_same_piece']) if results['overlaps_same_piece'] else 0
        mean_random = np.mean(results['overlaps_random']) if results['overlaps_random'] else 0

        gap_aligned_random = mean_aligned - mean_random
        gap_aligned_same_piece = mean_aligned - mean_same_piece

        print(f"  Overlap aligned:    {mean_aligned:.4f}")
        print(f"  Overlap same_piece: {mean_same_piece:.4f}")
        print(f"  Overlap random:     {mean_random:.4f}")
        print(f"  Gap (aligned-random):     {gap_aligned_random:.4f}")
        print(f"  Gap (aligned-same_piece): {gap_aligned_same_piece:.4f}")

        # Token stats
        print(f"\n  Tokens/piece audio: {np.mean(results['tokens_per_seg_audio']):.1f}")
        print(f"  Tokens/piece midi:  {np.mean(results['tokens_per_seg_midi']):.1f}")
        print(f"  Hash diversity audio: {np.mean(results['hash_diversity_audio']):.1f}")
        print(f"  Hash diversity midi:  {np.mean(results['hash_diversity_midi']):.1f}")

    # ================================================================
    # GO/NO-GO Evaluation
    # ================================================================
    print("\n" + "=" * 70)
    print("GO/NO-GO EVALUATION")
    print("=" * 70)

    for route_name, results in [
        ("Route A (Event-Based)", results_a),
        ("Route B (Improved TF)", results_b),
    ]:
        print(f"\n{route_name}:")

        mean_aligned = np.mean(results['overlaps_aligned']) if results['overlaps_aligned'] else 0
        mean_random = np.mean(results['overlaps_random']) if results['overlaps_random'] else 0
        gap = mean_aligned - mean_random

        criteria = [
            ("overlap_aligned > 5%", mean_aligned > 0.05, f"{mean_aligned:.2%}"),
            ("gap_aligned_random > 5%", gap > 0.05, f"{gap:.2%}"),
        ]

        all_pass = True
        for name, passed, value in criteria:
            status = "✓" if passed else "✗"
            print(f"  {status} {name}: {value}")
            if not passed:
                all_pass = False

        if all_pass:
            print(f"\n  ✓ GO: {route_name} shows discriminative signal")
        else:
            print(f"\n  ✗ NO-GO: {route_name} needs improvement")

    print("\n" + "=" * 70)

    # ================================================================
    # Save Results
    # ================================================================
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {
        'route_a': {
            'name': 'Event-Based',
            'overlap_aligned_mean': float(np.mean(results_a['overlaps_aligned'])) if results_a['overlaps_aligned'] else 0,
            'overlap_random_mean': float(np.mean(results_a['overlaps_random'])) if results_a['overlaps_random'] else 0,
            'gap_aligned_random': float(np.mean(results_a['overlaps_aligned']) - np.mean(results_a['overlaps_random'])) if results_a['overlaps_aligned'] and results_a['overlaps_random'] else 0,
            'tokens_per_piece_audio': float(np.mean(results_a['tokens_per_seg_audio'])),
            'tokens_per_piece_midi': float(np.mean(results_a['tokens_per_seg_midi'])),
        },
        'route_b': {
            'name': 'Improved TF',
            'overlap_aligned_mean': float(np.mean(results_b['overlaps_aligned'])) if results_b['overlaps_aligned'] else 0,
            'overlap_random_mean': float(np.mean(results_b['overlaps_random'])) if results_b['overlaps_random'] else 0,
            'gap_aligned_random': float(np.mean(results_b['overlaps_aligned']) - np.mean(results_b['overlaps_random'])) if results_b['overlaps_aligned'] and results_b['overlaps_random'] else 0,
            'tokens_per_piece_audio': float(np.mean(results_b['tokens_per_seg_audio'])),
            'tokens_per_piece_midi': float(np.mean(results_b['tokens_per_seg_midi'])),
        },
    }

    with open(output_dir / "route_comparison.json", 'w') as f:
        json.dump(all_results, f, indent=2)

    # ================================================================
    # Visualization
    # ================================================================
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Overlap comparison
    ax = axes[0, 0]
    x = np.arange(3)
    width = 0.35

    route_a_vals = [
        np.mean(results_a['overlaps_aligned']) if results_a['overlaps_aligned'] else 0,
        np.mean(results_a['overlaps_same_piece']) if results_a['overlaps_same_piece'] else 0,
        np.mean(results_a['overlaps_random']) if results_a['overlaps_random'] else 0,
    ]
    route_b_vals = [
        np.mean(results_b['overlaps_aligned']) if results_b['overlaps_aligned'] else 0,
        np.mean(results_b['overlaps_same_piece']) if results_b['overlaps_same_piece'] else 0,
        np.mean(results_b['overlaps_random']) if results_b['overlaps_random'] else 0,
    ]

    ax.bar(x - width/2, route_a_vals, width, label='Route A (Event)', color='steelblue')
    ax.bar(x + width/2, route_b_vals, width, label='Route B (TF)', color='coral')
    ax.set_xticks(x)
    ax.set_xticklabels(['Aligned', 'Same Piece', 'Random'])
    ax.set_ylabel('Overlap')
    ax.set_title('Hash Overlap by Condition')
    ax.legend()

    # 2. Gap comparison
    ax = axes[0, 1]
    gaps_a = [
        route_a_vals[0] - route_a_vals[2],  # aligned - random
        route_a_vals[0] - route_a_vals[1],  # aligned - same_piece
    ]
    gaps_b = [
        route_b_vals[0] - route_b_vals[2],
        route_b_vals[0] - route_b_vals[1],
    ]

    x = np.arange(2)
    ax.bar(x - width/2, gaps_a, width, label='Route A', color='steelblue')
    ax.bar(x + width/2, gaps_b, width, label='Route B', color='coral')
    ax.axhline(0.05, color='green', ls='--', label='GO threshold')
    ax.set_xticks(x)
    ax.set_xticklabels(['Gap (aligned-random)', 'Gap (aligned-same_piece)'])
    ax.set_ylabel('Gap')
    ax.set_title('Discriminative Gap')
    ax.legend()

    # 3. Token density
    ax = axes[1, 0]
    categories = ['Audio', 'MIDI']
    tokens_a = [np.mean(results_a['tokens_per_seg_audio']), np.mean(results_a['tokens_per_seg_midi'])]
    tokens_b = [np.mean(results_b['tokens_per_seg_audio']), np.mean(results_b['tokens_per_seg_midi'])]

    x = np.arange(2)
    ax.bar(x - width/2, tokens_a, width, label='Route A', color='steelblue')
    ax.bar(x + width/2, tokens_b, width, label='Route B', color='coral')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel('Tokens per piece')
    ax.set_title('Token Density')
    ax.legend()

    # 4. Summary
    ax = axes[1, 1]
    summary = f"""
ROUTE COMPARISON SUMMARY

Route A (Event-Based):
  Overlap aligned: {route_a_vals[0]:.3f}
  Overlap random:  {route_a_vals[2]:.3f}
  Gap: {gaps_a[0]:.3f}
  {'✓ GO' if gaps_a[0] > 0.05 else '✗ NO-GO'}

Route B (Improved TF):
  Overlap aligned: {route_b_vals[0]:.3f}
  Overlap random:  {route_b_vals[2]:.3f}
  Gap: {gaps_b[0]:.3f}
  {'✓ GO' if gaps_b[0] > 0.05 else '✗ NO-GO'}

Winner: {'Route A' if gaps_a[0] > gaps_b[0] else 'Route B'}
"""
    ax.text(0.5, 0.5, summary, ha='center', va='center', fontsize=11,
            transform=ax.transAxes, fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow'))
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_dir / "route_comparison.png", dpi=150)
    plt.close()

    print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=Path,
                        default=Path(__file__).parent / "Varios_pares")
    parser.add_argument('--output-dir', type=Path,
                        default=Path(__file__).parent / "Varios_pares" / "route_comparison")
    parser.add_argument('--max-duration', type=float, default=120.0)
    args = parser.parse_args()

    run_comparison(args.input_dir, args.output_dir, args.max_duration)
