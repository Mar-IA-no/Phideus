#!/usr/bin/env python3
"""
PASO 0: Diagnóstico de colisión de hashes Audio↔MIDI.

Mide:
1. overlap_aligned: % de hashes compartidos entre Audio y MIDI del MISMO segmento
2. overlap_same_piece_diff_time: % compartidos entre mismo MIDI, diferente ventana temporal
3. overlap_random: % compartidos entre Audio y MIDI de DIFERENTES piezas

También analiza:
- Top hashes por document frequency (hashes "comunes")
- Distribución de overlap por segmento

GO/NO-GO:
- Si overlap_aligned ≈ 0 → problema de NO-COLISIÓN (hashes no coinciden)
- Si overlap_aligned ≈ overlap_random → problema de COLISIÓN GENÉRICA (hashes no discriminan)
- Si overlap_aligned >> overlap_random → hay señal, el problema está en otro lado
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

from test_single_pair_v2_parallel import (
    extract_audio_constellation_parallel,
    extract_midi_constellation_parallel,
    N_WORKERS,
)

# Same config as crossmodal test
DT_BIN_SIZE = 2
LOG_RATIO_BIN_SIZE = 1/24
N_ANCHOR_BANDS = 8


def token_to_hash(token: np.ndarray) -> int:
    """Convert token to hash (same as crossmodal test)."""
    log_ratio = token[0]
    delta_t = token[1]
    anchor_band = int(token[3])

    dt_bin = int(delta_t / DT_BIN_SIZE) % 64
    log_ratio_bin = int(log_ratio / LOG_RATIO_BIN_SIZE) % 64
    band = min(anchor_band * N_ANCHOR_BANDS // 16, N_ANCHOR_BANDS - 1)

    return (dt_bin << 8) | (log_ratio_bin << 3) | band


def extract_segment_hashes(
    tokens_per_frame: List[np.ndarray],
    start_frame: int,
    end_frame: int
) -> Set[int]:
    """Extract unique hashes from a segment."""
    hashes = set()
    for t_frame in range(start_frame, min(end_frame, len(tokens_per_frame))):
        if len(tokens_per_frame[t_frame]) == 0:
            continue
        for token in tokens_per_frame[t_frame]:
            h = token_to_hash(token)
            hashes.add(h)
    return hashes


def compute_overlap(set_a: Set[int], set_b: Set[int]) -> float:
    """Compute Jaccard-like overlap: |A ∩ B| / min(|A|, |B|)."""
    if len(set_a) == 0 or len(set_b) == 0:
        return 0.0
    intersection = len(set_a & set_b)
    return intersection / min(len(set_a), len(set_b))


def run_diagnosis(input_dir: Path, output_dir: Path, max_duration: float = 120.0):
    """Run hash collision diagnosis."""

    print("=" * 70)
    print("PASO 0: DIAGNÓSTICO DE COLISIÓN DE HASHES")
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

    # Process pieces
    pieces_data = []  # (piece_id, audio_tokens, midi_tokens, n_frames)

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

        # MIDI
        notes = parse_midi(midi_path)
        notes = [n for n in notes if n.onset < duration]

        midi_tokens, _, _ = extract_midi_constellation_parallel(
            notes, duration=duration, sr=sr, hop_length=hop_length, n_workers=N_WORKERS,
        )

        pieces_data.append((i, audio_tokens, midi_tokens, len(audio_tokens)))

        # Count hashes
        audio_hash_set = extract_segment_hashes(audio_tokens, 0, len(audio_tokens))
        midi_hash_set = extract_segment_hashes(midi_tokens, 0, len(midi_tokens))

        print(f"    Audio: {len(audio_hash_set):,} unique hashes, "
              f"MIDI: {len(midi_hash_set):,} unique hashes", flush=True)

    print(f"\n{'=' * 70}")
    print(f"Processed {len(pieces_data)} pieces in {time.time()-t0:.1f}s")
    print("=" * 70, flush=True)

    # Build global hash frequency
    print("\n=== Analyzing global hash frequencies ===", flush=True)
    global_audio_df = Counter()  # document frequency for audio
    global_midi_df = Counter()   # document frequency for midi

    for piece_id, audio_tokens, midi_tokens, n_frames in pieces_data:
        audio_hashes = extract_segment_hashes(audio_tokens, 0, n_frames)
        midi_hashes = extract_segment_hashes(midi_tokens, 0, n_frames)

        for h in audio_hashes:
            global_audio_df[h] += 1
        for h in midi_hashes:
            global_midi_df[h] += 1

    n_pieces = len(pieces_data)

    # Top hashes by DF (most common)
    print(f"\nTop 10 Audio hashes (by document frequency):")
    for h, df in global_audio_df.most_common(10):
        print(f"  Hash {h:5d}: appears in {df}/{n_pieces} pieces ({df/n_pieces:.0%})")

    print(f"\nTop 10 MIDI hashes (by document frequency):")
    for h, df in global_midi_df.most_common(10):
        print(f"  Hash {h:5d}: appears in {df}/{n_pieces} pieces ({df/n_pieces:.0%})")

    # Check overlap between top audio and midi hashes
    top_audio = set(h for h, _ in global_audio_df.most_common(100))
    top_midi = set(h for h, _ in global_midi_df.most_common(100))
    top_overlap = len(top_audio & top_midi)
    print(f"\nOverlap in top-100 hashes: {top_overlap}/100 ({top_overlap:.0%})")

    # Segment-level overlap analysis
    print("\n=== Segment-level overlap analysis ===", flush=True)

    segments = []  # (piece_id, seg_id, start, audio_hashes, midi_hashes)

    for piece_id, audio_tokens, midi_tokens, n_frames in pieces_data:
        start = 0
        seg_id = 0
        while start + segment_frames <= n_frames:
            audio_seg_hashes = extract_segment_hashes(audio_tokens, start, start + segment_frames)
            midi_seg_hashes = extract_segment_hashes(midi_tokens, start, start + segment_frames)
            segments.append((piece_id, seg_id, start, audio_seg_hashes, midi_seg_hashes))
            start += hop_frames
            seg_id += 1

    print(f"Total segments: {len(segments)}")

    # 1. overlap_aligned: same piece, same time
    overlaps_aligned = []
    for piece_id, seg_id, start, audio_hashes, midi_hashes in segments:
        overlap = compute_overlap(audio_hashes, midi_hashes)
        overlaps_aligned.append(overlap)

    # 2. overlap_same_piece_diff_time: same piece, different segment
    overlaps_same_piece = []
    for i, (pid1, sid1, _, ah1, mh1) in enumerate(segments):
        for j, (pid2, sid2, _, ah2, mh2) in enumerate(segments):
            if pid1 == pid2 and abs(sid1 - sid2) >= 2:  # Same piece, different time
                overlap = compute_overlap(ah1, mh2)
                overlaps_same_piece.append(overlap)
                if len(overlaps_same_piece) >= 500:  # Sample
                    break
        if len(overlaps_same_piece) >= 500:
            break

    # 3. overlap_random: different pieces
    overlaps_random = []
    np.random.seed(42)
    n_samples = min(500, len(segments) * (len(segments) - 1) // 2)
    for _ in range(n_samples):
        i, j = np.random.choice(len(segments), 2, replace=False)
        if segments[i][0] != segments[j][0]:  # Different pieces
            overlap = compute_overlap(segments[i][3], segments[j][4])  # audio_i vs midi_j
            overlaps_random.append(overlap)

    # Statistics
    print("\n" + "=" * 70)
    print("OVERLAP STATISTICS")
    print("=" * 70)
    print(f"\n1. ALIGNED (same piece, same time):")
    print(f"   Mean:   {np.mean(overlaps_aligned):.4f}")
    print(f"   Median: {np.median(overlaps_aligned):.4f}")
    print(f"   Std:    {np.std(overlaps_aligned):.4f}")
    print(f"   Max:    {np.max(overlaps_aligned):.4f}")

    print(f"\n2. SAME PIECE, DIFF TIME (within-piece negative):")
    print(f"   Mean:   {np.mean(overlaps_same_piece):.4f}")
    print(f"   Median: {np.median(overlaps_same_piece):.4f}")
    print(f"   Std:    {np.std(overlaps_same_piece):.4f}")

    print(f"\n3. RANDOM (different pieces):")
    print(f"   Mean:   {np.mean(overlaps_random):.4f}")
    print(f"   Median: {np.median(overlaps_random):.4f}")
    print(f"   Std:    {np.std(overlaps_random):.4f}")

    # Diagnosis
    print("\n" + "=" * 70)
    print("DIAGNOSIS")
    print("=" * 70)

    mean_aligned = np.mean(overlaps_aligned)
    mean_same_piece = np.mean(overlaps_same_piece)
    mean_random = np.mean(overlaps_random)

    gap_aligned_random = mean_aligned - mean_random
    gap_aligned_same_piece = mean_aligned - mean_same_piece

    print(f"\nGap (aligned - random):     {gap_aligned_random:.4f}")
    print(f"Gap (aligned - same_piece): {gap_aligned_same_piece:.4f}")

    if mean_aligned < 0.01:
        diagnosis = "NO-COLISIÓN: Los hashes Audio y MIDI casi NO coinciden (<1%)"
        diagnosis_code = "NO_COLLISION"
    elif gap_aligned_random < 0.02:
        diagnosis = "COLISIÓN GENÉRICA: Los hashes coinciden, pero igual para aligned y random"
        diagnosis_code = "GENERIC_COLLISION"
    elif gap_aligned_same_piece < 0.02:
        diagnosis = "COLISIÓN TEMPORAL: Coinciden dentro de la pieza pero no discriminan tiempo"
        diagnosis_code = "TEMPORAL_COLLISION"
    else:
        diagnosis = "HAY SEÑAL: overlap_aligned > overlap_negatives"
        diagnosis_code = "HAS_SIGNAL"

    print(f"\n>>> DIAGNÓSTICO: {diagnosis}")

    # GO/NO-GO
    print("\n" + "=" * 70)
    print("GO/NO-GO CRITERIA")
    print("=" * 70)

    criteria = [
        ("overlap_aligned > 5%", mean_aligned > 0.05, f"{mean_aligned:.2%}"),
        ("gap_aligned_random > 2%", gap_aligned_random > 0.02, f"{gap_aligned_random:.2%}"),
        ("gap_aligned_same_piece > 1%", gap_aligned_same_piece > 0.01, f"{gap_aligned_same_piece:.2%}"),
    ]

    all_pass = True
    for name, passed, value in criteria:
        status = "✓" if passed else "✗"
        print(f"  {status} {name}: {value}")
        if not passed:
            all_pass = False

    print()
    if all_pass:
        print("✓ GO: Hay señal en los hashes, el problema puede estar en el voting")
    else:
        print("✗ NO-GO: Los hashes no tienen suficiente overlap discriminativo")
        print("\n   Recomendación: Cambiar a Ruta A (Event-Based) o aplicar")
        print("   harmonic folding + onset anchoring (Ruta B mejorada)")

    print("=" * 70)

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        'n_pieces': n_pieces,
        'n_segments': len(segments),
        'overlap_aligned': {
            'mean': float(np.mean(overlaps_aligned)),
            'median': float(np.median(overlaps_aligned)),
            'std': float(np.std(overlaps_aligned)),
            'max': float(np.max(overlaps_aligned)),
        },
        'overlap_same_piece_diff_time': {
            'mean': float(np.mean(overlaps_same_piece)),
            'median': float(np.median(overlaps_same_piece)),
            'std': float(np.std(overlaps_same_piece)),
        },
        'overlap_random': {
            'mean': float(np.mean(overlaps_random)),
            'median': float(np.median(overlaps_random)),
            'std': float(np.std(overlaps_random)),
        },
        'gap_aligned_random': float(gap_aligned_random),
        'gap_aligned_same_piece': float(gap_aligned_same_piece),
        'diagnosis': diagnosis,
        'diagnosis_code': diagnosis_code,
        'top_100_hash_overlap': top_overlap,
    }

    with open(output_dir / "diagnosis_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Overlap distributions
    ax = axes[0, 0]
    bins = np.linspace(0, max(0.3, np.max(overlaps_aligned)), 30)
    ax.hist(overlaps_aligned, bins=bins, alpha=0.7, label='Aligned', color='green')
    ax.hist(overlaps_same_piece, bins=bins, alpha=0.7, label='Same piece, diff time', color='orange')
    ax.hist(overlaps_random, bins=bins, alpha=0.7, label='Random', color='red')
    ax.axvline(np.mean(overlaps_aligned), color='green', ls='--', lw=2)
    ax.axvline(np.mean(overlaps_random), color='red', ls='--', lw=2)
    ax.set_xlabel('Overlap (Jaccard-like)')
    ax.set_ylabel('Count')
    ax.set_title('Hash Overlap Distribution')
    ax.legend()

    # 2. Scatter: audio hash count vs midi hash count
    ax = axes[0, 1]
    audio_counts = [len(s[3]) for s in segments]
    midi_counts = [len(s[4]) for s in segments]
    ax.scatter(audio_counts, midi_counts, alpha=0.5, s=20)
    ax.set_xlabel('Audio unique hashes per segment')
    ax.set_ylabel('MIDI unique hashes per segment')
    ax.set_title('Hash Density Comparison')
    ax.plot([0, max(audio_counts)], [0, max(audio_counts)], 'k--', alpha=0.3)

    # 3. Per-segment overlap
    ax = axes[1, 0]
    ax.scatter(range(len(overlaps_aligned)), sorted(overlaps_aligned, reverse=True),
               alpha=0.7, s=10, label='Aligned overlap')
    ax.axhline(np.mean(overlaps_random), color='red', ls='--', label=f'Random mean: {np.mean(overlaps_random):.3f}')
    ax.set_xlabel('Segment (sorted by overlap)')
    ax.set_ylabel('Overlap')
    ax.set_title('Per-Segment Aligned Overlap')
    ax.legend()

    # 4. Summary box
    ax = axes[1, 1]
    summary_text = f"""
DIAGNOSIS: {diagnosis_code}

Overlap Statistics:
  Aligned:           {mean_aligned:.4f}
  Same piece/diff t: {mean_same_piece:.4f}
  Random:            {mean_random:.4f}

Gaps:
  Aligned - Random:     {gap_aligned_random:.4f}
  Aligned - Same piece: {gap_aligned_same_piece:.4f}

Top-100 Hash Overlap: {top_overlap}%

{'✓ GO' if all_pass else '✗ NO-GO'}
"""
    ax.text(0.5, 0.5, summary_text, ha='center', va='center', fontsize=12,
            transform=ax.transAxes, fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgreen' if all_pass else 'lightcoral'))
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_dir / "diagnosis_results.png", dpi=150)
    plt.close()

    print(f"\nResults saved to: {output_dir}")

    return diagnosis_code


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=Path,
                        default=Path(__file__).parent / "Varios_pares")
    parser.add_argument('--output-dir', type=Path,
                        default=Path(__file__).parent / "Varios_pares" / "diagnosis")
    parser.add_argument('--max-duration', type=float, default=120.0)
    args = parser.parse_args()

    run_diagnosis(args.input_dir, args.output_dir, args.max_duration)
