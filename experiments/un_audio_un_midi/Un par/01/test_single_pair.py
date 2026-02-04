#!/usr/bin/env python3
"""
Test Single Audio-MIDI Pair
============================

Extracts constellation tokens from a single audio-MIDI pair and compares them.
This is a sanity check to verify the extractors produce coherent results.

Usage:
------
python experiments/un_audio_un_midi/test_single_pair.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import librosa
except ImportError:
    print("ERROR: librosa required. Install with: pip install librosa")
    sys.exit(1)

from src.utils.midi_utils import parse_midi, midi_to_constellation_tokens
from src.analizador.analizador_maestro import extract_audio_constellation


def load_audio(audio_path: Path, sr: int = 22050) -> np.ndarray:
    """Load and preprocess audio."""
    audio, _ = librosa.load(audio_path, sr=sr, mono=True)
    # RMS normalize
    rms = np.sqrt(np.mean(audio ** 2))
    if rms > 0:
        audio = audio / rms * 0.1
    return audio


def visualize_tokens(
    audio_tokens: list,
    midi_tokens: list,
    audio_frame_times: np.ndarray,
    midi_frame_times: np.ndarray,
    output_path: Path,
):
    """Visualize and compare tokens from audio and MIDI."""

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # --- Row 1: Audio ---
    ax = axes[0, 0]
    # Scatter plot of tokens (log_ratio vs delta_t)
    all_log_ratios_a = []
    all_delta_t_a = []
    all_weights_a = []
    for frame_tokens in audio_tokens:
        if len(frame_tokens) > 0:
            all_log_ratios_a.extend(frame_tokens[:, 0])  # log_ratio
            all_delta_t_a.extend(frame_tokens[:, 1])     # delta_t
            all_weights_a.extend(frame_tokens[:, 2])     # weight

    if all_log_ratios_a:
        sc = ax.scatter(all_delta_t_a, all_log_ratios_a, c=all_weights_a,
                       alpha=0.3, s=1, cmap='viridis')
        ax.set_xlabel('delta_t')
        ax.set_ylabel('log_ratio')
        ax.set_title(f'AUDIO Tokens (n={len(all_log_ratios_a)})')
        plt.colorbar(sc, ax=ax, label='weight')

    # Histogram of log_ratios
    ax = axes[0, 1]
    if all_log_ratios_a:
        ax.hist(all_log_ratios_a, bins=50, alpha=0.7, color='blue', density=True)
        ax.set_xlabel('log_ratio')
        ax.set_ylabel('Density')
        ax.set_title('AUDIO: log_ratio distribution')
        ax.axvline(0, color='red', linestyle='--', label='ratio=1')
        ax.axvline(np.log2(1.5), color='green', linestyle='--', label='ratio=3:2 (fifth)')
        ax.axvline(1.0, color='orange', linestyle='--', label='ratio=2 (octave)')
        ax.legend(fontsize=8)

    # Tokens per frame over time
    ax = axes[0, 2]
    tokens_per_frame_a = [len(t) for t in audio_tokens]
    ax.plot(audio_frame_times[:len(tokens_per_frame_a)], tokens_per_frame_a, 'b-', alpha=0.7)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Tokens per frame')
    ax.set_title(f'AUDIO: Tokens over time (avg={np.mean(tokens_per_frame_a):.1f})')

    # --- Row 2: MIDI ---
    ax = axes[1, 0]
    all_log_ratios_m = []
    all_delta_t_m = []
    all_weights_m = []
    for frame_tokens in midi_tokens:
        if len(frame_tokens) > 0:
            all_log_ratios_m.extend(frame_tokens[:, 0])
            all_delta_t_m.extend(frame_tokens[:, 1])
            all_weights_m.extend(frame_tokens[:, 2])

    if all_log_ratios_m:
        sc = ax.scatter(all_delta_t_m, all_log_ratios_m, c=all_weights_m,
                       alpha=0.3, s=1, cmap='viridis')
        ax.set_xlabel('delta_t')
        ax.set_ylabel('log_ratio')
        ax.set_title(f'MIDI Tokens (n={len(all_log_ratios_m)})')
        plt.colorbar(sc, ax=ax, label='weight')

    # Histogram of log_ratios
    ax = axes[1, 1]
    if all_log_ratios_m:
        ax.hist(all_log_ratios_m, bins=50, alpha=0.7, color='red', density=True)
        ax.set_xlabel('log_ratio')
        ax.set_ylabel('Density')
        ax.set_title('MIDI: log_ratio distribution')
        ax.axvline(0, color='red', linestyle='--', label='ratio=1')
        ax.axvline(np.log2(1.5), color='green', linestyle='--', label='ratio=3:2 (fifth)')
        ax.axvline(1.0, color='orange', linestyle='--', label='ratio=2 (octave)')
        ax.legend(fontsize=8)

    # Tokens per frame over time
    ax = axes[1, 2]
    tokens_per_frame_m = [len(t) for t in midi_tokens]
    ax.plot(midi_frame_times[:len(tokens_per_frame_m)], tokens_per_frame_m, 'r-', alpha=0.7)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Tokens per frame')
    ax.set_title(f'MIDI: Tokens over time (avg={np.mean(tokens_per_frame_m):.1f})')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Visualization saved to {output_path}")


def compare_histograms(audio_tokens: list, midi_tokens: list) -> dict:
    """Compare histograms of audio and MIDI tokens."""

    # Extract log_ratios
    audio_log_ratios = []
    midi_log_ratios = []

    for frame_tokens in audio_tokens:
        if len(frame_tokens) > 0:
            audio_log_ratios.extend(frame_tokens[:, 0])

    for frame_tokens in midi_tokens:
        if len(frame_tokens) > 0:
            midi_log_ratios.extend(frame_tokens[:, 0])

    audio_log_ratios = np.array(audio_log_ratios)
    midi_log_ratios = np.array(midi_log_ratios)

    # Create histograms with same bins
    bins = np.linspace(-0.5, 2.5, 61)  # log2 ratios from ~0.7 to ~5.6

    hist_audio, _ = np.histogram(audio_log_ratios, bins=bins, density=True)
    hist_midi, _ = np.histogram(midi_log_ratios, bins=bins, density=True)

    # Normalize to sum to 1
    hist_audio = hist_audio / (hist_audio.sum() + 1e-8)
    hist_midi = hist_midi / (hist_midi.sum() + 1e-8)

    # Compute similarity metrics
    # Cosine similarity
    cosine = np.dot(hist_audio, hist_midi) / (
        np.linalg.norm(hist_audio) * np.linalg.norm(hist_midi) + 1e-8
    )

    # Histogram intersection
    intersection = np.minimum(hist_audio, hist_midi).sum()

    # KL divergence (symmetrized)
    eps = 1e-10
    kl_am = np.sum(hist_audio * np.log((hist_audio + eps) / (hist_midi + eps)))
    kl_ma = np.sum(hist_midi * np.log((hist_midi + eps) / (hist_audio + eps)))
    kl_sym = (kl_am + kl_ma) / 2

    return {
        'cosine_similarity': float(cosine),
        'histogram_intersection': float(intersection),
        'kl_divergence_sym': float(kl_sym),
        'n_audio_tokens': len(audio_log_ratios),
        'n_midi_tokens': len(midi_log_ratios),
        'audio_mean_log_ratio': float(audio_log_ratios.mean()) if len(audio_log_ratios) > 0 else 0,
        'midi_mean_log_ratio': float(midi_log_ratios.mean()) if len(midi_log_ratios) > 0 else 0,
    }


def main():
    # Paths
    base_dir = Path(__file__).parent
    audio_path = base_dir / "MIDI-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_05_Track05_wav.wav"
    midi_path = base_dir / "MIDI-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_05_Track05_wav.midi"

    print("=" * 70)
    print("TEST: Single Audio-MIDI Pair Constellation Extraction")
    print("=" * 70)

    # Check files exist
    if not audio_path.exists():
        print(f"ERROR: Audio file not found: {audio_path}")
        return
    if not midi_path.exists():
        print(f"ERROR: MIDI file not found: {midi_path}")
        return

    print(f"\nAudio: {audio_path.name}")
    print(f"MIDI:  {midi_path.name}")

    # Load audio
    print("\n[1] Loading audio...")
    sr = 22050
    audio = load_audio(audio_path, sr=sr)
    duration = len(audio) / sr
    print(f"    Duration: {duration:.1f}s")
    print(f"    Samples: {len(audio)}")

    # Extract audio constellations
    print("\n[2] Extracting AUDIO constellations...")
    audio_tokens, audio_frame_times = extract_audio_constellation(
        audio, sr=sr,
        max_anchors=16,
        max_targets_per_anchor=4,
    )
    n_audio_tokens = sum(len(t) for t in audio_tokens)
    print(f"    Frames: {len(audio_tokens)}")
    print(f"    Total tokens: {n_audio_tokens}")
    print(f"    Avg tokens/frame: {n_audio_tokens / len(audio_tokens):.1f}")

    # Load and parse MIDI
    print("\n[3] Extracting MIDI constellations...")
    notes = parse_midi(midi_path)
    print(f"    Notes in MIDI: {len(notes)}")

    # Extract MIDI constellations
    # Use same frame rate as audio
    hop_length = 512
    frame_rate = sr / hop_length
    midi_tokens, midi_frame_times = midi_to_constellation_tokens(
        notes,
        duration=duration,
        frame_rate=frame_rate,
        max_anchors=16,
        max_targets_per_anchor=4,
    )
    n_midi_tokens = sum(len(t) for t in midi_tokens)
    print(f"    Frames: {len(midi_tokens)}")
    print(f"    Total tokens: {n_midi_tokens}")
    print(f"    Avg tokens/frame: {n_midi_tokens / len(midi_tokens):.1f}")

    # Compare histograms
    print("\n[4] Comparing histograms...")
    metrics = compare_histograms(audio_tokens, midi_tokens)

    print(f"\n    COMPARISON METRICS:")
    print(f"    -------------------")
    print(f"    Cosine similarity:      {metrics['cosine_similarity']:.4f}")
    print(f"    Histogram intersection: {metrics['histogram_intersection']:.4f}")
    print(f"    KL divergence (sym):    {metrics['kl_divergence_sym']:.4f}")
    print(f"    Audio mean log_ratio:   {metrics['audio_mean_log_ratio']:.4f}")
    print(f"    MIDI mean log_ratio:    {metrics['midi_mean_log_ratio']:.4f}")

    # Interpretation
    print("\n[5] INTERPRETATION:")
    print("    -----------------")
    if metrics['cosine_similarity'] > 0.7:
        print("    ✓ HIGH similarity - Audio and MIDI have similar ratio distributions")
    elif metrics['cosine_similarity'] > 0.4:
        print("    ~ MODERATE similarity - Some overlap in ratio distributions")
    else:
        print("    ✗ LOW similarity - Audio and MIDI have different ratio distributions")

    # Key ratios check
    print("\n    Expected peaks for piano music:")
    print("    - log_ratio ≈ 0.0 (unison/octave doubling)")
    print("    - log_ratio ≈ 0.58 (perfect fifth, 3:2)")
    print("    - log_ratio ≈ 0.32 (perfect fourth, 4:3)")
    print("    - log_ratio ≈ 1.0 (octave, 2:1)")

    # Visualize
    print("\n[6] Creating visualization...")
    output_path = base_dir / "comparison_visualization.png"
    visualize_tokens(
        audio_tokens, midi_tokens,
        audio_frame_times, midi_frame_times,
        output_path
    )

    # Save metrics
    import json
    metrics_path = base_dir / "comparison_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
