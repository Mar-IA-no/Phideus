#!/usr/bin/env python3
"""H4.2-6 Pre-Red Test: Audio ratio histogram feasibility.

DEC-005 Track 2 — Script 5.

Tests whether harmonic ratio histograms can be extracted from audio (via CQT)
and meaningfully compared with MIDI ratio histograms, BEFORE any training.

Phase P0 (Oracle): synthesized audio from MIDI -> should match perfectly.
Phase P1 (Real):   real audio recordings     -> tests real-world viability.

Thresholds (DEC-005):
  P0: GO if AUC >= 0.80 and delta_sim >= 0.10
  P1: GO if AUC >= 0.70 and delta_sim >= 0.05 and Wilcoxon p < 0.01

No GPU required. Uses CQT peak detection + FluidSynth for synthesis.

Output:
    data/bias_control_medium/evaluations/gate42/h426_prered_results.json
    data/bias_control_medium/evaluations/gate42/fig_histogram_overlay.png
    data/bias_control_medium/evaluations/gate42/fig_roc_p0_p1.png
    data/bias_control_medium/evaluations/gate42/fig_similarity_scatter.png
"""

import json
import logging
import sys
import time
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pretty_midi
from scipy.spatial.distance import cosine as cosine_dist
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.bias_control.datasets.maestro_segments import MaestroSegmentDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAESTRO_DIR = ROOT / "data/maestro_v3/maestro-v3.0.0"
SF2_PATH = "/usr/share/sounds/sf2/TimGM6mb.sf2"
OUTPUT_DIR = ROOT / "data/bias_control_medium/evaluations/gate42"

N_SEGMENTS = 100
SR = 16000
N_BINS = 256
N_PEAKS = 12
RATIO_MIN = 0.5
RATIO_MAX = 2.0
SEED = 42


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def suppress_harmonics(freqs: np.ndarray, tolerance: float = 0.05) -> np.ndarray:
    """Remove frequencies that are harmonics (2x, 3x) of lower frequencies."""
    freqs = np.sort(freqs)
    keep = []
    for f in freqs:
        is_harmonic = False
        for fundamental in keep:
            for harmonic in [2, 3]:
                if abs(f - fundamental * harmonic) / (fundamental * harmonic) < tolerance:
                    is_harmonic = True
                    break
            if is_harmonic:
                break
        if not is_harmonic:
            keep.append(f)
    return np.array(keep) if keep else np.array([])


def soft_bin_ratios(
    ratios: list,
    n_bins: int = N_BINS,
    ratio_min: float = RATIO_MIN,
    ratio_max: float = RATIO_MAX,
) -> np.ndarray:
    """Gaussian soft binning of ratio values into a histogram."""
    histogram = np.zeros(n_bins)
    if len(ratios) == 0:
        return histogram
    bin_edges = np.linspace(ratio_min, ratio_max, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]
    sigma = bin_width * 0.5

    for r in ratios:
        weights = np.exp(-0.5 * ((bin_centers - r) / sigma) ** 2)
        histogram += weights

    total = histogram.sum()
    if total > 0:
        histogram /= total
    return histogram


def extract_audio_ratio_histogram(
    audio: np.ndarray,
    sr: int = SR,
    n_bins: int = N_BINS,
    n_peaks: int = N_PEAKS,
    ratio_min: float = RATIO_MIN,
    ratio_max: float = RATIO_MAX,
) -> np.ndarray:
    """Extract ratio histogram from audio via CQT peak detection."""
    # 1. CQT: 84 bins, 12 bins/octave, fmin=27.5Hz (A0)
    C = np.abs(librosa.cqt(audio, sr=sr, n_bins=84, bins_per_octave=12, fmin=27.5))

    # 2. Average across time for segment-level spectrum
    spectrum = C.mean(axis=1)  # [84]

    # 3. Peak detection: top-K peaks by amplitude
    peak_indices = np.argsort(spectrum)[-n_peaks:]

    # 4. Convert CQT bins to frequency
    freqs = librosa.cqt_frequencies(n_bins=84, fmin=27.5, bins_per_octave=12)
    peak_freqs = freqs[peak_indices]

    # 5. Harmonic suppression
    peak_freqs = suppress_harmonics(peak_freqs, tolerance=0.05)

    if len(peak_freqs) < 2:
        return np.zeros(n_bins)

    # 6. Pairwise ratios (upper triangle + reciprocals in range)
    n = len(peak_freqs)
    ratios = []
    for i in range(n):
        for j in range(i + 1, n):
            r = peak_freqs[j] / peak_freqs[i]
            if ratio_min <= r <= ratio_max:
                ratios.append(r)
            r_inv = peak_freqs[i] / peak_freqs[j]
            if ratio_min <= r_inv <= ratio_max:
                ratios.append(r_inv)

    # 7. Soft binning + normalize
    return soft_bin_ratios(ratios, n_bins, ratio_min, ratio_max)


def extract_midi_ratio_histogram(
    midi_path: str,
    start_time: float,
    end_time: float,
    n_bins: int = N_BINS,
    ratio_min: float = RATIO_MIN,
    ratio_max: float = RATIO_MAX,
) -> np.ndarray:
    """Extract ratio histogram from MIDI notes in time window."""
    midi = pretty_midi.PrettyMIDI(str(midi_path))

    # Get all note frequencies in window
    freqs = []
    for instrument in midi.instruments:
        if instrument.is_drum:
            continue
        for note in instrument.notes:
            if note.start < end_time and note.end > start_time:
                freqs.append(pretty_midi.note_number_to_hz(note.pitch))

    if len(freqs) < 2:
        return np.zeros(n_bins)

    # Deduplicate by rounding to 0.1 Hz
    unique_freqs = np.unique(np.round(freqs, 1))

    # Pairwise ratios
    n = len(unique_freqs)
    ratios = []
    for i in range(n):
        for j in range(i + 1, n):
            r = unique_freqs[j] / unique_freqs[i]
            if ratio_min <= r <= ratio_max:
                ratios.append(r)
            r_inv = unique_freqs[i] / unique_freqs[j]
            if ratio_min <= r_inv <= ratio_max:
                ratios.append(r_inv)

    return soft_bin_ratios(ratios, n_bins, ratio_min, ratio_max)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity (1 - cosine distance), with zero-vector handling."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return float(1.0 - cosine_dist(a, b))


def bootstrap_ci(
    aligned_sims: np.ndarray,
    random_sims: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = SEED,
) -> tuple:
    """Bootstrap 95% CI for AUC."""
    rng = np.random.RandomState(seed)
    aucs = []
    for _ in range(n_bootstrap):
        idx_a = rng.choice(len(aligned_sims), len(aligned_sims), replace=True)
        idx_r = rng.choice(len(random_sims), len(random_sims), replace=True)
        labels = np.concatenate([np.ones(len(idx_a)), np.zeros(len(idx_r))])
        scores = np.concatenate([aligned_sims[idx_a], random_sims[idx_r]])
        try:
            aucs.append(roc_auc_score(labels, scores))
        except ValueError:
            pass
    if len(aucs) == 0:
        return (0.5, 0.5)
    return tuple(np.percentile(aucs, [2.5, 97.5]).tolist())


def decide(phase: str, auc: float, delta_sim: float, ci_low: float,
           wilcoxon_p: float = None) -> str:
    """Apply GO/INCONCLUSO/NO-GO thresholds from DEC-005."""
    if phase == "P0":
        if auc >= 0.80 and delta_sim >= 0.10:
            return "GO"
        elif auc < 0.65:
            return "NO-GO"
        else:
            return "INCONCLUSO"
    else:  # P1
        if auc >= 0.70 and delta_sim >= 0.05 and wilcoxon_p is not None and wilcoxon_p < 0.01:
            return "GO"
        elif auc < 0.55:
            return "NO-GO"
        else:
            return "INCONCLUSO"


# ---------------------------------------------------------------------------
# Visualization helpers (dark theme)
# ---------------------------------------------------------------------------

DARK_BG = "#0a0a0a"
COLOR_AUDIO = "#00e5ff"
COLOR_MIDI = "#ff1493"


def setup_dark_axes(ax):
    ax.set_facecolor(DARK_BG)
    ax.tick_params(colors="white", which="both")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for spine in ax.spines.values():
        spine.set_color("#333333")


def plot_histogram_overlay(segments_data, output_path):
    """Fig 1: overlay of audio vs MIDI histograms for 6 example segments."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.patch.set_facecolor(DARK_BG)
    fig.suptitle("Audio vs MIDI Ratio Histograms (P0 Oracle)", color="white", fontsize=14)
    x = np.linspace(RATIO_MIN, RATIO_MAX, N_BINS)

    for idx, ax in enumerate(axes.flat):
        if idx >= len(segments_data):
            ax.set_visible(False)
            continue
        seg = segments_data[idx]
        setup_dark_axes(ax)
        ax.plot(x, seg["audio_hist"], color=COLOR_AUDIO, alpha=0.8, linewidth=1.2, label="Audio")
        ax.fill_between(x, seg["audio_hist"], alpha=0.15, color=COLOR_AUDIO)
        ax.plot(x, seg["midi_hist"], color=COLOR_MIDI, alpha=0.8, linewidth=1.2, label="MIDI")
        ax.fill_between(x, seg["midi_hist"], alpha=0.15, color=COLOR_MIDI)
        ax.set_title(f'Seg {seg["seg_idx"]}  sim={seg["sim"]:.3f}', fontsize=9)
        ax.set_xlabel("ratio", fontsize=8)
        ax.set_ylabel("density", fontsize=8)
        if idx == 0:
            ax.legend(fontsize=7, loc="upper right",
                      facecolor="#1a1a1a", edgecolor="#333333", labelcolor="white")

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    logger.info(f"Saved: {output_path}")


def plot_roc(p0_labels, p0_scores, p1_labels, p1_scores, p0_auc, p1_auc, output_path):
    """Fig 2: ROC curves for P0 and P1."""
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor(DARK_BG)
    setup_dark_axes(ax)

    fpr0, tpr0, _ = roc_curve(p0_labels, p0_scores)
    fpr1, tpr1, _ = roc_curve(p1_labels, p1_scores)

    ax.plot(fpr0, tpr0, color=COLOR_AUDIO, linewidth=2, label=f"P0 Oracle (AUC={p0_auc:.3f})")
    ax.plot(fpr1, tpr1, color=COLOR_MIDI, linewidth=2, label=f"P1 Real (AUC={p1_auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="#555555", linewidth=1, label="Random")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves: Audio vs MIDI Ratio Histograms")
    ax.legend(fontsize=10, facecolor="#1a1a1a", edgecolor="#333333", labelcolor="white")

    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    logger.info(f"Saved: {output_path}")


def plot_similarity_scatter(p0_aligned, p0_random, p1_aligned, p1_random,
                            p0_ci, p1_ci, output_path):
    """Fig 3: scatter of aligned vs random similarities with bootstrap CI band."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor(DARK_BG)
    fig.suptitle("Aligned vs Random Cosine Similarity", color="white", fontsize=13)

    for ax, aligned, rand, ci, phase in [
        (axes[0], p0_aligned, p0_random, p0_ci, "P0 Oracle"),
        (axes[1], p1_aligned, p1_random, p1_ci, "P1 Real"),
    ]:
        setup_dark_axes(ax)
        ax.scatter(range(len(aligned)), aligned, s=14, color=COLOR_AUDIO,
                   alpha=0.7, label="Aligned", zorder=3)
        ax.scatter(range(len(rand)), rand, s=14, color=COLOR_MIDI,
                   alpha=0.7, label="Random", zorder=3)

        mean_a = np.mean(aligned)
        mean_r = np.mean(rand)
        ax.axhline(mean_a, color=COLOR_AUDIO, linestyle="--", linewidth=1, alpha=0.8)
        ax.axhline(mean_r, color=COLOR_MIDI, linestyle="--", linewidth=1, alpha=0.8)

        # CI band for AUC (shown as text annotation)
        ax.text(0.02, 0.95, f"AUC CI: [{ci[0]:.3f}, {ci[1]:.3f}]",
                transform=ax.transAxes, color="white", fontsize=8,
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a1a1a", edgecolor="#333333"))

        delta = mean_a - mean_r
        ax.set_title(f"{phase}  delta_sim={delta:.4f}", fontsize=10)
        ax.set_xlabel("Segment index")
        ax.set_ylabel("Cosine similarity")
        ax.legend(fontsize=8, loc="lower right",
                  facecolor="#1a1a1a", edgecolor="#333333", labelcolor="white")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    logger.info(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load dataset segments (validation split, segment_len=4.0)
    # ------------------------------------------------------------------
    logger.info("Loading validation dataset...")
    dataset = MaestroSegmentDataset(
        maestro_dir=str(MAESTRO_DIR),
        segment_len=4.0,
        hop=1.0,
        split="validation",
        load_audio=False,
        load_midi=False,
    )
    logger.info(f"Validation segments: {len(dataset.segments)}")

    # Select N_SEGMENTS via seed
    rng = np.random.RandomState(SEED)
    indices = rng.choice(len(dataset.segments), size=min(N_SEGMENTS, len(dataset.segments)),
                         replace=False)
    indices.sort()
    selected = [dataset.segments[i] for i in indices]
    logger.info(f"Selected {len(selected)} segments for evaluation")

    # ------------------------------------------------------------------
    # Phase P0 — Oracle (synthesized audio)
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("PHASE P0 — Oracle (MIDI -> FluidSynth -> CQT ratios)")
    logger.info("=" * 60)

    p0_aligned_sims = []
    p0_audio_hists = []
    p0_midi_hists = []
    p0_overlay_data = []
    p0_skipped = 0

    for i, seg in enumerate(selected):
        try:
            # Extract MIDI ratio histogram
            midi_hist = extract_midi_ratio_histogram(
                seg.midi_path, seg.start_time, seg.end_time
            )

            if midi_hist.sum() < 1e-12:
                p0_skipped += 1
                continue

            # Load full MIDI and synthesize audio
            midi_obj = pretty_midi.PrettyMIDI(str(seg.midi_path))
            synth_audio = midi_obj.fluidsynth(fs=SR, sf2_path=SF2_PATH)

            # Trim to segment window
            start_sample = int(seg.start_time * SR)
            end_sample = int(seg.end_time * SR)
            if end_sample > len(synth_audio):
                # Pad if needed
                synth_audio = np.pad(synth_audio, (0, max(0, end_sample - len(synth_audio))))
            segment_audio = synth_audio[start_sample:end_sample].astype(np.float32)

            if len(segment_audio) < SR:  # Less than 1 second
                p0_skipped += 1
                continue

            # Extract audio ratio histogram
            audio_hist = extract_audio_ratio_histogram(segment_audio, sr=SR)

            if audio_hist.sum() < 1e-12:
                p0_skipped += 1
                continue

            sim = cosine_similarity(audio_hist, midi_hist)
            p0_aligned_sims.append(sim)
            p0_audio_hists.append(audio_hist)
            p0_midi_hists.append(midi_hist)

            # Save first 6 for overlay plot
            if len(p0_overlay_data) < 6:
                p0_overlay_data.append({
                    "seg_idx": int(indices[i]),
                    "audio_hist": audio_hist,
                    "midi_hist": midi_hist,
                    "sim": sim,
                })

        except Exception as e:
            logger.warning(f"P0 segment {i} failed: {e}")
            p0_skipped += 1

        if (i + 1) % 20 == 0:
            logger.info(f"  P0 progress: {i + 1}/{len(selected)}")

    logger.info(f"P0 completed: {len(p0_aligned_sims)} segments, {p0_skipped} skipped")

    # Random pairs for P0
    p0_random_sims = []
    n_valid = len(p0_audio_hists)
    rng2 = np.random.RandomState(SEED + 1)
    for i in range(n_valid):
        j = rng2.randint(0, n_valid)
        while j == i:
            j = rng2.randint(0, n_valid)
        sim = cosine_similarity(p0_audio_hists[i], p0_midi_hists[j])
        p0_random_sims.append(sim)

    p0_aligned_sims = np.array(p0_aligned_sims)
    p0_random_sims = np.array(p0_random_sims)

    # P0 metrics
    p0_labels = np.concatenate([np.ones(len(p0_aligned_sims)), np.zeros(len(p0_random_sims))])
    p0_scores = np.concatenate([p0_aligned_sims, p0_random_sims])
    p0_auc = roc_auc_score(p0_labels, p0_scores)
    p0_delta = float(np.mean(p0_aligned_sims) - np.mean(p0_random_sims))
    p0_ci = bootstrap_ci(p0_aligned_sims, p0_random_sims)
    p0_decision = decide("P0", p0_auc, p0_delta, p0_ci[0])

    logger.info(f"P0 Results:")
    logger.info(f"  AUC           = {p0_auc:.4f}  (CI: [{p0_ci[0]:.4f}, {p0_ci[1]:.4f}])")
    logger.info(f"  delta_sim     = {p0_delta:.4f}")
    logger.info(f"  mean_aligned  = {np.mean(p0_aligned_sims):.4f}")
    logger.info(f"  mean_random   = {np.mean(p0_random_sims):.4f}")
    logger.info(f"  Decision      = {p0_decision}")

    # ------------------------------------------------------------------
    # Phase P1 — Real audio
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("PHASE P1 — Real Audio")
    logger.info("=" * 60)

    p1_aligned_sims = []
    p1_audio_hists = []
    p1_midi_hists_for_p1 = []
    p1_skipped = 0

    for i, seg in enumerate(selected):
        try:
            # Load real audio
            audio, _ = librosa.load(
                str(seg.audio_path), sr=SR, mono=True,
                offset=seg.start_time, duration=4.0,
            )

            if len(audio) < SR:
                p1_skipped += 1
                continue

            # Extract audio ratio histogram from real audio
            audio_hist = extract_audio_ratio_histogram(audio, sr=SR)

            if audio_hist.sum() < 1e-12:
                p1_skipped += 1
                continue

            # Extract MIDI ratio histogram (same as P0)
            midi_hist = extract_midi_ratio_histogram(
                seg.midi_path, seg.start_time, seg.end_time
            )

            if midi_hist.sum() < 1e-12:
                p1_skipped += 1
                continue

            sim = cosine_similarity(audio_hist, midi_hist)
            p1_aligned_sims.append(sim)
            p1_audio_hists.append(audio_hist)
            p1_midi_hists_for_p1.append(midi_hist)

        except Exception as e:
            logger.warning(f"P1 segment {i} failed: {e}")
            p1_skipped += 1

        if (i + 1) % 20 == 0:
            logger.info(f"  P1 progress: {i + 1}/{len(selected)}")

    logger.info(f"P1 completed: {len(p1_aligned_sims)} segments, {p1_skipped} skipped")

    # Random pairs for P1
    p1_random_sims = []
    n_valid_p1 = len(p1_audio_hists)
    rng3 = np.random.RandomState(SEED + 2)
    for i in range(n_valid_p1):
        j = rng3.randint(0, n_valid_p1)
        while j == i:
            j = rng3.randint(0, n_valid_p1)
        sim = cosine_similarity(p1_audio_hists[i], p1_midi_hists_for_p1[j])
        p1_random_sims.append(sim)

    p1_aligned_sims = np.array(p1_aligned_sims)
    p1_random_sims = np.array(p1_random_sims)

    # P1 metrics
    p1_labels = np.concatenate([np.ones(len(p1_aligned_sims)), np.zeros(len(p1_random_sims))])
    p1_scores = np.concatenate([p1_aligned_sims, p1_random_sims])
    p1_auc = roc_auc_score(p1_labels, p1_scores)
    p1_delta = float(np.mean(p1_aligned_sims) - np.mean(p1_random_sims))
    p1_ci = bootstrap_ci(p1_aligned_sims, p1_random_sims)

    # Wilcoxon signed-rank test (paired: aligned vs random)
    n_paired = min(len(p1_aligned_sims), len(p1_random_sims))
    if n_paired >= 10:
        stat, wilcoxon_p = wilcoxon(p1_aligned_sims[:n_paired], p1_random_sims[:n_paired])
    else:
        wilcoxon_p = 1.0

    p1_decision = decide("P1", p1_auc, p1_delta, p1_ci[0], wilcoxon_p)

    # Degradation ratio
    degradation = p1_auc / p0_auc if p0_auc > 0 else 0.0

    logger.info(f"P1 Results:")
    logger.info(f"  AUC           = {p1_auc:.4f}  (CI: [{p1_ci[0]:.4f}, {p1_ci[1]:.4f}])")
    logger.info(f"  delta_sim     = {p1_delta:.4f}")
    logger.info(f"  mean_aligned  = {np.mean(p1_aligned_sims):.4f}")
    logger.info(f"  mean_random   = {np.mean(p1_random_sims):.4f}")
    logger.info(f"  Wilcoxon p    = {wilcoxon_p:.6f}")
    logger.info(f"  Degradation   = {degradation:.4f} (P1/P0)")
    logger.info(f"  Decision      = {p1_decision}")

    # ------------------------------------------------------------------
    # Save JSON results
    # ------------------------------------------------------------------
    results = {
        "script": "h426_prered_test.py",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {
            "n_segments": N_SEGMENTS,
            "sr": SR,
            "n_bins": N_BINS,
            "n_peaks": N_PEAKS,
            "ratio_min": RATIO_MIN,
            "ratio_max": RATIO_MAX,
            "seed": SEED,
            "sf2_path": SF2_PATH,
            "segment_len": 4.0,
        },
        "P0_oracle": {
            "n_valid": len(p0_aligned_sims),
            "n_skipped": p0_skipped,
            "auc": round(p0_auc, 4),
            "auc_ci_95": [round(p0_ci[0], 4), round(p0_ci[1], 4)],
            "delta_sim": round(p0_delta, 4),
            "mean_aligned_sim": round(float(np.mean(p0_aligned_sims)), 4),
            "mean_random_sim": round(float(np.mean(p0_random_sims)), 4),
            "std_aligned_sim": round(float(np.std(p0_aligned_sims)), 4),
            "std_random_sim": round(float(np.std(p0_random_sims)), 4),
            "decision": p0_decision,
        },
        "P1_real": {
            "n_valid": len(p1_aligned_sims),
            "n_skipped": p1_skipped,
            "auc": round(p1_auc, 4),
            "auc_ci_95": [round(p1_ci[0], 4), round(p1_ci[1], 4)],
            "delta_sim": round(p1_delta, 4),
            "mean_aligned_sim": round(float(np.mean(p1_aligned_sims)), 4),
            "mean_random_sim": round(float(np.mean(p1_random_sims)), 4),
            "std_aligned_sim": round(float(np.std(p1_aligned_sims)), 4),
            "std_random_sim": round(float(np.std(p1_random_sims)), 4),
            "wilcoxon_p": round(float(wilcoxon_p), 6),
            "degradation_p1_over_p0": round(degradation, 4),
            "decision": p1_decision,
        },
        "decision_matrix": {
            "P0": p0_decision,
            "P1": p1_decision,
        },
        "elapsed_seconds": round(time.time() - t0, 1),
    }

    json_path = OUTPUT_DIR / "h426_prered_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved: {json_path}")

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------
    logger.info("\nGenerating figures...")

    # Fig 1: histogram overlay (P0 oracle, 6 example segments)
    if len(p0_overlay_data) > 0:
        plot_histogram_overlay(
            p0_overlay_data,
            OUTPUT_DIR / "fig_histogram_overlay.png",
        )

    # Fig 2: ROC curves P0 and P1
    plot_roc(
        p0_labels, p0_scores,
        p1_labels, p1_scores,
        p0_auc, p1_auc,
        OUTPUT_DIR / "fig_roc_p0_p1.png",
    )

    # Fig 3: similarity scatter with CI
    plot_similarity_scatter(
        p0_aligned_sims, p0_random_sims,
        p1_aligned_sims, p1_random_sims,
        p0_ci, p1_ci,
        OUTPUT_DIR / "fig_similarity_scatter.png",
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    elapsed = time.time() - t0
    logger.info("\n" + "=" * 60)
    logger.info("H4.2-6 PRE-RED TEST SUMMARY")
    logger.info("=" * 60)
    logger.info(f"P0 (Oracle): AUC={p0_auc:.4f}, delta={p0_delta:.4f} -> {p0_decision}")
    logger.info(f"P1 (Real):   AUC={p1_auc:.4f}, delta={p1_delta:.4f} -> {p1_decision}")
    logger.info(f"Degradation: {degradation:.4f}")
    logger.info(f"Elapsed: {elapsed:.0f}s")

    # 2x2 decision matrix interpretation
    logger.info("\n--- 2x2 Decision Matrix (DEC-005) ---")
    logger.info("  Gate 6 drift asymmetric? | H4.2-6 P1 GO? | Next step")
    logger.info("  Yes                      | Yes            | H4.2-6 training + H4.2-2 adapter + S-control")
    logger.info("  Yes                      | No             | H4.2-2 adapter + H4.2-1 audio-only + S-control")
    logger.info("  No                       | Yes            | H4.2-6 training + S-control")
    logger.info("  No                       | No             | S-control only -> re-evaluate branch 4.x")
    logger.info(f"\n  H4.2-6 cell: P1 = {p1_decision}")
    logger.info("  (Gate 6 drift result comes from compare_layer_drift.py)")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
