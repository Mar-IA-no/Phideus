#!/usr/bin/env python3
"""
GATE 2: Non-DL Baselines for MAESTRO Cross-Modal Matching
==========================================================

Implements baseline methods without deep learning:
- Baseline A: Chroma vs Pitch-Class histogram matching
- Baseline B: CCA/Ridge regression on mel + piano-roll features

GO Criterion: Piece-level Top-1 > 10x random

Usage:
------
python experiments/maestro/gate2_baselines.py \
    --data data/maestro_v3/processed \
    --maestro-dir data/maestro_v3/maestro-v3.0.0 \
    --output data/evaluations/maestro_baselines
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.cross_decomposition import CCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import librosa
except ImportError:
    librosa = None
    print("Warning: librosa not installed")

from src.utils.midi_utils import parse_midi, notes_to_pitch_histogram


# ═══════════════════════════════════════════════════════════════════════════════
# 1. FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

def extract_chroma_features(
    audio: np.ndarray,
    sr: int = 22050,
    n_chroma: int = 12,
    hop_length: int = 512,
) -> np.ndarray:
    """
    Extract chroma features from audio using CQT.

    Args:
        audio: Audio waveform
        sr: Sample rate
        n_chroma: Number of chroma bins (12 for pitch classes)
        hop_length: Hop length for analysis

    Returns:
        chroma: [T, 12] chroma features
    """
    if librosa is None:
        raise ImportError("librosa required")

    # CQT-based chroma (more robust than STFT-based)
    chroma = librosa.feature.chroma_cqt(
        y=audio,
        sr=sr,
        hop_length=hop_length,
        n_chroma=n_chroma,
    )

    return chroma.T  # [T, 12]


def extract_mel_features(
    audio: np.ndarray,
    sr: int = 22050,
    n_mels: int = 128,
    hop_length: int = 512,
) -> np.ndarray:
    """
    Extract log-mel spectrogram features.

    Args:
        audio: Audio waveform
        sr: Sample rate
        n_mels: Number of mel bands
        hop_length: Hop length

    Returns:
        mel: [T, n_mels] mel features
    """
    if librosa is None:
        raise ImportError("librosa required")

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=n_mels,
        hop_length=hop_length,
    )

    # Log compression
    mel_db = librosa.power_to_db(mel, ref=np.max)

    return mel_db.T  # [T, n_mels]


def extract_midi_pitch_histogram(
    midi_path: Path,
    start_time: float = 0.0,
    end_time: Optional[float] = None,
) -> np.ndarray:
    """
    Extract pitch-class histogram from MIDI.

    Returns:
        histogram: [12] pitch class distribution
    """
    notes = parse_midi(midi_path)
    return notes_to_pitch_histogram(notes, start_time, end_time)


def extract_midi_piano_roll_stats(
    midi_path: Path,
    start_time: float = 0.0,
    end_time: Optional[float] = None,
    frame_rate: float = 50.0,
) -> np.ndarray:
    """
    Extract statistical features from piano roll.

    Returns:
        features: [128] combining pitch histogram + temporal stats
    """
    from src.utils.midi_utils import parse_midi, notes_to_piano_roll, filter_notes_by_time

    notes = parse_midi(midi_path)

    if end_time is not None:
        notes = filter_notes_by_time(notes, start_time, end_time)
        duration = end_time - start_time
    else:
        duration = max((n.offset for n in notes), default=1.0)

    if len(notes) == 0:
        return np.zeros(128, dtype=np.float32)

    # Shift times for piano roll
    for n in notes:
        n.onset = max(0, n.onset - start_time)
        n.offset = max(n.onset + 0.01, n.offset - start_time)

    # Get piano roll [T, 88]
    pr = notes_to_piano_roll(notes, duration, frame_rate, piano_range=True)

    # Features:
    # - Pitch histogram (mean over time): [88]
    # - Pitch variance: [1]
    # - Note density stats: [3] (mean, std, max)
    # - Register usage: [8] (octave bins)
    # - Chord density: [3]

    pitch_mean = pr.mean(axis=0)  # [88]

    # Pad to 128
    features = np.zeros(128, dtype=np.float32)
    features[:88] = pitch_mean

    # Additional stats
    note_density = pr.sum(axis=1)  # [T]
    if len(note_density) > 0:
        features[88] = note_density.mean()
        features[89] = note_density.std()
        features[90] = note_density.max()

    # Octave distribution
    octave_bins = np.zeros(8)
    for i in range(88):
        octave = i // 12
        if octave < 8:
            octave_bins[octave] += pitch_mean[i]
    features[91:99] = octave_bins / (octave_bins.sum() + 1e-8)

    return features


# ═══════════════════════════════════════════════════════════════════════════════
# 2. BASELINE A: CHROMA vs PITCH-CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class ChromaBaseline:
    """
    Baseline A: Match audio chroma to MIDI pitch-class histogram.

    Simple but effective for piece-level matching.
    """

    def __init__(self):
        self.name = "Chroma-PitchClass"

    def compute_similarity(
        self,
        audio_chroma: np.ndarray,
        midi_pitch_hist: np.ndarray,
    ) -> float:
        """
        Compute similarity between audio chroma and MIDI pitch histogram.

        Args:
            audio_chroma: [T, 12] chroma features
            midi_pitch_hist: [12] pitch class histogram

        Returns:
            Cosine similarity
        """
        # Average chroma over time
        chroma_mean = audio_chroma.mean(axis=0)

        # Normalize
        chroma_norm = chroma_mean / (np.linalg.norm(chroma_mean) + 1e-8)
        midi_norm = midi_pitch_hist / (np.linalg.norm(midi_pitch_hist) + 1e-8)

        return float(np.dot(chroma_norm, midi_norm))

    def evaluate(
        self,
        audio_features: List[np.ndarray],
        midi_features: List[np.ndarray],
        k_values: List[int] = [1, 5, 10, 20],
    ) -> Dict[str, float]:
        """
        Evaluate retrieval performance.

        Args:
            audio_features: List of [T, 12] chroma arrays
            midi_features: List of [12] pitch histograms

        Returns:
            Metrics dict
        """
        N = len(audio_features)

        # Build similarity matrix
        sim = np.zeros((N, N), dtype=np.float32)
        for i in range(N):
            for j in range(N):
                sim[i, j] = self.compute_similarity(audio_features[i], midi_features[j])

        # Metrics
        labels = np.arange(N)
        results = {}

        for k in k_values:
            k_actual = min(k, N)
            topk = np.argsort(-sim, axis=1)[:, :k_actual]
            correct = np.any(topk == labels[:, None], axis=1)
            results[f'recall@{k}'] = float(correct.mean())

        # MRR
        sorted_idx = np.argsort(-sim, axis=1)
        ranks = np.where(sorted_idx == labels[:, None])[1] + 1
        results['mrr'] = float((1.0 / ranks).mean())
        results['mean_rank'] = float(ranks.mean())

        # Random baseline
        results['random_baseline'] = 1.0 / N
        results['ratio_vs_random'] = results['recall@1'] / results['random_baseline']

        return results


# ═══════════════════════════════════════════════════════════════════════════════
# 3. BASELINE B: CCA / RIDGE
# ═══════════════════════════════════════════════════════════════════════════════

class CCABaseline:
    """
    Baseline B: CCA projection of mel and piano-roll features.

    Projects both modalities to shared space using CCA.
    """

    def __init__(self, n_components: int = 32):
        self.name = "CCA"
        self.n_components = n_components
        self.cca = None
        self.scaler_audio = StandardScaler()
        self.scaler_midi = StandardScaler()

    def fit(
        self,
        audio_features: np.ndarray,
        midi_features: np.ndarray,
    ):
        """
        Fit CCA on training data.

        Args:
            audio_features: [N, D_audio] audio features
            midi_features: [N, D_midi] MIDI features
        """
        # Standardize
        audio_scaled = self.scaler_audio.fit_transform(audio_features)
        midi_scaled = self.scaler_midi.fit_transform(midi_features)

        # Fit CCA
        self.cca = CCA(n_components=self.n_components)
        self.cca.fit(audio_scaled, midi_scaled)

    def transform(
        self,
        audio_features: np.ndarray,
        midi_features: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform features to shared CCA space.

        Returns:
            z_audio, z_midi: [N, n_components] projected features
        """
        audio_scaled = self.scaler_audio.transform(audio_features)
        midi_scaled = self.scaler_midi.transform(midi_features)

        z_audio, z_midi = self.cca.transform(audio_scaled, midi_scaled)
        return z_audio, z_midi

    def evaluate(
        self,
        z_audio: np.ndarray,
        z_midi: np.ndarray,
        k_values: List[int] = [1, 5, 10, 20],
    ) -> Dict[str, float]:
        """Evaluate retrieval in CCA space."""
        N = len(z_audio)

        # Cosine similarity
        z_a = z_audio / (np.linalg.norm(z_audio, axis=1, keepdims=True) + 1e-8)
        z_m = z_midi / (np.linalg.norm(z_midi, axis=1, keepdims=True) + 1e-8)
        sim = z_a @ z_m.T

        labels = np.arange(N)
        results = {}

        for k in k_values:
            k_actual = min(k, N)
            topk = np.argsort(-sim, axis=1)[:, :k_actual]
            correct = np.any(topk == labels[:, None], axis=1)
            results[f'recall@{k}'] = float(correct.mean())

        sorted_idx = np.argsort(-sim, axis=1)
        ranks = np.where(sorted_idx == labels[:, None])[1] + 1
        results['mrr'] = float((1.0 / ranks).mean())
        results['mean_rank'] = float(ranks.mean())

        results['random_baseline'] = 1.0 / N
        results['ratio_vs_random'] = results['recall@1'] / results['random_baseline']

        return results


class RidgeBaseline:
    """
    Baseline B variant: Ridge regression to predict MIDI features from audio.
    """

    def __init__(self, alpha: float = 1.0):
        self.name = "Ridge"
        self.alpha = alpha
        self.ridge = None
        self.scaler_audio = StandardScaler()
        self.scaler_midi = StandardScaler()

    def fit(
        self,
        audio_features: np.ndarray,
        midi_features: np.ndarray,
    ):
        """Fit Ridge regression."""
        audio_scaled = self.scaler_audio.fit_transform(audio_features)
        midi_scaled = self.scaler_midi.fit_transform(midi_features)

        self.ridge = Ridge(alpha=self.alpha)
        self.ridge.fit(audio_scaled, midi_scaled)

    def predict(self, audio_features: np.ndarray) -> np.ndarray:
        """Predict MIDI features from audio."""
        audio_scaled = self.scaler_audio.transform(audio_features)
        return self.ridge.predict(audio_scaled)

    def evaluate(
        self,
        audio_features: np.ndarray,
        midi_features: np.ndarray,
        k_values: List[int] = [1, 5, 10, 20],
    ) -> Dict[str, float]:
        """Evaluate by predicting MIDI and matching."""
        N = len(audio_features)

        # Predict MIDI features
        midi_pred = self.predict(audio_features)
        midi_scaled = self.scaler_midi.transform(midi_features)

        # Cosine similarity between predicted and actual
        pred_norm = midi_pred / (np.linalg.norm(midi_pred, axis=1, keepdims=True) + 1e-8)
        actual_norm = midi_scaled / (np.linalg.norm(midi_scaled, axis=1, keepdims=True) + 1e-8)
        sim = pred_norm @ actual_norm.T

        labels = np.arange(N)
        results = {}

        for k in k_values:
            k_actual = min(k, N)
            topk = np.argsort(-sim, axis=1)[:, :k_actual]
            correct = np.any(topk == labels[:, None], axis=1)
            results[f'recall@{k}'] = float(correct.mean())

        sorted_idx = np.argsort(-sim, axis=1)
        ranks = np.where(sorted_idx == labels[:, None])[1] + 1
        results['mrr'] = float((1.0 / ranks).mean())
        results['mean_rank'] = float(ranks.mean())

        results['random_baseline'] = 1.0 / N
        results['ratio_vs_random'] = results['recall@1'] / results['random_baseline']

        return results


# ═══════════════════════════════════════════════════════════════════════════════
# 4. MAIN EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

def load_and_extract_features(
    processed_dir: Path,
    maestro_dir: Path,
    split: str = 'train',
    max_samples: Optional[int] = None,
    sr: int = 22050,
) -> Tuple[Dict[str, List], Dict[str, np.ndarray]]:
    """
    Load segments and extract features.

    Returns:
        features_list: Dict of feature lists (chroma, mel, etc.)
        features_array: Dict of feature arrays for CCA
    """
    processed_dir = Path(processed_dir)
    maestro_dir = Path(maestro_dir)

    # Load metadata
    with open(processed_dir / 'dataset_metadata.json') as f:
        metadata = json.load(f)

    segments = [s for s in metadata['segments'] if s['split'] == split]

    if max_samples is not None:
        segments = segments[:max_samples]

    print(f"Extracting features for {len(segments)} {split} segments...")

    # Feature storage
    chroma_list = []
    mel_stats_list = []
    pitch_hist_list = []
    piano_roll_list = []
    piece_ids = []

    for seg in tqdm(segments, desc="Extracting"):
        try:
            # Load audio
            audio_path = processed_dir / seg['audio_path']
            audio = np.load(audio_path)

            # Audio features
            chroma = extract_chroma_features(audio, sr)
            chroma_list.append(chroma)

            # Mel stats (mean, std over time)
            mel = extract_mel_features(audio, sr)
            mel_mean = mel.mean(axis=0)
            mel_std = mel.std(axis=0)
            mel_stats = np.concatenate([mel_mean, mel_std])  # [256]
            mel_stats_list.append(mel_stats)

            # MIDI features
            midi_path = maestro_dir / seg['midi_path']
            pitch_hist = extract_midi_pitch_histogram(
                midi_path, seg['start_time'], seg['end_time']
            )
            pitch_hist_list.append(pitch_hist)

            # Piano roll stats
            pr_stats = extract_midi_piano_roll_stats(
                midi_path, seg['start_time'], seg['end_time']
            )
            piano_roll_list.append(pr_stats)

            piece_ids.append(seg['piece_id'])

        except Exception as e:
            print(f"Error processing {seg['segment_id']}: {e}")
            continue

    return {
        'chroma': chroma_list,
        'mel_stats': np.array(mel_stats_list),
        'pitch_hist': np.array(pitch_hist_list),
        'piano_roll': np.array(piano_roll_list),
        'piece_ids': piece_ids,
    }


def run_baselines(
    processed_dir: Path,
    maestro_dir: Path,
    output_dir: Path,
    max_train: int = 1000,
    max_test: int = 500,
) -> Dict:
    """
    Run all baseline methods.

    Args:
        processed_dir: Path to processed MAESTRO segments
        maestro_dir: Path to original MAESTRO
        output_dir: Output directory for results
        max_train: Max training samples
        max_test: Max test samples

    Returns:
        Results dict
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("GATE 2: Non-DL Baselines")
    print("=" * 60)

    # Extract features
    print("\nExtracting training features...")
    train_features = load_and_extract_features(
        processed_dir, maestro_dir, split='train', max_samples=max_train
    )

    print("\nExtracting test features...")
    test_features = load_and_extract_features(
        processed_dir, maestro_dir, split='test', max_samples=max_test
    )

    results = {}

    # ─────────────────────────────────────────────────────────────────────────
    # Baseline A: Chroma vs Pitch-Class
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[Baseline A] Chroma vs Pitch-Class")
    print("-" * 40)

    chroma_baseline = ChromaBaseline()
    chroma_results = chroma_baseline.evaluate(
        test_features['chroma'],
        test_features['pitch_hist'],
    )
    results['chroma'] = chroma_results

    print(f"  Recall@1: {chroma_results['recall@1']:.4f}")
    print(f"  Recall@5: {chroma_results['recall@5']:.4f}")
    print(f"  MRR: {chroma_results['mrr']:.4f}")
    print(f"  vs Random: {chroma_results['ratio_vs_random']:.1f}x")

    # ─────────────────────────────────────────────────────────────────────────
    # Baseline B: CCA
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[Baseline B] CCA (Mel + Piano-roll)")
    print("-" * 40)

    cca_baseline = CCABaseline(n_components=32)

    # Fit on training
    cca_baseline.fit(
        train_features['mel_stats'],
        train_features['piano_roll'],
    )

    # Evaluate on test
    z_audio, z_midi = cca_baseline.transform(
        test_features['mel_stats'],
        test_features['piano_roll'],
    )
    cca_results = cca_baseline.evaluate(z_audio, z_midi)
    results['cca'] = cca_results

    print(f"  Recall@1: {cca_results['recall@1']:.4f}")
    print(f"  Recall@5: {cca_results['recall@5']:.4f}")
    print(f"  MRR: {cca_results['mrr']:.4f}")
    print(f"  vs Random: {cca_results['ratio_vs_random']:.1f}x")

    # ─────────────────────────────────────────────────────────────────────────
    # Baseline B variant: Ridge
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[Baseline B'] Ridge Regression")
    print("-" * 40)

    ridge_baseline = RidgeBaseline(alpha=1.0)
    ridge_baseline.fit(
        train_features['mel_stats'],
        train_features['piano_roll'],
    )
    ridge_results = ridge_baseline.evaluate(
        test_features['mel_stats'],
        test_features['piano_roll'],
    )
    results['ridge'] = ridge_results

    print(f"  Recall@1: {ridge_results['recall@1']:.4f}")
    print(f"  Recall@5: {ridge_results['recall@5']:.4f}")
    print(f"  MRR: {ridge_results['mrr']:.4f}")
    print(f"  vs Random: {ridge_results['ratio_vs_random']:.1f}x")

    # ─────────────────────────────────────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("GATE 2 SUMMARY")
    print("=" * 60)

    best_method = max(results, key=lambda k: results[k]['recall@1'])
    best_recall = results[best_method]['recall@1']
    best_ratio = results[best_method]['ratio_vs_random']

    print(f"\nBest method: {best_method}")
    print(f"  Recall@1: {best_recall:.4f}")
    print(f"  vs Random: {best_ratio:.1f}x")

    # GO/NO-GO
    go_threshold = 10.0  # 10x random
    results['go_criteria'] = {
        'threshold': go_threshold,
        'best_method': best_method,
        'best_ratio': best_ratio,
        'pass': best_ratio > go_threshold,
    }

    if best_ratio > go_threshold:
        print(f"\n✓ GATE 2 PASS: Best baseline > {go_threshold}x random")
    else:
        print(f"\n✗ GATE 2 FAIL: Best baseline < {go_threshold}x random")

    # Save results
    results_path = output_dir / 'gate2_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Generate report
    report_path = output_dir / 'GATE2_REPORT.md'
    with open(report_path, 'w') as f:
        f.write("# GATE 2: Non-DL Baselines Report\n\n")

        f.write("## Results Summary\n\n")
        f.write("| Method | Recall@1 | Recall@5 | MRR | vs Random |\n")
        f.write("|--------|----------|----------|-----|----------|\n")
        for method, m in results.items():
            if isinstance(m, dict) and 'recall@1' in m:
                f.write(f"| {method} | {m['recall@1']:.4f} | {m['recall@5']:.4f} | "
                        f"{m['mrr']:.4f} | {m['ratio_vs_random']:.1f}x |\n")

        f.write(f"\n## GO/NO-GO\n\n")
        f.write(f"- Threshold: > {go_threshold}x random\n")
        f.write(f"- Best method: {best_method} ({best_ratio:.1f}x)\n")
        f.write(f"- **Result**: {'**PASS**' if best_ratio > go_threshold else '**FAIL**'}\n")

    print(f"Report saved to {report_path}")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 5. CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description='GATE 2: Non-DL Baselines')
    parser.add_argument('--data', type=Path, required=True,
                        help='Path to processed MAESTRO segments')
    parser.add_argument('--maestro-dir', type=Path, required=True,
                        help='Path to original MAESTRO dataset')
    parser.add_argument('--output', type=Path, default=Path('data/evaluations/maestro_baselines'),
                        help='Output directory')
    parser.add_argument('--max-train', type=int, default=1000,
                        help='Max training samples')
    parser.add_argument('--max-test', type=int, default=500,
                        help='Max test samples')
    return parser.parse_args()


def main():
    args = parse_args()

    run_baselines(
        processed_dir=args.data,
        maestro_dir=args.maestro_dir,
        output_dir=args.output,
        max_train=args.max_train,
        max_test=args.max_test,
    )


if __name__ == '__main__':
    main()
