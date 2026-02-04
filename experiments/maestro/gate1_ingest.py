#!/usr/bin/env python3
"""
GATE 1: Ingestion and Alignment for MAESTRO Dataset
====================================================

Downloads, processes, and segments MAESTRO v3.0.0 dataset.
Creates aligned audio-MIDI pairs for cross-modal training.

Features:
- Download MAESTRO v3.0.0 (101GB compressed)
- Audio canonicalization: 22050Hz, mono, RMS normalized
- Segment generation: 4s windows with 2s hop
- Alignment verification: energy-density correlation

GO Criterion: Correlation (audio_energy, note_density) > 0.7

Usage:
------
# Download dataset
python experiments/maestro/gate1_ingest.py download \
    --output data/maestro_v3

# Process and segment
python experiments/maestro/gate1_ingest.py process \
    --input-dir data/maestro_v3/maestro-v3.0.0 \
    --output data/maestro_v3/processed \
    --window-len 4.0 --hop 2.0 --sr 22050 --workers 14

# Verify alignment
python experiments/maestro/gate1_ingest.py verify \
    --processed data/maestro_v3/processed
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import librosa
except ImportError:
    librosa = None
    print("Warning: librosa not installed. Install with: pip install librosa")

from src.utils.midi_utils import parse_midi, notes_to_piano_roll, get_note_density


# ═══════════════════════════════════════════════════════════════════════════════
# 1. CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

MAESTRO_URL = "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0.zip"
MAESTRO_MD5 = "7a6c23536ebcf3f50b1f00ac253886a7"  # For v3.0.0

DEFAULT_SR = 22050
DEFAULT_WINDOW_LEN = 4.0  # seconds
DEFAULT_HOP = 2.0  # seconds


@dataclass
class Segment:
    """Represents a processed segment."""
    audio_id: str           # Unique ID
    piece_id: str           # Original piece ID
    composer: str           # Composer name
    start_time: float       # Start time in original recording
    end_time: float         # End time
    audio_path: str         # Path to audio segment
    midi_path: str          # Path to original MIDI
    split: str              # train/validation/test


# ═══════════════════════════════════════════════════════════════════════════════
# 2. DOWNLOAD
# ═══════════════════════════════════════════════════════════════════════════════

def download_maestro(output_dir: Path, skip_download: bool = False) -> Path:
    """
    Download MAESTRO v3.0.0 dataset.

    Args:
        output_dir: Directory to save dataset
        skip_download: If True, assume already downloaded

    Returns:
        Path to extracted dataset
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    zip_path = output_dir / "maestro-v3.0.0.zip"
    extract_dir = output_dir / "maestro-v3.0.0"

    if extract_dir.exists() and (extract_dir / "maestro-v3.0.0.json").exists():
        print(f"Dataset already extracted at {extract_dir}")
        return extract_dir

    if not skip_download:
        if zip_path.exists():
            print(f"Zip file already exists at {zip_path}")
        else:
            print(f"Downloading MAESTRO v3.0.0 (~101GB)...")
            print(f"URL: {MAESTRO_URL}")
            print("This may take a while...")

            # Use wget for progress and resume
            cmd = [
                "wget", "-c",  # Continue partial downloads
                "-O", str(zip_path),
                MAESTRO_URL
            ]
            subprocess.run(cmd, check=True)

        # Verify MD5 (optional, takes time)
        print("Verifying download...")
        md5 = hashlib.md5()
        with open(zip_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192 * 1024), b""):
                md5.update(chunk)
        actual_md5 = md5.hexdigest()

        if actual_md5 != MAESTRO_MD5:
            print(f"Warning: MD5 mismatch!")
            print(f"  Expected: {MAESTRO_MD5}")
            print(f"  Got: {actual_md5}")
            # Don't fail, MAESTRO might have been updated

    # Extract
    if not extract_dir.exists():
        print(f"Extracting to {extract_dir}...")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(output_dir)
        print("Extraction complete!")

    return extract_dir


# ═══════════════════════════════════════════════════════════════════════════════
# 3. PROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

def load_maestro_metadata(maestro_dir: Path) -> List[Dict]:
    """Load MAESTRO metadata from JSON."""
    json_path = maestro_dir / "maestro-v3.0.0.json"

    if not json_path.exists():
        raise FileNotFoundError(f"Metadata not found: {json_path}")

    with open(json_path) as f:
        data = json.load(f)

    return data


def canonicalize_audio(
    audio_path: Path,
    sr: int = DEFAULT_SR,
) -> Tuple[np.ndarray, int]:
    """
    Load and canonicalize audio.

    - Resample to target SR
    - Convert stereo to mono
    - RMS normalize to target level
    """
    if librosa is None:
        raise ImportError("librosa required for audio processing")

    # Load with resampling
    y, sr_actual = librosa.load(str(audio_path), sr=sr, mono=True)

    # RMS normalize (target RMS = 0.1)
    rms = np.sqrt(np.mean(y ** 2))
    if rms > 1e-8:
        y = y * (0.1 / rms)

    # Clip to prevent distortion
    y = np.clip(y, -1.0, 1.0)

    return y.astype(np.float32), sr


def process_single_piece(args: Tuple) -> List[Dict]:
    """
    Process a single piece (worker function).

    Args:
        args: (piece_metadata, maestro_dir, output_dir, params)

    Returns:
        List of segment metadata dicts
    """
    piece, maestro_dir, output_dir, params = args

    sr = params.get('sr', DEFAULT_SR)
    window_len = params.get('window_len', DEFAULT_WINDOW_LEN)
    hop = params.get('hop', DEFAULT_HOP)

    maestro_dir = Path(maestro_dir)
    output_dir = Path(output_dir)

    audio_path = maestro_dir / piece['audio_filename']
    midi_path = maestro_dir / piece['midi_filename']

    if not audio_path.exists() or not midi_path.exists():
        return []

    try:
        # Load and canonicalize audio
        y, _ = canonicalize_audio(audio_path, sr)
        duration = len(y) / sr

        # Parse MIDI
        notes = parse_midi(midi_path)

        if len(notes) == 0:
            return []

        # Generate segments
        segments = []
        n_samples_window = int(window_len * sr)
        n_samples_hop = int(hop * sr)

        segment_idx = 0
        start_sample = 0

        while start_sample + n_samples_window <= len(y):
            start_time = start_sample / sr
            end_time = start_time + window_len

            # Extract audio segment
            audio_segment = y[start_sample:start_sample + n_samples_window]

            # Create unique ID
            piece_id = Path(piece['audio_filename']).stem
            segment_id = f"{piece_id}_seg{segment_idx:04d}"

            # Save segment
            segment_dir = output_dir / piece['split'] / piece_id
            segment_dir.mkdir(parents=True, exist_ok=True)

            audio_out = segment_dir / f"{segment_id}.npy"
            np.save(audio_out, audio_segment)

            segments.append({
                'segment_id': segment_id,
                'piece_id': piece_id,
                'composer': piece.get('canonical_composer', 'Unknown'),
                'title': piece.get('canonical_title', 'Unknown'),
                'start_time': start_time,
                'end_time': end_time,
                'duration': window_len,
                'audio_path': str(audio_out.relative_to(output_dir)),
                'midi_path': str(midi_path.relative_to(maestro_dir)),
                'split': piece['split'],
                'year': piece.get('year', 0),
            })

            start_sample += n_samples_hop
            segment_idx += 1

        return segments

    except Exception as e:
        print(f"Error processing {audio_path.name}: {e}")
        return []


def process_maestro(
    maestro_dir: Path,
    output_dir: Path,
    sr: int = DEFAULT_SR,
    window_len: float = DEFAULT_WINDOW_LEN,
    hop: float = DEFAULT_HOP,
    workers: int = 14,
    max_pieces: Optional[int] = None,
) -> Dict:
    """
    Process entire MAESTRO dataset.

    Args:
        maestro_dir: Path to extracted MAESTRO directory
        output_dir: Output directory for processed segments
        sr: Target sample rate
        window_len: Segment length in seconds
        hop: Hop between segments in seconds
        workers: Number of parallel workers
        max_pieces: Maximum pieces to process (for testing)

    Returns:
        Dataset metadata dict
    """
    maestro_dir = Path(maestro_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing MAESTRO dataset")
    print(f"  Source: {maestro_dir}")
    print(f"  Output: {output_dir}")
    print(f"  SR: {sr}, Window: {window_len}s, Hop: {hop}s")
    print(f"  Workers: {workers}")

    # Load metadata
    metadata = load_maestro_metadata(maestro_dir)
    pieces = metadata

    if max_pieces is not None:
        pieces = pieces[:max_pieces]
        print(f"  Limited to {max_pieces} pieces")

    print(f"  Total pieces: {len(pieces)}")

    # Prepare tasks
    params = {'sr': sr, 'window_len': window_len, 'hop': hop}
    tasks = [(p, maestro_dir, output_dir, params) for p in pieces]

    # Process in parallel
    all_segments = []

    if workers == 1:
        # Sequential for debugging
        for task in tqdm(tasks, desc="Processing"):
            segments = process_single_piece(task)
            all_segments.extend(segments)
    else:
        with mp.Pool(workers) as pool:
            results = list(tqdm(
                pool.imap(process_single_piece, tasks),
                total=len(tasks),
                desc="Processing"
            ))
            for segments in results:
                all_segments.extend(segments)

    # Split statistics
    splits = {'train': 0, 'validation': 0, 'test': 0}
    composers = set()
    for seg in all_segments:
        splits[seg['split']] = splits.get(seg['split'], 0) + 1
        composers.add(seg['composer'])

    # Save metadata
    dataset_meta = {
        'n_segments': len(all_segments),
        'splits': splits,
        'n_composers': len(composers),
        'sr': sr,
        'window_len': window_len,
        'hop': hop,
        'segments': all_segments,
    }

    meta_path = output_dir / 'dataset_metadata.json'
    with open(meta_path, 'w') as f:
        json.dump(dataset_meta, f, indent=2)

    print(f"\nProcessing complete!")
    print(f"  Segments: {len(all_segments)}")
    print(f"  Splits: {splits}")
    print(f"  Composers: {len(composers)}")
    print(f"  Metadata: {meta_path}")

    return dataset_meta


# ═══════════════════════════════════════════════════════════════════════════════
# 4. ALIGNMENT VERIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

def compute_audio_energy(
    audio: np.ndarray,
    sr: int,
    frame_rate: float = 50.0,
) -> np.ndarray:
    """Compute RMS energy per frame."""
    hop_length = int(sr / frame_rate)
    n_frames = len(audio) // hop_length

    energy = np.zeros(n_frames, dtype=np.float32)
    for i in range(n_frames):
        start = i * hop_length
        end = start + hop_length
        frame = audio[start:end]
        energy[i] = np.sqrt(np.mean(frame ** 2))

    return energy


def verify_alignment(
    processed_dir: Path,
    maestro_dir: Path,
    n_samples: int = 100,
) -> Dict:
    """
    Verify audio-MIDI alignment using energy-density correlation.

    GO Criterion: Correlation > 0.7

    Args:
        processed_dir: Path to processed segments
        maestro_dir: Path to original MAESTRO (for MIDI)
        n_samples: Number of segments to verify

    Returns:
        Verification results
    """
    processed_dir = Path(processed_dir)
    maestro_dir = Path(maestro_dir)

    # Load metadata
    meta_path = processed_dir / 'dataset_metadata.json'
    with open(meta_path) as f:
        metadata = json.load(f)

    segments = metadata['segments']
    sr = metadata['sr']

    # Sample segments
    if len(segments) > n_samples:
        indices = np.random.choice(len(segments), n_samples, replace=False)
        segments = [segments[i] for i in indices]

    print(f"Verifying alignment on {len(segments)} segments...")

    correlations = []

    for seg in tqdm(segments, desc="Verifying"):
        try:
            # Load audio
            audio_path = processed_dir / seg['audio_path']
            audio = np.load(audio_path)

            # Load MIDI and extract notes for this segment
            midi_path = maestro_dir / seg['midi_path']
            notes = parse_midi(midi_path)

            # Filter notes to segment time range
            start_time = seg['start_time']
            end_time = seg['end_time']
            seg_notes = [
                n for n in notes
                if n.offset > start_time and n.onset < end_time
            ]

            # Shift note times to segment-relative
            for n in seg_notes:
                n.onset = max(0, n.onset - start_time)
                n.offset = min(seg['duration'], n.offset - start_time)

            if len(seg_notes) < 2:
                continue

            # Compute energy and density
            frame_rate = 50.0
            energy = compute_audio_energy(audio, sr, frame_rate)
            density = get_note_density(seg_notes, seg['duration'], frame_rate)

            # Ensure same length
            min_len = min(len(energy), len(density))
            energy = energy[:min_len]
            density = density[:min_len]

            if min_len < 10:
                continue

            # Correlation
            if energy.std() > 0 and density.std() > 0:
                corr = np.corrcoef(energy, density)[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)

        except Exception as e:
            continue

    # Statistics
    correlations = np.array(correlations)
    mean_corr = float(correlations.mean())
    std_corr = float(correlations.std())
    min_corr = float(correlations.min())
    max_corr = float(correlations.max())

    results = {
        'n_verified': len(correlations),
        'mean_correlation': mean_corr,
        'std_correlation': std_corr,
        'min_correlation': min_corr,
        'max_correlation': max_corr,
        'go_threshold': 0.7,
        'pass': mean_corr > 0.7,
    }

    print(f"\nAlignment Verification Results:")
    print(f"  Segments verified: {len(correlations)}")
    print(f"  Mean correlation: {mean_corr:.4f} +/- {std_corr:.4f}")
    print(f"  Range: [{min_corr:.4f}, {max_corr:.4f}]")

    if mean_corr > 0.7:
        print(f"\n✓ GATE 1 PASS: Correlation > 0.7")
    else:
        print(f"\n✗ GATE 1 FAIL: Correlation < 0.7")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 5. CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description='GATE 1: MAESTRO Ingestion and Alignment'
    )
    subparsers = parser.add_subparsers(dest='command', help='Command')

    # Download
    dl = subparsers.add_parser('download', help='Download MAESTRO dataset')
    dl.add_argument('--output', type=Path, default=Path('data/maestro_v3'),
                    help='Output directory')
    dl.add_argument('--skip-download', action='store_true',
                    help='Skip download, assume already present')

    # Process
    proc = subparsers.add_parser('process', help='Process and segment dataset')
    proc.add_argument('--input-dir', type=Path, required=True,
                      help='Path to extracted MAESTRO directory')
    proc.add_argument('--output', type=Path, required=True,
                      help='Output directory for processed data')
    proc.add_argument('--sr', type=int, default=DEFAULT_SR,
                      help='Target sample rate')
    proc.add_argument('--window-len', type=float, default=DEFAULT_WINDOW_LEN,
                      help='Segment length in seconds')
    proc.add_argument('--hop', type=float, default=DEFAULT_HOP,
                      help='Hop between segments in seconds')
    proc.add_argument('--workers', type=int, default=14,
                      help='Number of parallel workers')
    proc.add_argument('--max-pieces', type=int, default=None,
                      help='Max pieces to process (for testing)')

    # Verify
    ver = subparsers.add_parser('verify', help='Verify alignment')
    ver.add_argument('--processed', type=Path, required=True,
                     help='Path to processed directory')
    ver.add_argument('--maestro-dir', type=Path, default=None,
                     help='Path to original MAESTRO (auto-detect if None)')
    ver.add_argument('--n-samples', type=int, default=100,
                     help='Number of samples to verify')
    ver.add_argument('--output', type=Path, default=None,
                     help='Output path for results JSON')

    return parser.parse_args()


def main():
    args = parse_args()

    if args.command == 'download':
        print("GATE 1: Download MAESTRO Dataset")
        print("=" * 60)
        download_maestro(args.output, args.skip_download)

    elif args.command == 'process':
        print("GATE 1: Process MAESTRO Dataset")
        print("=" * 60)
        process_maestro(
            maestro_dir=args.input_dir,
            output_dir=args.output,
            sr=args.sr,
            window_len=args.window_len,
            hop=args.hop,
            workers=args.workers,
            max_pieces=args.max_pieces,
        )

    elif args.command == 'verify':
        print("GATE 1: Verify Alignment")
        print("=" * 60)

        # Auto-detect MAESTRO dir
        if args.maestro_dir is None:
            # Try to find it relative to processed
            maestro_dir = args.processed.parent / 'maestro-v3.0.0'
            if not maestro_dir.exists():
                print(f"Error: Could not find MAESTRO dir. Specify with --maestro-dir")
                return
        else:
            maestro_dir = args.maestro_dir

        results = verify_alignment(
            processed_dir=args.processed,
            maestro_dir=maestro_dir,
            n_samples=args.n_samples,
        )

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {args.output}")

    else:
        print("Usage: python gate1_ingest.py {download|process|verify} ...")
        print("Run with -h for help")


if __name__ == '__main__':
    main()
