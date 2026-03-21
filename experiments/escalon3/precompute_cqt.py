#!/usr/bin/env python3
"""Precompute CQT representations for Escalón 3 audio.

Reads audio.npy per split, computes CQT per channel via librosa,
saves as audio_cqt.npy (memmapable float16) + cqt_config.json.

Usage:
    python experiments/escalon3/precompute_cqt.py \
        --data data/escalon3/bundled \
        --workers 14
"""

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import librosa
import numpy as np
from tqdm import tqdm


_GLOBAL_AUDIO = None

# CQT config
CQT_CONFIG = {
    'source_sr': 44100,
    'target_sr': 22050,
    'hop_length': 256,
    'fmin': 55.0,
    'bins_per_octave': 36,
    'n_bins': 252,  # 7 octaves
    'representation': 'log1p_abs',
    'dtype_disk': 'float16',
}

SPLITS = ['train', 'val', 'test', 'ratio_ood', 'scale_ood', 'equiv_ood']


def compute_cqt_single(audio_stereo: np.ndarray, config: dict) -> np.ndarray:
    """Compute CQT for a single stereo scene.

    Args:
        audio_stereo: [2, 88200] float32 at 44100 Hz
    Returns:
        [2, n_bins, T'] float16
    """
    results = []
    for ch in range(2):
        # Resample 44100 → 22050
        wav = librosa.resample(audio_stereo[ch].astype(np.float32),
                               orig_sr=config['source_sr'],
                               target_sr=config['target_sr'])
        # CQT
        cqt = librosa.cqt(wav,
                           sr=config['target_sr'],
                           hop_length=config['hop_length'],
                           fmin=config['fmin'],
                           n_bins=config['n_bins'],
                           bins_per_octave=config['bins_per_octave'])
        # log1p(|CQT|)
        mag = np.log1p(np.abs(cqt)).astype(np.float16)
        results.append(mag)

    return np.stack(results, axis=0)  # [2, F, T']


def init_worker(audio_path: str):
    """Open split audio memmap once per worker process."""
    global _GLOBAL_AUDIO
    _GLOBAL_AUDIO = np.load(audio_path, mmap_mode='r')


def process_scene(args):
    """Worker for parallel CQT computation."""
    idx, config = args
    try:
        audio = np.array(_GLOBAL_AUDIO[idx])
        cqt = compute_cqt_single(audio, config)
        return idx, cqt
    except Exception as e:
        return idx, str(e)


def precompute_split(data_dir: Path, split: str, config: dict, workers: int = 14):
    """Precompute CQT for one split."""
    global _GLOBAL_AUDIO
    split_dir = data_dir / split
    audio_path = split_dir / 'audio.npy'

    if not audio_path.exists():
        print(f"  {split}: no audio.npy — skipping")
        return

    audio = np.load(str(audio_path), mmap_mode='r')
    _GLOBAL_AUDIO = audio
    N = audio.shape[0]
    print(f"  {split}: {N} scenes...")

    # Compute CQT for first scene to get output shape
    first_cqt = compute_cqt_single(np.array(audio[0]), config)
    F, T = first_cqt.shape[1], first_cqt.shape[2]
    print(f"  CQT shape per scene: [2, {F}, {T}]")

    # Pre-allocate output memmap on disk to keep RAM bounded.
    cqt_path = split_dir / 'audio_cqt.npy'
    out = np.lib.format.open_memmap(
        str(cqt_path), mode='w+', dtype=np.float16, shape=(N, 2, F, T))
    out[0] = first_cqt

    # Process remaining scenes in parallel
    if N > 1:
        work = ((i, config) for i in range(1, N))
        errors = []

        if workers > 1:
            with Pool(workers, initializer=init_worker, initargs=(str(audio_path),)) as pool:
                for idx, result in tqdm(pool.imap_unordered(process_scene, work),
                                        total=N - 1, desc=f"  {split}", leave=False):
                    if isinstance(result, str):
                        errors.append((idx, result))
                    else:
                        out[idx] = result
        else:
            for item in tqdm(work, desc=f"  {split}", leave=False):
                idx, result = process_scene(item)
                if isinstance(result, str):
                    errors.append((idx, result))
                else:
                    out[idx] = result

        if errors:
            try:
                del out
            finally:
                if cqt_path.exists():
                    cqt_path.unlink()
            first = errors[0]
            raise RuntimeError(
                f"CQT precompute failed for split '{split}' on {len(errors)} scenes. "
                f"First error at idx={first[0]}: {first[1]}"
            )

    # Flush memmap to disk before verification.
    out.flush()
    del out

    # Verify memmap
    test = np.load(str(cqt_path), mmap_mode='r')
    assert isinstance(test, np.memmap), "memmap failed!"
    assert test.shape == (N, 2, F, T), f"shape mismatch: {test.shape}"

    mb = test.nbytes / 1e6
    print(f"  {split}: saved {cqt_path.name} — {N} scenes, shape [N,2,{F},{T}], {mb:.0f} MB")

    # Save config
    cfg = {**config, 'n_scenes': N, 'freq_bins': F, 'time_frames': T}
    with open(split_dir / 'cqt_config.json', 'w') as f:
        json.dump(cfg, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Precompute CQT for Escalón 3")
    parser.add_argument("--data", default="data/escalon3/bundled")
    parser.add_argument("--workers", type=int, default=14)
    args = parser.parse_args()

    data_dir = Path(args.data)
    print(f"Precomputing CQT with config: {CQT_CONFIG}")

    for split in SPLITS:
        precompute_split(data_dir, split, CQT_CONFIG, args.workers)

    print("\nDone.")


if __name__ == "__main__":
    main()
