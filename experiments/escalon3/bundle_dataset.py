#!/usr/bin/env python3
"""E3-P0 Step 2: Bundle generated scenes into per-split .npy/.npz files.

Reads data/escalon3/scenes/ and produces data/escalon3/bundled/{split}/ with:
  audio.npy   — [N, 2, 88200] float32 (memmapable)
  image.npy   — [N, 128, 128] uint8 (memmapable)
  meta.npz    — small metadata arrays (loaded to RAM)

Usage:
    python experiments/escalon3/bundle_dataset.py \
        --scenes data/escalon3/scenes \
        --output data/escalon3/bundled
"""

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image
import scipy.io.wavfile as wavfile
from tqdm import tqdm


SPLITS = ['train', 'val', 'test', 'ratio_ood', 'scale_ood', 'equiv_ood']


def load_scene(scene_dir: Path) -> dict:
    """Load one scene's audio, image, and metadata."""
    with open(scene_dir / 'meta.json') as f:
        meta = json.load(f)

    # Audio: WAV int16 → float32 [-1, 1], transpose to [2, N]
    sr, audio_int16 = wavfile.read(str(scene_dir / 'xy_audio.wav'))
    audio = audio_int16.astype(np.float32) / 32767.0  # [88200, 2]
    audio = audio.T  # [2, 88200]

    # Image: figure_clean only
    img = np.array(Image.open(scene_dir / 'figure_clean.png'))  # [128, 128] uint8

    return {
        'audio': audio,
        'image': img,
        'meta': meta,
    }


def load_scene_worker(scene_dir_str: str) -> dict:
    """Worker wrapper for multiprocessing."""
    try:
        return load_scene(Path(scene_dir_str))
    except Exception as e:
        return {'error': str(e), 'scene_dir': scene_dir_str}


def bundle_split(scenes_dir: Path, output_dir: Path, split: str, workers: int = 14):
    """Bundle all scenes of a given split into .npy + .npz."""
    # Find all scene directories for this split
    scene_dirs = []
    for scene_path in sorted(scenes_dir.iterdir()):
        if not scene_path.is_dir():
            continue
        meta_path = scene_path / 'meta.json'
        if not meta_path.exists():
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        if meta['split'] == split:
            scene_dirs.append(str(scene_path))

    if not scene_dirs:
        print(f"  {split}: 0 scenes — skipping")
        return

    print(f"  {split}: loading {len(scene_dirs)} scenes...")

    # Load all scenes in parallel
    if workers > 1:
        with Pool(workers) as pool:
            results = list(tqdm(
                pool.imap(load_scene_worker, scene_dirs),
                total=len(scene_dirs), desc=f"  {split}", leave=False
            ))
    else:
        results = [load_scene_worker(d) for d in tqdm(scene_dirs, desc=f"  {split}")]

    # Check for errors
    errors = [r for r in results if 'error' in r]
    if errors:
        for e in errors[:5]:
            print(f"    ERROR: {e['scene_dir']}: {e['error']}")
        results = [r for r in results if 'error' not in r]

    N = len(results)
    print(f"  {split}: {N} scenes loaded OK")

    # Stack arrays
    audio_arr = np.stack([r['audio'] for r in results])   # [N, 2, 88200]
    image_arr = np.stack([r['image'] for r in results])   # [N, 128, 128]

    # Build metadata arrays
    meta_keys = [
        ('ratio_id', np.int64),
        ('equiv_id', np.int64),
        ('p', np.int64),
        ('q', np.int64),
        ('p_raw', np.int64),
        ('q_raw', np.int64),
        ('ratio_float', np.float32),
        ('phase_idx', np.int64),
        ('phase_rad', np.float32),
        ('amp_ratio', np.float32),
        ('base_frequency', np.float32),
        ('harmonic_order', np.int64),
        ('is_reduced', bool),
    ]

    meta_arrays = {}
    for key, dtype in meta_keys:
        meta_arrays[key] = np.array([r['meta'][key] for r in results], dtype=dtype)

    # scene_id as fixed unicode (no pickle)
    meta_arrays['scene_id'] = np.array(
        [r['meta']['scene_id'] for r in results], dtype='<U64'
    )
    # Rename base_frequency → base_freq for consistency with dataset class
    meta_arrays['base_freq'] = meta_arrays.pop('base_frequency')

    # Save
    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)

    np.save(str(split_dir / 'audio.npy'), audio_arr)
    np.save(str(split_dir / 'image.npy'), image_arr)
    np.savez(str(split_dir / 'meta.npz'), **meta_arrays)

    # Verify mmap works
    test_mmap = np.load(str(split_dir / 'audio.npy'), mmap_mode='r')
    assert isinstance(test_mmap, np.memmap), f"mmap failed for {split}/audio.npy"

    # Print stats
    audio_mb = audio_arr.nbytes / 1e6
    image_mb = image_arr.nbytes / 1e6
    meta_kb = sum(v.nbytes for v in meta_arrays.values()) / 1e3
    print(f"  {split}: saved {N} scenes — audio {audio_mb:.0f}MB, image {image_mb:.1f}MB, meta {meta_kb:.0f}KB")


def main():
    parser = argparse.ArgumentParser(description="E3-P0: Bundle scenes into .npy/.npz")
    parser.add_argument("--scenes", default="data/escalon3/scenes")
    parser.add_argument("--output", default="data/escalon3/bundled")
    parser.add_argument("--workers", type=int, default=14)
    args = parser.parse_args()

    scenes_dir = Path(args.scenes)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Bundling scenes from {scenes_dir} → {output_dir}")

    for split in SPLITS:
        bundle_split(scenes_dir, output_dir, split, workers=args.workers)

    # Final summary
    print("\nDone. Split summary:")
    for split in SPLITS:
        split_dir = output_dir / split
        if (split_dir / 'audio.npy').exists():
            audio = np.load(str(split_dir / 'audio.npy'), mmap_mode='r')
            print(f"  {split}: {audio.shape[0]} scenes, audio {audio.shape}, "
                  f"dtype={audio.dtype}")
        else:
            print(f"  {split}: not generated")


if __name__ == "__main__":
    main()
