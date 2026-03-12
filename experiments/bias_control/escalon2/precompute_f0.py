#!/usr/bin/env python3
"""
Precompute F0 cache for S2-P2-main descriptor arms.

Extracts F0 per-modality (no cross-modal leakage):
  - Speech: PYIN (librosa.pyin, frame_length=2048, fmin=50, fmax=500)
  - EGG: Autocorrelation (custom, quasi-periodic signal)

Output: data/lombard/f0_cache_noise0.npz (~20 MB)
  Keys: f0_speech_{clip_id}, voiced_speech_{clip_id},
        f0_egg_{clip_id}, voiced_egg_{clip_id}
  + metadata dict

Usage:
  python experiments/bias_control/escalon2/precompute_f0.py \
      --lombard-dir data/lombard/FLombard \
      --manifest data/lombard/manifest.json \
      --output data/lombard/f0_cache_noise0.npz \
      --workers 14

Estimated time: ~30-60 min on 14 cores (2280 clips × 2 modalities).
"""

import argparse
import json
import logging
import sys
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.bias_control.vocal_descriptors import (
    extract_f0_egg,
    F0_HOP_LENGTH, F0_SR, F0_FMIN, F0_FMAX, F0_FRAME_LENGTH,
    SEGMENT_LEN, FRAMES_PER_SEGMENT, F0_HOP_SEC,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)


def extract_f0_speech_pyin(
    audio: np.ndarray,
    sr: int = F0_SR,
    fmin: float = F0_FMIN,
    fmax: float = F0_FMAX,
    hop_length: int = F0_HOP_LENGTH,
    frame_length: int = F0_FRAME_LENGTH,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract F0 from speech using librosa PYIN.

    Returns:
        f0: [N_frames] F0 in Hz (NaN for unvoiced, converted to 0.0)
        voiced: [N_frames] boolean voicing flag
    """
    import librosa

    f0, voiced_flag, voiced_probs = librosa.pyin(
        audio,
        fmin=fmin,
        fmax=fmax,
        sr=sr,
        hop_length=hop_length,
        frame_length=frame_length,
        center=True,
    )

    # Convert NaN to 0.0
    voiced = ~np.isnan(f0)
    f0 = np.nan_to_num(f0, nan=0.0).astype(np.float32)
    voiced = voiced.astype(bool)

    return f0, voiced


def process_clip(args: Tuple) -> Dict:
    """Process a single clip: extract F0 for both speech and EGG.

    Args:
        args: (clip_dict, lombard_dir_str)

    Returns:
        dict with f0/voiced arrays for both modalities + clip_id
    """
    clip, lombard_dir_str = args
    lombard_dir = Path(lombard_dir_str)
    clip_id = clip['clip_id']

    try:
        # Load speech
        speech_path = lombard_dir / clip['speech_path']
        speech, sr = sf.read(str(speech_path), dtype='float32')
        assert sr == F0_SR, f"Speech SR mismatch: {sr} != {F0_SR}"

        # Load EGG
        egg_path = lombard_dir / clip['egg_path']
        egg, sr = sf.read(str(egg_path), dtype='float32')
        assert sr == F0_SR, f"EGG SR mismatch: {sr} != {F0_SR}"

        # Extract F0 per-modality
        f0_speech, voiced_speech = extract_f0_speech_pyin(speech)
        f0_egg, voiced_egg = extract_f0_egg(egg)

        return {
            'clip_id': clip_id,
            f'f0_speech_{clip_id}': f0_speech,
            f'voiced_speech_{clip_id}': voiced_speech,
            f'f0_egg_{clip_id}': f0_egg,
            f'voiced_egg_{clip_id}': voiced_egg,
            'n_frames_speech': len(f0_speech),
            'n_frames_egg': len(f0_egg),
            'voiced_frac_speech': float(voiced_speech.mean()),
            'voiced_frac_egg': float(voiced_egg.mean()),
            'error': None,
        }
    except Exception as e:
        return {
            'clip_id': clip_id,
            'error': str(e),
        }


def main():
    parser = argparse.ArgumentParser(description='Precompute F0 cache for S2-P2-main')
    parser.add_argument('--lombard-dir', type=str, required=True)
    parser.add_argument('--manifest', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--noise-condition', type=str, default='noise0')
    parser.add_argument('--workers', type=int, default=14)
    args = parser.parse_args()

    # Load manifest
    with open(args.manifest) as f:
        manifest = json.load(f)

    clips = [c for c in manifest['clips'] if c['noise_condition'] == args.noise_condition]
    logger.info(f"Processing {len(clips)} clips ({args.noise_condition})")

    # Process clips in parallel
    t0 = time.time()
    work_args = [(clip, args.lombard_dir) for clip in clips]

    cache_data = {}
    errors = []
    stats = {
        'voiced_frac_speech': [],
        'voiced_frac_egg': [],
        'f0_agreement': [],
    }

    if args.workers > 1:
        with Pool(args.workers) as pool:
            results = list(tqdm(
                pool.imap_unordered(process_clip, work_args),
                total=len(work_args),
                desc="Extracting F0",
            ))
    else:
        results = [process_clip(wa) for wa in tqdm(work_args, desc="Extracting F0")]

    for result in results:
        clip_id = result['clip_id']
        if result.get('error'):
            errors.append((clip_id, result['error']))
            continue

        # Store arrays
        for key in [f'f0_speech_{clip_id}', f'voiced_speech_{clip_id}',
                     f'f0_egg_{clip_id}', f'voiced_egg_{clip_id}']:
            cache_data[key] = result[key]

        stats['voiced_frac_speech'].append(result['voiced_frac_speech'])
        stats['voiced_frac_egg'].append(result['voiced_frac_egg'])

        # F0 agreement in jointly voiced frames
        n_min = min(len(result[f'f0_speech_{clip_id}']),
                     len(result[f'f0_egg_{clip_id}']))
        v_both = result[f'voiced_speech_{clip_id}'][:n_min] & result[f'voiced_egg_{clip_id}'][:n_min]
        if v_both.sum() > 10:
            f0_s = result[f'f0_speech_{clip_id}'][:n_min][v_both]
            f0_e = result[f'f0_egg_{clip_id}'][:n_min][v_both]
            agreement = np.mean(np.abs(f0_s - f0_e) < 10.0)  # within 10 Hz
            stats['f0_agreement'].append(float(agreement))

    elapsed = time.time() - t0
    logger.info(f"Processed {len(clips)} clips in {elapsed:.1f}s ({elapsed/len(clips):.2f}s/clip)")

    if errors:
        logger.warning(f"{len(errors)} errors:")
        for cid, err in errors[:5]:
            logger.warning(f"  {cid}: {err}")

    # Summary stats
    logger.info(f"Voiced fraction speech: {np.mean(stats['voiced_frac_speech']):.3f} "
                f"(std={np.std(stats['voiced_frac_speech']):.3f})")
    logger.info(f"Voiced fraction EGG:    {np.mean(stats['voiced_frac_egg']):.3f} "
                f"(std={np.std(stats['voiced_frac_egg']):.3f})")
    if stats['f0_agreement']:
        logger.info(f"F0 agreement (±10Hz):   {np.mean(stats['f0_agreement']):.3f} "
                    f"(std={np.std(stats['f0_agreement']):.3f})")

    # Save cache with metadata
    metadata = {
        'hop_length': F0_HOP_LENGTH,
        'sr': F0_SR,
        'center': True,
        'fmin': F0_FMIN,
        'fmax': F0_FMAX,
        'frame_length': F0_FRAME_LENGTH,
        'segment_len': SEGMENT_LEN,
        'frames_per_segment': FRAMES_PER_SEGMENT,
        'f0_hop_sec': F0_HOP_SEC,
        'noise_condition': args.noise_condition,
        'n_clips': len(clips) - len(errors),
        'n_errors': len(errors),
        'speech_method': 'pyin',
        'egg_method': 'autocorrelation',
        'voiced_frac_speech_mean': float(np.mean(stats['voiced_frac_speech'])),
        'voiced_frac_egg_mean': float(np.mean(stats['voiced_frac_egg'])),
    }

    # Store metadata as JSON string in a special key
    cache_data['__metadata__'] = np.array(json.dumps(metadata))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(output_path), **cache_data)
    size_mb = output_path.stat().st_size / 1024 / 1024
    logger.info(f"Saved cache: {output_path} ({size_mb:.1f} MB, {len(cache_data)-1} arrays)")


if __name__ == '__main__':
    main()
