#!/usr/bin/env python3
"""
Precompute WavLM-Large features for S2-P3.

Extracts mean-pooled [1024] embeddings from frozen WavLM-Large for all
noise0 speech segments. Features are saved to NPZ for fast training
(no encoder forward pass during training loop).

Output: data/lombard/wavlm_features_noise0.npz (~106 MB)
  Keys: wavlm_{clip_id}_{segment_idx} → [1024] float32
  + __metadata__ key with JSON string

Usage:
  python experiments/bias_control/escalon2/precompute_wavlm.py \
      --lombard-dir data/lombard/FLombard \
      --segment-index data/lombard/segment_index.json \
      --output data/lombard/wavlm_features_noise0.npz \
      --batch-size 32

Estimated time: 2-5 min on RTX 3090.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.bias_control.encoders.wavlm_encoder import WavLMEncoder

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
SEGMENT_SAMPLES = 32000  # 2s @ 16kHz


def load_segment_waveform(lombard_dir: Path, clip_id: str, segment_idx: int,
                          start_sec: float) -> np.ndarray:
    """Load a single speech segment waveform."""
    speaker_id = clip_id.split('_')[0]
    speech_path = lombard_dir / 'process' / 'wav' / speaker_id / f"{clip_id}.wav"

    start_sample = int(round(start_sec * SAMPLE_RATE))
    audio, sr = sf.read(str(speech_path), start=start_sample,
                        stop=start_sample + SEGMENT_SAMPLES, dtype='float32')
    assert sr == SAMPLE_RATE, f"Expected {SAMPLE_RATE}Hz, got {sr}Hz"

    if len(audio) < SEGMENT_SAMPLES:
        audio = np.pad(audio, (0, SEGMENT_SAMPLES - len(audio)))

    return audio


def main():
    parser = argparse.ArgumentParser(description="Precompute WavLM features for S2-P3")
    parser.add_argument('--lombard-dir', type=str, required=True,
                        help='Path to FLombard directory')
    parser.add_argument('--segment-index', type=str, required=True,
                        help='Path to segment_index.json')
    parser.add_argument('--output', type=str, required=True,
                        help='Output NPZ path')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--noise-condition', type=str, default='noise0')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    lombard_dir = Path(args.lombard_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load segment index
    with open(args.segment_index) as f:
        index = json.load(f)

    segments = [s for s in index['segments']
                if s['noise_condition'] == args.noise_condition]
    logger.info(f"Found {len(segments)} segments for {args.noise_condition}")

    # Load WavLM
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    encoder = WavLMEncoder(freeze=True)
    encoder.to(device)
    encoder._load_model()
    logger.info(f"WavLM loaded on {device}, output_dim={encoder.output_dim}")

    # Determinism check
    test_wav = torch.randn(1, SEGMENT_SAMPLES, device=device)
    assert encoder.verify_determinism(test_wav), "WavLM determinism check FAILED"
    logger.info("Determinism check PASSED")

    # Precompute in batches
    features = {}
    t0 = time.time()

    for batch_start in tqdm(range(0, len(segments), args.batch_size),
                            desc="Precomputing WavLM"):
        batch_segs = segments[batch_start:batch_start + args.batch_size]

        # Load waveforms
        waveforms = []
        keys = []
        for seg in batch_segs:
            wav = load_segment_waveform(
                lombard_dir, seg['clip_id'], seg['segment_idx'], seg['start_sec']
            )
            waveforms.append(wav)
            keys.append(f"wavlm_{seg['clip_id']}_{seg['segment_idx']}")

        # Stack and encode
        batch_tensor = torch.tensor(np.stack(waveforms), dtype=torch.float32, device=device)
        with torch.no_grad():
            embeddings = encoder(batch_tensor)  # [B, 1024]

        # Store
        embeddings_np = embeddings.cpu().numpy()
        for i, key in enumerate(keys):
            features[key] = embeddings_np[i].astype(np.float32)

    elapsed = time.time() - t0
    logger.info(f"Precomputed {len(features)} features in {elapsed:.1f}s "
                f"({len(features)/elapsed:.0f} segments/sec)")

    # Save with metadata
    metadata = {
        'model_name': encoder.model_name,
        'output_dim': encoder.output_dim,
        'aggregation': encoder.aggregation,
        'sample_rate': SAMPLE_RATE,
        'segment_samples': SEGMENT_SAMPLES,
        'noise_condition': args.noise_condition,
        'n_segments': len(features),
        'time_elapsed_sec': elapsed,
    }
    features['__metadata__'] = np.array(json.dumps(metadata))

    np.savez_compressed(str(output_path), **features)
    size_mb = output_path.stat().st_size / 1e6
    logger.info(f"Saved to {output_path} ({size_mb:.1f} MB, {len(features)-1} segments)")


if __name__ == '__main__':
    main()
