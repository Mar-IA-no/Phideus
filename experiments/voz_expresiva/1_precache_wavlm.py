#!/usr/bin/env python3
"""Pre-cache WavLM-large frozen embeddings sobre ESD English para Fase 1.

Salida:
    data/voz_expresiva/wavlm_cache/wavlm_features.npy  (memmap [N, T_max, 1024])
    data/voz_expresiva/wavlm_cache/wavlm_lengths.npy   ([N])
    data/voz_expresiva/wavlm_cache/wavlm_index.json    (metadata per utt)

Run:
    python experiments/voz_expresiva/1_precache_wavlm.py \\
        --esd-root data/esd/raw/'Emotion Speech Dataset' \\
        --output-dir data/voz_expresiva/wavlm_cache \\
        --batch-size 8
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.bias_control.encoders.wavlm_encoder import WavLMEncoder  # noqa: E402
from src.voz_expresiva.esd_loader import ESDLoader, ESDUtterance  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _load_audio_16k(wav_path: Path) -> np.ndarray:
    import librosa
    wav, _ = librosa.load(str(wav_path), sr=16000, mono=True)
    return wav.astype(np.float32)


def _detect_T_max(utterances: List[ESDUtterance], sample_n: int = 200) -> int:
    """Estimate T_max (50 Hz frames) by sampling audio durations."""
    import soundfile as sf
    durations = []
    step = max(1, len(utterances) // sample_n)
    for u in utterances[::step]:
        try:
            info = sf.info(str(u.wav_path))
            durations.append(info.frames / info.samplerate)
        except Exception:
            pass
    if not durations:
        return 350
    max_dur = max(durations)
    # 50 Hz frames + safety margin (10%)
    T_max = int(np.ceil(max_dur * 50 * 1.1))
    return max(T_max, 50)


@torch.no_grad()
def _extract_batch(encoder: WavLMEncoder, batch_wavs: List[np.ndarray],
                   device: torch.device) -> List[np.ndarray]:
    """Run WavLM on a batch, return list of [T, 1024] arrays per utt."""
    # Pad to max length in batch
    lens = [len(w) for w in batch_wavs]
    max_len = max(lens)
    padded = np.zeros((len(batch_wavs), max_len), dtype=np.float32)
    for i, w in enumerate(batch_wavs):
        padded[i, :len(w)] = w
    waveform = torch.from_numpy(padded).to(device)
    outputs = encoder(waveform, return_sequence=True)   # [B, T, 1024]
    outputs_np = outputs.cpu().numpy()
    # Slice each to actual length (T_50 = samples / 320 approx)
    # WavLM downsamples ~320x via its conv stack.
    out_per_utt = []
    for i, L in enumerate(lens):
        T_i = max(1, L // 320)
        # WavLM's actual T may differ slightly; cap to model output T
        T_i = min(T_i, outputs_np.shape[1])
        out_per_utt.append(outputs_np[i, :T_i, :].astype(np.float32))
    return out_per_utt


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--esd-root", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--language", default="EN")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--limit", type=int, default=None,
                   help="Debug: process only first N utterances")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    loader = ESDLoader(args.esd_root, language=args.language)
    utterances = loader.list_all()
    if args.limit:
        utterances = utterances[: args.limit]
    N = len(utterances)
    logger.info("ESD %s: %d utterances", args.language, N)

    # Detect T_max
    T_max = _detect_T_max(utterances)
    logger.info("Detected T_max (50 Hz frames, +10%% margin): %d", T_max)

    # Load WavLM
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Loading WavLM on %s", device)
    encoder = WavLMEncoder(freeze=True, device=str(device))
    encoder._load_model()

    # Allocate memmap
    feat_path = out_dir / "wavlm_features.npy"
    features_mm = np.memmap(
        feat_path, dtype=np.float32, mode="w+",
        shape=(N, T_max, 1024),
    )
    lengths = np.zeros(N, dtype=np.int32)
    utt_records = []

    t0 = time.time()
    batch_size = args.batch_size

    for batch_start in range(0, N, batch_size):
        batch_end = min(batch_start + batch_size, N)
        batch_utts = utterances[batch_start:batch_end]
        batch_wavs = [_load_audio_16k(u.wav_path) for u in batch_utts]

        outs = _extract_batch(encoder, batch_wavs, device)

        for i, (utt, out) in enumerate(zip(batch_utts, outs)):
            row = batch_start + i
            T = out.shape[0]
            if T > T_max:
                logger.warning("Utt %s exceeds T_max (%d > %d), truncating",
                               utt.wav_path.name, T, T_max)
                T = T_max
                out = out[:T_max]
            features_mm[row, :T, :] = out
            lengths[row] = T
            utt_records.append({
                "speaker_id": utt.speaker_id,
                "emotion": utt.emotion,
                "sentence_id": utt.sentence_id,
                "language": utt.language,
                "wav_path": str(utt.wav_path),
                "row_idx": row,
                "T_50Hz": int(T),
            })

        if (batch_end % (batch_size * 25)) == 0 or batch_end == N:
            elapsed = time.time() - t0
            rate = batch_end / elapsed
            eta = (N - batch_end) / rate if rate > 0 else 0
            logger.info("Progress %d/%d (%.1f utt/s, ETA %.1f min)",
                        batch_end, N, rate, eta / 60)

    features_mm.flush()
    np.save(out_dir / "wavlm_lengths.npy", lengths)

    index = {
        "T_max": T_max,
        "N": N,
        "sample_rate": 16000,
        "frame_rate_hz": 50,
        "feature_dim": 1024,
        "wavlm_model": "microsoft/wavlm-large",
        "utterances": utt_records,
    }
    (out_dir / "wavlm_index.json").write_text(json.dumps(index, indent=2))

    total = time.time() - t0
    feat_size_gb = (N * T_max * 1024 * 4) / 1e9
    logger.info("Done in %.1f min. Cache size: %.2f GB", total / 60, feat_size_gb)


if __name__ == "__main__":
    main()
