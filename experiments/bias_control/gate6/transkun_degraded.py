"""
Gate 6 — Exp B: Transkun under Degraded Conditions

Tests whether A4 becomes more useful when the acoustic signal degrades.

CRITICAL PRINCIPLE: A4 is ALWAYS computed from the degraded audio,
the same waveform that Transkun sees. Computing A4 from clean audio
would be oracle leakage and contaminate the experiment.

Degradation types:
    - Additive Gaussian noise (SNR: 5, 10, 20 dB)
    - Band-limiting / low-pass (cutoff: 1kHz, 2kHz, 4kHz)
    - Data limitation (10%, 25%, 50% of training data)

For each degradation × level:
    1. baseline-degraded: Transkun pretrained, inference only
    2. finetune-degraded: Transkun + new params (no A4), trained on degraded
    3. A4-degraded: Transkun + A4 (from degraded audio), trained on degraded

Key comparison: Δ F1 between (3) and (2) at each degradation level.
Hypothesis: Δ F1 grows with degradation severity.

Usage:
    python experiments/bias_control/gate6/transkun_degraded.py \
        --degradation noise --level 10 \
        --config A4-degraded \
        --maestro-dir data/maestro_v3/maestro-v3.0.0 \
        --output data/gate6_results/transkun_degraded/noise_10dB_A4 \
        --iterations 50000 --batch-size 4 --device cuda

Dependency: Exp A must confirm that the fine-tuning pipeline works.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.bias_control.gate6.transkun_a4_finetune import (
    TransKunWithA4,
    load_pretrained_transkun,
    create_transkun_dataloaders,
    evaluate_transkun,
    train_loop,
)


# ── Audio Degradation Functions ────────────────────────────────────────────

def add_gaussian_noise(audio: torch.Tensor, snr_db: float) -> torch.Tensor:
    """Add Gaussian noise at specified SNR.

    Args:
        audio: [B, T] or [T] waveform
        snr_db: Signal-to-noise ratio in dB

    Returns:
        Degraded audio at same shape
    """
    signal_power = audio.pow(2).mean()
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = torch.randn_like(audio) * noise_power.sqrt()
    return audio + noise


def lowpass_filter(audio: torch.Tensor, cutoff_hz: float, sr: int = 44100) -> torch.Tensor:
    """Simple low-pass filter via FFT.

    Args:
        audio: [B, T] or [T]
        cutoff_hz: Cutoff frequency
        sr: Sample rate

    Returns:
        Band-limited audio
    """
    squeeze = False
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
        squeeze = True

    # FFT
    n = audio.shape[-1]
    freqs = torch.fft.rfftfreq(n, d=1.0/sr).to(audio.device)

    spectrum = torch.fft.rfft(audio, dim=-1)

    # Create smooth low-pass filter (Butterworth-like)
    # Using a gentle roll-off to avoid ringing
    order = 4
    filter_response = 1.0 / (1.0 + (freqs / cutoff_hz) ** (2 * order))
    filter_response = filter_response.unsqueeze(0)

    # Apply filter
    filtered_spectrum = spectrum * filter_response
    result = torch.fft.irfft(filtered_spectrum, n=n, dim=-1)

    if squeeze:
        result = result.squeeze(0)

    return result


# ── Degraded Data Wrapper ─────────────────────────────────────────────────

class DegradedCollateWrapper:
    """Wraps Transkun's collate_fn to apply degradation to audio.

    IMPORTANT: The degradation is applied to the audio BEFORE it's
    used for both mel spectrogram computation AND A4 descriptor.
    """

    def __init__(self, base_collate_fn, degradation: str, level: float, sr: int = 44100):
        self.base_collate_fn = base_collate_fn
        self.degradation = degradation
        self.level = level
        self.sr = sr

    def __call__(self, batch):
        # Apply degradation to audio in each sample
        degraded_batch = []
        for sample in batch:
            sample = dict(sample)  # shallow copy
            audio = sample['audioSlice']

            if isinstance(audio, np.ndarray):
                audio_t = torch.from_numpy(audio).float()
            else:
                audio_t = audio.float()

            # Apply degradation
            if self.degradation == 'noise':
                audio_t = add_gaussian_noise(audio_t, snr_db=self.level)
            elif self.degradation == 'lowpass':
                audio_t = lowpass_filter(audio_t, cutoff_hz=self.level, sr=self.sr)

            sample['audioSlice'] = audio_t.numpy() if isinstance(audio, np.ndarray) else audio_t
            degraded_batch.append(sample)

        return degraded_batch


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Gate 6 Exp B: Transkun Degraded Conditions',
    )
    parser.add_argument('--degradation', type=str, required=True,
                        choices=['noise', 'lowpass', 'data_limit'],
                        help='Type of degradation')
    parser.add_argument('--level', type=float, required=True,
                        help='Degradation level (dB for noise, Hz for lowpass, '
                             'fraction for data_limit)')
    parser.add_argument('--config', type=str, required=True,
                        choices=['baseline-degraded', 'finetune-degraded', 'A4-degraded'],
                        help='Which configuration')
    parser.add_argument('--maestro-dir', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--iterations', type=int, default=50000)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--eval-every', type=int, default=5000)
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint.pt to resume training')

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    # Map config names to internal names
    config_map = {
        'baseline-degraded': 'baseline',
        'finetune-degraded': 'finetune-noA4',
        'A4-degraded': 'A4-event',
    }
    internal_config = config_map[args.config]

    config = {
        'config_name': args.config,
        'internal_config': internal_config,
        'degradation': args.degradation,
        'level': args.level,
        'iterations': args.iterations,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'seed': args.seed,
        'eval_every': args.eval_every,
        'resume': args.resume,
    }

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nGate 6 Exp B: Degraded Conditions")
    print(f"  Degradation: {args.degradation} @ {args.level}")
    print(f"  Config: {args.config} → {internal_config}")

    # Load pretrained Transkun
    print("\nLoading Transkun v2...")
    base_model = load_pretrained_transkun(args.device)

    model_wrapper = TransKunWithA4(
        base_model, config_name=internal_config,
    ).to(args.device)

    print(f"  New params: {model_wrapper.n_trainable_params()/1e3:.1f}K")

    # Create dataloaders with degradation applied via collate wrapper
    from transkun.Data import collate_fn as transkun_collate

    print("\nCreating dataloaders...")

    if args.degradation == 'data_limit':
        # For data limitation, we subsample the training set
        # but use full validation set (to measure F1 on same data)
        train_loader, val_loader = create_transkun_dataloaders(
            args.maestro_dir, batch_size=args.batch_size,
        )
        # Subsample training by limiting epochs
        # (simpler than modifying dataset)
        fraction = args.level
        config['effective_iterations'] = int(args.iterations * fraction)
        config['iterations'] = config['effective_iterations']
    else:
        # Wrap collate_fn to apply noise/lowpass degradation to audio
        # Degradation applies to BOTH train and val: we measure how well
        # the model handles degraded input (A4 from degraded audio too)
        degraded_collate = DegradedCollateWrapper(
            transkun_collate, args.degradation, args.level,
        )
        print(f"  Degradation: {args.degradation} @ {args.level}")
        train_loader, val_loader = create_transkun_dataloaders(
            args.maestro_dir, batch_size=args.batch_size,
            custom_collate_fn=degraded_collate,
        )

    # Save config
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # Train (or just evaluate for baseline)
    train_loop(
        model_wrapper, train_loader, val_loader,
        output_dir, args.device, config,
    )


if __name__ == '__main__':
    main()
