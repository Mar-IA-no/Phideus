#!/usr/bin/env python3
"""
Gate 5B — Test 9: Invariance Suite.

Extends transposition invariance (Test 4) with additional transformation tests
to characterize model robustness across four axes:

1. Temporal Shift     — shift audio waveform by +/-N samples, MIDI unchanged
2. Velocity Scaling   — scale MIDI velocity by factor, audio unchanged
3. Octave Transposition — shift MIDI pitch by +/-12/24 semitones, audio unchanged
4. Audio Noise        — add calibrated Gaussian noise at various SNR, MIDI unchanged

For each transformation, we measure retrieval S = min(A2M_R@10, M2A_R@10) and
compute delta vs the unmodified baseline.

Usage:
    python experiments/bias_control/gate5b/test09_invariance_suite.py \
        --model models/gate5b/d4a4/best_model.pt

    python experiments/bias_control/gate5b/test09_invariance_suite.py --all
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any, Tuple, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINTS = {
    'D0': 'models/gate5b/D0/best_model.pt',
    'd4a4': 'models/gate5b/d4a4/best_model.pt',
    'a4r': 'models/gate5b/a4r/best_model.pt',
    'd4-a4r': 'models/gate5b/d4-a4r/best_model.pt',
}

# --- Transformation parameters ---
TEMPORAL_SHIFTS = [-8000, -4000, 0, +4000, +8000]       # samples at 16kHz
VELOCITY_FACTORS = [0.5, 0.8, 1.0, 1.2, 1.5]
OCTAVE_SHIFTS = [-24, -12, 0, +12, +24]                 # semitones
NOISE_SNRS = [40, 30, 20, 10, 5]                         # dB


# ---------------------------------------------------------------
#  Extraction helpers
# ---------------------------------------------------------------

@torch.no_grad()
def extract_audio_embeddings(model, dataset, device, batch_size, num_workers):
    """Extract audio embeddings with unmodified inputs."""
    from src.bias_control.datasets.maestro_segments import collate_segments

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_segments,
        pin_memory=True, prefetch_factor=2,
    )

    audio_embs = []
    for batch in tqdm(loader, desc="Extracting audio embeddings"):
        audio = batch['audio'].to(device, non_blocking=True)
        midi_pitch = batch['midi_pitch'].to(device, non_blocking=True)
        midi_velocity = batch['midi_velocity'].to(device, non_blocking=True)
        midi_duration = batch['midi_duration'].to(device, non_blocking=True)
        midi_mask = batch['midi_mask'].to(device, non_blocking=True)

        z_audio, _ = model(audio, midi_pitch, midi_velocity, midi_duration, midi_mask)
        audio_embs.append(z_audio.cpu())

    return torch.cat(audio_embs, dim=0)


@torch.no_grad()
def extract_midi_embeddings(model, dataset, device, batch_size, num_workers):
    """Extract MIDI embeddings with unmodified inputs."""
    from src.bias_control.datasets.maestro_segments import collate_segments

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_segments,
        pin_memory=True, prefetch_factor=2,
    )

    midi_embs = []
    for batch in tqdm(loader, desc="Extracting MIDI embeddings"):
        audio = batch['audio'].to(device, non_blocking=True)
        midi_pitch = batch['midi_pitch'].to(device, non_blocking=True)
        midi_velocity = batch['midi_velocity'].to(device, non_blocking=True)
        midi_duration = batch['midi_duration'].to(device, non_blocking=True)
        midi_mask = batch['midi_mask'].to(device, non_blocking=True)

        _, z_midi = model(audio, midi_pitch, midi_velocity, midi_duration, midi_mask)
        midi_embs.append(z_midi.cpu())

    return torch.cat(midi_embs, dim=0)


@torch.no_grad()
def extract_audio_embeddings_shifted(model, dataset, device, batch_size, num_workers, shift):
    """Extract audio embeddings with temporally shifted waveform."""
    from src.bias_control.datasets.maestro_segments import collate_segments

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_segments,
        pin_memory=True, prefetch_factor=2,
    )

    audio_embs = []
    for batch in tqdm(loader, desc=f"Audio shift={shift:+d}"):
        audio = batch['audio'].to(device, non_blocking=True)
        midi_pitch = batch['midi_pitch'].to(device, non_blocking=True)
        midi_velocity = batch['midi_velocity'].to(device, non_blocking=True)
        midi_duration = batch['midi_duration'].to(device, non_blocking=True)
        midi_mask = batch['midi_mask'].to(device, non_blocking=True)

        # Temporal shift: roll + zero-pad the wrapped region
        audio_t = torch.roll(audio, shifts=shift, dims=1)
        if shift > 0:
            audio_t[:, :shift] = 0.0
        elif shift < 0:
            audio_t[:, shift:] = 0.0

        z_audio, _ = model(audio_t, midi_pitch, midi_velocity, midi_duration, midi_mask)
        audio_embs.append(z_audio.cpu())

    return torch.cat(audio_embs, dim=0)


@torch.no_grad()
def extract_midi_embeddings_velocity(model, dataset, device, batch_size, num_workers, factor):
    """Extract MIDI embeddings with scaled velocity."""
    from src.bias_control.datasets.maestro_segments import collate_segments

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_segments,
        pin_memory=True, prefetch_factor=2,
    )

    midi_embs = []
    for batch in tqdm(loader, desc=f"Velocity x{factor:.1f}"):
        audio = batch['audio'].to(device, non_blocking=True)
        midi_pitch = batch['midi_pitch'].to(device, non_blocking=True)
        midi_velocity = batch['midi_velocity'].to(device, non_blocking=True)
        midi_duration = batch['midi_duration'].to(device, non_blocking=True)
        midi_mask = batch['midi_mask'].to(device, non_blocking=True)

        # Scale velocity on valid positions only
        vel_t = midi_velocity.clone()
        valid = ~midi_mask  # mask is True=padding, so ~mask = valid
        vel_t[valid] = (vel_t[valid].float() * factor).long().clamp(0, 127)

        _, z_midi = model(audio, midi_pitch, vel_t, midi_duration, midi_mask)
        midi_embs.append(z_midi.cpu())

    return torch.cat(midi_embs, dim=0)


@torch.no_grad()
def extract_midi_embeddings_octave(model, dataset, device, batch_size, num_workers, shift):
    """Extract MIDI embeddings with octave-transposed pitch."""
    from src.bias_control.datasets.maestro_segments import collate_segments

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_segments,
        pin_memory=True, prefetch_factor=2,
    )

    midi_embs = []
    for batch in tqdm(loader, desc=f"Octave shift={shift:+d}"):
        audio = batch['audio'].to(device, non_blocking=True)
        midi_pitch = batch['midi_pitch'].to(device, non_blocking=True)
        midi_velocity = batch['midi_velocity'].to(device, non_blocking=True)
        midi_duration = batch['midi_duration'].to(device, non_blocking=True)
        midi_mask = batch['midi_mask'].to(device, non_blocking=True)

        # Transpose pitch: clone, shift valid positions, clamp
        pitch_t = midi_pitch.clone()
        valid = ~midi_mask
        pitch_t[valid] = (pitch_t[valid] + shift).clamp(0, 127)

        _, z_midi = model(audio, pitch_t, midi_velocity, midi_duration, midi_mask)
        midi_embs.append(z_midi.cpu())

    return torch.cat(midi_embs, dim=0)


@torch.no_grad()
def extract_audio_embeddings_noisy(model, dataset, device, batch_size, num_workers, snr_db):
    """Extract audio embeddings with additive Gaussian noise at given SNR."""
    from src.bias_control.datasets.maestro_segments import collate_segments

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_segments,
        pin_memory=True, prefetch_factor=2,
    )

    audio_embs = []
    for batch in tqdm(loader, desc=f"Audio noise SNR={snr_db}dB"):
        audio = batch['audio'].to(device, non_blocking=True)
        midi_pitch = batch['midi_pitch'].to(device, non_blocking=True)
        midi_velocity = batch['midi_velocity'].to(device, non_blocking=True)
        midi_duration = batch['midi_duration'].to(device, non_blocking=True)
        midi_mask = batch['midi_mask'].to(device, non_blocking=True)

        # Compute per-sample RMS, add calibrated noise
        # audio shape: [B, 64000]
        audio_std = audio.std(dim=1, keepdim=True).clamp(min=1e-8)
        noise_scale = audio_std * (10.0 ** (-snr_db / 20.0))
        noise = noise_scale * torch.randn_like(audio)
        audio_noisy = audio + noise

        z_audio, _ = model(audio_noisy, midi_pitch, midi_velocity, midi_duration, midi_mask)
        audio_embs.append(z_audio.cpu())

    return torch.cat(audio_embs, dim=0)


# ---------------------------------------------------------------
#  Retrieval evaluation
# ---------------------------------------------------------------

def evaluate_retrieval_from_embeddings(audio_embs, midi_embs, dataset, index, seed=42):
    """Run structured retrieval evaluation with pre-computed embeddings."""
    from experiments.bias_control.evaluate_structured_pool import (
        evaluate_with_precomputed_embeddings, PoolConfig,
    )

    config = PoolConfig(pool_size=256, n_hard_negatives=64,
                        n_semi_hard_negatives=32, n_queries=500)

    a2m = evaluate_with_precomputed_embeddings(
        audio_embs, midi_embs, dataset, index, config,
        direction='a2m', seed=seed,
    )
    m2a = evaluate_with_precomputed_embeddings(
        audio_embs, midi_embs, dataset, index, config,
        direction='m2a', seed=seed,
    )

    S = min(a2m['mean_recall@10'], m2a['mean_recall@10'])
    return S, a2m['mean_recall@10'], m2a['mean_recall@10']


# ---------------------------------------------------------------
#  Per-subtest runners
# ---------------------------------------------------------------

def run_temporal_shift_test(model, dataset, index, device, batch_size, num_workers,
                            midi_embs_baseline, seed):
    """Run temporal shift sub-test (audio shifted, MIDI unchanged)."""
    logger.info("\n--- Temporal Shift Test ---")
    results = {}

    for shift in TEMPORAL_SHIFTS:
        logger.info(f"  Temporal shift: {shift:+d} samples ({shift / 16000:+.3f}s)")
        if shift == 0:
            audio_embs = extract_audio_embeddings(model, dataset, device, batch_size, num_workers)
        else:
            audio_embs = extract_audio_embeddings_shifted(
                model, dataset, device, batch_size, num_workers, shift,
            )

        S, a2m, m2a = evaluate_retrieval_from_embeddings(
            audio_embs, midi_embs_baseline, dataset, index, seed,
        )
        results[shift] = {'S': S, 'a2m_r10': a2m, 'm2a_r10': m2a}
        logger.info(f"    S={S:.1%}")

    S_baseline = results[0]['S']
    for shift, data in results.items():
        data['delta_S'] = data['S'] - S_baseline

    return {
        'shifts': {str(k): v for k, v in results.items()},
        'S_baseline': S_baseline,
    }


def run_velocity_scaling_test(model, dataset, index, device, batch_size, num_workers,
                               audio_embs_baseline, seed):
    """Run velocity scaling sub-test (MIDI velocity scaled, audio unchanged)."""
    logger.info("\n--- Velocity Scaling Test ---")
    results = {}

    for factor in VELOCITY_FACTORS:
        logger.info(f"  Velocity factor: {factor:.1f}")
        if factor == 1.0:
            midi_embs = extract_midi_embeddings(model, dataset, device, batch_size, num_workers)
        else:
            midi_embs = extract_midi_embeddings_velocity(
                model, dataset, device, batch_size, num_workers, factor,
            )

        S, a2m, m2a = evaluate_retrieval_from_embeddings(
            audio_embs_baseline, midi_embs, dataset, index, seed,
        )
        results[factor] = {'S': S, 'a2m_r10': a2m, 'm2a_r10': m2a}
        logger.info(f"    S={S:.1%}")

    S_baseline = results[1.0]['S']
    for factor, data in results.items():
        data['delta_S'] = data['S'] - S_baseline

    return {
        'factors': {str(k): v for k, v in results.items()},
        'S_baseline': S_baseline,
    }


def run_octave_transposition_test(model, dataset, index, device, batch_size, num_workers,
                                   audio_embs_baseline, seed):
    """Run octave transposition sub-test (MIDI pitch shifted by octaves, audio unchanged)."""
    logger.info("\n--- Octave Transposition Test ---")
    results = {}

    for shift in OCTAVE_SHIFTS:
        logger.info(f"  Octave shift: {shift:+d} semitones")
        if shift == 0:
            midi_embs = extract_midi_embeddings(model, dataset, device, batch_size, num_workers)
        else:
            midi_embs = extract_midi_embeddings_octave(
                model, dataset, device, batch_size, num_workers, shift,
            )

        S, a2m, m2a = evaluate_retrieval_from_embeddings(
            audio_embs_baseline, midi_embs, dataset, index, seed,
        )
        results[shift] = {'S': S, 'a2m_r10': a2m, 'm2a_r10': m2a}
        logger.info(f"    S={S:.1%}")

    S_baseline = results[0]['S']
    for shift, data in results.items():
        data['delta_S'] = data['S'] - S_baseline

    return {
        'shifts': {str(k): v for k, v in results.items()},
        'S_baseline': S_baseline,
    }


def run_audio_noise_test(model, dataset, index, device, batch_size, num_workers,
                          midi_embs_baseline, seed):
    """Run audio noise sub-test (Gaussian noise at calibrated SNR, MIDI unchanged)."""
    logger.info("\n--- Audio Noise Test ---")
    results = {}

    # Baseline = clean audio (effectively infinite SNR)
    logger.info("  Baseline (clean audio)")
    audio_embs_clean = extract_audio_embeddings(model, dataset, device, batch_size, num_workers)
    S_clean, a2m_clean, m2a_clean = evaluate_retrieval_from_embeddings(
        audio_embs_clean, midi_embs_baseline, dataset, index, seed,
    )
    S_baseline = S_clean
    results['clean'] = {'S': S_clean, 'a2m_r10': a2m_clean, 'm2a_r10': m2a_clean, 'delta_S': 0.0}
    logger.info(f"    S={S_clean:.1%}")

    for snr in NOISE_SNRS:
        logger.info(f"  SNR: {snr} dB")
        audio_embs = extract_audio_embeddings_noisy(
            model, dataset, device, batch_size, num_workers, snr,
        )

        S, a2m, m2a = evaluate_retrieval_from_embeddings(
            audio_embs, midi_embs_baseline, dataset, index, seed,
        )
        results[snr] = {
            'S': S, 'a2m_r10': a2m, 'm2a_r10': m2a,
            'delta_S': S - S_baseline,
        }
        logger.info(f"    S={S:.1%} (delta={S - S_baseline:+.1%})")

    return {
        'snr_levels': {str(k): v for k, v in results.items()},
        'S_baseline': S_baseline,
    }


# ---------------------------------------------------------------
#  Main evaluation
# ---------------------------------------------------------------

def evaluate_single(model_path, maestro_dir, device, num_workers, seed):
    """Run full invariance suite on one checkpoint."""
    from experiments.bias_control.gate5b.harness import (
        setup_gate5b_test, get_output_dir, save_test_result,
    )
    from experiments.bias_control.gate5b.checkpoint_loader import get_eval_batch_size

    class Args:
        pass
    args = Args()
    args.model = model_path
    args.maestro_dir = maestro_dir
    args.device = device
    args.seed = seed
    args.num_workers = num_workers
    args.output = None

    model, meta, dataset, index = setup_gate5b_test(args)
    descriptor = meta['descriptor']
    batch_size = get_eval_batch_size(descriptor)

    dev = torch.device(device if torch.cuda.is_available() else 'cpu')

    # ---- Extract baseline embeddings (unmodified) ----
    logger.info("Extracting baseline audio embeddings...")
    audio_embs_baseline = extract_audio_embeddings(
        model, dataset, dev, batch_size, num_workers,
    )
    logger.info("Extracting baseline MIDI embeddings...")
    midi_embs_baseline = extract_midi_embeddings(
        model, dataset, dev, batch_size, num_workers,
    )

    # ---- Run all four sub-tests ----
    result = {}

    result['temporal_shift'] = run_temporal_shift_test(
        model, dataset, index, dev, batch_size, num_workers,
        midi_embs_baseline, seed,
    )

    result['velocity_scaling'] = run_velocity_scaling_test(
        model, dataset, index, dev, batch_size, num_workers,
        audio_embs_baseline, seed,
    )

    result['octave_transposition'] = run_octave_transposition_test(
        model, dataset, index, dev, batch_size, num_workers,
        audio_embs_baseline, seed,
    )

    result['audio_noise'] = run_audio_noise_test(
        model, dataset, index, dev, batch_size, num_workers,
        midi_embs_baseline, seed,
    )

    # ---- Save ----
    output_dir = get_output_dir(args, meta)
    save_test_result(result, output_dir, 'test09_invariance_suite', meta, seed=seed)

    return result, meta


def main():
    parser = argparse.ArgumentParser(
        description='Gate 5B - Test 9: Invariance Suite'
    )
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--maestro-dir', type=str, default='data/maestro_v3/maestro-v3.0.0')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num-workers', type=int, default=8)

    args = parser.parse_args()

    if not args.model and not args.all:
        parser.error("Provide --model PATH or --all")

    models = {}
    if args.all:
        models = dict(DEFAULT_CHECKPOINTS)
    else:
        models = {'custom': args.model}

    all_results = {}

    for arm, path in models.items():
        if not Path(path).exists():
            logger.warning(f"Checkpoint not found: {path} -- skipping {arm}")
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"INVARIANCE SUITE: {arm}")
        logger.info(f"{'='*60}")

        start = time.time()
        result, meta = evaluate_single(
            path, args.maestro_dir, args.device, args.num_workers, args.seed,
        )
        elapsed = time.time() - start

        all_results[arm] = {
            'descriptor': meta['descriptor'],
            'temporal_shift': result['temporal_shift'],
            'velocity_scaling': result['velocity_scaling'],
            'octave_transposition': result['octave_transposition'],
            'audio_noise': result['audio_noise'],
            'time_s': elapsed,
        }

    # ---- Summary table ----
    if not all_results:
        logger.warning("No checkpoints evaluated.")
        return

    logger.info(f"\n{'='*60}")
    logger.info("INVARIANCE SUITE SUMMARY")
    logger.info(f"{'='*60}")

    # Temporal shift summary
    logger.info("\n[Temporal Shift] delta_S by shift (samples)")
    header = f"{'Arm':<10} {'S(0)':>6}"
    for shift in TEMPORAL_SHIFTS:
        if shift != 0:
            header += f" {'d(' + str(shift) + ')':>10}"
    logger.info(header)
    logger.info("-" * len(header))

    for arm, data in all_results.items():
        ts = data['temporal_shift']
        line = f"{arm:<10} {ts['S_baseline']:>5.1%}"
        for shift in TEMPORAL_SHIFTS:
            if shift != 0:
                delta = ts['shifts'].get(str(shift), {}).get('delta_S', 0)
                line += f" {delta:>+9.1%}"
        logger.info(line)

    # Velocity scaling summary
    logger.info("\n[Velocity Scaling] delta_S by factor")
    header = f"{'Arm':<10} {'S(1.0)':>6}"
    for f in VELOCITY_FACTORS:
        if f != 1.0:
            header += f" {'d(' + str(f) + ')':>10}"
    logger.info(header)
    logger.info("-" * len(header))

    for arm, data in all_results.items():
        vs = data['velocity_scaling']
        line = f"{arm:<10} {vs['S_baseline']:>5.1%}"
        for f in VELOCITY_FACTORS:
            if f != 1.0:
                delta = vs['factors'].get(str(f), {}).get('delta_S', 0)
                line += f" {delta:>+9.1%}"
        logger.info(line)

    # Octave transposition summary
    logger.info("\n[Octave Transposition] delta_S by semitones")
    header = f"{'Arm':<10} {'S(0)':>6}"
    for shift in OCTAVE_SHIFTS:
        if shift != 0:
            header += f" {'d(' + str(shift) + ')':>10}"
    logger.info(header)
    logger.info("-" * len(header))

    for arm, data in all_results.items():
        ot = data['octave_transposition']
        line = f"{arm:<10} {ot['S_baseline']:>5.1%}"
        for shift in OCTAVE_SHIFTS:
            if shift != 0:
                delta = ot['shifts'].get(str(shift), {}).get('delta_S', 0)
                line += f" {delta:>+9.1%}"
        logger.info(line)

    # Audio noise summary
    logger.info("\n[Audio Noise] delta_S by SNR (dB)")
    header = f"{'Arm':<10} {'S(clean)':>8}"
    for snr in NOISE_SNRS:
        header += f" {'d(' + str(snr) + 'dB)':>10}"
    logger.info(header)
    logger.info("-" * len(header))

    for arm, data in all_results.items():
        an = data['audio_noise']
        line = f"{arm:<10} {an['S_baseline']:>7.1%}"
        for snr in NOISE_SNRS:
            delta = an['snr_levels'].get(str(snr), {}).get('delta_S', 0)
            line += f" {delta:>+9.1%}"
        logger.info(line)


if __name__ == '__main__':
    main()
