#!/usr/bin/env python3
"""
Gate 5B — Test 8: Ratio Decoding Report.

Analyzes which descriptor features contribute most to the final embedding.

Limitation: descriptors are computed under torch.no_grad() + .detach(),
so gradient-based attribution is not possible. Instead, we use:

1. Perturbation-based sensitivity: For each dim of the descriptor, perturb
   ±epsilon and measure change in embedding (no grad required).
2. Ablation by feature: Zero one dim at a time, measure delta S.
3. Correlation analysis: Pearson correlation between each descriptor dim
   and embedding components.

Usage:
    python experiments/bias_control/gate5b/test08_ratio_decoding.py \
        --model models/gate5b/d4a4/best_model.pt

    python experiments/bias_control/gate5b/test08_ratio_decoding.py --all
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
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINTS = {
    'd4a4': 'models/gate5b/d4a4/best_model.pt',
    'a4r': 'models/gate5b/a4r/best_model.pt',
    'd4-a4r': 'models/gate5b/d4-a4r/best_model.pt',
}

# Descriptor dimensions
# A4: temporal deltas of log-magnitude in 8 octave frequency bands (STFT-based)
# See src/bias_control/audio_descriptors.py::compute_audio_descriptor_a4()
DESCRIPTOR_DIMS = {
    'a4': {'name': 'Audio A4 (octave-band log-mag delta)', 'dim': 8,
           'features': ['band0_47Hz', 'band1_94Hz', 'band2_188Hz', 'band3_375Hz',
                        'band4_750Hz', 'band5_1500Hz', 'band6_3000Hz', 'band7_6000Hz']},
    'd4': {'name': 'MIDI D4 (local intervals)', 'dim': 4,
           'features': ['interval_prev', 'interval_next', 'duration_ratio', 'velocity_diff']},
}

# Which descriptors each arm has
DESCRIPTOR_MAP = {
    'd4a4': {'audio': 'a4', 'midi': 'd4'},
    'a4r': {'audio': 'a4', 'midi': None},
    'd4-a4r': {'audio': 'a4', 'midi': 'd4'},
}


@torch.no_grad()
def collect_descriptors_and_embeddings(model, dataset, device, descriptor, num_workers=8,
                                        n_batches=30):
    """Collect raw descriptors and corresponding embeddings."""
    import experiments.bias_control.gate43_scratch.gate43_scratch_training as g43
    from src.bias_control.datasets.maestro_segments import collate_segments
    from experiments.bias_control.gate5b.checkpoint_loader import get_eval_batch_size

    batch_size = get_eval_batch_size(descriptor)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_segments,
        pin_memory=True,
    )

    desc_info = DESCRIPTOR_MAP.get(descriptor, {})
    dev = torch.device(device if torch.cuda.is_available() else 'cpu')

    audio_descs, midi_descs = [], []
    audio_embs, midi_embs = [], []

    for i, batch in enumerate(tqdm(loader, desc="Collecting descriptors")):
        if i >= n_batches:
            break

        audio = batch['audio'].to(dev, non_blocking=True)
        midi_pitch = batch['midi_pitch'].to(dev, non_blocking=True)
        midi_velocity = batch['midi_velocity'].to(dev, non_blocking=True)
        midi_duration = batch['midi_duration'].to(dev, non_blocking=True)
        midi_mask = batch['midi_mask'].to(dev, non_blocking=True)

        # Get embeddings
        z_audio, z_midi = model(audio, midi_pitch, midi_velocity, midi_duration, midi_mask)
        audio_embs.append(z_audio.cpu())
        midi_embs.append(z_midi.cpu())

        # Get raw descriptors
        if desc_info.get('audio'):
            a4 = g43.compute_audio_descriptor_a4(audio, target_length=None)
            # Mean pool over time: [B, T, 8] -> [B, 8]
            audio_descs.append(a4.mean(dim=1).cpu())

        if desc_info.get('midi'):
            d4 = g43.compute_local_interval_features(midi_pitch, midi_mask)
            # Mean pool valid positions: [B, T, 4] -> [B, 4]
            valid = ~midi_mask
            pooled = []
            for b in range(d4.size(0)):
                v = d4[b][valid[b]]
                pooled.append(v.mean(dim=0) if v.size(0) > 0 else d4[b].mean(dim=0))
            midi_descs.append(torch.stack(pooled).cpu())

    result = {
        'audio_embs': torch.cat(audio_embs, dim=0),
        'midi_embs': torch.cat(midi_embs, dim=0),
    }
    if audio_descs:
        result['audio_descs'] = torch.cat(audio_descs, dim=0)
    if midi_descs:
        result['midi_descs'] = torch.cat(midi_descs, dim=0)

    return result


def correlation_analysis(descs, embs, feature_names):
    """Pearson correlation between each descriptor dim and embedding components."""
    N, D_desc = descs.shape
    N2, D_emb = embs.shape
    assert N == N2

    descs_np = descs.numpy()
    embs_np = embs.numpy()

    # Per descriptor dim: mean |correlation| with all embedding dims
    importance = []
    for d in range(D_desc):
        corrs = []
        for e in range(D_emb):
            r = np.corrcoef(descs_np[:, d], embs_np[:, e])[0, 1]
            if np.isnan(r):
                r = 0.0
            corrs.append(abs(r))
        mean_abs_corr = float(np.mean(corrs))
        max_abs_corr = float(np.max(corrs))
        importance.append({
            'feature': feature_names[d] if d < len(feature_names) else f'dim_{d}',
            'mean_abs_corr': mean_abs_corr,
            'max_abs_corr': max_abs_corr,
        })

    # Sort by importance
    importance.sort(key=lambda x: -x['mean_abs_corr'])
    return importance


@torch.no_grad()
def perturbation_sensitivity(model, dataset, device, descriptor, num_workers=8,
                             epsilon=0.1, n_batches=20):
    """Measure embedding sensitivity to perturbation of each descriptor dim."""
    import experiments.bias_control.gate43_scratch.gate43_scratch_training as g43
    from src.bias_control.datasets.maestro_segments import collate_segments
    from experiments.bias_control.gate5b.checkpoint_loader import get_eval_batch_size

    desc_info = DESCRIPTOR_MAP.get(descriptor, {})
    batch_size = get_eval_batch_size(descriptor)
    dev = torch.device(device if torch.cuda.is_available() else 'cpu')

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_segments,
        pin_memory=True,
    )

    # For each descriptor type, perturb one dim at a time
    results = {}

    for side, desc_type in [('audio', desc_info.get('audio')), ('midi', desc_info.get('midi'))]:
        if desc_type is None:
            continue

        n_dims = DESCRIPTOR_DIMS[desc_type]['dim']
        feature_names = DESCRIPTOR_DIMS[desc_type]['features']

        # Sensitivity per dim
        sensitivities = [[] for _ in range(n_dims)]

        for batch_i, batch in enumerate(tqdm(loader, desc=f"Perturbation ({side})")):
            if batch_i >= n_batches:
                break

            audio = batch['audio'].to(dev, non_blocking=True)
            midi_pitch = batch['midi_pitch'].to(dev, non_blocking=True)
            midi_velocity = batch['midi_velocity'].to(dev, non_blocking=True)
            midi_duration = batch['midi_duration'].to(dev, non_blocking=True)
            midi_mask = batch['midi_mask'].to(dev, non_blocking=True)

            # Baseline embeddings
            z_audio_base, z_midi_base = model(audio, midi_pitch, midi_velocity, midi_duration, midi_mask)

            for dim in range(n_dims):
                # Create perturbation wrapper for this dim
                if side == 'audio':
                    orig_fn = g43.compute_audio_descriptor_a4

                    def perturbed_fn(*args, _orig=orig_fn, _dim=dim, _eps=epsilon, **kwargs):
                        result = _orig(*args, **kwargs)
                        perturbation = torch.zeros_like(result)
                        perturbation[:, :, _dim] = _eps
                        return result + perturbation

                    g43.compute_audio_descriptor_a4 = perturbed_fn
                else:
                    orig_fn = g43.compute_local_interval_features

                    def perturbed_fn(*args, _orig=orig_fn, _dim=dim, _eps=epsilon, **kwargs):
                        result = _orig(*args, **kwargs)
                        perturbation = torch.zeros_like(result)
                        perturbation[:, :, _dim] = _eps
                        return result + perturbation

                    g43.compute_local_interval_features = perturbed_fn

                # Forward with perturbation
                z_audio_pert, z_midi_pert = model(audio, midi_pitch, midi_velocity, midi_duration, midi_mask)

                # Restore
                if side == 'audio':
                    g43.compute_audio_descriptor_a4 = orig_fn
                else:
                    g43.compute_local_interval_features = orig_fn

                # Measure change
                if side == 'audio':
                    diff = (z_audio_pert - z_audio_base).norm(dim=1).mean().item()
                else:
                    diff = (z_midi_pert - z_midi_base).norm(dim=1).mean().item()

                sensitivities[dim].append(diff)

        dim_results = []
        for dim in range(n_dims):
            mean_sens = float(np.mean(sensitivities[dim])) if sensitivities[dim] else 0.0
            dim_results.append({
                'feature': feature_names[dim] if dim < len(feature_names) else f'dim_{dim}',
                'mean_sensitivity': mean_sens,
            })

        dim_results.sort(key=lambda x: -x['mean_sensitivity'])
        results[side] = dim_results

    return results


def evaluate_single(model_path, maestro_dir, device, num_workers, seed):
    """Run ratio decoding analysis on one checkpoint."""
    from experiments.bias_control.gate5b.harness import (
        setup_gate5b_test, get_output_dir, save_test_result,
    )

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

    desc_info = DESCRIPTOR_MAP.get(descriptor)
    if not desc_info:
        logger.info(f"Descriptor {descriptor} not in DESCRIPTOR_MAP — skipping")
        return None, meta

    # 1. Collect descriptors and embeddings
    logger.info("Collecting descriptors and embeddings...")
    data = collect_descriptors_and_embeddings(
        model, dataset, device, descriptor, num_workers=num_workers,
    )

    # 2. Correlation analysis
    logger.info("Running correlation analysis...")
    correlation_results = {}
    if 'audio_descs' in data:
        info = DESCRIPTOR_DIMS['a4']
        correlation_results['audio'] = correlation_analysis(
            data['audio_descs'], data['audio_embs'], info['features'],
        )
    if 'midi_descs' in data:
        info = DESCRIPTOR_DIMS['d4']
        correlation_results['midi'] = correlation_analysis(
            data['midi_descs'], data['midi_embs'], info['features'],
        )

    # 3. Perturbation sensitivity
    logger.info("Running perturbation sensitivity analysis...")
    perturbation_results = perturbation_sensitivity(
        model, dataset, device, descriptor, num_workers=num_workers,
    )

    result = {
        'correlation': correlation_results,
        'perturbation_sensitivity': perturbation_results,
        'n_samples': len(data['audio_embs']),
        'descriptor_dims': {
            side: DESCRIPTOR_DIMS[dt]
            for side, dt in desc_info.items()
            if dt is not None
        },
    }

    output_dir = get_output_dir(args, meta)
    save_test_result(result, output_dir, 'test08_ratio_decoding', meta, seed=seed)

    return result, meta


def main():
    parser = argparse.ArgumentParser(description='Gate 5B — Test 8: Ratio Decoding Report')
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
            logger.warning(f"Checkpoint not found: {path} — skipping {arm}")
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"RATIO DECODING: {arm}")
        logger.info(f"{'='*60}")

        start = time.time()
        result, meta = evaluate_single(
            path, args.maestro_dir, args.device, args.num_workers, args.seed,
        )
        elapsed = time.time() - start

        if result:
            all_results[arm] = result
            logger.info(f"  Completed in {elapsed:.0f}s")

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("RATIO DECODING SUMMARY")
    logger.info(f"{'='*60}")

    for arm, data in all_results.items():
        logger.info(f"\n{arm}:")
        for side in ['audio', 'midi']:
            if side in data.get('perturbation_sensitivity', {}):
                logger.info(f"  {side} perturbation sensitivity (top features):")
                for feat in data['perturbation_sensitivity'][side][:4]:
                    logger.info(f"    {feat['feature']:<20} sens={feat['mean_sensitivity']:.4f}")
            if side in data.get('correlation', {}):
                logger.info(f"  {side} correlation (top features):")
                for feat in data['correlation'][side][:4]:
                    logger.info(f"    {feat['feature']:<20} |r|={feat['mean_abs_corr']:.4f}")


if __name__ == '__main__':
    main()
