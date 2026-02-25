#!/usr/bin/env python3
"""
Gate 5B — Test 1: Causal Ablation (zero-out injection).

Zeroes out the descriptor in the forward pass and measures how much S drops.
This tests whether the descriptor causally contributes to the model's performance,
or if the augmentation layers merely add parameters.

Ablation modes:
  - zero_audio:  Zero the audio descriptor (A4) before injection
  - zero_midi:   Zero the MIDI descriptor (D4 intervals) before injection
  - zero_both:   Zero both descriptors
  - noise:       Replace descriptors with Gaussian noise (same mean/std)
  - shuffle:     Permute descriptors across batch samples

For D0 (no descriptor), all modes should produce delta~0 (sanity check).

Usage:
    python experiments/bias_control/gate5b/test01_causal_ablation.py \
        --model models/gate5b/d4a4/best_model.pt

    python experiments/bias_control/gate5b/test01_causal_ablation.py --all
"""

import argparse
import json
import logging
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Any, Tuple

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default checkpoint paths
DEFAULT_CHECKPOINTS = {
    'D0': 'models/gate5b/D0/best_model.pt',
    'd4a4': 'models/gate5b/d4a4/best_model.pt',
    'a4r': 'models/gate5b/a4r/best_model.pt',
    'd4-a4r': 'models/gate5b/d4-a4r/best_model.pt',
}

# Which descriptors each arm uses
DESCRIPTOR_MAP = {
    'd0': {'audio': None, 'midi': None},
    'd4a4': {'audio': 'a4', 'midi': 'd4'},
    'a4r': {'audio': 'a4', 'midi': None},
    'd4-a4r': {'audio': 'a4', 'midi': 'd4'},
}


def _make_zero_wrapper(original_fn, mode='zero', stats=None):
    """Create a wrapper that ablates the descriptor output."""
    def zero_wrapper(*args, **kwargs):
        result = original_fn(*args, **kwargs)
        if mode == 'zero':
            return torch.zeros_like(result)
        elif mode == 'noise':
            if stats is not None:
                return torch.randn_like(result) * stats['std'] + stats['mean']
            return torch.randn_like(result)
        elif mode == 'shuffle':
            # Permute across batch dimension
            B = result.size(0)
            perm = torch.randperm(B, device=result.device)
            return result[perm]
        else:
            return torch.zeros_like(result)
    return zero_wrapper


@contextmanager
def ablate_descriptors(descriptor: str, ablation_mode: str, which: str = 'both',
                       stats_audio=None, stats_midi=None):
    """
    Context manager that monkey-patches descriptor functions in the gate43 module.

    Args:
        descriptor: model descriptor (d4a4, a4r, d4-a4r)
        ablation_mode: 'zero', 'noise', 'shuffle'
        which: 'audio', 'midi', or 'both'
        stats_audio: dict with 'mean', 'std' for noise mode (audio descriptor)
        stats_midi: dict with 'mean', 'std' for noise mode (midi descriptor)
    """
    import experiments.bias_control.gate43_scratch.gate43_scratch_training as g43

    desc_info = DESCRIPTOR_MAP.get(descriptor, {})
    originals = {}

    try:
        # Patch audio descriptor if applicable
        if which in ('audio', 'both') and desc_info.get('audio'):
            orig_a4 = g43.compute_audio_descriptor_a4
            originals['audio_a4'] = orig_a4
            g43.compute_audio_descriptor_a4 = _make_zero_wrapper(
                orig_a4, mode=ablation_mode, stats=stats_audio
            )

        # Patch MIDI descriptor if applicable
        if which in ('midi', 'both') and desc_info.get('midi'):
            orig_d4 = g43.compute_local_interval_features
            originals['midi_d4'] = orig_d4
            g43.compute_local_interval_features = _make_zero_wrapper(
                orig_d4, mode=ablation_mode, stats=stats_midi
            )

        yield
    finally:
        # Restore originals
        if 'audio_a4' in originals:
            g43.compute_audio_descriptor_a4 = originals['audio_a4']
        if 'midi_d4' in originals:
            g43.compute_local_interval_features = originals['midi_d4']


def collect_descriptor_stats(model, dataset, device, descriptor, num_batches=20, num_workers=8):
    """Collect mean/std of descriptors for noise-mode ablation."""
    import experiments.bias_control.gate43_scratch.gate43_scratch_training as g43
    from src.bias_control.datasets.maestro_segments import collate_segments
    from torch.utils.data import DataLoader

    loader = DataLoader(
        dataset, batch_size=16, shuffle=False, num_workers=num_workers,
        collate_fn=collate_segments, pin_memory=True,
    )

    desc_info = DESCRIPTOR_MAP.get(descriptor, {})
    audio_vals, midi_vals = [], []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= num_batches:
                break
            audio = batch['audio'].to(device, non_blocking=True)
            midi_pitch = batch['midi_pitch'].to(device, non_blocking=True)
            midi_mask = batch['midi_mask'].to(device, non_blocking=True)

            if desc_info.get('audio'):
                a4 = g43.compute_audio_descriptor_a4(audio, target_length=None)
                audio_vals.append(a4.cpu())

            if desc_info.get('midi'):
                d4 = g43.compute_local_interval_features(midi_pitch, midi_mask)
                midi_vals.append(d4.cpu())

    stats_audio = None
    stats_midi = None
    if audio_vals:
        all_audio = torch.cat(audio_vals, dim=0)
        stats_audio = {'mean': all_audio.mean().item(), 'std': all_audio.std().item()}
    if midi_vals:
        # D4 features have variable seq length per batch — flatten each to [B*N, 4]
        flat_midi = torch.cat([v.reshape(-1, v.size(-1)) for v in midi_vals], dim=0)
        stats_midi = {'mean': flat_midi.mean().item(), 'std': flat_midi.std().item()}

    return stats_audio, stats_midi


def run_eval(model, dataset, index, device, descriptor, num_workers=8, seed=42):
    """Run structured pool evaluation and return S."""
    from experiments.bias_control.gate5b.harness import extract_embeddings
    from experiments.bias_control.evaluate_structured_pool import (
        evaluate_with_precomputed_embeddings, PoolConfig,
    )

    audio_embs, midi_embs = extract_embeddings(
        model, dataset, device, descriptor, num_workers=num_workers,
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


def verify_ablation_effective(model, dataset, device, descriptor, num_workers=8):
    """Sanity check: verify that patching actually changes embeddings."""
    from experiments.bias_control.gate5b.harness import extract_embeddings

    # Normal embeddings
    audio_normal, midi_normal = extract_embeddings(
        model, dataset, device, descriptor, num_workers=num_workers,
    )

    # Ablated embeddings
    with ablate_descriptors(descriptor, 'zero', 'both'):
        audio_ablated, midi_ablated = extract_embeddings(
            model, dataset, device, descriptor, num_workers=num_workers,
        )

    audio_diff = (audio_normal - audio_ablated).norm(dim=1).mean().item()
    midi_diff = (midi_normal - midi_ablated).norm(dim=1).mean().item()

    logger.info(f"Ablation verification: audio_diff={audio_diff:.4f}, midi_diff={midi_diff:.4f}")

    desc_info = DESCRIPTOR_MAP.get(descriptor, {})
    if desc_info.get('audio') and audio_diff < 1e-6:
        logger.warning("ABLATION NOT EFFECTIVE on audio side!")
        return False
    if desc_info.get('midi') and midi_diff < 1e-6:
        logger.warning("ABLATION NOT EFFECTIVE on midi side!")
        return False

    return True


def evaluate_single(model_path, maestro_dir, device, num_workers, seed):
    """Run full causal ablation on one checkpoint."""
    from experiments.bias_control.gate5b.harness import (
        setup_gate5b_test, get_output_dir, save_test_result,
        get_normal_embeddings,
    )
    from experiments.bias_control.evaluate_structured_pool import (
        evaluate_with_precomputed_embeddings, PoolConfig,
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
    output_dir = get_output_dir(args, meta)

    logger.info(f"\nCausal ablation for descriptor={descriptor}")
    desc_info = DESCRIPTOR_MAP.get(descriptor, {})
    has_audio = desc_info.get('audio') is not None
    has_midi = desc_info.get('midi') is not None

    # 1. Normal evaluation — use cache if available
    logger.info("Evaluating: NORMAL")
    audio_normal, midi_normal = get_normal_embeddings(
        model, dataset, device, descriptor, output_dir,
        num_workers=num_workers,
    )
    config = PoolConfig(pool_size=256, n_hard_negatives=64,
                        n_semi_hard_negatives=32, n_queries=500)
    a2m_res = evaluate_with_precomputed_embeddings(
        audio_normal, midi_normal, dataset, index, config,
        direction='a2m', seed=seed,
    )
    m2a_res = evaluate_with_precomputed_embeddings(
        audio_normal, midi_normal, dataset, index, config,
        direction='m2a', seed=seed,
    )
    S_normal = min(a2m_res['mean_recall@10'], m2a_res['mean_recall@10'])
    a2m_normal = a2m_res['mean_recall@10']
    m2a_normal = m2a_res['mean_recall@10']
    logger.info(f"  S_normal={S_normal:.1%}")

    # 2. Collect stats for noise mode (fast — only 20 batches)
    stats_audio, stats_midi = None, None
    if has_audio or has_midi:
        logger.info("Collecting descriptor stats for noise mode...")
        stats_audio, stats_midi = collect_descriptor_stats(
            model, dataset, device, descriptor, num_workers=num_workers,
        )

    # 4. Run ablation modes
    ablation_results = {}
    modes = []

    if has_audio and has_midi:
        modes = [
            ('zero', 'audio'), ('zero', 'midi'), ('zero', 'both'),
            ('noise', 'audio'), ('noise', 'midi'), ('noise', 'both'),
            ('shuffle', 'audio'), ('shuffle', 'midi'), ('shuffle', 'both'),
        ]
    elif has_audio:
        modes = [('zero', 'audio'), ('noise', 'audio'), ('shuffle', 'audio')]
    elif has_midi:
        modes = [('zero', 'midi'), ('noise', 'midi'), ('shuffle', 'midi')]
    else:
        # D0: no descriptors, skip
        logger.info("D0: no descriptors to ablate, recording baseline only")

    for ablation_mode, which in modes:
        label = f"{ablation_mode}_{which}"
        logger.info(f"Evaluating: {label}")

        with ablate_descriptors(descriptor, ablation_mode, which,
                                stats_audio=stats_audio, stats_midi=stats_midi):
            S_abl, a2m_abl, m2a_abl = run_eval(
                model, dataset, index, device, descriptor, num_workers, seed
            )

        delta = S_normal - S_abl
        ablation_results[label] = {
            'S': S_abl,
            'a2m_r10': a2m_abl,
            'm2a_r10': m2a_abl,
            'delta_S': delta,
        }
        logger.info(f"  S={S_abl:.1%}, delta={delta:+.1%}")

    result = {
        'S_normal': S_normal,
        'a2m_normal': a2m_normal,
        'm2a_normal': m2a_normal,
        'ablations': ablation_results,
        'descriptor_stats': {
            'audio': stats_audio,
            'midi': stats_midi,
        },
        'has_audio_descriptor': has_audio,
        'has_midi_descriptor': has_midi,
    }

    output_dir = get_output_dir(args, meta)
    save_test_result(result, output_dir, 'test01_causal_ablation', meta, seed=seed)

    return result, meta


def main():
    parser = argparse.ArgumentParser(description='Gate 5B — Test 1: Causal Ablation')
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
        logger.info(f"CAUSAL ABLATION: {arm}")
        logger.info(f"{'='*60}")

        start = time.time()
        result, meta = evaluate_single(
            path, args.maestro_dir, args.device, args.num_workers, args.seed,
        )
        elapsed = time.time() - start

        if result:
            all_results[arm] = {
                'descriptor': meta['descriptor'],
                'S_normal': result['S_normal'],
                'ablations': result['ablations'],
                'time_s': elapsed,
            }

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("CAUSAL ABLATION SUMMARY")
    logger.info(f"{'='*60}")

    for arm, data in all_results.items():
        logger.info(f"\n{arm} (descriptor={data['descriptor']}, S_normal={data['S_normal']:.1%}):")
        for mode, vals in data['ablations'].items():
            logger.info(f"  {mode:<20} S={vals['S']:.1%}  delta={vals['delta_S']:+.1%}")


if __name__ == '__main__':
    main()
