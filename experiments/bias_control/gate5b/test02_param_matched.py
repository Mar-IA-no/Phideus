#!/usr/bin/env python3
"""
Gate 5B — Test 02: Parameter-Matched Ablations.

Controls that d4a4's improvement over D0 is CAUSAL (from ratio information)
and not merely from the ~4.5M additional parameters in descriptor projections.

Trains 4 models with d4a4 architecture (~64-68.5M params, run-d freeze) but
with descriptors modified:

  real:     Pass-through (control positive, should match d4a4 ~83%)
  random:   Replace descriptor output with per-dim matched Gaussian noise
  shuffled: Deterministic derangement of descriptors across batch
  zero:     Zero descriptors (params exist but get no signal)

If d4a4's advantage is causal, random/shuffled/zero should fall to ~D0 level (~73%).

Usage (UNC via SLURM — see gate5b_param_matched.sh):
    python experiments/bias_control/gate5b/test02_param_matched.py \\
        --mode zero --epochs 30 --output results_unc/gate5b_param_matched/zero

Local dry-run (1 epoch, 10 batches):
    python experiments/bias_control/gate5b/test02_param_matched.py \\
        --mode zero --epochs 1 --max-batches-per-epoch 10 \\
        --output /tmp/test02_dryrun --device cuda
"""

import argparse
import hashlib
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────
DESCRIPTOR = 'd4a4'
FREEZE_POLICY = 'run-d'
ABLATION_MODES = ('real', 'random', 'shuffled', 'zero')


# ── Deterministic derangement ──────────────────────────────────────────────

def _fixed_derangement(n: int, generator: torch.Generator) -> torch.Tensor:
    """Permutation where no element stays in place (Sattolo's algorithm).

    Args:
        n: Number of elements (must be >= 2).
        generator: Torch generator for determinism.

    Returns:
        Tensor of shape [n] with the derangement indices.
    """
    if n < 2:
        return torch.arange(n)
    perm = torch.arange(n)
    for i in range(n - 1, 0, -1):
        j = torch.randint(0, i, (1,), generator=generator).item()  # [0, i)
        perm[i], perm[j] = perm[j].item(), perm[i].item()
    return perm


# ── Descriptor statistics collection ──────────────────────────────────────

def collect_descriptor_stats(
    train_dataset,
    device: torch.device,
    n_batches: int = 50,
    num_workers: int = 8,
) -> Dict[str, Dict[str, list]]:
    """Collect per-feature-dim mean/std of A4 and D4 descriptors.

    Returns:
        dict with 'a4' and 'd4' keys, each containing:
            'mean': list of floats (per dim)
            'std': list of floats (per dim)
    """
    import experiments.bias_control.gate43_scratch.gate43_scratch_training as g43
    from src.bias_control.datasets.maestro_segments import collate_segments

    loader = DataLoader(
        train_dataset, batch_size=16, shuffle=False,
        num_workers=num_workers, collate_fn=collate_segments,
        pin_memory=True,
    )

    a4_flat_accum = []  # list of [B*T, 8] tensors
    d4_valid_accum = []  # list of [n_valid, 4] tensors

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= n_batches:
                break
            audio = batch['audio'].to(device, non_blocking=True)
            midi_pitch = batch['midi_pitch'].to(device, non_blocking=True)
            midi_mask = batch['midi_mask'].to(device, non_blocking=True)

            # A4: [B, T, 8] → flatten to [B*T, 8]
            a4 = g43.compute_audio_descriptor_a4(audio, target_length=None)
            a4_flat_accum.append(a4.reshape(-1, a4.size(-1)).cpu())

            # D4: [B, N, 4] — variable N per batch, extract valid only
            d4 = g43.compute_local_interval_features(midi_pitch, midi_mask)
            valid_mask = ~midi_mask  # [B, N], True=valid
            d4_valid = d4[valid_mask]  # [n_valid, 4]
            d4_valid_accum.append(d4_valid.cpu())

    # A4: per-dim stats across all frames
    a4_flat = torch.cat(a4_flat_accum, dim=0)  # [total_frames, 8]
    a4_stats = {
        'mean': a4_flat.mean(dim=0).tolist(),
        'std': a4_flat.std(dim=0).tolist(),
    }

    # D4: per-dim stats across valid tokens
    d4_valid = torch.cat(d4_valid_accum, dim=0)  # [total_valid, 4]
    d4_stats = {
        'mean': d4_valid.mean(dim=0).tolist(),
        'std': d4_valid.std(dim=0).tolist(),
    }

    logger.info(f"Descriptor stats collected ({n_batches} batches):")
    logger.info(f"  A4: mean={[f'{v:.4f}' for v in a4_stats['mean']]}")
    logger.info(f"  A4: std={[f'{v:.4f}' for v in a4_stats['std']]}")
    logger.info(f"  D4: mean={[f'{v:.4f}' for v in d4_stats['mean']]}")
    logger.info(f"  D4: std={[f'{v:.4f}' for v in d4_stats['std']]}")
    logger.info(f"  D4 valid tokens: {d4_valid.shape[0]:,}")

    return {'a4': a4_stats, 'd4': d4_stats}


def _stats_hash(stats: dict) -> str:
    """Deterministic hash of stats dict for checkpoint verification."""
    s = json.dumps(stats, sort_keys=True)
    return hashlib.md5(s.encode()).hexdigest()[:8]


# ── Ablation wrappers ─────────────────────────────────────────────────────

def _make_ablation_wrapper(
    original_fn,
    mode: str,
    stats: Dict[str, list],
    base_seed: int,
    is_d4: bool = False,
):
    """Create a wrapper that replaces descriptor output according to mode.

    Args:
        original_fn: The original compute function.
        mode: 'real', 'zero', 'random', 'shuffled'.
        stats: Dict with 'mean' and 'std' (per-dim lists).
        base_seed: Base seed for deterministic generation.
        is_d4: If True, this wraps D4 (uses mask arg for padding).

    Returns:
        (wrapper_fn, call_count) where call_count is [int] (mutable).
    """
    call_count = [0]
    mean_t = torch.tensor(stats['mean'], dtype=torch.float32) if stats else None
    std_t = torch.tensor(stats['std'], dtype=torch.float32) if stats else None

    def wrapper(*args, **kwargs):
        result = original_fn(*args, **kwargs)

        if mode == 'real':
            call_count[0] += 1
            return result

        if mode == 'zero':
            call_count[0] += 1
            return torch.zeros_like(result)

        call_count[0] += 1
        seed = base_seed + call_count[0]
        rng = torch.Generator(device=result.device)
        rng.manual_seed(seed)

        if mode == 'random':
            # Per-dim matched noise
            m = mean_t.to(result.device)
            s = std_t.to(result.device)
            noise = torch.randn(result.shape, generator=rng, device=result.device,
                                dtype=result.dtype)
            # Broadcast: result is [B, T, D] and m/s are [D]
            noise = noise * s + m

            # Preserve D4 padding via mask (second positional arg)
            if is_d4:
                mask = args[1] if len(args) > 1 else kwargs.get('midi_mask')
                if mask is not None:
                    valid = (~mask).unsqueeze(-1).float()  # [B, N, 1]
                    noise = noise * valid
            return noise

        if mode == 'shuffled':
            B = result.size(0)
            perm = _fixed_derangement(B, rng)
            return result[perm.to(result.device)]

        return result  # fallback

    return wrapper, call_count


def apply_permanent_patches(
    mode: str,
    audio_stats: Dict[str, list],
    midi_stats: Dict[str, list],
    base_seed: int,
) -> Dict[str, list]:
    """Monkey-patch descriptor functions in the gate43 module.

    Args:
        mode: Ablation mode. If 'real', nothing is patched.
        audio_stats: Per-dim stats for A4.
        midi_stats: Per-dim stats for D4.
        base_seed: Base seed for deterministic wrappers.

    Returns:
        Dict mapping 'a4' and 'd4' to their mutable call_count lists.
    """
    import experiments.bias_control.gate43_scratch.gate43_scratch_training as g43

    call_counts = {}

    if mode == 'real':
        # No patching — use original functions
        call_counts['a4'] = [0]
        call_counts['d4'] = [0]
        return call_counts

    # A4 wrapper
    a4_wrapper, a4_cc = _make_ablation_wrapper(
        g43.compute_audio_descriptor_a4, mode, audio_stats,
        base_seed=base_seed, is_d4=False,
    )
    g43.compute_audio_descriptor_a4 = a4_wrapper
    call_counts['a4'] = a4_cc

    # D4 wrapper
    d4_wrapper, d4_cc = _make_ablation_wrapper(
        g43.compute_local_interval_features, mode, midi_stats,
        base_seed=base_seed + 1_000_000,  # offset to avoid seed collision
        is_d4=True,
    )
    g43.compute_local_interval_features = d4_wrapper
    call_counts['d4'] = d4_cc

    logger.info(f"Permanent patches applied: mode={mode}")
    return call_counts


def verify_patches(
    mode: str,
    train_loader: DataLoader,
    device: torch.device,
) -> bool:
    """Verify that patches produce the expected descriptor behavior.

    Returns True if verification passes.
    """
    import experiments.bias_control.gate43_scratch.gate43_scratch_training as g43

    batch = next(iter(train_loader))
    audio = batch['audio'].to(device, non_blocking=True)
    midi_pitch = batch['midi_pitch'].to(device, non_blocking=True)
    midi_mask = batch['midi_mask'].to(device, non_blocking=True)

    with torch.no_grad():
        a4 = g43.compute_audio_descriptor_a4(audio, target_length=None)
        d4 = g43.compute_local_interval_features(midi_pitch, midi_mask)

    ok = True

    if mode == 'zero':
        if a4.abs().max().item() > 1e-8:
            logger.error(f"VERIFY FAIL: A4 not zero (max={a4.abs().max().item():.6f})")
            ok = False
        if d4.abs().max().item() > 1e-8:
            logger.error(f"VERIFY FAIL: D4 not zero (max={d4.abs().max().item():.6f})")
            ok = False
        logger.info("  Verify zero: A4 max=%.6f, D4 max=%.6f ✓" %
                     (a4.abs().max().item(), d4.abs().max().item()))

    elif mode == 'random':
        a4_mean = a4.mean().item()
        a4_std = a4.std().item()
        d4_mean = d4.mean().item()
        d4_std = d4.std().item()
        logger.info(f"  Verify random: A4 mean={a4_mean:.4f} std={a4_std:.4f}")
        logger.info(f"  Verify random: D4 mean={d4_mean:.4f} std={d4_std:.4f}")
        # D4 padding should be zeros
        padding_mask = midi_mask.unsqueeze(-1)  # True=padding
        d4_padding = d4[padding_mask.expand_as(d4)]
        if d4_padding.numel() > 0 and d4_padding.abs().max().item() > 1e-8:
            logger.error("VERIFY FAIL: D4 padding not zeroed in random mode")
            ok = False
        else:
            logger.info("  Verify random: D4 padding correctly zeroed ✓")

    elif mode == 'shuffled':
        # Check that A4 has nonzero values (real data, just permuted)
        if a4.abs().max().item() < 1e-8:
            logger.error("VERIFY FAIL: A4 all zeros in shuffled mode")
            ok = False
        logger.info(f"  Verify shuffled: A4 max={a4.abs().max().item():.4f} ✓")

    elif mode == 'real':
        if a4.abs().max().item() < 1e-8:
            logger.error("VERIFY FAIL: A4 all zeros in real mode — patch active?")
            ok = False
        logger.info(f"  Verify real: A4 max={a4.abs().max().item():.4f}, "
                     f"D4 max={d4.abs().max().item():.4f} ✓")

    return ok


# ── Main training function ────────────────────────────────────────────────

def train_param_matched(args) -> dict:
    """Train a parameter-matched ablation arm."""
    import experiments.bias_control.gate43_scratch.gate43_scratch_training as g43
    from src.bias_control.architectures.cross_modal_model import CrossModalModel
    from src.bias_control.datasets.maestro_segments import (
        MaestroSegmentDataset, collate_segments,
    )
    from src.bias_control.training.preflight import (
        validate_training_setup, PARAM_RANGES,
    )

    mode = args.mode
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info(f"TEST 02 — PARAMETER-MATCHED ABLATION: mode={mode.upper()}")
    logger.info("=" * 60)

    # 1. Seed everything
    g43.seed_everything(args.seed)

    # 2. Create datasets
    logger.info("Creating datasets...")
    train_dataset = MaestroSegmentDataset(
        maestro_dir=args.maestro_dir, segment_len=4.0, hop=1.0, split='train',
    )
    val_dataset = MaestroSegmentDataset(
        maestro_dir=args.maestro_dir, segment_len=4.0, hop=1.0, split='validation',
    )
    logger.info(f"  Train: {len(train_dataset)} segments, Val: {len(val_dataset)} segments")

    # 3. Collect or load descriptor stats
    stats_path = output_dir / 'descriptor_stats.json'
    if stats_path.exists():
        logger.info(f"Loading cached descriptor stats from {stats_path}")
        with open(stats_path) as f:
            desc_stats = json.load(f)
    else:
        logger.info(f"Collecting descriptor stats ({args.stats_batches} batches)...")
        desc_stats = collect_descriptor_stats(
            train_dataset, device, n_batches=args.stats_batches,
            num_workers=args.num_workers,
        )
        with open(stats_path, 'w') as f:
            json.dump(desc_stats, f, indent=2)
        logger.info(f"  Stats saved to {stats_path}")

    sh = _stats_hash(desc_stats)
    logger.info(f"  Stats hash: {sh}")

    # 4. Apply permanent patches (before model creation, affects forward())
    call_counts = apply_permanent_patches(
        mode, desc_stats['a4'], desc_stats['d4'], base_seed=args.seed,
    )

    # 5. Handle resume
    start_epoch = 1
    initial_best_S = 0.0
    initial_best_epoch = 0
    resume_ckpt = None

    if args.resume:
        logger.info(f"Resuming from: {args.resume}")
        resume_ckpt = torch.load(args.resume, map_location=device)
        resume_arch = resume_ckpt.get('arch_config', {})

        # Verify mode matches
        ckpt_mode = resume_arch.get('ablation_mode')
        if ckpt_mode and ckpt_mode != mode:
            raise ValueError(
                f"Resume checkpoint ablation_mode={ckpt_mode} != --mode={mode}"
            )

        start_epoch = resume_ckpt['epoch'] + 1
        initial_best_S = resume_ckpt.get('best_S', 0.0)
        initial_best_epoch = resume_ckpt.get('epoch', 0)

        # Restore call_counts
        raw_cc = resume_arch.get('call_counts', {})
        for key in ('a4', 'd4'):
            val = raw_cc.get(key, 0)
            if isinstance(val, (list, tuple)):
                val = int(val[0])
            call_counts[key][0] = int(val)

        logger.info(f"  Restored call_counts: a4={call_counts['a4'][0]}, d4={call_counts['d4'][0]}")
        logger.info(f"  Resume from epoch {resume_ckpt['epoch']}, best_S={initial_best_S:.1%}")

        if start_epoch > args.epochs:
            raise ValueError(
                f"Resume checkpoint at epoch {resume_ckpt['epoch']} but --epochs={args.epochs}"
            )

    # 6. Create model (d4a4 architecture, from scratch, run-d freeze)
    logger.info(f"Creating model: descriptor={DESCRIPTOR}, freeze_policy={FREEZE_POLICY}")
    base_model = CrossModalModel(audio_encoder='lite', use_dann=False)
    g43.apply_freeze_policy(base_model, policy=FREEZE_POLICY)
    model = g43.create_gate42_model(DESCRIPTOR, base_model, ratio_weight=0.0)

    if resume_ckpt:
        model.load_state_dict(resume_ckpt['model_state_dict'], strict=True)
        logger.info("  Model weights restored from checkpoint")

    model = model.to(device)

    # Verify param count (preflight checks TRAINABLE params)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lo, hi = g43.GATE42_PARAM_RANGES[FREEZE_POLICY][DESCRIPTOR]
    logger.info(f"  Total params: {n_params:,} (trainable: {n_trainable:,})")
    logger.info(f"  Expected trainable range: [{lo:,}, {hi:,}]")
    if not (lo <= n_trainable <= hi):
        raise RuntimeError(
            f"Trainable param count {n_trainable:,} outside expected range [{lo:,}, {hi:,}]"
        )

    # 7. Create optimizer
    optimizer = g43.create_gate42_optimizer(
        model, DESCRIPTOR, freeze_policy=FREEZE_POLICY,
    )
    if resume_ckpt and 'optimizer_state_dict' in resume_ckpt:
        optimizer.load_state_dict(resume_ckpt['optimizer_state_dict'])
        logger.info("  Optimizer state restored")

    # 8. arch_config — mutable dict shared with checkpointing
    arch_config = {
        'gate': '5b-test02',
        'descriptor': DESCRIPTOR,
        'ablation_mode': mode,
        'from_scratch': True,
        'freeze_policy': FREEZE_POLICY,
        'ratio_weight': 0.0,
        'wrapper_base_seed': args.seed,
        'descriptor_stats_hash': sh,
        'call_counts': {
            'a4': call_counts['a4'],   # mutable [int] refs
            'd4': call_counts['d4'],
        },
    }
    if resume_ckpt:
        arch_config['resumed_from'] = args.resume
        arch_config['resumed_epoch'] = resume_ckpt['epoch']

    # 9. Preflight validation
    mode_key = f'gate42_{DESCRIPTOR}'
    PARAM_RANGES[mode_key] = g43.GATE42_PARAM_RANGES[FREEZE_POLICY][DESCRIPTOR]
    contract = g43.get_gate42_preflight_contract(DESCRIPTOR, freeze_policy=FREEZE_POLICY)
    validate_training_setup(model, optimizer, mode=mode_key, **contract)
    logger.info("  Preflight validation passed ✓")

    # 10. Verify patches on first batch
    g_dl = torch.Generator()
    g_dl.manual_seed(args.seed)
    dl_kwargs = g43._dataloader_kwargs(args.num_workers)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_segments,
        drop_last=True, generator=g_dl, worker_init_fn=g43.seed_worker,
        **dl_kwargs,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_segments,
        **dl_kwargs,
    )

    if not args.resume:
        logger.info("Verifying patches...")
        if not verify_patches(mode, train_loader, device):
            raise RuntimeError("Patch verification FAILED — aborting")
        logger.info("  Patch verification passed ✓")

    # 11. Scheduler
    total_steps = args.epochs * args.max_batches_per_epoch
    scheduler = g43.LinearWarmupCosineScheduler(
        optimizer, warmup_steps=200, total_steps=total_steps,
    )
    if resume_ckpt and 'scheduler_state_dict' in resume_ckpt:
        scheduler.load_state_dict(resume_ckpt['scheduler_state_dict'])
        logger.info(f"  Scheduler restored: step_count={scheduler.step_count}")

    # Parse structured eval epochs
    eval_epochs = args.structured_eval_epochs
    if eval_epochs is None:
        eval_epochs = [5, 10, 15, 20, 25, 28, 29, 30]

    # 12. Save config
    config = {
        'test': 'test02_param_matched',
        'mode': mode,
        'descriptor': DESCRIPTOR,
        'freeze_policy': FREEZE_POLICY,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'max_batches_per_epoch': args.max_batches_per_epoch,
        'seed': args.seed,
        'structured_eval_epochs': eval_epochs,
        'stats_hash': sh,
        'n_params': n_params,
        'n_trainable': n_trainable,
    }
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # 13. Train
    logger.info(f"\nStarting training: mode={mode}, epochs={args.epochs}, "
                f"batch_size={args.batch_size}, max_batches={args.max_batches_per_epoch}")
    logger.info(f"Structured eval epochs: {eval_epochs}")

    train_results = g43.train_loop_gate42(
        model=model, optimizer=optimizer, scheduler=scheduler,
        train_loader=train_loader, val_loader=val_loader,
        descriptor=DESCRIPTOR, arch_config=arch_config,
        output_dir=output_dir, maestro_dir=Path(args.maestro_dir),
        device=device, epochs=args.epochs,
        max_batches_per_epoch=args.max_batches_per_epoch,
        seed=args.seed, start_epoch=start_epoch,
        initial_best_S=initial_best_S,
        initial_best_epoch=initial_best_epoch,
        structured_eval_epochs=eval_epochs,
    )

    # 14. Final results
    best_ep = train_results['best_epoch']
    best_eval_path = output_dir / 'eval_per_epoch' / f'eval_epoch{best_ep}.json'
    last_eval_path = output_dir / 'eval_per_epoch' / f'eval_epoch{args.epochs}.json'

    best_eval = json.loads(best_eval_path.read_text()) if best_eval_path.exists() else None
    last_eval = json.loads(last_eval_path.read_text()) if last_eval_path.exists() else None

    final = {
        'test': 'test02_param_matched',
        'ablation_mode': mode,
        'descriptor': DESCRIPTOR,
        'freeze_policy': FREEZE_POLICY,
        'n_params': n_params,
        'n_trainable': n_trainable,
        'stats_hash': sh,
        'training': train_results,
        'evaluation_best': best_eval,
        'evaluation_final': last_eval,
        'call_counts_final': {
            'a4': call_counts['a4'][0],
            'd4': call_counts['d4'][0],
        },
    }
    with open(output_dir / 'final_results.json', 'w') as f:
        json.dump(final, f, indent=2)

    logger.info("=" * 60)
    logger.info(f"TEST 02 COMPLETE: mode={mode}")
    logger.info(f"  Best S: {train_results['best_S']:.1%} (epoch {best_ep})")
    logger.info(f"  Training time: {train_results['training_time_minutes']:.0f} min")
    logger.info(f"  Call counts: a4={call_counts['a4'][0]}, d4={call_counts['d4'][0]}")
    logger.info(f"  Output: {output_dir}")
    logger.info("=" * 60)

    return final


# ── CLI ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Gate 5B — Test 02: Parameter-Matched Ablations'
    )
    parser.add_argument('--mode', type=str, required=True,
                        choices=ABLATION_MODES,
                        help='Ablation mode: real/random/shuffled/zero')
    parser.add_argument('--maestro-dir', type=str,
                        default='data/maestro_v3/maestro-v3.0.0')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory for this arm')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--max-batches-per-epoch', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint path')
    parser.add_argument('--structured-eval-epochs', type=int, nargs='+',
                        default=None,
                        help='Epochs for structured eval (default: 5 10 15 20 25 28 29 30)')
    parser.add_argument('--stats-batches', type=int, default=50,
                        help='Number of batches for descriptor stats collection')

    args = parser.parse_args()
    train_param_matched(args)


if __name__ == '__main__':
    main()
