#!/usr/bin/env python3
"""
Gate 8: Descriptor-Conditioned Projection Heads.
(Promoted from Gate 5A C1 — trazabilidad preservada.)

Tests whether conditioning the projection head (the diagnosed information
bottleneck) with descriptor vectors improves cross-modal retrieval.

Evidence motivating this gate:
  - Test 11 Pre-Proj: MIDI projection 512->256 destroys 88% of conditioning info
  - Gate 7.1a: stronger frozen audio encoder doesn't improve retrieval
  Both point to projection/MIDI-side as the operational bottleneck.

Approach: Replace standard ProjectionHead with ConditionedProjectionHead
that uses FiLM modulation (zero-init -> identity at start, grows from there).
No encoder changes — only the projection heads are modified.

Audio conditioning uses compute_audio_band_energy (mean log-magnitude per
A4 band) — NOT the z-scored A4 temporal deltas (which have mean=0 by
construction and would produce degenerate conditioning).

Arms:
  a4r-ctrl     — Control (standard a4r, reproduced in this script)
  a4r-pca      — Conditioned audio projection (band energy -> audio)
  a4r-pcm      — Conditioned MIDI projection (D4 -> midi)
  a4r-pcd      — Both projections conditioned (band energy + D4)
  a4r-pcd-zero — Both conditioned but cond=zeros (overhead control)

Usage:
    python experiments/bias_control/gate5a_proj_cond.py \\
        --arm a4r-pcm \\
        --maestro-dir data/maestro_v3/maestro-v3.0.0 \\
        --output data/gate8_results/a4r-pcm \\
        --epochs 30 --batch-size 16 --device cuda --seed 42
"""

import argparse
import json
import logging
import math
import random
import sys
import time
import types
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.bias_control.architectures.cross_modal_model import CrossModalModel
from src.bias_control.encoders.projection import (
    ProjectionHead,
    ConditionedProjectionHead,
)
from src.bias_control.training.preflight import DriftSentinel
from src.bias_control.datasets.maestro_segments import (
    MaestroSegmentDataset,
    collate_segments,
)
from src.bias_control.audio_descriptors import compute_audio_band_energy
from src.bias_control.ratio_descriptors import compute_local_interval_features

from experiments.bias_control.gate43_scratch.gate43_scratch_training import (
    Gate42AudioReverseCrossAttModel,
    create_gate42_model,
    create_gate42_optimizer,
    apply_freeze_policy,
    load_foundation,
    LinearWarmupCosineScheduler,
    seed_everything,
    seed_worker,
    save_gate42_checkpoint,
    run_structured_eval,
    quick_val_eval,
    train_loop_gate42,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Arm configurations
# ---------------------------------------------------------------------------

ARM_CONFIGS = {
    'a4r-ctrl': {
        'proj_cond_audio': False,
        'proj_cond_midi': False,
        'zero_cond': False,
        'description': 'Control: standard a4r, same pipeline',
    },
    'a4r-pca': {
        'proj_cond_audio': True,
        'proj_cond_midi': False,
        'zero_cond': False,
        'description': 'Conditioned audio projection (A4→audio)',
    },
    'a4r-pcm': {
        'proj_cond_audio': False,
        'proj_cond_midi': True,
        'zero_cond': False,
        'description': 'Conditioned MIDI projection (D4→midi)',
    },
    'a4r-pcd': {
        'proj_cond_audio': True,
        'proj_cond_midi': True,
        'zero_cond': False,
        'description': 'Both projections conditioned (A4+D4)',
    },
    'a4r-pcd-zero': {
        'proj_cond_audio': True,
        'proj_cond_midi': True,
        'zero_cond': True,
        'description': 'Both conditioned, cond=zeros (overhead control)',
    },
}

# FiLM cond_dim: A4 descriptor has 8 dims, D4 has 4 dims
AUDIO_COND_DIM = 8
MIDI_COND_DIM = 4


# ---------------------------------------------------------------------------
# In-place model modification: setup_conditioned_projections
# ---------------------------------------------------------------------------

def setup_conditioned_projections(
    model: nn.Module,
    proj_cond_audio: bool = False,
    proj_cond_midi: bool = False,
    zero_cond: bool = False,
) -> nn.Module:
    """
    Modify a4r model in-place with conditioned projection heads.

    - Replaces ProjectionHead -> ConditionedProjectionHead (copying weights)
    - Monkey-patches model.forward to set/clear descriptor caches
    - No wrapper -> optimizer, state_dict, compute_total_loss all work unchanged
    """
    base = model.base_model

    # 1. Replace projections in-place (copying ALL weights+buffers)
    if proj_cond_audio:
        old = base.audio_projection
        assert isinstance(old, ProjectionHead), (
            f"Expected ProjectionHead, got {type(old)}"
        )
        base.audio_projection = ConditionedProjectionHead.from_projection_head(
            old, cond_dim=AUDIO_COND_DIM,
        )
        logger.info(f"  Replaced audio_projection with ConditionedProjectionHead "
                     f"(cond_dim={AUDIO_COND_DIM})")

    if proj_cond_midi:
        old = base.midi_projection
        assert isinstance(old, ProjectionHead), (
            f"Expected ProjectionHead, got {type(old)}"
        )
        base.midi_projection = ConditionedProjectionHead.from_projection_head(
            old, cond_dim=MIDI_COND_DIM,
        )
        logger.info(f"  Replaced midi_projection with ConditionedProjectionHead "
                     f"(cond_dim={MIDI_COND_DIM})")

    # 2. Store config as model attribute
    model._g5a_config = {
        'proj_cond_audio': proj_cond_audio,
        'proj_cond_midi': proj_cond_midi,
        'zero_cond': zero_cond,
    }

    # 3. Monkey-patch forward to set/clear caches
    #    compute_total_loss calls self.forward() -> finds patched method
    original_forward = model.__class__.forward  # unbound class method

    def patched_forward(self, audio, midi_pitch, midi_velocity, midi_duration,
                        midi_mask=None):
        try:
            # Set audio projection cache
            if proj_cond_audio:
                if zero_cond:
                    cond_a = torch.zeros(
                        audio.shape[0], AUDIO_COND_DIM, device=audio.device,
                    )
                else:
                    with torch.no_grad():
                        cond_a = compute_audio_band_energy(audio)  # [B, 8]
                self.base_model.audio_projection._cached_cond = cond_a

            # Set MIDI projection cache
            if proj_cond_midi:
                if zero_cond:
                    cond_m = torch.zeros(
                        midi_pitch.shape[0], MIDI_COND_DIM,
                        device=midi_pitch.device,
                    )
                else:
                    with torch.no_grad():
                        d4 = compute_local_interval_features(midi_pitch, midi_mask)
                        if midi_mask is not None:
                            valid = (~midi_mask).unsqueeze(-1).float()
                            cond_m = (d4 * valid).sum(1) / valid.sum(1).clamp(min=1)
                        else:
                            cond_m = d4.mean(dim=1)
                self.base_model.midi_projection._cached_cond = cond_m

            # Call original forward
            return original_forward(
                self, audio, midi_pitch, midi_velocity, midi_duration, midi_mask,
            )
        finally:
            # Always clear caches
            if proj_cond_audio:
                self.base_model.audio_projection._cached_cond = None
            if proj_cond_midi:
                self.base_model.midi_projection._cached_cond = None

    model.forward = types.MethodType(patched_forward, model)
    return model


# ---------------------------------------------------------------------------
# Model reconstruction from checkpoint
# ---------------------------------------------------------------------------

def reconstruct_model(checkpoint_path: str, device: torch.device) -> nn.Module:
    """Reconstruct a Gate 5A conditioned model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    arch_cfg = ckpt['arch_config']

    # 1. Create base a4r model
    base_model = CrossModalModel(audio_encoder='lite', use_dann=False)
    model = create_gate42_model('a4r', base_model)

    # 2. Apply conditioned projections if flagged
    pca = arch_cfg.get('proj_cond_audio', False)
    pcm = arch_cfg.get('proj_cond_midi', False)
    zc = arch_cfg.get('zero_cond', False)
    if pca or pcm:
        setup_conditioned_projections(
            model, proj_cond_audio=pca, proj_cond_midi=pcm, zero_cond=zc,
        )

    # 3. Load weights
    model.load_state_dict(ckpt['model_state_dict'], strict=True)
    model = model.to(device)
    model.eval()

    logger.info(f"Reconstructed model: arm={arch_cfg.get('arm')}, "
                f"epoch={ckpt.get('epoch')}, best_S={ckpt.get('best_S', 0):.1%}")
    return model


# ---------------------------------------------------------------------------
# Dataloader helper
# ---------------------------------------------------------------------------

def _dataloader_kwargs(num_workers: int) -> dict:
    if num_workers > 0:
        return {'pin_memory': True, 'prefetch_factor': 2}
    return {}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def run_train(args):
    """Train a Gate 5A arm."""
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)

    arm_cfg = ARM_CONFIGS[args.arm]

    logger.info("=" * 60)
    logger.info(f"GATE 8 — ARM: {args.arm.upper()}")
    logger.info(f"  {arm_cfg['description']}")
    logger.info("=" * 60)

    # --- Create base a4r model from scratch ---
    base_model = CrossModalModel(audio_encoder='lite', use_dann=False)
    freeze_policy = 'run-d'
    apply_freeze_policy(base_model, policy=freeze_policy)
    model = create_gate42_model('a4r', base_model)

    # --- Apply conditioned projections (in-place) ---
    if arm_cfg['proj_cond_audio'] or arm_cfg['proj_cond_midi']:
        setup_conditioned_projections(
            model,
            proj_cond_audio=arm_cfg['proj_cond_audio'],
            proj_cond_midi=arm_cfg['proj_cond_midi'],
            zero_cond=arm_cfg['zero_cond'],
        )

    model = model.to(device)

    # Count params
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  Parameters: total={total_params:,}, trainable={trainable_params:,}")
    film_params = 0
    if arm_cfg['proj_cond_audio']:
        film_params += sum(
            p.numel() for p in model.base_model.audio_projection.film_generators.parameters()
        )
    if arm_cfg['proj_cond_midi']:
        film_params += sum(
            p.numel() for p in model.base_model.midi_projection.film_generators.parameters()
        )
    if film_params > 0:
        logger.info(f"  FiLM params: {film_params:,} ({film_params/total_params:.2%} of total)")

    # --- Create optimizer (reuses a4r param groups; FiLM params auto-included) ---
    optimizer = create_gate42_optimizer(
        model, 'a4r',
        freeze_policy=freeze_policy,
        lr_audio=args.lr_audio,
        lr_audio_low=args.lr_audio_low,
        lr_midi=args.lr_midi,
        lr_proj=args.lr_proj,
        lr_ratio=args.lr_ratio,
    )

    # --- arch_config with Gate 8 flags (promoted from Gate 5A C1) ---
    arch_config = {
        'gate': '8',
        'descriptor': 'a4r',
        'arm': args.arm,
        'freeze_policy': freeze_policy,
        'from_scratch': True,
        'proj_cond_audio': arm_cfg['proj_cond_audio'],
        'proj_cond_midi': arm_cfg['proj_cond_midi'],
        'zero_cond': arm_cfg.get('zero_cond', False),
        'film_params': film_params,
        'total_params': total_params,
        'trainable_params': trainable_params,
    }

    # --- DataLoaders ---
    g = torch.Generator()
    g.manual_seed(args.seed)

    train_dataset = MaestroSegmentDataset(
        maestro_dir=args.maestro_dir, segment_len=4.0, hop=1.0, split='train',
    )
    val_dataset = MaestroSegmentDataset(
        maestro_dir=args.maestro_dir, segment_len=4.0, hop=1.0, split='validation',
    )

    dl_kwargs = _dataloader_kwargs(args.num_workers)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_segments,
        drop_last=True, generator=g, worker_init_fn=seed_worker,
        **dl_kwargs,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_segments,
        **dl_kwargs,
    )

    # --- Scheduler ---
    total_steps = args.epochs * args.max_batches_per_epoch
    scheduler = LinearWarmupCosineScheduler(
        optimizer, warmup_steps=args.warmup_steps, total_steps=total_steps,
    )

    logger.info(f"  LR schedule: warmup {args.warmup_steps} steps -> "
                f"cosine decay, total={total_steps} steps")

    # --- Save config ---
    config = {**vars(args), **arch_config}
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2, default=str)

    # --- Conditioning sanity check (first batch) ---
    if arm_cfg['proj_cond_audio'] or arm_cfg['proj_cond_midi']:
        logger.info("  [sanity] Probing conditioning vectors on first batch...")
        model.eval()
        with torch.no_grad():
            batch = next(iter(train_loader))
            audio = batch['audio'].to(device)
            if arm_cfg['proj_cond_audio'] and not arm_cfg['zero_cond']:
                cond_a = compute_audio_band_energy(audio)
                logger.info(f"  [sanity] cond_audio: shape={list(cond_a.shape)}, "
                            f"mean={cond_a.mean():.4f}, std={cond_a.std():.4f}, "
                            f"min={cond_a.min():.4f}, max={cond_a.max():.4f}")
            if arm_cfg['proj_cond_midi'] and not arm_cfg['zero_cond']:
                midi_pitch = batch['midi_pitch'].to(device)
                midi_mask = batch.get('midi_mask')
                if midi_mask is not None:
                    midi_mask = midi_mask.to(device)
                d4 = compute_local_interval_features(midi_pitch, midi_mask)
                if midi_mask is not None:
                    valid = (~midi_mask).unsqueeze(-1).float()
                    cond_m = (d4 * valid).sum(1) / valid.sum(1).clamp(min=1)
                else:
                    cond_m = d4.mean(dim=1)
                logger.info(f"  [sanity] cond_midi:  shape={list(cond_m.shape)}, "
                            f"mean={cond_m.mean():.4f}, std={cond_m.std():.4f}, "
                            f"min={cond_m.min():.4f}, max={cond_m.max():.4f}")
        model.train()

    # --- Train ---
    structured_eval_epochs = args.structured_eval_epochs
    if structured_eval_epochs is None:
        structured_eval_epochs = [5, 10, 15, 20, 25, 28, 29, 30]

    train_results = train_loop_gate42(
        model=model, optimizer=optimizer, scheduler=scheduler,
        train_loader=train_loader, val_loader=val_loader,
        descriptor='a4r', arch_config=arch_config,
        output_dir=output_dir, maestro_dir=Path(args.maestro_dir),
        device=device, epochs=args.epochs,
        max_batches_per_epoch=args.max_batches_per_epoch,
        max_val_batches=args.max_val_batches,
        seed=args.seed, embed_batch_size=args.embed_batch_size,
        structured_eval_epochs=structured_eval_epochs,
    )

    # --- Final results ---
    best_ep = train_results['best_epoch']
    best_eval_path = output_dir / 'eval_per_epoch' / f'eval_epoch{best_ep}.json'
    last_eval_path = output_dir / 'eval_per_epoch' / f'eval_epoch{args.epochs}.json'

    best_eval = json.loads(best_eval_path.read_text()) if best_eval_path.exists() else None
    last_eval = json.loads(last_eval_path.read_text()) if last_eval_path.exists() else None

    final = {
        'gate': '8',
        'arm': args.arm,
        'descriptor': 'a4r',
        'training': train_results,
        'evaluation_best': best_eval,
        'evaluation_final': last_eval,
        'param_count_total': total_params,
        'param_count_trainable': trainable_params,
        'film_params': film_params,
        'seed': args.seed,
        'epochs': args.epochs,
    }
    with open(output_dir / 'final_results.json', 'w') as f:
        json.dump(final, f, indent=2)

    logger.info("=" * 60)
    logger.info(f"GATE 8 — {args.arm.upper()} COMPLETE")
    logger.info(f"  Best S={train_results['best_S']:.1%} (epoch {best_ep})")
    logger.info(f"  Training time: {train_results['training_time_minutes']:.1f} min")
    logger.info("=" * 60)

    return final


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def run_evaluate(args):
    """Evaluate a Gate 5A checkpoint."""
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    model = reconstruct_model(args.checkpoint, device)

    results = run_structured_eval(
        model=model, maestro_dir=Path(args.maestro_dir), device=device,
        seed=args.seed,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    S = results['gate_metrics']['S']
    logger.info(f"Evaluation complete: S={S:.1%}")
    return results


# ---------------------------------------------------------------------------
# Verification tests
# ---------------------------------------------------------------------------

def run_verify(args):
    """Run all verification tests for Gate 5A."""
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    passed = 0
    total = 0

    # Test 1: Import
    total += 1
    try:
        from src.bias_control.encoders.projection import ConditionedProjectionHead
        logger.info("[PASS] Test 1: Import ConditionedProjectionHead")
        passed += 1
    except Exception as e:
        logger.error(f"[FAIL] Test 1: Import — {e}")

    # Test 2: Identity (from_projection_head copies ALL weights)
    total += 1
    try:
        orig = ProjectionHead(1024, 512, 256, n_layers=3)
        cond = ConditionedProjectionHead.from_projection_head(orig, cond_dim=8)
        x = torch.randn(4, 1024)
        orig.eval()
        cond.eval()
        out_orig = orig(x)
        out_cond = cond(x, cond=torch.zeros(4, 8))
        assert torch.allclose(out_orig, out_cond, atol=1e-6), (
            f"Identity broken! max diff={torch.abs(out_orig - out_cond).max():.2e}"
        )
        logger.info("[PASS] Test 2: Identity (from_projection_head)")
        passed += 1
    except Exception as e:
        logger.error(f"[FAIL] Test 2: Identity — {e}")

    # Test 3: FiLM zero-init
    total += 1
    try:
        for gen in cond.film_generators:
            out = gen(torch.randn(4, 8))
            assert torch.allclose(out, torch.zeros_like(out), atol=1e-7), (
                f"FiLM not zero-init! max={out.abs().max():.2e}"
            )
        logger.info("[PASS] Test 3: FiLM zero-init")
        passed += 1
    except Exception as e:
        logger.error(f"[FAIL] Test 3: FiLM zero-init — {e}")

    # Test 4: Monkey-patch (setup_conditioned_projections + compute_total_loss)
    total += 1
    try:
        base_model = CrossModalModel(audio_encoder='lite', use_dann=False)
        model = create_gate42_model('a4r', base_model)

        setup_conditioned_projections(
            model, proj_cond_audio=True, proj_cond_midi=True, zero_cond=False,
        )
        # .to(device) AFTER setup — ConditionedProjectionHead created on CPU
        model = model.to(device)

        # Verify compute_total_loss works
        B = 2
        audio = torch.randn(B, 96000, device=device)
        midi_pitch = torch.randint(21, 108, (B, 128), device=device)
        midi_velocity = torch.randint(32, 127, (B, 128), device=device)
        midi_duration = torch.randint(1, 16, (B, 128), device=device)
        midi_mask = torch.zeros(B, 128, dtype=torch.bool, device=device)

        loss, metrics = model.compute_total_loss(
            audio, midi_pitch, midi_velocity, midi_duration, midi_mask,
        )
        assert loss.requires_grad, "Loss has no grad!"
        assert 'vicreg_loss' in metrics
        logger.info(f"[PASS] Test 4: Monkey-patch (loss={loss.item():.4f})")
        passed += 1
    except Exception as e:
        logger.error(f"[FAIL] Test 4: Monkey-patch — {e}")
        import traceback
        traceback.print_exc()

    # Test 5: Param count
    total += 1
    try:
        total_p = sum(p.numel() for p in model.parameters())
        # a4r ~78.6M, + FiLM ~265K
        assert total_p > 78_000_000, f"Too few params: {total_p:,}"
        assert total_p < 80_000_000, f"Too many params: {total_p:,}"
        film_p = sum(
            p.numel()
            for p in model.base_model.audio_projection.film_generators.parameters()
        ) + sum(
            p.numel()
            for p in model.base_model.midi_projection.film_generators.parameters()
        )
        logger.info(f"[PASS] Test 5: Param count (total={total_p:,}, film={film_p:,})")
        passed += 1
    except Exception as e:
        logger.error(f"[FAIL] Test 5: Param count — {e}")

    # Test 6: Cache cleanup after exception
    total += 1
    try:
        try:
            bad_audio = torch.randn(2, 100, device=device)  # wrong shape
            model(bad_audio, midi_pitch, midi_velocity, midi_duration, midi_mask)
        except Exception:
            pass
        assert model.base_model.audio_projection._cached_cond is None, "Cache not cleared!"
        assert model.base_model.midi_projection._cached_cond is None, "Cache not cleared!"
        logger.info("[PASS] Test 6: Cache cleanup after exception")
        passed += 1
    except Exception as e:
        logger.error(f"[FAIL] Test 6: Cache cleanup — {e}")

    # Test 7: Optimizer creation
    total += 1
    try:
        opt = create_gate42_optimizer(model, 'a4r', freeze_policy='run-d')
        # Verify FiLM params are in optimizer
        opt_param_ids = set()
        for pg in opt.param_groups:
            for p in pg['params']:
                opt_param_ids.add(id(p))
        for p in model.base_model.audio_projection.film_generators.parameters():
            assert id(p) in opt_param_ids, "Audio FiLM param not in optimizer!"
        for p in model.base_model.midi_projection.film_generators.parameters():
            assert id(p) in opt_param_ids, "MIDI FiLM param not in optimizer!"
        logger.info("[PASS] Test 7: Optimizer includes FiLM params")
        passed += 1
    except Exception as e:
        logger.error(f"[FAIL] Test 7: Optimizer — {e}")

    # Test 8: Gradient flow through FiLM
    total += 1
    try:
        model.train()
        opt.zero_grad()
        audio = torch.randn(2, 96000, device=device)
        loss, _ = model.compute_total_loss(
            audio, midi_pitch, midi_velocity, midi_duration, midi_mask,
        )
        loss.backward()
        film_grads = []
        for gen in model.base_model.audio_projection.film_generators:
            for p in gen.parameters():
                if p.grad is not None:
                    film_grads.append(p.grad.abs().mean().item())
        assert len(film_grads) > 0, "No FiLM gradients!"
        logger.info(f"[PASS] Test 8: Gradient flow (mean FiLM grad={np.mean(film_grads):.2e})")
        passed += 1
    except Exception as e:
        logger.error(f"[FAIL] Test 8: Gradient flow — {e}")

    logger.info("=" * 60)
    logger.info(f"Verification: {passed}/{total} tests passed")
    logger.info("=" * 60)

    if passed < total:
        sys.exit(1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Gate 5A: Descriptor-Conditioned Projection Heads"
    )
    parser.add_argument(
        '--mode', type=str, default='train',
        choices=['train', 'evaluate', 'verify'],
        help='Execution mode',
    )
    parser.add_argument(
        '--arm', type=str, default='a4r-pcd',
        choices=list(ARM_CONFIGS.keys()),
        help='Experimental arm',
    )
    parser.add_argument(
        '--checkpoint', type=str, default=None,
        help='Checkpoint path (for evaluate mode)',
    )
    parser.add_argument(
        '--output', type=str, required=True,
        help='Output directory (train) or output file (evaluate)',
    )
    parser.add_argument(
        '--maestro-dir', type=str, default='data/maestro_v3/maestro-v3.0.0',
    )

    # Training params
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--max-batches-per-epoch', type=int, default=1000)
    parser.add_argument('--max-val-batches', type=int, default=846)
    parser.add_argument('--embed-batch-size', type=int, default=16)
    parser.add_argument('--warmup-steps', type=int, default=200)

    # Learning rates (match a4r baseline)
    parser.add_argument('--lr-audio', type=float, default=1e-5)
    parser.add_argument('--lr-audio-low', type=float, default=5e-6)
    parser.add_argument('--lr-midi', type=float, default=5e-5)
    parser.add_argument('--lr-proj', type=float, default=1e-4)
    parser.add_argument('--lr-ratio', type=float, default=5e-4)

    # Eval schedule
    parser.add_argument(
        '--structured-eval-epochs', type=int, nargs='+', default=None,
        help='Epochs for structured eval (default: 5,10,15,20,25,28,29,30)',
    )

    # Misc
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    if args.mode == 'train':
        run_train(args)
    elif args.mode == 'evaluate':
        if not args.checkpoint:
            parser.error("--checkpoint is required for evaluate mode")
        run_evaluate(args)
    elif args.mode == 'verify':
        run_verify(args)


if __name__ == '__main__':
    main()
