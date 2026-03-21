#!/usr/bin/env python3
"""E3-P5: Mixed Geometry Latent (euclidean + toroidal branches).

Eval phases (all eval-only, no retraining):
  Phase 1 (canonical):        final_results.json
  Phase 2 (checkpoint-sweep): checkpoint_sweep.json
  Phase 3 (weight-sweep):     euclidean_weight_sweep.json
  Phase 4 (ablation):         torus_ablation.json

Usage:
    # Training
    python experiments/escalon3/f5_mixed_geometry.py \
        --audio-encoder baseline --data data/escalon3/bundled \
        --scenes data/escalon3/scenes \
        --output data/escalon3/p5_mixed_flat_seed42 --epochs 50 --seed 42

    # Phase 1: canonical artefact
    python experiments/escalon3/f5_mixed_geometry.py --eval-only \
        --output data/escalon3/p5_mixed_flat_seed42

    # Phase 2: checkpoint sweep on val
    python experiments/escalon3/f5_mixed_geometry.py --eval-only --phase checkpoint-sweep \
        --output data/escalon3/p5_mixed_flat_seed42

    # All phases at once
    python experiments/escalon3/f5_mixed_geometry.py --eval-only --phase all \
        --output data/escalon3/p5_mixed_flat_seed42
"""

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.escalon3.lissajous_dataset import (
    create_lissajous_dataloaders, load_ood_dataset, load_reduced_gallery_dataset,
    LissajousDataset,
)
from src.escalon3.encoders import build_lissajous_audio_encoder, build_lissajous_image_encoder
from src.escalon3.torus_projector import MixedProjector
from src.escalon3.toroidal_vicreg import MixedVICRegLoss
from experiments.escalon3.eval_torus_escalon3 import (
    extract_mixed_embeddings, compute_torus_structural_metrics,
    torus_retrieval_score, euclidean_retrieval_score, mixed_retrieval_score,
    build_torus_pool_cache, build_torus_pool_cache_val,
    evaluate_render_ood_mixed,
    build_lissajous_index, sample_structured_pool,
)


# ============================================================
# Utilities
# ============================================================

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class LinearWarmupCosineScheduler:
    def __init__(self, optimizer, warmup_steps: int, total_steps: int):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.base_lrs = [pg['lr'] for pg in optimizer.param_groups]
        self.step_count = 0

    def step(self):
        self.step_count += 1
        if self.step_count <= self.warmup_steps:
            scale = self.step_count / max(1, self.warmup_steps)
        else:
            progress = (self.step_count - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps)
            scale = 0.5 * (1 + math.cos(math.pi * progress))
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg['lr'] = base_lr * scale

    def state_dict(self):
        return {'step_count': self.step_count, 'base_lrs': self.base_lrs}

    def load_state_dict(self, d):
        self.step_count = d['step_count']
        self.base_lrs = d['base_lrs']


def _audio_key(enc):
    return 'audio_cqt' if getattr(enc, 'input_kind', 'waveform') == 'cqt' else 'audio'


# ============================================================
# Training
# ============================================================

def train_one_epoch(audio_enc, image_enc, proj_audio, proj_image,
                    loss_fn, optimizer, scheduler, loader, device):
    audio_enc.train(); image_enc.train()
    proj_audio.train(); proj_image.train()

    total_loss = 0
    comps = {}
    n = 0
    akey = _audio_key(audio_enc)

    for batch in tqdm(loader, desc="Train", leave=False):
        audio = batch[akey].to(device)
        image = batch['image'].to(device)

        out_a = proj_audio(audio_enc(audio))   # {'euclidean': [B,128], 'torus': [B,64]}
        out_i = proj_image(image_enc(image))

        loss_dict = loss_fn(
            out_a['euclidean'], out_i['euclidean'],
            out_a['torus'], out_i['torus'],
        )

        optimizer.zero_grad()
        loss_dict['total'].backward()
        nn.utils.clip_grad_norm_(
            list(audio_enc.parameters()) + list(image_enc.parameters()) +
            list(proj_audio.parameters()) + list(proj_image.parameters()), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss_dict['total'].item()
        for k, v in loss_dict.items():
            if k != 'total' and isinstance(v, torch.Tensor):
                comps[k] = comps.get(k, 0) + v.item()
        n += 1

    avg = {k: v / max(n, 1) for k, v in comps.items()}
    avg['total'] = total_loss / max(n, 1)
    return avg


@torch.no_grad()
def quick_val_eval(audio_enc, image_enc, proj_audio, proj_image,
                   dataset, device, pool_size=128, n_queries=200, seed=42):
    """Quick validation eval returning S_euclidean, S_torus, S_mixed."""
    embs = extract_mixed_embeddings(audio_enc, image_enc, proj_audio, proj_image,
                                     dataset, device, num_workers=0)

    # Simple nearest-neighbor eval on small subset (no structured pool, just fast check)
    rng = random.Random(seed)
    N = len(dataset)
    queries = rng.sample(range(N), min(n_queries, N))

    results = {}
    for mode in ['euclidean', 'torus', 'mixed']:
        hits_a2i, hits_i2a = [], []
        for q in queries:
            pool = rng.sample([i for i in range(N) if i != q], min(pool_size, N - 1))
            pool_with_pos = pool[:pool_size - 1] + [q]
            rng.shuffle(pool_with_pos)
            pos_in_pool = pool_with_pos.index(q)

            for direction, hits in [('a2i', hits_a2i), ('i2a', hits_i2a)]:
                if direction == 'a2i':
                    if mode == 'euclidean':
                        scores = euclidean_retrieval_score(embs['audio_euc'][q], embs['image_euc'][pool_with_pos])
                    elif mode == 'torus':
                        scores = torus_retrieval_score(embs['audio_torus'][q], embs['image_torus'][pool_with_pos])
                    else:
                        scores = mixed_retrieval_score(
                            embs['audio_euc'][q], embs['audio_torus'][q],
                            embs['image_euc'][pool_with_pos], embs['image_torus'][pool_with_pos])
                else:
                    if mode == 'euclidean':
                        scores = euclidean_retrieval_score(embs['image_euc'][q], embs['audio_euc'][pool_with_pos])
                    elif mode == 'torus':
                        scores = torus_retrieval_score(embs['image_torus'][q], embs['audio_torus'][pool_with_pos])
                    else:
                        scores = mixed_retrieval_score(
                            embs['image_euc'][q], embs['image_torus'][q],
                            embs['audio_euc'][pool_with_pos], embs['audio_torus'][pool_with_pos])

                rank = (scores.argsort(descending=True) == pos_in_pool).nonzero(as_tuple=True)[0].item() + 1
                hits.append(float(rank <= 10))

        a2i = np.mean(hits_a2i)
        i2a = np.mean(hits_i2a)
        results[f'S_{mode}'] = min(a2i, i2a)

    return results


# ============================================================
# Shared evaluation helpers
# ============================================================

def _load_checkpoint(path, audio_enc, image_enc, proj_audio, proj_image, device):
    """Load checkpoint into models. Returns checkpoint dict."""
    ckpt = torch.load(path, map_location=device)
    audio_enc.load_state_dict(ckpt['audio_enc'])
    image_enc.load_state_dict(ckpt['image_enc'])
    proj_audio.load_state_dict(ckpt['proj_audio'])
    proj_image.load_state_dict(ckpt['proj_image'])
    return ckpt


def _resolve_checkpoint(args, output_dir: Path) -> Path:
    """Return explicit --checkpoint if given, else output_dir/best_model.pt."""
    return Path(args.checkpoint) if args.checkpoint else (output_dir / 'best_model.pt')


def _get_or_build_pool_cache_eval(pool_cache_dir, data_dir, seed=42):
    """Load or build pool_cache_eval.pt (test + OOD splits)."""
    cache_path = Path(pool_cache_dir) / 'pool_cache_eval.pt'
    if cache_path.exists():
        print(f"  Loading pool cache: {cache_path}")
        return torch.load(cache_path)
    print(f"  Building pool_cache_eval...")
    cache = build_torus_pool_cache(
        Path(data_dir), pool_size=128, n_queries_iid=500, n_queries_ood=500, seed=seed)
    torch.save(cache, cache_path)
    print(f"  Saved: {cache_path}")
    return cache


def _get_or_build_pool_cache_val(pool_cache_dir, data_dir, seed=42):
    """Load or build pool_cache_val.pt (val same-set only)."""
    cache_path = Path(pool_cache_dir) / 'pool_cache_val.pt'
    if cache_path.exists():
        print(f"  Loading pool cache: {cache_path}")
        return torch.load(cache_path)
    print(f"  Building pool_cache_val...")
    cache = build_torus_pool_cache_val(
        Path(data_dir), pool_size=128, n_queries=500, seed=seed)
    torch.save(cache, cache_path)
    print(f"  Saved: {cache_path}")
    return cache


def _score_split(q_embs, g_embs, pool_data,
                 modes=('euclidean', 'torus', 'mixed'),
                 directions=('a2i', 'i2a'),
                 euclidean_weight=0.5,
                 torus_perm=None):
    """Score a split using precomputed pools.

    pool_data: dict from precompute_torus_pools() with 'query_ids' and 'pools'.
    torus_perm: optional RNG for shuffling candidate torus embeddings.
    Returns: dict[direction][mode] = {'recall_at_10': float, 'n_queries': int}
             plus S_{mode} keys.
    """
    results = {}
    for direction in directions:
        results[direction] = {}
        q_key = 'audio' if direction == 'a2i' else 'image'
        c_key = 'image' if direction == 'a2i' else 'audio'

        for mode in modes:
            hits = []
            for q_idx in pool_data['query_ids']:
                q_idx = int(q_idx)
                pool = pool_data['pools'][q_idx]
                candidates = [int(c) for c in pool['candidate_indices']]
                positive_positions = [int(p) for p in pool['positive_positions']]

                c_torus_pool = g_embs[f'{c_key}_torus'][candidates]
                c_euc_pool = g_embs.get(f'{c_key}_euc')
                if c_euc_pool is not None:
                    c_euc_pool = c_euc_pool[candidates]

                # Apply torus shuffle if requested
                if torus_perm is not None and mode in ('torus', 'mixed'):
                    perm = torch.randperm(len(candidates), generator=torus_perm)
                    c_torus_pool = c_torus_pool[perm]

                if mode == 'euclidean':
                    if c_euc_pool is None:
                        continue
                    scores = euclidean_retrieval_score(
                        q_embs[f'{q_key}_euc'][q_idx], c_euc_pool)
                elif mode == 'torus':
                    scores = torus_retrieval_score(
                        q_embs[f'{q_key}_torus'][q_idx], c_torus_pool)
                else:  # mixed
                    if c_euc_pool is None:
                        continue
                    scores = mixed_retrieval_score(
                        q_embs[f'{q_key}_euc'][q_idx], q_embs[f'{q_key}_torus'][q_idx],
                        c_euc_pool, c_torus_pool,
                        euclidean_weight=euclidean_weight)

                sorted_idx = scores.argsort(descending=True).tolist()
                best_rank = min(sorted_idx.index(p) + 1 for p in positive_positions)
                hits.append(float(best_rank <= 10))

            r10 = float(np.mean(hits)) if hits else 0.0
            results[direction][mode] = {'recall_at_10': r10, 'n_queries': len(hits)}

    # S = min(a2i, i2a) per mode
    for mode in modes:
        a2i_r = results.get('a2i', {}).get(mode, {}).get('recall_at_10', 0)
        i2a_r = results.get('i2a', {}).get(mode, {}).get('recall_at_10', 0)
        results[f'S_{mode}'] = min(a2i_r, i2a_r)

    return results


# ============================================================
# Phase 1: Complete canonical artefact
# ============================================================

def run_phase1_canonical(audio_enc, image_enc, proj_audio, proj_image,
                          args, output_dir, device):
    """Phase 1: Re-eval best_model.pt with full eval suite.

    Outputs: final_results.json with iid, scale_ood, equiv_ood, render_ood_noisy,
    render_ood_thick, torus_structural_test, torus_structural_equiv_ood.
    """
    data_dir = Path(args.data)
    pool_cache_dir = Path(args.pool_cache) if args.pool_cache else data_dir

    print("\n=== PHASE 1: CANONICAL ARTEFACT ===")

    # Load checkpoint
    ckpt_path = _resolve_checkpoint(args, output_dir)
    ckpt = _load_checkpoint(ckpt_path, audio_enc, image_enc, proj_audio, proj_image, device)
    print(f"  Loaded {ckpt_path.name} (epoch {ckpt['epoch']})")

    # Read trainselect values from checkpoint if available, else fallback to re-eval
    results = {
        'config': vars(args),
        'best_epoch': ckpt.get('best_epoch', ckpt['epoch']),
        'best_val_S_mixed_trainselect': ckpt.get('best_val_S_mixed_trainselect'),
        'best_val_S_euclidean_trainselect': ckpt.get('best_val_S_euclidean_trainselect'),
        'best_val_S_torus_trainselect': ckpt.get('best_val_S_torus_trainselect'),
    }

    if results['best_val_S_mixed_trainselect'] is None:
        print("  Trainselect not in checkpoint, computing via quick_val_eval...")
        val_ds = LissajousDataset(data_dir / 'val')
        val_scores = quick_val_eval(audio_enc, image_enc, proj_audio, proj_image,
                                     val_ds, device)
        results['best_val_S_mixed_trainselect'] = val_scores['S_mixed']
        results['best_val_S_euclidean_trainselect'] = val_scores['S_euclidean']
        results['best_val_S_torus_trainselect'] = val_scores['S_torus']

    print(f"  Trainselect: S_euc={results['best_val_S_euclidean_trainselect']:.3f} "
          f"S_torus={results['best_val_S_torus_trainselect']:.3f} "
          f"S_mixed={results['best_val_S_mixed_trainselect']:.3f}")

    # Pool cache
    pool_cache = _get_or_build_pool_cache_eval(pool_cache_dir, data_dir, args.seed)

    # Extract embeddings
    print("  Extracting embeddings...")
    test_ds = LissajousDataset(data_dir / 'test')
    gallery_ds = load_reduced_gallery_dataset(data_dir)
    scale_ds = load_ood_dataset(data_dir, 'scale_ood')
    equiv_ds = load_ood_dataset(data_dir, 'equiv_ood')

    test_embs = extract_mixed_embeddings(audio_enc, image_enc, proj_audio, proj_image,
                                          test_ds, device, num_workers=0)
    gallery_embs = extract_mixed_embeddings(audio_enc, image_enc, proj_audio, proj_image,
                                             gallery_ds, device, num_workers=0)
    scale_embs = extract_mixed_embeddings(audio_enc, image_enc, proj_audio, proj_image,
                                           scale_ds, device, num_workers=0)
    equiv_embs = extract_mixed_embeddings(audio_enc, image_enc, proj_audio, proj_image,
                                           equiv_ds, device, num_workers=0)

    # Retrieval per split (iid + scale_ood + equiv_ood), all 3 modes
    split_cfg = {
        'iid':       {'q_embs': test_embs,  'g_embs': test_embs},
        'scale_ood': {'q_embs': scale_embs, 'g_embs': gallery_embs},
        'equiv_ood': {'q_embs': equiv_embs, 'g_embs': gallery_embs},
    }

    for split_name, cfg in split_cfg.items():
        pool_data = pool_cache['splits'][split_name]
        results[split_name] = _score_split(
            cfg['q_embs'], cfg['g_embs'], pool_data)
        print(f"  {split_name}: S_euc={results[split_name]['S_euclidean']:.3f} "
              f"S_torus={results[split_name]['S_torus']:.3f} "
              f"S_mixed={results[split_name]['S_mixed']:.3f}")

    # Render OOD (noisy + thick), i2a only, 3 modes
    scene_dir = args.scenes
    iid_pool_cache = {
        int(q): pool_cache['splits']['iid']['pools'][q]
        for q in pool_cache['splits']['iid']['query_ids']
    } if pool_cache else None

    for style in ['noisy', 'thick']:
        print(f"  Render OOD ({style})...")
        render_res = evaluate_render_ood_mixed(
            audio_enc, image_enc, proj_audio, proj_image,
            test_ds, scene_dir, device,
            render_style=style,
            pool_cache_iid=iid_pool_cache,
            euclidean_weight=0.5,
            n_queries=500, seed=args.seed)
        results[f'render_ood_{style}'] = render_res
        for mode in ['euclidean', 'torus', 'mixed']:
            r10 = render_res.get(mode, {}).get('recall_at_10', 0)
            print(f"    {mode}: R@10={r10:.3f}")

    # Torus structural on test AND equiv_ood (Phase 5 requirement)
    print("  Torus structural metrics...")
    results['torus_structural_test'] = compute_torus_structural_metrics(
        test_embs['audio_torus'], test_ds)
    results['torus_structural_equiv_ood'] = compute_torus_structural_metrics(
        equiv_embs['audio_torus'], equiv_ds)
    print(f"  Test sil_ratio={results['torus_structural_test']['torus_silhouette_ratio']:.3f}")
    print(f"  Equiv_ood sil_ratio={results['torus_structural_equiv_ood']['torus_silhouette_ratio']:.3f}")

    # Save
    with open(output_dir / 'final_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n  Phase 1 saved: {output_dir / 'final_results.json'}")
    print(f"  Best epoch: {ckpt['epoch']}")
    return results


# ============================================================
# Phase 2: Checkpoint sweep on VALIDATION only
# ============================================================

def run_phase2_checkpoint_sweep(audio_enc, image_enc, proj_audio, proj_image,
                                 args, output_dir, device):
    """Phase 2: Evaluate checkpoints on val same-set only (no test/OOD = no leakage)."""
    data_dir = Path(args.data)
    pool_cache_dir = Path(args.pool_cache) if args.pool_cache else data_dir

    print("\n=== PHASE 2: CHECKPOINT SWEEP (val only) ===")

    val_ds = LissajousDataset(data_dir / 'val')
    pool_cache = _get_or_build_pool_cache_val(pool_cache_dir, data_dir, args.seed)
    val_pool = pool_cache['splits']['val']

    checkpoint_epochs = [5, 10, 20, 30, 40, 50]
    sweep_results = {'config': vars(args), 'checkpoints': []}

    best_val_S_mixed_structured = 0.0
    best_epoch_structured = -1

    for ep in checkpoint_epochs:
        ckpt_path = output_dir / f'checkpoint_e{ep:02d}.pt'
        if not ckpt_path.exists():
            print(f"  Skipping epoch {ep} (checkpoint not found)")
            continue

        _load_checkpoint(ckpt_path, audio_enc, image_enc, proj_audio, proj_image, device)
        embs = extract_mixed_embeddings(audio_enc, image_enc, proj_audio, proj_image,
                                         val_ds, device, num_workers=0)

        scores = _score_split(embs, embs, val_pool)
        entry = {
            'epoch': ep,
            'S_euclidean': scores['S_euclidean'],
            'S_torus': scores['S_torus'],
            'S_mixed': scores['S_mixed'],
            'detail': {d: {m: scores[d][m] for m in ['euclidean', 'torus', 'mixed']}
                       for d in ['a2i', 'i2a']},
        }
        sweep_results['checkpoints'].append(entry)

        if scores['S_mixed'] > best_val_S_mixed_structured:
            best_val_S_mixed_structured = scores['S_mixed']
            best_epoch_structured = ep

        print(f"  E{ep:02d}: S_euc={scores['S_euclidean']:.3f} "
              f"S_torus={scores['S_torus']:.3f} S_mixed={scores['S_mixed']:.3f}")

    sweep_results['best_epoch_structured_val'] = best_epoch_structured
    sweep_results['best_val_S_mixed_structured_val'] = best_val_S_mixed_structured

    with open(output_dir / 'checkpoint_sweep.json', 'w') as f:
        json.dump(sweep_results, f, indent=2, default=str)

    print(f"\n  Phase 2 saved: {output_dir / 'checkpoint_sweep.json'}")
    print(f"  Best structured val: epoch {best_epoch_structured} "
          f"(S_mixed={best_val_S_mixed_structured:.3f})")
    return sweep_results


# ============================================================
# Phase 3: Euclidean weight sweep (post-hoc)
# ============================================================

def run_phase3_weight_sweep(audio_enc, image_enc, proj_audio, proj_image,
                             args, output_dir, device):
    """Phase 3: Sweep euclidean_weight on best checkpoint."""
    data_dir = Path(args.data)
    pool_cache_dir = Path(args.pool_cache) if args.pool_cache else data_dir

    print("\n=== PHASE 3: EUCLIDEAN WEIGHT SWEEP ===")

    ckpt_path = _resolve_checkpoint(args, output_dir)
    ckpt = _load_checkpoint(ckpt_path, audio_enc, image_enc, proj_audio, proj_image, device)

    # Pool cache and embeddings
    pool_cache = _get_or_build_pool_cache_eval(pool_cache_dir, data_dir, args.seed)

    test_ds = LissajousDataset(data_dir / 'test')
    gallery_ds = load_reduced_gallery_dataset(data_dir)
    scale_ds = load_ood_dataset(data_dir, 'scale_ood')
    equiv_ds = load_ood_dataset(data_dir, 'equiv_ood')

    print("  Extracting embeddings...")
    test_embs = extract_mixed_embeddings(audio_enc, image_enc, proj_audio, proj_image,
                                          test_ds, device, num_workers=0)
    gallery_embs = extract_mixed_embeddings(audio_enc, image_enc, proj_audio, proj_image,
                                             gallery_ds, device, num_workers=0)
    scale_embs = extract_mixed_embeddings(audio_enc, image_enc, proj_audio, proj_image,
                                           scale_ds, device, num_workers=0)
    equiv_embs = extract_mixed_embeddings(audio_enc, image_enc, proj_audio, proj_image,
                                           equiv_ds, device, num_workers=0)

    split_cfg = {
        'iid':       {'q_embs': test_embs,  'g_embs': test_embs},
        'scale_ood': {'q_embs': scale_embs, 'g_embs': gallery_embs},
        'equiv_ood': {'q_embs': equiv_embs, 'g_embs': gallery_embs},
    }

    weights = [0.0, 0.25, 0.5, 0.75, 1.0]
    sweep_results = {'config': vars(args), 'best_epoch': ckpt['epoch'], 'weights': []}

    for w in weights:
        entry = {'euclidean_weight': w}
        print(f"  weight={w:.2f}:")
        for split_name, cfg in split_cfg.items():
            pool_data = pool_cache['splits'][split_name]
            # Only score 'mixed' mode (euc/torus modes are weight-independent)
            scores = _score_split(
                cfg['q_embs'], cfg['g_embs'], pool_data,
                modes=('mixed',), euclidean_weight=w)
            entry[split_name] = {'S_mixed': scores['S_mixed'],
                                 'a2i_R10': scores['a2i']['mixed']['recall_at_10'],
                                 'i2a_R10': scores['i2a']['mixed']['recall_at_10']}
            print(f"    {split_name}: S_mixed={scores['S_mixed']:.3f}")
        sweep_results['weights'].append(entry)

    with open(output_dir / 'euclidean_weight_sweep.json', 'w') as f:
        json.dump(sweep_results, f, indent=2, default=str)

    print(f"\n  Phase 3 saved: {output_dir / 'euclidean_weight_sweep.json'}")
    return sweep_results


# ============================================================
# Phase 4: Torus ablation
# ============================================================

def run_phase4_ablation(audio_enc, image_enc, proj_audio, proj_image,
                         args, output_dir, device):
    """Phase 4: Torus ablation with 4 conditions.

    Conditions:
      euclidean_only: euclidean_weight=1.0
      torus_only:     euclidean_weight=0.0
      mixed:          euclidean_weight=0.5
      torus_shuffle:  euclidean_weight=0.5, candidate torus embeddings permuted
    """
    data_dir = Path(args.data)
    pool_cache_dir = Path(args.pool_cache) if args.pool_cache else data_dir

    print("\n=== PHASE 4: TORUS ABLATION ===")

    ckpt_path = _resolve_checkpoint(args, output_dir)
    ckpt = _load_checkpoint(ckpt_path, audio_enc, image_enc, proj_audio, proj_image, device)

    pool_cache = _get_or_build_pool_cache_eval(pool_cache_dir, data_dir, args.seed)

    test_ds = LissajousDataset(data_dir / 'test')
    gallery_ds = load_reduced_gallery_dataset(data_dir)
    scale_ds = load_ood_dataset(data_dir, 'scale_ood')
    equiv_ds = load_ood_dataset(data_dir, 'equiv_ood')

    print("  Extracting embeddings...")
    test_embs = extract_mixed_embeddings(audio_enc, image_enc, proj_audio, proj_image,
                                          test_ds, device, num_workers=0)
    gallery_embs = extract_mixed_embeddings(audio_enc, image_enc, proj_audio, proj_image,
                                             gallery_ds, device, num_workers=0)
    scale_embs = extract_mixed_embeddings(audio_enc, image_enc, proj_audio, proj_image,
                                           scale_ds, device, num_workers=0)
    equiv_embs = extract_mixed_embeddings(audio_enc, image_enc, proj_audio, proj_image,
                                           equiv_ds, device, num_workers=0)

    split_cfg = {
        'iid':       {'q_embs': test_embs,  'g_embs': test_embs},
        'scale_ood': {'q_embs': scale_embs, 'g_embs': gallery_embs},
        'equiv_ood': {'q_embs': equiv_embs, 'g_embs': gallery_embs},
    }

    conditions = {
        'euclidean_only': {'euclidean_weight': 1.0, 'modes': ('euclidean',), 'shuffle': False},
        'torus_only':     {'euclidean_weight': 0.0, 'modes': ('torus',),     'shuffle': False},
        'mixed':          {'euclidean_weight': 0.5, 'modes': ('mixed',),     'shuffle': False},
        'torus_shuffle':  {'euclidean_weight': 0.5, 'modes': ('mixed',),     'shuffle': True},
    }

    ablation_results = {'config': vars(args), 'best_epoch': ckpt['epoch'], 'conditions': {}}

    for cond_name, cond in conditions.items():
        print(f"  Condition: {cond_name}")
        ablation_results['conditions'][cond_name] = {}

        # Fixed permutation generator per condition (seed=42, shared between flat/cqtshift)
        torus_perm = torch.Generator().manual_seed(42) if cond['shuffle'] else None

        for split_name, cfg in split_cfg.items():
            pool_data = pool_cache['splits'][split_name]
            scores = _score_split(
                cfg['q_embs'], cfg['g_embs'], pool_data,
                modes=cond['modes'],
                euclidean_weight=cond['euclidean_weight'],
                torus_perm=torus_perm)

            mode_key = cond['modes'][0]
            ablation_results['conditions'][cond_name][split_name] = {
                f'S_{mode_key}': scores[f'S_{mode_key}'],
                'a2i_R10': scores['a2i'][mode_key]['recall_at_10'],
                'i2a_R10': scores['i2a'][mode_key]['recall_at_10'],
            }
            print(f"    {split_name}: S={scores[f'S_{mode_key}']:.3f}")

    # render_ood for each condition (i2a only)
    scene_dir = args.scenes
    iid_pool_cache = {
        int(q): pool_cache['splits']['iid']['pools'][q]
        for q in pool_cache['splits']['iid']['query_ids']
    }

    for cond_name, cond in conditions.items():
        torus_perm = torch.Generator().manual_seed(42) if cond['shuffle'] else None
        for style in ['noisy', 'thick']:
            render_res = evaluate_render_ood_mixed(
                audio_enc, image_enc, proj_audio, proj_image,
                test_ds, scene_dir, device,
                render_style=style,
                pool_cache_iid=iid_pool_cache,
                euclidean_weight=cond['euclidean_weight'],
                torus_perm=torus_perm,
                n_queries=500, seed=args.seed)

            mode_key = cond['modes'][0]
            r10 = render_res.get(mode_key, {}).get('recall_at_10', 0)
            ablation_results['conditions'][cond_name][f'render_ood_{style}'] = {
                f'R10_{mode_key}': r10}
            print(f"  {cond_name} render_{style}: R@10={r10:.3f}")

    with open(output_dir / 'torus_ablation.json', 'w') as f:
        json.dump(ablation_results, f, indent=2, default=str)

    print(f"\n  Phase 4 saved: {output_dir / 'torus_ablation.json'}")
    return ablation_results


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="E3-P5: Mixed Geometry Latent")
    parser.add_argument("--data", default="data/escalon3/bundled")
    parser.add_argument("--scenes", default="data/escalon3/scenes")
    parser.add_argument("--audio-encoder", default="baseline",
                        choices=["baseline", "cqtshift"])
    parser.add_argument("--image-encoder", default="baseline")
    parser.add_argument("--output", default=None)
    parser.add_argument("--euc-dim", type=int, default=128)
    parser.add_argument("--n-angles", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--lambda-inv", type=float, default=25.0)
    parser.add_argument("--lambda-var", type=float, default=25.0)
    parser.add_argument("--lambda-cov", type=float, default=1.0)
    parser.add_argument("--lambda-t-inv", type=float, default=10.0)
    parser.add_argument("--lambda-t-var", type=float, default=10.0)
    parser.add_argument("--lambda-t-cov", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr-enc", type=float, default=5e-4)
    parser.add_argument("--lr-proj", type=float, default=1e-3)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval-epochs", nargs='+', type=int,
                        default=[5, 10, 20, 30, 40, 50])
    parser.add_argument("--num-workers", type=int, default=0)
    # New eval-only args
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training, run eval phases")
    parser.add_argument("--phase", default="canonical",
                        choices=["canonical", "checkpoint-sweep", "weight-sweep",
                                 "ablation", "all"],
                        help="Which eval phase to run (requires --eval-only)")
    parser.add_argument("--pool-cache", default=None,
                        help="Directory for shared pool caches (default: data dir)")
    parser.add_argument("--euclidean-weight", type=float, default=0.5,
                        help="1.0=pure euclidean, 0.0=pure torus (for manual override)")
    parser.add_argument("--checkpoint", default=None,
                        help="Explicit checkpoint path for eval-only phases")
    args = parser.parse_args()

    if args.output is None:
        args.output = f"data/escalon3/p5_mixed_{args.audio_encoder}_seed{args.seed}"

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Models
    audio_enc = build_lissajous_audio_encoder(args.audio_encoder, 256).to(device)
    image_enc = build_lissajous_image_encoder(args.image_encoder, 256).to(device)
    proj_audio = MixedProjector(256, args.euc_dim, args.n_angles, args.hidden_dim).to(device)
    proj_image = MixedProjector(256, args.euc_dim, args.n_angles, args.hidden_dim).to(device)

    n_params = sum(p.numel() for m in [audio_enc, image_enc, proj_audio, proj_image]
                   for p in m.parameters())
    print(f"Audio encoder: {args.audio_encoder} | Geometry: mixed (euc={args.euc_dim}, torus={args.n_angles})")
    print(f"Total params: {n_params:,}")

    # ── Eval-only mode ──
    if args.eval_only:
        phases = (['canonical', 'checkpoint-sweep', 'weight-sweep', 'ablation']
                  if args.phase == 'all' else [args.phase])

        for phase in phases:
            if phase == 'canonical':
                run_phase1_canonical(audio_enc, image_enc, proj_audio, proj_image,
                                      args, output_dir, device)
            elif phase == 'checkpoint-sweep':
                run_phase2_checkpoint_sweep(audio_enc, image_enc, proj_audio, proj_image,
                                             args, output_dir, device)
            elif phase == 'weight-sweep':
                run_phase3_weight_sweep(audio_enc, image_enc, proj_audio, proj_image,
                                         args, output_dir, device)
            elif phase == 'ablation':
                run_phase4_ablation(audio_enc, image_enc, proj_audio, proj_image,
                                     args, output_dir, device)
        return

    # ── Training mode ──
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    loss_fn = MixedVICRegLoss(
        args.lambda_inv, args.lambda_var, args.lambda_cov,
        args.lambda_t_inv, args.lambda_t_var, args.lambda_t_cov)

    # Data
    train_loader, val_loader, test_loader = create_lissajous_dataloaders(
        args.data, args.batch_size, num_workers=args.num_workers)
    print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")

    # Optimizer
    optimizer = AdamW([
        {'params': audio_enc.parameters(), 'lr': args.lr_enc},
        {'params': image_enc.parameters(), 'lr': args.lr_enc},
        {'params': proj_audio.parameters(), 'lr': args.lr_proj},
        {'params': proj_image.parameters(), 'lr': args.lr_proj},
    ], weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = LinearWarmupCosineScheduler(optimizer, args.warmup_steps, total_steps)

    # Training loop
    history = []
    best_val_S_mixed = 0.0

    for epoch in range(1, args.epochs + 1):
        train_m = train_one_epoch(audio_enc, image_enc, proj_audio, proj_image,
                                   loss_fn, optimizer, scheduler, train_loader, device)
        log = {'epoch': epoch, **train_m}

        euc_t = train_m.get('euc_total', 0)
        tor_t = train_m.get('torus_total', 0)
        print(f"E{epoch:02d} | total={train_m['total']:.3f} euc={euc_t:.3f} torus={tor_t:.3f}")

        # Checkpoint
        ckpt = {
            'epoch': epoch,
            'audio_enc': audio_enc.state_dict(),
            'image_enc': image_enc.state_dict(),
            'proj_audio': proj_audio.state_dict(),
            'proj_image': proj_image.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'config': vars(args),
        }
        torch.save(ckpt, output_dir / f'checkpoint_e{epoch:02d}.pt')

        # Periodic eval on val
        if epoch in args.eval_epochs:
            print(f"  Val eval at epoch {epoch}...")
            val_scores = quick_val_eval(audio_enc, image_enc, proj_audio, proj_image,
                                         val_loader.dataset, device)
            log.update(val_scores)
            print(f"  S_euc={val_scores['S_euclidean']:.3f} "
                  f"S_torus={val_scores['S_torus']:.3f} "
                  f"S_mixed={val_scores['S_mixed']:.3f}")

            if val_scores['S_mixed'] > best_val_S_mixed:
                best_val_S_mixed = val_scores['S_mixed']
                ckpt['best_val_S_mixed_trainselect'] = val_scores['S_mixed']
                ckpt['best_val_S_euclidean_trainselect'] = val_scores['S_euclidean']
                ckpt['best_val_S_torus_trainselect'] = val_scores['S_torus']
                ckpt['best_epoch'] = epoch
                torch.save(ckpt, output_dir / 'best_model.pt')
                with open(output_dir / 'best_checkpoint_meta.json', 'w') as f_meta:
                    json.dump({
                        'best_epoch': epoch,
                        'best_val_S_mixed_trainselect': val_scores['S_mixed'],
                        'best_val_S_euclidean_trainselect': val_scores['S_euclidean'],
                        'best_val_S_torus_trainselect': val_scores['S_torus'],
                    }, f_meta, indent=2)
                print(f"  New best val_S_mixed={best_val_S_mixed:.3f}")

        history.append(log)

    # Post-training: run Phase 1
    run_phase1_canonical(audio_enc, image_enc, proj_audio, proj_image,
                          args, output_dir, device)


if __name__ == "__main__":
    main()
