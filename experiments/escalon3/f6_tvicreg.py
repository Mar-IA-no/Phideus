#!/usr/bin/env python3
"""E3-P6: Full T-VICReg (toroidal latent, no euclidean branch).

Tests the strong geometric hypothesis: is the full latent better organized
when trained entirely in toroidal geometry?

Usage:
    python experiments/escalon3/f6_tvicreg.py \
        --audio-encoder baseline \
        --data data/escalon3/bundled \
        --output data/escalon3/p6_tvicreg_flat_seed42 \
        --epochs 50 --seed 42
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
    create_lissajous_dataloaders, LissajousDataset,
    load_ood_dataset, load_reduced_gallery_dataset,
)
from src.escalon3.encoders import build_lissajous_audio_encoder, build_lissajous_image_encoder
from src.escalon3.torus_projector import TorusProjector
from src.escalon3.toroidal_vicreg import TVICRegLoss
from experiments.escalon3.eval_torus_escalon3 import (
    extract_mixed_embeddings, compute_torus_structural_metrics,
    torus_retrieval_score, geodesic_distance_torus,
    evaluate_render_ood_mixed,
    build_torus_pool_cache, build_torus_pool_cache_val,
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

        theta_a = proj_audio(audio_enc(audio))   # [B, n_angles] angles
        theta_i = proj_image(image_enc(image))

        loss_dict = loss_fn(theta_a, theta_i)

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
def quick_val_geodesic(audio_enc, image_enc, proj_audio, proj_image,
                       dataset, device, pool_size=128, n_queries=200, seed=42):
    """Quick val eval: S_geodesic (geodesic nearest neighbor)."""
    embs = extract_mixed_embeddings(audio_enc, image_enc, proj_audio, proj_image,
                                     dataset, device, num_workers=0)
    rng = random.Random(seed)
    N = len(dataset)
    queries = rng.sample(range(N), min(n_queries, N))

    hits_a2i, hits_i2a = [], []
    for q in queries:
        pool = rng.sample([i for i in range(N) if i != q], min(pool_size, N - 1))
        pool_with_pos = pool[:pool_size - 1] + [q]
        rng.shuffle(pool_with_pos)
        pos_in_pool = pool_with_pos.index(q)

        # a2i
        scores = torus_retrieval_score(embs['audio_torus'][q], embs['image_torus'][pool_with_pos])
        rank = (scores.argsort(descending=True) == pos_in_pool).nonzero(as_tuple=True)[0].item() + 1
        hits_a2i.append(float(rank <= 10))

        # i2a
        scores = torus_retrieval_score(embs['image_torus'][q], embs['audio_torus'][pool_with_pos])
        rank = (scores.argsort(descending=True) == pos_in_pool).nonzero(as_tuple=True)[0].item() + 1
        hits_i2a.append(float(rank <= 10))

    return {'S_geodesic': min(np.mean(hits_a2i), np.mean(hits_i2a))}


# ============================================================
# Full eval (comparable to P5 Phase 1, torus-only mode)
# ============================================================

def _get_or_build_pool_cache_eval(pool_cache_dir, data_dir, seed=42):
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


def _score_split_torus(q_embs, g_embs, pool_data, directions=('a2i', 'i2a')):
    """Score a split using torus-only retrieval (geodesic)."""
    results = {}
    for direction in directions:
        q_key = 'audio' if direction == 'a2i' else 'image'
        c_key = 'image' if direction == 'a2i' else 'audio'
        hits = []
        for q_idx in pool_data['query_ids']:
            q_idx = int(q_idx)
            pool = pool_data['pools'][q_idx]
            candidates = [int(c) for c in pool['candidate_indices']]
            positive_positions = [int(p) for p in pool['positive_positions']]

            scores = torus_retrieval_score(
                q_embs[f'{q_key}_torus'][q_idx],
                g_embs[f'{c_key}_torus'][candidates])
            sorted_idx = scores.argsort(descending=True).tolist()
            best_rank = min(sorted_idx.index(p) + 1 for p in positive_positions)
            hits.append(float(best_rank <= 10))

        r10 = float(np.mean(hits)) if hits else 0.0
        results[direction] = {'recall_at_10': r10, 'n_queries': len(hits)}

    a2i_r = results.get('a2i', {}).get('recall_at_10', 0)
    i2a_r = results.get('i2a', {}).get('recall_at_10', 0)
    results['S_torus'] = min(a2i_r, i2a_r)
    return results


def run_p6_full_eval(audio_enc, image_enc, proj_audio, proj_image,
                      args, output_dir, device, history=None):
    """Full P6 eval: iid + OOD + render_ood + structural. Torus-only."""
    data_dir = Path(args.data)
    pool_cache_dir = Path(args.pool_cache) if args.pool_cache else data_dir

    print("\n=== P6 FULL EVALUATION ===")

    ckpt_path = Path(args.checkpoint) if args.checkpoint else (output_dir / 'best_model.pt')
    ckpt = torch.load(ckpt_path, map_location=device)
    audio_enc.load_state_dict(ckpt['audio_enc'])
    image_enc.load_state_dict(ckpt['image_enc'])
    proj_audio.load_state_dict(ckpt['proj_audio'])
    proj_image.load_state_dict(ckpt['proj_image'])
    print(f"  Loaded {ckpt_path.name} (epoch {ckpt['epoch']})")

    # Val trainselect
    val_ds = LissajousDataset(data_dir / 'val')
    val_scores = quick_val_geodesic(audio_enc, image_enc, proj_audio, proj_image,
                                     val_ds, device)
    results = {
        'config': vars(args),
        'history': history or [],
        'best_epoch': ckpt['epoch'],
        'best_val_S_geodesic_trainselect': val_scores['S_geodesic'],
    }
    print(f"  Trainselect: S_geodesic={val_scores['S_geodesic']:.3f}")

    # Pool cache + embeddings
    pool_cache = _get_or_build_pool_cache_eval(pool_cache_dir, data_dir, args.seed)

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

    # Retrieval per split
    split_cfg = {
        'iid':       {'q_embs': test_embs,  'g_embs': test_embs},
        'scale_ood': {'q_embs': scale_embs, 'g_embs': gallery_embs},
        'equiv_ood': {'q_embs': equiv_embs, 'g_embs': gallery_embs},
    }

    for split_name, cfg in split_cfg.items():
        pool_data = pool_cache['splits'][split_name]
        results[split_name] = _score_split_torus(cfg['q_embs'], cfg['g_embs'], pool_data)
        print(f"  {split_name}: S_torus={results[split_name]['S_torus']:.3f}")

    # Render OOD (i2a only, torus mode)
    scene_dir = args.scenes
    iid_pool_cache = {
        int(q): pool_cache['splits']['iid']['pools'][q]
        for q in pool_cache['splits']['iid']['query_ids']
    }

    for style in ['noisy', 'thick']:
        render_res = evaluate_render_ood_mixed(
            audio_enc, image_enc, proj_audio, proj_image,
            test_ds, scene_dir, device,
            render_style=style,
            pool_cache_iid=iid_pool_cache,
            n_queries=500, seed=args.seed)
        results[f'render_ood_{style}'] = render_res
        r10 = render_res.get('torus', {}).get('recall_at_10', 0)
        print(f"  render_ood_{style}: R@10={r10:.3f}")

    # Torus structural on test + equiv_ood
    results['torus_structural_test'] = compute_torus_structural_metrics(
        test_embs['audio_torus'], test_ds)
    results['torus_structural_equiv_ood'] = compute_torus_structural_metrics(
        equiv_embs['audio_torus'], equiv_ds)
    print(f"  Structural test sil={results['torus_structural_test']['torus_silhouette_ratio']:.3f}")
    print(f"  Structural equiv_ood sil={results['torus_structural_equiv_ood']['torus_silhouette_ratio']:.3f}")

    with open(output_dir / 'final_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n  Saved: {output_dir / 'final_results.json'}")
    print(f"  Best epoch: {ckpt['epoch']}")
    return results


# ============================================================
# Phase 2: Checkpoint sweep on VALIDATION only
# ============================================================

def _get_or_build_pool_cache_val(pool_cache_dir, data_dir, seed=42):
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


def run_p6_checkpoint_sweep(audio_enc, image_enc, proj_audio, proj_image,
                             args, output_dir, device):
    """Phase 2: Evaluate checkpoints on val same-set only. Torus geodesic."""
    data_dir = Path(args.data)
    pool_cache_dir = Path(args.pool_cache) if args.pool_cache else data_dir

    print("\n=== P6 CHECKPOINT SWEEP (val only) ===")

    val_ds = LissajousDataset(data_dir / 'val')
    pool_cache = _get_or_build_pool_cache_val(pool_cache_dir, data_dir, args.seed)
    val_pool = pool_cache['splits']['val']

    checkpoint_epochs = [5, 10, 20, 30, 40, 50]
    sweep_results = {'config': vars(args), 'checkpoints': []}

    best_val_S_torus_structured = 0.0
    best_epoch_structured = -1

    for ep in checkpoint_epochs:
        ckpt_path = output_dir / f'checkpoint_e{ep:02d}.pt'
        if not ckpt_path.exists():
            print(f"  Skipping epoch {ep} (checkpoint not found)")
            continue

        ckpt = torch.load(ckpt_path, map_location=device)
        audio_enc.load_state_dict(ckpt['audio_enc'])
        image_enc.load_state_dict(ckpt['image_enc'])
        proj_audio.load_state_dict(ckpt['proj_audio'])
        proj_image.load_state_dict(ckpt['proj_image'])

        embs = extract_mixed_embeddings(audio_enc, image_enc, proj_audio, proj_image,
                                         val_ds, device, num_workers=0)

        scores = _score_split_torus(embs, embs, val_pool)
        entry = {
            'epoch': ep,
            'S_torus': scores['S_torus'],
            'a2i': scores['a2i'],
            'i2a': scores['i2a'],
        }
        sweep_results['checkpoints'].append(entry)

        if scores['S_torus'] > best_val_S_torus_structured:
            best_val_S_torus_structured = scores['S_torus']
            best_epoch_structured = ep

        print(f"  E{ep:02d}: S_torus={scores['S_torus']:.3f} "
              f"(a2i={scores['a2i']['recall_at_10']:.3f} "
              f"i2a={scores['i2a']['recall_at_10']:.3f})")

    sweep_results['best_epoch_structured_val'] = best_epoch_structured
    sweep_results['best_val_S_torus_structured_val'] = best_val_S_torus_structured

    with open(output_dir / 'checkpoint_sweep.json', 'w') as f:
        json.dump(sweep_results, f, indent=2, default=str)

    print(f"\n  Saved: {output_dir / 'checkpoint_sweep.json'}")
    print(f"  Best structured val: epoch {best_epoch_structured} "
          f"(S_torus={best_val_S_torus_structured:.3f})")
    return sweep_results


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="E3-P6: Full T-VICReg")
    parser.add_argument("--data", default="data/escalon3/bundled")
    parser.add_argument("--scenes", default="data/escalon3/scenes")
    parser.add_argument("--audio-encoder", default="baseline",
                        choices=["baseline", "cqtshift"])
    parser.add_argument("--image-encoder", default="baseline")
    parser.add_argument("--output", default=None)
    parser.add_argument("--n-angles", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=512)
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
    parser.add_argument("--pool-cache", default=None)
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training, run eval phases")
    parser.add_argument("--phase", default="canonical",
                        choices=["canonical", "checkpoint-sweep"],
                        help="Which eval phase to run (requires --eval-only)")
    parser.add_argument("--checkpoint", default=None,
                        help="Explicit checkpoint path for eval-only")
    args = parser.parse_args()

    if args.output is None:
        args.output = f"data/escalon3/p6_tvicreg_{args.audio_encoder}_seed{args.seed}"

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Models
    audio_enc = build_lissajous_audio_encoder(args.audio_encoder, 256).to(device)
    image_enc = build_lissajous_image_encoder(args.image_encoder, 256).to(device)
    proj_audio = TorusProjector(256, args.n_angles, args.hidden_dim).to(device)
    proj_image = TorusProjector(256, args.n_angles, args.hidden_dim).to(device)

    n_params = sum(p.numel() for m in [audio_enc, image_enc, proj_audio, proj_image]
                   for p in m.parameters())
    print(f"Audio encoder: {args.audio_encoder} | Geometry: full toroidal ({args.n_angles} angles)")
    print(f"Total params: {n_params:,}")

    # Eval-only mode
    if args.eval_only:
        if args.phase == 'checkpoint-sweep':
            run_p6_checkpoint_sweep(audio_enc, image_enc, proj_audio, proj_image,
                                     args, output_dir, device)
        else:  # canonical
            run_p6_full_eval(audio_enc, image_enc, proj_audio, proj_image,
                              args, output_dir, device)
        return

    loss_fn = TVICRegLoss(args.lambda_t_inv, args.lambda_t_var, args.lambda_t_cov)

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

    # Training
    history = []
    best_val_S_geodesic = 0.0

    for epoch in range(1, args.epochs + 1):
        train_m = train_one_epoch(audio_enc, image_enc, proj_audio, proj_image,
                                   loss_fn, optimizer, scheduler, train_loader, device)
        log = {'epoch': epoch, **train_m}

        inv = train_m.get('chordal_invariance', 0)
        var = train_m.get('circular_variance', 0)
        cov = train_m.get('circular_covariance', 0)
        print(f"E{epoch:02d} | total={train_m['total']:.3f} inv={inv:.4f} var={var:.4f} cov={cov:.4f}")

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

        if epoch in args.eval_epochs:
            print(f"  Val eval at epoch {epoch}...")
            val_scores = quick_val_geodesic(audio_enc, image_enc, proj_audio, proj_image,
                                             val_loader.dataset, device)
            log.update(val_scores)
            print(f"  S_geodesic={val_scores['S_geodesic']:.3f}")

            if val_scores['S_geodesic'] > best_val_S_geodesic:
                best_val_S_geodesic = val_scores['S_geodesic']
                torch.save(ckpt, output_dir / 'best_model.pt')
                print(f"  New best val_S_geodesic={best_val_S_geodesic:.3f}")

        history.append(log)

    # Final eval
    run_p6_full_eval(audio_enc, image_enc, proj_audio, proj_image,
                      args, output_dir, device, history=history)


if __name__ == "__main__":
    main()
