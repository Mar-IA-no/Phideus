#!/usr/bin/env python3
"""E3 Toroidal Evaluation: geodesic retrieval, torus probes, structural metrics.

Used by f5_mixed_geometry.py and f6_tvicreg.py.

Key distinctions:
- Training loss uses chordal invariance (1 - cos(Δθ)) — smooth, differentiable.
- Retrieval and structural metrics use geodesic distance (wrap-aware min(|Δ|, 2π-|Δ|)).
- Structural metrics computed on sin/cos embedding to avoid 0↔2π seam artifacts.
"""

import json
import math
import random
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.escalon3.lissajous_dataset import (
    LissajousDataset, collate_lissajous, load_ood_dataset, load_reduced_gallery_dataset,
)
from experiments.escalon3.eval_escalon3 import (
    build_lissajous_index, sample_structured_pool,
    find_positives_scale_ood, find_positives_equiv_ood,
    _build_key_index, _build_equiv_key_index,
    summarize_positive_coverage,
)


# ============================================================
# Geodesic distance (for retrieval and metrics, NOT training)
# ============================================================

def wrap_distance(theta_a: torch.Tensor, theta_b: torch.Tensor) -> torch.Tensor:
    """Wrap-aware angular distance per dimension. Returns [B, N] or [N]."""
    diff = torch.abs(theta_a - theta_b)
    return torch.minimum(diff, 2 * math.pi - diff)


def geodesic_distance_torus(theta_a: torch.Tensor, theta_b: torch.Tensor) -> torch.Tensor:
    """Mean geodesic distance over angular dimensions. Scalar or [B]."""
    return wrap_distance(theta_a, theta_b).mean(dim=-1)


# ============================================================
# Retrieval scores
# ============================================================

def torus_retrieval_score(query_theta: torch.Tensor, cand_theta: torch.Tensor) -> torch.Tensor:
    """Score = negative mean geodesic distance. Higher = closer.
    query: [N], cand: [M, N] → [M] scores.
    """
    d = wrap_distance(query_theta.unsqueeze(0), cand_theta)  # [M, N]
    return -d.mean(dim=-1)  # [M]


def euclidean_retrieval_score(query_euc: torch.Tensor, cand_euc: torch.Tensor) -> torch.Tensor:
    """Cosine similarity for euclidean branch. query: [D], cand: [M, D] → [M]."""
    return F.cosine_similarity(query_euc.unsqueeze(0), cand_euc, dim=-1)


def mixed_retrieval_score(query_euc: torch.Tensor, query_theta: torch.Tensor,
                          cand_euc: torch.Tensor, cand_theta: torch.Tensor,
                          euclidean_weight: float = 0.5) -> torch.Tensor:
    """Normalized mixed score: w * d_euc_norm + (1-w) * d_torus_norm.
    d_euc_norm = (1 - cosine) / 2 ∈ [0, 1]
    d_torus_norm = mean(wrap_dist²) / π² ∈ [0, 1]
    score = -d_mix (higher = closer).
    euclidean_weight: 1.0 = pure euclidean, 0.0 = pure torus.
    """
    cos_sim = F.cosine_similarity(query_euc.unsqueeze(0), cand_euc, dim=-1)
    d_euc_norm = (1 - cos_sim) / 2  # [M]

    wd = wrap_distance(query_theta.unsqueeze(0), cand_theta)  # [M, N]
    d_torus_norm = (wd ** 2).mean(dim=-1) / (math.pi ** 2)  # [M]

    d_mix = euclidean_weight * d_euc_norm + (1 - euclidean_weight) * d_torus_norm
    return -d_mix


# ============================================================
# Torus Probe Engine (torus-native, NOT S^(D-1))
# ============================================================

class TorusProbeEngine:
    """Torus-native probe: theta_probe(t) = wrap(theta_query + t * alpha).

    Alpha is added uniformly to ALL angular dimensions (Kronecker sequence on T^N).
    """

    def __init__(self, k_walk: int = 32):
        self.k_walk = k_walk

    def generate_walk(self, query_theta: torch.Tensor, alpha: float) -> torch.Tensor:
        """Generate K probe points on T^N.
        query_theta: [N] angles. Returns [K, N] probes.
        """
        steps = torch.arange(1, self.k_walk + 1, device=query_theta.device, dtype=query_theta.dtype)
        # theta_probe_i(t) = query_i + t * alpha, wrapped to (-π, π]
        probes = query_theta.unsqueeze(0) + steps.unsqueeze(1) * alpha  # [K, N]
        # Wrap to (-π, π]
        probes = (probes + math.pi) % (2 * math.pi) - math.pi
        return probes

    def probe_scores(self, query_theta: torch.Tensor, cand_theta: torch.Tensor,
                     alpha: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Score candidates using torus probe walk.
        Returns: (aggregated_scores [M], walk_top1_indices [K]).
        """
        probes = self.generate_walk(query_theta, alpha)  # [K, N]
        # Distance from each probe to each candidate
        # probes: [K, 1, N], cand: [1, M, N]
        d = wrap_distance(probes.unsqueeze(1), cand_theta.unsqueeze(0))  # [K, M, N]
        sim = -d.mean(dim=-1)  # [K, M] negative mean geodesic

        agg_scores = sim.max(dim=0).values  # [M] best probe per candidate
        walk_top1 = sim.argmax(dim=1)  # [K] best candidate per step
        return agg_scores, walk_top1

    def probe_mixed_scores(self, query_euc: torch.Tensor, query_theta: torch.Tensor,
                           cand_euc: torch.Tensor, cand_theta: torch.Tensor,
                           alpha: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """For P5: probe on torus branch, euclidean fixed.
        score = -[0.5 * d_euc_norm(fixed) + 0.5 * d_torus_norm(probed)]
        """
        # Fixed euclidean component
        cos_sim = F.cosine_similarity(query_euc.unsqueeze(0), cand_euc, dim=-1)  # [M]
        d_euc_norm = (1 - cos_sim) / 2  # [M]

        probes = self.generate_walk(query_theta, alpha)  # [K, N]
        # Torus distance from each probe to each candidate
        d = wrap_distance(probes.unsqueeze(1), cand_theta.unsqueeze(0))  # [K, M, N]
        d_torus_per_probe = (d ** 2).mean(dim=-1) / (math.pi ** 2)  # [K, M]

        # Mixed score per probe
        d_mix = 0.5 * d_euc_norm.unsqueeze(0) + 0.5 * d_torus_per_probe  # [K, M]
        sim = -d_mix

        agg_scores = sim.max(dim=0).values  # [M]
        walk_top1 = sim.argmax(dim=1)  # [K]
        return agg_scores, walk_top1


# ============================================================
# Walk metrics (per direction — in torus, "direction" = single walk)
# ============================================================

def walk_metrics(visited_class_ids: Sequence[int], pool_class_ids: Sequence[int],
                 max_depth: int = 32) -> Dict[str, float]:
    """Compute activation metrics from a single walk trace."""
    if not visited_class_ids:
        return {'locking_selectivity': 0.0, 'coverage_uniformity': 0.0,
                'relocking_depth': float(max_depth + 1), 'basin_exposure': 0.0}

    arr = np.asarray(visited_class_ids)
    _, counts = np.unique(arr, return_counts=True)
    dominant = counts.max()
    distinct_visited = len(np.unique(arr))
    distinct_pool = len(np.unique(np.asarray(pool_class_ids)))

    # Normalized entropy
    probs = counts / counts.sum()
    entropy = -(probs * np.log(probs + 1e-8)).sum()
    max_ent = math.log(max(len(np.unique(arr)), 2))
    norm_entropy = float(entropy / max_ent) if max_ent > 0 else 0.0

    # Relocking depth: first step with 3 consecutive same class
    relocking = float(max_depth + 1)
    for i in range(len(arr) - 2):
        if arr[i] == arr[i + 1] == arr[i + 2]:
            relocking = float(i + 1)
            break

    return {
        'locking_selectivity': float(dominant / len(arr)),
        'coverage_uniformity': norm_entropy,
        'relocking_depth': relocking,
        'basin_exposure': float(distinct_visited / max(distinct_pool, 1)),
    }


# ============================================================
# Structural metrics (geodesic-aware, sin/cos representation)
# ============================================================

def compute_torus_structural_metrics(theta_embs: torch.Tensor,
                                      dataset: LissajousDataset) -> Dict[str, float]:
    """Compute torus-specific structural metrics.
    Uses sin/cos representation to avoid seam artifacts at 0↔2π.
    """
    from sklearn.metrics import silhouette_score

    ratio_ids = np.array(dataset.ratio_id)
    equiv_ids = np.array(dataset.equiv_id)

    # Convert angles to sin/cos for silhouette (avoids seam)
    sincos = torch.cat([torch.sin(theta_embs), torch.cos(theta_embs)], dim=-1).numpy()  # [N, 2*n_angles]

    results = {}
    unique_ratios = np.unique(ratio_ids)

    if len(unique_ratios) > 1:
        # Silhouette on sin/cos with euclidean metric (equivalent to geodesic on angles)
        results['torus_silhouette_ratio'] = float(silhouette_score(
            sincos, ratio_ids, metric='euclidean',
            sample_size=min(1000, len(sincos))))
    else:
        results['torus_silhouette_ratio'] = 0.0

    unique_equiv = np.unique(equiv_ids)
    if len(unique_equiv) > 1:
        results['torus_silhouette_equiv'] = float(silhouette_score(
            sincos, equiv_ids, metric='euclidean',
            sample_size=min(1000, len(sincos))))
    else:
        results['torus_silhouette_equiv'] = 0.0

    # Mean geodesic distance for positive pairs (same ratio) vs negative
    # Sample 500 pairs each
    rng = np.random.RandomState(42)
    n_sample = min(500, len(theta_embs) // 2)
    pos_dists, neg_dists = [], []

    for _ in range(n_sample):
        i = rng.randint(0, len(theta_embs))
        # Find a same-ratio partner
        same_ratio = np.where(ratio_ids == ratio_ids[i])[0]
        if len(same_ratio) > 1:
            j = rng.choice(same_ratio[same_ratio != i])
            pos_dists.append(geodesic_distance_torus(theta_embs[i], theta_embs[j]).item())
        # Random negative
        j_neg = rng.randint(0, len(theta_embs))
        neg_dists.append(geodesic_distance_torus(theta_embs[i], theta_embs[j_neg]).item())

    results['mean_geodesic_pos'] = float(np.mean(pos_dists)) if pos_dists else 0.0
    results['mean_geodesic_neg'] = float(np.mean(neg_dists)) if neg_dists else 0.0

    # Angular uniformity: mean resultant length across all dimensions (lower = more uniform)
    cos_mean = torch.cos(theta_embs).mean(dim=0)
    sin_mean = torch.sin(theta_embs).mean(dim=0)
    R_bar = (cos_mean ** 2 + sin_mean ** 2).sqrt()
    results['angular_uniformity'] = float(1.0 - R_bar.mean().item())  # 1 - R̄: higher = more uniform

    return results


# ============================================================
# Pool cache for toroidal evaluation
# ============================================================

def precompute_torus_pools(
    split_name: str, query_ds, gallery_ds,
    pool_size: int, n_queries: int, seed: int,
    positive_finder=None,
) -> Dict:
    """Precompute fixed pools per query. Same as P4 pattern."""
    rng = random.Random(seed)
    gallery_index = build_lissajous_index(gallery_ds)
    query_ids = rng.sample(range(len(query_ds)), min(n_queries, len(query_ds)))
    pools = {}

    for q_idx in query_ids:
        if split_name == "iid":
            candidates, pos_pos = sample_structured_pool(q_idx, gallery_index, pool_size, rng)
            pool = {
                'positive_positions': [int(pos_pos)],
                'candidate_indices': [int(c) for c in candidates],
                'candidate_ratio_ids': [int(gallery_ds.ratio_id[c]) for c in candidates],
                'candidate_equiv_ids': [int(gallery_ds.equiv_id[c]) for c in candidates],
            }
        else:
            positives = positive_finder(q_idx)
            if not positives:
                continue
            pos_gal_idx = rng.choice(positives)
            candidates, _ = sample_structured_pool(pos_gal_idx, gallery_index, pool_size, rng)
            positive_set = set(int(i) for i in positives)
            positive_positions = [i for i, c in enumerate(candidates) if int(c) in positive_set]
            if not positive_positions:
                continue
            pool = {
                'positive_positions': [int(p) for p in positive_positions],
                'candidate_indices': [int(c) for c in candidates],
                'candidate_ratio_ids': [int(gallery_ds.ratio_id[c]) for c in candidates],
                'candidate_equiv_ids': [int(gallery_ds.equiv_id[c]) for c in candidates],
            }

        pools[int(q_idx)] = pool

    return {'split_name': split_name, 'query_ids': list(pools.keys()), 'pools': pools}


def build_torus_pool_cache(data_dir: Path, pool_size: int, n_queries_iid: int,
                            n_queries_ood: int, seed: int) -> Dict:
    """Build master pool cache for all splits."""
    test_ds = LissajousDataset(data_dir / 'test')
    gallery_ds = load_reduced_gallery_dataset(data_dir)
    gallery_key_idx = _build_key_index(gallery_ds)
    gallery_equiv_idx = _build_equiv_key_index(gallery_ds)
    scale_ds = load_ood_dataset(data_dir, 'scale_ood')
    equiv_ds = load_ood_dataset(data_dir, 'equiv_ood')

    cache = {'config': {'pool_size': pool_size, 'seed': seed}, 'splits': {}}

    cache['splits']['iid'] = precompute_torus_pools(
        'iid', test_ds, test_ds, pool_size, n_queries_iid, seed)
    cache['splits']['scale_ood'] = precompute_torus_pools(
        'scale_ood', scale_ds, gallery_ds, pool_size, n_queries_ood, seed,
        lambda q: find_positives_scale_ood(q, scale_ds, gallery_key_idx))
    cache['splits']['equiv_ood'] = precompute_torus_pools(
        'equiv_ood', equiv_ds, gallery_ds, pool_size, n_queries_ood, seed,
        lambda q: find_positives_equiv_ood(q, equiv_ds, gallery_equiv_idx))

    return cache


def build_torus_pool_cache_val(data_dir: Path, pool_size: int,
                                n_queries: int, seed: int) -> Dict:
    """Build pool cache for val same-set only (Phase 2)."""
    val_ds = LissajousDataset(data_dir / 'val')
    cache = {'config': {'pool_size': pool_size, 'seed': seed}, 'splits': {}}
    cache['splits']['val'] = precompute_torus_pools(
        'iid', val_ds, val_ds, pool_size, n_queries, seed)
    return cache


# ============================================================
# Render OOD for mixed geometry
# ============================================================

@torch.no_grad()
def evaluate_render_ood_mixed(
    audio_enc, image_enc, proj_audio, proj_image,
    test_ds: LissajousDataset,
    scene_dir: str,
    device: torch.device,
    render_style: str = 'noisy',
    pool_cache_iid: Optional[Dict] = None,
    euclidean_weight: float = 0.5,
    torus_perm: Optional[torch.Generator] = None,
    n_queries: int = 500,
    seed: int = 42,
) -> Dict[str, Dict]:
    """Render-OOD for mixed geometry. i2a direction only, 3 modes.

    Loads render variant PNGs as image query → find matching audio in clean gallery.
    torus_perm: optional Generator for shuffling candidate torus embeddings.
    Returns dict keyed by mode ('euclidean', 'torus', 'mixed').
    """
    from PIL import Image as PILImage

    rng = random.Random(seed)

    # Gallery: clean embeddings from test_ds
    gallery_embs = extract_mixed_embeddings(audio_enc, image_enc, proj_audio, proj_image,
                                             test_ds, device, num_workers=0)
    gallery_index = build_lissajous_index(test_ds)

    N = len(test_ds)
    query_indices = rng.sample(range(N), min(n_queries, N))

    style_fname = f"figure_style_{render_style}.png"
    scene_dir = Path(scene_dir)

    mode_ranks = {m: [] for m in ['euclidean', 'torus', 'mixed']}

    for q_idx in query_indices:
        sid = str(test_ds.scene_id[q_idx])
        img_path = scene_dir / sid / style_fname
        if not img_path.exists():
            continue

        img = np.array(PILImage.open(img_path))
        img_t = torch.from_numpy(img).unsqueeze(0).float() / 255.0  # [1, H, W]
        img_t = img_t.unsqueeze(0).to(device)  # [1, 1, H, W]

        out_q = proj_image(image_enc(img_t))
        if isinstance(out_q, dict):
            q_euc = out_q['euclidean'].cpu().squeeze(0)
            q_torus = out_q['torus'].cpu().squeeze(0)
        else:
            q_euc = None
            q_torus = out_q.cpu().squeeze(0)

        # Pool: reuse iid pool if cached, else build fresh
        if pool_cache_iid and q_idx in pool_cache_iid:
            candidates = pool_cache_iid[q_idx]['candidate_indices']
            positive_positions = pool_cache_iid[q_idx]['positive_positions']
        else:
            candidates, pos_pos = sample_structured_pool(q_idx, gallery_index, 128, rng)
            positive_positions = [pos_pos]

        cand_idx = [int(c) for c in candidates]

        cand_audio_euc = gallery_embs['audio_euc'][cand_idx] if 'audio_euc' in gallery_embs else None
        cand_audio_torus = gallery_embs['audio_torus'][cand_idx]

        if torus_perm is not None:
            perm = torch.randperm(len(cand_idx), generator=torus_perm)
            cand_audio_torus = cand_audio_torus[perm]

        for mode in ['euclidean', 'torus', 'mixed']:
            if mode == 'euclidean' and cand_audio_euc is not None:
                scores = euclidean_retrieval_score(q_euc, cand_audio_euc)
            elif mode == 'torus':
                scores = torus_retrieval_score(q_torus, cand_audio_torus)
            elif mode == 'mixed' and cand_audio_euc is not None:
                scores = mixed_retrieval_score(
                    q_euc, q_torus,
                    cand_audio_euc, cand_audio_torus,
                    euclidean_weight=euclidean_weight)
            else:
                continue

            sorted_idx = scores.argsort(descending=True).tolist()
            best_rank = min(sorted_idx.index(p) + 1 for p in positive_positions)
            mode_ranks[mode].append(best_rank)

    results = {}
    for mode, ranks in mode_ranks.items():
        if not ranks:
            results[mode] = {'error': f'no {style_fname} found'}
            continue
        ranks_np = np.array(ranks)
        results[mode] = {
            'recall_at_10': float((ranks_np <= 10).mean()),
            'recall_at_1': float((ranks_np <= 1).mean()),
            'mean_rank': float(ranks_np.mean()),
            'n_queries': len(ranks),
            'render_style': render_style,
            'direction': 'i2a',
        }
    return results


# ============================================================
# Embedding extraction for mixed/torus projectors
# ============================================================

@torch.no_grad()
def extract_mixed_embeddings(
    audio_enc, image_enc, proj_audio, proj_image,
    dataset: LissajousDataset, device: torch.device,
    batch_size: int = 64, num_workers: int = 0,
) -> Dict[str, torch.Tensor]:
    """Extract embeddings for MixedProjector. Returns dict with audio/image × euc/torus."""
    audio_enc.eval(); image_enc.eval(); proj_audio.eval(); proj_image.eval()

    akey = 'audio_cqt' if getattr(audio_enc, 'input_kind', 'waveform') == 'cqt' else 'audio'
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=collate_lissajous, num_workers=num_workers, pin_memory=True)

    all_a_euc, all_a_torus, all_i_euc, all_i_torus = [], [], [], []

    for batch in loader:
        out_a = proj_audio(audio_enc(batch[akey].to(device)))
        out_i = proj_image(image_enc(batch['image'].to(device)))

        if isinstance(out_a, dict):  # MixedProjector
            all_a_euc.append(out_a['euclidean'].cpu())
            all_a_torus.append(out_a['torus'].cpu())
            all_i_euc.append(out_i['euclidean'].cpu())
            all_i_torus.append(out_i['torus'].cpu())
        else:  # TorusProjector (pure angles)
            all_a_torus.append(out_a.cpu())
            all_i_torus.append(out_i.cpu())

    result = {
        'audio_torus': torch.cat(all_a_torus),
        'image_torus': torch.cat(all_i_torus),
    }
    if all_a_euc:
        result['audio_euc'] = torch.cat(all_a_euc)
        result['image_euc'] = torch.cat(all_i_euc)

    return result
