#!/usr/bin/env python3
"""E3 Evaluation: Structured pool retrieval for Lissajous cross-modal.

Two evaluation modes:
  1. Same-set (IID): query and gallery from the same dataset. Positive = same scene_id.
  2. Cross-set (OOD): query from OOD split, gallery from IID. Positive resolved by split:
     - scale_ood: same ratio_id + phase_idx + amp_ratio in gallery
     - equiv_ood: same equiv_id + phase_idx + amp_ratio + base_freq in gallery
     - ratio_ood: NO positive in gallery (structural diagnostics only)

Pools have ratio-aware hard negatives:
  L1: Same ratio, different phase
  L2: Farey-neighbor ratios
  L3: Different ratio, same phase
  L4: Random
"""

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.escalon3.lissajous_dataset import LissajousDataset, collate_lissajous


# ============================================================
# Index building
# ============================================================

def build_lissajous_index(dataset: LissajousDataset) -> Dict:
    """Build ratio/phase indices for structured pool construction."""
    by_ratio = defaultdict(list)
    by_phase = defaultdict(list)
    ratio_of = {}
    phase_of = {}

    for i in range(len(dataset)):
        rid = int(dataset.ratio_id[i])
        pid = int(dataset.phase_idx[i])
        by_ratio[rid].append(i)
        by_phase[pid].append(i)
        ratio_of[i] = rid
        phase_of[i] = pid

    # Farey neighbors: closest ratios by float value
    ratio_ids = sorted(by_ratio.keys())
    ratio_floats = {}
    for i in range(len(dataset)):
        rid = int(dataset.ratio_id[i])
        rf = float(dataset.ratio_float[i])
        ratio_floats[rid] = rf

    farey_neighbors = defaultdict(set)
    for rid in ratio_ids:
        rf = ratio_floats[rid]
        dists = [(abs(ratio_floats[other] - rf), other)
                 for other in ratio_ids if other != rid]
        dists.sort()
        for _, neighbor in dists[:3]:
            farey_neighbors[rid].add(neighbor)

    return {
        'by_ratio': dict(by_ratio),
        'by_phase': dict(by_phase),
        'ratio_of': ratio_of,
        'phase_of': phase_of,
        'farey_neighbors': dict(farey_neighbors),
    }


def _build_key_index(dataset: LissajousDataset) -> Dict[tuple, List[int]]:
    """Index gallery scenes by (ratio_id, phase_idx, amp_ratio)."""
    key_to_indices = defaultdict(list)
    for i in range(len(dataset)):
        key = (int(dataset.ratio_id[i]),
               int(dataset.phase_idx[i]),
               round(float(dataset.amp_ratio[i]), 4))
        key_to_indices[key].append(i)
    return dict(key_to_indices)


def _build_equiv_key_index(dataset: LissajousDataset) -> Dict[tuple, List[int]]:
    """Index gallery scenes by (equiv_id, phase_idx, amp_ratio, base_freq)."""
    key_to_indices = defaultdict(list)
    for i in range(len(dataset)):
        key = (int(dataset.equiv_id[i]),
               int(dataset.phase_idx[i]),
               round(float(dataset.amp_ratio[i]), 4),
               round(float(dataset.base_freq[i]), 1))
        key_to_indices[key].append(i)
    return dict(key_to_indices)


# ============================================================
# Pool sampling
# ============================================================

def sample_structured_pool(
    positive_idx: int,
    gallery_index: Dict,
    pool_size: int = 128,
    rng: random.Random = None,
) -> Tuple[List[int], int]:
    """Build a structured pool from gallery with ratio-aware hard negatives.

    positive_idx: index in the gallery that is the correct match.
    gallery_index: index built from the gallery dataset.
    Returns (candidate_indices_in_gallery, positive_position_in_list).
    """
    if rng is None:
        rng = random.Random(42)

    ratio_of = gallery_index['ratio_of']
    phase_of = gallery_index['phase_of']
    by_ratio = gallery_index['by_ratio']
    by_phase = gallery_index['by_phase']
    farey = gallery_index['farey_neighbors']

    q_ratio = ratio_of[positive_idx]
    q_phase = phase_of[positive_idx]
    all_indices = set(ratio_of.keys())
    candidates = set()

    # L1: Same ratio, different phase
    l1_pool = [i for i in by_ratio.get(q_ratio, [])
               if i != positive_idx and phase_of[i] != q_phase]
    l1 = rng.sample(l1_pool, min(16, len(l1_pool))) if l1_pool else []
    candidates.update(l1)

    # L2: Farey-neighbor ratios
    l2_pool = []
    for nr in farey.get(q_ratio, set()):
        l2_pool.extend(by_ratio.get(nr, []))
    l2_pool = [i for i in l2_pool if i not in candidates and i != positive_idx]
    l2 = rng.sample(l2_pool, min(16, len(l2_pool))) if l2_pool else []
    candidates.update(l2)

    # L3: Different ratio, same phase
    l3_pool = [i for i in by_phase.get(q_phase, [])
               if ratio_of[i] != q_ratio and i not in candidates and i != positive_idx]
    l3 = rng.sample(l3_pool, min(16, len(l3_pool))) if l3_pool else []
    candidates.update(l3)

    # L4: Random fill
    remaining = pool_size - len(candidates)
    if remaining > 0:
        available = [i for i in all_indices if i not in candidates and i != positive_idx]
        l4 = rng.sample(available, min(remaining, len(available)))
        candidates.update(l4)

    candidates = list(candidates)[:pool_size]
    pos_position = rng.randint(0, len(candidates))
    candidates.insert(pos_position, positive_idx)

    return candidates, pos_position


# ============================================================
# Embedding extraction
# ============================================================

@torch.no_grad()
def extract_embeddings(
    audio_enc, image_enc, proj_audio, proj_image,
    dataset: LissajousDataset,
    device: torch.device,
    batch_size: int = 64,
    num_workers: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract all embeddings. Returns (audio_embs [N,D], image_embs [N,D]) on CPU."""
    audio_enc.eval(); image_enc.eval()
    proj_audio.eval(); proj_image.eval()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=collate_lissajous, num_workers=num_workers,
                        pin_memory=True)
    akey = 'audio_cqt' if getattr(audio_enc, 'input_kind', 'waveform') == 'cqt' else 'audio'
    all_a, all_i = [], []
    for batch in loader:
        z_a = proj_audio(audio_enc(batch[akey].to(device)))
        z_i = proj_image(image_enc(batch['image'].to(device)))
        all_a.append(z_a.cpu()); all_i.append(z_i.cpu())

    return torch.cat(all_a), torch.cat(all_i)


# ============================================================
# Same-set retrieval (IID)
# ============================================================

def evaluate_retrieval_sameset(
    audio_embs: torch.Tensor,
    image_embs: torch.Tensor,
    dataset: LissajousDataset,
    direction: str = 'a2i',
    pool_size: int = 128,
    n_queries: int = 500,
    seed: int = 42,
) -> Dict:
    """Same-set retrieval: query and gallery from same dataset. Positive = same index."""
    rng = random.Random(seed)
    index = build_lissajous_index(dataset)
    N = len(dataset)
    query_indices = rng.sample(range(N), min(n_queries, N))

    ranks = []
    for q_idx in query_indices:
        candidates, pos_pos = sample_structured_pool(q_idx, index, pool_size, rng)

        if direction == 'a2i':
            query_emb = audio_embs[q_idx]
            cand_embs = image_embs[candidates]
        else:
            query_emb = image_embs[q_idx]
            cand_embs = audio_embs[candidates]

        sim = torch.nn.functional.cosine_similarity(query_emb.unsqueeze(0), cand_embs, dim=-1)
        sorted_idx = sim.argsort(descending=True)
        rank = (sorted_idx == pos_pos).nonzero(as_tuple=True)[0].item() + 1
        ranks.append(rank)

    ranks = np.array(ranks)
    return {
        'mean_rank': float(ranks.mean()),
        'mrr': float((1.0 / ranks).mean()),
        'recall_at_1': float((ranks <= 1).mean()),
        'recall_at_5': float((ranks <= 5).mean()),
        'recall_at_10': float((ranks <= 10).mean()),
        'recall_at_20': float((ranks <= 20).mean()),
        'pool_size': pool_size + 1,
        'n_queries': len(ranks),
        'direction': direction,
        'mode': 'sameset',
    }


# ============================================================
# Cross-set retrieval (OOD query → IID gallery)
# ============================================================

def summarize_positive_coverage(query_ds, positive_finder, n_queries=None) -> Dict:
    """Check how many queries have at least one positive in the gallery.

    Returns dict with coverage stats. n_no_positive MUST be 0 for valid OOD eval.
    """
    N = len(query_ds) if n_queries is None else min(n_queries, len(query_ds))
    has_positive = 0
    set_sizes = []
    for i in range(N):
        positives = positive_finder(i)
        if positives:
            has_positive += 1
            set_sizes.append(len(positives))
    n_no_pos = N - has_positive
    return {
        'n_total_queries': N,
        'n_with_positive': has_positive,
        'n_no_positive': n_no_pos,
        'positive_set_size_min': min(set_sizes) if set_sizes else 0,
        'positive_set_size_max': max(set_sizes) if set_sizes else 0,
        'positive_set_size_mean': float(np.mean(set_sizes)) if set_sizes else 0.0,
    }


def find_positives_scale_ood(q_idx, query_ds, gallery_key_index):
    """scale_ood positive: same ratio_id + phase_idx + amp_ratio in gallery."""
    key = (int(query_ds.ratio_id[q_idx]),
           int(query_ds.phase_idx[q_idx]),
           round(float(query_ds.amp_ratio[q_idx]), 4))
    return gallery_key_index.get(key, [])


def find_positives_equiv_ood(q_idx, query_ds, gallery_equiv_index):
    """equiv_ood positive: same equiv_id + phase_idx + amp_ratio + base_freq in gallery."""
    key = (int(query_ds.equiv_id[q_idx]),
           int(query_ds.phase_idx[q_idx]),
           round(float(query_ds.amp_ratio[q_idx]), 4),
           round(float(query_ds.base_freq[q_idx]), 1))
    return gallery_equiv_index.get(key, [])


def evaluate_retrieval_crossset(
    query_audio_embs: torch.Tensor,
    query_image_embs: torch.Tensor,
    query_ds: LissajousDataset,
    gallery_audio_embs: torch.Tensor,
    gallery_image_embs: torch.Tensor,
    gallery_ds: LissajousDataset,
    direction: str = 'a2i',
    positive_finder=None,
    pool_size: int = 128,
    n_queries: int = 500,
    seed: int = 42,
) -> Dict:
    """Cross-set retrieval: query from OOD, gallery from IID.

    positive_finder: callable(q_idx, query_ds, ...) → list of gallery indices.
    Multi-positive: rank = rank of best positive in pool.
    """
    rng = random.Random(seed)
    gallery_index = build_lissajous_index(gallery_ds)
    N_query = len(query_ds)
    query_indices = rng.sample(range(N_query), min(n_queries, N_query))

    ranks = []
    n_no_positive = 0

    for q_idx in query_indices:
        positive_gallery_indices = positive_finder(q_idx)
        if not positive_gallery_indices:
            n_no_positive += 1
            continue

        # Pick one positive for pool construction
        pos_gal_idx = rng.choice(positive_gallery_indices)
        candidates, pos_pos = sample_structured_pool(pos_gal_idx, gallery_index, pool_size, rng)

        # Mark ALL positives in the candidate list
        positive_set = set(positive_gallery_indices)
        positive_positions = [i for i, c in enumerate(candidates) if c in positive_set]

        if not positive_positions:
            # The sampled positive should be in there
            n_no_positive += 1
            continue

        if direction == 'a2i':
            query_emb = query_audio_embs[q_idx]
            cand_embs = gallery_image_embs[candidates]
        else:
            query_emb = query_image_embs[q_idx]
            cand_embs = gallery_audio_embs[candidates]

        sim = torch.nn.functional.cosine_similarity(query_emb.unsqueeze(0), cand_embs, dim=-1)
        sorted_idx = sim.argsort(descending=True).tolist()

        # Rank = position of the BEST positive
        best_rank = min(sorted_idx.index(pp) + 1 for pp in positive_positions)
        ranks.append(best_rank)

    if not ranks:
        return {'error': 'no valid queries with positives', 'n_no_positive': n_no_positive}

    ranks = np.array(ranks)
    return {
        'mean_rank': float(ranks.mean()),
        'mrr': float((1.0 / ranks).mean()),
        'recall_at_1': float((ranks <= 1).mean()),
        'recall_at_5': float((ranks <= 5).mean()),
        'recall_at_10': float((ranks <= 10).mean()),
        'recall_at_20': float((ranks <= 20).mean()),
        'pool_size': pool_size + 1,
        'n_queries': len(ranks),
        'n_no_positive': n_no_positive,
        'direction': direction,
        'mode': 'crossset',
    }


# ============================================================
# Structural metrics
# ============================================================

def compute_structural_metrics(
    audio_embs: torch.Tensor,
    image_embs: torch.Tensor,
    dataset: LissajousDataset,
) -> Dict:
    """Silhouette score by ratio_id."""
    from sklearn.metrics import silhouette_score

    ratio_ids = np.array(dataset.ratio_id)
    unique = np.unique(ratio_ids)

    results = {}
    if len(unique) > 1:
        results['silhouette_audio'] = float(silhouette_score(
            audio_embs.numpy(), ratio_ids, metric='cosine',
            sample_size=min(1000, len(audio_embs))))
        results['silhouette_image'] = float(silhouette_score(
            image_embs.numpy(), ratio_ids, metric='cosine',
            sample_size=min(1000, len(image_embs))))
        all_embs = torch.cat([audio_embs, image_embs]).numpy()
        all_ids = np.concatenate([ratio_ids, ratio_ids])
        results['silhouette_combined'] = float(silhouette_score(
            all_embs, all_ids, metric='cosine',
            sample_size=min(2000, len(all_embs))))
    else:
        results['silhouette_audio'] = 0.0
        results['silhouette_image'] = 0.0
        results['silhouette_combined'] = 0.0

    return results


# ============================================================
# Ratio-OOD diagnostics (no positive, structural only)
# ============================================================

def diagnose_ratio_ood(
    query_audio_embs: torch.Tensor,
    query_image_embs: torch.Tensor,
    query_ds: LissajousDataset,
    gallery_audio_embs: torch.Tensor,
    gallery_image_embs: torch.Tensor,
    gallery_ds: LissajousDataset,
) -> Dict:
    """For ratio-OOD: no positive exists. Report nearest gallery ratios."""
    results = {}

    # Audio: for each query, find nearest gallery audio embedding
    sims = torch.mm(
        torch.nn.functional.normalize(query_audio_embs, dim=-1),
        torch.nn.functional.normalize(gallery_audio_embs, dim=-1).T
    )
    nn_indices = sims.argmax(dim=-1)

    # What ratio does each OOD query map to?
    gallery_ratios = np.array(gallery_ds.ratio_id)
    query_ratios = np.array(query_ds.ratio_id)
    query_floats = np.array(query_ds.ratio_float)

    mapping = defaultdict(lambda: defaultdict(int))
    for i in range(len(query_ds)):
        q_rid = int(query_ratios[i])
        g_rid = int(gallery_ratios[nn_indices[i]])
        mapping[q_rid][g_rid] += 1

    for q_rid, counts in mapping.items():
        total = sum(counts.values())
        top3 = sorted(counts.items(), key=lambda x: -x[1])[:3]
        results[f'ratio_id_{q_rid}'] = {
            'n_queries': total,
            'nearest_gallery_ratios': [(rid, cnt / total) for rid, cnt in top3],
        }

    return results


# ============================================================
# Render-OOD evaluation
# ============================================================

def evaluate_render_ood(
    audio_enc, image_enc, proj_audio, proj_image,
    test_ds: LissajousDataset,
    scene_dir: str,
    device: torch.device,
    render_style: str = 'noisy',
    n_queries: int = 500,
    seed: int = 42,
) -> Dict:
    """Render-OOD: query with noisy/thick images against clean gallery.

    Loads render variant PNGs from scene_dir, uses clean audio+image gallery from test_ds.
    """
    from PIL import Image as PILImage

    rng = random.Random(seed)

    # Gallery: clean embeddings from test_ds
    gallery_a, gallery_i = [], []
    loader = DataLoader(test_ds, batch_size=64, shuffle=False,
                        collate_fn=collate_lissajous, num_workers=4)
    with torch.no_grad():
        audio_enc.eval(); image_enc.eval(); proj_audio.eval(); proj_image.eval()
        for batch in loader:
            akey_r = 'audio_cqt' if getattr(audio_enc, 'input_kind', 'waveform') == 'cqt' else 'audio'
            gallery_a.append(proj_audio(audio_enc(batch[akey_r].to(device))).cpu())
            gallery_i.append(proj_image(image_enc(batch['image'].to(device))).cpu())

    gallery_a = torch.cat(gallery_a)
    gallery_i = torch.cat(gallery_i)
    gallery_index = build_lissajous_index(test_ds)

    # Query: load render variant images for test scenes
    scene_dir = Path(scene_dir)
    N = len(test_ds)
    query_indices = rng.sample(range(N), min(n_queries, N))
    ranks = []

    style_fname = f"figure_style_{render_style}.png"

    for q_idx in query_indices:
        sid = str(test_ds.scene_id[q_idx])
        img_path = scene_dir / sid / style_fname
        if not img_path.exists():
            continue

        img = np.array(PILImage.open(img_path))
        img_t = torch.from_numpy(img).unsqueeze(0).float() / 255.0  # [1, 128, 128]
        img_t = img_t.unsqueeze(0).to(device)  # [1, 1, 128, 128]

        with torch.no_grad():
            query_emb = proj_image(image_enc(img_t)).cpu().squeeze(0)

        # Pool from gallery (image query → audio gallery = i2a direction)
        candidates, pos_pos = sample_structured_pool(q_idx, gallery_index, 128, rng)
        cand_embs = gallery_a[candidates]  # query noisy image → find matching audio

        sim = torch.nn.functional.cosine_similarity(query_emb.unsqueeze(0), cand_embs, dim=-1)
        sorted_idx = sim.argsort(descending=True)
        rank = (sorted_idx == pos_pos).nonzero(as_tuple=True)[0].item() + 1
        ranks.append(rank)

    if not ranks:
        return {'error': f'no {style_fname} found'}

    ranks = np.array(ranks)
    return {
        'recall_at_10': float((ranks <= 10).mean()),
        'recall_at_1': float((ranks <= 1).mean()),
        'mean_rank': float(ranks.mean()),
        'n_queries': len(ranks),
        'render_style': render_style,
        'direction': 'i2a',
        'mode': 'render_ood',
    }


# ============================================================
# Full evaluation suite
# ============================================================

def run_full_eval(
    audio_embs: torch.Tensor,
    image_embs: torch.Tensor,
    dataset: LissajousDataset,
    pool_size: int = 128,
    n_queries: int = 500,
    seed: int = 42,
) -> Dict:
    """Run same-set evaluation (for IID or val). Both directions + structural."""
    a2i = evaluate_retrieval_sameset(audio_embs, image_embs, dataset, 'a2i',
                                     pool_size, n_queries, seed)
    i2a = evaluate_retrieval_sameset(audio_embs, image_embs, dataset, 'i2a',
                                     pool_size, n_queries, seed)
    S = min(a2i['recall_at_10'], i2a['recall_at_10'])
    structural = compute_structural_metrics(audio_embs, image_embs, dataset)

    return {'a2i': a2i, 'i2a': i2a, 'S': S, **structural}
