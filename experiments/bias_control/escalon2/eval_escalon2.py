#!/usr/bin/env python3
"""
Escalón 2 — Evaluation: Structured Pool Retrieval for Speech <-> EGG

Pool builder with hard negative hierarchy + grouped bootstrap CI.
Used by train_escalon2.py for structured eval during training.

Pool structure (noise0 pilot):
  L1 (16): same clip / different non-overlapping window (>=2s apart)
  L2 (16): same speaker / different utterance
  L3 (16): different speaker / same sentence_id
  L4 (80): different speaker / different utterance (random)

Metrics:
  S = min(Speech2EGG@10, EGG2Speech@10)
  CI: grouped bootstrap by speaker, 1000 resamples
"""

import json
import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

logger = logging.getLogger(__name__)

SEGMENT_LEN = 2.0


# ── Index builders ───────────────────────────────────────────────────────────

def build_lombard_index(segments: List[Dict]):
    """Build lookup indices for pool construction."""
    by_clip = defaultdict(list)
    by_speaker = defaultdict(list)
    by_sentence = defaultdict(list)

    for i, seg in enumerate(segments):
        by_clip[seg['clip_id']].append(i)
        by_speaker[seg['speaker_id']].append(i)
        by_sentence[seg['sentence_id']].append(i)

    return by_clip, by_speaker, by_sentence


# ── Pool builder ─────────────────────────────────────────────────────────────

def build_lombard_pool(query_idx, segments, by_clip, by_speaker, by_sentence,
                       pool_size=128, rng=None):
    """Build structured pool with hard negative hierarchy.

    Returns:
      pool_indices: list of segment indices (negatives only, positive is implicit)
      strata_counts: dict with actual counts per stratum
    """
    if rng is None:
        rng = np.random.RandomState(42)

    query = segments[query_idx]
    q_clip = query['clip_id']
    q_speaker = query['speaker_id']
    q_sentence = query['sentence_id']
    q_start = query['start_sec']

    pool = []
    strata_counts = {'L1': 0, 'L2': 0, 'L3': 0, 'L4': 0}

    # L1: same clip, different window, non-overlapping (>=2s separation)
    l1_candidates = [
        idx for idx in by_clip[q_clip]
        if idx != query_idx and abs(segments[idx]['start_sec'] - q_start) >= SEGMENT_LEN
    ]
    rng.shuffle(l1_candidates)
    l1_sel = l1_candidates[:16]
    pool.extend(l1_sel)
    strata_counts['L1'] = len(l1_sel)

    # L2: same speaker, different clip
    l2_candidates = [
        idx for idx in by_speaker[q_speaker]
        if segments[idx]['clip_id'] != q_clip
    ]
    rng.shuffle(l2_candidates)
    l2_sel = l2_candidates[:16]
    pool.extend(l2_sel)
    strata_counts['L2'] = len(l2_sel)

    # L3: different speaker, same sentence_id
    l3_candidates = [
        idx for idx in by_sentence[q_sentence]
        if segments[idx]['speaker_id'] != q_speaker
    ]
    rng.shuffle(l3_candidates)
    l3_sel = l3_candidates[:16]
    pool.extend(l3_sel)
    strata_counts['L3'] = len(l3_sel)

    # L4: different speaker, different utterance (fill remaining)
    used = set(pool)
    used.add(query_idx)
    target_l4 = pool_size - 1 - len(pool)
    l4_candidates = [
        idx for spk, indices in by_speaker.items() if spk != q_speaker
        for idx in indices if idx not in used
    ]
    rng.shuffle(l4_candidates)
    l4_sel = l4_candidates[:target_l4]
    pool.extend(l4_sel)
    strata_counts['L4'] = len(l4_sel)

    return pool, strata_counts


# ── Embedding extraction ────────────────────────────────────────────────────

@torch.no_grad()
def extract_embeddings_lombard(model_speech, model_egg, proj_speech, proj_egg,
                               dataloader, device='cuda'):
    """Extract embeddings for all segments in dataloader.

    Returns:
      speech_embs: dict (clip_id, seg_idx) -> np.array [D]
      egg_embs: dict (clip_id, seg_idx) -> np.array [D]
    """
    model_speech.eval()
    model_egg.eval()
    proj_speech.eval()
    proj_egg.eval()

    speech_embs = {}
    egg_embs = {}

    for batch in tqdm(dataloader, desc="Extracting embeddings", leave=False):
        speech_wav = batch['speech'].to(device)
        egg_wav = batch['egg'].to(device)

        z_speech = proj_speech(model_speech(speech_wav))
        z_egg = proj_egg(model_egg(egg_wav))

        z_speech = z_speech.cpu().numpy()
        z_egg = z_egg.cpu().numpy()

        for i in range(len(batch['clip_id'])):
            key = (batch['clip_id'][i], batch['segment_idx'][i])
            speech_embs[key] = z_speech[i]
            egg_embs[key] = z_egg[i]

    return speech_embs, egg_embs


# ── Retrieval evaluation ────────────────────────────────────────────────────

def evaluate_retrieval_lombard(speech_embs, egg_embs, segments,
                               pool_size=128, n_queries=500, seed=42, k=10):
    """Evaluate cross-modal retrieval with structured pool.

    Returns dict with S2E@k, E2S@k, S, per-query details.
    """
    rng = np.random.RandomState(seed)
    by_clip, by_speaker, by_sentence = build_lombard_index(segments)

    n_queries = min(n_queries, len(segments))
    query_indices = rng.choice(len(segments), size=n_queries, replace=False)

    s2e_hits = []
    e2s_hits = []
    per_query = []

    for qi in query_indices:
        seg = segments[qi]
        key = (seg['clip_id'], seg['segment_idx'])

        if key not in speech_embs or key not in egg_embs:
            continue

        pool_indices, strata = build_lombard_pool(
            qi, segments, by_clip, by_speaker, by_sentence,
            pool_size=pool_size, rng=rng,
        )

        q_speech = speech_embs[key]
        q_egg = egg_embs[key]
        pos_egg = egg_embs[key]
        pos_speech = speech_embs[key]

        # Speech -> EGG
        sims = [(qi, float(_cosine(q_speech, pos_egg)))]
        for pi in pool_indices:
            ps = segments[pi]
            pk = (ps['clip_id'], ps['segment_idx'])
            if pk in egg_embs:
                sims.append((pi, float(_cosine(q_speech, egg_embs[pk]))))
        sims.sort(key=lambda x: -x[1])
        rank_s2e = next(i for i, (idx, _) in enumerate(sims) if idx == qi)
        s2e_hits.append(1 if rank_s2e < k else 0)

        # EGG -> Speech
        sims = [(qi, float(_cosine(q_egg, pos_speech)))]
        for pi in pool_indices:
            ps = segments[pi]
            pk = (ps['clip_id'], ps['segment_idx'])
            if pk in speech_embs:
                sims.append((pi, float(_cosine(q_egg, speech_embs[pk]))))
        sims.sort(key=lambda x: -x[1])
        rank_e2s = next(i for i, (idx, _) in enumerate(sims) if idx == qi)
        e2s_hits.append(1 if rank_e2s < k else 0)

        per_query.append({
            'clip_id': seg['clip_id'],
            'segment_idx': seg['segment_idx'],
            'speaker_id': seg['speaker_id'],
            'sentence_id': seg['sentence_id'],
            'rank_s2e': rank_s2e,
            'rank_e2s': rank_e2s,
            'strata': strata,
        })

    s2e_at_k = np.mean(s2e_hits) if s2e_hits else 0.0
    e2s_at_k = np.mean(e2s_hits) if e2s_hits else 0.0
    S = min(s2e_at_k, e2s_at_k)

    return {
        'S2E_at_k': float(s2e_at_k),
        'E2S_at_k': float(e2s_at_k),
        'S': float(S),
        'k': k,
        'pool_size': pool_size,
        'n_queries': len(s2e_hits),
        'random_baseline': k / pool_size,
        'per_query': per_query,
    }


def _cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return np.dot(a, b) / (na * nb)


# ── CI bootstrap (grouped by speaker) ───────────────────────────────────────

def grouped_bootstrap_ci(per_query, alpha=0.05, n_bootstrap=1000, seed=42, k=10):
    """Grouped bootstrap CI by speaker."""
    rng = np.random.RandomState(seed)

    by_speaker = defaultdict(list)
    for q in per_query:
        by_speaker[q['speaker_id']].append(q)

    speakers = list(by_speaker.keys())
    n_spk = len(speakers)
    bootstrap_S = []

    for _ in range(n_bootstrap):
        sampled = rng.choice(speakers, size=n_spk, replace=True)
        s2e, e2s = [], []
        for spk in sampled:
            for q in by_speaker[spk]:
                s2e.append(1 if q['rank_s2e'] < k else 0)
                e2s.append(1 if q['rank_e2s'] < k else 0)
        if s2e:
            bootstrap_S.append(min(np.mean(s2e), np.mean(e2s)))

    bootstrap_S = np.array(bootstrap_S)
    return {
        'S_mean': float(np.mean(bootstrap_S)),
        'S_ci_lo': float(np.percentile(bootstrap_S, 100 * alpha / 2)),
        'S_ci_hi': float(np.percentile(bootstrap_S, 100 * (1 - alpha / 2))),
        'n_speakers': n_spk,
        'n_bootstrap': n_bootstrap,
    }
