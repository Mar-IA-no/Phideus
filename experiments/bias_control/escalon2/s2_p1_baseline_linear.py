#!/usr/bin/env python3
"""
S2-P1: Baseline Lineal — Speech <-> EGG Cross-Modal Retrieval

Extracts simple features from speech and EGG segments, then evaluates
cross-modal retrieval using CCA and Ridge regression with a structured
pool containing hard negatives.

Protocol (from S2-P0):
  sr=16kHz, segment=2.0s, hop=0.5s
  Pilot: noise0 (clean condition) only
  Pool: 128, R@10 random = 7.8%
  S = min(Speech2EGG@10, EGG2Speech@10)
  CI: grouped bootstrap by speaker, 1000 resamples

Hard negative hierarchy (pilot, noise0 only):
  L1 (16): same clip / different non-overlapping window (>=2s apart)
  L2 (16): same speaker / different utterance
  L3 (16): different speaker / same sentence_id
  L4 (80): different speaker / different utterance (random)

Usage:
  python experiments/bias_control/escalon2/s2_p1_baseline_linear.py \
      --lombard-dir data/lombard/FLombard \
      --manifest data/lombard/manifest.json \
      --segment-index data/lombard/segment_index.json \
      --output-dir data/lombard/p1_results \
      --condition noise0 --workers 14
"""

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import soundfile as sf

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ── Protocol constants (must match P0) ───────────────────────────────────────

SAMPLE_RATE = 16000
SEGMENT_LEN = 2.0
SEGMENT_HOP = 0.5
SEGMENT_SAMPLES = int(SEGMENT_LEN * SAMPLE_RATE)  # 32000

# Band edges for 16kHz STFT (n_fft=1024, freq_res=15.625 Hz/bin)
# Each tuple: (bin_start, bin_end) inclusive
BAND_EDGES_16K = [
    (3, 6),     # ~47-94 Hz
    (6, 12),    # ~94-188 Hz
    (12, 24),   # ~188-375 Hz
    (24, 48),   # ~375-750 Hz
    (48, 96),   # ~750-1500 Hz
    (96, 192),  # ~1500-3000 Hz
    (192, 384), # ~3000-6000 Hz
    (384, 513), # ~6000-8000 Hz
]


# ── Feature extraction ───────────────────────────────────────────────────────

def extract_segment_features(waveform, sr=SAMPLE_RATE):
    """Extract simple features from a 2s waveform segment.

    Returns ~20-dim feature vector:
      - 8 band energies (mean log-magnitude per band)
      - 8 band energy stds
      - F0 mean, F0 std, F0 range (via autocorrelation)
      - voicing fraction (energy-based)
    """
    import warnings

    n_fft = 1024
    hop_length = 256

    # STFT
    # Pad if needed
    if len(waveform) < n_fft:
        waveform = np.pad(waveform, (0, n_fft - len(waveform)))

    from numpy.fft import rfft
    n_frames = 1 + (len(waveform) - n_fft) // hop_length
    frames = np.lib.stride_tricks.as_strided(
        waveform,
        shape=(n_frames, n_fft),
        strides=(hop_length * waveform.strides[0], waveform.strides[0]),
    )
    window = np.hanning(n_fft)
    spec = np.abs(rfft(frames * window, axis=1))  # [n_frames, n_fft//2+1]

    # Band energies (mean/std of log-magnitude per band)
    band_means = []
    band_stds = []
    for lo, hi in BAND_EDGES_16K:
        band = spec[:, lo:hi]
        log_band = np.log1p(band)  # log(1+x) for stability
        band_means.append(np.mean(log_band))
        band_stds.append(np.std(log_band))

    # F0 estimation via autocorrelation (simple, no external deps)
    f0_values = _estimate_f0_autocorr(waveform, sr)

    if len(f0_values) > 0:
        f0_mean = np.mean(f0_values)
        f0_std = np.std(f0_values)
        f0_range = np.max(f0_values) - np.min(f0_values)
    else:
        f0_mean = 0.0
        f0_std = 0.0
        f0_range = 0.0

    # Voicing fraction (energy-based: fraction of frames above threshold)
    frame_energy = np.sum(spec ** 2, axis=1)
    if frame_energy.max() > 0:
        threshold = 0.01 * frame_energy.max()
        voicing_frac = np.mean(frame_energy > threshold)
    else:
        voicing_frac = 0.0

    features = np.array(
        band_means + band_stds + [f0_mean, f0_std, f0_range, voicing_frac],
        dtype=np.float32,
    )
    return features  # 8+8+4 = 20 dims


def _estimate_f0_autocorr(waveform, sr, frame_len=1024, hop=256,
                           fmin=50, fmax=500):
    """Simple autocorrelation F0 estimator. Returns array of F0 values (Hz)
    for voiced frames."""
    lag_min = sr // fmax  # ~32 samples for 500Hz
    lag_max = sr // fmin  # ~320 samples for 50Hz

    n_frames = 1 + (len(waveform) - frame_len) // hop
    f0_values = []

    for i in range(n_frames):
        start = i * hop
        frame = waveform[start:start + frame_len]
        frame = frame - np.mean(frame)

        if np.max(np.abs(frame)) < 1e-6:
            continue

        # Normalized autocorrelation in lag range
        autocorr = np.correlate(frame, frame, mode='full')
        autocorr = autocorr[frame_len - 1:]  # positive lags only
        if autocorr[0] == 0:
            continue
        autocorr = autocorr / autocorr[0]

        # Find peak in valid lag range
        search = autocorr[lag_min:lag_max + 1]
        if len(search) == 0:
            continue

        peak_idx = np.argmax(search)
        peak_val = search[peak_idx]

        # Threshold for voicing decision
        if peak_val > 0.3:
            lag = lag_min + peak_idx
            f0 = sr / lag
            f0_values.append(f0)

    return np.array(f0_values) if f0_values else np.array([])


# ── Segment loading ──────────────────────────────────────────────────────────

def load_segment_audio(lombard_dir, clip_id, speaker_id, start_sec, end_sec,
                       modality='speech'):
    """Load a segment from speech or EGG file."""
    if modality == 'speech':
        subdir = 'process/wav'
    else:
        subdir = 'process/egg'

    path = os.path.join(lombard_dir, subdir, speaker_id,
                        f"{clip_id}.wav")
    start_sample = int(start_sec * SAMPLE_RATE)
    end_sample = int(end_sec * SAMPLE_RATE)

    waveform, sr = sf.read(path, start=start_sample, stop=end_sample,
                           dtype='float32')
    assert sr == SAMPLE_RATE, f"Expected {SAMPLE_RATE}, got {sr}"
    assert len(waveform) == SEGMENT_SAMPLES, \
        f"Expected {SEGMENT_SAMPLES} samples, got {len(waveform)}"
    return waveform


def extract_features_for_segment(args):
    """Worker function for parallel feature extraction."""
    lombard_dir, seg, modality = args
    try:
        wav = load_segment_audio(
            lombard_dir, seg['clip_id'], seg['speaker_id'],
            seg['start_sec'], seg['end_sec'], modality,
        )
        feat = extract_segment_features(wav)
        return seg['clip_id'], seg['segment_idx'], feat, None
    except Exception as e:
        return seg['clip_id'], seg['segment_idx'], None, str(e)


# ── Pool builder ─────────────────────────────────────────────────────────────

def build_segment_indices(segments):
    """Build lookup indices for pool construction.

    Returns:
      by_clip: {clip_id: [seg_indices]}
      by_speaker: {speaker_id: [seg_indices]}
      by_sentence: {sentence_id: [seg_indices]}
    """
    by_clip = defaultdict(list)
    by_speaker = defaultdict(list)
    by_sentence = defaultdict(list)

    for i, seg in enumerate(segments):
        by_clip[seg['clip_id']].append(i)
        by_speaker[seg['speaker_id']].append(i)
        by_sentence[seg['sentence_id']].append(i)

    return by_clip, by_speaker, by_sentence


def build_lombard_pool(query_idx, segments, by_clip, by_speaker, by_sentence,
                       pool_size=128, rng=None):
    """Build structured pool with hard negative hierarchy.

    Positive: same segment (same clip, same segment_idx) — other modality.

    Hard negatives (noise0 pilot):
      L1 (16): same clip / different window, non-overlapping (>=2s apart)
      L2 (16): same speaker / different utterance
      L3 (16): different speaker / same sentence_id
      L4 (80): different speaker / different utterance

    Returns:
      pool_indices: list of segment indices (first is positive)
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

    # Positive is the query itself (same index, other modality)
    # Not added to pool — it's implicit

    # L1: same clip, different window, non-overlapping (>=2s separation)
    l1_candidates = []
    for idx in by_clip[q_clip]:
        if idx == query_idx:
            continue
        seg = segments[idx]
        # Non-overlapping: separation >= segment_len
        if abs(seg['start_sec'] - q_start) >= SEGMENT_LEN:
            l1_candidates.append(idx)
    rng.shuffle(l1_candidates)
    l1_selected = l1_candidates[:16]
    pool.extend(l1_selected)
    strata_counts['L1'] = len(l1_selected)

    # L2: same speaker, different utterance (different clip)
    l2_candidates = []
    for idx in by_speaker[q_speaker]:
        seg = segments[idx]
        if seg['clip_id'] != q_clip:
            l2_candidates.append(idx)
    rng.shuffle(l2_candidates)
    l2_selected = l2_candidates[:16]
    pool.extend(l2_selected)
    strata_counts['L2'] = len(l2_selected)

    # L3: different speaker, same sentence_id
    l3_candidates = []
    for idx in by_sentence[q_sentence]:
        seg = segments[idx]
        if seg['speaker_id'] != q_speaker:
            l3_candidates.append(idx)
    rng.shuffle(l3_candidates)
    l3_selected = l3_candidates[:16]
    pool.extend(l3_selected)
    strata_counts['L3'] = len(l3_selected)

    # L4: different speaker, different utterance (fill remaining)
    used = set(pool)
    used.add(query_idx)
    target_l4 = pool_size - 1 - len(pool)  # -1 for positive
    l4_candidates = []
    for spk, indices in by_speaker.items():
        if spk == q_speaker:
            continue
        for idx in indices:
            if idx not in used:
                l4_candidates.append(idx)
    rng.shuffle(l4_candidates)
    l4_selected = l4_candidates[:target_l4]
    pool.extend(l4_selected)
    strata_counts['L4'] = len(l4_selected)

    # If still short, fill with any remaining
    if len(pool) < pool_size - 1:
        all_used = set(pool)
        all_used.add(query_idx)
        remaining = [i for i in range(len(segments)) if i not in all_used]
        rng.shuffle(remaining)
        extra = remaining[:pool_size - 1 - len(pool)]
        pool.extend(extra)
        strata_counts['L4'] += len(extra)

    return pool, strata_counts


# ── Retrieval evaluation ─────────────────────────────────────────────────────

def cosine_similarity(a, b):
    """Cosine similarity between vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return np.dot(a, b) / (norm_a * norm_b)


def evaluate_retrieval(speech_features, egg_features, segments,
                       pool_size=128, n_queries=500, seed=42, k=10):
    """Evaluate cross-modal retrieval with structured pool.

    Args:
      speech_features: dict of (clip_id, seg_idx) -> feature vector
      egg_features: dict of (clip_id, seg_idx) -> feature vector
      segments: list of segment dicts (test set, noise0 only)
      pool_size: pool size (128)
      n_queries: max queries
      k: R@k

    Returns dict with S2E_at_k, E2S_at_k, S, per-stratum stats, per-query details.
    """
    rng = np.random.RandomState(seed)

    # Build indices
    by_clip, by_speaker, by_sentence = build_segment_indices(segments)

    # Select queries
    n_queries = min(n_queries, len(segments))
    query_indices = rng.choice(len(segments), size=n_queries, replace=False)

    results = {
        'speech2egg': [],
        'egg2speech': [],
        'strata_counts': defaultdict(lambda: defaultdict(int)),
        'per_query': [],
    }

    for qi in query_indices:
        seg = segments[qi]
        key = (seg['clip_id'], seg['segment_idx'])

        if key not in speech_features or key not in egg_features:
            continue

        pool_indices, strata = build_lombard_pool(
            qi, segments, by_clip, by_speaker, by_sentence,
            pool_size=pool_size, rng=rng,
        )

        # Accumulate strata counts
        for stratum, count in strata.items():
            results['strata_counts'][stratum]['total'] += count
            results['strata_counts'][stratum]['n_queries'] += 1

        # Speech -> EGG retrieval
        query_speech = speech_features[key]
        # Positive is the same segment's EGG
        positive_egg = egg_features[key]
        # Pool EGG features
        pool_eggs = []
        for pi in pool_indices:
            ps = segments[pi]
            pk = (ps['clip_id'], ps['segment_idx'])
            if pk in egg_features:
                pool_eggs.append((pi, egg_features[pk]))

        if len(pool_eggs) == 0:
            continue

        # Compute similarities: positive + pool
        sims_s2e = [(qi, cosine_similarity(query_speech, positive_egg))]
        for pi, ef in pool_eggs:
            sims_s2e.append((pi, cosine_similarity(query_speech, ef)))
        sims_s2e.sort(key=lambda x: -x[1])
        rank_s2e = next(i for i, (idx, _) in enumerate(sims_s2e) if idx == qi)
        hit_s2e = 1 if rank_s2e < k else 0
        results['speech2egg'].append(hit_s2e)

        # EGG -> Speech retrieval
        query_egg = egg_features[key]
        positive_speech = speech_features[key]
        pool_speeches = []
        for pi in pool_indices:
            ps = segments[pi]
            pk = (ps['clip_id'], ps['segment_idx'])
            if pk in speech_features:
                pool_speeches.append((pi, speech_features[pk]))

        sims_e2s = [(qi, cosine_similarity(query_egg, positive_speech))]
        for pi, sf_feat in pool_speeches:
            sims_e2s.append((pi, cosine_similarity(query_egg, sf_feat)))
        sims_e2s.sort(key=lambda x: -x[1])
        rank_e2s = next(i for i, (idx, _) in enumerate(sims_e2s) if idx == qi)
        hit_e2s = 1 if rank_e2s < k else 0
        results['egg2speech'].append(hit_e2s)

        # Per-query detail
        results['per_query'].append({
            'clip_id': seg['clip_id'],
            'segment_idx': seg['segment_idx'],
            'speaker_id': seg['speaker_id'],
            'sentence_id': seg['sentence_id'],
            'rank_s2e': rank_s2e,
            'rank_e2s': rank_e2s,
            'strata': strata,
        })

    s2e_at_k = np.mean(results['speech2egg']) if results['speech2egg'] else 0.0
    e2s_at_k = np.mean(results['egg2speech']) if results['egg2speech'] else 0.0
    S = min(s2e_at_k, e2s_at_k)

    return {
        'S2E_at_k': float(s2e_at_k),
        'E2S_at_k': float(e2s_at_k),
        'S': float(S),
        'k': k,
        'pool_size': pool_size,
        'n_queries': len(results['speech2egg']),
        'random_baseline': k / pool_size,
        'per_query': results['per_query'],
        'strata_counts': {k: dict(v) for k, v in results['strata_counts'].items()},
    }


def evaluate_retrieval_cca(speech_features_dict, egg_features_dict,
                           train_segments, test_segments,
                           pool_size=128, n_queries=500, seed=42, k=10,
                           n_components=10):
    """CCA-based retrieval: project both modalities to shared CCA space,
    then do cosine retrieval with structured pool.

    Steps:
      1. Fit CCA on train features
      2. Project test features to CCA space
      3. Evaluate retrieval with structured pool
    """
    from sklearn.cross_decomposition import CCA

    # Collect aligned train pairs
    train_keys = []
    for seg in train_segments:
        key = (seg['clip_id'], seg['segment_idx'])
        if key in speech_features_dict and key in egg_features_dict:
            train_keys.append(key)

    if len(train_keys) < n_components * 2:
        logger.warning(f"Too few train pairs ({len(train_keys)}) for CCA")
        return None

    X_train = np.array([speech_features_dict[k] for k in train_keys])
    Y_train = np.array([egg_features_dict[k] for k in train_keys])

    # Fit CCA
    n_comp = min(n_components, X_train.shape[1], Y_train.shape[1])
    cca = CCA(n_components=n_comp, max_iter=1000)
    cca.fit(X_train, Y_train)

    # CCA correlations (diagnostic)
    X_c, Y_c = cca.transform(X_train, Y_train)
    correlations = [np.corrcoef(X_c[:, i], Y_c[:, i])[0, 1]
                    for i in range(n_comp)]
    logger.info(f"CCA correlations (train): {[f'{c:.3f}' for c in correlations]}")

    # Project test features
    test_speech_proj = {}
    test_egg_proj = {}
    for seg in test_segments:
        key = (seg['clip_id'], seg['segment_idx'])
        if key in speech_features_dict:
            proj = cca.transform(speech_features_dict[key].reshape(1, -1),
                                 np.zeros((1, Y_train.shape[1])))[0]
            test_speech_proj[key] = proj.flatten()
        if key in egg_features_dict:
            proj = cca.transform(np.zeros((1, X_train.shape[1])),
                                 egg_features_dict[key].reshape(1, -1))[1]
            test_egg_proj[key] = proj.flatten()

    # Retrieval with CCA-projected features
    retrieval_results = evaluate_retrieval(
        test_speech_proj, test_egg_proj, test_segments,
        pool_size=pool_size, n_queries=n_queries, seed=seed, k=k,
    )
    retrieval_results['method'] = 'CCA'
    retrieval_results['cca_correlations'] = [float(c) for c in correlations]
    retrieval_results['n_components'] = n_comp
    retrieval_results['n_train_pairs'] = len(train_keys)

    return retrieval_results


def evaluate_ridge(speech_features_dict, egg_features_dict,
                   train_segments, test_segments):
    """Ridge regression: predict EGG features from Speech (and vice versa).
    Reports R² as auxiliary diagnostic (not a criterion for advancement)."""
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score

    # Collect aligned train pairs
    train_keys = []
    for seg in train_segments:
        key = (seg['clip_id'], seg['segment_idx'])
        if key in speech_features_dict and key in egg_features_dict:
            train_keys.append(key)

    test_keys = []
    for seg in test_segments:
        key = (seg['clip_id'], seg['segment_idx'])
        if key in speech_features_dict and key in egg_features_dict:
            test_keys.append(key)

    X_train = np.array([speech_features_dict[k] for k in train_keys])
    Y_train = np.array([egg_features_dict[k] for k in train_keys])
    X_test = np.array([speech_features_dict[k] for k in test_keys])
    Y_test = np.array([egg_features_dict[k] for k in test_keys])

    # Speech -> EGG
    ridge_s2e = Ridge(alpha=1.0)
    ridge_s2e.fit(X_train, Y_train)
    Y_pred = ridge_s2e.predict(X_test)
    r2_s2e = r2_score(Y_test, Y_pred, multioutput='variance_weighted')

    # EGG -> Speech
    ridge_e2s = Ridge(alpha=1.0)
    ridge_e2s.fit(Y_train, X_train)
    X_pred = ridge_e2s.predict(Y_test)
    r2_e2s = r2_score(X_test, X_pred, multioutput='variance_weighted')

    return {
        'r2_speech2egg': float(r2_s2e),
        'r2_egg2speech': float(r2_e2s),
        'n_train_pairs': len(train_keys),
        'n_test_pairs': len(test_keys),
    }


# ── CI bootstrap (grouped by speaker) ───────────────────────────────────────

def grouped_bootstrap_ci(per_query_results, alpha=0.05, n_bootstrap=1000,
                         seed=42):
    """Compute CI for S using grouped bootstrap by speaker.

    With only 5 test speakers, naive per-query bootstrap would be
    too optimistic due to intra-speaker correlation.
    """
    rng = np.random.RandomState(seed)

    # Group by speaker
    by_speaker = defaultdict(list)
    for q in per_query_results:
        by_speaker[q['speaker_id']].append(q)

    speakers = list(by_speaker.keys())
    n_speakers = len(speakers)

    bootstrap_S = []
    for _ in range(n_bootstrap):
        # Resample speakers with replacement
        sampled_speakers = rng.choice(speakers, size=n_speakers, replace=True)

        s2e_hits = []
        e2s_hits = []
        for spk in sampled_speakers:
            for q in by_speaker[spk]:
                s2e_hits.append(1 if q['rank_s2e'] < 10 else 0)
                e2s_hits.append(1 if q['rank_e2s'] < 10 else 0)

        if len(s2e_hits) > 0:
            s2e = np.mean(s2e_hits)
            e2s = np.mean(e2s_hits)
            bootstrap_S.append(min(s2e, e2s))

    bootstrap_S = np.array(bootstrap_S)
    lo = np.percentile(bootstrap_S, 100 * alpha / 2)
    hi = np.percentile(bootstrap_S, 100 * (1 - alpha / 2))
    mean = np.mean(bootstrap_S)

    return {
        'S_mean': float(mean),
        'S_ci_lo': float(lo),
        'S_ci_hi': float(hi),
        'alpha': alpha,
        'n_bootstrap': n_bootstrap,
        'n_speakers': n_speakers,
        'speakers': speakers,
    }


# ── Per-stratum analysis ────────────────────────────────────────────────────

def analyze_per_stratum(per_query_results, segments, pool_size=128, k=10,
                        seed=42):
    """Analyze retrieval performance by hard negative stratum.

    For each query, check: does the positive beat each stratum?
    Key diagnostic: if L1 (same clip/diff window) is at chance,
    the signal is spurious.
    """
    # Count how often positive beats each stratum's negatives
    # We approximate by checking if rank < stratum start position
    # Actually, we need the per-query similarity details for this.
    # For now, report aggregate rank statistics.

    ranks_s2e = [q['rank_s2e'] for q in per_query_results]
    ranks_e2s = [q['rank_e2s'] for q in per_query_results]

    # Median rank
    return {
        'median_rank_s2e': float(np.median(ranks_s2e)),
        'median_rank_e2s': float(np.median(ranks_e2s)),
        'mean_rank_s2e': float(np.mean(ranks_s2e)),
        'mean_rank_e2s': float(np.mean(ranks_e2s)),
        'n_queries': len(ranks_s2e),
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='S2-P1: Baseline Linear')
    parser.add_argument('--lombard-dir', type=str, required=True,
                        help='Path to FLombard directory')
    parser.add_argument('--manifest', type=str, required=True,
                        help='Path to manifest.json')
    parser.add_argument('--segment-index', type=str, required=True,
                        help='Path to segment_index.json')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for results')
    parser.add_argument('--condition', type=str, default='noise0',
                        help='Noise condition for pilot (default: noise0)')
    parser.add_argument('--pool-size', type=int, default=128)
    parser.add_argument('--n-queries', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--workers', type=int, default=14)
    parser.add_argument('--feature-cache', type=str, default=None,
                        help='Path to save/load feature cache (.npz)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load segment index
    logger.info(f"Loading segment index from {args.segment_index}")
    with open(args.segment_index) as f:
        si = json.load(f)
    all_segments = si['segments']

    # Filter by condition
    cond = args.condition
    logger.info(f"Filtering segments for condition: {cond}")
    train_segments = [s for s in all_segments
                      if s['split'] == 'train' and s['noise_condition'] == cond]
    test_segments = [s for s in all_segments
                     if s['split'] == 'test' and s['noise_condition'] == cond]
    val_segments = [s for s in all_segments
                    if s['split'] == 'validation' and s['noise_condition'] == cond]

    logger.info(f"Train: {len(train_segments)}, Val: {len(val_segments)}, "
                f"Test: {len(test_segments)} segments ({cond})")

    # Feature extraction or cache loading
    feature_cache = args.feature_cache or os.path.join(
        args.output_dir, f'features_{cond}.npz')

    if os.path.exists(feature_cache):
        logger.info(f"Loading cached features from {feature_cache}")
        data = np.load(feature_cache, allow_pickle=True)
        speech_features = {eval(k): v for k, v in zip(
            data['keys'], data['speech_features'])}
        egg_features = {eval(k): v for k, v in zip(
            data['keys'], data['egg_features'])}
        logger.info(f"Loaded {len(speech_features)} cached feature pairs")
    else:
        # Extract features for train + test + val
        segments_to_extract = train_segments + test_segments + val_segments
        logger.info(f"Extracting features for {len(segments_to_extract)} segments "
                     f"(2 modalities x {len(segments_to_extract)} = "
                     f"{2*len(segments_to_extract)} total, {args.workers} workers)")

        speech_features = {}
        egg_features = {}

        t0 = time.time()

        # Extract speech features
        logger.info("Extracting speech features...")
        speech_args = [(args.lombard_dir, seg, 'speech')
                       for seg in segments_to_extract]
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(extract_features_for_segment, a): i
                       for i, a in enumerate(speech_args)}
            done = 0
            for future in as_completed(futures):
                clip_id, seg_idx, feat, err = future.result()
                if err:
                    logger.warning(f"Speech error {clip_id}:{seg_idx}: {err}")
                else:
                    speech_features[(clip_id, seg_idx)] = feat
                done += 1
                if done % 5000 == 0:
                    logger.info(f"  Speech: {done}/{len(speech_args)}")

        logger.info(f"Speech features: {len(speech_features)} "
                     f"({time.time()-t0:.1f}s)")

        # Extract EGG features
        t1 = time.time()
        logger.info("Extracting EGG features...")
        egg_args = [(args.lombard_dir, seg, 'egg')
                    for seg in segments_to_extract]
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(extract_features_for_segment, a): i
                       for i, a in enumerate(egg_args)}
            done = 0
            for future in as_completed(futures):
                clip_id, seg_idx, feat, err = future.result()
                if err:
                    logger.warning(f"EGG error {clip_id}:{seg_idx}: {err}")
                else:
                    egg_features[(clip_id, seg_idx)] = feat
                done += 1
                if done % 5000 == 0:
                    logger.info(f"  EGG: {done}/{len(egg_args)}")

        logger.info(f"EGG features: {len(egg_features)} "
                     f"({time.time()-t1:.1f}s)")

        # Save cache
        keys = sorted(speech_features.keys())
        keys = [k for k in keys if k in egg_features]  # only paired
        np.savez(feature_cache,
                 keys=np.array([str(k) for k in keys]),
                 speech_features=np.array([speech_features[k] for k in keys]),
                 egg_features=np.array([egg_features[k] for k in keys]))
        logger.info(f"Saved {len(keys)} feature pairs to {feature_cache}")

    # ── 1. Ridge regression (auxiliary diagnostic) ───────────────────────────
    logger.info("=" * 60)
    logger.info("RIDGE REGRESSION (auxiliary diagnostic)")
    logger.info("=" * 60)
    ridge_results = evaluate_ridge(
        speech_features, egg_features,
        train_segments, test_segments,
    )
    logger.info(f"  R² Speech→EGG: {ridge_results['r2_speech2egg']:.4f}")
    logger.info(f"  R² EGG→Speech: {ridge_results['r2_egg2speech']:.4f}")
    logger.info(f"  Train pairs: {ridge_results['n_train_pairs']}, "
                f"Test pairs: {ridge_results['n_test_pairs']}")

    # ── 2. Raw feature retrieval (no projection) ────────────────────────────
    logger.info("=" * 60)
    logger.info("RAW FEATURE RETRIEVAL (cosine on raw features)")
    logger.info("=" * 60)
    raw_retrieval = evaluate_retrieval(
        speech_features, egg_features, test_segments,
        pool_size=args.pool_size, n_queries=args.n_queries,
        seed=args.seed,
    )
    logger.info(f"  S2E@{raw_retrieval['k']}: {raw_retrieval['S2E_at_k']:.4f}")
    logger.info(f"  E2S@{raw_retrieval['k']}: {raw_retrieval['E2S_at_k']:.4f}")
    logger.info(f"  S = min(S2E, E2S) = {raw_retrieval['S']:.4f}")
    logger.info(f"  Random baseline: {raw_retrieval['random_baseline']:.4f}")
    logger.info(f"  N queries: {raw_retrieval['n_queries']}")

    # Strata
    for stratum, counts in sorted(raw_retrieval['strata_counts'].items()):
        avg = counts['total'] / max(counts['n_queries'], 1)
        logger.info(f"  {stratum}: avg {avg:.1f} per query")

    # CI
    raw_ci = grouped_bootstrap_ci(raw_retrieval['per_query'])
    logger.info(f"  CI (grouped by speaker): [{raw_ci['S_ci_lo']:.4f}, "
                f"{raw_ci['S_ci_hi']:.4f}]")

    # Per-stratum rank analysis
    rank_analysis = analyze_per_stratum(raw_retrieval['per_query'],
                                        test_segments)
    logger.info(f"  Median rank S2E: {rank_analysis['median_rank_s2e']:.1f}, "
                f"E2S: {rank_analysis['median_rank_e2s']:.1f}")

    # ── 3. CCA retrieval ────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("CCA RETRIEVAL (project to shared CCA space)")
    logger.info("=" * 60)
    cca_retrieval = evaluate_retrieval_cca(
        speech_features, egg_features,
        train_segments, test_segments,
        pool_size=args.pool_size, n_queries=args.n_queries,
        seed=args.seed,
    )
    if cca_retrieval is not None:
        logger.info(f"  CCA components: {cca_retrieval['n_components']}")
        logger.info(f"  CCA correlations: "
                    f"{[f'{c:.3f}' for c in cca_retrieval['cca_correlations']]}")
        logger.info(f"  S2E@{cca_retrieval['k']}: "
                    f"{cca_retrieval['S2E_at_k']:.4f}")
        logger.info(f"  E2S@{cca_retrieval['k']}: "
                    f"{cca_retrieval['E2S_at_k']:.4f}")
        logger.info(f"  S = min(S2E, E2S) = {cca_retrieval['S']:.4f}")
        logger.info(f"  Random baseline: {cca_retrieval['random_baseline']:.4f}")

        cca_ci = grouped_bootstrap_ci(cca_retrieval['per_query'])
        logger.info(f"  CI (grouped by speaker): [{cca_ci['S_ci_lo']:.4f}, "
                    f"{cca_ci['S_ci_hi']:.4f}]")
    else:
        logger.warning("  CCA skipped (insufficient train pairs)")
        cca_ci = None

    # ── 4. Summary ──────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("SUMMARY — S2-P1 Baseline Linear")
    logger.info("=" * 60)
    logger.info(f"  Condition: {cond}")
    logger.info(f"  Pool size: {args.pool_size}, k: 10")
    logger.info(f"  R@10 random: {10/args.pool_size:.4f}")
    logger.info(f"  Ridge R² S→E: {ridge_results['r2_speech2egg']:.4f}, "
                f"E→S: {ridge_results['r2_egg2speech']:.4f}")
    logger.info(f"  Raw retrieval S: {raw_retrieval['S']:.4f} "
                f"CI [{raw_ci['S_ci_lo']:.4f}, {raw_ci['S_ci_hi']:.4f}]")
    if cca_retrieval:
        logger.info(f"  CCA retrieval S: {cca_retrieval['S']:.4f} "
                    f"CI [{cca_ci['S_ci_lo']:.4f}, {cca_ci['S_ci_hi']:.4f}]")

    # ── Save results ────────────────────────────────────────────────────────
    results = {
        'protocol': {
            'condition': cond,
            'pool_size': args.pool_size,
            'k': 10,
            'random_baseline': 10 / args.pool_size,
            'n_queries': raw_retrieval['n_queries'],
            'seed': args.seed,
        },
        'ridge': ridge_results,
        'raw_retrieval': {
            'S2E_at_k': raw_retrieval['S2E_at_k'],
            'E2S_at_k': raw_retrieval['E2S_at_k'],
            'S': raw_retrieval['S'],
            'strata_counts': raw_retrieval['strata_counts'],
            'ci': raw_ci,
            'rank_analysis': rank_analysis,
        },
    }
    if cca_retrieval:
        results['cca_retrieval'] = {
            'S2E_at_k': cca_retrieval['S2E_at_k'],
            'E2S_at_k': cca_retrieval['E2S_at_k'],
            'S': cca_retrieval['S'],
            'cca_correlations': cca_retrieval['cca_correlations'],
            'n_components': cca_retrieval['n_components'],
            'ci': cca_ci,
        }

    output_path = os.path.join(args.output_dir, f'p1_results_{cond}.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {output_path}")


if __name__ == '__main__':
    main()
