"""Conversión parciales → tokens + pair features anti-leakage + targets — Fase 0 Atención Armónica.

A partir de una mezcla (lista de picos con freq, amp, source_id) produce:
    tokens         [N, 2]      = [log-f centrado (invariante a transposición), log-amp]
    pair_feats     [N, N, F]   pair features ANTI-LEAKAGE (solo de freqs/amps observadas)
    ratio_class_id [N, N]       id de clase de ratio (para nn.Embedding; NO one-hot)
    target         [N, N]       binario mismo-fuente (de source_id) — solo para supervisión
    pair_valid     [N, N]       True donde el par entra al loss (excluye diagonal y near-collisions)

CRÍTICO (anti-leakage, Codex #3): `compute_pair_features` recibe SOLO (freqs, amps).
NUNCA source_id ni los f0 verdaderos. Test de anti-leakage: la firma no los acepta.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import List, Tuple

import numpy as np

Q_RATIO = 8           # grilla racional p,q ∈ {1..Q}
K_HARMONICS = 8       # para common_f0_residual (m,n ∈ {1..K})

# Número de features continuas en pair_feats (dlogf, ratio_residual, common_f0_residual, log_amp_diff)
N_PAIR_CONT_FEATS = 4


@lru_cache(maxsize=1)
def _simple_ratio_grid() -> Tuple[np.ndarray, int]:
    """Grilla de ratios simples p/q ≥ 1 con gcd(p,q)=1, p,q ≤ Q. Devuelve (log_ratios ordenados, n_clases)."""
    ratios = set()
    for p in range(1, Q_RATIO + 1):
        for q in range(1, Q_RATIO + 1):
            if math.gcd(p, q) == 1 and p >= q:
                ratios.add((p, q))
    log_ratios = sorted(math.log(p / q) for (p, q) in ratios)
    return np.array(log_ratios, dtype=np.float64), len(log_ratios)


def n_ratio_classes() -> int:
    _, n = _simple_ratio_grid()
    return n


@lru_cache(maxsize=1)
def _harmonic_log_grid() -> np.ndarray:
    """log(m) para m ∈ {1..K} — para common_f0_residual."""
    return np.log(np.arange(1, K_HARMONICS + 1, dtype=np.float64))


def compute_tokens(freqs: np.ndarray, amps: np.ndarray) -> np.ndarray:
    """tokens [N,2] = [log-f centrado por la mezcla, log-amp]. Anti-leakage (solo freqs/amps)."""
    logf = np.log(freqs)
    logf_centered = logf - logf.mean()
    logamp = np.log(np.maximum(amps, 1e-12))
    return np.stack([logf_centered, logamp], axis=1).astype(np.float32)


def compute_pair_features(
    freqs: np.ndarray, amps: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Pair features ANTI-LEAKAGE. Recibe SOLO freqs/amps observadas.

    Returns:
        pair_cont [N, N, 4] = [dlogf, ratio_residual, common_f0_residual, log_amp_diff] (simétricas)
        ratio_class_id [N, N] int = índice (p,q) más cercano al ratio observado
    """
    N = len(freqs)
    logf = np.log(freqs)
    logamp = np.log(np.maximum(amps, 1e-12))

    # dlogf simétrico
    dlogf = np.abs(logf[:, None] - logf[None, :])                       # [N,N]
    log_amp_diff = np.abs(logamp[:, None] - logamp[None, :])            # [N,N]

    # ratio observado r = max/min ≥ 1 → log r = |dlogf|
    log_r = dlogf                                                       # ya es |Δlog f| ≥ 0

    # ratio_residual: distancia al racional simple más cercano + clase
    grid_log, _n = _simple_ratio_grid()                                # [G]
    # |log_r - grid|  → [N,N,G]
    diff = np.abs(log_r[:, :, None] - grid_log[None, None, :])
    ratio_class_id = diff.argmin(axis=2).astype(np.int64)              # [N,N]
    ratio_residual = diff.min(axis=2)                                  # [N,N]

    # common_f0_residual: min_{m,n} |log(f_i/m) - log(f_j/n)|
    hlog = _harmonic_log_grid()                                        # [K]
    # log(f_i/m) = logf_i - hlog_m  → [N,K]
    a = logf[:, None] - hlog[None, :]                                  # [N,K]
    # para cada par (i,j): min_{m,n} |a[i,m] - a[j,n]|
    # broadcasting [N,1,K,1] - [1,N,1,K] = [N,N,K,K]
    common = np.abs(a[:, None, :, None] - a[None, :, None, :])         # [N,N,K,K]
    common_f0_residual = common.reshape(N, N, -1).min(axis=2)         # [N,N]

    pair_cont = np.stack(
        [dlogf, ratio_residual, common_f0_residual, log_amp_diff], axis=2,
    ).astype(np.float32)                                               # [N,N,4]
    return pair_cont, ratio_class_id


def compute_target_and_mask(
    source_ids: np.ndarray, masked_pairs: List[Tuple[int, int]], N: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """target [N,N] binario mismo-fuente (usa source_id — esto es SUPERVISIÓN, no feature);
    pair_valid [N,N] bool: True fuera de diagonal y fuera de near-collisions."""
    target = (source_ids[:, None] == source_ids[None, :]).astype(np.float32)  # [N,N]
    valid = ~np.eye(N, dtype=bool)
    for (i, j) in masked_pairs:
        valid[i, j] = False
        valid[j, i] = False
    return target, valid


def mixture_to_arrays(mixture: dict) -> dict:
    """Convierte una mezcla (dict del JSONL) a arrays listos para el dataset."""
    peaks = mixture["peaks"]
    freqs = np.array([p["freq"] for p in peaks], dtype=np.float64)
    amps = np.array([p["amp"] for p in peaks], dtype=np.float64)
    source_ids = np.array([p["source_id"] for p in peaks], dtype=np.int64)
    masked_pairs = [tuple(pr) for pr in mixture["masked_pairs"]]
    N = len(peaks)

    tokens = compute_tokens(freqs, amps)
    pair_cont, ratio_class_id = compute_pair_features(freqs, amps)
    target, pair_valid = compute_target_and_mask(source_ids, masked_pairs, N)

    return {
        "tokens": tokens,                       # [N,2] float32
        "pair_cont": pair_cont,                 # [N,N,4] float32
        "ratio_class_id": ratio_class_id,       # [N,N] int64
        "target": target,                       # [N,N] float32
        "pair_valid": pair_valid,               # [N,N] bool
        "polyphony": mixture["polyphony"],
        "regime": mixture["regime"],
        "mixture_id": mixture["mixture_id"],
        "n_peaks": N,
    }
