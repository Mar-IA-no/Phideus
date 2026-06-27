"""Generador de mezclas armónicas polifónicas con ground truth exacto — Fase 0 Atención Armónica.

Produce un POOL de mezclas. Cada mezcla es un conjunto de fuentes (cada una con un f0 y K
parciales) sumadas. Ground truth exacto: qué parcial pertenece a qué fuente. NO se renderiza
audio en Fase 0 — los parciales exactos SON los tokens.

La relación "mismo-fuente" entre picos es el target del experimento (matriz de equivalencia).

Grilla v2.1 (β/dropout/amplitud calibrados por sweep; ver pool_meta.json frozen_params):
    K = 8 parciales por fuente (1..8)
    f0 base ∈ [100, 500] Hz log-uniforme
    polifonía ∈ {1, 2, 3}
    régimen fácil:   ratios entre f0s irracionales/disonantes
    régimen difícil: ratios entre f0s casi-consonantes (3:2,4:3,5:4,2:1) + jitter ±10..30 cents
    inarmonicidad: f_n = n·f0·√(1+β·n²), β per-source >0 (v2.1; β=0 solo legacy/regresión)
    amplitud: amp_n = (1/n^α)·exp(ε_n), α per-source, ε_n~N(0,σ_amp²) (v2.1; rompe leak amp=1/n)
    dropout: p_drop por parcial, piso min_partials=4 (restauración por amplitud)
    ε loss-mask = 10 cents (pares de parciales más cercanos que esto se excluyen del loss)

Sidecar JSONL — una mezcla por línea — auditable:
    {mixture_id, polyphony, regime,
     sources:[{f0, beta, alpha, n_partials_surviving, was_restored, partials:[{harmonic,freq,amp}]}],
     peaks:[{freq,amp,source_id,harmonic}], masked_pairs:[[i,j],...]}

Uso:
    python experiments/atencion_armonica/0_generate.py \\
        --output-dir data/atencion_armonica/pool --workers 14
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Grilla congelada (Fase 0)
# ---------------------------------------------------------------------------

K_HARMONICS = 8                       # múltiplos 1..K por fuente
F0_MIN_HZ = 100.0
F0_MAX_HZ = 500.0
POLYPHONIES = (1, 2, 3)
REGIMES = ("easy", "hard")
EPS_CENTS = 10.0                      # pares de parciales más cercanos → excluidos del loss
JITTER_CENTS_MIN = 10.0
JITTER_CENTS_MAX = 30.0

# Inarmonicidad (v2): f_n = n·f0·√(1+β·n²), β per-source log-uniforme [center/width, center·width].
# El center y p_drop se CONGELAN tras el sweep de calibración; estos son placeholders.
INHARM_BETA_WIDTH = math.sqrt(10.0)   # factor multiplicativo del rango (fijo)
DEFAULT_BETA_CENTER = 1e-3            # placeholder; valor congelado sale del sweep
DEFAULT_P_DROP = 0.15                 # placeholder; valor congelado sale del sweep
MIN_PARTIALS = 4                      # piso de parciales por fuente tras dropout

# Amplitud randomizada (v2.1): amp_n = (1/n^α)·exp(ε_n), α per-source ~ U[α_lo,α_hi],
# ε_n ~ N(0, σ_amp²) iid. Rompe el leak log_amp_diff≈dlogf de la envolvente 1/n determinística.
# α-range y σ_amp se CONGELAN tras el sweep v2.1; estos son placeholders.
DEFAULT_ALPHA_RANGE = (0.5, 2.5)     # placeholder; rango congelado sale del sweep
DEFAULT_SIGMA_AMP = 1.0              # placeholder; valor congelado sale del sweep

# Régimen difícil: ratios casi-consonantes entre f0s
HARD_RATIOS = (
    (3, 2), (4, 3), (5, 4), (2, 1),
)
# Régimen fácil: multiplicadores irracionales/disonantes entre f0s
EASY_MULTIPLIERS = (
    math.sqrt(2.0),        # 1.41421
    (1.0 + math.sqrt(5.0)) / 2.0,   # phi ≈ 1.61803
    math.sqrt(3.0),        # 1.73205
    math.sqrt(2.0) * 1.18, # ~1.668, dissonant-ish
    2.0 ** (1.0 / 3.0) * 1.41,  # cube-root-2 scaled, irrational
)

# Cota superior de f0 derivado para que K·f0 quede holgadamente < Nyquist (22050 @ 44.1k)
F0_DERIVED_MAX_HZ = 1400.0


def cents_between(f1: float, f2: float) -> float:
    """Distancia absoluta en cents entre dos frecuencias."""
    return abs(1200.0 * math.log2(f1 / f2))


@dataclass
class Peak:
    freq: float
    amp: float
    source_id: int
    harmonic: int


@dataclass
class Mixture:
    mixture_id: int
    polyphony: int
    regime: str
    sources: List[Dict]          # [{f0, partials:[{harmonic,freq,amp}]}]
    peaks: List[Peak]
    masked_pairs: List[Tuple[int, int]]

    def to_json(self) -> Dict:
        return {
            "mixture_id": self.mixture_id,
            "polyphony": self.polyphony,
            "regime": self.regime,
            "sources": self.sources,
            "peaks": [
                {"freq": p.freq, "amp": p.amp, "source_id": p.source_id,
                 "harmonic": p.harmonic}
                for p in self.peaks
            ],
            "masked_pairs": [list(pr) for pr in self.masked_pairs],
        }


def _source_partials(f0: float, source_id: int, beta: float, alpha: float, sigma_amp: float,
                     p_drop: float, rng: np.random.RandomState,
                     min_partials: int = MIN_PARTIALS) -> Tuple[List[Dict], List[Peak], bool]:
    """Genera los parciales de una fuente con inarmonicidad β, amplitud randomizada y dropout.

    f_n = n·f0·√(1+β·n²). amp_raw_n = (1/n^α)·exp(ε_n), ε_n~N(0,σ_amp²).
    Orden EXACTO (Codex r6 #4): freqs → sample amps → dropout → restaurar por amp_raw → normalizar.
    """
    ns = np.arange(1, K_HARMONICS + 1)
    freqs = ns * f0 * np.sqrt(1.0 + beta * ns.astype(np.float64) ** 2)   # 1. inarmónico
    # 2. amplitudes randomizadas (rompe el leak amp=1/n)
    eps = rng.normal(0.0, sigma_amp, size=K_HARMONICS)
    amps_raw = (1.0 / ns.astype(np.float64) ** alpha) * np.exp(eps)

    keep = rng.rand(K_HARMONICS) >= p_drop                              # 3. dropout iid
    was_restored = bool(keep.sum() < min_partials)                     # ¿hubo que restaurar?
    if was_restored:                                                    # 4. restaurar por amp_raw
        for idx in np.argsort(-amps_raw):
            if keep.sum() >= min_partials:
                break
            keep[idx] = True

    kept = np.where(keep)[0]
    amps = amps_raw[kept] / np.sqrt((amps_raw[kept] ** 2).sum())        # 5. energía unidad (último)
    partials, peaks = [], []
    for k, idx in enumerate(kept):
        n = int(ns[idx]); fr = float(freqs[idx]); am = float(amps[k])
        partials.append({"harmonic": n, "freq": fr, "amp": am})
        peaks.append(Peak(freq=fr, amp=am, source_id=source_id, harmonic=n))
    return partials, peaks, was_restored


def _draw_f0s(polyphony: int, regime: str, rng: np.random.RandomState) -> List[float]:
    """Elige los f0 de las fuentes según el régimen.

    f0_base ∈ [100,500] log-uniforme. Las fuentes adicionales se obtienen multiplicando
    el f0_base por un factor de régimen (irracional para easy; casi-consonante con jitter
    en cents para hard). Se reintenta si algún f0 derivado se sale de la cota.
    """
    log_min, log_max = math.log(F0_MIN_HZ), math.log(F0_MAX_HZ)
    for _attempt in range(64):
        f0_base = math.exp(rng.uniform(log_min, log_max))
        f0s = [f0_base]
        if polyphony == 1:
            return f0s

        # multiplicadores para las (polyphony-1) fuentes adicionales
        if regime == "easy":
            mults = list(rng.choice(len(EASY_MULTIPLIERS), size=polyphony - 1, replace=True))
            factors = []
            for mi in mults:
                base = EASY_MULTIPLIERS[mi]
                # mitad de las veces invertir el factor (fuente más grave)
                if rng.rand() < 0.5:
                    base = 1.0 / base
                factors.append(base)
        else:  # hard: casi-consonante + jitter en cents
            factors = []
            for _ in range(polyphony - 1):
                p, q = HARD_RATIOS[rng.randint(len(HARD_RATIOS))]
                ratio = p / q
                if rng.rand() < 0.5:
                    ratio = 1.0 / ratio
                jitter_c = rng.uniform(JITTER_CENTS_MIN, JITTER_CENTS_MAX)
                if rng.rand() < 0.5:
                    jitter_c = -jitter_c
                ratio *= 2.0 ** (jitter_c / 1200.0)
                factors.append(ratio)

        f0s = [f0_base] + [f0_base * fac for fac in factors]
        if all(F0_MIN_HZ * 0.5 <= f <= F0_DERIVED_MAX_HZ for f in f0s):
            # evitar f0s demasiado cercanos entre sí (degenera la tarea)
            ok = True
            for i in range(len(f0s)):
                for j in range(i + 1, len(f0s)):
                    if cents_between(f0s[i], f0s[j]) < 50.0:
                        ok = False
            if ok:
                return f0s
    # fallback: devuelve lo último (raro)
    return f0s


def generate_mixture(mixture_id: int, polyphony: int, regime: str,
                     master_seed: int,
                     beta_center: float = DEFAULT_BETA_CENTER,
                     p_drop: float = DEFAULT_P_DROP,
                     alpha_range: Tuple[float, float] = DEFAULT_ALPHA_RANGE,
                     sigma_amp: float = DEFAULT_SIGMA_AMP,
                     beta_width: float = INHARM_BETA_WIDTH,
                     min_partials: int = MIN_PARTIALS) -> Mixture:
    """Genera una mezcla determinística desde (mixture_id, master_seed).

    β per-source log-uniforme [beta_center/beta_width, beta_center·beta_width]. α per-source
    ~ U[alpha_range]. amplitud (1/n^α)·exp(N(0,σ_amp²)). Dropout p_drop con piso min_partials.
    f0s se sortean ANTES de β/α/dropout → estables entre combos del sweep para un mismo mixture_id.
    Legacy (β=0, amp=1/n): beta_center→0, alpha_range=(1,1), sigma_amp=0 (solo por flag explícito).
    """
    rng = np.random.RandomState(master_seed + mixture_id)
    f0s = _draw_f0s(polyphony, regime, rng)

    legacy_beta0 = beta_center <= 0.0                                   # path legacy (β=0, sin log)
    if not legacy_beta0:
        log_lo, log_hi = math.log(beta_center / beta_width), math.log(beta_center * beta_width)
    a_lo, a_hi = alpha_range
    sources: List[Dict] = []
    peaks: List[Peak] = []
    for sid, f0 in enumerate(f0s):
        beta = 0.0 if legacy_beta0 else float(math.exp(rng.uniform(log_lo, log_hi)))  # β per-source
        alpha = float(rng.uniform(a_lo, a_hi))                          # α per-source
        partials, src_peaks, was_restored = _source_partials(
            f0, sid, beta, alpha, sigma_amp, p_drop, rng, min_partials)
        sources.append({"f0": f0, "beta": beta, "alpha": alpha,
                        "n_partials_surviving": len(partials),
                        "was_restored": was_restored, "partials": partials})
        peaks.extend(src_peaks)

    # pares enmascarados (near-collisions < EPS_CENTS) — se excluyen del loss
    masked_pairs: List[Tuple[int, int]] = []
    n = len(peaks)
    for i in range(n):
        for j in range(i + 1, n):
            if cents_between(peaks[i].freq, peaks[j].freq) < EPS_CENTS:
                masked_pairs.append((i, j))

    return Mixture(
        mixture_id=mixture_id, polyphony=polyphony, regime=regime,
        sources=sources, peaks=peaks, masked_pairs=masked_pairs,
    )
